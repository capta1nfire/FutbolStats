"""
Team Home City Cascade — fallback pipeline to populate team_home_city_profile.

Cascade order:
  0. National teams → source='excluded_national', home_city='N/A'
  1. Manual overrides (team_home_city_overrides WHERE active=true)
  2. venue_city mode (≥3 home FT matches with venue_city)
  3. venue_name geocoding (Open-Meteo, extracts city from admin fields)
  4. LLM candidate (Gemini, if enabled) — always needs_review=true

Delta mode: only teams without profile OR llm_candidates without active override.
Full mode: all active clubs + nationals bulk.

Reference: Plan "Fallback Cascade para team_home_city_profile"
ABE-approved with corrections #1-#8.
"""

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)

# Minimum home matches with venue_city to derive home_city (Step 1)
MIN_HOME_MATCHES = 3

# Geocoding API (same as Open-Meteo provider, but used directly for timezone)
GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"


@dataclass
class CascadeResult:
    """Result of cascade resolution for a single team."""

    team_id: int
    home_city: Optional[str]
    timezone: Optional[str]
    source: str
    confidence: float
    needs_review: bool
    resolved: bool


@dataclass
class CascadeMetrics:
    """Aggregate metrics for a cascade batch run."""

    scanned: int = 0
    resolved: int = 0
    unresolved: int = 0
    by_source: dict = field(default_factory=dict)
    nationals_excluded: int = 0
    errors: int = 0
    mode: str = "delta"


# =========================================================================
# Step 0: National teams — excluded from coverage
# =========================================================================
async def _bulk_exclude_nationals(session: AsyncSession) -> int:
    """
    Bulk-insert national teams with source='excluded_national'.

    ABE correction #1: home_city='N/A' (NOT NULL constraint), excluded from
    thermal_shock downstream + coverage denominator.
    """
    result = await session.execute(text("""
        INSERT INTO team_home_city_profile
          (team_id, home_city, timezone, source, confidence, needs_review, last_updated_at)
        SELECT t.id, 'N/A', 'UTC', 'excluded_national', 1.0, false, NOW()
        FROM teams t
        LEFT JOIN team_home_city_profile thcp ON t.id = thcp.team_id
        WHERE t.team_type = 'national'
          AND thcp.team_id IS NULL
        ON CONFLICT (team_id) DO UPDATE SET
          source = 'excluded_national',
          home_city = 'N/A',
          timezone = 'UTC',
          confidence = 1.0,
          needs_review = false,
          last_updated_at = NOW()
    """))
    count = result.rowcount or 0
    if count > 0:
        logger.info(f"[CASCADE] Excluded {count} national teams (source=excluded_national)")
    return count


# =========================================================================
# Step 1: Manual overrides
# =========================================================================
async def _check_manual_override(
    session: AsyncSession, team_id: int
) -> Optional[CascadeResult]:
    """Check if team has an active manual override."""
    result = await session.execute(text("""
        SELECT home_city, timezone
        FROM team_home_city_overrides
        WHERE team_id = :team_id AND active = true
    """), {"team_id": team_id})
    row = result.fetchone()
    if row:
        return CascadeResult(
            team_id=team_id,
            home_city=row[0],
            timezone=row[1],
            source="manual_override",
            confidence=1.0,
            needs_review=False,
            resolved=True,
        )
    return None


# =========================================================================
# Step 2: venue_city mode (historical home matches)
# =========================================================================
async def _derive_venue_city(
    session: AsyncSession, team_id: int
) -> Optional[CascadeResult]:
    """
    Derive home_city from most frequent venue_city in home FT matches.

    Requires ≥ MIN_HOME_MATCHES for confidence.
    """
    result = await session.execute(text("""
        SELECT m.venue_city, COUNT(*) AS cnt
        FROM matches m
        WHERE m.home_team_id = :team_id
          AND m.venue_city IS NOT NULL
          AND m.venue_city != ''
          AND m.status IN ('FT', 'AET', 'PEN')
        GROUP BY m.venue_city
        ORDER BY cnt DESC
        LIMIT 1
    """), {"team_id": team_id})
    row = result.fetchone()

    if row and row[1] >= MIN_HOME_MATCHES:
        city = row[0].strip()
        return CascadeResult(
            team_id=team_id,
            home_city=city,
            timezone=None,  # Will be resolved separately
            source="venue_city",
            confidence=0.9,
            needs_review=False,
            resolved=True,
        )
    return None


# =========================================================================
# Step 3: venue_name geocoding (Open-Meteo)
# ABE audit: venue_geo models cities, NOT stadium names — removed erroneous
# internal lookup. Step 3 is pure geocoding of venue_name (free text).
# =========================================================================
async def _derive_venue_name(
    session: AsyncSession,
    team_id: int,
    country: str,
    geocode_cache: dict,
    http_session=None,
) -> Optional[CascadeResult]:
    """
    Derive home_city from venue_name via external geocoding (Open-Meteo).

    Steps:
      1. Get most frequent venue_name from home matches
      2. Geocode venue_name via Open-Meteo API (extracts city from admin fields)
      3. Cache results per venue_name per run
    """
    # Get most frequent venue_name
    result = await session.execute(text("""
        SELECT m.venue_name, COUNT(*) AS cnt
        FROM matches m
        WHERE m.home_team_id = :team_id
          AND m.venue_name IS NOT NULL
          AND m.venue_name != ''
          AND m.status IN ('FT', 'AET', 'PEN')
        GROUP BY m.venue_name
        ORDER BY cnt DESC
        LIMIT 1
    """), {"team_id": team_id})
    row = result.fetchone()

    if not row or row[1] < 1:
        return None

    venue_name = row[0].strip()
    cache_key = f"{venue_name}|{country}"

    # Check in-memory cache first
    if cache_key in geocode_cache:
        cached = geocode_cache[cache_key]
        if cached is None:
            return None
        return CascadeResult(
            team_id=team_id,
            home_city=cached["city"],
            timezone=cached["timezone"],
            source="venue_name_geocoded",
            confidence=0.7,
            needs_review=False,
            resolved=True,
        )

    # Geocode venue_name via Open-Meteo
    try:
        geo_data = await _geocode_venue(venue_name, country, http_session)
        if geo_data and geo_data.get("city"):
            city = geo_data["city"]
            tz = geo_data.get("timezone")
            geocode_cache[cache_key] = {"city": city, "timezone": tz}
            return CascadeResult(
                team_id=team_id,
                home_city=city,
                timezone=tz,
                source="venue_name_geocoded",
                confidence=0.7,
                needs_review=False,
                resolved=True,
            )
    except Exception as e:
        logger.warning(f"[CASCADE] Geocoding failed for venue '{venue_name}': {e}")

    geocode_cache[cache_key] = None
    return None


async def _geocode_venue(
    venue_name: str, country: str, http_session=None
) -> Optional[dict]:
    """
    Geocode a venue name to find the city it's in.

    Uses Open-Meteo Geocoding API (free, no key).
    Extracts city from admin2/admin1/name fields of the result.

    Args:
        venue_name: Stadium / venue name (free text).
        country: Country for disambiguation.
        http_session: Optional reusable aiohttp.ClientSession.
    """
    import aiohttp

    owns_session = http_session is None
    session = http_session or aiohttp.ClientSession()

    try:
        params = {
            "name": venue_name,
            "count": 5,
            "language": "en",
            "format": "json",
        }
        async with session.get(GEOCODING_URL, params=params) as resp:
            if resp.status != 200:
                return None
            data = await resp.json()

        results = data.get("results", [])
        if not results:
            return None

        # Country alias mapping (API-Football uses constituent countries)
        country_aliases = {
            "england": ["united kingdom", "england"],
            "scotland": ["united kingdom", "scotland"],
            "wales": ["united kingdom", "wales"],
            "northern-ireland": ["united kingdom", "northern ireland"],
            "usa": ["united states", "usa"],
            "south-korea": ["south korea", "korea"],
            "czech-republic": ["czech republic", "czechia"],
            "north-macedonia": ["north macedonia", "macedonia"],
            "ivory-coast": ["ivory coast", "côte d'ivoire"],
        }
        country_lower = country.lower()
        match_terms = country_aliases.get(country_lower, [country_lower])

        # Find best match by country — NO fallback to avoid wrong-country results
        for r in results:
            r_country = (r.get("country") or "").lower()
            r_admin1 = (r.get("admin1") or "").lower()
            if any(
                term in r_country or r_country in term or term in r_admin1
                for term in match_terms
            ):
                city = r.get("admin2") or r.get("admin1") or r.get("name")
                if city:
                    return {
                        "city": city,
                        "timezone": r.get("timezone"),
                        "lat": r.get("latitude"),
                        "lon": r.get("longitude"),
                    }

        # No country match found — return None instead of wrong-country result
        logger.debug(
            f"[CASCADE] Geocode for '{venue_name}': no result matched country '{country}'"
        )

    except Exception as e:
        logger.warning(f"[CASCADE] Geocode error for '{venue_name}': {e}")
    finally:
        if owns_session:
            await session.close()

    return None


async def _geocode_timezone(
    city: str, country: str, cache: dict, http_session=None
) -> Optional[str]:
    """Resolve timezone for a city via Open-Meteo geocoding."""
    tz_key = f"tz|{city}|{country}"
    if tz_key in cache:
        return cache[tz_key]

    import aiohttp

    owns_session = http_session is None
    session = http_session or aiohttp.ClientSession()

    try:
        params = {
            "name": f"{city}, {country}",
            "count": 3,
            "language": "en",
            "format": "json",
        }
        async with session.get(GEOCODING_URL, params=params) as resp:
            if resp.status != 200:
                cache[tz_key] = None
                return None
            data = await resp.json()

        results = data.get("results", [])
        if results:
            tz = results[0].get("timezone")
            cache[tz_key] = tz
            return tz

    except Exception as e:
        logger.debug(f"[CASCADE] Timezone lookup failed for '{city}': {e}")
    finally:
        if owns_session:
            await session.close()

    cache[tz_key] = None
    return None


# =========================================================================
# Step 4: LLM candidate (Gemini)
# ABE correction #4: always needs_review=true, confidence=0.5
# =========================================================================
async def _derive_llm(
    team_id: int,
    team_name: str,
    country: str,
    geocode_cache: dict,
) -> Optional[CascadeResult]:
    """
    Use Gemini LLM to guess home city for a team.

    Only used as candidate generator — always needs_review=true.
    Validates by geocoding the suggested city.
    """
    try:
        from app.llm.gemini_client import GeminiClient

        client = GeminiClient()
        prompt = (
            f"What is the home city of the football/soccer club '{team_name}' "
            f"from {country}? Reply ONLY with a JSON object: "
            f'{{"city": "CityName", "timezone": "Continent/City", "reasoning": "brief explanation"}}. '
            f"No other text."
        )

        result = await client.generate(
            prompt=prompt,
            max_tokens=150,
            temperature=0.1,
        )
        await client.close()

        if result.status != "COMPLETED" or not result.text:
            return None

        # Parse JSON response
        text_clean = result.text.strip()
        if text_clean.startswith("```"):
            text_clean = text_clean.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

        parsed = json.loads(text_clean)
        city = parsed.get("city")
        timezone = parsed.get("timezone")

        if not city or not isinstance(city, str):
            return None

        # ABE correction #3: validate with geocoding
        geo_data = await _geocode_venue(city, country)
        if not geo_data:
            logger.info(
                f"[CASCADE] LLM suggested '{city}' for team {team_id} "
                f"but geocoding couldn't validate it"
            )
            # Still accept as candidate but with lower confidence signal
            pass

        return CascadeResult(
            team_id=team_id,
            home_city=city,
            timezone=timezone,
            source="llm_candidate",
            confidence=0.5,
            needs_review=True,
            resolved=True,
        )

    except json.JSONDecodeError:
        logger.warning(f"[CASCADE] LLM returned non-JSON for team {team_id}")
        return None
    except Exception as e:
        logger.warning(f"[CASCADE] LLM failed for team {team_id}: {e}")
        return None


# =========================================================================
# Timezone resolution for venue_city step (doesn't have tz yet)
# =========================================================================
async def _resolve_timezone_for_result(
    result: CascadeResult, country: str, geocode_cache: dict,
    http_session=None,
) -> CascadeResult:
    """Resolve timezone if missing from cascade result."""
    if result.timezone:
        return result

    if result.home_city and result.home_city != "N/A":
        tz = await _geocode_timezone(result.home_city, country, geocode_cache, http_session)
        if tz:
            result.timezone = tz
        else:
            # Fallback: use country-based default
            result.timezone = _country_fallback_timezone(country)
    else:
        result.timezone = "UTC"

    return result


def _country_fallback_timezone(country: str) -> str:
    """Fallback timezone based on country name."""
    mapping = {
        "england": "Europe/London", "scotland": "Europe/London",
        "wales": "Europe/London", "northern-ireland": "Europe/London",
        "spain": "Europe/Madrid", "germany": "Europe/Berlin",
        "france": "Europe/Paris", "italy": "Europe/Rome",
        "portugal": "Europe/Lisbon", "netherlands": "Europe/Amsterdam",
        "belgium": "Europe/Brussels", "turkey": "Europe/Istanbul",
        "argentina": "America/Buenos_Aires", "brazil": "America/Sao_Paulo",
        "mexico": "America/Mexico_City", "usa": "America/New_York",
        "japan": "Asia/Tokyo", "australia": "Australia/Sydney",
        "colombia": "America/Bogota", "chile": "America/Santiago",
        "peru": "America/Lima", "ecuador": "America/Guayaquil",
        "uruguay": "America/Montevideo", "paraguay": "America/Asuncion",
        "bolivia": "America/La_Paz", "venezuela": "America/Caracas",
        "greece": "Europe/Athens", "austria": "Europe/Vienna",
        "switzerland": "Europe/Zurich", "denmark": "Europe/Copenhagen",
        "norway": "Europe/Oslo", "sweden": "Europe/Stockholm",
        "finland": "Europe/Helsinki", "poland": "Europe/Warsaw",
        "czech-republic": "Europe/Prague", "croatia": "Europe/Zagreb",
        "serbia": "Europe/Belgrade", "romania": "Europe/Bucharest",
        "ukraine": "Europe/Kiev", "russia": "Europe/Moscow",
        "china": "Asia/Shanghai", "south-korea": "Asia/Seoul",
        "saudi-arabia": "Asia/Riyadh", "egypt": "Africa/Cairo",
        "south-africa": "Africa/Johannesburg", "morocco": "Africa/Casablanca",
        "nigeria": "Africa/Lagos", "algeria": "Africa/Algiers",
        "tunisia": "Africa/Tunis",
    }
    return mapping.get(country.lower(), "UTC")


# =========================================================================
# Single-team resolution
# =========================================================================
async def _resolve_single_team(
    session: AsyncSession,
    team_id: int,
    team_name: str,
    country: str,
    team_type: str,
    llm_enabled: bool,
    geocode_cache: dict,
    http_session=None,
) -> CascadeResult:
    """
    Run the full cascade for a single team.

    Returns CascadeResult (resolved=True if a source produced a result).
    """
    # National teams: excluded
    if team_type == "national":
        return CascadeResult(
            team_id=team_id,
            home_city="N/A",
            timezone="UTC",
            source="excluded_national",
            confidence=1.0,
            needs_review=False,
            resolved=True,
        )

    # Step 1: Manual override
    override = await _check_manual_override(session, team_id)
    if override:
        return await _resolve_timezone_for_result(override, country, geocode_cache, http_session)

    # Step 2: venue_city
    vc = await _derive_venue_city(session, team_id)
    if vc:
        return await _resolve_timezone_for_result(vc, country, geocode_cache, http_session)

    # Step 3: venue_name (geocoding)
    vn = await _derive_venue_name(session, team_id, country, geocode_cache, http_session)
    if vn:
        return vn  # Already has timezone from geocoding

    # Step 4: LLM (if enabled)
    if llm_enabled:
        llm = await _derive_llm(team_id, team_name, country, geocode_cache)
        if llm:
            return llm

    # Unresolved
    return CascadeResult(
        team_id=team_id,
        home_city=None,
        timezone=None,
        source="unresolved",
        confidence=0.0,
        needs_review=False,
        resolved=False,
    )


# =========================================================================
# Upsert profile
# =========================================================================
async def _upsert_profile(session: AsyncSession, result: CascadeResult) -> None:
    """Upsert a single team profile from cascade result."""
    if not result.resolved or not result.home_city:
        return

    # Note: climate_normals_by_month is NOT touched by cascade —
    # it's populated by a separate climate normals job.
    await session.execute(text("""
        INSERT INTO team_home_city_profile
          (team_id, home_city, timezone, source, confidence, needs_review, last_updated_at)
        VALUES
          (:team_id, :home_city, :timezone, :source, :confidence, :needs_review, NOW())
        ON CONFLICT (team_id) DO UPDATE SET
          home_city = EXCLUDED.home_city,
          timezone = EXCLUDED.timezone,
          source = EXCLUDED.source,
          confidence = EXCLUDED.confidence,
          needs_review = EXCLUDED.needs_review,
          last_updated_at = NOW()
    """), {
        "team_id": result.team_id,
        "home_city": result.home_city,
        "timezone": result.timezone or "UTC",
        "source": result.source,
        "confidence": result.confidence,
        "needs_review": result.needs_review,
    })


# =========================================================================
# Batch runner
# =========================================================================
async def run_cascade_batch(
    session: AsyncSession,
    mode: str = "delta",
    limit: int = 200,
    llm_enabled: bool = False,
    dry_run: bool = False,
) -> dict:
    """
    Run cascade for a batch of teams.

    Args:
        session: DB session.
        mode: 'delta' (only missing/llm_candidates) or 'full' (all active + nationals).
        limit: Max teams to process per run.
        llm_enabled: Whether to use LLM as fallback (Step 4).
        dry_run: If True, don't persist changes.

    Returns:
        Metrics dict for job_runs recording.
    """
    import aiohttp

    metrics = CascadeMetrics(mode=mode)
    geocode_cache: dict = {}
    COMMIT_BATCH_SIZE = 25

    logger.info(f"[CASCADE] Starting batch: mode={mode}, limit={limit}, llm={llm_enabled}, dry_run={dry_run}")

    # Step 0: Bulk-exclude nationals (always, even in delta)
    if not dry_run:
        metrics.nationals_excluded = await _bulk_exclude_nationals(session)
        await session.commit()

    # Get teams to process
    if mode == "full":
        # All active clubs without profile
        teams_result = await session.execute(text("""
            SELECT DISTINCT t.id, t.name, t.country, t.team_type
            FROM teams t
            JOIN matches m ON (m.home_team_id = t.id OR m.away_team_id = t.id)
            LEFT JOIN team_home_city_profile thcp ON t.id = thcp.team_id
            WHERE t.team_type = 'club'
              AND t.country IS NOT NULL
              AND t.country != ''
              AND m.date >= NOW() - INTERVAL '30 days'
              AND thcp.team_id IS NULL
            LIMIT :limit
        """), {"limit": limit})
    else:
        # Delta: clubs without profile + llm_candidates without active override
        teams_result = await session.execute(text("""
            SELECT DISTINCT t.id, t.name, t.country, t.team_type
            FROM teams t
            JOIN matches m ON (m.home_team_id = t.id OR m.away_team_id = t.id)
            LEFT JOIN team_home_city_profile thcp ON t.id = thcp.team_id
            WHERE t.team_type = 'club'
              AND t.country IS NOT NULL
              AND t.country != ''
              AND m.date >= NOW() - INTERVAL '30 days'
              AND (
                thcp.team_id IS NULL
                OR (
                  thcp.source = 'llm_candidate'
                  AND NOT EXISTS (
                    SELECT 1 FROM team_home_city_overrides o
                    WHERE o.team_id = t.id AND o.active = true
                  )
                )
              )
            LIMIT :limit
        """), {"limit": limit})

    teams = teams_result.fetchall()
    metrics.scanned = len(teams)
    logger.info(f"[CASCADE] Found {len(teams)} teams to process ({mode} mode)")

    # Reuse a single aiohttp session for all geocoding calls in this batch
    async with aiohttp.ClientSession() as http_session:
        pending_upserts = 0

        for team_row in teams:
            team_id, team_name, country, team_type = team_row[0], team_row[1], team_row[2], team_row[3]

            try:
                result = await _resolve_single_team(
                    session, team_id, team_name, country, team_type,
                    llm_enabled, geocode_cache, http_session,
                )

                if result.resolved:
                    metrics.resolved += 1
                    metrics.by_source[result.source] = metrics.by_source.get(result.source, 0) + 1

                    if not dry_run:
                        await _upsert_profile(session, result)
                        pending_upserts += 1

                        # Commit every COMMIT_BATCH_SIZE teams
                        if pending_upserts >= COMMIT_BATCH_SIZE:
                            await session.commit()
                            pending_upserts = 0

                    logger.debug(
                        f"[CASCADE] {team_name} ({country}): "
                        f"{result.source} -> {result.home_city} (tz={result.timezone})"
                    )
                else:
                    metrics.unresolved += 1
                    logger.debug(f"[CASCADE] {team_name} ({country}): unresolved")

                # Rate limit geocoding calls
                await asyncio.sleep(0.15)

            except Exception as e:
                metrics.errors += 1
                logger.error(f"[CASCADE] Error for team {team_id} ({team_name}): {e}")
                try:
                    await session.rollback()
                    pending_upserts = 0
                except Exception:
                    pass

        # Flush remaining upserts
        if pending_upserts > 0 and not dry_run:
            await session.commit()

    logger.info(
        f"[CASCADE] Batch complete: "
        f"scanned={metrics.scanned}, resolved={metrics.resolved}, "
        f"unresolved={metrics.unresolved}, errors={metrics.errors}, "
        f"nationals={metrics.nationals_excluded}, "
        f"by_source={metrics.by_source}"
    )

    return {
        "mode": metrics.mode,
        "scanned": metrics.scanned,
        "resolved": metrics.resolved,
        "unresolved": metrics.unresolved,
        "nationals_excluded": metrics.nationals_excluded,
        "errors": metrics.errors,
        "by_source": metrics.by_source,
    }
