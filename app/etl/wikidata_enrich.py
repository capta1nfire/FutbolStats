"""
Wikidata Team Enrichment ETL.

Uses SPARQL to extract structured team data from Wikidata.
Fallback: Wikipedia REST API for missing fields (per Kimi recommendation).
Guardrails: offline job, rate limiting, provenance, fail-open.

ABE Notes:
- Coords come from STADIUM (P115->P625), not the club
- admin_location_label from P131 is informational only, not normalized
- Timezone resolved downstream via _resolve_timezone_for_result()

Cascade (per Kimi):
- override (team_enrichment_overrides) -> wikidata -> wikipedia -> basic name
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Any, Optional

import httpx
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


SPARQL_QUERY_TEMPLATE = """
SELECT ?team ?teamLabel ?fullName ?shortName
       ?stadium ?stadiumLabel ?capacity ?altitude ?stadiumCoords
       ?adminLocation ?adminLocationLabel
       ?website ?twitter ?instagram
WHERE {{
  BIND(wd:{qid} AS ?team)

  # Official names
  OPTIONAL {{ ?team wdt:P1448 ?fullName . }}
  OPTIONAL {{ ?team wdt:P1813 ?shortName . }}

  # PRIORITY: Stadium (P115) and its properties - coords from STADIUM
  OPTIONAL {{
    ?team wdt:P115 ?stadium .
    OPTIONAL {{ ?stadium wdt:P1083 ?capacity . }}
    OPTIONAL {{ ?stadium wdt:P2044 ?altitude . }}
    OPTIONAL {{ ?stadium wdt:P625 ?stadiumCoords . }}
  }}

  # Admin location (P131) - ONLY as informational label, NOT as truth
  # ABE: P131 is noisy, can be district/neighborhood/region
  OPTIONAL {{ ?team wdt:P131 ?adminLocation . }}

  # Web/Social
  OPTIONAL {{ ?team wdt:P856 ?website . }}
  OPTIONAL {{ ?team wdt:P2002 ?twitter . }}
  OPTIONAL {{ ?team wdt:P2003 ?instagram . }}

  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en,es". }}
}}
LIMIT 1
"""


async def fetch_wikidata_for_team(
    qid: str,
    client: httpx.AsyncClient,
) -> Optional[dict[str, Any]]:
    """
    Fetch structured data from Wikidata for a single team.

    Returns raw response for provenance + parsed fields.
    Fail-open: returns None on error (no crash).
    """
    query = SPARQL_QUERY_TEMPLATE.format(qid=qid)

    try:
        response = await client.get(
            settings.WIKIDATA_SPARQL_ENDPOINT,
            params={"query": query, "format": "json"},
            headers={"User-Agent": "FutbolStats/1.0 (contact@futbolstats.app)"},
            timeout=30.0,
        )
        response.raise_for_status()
        data = response.json()

        bindings = data.get("results", {}).get("bindings", [])
        if not bindings:
            logger.debug(f"[WIKIDATA] No results for {qid}")
            return None

        return _parse_wikidata_response(bindings[0], raw=data)

    except httpx.TimeoutException:
        logger.warning(f"[WIKIDATA] Timeout fetching {qid}")
        return None
    except httpx.HTTPStatusError as e:
        logger.warning(f"[WIKIDATA] HTTP error for {qid}: {e.response.status_code}")
        return None
    except Exception as e:
        logger.warning(f"[WIKIDATA] Fetch failed for {qid}: {e}")
        return None  # Fail-open


def _parse_wikidata_response(binding: dict, raw: dict) -> dict[str, Any]:
    """Parse SPARQL binding to structured dict."""

    def get_value(key: str) -> Optional[str]:
        return binding.get(key, {}).get("value")

    def get_qid(key: str) -> Optional[str]:
        val = get_value(key)
        if val and "/entity/Q" in val:
            return val.split("/")[-1]
        return None

    # Parse stadium coordinates (format: "Point(lon lat)")
    # ABE: Coords come from STADIUM (P115->P625), not the club
    lat, lon = None, None
    coords = get_value("stadiumCoords")
    if coords and coords.startswith("Point("):
        try:
            parts = coords[6:-1].split()
            lon, lat = float(parts[0]), float(parts[1])
        except (ValueError, IndexError):
            pass

    # Social handles cleanup (remove URLs, keep only username)
    twitter = get_value("twitter")
    if twitter and "/" in twitter:
        twitter = twitter.rstrip("/").split("/")[-1]
    instagram = get_value("instagram")
    if instagram and "/" in instagram:
        instagram = instagram.rstrip("/").split("/")[-1]

    # Parse capacity and altitude as integers
    capacity = None
    capacity_str = get_value("capacity")
    if capacity_str:
        try:
            capacity = int(float(capacity_str))
        except (ValueError, TypeError):
            pass

    altitude = None
    altitude_str = get_value("altitude")
    if altitude_str:
        try:
            altitude = int(float(altitude_str))
        except (ValueError, TypeError):
            pass

    return {
        "raw_jsonb": raw,
        "full_name": get_value("fullName"),
        "short_name": get_value("shortName"),
        "stadium_name": get_value("stadiumLabel"),
        "stadium_wikidata_id": get_qid("stadium"),
        "stadium_capacity": capacity,
        "stadium_altitude_m": altitude,
        "lat": lat,  # From stadium
        "lon": lon,  # From stadium
        # ABE: admin_location_label is informational, not normalized
        "admin_location_label": get_value("adminLocationLabel"),
        "website": get_value("website"),
        "social_handles": {
            "twitter": twitter,
            "instagram": instagram,
        },
        "enrichment_source": "wikidata",
    }


# ============================================================================
# Wikipedia REST API Fallback (per Kimi recommendation)
# ============================================================================

WIKIPEDIA_API_BASE = "https://en.wikipedia.org/api/rest_v1/page/summary"


async def fetch_wikipedia_for_team(
    team_name: str,
    client: httpx.AsyncClient,
) -> Optional[dict[str, Any]]:
    """
    Fallback: Fetch team info from Wikipedia REST API.

    API: https://en.wikipedia.org/api/rest_v1/page/summary/{title}
    Rate limit: 200 req/min (we use 5 req/sec max)

    Returns partial enrichment data for fields missing in Wikidata.
    Fail-open: returns None on error.
    """
    # Build Wikipedia title (spaces -> underscores, handle special chars)
    # Try with " F.C." suffix first (common pattern), then plain name
    titles_to_try = [
        f"{team_name}_F.C.",
        f"{team_name}_FC",
        team_name.replace(" ", "_"),
    ]

    for title in titles_to_try:
        try:
            response = await client.get(
                f"{WIKIPEDIA_API_BASE}/{title}",
                headers={
                    "User-Agent": "FutbolStats/1.0 (contact@futbolstats.app)",
                    "Accept": "application/json",
                },
                timeout=15.0,
                follow_redirects=True,
            )

            if response.status_code == 404:
                continue  # Try next title variant

            response.raise_for_status()
            data = response.json()

            # Verify it's about a football club (not disambiguation)
            if data.get("type") == "disambiguation":
                continue

            return _parse_wikipedia_response(data, team_name)

        except httpx.TimeoutException:
            logger.warning(f"[WIKIPEDIA] Timeout fetching {title}")
            continue
        except httpx.HTTPStatusError as e:
            if e.response.status_code != 404:
                logger.warning(f"[WIKIPEDIA] HTTP error for {title}: {e.response.status_code}")
            continue
        except Exception as e:
            logger.warning(f"[WIKIPEDIA] Fetch failed for {title}: {e}")
            continue

    logger.debug(f"[WIKIPEDIA] No results for {team_name}")
    return None


def _parse_wikipedia_response(data: dict, team_name: str) -> dict[str, Any]:
    """
    Parse Wikipedia REST API response.

    Fields available:
    - title: Page title (often includes "F.C." suffix)
    - description: Short description (e.g., "Colombian football club")
    - extract: First paragraph summary
    """
    title = data.get("title", team_name)
    description = data.get("description", "")
    extract = data.get("extract", "")

    # Try to extract full name from extract (first sentence often has it)
    # Pattern: "Club Deportivo Cali, commonly known as Deportivo Cali, is..."
    full_name = None
    if extract:
        # Look for pattern: "X, commonly known as Y, is..."
        # Or: "X (Spanish: Y) is..."
        first_sentence = extract.split(".")[0] if extract else ""
        if "commonly known as" in first_sentence.lower():
            # Extract the official name before "commonly known as"
            parts = first_sentence.split(",")
            if parts:
                full_name = parts[0].strip()
        elif first_sentence and len(first_sentence) < 200:
            # Use first part before comma/parenthesis as potential full name
            for sep in [",", " (", " is "]:
                if sep in first_sentence:
                    candidate = first_sentence.split(sep)[0].strip()
                    if len(candidate) > len(team_name) and len(candidate) < 150:
                        full_name = candidate
                    break

    return {
        "raw_jsonb": {"wikipedia_response": data, "source": "wikipedia_rest_api"},
        "full_name": full_name,
        "short_name": None,  # Wikipedia doesn't provide this reliably
        "stadium_name": None,  # Would need separate stadium page query
        "stadium_wikidata_id": None,
        "stadium_capacity": None,
        "stadium_altitude_m": None,
        "lat": None,
        "lon": None,
        "admin_location_label": None,
        "website": None,
        "social_handles": {"twitter": None, "instagram": None},
        "enrichment_source": "wikipedia",
        "wikipedia_title": title,
        "wikipedia_description": description,
    }


async def get_override_for_team(
    session: AsyncSession,
    team_id: int,
) -> Optional[dict[str, Any]]:
    """
    Check if team has manual override in team_enrichment_overrides.

    Overrides take priority over Wikidata and Wikipedia.
    """
    try:
        result = await session.execute(
            text("""
                SELECT
                    full_name, short_name, stadium_name, admin_location_label,
                    lat, lon, website, twitter_handle, instagram_handle,
                    source, notes
                FROM team_enrichment_overrides
                WHERE team_id = :team_id
            """),
            {"team_id": team_id},
        )
        row = result.fetchone()

        if not row:
            return None

        return {
            "full_name": row.full_name,
            "short_name": row.short_name,
            "stadium_name": row.stadium_name,
            "admin_location_label": row.admin_location_label,
            "lat": row.lat,
            "lon": row.lon,
            "website": row.website,
            "social_handles": {
                "twitter": row.twitter_handle,
                "instagram": row.instagram_handle,
            },
            "enrichment_source": f"override:{row.source}",
            "override_notes": row.notes,
        }
    except Exception as e:
        # Table might not exist yet
        logger.debug(f"[OVERRIDE] Could not check overrides: {e}")
        return None


def merge_enrichment_data(
    wikidata: Optional[dict],
    wikipedia: Optional[dict],
    override: Optional[dict],
) -> dict[str, Any]:
    """
    Merge enrichment data with cascade priority: override > wikidata > wikipedia.

    Only fills missing fields from lower priority sources.
    """
    # Start with empty base
    merged = {
        "raw_jsonb": None,
        "full_name": None,
        "short_name": None,
        "stadium_name": None,
        "stadium_wikidata_id": None,
        "stadium_capacity": None,
        "stadium_altitude_m": None,
        "lat": None,
        "lon": None,
        "admin_location_label": None,
        "website": None,
        "social_handles": {"twitter": None, "instagram": None},
        "enrichment_source": "none",
    }

    # Track which source contributed
    sources_used = []

    # Apply in reverse priority order (lowest first, highest last wins)
    for source_name, source_data in [
        ("wikipedia", wikipedia),
        ("wikidata", wikidata),
        ("override", override),
    ]:
        if not source_data:
            continue

        sources_used.append(source_name)

        for key, value in source_data.items():
            if key == "social_handles" and value:
                # Merge social handles specially
                for handle_key, handle_val in value.items():
                    if handle_val and not merged["social_handles"].get(handle_key):
                        merged["social_handles"][handle_key] = handle_val
            elif key == "raw_jsonb":
                # Keep wikidata raw_jsonb as primary (for provenance)
                if source_name == "wikidata" and value:
                    merged["raw_jsonb"] = value
                elif source_name == "wikipedia" and value and not merged["raw_jsonb"]:
                    merged["raw_jsonb"] = value
            elif value is not None and merged.get(key) is None:
                merged[key] = value

    # Set enrichment_source based on primary contributor
    if override:
        merged["enrichment_source"] = override.get("enrichment_source", "override")
    elif wikidata:
        merged["enrichment_source"] = "wikidata"
        if wikipedia:
            merged["enrichment_source"] = "wikidata+wikipedia"
    elif wikipedia:
        merged["enrichment_source"] = "wikipedia"

    return merged


async def run_wikidata_enrichment_batch(
    session: AsyncSession,
    batch_size: int = 100,
    mode: str = "catch-up",
    use_wikipedia_fallback: bool = True,
) -> dict[str, Any]:
    """
    Run batch enrichment for teams with wikidata_id.

    Cascade (per Kimi recommendation):
    1. Check team_enrichment_overrides (manual corrections)
    2. Fetch from Wikidata SPARQL
    3. If missing fields, fallback to Wikipedia REST API
    4. Merge and upsert

    Modes:
    - catch-up: Process teams without enrichment (priority)
    - refresh: Process teams with enrichment > 30 days old

    736 teams / 100 batch = ~8 days for complete catch-up.
    """
    if mode == "catch-up":
        # Only teams without enrichment
        result = await session.execute(
            text("""
            SELECT t.id, t.wikidata_id, t.name
            FROM teams t
            LEFT JOIN team_wikidata_enrichment twe ON t.id = twe.team_id
            WHERE t.wikidata_id IS NOT NULL
              AND twe.team_id IS NULL
            LIMIT :batch_size
        """),
            {"batch_size": batch_size},
        )
    else:
        # Refresh: teams with enrichment > 30 days old
        result = await session.execute(
            text("""
            SELECT t.id, t.wikidata_id, t.name
            FROM teams t
            JOIN team_wikidata_enrichment twe ON t.id = twe.team_id
            WHERE twe.fetched_at < (NOW() AT TIME ZONE 'UTC') - INTERVAL '30 days'
            ORDER BY twe.fetched_at ASC
            LIMIT :batch_size
        """),
            {"batch_size": batch_size},
        )

    teams = result.fetchall()

    if not teams:
        return {
            "status": "ok",
            "mode": mode,
            "processed": 0,
            "message": "No teams to enrich",
        }

    enriched = 0
    errors = 0
    skipped = 0
    wikipedia_fallbacks = 0
    override_applied = 0

    async with httpx.AsyncClient() as client:
        for team_id, wikidata_id, team_name in teams:
            # Rate limiting (conservative: 5 req/sec)
            await asyncio.sleep(settings.WIKIDATA_RATE_LIMIT_DELAY)

            # Step 1: Check for manual override
            override_data = await get_override_for_team(session, team_id)
            if override_data:
                override_applied += 1

            # Step 2: Fetch from Wikidata
            wikidata_data = await fetch_wikidata_for_team(wikidata_id, client)

            # Step 3: Wikipedia fallback if missing critical fields
            wikipedia_data = None
            if use_wikipedia_fallback:
                needs_fallback = (
                    not wikidata_data
                    or not wikidata_data.get("full_name")
                )
                if needs_fallback and team_name:
                    await asyncio.sleep(0.1)  # Extra rate limit for Wikipedia
                    wikipedia_data = await fetch_wikipedia_for_team(team_name, client)
                    if wikipedia_data:
                        wikipedia_fallbacks += 1

            # Step 4: Merge with cascade priority
            merged_data = merge_enrichment_data(
                wikidata=wikidata_data,
                wikipedia=wikipedia_data,
                override=override_data,
            )

            # Only upsert if we have any data
            if wikidata_data or wikipedia_data or override_data:
                await _upsert_enrichment(session, team_id, wikidata_id, merged_data)
                enriched += 1
            else:
                errors += 1

    await session.commit()

    logger.info(
        f"[WIKIDATA] Batch complete: mode={mode}, processed={len(teams)}, "
        f"enriched={enriched}, errors={errors}, "
        f"wikipedia_fallbacks={wikipedia_fallbacks}, overrides={override_applied}"
    )

    return {
        "status": "ok",
        "mode": mode,
        "processed": len(teams),
        "enriched": enriched,
        "errors": errors,
        "skipped": skipped,
        "wikipedia_fallbacks": wikipedia_fallbacks,
        "override_applied": override_applied,
    }


async def _upsert_enrichment(
    session: AsyncSession,
    team_id: int,
    wikidata_id: str,
    data: dict[str, Any],
) -> None:
    """
    UPSERT enrichment data.

    ABE: Explicit ::jsonb cast to avoid type ambiguity in SQLAlchemy text().
    Includes enrichment_source to track data origin (per Kimi).
    """
    await session.execute(
        text("""
        INSERT INTO team_wikidata_enrichment (
            team_id, wikidata_id, fetched_at, raw_jsonb,
            stadium_name, stadium_wikidata_id, stadium_capacity, stadium_altitude_m,
            lat, lon, admin_location_label,
            full_name, short_name, social_handles, website,
            enrichment_source
        ) VALUES (
            :team_id, :wikidata_id, (NOW() AT TIME ZONE 'UTC'), :raw_jsonb::jsonb,
            :stadium_name, :stadium_wikidata_id, :stadium_capacity, :stadium_altitude_m,
            :lat, :lon, :admin_location_label,
            :full_name, :short_name, :social_handles::jsonb, :website,
            :enrichment_source
        )
        ON CONFLICT (team_id) DO UPDATE SET
            wikidata_id = EXCLUDED.wikidata_id,
            fetched_at = (NOW() AT TIME ZONE 'UTC'),
            raw_jsonb = EXCLUDED.raw_jsonb,
            stadium_name = EXCLUDED.stadium_name,
            stadium_wikidata_id = EXCLUDED.stadium_wikidata_id,
            stadium_capacity = EXCLUDED.stadium_capacity,
            stadium_altitude_m = EXCLUDED.stadium_altitude_m,
            lat = EXCLUDED.lat,
            lon = EXCLUDED.lon,
            admin_location_label = EXCLUDED.admin_location_label,
            full_name = EXCLUDED.full_name,
            short_name = EXCLUDED.short_name,
            social_handles = EXCLUDED.social_handles,
            website = EXCLUDED.website,
            enrichment_source = EXCLUDED.enrichment_source
    """),
        {
            "team_id": team_id,
            "wikidata_id": wikidata_id,
            "raw_jsonb": json.dumps(data.get("raw_jsonb")),
            "stadium_name": data.get("stadium_name"),
            "stadium_wikidata_id": data.get("stadium_wikidata_id"),
            "stadium_capacity": data.get("stadium_capacity"),
            "stadium_altitude_m": data.get("stadium_altitude_m"),
            "lat": data.get("lat"),
            "lon": data.get("lon"),
            "admin_location_label": data.get("admin_location_label"),
            "full_name": data.get("full_name"),
            "short_name": data.get("short_name"),
            "social_handles": json.dumps(data.get("social_handles")),
            "website": data.get("website"),
            "enrichment_source": data.get("enrichment_source", "wikidata"),
        },
    )


async def get_enrichment_stats(session: AsyncSession) -> dict[str, Any]:
    """Get stats about wikidata enrichment coverage including source breakdown."""
    result = await session.execute(
        text("""
        SELECT
            (SELECT COUNT(*) FROM teams WHERE wikidata_id IS NOT NULL) as total_with_wikidata,
            (SELECT COUNT(*) FROM team_wikidata_enrichment) as enriched,
            (SELECT COUNT(*) FROM team_wikidata_enrichment WHERE lat IS NOT NULL) as with_coords,
            (SELECT COUNT(*) FROM team_wikidata_enrichment WHERE stadium_name IS NOT NULL) as with_stadium,
            (SELECT COUNT(*) FROM team_wikidata_enrichment WHERE admin_location_label IS NOT NULL) as with_admin_label,
            (SELECT COUNT(*) FROM team_wikidata_enrichment WHERE full_name IS NOT NULL) as with_full_name
    """)
    )
    row = result.fetchone()

    total = row.total_with_wikidata or 0
    enriched = row.enriched or 0

    # Get source breakdown (column might not exist yet)
    source_breakdown = {}
    try:
        source_result = await session.execute(
            text("""
            SELECT
                COALESCE(enrichment_source, 'wikidata') as source,
                COUNT(*) as count
            FROM team_wikidata_enrichment
            GROUP BY enrichment_source
            ORDER BY count DESC
        """)
        )
        for src_row in source_result.fetchall():
            source_breakdown[src_row.source] = src_row.count
    except Exception:
        # Column doesn't exist yet
        source_breakdown = {"wikidata": enriched}

    return {
        "total_with_wikidata": total,
        "enriched": enriched,
        "pct_complete": round(100.0 * enriched / total, 1) if total > 0 else 0,
        "with_coords": row.with_coords or 0,
        "with_stadium": row.with_stadium or 0,
        "with_admin_label": row.with_admin_label or 0,
        "with_full_name": row.with_full_name or 0,
        "by_source": source_breakdown,
    }
