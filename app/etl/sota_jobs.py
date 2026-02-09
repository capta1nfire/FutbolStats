"""
SOTA Enrichment Jobs - Reusable async functions for scheduler.

This module contains the core logic for SOTA enrichment jobs:
- sync_understat_refs: Link matches to Understat IDs
- backfill_understat_ft: Fetch xG data for finished matches
- capture_weather_prekickoff: Capture weather forecasts for upcoming matches
- expand_venue_geo: Geocode new venues (placeholder)

All functions are designed to be called from scheduler or CLI scripts.
They take a session and return a metrics dict (never raise, best-effort).

Reference: docs/ARCHITECTURE_SOTA.md
"""

import logging
from datetime import datetime, timedelta
from typing import Optional

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.etl.name_normalization import normalize_team_name
from app.etl.sofascore_aliases import build_alias_index, names_are_aliases

logger = logging.getLogger(__name__)

from app.etl.sota_constants import LEAGUE_PROXY_COUNTRY, SOFASCORE_SUPPORTED_LEAGUES, UNDERSTAT_SUPPORTED_LEAGUES  # noqa: E402

# Mapping from API-Football league IDs to Understat league names
LEAGUE_ID_TO_UNDERSTAT = {
    39: "EPL",
    140: "La_Liga",
    135: "Serie_A",
    78: "Bundesliga",
    61: "Ligue_1",
}


# =============================================================================
# UNDERSTAT REFS SYNC
# =============================================================================

async def sync_understat_refs(
    session: AsyncSession,
    days: int = 7,
    limit: int = 200,
) -> dict:
    """
    Sync Understat external refs for recent finished matches.

    Links internal matches to Understat match IDs using team name + kickoff matching.

    Args:
        session: Database session.
        days: Days back to scan for unlinked matches.
        limit: Max matches to process per run.

    Returns:
        Dict with metrics: scanned, linked_auto, linked_review, skipped_*, errors.
    """
    from app.etl.match_external_refs import (
        compute_match_score,
        get_match_decision,
        upsert_match_external_ref,
    )
    from app.etl.understat_provider import UnderstatProvider

    metrics = {
        "scanned": 0,
        "linked_auto": 0,
        "linked_review": 0,
        "skipped_no_candidates": 0,
        "skipped_low_score": 0,
        "errors": 0,
    }

    # Cache for Understat league matches
    understat_cache: dict[str, list[dict]] = {}
    provider = UnderstatProvider(use_mock=False)
    alias_index = build_alias_index()

    try:
        # Get candidate matches (FT without understat ref, in supported leagues)
        league_ids_str = ",".join(str(lid) for lid in UNDERSTAT_SUPPORTED_LEAGUES)
        cutoff = datetime.utcnow() - timedelta(days=days)
        limit_clause = f"LIMIT {limit}" if limit else ""

        result = await session.execute(text(f"""
            SELECT
                m.id AS match_id,
                m.external_id,
                m.date AS kickoff_utc,
                m.league_id,
                m.season,
                m.venue_city,
                t_home.name AS home_team,
                t_away.name AS away_team
            FROM matches m
            JOIN teams t_home ON m.home_team_id = t_home.id
            JOIN teams t_away ON m.away_team_id = t_away.id
            LEFT JOIN match_external_refs mer
                ON m.id = mer.match_id AND mer.source = 'understat'
            WHERE m.status IN ('FT', 'AET', 'PEN')
              AND m.date >= :cutoff
              AND m.home_goals IS NOT NULL
              AND m.away_goals IS NOT NULL
              AND m.league_id IN ({league_ids_str})
              AND mer.match_id IS NULL
            ORDER BY m.date DESC
            {limit_clause}
        """), {"cutoff": cutoff})

        matches = result.fetchall()
        metrics["scanned"] = len(matches)

        if not matches:
            logger.info(f"[SOTA_REFS] No unlinked matches found (last {days} days)")
            return metrics

        logger.info(f"[SOTA_REFS] Found {len(matches)} unlinked matches to process")

        for match in matches:
            match_id = match.match_id
            try:
                api_match = {
                    "match_id": match_id,
                    "kickoff_utc": match.kickoff_utc,
                    "home_team": match.home_team,
                    "away_team": match.away_team,
                    "league_id": match.league_id,
                    "season": match.season,
                    "venue_city": match.venue_city,
                }

                # Fetch Understat candidates
                candidates = await _fetch_understat_candidates(
                    api_match, provider, understat_cache, alias_index=alias_index
                )

                if not candidates:
                    metrics["skipped_no_candidates"] += 1
                    continue

                # Find best match
                best_candidate = None
                best_score = 0.0
                for candidate in candidates:
                    score = compute_match_score(api_match, candidate, alias_index=alias_index)
                    if score > best_score:
                        best_score = score
                        best_candidate = candidate

                # Decide whether to link
                should_link, needs_review = get_match_decision(best_score)

                if not should_link:
                    metrics["skipped_low_score"] += 1
                    continue

                # Build matched_by string
                matched_by = "kickoff+teams"
                if match.league_id:
                    matched_by += "+league"
                if match.venue_city:
                    matched_by += "+venue"
                if needs_review:
                    matched_by += ";needs_review"

                # Upsert to DB
                await upsert_match_external_ref(
                    session=session,
                    match_id=match_id,
                    source="understat",
                    source_match_id=best_candidate["source_match_id"],
                    confidence=best_score,
                    matched_by=matched_by,
                )

                if needs_review:
                    metrics["linked_review"] += 1
                else:
                    metrics["linked_auto"] += 1

            except Exception as e:
                metrics["errors"] += 1
                logger.error(f"[SOTA_REFS] Error processing match {match_id}: {e}")
                continue

        await session.commit()
        logger.info(
            f"[SOTA_REFS] Complete: linked_auto={metrics['linked_auto']}, "
            f"linked_review={metrics['linked_review']}, errors={metrics['errors']}"
        )

    except Exception as e:
        metrics["errors"] += 1
        logger.error(f"[SOTA_REFS] Job failed: {e}")

    finally:
        await provider.close()
        understat_cache.clear()

    return metrics


async def _fetch_understat_candidates(
    api_match: dict,
    provider,
    cache: dict,
    alias_index: Optional[dict[str, set[str]]] = None,
) -> list[dict]:
    """Fetch Understat match candidates using provider with caching."""
    league_id = api_match.get("league_id")
    if league_id not in LEAGUE_ID_TO_UNDERSTAT:
        return []

    understat_league = LEAGUE_ID_TO_UNDERSTAT[league_id]

    # Determine season
    season = api_match.get("season", "")
    if not season:
        kickoff = api_match.get("kickoff_utc")
        if kickoff:
            season = str(kickoff.year if kickoff.month >= 8 else kickoff.year - 1)
        else:
            return []

    cache_key = f"{understat_league}_{season}"

    # Check cache
    if cache_key not in cache:
        logger.debug(f"[SOTA_REFS] Fetching Understat matches for {cache_key}...")
        matches = await provider.get_league_matches(understat_league, season)
        cache[cache_key] = matches
        logger.debug(f"[SOTA_REFS] Cached {len(matches)} matches for {cache_key}")

    # Filter to finished matches that could be candidates
    all_matches = cache[cache_key]
    candidates = []

    for u_match in all_matches:
        if not u_match.get("is_result"):
            continue

        if _is_match_candidate(api_match, u_match, alias_index=alias_index):
            try:
                candidates.append({
                    "source_match_id": str(u_match.get("id")),
                    "kickoff_utc": datetime.strptime(
                        u_match.get("datetime", ""), "%Y-%m-%d %H:%M:%S"
                    ),
                    "home_team": u_match.get("home_team"),
                    "away_team": u_match.get("away_team"),
                    "league_id": league_id,
                    "season": season,
                })
            except (ValueError, TypeError):
                continue

    return candidates


def _is_match_candidate(
    api_match: dict,
    u_match: dict,
    tolerance_hours: int = 2,
    alias_index: Optional[dict[str, set[str]]] = None,
) -> bool:
    """Check if Understat match could match the API match."""
    try:
        u_kickoff_str = u_match.get("datetime", "")
        u_kickoff = datetime.strptime(u_kickoff_str, "%Y-%m-%d %H:%M:%S")
    except (ValueError, TypeError):
        return False

    api_kickoff = api_match.get("kickoff_utc")
    if not api_kickoff:
        return False

    time_diff = abs((u_kickoff - api_kickoff).total_seconds() / 3600)
    if time_diff > tolerance_hours:
        return False

    api_home = normalize_team_name(api_match.get("home_team", ""))
    api_away = normalize_team_name(api_match.get("away_team", ""))
    u_home = normalize_team_name(u_match.get("home_team", ""))
    u_away = normalize_team_name(u_match.get("away_team", ""))

    home_match = (
        api_home == u_home
        or api_home in u_home or u_home in api_home
        or (alias_index and names_are_aliases(api_match.get("home_team", ""), u_match.get("home_team", ""), alias_index))
    )
    away_match = (
        api_away == u_away
        or api_away in u_away or u_away in api_away
        or (alias_index and names_are_aliases(api_match.get("away_team", ""), u_match.get("away_team", ""), alias_index))
    )

    return home_match and away_match


# =============================================================================
# UNDERSTAT FT BACKFILL
# =============================================================================

async def backfill_understat_ft(
    session: AsyncSession,
    days: int = 14,
    limit: int = 100,
    with_ref_only: bool = True,
) -> dict:
    """
    Backfill Understat xG data for finished matches.

    Fetches xG/xPTS from Understat for matches that have refs but no xG data.

    Args:
        session: Database session.
        days: Days back to scan.
        limit: Max matches to process.
        with_ref_only: Only process matches that have understat refs (faster).

    Returns:
        Dict with metrics: scanned, inserted, updated, skipped_*, errors.
    """
    from app.etl.understat_provider import UnderstatProvider

    metrics = {
        "scanned": 0,
        "inserted": 0,
        "updated": 0,
        "skipped_no_ref": 0,
        "skipped_no_data": 0,
        "errors": 0,
    }

    cutoff = datetime.utcnow() - timedelta(days=days)
    join_type = "JOIN" if with_ref_only else "LEFT JOIN"
    limit_clause = f"LIMIT {limit}" if limit else ""

    provider = UnderstatProvider(use_mock=False)

    try:
        result = await session.execute(text(f"""
            SELECT
                m.id AS match_id,
                m.external_id,
                m.date,
                m.home_goals,
                m.away_goals,
                mer.source_match_id AS understat_id
            FROM matches m
            {join_type} match_external_refs mer
                ON m.id = mer.match_id AND mer.source = 'understat'
            LEFT JOIN match_understat_team mut ON m.id = mut.match_id
            WHERE m.status IN ('FT', 'AET', 'PEN')
              AND m.date >= :cutoff
              AND m.home_goals IS NOT NULL
              AND m.away_goals IS NOT NULL
              AND mut.match_id IS NULL
            ORDER BY m.date DESC
            {limit_clause}
        """), {"cutoff": cutoff})

        matches = result.fetchall()
        metrics["scanned"] = len(matches)

        if not matches:
            logger.info(f"[SOTA_XG] No matches need xG data (last {days} days)")
            return metrics

        logger.info(f"[SOTA_XG] Found {len(matches)} matches to backfill xG")

        for match in matches:
            match_id = match.match_id
            understat_id = match.understat_id

            try:
                if not understat_id:
                    metrics["skipped_no_ref"] += 1
                    continue

                # Fetch xG data from provider
                xg_data = await provider.get_match_team_xg(understat_id)

                if xg_data is None:
                    metrics["skipped_no_data"] += 1
                    continue

                # Upsert to match_understat_team
                result_action = await _upsert_understat_data(session, match_id, xg_data)
                metrics[result_action] += 1

            except Exception as e:
                metrics["errors"] += 1
                logger.error(
                    f"[SOTA_XG] Error processing match {match_id} ({type(e).__name__}): {e!r}",
                    exc_info=True
                )
                continue

        await session.commit()
        logger.info(
            f"[SOTA_XG] Complete: inserted={metrics['inserted']}, "
            f"updated={metrics['updated']}, errors={metrics['errors']}"
        )

    except Exception as e:
        metrics["errors"] += 1
        metrics["job_failed"] = True
        logger.error(
            f"[SOTA_XG] Job failed ({type(e).__name__}): {e!r}",
            exc_info=True
        )
        # Rollback defensivo para evitar dejar la conexión en estado "failed transaction"
        try:
            await session.rollback()
        except Exception:
            pass

    finally:
        await provider.close()

    return metrics


async def _upsert_understat_data(session: AsyncSession, match_id: int, data) -> str:
    """Upsert xG data to match_understat_team. Returns 'inserted' or 'updated'."""
    check = await session.execute(
        text("SELECT 1 FROM match_understat_team WHERE match_id = :match_id"),
        {"match_id": match_id}
    )
    exists = check.scalar() is not None

    params = {
        "match_id": match_id,
        "xg_home": data.xg_home,
        "xg_away": data.xg_away,
        "xpts_home": data.xpts_home,
        "xpts_away": data.xpts_away,
        "npxg_home": data.npxg_home,
        "npxg_away": data.npxg_away,
        "xga_home": data.xga_home,
        "xga_away": data.xga_away,
        "captured_at": data.captured_at or datetime.utcnow(),
        "source_version": data.source_version,
    }

    if exists:
        await session.execute(text("""
            UPDATE match_understat_team SET
                xg_home = :xg_home, xg_away = :xg_away,
                xpts_home = :xpts_home, xpts_away = :xpts_away,
                npxg_home = :npxg_home, npxg_away = :npxg_away,
                xga_home = :xga_home, xga_away = :xga_away,
                captured_at = :captured_at, source_version = :source_version
            WHERE match_id = :match_id
        """), params)
        return "updated"
    else:
        await session.execute(text("""
            INSERT INTO match_understat_team (
                match_id, xg_home, xg_away, xpts_home, xpts_away,
                npxg_home, npxg_away, xga_home, xga_away,
                captured_at, source_version
            ) VALUES (
                :match_id, :xg_home, :xg_away, :xpts_home, :xpts_away,
                :npxg_home, :npxg_away, :xga_home, :xga_away,
                :captured_at, :source_version
            )
        """), params)
        return "inserted"


# =============================================================================
# WEATHER CAPTURE (REAL)
# =============================================================================

async def capture_weather_prekickoff(
    session: AsyncSession,
    hours: int = 48,
    limit: int = 100,
    horizon: int = 24,
) -> dict:
    """
    Capture weather forecasts for upcoming matches.

    Uses Open-Meteo API to fetch weather data for venues with geo coordinates.

    Args:
        session: Database session.
        hours: Hours ahead to look for NS matches.
        limit: Max matches to process.
        horizon: Forecast horizon in hours (default 24).

    Returns:
        Dict with metrics: matches_checked, with_geo, inserted, updated, skipped_*, errors.
    """
    from app.etl.open_meteo_provider import OpenMeteoProvider

    metrics = {
        "matches_checked": 0,
        "with_geo": 0,
        "with_geo_venue": 0,  # From venue_geo table
        "with_geo_wikidata": 0,  # From team_wikidata_enrichment (fallback)
        "inserted": 0,
        "updated": 0,
        "skipped_no_geo": 0,
        "skipped_already_captured": 0,
        "errors": 0,
    }

    provider = OpenMeteoProvider(use_mock=False)

    try:
        # Find NS matches in next N hours with venue geo coordinates
        # Join via home team's country to resolve venue_geo
        # Fallback: COALESCE to team_wikidata_enrichment (stadium coords) if venue_geo missing
        # ABE: Using home_team stadium coords for neutral venues is acceptable for MVP
        result = await session.execute(text(f"""
            SELECT
                m.id AS match_id,
                m.date AS kickoff_utc,
                m.venue_city,
                t_home.country AS home_country,
                COALESCE(vg.lat, twe.lat) AS lat,
                COALESCE(vg.lon, twe.lon) AS lon,
                CASE WHEN vg.lat IS NOT NULL THEN 'venue_geo'
                     WHEN twe.lat IS NOT NULL THEN 'wikidata_stadium'
                     ELSE NULL END AS geo_source
            FROM matches m
            JOIN teams t_home ON m.home_team_id = t_home.id
            LEFT JOIN venue_geo vg
                ON m.venue_city = vg.venue_city
                AND t_home.country = vg.country
            LEFT JOIN team_wikidata_enrichment twe
                ON m.home_team_id = twe.team_id
            LEFT JOIN match_weather mw
                ON m.id = mw.match_id
                AND mw.forecast_horizon_hours = :horizon
            WHERE m.status = 'NS'
              AND m.date >= NOW()
              AND m.date < NOW() + INTERVAL '{hours} hours'
              AND mw.match_id IS NULL
            ORDER BY m.date ASC
            LIMIT :limit
        """), {"horizon": horizon, "limit": limit})

        matches = result.fetchall()
        metrics["matches_checked"] = len(matches)

        if not matches:
            logger.debug(f"[SOTA_WEATHER] No matches need weather (next {hours}h)")
            return metrics

        logger.info(f"[SOTA_WEATHER] Found {len(matches)} matches to capture weather")

        for match in matches:
            match_id = match.match_id
            kickoff_utc = match.kickoff_utc
            lat = match.lat
            lon = match.lon
            geo_source = match.geo_source

            try:
                if lat is None or lon is None:
                    metrics["skipped_no_geo"] += 1
                    continue

                metrics["with_geo"] += 1

                # Track geo source for observability
                if geo_source == "venue_geo":
                    metrics["with_geo_venue"] += 1
                elif geo_source == "wikidata_stadium":
                    metrics["with_geo_wikidata"] += 1
                    logger.debug(f"[SOTA_WEATHER] Using wikidata fallback for match {match_id}")

                # Fetch weather forecast
                forecast = await provider.get_forecast(
                    lat=lat,
                    lon=lon,
                    kickoff_utc=kickoff_utc,
                    horizon_hours=horizon,
                )

                if forecast is None:
                    metrics["errors"] += 1
                    logger.warning(f"[SOTA_WEATHER] No forecast for match {match_id}")
                    continue

                # Upsert to match_weather
                result_action = await _upsert_weather_data(
                    session, match_id, forecast, horizon
                )
                metrics[result_action] += 1

            except Exception as e:
                metrics["errors"] += 1
                logger.error(f"[SOTA_WEATHER] Error for match {match_id}: {e}")
                continue

        await session.commit()
        logger.info(
            f"[SOTA_WEATHER] Complete: inserted={metrics['inserted']}, "
            f"updated={metrics['updated']}, skipped_no_geo={metrics['skipped_no_geo']}"
        )

    except Exception as e:
        metrics["errors"] += 1
        logger.error(f"[SOTA_WEATHER] Job failed: {e}")

    finally:
        await provider.close()

    return metrics


async def _upsert_weather_data(
    session: AsyncSession,
    match_id: int,
    forecast,
    horizon: int,
) -> str:
    """Upsert weather data to match_weather. Returns 'inserted' or 'updated'."""
    check = await session.execute(
        text("""
            SELECT 1 FROM match_weather
            WHERE match_id = :match_id AND forecast_horizon_hours = :horizon
        """),
        {"match_id": match_id, "horizon": horizon}
    )
    exists = check.scalar() is not None

    params = {
        "match_id": match_id,
        "temp_c": forecast.temp_c,
        "humidity": forecast.humidity,
        "wind_ms": forecast.wind_ms,
        "precip_mm": forecast.precip_mm,
        "precip_prob": forecast.precip_prob,
        "pressure_hpa": forecast.pressure_hpa,
        "cloudcover": forecast.cloudcover,
        "is_daylight": forecast.is_daylight,
        "forecast_horizon_hours": horizon,
        "captured_at": forecast.captured_at or datetime.utcnow(),
    }

    if exists:
        await session.execute(text("""
            UPDATE match_weather SET
                temp_c = :temp_c, humidity = :humidity, wind_ms = :wind_ms,
                precip_mm = :precip_mm, precip_prob = :precip_prob,
                pressure_hpa = :pressure_hpa, cloudcover = :cloudcover,
                is_daylight = :is_daylight, captured_at = :captured_at
            WHERE match_id = :match_id AND forecast_horizon_hours = :forecast_horizon_hours
        """), params)
        return "updated"
    else:
        await session.execute(text("""
            INSERT INTO match_weather (
                match_id, temp_c, humidity, wind_ms, precip_mm, precip_prob,
                pressure_hpa, cloudcover, is_daylight,
                forecast_horizon_hours, captured_at
            ) VALUES (
                :match_id, :temp_c, :humidity, :wind_ms, :precip_mm, :precip_prob,
                :pressure_hpa, :cloudcover, :is_daylight,
                :forecast_horizon_hours, :captured_at
            )
        """), params)
        return "inserted"


# =============================================================================
# VENUE GEO EXPAND (REAL)
# =============================================================================

# Rate limiting for geocoding (be nice to Open-Meteo)
GEOCODING_DELAY_SECONDS = 0.5  # 2 requests/sec max


# =============================================================================
# SOFASCORE XI CAPTURE
# =============================================================================

# SOFASCORE_SUPPORTED_LEAGUES is imported from app.etl.sota_constants


async def sync_sofascore_refs(
    session: AsyncSession,
    hours: int = 72,
    days_back: int = 2,
    limit: int = 200,
    use_mock: bool = False,
) -> dict:
    """
    Sync Sofascore external refs for upcoming matches.

    Links internal matches to Sofascore event IDs using deterministic matching
    based on team names and kickoff time.

    Matching algorithm:
    - Score S per candidate based on:
      - Kickoff UTC (tolerance ±2h): 0.3 weight
      - Home team name (normalized): 0.35 weight
      - Away team name (normalized): 0.35 weight
    - Decision:
      - S >= 0.90: upsert with confidence=S
      - 0.75 <= S < 0.90: upsert with ";needs_review" flag
      - S < 0.75: skip (no link)

    Args:
        session: Database session.
        hours: Hours ahead to scan for NS matches.
        days_back: Days back to also scan (for recently scheduled matches).
        limit: Max matches to process per run.
        use_mock: Use mock data for testing.

    Returns:
        Dict with metrics: scanned, already_linked, linked_auto, linked_review,
        skipped_no_candidates, skipped_low_score, errors.
    """
    from app.etl.sofascore_provider import (
        SofascoreProvider,
        calculate_match_score,
        get_sofascore_threshold,
        normalize_team_name,
    )
    from app.etl.sofascore_aliases import build_alias_index

    # Build alias index once per job run
    alias_index = build_alias_index()

    metrics = {
        "scanned": 0,
        "already_linked": 0,
        "linked_auto": 0,
        "linked_review": 0,
        "skipped_no_candidates": 0,
        "skipped_low_score": 0,
        "near_misses": 0,
        "errors": 0,
    }

    provider = SofascoreProvider(use_mock=use_mock)

    try:
        # Find NS matches in supported leagues without sofascore ref
        league_ids_str = ",".join(str(lid) for lid in SOFASCORE_SUPPORTED_LEAGUES)
        limit_clause = f"LIMIT {limit}" if limit else ""

        result = await session.execute(text(f"""
            SELECT
                m.id AS match_id,
                m.external_id,
                m.date AS kickoff_utc,
                m.league_id,
                t_home.name AS home_team,
                t_away.name AS away_team,
                mer.source_match_id AS existing_sofascore_id
            FROM matches m
            JOIN teams t_home ON m.home_team_id = t_home.id
            JOIN teams t_away ON m.away_team_id = t_away.id
            LEFT JOIN match_external_refs mer
                ON m.id = mer.match_id AND mer.source = 'sofascore'
            WHERE m.status = 'NS'
              AND m.date >= NOW() - INTERVAL '{days_back} days'
              AND m.date < NOW() + INTERVAL '{hours} hours'
              AND m.league_id IN ({league_ids_str})
            ORDER BY m.date ASC
            {limit_clause}
        """))

        matches = result.fetchall()
        metrics["scanned"] = len(matches)

        if not matches:
            logger.debug("[SOFASCORE_REFS] No matches to process")
            return metrics

        # Group matches by date for efficient API calls
        matches_by_date: dict[str, list] = {}
        for match in matches:
            date_key = match.kickoff_utc.strftime("%Y-%m-%d")
            if date_key not in matches_by_date:
                matches_by_date[date_key] = []
            matches_by_date[date_key].append(match)

        # Fetch Sofascore events for each date
        sofascore_events_by_date: dict[str, list] = {}
        for date_str in matches_by_date.keys():
            date_obj = datetime.strptime(date_str, "%Y-%m-%d")
            events = await provider.get_scheduled_events(date_obj)
            sofascore_events_by_date[date_str] = events
            logger.debug(f"[SOFASCORE_REFS] Fetched {len(events)} Sofascore events for {date_str}")

        # Process each match
        for match in matches:
            match_id = match.match_id
            kickoff_utc = match.kickoff_utc
            home_team = match.home_team
            away_team = match.away_team
            existing_ref = match.existing_sofascore_id

            try:
                # Skip if already linked
                if existing_ref:
                    metrics["already_linked"] += 1
                    continue

                # Get candidates from same date
                date_key = kickoff_utc.strftime("%Y-%m-%d")
                candidates = sofascore_events_by_date.get(date_key, [])

                if not candidates:
                    metrics["skipped_no_candidates"] += 1
                    continue

                # Find best match — track top-3 for near-miss logging
                best_score = 0.0
                best_event = None
                best_matched_by = ""
                top_candidates: list[tuple[float, str, dict]] = []

                for event in candidates:
                    score, matched_by = calculate_match_score(
                        our_home=home_team,
                        our_away=away_team,
                        our_kickoff=kickoff_utc,
                        sf_home=event["home_team"],
                        sf_away=event["away_team"],
                        sf_kickoff=event["kickoff_utc"],
                        alias_index=alias_index,
                    )

                    # Maintain top-3 candidates for near-miss logging
                    top_candidates.append((score, matched_by, event))
                    top_candidates.sort(key=lambda x: x[0], reverse=True)
                    top_candidates = top_candidates[:3]

                    if score > best_score:
                        best_score = score
                        best_event = event
                        best_matched_by = matched_by

                # Decision based on score (configurable threshold per league)
                threshold = get_sofascore_threshold(match.league_id)
                if best_score < threshold:
                    metrics["skipped_low_score"] += 1

                    # Near-miss logging: top-3 candidates when 0.50 <= score < threshold
                    if best_score >= 0.50:
                        metrics["near_misses"] += 1
                        candidates_summary = " | ".join(
                            f"#{i+1} sf={c[2]['home_team']} vs {c[2]['away_team']} "
                            f"score={c[0]:.3f} ({c[1]})"
                            for i, c in enumerate(top_candidates[:3])
                        )
                        logger.warning(
                            "[SOFASCORE_REFS] Near-miss: match=%d league=%d "
                            "our=%s vs %s | best_score=%.3f | "
                            "our_norm=%s|%s | candidates: %s",
                            match_id, match.league_id,
                            home_team, away_team,
                            best_score,
                            normalize_team_name(home_team),
                            normalize_team_name(away_team),
                            candidates_summary,
                        )
                    else:
                        logger.debug(
                            "[SOFASCORE_REFS] No match for %s vs %s (best_score=%.2f)",
                            home_team, away_team, best_score,
                        )
                    continue

                # Add needs_review flag if score is borderline
                if best_score < 0.90:
                    best_matched_by += ";needs_review"
                    metrics["linked_review"] += 1
                else:
                    metrics["linked_auto"] += 1

                # Upsert to match_external_refs
                await session.execute(text("""
                    INSERT INTO match_external_refs
                        (match_id, source, source_match_id, confidence, matched_by, created_at)
                    VALUES
                        (:match_id, 'sofascore', :source_match_id, :confidence, :matched_by, NOW())
                    ON CONFLICT (match_id, source) DO UPDATE SET
                        source_match_id = EXCLUDED.source_match_id,
                        confidence = EXCLUDED.confidence,
                        matched_by = EXCLUDED.matched_by
                """), {
                    "match_id": match_id,
                    "source_match_id": best_event["event_id"],
                    "confidence": best_score,
                    "matched_by": best_matched_by,
                })

                logger.debug(
                    f"[SOFASCORE_REFS] Linked match {match_id} ({home_team} vs {away_team}) "
                    f"-> sofascore:{best_event['event_id']} (score={best_score:.2f}, {best_matched_by})"
                )

            except Exception as e:
                metrics["errors"] += 1
                logger.error(f"[SOFASCORE_REFS] Error processing match {match_id}: {e}")

        await session.commit()

        # Log summary
        total_linked = metrics["linked_auto"] + metrics["linked_review"]
        logger.info(
            f"[SOFASCORE_REFS] Complete: scanned={metrics['scanned']}, "
            f"linked_auto={metrics['linked_auto']}, linked_review={metrics['linked_review']}, "
            f"skipped_low={metrics['skipped_low_score']}, near_misses={metrics['near_misses']}, "
            f"skipped_no_cand={metrics['skipped_no_candidates']}, "
            f"already={metrics['already_linked']}, errors={metrics['errors']}"
        )

    except Exception as e:
        metrics["errors"] += 1
        logger.error(f"[SOFASCORE_REFS] Job failed: {e}")

    finally:
        await provider.close()

    return metrics


async def capture_sofascore_xi_prekickoff(
    session: AsyncSession,
    hours: int = 48,
    limit: int = 100,
) -> dict:
    """
    Capture Sofascore XI data for upcoming matches.

    Fetches lineup/formation/ratings from Sofascore for matches that:
    - Have sofascore ref
    - Are NS and within N hours of kickoff
    - Don't already have XI data captured

    Args:
        session: Database session.
        hours: Hours ahead to look for NS matches.
        limit: Max matches to process.

    Returns:
        Dict with metrics: matches_checked, captured, skipped_*, errors.
    """
    from app.etl.sofascore_provider import SofascoreProvider

    metrics = {
        "matches_checked": 0,
        "with_ref": 0,
        "captured": 0,
        "skipped_no_ref": 0,
        "skipped_already_captured": 0,
        "skipped_no_data": 0,
        "skipped_low_integrity": 0,
        "errors": 0,
    }

    provider = SofascoreProvider(use_mock=False)

    try:
        # Find NS matches with sofascore ref, without XI data
        league_ids_str = ",".join(str(lid) for lid in SOFASCORE_SUPPORTED_LEAGUES)
        limit_clause = f"LIMIT {limit}" if limit else ""

        result = await session.execute(text(f"""
            SELECT
                m.id AS match_id,
                m.date AS kickoff_utc,
                m.league_id,
                t_home.name AS home_team,
                t_away.name AS away_team,
                mer.source_match_id AS sofascore_id
            FROM matches m
            JOIN teams t_home ON m.home_team_id = t_home.id
            JOIN teams t_away ON m.away_team_id = t_away.id
            LEFT JOIN match_external_refs mer
                ON m.id = mer.match_id AND mer.source = 'sofascore'
            LEFT JOIN match_sofascore_lineup msl
                ON m.id = msl.match_id
            WHERE m.status = 'NS'
              AND m.date >= NOW()
              AND m.date < NOW() + INTERVAL '{hours} hours'
              AND m.league_id IN ({league_ids_str})
              AND msl.match_id IS NULL
            ORDER BY m.date ASC
            {limit_clause}
        """))

        matches = result.fetchall()
        metrics["matches_checked"] = len(matches)

        if not matches:
            logger.debug(f"[SOFASCORE_XI] No matches need XI capture (next {hours}h)")
            return metrics

        logger.info(f"[SOFASCORE_XI] Found {len(matches)} matches to check for XI")

        for match in matches:
            match_id = match.match_id
            sofascore_id = match.sofascore_id
            kickoff_utc = match.kickoff_utc
            cc = LEAGUE_PROXY_COUNTRY.get(match.league_id)

            try:
                if not sofascore_id:
                    metrics["skipped_no_ref"] += 1
                    continue

                metrics["with_ref"] += 1

                # Fetch lineup from Sofascore (geo-proxy by league country)
                lineup_data = await provider.get_match_lineup(sofascore_id, country_code=cc)

                if lineup_data.error:
                    if lineup_data.error == "not_found":
                        metrics["skipped_no_data"] += 1
                    else:
                        metrics["errors"] += 1
                        logger.warning(
                            f"[SOFASCORE_XI] Error fetching XI for match {match_id}: "
                            f"{lineup_data.error}"
                        )
                    continue

                # Check integrity score (need at least basic lineup)
                if lineup_data.integrity_score < 0.3:
                    metrics["skipped_low_integrity"] += 1
                    logger.debug(
                        f"[SOFASCORE_XI] Low integrity ({lineup_data.integrity_score}) "
                        f"for match {match_id}"
                    )
                    continue

                # Upsert lineup and player data
                await _upsert_sofascore_lineup(
                    session, match_id, lineup_data, kickoff_utc
                )
                metrics["captured"] += 1

                logger.debug(
                    f"[SOFASCORE_XI] Captured XI for match {match_id}: "
                    f"integrity={lineup_data.integrity_score:.2f}"
                )

            except Exception as e:
                metrics["errors"] += 1
                logger.error(f"[SOFASCORE_XI] Error processing match {match_id}: {e}")
                continue

        await session.commit()
        logger.info(
            f"[SOFASCORE_XI] Complete: captured={metrics['captured']}, "
            f"skipped_no_ref={metrics['skipped_no_ref']}, "
            f"errors={metrics['errors']}"
        )

    except Exception as e:
        metrics["errors"] += 1
        logger.error(f"[SOFASCORE_XI] Job failed: {e}")

    finally:
        await provider.close()

    return metrics


async def _upsert_sofascore_lineup(
    session: AsyncSession,
    match_id: int,
    lineup_data,
    kickoff_utc: datetime,
) -> None:
    """
    Upsert Sofascore lineup and player data to DB.

    Ensures PIT correctness: captured_at < kickoff_utc.
    """
    captured_at = lineup_data.captured_at or datetime.utcnow()

    # Validate PIT: captured_at must be before kickoff
    if kickoff_utc and captured_at >= kickoff_utc:
        logger.warning(
            f"[SOFASCORE_XI] PIT violation for match {match_id}: "
            f"captured_at={captured_at} >= kickoff={kickoff_utc}"
        )
        # Still save but flag it (data might be post-kickoff)

    # Process home lineup
    if lineup_data.home:
        await _upsert_team_lineup(
            session, match_id, "home", lineup_data.home, captured_at
        )

    # Process away lineup
    if lineup_data.away:
        await _upsert_team_lineup(
            session, match_id, "away", lineup_data.away, captured_at
        )


async def _upsert_team_lineup(
    session: AsyncSession,
    match_id: int,
    team_side: str,
    team_lineup,
    captured_at: datetime,
) -> None:
    """Upsert lineup for a single team (home/away)."""
    # Upsert lineup record
    formation = team_lineup.formation or "unknown"

    await session.execute(text("""
        INSERT INTO match_sofascore_lineup (match_id, team_side, formation, captured_at)
        VALUES (:match_id, :team_side, :formation, :captured_at)
        ON CONFLICT (match_id, team_side)
        DO UPDATE SET
            formation = EXCLUDED.formation,
            captured_at = EXCLUDED.captured_at
    """), {
        "match_id": match_id,
        "team_side": team_side,
        "formation": formation,
        "captured_at": captured_at,
    })

    # Upsert player records
    for player in team_lineup.players:
        await session.execute(text("""
            INSERT INTO match_sofascore_player (
                match_id, team_side, player_id_ext, position,
                is_starter, rating_pre_match, rating_recent_form,
                minutes_expected, captured_at
            ) VALUES (
                :match_id, :team_side, :player_id_ext, :position,
                :is_starter, :rating_pre_match, :rating_recent_form,
                :minutes_expected, :captured_at
            )
            ON CONFLICT (match_id, team_side, player_id_ext)
            DO UPDATE SET
                position = EXCLUDED.position,
                is_starter = EXCLUDED.is_starter,
                rating_pre_match = EXCLUDED.rating_pre_match,
                rating_recent_form = EXCLUDED.rating_recent_form,
                minutes_expected = EXCLUDED.minutes_expected,
                captured_at = EXCLUDED.captured_at
        """), {
            "match_id": match_id,
            "team_side": team_side,
            "player_id_ext": player.player_id_ext,
            "position": player.position,
            "is_starter": player.is_starter,
            "rating_pre_match": player.rating_pre_match,
            "rating_recent_form": player.rating_recent_form,
            "minutes_expected": player.minutes_expected,
            "captured_at": captured_at,
        })


# =============================================================================
# VENUE GEO EXPAND (REAL)
# =============================================================================

async def expand_venue_geo(
    session: AsyncSession,
    limit: int = 50,
) -> dict:
    """
    Expand venue_geo table with coordinates for new venues.

    Uses Open-Meteo Geocoding API to resolve city names to lat/lon.
    Priority: venues with most upcoming matches first.

    Args:
        session: Database session.
        limit: Max venues to process per run.

    Returns:
        Dict with metrics: venues_missing, venues_geocoded, skipped_*, errors.
    """
    import asyncio
    from app.etl.open_meteo_provider import OpenMeteoProvider

    metrics = {
        "venues_missing": 0,
        "venues_geocoded": 0,
        "skipped_no_city": 0,
        "skipped_no_country": 0,
        "skipped_no_result": 0,
        "errors": 0,
    }

    provider = OpenMeteoProvider(use_mock=False)

    try:
        # Find venue cities from recent/upcoming matches without geo data
        # Order by frequency (most used venues first) to maximize impact
        result = await session.execute(text("""
            SELECT m.venue_city, t_home.country, COUNT(*) as match_count
            FROM matches m
            JOIN teams t_home ON m.home_team_id = t_home.id
            LEFT JOIN venue_geo vg
                ON m.venue_city = vg.venue_city
                AND t_home.country = vg.country
            WHERE m.venue_city IS NOT NULL
              AND t_home.country IS NOT NULL
              AND m.date >= NOW() - INTERVAL '30 days'
              AND m.date < NOW() + INTERVAL '30 days'
              AND vg.venue_city IS NULL
            GROUP BY m.venue_city, t_home.country
            ORDER BY COUNT(*) DESC
            LIMIT :limit
        """), {"limit": limit})

        venues = result.fetchall()
        metrics["venues_missing"] = len(venues)

        if not venues:
            logger.debug("[SOTA_GEO] No missing venue geo data")
            return metrics

        logger.info(f"[SOTA_GEO] Found {len(venues)} venues to geocode")

        for row in venues:
            venue_city = row.venue_city
            country = row.country

            try:
                # Validate inputs
                if not venue_city or not venue_city.strip():
                    metrics["skipped_no_city"] += 1
                    continue

                if not country or not country.strip():
                    metrics["skipped_no_country"] += 1
                    continue

                # Rate limiting
                await asyncio.sleep(GEOCODING_DELAY_SECONDS)

                # Geocode the city
                geo_result = await provider.geocode_city(venue_city, country)

                if geo_result is None:
                    metrics["skipped_no_result"] += 1
                    logger.debug(f"[SOTA_GEO] No geocoding result for: {venue_city}, {country}")
                    continue

                # UPSERT into venue_geo
                await session.execute(text("""
                    INSERT INTO venue_geo (venue_city, country, lat, lon, source, confidence)
                    VALUES (:venue_city, :country, :lat, :lon, :source, :confidence)
                    ON CONFLICT (venue_city, country)
                    DO UPDATE SET
                        lat = EXCLUDED.lat,
                        lon = EXCLUDED.lon,
                        source = EXCLUDED.source,
                        confidence = EXCLUDED.confidence
                """), {
                    "venue_city": venue_city,
                    "country": country,
                    "lat": geo_result.lat,
                    "lon": geo_result.lon,
                    "source": geo_result.source,
                    "confidence": geo_result.confidence,
                })

                metrics["venues_geocoded"] += 1
                logger.debug(
                    f"[SOTA_GEO] Geocoded: {venue_city}, {country} -> "
                    f"({geo_result.lat}, {geo_result.lon}) conf={geo_result.confidence}"
                )

            except Exception as e:
                metrics["errors"] += 1
                logger.error(f"[SOTA_GEO] Error geocoding {venue_city}, {country}: {e}")
                continue

        await session.commit()
        logger.info(
            f"[SOTA_GEO] Complete: geocoded={metrics['venues_geocoded']}, "
            f"skipped_no_result={metrics['skipped_no_result']}, errors={metrics['errors']}"
        )

    except Exception as e:
        metrics["errors"] += 1
        logger.error(f"[SOTA_GEO] Job failed: {e}")

    finally:
        await provider.close()

    return metrics


# =============================================================================
# SOFASCORE POST-MATCH RATINGS BACKFILL
# =============================================================================

async def backfill_sofascore_ratings_ft(
    session: AsyncSession,
    days: int = 14,
    limit: int = 100,
) -> dict:
    """
    Backfill Sofascore player ratings for finished matches.

    Re-fetches /event/{id}/lineups for FT matches (ratings now available post-match)
    and inserts into sofascore_player_rating_history.

    Args:
        session: Database session.
        days: Days back to scan.
        limit: Max matches to process per run.

    Returns:
        Dict with metrics: scanned, inserted, skipped_*, errors.
    """
    from app.etl.sofascore_provider import SofascoreProvider

    metrics = {
        "scanned": 0,
        "matches_processed": 0,
        "players_inserted": 0,
        "skipped_no_ref": 0,
        "skipped_no_data": 0,
        "skipped_no_ratings": 0,
        "errors": 0,
    }

    provider = SofascoreProvider(use_mock=False)
    cutoff = datetime.utcnow() - timedelta(days=days)

    try:
        league_ids_str = ",".join(str(lid) for lid in SOFASCORE_SUPPORTED_LEAGUES)

        # FT matches with sofascore ref, without entries in rating history
        result = await session.execute(text(f"""
            SELECT
                m.id AS match_id,
                m.date AS match_date,
                m.league_id,
                mer.source_match_id AS sofascore_id
            FROM matches m
            JOIN match_external_refs mer
                ON m.id = mer.match_id AND mer.source = 'sofascore'
            WHERE m.status IN ('FT', 'AET', 'PEN')
              AND m.date >= :cutoff
              AND m.league_id IN ({league_ids_str})
              AND NOT EXISTS (
                  SELECT 1 FROM sofascore_player_rating_history sprh
                  WHERE sprh.match_id = m.id
              )
            ORDER BY m.date DESC
            LIMIT :limit
        """), {"cutoff": cutoff, "limit": limit})

        matches = result.fetchall()
        metrics["scanned"] = len(matches)

        if not matches:
            logger.debug(f"[SOFASCORE_RATINGS] No matches need ratings (last {days}d)")
            return metrics

        logger.info(f"[SOFASCORE_RATINGS] Found {len(matches)} matches to backfill ratings")

        for match in matches:
            match_id = match.match_id
            sofascore_id = match.sofascore_id
            match_date = match.match_date
            cc = LEAGUE_PROXY_COUNTRY.get(match.league_id)

            try:
                # Re-fetch lineups (post-match: ratings now available, geo-proxy)
                lineup_data = await provider.get_match_lineup(sofascore_id, country_code=cc)

                if lineup_data.error:
                    metrics["skipped_no_data"] += 1
                    continue

                # Extract ratings from both sides
                players_inserted = 0
                for side_data in [lineup_data.home, lineup_data.away]:
                    if not side_data:
                        continue
                    for player in side_data.players:
                        # rating_recent_form contains the post-match rating
                        rating = player.rating_recent_form
                        if rating is None:
                            continue

                        await session.execute(text("""
                            INSERT INTO sofascore_player_rating_history (
                                player_id_ext, match_id, team_side, position,
                                rating, minutes_played, is_starter, match_date,
                                captured_at
                            ) VALUES (
                                :player_id_ext, :match_id, :team_side, :position,
                                :rating, :minutes_played, :is_starter, :match_date,
                                NOW()
                            )
                            ON CONFLICT (player_id_ext, match_id) DO UPDATE SET
                                rating = EXCLUDED.rating,
                                captured_at = NOW()
                        """), {
                            "player_id_ext": player.player_id_ext,
                            "match_id": match_id,
                            "team_side": side_data.team_side,
                            "position": player.position,
                            "rating": rating,
                            "minutes_played": None,  # Not available from lineups endpoint
                            "is_starter": player.is_starter,
                            "match_date": match_date,
                        })
                        players_inserted += 1

                if players_inserted == 0:
                    metrics["skipped_no_ratings"] += 1
                else:
                    metrics["matches_processed"] += 1
                    metrics["players_inserted"] += players_inserted

            except Exception as e:
                metrics["errors"] += 1
                logger.error(f"[SOFASCORE_RATINGS] Error processing match {match_id}: {e}")
                continue

        await session.commit()
        logger.info(
            f"[SOFASCORE_RATINGS] Complete: matches={metrics['matches_processed']}, "
            f"players={metrics['players_inserted']}, errors={metrics['errors']}"
        )

    except Exception as e:
        metrics["errors"] += 1
        logger.error(f"[SOFASCORE_RATINGS] Job failed: {e}")
        try:
            await session.rollback()
        except Exception:
            pass

    finally:
        await provider.close()

    return metrics


# =============================================================================
# SOFASCORE POST-MATCH STATS BACKFILL
# =============================================================================

async def backfill_sofascore_stats_ft(
    session: AsyncSession,
    days: int = 14,
    limit: int = 100,
) -> dict:
    """
    Backfill Sofascore post-match statistics for finished matches.

    Fetches /event/{id}/statistics and stores xG, big chances, etc.

    Args:
        session: Database session.
        days: Days back to scan.
        limit: Max matches to process per run.

    Returns:
        Dict with metrics: scanned, inserted, skipped_*, errors.
    """
    from app.etl.sofascore_provider import SofascoreProvider

    metrics = {
        "scanned": 0,
        "inserted": 0,
        "skipped_no_data": 0,
        "skipped_empty_stats": 0,
        "errors": 0,
    }

    provider = SofascoreProvider(use_mock=False)
    cutoff = datetime.utcnow() - timedelta(days=days)

    try:
        league_ids_str = ",".join(str(lid) for lid in SOFASCORE_SUPPORTED_LEAGUES)

        # FT matches with sofascore ref, without stats entry
        result = await session.execute(text(f"""
            SELECT
                m.id AS match_id,
                m.league_id,
                mer.source_match_id AS sofascore_id
            FROM matches m
            JOIN match_external_refs mer
                ON m.id = mer.match_id AND mer.source = 'sofascore'
            WHERE m.status IN ('FT', 'AET', 'PEN')
              AND m.date >= :cutoff
              AND m.league_id IN ({league_ids_str})
              AND NOT EXISTS (
                  SELECT 1 FROM match_sofascore_stats mss
                  WHERE mss.match_id = m.id
              )
            ORDER BY m.date DESC
            LIMIT :limit
        """), {"cutoff": cutoff, "limit": limit})

        matches = result.fetchall()
        metrics["scanned"] = len(matches)

        if not matches:
            logger.debug(f"[SOFASCORE_STATS] No matches need stats (last {days}d)")
            return metrics

        logger.info(f"[SOFASCORE_STATS] Found {len(matches)} matches to backfill stats")

        for match in matches:
            match_id = match.match_id
            sofascore_id = match.sofascore_id
            cc = LEAGUE_PROXY_COUNTRY.get(match.league_id)

            try:
                stats_data, error = await provider.get_match_statistics(sofascore_id, country_code=cc)

                if error:
                    metrics["skipped_no_data"] += 1
                    continue

                if not stats_data or len(stats_data) <= 1:  # Only raw_stats key
                    metrics["skipped_empty_stats"] += 1
                    continue

                # Build INSERT params from parsed stats
                import json
                raw_json = json.dumps(stats_data.get("raw_stats")) if stats_data.get("raw_stats") else None

                await session.execute(text("""
                    INSERT INTO match_sofascore_stats (
                        match_id, possession_home, possession_away,
                        total_shots_home, total_shots_away,
                        shots_on_target_home, shots_on_target_away,
                        xg_home, xg_away,
                        corners_home, corners_away,
                        fouls_home, fouls_away,
                        big_chances_home, big_chances_away,
                        big_chances_missed_home, big_chances_missed_away,
                        accurate_passes_home, accurate_passes_away,
                        pass_accuracy_home, pass_accuracy_away,
                        raw_stats, captured_at
                    ) VALUES (
                        :match_id, :possession_home, :possession_away,
                        :total_shots_home, :total_shots_away,
                        :shots_on_target_home, :shots_on_target_away,
                        :xg_home, :xg_away,
                        :corners_home, :corners_away,
                        :fouls_home, :fouls_away,
                        :big_chances_home, :big_chances_away,
                        :big_chances_missed_home, :big_chances_missed_away,
                        :accurate_passes_home, :accurate_passes_away,
                        :pass_accuracy_home, :pass_accuracy_away,
                        CAST(:raw_stats AS jsonb), NOW()
                    )
                    ON CONFLICT (match_id) DO UPDATE SET
                        possession_home = EXCLUDED.possession_home,
                        possession_away = EXCLUDED.possession_away,
                        total_shots_home = EXCLUDED.total_shots_home,
                        total_shots_away = EXCLUDED.total_shots_away,
                        shots_on_target_home = EXCLUDED.shots_on_target_home,
                        shots_on_target_away = EXCLUDED.shots_on_target_away,
                        xg_home = EXCLUDED.xg_home,
                        xg_away = EXCLUDED.xg_away,
                        corners_home = EXCLUDED.corners_home,
                        corners_away = EXCLUDED.corners_away,
                        fouls_home = EXCLUDED.fouls_home,
                        fouls_away = EXCLUDED.fouls_away,
                        big_chances_home = EXCLUDED.big_chances_home,
                        big_chances_away = EXCLUDED.big_chances_away,
                        big_chances_missed_home = EXCLUDED.big_chances_missed_home,
                        big_chances_missed_away = EXCLUDED.big_chances_missed_away,
                        accurate_passes_home = EXCLUDED.accurate_passes_home,
                        accurate_passes_away = EXCLUDED.accurate_passes_away,
                        pass_accuracy_home = EXCLUDED.pass_accuracy_home,
                        pass_accuracy_away = EXCLUDED.pass_accuracy_away,
                        raw_stats = EXCLUDED.raw_stats,
                        captured_at = NOW()
                """), {
                    "match_id": match_id,
                    "possession_home": stats_data.get("possession_home"),
                    "possession_away": stats_data.get("possession_away"),
                    "total_shots_home": stats_data.get("total_shots_home"),
                    "total_shots_away": stats_data.get("total_shots_away"),
                    "shots_on_target_home": stats_data.get("shots_on_target_home"),
                    "shots_on_target_away": stats_data.get("shots_on_target_away"),
                    "xg_home": stats_data.get("xg_home"),
                    "xg_away": stats_data.get("xg_away"),
                    "corners_home": stats_data.get("corners_home"),
                    "corners_away": stats_data.get("corners_away"),
                    "fouls_home": stats_data.get("fouls_home"),
                    "fouls_away": stats_data.get("fouls_away"),
                    "big_chances_home": stats_data.get("big_chances_home"),
                    "big_chances_away": stats_data.get("big_chances_away"),
                    "big_chances_missed_home": stats_data.get("big_chances_missed_home"),
                    "big_chances_missed_away": stats_data.get("big_chances_missed_away"),
                    "accurate_passes_home": stats_data.get("accurate_passes_home"),
                    "accurate_passes_away": stats_data.get("accurate_passes_away"),
                    "pass_accuracy_home": stats_data.get("pass_accuracy_home"),
                    "pass_accuracy_away": stats_data.get("pass_accuracy_away"),
                    "raw_stats": raw_json,
                })

                metrics["inserted"] += 1

            except Exception as e:
                metrics["errors"] += 1
                logger.error(f"[SOFASCORE_STATS] Error processing match {match_id}: {e}")
                continue

        await session.commit()
        logger.info(
            f"[SOFASCORE_STATS] Complete: inserted={metrics['inserted']}, "
            f"skipped_no_data={metrics['skipped_no_data']}, errors={metrics['errors']}"
        )

    except Exception as e:
        metrics["errors"] += 1
        logger.error(f"[SOFASCORE_STATS] Job failed: {e}")
        try:
            await session.rollback()
        except Exception:
            pass

    finally:
        await provider.close()

    return metrics


# =============================================================================
# FOTMOB xG JOBS (ABE P0 2026-02-08)
# =============================================================================


async def sync_fotmob_refs(
    session: AsyncSession,
    days: int = 7,
    limit: int = 200,
) -> dict:
    """
    Link finished matches to FotMob match IDs for xG-eligible leagues.

    P0-2: Auto-link at score >= 0.90, needs_review at 0.75-0.90.
    P0-8: Only confirmed leagues (eligible = config ∩ FOTMOB_CONFIRMED_XG_LEAGUES).

    Args:
        session: Database session.
        days: Days back to scan for unlinked FT matches.
        limit: Max matches to process per run.

    Returns:
        Dict with metrics.
    """
    from app.config import get_settings
    from app.etl.fotmob_provider import FotmobProvider
    from app.etl.sofascore_provider import calculate_match_score
    from app.etl.sofascore_aliases import build_alias_index
    from app.etl.sota_constants import (
        FOTMOB_CONFIRMED_XG_LEAGUES,
        LEAGUE_ID_TO_FOTMOB,
    )

    settings = get_settings()
    metrics = {
        "scanned": 0,
        "already_linked": 0,
        "linked_auto": 0,
        "linked_review": 0,
        "skipped_no_candidates": 0,
        "skipped_low_score": 0,
        "skipped_no_mapping": 0,
        "errors": 0,
    }

    # P0-8: Only process confirmed leagues
    parsed_leagues = {int(x) for x in settings.FOTMOB_XG_LEAGUES.split(",") if x.strip()}
    eligible_leagues = parsed_leagues & FOTMOB_CONFIRMED_XG_LEAGUES

    if not eligible_leagues:
        logger.info("[FOTMOB-REFS] No eligible leagues (config ∩ confirmed = ∅)")
        return metrics

    alias_index = build_alias_index()
    provider = FotmobProvider()

    try:
        league_ids_str = ",".join(str(lid) for lid in eligible_leagues)
        cutoff = datetime.utcnow() - timedelta(days=days)

        # FT matches in eligible leagues without fotmob ref
        result = await session.execute(text(f"""
            SELECT
                m.id AS match_id,
                m.date AS kickoff_utc,
                m.league_id,
                t_home.name AS home_team,
                t_away.name AS away_team
            FROM matches m
            JOIN teams t_home ON m.home_team_id = t_home.id
            JOIN teams t_away ON m.away_team_id = t_away.id
            LEFT JOIN match_external_refs mer
                ON m.id = mer.match_id AND mer.source = 'fotmob'
            WHERE m.status IN ('FT', 'AET', 'PEN')
              AND m.date >= :cutoff
              AND m.league_id IN ({league_ids_str})
              AND mer.id IS NULL
            ORDER BY m.date DESC
            LIMIT :limit
        """), {"cutoff": cutoff, "limit": limit})

        matches = result.fetchall()
        metrics["scanned"] = len(matches)

        if not matches:
            logger.debug("[FOTMOB-REFS] No unlinked FT matches found")
            return metrics

        # Group by league for efficient fixture fetching
        matches_by_league: dict[int, list] = {}
        for match in matches:
            lid = match.league_id
            if lid not in matches_by_league:
                matches_by_league[lid] = []
            matches_by_league[lid].append(match)

        # Process each league
        for league_id, league_matches in matches_by_league.items():
            fotmob_league_id = LEAGUE_ID_TO_FOTMOB.get(league_id)
            if not fotmob_league_id:
                metrics["skipped_no_mapping"] += len(league_matches)
                logger.warning("[FOTMOB-REFS] No FotMob mapping for league %d", league_id)
                continue

            cc = LEAGUE_PROXY_COUNTRY.get(league_id)
            fm_fixtures, error = await provider.get_league_fixtures(fotmob_league_id, cc)

            if error or not fm_fixtures:
                metrics["errors"] += 1
                logger.warning("[FOTMOB-REFS] Failed to fetch fixtures for league %d: %s",
                               league_id, error)
                continue

            # Only finished FotMob fixtures
            fm_finished = [f for f in fm_fixtures if f.status == "finished"]

            for match in league_matches:
                try:
                    best_score = 0.0
                    best_fixture = None
                    best_matched_by = ""

                    for fm in fm_finished:
                        score, matched_by = calculate_match_score(
                            our_home=match.home_team,
                            our_away=match.away_team,
                            our_kickoff=match.kickoff_utc,
                            sf_home=fm.home_team,
                            sf_away=fm.away_team,
                            sf_kickoff=fm.kickoff_utc,
                            alias_index=alias_index,
                        )
                        if score > best_score:
                            best_score = score
                            best_fixture = fm
                            best_matched_by = matched_by

                    if best_score >= 0.90 and best_fixture:
                        # Auto-link (P0-2)
                        await session.execute(text("""
                            INSERT INTO match_external_refs
                                (match_id, source, source_match_id, confidence, matched_by)
                            VALUES (:match_id, 'fotmob', :source_match_id, :confidence, :matched_by)
                            ON CONFLICT (match_id, source) DO UPDATE SET
                                source_match_id = EXCLUDED.source_match_id,
                                confidence = EXCLUDED.confidence,
                                matched_by = EXCLUDED.matched_by
                        """), {
                            "match_id": match.match_id,
                            "source_match_id": str(best_fixture.fotmob_id),
                            "confidence": best_score,
                            "matched_by": best_matched_by,
                        })
                        metrics["linked_auto"] += 1

                    elif best_score >= 0.75 and best_fixture:
                        # Needs review
                        await session.execute(text("""
                            INSERT INTO match_external_refs
                                (match_id, source, source_match_id, confidence, matched_by)
                            VALUES (:match_id, 'fotmob', :source_match_id, :confidence, :matched_by)
                            ON CONFLICT (match_id, source) DO UPDATE SET
                                source_match_id = EXCLUDED.source_match_id,
                                confidence = EXCLUDED.confidence,
                                matched_by = EXCLUDED.matched_by
                        """), {
                            "match_id": match.match_id,
                            "source_match_id": str(best_fixture.fotmob_id),
                            "confidence": best_score,
                            "matched_by": f"{best_matched_by};needs_review",
                        })
                        metrics["linked_review"] += 1

                    elif best_score > 0:
                        metrics["skipped_low_score"] += 1
                    else:
                        metrics["skipped_no_candidates"] += 1

                except Exception as e:
                    metrics["errors"] += 1
                    logger.error("[FOTMOB-REFS] Error linking match %d: %s", match.match_id, e)
                    continue

        await session.commit()
        logger.info(
            "[FOTMOB-REFS] Complete: scanned=%d linked_auto=%d linked_review=%d "
            "skipped_no_mapping=%d errors=%d",
            metrics["scanned"], metrics["linked_auto"], metrics["linked_review"],
            metrics["skipped_no_mapping"], metrics["errors"],
        )

    except Exception as e:
        metrics["errors"] += 1
        logger.error("[FOTMOB-REFS] Job failed: %s", e)
        try:
            await session.rollback()
        except Exception:
            pass

    finally:
        await provider.close()

    return metrics


async def backfill_fotmob_xg_ft(
    session: AsyncSession,
    days: int = 7,
    limit: int = 100,
) -> dict:
    """
    Fetch xG from FotMob for linked FT matches.

    P0-4: captured_at = match.date + 6h for backfill.
    P0-6: Only team-level xG/xGOT.
    P0-8: Only confirmed leagues.

    Args:
        session: Database session.
        days: Days back to scan.
        limit: Max matches to process per run.

    Returns:
        Dict with metrics.
    """
    import json as _json

    from app.config import get_settings
    from app.etl.fotmob_provider import FotmobProvider
    from app.etl.sota_constants import FOTMOB_CONFIRMED_XG_LEAGUES

    settings = get_settings()
    metrics = {
        "scanned": 0,
        "captured": 0,
        "skipped_no_xg": 0,
        "errors": 0,
    }

    # P0-8: Only process confirmed leagues
    parsed_leagues = {int(x) for x in settings.FOTMOB_XG_LEAGUES.split(",") if x.strip()}
    eligible_leagues = parsed_leagues & FOTMOB_CONFIRMED_XG_LEAGUES

    if not eligible_leagues:
        logger.info("[FOTMOB-XG] No eligible leagues (config ∩ confirmed = ∅)")
        return metrics

    provider = FotmobProvider()
    cutoff = datetime.utcnow() - timedelta(days=days)

    try:
        league_ids_str = ",".join(str(lid) for lid in eligible_leagues)

        # FT matches with fotmob ref, without stats entry
        result = await session.execute(text(f"""
            SELECT
                m.id AS match_id,
                m.date AS kickoff_utc,
                m.league_id,
                mer.source_match_id AS fotmob_id
            FROM matches m
            JOIN match_external_refs mer
                ON m.id = mer.match_id AND mer.source = 'fotmob'
            WHERE m.status IN ('FT', 'AET', 'PEN')
              AND m.date >= :cutoff
              AND m.league_id IN ({league_ids_str})
              AND NOT EXISTS (
                  SELECT 1 FROM match_fotmob_stats mfs
                  WHERE mfs.match_id = m.id
              )
            ORDER BY m.date DESC
            LIMIT :limit
        """), {"cutoff": cutoff, "limit": limit})

        matches = result.fetchall()
        metrics["scanned"] = len(matches)

        if not matches:
            logger.debug("[FOTMOB-XG] No matches need xG (last %dd)", days)
            return metrics

        logger.info("[FOTMOB-XG] Found %d matches to backfill xG", len(matches))

        for match in matches:
            match_id = match.match_id
            fotmob_id = int(match.fotmob_id)
            cc = LEAGUE_PROXY_COUNTRY.get(match.league_id)

            try:
                xg_data, error = await provider.get_match_xg(fotmob_id, cc)

                if error or not xg_data:
                    metrics["skipped_no_xg"] += 1
                    continue

                # P0-4: captured_at = match_date + 6h for backfill
                captured_at = match.kickoff_utc + timedelta(hours=6)

                raw_json = _json.dumps(xg_data.raw_stats) if xg_data.raw_stats else None

                await session.execute(text("""
                    INSERT INTO match_fotmob_stats (
                        match_id, xg_home, xg_away, xgot_home, xgot_away,
                        xg_open_play_home, xg_open_play_away,
                        xg_set_play_home, xg_set_play_away,
                        raw_stats, captured_at, source_version
                    ) VALUES (
                        :match_id, :xg_home, :xg_away, :xgot_home, :xgot_away,
                        :xg_open_play_home, :xg_open_play_away,
                        :xg_set_play_home, :xg_set_play_away,
                        CAST(:raw_stats AS jsonb), :captured_at, :source_version
                    )
                    ON CONFLICT (match_id) DO UPDATE SET
                        xg_home = EXCLUDED.xg_home, xg_away = EXCLUDED.xg_away,
                        xgot_home = EXCLUDED.xgot_home, xgot_away = EXCLUDED.xgot_away,
                        xg_open_play_home = EXCLUDED.xg_open_play_home,
                        xg_open_play_away = EXCLUDED.xg_open_play_away,
                        xg_set_play_home = EXCLUDED.xg_set_play_home,
                        xg_set_play_away = EXCLUDED.xg_set_play_away,
                        raw_stats = EXCLUDED.raw_stats,
                        captured_at = EXCLUDED.captured_at
                """), {
                    "match_id": match_id,
                    "xg_home": xg_data.xg_home,
                    "xg_away": xg_data.xg_away,
                    "xgot_home": xg_data.xgot_home,
                    "xgot_away": xg_data.xgot_away,
                    "xg_open_play_home": xg_data.xg_open_play_home,
                    "xg_open_play_away": xg_data.xg_open_play_away,
                    "xg_set_play_home": xg_data.xg_set_play_home,
                    "xg_set_play_away": xg_data.xg_set_play_away,
                    "raw_stats": raw_json,
                    "captured_at": captured_at,
                    "source_version": provider.SCHEMA_VERSION,
                })

                metrics["captured"] += 1

            except Exception as e:
                metrics["errors"] += 1
                logger.error("[FOTMOB-XG] Error processing match %d: %s", match_id, e)
                continue

        await session.commit()
        logger.info(
            "[FOTMOB-XG] Complete: captured=%d skipped_no_xg=%d errors=%d",
            metrics["captured"], metrics["skipped_no_xg"], metrics["errors"],
        )

    except Exception as e:
        metrics["errors"] += 1
        logger.error("[FOTMOB-XG] Job failed: %s", e)
        try:
            await session.rollback()
        except Exception:
            pass

    finally:
        await provider.close()

    return metrics
