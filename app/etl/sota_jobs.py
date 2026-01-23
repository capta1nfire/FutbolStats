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

logger = logging.getLogger(__name__)

# Understat covers these leagues (API-Football league IDs)
UNDERSTAT_SUPPORTED_LEAGUES = {
    39,   # Premier League (England)
    140,  # La Liga (Spain)
    135,  # Serie A (Italy)
    78,   # Bundesliga (Germany)
    61,   # Ligue 1 (France)
}

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
                    api_match, provider, understat_cache
                )

                if not candidates:
                    metrics["skipped_no_candidates"] += 1
                    continue

                # Find best match
                best_candidate = None
                best_score = 0.0
                for candidate in candidates:
                    score = compute_match_score(api_match, candidate)
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

        if _is_match_candidate(api_match, u_match):
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


def _is_match_candidate(api_match: dict, u_match: dict, tolerance_hours: int = 2) -> bool:
    """Check if Understat match could match the API match."""
    import unicodedata

    def normalize(name: str) -> str:
        if not name:
            return ""
        name = name.lower()
        name = unicodedata.normalize('NFKD', name).encode('ascii', 'ignore').decode('ascii')
        name = name.replace(" fc", "").replace("fc ", "")
        name = name.replace(" cf", "").replace("cf ", "")
        return " ".join(name.split())

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

    api_home = normalize(api_match.get("home_team", ""))
    api_away = normalize(api_match.get("away_team", ""))
    u_home = normalize(u_match.get("home_team", ""))
    u_away = normalize(u_match.get("away_team", ""))

    home_match = api_home in u_home or u_home in api_home or api_home == u_home
    away_match = api_away in u_away or u_away in api_away or api_away == u_away

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
                logger.error(f"[SOTA_XG] Error processing match {match_id}: {e}")
                continue

        await session.commit()
        logger.info(
            f"[SOTA_XG] Complete: inserted={metrics['inserted']}, "
            f"updated={metrics['updated']}, errors={metrics['errors']}"
        )

    except Exception as e:
        metrics["errors"] += 1
        logger.error(f"[SOTA_XG] Job failed: {e}")

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
        result = await session.execute(text(f"""
            SELECT
                m.id AS match_id,
                m.date AS kickoff_utc,
                m.venue_city,
                t_home.country AS home_country,
                vg.lat,
                vg.lon
            FROM matches m
            JOIN teams t_home ON m.home_team_id = t_home.id
            LEFT JOIN venue_geo vg
                ON m.venue_city = vg.venue_city
                AND t_home.country = vg.country
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

            try:
                if lat is None or lon is None:
                    metrics["skipped_no_geo"] += 1
                    continue

                metrics["with_geo"] += 1

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
                precip_mm = :precip_mm, pressure_hpa = :pressure_hpa,
                cloudcover = :cloudcover, is_daylight = :is_daylight,
                captured_at = :captured_at
            WHERE match_id = :match_id AND forecast_horizon_hours = :forecast_horizon_hours
        """), params)
        return "updated"
    else:
        await session.execute(text("""
            INSERT INTO match_weather (
                match_id, temp_c, humidity, wind_ms, precip_mm,
                pressure_hpa, cloudcover, is_daylight,
                forecast_horizon_hours, captured_at
            ) VALUES (
                :match_id, :temp_c, :humidity, :wind_ms, :precip_mm,
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
