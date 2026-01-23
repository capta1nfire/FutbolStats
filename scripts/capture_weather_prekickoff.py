#!/usr/bin/env python3
"""
Capture weather forecasts for upcoming matches.

Selects NS matches in the next 48 hours and captures weather forecasts
from Open-Meteo, storing them in match_weather.

Usage:
    # Default: next 48 hours, horizon=24h
    DATABASE_URL="postgresql://..." python scripts/capture_weather_prekickoff.py

    # Custom hours ahead
    DATABASE_URL="postgresql://..." python scripts/capture_weather_prekickoff.py --hours 72

    # Custom forecast horizon
    DATABASE_URL="postgresql://..." python scripts/capture_weather_prekickoff.py --horizon 1

    # With mock data (for testing)
    DATABASE_URL="postgresql://..." python scripts/capture_weather_prekickoff.py --mock

Requirements:
    - venue_geo must have entries for match venues to get lat/lon
    - Matches without venue_geo are skipped
    - matches.venue_city must be populated

Reference: docs/ARCHITECTURE_SOTA.md section 1.3
"""

import argparse
import asyncio
import logging
import os
import sys
from datetime import datetime, timedelta

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

# Add app to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.etl.open_meteo_provider import OpenMeteoProvider

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def get_upcoming_matches_with_geo(
    session: AsyncSession,
    hours_ahead: int = 48,
) -> list[dict]:
    """
    Get upcoming NS matches with venue geolocation.

    Joins matches with venue_geo to get lat/lon for weather lookup.
    """
    now = datetime.utcnow()
    cutoff = now + timedelta(hours=hours_ahead)

    query = text("""
        SELECT
            m.id AS match_id,
            m.external_id,
            m.date AS kickoff_utc,
            m.venue_city,
            t_home.country AS home_country,
            vg.lat,
            vg.lon
        FROM matches m
        LEFT JOIN teams t_home ON m.home_team_id = t_home.id
        LEFT JOIN venue_geo vg
            ON LOWER(TRIM(m.venue_city)) = LOWER(TRIM(vg.venue_city))
            AND (
                LOWER(TRIM(t_home.country)) = LOWER(TRIM(vg.country))
                OR vg.country IS NULL
            )
        WHERE m.status = 'NS'
          AND m.date >= :now
          AND m.date <= :cutoff
        ORDER BY m.date ASC
    """)

    result = await session.execute(query, {"now": now, "cutoff": cutoff})
    rows = result.fetchall()

    return [
        {
            "match_id": row.match_id,
            "external_id": row.external_id,
            "kickoff_utc": row.kickoff_utc,
            "venue_city": row.venue_city,
            "home_country": row.home_country,
            "lat": row.lat,
            "lon": row.lon,
        }
        for row in rows
    ]


async def upsert_weather_data(
    session: AsyncSession,
    match_id: int,
    horizon_hours: int,
    data: dict,
) -> str:
    """
    Upsert weather data into match_weather.

    PK: (match_id, forecast_horizon_hours)
    Returns: 'inserted' or 'updated'
    """
    # Check if exists
    check = await session.execute(
        text("""
            SELECT 1 FROM match_weather
            WHERE match_id = :match_id AND forecast_horizon_hours = :horizon
        """),
        {"match_id": match_id, "horizon": horizon_hours}
    )
    exists = check.scalar() is not None

    if exists:
        # Update
        await session.execute(
            text("""
                UPDATE match_weather SET
                    temp_c = :temp_c,
                    humidity = :humidity,
                    wind_ms = :wind_ms,
                    precip_mm = :precip_mm,
                    pressure_hpa = :pressure_hpa,
                    cloudcover = :cloudcover,
                    is_daylight = :is_daylight,
                    captured_at = :captured_at
                WHERE match_id = :match_id AND forecast_horizon_hours = :horizon
            """),
            {
                "match_id": match_id,
                "horizon": horizon_hours,
                "temp_c": data["temp_c"],
                "humidity": data["humidity"],
                "wind_ms": data["wind_ms"],
                "precip_mm": data["precip_mm"],
                "pressure_hpa": data.get("pressure_hpa"),
                "cloudcover": data.get("cloudcover"),
                "is_daylight": data["is_daylight"],
                "captured_at": data["captured_at"],
            }
        )
        return "updated"
    else:
        # Insert
        await session.execute(
            text("""
                INSERT INTO match_weather (
                    match_id, temp_c, humidity, wind_ms, precip_mm,
                    pressure_hpa, cloudcover, is_daylight,
                    forecast_horizon_hours, captured_at
                ) VALUES (
                    :match_id, :temp_c, :humidity, :wind_ms, :precip_mm,
                    :pressure_hpa, :cloudcover, :is_daylight,
                    :horizon, :captured_at
                )
            """),
            {
                "match_id": match_id,
                "horizon": horizon_hours,
                "temp_c": data["temp_c"],
                "humidity": data["humidity"],
                "wind_ms": data["wind_ms"],
                "precip_mm": data["precip_mm"],
                "pressure_hpa": data.get("pressure_hpa"),
                "cloudcover": data.get("cloudcover"),
                "is_daylight": data["is_daylight"],
                "captured_at": data["captured_at"],
            }
        )
        return "inserted"


async def main(
    hours_ahead: int = 48,
    horizon_hours: int = 24,
    use_mock: bool = False,
):
    """Main weather capture logic."""
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        raise ValueError("DATABASE_URL environment variable not set")

    # Convert to async URL
    if database_url.startswith("postgresql://"):
        database_url = database_url.replace("postgresql://", "postgresql+asyncpg://", 1)

    engine = create_async_engine(database_url, echo=False)
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    # Counters
    stats = {
        "scanned": 0,
        "inserted": 0,
        "updated": 0,
        "skipped_no_venue": 0,
        "skipped_no_geo": 0,
        "skipped_no_data": 0,
        "errors": 0,
    }

    provider = OpenMeteoProvider(use_mock=use_mock)

    try:
        async with async_session() as session:
            # Get upcoming matches with geo
            matches = await get_upcoming_matches_with_geo(session, hours_ahead=hours_ahead)
            stats["scanned"] = len(matches)
            logger.info(f"Found {len(matches)} NS matches in next {hours_ahead} hours")

            for match in matches:
                match_id = match["match_id"]

                try:
                    # Check if we have venue_city
                    if not match["venue_city"]:
                        stats["skipped_no_venue"] += 1
                        logger.debug(f"Match {match_id}: No venue_city, skipping")
                        continue

                    # Check if we have geo coordinates
                    if match["lat"] is None or match["lon"] is None:
                        stats["skipped_no_geo"] += 1
                        logger.debug(
                            f"Match {match_id}: No geo for venue '{match['venue_city']}', skipping"
                        )
                        continue

                    # Fetch weather from provider
                    weather = await provider.get_forecast(
                        lat=match["lat"],
                        lon=match["lon"],
                        kickoff_utc=match["kickoff_utc"],
                        horizon_hours=horizon_hours,
                    )

                    if weather is None:
                        stats["skipped_no_data"] += 1
                        logger.debug(f"Match {match_id}: No weather data from provider")
                        continue

                    # Upsert to DB
                    data = {
                        "temp_c": weather.temp_c,
                        "humidity": weather.humidity,
                        "wind_ms": weather.wind_ms,
                        "precip_mm": weather.precip_mm,
                        "pressure_hpa": weather.pressure_hpa,
                        "cloudcover": weather.cloudcover,
                        "is_daylight": weather.is_daylight,
                        "captured_at": weather.captured_at or datetime.utcnow(),
                    }

                    result = await upsert_weather_data(
                        session, match_id, horizon_hours, data
                    )
                    stats[result] += 1
                    logger.debug(f"Match {match_id}: {result}")

                    # Small delay to avoid rate limiting (Open-Meteo is generous but be polite)
                    await asyncio.sleep(0.1)

                except Exception as e:
                    stats["errors"] += 1
                    logger.error(f"Match {match_id}: Error - {e}")
                    continue

            await session.commit()

    finally:
        await provider.close()
        await engine.dispose()

    # Print summary
    logger.info("=" * 60)
    logger.info("WEATHER CAPTURE SUMMARY:")
    logger.info("=" * 60)
    logger.info(f"  Scanned:           {stats['scanned']}")
    logger.info(f"  Inserted:          {stats['inserted']}")
    logger.info(f"  Updated:           {stats['updated']}")
    logger.info(f"  Skipped (no venue): {stats['skipped_no_venue']}")
    logger.info(f"  Skipped (no geo):   {stats['skipped_no_geo']}")
    logger.info(f"  Skipped (no data):  {stats['skipped_no_data']}")
    logger.info(f"  Errors:            {stats['errors']}")
    logger.info("=" * 60)

    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Capture weather for upcoming matches")
    parser.add_argument(
        "--hours", type=int, default=48, help="Hours ahead to scan (default: 48)"
    )
    parser.add_argument(
        "--horizon", type=int, default=24, help="Forecast horizon in hours (default: 24)"
    )
    parser.add_argument("--mock", action="store_true", help="Use mock data for testing")
    args = parser.parse_args()

    asyncio.run(main(hours_ahead=args.hours, horizon_hours=args.horizon, use_mock=args.mock))
