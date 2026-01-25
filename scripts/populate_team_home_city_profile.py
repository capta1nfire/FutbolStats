#!/usr/bin/env python3
"""
Populate team_home_city_profile for bio-adaptability features.

Derives home_city from historical match data (venue_city mode where team played home).
Resolves timezone using geocoding. Climate normals are stubbed for now.

Usage:
    # Dry-run to see what would be processed
    DATABASE_URL="postgresql://..." python scripts/populate_team_home_city_profile.py --dry-run --active-only

    # Real mode with geocoding (active teams only)
    DATABASE_URL="postgresql://..." python scripts/populate_team_home_city_profile.py --active-only --limit 500

    # Mock mode (for testing)
    DATABASE_URL="postgresql://..." python scripts/populate_team_home_city_profile.py --mock

    # Verbose logging
    DATABASE_URL="postgresql://..." python scripts/populate_team_home_city_profile.py -v

Notes:
    - Only processes clubs (team_type='club') with country set
    - home_city derived from most frequent venue_city in home matches
    - Minimum 3 home matches required for confidence
    - timezone resolved via TimeZoneDB or left NULL if unavailable

Reference: docs/ARCHITECTURE_SOTA.md section 1.3
"""

import argparse
import asyncio
import hashlib
import json
import logging
import os
import sys
from datetime import datetime
from typing import Optional

import httpx
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

# Add app to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Minimum home matches to derive home_city with confidence
MIN_HOME_MATCHES = 3

# Open-Meteo Geocoding API for timezone lookup
GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"


async def get_club_teams_needing_profile(
    session: AsyncSession,
    limit: Optional[int] = None,
    active_only: bool = False,
) -> list[dict]:
    """
    Get club teams that don't have a home city profile yet.

    Only includes teams with country set (needed for geocoding).
    If active_only=True, only includes teams with matches in last 30 days.
    """
    limit_clause = f"LIMIT {limit}" if limit else ""

    if active_only:
        # Only teams with matches in last 30 days
        query = text(f"""
            SELECT DISTINCT
                t.id AS team_id,
                t.name,
                t.country
            FROM teams t
            JOIN matches m ON t.id = m.home_team_id OR t.id = m.away_team_id
            LEFT JOIN team_home_city_profile thcp ON t.id = thcp.team_id
            WHERE t.team_type = 'club'
              AND t.country IS NOT NULL
              AND t.country != ''
              AND thcp.team_id IS NULL
              AND m.date >= NOW() - INTERVAL '30 days'
            ORDER BY t.id
            {limit_clause}
        """)
    else:
        query = text(f"""
            SELECT
                t.id AS team_id,
                t.name,
                t.country
            FROM teams t
            LEFT JOIN team_home_city_profile thcp ON t.id = thcp.team_id
            WHERE t.team_type = 'club'
              AND t.country IS NOT NULL
              AND t.country != ''
              AND thcp.team_id IS NULL
            ORDER BY t.id
            {limit_clause}
        """)

    result = await session.execute(query)
    rows = result.fetchall()

    return [
        {
            "team_id": row.team_id,
            "name": row.name,
            "country": row.country,
        }
        for row in rows
    ]


async def derive_home_city(
    session: AsyncSession,
    team_id: int,
) -> Optional[tuple[str, int]]:
    """
    Derive home_city from historical home match venue_city mode.

    Returns (home_city, match_count) or None if insufficient data.
    """
    query = text("""
        SELECT
            m.venue_city,
            COUNT(*) AS match_count
        FROM matches m
        WHERE m.home_team_id = :team_id
          AND m.venue_city IS NOT NULL
          AND m.venue_city != ''
          AND m.status IN ('FT', 'AET', 'PEN')
        GROUP BY m.venue_city
        ORDER BY match_count DESC
        LIMIT 1
    """)

    result = await session.execute(query, {"team_id": team_id})
    row = result.fetchone()

    if row and row.match_count >= MIN_HOME_MATCHES:
        return (row.venue_city.strip(), row.match_count)

    return None


async def resolve_timezone(
    client: httpx.AsyncClient,
    city: str,
    country: str,
) -> Optional[str]:
    """
    Resolve timezone for a city using Open-Meteo geocoding.

    Open-Meteo returns timezone in the geocoding response.
    Note: Open-Meteo search works best with city name only, then filter by country.
    """
    try:
        # Search by city name only (Open-Meteo doesn't support "city, country" format well)
        params = {
            "name": city,
            "count": 10,  # Get multiple results to filter by country
            "language": "en",
            "format": "json",
        }

        response = await client.get(GEOCODING_URL, params=params)
        response.raise_for_status()

        data = response.json()
        results = data.get("results", [])

        if not results:
            logger.debug(f"No geocoding results for '{city}'")
            return None

        # Try to find a result matching the country
        country_lower = country.lower()
        for result in results:
            result_country = result.get("country", "").lower()
            result_admin1 = result.get("admin1", "").lower()

            # Match on country or admin1 (for UK: "England" is admin1, country is "United Kingdom")
            if (country_lower in result_country or
                result_country in country_lower or
                country_lower in result_admin1 or
                result_admin1 in country_lower):
                return result.get("timezone")

        # Fallback: return first result's timezone
        logger.debug(f"No country match for '{city}' in '{country}', using first result")
        return results[0].get("timezone")

    except Exception as e:
        logger.debug(f"Timezone lookup failed for '{city}, {country}': {e}")
        return None


def generate_mock_timezone(city: str, country: str) -> str:
    """Generate deterministic mock timezone based on country."""
    # Common country -> timezone mapping for mocking
    country_tz = {
        "england": "Europe/London",
        "spain": "Europe/Madrid",
        "germany": "Europe/Berlin",
        "france": "Europe/Paris",
        "italy": "Europe/Rome",
        "portugal": "Europe/Lisbon",
        "netherlands": "Europe/Amsterdam",
        "belgium": "Europe/Brussels",
        "turkey": "Europe/Istanbul",
        "argentina": "America/Buenos_Aires",
        "brazil": "America/Sao_Paulo",
        "mexico": "America/Mexico_City",
        "usa": "America/New_York",
        "japan": "Asia/Tokyo",
        "australia": "Australia/Sydney",
    }

    country_lower = country.lower()
    for key, tz in country_tz.items():
        if key in country_lower:
            return tz

    return "UTC"


def generate_mock_climate_normals(city: str, country: str) -> dict:
    """
    Generate deterministic mock climate normals.

    Returns a dict with keys "01".."12" and temp_c_mean, humidity_mean.
    """
    # Use hash for deterministic values
    seed = f"{city.lower()}_{country.lower()}"
    hash_val = int(hashlib.md5(seed.encode()).hexdigest()[:8], 16)

    # Base temperature varies by "latitude" (determined by hash)
    base_temp = 10 + (hash_val % 15)  # 10-25 base
    humidity_base = 50 + (hash_val % 30)  # 50-80 base

    normals = {}
    for month in range(1, 13):
        # Seasonal variation (northern hemisphere pattern)
        seasonal = 10 * ((month - 7) ** 2 / 36 - 1)  # Peak in July
        temp = round(base_temp + seasonal, 1)
        humidity = round(humidity_base + (6 - abs(month - 6)) * 2, 0)

        normals[f"{month:02d}"] = {
            "temp_c_mean": temp,
            "humidity_mean": min(95, max(30, humidity)),
        }

    return normals


async def upsert_team_home_city_profile(
    session: AsyncSession,
    team_id: int,
    home_city: str,
    timezone: Optional[str],
    climate_normals_by_month: dict,
) -> str:
    """
    Upsert team_home_city_profile entry.

    PK: team_id
    Returns: 'inserted' or 'updated'
    """
    # Check if exists
    check = await session.execute(
        text("SELECT 1 FROM team_home_city_profile WHERE team_id = :team_id"),
        {"team_id": team_id}
    )
    exists = check.scalar() is not None

    # Convert dict to JSON string for JSONB
    # Note: asyncpg handles JSON serialization automatically when passed as string
    climate_json = json.dumps(climate_normals_by_month)

    if exists:
        await session.execute(
            text("""
                UPDATE team_home_city_profile SET
                    home_city = :home_city,
                    timezone = :timezone,
                    climate_normals_by_month = CAST(:climate_normals AS jsonb)
                WHERE team_id = :team_id
            """),
            {
                "team_id": team_id,
                "home_city": home_city,
                "timezone": timezone,
                "climate_normals": climate_json,
            }
        )
        return "updated"
    else:
        await session.execute(
            text("""
                INSERT INTO team_home_city_profile (
                    team_id, home_city, timezone, climate_normals_by_month
                ) VALUES (
                    :team_id, :home_city, :timezone, CAST(:climate_normals AS jsonb)
                )
            """),
            {
                "team_id": team_id,
                "home_city": home_city,
                "timezone": timezone,
                "climate_normals": climate_json,
            }
        )
        return "inserted"


async def main(
    use_mock: bool = False,
    limit: Optional[int] = None,
    verbose: bool = False,
    dry_run: bool = False,
    active_only: bool = False,
):
    """Main population logic."""
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

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
        "skipped_no_home_city": 0,
        "skipped_no_timezone": 0,
        "errors": 0,
    }

    examples = []

    mode_str = "DRY-RUN" if dry_run else "LIVE"
    scope_str = "active teams (30d)" if active_only else "all teams"
    logger.info(f"Mode: {mode_str}, Scope: {scope_str}, Limit: {limit or 'none'}")

    try:
        async with async_session() as session:
            # Get teams needing profile
            teams = await get_club_teams_needing_profile(session, limit=limit, active_only=active_only)
            stats["scanned"] = len(teams)
            logger.info(f"Found {len(teams)} club teams needing home city profile ({scope_str})")

            if not teams:
                logger.info("No teams need profiling")
                return stats

            async with httpx.AsyncClient(timeout=30.0) as client:
                for team in teams:
                    team_id = team["team_id"]
                    team_name = team["name"]
                    country = team["country"]

                    try:
                        # 1) Derive home_city from match history
                        home_city_result = await derive_home_city(session, team_id)

                        if home_city_result is None:
                            stats["skipped_no_home_city"] += 1
                            logger.debug(
                                f"Team {team_id} ({team_name}): "
                                f"Insufficient home match data, skipping"
                            )
                            continue

                        home_city, match_count = home_city_result

                        # 2) Resolve timezone
                        if use_mock:
                            timezone = generate_mock_timezone(home_city, country)
                        else:
                            timezone = await resolve_timezone(client, home_city, country)
                            # Rate limit
                            await asyncio.sleep(0.2)

                            # Skip if no timezone found (NOT NULL constraint)
                            if timezone is None:
                                # Fallback to mock timezone based on country
                                timezone = generate_mock_timezone(home_city, country)
                                logger.debug(
                                    f"Team {team_id} ({team_name}): Using fallback tz={timezone}"
                                )

                        # 3) Climate normals (mock for now, real requires historical weather API)
                        if use_mock:
                            climate_normals = generate_mock_climate_normals(home_city, country)
                        else:
                            # Real climate normals would require Open-Meteo historical API
                            # For now, use empty dict as placeholder
                            climate_normals = {}

                        # Store examples for summary
                        if len(examples) < 10:
                            examples.append({
                                "team_id": team_id,
                                "team": team_name,
                                "home_city": home_city,
                                "country": country,
                                "timezone": timezone,
                                "matches": match_count,
                            })

                        # 4) Upsert (skip in dry-run)
                        if dry_run:
                            stats["inserted"] += 1  # Would be inserted
                            logger.debug(
                                f"[DRY-RUN] Team {team_id} ({team_name}): would insert "
                                f"home_city={home_city}, tz={timezone}"
                            )
                        else:
                            result = await upsert_team_home_city_profile(
                                session=session,
                                team_id=team_id,
                                home_city=home_city,
                                timezone=timezone,
                                climate_normals_by_month=climate_normals,
                            )
                            stats[result] += 1

                            # Commit after each successful insert
                            await session.commit()

                            logger.debug(
                                f"Team {team_id} ({team_name}): {result} "
                                f"home_city={home_city}, tz={timezone}"
                            )

                    except Exception as e:
                        stats["errors"] += 1
                        logger.error(f"Team {team_id} ({team_name}): Error - {e}")
                        await session.rollback()
                        continue

    finally:
        await engine.dispose()

    # Print summary
    logger.info("=" * 60)
    logger.info("TEAM HOME CITY PROFILE SUMMARY:")
    logger.info("=" * 60)
    logger.info(f"  Scanned:              {stats['scanned']}")
    logger.info(f"  Inserted:             {stats['inserted']}")
    logger.info(f"  Updated:              {stats['updated']}")
    logger.info(f"  Skipped (no home city): {stats['skipped_no_home_city']}")
    logger.info(f"  Errors:               {stats['errors']}")
    logger.info("")
    if examples:
        logger.info("Examples:")
        for ex in examples:
            logger.info(
                f"  - {ex['team']} ({ex['country']}): "
                f"{ex['home_city']}, tz={ex['timezone']}, {ex['matches']} home matches"
            )
    logger.info("=" * 60)

    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Populate team_home_city_profile for bio-adaptability"
    )
    parser.add_argument(
        "--limit", type=int, default=500, help="Limit teams to process (default: 500)"
    )
    parser.add_argument(
        "--mock", action="store_true", help="Use mock data for testing"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Show what would be done without making changes"
    )
    parser.add_argument(
        "--active-only", action="store_true", help="Only process teams with matches in last 30 days"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Verbose logging"
    )
    args = parser.parse_args()

    asyncio.run(main(
        use_mock=args.mock,
        limit=args.limit,
        verbose=args.verbose,
        dry_run=args.dry_run,
        active_only=args.active_only,
    ))
