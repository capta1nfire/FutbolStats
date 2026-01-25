#!/usr/bin/env python3
"""
Populate venue_geo with geographic coordinates for venues.

Resolves venue_city to lat/lon using geocoding services.
Required for weather feature capture.

Usage:
    # Mock mode (deterministic coords for testing)
    DATABASE_URL="postgresql://..." python scripts/populate_venue_geo.py --mock

    # Real mode with Open-Meteo geocoding
    DATABASE_URL="postgresql://..." python scripts/populate_venue_geo.py

    # Limit for testing
    DATABASE_URL="postgresql://..." python scripts/populate_venue_geo.py --limit 50

    # Verbose logging
    DATABASE_URL="postgresql://..." python scripts/populate_venue_geo.py -v

Notes:
    - Uses Open-Meteo Geocoding API (free, no key required)
    - Falls back to home team's country if no country in venue
    - UPSERT by (venue_city, country)
    - Rate limited to avoid API abuse

Reference: docs/ARCHITECTURE_SOTA.md section 1.3
"""

import argparse
import asyncio
import hashlib
import logging
import os
import sys
from datetime import datetime, timezone
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

# Open-Meteo Geocoding API
GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"


async def get_unique_venues(
    session: AsyncSession,
    limit: Optional[int] = None,
    recent_days: int = 30,
) -> list[dict]:
    """
    Get unique venues from matches that need geocoding.

    Joins with home team to get country if venue doesn't have one.
    Only returns venues not already in venue_geo.
    Prioritizes by match_count (most used venues first).

    Args:
        session: Database session.
        limit: Max venues to return.
        recent_days: Only consider matches from last N days (default: 30).
    """
    limit_clause = f"LIMIT {limit}" if limit else ""

    query = text(f"""
        WITH recent_venues AS (
            SELECT
                m.venue_city,
                t.country AS team_country,
                COUNT(*) AS match_count
            FROM matches m
            JOIN teams t ON m.home_team_id = t.id
            LEFT JOIN venue_geo vg
                ON LOWER(TRIM(m.venue_city)) = LOWER(TRIM(vg.venue_city))
                AND (
                    LOWER(TRIM(COALESCE(t.country, 'unknown'))) = LOWER(TRIM(vg.country))
                    OR vg.country = 'unknown'
                )
            WHERE m.venue_city IS NOT NULL
              AND m.venue_city != ''
              AND vg.venue_city IS NULL
              AND m.date >= NOW() - INTERVAL '{recent_days} days'
            GROUP BY m.venue_city, t.country
        )
        SELECT venue_city, team_country, match_count
        FROM recent_venues
        ORDER BY match_count DESC
        {limit_clause}
    """)

    result = await session.execute(query)
    rows = result.fetchall()

    return [
        {
            "venue_city": row.venue_city.strip() if row.venue_city else None,
            "country": row.team_country.strip() if row.team_country else "unknown",
            "match_count": row.match_count,
        }
        for row in rows
        if row.venue_city and row.venue_city.strip()
    ]


async def geocode_with_open_meteo(
    client: httpx.AsyncClient,
    city: str,
    country: Optional[str] = None,
) -> Optional[dict]:
    """
    Geocode a city using Open-Meteo Geocoding API.

    Args:
        client: httpx async client.
        city: City name.
        country: Optional country name/code for filtering.

    Returns:
        Dict with lat, lon, confidence or None if not found.
    """
    # Country name mappings (Open-Meteo uses native names)
    COUNTRY_ALIASES = {
        "turkey": ["türkiye", "turkey"],
        "saudi-arabia": ["saudi arabia", "saudi-arabia"],
        "usa": ["united states", "usa", "us"],
        "england": ["united kingdom", "england", "uk"],
        "scotland": ["united kingdom", "scotland", "uk"],
        "wales": ["united kingdom", "wales", "uk"],
        "northern-ireland": ["united kingdom", "northern ireland", "uk"],
    }

    def country_matches(result_country: str, target_country: str) -> bool:
        """Check if result country matches target, considering aliases."""
        result_lower = result_country.lower()
        target_lower = target_country.lower()

        # Direct match
        if target_lower in result_lower or result_lower in target_lower:
            return True

        # Check aliases
        aliases = COUNTRY_ALIASES.get(target_lower, [])
        return any(alias in result_lower for alias in aliases)

    try:
        # Search by city name only (Open-Meteo handles country names poorly)
        params = {
            "name": city,
            "count": 10,  # Get more results to filter by country
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

        # Filter by country if provided
        best = None
        confidence = 0.9

        if country and country != "unknown":
            # Find first result matching country
            for r in results:
                result_country = r.get("country", "")
                if country_matches(result_country, country):
                    best = r
                    break

            # If no country match, use first result with lower confidence
            if best is None:
                best = results[0]
                confidence = 0.6
                logger.debug(f"No country match for '{city}' in {country}, using first result from {best.get('country')}")
        else:
            best = results[0]

        # Lower confidence for ambiguous cities
        if len(results) > 5:
            confidence -= 0.1

        return {
            "lat": best["latitude"],
            "lon": best["longitude"],
            "confidence": max(0.5, confidence),
            "matched_name": best.get("name", city),
            "matched_country": best.get("country", country),
        }

    except httpx.HTTPError as e:
        logger.warning(f"Geocoding HTTP error for '{city}': {e}")
        return None
    except Exception as e:
        logger.error(f"Geocoding error for '{city}': {e}")
        return None


def generate_mock_geocode(city: str, country: str) -> dict:
    """
    Generate deterministic mock coordinates for testing.

    Uses hash of city+country to generate consistent lat/lon.
    """
    seed = f"{city.lower()}_{country.lower()}"
    hash_bytes = hashlib.md5(seed.encode()).digest()

    # Generate lat in [-60, 70] and lon in [-180, 180]
    lat = ((hash_bytes[0] + hash_bytes[1] * 256) / 65535) * 130 - 60
    lon = ((hash_bytes[2] + hash_bytes[3] * 256) / 65535) * 360 - 180

    return {
        "lat": round(lat, 4),
        "lon": round(lon, 4),
        "confidence": 0.95,
        "matched_name": city,
        "matched_country": country,
    }


async def upsert_venue_geo(
    session: AsyncSession,
    venue_city: str,
    country: str,
    lat: float,
    lon: float,
    source: str,
    confidence: float,
) -> str:
    """
    Upsert venue_geo entry.

    PK: (venue_city, country)
    Returns: 'inserted' or 'updated'
    """
    # Normalize for comparison
    venue_city_lower = venue_city.lower().strip()
    country_lower = country.lower().strip()

    # Check if exists
    check = await session.execute(
        text("""
            SELECT 1 FROM venue_geo
            WHERE LOWER(TRIM(venue_city)) = :venue_city
              AND LOWER(TRIM(country)) = :country
        """),
        {"venue_city": venue_city_lower, "country": country_lower}
    )
    exists = check.scalar() is not None

    if exists:
        # Update
        await session.execute(
            text("""
                UPDATE venue_geo SET
                    lat = :lat,
                    lon = :lon,
                    source = :source,
                    confidence = :confidence
                WHERE LOWER(TRIM(venue_city)) = :venue_city_lower
                  AND LOWER(TRIM(country)) = :country_lower
            """),
            {
                "venue_city_lower": venue_city_lower,
                "country_lower": country_lower,
                "lat": lat,
                "lon": lon,
                "source": source,
                "confidence": confidence,
            }
        )
        return "updated"
    else:
        # Insert (use original case for display)
        await session.execute(
            text("""
                INSERT INTO venue_geo (venue_city, country, lat, lon, source, confidence)
                VALUES (:venue_city, :country, :lat, :lon, :source, :confidence)
            """),
            {
                "venue_city": venue_city.strip(),
                "country": country.strip() if country else "unknown",
                "lat": lat,
                "lon": lon,
                "source": source,
                "confidence": confidence,
            }
        )
        return "inserted"


async def main(
    use_mock: bool = False,
    limit: Optional[int] = None,
    verbose: bool = False,
    dry_run: bool = False,
    recent_days: int = 30,
):
    """Main population logic."""
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    mode = "DRY-RUN" if dry_run else "LIVE"
    logger.info(f"Mode: {mode}, Scope: matches last {recent_days}d, Limit: {limit or 'none'}")

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
        "skipped_no_result": 0,
        "errors": 0,
    }

    # Track examples for dry-run reporting
    examples = []

    source = "mock" if use_mock else "open-meteo-geocoding"

    try:
        async with async_session() as session:
            # Get unique venues needing geocoding
            venues = await get_unique_venues(session, limit=limit, recent_days=recent_days)
            stats["scanned"] = len(venues)
            logger.info(f"Found {len(venues)} unique venues to geocode")

            if not venues:
                logger.info("No venues need geocoding")
                return stats

            # Use httpx client for real geocoding
            async with httpx.AsyncClient(timeout=30.0) as client:
                for venue in venues:
                    city = venue["venue_city"]
                    country = venue["country"]
                    match_count = venue.get("match_count", 0)

                    try:
                        # Geocode
                        if use_mock:
                            geo = generate_mock_geocode(city, country)
                        else:
                            geo = await geocode_with_open_meteo(client, city, country)
                            # Rate limit: Open-Meteo is generous but be polite
                            await asyncio.sleep(0.2)

                        if geo is None:
                            stats["skipped_no_result"] += 1
                            logger.debug(f"Venue '{city}' ({country}): No geocoding result")
                            continue

                        if dry_run:
                            # Don't write to DB, just count
                            stats["inserted"] += 1
                            if len(examples) < 10:
                                examples.append(f"  - {city} ({country}): lat={geo['lat']:.4f}, lon={geo['lon']:.4f}, matches={match_count}")
                            logger.debug(f"Venue '{city}' ({country}): would insert lat={geo['lat']:.4f} lon={geo['lon']:.4f}")
                        else:
                            # Upsert to DB
                            result = await upsert_venue_geo(
                                session=session,
                                venue_city=city,
                                country=country,
                                lat=geo["lat"],
                                lon=geo["lon"],
                                source=source,
                                confidence=geo["confidence"],
                            )
                            stats[result] += 1
                            logger.debug(
                                f"Venue '{city}' ({country}): {result} "
                                f"lat={geo['lat']:.4f} lon={geo['lon']:.4f}"
                            )

                    except Exception as e:
                        stats["errors"] += 1
                        logger.error(f"Venue '{city}' ({country}): Error - {e}")
                        continue

            if not dry_run:
                await session.commit()

    finally:
        await engine.dispose()

    # Print summary
    logger.info("=" * 60)
    logger.info(f"VENUE GEO POPULATION SUMMARY ({mode}):")
    logger.info("=" * 60)
    logger.info(f"  Scanned:             {stats['scanned']}")
    logger.info(f"  {'Would insert:' if dry_run else 'Inserted:'}        {stats['inserted']}")
    logger.info(f"  Updated:             {stats['updated']}")
    logger.info(f"  Skipped (no result): {stats['skipped_no_result']}")
    logger.info(f"  Errors:              {stats['errors']}")
    if dry_run and examples:
        logger.info("")
        logger.info("Examples:")
        for ex in examples:
            logger.info(ex)
    logger.info("=" * 60)

    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Populate venue_geo with coordinates")
    parser.add_argument(
        "--mock", action="store_true", help="Use mock coordinates for testing"
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Limit number of venues to process"
    )
    parser.add_argument(
        "--days", type=int, default=30, help="Only consider matches from last N days (default: 30)"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Preview changes without writing to DB"
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
        recent_days=args.days,
    ))
