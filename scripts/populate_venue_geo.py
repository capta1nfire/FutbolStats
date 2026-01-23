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
) -> list[dict]:
    """
    Get unique venues from matches that need geocoding.

    Joins with home team to get country if venue doesn't have one.
    Only returns venues not already in venue_geo.
    """
    limit_clause = f"LIMIT {limit}" if limit else ""

    query = text(f"""
        SELECT DISTINCT ON (LOWER(TRIM(m.venue_city)), COALESCE(LOWER(TRIM(t.country)), 'unknown'))
            m.venue_city,
            t.country AS team_country,
            COUNT(*) OVER (PARTITION BY LOWER(TRIM(m.venue_city))) AS match_count
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
        ORDER BY LOWER(TRIM(m.venue_city)), COALESCE(LOWER(TRIM(t.country)), 'unknown'), match_count DESC
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
    try:
        # Build search query
        query = city
        if country and country != "unknown":
            query = f"{city}, {country}"

        params = {
            "name": query,
            "count": 5,
            "language": "en",
            "format": "json",
        }

        response = await client.get(GEOCODING_URL, params=params)
        response.raise_for_status()

        data = response.json()
        results = data.get("results", [])

        if not results:
            logger.debug(f"No geocoding results for '{query}'")
            return None

        # Find best match (first result is usually best)
        best = results[0]

        # Calculate confidence based on match quality
        confidence = 0.9  # Default high confidence for first result

        # Lower confidence if country doesn't match
        if country and country != "unknown":
            result_country = best.get("country", "").lower()
            if country.lower() not in result_country and result_country not in country.lower():
                confidence = 0.7

        # Lower confidence for ambiguous cities
        if len(results) > 3:
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
        "skipped_no_result": 0,
        "errors": 0,
    }

    source = "mock" if use_mock else "open-meteo-geocoding"

    try:
        async with async_session() as session:
            # Get unique venues needing geocoding
            venues = await get_unique_venues(session, limit=limit)
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

            await session.commit()

    finally:
        await engine.dispose()

    # Print summary
    logger.info("=" * 60)
    logger.info("VENUE GEO POPULATION SUMMARY:")
    logger.info("=" * 60)
    logger.info(f"  Scanned:           {stats['scanned']}")
    logger.info(f"  Inserted:          {stats['inserted']}")
    logger.info(f"  Updated:           {stats['updated']}")
    logger.info(f"  Skipped (no result): {stats['skipped_no_result']}")
    logger.info(f"  Errors:            {stats['errors']}")
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
        "-v", "--verbose", action="store_true", help="Verbose logging"
    )
    args = parser.parse_args()

    asyncio.run(main(use_mock=args.mock, limit=args.limit, verbose=args.verbose))
