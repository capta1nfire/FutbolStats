#!/usr/bin/env python3
"""
Backfill Venues: Populate venue_name and venue_city for matches in DB.

Fetches fixture data from API-Football to extract venue information.
This allows showing stadium info in match details without runtime API calls.

Features:
- Rate limiting (0.5s between requests)
- Backoff on failures (exponential)
- Skip matches that already have venue data
- Configurable date range and limit

Usage:
    # Backfill venues for last 30 days (default)
    python scripts/backfill_venues.py

    # Backfill venues for last 90 days
    python scripts/backfill_venues.py --days 90

    # Dry run (no writes)
    python scripts/backfill_venues.py --dry-run

    # Limit number of matches
    python scripts/backfill_venues.py --limit 100
"""

import asyncio
import argparse
import logging
import os
from datetime import datetime, timedelta
from typing import Optional

from dotenv import load_dotenv
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
import httpx

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
RAPIDAPI_KEY = os.getenv("RAPIDAPI_KEY")
RAPIDAPI_HOST = os.getenv("RAPIDAPI_HOST", "v3.football.api-sports.io")
DATABASE_URL = os.getenv("DATABASE_URL", "").replace("postgresql://", "postgresql+asyncpg://")


async def get_engine():
    """Create async database engine."""
    return create_async_engine(DATABASE_URL, echo=False)


async def fetch_venue_from_api(fixture_id: int) -> Optional[dict]:
    """Fetch venue info from API-Football fixture endpoint."""
    if not RAPIDAPI_KEY:
        logger.error("RAPIDAPI_KEY not set")
        return None

    # Determine API host and headers
    if "api-sports.io" in RAPIDAPI_HOST:
        url = f"https://{RAPIDAPI_HOST}/fixtures"
        headers = {"x-apisports-key": RAPIDAPI_KEY}
    else:
        url = f"https://{RAPIDAPI_HOST}/v3/fixtures"
        headers = {
            "X-RapidAPI-Key": RAPIDAPI_KEY,
            "X-RapidAPI-Host": RAPIDAPI_HOST,
        }
    params = {"id": fixture_id}

    async with httpx.AsyncClient() as client:
        response = await client.get(url, headers=headers, params=params, timeout=30)
        if response.status_code != 200:
            logger.warning(f"API error for fixture {fixture_id}: {response.status_code}")
            return None

        data = response.json()
        fixtures = data.get("response", [])
        if not fixtures:
            return None

        fixture = fixtures[0]
        venue = fixture.get("fixture", {}).get("venue", {})

        if not venue:
            return None

        return {
            "name": venue.get("name"),
            "city": venue.get("city"),
        }


async def get_matches_without_venue(
    session: AsyncSession,
    days: int,
    limit: int,
) -> list:
    """Get matches that need venue backfill."""
    cutoff = datetime.now() - timedelta(days=days)

    result = await session.execute(
        text("""
            SELECT id, external_id
            FROM matches
            WHERE date >= :cutoff
              AND venue_name IS NULL
            ORDER BY date DESC
            LIMIT :limit
        """),
        {"cutoff": cutoff, "limit": limit}
    )
    return result.fetchall()


async def update_venue(
    session: AsyncSession,
    match_id: int,
    venue_name: Optional[str],
    venue_city: Optional[str],
) -> None:
    """Update venue fields for a match."""
    await session.execute(
        text("""
            UPDATE matches
            SET venue_name = :venue_name, venue_city = :venue_city
            WHERE id = :match_id
        """),
        {"match_id": match_id, "venue_name": venue_name, "venue_city": venue_city}
    )
    await session.commit()


async def backfill_venues(
    days: int = 30,
    limit: int = 500,
    dry_run: bool = False,
):
    """Main backfill function."""
    engine = await get_engine()
    AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async with AsyncSessionLocal() as session:
        # Get matches needing venue data
        matches = await get_matches_without_venue(session, days, limit)

        if not matches:
            logger.info("No matches need venue backfill")
            return

        logger.info(f"Found {len(matches)} matches needing venue data")

        stats = {
            "processed": 0,
            "updated": 0,
            "no_venue": 0,
            "failed": 0,
        }
        consecutive_failures = 0
        max_failures = 5

        for match_id, external_id in matches:
            # Abort on too many failures
            if consecutive_failures >= max_failures:
                logger.error(f"Aborting after {consecutive_failures} consecutive failures")
                break

            try:
                venue = await fetch_venue_from_api(external_id)

                if venue and venue.get("name"):
                    if not dry_run:
                        await update_venue(session, match_id, venue["name"], venue.get("city"))
                    logger.info(f"Match {match_id}: {'would update' if dry_run else 'updated'} venue: {venue['name']}, {venue.get('city')}")
                    stats["updated"] += 1
                    consecutive_failures = 0
                else:
                    logger.info(f"Match {match_id}: no venue data available")
                    stats["no_venue"] += 1
                    consecutive_failures = 0

                # Rate limiting
                await asyncio.sleep(0.5)

            except Exception as e:
                logger.error(f"Match {match_id}: error - {e}")
                stats["failed"] += 1
                consecutive_failures += 1
                # Backoff
                await asyncio.sleep(min(2 ** consecutive_failures, 8))

            stats["processed"] += 1

    await engine.dispose()

    logger.info(f"Backfill complete: {stats}")
    return stats


def main():
    parser = argparse.ArgumentParser(description="Backfill venue data for matches")
    parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="Number of days back to look for matches (default: 30)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=500,
        help="Maximum number of matches to process (default: 500)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Don't write to DB, just log what would be done"
    )
    args = parser.parse_args()

    asyncio.run(backfill_venues(
        days=args.days,
        limit=args.limit,
        dry_run=args.dry_run,
    ))


if __name__ == "__main__":
    main()
