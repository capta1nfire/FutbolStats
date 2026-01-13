#!/usr/bin/env python3
"""
Backfill Events: Populate match.events for DB-first timeline architecture.

Fetches events from API-Football for finished matches that don't have events stored.
This ensures /matches/{id}/timeline serves from DB instead of external API.

Features:
- Rate limiting (0.5s between requests)
- Backoff on failures (exponential)
- Skip matches that already have events
- Configurable date range

Usage:
    # Backfill events for last 7 days (default)
    python scripts/backfill_events.py

    # Backfill events for last 30 days
    python scripts/backfill_events.py --days 30

    # Dry run (no writes)
    python scripts/backfill_events.py --dry-run

    # Limit number of matches
    python scripts/backfill_events.py --limit 50
"""

import asyncio
import argparse
import json
import logging
import os
import sys
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
API_FOOTBALL_KEY = os.getenv("API_FOOTBALL_KEY")
DATABASE_URL = os.getenv("DATABASE_URL", "").replace("postgresql://", "postgresql+asyncpg://")


async def get_engine():
    """Create async database engine."""
    return create_async_engine(DATABASE_URL, echo=False)


async def fetch_events_from_api(fixture_id: int) -> Optional[list]:
    """Fetch events from API-Football."""
    if not API_FOOTBALL_KEY:
        logger.error("API_FOOTBALL_KEY not set")
        return None

    url = f"https://v3.football.api-sports.io/fixtures/events"
    headers = {"x-apisports-key": API_FOOTBALL_KEY}
    params = {"fixture": fixture_id}

    async with httpx.AsyncClient() as client:
        response = await client.get(url, headers=headers, params=params, timeout=30)
        if response.status_code != 200:
            logger.warning(f"API error for fixture {fixture_id}: {response.status_code}")
            return None

        data = response.json()
        events = data.get("response", [])
        if not events:
            return []  # Empty is valid (goalless draw)

        # Normalize event structure
        normalized = []
        for event in events:
            normalized.append({
                "time": {
                    "elapsed": event.get("time", {}).get("elapsed"),
                    "extra": event.get("time", {}).get("extra"),
                },
                "team": {
                    "id": event.get("team", {}).get("id"),
                    "name": event.get("team", {}).get("name"),
                    "logo": event.get("team", {}).get("logo"),
                },
                "player": {
                    "id": event.get("player", {}).get("id"),
                    "name": event.get("player", {}).get("name"),
                },
                "assist": {
                    "id": event.get("assist", {}).get("id"),
                    "name": event.get("assist", {}).get("name"),
                },
                "type": event.get("type"),
                "detail": event.get("detail"),
                "comments": event.get("comments"),
                # Legacy format compatibility
                "minute": event.get("time", {}).get("elapsed"),
                "extra_minute": event.get("time", {}).get("extra"),
            })

        return normalized


async def get_matches_without_events(
    session: AsyncSession,
    days_back: int = 7,
    limit: int = 100,
) -> list:
    """Get finished matches that don't have events stored."""
    cutoff = datetime.now() - timedelta(days=days_back)

    result = await session.execute(
        text("""
            SELECT id, external_id, date, home_goals, away_goals
            FROM matches
            WHERE status IN ('FT', 'AET', 'PEN')
              AND date >= :cutoff
              AND (events IS NULL OR events::text = '[]' OR events::text = 'null')
            ORDER BY date DESC
            LIMIT :limit
        """),
        {"cutoff": cutoff, "limit": limit}
    )
    return [
        {"id": row[0], "external_id": row[1], "date": row[2], "home_goals": row[3], "away_goals": row[4]}
        for row in result.fetchall()
    ]


async def save_events(session: AsyncSession, match_id: int, events: list) -> None:
    """Persist events to match record."""
    await session.execute(
        text("""
            UPDATE matches
            SET events = :events
            WHERE id = :match_id
        """),
        {"match_id": match_id, "events": json.dumps(events)}
    )
    await session.commit()


async def backfill_events(
    days_back: int = 7,
    limit: int = 100,
    dry_run: bool = False,
):
    """Main backfill function."""
    engine = await get_engine()
    AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async with AsyncSessionLocal() as session:
        matches = await get_matches_without_events(session, days_back, limit)

        if not matches:
            logger.info("No matches need events backfill")
            return

        logger.info(f"Processing {len(matches)} matches from last {days_back} days")

        stats = {
            "processed": 0,
            "fetched": 0,
            "empty": 0,  # Valid 0-0 matches
            "failed": 0,
        }
        consecutive_failures = 0
        max_failures = 5

        for match in matches:
            # Abort on too many failures
            if consecutive_failures >= max_failures:
                logger.error(f"Aborting after {consecutive_failures} consecutive failures")
                break

            try:
                events = await fetch_events_from_api(match["external_id"])
                if events is not None:
                    if not dry_run:
                        await save_events(session, match["id"], events)

                    if len(events) == 0:
                        stats["empty"] += 1
                        logger.info(f"Match {match['id']} ({match['external_id']}): {'would save' if dry_run else 'saved'} 0 events (0-0?)")
                    else:
                        stats["fetched"] += 1
                        logger.info(f"Match {match['id']} ({match['external_id']}): {'would save' if dry_run else 'saved'} {len(events)} events")

                    consecutive_failures = 0
                else:
                    logger.warning(f"Match {match['id']} ({match['external_id']}): API returned None")
                    stats["failed"] += 1
                    consecutive_failures += 1

                # Rate limiting
                await asyncio.sleep(0.5)

            except Exception as e:
                logger.error(f"Match {match['id']} ({match['external_id']}): error - {e}")
                stats["failed"] += 1
                consecutive_failures += 1
                # Backoff
                await asyncio.sleep(min(2 ** consecutive_failures, 8))

            stats["processed"] += 1

    await engine.dispose()

    logger.info(f"Backfill complete: {stats}")
    return stats


def main():
    parser = argparse.ArgumentParser(description="Backfill match events to DB")
    parser.add_argument(
        "--days",
        type=int,
        default=7,
        help="Days to look back for matches (default: 7)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Max matches to process (default: 100)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Don't write to DB, just log what would be done"
    )
    args = parser.parse_args()

    asyncio.run(backfill_events(
        days_back=args.days,
        limit=args.limit,
        dry_run=args.dry_run,
    ))


if __name__ == "__main__":
    main()
