#!/usr/bin/env python3
"""
Backfill Historical Match Stats: Populate matches.stats for all FT matches since 2023-08-01.

This script addresses the gap caused by the stats backfill job being added late (2026-01-09)
with only 72h lookback. Result: ~23,000 matches have NULL stats, degrading XGBoost model
(shots/corners features always imputed as 0).

Features:
- Processes ALL configured leagues (from competitions.py)
- STATELESS: no local state file (Railway ephemeral filesystem safe)
- Rate limiting: 0.5s between requests (respects API limits)
- Batch commits: commits every 100 matches
- Reuses httpx client for efficiency
- Dry-run mode for testing

Usage:
    # Run backfill (default: all leagues from 2023-08-01)
    python scripts/backfill_historical_stats.py

    # Dry run
    python scripts/backfill_historical_stats.py --dry-run

    # Limit to specific leagues
    python scripts/backfill_historical_stats.py --leagues 39,140,135

    # Limit matches per run (useful for testing or daily budget)
    python scripts/backfill_historical_stats.py --limit 7500

Estimated time:
    - ~23,000 matches / 7,500 per day = ~3-4 days
    - Run once per day with --limit 7500
"""

import asyncio
import argparse
import json
import logging
import os
import sys
from datetime import datetime
from typing import Optional

from dotenv import load_dotenv
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
import httpx

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Configuration - try both env var names (ABE fix #2)
API_KEY = os.getenv("API_FOOTBALL_KEY") or os.getenv("RAPIDAPI_KEY")
DATABASE_URL = os.getenv("DATABASE_URL", "").replace("postgresql://", "postgresql+asyncpg://")

# Backfill settings
CUTOFF_DATE = "2023-08-01"  # Start of training window
DEFAULT_LIMIT = 7500        # Default daily budget
BATCH_SIZE = 100            # Commit every N matches
REQUEST_DELAY = 0.5         # Seconds between API calls

# All leagues from competitions.py
ALL_LEAGUES = [
    # HIGH priority - World Cup & Qualifiers
    1, 34, 32, 31, 30, 29, 33, 37,
    # HIGH priority - Top 5 Leagues
    39, 140, 135, 78, 61,
    # HIGH priority - UEFA & CONMEBOL
    2, 13,
    # MEDIUM - International tournaments
    9, 4, 5, 22, 6, 7,
    # MEDIUM - Secondary leagues
    40, 88, 94, 144,
    # MEDIUM - North America / Middle East
    253, 307,
    # MEDIUM - UEFA Club
    3, 848,
    # MEDIUM - LATAM Pack1
    71, 262, 128,
    # MEDIUM - LATAM Pack2
    239, 242, 250, 265, 268, 281, 299, 344,
    # MEDIUM - CONMEBOL
    11,
    # MEDIUM - Domestic Cups
    143, 45,
    # LOW - Friendlies
    10,
]


async def get_engine():
    """Create async database engine."""
    return create_async_engine(DATABASE_URL, echo=False)


async def fetch_stats_from_api(client: httpx.AsyncClient, external_id: int) -> Optional[dict]:
    """Fetch fixture statistics from API-Football.

    Args:
        client: Reusable httpx client (ABE fix #3)
        external_id: API-Football fixture ID
    """
    url = "https://v3.football.api-sports.io/fixtures/statistics"
    headers = {"x-apisports-key": API_KEY}
    params = {"fixture": external_id}

    response = await client.get(url, headers=headers, params=params, timeout=30)

    if response.status_code == 429:
        logger.warning("Rate limit hit (429)")
        raise Exception("429 Rate Limit")

    if response.status_code != 200:
        logger.warning(f"API error for fixture {external_id}: {response.status_code}")
        return None

    data = response.json()
    results = data.get("response", [])

    if not results or len(results) < 2:
        return None

    # Parse stats into {home: {...}, away: {...}} format
    stats = {"home": {}, "away": {}}

    # ABE fix #4: Use enumerate instead of results.index()
    for i, team_stats in enumerate(results):
        statistics = team_stats.get("statistics", [])
        side = "home" if i == 0 else "away"

        for stat in statistics:
            stat_type = stat.get("type", "").lower().replace(" ", "_")
            value = stat.get("value")

            # Convert percentages to floats
            if value and isinstance(value, str) and value.endswith("%"):
                try:
                    value = float(value.rstrip("%"))
                except ValueError:
                    pass

            stats[side][stat_type] = value

    return stats if stats["home"] or stats["away"] else None


async def get_matches_needing_stats(
    session: AsyncSession,
    leagues: list[int],
    cutoff_date: str,
    limit: int,
) -> list:
    """Get matches that need stats backfill.

    STATELESS design (ABE fix #1): Always selects first N matches with NULL stats.
    No need for last_processed_id - query naturally advances as stats are filled.
    """
    result = await session.execute(text("""
        SELECT id, external_id, date, league_id, home_goals, away_goals
        FROM matches
        WHERE status IN ('FT', 'AET', 'PEN')
          AND date >= :cutoff_date
          AND league_id = ANY(:leagues)
          AND external_id IS NOT NULL
          AND (stats IS NULL OR stats::text = '{}' OR stats::text = 'null')
        ORDER BY id ASC
        LIMIT :limit
    """), {
        "cutoff_date": cutoff_date,
        "leagues": leagues,
        "limit": limit,
    })
    return result.fetchall()


async def get_remaining_count(
    session: AsyncSession,
    leagues: list[int],
    cutoff_date: str,
) -> int:
    """Get count of matches still needing stats."""
    result = await session.execute(text("""
        SELECT COUNT(*) as cnt
        FROM matches
        WHERE status IN ('FT', 'AET', 'PEN')
          AND date >= :cutoff_date
          AND league_id = ANY(:leagues)
          AND external_id IS NOT NULL
          AND (stats IS NULL OR stats::text = '{}' OR stats::text = 'null')
    """), {
        "cutoff_date": cutoff_date,
        "leagues": leagues,
    })
    row = result.fetchone()
    return row.cnt if row else 0


async def update_match_stats(session: AsyncSession, match_id: int, stats: dict):
    """Update matches.stats with fetched data."""
    await session.execute(text("""
        UPDATE matches
        SET stats = CAST(:stats_json AS JSON)
        WHERE id = :match_id
    """), {
        "match_id": match_id,
        "stats_json": json.dumps(stats),
    })


async def backfill_stats(
    leagues: Optional[list[int]] = None,
    limit: Optional[int] = None,
    dry_run: bool = False,
):
    """Main backfill function (STATELESS - Railway safe)."""

    # ABE fix #2: Fail fast if no API key
    if not API_KEY:
        logger.error("FATAL: No API key found. Set API_FOOTBALL_KEY or RAPIDAPI_KEY")
        sys.exit(1)

    if not DATABASE_URL:
        logger.error("FATAL: DATABASE_URL not set")
        sys.exit(1)

    engine = await get_engine()
    AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    effective_limit = limit or DEFAULT_LIMIT
    target_leagues = leagues or ALL_LEAGUES

    logger.info(f"Starting backfill (STATELESS mode):")
    logger.info(f"  - Leagues: {len(target_leagues)}")
    logger.info(f"  - Cutoff date: {CUTOFF_DATE}")
    logger.info(f"  - Limit: {effective_limit}")
    logger.info(f"  - Dry run: {dry_run}")

    metrics = {
        "fetched": 0,
        "updated": 0,
        "skipped_no_stats": 0,
        "errors": 0,
        "api_calls": 0,
    }

    async with AsyncSessionLocal() as session:
        # Get remaining count for progress tracking
        remaining_before = await get_remaining_count(session, target_leagues, CUTOFF_DATE)
        logger.info(f"Matches needing stats: {remaining_before}")

        if remaining_before == 0:
            logger.info("All matches already have stats!")
            return {"status": "complete", "remaining": 0}

        # Get matches needing stats
        matches = await get_matches_needing_stats(
            session,
            target_leagues,
            CUTOFF_DATE,
            effective_limit,
        )

        if not matches:
            logger.info("No matches to process (query returned empty)")
            return {"status": "complete", "remaining": remaining_before}

        logger.info(f"Processing {len(matches)} matches this run")

        batch_count = 0

        # ABE fix #3: Reuse httpx client
        async with httpx.AsyncClient() as client:
            for match in matches:
                match_id = match.id
                external_id = match.external_id

                try:
                    # Fetch stats
                    stats = await fetch_stats_from_api(client, external_id)
                    metrics["api_calls"] += 1

                    if stats:
                        metrics["fetched"] += 1

                        if not dry_run:
                            await update_match_stats(session, match_id, stats)
                            metrics["updated"] += 1
                        else:
                            logger.debug(f"[DRY RUN] Would update match {match_id}")
                    else:
                        metrics["skipped_no_stats"] += 1
                        logger.debug(f"No stats available for match {match_id} (external: {external_id})")

                    # Batch commit
                    batch_count += 1
                    if batch_count >= BATCH_SIZE and not dry_run:
                        await session.commit()
                        batch_count = 0
                        logger.info(f"Progress: {metrics['api_calls']}/{len(matches)} calls, "
                                   f"{metrics['updated']} updated")

                    # Rate limiting
                    await asyncio.sleep(REQUEST_DELAY)

                except Exception as e:
                    error_str = str(e)
                    if "429" in error_str:
                        logger.warning("Rate limit hit, stopping run")
                        break
                    else:
                        metrics["errors"] += 1
                        logger.warning(f"Error for match {match_id}: {e}")

        # Final commit
        if not dry_run:
            await session.commit()

        # Get remaining count after
        remaining_after = await get_remaining_count(session, target_leagues, CUTOFF_DATE)

    await engine.dispose()

    # Summary
    logger.info("=" * 60)
    logger.info("Backfill run complete:")
    logger.info(f"  - API calls: {metrics['api_calls']}")
    logger.info(f"  - Matches updated: {metrics['updated']}")
    logger.info(f"  - Skipped (no stats from API): {metrics['skipped_no_stats']}")
    logger.info(f"  - Errors: {metrics['errors']}")
    logger.info(f"  - Remaining before: {remaining_before}")
    logger.info(f"  - Remaining after: {remaining_after}")
    logger.info(f"  - Progress: {remaining_before - remaining_after} matches filled")
    logger.info("=" * 60)

    return {
        "status": "ok",
        "metrics": metrics,
        "remaining_before": remaining_before,
        "remaining_after": remaining_after,
    }


def main():
    parser = argparse.ArgumentParser(description="Backfill historical match stats")
    parser.add_argument(
        "--leagues",
        type=str,
        help="Comma-separated league IDs (default: all configured leagues)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        help=f"Max matches to process this run (default: {DEFAULT_LIMIT})"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Don't write to DB, just log what would be done"
    )
    args = parser.parse_args()

    leagues = None
    if args.leagues:
        leagues = [int(x.strip()) for x in args.leagues.split(",")]

    asyncio.run(backfill_stats(
        leagues=leagues,
        limit=args.limit,
        dry_run=args.dry_run,
    ))


if __name__ == "__main__":
    main()
