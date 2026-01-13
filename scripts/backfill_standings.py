#!/usr/bin/env python3
"""
Backfill Standings: Populate league_standings table for DB-first architecture.

Fetches standings from API-Football for active leagues and persists to DB.
This ensures /matches/{id}/details serves standings from DB instead of external API.

Features:
- Rate limiting (0.5s between requests)
- Backoff on failures (exponential)
- Skip already-fresh data (< 6h old)
- Dry-run mode for testing

Usage:
    # Backfill all leagues with upcoming matches (default)
    python scripts/backfill_standings.py

    # Backfill specific leagues
    python scripts/backfill_standings.py --leagues 39,140,78

    # Dry run (no writes)
    python scripts/backfill_standings.py --dry-run

    # Force refresh even if data is fresh
    python scripts/backfill_standings.py --force
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
STANDINGS_TTL_HOURS = 6  # Consider data stale after this


async def get_engine():
    """Create async database engine."""
    return create_async_engine(DATABASE_URL, echo=False)


async def fetch_standings_from_api(league_id: int, season: int) -> Optional[list]:
    """Fetch standings from API-Football."""
    if not API_FOOTBALL_KEY:
        logger.error("API_FOOTBALL_KEY not set")
        return None

    url = f"https://v3.football.api-sports.io/standings"
    headers = {"x-apisports-key": API_FOOTBALL_KEY}
    params = {"league": league_id, "season": season}

    async with httpx.AsyncClient() as client:
        response = await client.get(url, headers=headers, params=params, timeout=30)
        if response.status_code != 200:
            logger.warning(f"API error for league {league_id}: {response.status_code}")
            return None

        data = response.json()
        results = data.get("response", [])
        if not results:
            return None

        # Extract standings from nested structure
        league_data = results[0].get("league", {})
        standings_groups = league_data.get("standings", [])
        if not standings_groups:
            return None

        # Flatten standings (handle groups like Champions League)
        standings = []
        for group in standings_groups:
            for team_standing in group:
                standings.append({
                    "team_id": team_standing.get("team", {}).get("id"),
                    "team_name": team_standing.get("team", {}).get("name"),
                    "team_logo": team_standing.get("team", {}).get("logo"),
                    "position": team_standing.get("rank"),
                    "points": team_standing.get("points"),
                    "played": team_standing.get("all", {}).get("played"),
                    "win": team_standing.get("all", {}).get("win"),
                    "draw": team_standing.get("all", {}).get("draw"),
                    "lose": team_standing.get("all", {}).get("lose"),
                    "goals_for": team_standing.get("all", {}).get("goals", {}).get("for"),
                    "goals_against": team_standing.get("all", {}).get("goals", {}).get("against"),
                    "goal_diff": team_standing.get("goalsDiff"),
                    "form": team_standing.get("form"),
                    "group": team_standing.get("group"),
                })

        return standings


async def is_standings_fresh(session: AsyncSession, league_id: int, season: int) -> bool:
    """Check if standings in DB are fresh (< TTL hours old)."""
    result = await session.execute(
        text("""
            SELECT captured_at
            FROM league_standings
            WHERE league_id = :league_id AND season = :season
        """),
        {"league_id": league_id, "season": season}
    )
    row = result.fetchone()
    if not row:
        return False

    captured_at = row[0]
    age_hours = (datetime.now() - captured_at).total_seconds() / 3600
    return age_hours < STANDINGS_TTL_HOURS


async def save_standings(session: AsyncSession, league_id: int, season: int, standings: list) -> None:
    """Persist standings to DB with upsert."""
    expires_at = datetime.now() + timedelta(hours=STANDINGS_TTL_HOURS)
    await session.execute(
        text("""
            INSERT INTO league_standings (league_id, season, standings, captured_at, expires_at, source)
            VALUES (:league_id, :season, :standings, NOW(), :expires_at, 'backfill')
            ON CONFLICT (league_id, season)
            DO UPDATE SET standings = :standings, captured_at = NOW(), expires_at = :expires_at, source = 'backfill'
        """),
        {"league_id": league_id, "season": season, "standings": json.dumps(standings), "expires_at": expires_at}
    )
    await session.commit()


async def get_active_leagues(session: AsyncSession, days_ahead: int = 14) -> list:
    """Get unique league_ids from upcoming matches."""
    now = datetime.now()
    future = now + timedelta(days=days_ahead)

    result = await session.execute(
        text("""
            SELECT DISTINCT league_id
            FROM matches
            WHERE league_id IS NOT NULL
              AND date >= :start_date
              AND date <= :end_date
              AND status = 'NS'
            ORDER BY league_id
        """),
        {"start_date": now.date(), "end_date": future.date()}
    )
    return [row[0] for row in result.fetchall()]


async def backfill_standings(
    leagues: Optional[list] = None,
    dry_run: bool = False,
    force: bool = False,
):
    """Main backfill function."""
    engine = await get_engine()
    AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    # Determine season
    now = datetime.now()
    season = now.year if now.month >= 7 else now.year - 1

    async with AsyncSessionLocal() as session:
        # Get leagues to process
        if leagues:
            league_ids = leagues
        else:
            league_ids = await get_active_leagues(session)

        if not league_ids:
            logger.info("No leagues to process")
            return

        logger.info(f"Processing {len(league_ids)} leagues for season {season}")

        stats = {
            "processed": 0,
            "fetched": 0,
            "skipped_fresh": 0,
            "failed": 0,
        }
        consecutive_failures = 0
        max_failures = 5

        for league_id in league_ids:
            # Check if data is fresh
            if not force and await is_standings_fresh(session, league_id, season):
                logger.info(f"League {league_id}: skipped (fresh)")
                stats["skipped_fresh"] += 1
                continue

            # Abort on too many failures
            if consecutive_failures >= max_failures:
                logger.error(f"Aborting after {consecutive_failures} consecutive failures")
                break

            try:
                standings = await fetch_standings_from_api(league_id, season)
                if standings:
                    if not dry_run:
                        await save_standings(session, league_id, season, standings)
                    logger.info(f"League {league_id}: {'would save' if dry_run else 'saved'} {len(standings)} teams")
                    stats["fetched"] += 1
                    consecutive_failures = 0
                else:
                    logger.warning(f"League {league_id}: no standings returned")
                    stats["failed"] += 1
                    consecutive_failures += 1

                # Rate limiting
                await asyncio.sleep(0.5)

            except Exception as e:
                logger.error(f"League {league_id}: error - {e}")
                stats["failed"] += 1
                consecutive_failures += 1
                # Backoff
                await asyncio.sleep(min(2 ** consecutive_failures, 8))

            stats["processed"] += 1

    await engine.dispose()

    logger.info(f"Backfill complete: {stats}")
    return stats


def main():
    parser = argparse.ArgumentParser(description="Backfill league standings to DB")
    parser.add_argument(
        "--leagues",
        type=str,
        help="Comma-separated league IDs (default: auto-detect from upcoming matches)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Don't write to DB, just log what would be done"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force refresh even if data is fresh"
    )
    args = parser.parse_args()

    leagues = None
    if args.leagues:
        leagues = [int(x.strip()) for x in args.leagues.split(",")]

    asyncio.run(backfill_standings(
        leagues=leagues,
        dry_run=args.dry_run,
        force=args.force,
    ))


if __name__ == "__main__":
    main()
