#!/usr/bin/env python3
"""
Backfill Standings: Populate league_standings table for DB-first architecture.

Fetches standings from API-Football for active leagues and persists to DB.
This ensures /matches/{id}/details serves standings from DB instead of external API.

League selection logic (when no --leagues specified):
1. Core leagues (Top 5, MX, BR, MLS, etc.) - always included
2. Leagues with matches in DB (historical coverage)
3. Leagues with fixtures in next 7 days (lookahead)
4. Skip leagues that return "no data" (cups/tournaments without tables)

Features:
- Rate limiting (0.5s between requests)
- Backoff on failures (exponential)
- Skip already-fresh data (< 6h old)
- Mark "no data" leagues as skipped (avoid retries)
- Dry-run mode for testing

Usage:
    # Backfill with smart league selection (default)
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
STANDINGS_TTL_HOURS = 6  # Consider standings data stale after this
NO_TABLE_TTL_DAYS = 30  # Don't retry "no table" leagues for this long
LOOKAHEAD_DAYS = 7  # Check for fixtures this many days ahead

# Core leagues: always backfill standings (leagues with regular tables)
CORE_LEAGUES = [
    # Top 5 European leagues
    39,   # England - Premier League
    140,  # Spain - La Liga
    135,  # Italy - Serie A
    78,   # Germany - Bundesliga
    61,   # France - Ligue 1
    # Other major leagues
    94,   # Portugal - Primeira Liga
    88,   # Netherlands - Eredivisie
    203,  # Turkey - Super Lig
    71,   # Brazil - Serie A
    262,  # Mexico - Liga MX
    128,  # Argentina - Primera Division
    253,  # USA - MLS
    # LATAM
    239,  # Colombia - Primera A
    242,  # Ecuador - Liga Pro
    265,  # Chile - Primera Division
    281,  # Peru - Primera Division
]

# Leagues that don't have standings tables (cups, tournaments, qualifiers)
# These return empty from API and should be skipped
NO_TABLE_LEAGUES = [
    # UEFA club competitions
    2,    # Champions League (groups phase has tables, knockout doesn't)
    3,    # Europa League
    848,  # Conference League
    # Domestic cups
    45,   # FA Cup
    143,  # Copa del Rey
    # CONMEBOL club competitions
    13,   # CONMEBOL Libertadores
    11,   # CONMEBOL Sudamericana
    # World Cup Qualifiers
    29, 30, 31, 32, 33, 34, 37,
    # International tournaments / friendlies (no league table)
    1,    # World Cup
    4,    # Euro Championship
    5,    # UEFA Nations League
    7,    # African Cup of Nations
    9,    # Copa America
    10,   # Friendlies
    28,   # Club Friendlies
]


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


async def mark_no_table(session: AsyncSession, league_id: int, season: int) -> None:
    """Mark a league as having no standings table (persisted for NO_TABLE_TTL_DAYS)."""
    expires_at = datetime.now() + timedelta(days=NO_TABLE_TTL_DAYS)
    await session.execute(
        text("""
            INSERT INTO league_standings (league_id, season, standings, captured_at, expires_at, source)
            VALUES (:league_id, :season, NULL, NOW(), :expires_at, 'no_table')
            ON CONFLICT (league_id, season)
            DO UPDATE SET standings = NULL, captured_at = NOW(), expires_at = :expires_at, source = 'no_table'
        """),
        {"league_id": league_id, "season": season, "expires_at": expires_at}
    )
    await session.commit()


async def is_marked_no_table(session: AsyncSession, league_id: int, season: int) -> bool:
    """Check if a league is marked as 'no_table' and still within TTL."""
    result = await session.execute(
        text("""
            SELECT source, expires_at
            FROM league_standings
            WHERE league_id = :league_id AND season = :season
        """),
        {"league_id": league_id, "season": season}
    )
    row = result.fetchone()
    if not row:
        return False

    source, expires_at = row
    if source != 'no_table':
        return False

    # Check if still within TTL
    return expires_at and datetime.now() < expires_at


async def get_db_no_table_leagues(session: AsyncSession, season: int) -> set:
    """Get league_ids marked as 'no_table' in DB (within TTL)."""
    result = await session.execute(
        text("""
            SELECT league_id
            FROM league_standings
            WHERE season = :season
              AND source = 'no_table'
              AND expires_at > NOW()
        """),
        {"season": season}
    )
    return {row[0] for row in result.fetchall()}


async def get_leagues_with_matches(session: AsyncSession) -> set:
    """Get all unique league_ids that have any matches in DB."""
    result = await session.execute(
        text("""
            SELECT DISTINCT league_id
            FROM matches
            WHERE league_id IS NOT NULL
        """)
    )
    return {row[0] for row in result.fetchall()}


async def get_leagues_with_upcoming_fixtures(session: AsyncSession) -> set:
    """Get league_ids with fixtures in the next LOOKAHEAD_DAYS."""
    now = datetime.now()
    future = now + timedelta(days=LOOKAHEAD_DAYS)

    result = await session.execute(
        text("""
            SELECT DISTINCT league_id
            FROM matches
            WHERE league_id IS NOT NULL
              AND date >= :start_date
              AND date <= :end_date
              AND status = 'NS'
        """),
        {"start_date": now.date(), "end_date": future.date()}
    )
    return {row[0] for row in result.fetchall()}


async def get_target_leagues(session: AsyncSession, season: int) -> list:
    """
    Get leagues to backfill using smart selection:
    1. Core leagues (always)
    2. Leagues with matches in DB
    3. Leagues with upcoming fixtures
    4. Exclude NO_TABLE_LEAGUES (hardcoded known)
    5. Exclude DB-persisted 'no_table' marks (auto-discovered, TTL 30 days)
    """
    # Gather all candidate leagues
    db_leagues = await get_leagues_with_matches(session)
    upcoming_leagues = await get_leagues_with_upcoming_fixtures(session)

    # Union: core + DB + upcoming
    all_candidates = set(CORE_LEAGUES) | db_leagues | upcoming_leagues

    # Get DB-persisted no_table marks
    db_no_table = await get_db_no_table_leagues(session, season)

    # Exclude leagues without tables (hardcoded + DB-persisted)
    all_no_table = set(NO_TABLE_LEAGUES) | db_no_table
    target_leagues = all_candidates - all_no_table

    logger.info(f"League selection: {len(CORE_LEAGUES)} core, {len(db_leagues)} in DB, "
                f"{len(upcoming_leagues)} upcoming")
    logger.info(f"Excluded: {len(NO_TABLE_LEAGUES)} hardcoded + {len(db_no_table)} DB-marked = {len(all_no_table)} no_table")
    logger.info(f"Target leagues: {len(target_leagues)} total")

    return sorted(target_leagues)


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
            logger.info(f"Using specified leagues: {league_ids}")
        else:
            league_ids = await get_target_leagues(session, season)

        if not league_ids:
            logger.info("No leagues to process")
            return

        logger.info(f"Processing {len(league_ids)} leagues for season {season}")

        stats = {
            "processed": 0,
            "fetched": 0,
            "skipped_fresh": 0,
            "skipped_no_table": 0,
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
                    # No data returned - mark as no_table in DB (TTL 30 days)
                    if not dry_run:
                        await mark_no_table(session, league_id, season)
                    logger.info(f"League {league_id}: {'would mark' if dry_run else 'marked'} no_table (TTL {NO_TABLE_TTL_DAYS}d)")
                    stats["skipped_no_table"] += 1
                    # Don't count as failure - this is expected for some leagues
                    consecutive_failures = 0

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
