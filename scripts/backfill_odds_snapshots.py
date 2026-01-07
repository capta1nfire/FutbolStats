#!/usr/bin/env python3
"""
Backfill odds_snapshots table from existing matches data.

For historical matches, we create snapshots from:
1. 'opening' - from opening_odds_* columns (if available)
2. 'closing' - from odds_* columns (closing line)

For the lineup_confirmed snapshot, we attempt to reconstruct the odds that existed
at (or immediately before) lineup_confirmed_at using odds_history (time-stamped).
If no odds_history exists before lineup_confirmed_at, we fall back to opening odds.

In production, we'll capture real-time odds at lineup announcement and insert
directly into odds_snapshots with snapshot_type='lineup_confirmed'.
"""

import asyncio
import logging
import os
import sys
from decimal import Decimal

from dotenv import load_dotenv
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def calculate_implied_probs(home: float, draw: float, away: float) -> tuple:
    """Calculate normalized implied probabilities from odds."""
    if not all([home, draw, away]) or any(o <= 1 for o in [home, draw, away]):
        return None, None, None, None

    raw_home = 1 / home
    raw_draw = 1 / draw
    raw_away = 1 / away

    total = raw_home + raw_draw + raw_away
    overround = total - 1

    # Normalize to sum=1
    prob_home = raw_home / total
    prob_draw = raw_draw / total
    prob_away = raw_away / total

    return prob_home, prob_draw, prob_away, overround


async def backfill_odds_snapshots():
    """Backfill odds snapshots from existing matches data."""
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        raise ValueError("DATABASE_URL required")

    if database_url.startswith("postgresql://"):
        database_url = database_url.replace("postgresql://", "postgresql+asyncpg://", 1)

    engine = create_async_engine(database_url)
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async with async_session() as session:
        # Get matches with odds that don't have snapshots yet
        result = await session.execute(text("""
            SELECT m.id, m.date,
                   m.opening_odds_home, m.opening_odds_draw, m.opening_odds_away,
                   m.odds_home, m.odds_draw, m.odds_away
            FROM matches m
            LEFT JOIN odds_snapshots os ON m.id = os.match_id
            WHERE m.status = 'FT'
              AND (m.odds_home IS NOT NULL OR m.opening_odds_home IS NOT NULL)
              AND os.id IS NULL
            ORDER BY m.date
        """))
        matches = result.fetchall()

        logger.info(f"Found {len(matches)} matches to process")

        snapshots_created = 0

        for match in matches:
            match_id = match.id
            match_date = match.date

            # Create opening snapshot if we have opening odds
            if match.opening_odds_home and match.opening_odds_draw and match.opening_odds_away:
                prob_h, prob_d, prob_a, ovr = calculate_implied_probs(
                    float(match.opening_odds_home),
                    float(match.opening_odds_draw),
                    float(match.opening_odds_away)
                )

                if prob_h is not None:
                    await session.execute(text("""
                        INSERT INTO odds_snapshots (
                            match_id, snapshot_type, snapshot_at,
                            odds_home, odds_draw, odds_away,
                            prob_home, prob_draw, prob_away,
                            overround, bookmaker
                        ) VALUES (
                            :match_id, 'opening', :snapshot_at,
                            :odds_home, :odds_draw, :odds_away,
                            :prob_home, :prob_draw, :prob_away,
                            :overround, 'avg'
                        )
                        ON CONFLICT (match_id, snapshot_type, bookmaker) DO NOTHING
                    """), {
                        "match_id": match_id,
                        "snapshot_at": match_date,  # Approximate
                        "odds_home": match.opening_odds_home,
                        "odds_draw": match.opening_odds_draw,
                        "odds_away": match.opening_odds_away,
                        "prob_home": prob_h,
                        "prob_draw": prob_d,
                        "prob_away": prob_a,
                        "overround": ovr,
                    })
                    snapshots_created += 1

            # Create closing snapshot
            if match.odds_home and match.odds_draw and match.odds_away:
                prob_h, prob_d, prob_a, ovr = calculate_implied_probs(
                    float(match.odds_home),
                    float(match.odds_draw),
                    float(match.odds_away)
                )

                if prob_h is not None:
                    await session.execute(text("""
                        INSERT INTO odds_snapshots (
                            match_id, snapshot_type, snapshot_at,
                            odds_home, odds_draw, odds_away,
                            prob_home, prob_draw, prob_away,
                            overround, bookmaker
                        ) VALUES (
                            :match_id, 'closing', :snapshot_at,
                            :odds_home, :odds_draw, :odds_away,
                            :prob_home, :prob_draw, :prob_away,
                            :overround, 'avg'
                        )
                        ON CONFLICT (match_id, snapshot_type, bookmaker) DO NOTHING
                    """), {
                        "match_id": match_id,
                        "snapshot_at": match_date,
                        "odds_home": match.odds_home,
                        "odds_draw": match.odds_draw,
                        "odds_away": match.odds_away,
                        "prob_home": prob_h,
                        "prob_draw": prob_d,
                        "prob_away": prob_a,
                        "overround": ovr,
                    })
                    snapshots_created += 1

            # Create lineup_confirmed snapshot (best-effort reconstruction)
            # 1) Find lineup_confirmed_at (we store it in match_lineups; take MIN across teams)
            lineup_time_res = await session.execute(text("""
                SELECT MIN(lineup_confirmed_at) AS lineup_confirmed_at
                FROM match_lineups
                WHERE match_id = :match_id
                  AND lineup_confirmed_at IS NOT NULL
            """), {"match_id": match_id})
            lineup_confirmed_at = lineup_time_res.scalar_one_or_none()

            if lineup_confirmed_at is not None:
                # 2) Find the latest odds_history snapshot at or before lineup_confirmed_at
                odds_hist_res = await session.execute(text("""
                    SELECT odds_home, odds_draw, odds_away, recorded_at, source
                    FROM odds_history
                    WHERE match_id = :match_id
                      AND recorded_at <= :cutoff
                      AND odds_home IS NOT NULL AND odds_draw IS NOT NULL AND odds_away IS NOT NULL
                    ORDER BY recorded_at DESC
                    LIMIT 1
                """), {"match_id": match_id, "cutoff": lineup_confirmed_at})
                oh = odds_hist_res.fetchone()

                if oh:
                    lh, ld, la = float(oh.odds_home), float(oh.odds_draw), float(oh.odds_away)
                    prob_h, prob_d, prob_a, ovr = calculate_implied_probs(lh, ld, la)
                    snapshot_at = oh.recorded_at
                    bookmaker = oh.source or "api_football"
                else:
                    # Fallback to opening odds as proxy, but timestamp at lineup_confirmed_at
                    if match.opening_odds_home and match.opening_odds_draw and match.opening_odds_away:
                        lh, ld, la = float(match.opening_odds_home), float(match.opening_odds_draw), float(match.opening_odds_away)
                        prob_h, prob_d, prob_a, ovr = calculate_implied_probs(lh, ld, la)
                        snapshot_at = lineup_confirmed_at
                        bookmaker = "avg"
                    else:
                        prob_h = prob_d = prob_a = ovr = None
                        snapshot_at = lineup_confirmed_at
                        bookmaker = "avg"

                if prob_h is not None:
                    await session.execute(text("""
                        INSERT INTO odds_snapshots (
                            match_id, snapshot_type, snapshot_at,
                            odds_home, odds_draw, odds_away,
                            prob_home, prob_draw, prob_away,
                            overround, bookmaker
                        ) VALUES (
                            :match_id, 'lineup_confirmed', :snapshot_at,
                            :odds_home, :odds_draw, :odds_away,
                            :prob_home, :prob_draw, :prob_away,
                            :overround, :bookmaker
                        )
                        ON CONFLICT (match_id, snapshot_type, bookmaker) DO NOTHING
                    """), {
                        "match_id": match_id,
                        "snapshot_at": snapshot_at,
                        "odds_home": lh if prob_h is not None else None,
                        "odds_draw": ld if prob_h is not None else None,
                        "odds_away": la if prob_h is not None else None,
                        "prob_home": prob_h,
                        "prob_draw": prob_d,
                        "prob_away": prob_a,
                        "overround": ovr,
                        "bookmaker": bookmaker,
                    })
                    snapshots_created += 1

            # Commit every 500 matches
            if snapshots_created % 500 == 0 and snapshots_created > 0:
                await session.commit()
                logger.info(f"Progress: {snapshots_created} snapshots created")

        await session.commit()

    await engine.dispose()

    logger.info("=" * 60)
    logger.info("BACKFILL COMPLETE")
    logger.info(f"Snapshots created: {snapshots_created}")
    logger.info("=" * 60)


if __name__ == "__main__":
    asyncio.run(backfill_odds_snapshots())
