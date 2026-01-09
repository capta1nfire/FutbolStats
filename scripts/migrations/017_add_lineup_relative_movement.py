#!/usr/bin/env python3
"""
Migration 017: Add lineup-relative market movement tracking

This addresses the auditor's critical feedback:
1. Track movement RELATIVE to lineup_detected_at (not just pre-kickoff)
2. Store movement metrics on normalized probabilities

New table: lineup_movement_snapshots
- Captures odds at T-30, T-15, T-5, T+0 (lineup), T+5, T+10 relative to lineup detection
- Stores delta_p (max movement on normalized probabilities)
- Links to lineup_detected_at from odds_snapshots

This allows measuring "Did the market move BECAUSE of lineup announcement?"
vs the current "How did odds drift pre-kickoff?"
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

DATABASE_URL = os.environ.get("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("DATABASE_URL required")

if DATABASE_URL.startswith("postgresql://"):
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://")


async def migrate():
    engine = create_async_engine(DATABASE_URL)

    async with engine.begin() as conn:
        # Create lineup_movement_snapshots table
        # This tracks odds RELATIVE to lineup detection time
        await conn.execute(text("""
            CREATE TABLE IF NOT EXISTS lineup_movement_snapshots (
                id SERIAL PRIMARY KEY,
                match_id INTEGER NOT NULL REFERENCES matches(id),

                -- Reference point: when lineup was detected
                lineup_detected_at TIMESTAMP NOT NULL,

                -- Snapshot timing relative to lineup detection
                -- Negative = before lineup, Positive = after lineup
                snapshot_type VARCHAR(20) NOT NULL,  -- 'L-30', 'L-15', 'L-5', 'L0', 'L+5', 'L+10'
                minutes_from_lineup NUMERIC(6,2),     -- Actual offset in minutes

                -- Absolute timestamps
                captured_at TIMESTAMP NOT NULL,
                kickoff_time TIMESTAMP,

                -- Odds data
                odds_home NUMERIC(6,3),
                odds_draw NUMERIC(6,3),
                odds_away NUMERIC(6,3),
                bookmaker VARCHAR(50),
                odds_freshness VARCHAR(20),  -- 'live', 'cached'

                -- Normalized probabilities (after removing overround)
                prob_home NUMERIC(6,4),
                prob_draw NUMERIC(6,4),
                prob_away NUMERIC(6,4),
                overround NUMERIC(6,4),

                -- Movement metrics (computed vs baseline L-30 or L0)
                -- delta_p = max(|prob_h(t) - prob_h(baseline)|, |prob_d(t) - prob_d(baseline)|, |prob_a(t) - prob_a(baseline)|)
                delta_p_vs_baseline NUMERIC(6,4),
                baseline_snapshot_type VARCHAR(20),  -- Which snapshot we compared against

                created_at TIMESTAMP DEFAULT NOW(),

                UNIQUE(match_id, snapshot_type, bookmaker)
            )
        """))
        print("Created table: lineup_movement_snapshots")

        # Create indices
        await conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_lms_match_type
            ON lineup_movement_snapshots(match_id, snapshot_type)
        """))
        print("Created index: idx_lms_match_type")

        await conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_lms_lineup_detected
            ON lineup_movement_snapshots(lineup_detected_at)
        """))
        print("Created index: idx_lms_lineup_detected")

        # Add column to matches for tracking lineup movement completion
        await conn.execute(text("""
            ALTER TABLE matches
            ADD COLUMN IF NOT EXISTS lineup_movement_tracked BOOLEAN DEFAULT FALSE
        """))
        print("Added column: matches.lineup_movement_tracked")

        # Create view for easy movement analysis
        # NOTE: CASE expressions ensure delta_p is NULL when we don't have both snapshots
        await conn.execute(text("""
            CREATE OR REPLACE VIEW v_lineup_movement_analysis AS
            SELECT
                m.id as match_id,
                m.external_id,
                m.date as kickoff_time,
                m.league_id,
                ht.name as home_team,
                at.name as away_team,

                -- Baseline (L0 = lineup detection moment)
                l0.prob_home as prob_home_l0,
                l0.prob_draw as prob_draw_l0,
                l0.prob_away as prob_away_l0,
                l0.captured_at as lineup_detected_at,

                -- Pre-lineup (L-30)
                l_30.prob_home as prob_home_l_30,
                l_30.prob_draw as prob_draw_l_30,
                l_30.prob_away as prob_away_l_30,

                -- Post-lineup (L+10)
                l_10.prob_home as prob_home_l_10,
                l_10.prob_draw as prob_draw_l_10,
                l_10.prob_away as prob_away_l_10,

                -- Movement metrics (only compute when we have both snapshots)
                CASE WHEN l_10.prob_home IS NOT NULL AND l0.prob_home IS NOT NULL THEN
                    GREATEST(
                        ABS(l_10.prob_home - l0.prob_home),
                        ABS(l_10.prob_draw - l0.prob_draw),
                        ABS(l_10.prob_away - l0.prob_away)
                    )
                END as delta_p_post_lineup,

                CASE WHEN l_30.prob_home IS NOT NULL AND l0.prob_home IS NOT NULL THEN
                    GREATEST(
                        ABS(l0.prob_home - l_30.prob_home),
                        ABS(l0.prob_draw - l_30.prob_draw),
                        ABS(l0.prob_away - l_30.prob_away)
                    )
                END as delta_p_pre_lineup

            FROM matches m
            JOIN teams ht ON m.home_team_id = ht.id
            JOIN teams at ON m.away_team_id = at.id
            LEFT JOIN lineup_movement_snapshots l0
                ON m.id = l0.match_id AND l0.snapshot_type = 'L0'
            LEFT JOIN lineup_movement_snapshots l_30
                ON m.id = l_30.match_id AND l_30.snapshot_type = 'L-30'
            LEFT JOIN lineup_movement_snapshots l_10
                ON m.id = l_10.match_id AND l_10.snapshot_type = 'L+10'
            WHERE m.lineup_confirmed = TRUE
        """))
        print("Created view: v_lineup_movement_analysis")

    await engine.dispose()
    print("\nMigration 017 completed successfully!")
    print("""
Next steps:
1. Update scheduler.py to capture L-30, L-15, L-5, L0, L+5, L+10 snapshots
2. Use lineup_detected_at from odds_snapshots as reference point
3. Compute delta_p on normalized probabilities
""")


if __name__ == "__main__":
    asyncio.run(migrate())
