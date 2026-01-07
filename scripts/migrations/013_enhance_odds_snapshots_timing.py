#!/usr/bin/env python3
"""
Migration 013: Enhance odds_snapshots with timing metadata.

Adds columns to track:
- kickoff_time: The scheduled match kickoff
- delta_to_kickoff_seconds: Time between snapshot and kickoff (for analysis)
- odds_source_freshness: 'live', 'stale_history', 'stale_match'

This allows us to validate that lineup_confirmed snapshots are:
1. Captured at ~60±15 min before kickoff
2. Using fresh (live) odds, not stale data
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine


async def run_migration():
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        raise ValueError("DATABASE_URL environment variable required")

    if database_url.startswith("postgresql://"):
        database_url = database_url.replace("postgresql://", "postgresql+asyncpg://", 1)

    engine = create_async_engine(database_url)

    async with engine.begin() as conn:
        # Add timing columns to odds_snapshots
        await conn.execute(text("""
            ALTER TABLE odds_snapshots
            ADD COLUMN IF NOT EXISTS kickoff_time TIMESTAMP,
            ADD COLUMN IF NOT EXISTS delta_to_kickoff_seconds INTEGER,
            ADD COLUMN IF NOT EXISTS odds_freshness VARCHAR(20) DEFAULT 'unknown'
        """))
        print("✓ Added timing columns to odds_snapshots")

        # Add comment explaining the columns
        await conn.execute(text("""
            COMMENT ON COLUMN odds_snapshots.kickoff_time IS 'Scheduled match kickoff time';
        """))
        await conn.execute(text("""
            COMMENT ON COLUMN odds_snapshots.delta_to_kickoff_seconds IS 'Seconds between snapshot_at and kickoff (positive = before kickoff)';
        """))
        await conn.execute(text("""
            COMMENT ON COLUMN odds_snapshots.odds_freshness IS 'live (API call at moment), stale_history, stale_match';
        """))
        print("✓ Added column comments")

        # Backfill kickoff_time and delta for existing snapshots
        await conn.execute(text("""
            UPDATE odds_snapshots os
            SET
                kickoff_time = m.date,
                delta_to_kickoff_seconds = EXTRACT(EPOCH FROM (m.date - os.snapshot_at))::INTEGER
            FROM matches m
            WHERE os.match_id = m.id
              AND os.kickoff_time IS NULL
        """))
        print("✓ Backfilled timing data for existing snapshots")

    await engine.dispose()
    print("\n✅ Migration 013 completed successfully!")


if __name__ == "__main__":
    asyncio.run(run_migration())
