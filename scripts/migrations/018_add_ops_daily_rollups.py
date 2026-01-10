#!/usr/bin/env python3
"""
Migration 018: Add ops_daily_rollups table for persistent KPI tracking.

This table stores aggregated daily metrics for the ops dashboard,
replacing ephemeral log-based monitoring with historical data.

Schema:
- day: DATE PRIMARY KEY (UTC date)
- payload: JSONB with metrics (pit_snapshots, bets_evaluable, baseline_coverage, etc.)
- created_at/updated_at: Timestamps for audit
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
        # Create ops_daily_rollups table
        await conn.execute(text("""
            CREATE TABLE IF NOT EXISTS ops_daily_rollups (
                day DATE PRIMARY KEY,
                payload JSONB NOT NULL,
                created_at TIMESTAMP DEFAULT NOW(),
                updated_at TIMESTAMP DEFAULT NOW()
            )
        """))
        print("✓ Created ops_daily_rollups table")

        # Add index on updated_at for efficient recent queries
        await conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_ops_daily_rollups_updated
            ON ops_daily_rollups (updated_at DESC)
        """))
        print("✓ Added index on updated_at")

        # Add comment explaining the table
        await conn.execute(text("""
            COMMENT ON TABLE ops_daily_rollups IS
            'Daily aggregated KPIs for ops dashboard. Populated by daily_ops_rollup scheduler job at 09:05 UTC.'
        """))
        print("✓ Added table comment")

    await engine.dispose()
    print("\n✅ Migration 018 completed successfully!")


if __name__ == "__main__":
    asyncio.run(run_migration())
