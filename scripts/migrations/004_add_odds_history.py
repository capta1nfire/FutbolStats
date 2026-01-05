#!/usr/bin/env python3
"""
Migration 004: Add odds_history table for tracking odds movements.

This table stores snapshots of odds over time, allowing:
- Historical analysis of odds before match starts
- Line movement detection (steam moves, sharp action)
- Backtesting value bets with actual closing odds
- Comparison across bookmakers (future)
"""

import asyncio
import os
from datetime import datetime

from sqlalchemy import text

# Get database URL from environment
DATABASE_URL = os.environ.get("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("DATABASE_URL environment variable is required")


async def run_migration():
    """Create the odds_history table."""
    from sqlalchemy.ext.asyncio import create_async_engine

    # Convert postgres:// to postgresql+asyncpg://
    db_url = DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://")
    engine = create_async_engine(db_url)

    async with engine.begin() as conn:
        # Check if table already exists
        result = await conn.execute(text("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables
                WHERE table_name = 'odds_history'
            )
        """))
        exists = result.scalar()

        if exists:
            print("Table 'odds_history' already exists, skipping creation")
            return

        # Create odds_history table
        await conn.execute(text("""
            CREATE TABLE odds_history (
                id SERIAL PRIMARY KEY,
                match_id INTEGER NOT NULL REFERENCES matches(id) ON DELETE CASCADE,
                recorded_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),

                -- Odds values
                odds_home FLOAT,
                odds_draw FLOAT,
                odds_away FLOAT,

                -- Metadata
                source VARCHAR(50) DEFAULT 'api_football',  -- bookmaker source
                is_opening BOOLEAN DEFAULT FALSE,           -- first recorded odds
                is_closing BOOLEAN DEFAULT FALSE,           -- last odds before kickoff

                -- Computed fields for quick analysis
                implied_home FLOAT,  -- 1/odds_home
                implied_draw FLOAT,  -- 1/odds_draw
                implied_away FLOAT,  -- 1/odds_away
                overround FLOAT,     -- sum of implied probs (margin)

                -- Index for efficient queries
                CONSTRAINT unique_match_timestamp UNIQUE (match_id, recorded_at, source)
            )
        """))
        print("Created table: odds_history")

        # Create indexes for common query patterns
        await conn.execute(text("""
            CREATE INDEX idx_odds_history_match_id ON odds_history(match_id)
        """))
        print("Created index: idx_odds_history_match_id")

        await conn.execute(text("""
            CREATE INDEX idx_odds_history_recorded_at ON odds_history(recorded_at)
        """))
        print("Created index: idx_odds_history_recorded_at")

        await conn.execute(text("""
            CREATE INDEX idx_odds_history_match_time ON odds_history(match_id, recorded_at DESC)
        """))
        print("Created index: idx_odds_history_match_time")

        # Add odds_recorded_at to matches table for quick reference
        result = await conn.execute(text("""
            SELECT EXISTS (
                SELECT FROM information_schema.columns
                WHERE table_name = 'matches' AND column_name = 'odds_recorded_at'
            )
        """))
        col_exists = result.scalar()

        if not col_exists:
            await conn.execute(text("""
                ALTER TABLE matches ADD COLUMN odds_recorded_at TIMESTAMP WITH TIME ZONE
            """))
            print("Added column: matches.odds_recorded_at")

        print("\nMigration 004 completed successfully!")

    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(run_migration())
