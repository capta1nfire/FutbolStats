#!/usr/bin/env python3
"""
Migration 005: Add frozen prediction fields.

When a match transitions from NS (Not Started) to in-play or finished,
we freeze the prediction so the user sees the ORIGINAL prediction they
saw before the match, not a recalculated one after model retraining.

This preserves:
- The model's original probability predictions
- The bookmaker odds at freeze time
- The EV calculations at freeze time
- The confidence tier at freeze time
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
    """Add frozen prediction fields to the predictions table."""
    from sqlalchemy.ext.asyncio import create_async_engine

    # Convert postgres:// to postgresql+asyncpg://
    db_url = DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://")
    engine = create_async_engine(db_url)

    async with engine.begin() as conn:
        # Check if is_frozen column already exists
        result = await conn.execute(text("""
            SELECT EXISTS (
                SELECT FROM information_schema.columns
                WHERE table_name = 'predictions' AND column_name = 'is_frozen'
            )
        """))
        exists = result.scalar()

        if exists:
            print("Column 'is_frozen' already exists, skipping migration")
            return

        # Add frozen state columns
        print("Adding frozen prediction columns...")

        # is_frozen - Whether the prediction is locked
        await conn.execute(text("""
            ALTER TABLE predictions ADD COLUMN is_frozen BOOLEAN DEFAULT FALSE
        """))
        print("  Added: is_frozen")

        # frozen_at - When the prediction was frozen
        await conn.execute(text("""
            ALTER TABLE predictions ADD COLUMN frozen_at TIMESTAMP WITH TIME ZONE
        """))
        print("  Added: frozen_at")

        # Frozen bookmaker odds at freeze time
        await conn.execute(text("""
            ALTER TABLE predictions ADD COLUMN frozen_odds_home FLOAT
        """))
        print("  Added: frozen_odds_home")

        await conn.execute(text("""
            ALTER TABLE predictions ADD COLUMN frozen_odds_draw FLOAT
        """))
        print("  Added: frozen_odds_draw")

        await conn.execute(text("""
            ALTER TABLE predictions ADD COLUMN frozen_odds_away FLOAT
        """))
        print("  Added: frozen_odds_away")

        # Frozen EV calculations at freeze time
        await conn.execute(text("""
            ALTER TABLE predictions ADD COLUMN frozen_ev_home FLOAT
        """))
        print("  Added: frozen_ev_home")

        await conn.execute(text("""
            ALTER TABLE predictions ADD COLUMN frozen_ev_draw FLOAT
        """))
        print("  Added: frozen_ev_draw")

        await conn.execute(text("""
            ALTER TABLE predictions ADD COLUMN frozen_ev_away FLOAT
        """))
        print("  Added: frozen_ev_away")

        # Frozen confidence tier at freeze time
        await conn.execute(text("""
            ALTER TABLE predictions ADD COLUMN frozen_confidence_tier VARCHAR(10)
        """))
        print("  Added: frozen_confidence_tier")

        # Frozen value bets JSON (preserves which bets had value at freeze time)
        await conn.execute(text("""
            ALTER TABLE predictions ADD COLUMN frozen_value_bets JSONB
        """))
        print("  Added: frozen_value_bets")

        # Create index for efficient queries on frozen predictions
        await conn.execute(text("""
            CREATE INDEX idx_predictions_is_frozen ON predictions(is_frozen) WHERE is_frozen = TRUE
        """))
        print("  Created index: idx_predictions_is_frozen")

        print("\nMigration 005 completed successfully!")

    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(run_migration())
