"""Migration 025: Prediction reruns audit table.

Creates the prediction_reruns table for tracking manual re-prediction runs
with before/after stats and evaluation metrics.

Also adds run_id column to predictions table for tracking which rerun
generated each prediction.
"""

import asyncio
import logging

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


MIGRATION_SQL = """
-- Create prediction_reruns table for audit/tracking
CREATE TABLE IF NOT EXISTS prediction_reruns (
    id SERIAL PRIMARY KEY,
    run_id UUID NOT NULL UNIQUE,
    run_type VARCHAR(50) NOT NULL,  -- 'manual_rerun', 'model_promotion', 'rollback'

    -- Configuration
    window_hours INT NOT NULL,
    architecture_before VARCHAR(50) NOT NULL,
    architecture_after VARCHAR(50) NOT NULL,
    model_version_before VARCHAR(50) NOT NULL,
    model_version_after VARCHAR(50) NOT NULL,

    -- Scope
    matches_total INT NOT NULL,
    matches_with_odds INT NOT NULL,

    -- Before/After stats (JSON)
    stats_before JSONB NOT NULL,
    stats_after JSONB NOT NULL,

    -- Top changes for review
    top_deltas JSONB,  -- Array of {match_id, delta_draw, before_probs, after_probs}

    -- Outcome metrics (filled when matches complete via scheduled job)
    evaluation_window_days INT DEFAULT 14,
    evaluated_matches INT DEFAULT 0,
    evaluation_report JSONB,  -- {baseline_accuracy, rerun_accuracy, baseline_brier, rerun_brier, ...}

    -- Status for serving preference
    is_active BOOLEAN DEFAULT TRUE,  -- If false, rollback: serve baseline instead

    -- Metadata
    triggered_by VARCHAR(100),
    notes TEXT,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    evaluated_at TIMESTAMP
);

-- Create index for active reruns lookup
CREATE INDEX IF NOT EXISTS ix_prediction_reruns_active
ON prediction_reruns (is_active, created_at DESC);

-- Add run_id column to predictions table (nullable for existing rows)
ALTER TABLE predictions
ADD COLUMN IF NOT EXISTS run_id UUID;

-- Create index for run_id lookup
CREATE INDEX IF NOT EXISTS ix_predictions_run_id
ON predictions (run_id) WHERE run_id IS NOT NULL;
"""


async def main():
    """Run migration from command line."""
    import os

    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        raise ValueError("DATABASE_URL environment variable not set")

    # Convert to async URL if needed
    if database_url.startswith("postgresql://"):
        database_url = database_url.replace("postgresql://", "postgresql+asyncpg://", 1)

    engine = create_async_engine(database_url, echo=False)

    async with engine.begin() as conn:
        # Check if table exists
        result = await conn.execute(text("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables
                WHERE table_name = 'prediction_reruns'
            )
        """))
        table_exists = result.scalar()

        if table_exists:
            logger.info("Table prediction_reruns already exists, checking for updates...")
        else:
            logger.info("Creating prediction_reruns table...")

        # Run migration (all statements are idempotent)
        for statement in MIGRATION_SQL.strip().split(";"):
            statement = statement.strip()
            if statement and not statement.startswith("--"):
                logger.info(f"Executing: {statement[:60]}...")
                try:
                    await conn.execute(text(statement))
                except Exception as e:
                    # ALTER TABLE ADD COLUMN IF NOT EXISTS may fail on some PG versions
                    if "already exists" in str(e).lower():
                        logger.info(f"Column already exists, skipping...")
                    else:
                        raise

    # Verify in separate connection after DDL commits
    async with engine.begin() as conn:
        # Verify table created
        result = await conn.execute(text("SELECT COUNT(*) FROM prediction_reruns"))
        count = result.scalar() or 0

        # Check run_id column exists
        result = await conn.execute(text("""
            SELECT EXISTS (
                SELECT FROM information_schema.columns
                WHERE table_name = 'predictions' AND column_name = 'run_id'
            )
        """))
        run_id_exists = result.scalar()

        logger.info(f"Migration complete. {count} rerun(s) in table, run_id column: {run_id_exists}")


if __name__ == "__main__":
    asyncio.run(main())
