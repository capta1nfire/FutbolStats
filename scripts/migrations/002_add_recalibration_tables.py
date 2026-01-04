"""
Migration 002: Add recalibration tables for Phase 2.

Creates:
- team_adjustments: Per-team confidence adjustments based on anomalies
- model_snapshots: Model version history for rollback capability

Run with:
    DATABASE_URL="postgresql://..." python scripts/migrations/002_add_recalibration_tables.py
"""

import asyncio
import logging
import os
import sys

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


MIGRATION_SQL = [
    # Table 1: team_adjustments
    """
    CREATE TABLE IF NOT EXISTS team_adjustments (
        id SERIAL PRIMARY KEY,
        team_id INTEGER NOT NULL UNIQUE REFERENCES teams(id) ON DELETE CASCADE,

        -- Confidence adjustment (1.0 = no change, 0.9 = reduce 10%)
        confidence_multiplier FLOAT NOT NULL DEFAULT 1.0,

        -- Metrics justifying the adjustment
        total_predictions INTEGER NOT NULL DEFAULT 0,
        correct_predictions INTEGER NOT NULL DEFAULT 0,
        anomaly_count INTEGER NOT NULL DEFAULT 0,
        avg_deviation_score FLOAT NOT NULL DEFAULT 0.0,

        -- Control
        last_updated TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
        adjustment_reason VARCHAR(200),

        -- Index for fast lookups
        CONSTRAINT chk_multiplier CHECK (confidence_multiplier > 0 AND confidence_multiplier <= 2.0)
    )
    """,

    # Index for team_adjustments
    """
    CREATE INDEX IF NOT EXISTS idx_team_adjustments_team_id ON team_adjustments(team_id)
    """,

    # Table 2: model_snapshots
    """
    CREATE TABLE IF NOT EXISTS model_snapshots (
        id SERIAL PRIMARY KEY,
        model_version VARCHAR(50) NOT NULL,

        -- Model file path
        model_path VARCHAR(500) NOT NULL,

        -- Validation metrics
        brier_score FLOAT NOT NULL,
        cv_brier_scores JSONB,
        samples_trained INTEGER NOT NULL,

        -- Status
        is_active BOOLEAN NOT NULL DEFAULT FALSE,
        is_baseline BOOLEAN NOT NULL DEFAULT FALSE,

        -- Metadata
        created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
        training_config JSONB
    )
    """,

    # Indexes for model_snapshots
    """
    CREATE INDEX IF NOT EXISTS idx_model_snapshots_version ON model_snapshots(model_version)
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_model_snapshots_active ON model_snapshots(is_active) WHERE is_active = TRUE
    """,

    # Ensure only one active snapshot at a time (partial unique index)
    """
    CREATE UNIQUE INDEX IF NOT EXISTS idx_model_snapshots_one_active
    ON model_snapshots(is_active) WHERE is_active = TRUE
    """,
]


async def run_migration():
    """Execute the migration."""
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        logger.error("DATABASE_URL environment variable not set")
        sys.exit(1)

    # Convert to async URL if needed
    if database_url.startswith("postgresql://"):
        database_url = database_url.replace("postgresql://", "postgresql+asyncpg://", 1)

    logger.info("Connecting to database...")
    async_engine = create_async_engine(database_url, echo=False)

    async with async_engine.begin() as conn:
        for i, sql in enumerate(MIGRATION_SQL):
            try:
                await conn.execute(text(sql))
                logger.info(f"Statement {i + 1}/{len(MIGRATION_SQL)} executed successfully")
            except Exception as e:
                if "already exists" in str(e).lower():
                    logger.info(f"Statement {i + 1} skipped (already exists)")
                else:
                    logger.error(f"Error in statement {i + 1}: {e}")
                    raise

    logger.info("Migration 002 completed successfully!")

    # Verify tables were created
    async with async_engine.connect() as conn:
        try:
            result = await conn.execute(text("""
                SELECT table_name FROM information_schema.tables
                WHERE table_schema = 'public'
                AND table_name IN ('team_adjustments', 'model_snapshots')
            """))
            tables = [row[0] for row in result.fetchall()]
            logger.info(f"Verified tables: {tables}")
        except Exception as e:
            logger.warning(f"Could not verify tables: {e}")

    await async_engine.dispose()


if __name__ == "__main__":
    asyncio.run(run_migration())
