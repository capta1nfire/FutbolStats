"""
Migration 003: Enhance team_adjustments for contextual intelligence.

Adds:
- home_multiplier / away_multiplier (split adjustments)
- consecutive_minimal_count (for recovery factor)
- last_anomaly_date (tracking)
- international_penalty (for tournament context)

Run with:
    DATABASE_URL="postgresql://..." python scripts/migrations/003_enhance_team_adjustments.py
"""

import asyncio
import logging
import os
import sys

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


MIGRATION_SQL = [
    # Add home/away split multipliers
    """
    ALTER TABLE team_adjustments
    ADD COLUMN IF NOT EXISTS home_multiplier FLOAT NOT NULL DEFAULT 1.0
    """,
    """
    ALTER TABLE team_adjustments
    ADD COLUMN IF NOT EXISTS away_multiplier FLOAT NOT NULL DEFAULT 1.0
    """,

    # Add recovery tracking (consecutive minimal audits)
    """
    ALTER TABLE team_adjustments
    ADD COLUMN IF NOT EXISTS consecutive_minimal_count INT NOT NULL DEFAULT 0
    """,

    # Add last anomaly tracking
    """
    ALTER TABLE team_adjustments
    ADD COLUMN IF NOT EXISTS last_anomaly_date TIMESTAMP
    """,

    # Add international penalty factor
    """
    ALTER TABLE team_adjustments
    ADD COLUMN IF NOT EXISTS international_penalty FLOAT NOT NULL DEFAULT 1.0
    """,

    # Add home/away specific metrics
    """
    ALTER TABLE team_adjustments
    ADD COLUMN IF NOT EXISTS home_predictions INT NOT NULL DEFAULT 0
    """,
    """
    ALTER TABLE team_adjustments
    ADD COLUMN IF NOT EXISTS home_correct INT NOT NULL DEFAULT 0
    """,
    """
    ALTER TABLE team_adjustments
    ADD COLUMN IF NOT EXISTS home_anomalies INT NOT NULL DEFAULT 0
    """,
    """
    ALTER TABLE team_adjustments
    ADD COLUMN IF NOT EXISTS away_predictions INT NOT NULL DEFAULT 0
    """,
    """
    ALTER TABLE team_adjustments
    ADD COLUMN IF NOT EXISTS away_correct INT NOT NULL DEFAULT 0
    """,
    """
    ALTER TABLE team_adjustments
    ADD COLUMN IF NOT EXISTS away_anomalies INT NOT NULL DEFAULT 0
    """,

    # Migrate existing data: copy confidence_multiplier to both home and away
    """
    UPDATE team_adjustments
    SET home_multiplier = confidence_multiplier,
        away_multiplier = confidence_multiplier
    WHERE home_multiplier = 1.0 AND confidence_multiplier != 1.0
    """,
]


async def run_migration():
    """Execute the migration."""
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        logger.error("DATABASE_URL environment variable not set")
        sys.exit(1)

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
                if "already exists" in str(e).lower() or "duplicate column" in str(e).lower():
                    logger.info(f"Statement {i + 1} skipped (already exists)")
                else:
                    logger.error(f"Error in statement {i + 1}: {e}")
                    raise

    logger.info("Migration 003 completed successfully!")

    # Verify columns
    async with async_engine.connect() as conn:
        try:
            result = await conn.execute(text("""
                SELECT column_name FROM information_schema.columns
                WHERE table_name = 'team_adjustments'
                ORDER BY ordinal_position
            """))
            columns = [row[0] for row in result.fetchall()]
            logger.info(f"team_adjustments columns: {columns}")
        except Exception as e:
            logger.warning(f"Could not verify columns: {e}")

    await async_engine.dispose()


if __name__ == "__main__":
    asyncio.run(run_migration())
