"""Migration 023: Shadow predictions idempotency fix.

Removes duplicate shadow predictions keeping the most recent per match_id,
and changes the unique constraint from (match_id, created_at) to just (match_id).

This ensures log_shadow_prediction() is idempotent.
"""

import asyncio
import logging

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


MIGRATION_SQL = """
-- Step 1: Remove duplicates keeping the most recent per match_id
DELETE FROM shadow_predictions sp1
WHERE sp1.id NOT IN (
    SELECT DISTINCT ON (match_id) id
    FROM shadow_predictions
    ORDER BY match_id, created_at DESC
);

-- Step 2: Drop the old unique constraint on (match_id, created_at)
ALTER TABLE shadow_predictions DROP CONSTRAINT IF EXISTS uq_shadow_match_created;

-- Step 3: Add new unique constraint on just match_id
ALTER TABLE shadow_predictions ADD CONSTRAINT uq_shadow_match_id UNIQUE (match_id);
"""


async def run_migration(session: AsyncSession) -> dict:
    """Run the migration to fix shadow predictions idempotency."""

    # Get count of duplicates before
    result = await session.execute(text("""
        SELECT COUNT(*) - COUNT(DISTINCT match_id) as duplicates
        FROM shadow_predictions
    """))
    duplicates_before = result.scalar() or 0

    logger.info(f"Found {duplicates_before} duplicate shadow predictions to remove")

    # Run migration
    for statement in MIGRATION_SQL.strip().split(";"):
        statement = statement.strip()
        if statement and not statement.startswith("--"):
            logger.info(f"Executing: {statement[:60]}...")
            await session.execute(text(statement))

    await session.commit()

    # Verify
    result = await session.execute(text("""
        SELECT COUNT(*) FROM shadow_predictions
    """))
    remaining = result.scalar() or 0

    logger.info(f"Migration complete. Removed {duplicates_before} duplicates. {remaining} records remaining.")

    return {
        "duplicates_removed": duplicates_before,
        "records_remaining": remaining,
        "status": "ok",
    }


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
        # Use raw connection for migration
        result = await conn.execute(text("""
            SELECT COUNT(*) - COUNT(DISTINCT match_id) as duplicates
            FROM shadow_predictions
        """))
        duplicates_before = result.scalar() or 0

        logger.info(f"Found {duplicates_before} duplicate shadow predictions to remove")

        for statement in MIGRATION_SQL.strip().split(";"):
            statement = statement.strip()
            if statement and not statement.startswith("--"):
                logger.info(f"Executing: {statement[:60]}...")
                await conn.execute(text(statement))

        result = await conn.execute(text("SELECT COUNT(*) FROM shadow_predictions"))
        remaining = result.scalar() or 0

        logger.info(f"Migration complete. Removed {duplicates_before} duplicates. {remaining} records remaining.")


if __name__ == "__main__":
    asyncio.run(main())
