"""Migration 027: Add elapsed field to matches table.

Adds the `elapsed` column to store the current minute for live matches,
enabling the iOS app to display "32'" instead of "1H".
"""

import asyncio
import logging

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


MIGRATION_SQL = """
-- Add elapsed column for live match minute tracking
ALTER TABLE matches ADD COLUMN IF NOT EXISTS elapsed INTEGER DEFAULT NULL;

-- Add comment for documentation
COMMENT ON COLUMN matches.elapsed IS 'Current minute for live matches (from API-Football status.elapsed)';
"""


async def run_migration(database_url: str):
    """Run the migration."""
    # Convert sync URL to async if needed
    if database_url.startswith("postgresql://"):
        database_url = database_url.replace("postgresql://", "postgresql+asyncpg://")

    engine = create_async_engine(database_url)

    async with engine.begin() as conn:
        logger.info("Running migration 027: Add elapsed field to matches")
        await conn.execute(text(MIGRATION_SQL))
        logger.info("Migration 027 completed successfully")

    await engine.dispose()


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv

    load_dotenv()

    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise ValueError("DATABASE_URL environment variable not set")

    asyncio.run(run_migration(database_url))
