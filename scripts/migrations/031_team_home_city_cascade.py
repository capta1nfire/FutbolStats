"""Migration 031: Team Home City Cascade.

Extends team_home_city_profile with source tracking columns and creates
team_home_city_overrides table for manual corrections.

Part of the fallback cascade pipeline:
  venue_city -> venue_name geocoding -> LLM candidate -> manual override

Reference: Plan "Fallback Cascade para team_home_city_profile"
"""

import asyncio
import logging

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# 1a) Extend team_home_city_profile with source tracking columns
# =============================================================================
SQL_ALTER_PROFILE = """
ALTER TABLE team_home_city_profile
  ADD COLUMN IF NOT EXISTS source VARCHAR(50) NOT NULL DEFAULT 'venue_city';

ALTER TABLE team_home_city_profile
  ADD COLUMN IF NOT EXISTS confidence FLOAT NOT NULL DEFAULT 0.9;

ALTER TABLE team_home_city_profile
  ADD COLUMN IF NOT EXISTS needs_review BOOLEAN NOT NULL DEFAULT false;

ALTER TABLE team_home_city_profile
  ADD COLUMN IF NOT EXISTS last_updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW();
"""

SQL_PROFILE_INDEXES = """
CREATE INDEX IF NOT EXISTS ix_thcp_source ON team_home_city_profile(source);
CREATE INDEX IF NOT EXISTS ix_thcp_needs_review ON team_home_city_profile(needs_review) WHERE needs_review = true;
"""

# =============================================================================
# 1b) Create team_home_city_overrides table
# =============================================================================
SQL_OVERRIDES_TABLE = """
CREATE TABLE IF NOT EXISTS team_home_city_overrides (
    team_id INTEGER NOT NULL REFERENCES teams(id) ON DELETE CASCADE PRIMARY KEY,
    home_city VARCHAR(255) NOT NULL,
    timezone VARCHAR(50),
    reason VARCHAR(500),
    active BOOLEAN NOT NULL DEFAULT true,
    created_by VARCHAR(100) NOT NULL DEFAULT 'manual',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_by VARCHAR(100)
);
"""

SQL_OVERRIDES_COMMENT = """
COMMENT ON TABLE team_home_city_overrides IS 'Manual home city overrides for teams where automated cascade is insufficient';
"""


async def main():
    """Run migration from command line."""
    import os

    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        raise ValueError("DATABASE_URL environment variable not set")

    if database_url.startswith("postgresql://"):
        database_url = database_url.replace("postgresql://", "postgresql+asyncpg://", 1)

    engine = create_async_engine(database_url, echo=False)

    async with engine.begin() as conn:
        # 1a) ALTER TABLE: add columns
        logger.info("Adding source tracking columns to team_home_city_profile...")
        for stmt in SQL_ALTER_PROFILE.strip().split(";"):
            stmt = stmt.strip()
            if stmt and not stmt.startswith("--"):
                await conn.execute(text(stmt))
        logger.info("  Columns added (or already exist).")

        # 1a) Indexes
        logger.info("Creating indexes on team_home_city_profile...")
        for stmt in SQL_PROFILE_INDEXES.strip().split(";"):
            stmt = stmt.strip()
            if stmt and not stmt.startswith("--"):
                await conn.execute(text(stmt))
        logger.info("  Indexes created (or already exist).")

        # 1b) CREATE TABLE overrides
        logger.info("Creating team_home_city_overrides table...")
        await conn.execute(text(SQL_OVERRIDES_TABLE.strip()))
        logger.info("  Table created (or already exists).")

        await conn.execute(text(SQL_OVERRIDES_COMMENT.strip()))

        # Verification
        logger.info("\n" + "=" * 60)
        logger.info("VERIFICATION:")
        logger.info("=" * 60)

        # Check profile columns
        result = await conn.execute(text("""
            SELECT column_name, data_type, is_nullable
            FROM information_schema.columns
            WHERE table_name = 'team_home_city_profile'
            ORDER BY ordinal_position
        """))
        rows = result.fetchall()
        logger.info("team_home_city_profile columns:")
        for row in rows:
            logger.info(f"  {row[0]}: {row[1]} (nullable={row[2]})")

        # Check overrides table exists
        result = await conn.execute(text("""
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_name = 'team_home_city_overrides'
            ORDER BY ordinal_position
        """))
        rows = result.fetchall()
        logger.info("team_home_city_overrides columns:")
        for row in rows:
            logger.info(f"  {row[0]}: {row[1]}")

    await engine.dispose()
    logger.info("\nMigration 031 complete.")


if __name__ == "__main__":
    asyncio.run(main())
