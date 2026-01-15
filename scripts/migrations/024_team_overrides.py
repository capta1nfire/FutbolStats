"""Migration 024: Team identity overrides.

Creates the team_overrides table for handling rebranding cases where
API-Football hasn't updated team names/logos but we need to show
the correct identity to users.

Example: La Equidad → Internacional de Bogotá (effective 2026-01-01)
"""

import asyncio
import logging

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


MIGRATION_SQL = """
-- Create team_overrides table
CREATE TABLE IF NOT EXISTS team_overrides (
    id SERIAL PRIMARY KEY,
    provider VARCHAR(50) NOT NULL DEFAULT 'api_football',
    external_team_id INTEGER NOT NULL,
    display_name VARCHAR(255) NOT NULL,
    display_logo_url VARCHAR(500),
    effective_from TIMESTAMP NOT NULL,
    effective_to TIMESTAMP,
    reason VARCHAR(500),
    updated_by VARCHAR(100),
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_team_override UNIQUE (provider, external_team_id, effective_from)
);

-- Create index for efficient lookup
CREATE INDEX IF NOT EXISTS ix_team_overrides_lookup
ON team_overrides (provider, external_team_id, effective_from);

-- Create index on external_team_id for batch preloading
CREATE INDEX IF NOT EXISTS ix_team_overrides_external_id
ON team_overrides (external_team_id);
"""


# Initial data: Internacional de Bogotá override
SEED_DATA_SQL = """
INSERT INTO team_overrides (
    provider, external_team_id, display_name, display_logo_url,
    effective_from, effective_to, reason, updated_by
) VALUES (
    'api_football',
    1134,
    'Internacional de Bogotá',
    'https://upload.wikimedia.org/wikipedia/commons/e/e6/Internacional_de_Bogot%C3%A1_Logo.svg',
    '2026-01-01 00:00:00',
    NULL,
    'Rebranding: La Equidad → Internacional de Bogotá (adquisición Tylis-Porter Group, dic 2025)',
    'migration_024'
) ON CONFLICT (provider, external_team_id, effective_from) DO UPDATE SET
    display_name = EXCLUDED.display_name,
    display_logo_url = EXCLUDED.display_logo_url,
    reason = EXCLUDED.reason,
    updated_at = NOW();
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

    # Step 1: Create table (DDL)
    async with engine.begin() as conn:
        # Check if table exists
        result = await conn.execute(text("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables
                WHERE table_name = 'team_overrides'
            )
        """))
        table_exists = result.scalar()

        if table_exists:
            logger.info("Table team_overrides already exists, skipping DDL...")
        else:
            logger.info("Creating team_overrides table...")
            # Run migration (CREATE IF NOT EXISTS is safe)
            for statement in MIGRATION_SQL.strip().split(";"):
                statement = statement.strip()
                if statement and not statement.startswith("--"):
                    logger.info(f"Executing: {statement[:60]}...")
                    await conn.execute(text(statement))
            logger.info("Table and indexes created.")

    # Step 2: Insert seed data (separate transaction after DDL commits)
    async with engine.begin() as conn:
        logger.info("Inserting seed data...")
        await conn.execute(text(SEED_DATA_SQL))

        # Verify
        result = await conn.execute(text("SELECT COUNT(*) FROM team_overrides"))
        count = result.scalar() or 0

        result = await conn.execute(text("""
            SELECT display_name, effective_from
            FROM team_overrides
            WHERE external_team_id = 1134
        """))
        row = result.fetchone()

        if row:
            logger.info(f"Seed data verified: {row[0]} effective from {row[1]}")

        logger.info(f"Migration complete. {count} override(s) in table.")


if __name__ == "__main__":
    asyncio.run(main())
