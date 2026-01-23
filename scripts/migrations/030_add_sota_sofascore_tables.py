"""Migration 030: SOTA Sofascore Tables (XI / ratings / formation).

Creates Sofascore enrichment tables for SOTA feature engineering:
- match_sofascore_player: XI player ratings/positions captured pre-kickoff
- match_sofascore_lineup: formation snapshot per team_side captured pre-kickoff

Reference:
- docs/ARCHITECTURE_SOTA.md section 1.3 (Sofascore tables)
- docs/FEATURE_DICTIONARY_SOTA.md section 3 (xi_* features)

Guardrails:
- Column names are EXACT as defined in the architecture doc.
- Idempotent: safe to run multiple times.
"""

import asyncio
import logging

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# Table 1: match_sofascore_player
# Stores one latest pre-kickoff snapshot per player per match+side.
# =============================================================================
SQL_MATCH_SOFASCORE_PLAYER = """
CREATE TABLE IF NOT EXISTS match_sofascore_player (
    match_id INTEGER NOT NULL REFERENCES matches(id) ON DELETE CASCADE,
    team_side VARCHAR(10) NOT NULL,  -- 'home'|'away'
    player_id_ext VARCHAR(100) NOT NULL,
    position VARCHAR(20) NOT NULL,  -- GK/DEF/MID/FWD (or sub-role if available)
    is_starter BOOLEAN NOT NULL,
    rating_pre_match FLOAT,  -- nullable
    rating_recent_form FLOAT,  -- nullable
    minutes_expected INTEGER,  -- nullable
    captured_at TIMESTAMP NOT NULL DEFAULT NOW(),
    PRIMARY KEY (match_id, team_side, player_id_ext)
);

CREATE INDEX IF NOT EXISTS ix_match_sofascore_player_captured_at
ON match_sofascore_player (captured_at);

CREATE INDEX IF NOT EXISTS ix_match_sofascore_player_match_side
ON match_sofascore_player (match_id, team_side);

COMMENT ON TABLE match_sofascore_player IS 'Sofascore XI players/ratings snapshot per match (ARCHITECTURE_SOTA.md 1.3)';
"""


# =============================================================================
# Table 2: match_sofascore_lineup
# Stores one latest pre-kickoff formation snapshot per match+side.
# =============================================================================
SQL_MATCH_SOFASCORE_LINEUP = """
CREATE TABLE IF NOT EXISTS match_sofascore_lineup (
    match_id INTEGER NOT NULL REFERENCES matches(id) ON DELETE CASCADE,
    team_side VARCHAR(10) NOT NULL,  -- 'home'|'away'
    formation VARCHAR(20) NOT NULL,  -- e.g., '4-3-3', '4-2-3-1'
    captured_at TIMESTAMP NOT NULL DEFAULT NOW(),
    PRIMARY KEY (match_id, team_side)
);

CREATE INDEX IF NOT EXISTS ix_match_sofascore_lineup_captured_at
ON match_sofascore_lineup (captured_at);

COMMENT ON TABLE match_sofascore_lineup IS 'Sofascore formation snapshot per match (ARCHITECTURE_SOTA.md 1.3)';
"""


TABLES = [
    ("match_sofascore_player", SQL_MATCH_SOFASCORE_PLAYER),
    ("match_sofascore_lineup", SQL_MATCH_SOFASCORE_LINEUP),
]


async def table_exists(conn, table_name: str) -> bool:
    """Check if a table exists in the database."""
    result = await conn.execute(
        text(
            """
            SELECT EXISTS (
                SELECT FROM information_schema.tables
                WHERE table_schema = 'public' AND table_name = :table_name
            )
            """
        ),
        {"table_name": table_name},
    )
    return bool(result.scalar())


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
    results = {"created": [], "already_exists": [], "errors": []}

    async with engine.begin() as conn:
        for table_name, sql in TABLES:
            try:
                exists = await table_exists(conn, table_name)
                if exists:
                    logger.info(f"Table {table_name} already exists, skipping...")
                    results["already_exists"].append(table_name)
                    continue

                logger.info(f"Creating table {table_name}...")
                for statement in sql.strip().split(";"):
                    statement = statement.strip()
                    if statement and not statement.startswith("--"):
                        await conn.execute(text(statement))
                results["created"].append(table_name)
                logger.info(f"Table {table_name} created.")
            except Exception as e:
                logger.error(f"Error creating table {table_name}: {e}")
                results["errors"].append((table_name, str(e)))

        logger.info("\n" + "=" * 60)
        logger.info("VERIFICATION - Tables status:")
        logger.info("=" * 60)
        for table_name, _ in TABLES:
            try:
                exists = await table_exists(conn, table_name)
                if exists:
                    result = await conn.execute(text(f"SELECT COUNT(*) FROM {table_name}"))
                    count = int(result.scalar() or 0)
                    logger.info(f"  {table_name}: EXISTS ({count} rows)")
                else:
                    logger.info(f"  {table_name}: NOT EXISTS")
            except Exception as e:
                logger.info(f"  {table_name}: ERROR ({e})")

    logger.info("\n" + "=" * 60)
    logger.info("MIGRATION SUMMARY:")
    logger.info("=" * 60)
    logger.info(f"  Created: {results['created']}")
    logger.info(f"  Already existed: {results['already_exists']}")

    if results["errors"]:
        logger.error(f"  Errors: {results['errors']}")
        raise RuntimeError(f"Migration failed with errors: {results['errors']}")

    logger.info("\nMigration 030 complete.")


if __name__ == "__main__":
    asyncio.run(main())

