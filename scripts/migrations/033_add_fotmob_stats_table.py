"""Migration 033: FotMob xG stats table.

Stores team-level xG data from FotMob (Opta source) for leagues without Understat coverage.
ABE P0 2026-02-08: Only team-level xG/xGOT + raw_stats JSONB. No shotmap, no player xG.

Guardrails:
- Idempotent: safe to run multiple times (CREATE TABLE IF NOT EXISTS).
- ON CONFLICT upsert in jobs.
- Separate from Understat tables (029) — fixed source per league.
"""

import asyncio
import logging

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


SQL_MATCH_FOTMOB_STATS = """
CREATE TABLE IF NOT EXISTS match_fotmob_stats (
    match_id INTEGER NOT NULL PRIMARY KEY
        REFERENCES matches(id) ON DELETE CASCADE,
    xg_home DOUBLE PRECISION,
    xg_away DOUBLE PRECISION,
    xgot_home DOUBLE PRECISION,
    xgot_away DOUBLE PRECISION,
    xg_open_play_home DOUBLE PRECISION,
    xg_open_play_away DOUBLE PRECISION,
    xg_set_play_home DOUBLE PRECISION,
    xg_set_play_away DOUBLE PRECISION,
    raw_stats JSONB,
    captured_at TIMESTAMP NOT NULL DEFAULT NOW(),
    source_version VARCHAR(50) DEFAULT 'fotmob_opta_v1'
);

CREATE INDEX IF NOT EXISTS ix_match_fotmob_stats_captured_at
ON match_fotmob_stats (captured_at);

COMMENT ON TABLE match_fotmob_stats IS 'Post-match FotMob team xG statistics (Opta source) for non-Understat leagues';
"""


TABLES = [
    ("match_fotmob_stats", SQL_MATCH_FOTMOB_STATS),
]


async def table_exists(conn, table_name: str) -> bool:
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
    import os

    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        raise ValueError("DATABASE_URL environment variable not set")

    if database_url.startswith("postgresql://"):
        database_url = database_url.replace("postgresql://", "postgresql+asyncpg://", 1)

    engine = create_async_engine(database_url, echo=False)
    results = {"created": [], "already_exists": [], "errors": []}

    async with engine.begin() as conn:
        for table_name, sql in TABLES:
            try:
                exists = await table_exists(conn, table_name)
                if exists:
                    logger.info(f"Table {table_name} already exists, ensuring indexes/comments...")
                else:
                    logger.info(f"Creating table {table_name}...")

                for statement in sql.strip().split(";"):
                    statement = statement.strip()
                    if statement and not statement.startswith("--"):
                        await conn.execute(text(statement))

                if exists:
                    results["already_exists"].append(table_name)
                else:
                    results["created"].append(table_name)
                logger.info(f"Table {table_name}: {'ensured' if exists else 'created'}.")
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

    logger.info("\nMigration 033 complete.")


if __name__ == "__main__":
    asyncio.run(main())
