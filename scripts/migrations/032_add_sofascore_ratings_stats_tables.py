"""Migration 032: Sofascore Player Rating History + Match Stats tables.

Phase A: sofascore_player_rating_history — post-match player ratings for rolling averages
Phase B: match_sofascore_stats — post-match team statistics (xG, big chances, etc.)

Guardrails:
- Idempotent: safe to run multiple times (CREATE TABLE IF NOT EXISTS).
- ON CONFLICT upsert for ratings corrections.
- Separate from pre-kickoff tables (030) to maintain PIT semantics.
"""

import asyncio
import logging

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# Table 1: sofascore_player_rating_history
# Post-match player ratings. Separate from match_sofascore_player (pre-kickoff).
# =============================================================================
SQL_PLAYER_RATING_HISTORY = """
CREATE TABLE IF NOT EXISTS sofascore_player_rating_history (
    id SERIAL PRIMARY KEY,
    player_id_ext VARCHAR(100) NOT NULL,
    match_id INTEGER NOT NULL REFERENCES matches(id) ON DELETE CASCADE,
    team_side VARCHAR(10) NOT NULL,
    position VARCHAR(20) NOT NULL,
    rating DOUBLE PRECISION NOT NULL,
    minutes_played INTEGER,
    is_starter BOOLEAN NOT NULL DEFAULT TRUE,
    match_date TIMESTAMP NOT NULL,
    captured_at TIMESTAMP NOT NULL DEFAULT NOW(),
    UNIQUE (player_id_ext, match_id)
);

CREATE INDEX IF NOT EXISTS ix_sfpr_player_date
ON sofascore_player_rating_history (player_id_ext, match_date DESC);

CREATE INDEX IF NOT EXISTS ix_sfpr_match_id
ON sofascore_player_rating_history (match_id);

COMMENT ON TABLE sofascore_player_rating_history IS 'Post-match Sofascore player ratings for rolling average feature engineering';
"""


# =============================================================================
# Table 2: match_sofascore_stats
# Post-match team statistics from Sofascore /event/{id}/statistics endpoint.
# =============================================================================
SQL_MATCH_SOFASCORE_STATS = """
CREATE TABLE IF NOT EXISTS match_sofascore_stats (
    match_id INTEGER NOT NULL PRIMARY KEY REFERENCES matches(id) ON DELETE CASCADE,
    possession_home SMALLINT,
    possession_away SMALLINT,
    total_shots_home SMALLINT,
    total_shots_away SMALLINT,
    shots_on_target_home SMALLINT,
    shots_on_target_away SMALLINT,
    xg_home DOUBLE PRECISION,
    xg_away DOUBLE PRECISION,
    corners_home SMALLINT,
    corners_away SMALLINT,
    fouls_home SMALLINT,
    fouls_away SMALLINT,
    big_chances_home SMALLINT,
    big_chances_away SMALLINT,
    big_chances_missed_home SMALLINT,
    big_chances_missed_away SMALLINT,
    accurate_passes_home SMALLINT,
    accurate_passes_away SMALLINT,
    pass_accuracy_home SMALLINT,
    pass_accuracy_away SMALLINT,
    raw_stats JSONB,
    captured_at TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS ix_match_sofascore_stats_captured_at
ON match_sofascore_stats (captured_at);

COMMENT ON TABLE match_sofascore_stats IS 'Post-match Sofascore team statistics (xG, big chances, etc.)';
"""


TABLES = [
    ("sofascore_player_rating_history", SQL_PLAYER_RATING_HISTORY),
    ("match_sofascore_stats", SQL_MATCH_SOFASCORE_STATS),
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

                # Always run all statements (CREATE TABLE/INDEX IF NOT EXISTS is idempotent)
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

    logger.info("\nMigration 032 complete.")


if __name__ == "__main__":
    asyncio.run(main())
