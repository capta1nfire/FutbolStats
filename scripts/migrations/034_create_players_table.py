"""Migration 034: Players catalog table.

Stores player information from API-Football squads endpoint.
Used for XI continuity scoring (cross-reference with match_lineups.starting_xi_ids).

Guardrails:
- Idempotent: safe to run multiple times (IF NOT EXISTS).
- external_id UNIQUE: upsert-safe via ON CONFLICT.
- team_id FK to teams(id): resolved during sync.
"""

import asyncio
import logging
import os

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


SQL_CREATE_PLAYERS = """
CREATE TABLE IF NOT EXISTS players (
    id SERIAL PRIMARY KEY,
    external_id INTEGER NOT NULL UNIQUE,
    name VARCHAR(200) NOT NULL,
    position VARCHAR(20),
    team_id INTEGER REFERENCES teams(id),
    team_external_id INTEGER,
    jersey_number INTEGER,
    age INTEGER,
    photo_url VARCHAR(500),
    last_synced_at TIMESTAMP DEFAULT NOW(),
    created_at TIMESTAMP DEFAULT NOW()
);
"""

SQL_IDX_TEAM = "CREATE INDEX IF NOT EXISTS idx_players_team ON players(team_id);"
SQL_IDX_EXT = "CREATE INDEX IF NOT EXISTS idx_players_ext ON players(external_id);"


async def main():
    db_url = os.environ.get("DATABASE_URL_ASYNC") or os.environ.get("DATABASE_URL", "")
    if not db_url:
        raise RuntimeError("DATABASE_URL_ASYNC or DATABASE_URL must be set")

    # Ensure async driver
    if db_url.startswith("postgresql://"):
        db_url = db_url.replace("postgresql://", "postgresql+asyncpg://", 1)

    engine = create_async_engine(db_url, echo=False)

    async with engine.begin() as conn:
        logger.info("Creating players table...")
        await conn.execute(text(SQL_CREATE_PLAYERS))
        logger.info("Creating indexes...")
        await conn.execute(text(SQL_IDX_TEAM))
        await conn.execute(text(SQL_IDX_EXT))
        logger.info("Migration 034 complete: players table created.")

    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
