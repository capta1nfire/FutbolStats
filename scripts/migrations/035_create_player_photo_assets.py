"""Migration 035: Player photo assets table.

Stores HQ player photos with immutable R2 keys (content_hash prefix).
Supports global headshots (context_team_id=NULL) and contextual composed cards.

Guardrails:
- Idempotent: IF NOT EXISTS on table, indexes, and unique constraint.
- Partial unique index ensures max 1 active asset per slot (fix #2).
- COALESCE handles NULL != NULL in Postgres (condition C).
"""

import asyncio
import logging
import os

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


SQL_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS player_photo_assets (
    id SERIAL PRIMARY KEY,
    player_external_id INTEGER NOT NULL,
    context_team_id INTEGER,
    season VARCHAR(20),
    role VARCHAR(10),
    kit_variant VARCHAR(10),

    -- Asset dimensions (fix #3)
    asset_type VARCHAR(10) NOT NULL,
    style VARCHAR(20) NOT NULL DEFAULT 'raw',

    -- Storage (fix #1: content_hash for immutable keys)
    r2_key VARCHAR(500) NOT NULL,
    cdn_url VARCHAR(500) NOT NULL,
    content_hash VARCHAR(64) NOT NULL,
    revision INTEGER NOT NULL DEFAULT 1,

    -- Provenance
    source VARCHAR(50) NOT NULL,
    processor VARCHAR(50),
    quality_score INTEGER,
    photo_meta JSONB,

    -- Lifecycle
    review_status VARCHAR(20) NOT NULL DEFAULT 'pending_review',
    is_active BOOLEAN NOT NULL DEFAULT false,
    activated_at TIMESTAMP,
    deactivated_at TIMESTAMP,
    changed_by VARCHAR(20),
    run_id VARCHAR(36),

    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);
"""

SQL_IDX_PLAYER = """
CREATE INDEX IF NOT EXISTS idx_player_photo_assets_ext_id
ON player_photo_assets(player_external_id);
"""

SQL_IDX_ACTIVE = """
CREATE INDEX IF NOT EXISTS idx_player_photo_assets_active
ON player_photo_assets(player_external_id, is_active)
WHERE is_active = true;
"""

SQL_IDX_HASH = """
CREATE INDEX IF NOT EXISTS idx_player_photo_assets_hash
ON player_photo_assets(content_hash);
"""

# Fix #2 + Condition C: Partial unique index with COALESCE for NULL safety.
# Ensures max 1 active asset per (player, team_context, season, role, kit, type, style).
SQL_UNIQUE_ACTIVE_SLOT = """
CREATE UNIQUE INDEX IF NOT EXISTS uq_player_photo_active_slot
ON player_photo_assets (
    player_external_id,
    COALESCE(context_team_id, 0),
    COALESCE(season, ''),
    COALESCE(role, ''),
    COALESCE(kit_variant, ''),
    asset_type,
    style
) WHERE is_active = true;
"""


async def main():
    db_url = os.environ.get("DATABASE_URL_ASYNC") or os.environ.get("DATABASE_URL", "")
    if not db_url:
        raise RuntimeError("DATABASE_URL_ASYNC or DATABASE_URL must be set")

    if db_url.startswith("postgresql://"):
        db_url = db_url.replace("postgresql://", "postgresql+asyncpg://", 1)

    engine = create_async_engine(db_url, echo=False)

    async with engine.begin() as conn:
        logger.info("Creating player_photo_assets table...")
        await conn.execute(text(SQL_CREATE_TABLE))

        logger.info("Creating indexes...")
        await conn.execute(text(SQL_IDX_PLAYER))
        await conn.execute(text(SQL_IDX_ACTIVE))
        await conn.execute(text(SQL_IDX_HASH))

        logger.info("Creating partial unique index (active slot)...")
        await conn.execute(text(SQL_UNIQUE_ACTIVE_SLOT))

        logger.info("Migration 035 complete: player_photo_assets table created.")

    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
