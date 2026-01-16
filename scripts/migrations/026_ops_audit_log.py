"""Migration 026: OPS audit log table.

Creates the ops_audit_log table for tracking manual dashboard actions
(predictions trigger, odds sync, sync window, etc.) with actor identification
and request context.
"""

import asyncio
import logging

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


MIGRATION_SQL = """
-- Create ops_audit_log table for dashboard action tracking
CREATE TABLE IF NOT EXISTS ops_audit_log (
    id SERIAL PRIMARY KEY,

    -- Action identification
    action VARCHAR(100) NOT NULL,
    request_id VARCHAR(36) NOT NULL,

    -- Actor identification
    actor VARCHAR(100) NOT NULL,
    actor_id VARCHAR(32) NOT NULL,

    -- Request context
    ip_address VARCHAR(45),
    user_agent VARCHAR(500),

    -- Action parameters and result
    params JSONB,
    result VARCHAR(20) NOT NULL,
    result_detail JSONB,
    error_message VARCHAR(500),

    -- Timestamps
    created_at TIMESTAMP DEFAULT NOW(),
    duration_ms INT
);

-- Create indexes for common queries
CREATE INDEX IF NOT EXISTS idx_ops_audit_log_created_at ON ops_audit_log(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_ops_audit_log_action ON ops_audit_log(action);
CREATE INDEX IF NOT EXISTS idx_ops_audit_log_actor_id ON ops_audit_log(actor_id);
"""


async def run_migration(database_url: str):
    """Run the migration."""
    # Convert sync URL to async if needed
    if database_url.startswith("postgresql://"):
        database_url = database_url.replace("postgresql://", "postgresql+asyncpg://")

    engine = create_async_engine(database_url)

    async with engine.begin() as conn:
        logger.info("Running migration 026: OPS audit log table")
        await conn.execute(text(MIGRATION_SQL))
        logger.info("Migration 026 completed successfully")

    await engine.dispose()


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv

    load_dotenv()

    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise ValueError("DATABASE_URL environment variable not set")

    asyncio.run(run_migration(database_url))
