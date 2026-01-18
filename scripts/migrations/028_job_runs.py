"""Migration 028: Job runs tracking table.

Creates job_runs table for tracking scheduler job executions.
Used as fallback for jobs_health in ops.json when Prometheus
metrics are unavailable (cold-start after deploy).

Example: After deploy, ops.json shows last_success_at from DB
instead of "unknown" while Prometheus warms up.
"""

import asyncio
import logging

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


MIGRATION_SQL = """
-- Create job_runs table for tracking scheduler job executions
CREATE TABLE IF NOT EXISTS job_runs (
    id SERIAL PRIMARY KEY,
    job_name VARCHAR(100) NOT NULL,
    status VARCHAR(20) NOT NULL DEFAULT 'ok',  -- ok, error, rate_limited, budget_exceeded
    started_at TIMESTAMP NOT NULL DEFAULT NOW(),
    finished_at TIMESTAMP,
    duration_ms INTEGER,
    error_message TEXT,
    metrics JSONB,  -- Optional job-specific metrics (e.g., rows_updated, fixtures_scanned)
    created_at TIMESTAMP NOT NULL DEFAULT NOW()
);

-- Index for efficient lookup of last success per job
CREATE INDEX IF NOT EXISTS ix_job_runs_job_status_started
ON job_runs (job_name, status, started_at DESC);

-- Index for cleanup queries (prune old runs)
CREATE INDEX IF NOT EXISTS ix_job_runs_created_at
ON job_runs (created_at);

-- Partial index for fast "last success" queries
CREATE INDEX IF NOT EXISTS ix_job_runs_last_success
ON job_runs (job_name, finished_at DESC)
WHERE status = 'ok';

-- Comment for documentation
COMMENT ON TABLE job_runs IS 'Tracks scheduler job executions for ops dashboard fallback (P1-B)';
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

    async with engine.begin() as conn:
        # Check if table exists
        result = await conn.execute(text("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables
                WHERE table_name = 'job_runs'
            )
        """))
        table_exists = result.scalar()

        if table_exists:
            logger.info("Table job_runs already exists, skipping DDL...")
        else:
            logger.info("Creating job_runs table...")
            for statement in MIGRATION_SQL.strip().split(";"):
                statement = statement.strip()
                if statement and not statement.startswith("--"):
                    logger.info(f"Executing: {statement[:60]}...")
                    await conn.execute(text(statement))
            logger.info("Table and indexes created.")

        # Verify
        result = await conn.execute(text("SELECT COUNT(*) FROM job_runs"))
        count = result.scalar() or 0
        logger.info(f"Migration complete. {count} run(s) in table.")


if __name__ == "__main__":
    asyncio.run(main())
