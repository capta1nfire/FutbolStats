"""
Migration 021: Add prediction_performance_reports table.

Stores daily performance reports for model evaluation.
Enables distinguishing variance from bugs via proper probability metrics.
"""

import asyncio
import logging

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

from app.config import get_settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

settings = get_settings()


async def migrate():
    """Add prediction_performance_reports table."""
    engine = create_async_engine(settings.DATABASE_URL)
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async with async_session() as session:
        logger.info("Creating prediction_performance_reports table...")
        try:
            await session.execute(text("""
                CREATE TABLE IF NOT EXISTS prediction_performance_reports (
                    id SERIAL PRIMARY KEY,
                    generated_at TIMESTAMP NOT NULL DEFAULT NOW(),
                    window_days INTEGER NOT NULL,
                    report_date DATE NOT NULL,
                    payload JSONB NOT NULL,
                    source VARCHAR(50) NOT NULL DEFAULT 'scheduler',
                    CONSTRAINT uq_report_window_date UNIQUE (window_days, report_date)
                )
            """))
            logger.info("Created prediction_performance_reports table")
        except Exception as e:
            if "already exists" in str(e).lower():
                logger.info("Table already exists, skipping")
            else:
                raise

        # Create indexes
        logger.info("Creating indexes...")
        try:
            await session.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_perf_reports_generated
                ON prediction_performance_reports(generated_at DESC)
            """))
            await session.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_perf_reports_window_date
                ON prediction_performance_reports(window_days, report_date DESC)
            """))
            logger.info("Created indexes")
        except Exception as e:
            logger.warning(f"Index creation note: {e}")

        await session.commit()
        logger.info("Migration 021 completed successfully")

    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(migrate())
