"""
Migration 019: Add data quality flags for training/backtest exclusion.

Adds:
- tainted, tainted_reason columns to matches table
- Ensures quarantined, tainted columns exist in odds_history

These flags allow excluding contaminated data from ML training and backtesting.
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
    """Add data quality flags to matches and odds_history."""
    engine = create_async_engine(settings.DATABASE_URL)
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async with async_session() as session:
        # 1. Add tainted columns to matches
        logger.info("Adding tainted columns to matches table...")
        try:
            await session.execute(text("""
                ALTER TABLE matches
                ADD COLUMN IF NOT EXISTS tainted BOOLEAN DEFAULT false,
                ADD COLUMN IF NOT EXISTS tainted_reason VARCHAR(100) DEFAULT NULL
            """))
            logger.info("Added tainted columns to matches")
        except Exception as e:
            if "already exists" in str(e).lower():
                logger.info("Columns already exist in matches, skipping")
            else:
                raise

        # 2. Ensure quarantined/tainted columns exist in odds_history
        logger.info("Ensuring data quality columns in odds_history...")
        try:
            await session.execute(text("""
                ALTER TABLE odds_history
                ADD COLUMN IF NOT EXISTS quarantined BOOLEAN DEFAULT false,
                ADD COLUMN IF NOT EXISTS quarantine_reason VARCHAR(100) DEFAULT NULL,
                ADD COLUMN IF NOT EXISTS tainted BOOLEAN DEFAULT false,
                ADD COLUMN IF NOT EXISTS taint_reason VARCHAR(100) DEFAULT NULL
            """))
            logger.info("Ensured data quality columns in odds_history")
        except Exception as e:
            if "already exists" in str(e).lower():
                logger.info("Columns already exist in odds_history, skipping")
            else:
                raise

        # 3. Create indexes for efficient filtering
        logger.info("Creating indexes for data quality flags...")
        try:
            await session.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_matches_tainted ON matches(tainted) WHERE tainted = true
            """))
            await session.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_odds_history_quarantined ON odds_history(quarantined) WHERE quarantined = true
            """))
            logger.info("Created data quality indexes")
        except Exception as e:
            logger.warning(f"Index creation note: {e}")

        await session.commit()
        logger.info("Migration 019 completed successfully")

    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(migrate())
