"""
Migration 020: Add venue fields to matches table.

Adds:
- venue_name: Stadium name from API-Football
- venue_city: Stadium city from API-Football

These fields allow showing stadium information in match details.
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
    """Add venue fields to matches table."""
    engine = create_async_engine(settings.DATABASE_URL)
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async with async_session() as session:
        logger.info("Adding venue columns to matches table...")
        try:
            await session.execute(text("""
                ALTER TABLE matches
                ADD COLUMN IF NOT EXISTS venue_name VARCHAR(255) DEFAULT NULL,
                ADD COLUMN IF NOT EXISTS venue_city VARCHAR(100) DEFAULT NULL
            """))
            logger.info("Added venue_name and venue_city columns to matches")
        except Exception as e:
            if "already exists" in str(e).lower():
                logger.info("Columns already exist in matches, skipping")
            else:
                raise

        await session.commit()
        logger.info("Migration 020 completed successfully")

    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(migrate())
