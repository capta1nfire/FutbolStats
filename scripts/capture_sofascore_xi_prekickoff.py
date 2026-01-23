#!/usr/bin/env python3
"""
Capture Sofascore XI data for upcoming matches.

CLI wrapper for app.etl.sota_jobs.capture_sofascore_xi_prekickoff().

Usage:
    # Default: next 48h, max 100 matches
    DATABASE_URL="postgresql://..." python scripts/capture_sofascore_xi_prekickoff.py

    # Custom window
    DATABASE_URL="postgresql://..." python scripts/capture_sofascore_xi_prekickoff.py --hours 24 --limit 50

Reference: docs/ARCHITECTURE_SOTA.md section 1.3 (match_sofascore_*)
"""

import argparse
import asyncio
import logging
import os
import sys

# Add app to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def main(hours: int = 48, limit: int = 100):
    """CLI entry point - wraps sota_jobs.capture_sofascore_xi_prekickoff."""
    from app.etl.sota_jobs import capture_sofascore_xi_prekickoff

    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        raise ValueError("DATABASE_URL environment variable not set")

    # Convert to async URL
    if database_url.startswith("postgresql://"):
        database_url = database_url.replace("postgresql://", "postgresql+asyncpg://", 1)

    engine = create_async_engine(database_url, echo=False)
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    try:
        async with async_session() as session:
            stats = await capture_sofascore_xi_prekickoff(session, hours=hours, limit=limit)

        # Print summary
        logger.info("=" * 60)
        logger.info("SOFASCORE XI CAPTURE SUMMARY:")
        logger.info("=" * 60)
        logger.info(f"  Matches checked:      {stats['matches_checked']}")
        logger.info(f"  With sofascore ref:   {stats['with_ref']}")
        logger.info(f"  Captured:             {stats['captured']}")
        logger.info(f"  Skipped (no ref):     {stats['skipped_no_ref']}")
        logger.info(f"  Skipped (no data):    {stats['skipped_no_data']}")
        logger.info(f"  Skipped (low integrity): {stats.get('skipped_low_integrity', 0)}")
        logger.info(f"  Errors:               {stats['errors']}")
        logger.info("=" * 60)

        return stats

    finally:
        await engine.dispose()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Capture Sofascore XI data for upcoming matches"
    )
    parser.add_argument(
        "--hours", type=int, default=48, help="Hours ahead to look (default: 48)"
    )
    parser.add_argument(
        "--limit", type=int, default=100, help="Max matches to process (default: 100)"
    )
    args = parser.parse_args()

    asyncio.run(main(hours=args.hours, limit=args.limit))
