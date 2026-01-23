#!/usr/bin/env python3
"""
Check Sofascore external refs status for upcoming matches.

CLI wrapper for app.etl.sota_jobs.sync_sofascore_refs().

Usage:
    # Default: next 72h, max 200 matches
    DATABASE_URL="postgresql://..." python scripts/check_sofascore_refs.py

    # Custom window
    DATABASE_URL="postgresql://..." python scripts/check_sofascore_refs.py --hours 24 --limit 100

Note: This script doesn't populate refs (Sofascore lacks public search API).
It only reports how many matches have/need refs for manual population.

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


async def main(hours: int = 72, limit: int = 200):
    """CLI entry point - wraps sota_jobs.sync_sofascore_refs."""
    from app.etl.sota_jobs import sync_sofascore_refs

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
            stats = await sync_sofascore_refs(session, hours=hours, limit=limit)

        # Print summary
        logger.info("=" * 60)
        logger.info("SOFASCORE REFS STATUS SUMMARY:")
        logger.info("=" * 60)
        logger.info(f"  Scanned:              {stats['scanned']}")
        logger.info(f"  With ref:             {stats['with_ref']}")
        logger.info(f"  Missing ref:          {stats['missing_ref']}")
        logger.info(f"  Errors:               {stats['errors']}")
        logger.info("=" * 60)

        if stats['missing_ref'] > 0:
            logger.warning(
                f"  {stats['missing_ref']} matches need sofascore refs populated manually.\n"
                "  See: docs/ARCHITECTURE_SOTA.md for manual population process."
            )

        return stats

    finally:
        await engine.dispose()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Check Sofascore refs status for upcoming matches"
    )
    parser.add_argument(
        "--hours", type=int, default=72, help="Hours ahead to scan (default: 72)"
    )
    parser.add_argument(
        "--limit", type=int, default=200, help="Max matches to check (default: 200)"
    )
    args = parser.parse_args()

    asyncio.run(main(hours=args.hours, limit=args.limit))
