#!/usr/bin/env python3
"""
Populate match_external_refs with Understat links.

CLI wrapper for app.etl.sota_jobs.sync_understat_refs().

Usage:
    # Default: last 30 days, max 200 matches
    DATABASE_URL="postgresql://..." python scripts/populate_match_external_refs_understat.py

    # Custom window
    DATABASE_URL="postgresql://..." python scripts/populate_match_external_refs_understat.py --days 7 --limit 100

Reference: docs/ARCHITECTURE_SOTA.md section 1.2
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


async def main(days: int = 30, limit: int = 200):
    """CLI entry point - wraps sota_jobs.sync_understat_refs."""
    from app.etl.sota_jobs import sync_understat_refs

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
            stats = await sync_understat_refs(session, days=days, limit=limit)

        # Print summary
        logger.info("=" * 60)
        logger.info("UNDERSTAT REFS POPULATION SUMMARY:")
        logger.info("=" * 60)
        logger.info(f"  Scanned:              {stats['scanned']}")
        logger.info(f"  Linked (auto):        {stats['linked_auto']}")
        logger.info(f"  Linked (needs_review): {stats['linked_review']}")
        logger.info(f"  Skipped (no candidates): {stats['skipped_no_candidates']}")
        logger.info(f"  Skipped (low score):  {stats['skipped_low_score']}")
        logger.info(f"  Errors:               {stats['errors']}")
        logger.info("=" * 60)

        return stats

    finally:
        await engine.dispose()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Populate match_external_refs with Understat links"
    )
    parser.add_argument(
        "--days", type=int, default=30, help="Days back to scan (default: 30)"
    )
    parser.add_argument(
        "--limit", type=int, default=200, help="Max matches to process (default: 200)"
    )
    args = parser.parse_args()

    asyncio.run(main(days=args.days, limit=args.limit))
