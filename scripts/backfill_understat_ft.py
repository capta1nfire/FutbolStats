#!/usr/bin/env python3
"""
Backfill Understat xG/xPTS data for finished matches.

CLI wrapper for app.etl.sota_jobs.backfill_understat_ft().

Usage:
    # Default: last 14 days
    DATABASE_URL="postgresql://..." python scripts/backfill_understat_ft.py

    # Custom window
    DATABASE_URL="postgresql://..." python scripts/backfill_understat_ft.py --days 30

    # With ref only (faster, skips matches without understat ref)
    DATABASE_URL="postgresql://..." python scripts/backfill_understat_ft.py --with-ref-only

Reference: docs/ARCHITECTURE_SOTA.md section 1.3
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


async def main(
    days: int = 14,
    limit: int = 100,
    with_ref_only: bool = True,
):
    """CLI entry point - wraps sota_jobs.backfill_understat_ft."""
    from app.etl.sota_jobs import backfill_understat_ft

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
            stats = await backfill_understat_ft(
                session,
                days=days,
                limit=limit,
                with_ref_only=with_ref_only,
            )

        # Print summary
        logger.info("=" * 60)
        logger.info("BACKFILL SUMMARY:")
        logger.info("=" * 60)
        logger.info(f"  Scanned:          {stats['scanned']}")
        logger.info(f"  Inserted:         {stats['inserted']}")
        logger.info(f"  Updated:          {stats['updated']}")
        logger.info(f"  Skipped (no ref): {stats['skipped_no_ref']}")
        logger.info(f"  Skipped (no data): {stats['skipped_no_data']}")
        logger.info(f"  Errors:           {stats['errors']}")
        logger.info("=" * 60)

        return stats

    finally:
        await engine.dispose()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backfill Understat xG data")
    parser.add_argument("--days", type=int, default=14, help="Days back to scan (default: 14)")
    parser.add_argument("--limit", type=int, default=100, help="Limit number of matches (default: 100)")
    parser.add_argument(
        "--with-ref-only", action="store_true", default=True,
        help="Only process matches with understat refs (default: True)"
    )
    args = parser.parse_args()

    asyncio.run(main(
        days=args.days,
        limit=args.limit,
        with_ref_only=args.with_ref_only,
    ))
