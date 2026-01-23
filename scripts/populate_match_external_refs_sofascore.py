#!/usr/bin/env python3
"""
CLI wrapper for Sofascore refs sync (match_external_refs population).

Usage:
    # Sync refs for matches in next 72h
    DATABASE_URL="postgresql://..." python3 scripts/populate_match_external_refs_sofascore.py

    # Custom time window
    DATABASE_URL="..." python3 scripts/populate_match_external_refs_sofascore.py --hours 48 --days-back 1

    # With mock data (testing)
    DATABASE_URL="..." python3 scripts/populate_match_external_refs_sofascore.py --mock

Reference:
    - app/etl/sota_jobs.py: sync_sofascore_refs()
    - docs/ARCHITECTURE_SOTA.md
"""

import argparse
import asyncio
import logging
import os
import sys

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


async def main():
    parser = argparse.ArgumentParser(
        description="Sync Sofascore refs to match_external_refs table"
    )
    parser.add_argument(
        "--hours",
        type=int,
        default=72,
        help="Hours ahead to scan for NS matches (default: 72)",
    )
    parser.add_argument(
        "--days-back",
        type=int,
        default=2,
        help="Days back to also scan (default: 2)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=200,
        help="Max matches to process (default: 200)",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mock data for testing",
    )

    args = parser.parse_args()

    # Validate DATABASE_URL
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        logger.error("DATABASE_URL environment variable not set")
        sys.exit(1)

    # Convert to async URL if needed
    if database_url.startswith("postgresql://"):
        database_url = database_url.replace("postgresql://", "postgresql+asyncpg://", 1)

    # Import after path setup
    from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
    from sqlalchemy.orm import sessionmaker

    from app.etl.sota_jobs import sync_sofascore_refs

    engine = create_async_engine(database_url, echo=False)
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async with async_session() as session:
        result = await sync_sofascore_refs(
            session=session,
            hours=args.hours,
            days_back=args.days_back,
            limit=args.limit,
            use_mock=args.mock,
        )

    await engine.dispose()

    # Print summary
    logger.info("=" * 60)
    logger.info("SOFASCORE REFS SYNC SUMMARY:")
    logger.info("=" * 60)
    logger.info(f"  Scanned:              {result['scanned']}")
    logger.info(f"  Already linked:       {result['already_linked']}")
    logger.info(f"  Linked (auto):        {result['linked_auto']}")
    logger.info(f"  Linked (needs review):{result['linked_review']}")
    logger.info(f"  Skipped (no cand.):   {result['skipped_no_candidates']}")
    logger.info(f"  Skipped (low score):  {result['skipped_low_score']}")
    logger.info(f"  Errors:               {result['errors']}")
    logger.info("=" * 60)

    total_linked = result["linked_auto"] + result["linked_review"]
    if total_linked > 0:
        logger.info(f"  Successfully linked {total_linked} matches to Sofascore.")
    elif result["scanned"] > 0:
        logger.warning("  No matches were linked. Check logs for details.")


if __name__ == "__main__":
    asyncio.run(main())
