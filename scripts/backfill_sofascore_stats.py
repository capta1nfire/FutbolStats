#!/usr/bin/env python3
"""
Backfill historical Sofascore post-match statistics for FT matches with refs.

Fetches xG, big chances, possession etc. for all historical matches.

Usage:
    source .env && export DATABASE_URL && python3 scripts/backfill_sofascore_stats.py

Estimated time: ~1,142 matches × 1 req/s ≈ 19 minutes.
"""

import asyncio
import logging
import os
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


async def main():
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        raise ValueError("DATABASE_URL environment variable not set")

    if database_url.startswith("postgresql://"):
        database_url = database_url.replace("postgresql://", "postgresql+asyncpg://", 1)

    from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
    from sqlalchemy.orm import sessionmaker

    engine = create_async_engine(database_url, echo=False, pool_size=5)
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    from app.etl.sota_jobs import backfill_sofascore_stats_ft

    total_metrics = {
        "scanned": 0,
        "inserted": 0,
        "skipped_no_data": 0,
        "skipped_empty_stats": 0,
        "errors": 0,
    }

    batch = 0
    while True:
        batch += 1
        logger.info(f"--- Batch {batch} ---")

        async with async_session() as session:
            metrics = await backfill_sofascore_stats_ft(
                session, days=180, limit=50
            )

        for k in total_metrics:
            total_metrics[k] += metrics.get(k, 0)

        scanned = metrics.get("scanned", 0)
        inserted = metrics.get("inserted", 0)
        logger.info(
            f"Batch {batch}: scanned={scanned}, inserted={inserted}, "
            f"errors={metrics.get('errors', 0)}"
        )

        if scanned == 0:
            logger.info("No more matches to process.")
            break

    logger.info("=" * 60)
    logger.info("BACKFILL COMPLETE")
    logger.info(f"  Total scanned: {total_metrics['scanned']}")
    logger.info(f"  Total inserted: {total_metrics['inserted']}")
    logger.info(f"  Total errors: {total_metrics['errors']}")

    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
