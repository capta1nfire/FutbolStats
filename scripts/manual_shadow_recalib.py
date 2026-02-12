#!/usr/bin/env python3
"""
Manual trigger for shadow_recalibration().

ATI-approved 2026-02-09 (first run = rebaseline, P0-3).
Must run with Python 3.12, from repo root, with .env loaded:

    set -a && source .env && set +a
    python scripts/manual_shadow_recalib.py

This calls shadow_recalibration() directly — same code path as the scheduler.
One-shot: exits after single run.
"""
import asyncio
import logging
import sys
import os

# Ensure project root is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("manual_shadow_recalib")


async def main():
    logger.info("=" * 60)
    logger.info("MANUAL SHADOW RECALIBRATION — one-shot trigger")
    logger.info("=" * 60)

    # Validate DATABASE_URL is set
    db_url = os.environ.get("DATABASE_URL") or os.environ.get("DATABASE_URL_ASYNC")
    if not db_url:
        logger.error("DATABASE_URL not set. Run: set -a && source .env && set +a")
        sys.exit(1)

    logger.info(f"DB host: {db_url.split('@')[1].split('/')[0] if '@' in db_url else 'unknown'}")

    # Import after sys.path setup
    from app.scheduler import shadow_recalibration

    logger.info("Calling shadow_recalibration()...")
    await shadow_recalibration()
    logger.info("Done.")


if __name__ == "__main__":
    asyncio.run(main())
