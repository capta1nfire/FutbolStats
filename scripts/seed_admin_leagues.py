#!/usr/bin/env python3
"""
Seed admin_leagues table from COMPETITIONS dict.

Usage:
    python scripts/seed_admin_leagues.py [--dry-run]
"""

import asyncio
import os
import sys

# Add parent dir to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

from app.dashboard.admin_seed import full_sync, get_sync_status


async def main():
    dry_run = "--dry-run" in sys.argv

    # Get database URL (required)
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        print("ERROR: DATABASE_URL environment variable is required.")
        print("Set it before running: export DATABASE_URL='postgresql://...'")
        sys.exit(1)

    # Convert to async URL
    if database_url.startswith("postgresql://"):
        database_url = database_url.replace("postgresql://", "postgresql+asyncpg://", 1)

    print(f"Connecting to database...")
    engine = create_async_engine(database_url, echo=False)

    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async with async_session() as session:
        # Check current status
        print("\n=== Current Status ===")
        status = await get_sync_status(session)
        print(f"Total in DB: {status['total_in_db']}")
        print(f"COMPETITIONS dict: {status['competitions_dict_count']}")
        print(f"By source: {status['by_source']}")

        # Run sync
        print(f"\n=== Running {'DRY RUN' if dry_run else 'FULL SYNC'} ===")
        result = await full_sync(session, dry_run=dry_run)

        print(f"\nSeed result:")
        print(f"  - Inserted: {result['seed']['inserted']}")
        print(f"  - Skipped: {result['seed']['skipped']}")
        print(f"  - Groups created: {result['seed']['groups_created']}")

        print(f"\nObserved result:")
        print(f"  - Discovered: {result['observed']['discovered']}")
        print(f"  - Inserted: {result['observed']['inserted']}")

        # Final status
        print("\n=== Final Status ===")
        status = await get_sync_status(session)
        print(f"Total in DB: {status['total_in_db']}")
        print(f"By source: {status['by_source']}")
        print(f"Synced: {status['synced']}")

    await engine.dispose()
    print("\nDone!")


if __name__ == "__main__":
    asyncio.run(main())
