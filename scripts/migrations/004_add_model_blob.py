"""
Migration 004: Add model_blob column to model_snapshots table.

This migration adds the model_blob BYTEA column that stores XGBoost models
as binary data directly in PostgreSQL, enabling fast startup without
filesystem dependency.

Run with:
    DATABASE_URL="postgresql://..." python scripts/migrations/004_add_model_blob.py
"""

import asyncio
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine


async def run_migration():
    """Add model_blob column to model_snapshots table."""
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        print("ERROR: DATABASE_URL environment variable not set")
        sys.exit(1)

    # Ensure async driver
    if database_url.startswith("postgresql://"):
        database_url = database_url.replace("postgresql://", "postgresql+asyncpg://", 1)

    engine = create_async_engine(database_url, echo=True)

    async with engine.begin() as conn:
        # Check if column already exists
        result = await conn.execute(text("""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = 'model_snapshots' AND column_name = 'model_blob'
        """))
        if result.fetchone():
            print("Column 'model_blob' already exists. Skipping migration.")
            return

        # Add model_blob column (BYTEA for binary data)
        print("Adding 'model_blob' column to model_snapshots table...")
        await conn.execute(text("""
            ALTER TABLE model_snapshots
            ADD COLUMN model_blob BYTEA
        """))

        # Make model_path nullable (it's now optional)
        print("Making 'model_path' column nullable...")
        await conn.execute(text("""
            ALTER TABLE model_snapshots
            ALTER COLUMN model_path DROP NOT NULL
        """))

        print("Migration complete!")

    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(run_migration())
