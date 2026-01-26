#!/usr/bin/env python3
"""
Run admin_leagues migration on production database.

Usage:
    DATABASE_URL='postgresql://...' python scripts/run_admin_migration.py

Requires DATABASE_URL environment variable.
"""

import asyncio
import os
import sys

# Add parent dir to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine


async def run_migration():
    """Execute the admin_001_create_admin_leagues.sql migration."""

    # Get database URL (required)
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        print("ERROR: DATABASE_URL environment variable is required.")
        print("Set it before running: export DATABASE_URL='postgresql://...'")
        import sys
        sys.exit(1)

    # Convert to async URL if needed
    if database_url.startswith("postgresql://"):
        database_url = database_url.replace("postgresql://", "postgresql+asyncpg://", 1)

    print(f"Connecting to database...")
    engine = create_async_engine(database_url, echo=False)

    # Read migration file
    migration_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "migrations",
        "admin_001_create_admin_leagues.sql"
    )

    with open(migration_path, "r") as f:
        migration_sql = f.read()

    # Split by semicolons and filter comments/empty
    statements = []
    current_stmt = []
    in_function = False

    for line in migration_sql.split("\n"):
        stripped = line.strip()

        # Track function blocks (they contain semicolons)
        if "CREATE OR REPLACE FUNCTION" in line or "CREATE FUNCTION" in line:
            in_function = True

        current_stmt.append(line)

        # End of statement
        if stripped.endswith(";") and not in_function:
            stmt = "\n".join(current_stmt).strip()
            if stmt and not stmt.startswith("--"):
                statements.append(stmt)
            current_stmt = []
        elif in_function and stripped == "$$ LANGUAGE plpgsql;":
            stmt = "\n".join(current_stmt).strip()
            statements.append(stmt)
            current_stmt = []
            in_function = False

    # Execute each statement
    async with engine.begin() as conn:
        for i, stmt in enumerate(statements):
            if not stmt.strip() or stmt.strip().startswith("--"):
                continue
            try:
                print(f"[{i+1}/{len(statements)}] Executing: {stmt[:60]}...")
                await conn.execute(text(stmt))
                print(f"    OK")
            except Exception as e:
                if "already exists" in str(e).lower():
                    print(f"    SKIP (already exists)")
                else:
                    print(f"    ERROR: {e}")
                    raise

    print("\nMigration completed successfully!")

    # Verify tables exist
    async with engine.begin() as conn:
        result = await conn.execute(text("""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_name IN ('admin_leagues', 'admin_league_groups')
            ORDER BY table_name
        """))
        tables = [r[0] for r in result.fetchall()]
        print(f"Created tables: {tables}")

    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(run_migration())
