#!/usr/bin/env python3
"""
One-off backfill: predictions.asof_timestamp NULL → PIT-safe derived value.

GDT Mandato 3: Prohibido DELETE. UPDATE PIT-safe derivando asof_timestamp
desde created_at, garantizando asof_timestamp < kickoff_utc SIEMPRE.

Formula:
    asof_timestamp = LEAST(
        p.created_at AT TIME ZONE 'UTC',
        (m.date AT TIME ZONE 'UTC') - INTERVAL '1 minute'
    )

Usage:
    source .env
    python scripts/backfill_predictions_asof_timestamp.py          # dry-run
    python scripts/backfill_predictions_asof_timestamp.py --apply  # apply
"""

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    import asyncpg
except ImportError:
    print("ERROR: asyncpg required. pip install asyncpg")
    sys.exit(1)


async def main():
    apply = "--apply" in sys.argv
    db_url = os.environ.get("DATABASE_URL", "")
    if not db_url:
        print("ERROR: DATABASE_URL not set. Run: source .env")
        return

    # asyncpg needs postgresql:// (not postgres://)
    if db_url.startswith("postgres://"):
        db_url = db_url.replace("postgres://", "postgresql://", 1)
    db_url = db_url.replace("postgresql+asyncpg://", "postgresql://")

    conn = await asyncpg.connect(db_url)

    # Pre-check: list NULL rows
    print("=" * 60)
    print("PRE-CHECK: predictions with asof_timestamp IS NULL")
    print("=" * 60)

    nulls = await conn.fetch("""
        SELECT p.id, p.match_id, p.model_version, p.created_at,
               m.date as kickoff_utc,
               CASE WHEN p.created_at >= m.date THEN 'late_write' ELSE 'ok' END as timing
        FROM predictions p
        JOIN matches m ON m.id = p.match_id
        WHERE p.asof_timestamp IS NULL
        ORDER BY p.created_at
    """)

    print(f"Found {len(nulls)} rows with NULL asof_timestamp")
    for r in nulls:
        print(f"  match_id={r['match_id']} model={r['model_version']} "
              f"created={r['created_at']} kickoff={r['kickoff_utc']} timing={r['timing']}")

    if not nulls:
        print("\nNothing to backfill. All predictions have asof_timestamp.")
        await conn.close()
        return

    late_writes = [r for r in nulls if r['timing'] == 'late_write']
    if late_writes:
        print(f"\n[WARNING] {len(late_writes)} late_write cases (created_at >= kickoff):")
        for r in late_writes:
            print(f"  match_id={r['match_id']} — will be corrected to kickoff-1min")

    if not apply:
        print("\n[DRY-RUN] No changes made. Pass --apply to execute UPDATE.")
        await conn.close()
        return

    # UPDATE: PIT-safe derivation
    print("\nApplying UPDATE...")
    result = await conn.execute("""
        UPDATE predictions p
        SET asof_timestamp = LEAST(
            p.created_at AT TIME ZONE 'UTC',
            (m.date AT TIME ZONE 'UTC') - INTERVAL '1 minute'
        )
        FROM matches m
        WHERE m.id = p.match_id
          AND p.asof_timestamp IS NULL
    """)
    rowcount = int(result.split(" ")[-1]) if result else 0
    print(f"Updated {rowcount} rows")

    # Post-check
    print("\nPOST-CHECK:")
    remaining = await conn.fetchval(
        "SELECT COUNT(*) FROM predictions WHERE asof_timestamp IS NULL"
    )
    violations = await conn.fetchval("""
        SELECT COUNT(*) FROM predictions p
        JOIN matches m ON m.id = p.match_id
        WHERE p.asof_timestamp >= (m.date AT TIME ZONE 'UTC')
    """)

    print(f"  NULL asof_timestamp remaining: {remaining}")
    print(f"  PIT violations (asof >= kickoff): {violations}")

    if remaining == 0 and violations == 0:
        print("\n[OK] Backfill complete. Zero NULLs, zero PIT violations.")
    else:
        print(f"\n[WARNING] Issues remain: {remaining} NULLs, {violations} violations")

    await conn.close()


if __name__ == "__main__":
    asyncio.run(main())
