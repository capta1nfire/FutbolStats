#!/usr/bin/env python3
"""
One-time backfill script for Sensor B missing predictions.

Fills b_* fields for sensor_predictions rows where b_home_prob IS NULL.
Uses proper feature engineering (same as normal prediction flow).

Usage:
    python scripts/backfill_sensor_b.py [--dry-run] [--limit N] [--match-ids 123,456]

Options:
    --dry-run       Show what would be updated without making changes
    --limit N       Max number of matches to process (default: 200)
    --match-ids     Comma-separated list of specific match IDs to process
    --include-ft    Include FT/AET/PEN matches (default: only NS)

AUDIT GUARDRAILS:
- Only updates rows where b_home_prob IS NULL (idempotent)
- Requires sensor.is_ready = True
- Does not affect A predictions (a_* fields preserved)
"""

import argparse
import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


async def main():
    parser = argparse.ArgumentParser(description="Backfill Sensor B missing predictions")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be updated")
    parser.add_argument("--limit", type=int, default=200, help="Max matches to process")
    parser.add_argument("--match-ids", type=str, help="Comma-separated match IDs")
    parser.add_argument("--include-ft", action="store_true", help="Include FT/AET/PEN matches")
    args = parser.parse_args()

    # Parse match IDs if provided
    match_ids = None
    if args.match_ids:
        match_ids = [int(x.strip()) for x in args.match_ids.split(",")]
        print(f"Targeting specific match IDs: {match_ids}")

    from app.database import AsyncSessionLocal
    from app.ml.sensor import (
        get_sensor_engine,
        is_sensor_ready,
        retrain_sensor,
        retry_missing_b_predictions,
    )

    # Ensure Sensor B is initialized in this process.
    # In the API, the scheduler retrain job populates the global sensor engine.
    # In this standalone script, we need to retrain/load it before using it.
    async with AsyncSessionLocal() as session:
        retrain_result = await retrain_sensor(session)
        sensor = get_sensor_engine()

    if sensor is None or not sensor.is_ready:
        print("ERROR: Sensor is not ready. Cannot backfill B predictions.")
        print(f"  retrain_result={retrain_result}")
        print(f"  sensor={sensor}, is_ready={is_sensor_ready()}")
        return 1

    print(f"Sensor is READY: version={sensor.model_version}")
    print(f"Mode: {'DRY-RUN' if args.dry_run else 'LIVE'}")
    print(f"Include FT: {args.include_ft}")
    print(f"Limit: {args.limit}")
    print()

    if args.dry_run:
        # Just show what would be processed
        import pandas as pd
        from sqlalchemy import text
        from app.features.engineering import FeatureEngineer
        from app.models import Match
        async with AsyncSessionLocal() as session:
            feature_engineer = FeatureEngineer(session=session)

            if match_ids:
                placeholders = ",".join([str(m) for m in match_ids])
                query = text(f"""
                    SELECT sp.match_id, m.status, m.date,
                           sp.sensor_state, sp.a_home_prob, sp.b_home_prob
                    FROM sensor_predictions sp
                    JOIN matches m ON m.id = sp.match_id
                    WHERE sp.b_home_prob IS NULL
                      AND sp.match_id IN ({placeholders})
                    ORDER BY m.date DESC
                    LIMIT :limit
                """)
            elif args.include_ft:
                query = text("""
                    SELECT sp.match_id, m.status, m.date,
                           sp.sensor_state, sp.a_home_prob, sp.b_home_prob
                    FROM sensor_predictions sp
                    JOIN matches m ON m.id = sp.match_id
                    WHERE sp.b_home_prob IS NULL
                      AND m.status IN ('NS', 'FT', 'AET', 'PEN')
                    ORDER BY m.date DESC
                    LIMIT :limit
                """)
            else:
                query = text("""
                    SELECT sp.match_id, m.status, m.date,
                           sp.sensor_state, sp.a_home_prob, sp.b_home_prob
                    FROM sensor_predictions sp
                    JOIN matches m ON m.id = sp.match_id
                    WHERE sp.b_home_prob IS NULL
                      AND m.status = 'NS'
                      AND m.date > NOW()
                    ORDER BY m.date DESC
                    LIMIT :limit
                """)

            result = await session.execute(query, {"limit": args.limit})
            rows = result.fetchall()

            print(f"Found {len(rows)} matches with missing B predictions:")
            print("-" * 80)
            for row in rows[:20]:  # Show first 20
                a_home = f"{row.a_home_prob:.3f}" if row.a_home_prob is not None else "NULL"
                print(
                    f"  match_id={row.match_id}, status={row.status}, "
                    f"date={row.date}, state={row.sensor_state}, a_home={a_home}"
                )
            if len(rows) > 20:
                print(f"  ... and {len(rows) - 20} more")
            print()

            # Provide 5 concrete before/after expected examples (no DB writes)
            print("Examples (before → expected after):")
            print("-" * 80)
            examples = rows[:5]
            for row in examples:
                match = await session.get(Match, row.match_id)
                if not match:
                    print(f"  match_id={row.match_id}: missing match record (skipped)")
                    continue

                features = await feature_engineer.get_match_features(match)
                if not features or features.get("has_features") is False:
                    print(f"  match_id={row.match_id}: no features available (skipped)")
                    continue

                df = pd.DataFrame([features])
                missing_cols = set(sensor.FEATURE_COLUMNS) - set(df.columns)
                if missing_cols:
                    print(f"  match_id={row.match_id}: missing columns {sorted(missing_cols)} (skipped)")
                    continue

                b_probs = sensor.predict_proba(df)
                if b_probs is None:
                    print(f"  match_id={row.match_id}: predict_proba=None (skipped)")
                    continue

                b_home, b_draw, b_away = b_probs[0].tolist()
                print(
                    f"  match_id={row.match_id}: b_home NULL → {b_home:.4f}, "
                    f"b_draw NULL → {b_draw:.4f}, b_away NULL → {b_away:.4f}"
                )

            print("Run without --dry-run to apply changes.")
        return 0

    # Live mode: actually update
    async with AsyncSessionLocal() as session:
        result = await retry_missing_b_predictions(
            session,
            include_ft=args.include_ft,
            match_ids=match_ids,
            limit=args.limit,
        )

        print("Backfill result:")
        print(f"  Status: {result.get('status')}")
        print(f"  Mode: {result.get('mode')}")
        print(f"  Checked: {result.get('checked')}")
        print(f"  Updated: {result.get('updated')}")
        print(f"  Skipped (no features): {result.get('skipped_no_features', 0)}")
        print(f"  Errors: {result.get('errors')}")

        if result.get("updated", 0) > 0:
            print()
            print("SUCCESS: B predictions filled for updated matches.")
        elif result.get("checked", 0) == 0:
            print()
            print("No matches needed backfill (all already have B predictions).")
        else:
            print()
            print("WARNING: Checked matches but none updated. Check logs for details.")

    return 0


if __name__ == "__main__":
    exit(asyncio.run(main()))
