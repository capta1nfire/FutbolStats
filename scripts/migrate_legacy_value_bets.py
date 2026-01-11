#!/usr/bin/env python3
"""
One-off migration script to normalize legacy frozen_value_bets.

Legacy format: {outcome, odds, model_prob, ev}
New format: {outcome, our_probability, implied_probability, edge, edge_percentage,
             expected_value, ev_percentage, market_odds, fair_odds, is_value_bet}

Usage:
    python scripts/migrate_legacy_value_bets.py [--dry-run]
"""

import asyncio
import json
import sys
from datetime import datetime

# Add app to path
sys.path.insert(0, ".")

from sqlalchemy import select, update
from app.database import get_async_session
from app.models import Prediction


def is_legacy_format(vb: dict) -> bool:
    """Check if value_bet is in legacy format."""
    # Legacy has 'ev' but not 'expected_value'
    return "ev" in vb and "expected_value" not in vb


def normalize_value_bet(vb: dict) -> dict:
    """Convert legacy value_bet to normalized format."""
    if not is_legacy_format(vb):
        return vb  # Already normalized

    outcome = vb.get("outcome", "unknown")
    odds = vb.get("odds", 0)
    model_prob = vb.get("model_prob", 0)
    ev = vb.get("ev", 0)

    # Calculate derived fields
    implied_prob = 1 / odds if odds > 0 else 0
    edge = model_prob - implied_prob if model_prob and implied_prob else 0
    fair_odds = 1 / model_prob if model_prob > 0 else None

    return {
        "outcome": outcome,
        "our_probability": round(model_prob, 4) if model_prob else None,
        "implied_probability": round(implied_prob, 4) if implied_prob else None,
        "edge": round(edge, 4) if edge else None,
        "edge_percentage": round(edge * 100, 1) if edge else None,
        "expected_value": round(ev, 4) if ev else None,
        "ev_percentage": round(ev * 100, 1) if ev else None,
        "market_odds": float(odds) if odds else None,
        "fair_odds": round(fair_odds, 2) if fair_odds else None,
        "is_value_bet": True,
    }


async def migrate_legacy_value_bets(dry_run: bool = True):
    """Migrate all legacy frozen_value_bets to normalized format."""
    print(f"{'[DRY RUN] ' if dry_run else ''}Starting legacy value_bets migration...")
    print(f"Timestamp: {datetime.utcnow().isoformat()}Z")
    print("-" * 60)

    migrated_count = 0
    skipped_count = 0
    errors = []

    async for session in get_async_session():
        # Find all predictions with frozen_value_bets
        stmt = select(Prediction).where(Prediction.frozen_value_bets.isnot(None))
        result = await session.execute(stmt)
        predictions = result.scalars().all()

        print(f"Found {len(predictions)} predictions with frozen_value_bets")

        for pred in predictions:
            try:
                value_bets = pred.frozen_value_bets
                if not value_bets:
                    continue

                # Check if any are legacy format
                has_legacy = any(is_legacy_format(vb) for vb in value_bets)
                if not has_legacy:
                    skipped_count += 1
                    continue

                # Normalize all value_bets
                normalized = [normalize_value_bet(vb) for vb in value_bets]

                print(f"\nPrediction {pred.id} (match_id={pred.match_id}):")
                print(f"  Before: {json.dumps(value_bets, indent=4)}")
                print(f"  After:  {json.dumps(normalized, indent=4)}")

                if not dry_run:
                    pred.frozen_value_bets = normalized

                migrated_count += 1

            except Exception as e:
                errors.append(f"Prediction {pred.id}: {e}")
                print(f"  ERROR: {e}")

        if not dry_run:
            await session.commit()
            print("\nChanges committed to database.")

    print("\n" + "=" * 60)
    print("MIGRATION SUMMARY")
    print("=" * 60)
    print(f"Migrated: {migrated_count}")
    print(f"Skipped (already normalized): {skipped_count}")
    print(f"Errors: {len(errors)}")
    if errors:
        for err in errors[:10]:
            print(f"  - {err}")

    return {
        "migrated": migrated_count,
        "skipped": skipped_count,
        "errors": errors,
        "dry_run": dry_run,
    }


if __name__ == "__main__":
    dry_run = "--dry-run" in sys.argv or "-n" in sys.argv

    if not dry_run:
        print("=" * 60)
        print("WARNING: This will modify the database!")
        print("Run with --dry-run to preview changes first.")
        print("=" * 60)
        response = input("Continue? (y/N): ")
        if response.lower() != "y":
            print("Aborted.")
            sys.exit(0)

    result = asyncio.run(migrate_legacy_value_bets(dry_run=dry_run))

    if dry_run and result["migrated"] > 0:
        print("\nTo apply changes, run without --dry-run flag.")
