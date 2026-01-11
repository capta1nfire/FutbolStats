#!/usr/bin/env python3
"""
Backfill Predictions for upcoming matches.

Usage:
    DATABASE_URL=<url> python3 scripts/backfill_predictions.py --days 7 --sync
    DATABASE_URL=<url> python3 scripts/backfill_predictions.py --days 3 --dry-run

This script:
1. Optionally syncs fixtures from API for EXTENDED_LEAGUES
2. Loads ML model and generates predictions for upcoming NS matches
3. Saves predictions to database

Options:
    --days N       Include matches from last N days + upcoming (default: 3)
    --sync         Sync fixtures from API before predicting (requires RAPIDAPI_KEY)
    --dry-run      Show what would be done without saving
    --league-ids   Comma-separated list of league IDs to process (default: EXTENDED_LEAGUES)
"""

import argparse
import asyncio
import logging
import os
import sys
from datetime import datetime

# Ensure app module is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def run_backfill(
    days: int = 3,
    sync_first: bool = False,
    dry_run: bool = False,
    league_ids: list[int] = None,
) -> dict:
    """Run the prediction backfill process."""
    from app.database import AsyncSessionLocal
    from app.db_utils import upsert
    from app.features.engineering import FeatureEngineer
    from app.ml import XGBoostEngine
    from app.models import Prediction
    from app.scheduler import EXTENDED_LEAGUES, CURRENT_SEASON

    if league_ids is None:
        league_ids = EXTENDED_LEAGUES

    stats = {
        "matches_found": 0,
        "predictions_saved": 0,
        "predictions_skipped": 0,
        "sync_matches": 0,
        "errors": [],
    }

    async with AsyncSessionLocal() as session:
        # Step 1: Optionally sync fixtures
        if sync_first:
            logger.info(f"Syncing fixtures for {len(league_ids)} leagues...")
            try:
                from app.etl.pipeline import create_etl_pipeline
                pipeline = await create_etl_pipeline(session)
                sync_result = await pipeline.sync_multiple_leagues(
                    league_ids=league_ids,
                    season=CURRENT_SEASON,
                    fetch_odds=False,
                )
                stats["sync_matches"] = sync_result.get("total_matches_synced", 0)
                logger.info(f"Synced {stats['sync_matches']} matches")
            except Exception as e:
                logger.error(f"Sync failed: {e}")
                stats["errors"].append(f"Sync failed: {e}")

        # Step 2: Load ML model
        engine = XGBoostEngine()
        if not engine.load_model():
            logger.error("Could not load ML model")
            return {"error": "ML model not loaded", **stats}

        # Step 3: Get features for upcoming + recent matches
        logger.info(f"Building features for matches (include_recent_days={days})...")
        feature_engineer = FeatureEngineer(session=session)
        df = await feature_engineer.get_upcoming_matches_features(
            league_ids=league_ids,
            include_recent_days=days,
        )

        if len(df) == 0:
            logger.info("No matches found for prediction")
            return stats

        stats["matches_found"] = len(df)
        logger.info(f"Found {len(df)} matches")

        # Filter to only NS (not started) matches for prediction
        df_ns = df[df["status"] == "NS"].copy()
        logger.info(f"Filtered to {len(df_ns)} NS (upcoming) matches")

        if len(df_ns) == 0:
            logger.info("No upcoming NS matches to predict")
            return stats

        # Step 4: Generate predictions
        predictions = engine.predict(df_ns)
        logger.info(f"Generated {len(predictions)} predictions")

        # Print preview
        print(f"\n{'='*70}")
        print(f"Predictions to save: {len(predictions)}")
        print(f"{'='*70}")
        for pred in predictions[:10]:
            match_id = pred.get("match_id")
            probs = pred["probabilities"]
            print(f"  {match_id}: H={probs['home']:.2f} D={probs['draw']:.2f} A={probs['away']:.2f}")
        if len(predictions) > 10:
            print(f"  ... and {len(predictions) - 10} more")

        if dry_run:
            logger.info("DRY RUN - not saving")
            return {**stats, "dry_run": True}

        # Step 5: Save predictions
        saved = 0
        for pred in predictions:
            match_id = pred.get("match_id")
            if not match_id:
                continue

            probs = pred["probabilities"]
            try:
                await upsert(
                    session,
                    Prediction,
                    values={
                        "match_id": match_id,
                        "model_version": engine.model_version,
                        "home_prob": probs["home"],
                        "draw_prob": probs["draw"],
                        "away_prob": probs["away"],
                    },
                    conflict_columns=["match_id", "model_version"],
                    update_columns=["home_prob", "draw_prob", "away_prob"],
                )
                saved += 1
            except Exception as e:
                logger.warning(f"Error saving prediction for match {match_id}: {e}")
                stats["errors"].append(f"Match {match_id}: {e}")

        await session.commit()
        stats["predictions_saved"] = saved
        logger.info(f"Saved {saved} predictions")

    return stats


async def main():
    parser = argparse.ArgumentParser(description="Backfill predictions for upcoming matches")
    parser.add_argument("--days", type=int, default=3, help="Include matches from last N days")
    parser.add_argument("--sync", action="store_true", help="Sync fixtures from API first")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done")
    parser.add_argument("--league-ids", type=str, help="Comma-separated league IDs")
    args = parser.parse_args()

    # Check environment
    if not os.environ.get("DATABASE_URL"):
        print("ERROR: DATABASE_URL environment variable required")
        sys.exit(1)

    if args.sync and not os.environ.get("RAPIDAPI_KEY"):
        print("ERROR: RAPIDAPI_KEY required for --sync")
        sys.exit(1)

    # Parse league IDs
    league_ids = None
    if args.league_ids:
        league_ids = [int(x.strip()) for x in args.league_ids.split(",")]
        print(f"Using custom league IDs: {league_ids}")

    # Run backfill
    stats = await run_backfill(
        days=args.days,
        sync_first=args.sync,
        dry_run=args.dry_run,
        league_ids=league_ids,
    )

    # Print summary
    print(f"\n{'='*70}")
    print("BACKFILL SUMMARY")
    print(f"{'='*70}")
    print(f"  Matches found:       {stats.get('matches_found', 0)}")
    print(f"  Predictions saved:   {stats.get('predictions_saved', 0)}")
    if stats.get("sync_matches"):
        print(f"  Matches synced:      {stats['sync_matches']}")
    if stats.get("dry_run"):
        print("  (DRY RUN - no changes made)")
    if stats.get("errors"):
        print(f"  Errors: {len(stats['errors'])}")
        for err in stats["errors"][:5]:
            print(f"    - {err}")
    print(f"{'='*70}")

    sys.exit(0 if not stats.get("errors") else 1)


if __name__ == "__main__":
    asyncio.run(main())
