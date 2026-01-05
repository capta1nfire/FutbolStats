"""Background scheduler for weekly sync, audit, and training jobs."""

import logging
import os
from datetime import datetime

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from app.database import AsyncSessionLocal
from app.etl.pipeline import create_etl_pipeline, ETLPipeline
from app.etl.api_football import APIFootballProvider
from app.features.engineering import FeatureEngineer

logger = logging.getLogger(__name__)

# Top 5 European leagues
SYNC_LEAGUES = [39, 140, 135, 78, 61]  # EPL, La Liga, Serie A, Bundesliga, Ligue 1
CURRENT_SEASON = 2025

# Flag to prevent multiple scheduler instances (e.g., with --reload)
_scheduler_started = False
scheduler = AsyncIOScheduler()

from typing import Optional

# Global state for live sync tracking
_last_live_sync: Optional[datetime] = None


def get_last_sync_time() -> Optional[datetime]:
    """Get the last live sync timestamp (for API endpoint)."""
    return _last_live_sync


async def global_sync_today() -> dict:
    """
    Global Sync: 1 API call for ALL fixtures worldwide for today.
    Filters to our 5 leagues in memory and updates the DB.

    Uses: GET /fixtures?date=YYYY-MM-DD (1 single API call)
    Budget: 1 call/min × 60 min × 24 hrs = 1,440 calls/day (of 7,500 available)
    """
    global _last_live_sync

    today = datetime.utcnow()

    try:
        async with AsyncSessionLocal() as session:
            provider = APIFootballProvider()

            # 1 SINGLE API CALL - all fixtures worldwide, filtered to our leagues
            our_fixtures = await provider.get_fixtures_by_date(
                date=today,
                league_ids=SYNC_LEAGUES  # Filter in memory
            )

            # Upsert to DB
            pipeline = ETLPipeline(provider, session)
            synced = 0
            for fixture in our_fixtures:
                try:
                    await pipeline._upsert_match(fixture)
                    synced += 1
                except Exception as e:
                    logger.warning(f"Error upserting fixture: {e}")

            await session.commit()
            await provider.close()

        _last_live_sync = datetime.utcnow()
        logger.info(f"Global sync complete: {synced} matches updated")

        return {"matches_synced": synced, "last_sync_at": _last_live_sync}

    except Exception as e:
        logger.error(f"Global sync failed: {e}")
        return {"matches_synced": 0, "error": str(e)}


async def daily_save_predictions():
    """
    Daily job to save predictions for upcoming matches.
    Runs every day at 7:00 AM UTC (before audit).
    """
    logger.info("Starting daily prediction save job...")

    try:
        from app.db_utils import upsert
        from app.ml import XGBoostEngine
        from app.models import Prediction

        async with AsyncSessionLocal() as session:
            # Load ML engine
            engine = XGBoostEngine()
            if not engine.load_model():
                logger.error("Could not load ML model for prediction save")
                return

            # Get upcoming matches features
            feature_engineer = FeatureEngineer(session=session)
            df = await feature_engineer.get_upcoming_matches_features()

            if len(df) == 0:
                logger.info("No upcoming matches to save predictions for")
                return

            # Make predictions
            predictions = engine.predict(df)

            # Save to database using generic upsert
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
                    logger.warning(f"Error saving prediction: {e}")

            await session.commit()
            logger.info(f"Daily prediction save complete: {saved} predictions saved")

    except Exception as e:
        logger.error(f"Daily prediction save failed: {e}")


async def daily_sync_results():
    """
    Daily job to sync match results from API.
    Runs every day at 6:00 AM UTC (before predictions and audit).

    This ensures we have the latest results for:
    - Yesterday's completed matches
    - Matches that finished after the last sync
    """
    logger.info("Starting daily results sync job...")

    try:
        async with AsyncSessionLocal() as session:
            pipeline = await create_etl_pipeline(session)
            result = await pipeline.sync_multiple_leagues(
                league_ids=SYNC_LEAGUES,
                season=CURRENT_SEASON,
                fetch_odds=False,  # Only sync results, not odds
            )

            logger.info(
                f"Daily sync complete: {result['total_matches_synced']} matches synced"
            )

    except Exception as e:
        logger.error(f"Daily sync failed: {e}")


async def daily_audit():
    """
    Daily job to audit completed matches and update team adjustments.
    Runs every day at 8:00 AM UTC.

    Hybrid Daily Adjustment:
    - Audits completed matches from the last 3 days
    - IMMEDIATELY triggers calculate_team_adjustments() after audit
    - Updates recovery counters for teams with consecutive MINIMAL audits
    - Applies international commitment penalties for upcoming matches

    This ensures a Tuesday anomaly affects Friday predictions.
    """
    logger.info("Starting daily audit job...")

    try:
        from app.audit import create_audit_service
        from app.ml.recalibration import RecalibrationEngine

        async with AsyncSessionLocal() as session:
            # Step 1: Run the audit
            audit_service = await create_audit_service(session)
            result = await audit_service.audit_recent_matches(days=3)
            await audit_service.close()

            logger.info(
                f"Daily audit complete: {result['matches_audited']} matches, "
                f"{result['accuracy']:.1f}% accuracy, {result['anomalies_detected']} anomalies"
            )

            # Step 2: IMMEDIATELY update team adjustments (Hybrid Daily Adjustment)
            if result['matches_audited'] > 0:
                logger.info("Triggering immediate team adjustments recalculation...")
                recalibrator = RecalibrationEngine(session)

                # Calculate new adjustments based on recent performance
                adj_result = await recalibrator.calculate_team_adjustments(days=14)
                logger.info(
                    f"Team adjustments updated: {adj_result['teams_analyzed']} analyzed, "
                    f"{adj_result['adjustments_made']} adjusted, "
                    f"{adj_result.get('recoveries_applied', 0)} recoveries"
                )

                # Step 3: Update recovery counters (El Perdón)
                recovery_result = await recalibrator.update_recovery_counters()
                logger.info(
                    f"Recovery counters updated: {recovery_result['teams_updated']} teams, "
                    f"{recovery_result['forgiveness_applied']} forgiveness applied"
                )

                # Step 4: Apply international commitment penalties
                intl_result = await recalibrator.apply_international_penalties(days_ahead=7)
                logger.info(
                    f"International penalties applied: {intl_result['teams_checked']} teams, "
                    f"{intl_result['penalties_applied']} penalties"
                )

                # Step 5: Detect league drift (Fase 2.2)
                drift_result = await recalibrator.detect_league_drift()
                if drift_result['unstable_leagues'] > 0:
                    logger.warning(
                        f"LEAGUE DRIFT DETECTED: {drift_result['unstable_leagues']} unstable leagues"
                    )
                    for alert in drift_result['drift_alerts']:
                        logger.warning(f"  - League {alert['league_id']}: {alert['insight']}")
                else:
                    logger.info(f"League drift check: All {drift_result['leagues_analyzed']} leagues stable")

                # Step 6: Check market movements (Fase 2.2)
                odds_result = await recalibrator.check_all_upcoming_odds_movements(days_ahead=3)
                if odds_result['movements_detected'] > 0:
                    logger.warning(
                        f"ODDS MOVEMENT DETECTED: {odds_result['movements_detected']} matches with significant movement"
                    )
                    for alert in odds_result['alerts']:
                        logger.warning(f"  - Match {alert['match_id']}: {alert['insight']}")
                else:
                    logger.info(f"Odds movement check: {odds_result['matches_checked']} matches stable")

    except Exception as e:
        logger.error(f"Daily audit failed: {e}")


async def weekly_recalibration(ml_engine):
    """
    Weekly intelligent recalibration job.
    Runs every Monday at 6:00 AM UTC.

    Steps:
    1. Sync latest fixtures/results
    2. Run audit on recent matches
    3. Update team confidence adjustments
    4. Evaluate if retraining is needed
    5. If retraining: validate new model before deploying
    """
    logger.info("Starting weekly recalibration job...")

    try:
        from app.audit import create_audit_service
        from app.ml.recalibration import RecalibrationEngine

        async with AsyncSessionLocal() as session:
            recalibrator = RecalibrationEngine(session)

            # Step 1: Sync latest fixtures/results
            logger.info(f"Syncing leagues: {SYNC_LEAGUES}")
            pipeline = await create_etl_pipeline(session)
            sync_result = await pipeline.sync_multiple_leagues(
                league_ids=SYNC_LEAGUES,
                season=CURRENT_SEASON,
                fetch_odds=True,
            )
            logger.info(f"Sync complete: {sync_result['total_matches_synced']} matches")

            # Step 2: Run audit on all unaudited matches
            logger.info("Running post-sync audit...")
            audit_service = await create_audit_service(session)
            audit_result = await audit_service.audit_recent_matches(days=7)
            await audit_service.close()

            logger.info(
                f"Audit complete: {audit_result['matches_audited']} matches audited, "
                f"{audit_result['accuracy']:.1f}% accuracy"
            )

            # Step 3: Update team confidence adjustments
            logger.info("Calculating team adjustments...")
            adj_result = await recalibrator.calculate_team_adjustments(days=30)
            logger.info(
                f"Team adjustments updated: {adj_result['teams_analyzed']} analyzed, "
                f"{adj_result['adjustments_made']} adjusted"
            )

            # Step 4: Evaluate if retraining is needed
            should_retrain, reason = await recalibrator.should_trigger_retrain(days=7)
            logger.info(f"Retrain evaluation: {reason}")

            if not should_retrain:
                logger.info("Skipping retrain - metrics within thresholds")
                return

            # Step 5: Retrain the model
            logger.info(f"Triggering retrain: {reason}")
            feature_engineer = FeatureEngineer(session=session)
            df = await feature_engineer.build_training_dataset()

            if len(df) < 100:
                logger.error(f"Insufficient training data: {len(df)} samples")
                return

            # Train in executor to avoid blocking the event loop
            import asyncio
            from concurrent.futures import ThreadPoolExecutor

            loop = asyncio.get_event_loop()
            with ThreadPoolExecutor() as executor:
                train_result = await loop.run_in_executor(executor, ml_engine.train, df)

            new_brier = train_result["brier_score"]
            logger.info(f"Training complete: Brier Score = {new_brier:.4f}")

            # Step 6: Validate new model against baseline
            is_valid, validation_msg = await recalibrator.validate_new_model(new_brier)
            logger.info(f"Validation result: {validation_msg}")

            if not is_valid:
                logger.warning(f"ROLLBACK: New model rejected - keeping previous version")
                # Reload the previous model
                ml_engine.load_model()
                return

            # Step 7: Create snapshot and activate new model
            snapshot = await recalibrator.create_snapshot(
                model_version=ml_engine.model_version,
                model_path=train_result["model_path"],
                brier_score=new_brier,
                cv_scores=train_result["cv_scores"],
                samples_trained=train_result["samples_trained"],
                training_config=None,  # Could add hyperparams here
            )
            logger.info(f"New model deployed: {snapshot.model_version} (Brier: {new_brier:.4f})")

    except Exception as e:
        logger.error(f"Weekly recalibration failed: {e}")


def start_scheduler(ml_engine):
    """
    Start the background scheduler.

    Uses a module-level flag to prevent duplicate scheduler instances
    when running with --reload or multiple workers.
    """
    global _scheduler_started

    # Prevent duplicate schedulers
    if _scheduler_started:
        logger.warning("Scheduler already started, skipping duplicate initialization")
        return

    # Check if running in reload mode - skip scheduler in child process
    # Uvicorn sets this env var in the reloader subprocess
    if os.environ.get("UVICORN_RELOADED"):
        logger.info("Skipping scheduler in reload subprocess")
        return

    # Daily results sync: Every day at 6:00 AM UTC (first job of the day)
    scheduler.add_job(
        daily_sync_results,
        trigger=CronTrigger(hour=6, minute=0),
        id="daily_sync_results",
        name="Daily Results Sync",
        replace_existing=True,
    )

    # Daily prediction save: Every day at 7:00 AM UTC (after sync)
    scheduler.add_job(
        daily_save_predictions,
        trigger=CronTrigger(hour=7, minute=0),
        id="daily_save_predictions",
        name="Daily Save Predictions",
        replace_existing=True,
    )

    # Daily audit job: Every day at 8:00 AM UTC (after results are synced)
    scheduler.add_job(
        daily_audit,
        trigger=CronTrigger(hour=8, minute=0),
        id="daily_audit",
        name="Daily Post-Match Audit",
        replace_existing=True,
    )

    # Weekly recalibration job: Monday at 5:00 AM UTC (before daily sync)
    scheduler.add_job(
        weekly_recalibration,
        trigger=CronTrigger(day_of_week="mon", hour=5, minute=0),
        args=[ml_engine],
        id="weekly_recalibration",
        name="Weekly Recalibration",
        replace_existing=True,
    )

    # Live Global Sync: Every 60 seconds (real-time results)
    # Uses 1 API call per minute = 1,440 calls/day (of 7,500 available)
    scheduler.add_job(
        global_sync_today,
        trigger=IntervalTrigger(seconds=60),
        id="live_global_sync",
        name="Live Global Sync (every minute)",
        replace_existing=True,
    )

    scheduler.start()
    _scheduler_started = True
    logger.info(
        "Scheduler started:\n"
        "  - Live global sync: Every 60 seconds\n"
        "  - Daily results sync: 6:00 AM UTC\n"
        "  - Daily save predictions: 7:00 AM UTC\n"
        "  - Daily audit: 8:00 AM UTC\n"
        "  - Weekly recalibration: Mondays 5:00 AM UTC"
    )


def stop_scheduler():
    """Stop the background scheduler."""
    global _scheduler_started
    if scheduler.running:
        scheduler.shutdown()
        _scheduler_started = False
        logger.info("Scheduler stopped")
