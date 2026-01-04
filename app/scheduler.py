"""Background scheduler for weekly sync, audit, and training jobs."""

import logging
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from app.database import AsyncSessionLocal
from app.etl.pipeline import create_etl_pipeline
from app.features.engineering import FeatureEngineer

logger = logging.getLogger(__name__)

# Top 5 European leagues
SYNC_LEAGUES = [39, 140, 135, 78, 61]  # EPL, La Liga, Serie A, Bundesliga, Ligue 1
CURRENT_SEASON = 2025

scheduler = AsyncIOScheduler()


async def daily_save_predictions():
    """
    Daily job to save predictions for upcoming matches.
    Runs every day at 7:00 AM UTC (before audit).
    """
    logger.info("Starting daily prediction save job...")

    try:
        from sqlalchemy.dialects.postgresql import insert as pg_insert

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

            # Save to database
            saved = 0
            for pred in predictions:
                match_id = pred.get("match_id")
                if not match_id:
                    continue

                probs = pred["probabilities"]
                stmt = pg_insert(Prediction).values(
                    match_id=match_id,
                    model_version=engine.model_version,
                    home_prob=probs["home"],
                    draw_prob=probs["draw"],
                    away_prob=probs["away"],
                ).on_conflict_do_update(
                    constraint="uq_match_model",
                    set_={
                        "home_prob": probs["home"],
                        "draw_prob": probs["draw"],
                        "away_prob": probs["away"],
                    }
                )
                try:
                    await session.execute(stmt)
                    saved += 1
                except Exception as e:
                    logger.warning(f"Error saving prediction: {e}")

            await session.commit()
            logger.info(f"Daily prediction save complete: {saved} predictions saved")

    except Exception as e:
        logger.error(f"Daily prediction save failed: {e}")


async def daily_audit():
    """
    Daily job to audit completed matches from the last 3 days.
    Runs every day at 8:00 AM UTC.
    """
    logger.info("Starting daily audit job...")

    try:
        from app.audit import create_audit_service

        async with AsyncSessionLocal() as session:
            audit_service = await create_audit_service(session)
            result = await audit_service.audit_recent_matches(days=3)
            await audit_service.close()

            logger.info(
                f"Daily audit complete: {result['matches_audited']} matches, "
                f"{result['accuracy']:.1f}% accuracy, {result['anomalies_detected']} anomalies"
            )

    except Exception as e:
        logger.error(f"Daily audit failed: {e}")


async def weekly_sync_and_train(ml_engine):
    """
    Weekly job to sync latest results and retrain the model.
    Runs every Monday at 6:00 AM UTC.
    """
    logger.info("Starting weekly sync and train job...")

    try:
        async with AsyncSessionLocal() as session:
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
            from app.audit import create_audit_service

            audit_service = await create_audit_service(session)
            audit_result = await audit_service.audit_recent_matches(days=7)
            await audit_service.close()

            logger.info(
                f"Audit complete: {audit_result['matches_audited']} matches audited, "
                f"{audit_result['accuracy']:.1f}% accuracy"
            )

            # Step 3: Retrain the model
            logger.info("Retraining ML model...")
            feature_engineer = FeatureEngineer(session=session)
            df = await feature_engineer.build_training_dataset()

            if len(df) < 100:
                logger.error(f"Insufficient training data: {len(df)} samples")
                return

            ml_engine.train(df)
            logger.info(f"Training complete: {ml_engine.model_version} with {len(df)} samples")

    except Exception as e:
        logger.error(f"Weekly sync and train failed: {e}")


def start_scheduler(ml_engine):
    """Start the background scheduler."""
    # Daily prediction save: Every day at 7:00 AM UTC (before audit)
    scheduler.add_job(
        daily_save_predictions,
        trigger=CronTrigger(hour=7, minute=0),
        id="daily_save_predictions",
        name="Daily Save Predictions",
        replace_existing=True,
    )

    # Daily audit job: Every day at 8:00 AM UTC
    scheduler.add_job(
        daily_audit,
        trigger=CronTrigger(hour=8, minute=0),
        id="daily_audit",
        name="Daily Post-Match Audit",
        replace_existing=True,
    )

    # Weekly sync + audit + train job: Monday at 6:00 AM UTC
    scheduler.add_job(
        weekly_sync_and_train,
        trigger=CronTrigger(day_of_week="mon", hour=6, minute=0),
        args=[ml_engine],
        id="weekly_sync_train",
        name="Weekly Sync, Audit and Train",
        replace_existing=True,
    )

    scheduler.start()
    logger.info(
        "Scheduler started:\n"
        "  - Daily save predictions: 7:00 AM UTC\n"
        "  - Daily audit: 8:00 AM UTC\n"
        "  - Weekly sync/audit/train: Mondays 6:00 AM UTC"
    )


def stop_scheduler():
    """Stop the background scheduler."""
    if scheduler.running:
        scheduler.shutdown()
        logger.info("Scheduler stopped")
