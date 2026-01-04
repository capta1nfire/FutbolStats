"""Background scheduler for weekly sync and training jobs."""

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

            # Step 2: Retrain the model
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
    # Weekly job: Monday at 6:00 AM UTC
    scheduler.add_job(
        weekly_sync_and_train,
        trigger=CronTrigger(day_of_week="mon", hour=6, minute=0),
        args=[ml_engine],
        id="weekly_sync_train",
        name="Weekly Sync and Train",
        replace_existing=True,
    )

    scheduler.start()
    logger.info("Scheduler started - Weekly sync/train job scheduled for Mondays at 6:00 AM UTC")


def stop_scheduler():
    """Stop the background scheduler."""
    if scheduler.running:
        scheduler.shutdown()
        logger.info("Scheduler stopped")
