"""Background scheduler for weekly sync, audit, and training jobs."""

import asyncio
import logging
import os
from datetime import datetime, timedelta
from decimal import Decimal

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from sqlalchemy import select, and_, text
from sqlalchemy.orm import selectinload

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

        # Freeze predictions for matches that have started
        freeze_result = await freeze_predictions_before_kickoff()

        _last_live_sync = datetime.utcnow()
        logger.info(f"Global sync complete: {synced} matches updated, {freeze_result.get('frozen_count', 0)} predictions frozen")

        return {
            "matches_synced": synced,
            "predictions_frozen": freeze_result.get("frozen_count", 0),
            "last_sync_at": _last_live_sync,
        }

    except Exception as e:
        logger.error(f"Global sync failed: {e}")
        return {"matches_synced": 0, "error": str(e)}


async def freeze_predictions_before_kickoff() -> dict:
    """
    Freeze predictions for matches that are about to start or have started.

    This preserves the original prediction the user saw BEFORE the match,
    including:
    - The model's probability predictions
    - The bookmaker odds at freeze time
    - The EV calculations at freeze time
    - The confidence tier at freeze time
    - The value bets at freeze time

    A prediction is frozen when:
    - Match status changes from NS to any other status (1H, HT, 2H, FT, etc.)
    - Or match date is in the past (safety net)

    Once frozen, the prediction is immutable even if the model is retrained.
    """
    from datetime import timedelta
    from app.models import Match, Prediction

    frozen_count = 0
    errors = []

    try:
        async with AsyncSessionLocal() as session:
            # Find predictions that need freezing:
            # 1. Prediction is not frozen yet
            # 2. Match status is NOT 'NS' (match has started or finished)
            # OR match date is in the past (safety net)
            now = datetime.utcnow()

            result = await session.execute(
                select(Prediction)
                .options(selectinload(Prediction.match))
                .where(
                    and_(
                        Prediction.is_frozen == False,  # noqa: E712
                    )
                )
            )
            predictions = result.scalars().all()

            for pred in predictions:
                match = pred.match
                if not match:
                    continue

                # Check if match has started or is in the past
                should_freeze = (
                    match.status != "NS" or  # Match has started/finished
                    match.date < now  # Match date is in the past (safety net)
                )

                if not should_freeze:
                    continue

                try:
                    # Freeze the prediction with current data
                    pred.is_frozen = True
                    pred.frozen_at = now

                    # Capture bookmaker odds at freeze time
                    pred.frozen_odds_home = match.odds_home
                    pred.frozen_odds_draw = match.odds_draw
                    pred.frozen_odds_away = match.odds_away

                    # Calculate and freeze EV values
                    if match.odds_home and pred.home_prob > 0:
                        pred.frozen_ev_home = (pred.home_prob * match.odds_home) - 1
                    if match.odds_draw and pred.draw_prob > 0:
                        pred.frozen_ev_draw = (pred.draw_prob * match.odds_draw) - 1
                    if match.odds_away and pred.away_prob > 0:
                        pred.frozen_ev_away = (pred.away_prob * match.odds_away) - 1

                    # Calculate and freeze confidence tier
                    max_prob = max(pred.home_prob, pred.draw_prob, pred.away_prob)
                    if max_prob >= 0.50:
                        pred.frozen_confidence_tier = "gold"
                    elif max_prob >= 0.40:
                        pred.frozen_confidence_tier = "silver"
                    else:
                        pred.frozen_confidence_tier = "copper"

                    # Calculate and freeze value bets
                    value_bets = []
                    ev_threshold = 0.05  # 5% EV minimum for value bet

                    if pred.frozen_ev_home and pred.frozen_ev_home >= ev_threshold:
                        value_bets.append({
                            "outcome": "home",
                            "odds": match.odds_home,
                            "model_prob": pred.home_prob,
                            "ev": pred.frozen_ev_home,
                        })
                    if pred.frozen_ev_draw and pred.frozen_ev_draw >= ev_threshold:
                        value_bets.append({
                            "outcome": "draw",
                            "odds": match.odds_draw,
                            "model_prob": pred.draw_prob,
                            "ev": pred.frozen_ev_draw,
                        })
                    if pred.frozen_ev_away and pred.frozen_ev_away >= ev_threshold:
                        value_bets.append({
                            "outcome": "away",
                            "odds": match.odds_away,
                            "model_prob": pred.away_prob,
                            "ev": pred.frozen_ev_away,
                        })

                    pred.frozen_value_bets = value_bets if value_bets else None

                    frozen_count += 1

                except Exception as e:
                    errors.append(f"Error freezing prediction {pred.id}: {e}")
                    logger.warning(f"Error freezing prediction {pred.id}: {e}")

            await session.commit()

            if frozen_count > 0:
                logger.info(f"Frozen {frozen_count} predictions")

            return {
                "frozen_count": frozen_count,
                "errors": errors[:10] if errors else None,  # Limit error count
            }

    except Exception as e:
        logger.error(f"freeze_predictions_before_kickoff failed: {e}")
        return {"frozen_count": 0, "error": str(e)}


async def monitor_lineups_and_capture_odds() -> dict:
    """
    Monitor upcoming matches for lineup announcements and capture odds snapshots.

    This is the CRITICAL job for the Lineup Arbitrage (Value Betting Táctico) strategy.

    When a lineup is announced (~60min before kickoff):
    1. Fetch lineups from API-Football
    2. If lineup is confirmed and we don't have a snapshot yet:
       - Get current odds (from odds_history or match.odds)
       - Create an odds_snapshot with snapshot_type='lineup_confirmed'
       - Store the exact timestamp for evaluation

    This provides the TRUE baseline for testing if our lineup model beats the
    market odds at the moment of lineup announcement.

    Runs every 5 minutes to catch lineup announcements in time.
    """
    from app.models import Match

    captured_count = 0
    checked_count = 0
    errors = []

    try:
        async with AsyncSessionLocal() as session:
            # Find matches starting in the next 90 minutes that don't have lineup_confirmed snapshot
            now = datetime.utcnow()
            window_start = now - timedelta(minutes=10)  # Include matches that just started
            window_end = now + timedelta(minutes=90)

            # Get matches in the window that:
            # 1. Are in NS (not started) or just started (1H)
            # 2. Don't have a lineup_confirmed odds snapshot yet
            result = await session.execute(text("""
                SELECT m.id, m.external_id, m.date, m.odds_home, m.odds_draw, m.odds_away
                FROM matches m
                WHERE m.date BETWEEN :window_start AND :window_end
                  AND m.status IN ('NS', '1H')
                  AND NOT EXISTS (
                      SELECT 1 FROM odds_snapshots os
                      WHERE os.match_id = m.id
                        AND os.snapshot_type = 'lineup_confirmed'
                  )
                ORDER BY m.date
            """), {"window_start": window_start, "window_end": window_end})

            matches = result.fetchall()
            
            # Limit processing to avoid overload (max 50 matches per run)
            if len(matches) > 50:
                logger.warning(
                    f"Too many matches in window ({len(matches)}), processing first 50 "
                    f"to avoid overload. Remaining will be processed in next run."
                )
                matches = matches[:50]

            if not matches:
                return {"checked": 0, "captured": 0, "message": "No matches in window"}

            # Use API-Football to check lineups
            provider = APIFootballProvider()

            try:
                for match in matches:
                    match_id = match.id
                    external_id = match.external_id
                    checked_count += 1

                    try:
                        # Fetch lineup from API with retry logic
                        lineup_data = None
                        max_retries = 3
                        for attempt in range(max_retries):
                            try:
                                lineup_data = await provider.get_lineups(external_id)
                                break
                            except Exception as e:
                                if attempt == max_retries - 1:
                                    logger.error(
                                        f"Failed to fetch lineup for match {match_id} "
                                        f"(external: {external_id}) after {max_retries} attempts: {e}"
                                    )
                                    raise
                                # Exponential backoff: 2s, 4s, 8s
                                await asyncio.sleep(2 ** attempt)
                                logger.debug(
                                    f"Retry {attempt + 1}/{max_retries} for lineup fetch "
                                    f"(match {match_id})"
                                )

                        if not lineup_data:
                            # Lineup not yet announced
                            continue

                        # Check if we have valid starting XI
                        home_lineup = lineup_data.get("home")
                        away_lineup = lineup_data.get("away")

                        if not home_lineup or not away_lineup:
                            continue

                        home_xi = home_lineup.get("starting_xi", [])
                        away_xi = away_lineup.get("starting_xi", [])

                        # Consider lineup confirmed if we have 11 players for each team
                        if len(home_xi) < 11 or len(away_xi) < 11:
                            continue
                        
                        # Validate match hasn't started (double-check after API call delay)
                        if match.status != 'NS':
                            logger.debug(
                                f"Match {match_id} status changed to {match.status} "
                                f"during processing, skipping"
                            )
                            continue
                        
                        # Validate data quality: check for None player IDs
                        if any(p is None for p in home_xi) or any(p is None for p in away_xi):
                            logger.warning(
                                f"Invalid player IDs (None values) in lineup for match {match_id}. "
                                f"Skipping to avoid data quality issues."
                            )
                            continue

                        # LINEUP CONFIRMED! Now capture the odds
                        # CRITICAL: Fetch FRESH odds directly from API at this exact moment
                        logger.info(f"Lineup confirmed for match {match_id} (external: {external_id})")

                        kickoff_time = match.date
                        lineup_detected_at = datetime.utcnow()

                        # PRIMARY: Get LIVE odds from API-Football (most accurate)
                        # CRITICAL: We MUST have fresh odds for valid baseline measurement
                        # Priority: Bet365 > Pinnacle > 1xBet (sharp bookmakers)
                        fresh_odds = await provider.get_odds(external_id)

                        if fresh_odds and fresh_odds.get("odds_home"):
                            odds_home = float(fresh_odds["odds_home"])
                            odds_draw = float(fresh_odds["odds_draw"])
                            odds_away = float(fresh_odds["odds_away"])
                            bookmaker_name = fresh_odds.get("bookmaker", "unknown")
                            source = f"{bookmaker_name}_live"
                            logger.info(f"Got FRESH odds from {bookmaker_name} for match {match_id}")
                        else:
                            # NO FALLBACK: We cannot use stale odds as baseline
                            # This would invalidate the evaluation metric
                            # The baseline MUST be the market odds at the exact moment of lineup detection
                            logger.error(
                                f"Cannot capture fresh odds for match {match_id} "
                                f"(external: {external_id}) - API returned: {fresh_odds}. "
                                f"Skipping this match. Will retry in next run (5 min)."
                            )
                            continue

                        # Calculate implied probabilities
                        if odds_home > 1 and odds_draw > 1 and odds_away > 1:
                            raw_home = 1 / odds_home
                            raw_draw = 1 / odds_draw
                            raw_away = 1 / odds_away
                            total = raw_home + raw_draw + raw_away
                            overround = total - 1

                            prob_home = raw_home / total
                            prob_draw = raw_draw / total
                            prob_away = raw_away / total
                        else:
                            logger.warning(f"Invalid odds for match {match_id}: {odds_home}, {odds_draw}, {odds_away}")
                            continue

                        # Insert the lineup_confirmed snapshot with timing metadata
                        snapshot_at = datetime.utcnow()

                        # CRITICAL VALIDATION: snapshot must be BEFORE kickoff
                        if kickoff_time and snapshot_at >= kickoff_time:
                            logger.warning(
                                f"Snapshot AFTER kickoff for match {match_id}: "
                                f"snapshot_at={snapshot_at}, kickoff={kickoff_time}. Skipping."
                            )
                            continue

                        # Calculate delta to kickoff (positive = before kickoff)
                        delta_to_kickoff = None
                        if kickoff_time:
                            delta_to_kickoff = int((kickoff_time - snapshot_at).total_seconds())
                            
                            # Validate delta is in expected range (0-120 minutes before kickoff)
                            minutes_to_kickoff = delta_to_kickoff / 60
                            if minutes_to_kickoff < 0:
                                logger.error(
                                    f"Negative delta for match {match_id}: {minutes_to_kickoff:.1f} min. "
                                    f"This should not happen after validation above."
                                )
                                continue
                            elif minutes_to_kickoff > 120:
                                logger.warning(
                                    f"Delta very large for match {match_id}: {minutes_to_kickoff:.1f} min. "
                                    f"Lineup detected very early - may be incorrect."
                                )
                                # Don't skip, but log warning for monitoring

                        # Determine odds freshness based on source
                        if "_live" in source:
                            odds_freshness = "live"
                        elif "_stale" in source:
                            odds_freshness = "stale"
                        else:
                            odds_freshness = "unknown"

                        await session.execute(text("""
                            INSERT INTO odds_snapshots (
                                match_id, snapshot_type, snapshot_at,
                                odds_home, odds_draw, odds_away,
                                prob_home, prob_draw, prob_away,
                                overround, bookmaker,
                                kickoff_time, delta_to_kickoff_seconds, odds_freshness
                            ) VALUES (
                                :match_id, 'lineup_confirmed', :snapshot_at,
                                :odds_home, :odds_draw, :odds_away,
                                :prob_home, :prob_draw, :prob_away,
                                :overround, :bookmaker,
                                :kickoff_time, :delta_to_kickoff, :odds_freshness
                            )
                            ON CONFLICT (match_id, snapshot_type, bookmaker) DO NOTHING
                        """), {
                            "match_id": match_id,
                            "snapshot_at": snapshot_at,
                            "odds_home": odds_home,
                            "odds_draw": odds_draw,
                            "odds_away": odds_away,
                            "prob_home": prob_home,
                            "prob_draw": prob_draw,
                            "prob_away": prob_away,
                            "overround": overround,
                            "bookmaker": source,
                            "kickoff_time": kickoff_time,
                            "delta_to_kickoff": delta_to_kickoff,
                            "odds_freshness": odds_freshness,
                        })

                        # Also update match_lineups with lineup_confirmed_at if not already set
                        await session.execute(text("""
                            UPDATE match_lineups
                            SET lineup_confirmed_at = COALESCE(lineup_confirmed_at, :confirmed_at)
                            WHERE match_id = :match_id
                        """), {"match_id": match_id, "confirmed_at": snapshot_at})

                        captured_count += 1
                        logger.info(
                            f"Captured lineup_confirmed odds for match {match_id}: "
                            f"H={odds_home:.2f}, D={odds_draw:.2f}, A={odds_away:.2f} at {snapshot_at}"
                        )

                    except Exception as e:
                        errors.append(f"Match {match_id}: {str(e)}")
                        logger.warning(f"Error checking lineup for match {match_id}: {e}")

                await session.commit()

            finally:
                await provider.close()

            if captured_count > 0:
                logger.info(f"Lineup monitoring: captured {captured_count} lineup_confirmed snapshots")

            return {
                "checked": checked_count,
                "captured": captured_count,
                "errors": errors[:5] if errors else None,
            }

    except Exception as e:
        logger.error(f"monitor_lineups_and_capture_odds failed: {e}")
        return {"checked": 0, "captured": 0, "error": str(e)}


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

    # Lineup Monitoring: Every 5 minutes
    # Checks for lineup announcements and captures odds at lineup_confirmed time
    # CRITICAL for Value Betting Táctico (Lineup Arbitrage) evaluation
    scheduler.add_job(
        monitor_lineups_and_capture_odds,
        trigger=IntervalTrigger(minutes=5),
        id="lineup_monitoring",
        name="Lineup Monitoring (every 5 min)",
        replace_existing=True,
    )

    scheduler.start()
    _scheduler_started = True
    logger.info(
        "Scheduler started:\n"
        "  - Live global sync: Every 60 seconds\n"
        "  - Lineup monitoring: Every 5 minutes\n"
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
