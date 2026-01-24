"""Background scheduler for weekly sync, audit, and training jobs."""

import asyncio
import json
import logging
import os
import time
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Optional

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from sqlalchemy import select, and_, text
from sqlalchemy.orm import selectinload

from app.database import AsyncSessionLocal, get_session_with_retry
from app.etl.pipeline import create_etl_pipeline, ETLPipeline

# NOTE: This import is intentionally defensive. In case of a partial deploy/version skew,
# we prefer the service to start without the budget guardrail rather than crash-loop.
try:
    from app.etl.api_football import APIFootballProvider, APIBudgetExceeded, get_api_budget_status
except ImportError:  # pragma: no cover
    from app.etl.api_football import APIFootballProvider  # type: ignore

    APIBudgetExceeded = Exception  # type: ignore

    def get_api_budget_status() -> dict:  # type: ignore
        return {"status": "unavailable"}
from app.features.engineering import FeatureEngineer

# Telemetry imports for shadow/sensor instrumentation
from app.telemetry.metrics import (
    record_shadow_predictions_batch,
    record_shadow_evaluation_batch,
    set_shadow_health_metrics,
    record_sensor_predictions_batch,
    record_sensor_evaluation_batch,
    record_sensor_retrain,
    set_sensor_health_metrics,
    # Job health metrics (P0 jobs instrumentation)
    record_job_run,
    record_stats_backfill_result,
    record_fastpath_tick,
)

# Sentry context for job error tracking
from app.telemetry.sentry import sentry_job_context, capture_exception as sentry_capture_exception

logger = logging.getLogger(__name__)

# Top 5 European leagues (core)
TOP5_LEAGUES = [39, 140, 135, 78, 61]  # EPL, La Liga, Serie A, Bundesliga, Ligue 1

# Extended leagues for lineup monitoring (more volume)
EXTENDED_LEAGUES = [
    39, 140, 135, 78, 61,  # Top 5
    40,   # England - EFL Championship (2nd division, high ML volume)
    94,   # Portugal - Primeira Liga
    88,   # Netherlands - Eredivisie
    203,  # Turkey - Super Lig
    71,   # Brazil - Serie A
    262,  # Mexico - Liga MX
    128,  # Argentina - Primera División
    253,  # USA - MLS
    144,  # Belgium - Pro League
    307,  # Saudi Arabia - Pro League
    2,    # Champions League
    3,    # Europa League
    848,  # Conference League
    45,   # FA Cup
    143,  # Copa del Rey
    13,   # CONMEBOL Libertadores
    11,   # CONMEBOL Sudamericana
    # LATAM Pack2 - extend monitoring coverage
    239,  # Colombia Primera A
    242,  # Ecuador Liga Pro
    250,  # Paraguay Division Profesional - Apertura
    252,  # Paraguay Division Profesional - Clausura
    265,  # Chile Primera Division
    268,  # Uruguay Primera - Apertura
    270,  # Uruguay Primera - Clausura
    281,  # Peru Primera Division
    299,  # Venezuela Primera Division
    344,  # Bolivia Primera Division
    # WC 2026 Qualifiers (fixtures/teams ingested) - enable extended live monitoring
    34,   # World Cup - Qualification South America (CONMEBOL)
    30,   # World Cup - Qualification Asia (AFC)
    31,   # World Cup - Qualification CONCACAF
    33,   # World Cup - Qualification Oceania (OFC)
    32,   # World Cup - Qualification Europe (UEFA)
    29,   # World Cup - Qualification Africa (CAF) (season may be pending)
    37,   # World Cup - Qualification Intercontinental Play-offs (fixtures may be pending)
]

# League scopes (dynamic via LEAGUE_MODE env var):
# - tracked (default): ALL leagues present in DB (supports "todas las ligas")
# - extended: EXTENDED_LEAGUES
# - top5: TOP5_LEAGUES
#
# SYNC_LEAGUES now respects LEAGUE_MODE for daily sync jobs.
# Default: EXTENDED_LEAGUES (broader coverage for predictions/audits)
def get_sync_leagues() -> list[int]:
    """Get leagues for daily sync based on LEAGUE_MODE env var."""
    mode = os.environ.get("LEAGUE_MODE", "extended").strip().lower()
    if mode == "top5":
        return TOP5_LEAGUES
    # extended or tracked both use EXTENDED_LEAGUES for sync
    # (tracked uses all DB leagues for live sync, but EXTENDED for daily batch)
    return EXTENDED_LEAGUES


# Legacy constant for backwards compatibility
SYNC_LEAGUES = EXTENDED_LEAGUES
CURRENT_SEASON = 2025

# Flag to prevent multiple scheduler instances (e.g., with --reload)
_scheduler_started = False
scheduler = AsyncIOScheduler()

# Global state for live sync tracking
_last_live_sync: Optional[datetime] = None

# Global state for fast-path metrics (for ops dashboard)
_fastpath_metrics: dict = {
    "last_tick_at": None,
    "last_tick_result": None,
    "ticks_total": 0,
    "ticks_with_activity": 0,
}


def get_fastpath_metrics() -> dict:
    """Get current fast-path metrics for ops dashboard."""
    return _fastpath_metrics.copy()


# Global state: tracked leagues cache (to support "all leagues" without hardcoding)
_tracked_leagues_cache: Optional[list[int]] = None
_tracked_leagues_cache_at: Optional[datetime] = None
_TRACKED_LEAGUES_TTL_SECONDS = int(os.environ.get("TRACKED_LEAGUES_TTL_SECONDS", "21600"))  # 6h


def get_last_sync_time() -> Optional[datetime]:
    """Get the last live sync timestamp (for API endpoint)."""
    return _last_live_sync


async def get_tracked_leagues(session) -> list[int]:
    """
    Return league_ids we consider "tracked" (all leagues present in DB),
    cached for TTL to avoid DB churn.
    """
    global _tracked_leagues_cache, _tracked_leagues_cache_at

    now = datetime.utcnow()
    if _tracked_leagues_cache and _tracked_leagues_cache_at:
        if (now - _tracked_leagues_cache_at).total_seconds() < _TRACKED_LEAGUES_TTL_SECONDS:
            return _tracked_leagues_cache

    result = await session.execute(text("SELECT DISTINCT league_id FROM matches"))
    leagues = sorted([row[0] for row in result.fetchall() if row[0] is not None])
    _tracked_leagues_cache = leagues
    _tracked_leagues_cache_at = now
    return leagues


def _league_mode() -> str:
    """
    Scheduler league mode:
    - tracked: all leagues present in DB (default)
    - extended: EXTENDED_LEAGUES list
    - top5: TOP5_LEAGUES list
    """
    return os.environ.get("LEAGUE_MODE", "tracked").strip().lower()


async def resolve_live_sync_leagues(session) -> Optional[list[int]]:
    mode = _league_mode()
    if mode == "top5":
        return TOP5_LEAGUES
    if mode == "extended":
        return EXTENDED_LEAGUES
    # tracked (default): all leagues in DB
    return await get_tracked_leagues(session)


async def resolve_lineup_monitoring_leagues(session) -> Optional[list[int]]:
    mode = _league_mode()
    if mode == "top5":
        return TOP5_LEAGUES
    if mode == "extended":
        return EXTENDED_LEAGUES
    return await get_tracked_leagues(session)


async def global_sync_today() -> dict:
    """
    Global Sync: 2 API calls for fixtures today + yesterday (for late matches crossing midnight).
    Filters to our leagues in memory and updates the DB.

    Uses: GET /fixtures?date=YYYY-MM-DD (2 API calls: today + yesterday)
    Budget: 2 calls/min × 60 min × 24 hrs = 2,880 calls/day (of 7,500 available)

    Why yesterday? Late matches (e.g., LATAM kickoff at 23:20 UTC) may still be live
    after midnight UTC. Without yesterday sync, live scores won't update.
    """
    global _last_live_sync

    today = datetime.utcnow()
    yesterday = today - timedelta(days=1)

    try:
        # Use retry context manager to handle stale connections (Railway can drop idle connections)
        async with get_session_with_retry(max_retries=3, retry_delay=1.0) as session:
            provider = APIFootballProvider()

            league_ids = await resolve_live_sync_leagues(session)

            # 2 API CALLS: today + yesterday (for late matches crossing midnight)
            our_fixtures = []

            # Today's fixtures
            today_fixtures = await provider.get_fixtures_by_date(
                date=today,
                league_ids=league_ids
            )
            our_fixtures.extend(today_fixtures)

            # Yesterday's fixtures (catch late matches still live after midnight UTC)
            yesterday_fixtures = await provider.get_fixtures_by_date(
                date=yesterday,
                league_ids=league_ids
            )
            our_fixtures.extend(yesterday_fixtures)

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

        # Ensure predictions exist for imminent matches (hard guardrail)
        kickoff_result = await ensure_kickoff_predictions()
        if kickoff_result.get("generated", 0) > 0:
            logger.warning(f"[KICKOFF-SAFETY] Generated {kickoff_result['generated']} last-minute predictions")

        # Freeze predictions for matches that have started
        freeze_result = await freeze_predictions_before_kickoff()

        _last_live_sync = datetime.utcnow()
        logger.info(f"Global sync complete: {synced} matches updated, {freeze_result.get('frozen_count', 0)} predictions frozen")

        return {
            "matches_synced": synced,
            "predictions_frozen": freeze_result.get("frozen_count", 0),
            "last_sync_at": _last_live_sync,
        }

    except APIBudgetExceeded as e:
        logger.warning(f"Global sync stopped: {e}. Budget status: {get_api_budget_status()}")
        return {"matches_synced": 0, "error": str(e), "budget": get_api_budget_status()}
    except Exception as e:
        logger.error(f"Global sync failed: {e}")
        return {"matches_synced": 0, "error": str(e)}


async def global_sync_window(days_ahead: int = 10, days_back: int = 1) -> dict:
    """
    Window Sync: Load fixtures for a date range (default: yesterday to 10 days ahead).

    This job ensures fixtures for upcoming matches are loaded in advance,
    regardless of season. Works for both European (season=2025) and LATAM
    (season=2026) leagues by using date-based queries instead of season.

    IMPORTANT: This solves the LATAM 2026 fixture loading issue where
    CURRENT_SEASON=2025 was preventing new season fixtures from loading.

    Uses: GET /fixtures?date=YYYY-MM-DD (1 API call per day in window)
    Budget: ~11 calls per run (10 days ahead + 1 back) = minimal impact

    Schedule: Daily at 05:30 UTC (before other sync jobs)
    """
    try:
        async with AsyncSessionLocal() as session:
            provider = APIFootballProvider()
            league_ids = await resolve_live_sync_leagues(session)

            if not league_ids:
                logger.warning("[WINDOW_SYNC] No leagues resolved, skipping")
                return {"matches_synced": 0, "days_processed": 0}

            pipeline = ETLPipeline(provider, session)
            now = datetime.utcnow()
            total_synced = 0
            days_processed = 0
            by_date = {}

            # Process each day in the window
            for day_offset in range(-days_back, days_ahead + 1):
                target_date = now + timedelta(days=day_offset)

                try:
                    fixtures = await provider.get_fixtures_by_date(
                        date=target_date,
                        league_ids=league_ids
                    )

                    day_synced = 0
                    for fixture in fixtures:
                        try:
                            await pipeline._upsert_match(fixture)
                            day_synced += 1
                        except Exception as e:
                            logger.warning(f"[WINDOW_SYNC] Error upserting fixture: {e}")

                    total_synced += day_synced
                    days_processed += 1
                    date_str = target_date.strftime("%Y-%m-%d")
                    by_date[date_str] = day_synced

                    if day_synced > 0:
                        logger.debug(f"[WINDOW_SYNC] {date_str}: {day_synced} fixtures")

                except APIBudgetExceeded as e:
                    logger.warning(f"[WINDOW_SYNC] Budget exceeded at day {day_offset}: {e}")
                    break
                except Exception as e:
                    logger.warning(f"[WINDOW_SYNC] Error processing day {day_offset}: {e}")
                    continue

            await session.commit()
            await provider.close()

            logger.info(
                f"[WINDOW_SYNC] Complete: {total_synced} fixtures synced, "
                f"{days_processed} days processed ({-days_back} to +{days_ahead})"
            )

            return {
                "matches_synced": total_synced,
                "days_processed": days_processed,
                "window": {"days_back": days_back, "days_ahead": days_ahead},
                "by_date": by_date,
            }

    except Exception as e:
        logger.error(f"[WINDOW_SYNC] Failed: {e}")
        return {"matches_synced": 0, "error": str(e)}


async def ensure_kickoff_predictions() -> dict:
    """
    Hard guardrail: Generate predictions for matches about to kick off without one.

    This is the last line of defense - if a match is starting in the next 30 minutes
    and has no prediction in DB, generate and save one immediately.

    This ensures no match can start without a persisted prediction.
    """
    from sqlalchemy import text
    from app.telemetry.metrics import record_job_run

    start_time = time.time()
    generated = 0
    errors = []

    try:
        from app.ml import XGBoostEngine
        from app.models import Prediction

        async with AsyncSessionLocal() as session:
            # Find NS matches in next 30 minutes WITHOUT any prediction
            result = await session.execute(
                text("""
                    SELECT m.id, m.external_id, m.date, m.league_id,
                           ht.name as home_team, at.name as away_team
                    FROM matches m
                    JOIN teams ht ON ht.id = m.home_team_id
                    JOIN teams at ON at.id = m.away_team_id
                    WHERE m.status = 'NS'
                      AND m.date > NOW()
                      AND m.date <= NOW() + INTERVAL '30 minutes'
                      AND NOT EXISTS (
                          SELECT 1 FROM predictions p WHERE p.match_id = m.id
                      )
                    ORDER BY m.date ASC
                """)
            )
            imminent_gaps = result.fetchall()

            if not imminent_gaps:
                return {"generated": 0, "message": "no_imminent_gaps"}

            logger.warning(
                f"[KICKOFF-SAFETY] Found {len(imminent_gaps)} matches starting in <30min WITHOUT prediction! "
                f"Generating now..."
            )

            # Load ML engine
            engine = XGBoostEngine()
            if not engine.load_model():
                logger.error("[KICKOFF-SAFETY] Could not load ML model")
                return {"generated": 0, "error": "model_not_loaded"}

            # Get features for these matches
            match_ids = [g[0] for g in imminent_gaps]
            feature_engineer = FeatureEngineer(session=session)

            if hasattr(feature_engineer, 'get_matches_features_by_ids'):
                df = await feature_engineer.get_matches_features_by_ids(match_ids)
            else:
                df = await feature_engineer.get_upcoming_matches_features()
                if len(df) > 0:
                    df = df[df["match_id"].isin(match_ids)]

            if len(df) == 0:
                logger.error(f"[KICKOFF-SAFETY] No features for imminent matches: {match_ids}")
                return {"generated": 0, "error": "no_features", "match_ids": match_ids}

            # Generate predictions
            predictions = engine.predict(df)

            for pred in predictions:
                match_id = pred.get("match_id")
                if not match_id:
                    continue

                try:
                    probs = pred.get("probabilities", {})

                    # Create prediction - table only has home_prob/draw_prob/away_prob
                    prediction = Prediction(
                        match_id=match_id,
                        model_version=engine.model_version or "v1.0.0",
                        home_prob=probs.get("home", 0.33),
                        draw_prob=probs.get("draw", 0.33),
                        away_prob=probs.get("away", 0.34),
                    )

                    session.add(prediction)
                    generated += 1

                    # Derive pick for logging
                    pick = max(probs, key=probs.get) if probs else "home"

                    logger.info(
                        f"[KICKOFF-SAFETY] Generated prediction for {match_id} "
                        f"(pick={pick}, home={probs.get('home', 0):.2%})"
                    )

                except Exception as e:
                    errors.append(f"match {match_id}: {e}")
                    logger.error(f"[KICKOFF-SAFETY] Error for {match_id}: {e}")

            await session.commit()

            duration_ms = (time.time() - start_time) * 1000

            if generated > 0:
                record_job_run(job="kickoff_safety_net", status="ok", duration_ms=duration_ms)
                logger.warning(
                    f"[KICKOFF-SAFETY] Complete: generated {generated} predictions for imminent matches"
                )

            return {
                "generated": generated,
                "duration_ms": duration_ms,
                "errors": errors[:5] if errors else None,
            }

    except Exception as e:
        logger.error(f"[KICKOFF-SAFETY] Failed: {e}")
        return {"generated": 0, "error": str(e)}


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
    from app.database import get_session_with_retry

    frozen_count = 0
    errors = []

    try:
        async with get_session_with_retry(max_retries=3, retry_delay=1.0) as session:
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

                    # OPS Daily Comparison: persist a "market snapshot" row for this match.
                    # This avoids a disconnect where odds_sync updates matches.odds_* but the
                    # OPS dashboard reads from match_odds_snapshot.
                    #
                    # Source of truth for "market at freeze": predictions.frozen_odds_*
                    # (these are copied from matches at freeze time).
                    try:
                        if (
                            pred.frozen_odds_home
                            and pred.frozen_odds_draw
                            and pred.frozen_odds_away
                            and pred.frozen_odds_home > 1
                            and pred.frozen_odds_draw > 1
                            and pred.frozen_odds_away > 1
                        ):
                            # Resolve primary bookmaker for this league (default bet365)
                            bookmaker_row = await session.execute(
                                text("""
                                    SELECT bookmaker
                                    FROM league_bookmaker_config
                                    WHERE league_id = :league_id AND is_primary = TRUE
                                    LIMIT 1
                                """),
                                {"league_id": match.league_id},
                            )
                            bookmaker = (bookmaker_row.scalar() or "bet365").lower()

                            # Compute implied probabilities (normalized, removes overround)
                            raw_home = 1 / float(pred.frozen_odds_home)
                            raw_draw = 1 / float(pred.frozen_odds_draw)
                            raw_away = 1 / float(pred.frozen_odds_away)
                            total = raw_home + raw_draw + raw_away
                            implied_home = raw_home / total
                            implied_draw = raw_draw / total
                            implied_away = raw_away / total
                            market_pick = max(
                                {"home": implied_home, "draw": implied_draw, "away": implied_away},
                                key=lambda k: {"home": implied_home, "draw": implied_draw, "away": implied_away}[k],
                            )

                            await session.execute(
                                text("""
                                    INSERT INTO match_odds_snapshot (
                                        match_id, bookmaker,
                                        odds_home, odds_draw, odds_away,
                                        implied_home, implied_draw, implied_away,
                                        market_pick, snapshot_at, is_primary
                                    ) VALUES (
                                        :match_id, :bookmaker,
                                        :odds_home, :odds_draw, :odds_away,
                                        :implied_home, :implied_draw, :implied_away,
                                        :market_pick, :snapshot_at, TRUE
                                    )
                                    ON CONFLICT (match_id, bookmaker) DO UPDATE SET
                                        odds_home = EXCLUDED.odds_home,
                                        odds_draw = EXCLUDED.odds_draw,
                                        odds_away = EXCLUDED.odds_away,
                                        implied_home = EXCLUDED.implied_home,
                                        implied_draw = EXCLUDED.implied_draw,
                                        implied_away = EXCLUDED.implied_away,
                                        market_pick = EXCLUDED.market_pick,
                                        snapshot_at = EXCLUDED.snapshot_at,
                                        is_primary = TRUE
                                """),
                                {
                                    "match_id": match.id,
                                    "bookmaker": bookmaker,
                                    "odds_home": float(pred.frozen_odds_home),
                                    "odds_draw": float(pred.frozen_odds_draw),
                                    "odds_away": float(pred.frozen_odds_away),
                                    "implied_home": float(round(implied_home, 6)),
                                    "implied_draw": float(round(implied_draw, 6)),
                                    "implied_away": float(round(implied_away, 6)),
                                    "market_pick": market_pick,
                                    "snapshot_at": pred.frozen_at or now,
                                },
                            )
                    except Exception as e:
                        # Don't block freezing if OPS snapshot fails.
                        logger.warning(
                            f"[FREEZE] Failed to upsert match_odds_snapshot for match_id={getattr(match, 'id', None)}: {e}"
                        )

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

                    # Calculate and freeze value bets (normalized format matching engine.py)
                    value_bets = []
                    ev_threshold = 0.05  # 5% EV minimum for value bet

                    def _build_value_bet(outcome: str, prob: float, odds: float, ev: float) -> dict:
                        """Build a normalized value_bet dict matching engine.py format."""
                        if not prob or not odds or odds <= 0:
                            return None
                        implied_prob = 1 / odds
                        edge = prob - implied_prob
                        return {
                            "outcome": outcome,
                            "our_probability": round(prob, 4),
                            "implied_probability": round(implied_prob, 4),
                            "edge": round(edge, 4),
                            "edge_percentage": round(edge * 100, 1),
                            "expected_value": round(ev, 4),
                            "ev_percentage": round(ev * 100, 1),
                            "market_odds": float(odds),
                            "fair_odds": round(1 / prob, 2) if prob > 0 else None,
                            "is_value_bet": True,
                        }

                    if pred.frozen_ev_home and pred.frozen_ev_home >= ev_threshold and match.odds_home:
                        vb = _build_value_bet("home", pred.home_prob, match.odds_home, pred.frozen_ev_home)
                        if vb:
                            value_bets.append(vb)
                    if pred.frozen_ev_draw and pred.frozen_ev_draw >= ev_threshold and match.odds_draw:
                        vb = _build_value_bet("draw", pred.draw_prob, match.odds_draw, pred.frozen_ev_draw)
                        if vb:
                            value_bets.append(vb)
                    if pred.frozen_ev_away and pred.frozen_ev_away >= ev_threshold and match.odds_away:
                        vb = _build_value_bet("away", pred.away_prob, match.odds_away, pred.frozen_ev_away)
                        if vb:
                            value_bets.append(vb)

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
        # Log full exception info to capture root cause (not just rollback failure)
        import traceback
        error_type = type(e).__name__
        error_msg = str(e)
        # Check if this is a connection-related error
        is_connection_error = any(
            keyword in error_msg.lower()
            for keyword in ["connection", "closed", "terminated", "rollback", "interface"]
        )
        if is_connection_error:
            logger.error(
                f"freeze_predictions_before_kickoff DB connection error [{error_type}]: {error_msg}\n"
                f"Traceback: {traceback.format_exc()}"
            )
        else:
            logger.error(f"freeze_predictions_before_kickoff failed [{error_type}]: {error_msg}")
        return {"frozen_count": 0, "error": f"[{error_type}] {error_msg}"}


# Module-level metrics for monitoring (reset weekly)
_lineup_capture_metrics = {
    "critical": {"api_errors_429": 0, "api_errors_timeout": 0, "api_errors_other": 0, "captures": 0, "latencies_ms": []},
    "full": {"api_errors_429": 0, "api_errors_timeout": 0, "api_errors_other": 0, "captures": 0, "latencies_ms": []},
    "last_reset": None,
}

# In-memory cooldown to reduce redundant lineup checks (per process).
# Keyed by external fixture id; value is datetime of last check that found no confirmed lineup.
_lineup_check_cooldown: dict[int, datetime] = {}
_LINEUP_COOLDOWN_SECONDS = int(os.environ.get("LINEUP_CHECK_COOLDOWN_SECONDS", "120"))  # 2 min

# Per-run caps to keep API usage bounded when monitoring "all leagues".
LINEUP_MAX_LINEUPS_PER_RUN_CRITICAL = int(os.environ.get("LINEUP_MAX_LINEUPS_PER_RUN_CRITICAL", "20"))
LINEUP_MAX_LINEUPS_PER_RUN_FULL = int(os.environ.get("LINEUP_MAX_LINEUPS_PER_RUN_FULL", "10"))
LINEUP_MAX_ODDS_PER_RUN = int(os.environ.get("LINEUP_MAX_ODDS_PER_RUN", "10"))


def get_lineup_capture_metrics() -> dict:
    """Get current lineup capture metrics for weekly report."""
    global _lineup_capture_metrics
    return {
        "critical_job": {
            "api_errors_429": _lineup_capture_metrics["critical"]["api_errors_429"],
            "api_errors_timeout": _lineup_capture_metrics["critical"]["api_errors_timeout"],
            "api_errors_other": _lineup_capture_metrics["critical"]["api_errors_other"],
            "captures": _lineup_capture_metrics["critical"]["captures"],
            "avg_latency_ms": (
                sum(_lineup_capture_metrics["critical"]["latencies_ms"]) /
                len(_lineup_capture_metrics["critical"]["latencies_ms"])
                if _lineup_capture_metrics["critical"]["latencies_ms"] else None
            ),
            "max_latency_ms": max(_lineup_capture_metrics["critical"]["latencies_ms"]) if _lineup_capture_metrics["critical"]["latencies_ms"] else None,
        },
        "full_job": {
            "api_errors_429": _lineup_capture_metrics["full"]["api_errors_429"],
            "api_errors_timeout": _lineup_capture_metrics["full"]["api_errors_timeout"],
            "api_errors_other": _lineup_capture_metrics["full"]["api_errors_other"],
            "captures": _lineup_capture_metrics["full"]["captures"],
            "avg_latency_ms": (
                sum(_lineup_capture_metrics["full"]["latencies_ms"]) /
                len(_lineup_capture_metrics["full"]["latencies_ms"])
                if _lineup_capture_metrics["full"]["latencies_ms"] else None
            ),
            "max_latency_ms": max(_lineup_capture_metrics["full"]["latencies_ms"]) if _lineup_capture_metrics["full"]["latencies_ms"] else None,
        },
        "last_reset": _lineup_capture_metrics["last_reset"],
    }


def reset_lineup_capture_metrics():
    """Reset metrics (called weekly before report generation)."""
    global _lineup_capture_metrics
    _lineup_capture_metrics = {
        "critical": {"api_errors_429": 0, "api_errors_timeout": 0, "api_errors_other": 0, "captures": 0, "latencies_ms": []},
        "full": {"api_errors_429": 0, "api_errors_timeout": 0, "api_errors_other": 0, "captures": 0, "latencies_ms": []},
        "last_reset": datetime.utcnow().isoformat(),
    }


async def monitor_lineups_and_capture_odds(critical_window_only: bool = False) -> dict:
    """
    Monitor upcoming matches for lineup announcements and capture odds snapshots.

    This is the CRITICAL job for the Lineup Arbitrage (Value Betting Táctico) strategy.

    OPTIMIZED for PIT evaluation (2026-01-07):
    - Target window: 45-75 minutes before kickoff (when lineups are announced)
    - Sweet spot: 60-80 min (most lineups announced here)
    - Extended monitoring window: 120 minutes ahead
    - Includes extended leagues for higher volume

    ADAPTIVE FREQUENCY SYSTEM:
    - critical_window_only=False: Full scan every 2 min (all matches in 120 min window)
    - critical_window_only=True: Aggressive scan every 60s (only 45-90 min matches)

    METRICS TRACKING (for weekly report):
    - API errors by type (429, timeout, other)
    - Internal latency (lineup detected -> snapshot saved)
    - Capture counts per job type

    When a lineup is announced (~60min before kickoff):
    1. Fetch lineups from API-Football
    2. If lineup is confirmed and we don't have a snapshot yet:
       - Get current LIVE odds (MUST be fresh)
       - Create an odds_snapshot with snapshot_type='lineup_confirmed'
       - Store the exact timestamp for evaluation

    This provides the TRUE baseline for testing if our lineup model beats the
    market odds at the moment of lineup announcement.
    """
    global _lineup_capture_metrics
    from app.models import Match

    # Track which job type for metrics
    job_type = "critical" if critical_window_only else "full"

    captured_count = 0
    checked_count = 0
    errors = []

    try:
        async with AsyncSessionLocal() as session:
            now = datetime.utcnow()
            try:
                league_ids = await resolve_lineup_monitoring_leagues(session)
            except Exception as e:
                logger.warning(f"Could not resolve tracked leagues; falling back to EXTENDED_LEAGUES: {e}")
                league_ids = EXTENDED_LEAGUES

            # ADAPTIVE WINDOW based on mode:
            # - Critical mode: Only 45-90 min window (aggressive 60s polling)
            # - Full mode: 0-120 min window (regular 2 min polling)
            if critical_window_only:
                # CRITICAL WINDOW: Focus on 45-90 min where lineups are typically announced
                # This runs every 60s to maximize capture probability in ideal window
                window_start = now + timedelta(minutes=45)
                window_end = now + timedelta(minutes=90)
                log_prefix = "[CRITICAL]"
            else:
                # FULL WINDOW: Extended monitoring
                window_start = now - timedelta(minutes=5)  # Very short past window
                window_end = now + timedelta(minutes=120)
                log_prefix = "[FULL]"

            # Get matches in the window that:
            # 1. Are in NS (not started) status only
            # 2. Don't have a lineup_confirmed odds snapshot yet
            # 3. Are in our extended leagues (for higher volume)
            # 4. PRIORITIZED by sweet spot (60-80 min) then target window (45-75 min)
            result = await session.execute(text("""
                SELECT m.id, m.external_id, m.date, m.odds_home, m.odds_draw, m.odds_away,
                       m.status, m.league_id,
                       EXTRACT(EPOCH FROM (m.date - NOW())) / 60 as minutes_to_kickoff
                FROM matches m
                WHERE m.date BETWEEN :window_start AND :window_end
                  AND m.status = 'NS'
                  AND (:league_ids_is_null = TRUE OR m.league_id = ANY(:league_ids))
                  AND NOT EXISTS (
                      SELECT 1 FROM odds_snapshots os
                      WHERE os.match_id = m.id
                        AND os.snapshot_type = 'lineup_confirmed'
                  )
                ORDER BY
                  -- PRIORITY 0: Sweet spot 60-80 min (most lineups announced here)
                  -- PRIORITY 1: Ideal window 45-75 min
                  -- PRIORITY 2: Extended window 30-90 min
                  -- PRIORITY 3: Rest
                  CASE
                    WHEN EXTRACT(EPOCH FROM (m.date - NOW())) / 60 BETWEEN 60 AND 80 THEN 0
                    WHEN EXTRACT(EPOCH FROM (m.date - NOW())) / 60 BETWEEN 45 AND 75 THEN 1
                    WHEN EXTRACT(EPOCH FROM (m.date - NOW())) / 60 BETWEEN 30 AND 90 THEN 2
                    ELSE 3
                  END,
                  m.date ASC
            """), {
                "window_start": window_start,
                "window_end": window_end,
                "league_ids": league_ids,
                "league_ids_is_null": league_ids is None,
            })

            matches = result.fetchall()

            # Log window status for monitoring
            if matches:
                in_sweet_spot = sum(1 for m in matches if 60 <= (m.minutes_to_kickoff or 0) <= 80)
                in_ideal = sum(1 for m in matches if 45 <= (m.minutes_to_kickoff or 0) <= 75)
                logger.info(
                    f"{log_prefix} Lineup monitor: {len(matches)} matches, "
                    f"{in_sweet_spot} in sweet spot (60-80), {in_ideal} in ideal (45-75)"
                )

            # Limit processing to avoid API rate limits
            # Hard cap per run (independent of match count) to keep API bounded when tracking all leagues.
            max_lineup_checks = LINEUP_MAX_LINEUPS_PER_RUN_CRITICAL if critical_window_only else LINEUP_MAX_LINEUPS_PER_RUN_FULL
            max_matches = max_lineup_checks  # 1 lineup call per match baseline
            if len(matches) > max_matches:
                logger.warning(
                    f"{log_prefix} Too many matches ({len(matches)}), processing first {max_matches} "
                    f"(prioritized by sweet spot). Remaining will be processed in next run."
                )
                matches = matches[:max_matches]

            if not matches:
                return {"checked": 0, "captured": 0, "message": f"{log_prefix} No matches in window"}

            # Use API-Football to check lineups
            provider = APIFootballProvider()

            try:
                lineup_calls = 0
                odds_calls = 0
                for match in matches:
                    match_id = match.id
                    external_id = match.external_id
                    checked_count += 1

                    try:
                        # Cooldown: if we recently checked this fixture and it wasn't confirmed, skip.
                        if external_id:
                            last = _lineup_check_cooldown.get(int(external_id))
                            if last and (datetime.utcnow() - last).total_seconds() < _LINEUP_COOLDOWN_SECONDS:
                                continue

                        # Per-run cap (lineups)
                        if lineup_calls >= max_lineup_checks:
                            break

                        # Track start time for latency measurement
                        capture_start_time = datetime.utcnow()

                        # Fetch lineup from API with retry logic
                        lineup_data = None
                        max_retries = 3
                        for attempt in range(max_retries):
                            try:
                                lineup_data = await provider.get_lineups(external_id)
                                lineup_calls += 1
                                break
                            except Exception as e:
                                error_str = str(e).lower()
                                # Track error types for metrics
                                if "429" in error_str or "rate limit" in error_str:
                                    _lineup_capture_metrics[job_type]["api_errors_429"] += 1
                                elif "timeout" in error_str or "timed out" in error_str:
                                    _lineup_capture_metrics[job_type]["api_errors_timeout"] += 1
                                else:
                                    _lineup_capture_metrics[job_type]["api_errors_other"] += 1

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
                            if external_id:
                                _lineup_check_cooldown[int(external_id)] = datetime.utcnow()
                            continue

                        # Check if we have valid starting XI
                        home_lineup = lineup_data.get("home")
                        away_lineup = lineup_data.get("away")

                        if not home_lineup or not away_lineup:
                            if external_id:
                                _lineup_check_cooldown[int(external_id)] = datetime.utcnow()
                            continue

                        home_xi = home_lineup.get("starting_xi", [])
                        away_xi = away_lineup.get("starting_xi", [])

                        # DIAGNOSTIC: Log partial lineups (8-10 players) for timing analysis
                        # This helps understand when API publishes lineups gradually
                        if 8 <= len(home_xi) < 11 or 8 <= len(away_xi) < 11:
                            minutes_to_ko = (match.date - datetime.utcnow()).total_seconds() / 60 if match.date else 0
                            logger.info(
                                f"PARTIAL_LINEUP: match_id={match_id} external={external_id} "
                                f"home={len(home_xi)}/11 away={len(away_xi)}/11 "
                                f"minutes_to_kickoff={minutes_to_ko:.1f}"
                            )

                        # Consider lineup confirmed if we have 11 players for each team
                        if len(home_xi) < 11 or len(away_xi) < 11:
                            if external_id:
                                _lineup_check_cooldown[int(external_id)] = datetime.utcnow()
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
                        if odds_calls >= LINEUP_MAX_ODDS_PER_RUN:
                            logger.info(f"{log_prefix} Odds cap reached ({LINEUP_MAX_ODDS_PER_RUN}), deferring odds capture.")
                            continue

                        fresh_odds = await provider.get_odds(external_id)
                        odds_calls += 1

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
                            # Note: This is expected when odds aren't published yet (e.g., >2h before kickoff)
                            logger.warning(
                                f"Cannot capture fresh odds for match {match_id} "
                                f"(external: {external_id}) - API returned: {fresh_odds}. "
                                f"Skipping this match. Will retry in next run."
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

                        # CRITICAL FIX (2026-01-08): Update matches.lineup_confirmed flag
                        # This was missing - snapshots were created but the match wasn't marked
                        await session.execute(text("""
                            UPDATE matches
                            SET lineup_confirmed = TRUE,
                                home_formation = :home_formation,
                                away_formation = :away_formation,
                                lineup_features_computed_at = :computed_at
                            WHERE id = :match_id
                        """), {
                            "match_id": match_id,
                            "home_formation": home_lineup.get("formation"),
                            "away_formation": away_lineup.get("formation"),
                            "computed_at": snapshot_at,
                        })

                        # P0 FIX (2026-01-14): Write-through odds to matches table
                        # odds_snapshots was being populated but matches.odds_* stayed NULL
                        # This broke /predictions/upcoming → market_odds null → iOS no Bookie/EV
                        if odds_freshness == "live":
                            await session.execute(text("""
                                UPDATE matches
                                SET odds_home = :odds_home,
                                    odds_draw = :odds_draw,
                                    odds_away = :odds_away,
                                    odds_recorded_at = :recorded_at
                                WHERE id = :match_id
                                  AND (odds_recorded_at IS NULL OR odds_recorded_at < :recorded_at)
                            """), {
                                "match_id": match_id,
                                "odds_home": odds_home,
                                "odds_draw": odds_draw,
                                "odds_away": odds_away,
                                "recorded_at": snapshot_at,
                            })
                            logger.info(
                                f"Synced live odds to matches: match_id={match_id}, "
                                f"H={odds_home:.2f}, D={odds_draw:.2f}, A={odds_away:.2f}, "
                                f"bookmaker={source}"
                            )

                        # Also update match_lineups with lineup_confirmed_at if not already set
                        await session.execute(text("""
                            UPDATE match_lineups
                            SET lineup_confirmed_at = COALESCE(lineup_confirmed_at, :confirmed_at)
                            WHERE match_id = :match_id
                        """), {"match_id": match_id, "confirmed_at": snapshot_at})

                        captured_count += 1

                        # Track metrics: capture count and latency
                        capture_end_time = datetime.utcnow()
                        latency_ms = int((capture_end_time - capture_start_time).total_seconds() * 1000)
                        _lineup_capture_metrics[job_type]["captures"] += 1
                        _lineup_capture_metrics[job_type]["latencies_ms"].append(latency_ms)

                        # Keep latencies list bounded (last 1000 captures)
                        if len(_lineup_capture_metrics[job_type]["latencies_ms"]) > 1000:
                            _lineup_capture_metrics[job_type]["latencies_ms"] = \
                                _lineup_capture_metrics[job_type]["latencies_ms"][-1000:]

                        logger.info(
                            f"Captured lineup_confirmed odds for match {match_id}: "
                            f"H={odds_home:.2f}, D={odds_draw:.2f}, A={odds_away:.2f} at {snapshot_at} "
                            f"(latency: {latency_ms}ms, job: {job_type})"
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
                "lineup_calls": lineup_calls,
                "odds_calls": odds_calls,
                "errors": errors[:5] if errors else None,
            }

    except APIBudgetExceeded as e:
        logger.warning(f"Lineup monitoring stopped: {e}. Budget status: {get_api_budget_status()}")
        return {"checked": checked_count, "captured": captured_count, "error": str(e), "budget": get_api_budget_status()}
    except Exception as e:
        logger.error(f"monitor_lineups_and_capture_odds failed: {e}")
        return {"checked": 0, "captured": 0, "error": str(e)}


# =============================================================================
# MARKET MOVEMENT TRACKING
# =============================================================================
# Time buckets for market movement analysis (minutes before kickoff)
MARKET_MOVEMENT_BUCKETS = [
    (58, 62, "T60"),   # ~60 min before kickoff
    (28, 32, "T30"),   # ~30 min before kickoff
    (13, 17, "T15"),   # ~15 min before kickoff
    (3, 7, "T5"),      # ~5 min before kickoff
]


async def capture_market_movement_snapshots() -> dict:
    """
    Capture odds at predefined time points before kickoff for market movement analysis.

    This helps understand:
    1. How much the market moves between lineup announcement and kickoff
    2. Whether "late" lineup detection (20-30 min) means edge is already priced in
    3. Optimal timing windows for value capture

    Time buckets: T-60, T-30, T-15, T-5 (minutes before kickoff)

    Run frequency: Every 5 minutes

    Environment variables:
    - MARKET_MOVEMENT_REQUIRE_LINEUP: If "0", capture snapshots even without lineup_confirmed.
      Default "1" requires lineup_confirmed=TRUE (legacy behavior).
      Set to "0" to enable CLV proxy baseline capture (T-60/T-30 before lineup).
    """
    from app.etl.api_football import APIFootballProvider

    captured_count = 0
    checked_count = 0

    # Guardrail: require lineup_confirmed by default to avoid extra API calls
    require_lineup = os.environ.get("MARKET_MOVEMENT_REQUIRE_LINEUP", "1") == "1"

    try:
        async with AsyncSessionLocal() as session:
            now = datetime.utcnow()
            league_ids = await resolve_lineup_monitoring_leagues(session)

            # Get matches that need market movement data
            # Focus on matches 5-65 minutes from now (covers all buckets)
            window_start = now + timedelta(minutes=3)
            window_end = now + timedelta(minutes=65)

            # Build query conditionally based on require_lineup setting
            lineup_condition = "AND m.lineup_confirmed = TRUE" if require_lineup else ""

            result = await session.execute(text(f"""
                SELECT m.id, m.external_id, m.date, m.league_id,
                       EXTRACT(EPOCH FROM (m.date - NOW())) / 60 as minutes_to_kickoff
                FROM matches m
                WHERE m.date BETWEEN :window_start AND :window_end
                  AND m.status = 'NS'
                  {lineup_condition}
                  AND m.market_movement_complete = FALSE
                  AND (:league_ids_is_null = TRUE OR m.league_id = ANY(:league_ids))
                ORDER BY m.date
            """), {
                "window_start": window_start,
                "window_end": window_end,
                "league_ids": league_ids,
                "league_ids_is_null": league_ids is None,
            })

            matches = result.fetchall()

            if not matches:
                return {"checked": 0, "captured": 0}

            provider = APIFootballProvider()

            try:
                api_calls_this_run = 0
                MAX_API_CALLS_PER_RUN = int(os.environ.get("MARKET_MOVEMENT_MAX_CALLS", "15"))
                for match in matches:
                    match_id = match.id
                    external_id = match.external_id
                    minutes_to_kickoff = float(match.minutes_to_kickoff)
                    kickoff_time = match.date
                    checked_count += 1

                    # Check which bucket this falls into
                    current_bucket = None
                    for min_m, max_m, bucket_name in MARKET_MOVEMENT_BUCKETS:
                        if min_m <= minutes_to_kickoff <= max_m:
                            current_bucket = bucket_name
                            break

                    if not current_bucket:
                        continue

                    # Check if we already have this bucket for this match
                    existing = await session.execute(text("""
                        SELECT 1 FROM market_movement_snapshots
                        WHERE match_id = :match_id AND snapshot_type = :bucket
                    """), {"match_id": match_id, "bucket": current_bucket})

                    if existing.fetchone():
                        continue

                    if api_calls_this_run >= MAX_API_CALLS_PER_RUN:
                        break

                    # Fetch fresh odds
                    fresh_odds = await provider.get_odds(external_id)
                    api_calls_this_run += 1

                    if not fresh_odds or not fresh_odds.get("odds_home"):
                        continue

                    odds_home = float(fresh_odds["odds_home"])
                    odds_draw = float(fresh_odds["odds_draw"])
                    odds_away = float(fresh_odds["odds_away"])
                    bookmaker = fresh_odds.get("bookmaker", "unknown")

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
                        continue

                    snapshot_at = datetime.utcnow()

                    # Insert market movement snapshot
                    await session.execute(text("""
                        INSERT INTO market_movement_snapshots (
                            match_id, snapshot_type, captured_at, kickoff_time,
                            minutes_to_kickoff, odds_home, odds_draw, odds_away,
                            bookmaker, odds_freshness,
                            prob_home, prob_draw, prob_away, overround
                        ) VALUES (
                            :match_id, :bucket, :captured_at, :kickoff_time,
                            :minutes_to_kickoff, :odds_home, :odds_draw, :odds_away,
                            :bookmaker, 'live',
                            :prob_home, :prob_draw, :prob_away, :overround
                        )
                        ON CONFLICT (match_id, snapshot_type, bookmaker) DO NOTHING
                    """), {
                        "match_id": match_id,
                        "bucket": current_bucket,
                        "captured_at": snapshot_at,
                        "kickoff_time": kickoff_time,
                        "minutes_to_kickoff": minutes_to_kickoff,
                        "odds_home": odds_home,
                        "odds_draw": odds_draw,
                        "odds_away": odds_away,
                        "bookmaker": f"{bookmaker}_live",
                        "prob_home": prob_home,
                        "prob_draw": prob_draw,
                        "prob_away": prob_away,
                        "overround": overround,
                    })

                    captured_count += 1
                    logger.info(
                        f"Market movement {current_bucket} captured for match {match_id}: "
                        f"H={odds_home:.2f} D={odds_draw:.2f} A={odds_away:.2f}"
                    )

                    # Check if match has all buckets now
                    bucket_count = await session.execute(text("""
                        SELECT COUNT(DISTINCT snapshot_type)
                        FROM market_movement_snapshots
                        WHERE match_id = :match_id
                    """), {"match_id": match_id})

                    if bucket_count.scalar() >= 4:
                        await session.execute(text("""
                            UPDATE matches SET market_movement_complete = TRUE
                            WHERE id = :match_id
                        """), {"match_id": match_id})

                await session.commit()

            finally:
                await provider.close()

            if captured_count > 0:
                logger.info(f"Market movement: captured {captured_count} snapshots")

            return {"checked": checked_count, "captured": captured_count}

    except APIBudgetExceeded as e:
        logger.warning(f"Market movement stopped: {e}. Budget status: {get_api_budget_status()}")
        return {"checked": checked_count, "captured": captured_count, "error": str(e), "budget": get_api_budget_status()}
    except Exception as e:
        logger.error(f"capture_market_movement_snapshots failed: {e}")
        return {"checked": 0, "captured": 0, "error": str(e)}


# =============================================================================
# LINEUP-RELATIVE MOVEMENT TRACKING (Auditor Critical Fix)
# =============================================================================
# Tracks odds movement RELATIVE to lineup_detected_at, not just pre-kickoff
# This measures "Did the market move BECAUSE of lineup announcement?"
#
# Snapshot types:
# - L-30: 30 min BEFORE lineup detection (baseline for "pre-lineup odds")
# - L-15: 15 min BEFORE lineup detection
# - L-5:  5 min BEFORE lineup detection
# - L0:   At lineup detection (captured in odds_snapshots already)
# - L+5:  5 min AFTER lineup detection
# - L+10: 10 min AFTER lineup detection (captures market reaction)
#
# Movement metric: delta_p = max(|p_H(t2)-p_H(t1)|, |p_D(t2)-p_D(t1)|, |p_A(t2)-p_A(t1)|)
# where p_X are NORMALIZED probabilities (after removing overround)

LINEUP_MOVEMENT_BUCKETS = [
    (-32, -28, "L-30"),  # ~30 min before lineup
    (-17, -13, "L-15"),  # ~15 min before lineup
    (-7, -3, "L-5"),     # ~5 min before lineup
    (3, 7, "L+5"),       # ~5 min after lineup
    (8, 12, "L+10"),     # ~10 min after lineup
]


def compute_delta_p(
    prob_h1: float, prob_d1: float, prob_a1: float,
    prob_h2: float, prob_d2: float, prob_a2: float
) -> float:
    """
    Compute max absolute movement on normalized probabilities.
    delta_p = max(|p_H(t2)-p_H(t1)|, |p_D(t2)-p_D(t1)|, |p_A(t2)-p_A(t1)|)
    """
    return max(
        abs(prob_h2 - prob_h1),
        abs(prob_d2 - prob_d1),
        abs(prob_a2 - prob_a1)
    )


async def capture_lineup_relative_movement() -> dict:
    """
    Capture odds at time points relative to lineup_detected_at.

    This addresses auditor's critical feedback:
    1. Track movement RELATIVE to lineup detection, not just pre-kickoff
    2. Use normalized probabilities for movement metrics

    Run frequency: Every 3 minutes (matches are scarce, captures are time-sensitive)
    """
    from app.etl.api_football import APIFootballProvider

    captured_count = 0
    checked_count = 0

    try:
        async with AsyncSessionLocal() as session:
            now = datetime.utcnow()
            league_ids = await resolve_lineup_monitoring_leagues(session)

            # Find matches that have lineup_confirmed but need movement tracking
            # We need matches where:
            # 1. lineup_confirmed = TRUE
            # 2. lineup_movement_tracked = FALSE
            # 3. We have the lineup_detected_at timestamp from odds_snapshots
            # 4. Match hasn't started yet (status = 'NS')

            result = await session.execute(text("""
                SELECT
                    m.id as match_id,
                    m.external_id,
                    m.date as kickoff_time,
                    m.league_id,
                    os.snapshot_at as lineup_detected_at,
                    os.prob_home as lineup_prob_home,
                    os.prob_draw as lineup_prob_draw,
                    os.prob_away as lineup_prob_away,
                    os.odds_home as lineup_odds_home,
                    os.odds_draw as lineup_odds_draw,
                    os.odds_away as lineup_odds_away,
                    os.bookmaker as lineup_bookmaker,
                    EXTRACT(EPOCH FROM (NOW() - os.snapshot_at)) / 60 as minutes_since_lineup
                FROM matches m
                JOIN odds_snapshots os ON m.id = os.match_id
                WHERE m.lineup_confirmed = TRUE
                  AND m.status = 'NS'
                  AND os.snapshot_type = 'lineup_confirmed'
                  AND COALESCE(m.lineup_movement_tracked, FALSE) = FALSE
                  AND os.snapshot_at IS NOT NULL
                  AND (:league_ids_is_null = TRUE OR m.league_id = ANY(:league_ids))
                  -- TTL: Only process if lineup was detected in last 45 minutes
                  -- (enough time for L-30 to L+10 window)
                  AND os.snapshot_at > NOW() - INTERVAL '45 minutes'
                  -- TTL: Stop tracking 15 min after kickoff (match already started)
                  AND m.date > NOW() - INTERVAL '15 minutes'
                ORDER BY os.snapshot_at DESC
                LIMIT 20
            """), {"league_ids": league_ids, "league_ids_is_null": league_ids is None})

            matches = result.fetchall()

            if not matches:
                return {"checked": 0, "captured": 0}

            provider = APIFootballProvider()

            # Rate limiting: configurable via env var, default 10
            # 10 calls × 20 runs/hour = 200 calls/hour max for this job
            api_calls_this_run = 0
            MAX_API_CALLS_PER_RUN = int(os.environ.get("LINEUP_MOVEMENT_MAX_CALLS", "10"))

            try:
                for match in matches:
                    match_id = match.match_id
                    external_id = match.external_id
                    lineup_detected_at = match.lineup_detected_at
                    minutes_since_lineup = float(match.minutes_since_lineup)
                    kickoff_time = match.kickoff_time

                    checked_count += 1

                    # L0 snapshot already exists in odds_snapshots
                    # Check if we need to insert it into lineup_movement_snapshots
                    l0_exists = await session.execute(text("""
                        SELECT 1 FROM lineup_movement_snapshots
                        WHERE match_id = :match_id AND snapshot_type = 'L0'
                    """), {"match_id": match_id})

                    if not l0_exists.fetchone():
                        # Insert L0 from odds_snapshots data
                        await session.execute(text("""
                            INSERT INTO lineup_movement_snapshots (
                                match_id, lineup_detected_at, snapshot_type,
                                minutes_from_lineup, captured_at, kickoff_time,
                                odds_home, odds_draw, odds_away, bookmaker, odds_freshness,
                                prob_home, prob_draw, prob_away, overround,
                                delta_p_vs_baseline, baseline_snapshot_type
                            ) VALUES (
                                :match_id, :lineup_detected_at, 'L0',
                                0, :lineup_detected_at, :kickoff_time,
                                :odds_home, :odds_draw, :odds_away, :bookmaker, 'live',
                                :prob_home, :prob_draw, :prob_away, :overround,
                                0, 'L0'
                            )
                            ON CONFLICT (match_id, snapshot_type, bookmaker) DO NOTHING
                        """), {
                            "match_id": match_id,
                            "lineup_detected_at": lineup_detected_at,
                            "kickoff_time": kickoff_time,
                            "odds_home": match.lineup_odds_home,
                            "odds_draw": match.lineup_odds_draw,
                            "odds_away": match.lineup_odds_away,
                            "bookmaker": match.lineup_bookmaker,
                            "prob_home": match.lineup_prob_home,
                            "prob_draw": match.lineup_prob_draw,
                            "prob_away": match.lineup_prob_away,
                            "overround": (1/float(match.lineup_odds_home) +
                                          1/float(match.lineup_odds_draw) +
                                          1/float(match.lineup_odds_away)) - 1
                                          if match.lineup_odds_home else 0,
                        })
                        captured_count += 1

                    # Check which bucket we should capture now
                    current_bucket = None
                    for min_m, max_m, bucket_name in LINEUP_MOVEMENT_BUCKETS:
                        if min_m <= minutes_since_lineup <= max_m:
                            current_bucket = bucket_name
                            break

                    if not current_bucket:
                        # Not in a capture window, skip
                        continue

                    # Check if we already have this bucket
                    existing = await session.execute(text("""
                        SELECT 1 FROM lineup_movement_snapshots
                        WHERE match_id = :match_id AND snapshot_type = :bucket
                    """), {"match_id": match_id, "bucket": current_bucket})

                    if existing.fetchone():
                        continue

                    # Rate limit check
                    if api_calls_this_run >= MAX_API_CALLS_PER_RUN:
                        logger.info(
                            f"Lineup movement: rate limit reached ({MAX_API_CALLS_PER_RUN} calls), "
                            f"deferring remaining matches to next run"
                        )
                        break

                    # Fetch fresh odds for this time point
                    try:
                        fresh_odds = await provider.get_odds(external_id)
                        api_calls_this_run += 1
                    except Exception as e:
                        error_str = str(e).lower()
                        if "429" in error_str or "rate limit" in error_str:
                            # Auto-throttle: stop immediately on rate limit
                            logger.warning(
                                f"Lineup movement: 429 rate limit hit, stopping run. "
                                f"Captured {captured_count} before throttle."
                            )
                            break
                        else:
                            logger.warning(f"API error for match {match_id}: {e}")
                            continue

                    if not fresh_odds or not fresh_odds.get("odds_home"):
                        logger.warning(
                            f"No odds available for lineup movement {current_bucket}, "
                            f"match {match_id}"
                        )
                        continue

                    odds_home = float(fresh_odds["odds_home"])
                    odds_draw = float(fresh_odds["odds_draw"])
                    odds_away = float(fresh_odds["odds_away"])
                    bookmaker = fresh_odds.get("bookmaker", "unknown")

                    # Calculate normalized probabilities
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
                        continue

                    # Calculate delta_p vs L0 baseline
                    delta_p = compute_delta_p(
                        float(match.lineup_prob_home or 0),
                        float(match.lineup_prob_draw or 0),
                        float(match.lineup_prob_away or 0),
                        prob_home, prob_draw, prob_away
                    )

                    snapshot_at = datetime.utcnow()

                    # Insert the snapshot
                    await session.execute(text("""
                        INSERT INTO lineup_movement_snapshots (
                            match_id, lineup_detected_at, snapshot_type,
                            minutes_from_lineup, captured_at, kickoff_time,
                            odds_home, odds_draw, odds_away, bookmaker, odds_freshness,
                            prob_home, prob_draw, prob_away, overround,
                            delta_p_vs_baseline, baseline_snapshot_type
                        ) VALUES (
                            :match_id, :lineup_detected_at, :bucket,
                            :minutes_from_lineup, :captured_at, :kickoff_time,
                            :odds_home, :odds_draw, :odds_away, :bookmaker, 'live',
                            :prob_home, :prob_draw, :prob_away, :overround,
                            :delta_p, 'L0'
                        )
                        ON CONFLICT (match_id, snapshot_type, bookmaker) DO NOTHING
                    """), {
                        "match_id": match_id,
                        "lineup_detected_at": lineup_detected_at,
                        "bucket": current_bucket,
                        "minutes_from_lineup": minutes_since_lineup,
                        "captured_at": snapshot_at,
                        "kickoff_time": kickoff_time,
                        "odds_home": odds_home,
                        "odds_draw": odds_draw,
                        "odds_away": odds_away,
                        "bookmaker": f"{bookmaker}_live",
                        "prob_home": prob_home,
                        "prob_draw": prob_draw,
                        "prob_away": prob_away,
                        "overround": overround,
                        "delta_p": delta_p,
                    })

                    captured_count += 1
                    logger.info(
                        f"Lineup movement {current_bucket} captured for match {match_id}: "
                        f"delta_p={delta_p:.4f} ({minutes_since_lineup:.1f} min since lineup)"
                    )

                    # Check if match has enough snapshots to mark as tracked
                    # We need at least L0 and one post-lineup (L+5 or L+10)
                    snapshot_count = await session.execute(text("""
                        SELECT
                            COUNT(*) FILTER (WHERE snapshot_type = 'L0') as has_l0,
                            COUNT(*) FILTER (WHERE snapshot_type IN ('L+5', 'L+10')) as has_post
                        FROM lineup_movement_snapshots
                        WHERE match_id = :match_id
                    """), {"match_id": match_id})

                    counts = snapshot_count.fetchone()
                    if counts and counts.has_l0 > 0 and counts.has_post > 0:
                        await session.execute(text("""
                            UPDATE matches
                            SET lineup_movement_tracked = TRUE
                            WHERE id = :match_id
                        """), {"match_id": match_id})
                        logger.info(f"Match {match_id} marked as lineup_movement_tracked")

                await session.commit()

            finally:
                await provider.close()

            if captured_count > 0:
                logger.info(
                    f"Lineup-relative movement: captured {captured_count} snapshots "
                    f"for {checked_count} matches"
                )

            return {"checked": checked_count, "captured": captured_count}

    except APIBudgetExceeded as e:
        logger.warning(f"Lineup-relative movement stopped: {e}. Budget status: {get_api_budget_status()}")
        return {"checked": checked_count, "captured": captured_count, "error": str(e), "budget": get_api_budget_status()}
    except Exception as e:
        logger.error(f"capture_lineup_relative_movement failed: {e}")
        return {"checked": 0, "captured": 0, "error": str(e)}


async def daily_save_predictions():
    """
    Daily job to save predictions for upcoming matches.
    Runs every day at 7:00 AM UTC (before audit).
    """
    start_time = time.time()
    logger.info("[DAILY-SAVE] Starting daily prediction save job...")

    try:
        from app.db_utils import upsert
        from app.ml import XGBoostEngine
        from app.models import Prediction, Match
        from sqlalchemy import select, func

        async with AsyncSessionLocal() as session:
            # Load ML engine
            engine = XGBoostEngine()
            if not engine.load_model():
                logger.error("Could not load ML model for prediction save")
                return

            # Get upcoming matches features
            feature_engineer = FeatureEngineer(session=session)
            df = await feature_engineer.get_upcoming_matches_features()

            # Enhanced logging: count NS matches and get next match date
            total_matches = len(df)
            ns_matches = len(df[df["status"] == "NS"]) if total_matches > 0 else 0

            # Query next NS match date directly from DB for accurate logging
            next_ns_result = await session.execute(
                select(func.min(Match.date))
                .where(Match.status == "NS", Match.date > datetime.utcnow())
            )
            next_ns_date = next_ns_result.scalar()

            logger.info(
                f"daily_save_predictions: total_matches={total_matches}, "
                f"ns_matches={ns_matches}, next_ns_utc={next_ns_date.isoformat() if next_ns_date else 'None'}"
            )

            if len(df) == 0:
                logger.info("No upcoming matches to save predictions for (no scheduled matches in window)")
                return

            # Make predictions
            predictions = engine.predict(df)

            # Shadow mode: log parallel predictions if enabled
            from app.ml.shadow import is_shadow_enabled, log_shadow_prediction
            shadow_logged = 0
            shadow_errors = 0

            # Sensor B: log A vs B predictions (internal diagnostics only)
            from app.ml.sensor import log_sensor_prediction
            from app.config import get_settings
            sensor_settings = get_settings()
            sensor_logged = 0
            sensor_errors = 0

            # Save to database using generic upsert (only NS matches)
            saved = 0
            skipped = 0
            for idx, pred in enumerate(predictions):
                match_id = pred.get("match_id")
                if not match_id:
                    continue

                # Only save predictions for NS (not started) matches
                match_status = pred.get("status", "")
                if match_status != "NS":
                    skipped += 1
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

                    # Shadow prediction: log parallel two-stage prediction (never affects main flow)
                    if is_shadow_enabled():
                        try:
                            match_df = df.iloc[[idx]]
                            shadow_result = await log_shadow_prediction(
                                session=session,
                                match_id=match_id,
                                df=match_df,
                                baseline_engine=engine,
                                skip_commit=True,
                            )
                            if shadow_result:
                                shadow_logged += 1
                        except Exception as shadow_err:
                            shadow_errors += 1
                            logger.warning(f"Shadow prediction failed for match {match_id}: {shadow_err}")

                    # Sensor B: log A vs B predictions (internal diagnostics, never affects main flow)
                    if sensor_settings.SENSOR_ENABLED:
                        try:
                            import numpy as np
                            match_df = df.iloc[[idx]]
                            model_a_probs = np.array([probs["home"], probs["draw"], probs["away"]])
                            sensor_result = await log_sensor_prediction(
                                session=session,
                                match_id=match_id,
                                df=match_df,
                                model_a_probs=model_a_probs,
                                model_a_version=engine.model_version,
                            )
                            if sensor_result:
                                sensor_logged += 1
                        except Exception as sensor_err:
                            sensor_errors += 1
                            logger.warning(f"Sensor prediction failed for match {match_id}: {sensor_err}")

                except Exception as e:
                    logger.warning(f"Error saving prediction: {e}")

            await session.commit()

            # Telemetry: record batch counters for shadow/sensor
            if is_shadow_enabled() and (shadow_logged > 0 or shadow_errors > 0):
                record_shadow_predictions_batch(logged=shadow_logged, errors=shadow_errors)
            if sensor_settings.SENSOR_ENABLED and (sensor_logged > 0 or sensor_errors > 0):
                record_sensor_predictions_batch(logged=sensor_logged, errors=sensor_errors)

            # Build log message with all stats for audit trail
            duration_ms = (time.time() - start_time) * 1000
            log_msg = (
                f"[DAILY-SAVE] Complete: saved={saved}, skipped={skipped}, "
                f"ns_matches={ns_matches}, model_version={engine.model_version}, "
                f"duration_ms={duration_ms:.0f}"
            )
            if is_shadow_enabled():
                log_msg += f", shadow={shadow_logged}/{shadow_errors}"
            if sensor_settings.SENSOR_ENABLED:
                log_msg += f", sensor={sensor_logged}/{sensor_errors}"
            logger.info(log_msg)

            # Record job run for monitoring
            from app.telemetry.metrics import record_job_run
            record_job_run(job="daily_save_predictions", status="ok", duration_ms=duration_ms)

    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        logger.error(f"[DAILY-SAVE] Failed after {duration_ms:.0f}ms: {e}")
        from app.telemetry.metrics import record_job_run
        record_job_run(job="daily_save_predictions", status="error", duration_ms=duration_ms)


async def prediction_gap_safety_net():
    """
    Safety net job to catch matches without predictions before kickoff.

    Runs every 30 minutes to detect NS matches in the next 12 hours that
    don't have a prediction yet, and generates predictions for them.

    This covers edge cases like:
    - Isolated midweek matches that fall between daily batch runs
    - Matches added after the daily prediction job ran
    - Any timing gaps in the prediction pipeline
    - LATAM matches with late-night kickoffs

    Safety: Only generates predictions for matches that haven't started yet.

    Telemetry:
    - prediction_gap_safety_net_runs_total{status}: ok, no_gaps, error
    - prediction_gap_safety_net_generated_total: predictions generated
    """
    from sqlalchemy import text
    from app.telemetry.metrics import record_job_run

    start_time = time.time()
    job_name = "prediction_gap_safety_net"

    try:
        from app.db_utils import upsert
        from app.ml import XGBoostEngine
        from app.models import Prediction, Match

        async with AsyncSessionLocal() as session:
            # Find NS matches in next 12 hours without any prediction
            # Extended from 2h to 12h to catch LATAM late-night matches
            lookahead_hours = 12
            result = await session.execute(
                text("""
                    SELECT m.id, m.external_id, m.date, m.league_id,
                           ht.name as home_team, at.name as away_team
                    FROM matches m
                    JOIN teams ht ON ht.id = m.home_team_id
                    JOIN teams at ON at.id = m.away_team_id
                    WHERE m.status = 'NS'
                      AND m.date > NOW()
                      AND m.date <= NOW() + INTERVAL '12 hours'
                      AND NOT EXISTS (
                          SELECT 1 FROM predictions p WHERE p.match_id = m.id
                      )
                    ORDER BY m.date ASC
                """)
            )
            gaps = result.fetchall()

            if not gaps:
                duration_ms = (time.time() - start_time) * 1000
                logger.info(f"[SAFETY-NET] No prediction gaps in next {lookahead_hours}h (checked in {duration_ms:.0f}ms)")
                record_job_run(job=job_name, status="ok", duration_ms=duration_ms)
                return {"status": "no_gaps", "checked": True}

            logger.info(f"[SAFETY-NET] Found {len(gaps)} matches without predictions in next {lookahead_hours}h")

            # Load ML engine
            engine = XGBoostEngine()
            if not engine.load_model():
                logger.error("[SAFETY-NET] Could not load ML model")
                record_job_run(job=job_name, status="error", duration_ms=(time.time() - start_time) * 1000)
                return {"status": "error", "error": "model_not_loaded"}

            # Get features only for gap matches (optimized for 12h window)
            match_ids = [g[0] for g in gaps]
            feature_engineer = FeatureEngineer(session=session)

            # Try to use optimized method if available, else fallback
            if hasattr(feature_engineer, 'get_matches_features_by_ids'):
                df = await feature_engineer.get_matches_features_by_ids(match_ids)
            else:
                # Fallback: get all upcoming and filter
                df = await feature_engineer.get_upcoming_matches_features()
                if len(df) > 0:
                    df = df[df["match_id"].isin(match_ids)]

            if len(df) == 0:
                logger.warning(f"[SAFETY-NET] No features available for gap matches: {match_ids}")
                record_job_run(job=job_name, status="ok", duration_ms=(time.time() - start_time) * 1000)
                return {"status": "no_features", "match_ids": match_ids}

            # Generate predictions
            predictions = engine.predict(df)

            # Shadow/Sensor imports for parallel logging
            from app.ml.shadow import is_shadow_enabled, log_shadow_prediction
            from app.ml.sensor import log_sensor_prediction
            from app.config import get_settings
            sensor_settings = get_settings()

            saved = 0
            for idx, pred in enumerate(predictions):
                match_id = pred.get("match_id")
                if not match_id:
                    continue

                # Double-check match is still NS (safety)
                match_status = pred.get("status", "")
                if match_status != "NS":
                    logger.debug(f"[SAFETY-NET] Skipping {match_id}, status={match_status}")
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

                    # Log match info for debugging
                    gap_info = next((g for g in gaps if g[0] == match_id), None)
                    if gap_info:
                        logger.info(
                            f"[SAFETY-NET] Generated prediction for {gap_info[4]} vs {gap_info[5]} "
                            f"(id={match_id}, kickoff={gap_info[2]})"
                        )

                    # Shadow prediction (if enabled)
                    if is_shadow_enabled():
                        try:
                            match_df = df[df["match_id"] == match_id]
                            if len(match_df) > 0:
                                await log_shadow_prediction(
                                    session=session,
                                    match_id=match_id,
                                    df=match_df,
                                    baseline_engine=engine,
                                    skip_commit=True,
                                )
                        except Exception as shadow_err:
                            logger.warning(f"[SAFETY-NET] Shadow failed for {match_id}: {shadow_err}")

                    # Sensor B prediction (if enabled)
                    if sensor_settings.SENSOR_ENABLED:
                        try:
                            import numpy as np
                            match_df = df[df["match_id"] == match_id]
                            if len(match_df) > 0:
                                model_a_probs = np.array([probs["home"], probs["draw"], probs["away"]])
                                await log_sensor_prediction(
                                    session=session,
                                    match_id=match_id,
                                    df=match_df,
                                    model_a_probs=model_a_probs,
                                    model_a_version=engine.model_version,
                                )
                        except Exception as sensor_err:
                            logger.warning(f"[SAFETY-NET] Sensor failed for {match_id}: {sensor_err}")

                except Exception as e:
                    logger.warning(f"[SAFETY-NET] Error saving prediction for {match_id}: {e}")

            await session.commit()

            duration_ms = (time.time() - start_time) * 1000
            logger.info(f"[SAFETY-NET] Complete: {saved} predictions generated for gap matches")
            record_job_run(job=job_name, status="ok", duration_ms=duration_ms)

            return {"status": "ok", "generated": saved, "gaps_found": len(gaps)}

    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        logger.error(f"[SAFETY-NET] Failed: {e}")
        record_job_run(job=job_name, status="error", duration_ms=duration_ms)
        return {"status": "error", "error": str(e)}


# Live status codes that indicate match is in progress
LIVE_STATUSES = frozenset(["1H", "HT", "2H", "ET", "BT", "P", "LIVE", "INT", "SUSP"])


async def live_tick():
    """
    Global live tick - updates all live matches in batch.

    Runs every 30s. Only fetches if there are live matches.
    Uses batch API call to minimize requests.

    Guardrails (per Director/Auditor approval):
    - Only status IN LIVE_STATUSES (1H, HT, 2H, ET, BT, P, LIVE, INT, SUSP)
    - Stuck guard: date > NOW() - 4 hours
    - Max 50 fixtures per tick (3 API requests max)
    - Updates: status, elapsed, home_goals, away_goals (NO events)
    - Degrade on rate_limited/budget_exceeded

    Telemetry:
    - live_tick_runs_total{status}: ok, skipped, rate_limited, error
    - live_tick_matches_updated_total: matches updated
    - live_tick_matches_live_gauge: current live match count
    """
    import time
    from sqlalchemy import text
    from app.telemetry.metrics import record_job_run
    from app.etl import APIFootballProvider

    start_time = time.time()
    job_name = "live_tick"

    try:
        async with AsyncSessionLocal() as session:
            # Find live matches (stuck guard: within 4h)
            result = await session.execute(
                text("""
                    SELECT id, external_id
                    FROM matches
                    WHERE status IN ('1H', 'HT', '2H', 'ET', 'BT', 'P', 'LIVE', 'INT', 'SUSP')
                      AND date > NOW() - INTERVAL '4 hours'
                    ORDER BY date ASC
                    LIMIT 50
                """)
            )
            live_matches = result.fetchall()

            if not live_matches:
                duration_ms = (time.time() - start_time) * 1000
                record_job_run(job=job_name, status="ok", duration_ms=duration_ms)
                # Don't log when no live matches - too noisy
                return {"status": "skipped", "reason": "no_live_matches"}

            live_count = len(live_matches)
            logger.info(f"[LIVE_TICK] Found {live_count} live matches")

            # Build lookup: external_id -> internal_id
            id_map = {ext_id: int_id for int_id, ext_id in live_matches}
            external_ids = list(id_map.keys())

            # Batch fetch from API-Football (max 20 per request)
            provider = APIFootballProvider()
            updated = 0
            api_errors = 0

            try:
                # Process in chunks of 20 (API limit)
                for i in range(0, len(external_ids), 20):
                    chunk = external_ids[i:i + 20]

                    try:
                        fixtures = await provider.get_fixtures_by_ids(chunk)

                        for f in fixtures:
                            ext_id = f.get("external_id")
                            if ext_id not in id_map:
                                continue

                            match_id = id_map[ext_id]
                            new_status = f.get("status")
                            new_elapsed = f.get("elapsed")
                            new_elapsed_extra = f.get("elapsed_extra")  # Injury time
                            new_home = f.get("home_goals")
                            new_away = f.get("away_goals")
                            new_events = f.get("events")  # FASE 1: Live events (full schema)

                            # Update match in DB (FASE 1: now includes events)
                            # GUARDRAIL: Only update events if we have new data (don't overwrite with NULL)
                            if new_events:
                                await session.execute(
                                    text("""
                                        UPDATE matches
                                        SET status = :status,
                                            elapsed = :elapsed,
                                            elapsed_extra = :elapsed_extra,
                                            home_goals = :home_goals,
                                            away_goals = :away_goals,
                                            events = :events
                                        WHERE id = :match_id
                                    """),
                                    {
                                        "match_id": match_id,
                                        "status": new_status,
                                        "elapsed": new_elapsed,
                                        "elapsed_extra": new_elapsed_extra,
                                        "home_goals": new_home,
                                        "away_goals": new_away,
                                        "events": json.dumps(new_events),
                                    }
                                )
                            else:
                                # No events - update only score/status, preserve existing events
                                await session.execute(
                                    text("""
                                        UPDATE matches
                                        SET status = :status,
                                            elapsed = :elapsed,
                                            elapsed_extra = :elapsed_extra,
                                            home_goals = :home_goals,
                                            away_goals = :away_goals
                                        WHERE id = :match_id
                                    """),
                                    {
                                        "match_id": match_id,
                                        "status": new_status,
                                        "elapsed": new_elapsed,
                                        "elapsed_extra": new_elapsed_extra,
                                        "home_goals": new_home,
                                        "away_goals": new_away,
                                    }
                                )
                            updated += 1

                    except Exception as chunk_err:
                        api_errors += 1
                        err_str = str(chunk_err).lower()

                        # Check for rate limit or budget exceeded
                        if "rate" in err_str or "limit" in err_str:
                            logger.warning(f"[LIVE_TICK] Rate limited, stopping tick early")
                            record_job_run(job=job_name, status="rate_limited", duration_ms=(time.time() - start_time) * 1000)
                            return {"status": "rate_limited", "updated": updated}

                        if "budget" in err_str or "exceeded" in err_str:
                            logger.error(f"[LIVE_TICK] Budget exceeded, disabling tick")
                            record_job_run(job=job_name, status="budget_exceeded", duration_ms=(time.time() - start_time) * 1000)
                            return {"status": "budget_exceeded", "updated": updated}

                        logger.warning(f"[LIVE_TICK] Chunk error: {chunk_err}")

                await session.commit()

            finally:
                await provider.close()

            duration_ms = (time.time() - start_time) * 1000
            logger.info(f"[LIVE_TICK] Updated {updated}/{live_count} live matches in {duration_ms:.0f}ms")
            record_job_run(job=job_name, status="ok", duration_ms=duration_ms)

            return {
                "status": "ok",
                "live_count": live_count,
                "updated": updated,
                "api_errors": api_errors,
                "duration_ms": duration_ms,
            }

    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        logger.error(f"[LIVE_TICK] Failed: {e}")
        record_job_run(job=job_name, status="error", duration_ms=duration_ms)
        return {"status": "error", "error": str(e)}


async def evaluate_shadow_predictions():
    """
    Evaluate shadow predictions against actual outcomes.
    Runs every 30 minutes to update completed matches.

    Updates shadow_predictions records with actual_result, correctness, and Brier scores.

    Telemetry:
    - selected: FT/AET/PEN candidates found
    - updated: rows actually updated
    - Logs only if selected > 0 or error
    - Warns if pending_ft > 0 AND updated == 0 (silent failure detection)
    """
    from app.ml.shadow import is_shadow_enabled, evaluate_shadow_outcomes, get_shadow_health_metrics
    from app.config import get_settings

    settings = get_settings()

    if not is_shadow_enabled():
        return {"status": "disabled", "message": "Shadow mode not enabled"}

    try:
        async with AsyncSessionLocal() as session:
            result = await evaluate_shadow_outcomes(session)
            selected = result.get("selected", 0)
            updated = result.get("updated", 0)

            # Telemetry: record evaluation batch
            if updated > 0:
                record_shadow_evaluation_batch(updated)

            # Log only if there was work or mismatch
            if selected > 0:
                if updated > 0:
                    logger.info(
                        f"Shadow evaluation complete: selected={selected}, updated={updated}, "
                        f"baseline_acc={result['baseline_accuracy']:.1%}, "
                        f"shadow_acc={result['shadow_accuracy']:.1%}, "
                        f"delta_brier={result['delta_brier']:+.4f}"
                    )
                else:
                    # Silent failure: found FT matches but didn't update any
                    logger.warning(
                        f"[SHADOW] Silent failure detected: selected={selected} FT matches but updated=0. "
                        "Check evaluation logic."
                    )

            # Update health gauges
            health = await get_shadow_health_metrics(session)
            set_shadow_health_metrics(
                eval_lag_minutes=health["eval_lag_minutes"],
                pending_ft=health["pending_ft"],
            )

            # Warn if lag exceeds threshold
            if health["eval_lag_minutes"] > settings.SHADOW_EVAL_STALE_MINUTES:
                logger.warning(
                    f"[SHADOW] Evaluation lag stale: {health['eval_lag_minutes']:.0f}min > "
                    f"{settings.SHADOW_EVAL_STALE_MINUTES}min threshold"
                )

            return result
    except Exception as e:
        logger.error(f"Shadow predictions evaluation failed: {e}")
        return {"status": "error", "error": str(e)}


async def retrain_sensor_model():
    """
    Retrain Sensor B (LogReg L2) on recent finished matches.
    Runs every SENSOR_RETRAIN_INTERVAL_HOURS (default 6h).

    Sensor B is for INTERNAL DIAGNOSTICS ONLY - never affects production picks.

    Telemetry: records retrain status (ok, learning, error)
    """
    from app.ml.sensor import retrain_sensor
    from app.config import get_settings

    settings = get_settings()
    if not settings.SENSOR_ENABLED:
        return {"status": "disabled", "message": "SENSOR_ENABLED=false"}

    logger.info("[SENSOR] Starting sensor retrain job...")

    try:
        async with AsyncSessionLocal() as session:
            result = await retrain_sensor(session)
            status = result.get("status", "ERROR").lower()

            # Telemetry: record retrain run
            if status == "ready":
                record_sensor_retrain("ok")
                logger.info(
                    f"[SENSOR] Retrain complete: n={result.get('samples')}, "
                    f"window={result.get('window_size')}, "
                    f"version={result.get('model_version')}"
                )
            elif status == "learning":
                record_sensor_retrain("learning")
                logger.info(f"[SENSOR] Still learning: {result.get('reason')}")
            else:
                record_sensor_retrain("error")
                logger.warning(f"[SENSOR] Retrain issue: {result}")

            return result
    except Exception as e:
        record_sensor_retrain("error")
        logger.error(f"[SENSOR] Retrain failed: {e}")
        return {"status": "error", "error": str(e)}


async def evaluate_sensor_predictions_job():
    """
    Evaluate Sensor B predictions against actual outcomes.
    Runs every 30 minutes alongside shadow evaluation.

    Updates sensor_predictions records with actual_outcome, correctness, and Brier scores.

    Telemetry:
    - selected: FT/AET/PEN candidates found
    - updated: rows actually updated
    - Logs only if selected > 0 or error
    - Warns if pending_ft > 0 AND updated == 0 (silent failure detection)
    """
    from app.ml.sensor import evaluate_sensor_predictions, get_sensor_health_metrics
    from app.config import get_settings

    settings = get_settings()
    if not settings.SENSOR_ENABLED:
        return {"status": "disabled", "message": "SENSOR_ENABLED=false"}

    try:
        async with AsyncSessionLocal() as session:
            result = await evaluate_sensor_predictions(session)
            selected = result.get("selected", 0)
            updated = result.get("updated", 0)

            # Telemetry: record evaluation batch
            if updated > 0:
                record_sensor_evaluation_batch(updated)

            # Log only if there was work or mismatch
            if selected > 0:
                if updated > 0:
                    logger.info(
                        f"[SENSOR] Evaluation complete: selected={selected}, updated={updated}, "
                        f"A_correct={result.get('a_correct', 0)}, "
                        f"B_correct={result.get('b_correct', 0)}"
                    )
                else:
                    # Silent failure: found FT matches but didn't update any
                    logger.warning(
                        f"[SENSOR] Silent failure detected: selected={selected} FT matches but updated=0. "
                        "Check evaluation logic."
                    )

            # Update health gauges
            health = await get_sensor_health_metrics(session)
            set_sensor_health_metrics(
                eval_lag_minutes=health["eval_lag_minutes"],
                pending_ft=health["pending_ft"],
                state=health["state"],
            )

            # Warn if lag exceeds threshold
            if health["eval_lag_minutes"] > settings.SENSOR_EVAL_STALE_MINUTES:
                logger.warning(
                    f"[SENSOR] Evaluation lag stale: {health['eval_lag_minutes']:.0f}min > "
                    f"{settings.SENSOR_EVAL_STALE_MINUTES}min threshold"
                )

            return result
    except Exception as e:
        logger.error(f"[SENSOR] Evaluation failed: {e}")
        return {"status": "error", "error": str(e)}


async def daily_refresh_aggregates():
    """
    Daily job to refresh league baselines and team profiles.
    Runs every day at 6:30 AM UTC (after results sync).

    Uses season-to-date data to compute:
    - League baselines: goals_avg, over_X_pct, btts, corners, cards
    - Team profiles: rates, ranks, by_time metrics (0-15, 76-90+)

    This data enriches narratives with relative context (team vs league).
    """
    logger.info("Starting daily aggregates refresh job...")

    try:
        from app.aggregates.refresh_job import refresh_all_aggregates

        async with AsyncSessionLocal() as session:
            result = await refresh_all_aggregates(session)

            logger.info(
                f"Aggregates refresh complete: {result['leagues_processed']} leagues, "
                f"{result['baselines_created']} baselines, {result['profiles_created']} profiles"
            )

            if result.get("errors"):
                for err in result["errors"][:5]:  # Log first 5 errors
                    logger.warning(f"  - {err}")

    except Exception as e:
        logger.error(f"Aggregates refresh failed: {e}")


async def daily_sync_results():
    """
    Daily job to sync match results from API.
    Runs every day at 6:00 AM UTC (before predictions and audit).

    This ensures we have the latest results for:
    - Yesterday's completed matches
    - Matches that finished after the last sync
    """
    sync_leagues = get_sync_leagues()
    logger.info(f"Starting daily results sync job for {len(sync_leagues)} leagues...")

    try:
        async with AsyncSessionLocal() as session:
            pipeline = await create_etl_pipeline(session)
            result = await pipeline.sync_multiple_leagues(
                league_ids=sync_leagues,
                season=CURRENT_SEASON,
                fetch_odds=False,  # Only sync results, not odds
            )

            logger.info(
                f"Daily sync complete: {result['total_matches_synced']} matches synced "
                f"from {len(sync_leagues)} leagues"
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
            sync_leagues = get_sync_leagues()
            logger.info(f"Syncing {len(sync_leagues)} leagues...")
            pipeline = await create_etl_pipeline(session)
            sync_result = await pipeline.sync_multiple_leagues(
                league_ids=sync_leagues,
                season=CURRENT_SEASON,
                # Guardrail: weekly recalibration must not write odds history.
                # PIT odds capture is handled by lineup jobs into odds_snapshots.
                fetch_odds=False,
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


# =============================================================================
# FINISHED MATCH STATS BACKFILL
# =============================================================================
# Captures post-match statistics (possession, shots, corners, cards, etc.)
# for recently finished matches that don't have stats yet.
# Uses APIFootballProvider.get_fixture_statistics() which already exists.

async def capture_finished_match_stats() -> dict:
    """
    Backfill job to fetch detailed statistics for recently finished matches.

    This job:
    1. Selects matches with status FT/AET/PEN in last N hours without stats
    2. Fetches statistics from API-Football per external_id
    3. Updates matches.stats with the JSON dict {home: {...}, away: {...}}

    Guardrails:
    - STATS_BACKFILL_ENABLED: If false, job returns immediately
    - STATS_BACKFILL_MAX_CALLS_PER_RUN: Hard cap on API calls per run
    - STATS_BACKFILL_LOOKBACK_HOURS: Only look at matches this recent
    - Respects global API budget (APIBudgetExceeded)
    - Auto-throttle on 429 errors

    Run frequency: Every 60 minutes (configurable)
    """
    import json
    import os
    import time
    from datetime import datetime
    from pathlib import Path

    from app.config import get_settings

    settings = get_settings()
    start_time = time.time()

    # Check if job is enabled
    enabled = os.environ.get("STATS_BACKFILL_ENABLED", str(settings.STATS_BACKFILL_ENABLED)).lower()
    if enabled in ("false", "0", "no"):
        logger.info("Stats backfill job disabled via STATS_BACKFILL_ENABLED=false")
        return {"status": "disabled"}

    # Get configuration from env or settings
    lookback_hours = int(os.environ.get("STATS_BACKFILL_LOOKBACK_HOURS", settings.STATS_BACKFILL_LOOKBACK_HOURS))
    max_calls = int(os.environ.get("STATS_BACKFILL_MAX_CALLS_PER_RUN", settings.STATS_BACKFILL_MAX_CALLS_PER_RUN))

    # Metrics tracking
    metrics = {
        "checked": 0,
        "fetched": 0,
        "updated": 0,
        "skipped_already_has_stats": 0,
        "skipped_no_external_id": 0,
        "api_calls": 0,
        "errors_429": 0,
        "errors_other": 0,
        "started_at": datetime.utcnow().isoformat(),
    }

    try:
        async with AsyncSessionLocal() as session:
            now = datetime.utcnow()

            # Select finished matches that need stats
            # - status IN ('FT', 'AET', 'PEN')
            # - date >= NOW() - INTERVAL lookback_hours
            # - stats IS NULL OR stats = '{}'
            # - external_id IS NOT NULL
            # ORDER BY date DESC (most recent first)
            # LIMIT max_calls
            result = await session.execute(text("""
                SELECT id, external_id, date, status, stats, league_id, home_goals, away_goals
                FROM matches
                WHERE status IN ('FT', 'AET', 'PEN')
                  AND date >= NOW() - INTERVAL ':lookback hours'
                  AND external_id IS NOT NULL
                  AND (stats IS NULL OR stats::text = '{}' OR stats::text = 'null')
                ORDER BY date DESC
                LIMIT :max_matches
            """.replace(":lookback", str(lookback_hours))), {
                "max_matches": max_calls,
            })

            matches = result.fetchall()
            metrics["checked"] = len(matches)

            if not matches:
                duration_ms = (time.time() - start_time) * 1000
                logger.info(f"Stats backfill: No matches need stats (lookback={lookback_hours}h)")
                record_job_run(job="stats_backfill", status="ok", duration_ms=duration_ms)
                return {**metrics, "status": "no_matches"}

            logger.info(f"Stats backfill: Found {len(matches)} matches needing stats")

            # Use APIFootballProvider to fetch stats
            provider = APIFootballProvider()

            try:
                for match in matches:
                    match_id = match.id
                    external_id = match.external_id

                    if not external_id:
                        metrics["skipped_no_external_id"] += 1
                        continue

                    # Double-check: skip if already has stats
                    if match.stats and match.stats != {} and str(match.stats) != 'null':
                        metrics["skipped_already_has_stats"] += 1
                        continue

                    # Check API call cap
                    if metrics["api_calls"] >= max_calls:
                        logger.info(f"Stats backfill: Hit call cap ({max_calls}), stopping")
                        break

                    try:
                        # Fetch stats from API
                        stats_data = await provider.get_fixture_statistics(external_id)
                        metrics["api_calls"] += 1
                        metrics["fetched"] += 1

                        if not stats_data:
                            # API returned no stats (match might be too old or stats unavailable)
                            logger.debug(f"No stats available for match {match_id} (external: {external_id})")
                            continue

                        # Update matches.stats with the JSON
                        # _parse_stats already returns {home: {...}, away: {...}}
                        # Cast explicitly to JSON to avoid Postgres text->json type error
                        await session.execute(text("""
                            UPDATE matches
                            SET stats = CAST(:stats_json AS JSON)
                            WHERE id = :match_id
                        """), {
                            "match_id": match_id,
                            "stats_json": json.dumps(stats_data),
                        })
                        metrics["updated"] += 1

                        logger.debug(f"Updated stats for match {match_id}: {list(stats_data.get('home', {}).keys())}")

                    except Exception as e:
                        error_str = str(e).lower()
                        if "429" in error_str or "rate limit" in error_str:
                            metrics["errors_429"] += 1
                            logger.warning(f"Stats backfill: 429 rate limit hit, stopping run")
                            break
                        else:
                            metrics["errors_other"] += 1
                            logger.warning(f"Error fetching stats for match {match_id}: {e}")
                            continue

                await session.commit()

            finally:
                await provider.close()

        # Query remaining pending FT matches for telemetry
        async with AsyncSessionLocal() as session2:
            result_pending = await session2.execute(text("""
                SELECT COUNT(*) as cnt
                FROM matches
                WHERE status IN ('FT', 'AET', 'PEN')
                  AND date >= NOW() - INTERVAL ':lookback hours'
                  AND external_id IS NOT NULL
                  AND (stats IS NULL OR stats::text = '{}' OR stats::text = 'null')
            """.replace(":lookback", str(lookback_hours))))
            ft_pending = result_pending.scalar() or 0

        # Log summary
        duration_ms = (time.time() - start_time) * 1000
        metrics["completed_at"] = datetime.utcnow().isoformat()
        metrics["duration_ms"] = round(duration_ms, 1)
        metrics["ft_pending"] = ft_pending

        logger.info(
            f"Stats backfill complete: "
            f"checked={metrics['checked']}, fetched={metrics['fetched']}, "
            f"updated={metrics['updated']}, api_calls={metrics['api_calls']}, "
            f"errors_429={metrics['errors_429']}, errors_other={metrics['errors_other']}, "
            f"ft_pending={ft_pending}"
        )

        # Record telemetry
        record_job_run(job="stats_backfill", status="ok", duration_ms=duration_ms)
        record_stats_backfill_result(rows_updated=metrics["updated"], ft_pending=ft_pending)

        # Save log file
        logs_dir = Path("logs")
        logs_dir.mkdir(exist_ok=True)
        log_file = logs_dir / f"finished_match_stats_backfill_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.log"
        with open(log_file, "w") as f:
            f.write(json.dumps(metrics, indent=2))

        return {**metrics, "status": "completed", "log_file": str(log_file)}

    except APIBudgetExceeded as e:
        duration_ms = (time.time() - start_time) * 1000
        logger.warning(f"Stats backfill stopped: {e}. Budget status: {get_api_budget_status()}")
        record_job_run(job="stats_backfill", status="budget_exceeded", duration_ms=duration_ms)
        metrics["budget_status"] = get_api_budget_status()
        return {**metrics, "status": "budget_exceeded", "error": str(e)}
    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        logger.error(f"Stats backfill failed: {e}")
        sentry_capture_exception(e, job_id="stats_backfill", metrics=metrics)
        record_job_run(job="stats_backfill", status="error", duration_ms=duration_ms)
        return {**metrics, "status": "error", "error": str(e)}


# =============================================================================
# STATS REFRESH JOB (Fix: late events like red cards)
# =============================================================================
# Re-fetches stats for recently finished matches even if they already have stats.
# Solves: Live sync captures stats mid-match, missing late events (red cards, etc.)


async def refresh_recent_ft_stats(lookback_hours: int = 6, max_calls: int = 50) -> dict:
    """
    Refresh stats for recently finished matches to capture late events.

    Unlike stats_backfill (which only fills NULL stats), this job:
    1. Selects FT/AET/PEN matches that finished in the last N hours
    2. Re-fetches stats from API-Football regardless of existing stats
    3. Overwrites with fresh complete stats

    This captures late events like:
    - Red cards in injury time
    - Late goals
    - Updated possession/shots after final whistle

    Default: 6h lookback, 50 calls max (runs every 2h = ~600 calls/day)

    Args:
        lookback_hours: Hours to look back for finished matches (default 6)
        max_calls: Maximum API calls per run (default 50)

    Returns:
        Dict with metrics: checked, refreshed, errors, etc.
    """
    import json
    import time
    from datetime import datetime

    start_time = time.time()

    # Check if job is enabled (same flag as stats_backfill)
    enabled = os.environ.get("STATS_REFRESH_ENABLED", "true").lower()
    if enabled in ("false", "0", "no"):
        logger.info("Stats refresh job disabled via STATS_REFRESH_ENABLED=false")
        return {"status": "disabled"}

    # Allow override via env
    lookback_hours = int(os.environ.get("STATS_REFRESH_LOOKBACK_HOURS", lookback_hours))
    max_calls = int(os.environ.get("STATS_REFRESH_MAX_CALLS", max_calls))

    metrics = {
        "checked": 0,
        "refreshed": 0,
        "skipped_no_external_id": 0,
        "api_calls": 0,
        "errors_429": 0,
        "errors_other": 0,
        "started_at": datetime.utcnow().isoformat(),
    }

    try:
        async with AsyncSessionLocal() as session:
            # Select recently finished matches (regardless of existing stats)
            # Key difference: no stats filter, we want to refresh ALL recent FT
            result = await session.execute(text("""
                SELECT id, external_id, date, status, finished_at
                FROM matches
                WHERE status IN ('FT', 'AET', 'PEN')
                  AND finished_at >= NOW() - INTERVAL ':lookback hours'
                  AND external_id IS NOT NULL
                ORDER BY finished_at DESC
                LIMIT :max_matches
            """.replace(":lookback", str(lookback_hours))), {
                "max_matches": max_calls,
            })

            matches = result.fetchall()
            metrics["checked"] = len(matches)

            if not matches:
                duration_ms = (time.time() - start_time) * 1000
                logger.info(f"Stats refresh: No recently finished matches (lookback={lookback_hours}h)")
                record_job_run(job="stats_refresh", status="ok", duration_ms=duration_ms)
                return {**metrics, "status": "no_matches"}

            logger.info(f"Stats refresh: Found {len(matches)} recently finished matches to refresh")

            # Use APIFootballProvider to fetch stats
            provider = APIFootballProvider()

            try:
                for match in matches:
                    match_id = match.id
                    external_id = match.external_id

                    if not external_id:
                        metrics["skipped_no_external_id"] += 1
                        continue

                    if metrics["api_calls"] >= max_calls:
                        logger.info(f"Stats refresh: Hit call cap ({max_calls}), stopping")
                        break

                    try:
                        # Fetch fresh stats from API
                        stats_data = await provider.get_fixture_statistics(external_id)
                        metrics["api_calls"] += 1

                        if not stats_data:
                            logger.debug(f"No stats available for match {match_id} (external: {external_id})")
                            continue

                        # Overwrite existing stats with fresh data
                        await session.execute(text("""
                            UPDATE matches
                            SET stats = CAST(:stats_json AS JSON)
                            WHERE id = :match_id
                        """), {
                            "match_id": match_id,
                            "stats_json": json.dumps(stats_data),
                        })
                        metrics["refreshed"] += 1

                        logger.debug(f"Refreshed stats for match {match_id}: {list(stats_data.get('home', {}).keys())}")

                    except Exception as e:
                        error_str = str(e).lower()
                        if "429" in error_str or "rate limit" in error_str:
                            metrics["errors_429"] += 1
                            logger.warning(f"Stats refresh: 429 rate limit hit, stopping run")
                            break
                        else:
                            metrics["errors_other"] += 1
                            logger.warning(f"Error refreshing stats for match {match_id}: {e}")
                            continue

                await session.commit()

            finally:
                await provider.close()

        duration_ms = (time.time() - start_time) * 1000
        metrics["completed_at"] = datetime.utcnow().isoformat()
        metrics["duration_ms"] = round(duration_ms, 1)

        logger.info(
            f"Stats refresh complete: "
            f"checked={metrics['checked']}, refreshed={metrics['refreshed']}, "
            f"api_calls={metrics['api_calls']}, errors={metrics['errors_other']}"
        )

        record_job_run(job="stats_refresh", status="ok", duration_ms=duration_ms)
        return {**metrics, "status": "completed"}

    except APIBudgetExceeded as e:
        duration_ms = (time.time() - start_time) * 1000
        logger.warning(f"Stats refresh stopped: {e}")
        record_job_run(job="stats_refresh", status="budget_exceeded", duration_ms=duration_ms)
        return {**metrics, "status": "budget_exceeded", "error": str(e)}
    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        logger.error(f"Stats refresh failed: {e}")
        sentry_capture_exception(e, job_id="stats_refresh", metrics=metrics)
        record_job_run(job="stats_refresh", status="error", duration_ms=duration_ms)
        return {**metrics, "status": "error", "error": str(e)}


# =============================================================================
# ODDS SYNC JOB (Auditor Decision 2026-01-16)
# =============================================================================
# Dedicated job to sync 1X2 odds for upcoming matches.
# Solves: API-Football DOES provide LATAM odds, but we weren't fetching them.
# Budget: ~250-400 requests/day with 48h window, 6h interval, 100 cap.


async def sync_odds_for_upcoming_matches() -> dict:
    """
    Sync 1X2 odds for upcoming matches from API-Football.

    This job:
    1. Selects NS matches in configurable window (default 48h ahead)
    2. Skips matches with recent odds (freshness check)
    3. Fetches 1X2 odds from API-Football
    4. Updates matches.odds_* columns with validated odds

    Guardrails:
    - ODDS_SYNC_ENABLED: Kill-switch (default True)
    - ODDS_SYNC_WINDOW_HOURS: Look-ahead window (default 48h)
    - ODDS_SYNC_MAX_FIXTURES: Cap per run (default 100)
    - ODDS_SYNC_FRESHNESS_HOURS: Skip if odds < N hours old (default 6h)
    - Respects global API budget (APIBudgetExceeded)
    - Auto-throttle on 429 errors

    Run frequency: Every 6 hours (configurable via ODDS_SYNC_INTERVAL_HOURS)
    """
    import time
    from datetime import datetime
    from app.config import get_settings
    from app.telemetry.metrics import (
        record_odds_sync_request,
        record_odds_sync_batch,
        record_odds_sync_run,
        record_job_run,
    )
    from app.telemetry.validators import validate_odds_1x2

    settings = get_settings()
    start_time = time.time()

    # Check kill-switch
    if not settings.ODDS_SYNC_ENABLED:
        logger.info("Odds sync job disabled via ODDS_SYNC_ENABLED=false")
        record_odds_sync_run("disabled", 0)
        return {"status": "disabled"}

    # Configuration
    window_hours = settings.ODDS_SYNC_WINDOW_HOURS
    max_fixtures = settings.ODDS_SYNC_MAX_FIXTURES
    freshness_hours = settings.ODDS_SYNC_FRESHNESS_HOURS

    # Metrics tracking
    metrics = {
        "scanned": 0,
        "updated": 0,
        "skipped_fresh": 0,
        "skipped_no_external_id": 0,
        "api_calls": 0,
        "api_empty": 0,
        "api_errors": 0,
        "errors_429": 0,
        "started_at": datetime.utcnow().isoformat(),
    }

    try:
        async with AsyncSessionLocal() as session:
            now = datetime.utcnow()

            # Select NS matches needing odds:
            # - status = 'NS' (not started)
            # - date >= NOW() AND date <= NOW() + window_hours
            # - external_id IS NOT NULL
            # - odds_recorded_at IS NULL OR odds_recorded_at < NOW() - freshness_hours
            # ORDER BY date ASC (soonest first - prioritize imminent matches)
            # LIMIT max_fixtures
            result = await session.execute(text("""
                SELECT id, external_id, date, league_id,
                       odds_home, odds_draw, odds_away, odds_recorded_at
                FROM matches
                WHERE status = 'NS'
                  AND date >= NOW()
                  AND date <= NOW() + INTERVAL ':window hours'
                  AND external_id IS NOT NULL
                  AND (
                      odds_recorded_at IS NULL
                      OR odds_recorded_at < NOW() - INTERVAL ':freshness hours'
                  )
                ORDER BY date ASC
                LIMIT :max_fixtures
            """.replace(":window", str(window_hours)).replace(":freshness", str(freshness_hours))), {
                "max_fixtures": max_fixtures,
            })

            matches = result.fetchall()
            metrics["scanned"] = len(matches)

            if not matches:
                logger.info(
                    f"Odds sync: No matches need odds update "
                    f"(window={window_hours}h, freshness={freshness_hours}h)"
                )
                duration_ms = (time.time() - start_time) * 1000
                record_odds_sync_run("ok", duration_ms)
                record_job_run(job="odds_sync", status="ok", duration_ms=duration_ms)
                return {**metrics, "status": "no_matches"}

            logger.info(f"Odds sync: Found {len(matches)} matches needing odds")

            # Use APIFootballProvider to fetch odds
            provider = APIFootballProvider()

            try:
                for match in matches:
                    match_id = match.id
                    external_id = match.external_id

                    if not external_id:
                        metrics["skipped_no_external_id"] += 1
                        continue

                    # Check API call cap
                    if metrics["api_calls"] >= max_fixtures:
                        logger.info(f"Odds sync: Hit call cap ({max_fixtures}), stopping")
                        break

                    try:
                        # Fetch odds from API
                        odds_data = await provider.get_odds(external_id)
                        metrics["api_calls"] += 1

                        if not odds_data:
                            # API returned no odds (not available for this fixture)
                            metrics["api_empty"] += 1
                            record_odds_sync_request("empty", 0)
                            logger.debug(f"No odds available for match {match_id} (external: {external_id})")
                            continue

                        # Validate odds before writing
                        odds_home = odds_data.get("odds_home")
                        odds_draw = odds_data.get("odds_draw")
                        odds_away = odds_data.get("odds_away")

                        validation = validate_odds_1x2(
                            odds_home=odds_home,
                            odds_draw=odds_draw,
                            odds_away=odds_away,
                            book=odds_data.get("bookmaker", "unknown"),
                        )

                        if not validation.is_usable:
                            logger.warning(
                                f"Odds sync: Rejecting invalid odds for match {match_id}: "
                                f"H={odds_home}, D={odds_draw}, A={odds_away}, "
                                f"violations={validation.violations}"
                            )
                            record_odds_sync_request("error", 0)
                            metrics["api_errors"] += 1
                            continue

                        # Update match with validated odds
                        await session.execute(text("""
                            UPDATE matches
                            SET odds_home = :odds_home,
                                odds_draw = :odds_draw,
                                odds_away = :odds_away,
                                odds_recorded_at = NOW()
                            WHERE id = :match_id
                        """), {
                            "match_id": match_id,
                            "odds_home": odds_home,
                            "odds_draw": odds_draw,
                            "odds_away": odds_away,
                        })
                        metrics["updated"] += 1
                        record_odds_sync_request("ok", 0)

                        logger.debug(
                            f"Odds sync: Updated match {match_id}: "
                            f"H={odds_home:.2f}, D={odds_draw:.2f}, A={odds_away:.2f} "
                            f"(source: {odds_data.get('bookmaker', 'unknown')})"
                        )

                    except APIBudgetExceeded as e:
                        logger.warning(f"Odds sync: Budget exceeded, stopping. {e}")
                        break

                    except Exception as e:
                        error_str = str(e).lower()
                        if "429" in error_str or "rate limit" in error_str:
                            metrics["errors_429"] += 1
                            record_odds_sync_request("rate_limited", 0)
                            logger.warning("Odds sync: 429 rate limit hit, stopping run")
                            break
                        else:
                            metrics["api_errors"] += 1
                            record_odds_sync_request("error", 0)
                            logger.warning(f"Error fetching odds for match {match_id}: {e}")
                            continue

                await session.commit()

            finally:
                await provider.close()

        # Log summary
        duration_ms = (time.time() - start_time) * 1000
        metrics["completed_at"] = datetime.utcnow().isoformat()
        metrics["duration_ms"] = round(duration_ms, 1)

        record_odds_sync_batch(metrics["scanned"], metrics["updated"])
        record_odds_sync_run("ok", duration_ms)
        record_job_run(job="odds_sync", status="ok", duration_ms=duration_ms)

        logger.info(
            f"Odds sync complete: "
            f"scanned={metrics['scanned']}, updated={metrics['updated']}, "
            f"api_calls={metrics['api_calls']}, empty={metrics['api_empty']}, "
            f"errors_429={metrics['errors_429']}, duration={duration_ms:.0f}ms"
        )

        return {**metrics, "status": "completed"}

    except APIBudgetExceeded as e:
        duration_ms = (time.time() - start_time) * 1000
        logger.warning(f"Odds sync stopped: {e}. Budget status: {get_api_budget_status()}")
        record_odds_sync_run("error", duration_ms)
        record_job_run(job="odds_sync", status="budget_exceeded", duration_ms=duration_ms)
        metrics["budget_status"] = get_api_budget_status()
        return {**metrics, "status": "budget_exceeded", "error": str(e)}
    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        logger.error(f"Odds sync failed: {e}")
        sentry_capture_exception(e, job_id="odds_sync", metrics=metrics)
        record_odds_sync_run("error", duration_ms)
        record_job_run(job="odds_sync", status="error", duration_ms=duration_ms)
        return {**metrics, "status": "error", "error": str(e)}


# =============================================================================
# PIT EVALUATION JOBS (Protocol 2026-01-07 v2.1)
# =============================================================================

async def daily_pit_evaluation():
    """
    Daily PIT evaluation job - runs silently and saves to DB + logs/.

    DOES NOT spam daily reports. Only saves data for weekly consolidation.
    Weekly report will analyze all daily JSONs and produce a single report.

    Runs daily at 9:00 AM UTC (after audit, after matches have finished).

    Now also persists to pit_reports table for Railway deploy resilience.
    """
    logger.info("Starting daily PIT evaluation (silent save)...")

    try:
        import subprocess
        import os
        import json
        from glob import glob

        # Run the evaluation script
        env = os.environ.copy()
        env["DATABASE_URL"] = os.environ.get("DATABASE_URL", "")

        result = subprocess.run(
            ["python", "scripts/evaluate_pit_live_only.py"],
            capture_output=True,
            text=True,
            env=env,
            timeout=300,  # 5 min timeout
        )

        saved_file = None
        if result.returncode == 0:
            # Parse output to find saved file
            for line in result.stdout.split('\n'):
                if 'Resultados guardados en:' in line:
                    saved_file = line.split(':')[-1].strip()
                    logger.info(f"PIT evaluation saved: {saved_file}")
                    break

            # If no explicit file found, try to find the most recent one
            if not saved_file:
                logs_dir = "logs"
                files = sorted(glob(f"{logs_dir}/pit_evaluation_live_only_*.json"))
                if files:
                    saved_file = files[-1]

            # Persist to database for Railway deploy resilience
            if saved_file and os.path.exists(saved_file):
                try:
                    with open(saved_file, 'r') as f:
                        payload = json.load(f)

                    await _save_pit_report_to_db(
                        report_type="daily",
                        payload=payload,
                        source="scheduler"
                    )
                    logger.info("PIT evaluation persisted to DB (pit_reports table)")
                except Exception as db_err:
                    logger.warning(f"Failed to persist PIT report to DB: {db_err}")

            logger.info("Daily PIT evaluation complete (silent mode)")
        else:
            logger.warning(f"PIT evaluation returned non-zero: {result.stderr[:500]}")

    except subprocess.TimeoutExpired:
        logger.error("PIT evaluation timed out after 5 minutes")
    except Exception as e:
        logger.error(f"Daily PIT evaluation failed: {e}")


async def _save_pit_report_to_db(report_type: str, payload: dict, source: str = "scheduler"):
    """
    Upsert PIT report to pit_reports table.
    Uses report_date = today (UTC) for uniqueness.
    """
    from sqlalchemy import text
    from app.database import AsyncSessionLocal
    import json

    today = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)

    async with AsyncSessionLocal() as session:
        try:
            # UPSERT: insert or update if exists for same type+date
            await session.execute(text("""
                INSERT INTO pit_reports (report_type, report_date, payload, source, created_at, updated_at)
                VALUES (:report_type, :report_date, CAST(:payload AS JSON), :source, NOW(), NOW())
                ON CONFLICT (report_type, report_date) DO UPDATE SET
                    payload = CAST(EXCLUDED.payload AS JSON),
                    source = EXCLUDED.source,
                    updated_at = NOW()
            """), {
                "report_type": report_type,
                "report_date": today,
                "payload": json.dumps(payload, default=str),
                "source": source,
            })
            await session.commit()
            logger.info(f"Saved {report_type} PIT report to DB for {today.date()}")
        except Exception as e:
            logger.error(f"Failed to save {report_type} PIT report for {today.date()}: {e}")
            await session.rollback()
            raise


async def weekly_pit_report():
    """
    Weekly consolidated PIT report - the ONLY report that gets published.

    Runs every Tuesday at 10:00 AM UTC (to include full weekend data).

    This job:
    1. Reads all daily pit_evaluation JSONs from the week
    2. Queries database for FULL capture visibility (avoids operational blindness)
    3. Produces a consolidated report with:
       - Checkpoint status (principal + ideal)
       - Edge decay curve/diagnosis
       - Data quality trends
       - ALL live captures (any window) + full minutes distribution
       - Capture delta % (this week vs previous week in [45-75])
       - API error counts by job type (CRITICAL vs FULL)
       - Internal latency metrics (lineup detected -> snapshot saved)
       - Operational recommendation
    4. Logs summary and saves consolidated report
    5. Resets weekly capture metrics for next period
    """
    logger.info("Starting weekly PIT report generation...")

    try:
        import json
        import os
        from datetime import datetime, timedelta

        from sqlalchemy import text

        from app.database import async_engine, AsyncSessionLocal

        # =====================================================================
        # Get capture metrics BEFORE resetting (for this week's report)
        # =====================================================================
        capture_metrics = get_lineup_capture_metrics()

        # ============================================================
        # Load daily PIT reports from DB (persistent across deploys)
        # ============================================================
        cutoff_dt = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=7)
        daily_reports: list[dict] = []
        async with AsyncSessionLocal() as session:
            res = await session.execute(text("""
                SELECT payload, report_date
                FROM pit_reports
                WHERE report_type = 'daily'
                  AND report_date >= :cutoff
                ORDER BY report_date ASC
            """), {"cutoff": cutoff_dt})
            rows = res.fetchall()
            for r in rows:
                daily_reports.append({"payload": r[0], "report_date": r[1]})

        if not daily_reports:
            logger.info("No PIT daily reports in the last 7 days (DB) - skipping weekly report")
            return

        latest = daily_reports[-1]["payload"] or {}
        latest_ref = f"db:daily:{daily_reports[-1]['report_date'].date().isoformat()}"

        # =====================================================================
        # FULL CAPTURE VISIBILITY - Query ALL live captures (any window)
        # This avoids "operational blindness" if system captures too early/late
        # =====================================================================
        full_visibility = {
            "total_live_pre_kickoff_any_window": 0,
            "minutes_to_kickoff_distribution": {},
            "captures_by_range": {
                "very_early_90plus": 0,
                "early_75_90": 0,
                "ideal_45_75": 0,
                "late_30_45": 0,
                "very_late_10_30": 0,
                "too_late_under_10": 0,
            }
        }

        # Track capture delta (this week vs last week in ideal window [45-75])
        capture_delta = {
            "this_week_ideal": 0,
            "last_week_ideal": 0,
            "delta_percent": None,
            "trend": "unknown",
        }

        try:
            async with async_engine.connect() as conn:
                # Count ALL live captures with delta > 0 (before kickoff)
                # Uses odds_snapshots table with snapshot_type = 'lineup_confirmed'
                result = await conn.execute(text("""
                    SELECT
                        COUNT(*) as total,
                        ROUND(delta_to_kickoff_seconds / 60.0) as minutes
                    FROM odds_snapshots
                    WHERE snapshot_type = 'lineup_confirmed'
                      AND odds_freshness = 'live'
                      AND delta_to_kickoff_seconds > 0
                    GROUP BY ROUND(delta_to_kickoff_seconds / 60.0)
                    ORDER BY minutes
                """))
                rows = result.fetchall()

                total_any_window = 0
                minutes_distribution = {}

                for row in rows:
                    count = row[0]
                    minutes = int(row[1]) if row[1] is not None else 0
                    total_any_window += count
                    minutes_distribution[minutes] = count

                    # Categorize by range
                    if minutes >= 90:
                        full_visibility["captures_by_range"]["very_early_90plus"] += count
                    elif minutes >= 75:
                        full_visibility["captures_by_range"]["early_75_90"] += count
                    elif minutes >= 45:
                        full_visibility["captures_by_range"]["ideal_45_75"] += count
                    elif minutes >= 30:
                        full_visibility["captures_by_range"]["late_30_45"] += count
                    elif minutes >= 10:
                        full_visibility["captures_by_range"]["very_late_10_30"] += count
                    else:
                        full_visibility["captures_by_range"]["too_late_under_10"] += count

                full_visibility["total_live_pre_kickoff_any_window"] = total_any_window
                full_visibility["minutes_to_kickoff_distribution"] = minutes_distribution

                logger.info(f"Full visibility: {total_any_window} total live captures (any window)")

                # =====================================================================
                # CAPTURE DELTA: Compare this week vs last week in ideal [45-75] window
                # This tracks if our adaptive frequency optimization is working
                # =====================================================================
                now = datetime.utcnow()
                this_week_start = now - timedelta(days=7)
                last_week_start = now - timedelta(days=14)
                last_week_end = now - timedelta(days=7)

                # This week's ideal captures
                result = await conn.execute(text("""
                    SELECT COUNT(*) FROM odds_snapshots
                    WHERE snapshot_type = 'lineup_confirmed'
                      AND odds_freshness = 'live'
                      AND delta_to_kickoff_seconds BETWEEN 2700 AND 4500  -- 45-75 min in seconds
                      AND snapshot_at >= :this_week_start
                """), {"this_week_start": this_week_start})
                this_week_ideal = result.scalar() or 0

                # Last week's ideal captures
                result = await conn.execute(text("""
                    SELECT COUNT(*) FROM odds_snapshots
                    WHERE snapshot_type = 'lineup_confirmed'
                      AND odds_freshness = 'live'
                      AND delta_to_kickoff_seconds BETWEEN 2700 AND 4500  -- 45-75 min in seconds
                      AND snapshot_at >= :last_week_start
                      AND snapshot_at < :last_week_end
                """), {"last_week_start": last_week_start, "last_week_end": last_week_end})
                last_week_ideal = result.scalar() or 0

                capture_delta["this_week_ideal"] = this_week_ideal
                capture_delta["last_week_ideal"] = last_week_ideal

                if last_week_ideal > 0:
                    delta_pct = ((this_week_ideal - last_week_ideal) / last_week_ideal) * 100
                    capture_delta["delta_percent"] = round(delta_pct, 1)
                    if delta_pct > 10:
                        capture_delta["trend"] = "IMPROVING"
                    elif delta_pct < -10:
                        capture_delta["trend"] = "DECLINING"
                    else:
                        capture_delta["trend"] = "STABLE"
                elif this_week_ideal > 0:
                    capture_delta["trend"] = "NEW_DATA"
                else:
                    capture_delta["trend"] = "NO_DATA"

                logger.info(
                    f"Capture delta: This week={this_week_ideal}, Last week={last_week_ideal}, "
                    f"Delta={capture_delta['delta_percent']}%, Trend={capture_delta['trend']}"
                )

        except Exception as db_err:
            logger.warning(f"Could not query full capture visibility: {db_err}")

        # Extract key metrics for weekly report
        # Support both:
        # - legacy weekly/daily schema (checkpoints/data_quality/edge_decay_diagnostic)
        # - daily live_only schema from scripts/evaluate_pit_live_only.py (counts/brier/betting/phase)
        is_daily_live_only = (
            latest.get("protocol_version") is not None or
            latest.get("counts", {}).get("n_pit_valid_10_90") is not None
        )

        data_quality = latest.get("data_quality", {}) or {}
        checkpoints = latest.get("checkpoints", {}) or {}
        edge_decay = latest.get("edge_decay_diagnostic", {}) or {}

        if is_daily_live_only:
            counts_daily = latest.get("counts", {}) or {}
            principal_n = int(counts_daily.get("n_pit_valid_10_90", 0) or 0)
            ideal_n = int(counts_daily.get("n_pit_valid_ideal_45_75", 0) or 0)

            phase = (latest.get("phase") or "unknown").strip().lower()
            phase_to_status = {
                "formal": "formal",
                "preliminar": "preliminary",
                "piloto": "piloto",
                "insufficient": "insufficient",
            }
            principal_status = phase_to_status.get(phase, phase)
            ideal_status = principal_status

            # Derive an "edge diagnostic" proxy from skill_vs_market when available
            brier = latest.get("brier", {}) or {}
            skill = brier.get("skill_vs_market")
            if skill is None:
                edge_diagnostic = "INSUFFICIENT_DATA"
            else:
                try:
                    skill_f = float(skill)
                except Exception:
                    skill_f = 0.0
                if skill_f > 0.05:
                    edge_diagnostic = "EDGE_PERSISTS"
                elif skill_f > -0.05:
                    edge_diagnostic = "INCONCLUSIVE"
                else:
                    edge_diagnostic = "NO_ALPHA"

            # Proxy quality score for weekly summary: % captures in ideal window
            if principal_n > 0:
                data_quality = {"quality_score": round((ideal_n / principal_n) * 100, 1)}
            else:
                data_quality = {"quality_score": None}
        else:
            principal_n = checkpoints.get("principal", {}).get("n", 0)
            principal_status = checkpoints.get("principal", {}).get("status", "insufficient")
            ideal_n = checkpoints.get("ideal", {}).get("n", 0)
            ideal_status = checkpoints.get("ideal", {}).get("status", "insufficient")
            edge_diagnostic = edge_decay.get("diagnostic", "UNKNOWN")

        # Determine operational recommendation
        if principal_status == "formal" and edge_diagnostic == "EDGE_PERSISTS":
            recommendation = "CONTINUE - Alpha confirmed, operate normally"
        elif principal_status == "formal" and edge_diagnostic == "EDGE_DECAYS":
            recommendation = "OPTIMIZE - Maximize [45-75] min captures"
        elif principal_status == "formal" and edge_diagnostic == "NO_ALPHA":
            recommendation = "REVIEW MODEL - No significant alpha detected"
        elif principal_status == "preliminary":
            recommendation = f"PRELIMINARY - Wait for N>=200 (current: {principal_n})"
        else:
            recommendation = f"ACCUMULATING - Need more data (current: {principal_n})"

        # Check if capturing in target window
        # For live_only daily schema, synthesize a minimal bin distribution.
        if is_daily_live_only:
            counts = {
                "valid_10_90": int(principal_n or 0),
                "ideal_45_75": int(ideal_n or 0),
            }
        else:
            counts = latest.get("counts", {}).get("by_bin", {}) or {}
        ideal_45_75 = counts.get("ideal_45_75", 0)
        late_10_30 = counts.get("late_10_30", 0)

        if ideal_45_75 < late_10_30 and principal_n > 20:
            recommendation += " | WARNING: More late captures than ideal - adjust timing"

        # Check for operational blindness using full visibility
        total_any = full_visibility["total_live_pre_kickoff_any_window"]
        in_window = principal_n
        if total_any > 0 and in_window < total_any * 0.5:
            pct_outside = ((total_any - in_window) / total_any) * 100
            recommendation += f" | ALERT: {pct_outside:.0f}% of captures outside [10-90] window"

        # Log the weekly summary
        logger.info("=" * 60)
        logger.info("WEEKLY PIT REPORT")
        logger.info("=" * 60)
        logger.info(f"Daily reports analyzed this week: {len(daily_reports)}")
        logger.info(f"Principal [10-90]: N={principal_n}, Status={principal_status}")
        logger.info(f"Ideal [45-75]: N={ideal_n}, Status={ideal_status}")
        logger.info(f"Edge Diagnostic: {edge_diagnostic}")
        logger.info(f"Quality Score: {data_quality.get('quality_score', 'N/A')}%")
        logger.info("-" * 60)
        logger.info("FULL CAPTURE VISIBILITY (avoids operational blindness):")
        logger.info(f"  Total live pre-kickoff (ANY window): {total_any}")
        logger.info(f"  In [10-90] window: {in_window} ({(in_window/total_any*100) if total_any else 0:.1f}%)")
        logger.info(f"  Captures by range: {full_visibility['captures_by_range']}")
        logger.info("-" * 60)
        logger.info("CAPTURE DELTA (this week vs last week in [45-75]):")
        logger.info(f"  This week: {capture_delta['this_week_ideal']}")
        logger.info(f"  Last week: {capture_delta['last_week_ideal']}")
        logger.info(f"  Delta: {capture_delta['delta_percent']}%")
        logger.info(f"  Trend: {capture_delta['trend']}")
        logger.info("-" * 60)
        logger.info("API ERROR TRACKING (by job type):")
        logger.info(f"  CRITICAL job: 429s={capture_metrics['critical_job']['api_errors_429']}, "
                   f"timeouts={capture_metrics['critical_job']['api_errors_timeout']}, "
                   f"other={capture_metrics['critical_job']['api_errors_other']}")
        logger.info(f"  FULL job: 429s={capture_metrics['full_job']['api_errors_429']}, "
                   f"timeouts={capture_metrics['full_job']['api_errors_timeout']}, "
                   f"other={capture_metrics['full_job']['api_errors_other']}")
        logger.info("-" * 60)
        logger.info("INTERNAL LATENCY (lineup detected -> snapshot saved):")
        logger.info(f"  CRITICAL job: avg={capture_metrics['critical_job']['avg_latency_ms']}ms, "
                   f"max={capture_metrics['critical_job']['max_latency_ms']}ms, "
                   f"captures={capture_metrics['critical_job']['captures']}")
        logger.info(f"  FULL job: avg={capture_metrics['full_job']['avg_latency_ms']}ms, "
                   f"max={capture_metrics['full_job']['max_latency_ms']}ms, "
                   f"captures={capture_metrics['full_job']['captures']}")
        logger.info("-" * 60)
        logger.info(f"RECOMMENDATION: {recommendation}")
        logger.info("=" * 60)

        # Save weekly consolidated report
        weekly_report = {
            "report_type": "pit_weekly_consolidated",
            "generated_at": datetime.utcnow().isoformat(),
            "evaluations_analyzed": len(daily_reports),
            "latest_evaluation": latest_ref,
            # Carry latest evaluation metrics for auditing (Brier vs market/uniform, ROI/EV + CI when available)
            "latest_metrics": {
                "phase": latest.get("phase"),
                "brier": latest.get("brier"),
                "betting": latest.get("betting"),
                "interpretation": latest.get("interpretation"),
                "prediction_integrity": latest.get("prediction_integrity"),
            },
            "summary": {
                "principal_n": principal_n,
                "principal_status": principal_status,
                "ideal_n": ideal_n,
                "ideal_status": ideal_status,
                "edge_diagnostic": edge_diagnostic,
                "quality_score": data_quality.get("quality_score"),
            },
            "full_capture_visibility": full_visibility,
            "capture_delta": capture_delta,
            "api_error_tracking": {
                "critical_job": {
                    "errors_429": capture_metrics["critical_job"]["api_errors_429"],
                    "errors_timeout": capture_metrics["critical_job"]["api_errors_timeout"],
                    "errors_other": capture_metrics["critical_job"]["api_errors_other"],
                },
                "full_job": {
                    "errors_429": capture_metrics["full_job"]["api_errors_429"],
                    "errors_timeout": capture_metrics["full_job"]["api_errors_timeout"],
                    "errors_other": capture_metrics["full_job"]["api_errors_other"],
                },
            },
            "internal_latency": {
                "critical_job": {
                    "avg_ms": capture_metrics["critical_job"]["avg_latency_ms"],
                    "max_ms": capture_metrics["critical_job"]["max_latency_ms"],
                    "captures": capture_metrics["critical_job"]["captures"],
                },
                "full_job": {
                    "avg_ms": capture_metrics["full_job"]["avg_latency_ms"],
                    "max_ms": capture_metrics["full_job"]["max_latency_ms"],
                    "captures": capture_metrics["full_job"]["captures"],
                },
            },
            "recommendation": recommendation,
            "bin_distribution": counts,
        }

        # Persist to DB so PIT dashboard survives Railway deploys
        try:
            await _save_pit_report_to_db(report_type="weekly", payload=weekly_report, source="scheduler")
            logger.info("Weekly PIT report persisted to DB (pit_reports table)")
        except Exception as db_err:
            logger.warning(f"Failed to persist weekly PIT report to DB: {db_err}")

        # Optional: also write to filesystem for debugging (ephemeral on Railway)
        try:
            logs_dir = "logs"
            os.makedirs(logs_dir, exist_ok=True)
            report_file = f"{logs_dir}/pit_weekly_{datetime.utcnow().strftime('%Y%m%d')}.json"
            with open(report_file, "w") as f:
                json.dump(weekly_report, f, indent=2)
            logger.info(f"Weekly report saved (filesystem): {report_file}")
        except Exception as fs_err:
            logger.debug(f"Could not write weekly PIT report to filesystem: {fs_err}")

        # Reset metrics for next week
        reset_lineup_capture_metrics()
        logger.info("Lineup capture metrics reset for next week")

    except Exception as e:
        logger.error(f"Weekly PIT report failed: {e}")


async def pit_reports_retention():
    """
    Monthly cleanup of old PIT reports to prevent unbounded growth.

    Retention policy:
    - daily reports: 180 days
    - weekly reports: 365 days

    Runs monthly on the 1st at 4:00 AM UTC.
    """
    logger.info("Starting PIT reports retention cleanup...")

    try:
        from sqlalchemy import text
        from app.database import AsyncSessionLocal

        async with AsyncSessionLocal() as session:
            # Delete old daily reports (> 180 days)
            daily_result = await session.execute(text("""
                DELETE FROM pit_reports
                WHERE report_type = 'daily'
                  AND report_date < (NOW() - INTERVAL '180 days')
                RETURNING id
            """))
            daily_deleted = len(daily_result.fetchall())

            # Delete old weekly reports (> 365 days)
            weekly_result = await session.execute(text("""
                DELETE FROM pit_reports
                WHERE report_type = 'weekly'
                  AND report_date < (NOW() - INTERVAL '365 days')
                RETURNING id
            """))
            weekly_deleted = len(weekly_result.fetchall())

            await session.commit()

            logger.info(f"PIT retention: deleted {daily_deleted} daily (>180d), {weekly_deleted} weekly (>365d) reports")

    except Exception as e:
        logger.error(f"PIT reports retention failed: {e}")


async def llm_raw_output_cleanup():
    """
    Weekly cleanup of llm_output_raw to prevent unbounded growth.

    Retention policy:
    - Keep llm_output_raw for 14 days (enough for debugging)
    - After 14 days, set to NULL but keep other traceability fields
    """
    logger.info("Starting LLM raw output cleanup...")

    try:
        async with AsyncSessionLocal() as session:
            result = await session.execute(text("""
                UPDATE post_match_audits
                SET llm_output_raw = NULL
                WHERE llm_output_raw IS NOT NULL
                  AND llm_narrative_generated_at < (NOW() - INTERVAL '14 days')
                RETURNING id
            """))
            cleaned = len(result.fetchall())

            await session.commit()

            logger.info(f"LLM cleanup: cleared llm_output_raw for {cleaned} audits (>14d)")

    except Exception as e:
        logger.error(f"LLM raw output cleanup failed: {e}")


async def daily_alpha_progress_snapshot() -> dict:
    """
    Daily Alpha Progress snapshot - captures progress state for auditing.

    Runs at 09:10 UTC daily (after ops rollup), saves to alpha_progress_snapshots table.
    Allows tracking evolution of "Progreso hacia Re-test/Alpha" over time.
    """
    import os
    from app.models import AlphaProgressSnapshot

    logger.info("Starting daily Alpha Progress snapshot...")

    try:
        # Import here to avoid circular imports
        from app.main import _get_cached_ops_data

        # Get current ops data
        data = await _get_cached_ops_data()

        # Extract relevant fields for the snapshot
        payload = {
            "generated_at": data.get("generated_at"),
            "league_mode": data.get("league_mode"),
            "tracked_leagues_count": data.get("tracked_leagues_count"),
            "progress": data.get("progress"),
            "budget": {
                "status": data.get("budget", {}).get("status"),
                "plan": data.get("budget", {}).get("plan"),
                "requests_today": data.get("budget", {}).get("requests_today"),
                "requests_limit": data.get("budget", {}).get("requests_limit"),
            },
            "pit": {
                "live_60m": data.get("pit", {}).get("live_60m"),
                "live_24h": data.get("pit", {}).get("live_24h"),
            },
        }

        # Get git commit SHA from env if available
        app_commit = os.environ.get("RAILWAY_GIT_COMMIT_SHA") or os.environ.get("GIT_COMMIT_SHA")

        # Save to DB
        async with AsyncSessionLocal() as session:
            snapshot = AlphaProgressSnapshot(
                payload=payload,
                source="scheduler_daily",
                app_commit=app_commit[:40] if app_commit else None,
            )
            session.add(snapshot)
            await session.commit()
            await session.refresh(snapshot)

            logger.info(f"Alpha Progress snapshot captured: id={snapshot.id}, progress={payload.get('progress', {})}")

            return {
                "status": "captured",
                "id": snapshot.id,
                "captured_at": snapshot.captured_at.isoformat(),
                "progress": payload.get("progress"),
            }

    except Exception as e:
        logger.error(f"Daily Alpha Progress snapshot failed: {e}")
        return {"status": "error", "error": str(e)}


# =============================================================================
# DAILY OPS ROLLUP
# =============================================================================
# League name mapping for rollups
LEAGUE_NAMES_ROLLUP = {
    39: "Premier League",
    140: "La Liga",
    135: "Serie A",
    78: "Bundesliga",
    61: "Ligue 1",
    94: "Primeira Liga",
    88: "Eredivisie",
    203: "Super Lig",
    71: "Brazil Serie A",
    262: "Liga MX",
    128: "Argentina Primera",
    253: "MLS",
    2: "Champions League",
    3: "Europa League",
    848: "Conference League",
}


async def daily_ops_rollup() -> dict:
    """
    Daily ops rollup job - aggregates metrics for the current day (UTC).

    Runs at 09:05 UTC daily, UPSERT by day (idempotent).
    Collects:
    - PIT snapshots (total, live, evaluable)
    - Delta KO bins distribution
    - Baseline coverage metrics
    - Market movement counts by type
    - Per-league breakdown for key leagues
    - Error summary (429s, budget exceeded) if available

    Returns dict with rollup summary.
    """
    import json
    from datetime import date

    today = date.today()
    logger.info(f"Daily ops rollup starting for {today}")

    try:
        async with AsyncSessionLocal() as session:
            payload = {
                "generated_at": datetime.utcnow().isoformat(),
                "day": str(today),
            }

            # =================================================================
            # GLOBAL METRICS (today, UTC)
            # =================================================================

            # PIT snapshots total (lineup_confirmed)
            res = await session.execute(text("""
                SELECT COUNT(*)
                FROM odds_snapshots
                WHERE snapshot_type = 'lineup_confirmed'
                  AND snapshot_at::date = :today
            """), {"today": today})
            payload["pit_snapshots_total"] = int(res.scalar() or 0)

            # PIT snapshots live
            res = await session.execute(text("""
                SELECT COUNT(*)
                FROM odds_snapshots
                WHERE snapshot_type = 'lineup_confirmed'
                  AND odds_freshness = 'live'
                  AND snapshot_at::date = :today
            """), {"today": today})
            payload["pit_snapshots_live"] = int(res.scalar() or 0)

            # PIT bets evaluable (have prediction before snapshot)
            res = await session.execute(text("""
                SELECT COUNT(DISTINCT os.id)
                FROM odds_snapshots os
                WHERE os.snapshot_type = 'lineup_confirmed'
                  AND os.odds_freshness = 'live'
                  AND os.snapshot_at::date = :today
                  AND EXISTS (
                      SELECT 1 FROM predictions p
                      WHERE p.match_id = os.match_id
                        AND p.created_at < os.snapshot_at
                  )
            """), {"today": today})
            payload["pit_bets_evaluable"] = int(res.scalar() or 0)

            # Delta KO bins distribution
            res = await session.execute(text("""
                SELECT
                    CASE
                        WHEN delta_to_kickoff_seconds < 0 THEN 'after_ko'
                        WHEN delta_to_kickoff_seconds < 600 THEN '0-10'
                        WHEN delta_to_kickoff_seconds < 2700 THEN '10-45'
                        WHEN delta_to_kickoff_seconds < 5400 THEN '45-90'
                        ELSE '90+'
                    END AS bin,
                    COUNT(*) AS cnt
                FROM odds_snapshots
                WHERE snapshot_type = 'lineup_confirmed'
                  AND odds_freshness = 'live'
                  AND snapshot_at::date = :today
                  AND delta_to_kickoff_seconds IS NOT NULL
                GROUP BY 1
            """), {"today": today})
            delta_ko_bins = {row[0]: int(row[1]) for row in res.fetchall()}
            payload["delta_ko_bins"] = delta_ko_bins

            # =================================================================
            # BASELINE COVERAGE
            # =================================================================
            # PIT with market_movement pre-KO (for CLV proxy)
            res = await session.execute(text("""
                SELECT
                    COUNT(*) FILTER (WHERE has_baseline) AS with_baseline,
                    COUNT(*) AS total
                FROM (
                    SELECT os.id,
                           EXISTS (
                               SELECT 1 FROM market_movement_snapshots mms
                               WHERE mms.match_id = os.match_id
                                 AND mms.captured_at < (
                                     SELECT m.date FROM matches m WHERE m.id = os.match_id
                                 )
                           ) AS has_baseline
                    FROM odds_snapshots os
                    WHERE os.snapshot_type = 'lineup_confirmed'
                      AND os.odds_freshness = 'live'
                      AND os.snapshot_at::date = :today
                ) sub
            """), {"today": today})
            row = res.first()
            pit_with_baseline = int(row[0] or 0) if row else 0
            pit_total_baseline = int(row[1] or 0) if row else 0
            baseline_pct = round((pit_with_baseline / pit_total_baseline) * 100, 1) if pit_total_baseline > 0 else 0

            payload["baseline_coverage"] = {
                "pit_with_market_baseline": pit_with_baseline,
                "pit_total": pit_total_baseline,
                "baseline_pct": baseline_pct,
            }

            # =================================================================
            # MARKET MOVEMENT COUNTS
            # =================================================================
            res = await session.execute(text("""
                SELECT snapshot_type, COUNT(*) AS cnt
                FROM market_movement_snapshots
                WHERE captured_at::date = :today
                GROUP BY snapshot_type
            """), {"today": today})
            market_movement_by_type = {row[0]: int(row[1]) for row in res.fetchall()}
            market_movement_total = sum(market_movement_by_type.values())

            payload["market_movement"] = {
                "total": market_movement_total,
                "by_type": market_movement_by_type,
            }

            # Lineup movement counts
            try:
                res = await session.execute(text("""
                    SELECT COUNT(*)
                    FROM lineup_movement_snapshots
                    WHERE captured_at::date = :today
                """), {"today": today})
                payload["lineup_movement_total"] = int(res.scalar() or 0)
            except Exception:
                payload["lineup_movement_total"] = None

            # =================================================================
            # PER-LEAGUE BREAKDOWN (Top 5 + key leagues)
            # =================================================================
            key_leagues = TOP5_LEAGUES + [2, 3]  # Top 5 + UCL + UEL
            res = await session.execute(text("""
                SELECT
                    m.league_id,
                    COUNT(*) FILTER (WHERE os.odds_freshness = 'live') AS pit_live,
                    COUNT(*) FILTER (
                        WHERE os.odds_freshness = 'live'
                          AND EXISTS (
                              SELECT 1 FROM predictions p
                              WHERE p.match_id = os.match_id
                                AND p.created_at < os.snapshot_at
                          )
                    ) AS bets_evaluable
                FROM odds_snapshots os
                JOIN matches m ON m.id = os.match_id
                WHERE os.snapshot_type = 'lineup_confirmed'
                  AND os.snapshot_at::date = :today
                  AND m.league_id = ANY(:leagues)
                GROUP BY m.league_id
            """), {"today": today, "leagues": key_leagues})

            by_league = {}
            for row in res.fetchall():
                league_id = int(row[0])
                league_name = LEAGUE_NAMES_ROLLUP.get(league_id, f"League {league_id}")
                by_league[league_name] = {
                    "league_id": league_id,
                    "pit_snapshots_live": int(row[1] or 0),
                    "bets_evaluable": int(row[2] or 0),
                }

            # Add baseline % per league (separate query for clarity)
            for league_name, data in by_league.items():
                league_id = data["league_id"]
                res = await session.execute(text("""
                    SELECT
                        COUNT(*) FILTER (WHERE has_baseline) AS with_baseline,
                        COUNT(*) AS total
                    FROM (
                        SELECT os.id,
                               EXISTS (
                                   SELECT 1 FROM market_movement_snapshots mms
                                   WHERE mms.match_id = os.match_id
                                     AND mms.captured_at < m.date
                               ) AS has_baseline
                        FROM odds_snapshots os
                        JOIN matches m ON m.id = os.match_id
                        WHERE os.snapshot_type = 'lineup_confirmed'
                          AND os.odds_freshness = 'live'
                          AND os.snapshot_at::date = :today
                          AND m.league_id = :league_id
                    ) sub
                """), {"today": today, "league_id": league_id})
                row = res.first()
                with_b = int(row[0] or 0) if row else 0
                total_b = int(row[1] or 0) if row else 0
                data["baseline_pct"] = round((with_b / total_b) * 100, 1) if total_b > 0 else 0.0

                # Market movement total per league
                res = await session.execute(text("""
                    SELECT COUNT(*)
                    FROM market_movement_snapshots mms
                    JOIN matches m ON m.id = mms.match_id
                    WHERE mms.captured_at::date = :today
                      AND m.league_id = :league_id
                """), {"today": today, "league_id": league_id})
                data["market_movement_total"] = int(res.scalar() or 0)

            payload["by_league"] = by_league

            # =================================================================
            # ERROR SUMMARY (best-effort from in-memory counters)
            # =================================================================
            # Get error counts from lineup capture metrics
            error_summary = {
                "api_429_critical": _lineup_capture_metrics.get("critical_window", {}).get("api_errors_429", 0),
                "api_429_full": _lineup_capture_metrics.get("full_window", {}).get("api_errors_429", 0),
                "timeouts_critical": _lineup_capture_metrics.get("critical_window", {}).get("api_errors_timeout", 0),
                "timeouts_full": _lineup_capture_metrics.get("full_window", {}).get("api_errors_timeout", 0),
            }
            # Add budget status
            try:
                budget_status = get_api_budget_status()
                error_summary["budget_used"] = budget_status.get("used")
                error_summary["budget_limit"] = budget_status.get("limit")
                error_summary["budget_pct"] = budget_status.get("used_pct")
            except Exception:
                pass

            payload["errors_summary"] = error_summary

            # =================================================================
            # NOTE FIELD (explain zeros)
            # =================================================================
            notes = []
            if payload["pit_snapshots_live"] == 0:
                notes.append("no PIT snapshots in window")
            if market_movement_total == 0:
                notes.append("no market movement captures")
            if payload["pit_bets_evaluable"] == 0 and payload["pit_snapshots_live"] > 0:
                notes.append("PIT snapshots exist but no predictions found")

            payload["note"] = "; ".join(notes) if notes else None

            # =================================================================
            # UPSERT INTO ops_daily_rollups
            # =================================================================
            payload_json = json.dumps(payload)
            await session.execute(text("""
                INSERT INTO ops_daily_rollups (day, payload, created_at, updated_at)
                VALUES (:day, :payload, NOW(), NOW())
                ON CONFLICT (day) DO UPDATE SET
                    payload = :payload,
                    updated_at = NOW()
            """), {"day": today, "payload": payload_json})
            await session.commit()

            logger.info(
                f"Daily ops rollup complete for {today}: "
                f"pit_live={payload['pit_snapshots_live']}, "
                f"bets_evaluable={payload['pit_bets_evaluable']}, "
                f"baseline_pct={baseline_pct}%"
            )

            return {
                "status": "success",
                "day": str(today),
                "pit_snapshots_live": payload["pit_snapshots_live"],
                "pit_bets_evaluable": payload["pit_bets_evaluable"],
                "baseline_pct": baseline_pct,
                "note": payload.get("note"),
            }

    except Exception as e:
        logger.error(f"Daily ops rollup failed: {e}")
        return {"status": "error", "error": str(e)}


async def daily_predictions_performance_report() -> dict:
    """
    Daily prediction performance report - generates 7d and 14d reports.

    Runs at 09:15 UTC daily (after ops rollup).
    Calculates proper probability metrics:
    - Brier score (primary)
    - Log loss (secondary)
    - Calibration bins
    - Market comparison

    These metrics allow distinguishing variance from bugs.
    """
    logger.info("Daily predictions performance report starting")

    try:
        from app.ml.performance_metrics import (
            generate_performance_report,
            save_performance_report,
        )

        results = {}

        async with AsyncSessionLocal() as session:
            # Generate 7-day report
            report_7d = await generate_performance_report(session, window_days=7)
            await save_performance_report(session, report_7d, window_days=7, source="scheduler")
            results["7d"] = {
                "n": report_7d.get("global", {}).get("n", 0),
                "brier": report_7d.get("global", {}).get("metrics", {}).get("brier_score"),
                "confidence": report_7d.get("confidence"),
            }

            # Generate 14-day report
            report_14d = await generate_performance_report(session, window_days=14)
            await save_performance_report(session, report_14d, window_days=14, source="scheduler")
            results["14d"] = {
                "n": report_14d.get("global", {}).get("n", 0),
                "brier": report_14d.get("global", {}).get("metrics", {}).get("brier_score"),
                "confidence": report_14d.get("confidence"),
            }

        # Log diagnostic summary
        diag_7d = report_7d.get("diagnostics", {})
        recommendation = diag_7d.get("recommendation", "unknown")

        logger.info(
            f"Daily predictions performance report complete: "
            f"7d_n={results['7d']['n']}, 7d_brier={results['7d']['brier']}, "
            f"14d_n={results['14d']['n']}, 14d_brier={results['14d']['brier']}, "
            f"recommendation={recommendation}"
        )

        return {
            "status": "success",
            "7d": results["7d"],
            "14d": results["14d"],
            "recommendation": recommendation,
        }

    except Exception as e:
        logger.error(f"Daily predictions performance report failed: {e}")
        return {"status": "error", "error": str(e)}


def _log_scheduler_jobs():
    """Log all registered scheduler jobs and their next run times."""
    jobs = scheduler.get_jobs()
    if not jobs:
        logger.warning("SCHEDULER HEARTBEAT: No jobs registered!")
        return

    job_info = []
    for job in jobs:
        next_run = job.next_run_time
        next_str = next_run.strftime("%Y-%m-%d %H:%M:%S UTC") if next_run else "None"
        job_info.append(f"  - {job.id}: next={next_str}")

    logger.info(
        f"SCHEDULER HEARTBEAT: {len(jobs)} jobs registered:\n" +
        "\n".join(job_info)
    )


async def update_predictions_health_metrics():
    """
    Update predictions health Prometheus gauges for Grafana alerting.

    Runs every 5 minutes to ensure metrics are fresh even if nobody opens
    /dashboard/ops.json. This enables reliable alerting on:
      (predictions_hours_since_last_saved > 12) AND (predictions_ns_next_48h > 0)

    Emits gauges:
    - predictions_hours_since_last_saved
    - predictions_ns_next_48h
    - predictions_ns_missing_next_48h
    - predictions_coverage_ns_pct
    - predictions_health_status (0=ok, 1=warn, 2=red)
    """
    from sqlalchemy import text
    from app.database import AsyncSessionLocal
    from app.telemetry.metrics import set_predictions_health_metrics

    try:
        async with AsyncSessionLocal() as session:
            now = datetime.utcnow()

            # 1) Hours since last prediction saved
            res = await session.execute(
                text("SELECT MAX(created_at) FROM predictions")
            )
            last_pred_at = res.scalar()

            hours_since_last = None
            if last_pred_at:
                delta = now - last_pred_at
                hours_since_last = round(delta.total_seconds() / 3600, 2)

            # 2) NS matches in next 48h
            res = await session.execute(
                text("""
                    SELECT COUNT(*) FROM matches
                    WHERE status = 'NS'
                      AND date > NOW()
                      AND date <= NOW() + INTERVAL '48 hours'
                """)
            )
            ns_next_48h = int(res.scalar() or 0)

            # 3) NS matches missing prediction
            res = await session.execute(
                text("""
                    SELECT COUNT(*) FROM matches m
                    WHERE m.status = 'NS'
                      AND m.date > NOW()
                      AND m.date <= NOW() + INTERVAL '48 hours'
                      AND NOT EXISTS (
                          SELECT 1 FROM predictions p WHERE p.match_id = m.id
                      )
                """)
            )
            ns_missing = int(res.scalar() or 0)

            # 4) Coverage percentage (NS)
            coverage_pct = 100.0
            if ns_next_48h > 0:
                coverage_pct = round(((ns_next_48h - ns_missing) / ns_next_48h) * 100, 1)

            # 5) FT matches in last 48h missing prediction (IMPACT METRIC)
            res = await session.execute(
                text("""
                    SELECT COUNT(*) FROM matches m
                    WHERE m.status IN ('FT', 'AET', 'PEN')
                      AND m.date >= NOW() - INTERVAL '48 hours'
                      AND NOT EXISTS (
                          SELECT 1 FROM predictions p WHERE p.match_id = m.id
                      )
                """)
            )
            ft_missing = int(res.scalar() or 0)

            # 6) Total FT matches in last 48h for coverage calc
            res = await session.execute(
                text("""
                    SELECT COUNT(*) FROM matches
                    WHERE status IN ('FT', 'AET', 'PEN')
                      AND date >= NOW() - INTERVAL '48 hours'
                """)
            )
            ft_total = int(res.scalar() or 0)

            # 7) FT coverage percentage
            ft_coverage_pct = 100.0
            if ft_total > 0:
                ft_coverage_pct = round(((ft_total - ft_missing) / ft_total) * 100, 1)

            # 8) Determine status (same logic as _calculate_predictions_health in main.py)
            status = "ok"
            if ns_next_48h == 0:
                status = "ok"  # No upcoming matches, no concern
            elif hours_since_last and hours_since_last > 12:
                # Smart bypass: if coverage is 100%, staleness is informational only
                if ns_missing == 0 and ft_missing == 0:
                    status = "ok"
                else:
                    status = "warn"
            elif coverage_pct < 50:
                status = "red"
            elif coverage_pct < 80:
                status = "warn"

            # Emit metrics
            set_predictions_health_metrics(
                hours_since_last=hours_since_last,
                ns_next_48h=ns_next_48h,
                ns_missing_next_48h=ns_missing,
                coverage_ns_pct=coverage_pct,
                status=status,
                ft_missing_48h=ft_missing,
                ft_coverage_pct=ft_coverage_pct,
            )

            hours_str = f"{hours_since_last:.1f}" if hours_since_last else "N/A"
            logger.info(
                f"[METRICS] predictions gauges updated: hours={hours_str}, "
                f"ns_48h={ns_next_48h}, ns_missing={ns_missing}, ns_coverage={coverage_pct}%, "
                f"ft_missing={ft_missing}, ft_coverage={ft_coverage_pct}%, status={status}"
            )

            # Send email alert if status is warn or red
            if status in ("warn", "red"):
                from app.alerting.email import send_alert_email, AlertType

                alert_type = (
                    AlertType.PREDICTIONS_HEALTH_RED
                    if status == "red"
                    else AlertType.PREDICTIONS_HEALTH_WARN
                )
                await send_alert_email(
                    alert_type=alert_type,
                    status=status,
                    hours_since_last=hours_since_last,
                    ns_next_48h=ns_next_48h,
                    ns_missing=ns_missing,
                    coverage_pct=coverage_pct,
                )

    except Exception as e:
        logger.error(f"[METRICS] Failed to update predictions health metrics: {e}")


async def scheduler_heartbeat():
    """Periodic heartbeat to confirm scheduler is running and log job status."""
    _log_scheduler_jobs()


async def fast_postmatch_narratives() -> dict:
    """
    Fast-path LLM narrative generation for recently finished matches.

    This job runs every 2 minutes (configurable via FASTPATH_INTERVAL_SECONDS)
    and generates LLM narratives within minutes of match completion instead
    of waiting for the daily audit job at 08:00 UTC.

    Flow:
    1. Select matches finished in last FASTPATH_LOOKBACK_MINUTES with predictions
    2. Refresh stats for matches that need them (with backoff)
    3. Enqueue ready matches to RunPod (batch)
    4. Poll completions and persist results

    Guardrails:
    - FASTPATH_ENABLED: If false, job returns immediately
    - FASTPATH_MAX_CONCURRENT_JOBS: Cap on simultaneous RunPod jobs
    - Respects stats gating (possession, shots required)
    - Idempotent: skips matches that already have narratives
    """
    global _fastpath_metrics
    import os
    import time
    from app.config import get_settings
    from app.llm.fastpath import FastPathService
    from app.telemetry.metrics import record_job_run, record_fastpath_tick

    settings = get_settings()
    start_time = time.time()

    # Check if job is enabled
    enabled = os.environ.get("FASTPATH_ENABLED", str(settings.FASTPATH_ENABLED)).lower()
    if enabled in ("false", "0", "no"):
        _fastpath_metrics["last_tick_at"] = datetime.utcnow()
        _fastpath_metrics["last_tick_result"] = {"status": "disabled"}
        return {"status": "disabled"}

    logger.info("[FASTPATH] Job started")
    try:
        async with AsyncSessionLocal() as session:
            service = FastPathService(session)
            try:
                result = await service.run_tick()
                duration_ms = int((time.time() - start_time) * 1000)

                # Update in-memory metrics
                now = datetime.utcnow()
                _fastpath_metrics["last_tick_at"] = now
                _fastpath_metrics["last_tick_result"] = result
                _fastpath_metrics["ticks_total"] += 1
                if result.get("selected", 0) > 0 or result.get("enqueued", 0) > 0 or result.get("completed", 0) > 0:
                    _fastpath_metrics["ticks_with_activity"] += 1

                # Persist tick to DB for ops dashboard (survives restarts)
                try:
                    await session.execute(
                        text("""
                            INSERT INTO fastpath_ticks
                            (tick_at, selected, refreshed, ready, enqueued, completed, errors, skipped, duration_ms)
                            VALUES (:tick_at, :selected, :refreshed, :ready, :enqueued, :completed, :errors, :skipped, :duration_ms)
                        """),
                        {
                            "tick_at": now,
                            "selected": result.get("selected", 0),
                            "refreshed": result.get("refreshed", 0),
                            "ready": result.get("stats_ready", 0),
                            "enqueued": result.get("enqueued", 0),
                            "completed": result.get("completed", 0),
                            "errors": result.get("errors", 0),
                            "skipped": result.get("skipped", 0),
                            "duration_ms": duration_ms,
                        }
                    )
                    await session.commit()
                except Exception as db_err:
                    # Log error visibly and rollback to clean session state
                    logger.error(f"[FASTPATH] Persist tick failed: {db_err}", exc_info=True)
                    await session.rollback()

                # Log summary if there was activity
                if result.get("selected", 0) > 0 or result.get("enqueued", 0) > 0:
                    logger.info(
                        f"[FASTPATH] tick complete: "
                        f"selected={result.get('selected', 0)}, "
                        f"stats_refreshed={result.get('refreshed', 0)}, "
                        f"ready={result.get('stats_ready', 0)}, "
                        f"enqueued={result.get('enqueued', 0)}, "
                        f"completed={result.get('completed', 0)}, "
                        f"errors={result.get('errors', 0)}"
                    )

                # Record telemetry
                record_job_run(job="fastpath", status="ok", duration_ms=duration_ms)
                record_fastpath_tick(
                    status="ok",
                    completed_ok=result.get("completed", 0),
                    completed_rejected=0,
                    completed_error=result.get("errors", 0),
                    backlog_ready=result.get("stats_ready", 0),
                )

                return result

            finally:
                await service.close()

    except Exception as e:
        logger.error(f"[FASTPATH] tick failed: {e}", exc_info=True)
        sentry_capture_exception(e, job_id="fastpath")
        now = datetime.utcnow()
        duration_ms = int((time.time() - start_time) * 1000)
        _fastpath_metrics["last_tick_at"] = now
        _fastpath_metrics["last_tick_result"] = {"status": "error", "error": str(e)}

        # Record telemetry (error case)
        record_job_run(job="fastpath", status="error", duration_ms=duration_ms)
        record_fastpath_tick(status="error", completed_error=1)

        # Persist error tick to DB (separate session to avoid state issues)
        try:
            async with AsyncSessionLocal() as err_session:
                await err_session.execute(
                    text("""
                        INSERT INTO fastpath_ticks
                        (tick_at, selected, refreshed, ready, enqueued, completed, errors, skipped, duration_ms, error_detail)
                        VALUES (:tick_at, 0, 0, 0, 0, 0, 1, 0, :duration_ms, :error_detail)
                    """),
                    {"tick_at": now, "duration_ms": duration_ms, "error_detail": str(e)[:500]}
                )
                await err_session.commit()
        except Exception as db_err:
            logger.error(f"[FASTPATH] Failed to persist error tick: {db_err}")

        return {"status": "error", "error": str(e)}


# =============================================================================
# SOTA ENRICHMENT JOBS
# =============================================================================
# Jobs for SOTA feature pipeline: Understat xG, Weather, Venue Geo
# All jobs are best-effort: errors logged, no crash-loops
# Reference: docs/ARCHITECTURE_SOTA.md

async def sota_understat_refs_sync() -> dict:
    """
    Sync Understat external refs for recent matches (Top-5 leagues).

    Links internal matches to Understat match IDs for xG data retrieval.
    Conservative scope: --days 7 --limit 200

    Frequency: Every 12 hours
    Guardrail: SOTA_UNDERSTAT_REFS_ENABLED env var
    """
    import time as _time
    from datetime import datetime

    start_time = _time.time()
    started_at = datetime.utcnow()
    job_name = "sota_understat_refs_sync"

    # Check if enabled
    if os.environ.get("SOTA_UNDERSTAT_REFS_ENABLED", "true").lower() in ("false", "0", "no"):
        logger.info(f"[{job_name}] Disabled via env var")
        return {"status": "disabled"}

    metrics = {
        "scanned": 0,
        "linked_auto": 0,
        "linked_review": 0,
        "skipped_no_candidates": 0,
        "skipped_low_score": 0,
        "errors": 0,
        "started_at": started_at.isoformat(),
    }

    try:
        # Use centralized sota_jobs module (not scripts/)
        from app.etl.sota_jobs import sync_understat_refs
        from app.jobs.tracking import record_job_run as record_job_run_db

        async with AsyncSessionLocal() as session:
            # Run with conservative settings: 7 days, max 200 matches
            stats = await sync_understat_refs(session, days=7, limit=200)
            metrics.update(stats)

            # Record in DB for ops dashboard fallback
            duration_ms = (_time.time() - start_time) * 1000
            status = "ok" if metrics.get("errors", 0) == 0 else "partial"
            await record_job_run_db(session, job_name, status, started_at, metrics=metrics)

        # Record in Prometheus
        record_job_run(job=job_name, status=status, duration_ms=duration_ms)

        logger.info(
            f"[{job_name}] Complete: scanned={metrics['scanned']}, "
            f"linked_auto={metrics['linked_auto']}, linked_review={metrics['linked_review']}, "
            f"errors={metrics['errors']}"
        )
        return {**metrics, "status": status, "duration_ms": duration_ms}

    except Exception as e:
        duration_ms = (_time.time() - start_time) * 1000
        logger.error(f"[{job_name}] Failed: {e}", exc_info=True)
        sentry_capture_exception(e, job_id=job_name)
        record_job_run(job=job_name, status="error", duration_ms=duration_ms)
        # Try to record error in DB too
        try:
            from app.jobs.tracking import record_job_run as record_job_run_db
            async with AsyncSessionLocal() as session:
                await record_job_run_db(session, job_name, "error", started_at, error=str(e))
        except Exception:
            pass  # Best-effort DB recording
        return {"status": "error", "error": str(e), "duration_ms": duration_ms}


async def sota_understat_ft_backfill() -> dict:
    """
    Backfill Understat xG data for finished matches.

    Fetches xG/xPTS from Understat for matches that have refs but no xG data.
    Conservative scope: --days 14 --limit 100 --with-ref-only

    Frequency: Every 6 hours
    Guardrail: SOTA_UNDERSTAT_BACKFILL_ENABLED env var
    Rate limit: 1 req/s (enforced in UnderstatProvider)
    """
    import time as _time
    from datetime import datetime

    start_time = _time.time()
    started_at = datetime.utcnow()
    job_name = "sota_understat_ft_backfill"

    # Check if enabled
    if os.environ.get("SOTA_UNDERSTAT_BACKFILL_ENABLED", "true").lower() in ("false", "0", "no"):
        logger.info(f"[{job_name}] Disabled via env var")
        return {"status": "disabled"}

    metrics = {
        "scanned": 0,
        "inserted": 0,
        "updated": 0,
        "skipped_no_ref": 0,
        "skipped_no_data": 0,
        "errors": 0,
        "started_at": started_at.isoformat(),
    }

    try:
        # Use centralized sota_jobs module (not scripts/)
        from app.etl.sota_jobs import backfill_understat_ft
        from app.jobs.tracking import record_job_run as record_job_run_db

        async with AsyncSessionLocal() as session:
            # Run with conservative settings: 14 days, max 100, only matches with refs
            stats = await backfill_understat_ft(session, days=14, limit=100, with_ref_only=True)
            metrics.update(stats)

            # Record in DB for ops dashboard fallback
            duration_ms = (_time.time() - start_time) * 1000
            status = "ok" if metrics.get("errors", 0) == 0 else "partial"
            await record_job_run_db(session, job_name, status, started_at, metrics=metrics)

        # Record in Prometheus
        record_job_run(job=job_name, status=status, duration_ms=duration_ms)

        logger.info(
            f"[{job_name}] Complete: scanned={metrics['scanned']}, "
            f"inserted={metrics['inserted']}, updated={metrics['updated']}, "
            f"errors={metrics['errors']}"
        )
        return {**metrics, "status": status, "duration_ms": duration_ms}

    except Exception as e:
        duration_ms = (_time.time() - start_time) * 1000
        logger.error(f"[{job_name}] Failed: {e}", exc_info=True)
        sentry_capture_exception(e, job_id=job_name)
        record_job_run(job=job_name, status="error", duration_ms=duration_ms)
        # Try to record error in DB too
        try:
            from app.jobs.tracking import record_job_run as record_job_run_db
            async with AsyncSessionLocal() as session:
                await record_job_run_db(session, job_name, "error", started_at, error=str(e))
        except Exception:
            pass  # Best-effort DB recording
        return {"status": "error", "error": str(e), "duration_ms": duration_ms}


async def sota_weather_capture_prekickoff() -> dict:
    """
    Capture weather forecasts for matches kicking off in next 48h.

    Uses Open-Meteo API (free, no key required) to fetch weather data.
    Best-effort: skips matches without venue geo data (no crash).

    Frequency: Every 60 minutes
    Guardrail: SOTA_WEATHER_ENABLED env var
    """
    import time as _time
    from datetime import datetime

    start_time = _time.time()
    started_at = datetime.utcnow()
    job_name = "sota_weather_capture"

    # Check if enabled
    if os.environ.get("SOTA_WEATHER_ENABLED", "false").lower() in ("false", "0", "no"):
        logger.debug(f"[{job_name}] Disabled via env var (set SOTA_WEATHER_ENABLED=true to enable)")
        return {"status": "disabled"}

    metrics = {
        "matches_checked": 0,
        "matches_with_geo": 0,
        "forecasts_captured": 0,
        "skipped_no_geo": 0,
        "skipped_already_captured": 0,
        "errors": 0,
        "started_at": started_at.isoformat(),
    }

    try:
        # Use centralized sota_jobs module with real OpenMeteo implementation
        from app.etl.sota_jobs import capture_weather_prekickoff
        from app.jobs.tracking import record_job_run as record_job_run_db

        async with AsyncSessionLocal() as session:
            # Run with 48h lookahead, max 100 matches, 24h horizon
            stats = await capture_weather_prekickoff(
                session,
                hours=48,
                limit=100,
                horizon=24,
            )
            metrics.update(stats)

            # Record in DB for ops dashboard fallback
            duration_ms = (_time.time() - start_time) * 1000
            status = "ok" if metrics.get("errors", 0) == 0 else "partial"
            await record_job_run_db(session, job_name, status, started_at, metrics=metrics)

        # Record in Prometheus
        record_job_run(job=job_name, status=status, duration_ms=duration_ms)

        logger.info(
            f"[{job_name}] Complete: checked={metrics['matches_checked']}, "
            f"with_geo={metrics['matches_with_geo']}, captured={metrics['forecasts_captured']}"
        )
        return {**metrics, "status": status, "duration_ms": duration_ms}

    except Exception as e:
        duration_ms = (_time.time() - start_time) * 1000
        logger.error(f"[{job_name}] Failed: {e}", exc_info=True)
        sentry_capture_exception(e, job_id=job_name)
        record_job_run(job=job_name, status="error", duration_ms=duration_ms)
        # Try to record error in DB too
        try:
            from app.jobs.tracking import record_job_run as record_job_run_db
            async with AsyncSessionLocal() as session:
                await record_job_run_db(session, job_name, "error", started_at, error=str(e))
        except Exception:
            pass  # Best-effort DB recording
        return {"status": "error", "error": str(e), "duration_ms": duration_ms}


async def sota_venue_geo_expand() -> dict:
    """
    Expand venue_geo table with coordinates for new venues.

    Finds venues from recent matches that don't have geo data yet.
    Uses geocoding API to get lat/lon/timezone.

    Frequency: Daily (or weekly)
    Guardrail: SOTA_VENUE_GEO_ENABLED env var
    """
    import time as _time
    from datetime import datetime

    start_time = _time.time()
    started_at = datetime.utcnow()
    job_name = "sota_venue_geo_expand"

    # Check if enabled (default off - requires geocoding API)
    if os.environ.get("SOTA_VENUE_GEO_ENABLED", "false").lower() in ("false", "0", "no"):
        logger.debug(f"[{job_name}] Disabled via env var (default off until geocoding API configured)")
        return {"status": "disabled"}

    metrics = {
        "venues_missing": 0,
        "venues_geocoded": 0,
        "skipped_no_city": 0,
        "errors": 0,
        "started_at": started_at.isoformat(),
    }

    try:
        # Use centralized sota_jobs module (placeholder - geocoding not yet implemented)
        from app.etl.sota_jobs import expand_venue_geo
        from app.jobs.tracking import record_job_run as record_job_run_db

        async with AsyncSessionLocal() as session:
            # Run venue geocoding (placeholder for now)
            stats = await expand_venue_geo(session, limit=50)
            metrics.update(stats)

            # Record in DB for ops dashboard fallback
            duration_ms = (_time.time() - start_time) * 1000
            status = "ok" if metrics.get("errors", 0) == 0 else "partial"
            await record_job_run_db(session, job_name, status, started_at, metrics=metrics)

        # Record in Prometheus
        record_job_run(job=job_name, status=status, duration_ms=duration_ms)

        logger.info(
            f"[{job_name}] Complete: missing={metrics['venues_missing']}, "
            f"geocoded={metrics['venues_geocoded']}"
        )
        return {**metrics, "status": status, "duration_ms": duration_ms}

    except Exception as e:
        duration_ms = (_time.time() - start_time) * 1000
        logger.error(f"[{job_name}] Failed: {e}", exc_info=True)
        sentry_capture_exception(e, job_id=job_name)
        record_job_run(job=job_name, status="error", duration_ms=duration_ms)
        # Try to record error in DB too
        try:
            from app.jobs.tracking import record_job_run as record_job_run_db
            async with AsyncSessionLocal() as session:
                await record_job_run_db(session, job_name, "error", started_at, error=str(e))
        except Exception:
            pass  # Best-effort DB recording
        return {"status": "error", "error": str(e), "duration_ms": duration_ms}


async def sota_sofascore_refs_sync() -> dict:
    """
    Sync Sofascore refs (match_external_refs) for upcoming matches.

    Links internal matches to Sofascore event IDs using deterministic matching
    based on team names and kickoff time. This enables XI capture.

    Frequency: Every 6 hours
    Guardrail: SOTA_SOFASCORE_REFS_ENABLED env var
    """
    import time as _time
    from datetime import datetime

    start_time = _time.time()
    started_at = datetime.utcnow()
    job_name = "sota_sofascore_refs_sync"

    # Check if enabled (default off)
    if os.environ.get("SOTA_SOFASCORE_REFS_ENABLED", "false").lower() in ("false", "0", "no"):
        logger.debug(f"[{job_name}] Disabled via env var (set SOTA_SOFASCORE_REFS_ENABLED=true to enable)")
        return {"status": "disabled"}

    metrics = {
        "scanned": 0,
        "already_linked": 0,
        "linked_auto": 0,
        "linked_review": 0,
        "skipped_no_candidates": 0,
        "skipped_low_score": 0,
        "errors": 0,
        "started_at": started_at.isoformat(),
    }

    try:
        from app.etl.sota_jobs import sync_sofascore_refs
        from app.jobs.tracking import record_job_run as record_job_run_db

        async with AsyncSessionLocal() as session:
            # Run with 72h lookahead, 2 days back, max 200 matches
            stats = await sync_sofascore_refs(
                session,
                hours=72,
                days_back=2,
                limit=200,
            )
            metrics.update(stats)

            # Record in DB for ops dashboard fallback
            duration_ms = (_time.time() - start_time) * 1000
            status = "ok" if metrics.get("errors", 0) == 0 else "partial"
            await record_job_run_db(session, job_name, status, started_at, metrics=metrics)

        # Record in Prometheus
        record_job_run(job=job_name, status=status, duration_ms=duration_ms)

        total_linked = metrics["linked_auto"] + metrics["linked_review"]
        logger.info(
            f"[{job_name}] Complete: scanned={metrics['scanned']}, "
            f"linked={total_linked}, already={metrics['already_linked']}"
        )
        return {**metrics, "status": status, "duration_ms": duration_ms}

    except Exception as e:
        duration_ms = (_time.time() - start_time) * 1000
        logger.error(f"[{job_name}] Failed: {e}", exc_info=True)
        sentry_capture_exception(e, job_id=job_name)
        record_job_run(job=job_name, status="error", duration_ms=duration_ms)
        # Try to record error in DB too
        try:
            from app.jobs.tracking import record_job_run as record_job_run_db
            async with AsyncSessionLocal() as session:
                await record_job_run_db(session, job_name, "error", started_at, error=str(e))
        except Exception:
            pass  # Best-effort DB recording
        return {"status": "error", "error": str(e), "duration_ms": duration_ms}


async def sota_sofascore_xi_capture() -> dict:
    """
    Capture Sofascore XI (lineup/formation/ratings) for upcoming matches.

    Fetches pre-match XI data from Sofascore API for matches in supported leagues.
    Best-effort: skips matches without sofascore ref (no crash).

    Frequency: Every 30 minutes
    Guardrail: SOTA_SOFASCORE_ENABLED env var

    PIT safety: Only captures data with captured_at < kickoff_utc.
    """
    import time as _time
    from datetime import datetime

    start_time = _time.time()
    started_at = datetime.utcnow()
    job_name = "sota_sofascore_xi_capture"

    # Check if enabled (default off - scraping requires careful rate limiting)
    if os.environ.get("SOTA_SOFASCORE_ENABLED", "false").lower() in ("false", "0", "no"):
        logger.debug(f"[{job_name}] Disabled via env var (set SOTA_SOFASCORE_ENABLED=true to enable)")
        return {"status": "disabled"}

    metrics = {
        "matches_checked": 0,
        "with_ref": 0,
        "captured": 0,
        "skipped_no_ref": 0,
        "skipped_no_data": 0,
        "errors": 0,
        "started_at": started_at.isoformat(),
    }

    try:
        from app.etl.sota_jobs import capture_sofascore_xi_prekickoff
        from app.jobs.tracking import record_job_run as record_job_run_db

        async with AsyncSessionLocal() as session:
            # Run with 48h lookahead, max 100 matches
            stats = await capture_sofascore_xi_prekickoff(
                session,
                hours=48,
                limit=100,
            )
            metrics.update(stats)

            # Record in DB for ops dashboard fallback
            duration_ms = (_time.time() - start_time) * 1000
            status = "ok" if metrics.get("errors", 0) == 0 else "partial"
            await record_job_run_db(session, job_name, status, started_at, metrics=metrics)

        # Record in Prometheus
        record_job_run(job=job_name, status=status, duration_ms=duration_ms)

        logger.info(
            f"[{job_name}] Complete: checked={metrics['matches_checked']}, "
            f"with_ref={metrics['with_ref']}, captured={metrics['captured']}"
        )
        return {**metrics, "status": status, "duration_ms": duration_ms}

    except Exception as e:
        duration_ms = (_time.time() - start_time) * 1000
        logger.error(f"[{job_name}] Failed: {e}", exc_info=True)
        sentry_capture_exception(e, job_id=job_name)
        record_job_run(job=job_name, status="error", duration_ms=duration_ms)
        # Try to record error in DB too
        try:
            from app.jobs.tracking import record_job_run as record_job_run_db
            async with AsyncSessionLocal() as session:
                await record_job_run_db(session, job_name, "error", started_at, error=str(e))
        except Exception:
            pass  # Best-effort DB recording
        return {"status": "error", "error": str(e), "duration_ms": duration_ms}


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

    # Window Sync: Daily at 05:30 UTC (BEFORE other jobs)
    # Loads fixtures for next 10 days using date-based API calls.
    # Solves LATAM 2026 issue: works regardless of CURRENT_SEASON setting.
    scheduler.add_job(
        global_sync_window,
        trigger=CronTrigger(hour=5, minute=30),
        id="global_sync_window",
        name="Daily Window Sync (10 days ahead)",
        replace_existing=True,
    )

    # Daily results sync: Every day at 6:00 AM UTC (first job of the day)
    scheduler.add_job(
        daily_sync_results,
        trigger=CronTrigger(hour=6, minute=0),
        id="daily_sync_results",
        name="Daily Results Sync",
        replace_existing=True,
    )

    # Daily aggregates refresh: Every day at 6:30 AM UTC (after results sync)
    # Computes league baselines and team profiles for narrative context
    scheduler.add_job(
        daily_refresh_aggregates,
        trigger=CronTrigger(hour=6, minute=30),
        id="daily_refresh_aggregates",
        name="Daily Aggregates Refresh (league baselines + team profiles)",
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

    # Prediction save safety net: Every 6 hours to catch missed daily runs
    # This ensures predictions are generated even if deploys interrupt the 7:00 UTC job
    scheduler.add_job(
        daily_save_predictions,
        trigger=IntervalTrigger(hours=6),
        id="predictions_safety_net",
        name="Predictions Safety Net (every 6h)",
        replace_existing=True,
    )

    # Prediction gap safety net: Every 30 minutes to catch isolated matches
    # Detects NS matches in next 2h without predictions and generates them
    # Covers midweek isolated matches that fall between daily batch runs
    scheduler.add_job(
        prediction_gap_safety_net,
        trigger=IntervalTrigger(minutes=30),
        id="prediction_gap_safety_net",
        name="Prediction Gap Safety Net (every 30min)",
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

    # Shadow predictions evaluation: Every 30 minutes
    # Updates shadow_predictions with actual outcomes for A/B comparison
    scheduler.add_job(
        evaluate_shadow_predictions,
        trigger=IntervalTrigger(minutes=30),
        id="evaluate_shadow_predictions",
        name="Shadow Predictions Evaluation (every 30min)",
        replace_existing=True,
    )

    # Sensor B retrain: Every 6 hours (configurable via SENSOR_RETRAIN_INTERVAL_HOURS)
    # LogReg L2 calibration diagnostics - INTERNAL ONLY, never affects production
    from app.config import get_settings
    sensor_settings = get_settings()
    if sensor_settings.SENSOR_ENABLED:
        scheduler.add_job(
            retrain_sensor_model,
            trigger=IntervalTrigger(hours=sensor_settings.SENSOR_RETRAIN_INTERVAL_HOURS),
            id="retrain_sensor_model",
            name=f"Sensor B Retrain (every {sensor_settings.SENSOR_RETRAIN_INTERVAL_HOURS}h)",
            replace_existing=True,
        )

        # Sensor B evaluation: Every 30 minutes (same as shadow)
        scheduler.add_job(
            evaluate_sensor_predictions_job,
            trigger=IntervalTrigger(minutes=30),
            id="evaluate_sensor_predictions",
            name="Sensor B Evaluation (every 30min)",
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

    # Live Tick: Every 10 seconds (batch update for live matches)
    # Only runs if there are matches in progress (status IN LIVE_STATUSES)
    # Uses batch API calls: 1-20 matches = 1 req, 21-40 = 2 req, etc.
    # Budget (ULTRA 75K/day): ~6 req/min peak = ~3K req/day (4% of limit)
    scheduler.add_job(
        live_tick,
        trigger=IntervalTrigger(seconds=10),
        id="live_tick",
        name="Live Tick (every 10s, batch update)",
        replace_existing=True,
    )

    # Lineup Monitoring - ADAPTIVE FREQUENCY SYSTEM (2026-01-07)
    # Two jobs working together to maximize capture in ideal window:

    # Job 1: CRITICAL WINDOW (45-90 min) - Aggressive 60s polling
    # This catches lineups right when they're announced (typically 60-70 min before kickoff)
    # Only processes matches in the critical window to minimize API calls
    scheduler.add_job(
        monitor_lineups_and_capture_odds,
        trigger=IntervalTrigger(seconds=60),
        kwargs={"critical_window_only": True},
        id="lineup_monitoring_critical",
        name="Lineup Monitoring CRITICAL (every 60s, 45-90min window)",
        replace_existing=True,
    )

    # Job 2: FULL WINDOW (0-120 min) - Regular 2 min polling
    # This catches any lineups we might have missed and handles early/late announcements
    scheduler.add_job(
        monitor_lineups_and_capture_odds,
        trigger=IntervalTrigger(minutes=2),
        kwargs={"critical_window_only": False},
        id="lineup_monitoring_full",
        name="Lineup Monitoring FULL (every 2 min, 0-120min window)",
        replace_existing=True,
    )

    # Job 3: MARKET MOVEMENT - Track odds at T-60, T-30, T-15, T-5
    # For matches with confirmed lineups, capture odds at predefined time points
    # This helps measure if the market moves before/after lineup announcement
    scheduler.add_job(
        capture_market_movement_snapshots,
        trigger=IntervalTrigger(minutes=5),
        id="market_movement_tracking",
        name="Market Movement Tracking (every 5 min)",
        replace_existing=True,
    )

    # Job 4: LINEUP-RELATIVE MOVEMENT (Auditor Critical Fix 2026-01-09)
    # Track odds RELATIVE to lineup_detected_at, not just pre-kickoff
    # Snapshots: L-30, L-15, L-5, L0, L+5, L+10 (minutes from lineup detection)
    # Uses normalized probabilities for delta_p metric
    scheduler.add_job(
        capture_lineup_relative_movement,
        trigger=IntervalTrigger(minutes=3),
        id="lineup_relative_movement",
        name="Lineup-Relative Movement (every 3 min)",
        replace_existing=True,
    )

    # Daily PIT Evaluation: 9:00 AM UTC (after audit, saves JSON silently)
    # Part of Protocol v2.1 - does NOT produce daily spam
    scheduler.add_job(
        daily_pit_evaluation,
        trigger=CronTrigger(hour=9, minute=0),
        id="daily_pit_evaluation",
        name="Daily PIT Evaluation (silent save)",
        replace_existing=True,
    )

    # Finished Match Stats Backfill: Every 60 minutes
    # Fetches detailed statistics (possession, shots, corners, cards) for finished matches
    # Guardrails: STATS_BACKFILL_ENABLED, STATS_BACKFILL_MAX_CALLS_PER_RUN (200), lookback 72h
    scheduler.add_job(
        capture_finished_match_stats,
        trigger=IntervalTrigger(minutes=60),
        id="finished_match_stats_backfill",
        name="Finished Match Stats Backfill (every 60 min)",
        replace_existing=True,
    )

    # Stats Refresh: Every 2 hours, re-fetch stats for recently finished matches
    # Captures late events (red cards, late goals) missed by live sync
    # Guardrails: STATS_REFRESH_ENABLED, lookback 6h, max 50 calls/run (~600/day)
    scheduler.add_job(
        refresh_recent_ft_stats,
        trigger=IntervalTrigger(hours=2),
        id="stats_refresh_recent",
        name="Stats Refresh Recent FT (every 2h)",
        replace_existing=True,
    )

    # Odds Sync: Every 6 hours (configurable via ODDS_SYNC_INTERVAL_HOURS)
    # Fetches 1X2 odds for upcoming NS matches in 48h window
    # Guardrails: ODDS_SYNC_ENABLED, ODDS_SYNC_MAX_FIXTURES (100), freshness 6h
    # Budget: ~250-400 requests/day (well within Pro plan 7,500/day)
    _odds_settings = get_settings()
    if _odds_settings.ODDS_SYNC_ENABLED:
        scheduler.add_job(
            sync_odds_for_upcoming_matches,
            trigger=IntervalTrigger(hours=_odds_settings.ODDS_SYNC_INTERVAL_HOURS),
            id="odds_sync_upcoming",
            name=f"Odds Sync (every {_odds_settings.ODDS_SYNC_INTERVAL_HOURS}h)",
            replace_existing=True,
        )

    # Fast-Path LLM Narratives: Every 2 minutes (configurable via FASTPATH_INTERVAL_SECONDS)
    # Generates narratives within minutes of match completion instead of daily audit
    # Guardrails: FASTPATH_ENABLED, FASTPATH_MAX_CONCURRENT_JOBS (10), lookback 90 min
    from app.config import get_settings
    _fp_settings = get_settings()
    scheduler.add_job(
        fast_postmatch_narratives,
        trigger=IntervalTrigger(seconds=_fp_settings.FASTPATH_INTERVAL_SECONDS),
        id="fast_postmatch_narratives",
        name=f"Fast-Path Narratives (every {_fp_settings.FASTPATH_INTERVAL_SECONDS}s)",
        replace_existing=True,
    )

    # Weekly PIT Report: Tuesdays 10:00 AM UTC (consolidated weekly report)
    # The ONLY PIT report that gets logged/published - not daily spam
    # Tuesday chosen to include full weekend football data
    scheduler.add_job(
        weekly_pit_report,
        trigger=CronTrigger(day_of_week="tue", hour=10, minute=0),
        id="weekly_pit_report",
        name="Weekly PIT Report (consolidated)",
        replace_existing=True,
    )

    # Daily Ops Rollup: 09:05 UTC (after PIT evaluation)
    # Aggregates daily KPIs to ops_daily_rollups table (idempotent UPSERT)
    scheduler.add_job(
        daily_ops_rollup,
        trigger=CronTrigger(hour=9, minute=5),
        id="daily_ops_rollup",
        name="Daily Ops Rollup (KPI aggregation)",
        replace_existing=True,
    )

    # Daily Alpha Progress Snapshot: 09:10 UTC (after ops rollup)
    # Captures progress towards Re-test/Alpha for auditing
    scheduler.add_job(
        daily_alpha_progress_snapshot,
        trigger=CronTrigger(hour=9, minute=10),
        id="daily_alpha_progress_snapshot",
        name="Daily Alpha Progress Snapshot",
        replace_existing=True,
    )

    # Daily Predictions Performance Report: 09:15 UTC (after alpha snapshot)
    # Generates 7d and 14d performance reports with Brier, log loss, calibration
    # Allows distinguishing variance from bugs
    scheduler.add_job(
        daily_predictions_performance_report,
        trigger=CronTrigger(hour=9, minute=15),
        id="daily_predictions_performance_report",
        name="Daily Predictions Performance Report (7d/14d)",
        replace_existing=True,
    )

    # Monthly PIT Reports Retention: 1st of month at 04:00 UTC
    # Deletes old reports: daily > 180 days, weekly > 365 days
    scheduler.add_job(
        pit_reports_retention,
        trigger=CronTrigger(day=1, hour=4, minute=0),
        id="pit_reports_retention",
        name="Monthly PIT Reports Retention",
        replace_existing=True,
    )

    # Weekly LLM Raw Output Cleanup: Sundays at 05:00 UTC
    # Clears llm_output_raw after 14 days to save space (keeps other traceability)
    scheduler.add_job(
        llm_raw_output_cleanup,
        trigger=CronTrigger(day_of_week="sun", hour=5, minute=0),
        id="llm_raw_output_cleanup",
        name="Weekly LLM Raw Output Cleanup (14d TTL)",
        replace_existing=True,
    )

    # Scheduler Heartbeat: Every 30 minutes, log registered jobs and next run times
    scheduler.add_job(
        scheduler_heartbeat,
        trigger=IntervalTrigger(minutes=30),
        id="scheduler_heartbeat",
        name="Scheduler Heartbeat (every 30 min)",
        replace_existing=True,
    )

    # Predictions Health Metrics: Every 5 minutes, update Prometheus gauges for alerting
    # Ensures metrics are fresh even if nobody opens /dashboard/ops.json
    scheduler.add_job(
        update_predictions_health_metrics,
        trigger=IntervalTrigger(minutes=5),
        id="predictions_health_metrics",
        name="Predictions Health Metrics (every 5 min)",
        replace_existing=True,
    )

    # =========================================================================
    # SOTA ENRICHMENT JOBS (Understat xG, Weather, Venue Geo)
    # All disabled by default except Understat jobs (core SOTA pipeline)
    # =========================================================================

    # SOTA: Understat refs sync - every 12 hours
    # Links matches to Understat IDs for xG retrieval
    # NOTE: next_run_time ensures job runs immediately on startup
    scheduler.add_job(
        sota_understat_refs_sync,
        trigger=IntervalTrigger(hours=12),
        id="sota_understat_refs_sync",
        name="SOTA Understat Refs Sync (every 12h)",
        replace_existing=True,
        next_run_time=datetime.utcnow(),
    )

    # SOTA: Understat xG backfill - every 6 hours
    # Fetches actual xG data for matches with refs
    # NOTE: next_run_time ensures job runs immediately on startup
    scheduler.add_job(
        sota_understat_ft_backfill,
        trigger=IntervalTrigger(hours=6),
        id="sota_understat_ft_backfill",
        name="SOTA Understat xG Backfill (every 6h)",
        replace_existing=True,
        next_run_time=datetime.utcnow(),
    )

    # SOTA: Weather capture - every 60 minutes
    # Captures weather forecasts for upcoming matches (disabled by default)
    # NOTE: next_run_time ensures job runs immediately on startup
    scheduler.add_job(
        sota_weather_capture_prekickoff,
        trigger=IntervalTrigger(minutes=60),
        id="sota_weather_capture",
        name="SOTA Weather Capture (every 60 min)",
        replace_existing=True,
        next_run_time=datetime.utcnow(),
    )

    # SOTA: Venue geo expand - daily at 03:00 UTC
    # Geocodes new venues (disabled by default)
    scheduler.add_job(
        sota_venue_geo_expand,
        trigger=CronTrigger(hour=3, minute=0),
        id="sota_venue_geo_expand",
        name="SOTA Venue Geo Expand (daily 03:00 UTC)",
        replace_existing=True,
    )

    # SOTA: Sofascore refs sync - every 6 hours
    # Links matches to Sofascore event IDs (disabled by default)
    # NOTE: next_run_time ensures job runs immediately on startup, then every 6h
    scheduler.add_job(
        sota_sofascore_refs_sync,
        trigger=IntervalTrigger(hours=6),
        id="sota_sofascore_refs_sync",
        name="SOTA Sofascore Refs Sync (every 6h)",
        replace_existing=True,
        next_run_time=datetime.utcnow(),
    )

    # SOTA: Sofascore XI capture - every 30 minutes
    # Captures lineup/formation/ratings for upcoming matches (disabled by default)
    # NOTE: next_run_time ensures job runs immediately on startup, then every 30min
    scheduler.add_job(
        sota_sofascore_xi_capture,
        trigger=IntervalTrigger(minutes=30),
        id="sota_sofascore_xi_capture",
        name="SOTA Sofascore XI Capture (every 30 min)",
        replace_existing=True,
        next_run_time=datetime.utcnow(),
    )

    scheduler.start()
    _scheduler_started = True

    # Log initial heartbeat immediately after start
    _log_scheduler_jobs()

    logger.info(
        f"Scheduler started:\n"
        f"  - Live global sync: Every 60 seconds\n"
        f"  - Lineup monitoring CRITICAL: Every 60s (45-90 min window - aggressive)\n"
        f"  - Lineup monitoring FULL: Every 2 min (0-120 min window - backup)\n"
        f"  - Market movement tracking: Every 5 min (T-60/T-30/T-15/T-5 pre-kickoff)\n"
        f"  - Lineup-relative movement: Every 3 min (L-30 to L+10 around lineup)\n"
        f"  - Finished match stats backfill: Every 60 min\n"
        f"  - Odds sync: Every {_odds_settings.ODDS_SYNC_INTERVAL_HOURS}h (1X2 for NS matches)\n"
        f"  - Fast-path narratives: Every {_fp_settings.FASTPATH_INTERVAL_SECONDS}s (post-match LLM)\n"
        f"  - Daily results sync: 6:00 AM UTC\n"
        f"  - Daily save predictions: 7:00 AM UTC\n"
        f"  - Daily audit: 8:00 AM UTC\n"
        f"  - Daily PIT evaluation: 9:00 AM UTC (silent save)\n"
        f"  - Daily ops rollup: 9:05 AM UTC (KPI aggregation)\n"
        f"  - Daily Alpha Progress snapshot: 9:10 AM UTC\n"
        f"  - Weekly recalibration: Mondays 5:00 AM UTC\n"
        f"  - Weekly PIT report: Tuesdays 10:00 AM UTC\n"
        f"  - Monthly PIT retention: 1st of month 04:00 UTC\n"
        f"  - Scheduler heartbeat: Every 30 min (logs job status)\n"
        f"  - SOTA Understat refs sync: Every 12h\n"
        f"  - SOTA Understat xG backfill: Every 6h\n"
        f"  - SOTA Weather capture: Every 60 min (disabled by default)\n"
        f"  - SOTA Venue geo expand: Daily 03:00 UTC (disabled by default)\n"
        f"  - SOTA Sofascore XI capture: Every 30 min (disabled by default)"
    )


def stop_scheduler():
    """Stop the background scheduler."""
    global _scheduler_started
    if scheduler.running:
        scheduler.shutdown()
        _scheduler_started = False
        logger.info("Scheduler stopped")
