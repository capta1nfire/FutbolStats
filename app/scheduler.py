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
    # ATI: Ext shadow observability
    record_ext_shadow_rejection,
)

# Sentry context for job error tracking
from app.telemetry.sentry import sentry_job_context, capture_exception as sentry_capture_exception

logger = logging.getLogger(__name__)

# Top 5 European leagues (core) — imported from single source of truth
from app.etl.sota_constants import UNDERSTAT_SUPPORTED_LEAGUES
TOP5_LEAGUES = list(UNDERSTAT_SUPPORTED_LEAGUES)

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
            # Find NS matches in next 30 minutes WITHOUT any prediction.
            #
            # NOTE (2026-02): Keep this query extremely lightweight.
            # - Avoid JOIN teams (not needed for gating)
            # - Use parameters instead of NOW() to help planner/cache stability
            # - LIMIT to bound work under contention
            now = datetime.utcnow()
            end = now + timedelta(minutes=30)
            max_candidates = 50

            try:
                result = await session.execute(
                    text("""
                        SELECT m.id
                        FROM matches m
                        LEFT JOIN predictions p ON p.match_id = m.id
                        WHERE m.status = 'NS'
                          AND m.date > :now
                          AND m.date <= :end
                          AND p.match_id IS NULL
                        ORDER BY m.date ASC
                        LIMIT :limit
                    """),
                    {"now": now, "end": end, "limit": max_candidates},
                )
                match_ids = [row[0] for row in result.fetchall()]
            except Exception as e:
                # Avoid noisy Sentry errors on transient DB contention.
                # When Postgres cancels the statement due to statement_timeout,
                # SQLAlchemy wraps asyncpg.exceptions.QueryCanceledError.
                msg = str(e).lower()
                if "statement timeout" in msg or "querycancelederror" in msg:
                    duration_ms = (time.time() - start_time) * 1000
                    logger.warning(
                        "[KICKOFF-SAFETY] Query timed out (statement_timeout); skipping this run",
                        extra={"duration_ms": int(duration_ms)},
                    )
                    record_job_run(job="kickoff_safety_net", status="error", duration_ms=duration_ms)
                    return {"generated": 0, "error": "statement_timeout"}
                raise

            if not match_ids:
                return {"generated": 0, "message": "no_imminent_gaps"}

            # Fetch match context for logging only (small IN-list)
            rows = await session.execute(
                text("""
                    SELECT m.id, m.external_id, m.date, m.league_id,
                           ht.name as home_team, at.name as away_team
                    FROM matches m
                    JOIN teams ht ON ht.id = m.home_team_id
                    JOIN teams at ON at.id = m.away_team_id
                    WHERE m.id = ANY(:match_ids)
                    ORDER BY m.date ASC
                """),
                {"match_ids": match_ids},
            )
            imminent_gaps = rows.fetchall()

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

            # Get features ONLY for imminent matches (O(k) not O(N))
            match_ids = [g[0] for g in imminent_gaps]
            feature_engineer = FeatureEngineer(session=session)
            df = await feature_engineer.get_matches_features_by_ids(
                match_ids, league_only=True, statuses=["NS"]
            )

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

    OPTIMIZATION (2026-01-25):
    - Filter in SQL (not Python) to avoid loading thousands of rows
    - Batch processing with LIMIT to prevent memory spikes
    - Only fetch IDs first, then process in batches
    """
    import time as time_module
    from app.models import Match, Prediction
    from app.database import get_session_with_retry

    start_time = time_module.time()
    frozen_count = 0
    errors = []
    batches_processed = 0
    batch_size = 500  # Process up to 500 predictions per batch

    try:
        async with get_session_with_retry(max_retries=3, retry_delay=1.0) as session:
            now = datetime.utcnow()

            # STEP 1: Count total not frozen (for observability only, lightweight)
            count_result = await session.execute(
                text("SELECT COUNT(*) FROM predictions WHERE is_frozen = FALSE")
            )
            not_frozen_total = count_result.scalar() or 0

            # STEP 2: Count eligible to freeze (for observability, before processing)
            eligible_count_result = await session.execute(
                text("""
                    SELECT COUNT(*)
                    FROM predictions p
                    JOIN matches m ON m.id = p.match_id
                    WHERE p.is_frozen = FALSE
                      AND (m.status != 'NS' OR m.date < :now)
                """),
                {"now": now},
            )
            eligible_total = eligible_count_result.scalar() or 0

            # Early return if nothing to freeze (avoid loading any objects)
            if eligible_total == 0:
                duration_ms = (time_module.time() - start_time) * 1000
                logger.info(
                    f"[FREEZE] Complete: not_frozen_total={not_frozen_total}, "
                    f"eligible_total=0, frozen_count=0, batches=0, duration_ms={duration_ms:.0f}"
                )
                return {
                    "frozen_count": 0,
                    "not_frozen_total": not_frozen_total,
                    "eligible_total": 0,
                    "batches_processed": 0,
                    "duration_ms": round(duration_ms),
                }

            # STEP 3: Find predictions eligible to freeze using SQL filter
            # Only fetch IDs to minimize memory, ordered by match date for determinism
            eligible_query = text("""
                SELECT p.id
                FROM predictions p
                JOIN matches m ON m.id = p.match_id
                WHERE p.is_frozen = FALSE
                  AND (m.status != 'NS' OR m.date < :now)
                ORDER BY m.date ASC
                LIMIT :limit
            """)

            # Process in batches until no more eligible
            while True:
                result = await session.execute(
                    eligible_query, {"now": now, "limit": batch_size}
                )
                prediction_ids = [row[0] for row in result.fetchall()]

                if not prediction_ids:
                    break  # No more to process

                batches_processed += 1

                # Fetch full prediction+match objects for this batch only
                batch_result = await session.execute(
                    select(Prediction)
                    .options(selectinload(Prediction.match))
                    .where(Prediction.id.in_(prediction_ids))
                )
                predictions = batch_result.scalars().all()

                for pred in predictions:
                    match = pred.match
                    if not match:
                        continue

                    # Double-check eligibility (idempotent guard)
                    if pred.is_frozen:
                        continue

                    try:
                        # Freeze the prediction with current data
                        pred.is_frozen = True
                        pred.frozen_at = now

                        # Capture bookmaker odds at freeze time
                        pred.frozen_odds_home = match.odds_home
                        pred.frozen_odds_draw = match.odds_draw
                        pred.frozen_odds_away = match.odds_away

                        # OPS Daily Comparison: persist a "market snapshot" row for this match
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
                            # Don't block freezing if OPS snapshot fails
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

                        # Calculate and freeze value bets
                        value_bets = []
                        ev_threshold = 0.05  # 5% EV minimum for value bet

                        def _build_value_bet(outcome: str, prob: float, odds: float, ev: float) -> dict:
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

                # Commit after each batch
                await session.commit()

            duration_ms = (time_module.time() - start_time) * 1000

            # Observability: log metrics
            logger.info(
                f"[FREEZE] Complete: not_frozen_total={not_frozen_total}, "
                f"eligible_total={eligible_total}, frozen_count={frozen_count}, "
                f"batches={batches_processed}, duration_ms={duration_ms:.0f}"
            )

            return {
                "frozen_count": frozen_count,
                "not_frozen_total": not_frozen_total,
                "eligible_total": eligible_total,
                "batches_processed": batches_processed,
                "duration_ms": round(duration_ms),
                "errors": errors[:10] if errors else None,
            }

    except Exception as e:
        import traceback
        error_type = type(e).__name__
        error_msg = str(e)
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


def _bridge_sofascore_to_lineup_dict(sc_lineup) -> dict | None:
    """Convert SofascoreMatchLineup to API-Football lineup dict format.

    Returns None if data is incomplete or malformed.
    Bridge needed because downstream code expects API-Football dict structure.
    """
    if not sc_lineup.home or not sc_lineup.away:
        return None

    def _safe_player_id(player_id_ext: str) -> int:
        try:
            return int(player_id_ext)
        except (ValueError, TypeError):
            return 0

    def _side_to_dict(side) -> dict:
        starters = [p for p in side.players if p.is_starter]
        subs = [p for p in side.players if not p.is_starter]
        return {
            "formation": side.formation,
            "starting_xi": [
                {"id": _safe_player_id(p.player_id_ext), "name": p.name or "", "pos": p.position}
                for p in starters
            ],
            "substitutes": [
                {"id": _safe_player_id(p.player_id_ext), "name": p.name or "", "pos": p.position}
                for p in subs
            ],
        }

    return {
        "home": _side_to_dict(sc_lineup.home),
        "away": _side_to_dict(sc_lineup.away),
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

    DEADLOCK FIX (2026-02-15): Refactored to short per-match transactions.
    - Phase 1: Read candidates into memory (read-only session, closed before API calls).
    - Phase 2: For each candidate, API calls (lineups + odds) OUTSIDE any DB transaction,
      then short tx with FOR NO KEY UPDATE lock ordering + deadlock retry.

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
    lineup_calls = 0
    odds_calls = 0
    errors = []

    try:
        # --- Phase 1: Read candidates into memory ---
        async with AsyncSessionLocal() as session:
            now = datetime.utcnow()
            try:
                league_ids = await resolve_lineup_monitoring_leagues(session)
            except Exception as e:
                logger.warning(f"Could not resolve tracked leagues; falling back to EXTENDED_LEAGUES: {e}")
                league_ids = EXTENDED_LEAGUES

            if critical_window_only:
                window_start = now + timedelta(minutes=45)
                window_end = now + timedelta(minutes=90)
                log_prefix = "[CRITICAL]"
            else:
                window_start = now - timedelta(minutes=5)
                window_end = now + timedelta(minutes=120)
                log_prefix = "[FULL]"

            result = await session.execute(text("""
                SELECT m.id, m.external_id, m.date, m.odds_home, m.odds_draw, m.odds_away,
                       m.status, m.league_id,
                       mer.source_match_id AS sofascore_id,
                       EXTRACT(EPOCH FROM (m.date - NOW())) / 60 as minutes_to_kickoff
                FROM matches m
                LEFT JOIN match_external_refs mer
                  ON m.id = mer.match_id AND mer.source = 'sofascore'
                WHERE m.date BETWEEN :window_start AND :window_end
                  AND m.status = 'NS'
                  AND (:league_ids_is_null = TRUE OR m.league_id = ANY(:league_ids))
                  AND NOT EXISTS (
                      SELECT 1 FROM odds_snapshots os
                      WHERE os.match_id = m.id
                        AND os.snapshot_type = 'lineup_confirmed'
                  )
                ORDER BY
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

            if matches:
                in_sweet_spot = sum(1 for m in matches if 60 <= (m.minutes_to_kickoff or 0) <= 80)
                in_ideal = sum(1 for m in matches if 45 <= (m.minutes_to_kickoff or 0) <= 75)
                logger.info(
                    f"{log_prefix} Lineup monitor: {len(matches)} matches, "
                    f"{in_sweet_spot} in sweet spot (60-80), {in_ideal} in ideal (45-75)"
                )

            max_lineup_checks = LINEUP_MAX_LINEUPS_PER_RUN_CRITICAL if critical_window_only else LINEUP_MAX_LINEUPS_PER_RUN_FULL
            if len(matches) > max_lineup_checks:
                logger.warning(
                    f"{log_prefix} Too many matches ({len(matches)}), processing first {max_lineup_checks} "
                    f"(prioritized by sweet spot). Remaining will be processed in next run."
                )
                matches = matches[:max_lineup_checks]
        # --- Session closed: no DB locks held during API calls ---

        if not matches:
            return {"checked": 0, "captured": 0, "message": f"{log_prefix} No matches in window"}

        # --- Phase 2: API calls + short tx per match ---
        provider = APIFootballProvider()
        sofascore_provider = None
        try:
            # Lazy-init Sofascore provider only if any candidate has sofascore_id
            if any(getattr(m, 'sofascore_id', None) for m in matches):
                from app.etl.sofascore_provider import SofascoreProvider, SofascoreMatchLineup, SofascoreLineupData
                sofascore_provider = SofascoreProvider()

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

                    if lineup_calls >= max_lineup_checks:
                        break

                    capture_start_time = datetime.utcnow()

                    # --- Dual-source lineup fetch (API-Football + Sofascore) ---
                    # ABE guardrail: asyncio.gather(return_exceptions=True)
                    # Sofascore NEVER blocks API-Football

                    async def _fetch_apifootball(_ext_id):
                        """Fetch lineup from API-Football with retries."""
                        _max_retries = 3
                        for _attempt in range(_max_retries):
                            try:
                                data = await provider.get_lineups(_ext_id)
                                return data
                            except Exception as _e:
                                error_str = str(_e).lower()
                                if "429" in error_str or "rate limit" in error_str:
                                    _lineup_capture_metrics[job_type]["api_errors_429"] += 1
                                elif "timeout" in error_str or "timed out" in error_str:
                                    _lineup_capture_metrics[job_type]["api_errors_timeout"] += 1
                                else:
                                    _lineup_capture_metrics[job_type]["api_errors_other"] += 1
                                if _attempt == _max_retries - 1:
                                    return None
                                await asyncio.sleep(2 ** _attempt)
                        return None

                    async def _fetch_sofascore(_sof_id):
                        """Fetch lineup from Sofascore via IPRoyal proxy.

                        ABE guardrail: NO country_code — pool genérico DE.
                        Timeout 5s, max 1 retry (configured in SofascoreProvider).
                        """
                        try:
                            result = await sofascore_provider.get_match_lineup(str(_sof_id))
                            if result.error or result.integrity_score < 0.6:
                                return None
                            return _bridge_sofascore_to_lineup_dict(result)
                        except Exception:
                            return None

                    sofascore_id = getattr(match, 'sofascore_id', None)
                    tasks = [_fetch_apifootball(external_id)]
                    if sofascore_id and sofascore_provider:
                        tasks.append(_fetch_sofascore(sofascore_id))

                    fetch_results = await asyncio.gather(*tasks, return_exceptions=True)
                    lineup_calls += 1

                    # First Valid Wins: pick first result with 11+11 valid starters
                    lineup_data = None
                    lineup_source = "api-football"
                    for _i, _res in enumerate(fetch_results):
                        if isinstance(_res, Exception) or _res is None:
                            continue
                        _hxi = _res.get("home", {}).get("starting_xi", [])
                        _axi = _res.get("away", {}).get("starting_xi", [])
                        if len(_hxi) == 11 and len(_axi) == 11:
                            # Kimi hardening: reject ghost IDs
                            if any(p.get("id") == 0 for p in _hxi + _axi):
                                continue
                            lineup_data = _res
                            lineup_source = "api-football" if _i == 0 else "sofascore"
                            break

                    if not lineup_data:
                        if external_id:
                            _lineup_check_cooldown[int(external_id)] = datetime.utcnow()
                        continue

                    home_lineup = lineup_data.get("home")
                    away_lineup = lineup_data.get("away")

                    if not home_lineup or not away_lineup:
                        if external_id:
                            _lineup_check_cooldown[int(external_id)] = datetime.utcnow()
                        continue

                    home_xi = home_lineup.get("starting_xi", [])
                    away_xi = away_lineup.get("starting_xi", [])

                    if 8 <= len(home_xi) < 11 or 8 <= len(away_xi) < 11:
                        minutes_to_ko = (match.date - datetime.utcnow()).total_seconds() / 60 if match.date else 0
                        logger.info(
                            f"PARTIAL_LINEUP: match_id={match_id} external={external_id} "
                            f"home={len(home_xi)}/11 away={len(away_xi)}/11 "
                            f"minutes_to_kickoff={minutes_to_ko:.1f}"
                        )

                    if len(home_xi) < 11 or len(away_xi) < 11:
                        if external_id:
                            _lineup_check_cooldown[int(external_id)] = datetime.utcnow()
                        continue

                    if any(p is None for p in home_xi) or any(p is None for p in away_xi):
                        logger.warning(
                            f"Invalid player IDs (None values) in lineup for match {match_id}. "
                            f"Skipping to avoid data quality issues."
                        )
                        continue

                    # LINEUP CONFIRMED! Now capture the odds
                    logger.info(f"Lineup confirmed for match {match_id} (external: {external_id}, provider: {lineup_source})")

                    kickoff_time = match.date
                    lineup_detected_at = datetime.utcnow()

                    if odds_calls >= LINEUP_MAX_ODDS_PER_RUN:
                        logger.info(f"{log_prefix} Odds cap reached ({LINEUP_MAX_ODDS_PER_RUN}), deferring odds capture.")
                        continue

                    # API call: get odds (OUTSIDE any DB transaction)
                    fresh_odds = await provider.get_odds(external_id)
                    odds_calls += 1

                    if not (fresh_odds and fresh_odds.get("odds_home")):
                        logger.warning(
                            f"Cannot capture fresh odds for match {match_id} "
                            f"(external: {external_id}) - API returned: {fresh_odds}. "
                            f"Skipping this match. Will retry in next run."
                        )
                        continue

                    odds_home = float(fresh_odds["odds_home"])
                    odds_draw = float(fresh_odds["odds_draw"])
                    odds_away = float(fresh_odds["odds_away"])
                    bookmaker_name = fresh_odds.get("bookmaker", "unknown")
                    source = f"{bookmaker_name}_live"
                    logger.info(f"Got FRESH odds from {bookmaker_name} for match {match_id}")

                    if not (odds_home > 1 and odds_draw > 1 and odds_away > 1):
                        logger.warning(f"Invalid odds for match {match_id}: {odds_home}, {odds_draw}, {odds_away}")
                        continue

                    raw_home = 1 / odds_home
                    raw_draw = 1 / odds_draw
                    raw_away = 1 / odds_away
                    total = raw_home + raw_draw + raw_away
                    overround = total - 1
                    prob_home = raw_home / total
                    prob_draw = raw_draw / total
                    prob_away = raw_away / total

                    snapshot_at = datetime.utcnow()

                    if kickoff_time and snapshot_at >= kickoff_time:
                        logger.warning(
                            f"Snapshot AFTER kickoff for match {match_id}: "
                            f"snapshot_at={snapshot_at}, kickoff={kickoff_time}. Skipping."
                        )
                        continue

                    delta_to_kickoff = None
                    if kickoff_time:
                        delta_to_kickoff = int((kickoff_time - snapshot_at).total_seconds())
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

                    odds_freshness = "live" if "_live" in source else ("stale" if "_stale" in source else "unknown")

                    # --- Short tx: all DB writes for this match ---
                    home_formation = home_lineup.get("formation")
                    away_formation = away_lineup.get("formation")

                    async def _write_lineup_snapshot(
                        _mid=match_id, _sat=snapshot_at,
                        _oh=odds_home, _od=odds_draw, _oa=odds_away,
                        _ph=prob_home, _pd=prob_draw, _pa=prob_away,
                        _ov=overround, _src=source, _kt=kickoff_time,
                        _dtk=delta_to_kickoff, _of=odds_freshness,
                        _hf=home_formation, _af=away_formation,
                        _lda=lineup_detected_at,
                    ):
                        async with AsyncSessionLocal() as s:
                            # Lock match row first → stable lock order
                            await s.execute(text(
                                "SELECT 1 FROM matches WHERE id = :mid FOR NO KEY UPDATE"
                            ), {"mid": _mid})

                            # INSERT odds_snapshot
                            await s.execute(text("""
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
                                "match_id": _mid, "snapshot_at": _sat,
                                "odds_home": _oh, "odds_draw": _od, "odds_away": _oa,
                                "prob_home": _ph, "prob_draw": _pd, "prob_away": _pa,
                                "overround": _ov, "bookmaker": _src,
                                "kickoff_time": _kt, "delta_to_kickoff": _dtk,
                                "odds_freshness": _of,
                            })

                            # UPDATE matches: lineup_confirmed + formations
                            await s.execute(text("""
                                UPDATE matches
                                SET lineup_confirmed = TRUE,
                                    home_formation = :home_formation,
                                    away_formation = :away_formation,
                                    lineup_features_computed_at = :computed_at
                                WHERE id = :match_id
                            """), {
                                "match_id": _mid,
                                "home_formation": _hf,
                                "away_formation": _af,
                                "computed_at": _sat,
                            })

                            # Write-through odds to matches table
                            if _of == "live":
                                await s.execute(text("""
                                    UPDATE matches
                                    SET odds_home = :odds_home,
                                        odds_draw = :odds_draw,
                                        odds_away = :odds_away,
                                        odds_recorded_at = :recorded_at
                                    WHERE id = :match_id
                                      AND (odds_recorded_at IS NULL OR odds_recorded_at < :recorded_at)
                                """), {
                                    "match_id": _mid,
                                    "odds_home": _oh, "odds_draw": _od, "odds_away": _oa,
                                    "recorded_at": _sat,
                                })

                            # UPDATE match_lineups
                            await s.execute(text("""
                                UPDATE match_lineups
                                SET lineup_confirmed_at = COALESCE(lineup_confirmed_at, :confirmed_at),
                                    lineup_detected_at = COALESCE(lineup_detected_at, :detected_at)
                                WHERE match_id = :match_id
                            """), {
                                "match_id": _mid,
                                "confirmed_at": _sat,
                                "detected_at": _lda,
                            })

                            await s.commit()

                    await _deadlock_retry_write(_write_lineup_snapshot, match_id=match_id)

                    captured_count += 1

                    if odds_freshness == "live":
                        logger.info(
                            f"Synced live odds to matches: match_id={match_id}, "
                            f"H={odds_home:.2f}, D={odds_draw:.2f}, A={odds_away:.2f}, "
                            f"bookmaker={source}"
                        )

                    # Emit event AFTER this match's commit (PIT-correct)
                    try:
                        from app.events import get_event_bus, LINEUP_CONFIRMED
                        bus = get_event_bus()
                        await bus.emit(LINEUP_CONFIRMED, {
                            "match_id": match_id,
                            "lineup_detected_at": lineup_detected_at,
                            "source": "lineup_monitoring",
                            "provider": lineup_source,
                        })
                    except Exception as evt_err:
                        logger.warning(f"Failed to emit LINEUP_CONFIRMED for match {match_id}: {evt_err}")

                    # Track metrics
                    capture_end_time = datetime.utcnow()
                    latency_ms = int((capture_end_time - capture_start_time).total_seconds() * 1000)
                    _lineup_capture_metrics[job_type]["captures"] += 1
                    _lineup_capture_metrics[job_type]["latencies_ms"].append(latency_ms)
                    if len(_lineup_capture_metrics[job_type]["latencies_ms"]) > 1000:
                        _lineup_capture_metrics[job_type]["latencies_ms"] = \
                            _lineup_capture_metrics[job_type]["latencies_ms"][-1000:]

                    logger.info(
                        f"Captured lineup_confirmed odds for match {match_id}: "
                        f"H={odds_home:.2f}, D={odds_draw:.2f}, A={odds_away:.2f} at {snapshot_at} "
                        f"(latency: {latency_ms}ms, job: {job_type}, provider: {lineup_source})"
                    )

                except Exception as e:
                    errors.append(f"Match {match_id}: {str(e)}")
                    logger.warning(f"Error checking lineup for match {match_id}: {e}")

        finally:
            try:
                await provider.close()
            except Exception:
                pass
            if sofascore_provider:
                try:
                    await sofascore_provider.close()
                except Exception:
                    pass

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
# DEADLOCK RETRY HELPER (shared by market movement & lineup tracking jobs)
# =============================================================================
import random
from sqlalchemy.exc import DBAPIError

_DEADLOCK_MAX_RETRIES = 3
_DEADLOCK_BASE_WAIT = 0.1  # 100ms


async def _deadlock_retry_write(write_fn, match_id=None):
    """Execute a short DB write with deadlock retry + exponential backoff/jitter.

    write_fn is an async callable that opens its own AsyncSessionLocal(),
    does writes, and commits.  If a deadlock is detected, we retry up to
    _DEADLOCK_MAX_RETRIES times.
    """
    for attempt in range(_DEADLOCK_MAX_RETRIES):
        try:
            return await write_fn()
        except DBAPIError as e:
            err_msg = str(e.orig) if hasattr(e, "orig") else str(e)
            if "deadlock detected" in err_msg.lower():
                if attempt < _DEADLOCK_MAX_RETRIES - 1:
                    wait = (_DEADLOCK_BASE_WAIT * (2 ** attempt)) + random.uniform(0, 0.1)
                    logger.warning(
                        f"Deadlock detected (match {match_id}), "
                        f"retry {attempt + 1}/{_DEADLOCK_MAX_RETRIES} after {wait:.2f}s"
                    )
                    await asyncio.sleep(wait)
                else:
                    logger.error(
                        f"Deadlock persisted after {_DEADLOCK_MAX_RETRIES} retries "
                        f"for match {match_id}: {err_msg}"
                    )
                    raise
            else:
                raise
    return None  # Should not reach here


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

    DEADLOCK FIX (2026-02-15): Refactored to short per-match transactions.
    - Phase 1: Read candidates into memory (read-only session, closed before API calls).
    - Phase 2: For each candidate, call API OUTSIDE any DB transaction,
      then open a SHORT tx with FOR NO KEY UPDATE lock ordering + deadlock retry.

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
        # --- Phase 1: Read candidates into memory ---
        candidates = []  # list of (match_id, external_id, kickoff_time, minutes_to_kickoff, bucket)
        async with AsyncSessionLocal() as session:
            now = datetime.utcnow()
            league_ids = await resolve_lineup_monitoring_leagues(session)

            window_start = now + timedelta(minutes=3)
            window_end = now + timedelta(minutes=65)

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

            MAX_API_CALLS_PER_RUN = int(os.environ.get("MARKET_MOVEMENT_MAX_CALLS", "15"))

            for match in matches:
                minutes_to_kickoff = float(match.minutes_to_kickoff)
                checked_count += 1

                current_bucket = None
                for min_m, max_m, bucket_name in MARKET_MOVEMENT_BUCKETS:
                    if min_m <= minutes_to_kickoff <= max_m:
                        current_bucket = bucket_name
                        break

                if not current_bucket:
                    continue

                # Check if bucket already captured
                existing = await session.execute(text("""
                    SELECT 1 FROM market_movement_snapshots
                    WHERE match_id = :match_id AND snapshot_type = :bucket
                """), {"match_id": match.id, "bucket": current_bucket})

                if existing.fetchone():
                    continue

                candidates.append((match.id, match.external_id, match.date, minutes_to_kickoff, current_bucket))

                if len(candidates) >= MAX_API_CALLS_PER_RUN:
                    break
        # --- Session closed: no DB locks held during API calls ---

        if not candidates:
            return {"checked": checked_count, "captured": 0}

        # --- Phase 2: API call + short tx per match ---
        provider = APIFootballProvider()
        try:
            for match_id, external_id, kickoff_time, minutes_to_kickoff, current_bucket in candidates:
                # API call OUTSIDE any DB transaction
                fresh_odds = await provider.get_odds(external_id)

                if not fresh_odds or not fresh_odds.get("odds_home"):
                    continue

                odds_home = float(fresh_odds["odds_home"])
                odds_draw = float(fresh_odds["odds_draw"])
                odds_away = float(fresh_odds["odds_away"])
                bookmaker = fresh_odds.get("bookmaker", "unknown")

                if not (odds_home > 1 and odds_draw > 1 and odds_away > 1):
                    continue

                raw_home = 1 / odds_home
                raw_draw = 1 / odds_draw
                raw_away = 1 / odds_away
                total = raw_home + raw_draw + raw_away
                overround = total - 1
                prob_home = raw_home / total
                prob_draw = raw_draw / total
                prob_away = raw_away / total
                snapshot_at = datetime.utcnow()

                # Short tx with deadlock retry + consistent lock ordering
                async def _write_snapshot(
                    _mid=match_id, _bucket=current_bucket, _snapshot_at=snapshot_at,
                    _kickoff=kickoff_time, _mtk=minutes_to_kickoff,
                    _oh=odds_home, _od=odds_draw, _oa=odds_away,
                    _bk=bookmaker, _ph=prob_home, _pd=prob_draw, _pa=prob_away,
                    _ov=overround,
                ):
                    async with AsyncSessionLocal() as s:
                        # Lock match row first → stable lock order across concurrent jobs
                        await s.execute(text(
                            "SELECT 1 FROM matches WHERE id = :mid FOR NO KEY UPDATE"
                        ), {"mid": _mid})

                        await s.execute(text("""
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
                            "match_id": _mid, "bucket": _bucket,
                            "captured_at": _snapshot_at, "kickoff_time": _kickoff,
                            "minutes_to_kickoff": _mtk,
                            "odds_home": _oh, "odds_draw": _od, "odds_away": _oa,
                            "bookmaker": f"{_bk}_live",
                            "prob_home": _ph, "prob_draw": _pd, "prob_away": _pa,
                            "overround": _ov,
                        })

                        # Check completion
                        bucket_count = await s.execute(text("""
                            SELECT COUNT(DISTINCT snapshot_type)
                            FROM market_movement_snapshots
                            WHERE match_id = :match_id
                        """), {"match_id": _mid})

                        if bucket_count.scalar() >= 4:
                            await s.execute(text("""
                                UPDATE matches SET market_movement_complete = TRUE
                                WHERE id = :match_id
                            """), {"match_id": _mid})

                        await s.commit()

                await _deadlock_retry_write(_write_snapshot, match_id=match_id)

                captured_count += 1
                logger.info(
                    f"Market movement {current_bucket} captured for match {match_id}: "
                    f"H={odds_home:.2f} D={odds_draw:.2f} A={odds_away:.2f}"
                )
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

    DEADLOCK FIX (2026-02-15): Refactored to short per-match transactions.
    """
    from app.etl.api_football import APIFootballProvider

    captured_count = 0
    checked_count = 0

    try:
        # --- Phase 1: Read candidates into memory ---
        # Each candidate: (match row data, needs_l0, bucket_to_capture)
        candidates = []
        async with AsyncSessionLocal() as session:
            league_ids = await resolve_lineup_monitoring_leagues(session)

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
                  AND os.snapshot_at > NOW() - INTERVAL '45 minutes'
                  AND m.date > NOW() - INTERVAL '15 minutes'
                ORDER BY os.snapshot_at DESC
                LIMIT 20
            """), {"league_ids": league_ids, "league_ids_is_null": league_ids is None})

            matches = result.fetchall()
            if not matches:
                return {"checked": 0, "captured": 0}

            MAX_API_CALLS_PER_RUN = int(os.environ.get("LINEUP_MOVEMENT_MAX_CALLS", "10"))

            for match in matches:
                match_id = match.match_id
                minutes_since_lineup = float(match.minutes_since_lineup)
                checked_count += 1

                # Check L0 existence
                l0_exists = await session.execute(text("""
                    SELECT 1 FROM lineup_movement_snapshots
                    WHERE match_id = :match_id AND snapshot_type = 'L0'
                """), {"match_id": match_id})
                needs_l0 = not l0_exists.fetchone()

                # Check which bucket
                current_bucket = None
                for min_m, max_m, bucket_name in LINEUP_MOVEMENT_BUCKETS:
                    if min_m <= minutes_since_lineup <= max_m:
                        current_bucket = bucket_name
                        break

                # Check if bucket already exists
                if current_bucket:
                    existing = await session.execute(text("""
                        SELECT 1 FROM lineup_movement_snapshots
                        WHERE match_id = :match_id AND snapshot_type = :bucket
                    """), {"match_id": match_id, "bucket": current_bucket})
                    if existing.fetchone():
                        current_bucket = None  # Already captured

                if not needs_l0 and not current_bucket:
                    continue

                candidates.append({
                    "match": match,
                    "needs_l0": needs_l0,
                    "bucket": current_bucket,
                    "minutes_since_lineup": minutes_since_lineup,
                })

                # Only count API-requiring candidates against limit
                if current_bucket and sum(1 for c in candidates if c["bucket"]) >= MAX_API_CALLS_PER_RUN:
                    break
        # --- Session closed: no DB locks held during API calls ---

        if not candidates:
            return {"checked": checked_count, "captured": 0}

        # --- Phase 2: API call + short tx per match ---
        provider = APIFootballProvider()
        try:
            for cand in candidates:
                match = cand["match"]
                match_id = match.match_id
                needs_l0 = cand["needs_l0"]
                current_bucket = cand["bucket"]
                minutes_since_lineup = cand["minutes_since_lineup"]

                # Insert L0 in its own short tx (no API call needed)
                if needs_l0:
                    async def _write_l0(
                        _mid=match_id,
                        _lda=match.lineup_detected_at,
                        _kt=match.kickoff_time,
                        _oh=match.lineup_odds_home, _od=match.lineup_odds_draw, _oa=match.lineup_odds_away,
                        _bk=match.lineup_bookmaker,
                        _ph=match.lineup_prob_home, _pd=match.lineup_prob_draw, _pa=match.lineup_prob_away,
                    ):
                        async with AsyncSessionLocal() as s:
                            await s.execute(text(
                                "SELECT 1 FROM matches WHERE id = :mid FOR NO KEY UPDATE"
                            ), {"mid": _mid})
                            overround = (1/float(_oh) + 1/float(_od) + 1/float(_oa)) - 1 if _oh else 0
                            await s.execute(text("""
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
                                "match_id": _mid, "lineup_detected_at": _lda,
                                "kickoff_time": _kt,
                                "odds_home": _oh, "odds_draw": _od, "odds_away": _oa,
                                "bookmaker": _bk,
                                "prob_home": _ph, "prob_draw": _pd, "prob_away": _pa,
                                "overround": overround,
                            })
                            await s.commit()

                    await _deadlock_retry_write(_write_l0, match_id=match_id)
                    captured_count += 1

                if not current_bucket:
                    continue

                # API call OUTSIDE any DB transaction
                try:
                    fresh_odds = await provider.get_odds(match.external_id)
                except Exception as e:
                    error_str = str(e).lower()
                    if "429" in error_str or "rate limit" in error_str:
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

                if not (odds_home > 1 and odds_draw > 1 and odds_away > 1):
                    continue

                raw_home = 1 / odds_home
                raw_draw = 1 / odds_draw
                raw_away = 1 / odds_away
                total = raw_home + raw_draw + raw_away
                overround = total - 1
                prob_home = raw_home / total
                prob_draw = raw_draw / total
                prob_away = raw_away / total

                delta_p = compute_delta_p(
                    float(match.lineup_prob_home or 0),
                    float(match.lineup_prob_draw or 0),
                    float(match.lineup_prob_away or 0),
                    prob_home, prob_draw, prob_away
                )
                snapshot_at = datetime.utcnow()

                # Short tx: INSERT snapshot + check completion + UPDATE matches
                async def _write_movement(
                    _mid=match_id, _lda=match.lineup_detected_at,
                    _bucket=current_bucket, _msl=minutes_since_lineup,
                    _sat=snapshot_at, _kt=match.kickoff_time,
                    _oh=odds_home, _od=odds_draw, _oa=odds_away,
                    _bk=bookmaker, _ph=prob_home, _pd=prob_draw, _pa=prob_away,
                    _ov=overround, _dp=delta_p,
                ):
                    async with AsyncSessionLocal() as s:
                        await s.execute(text(
                            "SELECT 1 FROM matches WHERE id = :mid FOR NO KEY UPDATE"
                        ), {"mid": _mid})

                        await s.execute(text("""
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
                            "match_id": _mid, "lineup_detected_at": _lda,
                            "bucket": _bucket, "minutes_from_lineup": _msl,
                            "captured_at": _sat, "kickoff_time": _kt,
                            "odds_home": _oh, "odds_draw": _od, "odds_away": _oa,
                            "bookmaker": f"{_bk}_live",
                            "prob_home": _ph, "prob_draw": _pd, "prob_away": _pa,
                            "overround": _ov, "delta_p": _dp,
                        })

                        # Check completion: L0 + at least one post-lineup
                        snapshot_count = await s.execute(text("""
                            SELECT
                                COUNT(*) FILTER (WHERE snapshot_type = 'L0') as has_l0,
                                COUNT(*) FILTER (WHERE snapshot_type IN ('L+5', 'L+10')) as has_post
                            FROM lineup_movement_snapshots
                            WHERE match_id = :match_id
                        """), {"match_id": _mid})

                        counts = snapshot_count.fetchone()
                        if counts and counts.has_l0 > 0 and counts.has_post > 0:
                            await s.execute(text("""
                                UPDATE matches SET lineup_movement_tracked = TRUE
                                WHERE id = :match_id
                            """), {"match_id": _mid})
                            logger.info(f"Match {_mid} marked as lineup_movement_tracked")

                        await s.commit()

                await _deadlock_retry_write(_write_movement, match_id=match_id)

                captured_count += 1
                logger.info(
                    f"Lineup movement {current_bucket} captured for match {match_id}: "
                    f"delta_p={delta_p:.4f} ({minutes_since_lineup:.1f} min since lineup)"
                )
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


async def daily_save_predictions(return_metrics: bool = False) -> dict | None:
    """
    Daily job to save predictions for upcoming matches.
    Runs every day at 7:00 AM UTC (before audit).

    AUDIT FIX (2026-01-25): Batching + time budget to prevent connection timeouts.
    - Filters NS-only BEFORE predicting (reduces work)
    - Commits every BATCH_SIZE matches (avoids long transactions)
    - Time budget guardrail (exits cleanly if exceeded)
    - Safe rollback on DB errors (no InterfaceError on closed connection)

    Args:
        return_metrics: If True, return detailed metrics dict instead of None.
                       Used by /dashboard/predictions/trigger-fase0 endpoint.
    """
    import numpy as np
    from app.db_utils import upsert
    from app.ml import XGBoostEngine
    from app.models import Prediction, Match
    from app.ml.shadow import is_shadow_enabled, log_shadow_prediction
    from app.ml.sensor import log_sensor_prediction
    from app.config import get_settings
    from app.telemetry.metrics import record_job_run
    from sqlalchemy import select, func
    from sqlalchemy.exc import InterfaceError, DBAPIError

    # Configuration
    BATCH_SIZE = 200
    TIME_BUDGET_MS = 600_000  # 10 minutes max (3500+ NS matches need ~400s for features)

    start_time = time.time()
    logger.info("[DAILY-SAVE] Starting daily prediction save job...")

    # Counters for observability
    saved = 0
    skipped_no_features = 0
    errors = 0
    batches_processed = 0
    shadow_logged = 0
    shadow_errors = 0
    sensor_logged = 0
    sensor_errors = 0
    job_status = "ok"

    async def safe_rollback(session):
        """Safely attempt rollback without raising on closed connection."""
        try:
            await session.rollback()
        except Exception:
            pass

    def check_time_budget() -> bool:
        """Returns True if time budget exceeded."""
        elapsed_ms = (time.time() - start_time) * 1000
        return elapsed_ms > TIME_BUDGET_MS

    try:
        # Load ML engine (outside session - doesn't need DB)
        engine = XGBoostEngine()
        if not engine.load_model():
            logger.error("[DAILY-SAVE] Could not load ML model")
            record_job_run(job="daily_save_predictions", status="error", duration_ms=0)
            if return_metrics:
                return {"status": "error", "error": "Could not load ML model", "n_matches_total": 0, "n_eligible": 0, "n_filtered": 0, "filtered_by_reason": {}, "duration_ms": 0}
            return

        sensor_settings = get_settings()

        # =================================================================
        # PHASE 1: Fetch features (separate session to avoid idle timeout)
        # =================================================================
        df = None
        ns_total = 0
        next_ns_date = None

        # Use get_session_with_retry to handle Railway connection drops (InterfaceError fix)
        async with get_session_with_retry(max_retries=3, retry_delay=1.0) as session:
            # Get upcoming matches features (can be slow - wrap in try/except)
            # FASE 0 FIX: league_only=True prevents "Exeter mode" where cup matches
            # against amateur teams inflate rolling averages for lower-division teams
            try:
                feature_engineer = FeatureEngineer(session=session)
                df = await feature_engineer.get_upcoming_matches_features(league_only=True, days_ahead=7)
            except (InterfaceError, DBAPIError) as fetch_err:
                duration_ms = (time.time() - start_time) * 1000
                logger.error(f"[DAILY-SAVE] DB connection lost during feature fetch: {fetch_err}")
                record_job_run(job="daily_save_predictions", status="error", duration_ms=duration_ms)
                if return_metrics:
                    return {"status": "error", "error": f"DB connection lost: {fetch_err}", "n_matches_total": 0, "n_eligible": 0, "n_filtered": 0, "filtered_by_reason": {}, "duration_ms": duration_ms}
                return

            # P0 FIX: Filter to NS-only BEFORE predicting (reduces work significantly)
            total_fetched = len(df)
            if total_fetched > 0 and "status" in df.columns:
                df = df[df["status"] == "NS"].reset_index(drop=True)
            ns_total = len(df)

            # ═══════════════════════════════════════════════════════════════════
            # KILL-SWITCH ROUTER (FASE 0/1)
            # Criterio: Ambos equipos deben tener >= MIN_LEAGUE_MATCHES partidos de LIGA
            #           en los LOOKBACK_DAYS días ANTERIORES al kickoff del partido
            # Config: KILLSWITCH_ENABLED, KILLSWITCH_MIN_LEAGUE_MATCHES, KILLSWITCH_LOOKBACK_DAYS
            # ═══════════════════════════════════════════════════════════════════
            from app.config import get_settings
            _ks_settings = get_settings()
            KILLSWITCH_ENABLED = _ks_settings.KILLSWITCH_ENABLED
            MIN_LEAGUE_MATCHES = _ks_settings.KILLSWITCH_MIN_LEAGUE_MATCHES
            LOOKBACK_DAYS = _ks_settings.KILLSWITCH_LOOKBACK_DAYS

            # Initialize kill-switch metrics (for return_metrics)
            killswitch_eligible = 0
            n_filtered = 0
            filtered_by_reason = {"home_insufficient": 0, "away_insufficient": 0, "both_insufficient": 0}

            if ns_total > 0 and KILLSWITCH_ENABLED:
                from collections import defaultdict
                from app.telemetry.metrics import (
                    PREDICTIONS_KILLSWITCH_FILTERED,
                    PREDICTIONS_KILLSWITCH_ELIGIBLE,
                )

                # PASO 1: Query BATCH - traer partidos de liga relevantes
                all_team_ids = list(set(
                    df["home_team_id"].tolist() + df["away_team_id"].tolist()
                ))

                # FIX #1: Calcular earliest_needed desde MIN(match.date), NO desde NOW()
                min_match_date = df["date"].min()
                earliest_needed = min_match_date - timedelta(days=LOOKBACK_DAYS + 7)

                league_matches_result = await session.execute(text("""
                    SELECT team_id, match_date
                    FROM (
                        SELECT home_team_id as team_id, date as match_date
                        FROM matches m
                        JOIN admin_leagues al ON m.league_id = al.league_id
                        WHERE m.status = 'FT'
                          AND al.kind = 'league'
                          AND m.date >= :earliest_needed
                        UNION ALL
                        SELECT away_team_id as team_id, date as match_date
                        FROM matches m
                        JOIN admin_leagues al ON m.league_id = al.league_id
                        WHERE m.status = 'FT'
                          AND al.kind = 'league'
                          AND m.date >= :earliest_needed
                    ) sub
                    WHERE team_id = ANY(:team_ids)
                    ORDER BY team_id, match_date DESC
                """), {"team_ids": all_team_ids, "earliest_needed": earliest_needed})

                # Construir dict: team_id -> [lista de fechas de partidos de liga]
                team_match_dates = defaultdict(list)
                for row in league_matches_result.fetchall():
                    team_match_dates[row.team_id].append(row.match_date)

                # PASO 2: Filtrar en MEMORIA con cutoff POR MATCH (match.date)
                n_before = len(df)
                eligible_match_ids = []
                # Track filtered reasons for return_metrics
                filtered_by_reason = {"home_insufficient": 0, "away_insufficient": 0, "both_insufficient": 0}

                for _, match_row in df.iterrows():
                    # CRÍTICO: El cutoff es relativo al kickoff del PARTIDO A PREDECIR
                    match_kickoff = match_row["date"]
                    cutoff = match_kickoff - timedelta(days=LOOKBACK_DAYS)

                    home_id = match_row["home_team_id"]
                    away_id = match_row["away_team_id"]

                    # Contar partidos de liga en ventana [cutoff, match_kickoff)
                    home_count = sum(
                        1 for d in team_match_dates.get(home_id, [])
                        if cutoff <= d < match_kickoff
                    )
                    away_count = sum(
                        1 for d in team_match_dates.get(away_id, [])
                        if cutoff <= d < match_kickoff
                    )

                    home_ok = home_count >= MIN_LEAGUE_MATCHES
                    away_ok = away_count >= MIN_LEAGUE_MATCHES

                    if home_ok and away_ok:
                        eligible_match_ids.append(match_row["match_id"])
                    else:
                        # FIX #2: Labels unificados con sufijo _insufficient
                        if not home_ok and not away_ok:
                            reason = "both_insufficient"
                        elif not home_ok:
                            reason = "home_insufficient"
                        else:
                            reason = "away_insufficient"

                        filtered_by_reason[reason] += 1
                        logger.info(
                            f"[KILL-SWITCH] Skipping match {match_row['match_id']} "
                            f"(reason={reason}, home={home_count}, away={away_count})"
                        )
                        PREDICTIONS_KILLSWITCH_FILTERED.labels(reason=reason).inc()

                # FIX #3: Calcular filtered correctamente ANTES de modificar df
                n_filtered = n_before - len(eligible_match_ids)
                killswitch_eligible = len(eligible_match_ids)
                PREDICTIONS_KILLSWITCH_ELIGIBLE.set(killswitch_eligible)
                df = df[df["match_id"].isin(eligible_match_ids)].reset_index(drop=True)
                logger.info(f"[KILL-SWITCH] {killswitch_eligible} eligible, {n_filtered} filtered")
            elif ns_total > 0 and not KILLSWITCH_ENABLED:
                # Kill-switch disabled - all matches are eligible
                killswitch_eligible = ns_total
                logger.info(f"[KILL-SWITCH] DISABLED - all {ns_total} matches eligible")
            # ═══════════════════════════════════════════════════════════════════

            # Query next NS match date for logging
            next_ns_result = await session.execute(
                select(func.min(Match.date))
                .where(Match.status == "NS", Match.date > datetime.utcnow())
            )
            next_ns_date = next_ns_result.scalar()
        # Session closed here - connection returned to pool

        logger.info(
            f"[DAILY-SAVE] Fetched {total_fetched} matches, filtered to {ns_total} NS, "
            f"next_ns_utc={next_ns_date.isoformat() if next_ns_date else 'None'}, "
            f"batch_size={BATCH_SIZE}, time_budget_ms={TIME_BUDGET_MS}"
        )

        if ns_total == 0:
            logger.info("[DAILY-SAVE] No NS matches to process")
            record_job_run(job="daily_save_predictions", status="ok", duration_ms=0)
            if return_metrics:
                return {"status": "ok", "n_matches_total": total_fetched, "n_eligible": 0, "n_filtered": 0, "filtered_by_reason": {}, "saved": 0, "duration_ms": 0}
            return

        # Time budget check BEFORE predict() (feature fetch can be slow)
        if check_time_budget():
            elapsed_ms = (time.time() - start_time) * 1000
            logger.warning(
                f"[DAILY-SAVE] Time budget exceeded before predict (elapsed={elapsed_ms:.0f}ms), "
                f"exiting early with ns_total={ns_total}"
            )
            record_job_run(job="daily_save_predictions", status="partial", duration_ms=elapsed_ms)
            if return_metrics:
                return {"status": "partial", "error": "Time budget exceeded before predict", "n_matches_total": total_fetched, "n_eligible": killswitch_eligible, "n_filtered": n_filtered, "filtered_by_reason": filtered_by_reason, "saved": 0, "duration_ms": elapsed_ms}
            return

        # =================================================================
        # PHASE 2: Make predictions (CPU-bound, no DB connection needed)
        # =================================================================
        # ABE P2: Capture PIT boundary BEFORE predicting — all information
        # used for these predictions was available at this timestamp.
        asof_timestamp = datetime.utcnow()

        # League Router: log tier distribution for observability (GDT M3)
        from app.ml.league_router import get_league_tier
        if "league_id" in df.columns:
            tier_counts = df["league_id"].apply(get_league_tier).value_counts().to_dict()
            logger.info(f"[DAILY-SAVE] League Router tiers: {tier_counts}")

        predictions = engine.predict(df)

        # PHASE 2a: Two-Stage W3 overlay for TS leagues (ABE routing)
        from app.ml.twostage_serving import overlay_ts_predictions
        predictions, ts_stats = overlay_ts_predictions(predictions, ml_engine=engine)
        if ts_stats.get("ts_hits", 0) > 0:
            logger.info(
                "[DAILY-SAVE] ts_serving | hits=%d no_odds=%d eligible=%d os_kept=%d",
                ts_stats["ts_hits"], ts_stats["ts_no_odds"],
                ts_stats["ts_eligible"], ts_stats["os_kept"],
            )

        # PHASE 2b: Market anchor — blend with market for low-signal leagues
        from app.ml.policy import apply_market_anchor, get_policy_config
        _policy_cfg = get_policy_config()
        predictions, _anchor_meta = apply_market_anchor(
            predictions,
            alpha_default=_policy_cfg["market_anchor_alpha_default"],
            league_overrides=_policy_cfg["market_anchor_league_overrides"],
            enabled=_policy_cfg["market_anchor_enabled"],
        )

        # PHASE 2c: Stamp serving telemetry (SSOT)
        from app.ml.serving_telemetry import stamp_serving_metadata
        stamp_serving_metadata(predictions, engine.model_version)

        # =================================================================
        # PHASE 3: Save predictions (new session, fresh connection)
        # =================================================================
        # Use get_session_with_retry to handle Railway connection drops (InterfaceError fix)
        async with get_session_with_retry(max_retries=3, retry_delay=1.0) as session:
            # Process in batches with commit per batch
            for batch_start in range(0, len(predictions), BATCH_SIZE):
                # Time budget check at start of each batch
                if check_time_budget():
                    remaining = len(predictions) - batch_start
                    logger.warning(
                        f"[DAILY-SAVE] Time budget reached, exiting early "
                        f"(saved={saved}, remaining={remaining})"
                    )
                    job_status = "partial"
                    break

                batch_end = min(batch_start + BATCH_SIZE, len(predictions))
                batch_predictions = predictions[batch_start:batch_end]

                try:
                    for idx, pred in enumerate(batch_predictions):
                        global_idx = batch_start + idx
                        match_id = pred.get("match_id")
                        if not match_id:
                            continue

                        probs = pred.get("probabilities")
                        if not probs:
                            skipped_no_features += 1
                            continue

                        # Upsert prediction
                        # ABE P2: Include asof_timestamp for PIT attribution
                        # ABE routing: use TS model_version if TS overlay was applied
                        served_version = pred.get("model_version_served", engine.model_version)
                        await upsert(
                            session,
                            Prediction,
                            values={
                                "match_id": match_id,
                                "model_version": served_version,
                                "home_prob": probs["home"],
                                "draw_prob": probs["draw"],
                                "away_prob": probs["away"],
                                "asof_timestamp": asof_timestamp,
                            },
                            conflict_columns=["match_id", "model_version"],
                            update_columns=["home_prob", "draw_prob", "away_prob", "asof_timestamp"],
                        )
                        saved += 1

                        # Shadow prediction (never affects main flow)
                        if is_shadow_enabled():
                            try:
                                match_df = df.iloc[[global_idx]]
                                shadow_result = await log_shadow_prediction(
                                    session=session,
                                    match_id=match_id,
                                    df=match_df,
                                    baseline_engine=engine,
                                    skip_commit=True,
                                )
                                if shadow_result:
                                    shadow_logged += 1
                            except Exception:
                                shadow_errors += 1

                        # Sensor B prediction (never affects main flow)
                        if sensor_settings.SENSOR_ENABLED:
                            try:
                                match_df = df.iloc[[global_idx]]
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
                            except Exception:
                                sensor_errors += 1

                    # Commit this batch
                    await session.commit()
                    batches_processed += 1

                except (InterfaceError, DBAPIError) as db_err:
                    # Connection lost - safe rollback and exit
                    logger.error(f"[DAILY-SAVE] DB connection lost: {db_err}")
                    await safe_rollback(session)
                    job_status = "error"
                    break

                except Exception as batch_err:
                    # Other error - try to rollback and continue with next batch
                    logger.warning(f"[DAILY-SAVE] Batch error: {batch_err}")
                    await safe_rollback(session)
                    errors += 1

        # Final telemetry
        duration_ms = (time.time() - start_time) * 1000

        # Record shadow/sensor batch counters
        if is_shadow_enabled() and (shadow_logged > 0 or shadow_errors > 0):
            record_shadow_predictions_batch(logged=shadow_logged, errors=shadow_errors)
        elif not is_shadow_enabled() and killswitch_eligible > 0:
            from app.telemetry.metrics import record_shadow_engine_not_loaded_skip
            record_shadow_engine_not_loaded_skip(killswitch_eligible)
            logger.info(
                f"[DAILY-SAVE] Shadow engine not loaded, "
                f"{killswitch_eligible} eligible predictions without shadow"
            )
        if sensor_settings.SENSOR_ENABLED and (sensor_logged > 0 or sensor_errors > 0):
            record_sensor_predictions_batch(logged=sensor_logged, errors=sensor_errors)

        log_msg = (
            f"[DAILY-SAVE] Complete: status={job_status}, saved={saved}, "
            f"ns_total={ns_total}, batches={batches_processed}, "
            f"skipped_no_features={skipped_no_features}, errors={errors}, "
            f"model={engine.model_version}, duration_ms={duration_ms:.0f}"
        )
        if is_shadow_enabled():
            log_msg += f", shadow={shadow_logged}/{shadow_errors}"
        if sensor_settings.SENSOR_ENABLED:
            log_msg += f", sensor={sensor_logged}/{sensor_errors}"
        logger.info(log_msg)

        record_job_run(job="daily_save_predictions", status=job_status, duration_ms=duration_ms)

        # Return metrics dict if requested (for trigger-fase0 endpoint)
        if return_metrics:
            return {
                "status": job_status,
                "n_matches_total": total_fetched,
                "n_ns": ns_total,
                "n_eligible": killswitch_eligible,
                "n_filtered": n_filtered,
                "filtered_by_reason": filtered_by_reason,
                "saved": saved,
                "skipped_no_features": skipped_no_features,
                "errors": errors,
                "model_version": engine.model_version,
                "duration_ms": round(duration_ms),
            }

    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        logger.error(f"[DAILY-SAVE] Failed after {duration_ms:.0f}ms: {e}")
        record_job_run(job="daily_save_predictions", status="error", duration_ms=duration_ms)
        if return_metrics:
            return {"status": "error", "error": str(e), "n_matches_total": 0, "n_eligible": 0, "n_filtered": 0, "filtered_by_reason": {}, "duration_ms": round(duration_ms)}


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
            # FASE 0 FIX: league_only=True prevents "Exeter mode"
            match_ids = [g[0] for g in gaps]
            feature_engineer = FeatureEngineer(session=session)

            # Optimized: compute features ONLY for gap matches (O(k) not O(N))
            df = await feature_engineer.get_matches_features_by_ids(
                match_ids, league_only=True, statuses=["NS"]
            )

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
        # Use get_session_with_retry to handle Railway connection drops
        async with get_session_with_retry(max_retries=3, retry_delay=1.0) as session:
            # Find live matches
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
            return {"status": "skipped", "reason": "no_live_matches"}

        live_count = len(live_matches)
        logger.info(f"[LIVE_TICK] Found {live_count} live matches")

        id_map = {ext_id: int_id for int_id, ext_id in live_matches}
        external_ids = list(id_map.keys())

        # Batch fetch from API-Football
        provider = APIFootballProvider()
        updated = 0
        api_errors = 0
        all_fixtures = []

        try:
            # 1. Fetch all data FIRST (Network I/O) without holding DB locks
            for i in range(0, len(external_ids), 20):
                chunk = external_ids[i:i + 20]
                try:
                    fixtures = await provider.get_fixtures_by_ids(chunk)
                    all_fixtures.extend(fixtures)
                except Exception as chunk_err:
                    api_errors += 1
                    err_str = str(chunk_err).lower()
                    if "rate" in err_str or "limit" in err_str:
                        logger.warning(f"[LIVE_TICK] Rate limited, stopping fetch")
                        break
                    if "budget" in err_str or "exceeded" in err_str:
                        logger.error(f"[LIVE_TICK] Budget exceeded, disabling tick")
                        break
                    logger.warning(f"[LIVE_TICK] Chunk error: {chunk_err}")

            # 2. Open new session to perform FAST updates
            if all_fixtures:
                async with get_session_with_retry(max_retries=3, retry_delay=1.0) as session:
                    for f in all_fixtures:
                        ext_id = f.get("external_id")
                        if ext_id not in id_map:
                            continue

                        match_id = id_map[ext_id]
                        new_status = f.get("status")
                        new_elapsed = f.get("elapsed")
                        new_elapsed_extra = f.get("elapsed_extra")
                        new_home = f.get("home_goals")
                        new_away = f.get("away_goals")
                        new_events = f.get("events")

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


# ═══════════════════════════════════════════════════════════════
# SHADOW ext-A/B/C (experimental model predictions)
# ATI: Un solo job genérico procesa todas las variantes habilitadas
# ═══════════════════════════════════════════════════════════════

# Features for ext models (same as training - shared by A/B/C)
EXT_14_FEATURES = [
    "home_goals_scored_avg",
    "home_goals_conceded_avg",
    "home_shots_avg",
    "home_corners_avg",
    "home_rest_days",
    "home_matches_played",
    "away_goals_scored_avg",
    "away_goals_conceded_avg",
    "away_shots_avg",
    "away_corners_avg",
    "away_rest_days",
    "away_matches_played",
    "goal_diff_avg",
    "rest_diff",
]

# Legacy alias
EXTC_14_FEATURES = EXT_14_FEATURES


async def generate_ext_shadow_predictions():
    """
    Generate shadow predictions for ext-A/B/C/D models.

    ATI: Un solo job genérico que procesa todas las variantes habilitadas.
    Cada variante se procesa independientemente (fail-closed por variante).
    - ext-A/B/C: Modelos con diferentes min_date de training
    - ext-D: Candidato league-only (v1.0.1-league-only-20260202)

    Writes ONLY to predictions_experiments (never to predictions).
    """
    from app.config import get_settings
    from app.telemetry.metrics import (
        EXT_SHADOW_INSERTED,
        EXT_SHADOW_SKIPPED,
        EXT_SHADOW_ERRORS,
        EXT_SHADOW_LAST_SUCCESS,
    )

    settings = get_settings()
    job_name = "ext_shadow"
    start_time = time.time()

    # Build list of enabled variants
    variants = []
    if settings.EXTA_SHADOW_ENABLED:
        variants.append({
            "name": "A",
            "model_version": settings.EXTA_SHADOW_MODEL_VERSION,
            "model_path": settings.EXTA_SHADOW_MODEL_PATH,
        })
    if settings.EXTB_SHADOW_ENABLED:
        variants.append({
            "name": "B",
            "model_version": settings.EXTB_SHADOW_MODEL_VERSION,
            "model_path": settings.EXTB_SHADOW_MODEL_PATH,
        })
    if settings.EXTC_SHADOW_ENABLED:
        variants.append({
            "name": "C",
            "model_version": settings.EXTC_SHADOW_MODEL_VERSION,
            "model_path": settings.EXTC_SHADOW_MODEL_PATH,
        })
    if settings.EXTD_SHADOW_ENABLED:
        variants.append({
            "name": "D",
            "model_version": settings.EXTD_SHADOW_MODEL_VERSION,
            "model_path": settings.EXTD_SHADOW_MODEL_PATH,
        })

    if not variants:
        return {"status": "disabled", "variants": []}

    results = {}
    for variant in variants:
        try:
            result = await _generate_ext_shadow_for_variant(
                variant_name=variant["name"],
                model_version=variant["model_version"],
                model_path=variant["model_path"],
                settings=settings,
            )
            results[variant["name"]] = result

            # Update metrics
            if result.get("inserted", 0) > 0:
                EXT_SHADOW_INSERTED.labels(variant=variant["name"]).inc(result["inserted"])
            if result.get("skipped", 0) > 0:
                EXT_SHADOW_SKIPPED.labels(variant=variant["name"]).inc(result["skipped"])
            EXT_SHADOW_LAST_SUCCESS.labels(variant=variant["name"]).set_to_current_time()

        except Exception as e:
            logger.error(f"[EXT_SHADOW] Variant {variant['name']} failed: {e}")
            EXT_SHADOW_ERRORS.labels(variant=variant["name"]).inc()
            results[variant["name"]] = {"status": "error", "error": str(e)}

    duration_ms = (time.time() - start_time) * 1000
    total_inserted = sum(r.get("inserted", 0) for r in results.values() if isinstance(r, dict))
    record_job_run(job=job_name, status="ok", duration_ms=duration_ms)

    if total_inserted > 0:
        logger.info(f"[EXT_SHADOW] Total inserted: {total_inserted} across {len(variants)} variants")

    return {"status": "ok", "variants": results}


async def _generate_ext_shadow_for_variant(
    variant_name: str,
    model_version: str,
    model_path: str,
    settings,
) -> dict:
    """
    Generate predictions for a single ext variant.

    ATI: Fail-closed - if model not found or error, return error dict without crashing.
    """
    from pathlib import Path
    from collections import defaultdict
    import numpy as np
    import xgboost as xgb

    # Load model (fail-closed if not found)
    if not Path(model_path).exists():
        logger.warning(f"[EXT_SHADOW] ext-{variant_name}: Model not found: {model_path}")
        record_ext_shadow_rejection(variant_name, "model_not_found")
        return {"status": "error", "reason": "model_not_found", "path": model_path}

    model = xgb.XGBClassifier()
    model.load_model(model_path)

    async with AsyncSessionLocal() as session:
        # ATI FIX v2: Siempre usar start_at fijo para evitar gaps si job cae >2h
        start_filter = f"AND os.snapshot_at >= '{settings.EXT_SHADOW_START_AT}'"

        # ATI FIX: Anti-join para encontrar snapshots pendientes
        snapshots_result = await session.execute(text(f"""
            SELECT
                os.id as snapshot_id,
                os.match_id,
                os.snapshot_at,
                m.home_team_id,
                m.away_team_id,
                m.date as match_date
            FROM odds_snapshots os
            JOIN matches m ON os.match_id = m.id
            LEFT JOIN predictions_experiments pe
                ON pe.snapshot_id = os.id
                AND pe.model_version = :model_version
            WHERE os.snapshot_type = 'lineup_confirmed'
              {start_filter}
              AND pe.snapshot_id IS NULL
              AND m.date > os.snapshot_at + INTERVAL '10 minutes'
              AND m.date < os.snapshot_at + INTERVAL '90 minutes'
            ORDER BY os.snapshot_at
            LIMIT :batch_size
        """), {
            "model_version": model_version,
            "batch_size": settings.EXT_SHADOW_BATCH_SIZE
        })
        snapshots = [dict(r._mapping) for r in snapshots_result.fetchall()]

        if not snapshots:
            record_ext_shadow_rejection(variant_name, "no_pending_snapshots")
            logger.info(
                f"[EXT_SHADOW] ext_shadow_no_snapshots variant={variant_name} "
                f"model_version={model_version} start_at={settings.EXT_SHADOW_START_AT}"
            )
            return {"status": "ok", "inserted": 0, "skipped": 0}

        # Get all team IDs and build match history index
        all_team_ids = set()
        for s in snapshots:
            all_team_ids.add(s['home_team_id'])
            all_team_ids.add(s['away_team_id'])

        # Get league matches for feature calculation (last 365 days from earliest snapshot)
        min_snapshot_at = min(s['snapshot_at'] for s in snapshots)
        earliest_date = min_snapshot_at - timedelta(days=365)

        # ATI FIX v2: Coherencia dataset - away_goals NOT NULL + tainted filter
        league_matches_result = await session.execute(text("""
            SELECT
                m.id,
                m.date,
                m.home_team_id,
                m.away_team_id,
                m.home_goals,
                m.away_goals,
                COALESCE((m.stats->'home'->>'total_shots')::int, 0) as home_shots,
                COALESCE((m.stats->'away'->>'total_shots')::int, 0) as away_shots,
                COALESCE((m.stats->'home'->>'corner_kicks')::int, 0) as home_corners,
                COALESCE((m.stats->'away'->>'corner_kicks')::int, 0) as away_corners
            FROM matches m
            JOIN admin_leagues al ON m.league_id = al.league_id
            WHERE m.status = 'FT'
              AND m.home_goals IS NOT NULL
              AND m.away_goals IS NOT NULL
              AND (m.tainted IS NULL OR m.tainted = false)
              AND al.kind = 'league'
              AND m.date >= :earliest
              AND (m.home_team_id = ANY(:team_ids) OR m.away_team_id = ANY(:team_ids))
            ORDER BY m.date
        """), {"earliest": earliest_date, "team_ids": list(all_team_ids)})
        league_matches = [dict(r._mapping) for r in league_matches_result.fetchall()]

        # Build team match index
        team_index = defaultdict(list)
        for m in league_matches:
            for tid in [m['home_team_id'], m['away_team_id']]:
                team_index[tid].append((m['date'], m))

        # Sort by date for each team
        for tid in team_index:
            team_index[tid].sort(key=lambda x: x[0])

        # Generate predictions
        inserted = 0
        skipped = 0
        errors = 0

        for snap in snapshots:
            try:
                # Calculate features as-of snapshot_at
                features = _calculate_ext_features(
                    snap, team_index, snap['snapshot_at']
                )
                feature_vector = np.array([[features.get(f, 0) for f in EXT_14_FEATURES]])
                probs = model.predict_proba(feature_vector)[0]

                # ATI FIX: RETURNING 1 para detectar insert real vs conflict skip
                # ATI FIX: CAST explícito a jsonb para asyncpg
                result = await session.execute(text("""
                    INSERT INTO predictions_experiments
                    (snapshot_id, match_id, snapshot_at, model_version,
                     home_prob, draw_prob, away_prob, feature_set, created_at)
                    VALUES (:snapshot_id, :match_id, :snapshot_at, :model_version,
                            :home_prob, :draw_prob, :away_prob,
                            CAST(:feature_set AS jsonb), :created_at)
                    ON CONFLICT (snapshot_id, model_version) DO NOTHING
                    RETURNING 1
                """), {
                    "snapshot_id": snap['snapshot_id'],
                    "match_id": snap['match_id'],
                    "snapshot_at": snap['snapshot_at'],
                    "model_version": model_version,
                    "home_prob": float(probs[0]),
                    "draw_prob": float(probs[1]),
                    "away_prob": float(probs[2]),
                    "feature_set": json.dumps(EXT_14_FEATURES),
                    "created_at": snap['snapshot_at'] - timedelta(seconds=1),
                })
                # Si RETURNING devuelve algo → insertó; si None → conflict (skipped)
                if result.fetchone() is not None:
                    inserted += 1
                else:
                    skipped += 1
            except Exception as e:
                logger.warning(f"[EXT_SHADOW] ext-{variant_name} error for snapshot {snap['snapshot_id']}: {e}")
                record_ext_shadow_rejection(variant_name, "insert_error")
                errors += 1

        await session.commit()

        # ATI: Log resumen con batch_size para contexto
        logger.info(
            f"[EXT_SHADOW] ext-{variant_name} run_summary: "
            f"batch_size={settings.EXT_SHADOW_BATCH_SIZE}, "
            f"processed={len(snapshots)}, "
            f"inserted={inserted}, skipped={skipped}, errors={errors}"
        )

        return {"status": "ok", "inserted": inserted, "skipped": skipped, "errors": errors}


def _calculate_ext_features(snap: dict, team_index: dict, snapshot_at: datetime) -> dict:
    """
    Calculate ext features as-of snapshot_at.

    Uses the same 14 features as the ext models were trained on.
    PIT-safe: only uses matches with date < snapshot_at.
    """
    import numpy as np

    home_id = snap['home_team_id']
    away_id = snap['away_team_id']

    # Get team history BEFORE snapshot_at (last 10 matches)
    def get_history(team_id, max_matches=10):
        history = []
        for dt, m in reversed(team_index.get(team_id, [])):
            if dt < snapshot_at:
                history.append(m)
                if len(history) >= max_matches:
                    break
        return history

    home_history = get_history(home_id)
    away_history = get_history(away_id)

    features = {}

    # Home features
    if home_history:
        goals_s, goals_c, shots, corners = [], [], [], []
        for m in home_history:
            if m['home_team_id'] == home_id:
                goals_s.append(m['home_goals'] or 0)
                goals_c.append(m['away_goals'] or 0)
                shots.append(m['home_shots'] or 0)
                corners.append(m['home_corners'] or 0)
            else:
                goals_s.append(m['away_goals'] or 0)
                goals_c.append(m['home_goals'] or 0)
                shots.append(m['away_shots'] or 0)
                corners.append(m['away_corners'] or 0)

        features['home_goals_scored_avg'] = np.mean(goals_s)
        features['home_goals_conceded_avg'] = np.mean(goals_c)
        features['home_shots_avg'] = np.mean(shots)
        features['home_corners_avg'] = np.mean(corners)
        features['home_matches_played'] = len(home_history)

        last_date = home_history[0]['date']
        delta = (snapshot_at - last_date).total_seconds() / 86400
        features['home_rest_days'] = max(1, min(30, delta))
    else:
        features['home_goals_scored_avg'] = 0
        features['home_goals_conceded_avg'] = 0
        features['home_shots_avg'] = 0
        features['home_corners_avg'] = 0
        features['home_rest_days'] = 7
        features['home_matches_played'] = 0

    # Away features
    if away_history:
        goals_s, goals_c, shots, corners = [], [], [], []
        for m in away_history:
            if m['home_team_id'] == away_id:
                goals_s.append(m['home_goals'] or 0)
                goals_c.append(m['away_goals'] or 0)
                shots.append(m['home_shots'] or 0)
                corners.append(m['home_corners'] or 0)
            else:
                goals_s.append(m['away_goals'] or 0)
                goals_c.append(m['home_goals'] or 0)
                shots.append(m['away_shots'] or 0)
                corners.append(m['away_corners'] or 0)

        features['away_goals_scored_avg'] = np.mean(goals_s)
        features['away_goals_conceded_avg'] = np.mean(goals_c)
        features['away_shots_avg'] = np.mean(shots)
        features['away_corners_avg'] = np.mean(corners)
        features['away_matches_played'] = len(away_history)

        last_date = away_history[0]['date']
        delta = (snapshot_at - last_date).total_seconds() / 86400
        features['away_rest_days'] = max(1, min(30, delta))
    else:
        features['away_goals_scored_avg'] = 0
        features['away_goals_conceded_avg'] = 0
        features['away_shots_avg'] = 0
        features['away_corners_avg'] = 0
        features['away_rest_days'] = 7
        features['away_matches_played'] = 0

    # Derived features
    features['goal_diff_avg'] = features['home_goals_scored_avg'] - features['away_goals_scored_avg']
    features['rest_diff'] = features['home_rest_days'] - features['away_rest_days']

    return features


# Legacy alias (calls new generic function)
# ATI: Mantenido para compatibilidad con jobs existentes
_calculate_extc_features = _calculate_ext_features


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
        async with get_session_with_retry(max_retries=3, retry_delay=1.0) as session:
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

                # AUTO-HEALING: Retry missing B predictions now that sensor is ready
                # This fills b_* for rows created when sensor was LEARNING
                try:
                    from app.ml.sensor import retry_missing_b_predictions
                    retry_result = await retry_missing_b_predictions(session, include_ft=False)
                    if retry_result.get("updated", 0) > 0:
                        logger.info(
                            f"[SENSOR] Auto-healing complete: updated={retry_result['updated']}, "
                            f"checked={retry_result['checked']}"
                        )
                except Exception as retry_err:
                    logger.warning(f"[SENSOR] Auto-healing failed: {retry_err}")

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
    start_time = time.time()

    try:
        from app.aggregates.refresh_job import refresh_all_aggregates

        # Use get_session_with_retry to handle Railway connection drops (InterfaceError fix)
        async with get_session_with_retry(max_retries=3, retry_delay=1.0) as session:
            result = await refresh_all_aggregates(session)

            duration_ms = (time.time() - start_time) * 1000
            logger.info(
                f"Aggregates refresh complete: {result['leagues_processed']} leagues, "
                f"{result['baselines_created']} baselines, {result['profiles_created']} profiles"
            )
            record_job_run(job="daily_refresh_aggregates", status="ok", duration_ms=duration_ms)

            if result.get("errors"):
                for err in result["errors"][:5]:  # Log first 5 errors
                    logger.warning(f"  - {err}")

    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        logger.error(f"Aggregates refresh failed: {e}")
        sentry_capture_exception(e, job_id="daily_refresh_aggregates")
        record_job_run(job="daily_refresh_aggregates", status="error", duration_ms=duration_ms)


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
    start_time = time.time()

    try:
        # Use get_session_with_retry to handle Railway connection drops
        async with get_session_with_retry(max_retries=3, retry_delay=1.0) as session:
            pipeline = await create_etl_pipeline(session)
            result = await pipeline.sync_multiple_leagues(
                league_ids=sync_leagues,
                season=CURRENT_SEASON,
                fetch_odds=False,  # Only sync results, not odds
            )

            duration_ms = (time.time() - start_time) * 1000
            logger.info(
                f"Daily sync complete: {result['total_matches_synced']} matches synced "
                f"from {len(sync_leagues)} leagues"
            )
            record_job_run(job="daily_sync_results", status="ok", duration_ms=duration_ms)

    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        logger.error(f"Daily sync failed: {e}")
        sentry_capture_exception(e, job_id="daily_sync_results")
        record_job_run(job="daily_sync_results", status="error", duration_ms=duration_ms)


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


RECALIB_COOLDOWN_REJECTS = 3       # Skip retrain after N consecutive rejects
RECALIB_COOLDOWN_DAYS = 14         # Cooldown duration in days

# Shadow (Two-Stage) retrain constants
SHADOW_RETRAIN_INTERVAL_DAYS = 14     # Retrain every 14 days
SHADOW_RETRAIN_VOLUME_TRIGGER = 1500  # OR: N new FT matches since last approved retrain
SHADOW_COOLDOWN_REJECTS = 3           # Skip after N consecutive rejects
SHADOW_COOLDOWN_DAYS = 14             # Cooldown duration
SHADOW_MODEL_VERSION = "v1.1.0-twostage"
SHADOW_DRAW_WEIGHT = 1.2


async def _check_recalib_cooldown() -> tuple[bool, int]:
    """Check if we're in cooldown from consecutive retrain→reject cycles.

    Returns (in_cooldown: bool, consecutive_rejects: int).
    Cooldown is bypassed if trigger is anomaly_rate (not gold accuracy).
    """
    try:
        from app.database import AsyncSessionLocal
        from sqlalchemy import text

        async with AsyncSessionLocal() as session:
            result = await session.execute(text("""
                SELECT metrics
                FROM job_runs
                WHERE job_name = 'weekly_recalibration'
                  AND status = 'ok'
                  AND metrics IS NOT NULL
                ORDER BY started_at DESC
                LIMIT :n
            """), {"n": RECALIB_COOLDOWN_REJECTS})
            rows = result.fetchall()

        if len(rows) < RECALIB_COOLDOWN_REJECTS:
            return False, 0

        consecutive_rejects = 0
        for row in rows:
            m = row[0] if row[0] else {}
            if m.get("validation_verdict") == "rejected":
                consecutive_rejects += 1
            else:
                break

        return consecutive_rejects >= RECALIB_COOLDOWN_REJECTS, consecutive_rejects
    except Exception as e:
        logger.warning(f"Cooldown check failed (proceeding): {e}")
        return False, 0


async def _record_recalib_run(
    status: str, run_metrics: dict, start_time: float, error: str = None,
) -> None:
    """Persist weekly_recalibration run to DB (session-based) + Prometheus.

    Uses app.jobs.tracking.record_job_run (proven to persist metrics correctly)
    instead of telemetry fire-and-forget which stores JSONB null.
    """
    from datetime import datetime, timedelta
    duration_ms = (time.time() - start_time) * 1000
    job_started_at = datetime.utcnow() - timedelta(milliseconds=duration_ms)
    # DB persist (reliable, session-based)
    try:
        async with get_session_with_retry(max_retries=2, retry_delay=0.5) as s:
            from app.jobs.tracking import record_job_run as record_job_run_db
            await record_job_run_db(
                s, "weekly_recalibration", status, job_started_at,
                error=error, metrics=run_metrics,
            )
    except Exception as db_err:
        logger.warning(f"Failed to persist recalib run to DB: {db_err}")
    # Prometheus (no metrics needed, just counters)
    record_job_run(job="weekly_recalibration", status=status, duration_ms=duration_ms)


async def _get_training_league_ids(session) -> list[int]:
    """Get active domestic league IDs for training cohort from admin_leagues."""
    result = await session.execute(text(
        "SELECT league_id FROM admin_leagues "
        "WHERE is_active = true AND kind = 'league' ORDER BY league_id"
    ))
    return [row[0] for row in result.fetchall()]


# Training cohort: matches from 2023+ in active domestic leagues, league_only features
TRAINING_MIN_DATE = datetime(2023, 1, 1)


async def _check_shadow_cooldown() -> tuple[bool, int]:
    """Check if shadow retrain is in cooldown from consecutive rejects.

    Same logic as _check_recalib_cooldown but for shadow_recalibration job.
    Returns (in_cooldown: bool, consecutive_rejects: int).
    """
    try:
        from app.database import AsyncSessionLocal

        async with AsyncSessionLocal() as session:
            result = await session.execute(text("""
                SELECT metrics
                FROM job_runs
                WHERE job_name = 'shadow_recalibration'
                  AND status = 'ok'
                  AND metrics IS NOT NULL
                ORDER BY started_at DESC
                LIMIT :n
            """), {"n": SHADOW_COOLDOWN_REJECTS})
            rows = result.fetchall()

        if len(rows) < SHADOW_COOLDOWN_REJECTS:
            return False, 0

        consecutive_rejects = 0
        for row in rows:
            m = row[0] if row[0] else {}
            if m.get("validation_verdict") == "rejected":
                consecutive_rejects += 1
            else:
                break

        return consecutive_rejects >= SHADOW_COOLDOWN_REJECTS, consecutive_rejects
    except Exception as e:
        logger.warning(f"Shadow cooldown check failed (proceeding): {e}")
        return False, 0


async def _record_shadow_run(
    status: str, run_metrics: dict, start_time: float, error: str = None,
) -> None:
    """Persist shadow_recalibration run to DB (session-based) + Prometheus.

    Uses app.jobs.tracking.record_job_run (proven to persist metrics correctly)
    instead of telemetry fire-and-forget which stores JSONB null.
    """
    from datetime import timedelta
    duration_ms = (time.time() - start_time) * 1000
    job_started_at = datetime.utcnow() - timedelta(milliseconds=duration_ms)
    try:
        async with get_session_with_retry(max_retries=2, retry_delay=0.5) as s:
            from app.jobs.tracking import record_job_run as record_job_run_db
            await record_job_run_db(
                s, "shadow_recalibration", status, job_started_at,
                error=error, metrics=run_metrics,
            )
    except Exception as db_err:
        logger.warning(f"Failed to persist shadow recalib run to DB: {db_err}")
    record_job_run(job="shadow_recalibration", status=status, duration_ms=duration_ms)


async def _should_trigger_shadow_retrain(session) -> tuple[bool, str]:
    """Evaluate if shadow model retrain should be triggered.

    ATI P0-1: Anchor on validation_verdict='approved', NOT status='ok'
    (ok includes no_retrain/skipped_cooldown runs).

    ATI P0-2: Volume trigger uses cohort-aware count (active leagues, tainted=false).

    Returns (should_trigger, reason).
    """
    # P0-1: Last approved shadow retrain
    result = await session.execute(text("""
        SELECT finished_at FROM job_runs
        WHERE job_name = 'shadow_recalibration'
          AND metrics IS NOT NULL
          AND metrics->>'validation_verdict' = 'approved'
        ORDER BY finished_at DESC LIMIT 1
    """))
    row = result.first()
    last_approved_at = row[0] if row else None

    # First run ever
    if last_approved_at is None:
        return True, "first_run"

    # Interval trigger
    days_since = (datetime.utcnow() - last_approved_at).days
    if days_since >= SHADOW_RETRAIN_INTERVAL_DAYS:
        return True, f"interval_{days_since}d"

    # P0-2: Volume trigger — cohort-aware (active domestic leagues, tainted=false)
    # P1: status='FT' only (matches build_training_dataset which excludes AET/PEN)
    training_league_ids = await _get_training_league_ids(session)
    result = await session.execute(text("""
        SELECT COUNT(*) FROM matches
        WHERE status = 'FT'
          AND home_goals IS NOT NULL AND away_goals IS NOT NULL
          AND (tainted IS NULL OR tainted = false)
          AND league_id = ANY(:league_ids)
          AND date >= :cutoff
    """), {"league_ids": training_league_ids, "cutoff": last_approved_at})
    new_ft_count = result.scalar() or 0

    if new_ft_count >= SHADOW_RETRAIN_VOLUME_TRIGGER:
        return True, f"volume_{new_ft_count}_ft"

    return False, f"no_trigger (days={days_since}, new_ft={new_ft_count})"


async def weekly_recalibration(ml_engine):
    """
    Weekly intelligent recalibration job.
    Runs every Monday at 5:00 AM UTC.

    Steps:
    1. Sync latest fixtures/results
    2. Run audit on recent matches
    3. Update team confidence adjustments
    4. Evaluate if retraining is needed (with cooldown check)
    5. If retraining: build cohort-matched dataset, train, validate, deploy
    """
    logger.info("Starting weekly recalibration job...")
    start_time = time.time()
    # Metrics payload for job_runs (ATI P0-2)
    run_metrics: dict = {}

    try:
        from app.audit import create_audit_service
        from app.ml.recalibration import RecalibrationEngine

        # IMPORTANT (P0): Avoid holding a DB connection while running long CPU work
        # (ml_engine.train). Railway/Postgres may drop idle checked-out connections,
        # which can surface later as "rollback() underlying connection is closed".
        #
        # We split the job into phases with short-lived sessions:
        # - Phase 1: sync + audit + adjustments + dataset build (DB)
        # - Phase 2: training (CPU, no DB session)
        # - Phase 3: validation + snapshot (DB)

        sync_leagues = get_sync_leagues()

        # ── Phase 1: DB work up to dataset ──
        async with get_session_with_retry(max_retries=3, retry_delay=1.0) as session:
            recalibrator = RecalibrationEngine(session)

            # Step 1: Sync latest fixtures/results
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

            # Step 2: Run audit on recent matches
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

            # Parse trigger metrics from reason string for logging
            is_anomaly_trigger = "anomaly" in reason.lower()
            run_metrics["trigger_reason"] = reason

            if not should_retrain:
                logger.info("Skipping retrain - metrics within thresholds")
                run_metrics["validation_verdict"] = "no_retrain"
                await _record_recalib_run("ok", run_metrics, start_time)
                return

            # P0-2: Cooldown — skip retrain if last N runs were all rejected
            # Exception: anomaly_rate triggers always bypass cooldown
            if not is_anomaly_trigger:
                in_cooldown, n_rejects = await _check_recalib_cooldown()
                if in_cooldown:
                    logger.info(
                        f"COOLDOWN: {n_rejects} consecutive rejects, skipping retrain "
                        f"for {RECALIB_COOLDOWN_DAYS}d (non-anomaly trigger)"
                    )
                    run_metrics["validation_verdict"] = "skipped_cooldown"
                    run_metrics["consecutive_rejects"] = n_rejects
                    await _record_recalib_run("ok", run_metrics, start_time)
                    return

            # Step 5a: Build training dataset (DB)
            # ATI P0: Cohort-matched — active domestic leagues + 2023+ + league_only features
            training_league_ids = await _get_training_league_ids(session)
            logger.info(
                f"Triggering retrain: {reason} | cohort: "
                f"league_ids_count={len(training_league_ids)}, "
                f"min_date={TRAINING_MIN_DATE.date()}, league_only=True"
            )
            feature_engineer = FeatureEngineer(session=session)
            df = await feature_engineer.build_training_dataset(
                league_only=True,
                league_ids=training_league_ids,
                min_date=TRAINING_MIN_DATE,
            )

        # Cohort metadata for traceability
        cohort_meta = {
            "dataset_mode": "league_only_active_domestic_recent",
            "league_only": True,
            "league_ids_source": "admin_leagues(is_active=true, kind='league')",
            "league_ids_count": len(training_league_ids),
            "min_date": str(TRAINING_MIN_DATE.date()),
        }
        run_metrics.update(cohort_meta)

        # ── Phase 2: CPU work (no DB session held) ──
        if len(df) < 100:
            logger.error(f"Insufficient training data: {len(df)} samples")
            run_metrics["samples_trained"] = len(df)
            run_metrics["validation_verdict"] = "insufficient_data"
            await _record_recalib_run("error", run_metrics, start_time)
            return

        run_metrics["samples_trained"] = len(df)

        # Train in executor to avoid blocking the event loop
        from concurrent.futures import ThreadPoolExecutor

        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as executor:
            train_result = await loop.run_in_executor(executor, ml_engine.train, df)

        new_brier = train_result["brier_score"]
        logger.info(f"Training complete: Brier Score = {new_brier:.4f}")
        run_metrics["new_brier"] = round(new_brier, 6)

        # ── Phase 3: DB work (validate + snapshot) ──
        async with get_session_with_retry(max_retries=3, retry_delay=1.0) as session:
            recalibrator = RecalibrationEngine(session)

            # Step 6: Validate new model against baseline AND active (P1)
            is_valid, validation_msg = await recalibrator.validate_new_model(new_brier)
            run_metrics["baseline_brier_used"] = 0.2063  # BRIER_SCORE_BASELINE constant

            # P1: Anti-regression check vs active model
            active_snapshot = await recalibrator.get_active_snapshot()
            if active_snapshot:
                run_metrics["active_brier"] = round(active_snapshot.brier_score, 6)
                if is_valid and new_brier >= active_snapshot.brier_score:
                    is_valid = False
                    validation_msg = (
                        f"New Brier ({new_brier:.4f}) >= active "
                        f"({active_snapshot.brier_score:.4f}) - REJECTED (anti-regression)"
                    )

            logger.info(f"Validation result: {validation_msg}")
            run_metrics["validation_msg"] = validation_msg

            if not is_valid:
                logger.warning("ROLLBACK: New model rejected - keeping previous version")
                ml_engine.load_model()
                run_metrics["validation_verdict"] = "rejected"
                await _record_recalib_run("ok", run_metrics, start_time)
                return

            # Step 7: Create snapshot and activate new model
            training_config = {
                **cohort_meta,
                "samples_trained": train_result["samples_trained"],
                "cv_scores": train_result["cv_scores"],
                "brier_score": round(new_brier, 6),
            }
            snapshot = await recalibrator.create_snapshot(
                model_version=ml_engine.model_version,
                model_path=train_result["model_path"],
                brier_score=new_brier,
                cv_scores=train_result["cv_scores"],
                samples_trained=train_result["samples_trained"],
                training_config=training_config,
            )
            logger.info(f"New model deployed: {snapshot.model_version} (Brier: {new_brier:.4f})")

        run_metrics["validation_verdict"] = "approved"
        await _record_recalib_run("ok", run_metrics, start_time)

    except Exception as e:
        logger.error(f"Weekly recalibration failed: {e}")
        sentry_capture_exception(e, job_id="weekly_recalibration")
        run_metrics["validation_verdict"] = "error"
        await _record_recalib_run("error", run_metrics, start_time, error=str(e))


# =============================================================================
# SHADOW (TWO-STAGE) AUTOMATIC RETRAINING
# =============================================================================
# Separate from weekly_recalibration (Model A). Does NOT affect production
# predictions (is_active always False). ATI-approved 2026-02-09.


async def shadow_recalibration():
    """
    Shadow (Two-Stage) model automatic retraining.
    Runs every Tuesday at 5:00 AM UTC (bi-weekly via internal interval check).

    Separate from weekly_recalibration (Model A).
    Does NOT affect production predictions (is_active remains False).

    Steps:
    1. Check trigger (interval >= 14d OR volume >= 1500 new FT)
    2. Check cooldown (3 consecutive rejects → skip 14d)
    3. Build cohort-matched dataset (same as Model A)
    4. Train TwoStageEngine in ThreadPoolExecutor
    5. Validate vs last shadow snapshot (rebaseline if cohort changed)
    6. Save snapshot + hot-reload shadow engine
    """
    logger.info("[SHADOW_RECALIB] Starting shadow recalibration job...")
    start_time = time.time()
    run_metrics: dict = {}

    try:
        from app.ml.engine import TwoStageEngine
        from app.models import ModelSnapshot

        # ── Phase 1: DB work (trigger + cooldown + dataset) ──
        async with get_session_with_retry(max_retries=3, retry_delay=1.0) as session:
            # Step 1: Check trigger
            should_trigger, reason = await _should_trigger_shadow_retrain(session)
            run_metrics["trigger_reason"] = reason
            logger.info(f"[SHADOW_RECALIB] Trigger evaluation: {reason}")

            if not should_trigger:
                logger.info("[SHADOW_RECALIB] Skipping - no trigger")
                run_metrics["validation_verdict"] = "no_retrain"
                await _record_shadow_run("ok", run_metrics, start_time)
                return

            # Step 2: Cooldown check
            in_cooldown, n_rejects = await _check_shadow_cooldown()
            if in_cooldown:
                logger.info(
                    f"[SHADOW_RECALIB] COOLDOWN: {n_rejects} consecutive rejects, "
                    f"skipping for {SHADOW_COOLDOWN_DAYS}d"
                )
                run_metrics["validation_verdict"] = "skipped_cooldown"
                run_metrics["consecutive_rejects"] = n_rejects
                await _record_shadow_run("ok", run_metrics, start_time)
                return

            # Step 3: Build cohort-matched dataset (same as Model A)
            training_league_ids = await _get_training_league_ids(session)
            logger.info(
                f"[SHADOW_RECALIB] Building dataset: "
                f"league_ids_count={len(training_league_ids)}, "
                f"min_date={TRAINING_MIN_DATE.date()}, league_only=True"
            )
            feature_engineer = FeatureEngineer(session=session)
            df = await feature_engineer.build_training_dataset(
                league_only=True,
                league_ids=training_league_ids,
                min_date=TRAINING_MIN_DATE,
            )

        # Cohort metadata for traceability
        cohort_meta = {
            "dataset_mode": "league_only_active_domestic_recent",
            "league_only": True,
            "league_ids_source": "admin_leagues(is_active=true, kind='league')",
            "league_ids_count": len(training_league_ids),
            "min_date": str(TRAINING_MIN_DATE.date()),
        }
        run_metrics.update(cohort_meta)

        # ── Phase 2: CPU work (no DB session held) ──
        if len(df) < 100:
            logger.error(f"[SHADOW_RECALIB] Insufficient data: {len(df)} samples")
            run_metrics["samples_trained"] = len(df)
            run_metrics["validation_verdict"] = "insufficient_data"
            await _record_shadow_run("error", run_metrics, start_time)
            return

        run_metrics["samples_trained"] = len(df)

        engine = TwoStageEngine(
            model_version=SHADOW_MODEL_VERSION,
            draw_weight=SHADOW_DRAW_WEIGHT,
        )

        from concurrent.futures import ThreadPoolExecutor
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as executor:
            train_result = await loop.run_in_executor(executor, engine.train, df)

        new_brier = train_result["brier_score"]
        cv_scores = train_result["cv_scores"]
        logger.info(f"[SHADOW_RECALIB] Training complete: Brier = {new_brier:.4f}")
        run_metrics["new_brier"] = round(new_brier, 6)
        run_metrics["cv_scores"] = cv_scores

        # ── Phase 3: DB work (validate + snapshot + hot-reload) ──
        async with get_session_with_retry(max_retries=3, retry_delay=1.0) as session:
            # Get last shadow snapshot for comparison
            from sqlalchemy import select
            result = await session.execute(
                select(ModelSnapshot)
                .where(ModelSnapshot.model_version.like("%twostage%"))
                .order_by(ModelSnapshot.created_at.desc())
                .limit(1)
            )
            last_shadow = result.scalar_one_or_none()

            # P0-3: Rebaseline check — if previous snapshot lacks dataset_mode,
            # it was trained on old regime (69K). Don't compare apples to oranges.
            is_rebaseline = False
            if last_shadow:
                prev_config = last_shadow.training_config or {}
                if not prev_config.get("dataset_mode"):
                    is_rebaseline = True
                    logger.info(
                        f"[SHADOW_RECALIB] REBASELINE: Previous snapshot "
                        f"(id={last_shadow.id}, brier={last_shadow.brier_score:.4f}) "
                        f"lacks dataset_mode — old regime. Auto-approving."
                    )

            run_metrics["is_rebaseline"] = is_rebaseline

            # Retrain gate
            is_valid = True
            if last_shadow and not is_rebaseline:
                run_metrics["last_shadow_brier"] = round(last_shadow.brier_score, 6)
                if new_brier >= last_shadow.brier_score:
                    is_valid = False
                    validation_msg = (
                        f"New Brier ({new_brier:.4f}) >= last shadow "
                        f"({last_shadow.brier_score:.4f}) - REJECTED"
                    )
                else:
                    validation_msg = (
                        f"Improved: {last_shadow.brier_score:.4f} → {new_brier:.4f} "
                        f"(Δ={last_shadow.brier_score - new_brier:.4f}) - APPROVED"
                    )
            elif is_rebaseline:
                validation_msg = f"REBASELINE: New cohort baseline Brier = {new_brier:.4f} - APPROVED"
            else:
                validation_msg = f"First shadow snapshot: Brier = {new_brier:.4f} - APPROVED"

            logger.info(f"[SHADOW_RECALIB] Validation: {validation_msg}")
            run_metrics["validation_msg"] = validation_msg

            if not is_valid:
                logger.warning("[SHADOW_RECALIB] New shadow model rejected")
                run_metrics["validation_verdict"] = "rejected"
                await _record_shadow_run("ok", run_metrics, start_time)
                return

            # Save snapshot — insert directly, NOT via create_snapshot()
            # (create_snapshot sets is_active=False on ALL snapshots, killing Model A)
            blob = engine.save_to_bytes()
            training_config = {
                **cohort_meta,
                "architecture": "two_stage",
                "draw_weight": SHADOW_DRAW_WEIGHT,
                "samples_trained": len(df),
                "cv_scores": cv_scores,
                "brier_score": round(new_brier, 6),
                "is_rebaseline": is_rebaseline,
            }

            snapshot = ModelSnapshot(
                model_version=SHADOW_MODEL_VERSION,
                model_path="db_blob",
                model_blob=blob,
                brier_score=new_brier,
                cv_brier_scores={"scores": cv_scores},
                samples_trained=len(df),
                is_active=False,
                is_baseline=False,
                training_config=training_config,
            )
            session.add(snapshot)
            await session.commit()
            await session.refresh(snapshot)
            logger.info(
                f"[SHADOW_RECALIB] Snapshot saved: id={snapshot.id}, "
                f"brier={new_brier:.4f}"
            )

        # Hot-reload shadow engine (P0-5: safe fallback)
        from app.ml.shadow import reload_shadow_engine
        if reload_shadow_engine(blob):
            run_metrics["hot_reload"] = "success"
        else:
            run_metrics["hot_reload"] = "failed"
            run_metrics["validation_verdict"] = "reload_failed"
            logger.warning(
                "[SHADOW_RECALIB] Hot-reload failed — snapshot saved, "
                "engine will pick up on next restart"
            )
            await _record_shadow_run("ok", run_metrics, start_time)
            return

        run_metrics["validation_verdict"] = "approved"
        run_metrics["snapshot_id"] = snapshot.id
        await _record_shadow_run("ok", run_metrics, start_time)
        logger.info(
            f"[SHADOW_RECALIB] Complete: Brier {new_brier:.4f}, "
            f"snapshot_id={snapshot.id}, hot-reload OK"
        )

    except Exception as e:
        logger.error(f"[SHADOW_RECALIB] Failed: {e}")
        sentry_capture_exception(e, job_id="shadow_recalibration")
        run_metrics["validation_verdict"] = "error"
        await _record_shadow_run("error", run_metrics, start_time, error=str(e))


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
# FINISHED MATCH PLAYER STATS SYNC (API-Football /fixtures/players)
# =============================================================================
# Captures per-player statistics and rating for recently finished matches.
# Writes to match_player_stats (data layer) and is guarded by feature flags.

async def capture_finished_match_player_stats() -> dict:
    """
    Going-forward job: ingest per-player per-match stats from API-Football /fixtures/players.

    Guardrails:
    - PLAYER_STATS_SYNC_ENABLED: If false, job returns immediately (default OFF during incubation)
    - PLAYER_STATS_SYNC_DELAY_HOURS: Only process matches finished >= delay hours ago
    - PLAYER_STATS_SYNC_MAX_CALLS: Hard cap on API calls per run

    PIT note:
    The API-Football rating is POST-match. It must ONLY be used for pre-match PTS with:
        matches.date < asof_timestamp
    Never include same-match ratings in pre-match feature computation.

    Run frequency: Every 60 minutes (scheduler job), but disabled by default.
    """
    import json
    import os
    import time
    from datetime import datetime
    from pathlib import Path

    from app.config import get_settings

    settings = get_settings()
    start_time = time.time()

    enabled = os.environ.get("PLAYER_STATS_SYNC_ENABLED", str(settings.PLAYER_STATS_SYNC_ENABLED)).lower()
    if enabled in ("false", "0", "no"):
        logger.info("Player stats sync job disabled via PLAYER_STATS_SYNC_ENABLED=false")
        return {"status": "disabled"}

    delay_hours = int(os.environ.get("PLAYER_STATS_SYNC_DELAY_HOURS", settings.PLAYER_STATS_SYNC_DELAY_HOURS))
    max_calls = int(os.environ.get("PLAYER_STATS_SYNC_MAX_CALLS", settings.PLAYER_STATS_SYNC_MAX_CALLS))

    metrics = {
        "checked": 0,
        "api_calls": 0,
        "matches_ingested": 0,
        "rows_upserted": 0,
        "skipped_no_external_id": 0,
        "skipped_no_data": 0,
        "skipped_no_players": 0,
        "errors": 0,
        "started_at": datetime.utcnow().isoformat(),
        "delay_hours": delay_hours,
        "max_calls": max_calls,
    }

    def _parse_smallint(value) -> Optional[int]:
        if value is None:
            return None
        try:
            if isinstance(value, bool):
                return int(value)
            if isinstance(value, (int,)):
                return int(value)
            if isinstance(value, (float, Decimal)):
                return int(value)
            if isinstance(value, str):
                s = value.strip()
                if s in ("", "-", "null", "None"):
                    return None
                # Handle stoppage-time formats like "90+4"
                if "+" in s:
                    s = s.split("+", 1)[0].strip()
                if s.endswith("%"):
                    s = s[:-1]
                return int(float(s))
        except Exception:
            return None
        return None

    def _parse_rating(value) -> Optional[Decimal]:
        if value is None:
            return None
        try:
            if isinstance(value, Decimal):
                return value.quantize(Decimal("0.01"))
            if isinstance(value, (int, float)):
                return Decimal(str(value)).quantize(Decimal("0.01"))
            if isinstance(value, str):
                s = value.strip()
                if s in ("", "-", "null", "None"):
                    return None
                return Decimal(s).quantize(Decimal("0.01"))
        except Exception:
            return None
        return None

    insert_sql = text("""
        INSERT INTO match_player_stats (
            match_id, player_external_id, player_name,
            team_external_id, team_id, match_date,
            rating, minutes, position, is_substitute, is_captain,
            goals, assists, saves,
            shots_total, shots_on_target,
            passes_total, passes_key, passes_accuracy,
            tackles, interceptions, blocks,
            duels_total, duels_won,
            dribbles_attempts, dribbles_success,
            fouls_drawn, fouls_committed,
            yellow_cards, red_cards,
            raw_json, captured_at
        ) VALUES (
            :match_id, :player_external_id, :player_name,
            :team_external_id, :team_id, :match_date,
            :rating, :minutes, :position, :is_substitute, :is_captain,
            :goals, :assists, :saves,
            :shots_total, :shots_on_target,
            :passes_total, :passes_key, :passes_accuracy,
            :tackles, :interceptions, :blocks,
            :duels_total, :duels_won,
            :dribbles_attempts, :dribbles_success,
            :fouls_drawn, :fouls_committed,
            :yellow_cards, :red_cards,
            CAST(:raw_json AS JSONB), NOW()
        )
        ON CONFLICT (match_id, player_external_id)
        DO UPDATE SET
            player_name = EXCLUDED.player_name,
            team_external_id = EXCLUDED.team_external_id,
            team_id = COALESCE(EXCLUDED.team_id, match_player_stats.team_id),
            match_date = COALESCE(EXCLUDED.match_date, match_player_stats.match_date),
            rating = EXCLUDED.rating,
            minutes = EXCLUDED.minutes,
            position = EXCLUDED.position,
            is_substitute = EXCLUDED.is_substitute,
            is_captain = EXCLUDED.is_captain,
            goals = EXCLUDED.goals,
            assists = EXCLUDED.assists,
            saves = EXCLUDED.saves,
            shots_total = EXCLUDED.shots_total,
            shots_on_target = EXCLUDED.shots_on_target,
            passes_total = EXCLUDED.passes_total,
            passes_key = EXCLUDED.passes_key,
            passes_accuracy = EXCLUDED.passes_accuracy,
            tackles = EXCLUDED.tackles,
            interceptions = EXCLUDED.interceptions,
            blocks = EXCLUDED.blocks,
            duels_total = EXCLUDED.duels_total,
            duels_won = EXCLUDED.duels_won,
            dribbles_attempts = EXCLUDED.dribbles_attempts,
            dribbles_success = EXCLUDED.dribbles_success,
            fouls_drawn = EXCLUDED.fouls_drawn,
            fouls_committed = EXCLUDED.fouls_committed,
            yellow_cards = EXCLUDED.yellow_cards,
            red_cards = EXCLUDED.red_cards,
            raw_json = EXCLUDED.raw_json,
            captured_at = NOW()
    """)

    try:
        async with AsyncSessionLocal() as session:
            # Pre-load team external_id → internal id mapping
            team_rows = (await session.execute(text("""
                SELECT id, external_id FROM teams WHERE external_id IS NOT NULL
            """))).fetchall()
            team_map = {r.external_id: r.id for r in team_rows}

            # Select finished matches older than delay with no player stats yet
            result = await session.execute(text("""
                SELECT m.id, m.external_id, m.date
                FROM matches m
                WHERE m.status IN ('FT', 'AET', 'PEN')
                  AND m.date <= NOW() - INTERVAL ':delay hours'
                  AND m.external_id IS NOT NULL
                  AND NOT EXISTS (
                      SELECT 1 FROM match_player_stats mps WHERE mps.match_id = m.id
                  )
                ORDER BY m.date DESC
                LIMIT :lim
            """.replace(":delay", str(delay_hours))), {"lim": max_calls})

            matches = result.fetchall()
            metrics["checked"] = len(matches)

            if not matches:
                duration_ms = (time.time() - start_time) * 1000
                record_job_run(job="player_stats_sync", status="ok", duration_ms=duration_ms)
                return {**metrics, "status": "no_matches", "duration_ms": round(duration_ms, 1)}

            provider = APIFootballProvider()
            try:
                for m in matches:
                    match_id = m.id
                    fixture_id = m.external_id
                    match_date = m.date.date() if getattr(m, "date", None) else None

                    if not fixture_id:
                        metrics["skipped_no_external_id"] += 1
                        continue

                    if metrics["api_calls"] >= max_calls:
                        break

                    try:
                        resp = await provider.get_fixture_players(int(fixture_id))
                        metrics["api_calls"] += 1
                    except Exception as e:
                        metrics["errors"] += 1
                        logger.warning(f"Player stats sync: API error match_id={match_id}: {e}")
                        continue

                    if not resp:
                        metrics["skipped_no_data"] += 1
                        continue

                    # Deduplicate per fixture (GDT #1): keep best row per player_external_id.
                    # Prefer non-null rating; otherwise prefer higher minutes.
                    rows_by_player: dict[int, dict] = {}
                    for team_block in resp:
                        team = team_block.get("team") or {}
                        team_external_id = team.get("id")
                        team_id = team_map.get(team_external_id) if team_external_id is not None else None

                        for pl in (team_block.get("players") or []):
                            player = pl.get("player") or {}
                            player_external_id = player.get("id")
                            if not player_external_id:
                                continue

                            stats_list = pl.get("statistics") or []
                            st = stats_list[0] if stats_list else {}
                            games = st.get("games") or {}

                            rating = _parse_rating(games.get("rating"))
                            minutes = _parse_smallint(games.get("minutes"))
                            position = games.get("position")

                            # Ghost filter (GDT #2): bench warmers → force rating NULL
                            if minutes is None or minutes == 0:
                                rating = None

                            row = {
                                "match_id": match_id,
                                "player_external_id": int(player_external_id),
                                "player_name": player.get("name"),
                                "team_external_id": int(team_external_id) if team_external_id is not None else None,
                                "team_id": int(team_id) if team_id is not None else None,
                                "match_date": match_date,
                                "rating": rating,
                                "minutes": minutes,
                                "position": position,
                                "is_substitute": games.get("substitute"),
                                "is_captain": games.get("captain"),
                                "goals": _parse_smallint((st.get("goals") or {}).get("total")),
                                "assists": _parse_smallint((st.get("goals") or {}).get("assists")),
                                "saves": _parse_smallint((st.get("goals") or {}).get("saves")),
                                "shots_total": _parse_smallint((st.get("shots") or {}).get("total")),
                                "shots_on_target": _parse_smallint((st.get("shots") or {}).get("on")),
                                "passes_total": _parse_smallint((st.get("passes") or {}).get("total")),
                                "passes_key": _parse_smallint((st.get("passes") or {}).get("key")),
                                "passes_accuracy": _parse_smallint((st.get("passes") or {}).get("accuracy")),
                                "tackles": _parse_smallint((st.get("tackles") or {}).get("total")),
                                "interceptions": _parse_smallint((st.get("tackles") or {}).get("interceptions")),
                                "blocks": _parse_smallint((st.get("tackles") or {}).get("blocks")),
                                "duels_total": _parse_smallint((st.get("duels") or {}).get("total")),
                                "duels_won": _parse_smallint((st.get("duels") or {}).get("won")),
                                "dribbles_attempts": _parse_smallint((st.get("dribbles") or {}).get("attempts")),
                                "dribbles_success": _parse_smallint((st.get("dribbles") or {}).get("success")),
                                "fouls_drawn": _parse_smallint((st.get("fouls") or {}).get("drawn")),
                                "fouls_committed": _parse_smallint((st.get("fouls") or {}).get("committed")),
                                "yellow_cards": _parse_smallint((st.get("cards") or {}).get("yellow")),
                                "red_cards": _parse_smallint((st.get("cards") or {}).get("red")),
                                "raw_json": json.dumps(st),
                            }

                            pid = row["player_external_id"]
                            existing = rows_by_player.get(pid)
                            if not existing:
                                rows_by_player[pid] = row
                            else:
                                # Prefer row with a rating
                                if existing.get("rating") is None and row.get("rating") is not None:
                                    rows_by_player[pid] = row
                                else:
                                    # Prefer higher minutes when available
                                    m_new = row.get("minutes")
                                    m_old = existing.get("minutes")
                                    if m_new is not None and (m_old is None or m_new > m_old):
                                        rows_by_player[pid] = row

                    rows = list(rows_by_player.values())
                    if not rows:
                        metrics["skipped_no_players"] += 1
                        continue

                    # Atomic per match: savepoint + commit once match rows are upserted
                    await session.execute(text("SAVEPOINT sp_player_stats_match"))
                    try:
                        await session.execute(insert_sql, rows)
                        await session.execute(text("RELEASE SAVEPOINT sp_player_stats_match"))
                        await session.commit()
                        metrics["matches_ingested"] += 1
                        metrics["rows_upserted"] += len(rows)
                    except Exception as e:
                        await session.execute(text("ROLLBACK TO SAVEPOINT sp_player_stats_match"))
                        await session.commit()
                        metrics["errors"] += 1
                        logger.warning(f"Player stats sync: DB upsert failed match_id={match_id}: {e}")
                        continue

            finally:
                await provider.close()

        duration_ms = (time.time() - start_time) * 1000
        metrics["completed_at"] = datetime.utcnow().isoformat()
        metrics["duration_ms"] = round(duration_ms, 1)
        record_job_run(job="player_stats_sync", status="ok", duration_ms=duration_ms)

        # Save log file (best-effort)
        try:
            logs_dir = Path("logs")
            logs_dir.mkdir(exist_ok=True)
            log_file = logs_dir / f"player_stats_sync_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.log"
            with open(log_file, "w") as f:
                f.write(json.dumps(metrics, indent=2))
            metrics["log_file"] = str(log_file)
        except Exception:
            pass

        return {**metrics, "status": "completed"}

    except APIBudgetExceeded as e:
        duration_ms = (time.time() - start_time) * 1000
        logger.warning(f"Player stats sync stopped: {e}. Budget status: {get_api_budget_status()}")
        record_job_run(job="player_stats_sync", status="budget_exceeded", duration_ms=duration_ms)
        metrics["budget_status"] = get_api_budget_status()
        return {**metrics, "status": "budget_exceeded", "error": str(e)}
    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        logger.error(f"Player stats sync failed: {e}")
        sentry_capture_exception(e, job_id="player_stats_sync", metrics=metrics)
        record_job_run(job="player_stats_sync", status="error", duration_ms=duration_ms)
        return {**metrics, "status": "error", "error": str(e)}


# =============================================================================
# HISTORICAL STATS BACKFILL JOB
# =============================================================================
# Backfills stats for matches from 2023-08-01 onwards that have NULL stats.
# This addresses the gap from stats_backfill being added late (2026-01-09)
# with only 72h lookback, leaving ~23,000 historical matches without stats.
#
# Runs every hour with a small batch (500) to avoid timeouts and stay within budget.
# Stateless design: always picks first N matches with NULL stats, ordered by ID.
#
# IMPORTANT: When API returns no stats for a match, we mark it with
# {"_no_stats": true} so we don't retry it. This avoids wasting API calls
# on matches that will never have stats (friendlies, old matches, etc.)


async def historical_stats_backfill() -> dict:
    """
    Backfill historical match stats for training data quality.

    This job fills stats for matches since 2023-08-01 that have NULL stats.
    Unlike the regular stats_backfill (72h lookback), this covers all historical data.

    Guardrails:
    - HISTORICAL_STATS_BACKFILL_ENABLED: If false, job returns immediately
    - HISTORICAL_STATS_BACKFILL_BATCH_SIZE: Max matches per run (default 500)
    - Rate limiting: 0.5s between API calls
    - Stops on 429 or budget exceeded

    Design:
    - Stateless: queries matches with NULL stats ordered by ID
    - Idempotent: safe to run multiple times, naturally advances
    - Auto-completes: returns early when no matches left
    - Marks matches without stats as {"_no_stats": true} to skip on future runs
    """
    import asyncio
    import json
    import os
    import time
    from datetime import date

    import httpx
    from app.config import get_settings

    settings = get_settings()
    start_time = time.time()

    # Check if job is enabled
    enabled = os.environ.get("HISTORICAL_STATS_BACKFILL_ENABLED", "true").lower()
    if enabled in ("false", "0", "no"):
        logger.info("Historical stats backfill job disabled")
        return {"status": "disabled"}

    # Configuration
    CUTOFF_DATE = date(2023, 8, 1)
    BATCH_SIZE = int(os.environ.get("HISTORICAL_STATS_BACKFILL_BATCH_SIZE", "500"))
    REQUEST_DELAY = 0.5

    # All leagues from competitions.py
    ALL_LEAGUES = [
        1, 34, 32, 31, 30, 29, 33, 37,  # World Cup & Qualifiers
        39, 140, 135, 78, 61,  # Top 5
        2, 13,  # UCL, Libertadores
        9, 4, 5, 22, 6, 7,  # International
        40, 88, 94, 144, 203,  # Secondary + Süper Lig
        253, 307,  # MLS, Saudi
        3, 848,  # Europa, Conference
        71, 262, 128,  # LATAM Pack1
        239, 242, 250, 252, 265, 268, 270, 281, 299, 344,  # LATAM Pack2 (252=PY Clausura, 270=UY Clausura)
        11,  # Sudamericana
        143, 45,  # Domestic Cups
        10,  # Friendlies
    ]

    metrics = {
        "fetched": 0,
        "updated": 0,
        "marked_no_stats": 0,
        "errors": 0,
        "api_calls": 0,
        "started_at": datetime.utcnow().isoformat(),
    }

    try:
        async with AsyncSessionLocal() as session:
            # Get remaining count (exclude already marked as no_stats)
            result = await session.execute(text("""
                SELECT COUNT(*) as cnt
                FROM matches
                WHERE status IN ('FT', 'AET', 'PEN')
                  AND date >= :cutoff_date
                  AND league_id = ANY(:leagues)
                  AND external_id IS NOT NULL
                  AND (stats IS NULL OR stats::text = '{}' OR stats::text = 'null')
            """), {"cutoff_date": CUTOFF_DATE, "leagues": ALL_LEAGUES})
            remaining_before = result.scalar() or 0

            if remaining_before == 0:
                logger.info("Historical stats backfill: COMPLETE - all matches processed!")
                record_job_run(job="historical_stats_backfill", status="complete", duration_ms=0)
                return {"status": "complete", "remaining": 0}

            # Get matches needing stats (exclude already marked)
            result = await session.execute(text("""
                SELECT id, external_id
                FROM matches
                WHERE status IN ('FT', 'AET', 'PEN')
                  AND date >= :cutoff_date
                  AND league_id = ANY(:leagues)
                  AND external_id IS NOT NULL
                  AND (stats IS NULL OR stats::text = '{}' OR stats::text = 'null')
                ORDER BY id ASC
                LIMIT :limit
            """), {"cutoff_date": CUTOFF_DATE, "leagues": ALL_LEAGUES, "limit": BATCH_SIZE})
            matches = result.fetchall()

            if not matches:
                logger.info("Historical stats backfill: no matches to process")
                return {"status": "complete", "remaining": remaining_before}

            logger.info(f"Historical stats backfill: processing {len(matches)} matches (remaining: {remaining_before})")

            api_key = settings.RAPIDAPI_KEY
            batch_count = 0
            COMMIT_BATCH = 100

            async with httpx.AsyncClient() as client:
                for match in matches:
                    match_id, external_id = match

                    try:
                        # Fetch stats from API-Football
                        url = "https://v3.football.api-sports.io/fixtures/statistics"
                        headers = {"x-apisports-key": api_key}
                        params = {"fixture": external_id}

                        response = await client.get(url, headers=headers, params=params, timeout=30)
                        metrics["api_calls"] += 1

                        if response.status_code == 429:
                            logger.warning("Historical stats backfill: 429 rate limit, stopping")
                            break

                        if response.status_code != 200:
                            metrics["errors"] += 1
                            continue

                        data = response.json()
                        results = data.get("response", [])

                        if not results or len(results) < 2:
                            # Mark as no_stats so we don't retry this match
                            metrics["marked_no_stats"] += 1
                            await session.execute(text("""
                                UPDATE matches
                                SET stats = CAST(:stats_json AS JSON)
                                WHERE id = :match_id
                            """), {"match_id": match_id, "stats_json": json.dumps({"_no_stats": True})})
                            batch_count += 1
                            if batch_count >= COMMIT_BATCH:
                                await session.commit()
                                batch_count = 0
                            await asyncio.sleep(REQUEST_DELAY)
                            continue

                        # Parse stats
                        stats = {"home": {}, "away": {}}
                        for i, team_stats in enumerate(results):
                            statistics = team_stats.get("statistics", [])
                            side = "home" if i == 0 else "away"
                            for stat in statistics:
                                stat_type = stat.get("type", "").lower().replace(" ", "_")
                                value = stat.get("value")
                                if value and isinstance(value, str) and value.endswith("%"):
                                    try:
                                        value = float(value.rstrip("%"))
                                    except ValueError:
                                        pass
                                stats[side][stat_type] = value

                        if stats["home"] or stats["away"]:
                            metrics["fetched"] += 1
                            await session.execute(text("""
                                UPDATE matches
                                SET stats = CAST(:stats_json AS JSON)
                                WHERE id = :match_id
                            """), {"match_id": match_id, "stats_json": json.dumps(stats)})
                            metrics["updated"] += 1

                        # Batch commit
                        batch_count += 1
                        if batch_count >= COMMIT_BATCH:
                            await session.commit()
                            batch_count = 0

                        # Rate limiting
                        await asyncio.sleep(REQUEST_DELAY)

                    except Exception as e:
                        metrics["errors"] += 1
                        logger.warning(f"Historical stats backfill error for match {match_id}: {e}")

            # Final commit
            await session.commit()

            # Get remaining count after
            result = await session.execute(text("""
                SELECT COUNT(*) as cnt
                FROM matches
                WHERE status IN ('FT', 'AET', 'PEN')
                  AND date >= :cutoff_date
                  AND league_id = ANY(:leagues)
                  AND external_id IS NOT NULL
                  AND (stats IS NULL OR stats::text = '{}' OR stats::text = 'null')
            """), {"cutoff_date": CUTOFF_DATE, "leagues": ALL_LEAGUES})
            remaining_after = result.scalar() or 0

        # Log summary
        duration_ms = (time.time() - start_time) * 1000
        progress = remaining_before - remaining_after
        pct_complete = round((1 - remaining_after / max(remaining_before, 1)) * 100, 1)

        logger.info(
            f"Historical stats backfill: "
            f"api_calls={metrics['api_calls']}, updated={metrics['updated']}, "
            f"marked_no_stats={metrics['marked_no_stats']}, errors={metrics['errors']}, "
            f"progress={progress}, remaining={remaining_after} ({pct_complete}% total complete)"
        )

        record_job_run(job="historical_stats_backfill", status="ok", duration_ms=duration_ms)

        return {
            "status": "ok",
            "metrics": metrics,
            "remaining_before": remaining_before,
            "remaining_after": remaining_after,
            "progress": progress,
            "duration_ms": round(duration_ms, 1),
        }

    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        logger.error(f"Historical stats backfill failed: {e}")
        sentry_capture_exception(e, job_id="historical_stats_backfill", metrics=metrics)
        record_job_run(job="historical_stats_backfill", status="error", duration_ms=duration_ms)
        return {"status": "error", "error": str(e), "metrics": metrics}


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
# CLV SCORING JOB (Phase 2: P2-04)
# =============================================================================
# Scores CLV for predictions of recently finished matches.
# Uses canonical bookmaker (Bet365 > Pinnacle > 1xBet) from odds_history.
# CLV_k = ln(odds_asof / odds_close) per outcome. Positive = timing edge.


async def canonical_odds_sweeper() -> dict:
    """
    Materialize canonical odds for recently changed matches.

    Runs every 6 hours. Scope: matches modified in last 7 days + next 3 days.
    Anti-downgrade guardrail: if a row already has P1/P2/P3 data, NEVER overwrite
    with a lower-priority source (P4-P7). Quality can only go UP, never DOWN.

    Cascade (Market Truth > System Myopia):
      P1: FDUK Pinnacle (raw_odds_1x2 where provider='fduk', bookmaker='Pinnacle')
      P2: FDUK B365 / OddsPortal (raw_odds_1x2 excl. Pinnacle)
      P3: prediction_clv / match_odds_snapshot (bet365 frozen)
      P4: odds_snapshots Bet365_live
      P5: predictions.frozen_odds
      P6: matches.odds_home (API-Football)
      P7: odds_snapshots avg (fallback)
    """
    logger.info("canonical_odds_sweeper: starting")
    t0 = time.time()
    stats = {"inserted": 0, "upgraded": 0, "skipped_downgrade": 0, "errors": 0}

    try:
        async with AsyncSessionLocal() as session:
            # Scope: matches modified recently OR upcoming
            scope_rows = await session.execute(text("""
                SELECT m.id, m.status,
                       m.odds_home, m.odds_draw, m.odds_away,
                       co.match_id AS has_canonical, co.priority AS current_priority
                FROM matches m
                LEFT JOIN match_canonical_odds co ON co.match_id = m.id
                WHERE m.date >= NOW() - INTERVAL '7 days'
                  AND m.date <= NOW() + INTERVAL '3 days'
            """))
            rows = scope_rows.fetchall()
            match_ids = [row.id for row in rows]
            logger.info(f"canonical_odds_sweeper: {len(rows)} matches in scope")

            # Pre-load raw odds for P1/P2 resolution (batch, no N+1)
            raw_odds_map = {}  # match_id -> (odds_h, odds_d, odds_a, source, priority)
            if match_ids:
                raw_rows = await session.execute(text("""
                    SELECT DISTINCT ON (r.match_id)
                        r.match_id, r.odds_home, r.odds_draw, r.odds_away,
                        r.provider, r.bookmaker, r.odds_kind
                    FROM raw_odds_1x2 r
                    WHERE r.match_id = ANY(:ids)
                      AND r.provider IN ('fduk', 'oddsportal')
                    ORDER BY r.match_id,
                      CASE WHEN r.provider = 'fduk' AND r.bookmaker = 'Pinnacle' THEN 0
                           WHEN r.provider = 'fduk' THEN 1
                           WHEN r.provider = 'oddsportal' THEN 2
                           ELSE 3
                      END,
                      CASE r.bookmaker
                          WHEN 'Pinnacle' THEN 1
                          WHEN 'Bet365' THEN 2
                          WHEN 'unknown' THEN 3
                          WHEN 'bet-at-home' THEN 4
                          WHEN 'BetInAsia' THEN 5
                          WHEN 'avg' THEN 6
                          ELSE 7
                      END,
                      CASE r.odds_kind WHEN 'closing' THEN 1 ELSE 2 END
                """), {"ids": match_ids})
                for rr in raw_rows.fetchall():
                    is_pinnacle = (rr.provider == 'fduk' and rr.bookmaker == 'Pinnacle')
                    priority = 1 if is_pinnacle else 2
                    source = f"{rr.provider} ({rr.bookmaker})"
                    raw_odds_map[rr.match_id] = (
                        rr.odds_home, rr.odds_draw, rr.odds_away, source, priority
                    )

            for row in rows:
                match_id = row.id
                is_closing = row.status in ('FT', 'AET', 'PEN', 'AWD')
                current_p = row.current_priority  # None if not in canonical yet

                # Resolve best available odds with priority
                best = None  # (odds_h, odds_d, odds_a, source, priority)

                # P1/P2: from raw_odds_1x2 (pre-loaded)
                if match_id in raw_odds_map:
                    best = raw_odds_map[match_id]

                # P3-P5: require sub-queries (expensive), skip in sweeper — covered by one-shot script
                # Only P1, P2, P6 are cheap enough for a periodic job

                # P6: matches.odds_home (API-Football)
                if not best and row.odds_home and row.odds_draw and row.odds_away:
                    best = (row.odds_home, row.odds_draw, row.odds_away,
                            'matches.odds (API-Football)', 6)

                if not best:
                    continue  # No odds available for this match

                new_priority = best[4]

                # Anti-downgrade guardrail: NEVER overwrite high-quality with low-quality
                if current_p is not None and new_priority >= current_p:
                    stats["skipped_downgrade"] += 1
                    continue

                # Insert or upgrade
                await session.execute(text("""
                    INSERT INTO match_canonical_odds
                        (match_id, odds_home, odds_draw, odds_away, source, priority, is_closing, updated_at)
                    VALUES (:mid, :oh, :od, :oa, :src, :p, :ic, NOW())
                    ON CONFLICT (match_id) DO UPDATE SET
                        odds_home = EXCLUDED.odds_home,
                        odds_draw = EXCLUDED.odds_draw,
                        odds_away = EXCLUDED.odds_away,
                        source = EXCLUDED.source,
                        priority = EXCLUDED.priority,
                        is_closing = EXCLUDED.is_closing,
                        updated_at = NOW()
                    WHERE match_canonical_odds.priority > EXCLUDED.priority
                       OR match_canonical_odds.match_id IS NULL
                """), {
                    "mid": match_id,
                    "oh": float(best[0]), "od": float(best[1]), "oa": float(best[2]),
                    "src": best[3], "p": new_priority, "ic": is_closing,
                })

                if row.has_canonical is None:
                    stats["inserted"] += 1
                else:
                    stats["upgraded"] += 1

            await session.commit()

    except Exception as e:
        logger.error(f"canonical_odds_sweeper error: {e}", exc_info=True)
        stats["errors"] += 1

    elapsed = time.time() - t0
    stats["elapsed_s"] = round(elapsed, 1)
    logger.info(f"canonical_odds_sweeper: done in {elapsed:.1f}s — {stats}")
    return stats


async def score_clv_post_match() -> dict:
    """
    Score CLV for predictions of recently finished matches.

    Runs every 2 hours. For each prediction with asof_timestamp where
    the match is now FT/AET/PEN and no CLV exists yet:
    1. Find canonical bookmaker with odds at asof and close
    2. De-vig both, compute log-odds CLV per outcome
    3. Insert into prediction_clv

    Returns metrics dict.
    """
    import time

    from app.telemetry.metrics import record_job_run

    start_time = time.time()

    try:
        from app.ml.clv import score_clv_batch

        async with AsyncSessionLocal() as session:
            metrics = await score_clv_batch(session, lookback_hours=72, limit=200)

        duration_ms = (time.time() - start_time) * 1000
        logger.info(
            f"CLV scoring: scanned={metrics['scanned']}, "
            f"scored={metrics['scored']}, "
            f"skipped_no_odds={metrics['skipped_no_odds']}, "
            f"errors={metrics['errors']} ({duration_ms:.0f}ms)"
        )
        record_job_run(job="clv_scoring", status="ok", duration_ms=duration_ms)
        return metrics

    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        logger.error(f"CLV scoring failed: {e}")
        sentry_capture_exception(e, job_id="clv_scoring")
        record_job_run(job="clv_scoring", status="error", duration_ms=duration_ms)
        return {"status": "error", "error": str(e)}


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
    from app.models import OddsHistory

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
        "odds_history_saved": 0,
        "bookmakers_captured": 0,
        "consensus_calculated": 0,
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
                record_job_run(job="odds_sync", status="no_matches", duration_ms=duration_ms, metrics=metrics)
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
                        # Fetch ALL bookmaker odds (single API call)
                        all_odds_data = await provider.get_odds_all(external_id)
                        metrics["api_calls"] += 1

                        if not all_odds_data:
                            metrics["api_empty"] += 1
                            record_odds_sync_request("empty", 0)
                            logger.debug(f"No odds available for match {match_id} (external: {external_id})")
                            continue

                        # Pick primary bookmaker by priority for matches.odds_*
                        primary_odds = None
                        for priority_book in provider.PRIORITY_BOOKMAKERS:
                            for odds in all_odds_data:
                                if odds["bookmaker"].lower() == priority_book.lower():
                                    primary_odds = odds
                                    break
                            if primary_odds:
                                break
                        if not primary_odds:
                            primary_odds = all_odds_data[0]

                        # Validate primary before writing to matches
                        p_home = primary_odds.get("odds_home")
                        p_draw = primary_odds.get("odds_draw")
                        p_away = primary_odds.get("odds_away")
                        primary_validation = validate_odds_1x2(
                            odds_home=p_home, odds_draw=p_draw, odds_away=p_away,
                            book=primary_odds.get("bookmaker", "unknown"),
                        )

                        if primary_validation.is_usable:
                            await session.execute(text("""
                                UPDATE matches
                                SET odds_home = :odds_home,
                                    odds_draw = :odds_draw,
                                    odds_away = :odds_away,
                                    odds_recorded_at = NOW()
                                WHERE id = :match_id
                            """), {
                                "match_id": match_id,
                                "odds_home": p_home,
                                "odds_draw": p_draw,
                                "odds_away": p_away,
                            })
                            metrics["updated"] += 1
                            record_odds_sync_request("ok", 0)
                        else:
                            logger.warning(
                                f"Odds sync: Primary odds invalid for match {match_id}: "
                                f"violations={primary_validation.violations}"
                            )

                        # P1: Common snapshot timestamp for all books in this match
                        snapshot_ts = datetime.utcnow()

                        # Batch is_opening check — one query per match
                        existing_sources_result = await session.execute(text(
                            "SELECT DISTINCT source FROM odds_history WHERE match_id = :mid"
                        ), {"mid": match_id})
                        existing_sources = {row[0] for row in existing_sources_result.fetchall()}

                        # Loop ALL bookmakers: validate + insert with ON CONFLICT (P0-2 idempotency)
                        books_saved = 0
                        for book_odds in all_odds_data:
                            bk_name = book_odds.get("bookmaker", "unknown")
                            bk_home = book_odds.get("odds_home")
                            bk_draw = book_odds.get("odds_draw")
                            bk_away = book_odds.get("odds_away")

                            bk_validation = validate_odds_1x2(
                                odds_home=bk_home, odds_draw=bk_draw, odds_away=bk_away,
                                book=bk_name,
                            )
                            if not bk_validation.is_usable:
                                continue

                            is_opening = bk_name not in existing_sources
                            implied_h = 1 / bk_home if bk_home and bk_home > 0 else None
                            implied_d = 1 / bk_draw if bk_draw and bk_draw > 0 else None
                            implied_a = 1 / bk_away if bk_away and bk_away > 0 else None
                            overround = (implied_h + implied_d + implied_a) if (implied_h and implied_d and implied_a) else None

                            await session.execute(text("""
                                INSERT INTO odds_history
                                    (match_id, recorded_at, odds_home, odds_draw, odds_away,
                                     source, is_opening, implied_home, implied_draw, implied_away, overround)
                                VALUES
                                    (:mid, :ts, :oh, :od, :oa, :src, :opening, :ih, :id, :ia, :ov)
                                ON CONFLICT (match_id, recorded_at, source) DO NOTHING
                            """), {
                                "mid": match_id, "ts": snapshot_ts,
                                "oh": bk_home, "od": bk_draw, "oa": bk_away,
                                "src": bk_name, "opening": is_opening,
                                "ih": implied_h, "id": implied_d, "ia": implied_a, "ov": overround,
                            })
                            books_saved += 1
                            if is_opening:
                                existing_sources.add(bk_name)

                        metrics["odds_history_saved"] += books_saved
                        metrics["bookmakers_captured"] = metrics.get("bookmakers_captured", 0) + books_saved

                        # Calculate and persist consensus (same snapshot timestamp)
                        from app.ml.consensus import calculate_consensus
                        consensus = calculate_consensus(all_odds_data)
                        if consensus:
                            is_opening_c = "consensus" not in existing_sources
                            c_h = 1 / consensus["odds_home"] if consensus["odds_home"] else None
                            c_d = 1 / consensus["odds_draw"] if consensus["odds_draw"] else None
                            c_a = 1 / consensus["odds_away"] if consensus["odds_away"] else None
                            c_ov = (c_h + c_d + c_a) if (c_h and c_d and c_a) else None

                            await session.execute(text("""
                                INSERT INTO odds_history
                                    (match_id, recorded_at, odds_home, odds_draw, odds_away,
                                     source, is_opening, implied_home, implied_draw, implied_away, overround)
                                VALUES
                                    (:mid, :ts, :oh, :od, :oa, 'consensus', :opening, :ih, :id, :ia, :ov)
                                ON CONFLICT (match_id, recorded_at, source) DO NOTHING
                            """), {
                                "mid": match_id, "ts": snapshot_ts,
                                "oh": consensus["odds_home"], "od": consensus["odds_draw"], "oa": consensus["odds_away"],
                                "opening": is_opening_c,
                                "ih": c_h, "id": c_d, "ia": c_a, "ov": c_ov,
                            })
                            metrics["odds_history_saved"] += 1
                            metrics["consensus_calculated"] = metrics.get("consensus_calculated", 0) + 1

                        logger.debug(
                            f"Odds sync: match {match_id}: "
                            f"primary={primary_odds.get('bookmaker')}, "
                            f"books={books_saved}, consensus={'yes' if consensus else 'no'}"
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
        record_job_run(job="odds_sync", status="ok", duration_ms=duration_ms, metrics=metrics)

        logger.info(
            f"Odds sync complete: "
            f"scanned={metrics['scanned']}, updated={metrics['updated']}, "
            f"odds_history={metrics['odds_history_saved']} "
            f"(books={metrics['bookmakers_captured']}, consensus={metrics['consensus_calculated']}), "
            f"api_calls={metrics['api_calls']}, empty={metrics['api_empty']}, "
            f"errors_429={metrics['errors_429']}, duration={duration_ms:.0f}ms"
        )

        return {**metrics, "status": "completed"}

    except APIBudgetExceeded as e:
        duration_ms = (time.time() - start_time) * 1000
        logger.warning(f"Odds sync stopped: {e}. Budget status: {get_api_budget_status()}")
        record_odds_sync_run("error", duration_ms)
        record_job_run(job="odds_sync", status="budget_exceeded", duration_ms=duration_ms, metrics=metrics)
        metrics["budget_status"] = get_api_budget_status()
        return {**metrics, "status": "budget_exceeded", "error": str(e)}
    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        logger.error(f"Odds sync failed: {e}")
        sentry_capture_exception(e, job_id="odds_sync", metrics=metrics)
        record_odds_sync_run("error", duration_ms)
        record_job_run(job="odds_sync", status="error", duration_ms=duration_ms, metrics=metrics)
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


async def ops_incidents_purge():
    """
    Daily purge of old resolved incidents from ops_incidents.

    Retention: 30 days after resolved_at.
    Safety cap: abort if > 50,000 rows (prevents accidental mass delete).
    Logs purge count to job_runs metrics.
    """
    logger.info("Starting ops_incidents purge...")

    try:
        async with AsyncSessionLocal() as session:
            # Safety check: count before delete
            count_result = await session.execute(text("""
                SELECT COUNT(*) AS cnt FROM ops_incidents
                WHERE status = 'resolved'
                  AND resolved_at IS NOT NULL
                  AND resolved_at < NOW() - INTERVAL '30 days'
            """))
            to_purge = count_result.scalar() or 0

            if to_purge == 0:
                logger.info("ops_incidents purge: nothing to purge")
                return

            if to_purge > 50000:
                logger.error(f"ops_incidents purge: SAFETY CAP — {to_purge} rows exceed 50,000 limit. Aborting.")
                return

            result = await session.execute(text("""
                DELETE FROM ops_incidents
                WHERE status = 'resolved'
                  AND resolved_at IS NOT NULL
                  AND resolved_at < NOW() - INTERVAL '30 days'
                RETURNING id
            """))
            purged = len(result.fetchall())
            await session.commit()

            logger.info(f"ops_incidents purge: deleted {purged} resolved incidents (>30d)")

    except Exception as e:
        logger.error(f"ops_incidents purge failed: {e}")


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
        from app.dashboard.ops_routes import get_cached_ops_data as _get_cached_ops_data

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
            # Determine status: job_failed=True means global failure, not just partial errors
            if metrics.get("job_failed"):
                status = "error"
            elif metrics.get("errors", 0) == 0:
                status = "ok"
            else:
                status = "partial"
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


async def sota_team_home_city_sync() -> dict:
    """
    Daily sync of team_home_city_profile via fallback cascade.

    Full mode on Sundays, delta otherwise.
    Cascade: venue_city -> venue_name geocoding -> LLM candidate -> manual overrides.

    Frequency: Daily 05:00 UTC (avoids ops_incidents_purge at 04:30)
    Guardrail: TEAM_PROFILE_SYNC_ENABLED env var (default off)
    """
    import time as _time
    from datetime import datetime

    start_time = _time.time()
    started_at = datetime.utcnow()
    job_name = "sota_team_home_city_sync"

    # Check if enabled (default off)
    if os.environ.get("TEAM_PROFILE_SYNC_ENABLED", "false").lower() in ("false", "0", "no"):
        logger.debug(f"[{job_name}] Disabled via env var (set TEAM_PROFILE_SYNC_ENABLED=true)")
        return {"status": "disabled"}

    try:
        from app.etl.team_home_city import run_cascade_batch
        from app.jobs.tracking import record_job_run as record_job_run_db

        # Full on Sundays (weekday 6), delta otherwise
        mode = "full" if datetime.utcnow().weekday() == 6 else "delta"
        llm_enabled = os.environ.get("TEAM_PROFILE_LLM_ENABLED", "false").lower() in ("true", "1")

        async with AsyncSessionLocal() as session:
            metrics = await run_cascade_batch(
                session, mode=mode, limit=200, llm_enabled=llm_enabled,
            )

            duration_ms = (_time.time() - start_time) * 1000
            status = "ok" if metrics.get("errors", 0) == 0 else "partial"
            await record_job_run_db(session, job_name, status, started_at, metrics=metrics)

        record_job_run(job=job_name, status=status, duration_ms=duration_ms)

        logger.info(
            f"[{job_name}] Complete ({mode}): resolved={metrics.get('resolved', 0)}, "
            f"unresolved={metrics.get('unresolved', 0)}, errors={metrics.get('errors', 0)}"
        )
        return {**metrics, "status": status, "duration_ms": duration_ms}

    except Exception as e:
        duration_ms = (_time.time() - start_time) * 1000
        logger.error(f"[{job_name}] Failed: {e}", exc_info=True)
        sentry_capture_exception(e, job_id=job_name)
        record_job_run(job=job_name, status="error", duration_ms=duration_ms)
        try:
            from app.jobs.tracking import record_job_run as record_job_run_db
            async with AsyncSessionLocal() as session:
                await record_job_run_db(session, job_name, "error", started_at, error=str(e))
        except Exception:
            pass
        return {"status": "error", "error": str(e), "duration_ms": duration_ms}


async def sota_wikidata_team_enrich() -> dict:
    """
    Enrich teams with Wikidata data (stadium coords, admin location, social).

    Dual mode:
    - CATCH-UP (daily): If teams lack enrichment, process batch_size=100
    - REFRESH (weekly): If all have enrichment, refresh > 30 days (Sundays only)

    736 teams / 100 batch = ~8 days for complete catch-up.

    Frequency: Daily 04:30 UTC (before team_home_city_sync at 05:00)
    Guardrail: WIKIDATA_ENRICH_ENABLED env var (default off)
    """
    import time as _time
    from datetime import datetime

    start_time = _time.time()
    started_at = datetime.utcnow()
    job_name = "sota_wikidata_team_enrich"

    # Check if enabled (default off)
    if os.environ.get("WIKIDATA_ENRICH_ENABLED", "false").lower() in ("false", "0", "no"):
        logger.debug(f"[{job_name}] Disabled via env var (set WIKIDATA_ENRICH_ENABLED=true)")
        return {"status": "disabled"}

    try:
        from app.etl.wikidata_enrich import get_enrichment_stats, run_wikidata_enrichment_batch
        from app.jobs.tracking import record_job_run as record_job_run_db

        async with AsyncSessionLocal() as session:
            # Check pending count to decide mode
            stats = await get_enrichment_stats(session)
            pending = stats["total_with_wikidata"] - stats["enriched"]

            if pending > 0:
                # CATCH-UP mode: process teams without enrichment
                mode = "catch-up"
                batch_size = int(os.environ.get("WIKIDATA_ENRICH_BATCH_SIZE", "100"))
            elif datetime.utcnow().weekday() == 6:  # Sunday
                # REFRESH mode: refresh > 30 days (only Sundays)
                mode = "refresh"
                batch_size = 50
            else:
                # No work to do
                duration_ms = (_time.time() - start_time) * 1000
                logger.debug(f"[{job_name}] No catch-up pending, not refresh day (Sunday)")
                return {
                    "status": "ok",
                    "mode": "idle",
                    "message": "No catch-up pending, not refresh day",
                    "stats": stats,
                    "duration_ms": duration_ms,
                }

            metrics = await run_wikidata_enrichment_batch(
                session, batch_size=batch_size, mode=mode
            )

            duration_ms = (_time.time() - start_time) * 1000
            status = "ok" if metrics.get("errors", 0) == 0 else "partial"
            await record_job_run_db(session, job_name, status, started_at, metrics=metrics)

        record_job_run(job=job_name, status=status, duration_ms=duration_ms)

        logger.info(
            f"[{job_name}] Complete ({mode}): enriched={metrics.get('enriched', 0)}, "
            f"errors={metrics.get('errors', 0)}, pct_complete={stats.get('pct_complete', 0)}%"
        )
        return {**metrics, "status": status, "duration_ms": duration_ms, "stats": stats}

    except Exception as e:
        duration_ms = (_time.time() - start_time) * 1000
        logger.error(f"[{job_name}] Failed: {e}", exc_info=True)
        sentry_capture_exception(e, job_id=job_name)
        record_job_run(job=job_name, status="error", duration_ms=duration_ms)
        try:
            from app.jobs.tracking import record_job_run as record_job_run_db

            async with AsyncSessionLocal() as session:
                await record_job_run_db(session, job_name, "error", started_at, error=str(e))
        except Exception:
            pass
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
            # Run with 72h lookahead, 7 days back, max 200 matches
            # P1a: days_back raised from 2→7 to catch matches missed by short window/downtime
            stats = await sync_sofascore_refs(
                session,
                hours=72,
                days_back=7,
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


async def sota_sofascore_ratings_backfill() -> dict:
    """
    Backfill Sofascore post-match player ratings for finished matches.

    Re-fetches lineups for FT matches to extract ratings (available post-match).
    Stores in sofascore_player_rating_history for rolling average features.

    Frequency: Every 6 hours
    Guardrail: SOTA_SOFASCORE_RATINGS_ENABLED env var
    """
    import time as _time
    from datetime import datetime

    start_time = _time.time()
    started_at = datetime.utcnow()
    job_name = "sota_sofascore_ratings_backfill"

    if os.environ.get("SOTA_SOFASCORE_RATINGS_ENABLED", "false").lower() in ("false", "0", "no"):
        logger.debug(f"[{job_name}] Disabled via env var (set SOTA_SOFASCORE_RATINGS_ENABLED=true to enable)")
        return {"status": "disabled"}

    metrics = {"started_at": started_at.isoformat()}

    try:
        from app.etl.sota_jobs import backfill_sofascore_ratings_ft
        from app.jobs.tracking import record_job_run as record_job_run_db

        async with AsyncSessionLocal() as session:
            stats = await backfill_sofascore_ratings_ft(session, days=14, limit=100)
            metrics.update(stats)

            duration_ms = (_time.time() - start_time) * 1000
            status = "ok" if metrics.get("errors", 0) == 0 else "partial"
            await record_job_run_db(session, job_name, status, started_at, metrics=metrics)

        record_job_run(job=job_name, status=status, duration_ms=duration_ms)
        logger.info(
            f"[{job_name}] Complete: matches={metrics.get('matches_processed', 0)}, "
            f"players={metrics.get('players_inserted', 0)}"
        )
        return {**metrics, "status": status, "duration_ms": duration_ms}

    except Exception as e:
        duration_ms = (_time.time() - start_time) * 1000
        logger.error(f"[{job_name}] Failed: {e}", exc_info=True)
        sentry_capture_exception(e, job_id=job_name)
        record_job_run(job=job_name, status="error", duration_ms=duration_ms)
        try:
            from app.jobs.tracking import record_job_run as record_job_run_db
            async with AsyncSessionLocal() as session:
                await record_job_run_db(session, job_name, "error", started_at, error=str(e))
        except Exception:
            pass
        return {"status": "error", "error": str(e), "duration_ms": duration_ms}


async def sota_sofascore_stats_backfill() -> dict:
    """
    Backfill Sofascore post-match statistics for finished matches.

    Fetches xG, big chances, possession etc. from Sofascore statistics endpoint.
    Stores in match_sofascore_stats.

    Frequency: Every 6 hours
    Guardrail: SOTA_SOFASCORE_STATS_ENABLED env var
    """
    import time as _time
    from datetime import datetime

    start_time = _time.time()
    started_at = datetime.utcnow()
    job_name = "sota_sofascore_stats_backfill"

    if os.environ.get("SOTA_SOFASCORE_STATS_ENABLED", "false").lower() in ("false", "0", "no"):
        logger.debug(f"[{job_name}] Disabled via env var (set SOTA_SOFASCORE_STATS_ENABLED=true to enable)")
        return {"status": "disabled"}

    metrics = {"started_at": started_at.isoformat()}

    try:
        from app.etl.sota_jobs import backfill_sofascore_stats_ft
        from app.jobs.tracking import record_job_run as record_job_run_db

        async with AsyncSessionLocal() as session:
            stats = await backfill_sofascore_stats_ft(session, days=14, limit=100)
            metrics.update(stats)

            duration_ms = (_time.time() - start_time) * 1000
            status = "ok" if metrics.get("errors", 0) == 0 else "partial"
            await record_job_run_db(session, job_name, status, started_at, metrics=metrics)

        record_job_run(job=job_name, status=status, duration_ms=duration_ms)
        logger.info(
            f"[{job_name}] Complete: inserted={metrics.get('inserted', 0)}"
        )
        return {**metrics, "status": status, "duration_ms": duration_ms}

    except Exception as e:
        duration_ms = (_time.time() - start_time) * 1000
        logger.error(f"[{job_name}] Failed: {e}", exc_info=True)
        sentry_capture_exception(e, job_id=job_name)
        record_job_run(job=job_name, status="error", duration_ms=duration_ms)
        try:
            from app.jobs.tracking import record_job_run as record_job_run_db
            async with AsyncSessionLocal() as session:
                await record_job_run_db(session, job_name, "error", started_at, error=str(e))
        except Exception:
            pass
        return {"status": "error", "error": str(e), "duration_ms": duration_ms}


async def sota_fotmob_refs_sync() -> dict:
    """
    Sync FotMob refs (match_external_refs) for FT matches in eligible leagues.

    Links internal matches to FotMob match IDs for xG capture.
    ABE P0-8: Only confirmed leagues execute.

    Frequency: Every 12 hours
    Guardrail: FOTMOB_REFS_ENABLED setting
    """
    import time as _time
    from datetime import datetime

    start_time = _time.time()
    started_at = datetime.utcnow()
    job_name = "sota_fotmob_refs_sync"

    try:
        from app.config import get_settings
        settings = get_settings()
        if not settings.FOTMOB_REFS_ENABLED:
            logger.debug(f"[{job_name}] Disabled (set FOTMOB_REFS_ENABLED=true to enable)")
            return {"status": "disabled"}
    except Exception:
        return {"status": "disabled"}

    metrics = {"started_at": started_at.isoformat()}

    try:
        from app.etl.sota_jobs import sync_fotmob_refs
        from app.jobs.tracking import record_job_run as record_job_run_db

        async with AsyncSessionLocal() as session:
            stats = await sync_fotmob_refs(session, days=7, limit=200)
            metrics.update(stats)

            duration_ms = (_time.time() - start_time) * 1000
            status = "ok" if metrics.get("errors", 0) == 0 else "partial"
            await record_job_run_db(session, job_name, status, started_at, metrics=metrics)

        record_job_run(job=job_name, status=status, duration_ms=duration_ms)
        logger.info(
            f"[{job_name}] Complete: linked_auto={metrics.get('linked_auto', 0)} "
            f"linked_review={metrics.get('linked_review', 0)}"
        )
        return {**metrics, "status": status, "duration_ms": duration_ms}

    except Exception as e:
        duration_ms = (_time.time() - start_time) * 1000
        logger.error(f"[{job_name}] Failed: {e}", exc_info=True)
        sentry_capture_exception(e, job_id=job_name)
        record_job_run(job=job_name, status="error", duration_ms=duration_ms)
        try:
            from app.jobs.tracking import record_job_run as record_job_run_db
            async with AsyncSessionLocal() as session:
                await record_job_run_db(session, job_name, "error", started_at, error=str(e))
        except Exception:
            pass
        return {"status": "error", "error": str(e), "duration_ms": duration_ms}


async def sota_fotmob_xg_backfill() -> dict:
    """
    Backfill FotMob xG for linked FT matches.

    Fetches team-level xG/xGOT from FotMob matchDetails endpoint.
    ABE P0-4: captured_at = match_date + 6h.
    ABE P0-8: Only confirmed leagues execute.

    Frequency: Every 6 hours
    Guardrail: FOTMOB_XG_ENABLED setting
    """
    import time as _time
    from datetime import datetime

    start_time = _time.time()
    started_at = datetime.utcnow()
    job_name = "sota_fotmob_xg_backfill"

    try:
        from app.config import get_settings
        settings = get_settings()
        if not settings.FOTMOB_XG_ENABLED:
            logger.debug(f"[{job_name}] Disabled (set FOTMOB_XG_ENABLED=true to enable)")
            return {"status": "disabled"}
    except Exception:
        return {"status": "disabled"}

    metrics = {"started_at": started_at.isoformat()}

    try:
        from app.etl.sota_jobs import backfill_fotmob_xg_ft
        from app.jobs.tracking import record_job_run as record_job_run_db

        async with AsyncSessionLocal() as session:
            stats = await backfill_fotmob_xg_ft(session, days=7, limit=100)
            metrics.update(stats)

            duration_ms = (_time.time() - start_time) * 1000
            status = "ok" if metrics.get("errors", 0) == 0 else "partial"
            await record_job_run_db(session, job_name, status, started_at, metrics=metrics)

        record_job_run(job=job_name, status=status, duration_ms=duration_ms)
        logger.info(
            f"[{job_name}] Complete: captured={metrics.get('captured', 0)} "
            f"skipped_no_xg={metrics.get('skipped_no_xg', 0)}"
        )
        return {**metrics, "status": status, "duration_ms": duration_ms}

    except Exception as e:
        duration_ms = (_time.time() - start_time) * 1000
        logger.error(f"[{job_name}] Failed: {e}", exc_info=True)
        sentry_capture_exception(e, job_id=job_name)
        record_job_run(job=job_name, status="error", duration_ms=duration_ms)
        try:
            from app.jobs.tracking import record_job_run as record_job_run_db
            async with AsyncSessionLocal() as session:
                await record_job_run_db(session, job_name, "error", started_at, error=str(e))
        except Exception:
            pass
        return {"status": "error", "error": str(e), "duration_ms": duration_ms}


async def titan_feature_matrix_runner() -> dict:
    """
    TITAN Feature Matrix Runner - materializes features for upcoming matches.

    Runs the TITAN pipeline to extract API-Football data and materialize
    feature_matrix rows with Tier 1/1b/1c/1d/2/3 features.

    Frequency: Every 2 hours
    Guardrail: TITAN_RUNNER_ENABLED env var (default: true)

    Processes:
    - Today's matches (pre-kickoff for betting)
    - Tomorrow's matches (lookahead for early signals)

    PIT safety: Only materializes data with captured_at < kickoff_utc.
    """
    import time as _time
    from datetime import datetime, date, timedelta

    start_time = _time.time()
    started_at = datetime.utcnow()
    job_name = "titan_feature_matrix_runner"

    # Check if enabled (default ON for TITAN)
    if os.environ.get("TITAN_RUNNER_ENABLED", "true").lower() in ("false", "0", "no"):
        logger.debug(f"[{job_name}] Disabled via env var (set TITAN_RUNNER_ENABLED=true to enable)")
        return {"status": "disabled"}

    metrics = {
        "today_processed": 0,
        "tomorrow_processed": 0,
        "total_extracted": 0,
        "total_failed": 0,
        "errors": [],
        "started_at": started_at.isoformat(),
    }

    try:
        from app.titan.runner import run_titan_pipeline
        from app.jobs.tracking import record_job_run as record_job_run_db

        # Process today's matches
        today = date.today()
        try:
            today_result = await run_titan_pipeline(
                target_date=today,
                limit=50,
                dry_run=False,
            )
            metrics["today_processed"] = today_result.get("matches_found", 0)
            metrics["total_extracted"] += today_result.get("extracted_success", 0)
            metrics["total_failed"] += today_result.get("extracted_failed", 0)
            if today_result.get("errors"):
                metrics["errors"].extend(today_result["errors"][:5])  # Max 5 errors
        except Exception as e:
            logger.error(f"[{job_name}] Error processing today: {e}")
            metrics["errors"].append(f"today: {str(e)[:100]}")

        # Process tomorrow's matches (lookahead)
        tomorrow = today + timedelta(days=1)
        try:
            tomorrow_result = await run_titan_pipeline(
                target_date=tomorrow,
                limit=50,
                dry_run=False,
            )
            metrics["tomorrow_processed"] = tomorrow_result.get("matches_found", 0)
            metrics["total_extracted"] += tomorrow_result.get("extracted_success", 0)
            metrics["total_failed"] += tomorrow_result.get("extracted_failed", 0)
            if tomorrow_result.get("errors"):
                metrics["errors"].extend(tomorrow_result["errors"][:5])
        except Exception as e:
            logger.error(f"[{job_name}] Error processing tomorrow: {e}")
            metrics["errors"].append(f"tomorrow: {str(e)[:100]}")

        duration_ms = (_time.time() - start_time) * 1000
        status = "ok" if not metrics["errors"] else "partial"

        # Record in DB for ops dashboard
        async with AsyncSessionLocal() as session:
            await record_job_run_db(session, job_name, status, started_at, metrics=metrics)

        # Record in Prometheus
        record_job_run(job=job_name, status=status, duration_ms=duration_ms)

        logger.info(
            f"[{job_name}] Completed: today={metrics['today_processed']}, "
            f"tomorrow={metrics['tomorrow_processed']}, extracted={metrics['total_extracted']}, "
            f"failed={metrics['total_failed']}, duration={duration_ms:.0f}ms"
        )

        return {"status": status, "metrics": metrics, "duration_ms": duration_ms}

    except Exception as e:
        duration_ms = (_time.time() - start_time) * 1000
        logger.error(f"[{job_name}] Fatal error: {e}")
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


async def titan_outcome_sync() -> dict:
    """
    TITAN Outcome Sync - updates outcome column for finished matches.

    Syncs outcome ('home', 'draw', 'away') from public.matches to titan.feature_matrix
    for matches that have finished (FT/AET/PEN) but don't have outcome set yet.

    This enables the TITAN gate metrics (with_outcome count) to progress.

    Runs every 30 minutes.
    """
    import time as _time
    from sqlalchemy import text

    job_name = "titan_outcome_sync"
    start_time = _time.time()

    try:
        async with get_session_with_retry(max_retries=3, retry_delay=1.0) as session:
            # Batch update: join titan.feature_matrix with public.matches
            # and set outcome based on final score
            result = await session.execute(text("""
                UPDATE titan.feature_matrix fm
                SET
                    outcome = CASE
                        WHEN m.home_goals > m.away_goals THEN 'home'
                        WHEN m.home_goals = m.away_goals THEN 'draw'
                        WHEN m.home_goals < m.away_goals THEN 'away'
                    END,
                    updated_at = NOW()
                FROM matches m
                WHERE m.external_id = fm.match_id
                  AND fm.outcome IS NULL
                  AND m.status IN ('FT', 'AET', 'PEN')
                  AND m.home_goals IS NOT NULL
                  AND m.away_goals IS NOT NULL
            """))

            updated_count = result.rowcount
            await session.commit()

            duration_ms = (_time.time() - start_time) * 1000

            if updated_count > 0:
                logger.info(f"[{job_name}] Updated {updated_count} outcomes in {duration_ms:.0f}ms")
            else:
                logger.debug(f"[{job_name}] No pending outcomes to sync")

            record_job_run(job=job_name, status="ok", duration_ms=duration_ms)

            return {
                "status": "ok",
                "updated": updated_count,
                "duration_ms": duration_ms,
            }

    except Exception as e:
        duration_ms = (_time.time() - start_time) * 1000
        logger.error(f"[{job_name}] Failed: {e}")
        record_job_run(job=job_name, status="error", duration_ms=duration_ms)
        return {"status": "error", "error": str(e), "duration_ms": duration_ms}


async def titan_lineup_fixup() -> dict:
    """
    TITAN Lineup Fixup - fills missing Tier 1c/1d from SOTA data.

    Finds matches where sofascore_lineup_available=false in feature_matrix
    but match_sofascore_lineup has both sides captured pre-kickoff.
    Fill-only: never degrades existing True to False (ABE directive).

    Frequency: Every 1 hour
    Guardrail: TITAN_LINEUP_FIXUP_ENABLED env var (default: true)
    """
    import time as _time

    job_name = "titan_lineup_fixup"
    start_time = _time.time()

    if os.environ.get("TITAN_LINEUP_FIXUP_ENABLED", "true").lower() in ("false", "0", "no"):
        logger.debug(f"[{job_name}] Disabled via env var")
        return {"status": "disabled"}

    try:
        async with get_session_with_retry(max_retries=3, retry_delay=1.0) as session:
            from app.titan.materializers.feature_matrix import FeatureMatrixMaterializer

            materializer = FeatureMatrixMaterializer(session=session)
            stats = await materializer.fixup_missing_lineup_from_sota(limit=100)

            duration_ms = (_time.time() - start_time) * 1000

            if stats["fixed"] > 0:
                logger.info(
                    f"[{job_name}] Fixed {stats['fixed']} lineups in {duration_ms:.0f}ms "
                    f"({stats['skipped']} skipped, {stats['errors']} errors)"
                )
            else:
                logger.debug(f"[{job_name}] No fixable matches found ({duration_ms:.0f}ms)")

            record_job_run(job=job_name, status="ok", duration_ms=duration_ms)

            return {
                "status": "ok",
                **stats,
                "duration_ms": duration_ms,
            }

    except Exception as e:
        duration_ms = (_time.time() - start_time) * 1000
        logger.error(f"[{job_name}] Failed: {e}")
        record_job_run(job=job_name, status="error", duration_ms=duration_ms)
        return {"status": "error", "error": str(e), "duration_ms": duration_ms}


async def titan_xg_fixup() -> dict:
    """
    TITAN xG Fixup - fills missing Tier 1b from SOTA xG data.

    Finds matches where tier1b_complete=false in feature_matrix but SOTA has
    sufficient xG data (Understat or FotMob) for both teams (>=5 past matches).
    Fill-only: never degrades existing True to False (ABE directive).
    xg_captured_at set to pre-kickoff timestamp (ABE P0).

    Frequency: Every 1 hour
    Guardrail: TITAN_XG_FIXUP_ENABLED env var (default: true)
    """
    import time as _time

    job_name = "titan_xg_fixup"
    start_time = _time.time()

    if os.environ.get("TITAN_XG_FIXUP_ENABLED", "true").lower() in ("false", "0", "no"):
        logger.debug(f"[{job_name}] Disabled via env var")
        return {"status": "disabled"}

    try:
        async with get_session_with_retry(max_retries=3, retry_delay=1.0) as session:
            from app.titan.materializers.feature_matrix import FeatureMatrixMaterializer

            materializer = FeatureMatrixMaterializer(session=session)
            stats = await materializer.fixup_missing_xg_from_sota(limit=100)

            duration_ms = (_time.time() - start_time) * 1000

            if stats["fixed"] > 0:
                logger.info(
                    f"[{job_name}] Fixed {stats['fixed']} xG entries in {duration_ms:.0f}ms "
                    f"({stats['skipped']} skipped, {stats['errors']} errors)"
                )
            else:
                logger.debug(f"[{job_name}] No fixable matches found ({duration_ms:.0f}ms)")

            record_job_run(job=job_name, status="ok", duration_ms=duration_ms)

            return {
                "status": "ok",
                **stats,
                "duration_ms": duration_ms,
            }

    except Exception as e:
        duration_ms = (_time.time() - start_time) * 1000
        logger.error(f"[{job_name}] Failed: {e}")
        record_job_run(job=job_name, status="error", duration_ms=duration_ms)
        return {"status": "error", "error": str(e), "duration_ms": duration_ms}


async def logo_resize_pending() -> dict:
    """
    Logo Resize Job - processes pending_resize logos into thumbnails.

    Finds team_logos with status='pending_resize' and generates
    WebP thumbnails (64, 128, 256, 512) for each variant.

    Runs every 5 minutes. Processes up to 10 teams per run.

    Requires LOGOS_R2_ENABLED=true to be active.
    """
    import time as _time
    from sqlalchemy import select, update

    job_name = "logo_resize_pending"
    start_time = _time.time()

    # Check if logos system is enabled
    from app.logos.config import get_logos_settings
    logos_settings = get_logos_settings()

    if not logos_settings.LOGOS_R2_ENABLED:
        logger.debug(f"[{job_name}] Skipped - LOGOS_R2_ENABLED=false")
        return {"status": "skipped", "reason": "disabled"}

    try:
        from app.models import TeamLogo
        from app.logos.batch_worker import process_team_thumbnails

        async with get_session_with_retry(max_retries=3, retry_delay=1.0) as session:
            # Find pending_resize logos (limit to batch size)
            result = await session.execute(
                select(TeamLogo.team_id)
                .where(TeamLogo.status == "pending_resize")
                .limit(10)
            )
            pending_ids = [row[0] for row in result.fetchall()]

            if not pending_ids:
                logger.debug(f"[{job_name}] No logos pending resize")
                return {"status": "ok", "processed": 0}

            processed = 0
            errors = 0

            for team_id in pending_ids:
                try:
                    success, error = await process_team_thumbnails(session, team_id)
                    if success:
                        processed += 1
                    else:
                        errors += 1
                        logger.warning(f"[{job_name}] Team {team_id} resize failed: {error}")
                except Exception as e:
                    errors += 1
                    logger.error(f"[{job_name}] Team {team_id} exception: {e}")

            duration_ms = (_time.time() - start_time) * 1000

            logger.info(
                f"[{job_name}] Processed {processed}/{len(pending_ids)} logos, "
                f"{errors} errors in {duration_ms:.0f}ms"
            )

            record_job_run(job=job_name, status="ok", duration_ms=duration_ms)

            return {
                "status": "ok",
                "processed": processed,
                "errors": errors,
                "duration_ms": duration_ms,
            }

    except Exception as e:
        duration_ms = (_time.time() - start_time) * 1000
        logger.error(f"[{job_name}] Failed: {e}")
        record_job_run(job=job_name, status="error", duration_ms=duration_ms)
        return {"status": "error", "error": str(e), "duration_ms": duration_ms}


async def player_injuries_sync() -> dict:
    """
    Sync player injuries from API-Football for all tracked leagues.

    Fetches injuries per league/season, resolves internal team_id and match_id,
    upserts into player_injuries table.

    Frequency: Every 6 hours (staggered: 06:00, 12:00, 18:00, 00:00 UTC)
    Guardrail: INJURIES_SYNC_ENABLED env var
    """
    import time as _time

    start_time = _time.time()
    started_at = datetime.utcnow()
    job_name = "player_injuries_sync"

    if os.environ.get("INJURIES_SYNC_ENABLED", "false").lower() in ("false", "0", "no"):
        logger.debug(f"[{job_name}] Disabled via env var")
        return {"status": "disabled"}

    metrics = {"started_at": started_at.isoformat()}

    try:
        from app.etl.player_jobs import sync_injuries
        from app.jobs.tracking import record_job_run as record_job_run_db

        async with AsyncSessionLocal() as session:
            stats = await sync_injuries(session)
            metrics.update(stats)

            duration_ms = (_time.time() - start_time) * 1000
            status = "ok" if metrics.get("errors", 0) == 0 else "partial"
            await record_job_run_db(session, job_name, status, started_at, metrics=metrics)

        record_job_run(job=job_name, status=status, duration_ms=duration_ms)
        logger.info(
            f"[{job_name}] Complete: leagues={metrics.get('leagues_ok', 0)}, "
            f"inserted={metrics.get('injuries_inserted', 0)}, "
            f"updated={metrics.get('injuries_updated', 0)}"
        )
        return {**metrics, "status": status, "duration_ms": duration_ms}

    except Exception as e:
        duration_ms = (_time.time() - start_time) * 1000
        logger.error(f"[{job_name}] Failed: {e}", exc_info=True)
        sentry_capture_exception(e, job_id=job_name)
        record_job_run(job=job_name, status="error", duration_ms=duration_ms)
        try:
            from app.jobs.tracking import record_job_run as record_job_run_db
            async with AsyncSessionLocal() as session:
                await record_job_run_db(session, job_name, "error", started_at, error=str(e))
        except Exception:
            pass
        return {"status": "error", "error": str(e), "duration_ms": duration_ms}


async def player_manager_sync() -> dict:
    """
    Sync managers from API-Football, detect coaching changes.

    Fetches current coach per team, upserts catalog, detects changes
    and records stints in team_manager_history.

    Frequency: Daily at 02:00 UTC
    Guardrail: MANAGER_SYNC_ENABLED env var
    """
    import time as _time

    start_time = _time.time()
    started_at = datetime.utcnow()
    job_name = "player_manager_sync"

    if os.environ.get("MANAGER_SYNC_ENABLED", "false").lower() in ("false", "0", "no"):
        logger.debug(f"[{job_name}] Disabled via env var")
        return {"status": "disabled"}

    metrics = {"started_at": started_at.isoformat()}

    try:
        from app.etl.player_jobs import sync_managers
        from app.jobs.tracking import record_job_run as record_job_run_db

        async with AsyncSessionLocal() as session:
            stats = await sync_managers(session)
            metrics.update(stats)

            duration_ms = (_time.time() - start_time) * 1000
            status = "ok" if metrics.get("errors", 0) == 0 else "partial"
            await record_job_run_db(session, job_name, status, started_at, metrics=metrics)

        record_job_run(job=job_name, status=status, duration_ms=duration_ms)
        logger.info(
            f"[{job_name}] Complete: teams={metrics.get('teams_ok', 0)}, "
            f"changes={metrics.get('changes_detected', 0)}"
        )
        return {**metrics, "status": status, "duration_ms": duration_ms}

    except Exception as e:
        duration_ms = (_time.time() - start_time) * 1000
        logger.error(f"[{job_name}] Failed: {e}", exc_info=True)
        sentry_capture_exception(e, job_id=job_name)
        record_job_run(job=job_name, status="error", duration_ms=duration_ms)
        try:
            from app.jobs.tracking import record_job_run as record_job_run_db
            async with AsyncSessionLocal() as session:
                await record_job_run_db(session, job_name, "error", started_at, error=str(e))
        except Exception:
            pass
        return {"status": "error", "error": str(e), "duration_ms": duration_ms}


async def player_squad_sync() -> dict:
    """
    Sync player squads from API-Football for all active teams.

    Fetches squad roster per team, upserts into players table.

    Frequency: Weekly on Monday at 03:00 UTC
    Guardrail: SQUAD_SYNC_ENABLED env var
    """
    import time as _time

    start_time = _time.time()
    started_at = datetime.utcnow()
    job_name = "player_squad_sync"

    if os.environ.get("SQUAD_SYNC_ENABLED", "false").lower() in ("false", "0", "no"):
        logger.debug(f"[{job_name}] Disabled via env var")
        return {"status": "disabled"}

    metrics = {"started_at": started_at.isoformat()}

    try:
        from app.etl.player_jobs import sync_squads
        from app.jobs.tracking import record_job_run as record_job_run_db

        async with AsyncSessionLocal() as session:
            stats = await sync_squads(session)
            metrics.update(stats)

            duration_ms = (_time.time() - start_time) * 1000
            status = "ok" if metrics.get("errors", 0) == 0 else "partial"
            await record_job_run_db(session, job_name, status, started_at, metrics=metrics)

        record_job_run(job=job_name, status=status, duration_ms=duration_ms)
        logger.info(
            f"[{job_name}] Complete: teams={metrics.get('teams_ok', 0)}, "
            f"players={metrics.get('players_upserted', 0)}"
        )
        return {**metrics, "status": status, "duration_ms": duration_ms}

    except Exception as e:
        duration_ms = (_time.time() - start_time) * 1000
        logger.error(f"[{job_name}] Failed: {e}", exc_info=True)
        sentry_capture_exception(e, job_id=job_name)
        record_job_run(job=job_name, status="error", duration_ms=duration_ms)
        try:
            from app.jobs.tracking import record_job_run as record_job_run_db
            async with AsyncSessionLocal() as session:
                await record_job_run_db(session, job_name, "error", started_at, error=str(e))
        except Exception:
            pass
        return {"status": "error", "error": str(e), "duration_ms": duration_ms}


async def lineup_sync() -> dict:
    """
    Sync lineups for recently finished matches missing lineup data.

    Finds FT matches in last 48h without match_lineups rows and fetches them.

    Frequency: Every 6 hours (offset from injuries: hour 3,9,15,21)
    Guardrail: LINEUP_SYNC_ENABLED env var
    """
    import time as _time

    start_time = _time.time()
    started_at = datetime.utcnow()
    job_name = "lineup_sync"

    if os.environ.get("LINEUP_SYNC_ENABLED", "false").lower() in ("false", "0", "no"):
        logger.debug(f"[{job_name}] Disabled via env var")
        return {"status": "disabled"}

    metrics = {"started_at": started_at.isoformat()}

    try:
        from app.etl.player_jobs import sync_match_lineups
        from app.jobs.tracking import record_job_run as record_job_run_db

        async with AsyncSessionLocal() as session:
            stats = await sync_match_lineups(session)
            metrics.update(stats)

            duration_ms = (_time.time() - start_time) * 1000
            status = "ok" if metrics.get("errors", 0) == 0 else "partial"
            await record_job_run_db(session, job_name, status, started_at, metrics=metrics)

        record_job_run(job=job_name, status=status, duration_ms=duration_ms)
        logger.info(
            f"[{job_name}] Complete: checked={metrics.get('checked', 0)}, "
            f"inserted={metrics.get('inserted', 0)}"
        )
        return {**metrics, "status": status, "duration_ms": duration_ms}

    except Exception as e:
        duration_ms = (_time.time() - start_time) * 1000
        logger.error(f"[{job_name}] Failed: {e}", exc_info=True)
        sentry_capture_exception(e, job_id=job_name)
        record_job_run(job=job_name, status="error", duration_ms=duration_ms)
        try:
            from app.jobs.tracking import record_job_run as record_job_run_db
            async with AsyncSessionLocal() as session:
                await record_job_run_db(session, job_name, "error", started_at, error=str(e))
        except Exception:
            pass
        return {"status": "error", "error": str(e), "duration_ms": duration_ms}


async def _run_event_bus_sweeper():
    """Wrapper for Event Bus Sweeper Queue (Phase 2, P2-09)."""
    try:
        from app.events import get_event_bus, run_sweeper
        result = await run_sweeper(get_event_bus(), AsyncSessionLocal)
        if result > 0:
            logger.info(f"[SWEEPER] Emitted {result} LINEUP_CONFIRMED events")
    except Exception as e:
        logger.error(f"[SWEEPER] Job failed: {e}", exc_info=True)


async def _refresh_serving_configs_job():
    """[P0-F] Periodic refresh of league_serving_config cache for multi-worker convergence."""
    try:
        from app.ml.league_router import load_serving_configs
        async with AsyncSessionLocal() as session:
            n = await load_serving_configs(session)
            logger.debug("[SERVING-CONFIG] Refreshed %d configs", n)
    except Exception as e:
        logger.warning("[SERVING-CONFIG] Refresh failed: %s", e)


async def _run_auto_lab_job():
    """[P2] Auto-Lab Online: evaluate feature sets for one league per run."""
    try:
        from app.ml.auto_lab import auto_lab_scheduler_job
        await auto_lab_scheduler_job()
    except Exception as e:
        logger.error("[AUTO_LAB] Job wrapper failed: %s", e)


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
        next_run_time=datetime.utcnow() + timedelta(seconds=55),  # Offset: +55s
        max_instances=1,
        coalesce=True,
        misfire_grace_time=6 * 3600,  # 6h grace for 6h interval
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
        max_instances=1,
        coalesce=True,
    )

    # Daily audit job: Every day at 8:00 AM UTC (after results are synced)
    scheduler.add_job(
        daily_audit,
        trigger=CronTrigger(hour=8, minute=0),
        id="daily_audit",
        name="Daily Post-Match Audit",
        replace_existing=True,
    )

    # Serving config refresh: Every 5 minutes [P0-F multi-worker convergence]
    # Ensures all Gunicorn workers pick up DB changes within 5 minutes
    scheduler.add_job(
        _refresh_serving_configs_job,
        trigger=IntervalTrigger(minutes=5),
        id="serving_config_refresh",
        name="Serving Config Refresh (every 5min)",
        replace_existing=True,
        max_instances=1,
        coalesce=True,
    )

    # Auto-Lab Online: CronTrigger 4:00 UTC daily (before sync/predictions)
    # Advisory only — evaluates feature sets per league, persists to DB
    # Kill-switch: AUTO_LAB_ENABLED (default False)
    # [P0-J] pg_try_advisory_lock(777001) + max_instances=1
    scheduler.add_job(
        _run_auto_lab_job,
        trigger=CronTrigger(hour=4, minute=0),
        id="auto_lab_online",
        name="Auto-Lab Online (daily 4:00 UTC)",
        replace_existing=True,
        max_instances=1,
        coalesce=True,
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
            next_run_time=datetime.utcnow() + timedelta(seconds=65),  # Offset: +65s
            max_instances=1,
            coalesce=True,
            misfire_grace_time=sensor_settings.SENSOR_RETRAIN_INTERVAL_HOURS * 3600,  # Grace = interval
        )

        # Sensor B evaluation: Every 30 minutes (same as shadow)
        scheduler.add_job(
            evaluate_sensor_predictions_job,
            trigger=IntervalTrigger(minutes=30),
            id="evaluate_sensor_predictions",
            name="Sensor B Evaluation (every 30min)",
            replace_existing=True,
        )

    # ext-A/B/C/D Shadow job: Generate experimental predictions for all enabled variants
    # ATI: Un solo job genérico procesa A/B/C/D en paralelo
    # ATI guardrails: default OFF per variant, max_instances=1, coalesce=True
    ext_shadow_enabled = (
        sensor_settings.EXTA_SHADOW_ENABLED or
        sensor_settings.EXTB_SHADOW_ENABLED or
        sensor_settings.EXTC_SHADOW_ENABLED or
        sensor_settings.EXTD_SHADOW_ENABLED
    )
    if ext_shadow_enabled:
        enabled_variants = []
        if sensor_settings.EXTA_SHADOW_ENABLED:
            enabled_variants.append("A")
        if sensor_settings.EXTB_SHADOW_ENABLED:
            enabled_variants.append("B")
        if sensor_settings.EXTC_SHADOW_ENABLED:
            enabled_variants.append("C")
        if sensor_settings.EXTD_SHADOW_ENABLED:
            enabled_variants.append("D")

        scheduler.add_job(
            generate_ext_shadow_predictions,
            trigger=IntervalTrigger(minutes=sensor_settings.EXT_SHADOW_INTERVAL_MINUTES),
            id="ext_shadow",
            name=f"ext-{'/'.join(enabled_variants)} Shadow (every {sensor_settings.EXT_SHADOW_INTERVAL_MINUTES}min)",
            replace_existing=True,
            max_instances=1,  # ATI: evitar solapes si una corrida tarda
            coalesce=True,    # ATI: fusionar runs acumulados
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

    # Shadow (Two-Stage) recalibration: Tuesday at 5:00 AM UTC (bi-weekly)
    # Only registered if shadow mode is enabled. Internal interval check ensures
    # actual retrain only runs every 14 days (or on volume trigger).
    from app.config import get_settings as _get_shadow_settings
    _shadow_cfg = _get_shadow_settings()
    if _shadow_cfg.MODEL_SHADOW_ARCHITECTURE == "two_stage":
        scheduler.add_job(
            shadow_recalibration,
            trigger=CronTrigger(day_of_week="tue", hour=5, minute=0),
            id="shadow_recalibration",
            name="Shadow Two-Stage Recalibration",
            replace_existing=True,
            max_instances=1,
            coalesce=True,
            misfire_grace_time=300,
        )
        logger.info("Registered shadow_recalibration job (bi-weekly Tuesdays 5AM UTC)")

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
        max_instances=1,
        coalesce=True,
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
        max_instances=1,
        coalesce=True,
        misfire_grace_time=120,
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
        max_instances=1,
        coalesce=True,
        misfire_grace_time=120,
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
        max_instances=1,
        coalesce=True,
        misfire_grace_time=120,
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
        max_instances=1,
        coalesce=True,
        misfire_grace_time=120,
    )

    # Job 5: EVENT BUS SWEEPER (Phase 2, P2-09)
    # Reconciliation every 2min: finds matches with confirmed lineups but no
    # post-lineup prediction and emits LINEUP_CONFIRMED for cascade re-prediction.
    # ATI Directive: FOR UPDATE SKIP LOCKED mandatory for dedupe.
    scheduler.add_job(
        _run_event_bus_sweeper,
        trigger=IntervalTrigger(minutes=2),
        id="event_bus_sweeper",
        name="Event Bus Sweeper (every 2min)",
        replace_existing=True,
        max_instances=1,
        coalesce=True,
        misfire_grace_time=120,
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

    # Finished Match Player Stats Sync: Every 60 minutes (going-forward)
    # Fetches per-player stats + rating from API-Football /fixtures/players
    # Guardrails: PLAYER_STATS_SYNC_ENABLED (default OFF), delay hours, max calls/run
    scheduler.add_job(
        capture_finished_match_player_stats,
        trigger=IntervalTrigger(minutes=60),
        id="finished_match_player_stats_sync",
        name="Finished Match Player Stats Sync (every 60 min)",
        replace_existing=True,
        next_run_time=datetime.utcnow() + timedelta(seconds=135),  # Offset from stats_backfill
        max_instances=1,
        coalesce=True,
        misfire_grace_time=3600,  # 1h grace for 60min interval
    )

    # Historical Stats Backfill: Every 60 minutes
    # Backfills stats for matches since 2023-08-01 that have NULL stats.
    # Addresses gap from stats_backfill being added late (2026-01-09) with 72h lookback.
    # Guardrails: HISTORICAL_STATS_BACKFILL_ENABLED, batch 500/run (~12,000/day)
    # Auto-disables when complete (returns early if no matches left)
    scheduler.add_job(
        historical_stats_backfill,
        trigger=IntervalTrigger(minutes=60),
        id="historical_stats_backfill",
        name="Historical Stats Backfill (every 60 min)",
        replace_existing=True,
        next_run_time=datetime.utcnow() + timedelta(seconds=120),  # Offset: +120s (after regular backfill)
        max_instances=1,
        coalesce=True,
        misfire_grace_time=3600,  # 1h grace
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
        next_run_time=datetime.utcnow() + timedelta(seconds=75),  # Offset: +75s
        max_instances=1,
        coalesce=True,
        misfire_grace_time=2 * 3600,  # 2h grace for 2h interval
    )

    # CLV Scoring: Every 2 hours (Phase 2: P2-04)
    # Scores CLV for predictions of recently finished matches
    # Uses canonical bookmaker from odds_history, log-odds CLV per outcome
    scheduler.add_job(
        score_clv_post_match,
        trigger=IntervalTrigger(hours=2),
        id="clv_scoring_post_match",
        name="CLV Scoring Post-Match (every 2h)",
        replace_existing=True,
        next_run_time=datetime.utcnow() + timedelta(seconds=95),
        max_instances=1,
        coalesce=True,
        misfire_grace_time=2 * 3600,
    )

    # Canonical Odds Sweeper: Every 6 hours
    # Materializes canonical odds for matches in [-7d, +3d] window.
    # Anti-downgrade guardrail: P1/P2/P3 data is IMMUTABLE from lower sources.
    # 0 API calls — reads only from existing DB tables.
    scheduler.add_job(
        canonical_odds_sweeper,
        trigger=IntervalTrigger(hours=6),
        id="canonical_odds_sweeper",
        name="Canonical Odds Sweeper (every 6h)",
        replace_existing=True,
        next_run_time=datetime.utcnow() + timedelta(seconds=105),  # Offset: +105s
        max_instances=1,
        coalesce=True,
        misfire_grace_time=6 * 3600,
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
            next_run_time=datetime.utcnow() + timedelta(seconds=85),  # Offset: +85s
            max_instances=1,
            coalesce=True,
            misfire_grace_time=_odds_settings.ODDS_SYNC_INTERVAL_HOURS * 3600,  # Grace = interval
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

    # Daily Ops Incidents Purge: 04:30 UTC
    # Deletes resolved incidents older than 30 days
    # Safety cap: aborts if > 50,000 rows to prevent mass deletion
    scheduler.add_job(
        ops_incidents_purge,
        trigger=CronTrigger(hour=4, minute=30),
        id="ops_incidents_purge",
        name="Daily Ops Incidents Purge (30d TTL)",
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
    #
    # HARDENING: All SOTA jobs use:
    #   - max_instances=1: Prevent concurrent execution
    #   - coalesce=True: Collapse missed runs into single execution
    #   - misfire_grace_time: Scaled to job interval (1h for hourly, 6h for 6h jobs, etc)
    #   - Staggered offsets: Prevent startup stampede
    # =========================================================================

    # SOTA: Understat refs sync - every 12 hours
    # Links matches to Understat IDs for xG retrieval
    scheduler.add_job(
        sota_understat_refs_sync,
        trigger=IntervalTrigger(hours=12),
        id="sota_understat_refs_sync",
        name="SOTA Understat Refs Sync (every 12h)",
        replace_existing=True,
        next_run_time=datetime.utcnow() + timedelta(seconds=5),  # Offset: +5s
        max_instances=1,
        coalesce=True,
        misfire_grace_time=12 * 3600,  # 12h grace for 12h interval
    )

    # SOTA: Understat xG backfill - every 6 hours
    # Fetches actual xG data for matches with refs
    scheduler.add_job(
        sota_understat_ft_backfill,
        trigger=IntervalTrigger(hours=6),
        id="sota_understat_ft_backfill",
        name="SOTA Understat xG Backfill (every 6h)",
        replace_existing=True,
        next_run_time=datetime.utcnow() + timedelta(seconds=15),  # Offset: +15s
        max_instances=1,
        coalesce=True,
        misfire_grace_time=6 * 3600,  # 6h grace for 6h interval
    )

    # SOTA: Weather capture - every 60 minutes
    # Captures weather forecasts for upcoming matches (disabled by default)
    scheduler.add_job(
        sota_weather_capture_prekickoff,
        trigger=IntervalTrigger(minutes=60),
        id="sota_weather_capture",
        name="SOTA Weather Capture (every 60 min)",
        replace_existing=True,
        next_run_time=datetime.utcnow() + timedelta(seconds=25),  # Offset: +25s
        max_instances=1,
        coalesce=True,
        misfire_grace_time=3600,  # 1h grace for 1h interval
    )

    # SOTA: Venue geo expand - daily at 03:00 UTC
    # Geocodes new venues (disabled by default)
    scheduler.add_job(
        sota_venue_geo_expand,
        trigger=CronTrigger(hour=3, minute=0),
        id="sota_venue_geo_expand",
        name="SOTA Venue Geo Expand (daily 03:00 UTC)",
        replace_existing=True,
        max_instances=1,
        coalesce=True,
        misfire_grace_time=24 * 3600,  # 24h grace for daily job
    )

    # SOTA: Wikidata Team Enrich - daily at 04:30 UTC
    # Enriches teams with Wikidata data (stadium coords, admin location, social)
    # Guardrail: WIKIDATA_ENRICH_ENABLED (default off)
    scheduler.add_job(
        sota_wikidata_team_enrich,
        trigger=CronTrigger(hour=4, minute=30),
        id="sota_wikidata_team_enrich",
        name="SOTA Wikidata Team Enrich (daily 04:30 UTC)",
        replace_existing=True,
        max_instances=1,
        coalesce=True,
        misfire_grace_time=24 * 3600,  # 24h grace for daily job
    )

    # SOTA: Team Home City Sync - daily at 05:00 UTC
    # Cascade pipeline: venue_city -> venue_name geocoding -> wikidata -> LLM -> overrides
    # Guardrail: TEAM_PROFILE_SYNC_ENABLED (default off)
    scheduler.add_job(
        sota_team_home_city_sync,
        trigger=CronTrigger(hour=5, minute=0),
        id="sota_team_home_city_sync",
        name="SOTA Team Home City Sync (daily 05:00 UTC)",
        replace_existing=True,
        max_instances=1,
        coalesce=True,
        misfire_grace_time=24 * 3600,  # 24h grace for daily job
    )

    # SOTA: Sofascore refs sync - every 6 hours
    # Links matches to Sofascore event IDs (disabled by default)
    scheduler.add_job(
        sota_sofascore_refs_sync,
        trigger=IntervalTrigger(hours=6),
        id="sota_sofascore_refs_sync",
        name="SOTA Sofascore Refs Sync (every 6h)",
        replace_existing=True,
        next_run_time=datetime.utcnow() + timedelta(seconds=35),  # Offset: +35s
        max_instances=1,
        coalesce=True,
        misfire_grace_time=6 * 3600,  # 6h grace for 6h interval
    )

    # SOTA: Sofascore XI capture - every 30 minutes
    # Captures lineup/formation/ratings for upcoming matches (disabled by default)
    scheduler.add_job(
        sota_sofascore_xi_capture,
        trigger=IntervalTrigger(minutes=30),
        id="sota_sofascore_xi_capture",
        name="SOTA Sofascore XI Capture (every 30 min)",
        replace_existing=True,
        next_run_time=datetime.utcnow() + timedelta(seconds=45),  # Offset: +45s
        max_instances=1,
        coalesce=True,
        misfire_grace_time=1800,  # 30min grace for 30min interval
    )

    # SOTA: Sofascore post-match ratings backfill - every 6 hours
    # Re-fetches lineups for FT matches to get player ratings (disabled by default)
    scheduler.add_job(
        sota_sofascore_ratings_backfill,
        trigger=IntervalTrigger(hours=6),
        id="sota_sofascore_ratings_backfill",
        name="SOTA Sofascore Ratings Backfill (every 6h)",
        replace_existing=True,
        next_run_time=datetime.utcnow() + timedelta(seconds=55),  # Offset: +55s
        max_instances=1,
        coalesce=True,
        misfire_grace_time=6 * 3600,
    )

    # SOTA: Sofascore post-match stats backfill - every 6 hours
    # Fetches xG, big chances etc. from Sofascore (disabled by default)
    scheduler.add_job(
        sota_sofascore_stats_backfill,
        trigger=IntervalTrigger(hours=6),
        id="sota_sofascore_stats_backfill",
        name="SOTA Sofascore Stats Backfill (every 6h)",
        replace_existing=True,
        next_run_time=datetime.utcnow() + timedelta(seconds=65),  # Offset: +65s
        max_instances=1,
        coalesce=True,
        misfire_grace_time=6 * 3600,
    )

    # SOTA: FotMob refs sync - every 12h (ABE P0-1: scheduler-only, P0-8: confirmed leagues)
    scheduler.add_job(
        sota_fotmob_refs_sync,
        trigger=IntervalTrigger(hours=12),
        id="sota_fotmob_refs_sync",
        name="SOTA FotMob Refs Sync (every 12h)",
        replace_existing=True,
        next_run_time=datetime.utcnow() + timedelta(seconds=75),  # Offset: +75s
        max_instances=1,
        coalesce=True,
        misfire_grace_time=12 * 3600,
    )

    # SOTA: FotMob xG backfill - every 6h (requires refs to exist first)
    scheduler.add_job(
        sota_fotmob_xg_backfill,
        trigger=IntervalTrigger(hours=6),
        id="sota_fotmob_xg_backfill",
        name="SOTA FotMob xG Backfill (every 6h)",
        replace_existing=True,
        next_run_time=datetime.utcnow() + timedelta(seconds=85),  # Offset: +85s
        max_instances=1,
        coalesce=True,
        misfire_grace_time=6 * 3600,
    )

    # TITAN: Feature Matrix Runner - every 2 hours
    # Materializes feature_matrix rows for upcoming matches (enabled by default)
    scheduler.add_job(
        titan_feature_matrix_runner,
        trigger=IntervalTrigger(hours=2),
        id="titan_feature_matrix_runner",
        name="TITAN Feature Matrix Runner (every 2h)",
        replace_existing=True,
        next_run_time=datetime.utcnow() + timedelta(seconds=60),  # Offset: +60s (run soon after start)
        max_instances=1,
        coalesce=True,
        misfire_grace_time=2 * 3600,  # 2h grace for 2h interval
    )

    # TITAN: Outcome Sync - every 30 minutes
    # Updates outcome column for finished matches (enables gate metrics)
    scheduler.add_job(
        titan_outcome_sync,
        trigger=IntervalTrigger(minutes=30),
        id="titan_outcome_sync",
        name="TITAN Outcome Sync (every 30 min)",
        replace_existing=True,
        next_run_time=datetime.utcnow() + timedelta(seconds=90),  # Offset: +90s
        max_instances=1,
        coalesce=True,
        misfire_grace_time=1800,  # 30min grace
    )

    # TITAN: Lineup Fixup - every 1 hour
    # Fills missing Tier 1c/1d from SOTA lineup data captured pre-kickoff
    # Fill-only: never degrades existing True to False
    scheduler.add_job(
        titan_lineup_fixup,
        trigger=IntervalTrigger(hours=1),
        id="titan_lineup_fixup",
        name="TITAN Lineup Fixup (every 1h)",
        replace_existing=True,
        next_run_time=datetime.utcnow() + timedelta(seconds=95),  # Offset: +95s
        max_instances=1,
        coalesce=True,
        misfire_grace_time=3600,  # 1h grace
    )

    # TITAN: xG Fixup - every 1 hour
    # Fills missing Tier 1b from SOTA xG data (Understat/FotMob)
    # Fill-only: never degrades existing True to False (ABE directive)
    # xg_captured_at set to pre-kickoff (ABE P0)
    scheduler.add_job(
        titan_xg_fixup,
        trigger=IntervalTrigger(hours=1),
        id="titan_xg_fixup",
        name="TITAN xG Fixup (every 1h)",
        replace_existing=True,
        next_run_time=datetime.utcnow() + timedelta(seconds=110),  # Offset: +110s
        max_instances=1,
        coalesce=True,
        misfire_grace_time=3600,  # 1h grace
    )

    # Logos: Resize pending thumbnails - every 5 minutes
    # Processes team_logos with status='pending_resize' into WebP thumbnails
    # Disabled by default (requires LOGOS_R2_ENABLED=true)
    scheduler.add_job(
        logo_resize_pending,
        trigger=IntervalTrigger(minutes=5),
        id="logo_resize_pending",
        name="Logo Resize Pending (every 5 min)",
        replace_existing=True,
        next_run_time=datetime.utcnow() + timedelta(seconds=120),  # Offset: +120s
        max_instances=1,
        coalesce=True,
        misfire_grace_time=300,  # 5min grace
    )

    # Players & Managers: Injuries sync - every 6 hours (staggered)
    # Fetches injuries from API-Football for all tracked leagues
    scheduler.add_job(
        player_injuries_sync,
        trigger=CronTrigger(hour="0,6,12,18", minute=0),
        id="player_injuries_sync",
        name="Player Injuries Sync (every 6h)",
        replace_existing=True,
        max_instances=1,
        coalesce=True,
        misfire_grace_time=6 * 3600,
    )

    # Players & Managers: Manager sync - daily at 02:00 UTC
    # Fetches coaches, detects changes, updates history
    scheduler.add_job(
        player_manager_sync,
        trigger=CronTrigger(hour=2, minute=0),
        id="player_manager_sync",
        name="Player Manager Sync (daily 02:00 UTC)",
        replace_existing=True,
        max_instances=1,
        coalesce=True,
        misfire_grace_time=24 * 3600,
    )

    # Players: Squad sync - weekly Monday at 03:00 UTC
    # Fetches squad roster per team, upserts players catalog
    scheduler.add_job(
        player_squad_sync,
        trigger=CronTrigger(day_of_week="mon", hour=3, minute=0),
        id="player_squad_sync",
        name="Player Squad Sync (weekly Mon 03:00 UTC)",
        replace_existing=True,
        max_instances=1,
        coalesce=True,
        misfire_grace_time=24 * 3600,
    )

    # Lineups: Sync lineups for FT matches - every 6h (staggered from injuries)
    # Fetches lineups from API-Football for recently finished matches
    scheduler.add_job(
        lineup_sync,
        trigger=CronTrigger(hour="3,9,15,21", minute=30),
        id="lineup_sync",
        name="Lineup Sync (every 6h, offset)",
        replace_existing=True,
        max_instances=1,
        coalesce=True,
        misfire_grace_time=6 * 3600,
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
        f"  - CLV scoring post-match: Every 2h (canonical bookmaker, log-odds)\n"
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
        f"  - SOTA Sofascore XI capture: Every 30 min (disabled by default)\n"
        f"  - TITAN Feature Matrix Runner: Every 2h (enabled by default)\n"
        f"  - Logo Resize Pending: Every 5 min (disabled by default, requires LOGOS_R2_ENABLED)"
    )


def stop_scheduler():
    """Stop the background scheduler."""
    global _scheduler_started
    if scheduler.running:
        scheduler.shutdown()
        _scheduler_started = False
        logger.info("Scheduler stopped")
