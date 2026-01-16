"""FastAPI application for FutbolStat MVP."""

import json
import logging
import os
import time
from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.database import close_db, get_async_session, init_db, AsyncSessionLocal
from app.etl import APIFootballProvider, ETLPipeline
from app.etl.competitions import ALL_LEAGUE_IDS, COMPETITIONS
from app.features import FeatureEngineer
from app.ml import XGBoostEngine
from app.ml.persistence import load_active_model, persist_model_snapshot
from app.models import Match, OddsHistory, PITReport, PostMatchAudit, Prediction, PredictionOutcome, Team, TeamAdjustment, TeamOverride
from app.teams.overrides import preload_team_overrides, resolve_team_display
from app.scheduler import start_scheduler, stop_scheduler, get_last_sync_time, get_sync_leagues, SYNC_LEAGUES, global_sync_window
from app.security import limiter, verify_api_key, verify_api_key_or_ops_session
from app.telemetry.sentry import init_sentry, sentry_job_context, is_sentry_enabled

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Initialize Sentry for error tracking (before FastAPI app creation)
# Only activates if SENTRY_DSN is set in environment
init_sentry()

settings = get_settings()

# =============================================================================
# OPS LOG BUFFER (in-memory, filtered)
# =============================================================================

OPS_LOG_BUFFER_MAX = int(os.environ.get("OPS_LOG_BUFFER_MAX", "500"))
OPS_LOG_DEFAULT_LIMIT = int(os.environ.get("OPS_LOG_DEFAULT_LIMIT", "200"))
OPS_LOG_DEFAULT_SINCE_MINUTES = int(os.environ.get("OPS_LOG_DEFAULT_SINCE_MINUTES", "1440"))  # 24h

_ops_log_buffer: "deque[dict]" = deque(maxlen=OPS_LOG_BUFFER_MAX)
_ops_log_handler_installed = False
_seen_scheduler_started = False  # Deduplicate "Scheduler started" (show only once per deploy)


def _is_relevant_ops_log(record: logging.LogRecord, message: str) -> bool:
    """
    Decide if a log line is relevant for the operator dashboard.
    Goal: include capture/sync/budget/errors, exclude noisy scheduler/httpx spam.
    """
    name = record.name or ""
    msg = message or ""

    # Always include warnings/errors
    if record.levelno >= logging.WARNING:
        return True

    # Exclude very noisy loggers
    if name.startswith("apscheduler"):
        return False
    if name.startswith("httpx"):
        return False
    if name.startswith("uvicorn.access"):
        return False

    # Exclude common spam patterns
    spam_substrings = [
        "executed successfully",
        "Job ",
        "HTTP Request: GET",
        "HTTP Request: POST",
        # Global sync spam (keep only "complete" and "Filtered to")
        "Global sync: Fetching all fixtures",
        "Global sync: Received",
    ]
    if any(s in msg for s in spam_substrings):
        return False

    # Include key operational events (high signal)
    include_substrings = [
        "Global sync complete",
        "Lineup confirmed for match",
        "Captured lineup_confirmed odds",
        "PIT_SNAPSHOT_CREATED",
        "Market movement",
        "lineup_movement",
        "finished_match_stats",
        "Stats backfill",
        "BUDGET",
        "APIBudgetExceeded",
        "budget exceeded",
        "Rate limited",
        "ERROR",
        "Error ",
        "WARNING",
    ]
    if any(s in msg for s in include_substrings):
        return True

    # Include some internal modules even at INFO
    if name.startswith("app.scheduler") or name.startswith("app.etl.api_football"):
        # Keep only if it doesn't look like low-signal spam
        low_signal = ["Adding job tentatively", "Database tables created successfully"]
        if any(s in msg for s in low_signal):
            return False
        return True

    return False


class OpsLogBufferHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        global _seen_scheduler_started

        try:
            message = record.getMessage()
        except Exception:
            message = "<unprintable>"

        if not _is_relevant_ops_log(record, message):
            return

        # Deduplicate "Scheduler started" - show only once per deploy
        if "Scheduler started:" in message:
            if _seen_scheduler_started:
                return  # Skip duplicate
            _seen_scheduler_started = True

        ts = getattr(record, "created", None)
        dt = datetime.utcfromtimestamp(ts) if isinstance(ts, (int, float)) else datetime.utcnow()

        _ops_log_buffer.append(
            {
                "ts_utc": dt.isoformat() + "Z",
                "level": record.levelname,
                "logger": record.name,
                "message": message,
            }
        )


def _install_ops_log_handler() -> None:
    global _ops_log_handler_installed
    if _ops_log_handler_installed:
        return

    root = logging.getLogger()
    handler = OpsLogBufferHandler(level=logging.INFO)
    root.addHandler(handler)
    _ops_log_handler_installed = True


def _get_ops_logs(
    since_minutes: int = OPS_LOG_DEFAULT_SINCE_MINUTES,
    limit: int = OPS_LOG_DEFAULT_LIMIT,
    level: Optional[str] = None,
    compact: bool = False,
) -> list[dict]:
    # Copy without holding locks (deque ops are atomic-ish for append; acceptable for dashboard use)
    rows = list(_ops_log_buffer)

    # Filter by since (UTC)
    cutoff = datetime.utcnow() - timedelta(minutes=since_minutes)
    filtered = []
    for r in rows:
        ts_str = r.get("ts_utc")
        try:
            dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00")).replace(tzinfo=None) if ts_str else None
        except Exception:
            dt = None
        if dt and dt < cutoff:
            continue
        filtered.append(r)

    # Filter by minimum level if provided
    if level:
        level = level.upper()
        order = {"DEBUG": 10, "INFO": 20, "WARNING": 30, "ERROR": 40, "CRITICAL": 50}
        min_lvl = order.get(level)
        if min_lvl is not None:
            filtered = [r for r in filtered if order.get(r.get("level", "INFO"), 20) >= min_lvl]

    # Compact mode: group by message, show count + first/last timestamp
    if compact:
        from collections import OrderedDict
        groups: OrderedDict[str, dict] = OrderedDict()
        for r in filtered:
            msg = r.get("message", "")
            # Normalize message for grouping (remove variable parts like match counts)
            # Keep first 80 chars for grouping key
            key = msg[:80] if len(msg) > 80 else msg
            if key not in groups:
                groups[key] = {
                    "message": msg,
                    "level": r.get("level"),
                    "logger": r.get("logger"),
                    "count": 1,
                    "first_ts": r.get("ts_utc"),
                    "last_ts": r.get("ts_utc"),
                }
            else:
                groups[key]["count"] += 1
                groups[key]["last_ts"] = r.get("ts_utc")
        # Sort by last_ts descending (newest groups first)
        result = list(groups.values())
        result.sort(key=lambda x: x.get("last_ts", ""), reverse=True)
        return result[: max(1, int(limit))]

    # Newest first
    filtered.reverse()
    return filtered[: max(1, int(limit))]


# Global ML engine
ml_engine = XGBoostEngine()

# =============================================================================
# TELEMETRY COUNTERS (aggregated, no high-cardinality labels)
# =============================================================================
# Thread-safe via GIL for simple increments; no locks needed for counters.

_telemetry = {
    # Predictions cache
    "predictions_cache_hit_full": 0,
    "predictions_cache_hit_priority": 0,
    "predictions_cache_miss_full": 0,
    "predictions_cache_miss_priority_upgrade": 0,
    # Standings source
    "standings_source_cache": 0,
    "standings_source_db": 0,
    "standings_source_miss": 0,
    # Timeline source
    "timeline_source_db": 0,
    "timeline_source_api_fallback": 0,
}


def _incr(key: str) -> None:
    """Increment a telemetry counter."""
    _telemetry[key] = _telemetry.get(key, 0) + 1


# Simple in-memory cache for predictions
_predictions_cache = {
    "data": None,
    "timestamp": 0,
    "ttl": 300,  # 5 minutes cache
}

# Standings cache: keyed by (league_id, season), stores standings list
# TTL 30 minutes - standings don't change frequently during a match detail view
# Standings: DB-first with L1 memory cache
# Architecture: memory cache (30min TTL) -> DB (6h TTL) -> provider fallback
_standings_cache = {}  # type: dict  # L1 cache: {(league_id, season): {"data": list, "timestamp": float}}
_STANDINGS_CACHE_TTL = 1800  # 30 minutes (L1 memory)
_STANDINGS_DB_TTL = 21600  # 6 hours (DB refresh threshold)


# Many LATAM leagues run on a calendar-year season (Jan-Dec). Our default season logic
# (Jul-Jun) is correct for most European leagues but wrong for these.
_CALENDAR_YEAR_SEASON_LEAGUES = {
    71,   # Brazil - Serie A
    128,  # Argentina - Primera División
    239,  # Colombia Primera A
    242,  # Ecuador Liga Pro
    253,  # USA - MLS (calendar)
    265,  # Chile Primera Division
    268,  # Uruguay Primera - Apertura
    270,  # Uruguay Primera - Clausura
    281,  # Peru Primera Division
    299,  # Venezuela Primera Division
    344,  # Bolivia Primera Division
}

# Leagues where we should NOT filter teams by "Relegation" in previous standings,
# because they use different systems (averages) or have no relegation.
# For these leagues, we use the full previous roster in placeholder generation.
_NO_RELEGATION_FILTER_LEAGUES = {
    239,  # Colombia - relegation by multi-season averages (not single table)
    262,  # Mexico - no relegation (varies by season, assume none)
}


def _season_for_league(league_id: Optional[int], dt: datetime) -> int:
    """
    Determine API-Football 'season' parameter for a league at a given date.

    - Default: European-style season year (Jul-Jun): Jan 2026 -> 2025.
    - Calendar-year leagues (LATAM/MLS): Jan 2026 -> 2026.
    """
    if league_id is not None and league_id in _CALENDAR_YEAR_SEASON_LEAGUES:
        return dt.year
    return dt.year if dt.month >= 7 else dt.year - 1


def _get_cached_standings(league_id: int, season: int) -> Optional[list]:
    """Get standings from L1 memory cache if still valid."""
    key = (league_id, season)
    if key in _standings_cache:
        entry = _standings_cache[key]
        if time.time() - entry["timestamp"] < _STANDINGS_CACHE_TTL:
            return entry["data"]
    return None


def _set_cached_standings(league_id: int, season: int, data: list) -> None:
    """Store standings in L1 memory cache."""
    key = (league_id, season)
    _standings_cache[key] = {"data": data, "timestamp": time.time()}


async def _get_standings_from_db(session, league_id: int, season: int) -> Optional[list]:
    """Get standings from DB (L2). Returns None if not found or expired."""
    from datetime import timedelta
    result = await session.execute(
        text("""
            SELECT standings, captured_at
            FROM league_standings
            WHERE league_id = :league_id AND season = :season
        """),
        {"league_id": league_id, "season": season}
    )
    row = result.fetchone()
    if row:
        standings, captured_at = row
        # Check if data is stale (older than 6h)
        if captured_at and (datetime.now() - captured_at).total_seconds() < _STANDINGS_DB_TTL:
            return standings
    return None


async def _save_standings_to_db(session, league_id: int, season: int, standings: list) -> None:
    """Persist standings to DB with upsert."""
    from datetime import timedelta
    expires_at = datetime.now() + timedelta(seconds=_STANDINGS_DB_TTL)
    await session.execute(
        text("""
            INSERT INTO league_standings (league_id, season, standings, captured_at, expires_at, source)
            VALUES (:league_id, :season, :standings, NOW(), :expires_at, 'warmup')
            ON CONFLICT (league_id, season)
            DO UPDATE SET standings = :standings, captured_at = NOW(), expires_at = :expires_at, source = 'warmup'
        """),
        {"league_id": league_id, "season": season, "standings": json.dumps(standings), "expires_at": expires_at}
    )
    await session.commit()


async def _generate_placeholder_standings(session, league_id: int, season: int) -> list:
    """
    Generate placeholder standings for a league when API data is not yet available.

    Strategy (in order of priority):
    1. Use teams from fixtures of the new season (most accurate - reflects actual roster)
    2. Use teams from previous season standings, filtering relegated teams
    3. Fall back to teams from recent matches (least accurate)

    Returns teams with zero stats, ordered alphabetically.

    Args:
        session: Database session
        league_id: League ID
        season: Season year

    Returns:
        List of standings dicts with all zeros, ordered alphabetically by team name.
    """
    teams_data = []

    # Strategy 1: Use teams from fixtures of the target season (most accurate)
    # This reflects the actual roster including promotions/relegations
    new_season_result = await session.execute(
        text("""
            SELECT DISTINCT t.external_id, t.name, t.logo_url
            FROM teams t
            JOIN matches m ON (t.id = m.home_team_id OR t.id = m.away_team_id)
            WHERE m.league_id = :league_id
              AND EXTRACT(YEAR FROM m.date) = :season
              AND t.team_type = 'club'
            ORDER BY t.name
        """),
        {"league_id": league_id, "season": season}
    )
    for row in new_season_result.fetchall():
        teams_data.append({
            "external_id": row[0],
            "name": row[1],
            "logo_url": row[2],
        })

    if teams_data:
        logger.info(f"Using {len(teams_data)} teams from {season} fixtures for placeholder")
    else:
        # Strategy 2: Use teams from previous season standings, filtering relegated teams
        prev_standings_result = await session.execute(
            text("""
                SELECT standings
                FROM league_standings
                WHERE league_id = :league_id
                  AND season < :season
                  AND json_array_length(standings) > 0
                ORDER BY season DESC
                LIMIT 1
            """),
            {"league_id": league_id, "season": season}
        )
        prev_row = prev_standings_result.fetchone()

        if prev_row and prev_row[0]:
            prev_standings = prev_row[0]
            relegated_teams = []
            for s in prev_standings:
                desc = s.get("description") or ""
                # Only filter by "Relegation" if the league uses traditional table-based relegation.
                # Skip filtering for leagues with averages-based or no relegation system.
                if league_id not in _NO_RELEGATION_FILTER_LEAGUES:
                    if "relegation" in desc.lower():
                        relegated_teams.append(s.get("team_name"))
                        continue  # Skip relegated teams
                teams_data.append({
                    "external_id": s.get("team_id"),
                    "name": s.get("team_name"),
                    "logo_url": s.get("team_logo"),
                })
            teams_data.sort(key=lambda x: x.get("name", ""))
            if relegated_teams:
                logger.info(f"Excluded {len(relegated_teams)} relegated teams: {relegated_teams}")
            logger.info(f"Using {len(teams_data)} teams from previous standings for placeholder")

    # Strategy 3: Fallback to teams from recent matches (less accurate)
    if not teams_data:
        result = await session.execute(
            text("""
                SELECT DISTINCT t.external_id, t.name, t.logo_url
                FROM teams t
                JOIN matches m ON (t.id = m.home_team_id OR t.id = m.away_team_id)
                WHERE m.league_id = :league_id
                  AND m.date > NOW() - INTERVAL '1 year'
                  AND t.team_type = 'club'
                ORDER BY t.name
            """),
            {"league_id": league_id}
        )
        for row in result.fetchall():
            teams_data.append({
                "external_id": row[0],
                "name": row[1],
                "logo_url": row[2],
            })
        logger.info(f"Using {len(teams_data)} teams from recent matches for placeholder")

    if not teams_data:
        return []

    # Build placeholder standings ordered alphabetically (position = row number)
    standings = []
    for idx, team in enumerate(teams_data, start=1):
        standings.append({
            "position": idx,
            "team_id": team["external_id"],
            "team_name": team["name"],
            "team_logo": team["logo_url"],
            "points": 0,
            "played": 0,
            "won": 0,
            "drawn": 0,
            "lost": 0,
            "goals_for": 0,
            "goals_against": 0,
            "goal_diff": 0,
            "form": "",
            "group": None,
            "is_placeholder": True,
        })

    logger.info(f"Generated placeholder standings for league {league_id} season {season}: {len(standings)} teams")
    return standings


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    import asyncio

    # Startup
    logger.info("Starting FutbolStat MVP...")
    _install_ops_log_handler()
    await init_db()

    # Ensure ops_daily_rollups table exists (idempotent)
    async with AsyncSessionLocal() as session:
        await session.execute(text("""
            CREATE TABLE IF NOT EXISTS ops_daily_rollups (
                day DATE PRIMARY KEY,
                payload JSONB NOT NULL,
                created_at TIMESTAMP DEFAULT NOW(),
                updated_at TIMESTAMP DEFAULT NOW()
            )
        """))
        # Ensure league_standings table exists (DB-first architecture)
        await session.execute(text("""
            CREATE TABLE IF NOT EXISTS league_standings (
                id SERIAL PRIMARY KEY,
                league_id INTEGER NOT NULL,
                season INTEGER NOT NULL,
                standings JSONB NOT NULL,
                captured_at TIMESTAMP DEFAULT NOW(),
                source VARCHAR(50) DEFAULT 'api_football',
                expires_at TIMESTAMP,
                UNIQUE(league_id, season)
            )
        """))
        await session.execute(text("""
            CREATE INDEX IF NOT EXISTS ix_league_standings_league_id ON league_standings(league_id)
        """))
        await session.execute(text("""
            CREATE INDEX IF NOT EXISTS ix_league_standings_season ON league_standings(season)
        """))
        await session.commit()

    # Try to load model from PostgreSQL first (fast path)
    model_loaded = False
    async with AsyncSessionLocal() as session:
        logger.info("Checking for model in PostgreSQL...")
        model_loaded = await load_active_model(session, ml_engine)

    if model_loaded:
        logger.info("ML model loaded from PostgreSQL (fast startup)")
    elif ml_engine.load_model():
        # Fallback: try loading from local filesystem (legacy)
        logger.info("ML model loaded from filesystem (legacy)")
    elif settings.SKIP_AUTO_TRAIN:
        logger.warning(
            "No ML model found, but SKIP_AUTO_TRAIN=true. "
            "Use POST /model/train to train manually."
        )
    else:
        # Start training in background to avoid startup timeout
        logger.info("No ML model found. Starting background training...")
        asyncio.create_task(_train_model_background())

    # Initialize shadow engine if configured (FASE 2: two-stage shadow mode)
    async with AsyncSessionLocal() as session:
        from app.ml.shadow import init_shadow_engine
        shadow_initialized = await init_shadow_engine(session)
        if shadow_initialized:
            logger.info("Shadow engine initialized (two-stage model for A/B comparison)")

    # Start background scheduler for weekly sync/train
    start_scheduler(ml_engine)

    # Warm up standings cache for active leagues (non-blocking)
    asyncio.create_task(_warmup_standings_cache())

    # Predictions catch-up on startup (P2 resilience)
    # If predictions job hasn't run recently and there are upcoming matches,
    # trigger a catch-up to avoid gaps from deploys interrupting the daily cron
    asyncio.create_task(_predictions_catchup_on_startup())

    yield

    # Shutdown
    logger.info("Shutting down...")
    stop_scheduler()
    await close_db()


async def _train_model_background():
    """Train the ML model in background after startup and save to PostgreSQL."""
    import asyncio
    from concurrent.futures import ThreadPoolExecutor

    # Small delay to let server fully start
    await asyncio.sleep(2)

    try:
        logger.info("Background training started...")
        async with AsyncSessionLocal() as session:
            feature_engineer = FeatureEngineer(session=session)
            df = await feature_engineer.build_training_dataset()

            if len(df) < 100:
                logger.error(f"Insufficient training data: {len(df)} samples. Need at least 100.")
                return

            # Train in executor to avoid blocking the event loop
            loop = asyncio.get_event_loop()
            with ThreadPoolExecutor() as executor:
                result = await loop.run_in_executor(executor, ml_engine.train, df)

            # Save model to PostgreSQL for fast startup on future deploys
            snapshot_id = await persist_model_snapshot(
                session=session,
                engine=ml_engine,
                brier_score=result["brier_score"],
                cv_scores=result["cv_scores"],
                samples_trained=result["samples_trained"],
            )

            logger.info(
                f"Background training complete: {ml_engine.model_version} with {len(df)} samples. "
                f"Saved to DB as snapshot {snapshot_id}"
            )
    except Exception as e:
        logger.error(f"Background training failed: {e}")


async def _warmup_standings_cache():
    """Pre-warm standings for leagues with upcoming matches.

    DB-first architecture: fetches from provider and persists to DB + L1 cache.
    Runs at startup to ensure most match_details requests hit cache/DB.
    This is fire-and-forget - failures don't affect app health.
    """
    import asyncio

    # Small delay to let server fully start
    await asyncio.sleep(2)
    _t_start = time.time()

    try:
        async with AsyncSessionLocal() as session:
            # Get unique league_ids from matches in the next 7 days
            from datetime import timedelta
            now = datetime.now()
            week_ahead = now + timedelta(days=7)

            result = await session.execute(
                text("""
                    SELECT DISTINCT league_id
                    FROM matches
                    WHERE league_id IS NOT NULL
                      AND date >= :start_date
                      AND date <= :end_date
                      AND status = 'NS'
                    LIMIT 20
                """),
                {"start_date": now.date(), "end_date": week_ahead.date()}
            )
            league_ids = [row[0] for row in result.fetchall()]

            if not league_ids:
                logger.info("[WARMUP] No upcoming leagues to warm up")
                return

            logger.info(f"[WARMUP] Warming up standings for {len(league_ids)} leagues: {league_ids}")

            provider = APIFootballProvider()
            warmed = 0
            skipped_cache = 0
            skipped_db = 0
            failed = 0
            consecutive_failures = 0
            max_consecutive_failures = 3  # Stop if API seems down

            for league_id in league_ids:
                season = _season_for_league(league_id, now)
                # Skip if already in L1 cache
                if _get_cached_standings(league_id, season) is not None:
                    skipped_cache += 1
                    continue

                # Skip if already in DB (fresh)
                db_standings = await _get_standings_from_db(session, league_id, season)
                if db_standings is not None:
                    # Populate L1 cache from DB
                    _set_cached_standings(league_id, season, db_standings)
                    skipped_db += 1
                    continue

                # Abort if too many consecutive failures (API budget/rate limit)
                if consecutive_failures >= max_consecutive_failures:
                    logger.warning(f"[WARMUP] Aborting after {consecutive_failures} consecutive failures")
                    break

                try:
                    standings = await provider.get_standings(league_id, season)
                    # Persist to DB (primary storage)
                    await _save_standings_to_db(session, league_id, season, standings)
                    # Populate L1 cache
                    _set_cached_standings(league_id, season, standings)
                    warmed += 1
                    consecutive_failures = 0  # Reset on success
                    # Rate limit: 0.5s between calls
                    await asyncio.sleep(0.5)
                except Exception as e:
                    failed += 1
                    consecutive_failures += 1
                    logger.warning(f"[WARMUP] Failed league {league_id}: {e}")
                    # Exponential backoff on failure: 1s, 2s, 4s
                    await asyncio.sleep(min(2 ** consecutive_failures, 4))

            await provider.close()
            elapsed_ms = int((time.time() - _t_start) * 1000)
            logger.info(
                f"[WARMUP] Complete: warmed={warmed}, skipped_cache={skipped_cache}, "
                f"skipped_db={skipped_db}, failed={failed}, total_leagues={len(league_ids)}, elapsed_ms={elapsed_ms}"
            )

    except Exception as e:
        logger.error(f"[WARMUP] Standings warmup failed: {e}")


async def _predictions_catchup_on_startup():
    """
    Predictions catch-up on startup (P2 resilience).

    Handles missed daily_save_predictions runs due to deploys/restarts.
    Conditions to trigger:
    - hours_since_last_prediction_saved > 6
    - ns_next_48h > 0 (there are upcoming matches to predict)

    This is fire-and-forget, idempotent (upsert), and non-blocking.
    """
    import asyncio

    # Small delay to let server fully start and ML model load
    await asyncio.sleep(5)

    try:
        async with AsyncSessionLocal() as session:
            now = datetime.utcnow()

            # 1) Check hours since last prediction saved
            res = await session.execute(
                text("SELECT MAX(created_at) FROM predictions")
            )
            last_pred_at = res.scalar()

            hours_since_last = None
            if last_pred_at:
                delta = now - last_pred_at
                hours_since_last = delta.total_seconds() / 3600

            # 2) Check NS matches in next 48h
            res = await session.execute(
                text("""
                    SELECT COUNT(*) FROM matches
                    WHERE status = 'NS'
                      AND date > NOW()
                      AND date <= NOW() + INTERVAL '48 hours'
                """)
            )
            ns_next_48h = int(res.scalar() or 0)

            # 3) Evaluate conditions
            should_catchup = (
                (hours_since_last is None or hours_since_last > 6)
                and ns_next_48h > 0
            )

            hours_str = f"{hours_since_last:.1f}" if hours_since_last else "N/A"

            if not should_catchup:
                logger.info(
                    f"[STARTUP] Predictions catch-up skipped: "
                    f"hours_since_last={hours_str}, ns_next_48h={ns_next_48h}"
                )
                return

            # 4) Trigger catch-up
            logger.warning(
                f"[OPS_ALERT] predictions catch-up on startup triggered: "
                f"hours_since_last={hours_str}, ns_next_48h={ns_next_48h}"
            )

            # Use same logic as /dashboard/predictions/trigger endpoint
            from app.db_utils import upsert

            # Check ML model is loaded
            if not ml_engine.is_loaded:
                logger.error("[STARTUP] Predictions catch-up aborted: ML model not loaded")
                return

            # Get features for upcoming matches
            feature_engineer = FeatureEngineer(session=session)
            df = await feature_engineer.get_upcoming_matches_features()

            if len(df) == 0:
                logger.info("[STARTUP] Predictions catch-up: no upcoming matches found")
                return

            # Filter to NS only
            df_ns = df[df["status"] == "NS"].copy()

            if len(df_ns) == 0:
                logger.info("[STARTUP] Predictions catch-up: no NS matches to predict")
                return

            # Generate predictions
            predictions = ml_engine.predict(df_ns)

            # Save to database (idempotent upsert)
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
                            "model_version": ml_engine.model_version,
                            "home_prob": probs["home"],
                            "draw_prob": probs["draw"],
                            "away_prob": probs["away"],
                        },
                        conflict_columns=["match_id", "model_version"],
                        update_columns=["home_prob", "draw_prob", "away_prob"],
                    )
                    saved += 1
                except Exception as e:
                    logger.warning(f"[STARTUP] Predictions catch-up: match {match_id} failed: {e}")

            await session.commit()
            logger.info(
                f"[STARTUP] Predictions catch-up complete: saved={saved}, "
                f"ns_matches={len(df_ns)}, model={ml_engine.model_version}"
            )

    except Exception as e:
        logger.error(f"[STARTUP] Predictions catch-up failed: {e}")


app = FastAPI(
    title="FutbolStat MVP",
    description="Football Prediction System for FIFA World Cup",
    version="1.0.0",
    lifespan=lifespan,
)

# Add session middleware for OPS console login
# Secret key is required in production; fallback to random for dev
_session_secret = settings.OPS_SESSION_SECRET or os.urandom(32).hex()
if not settings.OPS_SESSION_SECRET and os.getenv("RAILWAY_PROJECT_ID"):
    logger.warning("[SECURITY] OPS_SESSION_SECRET not set in production - sessions will be invalidated on restart")

from starlette.middleware.sessions import SessionMiddleware
app.add_middleware(
    SessionMiddleware,
    secret_key=_session_secret,
    session_cookie="ops_session",
    max_age=settings.OPS_SESSION_TTL_HOURS * 3600,
    same_site="lax",
    https_only=os.getenv("RAILWAY_PROJECT_ID") is not None,  # Secure cookie in prod
)

# Add rate limiting
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


# Request/Response Models
class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


class ETLSyncRequest(BaseModel):
    league_ids: list[int]
    season: int
    fetch_odds: bool = False


class ETLSyncResponse(BaseModel):
    matches_synced: int
    teams_synced: int
    details: list[dict]


class TrainRequest(BaseModel):
    min_date: Optional[str] = None  # YYYY-MM-DD
    max_date: Optional[str] = None
    league_ids: Optional[list[int]] = None


class TrainResponse(BaseModel):
    model_version: str
    brier_score: float
    samples_trained: int
    feature_importance: dict


class PredictionItem(BaseModel):
    """Prediction item with contextual intelligence for iOS consumption."""
    match_id: Optional[int] = None
    match_external_id: Optional[int] = None
    home_team: str
    away_team: str
    home_team_logo: Optional[str] = None
    away_team_logo: Optional[str] = None
    date: datetime
    status: Optional[str] = None  # Match status: NS, FT, 1H, 2H, HT, etc.
    home_goals: Optional[int] = None  # Final score (nil if not played)
    away_goals: Optional[int] = None  # Final score (nil if not played)
    league_id: Optional[int] = None
    venue: Optional[dict] = None  # Stadium: {"name": str, "city": str} or None

    # Adjusted probabilities (after team adjustments)
    probabilities: dict
    # Raw model output before adjustments
    raw_probabilities: Optional[dict] = None

    fair_odds: dict
    market_odds: Optional[dict] = None

    # Confidence tier with degradation tracking
    confidence_tier: Optional[str] = None  # gold, silver, copper
    original_tier: Optional[str] = None    # Original tier before degradation

    # Value betting
    value_bets: Optional[list[dict]] = None
    has_value_bet: Optional[bool] = False
    best_value_bet: Optional[dict] = None

    # Contextual adjustments applied
    adjustment_applied: Optional[bool] = False
    adjustments: Optional[dict] = None

    # Reasoning engine (human-readable insights)
    prediction_insights: Optional[list[str]] = None
    warnings: Optional[list[str]] = None

    # Frozen prediction data (for finished matches)
    is_frozen: Optional[bool] = False
    frozen_at: Optional[str] = None  # ISO datetime when prediction was frozen
    frozen_ev: Optional[dict] = None  # EV values at freeze time

    # Rerun serving (DB-first gated)
    served_from_rerun: Optional[bool] = None  # True if served from DB rerun prediction
    rerun_model_version: Optional[str] = None  # Model version of rerun prediction


class PredictionsResponse(BaseModel):
    predictions: list[PredictionItem]
    model_version: str
    # Metadata about contextual filters applied
    context_applied: Optional[dict] = None


# Endpoints
@app.get("/health", response_model=HealthResponse)
@limiter.limit("120/minute")
async def health_check(request: Request):
    """Health check endpoint."""
    return HealthResponse(
        status="ok",
        model_loaded=ml_engine.is_loaded,
    )


@app.get("/telemetry")
async def get_telemetry(request: Request):
    """
    Aggregated telemetry counters for cache hit/miss monitoring.

    No high-cardinality labels (no match_id, team names, URLs).
    Safe for Prometheus/Grafana scraping.

    Protected by X-Dashboard-Token (same as other dashboard endpoints).

    NOTE: Counters reset on redeploy/restart. This is diagnostic telemetry,
    not historical observability. For persistent metrics, export to Prometheus.
    """
    if not _verify_dashboard_token(request):
        raise HTTPException(status_code=401, detail="Telemetry access requires valid token.")

    # Calculate hit rates
    pred_hits = _telemetry["predictions_cache_hit_full"] + _telemetry["predictions_cache_hit_priority"]
    pred_misses = _telemetry["predictions_cache_miss_full"] + _telemetry["predictions_cache_miss_priority_upgrade"]
    pred_total = pred_hits + pred_misses
    pred_hit_rate = pred_hits / pred_total if pred_total > 0 else 0

    standings_hits = _telemetry["standings_source_cache"] + _telemetry["standings_source_db"]
    standings_total = standings_hits + _telemetry["standings_source_miss"]
    standings_hit_rate = standings_hits / standings_total if standings_total > 0 else 0

    timeline_total = _telemetry["timeline_source_db"] + _telemetry["timeline_source_api_fallback"]
    timeline_db_rate = _telemetry["timeline_source_db"] / timeline_total if timeline_total > 0 else 0

    return {
        "predictions_cache": {
            "hit_full": _telemetry["predictions_cache_hit_full"],
            "hit_priority": _telemetry["predictions_cache_hit_priority"],
            "miss_full": _telemetry["predictions_cache_miss_full"],
            "miss_priority_upgrade": _telemetry["predictions_cache_miss_priority_upgrade"],
            "hit_rate": round(pred_hit_rate, 3),
        },
        "standings_source": {
            "cache": _telemetry["standings_source_cache"],
            "db": _telemetry["standings_source_db"],
            "miss": _telemetry["standings_source_miss"],
            "hit_rate": round(standings_hit_rate, 3),
        },
        "timeline_source": {
            "db": _telemetry["timeline_source_db"],
            "api_fallback": _telemetry["timeline_source_api_fallback"],
            "db_rate": round(timeline_db_rate, 3),
        },
    }


@app.get("/metrics")
async def prometheus_metrics(
    authorization: str = Header(None, alias="Authorization"),
):
    """
    Prometheus metrics endpoint for Data Quality Telemetry.

    Exposes metrics for:
    - Provider ingestion (requests, errors, latency)
    - Anti-lookahead (event latency, tainted records)
    - Market integrity (odds validation, overround)
    - Entity mapping coverage

    Scrape this endpoint from Grafana Cloud or Prometheus.
    Requires Bearer token authentication via METRICS_BEARER_TOKEN env var.
    """
    from fastapi.responses import PlainTextResponse

    # Validate Bearer token if configured
    expected_token = getattr(settings, "METRICS_BEARER_TOKEN", None)
    if expected_token:
        if not authorization:
            return PlainTextResponse(
                content="# Unauthorized: Missing Authorization header\n",
                status_code=401,
                media_type="text/plain",
            )
        # Extract token from "Bearer <token>"
        parts = authorization.split(" ", 1)
        if len(parts) != 2 or parts[0].lower() != "bearer":
            return PlainTextResponse(
                content="# Unauthorized: Invalid Authorization format\n",
                status_code=401,
                media_type="text/plain",
            )
        if parts[1] != expected_token:
            return PlainTextResponse(
                content="# Unauthorized: Invalid token\n",
                status_code=401,
                media_type="text/plain",
            )

    try:
        from app.telemetry import get_metrics_text
        content, content_type = get_metrics_text()
        return PlainTextResponse(content=content, media_type=content_type)
    except ImportError:
        # Fallback if prometheus_client not installed
        return PlainTextResponse(
            content="# Telemetry module not available\n",
            media_type="text/plain",
        )


@app.get("/sync/status")
async def get_sync_status():
    """
    Get current sync status for iOS display.

    Returns last sync timestamp and API budget info.
    Used by mobile app to show data freshness.
    """
    last_sync = get_last_sync_time()

    # Best-effort: expose real budget numbers (prefer internal guardrail, optionally enrich with cached /status)
    daily_budget = int(getattr(settings, "API_DAILY_BUDGET", 0) or 0) or 75000
    daily_used = None
    remaining_pct = None
    api_account_status = None

    try:
        from app.etl.api_football import get_api_budget_status, get_api_account_status  # type: ignore

        internal = get_api_budget_status()
        # Internal is authoritative for guardrail; can be None early in day before first request
        daily_used = internal.get("budget_used")
        daily_budget = internal.get("budget_total") or daily_budget

        # Enrich with real API status (cached 10 min); don't fail endpoint if unavailable
        api_account_status = await get_api_account_status()  # type: ignore
        ext_used = api_account_status.get("requests_today")
        ext_limit = api_account_status.get("requests_limit")
        if isinstance(ext_used, int) and ext_used >= 0:
            # Use the max to avoid under-reporting after process restart
            if isinstance(daily_used, int):
                daily_used = max(daily_used, ext_used)
            else:
                daily_used = ext_used
        if isinstance(ext_limit, int) and ext_limit > 0:
            daily_budget = ext_limit
    except Exception:
        pass

    if isinstance(daily_used, int) and daily_budget > 0:
        remaining_pct = round((1 - (daily_used / daily_budget)) * 100, 1)

    return {
        "last_sync_at": last_sync.isoformat() if last_sync else None,
        "sync_interval_seconds": 60,
        "daily_api_calls": daily_used,
        "daily_budget": daily_budget,
        "budget_remaining_percent": remaining_pct,
        "leagues": get_sync_leagues(),
        "api_account_status": api_account_status,  # optional debug visibility for iOS
    }


@app.get("/")
@limiter.limit("60/minute")
async def root(request: Request):
    """Root endpoint with API info."""
    return {
        "name": "FutbolStat MVP",
        "version": "1.0.0",
        "description": "Football Prediction System for FIFA World Cup",
        "endpoints": {
            "health": "/health",
            "etl_sync": "POST /etl/sync",
            "train": "POST /model/train",
            "predictions": "GET /predictions/upcoming",
        },
    }


@app.post("/etl/sync", response_model=ETLSyncResponse)
@limiter.limit("10/minute")
async def etl_sync(
    request: Request,
    body: ETLSyncRequest,
    session: AsyncSession = Depends(get_async_session),
    _: bool = Depends(verify_api_key),
):
    """
    Sync fixtures from API-Football.

    Fetches matches for specified leagues and season.
    Requires API key authentication.
    """
    logger.info(f"ETL sync request: {body}")

    # Validate league IDs
    for league_id in body.league_ids:
        if league_id not in COMPETITIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown league ID: {league_id}. Valid IDs: {ALL_LEAGUE_IDS}",
            )

    provider = APIFootballProvider()
    try:
        pipeline = ETLPipeline(provider=provider, session=session)
        result = await pipeline.sync_multiple_leagues(
            league_ids=body.league_ids,
            season=body.season,
            fetch_odds=body.fetch_odds,
        )

        return ETLSyncResponse(
            matches_synced=result["total_matches_synced"],
            teams_synced=result["total_teams_synced"],
            details=result["details"],
        )
    finally:
        await provider.close()


@app.post("/etl/sync-historical")
@limiter.limit("5/minute")
async def etl_sync_historical(
    request: Request,
    start_year: int = 2018,
    end_year: Optional[int] = None,
    league_ids: Optional[list[int]] = None,
    session: AsyncSession = Depends(get_async_session),
    _: bool = Depends(verify_api_key),
):
    """
    Sync historical data for multiple seasons.

    This is a long-running operation. Use for initial data loading.
    Requires API key authentication.
    """
    if league_ids is None:
        league_ids = ALL_LEAGUE_IDS

    provider = APIFootballProvider()
    try:
        pipeline = ETLPipeline(provider=provider, session=session)
        result = await pipeline.sync_historical_data(
            league_ids=league_ids,
            start_year=start_year,
            end_year=end_year,
        )
        return result
    finally:
        await provider.close()


@app.post("/etl/sync-window")
@limiter.limit("5/minute")
async def etl_sync_window(
    request: Request,
    days_ahead: int = 10,
    days_back: int = 1,
    _: bool = Depends(verify_api_key),
):
    """
    Sync fixtures by date window (not by season).

    This endpoint triggers the global_sync_window job which loads fixtures
    for a range of dates regardless of season. Useful for loading LATAM 2026
    fixtures when CURRENT_SEASON is still set to 2025.

    Args:
        days_ahead: Days ahead to sync (default: 10)
        days_back: Days back to sync (default: 1)

    Requires API key authentication.
    """
    from app.ops.audit import log_ops_action

    logger.info(f"[ETL] sync-window request: days_back={days_back}, days_ahead={days_ahead}")

    start_time = time.time()
    result = await global_sync_window(days_ahead=days_ahead, days_back=days_back)
    duration_ms = int((time.time() - start_time) * 1000)

    # Audit log
    try:
        async with AsyncSessionLocal() as audit_session:
            await log_ops_action(
                session=audit_session,
                request=request,
                action="sync_window",
                params={"days_ahead": days_ahead, "days_back": days_back},
                result="ok" if not result.get("error") else "error",
                result_detail={
                    "matches_synced": result.get("matches_synced", 0),
                    "days_processed": result.get("days_processed", 0),
                },
                error_message=result.get("error"),
                duration_ms=duration_ms,
            )
    except Exception as audit_err:
        logger.warning(f"Failed to log audit for sync_window: {audit_err}")

    return {
        "status": "ok",
        "matches_synced": result.get("matches_synced", 0),
        "days_processed": result.get("days_processed", 0),
        "window": result.get("window", {}),
        "by_date": result.get("by_date", {}),
        "error": result.get("error"),
    }


@app.post("/etl/refresh-aggregates")
@limiter.limit("5/minute")
async def etl_refresh_aggregates(
    request: Request,
    session: AsyncSession = Depends(get_async_session),
    _: bool = Depends(verify_api_key),
):
    """
    Manually trigger league aggregates refresh.

    Computes league baselines and team profiles for all leagues with
    sufficient data. This is the same job that runs daily at 06:30 UTC.

    Returns metrics about the refresh operation.
    Requires API key authentication.
    """
    logger.info("[AGGREGATES] Manual refresh triggered via API")

    from app.aggregates.refresh_job import refresh_all_aggregates, get_aggregates_status

    # Get status before
    status_before = await get_aggregates_status(session)

    # Run refresh
    result = await refresh_all_aggregates(session)

    # Get status after
    status_after = await get_aggregates_status(session)

    return {
        "status": "ok",
        "refresh_result": result,
        "status_before": status_before,
        "status_after": status_after,
    }


@app.get("/aggregates/status")
@limiter.limit("30/minute")
async def get_aggregates_status_endpoint(
    request: Request,
    session: AsyncSession = Depends(get_async_session),
    _: bool = Depends(verify_api_key),
):
    """
    Get current status of league aggregates tables.

    Returns counts and latest computation timestamps.
    Requires API key authentication.
    """
    from app.aggregates.refresh_job import get_aggregates_status

    status = await get_aggregates_status(session)
    return {"status": "ok", **status}


@app.get("/aggregates/breakdown")
@limiter.limit("30/minute")
async def get_aggregates_breakdown(
    request: Request,
    session: AsyncSession = Depends(get_async_session),
    _: bool = Depends(verify_api_key),
):
    """
    Get detailed breakdown of aggregates by dimension.

    Clarifies what baselines_created and profiles_created represent.
    Requires API key authentication.
    """
    from sqlalchemy import select, func, distinct
    from app.models import LeagueSeasonBaseline, LeagueTeamProfile

    # Baselines breakdown
    total_baselines = (await session.execute(
        select(func.count(LeagueSeasonBaseline.id))
    )).scalar() or 0

    distinct_leagues = (await session.execute(
        select(func.count(distinct(LeagueSeasonBaseline.league_id)))
    )).scalar() or 0

    distinct_seasons = (await session.execute(
        select(func.count(distinct(LeagueSeasonBaseline.season)))
    )).scalar() or 0

    distinct_dates = (await session.execute(
        select(func.count(distinct(LeagueSeasonBaseline.as_of_date)))
    )).scalar() or 0

    # Profiles breakdown
    total_profiles = (await session.execute(
        select(func.count(LeagueTeamProfile.id))
    )).scalar() or 0

    distinct_teams = (await session.execute(
        select(func.count(distinct(LeagueTeamProfile.team_id)))
    )).scalar() or 0

    profiles_with_min_sample = (await session.execute(
        select(func.count(LeagueTeamProfile.id))
        .where(LeagueTeamProfile.min_sample_ok == True)
    )).scalar() or 0

    # Season distribution
    seasons_result = await session.execute(
        select(
            LeagueSeasonBaseline.season,
            func.count(LeagueSeasonBaseline.id)
        )
        .group_by(LeagueSeasonBaseline.season)
        .order_by(LeagueSeasonBaseline.season.desc())
    )
    seasons_breakdown = {str(row[0]): row[1] for row in seasons_result}

    return {
        "status": "ok",
        "baselines": {
            "total_rows": total_baselines,
            "distinct_league_id": distinct_leagues,
            "distinct_season": distinct_seasons,
            "distinct_as_of_date": distinct_dates,
            "note": "Each row = one (league_id, season, as_of_date) combination",
        },
        "profiles": {
            "total_rows": total_profiles,
            "distinct_team_id": distinct_teams,
            "with_min_sample_ok": profiles_with_min_sample,
            "note": "Each row = one (league_id, season, team_id, as_of_date) combination",
        },
        "seasons_breakdown": seasons_breakdown,
    }


@app.post("/model/train", response_model=TrainResponse)
@limiter.limit("5/minute")
async def train_model(
    request: Request,
    body: TrainRequest = None,
    session: AsyncSession = Depends(get_async_session),
    _: bool = Depends(verify_api_key),
):
    """
    Train the prediction model.

    Uses historical match data to train XGBoost model.
    Requires API key authentication.
    """
    body = body or TrainRequest()

    logger.info("Starting model training...")

    # Parse dates
    min_date = None
    max_date = None
    if body.min_date:
        min_date = datetime.strptime(body.min_date, "%Y-%m-%d")
    if body.max_date:
        max_date = datetime.strptime(body.max_date, "%Y-%m-%d")

    # Build training dataset
    feature_engineer = FeatureEngineer(session=session)
    df = await feature_engineer.build_training_dataset(
        min_date=min_date,
        max_date=max_date,
        league_ids=body.league_ids,
    )

    if len(df) < 100:
        raise HTTPException(
            status_code=400,
            detail=f"Insufficient training data: {len(df)} samples. Need at least 100.",
        )

    # Train model in executor to avoid blocking the event loop
    import asyncio
    from concurrent.futures import ThreadPoolExecutor

    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as executor:
        result = await loop.run_in_executor(executor, ml_engine.train, df)

    # Save model to PostgreSQL for fast startup on future deploys
    snapshot_id = await persist_model_snapshot(
        session=session,
        engine=ml_engine,
        brier_score=result["brier_score"],
        cv_scores=result["cv_scores"],
        samples_trained=result["samples_trained"],
    )
    logger.info(f"Model saved to PostgreSQL as snapshot {snapshot_id}")

    return TrainResponse(
        model_version=result["model_version"],
        brier_score=result["brier_score"],
        samples_trained=result["samples_trained"],
        feature_importance=result["feature_importance"],
    )


@app.get("/predictions/upcoming", response_model=PredictionsResponse)
@limiter.limit("30/minute")
async def get_predictions(
    request: Request,
    league_ids: Optional[str] = None,  # comma-separated
    days: int = 7,  # Legacy: applies to both back and ahead if specific params not set
    days_back: Optional[int] = None,  # Past N days (finished matches with scores)
    days_ahead: Optional[int] = None,  # Future N days (upcoming matches)
    save: bool = False,  # Save predictions to database
    with_context: bool = True,  # Apply contextual intelligence
    session: AsyncSession = Depends(get_async_session),
):
    """
    Get predictions for upcoming matches with contextual intelligence.

    Returns probabilities, fair odds, and reasoning insights for matches.
    Applies team adjustments, league drift detection, and market movement analysis.

    Args:
        league_ids: Comma-separated league IDs to filter
        days: Legacy param - applies to both directions if days_back/days_ahead not set
        days_back: Past N days for finished matches (overrides 'days' for past)
        days_ahead: Future N days for upcoming matches (overrides 'days' for future)
        save: Persist predictions to database for auditing
        with_context: Apply contextual intelligence (team adjustments, drift, odds)

    Priority window example: ?days_back=1&days_ahead=1 → yesterday/today/tomorrow
    Full window example: ?days_back=7&days_ahead=7 → 15-day range

    Uses in-memory caching (5 min TTL) for faster responses.
    """
    global _predictions_cache

    if not ml_engine.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Train a model first with POST /model/train",
        )

    # Resolve actual days_back and days_ahead (new params override legacy 'days')
    actual_days_back = days_back if days_back is not None else days
    actual_days_ahead = days_ahead if days_ahead is not None else days
    logger.info(f"Predictions params: days={days}, days_back={days_back}, days_ahead={days_ahead} -> actual_back={actual_days_back}, actual_ahead={actual_days_ahead}")

    # Cache key based on parameters
    cache_key = f"{league_ids or 'all'}_{actual_days_back}_{actual_days_ahead}_{with_context}"
    now = time.time()

    # Check cache (only for default full requests without league filter)
    is_default_full = (
        league_ids is None
        and actual_days_back == 7
        and actual_days_ahead == 7
        and not save
        and with_context
    )
    if is_default_full and _predictions_cache["data"] is not None:
        if now - _predictions_cache["timestamp"] < _predictions_cache["ttl"]:
            logger.info("Returning cached predictions")
            return _predictions_cache["data"]

    # Priority optimization: serve any cacheable request from full (7+7) cache
    # This applies to priority requests (1+1) or any subset of the default window
    is_cacheable_subset = (
        league_ids is None
        and actual_days_back <= 7
        and actual_days_ahead <= 7
        and not save
        and with_context
    )

    # Helper to filter predictions by date range
    def _filter_predictions_by_range(
        cached_response: PredictionsResponse,
        days_back: int,
        days_ahead: int
    ) -> PredictionsResponse:
        from datetime import timezone
        today_start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        range_start = today_start - timedelta(days=days_back)
        range_end = today_start + timedelta(days=days_ahead + 1)  # +1 to include full day

        filtered = [
            p for p in cached_response.predictions
            if range_start <= p.date.replace(tzinfo=timezone.utc) < range_end
        ]
        return PredictionsResponse(
            predictions=filtered,
            model_version=cached_response.model_version,
            context_applied=cached_response.context_applied,
        )

    # If cache is warm and request is a subset, filter and return immediately
    if is_cacheable_subset and _predictions_cache["data"] is not None:
        if now - _predictions_cache["timestamp"] < _predictions_cache["ttl"]:
            if is_default_full:
                # Full request with warm cache - return as-is (already handled above, but safety)
                _incr("predictions_cache_hit_full")
                logger.info("predictions_cache | cache_hit | type=full, count=%d", len(_predictions_cache["data"].predictions))
                return _predictions_cache["data"]
            else:
                # Subset request - filter from cache
                result = _filter_predictions_by_range(
                    _predictions_cache["data"],
                    actual_days_back,
                    actual_days_ahead
                )
                _incr("predictions_cache_hit_priority")
                logger.info(
                    "predictions_cache | cache_hit | type=priority, filtered_count=%d, full_count=%d, days_back=%d, days_ahead=%d",
                    len(result.predictions), len(_predictions_cache["data"].predictions),
                    actual_days_back, actual_days_ahead
                )
                return result

    # Parse league IDs
    league_id_list = None
    if league_ids:
        league_id_list = [int(x.strip()) for x in league_ids.split(",")]

    # Cold start optimization: when cache is cold and request is a subset,
    # always fetch full (7+7) to populate cache, then filter result
    # This ensures first priority request warms the cache for subsequent requests
    fetch_days_back = actual_days_back
    fetch_days_ahead = actual_days_ahead
    needs_filtering = False

    if is_cacheable_subset and not is_default_full and league_id_list is None:
        # Subset request with cold cache - fetch full range to warm cache
        _incr("predictions_cache_miss_priority_upgrade")
        logger.info(
            "predictions_cache | cache_miss | type=priority_upgrade, requested_days=%d+%d, fetching=7+7",
            actual_days_back, actual_days_ahead
        )
        fetch_days_back = 7
        fetch_days_ahead = 7
        needs_filtering = True
    elif is_cacheable_subset and is_default_full:
        _incr("predictions_cache_miss_full")
        logger.info("predictions_cache | cache_miss | type=full")
    else:
        logger.info("predictions_cache | cache_bypass | league_ids=%s, save=%s, with_context=%s",
                    league_ids, save, with_context)

    # Track compute time for cache miss (with per-stage timing)
    _compute_start = time.time()
    _stage_times = {}

    # Get features for upcoming matches
    # iOS progressive loading:
    #   Priority: days_back=1, days_ahead=1 → yesterday/today/tomorrow (~50-100 matches)
    #   Full: days_back=7, days_ahead=7 → 15-day window (~300 matches)
    _t0 = time.time()
    feature_engineer = FeatureEngineer(session=session)
    df = await feature_engineer.get_upcoming_matches_features(
        league_ids=league_id_list,
        include_recent_days=fetch_days_back,  # Past N days for finished matches
        days_ahead=fetch_days_ahead,  # Future N days for upcoming matches
    )
    _stage_times["features_ms"] = (time.time() - _t0) * 1000
    logger.info(f"Predictions query: days_back={fetch_days_back}, days_ahead={fetch_days_ahead}, matches={len(df)}")

    if len(df) == 0:
        return PredictionsResponse(
            predictions=[],
            model_version=ml_engine.model_version,
        )

    # Load contextual intelligence data
    team_adjustments = None
    context = None
    context_metadata = {
        "team_adjustments_loaded": False,
        "unstable_leagues": 0,
        "odds_movements_detected": 0,
    }

    if with_context:
        from app.ml.recalibration import RecalibrationEngine, load_team_adjustments, get_drift_cache_stats
        _t1 = time.time()

        try:
            # Load team adjustments (includes raw data to avoid duplicate query)
            _t_adj = time.time()
            team_adjustments = await load_team_adjustments(session)
            _stage_times["adjustments_ms"] = (time.time() - _t_adj) * 1000
            context_metadata["team_adjustments_loaded"] = True

            # Initialize recalibrator for context gathering
            recalibrator = RecalibrationEngine(session)

            # Detect unstable leagues (with TTL cache)
            _drift_stats_before = get_drift_cache_stats()
            _t_drift = time.time()
            drift_result = await recalibrator.detect_league_drift()
            unstable_leagues = {alert["league_id"] for alert in drift_result.get("drift_alerts", [])}
            _stage_times["drift_ms"] = (time.time() - _t_drift) * 1000
            _drift_stats_after = get_drift_cache_stats()
            _drift_was_hit = _drift_stats_after["hits"] > _drift_stats_before["hits"]
            _stage_times["drift_cache"] = "HIT" if _drift_was_hit else "MISS"
            context_metadata["unstable_leagues"] = len(unstable_leagues)

            # Check odds movements for upcoming matches (batch query, no N+1)
            _t_odds = time.time()
            odds_result = await recalibrator.check_all_upcoming_odds_movements(days_ahead=days)
            odds_movements = {
                alert["match_id"]: alert
                for alert in odds_result.get("alerts", [])
            }
            _stage_times["odds_ms"] = (time.time() - _t_odds) * 1000
            context_metadata["odds_movements_detected"] = len(odds_movements)

            # Build team details from already-loaded adjustments (no duplicate query)
            team_details = {}
            for adj in team_adjustments.get("raw", []):
                home_anomaly_rate = adj.home_anomalies / adj.home_predictions if adj.home_predictions > 0 else 0
                away_anomaly_rate = adj.away_anomalies / adj.away_predictions if adj.away_predictions > 0 else 0
                team_details[adj.team_id] = {
                    "home_anomaly_rate": home_anomaly_rate,
                    "away_anomaly_rate": away_anomaly_rate,
                    "consecutive_minimal": adj.consecutive_minimal_count,
                    "international_penalty": adj.international_penalty,
                }

            # Build context dictionary
            context = {
                "unstable_leagues": unstable_leagues,
                "odds_movements": odds_movements,
                "international_commitments": {},  # Filled from team_details
                "team_details": team_details,
            }

            # Add international commitments from team_details
            for team_id, details in team_details.items():
                if details["international_penalty"] < 1.0:
                    context["international_commitments"][team_id] = {
                        "penalty": details["international_penalty"],
                        "days": 3,  # Approximation
                    }

            _stage_times["context_ms"] = (time.time() - _t1) * 1000
            logger.info(
                f"Context loaded: {len(unstable_leagues)} unstable leagues, "
                f"{len(odds_movements)} odds movements"
            )

        except Exception as e:
            _stage_times["context_ms"] = (time.time() - _t1) * 1000
            logger.warning(f"Error loading context: {e}. Predictions will be made without context.")

    # Make predictions with context
    _t2 = time.time()
    predictions = ml_engine.predict(df, team_adjustments=team_adjustments, context=context)
    _stage_times["predict_ms"] = (time.time() - _t2) * 1000

    # For finished matches, overlay frozen prediction data if available
    _t3 = time.time()
    predictions = await _overlay_frozen_predictions(session, predictions)
    _stage_times["overlay_ms"] = (time.time() - _t3) * 1000

    # For NS matches, overlay rerun predictions if PREFER_RERUN_PREDICTIONS=true
    _t3b = time.time()
    # Build match_dates dict from DataFrame for freshness check
    match_dates = {}
    if "match_id" in df.columns and "date" in df.columns:
        for _, row in df.iterrows():
            mid = row.get("match_id")
            mdate = row.get("date")
            if mid and mdate:
                match_dates[int(mid)] = mdate if isinstance(mdate, datetime) else datetime.fromisoformat(str(mdate).replace("Z", "+00:00"))
    predictions, rerun_stats = await _overlay_rerun_predictions(session, predictions, match_dates)
    _stage_times["rerun_overlay_ms"] = (time.time() - _t3b) * 1000
    if rerun_stats.get("db_hits", 0) > 0 or rerun_stats.get("db_stale", 0) > 0:
        logger.info(
            f"rerun_serving | db_hits={rerun_stats['db_hits']} db_stale={rerun_stats['db_stale']} "
            f"live_fallback={rerun_stats['live_fallback']} total_ns={rerun_stats['total_ns']}"
        )

    # Apply team identity overrides (rebranding, e.g., La Equidad → Internacional de Bogotá)
    _t4 = time.time()
    predictions = await _apply_team_overrides(session, predictions)
    _stage_times["overrides_ms"] = (time.time() - _t4) * 1000

    # Save predictions to database if requested
    if save:
        saved_count = await _save_predictions_to_db(session, predictions, ml_engine.model_version)
        logger.info(f"Saved {saved_count} predictions to database")

    # Convert to response model
    prediction_items = []
    for pred in predictions:
        item = PredictionItem(
            match_id=pred.get("match_id"),
            match_external_id=pred.get("match_external_id"),
            home_team=pred["home_team"],
            away_team=pred["away_team"],
            home_team_logo=pred.get("home_team_logo"),
            away_team_logo=pred.get("away_team_logo"),
            date=pred["date"],
            status=pred.get("status"),
            home_goals=pred.get("home_goals"),
            away_goals=pred.get("away_goals"),
            league_id=pred.get("league_id"),
            venue=pred.get("venue"),
            probabilities=pred["probabilities"],
            raw_probabilities=pred.get("raw_probabilities"),
            fair_odds=pred["fair_odds"],
            market_odds=pred.get("market_odds"),
            confidence_tier=pred.get("confidence_tier"),
            original_tier=pred.get("original_tier"),
            value_bets=pred.get("value_bets"),
            has_value_bet=pred.get("has_value_bet", False),
            best_value_bet=pred.get("best_value_bet"),
            adjustment_applied=pred.get("adjustment_applied", False),
            adjustments=pred.get("adjustments"),
            prediction_insights=pred.get("prediction_insights"),
            warnings=pred.get("warnings"),
            # Frozen prediction fields
            is_frozen=pred.get("is_frozen", False),
            frozen_at=pred.get("frozen_at"),
            frozen_ev=pred.get("frozen_ev"),
            # Rerun serving fields
            served_from_rerun=pred.get("served_from_rerun"),
            rerun_model_version=pred.get("rerun_model_version"),
        )
        prediction_items.append(item)

    response = PredictionsResponse(
        predictions=prediction_items,
        model_version=ml_engine.model_version,
        context_applied=context_metadata if with_context else None,
    )

    # Compute time for telemetry
    _compute_ms = (time.time() - _compute_start) * 1000
    _stage_times["total_ms"] = _compute_ms

    # Log per-stage timing breakdown for performance monitoring
    # Separate numeric timings from string metadata (like drift_cache hit/miss)
    _timing_parts = [f"{k}={v:.0f}" for k, v in _stage_times.items() if isinstance(v, (int, float))]
    _meta_parts = [f"{k}={v}" for k, v in _stage_times.items() if isinstance(v, str)]
    logger.info(
        "predictions_timing | %s | %s | matches=%d",
        " | ".join(_timing_parts),
        " | ".join(_meta_parts) if _meta_parts else "no_meta",
        len(prediction_items)
    )

    # Cache the response (for 7+7 requests or upgraded priority requests)
    # This ensures the cache is always populated with full data
    should_cache = is_default_full or needs_filtering
    if should_cache:
        _predictions_cache["data"] = response
        _predictions_cache["timestamp"] = now
        logger.info(
            "predictions_cache | cached | compute_ms=%.1f, full_count=%d, type=%s",
            _compute_ms, len(prediction_items),
            "priority_upgrade" if needs_filtering else "full"
        )

    # If this was an upgraded priority request, filter the result before returning
    if needs_filtering:
        filtered_response = _filter_predictions_by_range(
            response,
            actual_days_back,
            actual_days_ahead
        )
        logger.info(
            "predictions_cache | filtered | filtered_count=%d, full_count=%d, days_back=%d, days_ahead=%d",
            len(filtered_response.predictions), len(response.predictions),
            actual_days_back, actual_days_ahead
        )
        return filtered_response

    return response


async def _save_predictions_to_db(
    session: AsyncSession,
    predictions: list[dict],
    model_version: str,
) -> int:
    """Save predictions to database for later auditing."""
    from app.db_utils import upsert

    saved = 0
    for pred in predictions:
        match_id = pred.get("match_id")
        if not match_id:
            continue

        probs = pred["probabilities"]

        try:
            # Use generic upsert for cross-database compatibility
            await upsert(
                session,
                Prediction,
                values={
                    "match_id": match_id,
                    "model_version": model_version,
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

    await session.commit()
    return saved


# Metrics counters for rerun serving (DB-first vs live fallback)
_rerun_serving_stats = {
    "db_hits": 0,
    "db_stale": 0,
    "live_fallback": 0,
    "total_ns_served": 0,
}


async def _overlay_rerun_predictions(
    session: AsyncSession,
    predictions: list[dict],
    match_dates: dict[int, datetime],  # match_id -> match date for freshness check
) -> tuple[list[dict], dict]:
    """
    Overlay rerun predictions from DB for NS matches when PREFER_RERUN_PREDICTIONS=true.

    This implements "DB-first gated" serving:
    1. If PREFER_RERUN_PREDICTIONS=false: return predictions unchanged (baseline)
    2. If true: for each NS match, try to serve from DB if:
       - A prediction with run_id exists (from a rerun)
       - The prediction is "fresh" (created within RERUN_FRESHNESS_HOURS of match kickoff)
    3. If no fresh DB prediction: fall back to live baseline

    Returns:
        tuple: (modified predictions, serving stats dict)
    """
    settings = get_settings()
    stats = {"db_hits": 0, "db_stale": 0, "live_fallback": 0, "total_ns": 0}

    # If flag is off, return unchanged
    if not settings.PREFER_RERUN_PREDICTIONS:
        return predictions, stats

    if not predictions:
        return predictions, stats

    # Get NS match IDs
    ns_match_ids = [
        p.get("match_id") for p in predictions
        if p.get("match_id") and p.get("status") == "NS"
    ]
    if not ns_match_ids:
        return predictions, stats

    stats["total_ns"] = len(ns_match_ids)

    # Query rerun predictions (those with run_id, most recent per match)
    # Using a subquery to get the latest prediction per match_id with run_id
    result = await session.execute(
        text("""
            SELECT DISTINCT ON (match_id)
                match_id, model_version, home_prob, draw_prob, away_prob,
                created_at, run_id
            FROM predictions
            WHERE match_id = ANY(:match_ids)
              AND run_id IS NOT NULL
            ORDER BY match_id, created_at DESC
        """),
        {"match_ids": ns_match_ids}
    )
    rerun_preds = {row[0]: row for row in result.fetchall()}

    if not rerun_preds:
        stats["live_fallback"] = len(ns_match_ids)
        return predictions, stats

    # Freshness threshold
    freshness_hours = settings.RERUN_FRESHNESS_HOURS
    now = datetime.utcnow()

    # Overlay rerun predictions where fresh
    for pred in predictions:
        match_id = pred.get("match_id")
        status = pred.get("status")

        if status != "NS" or match_id not in rerun_preds:
            if status == "NS":
                stats["live_fallback"] += 1
            continue

        db_pred = rerun_preds[match_id]
        pred_created_at = db_pred[5]  # created_at
        match_date = match_dates.get(match_id)

        # Freshness check: prediction must be within RERUN_FRESHNESS_HOURS of now
        # OR within RERUN_FRESHNESS_HOURS before match kickoff
        is_fresh = False
        hours_since_pred = (now - pred_created_at).total_seconds() / 3600

        if hours_since_pred <= freshness_hours:
            is_fresh = True
        elif match_date:
            # Also fresh if match is soon and pred was made recently enough
            hours_to_kickoff = (match_date - now).total_seconds() / 3600
            if hours_to_kickoff > 0 and hours_since_pred <= freshness_hours * 2:
                is_fresh = True

        if is_fresh:
            # Overlay DB prediction
            pred["probabilities"] = {
                "home": float(db_pred[2]),
                "draw": float(db_pred[3]),
                "away": float(db_pred[4]),
            }
            # Recalculate fair odds from new probabilities
            pred["fair_odds"] = {
                "home": round(1.0 / db_pred[2], 2) if db_pred[2] > 0 else None,
                "draw": round(1.0 / db_pred[3], 2) if db_pred[3] > 0 else None,
                "away": round(1.0 / db_pred[4], 2) if db_pred[4] > 0 else None,
            }
            # Mark as served from rerun
            pred["served_from_rerun"] = True
            pred["rerun_model_version"] = db_pred[1]
            stats["db_hits"] += 1
        else:
            stats["db_stale"] += 1
            stats["live_fallback"] += 1

    # Update global counters
    _rerun_serving_stats["db_hits"] += stats["db_hits"]
    _rerun_serving_stats["db_stale"] += stats["db_stale"]
    _rerun_serving_stats["live_fallback"] += stats["live_fallback"]
    _rerun_serving_stats["total_ns_served"] += stats["total_ns"]

    # Record Prometheus metrics
    try:
        from app.telemetry.metrics import record_rerun_serving_batch
        record_rerun_serving_batch(
            db_hits=stats["db_hits"],
            db_stale=stats["db_stale"],
            live_fallback=stats["live_fallback"],
            total_ns=stats["total_ns"],
        )
    except Exception as e:
        logger.warning(f"Failed to record rerun serving metrics: {e}")

    return predictions, stats


async def _overlay_frozen_predictions(
    session: AsyncSession,
    predictions: list[dict],
) -> list[dict]:
    """
    Overlay frozen prediction data for finished matches.

    For matches that have frozen predictions (is_frozen=True), we replace
    the dynamically calculated values with the frozen values. This ensures
    users see the ORIGINAL prediction they saw before the match, not a
    recalculated one after model retraining.

    Frozen data includes:
    - frozen_odds_home/draw/away: Bookmaker odds at freeze time
    - frozen_ev_home/draw/away: EV calculations at freeze time
    - frozen_confidence_tier: Confidence tier at freeze time
    - frozen_value_bets: Value bets at freeze time
    """
    if not predictions:
        return predictions

    # Get match IDs to look up frozen predictions
    match_ids = [p.get("match_id") for p in predictions if p.get("match_id")]
    if not match_ids:
        return predictions

    # Query frozen predictions for these matches
    result = await session.execute(
        select(Prediction)
        .where(
            Prediction.match_id.in_(match_ids),
            Prediction.is_frozen == True,  # noqa: E712
        )
    )
    frozen_preds = {p.match_id: p for p in result.scalars().all()}

    if not frozen_preds:
        return predictions

    # Overlay frozen data for finished matches
    for pred in predictions:
        match_id = pred.get("match_id")
        status = pred.get("status")

        # Only overlay for finished matches with frozen predictions
        if match_id in frozen_preds and status not in ("NS", None):
            frozen = frozen_preds[match_id]

            # Overlay frozen odds if available (from when prediction was frozen)
            if frozen.frozen_odds_home is not None:
                pred["market_odds"] = {
                    "home": frozen.frozen_odds_home,
                    "draw": frozen.frozen_odds_draw,
                    "away": frozen.frozen_odds_away,
                    "is_frozen": True,  # Flag to indicate these are frozen odds
                }

            # Overlay frozen confidence tier
            if frozen.frozen_confidence_tier:
                # Keep original for reference but use frozen as main
                pred["original_tier"] = pred.get("confidence_tier")
                pred["confidence_tier"] = frozen.frozen_confidence_tier

            # Overlay frozen value bets
            if frozen.frozen_value_bets:
                pred["value_bets"] = frozen.frozen_value_bets
                pred["has_value_bet"] = len(frozen.frozen_value_bets) > 0
                if frozen.frozen_value_bets:
                    # Find best value bet (highest EV) - support both old "ev" and new "expected_value" keys
                    best = max(frozen.frozen_value_bets, key=lambda x: x.get("expected_value", x.get("ev", 0)))
                    pred["best_value_bet"] = best

            # Add frozen metadata
            pred["is_frozen"] = True
            pred["frozen_at"] = frozen.frozen_at.isoformat() if frozen.frozen_at else None

            # Add frozen EV values for reference
            if frozen.frozen_ev_home is not None:
                pred["frozen_ev"] = {
                    "home": frozen.frozen_ev_home,
                    "draw": frozen.frozen_ev_draw,
                    "away": frozen.frozen_ev_away,
                }

    return predictions


async def _apply_team_overrides(
    session: AsyncSession,
    predictions: list[dict],
) -> list[dict]:
    """
    Apply team identity overrides to predictions.

    For rebranded teams (e.g., La Equidad → Internacional de Bogotá),
    replaces display names/logos based on match date and effective_from.

    Args:
        session: Database session.
        predictions: List of prediction dicts with team info.

    Returns:
        Predictions with overridden team names/logos where applicable.
    """
    if not predictions:
        return predictions

    # Collect all unique external team IDs
    external_ids = set()
    for pred in predictions:
        home_ext = pred.get("home_team_external_id")
        away_ext = pred.get("away_team_external_id")
        if home_ext:
            external_ids.add(home_ext)
        if away_ext:
            external_ids.add(away_ext)

    if not external_ids:
        return predictions

    # Batch load all overrides (single query)
    overrides = await preload_team_overrides(session, list(external_ids))

    if not overrides:
        return predictions

    # Apply overrides to each prediction
    override_count = 0
    for pred in predictions:
        match_date = pred.get("date")
        if not match_date:
            continue

        # Convert to datetime if string
        if isinstance(match_date, str):
            match_date = datetime.fromisoformat(match_date.replace("Z", "+00:00"))

        # Home team override
        home_ext = pred.get("home_team_external_id")
        if home_ext:
            home_display = resolve_team_display(
                overrides,
                home_ext,
                match_date,
                pred.get("home_team", "Unknown"),
                pred.get("home_team_logo"),
            )
            if home_display.is_override:
                pred["home_team"] = home_display.name
                if home_display.logo_url:
                    pred["home_team_logo"] = home_display.logo_url
                override_count += 1

        # Away team override
        away_ext = pred.get("away_team_external_id")
        if away_ext:
            away_display = resolve_team_display(
                overrides,
                away_ext,
                match_date,
                pred.get("away_team", "Unknown"),
                pred.get("away_team_logo"),
            )
            if away_display.is_override:
                pred["away_team"] = away_display.name
                if away_display.logo_url:
                    pred["away_team_logo"] = away_display.logo_url
                override_count += 1

    if override_count > 0:
        logger.info(f"Applied {override_count} team identity overrides to predictions")

    return predictions


@app.get("/predictions/match/{match_id}")
async def get_match_prediction(
    match_id: int,
    session: AsyncSession = Depends(get_async_session),
):
    """Get prediction for a specific match."""
    if not ml_engine.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Train a model first.",
        )

    # Get match
    result = await session.execute(select(Match).where(Match.id == match_id))
    match = result.scalar_one_or_none()

    if not match:
        raise HTTPException(status_code=404, detail="Match not found")

    # Get features
    feature_engineer = FeatureEngineer(session=session)
    features = await feature_engineer.get_match_features(match)

    # Get team names
    home_team = await session.get(Team, match.home_team_id)
    away_team = await session.get(Team, match.away_team_id)

    features["home_team_name"] = home_team.name if home_team else "Unknown"
    features["away_team_name"] = away_team.name if away_team else "Unknown"
    features["odds_home"] = match.odds_home
    features["odds_draw"] = match.odds_draw
    features["odds_away"] = match.odds_away

    import pandas as pd

    df = pd.DataFrame([features])
    predictions = ml_engine.predict(df)

    return predictions[0]


@app.get("/teams")
async def list_teams(
    team_type: Optional[str] = None,
    limit: int = 100,
    session: AsyncSession = Depends(get_async_session),
):
    """List teams in the database."""
    query = select(Team)
    if team_type:
        query = query.where(Team.team_type == team_type)
    query = query.limit(limit)

    result = await session.execute(query)
    teams = result.scalars().all()

    return [
        {
            "id": t.id,
            "external_id": t.external_id,
            "name": t.name,
            "country": t.country,
            "team_type": t.team_type,
            "logo_url": t.logo_url,
        }
        for t in teams
    ]


@app.get("/matches")
async def list_matches(
    league_id: Optional[int] = None,
    status: Optional[str] = None,
    limit: int = 50,
    session: AsyncSession = Depends(get_async_session),
):
    """List matches in the database with eager loading to avoid N+1 queries."""
    from sqlalchemy.orm import selectinload

    query = (
        select(Match)
        .options(
            selectinload(Match.home_team),
            selectinload(Match.away_team),
        )
        .order_by(Match.date.desc())
    )

    if league_id:
        query = query.where(Match.league_id == league_id)
    if status:
        query = query.where(Match.status == status)

    query = query.limit(limit)

    result = await session.execute(query)
    matches = result.scalars().all()

    # Build response using eager-loaded relationships
    return [
        {
            "id": m.id,
            "external_id": m.external_id,
            "date": m.date,
            "league_id": m.league_id,
            "home_team": m.home_team.name if m.home_team else "Unknown",
            "away_team": m.away_team.name if m.away_team else "Unknown",
            "home_goals": m.home_goals,
            "away_goals": m.away_goals,
            "status": m.status,
            "match_type": m.match_type,
        }
        for m in matches
    ]


@app.get("/competitions")
async def list_competitions():
    """List available competitions."""
    return [
        {
            "league_id": comp.league_id,
            "name": comp.name,
            "match_type": comp.match_type,
            "priority": comp.priority.value,
            "match_weight": comp.match_weight,
        }
        for comp in COMPETITIONS.values()
    ]


@app.get("/model/info")
async def model_info():
    """Get information about the current model."""
    if not ml_engine.is_loaded:
        return {
            "loaded": False,
            "message": "No model loaded. Train one with POST /model/train",
        }

    return {
        "loaded": True,
        "version": ml_engine.model_version,
        "features": ml_engine.FEATURE_COLUMNS,
    }


@app.get("/model/shadow-report")
@limiter.limit("30/minute")
async def get_shadow_report_endpoint(
    request: Request,
    session: AsyncSession = Depends(get_async_session),
    _: bool = Depends(verify_api_key_or_ops_session),
):
    """
    Get shadow model A/B comparison report.

    Returns accuracy and Brier score comparison between baseline and shadow (two-stage) model.
    Includes per-outcome breakdown and GO/NO-GO recommendation.
    Requires API key authentication.
    """
    from app.ml.shadow import is_shadow_enabled, get_shadow_report

    if not is_shadow_enabled():
        return {
            "status": "disabled",
            "message": "Shadow mode not enabled. Set MODEL_SHADOW_ARCHITECTURE=two_stage to enable.",
        }

    report = await get_shadow_report(session)
    return report


@app.get("/model/sensor-report")
@limiter.limit("30/minute")
async def get_sensor_report_endpoint(
    request: Request,
    session: AsyncSession = Depends(get_async_session),
    _: bool = Depends(verify_api_key_or_ops_session),
):
    """
    Get Sensor B (LogReg L2) calibration diagnostics report.

    Returns Model A vs Model B comparison with Brier scores, accuracy,
    signal score, and window analysis. INTERNAL USE ONLY - does not affect production.
    Requires API key authentication.
    """
    from app.ml.sensor import get_sensor_report
    from app.config import get_settings

    sensor_settings = get_settings()
    if not sensor_settings.SENSOR_ENABLED:
        return {
            "status": "disabled",
            "message": "Sensor B not enabled. Set SENSOR_ENABLED=true to enable.",
        }

    report = await get_sensor_report(session)
    return report


@app.post("/odds/refresh")
async def refresh_odds(
    session: AsyncSession = Depends(get_async_session),
):
    """
    Refresh odds for all upcoming matches.

    Fetches latest pre-match odds from API-Football for matches with status 'NS'.
    Prioritizes Bet365, Pinnacle for reliable odds.
    """
    # Get all upcoming matches
    query = select(Match).where(Match.status == "NS")
    result = await session.execute(query)
    matches = result.scalars().all()

    if not matches:
        return {"message": "No upcoming matches found", "updated": 0}

    provider = APIFootballProvider()
    updated_count = 0
    errors = []

    try:
        for match in matches:
            try:
                odds = await provider.get_odds(match.external_id)
                if odds:
                    match.odds_home = odds.get("odds_home")
                    match.odds_draw = odds.get("odds_draw")
                    match.odds_away = odds.get("odds_away")
                    updated_count += 1
                    logger.info(f"Updated odds for match {match.id}: H={match.odds_home}, D={match.odds_draw}, A={match.odds_away}")
            except Exception as e:
                errors.append({"match_id": match.id, "error": str(e)})
                logger.error(f"Error fetching odds for match {match.id}: {e}")

        await session.commit()

    finally:
        await provider.close()

    return {
        "message": f"Odds refresh complete",
        "total_matches": len(matches),
        "updated": updated_count,
        "errors": errors if errors else None,
    }


@app.get("/teams/{team_id}/history")
async def get_team_history(
    team_id: int,
    limit: int = 5,
    session: AsyncSession = Depends(get_async_session),
):
    """
    Get recent match history for a team.

    Returns the last N matches played by the team with results.
    Uses eager loading to avoid N+1 queries.
    """
    from sqlalchemy import or_
    from sqlalchemy.orm import selectinload

    # Get team info
    team = await session.get(Team, team_id)
    if not team:
        raise HTTPException(status_code=404, detail="Team not found")

    # Get last matches with eager loading of both teams (avoids N+1 queries)
    query = (
        select(Match)
        .where(
            or_(
                Match.home_team_id == team_id,
                Match.away_team_id == team_id,
            ),
            Match.status == "FT",  # Only finished matches
        )
        .options(
            selectinload(Match.home_team),
            selectinload(Match.away_team),
        )
        .order_by(Match.date.desc())
        .limit(limit)
    )

    result = await session.execute(query)
    matches = result.scalars().all()

    history = []
    for match in matches:
        # Get opponent from eager-loaded relationship
        if match.home_team_id == team_id:
            opponent = match.away_team
            team_goals = match.home_goals
            opponent_goals = match.away_goals
            is_home = True
        else:
            opponent = match.home_team
            team_goals = match.away_goals
            opponent_goals = match.home_goals
            is_home = False

        # Determine result
        if team_goals > opponent_goals:
            result_str = "W"
        elif team_goals < opponent_goals:
            result_str = "L"
        else:
            result_str = "D"

        history.append({
            "match_id": match.id,
            "date": match.date.isoformat() if match.date else None,
            "opponent": opponent.name if opponent else "Unknown",
            "opponent_logo": opponent.logo_url if opponent else None,
            "is_home": is_home,
            "team_goals": team_goals,
            "opponent_goals": opponent_goals,
            "result": result_str,
            "league_id": match.league_id,
        })

    return {
        "team_id": team_id,
        "team_name": team.name,
        "team_logo": team.logo_url,
        "matches": history,
    }


@app.get("/matches/{match_id}/details")
async def get_match_details(
    match_id: int,
    session: AsyncSession = Depends(get_async_session),
):
    """
    Get full match details including both teams' recent history and standings.

    Returns match info, prediction, standings positions, and last 5 matches for each team.
    """
    import time
    import asyncio
    _t_start = time.time()
    _timings = {}

    # Get match
    _t0 = time.time()
    match = await session.get(Match, match_id)
    _timings["get_match"] = int((time.time() - _t0) * 1000)
    if not match:
        raise HTTPException(status_code=404, detail="Match not found")

    # Get teams (parallel)
    _t0 = time.time()
    home_team, away_team = await asyncio.gather(
        session.get(Team, match.home_team_id),
        session.get(Team, match.away_team_id),
    )
    _timings["get_teams"] = int((time.time() - _t0) * 1000)

    # Determine season for standings lookup
    current_date = match.date or datetime.now()
    season = _season_for_league(match.league_id, current_date)

    # NON-BLOCKING standings: L1 cache -> DB -> skip (never call external API in hot path)
    # This ensures endpoint always responds <400ms regardless of league
    _t0 = time.time()
    standings = None
    standings_status = "skipped"  # skipped | cache_hit | db_hit | miss
    standings_source = None
    if match.league_id:
        # L1: memory cache (check truthiness - empty list means no data)
        standings = _get_cached_standings(match.league_id, season)
        if standings:
            standings_status = "cache_hit"
            standings_source = "cache"
            _incr("standings_source_cache")
        else:
            # L2: database
            standings = await _get_standings_from_db(session, match.league_id, season)
            if standings:
                standings_status = "db_hit"
                standings_source = "db"
                _incr("standings_source_db")
                # Populate L1 cache for next request
                _set_cached_standings(match.league_id, season, standings)
            else:
                # L3: Generate placeholder standings (zero stats, alphabetical order)
                standings = await _generate_placeholder_standings(session, match.league_id, season)
                if standings:
                    standings_status = "placeholder"
                    standings_source = "placeholder"
                    _incr("standings_source_placeholder")
                    _set_cached_standings(match.league_id, season, standings)
                else:
                    standings_status = "miss"
                    _incr("standings_source_miss")
    _timings["get_standings"] = int((time.time() - _t0) * 1000)
    _timings["standings_status"] = standings_status
    if standings_source:
        _timings["standings_source"] = standings_source

    # Get history for both teams (parallel)
    _t0 = time.time()
    home_history, away_history = await asyncio.gather(
        get_team_history(match.home_team_id, limit=5, session=session),
        get_team_history(match.away_team_id, limit=5, session=session),
    )
    _timings["get_history"] = int((time.time() - _t0) * 1000)

    # Extract standings positions (only if we have cached data)
    home_position = None
    away_position = None
    home_league_points = None
    away_league_points = None

    # Only use standings for club teams when cache hit
    if home_team and home_team.team_type == "club" and standings:
        try:
            for standing in standings:
                if home_team and standing.get("team_id") == home_team.external_id:
                    home_position = standing.get("position")
                    home_league_points = standing.get("points")
                if away_team and standing.get("team_id") == away_team.external_id:
                    away_position = standing.get("position")
                    away_league_points = standing.get("points")
        except Exception as e:
            logger.warning(f"Could not process standings: {e}")

    # Get prediction if model is loaded and match not played
    prediction = None
    if ml_engine.is_loaded and match.status == "NS":
        try:
            _t0 = time.time()
            feature_engineer = FeatureEngineer(session=session)
            features = await feature_engineer.get_match_features(match)
            _timings["get_features"] = int((time.time() - _t0) * 1000)

            features["home_team_name"] = home_team.name if home_team else "Unknown"
            features["away_team_name"] = away_team.name if away_team else "Unknown"

            _t0 = time.time()
            import pandas as pd
            df = pd.DataFrame([features])
            predictions = ml_engine.predict(df)
            prediction = predictions[0] if predictions else None
            _timings["ml_predict"] = int((time.time() - _t0) * 1000)
        except Exception as e:
            logger.error(f"Error getting prediction: {e}")

    _timings["total"] = int((time.time() - _t_start) * 1000)
    logger.info(f"[PERF] match_details match_id={match_id} timings={_timings}")

    # Resolve team display names/logos (handles rebranding like La Equidad → Internacional de Bogotá)
    home_name = home_team.name if home_team else "Unknown"
    home_logo = home_team.logo_url if home_team else None
    away_name = away_team.name if away_team else "Unknown"
    away_logo = away_team.logo_url if away_team else None

    # Apply team overrides if match date is after effective_from
    if match.date and (home_team or away_team):
        external_ids = []
        if home_team and home_team.external_id:
            external_ids.append(home_team.external_id)
        if away_team and away_team.external_id:
            external_ids.append(away_team.external_id)

        if external_ids:
            overrides = await preload_team_overrides(session, external_ids)
            if overrides:
                if home_team and home_team.external_id:
                    home_display = resolve_team_display(
                        overrides, home_team.external_id, match.date, home_name, home_logo
                    )
                    if home_display.is_override:
                        home_name = home_display.name
                        home_logo = home_display.logo_url or home_logo

                if away_team and away_team.external_id:
                    away_display = resolve_team_display(
                        overrides, away_team.external_id, match.date, away_name, away_logo
                    )
                    if away_display.is_override:
                        away_name = away_display.name
                        away_logo = away_display.logo_url or away_logo

    return {
        "match": {
            "id": match.id,
            "date": match.date.isoformat() if match.date else None,
            "league_id": match.league_id,
            "status": match.status,
            "home_goals": match.home_goals,
            "away_goals": match.away_goals,
            "venue": {
                "name": match.venue_name,
                "city": match.venue_city,
            } if match.venue_name else None,
        },
        "home_team": {
            "id": home_team.id if home_team else None,
            "name": home_name,
            "logo": home_logo,
            "history": home_history["matches"],
            "position": home_position,
            "league_points": home_league_points,
        },
        "away_team": {
            "id": away_team.id if away_team else None,
            "name": away_name,
            "logo": away_logo,
            "history": away_history["matches"],
            "position": away_position,
            "league_points": away_league_points,
        },
        "prediction": prediction,
        "standings_status": standings_status,  # hit | miss | skipped
    }


@app.get("/matches/{match_id}/insights")
async def get_match_insights(
    match_id: int,
    session: AsyncSession = Depends(get_async_session),
):
    """
    Get narrative insights for a finished match.

    Returns human-readable explanations of why the prediction succeeded or failed,
    including analysis of efficiency, clinical finishing, goalkeeper heroics, etc.

    Only available for matches that have been audited (finished + processed).
    """
    # Get match
    match = await session.get(Match, match_id)
    if not match:
        raise HTTPException(status_code=404, detail="Match not found")

    # Only for finished matches
    if match.status not in ("FT", "AET", "PEN"):
        raise HTTPException(
            status_code=400,
            detail=f"Insights only available for finished matches. Status: {match.status}"
        )

    # Get prediction outcome and audit for this match (canonical path)
    result = await session.execute(
        select(PredictionOutcome, PostMatchAudit)
        .join(PostMatchAudit, PredictionOutcome.id == PostMatchAudit.outcome_id)
        .where(PredictionOutcome.match_id == match_id)
    )
    row = result.first()

    if not row:
        # Fallback: generate narrative insights on-demand (non-canonical) so iOS can display something
        # even if the daily audit has not run yet or failed.
        #
        # Important: Keep response shape identical to MatchInsightsResponse in iOS (no optionals).
        from app.audit.service import PostMatchAuditService

        # Prefer a frozen prediction if available; otherwise use latest prediction.
        pred = None
        pred_res = await session.execute(
            select(Prediction)
            .where(Prediction.match_id == match_id)
            .where(Prediction.is_frozen == True)  # noqa: E712
            .order_by(Prediction.frozen_at.desc().nullslast(), Prediction.created_at.desc())
            .limit(1)
        )
        pred = pred_res.scalar_one_or_none()

        if pred is None:
            pred_res = await session.execute(
                select(Prediction)
                .where(Prediction.match_id == match_id)
                .order_by(Prediction.created_at.desc())
                .limit(1)
            )
            pred = pred_res.scalar_one_or_none()

        # If no saved prediction exists, compute one from features (best-effort).
        if pred is None:
            try:
                # Load team names for context
                home_team = await session.get(Team, match.home_team_id)
                away_team = await session.get(Team, match.away_team_id)
                feature_engineer = FeatureEngineer(session=session)
                features = await feature_engineer.get_match_features(match)
                features["home_team_name"] = home_team.name if home_team else "Local"
                features["away_team_name"] = away_team.name if away_team else "Visitante"

                import pandas as pd

                df = pd.DataFrame([features])
                preds = ml_engine.predict(df)
                p0 = preds[0] if preds else None
                probs = (p0 or {}).get("probabilities") or {}
                hp = float(probs.get("home") or 0.0)
                dp = float(probs.get("draw") or 0.0)
                ap = float(probs.get("away") or 0.0)
                pred = Prediction(
                    match_id=match_id,
                    model_version=ml_engine.model_version,
                    home_prob=hp,
                    draw_prob=dp,
                    away_prob=ap,
                )
            except Exception:
                # No prediction available; return empty insights but keep schema stable.
                pred = Prediction(
                    match_id=match_id,
                    model_version=ml_engine.model_version,
                    home_prob=0.0,
                    draw_prob=0.0,
                    away_prob=0.0,
                )

        service = PostMatchAuditService(session)
        try:
            predicted_result, confidence = service._get_predicted_result(pred)
        except Exception:
            predicted_result, confidence = ("draw", 0.0)

        actual_result = "draw"
        if match.home_goals is not None and match.away_goals is not None:
            if match.home_goals > match.away_goals:
                actual_result = "home"
            elif match.home_goals < match.away_goals:
                actual_result = "away"

        prediction_correct = predicted_result == actual_result

        home_team = await session.get(Team, match.home_team_id)
        away_team = await session.get(Team, match.away_team_id)

        narrative_result = service.generate_narrative_insights(
            prediction=pred,
            actual_result=actual_result,
            home_goals=match.home_goals or 0,
            away_goals=match.away_goals or 0,
            stats=match.stats or {},
            home_team_name=home_team.name if home_team else "Local",
            away_team_name=away_team.name if away_team else "Visitante",
            home_position=None,
            away_position=None,
        )

        await service.close()

        fallback_response = {
            "match_id": match_id,
            "prediction_correct": prediction_correct,
            "predicted_result": predicted_result,
            "actual_result": actual_result,
            "confidence": confidence,
            "deviation_type": "pending_audit",
            "insights": narrative_result.get("insights") or [],
            "momentum_analysis": narrative_result.get("momentum_analysis"),
            # No LLM narrative available - indicates match had no pre-match prediction
            "llm_narrative_status": "no_prediction",
        }
        # Include match stats for UI stats table
        if match.stats:
            fallback_response["match_stats"] = match.stats
        if match.events:
            fallback_response["match_events"] = match.events
        return fallback_response

    outcome, audit = row

    response = {
        "match_id": match_id,
        "prediction_correct": outcome.prediction_correct,
        "predicted_result": outcome.predicted_result,
        "actual_result": outcome.actual_result,
        "confidence": outcome.confidence,
        "deviation_type": audit.deviation_type,
        "insights": audit.narrative_insights or [],
        "momentum_analysis": audit.momentum_analysis,
    }

    # Include LLM narrative if available
    if audit.llm_narrative_status == "ok" and audit.llm_narrative_json:
        response["llm_narrative"] = audit.llm_narrative_json
        response["llm_narrative_status"] = "ok"
    elif audit.llm_narrative_status:
        response["llm_narrative_status"] = audit.llm_narrative_status

    # Include match stats for UI stats table (renders independently of narrative)
    if match.stats:
        response["match_stats"] = match.stats

    # Include events for UI
    if match.events:
        response["match_events"] = match.events

    return response


@app.get("/matches/{match_id}/timeline")
async def get_match_timeline(
    match_id: int,
    session: AsyncSession = Depends(get_async_session),
):
    """
    Get timeline data for a finished match.

    Returns goal events with minutes, and compares against our prediction
    to show when the prediction was "in line" vs "out of line" with the score.

    Only available for finished matches (FT, AET, PEN) with a saved prediction.
    """
    # Get match
    match = await session.get(Match, match_id)
    if not match:
        raise HTTPException(status_code=404, detail="Match not found")

    # Only for finished matches
    if match.status not in ("FT", "AET", "PEN"):
        raise HTTPException(
            status_code=400,
            detail=f"Timeline only available for finished matches. Status: {match.status}"
        )

    # Get saved prediction for this match
    result = await session.execute(
        select(Prediction).where(Prediction.match_id == match_id).limit(1)
    )
    prediction = result.scalar_one_or_none()

    if not prediction:
        raise HTTPException(
            status_code=404,
            detail="No prediction saved for this match"
        )

    # Determine what we predicted
    predicted_outcome = "home"
    if prediction.away_prob > prediction.home_prob and prediction.away_prob > prediction.draw_prob:
        predicted_outcome = "away"
    elif prediction.draw_prob > prediction.home_prob and prediction.draw_prob > prediction.away_prob:
        predicted_outcome = "draw"

    # Get goal events - prefer DB, fallback to API
    import time
    _t0 = time.time()
    events = []
    events_source = "none"

    # Try DB first (for finished matches, events should be cached)
    if match.events and len(match.events) > 0:
        events = match.events
        events_source = "db"
        _incr("timeline_source_db")
        logger.info(f"[PERF] timeline match_id={match_id} events_source=db count={len(events)} time_ms={int((time.time() - _t0) * 1000)}")
    else:
        # Fallback to API (and persist for next time)
        _incr("timeline_source_api_fallback")
        logger.info(f"[PERF] timeline match_id={match_id} events_source=api_fallback (db events empty)")
        provider = APIFootballProvider()
        try:
            events = await provider.get_fixture_events(match.external_id)
            events_source = "api"
            # Persist to DB for future requests (best-effort)
            if events:
                try:
                    match.events = events
                    await session.commit()
                    logger.info(f"[PERF] timeline match_id={match_id} persisted {len(events)} events to DB")
                except Exception as persist_err:
                    logger.warning(f"[PERF] timeline match_id={match_id} failed to persist events: {persist_err}")
        finally:
            await provider.close()
        logger.info(f"[PERF] timeline match_id={match_id} events_source=api count={len(events)} time_ms={int((time.time() - _t0) * 1000)}")

    # Filter only goals
    goals = [
        e for e in events
        if e.get("type") == "Goal"
    ]

    # Sort by minute
    goals.sort(key=lambda g: (g.get("minute") or 0, g.get("extra_minute") or 0))

    # Get team IDs
    home_team = await session.get(Team, match.home_team_id)
    away_team = await session.get(Team, match.away_team_id)
    home_external_id = home_team.external_id if home_team else None
    away_external_id = away_team.external_id if away_team else None

    # Build timeline segments
    # Each segment: {start_minute, end_minute, home_score, away_score, status}
    # status: "correct" (prediction in line), "neutral" (draw when we predicted win), "wrong" (losing)
    segments = []
    current_home = 0
    current_away = 0
    last_minute = 0

    # Calculate total match duration based on goals (including added time)
    # Default to 90, but extend if there are goals in added time
    total_minutes = 90
    if goals:
        # Consider both base minute and extra time (e.g., 90+3 = 93 effective)
        max_effective = max(
            (g.get("minute") or 0) + (g.get("extra_minute") or 0)
            for g in goals
        )
        # Use the maximum between 90 and the last goal's effective minute
        total_minutes = max(90, max_effective)

    for goal in goals:
        minute = goal.get("minute") or 0
        extra = goal.get("extra_minute") or 0
        effective_minute = minute + (extra * 0.1)  # For sorting 90+1, 90+2, etc.

        # Add segment before this goal
        if minute > last_minute:
            status = _calculate_segment_status(
                current_home, current_away, predicted_outcome
            )
            segments.append({
                "start_minute": last_minute,
                "end_minute": minute,
                "home_goals": current_home,
                "away_goals": current_away,
                "status": status,
            })

        # Determine which team scored (prefer team_id, fallback to team_name match)
        is_home_team = False
        is_away_team = False
        used_legacy_fallback = False

        if goal.get("team_id"):
            is_home_team = goal.get("team_id") == home_external_id
            is_away_team = goal.get("team_id") == away_external_id
        else:
            # Legacy fallback: match by team name
            used_legacy_fallback = True
            goal_team_name = goal.get("team_name") or goal.get("team")
            if goal_team_name:
                is_home_team = goal_team_name == (home_team.name if home_team else None)
                is_away_team = goal_team_name == (away_team.name if away_team else None)
            logger.info(f"[TIMELINE] match_id={match_id} legacy_fallback goal_team_name={goal_team_name} matched={'home' if is_home_team else 'away' if is_away_team else 'none'}")

        # Update score
        if is_home_team:
            if goal.get("detail") == "Own Goal":
                current_away += 1
            else:
                current_home += 1
        elif is_away_team:
            if goal.get("detail") == "Own Goal":
                current_home += 1
            else:
                current_away += 1

        last_minute = minute

    # Add final segment
    if last_minute < total_minutes:
        status = _calculate_segment_status(
            current_home, current_away, predicted_outcome
        )
        segments.append({
            "start_minute": last_minute,
            "end_minute": total_minutes,
            "home_goals": current_home,
            "away_goals": current_away,
            "status": status,
        })

    # Calculate time in correct prediction
    correct_minutes = sum(
        (s["end_minute"] - s["start_minute"]) for s in segments if s["status"] == "correct"
    )
    total_match_minutes = total_minutes
    correct_percentage = (correct_minutes / total_match_minutes) * 100 if total_match_minutes > 0 else 0

    # Determine final result
    final_result = "draw"
    if match.home_goals > match.away_goals:
        final_result = "home"
    elif match.away_goals > match.home_goals:
        final_result = "away"

    return {
        "match_id": match_id,
        "status": match.status,
        "final_score": {
            "home": match.home_goals,
            "away": match.away_goals,
        },
        "prediction": {
            "outcome": predicted_outcome,
            "home_prob": round(prediction.home_prob, 4),
            "draw_prob": round(prediction.draw_prob, 4),
            "away_prob": round(prediction.away_prob, 4),
            "correct": predicted_outcome == final_result,
        },
        "total_minutes": total_minutes,
        "goals": [
            {
                "minute": g.get("minute"),
                "extra_minute": g.get("extra_minute"),
                # Determine team: prefer team_id match, fallback to team_name match for legacy events
                "team": (
                    "home" if g.get("team_id") == home_external_id
                    else "away" if g.get("team_id") == away_external_id
                    else "home" if g.get("team_name") == (home_team.name if home_team else None) or g.get("team") == (home_team.name if home_team else None)
                    else "away"
                ),
                "team_name": g.get("team_name") or g.get("team"),  # Support legacy "team" field
                "player": g.get("player_name") or g.get("player"),  # Support legacy "player" field
                "is_own_goal": g.get("detail") == "Own Goal",
                "is_penalty": g.get("detail") == "Penalty",
            }
            for g in goals
        ],
        "segments": segments,
        "summary": {
            "correct_minutes": round(correct_minutes, 1),
            "correct_percentage": round(correct_percentage, 1),
        },
        "_meta": {
            "events_source": events_source,
            "events_count": len(events),
        },
    }


def _calculate_segment_status(home_score: int, away_score: int, predicted: str) -> str:
    """
    Calculate segment status based on current score vs prediction.

    Returns:
        "correct": Score aligns with prediction
        "neutral": Draw when we predicted a win (gray area)
        "wrong": Losing team is the one we predicted to win
    """
    if home_score == away_score:
        # It's a draw
        if predicted == "draw":
            return "correct"
        else:
            return "neutral"  # We predicted a win but it's tied

    if home_score > away_score:
        # Home is winning
        if predicted == "home":
            return "correct"
        elif predicted == "away":
            return "wrong"
        else:  # predicted draw
            return "neutral"
    else:
        # Away is winning
        if predicted == "away":
            return "correct"
        elif predicted == "home":
            return "wrong"
        else:  # predicted draw
            return "neutral"


@app.get("/matches/{match_id}/odds-history")
async def get_match_odds_history(
    match_id: int,
    session: AsyncSession = Depends(get_async_session),
):
    """
    Get odds history for a match showing how odds changed over time.

    Returns all recorded odds snapshots for the match, ordered by time.
    Useful for:
    - Analyzing line movements before the match
    - Seeing opening vs closing odds
    - Detecting sharp money movements
    """
    # Get match
    match = await session.get(Match, match_id)
    if not match:
        raise HTTPException(status_code=404, detail="Match not found")

    # Get team names
    home_team = await session.get(Team, match.home_team_id)
    away_team = await session.get(Team, match.away_team_id)

    # Get odds history
    result = await session.execute(
        select(OddsHistory)
        .where(OddsHistory.match_id == match_id)
        .order_by(OddsHistory.recorded_at.asc())
    )
    history = result.scalars().all()

    # Calculate line movement if we have opening and current odds
    movement = None
    if len(history) >= 2:
        opening = history[0]
        current = history[-1]
        if opening.odds_home and current.odds_home:
            movement = {
                "home_change": round(current.odds_home - opening.odds_home, 2),
                "draw_change": round((current.odds_draw or 0) - (opening.odds_draw or 0), 2),
                "away_change": round((current.odds_away or 0) - (opening.odds_away or 0), 2),
                "home_pct": round((current.odds_home - opening.odds_home) / opening.odds_home * 100, 1),
                "draw_pct": round(((current.odds_draw or 0) - (opening.odds_draw or 0)) / (opening.odds_draw or 1) * 100, 1) if opening.odds_draw else None,
                "away_pct": round(((current.odds_away or 0) - (opening.odds_away or 0)) / (opening.odds_away or 1) * 100, 1) if opening.odds_away else None,
            }

    return {
        "match_id": match_id,
        "home_team": home_team.name if home_team else "Unknown",
        "away_team": away_team.name if away_team else "Unknown",
        "match_date": match.date.isoformat() if match.date else None,
        "status": match.status,
        "current_odds": {
            "home": match.odds_home,
            "draw": match.odds_draw,
            "away": match.odds_away,
            "recorded_at": match.odds_recorded_at.isoformat() if match.odds_recorded_at else None,
        },
        "history": [
            {
                "recorded_at": h.recorded_at.isoformat(),
                "odds_home": h.odds_home,
                "odds_draw": h.odds_draw,
                "odds_away": h.odds_away,
                "implied_home": round(h.implied_home, 4) if h.implied_home else None,
                "implied_draw": round(h.implied_draw, 4) if h.implied_draw else None,
                "implied_away": round(h.implied_away, 4) if h.implied_away else None,
                "overround": round(h.overround, 4) if h.overround else None,
                "is_opening": h.is_opening,
                "is_closing": h.is_closing,
                "source": h.source,
            }
            for h in history
        ],
        "movement": movement,
        "total_snapshots": len(history),
    }


@app.get("/standings/{league_id}")
async def get_league_standings(
    league_id: int,
    season: int = None,
    session: AsyncSession = Depends(get_async_session),
):
    """
    Get full league standings/table for a given league.

    DB-first architecture: serves from DB, falls back to provider on miss.
    Returns all teams with position, points, matches played, goals, form, etc.
    """
    _t_start = time.time()
    source = None

    try:
        # Determine season if not provided
        if season is None:
            current_date = datetime.now()
            season = _season_for_league(league_id, current_date)

        # L1: Memory cache (check truthiness - empty list means no data)
        standings = _get_cached_standings(league_id, season)
        if standings:
            source = "cache"
        else:
            # L2: Database
            standings = await _get_standings_from_db(session, league_id, season)
            if standings:
                source = "db"
                # Populate L1 cache
                _set_cached_standings(league_id, season, standings)
            else:
                # L3: Provider fallback (and persist)
                source = "api_fallback"
                provider = APIFootballProvider()
                try:
                    standings = await provider.get_standings(league_id, season)
                    if standings:
                        # Persist to DB
                        await _save_standings_to_db(session, league_id, season, standings)
                        # Populate L1 cache
                        _set_cached_standings(league_id, season, standings)
                finally:
                    await provider.close()

        # L4: Placeholder fallback - generate zero-stats standings from known teams
        if not standings:
            standings = await _generate_placeholder_standings(session, league_id, season)
            if standings:
                source = "placeholder"
                # Cache placeholder standings (shorter TTL handled by is_placeholder flag)
                _set_cached_standings(league_id, season, standings)

        if not standings:
            raise HTTPException(
                status_code=404,
                detail=f"Standings not available yet for season {season}. No teams found for this league.",
            )

        # Apply team identity overrides (e.g., La Equidad -> Internacional de Bogotá)
        from app.teams.overrides import apply_team_overrides_to_standings
        standings = await apply_team_overrides_to_standings(
            session, standings, league_id, season
        )

        elapsed_ms = int((time.time() - _t_start) * 1000)
        logger.info(f"[PERF] get_standings league_id={league_id} season={season} source={source} time_ms={elapsed_ms}")

        # Determine if standings are placeholder (all zeros)
        is_placeholder = source == "placeholder" or (
            standings and standings[0].get("is_placeholder", False)
        )

        return {
            "league_id": league_id,
            "season": season,
            "standings": standings,
            "source": source,
            "is_placeholder": is_placeholder,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching standings: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch standings")


# ============================================================================
# AUDIT ENDPOINTS - Post-match analysis and model evaluation
# ============================================================================


class AuditResponse(BaseModel):
    matches_audited: int
    correct_predictions: int
    accuracy: float
    anomalies_detected: int
    period_days: int


class AuditSummaryResponse(BaseModel):
    total_outcomes: int
    correct_predictions: int
    overall_accuracy: float
    by_tier: dict
    by_deviation_type: dict
    recent_anomalies: list


@app.post("/audit/run", response_model=AuditResponse)
async def run_audit(
    days: int = 7,
    session: AsyncSession = Depends(get_async_session),
):
    """
    Run post-match audit for completed matches.

    Analyzes predictions vs actual results for matches finished in the last N days.
    Fetches xG, events (red cards, penalties, VAR) and classifies deviations.
    """
    from app.audit import create_audit_service

    logger.info(f"Running audit for last {days} days...")

    try:
        audit_service = await create_audit_service(session)
        result = await audit_service.audit_recent_matches(days=days)
        await audit_service.close()

        return AuditResponse(**result)

    except Exception as e:
        logger.error(f"Audit failed: {e}")
        raise HTTPException(status_code=500, detail=f"Audit failed: {str(e)}")


@app.get("/audit/summary", response_model=AuditSummaryResponse)
async def get_audit_summary(
    days: Optional[int] = None,
    session: AsyncSession = Depends(get_async_session),
):
    """
    Get summary of audit results.

    Returns accuracy by confidence tier, deviation distribution, and recent anomalies.
    """
    from sqlalchemy import func

    from app.models import PredictionOutcome, PostMatchAudit

    # Base query
    query = select(PredictionOutcome)

    if days:
        from datetime import timedelta
        cutoff = datetime.utcnow() - timedelta(days=days)
        query = query.where(PredictionOutcome.audited_at >= cutoff)

    result = await session.execute(query)
    outcomes = result.scalars().all()

    if not outcomes:
        return AuditSummaryResponse(
            total_outcomes=0,
            correct_predictions=0,
            overall_accuracy=0.0,
            by_tier={},
            by_deviation_type={},
            recent_anomalies=[],
        )

    # Calculate metrics
    total = len(outcomes)
    correct = sum(1 for o in outcomes if o.prediction_correct)
    overall_accuracy = (correct / total * 100) if total > 0 else 0

    # By tier
    tiers = {}
    for tier in ["gold", "silver", "copper"]:
        tier_outcomes = [o for o in outcomes if o.confidence_tier == tier]
        tier_correct = sum(1 for o in tier_outcomes if o.prediction_correct)
        tier_total = len(tier_outcomes)
        tiers[tier] = {
            "total": tier_total,
            "correct": tier_correct,
            "accuracy": (tier_correct / tier_total * 100) if tier_total > 0 else 0,
        }

    # Get audits for deviation breakdown
    outcome_ids = [o.id for o in outcomes]
    audit_result = await session.execute(
        select(PostMatchAudit).where(PostMatchAudit.outcome_id.in_(outcome_ids))
    )
    audits = audit_result.scalars().all()

    # By deviation type
    deviation_types = {}
    for dtype in ["minimal", "expected", "anomaly"]:
        count = sum(1 for a in audits if a.deviation_type == dtype)
        deviation_types[dtype] = count

    # Recent anomalies
    anomaly_audits = [a for a in audits if a.deviation_type == "anomaly"]
    recent_anomalies = []

    for audit in anomaly_audits[:10]:  # Last 10 anomalies
        outcome = next((o for o in outcomes if o.id == audit.outcome_id), None)
        if outcome:
            # Get match info
            match_result = await session.execute(
                select(Match).where(Match.id == outcome.match_id)
            )
            match = match_result.scalar_one_or_none()

            if match:
                home_team = await session.get(Team, match.home_team_id)
                away_team = await session.get(Team, match.away_team_id)

                recent_anomalies.append({
                    "match_id": match.id,
                    "date": match.date.isoformat() if match.date else None,
                    "home_team": home_team.name if home_team else "Unknown",
                    "away_team": away_team.name if away_team else "Unknown",
                    "score": f"{outcome.actual_home_goals}-{outcome.actual_away_goals}",
                    "predicted": outcome.predicted_result,
                    "actual": outcome.actual_result,
                    "confidence": round(outcome.confidence * 100, 1),
                    "primary_factor": audit.primary_factor,
                    "xg_home": outcome.xg_home,
                    "xg_away": outcome.xg_away,
                })

    return AuditSummaryResponse(
        total_outcomes=total,
        correct_predictions=correct,
        overall_accuracy=round(overall_accuracy, 2),
        by_tier=tiers,
        by_deviation_type=deviation_types,
        recent_anomalies=recent_anomalies,
    )


# ============================================================================
# RECALIBRATION ENDPOINTS - Model auto-adjustment and team confidence
# ============================================================================


class RecalibrationStatusResponse(BaseModel):
    current_model_version: str
    baseline_brier_score: float
    current_brier_score: Optional[float]
    last_retrain_date: Optional[str]
    gold_accuracy_current: float
    gold_accuracy_threshold: float
    retrain_needed: bool
    retrain_reason: str
    teams_with_adjustments: int


class TeamAdjustmentResponse(BaseModel):
    team_id: int
    team_name: str
    confidence_multiplier: float
    total_predictions: int
    correct_predictions: int
    anomaly_count: int
    avg_deviation_score: float
    last_updated: str
    reason: Optional[str]


class ModelSnapshotResponse(BaseModel):
    id: int
    model_version: str
    model_path: str
    brier_score: float
    samples_trained: int
    is_active: bool
    is_baseline: bool
    created_at: str


@app.get("/recalibration/status", response_model=RecalibrationStatusResponse)
async def get_recalibration_status(
    session: AsyncSession = Depends(get_async_session),
):
    """
    Get current recalibration status.

    Returns model health metrics, thresholds, and whether retraining is needed.
    """
    from app.ml.recalibration import RecalibrationEngine

    try:
        recalibrator = RecalibrationEngine(session)
        status = await recalibrator.get_recalibration_status()
        return RecalibrationStatusResponse(**status)
    except Exception as e:
        logger.error(f"Error getting recalibration status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/recalibration/team-adjustments", response_model=list[TeamAdjustmentResponse])
async def get_team_adjustments(
    session: AsyncSession = Depends(get_async_session),
):
    """
    Get all teams with confidence adjustments.

    Returns teams whose predictions are being adjusted due to high anomaly rates.
    """
    from app.ml.recalibration import RecalibrationEngine

    try:
        recalibrator = RecalibrationEngine(session)
        adjustments = await recalibrator.get_team_adjustments()
        return [TeamAdjustmentResponse(**adj) for adj in adjustments]
    except Exception as e:
        logger.error(f"Error getting team adjustments: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/recalibration/calculate-adjustments")
async def calculate_team_adjustments(
    days: int = 30,
    session: AsyncSession = Depends(get_async_session),
):
    """
    Manually trigger team adjustment calculation.

    Analyzes recent prediction outcomes and updates confidence multipliers.
    """
    from app.ml.recalibration import RecalibrationEngine

    try:
        recalibrator = RecalibrationEngine(session)
        result = await recalibrator.calculate_team_adjustments(days=days)
        return result
    except Exception as e:
        logger.error(f"Error calculating adjustments: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/recalibration/league-drift")
async def get_league_drift(
    session: AsyncSession = Depends(get_async_session),
):
    """
    Detect league-level accuracy drift.

    Compares weekly GOLD accuracy per league against historical baseline.
    Leagues with 15%+ accuracy drop are marked as 'Unstable'.

    Use this to identify structural changes in specific leagues.
    """
    from app.ml.recalibration import RecalibrationEngine

    try:
        recalibrator = RecalibrationEngine(session)
        result = await recalibrator.detect_league_drift()
        return result
    except Exception as e:
        logger.error(f"Error detecting league drift: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/recalibration/odds-movement")
async def get_odds_movements(
    days_ahead: int = 3,
    session: AsyncSession = Depends(get_async_session),
):
    """
    Check for significant market odds movements.

    Compares current market odds with our fair odds at prediction time.
    Movement of 25%+ triggers tier degradation warning.

    Returns matches with unusual market activity that may indicate
    information we don't have (injuries, lineup changes, etc).
    """
    from app.ml.recalibration import RecalibrationEngine

    try:
        recalibrator = RecalibrationEngine(session)
        result = await recalibrator.check_all_upcoming_odds_movements(days_ahead=days_ahead)
        return result
    except Exception as e:
        logger.error(f"Error checking odds movements: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/recalibration/odds-movement/{match_id}")
async def get_match_odds_movement(
    match_id: int,
    session: AsyncSession = Depends(get_async_session),
):
    """
    Check odds movement for a specific match.

    Returns detailed analysis of market movement and tier degradation recommendation.
    """
    from app.ml.recalibration import RecalibrationEngine

    try:
        recalibrator = RecalibrationEngine(session)
        result = await recalibrator.check_odds_movement(match_id)
        return result
    except Exception as e:
        logger.error(f"Error checking match odds movement: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/recalibration/lineup/{match_external_id}")
async def check_match_lineup(
    match_external_id: int,
    session: AsyncSession = Depends(get_async_session),
):
    """
    Check lineup validation for a specific match (Fase 3).

    Fetches announced lineup from API-Football (available ~60min before kickoff)
    and compares with expected best XI.

    Returns:
    - available: Whether lineups are announced
    - lineup_data: Formation and starters count for each team
    - tier_degradation: Recommended tier reduction (0, 1, or 2)
    - warnings: List of warnings (LINEUP_ROTATION_HOME, LINEUP_ROTATION_SEVERE_AWAY, etc.)
    - insights: Human-readable rotation analysis

    Variance thresholds:
    - 30%+ rotation = 1 tier degradation
    - 50%+ rotation = 2 tier degradation (severe)
    """
    from app.ml.recalibration import RecalibrationEngine

    try:
        recalibrator = RecalibrationEngine(session)
        result = await recalibrator.check_lineup_for_match(match_external_id)
        return result
    except Exception as e:
        logger.error(f"Error checking match lineup: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/matches/{match_id}/lineup")
async def get_match_lineup(
    match_id: int,
    session: AsyncSession = Depends(get_async_session),
):
    """
    Get full lineup information for a match.

    Fetches starting XI and substitutes for both teams.
    Available approximately 60 minutes before kickoff.
    """
    from app.etl.api_football import APIFootballProvider

    # Get match to find external ID
    match = await session.get(Match, match_id)
    if not match:
        raise HTTPException(status_code=404, detail="Match not found")

    provider = APIFootballProvider()
    try:
        lineup_data = await provider.get_lineups(match.external_id)

        if not lineup_data:
            return {
                "available": False,
                "match_id": match_id,
                "external_id": match.external_id,
                "message": "Lineups not yet announced (typically available ~60min before kickoff)",
            }

        return {
            "available": True,
            "match_id": match_id,
            "external_id": match.external_id,
            "home": lineup_data.get("home"),
            "away": lineup_data.get("away"),
        }
    finally:
        await provider.close()


@app.get("/recalibration/snapshots", response_model=list[ModelSnapshotResponse])
async def get_model_snapshots(
    session: AsyncSession = Depends(get_async_session),
):
    """
    Get all model snapshots.

    Returns history of model versions for rollback capability.
    """
    from app.models import ModelSnapshot

    try:
        query = select(ModelSnapshot).order_by(ModelSnapshot.created_at.desc())
        result = await session.execute(query)
        snapshots = result.scalars().all()

        return [
            ModelSnapshotResponse(
                id=s.id,
                model_version=s.model_version,
                model_path=s.model_path,
                brier_score=s.brier_score,
                samples_trained=s.samples_trained,
                is_active=s.is_active,
                is_baseline=s.is_baseline,
                created_at=s.created_at.isoformat(),
            )
            for s in snapshots
        ]
    except Exception as e:
        logger.error(f"Error getting snapshots: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/recalibration/create-baseline")
async def create_baseline_snapshot(
    brier_score: float = 0.2063,
    session: AsyncSession = Depends(get_async_session),
):
    """
    Create a baseline snapshot for the current model.

    This sets the reference point for model validation.
    New models must beat this Brier score to be deployed.
    """
    from app.ml.recalibration import RecalibrationEngine
    from app.config import get_settings
    from pathlib import Path

    settings = get_settings()

    try:
        recalibrator = RecalibrationEngine(session)

        # Find current model file
        model_path = Path(settings.MODEL_PATH)
        model_files = list(model_path.glob("xgb_*.json"))

        if not model_files:
            raise HTTPException(status_code=404, detail="No model files found")

        latest_model = max(model_files, key=lambda p: p.stat().st_mtime)

        snapshot = await recalibrator.create_snapshot(
            model_version=settings.MODEL_VERSION,
            model_path=str(latest_model),
            brier_score=brier_score,
            cv_scores=[brier_score],  # Single value for baseline
            samples_trained=0,  # Unknown for existing model
            is_baseline=True,
        )

        return {
            "message": "Baseline snapshot created",
            "snapshot_id": snapshot.id,
            "model_version": snapshot.model_version,
            "brier_score": snapshot.brier_score,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating baseline: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# LINEUP ARBITRAGE - Real-time odds capture at lineup announcement
# ============================================================================


@app.post("/lineup/monitor")
@limiter.limit("10/minute")
async def trigger_lineup_monitoring(
    request: Request,
    _: bool = Depends(verify_api_key),
):
    """
    Manually trigger lineup monitoring to capture odds at lineup_confirmed time.

    This is the same job that runs every 5 minutes automatically.
    Use this endpoint to test or force capture for matches in the next 90 minutes.

    The job:
    1. Finds matches starting within 90 minutes
    2. Checks if lineups are announced (11 players per team)
    3. If lineup is confirmed and no snapshot exists:
       - Captures current odds as 'lineup_confirmed' snapshot
       - Records exact timestamp for model evaluation

    This data is CRITICAL for evaluating the Lineup Arbitrage hypothesis:
    Can we beat the market odds AT THE MOMENT lineups are announced?

    Requires API key authentication.
    """
    from app.scheduler import monitor_lineups_and_capture_odds

    result = await monitor_lineups_and_capture_odds()
    return result


@app.get("/lineup/snapshots")
async def get_lineup_snapshots(
    days: int = 7,
    session: AsyncSession = Depends(get_async_session),
):
    """
    Get lineup_confirmed odds snapshots for recent matches.

    Returns matches where we captured odds at the moment of lineup announcement.
    This data is used to evaluate the Lineup Arbitrage model.

    Each snapshot includes:
    - match_id, date
    - snapshot_at: When we detected the lineup
    - odds at that moment (H/D/A)
    - implied probabilities (normalized)
    - TIMING METRICS: delta_to_kickoff, odds_freshness
    """
    from sqlalchemy import text

    cutoff = datetime.utcnow() - timedelta(days=days)

    result = await session.execute(text("""
        SELECT
            os.match_id,
            os.snapshot_at,
            os.odds_home,
            os.odds_draw,
            os.odds_away,
            os.prob_home,
            os.prob_draw,
            os.prob_away,
            os.overround,
            os.bookmaker,
            os.kickoff_time,
            os.delta_to_kickoff_seconds,
            os.odds_freshness,
            m.date as match_date,
            m.status,
            m.home_goals,
            m.away_goals,
            ht.name as home_team,
            at.name as away_team
        FROM odds_snapshots os
        JOIN matches m ON os.match_id = m.id
        LEFT JOIN teams ht ON m.home_team_id = ht.id
        LEFT JOIN teams at ON m.away_team_id = at.id
        WHERE os.snapshot_type = 'lineup_confirmed'
          AND os.snapshot_at >= :cutoff
        ORDER BY os.snapshot_at DESC
    """), {"cutoff": cutoff})

    snapshots = result.fetchall()

    # Calculate timing distribution
    deltas = [s.delta_to_kickoff_seconds for s in snapshots if s.delta_to_kickoff_seconds is not None]
    freshness_counts = {}
    for s in snapshots:
        f = s.odds_freshness or "unknown"
        freshness_counts[f] = freshness_counts.get(f, 0) + 1

    timing_stats = None
    if deltas:
        sorted_deltas = sorted(deltas)
        p50_idx = len(sorted_deltas) // 2
        p90_idx = int(len(sorted_deltas) * 0.9)
        timing_stats = {
            "count": len(deltas),
            "min_minutes": round(min(deltas) / 60, 1),
            "max_minutes": round(max(deltas) / 60, 1),
            "p50_minutes": round(sorted_deltas[p50_idx] / 60, 1),
            "p90_minutes": round(sorted_deltas[p90_idx] / 60, 1) if p90_idx < len(sorted_deltas) else None,
            "mean_minutes": round(sum(deltas) / len(deltas) / 60, 1),
        }

    return {
        "count": len(snapshots),
        "days": days,
        "timing_stats": timing_stats,
        "freshness_distribution": freshness_counts,
        "snapshots": [
            {
                "match_id": s.match_id,
                "home_team": s.home_team,
                "away_team": s.away_team,
                "match_date": s.match_date.isoformat() if s.match_date else None,
                "kickoff_time": s.kickoff_time.isoformat() if s.kickoff_time else None,
                "status": s.status,
                "final_score": f"{s.home_goals}-{s.away_goals}" if s.home_goals is not None else None,
                "snapshot_at": s.snapshot_at.isoformat() if s.snapshot_at else None,
                "delta_to_kickoff_minutes": round(s.delta_to_kickoff_seconds / 60, 1) if s.delta_to_kickoff_seconds else None,
                "odds_freshness": s.odds_freshness,
                "odds": {
                    "home": float(s.odds_home) if s.odds_home else None,
                    "draw": float(s.odds_draw) if s.odds_draw else None,
                    "away": float(s.odds_away) if s.odds_away else None,
                },
                "implied_probs": {
                    "home": float(s.prob_home) if s.prob_home else None,
                    "draw": float(s.prob_draw) if s.prob_draw else None,
                    "away": float(s.prob_away) if s.prob_away else None,
                },
                "overround": float(s.overround) if s.overround else None,
                "source": s.bookmaker,
            }
            for s in snapshots
        ],
    }


# =============================================================================
# PIT DASHBOARD (Minimalista - No DB queries)
# =============================================================================

# In-memory cache for dashboard data
_pit_dashboard_cache = {
    "data": None,
    "timestamp": 0,
    "ttl": 45,  # 45 seconds cache
}


def _load_latest_pit_report() -> dict:
    """
    Load the most recent PIT report from DB first, then filesystem fallback.

    Priority: DB (pit_reports table) > filesystem (logs/)
    This ensures reports survive Railway deploys.
    """
    import os
    from glob import glob

    result = {
        "weekly": None,
        "daily": None,
        "source": None,
        "error": None,
        "report_date": None,
        "created_at": None,
    }

    # Try DB first (persistent across deploys)
    try:
        db_result = _load_pit_reports_from_db()
        if db_result.get("daily") or db_result.get("weekly"):
            return db_result
    except Exception as e:
        # Log but don't fail - fall through to filesystem
        logger.debug(f"DB PIT lookup failed (will try filesystem): {e}")

    # Fallback to filesystem (for backwards compatibility)
    logs_dir = "logs"
    if not os.path.exists(logs_dir):
        result["error"] = "No PIT reports found (DB empty, no logs directory)"
        return result

    # Find latest weekly report
    weekly_files = glob(f"{logs_dir}/pit_weekly_*.json")
    if weekly_files:
        latest_weekly = max(weekly_files, key=os.path.getmtime)
        try:
            import json
            with open(latest_weekly) as f:
                result["weekly"] = json.load(f)
                result["weekly"]["_file"] = os.path.basename(latest_weekly)
                result["source"] = "filesystem_weekly"
        except Exception as e:
            result["error"] = f"Error reading weekly: {e}"

    # Find latest daily report (legacy fallbacks)
    # - pit_evaluation_live_only_*.json (current script)
    # - pit_evaluation_*.json (older/legacy)
    daily_files = glob(f"{logs_dir}/pit_evaluation_live_only_*.json") + glob(f"{logs_dir}/pit_evaluation_*.json")
    if daily_files:
        latest_daily = max(daily_files, key=os.path.getmtime)
        try:
            import json
            with open(latest_daily) as f:
                result["daily"] = json.load(f)
                result["daily"]["_file"] = os.path.basename(latest_daily)
                if not result["weekly"]:
                    result["source"] = "filesystem_daily"
        except Exception as e:
            if not result["error"]:
                result["error"] = f"Error reading daily: {e}"

    if not result["weekly"] and not result["daily"]:
        result["error"] = "No PIT reports found"

    return result


async def _load_pit_reports_from_db_async() -> dict:
    """
    Load latest PIT reports from pit_reports table (async version).
    Returns dict with weekly/daily payloads and metadata.
    """
    from sqlalchemy import text

    result = {
        "weekly": None,
        "daily": None,
        "source": None,
        "error": None,
        "report_date": None,
        "created_at": None,
    }

    try:
        async with AsyncSessionLocal() as session:
            # Get latest daily - process result INSIDE the session context
            daily_result = await session.execute(text("""
                SELECT payload, report_date, created_at, source
                FROM pit_reports
                WHERE report_type = 'daily'
                ORDER BY report_date DESC
                LIMIT 1
            """))
            daily_row = daily_result.fetchone()

            # Extract daily data while still in session
            if daily_row:
                result["daily"] = daily_row[0]  # payload is JSON
                result["report_date"] = str(daily_row[1])
                result["created_at"] = str(daily_row[2])
                result["source"] = f"db_{daily_row[3]}"

            # Get latest weekly
            weekly_result = await session.execute(text("""
                SELECT payload, report_date, created_at, source
                FROM pit_reports
                WHERE report_type = 'weekly'
                ORDER BY report_date DESC
                LIMIT 1
            """))
            weekly_row = weekly_result.fetchone()

            # Extract weekly data while still in session
            if weekly_row:
                result["weekly"] = weekly_row[0]
                if not result["source"]:
                    result["report_date"] = str(weekly_row[1])
                    result["created_at"] = str(weekly_row[2])
                    result["source"] = f"db_{weekly_row[3]}"

        # Check after session closes
        if not result["weekly"] and not result["daily"]:
            result["error"] = "No PIT reports in database"

    except Exception as e:
        logger.warning(f"PIT DB load error: {e}")
        result["error"] = f"DB error: {str(e)[:100]}"

    return result


def _load_pit_reports_from_db() -> dict:
    """
    Sync wrapper for _load_pit_reports_from_db_async.
    Used by cached functions that need sync interface.
    """
    import asyncio
    try:
        # Try to get existing event loop
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # We're inside an async context, can't use run_until_complete
            # Return empty to fall through to filesystem fallback
            return {"error": "Cannot run sync in async context"}
        return loop.run_until_complete(_load_pit_reports_from_db_async())
    except RuntimeError:
        # No event loop exists, create new one
        return asyncio.run(_load_pit_reports_from_db_async())


async def _get_cached_pit_data_async() -> dict:
    """Get PIT data with caching (async version for DB access)."""
    now = time.time()
    if _pit_dashboard_cache["data"] and (now - _pit_dashboard_cache["timestamp"]) < _pit_dashboard_cache["ttl"]:
        return _pit_dashboard_cache["data"]

    # Try DB first (async)
    data = await _load_pit_reports_from_db_async()
    if data.get("daily") or data.get("weekly"):
        _pit_dashboard_cache["data"] = data
        _pit_dashboard_cache["timestamp"] = now
        return data

    # Fallback to filesystem
    data = _load_latest_pit_report()
    _pit_dashboard_cache["data"] = data
    _pit_dashboard_cache["timestamp"] = now
    return data


def _get_cached_pit_data() -> dict:
    """Get PIT data with caching (sync fallback, uses filesystem only)."""
    now = time.time()
    if _pit_dashboard_cache["data"] and (now - _pit_dashboard_cache["timestamp"]) < _pit_dashboard_cache["ttl"]:
        return _pit_dashboard_cache["data"]

    data = _load_latest_pit_report()
    _pit_dashboard_cache["data"] = data
    _pit_dashboard_cache["timestamp"] = now
    return data


def _has_valid_session(request: Request) -> bool:
    """
    Check if request has a valid OPS session cookie.

    Returns True if session contains ops_authenticated=True and is not expired.
    """
    try:
        session = request.session
        if not session.get("ops_authenticated"):
            return False

        # Check expiration
        issued_at = session.get("issued_at")
        if issued_at:
            from datetime import datetime
            issued = datetime.fromisoformat(issued_at)
            ttl_hours = settings.OPS_SESSION_TTL_HOURS
            if datetime.utcnow() - issued > timedelta(hours=ttl_hours):
                return False

        return True
    except Exception:
        return False


def _verify_dashboard_token(request: Request) -> bool:
    """
    Verify dashboard access via token OR session.

    Auth methods (in order of preference):
    1. X-Dashboard-Token header (for services/automation)
    2. Valid session cookie (for web browser access)
    3. Query param token (dev only, disabled in prod)

    SECURITY: In production, query params are disabled.
    """
    # Method 1: Check header token
    token = settings.DASHBOARD_TOKEN
    if token:
        provided = request.headers.get("X-Dashboard-Token")
        if provided == token:
            return True

    # Method 2: Check valid session
    if _has_valid_session(request):
        return True

    # Method 3: Query param fallback ONLY in development
    if token and not os.getenv("RAILWAY_PROJECT_ID"):
        provided = request.query_params.get("token")
        if provided == token:
            return True

    return False


def _get_dashboard_token_from_request(request: Request) -> str | None:
    """
    Extract dashboard token from request.

    SECURITY: In production, only accepts token via X-Dashboard-Token header.
    Query params are only allowed in development (token leaks in logs/browser history).
    """
    # Header is preferred method
    token = request.headers.get("X-Dashboard-Token")

    # Query param fallback ONLY in development
    if not token and not os.getenv("RAILWAY_PROJECT_ID"):
        from fastapi import Query
        # Check if token was passed as query param
        token = request.query_params.get("token")

    return token


def _verify_debug_token(request: Request) -> None:
    """
    Verify dashboard token for debug endpoints. Raises HTTPException if invalid.

    Accepts either:
    - X-Dashboard-Token header
    - Valid session cookie

    SECURITY: Query params disabled in prod.
    """
    # Check session first (for browser access)
    if _has_valid_session(request):
        return

    # Then check header token
    expected = settings.DASHBOARD_TOKEN
    if not expected:
        raise HTTPException(status_code=503, detail="Dashboard token not configured")

    provided = _get_dashboard_token_from_request(request)
    if provided and provided == expected:
        return

    raise HTTPException(status_code=401, detail="Invalid token")


def _render_pit_dashboard_html(data: dict) -> str:
    """Render PIT dashboard as HTML."""
    weekly = data.get("weekly")
    daily = data.get("daily")
    source = data.get("source", "none")
    error = data.get("error")

    # Extract data preferring weekly, falling back to daily
    report = weekly or daily or {}

    # Detect if this is a daily live_only report (has protocol_version or counts.n_pit_valid_10_90)
    is_daily_live_only = (
        report.get("protocol_version") is not None or
        report.get("counts", {}).get("n_pit_valid_10_90") is not None
    )

    if is_daily_live_only:
        # Map daily live_only schema to dashboard variables
        counts = report.get("counts", {})
        principal_n = counts.get("n_pit_valid_10_90", 0)
        ideal_n = counts.get("n_pit_valid_ideal_45_75", 0)
        total_live = counts.get("n_total_snapshots", counts.get("n_pre_kickoff", 0))

        # Phase maps to status
        phase = report.get("phase", "unknown")
        phase_to_status = {
            "formal": "formal",
            "preliminar": "preliminary",
            "piloto": "piloto",
            "insufficient": "insufficient",
        }
        principal_status = phase_to_status.get(phase, phase)
        ideal_status = principal_status  # Same for daily

        # Quality score: use % ideal window as proxy, or N/A
        ideal_pct = round(ideal_n / principal_n * 100, 1) if principal_n > 0 else 0
        quality_score = ideal_pct  # Proxy: % in ideal window

        # Edge diagnostic: not available in daily, show phase info
        brier = report.get("brier", {})
        if brier.get("skill_vs_market") is not None:
            skill = brier.get("skill_vs_market", 0)
            if skill > 0.05:
                edge_diagnostic = "EDGE_PERSISTS"
            elif skill > -0.05:
                edge_diagnostic = "INCONCLUSIVE"
            else:
                edge_diagnostic = "NO_ALPHA"
        else:
            edge_diagnostic = "INSUFFICIENT_DATA"

        # Build captures_by_range from timing_distribution if available
        captures_by_range = {}
        timing = report.get("timing_distribution", {})
        if timing:
            # Approximate distribution based on counts
            captures_by_range = {
                "ideal_45_75": ideal_n,
                "valid_10_90": principal_n,
            }
        ideal_captures = ideal_n

        # Exclusions: not in daily schema
        exclusions = {}

        # Recommendation based on phase
        phase_recommendations = {
            "insufficient": f"Accumulating data (N={principal_n}, need 50+)",
            "piloto": f"Pilot phase (N={principal_n}, need 200+)",
            "preliminar": f"Preliminary phase (N={principal_n}, need 500+)",
            "formal": "Formal evaluation phase - metrics are statistically significant",
        }
        recommendation = phase_recommendations.get(phase, f"Phase: {phase}")

    else:
        # Original weekly schema mapping
        summary = report.get("summary", {})
        principal_n = summary.get("principal_n", report.get("counts", {}).get("total_principal", 0))
        ideal_n = summary.get("ideal_n", report.get("counts", {}).get("total_ideal", 0))
        principal_status = summary.get("principal_status", report.get("checkpoints", {}).get("principal", {}).get("status", "unknown"))
        ideal_status = summary.get("ideal_status", report.get("checkpoints", {}).get("ideal", {}).get("status", "unknown"))
        quality_score = summary.get("quality_score", report.get("data_quality", {}).get("quality_score", 0))
        edge_diagnostic = summary.get("edge_diagnostic", report.get("edge_decay_diagnostic", {}).get("diagnostic", "N/A"))

        # Full capture visibility
        visibility = report.get("full_capture_visibility", {})
        total_live = visibility.get("total_live_pre_kickoff_any_window", 0)
        captures_by_range = visibility.get("captures_by_range", {})
        ideal_captures = captures_by_range.get("ideal_45_75", 0)
        # % Ideal Window = ideal / valid_10_90 (not total_live which includes out-of-window)
        ideal_pct = round(ideal_n / principal_n * 100, 1) if principal_n > 0 else 0

        # Quality gate exclusions
        exclusions = report.get("data_quality", {}).get("exclusions", {})

        # Recommendation
        recommendation = report.get("recommendation", "N/A")

    # Calculate live % from weekly if available
    live_pct = 0
    if weekly:
        # From capture_delta or similar
        this_week = weekly.get("capture_delta", {}).get("this_week_ideal", 0)
        live_pct = 90  # Assume high if we have data (actual comes from freshness)

    # Timestamps - OPS style: use "—" for missing, with tooltips
    weekly_ts = weekly.get("generated_at") if weekly else None
    daily_ts = daily.get("generated_at", daily.get("timestamp")) if daily else None

    # Format timestamps OPS-style
    def format_ts_ops(ts, tooltip_missing):
        if ts:
            return f'<span>{ts[:19] if len(str(ts)) > 19 else ts}</span>'
        return f'<span class="muted" title="{tooltip_missing}">—</span>'

    weekly_display = format_ts_ops(weekly_ts, "Not generated yet. Runs Tuesdays 10:00 UTC.")
    daily_display = format_ts_ops(daily_ts, "Not generated yet. Runs daily 09:00 UTC.")

    # Source display - hide "File:" if db-backed
    source_is_db = source and source.startswith("db_")
    source_display = f"Source: {source}" if source else "Source: —"

    # Status icons
    def status_icon(status):
        if status == "formal":
            return "✅"
        elif status == "preliminary":
            return "🔶"
        else:
            return "⏳"

    def edge_icon(diag):
        icons = {
            "EDGE_PERSISTS": "✅",
            "EDGE_DECAYS": "⚠️",
            "NO_ALPHA": "❌",
            "INCONCLUSIVE": "⏳",
            "INSUFFICIENT_DATA": "📊",
        }
        return icons.get(diag, "❓")

    def format_edge_label(diag):
        """Format edge diagnostic for display (friendly labels)."""
        labels = {
            "EDGE_PERSISTS": "Edge Persists",
            "EDGE_DECAYS": "Edge Decays",
            "NO_ALPHA": "No Alpha",
            "INCONCLUSIVE": "Inconclusive",
            "INSUFFICIENT_DATA": "Insufficient Data",
        }
        return labels.get(diag, diag.replace("_", " ").title() if diag else "N/A")

    def format_bin_label(label):
        """Format bin label for display (friendly labels)."""
        labels = {
            "ideal_45_75": "Ideal [45-75]",
            "valid_10_90": "Valid [10-90]",
            "very_early_90plus": "Very Early [90+]",
            "early_75_90": "Early [75-90]",
            "late_30_45": "Late [30-45]",
            "very_late_10_30": "Very Late [10-30]",
            "too_late_under_10": "Too Late [<10]",
        }
        return labels.get(label, label.replace("_", " ").title())

    def get_diagnostic_context(n: int, phase: str, diag: str) -> tuple[str, str]:
        """
        Get context icon and tooltip based on sample size.
        Returns (context_icon, tooltip_text).
        """
        if phase == "insufficient" or n < 50:
            return (
                "ℹ️",
                "Early diagnostic (low N). High variance - not a business verdict. Accumulating data."
            )
        elif phase in ("piloto", "preliminar") or 50 <= n < 200:
            return (
                "⚠️",
                "Preliminary signal. Useful for monitoring, not conclusive. Review trend and wait for more N."
            )
        else:
            # Formal phase (n >= 200)
            if diag == "EDGE_PERSISTS":
                return (
                    "✅",
                    "Diagnostic with sufficient N (more reliable). Verify ROI/EV with CI before decisions."
                )
            else:
                return (
                    "❌",
                    "Diagnostic with sufficient N (more reliable). Verify ROI/EV with CI before decisions."
                )

    # Get diagnostic context based on sample size
    diag_context_icon, diag_tooltip = get_diagnostic_context(principal_n, principal_status, edge_diagnostic)

    # Bin data - OPS style empty state
    bins_html = ""
    for label, count in captures_by_range.items():
        pct = round(count / total_live * 100, 1) if total_live > 0 else 0
        highlight = 'class="highlight"' if "ideal" in label else ""
        display_label = format_bin_label(label)
        bins_html += f"<tr {highlight}><td>{display_label}</td><td>{count}</td><td>{pct}%</td></tr>"
    if not bins_html:
        bins_html = '<tr><td colspan="3" class="muted" style="text-align:center;">— No data yet —</td></tr>'

    # Exclusions table - OPS style empty state
    exclusions_html = ""
    sorted_excl = sorted(exclusions.items(), key=lambda x: x[1], reverse=True)
    for reason, count in sorted_excl[:5]:
        if count > 0:
            exclusions_html += f"<tr><td>{reason}</td><td>{count}</td></tr>"
    if not exclusions_html:
        exclusions_html = '<tr><td colspan="2" class="muted" style="text-align:center;">— No exclusions —</td></tr>'

    # Card color logic - OPS style (no alarm colors when no data)
    def card_color_ideal_pct(val):
        if val is None or val == 0:
            return ""  # No color when no data
        if val >= 60:
            return "green"
        if val >= 30:
            return "yellow"
        return "red"

    def card_color_quality(val):
        if val is None or val == 0:
            return ""  # No color when no data
        if val >= 60:
            return "green"
        if val >= 30:
            return "yellow"
        return "red"

    # Card values - OPS style "—" for missing
    def format_card_value(val, suffix=""):
        if val is None or (isinstance(val, (int, float)) and val == 0):
            return "—"
        return f"{val}{suffix}"

    live_snapshots_display = format_card_value(total_live) if total_live else "—"
    ideal_pct_display = format_card_value(ideal_pct, "%") if principal_n > 0 else "—"
    quality_score_display = format_card_value(quality_score, "%") if principal_n > 0 else "—"
    checkpoints_display = f"{status_icon(principal_status)} {principal_n}" if principal_n else "—"
    ideal_display = f"{status_icon(ideal_status)} {ideal_n} ideal" if ideal_n else "— ideal"

    # Subtitles - OPS style
    live_sub = "Total pre-kickoff" if total_live else "No data yet"
    ideal_pct_sub = f"[45-75] min: {ideal_captures} captures" if ideal_captures else "No data yet"
    quality_sub = "% ideal window (proxy)" if quality_score else "No data yet"
    checkpoints_sub = f"Principal [{ideal_display}]" if principal_n else "No data yet"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PIT Dashboard - FutbolStats</title>
    <style>
        :root {{
            --bg: #0f172a;
            --card: #1e293b;
            --border: #334155;
            --text: #e2e8f0;
            --muted: #94a3b8;
            --green: #22c55e;
            --yellow: #eab308;
            --red: #ef4444;
            --blue: #3b82f6;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'SF Pro', system-ui, sans-serif;
            background: var(--bg);
            color: var(--text);
            padding: 1.5rem;
            min-height: 100vh;
        }}
        .header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1.5rem;
            padding-bottom: 1rem;
            border-bottom: 1px solid var(--border);
        }}
        .header h1 {{ font-size: 1.5rem; font-weight: 600; }}
        .header .meta {{ color: var(--muted); font-size: 0.75rem; }}
        .cards {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin-bottom: 1.5rem;
        }}
        .card {{
            background: var(--card);
            border: 1px solid var(--border);
            border-radius: 0.75rem;
            padding: 1.25rem;
        }}
        .card-label {{ font-size: 0.75rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.05em; }}
        .card-value {{ font-size: 2rem; font-weight: 700; margin: 0.5rem 0; }}
        .card-sub {{ font-size: 0.875rem; color: var(--muted); }}
        .card.green .card-value {{ color: var(--green); }}
        .card.yellow .card-value {{ color: var(--yellow); }}
        .card.red .card-value {{ color: var(--red); }}
        .card.blue .card-value {{ color: var(--blue); }}
        .tables {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 1rem;
            margin-bottom: 1.5rem;
        }}
        .table-card {{
            background: var(--card);
            border: 1px solid var(--border);
            border-radius: 0.75rem;
            overflow: hidden;
        }}
        .table-card h3 {{
            padding: 1rem;
            font-size: 0.875rem;
            font-weight: 600;
            border-bottom: 1px solid var(--border);
        }}
        table {{ width: 100%; border-collapse: collapse; font-size: 0.875rem; }}
        th, td {{ padding: 0.75rem 1rem; text-align: left; }}
        th {{ color: var(--muted); font-weight: 500; }}
        tr:not(:last-child) {{ border-bottom: 1px solid var(--border); }}
        tr.highlight {{ background: rgba(34, 197, 94, 0.1); }}
        .decision-box {{
            background: var(--card);
            border: 2px solid var(--border);
            border-radius: 0.75rem;
            padding: 1.5rem;
            text-align: center;
        }}
        .decision-box h3 {{ margin-bottom: 1rem; font-size: 0.875rem; color: var(--muted); }}
        .decision {{ font-size: 1.25rem; font-weight: 600; }}
        .context-icon {{ cursor: help; margin-left: 0.5rem; }}
        .tooltip-hint {{
            font-size: 0.75rem;
            color: var(--muted);
            margin-top: 0.5rem;
            cursor: help;
            text-decoration: underline dotted;
        }}
        .tooltip-hint:hover, .context-icon:hover {{
            color: var(--blue);
        }}
        .muted {{ color: var(--muted); }}
        .error {{ background: rgba(239, 68, 68, 0.1); border-color: var(--red); padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem; }}
        .no-data {{ color: var(--muted); }}
        .footer {{ margin-top: 2rem; text-align: center; color: var(--muted); font-size: 0.75rem; }}
        .nav-tabs {{
            display: inline-flex;
            gap: 0.35rem;
            padding: 0.35rem;
            border: 1px solid var(--border);
            border-radius: 0.75rem;
            background: rgba(30, 41, 59, 0.55);
        }}
        .nav-tabs a {{
            display: inline-flex;
            align-items: center;
            padding: 0.35rem 0.6rem;
            border-radius: 0.6rem;
            color: var(--muted);
            font-size: 0.8rem;
            text-decoration: none;
            border: 1px solid transparent;
        }}
        .nav-tabs a:hover {{
            color: var(--text);
            border-color: rgba(59, 130, 246, 0.35);
            background: rgba(59, 130, 246, 0.12);
        }}
        .nav-tabs a.active {{
            color: var(--text);
            background: rgba(59, 130, 246, 0.18);
            border-color: rgba(59, 130, 246, 0.45);
        }}
        .json-dropdown {{
            position: relative;
            display: inline-block;
        }}
        .json-dropdown-btn {{
            display: inline-flex;
            align-items: center;
            padding: 0.35rem 0.6rem;
            border-radius: 0.6rem;
            color: var(--muted);
            font-size: 0.8rem;
            text-decoration: none;
            border: 1px solid transparent;
            cursor: pointer;
            background: transparent;
        }}
        .json-dropdown-btn:hover {{
            color: var(--text);
            border-color: rgba(59, 130, 246, 0.35);
            background: rgba(59, 130, 246, 0.12);
        }}
        .json-dropdown-content {{
            display: none;
            position: absolute;
            right: 0;
            top: 100%;
            margin-top: 0.25rem;
            background: var(--card);
            border: 1px solid var(--border);
            border-radius: 0.5rem;
            min-width: 180px;
            z-index: 100;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        }}
        .json-dropdown:hover .json-dropdown-content {{
            display: block;
        }}
        .json-dropdown-content a {{
            display: block;
            padding: 0.5rem 0.75rem;
            color: var(--muted);
            text-decoration: none;
            font-size: 0.75rem;
        }}
        .json-dropdown-content a:hover {{
            background: rgba(59, 130, 246, 0.12);
            color: var(--text);
        }}
        .json-dropdown-content a:first-child {{
            border-radius: 0.5rem 0.5rem 0 0;
        }}
        .copy-json-btn {{
            display: block;
            width: 100%;
            padding: 0.4rem 0.75rem;
            color: var(--muted);
            font-size: 0.75rem;
            text-align: left;
            background: rgba(59, 130, 246, 0.08);
            border: none;
            border-top: 1px solid var(--border);
            cursor: pointer;
        }}
        .copy-json-btn:hover {{
            background: rgba(59, 130, 246, 0.18);
            color: var(--text);
        }}
        .copy-json-btn:last-child {{
            border-radius: 0 0 0.5rem 0.5rem;
        }}
    </style>
</head>
<body>
    <div class="header">
        <div>
            <h1>📊 PIT Dashboard</h1>
            <div class="meta">{source_display}</div>
        </div>
        <div class="meta" style="text-align:right;">
            <div>Weekly: {weekly_display} | Daily: {daily_display}</div>
            <div style="margin-top: 0.35rem;">
                <div class="nav-tabs">
                    <a class="nav-link" data-path="/dashboard/ops" href="/dashboard/ops">Ops</a>
                    <a class="nav-link active" data-path="/dashboard/pit" href="/dashboard/pit">PIT</a>
                    <a class="nav-link" data-path="/dashboard/ops/logs" href="/dashboard/ops/logs">Logs</a>
                    <div class="json-dropdown">
                        <span class="json-dropdown-btn">JSON ▾</span>
                        <div class="json-dropdown-content">
                            <a data-path="/dashboard/ops.json" href="/dashboard/ops.json" target="_blank">Ops JSON</a>
                            <button class="copy-json-btn" data-endpoint="/dashboard/ops.json">📋 Copy Ops</button>
                            <a data-path="/dashboard/pit.json" href="/dashboard/pit.json" target="_blank">PIT JSON</a>
                            <button class="copy-json-btn" data-endpoint="/dashboard/pit.json">📋 Copy PIT</button>
                            <a data-path="/dashboard/ops/history.json?days=30" href="/dashboard/ops/history.json?days=30" target="_blank">History JSON</a>
                            <button class="copy-json-btn" data-endpoint="/dashboard/ops/history.json?days=30">📋 Copy History</button>
                            <a data-path="/dashboard/ops/logs.json?limit=200" href="/dashboard/ops/logs.json?limit=200" target="_blank">Logs JSON</a>
                            <button class="copy-json-btn" data-endpoint="/dashboard/ops/logs.json?limit=200">📋 Copy Logs</button>
                            <a data-path="/dashboard/ops/progress_snapshots.json" href="/dashboard/ops/progress_snapshots.json" target="_blank">Alpha Snapshots</a>
                            <button class="copy-json-btn" data-endpoint="/dashboard/ops/progress_snapshots.json">📋 Copy Alpha</button>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    {"<div class='error'>⚠️ " + error + "</div>" if error else ""}

    <div class="cards">
        <div class="card {'blue' if total_live else ''}">
            <div class="card-label">Live Snapshots</div>
            <div class="card-value {'no-data' if not total_live else ''}">{live_snapshots_display}</div>
            <div class="card-sub">{live_sub}</div>
        </div>
        <div class="card {card_color_ideal_pct(ideal_pct) if principal_n > 0 else ''}">
            <div class="card-label">% Ideal Window</div>
            <div class="card-value {'no-data' if principal_n == 0 else ''}">{ideal_pct_display}</div>
            <div class="card-sub">{ideal_pct_sub}</div>
        </div>
        <div class="card {card_color_quality(quality_score) if principal_n > 0 else ''}" title="Using % ideal window as proxy when no data_quality available">
            <div class="card-label">Quality Score</div>
            <div class="card-value {'no-data' if principal_n == 0 else ''}">{quality_score_display}</div>
            <div class="card-sub">{quality_sub}</div>
        </div>
        <div class="card">
            <div class="card-label">Checkpoints</div>
            <div class="card-value {'no-data' if not principal_n else ''}">{checkpoints_display}</div>
            <div class="card-sub">{checkpoints_sub}</div>
        </div>
    </div>

    <div class="tables">
        <div class="table-card">
            <h3>📍 Timing Distribution (Bins)</h3>
            <table>
                <thead><tr><th>Bin</th><th>Count</th><th>%</th></tr></thead>
                <tbody>{bins_html}</tbody>
            </table>
        </div>
        <div class="table-card">
            <h3>🚫 Quality Gate Exclusions</h3>
            <table>
                <thead><tr><th>Reason</th><th>Count</th></tr></thead>
                <tbody>{exclusions_html}</tbody>
            </table>
        </div>
    </div>

    <div class="decision-box">
        <h3>Edge Decay Diagnostic <span class="context-icon" title="{diag_tooltip}">{diag_context_icon}</span></h3>
        <div class="decision">{edge_icon(edge_diagnostic) + " " + format_edge_label(edge_diagnostic) if edge_diagnostic and edge_diagnostic != "N/A" else '<span class="no-data">—</span>'}</div>
        <div class="tooltip-hint" title="{diag_tooltip}">N={principal_n if principal_n else "—"} &bull; {principal_status if principal_status else "—"}</div>
        <div style="margin-top: 1rem; color: var(--muted);">{recommendation if recommendation and recommendation != "N/A" else "No recommendation yet"}</div>
    </div>

    <div class="footer">
        FutbolStats PIT Protocol v2.1 | Cache TTL: {_pit_dashboard_cache['ttl']}s
    </div>

    <script>
      // Preserve ?token= across dashboard navigation (for convenience).
      // Prefer X-Dashboard-Token header in production.
      (function() {{
        const params = new URLSearchParams(window.location.search);
        const token = params.get('token');
        if (!token) return;
        document.querySelectorAll('a.nav-link, .json-dropdown-content a').forEach(a => {{
          const path = a.getAttribute('data-path');
          if (!path) return;
          const joiner = path.includes('?') ? '&' : '?';
          a.setAttribute('href', path + joiner + 'token=' + encodeURIComponent(token));
        }});
        // Update copy buttons with token
        document.querySelectorAll('.copy-json-btn').forEach(btn => {{
          const endpoint = btn.getAttribute('data-endpoint');
          if (!endpoint) return;
          const joiner = endpoint.includes('?') ? '&' : '?';
          btn.setAttribute('data-endpoint', endpoint + joiner + 'token=' + encodeURIComponent(token));
        }});
      }})();

      // Copy JSON to clipboard
      document.querySelectorAll('.copy-json-btn').forEach(btn => {{
        btn.addEventListener('click', async () => {{
          const endpoint = btn.getAttribute('data-endpoint');
          try {{
            const res = await fetch(endpoint);
            const json = await res.json();
            await navigator.clipboard.writeText(JSON.stringify(json, null, 2));
            const orig = btn.textContent;
            btn.textContent = '✅ Copied!';
            setTimeout(() => btn.textContent = orig, 1500);
          }} catch (e) {{
            btn.textContent = '❌ Error';
            setTimeout(() => btn.textContent = btn.textContent.replace('❌ Error', '📋'), 1500);
          }}
        }});
      }});
    </script>
</body>
</html>"""
    return html


@app.get("/dashboard/pit")
async def pit_dashboard_html(request: Request):
    """
    PIT Dashboard - Visual overview of Point-In-Time evaluation status.

    Reads from logs/ files only (no DB queries).
    Auth: session cookie (web) or X-Dashboard-Token header (API).
    """
    from fastapi.responses import HTMLResponse, RedirectResponse

    if not _verify_dashboard_token(request):
        # Redirect to login for better UX
        return RedirectResponse(url="/ops/login", status_code=302)

    data = await _get_cached_pit_data_async()
    html = _render_pit_dashboard_html(data)
    return HTMLResponse(content=html)


@app.get("/dashboard/pit.json")
async def pit_dashboard_json(request: Request):
    """
    PIT Dashboard JSON - Raw data for programmatic access.

    Returns the latest weekly/daily PIT report data.
    """
    if not _verify_dashboard_token(request):
        raise HTTPException(
            status_code=401,
            detail="Dashboard access requires valid token.",
        )

    data = await _get_cached_pit_data_async()
    return {
        "source": data.get("source"),
        "error": data.get("error"),
        "report_date": data.get("report_date"),
        "created_at": data.get("created_at"),
        "weekly": data.get("weekly"),
        "daily": data.get("daily"),
        "cache_age_seconds": round(time.time() - _pit_dashboard_cache["timestamp"], 1) if _pit_dashboard_cache["timestamp"] else None,
    }


@app.get("/dashboard/pit/debug")
async def pit_dashboard_debug(request: Request):
    """
    Debug endpoint - shows raw pit_reports table content.
    Protected by dashboard token.
    """
    if not _verify_dashboard_token(request):
        raise HTTPException(status_code=401, detail="Dashboard access requires valid token.")

    from sqlalchemy import text

    result = {
        "table_contents": [],
        "weekly_count": 0,
        "daily_count": 0,
        "error": None,
    }

    try:
        async with AsyncSessionLocal() as session:
            # Get all rows
            rows_result = await session.execute(text("""
                SELECT id, report_type, report_date, source, created_at, updated_at,
                       LENGTH(payload::text) as payload_size
                FROM pit_reports
                ORDER BY report_date DESC, report_type DESC
                LIMIT 20
            """))
            rows = rows_result.fetchall()

            result["table_contents"] = [
                {
                    "id": row[0],
                    "report_type": row[1],
                    "report_date": str(row[2]),
                    "source": row[3],
                    "created_at": str(row[4]),
                    "updated_at": str(row[5]),
                    "payload_size": row[6],
                }
                for row in rows
            ]

            # Counts
            daily_result = await session.execute(text("SELECT COUNT(*) FROM pit_reports WHERE report_type='daily'"))
            result["daily_count"] = daily_result.scalar()

            weekly_result = await session.execute(text("SELECT COUNT(*) FROM pit_reports WHERE report_type='weekly'"))
            result["weekly_count"] = weekly_result.scalar()

    except Exception as e:
        result["error"] = str(e)

    return result


@app.post("/dashboard/pit/trigger")
async def pit_trigger_evaluation(request: Request):
    """
    Manually trigger PIT evaluation (for testing).
    Protected by dashboard token.
    """
    if not _verify_dashboard_token(request):
        raise HTTPException(status_code=401, detail="Dashboard access requires valid token.")

    from app.scheduler import daily_pit_evaluation

    try:
        await daily_pit_evaluation()
        return {"status": "ok", "message": "PIT evaluation triggered"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.post("/dashboard/predictions/trigger")
async def predictions_trigger_save(request: Request):
    """
    Manually trigger daily_save_predictions (for recovery).
    Protected by dashboard token.
    Use when predictions_health is RED/WARN.
    Returns detailed diagnostics for debugging.
    """
    if not _verify_dashboard_token(request):
        raise HTTPException(status_code=401, detail="Dashboard access requires valid token.")

    from app.db_utils import upsert
    from app.features import FeatureEngineer
    from app.ml.shadow import is_shadow_enabled, log_shadow_prediction
    from app.ml.sensor import log_sensor_prediction
    from app.config import get_settings
    from app.ops.audit import log_ops_action, OpsActionTimer

    sensor_settings = get_settings()
    start_time = time.time()

    diagnostics = {
        "status": "unknown",
        "model_loaded": False,
        "matches_found": 0,
        "ns_matches": 0,
        "predictions_generated": 0,
        "predictions_saved": 0,
        "shadow_logged": 0,
        "shadow_errors": 0,
        "sensor_logged": 0,
        "sensor_errors": 0,
        "errors": [],
    }

    try:
        async with AsyncSessionLocal() as session:
            # Step 1: Use global ml_engine (already loaded at startup)
            if not ml_engine.is_loaded:
                diagnostics["status"] = "error"
                diagnostics["errors"].append("Global ML model not loaded")
                return diagnostics
            diagnostics["model_loaded"] = True
            diagnostics["model_version"] = ml_engine.model_version

            # Step 2: Get features
            feature_engineer = FeatureEngineer(session=session)
            df = await feature_engineer.get_upcoming_matches_features()
            diagnostics["matches_found"] = len(df)

            if len(df) == 0:
                diagnostics["status"] = "ok"
                diagnostics["errors"].append("No upcoming matches found")
                return diagnostics

            # Filter to NS only
            df_ns = df[df["status"] == "NS"].copy()
            diagnostics["ns_matches"] = len(df_ns)

            if len(df_ns) == 0:
                diagnostics["status"] = "ok"
                diagnostics["errors"].append("No NS matches to predict")
                return diagnostics

            # Step 3: Generate predictions
            predictions = ml_engine.predict(df_ns)
            diagnostics["predictions_generated"] = len(predictions)

            # Step 4: Save to database with shadow + sensor logging
            saved = 0
            shadow_logged = 0
            shadow_errors = 0
            sensor_logged = 0
            sensor_errors = 0
            for idx, pred in enumerate(predictions):
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
                            "model_version": ml_engine.model_version,
                            "home_prob": probs["home"],
                            "draw_prob": probs["draw"],
                            "away_prob": probs["away"],
                        },
                        conflict_columns=["match_id", "model_version"],
                        update_columns=["home_prob", "draw_prob", "away_prob"],
                    )
                    saved += 1

                    # Shadow prediction: log parallel two-stage prediction
                    if is_shadow_enabled():
                        try:
                            match_df = df_ns.iloc[[idx]]
                            shadow_result = await log_shadow_prediction(
                                session=session,
                                match_id=match_id,
                                df=match_df,
                                baseline_engine=ml_engine,
                                skip_commit=True,
                            )
                            if shadow_result:
                                shadow_logged += 1
                        except Exception as shadow_err:
                            shadow_errors += 1
                            logger.warning(f"Shadow prediction failed for match {match_id}: {shadow_err}")

                    # Sensor B: log A vs B predictions (internal diagnostics only)
                    if sensor_settings.SENSOR_ENABLED:
                        try:
                            import numpy as np
                            match_df = df_ns.iloc[[idx]]
                            model_a_probs = np.array([probs["home"], probs["draw"], probs["away"]])
                            sensor_result = await log_sensor_prediction(
                                session=session,
                                match_id=match_id,
                                df=match_df,
                                model_a_probs=model_a_probs,
                                model_a_version=ml_engine.model_version,
                            )
                            if sensor_result:
                                sensor_logged += 1
                        except Exception as sensor_err:
                            sensor_errors += 1
                            logger.warning(f"Sensor prediction failed for match {match_id}: {sensor_err}")

                except Exception as e:
                    diagnostics["errors"].append(f"Match {match_id}: {str(e)[:50]}")

            await session.commit()
            diagnostics["predictions_saved"] = saved
            diagnostics["shadow_logged"] = shadow_logged
            diagnostics["shadow_errors"] = shadow_errors
            diagnostics["sensor_logged"] = sensor_logged
            diagnostics["sensor_errors"] = sensor_errors
            diagnostics["status"] = "ok" if saved > 0 else "no_new_predictions"

            shadow_info = f", shadow: {shadow_logged}" if is_shadow_enabled() else ""
            sensor_info = f", sensor: {sensor_logged}" if sensor_settings.SENSOR_ENABLED else ""
            logger.info(f"Predictions trigger complete: {saved} saved from {len(df_ns)} NS matches{shadow_info}{sensor_info}")

    except Exception as e:
        diagnostics["status"] = "error"
        diagnostics["errors"].append(str(e))
        logger.error(f"Predictions trigger failed: {e}")

    # Audit log
    duration_ms = int((time.time() - start_time) * 1000)
    try:
        async with AsyncSessionLocal() as audit_session:
            await log_ops_action(
                session=audit_session,
                request=request,
                action="predictions_trigger",
                params=None,
                result="ok" if diagnostics["status"] == "ok" else "error",
                result_detail={
                    "predictions_saved": diagnostics.get("predictions_saved", 0),
                    "ns_matches": diagnostics.get("ns_matches", 0),
                    "shadow_logged": diagnostics.get("shadow_logged", 0),
                    "sensor_logged": diagnostics.get("sensor_logged", 0),
                },
                error_message=diagnostics["errors"][0] if diagnostics["errors"] else None,
                duration_ms=duration_ms,
            )
    except Exception as audit_err:
        logger.warning(f"Failed to log audit for predictions_trigger: {audit_err}")

    return diagnostics


# =============================================================================
# OPS DASHBOARD (DB-backed, cached)
# =============================================================================

_ops_dashboard_cache = {
    "data": None,
    "timestamp": 0,
    "ttl": 45,  # seconds
}

# Rate-limit OPS_ALERT logging (once per 5 minutes max)
_predictions_health_alert_last: float = 0
_PREDICTIONS_HEALTH_ALERT_COOLDOWN = 300  # 5 minutes


async def _calculate_telemetry_summary(session) -> dict:
    """
    Calculate Data Quality Telemetry summary for ops dashboard.

    Queries DB for quarantine/taint/unmapped counts (24h window).
    Returns status: OK/WARN/RED based on data quality flags.

    Thresholds (conservative, protecting training/backtest/value):
    - RED: tainted_matches > 0 OR quarantined_odds > 0
    - WARN: unmapped_entities > 0 (and tainted/quarantined == 0)
    - OK: all counters == 0
    """
    now = datetime.utcnow()

    # 1) Quarantined odds in last 24h
    quarantined_odds_24h = 0
    try:
        res = await session.execute(
            text("""
                SELECT COUNT(*) FROM odds_history
                WHERE quarantined = true
                  AND recorded_at > NOW() - INTERVAL '24 hours'
            """)
        )
        quarantined_odds_24h = int(res.scalar() or 0)
    except Exception:
        pass  # Table may not exist yet

    # 2) Tainted matches (recent matches that are tainted)
    tainted_matches_24h = 0
    try:
        res = await session.execute(
            text("""
                SELECT COUNT(*) FROM matches
                WHERE tainted = true
                  AND date > NOW() - INTERVAL '7 days'
            """)
        )
        tainted_matches_24h = int(res.scalar() or 0)
    except Exception:
        pass  # Column may not exist yet

    # 3) Unmapped entities (teams without logo - proxy for incomplete data)
    unmapped_entities_24h = 0
    try:
        # Check for teams missing logo_url (proxy for unmapped/incomplete)
        res = await session.execute(
            text("""
                SELECT COUNT(DISTINCT t.id) FROM teams t
                WHERE t.logo_url IS NULL
            """)
        )
        unmapped_entities_24h = int(res.scalar() or 0)
    except Exception:
        pass

    # 4) Odds desync: matches with live snapshot but NULL odds in matches table
    # P1 sensor (2026-01-14): Detects when write-through fails silently
    odds_desync_6h = 0
    odds_desync_90m = 0
    try:
        # 6h window - early warning
        res = await session.execute(
            text("""
                SELECT COUNT(DISTINCT m.id)
                FROM matches m
                JOIN odds_snapshots os ON os.match_id = m.id
                WHERE m.status = 'NS'
                  AND m.date BETWEEN NOW() AND NOW() + INTERVAL '6 hours'
                  AND os.odds_freshness = 'live'
                  AND os.snapshot_type = 'lineup_confirmed'
                  AND os.snapshot_at >= NOW() - INTERVAL '120 minutes'
                  AND (m.odds_home IS NULL OR m.odds_draw IS NULL OR m.odds_away IS NULL)
            """)
        )
        odds_desync_6h = int(res.scalar() or 0)

        # 90m window - critical (near kickoff)
        res = await session.execute(
            text("""
                SELECT COUNT(DISTINCT m.id)
                FROM matches m
                JOIN odds_snapshots os ON os.match_id = m.id
                WHERE m.status = 'NS'
                  AND m.date BETWEEN NOW() AND NOW() + INTERVAL '90 minutes'
                  AND os.odds_freshness = 'live'
                  AND os.snapshot_type = 'lineup_confirmed'
                  AND os.snapshot_at >= NOW() - INTERVAL '120 minutes'
                  AND (m.odds_home IS NULL OR m.odds_draw IS NULL OR m.odds_away IS NULL)
            """)
        )
        odds_desync_90m = int(res.scalar() or 0)
    except Exception:
        pass  # Table/column may not exist

    # Determine status
    # RED: desync near kickoff (90m) OR tainted/quarantined
    # WARN: desync in 6h window OR unmapped entities
    if odds_desync_90m > 0 or tainted_matches_24h > 0 or quarantined_odds_24h > 0:
        status = "RED"
    elif odds_desync_6h > 0 or unmapped_entities_24h > 0:
        status = "WARN"
    else:
        status = "OK"

    # Build Grafana links from env vars (only if configured)
    links = []
    grafana_urls = {
        "Availability": os.environ.get("GRAFANA_DQ_AVAIL_URL"),
        "Freshness/Lag": os.environ.get("GRAFANA_DQ_LAG_URL"),
        "Market Integrity": os.environ.get("GRAFANA_DQ_MARKET_URL"),
        "Mapping Coverage": os.environ.get("GRAFANA_DQ_MAPPING_URL"),
    }
    for title, url in grafana_urls.items():
        if url:
            links.append({"title": f"Grafana: {title}", "url": url})

    return {
        "status": status,
        "updated_at": now.isoformat(),
        "summary": {
            "quarantined_odds_24h": quarantined_odds_24h,
            "tainted_matches_24h": tainted_matches_24h,
            "unmapped_entities_24h": unmapped_entities_24h,
            "odds_desync_6h": odds_desync_6h,
            "odds_desync_90m": odds_desync_90m,
        },
        "links": links,
    }


async def _calculate_shadow_mode_summary(session) -> dict:
    """
    Calculate Shadow Mode summary for ops dashboard.

    Returns state, counts, metrics, and recommendation for A/B model comparison.
    """
    from app.ml.shadow import is_shadow_enabled, get_shadow_engine
    from app.config import get_settings

    settings = get_settings()
    now = datetime.utcnow()

    # State info
    shadow_arch = settings.MODEL_SHADOW_ARCHITECTURE
    enabled = bool(shadow_arch)
    shadow_engine = get_shadow_engine()
    engine_loaded = shadow_engine is not None and shadow_engine.is_loaded if shadow_engine else False

    state = {
        "enabled": enabled,
        "shadow_architecture": shadow_arch or None,
        "shadow_model_version": shadow_engine.model_version if engine_loaded else None,
        "baseline_model_version": settings.MODEL_VERSION,
        "last_evaluation_at": None,
        "evaluation_job_interval_minutes": 30,
    }

    # Thresholds from settings
    min_samples = settings.SHADOW_MIN_SAMPLES
    brier_improvement_min = settings.SHADOW_BRIER_IMPROVEMENT_MIN
    accuracy_drop_max = settings.SHADOW_ACCURACY_DROP_MAX
    window_days = settings.SHADOW_WINDOW_DAYS

    # Default response if disabled
    if not enabled or not engine_loaded:
        return {
            "state": state,
            "counts": None,
            "metrics": None,
            "gating": {
                "min_samples_required": min_samples,
                "samples_evaluated": 0,
            },
            "recommendation": {
                "status": "DISABLED" if not enabled else "NOT_LOADED",
                "reason": "Shadow mode not configured" if not enabled else "Shadow model not loaded",
            },
        }

    # Counts query (window)
    counts = {
        "shadow_predictions_total": 0,
        "shadow_predictions_evaluated": 0,
        "shadow_predictions_pending": 0,
        "shadow_predictions_last_24h": 0,
        "shadow_evaluations_last_24h": 0,
    }

    try:
        # Rollback any previous failed transaction state
        await session.rollback()

        # Total predictions
        res = await session.execute(
            text(f"""
                SELECT COUNT(*) FROM shadow_predictions
                WHERE created_at > NOW() - INTERVAL '{window_days} days'
            """)
        )
        counts["shadow_predictions_total"] = int(res.scalar() or 0)

        # Evaluated vs pending
        res = await session.execute(
            text(f"""
                SELECT
                    COUNT(*) FILTER (WHERE evaluated_at IS NOT NULL) AS evaluated,
                    COUNT(*) FILTER (WHERE evaluated_at IS NULL) AS pending
                FROM shadow_predictions
                WHERE created_at > NOW() - INTERVAL '{window_days} days'
            """)
        )
        row = res.first()
        if row:
            counts["shadow_predictions_evaluated"] = int(row[0] or 0)
            counts["shadow_predictions_pending"] = int(row[1] or 0)

        # Last 24h predictions
        res = await session.execute(
            text("""
                SELECT COUNT(*) FROM shadow_predictions
                WHERE created_at > NOW() - INTERVAL '24 hours'
            """)
        )
        counts["shadow_predictions_last_24h"] = int(res.scalar() or 0)

        # Last 24h evaluations
        res = await session.execute(
            text("""
                SELECT COUNT(*) FROM shadow_predictions
                WHERE evaluated_at > NOW() - INTERVAL '24 hours'
            """)
        )
        counts["shadow_evaluations_last_24h"] = int(res.scalar() or 0)

        # Errors in last 24h (shadow prediction failures)
        res = await session.execute(
            text("""
                SELECT COUNT(*) FROM shadow_predictions
                WHERE error_code IS NOT NULL
                  AND created_at > NOW() - INTERVAL '24 hours'
            """)
        )
        counts["shadow_errors_last_24h"] = int(res.scalar() or 0)

        # Last evaluation timestamp
        res = await session.execute(
            text("SELECT MAX(evaluated_at) FROM shadow_predictions")
        )
        last_eval = res.scalar()
        if last_eval:
            state["last_evaluation_at"] = last_eval.isoformat() if hasattr(last_eval, 'isoformat') else str(last_eval)

    except Exception as e:
        logger.warning(f"Shadow mode counts query failed: {e}")

    # Metrics (only if enough evaluated samples)
    samples_evaluated = counts["shadow_predictions_evaluated"]
    metrics = None
    head_to_head = None
    draws_info = None

    if samples_evaluated >= min_samples:
        try:
            # Accuracy and Brier
            res = await session.execute(
                text(f"""
                    SELECT
                        AVG(CASE WHEN baseline_correct THEN 1.0 ELSE 0.0 END) AS baseline_acc,
                        AVG(CASE WHEN shadow_correct THEN 1.0 ELSE 0.0 END) AS shadow_acc,
                        AVG(baseline_brier) AS baseline_brier,
                        AVG(shadow_brier) AS shadow_brier,
                        SUM(CASE WHEN shadow_correct AND NOT baseline_correct THEN 1 ELSE 0 END) AS shadow_better,
                        SUM(CASE WHEN baseline_correct AND NOT shadow_correct THEN 1 ELSE 0 END) AS baseline_better,
                        SUM(CASE WHEN shadow_correct AND baseline_correct THEN 1 ELSE 0 END) AS both_correct,
                        SUM(CASE WHEN NOT shadow_correct AND NOT baseline_correct THEN 1 ELSE 0 END) AS both_wrong
                    FROM shadow_predictions
                    WHERE evaluated_at IS NOT NULL
                      AND created_at > NOW() - INTERVAL '{window_days} days'
                """)
            )
            row = res.first()
            if row:
                baseline_acc = float(row[0] or 0)
                shadow_acc = float(row[1] or 0)
                baseline_brier = float(row[2] or 0)
                shadow_brier = float(row[3] or 0)

                metrics = {
                    "baseline_accuracy": round(baseline_acc, 4),
                    "shadow_accuracy": round(shadow_acc, 4),
                    "baseline_brier": round(baseline_brier, 4),
                    "shadow_brier": round(shadow_brier, 4),
                    "delta_accuracy": round(shadow_acc - baseline_acc, 4),
                    "delta_brier": round(shadow_brier - baseline_brier, 4),
                }

                head_to_head = {
                    "shadow_better": int(row[4] or 0),
                    "baseline_better": int(row[5] or 0),
                    "both_correct": int(row[6] or 0),
                    "both_wrong": int(row[7] or 0),
                }

            # Draw prediction stats
            res = await session.execute(
                text(f"""
                    SELECT
                        AVG(CASE WHEN shadow_predicted = 'draw' THEN 1.0 ELSE 0.0 END) AS shadow_draw_pct,
                        AVG(CASE WHEN actual_result = 'draw' THEN 1.0 ELSE 0.0 END) AS actual_draw_pct
                    FROM shadow_predictions
                    WHERE evaluated_at IS NOT NULL
                      AND created_at > NOW() - INTERVAL '{window_days} days'
                """)
            )
            row = res.first()
            if row:
                draws_info = {
                    "shadow_draw_predicted_pct": round(float(row[0] or 0) * 100, 1),
                    "actual_draw_pct": round(float(row[1] or 0) * 100, 1),
                }

        except Exception as e:
            logger.warning(f"Shadow mode metrics query failed: {e}")

    # Recommendation logic
    if samples_evaluated < min_samples:
        recommendation = {
            "status": "NO_DATA",
            "reason": f"Need {min_samples} evaluated samples, have {samples_evaluated}",
        }
    elif metrics:
        delta_brier = metrics["delta_brier"]
        delta_acc = metrics["delta_accuracy"]

        # GO: shadow improves brier AND doesn't hurt accuracy too much
        if delta_brier <= -brier_improvement_min and delta_acc >= -accuracy_drop_max:
            recommendation = {
                "status": "GO",
                "reason": f"Shadow improves Brier by {-delta_brier:.4f}, accuracy delta {delta_acc:+.1%}",
            }
        # NO_GO: shadow degrades accuracy significantly
        elif delta_acc < -accuracy_drop_max:
            recommendation = {
                "status": "NO_GO",
                "reason": f"Shadow degrades accuracy by {-delta_acc:.1%} (max allowed: {accuracy_drop_max:.1%})",
            }
        # NO_GO: shadow makes brier worse
        elif delta_brier > brier_improvement_min:
            recommendation = {
                "status": "NO_GO",
                "reason": f"Shadow degrades Brier by {delta_brier:.4f}",
            }
        # HOLD: not enough improvement to switch
        else:
            recommendation = {
                "status": "HOLD",
                "reason": f"Shadow comparable to baseline (Brier delta: {delta_brier:+.4f}, accuracy delta: {delta_acc:+.1%})",
            }
    else:
        recommendation = {
            "status": "NO_DATA",
            "reason": "Metrics not available",
        }

    # Health signals for telemetry
    health = {
        "pending_ft_to_evaluate": 0,
        "eval_lag_minutes": 0.0,
        "stale_threshold_minutes": settings.SHADOW_EVAL_STALE_MINUTES,
        "is_stale": False,
    }
    try:
        from app.ml.shadow import get_shadow_health_metrics
        health_data = await get_shadow_health_metrics(session)
        health["pending_ft_to_evaluate"] = health_data.get("pending_ft", 0)
        health["eval_lag_minutes"] = health_data.get("eval_lag_minutes", 0.0)
        health["is_stale"] = health["eval_lag_minutes"] > settings.SHADOW_EVAL_STALE_MINUTES
    except Exception as e:
        logger.warning(f"Shadow health metrics query failed: {e}")

    return {
        "state": state,
        "counts": counts,
        "metrics": metrics,
        "head_to_head": head_to_head,
        "draws": draws_info,
        "gating": {
            "min_samples_required": min_samples,
            "samples_evaluated": samples_evaluated,
            "brier_improvement_min": brier_improvement_min,
            "accuracy_drop_max": accuracy_drop_max,
            "window_days": window_days,
        },
        "recommendation": recommendation,
        "health": health,
    }


async def _calculate_sensor_b_summary(session) -> dict:
    """
    Calculate Sensor B summary for ops dashboard.

    Sensor B is INTERNAL DIAGNOSTICS ONLY - never affects production picks.
    Returns flat structure for easy card rendering.

    States (Auditor-approved):
    - DISABLED: SENSOR_ENABLED=false
    - LEARNING: <min_samples evaluated, not ready to report metrics
    - TRACKING: >=min_samples, no conclusive signal yet
    - SIGNAL_DETECTED: signal_score > threshold, A may be stale
    - OVERFITTING_SUSPECTED: signal_score < threshold, B is noise
    - ERROR: exception during computation
    """
    from app.ml.sensor import get_sensor_report
    from app.config import get_settings

    sensor_settings = get_settings()

    if not sensor_settings.SENSOR_ENABLED:
        return {
            "state": "DISABLED",
            "reason": "SENSOR_ENABLED=false",
        }

    try:
        # Rollback any previous failed transaction
        await session.rollback()

        report = await get_sensor_report(session)

        # Determine state for card display (Auditor-approved statuses)
        rec = report.get("recommendation", {})
        rec_status = rec.get("status", "NO_DATA")

        if report.get("status") == "NO_DATA":
            state = "LEARNING"
        elif rec_status in ("SIGNAL_DETECTED", "OVERFITTING_SUSPECTED", "TRACKING"):
            state = rec_status
        else:
            state = "LEARNING"

        # Extract metrics for flat card display
        metrics = report.get("metrics", {})
        gating = report.get("gating", {})
        sensor_info = report.get("sensor_info", {})
        counts = report.get("counts", {})

        # Health signals for telemetry
        health = {
            "pending_ft_to_evaluate": 0,
            "eval_lag_minutes": 0.0,
            "stale_threshold_minutes": sensor_settings.SENSOR_EVAL_STALE_MINUTES,
            "is_stale": False,
        }
        try:
            from app.ml.sensor import get_sensor_health_metrics
            health_data = await get_sensor_health_metrics(session)
            health["pending_ft_to_evaluate"] = health_data.get("pending_ft", 0)
            health["eval_lag_minutes"] = health_data.get("eval_lag_minutes", 0.0)
            health["is_stale"] = health["eval_lag_minutes"] > sensor_settings.SENSOR_EVAL_STALE_MINUTES
        except Exception as he:
            logger.warning(f"Sensor health metrics query failed: {he}")

        # Compute accuracy percentages (only if samples >= min_samples)
        samples_evaluated = counts.get("evaluated", 0)
        min_samples = gating.get("min_samples_required", 50)
        has_enough_samples = samples_evaluated >= min_samples

        # Accuracy fields (null if not enough samples)
        accuracy_a_pct = None
        accuracy_b_pct = None
        delta_accuracy_pct = None

        if has_enough_samples:
            a_acc = metrics.get("a_accuracy")
            b_acc = metrics.get("b_accuracy")
            if a_acc is not None and b_acc is not None:
                accuracy_a_pct = round(a_acc * 100, 1)
                accuracy_b_pct = round(b_acc * 100, 1)
                delta_accuracy_pct = round((b_acc - a_acc) * 100, 1)

        return {
            "state": state,
            "reason": rec.get("reason", ""),
            # Counts
            "samples_evaluated": samples_evaluated,
            "samples_pending": counts.get("pending", 0),
            "min_samples": min_samples,
            # Accuracy A vs B (Auditor card) - only present if samples >= min_samples
            "accuracy_a_pct": accuracy_a_pct,
            "accuracy_b_pct": accuracy_b_pct,
            "delta_accuracy_pct": delta_accuracy_pct,
            "window_days": sensor_settings.SENSOR_EVAL_WINDOW_DAYS,
            "note": "solo FT evaluados",
            # Metrics (only show if we have enough samples - gating)
            "signal_score": metrics.get("signal_score") if state != "LEARNING" else None,
            "brier_a": metrics.get("a_brier") if state != "LEARNING" else None,
            "brier_b": metrics.get("b_brier") if state != "LEARNING" else None,
            "delta_brier": metrics.get("delta_brier") if state != "LEARNING" else None,
            "accuracy_a": metrics.get("a_accuracy") if state != "LEARNING" else None,
            "accuracy_b": metrics.get("b_accuracy") if state != "LEARNING" else None,
            # Sensor info
            "window_size": sensor_info.get("window_size", 50),
            "last_retrain_at": sensor_info.get("last_trained"),
            "retrain_interval_hours": sensor_settings.SENSOR_RETRAIN_INTERVAL_HOURS,
            "model_version": sensor_info.get("model_version"),
            "is_ready": sensor_info.get("is_ready", False),
            # Health (telemetry)
            "health": health,
        }
    except Exception as e:
        logger.warning(f"Sensor B summary failed: {e}")
        return {
            "state": "ERROR",
            "reason": str(e)[:100],
        }


async def _calculate_jobs_health_summary(session) -> dict:
    """
    Calculate P0 jobs health summary for OPS dashboard.

    Monitors the three critical jobs:
    - stats_backfill: Fetches match stats for finished matches
    - odds_sync: Syncs 1X2 odds for upcoming matches
    - fastpath: Generates LLM narratives for finished matches

    Each job reports:
    - last_success_at: Timestamp of last OK run
    - minutes_since_success: Gap since last OK
    - ft_pending (stats_backfill): FT matches without stats
    - backlog_ready (fastpath): Audits ready for narratives
    - status: ok/warn/red based on thresholds
    """
    from app.telemetry.metrics import (
        job_last_success_timestamp,
        stats_backfill_ft_pending_gauge,
        fastpath_backlog_ready_gauge,
    )

    now = datetime.utcnow()

    # Helper to format timestamp and calculate age
    def job_status(job_name: str, max_gap_minutes: int) -> dict:
        try:
            ts = job_last_success_timestamp.labels(job=job_name)._value.get()
            if ts and ts > 0:
                last_success = datetime.utcfromtimestamp(ts)
                gap_minutes = (now - last_success).total_seconds() / 60
                status = "ok"
                if gap_minutes > max_gap_minutes * 2:
                    status = "red"
                elif gap_minutes > max_gap_minutes:
                    status = "warn"
                return {
                    "last_success_at": last_success.isoformat() + "Z",
                    "minutes_since_success": round(gap_minutes, 1),
                    "status": status,
                }
        except Exception:
            pass
        return {
            "last_success_at": None,
            "minutes_since_success": None,
            "status": "unknown",
        }

    # Stats backfill: runs hourly, warn if >2h, red if >3h
    stats_health = job_status("stats_backfill", max_gap_minutes=120)
    try:
        ft_pending = int(stats_backfill_ft_pending_gauge._value.get() or 0)
        stats_health["ft_pending"] = ft_pending
        # Escalate status based on pending count
        if ft_pending > 10:
            stats_health["status"] = "red"
        elif ft_pending > 5 and stats_health["status"] == "ok":
            stats_health["status"] = "warn"
    except Exception:
        stats_health["ft_pending"] = None

    # Odds sync: runs every 6h, warn if >12h, red if >18h
    odds_health = job_status("odds_sync", max_gap_minutes=720)

    # Fastpath: runs every 2min, warn if >5min, red if >10min
    fastpath_health = job_status("fastpath", max_gap_minutes=5)
    try:
        backlog_ready = int(fastpath_backlog_ready_gauge._value.get() or 0)
        fastpath_health["backlog_ready"] = backlog_ready
        # Escalate status based on backlog
        if backlog_ready > 5:
            fastpath_health["status"] = "red"
        elif backlog_ready > 3 and fastpath_health["status"] == "ok":
            fastpath_health["status"] = "warn"
    except Exception:
        fastpath_health["backlog_ready"] = None

    # Overall status: worst of the three
    statuses = [stats_health["status"], odds_health["status"], fastpath_health["status"]]
    if "red" in statuses:
        overall = "red"
    elif "warn" in statuses:
        overall = "warn"
    elif all(s == "ok" for s in statuses):
        overall = "ok"
    else:
        overall = "unknown"

    # Add help URLs for oncall quick reference
    runbook_base = "docs/GRAFANA_ALERTS_CHECKLIST.md"
    stats_health["help_url"] = f"{runbook_base}#stats-backfill-job"
    odds_health["help_url"] = f"{runbook_base}#odds-sync-job"
    fastpath_health["help_url"] = f"{runbook_base}#fastpath-llm-narratives-job"

    return {
        "status": overall,
        "runbook_url": f"{runbook_base}#p0-jobs-health-scheduler-jobs",
        "stats_backfill": stats_health,
        "odds_sync": odds_health,
        "fastpath": fastpath_health,
    }


async def _calculate_rerun_serving_summary(session) -> dict:
    """
    Calculate rerun serving metrics for OPS dashboard.

    Shows canary status: how many predictions are served from DB (two-stage)
    vs live baseline, plus active rerun info.
    """
    settings = get_settings()
    try:
        # Check if enabled
        enabled = settings.PREFER_RERUN_PREDICTIONS
        freshness_hours = settings.RERUN_FRESHNESS_HOURS

        # Get active rerun info
        res = await session.execute(
            text("""
                SELECT run_id, architecture_after, model_version_after,
                       matches_total, created_at
                FROM prediction_reruns
                WHERE is_active = true
                ORDER BY created_at DESC
                LIMIT 1
            """)
        )
        active_rerun = res.fetchone()

        # Count NS matches with rerun predictions (fresh)
        res = await session.execute(
            text("""
                SELECT COUNT(DISTINCT p.match_id)
                FROM predictions p
                JOIN matches m ON m.id = p.match_id
                WHERE p.run_id IS NOT NULL
                  AND m.status = 'NS'
                  AND m.date > NOW()
                  AND p.created_at > NOW() - make_interval(hours => :hours)
            """),
            {"hours": freshness_hours * 2}  # Use 2x freshness for counting
        )
        ns_with_rerun = int(res.scalar() or 0)

        # Total NS matches in window
        res = await session.execute(
            text("""
                SELECT COUNT(*)
                FROM matches
                WHERE status = 'NS'
                  AND date > NOW()
                  AND date < NOW() + INTERVAL '7 days'
            """)
        )
        total_ns = int(res.scalar() or 0)

        # Get in-memory serving stats
        from_rerun_pct = round(100.0 * ns_with_rerun / total_ns, 1) if total_ns > 0 else 0.0
        from_baseline_pct = round(100.0 - from_rerun_pct, 1)

        return {
            "enabled": enabled,
            "freshness_hours": freshness_hours,
            "active_rerun": {
                "run_id": str(active_rerun[0]) if active_rerun else None,
                "architecture": active_rerun[1] if active_rerun else None,
                "model_version": active_rerun[2] if active_rerun else None,
                "matches_total": active_rerun[3] if active_rerun else None,
                "created_at": active_rerun[4].isoformat() if active_rerun else None,
            } if active_rerun else None,
            "coverage": {
                "ns_with_rerun": ns_with_rerun,
                "total_ns_7d": total_ns,
                "from_rerun_pct": from_rerun_pct,
                "from_baseline_pct": from_baseline_pct,
            },
            "in_memory_stats": _rerun_serving_stats.copy(),
        }
    except Exception as e:
        logger.warning(f"Rerun serving summary failed: {e}")
        return {
            "enabled": settings.PREFER_RERUN_PREDICTIONS,
            "error": str(e)[:100],
        }


async def _calculate_predictions_health(session) -> dict:
    """
    Calculate predictions health metrics for P0 observability.

    Detects when daily_save_predictions isn't running/persisting.
    Returns status: ok/warn/red based on recency and coverage.

    Smart logic: If there are no upcoming NS matches scheduled, we don't
    raise WARN/RED for stale predictions (false positive in low-activity periods).
    """
    now = datetime.utcnow()

    # 1) Last prediction saved (for any match)
    res = await session.execute(
        text("SELECT MAX(created_at) FROM predictions")
    )
    last_pred_at = res.scalar()

    # 2) Predictions saved in last 24h
    res = await session.execute(
        text("""
            SELECT COUNT(*) FROM predictions
            WHERE created_at > NOW() - INTERVAL '24 hours'
        """)
    )
    preds_last_24h = int(res.scalar() or 0)

    # 3) Predictions saved today (UTC)
    res = await session.execute(
        text("""
            SELECT COUNT(*) FROM predictions
            WHERE created_at::date = CURRENT_DATE
        """)
    )
    preds_today = int(res.scalar() or 0)

    # 4) FT matches in last 48h
    res = await session.execute(
        text("""
            SELECT COUNT(*) FROM matches
            WHERE status IN ('FT', 'AET', 'PEN')
              AND date > NOW() - INTERVAL '48 hours'
        """)
    )
    ft_48h = int(res.scalar() or 0)

    # 5) FT matches in last 48h MISSING prediction
    res = await session.execute(
        text("""
            SELECT COUNT(*) FROM matches m
            WHERE m.status IN ('FT', 'AET', 'PEN')
              AND m.date > NOW() - INTERVAL '48 hours'
              AND NOT EXISTS (
                  SELECT 1 FROM predictions p WHERE p.match_id = m.id
              )
        """)
    )
    ft_48h_missing = int(res.scalar() or 0)

    # 6) Coverage percentage
    coverage_48h_pct = 0.0
    if ft_48h > 0:
        coverage_48h_pct = round(((ft_48h - ft_48h_missing) / ft_48h) * 100, 1)

    # 7) NS matches in next 48h (for smart alerting)
    res = await session.execute(
        text("""
            SELECT COUNT(*) FROM matches
            WHERE status = 'NS'
              AND date > NOW()
              AND date <= NOW() + INTERVAL '48 hours'
        """)
    )
    ns_next_48h = int(res.scalar() or 0)

    # 8) Next NS match date (for visibility)
    res = await session.execute(
        text("""
            SELECT MIN(date) FROM matches
            WHERE status = 'NS' AND date > NOW()
        """)
    )
    next_ns_date = res.scalar()

    # 9) NS matches in next 48h MISSING prediction (key metric!)
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
    ns_next_48h_missing = int(res.scalar() or 0)

    # 10) NS coverage percentage (upcoming matches with predictions)
    ns_coverage_pct = 100.0
    if ns_next_48h > 0:
        ns_coverage_pct = round(((ns_next_48h - ns_next_48h_missing) / ns_next_48h) * 100, 1)

    # Calculate hours since last prediction (informational only)
    hours_since_last = None
    if last_pred_at:
        delta = now - last_pred_at
        hours_since_last = round(delta.total_seconds() / 3600, 1)

    # Determine status with smart logic
    # Primary metric: NS coverage (do upcoming matches have predictions?)
    # Secondary metric: FT coverage (did past matches have predictions?)
    status = "ok"
    status_reason = None

    # Smart bypass: no upcoming matches = no expectation of predictions
    if ns_next_48h == 0:
        status = "ok"
        status_reason = "No upcoming NS matches in 48h (low activity period)"
    # NEW: Check if prediction job hasn't run recently (stale job detection)
    elif hours_since_last and hours_since_last > 12 and ns_next_48h > 0:
        status = "warn"
        status_reason = f"Predictions job stale: {hours_since_last:.1f}h since last save with {ns_next_48h} NS upcoming"
    # Primary check: upcoming NS matches should have predictions
    elif ns_coverage_pct < 50:
        status = "red"
        status_reason = f"NS coverage {ns_coverage_pct}% < 50% ({ns_next_48h_missing}/{ns_next_48h} missing)"
    elif ns_coverage_pct < 80:
        status = "warn"
        status_reason = f"NS coverage {ns_coverage_pct}% < 80% ({ns_next_48h_missing}/{ns_next_48h} missing)"
    # Secondary check: past FT matches coverage
    elif coverage_48h_pct < 50:
        status = "red"
        status_reason = f"FT coverage {coverage_48h_pct}% < 50% threshold"
    elif coverage_48h_pct < 80:
        status = "warn"
        status_reason = f"FT coverage {coverage_48h_pct}% < 80% threshold"

    # Log OPS_ALERT if red/warn (rate-limited to avoid spam)
    global _predictions_health_alert_last
    import time as _time
    now_ts = _time.time()

    if status in ("red", "warn") and (now_ts - _predictions_health_alert_last) > _PREDICTIONS_HEALTH_ALERT_COOLDOWN:
        _predictions_health_alert_last = now_ts
        if status == "red":
            logger.error(
                f"[OPS_ALERT] predictions_health=RED: {status_reason}. "
                f"last_pred={last_pred_at}, preds_24h={preds_last_24h}, "
                f"ft_48h={ft_48h}, missing={ft_48h_missing}, ns_next_48h={ns_next_48h}"
            )
        else:
            logger.warning(
                f"[OPS_ALERT] predictions_health=WARN: {status_reason}. "
                f"last_pred={last_pred_at}, preds_24h={preds_last_24h}, ns_next_48h={ns_next_48h}"
            )

    # Emit Prometheus metrics for Grafana alerting (P1)
    try:
        from app.telemetry.metrics import set_predictions_health_metrics
        set_predictions_health_metrics(
            hours_since_last=hours_since_last,
            ns_next_48h=ns_next_48h,
            ns_missing_next_48h=ns_next_48h_missing,
            coverage_ns_pct=ns_coverage_pct,
            status=status,
        )
    except Exception as e:
        logger.debug(f"Failed to emit predictions health metrics: {e}")

    return {
        "status": status,
        "status_reason": status_reason,
        # NS (upcoming) metrics - primary
        "ns_matches_next_48h": ns_next_48h,
        "ns_matches_next_48h_missing_prediction": ns_next_48h_missing,
        "ns_coverage_pct": ns_coverage_pct,
        "next_ns_match_utc": next_ns_date.isoformat() if next_ns_date else None,
        # FT (past) metrics - secondary
        "ft_matches_last_48h": ft_48h,
        "ft_matches_last_48h_missing_prediction": ft_48h_missing,
        "ft_coverage_pct": coverage_48h_pct,
        # Informational (not used for status determination)
        "last_prediction_saved_at": last_pred_at.isoformat() if last_pred_at else None,
        "hours_since_last_prediction": hours_since_last,
        "predictions_saved_last_24h": preds_last_24h,
        "predictions_saved_today_utc": preds_today,
        "thresholds": {
            "ns_coverage_warn_pct": 80,
            "ns_coverage_red_pct": 50,
            "ft_coverage_warn_pct": 80,
            "ft_coverage_red_pct": 50,
        },
    }


async def _calculate_model_performance(session) -> dict:
    """
    Calculate model performance summary for OPS dashboard card.

    Returns compact summary from the latest 7d performance report:
    - Brier score
    - Skill vs market
    - Recommendation (OK/WATCH/INVESTIGATE)
    - Status color (green/yellow/red)
    """
    from app.ml.performance_metrics import get_latest_report

    try:
        report = await get_latest_report(session, window_days=7)

        if not report:
            return {
                "status": "gray",
                "status_reason": "No report available yet",
                "brier_score": None,
                "skill_vs_market": None,
                "recommendation": None,
                "n_predictions": 0,
                "confidence": "none",
                "report_generated_at": None,
            }

        global_metrics = report.get("global", {})
        metrics = global_metrics.get("metrics", {})
        diagnostics = report.get("diagnostics", {})
        market = metrics.get("market_comparison", {})

        brier = metrics.get("brier_score")
        skill = market.get("skill_vs_market") if market else None
        recommendation = diagnostics.get("recommendation", "OK")
        n = global_metrics.get("n", 0)
        confidence = report.get("confidence", "low")

        # Determine status color based on recommendation
        if "INVESTIGATE" in recommendation:
            status = "red"
        elif "MONITOR" in recommendation or "WATCH" in recommendation:
            status = "yellow"
        else:
            status = "green"

        # Format skill for display
        skill_display = None
        if skill is not None:
            skill_display = f"{skill * 100:+.1f}%"

        return {
            "status": status,
            "status_reason": recommendation,
            "brier_score": round(brier, 4) if brier else None,
            "skill_vs_market": skill_display,
            "skill_vs_market_raw": round(skill, 4) if skill is not None else None,
            "recommendation": recommendation,
            "n_predictions": n,
            "confidence": confidence,
            "report_generated_at": report.get("generated_at"),
            "endpoint": "/dashboard/ops/predictions_performance.json?window_days=7",
        }

    except Exception as e:
        logger.warning(f"Error calculating model performance: {e}")
        return {
            "status": "gray",
            "status_reason": f"Error: {str(e)[:50]}",
            "brier_score": None,
            "skill_vs_market": None,
            "recommendation": None,
            "n_predictions": 0,
            "confidence": "error",
            "report_generated_at": None,
        }


async def _calculate_fastpath_health(session) -> dict:
    """
    Calculate fast-path LLM narrative health metrics.

    Monitors the fast-path job that generates narratives within minutes
    of match completion instead of waiting for daily audit.

    Returns status: ok/warn/red based on tick recency and error rates.
    """
    from app.config import get_settings

    settings = get_settings()
    now = datetime.utcnow()

    # Read ticks from DB (canonical source - survives restarts/multi-process)
    last_tick_at = None
    last_tick_result = {}
    ticks_total = 0
    ticks_with_activity = 0
    db_unavailable = False

    try:
        # Get most recent tick from DB
        res = await session.execute(
            text("""
                SELECT tick_at, selected, refreshed, ready, enqueued, completed, errors, skipped
                FROM fastpath_ticks
                ORDER BY tick_at DESC
                LIMIT 1
            """)
        )
        row = res.fetchone()
        if row:
            last_tick_at = row[0]
            last_tick_result = {
                "selected": row[1], "refreshed": row[2], "stats_ready": row[3],
                "enqueued": row[4], "completed": row[5], "errors": row[6], "skipped": row[7]
            }

        # Get tick counts from last hour
        res = await session.execute(
            text("""
                SELECT COUNT(*), COUNT(*) FILTER (WHERE selected > 0 OR enqueued > 0 OR completed > 0)
                FROM fastpath_ticks
                WHERE tick_at > NOW() - INTERVAL '1 hour'
            """)
        )
        counts = res.fetchone()
        if counts:
            ticks_total = counts[0] or 0
            ticks_with_activity = counts[1] or 0
    except Exception as db_err:
        # DB unavailable - mark as red status (don't use in-memory fallback in prod)
        logger.warning(f"fastpath_ticks DB unavailable: {db_err}")
        db_unavailable = True

    # Check if fast-path is enabled
    enabled = os.environ.get("FASTPATH_ENABLED", str(settings.FASTPATH_ENABLED)).lower()
    is_enabled = enabled not in ("false", "0", "no")

    # Calculate minutes since last tick
    minutes_since_tick = None
    if last_tick_at:
        delta = now - last_tick_at
        minutes_since_tick = round(delta.total_seconds() / 60, 1)

    # Query DB for LLM stats in last 60 minutes
    llm_60m = {"ok": 0, "ok_retry": 0, "error": 0, "skipped": 0, "in_queue": 0, "running": 0}
    error_codes_60m = {}
    pending_ready = 0

    try:
        # LLM status breakdown (last 60 min)
        res = await session.execute(
            text("""
                SELECT llm_narrative_status, COUNT(*) as cnt
                FROM post_match_audits
                WHERE created_at > NOW() - INTERVAL '60 minutes'
                  AND llm_narrative_status IS NOT NULL
                GROUP BY llm_narrative_status
            """)
        )
        for row in res.fetchall():
            status_key = row[0] or "unknown"
            llm_60m[status_key] = int(row[1])

        # Error codes breakdown (last 60 min)
        res = await session.execute(
            text("""
                SELECT llm_narrative_error_code, COUNT(*) as cnt
                FROM post_match_audits
                WHERE created_at > NOW() - INTERVAL '60 minutes'
                  AND llm_narrative_error_code IS NOT NULL
                GROUP BY llm_narrative_error_code
                ORDER BY cnt DESC
                LIMIT 5
            """)
        )
        for row in res.fetchall():
            error_codes_60m[row[0]] = int(row[1])

        # Pending ready: FT matches with stats_ready but no successful narrative
        # Use COALESCE(finished_at, date) to align with fast-path selector logic
        res = await session.execute(
            text("""
                SELECT COUNT(*)
                FROM matches m
                WHERE m.status IN ('FT', 'AET', 'PEN')
                  AND COALESCE(m.finished_at, m.date) > NOW() - INTERVAL '180 minutes'
                  AND m.stats_ready_at IS NOT NULL
                  AND NOT EXISTS (
                      SELECT 1 FROM prediction_outcomes po
                      JOIN post_match_audits pma ON pma.outcome_id = po.id
                      WHERE po.match_id = m.id
                        AND pma.llm_narrative_status = 'ok'
                  )
            """)
        )
        pending_ready = int(res.scalar() or 0)

    except Exception as e:
        logger.warning(f"Error calculating fastpath_health DB metrics: {e}")

    # Calculate total errors and success
    total_ok = llm_60m.get("ok", 0) + llm_60m.get("ok_retry", 0)
    total_errors = llm_60m.get("error", 0)
    total_processed = total_ok + total_errors + llm_60m.get("skipped", 0)
    error_rate_60m = 0.0
    if total_processed > 0:
        error_rate_60m = round((total_errors / total_processed) * 100, 1)

    # Determine status
    status = "ok"
    status_reason = None

    if db_unavailable:
        status = "red"
        status_reason = "fastpath_ticks table unavailable"
    elif not is_enabled:
        status = "disabled"
        status_reason = "FASTPATH_ENABLED=false"
    elif minutes_since_tick is None:
        status = "warn"
        status_reason = "No tick recorded yet (job may not have run)"
    elif minutes_since_tick > 10:
        status = "red"
        status_reason = f"No tick in {minutes_since_tick:.0f} min (>10 min threshold)"
    elif error_rate_60m > 50:
        status = "red"
        status_reason = f"Error rate {error_rate_60m}% > 50% threshold"
    elif error_rate_60m > 20:
        status = "warn"
        status_reason = f"Error rate {error_rate_60m}% > 20% threshold"
    elif total_ok == 0 and llm_60m.get("skipped", 0) > 5:
        status = "warn"
        status_reason = f"0 ok, {llm_60m.get('skipped', 0)} skipped (gating issues?)"

    return {
        "status": status,
        "status_reason": status_reason,
        "enabled": is_enabled,
        "last_tick_at": last_tick_at.isoformat() if last_tick_at else None,
        "minutes_since_tick": minutes_since_tick,
        "last_tick_result": last_tick_result,
        "ticks_total": ticks_total,
        "ticks_with_activity": ticks_with_activity,
        "last_60m": {
            "ok": llm_60m.get("ok", 0),
            "ok_retry": llm_60m.get("ok_retry", 0),
            "error": llm_60m.get("error", 0),
            "skipped": llm_60m.get("skipped", 0),
            "in_queue": llm_60m.get("in_queue", 0),
            "running": llm_60m.get("running", 0),
            "total_processed": total_processed,
            "error_rate_pct": error_rate_60m,
        },
        "top_error_codes_60m": error_codes_60m,
        "pending_ready": pending_ready,
        "config": {
            "interval_seconds": settings.FASTPATH_INTERVAL_SECONDS,
            "lookback_minutes": settings.FASTPATH_LOOKBACK_MINUTES,
            "max_concurrent_jobs": settings.FASTPATH_MAX_CONCURRENT_JOBS,
        },
    }


@app.get("/dashboard/ops/fastpath_debug.json")
async def fastpath_debug_endpoint(
    request: Request,
    session: AsyncSession = Depends(get_async_session),
):
    """Debug endpoint to see skipped audits and their reasons."""
    _verify_debug_token(request)

    try:
        # Get skipped audits from last 60 min
        res = await session.execute(
            text("""
                SELECT
                    pma.id as audit_id,
                    po.match_id,
                    pma.llm_narrative_status,
                    pma.llm_narrative_error_code,
                    pma.llm_narrative_error_detail,
                    pma.created_at,
                    m.home_goals,
                    m.away_goals,
                    m.stats IS NOT NULL as has_stats,
                    m.stats_ready_at IS NOT NULL as stats_ready
                FROM post_match_audits pma
                JOIN prediction_outcomes po ON po.id = pma.outcome_id
                JOIN matches m ON m.id = po.match_id
                WHERE pma.created_at > NOW() - INTERVAL '60 minutes'
                  AND pma.llm_narrative_status = 'skipped'
                ORDER BY pma.created_at DESC
                LIMIT 20
            """)
        )
        skipped = []
        for r in res.fetchall():
            skipped.append({
                "audit_id": r[0],
                "match_id": r[1],
                "status": r[2],
                "error_code": r[3],
                "error_detail": r[4],
                "created_at": r[5].isoformat() if r[5] else None,
                "goals": f"{r[6]}-{r[7]}",
                "has_stats": r[8],
                "stats_ready": r[9],
            })

        # Get status breakdown
        res2 = await session.execute(
            text("""
                SELECT llm_narrative_status, COUNT(*)
                FROM post_match_audits
                WHERE created_at > NOW() - INTERVAL '60 minutes'
                GROUP BY llm_narrative_status
                ORDER BY COUNT(*) DESC
            """)
        )
        breakdown = {r[0]: r[1] for r in res2.fetchall()}

        return {
            "skipped_audits": skipped,
            "status_breakdown_60m": breakdown,
        }
    except Exception as e:
        logger.error(f"fastpath_debug error: {e}")
        return {"error": str(e)}


@app.get("/dashboard/ops/llm_audit/{match_id}.json")
async def llm_audit_endpoint(
    request: Request,
    match_id: int,
    session: AsyncSession = Depends(get_async_session),
):
    """
    Debug endpoint for LLM traceability.

    Returns the exact payload sent to Qwen for a specific match,
    allowing quick RCA for hallucination issues.
    """
    _verify_debug_token(request)

    try:
        # Get audit with traceability data
        res = await session.execute(
            text("""
                SELECT
                    pma.id as audit_id,
                    po.match_id,
                    pma.llm_prompt_version,
                    pma.llm_prompt_input_hash,
                    pma.llm_prompt_input_json,
                    pma.llm_output_raw,
                    pma.llm_validation_errors,
                    pma.llm_narrative_status,
                    pma.llm_narrative_error_code,
                    pma.llm_narrative_error_detail,
                    pma.llm_narrative_request_id,
                    pma.llm_narrative_json,
                    pma.llm_narrative_generated_at,
                    pma.llm_narrative_tokens_in,
                    pma.llm_narrative_tokens_out,
                    pma.llm_narrative_exec_ms,
                    pma.created_at
                FROM post_match_audits pma
                JOIN prediction_outcomes po ON po.id = pma.outcome_id
                WHERE po.match_id = :match_id
            """),
            {"match_id": match_id}
        )
        row = res.fetchone()

        if not row:
            return {"error": f"No audit found for match_id {match_id}"}

        return {
            "audit_id": row[0],
            "match_id": row[1],
            "prompt_version": row[2],
            "prompt_input_hash": row[3],
            "prompt_input_json": row[4],
            "output_raw_preview": row[5][:500] if row[5] else None,
            "output_raw_len": len(row[5]) if row[5] else 0,
            "validation_errors": row[6],
            "status": row[7],
            "error_code": row[8],
            "error_detail": row[9],
            "runpod_job_id": row[10],
            "narrative_json": row[11],
            "generated_at": row[12].isoformat() if row[12] else None,
            "tokens_in": row[13],
            "tokens_out": row[14],
            "exec_ms": row[15],
            "audit_created_at": row[16].isoformat() if row[16] else None,
        }
    except Exception as e:
        logger.error(f"llm_audit error for match {match_id}: {e}")
        return {"error": str(e)}


@app.get("/dashboard/ops/match_data.json")
async def match_data_debug_endpoint(
    request: Request,
    match_id: int = Query(...),
    session: AsyncSession = Depends(get_async_session),
):
    """Debug endpoint to see exact match_data sent to LLM."""
    _verify_debug_token(request)

    try:
        match = await session.get(Match, match_id)
        if not match:
            return {"error": f"Match {match_id} not found"}

        home_team = await session.get(Team, match.home_team_id)
        away_team = await session.get(Team, match.away_team_id)

        # Get prediction
        pred_result = await session.execute(
            select(Prediction)
            .where(Prediction.match_id == match_id)
            .order_by(Prediction.created_at.desc())
            .limit(1)
        )
        prediction = pred_result.scalar_one_or_none()

        # Build the exact match_data that would be sent to LLM
        probs = {}
        predicted_result = None
        confidence = None
        if prediction:
            probs = {
                "home": prediction.home_prob,
                "draw": prediction.draw_prob,
                "away": prediction.away_prob,
            }
            predicted_result = max(probs, key=probs.get)
            confidence = probs[predicted_result]

        home_goals = match.home_goals or 0
        away_goals = match.away_goals or 0
        if home_goals > away_goals:
            actual_result = "home"
        elif away_goals > home_goals:
            actual_result = "away"
        else:
            actual_result = "draw"

        match_data = {
            "match_id": match.id,
            "home_team": home_team.name if home_team else "Local",
            "away_team": away_team.name if away_team else "Visitante",
            "league_name": "",
            "date": match.date.isoformat() if match.date else "",
            "home_goals": home_goals,
            "away_goals": away_goals,
            "stats": match.stats or {},
            "events": match.events or [],
            "prediction": {
                "probabilities": probs,
                "predicted_result": predicted_result,
                "confidence": confidence,
                "correct": predicted_result == actual_result if predicted_result else None,
            },
            "market_odds": {
                "home": match.odds_home,
                "draw": match.odds_draw,
                "away": match.odds_away,
            } if match.odds_home else {},
        }

        return {
            "match_data_sent_to_llm": match_data,
            "raw_match_stats": match.stats,
            "stats_ready_at": match.stats_ready_at.isoformat() if match.stats_ready_at else None,
            "stats_last_checked_at": match.stats_last_checked_at.isoformat() if match.stats_last_checked_at else None,
        }
    except Exception as e:
        logger.error(f"match_data_debug error: {e}")
        return {"error": str(e)}


@app.get("/dashboard/ops/stats_rca.json")
async def stats_rca_endpoint(
    request: Request,
    match_id: int = Query(...),
    session: AsyncSession = Depends(get_async_session),
):
    """
    RCA endpoint: fetch stats from API-Football and show full diagnostic.
    Tests: API response, parsing, persistence.
    """
    _verify_debug_token(request)

    from app.etl.api_football import APIFootballProvider

    result = {
        "match_id": match_id,
        "steps": {},
        "diagnosis": None,
    }

    try:
        # Step 1: Get match from DB
        match = await session.get(Match, match_id)
        if not match:
            return {"error": f"Match {match_id} not found"}

        result["steps"]["1_match_info"] = {
            "id": match.id,
            "external_id": match.external_id,
            "stats_before": match.stats,
            "stats_ready_at": str(match.stats_ready_at) if match.stats_ready_at else None,
        }

        if not match.external_id:
            result["diagnosis"] = "NO_EXTERNAL_ID"
            return result

        # Step 2: Fetch from API-Football
        provider = APIFootballProvider()
        try:
            stats_data = await provider._rate_limited_request(
                "fixtures/statistics",
                {"fixture": match.external_id},
                entity="stats"
            )
            await provider.close()
        except Exception as api_err:
            result["steps"]["2_api_call"] = {"error": str(api_err)}
            result["diagnosis"] = "API_CALL_FAILED"
            return result

        response = stats_data.get("response", [])
        result["steps"]["2_api_call"] = {
            "raw_response_keys": list(stats_data.keys()),
            "response_len": len(response),
            "response_teams": [r.get("team", {}).get("name") for r in response] if response else [],
        }

        if len(response) < 2:
            result["diagnosis"] = "API_RESPONSE_EMPTY_OR_INCOMPLETE"
            result["steps"]["2_api_call"]["raw_response"] = stats_data
            return result

        # Step 3: Show raw statistics structure
        result["steps"]["3_raw_stats_structure"] = {
            "team_0_name": response[0].get("team", {}).get("name"),
            "team_0_statistics_count": len(response[0].get("statistics", [])),
            "team_0_statistics_sample": response[0].get("statistics", [])[:5],
            "team_1_name": response[1].get("team", {}).get("name"),
            "team_1_statistics_count": len(response[1].get("statistics", [])),
            "team_1_statistics_sample": response[1].get("statistics", [])[:5],
        }

        # Step 4: Parse stats using our key_map
        key_map = {
            "Ball Possession": "ball_possession",
            "Total Shots": "total_shots",
            "Shots on Goal": "shots_on_goal",
            "Shots off Goal": "shots_off_goal",
            "Blocked Shots": "blocked_shots",
            "Shots insidebox": "shots_insidebox",
            "Shots outsidebox": "shots_outsidebox",
            "Fouls": "fouls",
            "Corner Kicks": "corner_kicks",
            "Offsides": "offsides",
            "Yellow Cards": "yellow_cards",
            "Red Cards": "red_cards",
            "Goalkeeper Saves": "goalkeeper_saves",
            "Total passes": "total_passes",
            "Passes accurate": "passes_accurate",
            "Passes %": "passes_pct",
            "expected_goals": "expected_goals",
        }

        def parse_team_stats(stats_list):
            parsed = {}
            for stat in stats_list:
                stat_type = stat.get("type")
                stat_value = stat.get("value")
                if stat_type in key_map:
                    parsed[key_map[stat_type]] = stat_value
            return parsed

        home_stats = parse_team_stats(response[0].get("statistics", []))
        away_stats = parse_team_stats(response[1].get("statistics", []))

        result["steps"]["4_parsed_stats"] = {
            "home_stats": home_stats,
            "away_stats": away_stats,
            "home_keys": list(home_stats.keys()),
            "away_keys": list(away_stats.keys()),
        }

        # Step 5: Test persistence
        new_stats = {"home": home_stats, "away": away_stats}
        match.stats = new_stats
        await session.flush()
        await session.commit()

        # Step 6: Re-query to verify persistence
        await session.refresh(match)
        result["steps"]["5_persistence"] = {
            "stats_after_commit": match.stats,
            "persisted_successfully": match.stats is not None and match.stats != {},
        }

        if match.stats and match.stats != {}:
            result["diagnosis"] = "SUCCESS - Stats fetched, parsed, and persisted"
        else:
            result["diagnosis"] = "PERSISTENCE_FAILED - Stats were set but did not persist"

        return result

    except Exception as e:
        logger.error(f"stats_rca error: {e}", exc_info=True)
        result["diagnosis"] = f"EXCEPTION: {str(e)}"
        return result


@app.get("/dashboard/ops/bulk_stats_backfill.json")
async def bulk_stats_backfill_endpoint(
    request: Request,
    since_date: str = Query("2026-01-03", description="Start date YYYY-MM-DD"),
    limit: int = Query(50, description="Max matches to process per call"),
    dry_run: bool = Query(True, description="If true, only list matches without fetching"),
    session: AsyncSession = Depends(get_async_session),
):
    """
    Bulk backfill stats for all FT matches since a given date that are missing stats.
    Use dry_run=true first to see how many matches need backfill.
    """
    _verify_debug_token(request)

    from app.etl.api_football import APIFootballProvider
    from datetime import datetime
    import json as json_lib

    result = {
        "since_date": since_date,
        "dry_run": dry_run,
        "limit": limit,
        "matches_found": 0,
        "matches_processed": 0,
        "successes": [],
        "failures": [],
    }

    try:
        # Parse date string to date object
        from datetime import datetime as dt
        since_date_parsed = dt.strptime(since_date, "%Y-%m-%d").date()

        # Find all FT matches since date with missing stats
        res = await session.execute(text("""
            SELECT id, external_id, date, home_team_id, away_team_id
            FROM matches
            WHERE status IN ('FT', 'AET', 'PEN')
              AND date >= :since_date
              AND (stats IS NULL OR stats::text = '{}' OR stats::text = 'null')
            ORDER BY date ASC
            LIMIT :limit
        """), {"since_date": since_date_parsed, "limit": limit})

        matches = res.fetchall()
        result["matches_found"] = len(matches)

        if dry_run:
            result["matches_to_process"] = [
                {"id": m[0], "external_id": m[1], "date": str(m[2])}
                for m in matches
            ]
            # Also count total
            res_total = await session.execute(text("""
                SELECT COUNT(*) FROM matches
                WHERE status IN ('FT', 'AET', 'PEN')
                  AND date >= :since_date
                  AND (stats IS NULL OR stats::text = '{}' OR stats::text = 'null')
            """), {"since_date": since_date_parsed})
            result["total_missing"] = res_total.scalar()
            return result

        # Process matches
        provider = APIFootballProvider()
        key_map = {
            "Ball Possession": "ball_possession",
            "Total Shots": "total_shots",
            "Shots on Goal": "shots_on_goal",
            "Shots off Goal": "shots_off_goal",
            "Blocked Shots": "blocked_shots",
            "Shots insidebox": "shots_insidebox",
            "Shots outsidebox": "shots_outsidebox",
            "Fouls": "fouls",
            "Corner Kicks": "corner_kicks",
            "Offsides": "offsides",
            "Yellow Cards": "yellow_cards",
            "Red Cards": "red_cards",
            "Goalkeeper Saves": "goalkeeper_saves",
            "Total passes": "total_passes",
            "Passes accurate": "passes_accurate",
            "Passes %": "passes_pct",
            "expected_goals": "expected_goals",
        }

        def parse_team_stats(stats_list):
            parsed = {}
            for stat in stats_list:
                stat_type = stat.get("type")
                stat_value = stat.get("value")
                if stat_type in key_map:
                    parsed[key_map[stat_type]] = stat_value
            return parsed

        for match_row in matches:
            match_id, external_id, match_date, home_id, away_id = match_row

            if not external_id:
                result["failures"].append({"id": match_id, "reason": "NO_EXTERNAL_ID"})
                continue

            try:
                stats_data = await provider._rate_limited_request(
                    "fixtures/statistics",
                    {"fixture": external_id},
                    entity="stats"
                )
                response = stats_data.get("response", [])

                if len(response) < 2:
                    result["failures"].append({"id": match_id, "external_id": external_id, "reason": "API_EMPTY"})
                    continue

                home_stats = parse_team_stats(response[0].get("statistics", []))
                away_stats = parse_team_stats(response[1].get("statistics", []))

                new_stats = {"home": home_stats, "away": away_stats}

                # Update in DB
                await session.execute(
                    text("UPDATE matches SET stats = :stats, stats_ready_at = NOW() WHERE id = :id"),
                    {"stats": json_lib.dumps(new_stats), "id": match_id}
                )
                result["successes"].append({"id": match_id, "external_id": external_id})
                result["matches_processed"] += 1

            except Exception as e:
                result["failures"].append({"id": match_id, "external_id": external_id, "reason": str(e)})

        await provider.close()
        await session.commit()

        return result

    except Exception as e:
        logger.error(f"bulk_stats_backfill error: {e}", exc_info=True)
        return {"error": str(e)}


@app.get("/dashboard/ops/fetch_events.json")
async def fetch_events_endpoint(
    request: Request,
    match_id: int = Query(...),
    session: AsyncSession = Depends(get_async_session),
):
    """
    Fetch events from API-Football for a specific match and persist.
    Used for testing/verification.
    """
    _verify_debug_token(request)

    from app.etl.api_football import APIFootballProvider
    from app.llm.fastpath import FastPathService

    result = {
        "match_id": match_id,
        "events_before": None,
        "events_after": None,
        "api_response_count": 0,
        "diagnosis": None,
    }

    try:
        match = await session.get(Match, match_id)
        if not match:
            return {"error": f"Match {match_id} not found"}

        result["events_before"] = match.events
        result["external_id"] = match.external_id

        if not match.external_id:
            result["diagnosis"] = "NO_EXTERNAL_ID"
            return result

        # Fetch events from API-Football
        provider = APIFootballProvider()
        try:
            events_data = await provider._rate_limited_request(
                "fixtures/events",
                {"fixture": match.external_id},
                entity="events"
            )
            await provider.close()
        except Exception as api_err:
            result["diagnosis"] = f"API_CALL_FAILED: {api_err}"
            return result

        events_response = events_data.get("response", [])
        result["api_response_count"] = len(events_response)

        if not events_response:
            result["diagnosis"] = "API_RESPONSE_EMPTY"
            return result

        # Parse events using FastPathService method
        fastpath = FastPathService(session)
        parsed_events = fastpath._parse_events(events_response)
        result["parsed_events_count"] = len(parsed_events)
        result["parsed_events"] = parsed_events

        # Persist
        match.events = parsed_events
        await session.commit()
        await session.refresh(match)

        result["events_after"] = match.events
        result["diagnosis"] = "SUCCESS" if match.events else "PERSISTENCE_FAILED"

        return result

    except Exception as e:
        logger.error(f"fetch_events error: {e}", exc_info=True)
        result["diagnosis"] = f"EXCEPTION: {str(e)}"
        return result


@app.get("/dashboard/ops/audit_metrics.json")
async def audit_metrics_endpoint(
    request: Request,
    session: AsyncSession = Depends(get_async_session),
):
    """
    Audit endpoint: cross-check dashboard metrics with direct DB queries.
    Returns raw query results for manual verification.
    """
    _verify_debug_token(request)

    from sqlalchemy import text

    result = {
        "generated_at": datetime.utcnow().isoformat(),
        "audits": {},
    }

    # P0.1: fastpath_ticks verification
    try:
        # Last 5 ticks
        res = await session.execute(text("""
            SELECT tick_at, selected, refreshed, ready, enqueued, completed, errors, skipped
            FROM fastpath_ticks
            ORDER BY tick_at DESC
            LIMIT 5
        """))
        ticks = [{"tick_at": str(r[0]), "selected": r[1], "refreshed": r[2], "ready": r[3],
                  "enqueued": r[4], "completed": r[5], "errors": r[6], "skipped": r[7]}
                 for r in res.fetchall()]
        result["audits"]["fastpath_ticks_last_5"] = ticks

        # Tick count last hour
        res = await session.execute(text("""
            SELECT COUNT(*), COUNT(*) FILTER (WHERE selected > 0 OR enqueued > 0 OR completed > 0)
            FROM fastpath_ticks WHERE tick_at > NOW() - INTERVAL '1 hour'
        """))
        row = res.fetchone()
        result["audits"]["ticks_1h"] = {"total": row[0], "with_activity": row[1]}
    except Exception as e:
        result["audits"]["fastpath_ticks_error"] = str(e)

    # P0.1: pending_ready verification - sample 5 match_ids
    try:
        res = await session.execute(text("""
            SELECT m.id, m.status, m.stats_ready_at, m.finished_at, m.date,
                   COALESCE(m.finished_at, m.date) as effective_finished
            FROM matches m
            WHERE m.status IN ('FT', 'AET', 'PEN')
              AND COALESCE(m.finished_at, m.date) > NOW() - INTERVAL '180 minutes'
              AND m.stats_ready_at IS NOT NULL
              AND NOT EXISTS (
                  SELECT 1 FROM prediction_outcomes po
                  JOIN post_match_audits pma ON pma.outcome_id = po.id
                  WHERE po.match_id = m.id AND pma.llm_narrative_status = 'ok'
              )
            LIMIT 5
        """))
        pending = [{"match_id": r[0], "status": r[1], "stats_ready_at": str(r[2]) if r[2] else None,
                    "finished_at": str(r[3]) if r[3] else None, "date": str(r[4]) if r[4] else None}
                   for r in res.fetchall()]
        result["audits"]["pending_ready_sample"] = pending
        result["audits"]["pending_ready_count"] = len(pending)
    except Exception as e:
        result["audits"]["pending_ready_error"] = str(e)

    # P0.2: LLM status breakdown last 60m - direct query
    try:
        res = await session.execute(text("""
            SELECT llm_narrative_status, COUNT(*) as cnt
            FROM post_match_audits
            WHERE created_at > NOW() - INTERVAL '60 minutes'
              AND llm_narrative_status IS NOT NULL
            GROUP BY llm_narrative_status
        """))
        llm_breakdown = {r[0]: r[1] for r in res.fetchall()}
        result["audits"]["llm_60m_direct"] = llm_breakdown

        # Sample of audits with error
        res = await session.execute(text("""
            SELECT pma.id, pma.outcome_id, pma.llm_narrative_status, pma.llm_narrative_error_code, pma.created_at
            FROM post_match_audits pma
            WHERE pma.created_at > NOW() - INTERVAL '60 minutes'
              AND pma.llm_narrative_status = 'error'
            LIMIT 5
        """))
        errors = [{"audit_id": r[0], "outcome_id": r[1], "status": r[2], "error_code": r[3], "created_at": str(r[4])}
                  for r in res.fetchall()]
        result["audits"]["llm_errors_sample"] = errors
    except Exception as e:
        result["audits"]["llm_60m_error"] = str(e)

    # P1.1: predictions_health verification
    try:
        # Last prediction saved
        res = await session.execute(text("SELECT MAX(created_at) FROM predictions"))
        last_pred = res.scalar()
        result["audits"]["last_prediction_saved_at"] = str(last_pred) if last_pred else None

        # FT matches last 48h
        res = await session.execute(text("""
            SELECT COUNT(*) FROM matches
            WHERE status IN ('FT', 'AET', 'PEN')
              AND COALESCE(finished_at, date) > NOW() - INTERVAL '48 hours'
        """))
        ft_48h = res.scalar()
        result["audits"]["ft_matches_48h"] = ft_48h

        # FT matches missing prediction
        res = await session.execute(text("""
            SELECT COUNT(*) FROM matches m
            WHERE m.status IN ('FT', 'AET', 'PEN')
              AND COALESCE(m.finished_at, m.date) > NOW() - INTERVAL '48 hours'
              AND NOT EXISTS (SELECT 1 FROM predictions p WHERE p.match_id = m.id)
        """))
        missing = res.scalar()
        result["audits"]["ft_missing_prediction_48h"] = missing

        # Sample of missing
        res = await session.execute(text("""
            SELECT m.id, m.status, m.date, m.home_team_id, m.away_team_id
            FROM matches m
            WHERE m.status IN ('FT', 'AET', 'PEN')
              AND COALESCE(m.finished_at, m.date) > NOW() - INTERVAL '48 hours'
              AND NOT EXISTS (SELECT 1 FROM predictions p WHERE p.match_id = m.id)
            LIMIT 10
        """))
        missing_sample = [{"match_id": r[0], "status": r[1], "date": str(r[2]) if r[2] else None}
                          for r in res.fetchall()]
        result["audits"]["ft_missing_prediction_sample"] = missing_sample
    except Exception as e:
        result["audits"]["predictions_health_error"] = str(e)

    # P1.2: stats_backfill verification
    try:
        # Matches 72h with stats (stats is JSON, cast to text for comparison)
        res = await session.execute(text("""
            SELECT COUNT(*) FROM matches
            WHERE status IN ('FT', 'AET', 'PEN')
              AND COALESCE(finished_at, date) > NOW() - INTERVAL '72 hours'
              AND stats IS NOT NULL
              AND stats::text != '{}'
              AND stats::text != 'null'
        """))
        with_stats = res.scalar()
        result["audits"]["finished_72h_with_stats"] = with_stats

        # Matches 72h missing stats
        res = await session.execute(text("""
            SELECT COUNT(*) FROM matches
            WHERE status IN ('FT', 'AET', 'PEN')
              AND COALESCE(finished_at, date) > NOW() - INTERVAL '72 hours'
              AND (stats IS NULL OR stats::text = '{}' OR stats::text = 'null')
        """))
        missing_stats = res.scalar()
        result["audits"]["finished_72h_missing_stats"] = missing_stats

        # Sample missing stats
        res = await session.execute(text("""
            SELECT id, status, date, stats
            FROM matches
            WHERE status IN ('FT', 'AET', 'PEN')
              AND COALESCE(finished_at, date) > NOW() - INTERVAL '72 hours'
              AND (stats IS NULL OR stats::text = '{}' OR stats::text = 'null')
            LIMIT 10
        """))
        missing_sample = [{"match_id": r[0], "status": r[1], "date": str(r[2]) if r[2] else None,
                           "stats": r[3]} for r in res.fetchall()]
        result["audits"]["missing_stats_sample"] = missing_sample
    except Exception as e:
        result["audits"]["stats_backfill_error"] = str(e)

    # P0.3: Stats integrity check - matches with stats that might be overwritten
    try:
        res = await session.execute(text("""
            SELECT id, status, stats_ready_at, stats IS NOT NULL as has_stats,
                   events IS NOT NULL as has_events
            FROM matches
            WHERE status IN ('FT', 'AET', 'PEN')
              AND stats_ready_at IS NOT NULL
              AND stats IS NOT NULL AND stats::text != '{}'
            ORDER BY stats_ready_at DESC
            LIMIT 5
        """))
        integrity = [{"match_id": r[0], "status": r[1], "stats_ready_at": str(r[2]) if r[2] else None,
                      "has_stats": r[3], "has_events": r[4]} for r in res.fetchall()]
        result["audits"]["stats_integrity_sample"] = integrity
    except Exception as e:
        result["audits"]["stats_integrity_error"] = str(e)

    return result


@app.get("/dashboard/ops/predictions_performance.json")
async def predictions_performance_endpoint(
    request: Request,
    window_days: int = Query(default=7, ge=1, le=30),
    regenerate: bool = Query(default=False),
    session: AsyncSession = Depends(get_async_session),
):
    """
    Prediction performance report: proper probability metrics for model evaluation.

    Returns Brier score, log loss, calibration, and market comparison.
    Use this to distinguish variance from bugs.

    Args:
        window_days: 7 or 14 (default 7)
        regenerate: If True, generates fresh report instead of returning cached

    Auth: X-Dashboard-Token header required.
    """
    _verify_debug_token(request)

    from app.ml.performance_metrics import (
        generate_performance_report,
        get_latest_report,
        save_performance_report,
    )

    if regenerate:
        # Generate fresh report
        report = await generate_performance_report(session, window_days)
        await save_performance_report(session, report, window_days, source="api")
        return report

    # Try to get cached report
    cached = await get_latest_report(session, window_days)
    if cached:
        return cached

    # No cached report, generate one
    report = await generate_performance_report(session, window_days)
    await save_performance_report(session, report, window_days, source="api")
    return report


async def _load_ops_data() -> dict:
    """
    Ops dashboard: read-only aggregated metrics from DB + in-process state.
    Designed to be lightweight (few aggregated queries) and cached.
    """
    now = datetime.utcnow()

    # Budget status - fetch real API account status from API-Football
    budget_status: dict = {"status": "unavailable"}
    try:
        from app.etl.api_football import get_api_account_status

        budget_status = await get_api_account_status()
    except Exception as e:
        logger.warning(f"Could not fetch API account status: {e}")
        budget_status = {"status": "unavailable", "error": str(e)}

    # Observational metadata: API-Football daily budget refresh time (approx).
    # User-reported: ~4:00pm America/Los_Angeles. Best-effort only (ops UX).
    try:
        from zoneinfo import ZoneInfo

        tz_name = "America/Los_Angeles"
        reset_hour = 16
        reset_minute = 0

        now_utc = datetime.utcnow().replace(tzinfo=ZoneInfo("UTC"))
        now_la = now_utc.astimezone(ZoneInfo(tz_name))
        next_reset_la = now_la.replace(hour=reset_hour, minute=reset_minute, second=0, microsecond=0)
        if next_reset_la <= now_la:
            next_reset_la = next_reset_la + timedelta(days=1)
        next_reset_utc = next_reset_la.astimezone(ZoneInfo("UTC"))

        if not isinstance(budget_status, dict):
            budget_status = {"status": "unavailable"}

        budget_status.update(
            {
                "tokens_reset_tz": tz_name,
                "tokens_reset_local_time": f"{reset_hour:02d}:{reset_minute:02d}",
                "tokens_reset_at_la": next_reset_la.isoformat(),
                "tokens_reset_at_utc": next_reset_utc.isoformat(),
                "tokens_reset_note": "Observed daily refresh around 4:00pm America/Los_Angeles",
            }
        )
    except Exception:
        pass

    league_mode = os.environ.get("LEAGUE_MODE", "tracked").strip().lower()
    last_sync = get_last_sync_time()

    async with AsyncSessionLocal() as session:
        # Tracked leagues (distinct league_id)
        res = await session.execute(text("SELECT COUNT(DISTINCT league_id) FROM matches WHERE league_id IS NOT NULL"))
        tracked_leagues_count = int(res.scalar() or 0)

        # Upcoming matches (next 24h)
        res = await session.execute(
            text(
                """
                SELECT league_id, COUNT(*) AS upcoming
                FROM matches
                WHERE league_id IS NOT NULL
                  AND date >= NOW()
                  AND date < NOW() + INTERVAL '24 hours'
                GROUP BY league_id
                ORDER BY upcoming DESC
                LIMIT 20
                """
            )
        )
        upcoming_by_league = [{"league_id": int(r[0]), "upcoming_24h": int(r[1])} for r in res.fetchall()]

        # PIT snapshots (live, lineup_confirmed)
        res = await session.execute(
            text(
                """
                SELECT COUNT(*)
                FROM odds_snapshots
                WHERE snapshot_type = 'lineup_confirmed'
                  AND odds_freshness = 'live'
                  AND snapshot_at > NOW() - INTERVAL '60 minutes'
                """
            )
        )
        pit_live_60m = int(res.scalar() or 0)

        res = await session.execute(
            text(
                """
                SELECT COUNT(*)
                FROM odds_snapshots
                WHERE snapshot_type = 'lineup_confirmed'
                  AND odds_freshness = 'live'
                  AND snapshot_at > NOW() - INTERVAL '24 hours'
                """
            )
        )
        pit_live_24h = int(res.scalar() or 0)

        # ΔKO distribution (last 60m)
        res = await session.execute(
            text(
                """
                SELECT ROUND(delta_to_kickoff_seconds / 60.0) AS min_to_ko, COUNT(*) AS c
                FROM odds_snapshots
                WHERE snapshot_type = 'lineup_confirmed'
                  AND odds_freshness = 'live'
                  AND snapshot_at > NOW() - INTERVAL '60 minutes'
                  AND delta_to_kickoff_seconds IS NOT NULL
                GROUP BY 1
                ORDER BY 1
                """
            )
        )
        pit_dko_60m = [{"min_to_ko": int(r[0]), "count": int(r[1])} for r in res.fetchall()]

        # Latest PIT snapshots (last 10, any freshness)
        res = await session.execute(
            text(
                """
                SELECT os.snapshot_at, os.match_id, m.league_id, os.odds_freshness, os.delta_to_kickoff_seconds,
                       os.odds_home, os.odds_draw, os.odds_away, os.bookmaker
                FROM odds_snapshots os
                JOIN matches m ON m.id = os.match_id
                WHERE os.snapshot_type = 'lineup_confirmed'
                ORDER BY os.snapshot_at DESC
                LIMIT 10
                """
            )
        )
        latest_pit = []
        for r in res.fetchall():
            latest_pit.append(
                {
                    "snapshot_at": r[0].isoformat() if r[0] else None,
                    "match_id": int(r[1]) if r[1] is not None else None,
                    "league_id": int(r[2]) if r[2] is not None else None,
                    "odds_freshness": r[3],
                    "delta_to_kickoff_minutes": round(float(r[4]) / 60.0, 1) if r[4] is not None else None,
                    "odds": {
                        "home": float(r[5]) if r[5] is not None else None,
                        "draw": float(r[6]) if r[6] is not None else None,
                        "away": float(r[7]) if r[7] is not None else None,
                    },
                    "bookmaker": r[8],
                }
            )

        # Movement snapshots (last 24h)
        lineup_movement_24h = None
        market_movement_24h = None
        try:
            res = await session.execute(
                text(
                    """
                    SELECT COUNT(*)
                    FROM lineup_movement_snapshots
                    WHERE captured_at > NOW() - INTERVAL '24 hours'
                    """
                )
            )
            lineup_movement_24h = int(res.scalar() or 0)
        except Exception:
            lineup_movement_24h = None

        try:
            res = await session.execute(
                text(
                    """
                    SELECT COUNT(*)
                    FROM market_movement_snapshots
                    WHERE captured_at > NOW() - INTERVAL '24 hours'
                    """
                )
            )
            market_movement_24h = int(res.scalar() or 0)
        except Exception:
            market_movement_24h = None

        # Stats backfill health (last 72h finished matches)
        # Use COALESCE(finished_at, date) to get matches that FINISHED in last 72h
        # not matches that STARTED in last 72h (date is kickoff time)
        res = await session.execute(
            text(
                """
                SELECT
                    COUNT(*) FILTER (WHERE stats IS NOT NULL AND stats::text != '{}' AND stats::text != 'null') AS with_stats,
                    COUNT(*) FILTER (WHERE stats IS NULL OR stats::text = '{}' OR stats::text = 'null') AS missing_stats
                FROM matches
                WHERE status IN ('FT', 'AET', 'PEN')
                  AND COALESCE(finished_at, date) > NOW() - INTERVAL '72 hours'
                """
            )
        )
        row = res.first()
        stats_with = int(row[0] or 0) if row else 0
        stats_missing = int(row[1] or 0) if row else 0

        # =============================================================
        # PROGRESS METRICS (for re-test / Alpha readiness)
        # =============================================================
        # Configurable targets via env vars
        TARGET_PIT_SNAPSHOTS_30D = int(os.environ.get("TARGET_PIT_SNAPSHOTS_30D", "100"))
        TARGET_PIT_BETS_30D = int(os.environ.get("TARGET_PIT_BETS_30D", "100"))
        TARGET_BASELINE_COVERAGE_PCT = int(os.environ.get("TARGET_BASELINE_COVERAGE_PCT", "60"))

        # 1) PIT snapshots (30 days) - lineup_confirmed with live odds
        pit_snapshots_30d = 0
        try:
            res = await session.execute(
                text(
                    """
                    SELECT COUNT(*)
                    FROM odds_snapshots
                    WHERE snapshot_type = 'lineup_confirmed'
                      AND odds_freshness = 'live'
                      AND snapshot_at > NOW() - INTERVAL '30 days'
                    """
                )
            )
            pit_snapshots_30d = int(res.scalar() or 0)
        except Exception:
            pit_snapshots_30d = 0

        # 2) PIT with predictions as-of (evaluable bets, 30 days)
        # Count PIT snapshots that have a prediction created BEFORE the snapshot
        pit_bets_30d = 0
        try:
            res = await session.execute(
                text(
                    """
                    SELECT COUNT(DISTINCT os.id)
                    FROM odds_snapshots os
                    WHERE os.snapshot_type = 'lineup_confirmed'
                      AND os.odds_freshness = 'live'
                      AND os.snapshot_at > NOW() - INTERVAL '30 days'
                      AND EXISTS (
                          SELECT 1 FROM predictions p
                          WHERE p.match_id = os.match_id
                            AND p.created_at < os.snapshot_at
                      )
                    """
                )
            )
            pit_bets_30d = int(res.scalar() or 0)
        except Exception:
            pit_bets_30d = 0

        # 3) Baseline coverage (% of recent PIT matches with market_movement pre-KO)
        # This measures how many PIT snapshots have baseline odds for CLV proxy
        baseline_coverage_pct = 0
        pit_with_baseline = 0
        pit_total_for_baseline = 0
        try:
            res = await session.execute(
                text(
                    """
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
                          AND os.snapshot_at > NOW() - INTERVAL '30 days'
                    ) sub
                    """
                )
            )
            row = res.first()
            if row:
                pit_with_baseline = int(row[0] or 0)
                pit_total_for_baseline = int(row[1] or 0)
                if pit_total_for_baseline > 0:
                    baseline_coverage_pct = round((pit_with_baseline / pit_total_for_baseline) * 100, 1)
        except Exception:
            baseline_coverage_pct = 0
            pit_with_baseline = 0
            pit_total_for_baseline = 0

        progress_metrics = {
            "pit_snapshots_30d": pit_snapshots_30d,
            "target_pit_snapshots_30d": TARGET_PIT_SNAPSHOTS_30D,
            "pit_bets_30d": pit_bets_30d,
            "target_pit_bets_30d": TARGET_PIT_BETS_30D,
            "baseline_coverage_pct": baseline_coverage_pct,
            "pit_with_baseline": pit_with_baseline,
            "pit_total_for_baseline": pit_total_for_baseline,
            "target_baseline_coverage_pct": TARGET_BASELINE_COVERAGE_PCT,
            "ready_for_retest": (
                pit_bets_30d >= TARGET_PIT_BETS_30D and
                baseline_coverage_pct >= TARGET_BASELINE_COVERAGE_PCT
            ),
        }

        # =============================================================
        # PREDICTIONS HEALTH (P0 observability - detect scheduler issues)
        # =============================================================
        predictions_health = await _calculate_predictions_health(session)

        # =============================================================
        # FAST-PATH HEALTH (LLM narrative generation monitoring)
        # =============================================================
        fastpath_health = await _calculate_fastpath_health(session)

        # =============================================================
        # MODEL PERFORMANCE (7d probability metrics summary)
        # =============================================================
        model_performance = await _calculate_model_performance(session)

        # =============================================================
        # DATA QUALITY TELEMETRY (quarantine/taint/unmapped summary)
        # =============================================================
        telemetry_data = await _calculate_telemetry_summary(session)

        # =============================================================
        # SHADOW MODE (A/B model comparison monitoring)
        # =============================================================
        shadow_mode_data = await _calculate_shadow_mode_summary(session)

        # =============================================================
        # SENSOR B (LogReg L2 calibration diagnostics - INTERNAL ONLY)
        # =============================================================
        sensor_b_data = await _calculate_sensor_b_summary(session)

        # =============================================================
        # RERUN SERVING (DB-first canary for two-stage)
        # =============================================================
        rerun_serving_data = await _calculate_rerun_serving_summary(session)

        # =============================================================
        # JOBS HEALTH (P0 jobs monitoring - stats_backfill, odds_sync, fastpath)
        # =============================================================
        jobs_health_data = await _calculate_jobs_health_summary(session)

        # =============================================================
        # LLM COST (Gemini token usage from PostMatchAudit)
        # =============================================================
        llm_cost_data = {"provider": "gemini", "status": "unavailable"}
        try:
            # Rollback any previous failed transaction state
            await session.rollback()

            # Gemini pricing from settings (per 1M tokens)
            GEMINI_PRICE_IN = settings.GEMINI_PRICE_INPUT
            GEMINI_PRICE_OUT = settings.GEMINI_PRICE_OUTPUT

            # 24h metrics (includes 'ok' and legacy 'completed_sync' statuses)
            res_24h = await session.execute(
                text(
                    """
                    SELECT
                        COUNT(*) AS ok_count,
                        COALESCE(SUM(llm_narrative_tokens_in), 0) AS tokens_in,
                        COALESCE(SUM(llm_narrative_tokens_out), 0) AS tokens_out
                    FROM post_match_audits
                    WHERE llm_narrative_model LIKE 'gemini%'
                      AND llm_narrative_status IN ('ok', 'completed_sync')
                      AND created_at > NOW() - INTERVAL '24 hours'
                    """
                )
            )
            row_24h = res_24h.first()
            ok_24h = int(row_24h[0] or 0) if row_24h else 0
            tokens_in_24h = int(row_24h[1] or 0) if row_24h else 0
            tokens_out_24h = int(row_24h[2] or 0) if row_24h else 0
            cost_24h = (tokens_in_24h * GEMINI_PRICE_IN + tokens_out_24h * GEMINI_PRICE_OUT) / 1_000_000

            # 7d metrics
            res_7d = await session.execute(
                text(
                    """
                    SELECT
                        COUNT(*) AS ok_count,
                        COALESCE(SUM(llm_narrative_tokens_in), 0) AS tokens_in,
                        COALESCE(SUM(llm_narrative_tokens_out), 0) AS tokens_out
                    FROM post_match_audits
                    WHERE llm_narrative_model LIKE 'gemini%'
                      AND llm_narrative_status IN ('ok', 'completed_sync')
                      AND created_at > NOW() - INTERVAL '7 days'
                    """
                )
            )
            row_7d = res_7d.first()
            ok_7d = int(row_7d[0] or 0) if row_7d else 0
            tokens_in_7d = int(row_7d[1] or 0) if row_7d else 0
            tokens_out_7d = int(row_7d[2] or 0) if row_7d else 0
            cost_7d = (tokens_in_7d * GEMINI_PRICE_IN + tokens_out_7d * GEMINI_PRICE_OUT) / 1_000_000

            # Total accumulated cost (all time)
            res_total = await session.execute(
                text(
                    """
                    SELECT
                        COUNT(*) AS ok_count,
                        COALESCE(SUM(llm_narrative_tokens_in), 0) AS tokens_in,
                        COALESCE(SUM(llm_narrative_tokens_out), 0) AS tokens_out
                    FROM post_match_audits
                    WHERE llm_narrative_model LIKE 'gemini%'
                      AND llm_narrative_status IN ('ok', 'completed_sync')
                    """
                )
            )
            row_total = res_total.first()
            ok_total = int(row_total[0] or 0) if row_total else 0
            tokens_in_total = int(row_total[1] or 0) if row_total else 0
            tokens_out_total = int(row_total[2] or 0) if row_total else 0
            cost_total = (tokens_in_total * GEMINI_PRICE_IN + tokens_out_total * GEMINI_PRICE_OUT) / 1_000_000

            # Calculate avg cost per OK request
            avg_cost_per_ok = cost_24h / ok_24h if ok_24h > 0 else 0.0

            # Status: warn if cost_24h > $1 or avg_cost > $0.01
            status = "ok"
            if cost_24h > 1.0 or avg_cost_per_ok > 0.01:
                status = "warn"

            llm_cost_data = {
                "provider": "gemini",
                "cost_total_usd": round(cost_total, 2),
                "cost_24h_usd": round(cost_24h, 4),
                "cost_7d_usd": round(cost_7d, 4),
                "requests_ok_total": ok_total,
                "requests_ok_24h": ok_24h,
                "requests_ok_7d": ok_7d,
                "avg_cost_per_ok_24h": round(avg_cost_per_ok, 6),
                "tokens_in_24h": tokens_in_24h,
                "tokens_out_24h": tokens_out_24h,
                "tokens_in_7d": tokens_in_7d,
                "tokens_out_7d": tokens_out_7d,
                "tokens_in_total": tokens_in_total,
                "tokens_out_total": tokens_out_total,
                "status": status,
                "note": "Estimated from token usage. Gemini 2.0 Flash: $0.075/1M in, $0.30/1M out",
            }
        except Exception as e:
            logger.warning(f"Could not calculate LLM cost: {e}")
            llm_cost_data = {"provider": "gemini", "status": "error", "error": str(e)}

    # League names - comprehensive fallback for all known leagues
    # Includes EXTENDED_LEAGUES and other common leagues from API-Football
    LEAGUE_NAMES_FALLBACK: dict[int, str] = {
        1: "World Cup",
        2: "Champions League",
        3: "Europa League",
        4: "Euro",
        5: "Nations League",
        9: "Copa América",
        10: "Friendlies",
        11: "Sudamericana",
        13: "Libertadores",
        22: "Gold Cup",
        # Legacy: league_id=28 was previously (incorrectly) used for WCQ CONMEBOL in code.
        # In production DB it may contain SAFF Championship fixtures. Keep explicit to avoid confusion.
        28: "SAFF Championship (legacy)",
        # WC 2026 Qualifiers (correct API-Football league IDs)
        29: "WCQ CAF",
        30: "WCQ AFC",
        31: "WCQ CONCACAF",
        32: "WCQ UEFA",
        33: "WCQ OFC",
        34: "WCQ CONMEBOL",
        37: "WCQ Intercontinental Play-offs",
        39: "Premier League",
        45: "FA Cup",
        61: "Ligue 1",
        71: "Brazil Serie A",
        78: "Bundesliga",
        88: "Eredivisie",
        94: "Primeira Liga",
        128: "Argentina Primera",
        135: "Serie A",
        140: "La Liga",
        143: "Copa del Rey",
        203: "Super Lig",
        239: "Colombia Primera A",
        242: "Ecuador Liga Pro",
        250: "Paraguay Primera - Apertura",
        252: "Paraguay Primera - Clausura",
        253: "MLS",
        262: "Liga MX",
        265: "Chile Primera División",
        268: "Uruguay Primera - Apertura",
        270: "Uruguay Primera - Clausura",
        281: "Peru Primera División",
        299: "Venezuela Primera División",
        344: "Bolivia Primera División",
        848: "Conference League",
    }

    # Merge with COMPETITIONS (if available)
    league_name_by_id: dict[int, str] = LEAGUE_NAMES_FALLBACK.copy()
    try:
        for league_id, comp in (COMPETITIONS or {}).items():
            if league_id is not None and comp is not None:
                name = getattr(comp, "name", None)
                if name:
                    league_name_by_id[int(league_id)] = name
    except Exception:
        pass  # Keep fallback names

    for item in upcoming_by_league:
        lid = item["league_id"]
        item["league_name"] = league_name_by_id.get(lid)

    for item in latest_pit:
        lid = item.get("league_id")
        if isinstance(lid, int):
            item["league_name"] = league_name_by_id.get(lid)

    # =============================================================
    # COVERAGE BY LEAGUE (NS matches in next 48h with predictions/odds)
    # =============================================================
    coverage_by_league = []
    try:
        async with AsyncSessionLocal() as session:
            res = await session.execute(
                text(
                    """
                    SELECT
                        m.league_id,
                        COUNT(*) AS total_ns,
                        COUNT(p.id) AS with_prediction,
                        COUNT(m.odds_home) AS with_odds
                    FROM matches m
                    LEFT JOIN predictions p ON p.match_id = m.id
                    WHERE m.status = 'NS'
                      AND m.date >= NOW()
                      AND m.date < NOW() + INTERVAL '48 hours'
                      AND m.league_id IS NOT NULL
                    GROUP BY m.league_id
                    ORDER BY COUNT(*) DESC
                    LIMIT 15
                    """
                )
            )
            for row in res.fetchall():
                lid = int(row[0])
                total = int(row[1])
                with_pred = int(row[2])
                with_odds = int(row[3])
                coverage_by_league.append({
                    "league_id": lid,
                    "league_name": league_name_by_id.get(lid, f"League {lid}"),
                    "total_ns": total,
                    "with_prediction": with_pred,
                    "with_odds": with_odds,
                    "pred_pct": round(with_pred / total * 100, 1) if total > 0 else 0,
                    "odds_pct": round(with_odds / total * 100, 1) if total > 0 else 0,
                })
    except Exception as e:
        logger.warning(f"Could not calculate coverage by league: {e}")

    return {
        "generated_at": now.isoformat(),
        "league_mode": league_mode,
        "tracked_leagues_count": tracked_leagues_count,
        "last_sync_at": last_sync.isoformat() if last_sync else None,
        "budget": budget_status,
        "pit": {
            "live_60m": pit_live_60m,
            "live_24h": pit_live_24h,
            "delta_to_kickoff_60m": pit_dko_60m,
            "latest": latest_pit,
        },
        "movement": {
            "lineup_movement_24h": lineup_movement_24h,
            "market_movement_24h": market_movement_24h,
        },
        "stats_backfill": {
            "finished_72h_with_stats": stats_with,
            "finished_72h_missing_stats": stats_missing,
        },
        "upcoming": {
            "by_league_24h": upcoming_by_league,
        },
        "progress": progress_metrics,
        "predictions_health": predictions_health,
        "fastpath_health": fastpath_health,
        "model_performance": model_performance,
        "telemetry": telemetry_data,
        "llm_cost": llm_cost_data,
        "shadow_mode": shadow_mode_data,
        "sensor_b": sensor_b_data,
        "rerun_serving": rerun_serving_data,
        "jobs_health": jobs_health_data,
        "coverage_by_league": coverage_by_league,
    }


async def _get_cached_ops_data() -> dict:
    now = time.time()
    if _ops_dashboard_cache["data"] and (now - _ops_dashboard_cache["timestamp"]) < _ops_dashboard_cache["ttl"]:
        return _ops_dashboard_cache["data"]
    data = await _load_ops_data()
    _ops_dashboard_cache["data"] = data
    _ops_dashboard_cache["timestamp"] = now
    return data


def _format_timestamp_la(ts_str: str) -> str:
    """Convert UTC timestamp string to Los Angeles time in friendly format."""
    if not ts_str:
        return ""
    try:
        from zoneinfo import ZoneInfo
        # Parse UTC timestamp
        if "T" in ts_str:
            dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        else:
            dt = datetime.fromisoformat(ts_str)
        # If naive, assume UTC
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=ZoneInfo("UTC"))
        # Convert to LA time
        la_tz = ZoneInfo("America/Los_Angeles")
        dt_la = dt.astimezone(la_tz)
        return dt_la.strftime("%m/%d %H:%M PT")
    except Exception:
        # Fallback: just truncate to readable format
        return ts_str[:16].replace("T", " ") if len(ts_str) > 16 else ts_str


def _friendly_label(value: str) -> str:
    """Convert snake_case or technical values to friendly labels."""
    if not value:
        return ""
    # Known mappings
    LABELS = {
        "lineup_confirmed": "Lineup Confirmed",
        "pre_match": "Pre-Match",
        "live": "Live",
        "stale": "Stale",
        "unknown": "Unknown",
        "ok": "OK",
        "error": "Error",
        "unavailable": "Unavailable",
        "inactive": "Inactive",
    }
    if value in LABELS:
        return LABELS[value]
    # Fallback: convert snake_case to Title Case
    return value.replace("_", " ").title()


def _render_progress_bar(label: str, current: int, target: int, tooltip: str) -> str:
    """Render a progress bar HTML snippet for count-based metrics."""
    pct = min(100, round((current / target) * 100, 1)) if target > 0 else 0
    color = "rgba(34, 197, 94, 0.8)" if pct >= 100 else "rgba(59, 130, 246, 0.8)"
    return f"""
    <div style="background: var(--card); border: 1px solid var(--border); border-radius: 0.5rem; padding: 0.75rem;">
      <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.35rem;">
        <span style="font-size: 0.85rem; color: var(--text);">{label}<span class="info-icon" style="margin-left: 0.25rem;">i<span class="tooltip">{tooltip}</span></span></span>
        <span style="font-size: 0.85rem; font-weight: 600; color: var(--text);">{current} / {target}</span>
      </div>
      <div style="background: var(--border); border-radius: 0.25rem; height: 8px; overflow: hidden;">
        <div style="background: {color}; height: 100%; width: {pct}%; transition: width 0.3s;"></div>
      </div>
    </div>
    """


def _render_progress_bar_pct(label: str, current_pct: float, target_pct: int, with_val: int, total_val: int, tooltip: str) -> str:
    """Render a progress bar HTML snippet for percentage-based metrics."""
    pct = min(100, round((current_pct / target_pct) * 100, 1)) if target_pct > 0 else 0
    color = "rgba(34, 197, 94, 0.8)" if current_pct >= target_pct else "rgba(59, 130, 246, 0.8)"
    return f"""
    <div style="background: var(--card); border: 1px solid var(--border); border-radius: 0.5rem; padding: 0.75rem;">
      <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.35rem;">
        <span style="font-size: 0.85rem; color: var(--text);">{label}<span class="info-icon" style="margin-left: 0.25rem;">i<span class="tooltip">{tooltip}</span></span></span>
        <span style="font-size: 0.85rem; font-weight: 600; color: var(--text);">{current_pct}% ({with_val}/{total_val}) / {target_pct}%</span>
      </div>
      <div style="background: var(--border); border-radius: 0.25rem; height: 8px; overflow: hidden;">
        <div style="background: {color}; height: 100%; width: {pct}%; transition: width 0.3s;"></div>
      </div>
    </div>
    """


def _render_history_rows(history: list) -> str:
    """Render history table rows for the ops dashboard."""
    if not history:
        return "<tr><td colspan='6' style='text-align:center; color:var(--muted);'>— No historical data yet. Daily rollup runs at 09:05 UTC. —</td></tr>"

    def fmt(val, suffix=""):
        """Format value: show '—' for None/0, otherwise value with optional suffix."""
        if val is None:
            return "—"
        if isinstance(val, (int, float)) and val == 0:
            return "0" + suffix  # Show 0 explicitly (it's valid data)
        return f"{val}{suffix}"

    rows = ""
    for entry in history:
        day = entry.get("day", "—")
        p = entry.get("payload") or {}

        pit_live = p.get("pit_snapshots_live")
        bets_eval = p.get("pit_bets_evaluable")
        baseline_pct = p.get("baseline_coverage", {}).get("baseline_pct")
        market_total = p.get("market_movement", {}).get("total")
        note = p.get("note")

        bins = p.get("delta_ko_bins", {})
        bin_10_45 = bins.get("10-45")
        bin_45_90 = bins.get("45-90")

        # Show note indicator if present
        note_indicator = f' <span class="info-icon" style="font-size:0.7em;">i<span class="tooltip">{note}</span></span>' if note else ""

        # Format bins display
        bins_display = f"{fmt(bin_10_45)} / {fmt(bin_45_90)}" if bin_10_45 is not None or bin_45_90 is not None else "—"

        rows += f"""
        <tr>
            <td style="font-weight:500;">{day}{note_indicator}</td>
            <td style="text-align:center;">{fmt(pit_live)}</td>
            <td style="text-align:center;">{fmt(bets_eval)}</td>
            <td style="text-align:center;">{fmt(baseline_pct, '%')}</td>
            <td style="text-align:center;">{bins_display}</td>
            <td style="text-align:center;">{fmt(market_total)}</td>
        </tr>"""

    return rows


def _render_ops_dashboard_html(data: dict, history: list | None = None, audit_logs: list | None = None) -> str:
    # External URLs from settings (with fallbacks)
    grafana_base = settings.GRAFANA_BASE_URL or "https://grafana.com"
    github_repo = settings.GITHUB_REPO_URL or "https://github.com/capta1nfire/FutbolStats"

    # Helper to build Sentry search URL
    def sentry_url(query: str) -> str:
        if settings.SENTRY_ORG and settings.SENTRY_PROJECT_ID:
            return f"https://{settings.SENTRY_ORG}.sentry.io/issues/?project={settings.SENTRY_PROJECT_ID}&query={query}"
        return f"https://sentry.io/issues/?query={query}"

    # Sentry base URL for footer link
    sentry_base = f"https://{settings.SENTRY_ORG}.sentry.io/issues/?project={settings.SENTRY_PROJECT_ID}" if settings.SENTRY_ORG else "https://sentry.io"

    budget = data.get("budget") or {}
    budget_status = budget.get("status", "unknown")
    # New API account status fields
    budget_used = budget.get("requests_today") or budget.get("used")
    budget_limit = budget.get("requests_limit") or budget.get("budget")
    budget_remaining = budget.get("requests_remaining")
    budget_plan = budget.get("plan", "")
    budget_plan_end = budget.get("plan_end", "")
    budget_cached = budget.get("cached", False)
    budget_reset_time = budget.get("tokens_reset_local_time")
    budget_reset_tz = budget.get("tokens_reset_tz")
    budget_reset_at_la = budget.get("tokens_reset_at_la")

    pit = data.get("pit") or {}
    pit_60m = pit.get("live_60m", 0)
    pit_24h = pit.get("live_24h", 0)
    dko = pit.get("delta_to_kickoff_60m") or []
    latest = pit.get("latest") or []

    upcoming = (data.get("upcoming") or {}).get("by_league_24h") or []
    movement = data.get("movement") or {}
    stats = data.get("stats_backfill") or {}
    history = history or []
    progress = data.get("progress") or {}
    pred_health = data.get("predictions_health") or {}
    fp_health = data.get("fastpath_health") or {}
    fp_60m = fp_health.get("last_60m") or {}
    telemetry = data.get("telemetry") or {}
    telemetry_summary = telemetry.get("summary") or {}
    llm_cost = data.get("llm_cost") or {}
    shadow_mode = data.get("shadow_mode") or {}
    sensor_b = data.get("sensor_b") or {}
    jobs_health = data.get("jobs_health") or {}
    coverage_by_league = data.get("coverage_by_league") or []

    def budget_color() -> str:
        if budget_status in ("unavailable", "error"):
            return "yellow"
        if budget_status == "inactive":
            return "red"
        if isinstance(budget_used, int) and isinstance(budget_limit, int) and budget_limit > 0:
            pct = budget_used / budget_limit
            if pct >= 0.9:
                return "red"
            if pct >= 0.7:
                return "yellow"
            return "green"
        return "blue"

    def pred_health_color() -> str:
        status = pred_health.get("status", "unknown")
        if status == "red":
            return "red"
        if status == "warn":
            return "yellow"
        if status == "ok":
            return "green"
        return "blue"

    def fastpath_health_color() -> str:
        status = fp_health.get("status", "unknown")
        if status == "red":
            return "red"
        if status == "warn":
            return "yellow"
        if status == "ok":
            return "green"
        if status == "disabled":
            return "blue"
        return "blue"

    def telemetry_color() -> str:
        status = telemetry.get("status", "OK").upper()
        if status == "RED":
            return "red"
        if status == "WARN":
            return "yellow"
        if status == "OK":
            return "green"
        return "blue"

    def llm_cost_color() -> str:
        status = llm_cost.get("status", "unavailable")
        if status == "error":
            return "red"
        if status == "warn":
            return "yellow"
        if status == "ok":
            return "green"
        return "blue"

    def shadow_mode_color() -> str:
        rec = shadow_mode.get("recommendation") or {}
        status = rec.get("status", "DISABLED").upper()
        if status == "GO":
            return "green"
        if status == "NO_GO":
            return "red"
        if status == "HOLD":
            return "yellow"
        if status in ("NO_DATA", "DISABLED", "NOT_LOADED"):
            return "blue"
        return "blue"

    def sensor_b_color() -> str:
        """Color for Sensor B card based on state (Auditor-approved statuses)."""
        state = sensor_b.get("state", "DISABLED")
        if state == "DISABLED":
            return "blue"
        if state == "LEARNING":
            return "yellow"  # Collecting samples
        if state == "ERROR":
            return "red"
        if state == "TRACKING":
            return "green"   # Normal operation, no alarm
        if state == "SIGNAL_DETECTED":
            return "yellow"  # Attention: A may be stale
        if state == "OVERFITTING_SUSPECTED":
            return "blue"    # B is noise, A is fine
        return "blue"

    def jobs_health_color() -> str:
        """Color for Jobs Health card."""
        status = jobs_health.get("status", "unknown")
        if status == "red":
            return "red"
        if status == "warn":
            return "yellow"
        if status == "ok":
            return "green"
        return "blue"

    # Tables HTML
    upcoming_rows = ""
    for r in upcoming:
        lid = r.get("league_id")
        name = r.get("league_name") or "Unknown"
        upcoming_rows += f"<tr><td>{name} ({lid})</td><td>{r.get('upcoming_24h')}</td></tr>"
    if not upcoming_rows:
        upcoming_rows = "<tr><td colspan='2'>Sin partidos próximos en 24h</td></tr>"

    dko_rows = ""
    for r in dko:
        dko_rows += f"<tr><td>{r.get('min_to_ko')}</td><td>{r.get('count')}</td></tr>"
    if not dko_rows:
        dko_rows = "<tr><td colspan='2'>Sin PIT live en la última hora</td></tr>"

    latest_rows = ""
    for r in latest:
        lid = r.get("league_id")
        name = r.get("league_name") or "Unknown"
        odds = r.get("odds") or {}
        snapshot_time = _format_timestamp_la(r.get("snapshot_at") or "")
        freshness = _friendly_label(r.get("odds_freshness") or "")
        latest_rows += (
            "<tr>"
            f"<td>{snapshot_time}</td>"
            f"<td>{name} ({lid})</td>"
            f"<td>{freshness}</td>"
            f"<td>{r.get('delta_to_kickoff_minutes')}</td>"
            f"<td>{odds.get('home')}</td>"
            f"<td>{odds.get('draw')}</td>"
            f"<td>{odds.get('away')}</td>"
            f"<td>{r.get('bookmaker')}</td>"
            "</tr>"
        )
    if not latest_rows:
        latest_rows = "<tr><td colspan='8'>Sin snapshots PIT</td></tr>"

    # Audit log rows
    audit_logs = audit_logs or []
    audit_rows = ""
    for log in audit_logs:
        result_color = ""
        if log.get("result") == "ok":
            result_color = "color: var(--green);"
        elif log.get("result") == "error":
            result_color = "color: var(--red);"
        audit_time = _format_timestamp_la(log.get("created_at") or "")
        duration = f'{log.get("duration_ms")}ms' if log.get("duration_ms") else "—"
        audit_rows += (
            "<tr>"
            f"<td>{audit_time}</td>"
            f"<td>{log.get('action', '—')}</td>"
            f"<td>{log.get('actor_id', '—')[:8]}</td>"
            f"<td style='{result_color}'>{log.get('result', '—').upper()}</td>"
            f"<td>{log.get('result_summary', '') or '—'}</td>"
            f"<td>{duration}</td>"
            "</tr>"
        )
    if not audit_rows:
        audit_rows = "<tr><td colspan='6' style='text-align:center; color:var(--muted);'>— Sin acciones recientes —</td></tr>"

    # Coverage by league rows
    coverage_rows = ""
    for cov in coverage_by_league:
        pred_color = "color: var(--green);" if cov.get("pred_pct", 0) >= 90 else "color: var(--yellow);" if cov.get("pred_pct", 0) >= 70 else "color: var(--red);"
        odds_color = "color: var(--green);" if cov.get("odds_pct", 0) >= 90 else "color: var(--yellow);" if cov.get("odds_pct", 0) >= 50 else "color: var(--muted);"
        coverage_rows += (
            "<tr>"
            f"<td>{cov.get('league_name', 'Unknown')} ({cov.get('league_id')})</td>"
            f"<td>{cov.get('total_ns', 0)}</td>"
            f"<td style='{pred_color}'>{cov.get('with_prediction', 0)} ({cov.get('pred_pct', 0)}%)</td>"
            f"<td style='{odds_color}'>{cov.get('with_odds', 0)} ({cov.get('odds_pct', 0)}%)</td>"
            "</tr>"
        )
    if not coverage_rows:
        coverage_rows = "<tr><td colspan='4' style='text-align:center; color:var(--muted);'>— No hay partidos NS en próximas 48h —</td></tr>"

    html = f"""<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <meta http-equiv="refresh" content="60">
  <title>Ops Dashboard - FutbolStats</title>
  <!-- Sentry/Grafana URLs for deep-linking -->
  <!-- Sentry: https://sentry.io/organizations/YOUR_ORG/issues/?project=YOUR_PROJECT -->
  <!-- Grafana: https://YOUR_ORG.grafana.net/d/DASHBOARD_ID -->
  <style>
    :root {{
      --bg: #0f172a;
      --card: #1e293b;
      --border: #334155;
      --text: #e2e8f0;
      --muted: #94a3b8;
      --green: #22c55e;
      --yellow: #eab308;
      --red: #ef4444;
      --blue: #3b82f6;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, 'SF Pro', system-ui, sans-serif;
      background: var(--bg);
      color: var(--text);
      padding: 1.5rem;
      min-height: 100vh;
    }}
    .header {{
      display: flex;
      justify-content: space-between;
      align-items: flex-end;
      margin-bottom: 1.25rem;
      padding-bottom: 1rem;
      border-bottom: 1px solid var(--border);
      gap: 1rem;
    }}
    .header h1 {{ font-size: 1.5rem; font-weight: 650; }}
    .meta {{ color: var(--muted); font-size: 0.8rem; line-height: 1.3; text-align: right; }}
    .cards {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 1rem;
      margin-bottom: 1.25rem;
    }}
    .card {{
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 0.75rem;
      padding: 1.25rem;
    }}
    .card-label {{ font-size: 0.75rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.06em; }}
    .card-value {{ font-size: 2rem; font-weight: 800; margin-top: 0.5rem; }}
    .card-sub {{ margin-top: 0.4rem; font-size: 0.9rem; color: var(--muted); }}
    .card.green .card-value {{ color: var(--green); }}
    .card.yellow .card-value {{ color: var(--yellow); }}
    .card.red .card-value {{ color: var(--red); }}
    .card.blue .card-value {{ color: var(--blue); }}
    .tables {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(340px, 1fr));
      gap: 1rem;
    }}
    .table-card {{
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 0.75rem;
      overflow: hidden;
    }}
    .table-card h3 {{
      padding: 1rem;
      font-size: 0.9rem;
      font-weight: 650;
      border-bottom: 1px solid var(--border);
    }}
    table {{ width: 100%; border-collapse: collapse; font-size: 0.875rem; }}
    th, td {{ padding: 0.75rem 1rem; text-align: left; vertical-align: top; }}
    th {{ color: var(--muted); font-weight: 550; }}
    tr:not(:last-child) {{ border-bottom: 1px solid var(--border); }}
    .footer {{
      margin-top: 1.5rem;
      text-align: center;
      color: var(--muted);
      font-size: 0.8rem;
    }}
    a {{ color: var(--blue); text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
    .nav-tabs {{
      display: inline-flex;
      gap: 0.35rem;
      padding: 0.35rem;
      border: 1px solid var(--border);
      border-radius: 0.75rem;
      background: rgba(30, 41, 59, 0.55);
    }}
    .nav-tabs a {{
      display: inline-flex;
      align-items: center;
      padding: 0.35rem 0.6rem;
      border-radius: 0.6rem;
      color: var(--muted);
      font-size: 0.8rem;
      text-decoration: none;
      border: 1px solid transparent;
    }}
    .nav-tabs a:hover {{
      color: var(--text);
      border-color: rgba(59, 130, 246, 0.35);
      background: rgba(59, 130, 246, 0.12);
    }}
    .nav-tabs a.active {{
      color: var(--text);
      background: rgba(59, 130, 246, 0.18);
      border-color: rgba(59, 130, 246, 0.45);
    }}
    /* Tooltip styles */
    .info-icon {{
      display: inline-flex;
      align-items: center;
      justify-content: center;
      width: 16px;
      height: 16px;
      border-radius: 50%;
      background: var(--border);
      color: var(--muted);
      font-size: 10px;
      font-weight: 700;
      cursor: help;
      margin-left: 6px;
      position: relative;
      vertical-align: middle;
    }}
    .info-icon:hover {{
      background: var(--blue);
      color: white;
    }}
    .info-icon .tooltip {{
      visibility: hidden;
      opacity: 0;
      position: absolute;
      bottom: 125%;
      left: 50%;
      transform: translateX(-50%);
      background: #1a1a2e;
      color: var(--text);
      padding: 10px 12px;
      border-radius: 8px;
      font-size: 12px;
      font-weight: 400;
      width: 240px;
      text-align: left;
      line-height: 1.4;
      box-shadow: 0 4px 20px rgba(0,0,0,0.4);
      border: 1px solid var(--border);
      z-index: 100;
      transition: opacity 0.2s, visibility 0.2s;
      text-transform: none;
      letter-spacing: normal;
    }}
    .info-icon .tooltip::after {{
      content: "";
      position: absolute;
      top: 100%;
      left: 50%;
      transform: translateX(-50%);
      border: 6px solid transparent;
      border-top-color: #1a1a2e;
    }}
    .info-icon:hover .tooltip {{
      visibility: visible;
      opacity: 1;
    }}
    /* JSON Dropdown Menu */
    .json-dropdown {{
      position: relative;
      display: inline-block;
    }}
    .json-dropdown-btn {{
      display: inline-flex;
      align-items: center;
      padding: 0.35rem 0.6rem;
      border-radius: 0.6rem;
      color: var(--muted);
      font-size: 0.8rem;
      cursor: pointer;
      border: 1px solid transparent;
    }}
    .json-dropdown-btn:hover {{
      color: var(--text);
      border-color: rgba(59, 130, 246, 0.35);
      background: rgba(59, 130, 246, 0.12);
    }}
    .json-dropdown-content {{
      display: none;
      position: absolute;
      right: 0;
      top: 100%;
      min-width: 140px;
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 0.5rem;
      box-shadow: 0 4px 12px rgba(0,0,0,0.3);
      z-index: 100;
    }}
    .json-dropdown:hover .json-dropdown-content {{
      display: block;
    }}
    .json-dropdown-content a {{
      display: block;
      padding: 0.5rem 0.75rem;
      color: var(--muted);
      font-size: 0.8rem;
      text-decoration: none;
      border: none;
    }}
    .json-dropdown-content a:hover {{
      background: rgba(59, 130, 246, 0.12);
      color: var(--text);
    }}
    .json-dropdown-content a:first-child {{
      border-radius: 0.5rem 0.5rem 0 0;
    }}
    .copy-json-btn {{
      display: block;
      width: 100%;
      padding: 0.4rem 0.75rem;
      color: var(--muted);
      font-size: 0.75rem;
      text-align: left;
      background: rgba(59, 130, 246, 0.08);
      border: none;
      border-top: 1px solid var(--border);
      cursor: pointer;
    }}
    .copy-json-btn:hover {{
      background: rgba(59, 130, 246, 0.18);
      color: var(--text);
    }}
    .copy-json-btn:last-child {{
      border-radius: 0 0 0.5rem 0.5rem;
    }}
    /* Debug Pack button */
    .debug-pack-btn {{
      display: inline-flex;
      align-items: center;
      gap: 0.35rem;
      padding: 0.4rem 0.75rem;
      border-radius: 0.6rem;
      background: rgba(239, 68, 68, 0.15);
      border: 1px solid rgba(239, 68, 68, 0.4);
      color: #f87171;
      font-size: 0.8rem;
      cursor: pointer;
      transition: all 0.2s;
    }}
    .debug-pack-btn:hover {{
      background: rgba(239, 68, 68, 0.25);
      border-color: rgba(239, 68, 68, 0.6);
      color: #fca5a5;
    }}
    .debug-pack-btn.success {{
      background: rgba(34, 197, 94, 0.15);
      border-color: rgba(34, 197, 94, 0.4);
      color: var(--green);
    }}
    /* Controls section */
    .controls-section {{
      margin-top: 1.5rem;
      padding: 1rem;
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 0.75rem;
    }}
    .controls-section h3 {{
      font-size: 0.9rem;
      font-weight: 600;
      margin-bottom: 1rem;
      color: var(--text);
    }}
    .controls-grid {{
      display: flex;
      flex-wrap: wrap;
      gap: 0.75rem;
    }}
    .control-btn {{
      display: inline-flex;
      align-items: center;
      gap: 0.5rem;
      padding: 0.6rem 1rem;
      border-radius: 0.5rem;
      font-size: 0.85rem;
      cursor: pointer;
      border: 1px solid var(--border);
      background: var(--card);
      color: var(--text);
      transition: all 0.2s;
    }}
    .control-btn:hover {{
      border-color: var(--blue);
      background: rgba(59, 130, 246, 0.1);
    }}
    .control-btn:disabled {{
      opacity: 0.5;
      cursor: not-allowed;
    }}
    .control-btn.danger {{
      border-color: rgba(239, 68, 68, 0.4);
      color: #f87171;
    }}
    .control-btn.danger:hover {{
      background: rgba(239, 68, 68, 0.15);
      border-color: rgba(239, 68, 68, 0.6);
    }}
    .control-btn.success {{
      background: rgba(34, 197, 94, 0.15);
      border-color: rgba(34, 197, 94, 0.4);
      color: var(--green);
    }}
    .control-result {{
      font-size: 0.75rem;
      color: var(--muted);
      margin-top: 0.5rem;
    }}
    /* Confirmation modal */
    .confirm-overlay {{
      display: none;
      position: fixed;
      inset: 0;
      background: rgba(0,0,0,0.7);
      z-index: 1000;
      align-items: center;
      justify-content: center;
    }}
    .confirm-overlay.show {{
      display: flex;
    }}
    .confirm-modal {{
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 0.75rem;
      padding: 1.5rem;
      max-width: 400px;
      width: 90%;
    }}
    .confirm-modal h4 {{
      font-size: 1rem;
      margin-bottom: 0.5rem;
    }}
    .confirm-modal p {{
      color: var(--muted);
      font-size: 0.875rem;
      margin-bottom: 1rem;
    }}
    .confirm-modal .buttons {{
      display: flex;
      gap: 0.75rem;
      justify-content: flex-end;
    }}
    .confirm-modal .btn-cancel {{
      padding: 0.5rem 1rem;
      border-radius: 0.5rem;
      border: 1px solid var(--border);
      background: transparent;
      color: var(--muted);
      cursor: pointer;
    }}
    .confirm-modal .btn-confirm {{
      padding: 0.5rem 1rem;
      border-radius: 0.5rem;
      border: none;
      background: var(--blue);
      color: white;
      cursor: pointer;
    }}
    .confirm-modal .btn-confirm.danger {{
      background: var(--red);
    }}
    /* Countdown indicator */
    .refresh-countdown {{
      display: inline-flex;
      align-items: center;
      gap: 0.4rem;
      padding: 0.25rem 0.6rem;
      background: rgba(59, 130, 246, 0.12);
      border: 1px solid rgba(59, 130, 246, 0.25);
      border-radius: 0.5rem;
      font-size: 0.75rem;
      color: var(--muted);
    }}
    .refresh-countdown .dot {{
      width: 6px;
      height: 6px;
      background: var(--green);
      border-radius: 50%;
      animation: pulse 2s infinite;
    }}
    @keyframes pulse {{
      0%, 100% {{ opacity: 1; }}
      50% {{ opacity: 0.4; }}
    }}
    /* External links in cards */
    .card-links {{
      display: flex;
      gap: 0.75rem;
      margin-top: 0.35rem;
      font-size: 0.75rem;
    }}
    .card-links a {{
      color: var(--muted);
      text-decoration: none;
      opacity: 0.8;
    }}
    .card-links a:hover {{
      color: var(--blue);
      opacity: 1;
    }}
  </style>
</head>
<body>
  <div class="header">
    <div>
      <h1>Ops Dashboard</h1>
      <div class="meta" style="text-align:left;">
        Generado: {data.get("generated_at")} UTC<br/>
        LEAGUE_MODE: {data.get("league_mode")} | Tracked leagues: {data.get("tracked_leagues_count")}<br/>
        Last live sync: {data.get("last_sync_at")}
      </div>
    </div>
    <div class="meta">
      API: {budget_status} | Plan: {budget_plan or "N/A"} | Expires: {budget_plan_end[:10] if budget_plan_end else "N/A"}<br/>
      <div style="margin-top: 0.35rem;">
        <div class="nav-tabs">
          <a class="nav-link active" data-path="/dashboard/ops" href="/dashboard/ops">Ops</a>
          <a class="nav-link" data-path="/dashboard/pit" href="/dashboard/pit">PIT</a>
          <a class="nav-link" data-path="/dashboard/ops/history" href="/dashboard/ops/history">History</a>
          <a class="nav-link" data-path="/dashboard/ops/logs" href="/dashboard/ops/logs">Logs (debug)</a>
          <a class="nav-link" href="/ops/logout" style="margin-left: auto; color: #f87171;">Logout</a>
          <button class="debug-pack-btn" id="debugPackBtn">📦 Copy Debug Pack</button>
          <div class="json-dropdown">
            <span class="json-dropdown-btn">JSON ▾</span>
            <div class="json-dropdown-content">
              <a data-path="/dashboard/ops.json" href="/dashboard/ops.json" target="_blank">Ops JSON</a>
              <button class="copy-json-btn" data-endpoint="/dashboard/ops.json">📋 Copy Ops</button>
              <a data-path="/dashboard/pit.json" href="/dashboard/pit.json" target="_blank">PIT JSON</a>
              <button class="copy-json-btn" data-endpoint="/dashboard/pit.json">📋 Copy PIT</button>
              <a data-path="/dashboard/ops/history.json?days=30" href="/dashboard/ops/history.json?days=30" target="_blank">History JSON</a>
              <button class="copy-json-btn" data-endpoint="/dashboard/ops/history.json?days=30">📋 Copy History</button>
              <a data-path="/dashboard/ops/logs.json?limit=200" href="/dashboard/ops/logs.json?limit=200" target="_blank">Logs JSON</a>
              <button class="copy-json-btn" data-endpoint="/dashboard/ops/logs.json?limit=200">📋 Copy Logs</button>
              <a data-path="/dashboard/ops/progress_snapshots.json" href="/dashboard/ops/progress_snapshots.json" target="_blank">Alpha Snapshots</a>
              <button class="copy-json-btn" data-endpoint="/dashboard/ops/progress_snapshots.json">📋 Copy Alpha</button>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>

  <div class="cards">
    <div class="card blue">
      <div class="card-label">PIT Live (60 min)<span class="info-icon">i<span class="tooltip">Point-In-Time: Snapshots de odds capturados en el momento exacto que se confirman las alineaciones. Mide cuántos partidos tuvieron captura de odds "live" en la última hora.</span></span></div>
      <div class="card-value">{pit_60m}</div>
      <div class="card-sub">Lineup Confirmed + Live</div>
    </div>
    <div class="card blue">
      <div class="card-label">PIT Live (24 h)<span class="info-icon">i<span class="tooltip">Volumen total de snapshots PIT capturados en las últimas 24 horas. Indica la actividad general del sistema de monitoreo de odds.</span></span></div>
      <div class="card-value">{pit_24h}</div>
      <div class="card-sub">Volumen último día</div>
    </div>
    <div class="card {budget_color()}">
      <div class="card-label">API Budget{f" ({budget_plan})" if budget_plan else ""}<span class="info-icon">i<span class="tooltip">Consumo de la API de API-Football. Muestra requests usados hoy vs límite diario. Verde: &lt;70%, Amarillo: 70-90%, Rojo: &gt;90%.</span></span></div>
      <div class="card-value">{f"{budget_used:,}" if budget_used is not None else "?"} / {f"{budget_limit:,}" if budget_limit is not None else "?"}</div>
      <div class="card-sub">
        {f"{budget_remaining:,} remaining" if budget_remaining is not None else budget_status}{" (cached)" if budget_cached else ""}
        {f"<br/>Resets: 4:00 PM (Los Angeles)" if budget_reset_time else ""}
      </div>
    </div>
    <div class="card">
      <div class="card-label">Movimiento (24 h)<span class="info-icon">i<span class="tooltip">Snapshots de movimiento de odds capturados. Lineup Movement: cambios cerca del anuncio de alineaciones. Market Movement: cambios pre-partido (T-60 a T-5 min).</span></span></div>
      <div class="card-value">{movement.get("lineup_movement_24h")}</div>
      <div class="card-sub">Lineup Movement (Market: {movement.get("market_movement_24h")})</div>
    </div>
    <div class="card">
      <div class="card-label">Stats FT (72 h)<span class="info-icon">i<span class="tooltip">Partidos finalizados (FT) en las últimas 72 horas que tienen estadísticas completas vs los que faltan. El backfill automático rellena los faltantes.</span></span></div>
      <div class="card-value">{stats.get("finished_72h_with_stats")}</div>
      <div class="card-sub">Faltan: {stats.get("finished_72h_missing_stats")}</div>
    </div>
    <div class="card {pred_health_color()}">
      <div class="card-label">Predictions Health<span class="info-icon">i<span class="tooltip">Estado del scheduler de predicciones. ROJO: No se guardan predicciones en &gt;48h o cobertura &lt;50%. AMARILLO: &gt;24h o cobertura &lt;80%. VERDE: OK. Si se pone rojo, el scheduler no está funcionando y se rompe audit/LLM.</span></span></div>
      <div class="card-value">{pred_health.get("status", "?").upper()}</div>
      <div class="card-sub">
        Preds 24h: {pred_health.get("predictions_saved_last_24h", 0)} |
        Coverage 48h: {pred_health.get("ft_coverage_pct", 0)}%
        {f"<br/>Missing FT: {pred_health.get('ft_matches_last_48h_missing_prediction', 0)}/{pred_health.get('ft_matches_last_48h', 0)}" if pred_health.get("status") != "ok" else ""}
        {f"<br/><small style='color:var(--red)'>{pred_health.get('status_reason', '')}</small>" if pred_health.get("status_reason") else ""}
        <div class="card-links">
          <a href="{sentry_url('predictions')}" target="_blank">Sentry</a>
        </div>
      </div>
    </div>
    <div class="card {fastpath_health_color()}">
      <div class="card-label">LLM Fast-Path<span class="info-icon">i<span class="tooltip">Genera narrativas LLM minutos despues de FT (no espera daily audit 08:00 UTC). ROJO: Sin tick en &gt;10 min o error_rate &gt;50%. AMARILLO: error_rate &gt;20% o muchos skipped. VERDE: OK. DISABLED: FASTPATH_ENABLED=false.</span></span></div>
      <div class="card-value">{fp_health.get("status", "?").upper()}</div>
      <div class="card-sub">
        {f"Last tick: {fp_health.get('minutes_since_tick', '?'):.1f} min ago" if fp_health.get('minutes_since_tick') else "No tick yet"}
        <br/>60m: {fp_60m.get('ok', 0)} ok, {fp_60m.get('error', 0)} err, {fp_60m.get('skipped', 0)} skip
        {f"<br/>Pending: {fp_health.get('pending_ready', 0)}" if fp_health.get('pending_ready', 0) > 0 else ""}
        {f"<br/><small style='color:var(--red)'>{fp_health.get('status_reason', '')}</small>" if fp_health.get("status_reason") else ""}
        <div class="card-links">
          <a href="{sentry_url('fastpath+OR+gemini+OR+llm')}" target="_blank">Sentry</a>
          <a href="https://aistudio.google.com/u/1/usage" target="_blank">Gemini</a>
        </div>
      </div>
    </div>
    <div class="card {telemetry_color()}">
      <div class="card-label">Data Quality<span class="info-icon">i<span class="tooltip">Telemetría de calidad de datos. ROJO: Hay odds en cuarentena o partidos tainted (datos no confiables). AMARILLO: Hay entidades sin mapear. VERDE: Todo OK. Los datos cuarentenados/tainted son excluidos del training.</span></span></div>
      <div class="card-value">{telemetry.get("status", "?").upper()}</div>
      <div class="card-sub">
        Quarantine: {telemetry_summary.get("quarantined_odds_24h", 0)} |
        Tainted: {telemetry_summary.get("tainted_matches_24h", 0)} |
        Unmapped: {telemetry_summary.get("unmapped_entities_24h", 0)}
        <div class="card-links">
          <a href="{sentry_url('telemetry+OR+quarantine')}" target="_blank">Sentry</a>
          <a href="{grafana_base}" target="_blank">Grafana</a>
        </div>
      </div>
    </div>
    <div class="card {llm_cost_color()}">
      <div class="card-label">LLM Cost ({settings.GEMINI_MODEL})<span class="info-icon">i<span class="tooltip">Costo total acumulado de {settings.GEMINI_MODEL}. Pricing: ${settings.GEMINI_PRICE_INPUT}/1M tokens entrada, ${settings.GEMINI_PRICE_OUTPUT}/1M tokens salida. AMARILLO: &gt;$1/día o &gt;$0.01/request.</span></span></div>
      <div class="card-value">${llm_cost.get("cost_total_usd", 0):.2f}</div>
      <div class="card-sub">
        24h: ${llm_cost.get("cost_24h_usd", 0):.4f} ({llm_cost.get("requests_ok_24h", 0)} req) | 7d: ${llm_cost.get("cost_7d_usd", 0):.4f}
        <br/>Total: {llm_cost.get("requests_ok_total", 0):,} req | {(llm_cost.get("tokens_in_total", 0) + llm_cost.get("tokens_out_total", 0)):,} tokens
        <br/><a href="https://aistudio.google.com/u/1/usage?project=gen-lang-client-0385923148&amp;tab=billing" target="_blank" style="font-size:0.75rem;">Gemini Console →</a>
      </div>
    </div>
    <div class="card {shadow_mode_color()}">
      <div class="card-label">Shadow Mode<span class="info-icon">i<span class="tooltip">A/B testing del modelo two-stage vs baseline. GO: shadow mejora Brier sin degradar accuracy. NO_GO: shadow degrada métricas. HOLD: comparable, seguir monitoreando. NO_DATA: faltan muestras evaluadas (esperando partidos FT).</span></span></div>
      <div class="card-value">{(shadow_mode.get("recommendation") or {}).get("status", "DISABLED")}</div>
      <div class="card-sub">
        {f'Awaiting FT: {(shadow_mode.get("counts") or {}).get("shadow_predictions_pending", 0)} | Evaluated: {(shadow_mode.get("counts") or {}).get("shadow_predictions_evaluated", 0)}' if shadow_mode.get("counts") else 'Disabled'}
        {f' | Errors 24h: {(shadow_mode.get("counts") or {}).get("shadow_errors_last_24h", 0)}' if (shadow_mode.get("counts") or {}).get("shadow_errors_last_24h", 0) > 0 else ''}
        {f'<br/>Δ Brier: {(shadow_mode.get("metrics") or {}).get("delta_brier", 0):+.4f} | Δ Acc: {(shadow_mode.get("metrics") or {}).get("delta_accuracy", 0)*100:+.1f}%' if shadow_mode.get("metrics") else ''}
        <br/><span style="font-size:0.7rem;">Last eval: {_format_timestamp_la((shadow_mode.get("state") or {}).get("last_evaluation_at") or "") or "—"} | Next: every {(shadow_mode.get("state") or {}).get("evaluation_job_interval_minutes", 30)}m</span>
        <br/><span style="font-size:0.75rem;">{(shadow_mode.get("recommendation") or {}).get("reason", "")[:55]}</span>
        <br/><a href="/model/shadow-report" target="_blank" style="font-size:0.75rem;">Full report →</a>
      </div>
    </div>
    <div class="card {sensor_b_color()}">
      <div class="card-label">Sensor B<span class="info-icon">i<span class="tooltip">LogReg L2 calibration diagnostics (INTERNO). LEARNING: recolectando samples. TRACKING: monitoreando (normal). SIGNAL_DETECTED: revisar Model A. OVERFITTING_SUSPECTED: sensor es ruido. Solo diagnóstico, NO afecta producción.</span></span></div>
      <div class="card-value">{sensor_b.get("state", "DISABLED")}</div>
      <div class="card-sub">
        {f'Evaluated: {sensor_b.get("samples_evaluated", 0)}/{sensor_b.get("min_samples", 50)} | Pending: {sensor_b.get("samples_pending", 0)}' if sensor_b.get("state") not in ("DISABLED", None) else 'Disabled (SENSOR_ENABLED=false)'}
        {f'<br/>Signal: {sensor_b.get("signal_score"):.3f} | Δ Brier: {sensor_b.get("delta_brier"):+.4f}' if sensor_b.get("signal_score") is not None else ''}
        {f'<br/>A Brier: {sensor_b.get("brier_a"):.4f} | B Brier: {sensor_b.get("brier_b"):.4f}' if sensor_b.get("brier_a") is not None else ''}
        {f'<br/><span style="font-size:0.7rem;">Last retrain: {_format_timestamp_la(sensor_b.get("last_retrain_at") or "") or "—"} | Every {sensor_b.get("retrain_interval_hours", 6)}h</span>' if sensor_b.get("state") not in ("DISABLED", None) else ''}
        {f'<br/><span style="font-size:0.75rem;">{sensor_b.get("reason", "")[:55]}</span>' if sensor_b.get("reason") else ''}
        <br/><a href="/model/sensor-report" target="_blank" style="font-size:0.75rem;">Full report →</a>
      </div>
    </div>
    <div class="card {'neutral' if sensor_b.get('accuracy_a_pct') is None else 'green' if (sensor_b.get('delta_accuracy_pct') or 0) >= 0 else 'yellow'}">
      <div class="card-label">Accuracy A vs B<span class="info-icon">i<span class="tooltip">Comparación de % acierto entre Model A (producción) y Sensor B (LogReg L2). Solo diagnóstico interno, basado en partidos FT evaluados. NO afecta producción.</span></span></div>
      <div class="card-value">{f"A: {sensor_b.get('accuracy_a_pct'):.1f}% | B: {sensor_b.get('accuracy_b_pct'):.1f}%" if sensor_b.get('accuracy_a_pct') is not None else sensor_b.get('state', 'NO_DATA')}</div>
      <div class="card-sub">
        {f"Δ: {sensor_b.get('delta_accuracy_pct'):+.1f}%" if sensor_b.get('delta_accuracy_pct') is not None else f"Evaluated: {sensor_b.get('samples_evaluated', 0)}/{sensor_b.get('min_samples', 50)}"}
        <br/>Evaluated: {sensor_b.get('samples_evaluated', 0)} (window: {sensor_b.get('window_days', 14)}d)
        <br/><span style="font-size:0.7rem;">{sensor_b.get('note', 'solo FT evaluados')}</span>
        <br/><a href="/model/sensor-report" target="_blank" style="font-size:0.75rem;">Full report →</a>
      </div>
    </div>
    <div class="card {jobs_health_color()}">
      <div class="card-label">Jobs Health (P0)<span class="info-icon">i<span class="tooltip">Estado de los 3 jobs P0 del scheduler. stats_backfill: fetch stats de partidos FT (cada 60min). odds_sync: sync odds 1X2 (cada 6h). fastpath: narrativas LLM (cada 2min). RED: job falló o backlog. UNKNOWN: awaiting first run after deploy.</span></span></div>
      <div class="card-value">{jobs_health.get("status", "?").upper()}</div>
      <div class="card-sub">
        Stats: {(jobs_health.get("stats_backfill") or {}).get("status", "?").upper()} | Odds: {(jobs_health.get("odds_sync") or {}).get("status", "?").upper()} | FP: {(jobs_health.get("fastpath") or {}).get("status", "?").upper()}
        <br/><span style="font-size:0.7rem;">Stats pending: {(jobs_health.get("stats_backfill") or {}).get("ft_pending", "?")} | FP backlog: {(jobs_health.get("fastpath") or {}).get("backlog_ready", "?")}</span>
        <br/><span style="font-size:0.7rem;">Stats last: {_format_timestamp_la((jobs_health.get("stats_backfill") or {}).get("last_success_at") or "") or "awaiting (runs every 60m)"}</span>
        <div class="card-links">
          <a href="{github_repo}/blob/main/docs/GRAFANA_ALERTS_CHECKLIST.md#p0-jobs-health-scheduler-jobs" target="_blank">Runbook</a>
          <a href="{sentry_url('scheduler')}" target="_blank">Sentry</a>
        </div>
      </div>
    </div>
  </div>

  <div class="tables">
    <div class="table-card">
      <h3>Próximos partidos (24h) por liga<span class="info-icon">i<span class="tooltip">Partidos programados en las próximas 24 horas, agrupados por liga. Estas son las ligas que el sistema está monitoreando activamente.</span></span></h3>
      <table>
        <thead><tr><th>Liga (ID)</th><th>Upcoming</th></tr></thead>
        <tbody>{upcoming_rows}</tbody>
      </table>
    </div>

    <div class="table-card">
      <h3>Coverage by League (48h)<span class="info-icon">i<span class="tooltip">Cobertura de predicciones y odds para partidos NS en las próximas 48 horas. VERDE: ≥90% pred / ≥90% odds. AMARILLO: ≥70% pred / ≥50% odds. ROJO: &lt;70% predicciones.</span></span></h3>
      <table>
        <thead><tr><th>Liga (ID)</th><th>NS</th><th>Predictions</th><th>Odds</th></tr></thead>
        <tbody>{coverage_rows}</tbody>
      </table>
    </div>

    <div class="table-card">
      <h3>ΔKO PIT Live (últimos 60 min)<span class="info-icon">i<span class="tooltip">Delta to Kickoff: Minutos antes del inicio del partido cuando se capturó el snapshot PIT. Valores negativos indican captura antes del kickoff. Ideal: -45 a -90 minutos.</span></span></h3>
      <table>
        <thead><tr><th>min_to_ko</th><th>count</th></tr></thead>
        <tbody>{dko_rows}</tbody>
      </table>
    </div>

    <div class="table-card">
      <h3>Progreso hacia Re-test / Alpha<span class="info-icon">i<span class="tooltip">Métricas de preparación para re-evaluar el modelo. Se recomienda re-test cuando: Bets ≥ 100 y Baseline Coverage ≥ 60%.</span></span></h3>
      <div style="padding: 0.75rem;">
        {_render_progress_bar(
            "PIT Snapshots (30d)",
            progress.get("pit_snapshots_30d", 0),
            progress.get("target_pit_snapshots_30d", 100),
            "Snapshots PIT (lineup_confirmed + live) capturados en los últimos 30 días."
        )}
        {_render_progress_bar(
            "Bets Evaluables (30d)",
            progress.get("pit_bets_30d", 0),
            progress.get("target_pit_bets_30d", 100),
            "PIT snapshots con predicción válida (created_at < snapshot_at). Listos para evaluar ROI."
        )}
        {_render_progress_bar_pct(
            "Baseline Coverage",
            progress.get("baseline_coverage_pct", 0),
            progress.get("target_baseline_coverage_pct", 60),
            progress.get("pit_with_baseline", 0),
            progress.get("pit_total_for_baseline", 0),
            "% de PIT snapshots con market_movement pre-kickoff (para CLV proxy)."
        )}
        <div style="margin-top: 0.75rem; padding: 0.6rem; background: {'rgba(34, 197, 94, 0.12)' if progress.get('ready_for_retest') else 'rgba(234, 179, 8, 0.12)'}; border: 1px solid {'rgba(34, 197, 94, 0.35)' if progress.get('ready_for_retest') else 'rgba(234, 179, 8, 0.35)'}; border-radius: 0.5rem; font-size: 0.8rem; color: var(--text);">
          {'✅ Listo para re-test' if progress.get('ready_for_retest') else '⏳ N bets ≥ ' + str(progress.get('target_pit_bets_30d', 100)) + ' y baseline ≥ ' + str(progress.get('target_baseline_coverage_pct', 60)) + '%'}
        </div>
      </div>
    </div>

    <div class="table-card">
      <h3>KPI Histórico (14 días)<span class="info-icon">i<span class="tooltip">Métricas diarias persistentes (día UTC 00:00-23:59). Rollup generado a las 09:05 UTC. Nota: los valores del día actual pueden diferir del "PIT Live 24h" que cuenta últimas 24 horas móviles.</span></span></h3>
      <table style="font-size: 0.8rem;">
        <thead>
          <tr>
            <th>Día</th>
            <th>PIT</th>
            <th>Bets</th>
            <th>Base%</th>
            <th>ΔKO</th>
            <th>Mov</th>
          </tr>
        </thead>
        <tbody>
          {_render_history_rows(history[:14])}
        </tbody>
      </table>
    </div>

    <div class="table-card" style="grid-column: 1 / -1;">
      <h3>Últimos 10 PIT (Lineup Confirmed)<span class="info-icon">i<span class="tooltip">Los 10 snapshots PIT más recientes donde se confirmaron alineaciones. Muestra: hora (PT), liga, frescura de odds, minutos al kickoff, y odds H/D/A del bookmaker.</span></span></h3>
      <table>
        <thead>
          <tr>
            <th>snapshot_at</th><th>liga</th><th>freshness</th><th>ΔKO(min)</th>
            <th>H</th><th>D</th><th>A</th><th>bookmaker</th>
          </tr>
        </thead>
        <tbody>{latest_rows}</tbody>
      </table>
    </div>

    <div class="table-card" style="grid-column: 1 / -1;">
      <h3>Recent Actions (Audit Log)<span class="info-icon">i<span class="tooltip">Registro de acciones manuales ejecutadas desde el dashboard OPS. Incluye: predictions_trigger, odds_sync, sync_window. Actor ID es hash del token usado.</span></span></h3>
      <table>
        <thead>
          <tr>
            <th>Timestamp</th><th>Action</th><th>Actor</th><th>Result</th><th>Summary</th><th>Duration</th>
          </tr>
        </thead>
        <tbody>{audit_rows}</tbody>
      </table>
    </div>
  </div>

  <div class="controls-section">
    <h3>Controls<span class="info-icon">i<span class="tooltip">Acciones manuales del dashboard. Cada acción requiere confirmación y se registra en el Audit Log. Usar con precaución.</span></span></h3>
    <div class="controls-grid">
      <button class="control-btn" id="triggerPredictionsBtn" data-action="predictions_trigger" data-endpoint="/dashboard/predictions/trigger" data-method="POST" data-confirm="Esto generará predicciones para todos los partidos NS sin predicción. Puede tomar varios segundos.">
        Trigger Predictions
      </button>
      <button class="control-btn" id="syncOddsBtn" data-action="odds_sync" data-endpoint="/dashboard/ops/odds_sync" data-method="POST" data-confirm="Esto sincronizará odds 1X2 de API-Football para partidos próximos. Consume budget de API.">
        Sync Odds
      </button>
    </div>
    <div id="controlResult" class="control-result"></div>
  </div>

  <!-- Confirmation Modal -->
  <div class="confirm-overlay" id="confirmOverlay">
    <div class="confirm-modal">
      <h4 id="confirmTitle">Confirmar acción</h4>
      <p id="confirmMessage">¿Estás seguro?</p>
      <div class="buttons">
        <button class="btn-cancel" id="confirmCancel">Cancelar</button>
        <button class="btn-confirm" id="confirmOk">Ejecutar</button>
      </div>
    </div>
  </div>

  <div class="footer">
    <span class="refresh-countdown"><span class="dot"></span> Auto-refresh in <span id="countdown">60</span>s</span>
    | Cache TTL: {_ops_dashboard_cache["ttl"]}s
    | <a href="{sentry_base}" target="_blank">Sentry</a>
    | <a href="{grafana_base}" target="_blank">Grafana</a>
  </div>

  <script>
    // Preserve ?token= across dashboard navigation (for convenience).
    // Prefer X-Dashboard-Token header in production.
    (function() {{
      const params = new URLSearchParams(window.location.search);
      const token = params.get('token');
      if (!token) return;
      document.querySelectorAll('a.nav-link, .json-dropdown-content a').forEach(a => {{
        const path = a.getAttribute('data-path');
        if (!path) return;
        const joiner = path.includes('?') ? '&' : '?';
        a.setAttribute('href', path + joiner + 'token=' + encodeURIComponent(token));
      }});
      // Update copy buttons with token
      document.querySelectorAll('.copy-json-btn').forEach(btn => {{
        const endpoint = btn.getAttribute('data-endpoint');
        if (!endpoint) return;
        const joiner = endpoint.includes('?') ? '&' : '?';
        btn.setAttribute('data-endpoint', endpoint + joiner + 'token=' + encodeURIComponent(token));
      }});
    }})();

    // Copy JSON to clipboard
    document.querySelectorAll('.copy-json-btn').forEach(btn => {{
      btn.addEventListener('click', async () => {{
        const endpoint = btn.getAttribute('data-endpoint');
        try {{
          const res = await fetch(endpoint);
          const json = await res.json();
          await navigator.clipboard.writeText(JSON.stringify(json, null, 2));
          const orig = btn.textContent;
          btn.textContent = '✅ Copied!';
          setTimeout(() => btn.textContent = orig, 1500);
        }} catch (e) {{
          btn.textContent = '❌ Error';
          setTimeout(() => btn.textContent = btn.textContent.replace('❌ Error', '📋'), 1500);
        }}
      }});
    }});

    // Copy Debug Pack (combines ops.json + pit.json + logs.json)
    document.getElementById('debugPackBtn')?.addEventListener('click', async () => {{
      const btn = document.getElementById('debugPackBtn');
      const orig = btn.textContent;
      btn.textContent = '⏳ Loading...';
      btn.disabled = true;

      try {{
        const params = new URLSearchParams(window.location.search);
        const token = params.get('token');
        const addToken = (url) => token ? url + (url.includes('?') ? '&' : '?') + 'token=' + encodeURIComponent(token) : url;

        // Fetch all endpoints in parallel
        const [opsRes, pitRes, logsRes] = await Promise.all([
          fetch(addToken('/dashboard/ops.json')),
          fetch(addToken('/dashboard/pit.json')),
          fetch(addToken('/dashboard/ops/logs.json?limit=50'))
        ]);

        const [opsJson, pitJson, logsJson] = await Promise.all([
          opsRes.json(),
          pitRes.json(),
          logsRes.json()
        ]);

        // Combine into debug pack
        const debugPack = {{
          _meta: {{
            generated_at: new Date().toISOString(),
            source: 'OPS Dashboard Debug Pack',
            version: '1.0'
          }},
          ops: opsJson,
          pit: pitJson,
          logs: logsJson
        }};

        await navigator.clipboard.writeText(JSON.stringify(debugPack, null, 2));
        btn.textContent = '✅ Copied!';
        btn.classList.add('success');
        setTimeout(() => {{
          btn.textContent = orig;
          btn.classList.remove('success');
          btn.disabled = false;
        }}, 2000);
      }} catch (e) {{
        console.error('Debug pack error:', e);
        btn.textContent = '❌ Error';
        setTimeout(() => {{
          btn.textContent = orig;
          btn.disabled = false;
        }}, 2000);
      }}
    }});

    // Controls with confirmation
    (function() {{
      const overlay = document.getElementById('confirmOverlay');
      const confirmTitle = document.getElementById('confirmTitle');
      const confirmMessage = document.getElementById('confirmMessage');
      const confirmOk = document.getElementById('confirmOk');
      const confirmCancel = document.getElementById('confirmCancel');
      const controlResult = document.getElementById('controlResult');
      let pendingAction = null;

      const params = new URLSearchParams(window.location.search);
      const token = params.get('token');

      // Setup control buttons
      document.querySelectorAll('.control-btn').forEach(btn => {{
        btn.addEventListener('click', () => {{
          const action = btn.getAttribute('data-action');
          const endpoint = btn.getAttribute('data-endpoint');
          const method = btn.getAttribute('data-method') || 'POST';
          const confirmMsg = btn.getAttribute('data-confirm') || '¿Ejecutar esta acción?';

          pendingAction = {{ btn, action, endpoint, method }};
          confirmTitle.textContent = `Confirmar: ${{action}}`;
          confirmMessage.textContent = confirmMsg;
          overlay.classList.add('show');
        }});
      }});

      // Cancel confirmation
      confirmCancel.addEventListener('click', () => {{
        overlay.classList.remove('show');
        pendingAction = null;
      }});

      // Clicking overlay background cancels
      overlay.addEventListener('click', (e) => {{
        if (e.target === overlay) {{
          overlay.classList.remove('show');
          pendingAction = null;
        }}
      }});

      // Execute confirmed action
      confirmOk.addEventListener('click', async () => {{
        if (!pendingAction) return;

        const {{ btn, action, endpoint, method }} = pendingAction;
        overlay.classList.remove('show');

        const orig = btn.textContent;
        btn.textContent = '⏳ Running...';
        btn.disabled = true;
        controlResult.textContent = '';

        try {{
          const url = token ? endpoint + (endpoint.includes('?') ? '&' : '?') + 'token=' + encodeURIComponent(token) : endpoint;
          const res = await fetch(url, {{
            method: method,
            headers: {{ 'Content-Type': 'application/json' }}
          }});

          const data = await res.json();

          if (res.ok) {{
            btn.textContent = '✅ Done';
            btn.classList.add('success');
            controlResult.textContent = `✅ ${{action}}: ${{data.message || JSON.stringify(data).slice(0, 100)}}`;
            controlResult.style.color = 'var(--green)';
          }} else {{
            btn.textContent = '❌ Error';
            controlResult.textContent = `❌ ${{action}}: ${{data.detail || data.error || 'Unknown error'}}`;
            controlResult.style.color = 'var(--red)';
          }}
        }} catch (e) {{
          btn.textContent = '❌ Error';
          controlResult.textContent = `❌ ${{action}}: ${{e.message}}`;
          controlResult.style.color = 'var(--red)';
        }}

        setTimeout(() => {{
          btn.textContent = orig;
          btn.classList.remove('success');
          btn.disabled = false;
        }}, 3000);

        pendingAction = null;
      }});
    }})();

    // Countdown timer for auto-refresh
    (function() {{
      let seconds = 60;
      const countdownEl = document.getElementById('countdown');
      if (!countdownEl) return;

      setInterval(() => {{
        seconds--;
        if (seconds < 0) seconds = 60;
        countdownEl.textContent = seconds;
      }}, 1000);
    }})();
  </script>
</body>
</html>"""
    return html


@app.get("/dashboard/ops")
async def ops_dashboard_html(request: Request):
    """
    Ops Dashboard - Monitoreo en vivo del backend (DB-backed).

    Auth: session cookie (web) or X-Dashboard-Token header (API).
    """
    from fastapi.responses import HTMLResponse, RedirectResponse

    if not _verify_dashboard_token(request):
        # Redirect to login for better UX (instead of 401)
        return RedirectResponse(url="/ops/login", status_code=302)

    data = await _get_cached_ops_data()

    # Fetch KPI history (last 14 days for dashboard display)
    history = await _get_ops_history(days=14)

    # Fetch recent audit logs for dashboard display
    audit_logs = []
    try:
        from app.ops.audit import get_recent_audit_logs
        async with AsyncSessionLocal() as session:
            audit_logs = await get_recent_audit_logs(session, limit=10)
    except Exception as e:
        logger.warning(f"Could not fetch audit logs: {e}")

    html = _render_ops_dashboard_html(data, history=history, audit_logs=audit_logs)
    return HTMLResponse(content=html)


@app.get("/dashboard/ops.json")
async def ops_dashboard_json(request: Request):
    if not _verify_dashboard_token(request):
        raise HTTPException(status_code=401, detail="Dashboard access requires valid token.")
    data = await _get_cached_ops_data()
    return {
        "data": data,
        "cache_age_seconds": round(time.time() - _ops_dashboard_cache["timestamp"], 1) if _ops_dashboard_cache["timestamp"] else None,
    }


@app.get("/dashboard/ops/logs.json")
async def ops_dashboard_logs_json(
    request: Request,
    limit: int = OPS_LOG_DEFAULT_LIMIT,
    since_minutes: int = OPS_LOG_DEFAULT_SINCE_MINUTES,
    level: Optional[str] = None,
    mode: Optional[str] = None,
):
    """Filtered in-memory ops logs (copy/paste friendly). Use mode=compact for grouped view."""
    if not _verify_dashboard_token(request):
        raise HTTPException(status_code=401, detail="Dashboard access requires valid token.")
    compact = mode == "compact"
    return {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "limit": limit,
        "since_minutes": since_minutes,
        "level": level,
        "mode": mode,
        "entries": _get_ops_logs(since_minutes=since_minutes, limit=limit, level=level, compact=compact),
    }


@app.get("/dashboard/ops/logs")
async def ops_dashboard_logs_html(
    request: Request,
    limit: int = OPS_LOG_DEFAULT_LIMIT,
    since_minutes: int = OPS_LOG_DEFAULT_SINCE_MINUTES,
    level: Optional[str] = None,
    mode: Optional[str] = None,
):
    """HTML view of filtered in-memory ops logs. Use mode=compact for grouped view."""
    from fastapi.responses import HTMLResponse

    if not _verify_dashboard_token(request):
        raise HTTPException(status_code=401, detail="Dashboard access requires valid token.")

    compact = mode == "compact"
    entries = _get_ops_logs(since_minutes=since_minutes, limit=limit, level=level, compact=compact)

    rows_html = ""
    if compact:
        # Compact mode: show count, first_ts, last_ts
        for e in entries:
            count = e.get("count", 1)
            count_badge = f"<span style='background: rgba(59,130,246,0.25); padding: 0.15rem 0.4rem; border-radius: 0.3rem; font-size: 0.8rem;'>×{count}</span>" if count > 1 else ""
            rows_html += (
                "<tr>"
                f"<td>{e.get('first_ts', '')[:19]}</td>"
                f"<td>{e.get('last_ts', '')[:19] if count > 1 else ''}</td>"
                f"<td>{count_badge}</td>"
                f"<td>{e.get('level')}</td>"
                f"<td style='white-space: pre-wrap;'>{e.get('message')}</td>"
                "</tr>"
            )
    else:
        for e in entries:
            rows_html += (
                "<tr>"
                f"<td>{e.get('ts_utc')}</td>"
                f"<td>{e.get('level')}</td>"
                f"<td>{e.get('logger')}</td>"
                f"<td style='white-space: pre-wrap;'>{e.get('message')}</td>"
                "</tr>"
            )
    if not rows_html:
        cols = 5 if compact else 4
        rows_html = f"<tr><td colspan='{cols}'>Sin eventos relevantes en el buffer (aún).</td></tr>"

    json_link = f"/dashboard/ops/logs.json?limit={limit}&since_minutes={since_minutes}" + (f"&level={level}" if level else "") + (f"&mode={mode}" if mode else "")

    html = f"""<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <meta http-equiv="refresh" content="60">
  <title>Ops Logs - FutbolStats</title>
  <style>
    :root {{
      --bg: #0f172a; --card: #1e293b; --border: #334155; --text: #e2e8f0; --muted: #94a3b8; --blue: #3b82f6;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{ font-family: -apple-system, BlinkMacSystemFont, system-ui, sans-serif; background: var(--bg); color: var(--text); padding: 1.25rem; }}
    .header {{ display:flex; justify-content:space-between; align-items:flex-end; gap:1rem; margin-bottom:1rem; border-bottom:1px solid var(--border); padding-bottom:0.75rem; }}
    .meta {{ color: var(--muted); font-size:0.85rem; line-height:1.3; text-align:right; }}
    a {{ color: var(--blue); text-decoration:none; }}
    a:hover {{ text-decoration:underline; }}
    .card {{ background: var(--card); border: 1px solid var(--border); border-radius: 0.75rem; overflow:hidden; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 0.9rem; }}
    th, td {{ padding: 0.75rem 0.9rem; text-align:left; vertical-align: top; }}
    th {{ color: var(--muted); font-weight: 600; }}
    tr:not(:last-child) {{ border-bottom: 1px solid var(--border); }}
    .toolbar {{ display:flex; justify-content:space-between; align-items:center; gap:1rem; margin: 0.75rem 0 1rem; }}
    .btn {{ background: rgba(59,130,246,0.15); border: 1px solid rgba(59,130,246,0.35); color: var(--text); padding: 0.5rem 0.75rem; border-radius: 0.6rem; cursor:pointer; }}
    .btn:hover {{ background: rgba(59,130,246,0.25); }}
    .hint {{ color: var(--muted); font-size: 0.85rem; }}
    .nav-tabs {{
      display: inline-flex;
      gap: 0.35rem;
      padding: 0.35rem;
      border: 1px solid var(--border);
      border-radius: 0.75rem;
      background: rgba(30, 41, 59, 0.55);
    }}
    .nav-tabs a {{
      display: inline-flex;
      align-items: center;
      padding: 0.35rem 0.6rem;
      border-radius: 0.6rem;
      color: var(--muted);
      font-size: 0.8rem;
      text-decoration: none;
      border: 1px solid transparent;
    }}
    .nav-tabs a:hover {{
      color: var(--text);
      border-color: rgba(59, 130, 246, 0.35);
      background: rgba(59, 130, 246, 0.12);
    }}
    .nav-tabs a.active {{
      color: var(--text);
      background: rgba(59, 130, 246, 0.18);
      border-color: rgba(59, 130, 246, 0.45);
    }}
    /* JSON Dropdown Menu */
    .json-dropdown {{
      position: relative;
      display: inline-block;
    }}
    .json-dropdown-btn {{
      display: inline-flex;
      align-items: center;
      padding: 0.35rem 0.6rem;
      border-radius: 0.6rem;
      color: var(--muted);
      font-size: 0.8rem;
      cursor: pointer;
      border: 1px solid transparent;
    }}
    .json-dropdown-btn:hover {{
      color: var(--text);
      border-color: rgba(59, 130, 246, 0.35);
      background: rgba(59, 130, 246, 0.12);
    }}
    .json-dropdown-content {{
      display: none;
      position: absolute;
      right: 0;
      top: 100%;
      min-width: 140px;
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 0.5rem;
      box-shadow: 0 4px 12px rgba(0,0,0,0.3);
      z-index: 100;
    }}
    .json-dropdown:hover .json-dropdown-content {{
      display: block;
    }}
    .json-dropdown-content a {{
      display: block;
      padding: 0.5rem 0.75rem;
      color: var(--muted);
      font-size: 0.8rem;
      text-decoration: none;
      border: none;
    }}
    .json-dropdown-content a:hover {{
      background: rgba(59, 130, 246, 0.12);
      color: var(--text);
    }}
    .json-dropdown-content a:first-child {{
      border-radius: 0.5rem 0.5rem 0 0;
    }}
    .copy-json-btn {{
      display: block;
      width: 100%;
      padding: 0.4rem 0.75rem;
      color: var(--muted);
      font-size: 0.75rem;
      text-align: left;
      background: rgba(59, 130, 246, 0.08);
      border: none;
      border-top: 1px solid var(--border);
      cursor: pointer;
    }}
    .copy-json-btn:hover {{
      background: rgba(59, 130, 246, 0.18);
      color: var(--text);
    }}
    .copy-json-btn:last-child {{
      border-radius: 0 0 0.5rem 0.5rem;
    }}
  </style>
</head>
<body>
  <div class="header">
    <div>
      <h2>Logs relevantes (Ops){' - Compact' if compact else ''}</h2>
      <div class="hint">Solo eventos de captura/sync/movement/budget/errores. Se excluye spam de apscheduler/httpx.</div>
    </div>
    <div class="meta">
      <div>since_minutes={since_minutes} | limit={limit} | level={level or 'INFO+'} | mode={mode or 'normal'}</div>
      <div style="margin-top: 0.35rem;">
        <div class="nav-tabs">
          <a class="nav-link" data-path="/dashboard/ops" href="/dashboard/ops">Ops</a>
          <a class="nav-link" data-path="/dashboard/pit" href="/dashboard/pit">PIT</a>
          <a class="nav-link active" data-path="/dashboard/ops/logs" href="/dashboard/ops/logs">Logs</a>
          <div class="json-dropdown">
            <span class="json-dropdown-btn">JSON ▾</span>
            <div class="json-dropdown-content">
              <a data-path="/dashboard/ops.json" href="/dashboard/ops.json" target="_blank">Ops JSON</a>
              <button class="copy-json-btn" data-endpoint="/dashboard/ops.json">📋 Copy Ops</button>
              <a data-path="/dashboard/pit.json" href="/dashboard/pit.json" target="_blank">PIT JSON</a>
              <button class="copy-json-btn" data-endpoint="/dashboard/pit.json">📋 Copy PIT</button>
              <a data-path="/dashboard/ops/history.json?days=30" href="/dashboard/ops/history.json?days=30" target="_blank">History JSON</a>
              <button class="copy-json-btn" data-endpoint="/dashboard/ops/history.json?days=30">📋 Copy History</button>
              <a data-path="/dashboard/ops/logs.json?limit=200" href="/dashboard/ops/logs.json?limit=200" target="_blank">Logs JSON</a>
              <button class="copy-json-btn" data-endpoint="/dashboard/ops/logs.json?limit=200">📋 Copy Logs</button>
              <a data-path="/dashboard/ops/progress_snapshots.json" href="/dashboard/ops/progress_snapshots.json" target="_blank">Alpha Snapshots</a>
              <button class="copy-json-btn" data-endpoint="/dashboard/ops/progress_snapshots.json">📋 Copy Alpha</button>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>

  <div class="toolbar">
    <button class="btn" onclick="copyLogs()">Copiar (últimos {len(entries)})</button>
    <span class="hint">
      {'<a href="?since_minutes=' + str(since_minutes) + '&limit=' + str(limit) + ('&level=' + level if level else '') + '">Normal</a>' if compact else '<a href="?since_minutes=' + str(since_minutes) + '&limit=' + str(limit) + ('&level=' + level if level else '') + '&mode=compact">Compact</a>'}
      | Tip: `?level=WARNING` o `?since_minutes=180`
    </span>
  </div>

  <div class="card">
    <table>
      <thead><tr>{'<th>first_ts</th><th>last_ts</th><th>count</th><th>level</th><th>message</th>' if compact else '<th>ts_utc</th><th>level</th><th>logger</th><th>message</th>'}</tr></thead>
      <tbody>{rows_html}</tbody>
    </table>
  </div>

  <script>
    function copyLogs() {{
      const rows = Array.from(document.querySelectorAll("tbody tr"));
      const lines = rows.map(r => Array.from(r.children).map(td => td.innerText).join(" | "));
      const text = lines.join("\\n");
      navigator.clipboard.writeText(text);
    }}

    // Preserve ?token= across dashboard navigation (for convenience).
    (function() {{
      const params = new URLSearchParams(window.location.search);
      const token = params.get('token');
      if (!token) return;
      document.querySelectorAll('a.nav-link, .json-dropdown-content a').forEach(a => {{
        const path = a.getAttribute('data-path');
        if (!path) return;
        // If data-path already has query params, append with &
        const joiner = path.includes('?') ? '&' : '?';
        a.setAttribute('href', path + joiner + 'token=' + encodeURIComponent(token));
      }});
      // Update copy buttons with token
      document.querySelectorAll('.copy-json-btn').forEach(btn => {{
        const endpoint = btn.getAttribute('data-endpoint');
        if (!endpoint) return;
        const joiner = endpoint.includes('?') ? '&' : '?';
        btn.setAttribute('data-endpoint', endpoint + joiner + 'token=' + encodeURIComponent(token));
      }});
    }})();

    // Copy JSON to clipboard
    document.querySelectorAll('.copy-json-btn').forEach(btn => {{
      btn.addEventListener('click', async () => {{
        const endpoint = btn.getAttribute('data-endpoint');
        try {{
          const res = await fetch(endpoint);
          const json = await res.json();
          await navigator.clipboard.writeText(JSON.stringify(json, null, 2));
          const orig = btn.textContent;
          btn.textContent = '✅ Copied!';
          setTimeout(() => btn.textContent = orig, 1500);
        }} catch (e) {{
          btn.textContent = '❌ Error';
          setTimeout(() => btn.textContent = btn.textContent.replace('❌ Error', '📋'), 1500);
        }}
      }});
    }});
  </script>
</body>
</html>"""
    return HTMLResponse(content=html)


# =============================================================================
# OPS ADMIN ENDPOINTS (manual triggers, protected by token)
# =============================================================================


@app.post("/dashboard/ops/rollup")
async def trigger_ops_rollup(request: Request):
    """
    Manually trigger the daily ops rollup job.

    Protected by dashboard token. Use for testing/validation.
    """
    if not _verify_dashboard_token(request):
        raise HTTPException(status_code=401, detail="Dashboard access requires valid token.")

    from app.scheduler import daily_ops_rollup

    result = await daily_ops_rollup()
    return {
        "status": "executed",
        "result": result,
    }


@app.post("/dashboard/ops/odds_sync")
async def trigger_odds_sync(request: Request):
    """
    Manually trigger the odds sync job for upcoming matches.

    Protected by dashboard token. Use for testing/validation or immediate sync.
    """
    if not _verify_dashboard_token(request):
        raise HTTPException(status_code=401, detail="Dashboard access requires valid token.")

    from app.scheduler import sync_odds_for_upcoming_matches
    from app.ops.audit import log_ops_action

    start_time = time.time()
    result = await sync_odds_for_upcoming_matches()
    duration_ms = int((time.time() - start_time) * 1000)

    # Audit log
    try:
        async with AsyncSessionLocal() as audit_session:
            await log_ops_action(
                session=audit_session,
                request=request,
                action="odds_sync",
                params=None,
                result="ok" if result.get("status") == "completed" else "error",
                result_detail={
                    "scanned": result.get("scanned", 0),
                    "updated": result.get("updated", 0),
                    "api_calls": result.get("api_calls", 0),
                },
                duration_ms=duration_ms,
            )
    except Exception as audit_err:
        logger.warning(f"Failed to log audit for odds_sync: {audit_err}")

    return {
        "status": "executed",
        "result": result,
    }


# =============================================================================
# PREDICTION RERUN (controlled two-stage model promotion)
# =============================================================================


class PredictionRerunRequest(BaseModel):
    """Request body for predictions rerun endpoint."""
    window_hours: int = Field(default=168, ge=24, le=336, description="Time window (24-336h)")
    dry_run: bool = Field(default=True, description="If True, compute stats but don't save")
    architecture: str = Field(default="two_stage", description="Target architecture")
    max_matches: int = Field(default=500, ge=1, le=1000, description="Max matches to rerun")
    notes: Optional[str] = Field(default=None, description="Optional notes for audit")


@app.post("/dashboard/ops/predictions_rerun")
async def predictions_rerun(request: Request, body: PredictionRerunRequest):
    """
    Manual re-prediction of NS matches with two-stage architecture.

    ONE-OFF operation for controlled model promotion. Requires:
    - Dashboard token authentication
    - dry_run=true first to review changes
    - Saves before/after stats to prediction_reruns table

    Rollback: Set is_active=false on the rerun record (or PREFER_RERUN_PREDICTIONS=false).
    """
    import uuid
    from sqlalchemy import text

    if not _verify_dashboard_token(request):
        raise HTTPException(status_code=401, detail="Dashboard access requires valid token.")

    from app.ml.shadow import get_shadow_engine, is_shadow_enabled
    from app.models import Prediction, PredictionRerun
    from app.features import FeatureEngineer

    settings = get_settings()
    run_id = uuid.uuid4()

    async with AsyncSessionLocal() as session:
        try:
            # 1. Validate shadow engine is available
            shadow_engine = get_shadow_engine()
            if not shadow_engine or not shadow_engine.is_loaded:
                return {
                    "status": "error",
                    "error": "Shadow engine (two-stage) not loaded. Train it first.",
                    "hint": "Set MODEL_SHADOW_ARCHITECTURE=two_stage and trigger shadow training."
                }

            # 2. Query NS matches in window
            result = await session.execute(text("""
                SELECT m.id, m.external_id, m.date, m.league_id,
                       m.odds_home, m.odds_draw, m.odds_away
                FROM matches m
                WHERE m.status = 'NS'
                  AND m.date >= NOW()
                  AND m.date <= NOW() + make_interval(hours => :window_hours)
                ORDER BY m.date ASC
                LIMIT :max_matches
            """), {
                "window_hours": body.window_hours,
                "max_matches": body.max_matches,
            })
            ns_matches = result.fetchall()

            if not ns_matches:
                return {
                    "status": "no_matches",
                    "message": f"No NS matches found in {body.window_hours}h window",
                    "run_id": str(run_id),
                }

            match_ids = [m[0] for m in ns_matches]
            matches_with_odds = sum(1 for m in ns_matches if m[4] is not None)

            # 3. Get BEFORE predictions (baseline)
            result = await session.execute(text("""
                SELECT p.match_id, p.model_version, p.home_prob, p.draw_prob, p.away_prob
                FROM predictions p
                WHERE p.match_id = ANY(:match_ids)
                  AND p.model_version = :version
            """), {
                "match_ids": match_ids,
                "version": settings.MODEL_VERSION,
            })
            before_preds = {row[0]: {"home": row[2], "draw": row[3], "away": row[4]} for row in result.fetchall()}

            # 4. Compute BEFORE stats
            before_stats = _compute_prediction_stats(before_preds, "before")

            # 5. Get features for NS matches
            feature_engineer = FeatureEngineer(session=session)
            df = await feature_engineer.get_upcoming_matches_features()

            # Filter to our NS matches
            df_ns = df[df["match_id"].isin(match_ids)].copy()

            if len(df_ns) == 0:
                return {
                    "status": "no_features",
                    "error": "Could not compute features for NS matches",
                    "matches_total": len(match_ids),
                }

            # 6. Generate AFTER predictions (two-stage)
            after_preds = {}
            after_probas = shadow_engine.predict_proba(df_ns)

            for idx, (_, row) in enumerate(df_ns.iterrows()):
                match_id = row["match_id"]
                after_preds[match_id] = {
                    "home": float(after_probas[idx][0]),
                    "draw": float(after_probas[idx][1]),
                    "away": float(after_probas[idx][2]),
                }

            # 7. Compute AFTER stats
            after_stats = _compute_prediction_stats(after_preds, "after")

            # 8. Compute top deltas (largest draw probability changes)
            top_deltas = []
            for match_id in after_preds:
                if match_id in before_preds:
                    delta_draw = after_preds[match_id]["draw"] - before_preds[match_id]["draw"]
                    top_deltas.append({
                        "match_id": match_id,
                        "delta_draw": round(delta_draw, 4),
                        "before": {k: round(v, 4) for k, v in before_preds[match_id].items()},
                        "after": {k: round(v, 4) for k, v in after_preds[match_id].items()},
                    })

            top_deltas.sort(key=lambda x: abs(x["delta_draw"]), reverse=True)
            top_deltas = top_deltas[:20]  # Keep top 20

            # 9. Build response
            response = {
                "status": "dry_run" if body.dry_run else "executed",
                "run_id": str(run_id),
                "window_hours": body.window_hours,
                "architecture_before": settings.MODEL_ARCHITECTURE,
                "architecture_after": body.architecture,
                "model_version_before": settings.MODEL_VERSION,
                "model_version_after": f"v1.1.0-{body.architecture}",
                "matches_total": len(match_ids),
                "matches_with_features": len(df_ns),
                "matches_with_odds": matches_with_odds,
                "matches_with_before_pred": len(before_preds),
                "stats_before": before_stats,
                "stats_after": after_stats,
                "top_deltas": top_deltas[:10],  # Show top 10 in response
            }

            # 10. If not dry_run, save to database
            if not body.dry_run:
                # Save new predictions with run_id
                saved = 0
                errors = 0
                model_version_after = f"v1.1.0-{body.architecture}"

                for match_id, probs in after_preds.items():
                    try:
                        # Insert new prediction (with different model_version, so no conflict)
                        await session.execute(text("""
                            INSERT INTO predictions (match_id, model_version, home_prob, draw_prob, away_prob, run_id, created_at)
                            VALUES (:match_id, :model_version, :home_prob, :draw_prob, :away_prob, :run_id, NOW())
                            ON CONFLICT (match_id, model_version)
                            DO UPDATE SET
                                home_prob = EXCLUDED.home_prob,
                                draw_prob = EXCLUDED.draw_prob,
                                away_prob = EXCLUDED.away_prob,
                                run_id = EXCLUDED.run_id,
                                created_at = NOW()
                        """), {
                            "match_id": match_id,
                            "model_version": model_version_after,
                            "home_prob": probs["home"],
                            "draw_prob": probs["draw"],
                            "away_prob": probs["away"],
                            "run_id": run_id,
                        })
                        saved += 1
                    except Exception as e:
                        errors += 1
                        logger.warning(f"Rerun: failed to save match {match_id}: {e}")

                # Create audit record
                rerun_record = PredictionRerun(
                    run_id=run_id,
                    run_type="manual_rerun",
                    window_hours=body.window_hours,
                    architecture_before=settings.MODEL_ARCHITECTURE,
                    architecture_after=body.architecture,
                    model_version_before=settings.MODEL_VERSION,
                    model_version_after=model_version_after,
                    matches_total=len(match_ids),
                    matches_with_odds=matches_with_odds,
                    stats_before=before_stats,
                    stats_after=after_stats,
                    top_deltas=top_deltas,
                    is_active=True,
                    triggered_by="dashboard_ops",
                    notes=body.notes,
                )
                session.add(rerun_record)
                await session.commit()

                response["saved"] = saved
                response["errors"] = errors
                response["audit_record_created"] = True

                logger.info(
                    f"[RERUN] Predictions rerun complete: run_id={run_id}, "
                    f"saved={saved}, errors={errors}, matches={len(match_ids)}"
                )

            return response

        except Exception as e:
            logger.error(f"[RERUN] Predictions rerun failed: {e}")
            raise HTTPException(status_code=500, detail=f"Rerun failed: {str(e)}")


def _compute_prediction_stats(preds: dict, label: str) -> dict:
    """Compute stats for a set of predictions."""
    if not preds:
        return {"n": 0, "label": label}

    home_probs = [p["home"] for p in preds.values()]
    draw_probs = [p["draw"] for p in preds.values()]
    away_probs = [p["away"] for p in preds.values()]

    # Max prob for each prediction (confidence)
    max_probs = [max(p["home"], p["draw"], p["away"]) for p in preds.values()]

    # Count picks
    picks = {"home": 0, "draw": 0, "away": 0}
    for p in preds.values():
        if p["home"] >= p["draw"] and p["home"] >= p["away"]:
            picks["home"] += 1
        elif p["draw"] >= p["home"] and p["draw"] >= p["away"]:
            picks["draw"] += 1
        else:
            picks["away"] += 1

    n = len(preds)
    return {
        "label": label,
        "n": n,
        "avg_p_home": round(sum(home_probs) / n, 4),
        "avg_p_draw": round(sum(draw_probs) / n, 4),
        "avg_p_away": round(sum(away_probs) / n, 4),
        "avg_p_max": round(sum(max_probs) / n, 4),
        "draw_share_pct": round(100.0 * picks["draw"] / n, 2),
        "home_picks": picks["home"],
        "draw_picks": picks["draw"],
        "away_picks": picks["away"],
    }


@app.post("/dashboard/ops/predictions_rerun_rollback")
async def predictions_rerun_rollback(request: Request, run_id: str):
    """
    Rollback a prediction rerun by setting is_active=False.

    This doesn't delete data - it just marks the rerun as inactive,
    so the serving logic will prefer baseline predictions.
    """
    import uuid
    from sqlalchemy import text

    if not _verify_dashboard_token(request):
        raise HTTPException(status_code=401, detail="Dashboard access requires valid token.")

    try:
        parsed_run_id = uuid.UUID(run_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid run_id format")

    async with AsyncSessionLocal() as session:
        result = await session.execute(text("""
            UPDATE prediction_reruns
            SET is_active = FALSE
            WHERE run_id = :run_id
            RETURNING id, run_type, matches_total
        """), {"run_id": parsed_run_id})

        row = result.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Rerun not found")

        await session.commit()

        logger.info(f"[RERUN] Rollback executed: run_id={run_id}")

        return {
            "status": "rolled_back",
            "run_id": run_id,
            "rerun_id": row[0],
            "run_type": row[1],
            "matches_affected": row[2],
            "message": "Rerun deactivated. Baseline predictions will be served.",
        }


@app.get("/dashboard/ops/predictions_reruns.json")
async def list_prediction_reruns(request: Request):
    """List all prediction reruns with their status."""
    from sqlalchemy import text

    if not _verify_dashboard_token(request):
        raise HTTPException(status_code=401, detail="Dashboard access requires valid token.")

    async with AsyncSessionLocal() as session:
        result = await session.execute(text("""
            SELECT
                run_id, run_type, window_hours,
                architecture_before, architecture_after,
                model_version_before, model_version_after,
                matches_total, matches_with_odds,
                stats_before, stats_after,
                is_active, triggered_by, notes,
                created_at, evaluated_at, evaluated_matches
            FROM prediction_reruns
            ORDER BY created_at DESC
            LIMIT 20
        """))

        reruns = []
        for row in result.fetchall():
            reruns.append({
                "run_id": str(row[0]),
                "run_type": row[1],
                "window_hours": row[2],
                "architecture_before": row[3],
                "architecture_after": row[4],
                "model_version_before": row[5],
                "model_version_after": row[6],
                "matches_total": row[7],
                "matches_with_odds": row[8],
                "draw_share_before": row[9].get("draw_share_pct") if row[9] else None,
                "draw_share_after": row[10].get("draw_share_pct") if row[10] else None,
                "is_active": row[11],
                "triggered_by": row[12],
                "notes": row[13],
                "created_at": row[14].isoformat() if row[14] else None,
                "evaluated_at": row[15].isoformat() if row[15] else None,
                "evaluated_matches": row[16],
            })

        return {"reruns": reruns, "count": len(reruns)}


# =============================================================================
# ALPHA PROGRESS SNAPSHOTS (track Re-test/Alpha evolution over time)
# =============================================================================


@app.post("/dashboard/ops/progress_snapshot")
async def capture_progress_snapshot(request: Request, milestone: str | None = None):
    """
    Capture current Alpha Progress state to DB for auditing.

    Creates a snapshot with: generated_at, league_mode, tracked_leagues_count,
    progress metrics, and budget subset.
    Protected by dashboard token.

    Optional query param:
    - milestone: Label for this capture (e.g., "baseline_0", "pit_75", "pit_100", "bets_100", "ready_true")
    """
    import os
    from app.models import AlphaProgressSnapshot

    if not _verify_dashboard_token(request):
        raise HTTPException(status_code=401, detail="Dashboard access requires valid token.")

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

    # Add milestone label if provided
    if milestone:
        payload["milestone"] = milestone

    # Get git commit SHA from env if available
    app_commit = os.environ.get("RAILWAY_GIT_COMMIT_SHA") or os.environ.get("GIT_COMMIT_SHA")

    # Save to DB
    async with AsyncSessionLocal() as session:
        snapshot = AlphaProgressSnapshot(
            payload=payload,
            source="dashboard_manual" if milestone else "dashboard_manual",
            app_commit=app_commit[:40] if app_commit else None,
        )
        session.add(snapshot)
        await session.commit()
        await session.refresh(snapshot)

        return {
            "status": "captured",
            "id": snapshot.id,
            "milestone": milestone,
            "captured_at": snapshot.captured_at.isoformat(),
            "source": snapshot.source,
            "app_commit": snapshot.app_commit,
        }


@app.get("/dashboard/ops/progress_snapshots.json")
async def get_progress_snapshots(request: Request, limit: int = 50):
    """
    Get historical Alpha Progress snapshots for auditing.

    Returns list of snapshots ordered by captured_at DESC (most recent first).
    Protected by dashboard token.
    """
    import json

    if not _verify_dashboard_token(request):
        raise HTTPException(status_code=401, detail="Dashboard access requires valid token.")

    async with AsyncSessionLocal() as session:
        result = await session.execute(text("""
            SELECT id, captured_at, payload, source, app_commit
            FROM alpha_progress_snapshots
            ORDER BY captured_at DESC
            LIMIT :limit
        """), {"limit": limit})

        rows = result.fetchall()
        snapshots = []
        for row in rows:
            payload = row[2]
            if isinstance(payload, str):
                payload = json.loads(payload)

            snapshots.append({
                "id": row[0],
                "captured_at": row[1].isoformat() if row[1] else None,
                "payload": payload,
                "source": row[3],
                "app_commit": row[4],
            })

        return {
            "count": len(snapshots),
            "limit": limit,
            "snapshots": snapshots,
        }


# =============================================================================
# OPS HISTORY ENDPOINTS (KPI rollups from ops_daily_rollups table)
# =============================================================================


async def _get_ops_history(days: int = 30) -> list[dict]:
    """Fetch recent daily rollups from ops_daily_rollups table."""
    import json

    async with AsyncSessionLocal() as session:
        result = await session.execute(text("""
            SELECT day, payload, updated_at
            FROM ops_daily_rollups
            ORDER BY day DESC
            LIMIT :days
        """), {"days": days})

        rows = result.fetchall()
        history = []
        for row in rows:
            day = row[0]
            payload = row[1]
            updated_at = row[2]

            # Parse payload if it's a string
            if isinstance(payload, str):
                payload = json.loads(payload)

            history.append({
                "day": str(day),
                "payload": payload,
                "updated_at": updated_at.isoformat() if updated_at else None,
            })

        return history


@app.get("/dashboard/ops/history.json")
async def ops_history_json(request: Request, days: int = 30):
    """JSON endpoint for historical daily KPIs."""
    if not _verify_dashboard_token(request):
        raise HTTPException(status_code=401, detail="Dashboard access requires valid token.")

    history = await _get_ops_history(days=days)
    return {
        "days_requested": days,
        "days_available": len(history),
        "history": history,
    }


@app.get("/dashboard/ops/history")
async def ops_history_html(request: Request, days: int = 30):
    """HTML view of historical daily KPIs."""
    from fastapi.responses import HTMLResponse

    if not _verify_dashboard_token(request):
        raise HTTPException(status_code=401, detail="Dashboard access requires valid token.")

    history = await _get_ops_history(days=days)

    # Build table rows
    rows_html = ""
    for entry in history:
        day = entry["day"]
        p = entry.get("payload") or {}

        pit_live = p.get("pit_snapshots_live", 0)
        bets_eval = p.get("pit_bets_evaluable", 0)
        baseline_pct = p.get("baseline_coverage", {}).get("baseline_pct", 0)
        market_total = p.get("market_movement", {}).get("total", 0)

        # Delta KO bins
        bins = p.get("delta_ko_bins", {})
        bin_10_45 = bins.get("10-45", 0)
        bin_45_90 = bins.get("45-90", 0)

        # Errors - handle None/missing values with "—"
        errors = p.get("errors_summary", {})
        err_429_critical = errors.get("api_429_critical") or 0
        err_429_full = errors.get("api_429_full") or 0
        err_429 = err_429_critical + err_429_full
        budget_pct = errors.get("budget_pct")

        # Format budget_pct: show "—" if None or missing
        budget_pct_display = f"{budget_pct}%" if budget_pct is not None else "—"

        rows_html += f"""
        <tr>
            <td>{day}</td>
            <td>{pit_live}</td>
            <td>{bets_eval}</td>
            <td>{baseline_pct}%</td>
            <td>{bin_10_45} / {bin_45_90}</td>
            <td>{market_total}</td>
            <td>{err_429 if err_429 > 0 else '—'}</td>
            <td>{budget_pct_display}</td>
        </tr>"""

    if not rows_html:
        rows_html = "<tr><td colspan='8'>No hay datos históricos aún. El rollup diario corre a las 09:05 UTC.</td></tr>"

    html = f"""<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <meta http-equiv="refresh" content="300">
  <title>Ops History - FutbolStats</title>
  <style>
    :root {{
      --bg: #0f172a; --card: #1e293b; --border: #334155; --text: #e2e8f0; --muted: #94a3b8; --blue: #3b82f6;
      --green: #22c55e; --yellow: #eab308; --red: #ef4444;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{ font-family: -apple-system, BlinkMacSystemFont, system-ui, sans-serif; background: var(--bg); color: var(--text); padding: 1.25rem; }}
    .header {{ display:flex; justify-content:space-between; align-items:flex-end; gap:1rem; margin-bottom:1rem; border-bottom:1px solid var(--border); padding-bottom:0.75rem; }}
    .meta {{ color: var(--muted); font-size:0.85rem; line-height:1.3; text-align:right; }}
    a {{ color: var(--blue); text-decoration:none; }}
    a:hover {{ text-decoration:underline; }}
    .card {{ background: var(--card); border: 1px solid var(--border); border-radius: 0.75rem; overflow:hidden; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 0.9rem; }}
    th, td {{ padding: 0.75rem 0.9rem; text-align:center; vertical-align: middle; }}
    th {{ color: var(--muted); font-weight: 600; }}
    td:first-child {{ text-align: left; font-weight: 500; }}
    tr:not(:last-child) {{ border-bottom: 1px solid var(--border); }}
    .nav-tabs {{
      display: inline-flex;
      gap: 0.35rem;
      padding: 0.35rem;
      border: 1px solid var(--border);
      border-radius: 0.75rem;
      background: rgba(30, 41, 59, 0.55);
    }}
    .nav-tabs a {{
      display: inline-flex;
      align-items: center;
      padding: 0.35rem 0.6rem;
      border-radius: 0.6rem;
      color: var(--muted);
      font-size: 0.8rem;
      text-decoration: none;
      border: 1px solid transparent;
    }}
    .nav-tabs a:hover {{
      color: var(--text);
      border-color: rgba(59, 130, 246, 0.35);
      background: rgba(59, 130, 246, 0.12);
    }}
    .nav-tabs a.active {{
      color: var(--text);
      background: rgba(59, 130, 246, 0.18);
      border-color: rgba(59, 130, 246, 0.45);
    }}
    .hint {{ color: var(--muted); font-size: 0.85rem; }}
    /* JSON Dropdown Menu */
    .json-dropdown {{
      position: relative;
      display: inline-block;
    }}
    .json-dropdown-btn {{
      display: inline-flex;
      align-items: center;
      padding: 0.35rem 0.6rem;
      border-radius: 0.6rem;
      color: var(--muted);
      font-size: 0.8rem;
      cursor: pointer;
      border: 1px solid transparent;
    }}
    .json-dropdown-btn:hover {{
      color: var(--text);
      border-color: rgba(59, 130, 246, 0.35);
      background: rgba(59, 130, 246, 0.12);
    }}
    .json-dropdown-content {{
      display: none;
      position: absolute;
      right: 0;
      top: 100%;
      min-width: 140px;
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 0.5rem;
      box-shadow: 0 4px 12px rgba(0,0,0,0.3);
      z-index: 100;
    }}
    .json-dropdown:hover .json-dropdown-content {{
      display: block;
    }}
    .json-dropdown-content a {{
      display: block;
      padding: 0.5rem 0.75rem;
      color: var(--muted);
      font-size: 0.8rem;
      text-decoration: none;
      border: none;
    }}
    .json-dropdown-content a:hover {{
      background: rgba(59, 130, 246, 0.12);
      color: var(--text);
    }}
    .json-dropdown-content a:first-child {{
      border-radius: 0.5rem 0.5rem 0 0;
    }}
    .copy-json-btn {{
      display: block;
      width: 100%;
      padding: 0.4rem 0.75rem;
      color: var(--muted);
      font-size: 0.75rem;
      text-align: left;
      background: rgba(59, 130, 246, 0.08);
      border: none;
      border-top: 1px solid var(--border);
      cursor: pointer;
    }}
    .copy-json-btn:hover {{
      background: rgba(59, 130, 246, 0.18);
      color: var(--text);
    }}
    .copy-json-btn:last-child {{
      border-radius: 0 0 0.5rem 0.5rem;
    }}
  </style>
</head>
<body>
  <div class="header">
    <div>
      <h2>KPI Histórico (últimos {days} días)</h2>
      <div class="hint">Métricas diarias persistentes desde ops_daily_rollups</div>
    </div>
    <div class="meta">
      <div>{len(history)} días con datos</div>
      <div style="margin-top: 0.35rem;">
        <div class="nav-tabs">
          <a class="nav-link" data-path="/dashboard/ops" href="/dashboard/ops">Ops</a>
          <a class="nav-link" data-path="/dashboard/pit" href="/dashboard/pit">PIT</a>
          <a class="nav-link active" data-path="/dashboard/ops/history" href="/dashboard/ops/history">History</a>
          <a class="nav-link" data-path="/dashboard/ops/logs" href="/dashboard/ops/logs">Logs</a>
          <div class="json-dropdown">
            <span class="json-dropdown-btn">JSON ▾</span>
            <div class="json-dropdown-content">
              <a data-path="/dashboard/ops.json" href="/dashboard/ops.json" target="_blank">Ops JSON</a>
              <button class="copy-json-btn" data-endpoint="/dashboard/ops.json">📋 Copy Ops</button>
              <a data-path="/dashboard/pit.json" href="/dashboard/pit.json" target="_blank">PIT JSON</a>
              <button class="copy-json-btn" data-endpoint="/dashboard/pit.json">📋 Copy PIT</button>
              <a data-path="/dashboard/ops/history.json?days=30" href="/dashboard/ops/history.json?days=30" target="_blank">History JSON</a>
              <button class="copy-json-btn" data-endpoint="/dashboard/ops/history.json?days=30">📋 Copy History</button>
              <a data-path="/dashboard/ops/logs.json?limit=200" href="/dashboard/ops/logs.json?limit=200" target="_blank">Logs JSON</a>
              <button class="copy-json-btn" data-endpoint="/dashboard/ops/logs.json?limit=200">📋 Copy Logs</button>
              <a data-path="/dashboard/ops/progress_snapshots.json" href="/dashboard/ops/progress_snapshots.json" target="_blank">Alpha Snapshots</a>
              <button class="copy-json-btn" data-endpoint="/dashboard/ops/progress_snapshots.json">📋 Copy Alpha</button>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>

  <div class="card">
    <table>
      <thead>
        <tr>
          <th>Día</th>
          <th>PIT Live</th>
          <th>Bets Eval</th>
          <th>Baseline %</th>
          <th>ΔKO (10-45 / 45-90)</th>
          <th>Mkt Mov</th>
          <th>429s</th>
          <th>Budget %</th>
        </tr>
      </thead>
      <tbody>
        {rows_html}
      </tbody>
    </table>
  </div>

  <script>
    // Preserve ?token= across dashboard navigation
    (function() {{
      const params = new URLSearchParams(window.location.search);
      const token = params.get('token');
      if (!token) return;
      document.querySelectorAll('a.nav-link, .json-dropdown-content a').forEach(a => {{
        const path = a.getAttribute('data-path');
        if (!path) return;
        const joiner = path.includes('?') ? '&' : '?';
        a.setAttribute('href', path + joiner + 'token=' + encodeURIComponent(token));
      }});
      // Update copy buttons with token
      document.querySelectorAll('.copy-json-btn').forEach(btn => {{
        const endpoint = btn.getAttribute('data-endpoint');
        if (!endpoint) return;
        const joiner = endpoint.includes('?') ? '&' : '?';
        btn.setAttribute('data-endpoint', endpoint + joiner + 'token=' + encodeURIComponent(token));
      }});
    }})();

    // Copy JSON to clipboard
    document.querySelectorAll('.copy-json-btn').forEach(btn => {{
      btn.addEventListener('click', async () => {{
        const endpoint = btn.getAttribute('data-endpoint');
        try {{
          const res = await fetch(endpoint);
          const json = await res.json();
          await navigator.clipboard.writeText(JSON.stringify(json, null, 2));
          const orig = btn.textContent;
          btn.textContent = '✅ Copied!';
          setTimeout(() => btn.textContent = orig, 1500);
        }} catch (e) {{
          btn.textContent = '❌ Error';
          setTimeout(() => btn.textContent = btn.textContent.replace('❌ Error', '📋'), 1500);
        }}
      }});
    }});
  </script>
</body>
</html>"""
    return HTMLResponse(content=html)


# =============================================================================
# OPS CONSOLE LOGIN / LOGOUT
# =============================================================================


def _render_login_page(error: str = "") -> str:
    """Render the OPS console login page HTML."""
    error_html = f'<div class="error">{error}</div>' if error else ""
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>OPS Console Login</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
        }}
        .login-container {{
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 16px;
            padding: 40px;
            width: 100%;
            max-width: 400px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        }}
        h1 {{
            color: #fff;
            text-align: center;
            margin-bottom: 8px;
            font-size: 24px;
        }}
        .subtitle {{
            color: rgba(255, 255, 255, 0.6);
            text-align: center;
            margin-bottom: 32px;
            font-size: 14px;
        }}
        .error {{
            background: rgba(239, 68, 68, 0.2);
            border: 1px solid rgba(239, 68, 68, 0.5);
            color: #fca5a5;
            padding: 12px;
            border-radius: 8px;
            margin-bottom: 20px;
            font-size: 14px;
            text-align: center;
        }}
        .form-group {{
            margin-bottom: 20px;
        }}
        label {{
            display: block;
            color: rgba(255, 255, 255, 0.8);
            margin-bottom: 8px;
            font-size: 14px;
        }}
        input[type="password"] {{
            width: 100%;
            padding: 12px 16px;
            border: 1px solid rgba(255, 255, 255, 0.2);
            border-radius: 8px;
            background: rgba(255, 255, 255, 0.1);
            color: #fff;
            font-size: 16px;
            transition: border-color 0.2s;
        }}
        input[type="password"]:focus {{
            outline: none;
            border-color: #3b82f6;
        }}
        button {{
            width: 100%;
            padding: 14px;
            background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
            border: none;
            border-radius: 8px;
            color: #fff;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            transition: transform 0.2s, box-shadow 0.2s;
        }}
        button:hover {{
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(59, 130, 246, 0.4);
        }}
        .footer {{
            text-align: center;
            margin-top: 24px;
            color: rgba(255, 255, 255, 0.4);
            font-size: 12px;
        }}
    </style>
</head>
<body>
    <div class="login-container">
        <h1>FutbolStats OPS</h1>
        <p class="subtitle">Admin Console</p>
        {error_html}
        <form method="POST" action="/ops/login">
            <div class="form-group">
                <label for="password">Password</label>
                <input type="password" id="password" name="password" required autofocus>
            </div>
            <button type="submit">Sign In</button>
        </form>
        <p class="footer">Secure access only</p>
    </div>
</body>
</html>"""


@app.get("/ops/login")
async def ops_login_page(request: Request, error: str = ""):
    """Display the OPS console login form."""
    from fastapi.responses import HTMLResponse, RedirectResponse

    # If already logged in, redirect to dashboard
    if _has_valid_session(request):
        return RedirectResponse(url="/dashboard/ops", status_code=302)

    # Check if login is enabled
    if not settings.OPS_ADMIN_PASSWORD:
        raise HTTPException(
            status_code=503,
            detail="OPS login disabled. Set OPS_ADMIN_PASSWORD env var."
        )

    return HTMLResponse(content=_render_login_page(error))


@app.post("/ops/login")
@limiter.limit("10/minute")
async def ops_login_submit(request: Request):
    """Process OPS console login."""
    from fastapi.responses import HTMLResponse, RedirectResponse

    # Check if login is enabled
    if not settings.OPS_ADMIN_PASSWORD:
        raise HTTPException(status_code=503, detail="OPS login disabled")

    # Parse form data
    form = await request.form()
    password = form.get("password", "")

    # Validate password
    if password != settings.OPS_ADMIN_PASSWORD:
        logger.warning(f"[OPS_LOGIN] Failed login attempt from {request.client.host}")
        return HTMLResponse(
            content=_render_login_page("Invalid password"),
            status_code=401
        )

    # Create session
    request.session["ops_authenticated"] = True
    request.session["issued_at"] = datetime.utcnow().isoformat()
    logger.info(f"[OPS_LOGIN] Successful login from {request.client.host}")

    # Redirect to dashboard
    return RedirectResponse(url="/dashboard/ops", status_code=302)


@app.get("/ops/logout")
async def ops_logout(request: Request):
    """Logout from OPS console."""
    from fastapi.responses import RedirectResponse

    # Clear session
    request.session.clear()
    logger.info(f"[OPS_LOGIN] Logout from {request.client.host}")

    return RedirectResponse(url="/ops/login", status_code=302)


@app.get("/dashboard")
async def dashboard_home(request: Request):
    """Unified dashboard entrypoint (redirects to Ops)."""
    from fastapi.responses import RedirectResponse

    if not _verify_dashboard_token(request):
        # Redirect to login instead of 401 for better UX
        return RedirectResponse(url="/ops/login", status_code=302)

    # Preserve token query param ONLY in development (not in prod - security risk)
    target = "/dashboard/ops"
    if not os.getenv("RAILWAY_PROJECT_ID"):
        token = request.query_params.get("token")
        if token:
            target = f"{target}?token={token}"
    return RedirectResponse(url=target, status_code=307)


# =============================================================================
# DEBUG ENDPOINT: Daily Counts (temporary for ops monitoring)
# =============================================================================


@app.get("/dashboard/ops/daily_counts.json")
async def ops_daily_counts(
    request: Request,
    date: str = None,
    session: AsyncSession = Depends(get_async_session),
):
    """
    Get daily counts for predictions, audits, and LLM narratives.

    Args:
        date: Date in YYYY-MM-DD format. Defaults to today (UTC).
    """
    if not _verify_dashboard_token(request):
        raise HTTPException(status_code=401, detail="Dashboard access requires valid token.")

    import re

    target_date = date or datetime.utcnow().strftime("%Y-%m-%d")
    # Validate date format (YYYY-MM-DD) to prevent SQL injection
    if not re.match(r"^\d{4}-\d{2}-\d{2}$", target_date):
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD.")

    # A) Predictions
    predictions_created_today = await session.execute(
        text(f"SELECT COUNT(*) FROM predictions WHERE created_at::date = '{target_date}'")
    )
    pred_created = predictions_created_today.scalar() or 0

    predictions_for_matches_today = await session.execute(
        text(f"""
            SELECT COUNT(*) FROM predictions p
            JOIN matches m ON p.match_id = m.id
            WHERE m.date::date = '{target_date}'
        """)
    )
    pred_for_matches = predictions_for_matches_today.scalar() or 0

    # B) Audits
    ft_matches_today = await session.execute(
        text(f"""
            SELECT COUNT(*) FROM matches
            WHERE status IN ('FT', 'AET', 'PEN')
            AND date::date = '{target_date}'
        """)
    )
    ft_count = ft_matches_today.scalar() or 0

    with_prediction_outcome = await session.execute(
        text(f"""
            SELECT COUNT(*) FROM prediction_outcomes po
            JOIN matches m ON po.match_id = m.id
            WHERE m.status IN ('FT', 'AET', 'PEN')
            AND m.date::date = '{target_date}'
        """)
    )
    po_count = with_prediction_outcome.scalar() or 0

    with_post_match_audit = await session.execute(
        text(f"""
            SELECT COUNT(*) FROM post_match_audits pma
            JOIN prediction_outcomes po ON pma.outcome_id = po.id
            JOIN matches m ON po.match_id = m.id
            WHERE m.status IN ('FT', 'AET', 'PEN')
            AND m.date::date = '{target_date}'
        """)
    )
    pma_count = with_post_match_audit.scalar() or 0

    # C) LLM Narratives
    llm_ok_today = await session.execute(
        text(f"""
            SELECT COUNT(*) FROM post_match_audits pma
            JOIN prediction_outcomes po ON pma.outcome_id = po.id
            JOIN matches m ON po.match_id = m.id
            WHERE m.date::date = '{target_date}'
            AND pma.llm_narrative_status = 'ok'
        """)
    )
    llm_ok = llm_ok_today.scalar() or 0

    llm_breakdown = await session.execute(
        text(f"""
            SELECT pma.llm_narrative_status, COUNT(*) as count
            FROM post_match_audits pma
            JOIN prediction_outcomes po ON pma.outcome_id = po.id
            JOIN matches m ON po.match_id = m.id
            WHERE m.date::date = '{target_date}'
            GROUP BY pma.llm_narrative_status
            ORDER BY count DESC
        """)
    )
    breakdown = {row[0] or "null": row[1] for row in llm_breakdown.all()}

    # D) LLM Error Details (for debugging)
    llm_error_details = await session.execute(
        text(f"""
            SELECT
                po.match_id,
                pma.id as audit_id,
                pma.llm_narrative_status,
                pma.llm_narrative_delay_ms,
                pma.llm_narrative_exec_ms,
                pma.llm_narrative_tokens_in,
                pma.llm_narrative_tokens_out,
                pma.llm_narrative_worker_id,
                pma.llm_narrative_model,
                pma.llm_narrative_generated_at,
                CASE WHEN pma.llm_narrative_json IS NULL THEN true ELSE false END as json_is_null,
                SUBSTRING(pma.llm_narrative_json::text, 1, 500) as json_preview,
                pma.llm_narrative_error_code,
                pma.llm_narrative_error_detail,
                pma.llm_narrative_request_id,
                pma.llm_narrative_attempts
            FROM post_match_audits pma
            JOIN prediction_outcomes po ON pma.outcome_id = po.id
            JOIN matches m ON po.match_id = m.id
            WHERE m.date::date = '{target_date}'
            AND (pma.llm_narrative_status IS NULL OR pma.llm_narrative_status != 'ok')
            ORDER BY po.match_id
        """)
    )
    error_rows = []
    for row in llm_error_details.all():
        error_rows.append({
            "match_id": row[0],
            "audit_id": row[1],
            "status": row[2],
            "delay_ms": row[3],
            "exec_ms": row[4],
            "tokens_in": row[5],
            "tokens_out": row[6],
            "worker_id": row[7],
            "model": row[8],
            "generated_at": str(row[9]) if row[9] else None,
            "json_is_null": row[10],
            "json_preview": row[11],
            "error_code": row[12],
            "error_detail": row[13],
            "request_id": row[14],
            "attempts": row[15],
        })

    return {
        "date": target_date,
        "predictions": {
            "created_today": pred_created,
            "for_matches_today": pred_for_matches,
        },
        "audits": {
            "ft_matches_today": ft_count,
            "with_prediction_outcome": po_count,
            "with_post_match_audit": pma_count,
        },
        "llm_narratives": {
            "ok_today": llm_ok,
            "breakdown": breakdown,
            "error_details": error_rows,
        },
    }


@app.post("/dashboard/ops/migrate_llm_error_fields")
async def migrate_llm_error_fields(
    request: Request,
    session: AsyncSession = Depends(get_async_session),
):
    """
    One-time migration to add LLM error observability fields.
    Safe to run multiple times (uses IF NOT EXISTS).
    """
    if not _verify_dashboard_token(request):
        raise HTTPException(status_code=401, detail="Dashboard access requires valid token.")

    migrations = [
        "ALTER TABLE post_match_audits ADD COLUMN IF NOT EXISTS llm_narrative_error_code VARCHAR(50)",
        "ALTER TABLE post_match_audits ADD COLUMN IF NOT EXISTS llm_narrative_error_detail VARCHAR(500)",
        "ALTER TABLE post_match_audits ADD COLUMN IF NOT EXISTS llm_narrative_request_id VARCHAR(100)",
        "ALTER TABLE post_match_audits ADD COLUMN IF NOT EXISTS llm_narrative_attempts INTEGER",
    ]

    results = []
    for sql in migrations:
        try:
            await session.execute(text(sql))
            results.append({"sql": sql[:60] + "...", "status": "ok"})
        except Exception as e:
            results.append({"sql": sql[:60] + "...", "status": "error", "error": str(e)})

    await session.commit()

    # Verify columns exist
    verify = await session.execute(
        text("""
            SELECT column_name FROM information_schema.columns
            WHERE table_name='post_match_audits'
            AND column_name LIKE 'llm_narrative_error%'
            OR column_name IN ('llm_narrative_request_id', 'llm_narrative_attempts')
            ORDER BY column_name
        """)
    )
    columns = [row[0] for row in verify.all()]

    return {
        "status": "ok",
        "migrations": results,
        "verified_columns": columns,
    }


@app.post("/dashboard/ops/migrate_fastpath_fields")
async def migrate_fastpath_fields(
    request: Request,
    session: AsyncSession = Depends(get_async_session),
):
    """
    One-time migration to add fast-path tracking fields to matches.
    Safe to run multiple times (uses IF NOT EXISTS).
    """
    if not _verify_dashboard_token(request):
        raise HTTPException(status_code=401, detail="Dashboard access requires valid token.")

    migrations = [
        "ALTER TABLE matches ADD COLUMN IF NOT EXISTS finished_at TIMESTAMP",
        "ALTER TABLE matches ADD COLUMN IF NOT EXISTS stats_ready_at TIMESTAMP",
        "ALTER TABLE matches ADD COLUMN IF NOT EXISTS stats_last_checked_at TIMESTAMP",
        "ALTER TABLE matches ADD COLUMN IF NOT EXISTS events JSON",
        """CREATE INDEX IF NOT EXISTS idx_matches_fastpath_candidates
           ON matches(finished_at, stats_ready_at)
           WHERE finished_at IS NOT NULL AND stats_ready_at IS NULL""",
        """CREATE INDEX IF NOT EXISTS idx_matches_finished_at
           ON matches(finished_at)
           WHERE finished_at IS NOT NULL""",
    ]

    results = []
    for sql in migrations:
        try:
            await session.execute(text(sql))
            results.append({"sql": sql[:60] + "...", "status": "ok"})
        except Exception as e:
            results.append({"sql": sql[:60] + "...", "status": "error", "error": str(e)})

    await session.commit()

    # Verify columns exist
    verify = await session.execute(
        text("""
            SELECT column_name FROM information_schema.columns
            WHERE table_name='matches'
            AND column_name IN ('finished_at', 'stats_ready_at', 'stats_last_checked_at')
            ORDER BY column_name
        """)
    )
    columns = [row[0] for row in verify.all()]

    return {
        "status": "ok",
        "migrations": results,
        "verified_columns": columns,
    }


# ---------------------------------------------------------------------------
# Debug Log Endpoint (for iOS performance instrumentation)
# ---------------------------------------------------------------------------

from pathlib import Path
from datetime import datetime as dt

# Environment flag: DEBUG_LOG_ENABLED=true allows logging without token (dev mode ONLY)
# SECURITY: In production (Railway), this flag is IGNORED - auth is always required
_DEBUG_LOG_ENABLED = os.getenv("DEBUG_LOG_ENABLED", "false").lower() == "true"
_IS_PRODUCTION = os.getenv("RAILWAY_PROJECT_ID") is not None


@app.post("/debug/log")
async def debug_log(request: Request):
    """
    Receives performance logs from iOS instrumentation.

    Security:
    - In production: always require valid X-Dashboard-Token (fail-closed)
    - In development: DEBUG_LOG_ENABLED=true allows without token

    Rate limit: handled by global rate limiter
    """
    # SECURITY: In production, ALWAYS require auth (ignore DEBUG_LOG_ENABLED)
    skip_auth = _DEBUG_LOG_ENABLED and not _IS_PRODUCTION
    if not skip_auth:
        token = request.headers.get("X-Dashboard-Token")
        expected = os.getenv("DASHBOARD_TOKEN", "")
        if not expected:
            # Fail-closed: no token configured = deny all
            return JSONResponse({"error": "service misconfigured"}, status_code=503)
        if not token or token != expected:
            return JSONResponse({"error": "unauthorized"}, status_code=401)

    # Parse and validate payload
    try:
        payload = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid json"}, status_code=400)

    if not isinstance(payload, dict):
        return JSONResponse({"error": "payload must be object"}, status_code=400)

    if "component" not in payload:
        return JSONResponse({"error": "missing component field"}, status_code=400)

    # Ensure logs directory exists
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / "perf_debug.log"

    # Build log entry (no token/headers logged)
    entry = {
        "ts": dt.utcnow().isoformat() + "Z",
        "component": payload.get("component"),
        "endpoint": payload.get("endpoint"),
        "message": payload.get("message"),
        "data": payload.get("data"),
        "hypothesisId": payload.get("hypothesisId"),
    }

    # Append to log file
    import json as _json
    with open(log_file, "a") as f:
        f.write(_json.dumps(entry) + "\n")

    return {"status": "ok"}


@app.get("/debug/log")
async def get_debug_logs(request: Request, tail: int = 50):
    """
    Returns the last N lines of perf_debug.log for analysis.

    Security: requires valid X-Dashboard-Token (always, even in debug mode)
    """
    token = request.headers.get("X-Dashboard-Token")
    expected = os.getenv("DASHBOARD_TOKEN", "")
    if not token or token != expected:
        return JSONResponse({"error": "unauthorized"}, status_code=401)

    log_file = Path("logs") / "perf_debug.log"
    if not log_file.exists():
        return {"logs": [], "count": 0, "message": "No logs yet"}

    import json as _json
    lines = []
    with open(log_file, "r") as f:
        all_lines = f.readlines()
        # Get last N lines
        for line in all_lines[-tail:]:
            try:
                lines.append(_json.loads(line.strip()))
            except Exception:
                lines.append({"raw": line.strip()})

    return {"logs": lines, "count": len(lines), "total": len(all_lines)}
