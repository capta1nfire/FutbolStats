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
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel, Field, model_validator
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from sqlalchemy import func, select, text, column
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.database import close_db, get_async_session, init_db, AsyncSessionLocal, get_pool_status
from app.etl import APIFootballProvider, ETLPipeline
from app.etl.competitions import ALL_LEAGUE_IDS, COMPETITIONS
from app.etl.sota_constants import SOFASCORE_SUPPORTED_LEAGUES
from app.features import FeatureEngineer
from app.ml import XGBoostEngine
from app.ml.persistence import load_active_model, persist_model_snapshot
from app.models import JobRun, Match, OddsHistory, OpsAlert, PITReport, PostMatchAudit, Prediction, PredictionOutcome, SensorPrediction, ShadowPrediction, Team, TeamAdjustment, TeamOverride
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
    "standings_source_calculated": 0,
    "standings_source_placeholder": 0,
    "standings_source_miss": 0,
    # Timeline source
    "timeline_source_db": 0,
    "timeline_source_api_fallback": 0,
}


def _incr(key: str) -> None:
    """Increment a telemetry counter."""
    _telemetry[key] = _telemetry.get(key, 0) + 1


# Simple in-memory cache for predictions
# TTL reduced to 60s for better live match updates (elapsed, goals)
_predictions_cache = {
    "data": None,
    "timestamp": 0,
    "ttl": 60,  # 1 minute cache (was 5 min, reduced for live matches)
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


# Standings calculated cache: shorter TTL (15 min) for calculated standings
_STANDINGS_CALCULATED_TTL = 900  # 15 minutes


async def _calculate_standings_from_results(session, league_id: int, season: int) -> list:
    """
    Calculate standings from FT match results when API-Football has no data yet.

    Guardrails (per Auditor approval):
    - Only activates if FT_count >= 2 in this league/season
    - Only for league competitions (not cups/knockouts)
    - Returns source='calculated', is_calculated=True for transparency
    - Uses shorter cache TTL (15 min)
    - Priority: API standings > calculated > placeholder

    Sorting: points DESC, goal_diff DESC, goals_for DESC, team_name ASC

    Args:
        session: Database session
        league_id: League ID
        season: Season year

    Returns:
        List of standings dicts with calculated stats, or empty list if not eligible.
    """
    # Check FT count threshold (guardrail: need at least 2 finished matches)
    ft_count_result = await session.execute(
        text("""
            SELECT COUNT(*)
            FROM matches
            WHERE league_id = :league_id
              AND EXTRACT(YEAR FROM date) = :season
              AND status IN ('FT', 'AET', 'PEN')
              AND home_goals IS NOT NULL
              AND away_goals IS NOT NULL
        """),
        {"league_id": league_id, "season": season}
    )
    ft_count = ft_count_result.scalar() or 0

    if ft_count < 2:
        logger.debug(f"Calculated standings skipped: league {league_id} season {season} has only {ft_count} FT matches (need >= 2)")
        return []

    # Get all teams from season fixtures (includes teams with 0 matches played)
    teams_result = await session.execute(
        text("""
            SELECT DISTINCT t.id, t.external_id, t.name, t.logo_url
            FROM teams t
            JOIN matches m ON (t.id = m.home_team_id OR t.id = m.away_team_id)
            WHERE m.league_id = :league_id
              AND EXTRACT(YEAR FROM m.date) = :season
              AND t.team_type = 'club'
        """),
        {"league_id": league_id, "season": season}
    )
    teams = {row[0]: {"id": row[0], "external_id": row[1], "name": row[2], "logo_url": row[3]} for row in teams_result.fetchall()}

    if not teams:
        return []

    # Initialize stats for all teams
    stats = {}
    for team_id, team_data in teams.items():
        stats[team_id] = {
            "team_id": team_data["external_id"],
            "team_name": team_data["name"],
            "team_logo": team_data["logo_url"],
            "points": 0,
            "played": 0,
            "won": 0,
            "drawn": 0,
            "lost": 0,
            "goals_for": 0,
            "goals_against": 0,
            "goal_diff": 0,
            "form": "",  # Last 5 results
            "form_results": [],  # For building form string
        }

    # Calculate stats from FT matches
    matches_result = await session.execute(
        text("""
            SELECT home_team_id, away_team_id, home_goals, away_goals, date
            FROM matches
            WHERE league_id = :league_id
              AND EXTRACT(YEAR FROM date) = :season
              AND status IN ('FT', 'AET', 'PEN')
              AND home_goals IS NOT NULL
              AND away_goals IS NOT NULL
            ORDER BY date ASC
        """),
        {"league_id": league_id, "season": season}
    )

    for home_id, away_id, home_goals, away_goals, match_date in matches_result.fetchall():
        if home_id not in stats or away_id not in stats:
            continue

        # Home team stats
        stats[home_id]["played"] += 1
        stats[home_id]["goals_for"] += home_goals
        stats[home_id]["goals_against"] += away_goals

        # Away team stats
        stats[away_id]["played"] += 1
        stats[away_id]["goals_for"] += away_goals
        stats[away_id]["goals_against"] += home_goals

        if home_goals > away_goals:
            # Home win
            stats[home_id]["won"] += 1
            stats[home_id]["points"] += 3
            stats[home_id]["form_results"].append("W")
            stats[away_id]["lost"] += 1
            stats[away_id]["form_results"].append("L")
        elif home_goals < away_goals:
            # Away win
            stats[away_id]["won"] += 1
            stats[away_id]["points"] += 3
            stats[away_id]["form_results"].append("W")
            stats[home_id]["lost"] += 1
            stats[home_id]["form_results"].append("L")
        else:
            # Draw
            stats[home_id]["drawn"] += 1
            stats[home_id]["points"] += 1
            stats[home_id]["form_results"].append("D")
            stats[away_id]["drawn"] += 1
            stats[away_id]["points"] += 1
            stats[away_id]["form_results"].append("D")

    # Calculate goal diff and form string (last 5)
    for team_id in stats:
        stats[team_id]["goal_diff"] = stats[team_id]["goals_for"] - stats[team_id]["goals_against"]
        stats[team_id]["form"] = "".join(stats[team_id]["form_results"][-5:])
        del stats[team_id]["form_results"]  # Remove helper field

    # Sort: points DESC, goal_diff DESC, goals_for DESC, team_name ASC
    sorted_teams = sorted(
        stats.values(),
        key=lambda x: (-x["points"], -x["goal_diff"], -x["goals_for"], x["team_name"])
    )

    # Build final standings with positions
    standings = []
    for idx, team_stats in enumerate(sorted_teams, start=1):
        standings.append({
            "position": idx,
            "team_id": team_stats["team_id"],
            "team_name": team_stats["team_name"],
            "team_logo": team_stats["team_logo"],
            "points": team_stats["points"],
            "played": team_stats["played"],
            "won": team_stats["won"],
            "drawn": team_stats["drawn"],
            "lost": team_stats["lost"],
            "goals_for": team_stats["goals_for"],
            "goals_against": team_stats["goals_against"],
            "goal_diff": team_stats["goal_diff"],
            "form": team_stats["form"],
            "group": None,
            "is_calculated": True,  # Transparency flag
            "source": "calculated",
        })

    logger.info(f"Calculated standings for league {league_id} season {season}: {len(standings)} teams from {ft_count} FT matches")
    return standings


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
    elapsed: Optional[int] = None  # Current minute for live matches (e.g., 32)
    elapsed_extra: Optional[int] = None  # Added/injury time (e.g., 3 for 90+3)
    home_goals: Optional[int] = None  # Final score (nil if not played)
    away_goals: Optional[int] = None  # Final score (nil if not played)
    league_id: Optional[int] = None
    venue: Optional[dict] = None  # Stadium: {"name": str, "city": str} or None
    events: Optional[list[dict]] = None  # Match events (goals, cards) for live timeline

    # Model pick derived from probabilities (home, draw, away)
    pick: Optional[str] = None

    # Adjusted probabilities (after team adjustments)
    probabilities: dict

    @model_validator(mode='after')
    def derive_pick_from_probabilities(self) -> 'PredictionItem':
        """Derive pick from probabilities if not set.

        Deterministic tie-breaker: home > draw > away (matches betting convention).
        """
        if self.pick is None and self.probabilities:
            probs = self.probabilities
            h = probs.get("home", 0)
            d = probs.get("draw", 0)
            a = probs.get("away", 0)
            if h or d or a:  # At least one prob exists
                # Deterministic: priority home > draw > away on ties
                if h >= d and h >= a:
                    self.pick = "home"
                elif d >= a:
                    self.pick = "draw"
                else:
                    self.pick = "away"
        return self
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
    _: bool = Depends(verify_api_key),
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
            elapsed=pred.get("elapsed"),
            elapsed_extra=pred.get("elapsed_extra"),
            home_goals=pred.get("home_goals"),
            away_goals=pred.get("away_goals"),
            league_id=pred.get("league_id"),
            venue=pred.get("venue"),
            events=pred.get("events"),
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
    Overlay rerun predictions from DB for NS matches.

    DISABLED (2025-01): Per audit directive, serving is baseline-only.
    Rerun/shadow predictions are for evaluation only, not production serving.
    This function now always returns predictions unchanged (baseline).

    The PREFER_RERUN_PREDICTIONS flag and rerun infrastructure remain for
    OPS/analysis endpoints but do not affect public prediction serving.

    Returns:
        tuple: (unchanged predictions, empty stats dict)
    """
    # AUDIT P0: Baseline-only serving - always return unchanged
    stats = {"db_hits": 0, "db_stale": 0, "live_fallback": 0, "total_ns": 0}
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
    _: bool = Depends(verify_api_key),
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


# =============================================================================
# LIVE SUMMARY ENDPOINT (iOS Live Score Polling)
# =============================================================================

# Cache for live summary (L1 cache with 5s TTL per Auditor requirement)
_live_summary_cache: dict = {
    "data": None,
    "timestamp": 0.0,
    "ttl": 5.0,  # 5 second TTL
}

# Live statuses that indicate a match is currently being played
LIVE_STATUSES = frozenset(["1H", "HT", "2H", "ET", "BT", "P", "LIVE", "INT", "SUSP"])


@app.get("/live-summary")
@limiter.limit("60/minute")  # Rate limit: 60 req/min per IP (4 req/15s is comfortable)
async def get_live_summary(
    request: Request,
    session: AsyncSession = Depends(get_async_session),
    _: bool = Depends(verify_api_key),  # Require API key authentication
):
    """
    Ultra-light endpoint for live score polling (iOS LiveScoreManager).

    Returns only LIVE matches with minimal payload (~50 bytes/match).
    Designed for 15s polling interval from iOS clients.

    Response schema (v2 - FASE 1: includes events):
    {
        "ts": 1705500000,  // Unix timestamp of cache
        "matches": {
            "12345": {
                "s": "2H", "e": 67, "ex": 0, "h": 2, "a": 1,
                "ev": [
                    {"m": 23, "t": "Goal", "d": "Normal Goal", "tm": 529, "p": "Messi", "a": "Di Maria"},
                    {"m": 45, "x": 2, "t": "Card", "d": "Yellow Card", "tm": 530, "p": "Martinez"}
                ]
            }
        }
    }

    Fields:
    - s: status (1H, HT, 2H, ET, FT, etc.)
    - e: elapsed minutes
    - ex: elapsed_extra (injury time, e.g., 3 for 90+3)
    - h: home goals
    - a: away goals
    - ev: events array (optional, only if events exist)
      - m: minute
      - x: extra minute (injury time)
      - t: type (Goal, Card)
      - d: detail (Normal Goal, Yellow Card, Red Card, Penalty, Own Goal, etc.)
      - tm: team_id
      - p: player name
      - a: assist name (goals only)

    Auth: Requires X-API-Key header.
    Rate limit: 60 requests/minute per IP.
    """
    from app.telemetry.metrics import record_live_summary_request

    start_time = time.time()
    now = time.time()

    try:
        # Check L1 cache (5s TTL)
        if (
            _live_summary_cache["data"] is not None
            and now - _live_summary_cache["timestamp"] < _live_summary_cache["ttl"]
        ):
            # Cache hit - return immediately
            cached_data = _live_summary_cache["data"]
            latency_ms = (time.time() - start_time) * 1000
            record_live_summary_request(
                status="ok",
                latency_ms=latency_ms,
                matches_count=len(cached_data.get("matches", {})),
            )
            return cached_data

        # Cache miss - query DB (FASE 1: now includes events column)
        query = text("""
            SELECT id, status, elapsed, elapsed_extra, home_goals, away_goals, events
            FROM matches
            WHERE status IN ('1H', 'HT', '2H', 'ET', 'BT', 'P', 'LIVE', 'INT', 'SUSP')
            LIMIT 50
        """)

        result = await session.execute(query)
        rows = result.fetchall()

        # Build compact response (keyed by internal match_id per Auditor requirement)
        # FASE 1: now includes events (ev) when available
        matches_dict = {}
        for row in rows:
            match_id = row[0]
            match_data = {
                "s": row[1],  # status
                "e": row[2] or 0,  # elapsed
                "ex": row[3] or 0,  # elapsed_extra
                "h": row[4] or 0,  # home_goals
                "a": row[5] or 0,  # away_goals
            }
            # FASE 1: Convert FULL schema events to COMPACT format for iOS
            # DB stores: {type, detail, minute, extra_minute, team_id, team_name, player_name, assist_name}
            # iOS expects: {m, x, t, d, tm, p, a}
            events = row[6]
            if events:
                # events is already JSON from DB, parse if string
                if isinstance(events, str):
                    try:
                        events = json.loads(events)
                    except json.JSONDecodeError:
                        events = None
                if events:
                    # Convert to compact format (only Goal and Card for iOS timeline)
                    compact_events = []
                    for ev in events:
                        ev_type = ev.get("type")
                        if ev_type not in ("Goal", "Card"):
                            continue
                        compact_events.append({
                            "m": ev.get("minute"),
                            "x": ev.get("extra_minute"),
                            "t": ev_type,
                            "d": ev.get("detail"),
                            "tm": ev.get("team_id"),
                            "p": ev.get("player_name"),
                            "a": ev.get("assist_name"),
                        })
                    if compact_events:
                        match_data["ev"] = compact_events
            matches_dict[match_id] = match_data

        response_data = {
            "ts": int(now),
            "matches": matches_dict,
        }

        # Update L1 cache
        _live_summary_cache["data"] = response_data
        _live_summary_cache["timestamp"] = now

        latency_ms = (time.time() - start_time) * 1000
        record_live_summary_request(
            status="ok",
            latency_ms=latency_ms,
            matches_count=len(matches_dict),
        )

        logger.debug(f"[live-summary] Returned {len(matches_dict)} live matches in {latency_ms:.1f}ms")

        return response_data

    except Exception as e:
        latency_ms = (time.time() - start_time) * 1000
        record_live_summary_request(status="error", latency_ms=latency_ms, matches_count=0)
        logger.error(f"[live-summary] Error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/teams")
async def list_teams(
    team_type: Optional[str] = None,
    limit: int = 100,
    session: AsyncSession = Depends(get_async_session),
    _: bool = Depends(verify_api_key),
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
    _: bool = Depends(verify_api_key),
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
async def list_competitions(
    _: bool = Depends(verify_api_key),
):
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
    _: bool = Depends(verify_api_key),
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
    _: bool = Depends(verify_api_key),
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
                # L3: Try calculated standings from FT results first
                standings = await _calculate_standings_from_results(session, match.league_id, season)
                if standings:
                    standings_status = "calculated"
                    standings_source = "calculated"
                    _incr("standings_source_calculated")
                    _set_cached_standings(match.league_id, season, standings)
                else:
                    # L4: Generate placeholder standings (zero stats, alphabetical order)
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
            "id": home_team.external_id if home_team else None,
            "name": home_name,
            "logo": home_logo,
            "history": home_history["matches"],
            "position": home_position,
            "league_points": home_league_points,
        },
        "away_team": {
            "id": away_team.external_id if away_team else None,
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
    _: bool = Depends(verify_api_key),
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
    _: bool = Depends(verify_api_key),
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
    # Use the FIRST frozen prediction (original baseline model)
    # The two_stage shadow model was added later for A/B testing but shouldn't
    # replace the original prediction for evaluation purposes
    prediction_source = "frozen_original"
    result = await session.execute(
        select(Prediction)
        .where(Prediction.match_id == match_id)
        .where(Prediction.is_frozen == True)
        .order_by(Prediction.created_at.asc())  # First/original prediction
        .limit(1)
    )
    prediction = result.scalar_one_or_none()

    # Last fallback: any prediction (mark as low confidence)
    if not prediction:
        prediction_source = "unfrozen_fallback"
        result = await session.execute(
            select(Prediction)
            .where(Prediction.match_id == match_id)
            .order_by(Prediction.created_at.asc())
            .limit(1)
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
            "prediction_source": prediction_source,
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
    _: bool = Depends(verify_api_key),
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
    _: bool = Depends(verify_api_key),
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

        # L3.5: Calculated standings from FT results (when API has no data yet)
        # Priority: API > calculated > placeholder
        # Guardrails: FT_count >= 2, transparency via is_calculated flag
        if not standings:
            standings = await _calculate_standings_from_results(session, league_id, season)
            if standings:
                source = "calculated"
                # Use shorter TTL for calculated standings (15 min)
                _set_cached_standings(league_id, season, standings)
                logger.info(f"Using calculated standings for league {league_id} season {season}")

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

        # Determine if standings are placeholder or calculated
        is_placeholder = source == "placeholder" or (
            standings and standings[0].get("is_placeholder", False)
        )
        is_calculated = source == "calculated" or (
            standings and standings[0].get("is_calculated", False)
        )

        return {
            "league_id": league_id,
            "season": season,
            "standings": standings,
            "source": source,
            "is_placeholder": is_placeholder,
            "is_calculated": is_calculated,
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
    _: bool = Depends(verify_api_key),
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


# =============================================================================
# TITAN OMNISCIENCE Dashboard
# =============================================================================


@app.get("/dashboard/titan.json")
async def titan_dashboard_json(request: Request):
    """
    TITAN OMNISCIENCE Dashboard JSON - Enterprise scraping status.

    Returns comprehensive TITAN operational status including:
    - Extraction counts and coverage
    - DLQ (Dead Letter Queue) status
    - PIT (Point-in-Time) compliance metrics
    - Feature matrix statistics

    Protected by X-Dashboard-Token (same auth as /dashboard/ops.json).
    """
    if not _verify_dashboard_token(request):
        raise HTTPException(
            status_code=401,
            detail="Dashboard access requires valid token.",
        )

    from app.titan.dashboard import get_titan_status

    async with AsyncSessionLocal() as session:
        return await get_titan_status(session)


# =============================================================================
# Feature Coverage Matrix (SOTA Dashboard)
# =============================================================================
# Cache for feature coverage matrix (expensive query, TTL 30 min)
_feature_coverage_cache = {
    "data": None,
    "timestamp": 0,
    "ttl": 1800,  # 30 minutes
}


@app.get("/dashboard/feature-coverage.json")
async def dashboard_feature_coverage_json(request: Request):
    """
    Feature Coverage Matrix - Shows % of non-NULL features by league and season.

    Used by SOTA dashboard to identify which leagues have sufficient data quality
    for ML training (avoid imputing missing values with 0).

    Windows:
    - 23/24: 2023-08-01 to 2024-07-31
    - 24/25: 2024-08-01 to 2025-07-31
    - 25/26: 2025-08-01 to 2026-07-31 (current season)

    Features (30 total):
    - Tier 1 (14): Core features from public.matches [PROD]
    - Tier 1b (6): xG features from titan.feature_matrix [TITAN]
    - Tier 1c (4): Lineup features from titan.feature_matrix [TITAN]
    - Tier 1d (6): XI Depth features from titan.feature_matrix [TITAN]

    Auth: X-Dashboard-Token header.
    Cache: 30 minutes TTL.
    """
    if not _verify_dashboard_token(request):
        raise HTTPException(
            status_code=401,
            detail="Dashboard access requires valid token.",
        )

    now = time.time()
    cached = False
    cache_age = None

    # Check cache
    if _feature_coverage_cache["data"] and (now - _feature_coverage_cache["timestamp"]) < _feature_coverage_cache["ttl"]:
        cached = True
        cache_age = round(now - _feature_coverage_cache["timestamp"], 1)
        return {
            "generated_at": _feature_coverage_cache["data"]["generated_at"],
            "cached": cached,
            "cache_age_seconds": cache_age,
            "data": _feature_coverage_cache["data"]["data"],
        }

    # Calculate fresh data
    async with AsyncSessionLocal() as session:
        data = await _calculate_feature_coverage(session)

    # Update cache
    _feature_coverage_cache["data"] = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "data": data,
    }
    _feature_coverage_cache["timestamp"] = now

    return {
        "generated_at": _feature_coverage_cache["data"]["generated_at"],
        "cached": False,
        "cache_age_seconds": None,
        "data": data,
    }


async def _calculate_feature_coverage(session) -> dict:
    """
    Calculate feature coverage matrix for all leagues and windows.

    Returns structured data matching the frontend contract.
    """
    from app.etl.competitions import COMPETITIONS

    # Define windows
    windows = [
        {"key": "23/24", "from": "2023-08-01", "to": "2024-07-31"},
        {"key": "24/25", "from": "2024-08-01", "to": "2025-07-31"},
        {"key": "25/26", "from": "2025-08-01", "to": "2026-07-31"},
    ]

    # Define tiers
    tiers = [
        {"id": "tier1", "label": "[PROD] Tier 1 - Core", "badge": "PROD"},
        {"id": "tier1b", "label": "[TITAN] Tier 1b - xG", "badge": "TITAN"},
        {"id": "tier1c", "label": "[TITAN] Tier 1c - Lineup", "badge": "TITAN"},
        {"id": "tier1d", "label": "[TITAN] Tier 1d - XI Depth", "badge": "TITAN"},
    ]

    # Define features with tier mapping
    features = [
        # Tier 1 - Core (calculated from matches data)
        {"key": "home_goals_scored_avg", "tier_id": "tier1", "badge": "PROD", "source": "public.matches"},
        {"key": "home_goals_conceded_avg", "tier_id": "tier1", "badge": "PROD", "source": "public.matches"},
        {"key": "home_shots_avg", "tier_id": "tier1", "badge": "PROD", "source": "public.matches.stats"},
        {"key": "home_corners_avg", "tier_id": "tier1", "badge": "PROD", "source": "public.matches.stats"},
        {"key": "home_rest_days", "tier_id": "tier1", "badge": "PROD", "source": "public.matches"},
        {"key": "home_matches_played", "tier_id": "tier1", "badge": "PROD", "source": "public.matches"},
        {"key": "away_goals_scored_avg", "tier_id": "tier1", "badge": "PROD", "source": "public.matches"},
        {"key": "away_goals_conceded_avg", "tier_id": "tier1", "badge": "PROD", "source": "public.matches"},
        {"key": "away_shots_avg", "tier_id": "tier1", "badge": "PROD", "source": "public.matches.stats"},
        {"key": "away_corners_avg", "tier_id": "tier1", "badge": "PROD", "source": "public.matches.stats"},
        {"key": "away_rest_days", "tier_id": "tier1", "badge": "PROD", "source": "public.matches"},
        {"key": "away_matches_played", "tier_id": "tier1", "badge": "PROD", "source": "public.matches"},
        {"key": "goal_diff_avg", "tier_id": "tier1", "badge": "PROD", "source": "derived"},
        {"key": "rest_diff", "tier_id": "tier1", "badge": "PROD", "source": "derived"},
        # Tier 1b - xG
        {"key": "xg_home_last5", "tier_id": "tier1b", "badge": "TITAN", "source": "titan.feature_matrix"},
        {"key": "xga_home_last5", "tier_id": "tier1b", "badge": "TITAN", "source": "titan.feature_matrix"},
        {"key": "npxg_home_last5", "tier_id": "tier1b", "badge": "TITAN", "source": "titan.feature_matrix"},
        {"key": "xg_away_last5", "tier_id": "tier1b", "badge": "TITAN", "source": "titan.feature_matrix"},
        {"key": "xga_away_last5", "tier_id": "tier1b", "badge": "TITAN", "source": "titan.feature_matrix"},
        {"key": "npxg_away_last5", "tier_id": "tier1b", "badge": "TITAN", "source": "titan.feature_matrix"},
        # Tier 1c - Lineup
        {"key": "sofascore_home_formation", "tier_id": "tier1c", "badge": "TITAN", "source": "titan.feature_matrix"},
        {"key": "sofascore_away_formation", "tier_id": "tier1c", "badge": "TITAN", "source": "titan.feature_matrix"},
        {"key": "lineup_home_starters_count", "tier_id": "tier1c", "badge": "TITAN", "source": "titan.feature_matrix"},
        {"key": "lineup_away_starters_count", "tier_id": "tier1c", "badge": "TITAN", "source": "titan.feature_matrix"},
        # Tier 1d - XI Depth
        {"key": "xi_home_def_count", "tier_id": "tier1d", "badge": "TITAN", "source": "titan.feature_matrix"},
        {"key": "xi_home_mid_count", "tier_id": "tier1d", "badge": "TITAN", "source": "titan.feature_matrix"},
        {"key": "xi_home_fwd_count", "tier_id": "tier1d", "badge": "TITAN", "source": "titan.feature_matrix"},
        {"key": "xi_away_def_count", "tier_id": "tier1d", "badge": "TITAN", "source": "titan.feature_matrix"},
        {"key": "xi_away_mid_count", "tier_id": "tier1d", "badge": "TITAN", "source": "titan.feature_matrix"},
        {"key": "xi_away_fwd_count", "tier_id": "tier1d", "badge": "TITAN", "source": "titan.feature_matrix"},
    ]

    # Build leagues list from COMPETITIONS
    leagues = [
        {"league_id": comp.league_id, "name": comp.name}
        for comp in COMPETITIONS.values()
    ]

    # =========================================================================
    # Query 1: Get match counts and Tier 1 coverage per league/window
    # =========================================================================
    tier1_query = text("""
        WITH match_counts AS (
            SELECT
                league_id,
                CASE
                    WHEN date >= '2023-08-01' AND date < '2024-08-01' THEN '23/24'
                    WHEN date >= '2024-08-01' AND date < '2025-08-01' THEN '24/25'
                    WHEN date >= '2025-08-01' AND date < '2026-08-01' THEN '25/26'
                END as time_window,
                COUNT(*) as total,
                -- Goals are always available for FT matches
                COUNT(*) as with_goals,
                -- Stats JSON with shots/corners (check if total_shots exists)
                COUNT(*) FILTER (WHERE
                    stats IS NOT NULL
                    AND stats::text NOT IN ('null', '{}', '')
                    AND (stats->'home'->>'total_shots') IS NOT NULL
                ) as with_stats
            FROM matches
            WHERE status = 'FT'
              AND date >= '2023-08-01'
              AND date < '2026-08-01'
            GROUP BY league_id, time_window
        )
        SELECT * FROM match_counts WHERE time_window IS NOT NULL
        ORDER BY league_id, time_window
    """)

    tier1_result = await session.execute(tier1_query)
    tier1_rows = tier1_result.fetchall()

    # Build tier1 data structure
    tier1_data = {}  # {league_id: {window: {total, with_goals, with_stats}}}
    for row in tier1_rows:
        lid = row.league_id
        win = row.time_window
        if lid not in tier1_data:
            tier1_data[lid] = {}
        tier1_data[lid][win] = {
            "total": int(row.total),
            "with_goals": int(row.with_goals),
            "with_stats": int(row.with_stats),
        }

    # =========================================================================
    # Query 2: Get TITAN feature coverage per league/window
    # =========================================================================
    titan_query = text("""
        SELECT
            competition_id as league_id,
            CASE
                WHEN kickoff_utc >= '2023-08-01' AND kickoff_utc < '2024-08-01' THEN '23/24'
                WHEN kickoff_utc >= '2024-08-01' AND kickoff_utc < '2025-08-01' THEN '24/25'
                WHEN kickoff_utc >= '2025-08-01' AND kickoff_utc < '2026-08-01' THEN '25/26'
            END as time_window,
            COUNT(*) as total,
            -- Tier 1b: xG features
            COUNT(*) FILTER (WHERE xg_home_last5 IS NOT NULL) as xg_home_last5,
            COUNT(*) FILTER (WHERE xga_home_last5 IS NOT NULL) as xga_home_last5,
            COUNT(*) FILTER (WHERE npxg_home_last5 IS NOT NULL) as npxg_home_last5,
            COUNT(*) FILTER (WHERE xg_away_last5 IS NOT NULL) as xg_away_last5,
            COUNT(*) FILTER (WHERE xga_away_last5 IS NOT NULL) as xga_away_last5,
            COUNT(*) FILTER (WHERE npxg_away_last5 IS NOT NULL) as npxg_away_last5,
            -- Tier 1c: Lineup features
            COUNT(*) FILTER (WHERE sofascore_home_formation IS NOT NULL) as sofascore_home_formation,
            COUNT(*) FILTER (WHERE sofascore_away_formation IS NOT NULL) as sofascore_away_formation,
            COUNT(*) FILTER (WHERE lineup_home_starters_count IS NOT NULL) as lineup_home_starters_count,
            COUNT(*) FILTER (WHERE lineup_away_starters_count IS NOT NULL) as lineup_away_starters_count,
            -- Tier 1d: XI Depth features
            COUNT(*) FILTER (WHERE xi_home_def_count IS NOT NULL) as xi_home_def_count,
            COUNT(*) FILTER (WHERE xi_home_mid_count IS NOT NULL) as xi_home_mid_count,
            COUNT(*) FILTER (WHERE xi_home_fwd_count IS NOT NULL) as xi_home_fwd_count,
            COUNT(*) FILTER (WHERE xi_away_def_count IS NOT NULL) as xi_away_def_count,
            COUNT(*) FILTER (WHERE xi_away_mid_count IS NOT NULL) as xi_away_mid_count,
            COUNT(*) FILTER (WHERE xi_away_fwd_count IS NOT NULL) as xi_away_fwd_count
        FROM titan.feature_matrix
        WHERE kickoff_utc >= '2023-08-01' AND kickoff_utc < '2026-08-01'
        GROUP BY competition_id, time_window
        HAVING CASE
            WHEN kickoff_utc >= '2023-08-01' AND kickoff_utc < '2024-08-01' THEN '23/24'
            WHEN kickoff_utc >= '2024-08-01' AND kickoff_utc < '2025-08-01' THEN '24/25'
            WHEN kickoff_utc >= '2025-08-01' AND kickoff_utc < '2026-08-01' THEN '25/26'
        END IS NOT NULL
        ORDER BY competition_id, time_window
    """)

    try:
        titan_result = await session.execute(titan_query)
        titan_rows = titan_result.fetchall()
    except Exception as e:
        logger.warning(f"TITAN feature_matrix query failed (schema may not exist): {e}")
        titan_rows = []

    # Build titan data structure
    titan_data = {}  # {league_id: {window: {feature: count}}}
    titan_features = [
        "xg_home_last5", "xga_home_last5", "npxg_home_last5",
        "xg_away_last5", "xga_away_last5", "npxg_away_last5",
        "sofascore_home_formation", "sofascore_away_formation",
        "lineup_home_starters_count", "lineup_away_starters_count",
        "xi_home_def_count", "xi_home_mid_count", "xi_home_fwd_count",
        "xi_away_def_count", "xi_away_mid_count", "xi_away_fwd_count",
    ]

    for row in titan_rows:
        lid = row.league_id
        win = row.time_window
        if lid not in titan_data:
            titan_data[lid] = {}
        titan_data[lid][win] = {"total": int(row.total)}
        for feat in titan_features:
            titan_data[lid][win][feat] = int(getattr(row, feat, 0) or 0)

    # =========================================================================
    # Build coverage and league_summaries structures
    # =========================================================================
    coverage = {}  # {feature_key: {league_id: {window: {pct, n}}}}
    league_summaries = {}  # {league_id: {window: {matches_total, avg_pct}}}

    # Initialize coverage structure
    for feat in features:
        coverage[feat["key"]] = {}

    # Process each league
    for league in leagues:
        lid = league["league_id"]
        league_summaries[str(lid)] = {}

        for win_def in windows:
            win = win_def["key"]

            # Get base match count from tier1_data
            t1 = tier1_data.get(lid, {}).get(win, {"total": 0, "with_goals": 0, "with_stats": 0})
            matches_total = t1["total"]

            # Calculate coverage for each feature
            feature_pcts = []

            for feat in features:
                fkey = feat["key"]

                if str(lid) not in coverage[fkey]:
                    coverage[fkey][str(lid)] = {}

                if matches_total == 0:
                    pct = 0.0
                    n = 0
                elif feat["tier_id"] == "tier1":
                    # Tier 1 features - use tier1_data
                    if fkey in ["home_shots_avg", "away_shots_avg", "home_corners_avg", "away_corners_avg"]:
                        # Stats-dependent features
                        n = t1["with_stats"]
                    else:
                        # Goals/rest/matches_played - always available for FT matches
                        n = t1["with_goals"]
                    pct = round(100.0 * n / matches_total, 1)
                else:
                    # Tier 1b/1c/1d - use titan_data
                    titan_win = titan_data.get(lid, {}).get(win, {})
                    # Use titan total as denominator if available, else matches_total
                    titan_total = titan_win.get("total", 0)
                    if titan_total > 0:
                        n = titan_win.get(fkey, 0)
                        pct = round(100.0 * n / titan_total, 1)
                    else:
                        n = 0
                        pct = 0.0

                coverage[fkey][str(lid)][win] = {"pct": pct, "n": n}
                feature_pcts.append(pct)

            # Get titan total for this window (for transparency in league_summaries)
            titan_win = titan_data.get(lid, {}).get(win, {})
            titan_total = titan_win.get("total", 0)

            # Calculate league summary (avg across all 30 features)
            avg_pct = round(sum(feature_pcts) / len(feature_pcts), 1) if feature_pcts else 0.0
            league_summaries[str(lid)][win] = {
                "matches_total_ft": matches_total,
                "matches_total_titan": titan_total,
                "avg_pct": avg_pct,
            }

        # Calculate total (combined windows) with correct denominators per tier
        # Tier 1: uses FT matches as denominator
        total_matches_ft = sum(
            league_summaries[str(lid)].get(w["key"], {}).get("matches_total_ft", 0)
            for w in windows
        )
        # TITAN tiers: uses titan.feature_matrix rows as denominator
        total_matches_titan = sum(
            titan_data.get(lid, {}).get(w["key"], {}).get("total", 0)
            for w in windows
        )

        total_pcts = []
        for feat in features:
            fkey = feat["key"]
            total_n = sum(
                coverage[fkey].get(str(lid), {}).get(w["key"], {}).get("n", 0)
                for w in windows
            )

            # Use correct denominator based on tier (ABE fix: avoid mixing denominators)
            if feat["tier_id"] == "tier1":
                denominator = total_matches_ft
            else:
                denominator = total_matches_titan

            total_pct = round(100.0 * total_n / denominator, 1) if denominator > 0 else 0.0
            coverage[fkey][str(lid)]["total"] = {"pct": total_pct, "n": total_n}
            total_pcts.append(total_pct)

        league_summaries[str(lid)]["total"] = {
            "matches_total_ft": total_matches_ft,
            "matches_total_titan": total_matches_titan,
            "avg_pct": round(sum(total_pcts) / len(total_pcts), 1) if total_pcts else 0.0,
        }

    return {
        "windows": windows,
        "tiers": tiers,
        "features": features,
        "leagues": leagues,
        "league_summaries": league_summaries,
        "coverage": coverage,
    }


# =============================================================================
# ML Health Dashboard (ATI v1.1)
# =============================================================================
# Cache for ML health (60s TTL - early warning for pipeline issues)
_ml_health_cache = {
    "data": None,
    "timestamp": 0,
    "ttl": 60,  # 60 seconds
    "previous_snapshot": None,  # For future top_regressions calculation
    "previous_snapshot_at": None,
}


@app.get("/dashboard/ml_health.json")
async def dashboard_ml_health_json(request: Request):
    """
    ML Health Dashboard - Comprehensive ML pipeline health metrics.

    ATI v1.1 - Addresses "vuelo a ciegas" finding where XGBoost ran
    with 0% coverage for shots/corners in 23/24 season.

    Returns:
    - fuel_gauge: Overall health status (ok/warn/error) with reasons
    - sota_stats_coverage: Stats coverage by season and league (P0 - root cause)
    - titan_coverage: TITAN feature_matrix coverage by tier
    - pit_compliance: Point-in-Time violations
    - freshness: Data staleness (age_hours_now for early warning)
    - prediction_confidence: Entropy and tier distribution
    - top_regressions: Not implemented yet (requires baseline)

    Fail-soft: If any section fails, returns partial data with health="partial"

    Auth: X-Dashboard-Token header.
    """
    import time

    if not _verify_dashboard_token(request):
        raise HTTPException(
            status_code=401,
            detail="Dashboard access requires valid token.",
        )

    now = time.time()

    # Check cache
    if _ml_health_cache["data"] and (now - _ml_health_cache["timestamp"]) < _ml_health_cache["ttl"]:
        cached_data = _ml_health_cache["data"]
        return {
            "generated_at": cached_data["generated_at"],
            "cached": True,
            "cache_age_seconds": round(now - _ml_health_cache["timestamp"], 1),
            "health": cached_data["health"],
            "data": cached_data["data"],
        }

    # Build fresh data
    from app.ml.health import build_ml_health_data

    async with AsyncSessionLocal() as session:
        result = await build_ml_health_data(session)

    # Update cache + snapshot for future regressions
    _ml_health_cache["previous_snapshot"] = _ml_health_cache["data"]
    _ml_health_cache["previous_snapshot_at"] = _ml_health_cache["timestamp"]
    _ml_health_cache["data"] = result
    _ml_health_cache["timestamp"] = now

    return {
        "generated_at": result["generated_at"],
        "cached": False,
        "cache_age_seconds": None,
        "health": result["health"],
        "data": result["data"],
    }


# =============================================================================
# Admin Panel P0 (Read-only)
# =============================================================================
# Cache for admin endpoints (120s for lists, 60s for details)
_admin_cache = {
    "overview": {"data": None, "timestamp": 0, "ttl": 120},
    "leagues": {"data": None, "timestamp": 0, "ttl": 120},
    "league_detail": {},  # keyed by league_id
    "teams": {},  # keyed by filter params
    "team_detail": {},  # keyed by team_id
}


@app.get("/dashboard/admin/overview.json")
async def dashboard_admin_overview(request: Request):
    """Admin Panel - System overview with counts and coverage summary."""
    import time

    if not _verify_dashboard_token(request):
        raise HTTPException(status_code=401, detail="Dashboard access requires valid token.")

    now = time.time()
    cache = _admin_cache["overview"]

    if cache["data"] and (now - cache["timestamp"]) < cache["ttl"]:
        return {
            "generated_at": cache["data"]["generated_at"],
            "cached": True,
            "cache_age_seconds": round(now - cache["timestamp"], 1),
            "data": cache["data"]["data"],
        }

    from app.dashboard.admin import build_overview

    async with AsyncSessionLocal() as session:
        data = await build_overview(session)

    result = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "data": data,
    }
    cache["data"] = result
    cache["timestamp"] = now

    return {
        "generated_at": result["generated_at"],
        "cached": False,
        "cache_age_seconds": None,
        "data": data,
    }


@app.get("/dashboard/admin/leagues.json")
async def dashboard_admin_leagues(request: Request):
    """Admin Panel - List all leagues with configured vs observed distinction."""
    import time

    if not _verify_dashboard_token(request):
        raise HTTPException(status_code=401, detail="Dashboard access requires valid token.")

    now = time.time()
    cache = _admin_cache["leagues"]

    if cache["data"] and (now - cache["timestamp"]) < cache["ttl"]:
        return {
            "generated_at": cache["data"]["generated_at"],
            "cached": True,
            "cache_age_seconds": round(now - cache["timestamp"], 1),
            "data": cache["data"]["data"],
        }

    from app.dashboard.admin import build_leagues_list

    async with AsyncSessionLocal() as session:
        data = await build_leagues_list(session)

    result = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "data": data,
    }
    cache["data"] = result
    cache["timestamp"] = now

    return {
        "generated_at": result["generated_at"],
        "cached": False,
        "cache_age_seconds": None,
        "data": data,
    }


@app.get("/dashboard/admin/league/{league_id}.json")
async def dashboard_admin_league_detail(request: Request, league_id: int):
    """Admin Panel - Detail for a specific league."""
    import time

    if not _verify_dashboard_token(request):
        raise HTTPException(status_code=401, detail="Dashboard access requires valid token.")

    now = time.time()
    cache_key = str(league_id)
    cache = _admin_cache["league_detail"]

    if cache_key in cache and cache[cache_key]["data"] and (now - cache[cache_key]["timestamp"]) < 60:
        cached = cache[cache_key]
        return {
            "generated_at": cached["data"]["generated_at"],
            "cached": True,
            "cache_age_seconds": round(now - cached["timestamp"], 1),
            "data": cached["data"]["data"],
        }

    from app.dashboard.admin import build_league_detail

    async with AsyncSessionLocal() as session:
        data = await build_league_detail(session, league_id)

    if data is None:
        raise HTTPException(status_code=404, detail=f"League {league_id} not found.")

    result = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "data": data,
    }
    cache[cache_key] = {"data": result, "timestamp": now}

    return {
        "generated_at": result["generated_at"],
        "cached": False,
        "cache_age_seconds": None,
        "data": data,
    }


@app.get("/dashboard/admin/teams.json")
async def dashboard_admin_teams(
    request: Request,
    type: str = "all",
    country: str = None,
    limit: int = 100,
    offset: int = 0,
):
    """Admin Panel - List teams with optional filters and pagination."""
    import time

    if not _verify_dashboard_token(request):
        raise HTTPException(status_code=401, detail="Dashboard access requires valid token.")

    now = time.time()
    cache_key = f"{type}:{country}:{limit}:{offset}"
    cache = _admin_cache["teams"]

    if cache_key in cache and cache[cache_key]["data"] and (now - cache[cache_key]["timestamp"]) < 120:
        cached = cache[cache_key]
        return {
            "generated_at": cached["data"]["generated_at"],
            "cached": True,
            "cache_age_seconds": round(now - cached["timestamp"], 1),
            "data": cached["data"]["data"],
        }

    from app.dashboard.admin import build_teams_list

    team_type = type if type != "all" else None

    async with AsyncSessionLocal() as session:
        data = await build_teams_list(session, team_type=team_type, country=country, limit=limit, offset=offset)

    result = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "data": data,
    }
    cache[cache_key] = {"data": result, "timestamp": now}

    return {
        "generated_at": result["generated_at"],
        "cached": False,
        "cache_age_seconds": None,
        "data": data,
    }


@app.get("/dashboard/admin/team/{team_id}.json")
async def dashboard_admin_team_detail(request: Request, team_id: int):
    """Admin Panel - Detail for a specific team."""
    import time

    if not _verify_dashboard_token(request):
        raise HTTPException(status_code=401, detail="Dashboard access requires valid token.")

    now = time.time()
    cache_key = str(team_id)
    cache = _admin_cache["team_detail"]

    if cache_key in cache and cache[cache_key]["data"] and (now - cache[cache_key]["timestamp"]) < 60:
        cached = cache[cache_key]
        return {
            "generated_at": cached["data"]["generated_at"],
            "cached": True,
            "cache_age_seconds": round(now - cached["timestamp"], 1),
            "data": cached["data"]["data"],
        }

    from app.dashboard.admin import build_team_detail

    async with AsyncSessionLocal() as session:
        data = await build_team_detail(session, team_id)

    if data is None:
        raise HTTPException(status_code=404, detail=f"Team {team_id} not found.")

    result = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "data": data,
    }
    cache[cache_key] = {"data": result, "timestamp": now}

    return {
        "generated_at": result["generated_at"],
        "cached": False,
        "cache_age_seconds": None,
        "data": data,
    }


# =============================================================================
# P2B - Admin Mutations
# =============================================================================

@app.patch("/dashboard/admin/leagues/{league_id}.json")
async def dashboard_admin_patch_league(request: Request, league_id: int):
    """
    Admin Panel - Update a league configuration.

    P2B: PATCH mutations with audit trail.
    Whitelist: is_active, country, kind, priority, match_type, match_weight,
               display_order, tags, rules_json, group_id, name
    """
    if not _verify_dashboard_token(request):
        raise HTTPException(status_code=401, detail="Dashboard access requires valid token.")

    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    from app.dashboard.admin import patch_league, ValidationError

    try:
        async with AsyncSessionLocal() as session:
            result = await patch_league(session, league_id, body, actor="dashboard")
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    # Invalidate caches
    _admin_cache["overview"]["data"] = None
    _admin_cache["leagues"]["data"] = None
    if str(league_id) in _admin_cache["league_detail"]:
        del _admin_cache["league_detail"][str(league_id)]

    return {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "data": result,
    }


@app.get("/dashboard/admin/audit.json")
async def dashboard_admin_audit(
    request: Request,
    entity_type: str = None,
    entity_id: str = None,
    limit: int = 50,
    offset: int = 0,
):
    """
    Admin Panel - View audit log entries.

    P2B: Audit trail for mutations.
    Optional filters: entity_type, entity_id
    Supported entity_types: admin_leagues, admin_league_groups
    """
    if not _verify_dashboard_token(request):
        raise HTTPException(status_code=401, detail="Dashboard access requires valid token.")

    from app.dashboard.admin import get_audit_log, ValidationError

    try:
        async with AsyncSessionLocal() as session:
            data = await get_audit_log(
                session,
                entity_type=entity_type,
                entity_id=entity_id,
                limit=limit,
                offset=offset
            )
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "cached": False,
        "data": data,
    }


@app.get("/dashboard/admin/league-groups.json")
async def dashboard_admin_league_groups(request: Request):
    """
    Admin Panel - List league groups with aggregated metrics.

    P2C: Paired leagues (Apertura/Clausura) as navigable entities.
    """
    if not _verify_dashboard_token(request):
        raise HTTPException(status_code=401, detail="Dashboard access requires valid token.")

    from app.dashboard.admin import build_league_groups_list

    async with AsyncSessionLocal() as session:
        data = await build_league_groups_list(session)

    return {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "cached": False,
        "cache_age_seconds": None,
        "data": data,
    }


@app.get("/dashboard/admin/league-group/{group_id}.json")
async def dashboard_admin_league_group_detail(request: Request, group_id: int):
    """
    Admin Panel - League group detail with member leagues.

    P2C: Full details for a paired league group.
    """
    if not _verify_dashboard_token(request):
        raise HTTPException(status_code=401, detail="Dashboard access requires valid token.")

    from app.dashboard.admin import build_league_group_detail

    async with AsyncSessionLocal() as session:
        data = await build_league_group_detail(session, group_id)

    if data is None:
        raise HTTPException(status_code=404, detail=f"Group {group_id} not found")

    return {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "cached": False,
        "cache_age_seconds": None,
        "data": data,
    }


# =============================================================================
# Football Navigation API (P3)
# =============================================================================

# Cache for football nav endpoints
_football_nav_cache: dict = {}
FOOTBALL_NAV_CACHE_TTL = 120  # 2 minutes for lists
FOOTBALL_NAV_DETAIL_CACHE_TTL = 60  # 1 minute for details


def _get_football_cache(key: str, ttl: int) -> tuple:
    """Get cached data if valid. Returns (data, age_seconds) or (None, None)."""
    if key in _football_nav_cache:
        cached = _football_nav_cache[key]
        age = time.time() - cached["timestamp"]
        if age < ttl:
            return cached["data"], int(age)
    return None, None


def _set_football_cache(key: str, data: dict):
    """Set cache entry."""
    _football_nav_cache[key] = {"data": data, "timestamp": time.time()}


@app.get("/dashboard/football/nav.json")
async def dashboard_football_nav(request: Request):
    """
    Football Navigation - Top-level categories.

    P3: Returns categories for Col 2 top area.
    """
    if not _verify_dashboard_token(request):
        raise HTTPException(status_code=401, detail="Dashboard access requires valid token.")

    cache_key = "football_nav"
    cached_data, cache_age = _get_football_cache(cache_key, FOOTBALL_NAV_CACHE_TTL)

    if cached_data:
        return {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "cached": True,
            "cache_age_seconds": cache_age,
            "data": cached_data,
        }

    from app.dashboard.football_nav import build_nav

    async with AsyncSessionLocal() as session:
        data = await build_nav(session)

    _set_football_cache(cache_key, data)

    return {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "cached": False,
        "cache_age_seconds": None,
        "data": data,
    }


@app.get("/dashboard/football/leagues/countries.json")
async def dashboard_football_countries(request: Request):
    """
    Football Navigation - Countries with active leagues.

    P3: For Col 2 when category=leagues_by_country.
    """
    if not _verify_dashboard_token(request):
        raise HTTPException(status_code=401, detail="Dashboard access requires valid token.")

    cache_key = "football_countries"
    cached_data, cache_age = _get_football_cache(cache_key, FOOTBALL_NAV_CACHE_TTL)

    if cached_data:
        return {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "cached": True,
            "cache_age_seconds": cache_age,
            "data": cached_data,
        }

    from app.dashboard.football_nav import build_countries_list

    async with AsyncSessionLocal() as session:
        data = await build_countries_list(session)

    _set_football_cache(cache_key, data)

    return {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "cached": False,
        "cache_age_seconds": None,
        "data": data,
    }


@app.get("/dashboard/football/leagues/country/{country}.json")
async def dashboard_football_country_detail(request: Request, country: str):
    """
    Football Navigation - Leagues for a specific country.

    P3: For Col 4 when a country is selected.
    """
    if not _verify_dashboard_token(request):
        raise HTTPException(status_code=401, detail="Dashboard access requires valid token.")

    # URL decode country name
    from urllib.parse import unquote
    country = unquote(country)

    cache_key = f"football_country_{country}"
    cached_data, cache_age = _get_football_cache(cache_key, FOOTBALL_NAV_DETAIL_CACHE_TTL)

    if cached_data:
        return {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "cached": True,
            "cache_age_seconds": cache_age,
            "data": cached_data,
        }

    from app.dashboard.football_nav import build_country_detail

    async with AsyncSessionLocal() as session:
        data = await build_country_detail(session, country)

    if data is None:
        raise HTTPException(status_code=404, detail=f"Country '{country}' not found or has no active leagues")

    _set_football_cache(cache_key, data)

    return {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "cached": False,
        "cache_age_seconds": None,
        "data": data,
    }


@app.get("/dashboard/football/league/{league_id}.json")
async def dashboard_football_league_detail(request: Request, league_id: int):
    """
    Football Navigation - League detail.

    P3: For Col 4 drilldown of a specific league.
    """
    if not _verify_dashboard_token(request):
        raise HTTPException(status_code=401, detail="Dashboard access requires valid token.")

    cache_key = f"football_league_{league_id}"
    cached_data, cache_age = _get_football_cache(cache_key, FOOTBALL_NAV_DETAIL_CACHE_TTL)

    if cached_data:
        return {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "cached": True,
            "cache_age_seconds": cache_age,
            "data": cached_data,
        }

    from app.dashboard.football_nav import build_league_nav_detail

    async with AsyncSessionLocal() as session:
        data = await build_league_nav_detail(session, league_id)

    if data is None:
        raise HTTPException(status_code=404, detail=f"League {league_id} not found or not active")

    _set_football_cache(cache_key, data)

    return {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "cached": False,
        "cache_age_seconds": None,
        "data": data,
    }


@app.get("/dashboard/football/group/{group_id}.json")
async def dashboard_football_group_detail(request: Request, group_id: int):
    """
    Football Navigation - League group detail (paired leagues).

    P3: For Col 4 drilldown of a paired league group.
    """
    if not _verify_dashboard_token(request):
        raise HTTPException(status_code=401, detail="Dashboard access requires valid token.")

    cache_key = f"football_group_{group_id}"
    cached_data, cache_age = _get_football_cache(cache_key, FOOTBALL_NAV_DETAIL_CACHE_TTL)

    if cached_data:
        return {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "cached": True,
            "cache_age_seconds": cache_age,
            "data": cached_data,
        }

    from app.dashboard.football_nav import build_group_nav_detail

    async with AsyncSessionLocal() as session:
        data = await build_group_nav_detail(session, group_id)

    if data is None:
        raise HTTPException(status_code=404, detail=f"Group {group_id} not found or has no active members")

    _set_football_cache(cache_key, data)

    return {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "cached": False,
        "cache_age_seconds": None,
        "data": data,
    }


@app.get("/dashboard/football/overview.json")
async def dashboard_football_overview(request: Request):
    """
    Football Navigation - Overview.

    P3.1: Shows summary counts, upcoming matches, top leagues, and alerts.
    All filtered by admin_leagues.is_active = true.
    """
    if not _verify_dashboard_token(request):
        raise HTTPException(status_code=401, detail="Dashboard access requires valid token.")

    cache_key = "football_overview"
    cached_data, cache_age = _get_football_cache(cache_key, FOOTBALL_NAV_CACHE_TTL)

    if cached_data:
        return {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "cached": True,
            "cache_age_seconds": cache_age,
            "data": cached_data,
        }

    from app.dashboard.football_nav import build_football_overview

    async with AsyncSessionLocal() as session:
        data = await build_football_overview(session)

    _set_football_cache(cache_key, data)

    return {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "cached": False,
        "cache_age_seconds": None,
        "data": data,
    }


# =============================================================================
# Football Navigation - National Teams (P3.3)
# =============================================================================


@app.get("/dashboard/football/nationals/countries.json")
async def dashboard_football_nationals_countries(request: Request):
    """
    Football Navigation - List countries with national teams.

    P3.3: Returns countries with national teams that have matches in active
    international competitions. Ordered by total_matches DESC.
    """
    if not _verify_dashboard_token(request):
        raise HTTPException(status_code=401, detail="Dashboard access requires valid token.")

    cache_key = "football_nationals_countries"
    cached_data, cache_age = _get_football_cache(cache_key, FOOTBALL_NAV_CACHE_TTL)

    if cached_data:
        return {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "cached": True,
            "cache_age_seconds": cache_age,
            "data": cached_data,
        }

    from app.dashboard.football_nav import build_nationals_countries_list

    async with AsyncSessionLocal() as session:
        data = await build_nationals_countries_list(session)

    _set_football_cache(cache_key, data)

    return {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "cached": False,
        "cache_age_seconds": None,
        "data": data,
    }


@app.get("/dashboard/football/nationals/country/{country}.json")
async def dashboard_football_nationals_country(request: Request, country: str):
    """
    Football Navigation - Country detail with national teams.

    P3.3: The {country} parameter is the team name (e.g., "Portugal", "Spain").
    Returns teams, competitions, recent matches, and stats.
    """
    if not _verify_dashboard_token(request):
        raise HTTPException(status_code=401, detail="Dashboard access requires valid token.")

    # URL decode the country parameter (handles spaces, accents, etc.)
    from urllib.parse import unquote
    country = unquote(country)

    cache_key = f"football_nationals_country_{country}"
    cached_data, cache_age = _get_football_cache(cache_key, FOOTBALL_NAV_DETAIL_CACHE_TTL)

    if cached_data:
        return {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "cached": True,
            "cache_age_seconds": cache_age,
            "data": cached_data,
        }

    from app.dashboard.football_nav import build_nationals_country_detail

    async with AsyncSessionLocal() as session:
        data = await build_nationals_country_detail(session, country)

    if data is None:
        raise HTTPException(status_code=404, detail=f"Country '{country}' not found")

    _set_football_cache(cache_key, data)

    return {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "cached": False,
        "cache_age_seconds": None,
        "data": data,
    }


@app.get("/dashboard/football/nationals/team/{team_id}.json")
async def dashboard_football_nationals_team(request: Request, team_id: int):
    """
    Football Navigation - Team 360 for national team.

    P3.3: Returns full team details including competitions, stats (overall and
    by competition), recent matches, and head-to-head records.
    Only works for teams with team_type='national'.
    """
    if not _verify_dashboard_token(request):
        raise HTTPException(status_code=401, detail="Dashboard access requires valid token.")

    cache_key = f"football_nationals_team_{team_id}"
    cached_data, cache_age = _get_football_cache(cache_key, FOOTBALL_NAV_DETAIL_CACHE_TTL)

    if cached_data:
        return {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "cached": True,
            "cache_age_seconds": cache_age,
            "data": cached_data,
        }

    from app.dashboard.football_nav import build_nationals_team_detail

    async with AsyncSessionLocal() as session:
        data = await build_nationals_team_detail(session, team_id)

    if data is None:
        raise HTTPException(status_code=404, detail=f"National team {team_id} not found")

    _set_football_cache(cache_key, data)

    return {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "cached": False,
        "cache_age_seconds": None,
        "data": data,
    }


# =============================================================================
# Football Navigation - Tournaments & Cups (P3.4)
# =============================================================================


@app.get("/dashboard/football/tournaments.json")
async def dashboard_football_tournaments(request: Request):
    """
    Football Navigation - List tournaments, cups and international competitions.

    P3.4: Returns tournaments filtered by kind IN ('cup', 'international', 'friendly')
    AND is_active=true. Includes stats per tournament: total_matches, matches_30d,
    seasons_range, last_match, next_match, coverage percentages, participants_count.
    """
    if not _verify_dashboard_token(request):
        raise HTTPException(status_code=401, detail="Dashboard access requires valid token.")

    cache_key = "football_tournaments"
    cached_data, cache_age = _get_football_cache(cache_key, FOOTBALL_NAV_CACHE_TTL)

    if cached_data:
        return {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "cached": True,
            "cache_age_seconds": cache_age,
            "data": cached_data,
        }

    from app.dashboard.football_nav import build_tournaments_list

    async with AsyncSessionLocal() as session:
        data = await build_tournaments_list(session)

    _set_football_cache(cache_key, data)

    return {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "cached": False,
        "cache_age_seconds": None,
        "data": data,
    }


# =============================================================================
# Football Navigation - World Cup 2026 (P3.5)
# =============================================================================


@app.get("/dashboard/football/world-cup-2026/overview.json")
async def dashboard_football_world_cup_overview(request: Request):
    """
    Football Navigation - World Cup 2026 overview.

    P3.5: Returns overview with summary, alerts, and upcoming matches.
    Status can be: "ok", "not_ready", or "disabled".
    Fail-soft: returns status="not_ready" if no data, never 500.
    """
    if not _verify_dashboard_token(request):
        raise HTTPException(status_code=401, detail="Dashboard access requires valid token.")

    cache_key = "football_world_cup_overview"
    cached_data, cache_age = _get_football_cache(cache_key, FOOTBALL_NAV_CACHE_TTL)

    if cached_data:
        return {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "cached": True,
            "cache_age_seconds": cache_age,
            "data": cached_data,
        }

    from app.dashboard.football_nav import build_world_cup_overview

    async with AsyncSessionLocal() as session:
        data = await build_world_cup_overview(session)

    _set_football_cache(cache_key, data)

    return {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "cached": False,
        "cache_age_seconds": None,
        "data": data,
    }


@app.get("/dashboard/football/world-cup-2026/groups.json")
async def dashboard_football_world_cup_groups(request: Request):
    """
    Football Navigation - World Cup 2026 groups list.

    P3.5: Returns all groups with team standings.
    Status can be: "ok", "not_ready", or "disabled".
    Fail-soft: returns empty groups if no standings data.
    """
    if not _verify_dashboard_token(request):
        raise HTTPException(status_code=401, detail="Dashboard access requires valid token.")

    cache_key = "football_world_cup_groups"
    cached_data, cache_age = _get_football_cache(cache_key, FOOTBALL_NAV_DETAIL_CACHE_TTL)

    if cached_data:
        return {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "cached": True,
            "cache_age_seconds": cache_age,
            "data": cached_data,
        }

    from app.dashboard.football_nav import build_world_cup_groups

    async with AsyncSessionLocal() as session:
        data = await build_world_cup_groups(session)

    _set_football_cache(cache_key, data)

    return {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "cached": False,
        "cache_age_seconds": None,
        "data": data,
    }


@app.get("/dashboard/football/world-cup-2026/group/{group}.json")
async def dashboard_football_world_cup_group_detail(request: Request, group: str):
    """
    Football Navigation - World Cup 2026 group detail.

    P3.5: Returns standings and matches for a specific group.
    Parameter group is URL-decoded (e.g., "Group A" or "Group%20A").
    Returns 404 if group not found.
    """
    if not _verify_dashboard_token(request):
        raise HTTPException(status_code=401, detail="Dashboard access requires valid token.")

    # URL decode the group name
    from urllib.parse import unquote
    group = unquote(group)

    cache_key = f"football_world_cup_group_{group}"
    cached_data, cache_age = _get_football_cache(cache_key, FOOTBALL_NAV_DETAIL_CACHE_TTL)

    if cached_data:
        return {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "cached": True,
            "cache_age_seconds": cache_age,
            "data": cached_data,
        }

    from app.dashboard.football_nav import build_world_cup_group_detail

    async with AsyncSessionLocal() as session:
        data = await build_world_cup_group_detail(session, group)

    if data is None:
        raise HTTPException(status_code=404, detail=f"Group '{group}' not found")

    _set_football_cache(cache_key, data)

    return {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "cached": False,
        "cache_age_seconds": None,
        "data": data,
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


# =============================================================================
# DASHBOARD V2 ENDPOINTS (wrapper-compliant)
# Wrapper: { generated_at, cached, cache_age_seconds, data }
# =============================================================================

def _make_v2_wrapper(data: dict, cached: bool = False, cache_age_seconds: float = 0) -> dict:
    """Build standard v2 wrapper for dashboard endpoints."""
    return {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "cached": cached,
        "cache_age_seconds": round(cache_age_seconds, 1),
        "data": data,
    }


# Cache for /dashboard/overview/rollup.json
_rollup_cache: dict = {"data": None, "timestamp": 0, "ttl": 60}


@app.get("/dashboard/overview/rollup.json")
async def dashboard_overview_rollup(request: Request):
    """
    V2 endpoint: Overview rollup data with standard wrapper.
    TTL: 60s
    Auth: X-Dashboard-Token
    """
    if not _verify_dashboard_token(request):
        raise HTTPException(status_code=401, detail="Dashboard access requires valid token.")

    import time as time_module
    now_ts = time_module.time()

    # Check cache
    if _rollup_cache["data"] and (now_ts - _rollup_cache["timestamp"]) < _rollup_cache["ttl"]:
        cache_age = now_ts - _rollup_cache["timestamp"]
        return _make_v2_wrapper(_rollup_cache["data"], cached=True, cache_age_seconds=cache_age)

    # Best-effort: get ops data and extract rollup subset
    try:
        ops_data = await _get_cached_ops_data()

        # Extract stable rollup fields - use REAL keys from ops_data
        # Required fields (core dashboard)
        rollup = {
            "generated_at": ops_data.get("generated_at"),
            "budget": ops_data.get("budget"),
            "sentry": ops_data.get("sentry"),
            "jobs_health": ops_data.get("jobs_health"),
            "predictions_health": ops_data.get("predictions_health"),
            "fastpath_health": ops_data.get("fastpath_health"),
            "pit": ops_data.get("pit"),
            "movement": ops_data.get("movement"),
            "llm_cost": ops_data.get("llm_cost"),
            "sota_enrichment": ops_data.get("sota_enrichment"),
            "providers": ops_data.get("providers"),
        }

        # Optional fields (include if present)
        optional_keys = [
            "coverage_by_league",
            "shadow_mode",
            "sensor_b",
            "rerun_serving",
            "ml_model",
            "telemetry",
        ]
        for key in optional_keys:
            if key in ops_data and ops_data[key] is not None:
                rollup[key] = ops_data[key]

        _rollup_cache["data"] = rollup
        _rollup_cache["timestamp"] = now_ts

        return _make_v2_wrapper(rollup, cached=False, cache_age_seconds=0)

    except Exception as e:
        logger.warning(f"Rollup endpoint error: {e}")
        return _make_v2_wrapper(
            {"status": "degraded", "note": f"upstream error: {str(e)[:50]}"},
            cached=False,
            cache_age_seconds=0,
        )


# Cache for /dashboard/sentry/issues.json
_sentry_issues_cache: dict = {"data": None, "timestamp": 0, "ttl": 90, "params": None}


@app.get("/dashboard/sentry/issues.json")
async def dashboard_sentry_issues(
    request: Request,
    range: str = "24h",
    page: int = 1,
    limit: int = 50,
):
    """
    V2 endpoint: Sentry issues with standard wrapper.
    TTL: 90s
    Auth: X-Dashboard-Token
    Query params: range (24h|1h|7d), page, limit (max 100)
    """
    if not _verify_dashboard_token(request):
        raise HTTPException(status_code=401, detail="Dashboard access requires valid token.")

    import time as time_module
    import httpx
    from datetime import timedelta

    # Clamp limit
    limit = min(max(1, limit), 100)
    page = max(1, page)

    # Validate range
    valid_ranges = {"1h": 1, "24h": 24, "7d": 168}
    if range not in valid_ranges:
        range = "24h"
    hours = valid_ranges[range]

    now_ts = time_module.time()
    cache_key = f"{range}:{page}:{limit}"

    # Check cache (only if same params)
    if (_sentry_issues_cache["data"] and
        _sentry_issues_cache["params"] == cache_key and
        (now_ts - _sentry_issues_cache["timestamp"]) < _sentry_issues_cache["ttl"]):
        cache_age = now_ts - _sentry_issues_cache["timestamp"]
        return _make_v2_wrapper(_sentry_issues_cache["data"], cached=True, cache_age_seconds=cache_age)

    # Base response (degraded fallback)
    base_data = {
        "issues": [],
        "total": 0,
        "page": page,
        "limit": limit,
        "pages": 0,
        "status": "degraded",
        "note": "upstream unavailable",
    }

    # Check credentials
    if not settings.SENTRY_AUTH_TOKEN or not settings.SENTRY_ORG or not settings.SENTRY_PROJECT_SLUG:
        base_data["note"] = "Sentry credentials not configured"
        return _make_v2_wrapper(base_data, cached=False, cache_age_seconds=0)

    org_slug = settings.SENTRY_ORG
    project_slug = settings.SENTRY_PROJECT_SLUG
    auth_token = settings.SENTRY_AUTH_TOKEN
    env_filter = settings.SENTRY_ENV or "production"

    headers = {
        "Authorization": f"Bearer {auth_token}",
        "Content-Type": "application/json",
    }

    try:
        async with httpx.AsyncClient(headers=headers, timeout=10.0) as client:
            issues_url = f"https://sentry.io/api/0/projects/{org_slug}/{project_slug}/issues/"

            # Time boundary
            now_dt = datetime.utcnow()
            time_ago = (now_dt - timedelta(hours=hours)).strftime("%Y-%m-%dT%H:%M:%S")

            env_query = f"environment:{env_filter}" if env_filter else ""

            # Fetch issues
            params = {
                "query": f"is:unresolved {env_query}".strip(),
                "sort": "date",
                "limit": 100,  # Fetch more, then paginate
            }
            resp = await client.get(issues_url, params=params)

            if resp.status_code != 200:
                base_data["note"] = f"Sentry API returned {resp.status_code}"
                return _make_v2_wrapper(base_data, cached=False, cache_age_seconds=0)

            all_issues = resp.json() if isinstance(resp.json(), list) else []

            # Filter by time range (lastSeen within range)
            filtered_issues = [
                i for i in all_issues
                if i.get("lastSeen", "") >= time_ago
            ]

            # Build issue list (no PII, no stacktraces)
            issues = []
            for issue in filtered_issues:
                title = issue.get("title", "Unknown")[:100]
                title = title.replace("@", "[at]")  # Basic sanitization
                issue_id = issue.get("id")

                issues.append({
                    "id": str(issue_id) if issue_id else None,
                    "title": title,
                    "level": issue.get("level", "error"),
                    "count": int(issue.get("count", 0)),
                    "last_seen_at": issue.get("lastSeen"),
                    "first_seen_at": issue.get("firstSeen"),
                    "issue_url": f"https://sentry.io/organizations/{org_slug}/issues/{issue_id}/" if issue_id else None,
                })

            # Sort by count descending
            issues.sort(key=lambda x: x["count"], reverse=True)

            # Paginate
            total = len(issues)
            pages = (total + limit - 1) // limit if total > 0 else 0
            start = (page - 1) * limit
            end = start + limit
            paginated_issues = issues[start:end]

            # Determine status
            status = "ok"
            if len([i for i in issues if i["level"] == "error"]) >= 5:
                status = "warn"
            if len([i for i in issues if i["level"] == "error"]) >= 20:
                status = "red"

            result = {
                "issues": paginated_issues,
                "total": total,
                "page": page,
                "limit": limit,
                "pages": pages,
                "status": status,
            }

            # Cache
            _sentry_issues_cache["data"] = result
            _sentry_issues_cache["timestamp"] = now_ts
            _sentry_issues_cache["params"] = cache_key

            return _make_v2_wrapper(result, cached=False, cache_age_seconds=0)

    except Exception as e:
        logger.warning(f"Sentry issues endpoint error: {e}")
        base_data["note"] = f"fetch error: {str(e)[:50]}"
        return _make_v2_wrapper(base_data, cached=False, cache_age_seconds=0)


# Cache for /dashboard/predictions/missing.json
_missing_preds_cache: dict = {"data": None, "timestamp": 0, "ttl": 60, "params": None}


@app.get("/dashboard/predictions/missing.json")
async def dashboard_predictions_missing(
    request: Request,
    hours: int = 48,
    league_ids: str = None,  # comma-separated
    page: int = 1,
    limit: int = 50,
    session: AsyncSession = Depends(get_async_session),
):
    """
    V2 endpoint: Matches missing predictions with standard wrapper.
    TTL: 60s
    Auth: X-Dashboard-Token
    Query params: hours (1-72), league_ids (comma-separated), page, limit (max 100)
    """
    if not _verify_dashboard_token(request):
        raise HTTPException(status_code=401, detail="Dashboard access requires valid token.")

    import time as time_module

    # Clamp params
    hours = min(max(1, hours), 72)
    limit = min(max(1, limit), 100)
    page = max(1, page)

    # Parse league_ids
    league_filter = []
    if league_ids:
        try:
            league_filter = [int(x.strip()) for x in league_ids.split(",") if x.strip().isdigit()]
        except Exception:
            pass

    now_ts = time_module.time()
    cache_key = f"{hours}:{','.join(map(str, league_filter))}:{page}:{limit}"

    # Check cache
    if (_missing_preds_cache["data"] and
        _missing_preds_cache["params"] == cache_key and
        (now_ts - _missing_preds_cache["timestamp"]) < _missing_preds_cache["ttl"]):
        cache_age = now_ts - _missing_preds_cache["timestamp"]
        return _make_v2_wrapper(_missing_preds_cache["data"], cached=True, cache_age_seconds=cache_age)

    base_data = {
        "missing": [],
        "missing_total": 0,
        "matches_total": 0,
        "coverage_pct": None,
        "total": 0,
        "page": page,
        "limit": limit,
        "pages": 0,
        "status": "degraded",
        "note": "upstream unavailable",
    }

    try:
        from sqlalchemy import text

        now_dt = datetime.utcnow()
        cutoff = now_dt + timedelta(hours=hours)

        # Build query for matches without predictions
        league_clause = ""
        if league_filter:
            league_clause = f"AND m.league_id IN ({','.join(map(str, league_filter))})"

        # Count total matches in window
        total_query = f"""
            SELECT COUNT(*) FROM matches m
            WHERE m.status = 'NS'
              AND m.date BETWEEN NOW() AND :cutoff
              {league_clause}
        """
        total_result = await session.execute(text(total_query), {"cutoff": cutoff})
        matches_total = total_result.scalar() or 0

        # Get matches missing predictions
        missing_query = f"""
            SELECT m.id, m.date, m.league_id, ht.name as home, at.name as away
            FROM matches m
            JOIN teams ht ON ht.id = m.home_team_id
            JOIN teams at ON at.id = m.away_team_id
            LEFT JOIN predictions p ON p.match_id = m.id
            WHERE m.status = 'NS'
              AND m.date BETWEEN NOW() AND :cutoff
              AND p.id IS NULL
              {league_clause}
            ORDER BY m.date ASC
        """
        missing_result = await session.execute(text(missing_query), {"cutoff": cutoff})
        all_missing = missing_result.fetchall()

        missing_total = len(all_missing)

        # Build missing list
        missing_list = []
        for row in all_missing:
            missing_list.append({
                "match_id": row[0],
                "kickoff_utc": row[1].isoformat() + "Z" if row[1] else None,
                "league_id": row[2],
                "home": row[3],
                "away": row[4],
            })

        # Paginate
        pages = (missing_total + limit - 1) // limit if missing_total > 0 else 0
        start = (page - 1) * limit
        end = start + limit
        paginated = missing_list[start:end]

        # Calculate coverage
        coverage_pct = None
        if matches_total > 0:
            coverage_pct = round(((matches_total - missing_total) / matches_total) * 100, 1)

        # Determine status
        status = "ok"
        if missing_total > 0:
            status = "warn"
        if matches_total > 0 and (missing_total / matches_total) > 0.1:
            status = "red"

        result = {
            "missing": paginated,
            "missing_total": missing_total,
            "matches_total": matches_total,
            "coverage_pct": coverage_pct,
            "total": missing_total,
            "page": page,
            "limit": limit,
            "pages": pages,
            "status": status,
        }

        _missing_preds_cache["data"] = result
        _missing_preds_cache["timestamp"] = now_ts
        _missing_preds_cache["params"] = cache_key

        return _make_v2_wrapper(result, cached=False, cache_age_seconds=0)

    except Exception as e:
        logger.warning(f"Missing predictions endpoint error: {e}")
        base_data["note"] = f"db error: {str(e)[:50]}"
        return _make_v2_wrapper(base_data, cached=False, cache_age_seconds=0)


# Cache for /dashboard/movement/recent.json (activity by recency)
_movement_recent_cache: dict = {"data": None, "timestamp": 0, "ttl": 60, "params": None}


@app.get("/dashboard/movement/recent.json")
async def dashboard_movement_recent(
    request: Request,
    range: str = "24h",
    type: str = None,  # "lineup" or "market"
    limit: int = 50,
    session: AsyncSession = Depends(get_async_session),
):
    """
    V2 endpoint: Recent lineup/market activity ordered by recency.
    TTL: 60s
    Auth: X-Dashboard-Token
    Query params: range (24h|7d), type (lineup|market), limit (max 100)
    """
    if not _verify_dashboard_token(request):
        raise HTTPException(status_code=401, detail="Dashboard access requires valid token.")

    import time as time_module
    from sqlalchemy import text

    # Clamp params
    limit = min(max(1, limit), 100)

    # Validate range
    valid_ranges = {"24h": 24, "7d": 168}
    if range not in valid_ranges:
        range = "24h"
    hours = valid_ranges[range]

    # Validate type
    valid_types = ["lineup", "market", None]
    if type not in valid_types:
        type = None

    now_ts = time_module.time()
    cache_key = f"{range}:{type}:{limit}"

    # Check cache
    if (_movement_recent_cache["data"] and
        _movement_recent_cache["params"] == cache_key and
        (now_ts - _movement_recent_cache["timestamp"]) < _movement_recent_cache["ttl"]):
        cache_age = now_ts - _movement_recent_cache["timestamp"]
        return _make_v2_wrapper(_movement_recent_cache["data"], cached=True, cache_age_seconds=cache_age)

    base_data = {
        "items": [],
        "status": "degraded",
        "note": "upstream unavailable",
    }

    try:
        now_dt = datetime.utcnow()
        cutoff = now_dt - timedelta(hours=hours)

        items = []

        # Get lineup changes (from match_sofascore_lineup captured_at)
        if type is None or type == "lineup":
            lineup_query = """
                SELECT DISTINCT ON (m.id) m.id, m.date, m.league_id,
                       ht.name as home, at.name as away,
                       msl.captured_at
                FROM match_sofascore_lineup msl
                JOIN matches m ON m.id = msl.match_id
                JOIN teams ht ON ht.id = m.home_team_id
                JOIN teams at ON at.id = m.away_team_id
                WHERE msl.captured_at > :cutoff
                ORDER BY m.id, msl.captured_at DESC
                LIMIT :limit
            """
            lineup_result = await session.execute(
                text(lineup_query),
                {"cutoff": cutoff, "limit": limit}
            )
            for row in lineup_result.fetchall():
                items.append({
                    "match_id": row[0],
                    "kickoff_utc": row[1].isoformat() + "Z" if row[1] else None,
                    "league_id": row[2],
                    "home": row[3],
                    "away": row[4],
                    "type": "lineup",
                    "captured_at": row[5].isoformat() + "Z" if row[5] else None,
                    "source": "sofascore",
                })

        # Get market/odds captures (from odds_history)
        if type is None or type == "market":
            market_query = """
                SELECT DISTINCT ON (m.id) m.id, m.date, m.league_id,
                       ht.name as home, at.name as away,
                       oh.recorded_at
                FROM odds_history oh
                JOIN matches m ON m.id = oh.match_id
                JOIN teams ht ON ht.id = m.home_team_id
                JOIN teams at ON at.id = m.away_team_id
                WHERE oh.recorded_at > :cutoff
                  AND NOT COALESCE(oh.quarantined, false)
                ORDER BY m.id, oh.recorded_at DESC
                LIMIT :limit
            """
            market_result = await session.execute(
                text(market_query),
                {"cutoff": cutoff, "limit": limit}
            )
            for row in market_result.fetchall():
                items.append({
                    "match_id": row[0],
                    "kickoff_utc": row[1].isoformat() + "Z" if row[1] else None,
                    "league_id": row[2],
                    "home": row[3],
                    "away": row[4],
                    "type": "market",
                    "captured_at": row[5].isoformat() + "Z" if row[5] else None,
                    "source": "api-football",
                })

        # Sort by captured_at (most recent first) and limit
        items.sort(key=lambda x: x["captured_at"] or "", reverse=True)
        items = items[:limit]

        result = {
            "items": items,
            "status": "ok" if items else "warn",
            "note": "recent activity ordered by recency",
        }

        _movement_recent_cache["data"] = result
        _movement_recent_cache["timestamp"] = now_ts
        _movement_recent_cache["params"] = cache_key

        return _make_v2_wrapper(result, cached=False, cache_age_seconds=0)

    except Exception as e:
        logger.warning(f"Movement recent endpoint error: {e}")
        base_data["note"] = f"db error: {str(e)[:50]}"
        return _make_v2_wrapper(base_data, cached=False, cache_age_seconds=0)


# Cache for /dashboard/movement/top.json (top movers by magnitude)
_movement_top_cache: dict = {"data": None, "timestamp": 0, "ttl": 60, "params": None}


@app.get("/dashboard/movement/top.json")
async def dashboard_movement_top(
    request: Request,
    range: str = "24h",
    type: str = None,  # "lineup" or "market"
    limit: int = 50,
    session: AsyncSession = Depends(get_async_session),
):
    """
    V2 endpoint: Top movers by movement magnitude.
    TTL: 60s
    Auth: X-Dashboard-Token
    Query params: range (24h|7d), type (lineup|market), limit (max 100)

    Movement magnitude:
    - market: max |Δimplied_prob| between first and last snapshot in window
    - lineup: placeholder (no magnitude metric yet, returns empty for type=lineup)
    """
    if not _verify_dashboard_token(request):
        raise HTTPException(status_code=401, detail="Dashboard access requires valid token.")

    import time as time_module
    from sqlalchemy import text

    # Clamp params
    limit = min(max(1, limit), 100)

    # Validate range
    valid_ranges = {"24h": 24, "7d": 168}
    if range not in valid_ranges:
        range = "24h"
    hours = valid_ranges[range]

    # Validate type
    valid_types = ["lineup", "market", None]
    if type not in valid_types:
        type = None

    now_ts = time_module.time()
    cache_key = f"{range}:{type}:{limit}"

    # Check cache
    if (_movement_top_cache["data"] and
        _movement_top_cache["params"] == cache_key and
        (now_ts - _movement_top_cache["timestamp"]) < _movement_top_cache["ttl"]):
        cache_age = now_ts - _movement_top_cache["timestamp"]
        return _make_v2_wrapper(_movement_top_cache["data"], cached=True, cache_age_seconds=cache_age)

    base_data = {
        "movers": [],
        "status": "degraded",
        "note": "upstream unavailable",
    }

    try:
        now_dt = datetime.utcnow()
        cutoff = now_dt - timedelta(hours=hours)

        movers = []

        # Market movers: calculate magnitude as max delta in implied probs
        if type is None or type == "market":
            # Get matches with multiple odds snapshots and calculate movement
            market_query = """
                WITH match_odds_range AS (
                    SELECT
                        match_id,
                        FIRST_VALUE(implied_home) OVER w AS first_implied_home,
                        FIRST_VALUE(implied_draw) OVER w AS first_implied_draw,
                        FIRST_VALUE(implied_away) OVER w AS first_implied_away,
                        LAST_VALUE(implied_home) OVER w AS last_implied_home,
                        LAST_VALUE(implied_draw) OVER w AS last_implied_draw,
                        LAST_VALUE(implied_away) OVER w AS last_implied_away,
                        MAX(recorded_at) OVER (PARTITION BY match_id) AS last_recorded_at,
                        ROW_NUMBER() OVER (PARTITION BY match_id ORDER BY recorded_at DESC) AS rn
                    FROM odds_history
                    WHERE recorded_at > :cutoff
                      AND NOT COALESCE(quarantined, false)
                      AND implied_home IS NOT NULL
                    WINDOW w AS (PARTITION BY match_id ORDER BY recorded_at
                                 ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING)
                ),
                match_movement AS (
                    SELECT
                        match_id,
                        last_recorded_at,
                        GREATEST(
                            ABS(COALESCE(last_implied_home, 0) - COALESCE(first_implied_home, 0)),
                            ABS(COALESCE(last_implied_draw, 0) - COALESCE(first_implied_draw, 0)),
                            ABS(COALESCE(last_implied_away, 0) - COALESCE(first_implied_away, 0))
                        ) AS magnitude
                    FROM match_odds_range
                    WHERE rn = 1
                )
                SELECT m.id, m.date, m.league_id, ht.name, at.name,
                       mm.magnitude, mm.last_recorded_at
                FROM match_movement mm
                JOIN matches m ON m.id = mm.match_id
                JOIN teams ht ON ht.id = m.home_team_id
                JOIN teams at ON at.id = m.away_team_id
                WHERE mm.magnitude > 0.01  -- Filter noise (>1% movement)
                ORDER BY mm.magnitude DESC
                LIMIT :limit
            """
            try:
                market_result = await session.execute(
                    text(market_query),
                    {"cutoff": cutoff, "limit": limit}
                )
                for row in market_result.fetchall():
                    movers.append({
                        "match_id": row[0],
                        "kickoff_utc": row[1].isoformat() + "Z" if row[1] else None,
                        "league_id": row[2],
                        "home": row[3],
                        "away": row[4],
                        "type": "market",
                        "value": round(float(row[5]) * 100, 2) if row[5] else 0,  # As percentage points
                        "captured_at": row[6].isoformat() + "Z" if row[6] else None,
                        "source": "api-football",
                    })
            except Exception as market_err:
                logger.debug(f"Market movers query failed: {market_err}")
                # Continue without market data

        # Lineup movers: no magnitude metric yet
        # For type=lineup, return empty with note
        if type == "lineup":
            result = {
                "movers": [],
                "status": "warn",
                "note": "lineup magnitude metric not implemented yet; use /movement/recent.json for lineup activity",
            }
            return _make_v2_wrapper(result, cached=False, cache_age_seconds=0)

        # Sort by magnitude (value) descending
        movers.sort(key=lambda x: abs(x.get("value", 0)), reverse=True)
        movers = movers[:limit]

        # Determine status
        status = "ok" if movers else "warn"
        note = "top movers by implied probability change (percentage points)"
        if not movers:
            note = "no significant market movements detected in window"

        result = {
            "movers": movers,
            "status": status,
            "note": note,
        }

        _movement_top_cache["data"] = result
        _movement_top_cache["timestamp"] = now_ts
        _movement_top_cache["params"] = cache_key

        return _make_v2_wrapper(result, cached=False, cache_age_seconds=0)

    except Exception as e:
        logger.warning(f"Movement top endpoint error: {e}")
        base_data["note"] = f"db error: {str(e)[:50]}"
        return _make_v2_wrapper(base_data, cached=False, cache_age_seconds=0)


# =============================================================================
# END DASHBOARD V2 ENDPOINTS
# =============================================================================


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
        # AUDIT P0: Derive state from recommendation.status (source of truth)
        rec = report.get("recommendation", {})
        rec_status = rec.get("status", "NO_DATA")
        report_status = report.get("status", "")

        if report_status in ("NO_DATA", "INSUFFICIENT_DATA", "DISABLED"):
            state = "LEARNING"
        elif rec_status in ("SIGNAL_DETECTED", "OVERFITTING_SUSPECTED", "TRACKING"):
            state = rec_status
        elif rec_status == "LEARNING":
            state = "LEARNING"
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
        # AUDIT P0: Use evaluated_with_b for A vs B comparison (where sensor produced predictions)
        samples_evaluated = counts.get("evaluated_with_b", 0)
        samples_evaluated_total = counts.get("evaluated_total", 0)
        samples_pending = counts.get("pending_with_b", 0)
        samples_pending_total = counts.get("pending_total", 0)
        # AUDIT: Expose missing B predictions (sensor was LEARNING when logged)
        missing_b_evaluated = counts.get("missing_b_evaluated", 0)
        missing_b_pending = counts.get("missing_b_pending", 0)
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
            # Counts - AUDIT P0: expose both total and with_b, use with_b for card
            "samples_evaluated": samples_evaluated,  # evaluated_with_b (A vs B comparison)
            "samples_evaluated_total": samples_evaluated_total,  # all evaluated
            "samples_pending": samples_pending,  # pending_with_b (will have B to compare)
            "samples_pending_total": samples_pending_total,  # all pending
            # AUDIT: Expose missing B predictions (sensor was LEARNING when logged)
            # These records are excluded from A vs B comparison metrics
            "missing_b_evaluated": missing_b_evaluated,
            "missing_b_pending": missing_b_pending,
            "missing_b_total": missing_b_evaluated + missing_b_pending,
            # Warning if sensor is ready but there are missing B predictions (needs retry)
            "has_missing_b": (missing_b_evaluated + missing_b_pending) > 0,
            "missing_b_warning": (
                "retry needed" if sensor_info.get("is_ready") and (missing_b_evaluated + missing_b_pending) > 0
                else None
            ),
            "min_samples": min_samples,
            # Accuracy A vs B (Auditor card) - only present if samples >= min_samples
            "accuracy_a_pct": accuracy_a_pct,
            "accuracy_b_pct": accuracy_b_pct,
            "delta_accuracy_pct": delta_accuracy_pct,
            "window_days": sensor_settings.SENSOR_EVAL_WINDOW_DAYS,
            "note": "solo FT evaluados (apples-to-apples con Model A)",
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

    P1-B: Falls back to DB (job_runs table) when Prometheus metrics are unavailable
    (e.g., cold-start after deploy).
    """
    from app.telemetry.metrics import (
        job_last_success_timestamp,
        stats_backfill_ft_pending_gauge,
        fastpath_backlog_ready_gauge,
    )

    now = datetime.utcnow()

    # P1-B: Preload DB fallback data for all jobs
    db_fallback = {}
    try:
        from app.jobs.tracking import get_jobs_health_from_db
        db_fallback = await get_jobs_health_from_db(session)
    except Exception as e:
        logger.debug(f"[JOBS_HEALTH] DB fallback unavailable: {e}")

    # Helper to format timestamp and calculate age
    def job_status(job_name: str, max_gap_minutes: int) -> dict:
        last_success = None
        source = "prometheus"

        # Try Prometheus first
        try:
            ts = job_last_success_timestamp.labels(job=job_name)._value.get()
            if ts and ts > 0:
                last_success = datetime.utcfromtimestamp(ts)
        except Exception:
            pass

        # P1-B: Fallback to DB if Prometheus has no data
        if last_success is None and job_name in db_fallback:
            db_data = db_fallback[job_name]
            if db_data.get("last_success_at"):
                try:
                    # Parse ISO format
                    ts_str = db_data["last_success_at"].rstrip("Z")
                    last_success = datetime.fromisoformat(ts_str)
                    source = "db_fallback"
                except Exception:
                    pass

        if last_success:
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
                "source": source,  # For debugging
            }

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

    # Build top_alert for warn/red status (Auditor Dashboard enhancement)
    top_alert = None
    alerts_count = 0

    if overall in ("warn", "red"):
        # Collect all jobs with their severity for ranking
        job_alerts = []
        jobs_meta = {
            "stats_backfill": {"data": stats_health, "label": "Stats Backfill"},
            "odds_sync": {"data": odds_health, "label": "Odds Sync"},
            "fastpath": {"data": fastpath_health, "label": "Fast-Path Narratives"},
        }

        for job_key, meta in jobs_meta.items():
            job_data = meta["data"]
            job_status_val = job_data.get("status", "unknown")

            if job_status_val in ("warn", "red"):
                alerts_count += 1
                minutes_since = job_data.get("minutes_since_success")

                # Build reason string
                if minutes_since is not None:
                    if minutes_since >= 60:
                        hours_ago = round(minutes_since / 60, 1)
                        reason = f"Last success {hours_ago}h ago"
                    else:
                        reason = f"Last success {int(minutes_since)}m ago"
                else:
                    reason = "No recent success recorded"

                # Add context for specific jobs
                if job_key == "stats_backfill" and job_data.get("ft_pending"):
                    reason += f" ({job_data['ft_pending']} FT pending)"
                elif job_key == "fastpath" and job_data.get("backlog_ready"):
                    reason += f" ({job_data['backlog_ready']} backlog)"

                job_alerts.append({
                    "job_key": job_key,
                    "label": meta["label"],
                    "severity": job_status_val,
                    "reason": reason,
                    "minutes_since_success": minutes_since,
                    "runbook_url": job_data.get("help_url"),
                    # Sort key: red=2, warn=1; then by minutes_since_success desc
                    "_sort_key": (2 if job_status_val == "red" else 1, minutes_since or 0),
                })

        # Select worst job as top_alert
        if job_alerts:
            job_alerts.sort(key=lambda x: x["_sort_key"], reverse=True)
            worst = job_alerts[0]
            top_alert = {
                "job_key": worst["job_key"],
                "label": worst["label"],
                "severity": worst["severity"],
                "reason": worst["reason"],
                "minutes_since_success": worst["minutes_since_success"],
                "runbook_url": worst["runbook_url"],
            }

    result = {
        "status": overall,
        "runbook_url": f"{runbook_base}#p0-jobs-health-scheduler-jobs",
        "stats_backfill": stats_health,
        "odds_sync": odds_health,
        "fastpath": fastpath_health,
    }

    # Only include top_alert fields when there are alerts
    if top_alert:
        result["top_alert"] = top_alert
        result["alerts_count"] = alerts_count

    return result


async def _calculate_sota_enrichment_summary(session) -> dict:
    """
    Calculate SOTA enrichment coverage metrics for OPS dashboard.

    Reports coverage and staleness for:
    - Understat xG data (match_understat_team)
    - Weather forecasts (match_weather)
    - Venue geo coordinates (venue_geo)
    - Team home city profiles (team_home_city_profile)
    - Sofascore XI lineups (match_sofascore_lineup)

    All metrics are best-effort: query failures return "unavailable" status.

    STANDARDIZED SHAPE (per component):
    {
        "status": "ok" | "warn" | "red" | "unavailable",
        "coverage_pct": float,
        "total": int,
        "with_data": int,
        "staleness_hours": float | null,
        "note": string | null
    }
    """
    now = datetime.utcnow()
    result = {"status": "ok", "generated_at": now.isoformat()}

    # Helper to build unavailable response
    def _unavailable(note: str) -> dict:
        return {
            "status": "unavailable",
            "coverage_pct": 0.0,
            "total": 0,
            "with_data": 0,
            "staleness_hours": None,
            "note": note,
        }

    # 1) Understat coverage: FT matches in last 14 days with xG data
    try:
        res = await session.execute(
            text("""
                SELECT
                    COUNT(*) FILTER (WHERE mut.match_id IS NOT NULL) AS with_xg,
                    COUNT(*) AS total_ft
                FROM matches m
                LEFT JOIN match_understat_team mut ON m.id = mut.match_id
                WHERE m.status IN ('FT', 'AET', 'PEN')
                  AND m.date >= NOW() - INTERVAL '14 days'
                  AND m.league_id IN (39, 140, 135, 78, 61)
            """)
        )
        row = res.first()
        with_data = int(row[0] or 0) if row else 0
        total = int(row[1] or 0) if row else 0
        coverage_pct = round(with_data / total * 100, 1) if total > 0 else 0.0

        # Get staleness (latest captured_at)
        res_stale = await session.execute(
            text("""
                SELECT MAX(captured_at) FROM match_understat_team
                WHERE captured_at > NOW() - INTERVAL '7 days'
            """)
        )
        latest_capture = res_stale.scalar()
        staleness_hours = None
        if latest_capture:
            staleness_hours = round((now - latest_capture).total_seconds() / 3600, 1)

        result["understat"] = {
            "status": "ok" if coverage_pct >= 50 else ("warn" if coverage_pct >= 20 else "red"),
            "coverage_pct": coverage_pct,
            "total": total,
            "with_data": with_data,
            "staleness_hours": staleness_hours,
            "note": "FT matches last 14d (top 5 leagues)",
        }
    except Exception as e:
        logger.debug(f"[SOTA] Understat metrics unavailable: {e}")
        result["understat"] = _unavailable(f"Query failed: {str(e)[:50]}")

    # 2) Weather coverage: NS matches in next 48h with weather forecasts
    try:
        res = await session.execute(
            text("""
                SELECT
                    COUNT(*) FILTER (WHERE mw.match_id IS NOT NULL) AS with_weather,
                    COUNT(*) AS total_ns
                FROM matches m
                LEFT JOIN match_weather mw ON m.id = mw.match_id
                WHERE m.status = 'NS'
                  AND m.date >= NOW()
                  AND m.date < NOW() + INTERVAL '48 hours'
            """)
        )
        row = res.first()
        with_data = int(row[0] or 0) if row else 0
        total = int(row[1] or 0) if row else 0
        coverage_pct = round(with_data / total * 100, 1) if total > 0 else 0.0

        result["weather"] = {
            "status": "ok" if coverage_pct >= 50 else ("warn" if coverage_pct >= 10 else "red"),
            "coverage_pct": coverage_pct,
            "total": total,
            "with_data": with_data,
            "staleness_hours": None,  # weather is forward-looking, no staleness
            "note": "NS matches next 48h",
        }
    except Exception as e:
        logger.debug(f"[SOTA] Weather metrics unavailable: {e}")
        result["weather"] = _unavailable(f"Query failed: {str(e)[:50]}")

    # 3) Venue geo coverage: venues from recent matches with coordinates
    try:
        res = await session.execute(
            text("""
                SELECT
                    COUNT(DISTINCT vg.venue_city) AS with_geo,
                    COUNT(DISTINCT m.venue_city) AS total_venues
                FROM matches m
                LEFT JOIN venue_geo vg ON m.venue_city = vg.venue_city
                WHERE m.venue_city IS NOT NULL
                  AND m.date >= NOW() - INTERVAL '30 days'
            """)
        )
        row = res.first()
        with_data = int(row[0] or 0) if row else 0
        total = int(row[1] or 0) if row else 0
        coverage_pct = round(with_data / total * 100, 1) if total > 0 else 0.0

        result["venue_geo"] = {
            "status": "ok" if coverage_pct >= 50 else ("warn" if coverage_pct >= 20 else "red"),
            "coverage_pct": coverage_pct,
            "total": total,
            "with_data": with_data,
            "staleness_hours": None,  # static data, no staleness
            "note": "Venues from matches last 30d",
        }
    except Exception as e:
        logger.debug(f"[SOTA] Venue geo metrics unavailable: {e}")
        result["venue_geo"] = _unavailable(f"Query failed: {str(e)[:50]}")

    # 4) Team profiles coverage: teams with home city profiles
    # ABE 2026-01-25: Report both "all teams" and "active teams (30d)" for clarity
    try:
        # Query 1: All teams (historical debt metric)
        res_all = await session.execute(
            text("""
                SELECT
                    COUNT(*) FILTER (WHERE thcp.team_id IS NOT NULL) AS with_profile,
                    COUNT(*) AS total_teams
                FROM teams t
                LEFT JOIN team_home_city_profile thcp ON t.id = thcp.team_id
            """)
        )
        row_all = res_all.first()
        with_data_all = int(row_all[0] or 0) if row_all else 0
        total_all = int(row_all[1] or 0) if row_all else 0
        coverage_pct_all = round(with_data_all / total_all * 100, 1) if total_all > 0 else 0.0

        # Query 2: Active teams (last 30d) - operational metric for dashboard card
        res_active = await session.execute(
            text("""
                SELECT
                    COUNT(DISTINCT CASE WHEN thcp.team_id IS NOT NULL THEN t.id END) AS with_profile,
                    COUNT(DISTINCT t.id) AS total_teams
                FROM teams t
                JOIN matches m ON t.id = m.home_team_id OR t.id = m.away_team_id
                LEFT JOIN team_home_city_profile thcp ON t.id = thcp.team_id
                WHERE m.date >= NOW() - INTERVAL '30 days'
            """)
        )
        row_active = res_active.first()
        with_data_active = int(row_active[0] or 0) if row_active else 0
        total_active = int(row_active[1] or 0) if row_active else 0
        coverage_pct_active = round(with_data_active / total_active * 100, 1) if total_active > 0 else 0.0

        # Primary metric: active teams (better operational signal)
        result["team_profiles"] = {
            "status": "ok" if coverage_pct_active >= 30 else ("warn" if coverage_pct_active >= 10 else "red"),
            "coverage_pct": coverage_pct_active,
            "total": total_active,
            "with_data": with_data_active,
            "staleness_hours": None,  # static data, no staleness
            "note": f"Active teams (30d). All teams: {with_data_all}/{total_all} ({coverage_pct_all}%)",
            # Detailed breakdown for audit
            "all_teams": {
                "with_data": with_data_all,
                "total": total_all,
                "coverage_pct": coverage_pct_all,
            },
            "active_teams_30d": {
                "with_data": with_data_active,
                "total": total_active,
                "coverage_pct": coverage_pct_active,
            },
        }
    except Exception as e:
        logger.debug(f"[SOTA] Team profiles metrics unavailable: {e}")
        result["team_profiles"] = _unavailable(f"Query failed: {str(e)[:50]}")

    # 5) Sofascore XI coverage: NS matches in next 48h with XI data
    try:
        # Check if tables exist first (they may not be deployed yet)
        table_check = await session.execute(
            text("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables
                    WHERE table_name = 'match_sofascore_lineup'
                )
            """)
        )
        tables_exist = table_check.scalar()

        if tables_exist:
            sofascore_league_ids = ",".join(str(lid) for lid in SOFASCORE_SUPPORTED_LEAGUES)
            res = await session.execute(
                text(f"""
                    SELECT
                        COUNT(*) FILTER (WHERE msl.match_id IS NOT NULL) AS with_xi,
                        COUNT(*) AS total_ns
                    FROM matches m
                    LEFT JOIN match_sofascore_lineup msl ON m.id = msl.match_id
                    WHERE m.status = 'NS'
                      AND m.date >= NOW()
                      AND m.date < NOW() + INTERVAL '48 hours'
                      AND m.league_id IN ({sofascore_league_ids})
                """)
            )
            row = res.first()
            with_data = int(row[0] or 0) if row else 0
            total = int(row[1] or 0) if row else 0
            coverage_pct = round(with_data / total * 100, 1) if total > 0 else 0.0

            # Get staleness (latest captured_at)
            res_stale = await session.execute(
                text("""
                    SELECT MAX(captured_at) FROM match_sofascore_lineup
                    WHERE captured_at > NOW() - INTERVAL '7 days'
                """)
            )
            latest_capture = res_stale.scalar()
            staleness_hours = None
            if latest_capture:
                staleness_hours = round((now - latest_capture).total_seconds() / 3600, 1)

            result["sofascore_xi"] = {
                "status": "ok" if coverage_pct >= 30 else ("warn" if coverage_pct >= 10 else "red"),
                "coverage_pct": coverage_pct,
                "total": total,
                "with_data": with_data,
                "staleness_hours": staleness_hours,
                "note": "NS matches next 48h (SOTA leagues)",
            }
        else:
            # Tables don't exist yet - waiting for migration
            result["sofascore_xi"] = _unavailable("Tables not deployed yet (migration 030)")
    except Exception as e:
        logger.debug(f"[SOTA] Sofascore XI metrics unavailable: {e}")
        result["sofascore_xi"] = _unavailable(f"Query failed: {str(e)[:50]}")

    # Overall status: worst of components (excluding unavailable)
    component_statuses = []
    for key in ["understat", "weather", "venue_geo", "team_profiles", "sofascore_xi"]:
        if result.get(key, {}).get("status") in ("ok", "warn", "red"):
            component_statuses.append(result[key]["status"])

    if "red" in component_statuses:
        result["status"] = "red"
    elif "warn" in component_statuses:
        result["status"] = "warn"
    elif component_statuses:
        result["status"] = "ok"
    else:
        result["status"] = "unavailable"

    return result


async def _calculate_titan_summary() -> dict:
    """
    Calculate TITAN OMNISCIENCE summary for OPS dashboard.

    Reports:
    - feature_matrix row count and tier coverage
    - Job status (last run, success/fail)
    - Progress toward N=50/200 gate

    Lightweight query for ops.json inclusion.
    Creates its own session to avoid scope issues.
    """
    now = datetime.utcnow()
    result = {
        "status": "ok",
        "generated_at": now.isoformat(),
        "feature_matrix": {},
        "job": {},
        "gate": {},
    }

    try:
        async with AsyncSessionLocal() as session:
            # Check if titan schema exists
            schema_check = await session.execute(text("""
                SELECT EXISTS (
                    SELECT FROM information_schema.schemata
                    WHERE schema_name = 'titan'
                )
            """))
            schema_exists = schema_check.scalar()

            if not schema_exists:
                result["status"] = "unavailable"
                result["note"] = "TITAN schema not deployed"
                return result

            # Feature matrix stats
            fm_stats = await session.execute(text("""
                SELECT
                    COUNT(*) as total,
                    COUNT(*) FILTER (WHERE tier1_complete = TRUE) as tier1,
                    COUNT(*) FILTER (WHERE tier1b_complete = TRUE) as tier1b,
                    COUNT(*) FILTER (WHERE tier1c_complete = TRUE) as tier1c,
                    COUNT(*) FILTER (WHERE tier1d_complete = TRUE) as tier1d,
                    COUNT(*) FILTER (WHERE outcome IS NOT NULL) as with_outcome
                FROM titan.feature_matrix
            """))
            row = fm_stats.first()
            if row:
                total = int(row[0] or 0)
                tier1 = int(row[1] or 0)
                tier1b = int(row[2] or 0)
                tier1c = int(row[3] or 0)
                tier1d = int(row[4] or 0)
                with_outcome = int(row[5] or 0)

                result["feature_matrix"] = {
                    "total_rows": total,
                    "tier1_complete": tier1,
                    "tier1b_complete": tier1b,
                    "tier1c_complete": tier1c,
                    "tier1d_complete": tier1d,
                    "with_outcome": with_outcome,
                    "tier1b_pct": round(tier1b / total * 100, 1) if total > 0 else 0,
                    "tier1c_pct": round(tier1c / total * 100, 1) if total > 0 else 0,
                }

                # Gate progress (ABE-defined thresholds)
                result["gate"] = {
                    "n_current": with_outcome,
                    "n_target_pilot": 50,
                    "n_target_prelim": 200,
                    "n_target_formal": 500,
                    "ready_for_pilot": with_outcome >= 50,
                    "ready_for_prelim": with_outcome >= 200,
                    "ready_for_formal": with_outcome >= 500,
                    "pct_to_pilot": round(min(100, with_outcome / 50 * 100), 1),
                    "pct_to_prelim": round(min(100, with_outcome / 200 * 100), 1),
                    "pct_to_formal": round(min(100, with_outcome / 500 * 100), 1),
                }

            # Job status (last TITAN runner run)
            job_stats = await session.execute(text("""
                SELECT status, started_at, metrics
                FROM job_runs
                WHERE job_name = 'titan_feature_matrix_runner'
                ORDER BY started_at DESC
                LIMIT 1
            """))
            job_row = job_stats.first()
            if job_row:
                result["job"] = {
                    "last_status": job_row[0],
                    "last_run_at": job_row[1].isoformat() if job_row[1] else None,
                    "last_metrics": job_row[2] if job_row[2] else {},
                }
            else:
                result["job"] = {
                    "last_status": "never_run",
                    "last_run_at": None,
                    "note": "Job has not run yet - will start on next 2h interval",
                }

            # Determine overall status
            if result["feature_matrix"].get("total_rows", 0) == 0:
                result["status"] = "warn"
                result["note"] = "No data in feature_matrix yet"
            elif result["gate"].get("ready_for_formal"):
                result["status"] = "ok"
                result["note"] = f"Ready for formal eval (N={with_outcome})"
            elif result["gate"].get("ready_for_prelim"):
                result["status"] = "ok"
                result["note"] = f"Ready for preliminary eval (N={with_outcome}/500)"
            elif result["gate"].get("ready_for_pilot"):
                result["status"] = "ok"
                result["note"] = f"Ready for pilot eval (N={with_outcome}/500)"
            else:
                result["status"] = "building"
                result["note"] = f"Accumulating data: {with_outcome}/500 for formal gate"

    except Exception as e:
        logger.warning(f"[TITAN] Summary calculation failed: {e}")
        result["status"] = "error"
        result["error"] = str(e)[:100]

    return result


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
    # P1: Only WARN for staleness if coverage is NOT 100% - if all NS and FT have predictions,
    # staleness is just informational (DAILY-SAVE runs once/day)
    elif hours_since_last and hours_since_last > 12 and ns_next_48h > 0:
        # Check if coverage is perfect - if so, staleness is just informational
        if ns_next_48h_missing == 0 and ft_48h_missing == 0:
            # Coverage is 100%, staleness is OK (just means daily job ran earlier)
            status = "ok"
            status_reason = None  # No alert needed
        else:
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


# =============================================================================
# SENTRY HEALTH (server-side aggregation for ops dashboard)
# =============================================================================

_sentry_health_cache: dict = {
    "data": None,
    "timestamp": 0,
    "ttl": 90,  # 90 seconds cache (balance between freshness and API limits)
}

# Sentry API thresholds for status determination
_SENTRY_CRITICAL_THRESHOLD_1H = 5  # new_issues_1h >= 5 → critical
_SENTRY_WARNING_THRESHOLD_24H = 20  # new_issues_24h >= 20 → warning


async def _fetch_sentry_health() -> dict:
    """
    Fetch Sentry health metrics via Sentry API (server-side only).

    Best-effort: returns degraded status if credentials missing or API fails.
    Uses in-memory cache with 90s TTL to avoid API rate limits.

    Sentry API endpoints used:
    - GET /api/0/projects/{org}/{project}/issues/ (for issue counts)
    - Query params: statsPeriod=1h, statsPeriod=24h, query=is:unresolved

    Returns:
        dict with status, counts, top_issues, etc.
    """
    import time as time_module
    import httpx

    now_ts = time_module.time()
    now_iso = datetime.utcnow().isoformat()

    # Check cache first
    if _sentry_health_cache["data"] and (now_ts - _sentry_health_cache["timestamp"]) < _sentry_health_cache["ttl"]:
        cached = _sentry_health_cache["data"].copy()
        cached["cached"] = True
        cached["cache_age_seconds"] = int(now_ts - _sentry_health_cache["timestamp"])
        return cached

    # Base response structure (degraded fallback)
    base_response = {
        "status": "degraded",
        "cached": False,
        "cache_age_seconds": 0,
        "generated_at": now_iso,
        "project": {
            "org_slug": settings.SENTRY_ORG or None,
            "project_slug": settings.SENTRY_PROJECT_SLUG or None,
            "env": settings.SENTRY_ENV or "production",
        },
        "counts": {
            "new_issues_1h": 0,
            "new_issues_24h": 0,
            "active_issues_1h": 0,
            "active_issues_24h": 0,
            "open_issues": 0,
        },
        "last_event_at": None,
        "top_issues": [],
        "note": "best-effort, aggregated server-side",
    }

    # Check if credentials are configured
    if not settings.SENTRY_AUTH_TOKEN or not settings.SENTRY_ORG or not settings.SENTRY_PROJECT_SLUG:
        base_response["error"] = "Sentry credentials not configured"
        _sentry_health_cache["data"] = base_response
        _sentry_health_cache["timestamp"] = now_ts
        return base_response

    org_slug = settings.SENTRY_ORG
    project_slug = settings.SENTRY_PROJECT_SLUG
    auth_token = settings.SENTRY_AUTH_TOKEN
    env_filter = settings.SENTRY_ENV or "production"

    headers = {
        "Authorization": f"Bearer {auth_token}",
        "Content-Type": "application/json",
    }

    try:
        async with httpx.AsyncClient(headers=headers, timeout=10.0) as client:
            issues_url = f"https://sentry.io/api/0/projects/{org_slug}/{project_slug}/issues/"

            # Calculate time boundaries for filtering by lastSeen/firstSeen
            from datetime import timedelta
            now_dt = datetime.utcnow()
            one_hour_ago = (now_dt - timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%S")
            one_day_ago = (now_dt - timedelta(hours=24)).strftime("%Y-%m-%dT%H:%M:%S")

            # Build query - try with env filter first, fallback without if empty
            env_query = f"environment:{env_filter}" if env_filter else ""

            # 1) Fetch ALL unresolved issues (open_issues - no time filter)
            #    Sort by lastSeen desc to get most recent activity first
            params_open = {
                "query": f"is:unresolved {env_query}".strip(),
                "sort": "date",  # sort by lastSeen descending
                "limit": 100,
            }
            resp_open = await client.get(issues_url, params=params_open)

            # Parse open issues response
            all_issues = []
            env_filter_excluded = False

            if resp_open.status_code == 200:
                all_issues = resp_open.json() if isinstance(resp_open.json(), list) else []

                # If env filter returned 0, try without env to check if filter excluded results
                if len(all_issues) == 0 and env_query:
                    params_no_env = {
                        "query": "is:unresolved",
                        "sort": "date",
                        "limit": 100,
                    }
                    resp_no_env = await client.get(issues_url, params=params_no_env)
                    if resp_no_env.status_code == 200:
                        issues_no_env = resp_no_env.json() if isinstance(resp_no_env.json(), list) else []
                        if len(issues_no_env) > 0:
                            env_filter_excluded = True
                            all_issues = issues_no_env  # Use unfiltered results

            # Calculate counts from the fetched issues
            open_issues = len(all_issues)
            new_issues_1h = 0
            new_issues_24h = 0
            active_issues_1h = 0
            active_issues_24h = 0
            last_event_at = None
            top_issues = []

            for issue in all_issues:
                first_seen = issue.get("firstSeen", "")
                last_seen = issue.get("lastSeen", "")

                # New issues (by firstSeen - when issue was created)
                if first_seen and first_seen >= one_hour_ago:
                    new_issues_1h += 1
                if first_seen and first_seen >= one_day_ago:
                    new_issues_24h += 1

                # Active issues (by lastSeen - recent activity)
                if last_seen and last_seen >= one_hour_ago:
                    active_issues_1h += 1
                if last_seen and last_seen >= one_day_ago:
                    active_issues_24h += 1

            # Get last_event_at from most recent issue (already sorted by lastSeen desc)
            if all_issues:
                last_event_at = all_issues[0].get("lastSeen")

                # Extract top issues (top 3 by recent activity, from active_24h)
                active_24h_issues = [
                    i for i in all_issues
                    if i.get("lastSeen", "") >= one_day_ago
                ]
                # Sort by count descending for top issues
                sorted_active = sorted(
                    active_24h_issues,
                    key=lambda x: int(x.get("count", 0)),
                    reverse=True
                )[:3]

                for issue in sorted_active:
                    title = issue.get("title", "Unknown")[:80]
                    title = title.replace("@", "[at]")  # Basic sanitization
                    top_issues.append({
                        "title": title,
                        "count": int(issue.get("count", 0)),
                        "level": issue.get("level", "error"),
                        "last_seen": issue.get("lastSeen"),
                    })

            # Determine status based on activity thresholds (use active, not new)
            status = "ok"
            if active_issues_1h >= _SENTRY_CRITICAL_THRESHOLD_1H:
                status = "critical"
            elif active_issues_24h >= _SENTRY_WARNING_THRESHOLD_24H:
                status = "warning"

            # Build note
            note = "best-effort, aggregated server-side"
            if env_filter_excluded:
                note += f"; env filter '{env_filter}' excluded results, showing all"

            result = {
                "status": status,
                "cached": False,
                "cache_age_seconds": 0,
                "generated_at": now_iso,
                "project": {
                    "org_slug": org_slug,
                    "project_slug": project_slug,
                    "env": env_filter if not env_filter_excluded else "(all)",
                },
                "counts": {
                    "new_issues_1h": new_issues_1h,
                    "new_issues_24h": new_issues_24h,
                    "active_issues_1h": active_issues_1h,
                    "active_issues_24h": active_issues_24h,
                    "open_issues": open_issues,
                },
                "last_event_at": last_event_at,
                "top_issues": top_issues,
                "note": note,
            }

            # Update cache
            _sentry_health_cache["data"] = result
            _sentry_health_cache["timestamp"] = now_ts

            return result

    except httpx.TimeoutException:
        base_response["error"] = "Sentry API timeout"
        logger.warning("Sentry health fetch timeout")
    except httpx.HTTPStatusError as e:
        base_response["error"] = f"Sentry API HTTP {e.response.status_code}"
        logger.warning(f"Sentry health fetch HTTP error: {e}")
    except Exception as e:
        base_response["error"] = f"Sentry fetch error: {str(e)[:50]}"
        logger.warning(f"Sentry health fetch error: {e}")

    # Cache degraded response too (to avoid hammering on errors)
    _sentry_health_cache["data"] = base_response
    _sentry_health_cache["timestamp"] = now_ts
    return base_response


def _get_providers_health() -> dict:
    """Get health status for external data providers."""
    try:
        from app.etl.api_football import get_provider_health
        api_football = get_provider_health()
    except Exception as e:
        logger.warning(f"Could not get API-Football provider health: {e}")
        api_football = {"status": "unknown", "error": str(e)}

    return {
        "api_football": api_football,
    }


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

    # Sentry health - fetch aggregated metrics (best-effort, cached)
    sentry_health: dict = await _fetch_sentry_health()

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
        # Per PIT Protocol v2: Piloto=50, Preliminar=200, Formal=500
        # Aligned with TITAN formal gate for consistency
        TARGET_PIT_SNAPSHOTS_30D = int(os.environ.get("TARGET_PIT_SNAPSHOTS_30D", "500"))
        TARGET_PIT_BETS_30D = int(os.environ.get("TARGET_PIT_BETS_30D", "500"))
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
        # SOTA ENRICHMENT (Understat xG, Weather, Venue Geo coverage)
        # =============================================================
        sota_enrichment_data = await _calculate_sota_enrichment_summary(session)

        # =============================================================
        # LLM COST (Gemini token usage from PostMatchAudit)
        # =============================================================
        llm_cost_data = {"provider": "gemini", "status": "unavailable"}
        try:
            # Rollback any previous failed transaction state
            await session.rollback()

            # Use pricing from settings (single source of truth)
            MODEL_PRICING = settings.GEMINI_PRICING
            DEFAULT_PRICE_IN = settings.GEMINI_PRICE_INPUT
            DEFAULT_PRICE_OUT = settings.GEMINI_PRICE_OUTPUT

            # Build dynamic CASE statements from MODEL_PRICING
            # Groups models by same pricing to reduce SQL complexity
            def build_pricing_case_sql() -> str:
                """Generate SQL CASE for model-specific pricing from settings."""
                # Group models by pricing tuple (input, output)
                pricing_groups: dict[tuple[float, float], list[str]] = {}
                for model, prices in MODEL_PRICING.items():
                    key = (prices["input"], prices["output"])
                    if key not in pricing_groups:
                        pricing_groups[key] = []
                    pricing_groups[key].append(model)

                case_parts = []
                for (price_in, price_out), models in pricing_groups.items():
                    models_sql = ", ".join(f"'{m}'" for m in models)
                    case_parts.append(
                        f"WHEN llm_narrative_model IN ({models_sql}) THEN "
                        f"(COALESCE(llm_narrative_tokens_in, 0) * {price_in} + "
                        f"COALESCE(llm_narrative_tokens_out, 0) * {price_out}) / 1000000.0"
                    )

                # Add ELSE for unknown models (uses default pricing via params)
                case_parts.append(
                    "ELSE (COALESCE(llm_narrative_tokens_in, 0) * :default_in + "
                    "COALESCE(llm_narrative_tokens_out, 0) * :default_out) / 1000000.0"
                )

                return "CASE " + " ".join(case_parts) + " END"

            pricing_case_sql = build_pricing_case_sql()

            # Helper function to build LLM cost query with specific interval
            # Note: INTERVAL cannot be parameterized in PostgreSQL, must use literal
            def llm_cost_query(interval_literal: str) -> str:
                return f"""
                    SELECT
                        COUNT(*) AS request_count,
                        COALESCE(SUM(llm_narrative_tokens_in), 0) AS tokens_in,
                        COALESCE(SUM(llm_narrative_tokens_out), 0) AS tokens_out,
                        COALESCE(SUM({pricing_case_sql}), 0) AS cost_usd
                    FROM post_match_audits
                    WHERE llm_narrative_model LIKE 'gemini%'
                      AND (COALESCE(llm_narrative_tokens_in, 0) > 0
                           OR COALESCE(llm_narrative_tokens_out, 0) > 0)
                      AND created_at > NOW() - INTERVAL '{interval_literal}'
                """

            query_params = {"default_in": DEFAULT_PRICE_IN, "default_out": DEFAULT_PRICE_OUT}

            # 24h metrics
            res_24h = await session.execute(
                text(llm_cost_query("24 hours")), query_params
            )
            row_24h = res_24h.first()
            requests_24h = int(row_24h[0] or 0) if row_24h else 0
            tokens_in_24h = int(row_24h[1] or 0) if row_24h else 0
            tokens_out_24h = int(row_24h[2] or 0) if row_24h else 0
            cost_24h = float(row_24h[3] or 0) if row_24h else 0.0

            # 7d metrics
            res_7d = await session.execute(
                text(llm_cost_query("7 days")), query_params
            )
            row_7d = res_7d.first()
            requests_7d = int(row_7d[0] or 0) if row_7d else 0
            tokens_in_7d = int(row_7d[1] or 0) if row_7d else 0
            tokens_out_7d = int(row_7d[2] or 0) if row_7d else 0
            cost_7d = float(row_7d[3] or 0) if row_7d else 0.0

            # 28d metrics (matches Google AI Studio billing window)
            res_28d = await session.execute(
                text(llm_cost_query("28 days")), query_params
            )
            row_28d = res_28d.first()
            requests_28d = int(row_28d[0] or 0) if row_28d else 0
            tokens_in_28d = int(row_28d[1] or 0) if row_28d else 0
            tokens_out_28d = int(row_28d[2] or 0) if row_28d else 0
            cost_28d = float(row_28d[3] or 0) if row_28d else 0.0

            # Total accumulated cost (all time)
            res_total = await session.execute(
                text(
                    f"""
                    SELECT
                        COUNT(*) AS request_count,
                        COALESCE(SUM(llm_narrative_tokens_in), 0) AS tokens_in,
                        COALESCE(SUM(llm_narrative_tokens_out), 0) AS tokens_out,
                        COALESCE(SUM({pricing_case_sql}), 0) AS cost_usd
                    FROM post_match_audits
                    WHERE llm_narrative_model LIKE 'gemini%'
                      AND (COALESCE(llm_narrative_tokens_in, 0) > 0
                           OR COALESCE(llm_narrative_tokens_out, 0) > 0)
                    """
                ),
                query_params,
            )
            row_total = res_total.first()
            requests_total = int(row_total[0] or 0) if row_total else 0
            tokens_in_total = int(row_total[1] or 0) if row_total else 0
            tokens_out_total = int(row_total[2] or 0) if row_total else 0
            cost_total = float(row_total[3] or 0) if row_total else 0.0

            # Calculate avg cost per request
            avg_cost_per_request = cost_24h / requests_24h if requests_24h > 0 else 0.0

            # Status: warn if cost_24h > $1 or avg_cost > $0.01
            status = "ok"
            if cost_24h > 1.0 or avg_cost_per_request > 0.01:
                status = "warn"

            # Model usage breakdown by window (for tooltip/audit)
            # Best-effort: if fails, omit model_usage_* (don't break llm_cost)
            model_usage_28d = None
            model_usage_7d = None
            model_usage_24h = None
            try:
                def model_usage_query(interval_literal: str) -> str:
                    return f"""
                        SELECT
                            llm_narrative_model,
                            COUNT(*) AS requests,
                            COALESCE(SUM(llm_narrative_tokens_in), 0) AS tokens_in,
                            COALESCE(SUM(llm_narrative_tokens_out), 0) AS tokens_out
                        FROM post_match_audits
                        WHERE llm_narrative_model LIKE 'gemini%'
                          AND (COALESCE(llm_narrative_tokens_in, 0) > 0
                               OR COALESCE(llm_narrative_tokens_out, 0) > 0)
                          AND created_at > NOW() - INTERVAL '{interval_literal}'
                        GROUP BY llm_narrative_model
                        ORDER BY requests DESC
                    """

                def parse_model_usage(rows) -> dict | None:
                    if not rows:
                        return None
                    models = {}
                    top_model = None
                    max_requests = 0
                    for row in rows:
                        model_name = row[0]
                        req_count = int(row[1] or 0)
                        t_in = int(row[2] or 0)
                        t_out = int(row[3] or 0)
                        models[model_name] = {
                            "requests": req_count,
                            "tokens_in": t_in,
                            "tokens_out": t_out,
                        }
                        if req_count > max_requests:
                            max_requests = req_count
                            top_model = model_name
                    if not models:
                        return None
                    return {"top_model": top_model, "models": models}

                # 28d usage
                res_usage_28d = await session.execute(text(model_usage_query("28 days")))
                model_usage_28d = parse_model_usage(res_usage_28d.fetchall())

                # 7d usage
                res_usage_7d = await session.execute(text(model_usage_query("7 days")))
                model_usage_7d = parse_model_usage(res_usage_7d.fetchall())

                # 24h usage
                res_usage_24h = await session.execute(text(model_usage_query("24 hours")))
                model_usage_24h = parse_model_usage(res_usage_24h.fetchall())

            except Exception as usage_err:
                logger.debug(f"Could not calculate model usage breakdown: {usage_err}")
                # Continue without model_usage_* (best-effort)

            # Get current model pricing for transparency
            current_model = settings.GEMINI_MODEL
            current_pricing = MODEL_PRICING.get(
                current_model, {"input": DEFAULT_PRICE_IN, "output": DEFAULT_PRICE_OUT}
            )

            llm_cost_data = {
                "provider": "gemini",
                "model": current_model,
                # Pricing transparency for auditing (current model)
                "pricing_input_per_1m": current_pricing["input"],
                "pricing_output_per_1m": current_pricing["output"],
                # All model pricing for reference
                "model_pricing": MODEL_PRICING,
                # Cost metrics (calculated with per-model pricing)
                "cost_24h_usd": round(cost_24h, 4),
                "cost_7d_usd": round(cost_7d, 4),
                "cost_28d_usd": round(cost_28d, 4),
                "cost_total_usd": round(cost_total, 2),
                # Request counts (all requests with tokens, not filtered by status)
                "requests_24h": requests_24h,
                "requests_7d": requests_7d,
                "requests_28d": requests_28d,
                "requests_total": requests_total,
                # Legacy fields for backward compatibility
                "requests_ok_24h": requests_24h,
                "requests_ok_7d": requests_7d,
                "requests_ok_total": requests_total,
                "avg_cost_per_ok_24h": round(avg_cost_per_request, 6),
                # Token breakdown
                "tokens_in_24h": tokens_in_24h,
                "tokens_out_24h": tokens_out_24h,
                "tokens_in_7d": tokens_in_7d,
                "tokens_out_7d": tokens_out_7d,
                "tokens_in_28d": tokens_in_28d,
                "tokens_out_28d": tokens_out_28d,
                "tokens_in_total": tokens_in_total,
                "tokens_out_total": tokens_out_total,
                "status": status,
                "note": "Cost calculated per-model from settings.GEMINI_PRICING. 28d window matches Google AI Studio billing.",
                "pricing_source": "config.GEMINI_PRICING",
                # Model usage breakdown (best-effort, may be None)
                **({"model_usage_28d": model_usage_28d} if model_usage_28d else {}),
                **({"model_usage_7d": model_usage_7d} if model_usage_7d else {}),
                **({"model_usage_24h": model_usage_24h} if model_usage_24h else {}),
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
    # LIVE SUMMARY STATS (iOS Live Score Polling)
    # =============================================================
    live_summary_stats = {
        "cache_ttl_seconds": _live_summary_cache["ttl"],
        "cache_timestamp": _live_summary_cache["timestamp"],
        "cache_age_seconds": round(time.time() - _live_summary_cache["timestamp"], 1) if _live_summary_cache["timestamp"] else None,
        "cached_live_matches": len(_live_summary_cache["data"]["matches"]) if _live_summary_cache["data"] else 0,
    }

    # =============================================================
    # ML MODEL STATUS
    # =============================================================
    ml_model_info = {
        "loaded": ml_engine.model is not None,
        "version": ml_engine.model_version,
        "source": "file",  # Currently only file-based loading
        "model_path": str(ml_engine.model_path),
    }
    if ml_engine.model is not None:
        try:
            ml_model_info["n_features"] = ml_engine.model.n_features_in_
        except AttributeError:
            pass

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
        "sentry": sentry_health,
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
        "sota_enrichment": sota_enrichment_data,
        "titan": await _calculate_titan_summary(),
        "coverage_by_league": coverage_by_league,
        "ml_model": ml_model_info,
        "live_summary": live_summary_stats,
        "db_pool": get_pool_status(),
        "providers": _get_providers_health(),
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
    /* Alerts Bell */
    .alerts-bell {{
      position: relative;
      cursor: pointer;
      font-size: 1.25rem;
      margin-left: 0.75rem;
      padding: 0.35rem;
    }}
    .alerts-bell:hover {{ opacity: 0.8; }}
    .alerts-badge {{
      position: absolute;
      top: -4px;
      right: -6px;
      background: var(--red);
      color: white;
      font-size: 0.65rem;
      font-weight: 700;
      min-width: 16px;
      height: 16px;
      border-radius: 50%;
      display: flex;
      align-items: center;
      justify-content: center;
      line-height: 1;
    }}
    .alerts-dropdown {{
      display: none;
      position: absolute;
      top: 100%;
      right: 0;
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 0.5rem;
      width: 320px;
      max-height: 400px;
      overflow-y: auto;
      z-index: 1000;
      box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    }}
    .alerts-dropdown.open {{ display: block; }}
    .alerts-header {{
      padding: 0.75rem;
      border-bottom: 1px solid var(--border);
      display: flex;
      justify-content: space-between;
      align-items: center;
      font-size: 0.8rem;
      font-weight: 600;
    }}
    .alerts-header button {{
      background: var(--border);
      border: none;
      color: var(--muted);
      padding: 0.25rem 0.5rem;
      border-radius: 0.25rem;
      font-size: 0.7rem;
      cursor: pointer;
    }}
    .alerts-header button:hover {{ background: var(--blue); color: white; }}
    .alerts-list {{ padding: 0; }}
    .alert-item {{
      padding: 0.6rem 0.75rem;
      border-bottom: 1px solid var(--border);
      font-size: 0.8rem;
    }}
    .alert-item:last-child {{ border-bottom: none; }}
    .alert-item.critical {{ border-left: 3px solid var(--red); }}
    .alert-item.warning {{ border-left: 3px solid var(--yellow); }}
    .alert-item.info {{ border-left: 3px solid var(--blue); }}
    .alert-item.resolved {{ opacity: 0.6; }}
    .alert-title {{ font-weight: 600; margin-bottom: 0.2rem; }}
    .alert-meta {{ color: var(--muted); font-size: 0.7rem; }}
    .alert-empty {{ padding: 1.5rem; text-align: center; color: var(--muted); font-size: 0.85rem; }}
    /* Toast Notifications */
    .toast-container {{
      position: fixed;
      top: 1rem;
      right: 1rem;
      z-index: 9999;
      display: flex;
      flex-direction: column;
      gap: 0.5rem;
    }}
    .toast {{
      background: var(--card);
      border: 1px solid var(--red);
      border-left: 4px solid var(--red);
      border-radius: 0.5rem;
      padding: 0.75rem 1rem;
      min-width: 280px;
      max-width: 380px;
      box-shadow: 0 4px 12px rgba(0,0,0,0.4);
      animation: slideIn 0.3s ease;
    }}
    .toast-title {{ font-weight: 600; font-size: 0.85rem; margin-bottom: 0.2rem; }}
    .toast-message {{ color: var(--muted); font-size: 0.75rem; }}
    .toast-close {{
      position: absolute;
      top: 0.5rem;
      right: 0.5rem;
      background: none;
      border: none;
      color: var(--muted);
      cursor: pointer;
      font-size: 1rem;
    }}
    @keyframes slideIn {{
      from {{ transform: translateX(100%); opacity: 0; }}
      to {{ transform: translateX(0); opacity: 1; }}
    }}
  </style>
</head>
<body>
  <div class="toast-container" id="toastContainer"></div>
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
          <a class="nav-link" data-path="/dashboard/ops/daily_comparison" href="/dashboard/ops/daily_comparison">Daily</a>
          <a class="nav-link" data-path="/dashboard/ops/league_stats" href="/dashboard/ops/league_stats">Ligas</a>
          <a class="nav-link" data-path="/dashboard/ops/history" href="/dashboard/ops/history">History</a>
          <a class="nav-link" data-path="/dashboard/ops/logs" href="/dashboard/ops/logs">Logs (debug)</a>
          <a class="nav-link" href="/ops/logout" style="margin-left: auto; color: #f87171;">Logout</a>
          <!-- Alerts Bell -->
          <div class="alerts-bell" id="alertsBell" onclick="toggleAlertsDropdown()">
            🔔
            <span class="alerts-badge" id="alertsBadge" style="display: none;">0</span>
            <div class="alerts-dropdown" id="alertsDropdown">
              <div class="alerts-header">
                <span>Alertas</span>
                <button onclick="event.stopPropagation(); ackAllAlerts();">Marcar leídas</button>
              </div>
              <div class="alerts-list" id="alertsList">
                <div class="alert-empty">Sin alertas</div>
              </div>
            </div>
          </div>
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
      <h3>Progreso hacia Re-test / Alpha<span class="info-icon">i<span class="tooltip">Métricas de preparación para re-evaluar el modelo. Fases PIT v2: Piloto=50 (✓), Preliminar=200 (actual), Formal=500. Re-test cuando: Bets ≥ 200 y Baseline Coverage ≥ 60%.</span></span></h3>
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

    // =========================================================================
    // ALERTS BELL: Polling + Toast Notifications
    // =========================================================================
    (function() {{
      const seenAlertIds = new Set(JSON.parse(localStorage.getItem('seenAlertIds') || '[]'));
      let alertsDropdownOpen = false;

      // Get token for API calls
      function getAuthHeaders() {{
        const params = new URLSearchParams(window.location.search);
        const token = params.get('token');
        const headers = {{}};
        if (token) headers['X-Dashboard-Token'] = token;
        return headers;
      }}

      // Toggle dropdown
      window.toggleAlertsDropdown = function() {{
        const dropdown = document.getElementById('alertsDropdown');
        alertsDropdownOpen = !alertsDropdownOpen;
        dropdown.classList.toggle('open', alertsDropdownOpen);
      }};

      // Close dropdown on outside click
      document.addEventListener('click', (e) => {{
        const bell = document.getElementById('alertsBell');
        if (bell && !bell.contains(e.target)) {{
          document.getElementById('alertsDropdown').classList.remove('open');
          alertsDropdownOpen = false;
        }}
      }});

      // Mark all as read
      window.ackAllAlerts = async function() {{
        try {{
          const token = new URLSearchParams(window.location.search).get('token');
          const url = token ? `/dashboard/ops/alerts/ack?token=${{encodeURIComponent(token)}}` : '/dashboard/ops/alerts/ack';
          await fetch(url, {{
            method: 'POST',
            headers: {{ 'Content-Type': 'application/json', ...getAuthHeaders() }},
            body: JSON.stringify({{ ack_all: true }})
          }});
          fetchAlerts();
        }} catch (e) {{
          console.error('Ack alerts error:', e);
        }}
      }};

      // Show toast for critical alerts
      function showToast(alert) {{
        const container = document.getElementById('toastContainer');
        const toast = document.createElement('div');
        toast.className = 'toast';
        toast.style.position = 'relative';
        toast.innerHTML = `
          <button class="toast-close" onclick="this.parentElement.remove()">×</button>
          <div class="toast-title">${{alert.title}}</div>
          <div class="toast-message">${{alert.message || ''}}</div>
        `;
        container.appendChild(toast);
        // Auto-remove after 10s
        setTimeout(() => toast.remove(), 10000);
      }}

      // Format time ago
      function timeAgo(isoString) {{
        if (!isoString) return '';
        const diff = Date.now() - new Date(isoString).getTime();
        const mins = Math.floor(diff / 60000);
        if (mins < 1) return 'ahora';
        if (mins < 60) return `${{mins}}m`;
        const hrs = Math.floor(mins / 60);
        if (hrs < 24) return `${{hrs}}h`;
        return `${{Math.floor(hrs / 24)}}d`;
      }}

      // Fetch and render alerts
      async function fetchAlerts() {{
        try {{
          const token = new URLSearchParams(window.location.search).get('token');
          const url = token
            ? `/dashboard/ops/alerts.json?status=firing&limit=20&token=${{encodeURIComponent(token)}}`
            : '/dashboard/ops/alerts.json?status=firing&limit=20';
          const res = await fetch(url, {{ headers: getAuthHeaders() }});
          if (!res.ok) return;
          const data = await res.json();

          // Update badge
          const badge = document.getElementById('alertsBadge');
          if (data.unread_count > 0) {{
            badge.textContent = data.unread_count > 9 ? '9+' : data.unread_count;
            badge.style.display = 'flex';
          }} else {{
            badge.style.display = 'none';
          }}

          // Render list
          const list = document.getElementById('alertsList');
          if (data.items && data.items.length > 0) {{
            list.innerHTML = data.items.map(a => `
              <div class="alert-item ${{a.severity}} ${{a.status === 'resolved' ? 'resolved' : ''}}">
                <div class="alert-title">${{a.title}}</div>
                <div class="alert-meta">
                  ${{a.severity.toUpperCase()}} · ${{timeAgo(a.last_seen_at)}}
                  ${{a.source_url ? `<a href="${{a.source_url}}" target="_blank" style="color:var(--blue);">→ Grafana</a>` : ''}}
                </div>
              </div>
            `).join('');

            // Show toast for new CRITICAL alerts
            data.items.forEach(a => {{
              if (a.severity === 'critical' && a.status === 'firing' && !seenAlertIds.has(a.id)) {{
                showToast(a);
                seenAlertIds.add(a.id);
              }}
            }});
            // Persist seen IDs
            localStorage.setItem('seenAlertIds', JSON.stringify([...seenAlertIds].slice(-100)));
          }} else {{
            list.innerHTML = '<div class="alert-empty">Sin alertas activas</div>';
          }}
        }} catch (e) {{
          console.error('Fetch alerts error:', e);
        }}
      }}

      // Initial fetch + polling every 20s
      fetchAlerts();
      setInterval(fetchAlerts, 20000);
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


# -----------------------------------------------------------------------------
# Upcoming Matches for Dashboard Overview Card
# -----------------------------------------------------------------------------
_upcoming_matches_cache = {
    "data": None,
    "timestamp": 0,
    "ttl": 60,  # seconds - recommended by auditor
}


@app.get("/dashboard/upcoming_matches.json")
async def get_upcoming_matches_dashboard(
    request: Request,
    hours: int = 24,
    limit: int = 20,
    session: AsyncSession = Depends(get_async_session),
):
    """
    Upcoming scheduled matches for dashboard Overview card.

    Returns next N hours of scheduled matches with prediction status.
    Uses EXISTS subquery to avoid N+1 queries (P0 requirement).

    Auth: X-Dashboard-Token header (read-only, separate from X-API-Key).

    Response:
    {
        "generated_at": "2026-01-22T...",
        "cached": true/false,
        "cache_age_seconds": 45,
        "data": {
            "upcoming": [
                {
                    "id": 12345,
                    "home": "Real Madrid",
                    "away": "Barcelona",
                    "kickoff_iso": "2026-01-22T20:00:00Z",
                    "league_name": "La Liga",
                    "has_prediction": true
                }
            ]
        }
    }
    """
    if not _verify_dashboard_token(request):
        raise HTTPException(status_code=401, detail="Dashboard access requires valid token.")

    # Enforce limits (P1 recommendation)
    hours = min(max(hours, 1), 72)  # 1-72 hours
    limit = min(max(limit, 1), 50)  # 1-50 matches

    now = time.time()

    # Check cache
    if (
        _upcoming_matches_cache["data"] is not None
        and (now - _upcoming_matches_cache["timestamp"]) < _upcoming_matches_cache["ttl"]
    ):
        cached_data = _upcoming_matches_cache["data"]
        return {
            "generated_at": cached_data["generated_at"],
            "cached": True,
            "cache_age_seconds": round(now - _upcoming_matches_cache["timestamp"], 1),
            "data": cached_data["data"],
        }

    # Build query with EXISTS subquery to avoid N+1 (P0 requirement)
    from sqlalchemy import exists, literal_column
    from sqlalchemy.orm import aliased

    now_dt = datetime.utcnow()
    cutoff_dt = now_dt + timedelta(hours=hours)

    # Subquery: EXISTS prediction for this match
    has_prediction_subq = (
        exists()
        .where(Prediction.match_id == Match.id)
        .correlate(Match)
    )

    # Main query with team names via JOIN
    home_team = aliased(Team, name="home_team")
    away_team = aliased(Team, name="away_team")

    query = (
        select(
            Match.id,
            Match.date,
            Match.league_id,
            home_team.name.label("home_name"),
            away_team.name.label("away_name"),
            has_prediction_subq.label("has_prediction"),
        )
        .join(home_team, Match.home_team_id == home_team.id)
        .join(away_team, Match.away_team_id == away_team.id)
        .where(Match.status == "NS")
        .where(Match.date >= now_dt)
        .where(Match.date <= cutoff_dt)
        .order_by(Match.date)
        .limit(limit)
    )

    result = await session.execute(query)
    rows = result.all()

    # Build league name lookup from COMPETITIONS (single source of truth)
    from app.etl.competitions import COMPETITIONS
    league_name_by_id: dict[int, str] = {
        lid: comp.name for lid, comp in COMPETITIONS.items() if comp.name
    }

    # Format response
    upcoming = []
    for row in rows:
        upcoming.append({
            "id": row.id,
            "home": row.home_name,
            "away": row.away_name,
            "kickoff_iso": row.date.isoformat() + "Z" if row.date else None,
            "league_name": league_name_by_id.get(row.league_id, f"League {row.league_id}"),
            "has_prediction": bool(row.has_prediction),
        })

    generated_at = datetime.utcnow().isoformat() + "Z"

    # Update cache
    _upcoming_matches_cache["data"] = {
        "generated_at": generated_at,
        "data": {"upcoming": upcoming},
    }
    _upcoming_matches_cache["timestamp"] = now

    return {
        "generated_at": generated_at,
        "cached": False,
        "cache_age_seconds": 0,
        "data": {"upcoming": upcoming},
    }


# -----------------------------------------------------------------------------
# Dashboard Matches Table Endpoint
# -----------------------------------------------------------------------------

# Cache for matches table - keyed by query params
# TTL: 15s for LIVE, 60s for NS/FT (P1 auditor: 10-15s LIVE for Ops responsiveness)
_matches_table_cache: dict[str, dict] = {}
_MATCHES_CACHE_TTL_LIVE = 15  # seconds
_MATCHES_CACHE_TTL_DEFAULT = 60  # seconds


def _get_matches_cache_key(
    status: str,
    hours: int,
    league_id: int | None,
    page: int,
    limit: int,
    match_id: int | None = None,
    from_time: str | None = None,
    to_time: str | None = None,
) -> str:
    """Generate cache key including all query params (P1 guardrail)."""
    if from_time and to_time:
        return f"matches:{status}:range:{from_time}:{to_time}:{league_id}:{page}:{limit}:{match_id}"
    return f"matches:{status}:{hours}:{league_id}:{page}:{limit}:{match_id}"


@app.get("/dashboard/matches.json")
async def get_matches_dashboard(
    request: Request,
    status: str = "NS",  # NS, LIVE, FT, or ALL
    hours: int = 168,  # 7 days default for table view
    from_time: str | None = None,  # ISO8601 UTC start time (overrides hours)
    to_time: str | None = None,  # ISO8601 UTC end time (overrides hours)
    league_id: int | None = None,
    match_id: int | None = None,  # optional exact match lookup (deep-link support)
    page: int = 1,
    limit: int = 50,
    session: AsyncSession = Depends(get_async_session),
):
    """
    Matches table for dashboard /matches page.

    Returns paginated matches with filtering by status, time window, and league.
    Uses EXISTS subquery to avoid N+1 queries.

    Auth: X-Dashboard-Token header (read-only).

    Query params:
    - status: NS (scheduled), LIVE (in-play), FT (finished), ALL
    - hours: time window (1-168, default 168 = 7 days)
    - from_time: ISO8601 UTC start time (overrides hours if both from_time and to_time provided)
    - to_time: ISO8601 UTC end time (overrides hours if both from_time and to_time provided)
    - league_id: optional filter by league
    - page: pagination (1-indexed)
    - limit: rows per page (1-100)

    Response:
    {
        "generated_at": "2026-01-22T...",
        "cached": true/false,
        "cache_age_seconds": 15,
        "data": {
            "matches": [...],
            "total": 234,
            "page": 1,
            "limit": 50,
            "pages": 5
        }
    }
    """
    if not _verify_dashboard_token(request):
        raise HTTPException(status_code=401, detail="Dashboard access requires valid token.")

    # Normalize and validate params
    status = status.upper()
    valid_statuses = {"NS", "LIVE", "FT", "ALL"}
    if status not in valid_statuses:
        status = "ALL"

    hours = min(max(hours, 1), 168)  # 1-168 hours (7 days max)
    page = max(page, 1)
    limit = min(max(limit, 1), 100)

    # Parse from_time/to_time if provided (ISO8601 UTC)
    parsed_from_time = None
    parsed_to_time = None
    if from_time and to_time:
        try:
            # Parse ISO8601 format (e.g., 2026-01-23T00:00:00Z)
            parsed_from_time = datetime.fromisoformat(from_time.replace("Z", "+00:00")).replace(tzinfo=None)
            parsed_to_time = datetime.fromisoformat(to_time.replace("Z", "+00:00")).replace(tzinfo=None)
        except ValueError:
            # Invalid format, ignore and use hours-based window
            pass

    # Cache TTL based on status
    cache_ttl = _MATCHES_CACHE_TTL_LIVE if status == "LIVE" else _MATCHES_CACHE_TTL_DEFAULT
    # Include from_time/to_time in cache key when used
    cache_key = _get_matches_cache_key(
        status, hours, league_id, page, limit, match_id=match_id,
        from_time=from_time if parsed_from_time else None,
        to_time=to_time if parsed_to_time else None
    )

    now = time.time()

    # Check cache
    if cache_key in _matches_table_cache:
        cached_entry = _matches_table_cache[cache_key]
        age = now - cached_entry["timestamp"]
        if age < cache_ttl:
            return {
                "generated_at": cached_entry["generated_at"],
                "cached": True,
                "cache_age_seconds": round(age, 1),
                "data": cached_entry["data"],
            }

    # Build query with GROUP BY to deduplicate prediction rows
    from sqlalchemy import func
    from sqlalchemy.orm import aliased

    now_dt = datetime.utcnow()

    # Time window: use from_time/to_time if provided, otherwise calculate from hours
    if parsed_from_time and parsed_to_time:
        # Explicit date range from calendar view
        start_dt = parsed_from_time
        end_dt = parsed_to_time
    elif status == "NS":
        # Upcoming: now to +hours
        start_dt = now_dt
        end_dt = now_dt + timedelta(hours=hours)
    elif status == "FT":
        # Finished: -hours to now
        start_dt = now_dt - timedelta(hours=hours)
        end_dt = now_dt
    else:
        # LIVE or ALL: both directions
        start_dt = now_dt - timedelta(hours=hours)
        end_dt = now_dt + timedelta(hours=hours)

    # Team aliases
    home_team = aliased(Team, name="home_team")
    away_team = aliased(Team, name="away_team")

    # Base query with LEFT JOINs for predictions and weather
    # Use GROUP BY and MAX to get one row per match when there are multiple predictions
    # Weather: use raw SQL subquery with DISTINCT ON to get latest forecast per match
    weather_subq = text("""
        SELECT DISTINCT ON (match_id)
            match_id,
            temp_c,
            humidity,
            wind_ms,
            precip_mm,
            precip_prob,
            cloudcover,
            is_daylight
        FROM match_weather
        ORDER BY match_id, forecast_horizon_hours ASC
    """).columns(
        column("match_id"),
        column("temp_c"),
        column("humidity"),
        column("wind_ms"),
        column("precip_mm"),
        column("precip_prob"),
        column("cloudcover"),
        column("is_daylight"),
    ).subquery("weather")

    base_query = (
        select(
            Match.id,
            Match.date,
            Match.league_id,
            Match.status,
            Match.home_goals,
            Match.away_goals,
            Match.elapsed,
            Match.elapsed_extra,
            Match.venue_name,
            Match.venue_city,
            home_team.name.label("home_name"),
            away_team.name.label("away_name"),
            # Model A (production) prediction - use MAX to pick one value
            func.max(Prediction.home_prob).label("model_a_home"),
            func.max(Prediction.draw_prob).label("model_a_draw"),
            func.max(Prediction.away_prob).label("model_a_away"),
            # Market odds (frozen at prediction time)
            func.max(Prediction.frozen_odds_home).label("market_home"),
            func.max(Prediction.frozen_odds_draw).label("market_draw"),
            func.max(Prediction.frozen_odds_away).label("market_away"),
            # Shadow/Two-Stage prediction
            func.max(ShadowPrediction.shadow_home_prob).label("shadow_home"),
            func.max(ShadowPrediction.shadow_draw_prob).label("shadow_draw"),
            func.max(ShadowPrediction.shadow_away_prob).label("shadow_away"),
            # Sensor B prediction
            func.max(SensorPrediction.b_home_prob).label("sensor_b_home"),
            func.max(SensorPrediction.b_draw_prob).label("sensor_b_draw"),
            func.max(SensorPrediction.b_away_prob).label("sensor_b_away"),
            # Weather forecast (use direct columns, added to GROUP BY since subquery has DISTINCT ON)
            weather_subq.c.temp_c.label("weather_temp_c"),
            weather_subq.c.humidity.label("weather_humidity"),
            weather_subq.c.wind_ms.label("weather_wind_ms"),
            weather_subq.c.precip_mm.label("weather_precip_mm"),
            weather_subq.c.precip_prob.label("weather_precip_prob"),
            weather_subq.c.cloudcover.label("weather_cloudcover"),
            weather_subq.c.is_daylight.label("weather_is_daylight"),
        )
        .join(home_team, Match.home_team_id == home_team.id)
        .join(away_team, Match.away_team_id == away_team.id)
        .outerjoin(Prediction, Prediction.match_id == Match.id)
        .outerjoin(ShadowPrediction, ShadowPrediction.match_id == Match.id)
        .outerjoin(SensorPrediction, SensorPrediction.match_id == Match.id)
        .outerjoin(weather_subq, weather_subq.c.match_id == Match.id)
        .group_by(
            Match.id,
            Match.date,
            Match.league_id,
            Match.status,
            Match.home_goals,
            Match.away_goals,
            Match.elapsed,
            Match.elapsed_extra,
            Match.venue_name,
            Match.venue_city,
            home_team.name,
            away_team.name,
            # Weather fields (from DISTINCT ON subquery, so safe to group by)
            weather_subq.c.temp_c,
            weather_subq.c.humidity,
            weather_subq.c.wind_ms,
            weather_subq.c.precip_mm,
            weather_subq.c.precip_prob,
            weather_subq.c.cloudcover,
            weather_subq.c.is_daylight,
        )
    )

    # Optional exact match lookup (deep-link support).
    # When match_id is provided we ignore time window/status/league filters to avoid false negatives.
    if match_id is not None:
        base_query = base_query.where(Match.id == match_id)
    else:
        base_query = base_query.where(Match.date >= start_dt).where(Match.date <= end_dt)

    # Status filter
    if match_id is None and status == "LIVE":
        # LIVE includes: 1H, HT, 2H, ET, BT, P, SUSP, INT, LIVE
        live_statuses = ["1H", "HT", "2H", "ET", "BT", "P", "SUSP", "INT", "LIVE"]
        base_query = base_query.where(Match.status.in_(live_statuses))
    elif match_id is None and status == "NS":
        base_query = base_query.where(Match.status == "NS")
    elif match_id is None and status == "FT":
        # FT includes: FT, AET, PEN
        ft_statuses = ["FT", "AET", "PEN"]
        base_query = base_query.where(Match.status.in_(ft_statuses))
    # ALL: no status filter

    # League filter
    if match_id is None and league_id is not None:
        base_query = base_query.where(Match.league_id == league_id)

    # Count total for pagination (count distinct match IDs)
    # Note: We just count Match rows, no need to join Team tables for count
    count_query = (
        select(func.count(Match.id))
    )

    if match_id is not None:
        count_query = count_query.where(Match.id == match_id)
    else:
        count_query = count_query.where(Match.date >= start_dt).where(Match.date <= end_dt)

    if match_id is None and status == "LIVE":
        count_query = count_query.where(Match.status.in_(["1H", "HT", "2H", "ET", "BT", "P", "SUSP", "INT", "LIVE"]))
    elif match_id is None and status == "NS":
        count_query = count_query.where(Match.status == "NS")
    elif match_id is None and status == "FT":
        count_query = count_query.where(Match.status.in_(["FT", "AET", "PEN"]))
    if match_id is None and league_id is not None:
        count_query = count_query.where(Match.league_id == league_id)

    total_result = await session.execute(count_query)
    total = total_result.scalar() or 0

    # Order and paginate
    # For match_id we always return a single match (if found).
    if match_id is not None:
        page = 1
        limit = 1
        offset = 0
        query = base_query.limit(1)
    else:
        offset = (page - 1) * limit
        query = base_query.order_by(Match.date.desc() if status == "FT" else Match.date).offset(offset).limit(limit)

    result = await session.execute(query)
    rows = result.all()

    # League names from COMPETITIONS with extended fallback
    from app.etl.competitions import COMPETITIONS

    # Extended fallback for leagues not in COMPETITIONS
    LEAGUE_NAMES_EXTENDED: dict[int, str] = {
        1: "World Cup", 2: "Champions League", 3: "Europa League",
        39: "Premier League", 40: "Championship", 61: "Ligue 1",
        78: "Bundesliga", 135: "Serie A", 140: "La Liga",
        94: "Primeira Liga", 88: "Eredivisie", 203: "Süper Lig",
        239: "Liga BetPlay", 253: "MLS", 262: "Liga MX",
        128: "Argentina Primera", 71: "Brasileirão",
        848: "Conference League", 45: "FA Cup", 143: "Copa del Rey",
        242: "Ecuador Liga Pro", 250: "Paraguay Primera",
    }

    # League ID to country mapping for flags
    LEAGUE_COUNTRY: dict[int, str] = {
        # International
        1: "World", 2: "World", 3: "World", 848: "World",
        4: "World", 5: "World", 6: "World", 7: "World",
        9: "World", 10: "World", 11: "World", 13: "World",
        22: "World", 29: "World", 30: "World", 31: "World",
        32: "World", 33: "World", 34: "World", 37: "World",
        # Europe Top 5
        39: "England", 40: "England", 45: "England",
        140: "Spain", 143: "Spain",
        135: "Italy",
        78: "Germany",
        61: "France",
        # Europe Secondary
        94: "Portugal", 88: "Netherlands", 144: "Belgium",
        203: "Turkey",
        # Americas
        253: "USA", 262: "Mexico",
        128: "Argentina", 71: "Brazil",
        239: "Colombia", 242: "Ecuador", 250: "Paraguay",
        265: "Chile", 268: "Uruguay", 281: "Peru",
        299: "Venezuela", 344: "Bolivia",
        # Middle East
        307: "Saudi-Arabia",
    }

    # Build league name lookup: COMPETITIONS takes priority, then extended fallback
    league_name_by_id: dict[int, str] = LEAGUE_NAMES_EXTENDED.copy()
    for lid, comp in COMPETITIONS.items():
        if comp.name:
            league_name_by_id[lid] = comp.name

    # Format response
    matches = []
    for row in rows:
        match_data = {
            "id": row.id,
            "kickoff_iso": row.date.isoformat() + "Z" if row.date else None,
            "league_id": row.league_id,
            "league_name": league_name_by_id.get(row.league_id, f"League {row.league_id}"),
            "league_country": LEAGUE_COUNTRY.get(row.league_id, ""),
            "home": row.home_name,
            "away": row.away_name,
            "status": row.status,
        }

        # Venue (stadium name and city)
        if row.venue_name or row.venue_city:
            match_data["venue"] = {
                "name": row.venue_name,
                "city": row.venue_city,
            }

        # Weather forecast (if available)
        if row.weather_temp_c is not None:
            match_data["weather"] = {
                "temp_c": round(row.weather_temp_c, 1),
                "humidity": round(row.weather_humidity, 0) if row.weather_humidity else None,
                "wind_ms": round(row.weather_wind_ms, 1) if row.weather_wind_ms else None,
                "precip_mm": round(row.weather_precip_mm, 1) if row.weather_precip_mm else None,
                "precip_prob": round(row.weather_precip_prob, 0) if row.weather_precip_prob else None,
                "cloudcover": round(row.weather_cloudcover, 0) if row.weather_cloudcover else None,
                "is_daylight": bool(row.weather_is_daylight) if row.weather_is_daylight is not None else None,
            }

        # Score (only if played/playing)
        if row.home_goals is not None and row.away_goals is not None:
            match_data["score"] = {"home": row.home_goals, "away": row.away_goals}

        # Elapsed (only if live)
        if row.elapsed is not None:
            match_data["elapsed"] = row.elapsed
            if row.elapsed_extra is not None:
                match_data["elapsed_extra"] = row.elapsed_extra

        # Market odds (converted from decimal odds to implied probabilities)
        if row.market_home is not None:
            match_data["market"] = {
                "home": round(1 / row.market_home, 3) if row.market_home > 0 else None,
                "draw": round(1 / row.market_draw, 3) if row.market_draw and row.market_draw > 0 else None,
                "away": round(1 / row.market_away, 3) if row.market_away and row.market_away > 0 else None,
            }

        # Model A prediction
        if row.model_a_home is not None:
            match_data["model_a"] = {
                "home": round(row.model_a_home, 3),
                "draw": round(row.model_a_draw, 3),
                "away": round(row.model_a_away, 3),
            }

        # Shadow/Two-Stage prediction
        if row.shadow_home is not None:
            match_data["shadow"] = {
                "home": round(row.shadow_home, 3),
                "draw": round(row.shadow_draw, 3),
                "away": round(row.shadow_away, 3),
            }

        # Sensor B prediction
        if row.sensor_b_home is not None:
            match_data["sensor_b"] = {
                "home": round(row.sensor_b_home, 3),
                "draw": round(row.sensor_b_draw, 3),
                "away": round(row.sensor_b_away, 3),
            }

        matches.append(match_data)

    generated_at = datetime.utcnow().isoformat() + "Z"
    pages = (total + limit - 1) // limit if limit > 0 else 1

    response_data = {
        "matches": matches,
        "total": total,
        "page": page,
        "limit": limit,
        "pages": pages,
    }

    # Update cache
    _matches_table_cache[cache_key] = {
        "generated_at": generated_at,
        "timestamp": now,
        "data": response_data,
    }

    # Clean old cache entries (simple LRU: keep last 50)
    if len(_matches_table_cache) > 50:
        oldest_key = min(_matches_table_cache.keys(), key=lambda k: _matches_table_cache[k]["timestamp"])
        del _matches_table_cache[oldest_key]

    return {
        "generated_at": generated_at,
        "cached": False,
        "cache_age_seconds": 0,
        "data": response_data,
    }


# -----------------------------------------------------------------------------
# Dashboard Jobs History Endpoint
# -----------------------------------------------------------------------------

# Cache for jobs table
_jobs_table_cache: dict[str, dict] = {}
_JOBS_CACHE_TTL = 30  # seconds


def _get_jobs_cache_key(status: str | None, job_name: str | None, hours: int, page: int, limit: int) -> str:
    """Generate cache key including all query params."""
    return f"jobs:{status}:{job_name}:{hours}:{page}:{limit}"


@app.get("/dashboard/jobs.json")
async def get_jobs_dashboard(
    request: Request,
    status: str | None = None,  # ok, error, rate_limited, budget_exceeded
    job_name: str | None = None,  # Filter by job name
    hours: int = 24,  # Time window (default 24h)
    page: int = 1,
    limit: int = 50,
    session: AsyncSession = Depends(get_async_session),
):
    """
    Job runs history for dashboard /jobs page.

    Returns paginated job runs with filtering by status and job name.

    Auth: X-Dashboard-Token header (read-only).

    Query params:
    - status: ok, error, rate_limited, budget_exceeded (optional)
    - job_name: stats_backfill, odds_sync, fastpath, etc. (optional)
    - hours: time window (1-168, default 24)
    - page: pagination (1-indexed)
    - limit: rows per page (1-100)

    Response:
    {
        "generated_at": "2026-01-22T...",
        "cached": true/false,
        "cache_age_seconds": 15,
        "data": {
            "runs": [...],
            "total": 234,
            "page": 1,
            "limit": 50,
            "pages": 5,
            "jobs_summary": {...}  // Per-job health summary
        }
    }
    """
    if not _verify_dashboard_token(request):
        raise HTTPException(status_code=401, detail="Dashboard access requires valid token.")

    # Normalize and validate params
    hours = min(max(hours, 1), 168)  # 1-168 hours (7 days max)
    page = max(page, 1)
    limit = min(max(limit, 1), 100)

    cache_key = _get_jobs_cache_key(status, job_name, hours, page, limit)
    now = time.time()

    # Check cache
    if cache_key in _jobs_table_cache:
        cached_entry = _jobs_table_cache[cache_key]
        age = now - cached_entry["timestamp"]
        if age < _JOBS_CACHE_TTL:
            return {
                "generated_at": cached_entry["generated_at"],
                "cached": True,
                "cache_age_seconds": round(age, 1),
                "data": cached_entry["data"],
            }

    from sqlalchemy import func

    now_dt = datetime.utcnow()
    cutoff_dt = now_dt - timedelta(hours=hours)

    # Base query
    base_query = (
        select(JobRun)
        .where(JobRun.started_at >= cutoff_dt)
    )

    # Status filter
    if status:
        base_query = base_query.where(JobRun.status == status)

    # Job name filter
    if job_name:
        base_query = base_query.where(JobRun.job_name == job_name)

    # Count total for pagination
    count_query = select(func.count()).select_from(base_query.subquery())
    total_result = await session.execute(count_query)
    total = total_result.scalar() or 0

    # Order and paginate
    offset = (page - 1) * limit
    query = base_query.order_by(JobRun.started_at.desc()).offset(offset).limit(limit)

    result = await session.execute(query)
    rows = result.scalars().all()

    # Format response
    runs = []
    for row in rows:
        run_data = {
            "id": row.id,
            "job_name": row.job_name,
            "status": row.status,
            "started_at": row.started_at.isoformat() + "Z" if row.started_at else None,
            "finished_at": row.finished_at.isoformat() + "Z" if row.finished_at else None,
            "duration_ms": row.duration_ms,
        }

        # Only include error if failed
        if row.status == "error" and row.error_message:
            run_data["error"] = row.error_message[:500]  # Truncate long errors

        # Include metrics if present
        if row.metrics:
            run_data["metrics"] = row.metrics

        runs.append(run_data)

    # Get jobs summary (per-job health) using existing function
    from app.jobs.tracking import get_jobs_health_from_db
    jobs_summary = await get_jobs_health_from_db(session)

    generated_at = datetime.utcnow().isoformat() + "Z"
    pages = (total + limit - 1) // limit if limit > 0 else 1

    response_data = {
        "runs": runs,
        "total": total,
        "page": page,
        "limit": limit,
        "pages": pages,
        "jobs_summary": jobs_summary,
    }

    # Update cache
    _jobs_table_cache[cache_key] = {
        "generated_at": generated_at,
        "timestamp": now,
        "data": response_data,
    }

    # Clean old cache entries (simple LRU: keep last 20)
    if len(_jobs_table_cache) > 20:
        oldest_key = min(_jobs_table_cache.keys(), key=lambda k: _jobs_table_cache[k]["timestamp"])
        del _jobs_table_cache[oldest_key]

    return {
        "generated_at": generated_at,
        "cached": False,
        "cache_age_seconds": 0,
        "data": response_data,
    }


# -----------------------------------------------------------------------------
# Dashboard Data Quality Endpoint (tabular checks for /data-quality page)
# -----------------------------------------------------------------------------
_data_quality_cache: dict[str, dict] = {}
_DATA_QUALITY_CACHE_TTL = 45  # seconds (auditor: 30–60s)
_data_quality_detail_cache: dict[str, dict] = {}
_DATA_QUALITY_DETAIL_CACHE_TTL = 45  # seconds


def _normalize_multi_param(values: list[str] | None) -> list[str]:
    """
    Normalize multi-select query params.
    Supports repeated params (?status=a&status=b) and comma-separated (?status=a,b).
    """
    if not values:
        return []
    out: list[str] = []
    for v in values:
        if not v:
            continue
        parts = [p.strip() for p in v.split(",")] if "," in v else [v.strip()]
        out.extend([p for p in parts if p])
    # Deduplicate, preserve order
    seen = set()
    deduped = []
    for v in out:
        k = v.lower()
        if k in seen:
            continue
        seen.add(k)
        deduped.append(v)
    return deduped


def _dq_status_from_count(
    count: int | None,
    *,
    fail_if_gt: int = 0,
    warn_if_gt: int = 0,
) -> str:
    """Map numeric check value to passing|warning|failing."""
    if count is None:
        return "warning"
    if count > fail_if_gt:
        return "failing"
    if count > warn_if_gt:
        return "warning"
    return "passing"


async def _build_data_quality_checks(session: AsyncSession) -> list[dict]:
    """
    Build a stable list of Data Quality checks.
    Best-effort: if a single source fails, degrade that check (do not fail whole endpoint).
    """
    now_iso = datetime.utcnow().isoformat() + "Z"
    checks: list[dict] = []

    # Helper to append checks with consistent shape
    def add_check(
        *,
        check_id: str,
        name: str,
        category: str,
        status: str,
        current_value,
        threshold,
        affected_count: int,
        description: str | None,
    ) -> None:
        checks.append(
            {
                "id": check_id,
                "name": name,
                "category": category,
                "status": status,
                "last_run_at": now_iso,
                "current_value": current_value,
                "threshold": threshold,
                "affected_count": affected_count,
                "description": description,
            }
        )

    # 1) Quarantined odds (24h) - should be 0
    try:
        res = await session.execute(
            text(
                """
                SELECT COUNT(*) FROM odds_history
                WHERE quarantined = true
                  AND recorded_at > NOW() - INTERVAL '24 hours'
                """
            )
        )
        quarantined_odds_24h = int(res.scalar() or 0)
        add_check(
            check_id="dq_quarantined_odds_24h",
            name="Quarantined odds (24h)",
            category="odds",
            status=_dq_status_from_count(quarantined_odds_24h, fail_if_gt=0, warn_if_gt=0),
            current_value=quarantined_odds_24h,
            threshold=0,
            affected_count=quarantined_odds_24h,
            description="Odds snapshots flagged as quarantined in the last 24h (should be 0).",
        )
    except Exception as e:
        add_check(
            check_id="dq_quarantined_odds_24h",
            name="Quarantined odds (24h)",
            category="odds",
            status="warning",
            current_value=None,
            threshold=0,
            affected_count=0,
            description=f"Degraded: could not query odds_history ({str(e)[:80]}).",
        )

    # 2) Tainted matches (7d) - should be 0
    try:
        res = await session.execute(
            text(
                """
                SELECT COUNT(*) FROM matches
                WHERE tainted = true
                  AND date > NOW() - INTERVAL '7 days'
                """
            )
        )
        tainted_matches_7d = int(res.scalar() or 0)
        add_check(
            check_id="dq_tainted_matches_7d",
            name="Tainted matches (7d)",
            category="consistency",
            status=_dq_status_from_count(tainted_matches_7d, fail_if_gt=0, warn_if_gt=0),
            current_value=tainted_matches_7d,
            threshold=0,
            affected_count=tainted_matches_7d,
            description="Matches flagged as tainted in the last 7 days (should be 0).",
        )
    except Exception as e:
        add_check(
            check_id="dq_tainted_matches_7d",
            name="Tainted matches (7d)",
            category="consistency",
            status="warning",
            current_value=None,
            threshold=0,
            affected_count=0,
            description=f"Degraded: could not query matches.tainted ({str(e)[:80]}).",
        )

    # 3) Unmapped teams (missing logo_url) - warning if > 0
    try:
        res = await session.execute(
            text(
                """
                SELECT COUNT(DISTINCT t.id) FROM teams t
                WHERE t.logo_url IS NULL
                """
            )
        )
        unmapped_teams = int(res.scalar() or 0)
        add_check(
            check_id="dq_unmapped_teams",
            name="Unmapped teams (missing logo)",
            category="completeness",
            status=_dq_status_from_count(unmapped_teams, fail_if_gt=999999999, warn_if_gt=0),
            current_value=unmapped_teams,
            threshold=0,
            affected_count=unmapped_teams,
            description="Teams missing logo_url (proxy for incomplete mapping).",
        )
    except Exception as e:
        add_check(
            check_id="dq_unmapped_teams",
            name="Unmapped teams (missing logo)",
            category="completeness",
            status="warning",
            current_value=None,
            threshold=0,
            affected_count=0,
            description=f"Degraded: could not query teams.logo_url ({str(e)[:80]}).",
        )

    # 4) Odds desync near-term (6h) - warning if > 0
    try:
        res = await session.execute(
            text(
                """
                SELECT COUNT(DISTINCT m.id)
                FROM matches m
                JOIN odds_snapshots os ON os.match_id = m.id
                WHERE m.status = 'NS'
                  AND m.date BETWEEN NOW() AND NOW() + INTERVAL '6 hours'
                  AND os.odds_freshness = 'live'
                  AND os.snapshot_type = 'lineup_confirmed'
                  AND os.snapshot_at >= NOW() - INTERVAL '120 minutes'
                  AND (m.odds_home IS NULL OR m.odds_draw IS NULL OR m.odds_away IS NULL)
                """
            )
        )
        odds_desync_6h = int(res.scalar() or 0)
        add_check(
            check_id="dq_odds_desync_6h",
            name="Odds desync (next 6h)",
            category="freshness",
            status=_dq_status_from_count(odds_desync_6h, fail_if_gt=999999999, warn_if_gt=0),
            current_value=odds_desync_6h,
            threshold=0,
            affected_count=odds_desync_6h,
            description="NS matches in next 6h with live lineup_confirmed snapshot but NULL odds in matches.",
        )
    except Exception as e:
        add_check(
            check_id="dq_odds_desync_6h",
            name="Odds desync (next 6h)",
            category="freshness",
            status="warning",
            current_value=None,
            threshold=0,
            affected_count=0,
            description=f"Degraded: could not query odds_snapshots/matches odds fields ({str(e)[:80]}).",
        )

    # 5) Odds desync critical (90m) - failing if > 0
    try:
        res = await session.execute(
            text(
                """
                SELECT COUNT(DISTINCT m.id)
                FROM matches m
                JOIN odds_snapshots os ON os.match_id = m.id
                WHERE m.status = 'NS'
                  AND m.date BETWEEN NOW() AND NOW() + INTERVAL '90 minutes'
                  AND os.odds_freshness = 'live'
                  AND os.snapshot_type = 'lineup_confirmed'
                  AND os.snapshot_at >= NOW() - INTERVAL '120 minutes'
                  AND (m.odds_home IS NULL OR m.odds_draw IS NULL OR m.odds_away IS NULL)
                """
            )
        )
        odds_desync_90m = int(res.scalar() or 0)
        add_check(
            check_id="dq_odds_desync_90m",
            name="Odds desync (next 90m)",
            category="odds",
            status=_dq_status_from_count(odds_desync_90m, fail_if_gt=0, warn_if_gt=0),
            current_value=odds_desync_90m,
            threshold=0,
            affected_count=odds_desync_90m,
            description="CRITICAL: NS matches in next 90m missing odds despite recent live lineup_confirmed snapshot.",
        )
    except Exception as e:
        add_check(
            check_id="dq_odds_desync_90m",
            name="Odds desync (next 90m)",
            category="odds",
            status="warning",
            current_value=None,
            threshold=0,
            affected_count=0,
            description=f"Degraded: could not query odds_snapshots/matches odds fields ({str(e)[:80]}).",
        )

    # =========================================================================
    # SOTA ENRICHMENT CHECKS (Understat, Weather, Venue Geo, Team Profiles)
    # =========================================================================

    # 6) Understat coverage: FT matches in last 14d (Top-5 leagues) with xG
    try:
        res = await session.execute(
            text(
                """
                SELECT
                    COUNT(*) FILTER (WHERE mut.match_id IS NOT NULL) AS with_xg,
                    COUNT(*) AS total_ft
                FROM matches m
                LEFT JOIN match_understat_team mut ON m.id = mut.match_id
                WHERE m.status IN ('FT', 'AET', 'PEN')
                  AND m.date >= NOW() - INTERVAL '14 days'
                  AND m.league_id IN (39, 140, 135, 78, 61)
                """
            )
        )
        row = res.first()
        with_xg = int(row[0] or 0) if row else 0
        total_ft = int(row[1] or 0) if row else 0
        missing_xg = total_ft - with_xg
        coverage_pct = round(with_xg / total_ft * 100, 1) if total_ft > 0 else 0.0
        # Status: warn if coverage < 60%, fail if < 30%
        if coverage_pct < 30:
            status = "failing"
        elif coverage_pct < 60:
            status = "warning"
        else:
            status = "passing"
        add_check(
            check_id="dq_understat_coverage_ft_14d",
            name="Understat xG coverage (14d)",
            category="coverage",
            status=status,
            current_value=f"{coverage_pct}%",
            threshold="≥60%",
            affected_count=missing_xg,
            description=f"FT matches (Top-5 leagues, 14d) with xG data: {with_xg}/{total_ft}.",
        )
    except Exception as e:
        add_check(
            check_id="dq_understat_coverage_ft_14d",
            name="Understat xG coverage (14d)",
            category="coverage",
            status="warning",
            current_value=None,
            threshold="≥60%",
            affected_count=0,
            description=f"Degraded: could not query match_understat_team ({str(e)[:80]}).",
        )

    # 7) Weather coverage: NS matches in next 48h with forecasts
    try:
        res = await session.execute(
            text(
                """
                SELECT
                    COUNT(*) FILTER (WHERE mw.match_id IS NOT NULL) AS with_weather,
                    COUNT(*) AS total_ns
                FROM matches m
                LEFT JOIN match_weather mw ON m.id = mw.match_id
                WHERE m.status = 'NS'
                  AND m.date >= NOW()
                  AND m.date < NOW() + INTERVAL '48 hours'
                """
            )
        )
        row = res.first()
        with_weather = int(row[0] or 0) if row else 0
        total_ns = int(row[1] or 0) if row else 0
        missing_weather = total_ns - with_weather
        coverage_pct = round(with_weather / total_ns * 100, 1) if total_ns > 0 else 0.0
        # Status: warn if coverage < 30%, fail if < 10% (weather is optional SOTA feature)
        if coverage_pct < 10:
            status = "failing"
        elif coverage_pct < 30:
            status = "warning"
        else:
            status = "passing"
        add_check(
            check_id="dq_weather_coverage_ns_48h",
            name="Weather coverage (NS 48h)",
            category="coverage",
            status=status,
            current_value=f"{coverage_pct}%",
            threshold="≥30%",
            affected_count=missing_weather,
            description=f"NS matches (48h) with weather forecast: {with_weather}/{total_ns}.",
        )
    except Exception as e:
        add_check(
            check_id="dq_weather_coverage_ns_48h",
            name="Weather coverage (NS 48h)",
            category="coverage",
            status="warning",
            current_value=None,
            threshold="≥30%",
            affected_count=0,
            description=f"Degraded: could not query match_weather ({str(e)[:80]}).",
        )

    # 8) Venue geo coverage: venues from last 30d matches with coordinates
    # Note: venue_geo uses venue_city as key
    try:
        res = await session.execute(
            text(
                """
                SELECT
                    COUNT(DISTINCT vg.venue_city) AS with_geo,
                    COUNT(DISTINCT m.venue_city) AS total_venues
                FROM matches m
                LEFT JOIN venue_geo vg ON m.venue_city = vg.venue_city
                WHERE m.venue_city IS NOT NULL
                  AND m.date >= NOW() - INTERVAL '30 days'
                """
            )
        )
        row = res.first()
        with_geo = int(row[0] or 0) if row else 0
        total_venues = int(row[1] or 0) if row else 0
        missing_geo = total_venues - with_geo
        coverage_pct = round(with_geo / total_venues * 100, 1) if total_venues > 0 else 0.0
        # Status: warn if coverage < 30%, fail if < 10%
        if coverage_pct < 10:
            status = "failing"
        elif coverage_pct < 30:
            status = "warning"
        else:
            status = "passing"
        add_check(
            check_id="dq_venue_geo_coverage",
            name="Venue geo coverage (30d)",
            category="coverage",
            status=status,
            current_value=f"{coverage_pct}%",
            threshold="≥30%",
            affected_count=missing_geo,
            description=f"Venues from recent matches with coordinates: {with_geo}/{total_venues}.",
        )
    except Exception as e:
        add_check(
            check_id="dq_venue_geo_coverage",
            name="Venue geo coverage (30d)",
            category="coverage",
            status="warning",
            current_value=None,
            threshold="≥30%",
            affected_count=0,
            description=f"Degraded: could not query venue_geo ({str(e)[:80]}).",
        )

    # 9) Team home city profile coverage
    try:
        res = await session.execute(
            text(
                """
                SELECT
                    COUNT(*) FILTER (WHERE thcp.team_id IS NOT NULL) AS with_profile,
                    COUNT(*) AS total_teams
                FROM teams t
                LEFT JOIN team_home_city_profile thcp ON t.id = thcp.team_id
                """
            )
        )
        row = res.first()
        with_profile = int(row[0] or 0) if row else 0
        total_teams = int(row[1] or 0) if row else 0
        missing_profile = total_teams - with_profile
        coverage_pct = round(with_profile / total_teams * 100, 1) if total_teams > 0 else 0.0
        # Status: warn if coverage < 20%, fail if < 5%
        if coverage_pct < 5:
            status = "failing"
        elif coverage_pct < 20:
            status = "warning"
        else:
            status = "passing"
        add_check(
            check_id="dq_team_profile_coverage",
            name="Team profile coverage",
            category="coverage",
            status=status,
            current_value=f"{coverage_pct}%",
            threshold="≥20%",
            affected_count=missing_profile,
            description=f"Teams with home city profile: {with_profile}/{total_teams}.",
        )
    except Exception as e:
        add_check(
            check_id="dq_team_profile_coverage",
            name="Team profile coverage",
            category="coverage",
            status="warning",
            current_value=None,
            threshold="≥20%",
            affected_count=0,
            description=f"Degraded: could not query team_home_city_profile ({str(e)[:80]}).",
        )

    # 10) Sofascore XI coverage: NS matches in next 48h (supported leagues) with XI data
    try:
        # Check if table exists first
        table_check = await session.execute(
            text("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables
                    WHERE table_name = 'match_sofascore_lineup'
                )
            """)
        )
        tables_exist = table_check.scalar()

        if tables_exist:
            sofascore_league_ids = ",".join(str(lid) for lid in SOFASCORE_SUPPORTED_LEAGUES)
            res = await session.execute(
                text(
                    f"""
                    SELECT
                        COUNT(*) FILTER (WHERE msl.match_id IS NOT NULL) AS with_xi,
                        COUNT(*) AS total_ns
                    FROM matches m
                    LEFT JOIN match_sofascore_lineup msl ON m.id = msl.match_id
                    WHERE m.status = 'NS'
                      AND m.date >= NOW()
                      AND m.date < NOW() + INTERVAL '48 hours'
                      AND m.league_id IN ({sofascore_league_ids})
                    """
                )
            )
            row = res.first()
            with_xi = int(row[0] or 0) if row else 0
            total_ns = int(row[1] or 0) if row else 0
            missing_xi = total_ns - with_xi
            coverage_pct = round(with_xi / total_ns * 100, 1) if total_ns > 0 else 0.0
            # Status: warn if coverage < 20%, fail if < 5% (XI is optional SOTA feature)
            if coverage_pct < 5:
                status = "failing"
            elif coverage_pct < 20:
                status = "warning"
            else:
                status = "passing"
            add_check(
                check_id="dq_sofascore_xi_coverage_ns_48h",
                name="Sofascore XI coverage (NS 48h)",
                category="coverage",
                status=status,
                current_value=f"{coverage_pct}%",
                threshold="≥20%",
                affected_count=missing_xi,
                description=f"NS matches (48h, supported leagues) with XI data: {with_xi}/{total_ns}.",
            )
        else:
            # Tables not deployed yet
            add_check(
                check_id="dq_sofascore_xi_coverage_ns_48h",
                name="Sofascore XI coverage (NS 48h)",
                category="coverage",
                status="warning",
                current_value="pending",
                threshold="≥20%",
                affected_count=0,
                description="Tables not deployed yet (migration 030 pending).",
            )
    except Exception as e:
        add_check(
            check_id="dq_sofascore_xi_coverage_ns_48h",
            name="Sofascore XI coverage (NS 48h)",
            category="coverage",
            status="warning",
            current_value=None,
            threshold="≥20%",
            affected_count=0,
            description=f"Degraded: could not query match_sofascore_lineup ({str(e)[:80]}).",
        )

    return checks


@app.get("/dashboard/data_quality.json")
async def dashboard_data_quality_json(
    request: Request,
    status: list[str] | None = Query(default=None),  # passing|warning|failing (multi)
    category: list[str] | None = Query(default=None),  # coverage|consistency|completeness|freshness|odds (multi)
    q: str | None = None,
    page: int = 1,
    limit: int = 50,
    session: AsyncSession = Depends(get_async_session),
):
    """
    Data Quality checks (tabular) for dashboard /data-quality page.
    Auth: X-Dashboard-Token header required.
    """
    if not _verify_dashboard_token(request):
        raise HTTPException(status_code=401, detail="Dashboard access requires valid token.")

    page = max(page, 1)
    limit = min(max(limit, 1), 100)

    status_filters = [s.lower() for s in _normalize_multi_param(status)]
    category_filters = [c.lower() for c in _normalize_multi_param(category)]
    q_norm = (q or "").strip().lower()

    cache_key = f"dq:{','.join(status_filters) or '-'}:{','.join(category_filters) or '-'}:{q_norm or '-'}:{page}:{limit}"
    now = time.time()

    if cache_key in _data_quality_cache:
        cached_entry = _data_quality_cache[cache_key]
        age = now - cached_entry["timestamp"]
        if age < _DATA_QUALITY_CACHE_TTL:
            return {
                "generated_at": cached_entry["generated_at"],
                "cached": True,
                "cache_age_seconds": round(age, 1),
                "data": cached_entry["data"],
            }

    checks = await _build_data_quality_checks(session)

    # Filter
    filtered = []
    for c in checks:
        c_status = str(c.get("status") or "").lower()
        c_cat = str(c.get("category") or "").lower()
        if status_filters and c_status not in status_filters:
            continue
        if category_filters and c_cat not in category_filters:
            continue
        if q_norm:
            hay = " ".join(
                [
                    str(c.get("id") or ""),
                    str(c.get("name") or ""),
                    str(c.get("description") or ""),
                    str(c.get("category") or ""),
                    str(c.get("status") or ""),
                ]
            ).lower()
            if q_norm not in hay:
                continue
        filtered.append(c)

    total = len(filtered)
    pages = (total + limit - 1) // limit if limit > 0 else 1
    start = (page - 1) * limit
    end = start + limit
    page_checks = filtered[start:end]

    generated_at = datetime.utcnow().isoformat() + "Z"
    response_data = {
        "checks": page_checks,
        "total": total,
        "page": page,
        "limit": limit,
        "pages": pages,
    }

    _data_quality_cache[cache_key] = {
        "generated_at": generated_at,
        "timestamp": now,
        "data": response_data,
    }
    # Simple cache pruning
    if len(_data_quality_cache) > 50:
        oldest_key = min(_data_quality_cache.keys(), key=lambda k: _data_quality_cache[k]["timestamp"])
        del _data_quality_cache[oldest_key]

    return {
        "generated_at": generated_at,
        "cached": False,
        "cache_age_seconds": 0,
        "data": response_data,
    }


@app.get("/dashboard/data_quality/{check_id}.json")
async def dashboard_data_quality_check_json(
    check_id: str,
    request: Request,
    limit: int = 50,
    session: AsyncSession = Depends(get_async_session),
):
    """
    Data Quality check detail endpoint.
    Best-effort and no PII. Returns affected_items (limited) and empty history for now.
    Auth: X-Dashboard-Token header required.
    """
    if not _verify_dashboard_token(request):
        raise HTTPException(status_code=401, detail="Dashboard access requires valid token.")

    limit = min(max(limit, 1), 100)
    cache_key = f"dq_detail:{check_id}:{limit}"
    now = time.time()

    if cache_key in _data_quality_detail_cache:
        cached_entry = _data_quality_detail_cache[cache_key]
        age = now - cached_entry["timestamp"]
        if age < _DATA_QUALITY_DETAIL_CACHE_TTL:
            return {
                "generated_at": cached_entry["generated_at"],
                "cached": True,
                "cache_age_seconds": round(age, 1),
                "data": cached_entry["data"],
            }

    # Build current check list and locate the check
    checks = await _build_data_quality_checks(session)
    check = next((c for c in checks if str(c.get("id")) == check_id), None)
    if not check:
        raise HTTPException(status_code=404, detail="Unknown check_id")

    affected_items: list[dict] = []

    try:
        if check_id == "dq_quarantined_odds_24h":
            res = await session.execute(
                text(
                    """
                    SELECT id, match_id, recorded_at, source
                    FROM odds_history
                    WHERE quarantined = true
                      AND recorded_at > NOW() - INTERVAL '24 hours'
                    ORDER BY recorded_at DESC
                    LIMIT :limit
                    """
                ),
                {"limit": limit},
            )
            affected_items = [
                {
                    "odds_history_id": int(r[0]),
                    "match_id": int(r[1]) if r[1] is not None else None,
                    "recorded_at": r[2].isoformat() + "Z" if r[2] else None,
                    "source": r[3],
                }
                for r in res.fetchall()
            ]

        elif check_id == "dq_tainted_matches_7d":
            res = await session.execute(
                text(
                    """
                    SELECT id, date, league_id, status
                    FROM matches
                    WHERE tainted = true
                      AND date > NOW() - INTERVAL '7 days'
                    ORDER BY date DESC
                    LIMIT :limit
                    """
                ),
                {"limit": limit},
            )
            affected_items = [
                {
                    "match_id": int(r[0]),
                    "date": r[1].isoformat() + "Z" if r[1] else None,
                    "league_id": int(r[2]) if r[2] is not None else None,
                    "status": r[3],
                }
                for r in res.fetchall()
            ]

        elif check_id == "dq_unmapped_teams":
            res = await session.execute(
                text(
                    """
                    SELECT id, name, country
                    FROM teams
                    WHERE logo_url IS NULL
                    ORDER BY id ASC
                    LIMIT :limit
                    """
                ),
                {"limit": limit},
            )
            affected_items = [
                {
                    "team_id": int(r[0]),
                    "name": r[1],
                    "country": r[2],
                }
                for r in res.fetchall()
            ]

        elif check_id in ("dq_odds_desync_6h", "dq_odds_desync_90m"):
            window = "6 hours" if check_id == "dq_odds_desync_6h" else "90 minutes"
            res = await session.execute(
                text(
                    f"""
                    SELECT DISTINCT m.id, m.date, m.league_id
                    FROM matches m
                    JOIN odds_snapshots os ON os.match_id = m.id
                    WHERE m.status = 'NS'
                      AND m.date BETWEEN NOW() AND NOW() + INTERVAL '{window}'
                      AND os.odds_freshness = 'live'
                      AND os.snapshot_type = 'lineup_confirmed'
                      AND os.snapshot_at >= NOW() - INTERVAL '120 minutes'
                      AND (m.odds_home IS NULL OR m.odds_draw IS NULL OR m.odds_away IS NULL)
                    ORDER BY m.date ASC
                    LIMIT :limit
                    """
                ),
                {"limit": limit},
            )
            affected_items = [
                {
                    "match_id": int(r[0]),
                    "kickoff_utc": r[1].isoformat() + "Z" if r[1] else None,
                    "league_id": int(r[2]) if r[2] is not None else None,
                }
                for r in res.fetchall()
            ]
    except Exception as e:
        # Best-effort: don't fail endpoint if affected_items query fails
        affected_items = [{"error": f"degraded: {str(e)[:120]}"}]

    generated_at = datetime.utcnow().isoformat() + "Z"
    response_data = {
        "check": check,
        "affected_items": affected_items,
        "history": [],  # Placeholder for future: time-series snapshots per check
    }

    _data_quality_detail_cache[cache_key] = {
        "generated_at": generated_at,
        "timestamp": now,
        "data": response_data,
    }
    if len(_data_quality_detail_cache) > 50:
        oldest_key = min(_data_quality_detail_cache.keys(), key=lambda k: _data_quality_detail_cache[k]["timestamp"])
        del _data_quality_detail_cache[oldest_key]

    return {
        "generated_at": generated_at,
        "cached": False,
        "cache_age_seconds": 0,
        "data": response_data,
    }


# =============================================================================
# DASHBOARD SETTINGS (read-only operational settings, no secrets)
# =============================================================================

# Cache for settings endpoints
_settings_summary_cache: dict = {"data": None, "timestamp": 0, "ttl": 300}
_settings_flags_cache: dict = {"data": None, "timestamp": 0, "ttl": 60}
_settings_models_cache: dict = {"data": None, "timestamp": 0, "ttl": 300}

# SECURITY: Secrets that MUST NEVER appear in settings responses
_SETTINGS_SECRET_KEYS = frozenset({
    "DATABASE_URL", "RAPIDAPI_KEY", "API_KEY", "DASHBOARD_TOKEN",
    "RUNPOD_API_KEY", "GEMINI_API_KEY", "METRICS_BEARER_TOKEN",
    "SMTP_PASSWORD", "OPS_ADMIN_PASSWORD", "OPS_SESSION_SECRET",
    "SENTRY_AUTH_TOKEN", "FUTBOLSTATS_API_KEY", "X_API_KEY",
})


def _is_env_configured(key: str) -> bool:
    """Check if an environment variable is configured (non-empty)."""
    import os
    val = os.environ.get(key, "")
    return bool(val and val.strip())


def _get_known_feature_flags() -> list[dict]:
    """
    Return list of known feature flags with metadata.

    NOTE: Values are bool or None, never secret strings.
    """
    import os

    flags = [
        # LLM/Narratives
        {"key": "FASTPATH_ENABLED", "scope": "llm", "description": "Enable FastPath LLM narrative generation"},
        {"key": "FASTPATH_DRY_RUN", "scope": "llm", "description": "FastPath dry-run mode (no writes)"},
        {"key": "GEMINI_ENABLED", "scope": "llm", "description": "Use Gemini as LLM provider"},
        {"key": "RUNPOD_ENABLED", "scope": "llm", "description": "Use RunPod as LLM provider"},
        # SOTA
        {"key": "SOTA_SOFASCORE_ENABLED", "scope": "sota", "description": "Enable Sofascore XI capture"},
        {"key": "SOTA_SOFASCORE_REFS_ENABLED", "scope": "sota", "description": "Enable Sofascore refs discovery"},
        {"key": "SOTA_WEATHER_ENABLED", "scope": "sota", "description": "Enable weather capture"},
        {"key": "SOTA_VENUE_GEO_ENABLED", "scope": "sota", "description": "Enable venue geocoding"},
        {"key": "SOTA_UNDERSTAT_ENABLED", "scope": "sota", "description": "Enable Understat xG capture"},
        # Sensor/Shadow
        {"key": "SENSOR_B_ENABLED", "scope": "sensor", "description": "Enable Sensor B calibration"},
        {"key": "SHADOW_MODE_ENABLED", "scope": "sensor", "description": "Enable shadow mode predictions"},
        {"key": "SHADOW_TWO_STAGE_ENABLED", "scope": "sensor", "description": "Enable two-stage shadow architecture"},
        # Jobs
        {"key": "SCHEDULER_ENABLED", "scope": "jobs", "description": "Enable background scheduler"},
        {"key": "ODDS_SYNC_ENABLED", "scope": "jobs", "description": "Enable odds sync job"},
        {"key": "STATS_BACKFILL_ENABLED", "scope": "jobs", "description": "Enable stats backfill job"},
        # Predictions
        {"key": "PREDICTIONS_ENABLED", "scope": "predictions", "description": "Enable prediction generation"},
        {"key": "PREDICTIONS_TWO_STAGE", "scope": "predictions", "description": "Use two-stage prediction model"},
        # Other
        {"key": "DEBUG", "scope": "other", "description": "Debug mode enabled"},
        {"key": "SENTRY_ENABLED", "scope": "other", "description": "Enable Sentry error tracking"},
    ]

    result = []
    for flag in flags:
        key = flag["key"]
        raw_val = os.environ.get(key, "").lower().strip()

        # Determine enabled state
        if raw_val in ("true", "1", "yes", "on"):
            enabled = True
        elif raw_val in ("false", "0", "no", "off"):
            enabled = False
        elif raw_val == "":
            enabled = None  # Not set
        else:
            enabled = None  # Unknown value

        result.append({
            "key": key,
            "enabled": enabled,
            "scope": flag["scope"],
            "description": flag["description"],
            "source": "env" if raw_val else "default",
        })

    return result


@app.get("/dashboard/settings/summary.json")
async def dashboard_settings_summary(request: Request):
    """
    Read-only summary of operational settings.

    Auth: X-Dashboard-Token required.
    TTL: 300s cache.

    SECURITY: No secrets or PII in response. Only configured: true/false.
    """
    if not _verify_dashboard_token(request):
        raise HTTPException(status_code=401, detail="Dashboard access requires valid token.")

    now = time.time()
    cache = _settings_summary_cache

    # Check cache
    if cache["data"] and (now - cache["timestamp"]) < cache["ttl"]:
        return {
            "generated_at": cache["data"]["generated_at"],
            "cached": True,
            "cache_age_seconds": round(now - cache["timestamp"], 1),
            "data": cache["data"]["payload"],
        }

    # Build fresh data
    generated_at = datetime.utcnow().isoformat() + "Z"

    try:
        integrations = {
            "rapidapi": {
                "configured": _is_env_configured("RAPIDAPI_KEY") or _is_env_configured("API_FOOTBALL_KEY"),
                "source": "env",
            },
            "sentry": {
                "configured": _is_env_configured("SENTRY_DSN"),
                "source": "env",
            },
            "metrics": {
                "configured": _is_env_configured("METRICS_BEARER_TOKEN"),
                "source": "env",
            },
            "gemini": {
                "configured": _is_env_configured("GEMINI_API_KEY"),
                "source": "env",
            },
            "runpod": {
                "configured": _is_env_configured("RUNPOD_API_KEY"),
                "source": "env",
            },
            "database": {
                "configured": _is_env_configured("DATABASE_URL"),
                "source": "env",
            },
        }

        payload = {
            "readonly": True,
            "sections": ["general", "feature_flags", "model_versions", "integrations"],
            "notes": "Read-only operational settings. No secrets returned.",
            "links": [
                {"title": "Ops Dashboard", "url": "/dashboard/ops"},
                {"title": "Data Quality", "url": "/dashboard/data_quality.json"},
                {"title": "Feature Flags", "url": "/dashboard/settings/feature_flags.json"},
                {"title": "Model Versions", "url": "/dashboard/settings/model_versions.json"},
            ],
            "integrations": integrations,
        }

        # Update cache
        cache["data"] = {"generated_at": generated_at, "payload": payload}
        cache["timestamp"] = now

        return {
            "generated_at": generated_at,
            "cached": False,
            "cache_age_seconds": 0,
            "data": payload,
        }

    except Exception as e:
        logger.error(f"[SETTINGS] summary.json error: {e}")
        return {
            "generated_at": generated_at,
            "cached": False,
            "cache_age_seconds": 0,
            "data": {
                "readonly": True,
                "sections": [],
                "notes": "Read-only operational settings. No secrets returned.",
                "status": "degraded",
                "error": str(e)[:100],
            },
        }


@app.get("/dashboard/settings/feature_flags.json")
async def dashboard_settings_feature_flags(
    request: Request,
    q: str | None = None,
    enabled: bool | None = None,
    scope: str | None = None,
    page: int = 1,
    limit: int = 50,
):
    """
    Read-only list of feature flags.

    Auth: X-Dashboard-Token required.
    TTL: 60s cache (flags can change with deploy).

    Query params:
    - q: Search by key or description
    - enabled: Filter by true/false
    - scope: Filter by scope (llm, sota, jobs, sensor, predictions, other)
    - page: Page number (default 1)
    - limit: Items per page (default 50, max 100)

    SECURITY: No secrets. Only boolean enabled state.
    """
    if not _verify_dashboard_token(request):
        raise HTTPException(status_code=401, detail="Dashboard access requires valid token.")

    # Clamp limit
    limit = min(max(1, limit), 100)
    page = max(1, page)

    now = time.time()
    cache = _settings_flags_cache
    generated_at = datetime.utcnow().isoformat() + "Z"

    try:
        # Get all flags (always fresh since they depend on env)
        all_flags = _get_known_feature_flags()

        # Apply filters
        filtered = all_flags

        if q:
            q_lower = q.lower()
            filtered = [
                f for f in filtered
                if q_lower in f["key"].lower() or q_lower in f["description"].lower()
            ]

        if enabled is not None:
            filtered = [f for f in filtered if f["enabled"] == enabled]

        if scope:
            filtered = [f for f in filtered if f["scope"] == scope]

        # Pagination
        total = len(filtered)
        pages = (total + limit - 1) // limit if limit > 0 else 1
        start = (page - 1) * limit
        end = start + limit
        paginated = filtered[start:end]

        return {
            "generated_at": generated_at,
            "cached": False,  # Always fresh for flags
            "cache_age_seconds": 0,
            "data": {
                "flags": paginated,
                "total": total,
                "page": page,
                "limit": limit,
                "pages": pages,
            },
        }

    except Exception as e:
        logger.error(f"[SETTINGS] feature_flags.json error: {e}")
        return {
            "generated_at": generated_at,
            "cached": False,
            "cache_age_seconds": 0,
            "data": {
                "flags": [],
                "total": 0,
                "page": page,
                "limit": limit,
                "pages": 0,
                "status": "degraded",
                "error": str(e)[:100],
            },
        }


@app.get("/dashboard/settings/model_versions.json")
async def dashboard_settings_model_versions(request: Request):
    """
    Read-only list of ML model versions.

    Auth: X-Dashboard-Token required.
    TTL: 300s cache.

    SECURITY: No secrets. Only version strings.
    """
    if not _verify_dashboard_token(request):
        raise HTTPException(status_code=401, detail="Dashboard access requires valid token.")

    now = time.time()
    cache = _settings_models_cache
    generated_at = datetime.utcnow().isoformat() + "Z"

    # Check cache
    if cache["data"] and (now - cache["timestamp"]) < cache["ttl"]:
        return {
            "generated_at": cache["data"]["generated_at"],
            "cached": True,
            "cache_age_seconds": round(now - cache["timestamp"], 1),
            "data": cache["data"]["payload"],
        }

    try:
        import os

        # Get model versions from settings/env
        baseline_version = getattr(settings, "MODEL_VERSION", None) or os.environ.get("MODEL_VERSION", "v1.0.0")
        architecture = os.environ.get("MODEL_ARCHITECTURE", "baseline")
        shadow_version = os.environ.get("SHADOW_MODEL_VERSION", "disabled")
        shadow_architecture = os.environ.get("SHADOW_ARCHITECTURE", "disabled")

        # Try to get updated_at from model files if available
        model_updated_at = None
        try:
            import glob
            model_files = glob.glob("models/xgb_*.json")
            if model_files:
                import os.path
                latest_file = max(model_files, key=os.path.getmtime)
                mtime = os.path.getmtime(latest_file)
                model_updated_at = datetime.utcfromtimestamp(mtime).isoformat() + "Z"
        except Exception:
            pass

        models = [
            {
                "name": "baseline",
                "version": baseline_version,
                "source": "settings",
                "updated_at": model_updated_at,
            },
            {
                "name": "architecture",
                "version": architecture,
                "source": "env",
                "updated_at": None,
            },
            {
                "name": "shadow_version",
                "version": shadow_version,
                "source": "env",
                "updated_at": None,
            },
            {
                "name": "shadow_architecture",
                "version": shadow_architecture,
                "source": "env",
                "updated_at": None,
            },
        ]

        payload = {"models": models}

        # Update cache
        cache["data"] = {"generated_at": generated_at, "payload": payload}
        cache["timestamp"] = now

        return {
            "generated_at": generated_at,
            "cached": False,
            "cache_age_seconds": 0,
            "data": payload,
        }

    except Exception as e:
        logger.error(f"[SETTINGS] model_versions.json error: {e}")
        return {
            "generated_at": generated_at,
            "cached": False,
            "cache_age_seconds": 0,
            "data": {
                "models": [],
                "status": "degraded",
                "error": str(e)[:100],
            },
        }


# =============================================================================
# DASHBOARD PREDICTIONS (read-only, for ops dashboard)
# =============================================================================

_dashboard_predictions_cache: dict = {"data": None, "timestamp": 0, "ttl": 45}


@app.get("/dashboard/predictions.json")
async def dashboard_predictions_json(
    request: Request,
    league_ids: str | None = Query(default=None, description="Comma-separated league IDs"),
    status: str | None = Query(default=None, description="Match status filter (NS,LIVE,FT,etc)"),
    model: str | None = Query(default=None, description="Model filter (baseline,shadow)"),
    q: str | None = Query(default=None, description="Search by team name"),
    days_back: int = Query(default=0, ge=0, le=30, description="Days back from today"),
    days_ahead: int = Query(default=3, ge=0, le=14, description="Days ahead from today"),
    page: int = Query(default=1, ge=1, description="Page number"),
    limit: int = Query(default=50, ge=1, le=100, description="Items per page"),
    session: AsyncSession = Depends(get_async_session),
):
    """
    Read-only predictions list for Dashboard.

    Auth: X-Dashboard-Token required.
    TTL: 45s cache.

    Query params:
    - league_ids: Comma-separated league IDs (e.g., "39,140,135")
    - status: Match status filter (NS, LIVE, FT, etc.)
    - model: Model filter (baseline, shadow)
    - q: Search by team name
    - days_back: Days back from today (default 0)
    - days_ahead: Days ahead from today (default 3)
    - page: Page number (default 1)
    - limit: Items per page (default 50, max 100)

    SECURITY: No secrets/PII. Public match data only.
    """
    if not _verify_dashboard_token(request):
        raise HTTPException(status_code=401, detail="Dashboard access requires valid token.")

    now = time.time()
    generated_at = datetime.utcnow().isoformat() + "Z"

    try:
        from sqlalchemy import text

        # Build date range
        today = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        from_date = today - timedelta(days=days_back)
        to_date = today + timedelta(days=days_ahead + 1)  # +1 to include full day

        # Build query with filters
        filters = ["m.date >= :from_date", "m.date < :to_date"]
        params: dict = {"from_date": from_date, "to_date": to_date}

        # League filter
        if league_ids:
            try:
                league_list = [int(lid.strip()) for lid in league_ids.split(",") if lid.strip()]
                if league_list:
                    filters.append(f"m.league_id IN ({','.join(str(l) for l in league_list)})")
            except ValueError:
                pass  # Invalid league_ids, ignore filter

        # Status filter
        if status:
            status_list = [s.strip().upper() for s in status.split(",") if s.strip()]
            if status_list:
                status_placeholders = ",".join(f"'{s}'" for s in status_list)
                filters.append(f"m.status IN ({status_placeholders})")

        # Model filter (baseline vs shadow via model_version)
        model_filter_sql = ""
        if model:
            model_lower = model.lower()
            if model_lower == "baseline":
                model_filter_sql = "AND p.model_version NOT LIKE '%shadow%' AND p.model_version NOT LIKE '%two_stage%'"
            elif model_lower == "shadow":
                model_filter_sql = "AND (p.model_version LIKE '%shadow%' OR p.model_version LIKE '%two_stage%')"

        # Search filter
        if q:
            filters.append("(t_home.name ILIKE :q OR t_away.name ILIKE :q)")
            params["q"] = f"%{q}%"

        where_clause = " AND ".join(filters)

        # League names fallback (no competitions table)
        league_names = {
            1: "World Cup", 2: "Champions League", 3: "Europa League",
            39: "Premier League", 40: "Championship", 61: "Ligue 1",
            78: "Bundesliga", 135: "Serie A", 140: "La Liga",
            94: "Primeira Liga", 88: "Eredivisie", 203: "Süper Lig",
            239: "Liga BetPlay", 253: "MLS", 262: "Liga MX",
            128: "Argentina Primera", 71: "Brasileirão",
            848: "Conference League", 45: "FA Cup", 143: "Copa del Rey",
        }

        # Count total
        count_query = f"""
            SELECT COUNT(DISTINCT p.id)
            FROM predictions p
            JOIN matches m ON p.match_id = m.id
            JOIN teams t_home ON m.home_team_id = t_home.id
            JOIN teams t_away ON m.away_team_id = t_away.id
            WHERE {where_clause} {model_filter_sql}
        """
        count_result = await session.execute(text(count_query), params)
        total = int(count_result.scalar() or 0)

        # Calculate pagination
        pages = (total + limit - 1) // limit if limit > 0 else 1
        offset = (page - 1) * limit

        # Fetch predictions with pagination
        data_query = f"""
            SELECT
                p.id,
                p.match_id,
                m.league_id,
                m.date AS kickoff_utc,
                t_home.name AS home_team,
                t_away.name AS away_team,
                m.status,
                m.home_goals,
                m.away_goals,
                p.model_version,
                p.home_prob,
                p.draw_prob,
                p.away_prob,
                p.is_frozen,
                p.frozen_at,
                p.frozen_confidence_tier,
                p.created_at
            FROM predictions p
            JOIN matches m ON p.match_id = m.id
            JOIN teams t_home ON m.home_team_id = t_home.id
            JOIN teams t_away ON m.away_team_id = t_away.id
            WHERE {where_clause} {model_filter_sql}
            ORDER BY m.date ASC, p.created_at DESC
            LIMIT :limit OFFSET :offset
        """
        params["limit"] = limit
        params["offset"] = offset

        result = await session.execute(text(data_query), params)
        rows = result.fetchall()

        # Format predictions
        predictions = []
        for row in rows:
            # Determine pick based on highest probability
            probs = {
                "home": round(row.home_prob, 3) if row.home_prob else None,
                "draw": round(row.draw_prob, 3) if row.draw_prob else None,
                "away": round(row.away_prob, 3) if row.away_prob else None,
            }

            pick = None
            if probs["home"] and probs["draw"] and probs["away"]:
                max_prob = max(probs["home"], probs["draw"], probs["away"])
                if probs["home"] == max_prob:
                    pick = "home"
                elif probs["draw"] == max_prob:
                    pick = "draw"
                else:
                    pick = "away"

            # Determine model type
            model_version = row.model_version or ""
            if "shadow" in model_version.lower() or "two_stage" in model_version.lower():
                model_type = "shadow"
            else:
                model_type = "baseline"

            predictions.append({
                "id": row.id,
                "match_id": row.match_id,
                "league_id": row.league_id,
                "league_name": league_names.get(row.league_id, f"League {row.league_id}"),
                "kickoff_utc": row.kickoff_utc.isoformat() + "Z" if row.kickoff_utc else None,
                "home_team": row.home_team,
                "away_team": row.away_team,
                "status": row.status,
                "score": f"{row.home_goals}-{row.away_goals}" if row.home_goals is not None else None,
                "model": model_type,
                "model_version": model_version,
                "pick": pick,
                "probs": probs,
                "is_frozen": row.is_frozen,
                "frozen_at": row.frozen_at.isoformat() + "Z" if row.frozen_at else None,
                "confidence_tier": row.frozen_confidence_tier,
                "created_at": row.created_at.isoformat() + "Z" if row.created_at else None,
            })

        return {
            "generated_at": generated_at,
            "cached": False,
            "cache_age_seconds": 0,
            "data": {
                "predictions": predictions,
                "total": total,
                "page": page,
                "limit": limit,
                "pages": pages,
                "filters_applied": {
                    "league_ids": league_ids,
                    "status": status,
                    "model": model,
                    "q": q,
                    "days_back": days_back,
                    "days_ahead": days_ahead,
                },
            },
        }

    except Exception as e:
        logger.error(f"[DASHBOARD] predictions.json error: {e}", exc_info=True)
        return {
            "generated_at": generated_at,
            "cached": False,
            "cache_age_seconds": 0,
            "data": {
                "predictions": [],
                "total": 0,
                "page": page,
                "limit": limit,
                "pages": 0,
                "status": "degraded",
                "error": str(e)[:100],
            },
        }


# =============================================================================
# DASHBOARD ANALYTICS REPORTS (catalog for analytics table)
# =============================================================================

_analytics_reports_cache: dict = {"data": None, "timestamp": 0, "ttl": 120}


async def _build_analytics_reports(session: AsyncSession) -> list[dict]:
    """
    Build the list of analytics reports from various sources.

    Sources:
    - prediction_performance_reports (model_performance)
    - pit_reports (prediction_accuracy)
    - ops_daily_rollups (system_metrics)
    - job_runs / api usage metrics (api_usage)

    Returns list of report dicts with stable IDs.
    """
    from sqlalchemy import text

    reports = []

    # 1) Model Performance Reports (from prediction_performance_reports)
    try:
        result = await session.execute(text("""
            SELECT id, generated_at, window_days, report_date, payload, source
            FROM prediction_performance_reports
            ORDER BY generated_at DESC
            LIMIT 10
        """))
        rows = result.fetchall()

        for row in rows:
            payload = row.payload or {}
            overall = payload.get("overall", {})
            accuracy = overall.get("accuracy")
            total_matches = overall.get("total_matches", 0)

            # Determine status
            status = "ok"
            if accuracy is not None and accuracy < 0.35:
                status = "warning"
            elif row.generated_at and (datetime.utcnow() - row.generated_at).days > 2:
                status = "stale"

            reports.append({
                "id": f"model_perf_{row.window_days}d_{row.id}",
                "type": "model_performance",
                "title": f"Model Performance ({row.window_days}d)",
                "subtitle": f"Report date: {row.report_date}" if row.report_date else None,
                "status": status,
                "updated_at": row.generated_at.isoformat() + "Z" if row.generated_at else None,
                "tags": ["xgb", f"{row.window_days}d", row.source or "auto"],
                "summary": {
                    "primary_label": "Accuracy",
                    "primary_value": f"{accuracy:.1%}" if accuracy else "N/A",
                    "secondary_label": "Matches",
                    "secondary_value": total_matches,
                },
                "links": [
                    {"title": "Performance API", "url": f"/dashboard/ops/predictions_performance.json?window_days={row.window_days}"},
                ],
            })
    except Exception as e:
        logger.warning(f"[ANALYTICS] Failed to load model_performance reports: {e}")
        reports.append({
            "id": "model_perf_error",
            "type": "model_performance",
            "title": "Model Performance",
            "subtitle": "Error loading reports",
            "status": "warning",
            "updated_at": None,
            "tags": ["error"],
            "summary": {"primary_label": "Status", "primary_value": "unavailable"},
            "links": [],
        })

    # 2) Prediction Accuracy Reports (from pit_reports)
    try:
        result = await session.execute(text("""
            SELECT id, report_type, report_date, payload, source, created_at, updated_at
            FROM pit_reports
            WHERE report_type IN ('daily', 'weekly')
            ORDER BY created_at DESC
            LIMIT 10
        """))
        rows = result.fetchall()

        for row in rows:
            payload = row.payload or {}
            summary_data = payload.get("summary", {})
            accuracy = summary_data.get("accuracy")
            total = summary_data.get("total_predictions", 0)

            # Determine status
            status = "ok"
            if row.created_at and (datetime.utcnow() - row.created_at).days > 3:
                status = "stale"
            elif accuracy is not None and accuracy < 0.30:
                status = "warning"

            reports.append({
                "id": f"pit_{row.report_type}_{row.id}",
                "type": "prediction_accuracy",
                "title": f"PIT Report ({row.report_type.capitalize()})",
                "subtitle": f"Date: {row.report_date.strftime('%Y-%m-%d')}" if row.report_date else None,
                "status": status,
                "updated_at": (row.updated_at or row.created_at).isoformat() + "Z" if (row.updated_at or row.created_at) else None,
                "tags": ["pit", row.report_type, row.source or "auto"],
                "summary": {
                    "primary_label": "Accuracy",
                    "primary_value": f"{accuracy:.1%}" if accuracy else "N/A",
                    "secondary_label": "Predictions",
                    "secondary_value": total,
                },
                "links": [
                    {"title": "PIT Protocol", "url": "docs/PIT_EVALUATION_PROTOCOL.md"},
                ],
            })
    except Exception as e:
        logger.warning(f"[ANALYTICS] Failed to load pit_reports: {e}")

    # 3) System Metrics (from ops_daily_rollups)
    try:
        result = await session.execute(text("""
            SELECT day, payload, created_at, updated_at
            FROM ops_daily_rollups
            ORDER BY day DESC
            LIMIT 7
        """))
        rows = result.fetchall()

        for row in rows:
            payload = row.payload or {}
            api_calls = payload.get("api_calls_total", 0)
            predictions_generated = payload.get("predictions_generated", 0)
            job_status = payload.get("jobs_status", "unknown")

            # Determine status
            status = "ok"
            if job_status == "error":
                status = "warning"
            elif row.day and (datetime.utcnow().date() - row.day).days > 2:
                status = "stale"

            reports.append({
                "id": f"ops_rollup_{row.day.isoformat()}",
                "type": "system_metrics",
                "title": f"Daily Ops Rollup",
                "subtitle": f"Date: {row.day.isoformat()}",
                "status": status,
                "updated_at": (row.updated_at or row.created_at).isoformat() + "Z" if (row.updated_at or row.created_at) else None,
                "tags": ["ops", "daily", "system"],
                "summary": {
                    "primary_label": "API Calls",
                    "primary_value": api_calls,
                    "secondary_label": "Predictions",
                    "secondary_value": predictions_generated,
                },
                "links": [
                    {"title": "History API", "url": "/dashboard/ops/history.json?days=30"},
                ],
            })
    except Exception as e:
        logger.warning(f"[ANALYTICS] Failed to load ops_daily_rollups: {e}")

    # 4) API Usage / Job Health (synthetic from job_runs)
    try:
        result = await session.execute(text("""
            SELECT
                job_name,
                COUNT(*) as runs_24h,
                COUNT(*) FILTER (WHERE status = 'ok') as success_count,
                MAX(started_at) as last_run
            FROM job_runs
            WHERE started_at > NOW() - INTERVAL '24 hours'
            GROUP BY job_name
            ORDER BY runs_24h DESC
            LIMIT 5
        """))
        rows = result.fetchall()

        if rows:
            total_runs = sum(r.runs_24h for r in rows)
            total_success = sum(r.success_count for r in rows)
            success_rate = total_success / total_runs if total_runs > 0 else 0

            status = "ok" if success_rate >= 0.95 else ("warning" if success_rate >= 0.80 else "stale")

            reports.append({
                "id": "api_usage_24h",
                "type": "api_usage",
                "title": "Job Executions (24h)",
                "subtitle": f"Top jobs: {', '.join(r.job_name[:20] for r in rows[:3])}",
                "status": status,
                "updated_at": datetime.utcnow().isoformat() + "Z",
                "tags": ["jobs", "api", "24h"],
                "summary": {
                    "primary_label": "Success Rate",
                    "primary_value": f"{success_rate:.1%}",
                    "secondary_label": "Total Runs",
                    "secondary_value": total_runs,
                },
                "links": [
                    {"title": "Job Runs API", "url": "/dashboard/ops/job_runs.json"},
                ],
            })
    except Exception as e:
        logger.warning(f"[ANALYTICS] Failed to load job_runs metrics: {e}")

    # 5) SOTA Enrichment Summary (synthetic)
    try:
        result = await session.execute(text("""
            SELECT
                (SELECT COUNT(*) FROM match_sofascore_lineup) as sofascore_lineups,
                (SELECT COUNT(*) FROM match_weather) as weather_forecasts,
                (SELECT COUNT(*) FROM venue_geo) as venue_geos,
                (SELECT COUNT(*) FROM match_understat_team) as understat_xg
        """))
        row = result.fetchone()

        if row:
            total_enrichments = (row.sofascore_lineups or 0) + (row.weather_forecasts or 0) + (row.venue_geos or 0) + (row.understat_xg or 0)

            reports.append({
                "id": "sota_enrichment_summary",
                "type": "system_metrics",
                "title": "SOTA Enrichment Summary",
                "subtitle": f"XI: {row.sofascore_lineups}, Weather: {row.weather_forecasts}, Geo: {row.venue_geos}",
                "status": "ok" if total_enrichments > 100 else "warning",
                "updated_at": datetime.utcnow().isoformat() + "Z",
                "tags": ["sota", "enrichment", "coverage"],
                "summary": {
                    "primary_label": "Total Enrichments",
                    "primary_value": total_enrichments,
                    "secondary_label": "Understat xG",
                    "secondary_value": row.understat_xg or 0,
                },
                "links": [
                    {"title": "Ops Dashboard", "url": "/dashboard/ops.json"},
                ],
            })
    except Exception as e:
        logger.warning(f"[ANALYTICS] Failed to load SOTA enrichment summary: {e}")

    return reports


@app.get("/dashboard/analytics/reports.json")
async def dashboard_analytics_reports(
    request: Request,
    type: str | None = Query(default=None, description="Report type filter"),
    q: str | None = Query(default=None, description="Search by title/subtitle"),
    status: str | None = Query(default=None, description="Status filter (ok|warning|stale)"),
    page: int = Query(default=1, ge=1, description="Page number"),
    limit: int = Query(default=50, ge=1, le=100, description="Items per page"),
    session: AsyncSession = Depends(get_async_session),
):
    """
    Analytics reports catalog for Dashboard table.

    Auth: X-Dashboard-Token required.
    TTL: 120s cache.

    Query params:
    - type: Filter by report type (model_performance, prediction_accuracy, system_metrics, api_usage)
    - q: Search by title or subtitle
    - status: Filter by status (ok, warning, stale)
    - page: Page number (default 1)
    - limit: Items per page (default 50, max 100)

    SECURITY: No secrets/PII. Aggregated metrics only.
    """
    if not _verify_dashboard_token(request):
        raise HTTPException(status_code=401, detail="Dashboard access requires valid token.")

    now = time.time()
    cache = _analytics_reports_cache
    generated_at = datetime.utcnow().isoformat() + "Z"

    try:
        # Check cache (only for unfiltered requests)
        cache_key = f"{type}:{q}:{status}:{page}:{limit}"
        use_cache = (type is None and q is None and status is None and page == 1 and limit == 50)

        if use_cache and cache["data"] and (now - cache["timestamp"]) < cache["ttl"]:
            return {
                "generated_at": cache["data"]["generated_at"],
                "cached": True,
                "cache_age_seconds": round(now - cache["timestamp"], 1),
                "data": cache["data"]["payload"],
            }

        # Build reports list
        all_reports = await _build_analytics_reports(session)

        # Apply filters
        filtered = all_reports

        if type:
            filtered = [r for r in filtered if r["type"] == type]

        if status:
            filtered = [r for r in filtered if r["status"] == status]

        if q:
            q_lower = q.lower()
            filtered = [
                r for r in filtered
                if q_lower in (r["title"] or "").lower()
                or q_lower in (r["subtitle"] or "").lower()
                or any(q_lower in tag.lower() for tag in r.get("tags", []))
            ]

        # Pagination
        total = len(filtered)
        pages = (total + limit - 1) // limit if limit > 0 else 1
        start = (page - 1) * limit
        end = start + limit
        paginated = filtered[start:end]

        payload = {
            "reports": paginated,
            "total": total,
            "page": page,
            "limit": limit,
            "pages": pages,
            "note": "best-effort, aggregated server-side",
        }

        # Update cache for default requests
        if use_cache:
            cache["data"] = {"generated_at": generated_at, "payload": payload}
            cache["timestamp"] = now

        return {
            "generated_at": generated_at,
            "cached": False,
            "cache_age_seconds": 0,
            "data": payload,
        }

    except Exception as e:
        logger.error(f"[ANALYTICS] reports.json error: {e}", exc_info=True)
        return {
            "generated_at": generated_at,
            "cached": False,
            "cache_age_seconds": 0,
            "data": {
                "reports": [],
                "total": 0,
                "page": page,
                "limit": limit,
                "pages": 0,
                "status": "degraded",
                "error": str(e)[:100],
                "note": "best-effort, aggregated server-side",
            },
        }


# =============================================================================
# AUDIT LOGS ENDPOINT
# =============================================================================
_audit_logs_cache: dict = {"data": None, "timestamp": 0}
_AUDIT_LOGS_TTL = 90  # 90 seconds cache


def _parse_range_to_hours(range_str: str | None) -> int:
    """Convert range string to hours. Default 24h."""
    if not range_str:
        return 24
    range_map = {"1h": 1, "24h": 24, "7d": 168, "30d": 720}
    return range_map.get(range_str, 24)


def _derive_severity(result: str | None, error_message: str | None) -> str:
    """Derive severity from result/error_message."""
    if error_message or result == "error":
        return "error"
    if result == "warning" or result == "partial":
        return "warning"
    return "info"


def _derive_actor_kind(actor: str | None) -> str:
    """Derive actor_kind from actor field."""
    if not actor:
        return "system"
    actor_lower = actor.lower()
    if "scheduler" in actor_lower or "job" in actor_lower or "system" in actor_lower:
        return "system"
    if "token" in actor_lower or "key" in actor_lower or "user" in actor_lower:
        return "user"
    return "system"


async def _build_audit_events(
    session: AsyncSession,
    hours: int,
    types: list[str] | None,
    severities: list[str] | None,
    actor_kinds: list[str] | None,
    search: str | None,
) -> list[dict]:
    """Build audit events from multiple sources."""
    events = []
    cutoff = datetime.utcnow() - timedelta(hours=hours)

    # Source 1: ops_audit_log
    try:
        query = text("""
            SELECT id, action, request_id, actor, actor_id, result, error_message,
                   created_at, duration_ms
            FROM ops_audit_log
            WHERE created_at >= :cutoff
            ORDER BY created_at DESC
            LIMIT 500
        """)
        result = await session.execute(query, {"cutoff": cutoff})
        rows = result.mappings().all()

        for row in rows:
            severity = _derive_severity(row.get("result"), row.get("error_message"))
            actor_kind = _derive_actor_kind(row.get("actor"))

            # Build display message
            action = row.get("action", "unknown")
            result_str = row.get("result", "")
            duration = row.get("duration_ms")
            msg_parts = [f"{action}"]
            if result_str:
                msg_parts.append(f"→ {result_str}")
            if duration:
                msg_parts.append(f"({duration}ms)")
            message = " ".join(msg_parts)

            # Actor display (redact full IDs)
            actor_id = row.get("actor_id", "")
            actor_display = row.get("actor", "system")
            if actor_id and len(actor_id) > 4:
                actor_display = f"{actor_display}:{actor_id[:4]}…"

            events.append({
                "id": f"audit_{row['id']}",
                "type": action,
                "severity": severity,
                "actor_kind": actor_kind,
                "actor_display": actor_display,
                "message": message,
                "created_at": row["created_at"].isoformat() + "Z" if row.get("created_at") else None,
                "correlation_id": row.get("request_id"),
                "runbook_url": None,
            })
    except Exception as e:
        logger.warning(f"[AUDIT] Failed to load ops_audit_log: {e}")

    # Source 2: job_runs (scheduler jobs)
    try:
        query = text("""
            SELECT id, job_name, status, started_at, finished_at, duration_ms, error_message
            FROM job_runs
            WHERE created_at >= :cutoff
            ORDER BY created_at DESC
            LIMIT 500
        """)
        result = await session.execute(query, {"cutoff": cutoff})
        rows = result.mappings().all()

        for row in rows:
            status = row.get("status", "")
            severity = "error" if status == "error" else ("warning" if status == "partial" else "info")

            job_name = row.get("job_name", "unknown_job")
            duration = row.get("duration_ms")
            msg_parts = [f"job:{job_name}", f"→ {status}"]
            if duration:
                msg_parts.append(f"({duration}ms)")
            if row.get("error_message"):
                # Truncate error message
                err = str(row["error_message"])[:50]
                msg_parts.append(f"- {err}")
            message = " ".join(msg_parts)

            events.append({
                "id": f"job_{row['id']}",
                "type": f"job_{job_name}",
                "severity": severity,
                "actor_kind": "system",
                "actor_display": "scheduler",
                "message": message,
                "created_at": row["started_at"].isoformat() + "Z" if row.get("started_at") else None,
                "correlation_id": None,
                "runbook_url": "/docs/OPS_RUNBOOK.md" if severity == "error" else None,
            })
    except Exception as e:
        logger.warning(f"[AUDIT] Failed to load job_runs: {e}")

    # Sort all events by created_at DESC
    events.sort(key=lambda x: x.get("created_at") or "", reverse=True)

    # Apply filters
    if types:
        type_set = set(types)
        events = [e for e in events if e["type"] in type_set]

    if severities:
        sev_set = set(severities)
        events = [e for e in events if e["severity"] in sev_set]

    if actor_kinds:
        ak_set = set(actor_kinds)
        events = [e for e in events if e["actor_kind"] in ak_set]

    if search:
        search_lower = search.lower()
        events = [
            e for e in events
            if search_lower in (e.get("message") or "").lower()
            or search_lower in (e.get("type") or "").lower()
            or search_lower in (e.get("actor_display") or "").lower()
        ]

    return events


@app.get("/dashboard/audit_logs.json")
async def dashboard_audit_logs(
    request: Request,
    type: str | None = Query(default=None, description="Type filter (comma-separated for multi)"),
    severity: str | None = Query(default=None, description="Severity filter: info,warning,error (comma-separated)"),
    actor_kind: str | None = Query(default=None, description="Actor kind filter: system,user (comma-separated)"),
    q: str | None = Query(default=None, description="Search in message/type/actor"),
    range: str | None = Query(default="24h", description="Time range: 1h, 24h, 7d, 30d"),
    page: int = Query(default=1, ge=1, description="Page number"),
    limit: int = Query(default=50, ge=1, le=100, description="Items per page"),
    session: AsyncSession = Depends(get_async_session),
):
    """
    Audit logs for Dashboard monitoring.

    Auth: X-Dashboard-Token required.
    TTL: 90s cache.

    Query params:
    - type: Filter by event type (comma-separated for multi)
    - severity: Filter by severity: info, warning, error (comma-separated)
    - actor_kind: Filter by actor kind: system, user (comma-separated)
    - q: Search in message, type, or actor_display
    - range: Time range - 1h, 24h (default), 7d, 30d
    - page: Page number (default 1)
    - limit: Items per page (default 50, max 100)

    SECURITY: No PII/secrets/payloads. Redacted actor IDs.
    """
    if not _verify_dashboard_token(request):
        raise HTTPException(status_code=401, detail="Dashboard access requires valid token.")

    generated_at = datetime.utcnow().isoformat() + "Z"

    # Parse multi-value filters
    types = [t.strip() for t in type.split(",")] if type else None
    severities = [s.strip() for s in severity.split(",")] if severity else None
    actor_kinds = [a.strip() for a in actor_kind.split(",")] if actor_kind else None
    hours = _parse_range_to_hours(range)

    # Check if we can use cache (only for default/simple requests)
    use_cache = not types and not severities and not actor_kinds and not q and range == "24h"
    now = time.time()
    cache = _audit_logs_cache

    try:
        if use_cache and cache["data"] and (now - cache["timestamp"]) < _AUDIT_LOGS_TTL:
            cached_data = cache["data"]
            # Apply pagination to cached data
            all_events = cached_data["all_events"]
            total = len(all_events)
            pages = (total + limit - 1) // limit if limit > 0 else 1
            start = (page - 1) * limit
            end = start + limit
            paginated = all_events[start:end]

            return {
                "generated_at": cached_data["generated_at"],
                "cached": True,
                "cache_age_seconds": int(now - cache["timestamp"]),
                "data": {
                    "events": paginated,
                    "total": total,
                    "page": page,
                    "limit": limit,
                    "pages": pages,
                },
            }

        # Build events list
        all_events = await _build_audit_events(session, hours, types, severities, actor_kinds, q)

        # Pagination
        total = len(all_events)
        pages = (total + limit - 1) // limit if limit > 0 else 1
        start = (page - 1) * limit
        end = start + limit
        paginated = all_events[start:end]

        payload = {
            "events": paginated,
            "total": total,
            "page": page,
            "limit": limit,
            "pages": pages,
        }

        # Update cache for default requests
        if use_cache:
            cache["data"] = {"generated_at": generated_at, "all_events": all_events}
            cache["timestamp"] = now

        return {
            "generated_at": generated_at,
            "cached": False,
            "cache_age_seconds": 0,
            "data": payload,
        }

    except Exception as e:
        logger.error(f"[AUDIT] audit_logs.json error: {e}", exc_info=True)
        return {
            "generated_at": generated_at,
            "cached": False,
            "cache_age_seconds": 0,
            "data": {
                "events": [],
                "total": 0,
                "page": page,
                "limit": limit,
                "pages": 0,
                "status": "degraded",
                "error": str(e)[:100],
            },
        }


# =============================================================================
# TEAM LOGOS ENDPOINT
# =============================================================================
_team_logos_cache: dict = {"data": None, "timestamp": 0}
_TEAM_LOGOS_TTL = 3600  # 1 hour cache (logos rarely change)


@app.get("/dashboard/team_logos.json")
async def dashboard_team_logos(
    request: Request,
    session: AsyncSession = Depends(get_async_session),
):
    """
    Team logos map for Dashboard.

    Auth: X-Dashboard-Token required.
    TTL: 3600s (1 hour) cache - logos rarely change.

    Returns a map of team_name -> logo_url for efficient frontend lookup.
    """
    if not _verify_dashboard_token(request):
        raise HTTPException(status_code=401, detail="Dashboard access requires valid token.")

    generated_at = datetime.utcnow().isoformat() + "Z"
    now = time.time()
    cache = _team_logos_cache

    # Check cache
    if cache["data"] and (now - cache["timestamp"]) < _TEAM_LOGOS_TTL:
        return {
            "generated_at": cache["data"]["generated_at"],
            "cached": True,
            "cache_age_seconds": int(now - cache["timestamp"]),
            "teams": cache["data"]["teams"],
            "count": cache["data"]["count"],
        }

    try:
        # Fetch all teams with logos
        result = await session.execute(
            text("SELECT name, logo_url FROM teams WHERE logo_url IS NOT NULL ORDER BY name")
        )
        rows = result.fetchall()

        # Build name -> logo_url map
        teams_map = {}
        for row in rows:
            name, logo_url = row
            if name and logo_url:
                teams_map[name] = logo_url

        # Update cache
        cache["data"] = {
            "generated_at": generated_at,
            "teams": teams_map,
            "count": len(teams_map),
        }
        cache["timestamp"] = now

        return {
            "generated_at": generated_at,
            "cached": False,
            "cache_age_seconds": 0,
            "teams": teams_map,
            "count": len(teams_map),
        }

    except Exception as e:
        logger.error(f"[TEAM_LOGOS] Error fetching team logos: {e}", exc_info=True)
        return {
            "generated_at": generated_at,
            "cached": False,
            "cache_age_seconds": 0,
            "teams": {},
            "count": 0,
            "status": "degraded",
            "error": str(e)[:100],
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
          <a class="nav-link" data-path="/dashboard/ops/daily_comparison" href="/dashboard/ops/daily_comparison">Daily</a>
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


@app.post("/dashboard/ops/sensor_retrain")
async def trigger_sensor_retrain(request: Request):
    """
    Manually trigger Sensor B retrain job.

    Protected by dashboard token. Use after deploy to force immediate retrain
    instead of waiting for the 6h interval.
    """
    if not _verify_dashboard_token(request):
        raise HTTPException(status_code=401, detail="Dashboard access requires valid token.")

    from app.scheduler import retrain_sensor_model

    start_time = time.time()
    result = await retrain_sensor_model()
    duration_ms = int((time.time() - start_time) * 1000)

    return {
        "status": "executed",
        "duration_ms": duration_ms,
        "result": result,
    }


@app.post("/dashboard/ops/sensor_eval")
async def trigger_sensor_eval(request: Request):
    """
    Manually trigger Sensor B evaluation job.

    Protected by dashboard token. Evaluates pending FT matches against
    sensor predictions.
    """
    if not _verify_dashboard_token(request):
        raise HTTPException(status_code=401, detail="Dashboard access requires valid token.")

    from app.scheduler import evaluate_sensor_predictions_job

    start_time = time.time()
    result = await evaluate_sensor_predictions_job()
    duration_ms = int((time.time() - start_time) * 1000)

    return {
        "status": "executed",
        "duration_ms": duration_ms,
        "result": result,
    }


@app.post("/dashboard/ops/shadow_eval")
async def trigger_shadow_eval(request: Request):
    """
    Manually trigger Shadow mode evaluation job.

    Protected by dashboard token. Evaluates pending FT matches against
    shadow predictions.
    """
    if not _verify_dashboard_token(request):
        raise HTTPException(status_code=401, detail="Dashboard access requires valid token.")

    from app.scheduler import evaluate_shadow_predictions

    start_time = time.time()
    result = await evaluate_shadow_predictions()
    duration_ms = int((time.time() - start_time) * 1000)

    return {
        "status": "executed",
        "duration_ms": duration_ms,
        "result": result,
    }


@app.post("/dashboard/ops/stats_backfill")
async def trigger_stats_backfill(request: Request):
    """
    Manually trigger stats backfill job.

    Protected by dashboard token. Use after deploy to force immediate execution
    instead of waiting for the 60min interval.
    """
    if not _verify_dashboard_token(request):
        raise HTTPException(status_code=401, detail="Dashboard access requires valid token.")

    from app.scheduler import capture_finished_match_stats

    start_time = time.time()
    result = await capture_finished_match_stats()
    duration_ms = int((time.time() - start_time) * 1000)

    return {
        "status": "executed",
        "duration_ms": duration_ms,
        "result": result,
    }


@app.post("/dashboard/ops/historical_stats_backfill")
async def trigger_historical_stats_backfill(request: Request):
    """
    Trigger historical stats backfill job for matches since 2023-08-01.

    This endpoint calls the scheduler job which:
    - Processes 500 matches per run (configurable via HISTORICAL_STATS_BACKFILL_BATCH_SIZE)
    - Marks matches without stats as {"_no_stats": true} to skip on future runs
    - Auto-advances through all leagues

    Protected by dashboard token.
    """
    if not _verify_dashboard_token(request):
        raise HTTPException(status_code=401, detail="Dashboard access requires valid token.")

    from app.scheduler import historical_stats_backfill

    start_time = time.time()
    result = await historical_stats_backfill()
    duration_ms = int((time.time() - start_time) * 1000)

    return {
        "status": "executed",
        "duration_ms": duration_ms,
        "result": result,
    }


@app.post("/dashboard/ops/match_link")
async def link_match_to_api_football(
    request: Request,
    match_id: int,
    external_id: int,
    fetch_stats: bool = True,
):
    """
    Link an orphan match to its API-Football fixture_id.

    Orphan matches (external_id=NULL) cannot receive odds or stats from API-Football.
    This endpoint allows manually linking them when the fixture_id is known.

    Args:
        match_id: Our internal match ID
        external_id: API-Football fixture ID
        fetch_stats: If True, also fetch and update stats from API-Football
    """
    if not _verify_dashboard_token(request):
        raise HTTPException(status_code=401, detail="Dashboard access requires valid token.")

    from sqlalchemy import text
    import json

    start_time = time.time()
    result = {"match_id": match_id, "external_id": external_id}

    try:
        # Update external_id first
        async with AsyncSessionLocal() as session:
            # Check if external_id already exists on another match
            check_result = await session.execute(text("""
                SELECT id FROM matches WHERE external_id = :external_id AND id != :match_id
            """), {"match_id": match_id, "external_id": external_id})
            existing = check_result.scalar()
            if existing:
                return {"status": "error", "error": f"external_id {external_id} already exists on match {existing}"}

            update_result = await session.execute(text("""
                UPDATE matches
                SET external_id = :external_id
                WHERE id = :match_id
            """), {"match_id": match_id, "external_id": external_id})
            await session.commit()
            result["external_id_updated"] = True
            result["rows_affected"] = update_result.rowcount

        # Optionally fetch and update stats in separate transaction
        if fetch_stats:
            from app.etl.api_football import APIFootballProvider
            provider = APIFootballProvider()
            try:
                stats_data = await provider.get_fixture_statistics(external_id)
                if stats_data:
                    async with AsyncSessionLocal() as session2:
                        await session2.execute(text("""
                            UPDATE matches
                            SET stats = CAST(:stats_json AS JSON)
                            WHERE id = :match_id
                        """), {"match_id": match_id, "stats_json": json.dumps(stats_data)})
                        await session2.commit()
                    result["stats_updated"] = True
                    result["stats_keys"] = list(stats_data.get("home", {}).keys())
                else:
                    result["stats_updated"] = False
                    result["stats_error"] = "No stats returned from API"
            finally:
                await provider.close()

        duration_ms = int((time.time() - start_time) * 1000)
        return {"status": "ok", "duration_ms": duration_ms, "result": result}

    except Exception as e:
        duration_ms = int((time.time() - start_time) * 1000)
        return {"status": "error", "duration_ms": duration_ms, "error": str(e)}


@app.post("/dashboard/ops/stats_refresh")
async def trigger_stats_refresh(request: Request, lookback_hours: int = 48, max_calls: int = 100):
    """
    Manually trigger stats refresh for recently finished matches.

    Unlike stats_backfill, this re-fetches stats for ALL recent FT matches,
    even if they already have stats. Captures late events like red cards.

    Args:
        lookback_hours: Hours to look back (default 48 for manual runs)
        max_calls: Max API calls (default 100)
    """
    if not _verify_dashboard_token(request):
        raise HTTPException(status_code=401, detail="Dashboard access requires valid token.")

    from app.scheduler import refresh_recent_ft_stats

    start_time = time.time()
    result = await refresh_recent_ft_stats(lookback_hours=lookback_hours, max_calls=max_calls)
    duration_ms = int((time.time() - start_time) * 1000)

    return {
        "status": "executed",
        "duration_ms": duration_ms,
        "result": result,
    }


@app.post("/dashboard/ops/narratives_regenerate")
async def trigger_narratives_regenerate(
    request: Request,
    lookback_hours: int = 48,
    max_matches: int = 100,
    force: bool = False,
):
    """
    Regenerate LLM narratives for recently finished matches.

    This endpoint resets narratives for matches that had stats refreshed,
    allowing FastPath to regenerate them with updated data.

    Args:
        lookback_hours: Hours to look back for finished matches (default 48)
        max_matches: Maximum matches to process (default 100)
        force: If True, regenerate even if narrative already exists
    """
    if not _verify_dashboard_token(request):
        raise HTTPException(status_code=401, detail="Dashboard access requires valid token.")

    from sqlalchemy import text

    start_time = time.time()
    metrics = {
        "checked": 0,
        "reset": 0,
        "already_pending": 0,
        "no_audit": 0,
        "errors": 0,
    }

    try:
        async with AsyncSessionLocal() as session:
            # Find FT matches in lookback window that have prediction_outcomes and post_match_audits
            result = await session.execute(text("""
                SELECT
                    m.id as match_id,
                    po.id as outcome_id,
                    pma.id as audit_id,
                    pma.llm_narrative_status,
                    ht.name as home_team,
                    at.name as away_team
                FROM matches m
                JOIN prediction_outcomes po ON po.match_id = m.id
                JOIN post_match_audits pma ON pma.outcome_id = po.id
                JOIN teams ht ON ht.id = m.home_team_id
                JOIN teams at ON at.id = m.away_team_id
                WHERE m.status IN ('FT', 'AET', 'PEN')
                  AND m.finished_at >= NOW() - INTERVAL ':lookback hours'
                  AND m.stats IS NOT NULL
                  AND m.stats::text != '{}'
                ORDER BY m.finished_at DESC
                LIMIT :max_matches
            """.replace(":lookback", str(lookback_hours))), {
                "max_matches": max_matches,
            })

            rows = result.fetchall()
            metrics["checked"] = len(rows)

            reset_ids = []
            match_ids = []
            for row in rows:
                if row.llm_narrative_status == "pending":
                    metrics["already_pending"] += 1
                    continue

                if row.llm_narrative_status == "ok" and not force:
                    # Skip if already has narrative and force=False
                    continue

                reset_ids.append(row.audit_id)
                match_ids.append(row.match_id)

            if reset_ids:
                # Reset narrative status to allow regeneration
                await session.execute(text("""
                    UPDATE post_match_audits
                    SET llm_narrative_status = 'pending',
                        llm_narrative_attempts = 0,
                        llm_narrative_json = NULL,
                        llm_output_raw = NULL,
                        llm_prompt_version = NULL,
                        llm_validation_errors = NULL
                    WHERE id = ANY(:ids)
                """), {"ids": reset_ids})

                # Trick: update finished_at to NOW() so FastPath picks them up
                # FastPath has a 90-min lookback, this makes old matches eligible
                await session.execute(text("""
                    UPDATE matches
                    SET finished_at = NOW()
                    WHERE id = ANY(:ids)
                """), {"ids": match_ids})

                await session.commit()
                metrics["reset"] = len(reset_ids)

        duration_ms = int((time.time() - start_time) * 1000)
        return {
            "status": "executed",
            "duration_ms": duration_ms,
            "result": {
                **metrics,
                "message": f"Reset {metrics['reset']} narratives. FastPath will regenerate in next ticks (~2 min each).",
            },
        }

    except Exception as e:
        duration_ms = int((time.time() - start_time) * 1000)
        metrics["errors"] = 1
        return {
            "status": "error",
            "duration_ms": duration_ms,
            "result": {**metrics, "error": str(e)},
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
          <a class="nav-link" data-path="/dashboard/ops/daily_comparison" href="/dashboard/ops/daily_comparison">Daily</a>
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


# =============================================================================
# DAILY COMPARISON: Model A vs Shadow vs Sensor B vs Market
# =============================================================================


@app.get("/dashboard/ops/daily_comparison.json")
async def ops_daily_comparison(
    request: Request,
    date: str = None,
    league_id: int = None,
    session: AsyncSession = Depends(get_async_session),
):
    """
    Get daily comparison of finished matches: Real vs Model A vs Shadow vs Sensor B vs Market.

    Args:
        date: Date in YYYY-MM-DD format (America/Los_Angeles timezone). Defaults to today.
        league_id: Optional filter by league ID.

    Returns:
        List of matches with predictions from all sources for comparison.
    """
    import pytz
    import re

    if not _verify_dashboard_token(request):
        raise HTTPException(status_code=401, detail="Dashboard access requires valid token.")

    # Default to today in LA timezone
    la_tz = pytz.timezone("America/Los_Angeles")
    if date:
        # Validate date format
        if not re.match(r"^\d{4}-\d{2}-\d{2}$", date):
            raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD.")
        target_date = datetime.strptime(date, "%Y-%m-%d")
    else:
        # Today in LA timezone
        target_date = datetime.now(la_tz).replace(hour=0, minute=0, second=0, microsecond=0)
        target_date = target_date.replace(tzinfo=None)  # Make naive for localize

    # Convert date_la to UTC range (CRITICAL for index usage per Auditor)
    start_la = la_tz.localize(target_date.replace(hour=0, minute=0, second=0))
    end_la = la_tz.localize(target_date.replace(hour=23, minute=59, second=59))
    start_utc = start_la.astimezone(pytz.UTC).replace(tzinfo=None)  # Naive for DB
    end_utc = (end_la.astimezone(pytz.UTC) + timedelta(seconds=1)).replace(tzinfo=None)

    # Build query with UTC range filter
    query = """
        SELECT
            match_id,
            kickoff_utc,
            match_day_la,
            league_id,
            status,
            home_team,
            away_team,
            home_goals,
            away_goals,
            actual_outcome,
            a_home_prob,
            a_draw_prob,
            a_away_prob,
            a_pick,
            a_version,
            a_is_frozen,
            shadow_home_prob,
            shadow_draw_prob,
            shadow_away_prob,
            shadow_pick,
            shadow_version,
            sensor_home_prob,
            sensor_draw_prob,
            sensor_away_prob,
            sensor_pick,
            sensor_version,
            sensor_state,
            market_bookmaker,
            market_odds_home,
            market_odds_draw,
            market_odds_away,
            market_implied_home,
            market_implied_draw,
            market_implied_away,
            market_pick
        FROM v_daily_match_comparison
        WHERE kickoff_utc >= :start_utc AND kickoff_utc < :end_utc
    """
    params = {"start_utc": start_utc, "end_utc": end_utc}

    if league_id:
        query += " AND league_id = :league_id"
        params["league_id"] = league_id

    query += " ORDER BY kickoff_utc"

    result = await session.execute(text(query), params)
    matches = result.mappings().all()

    # Calculate summary stats
    total = len(matches)
    a_correct = sum(1 for m in matches if m["a_pick"] == m["actual_outcome"])
    shadow_correct = sum(1 for m in matches if m["shadow_pick"] == m["actual_outcome"])
    sensor_correct = sum(1 for m in matches if m["sensor_pick"] == m["actual_outcome"])
    market_correct = sum(1 for m in matches if m["market_pick"] == m["actual_outcome"])

    return {
        "date_la": target_date.strftime("%Y-%m-%d"),
        "start_utc": start_utc.isoformat(),
        "end_utc": end_utc.isoformat(),
        "total_matches": total,
        "summary": {
            "model_a": {
                "correct": a_correct,
                "accuracy": round(a_correct / total, 3) if total > 0 else 0,
            },
            "shadow": {
                "correct": shadow_correct,
                "accuracy": round(shadow_correct / total, 3) if total > 0 else 0,
            },
            "sensor_b": {
                "correct": sensor_correct,
                "accuracy": round(sensor_correct / total, 3) if total > 0 else 0,
            },
            "market": {
                "correct": market_correct,
                "accuracy": round(market_correct / total, 3) if total > 0 else 0,
            },
        },
        "matches": [dict(m) for m in matches],
    }


@app.get("/dashboard/ops/daily_comparison", response_class=HTMLResponse)
async def ops_daily_comparison_html(
    request: Request,
    date: str = None,
    league_id: int = None,
    market: str = None,
    model_a: str = None,
    shadow: str = None,
    sensor: str = None,
    session: AsyncSession = Depends(get_async_session),
):
    """
    HTML table view of daily comparison - opens directly in browser.

    Model selection via query params (1=enabled, absent=disabled):
    - market: Include Market predictions
    - model_a: Include Model A predictions
    - shadow: Include Shadow predictions
    - sensor: Include Sensor B predictions
    """
    import pytz
    import re

    if not _verify_dashboard_token(request):
        raise HTTPException(status_code=401, detail="Dashboard access requires valid token.")

    # Parse model selection: present with value "1" = enabled, absent = disabled
    # First visit (no params) = all enabled by default
    query_params = dict(request.query_params)
    is_first_visit = not any(k in query_params for k in ["market", "model_a", "shadow", "sensor"])

    if is_first_visit:
        show_market = show_model_a = show_shadow = show_sensor = True
    else:
        show_market = market == "1"
        show_model_a = model_a == "1"
        show_shadow = shadow == "1"
        show_sensor = sensor == "1"

    # At least one model must be selected
    if not any([show_market, show_model_a, show_shadow, show_sensor]):
        show_market = show_model_a = True  # Default to Market + Model A

    selected_models = {
        "market": show_market,
        "model_a": show_model_a,
        "shadow": show_shadow,
        "sensor": show_sensor,
    }

    # Reuse the JSON endpoint logic
    la_tz = pytz.timezone("America/Los_Angeles")
    if date:
        if not re.match(r"^\d{4}-\d{2}-\d{2}$", date):
            raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD.")
        target_date = datetime.strptime(date, "%Y-%m-%d")
    else:
        target_date = datetime.now(la_tz).replace(hour=0, minute=0, second=0, microsecond=0)
        target_date = target_date.replace(tzinfo=None)

    start_la = la_tz.localize(target_date.replace(hour=0, minute=0, second=0))
    end_la = la_tz.localize(target_date.replace(hour=23, minute=59, second=59))
    start_utc = start_la.astimezone(pytz.UTC).replace(tzinfo=None)
    end_utc = (end_la.astimezone(pytz.UTC) + timedelta(seconds=1)).replace(tzinfo=None)

    # Query for FINISHED matches (from VIEW)
    query_finished = """
        SELECT * FROM v_daily_match_comparison
        WHERE kickoff_utc >= :start_utc AND kickoff_utc < :end_utc
    """
    params = {"start_utc": start_utc, "end_utc": end_utc}
    if league_id:
        query_finished += " AND league_id = :league_id"
        params["league_id"] = league_id
    query_finished += " ORDER BY kickoff_utc"

    result_finished = await session.execute(text(query_finished), params)
    finished_matches = result_finished.mappings().all()

    # Query for PENDING matches (NS status) - need to join manually
    query_pending = """
        WITH latest_predictions AS (
            SELECT DISTINCT ON (match_id)
                match_id, home_prob, draw_prob, away_prob, model_version
            FROM predictions
            WHERE model_version NOT LIKE '%two_stage%'
            ORDER BY match_id, is_frozen DESC NULLS LAST, created_at DESC
        )
        SELECT
            m.id AS match_id,
            m.date AS kickoff_utc,
            m.league_id,
            m.status,
            ht.name AS home_team,
            at.name AS away_team,
            p.home_prob AS a_home_prob,
            p.draw_prob AS a_draw_prob,
            p.away_prob AS a_away_prob,
            CASE
                WHEN p.home_prob >= p.draw_prob AND p.home_prob >= p.away_prob THEN 'home'
                WHEN p.draw_prob >= p.home_prob AND p.draw_prob >= p.away_prob THEN 'draw'
                ELSE 'away'
            END AS a_pick,
            sp.shadow_predicted AS shadow_pick,
            sen.b_pick AS sensor_pick,
            mos.market_pick
        FROM matches m
        JOIN teams ht ON ht.id = m.home_team_id
        JOIN teams at ON at.id = m.away_team_id
        LEFT JOIN latest_predictions p ON p.match_id = m.id
        LEFT JOIN shadow_predictions sp ON sp.match_id = m.id
        LEFT JOIN sensor_predictions sen ON sen.match_id = m.id
        LEFT JOIN match_odds_snapshot mos ON mos.match_id = m.id AND mos.is_primary = TRUE
        WHERE m.status = 'NS'
          AND m.date >= :start_utc AND m.date < :end_utc
    """
    if league_id:
        query_pending += " AND m.league_id = :league_id"
    query_pending += " ORDER BY m.date"

    result_pending = await session.execute(text(query_pending), params)
    pending_matches = result_pending.mappings().all()

    # Combine: pending first, then finished
    all_matches = list(pending_matches) + list(finished_matches)

    # Calculate summary (only finished matches)
    total_finished = len(finished_matches)
    total_pending = len(pending_matches)
    a_correct = sum(1 for m in finished_matches if m["a_pick"] == m["actual_outcome"])
    shadow_correct = sum(1 for m in finished_matches if m["shadow_pick"] == m["actual_outcome"])
    sensor_correct = sum(1 for m in finished_matches if m["sensor_pick"] == m["actual_outcome"])
    market_correct = sum(1 for m in finished_matches if m["market_pick"] == m["actual_outcome"])

    # Calculate GLOBAL accuracy (only for SELECTED models)
    # Build WHERE clause dynamically based on selected models
    global_where_clauses = []
    if show_market:
        global_where_clauses.append("market_pick IS NOT NULL")
    if show_model_a:
        global_where_clauses.append("a_pick IS NOT NULL")
    if show_shadow:
        global_where_clauses.append("shadow_pick IS NOT NULL")
    if show_sensor:
        global_where_clauses.append("sensor_pick IS NOT NULL")

    global_where = " AND ".join(global_where_clauses) if global_where_clauses else "1=1"

    global_query = f"""
        SELECT
            COUNT(*) as total,
            SUM(CASE WHEN market_pick = actual_outcome THEN 1 ELSE 0 END) as market_correct,
            SUM(CASE WHEN a_pick = actual_outcome THEN 1 ELSE 0 END) as a_correct,
            SUM(CASE WHEN shadow_pick = actual_outcome THEN 1 ELSE 0 END) as shadow_correct,
            SUM(CASE WHEN sensor_pick = actual_outcome THEN 1 ELSE 0 END) as sensor_correct
        FROM v_daily_match_comparison
        WHERE {global_where}
    """
    global_result = await session.execute(text(global_query))
    global_row = global_result.fetchone()
    global_total = global_row[0] or 0
    global_market = global_row[1] or 0
    global_a = global_row[2] or 0
    global_shadow = global_row[3] or 0
    global_sensor = global_row[4] or 0

    # Calculate daily wins per model (only for SELECTED models)
    # Build dynamic winner conditions based on selected models
    selected_model_names = []
    if show_market:
        selected_model_names.append(("market", "market_correct"))
    if show_model_a:
        selected_model_names.append(("model_a", "a_correct"))
    if show_shadow:
        selected_model_names.append(("shadow", "shadow_correct"))
    if show_sensor:
        selected_model_names.append(("sensor", "sensor_correct"))

    # Build winner CASE expressions dynamically
    def build_winner_case(model_col, all_cols):
        """Build CASE for when this model beats all others (only among selected models)."""
        # Check if this model is in the selected list
        selected_cols = [c for _, c in all_cols]
        if model_col not in selected_cols:
            return "0"  # Model not selected, never wins
        other_cols = [c for _, c in all_cols if c != model_col]
        if not other_cols:
            return "1"  # Only one model selected, always wins
        conditions = " AND ".join([f"{model_col} > {oc}" for oc in other_cols])
        return f"CASE WHEN {conditions} THEN 1 ELSE 0 END"

    wins_market = wins_a = wins_shadow = wins_sensor = 0

    if len(selected_model_names) >= 1:
        daily_wins_query = f"""
            WITH daily_stats AS (
                SELECT
                    match_day_la,
                    SUM(CASE WHEN market_pick = actual_outcome THEN 1 ELSE 0 END) as market_correct,
                    SUM(CASE WHEN a_pick = actual_outcome THEN 1 ELSE 0 END) as a_correct,
                    SUM(CASE WHEN shadow_pick = actual_outcome THEN 1 ELSE 0 END) as shadow_correct,
                    SUM(CASE WHEN sensor_pick = actual_outcome THEN 1 ELSE 0 END) as sensor_correct
                FROM v_daily_match_comparison
                WHERE {global_where}
                GROUP BY match_day_la
            )
            SELECT
                SUM({build_winner_case('market_correct', selected_model_names)}) as market_wins,
                SUM({build_winner_case('a_correct', selected_model_names)}) as a_wins,
                SUM({build_winner_case('shadow_correct', selected_model_names)}) as shadow_wins,
                SUM({build_winner_case('sensor_correct', selected_model_names)}) as sensor_wins
            FROM daily_stats
        """
        daily_wins_result = await session.execute(text(daily_wins_query))
        daily_wins_row = daily_wins_result.fetchone()
        wins_market = daily_wins_row[0] or 0
        wins_a = daily_wins_row[1] or 0
        wins_shadow = daily_wins_row[2] or 0
        wins_sensor = daily_wins_row[3] or 0

    # Build HTML
    date_str = target_date.strftime("%Y-%m-%d")

    def badge_only(pick):
        """Render pick as colored badge with initial (no check mark)."""
        if not pick:
            return '-'
        colors = {'home': '#3b82f6', 'draw': '#6b7280', 'away': '#ef4444'}
        initials = {'home': 'H', 'draw': 'D', 'away': 'A'}
        bg = colors.get(pick, '#6b7280')
        letter = initials.get(pick, '?')
        return f'<span style="background:{bg};color:white;padding:2px 8px;border-radius:4px;font-weight:600;">{letter}</span>'

    def badge(pick, actual):
        """Render pick as colored badge - green if correct, red if wrong."""
        if not pick:
            return '-'
        initials = {'home': 'H', 'draw': 'D', 'away': 'A'}
        letter = initials.get(pick, '?')
        is_correct = pick == actual
        bg = '#22c55e' if is_correct else '#ef4444'  # green if correct, red if wrong
        return f'<span style="background:{bg};color:white;padding:2px 8px;border-radius:4px;font-weight:600;">{letter}</span>'

    # Determine daily winner (model with most correct predictions)
    daily_scores = {
        'market': market_correct,
        'model_a': a_correct,
        'shadow': shadow_correct,
        'sensor': sensor_correct
    }
    max_score = max(daily_scores.values())
    daily_winner = [k for k, v in daily_scores.items() if v == max_score][0] if max_score > 0 else None

    # Helper to add green background if this column is the winner
    def winner_style(col_name):
        if col_name == daily_winner:
            return "background: #dcfce7;"
        return ""

    def format_time_la(kickoff_utc):
        """Convert UTC kickoff to LA time formatted as HH:MM AM/PM."""
        if not kickoff_utc:
            return "—"
        utc_dt = kickoff_utc.replace(tzinfo=pytz.UTC) if kickoff_utc.tzinfo is None else kickoff_utc
        la_dt = utc_dt.astimezone(la_tz)
        return la_dt.strftime("%-I:%M %p")

    rows_html = ""
    for m in all_matches:
        is_finished = m.get('status') in ('FT', 'AET', 'PEN') or m.get('actual_outcome') is not None
        actual = m.get('actual_outcome')

        if is_finished:
            # Finished match: show score, result, and checkmarks
            score_cell = f"{m['home_goals']}-{m['away_goals']}"
            real_cell = badge_only(actual)
            market_cell = badge(m.get('market_pick'), actual)
            model_a_cell = badge(m.get('a_pick'), actual)
            shadow_cell = badge(m.get('shadow_pick'), actual)
            sensor_cell = badge(m.get('sensor_pick'), actual)
        else:
            # Pending match: show time, "—" for real, badges without checkmarks
            score_cell = format_time_la(m.get('kickoff_utc'))
            real_cell = "—"
            market_cell = badge_only(m.get('market_pick'))
            model_a_cell = badge_only(m.get('a_pick'))
            shadow_cell = badge_only(m.get('shadow_pick'))
            sensor_cell = badge_only(m.get('sensor_pick'))

        # Build row with only selected model columns
        model_cells = ""
        if show_market:
            model_cells += f'<td style="text-align: center; {winner_style("market") if is_finished else ""}">{market_cell}</td>'
        if show_model_a:
            model_cells += f'<td style="text-align: center; {winner_style("model_a") if is_finished else ""}">{model_a_cell}</td>'
        if show_shadow:
            model_cells += f'<td style="text-align: center; {winner_style("shadow") if is_finished else ""}">{shadow_cell}</td>'
        if show_sensor:
            model_cells += f'<td style="text-align: center; {winner_style("sensor") if is_finished else ""}">{sensor_cell}</td>'

        rows_html += f"""
        <tr style="{'opacity: 0.7;' if not is_finished else ''}">
            <td>{m['home_team']}</td>
            <td>{m['away_team']}</td>
            <td style="text-align: center; font-weight: bold;">{score_cell}</td>
            <td style="text-align: center;">{real_cell}</td>
            {model_cells}
        </tr>
        """

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Daily Comparison - {date_str}</title>
        <style>
            * {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; }}
            body {{ background: #f8fafc; padding: 20px; }}
            .header-container {{ display: flex; justify-content: space-between; align-items: flex-start; }}
            .header-left {{ flex: 1; }}
            h1 {{ color: #1e293b; margin-bottom: 5px; }}
            .subtitle {{ color: #64748b; margin-bottom: 20px; }}
            .summary-box {{ background: #1e293b; color: white; padding: 12px 20px; border-radius: 8px; display: flex; align-items: center; gap: 24px; }}
            .summary-title {{ font-size: 12px; color: #94a3b8; font-weight: 500; white-space: nowrap; }}
            .summary-item {{ text-align: center; }}
            .summary-item .label {{ font-size: 11px; color: #94a3b8; margin-bottom: 2px; }}
            .summary-item .label .wins {{ font-size: 9px; background: #fbbf24; color: #1e293b; padding: 1px 4px; border-radius: 3px; font-weight: 700; margin-left: 2px; }}
            .summary-item .value {{ font-size: 18px; font-weight: 700; }}
            .summary-item .count {{ font-size: 10px; background: #374151; padding: 2px 6px; border-radius: 4px; margin-top: 4px; display: inline-block; }}
            table {{ width: 100%; border-collapse: collapse; background: white; border-radius: 8px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
            th {{ background: #1e293b; color: white; padding: 12px; text-align: left; font-weight: 600; }}
            td {{ padding: 12px; border-bottom: 1px solid #e2e8f0; }}
            tr:hover {{ background: #f1f5f9; }}
            .nav {{ margin-bottom: 20px; }}
            .nav a {{ color: #3b82f6; text-decoration: none; margin-right: 15px; }}
            .nav a:hover {{ text-decoration: underline; }}
            .legend {{ font-size: 11px; color: #94a3b8; margin: 8px 0 16px 0; font-style: italic; }}
            .summary-item .checkbox {{ margin-top: 6px; }}
            .summary-item input[type="checkbox"] {{ width: 14px; height: 14px; cursor: pointer; }}
            .apply-btn {{ background: #3b82f6; color: white; border: none; padding: 6px 12px; border-radius: 4px; cursor: pointer; font-size: 11px; margin-left: 8px; }}
            .apply-btn:hover {{ background: #2563eb; }}
            .search-box {{ margin: 16px 0; }}
            .search-box input {{ padding: 8px 12px; border: 1px solid #e2e8f0; border-radius: 6px; width: 300px; font-size: 14px; }}
            .search-box input:focus {{ outline: none; border-color: #3b82f6; }}
        </style>
    </head>
    <body>
        <div class="nav">
            <a href="/dashboard/ops">← Ops</a>
            <a href="?date={( target_date - timedelta(days=1)).strftime('%Y-%m-%d')}&market={market}&model_a={model_a}&shadow={shadow}&sensor={sensor}">Día anterior</a>
            <a href="?date={(target_date + timedelta(days=1)).strftime('%Y-%m-%d')}&market={market}&model_a={model_a}&shadow={shadow}&sensor={sensor}">Día siguiente →</a>
        </div>

        <div class="header-container">
            <div class="header-left">
                <h1>Daily Comparison</h1>
                <p class="subtitle">{date_str} (LA timezone) - {total_finished} terminados, {total_pending} pendientes</p>
            </div>
            <form class="summary-box" method="GET">
                <input type="hidden" name="date" value="{date_str}">
                <span class="summary-title">n={global_total}</span>
                <div class="summary-item">
                    <div class="label">Market <span class="wins">{wins_market}</span></div>
                    <div class="value">{round(global_market/global_total*100, 1) if global_total > 0 and show_market else '—'}{'%' if show_market and global_total > 0 else ''}</div>
                    <div class="count">{global_market if show_market else '—'}</div>
                    <div class="checkbox"><input type="checkbox" name="market" value="1" {'checked' if show_market else ''}></div>
                </div>
                <div class="summary-item">
                    <div class="label">Model A <span class="wins">{wins_a}</span></div>
                    <div class="value">{round(global_a/global_total*100, 1) if global_total > 0 and show_model_a else '—'}{'%' if show_model_a and global_total > 0 else ''}</div>
                    <div class="count">{global_a if show_model_a else '—'}</div>
                    <div class="checkbox"><input type="checkbox" name="model_a" value="1" {'checked' if show_model_a else ''}></div>
                </div>
                <div class="summary-item">
                    <div class="label">Shadow <span class="wins">{wins_shadow}</span></div>
                    <div class="value">{round(global_shadow/global_total*100, 1) if global_total > 0 and show_shadow else '—'}{'%' if show_shadow and global_total > 0 else ''}</div>
                    <div class="count">{global_shadow if show_shadow else '—'}</div>
                    <div class="checkbox"><input type="checkbox" name="shadow" value="1" {'checked' if show_shadow else ''}></div>
                </div>
                <div class="summary-item">
                    <div class="label">Sensor B <span class="wins">{wins_sensor}</span></div>
                    <div class="value">{round(global_sensor/global_total*100, 1) if global_total > 0 and show_sensor else '—'}{'%' if show_sensor and global_total > 0 else ''}</div>
                    <div class="count">{global_sensor if show_sensor else '—'}</div>
                    <div class="checkbox"><input type="checkbox" name="sensor" value="1" {'checked' if show_sensor else ''}></div>
                </div>
                <button type="submit" class="apply-btn">Aplicar</button>
            </form>
        </div>
        <p class="legend">* Resultados calculados solo con partidos donde los modelos seleccionados tienen datos</p>

        <div class="search-box">
            <input type="text" id="searchInput" placeholder="Buscar equipo..." onkeyup="filterTable()">
        </div>

        <table id="matchesTable">
            <thead>
                <tr>
                    <th>Home</th>
                    <th>Away</th>
                    <th style="text-align: center;">Score</th>
                    <th style="text-align: center;">Real</th>
                    {'<th style="text-align: center; ' + ('background:#166534;' if daily_winner == 'market' else '') + '">Market<br><span style="font-size:11px;background:#22c55e;color:white;padding:2px 6px;border-radius:4px;margin-top:4px;display:inline-block;">' + str(round(market_correct/total_finished*100, 1) if total_finished > 0 else 0) + '%</span></th>' if show_market else ''}
                    {'<th style="text-align: center; ' + ('background:#166534;' if daily_winner == 'model_a' else '') + '">Model A<br><span style="font-size:11px;background:#22c55e;color:white;padding:2px 6px;border-radius:4px;margin-top:4px;display:inline-block;">' + str(round(a_correct/total_finished*100, 1) if total_finished > 0 else 0) + '%</span></th>' if show_model_a else ''}
                    {'<th style="text-align: center; ' + ('background:#166534;' if daily_winner == 'shadow' else '') + '">Shadow<br><span style="font-size:11px;background:#22c55e;color:white;padding:2px 6px;border-radius:4px;margin-top:4px;display:inline-block;">' + str(round(shadow_correct/total_finished*100, 1) if total_finished > 0 else 0) + '%</span></th>' if show_shadow else ''}
                    {'<th style="text-align: center; ' + ('background:#166534;' if daily_winner == 'sensor' else '') + '">Sensor B<br><span style="font-size:11px;background:#22c55e;color:white;padding:2px 6px;border-radius:4px;margin-top:4px;display:inline-block;">' + str(round(sensor_correct/total_finished*100, 1) if total_finished > 0 else 0) + '%</span></th>' if show_sensor else ''}
                </tr>
            </thead>
            <tbody>
                {rows_html}
            </tbody>
        </table>

        <script>
        function filterTable() {{
            const input = document.getElementById('searchInput');
            const filter = input.value.toLowerCase();
            const table = document.getElementById('matchesTable');
            const rows = table.getElementsByTagName('tr');

            for (let i = 1; i < rows.length; i++) {{
                const cells = rows[i].getElementsByTagName('td');
                if (cells.length > 0) {{
                    const home = cells[0].textContent.toLowerCase();
                    const away = cells[1].textContent.toLowerCase();
                    if (home.includes(filter) || away.includes(filter)) {{
                        rows[i].style.display = '';
                    }} else {{
                        rows[i].style.display = 'none';
                    }}
                }}
            }}
        }}
        </script>
    </body>
    </html>
    """

    return HTMLResponse(content=html)


@app.get("/dashboard/ops/league_stats", response_class=HTMLResponse)
async def ops_league_stats_html(
    request: Request,
    league_id: int = 239,
    season: int = 2026,
    session: AsyncSession = Depends(get_async_session),
):
    """
    HTML dashboard showing league-wide statistics.

    Shows aggregated team stats: top scorers, best defense, etc.
    Also shows data availability for derived_facts.
    """
    if not _verify_dashboard_token(request):
        raise HTTPException(status_code=401, detail="Dashboard access requires valid token.")

    # League name mapping
    league_names = {
        239: "Colombia Primera A",
        128: "Argentina Primera División",
        250: "Paraguay Apertura",
        252: "Paraguay Clausura",
        39: "Premier League",
        140: "La Liga",
        135: "Serie A",
        78: "Bundesliga",
        61: "Ligue 1",
        71: "Brasil Serie A",
        253: "MLS",
        262: "Liga MX",
    }
    league_name = league_names.get(league_id, f"League {league_id}")

    # Team stats query
    team_stats_query = """
        WITH team_stats AS (
            SELECT
                t.id,
                t.name,
                COUNT(*) as matches_played,
                SUM(CASE WHEN m.home_team_id = t.id THEN m.home_goals ELSE m.away_goals END) as goals_for,
                SUM(CASE WHEN m.home_team_id = t.id THEN m.away_goals ELSE m.home_goals END) as goals_against,
                SUM(CASE WHEN m.home_team_id = t.id THEN 1 ELSE 0 END) as home_matches,
                SUM(CASE WHEN m.away_team_id = t.id THEN 1 ELSE 0 END) as away_matches,
                SUM(CASE WHEN m.home_team_id = t.id THEN m.home_goals ELSE 0 END) as home_goals,
                SUM(CASE WHEN m.away_team_id = t.id THEN m.away_goals ELSE 0 END) as away_goals,
                SUM(CASE WHEN
                    (m.home_team_id = t.id AND m.home_goals > m.away_goals) OR
                    (m.away_team_id = t.id AND m.away_goals > m.home_goals)
                    THEN 1 ELSE 0 END) as wins,
                SUM(CASE WHEN m.home_goals = m.away_goals THEN 1 ELSE 0 END) as draws,
                SUM(CASE WHEN
                    (m.home_team_id = t.id AND m.home_goals < m.away_goals) OR
                    (m.away_team_id = t.id AND m.away_goals < m.home_goals)
                    THEN 1 ELSE 0 END) as losses
            FROM matches m
            JOIN teams t ON t.id = m.home_team_id OR t.id = m.away_team_id
            WHERE m.league_id = :league_id
              AND m.status = 'FT'
              AND m.season = :season
            GROUP BY t.id, t.name
        )
        SELECT
            name,
            matches_played,
            wins,
            draws,
            losses,
            goals_for,
            goals_against,
            goals_for - goals_against as goal_diff,
            home_matches,
            away_matches,
            home_goals,
            away_goals,
            ROUND(goals_for::numeric / NULLIF(matches_played, 0), 2) as goals_per_match,
            ROUND(goals_against::numeric / NULLIF(matches_played, 0), 2) as conceded_per_match,
            wins * 3 + draws as points,
            id as team_id
        FROM team_stats
        WHERE matches_played >= 1
        ORDER BY points DESC, goal_diff DESC
    """
    result = await session.execute(text(team_stats_query), {"league_id": league_id, "season": season})
    teams = result.fetchall()

    # Data availability query
    data_availability_query = """
        SELECT
            COUNT(*) as total_matches,
            COUNT(CASE WHEN stats IS NOT NULL AND stats::text != 'null' THEN 1 END) as with_stats,
            COUNT(CASE WHEN home_goals IS NOT NULL THEN 1 END) as with_goals,
            COALESCE(SUM(home_goals), 0) + COALESCE(SUM(away_goals), 0) as total_goals
        FROM matches
        WHERE league_id = :league_id
          AND status = 'FT'
          AND season = :season
    """
    avail_result = await session.execute(text(data_availability_query), {"league_id": league_id, "season": season})
    availability = avail_result.fetchone()

    # Detailed stats query (from matches with stats)
    detailed_stats_query = """
        WITH team_detailed AS (
            SELECT
                t.id,
                t.name,
                COUNT(*) as matches_with_stats,
                SUM(COALESCE((CASE WHEN m.home_team_id = t.id THEN (m.stats->'home'->>'offsides')::int ELSE (m.stats->'away'->>'offsides')::int END), 0)) as total_offsides,
                SUM(COALESCE((CASE WHEN m.home_team_id = t.id THEN (m.stats->'home'->>'yellow_cards')::int ELSE (m.stats->'away'->>'yellow_cards')::int END), 0)) as total_yellows,
                SUM(COALESCE((CASE WHEN m.home_team_id = t.id THEN (m.stats->'home'->>'fouls')::int ELSE (m.stats->'away'->>'fouls')::int END), 0)) as total_fouls,
                SUM(COALESCE((CASE WHEN m.home_team_id = t.id THEN (m.stats->'home'->>'corner_kicks')::int ELSE (m.stats->'away'->>'corner_kicks')::int END), 0)) as total_corners,
                SUM(COALESCE((CASE WHEN m.home_team_id = t.id THEN (m.stats->'home'->>'total_shots')::int ELSE (m.stats->'away'->>'total_shots')::int END), 0)) as total_shots,
                SUM(COALESCE((CASE WHEN m.home_team_id = t.id THEN (m.stats->'home'->>'shots_on_goal')::int ELSE (m.stats->'away'->>'shots_on_goal')::int END), 0)) as shots_on_target
            FROM matches m
            JOIN teams t ON t.id = m.home_team_id OR t.id = m.away_team_id
            WHERE m.league_id = :league_id
              AND m.status = 'FT'
              AND m.season = :season
              AND m.stats IS NOT NULL
              AND m.stats::text != 'null'
            GROUP BY t.id, t.name
        )
        SELECT * FROM team_detailed WHERE matches_with_stats > 0 ORDER BY total_shots DESC
    """
    detailed_result = await session.execute(text(detailed_stats_query), {"league_id": league_id, "season": season})
    detailed_stats = detailed_result.fetchall()

    # Calculate detailed leaders
    most_offsides = max(detailed_stats, key=lambda t: t[3]) if detailed_stats else None
    most_yellows = max(detailed_stats, key=lambda t: t[4]) if detailed_stats else None
    most_fouls = max(detailed_stats, key=lambda t: t[5]) if detailed_stats else None
    most_corners = max(detailed_stats, key=lambda t: t[6]) if detailed_stats else None
    most_shots = max(detailed_stats, key=lambda t: t[7]) if detailed_stats else None
    best_accuracy = max(detailed_stats, key=lambda t: t[8] / t[7] if t[7] > 0 else 0) if detailed_stats else None

    # Calculate league-wide stats
    total_matches = availability[0] if availability else 0
    total_goals = int(availability[3]) if availability and availability[3] else 0
    avg_goals_per_match = round(total_goals / total_matches, 2) if total_matches > 0 else 0

    # Find extremes (return all teams tied for the lead)
    def get_all_max(teams_list, key_func, is_min=False):
        """Get all teams tied for max (or min) value."""
        if not teams_list:
            return []
        if is_min:
            extreme_val = min(key_func(t) for t in teams_list)
        else:
            extreme_val = max(key_func(t) for t in teams_list)
        return [t for t in teams_list if key_func(t) == extreme_val]

    if teams:
        top_scorers = get_all_max(teams, lambda t: t[5])  # goals_for
        worst_defenses = get_all_max(teams, lambda t: t[6])  # goals_against
        best_defenses = get_all_max(teams, lambda t: t[13] if t[13] is not None else 999, is_min=True)  # conceded_per_match
        most_wins_list = get_all_max(teams, lambda t: t[2])  # wins
        most_draws_list = get_all_max(teams, lambda t: t[3])  # draws
        most_losses_list = get_all_max(teams, lambda t: t[4])  # losses
        best_home_list = get_all_max(teams, lambda t: t[10] / t[8] if t[8] > 0 else 0)  # home_goals/home_matches
        best_away_list = get_all_max(teams, lambda t: t[11] / t[9] if t[9] > 0 else 0)  # away_goals/away_matches
    else:
        top_scorers = worst_defenses = best_defenses = most_wins_list = most_draws_list = most_losses_list = best_home_list = best_away_list = []

    # -------------------------------------------------------------------------
    # Modal data: per-team badges + stats + league ranks
    # -------------------------------------------------------------------------
    # Build quick lookup maps
    team_rows_by_id = {}
    for pos, t in enumerate(teams, 1):
        # Tuple layout:
        # 0 name, 1 PJ, 2 W, 3 D, 4 L, 5 GF, 6 GA, 7 GD, 8 home_matches, 9 away_matches,
        # 10 home_goals, 11 away_goals, 12 goals_per_match, 13 conceded_per_match, 14 points, 15 team_id
        team_id = t[15]
        team_rows_by_id[int(team_id)] = {"pos": pos, "row": t}

    def _rank_desc(values_by_team_id: dict[int, float]) -> dict[int, int]:
        """Dense rank (1..N) for descending values. Ties share rank."""
        # Sort unique values descending
        unique_vals = sorted(set(values_by_team_id.values()), reverse=True)
        rank_by_val = {v: i + 1 for i, v in enumerate(unique_vals)}
        return {tid: rank_by_val[val] for tid, val in values_by_team_id.items()}

    def _rank_asc(values_by_team_id: dict[int, float]) -> dict[int, int]:
        """Dense rank (1..N) for ascending values. Ties share rank."""
        unique_vals = sorted(set(values_by_team_id.values()))
        rank_by_val = {v: i + 1 for i, v in enumerate(unique_vals)}
        return {tid: rank_by_val[val] for tid, val in values_by_team_id.items()}

    # Base stats ranks (all teams)
    gf_by_id = {tid: float(data["row"][5] or 0) for tid, data in team_rows_by_id.items()}
    ga_by_id = {tid: float(data["row"][6] or 0) for tid, data in team_rows_by_id.items()}
    pts_by_id = {tid: float(data["row"][14] or 0) for tid, data in team_rows_by_id.items()}
    gd_by_id = {tid: float(data["row"][7] or 0) for tid, data in team_rows_by_id.items()}
    gf_rank = _rank_desc(gf_by_id)
    ga_rank = _rank_asc(ga_by_id)
    pts_rank = _rank_desc(pts_by_id)
    gd_rank = _rank_desc(gd_by_id)

    # Detailed stats ranks (only teams with stats)
    detailed_by_id = {}
    for d in detailed_stats:
        # d: (id, name, matches_with_stats, total_offsides, total_yellows, total_fouls,
        #     total_corners, total_shots, shots_on_target)
        detailed_by_id[int(d[0])] = {
            "matches_with_stats": int(d[2] or 0),
            "offsides": int(d[3] or 0),
            "yellows": int(d[4] or 0),
            "fouls": int(d[5] or 0),
            "corners": int(d[6] or 0),
            "shots": int(d[7] or 0),
            "shots_on_target": int(d[8] or 0),
        }

    # ranks among teams with stats only
    shots_rank = {}
    corners_rank = {}
    sot_rank = {}
    acc_rank = {}
    fouls_rank = {}
    if detailed_by_id:
        shots_by_id = {tid: float(v["shots"]) for tid, v in detailed_by_id.items()}
        corners_by_id = {tid: float(v["corners"]) for tid, v in detailed_by_id.items()}
        sot_by_id = {tid: float(v["shots_on_target"]) for tid, v in detailed_by_id.items()}
        fouls_by_id = {tid: float(v["fouls"]) for tid, v in detailed_by_id.items()}
        acc_by_id = {
            tid: (float(v["shots_on_target"]) / float(v["shots"])) if v["shots"] > 0 else 0.0
            for tid, v in detailed_by_id.items()
        }
        shots_rank = _rank_desc(shots_by_id)
        corners_rank = _rank_desc(corners_by_id)
        sot_rank = _rank_desc(sot_by_id)
        fouls_rank = _rank_desc(fouls_by_id)
        acc_rank = _rank_desc(acc_by_id)

    # Badge memberships (by team_id)
    def _ids(team_list):
        return {int(t[15]) for t in (team_list or [])}

    badge_defs = [
        ("top_scorer", "Más goleador", _ids(top_scorers), lambda tid: int(team_rows_by_id[tid]["row"][5] or 0), len(top_scorers)),
        ("best_defense", "Mejor defensa", _ids(best_defenses), lambda tid: int(team_rows_by_id[tid]["row"][6] or 0), len(best_defenses)),
        ("most_shots_on_target", "Más tiros al arco", {int(most_shots[0])} if most_shots else set(), lambda tid: int(detailed_by_id.get(tid, {}).get("shots_on_target", 0)), 1),
        ("most_corners", "Más córners", {int(most_corners[0])} if most_corners else set(), lambda tid: int(detailed_by_id.get(tid, {}).get("corners", 0)), 1),
        ("best_accuracy", "Mejor puntería", {int(best_accuracy[0])} if best_accuracy else set(), lambda tid: round((detailed_by_id.get(tid, {}).get("shots_on_target", 0) / detailed_by_id.get(tid, {}).get("shots", 1)) * 100, 1) if detailed_by_id.get(tid, {}).get("shots", 0) else 0, 1),
    ]

    # Build JSON payload for modal
    modal_data = {}
    for tid, data in team_rows_by_id.items():
        t = data["row"]
        team_name = t[0]
        team_payload = {
            "team_id": tid,
            "team_name": team_name,
            "position": data["pos"],
            "points": int(t[14] or 0),
            "goal_diff": int(t[7] or 0),
            "played": int(t[1] or 0),
            "wins": int(t[2] or 0),
            "draws": int(t[3] or 0),
            "losses": int(t[4] or 0),
            "goals_for": int(t[5] or 0),
            "goals_against": int(t[6] or 0),
            "ranks": {
                "points": pts_rank.get(tid),
                "goal_diff": gd_rank.get(tid),
                "goals_for": gf_rank.get(tid),
                "goals_against": ga_rank.get(tid),
                "shots": shots_rank.get(tid),
                "shots_on_target": sot_rank.get(tid),
                "corners": corners_rank.get(tid),
                "accuracy": acc_rank.get(tid),
                "fouls": fouls_rank.get(tid),
            },
            "detailed_stats": detailed_by_id.get(tid),
            "badges": [],
        }

        for key, title, members, value_fn, tied_n in badge_defs:
            if tid in members:
                try:
                    val = value_fn(tid)
                except Exception:
                    val = None
                team_payload["badges"].append({
                    "key": key,
                    "title": title,
                    "value": val,
                    "tied_n": tied_n,
                })

        modal_data[str(tid)] = team_payload

    # Helper to format team names (join multiple with comma)
    def format_teams(team_list, max_show=3):
        if not team_list:
            return '-'
        names = [t[0] for t in team_list[:max_show]]
        result = ', '.join(names)
        if len(team_list) > max_show:
            result += f' (+{len(team_list) - max_show})'
        return result

    # Build HTML
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>League Stats - {league_name}</title>
        <style>
            body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 20px; background: #f8fafc; }}
            h1 {{ color: #1e293b; margin-bottom: 5px; }}
            .subtitle {{ color: #64748b; margin-bottom: 20px; }}
            .nav {{ margin-bottom: 20px; }}
            .nav a {{ color: #3b82f6; text-decoration: none; margin-right: 15px; }}
            .nav a:hover {{ text-decoration: underline; }}
            .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 16px; margin-bottom: 24px; }}
            .card {{ background: white; padding: 16px; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
            .card-title {{ font-size: 12px; color: #64748b; text-transform: uppercase; margin-bottom: 8px; }}
            .card-value {{ font-size: 24px; font-weight: 700; color: #1e293b; }}
            .card-detail {{ font-size: 13px; color: #64748b; margin-top: 4px; }}
            .highlight {{ background: #166534; color: white; }}
            .highlight .card-title {{ color: #bbf7d0; }}
            .highlight .card-value {{ color: white; }}
            .highlight .card-detail {{ color: #bbf7d0; }}
            .warning {{ background: #fef3c7; }}
            .warning .card-title {{ color: #92400e; }}
            table {{ width: 100%; border-collapse: collapse; background: white; border-radius: 8px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.1); margin-top: 24px; }}
            th {{ background: #1e293b; color: white; padding: 12px; text-align: left; font-weight: 600; font-size: 12px; }}
            td {{ padding: 10px 12px; border-bottom: 1px solid #e2e8f0; font-size: 13px; }}
            tr:hover {{ background: #f1f5f9; }}
            .num {{ text-align: right; font-variant-numeric: tabular-nums; }}
            .pos {{ width: 30px; text-align: center; font-weight: 600; color: #64748b; }}
            .section-title {{ font-size: 18px; font-weight: 600; color: #1e293b; margin: 32px 0 16px 0; }}
            .data-status {{ display: flex; gap: 24px; margin-bottom: 24px; padding: 16px; background: #1e293b; border-radius: 8px; color: white; }}
            .data-item {{ text-align: center; }}
            .data-item .label {{ font-size: 11px; color: #94a3b8; }}
            .data-item .value {{ font-size: 20px; font-weight: 700; }}
            .available {{ color: #22c55e; }}
            .missing {{ color: #ef4444; }}
            .league-select {{ padding: 8px 12px; border-radius: 6px; border: 1px solid #e2e8f0; font-size: 14px; margin-left: 12px; }}
            /* Modal */
            .modal-overlay {{ position: fixed; inset: 0; background: rgba(15, 23, 42, 0.55); display: none; align-items: center; justify-content: center; padding: 24px; z-index: 9999; }}
            .modal {{ background: white; width: 100%; max-width: 720px; border-radius: 12px; box-shadow: 0 20px 50px rgba(0,0,0,0.25); overflow: hidden; }}
            .modal-header {{ display: flex; justify-content: space-between; align-items: flex-start; padding: 16px 18px; border-bottom: 1px solid #e2e8f0; }}
            .modal-title {{ font-size: 18px; font-weight: 700; color: #0f172a; }}
            .modal-subtitle {{ font-size: 13px; color: #64748b; margin-top: 4px; }}
            .modal-close {{ border: 0; background: transparent; font-size: 20px; cursor: pointer; color: #64748b; padding: 4px 8px; }}
            .modal-body {{ padding: 16px 18px; }}
            .badge {{ display: inline-block; background: #0f172a; color: white; padding: 3px 8px; border-radius: 999px; font-size: 12px; font-weight: 600; }}
            .badge-gray {{ background: #64748b; }}
            .badges {{ display: flex; flex-wrap: wrap; gap: 8px; margin: 10px 0 14px; }}
            .kpi-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 10px; margin-top: 12px; }}
            .kpi {{ background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 10px; padding: 10px; }}
            .kpi .label {{ font-size: 11px; color: #64748b; text-transform: uppercase; }}
            .kpi .value {{ font-size: 18px; font-weight: 800; color: #0f172a; margin-top: 4px; }}
            .kpi .rank {{ font-size: 12px; color: #334155; margin-top: 2px; }}
            .clickable-row {{ cursor: pointer; }}
        </style>
    </head>
    <body>
        <div class="nav">
            <a href="/dashboard/ops">← Ops</a>
            <a href="/dashboard/ops/daily_comparison">Daily</a>
            <a href="/dashboard/ops/history">History</a>
        </div>

        <h1>League Stats: {league_name}
            <select class="league-select" onchange="window.location.href='?league_id='+this.value+'&season='+document.getElementById('seasonSelect').value">
                <optgroup label="Sudamérica">
                    <option value="239" {'selected' if league_id == 239 else ''}>Colombia Primera A</option>
                    <option value="128" {'selected' if league_id == 128 else ''}>Argentina Primera</option>
                    <option value="250" {'selected' if league_id == 250 else ''}>Paraguay Apertura</option>
                    <option value="252" {'selected' if league_id == 252 else ''}>Paraguay Clausura</option>
                    <option value="71" {'selected' if league_id == 71 else ''}>Brasil Serie A</option>
                </optgroup>
                <optgroup label="Europa">
                    <option value="39" {'selected' if league_id == 39 else ''}>Premier League</option>
                    <option value="140" {'selected' if league_id == 140 else ''}>La Liga</option>
                    <option value="135" {'selected' if league_id == 135 else ''}>Serie A Italia</option>
                    <option value="78" {'selected' if league_id == 78 else ''}>Bundesliga</option>
                    <option value="61" {'selected' if league_id == 61 else ''}>Ligue 1</option>
                </optgroup>
                <optgroup label="Norteamérica">
                    <option value="253" {'selected' if league_id == 253 else ''}>MLS</option>
                    <option value="262" {'selected' if league_id == 262 else ''}>Liga MX</option>
                </optgroup>
            </select>
            <select id="seasonSelect" class="league-select" onchange="window.location.href='?league_id={league_id}&season='+this.value">
                <option value="2026" {'selected' if season == 2026 else ''}>2026</option>
                <option value="2025" {'selected' if season == 2025 else ''}>2025</option>
                <option value="2024" {'selected' if season == 2024 else ''}>2024</option>
                <option value="2023" {'selected' if season == 2023 else ''}>2023</option>
                <option value="2022" {'selected' if season == 2022 else ''}>2022</option>
            </select>
        </h1>
        <p class="subtitle">Temporada {season} - Datos calculados de {total_matches} partidos</p>

        <div class="section-title">Disponibilidad de Datos para Narrativas</div>
        <div class="data-status">
            <div class="data-item">
                <div class="label">Partidos FT</div>
                <div class="value available">{availability[0] if availability else 0}</div>
            </div>
            <div class="data-item">
                <div class="label">Con Stats Detalladas</div>
                <div class="value {'available' if availability and availability[1] > 0 else 'missing'}">{availability[1] if availability else 0}</div>
            </div>
            <div class="data-item">
                <div class="label">% Stats</div>
                <div class="value {'available' if availability and availability[1]/availability[0]*100 > 50 else 'missing'}">{round(availability[1]/availability[0]*100, 1) if availability and availability[0] > 0 else 0}%</div>
            </div>
            <div class="data-item">
                <div class="label">Goles Totales</div>
                <div class="value available">{total_goals}</div>
            </div>
            <div class="data-item">
                <div class="label">Prom. Goles/Partido</div>
                <div class="value available">{avg_goals_per_match}</div>
            </div>
        </div>

        <div class="section-title">Líderes de Liga (Datos Verificables)</div>
        <div class="grid">
            <div class="card highlight">
                <div class="card-title">Más Goleador ({len(top_scorers)} equipo{'s' if len(top_scorers) != 1 else ''})</div>
                <div class="card-value">{format_teams(top_scorers)}</div>
                <div class="card-detail">{top_scorers[0][5] if top_scorers else 0} goles</div>
            </div>
            <div class="card highlight">
                <div class="card-title">Mejor Defensa ({len(best_defenses)} equipo{'s' if len(best_defenses) != 1 else ''})</div>
                <div class="card-value">{format_teams(best_defenses)}</div>
                <div class="card-detail">{best_defenses[0][6] if best_defenses else 0} goles recibidos</div>
            </div>
            <div class="card">
                <div class="card-title">Valla Más Vencida ({len(worst_defenses)} equipo{'s' if len(worst_defenses) != 1 else ''})</div>
                <div class="card-value">{format_teams(worst_defenses)}</div>
                <div class="card-detail">{worst_defenses[0][6] if worst_defenses else 0} goles recibidos</div>
            </div>
            <div class="card">
                <div class="card-title">Más Victorias ({len(most_wins_list)} equipo{'s' if len(most_wins_list) != 1 else ''})</div>
                <div class="card-value">{format_teams(most_wins_list)}</div>
                <div class="card-detail">{most_wins_list[0][2] if most_wins_list else 0} victoria{'s' if most_wins_list and most_wins_list[0][2] != 1 else ''}</div>
            </div>
            <div class="card">
                <div class="card-title">Más Empates ({len(most_draws_list)} equipo{'s' if len(most_draws_list) != 1 else ''})</div>
                <div class="card-value">{format_teams(most_draws_list)}</div>
                <div class="card-detail">{most_draws_list[0][3] if most_draws_list else 0} empate{'s' if most_draws_list and most_draws_list[0][3] != 1 else ''}</div>
            </div>
            <div class="card">
                <div class="card-title">Más Derrotas ({len(most_losses_list)} equipo{'s' if len(most_losses_list) != 1 else ''})</div>
                <div class="card-value">{format_teams(most_losses_list)}</div>
                <div class="card-detail">{most_losses_list[0][4] if most_losses_list else 0} derrota{'s' if most_losses_list and most_losses_list[0][4] != 1 else ''}</div>
            </div>
            <div class="card">
                <div class="card-title">Mejor Local ({len(best_home_list)} equipo{'s' if len(best_home_list) != 1 else ''})</div>
                <div class="card-value">{format_teams(best_home_list)}</div>
                <div class="card-detail">{best_home_list[0][10] if best_home_list else 0} goles de local</div>
            </div>
            <div class="card">
                <div class="card-title">Mejor Visitante ({len(best_away_list)} equipo{'s' if len(best_away_list) != 1 else ''})</div>
                <div class="card-value">{format_teams(best_away_list)}</div>
                <div class="card-detail">{best_away_list[0][11] if best_away_list else 0} goles de visita</div>
            </div>
        </div>

        <div class="section-title">Stats Detalladas (de {availability[1] if availability else 0} partidos con stats)</div>
        <div class="grid">
            <div class="card {'highlight' if most_offsides else 'warning'}">
                <div class="card-title">Más Fueras de Lugar</div>
                <div class="card-value">{most_offsides[1] if most_offsides else '—'}</div>
                <div class="card-detail">{most_offsides[3] if most_offsides else 0} offsides en {most_offsides[2] if most_offsides else 0} partidos</div>
            </div>
            <div class="card {'highlight' if most_yellows else 'warning'}">
                <div class="card-title">Más Tarjetas Amarillas</div>
                <div class="card-value">{most_yellows[1] if most_yellows else '—'}</div>
                <div class="card-detail">{most_yellows[4] if most_yellows else 0} amarillas en {most_yellows[2] if most_yellows else 0} partidos</div>
            </div>
            <div class="card {'highlight' if most_fouls else 'warning'}">
                <div class="card-title">Más Faltas Cometidas</div>
                <div class="card-value">{most_fouls[1] if most_fouls else '—'}</div>
                <div class="card-detail">{most_fouls[5] if most_fouls else 0} faltas en {most_fouls[2] if most_fouls else 0} partidos</div>
            </div>
            <div class="card {'highlight' if most_corners else 'warning'}">
                <div class="card-title">Más Córners</div>
                <div class="card-value">{most_corners[1] if most_corners else '—'}</div>
                <div class="card-detail">{most_corners[6] if most_corners else 0} córners en {most_corners[2] if most_corners else 0} partidos</div>
            </div>
            <div class="card {'highlight' if most_shots else 'warning'}">
                <div class="card-title">Más Tiros</div>
                <div class="card-value">{most_shots[1] if most_shots else '—'}</div>
                <div class="card-detail">{most_shots[7] if most_shots else 0} tiros totales ({most_shots[8] if most_shots else 0} al arco)</div>
            </div>
            <div class="card {'highlight' if best_accuracy else 'warning'}">
                <div class="card-title">Mejor Puntería</div>
                <div class="card-value">{best_accuracy[1] if best_accuracy else '—'}</div>
                <div class="card-detail">{round(best_accuracy[8] / best_accuracy[7] * 100, 1) if best_accuracy and best_accuracy[7] > 0 else 0}% tiros al arco ({best_accuracy[8] if best_accuracy else 0}/{best_accuracy[7] if best_accuracy else 0})</div>
            </div>
        </div>

        <div class="section-title warning-box">Datos NO Disponibles</div>
        <div class="grid">
            <div class="card warning">
                <div class="card-title">Goles de Cabeza</div>
                <div class="card-value">—</div>
                <div class="card-detail">API no provee desglose por tipo de gol</div>
            </div>
            <div class="card warning">
                <div class="card-title">xG Acumulado</div>
                <div class="card-value">—</div>
                <div class="card-detail">Solo algunos partidos tienen xG</div>
            </div>
        </div>

        <div class="section-title">Tabla de Posiciones</div>
        <table>
            <thead>
                <tr>
                    <th class="pos">#</th>
                    <th>Equipo</th>
                    <th class="num">PJ</th>
                    <th class="num">G</th>
                    <th class="num">E</th>
                    <th class="num">P</th>
                    <th class="num">GF</th>
                    <th class="num">GC</th>
                    <th class="num">DIF</th>
                    <th class="num">PTS</th>
                </tr>
            </thead>
            <tbody>
    """

    for i, team in enumerate(teams, 1):
        team_id = team[15]
        html += f"""
                <tr class="clickable-row" data-team-id="{team_id}" title="Click para ver detalles">
                    <td class="pos">{i}</td>
                    <td>{team[0]}</td>
                    <td class="num">{team[1]}</td>
                    <td class="num">{team[2]}</td>
                    <td class="num">{team[3]}</td>
                    <td class="num">{team[4]}</td>
                    <td class="num">{team[5]}</td>
                    <td class="num">{team[6]}</td>
                    <td class="num">{team[7]}</td>
                    <td class="num"><strong>{team[14]}</strong></td>
                </tr>
        """

    import json as json_module
    modal_data_json = json_module.dumps(modal_data)

    html += f"""
            </tbody>
        </table>
        <div id="teamModalOverlay" class="modal-overlay" role="dialog" aria-modal="true" aria-hidden="true">
            <div class="modal" role="document">
                <div class="modal-header">
                    <div>
                        <div id="teamModalTitle" class="modal-title">Equipo</div>
                        <div id="teamModalSubtitle" class="modal-subtitle">—</div>
                    </div>
                    <button id="teamModalClose" class="modal-close" aria-label="Cerrar">×</button>
                </div>
                <div class="modal-body">
                    <div id="teamModalBadges" class="badges"></div>
                    <div id="teamModalKPIs" class="kpi-grid"></div>
                </div>
            </div>
        </div>
        <script>
        const TEAM_DATA = {modal_data_json};

        function fmtRank(rank, total) {{
            if (!rank) return '—';
            return `#${{rank}} de ${{total}}`;
        }}

        function openTeamModal(teamId) {{
            const data = TEAM_DATA[String(teamId)];
            if (!data) return;

            const totalTeams = Object.keys(TEAM_DATA).length;

            document.getElementById('teamModalTitle').textContent = data.team_name;
            document.getElementById('teamModalSubtitle').textContent =
                `Posición: ${{data.position}} | PTS: ${{data.points}} | DIF: ${{data.goal_diff}}`;

            // Badges (logros)
            const badgesEl = document.getElementById('teamModalBadges');
            badgesEl.innerHTML = '';
            if (data.badges && data.badges.length) {{
                for (const b of data.badges) {{
                    const val = (b.value === null || b.value === undefined) ? '' : ` (${{b.value}})`;
                    const tied = b.tied_n ? ` — ${{1}} de ${{b.tied_n}} equipos` : '';
                    const span = document.createElement('span');
                    span.className = 'badge';
                    span.textContent = `${{b.title}}${{val}}${{tied}}`;
                    badgesEl.appendChild(span);
                }}
            }} else {{
                const span = document.createElement('span');
                span.className = 'badge badge-gray';
                span.textContent = 'Sin insignias destacadas';
                badgesEl.appendChild(span);
            }}

            // KPIs con ranking
            const kpisEl = document.getElementById('teamModalKPIs');
            kpisEl.innerHTML = '';

            const metrics = [
                {{ key: 'goals_for', label: 'Goles a favor', value: data.goals_for, rank: data.ranks.goals_for }},
                {{ key: 'goals_against', label: 'Goles en contra', value: data.goals_against, rank: data.ranks.goals_against }},
                {{ key: 'shots', label: 'Tiros totales', value: data.detailed_stats ? data.detailed_stats.shots : null, rank: data.ranks.shots }},
                {{ key: 'shots_on_target', label: 'Tiros al arco', value: data.detailed_stats ? data.detailed_stats.shots_on_target : null, rank: data.ranks.shots_on_target }},
                {{ key: 'corners', label: 'Córners', value: data.detailed_stats ? data.detailed_stats.corners : null, rank: data.ranks.corners }},
                {{ key: 'accuracy', label: 'Puntería (%)', value: (() => {{
                    if (!data.detailed_stats) return null;
                    const s = data.detailed_stats.shots || 0;
                    const sot = data.detailed_stats.shots_on_target || 0;
                    if (s <= 0) return 0;
                    return Math.round((sot / s) * 1000) / 10;
                }})(), rank: data.ranks.accuracy }},
                {{ key: 'fouls', label: 'Faltas', value: data.detailed_stats ? data.detailed_stats.fouls : null, rank: data.ranks.fouls }},
            ];

            for (const m of metrics) {{
                const div = document.createElement('div');
                div.className = 'kpi';
                const value = (m.value === null || m.value === undefined) ? '—' : m.value;
                div.innerHTML = `
                    <div class="label">${{m.label}}</div>
                    <div class="value">${{value}}</div>
                    <div class="rank">${{fmtRank(m.rank, totalTeams)}}</div>
                `;
                kpisEl.appendChild(div);
            }}

            const overlay = document.getElementById('teamModalOverlay');
            overlay.style.display = 'flex';
            overlay.setAttribute('aria-hidden', 'false');
        }}

        function closeTeamModal() {{
            const overlay = document.getElementById('teamModalOverlay');
            overlay.style.display = 'none';
            overlay.setAttribute('aria-hidden', 'true');
        }}

        // Row click handlers
        document.querySelectorAll('tr[data-team-id]').forEach(row => {{
            row.addEventListener('click', () => {{
                const teamId = row.getAttribute('data-team-id');
                openTeamModal(teamId);
            }});
        }});

        // Close controls
        document.getElementById('teamModalClose').addEventListener('click', closeTeamModal);
        document.getElementById('teamModalOverlay').addEventListener('click', (e) => {{
            if (e.target && e.target.id === 'teamModalOverlay') closeTeamModal();
        }});
        document.addEventListener('keydown', (e) => {{
            if (e.key === 'Escape') closeTeamModal();
        }});
        </script>
    </body>
    </html>
    """

    return HTMLResponse(content=html)


@app.get("/dashboard/ops/team_overrides.json")
async def ops_team_overrides(
    request: Request,
    external_team_id: int = None,
    session: AsyncSession = Depends(get_async_session),
):
    """
    List all team identity overrides configured in the system.

    Args:
        external_team_id: Optional filter by API-Football team ID (e.g., 1134 for La Equidad).

    Used to verify rebranding configurations like La Equidad → Internacional de Bogotá.
    """
    if not _verify_dashboard_token(request):
        raise HTTPException(status_code=401, detail="Dashboard access requires valid token.")

    query = """
        SELECT
            id, provider, external_team_id, display_name, display_logo_url,
            effective_from, effective_to, reason, updated_by, created_at, updated_at
        FROM team_overrides
    """
    params = {}
    if external_team_id:
        query += " WHERE external_team_id = :external_team_id"
        params["external_team_id"] = external_team_id
    query += " ORDER BY external_team_id, effective_from DESC"

    result = await session.execute(text(query), params)
    rows = result.fetchall()

    overrides = []
    for row in rows:
        overrides.append({
            "id": row[0],
            "provider": row[1],
            "external_team_id": row[2],
            "display_name": row[3],
            "display_logo_url": row[4],
            "effective_from": str(row[5]) if row[5] else None,
            "effective_to": str(row[6]) if row[6] else None,
            "reason": row[7],
            "updated_by": row[8],
            "created_at": str(row[9]) if row[9] else None,
            "updated_at": str(row[10]) if row[10] else None,
        })

    return {
        "count": len(overrides),
        "overrides": overrides,
        "note": "These overrides replace team names/logos for matches on or after effective_from date.",
    }


@app.get("/dashboard/ops/job_runs.json")
async def ops_job_runs(
    request: Request,
    job_name: str = None,
    limit: int = 20,
    session: AsyncSession = Depends(get_async_session),
):
    """
    List recent job runs from the job_runs table (P1-B fallback).

    Args:
        job_name: Optional filter by job name (stats_backfill, odds_sync, fastpath).
        limit: Max rows to return (default 20).

    Used to verify job tracking is working and debug jobs_health fallback.
    """
    if not _verify_dashboard_token(request):
        raise HTTPException(status_code=401, detail="Dashboard access requires valid token.")

    # Check if table exists
    try:
        check_result = await session.execute(text("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables
                WHERE table_name = 'job_runs'
            )
        """))
        table_exists = check_result.scalar()
        if not table_exists:
            return {
                "count": 0,
                "runs": [],
                "note": "job_runs table does not exist. Run migration 028_job_runs.py first.",
            }
    except Exception as e:
        return {"error": str(e), "note": "Failed to check table existence"}

    query = """
        SELECT id, job_name, status, started_at, finished_at, duration_ms, error_message, metrics
        FROM job_runs
    """
    params = {"limit": min(limit, 100)}  # Cap at 100

    if job_name:
        query += " WHERE job_name = :job_name"
        params["job_name"] = job_name

    query += " ORDER BY finished_at DESC LIMIT :limit"

    result = await session.execute(text(query), params)
    rows = result.fetchall()

    runs = []
    for row in rows:
        runs.append({
            "id": row[0],
            "job_name": row[1],
            "status": row[2],
            "started_at": row[3].isoformat() + "Z" if row[3] else None,
            "finished_at": row[4].isoformat() + "Z" if row[4] else None,
            "duration_ms": row[5],
            "error_message": row[6],
            "metrics": row[7],
        })

    # Get last success per job for summary
    summary_result = await session.execute(text("""
        SELECT DISTINCT ON (job_name)
            job_name,
            finished_at as last_success_at
        FROM job_runs
        WHERE status = 'ok'
        ORDER BY job_name, finished_at DESC
    """))
    summary_rows = summary_result.fetchall()
    summary = {
        row[0]: row[1].isoformat() + "Z" if row[1] else None
        for row in summary_rows
    }

    return {
        "count": len(runs),
        "runs": runs,
        "last_success_by_job": summary,
        "note": "Job runs tracked for ops dashboard fallback when Prometheus is cold.",
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


# =============================================================================
# OPS ALERTS: Grafana Webhook → Bell + Toast Notifications
# =============================================================================


def _verify_alerts_webhook_secret(request: Request) -> bool:
    """Verify webhook authentication via X-Alerts-Secret header or Authorization header.

    Supports two formats:
    1. X-Alerts-Secret: <token>  (direct header)
    2. Authorization: X-Alerts-Secret <token>  (Grafana webhook format)
    """
    settings = get_settings()
    if not settings.ALERTS_WEBHOOK_SECRET:
        return False  # Webhook disabled if no secret configured

    # Try direct header first
    provided = request.headers.get("X-Alerts-Secret", "")
    if provided == settings.ALERTS_WEBHOOK_SECRET:
        return True

    # Try Authorization header with custom scheme (Grafana format)
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("X-Alerts-Secret "):
        provided = auth_header[len("X-Alerts-Secret "):]
        return provided == settings.ALERTS_WEBHOOK_SECRET

    return False


@app.post("/dashboard/ops/alerts/webhook")
async def ops_alerts_webhook(
    request: Request,
    session: AsyncSession = Depends(get_async_session),
):
    """
    Receive alerts from Grafana Alerting webhook.

    Auth: X-Alerts-Secret header (dedicated secret, not dashboard token).

    Expects Grafana Unified Alerting format. Tolerant parsing.
    Upserts by dedupe_key (fingerprint) for idempotence.
    """
    if not _verify_alerts_webhook_secret(request):
        raise HTTPException(status_code=401, detail="Invalid or missing X-Alerts-Secret")

    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON payload")

    # Grafana sends alerts in different formats depending on version
    # Handle both single alert and array of alerts
    alerts = []
    if isinstance(payload, list):
        alerts = payload
    elif isinstance(payload, dict):
        # Grafana Unified Alerting format: { alerts: [...] }
        if "alerts" in payload:
            alerts = payload.get("alerts", [])
        else:
            # Single alert object
            alerts = [payload]

    if not alerts:
        return {"status": "ok", "processed": 0, "message": "No alerts in payload"}

    processed = 0
    errors = []

    for alert in alerts:
        try:
            # Extract fields with fallbacks
            labels = alert.get("labels", {})
            annotations = alert.get("annotations", {})

            # Dedupe key: prefer fingerprint, fallback to alertname + labels hash
            fingerprint = alert.get("fingerprint", "")
            if not fingerprint:
                # Generate from alertname + sorted labels
                import hashlib
                alertname = labels.get("alertname", "unknown")
                labels_str = str(sorted(labels.items()))
                fingerprint = hashlib.sha256(f"{alertname}:{labels_str}".encode()).hexdigest()[:32]

            # Status: firing or resolved
            status = alert.get("status", "firing").lower()
            if status not in ("firing", "resolved"):
                status = "firing"

            # Severity from labels
            severity = labels.get("severity", "warning").lower()
            if severity not in ("critical", "warning", "info"):
                severity = "warning"

            # Title: prefer annotations.summary, fallback to labels.alertname
            title = (
                annotations.get("summary")
                or labels.get("alertname")
                or alert.get("title")
                or "Unknown Alert"
            )[:500]  # Truncate

            # Message: prefer annotations.description
            message = annotations.get("description") or annotations.get("message") or ""
            if len(message) > 1000:
                message = message[:997] + "..."

            # Timestamps (convert to naive UTC for DB compatibility)
            # Helper: normalize any timestamp to UTC naive (repo convention)
            def _to_utc_naive(value: str | datetime | None) -> datetime | None:
                """
                Convert timestamp to UTC naive datetime.

                - None -> None
                - ISO string with tz -> parse, convert to UTC, strip tzinfo
                - datetime aware -> convert to UTC, strip tzinfo
                - datetime naive -> assume UTC, return as-is
                """
                from datetime import timezone

                if value is None:
                    return None

                if isinstance(value, str):
                    if not value or value == "0001-01-01T00:00:00Z":
                        return None
                    try:
                        # Parse ISO format, handle Z suffix
                        value = datetime.fromisoformat(value.replace("Z", "+00:00"))
                    except ValueError:
                        logger.warning(f"[ALERTS] Invalid timestamp format: {value[:50]}")
                        return None

                if isinstance(value, datetime):
                    if value.tzinfo is not None:
                        # Convert to UTC then strip tzinfo
                        value = value.astimezone(timezone.utc).replace(tzinfo=None)
                    # else: already naive, assume UTC
                    return value

                return None

            starts_at = _to_utc_naive(alert.get("startsAt"))
            ends_at = _to_utc_naive(alert.get("endsAt"))

            # Source URL (Grafana panel/alert link)
            source_url = (
                alert.get("generatorURL")
                or alert.get("silenceURL")
                or annotations.get("runbook_url")
            )

            # Guardrail: ensure timestamps are naive UTC before DB insert
            # (asyncpg will reject aware datetimes for TIMESTAMP WITHOUT TIME ZONE)
            if starts_at is not None and starts_at.tzinfo is not None:
                logger.warning(f"[ALERTS] starts_at still has tzinfo after normalization, forcing naive: {fingerprint}")
                starts_at = starts_at.replace(tzinfo=None)
            if ends_at is not None and ends_at.tzinfo is not None:
                logger.warning(f"[ALERTS] ends_at still has tzinfo after normalization, forcing naive: {fingerprint}")
                ends_at = ends_at.replace(tzinfo=None)

            # Upsert into ops_alerts
            now = datetime.utcnow()

            # Check if exists
            existing = await session.execute(
                select(OpsAlert).where(OpsAlert.dedupe_key == fingerprint)
            )
            existing_alert = existing.scalar_one_or_none()

            if existing_alert:
                # Update existing
                existing_alert.status = status
                existing_alert.severity = severity
                existing_alert.title = title
                existing_alert.message = message
                existing_alert.labels = labels
                existing_alert.annotations = annotations
                existing_alert.starts_at = starts_at or existing_alert.starts_at
                existing_alert.ends_at = ends_at
                existing_alert.source_url = source_url or existing_alert.source_url
                existing_alert.last_seen_at = now
                existing_alert.updated_at = now
                # If resolved, mark as read (auto-clear)
                if status == "resolved":
                    existing_alert.is_read = True
            else:
                # Insert new
                new_alert = OpsAlert(
                    dedupe_key=fingerprint,
                    status=status,
                    severity=severity,
                    title=title,
                    message=message,
                    labels=labels,
                    annotations=annotations,
                    starts_at=starts_at,
                    ends_at=ends_at,
                    source="grafana",
                    source_url=source_url,
                    first_seen_at=now,
                    last_seen_at=now,
                    is_read=False,
                    is_ack=False,
                    created_at=now,
                    updated_at=now,
                )
                session.add(new_alert)

            processed += 1

        except Exception as e:
            errors.append(str(e)[:100])
            logger.warning(f"Failed to process alert: {e}")

    await session.commit()

    return {
        "status": "ok",
        "processed": processed,
        "errors": errors if errors else None,
    }


@app.get("/dashboard/ops/alerts.json")
async def ops_alerts_list(
    request: Request,
    limit: int = 50,
    status: str = "all",  # firing, resolved, all
    unread_only: bool = False,
    session: AsyncSession = Depends(get_async_session),
):
    """
    Get ops alerts for bell dropdown.

    Auth: X-Dashboard-Token (same as ops.json).

    Returns unread_count and list of recent alerts.
    """
    if not _verify_dashboard_token(request):
        raise HTTPException(status_code=401, detail="Dashboard access requires valid token.")

    # Clamp limit
    limit = min(max(1, limit), 100)

    # Build query
    query = select(OpsAlert).order_by(OpsAlert.last_seen_at.desc())

    # Status filter
    if status == "firing":
        query = query.where(OpsAlert.status == "firing")
    elif status == "resolved":
        query = query.where(OpsAlert.status == "resolved")
    # else: all

    # Unread filter
    if unread_only:
        query = query.where(OpsAlert.is_read == False)

    query = query.limit(limit)

    result = await session.execute(query)
    alerts = result.scalars().all()

    # Get unread count (always firing + unread)
    unread_result = await session.execute(
        select(func.count(OpsAlert.id)).where(
            OpsAlert.is_read == False,
            OpsAlert.status == "firing"
        )
    )
    unread_count = unread_result.scalar() or 0

    # Format response
    items = []
    for a in alerts:
        items.append({
            "id": a.id,
            "dedupe_key": a.dedupe_key,
            "status": a.status,
            "severity": a.severity,
            "title": a.title,
            "message": a.message[:200] if a.message else None,  # Truncate for list
            "starts_at": a.starts_at.isoformat() if a.starts_at else None,
            "ends_at": a.ends_at.isoformat() if a.ends_at else None,
            "last_seen_at": a.last_seen_at.isoformat() if a.last_seen_at else None,
            "source_url": a.source_url,
            "is_read": a.is_read,
            "is_ack": a.is_ack,
        })

    return {
        "unread_count": unread_count,
        "items": items,
    }


@app.post("/dashboard/ops/alerts/ack")
async def ops_alerts_ack(
    request: Request,
    session: AsyncSession = Depends(get_async_session),
):
    """
    Mark alerts as read/acknowledged.

    Auth: X-Dashboard-Token.

    Body: { "ids": [1,2,3] } or { "ack_all": true }
    """
    if not _verify_dashboard_token(request):
        raise HTTPException(status_code=401, detail="Dashboard access requires valid token.")

    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    now = datetime.utcnow()
    updated = 0

    if body.get("ack_all"):
        # Mark all unread firing alerts as read
        result = await session.execute(
            text("""
                UPDATE ops_alerts
                SET is_read = true, is_ack = true, updated_at = :now
                WHERE is_read = false AND status = 'firing'
            """),
            {"now": now}
        )
        updated = result.rowcount
    elif body.get("ids"):
        ids = body.get("ids", [])
        if not isinstance(ids, list):
            raise HTTPException(status_code=400, detail="ids must be an array")
        # Mark specific alerts as read
        result = await session.execute(
            text("""
                UPDATE ops_alerts
                SET is_read = true, is_ack = true, updated_at = :now
                WHERE id = ANY(:ids)
            """),
            {"now": now, "ids": ids}
        )
        updated = result.rowcount
    else:
        raise HTTPException(status_code=400, detail="Provide 'ids' array or 'ack_all': true")

    await session.commit()

    return {
        "status": "ok",
        "updated": updated,
    }


@app.post("/ops/migrate-weather-precip-prob", include_in_schema=False)
async def migrate_weather_precip_prob(
    session: AsyncSession = Depends(get_async_session),
    _: None = Depends(_verify_dashboard_token),
):
    """
    One-time migration to add precipitation probability field to match_weather.
    Also triggers a backfill for upcoming matches.
    """
    from sqlalchemy import text

    migrations = [
        "ALTER TABLE match_weather ADD COLUMN IF NOT EXISTS precip_prob double precision",
    ]

    results = []
    for sql in migrations:
        try:
            await session.execute(text(sql))
            await session.commit()
            results.append({"sql": sql[:60] + "...", "status": "ok"})
        except Exception as e:
            results.append({"sql": sql[:60] + "...", "status": "error", "error": str(e)})

    # Verify column was added
    verify = await session.execute(
        text("""
            SELECT column_name FROM information_schema.columns
            WHERE table_name = 'match_weather'
            ORDER BY ordinal_position
        """)
    )
    columns = [row[0] for row in verify.all()]

    return {
        "status": "ok",
        "migrations": results,
        "verified_columns": columns,
        "note": "Backfill will happen automatically on next weather_sync job run",
    }


@app.post("/ops/trigger-weather-sync", include_in_schema=False)
async def trigger_weather_sync(
    hours: int = 48,
    limit: int = 100,
    _: None = Depends(_verify_dashboard_token),
):
    """
    Manually trigger weather forecast capture for upcoming matches.

    Args:
        hours: Lookahead window (default 48h)
        limit: Max matches to process (default 100)

    Returns:
        Stats from the weather capture job.
    """
    from app.etl.sota_jobs import capture_weather_prekickoff

    async with AsyncSessionLocal() as session:
        stats = await capture_weather_prekickoff(
            session,
            hours=hours,
            limit=limit,
            horizon=24,
        )
        await session.commit()

    return {
        "status": "ok",
        "stats": stats,
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


# -----------------------------------------------------------------------------
# Dashboard Incidents Endpoint
# -----------------------------------------------------------------------------
_incidents_cache = {
    "data": None,
    "timestamp": 0,
    "ttl": 30,  # 30 seconds cache (balance freshness vs load)
}


async def _aggregate_incidents(session) -> list[dict]:
    """
    Aggregate incidents from multiple sources into a unified list.

    Sources:
    1. Sentry issues (from cached ops data)
    2. Health alerts (predictions, jobs, fastpath, budget)
    3. System-level incidents (from ops health checks)

    Returns list of incidents sorted by created_at DESC.
    """
    incidents = []
    now = datetime.utcnow()
    now_iso = now.isoformat() + "Z"

    # Helper to generate stable incident ID from source + key
    def make_id(source: str, key: str) -> int:
        import hashlib
        h = hashlib.md5(f"{source}:{key}".encode()).hexdigest()
        return int(h[:8], 16)  # First 8 hex chars → int

    # =========================================================================
    # SOURCE 1: Sentry Issues (from health API)
    # =========================================================================
    try:
        sentry_data = await _fetch_sentry_health()
        if sentry_data.get("status") != "degraded":
            top_issues = sentry_data.get("top_issues", [])
            for issue in top_issues[:10]:  # Limit to top 10
                title = issue.get("title", "Unknown Sentry Issue")
                level = issue.get("level", "error")
                count = issue.get("count", 0)
                last_seen = issue.get("last_seen")

                # Map Sentry level to severity
                severity = "warning"
                if level in ("error", "fatal"):
                    severity = "critical"
                elif level == "warning":
                    severity = "warning"
                else:
                    severity = "info"

                # Determine status based on recency
                status = "active"
                if last_seen:
                    try:
                        from datetime import timedelta
                        last_dt = datetime.fromisoformat(last_seen.replace("Z", ""))
                        if now - last_dt > timedelta(hours=24):
                            status = "resolved"
                    except Exception:
                        pass

                incidents.append({
                    "id": make_id("sentry", title[:50]),
                    "severity": severity,
                    "status": status,
                    "type": "sentry",
                    "title": title[:80],
                    "description": f"Sentry: {count} events. Level: {level}."[:200],
                    "created_at": last_seen or now_iso,
                    "updated_at": last_seen or now_iso,
                    "runbook_url": None,
                })
    except Exception as e:
        logger.warning(f"Could not fetch Sentry incidents: {e}")

    # =========================================================================
    # SOURCE 2: Predictions Health
    # =========================================================================
    try:
        pred_health = await _calculate_predictions_health(session)
        status_val = pred_health.get("status", "ok")
        # Backend may use "warn" - normalize to "warning"
        if status_val in ("warn", "warning", "critical"):
            reason = pred_health.get("status_reason", "Predictions health degraded")
            ns_missing = pred_health.get("ns_matches_next_48h_missing_prediction", 0)
            ns_total = pred_health.get("ns_matches_next_48h", 0)
            coverage = pred_health.get("ns_coverage_pct", 100)
            severity = "warning" if status_val == "warn" else status_val

            incidents.append({
                "id": make_id("predictions", "health"),
                "severity": severity,
                "status": "active",
                "type": "predictions",
                "title": f"Predictions coverage at {coverage}%"[:80],
                "description": f"{reason}. {ns_missing}/{ns_total} NS matches missing predictions."[:200],
                "created_at": now_iso,
                "updated_at": now_iso,
                "runbook_url": "docs/OPS_RUNBOOK.md#predictions-health",
            })
    except Exception as e:
        logger.warning(f"Could not check predictions health: {e}")

    # =========================================================================
    # SOURCE 3: Jobs Health
    # =========================================================================
    try:
        jobs_health = await _calculate_jobs_health_summary(session)
        overall_status = jobs_health.get("status", "ok")

        # Check individual jobs
        for job_name in ["stats_backfill", "odds_sync", "fastpath"]:
            job_data = jobs_health.get(job_name, {})
            job_status = job_data.get("status", "ok")
            # Backend uses "warn" but we normalize to "warning" for dashboard
            if job_status in ("warn", "warning", "critical"):
                mins_since = job_data.get("minutes_since_success")
                help_url = job_data.get("help_url")

                # Normalize "warn" to "warning"
                severity = "warning" if job_status == "warn" else job_status
                time_str = f"{int(mins_since)}m" if mins_since and mins_since < 60 else (
                    f"{int(mins_since/60)}h" if mins_since else "unknown"
                )

                incidents.append({
                    "id": make_id("jobs", job_name),
                    "severity": severity,
                    "status": "active",
                    "type": "scheduler",
                    "title": f"Job '{job_name}' unhealthy"[:80],
                    "description": f"Last success: {time_str} ago. Status: {job_status}."[:200],
                    "created_at": now_iso,
                    "updated_at": now_iso,
                    "runbook_url": help_url,
                })
    except Exception as e:
        logger.warning(f"Could not check jobs health: {e}")

    # =========================================================================
    # SOURCE 4: FastPath Health (LLM narratives)
    # =========================================================================
    try:
        fp_health = await _calculate_fastpath_health(session)
        fp_status = fp_health.get("status", "ok")
        # Backend may use "warn" - normalize to "warning"
        if fp_status in ("warn", "warning", "critical"):
            error_rate = fp_health.get("last_60m", {}).get("error_rate_pct", 0)
            in_queue = fp_health.get("last_60m", {}).get("in_queue", 0)
            reason = fp_health.get("status_reason", "Fastpath degraded")
            severity = "warning" if fp_status == "warn" else fp_status

            incidents.append({
                "id": make_id("fastpath", "health"),
                "severity": severity,
                "status": "active",
                "type": "llm",
                "title": f"Fastpath error rate {error_rate}%"[:80],
                "description": f"{reason}. Queue: {in_queue}."[:200],
                "created_at": now_iso,
                "updated_at": now_iso,
                "runbook_url": "docs/OPS_RUNBOOK.md#fastpath-health",
            })
    except Exception as e:
        logger.warning(f"Could not check fastpath health: {e}")

    # =========================================================================
    # SOURCE 5: API Budget
    # =========================================================================
    try:
        budget_data = await _fetch_api_football_budget()
        budget_status = budget_data.get("status", "ok")
        # Backend may use "warn" - normalize to "warning"
        if budget_status in ("warn", "warning", "critical"):
            pct_used = budget_data.get("pct_used", 0)
            remaining = budget_data.get("requests_remaining", 0)
            severity = "warning" if budget_status == "warn" else budget_status

            incidents.append({
                "id": make_id("budget", "api-football"),
                "severity": severity,
                "status": "active",
                "type": "api_budget",
                "title": f"API-Football budget at {pct_used}%"[:80],
                "description": f"Remaining requests: {remaining}."[:200],
                "created_at": now_iso,
                "updated_at": now_iso,
                "runbook_url": "docs/OPS_RUNBOOK.md#api-budget",
            })
    except Exception as e:
        logger.warning(f"Could not check API budget: {e}")

    # Sort by severity (critical first) then by created_at DESC
    severity_order = {"critical": 0, "warning": 1, "info": 2}
    incidents.sort(key=lambda x: (severity_order.get(x["severity"], 2), x["created_at"]), reverse=False)
    # Reverse to get DESC order for created_at within same severity
    incidents.sort(key=lambda x: severity_order.get(x["severity"], 2))

    return incidents


@app.get("/dashboard/incidents.json")
async def get_incidents_dashboard(
    request: Request,
    status: list[str] = Query(default=[]),
    severity: list[str] = Query(default=[]),
    type: str = Query(default=None, alias="type"),
    q: str = Query(default=None),
    page: int = Query(default=1, ge=1),
    limit: int = Query(default=50, ge=1, le=100),
    session: AsyncSession = Depends(get_async_session),
):
    """
    Unified incidents endpoint for Dashboard.

    Aggregates incidents from multiple sources:
    - Sentry issues (errors/exceptions)
    - Predictions health alerts
    - Scheduler jobs health
    - FastPath/LLM health
    - API budget warnings

    Query params:
    - status: active|acknowledged|resolved (multi-select)
    - severity: info|warning|critical (multi-select)
    - type: sentry|predictions|scheduler|llm|api_budget
    - q: search substring in title/description
    - page: pagination (default 1)
    - limit: page size (default 50, max 100)

    Response:
    {
        "generated_at": "2026-01-23T...",
        "cached": true,
        "cache_age_seconds": 12,
        "data": {
            "incidents": [...],
            "total": 15,
            "page": 1,
            "limit": 50,
            "pages": 1
        }
    }

    Auth: X-Dashboard-Token header required.
    """
    if not _verify_dashboard_token(request):
        raise HTTPException(status_code=401, detail="Dashboard access requires valid token.")

    now = time.time()
    now_iso = datetime.utcnow().isoformat() + "Z"

    # Check cache
    if (
        _incidents_cache["data"] is not None
        and (now - _incidents_cache["timestamp"]) < _incidents_cache["ttl"]
    ):
        all_incidents = _incidents_cache["data"]
        cached = True
        cache_age = round(now - _incidents_cache["timestamp"], 1)
    else:
        # Fetch fresh data
        try:
            all_incidents = await _aggregate_incidents(session)
            _incidents_cache["data"] = all_incidents
            _incidents_cache["timestamp"] = now
            cached = False
            cache_age = 0
        except Exception as e:
            logger.error(f"Failed to aggregate incidents: {e}")
            all_incidents = []
            cached = False
            cache_age = 0

    # Apply filters
    filtered = all_incidents

    # Filter by status (multi-select)
    if status:
        valid_statuses = {"active", "acknowledged", "resolved"}
        status_filter = set(s.lower() for s in status if s.lower() in valid_statuses)
        if status_filter:
            filtered = [i for i in filtered if i["status"] in status_filter]

    # Filter by severity (multi-select)
    if severity:
        valid_severities = {"info", "warning", "critical"}
        severity_filter = set(s.lower() for s in severity if s.lower() in valid_severities)
        if severity_filter:
            filtered = [i for i in filtered if i["severity"] in severity_filter]

    # Filter by type
    if type:
        filtered = [i for i in filtered if i["type"] == type]

    # Filter by search query (substring in title or description)
    if q:
        q_lower = q.lower()
        filtered = [
            i for i in filtered
            if q_lower in i["title"].lower() or q_lower in i["description"].lower()
        ]

    # Pagination
    total = len(filtered)
    pages = max(1, (total + limit - 1) // limit)
    page = min(page, pages)  # Clamp to valid range
    start = (page - 1) * limit
    end = start + limit
    paginated = filtered[start:end]

    return {
        "generated_at": now_iso,
        "cached": cached,
        "cache_age_seconds": cache_age,
        "data": {
            "incidents": paginated,
            "total": total,
            "page": page,
            "limit": limit,
            "pages": pages,
        },
    }
