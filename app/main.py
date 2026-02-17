"""FastAPI application for FutbolStat MVP.

Infrastructure-only: log buffer, lifespan, middleware, router registration.
All endpoints live in app/routes/ and app/dashboard/ modules.
"""

import logging
import os
from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import Optional

from fastapi import FastAPI
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from sqlalchemy import text

from app.config import get_settings
from app.database import close_db, init_db, AsyncSessionLocal
from app.ml.persistence import load_active_model
from app.scheduler import start_scheduler, stop_scheduler
from app.security import limiter
from app.state import ml_engine
from app.telemetry.sentry import init_sentry
from app.logos.routes import router as logos_router
from app.dashboard.model_benchmark import router as model_benchmark_router
from app.dashboard.benchmark_matrix import router as benchmark_matrix_router
from app.dashboard.football_routes import router as football_routes_router
from app.dashboard.admin_routes import router as admin_routes_router
from app.dashboard.settings_routes import router as settings_routes_router
from app.dashboard.ops_routes import router as ops_routes_router
from app.dashboard.dashboard_views_routes import router as dashboard_views_routes_router
from app.routes.core import router as core_router
from app.routes.api import (
    router as api_router,
    _train_model_background,
    _warmup_standings_cache,
    _predictions_catchup_on_startup,
)

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

    # Initialize Family S engine (Mandato D: Tier 3 MTV model)
    # P1: Load always so flipping LEAGUE_ROUTER_MTV_ENABLED doesn't need redeploy
    async with AsyncSessionLocal() as session:
        from app.ml.family_s import init_family_s_engine
        family_s_ok = await init_family_s_engine(session)
        if family_s_ok:
            logger.info("Family S engine loaded (activates when LEAGUE_ROUTER_MTV_ENABLED=true)")

    # Start background scheduler for weekly sync/train
    start_scheduler(ml_engine)

    # Phase 2: Start Event Bus for lineup cascade
    from app.events import get_event_bus, LINEUP_CONFIRMED
    from app.events.handlers import cascade_handler
    event_bus = get_event_bus()
    event_bus.subscribe(LINEUP_CONFIRMED, cascade_handler)
    await event_bus.start()
    logger.info("Event Bus started (Phase 2 lineup cascade)")

    # Warm up standings cache for active leagues (non-blocking)
    asyncio.create_task(_warmup_standings_cache())

    # Predictions catch-up on startup (P2 resilience)
    # If predictions job hasn't run recently and there are upcoming matches,
    # trigger a catch-up to avoid gaps from deploys interrupting the daily cron
    asyncio.create_task(_predictions_catchup_on_startup())

    yield

    # Shutdown
    logger.info("Shutting down...")
    await get_event_bus().stop()
    stop_scheduler()
    await close_db()



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

# Include routers
app.include_router(core_router)
app.include_router(logos_router)
app.include_router(model_benchmark_router)
app.include_router(benchmark_matrix_router)
app.include_router(football_routes_router)
app.include_router(admin_routes_router)
app.include_router(settings_routes_router)
app.include_router(ops_routes_router)
app.include_router(dashboard_views_routes_router)
app.include_router(api_router)
