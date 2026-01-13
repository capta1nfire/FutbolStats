"""FastAPI application for FutbolStat MVP."""

import logging
import os
import time
from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
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
from app.models import Match, OddsHistory, PITReport, PostMatchAudit, Prediction, PredictionOutcome, Team, TeamAdjustment
from app.scheduler import start_scheduler, stop_scheduler, get_last_sync_time, get_sync_leagues, SYNC_LEAGUES
from app.security import limiter, verify_api_key

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

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

# Simple in-memory cache for predictions
_predictions_cache = {
    "data": None,
    "timestamp": 0,
    "ttl": 300,  # 5 minutes cache
}

# Standings cache: keyed by (league_id, season), stores standings list
# TTL 30 minutes - standings don't change frequently during a match detail view
_standings_cache = {}  # type: dict  # {(league_id, season): {"data": list, "timestamp": float}}
_STANDINGS_CACHE_TTL = 1800  # 30 minutes


def _get_cached_standings(league_id: int, season: int) -> Optional[list]:
    """Get standings from cache if still valid."""
    key = (league_id, season)
    if key in _standings_cache:
        entry = _standings_cache[key]
        if time.time() - entry["timestamp"] < _STANDINGS_CACHE_TTL:
            return entry["data"]
    return None


def _set_cached_standings(league_id: int, season: int, data: list) -> None:
    """Store standings in cache."""
    key = (league_id, season)
    _standings_cache[key] = {"data": data, "timestamp": time.time()}


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

    # Start background scheduler for weekly sync/train
    start_scheduler(ml_engine)

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


app = FastAPI(
    title="FutbolStat MVP",
    description="Football Prediction System for FIFA World Cup",
    version="1.0.0",
    lifespan=lifespan,
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

    # Parse league IDs
    league_id_list = None
    if league_ids:
        league_id_list = [int(x.strip()) for x in league_ids.split(",")]

    # Get features for upcoming matches
    # iOS progressive loading:
    #   Priority: days_back=1, days_ahead=1 → yesterday/today/tomorrow (~50-100 matches)
    #   Full: days_back=7, days_ahead=7 → 15-day window (~300 matches)
    feature_engineer = FeatureEngineer(session=session)
    df = await feature_engineer.get_upcoming_matches_features(
        league_ids=league_id_list,
        include_recent_days=actual_days_back,  # Past N days for finished matches
        days_ahead=actual_days_ahead,  # Future N days for upcoming matches
    )
    logger.info(f"Predictions query: days_back={actual_days_back}, days_ahead={actual_days_ahead}, matches={len(df)}")

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
        from app.ml.recalibration import RecalibrationEngine, load_team_adjustments

        try:
            # Load team adjustments
            team_adjustments = await load_team_adjustments(session)
            context_metadata["team_adjustments_loaded"] = True

            # Initialize recalibrator for context gathering
            recalibrator = RecalibrationEngine(session)

            # Detect unstable leagues
            drift_result = await recalibrator.detect_league_drift()
            unstable_leagues = {alert["league_id"] for alert in drift_result.get("drift_alerts", [])}
            context_metadata["unstable_leagues"] = len(unstable_leagues)

            # Check odds movements for upcoming matches
            odds_result = await recalibrator.check_all_upcoming_odds_movements(days_ahead=days)
            odds_movements = {
                alert["match_id"]: alert
                for alert in odds_result.get("alerts", [])
            }
            context_metadata["odds_movements_detected"] = len(odds_movements)

            # Build team details for insights generation
            team_details = {}
            adj_query = select(TeamAdjustment)
            adj_result = await session.execute(adj_query)
            for adj in adj_result.scalars().all():
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

            logger.info(
                f"Context loaded: {len(unstable_leagues)} unstable leagues, "
                f"{len(odds_movements)} odds movements"
            )

        except Exception as e:
            logger.warning(f"Error loading context: {e}. Predictions will be made without context.")

    # Make predictions with context
    predictions = ml_engine.predict(df, team_adjustments=team_adjustments, context=context)

    # For finished matches, overlay frozen prediction data if available
    predictions = await _overlay_frozen_predictions(session, predictions)

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
        )
        prediction_items.append(item)

    response = PredictionsResponse(
        predictions=prediction_items,
        model_version=ml_engine.model_version,
        context_applied=context_metadata if with_context else None,
    )

    # Cache the response (only for default 7+7 requests with context)
    # This prevents non-default requests (e.g., days=30) from polluting the cache
    if is_default_full:
        _predictions_cache["data"] = response
        _predictions_cache["timestamp"] = now
        logger.info(f"Cached {len(prediction_items)} predictions (default 7+7)")

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
    # Get match
    match = await session.get(Match, match_id)
    if not match:
        raise HTTPException(status_code=404, detail="Match not found")

    # Get teams
    home_team = await session.get(Team, match.home_team_id)
    away_team = await session.get(Team, match.away_team_id)

    # Get history for both teams
    home_history = await get_team_history(match.home_team_id, limit=5, session=session)
    away_history = await get_team_history(match.away_team_id, limit=5, session=session)

    # Get standings for league (for club leagues only)
    home_position = None
    away_position = None
    home_league_points = None
    away_league_points = None

    # Only fetch standings for club leagues (not national teams)
    if home_team and home_team.team_type == "club" and match.league_id:
        try:
            # Determine season (current year or previous if early in year)
            current_date = match.date or datetime.now()
            season = current_date.year if current_date.month >= 7 else current_date.year - 1

            # Check cache first (avoids ~2s external API call)
            standings = _get_cached_standings(match.league_id, season)
            if standings is None:
                provider = APIFootballProvider()
                standings = await provider.get_standings(match.league_id, season)
                await provider.close()
                _set_cached_standings(match.league_id, season, standings)

            # Find positions for both teams
            for standing in standings:
                if home_team and standing.get("team_id") == home_team.external_id:
                    home_position = standing.get("position")
                    home_league_points = standing.get("points")
                if away_team and standing.get("team_id") == away_team.external_id:
                    away_position = standing.get("position")
                    away_league_points = standing.get("points")
        except Exception as e:
            logger.warning(f"Could not fetch standings: {e}")

    # Get prediction if model is loaded and match not played
    prediction = None
    if ml_engine.is_loaded and match.status == "NS":
        try:
            feature_engineer = FeatureEngineer(session=session)
            features = await feature_engineer.get_match_features(match)
            features["home_team_name"] = home_team.name if home_team else "Unknown"
            features["away_team_name"] = away_team.name if away_team else "Unknown"

            import pandas as pd
            df = pd.DataFrame([features])
            predictions = ml_engine.predict(df)
            prediction = predictions[0] if predictions else None
        except Exception as e:
            logger.error(f"Error getting prediction: {e}")

    return {
        "match": {
            "id": match.id,
            "date": match.date.isoformat() if match.date else None,
            "league_id": match.league_id,
            "status": match.status,
            "home_goals": match.home_goals,
            "away_goals": match.away_goals,
        },
        "home_team": {
            "id": home_team.id if home_team else None,
            "name": home_team.name if home_team else "Unknown",
            "logo": home_team.logo_url if home_team else None,
            "history": home_history["matches"],
            "position": home_position,
            "league_points": home_league_points,
        },
        "away_team": {
            "id": away_team.id if away_team else None,
            "name": away_team.name if away_team else "Unknown",
            "logo": away_team.logo_url if away_team else None,
            "history": away_history["matches"],
            "position": away_position,
            "league_points": away_league_points,
        },
        "prediction": prediction,
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

    # Get goal events from API
    provider = APIFootballProvider()
    try:
        events = await provider.get_fixture_events(match.external_id)
    finally:
        await provider.close()

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

        # Update score
        if goal.get("team_id") == home_external_id:
            # Check for own goal
            if goal.get("detail") == "Own Goal":
                current_away += 1
            else:
                current_home += 1
        elif goal.get("team_id") == away_external_id:
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
                "team": "home" if g.get("team_id") == home_external_id else "away",
                "team_name": g.get("team_name"),
                "player": g.get("player_name"),
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
async def get_league_standings(league_id: int, season: int = None):
    """
    Get full league standings/table for a given league.

    Returns all teams with position, points, matches played, goals, form, etc.
    """
    try:
        provider = APIFootballProvider()

        # Determine season if not provided
        if season is None:
            current_date = datetime.now()
            season = current_date.year if current_date.month >= 7 else current_date.year - 1

        standings = await provider.get_standings(league_id, season)
        await provider.close()

        if not standings:
            raise HTTPException(status_code=404, detail="No standings found for this league")

        return {
            "league_id": league_id,
            "season": season,
            "standings": standings,
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


def _verify_dashboard_token(request: Request) -> bool:
    """Verify dashboard access token."""
    token = settings.DASHBOARD_TOKEN
    if not token:  # Empty token = dashboard disabled
        return False

    # Check header first, then query param
    provided = request.headers.get("X-Dashboard-Token") or request.query_params.get("token")
    return provided == token


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
        ideal_pct = round(ideal_captures / total_live * 100, 1) if total_live > 0 else 0

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
    Protected by X-Dashboard-Token header (preferred) or ?token= query param.

    SECURITY NOTE: Prefer header over query param in production.
    Query params may be logged in server access logs and browser history.
    """
    from fastapi.responses import HTMLResponse

    if not _verify_dashboard_token(request):
        raise HTTPException(
            status_code=401,
            detail="Dashboard access requires valid token. Set DASHBOARD_TOKEN env var and provide via X-Dashboard-Token header or ?token= param.",
        )

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

    diagnostics = {
        "status": "unknown",
        "model_loaded": False,
        "matches_found": 0,
        "ns_matches": 0,
        "predictions_generated": 0,
        "predictions_saved": 0,
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

            # Step 4: Save to database
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
                    diagnostics["errors"].append(f"Match {match_id}: {str(e)[:50]}")

            await session.commit()
            diagnostics["predictions_saved"] = saved
            diagnostics["status"] = "ok" if saved > 0 else "no_new_predictions"

            logger.info(f"Predictions trigger complete: {saved} saved from {len(df_ns)} NS matches")

    except Exception as e:
        diagnostics["status"] = "error"
        diagnostics["errors"].append(str(e))
        logger.error(f"Predictions trigger failed: {e}")

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
                  AND captured_at > NOW() - INTERVAL '24 hours'
            """)
        )
        quarantined_odds_24h = int(res.scalar() or 0)
    except Exception:
        pass  # Table may not exist yet

    # 2) Tainted matches in last 24h (matches marked tainted recently)
    tainted_matches_24h = 0
    try:
        res = await session.execute(
            text("""
                SELECT COUNT(*) FROM matches
                WHERE tainted = true
                  AND updated_at > NOW() - INTERVAL '24 hours'
            """)
        )
        tainted_matches_24h = int(res.scalar() or 0)
    except Exception:
        pass  # Column may not exist yet

    # 3) Unmapped entities (teams without internal mapping, first seen in last 24h)
    # This checks for external teams that were seen but not yet mapped
    unmapped_entities_24h = 0
    try:
        # Check for teams referenced in matches but missing logo_url (proxy for unmapped)
        res = await session.execute(
            text("""
                SELECT COUNT(DISTINCT t.id) FROM teams t
                WHERE t.logo_url IS NULL
                  AND t.created_at > NOW() - INTERVAL '24 hours'
            """)
        )
        unmapped_entities_24h = int(res.scalar() or 0)
    except Exception:
        pass

    # Determine status
    if tainted_matches_24h > 0 or quarantined_odds_24h > 0:
        status = "RED"
    elif unmapped_entities_24h > 0:
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
        },
        "links": links,
    }


async def _calculate_predictions_health(session) -> dict:
    """
    Calculate predictions health metrics for P0 observability.

    Detects when daily_save_predictions isn't running/persisting.
    Returns status: ok/warn/red based on recency and coverage.
    """
    now = datetime.utcnow()

    # 1) Last prediction saved
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

    # Calculate hours since last prediction
    hours_since_last = None
    if last_pred_at:
        delta = now - last_pred_at
        hours_since_last = round(delta.total_seconds() / 3600, 1)

    # Determine status
    # - ok: last_pred <= 24h AND coverage >= 80%
    # - warn: last_pred > 24h OR coverage < 80%
    # - red: last_pred > 48h OR coverage < 50%
    status = "ok"
    status_reason = None

    if hours_since_last is None or hours_since_last > 48:
        status = "red"
        status_reason = f"No predictions in {hours_since_last or 'unknown'}h (>48h threshold)"
    elif coverage_48h_pct < 50:
        status = "red"
        status_reason = f"Coverage {coverage_48h_pct}% < 50% threshold"
    elif hours_since_last > 24:
        status = "warn"
        status_reason = f"No predictions in {hours_since_last}h (>24h threshold)"
    elif coverage_48h_pct < 80:
        status = "warn"
        status_reason = f"Coverage {coverage_48h_pct}% < 80% threshold"

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
                f"ft_48h={ft_48h}, missing={ft_48h_missing}"
            )
        else:
            logger.warning(
                f"[OPS_ALERT] predictions_health=WARN: {status_reason}. "
                f"last_pred={last_pred_at}, preds_24h={preds_last_24h}"
            )

    return {
        "status": status,
        "status_reason": status_reason,
        "last_prediction_saved_at": last_pred_at.isoformat() if last_pred_at else None,
        "hours_since_last_prediction": hours_since_last,
        "predictions_saved_last_24h": preds_last_24h,
        "predictions_saved_today_utc": preds_today,
        "ft_matches_last_48h": ft_48h,
        "ft_matches_last_48h_missing_prediction": ft_48h_missing,
        "coverage_last_48h_pct": coverage_48h_pct,
        "thresholds": {
            "hours_warn": 24,
            "hours_red": 48,
            "coverage_warn_pct": 80,
            "coverage_red_pct": 50,
        },
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
    token: str = Query(...),
    session: AsyncSession = Depends(get_async_session),
):
    """Debug endpoint to see skipped audits and their reasons."""
    if token != settings.DASHBOARD_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid token")

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


@app.get("/dashboard/ops/match_data.json")
async def match_data_debug_endpoint(
    token: str = Query(...),
    match_id: int = Query(...),
    session: AsyncSession = Depends(get_async_session),
):
    """Debug endpoint to see exact match_data sent to LLM."""
    if token != settings.DASHBOARD_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid token")

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
    token: str = Query(...),
    match_id: int = Query(...),
    session: AsyncSession = Depends(get_async_session),
):
    """
    RCA endpoint: fetch stats from API-Football and show full diagnostic.
    Tests: API response, parsing, persistence.
    """
    if token != settings.DASHBOARD_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid token")

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
    token: str = Query(...),
    since_date: str = Query("2026-01-03", description="Start date YYYY-MM-DD"),
    limit: int = Query(50, description="Max matches to process per call"),
    dry_run: bool = Query(True, description="If true, only list matches without fetching"),
    session: AsyncSession = Depends(get_async_session),
):
    """
    Bulk backfill stats for all FT matches since a given date that are missing stats.
    Use dry_run=true first to see how many matches need backfill.
    """
    if token != settings.DASHBOARD_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid token")

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
    token: str = Query(...),
    match_id: int = Query(...),
    session: AsyncSession = Depends(get_async_session),
):
    """
    Fetch events from API-Football for a specific match and persist.
    Used for testing/verification.
    """
    if token != settings.DASHBOARD_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid token")

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
    token: str = Query(...),
    session: AsyncSession = Depends(get_async_session),
):
    """
    Audit endpoint: cross-check dashboard metrics with direct DB queries.
    Returns raw query results for manual verification.
    """
    if token != settings.DASHBOARD_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid token")

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
        # DATA QUALITY TELEMETRY (quarantine/taint/unmapped summary)
        # =============================================================
        telemetry_data = await _calculate_telemetry_summary(session)

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
        "telemetry": telemetry_data,
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


def _render_ops_dashboard_html(data: dict, history: list | None = None) -> str:
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

    html = f"""<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <meta http-equiv="refresh" content="60">
  <title>Ops Dashboard - FutbolStats</title>
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
        Coverage 48h: {pred_health.get("coverage_last_48h_pct", 0)}%
        {f"<br/>Missing FT: {pred_health.get('ft_matches_last_48h_missing_prediction', 0)}/{pred_health.get('ft_matches_last_48h', 0)}" if pred_health.get("status") != "ok" else ""}
        {f"<br/><small style='color:var(--red)'>{pred_health.get('status_reason', '')}</small>" if pred_health.get("status_reason") else ""}
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
      </div>
    </div>
    <div class="card {telemetry_color()}">
      <div class="card-label">Data Quality<span class="info-icon">i<span class="tooltip">Telemetría de calidad de datos. ROJO: Hay odds en cuarentena o partidos tainted (datos no confiables). AMARILLO: Hay entidades sin mapear. VERDE: Todo OK. Los datos cuarentenados/tainted son excluidos del training.</span></span></div>
      <div class="card-value">{telemetry.get("status", "?").upper()}</div>
      <div class="card-sub">
        Quarantine: {telemetry_summary.get("quarantined_odds_24h", 0)} |
        Tainted: {telemetry_summary.get("tainted_matches_24h", 0)} |
        Unmapped: {telemetry_summary.get("unmapped_entities_24h", 0)}
        {f'<br/><a href="{telemetry.get("links", [{}])[0].get("url", "#")}" target="_blank" style="font-size:0.75rem;">Grafana →</a>' if telemetry.get("links") else ""}
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
  </div>

  <div class="footer">
    Refresh: 60s | Cache TTL: {_ops_dashboard_cache["ttl"]}s
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


@app.get("/dashboard/ops")
async def ops_dashboard_html(request: Request):
    """
    Ops Dashboard - Monitoreo en vivo del backend (DB-backed).

    Protegido por X-Dashboard-Token (preferido) o ?token=.
    """
    from fastapi.responses import HTMLResponse

    if not _verify_dashboard_token(request):
        raise HTTPException(status_code=401, detail="Dashboard access requires valid token.")

    data = await _get_cached_ops_data()

    # Fetch KPI history (last 14 days for dashboard display)
    history = await _get_ops_history(days=14)

    html = _render_ops_dashboard_html(data, history=history)
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


@app.get("/dashboard")
async def dashboard_home(request: Request):
    """Unified dashboard entrypoint (redirects to Ops)."""
    from fastapi.responses import RedirectResponse

    if not _verify_dashboard_token(request):
        raise HTTPException(status_code=401, detail="Dashboard access requires valid token.")

    # Preserve token query param if present (convenience). Prefer header in production.
    token = request.query_params.get("token")
    target = "/dashboard/ops"
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

# Environment flag: DEBUG_LOG_ENABLED=true allows logging without token (dev mode)
_DEBUG_LOG_ENABLED = os.getenv("DEBUG_LOG_ENABLED", "false").lower() == "true"


@app.post("/debug/log")
async def debug_log(request: Request):
    """
    Receives performance logs from iOS instrumentation.

    Security:
    - If DEBUG_LOG_ENABLED=true: allow without token (dev/debug mode)
    - Otherwise: require valid X-Dashboard-Token (401 if missing/invalid)

    Rate limit: handled by global rate limiter
    """
    # Auth check (unless debug mode enabled)
    if not _DEBUG_LOG_ENABLED:
        token = request.headers.get("X-Dashboard-Token")
        expected = os.getenv("DASHBOARD_TOKEN", "")
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
