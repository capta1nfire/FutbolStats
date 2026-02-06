"""Core routes: health, telemetry, metrics.

Auth per-endpoint (P0-1):
- /health: public, rate limited
- /telemetry: dashboard token
- /metrics: Bearer token
"""

from fastapi import APIRouter, Header, HTTPException, Request
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel

from app.config import get_settings
from app.security import limiter, verify_dashboard_token_bool
from app.state import _telemetry, ml_engine

router = APIRouter(tags=["core"])
settings = get_settings()


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


@router.get("/health", response_model=HealthResponse)
@limiter.limit("120/minute")
async def health_check(request: Request):
    """Health check endpoint."""
    return HealthResponse(
        status="ok",
        model_loaded=ml_engine.is_loaded,
    )


@router.get("/telemetry")
async def get_telemetry(request: Request):
    """
    Aggregated telemetry counters for cache hit/miss monitoring.

    No high-cardinality labels (no match_id, team names, URLs).
    Safe for Prometheus/Grafana scraping.

    Protected by X-Dashboard-Token (same as other dashboard endpoints).

    NOTE: Counters reset on redeploy/restart. This is diagnostic telemetry,
    not historical observability. For persistent metrics, export to Prometheus.
    """
    if not verify_dashboard_token_bool(request):
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


@router.get("/metrics")
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
    # P0-4: Preserve exact Bearer token auth from main.py
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
