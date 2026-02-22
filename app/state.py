"""Shared singletons for FutbolStats application.

Singleton-by-import pattern: main.py and routers import from this module
to share the same instances (ml_engine, telemetry counters).
"""

from app.ml.engine import TwoStageEngine

# Global ML engine (singleton) — V1.1.0 cross-wire: TwoStageEngine as baseline
# TwoStageEngine(XGBoostEngine) inherits predict(), _find_value_bets().
# load_from_bytes() auto-detects blob type (xgb_twostage or xgb_baseline fallback).
ml_engine = TwoStageEngine()

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


# =============================================================================
# LIVE SUMMARY CACHE (shared between main.py and ops_routes.py)
# =============================================================================
# Moved here to eliminate circular lazy import `from app.main import _live_summary_cache`.

_live_summary_cache: dict = {
    "data": None,
    "timestamp": 0.0,
    "ttl": 5.0,  # 5 second TTL
}
