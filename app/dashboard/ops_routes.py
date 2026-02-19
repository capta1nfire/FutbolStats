"""Ops Dashboard API — debug, triggers, login, alerts, incidents.

46 endpoints across /dashboard/ops/*, /ops/*, /debug/*, /dashboard/incidents*.
Auth patterns: dashboard_token, debug_token, alerts_webhook, Depends, public+rate-limit.
Extracted from main.py Step 4a.
"""

import asyncio
import hashlib
import json
import logging
import os
import time
from collections import deque
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from pydantic import BaseModel, Field
from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.database import AsyncSessionLocal, get_async_session, get_pool_status
from app.etl.sota_constants import SOFASCORE_SUPPORTED_LEAGUES, UNDERSTAT_SUPPORTED_LEAGUES
from app.models import OpsAlert
from app.security import (
    limiter,
    verify_dashboard_token_bool,
    verify_debug_token,
    _has_valid_ops_session,
)
from app.state import ml_engine, _live_summary_cache

router = APIRouter(tags=["ops"])

logger = logging.getLogger(__name__)
settings = get_settings()

# Constants for ops log defaults (read from env, same as main.py log buffer)
OPS_LOG_DEFAULT_LIMIT = int(os.environ.get("OPS_LOG_DEFAULT_LIMIT", "200"))
OPS_LOG_DEFAULT_SINCE_MINUTES = int(os.environ.get("OPS_LOG_DEFAULT_SINCE_MINUTES", "1440"))  # 24h


def _check_token(request: Request):
    """Verify dashboard token (header + session + query param dev)."""
    if not verify_dashboard_token_bool(request):
        raise HTTPException(status_code=401, detail="Dashboard access requires valid token.")


# =============================================================================
# OPS DASHBOARD (DB-backed, cached)
# =============================================================================

_ops_dashboard_cache = {
    "data": None,
    "timestamp": 0,
    "ttl": 45,  # seconds
    # Refresh state (avoid doing heavy DB work inside HTTP requests)
    "refreshing": False,
    "last_refresh_reason": None,
    "last_refresh_error": None,
    "last_refresh_started_at": None,   # epoch seconds
    "last_refresh_finished_at": None,  # epoch seconds
    "last_refresh_duration_ms": None,
    # Backoff (prevent tight retry loops if DB is unhealthy)
    "refresh_failures": 0,
    "next_refresh_after": 0,  # epoch seconds
}

# Best-effort handle for the latest refresh task (debug/visibility only)
_ops_dashboard_refresh_task = None

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

    # NOTE (2026-02): Keep /dashboard/ops.json cheap.
    # Consolidate multiple counts into a single statement to reduce
    # "Consecutive DB Queries" noise and DB round-trips.
    quarantined_odds_24h = 0
    tainted_matches_24h = 0
    unmapped_entities_24h = 0
    odds_desync_6h = 0
    odds_desync_90m = 0

    try:
        res = await session.execute(
            text("""
                SELECT
                  -- 1) Quarantined odds in last 24h
                  (SELECT COUNT(*)
                     FROM odds_history
                    WHERE quarantined = true
                      AND recorded_at > NOW() - INTERVAL '24 hours'
                  ) AS quarantined_odds_24h,

                  -- 2) Tainted matches (recent)
                  (SELECT COUNT(*)
                     FROM matches
                    WHERE tainted = true
                      AND date > NOW() - INTERVAL '7 days'
                  ) AS tainted_matches_7d,

                  -- 3) Unmapped entities (teams without logo)
                  (SELECT COUNT(DISTINCT t.id)
                     FROM teams t
                    WHERE t.logo_url IS NULL
                  ) AS unmapped_entities,

                  -- 4a) Odds desync 6h window (early warning)
                  (SELECT COUNT(DISTINCT m.id)
                     FROM matches m
                     JOIN odds_snapshots os ON os.match_id = m.id
                    WHERE m.status = 'NS'
                      AND m.date BETWEEN NOW() AND NOW() + INTERVAL '6 hours'
                      AND os.odds_freshness = 'live'
                      AND os.snapshot_type = 'lineup_confirmed'
                      AND os.snapshot_at >= NOW() - INTERVAL '120 minutes'
                      AND (m.odds_home IS NULL OR m.odds_draw IS NULL OR m.odds_away IS NULL)
                  ) AS odds_desync_6h,

                  -- 4b) Odds desync 90m window (near kickoff, critical)
                  (SELECT COUNT(DISTINCT m.id)
                     FROM matches m
                     JOIN odds_snapshots os ON os.match_id = m.id
                    WHERE m.status = 'NS'
                      AND m.date BETWEEN NOW() AND NOW() + INTERVAL '90 minutes'
                      AND os.odds_freshness = 'live'
                      AND os.snapshot_type = 'lineup_confirmed'
                      AND os.snapshot_at >= NOW() - INTERVAL '120 minutes'
                      AND (m.odds_home IS NULL OR m.odds_draw IS NULL OR m.odds_away IS NULL)
                  ) AS odds_desync_90m
            """)
        )
        row = res.first()
        if row:
            quarantined_odds_24h = int(row[0] or 0)
            tainted_matches_24h = int(row[1] or 0)
            unmapped_entities_24h = int(row[2] or 0)
            odds_desync_6h = int(row[3] or 0)
            odds_desync_90m = int(row[4] or 0)
    except Exception:
        # Fail-soft: keep ops dashboard alive if a table/column is missing.
        # (The older multi-query approach was more granular; this is the minimal-safe fallback.)
        pass

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


async def _refresh_ops_dashboard_cache(reason: str = "unknown") -> None:
    """Refresh ops dashboard cache in background (fail-soft)."""
    start_ts = time.time()
    _ops_dashboard_cache["refreshing"] = True
    _ops_dashboard_cache["last_refresh_reason"] = reason
    _ops_dashboard_cache["last_refresh_started_at"] = start_ts
    _ops_dashboard_cache["last_refresh_error"] = None

    try:
        data = await _load_ops_data()
        _ops_dashboard_cache["data"] = data
        _ops_dashboard_cache["timestamp"] = time.time()
        _ops_dashboard_cache["refresh_failures"] = 0
        _ops_dashboard_cache["next_refresh_after"] = 0
    except Exception as e:
        # Backoff: exponential up to 5 minutes
        failures = int(_ops_dashboard_cache.get("refresh_failures") or 0) + 1
        _ops_dashboard_cache["refresh_failures"] = failures
        backoff_seconds = min(300, 2 ** min(failures, 8))
        _ops_dashboard_cache["next_refresh_after"] = time.time() + backoff_seconds
        _ops_dashboard_cache["last_refresh_error"] = f"{type(e).__name__}: {e}"
        logger.warning(
            f"[OPS_DASHBOARD] Cache refresh failed ({reason}) ({type(e).__name__}): {e!r}",
            exc_info=True,
        )
    finally:
        end_ts = time.time()
        _ops_dashboard_cache["last_refresh_finished_at"] = end_ts
        _ops_dashboard_cache["last_refresh_duration_ms"] = int((end_ts - start_ts) * 1000)
        _ops_dashboard_cache["refreshing"] = False


def _schedule_ops_dashboard_cache_refresh(reason: str = "stale") -> None:
    """
    Schedule a cache refresh without inheriting the current request context.

    Why: Sentry performance tracing can attribute async child tasks to the
    current HTTP transaction (contextvars propagation), re-triggering the
    "Consecutive DB Queries" alert even if we don't await the refresh.
    """
    import asyncio
    import contextvars

    global _ops_dashboard_refresh_task

    # Avoid duplicate refreshes
    if _ops_dashboard_cache.get("refreshing"):
        return

    # Respect backoff window after failures
    now = time.time()
    next_after = float(_ops_dashboard_cache.get("next_refresh_after") or 0)
    if next_after and now < next_after:
        return

    # Mark refreshing early to prevent thundering herd scheduling
    _ops_dashboard_cache["refreshing"] = True

    try:
        # Run task creation in a fresh (empty) Context to detach request scope.
        ctx = contextvars.Context()
        _ops_dashboard_refresh_task = ctx.run(
            asyncio.create_task,
            _refresh_ops_dashboard_cache(reason=reason),
        )
    except Exception as e:
        _ops_dashboard_cache["refreshing"] = False
        _ops_dashboard_cache["last_refresh_error"] = f"schedule_failed: {type(e).__name__}: {e}"
        logger.warning(f"[OPS_DASHBOARD] Could not schedule cache refresh: {e!r}")


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
            # Sanity check (P0 ATI/ADA): detect overconfidence in last 24h
            "sanity": report.get("sanity"),
        }
    except Exception as e:
        logger.warning(f"Sensor B summary failed: {e}")
        return {
            "state": "ERROR",
            "reason": str(e)[:100],
        }


async def _calculate_extc_shadow_summary(session) -> dict:
    """
    Calculate ext-C shadow model summary for ops dashboard.

    ext-C is an experimental model evaluation job that generates
    shadow predictions for the v1.0.2-ext-C model variant.

    States:
    - DISABLED: EXTC_SHADOW_ENABLED=false
    - ACTIVE: Job is enabled and running
    - ERROR: Issue with job execution
    """
    from app.config import get_settings
    from app.telemetry.metrics import (
        EXTC_SHADOW_INSERTED,
        EXTC_SHADOW_ERRORS,
        EXTC_SHADOW_LAST_SUCCESS,
    )

    extc_settings = get_settings()

    if not extc_settings.EXTC_SHADOW_ENABLED:
        return {
            "state": "DISABLED",
            "reason": "EXTC_SHADOW_ENABLED=false",
            "model_version": extc_settings.EXTC_SHADOW_MODEL_VERSION,
        }

    try:
        # Get metrics from Prometheus
        inserted_total = 0
        errors_total = 0
        last_success_ts = None

        try:
            inserted_total = int(EXTC_SHADOW_INSERTED._value.get() or 0)
        except Exception:
            pass

        try:
            errors_total = int(EXTC_SHADOW_ERRORS._value.get() or 0)
        except Exception:
            pass

        try:
            ts = EXTC_SHADOW_LAST_SUCCESS._value.get()
            if ts and ts > 0:
                last_success_ts = datetime.utcfromtimestamp(ts).isoformat() + "Z"
        except Exception:
            pass

        # Get count from predictions_experiments table
        predictions_count = 0
        try:
            result = await session.execute(text("""
                SELECT COUNT(*) as n
                FROM predictions_experiments
                WHERE model_version = :model_version
            """), {"model_version": extc_settings.EXTC_SHADOW_MODEL_VERSION})
            row = result.first()
            predictions_count = int(row[0]) if row else 0
        except Exception as e:
            logger.debug(f"[EXTC_SHADOW] Count query failed: {e}")

        return {
            "state": "ACTIVE",
            "model_version": extc_settings.EXTC_SHADOW_MODEL_VERSION,
            "model_path": extc_settings.EXTC_SHADOW_MODEL_PATH,
            "interval_minutes": extc_settings.EXTC_SHADOW_INTERVAL_MINUTES,
            "batch_size": extc_settings.EXTC_SHADOW_BATCH_SIZE,
            "oos_only": extc_settings.EXTC_SHADOW_OOS_ONLY,
            "start_at": extc_settings.EXTC_SHADOW_START_AT,
            "predictions_count": predictions_count,
            "inserted_total": inserted_total,
            "errors_total": errors_total,
            "last_success_at": last_success_ts,
        }

    except Exception as e:
        logger.warning(f"ext-C shadow summary failed: {e}")
        return {
            "state": "ERROR",
            "reason": str(e)[:100],
            "model_version": extc_settings.EXTC_SHADOW_MODEL_VERSION,
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
        # Idle override: if no pending work, job is healthy regardless of timer
        if ft_pending == 0:
            stats_health["status"] = "ok"
            stats_health["note"] = "idle_no_pending"
        else:
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

    # Helper: compute stable incident_id for a job (same hash as _make_incident_id)
    # Uses "jobs:" prefix — canonical: id = md5("jobs:<job_key>") first 15 hex
    def _job_incident_id(job_key: str) -> int:
        import hashlib
        h = hashlib.md5(f"jobs:{job_key}".encode()).hexdigest()
        return int(h[:15], 16)

    # Add incident_id to each job for deep-linking from dashboard
    for _jk, _jd in [("stats_backfill", stats_health), ("odds_sync", odds_health), ("fastpath", fastpath_health)]:
        _jd["incident_id"] = _job_incident_id(_jk)

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
                    "incident_id": _job_incident_id(job_key),
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
                "incident_id": worst["incident_id"],
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
    - Wikidata enrichment: stadium + city (team_wikidata_enrichment)
    - Managers: current DT coverage (team_manager_history)

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
    understat_league_ids = ",".join(str(lid) for lid in UNDERSTAT_SUPPORTED_LEAGUES)
    try:
        res = await session.execute(
            text(f"""
                SELECT
                    COUNT(*) FILTER (WHERE mut.match_id IS NOT NULL) AS with_xg,
                    COUNT(*) AS total_ft
                FROM matches m
                LEFT JOIN match_understat_team mut ON m.id = mut.match_id
                WHERE m.status IN ('FT', 'AET', 'PEN')
                  AND m.date >= NOW() - INTERVAL '14 days'
                  AND m.league_id IN ({understat_league_ids})
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

    # 6) Wikidata enrichment: stadium + city coverage for active league teams
    try:
        res = await session.execute(
            text("""
                SELECT
                    COUNT(DISTINCT t.id) AS total_active,
                    COUNT(DISTINCT CASE WHEN t.wikidata_id IS NOT NULL THEN t.id END) AS with_qid,
                    COUNT(DISTINCT CASE WHEN twe.stadium_name IS NOT NULL THEN t.id END) AS with_stadium,
                    COUNT(DISTINCT CASE WHEN twe.admin_location_label IS NOT NULL THEN t.id END) AS with_city,
                    MAX(twe.fetched_at) AS latest_fetch,
                    MIN(twe.fetched_at) AS oldest_fetch
                FROM teams t
                JOIN matches m ON (m.home_team_id = t.id OR m.away_team_id = t.id)
                JOIN admin_leagues al ON al.league_id = m.league_id
                LEFT JOIN team_wikidata_enrichment twe ON twe.team_id = t.id
                WHERE m.date >= NOW() - INTERVAL '30 days'
                  AND al.kind = 'league' AND al.is_active = true
            """)
        )
        row = res.first()
        total = int(row.total_active or 0) if row else 0
        with_qid = int(row.with_qid or 0) if row else 0
        with_stadium = int(row.with_stadium or 0) if row else 0
        with_city = int(row.with_city or 0) if row else 0
        stadium_pct = round(with_stadium / total * 100, 1) if total > 0 else 0.0
        city_pct = round(with_city / total * 100, 1) if total > 0 else 0.0
        qid_pct = round(with_qid / total * 100, 1) if total > 0 else 0.0

        staleness_hours = None
        if row and row.oldest_fetch:
            staleness_hours = round((now - row.oldest_fetch).total_seconds() / 3600, 1)

        worst_pct = min(stadium_pct, city_pct)
        result["wikidata_enrichment"] = {
            "status": "ok" if worst_pct >= 95 else ("warn" if worst_pct >= 80 else "red"),
            "coverage_pct": worst_pct,
            "total": total,
            "with_data": min(with_stadium, with_city),
            "staleness_hours": staleness_hours,
            "note": f"Active league teams (30d). Stadium: {with_stadium}/{total} ({stadium_pct}%), City: {with_city}/{total} ({city_pct}%), QID: {with_qid}/{total} ({qid_pct}%)",
        }
    except Exception as e:
        logger.debug(f"[SOTA] Wikidata enrichment metrics unavailable: {e}")
        result["wikidata_enrichment"] = _unavailable(f"Query failed: {str(e)[:50]}")

    # 7) Managers: coverage for active league teams with current manager
    try:
        res = await session.execute(
            text("""
                SELECT
                    COUNT(DISTINCT t.id) AS total_active,
                    COUNT(DISTINCT CASE WHEN tmh.team_id IS NOT NULL THEN t.id END) AS with_manager,
                    MAX(tmh.detected_at) AS latest_detection
                FROM teams t
                JOIN matches m ON (m.home_team_id = t.id OR m.away_team_id = t.id)
                JOIN admin_leagues al ON al.league_id = m.league_id
                LEFT JOIN team_manager_history tmh ON tmh.team_id = t.id AND tmh.end_date IS NULL
                WHERE m.date >= NOW() - INTERVAL '30 days'
                  AND al.kind = 'league' AND al.is_active = true
            """)
        )
        row = res.first()
        total = int(row.total_active or 0) if row else 0
        with_manager = int(row.with_manager or 0) if row else 0
        manager_pct = round(with_manager / total * 100, 1) if total > 0 else 0.0

        staleness_hours = None
        if row and row.latest_detection:
            staleness_hours = round((now - row.latest_detection).total_seconds() / 3600, 1)

        result["managers"] = {
            "status": "ok" if manager_pct >= 90 else ("warn" if manager_pct >= 70 else "red"),
            "coverage_pct": manager_pct,
            "total": total,
            "with_data": with_manager,
            "staleness_hours": staleness_hours,
            "note": f"Active league teams with current manager (end_date IS NULL)",
        }
    except Exception as e:
        logger.debug(f"[SOTA] Manager metrics unavailable: {e}")
        result["managers"] = _unavailable(f"Query failed: {str(e)[:50]}")

    # Overall status: worst of components (excluding unavailable)
    component_statuses = []
    for key in ["understat", "weather", "venue_geo", "team_profiles", "sofascore_xi", "wikidata_enrichment", "managers"]:
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

    PERF: Single CTE query replaces 8 sequential queries (fixes N+1 pattern).
    """
    now = datetime.utcnow()

    # Single consolidated query using CTEs (8 queries → 1)
    res = await session.execute(
        text("""
            WITH
              pred_stats AS (
                SELECT
                  MAX(created_at) as last_pred_at,
                  COUNT(*) FILTER (WHERE created_at > NOW() - INTERVAL '24 hours') as preds_last_24h,
                  COUNT(*) FILTER (WHERE created_at::date = CURRENT_DATE) as preds_today
                FROM predictions
              ),
              ft_with_pred AS (
                SELECT
                  m.id,
                  p.id as pred_id
                FROM matches m
                LEFT JOIN predictions p ON p.match_id = m.id
                WHERE m.status IN ('FT', 'AET', 'PEN')
                  AND m.date > NOW() - INTERVAL '48 hours'
              ),
              ft_stats AS (
                SELECT
                  COUNT(*) as ft_48h,
                  COUNT(*) FILTER (WHERE pred_id IS NULL) as ft_48h_missing
                FROM ft_with_pred
              ),
              ns_with_pred AS (
                SELECT
                  m.id,
                  m.date,
                  p.id as pred_id
                FROM matches m
                LEFT JOIN predictions p ON p.match_id = m.id
                WHERE m.status = 'NS'
                  AND m.date > NOW()
              ),
              ns_stats AS (
                SELECT
                  COUNT(*) FILTER (WHERE date <= NOW() + INTERVAL '48 hours') as ns_next_48h,
                  COUNT(*) FILTER (WHERE date <= NOW() + INTERVAL '48 hours' AND pred_id IS NULL) as ns_next_48h_missing,
                  MIN(date) as next_ns_date
                FROM ns_with_pred
              )
            SELECT
              pred_stats.last_pred_at,
              pred_stats.preds_last_24h,
              pred_stats.preds_today,
              ft_stats.ft_48h,
              ft_stats.ft_48h_missing,
              ns_stats.ns_next_48h,
              ns_stats.ns_next_48h_missing,
              ns_stats.next_ns_date
            FROM pred_stats, ft_stats, ns_stats
        """)
    )
    row = res.fetchone()

    # Extract values from single row result
    last_pred_at = row[0]
    preds_last_24h = int(row[1] or 0)
    preds_today = int(row[2] or 0)
    ft_48h = int(row[3] or 0)
    ft_48h_missing = int(row[4] or 0)
    ns_next_48h = int(row[5] or 0)
    ns_next_48h_missing = int(row[6] or 0)
    next_ns_date = row[7]

    # Coverage percentages
    coverage_48h_pct = 0.0
    if ft_48h > 0:
        coverage_48h_pct = round(((ft_48h - ft_48h_missing) / ft_48h) * 100, 1)

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


async def _calculate_cascade_ab_test(session) -> dict:
    """
    A/B comparison: Cascade vs Daily predictions (Phase 2, P2-15).

    Compares CLV distribution between predictions that were updated post-lineup
    (cascade) vs those that remained pre-lineup (daily batch only).

    IMPORTANT: `asof_timestamp` is a PIT anchor and is populated for *all*
    predictions (GDT #1). Therefore, cascade-vs-daily must NOT be inferred
    from NULL-ness of `asof_timestamp`.

    Cascade prediction identifiers (DB-first):
    - `match_lineups.lineup_detected_at` exists AND `predictions.asof_timestamp >= lineup_detected_at`
      (i.e., prediction was saved after we detected the lineup).
    - Otherwise, treat as "daily" (pre-lineup or cascade not executed).
    """
    # CLV comparison: cascade vs daily
    result = await session.execute(text("""
        WITH pred_type AS (
            SELECT
                p.id AS prediction_id,
                p.match_id,
                p.asof_timestamp,
                ml.lineup_detected_at,
                CASE
                    WHEN ml.lineup_detected_at IS NOT NULL AND p.asof_timestamp >= ml.lineup_detected_at
                        THEN 'cascade'
                    ELSE 'daily'
                END AS pred_source
            FROM predictions p
            JOIN matches m ON m.id = p.match_id
            -- match_lineups has 2 rows per match (home/away). Join ONLY home row to avoid duplicates.
            LEFT JOIN match_lineups ml
                ON ml.match_id = m.id AND ml.team_id = m.home_team_id
            WHERE m.status IN ('FT', 'AET', 'PEN')
              AND p.asof_timestamp IS NOT NULL
        )
        SELECT
            pt.pred_source,
            COUNT(pc.id) AS n_scored,
            ROUND(AVG(pc.clv_home)::numeric, 5) AS mean_clv_home,
            ROUND(AVG(pc.clv_draw)::numeric, 5) AS mean_clv_draw,
            ROUND(AVG(pc.clv_away)::numeric, 5) AS mean_clv_away,
            ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY pc.clv_home)::numeric, 5) AS median_clv_home,
            ROUND(100.0 * COUNT(*) FILTER (WHERE pc.clv_home > 0) / NULLIF(COUNT(*), 0), 1) AS pct_pos_home,
            ROUND(100.0 * COUNT(*) FILTER (WHERE pc.clv_draw > 0) / NULLIF(COUNT(*), 0), 1) AS pct_pos_draw,
            ROUND(100.0 * COUNT(*) FILTER (WHERE pc.clv_away > 0) / NULLIF(COUNT(*), 0), 1) AS pct_pos_away
        FROM pred_type pt
        JOIN prediction_clv pc ON pc.prediction_id = pt.prediction_id
        WHERE pc.clv_home IS NOT NULL
        GROUP BY pt.pred_source
        ORDER BY pt.pred_source
    """))
    rows = result.fetchall()

    ab_by_source = {}
    for row in rows:
        ab_by_source[row.pred_source] = {
            "n_scored": row.n_scored,
            "mean_clv": {
                "home": float(row.mean_clv_home),
                "draw": float(row.mean_clv_draw),
                "away": float(row.mean_clv_away),
            },
            "median_clv_home": float(row.median_clv_home),
            "pct_positive": {
                "home": float(row.pct_pos_home),
                "draw": float(row.pct_pos_draw),
                "away": float(row.pct_pos_away),
            },
        }

    # Prediction counts (regardless of CLV availability), classified using the same rule.
    count_result = await session.execute(text("""
        SELECT
            SUM(
                CASE
                    WHEN ml.lineup_detected_at IS NOT NULL AND p.asof_timestamp >= ml.lineup_detected_at
                        THEN 1 ELSE 0
                END
            ) AS n_cascade,
            SUM(
                CASE
                    WHEN NOT (ml.lineup_detected_at IS NOT NULL AND p.asof_timestamp >= ml.lineup_detected_at)
                        THEN 1 ELSE 0
                END
            ) AS n_daily
        FROM predictions p
        JOIN matches m ON m.id = p.match_id
        LEFT JOIN match_lineups ml
            ON ml.match_id = m.id AND ml.team_id = m.home_team_id
        WHERE m.status IN ('FT', 'AET', 'PEN')
          AND p.asof_timestamp IS NOT NULL
    """))
    cr = count_result.fetchone()
    n_cascade = int(cr.n_cascade or 0) if cr else 0
    n_daily = int(cr.n_daily or 0) if cr else 0

    has_data = (
        "cascade" in ab_by_source
        and "daily" in ab_by_source
        and ab_by_source["cascade"]["n_scored"] >= 10
        and ab_by_source["daily"]["n_scored"] >= 10
    )

    return {
        "status": "OK" if has_data else "ACCUMULATING",
        "n_cascade_predictions": n_cascade,
        "n_daily_predictions": n_daily,
        "ab_comparison": ab_by_source if ab_by_source else None,
        "verdict": (
            _compute_ab_verdict(ab_by_source) if has_data
            else (
                "Need >=10 CLV scores in BOTH groups to compare "
                f"(have cascade={ab_by_source.get('cascade', {}).get('n_scored', 0)}, "
                f"daily={ab_by_source.get('daily', {}).get('n_scored', 0)})"
            )
        ),
    }


def _compute_ab_verdict(ab: dict) -> str:
    """Compute A/B verdict from CLV comparison."""
    daily = ab.get("daily", {})
    cascade = ab.get("cascade", {})
    if not daily or not cascade:
        return "Incomplete data for comparison"

    d_mean = daily.get("mean_clv", {})
    c_mean = cascade.get("mean_clv", {})
    if not d_mean or not c_mean:
        return "Missing CLV means"

    # Average CLV across outcomes
    d_avg = (d_mean.get("home", 0) + d_mean.get("draw", 0) + d_mean.get("away", 0)) / 3
    c_avg = (c_mean.get("home", 0) + c_mean.get("draw", 0) + c_mean.get("away", 0)) / 3
    delta = c_avg - d_avg

    if delta > 0.001:
        return f"CASCADE WINS: +{delta:.5f} avg CLV vs daily (N_daily={daily.get('n_scored')}, N_cascade={cascade.get('n_scored')})"
    elif delta < -0.001:
        return f"DAILY WINS: {delta:.5f} avg CLV delta (cascade worse) (N_daily={daily.get('n_scored')}, N_cascade={cascade.get('n_scored')})"
    else:
        return f"NEUTRAL: {delta:.5f} avg CLV delta (within noise) (N_daily={daily.get('n_scored')}, N_cascade={cascade.get('n_scored')})"


async def _calculate_clv_summary(session) -> dict:
    """
    CLV (Closing Line Value) rolling metrics per league (Phase 2, P2-12).

    Returns mean/median CLV per outcome per league, plus %> 0 (favorable CLV rate).
    Only uses scored CLV records from prediction_clv table.
    """
    result = await session.execute(text("""
        SELECT
            m.league_id,
            COUNT(*) as n,
            ROUND(AVG(pc.clv_home)::numeric, 5) as mean_clv_home,
            ROUND(AVG(pc.clv_draw)::numeric, 5) as mean_clv_draw,
            ROUND(AVG(pc.clv_away)::numeric, 5) as mean_clv_away,
            ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY pc.clv_home)::numeric, 5) as median_clv_home,
            ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY pc.clv_draw)::numeric, 5) as median_clv_draw,
            ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY pc.clv_away)::numeric, 5) as median_clv_away,
            ROUND(100.0 * COUNT(*) FILTER (WHERE pc.clv_home > 0) / NULLIF(COUNT(*), 0), 1)
                as pct_positive_home,
            ROUND(100.0 * COUNT(*) FILTER (WHERE pc.clv_draw > 0) / NULLIF(COUNT(*), 0), 1)
                as pct_positive_draw,
            ROUND(100.0 * COUNT(*) FILTER (WHERE pc.clv_away > 0) / NULLIF(COUNT(*), 0), 1)
                as pct_positive_away
        FROM prediction_clv pc
        JOIN matches m ON m.id = pc.match_id
        WHERE pc.clv_home IS NOT NULL
        GROUP BY m.league_id
        ORDER BY n DESC
    """))
    by_league = []
    total_n = 0
    for row in result.fetchall():
        total_n += row.n
        by_league.append({
            "league_id": row.league_id,
            "n": row.n,
            "mean": {"home": float(row.mean_clv_home), "draw": float(row.mean_clv_draw), "away": float(row.mean_clv_away)},
            "median": {"home": float(row.median_clv_home), "draw": float(row.median_clv_draw), "away": float(row.median_clv_away)},
            "pct_positive": {"home": float(row.pct_positive_home), "draw": float(row.pct_positive_draw), "away": float(row.pct_positive_away)},
        })

    # Global summary
    global_result = await session.execute(text("""
        SELECT
            COUNT(*) as n,
            ROUND(AVG(clv_home)::numeric, 5) as mean_h,
            ROUND(AVG(clv_draw)::numeric, 5) as mean_d,
            ROUND(AVG(clv_away)::numeric, 5) as mean_a,
            ROUND(100.0 * COUNT(*) FILTER (WHERE clv_home > 0) / NULLIF(COUNT(*), 0), 1) as pct_h,
            ROUND(100.0 * COUNT(*) FILTER (WHERE clv_draw > 0) / NULLIF(COUNT(*), 0), 1) as pct_d,
            ROUND(100.0 * COUNT(*) FILTER (WHERE clv_away > 0) / NULLIF(COUNT(*), 0), 1) as pct_a
        FROM prediction_clv
        WHERE clv_home IS NOT NULL
    """))
    g = global_result.fetchone()

    return {
        "total_scored": g.n if g else 0,
        "leagues_count": len(by_league),
        "global_mean": {"home": float(g.mean_h), "draw": float(g.mean_d), "away": float(g.mean_a)} if g and g.n > 0 else None,
        "global_pct_positive": {"home": float(g.pct_h), "draw": float(g.pct_d), "away": float(g.pct_a)} if g and g.n > 0 else None,
        "by_league": by_league,
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
        # PERF: Single query for tick stats (2 queries → 1)
        res = await session.execute(
            text("""
                WITH recent_tick AS (
                    SELECT tick_at, selected, refreshed, ready, enqueued, completed, errors, skipped
                    FROM fastpath_ticks
                    ORDER BY tick_at DESC
                    LIMIT 1
                ),
                hour_stats AS (
                    SELECT
                        COUNT(*) as total,
                        COUNT(*) FILTER (WHERE selected > 0 OR enqueued > 0 OR completed > 0) as with_activity
                    FROM fastpath_ticks
                    WHERE tick_at > NOW() - INTERVAL '1 hour'
                )
                SELECT
                    rt.tick_at, rt.selected, rt.refreshed, rt.ready, rt.enqueued, rt.completed, rt.errors, rt.skipped,
                    hs.total, hs.with_activity
                FROM hour_stats hs
                LEFT JOIN recent_tick rt ON true
            """)
        )
        row = res.fetchone()
        if row:
            if row[0]:  # tick_at exists
                last_tick_at = row[0]
                last_tick_result = {
                    "selected": row[1], "refreshed": row[2], "stats_ready": row[3],
                    "enqueued": row[4], "completed": row[5], "errors": row[6], "skipped": row[7]
                }
            ticks_total = row[8] or 0
            ticks_with_activity = row[9] or 0
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


@router.get("/dashboard/ops/fastpath_debug.json")
async def fastpath_debug_endpoint(
    request: Request,
    session: AsyncSession = Depends(get_async_session),
):
    """Debug endpoint to see skipped audits and their reasons."""
    verify_debug_token(request)

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


@router.get("/dashboard/ops/llm_audit/{match_id}.json")
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
    verify_debug_token(request)

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


@router.get("/dashboard/ops/match_data.json")
async def match_data_debug_endpoint(
    request: Request,
    match_id: int = Query(...),
    session: AsyncSession = Depends(get_async_session),
):
    """Debug endpoint to see exact match_data sent to LLM."""
    verify_debug_token(request)

    from app.models import Match, Team, Prediction  # lazy import

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


@router.get("/dashboard/ops/stats_rca.json")
async def stats_rca_endpoint(
    request: Request,
    match_id: int = Query(...),
    session: AsyncSession = Depends(get_async_session),
):
    """
    RCA endpoint: fetch stats from API-Football and show full diagnostic.
    Tests: API response, parsing, persistence.
    """
    verify_debug_token(request)

    from app.etl.api_football import APIFootballProvider
    from app.models import Match  # lazy import

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


@router.get("/dashboard/ops/bulk_stats_backfill.json")
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
    verify_debug_token(request)

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


@router.get("/dashboard/ops/fetch_events.json")
async def fetch_events_endpoint(
    request: Request,
    match_id: int = Query(...),
    session: AsyncSession = Depends(get_async_session),
):
    """
    Fetch events from API-Football for a specific match and persist.
    Used for testing/verification.
    """
    verify_debug_token(request)

    from app.etl.api_football import APIFootballProvider
    from app.llm.fastpath import FastPathService
    from app.models import Match  # lazy import

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


@router.get("/dashboard/ops/audit_metrics.json")
async def audit_metrics_endpoint(
    request: Request,
    session: AsyncSession = Depends(get_async_session),
):
    """
    Audit endpoint: cross-check dashboard metrics with direct DB queries.
    Returns raw query results for manual verification.
    """
    verify_debug_token(request)

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


@router.get("/dashboard/ops/predictions_performance.json")
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
    verify_debug_token(request)

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
_SENTRY_CRITICAL_THRESHOLD_1H = 3  # active_issues_1h >= 3 → critical
_SENTRY_WARNING_THRESHOLD_24H = 1  # active_issues_24h >= 1 → warning


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


# =====================================================================
# League names fallback (module-level for _build_league_name_map)
# =====================================================================
_LEAGUE_NAMES_FALLBACK: dict[int, str] = {
    1: "World Cup", 2: "Champions League", 3: "Europa League",
    4: "Euro", 5: "Nations League", 9: "Copa Am\u00e9rica",
    10: "Friendlies", 11: "Sudamericana", 13: "Libertadores",
    22: "Gold Cup",
    # Legacy: league_id=28 was previously (incorrectly) used for WCQ CONMEBOL in code.
    # In production DB it may contain SAFF Championship fixtures.
    28: "SAFF Championship (legacy)",
    # WC 2026 Qualifiers (correct API-Football league IDs)
    29: "WCQ CAF", 30: "WCQ AFC", 31: "WCQ CONCACAF",
    32: "WCQ UEFA", 33: "WCQ OFC", 34: "WCQ CONMEBOL",
    37: "WCQ Intercontinental Play-offs",
    39: "Premier League", 45: "FA Cup", 61: "Ligue 1",
    71: "Brazil Serie A", 78: "Bundesliga", 88: "Eredivisie",
    94: "Primeira Liga", 128: "Argentina Primera", 135: "Serie A",
    140: "La Liga", 143: "Copa del Rey", 203: "Super Lig",
    239: "Colombia Primera A", 242: "Ecuador Liga Pro",
    250: "Paraguay Primera - Apertura", 252: "Paraguay Primera - Clausura",
    253: "MLS", 262: "Liga MX", 265: "Chile Primera Divisi\u00f3n",
    268: "Uruguay Primera - Apertura", 270: "Uruguay Primera - Clausura",
    281: "Peru Primera Divisi\u00f3n", 299: "Venezuela Primera Divisi\u00f3n",
    344: "Bolivia Primera Divisi\u00f3n", 848: "Conference League",
}


def _build_league_name_map() -> dict[int, str]:
    """Build league_id -> name map from fallback + COMPETITIONS + display_name overlay."""
    from app.etl.competitions import COMPETITIONS
    from app.dashboard.admin import _league_cache

    league_name_by_id = _LEAGUE_NAMES_FALLBACK.copy()
    try:
        for league_id, comp in (COMPETITIONS or {}).items():
            if league_id is not None and comp is not None:
                name = getattr(comp, "name", None)
                if name:
                    league_name_by_id[int(league_id)] = name
    except Exception:
        pass
    # Overlay display_name from admin_leagues (highest priority)
    for lid, entry in _league_cache.items():
        ename = entry.get("effective_name")
        if ename:
            league_name_by_id[lid] = ename
    return league_name_by_id


# =====================================================================
# Parallel helpers for _load_ops_data (each opens its own DB session)
# =====================================================================

async def _fetch_budget_status() -> dict:
    """Fetch API-Football budget status (HTTP call + timezone enrichment)."""
    budget_status: dict = {"status": "unavailable"}
    try:
        from app.etl.api_football import get_api_account_status
        budget_status = await get_api_account_status()
    except Exception as e:
        logger.warning(f"Could not fetch API account status: {e}")
        budget_status = {"status": "unavailable", "error": str(e)}

    try:
        from zoneinfo import ZoneInfo
        tz_name = "America/Los_Angeles"
        reset_hour, reset_minute = 16, 0
        now_utc = datetime.utcnow().replace(tzinfo=ZoneInfo("UTC"))
        now_la = now_utc.astimezone(ZoneInfo(tz_name))
        next_reset_la = now_la.replace(hour=reset_hour, minute=reset_minute, second=0, microsecond=0)
        if next_reset_la <= now_la:
            next_reset_la += timedelta(days=1)
        next_reset_utc = next_reset_la.astimezone(ZoneInfo("UTC"))
        if not isinstance(budget_status, dict):
            budget_status = {"status": "unavailable"}
        budget_status.update({
            "tokens_reset_tz": tz_name,
            "tokens_reset_local_time": f"{reset_hour:02d}:{reset_minute:02d}",
            "tokens_reset_at_la": next_reset_la.isoformat(),
            "tokens_reset_at_utc": next_reset_utc.isoformat(),
            "tokens_reset_note": "Observed daily refresh around 4:00pm America/Los_Angeles",
        })
    except Exception:
        pass
    return budget_status


async def _run_inline_queries() -> dict:
    """Run simple inline DB queries for ops dashboard (single session)."""
    async with AsyncSessionLocal() as session:
        # Tracked leagues (distinct league_id)
        res = await session.execute(text("SELECT COUNT(DISTINCT league_id) FROM matches WHERE league_id IS NOT NULL"))
        tracked_leagues_count = int(res.scalar() or 0)

        # Upcoming matches (next 24h)
        res = await session.execute(
            text("""
                SELECT league_id, COUNT(*) AS upcoming
                FROM matches
                WHERE league_id IS NOT NULL
                  AND date >= NOW()
                  AND date < NOW() + INTERVAL '24 hours'
                GROUP BY league_id
                ORDER BY upcoming DESC
                LIMIT 20
            """)
        )
        upcoming_by_league = [{"league_id": int(r[0]), "upcoming_24h": int(r[1])} for r in res.fetchall()]

        # PIT snapshots (live, lineup_confirmed)
        res = await session.execute(
            text("""
                SELECT COUNT(*)
                FROM odds_snapshots
                WHERE snapshot_type = 'lineup_confirmed'
                  AND odds_freshness = 'live'
                  AND snapshot_at > NOW() - INTERVAL '60 minutes'
            """)
        )
        pit_live_60m = int(res.scalar() or 0)

        res = await session.execute(
            text("""
                SELECT COUNT(*)
                FROM odds_snapshots
                WHERE snapshot_type = 'lineup_confirmed'
                  AND odds_freshness = 'live'
                  AND snapshot_at > NOW() - INTERVAL '24 hours'
            """)
        )
        pit_live_24h = int(res.scalar() or 0)

        # DKO distribution (last 60m)
        res = await session.execute(
            text("""
                SELECT ROUND(delta_to_kickoff_seconds / 60.0) AS min_to_ko, COUNT(*) AS c
                FROM odds_snapshots
                WHERE snapshot_type = 'lineup_confirmed'
                  AND odds_freshness = 'live'
                  AND snapshot_at > NOW() - INTERVAL '60 minutes'
                  AND delta_to_kickoff_seconds IS NOT NULL
                GROUP BY 1
                ORDER BY 1
            """)
        )
        pit_dko_60m = [{"min_to_ko": int(r[0]), "count": int(r[1])} for r in res.fetchall()]

        # Latest PIT snapshots (last 10, any freshness)
        res = await session.execute(
            text("""
                SELECT os.snapshot_at, os.match_id, m.league_id, os.odds_freshness, os.delta_to_kickoff_seconds,
                       os.odds_home, os.odds_draw, os.odds_away, os.bookmaker
                FROM odds_snapshots os
                JOIN matches m ON m.id = os.match_id
                WHERE os.snapshot_type = 'lineup_confirmed'
                ORDER BY os.snapshot_at DESC
                LIMIT 10
            """)
        )
        latest_pit = []
        for r in res.fetchall():
            latest_pit.append({
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
            })

        # Movement snapshots (last 24h)
        lineup_movement_24h = None
        market_movement_24h = None
        try:
            res = await session.execute(
                text("SELECT COUNT(*) FROM lineup_movement_snapshots WHERE captured_at > NOW() - INTERVAL '24 hours'")
            )
            lineup_movement_24h = int(res.scalar() or 0)
        except Exception:
            lineup_movement_24h = None
        try:
            res = await session.execute(
                text("SELECT COUNT(*) FROM market_movement_snapshots WHERE captured_at > NOW() - INTERVAL '24 hours'")
            )
            market_movement_24h = int(res.scalar() or 0)
        except Exception:
            market_movement_24h = None

        # Stats backfill health (last 72h finished matches)
        res = await session.execute(
            text("""
                SELECT
                    COUNT(*) FILTER (WHERE stats IS NOT NULL AND stats::text != '{}' AND stats::text != 'null') AS with_stats,
                    COUNT(*) FILTER (WHERE stats IS NULL OR stats::text = '{}' OR stats::text = 'null') AS missing_stats
                FROM matches
                WHERE status IN ('FT', 'AET', 'PEN')
                  AND COALESCE(finished_at, date) > NOW() - INTERVAL '72 hours'
            """)
        )
        row = res.first()
        stats_with = int(row[0] or 0) if row else 0
        stats_missing = int(row[1] or 0) if row else 0

        # =============================================================
        # PROGRESS METRICS (for re-test / Alpha readiness)
        # =============================================================
        TARGET_PIT_SNAPSHOTS_30D = int(os.environ.get("TARGET_PIT_SNAPSHOTS_30D", "500"))
        TARGET_PIT_BETS_30D = int(os.environ.get("TARGET_PIT_BETS_30D", "500"))
        TARGET_BASELINE_COVERAGE_PCT = int(os.environ.get("TARGET_BASELINE_COVERAGE_PCT", "60"))

        pit_snapshots_30d = 0
        try:
            res = await session.execute(
                text("""
                    SELECT COUNT(*)
                    FROM odds_snapshots
                    WHERE snapshot_type = 'lineup_confirmed'
                      AND odds_freshness = 'live'
                      AND snapshot_at > NOW() - INTERVAL '30 days'
                """)
            )
            pit_snapshots_30d = int(res.scalar() or 0)
        except Exception:
            pit_snapshots_30d = 0

        pit_bets_30d = 0
        try:
            res = await session.execute(
                text("""
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
                """)
            )
            pit_bets_30d = int(res.scalar() or 0)
        except Exception:
            pit_bets_30d = 0

        baseline_coverage_pct = 0
        pit_with_baseline = 0
        pit_total_for_baseline = 0
        try:
            res = await session.execute(
                text("""
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
                """)
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

    return {
        "tracked_leagues_count": tracked_leagues_count,
        "upcoming_by_league": upcoming_by_league,
        "pit_live_60m": pit_live_60m,
        "pit_live_24h": pit_live_24h,
        "pit_dko_60m": pit_dko_60m,
        "latest_pit": latest_pit,
        "lineup_movement_24h": lineup_movement_24h,
        "market_movement_24h": market_movement_24h,
        "stats_with": stats_with,
        "stats_missing": stats_missing,
        "progress_metrics": progress_metrics,
    }


async def _run_llm_cost_queries() -> dict:
    """Calculate LLM cost metrics for ops dashboard."""
    llm_cost_data: dict = {"provider": "gemini", "status": "unavailable"}
    try:
        async with AsyncSessionLocal() as session:
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
    return llm_cost_data


async def _run_coverage_queries(league_name_by_id: dict[int, str]) -> list:
    """Coverage by league (NS matches in next 48h with predictions/odds)."""
    coverage_by_league = []
    try:
        async with AsyncSessionLocal() as session:
            res = await session.execute(
                text("""
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
                """)
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
    return coverage_by_league


async def _load_ops_data() -> dict:
    """
    Ops dashboard: read-only aggregated metrics from DB + in-process state.
    Parallelized with asyncio.gather -- ~16 independent sections run concurrently,
    each with its own DB session. Pool: 10+20=30, uses ~14 concurrent sessions.
    """
    from app.scheduler import get_last_sync_time

    now = datetime.utcnow()
    league_mode = os.environ.get("LEAGUE_MODE", "tracked").strip().lower()
    last_sync = get_last_sync_time()
    league_name_by_id = _build_league_name_map()

    # Fail-soft wrapper: degrade per-section instead of failing entire refresh
    async def _safe(coro, default, label):
        try:
            return await coro
        except Exception as e:
            logger.warning(f"[OPS_DASHBOARD] {label} failed: {type(e).__name__}: {e}")
            return default

    # Helper: run a _calculate_* function with its own DB session (fail-soft)
    async def _calc(fn):
        label = fn.__name__
        try:
            async with AsyncSessionLocal() as s:
                return await fn(s)
        except Exception as e:
            logger.warning(f"[OPS_DASHBOARD] {label} failed: {type(e).__name__}: {e}")
            return {"status": "error", "error": f"{type(e).__name__}: {e}"}

    # Fallback for inline queries (all zeros/empty — dashboard shows "Degraded" per-card)
    _inline_fallback = {
        "tracked_leagues_count": 0, "upcoming_by_league": [],
        "pit_live_60m": 0, "pit_live_24h": 0, "pit_dko_60m": [], "latest_pit": [],
        "lineup_movement_24h": None, "market_movement_24h": None,
        "stats_with": 0, "stats_missing": 0,
        "progress_metrics": {
            "pit_snapshots_30d": 0, "target_pit_snapshots_30d": 500,
            "pit_bets_30d": 0, "target_pit_bets_30d": 500,
            "baseline_coverage_pct": 0, "pit_with_baseline": 0,
            "pit_total_for_baseline": 0, "target_baseline_coverage_pct": 60,
            "ready_for_retest": False,
        },
    }

    # Run all independent sections in parallel (fail-soft per section)
    (
        budget_status,
        sentry_health,
        inline,
        predictions_health,
        fastpath_health,
        model_performance,
        telemetry_data,
        shadow_mode_data,
        sensor_b_data,
        extc_shadow_data,
        rerun_serving_data,
        jobs_health_data,
        sota_enrichment_data,
        titan_data,
        llm_cost_data,
        coverage_by_league,
        clv_data,
        cascade_ab_data,
    ) = await asyncio.gather(
        _fetch_budget_status(),
        _fetch_sentry_health(),
        _safe(_run_inline_queries(), _inline_fallback, "inline_queries"),
        _calc(_calculate_predictions_health),
        _calc(_calculate_fastpath_health),
        _calc(_calculate_model_performance),
        _calc(_calculate_telemetry_summary),
        _calc(_calculate_shadow_mode_summary),
        _calc(_calculate_sensor_b_summary),
        _calc(_calculate_extc_shadow_summary),
        _calc(_calculate_rerun_serving_summary),
        _calc(_calculate_jobs_health_summary),
        _calc(_calculate_sota_enrichment_summary),
        _safe(_calculate_titan_summary(), {"status": "error"}, "titan"),
        _run_llm_cost_queries(),
        _safe(_run_coverage_queries(league_name_by_id), [], "coverage"),
        _calc(_calculate_clv_summary),
        _calc(_calculate_cascade_ab_test),
    )

    # Post-processing: enrich with league names
    for item in inline["upcoming_by_league"]:
        item["league_name"] = league_name_by_id.get(item["league_id"])
    for item in inline["latest_pit"]:
        lid = item.get("league_id")
        if isinstance(lid, int):
            item["league_name"] = league_name_by_id.get(lid)

    # Live summary stats (from main.py cache, lazy imported)
    live_summary_stats = {
        "cache_ttl_seconds": _live_summary_cache["ttl"],
        "cache_timestamp": _live_summary_cache["timestamp"],
        "cache_age_seconds": round(time.time() - _live_summary_cache["timestamp"], 1) if _live_summary_cache["timestamp"] else None,
        "cached_live_matches": len(_live_summary_cache["data"]["matches"]) if _live_summary_cache["data"] else 0,
    }

    # ML model status
    ml_model_info = {
        "loaded": ml_engine.model is not None,
        "version": ml_engine.model_version,
        "source": "file",
        "model_path": str(ml_engine.model_path),
    }
    if ml_engine.model is not None:
        try:
            ml_model_info["n_features"] = ml_engine.model.n_features_in_
        except AttributeError:
            pass

    return {
        "generated_at": now.isoformat(),
        "league_mode": league_mode,
        "tracked_leagues_count": inline["tracked_leagues_count"],
        "last_sync_at": last_sync.isoformat() if last_sync else None,
        "budget": budget_status,
        "sentry": sentry_health,
        "pit": {
            "live_60m": inline["pit_live_60m"],
            "live_24h": inline["pit_live_24h"],
            "delta_to_kickoff_60m": inline["pit_dko_60m"],
            "latest": inline["latest_pit"],
        },
        "movement": {
            "lineup_movement_24h": inline["lineup_movement_24h"],
            "market_movement_24h": inline["market_movement_24h"],
        },
        "stats_backfill": {
            "finished_72h_with_stats": inline["stats_with"],
            "finished_72h_missing_stats": inline["stats_missing"],
        },
        "upcoming": {
            "by_league_24h": inline["upcoming_by_league"],
        },
        "progress": inline["progress_metrics"],
        "predictions_health": predictions_health,
        "fastpath_health": fastpath_health,
        "model_performance": model_performance,
        "telemetry": telemetry_data,
        "llm_cost": llm_cost_data,
        "shadow_mode": shadow_mode_data,
        "sensor_b": sensor_b_data,
        "extc_shadow": extc_shadow_data,
        "rerun_serving": rerun_serving_data,
        "jobs_health": jobs_health_data,
        "sota_enrichment": sota_enrichment_data,
        "titan": titan_data,
        "coverage_by_league": coverage_by_league,
        "clv": clv_data,
        "cascade_ab": cascade_ab_data,
        "ml_model": ml_model_info,
        "live_summary": live_summary_stats,
        "db_pool": get_pool_status(),
        "providers": _get_providers_health(),
    }


async def _get_cached_ops_data(blocking: bool = True) -> dict:
    now = time.time()
    if _ops_dashboard_cache["data"] and (now - _ops_dashboard_cache["timestamp"]) < _ops_dashboard_cache["ttl"]:
        return _ops_dashboard_cache["data"]

    # Non-blocking mode: return stale cache and refresh in background.
    # Used by /dashboard/ops.json to avoid heavy DB work in request path.
    if not blocking:
        if _ops_dashboard_cache["data"] is not None:
            _schedule_ops_dashboard_cache_refresh(reason="stale_nonblocking")
            return _ops_dashboard_cache["data"]
        # Cold start: schedule refresh and return a minimal placeholder
        _schedule_ops_dashboard_cache_refresh(reason="cold_start_nonblocking")
        return {
            "generated_at": datetime.utcnow().isoformat(),
            "status": "warming_cache",
            "note": "OPS cache warming in background (non-blocking). Retry in a few seconds.",
        }

    # Blocking mode (HTML dashboard): refresh cache, but avoid duplicate heavy work
    # if a background refresh is already running.
    if _ops_dashboard_cache.get("refreshing") and _ops_dashboard_cache.get("data") is not None:
        return _ops_dashboard_cache["data"]

    # Respect backoff window after failures: serve stale if we have it.
    next_after = float(_ops_dashboard_cache.get("next_refresh_after") or 0)
    if next_after and now < next_after and _ops_dashboard_cache.get("data") is not None:
        return _ops_dashboard_cache["data"]

    await _refresh_ops_dashboard_cache(reason="blocking_request")
    if _ops_dashboard_cache.get("data") is not None:
        return _ops_dashboard_cache["data"]

    # Extremely rare: still no data (DB down on cold start). Fail-soft.
    return {
        "generated_at": datetime.utcnow().isoformat(),
        "status": "unavailable",
        "note": "Could not load OPS data (DB unavailable).",
    }


async def get_cached_ops_data(blocking: bool = True) -> dict:
    """Exported for dashboard_views_routes (rollup endpoint). Wraps _get_cached_ops_data."""
    return await _get_cached_ops_data(blocking)




@router.get("/dashboard/ops.json")
async def ops_dashboard_json(request: Request):
    if not verify_dashboard_token_bool(request):
        raise HTTPException(status_code=401, detail="Dashboard access requires valid token.")
    # Non-blocking: avoid running the heavy ops query bundle in the request path.
    # Return stale cache (if any) and refresh in background.
    data = await _get_cached_ops_data(blocking=False)
    return {
        "data": data,
        "cache_age_seconds": round(time.time() - _ops_dashboard_cache["timestamp"], 1) if _ops_dashboard_cache["timestamp"] else None,
        "cache_ttl_seconds": _ops_dashboard_cache.get("ttl"),
        "cache_refreshing": bool(_ops_dashboard_cache.get("refreshing")),
        "cache_last_refresh_error": _ops_dashboard_cache.get("last_refresh_error"),
        "cache_last_refresh_duration_ms": _ops_dashboard_cache.get("last_refresh_duration_ms"),
        "cache_last_refresh_finished_at": _ops_dashboard_cache.get("last_refresh_finished_at"),
    }



@router.get("/dashboard/ops/logs.json")
async def ops_dashboard_logs_json(
    request: Request,
    limit: int = OPS_LOG_DEFAULT_LIMIT,
    since_minutes: int = OPS_LOG_DEFAULT_SINCE_MINUTES,
    level: Optional[str] = None,
    mode: Optional[str] = None,
):
    """Filtered in-memory ops logs (copy/paste friendly). Use mode=compact for grouped view."""
    if not verify_dashboard_token_bool(request):
        raise HTTPException(status_code=401, detail="Dashboard access requires valid token.")
    from app.main import _get_ops_logs  # lazy import (P0-11: no top-level app.main)

    compact = mode == "compact"
    return {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "limit": limit,
        "since_minutes": since_minutes,
        "level": level,
        "mode": mode,
        "entries": _get_ops_logs(since_minutes=since_minutes, limit=limit, level=level, compact=compact),
    }




# =============================================================================
# OPS ADMIN ENDPOINTS (manual triggers, protected by token)
# =============================================================================


@router.post("/dashboard/ops/rollup")
async def trigger_ops_rollup(request: Request):
    """
    Manually trigger the daily ops rollup job.

    Protected by dashboard token. Use for testing/validation.
    """
    if not verify_dashboard_token_bool(request):
        raise HTTPException(status_code=401, detail="Dashboard access requires valid token.")

    from app.scheduler import daily_ops_rollup

    result = await daily_ops_rollup()
    return {
        "status": "executed",
        "result": result,
    }


@router.post("/dashboard/ops/odds_sync")
async def trigger_odds_sync(request: Request):
    """
    Manually trigger the odds sync job for upcoming matches.

    Protected by dashboard token. Use for testing/validation or immediate sync.
    """
    if not verify_dashboard_token_bool(request):
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


@router.post("/dashboard/ops/sensor_retrain")
async def trigger_sensor_retrain(request: Request):
    """
    Manually trigger Sensor B retrain job.

    Protected by dashboard token. Use after deploy to force immediate retrain
    instead of waiting for the 6h interval.
    """
    if not verify_dashboard_token_bool(request):
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


@router.post("/dashboard/ops/sensor_eval")
async def trigger_sensor_eval(request: Request):
    """
    Manually trigger Sensor B evaluation job.

    Protected by dashboard token. Evaluates pending FT matches against
    sensor predictions.
    """
    if not verify_dashboard_token_bool(request):
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


@router.post("/dashboard/ops/shadow_eval")
async def trigger_shadow_eval(request: Request):
    """
    Manually trigger Shadow mode evaluation job.

    Protected by dashboard token. Evaluates pending FT matches against
    shadow predictions.
    """
    if not verify_dashboard_token_bool(request):
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


@router.post("/dashboard/ops/stats_backfill")
async def trigger_stats_backfill(request: Request):
    """
    Manually trigger stats backfill job.

    Protected by dashboard token. Use after deploy to force immediate execution
    instead of waiting for the 60min interval.
    """
    if not verify_dashboard_token_bool(request):
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


@router.post("/dashboard/ops/historical_stats_backfill")
async def trigger_historical_stats_backfill(request: Request):
    """
    Trigger historical stats backfill job for matches since 2023-08-01.

    This endpoint calls the scheduler job which:
    - Processes 500 matches per run (configurable via HISTORICAL_STATS_BACKFILL_BATCH_SIZE)
    - Marks matches without stats as {"_no_stats": true} to skip on future runs
    - Auto-advances through all leagues

    Protected by dashboard token.
    """
    if not verify_dashboard_token_bool(request):
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


@router.post("/dashboard/ops/weekly_recalibration")
async def trigger_weekly_recalibration(request: Request):
    """
    Manually trigger the weekly recalibration job.

    This runs: sync → audit → team adjustments → retrain evaluation → snapshot.
    Protected by dashboard token. Typically takes 1-5 minutes.
    """
    if not verify_dashboard_token_bool(request):
        raise HTTPException(status_code=401, detail="Dashboard access requires valid token.")

    from app.scheduler import weekly_recalibration
    from app.state import ml_engine

    start_time = time.time()
    await weekly_recalibration(ml_engine)
    duration_ms = int((time.time() - start_time) * 1000)

    return {
        "status": "executed",
        "duration_ms": duration_ms,
    }


@router.post("/dashboard/ops/match_link")
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
    if not verify_dashboard_token_bool(request):
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


@router.patch("/dashboard/matches/{match_id}/odds")
async def update_match_odds_manual(
    request: Request,
    match_id: int,
    odds_home: float,
    odds_draw: float,
    odds_away: float,
    source: str = "manual_audit",
):
    """
    Manually update 1X2 odds for a match (audit/backfill purposes).

    Use when API-Football doesn't have odds but we need them for tracking.
    Records source for audit trail.

    Args:
        match_id: Internal match ID
        odds_home: Home win odds (decimal, e.g. 2.50)
        odds_draw: Draw odds (decimal, e.g. 3.20)
        odds_away: Away win odds (decimal, e.g. 2.80)
        source: Source of odds (e.g. "manual_audit_bet365", "sportsgambler")
    """
    if not verify_dashboard_token_bool(request):
        raise HTTPException(status_code=401, detail="Dashboard access requires valid token.")

    from sqlalchemy import text
    from app.ops.audit import log_ops_action

    # Validate odds are reasonable (1.01 to 100.0)
    for name, value in [("odds_home", odds_home), ("odds_draw", odds_draw), ("odds_away", odds_away)]:
        if not (1.01 <= value <= 100.0):
            raise HTTPException(status_code=400, detail=f"{name} must be between 1.01 and 100.0")

    start_time = time.time()

    try:
        async with AsyncSessionLocal() as session:
            # Verify match exists
            check = await session.execute(
                text("SELECT id, status FROM matches WHERE id = :mid"),
                {"mid": match_id}
            )
            match = check.fetchone()
            if not match:
                raise HTTPException(status_code=404, detail=f"Match {match_id} not found")

            # Update odds
            await session.execute(
                text("""
                    UPDATE matches
                    SET odds_home = :oh, odds_draw = :od, odds_away = :oa,
                        odds_recorded_at = NOW()
                    WHERE id = :mid
                """),
                {"mid": match_id, "oh": odds_home, "od": odds_draw, "oa": odds_away}
            )
            await session.commit()

        duration_ms = int((time.time() - start_time) * 1000)

        # Audit log
        try:
            async with AsyncSessionLocal() as audit_session:
                await log_ops_action(
                    session=audit_session,
                    request=request,
                    action="manual_odds_update",
                    params={"match_id": match_id, "source": source},
                    result="ok",
                    result_detail={
                        "odds_home": odds_home,
                        "odds_draw": odds_draw,
                        "odds_away": odds_away,
                    },
                    duration_ms=duration_ms,
                )
        except Exception as audit_err:
            logger.warning(f"Failed to log audit for manual_odds_update: {audit_err}")

        return {
            "status": "ok",
            "match_id": match_id,
            "odds": {"home": odds_home, "draw": odds_draw, "away": odds_away},
            "source": source,
            "duration_ms": duration_ms,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update odds for match {match_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/dashboard/ops/stats_refresh")
async def trigger_stats_refresh(request: Request, lookback_hours: int = 48, max_calls: int = 100):
    """
    Manually trigger stats refresh for recently finished matches.

    Unlike stats_backfill, this re-fetches stats for ALL recent FT matches,
    even if they already have stats. Captures late events like red cards.

    Args:
        lookback_hours: Hours to look back (default 48 for manual runs)
        max_calls: Max API calls (default 100)
    """
    if not verify_dashboard_token_bool(request):
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


@router.post("/dashboard/ops/narratives_regenerate")
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
    if not verify_dashboard_token_bool(request):
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


@router.post("/dashboard/ops/predictions_rerun")
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

    if not verify_dashboard_token_bool(request):
        raise HTTPException(status_code=401, detail="Dashboard access requires valid token.")

    # GUARDRAIL: Prevent two_stage/shadow predictions from going to predictions table
    # Shadow/two-stage models must use shadow_predictions table, not predictions.
    # This prevents data inconsistency issues discovered in Model Benchmark Tile.
    forbidden_archs = ['two_stage', 'shadow', 'twostage']
    if any(arch in body.architecture.lower() for arch in forbidden_archs):
        raise HTTPException(
            status_code=400,
            detail=(
                f"Architecture '{body.architecture}' cannot write to predictions table. "
                "Shadow/two-stage predictions must use shadow_predictions table. "
                "Use the shadow mode pipeline instead of this endpoint."
            )
        )

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
            raise HTTPException(status_code=500, detail="Rerun failed. Check server logs for details.")


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


@router.post("/dashboard/ops/predictions_rerun_rollback")
async def predictions_rerun_rollback(request: Request, run_id: str):
    """
    Rollback a prediction rerun by setting is_active=False.

    This doesn't delete data - it just marks the rerun as inactive,
    so the serving logic will prefer baseline predictions.
    """
    import uuid
    from sqlalchemy import text

    if not verify_dashboard_token_bool(request):
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


@router.get("/dashboard/ops/predictions_reruns.json")
async def list_prediction_reruns(request: Request):
    """List all prediction reruns with their status."""
    from sqlalchemy import text

    if not verify_dashboard_token_bool(request):
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


@router.post("/dashboard/ops/progress_snapshot")
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

    if not verify_dashboard_token_bool(request):
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


@router.get("/dashboard/ops/progress_snapshots.json")
async def get_progress_snapshots(request: Request, limit: int = 50):
    """
    Get historical Alpha Progress snapshots for auditing.

    Returns list of snapshots ordered by captured_at DESC (most recent first).
    Protected by dashboard token.
    """
    import json

    if not verify_dashboard_token_bool(request):
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


@router.get("/dashboard/ops/history.json")
async def ops_history_json(request: Request, days: int = 30):
    """JSON endpoint for historical daily KPIs."""
    if not verify_dashboard_token_bool(request):
        raise HTTPException(status_code=401, detail="Dashboard access requires valid token.")

    history = await _get_ops_history(days=days)
    return {
        "days_requested": days,
        "days_available": len(history),
        "history": history,
    }




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
        <h1>Bon Jogo OPS</h1>
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


@router.get("/ops/login")
async def ops_login_page(request: Request, error: str = ""):
    """Display the OPS console login form."""
    from fastapi.responses import HTMLResponse, RedirectResponse

    # If already logged in, redirect to dashboard JSON
    if _has_valid_ops_session(request):
        return RedirectResponse(url="/dashboard/ops.json", status_code=302)

    # Check if login is enabled
    if not settings.OPS_ADMIN_PASSWORD:
        raise HTTPException(
            status_code=503,
            detail="OPS login disabled. Set OPS_ADMIN_PASSWORD env var."
        )

    return HTMLResponse(content=_render_login_page(error))


@router.post("/ops/login")
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

    # Redirect to dashboard JSON
    return RedirectResponse(url="/dashboard/ops.json", status_code=302)


@router.get("/ops/logout")
async def ops_logout(request: Request):
    """Logout from OPS console."""
    from fastapi.responses import RedirectResponse

    # Clear session
    request.session.clear()
    logger.info(f"[OPS_LOGIN] Logout from {request.client.host}")

    return RedirectResponse(url="/ops/login", status_code=302)


@router.get("/dashboard")
async def dashboard_home(request: Request):
    """Unified dashboard entrypoint (redirects to Ops)."""
    from fastapi.responses import RedirectResponse

    if not verify_dashboard_token_bool(request):
        # Redirect to login instead of 401 for better UX
        return RedirectResponse(url="/ops/login", status_code=302)

    # Preserve token query param ONLY in development (not in prod - security risk)
    target = "/dashboard/ops.json"
    if not os.getenv("RAILWAY_PROJECT_ID"):
        token = request.query_params.get("token")
        if token:
            target = f"{target}?token={token}"
    return RedirectResponse(url=target, status_code=307)


# =============================================================================
# DEBUG ENDPOINT: Daily Counts (temporary for ops monitoring)
# =============================================================================


@router.get("/dashboard/ops/daily_counts.json")
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
    if not verify_dashboard_token_bool(request):
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


@router.get("/dashboard/ops/daily_comparison.json")
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

    if not verify_dashboard_token_bool(request):
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






@router.get("/dashboard/ops/team_overrides.json")
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
    if not verify_dashboard_token_bool(request):
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


@router.get("/dashboard/ops/job_runs.json")
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
    if not verify_dashboard_token_bool(request):
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


@router.post("/dashboard/ops/migrate_llm_error_fields")
async def migrate_llm_error_fields(
    request: Request,
    session: AsyncSession = Depends(get_async_session),
):
    """
    One-time migration to add LLM error observability fields.
    Safe to run multiple times (uses IF NOT EXISTS).
    """
    if not verify_dashboard_token_bool(request):
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


@router.post("/dashboard/ops/migrate_fastpath_fields")
async def migrate_fastpath_fields(
    request: Request,
    session: AsyncSession = Depends(get_async_session),
):
    """
    One-time migration to add fast-path tracking fields to matches.
    Safe to run multiple times (uses IF NOT EXISTS).
    """
    if not verify_dashboard_token_bool(request):
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


@router.post("/dashboard/ops/alerts/webhook")
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


@router.get("/dashboard/ops/alerts.json")
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
    if not verify_dashboard_token_bool(request):
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


@router.post("/dashboard/ops/alerts/ack")
async def ops_alerts_ack(
    request: Request,
    session: AsyncSession = Depends(get_async_session),
):
    """
    Mark alerts as read/acknowledged.

    Auth: X-Dashboard-Token.

    Body: { "ids": [1,2,3] } or { "ack_all": true }
    """
    if not verify_dashboard_token_bool(request):
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


@router.post("/ops/migrate-weather-precip-prob", include_in_schema=False)
async def migrate_weather_precip_prob(
    request: Request,
    session: AsyncSession = Depends(get_async_session),
):
    """
    One-time migration to add precipitation probability field to match_weather.
    Also triggers a backfill for upcoming matches.
    """
    _check_token(request)
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


@router.post("/ops/trigger-weather-sync", include_in_schema=False)
async def trigger_weather_sync(
    request: Request,
    hours: int = 48,
    limit: int = 100,
):
    """
    Manually trigger weather forecast capture for upcoming matches.

    Args:
        hours: Lookahead window (default 48h)
        limit: Max matches to process (default 100)

    Returns:
        Stats from the weather capture job.
    """
    _check_token(request)
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


# Environment flag: DEBUG_LOG_ENABLED=true allows logging without token (dev mode ONLY)
# SECURITY: In production (Railway), this flag is IGNORED - auth is always required
_DEBUG_LOG_ENABLED = os.getenv("DEBUG_LOG_ENABLED", "false").lower() == "true"
_IS_PRODUCTION = os.getenv("RAILWAY_PROJECT_ID") is not None


@router.post("/debug/log")
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
        "ts": datetime.utcnow().isoformat() + "Z",
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


@router.get("/debug/log")
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


_RESOLVE_GRACE_MINUTES = 30  # Auto-resolve after 30 min not seen (per ABE guardrail)


def _make_incident_id(source: str, key: str) -> int:
    """Generate stable incident ID from source + key (MD5 first 15 hex → int).

    Uses 15 hex digits (60 bits) to fit safely within PostgreSQL BIGINT
    (max 2^63-1) while making hash collisions practically impossible.
    Previous 8-hex (32-bit) version caused UniqueViolationError on PK
    when two different (source, source_key) pairs collided.
    """
    import hashlib
    h = hashlib.md5(f"{source}:{key}".encode()).hexdigest()
    return int(h[:15], 16)


async def _detect_active_incidents(session) -> list[dict]:
    """
    Detect currently active incidents from all sources.

    Returns list of dicts with: id, source, source_key, severity, type, title,
    description, runbook_url, details.

    This is the "detection" phase only — does NOT persist anything.
    """
    incidents = []
    now = datetime.utcnow()
    now_iso = now.isoformat() + "Z"

    # =========================================================================
    # OPTIMIZATION: Launch HTTP-only sources as tasks
    # =========================================================================
    import asyncio
    sentry_task = asyncio.create_task(_fetch_sentry_health())

    # =========================================================================
    # SOURCE 1: Sentry Issues
    # =========================================================================
    try:
        sentry_data = await sentry_task
        if sentry_data.get("status") != "degraded":
            top_issues = sentry_data.get("top_issues", [])
            for issue in top_issues[:10]:
                title = issue.get("title", "Unknown Sentry Issue")
                level = issue.get("level", "error")
                count = issue.get("count", 0)
                last_seen = issue.get("last_seen")

                severity = "warning"
                if level in ("error", "fatal"):
                    severity = "critical"
                elif level == "warning":
                    severity = "warning"
                else:
                    severity = "info"

                _PERF_KEYWORDS = ("consecutive db", "n+1", "slow db", "slow http",
                                  "large http", "large render", "file io on main")
                if level == "info" and any(k in title.lower() for k in _PERF_KEYWORDS):
                    severity = "warning"

                source_key = title[:50]
                incidents.append({
                    "id": _make_incident_id("sentry", source_key),
                    "source": "sentry",
                    "source_key": source_key,
                    "severity": severity,
                    "type": "sentry",
                    "title": title[:80],
                    "description": f"Sentry: {count} events. Level: {level}."[:200],
                    "runbook_url": None,
                    "details": {"level": level, "count": count, "last_seen": last_seen},
                })
    except Exception as e:
        logger.warning(f"Could not fetch Sentry incidents: {e}")

    # =========================================================================
    # SOURCE 2: Predictions Health
    # =========================================================================
    try:
        pred_health = await _calculate_predictions_health(session)
        status_val = pred_health.get("status", "ok")
        if status_val in ("warn", "warning", "critical"):
            reason = pred_health.get("status_reason", "Predictions health degraded")
            ns_missing = pred_health.get("ns_matches_next_48h_missing_prediction", 0)
            ns_total = pred_health.get("ns_matches_next_48h", 0)
            coverage = pred_health.get("ns_coverage_pct", 100)
            severity = "warning" if status_val == "warn" else status_val

            incidents.append({
                "id": _make_incident_id("predictions", "health"),
                "source": "predictions",
                "source_key": "health",
                "severity": severity,
                "type": "predictions",
                "title": f"Predictions coverage at {coverage}%"[:80],
                "description": f"{reason}. {ns_missing}/{ns_total} NS matches missing predictions."[:200],
                "runbook_url": "docs/OPS_RUNBOOK.md#predictions-health",
                "details": {"coverage_pct": coverage, "ns_missing": ns_missing, "ns_total": ns_total},
            })
    except Exception as e:
        logger.warning(f"Could not check predictions health: {e}")

    # =========================================================================
    # SOURCE 3: Jobs Health
    # =========================================================================
    try:
        jobs_health = await _calculate_jobs_health_summary(session)

        for job_name in ["stats_backfill", "odds_sync", "fastpath"]:
            job_data = jobs_health.get(job_name, {})
            job_status = job_data.get("status", "ok")
            if job_status in ("warn", "warning", "red", "critical"):
                mins_since = job_data.get("minutes_since_success")
                help_url = job_data.get("help_url")

                severity = {
                    "warn": "warning", "warning": "warning",
                    "red": "critical", "critical": "critical",
                }.get(job_status, "warning")
                time_str = f"{int(mins_since)}m" if mins_since and mins_since < 60 else (
                    f"{int(mins_since/60)}h" if mins_since else "unknown"
                )

                job_labels = {
                    "stats_backfill": "Stats Backfill",
                    "odds_sync": "Odds Sync",
                    "fastpath": "Fast-Path Narratives",
                }
                expected_intervals = {
                    "stats_backfill": 120, "odds_sync": 720, "fastpath": 5,
                }
                job_label = job_labels.get(job_name, job_name)
                expected_min = expected_intervals.get(job_name)
                ft_pending = job_data.get("ft_pending")
                backlog_ready = job_data.get("backlog_ready")
                last_success_at = job_data.get("last_success_at")
                data_source = job_data.get("source", "unknown")

                desc_parts = [f"Job '{job_label}' last succeeded {time_str} ago (status: {job_status})."]
                if expected_min:
                    desc_parts.append(f"Expected interval: {expected_min}min.")
                if ft_pending is not None:
                    desc_parts.append(f"FT pending stats: {ft_pending}.")
                if backlog_ready is not None:
                    desc_parts.append(f"Backlog ready: {backlog_ready}.")
                desc_parts.append(f"Source: {data_source}.")

                details = {
                    "job_key": job_name,
                    "job_label": job_label,
                    "status": job_status,
                    "minutes_since_success": mins_since,
                    "expected_interval_min": expected_min,
                    "last_success_at": last_success_at,
                    "source": data_source,
                    "runbook_url": help_url,
                }
                if ft_pending is not None:
                    details["ft_pending"] = ft_pending
                if backlog_ready is not None:
                    details["backlog_ready"] = backlog_ready

                # Canonical: id = md5("jobs:<job_name>"), source="jobs"
                incidents.append({
                    "id": _make_incident_id("jobs", job_name),
                    "source": "jobs",
                    "source_key": job_name,
                    "severity": severity,
                    "type": "scheduler",
                    "title": f"Job '{job_label}' unhealthy"[:80],
                    "description": " ".join(desc_parts)[:300],
                    "runbook_url": help_url,
                    "details": details,
                })
    except Exception as e:
        logger.warning(f"Could not check jobs health: {e}")

    # =========================================================================
    # SOURCE 4: FastPath Health (LLM narratives)
    # =========================================================================
    try:
        fp_health = await _calculate_fastpath_health(session)
        fp_status = fp_health.get("status", "ok")
        if fp_status in ("warn", "warning", "red", "critical"):
            error_rate = fp_health.get("last_60m", {}).get("error_rate_pct", 0)
            in_queue = fp_health.get("last_60m", {}).get("in_queue", 0)
            reason = fp_health.get("status_reason", "Fastpath degraded")
            severity = {"warn": "warning", "warning": "warning", "red": "critical", "critical": "critical"}.get(fp_status, "warning")

            incidents.append({
                "id": _make_incident_id("fastpath", "health"),
                "source": "fastpath",
                "source_key": "health",
                "severity": severity,
                "type": "llm",
                "title": f"Fastpath error rate {error_rate}%"[:80],
                "description": f"{reason}. Queue: {in_queue}."[:200],
                "runbook_url": "docs/OPS_RUNBOOK.md#fastpath-health",
                "details": {"error_rate_pct": error_rate, "in_queue": in_queue},
            })
    except Exception as e:
        logger.warning(f"Could not check fastpath health: {e}")

    # =========================================================================
    # SOURCE 5: API Budget (not yet implemented)
    # =========================================================================
    # TODO: implement _fetch_api_football_budget() to enable this source

    # =========================================================================
    # SOURCE 6: Team Profile Coverage (data quality)
    # Excludes nationals from denominator (source='excluded_national')
    # =========================================================================
    try:
        profile_result = await session.execute(text("""
            WITH active_clubs AS (
                SELECT DISTINCT t.id
                FROM teams t
                JOIN matches m ON t.id = m.home_team_id OR t.id = m.away_team_id
                WHERE t.team_type = 'club'
                  AND t.country IS NOT NULL
                  AND m.date >= NOW() - INTERVAL '30 days'
            )
            SELECT
                COUNT(*) AS total,
                COUNT(thcp.team_id) AS with_profile
            FROM active_clubs ac
            LEFT JOIN team_home_city_profile thcp
              ON ac.id = thcp.team_id
              AND thcp.source != 'excluded_national'
        """))
        profile_row = profile_result.fetchone()
        if profile_row and profile_row[0] > 0:
            total_clubs = profile_row[0]
            with_profile = profile_row[1]
            coverage_pct = round(with_profile / total_clubs * 100, 1)

            if coverage_pct < 80:
                incidents.append({
                    "id": _make_incident_id("data_quality", "team_profile_coverage"),
                    "source": "data_quality",
                    "source_key": "team_profile_coverage",
                    "severity": "warning",
                    "type": "data_quality",
                    "title": f"Team profile coverage {coverage_pct}% (target 80%)"[:80],
                    "description": (
                        f"{total_clubs - with_profile} of {total_clubs} active clubs "
                        f"missing home city profile. Run cascade sync."
                    )[:200],
                    "runbook_url": None,
                    "details": {
                        "coverage_pct": coverage_pct,
                        "total_clubs": total_clubs,
                        "with_profile": with_profile,
                        "missing": total_clubs - with_profile,
                    },
                })
    except Exception as e:
        logger.warning(f"Could not check team profile coverage: {e}")

    return incidents


async def _upsert_incidents(session, detected: list[dict]) -> None:
    """
    Upsert detected incidents into ops_incidents table (batch, set-based).

    Single INSERT...ON CONFLICT DO UPDATE — no Python loops, no N+1 queries.

    - INSERT new incidents with timeline "created" event (built in SQL).
    - UPDATE existing: refresh title/description/details/severity, set last_seen_at.
    - REOPEN resolved incidents that reappear (status → active, timeline "reopened").
    - Does NOT touch user-set acknowledged status (unless reopening from resolved).
    """
    if not detected:
        return

    import json as _json
    now = datetime.utcnow()
    now_iso = now.isoformat() + "Z"

    # Build parallel arrays for unnest (ABE: text[] for JSONB to avoid asyncpg type issues)
    ids: list[int] = []
    sources: list[str] = []
    source_keys: list[str] = []
    severities: list[str] = []
    types: list[str] = []
    titles: list[str] = []
    descriptions: list[str | None] = []
    details_json: list[str | None] = []
    runbook_urls: list[str | None] = []
    titles_short: list[str] = []  # for "created" timeline message

    for inc in detected:
        ids.append(inc["id"])
        sources.append(inc["source"])
        source_keys.append(inc["source_key"])
        severities.append(inc["severity"])
        types.append(inc["type"])
        titles.append(inc["title"])
        descriptions.append(inc.get("description"))
        details_json.append(
            _json.dumps(inc["details"]) if inc.get("details") else None
        )
        runbook_urls.append(inc.get("runbook_url"))
        titles_short.append(inc["title"][:100])

    await session.execute(
        text("""
            WITH data AS (
                SELECT *
                FROM unnest(
                    :ids    ::BIGINT[],
                    :sources ::TEXT[],
                    :source_keys ::TEXT[],
                    :severities  ::TEXT[],
                    :types       ::TEXT[],
                    :titles      ::TEXT[],
                    :descriptions ::TEXT[],
                    :details_json ::TEXT[],
                    :runbook_urls ::TEXT[],
                    :titles_short ::TEXT[]
                ) AS t(id, source, source_key, severity, type, title,
                       description, details_json, runbook_url, title_short)
            )
            INSERT INTO ops_incidents
                (id, source, source_key, severity, status, type, title,
                 description, details, runbook_url, timeline,
                 created_at, last_seen_at, updated_at)
            SELECT
                d.id, d.source, d.source_key, d.severity, 'active', d.type, d.title,
                d.description,
                CASE WHEN d.details_json IS NOT NULL
                     THEN d.details_json::jsonb ELSE NULL END,
                d.runbook_url,
                jsonb_build_array(jsonb_build_object(
                    'ts',      :now_iso ::TEXT,
                    'message', 'Incident detected: ' || d.title_short,
                    'actor',   'system',
                    'action',  'created'
                )),
                :now ::TIMESTAMPTZ, :now ::TIMESTAMPTZ, :now ::TIMESTAMPTZ
            FROM data d
            ON CONFLICT (source, source_key) DO UPDATE SET
                severity     = EXCLUDED.severity,
                title        = EXCLUDED.title,
                description  = EXCLUDED.description,
                details      = EXCLUDED.details,
                runbook_url  = EXCLUDED.runbook_url,
                last_seen_at = EXCLUDED.last_seen_at,
                updated_at   = EXCLUDED.updated_at,
                status = CASE
                    WHEN ops_incidents.status = 'resolved' THEN 'active'
                    ELSE ops_incidents.status
                END,
                resolved_at = CASE
                    WHEN ops_incidents.status = 'resolved' THEN NULL
                    ELSE ops_incidents.resolved_at
                END,
                acknowledged_at = CASE
                    WHEN ops_incidents.status = 'resolved' THEN NULL
                    ELSE ops_incidents.acknowledged_at
                END,
                timeline = CASE
                    WHEN ops_incidents.status = 'resolved' THEN
                        COALESCE(ops_incidents.timeline, '[]'::jsonb)
                        || jsonb_build_array(jsonb_build_object(
                            'ts',      :now_iso ::TEXT,
                            'message', 'Incident reopened (detected again)',
                            'actor',   'system',
                            'action',  'reopened'
                        ))
                    ELSE ops_incidents.timeline
                END
        """),
        {
            "ids": ids,
            "sources": sources,
            "source_keys": source_keys,
            "severities": severities,
            "types": types,
            "titles": titles,
            "descriptions": descriptions,
            "details_json": details_json,
            "runbook_urls": runbook_urls,
            "titles_short": titles_short,
            "now": now,
            "now_iso": now_iso,
        },
    )

    await session.commit()


async def _auto_resolve_stale_incidents(session) -> int:
    """
    Auto-resolve incidents not seen within grace window (single UPDATE, no loops).

    Only resolves active/acknowledged incidents where last_seen_at is older
    than RESOLVE_GRACE_MINUTES. Appends "auto_resolved" timeline event in SQL.

    Returns count of auto-resolved incidents.
    """
    now = datetime.utcnow()
    now_iso = now.isoformat() + "Z"
    grace_cutoff = now - timedelta(minutes=_RESOLVE_GRACE_MINUTES)

    result = await session.execute(
        text("""
            UPDATE ops_incidents
            SET status      = 'resolved',
                resolved_at = :now ::TIMESTAMPTZ,
                updated_at  = :now ::TIMESTAMPTZ,
                timeline    = COALESCE(timeline, '[]'::jsonb)
                              || jsonb_build_array(jsonb_build_object(
                                  'ts',      :now_iso ::TEXT,
                                  'message', :resolve_msg ::TEXT,
                                  'actor',   'system',
                                  'action',  'auto_resolved'
                              ))
            WHERE status IN ('active', 'acknowledged')
              AND last_seen_at < :cutoff ::TIMESTAMPTZ
            RETURNING id
        """),
        {
            "now": now,
            "now_iso": now_iso,
            "resolve_msg": f"Auto-resolved (not seen for {_RESOLVE_GRACE_MINUTES}+ min)",
            "cutoff": grace_cutoff,
        },
    )
    resolved_ids = result.fetchall()

    if resolved_ids:
        await session.commit()

    return len(resolved_ids)


async def _aggregate_incidents(session) -> list[dict]:
    """
    Aggregate incidents from multiple sources, persist to ops_incidents,
    and return the full list from DB.

    Flow:
    1. Detect active incidents from all sources
    2. Upsert into ops_incidents (create/update/reopen)
    3. Auto-resolve stale incidents (grace window)
    4. Read all non-resolved from DB and return as dicts
    """
    # Phase 1: Detect
    detected = await _detect_active_incidents(session)

    # Phase 2: Upsert
    try:
        await _upsert_incidents(session, detected)
    except Exception as e:
        logger.error(f"Failed to upsert incidents: {e}")
        # Rollback and continue with read-only
        await session.rollback()

    # Phase 3: Auto-resolve stale
    try:
        resolved_count = await _auto_resolve_stale_incidents(session)
        if resolved_count > 0:
            logger.info(f"Auto-resolved {resolved_count} stale incidents (grace={_RESOLVE_GRACE_MINUTES}m)")
    except Exception as e:
        logger.warning(f"Failed to auto-resolve incidents: {e}")
        await session.rollback()

    # Phase 4: Read all from DB
    try:
        result = await session.execute(
            text("""
                SELECT id, source, source_key, severity, status, type, title,
                       description, details, runbook_url, timeline,
                       created_at, last_seen_at, acknowledged_at, resolved_at, updated_at
                FROM ops_incidents
                ORDER BY
                    CASE severity WHEN 'critical' THEN 0 WHEN 'warning' THEN 1 ELSE 2 END,
                    created_at DESC
            """)
        )
        rows = result.mappings().all()
        incidents = []

        def _ts_iso(dt) -> str | None:
            """Serialize TIMESTAMPTZ to ISO 8601 with Z suffix (no +00:00 duplication)."""
            if dt is None:
                return None
            s = dt.isoformat()
            # asyncpg returns aware datetimes with +00:00; replace with Z for JS compat
            if s.endswith("+00:00"):
                s = s[:-6] + "Z"
            elif not s.endswith("Z"):
                s += "Z"
            return s

        for row in rows:
            inc = {
                "id": row["id"],
                "severity": row["severity"],
                "status": row["status"],
                "type": row["type"],
                "title": row["title"],
                "description": row["description"] or "",
                "created_at": _ts_iso(row["created_at"]),
                "updated_at": _ts_iso(row["updated_at"]),
                "runbook_url": row["runbook_url"],
                "details": row["details"],
                "timeline": row["timeline"] or [],
                "acknowledged_at": _ts_iso(row["acknowledged_at"]),
                "resolved_at": _ts_iso(row["resolved_at"]),
                "source": row["source"],
                "last_seen_at": _ts_iso(row["last_seen_at"]),
            }
            incidents.append(inc)
        return incidents
    except Exception as e:
        logger.error(f"Failed to read incidents from DB: {e}")
        # Fallback: return detected incidents as dicts (ephemeral, like before)
        return detected


@router.get("/dashboard/incidents.json")
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
    if not verify_dashboard_token_bool(request):
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


@router.patch("/dashboard/incidents/{incident_id}")
async def patch_incident(
    incident_id: int,
    request: Request,
    session: AsyncSession = Depends(get_async_session),
):
    """
    Update incident status (acknowledge/resolve).

    Body: {"status": "acknowledged"|"resolved"}

    - Persists status change to ops_incidents.
    - Sets acknowledged_at / resolved_at timestamps.
    - Appends timeline event with actor="user".

    Auth: X-Dashboard-Token header required.
    """
    if not verify_dashboard_token_bool(request):
        raise HTTPException(status_code=401, detail="Dashboard access requires valid token.")

    import json as _json

    body = await request.json()
    new_status = body.get("status")
    if new_status not in ("acknowledged", "resolved"):
        raise HTTPException(status_code=400, detail="status must be 'acknowledged' or 'resolved'")

    # Fetch current incident
    result = await session.execute(
        text("SELECT id, status, timeline FROM ops_incidents WHERE id = :id"),
        {"id": incident_id},
    )
    row = result.mappings().first()
    if not row:
        raise HTTPException(status_code=404, detail="Incident not found")

    current_status = row["status"]
    if current_status == new_status:
        return {"ok": True, "message": f"Already {new_status}"}

    # Validate transition
    if new_status == "acknowledged" and current_status != "active":
        raise HTTPException(status_code=400, detail="Can only acknowledge active incidents")
    if new_status == "resolved" and current_status == "resolved":
        raise HTTPException(status_code=400, detail="Already resolved")

    now = datetime.utcnow()
    now_iso = now.isoformat() + "Z"

    # Build timeline event
    old_timeline = row["timeline"] or []
    if isinstance(old_timeline, str):
        old_timeline = _json.loads(old_timeline)
    new_timeline = list(old_timeline)
    new_timeline.append({
        "ts": now_iso,
        "message": f"Status changed: {current_status} → {new_status}",
        "actor": "user",
        "action": new_status,
    })

    # Update
    if new_status == "acknowledged":
        await session.execute(
            text("""
                UPDATE ops_incidents
                SET status = 'acknowledged', acknowledged_at = :now,
                    updated_at = :now, timeline = CAST(:timeline AS jsonb)
                WHERE id = :id
            """),
            {"id": incident_id, "now": now, "timeline": _json.dumps(new_timeline)},
        )
    elif new_status == "resolved":
        await session.execute(
            text("""
                UPDATE ops_incidents
                SET status = 'resolved', resolved_at = :now,
                    updated_at = :now, timeline = CAST(:timeline AS jsonb)
                WHERE id = :id
            """),
            {"id": incident_id, "now": now, "timeline": _json.dumps(new_timeline)},
        )

    await session.commit()

    # Invalidate cache
    _incidents_cache["data"] = None
    _incidents_cache["timestamp"] = 0

    return {"ok": True, "status": new_status, "updated_at": now_iso}


# =============================================================================
# SERVING CONFIG RELOAD (SSOT)
# =============================================================================

@router.post("/dashboard/ops/reload-serving-config")
async def reload_serving_config(request: Request):
    """
    Hot-reload league_serving_config from DB into this worker's memory cache.

    Note: Only reloads the current worker. Other workers rely on TTL-based
    lazy reload (60s) or the periodic APScheduler refresh job (5min).

    Protected by dashboard token.
    """
    if not verify_dashboard_token_bool(request):
        raise HTTPException(status_code=401, detail="Dashboard access requires valid token.")

    from app.ml.league_router import load_serving_configs, get_tier_summary
    from app.database import AsyncSessionLocal

    async with AsyncSessionLocal() as session:
        n = await load_serving_configs(session)

    summary = get_tier_summary()
    return {
        "status": "reloaded",
        "configs_loaded": n,
        "tier_summary": summary,
    }


@router.get("/dashboard/league-lab.json")
async def get_league_lab_data(request: Request):
    """
    Auto-Lab Online dashboard data.

    Returns recent runs (30d), per-league summary, and config.
    Protected by dashboard token.
    """
    if not verify_dashboard_token_bool(request):
        raise HTTPException(status_code=401, detail="Dashboard access requires valid token.")

    settings = get_settings()

    async with AsyncSessionLocal() as session:
        # Recent runs (last 30 days)
        recent_rows = (await session.execute(text("""
            SELECT r.id, r.league_id, al.name AS league_name,
                   r.started_at, r.finished_at, r.status, r.mode,
                   r.n_matches_used, r.n_tests_run, r.duration_ms,
                   r.best_test_name, r.best_brier, r.market_brier, r.delta_vs_market,
                   r.error_message
            FROM league_lab_runs r
            JOIN admin_leagues al ON al.league_id = r.league_id
            WHERE r.started_at >= NOW() - INTERVAL '30 days'
            ORDER BY r.started_at DESC
            LIMIT 50
        """))).fetchall()

        recent_runs = []
        for row in recent_rows:
            recent_runs.append({
                "run_id": row[0],
                "league_id": row[1],
                "league_name": row[2],
                "started_at": row[3].isoformat() if row[3] else None,
                "finished_at": row[4].isoformat() if row[4] else None,
                "status": row[5],
                "mode": row[6],
                "n_matches": row[7],
                "n_tests": row[8],
                "duration_ms": row[9],
                "best_test": row[10],
                "best_brier": row[11],
                "market_brier": row[12],
                "delta_vs_market": row[13],
                "error": row[14],
            })

        # Per-league summary (latest completed run per league)
        summary_rows = (await session.execute(text("""
            SELECT DISTINCT ON (r.league_id)
                   r.league_id, al.name AS league_name,
                   r.id AS run_id, r.finished_at, r.status,
                   r.best_test_name, r.best_brier, r.market_brier, r.delta_vs_market,
                   r.n_matches_used, r.n_tests_run
            FROM league_lab_runs r
            JOIN admin_leagues al ON al.league_id = r.league_id
            WHERE r.status = 'completed'
            ORDER BY r.league_id, r.finished_at DESC
        """))).fetchall()

        per_league = []
        for row in summary_rows:
            per_league.append({
                "league_id": row[0],
                "league_name": row[1],
                "last_run_id": row[2],
                "last_run_at": row[3].isoformat() if row[3] else None,
                "best_test": row[5],
                "best_brier": row[6],
                "market_brier": row[7],
                "delta_vs_market": row[8],
                "n_matches": row[9],
                "n_tests": row[10],
            })

    return {
        "auto_lab_enabled": settings.AUTO_LAB_ENABLED,
        "config": {
            "max_per_day": settings.AUTO_LAB_MAX_PER_DAY,
            "timeout_min": settings.AUTO_LAB_TIMEOUT_MIN,
            "cadence_high_days": settings.AUTO_LAB_CADENCE_HIGH_DAYS,
            "cadence_mid_days": settings.AUTO_LAB_CADENCE_MID_DAYS,
            "cadence_low_days": settings.AUTO_LAB_CADENCE_LOW_DAYS,
            "min_matches": settings.AUTO_LAB_MIN_MATCHES_TOTAL,
        },
        "recent_runs": recent_runs,
        "per_league_summary": per_league,
    }

