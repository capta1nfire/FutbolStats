"""Shadow prediction service for A/B model comparison.

FASE 2: Two-stage model shadow evaluation.

Runs experimental model (two-stage) in parallel with baseline,
logs both predictions to ShadowPrediction table for comparison.
Does NOT affect served predictions.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.ml.engine import TwoStageEngine, XGBoostEngine
from app.models import ModelSnapshot, ShadowPrediction

logger = logging.getLogger(__name__)
settings = get_settings()

# Global shadow engine (loaded once at startup if enabled)
_shadow_engine: Optional[TwoStageEngine] = None
_shadow_enabled: bool = False


async def init_shadow_engine(session: AsyncSession) -> bool:
    """
    Initialize shadow engine if MODEL_SHADOW_ARCHITECTURE is configured.

    Called during startup. Loads or trains shadow model.

    Args:
        session: Async database session.

    Returns:
        True if shadow engine initialized, False if disabled/failed.
    """
    global _shadow_engine, _shadow_enabled

    shadow_arch = settings.MODEL_SHADOW_ARCHITECTURE
    if not shadow_arch:
        logger.info("Shadow mode disabled (MODEL_SHADOW_ARCHITECTURE not set)")
        _shadow_enabled = False
        return False

    if shadow_arch != "two_stage":
        logger.warning(f"Unknown shadow architecture: {shadow_arch}")
        _shadow_enabled = False
        return False

    logger.info(f"Initializing shadow engine: {shadow_arch}")

    # Check if we have a stored two-stage model
    result = await session.execute(
        select(ModelSnapshot)
        .where(ModelSnapshot.model_version.like("%twostage%"))
        .order_by(ModelSnapshot.created_at.desc())
        .limit(1)
    )
    snapshot = result.scalar_one_or_none()

    _shadow_engine = TwoStageEngine()

    if snapshot and snapshot.model_blob:
        # Load from DB
        if _shadow_engine.load_from_bytes(snapshot.model_blob):
            logger.info(
                f"Shadow engine loaded from DB: version={snapshot.model_version}, "
                f"brier={snapshot.brier_score:.4f}"
            )
            _shadow_enabled = True
            return True
        else:
            logger.error("Failed to load shadow model from DB")

    # No stored model - need to train (will be done in scheduler or manual trigger)
    logger.warning(
        "No two-stage model found in DB. Shadow mode enabled but model not loaded. "
        "Run training to generate shadow model."
    )
    _shadow_enabled = False
    return False


def is_shadow_enabled() -> bool:
    """Check if shadow mode is enabled and model is loaded."""
    return _shadow_enabled and _shadow_engine is not None and _shadow_engine.is_loaded


def get_shadow_engine() -> Optional[TwoStageEngine]:
    """Get the shadow engine instance."""
    return _shadow_engine


async def log_shadow_prediction(
    session: AsyncSession,
    match_id: int,
    df: pd.DataFrame,
    baseline_engine: XGBoostEngine,
    skip_commit: bool = False,
) -> Optional[int]:
    """
    Generate and log shadow prediction alongside baseline.

    Uses UPSERT (ON CONFLICT DO UPDATE) for idempotency - calling multiple
    times for the same match_id will update the existing record.

    Args:
        session: Async database session.
        match_id: Match ID.
        df: DataFrame with features for prediction.
        baseline_engine: The baseline XGBoostEngine.
        skip_commit: If True, don't commit (caller will batch commit).

    Returns:
        Inserted/updated record id if successful, None if shadow disabled/failed.
    """
    from sqlalchemy import text

    if not is_shadow_enabled():
        return None

    try:
        # Baseline prediction
        baseline_proba = baseline_engine.predict_proba(df)
        baseline_pred = ["home", "draw", "away"][np.argmax(baseline_proba[0])]

        # Shadow prediction (two-stage)
        shadow_proba = _shadow_engine.predict_proba(df)
        shadow_pred = ["home", "draw", "away"][np.argmax(shadow_proba[0])]

        # UPSERT using raw SQL for ON CONFLICT DO UPDATE
        upsert_query = text("""
            INSERT INTO shadow_predictions (
                match_id, baseline_version, baseline_home_prob, baseline_draw_prob, baseline_away_prob,
                baseline_predicted, shadow_version, shadow_architecture, shadow_home_prob,
                shadow_draw_prob, shadow_away_prob, shadow_predicted, created_at
            ) VALUES (
                :match_id, :baseline_version, :baseline_home_prob, :baseline_draw_prob, :baseline_away_prob,
                :baseline_predicted, :shadow_version, :shadow_architecture, :shadow_home_prob,
                :shadow_draw_prob, :shadow_away_prob, :shadow_predicted, NOW()
            )
            ON CONFLICT (match_id) DO UPDATE SET
                baseline_version = EXCLUDED.baseline_version,
                baseline_home_prob = EXCLUDED.baseline_home_prob,
                baseline_draw_prob = EXCLUDED.baseline_draw_prob,
                baseline_away_prob = EXCLUDED.baseline_away_prob,
                baseline_predicted = EXCLUDED.baseline_predicted,
                shadow_version = EXCLUDED.shadow_version,
                shadow_architecture = EXCLUDED.shadow_architecture,
                shadow_home_prob = EXCLUDED.shadow_home_prob,
                shadow_draw_prob = EXCLUDED.shadow_draw_prob,
                shadow_away_prob = EXCLUDED.shadow_away_prob,
                shadow_predicted = EXCLUDED.shadow_predicted,
                created_at = NOW()
            RETURNING id
        """)

        result = await session.execute(upsert_query, {
            "match_id": match_id,
            "baseline_version": baseline_engine.model_version,
            "baseline_home_prob": float(baseline_proba[0][0]),
            "baseline_draw_prob": float(baseline_proba[0][1]),
            "baseline_away_prob": float(baseline_proba[0][2]),
            "baseline_predicted": baseline_pred,
            "shadow_version": _shadow_engine.model_version,
            "shadow_architecture": "two_stage",
            "shadow_home_prob": float(shadow_proba[0][0]),
            "shadow_draw_prob": float(shadow_proba[0][1]),
            "shadow_away_prob": float(shadow_proba[0][2]),
            "shadow_predicted": shadow_pred,
        })

        record_id = result.scalar()

        if not skip_commit:
            await session.commit()

        logger.debug(
            f"Shadow prediction logged: match={match_id}, id={record_id}, "
            f"baseline={baseline_pred}({baseline_proba[0][np.argmax(baseline_proba[0])]:.3f}), "
            f"shadow={shadow_pred}({shadow_proba[0][np.argmax(shadow_proba[0])]:.3f})"
        )

        return int(record_id) if record_id is not None else None

    except Exception as e:
        logger.error(f"Failed to log shadow prediction for match {match_id}: {e}")
        return None


async def evaluate_shadow_outcomes(session: AsyncSession) -> dict:
    """
    Evaluate shadow predictions against actual outcomes.

    Updates ShadowPrediction records with actual_result and metrics.

    Returns:
        Summary statistics with:
        - selected: FT matches found with pending evaluations
        - updated: rows actually updated
        - baseline/shadow accuracy and brier metrics
    """
    from sqlalchemy import and_
    from datetime import datetime

    from app.models import Match

    # Get shadow predictions without outcomes (FT candidates)
    result = await session.execute(
        select(ShadowPrediction, Match)
        .join(Match, ShadowPrediction.match_id == Match.id)
        .where(
            and_(
                ShadowPrediction.actual_result.is_(None),
                Match.status.in_(["FT", "AET", "PEN"]),
                Match.home_goals.isnot(None),
                Match.away_goals.isnot(None),
            )
        )
    )
    records = result.all()
    selected = len(records)

    if not records:
        return {"selected": 0, "updated": 0, "evaluated": 0, "message": "No pending shadow predictions to evaluate"}

    evaluated = 0
    baseline_correct = 0
    shadow_correct = 0
    baseline_brier_sum = 0.0
    shadow_brier_sum = 0.0

    for shadow_pred, match in records:
        # Determine actual result
        if match.home_goals > match.away_goals:
            actual = "home"
            actual_idx = 0
        elif match.home_goals == match.away_goals:
            actual = "draw"
            actual_idx = 1
        else:
            actual = "away"
            actual_idx = 2

        # Compute correctness
        b_correct = shadow_pred.baseline_predicted == actual
        s_correct = shadow_pred.shadow_predicted == actual

        # Compute Brier scores
        baseline_proba = [
            shadow_pred.baseline_home_prob,
            shadow_pred.baseline_draw_prob,
            shadow_pred.baseline_away_prob,
        ]
        shadow_proba = [
            shadow_pred.shadow_home_prob,
            shadow_pred.shadow_draw_prob,
            shadow_pred.shadow_away_prob,
        ]

        # One-hot actual
        actual_onehot = [0, 0, 0]
        actual_onehot[actual_idx] = 1

        b_brier = sum((p - a) ** 2 for p, a in zip(baseline_proba, actual_onehot))
        s_brier = sum((p - a) ** 2 for p, a in zip(shadow_proba, actual_onehot))

        # Update record
        shadow_pred.actual_result = actual
        shadow_pred.baseline_correct = b_correct
        shadow_pred.shadow_correct = s_correct
        shadow_pred.baseline_brier = b_brier
        shadow_pred.shadow_brier = s_brier
        shadow_pred.evaluated_at = datetime.utcnow()

        session.add(shadow_pred)

        # Accumulators
        evaluated += 1
        if b_correct:
            baseline_correct += 1
        if s_correct:
            shadow_correct += 1
        baseline_brier_sum += b_brier
        shadow_brier_sum += s_brier

    await session.commit()

    return {
        "selected": selected,
        "updated": evaluated,
        "evaluated": evaluated,
        "baseline_accuracy": baseline_correct / evaluated if evaluated > 0 else 0,
        "shadow_accuracy": shadow_correct / evaluated if evaluated > 0 else 0,
        "baseline_brier_avg": baseline_brier_sum / evaluated if evaluated > 0 else 0,
        "shadow_brier_avg": shadow_brier_sum / evaluated if evaluated > 0 else 0,
        "delta_accuracy": (shadow_correct - baseline_correct) / evaluated if evaluated > 0 else 0,
        "delta_brier": (shadow_brier_sum - baseline_brier_sum) / evaluated if evaluated > 0 else 0,
    }


async def get_shadow_report(session: AsyncSession) -> dict:
    """
    Generate comprehensive shadow evaluation report.

    Uses SHADOW_WINDOW_DAYS for temporal filtering.
    Uses SHADOW_BRIER_IMPROVEMENT_MIN, SHADOW_ACCURACY_DROP_MAX for thresholds.

    Returns:
        Report with baseline vs shadow comparison metrics.
    """
    from sqlalchemy import func, and_, case, text

    window_days = settings.SHADOW_WINDOW_DAYS
    min_samples = settings.SHADOW_MIN_SAMPLES

    # Get evaluated predictions within window
    result = await session.execute(
        select(
            func.count(ShadowPrediction.id).label("total"),
            func.sum(case((ShadowPrediction.baseline_correct == True, 1), else_=0)).label("baseline_correct"),
            func.sum(case((ShadowPrediction.shadow_correct == True, 1), else_=0)).label("shadow_correct"),
            func.avg(ShadowPrediction.baseline_brier).label("baseline_brier_avg"),
            func.avg(ShadowPrediction.shadow_brier).label("shadow_brier_avg"),
        )
        .where(
            and_(
                ShadowPrediction.actual_result.isnot(None),
                ShadowPrediction.evaluated_at.isnot(None),
                ShadowPrediction.evaluated_at > text(f"NOW() - INTERVAL '{window_days} days'"),
            )
        )
    )
    row = result.one()

    total = row.total or 0
    if total == 0:
        return {"status": "no_data", "message": "No evaluated shadow predictions"}

    baseline_correct = row.baseline_correct or 0
    shadow_correct = row.shadow_correct or 0
    baseline_brier = row.baseline_brier_avg or 0
    shadow_brier = row.shadow_brier_avg or 0

    # Per-outcome breakdown (also within window)
    outcome_stats = {}
    for outcome in ["home", "draw", "away"]:
        out_result = await session.execute(
            select(
                func.count(ShadowPrediction.id).label("total"),
                func.sum(case((ShadowPrediction.baseline_correct == True, 1), else_=0)).label("baseline_correct"),
                func.sum(case((ShadowPrediction.shadow_correct == True, 1), else_=0)).label("shadow_correct"),
            )
            .where(
                and_(
                    ShadowPrediction.actual_result == outcome,
                    ShadowPrediction.evaluated_at.isnot(None),
                    ShadowPrediction.evaluated_at > text(f"NOW() - INTERVAL '{window_days} days'"),
                )
            )
        )
        out_row = out_result.one()
        out_total = out_row.total or 0
        if out_total > 0:
            outcome_stats[outcome] = {
                "total": out_total,
                "baseline_accuracy": (out_row.baseline_correct or 0) / out_total,
                "shadow_accuracy": (out_row.shadow_correct or 0) / out_total,
            }

    return {
        "status": "ok",
        "total_evaluated": total,
        "window_days": window_days,
        "baseline": {
            "version": settings.MODEL_VERSION,
            "accuracy": baseline_correct / total,
            "brier_avg": float(baseline_brier),
        },
        "shadow": {
            "version": "v1.1.0-twostage",
            "architecture": "two_stage",
            "accuracy": shadow_correct / total,
            "brier_avg": float(shadow_brier),
        },
        "delta": {
            "accuracy": (shadow_correct - baseline_correct) / total,
            "brier": float(shadow_brier - baseline_brier),
        },
        "by_outcome": outcome_stats,
        "thresholds": {
            "brier_improvement_min": settings.SHADOW_BRIER_IMPROVEMENT_MIN,
            "accuracy_drop_max": settings.SHADOW_ACCURACY_DROP_MAX,
            "min_samples": min_samples,
        },
        "recommendation": _get_recommendation(
            baseline_correct / total,
            shadow_correct / total,
            float(baseline_brier),
            float(shadow_brier),
            total,
        ),
    }


def _get_recommendation(
    baseline_acc: float,
    shadow_acc: float,
    baseline_brier: float,
    shadow_brier: float,
    n: int,
) -> str:
    """Generate recommendation based on shadow evaluation.

    Uses settings thresholds:
    - SHADOW_BRIER_IMPROVEMENT_MIN: min brier improvement for GO
    - SHADOW_ACCURACY_DROP_MAX: max accuracy drop allowed
    """
    min_samples = settings.SHADOW_MIN_SAMPLES
    if n < min_samples:
        return f"INSUFFICIENT_DATA: Need at least {min_samples} evaluated predictions"

    # Use settings thresholds
    brier_improvement_min = settings.SHADOW_BRIER_IMPROVEMENT_MIN
    accuracy_drop_max = settings.SHADOW_ACCURACY_DROP_MAX

    brier_improved = shadow_brier < baseline_brier - brier_improvement_min
    brier_ok = shadow_brier <= baseline_brier + (brier_improvement_min / 2)  # Allow small regression
    acc_improved = shadow_acc > baseline_acc + accuracy_drop_max
    acc_ok = shadow_acc >= baseline_acc - accuracy_drop_max

    if brier_improved and acc_ok:
        return "GO: Shadow model improves calibration without hurting accuracy"
    elif brier_ok and acc_improved:
        return "GO: Shadow model improves accuracy with acceptable calibration"
    elif brier_ok and acc_ok:
        return "HOLD: Shadow model comparable to baseline, continue monitoring"
    elif not brier_ok:
        return f"NO_GO: Shadow model degrades Brier (delta={shadow_brier - baseline_brier:+.4f})"
    else:
        return f"NO_GO: Shadow model degrades accuracy (delta={shadow_acc - baseline_acc:+.1%})"


async def get_shadow_health_metrics(session: AsyncSession) -> dict:
    """
    Get health metrics for shadow mode telemetry.

    Returns:
        - pending_ft: FT matches with pending shadow evaluations (COUNT DISTINCT match_id)
        - eval_lag_minutes: Minutes since oldest pending prediction (0 if none)
    """
    from sqlalchemy import func, and_
    from datetime import datetime

    from app.models import Match

    # Count DISTINCT pending FT matches (not rows, which could have duplicates)
    result = await session.execute(
        select(
            func.count(func.distinct(ShadowPrediction.match_id)).label("pending"),
            func.min(ShadowPrediction.created_at).label("oldest_created"),
        )
        .join(Match, ShadowPrediction.match_id == Match.id)
        .where(
            and_(
                ShadowPrediction.actual_result.is_(None),
                Match.status.in_(["FT", "AET", "PEN"]),
                Match.home_goals.isnot(None),
                Match.away_goals.isnot(None),
            )
        )
    )
    row = result.one()

    pending_ft = row.pending or 0
    eval_lag_minutes = 0.0

    if row.oldest_created:
        delta = datetime.utcnow() - row.oldest_created
        eval_lag_minutes = delta.total_seconds() / 60.0

    return {
        "pending_ft": pending_ft,
        "eval_lag_minutes": round(eval_lag_minutes, 1),
    }
