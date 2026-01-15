"""Sensor B: LogReg L2 calibration diagnostics.

This is an INTERNAL diagnostic tool only. It does NOT affect production picks.
Purpose: Detect if Model A is stale/rigid or if there's no additional signal in features.

GOVERNANCE RULES:
- Sensor B never affects production predictions
- Sensor B is never exposed to iOS or public endpoints
- Actions based on Sensor B require human review
- Minimum N samples required before reporting metrics
"""

import logging
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# Global sensor instance (loaded/trained at runtime)
_sensor_engine: Optional["SensorEngine"] = None
_sensor_last_trained: Optional[datetime] = None


class SensorEngine:
    """
    LogReg L2 sensor for calibration diagnostics.

    Re-trains on sliding window of recent FT matches.
    Used to detect if Model A is stale or if features lack signal.
    """

    # Same features as XGBoostEngine for fair comparison
    FEATURE_COLUMNS = [
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

    # LogReg L2 config (conservative, avoids overfitting on small N)
    LOGREG_CONFIG = {
        "penalty": "l2",
        "C": 0.1,  # Strong regularization
        "solver": "lbfgs",
        "max_iter": 200,
        "class_weight": "balanced",  # Important for draws
        "random_state": 42,
        "multi_class": "multinomial",
    }

    def __init__(self, window_size: int = None):
        """
        Initialize sensor engine.

        Args:
            window_size: Number of recent FT matches to train on.
        """
        self.window_size = window_size or settings.SENSOR_WINDOW_SIZE
        self.model: Optional[LogisticRegression] = None
        self.model_version: Optional[str] = None
        self.trained_at: Optional[datetime] = None
        self.training_samples: int = 0
        self.is_ready: bool = False

    def train(self, df: pd.DataFrame) -> dict:
        """
        Train sensor on recent finished matches.

        Args:
            df: DataFrame with features and outcome labels.
                Must have columns: FEATURE_COLUMNS + 'label' (0=home, 1=draw, 2=away)

        Returns:
            Training result dict with status and metrics.
        """
        if len(df) < settings.SENSOR_MIN_SAMPLES:
            self.is_ready = False
            return {
                "status": "LEARNING",
                "reason": f"Need {settings.SENSOR_MIN_SAMPLES} samples, have {len(df)}",
                "samples": len(df),
            }

        try:
            # Prepare features
            X = self._prepare_features(df)
            y = df["label"].values

            # Train LogReg
            self.model = LogisticRegression(**self.LOGREG_CONFIG)
            self.model.fit(X, y)

            # Update state
            self.training_samples = len(df)
            self.trained_at = datetime.utcnow()
            self.model_version = f"logreg_l2_w{self.window_size}_v1"
            self.is_ready = True

            # Calculate training accuracy (not for evaluation, just sanity check)
            train_acc = self.model.score(X, y)

            logger.info(
                f"[SENSOR] Trained: n={len(df)}, window={self.window_size}, "
                f"train_acc={train_acc:.3f}, version={self.model_version}"
            )

            return {
                "status": "READY",
                "samples": len(df),
                "window_size": self.window_size,
                "train_accuracy": round(train_acc, 4),
                "model_version": self.model_version,
            }

        except Exception as e:
            logger.error(f"[SENSOR] Training failed: {e}")
            self.is_ready = False
            return {
                "status": "ERROR",
                "reason": str(e),
                "samples": len(df),
            }

    def predict_proba(self, df: pd.DataFrame) -> Optional[np.ndarray]:
        """
        Get probability predictions from sensor.

        Args:
            df: DataFrame with features for prediction.

        Returns:
            Array of shape (n_samples, 3) with [home, draw, away] probabilities,
            or None if sensor not ready.
        """
        if not self.is_ready or self.model is None:
            return None

        try:
            X = self._prepare_features(df)
            return self.model.predict_proba(X)
        except Exception as e:
            logger.warning(f"[SENSOR] Prediction failed: {e}")
            return None

    def _prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """Extract and prepare feature matrix from DataFrame."""
        df = df.copy()
        for col in self.FEATURE_COLUMNS:
            if col not in df.columns:
                df[col] = 0
        return df[self.FEATURE_COLUMNS].fillna(0).values


def get_sensor_engine() -> Optional[SensorEngine]:
    """Get the global sensor engine instance."""
    return _sensor_engine


def is_sensor_ready() -> bool:
    """Check if sensor is trained and ready for predictions."""
    return _sensor_engine is not None and _sensor_engine.is_ready


async def retrain_sensor(session: AsyncSession) -> dict:
    """
    Retrain sensor on recent finished matches.

    Called by scheduler job every SENSOR_RETRAIN_INTERVAL_HOURS.

    Args:
        session: Async database session.

    Returns:
        Training result dict.
    """
    global _sensor_engine, _sensor_last_trained

    if not settings.SENSOR_ENABLED:
        return {"status": "DISABLED", "reason": "SENSOR_ENABLED=false"}

    window_size = settings.SENSOR_WINDOW_SIZE

    # Query recent finished matches with features
    # We need to rebuild features from match data
    query = text("""
        SELECT
            m.id as match_id,
            m.home_goals,
            m.away_goals,
            -- We'll compute label in Python
            m.date
        FROM matches m
        WHERE m.status IN ('FT', 'AET', 'PEN')
          AND m.home_goals IS NOT NULL
          AND m.away_goals IS NOT NULL
        ORDER BY m.date DESC
        LIMIT :limit
    """)

    result = await session.execute(query, {"limit": window_size * 2})  # Get extra for safety
    rows = result.fetchall()

    if len(rows) < settings.SENSOR_MIN_SAMPLES:
        logger.info(f"[SENSOR] Not enough samples for training: {len(rows)} < {settings.SENSOR_MIN_SAMPLES}")
        return {
            "status": "LEARNING",
            "reason": f"Need {settings.SENSOR_MIN_SAMPLES} samples, have {len(rows)}",
            "samples": len(rows),
        }

    # Build training data with features
    # For MVP, we'll use a simplified approach: query features from feature engineer
    from app.features import FeatureEngineer

    feature_engineer = FeatureEngineer(session=session)

    # Get match IDs for feature building
    match_ids = [row.match_id for row in rows[:window_size]]

    # Build features for these matches
    try:
        df = await feature_engineer.get_matches_features_by_ids(match_ids)
    except AttributeError:
        # Fallback: get features differently if method doesn't exist
        logger.warning("[SENSOR] get_matches_features_by_ids not available, using alternative")
        # For now, return LEARNING state
        return {
            "status": "LEARNING",
            "reason": "Feature engineering method not available",
            "samples": len(rows),
        }

    if df is None or len(df) < settings.SENSOR_MIN_SAMPLES:
        return {
            "status": "LEARNING",
            "reason": f"Could not build features for enough samples",
            "samples": len(df) if df is not None else 0,
        }

    # Add labels (0=home, 1=draw, 2=away)
    df["label"] = df.apply(
        lambda row: 0 if row.get("home_goals", 0) > row.get("away_goals", 0)
        else (1 if row.get("home_goals", 0) == row.get("away_goals", 0) else 2),
        axis=1
    )

    # Initialize or get sensor
    if _sensor_engine is None:
        _sensor_engine = SensorEngine(window_size=window_size)

    # Train
    result = _sensor_engine.train(df)
    _sensor_last_trained = datetime.utcnow()

    return result


async def log_sensor_prediction(
    session: AsyncSession,
    match_id: int,
    df: pd.DataFrame,
    model_a_probs: np.ndarray,
    model_a_version: str,
) -> Optional[dict]:
    """
    Log Model A vs Sensor B predictions for a match.

    Args:
        session: Async database session.
        match_id: Match ID.
        df: DataFrame with features for the match.
        model_a_probs: Array [home, draw, away] from Model A.
        model_a_version: Version string of Model A.

    Returns:
        Dict with logged prediction info, or None on error.
    """
    if not settings.SENSOR_ENABLED:
        return None

    try:
        # Model A prediction
        a_pick = ["home", "draw", "away"][np.argmax(model_a_probs)]

        # Sensor B prediction (if ready)
        sensor = get_sensor_engine()
        sensor_state = "LEARNING"
        b_probs = None
        b_pick = None
        model_b_version = None

        if sensor is not None and sensor.is_ready:
            b_probs = sensor.predict_proba(df)
            if b_probs is not None:
                b_probs = b_probs[0]  # First row
                b_pick = ["home", "draw", "away"][np.argmax(b_probs)]
                model_b_version = sensor.model_version
                sensor_state = "READY"

        # Insert into sensor_predictions
        insert_query = text("""
            INSERT INTO sensor_predictions (
                match_id, window_size, model_a_version, model_b_version,
                a_home_prob, a_draw_prob, a_away_prob, a_pick,
                b_home_prob, b_draw_prob, b_away_prob, b_pick,
                sensor_state, created_at
            ) VALUES (
                :match_id, :window_size, :model_a_version, :model_b_version,
                :a_home_prob, :a_draw_prob, :a_away_prob, :a_pick,
                :b_home_prob, :b_draw_prob, :b_away_prob, :b_pick,
                :sensor_state, NOW()
            )
            ON CONFLICT (match_id) DO UPDATE SET
                window_size = EXCLUDED.window_size,
                model_a_version = EXCLUDED.model_a_version,
                model_b_version = EXCLUDED.model_b_version,
                a_home_prob = EXCLUDED.a_home_prob,
                a_draw_prob = EXCLUDED.a_draw_prob,
                a_away_prob = EXCLUDED.a_away_prob,
                a_pick = EXCLUDED.a_pick,
                b_home_prob = EXCLUDED.b_home_prob,
                b_draw_prob = EXCLUDED.b_draw_prob,
                b_away_prob = EXCLUDED.b_away_prob,
                b_pick = EXCLUDED.b_pick,
                sensor_state = EXCLUDED.sensor_state,
                created_at = NOW()
        """)

        await session.execute(insert_query, {
            "match_id": match_id,
            "window_size": settings.SENSOR_WINDOW_SIZE,
            "model_a_version": model_a_version,
            "model_b_version": model_b_version,
            "a_home_prob": float(model_a_probs[0]),
            "a_draw_prob": float(model_a_probs[1]),
            "a_away_prob": float(model_a_probs[2]),
            "a_pick": a_pick,
            "b_home_prob": float(b_probs[0]) if b_probs is not None else None,
            "b_draw_prob": float(b_probs[1]) if b_probs is not None else None,
            "b_away_prob": float(b_probs[2]) if b_probs is not None else None,
            "b_pick": b_pick,
            "sensor_state": sensor_state,
        })

        return {
            "match_id": match_id,
            "sensor_state": sensor_state,
            "a_pick": a_pick,
            "b_pick": b_pick,
        }

    except Exception as e:
        logger.warning(f"[SENSOR] Failed to log prediction for match {match_id}: {e}")
        return None


async def evaluate_sensor_predictions(session: AsyncSession) -> dict:
    """
    Evaluate sensor predictions against actual outcomes.

    Called by scheduler job every 30 minutes.

    Args:
        session: Async database session.

    Returns:
        Evaluation result dict.
    """
    if not settings.SENSOR_ENABLED:
        return {"status": "DISABLED"}

    # Find predictions to evaluate
    query = text("""
        SELECT sp.id, sp.match_id,
               sp.a_home_prob, sp.a_draw_prob, sp.a_away_prob, sp.a_pick,
               sp.b_home_prob, sp.b_draw_prob, sp.b_away_prob, sp.b_pick,
               m.home_goals, m.away_goals
        FROM sensor_predictions sp
        JOIN matches m ON sp.match_id = m.id
        WHERE sp.evaluated_at IS NULL
          AND m.status IN ('FT', 'AET', 'PEN')
          AND m.home_goals IS NOT NULL
          AND m.away_goals IS NOT NULL
    """)

    result = await session.execute(query)
    rows = result.fetchall()

    if not rows:
        return {"status": "ok", "evaluated": 0, "message": "No pending sensor predictions"}

    evaluated = 0
    a_correct_count = 0
    b_correct_count = 0

    for row in rows:
        # Determine actual outcome
        if row.home_goals > row.away_goals:
            actual = "home"
            actual_idx = 0
        elif row.home_goals == row.away_goals:
            actual = "draw"
            actual_idx = 1
        else:
            actual = "away"
            actual_idx = 2

        # Correctness
        a_correct = row.a_pick == actual
        b_correct = row.b_pick == actual if row.b_pick else None

        # Brier scores
        a_probs = [row.a_home_prob, row.a_draw_prob, row.a_away_prob]
        actual_onehot = [0, 0, 0]
        actual_onehot[actual_idx] = 1
        a_brier = sum((p - a) ** 2 for p, a in zip(a_probs, actual_onehot))

        b_brier = None
        if row.b_home_prob is not None:
            b_probs = [row.b_home_prob, row.b_draw_prob, row.b_away_prob]
            b_brier = sum((p - a) ** 2 for p, a in zip(b_probs, actual_onehot))

        # Update record
        update_query = text("""
            UPDATE sensor_predictions SET
                actual_outcome = :actual,
                a_correct = :a_correct,
                b_correct = :b_correct,
                a_brier = :a_brier,
                b_brier = :b_brier,
                evaluated_at = NOW()
            WHERE id = :id
        """)

        await session.execute(update_query, {
            "id": row.id,
            "actual": actual,
            "a_correct": a_correct,
            "b_correct": b_correct,
            "a_brier": a_brier,
            "b_brier": b_brier,
        })

        evaluated += 1
        if a_correct:
            a_correct_count += 1
        if b_correct:
            b_correct_count += 1

    await session.commit()

    logger.info(
        f"[SENSOR] Evaluation complete: {evaluated} predictions, "
        f"A correct={a_correct_count}, B correct={b_correct_count}"
    )

    return {
        "status": "ok",
        "evaluated": evaluated,
        "a_correct": a_correct_count,
        "b_correct": b_correct_count,
    }


async def get_sensor_report(session: AsyncSession) -> dict:
    """
    Generate comprehensive sensor evaluation report.

    Returns:
        Report with A vs B comparison metrics and signal score.
    """
    if not settings.SENSOR_ENABLED:
        return {"status": "DISABLED", "reason": "SENSOR_ENABLED=false"}

    window_days = settings.SENSOR_EVAL_WINDOW_DAYS
    min_samples = settings.SENSOR_MIN_SAMPLES

    # Get evaluated predictions in window
    query = text(f"""
        SELECT
            COUNT(*) AS total,
            COUNT(*) FILTER (WHERE a_correct) AS a_correct,
            COUNT(*) FILTER (WHERE b_correct) AS b_correct,
            COUNT(*) FILTER (WHERE b_correct AND NOT a_correct) AS b_wins,
            COUNT(*) FILTER (WHERE a_correct AND NOT b_correct) AS a_wins,
            COUNT(*) FILTER (WHERE a_correct AND b_correct) AS both_correct,
            COUNT(*) FILTER (WHERE NOT a_correct AND NOT b_correct) AS both_wrong,
            AVG(a_brier) AS a_brier_avg,
            AVG(b_brier) FILTER (WHERE b_brier IS NOT NULL) AS b_brier_avg
        FROM sensor_predictions
        WHERE evaluated_at IS NOT NULL
          AND created_at > NOW() - INTERVAL '{window_days} days'
          AND b_home_prob IS NOT NULL
    """)

    result = await session.execute(query)
    row = result.first()

    total = int(row.total or 0)

    if total < min_samples:
        # Get counts for pending
        pending_query = text(f"""
            SELECT COUNT(*) FROM sensor_predictions
            WHERE evaluated_at IS NULL
              AND created_at > NOW() - INTERVAL '{window_days} days'
        """)
        pending_result = await session.execute(pending_query)
        pending = int(pending_result.scalar() or 0)

        return {
            "status": "NO_DATA",
            "reason": f"Need {min_samples} evaluated samples, have {total}",
            "counts": {
                "total": total,
                "pending": pending,
                "evaluated": total,
            },
            "gating": {
                "min_samples_required": min_samples,
                "samples_evaluated": total,
                "window_days": window_days,
            },
            "recommendation": {
                "status": "NO_DATA",
                "reason": f"Need {min_samples} evaluated samples with sensor predictions",
            },
        }

    # Calculate metrics
    a_brier = float(row.a_brier_avg or 0)
    b_brier = float(row.b_brier_avg or 0)
    a_correct = int(row.a_correct or 0)
    b_correct = int(row.b_correct or 0)

    # Signal score: how much better is B at extracting signal?
    # brier_uniform = 2/3 ≈ 0.667 (predicting 33% each)
    brier_uniform = 2/3

    # signal_score = (uniform - B) / (uniform - A)
    # > 1.0 means B extracts more signal than A
    # ≈ 1.0 means B and A similar
    # < 1.0 means B worse (overfitting)
    denom = brier_uniform - a_brier
    if abs(denom) < 0.001:
        signal_score = 1.0  # A is at uniform level, can't compare
    else:
        signal_score = (brier_uniform - b_brier) / denom

    # Determine recommendation using Auditor-approved statuses:
    # LEARNING, TRACKING, SIGNAL_DETECTED, OVERFITTING_SUSPECTED
    signal_go = settings.SENSOR_SIGNAL_SCORE_GO
    signal_noise = settings.SENSOR_SIGNAL_SCORE_NOISE

    if signal_score > signal_go:
        rec_status = "SIGNAL_DETECTED"
        rec_reason = f"B extracts more signal (score={signal_score:.2f}), review Model A"
    elif signal_score < signal_noise:
        rec_status = "OVERFITTING_SUSPECTED"
        rec_reason = f"B worse than A (score={signal_score:.2f}), sensor may be overfitting"
    else:
        rec_status = "TRACKING"
        rec_reason = f"B and A comparable (score={signal_score:.2f}), continue monitoring"

    # Counts
    pending_query = text(f"""
        SELECT COUNT(*) FROM sensor_predictions
        WHERE evaluated_at IS NULL
          AND created_at > NOW() - INTERVAL '{window_days} days'
    """)
    pending_result = await session.execute(pending_query)
    pending = int(pending_result.scalar() or 0)

    return {
        "status": "ok",
        "counts": {
            "total": total + pending,
            "evaluated": total,
            "pending": pending,
        },
        "metrics": {
            "a_accuracy": round(a_correct / total, 4) if total > 0 else 0,
            "b_accuracy": round(b_correct / total, 4) if total > 0 else 0,
            "a_brier": round(a_brier, 4),
            "b_brier": round(b_brier, 4),
            "delta_brier": round(b_brier - a_brier, 4),
            "signal_score": round(signal_score, 3),
        },
        "head_to_head": {
            "b_wins": int(row.b_wins or 0),
            "a_wins": int(row.a_wins or 0),
            "both_correct": int(row.both_correct or 0),
            "both_wrong": int(row.both_wrong or 0),
        },
        "gating": {
            "min_samples_required": min_samples,
            "samples_evaluated": total,
            "window_days": window_days,
            "signal_go_threshold": signal_go,
            "signal_noise_threshold": signal_noise,
        },
        "recommendation": {
            "status": rec_status,
            "reason": rec_reason,
        },
        "sensor_info": {
            "window_size": settings.SENSOR_WINDOW_SIZE,
            "model_version": _sensor_engine.model_version if _sensor_engine else None,
            "last_trained": _sensor_last_trained.isoformat() if _sensor_last_trained else None,
            "is_ready": is_sensor_ready(),
        },
    }
