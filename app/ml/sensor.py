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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
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

    # Hard bounds for raw feature values (guardrail against outliers / bad feature rows).
    # These are intentionally wide and only prevent pathological values from dominating logits.
    FEATURE_CLIP_BOUNDS: dict[str, tuple[float, float]] = {
        "home_goals_scored_avg": (0.0, 6.0),
        "home_goals_conceded_avg": (0.0, 6.0),
        "home_shots_avg": (0.0, 40.0),
        "home_corners_avg": (0.0, 30.0),
        "home_rest_days": (0.0, 45.0),
        "home_matches_played": (0.0, 80.0),
        "away_goals_scored_avg": (0.0, 6.0),
        "away_goals_conceded_avg": (0.0, 6.0),
        "away_shots_avg": (0.0, 40.0),
        "away_corners_avg": (0.0, 30.0),
        "away_rest_days": (0.0, 45.0),
        "away_matches_played": (0.0, 80.0),
        "goal_diff_avg": (-6.0, 6.0),
        "rest_diff": (-45.0, 45.0),
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
        # Numerical stability / calibration controls
        self.temperature: float = float(getattr(settings, "SENSOR_TEMPERATURE", 2.0))
        self.prob_eps: float = float(getattr(settings, "SENSOR_PROB_EPS", 1e-12))

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

            # Train LogReg with feature scaling to prevent extreme logits/probabilities
            # (LogReg is sensitive to feature scale; unscaled/outlier rows can produce
            # very confident probabilities that break calibration diagnostics.)
            self.model = Pipeline(
                steps=[
                    ("scaler", StandardScaler(with_mean=True, with_std=True)),
                    ("clf", LogisticRegression(**self.LOGREG_CONFIG)),
                ]
            )
            self.model.fit(X, y)

            # Update state
            self.training_samples = len(df)
            self.trained_at = datetime.utcnow()
            self.model_version = f"logreg_l2_w{self.window_size}_v1"
            self.is_ready = True

            # Calculate training accuracy (not for evaluation, just sanity check)
            train_acc = float(self.model.score(X, y))

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
                "temperature": round(self.temperature, 3),
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
            # Apply optional temperature scaling to reduce overconfidence.
            # This is still a diagnostic tool; we want stable, comparable probabilities.
            probs = self._predict_proba_with_temperature(X)
            # Floor/ceiling probabilities to keep downstream metrics stable (logloss, etc.)
            eps = self.prob_eps
            probs = np.clip(probs, eps, 1.0 - eps)
            probs = probs / probs.sum(axis=1, keepdims=True)
            return probs
        except Exception as e:
            logger.warning(f"[SENSOR] Prediction failed: {e}")
            return None

    def _prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """Extract and prepare feature matrix from DataFrame."""
        df = df.copy()
        for col in self.FEATURE_COLUMNS:
            if col not in df.columns:
                df[col] = 0
        # Coerce to numeric, replace inf/NaN, and clip pathological values.
        for col in self.FEATURE_COLUMNS:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
        for col, (lo, hi) in self.FEATURE_CLIP_BOUNDS.items():
            if col in df.columns:
                df[col] = df[col].clip(lower=lo, upper=hi)
        return df[self.FEATURE_COLUMNS].values

    def _predict_proba_with_temperature(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probabilities with temperature scaling (multinomial).

        Using decision_function + our own softmax gives us a stable place
        to apply temperature without retraining or changing scikit internals.
        """
        # Pipeline: scaler -> clf
        if isinstance(self.model, Pipeline):
            scaler = self.model.named_steps["scaler"]
            clf = self.model.named_steps["clf"]
            Xs = scaler.transform(X)
            scores = clf.decision_function(Xs)
        else:
            # Fallback for legacy state (shouldn't happen after retrain)
            scores = self.model.decision_function(X)

        # Ensure 2D (n_samples, n_classes)
        if scores.ndim == 1:
            scores = np.stack([-scores, scores], axis=1)

        T = max(1e-6, float(self.temperature))
        scores = scores / T

        # Stable softmax
        scores = scores - np.max(scores, axis=1, keepdims=True)
        exp_scores = np.exp(scores)
        return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)


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
    # AUDIT P0: FT-only for apples-to-apples with Model A training
    query = text("""
        SELECT
            m.id as match_id,
            m.home_goals,
            m.away_goals,
            -- We'll compute label in Python
            m.date
        FROM matches m
        WHERE m.status = 'FT'
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
        # AUDIT GUARDRAIL: Use COALESCE to preserve existing B predictions
        # If row exists with b_home_prob NOT NULL, don't overwrite with NULL
        # This prevents regression if sensor temporarily becomes not-ready
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
                a_home_prob = EXCLUDED.a_home_prob,
                a_draw_prob = EXCLUDED.a_draw_prob,
                a_away_prob = EXCLUDED.a_away_prob,
                a_pick = EXCLUDED.a_pick,
                -- GUARDRAIL: Only update B fields if new value is NOT NULL (preserve existing)
                model_b_version = COALESCE(EXCLUDED.model_b_version, sensor_predictions.model_b_version),
                b_home_prob = COALESCE(EXCLUDED.b_home_prob, sensor_predictions.b_home_prob),
                b_draw_prob = COALESCE(EXCLUDED.b_draw_prob, sensor_predictions.b_draw_prob),
                b_away_prob = COALESCE(EXCLUDED.b_away_prob, sensor_predictions.b_away_prob),
                b_pick = COALESCE(EXCLUDED.b_pick, sensor_predictions.b_pick),
                sensor_state = CASE
                    WHEN EXCLUDED.b_home_prob IS NOT NULL THEN EXCLUDED.sensor_state
                    WHEN sensor_predictions.b_home_prob IS NOT NULL THEN sensor_predictions.sensor_state
                    ELSE EXCLUDED.sensor_state
                END
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
    # AUDIT P0: FT-only for apples-to-apples with Model A training
    query = text("""
        SELECT sp.id, sp.match_id,
               sp.a_home_prob, sp.a_draw_prob, sp.a_away_prob, sp.a_pick,
               sp.b_home_prob, sp.b_draw_prob, sp.b_away_prob, sp.b_pick,
               m.home_goals, m.away_goals
        FROM sensor_predictions sp
        JOIN matches m ON sp.match_id = m.id
        WHERE sp.evaluated_at IS NULL
          AND m.status = 'FT'
          AND m.home_goals IS NOT NULL
          AND m.away_goals IS NOT NULL
    """)

    result = await session.execute(query)
    rows = result.fetchall()
    selected = len(rows)

    if not rows:
        return {"status": "ok", "selected": 0, "updated": 0, "evaluated": 0, "message": "No pending sensor predictions"}

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
        f"[SENSOR] Evaluation complete: selected={selected}, updated={evaluated}, "
        f"A_correct={a_correct_count}, B_correct={b_correct_count}"
    )

    return {
        "status": "ok",
        "selected": selected,
        "updated": evaluated,
        "evaluated": evaluated,
        "a_correct": a_correct_count,
        "b_correct": b_correct_count,
    }


async def retry_missing_b_predictions(
    session: AsyncSession,
    include_ft: bool = False,
    match_ids: list[int] = None,
    limit: int = 100,
) -> dict:
    """
    Auto-healing: Generate Sensor B predictions for rows with b_home_prob IS NULL.

    Uses FeatureEngineer to reconstruct proper features for each match,
    then generates B prediction using the sensor model.

    AUDIT GUARDRAILS:
    - Only updates rows where b_home_prob IS NULL (idempotent, never overwrites existing B)
    - Requires sensor.is_ready = True to proceed
    - Default: only NS matches (include_ft=False prevents time-travel for evaluation)
    - Backfill mode: include_ft=True for one-time historical fill

    Args:
        session: Async database session
        include_ft: If True, also process FT matches (for backfill). Default False (auto-healing).
        match_ids: Optional list of specific match IDs to process (for targeted backfill)
        limit: Max matches to process per call

    Returns:
        Dict with retry stats
    """
    from app.features.engineering import FeatureEngineer
    from app.models import Match

    sensor = get_sensor_engine()

    # If sensor not ready, skip entirely
    if sensor is None or not sensor.is_ready:
        logger.info("[SENSOR_RETRY] Skipped: sensor not ready")
        return {
            "status": "skipped",
            "reason": "sensor_not_ready",
            "checked": 0,
            "updated": 0,
            "errors": 0,
        }

    # Build query based on mode
    if match_ids:
        # Targeted backfill: specific match IDs
        placeholders = ",".join([f":id_{i}" for i in range(len(match_ids))])
        query = text(f"""
            SELECT sp.match_id
            FROM sensor_predictions sp
            WHERE sp.b_home_prob IS NULL
              AND sp.match_id IN ({placeholders})
            LIMIT :limit
        """)
        params = {f"id_{i}": mid for i, mid in enumerate(match_ids)}
        params["limit"] = limit
    elif include_ft:
        # Backfill mode: include FT matches
        query = text("""
            SELECT sp.match_id
            FROM sensor_predictions sp
            JOIN matches m ON m.id = sp.match_id
            WHERE sp.b_home_prob IS NULL
              AND m.status IN ('NS', 'FT', 'AET', 'PEN')
            ORDER BY m.date DESC
            LIMIT :limit
        """)
        params = {"limit": limit}
    else:
        # Auto-healing mode: only future NS matches
        query = text("""
            SELECT sp.match_id
            FROM sensor_predictions sp
            JOIN matches m ON m.id = sp.match_id
            WHERE sp.b_home_prob IS NULL
              AND m.status = 'NS'
              AND m.date > NOW()
            LIMIT :limit
        """)
        params = {"limit": limit}

    result = await session.execute(query, params)
    rows = result.fetchall()

    if not rows:
        mode = "backfill" if include_ft else "auto-healing"
        logger.info(f"[SENSOR_RETRY] No matches with missing B predictions found (mode={mode})")
        return {
            "status": "ok",
            "checked": 0,
            "updated": 0,
            "errors": 0,
        }

    # Initialize feature engineer for computing features
    feature_engineer = FeatureEngineer(session=session)

    updated = 0
    errors = 0
    skipped_no_features = 0

    for row in rows:
        match_id = row.match_id
        try:
            # Get match object
            match_result = await session.execute(
                text("SELECT * FROM matches WHERE id = :match_id"),
                {"match_id": match_id}
            )
            match_row = match_result.first()
            if not match_row:
                logger.warning(f"[SENSOR_RETRY] Match {match_id} not found")
                errors += 1
                continue

            # Create Match object from row
            match = Match(
                id=match_row.id,
                home_team_id=match_row.home_team_id,
                away_team_id=match_row.away_team_id,
                date=match_row.date,
                status=match_row.status,
                home_goals=match_row.home_goals,
                away_goals=match_row.away_goals,
            )

            # Compute features using FeatureEngineer
            features = await feature_engineer.get_match_features(match)
            if not features or features.get("has_features") is False:
                logger.debug(f"[SENSOR_RETRY] Skipped match {match_id}: no features available")
                skipped_no_features += 1
                continue

            # Build DataFrame with sensor's expected columns
            df = pd.DataFrame([features])

            # Ensure all required columns exist
            missing_cols = set(sensor.FEATURE_COLUMNS) - set(df.columns)
            if missing_cols:
                logger.debug(f"[SENSOR_RETRY] Skipped match {match_id}: missing columns {missing_cols}")
                skipped_no_features += 1
                continue

            # Generate B prediction
            b_probs = sensor.predict_proba(df)
            if b_probs is None:
                logger.debug(f"[SENSOR_RETRY] Skipped match {match_id}: sensor predict_proba returned None")
                skipped_no_features += 1
                continue

            b_probs = b_probs[0]  # First row
            b_pick = ["home", "draw", "away"][np.argmax(b_probs)]

            # Update only B fields, preserve A fields
            # GUARDRAIL: WHERE b_home_prob IS NULL ensures no overwrite
            update_query = text("""
                UPDATE sensor_predictions
                SET b_home_prob = :b_home_prob,
                    b_draw_prob = :b_draw_prob,
                    b_away_prob = :b_away_prob,
                    b_pick = :b_pick,
                    model_b_version = :model_b_version,
                    sensor_state = 'READY'
                WHERE match_id = :match_id
                  AND b_home_prob IS NULL
            """)

            await session.execute(update_query, {
                "match_id": match_id,
                "b_home_prob": float(b_probs[0]),
                "b_draw_prob": float(b_probs[1]),
                "b_away_prob": float(b_probs[2]),
                "b_pick": b_pick,
                "model_b_version": sensor.model_version,
            })
            updated += 1

        except Exception as e:
            logger.warning(f"[SENSOR_RETRY] Error updating match {match_id}: {e}")
            errors += 1

    await session.commit()

    mode = "backfill" if include_ft else "auto-healing"
    logger.info(
        f"[SENSOR_RETRY] Complete (mode={mode}): checked={len(rows)}, "
        f"updated={updated}, skipped_no_features={skipped_no_features}, errors={errors}"
    )

    return {
        "status": "ok",
        "mode": mode,
        "checked": len(rows),
        "updated": updated,
        "skipped_no_features": skipped_no_features,
        "errors": errors,
    }


async def get_sensor_report(session: AsyncSession) -> dict:
    """
    Generate comprehensive sensor evaluation report.

    Returns:
        Report with A vs B comparison metrics and signal score.

    Counts semantics:
        - evaluated_total: All predictions with evaluated_at set (evaluator ran)
        - evaluated_with_b: Evaluated predictions where sensor produced probs (not LEARNING)
        - pending_total: Predictions awaiting evaluation
        - pending_with_b: Pending where sensor produced probs

    Gating for A vs B comparison uses evaluated_with_b (need sensor predictions to compare).
    """
    if not settings.SENSOR_ENABLED:
        return {"status": "DISABLED", "reason": "SENSOR_ENABLED=false"}

    window_days = settings.SENSOR_EVAL_WINDOW_DAYS
    min_samples = settings.SENSOR_MIN_SAMPLES

    # Get all counts in one query for efficiency
    counts_query = text(f"""
        SELECT
            -- Total counts
            COUNT(*) AS total,
            -- Evaluated counts (evaluator has run)
            COUNT(*) FILTER (WHERE evaluated_at IS NOT NULL) AS evaluated_total,
            -- Evaluated with B probs (sensor was READY, not LEARNING)
            COUNT(*) FILTER (WHERE evaluated_at IS NOT NULL AND b_home_prob IS NOT NULL) AS evaluated_with_b,
            -- Pending counts
            COUNT(*) FILTER (WHERE evaluated_at IS NULL) AS pending_total,
            COUNT(*) FILTER (WHERE evaluated_at IS NULL AND b_home_prob IS NOT NULL) AS pending_with_b,
            -- Sensor state distribution
            COUNT(*) FILTER (WHERE sensor_state = 'LEARNING') AS state_learning,
            COUNT(*) FILTER (WHERE sensor_state = 'READY') AS state_ready
        FROM sensor_predictions
        WHERE created_at > NOW() - INTERVAL '{window_days} days'
    """)
    counts_result = await session.execute(counts_query)
    counts_row = counts_result.first()

    evaluated_total = int(counts_row.evaluated_total or 0)
    evaluated_with_b = int(counts_row.evaluated_with_b or 0)
    pending_total = int(counts_row.pending_total or 0)
    pending_with_b = int(counts_row.pending_with_b or 0)
    state_learning = int(counts_row.state_learning or 0)
    state_ready = int(counts_row.state_ready or 0)
    # AUDIT: Expose missing B predictions (sensor was LEARNING when prediction was logged)
    missing_b_evaluated = evaluated_total - evaluated_with_b
    missing_b_pending = pending_total - pending_with_b

    # =========================================================================
    # Sanity Check: Detect overconfidence in last 24h predictions (P0 ATI/ADA)
    # Uses 1e-12 epsilon consistent with prob_eps clipping
    # =========================================================================
    sanity_query = text("""
        SELECT
            COUNT(*) AS total,
            COUNT(*) FILTER (
                WHERE GREATEST(b_home_prob, b_draw_prob, b_away_prob) > 0.9999
            ) AS overconfident_count,
            AVG(
                -b_home_prob * LN(b_home_prob + 1e-12)
                -b_draw_prob * LN(b_draw_prob + 1e-12)
                -b_away_prob * LN(b_away_prob + 1e-12)
            ) AS mean_entropy,
            COUNT(*) FILTER (
                WHERE (
                    -b_home_prob * LN(b_home_prob + 1e-12)
                    -b_draw_prob * LN(b_draw_prob + 1e-12)
                    -b_away_prob * LN(b_away_prob + 1e-12)
                ) < 0.25
            ) AS low_entropy_count,
            MIN(LEAST(b_home_prob, b_draw_prob, b_away_prob)) AS min_prob
        FROM sensor_predictions
        WHERE created_at > NOW() - INTERVAL '24 hours'
          AND b_home_prob IS NOT NULL
    """)
    sanity_result = await session.execute(sanity_query)
    sanity_row = sanity_result.first()

    total_sanity = int(sanity_row.total or 0)
    if total_sanity > 0:
        overconfident_ratio = (sanity_row.overconfident_count or 0) / total_sanity
        low_entropy_ratio = (sanity_row.low_entropy_count or 0) / total_sanity
    else:
        overconfident_ratio = 0.0
        low_entropy_ratio = 0.0

    # Determine sanity state (thresholds from ATI: 5% overconfident, 10% low entropy)
    if overconfident_ratio > 0.05 or low_entropy_ratio > 0.10:
        sanity_state = "OVERCONFIDENT"
        logger.warning(
            f"[SENSOR] Sanity check OVERCONFIDENT: overconfident_ratio={overconfident_ratio:.4f}, "
            f"low_entropy_ratio={low_entropy_ratio:.4f}, samples={total_sanity}"
        )
    else:
        sanity_state = "HEALTHY"

    sanity_metrics = {
        "state": sanity_state,
        "window_hours": 24,
        "samples": total_sanity,
        "overconfident_count": int(sanity_row.overconfident_count or 0),
        "overconfident_ratio": round(overconfident_ratio, 4),
        "mean_entropy": round(float(sanity_row.mean_entropy or 0), 4),
        "low_entropy_count": int(sanity_row.low_entropy_count or 0),
        "low_entropy_ratio": round(low_entropy_ratio, 4),
        "min_prob": float(sanity_row.min_prob) if sanity_row.min_prob is not None else None,
    }

    # Gating uses evaluated_with_b (need sensor predictions to compare A vs B)
    if evaluated_with_b < min_samples:
        # Determine current sensor state for messaging
        if state_ready > 0:
            sensor_state = "READY"
            state_msg = f"Sensor is READY, {evaluated_with_b} samples evaluated with B predictions"
        elif state_learning > 0:
            sensor_state = "LEARNING"
            state_msg = f"Sensor is LEARNING (needs {settings.SENSOR_MIN_SAMPLES} training samples), {evaluated_total} matches evaluated but B had no predictions"
        else:
            sensor_state = "UNKNOWN"
            state_msg = "No sensor predictions recorded yet"

        return {
            "status": "INSUFFICIENT_DATA",
            "reason": f"Need {min_samples} evaluated_with_b samples, have {evaluated_with_b}",
            "sensor_state": sensor_state,
            "sanity": sanity_metrics,
            "counts": {
                "total": int(counts_row.total or 0),
                "evaluated_total": evaluated_total,
                "evaluated_with_b": evaluated_with_b,
                "pending_total": pending_total,
                "pending_with_b": pending_with_b,
                # AUDIT: Expose records missing B due to sensor LEARNING state
                "missing_b_evaluated": missing_b_evaluated,
                "missing_b_pending": missing_b_pending,
            },
            "state_distribution": {
                "learning": state_learning,
                "ready": state_ready,
            },
            "gating": {
                "min_samples_required": min_samples,
                "samples_evaluated_with_b": evaluated_with_b,
                "samples_evaluated_total": evaluated_total,
                "window_days": window_days,
            },
            "recommendation": {
                "status": "LEARNING" if sensor_state == "LEARNING" else "INSUFFICIENT_DATA",
                "reason": state_msg,
            },
        }

    # Get metrics for evaluated_with_b samples only (where we can compare A vs B)
    metrics_query = text(f"""
        SELECT
            COUNT(*) AS total,
            COUNT(*) FILTER (WHERE a_correct) AS a_correct,
            COUNT(*) FILTER (WHERE b_correct) AS b_correct,
            COUNT(*) FILTER (WHERE b_correct AND NOT a_correct) AS b_wins,
            COUNT(*) FILTER (WHERE a_correct AND NOT b_correct) AS a_wins,
            COUNT(*) FILTER (WHERE a_correct AND b_correct) AS both_correct,
            COUNT(*) FILTER (WHERE NOT a_correct AND NOT b_correct) AS both_wrong,
            AVG(a_brier) AS a_brier_avg,
            AVG(b_brier) AS b_brier_avg
        FROM sensor_predictions
        WHERE evaluated_at IS NOT NULL
          AND created_at > NOW() - INTERVAL '{window_days} days'
          AND b_home_prob IS NOT NULL
    """)

    result = await session.execute(metrics_query)
    row = result.first()

    total = int(row.total or 0)

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

    return {
        "status": "ok",
        "sensor_state": "READY",
        "sanity": sanity_metrics,
        "counts": {
            "total": int(counts_row.total or 0),
            "evaluated_total": evaluated_total,
            "evaluated_with_b": evaluated_with_b,
            "pending_total": pending_total,
            "pending_with_b": pending_with_b,
            # AUDIT: Expose records missing B due to sensor LEARNING state
            "missing_b_evaluated": missing_b_evaluated,
            "missing_b_pending": missing_b_pending,
        },
        "state_distribution": {
            "learning": state_learning,
            "ready": state_ready,
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
            "samples_evaluated_with_b": evaluated_with_b,
            "samples_evaluated_total": evaluated_total,
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


async def get_sensor_health_metrics(session: AsyncSession) -> dict:
    """
    Get health metrics for sensor B telemetry.

    Returns:
        - pending_ft: FT matches with pending sensor evaluations (COUNT DISTINCT match_id)
        - eval_lag_minutes: Minutes since oldest pending prediction (0 if none)
        - state: Current sensor state (disabled, learning, ready, error)
    """
    if not settings.SENSOR_ENABLED:
        return {
            "pending_ft": 0,
            "eval_lag_minutes": 0.0,
            "state": "disabled",
        }

    # Count DISTINCT pending FT matches (not rows, which could have duplicates)
    # AUDIT P0: FT-only for apples-to-apples with Model A training
    query = text("""
        SELECT
            COUNT(DISTINCT sp.match_id) AS pending,
            MIN(sp.created_at) AS oldest_created
        FROM sensor_predictions sp
        JOIN matches m ON sp.match_id = m.id
        WHERE sp.evaluated_at IS NULL
          AND m.status = 'FT'
          AND m.home_goals IS NOT NULL
          AND m.away_goals IS NOT NULL
    """)

    result = await session.execute(query)
    row = result.first()

    pending_ft = int(row.pending or 0)
    eval_lag_minutes = 0.0

    if row.oldest_created:
        from datetime import datetime
        delta = datetime.utcnow() - row.oldest_created
        eval_lag_minutes = delta.total_seconds() / 60.0

    # Determine current state
    sensor = get_sensor_engine()
    if sensor is None:
        state = "learning"
    elif sensor.is_ready:
        state = "ready"
    else:
        state = "learning"

    return {
        "pending_ft": pending_ft,
        "eval_lag_minutes": round(eval_lag_minutes, 1),
        "state": state,
    }
