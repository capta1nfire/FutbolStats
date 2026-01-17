"""XGBoost model engine for match prediction."""

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit

from app.config import get_settings
from app.ml.metrics import calculate_brier_score

logger = logging.getLogger(__name__)
settings = get_settings()


# =============================================================================
# FASE 2: Two-Stage Model Architecture
# =============================================================================


class TwoStageEngine:
    """
    Two-stage model for improved draw prediction.

    Architecture:
    - Stage 1: Binary classifier (draw vs non-draw)
    - Stage 2: Binary classifier (home vs away) for non-draws

    Composition (soft, no hard override):
    - p_draw = P(draw | Stage1)
    - p_home = (1 - p_draw) × P(home | non-draw, Stage2)
    - p_away = (1 - p_draw) × P(away | non-draw, Stage2)

    All probabilities sum to 1.0 by construction.
    """

    # Stage 1 features (draw detection) - includes implied_draw from odds
    STAGE1_FEATURES = [
        "home_goals_scored_avg", "home_goals_conceded_avg", "home_shots_avg",
        "home_corners_avg", "home_rest_days", "home_matches_played",
        "away_goals_scored_avg", "away_goals_conceded_avg", "away_shots_avg",
        "away_corners_avg", "away_rest_days", "away_matches_played",
        "goal_diff_avg", "rest_diff",
        "abs_attack_diff", "abs_defense_diff", "abs_strength_gap",
        "implied_draw",  # Derived from odds
    ]

    # Stage 2 features (home vs away)
    STAGE2_FEATURES = [
        "home_goals_scored_avg", "home_goals_conceded_avg", "home_shots_avg",
        "home_corners_avg", "home_rest_days", "home_matches_played",
        "away_goals_scored_avg", "away_goals_conceded_avg", "away_shots_avg",
        "away_corners_avg", "away_rest_days", "away_matches_played",
        "goal_diff_avg", "rest_diff",
        "abs_attack_diff", "abs_defense_diff", "abs_strength_gap",
    ]

    # Stage 1 hyperparameters (binary draw vs non-draw)
    PARAMS_STAGE1 = {
        "objective": "binary:logistic",
        "max_depth": 3,
        "learning_rate": 0.05,
        "n_estimators": 100,
        "min_child_weight": 7,
        "subsample": 0.72,
        "colsample_bytree": 0.71,
        "random_state": 42,
    }

    # Stage 2 hyperparameters (binary home vs away)
    PARAMS_STAGE2 = {
        "objective": "binary:logistic",
        "max_depth": 3,
        "learning_rate": 0.05,
        "n_estimators": 100,
        "min_child_weight": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
    }

    def __init__(self, model_version: str = None, draw_weight: float = 1.2):
        """
        Initialize two-stage engine.

        Args:
            model_version: Version string for saving/loading.
            draw_weight: Sample weight for draws in Stage 1 (default 1.2).
        """
        self.model_version = model_version or "v1.1.0-twostage"
        self.draw_weight = draw_weight
        self.stage1: Optional[xgb.XGBClassifier] = None
        self.stage2: Optional[xgb.XGBClassifier] = None
        self.model_path = Path(settings.MODEL_PATH)
        self.model_path.mkdir(exist_ok=True)

    def _prepare_features_stage1(self, df: pd.DataFrame) -> np.ndarray:
        """Prepare features for Stage 1, adding implied_draw from odds."""
        df = df.copy()

        # Compute implied_draw from odds if available
        if "odds_draw" in df.columns:
            df["implied_draw_raw"] = 1 / df["odds_draw"].replace(0, np.nan)
            df["implied_home_raw"] = 1 / df["odds_home"].replace(0, np.nan)
            df["implied_away_raw"] = 1 / df["odds_away"].replace(0, np.nan)
            total = df["implied_draw_raw"] + df["implied_home_raw"] + df["implied_away_raw"]
            df["implied_draw"] = df["implied_draw_raw"] / total
            df["implied_draw"] = df["implied_draw"].fillna(0.25)
        else:
            df["implied_draw"] = 0.25  # Default league average

        # Fill missing features
        for col in self.STAGE1_FEATURES:
            if col not in df.columns:
                logger.warning(f"Missing Stage1 feature {col}, filling with 0")
                df[col] = 0

        return df[self.STAGE1_FEATURES].fillna(0).values

    def _prepare_features_stage2(self, df: pd.DataFrame) -> np.ndarray:
        """Prepare features for Stage 2."""
        df = df.copy()
        for col in self.STAGE2_FEATURES:
            if col not in df.columns:
                logger.warning(f"Missing Stage2 feature {col}, filling with 0")
                df[col] = 0
        return df[self.STAGE2_FEATURES].fillna(0).values

    def train(self, df: pd.DataFrame, n_splits: int = 3) -> dict:
        """
        Train both stages using TimeSeriesSplit.

        Args:
            df: DataFrame with features and 'result' target (0=home, 1=draw, 2=away).
            n_splits: Number of CV splits.

        Returns:
            Dictionary with training metrics.
        """
        logger.info(f"Training TwoStage model {self.model_version} with {len(df)} samples")

        # Prepare targets
        y = df["result"].values
        y_draw = (y == 1).astype(int)  # Stage 1: 1=draw, 0=non-draw

        # Non-draw subset for Stage 2
        nondraw_mask = y != 1
        y_home = (y[nondraw_mask] == 0).astype(int)  # Stage 2: 1=home, 0=away

        # Prepare feature matrices
        X_s1 = self._prepare_features_stage1(df)
        X_s2_full = self._prepare_features_stage2(df)
        X_s2 = X_s2_full[nondraw_mask]

        # Sample weights for Stage 1 (upweight draws)
        sample_weight_s1 = np.ones(len(y_draw), dtype=np.float32)
        sample_weight_s1[y_draw == 1] = self.draw_weight
        logger.info(
            f"Stage1 sample weights: non-draw=1.0, draw={self.draw_weight} "
            f"(draws: {(y_draw == 1).sum()}/{len(y_draw)} = {(y_draw == 1).mean():.1%})"
        )

        # Time series CV
        tscv = TimeSeriesSplit(n_splits=n_splits)
        cv_brier_scores = []

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X_s1)):
            # Stage 1 split
            X_s1_train, X_s1_val = X_s1[train_idx], X_s1[val_idx]
            y_draw_train, y_draw_val = y_draw[train_idx], y_draw[val_idx]
            w_s1_train = sample_weight_s1[train_idx]

            # Stage 2 split (only non-draws in training)
            nondraw_train_mask = y[train_idx] != 1
            train_idx_nondraw = train_idx[nondraw_train_mask]
            X_s2_train = X_s2_full[train_idx_nondraw]
            y_home_train = (y[train_idx_nondraw] == 0).astype(int)

            # Train Stage 1
            s1_model = xgb.XGBClassifier(**self.PARAMS_STAGE1)
            s1_model.fit(X_s1_train, y_draw_train, sample_weight=w_s1_train, verbose=False)

            # Train Stage 2
            s2_model = xgb.XGBClassifier(**self.PARAMS_STAGE2)
            s2_model.fit(X_s2_train, y_home_train, verbose=False)

            # Predict on validation
            p_draw = s1_model.predict_proba(X_s1_val)[:, 1]
            X_s2_val = X_s2_full[val_idx]
            p_home_given_nondraw = s2_model.predict_proba(X_s2_val)[:, 1]

            # Compose 3-class probabilities
            p_home = (1 - p_draw) * p_home_given_nondraw
            p_away = (1 - p_draw) * (1 - p_home_given_nondraw)
            y_proba = np.column_stack([p_home, p_draw, p_away])

            # Calculate Brier
            brier = calculate_brier_score(y[val_idx], y_proba)
            cv_brier_scores.append(brier)
            logger.info(f"Fold {fold + 1}: Brier Score = {brier:.4f}")

        avg_brier = np.mean(cv_brier_scores)
        logger.info(f"Average Brier Score: {avg_brier:.4f}")

        # Train final models on all data
        self.stage1 = xgb.XGBClassifier(**self.PARAMS_STAGE1)
        self.stage1.fit(X_s1, y_draw, sample_weight=sample_weight_s1, verbose=False)

        self.stage2 = xgb.XGBClassifier(**self.PARAMS_STAGE2)
        self.stage2.fit(X_s2, y_home, verbose=False)

        logger.info("TwoStage model training complete")

        return {
            "model_version": self.model_version,
            "architecture": "two_stage",
            "samples_trained": len(df),
            "brier_score": round(avg_brier, 4),
            "cv_scores": [round(s, 4) for s in cv_brier_scores],
            "draw_weight": self.draw_weight,
        }

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict 3-class probabilities using soft composition.

        Returns:
            Array of shape (n_samples, 3) with [p_home, p_draw, p_away].
        """
        if self.stage1 is None or self.stage2 is None:
            raise ValueError("Model not trained or loaded")

        X_s1 = self._prepare_features_stage1(df)
        X_s2 = self._prepare_features_stage2(df)

        # Stage 1: P(draw)
        p_draw = self.stage1.predict_proba(X_s1)[:, 1]

        # Stage 2: P(home | non-draw)
        p_home_given_nondraw = self.stage2.predict_proba(X_s2)[:, 1]

        # Compose (soft composition, no threshold override)
        p_home = (1 - p_draw) * p_home_given_nondraw
        p_away = (1 - p_draw) * (1 - p_home_given_nondraw)

        proba = np.column_stack([p_home, p_draw, p_away])

        # Clamp to [0, 1] and renormalize (safety)
        proba = np.clip(proba, 0, 1)
        row_sums = proba.sum(axis=1, keepdims=True)
        proba = proba / row_sums

        return proba

    def save_to_bytes(self) -> bytes:
        """Export both stages to compressed bytes for DB storage."""
        import pickle
        import zlib

        if self.stage1 is None or self.stage2 is None:
            raise ValueError("No model trained to export")

        payload = {
            "stage1": self.stage1,
            "stage2": self.stage2,
            "model_version": self.model_version,
            "draw_weight": self.draw_weight,
        }

        raw_bytes = pickle.dumps(payload)
        compressed = zlib.compress(raw_bytes, level=6)

        compression_ratio = (1 - len(compressed) / len(raw_bytes)) * 100
        logger.info(
            f"TwoStage model compressed: {len(raw_bytes)} -> {len(compressed)} bytes "
            f"({compression_ratio:.1f}% reduction)"
        )

        return compressed

    def load_from_bytes(self, blob: bytes) -> bool:
        """Load both stages from compressed bytes."""
        import pickle
        import zlib

        try:
            decompressed = zlib.decompress(blob)
            payload = pickle.loads(decompressed)

            self.stage1 = payload["stage1"]
            self.stage2 = payload["stage2"]
            self.model_version = payload.get("model_version", "v1.1.0-twostage")
            self.draw_weight = payload.get("draw_weight", 1.2)

            logger.info(
                f"TwoStage model loaded from bytes: {len(blob)} compressed -> "
                f"{len(decompressed)} decompressed"
            )
            return True
        except Exception as e:
            logger.error(f"Failed to load TwoStage model from bytes: {e}")
            self.stage1 = None
            self.stage2 = None
            return False

    @property
    def is_loaded(self) -> bool:
        """Check if both stages are loaded."""
        return self.stage1 is not None and self.stage2 is not None


# =============================================================================
# Original XGBoostEngine (baseline)
# =============================================================================


class XGBoostEngine:
    """
    XGBoost-based prediction engine for football matches.

    Uses multi-class classification to predict:
    - 0: Home Win
    - 1: Draw
    - 2: Away Win
    """

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
        # FASE 1: Competitiveness features for draw prediction
        "abs_attack_diff",
        "abs_defense_diff",
        "abs_strength_gap",
    ]

    def __init__(self, model_version: str = None):
        self.model_version = model_version or settings.MODEL_VERSION
        self.model: Optional[xgb.XGBClassifier] = None
        self.model_path = Path(settings.MODEL_PATH)
        self.model_path.mkdir(exist_ok=True)
        self._feature_compat_logged = False  # Log feature truncation only once

    def _get_model_filepath(self) -> Path:
        """Get the filepath for the current model version."""
        timestamp = datetime.now().strftime("%Y%m%d")
        return self.model_path / f"xgb_{self.model_version}_{timestamp}.json"

    def _get_model_expected_features(self) -> list[str]:
        """
        Get the features expected by the loaded model.

        Handles backward compatibility with models trained on fewer features.
        """
        if self.model is None:
            return self.FEATURE_COLUMNS

        # Try to get feature count from model
        try:
            n_features = self.model.n_features_in_
        except AttributeError:
            # Fallback for older models
            n_features = len(self.FEATURE_COLUMNS)

        # If model expects fewer features than current FEATURE_COLUMNS,
        # return only the first N features (they were added in order)
        if n_features < len(self.FEATURE_COLUMNS):
            if not self._feature_compat_logged:
                logger.info(
                    f"[FEATURE_COMPAT] Model expects {n_features} features, code has {len(self.FEATURE_COLUMNS)}. "
                    f"Truncating to first {n_features} features for backward compatibility."
                )
                self._feature_compat_logged = True
            return self.FEATURE_COLUMNS[:n_features]

        return self.FEATURE_COLUMNS

    def _prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """Extract and prepare feature matrix from DataFrame."""
        # Get features expected by the loaded model (handles backward compatibility)
        expected_features = self._get_model_expected_features()

        # Ensure all required columns exist
        for col in expected_features:
            if col not in df.columns:
                logger.warning(f"Missing column {col}, filling with 0")
                df[col] = 0

        return df[expected_features].fillna(0).values

    def train(
        self,
        df: pd.DataFrame,
        n_splits: int = 3,
        hyperparams: Optional[dict] = None,
        draw_weight: float = 1.5,
    ) -> dict:
        """
        Train the XGBoost model using TimeSeriesSplit.

        Args:
            df: DataFrame with features and 'result' target column.
            n_splits: Number of splits for time series cross-validation.
            hyperparams: Optional XGBoost hyperparameters.
            draw_weight: Weight multiplier for draw samples (default 1.5).
                         Higher values give more importance to predicting draws correctly.

        Returns:
            Dictionary with training metrics.
        """
        logger.info(f"Training model {self.model_version} with {len(df)} samples")

        # Prepare data
        X = self._prepare_features(df)
        y = df["result"].values

        # FASE 1: Compute sample weights to address draw imbalance
        # Draws (class 1) are naturally rarer (~28%) but model predicts them at ~0%
        # Giving higher weight to draws forces the model to learn their patterns
        sample_weight = np.ones(len(y), dtype=np.float32)
        sample_weight[y == 1] = draw_weight  # Upweight draws
        logger.info(
            f"Sample weights: home/away=1.0, draw={draw_weight} "
            f"(draws: {(y == 1).sum()}/{len(y)} = {(y == 1).mean():.1%})"
        )

        # Optimized hyperparameters (Optuna, 50 trials, Brier Score: 0.2054)
        params = {
            "objective": "multi:softprob",
            "num_class": 3,
            "max_depth": 3,
            "learning_rate": 0.0283,
            "n_estimators": 114,
            "min_child_weight": 7,
            "subsample": 0.72,
            "colsample_bytree": 0.71,
            "reg_alpha": 2.8e-05,
            "reg_lambda": 0.000904,
            "random_state": 42,
            "use_label_encoder": False,
            "eval_metric": "mlogloss",
        }
        if hyperparams:
            params.update(hyperparams)

        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=n_splits)
        cv_scores = []

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            w_train = sample_weight[train_idx]

            fold_model = xgb.XGBClassifier(**params)
            fold_model.fit(
                X_train,
                y_train,
                sample_weight=w_train,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )

            y_proba = fold_model.predict_proba(X_val)
            brier = calculate_brier_score(y_val, y_proba)
            cv_scores.append(brier)
            logger.info(f"Fold {fold + 1}: Brier Score = {brier:.4f}")

        avg_brier = np.mean(cv_scores)
        logger.info(f"Average Brier Score: {avg_brier:.4f}")

        # Train final model on all data
        self.model = xgb.XGBClassifier(**params)
        self.model.fit(X, y, sample_weight=sample_weight, verbose=False)

        # Save model
        model_path = self._get_model_filepath()
        self.model.save_model(str(model_path))
        logger.info(f"Model saved to {model_path}")

        # Get feature importance
        importance = dict(
            zip(self.FEATURE_COLUMNS, self.model.feature_importances_.tolist())
        )

        return {
            "model_version": self.model_version,
            "samples_trained": len(df),
            "brier_score": round(avg_brier, 4),
            "cv_scores": [round(s, 4) for s in cv_scores],
            "feature_importance": importance,
            "model_path": str(model_path),
        }

    def load_model(self, model_path: Optional[str] = None) -> bool:
        """
        Load a trained model from disk.

        Args:
            model_path: Optional specific path. If None, loads latest model.

        Returns:
            True if model loaded successfully.
        """
        if model_path:
            path = Path(model_path)
        else:
            # Find latest model file
            model_files = list(self.model_path.glob(f"xgb_{self.model_version}_*.json"))
            if not model_files:
                # Try any model
                model_files = list(self.model_path.glob("xgb_*.json"))

            if not model_files:
                logger.warning("No model files found")
                return False

            path = max(model_files, key=lambda p: p.stat().st_mtime)

        try:
            self.model = xgb.XGBClassifier()
            self.model.load_model(str(path))
            logger.info(f"Model loaded from {path}")
            return True
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict probabilities for matches.

        Args:
            df: DataFrame with features.

        Returns:
            Array of shape (n_samples, 3) with probabilities.
        """
        if self.model is None:
            if not self.load_model():
                raise ValueError("No model loaded")

        X = self._prepare_features(df)
        return self.model.predict_proba(X)

    def predict(
        self,
        df: pd.DataFrame,
        team_adjustments: dict = None,
        context: dict = None,
    ) -> list[dict]:
        """
        Make predictions with probabilities, fair odds, value bet detection,
        and contextual intelligence insights.

        Args:
            df: DataFrame with features and match info.
            team_adjustments: Optional dict with home/away split multipliers.
                Format: {"home": {team_id: multiplier}, "away": {team_id: multiplier}}
                Or legacy format: {team_id: multiplier} (applied to both)
            context: Optional dict with contextual intelligence data:
                - unstable_leagues: set of league_ids with drift detected
                - odds_movements: dict of match_id -> movement data
                - international_commitments: dict of team_id -> commitment info
                - team_details: dict of team_id -> TeamAdjustment details

        Returns:
            List of prediction dictionaries with value betting metrics and insights.
        """
        probas = self.predict_proba(df)
        context = context or {}

        # Extract context data
        unstable_leagues = context.get("unstable_leagues", set())
        odds_movements = context.get("odds_movements", {})
        international_commitments = context.get("international_commitments", {})
        team_details = context.get("team_details", {})

        predictions = []
        for idx, (_, row) in enumerate(df.iterrows()):
            # Store raw probabilities before adjustments
            # Use enumerate index, not DataFrame index (which may be non-contiguous)
            raw_home_prob = float(probas[idx][0])
            raw_draw_prob = float(probas[idx][1])
            raw_away_prob = float(probas[idx][2])

            home_prob = raw_home_prob
            draw_prob = raw_draw_prob
            away_prob = raw_away_prob

            # Track insights and warnings
            insights = []
            warnings = []

            # Get team and match info
            home_team_id = row.get("home_team_id")
            away_team_id = row.get("away_team_id")
            match_id = row.get("match_id")
            league_id = row.get("league_id")
            home_team_name = row.get("home_team_name", "Local")
            away_team_name = row.get("away_team_name", "Visitante")

            # Apply team adjustments if provided
            adjustment_applied = False
            home_adj = 1.0
            away_adj = 1.0
            intl_penalty_home = None
            intl_penalty_away = None

            if team_adjustments:
                # Check for new format with home/away split
                if "home" in team_adjustments and "away" in team_adjustments:
                    home_adjustments = team_adjustments["home"]
                    away_adjustments = team_adjustments["away"]

                    home_adj = home_adjustments.get(home_team_id, 1.0) if home_team_id else 1.0
                    away_adj = away_adjustments.get(away_team_id, 1.0) if away_team_id else 1.0
                else:
                    home_adj = team_adjustments.get(home_team_id, 1.0) if home_team_id else 1.0
                    away_adj = team_adjustments.get(away_team_id, 1.0) if away_team_id else 1.0

                # Generate insights for team adjustments
                if home_adj != 1.0:
                    home_detail = team_details.get(home_team_id, {})
                    anomaly_pct = home_detail.get("home_anomaly_rate", 0)
                    if anomaly_pct > 0:
                        insights.append(f"{home_team_name} home_mult={home_adj:.2f} (anomalias {anomaly_pct:.0%})")

                if away_adj != 1.0:
                    away_detail = team_details.get(away_team_id, {})
                    anomaly_pct = away_detail.get("away_anomaly_rate", 0)
                    if anomaly_pct > 0:
                        insights.append(f"{away_team_name} away_mult={away_adj:.2f} (anomalias {anomaly_pct:.0%})")

                # Check international commitments
                if home_team_id in international_commitments:
                    commitment = international_commitments[home_team_id]
                    intl_penalty_home = commitment.get("penalty", 1.0)
                    insights.append(f"{home_team_name} tiene compromiso internacional en {commitment.get('days', '?')} dias")
                    warnings.append("INTERNATIONAL_COMMITMENT")

                if away_team_id in international_commitments:
                    commitment = international_commitments[away_team_id]
                    intl_penalty_away = commitment.get("penalty", 1.0)
                    insights.append(f"{away_team_name} tiene compromiso internacional en {commitment.get('days', '?')} dias")
                    if "INTERNATIONAL_COMMITMENT" not in warnings:
                        warnings.append("INTERNATIONAL_COMMITMENT")

                # Apply adjustments to win probabilities
                if home_adj != 1.0 or away_adj != 1.0:
                    home_prob *= home_adj
                    away_prob *= away_adj

                    # Renormalize to sum to 1.0
                    total = home_prob + draw_prob + away_prob
                    home_prob /= total
                    draw_prob /= total
                    away_prob /= total
                    adjustment_applied = True

            # Calculate base confidence tier
            max_prob = max(home_prob, draw_prob, away_prob)
            if max_prob >= 0.55:
                confidence_tier = "gold"
            elif max_prob >= 0.45:
                confidence_tier = "silver"
            else:
                confidence_tier = "copper"

            original_tier = confidence_tier
            tier_degradations = 0

            # Check league drift
            league_drift_applied = False
            if league_id and league_id in unstable_leagues:
                tier_degradations += 1
                league_drift_applied = True
                insights.append(f"Liga {league_id} inestable - accuracy en caida")
                warnings.append("LEAGUE_DRIFT")

            # Check odds movement
            odds_movement_data = None
            if match_id and match_id in odds_movements:
                movement = odds_movements[match_id]
                if movement.get("has_movement"):
                    tier_deg = movement.get("tier_degradation", 0)
                    tier_degradations += tier_deg
                    odds_movement_data = movement
                    pct = movement.get("movement_percentage", 0)
                    outcome = movement.get("max_movement_outcome", "?")
                    insights.append(f"Movimiento de cuotas +{pct:.0f}% en {outcome}")
                    warnings.append("ODDS_MOVEMENT")

            # Apply tier degradations
            if tier_degradations > 0:
                tier_order = ["gold", "silver", "copper"]
                try:
                    current_idx = tier_order.index(confidence_tier)
                    new_idx = min(current_idx + tier_degradations, len(tier_order) - 1)
                    confidence_tier = tier_order[new_idx]
                except ValueError:
                    confidence_tier = "copper"

            pred = {
                "match_id": int(match_id) if match_id else None,
                "match_external_id": int(row.get("match_external_id")) if row.get("match_external_id") else None,
                "home_team": home_team_name,
                "away_team": away_team_name,
                "home_team_logo": row.get("home_team_logo"),
                "away_team_logo": row.get("away_team_logo"),
                # External IDs for team override resolution
                "home_team_external_id": int(row.get("home_team_external_id")) if row.get("home_team_external_id") else None,
                "away_team_external_id": int(row.get("away_team_external_id")) if row.get("away_team_external_id") else None,
                "date": row.get("date"),
                "status": row.get("status"),
                "elapsed": int(row.get("elapsed")) if pd.notna(row.get("elapsed")) else None,
                "home_goals": int(row.get("home_goals")) if pd.notna(row.get("home_goals")) else None,
                "away_goals": int(row.get("away_goals")) if pd.notna(row.get("away_goals")) else None,
                "league_id": league_id,
                "venue": {
                    "name": row.get("venue_name"),
                    "city": row.get("venue_city"),
                } if row.get("venue_name") else None,

                # Adjusted probabilities
                "probabilities": {
                    "home": round(home_prob, 4),
                    "draw": round(draw_prob, 4),
                    "away": round(away_prob, 4),
                },
                # Raw model output (before adjustments)
                "raw_probabilities": {
                    "home": round(raw_home_prob, 4),
                    "draw": round(raw_draw_prob, 4),
                    "away": round(raw_away_prob, 4),
                } if adjustment_applied else None,

                "fair_odds": {
                    "home": round(1 / home_prob, 2) if home_prob > 0 else None,
                    "draw": round(1 / draw_prob, 2) if draw_prob > 0 else None,
                    "away": round(1 / away_prob, 2) if away_prob > 0 else None,
                },

                # Confidence tier with degradation tracking
                "confidence_tier": confidence_tier,
                "original_tier": original_tier if confidence_tier != original_tier else None,

                "has_value_bet": False,
                "best_value_bet": None,

                # Detailed adjustment info
                "adjustment_applied": adjustment_applied or tier_degradations > 0,
                "adjustments": {
                    "home_multiplier": round(home_adj, 4) if home_adj != 1.0 else None,
                    "away_multiplier": round(away_adj, 4) if away_adj != 1.0 else None,
                    "international_penalty_home": intl_penalty_home,
                    "international_penalty_away": intl_penalty_away,
                    "league_drift_applied": league_drift_applied,
                    "odds_movement_degradation": tier_degradations if odds_movement_data else None,
                } if (adjustment_applied or tier_degradations > 0) else None,

                # Reasoning engine output
                "prediction_insights": insights if insights else None,
                "warnings": warnings if warnings else None,
            }

            # Add value bets if market odds available
            if row.get("odds_home") and row.get("odds_draw") and row.get("odds_away"):
                # Convert numpy types to native Python floats for JSON serialization
                odds_home = float(row["odds_home"])
                odds_draw = float(row["odds_draw"])
                odds_away = float(row["odds_away"])

                pred["market_odds"] = {
                    "home": odds_home,
                    "draw": odds_draw,
                    "away": odds_away,
                }
                value_bets = self._find_value_bets(
                    probas[idx],
                    [odds_home, odds_draw, odds_away],
                )
                pred["value_bets"] = value_bets

                # Flag if has any value bet and find best one
                if value_bets:
                    pred["has_value_bet"] = True
                    # Best value bet = highest EV
                    pred["best_value_bet"] = max(
                        value_bets, key=lambda x: x["expected_value"]
                    )

            predictions.append(pred)

        return predictions

    def _find_value_bets(
        self,
        probas: np.ndarray,
        market_odds: list[float],
        threshold: float = 0.05,
    ) -> list[dict]:
        """
        Find value betting opportunities with Expected Value calculation.

        A value bet exists when our probability is higher than
        the implied probability from market odds by more than threshold.

        EV (Expected Value) = (probability * odds) - 1
        - Positive EV = profitable bet in the long run
        - edge > 5% = value bet

        Args:
            probas: Model probabilities [home, draw, away].
            market_odds: Market odds [home, draw, away].
            threshold: Minimum edge required (default 5%).

        Returns:
            List of value bet opportunities with EV metrics.
        """
        outcomes = ["home", "draw", "away"]
        value_bets = []

        for i, (prob, odds) in enumerate(zip(probas, market_odds)):
            if odds <= 0:
                continue

            # Convert numpy types to native Python floats
            prob = float(prob)
            implied_prob = 1 / odds
            edge = prob - implied_prob

            # Calculate Expected Value: EV = (prob * odds) - 1
            # EV > 0 means profitable, EV of 0.10 means 10% expected return
            expected_value = (prob * odds) - 1

            if edge > threshold:
                value_bets.append({
                    "outcome": outcomes[i],
                    "our_probability": round(prob, 4),
                    "implied_probability": round(implied_prob, 4),
                    "edge": round(edge, 4),
                    "edge_percentage": round(edge * 100, 1),  # 5.2%
                    "expected_value": round(expected_value, 4),
                    "ev_percentage": round(expected_value * 100, 1),  # 10.5%
                    "market_odds": float(odds),
                    "fair_odds": round(1 / prob, 2),
                    "is_value_bet": True,
                })

        return value_bets

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self.model is not None

    def save_to_bytes(self) -> bytes:
        """
        Export the trained model to compressed bytes for database storage.

        Uses pickle + zlib for ~70% size reduction.
        This can be stored in PostgreSQL BYTEA and loaded without disk I/O.

        Returns:
            bytes: The compressed model as binary data.

        Raises:
            ValueError: If no model is trained.
        """
        import pickle
        import zlib

        if self.model is None:
            raise ValueError("No model trained to export")

        raw_bytes = pickle.dumps(self.model)
        compressed = zlib.compress(raw_bytes, level=6)

        compression_ratio = (1 - len(compressed) / len(raw_bytes)) * 100
        logger.info(
            f"Model compressed: {len(raw_bytes)} -> {len(compressed)} bytes "
            f"({compression_ratio:.1f}% reduction)"
        )

        return compressed

    def load_from_bytes(self, blob: bytes) -> bool:
        """
        Load a model directly from compressed bytes (from database).

        Args:
            blob: Compressed binary model data from save_to_bytes() or database.

        Returns:
            True if model loaded successfully, False otherwise.
        """
        import pickle
        import zlib

        try:
            decompressed = zlib.decompress(blob)
            self.model = pickle.loads(decompressed)

            logger.info(
                f"Model loaded from bytes: {len(blob)} compressed -> "
                f"{len(decompressed)} decompressed"
            )
            return True
        except Exception as e:
            logger.error(f"Failed to load model from bytes: {e}")
            self.model = None
            return False
