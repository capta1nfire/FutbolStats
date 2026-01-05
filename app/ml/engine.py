"""XGBoost model engine for match prediction."""

import json
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
    ]

    def __init__(self, model_version: str = None):
        self.model_version = model_version or settings.MODEL_VERSION
        self.model: Optional[xgb.XGBClassifier] = None
        self.model_path = Path(settings.MODEL_PATH)
        self.model_path.mkdir(exist_ok=True)

    def _get_model_filepath(self) -> Path:
        """Get the filepath for the current model version."""
        timestamp = datetime.now().strftime("%Y%m%d")
        return self.model_path / f"xgb_{self.model_version}_{timestamp}.json"

    def _prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """Extract and prepare feature matrix from DataFrame."""
        # Ensure all required columns exist
        for col in self.FEATURE_COLUMNS:
            if col not in df.columns:
                logger.warning(f"Missing column {col}, filling with 0")
                df[col] = 0

        return df[self.FEATURE_COLUMNS].fillna(0).values

    def train(
        self,
        df: pd.DataFrame,
        n_splits: int = 3,
        hyperparams: Optional[dict] = None,
    ) -> dict:
        """
        Train the XGBoost model using TimeSeriesSplit.

        Args:
            df: DataFrame with features and 'result' target column.
            n_splits: Number of splits for time series cross-validation.
            hyperparams: Optional XGBoost hyperparameters.

        Returns:
            Dictionary with training metrics.
        """
        logger.info(f"Training model {self.model_version} with {len(df)} samples")

        # Prepare data
        X = self._prepare_features(df)
        y = df["result"].values

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

            fold_model = xgb.XGBClassifier(**params)
            fold_model.fit(
                X_train,
                y_train,
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
        self.model.fit(X, y, verbose=False)

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
        for i, row in df.iterrows():
            # Store raw probabilities before adjustments
            raw_home_prob = float(probas[i][0])
            raw_draw_prob = float(probas[i][1])
            raw_away_prob = float(probas[i][2])

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
                "date": row.get("date"),
                "status": row.get("status"),
                "home_goals": int(row.get("home_goals")) if pd.notna(row.get("home_goals")) else None,
                "away_goals": int(row.get("away_goals")) if pd.notna(row.get("away_goals")) else None,
                "league_id": league_id,

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
                    probas[i],
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
