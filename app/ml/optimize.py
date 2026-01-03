"""Hyperparameter optimization using Optuna for XGBoost model."""

import logging
import sys
from pathlib import Path

import numpy as np
import optuna
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.features.engineering import FeatureEngineer
from app.ml.metrics import calculate_brier_score

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress Optuna's verbose output
optuna.logging.set_verbosity(optuna.logging.WARNING)


class XGBoostOptimizer:
    """Optuna-based hyperparameter optimizer for XGBoost."""

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

    def __init__(self, n_splits: int = 5):
        self.n_splits = n_splits
        self.X = None
        self.y = None
        self.best_params = None
        self.best_score = None
        self.baseline_score = 0.2172  # Current production score

    def load_data(self, database_url: str = None) -> tuple[np.ndarray, np.ndarray]:
        """Load and prepare training data."""
        import asyncio
        import os
        from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
        from sqlalchemy.orm import sessionmaker

        # Use provided URL, environment variable, or default to Railway production
        db_url = database_url or os.environ.get(
            "PROD_DATABASE_URL",
            "postgresql+asyncpg://postgres:hzvozcXijUpblVrQshuowYcEGwZnMrfO@maglev.proxy.rlwy.net:24997/railway"
        )

        async def fetch_data():
            engine = create_async_engine(db_url)
            async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

            async with async_session() as session:
                fe = FeatureEngineer(session)
                df = await fe.build_training_dataset()
                return df

        df = asyncio.run(fetch_data())

        # Prepare features
        for col in self.FEATURE_COLUMNS:
            if col not in df.columns:
                df[col] = 0

        self.X = df[self.FEATURE_COLUMNS].fillna(0).values
        self.y = df["result"].values

        logger.info(f"Loaded {len(self.X)} samples with {len(self.FEATURE_COLUMNS)} features")
        return self.X, self.y

    def objective(self, trial: optuna.Trial) -> float:
        """
        Optuna objective function.

        Minimizes Brier Score using TimeSeriesSplit cross-validation.
        """
        # Define search space
        params = {
            "objective": "multi:softprob",
            "num_class": 3,
            "eval_metric": "mlogloss",
            "use_label_encoder": False,
            "random_state": 42,
            "verbosity": 0,
            # Hyperparameters to optimize
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            # Additional regularization
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 1.0, log=True),
        }

        # TimeSeriesSplit cross-validation
        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        cv_scores = []

        for train_idx, val_idx in tscv.split(self.X):
            X_train, X_val = self.X[train_idx], self.X[val_idx]
            y_train, y_val = self.y[train_idx], self.y[val_idx]

            model = xgb.XGBClassifier(**params)
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )

            y_proba = model.predict_proba(X_val)
            brier = calculate_brier_score(y_val, y_proba)
            cv_scores.append(brier)

        return np.mean(cv_scores)

    def optimize(self, n_trials: int = 50) -> dict:
        """
        Run Optuna optimization.

        Args:
            n_trials: Number of trials to run.

        Returns:
            Dictionary with best parameters and scores.
        """
        if self.X is None:
            self.load_data()

        logger.info(f"Starting optimization with {n_trials} trials...")
        logger.info(f"Baseline Brier Score: {self.baseline_score}")

        # Create study (minimize Brier Score)
        study = optuna.create_study(
            direction="minimize",
            study_name="xgboost_football_prediction",
            sampler=optuna.samplers.TPESampler(seed=42),
        )

        # Add callback for progress
        def callback(study, trial):
            if trial.number % 10 == 0:
                logger.info(
                    f"Trial {trial.number}: Brier={trial.value:.4f} "
                    f"(Best so far: {study.best_value:.4f})"
                )

        study.optimize(
            self.objective,
            n_trials=n_trials,
            callbacks=[callback],
            show_progress_bar=True,
        )

        self.best_params = study.best_params
        self.best_score = study.best_value

        # Calculate improvement
        improvement = ((self.baseline_score - self.best_score) / self.baseline_score) * 100

        results = {
            "best_params": self.best_params,
            "best_brier_score": round(self.best_score, 4),
            "baseline_brier_score": self.baseline_score,
            "improvement_percent": round(improvement, 2),
            "n_trials": n_trials,
            "is_better": self.best_score < self.baseline_score,
        }

        logger.info("=" * 60)
        logger.info("OPTIMIZATION COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Baseline Brier Score: {self.baseline_score}")
        logger.info(f"New Best Brier Score: {self.best_score:.4f}")
        logger.info(f"Improvement: {improvement:.2f}%")
        logger.info("-" * 60)
        logger.info("Best Hyperparameters:")
        for key, value in self.best_params.items():
            logger.info(f"  {key}: {value}")
        logger.info("=" * 60)

        return results

    def get_engine_params(self) -> dict:
        """
        Get parameters formatted for XGBoostEngine.

        Returns:
            Dictionary ready to be used in engine.py.
        """
        if self.best_params is None:
            raise ValueError("No optimization has been run yet")

        return {
            "max_depth": self.best_params["max_depth"],
            "learning_rate": round(self.best_params["learning_rate"], 4),
            "n_estimators": self.best_params["n_estimators"],
            "subsample": round(self.best_params["subsample"], 2),
            "colsample_bytree": round(self.best_params["colsample_bytree"], 2),
            "min_child_weight": self.best_params["min_child_weight"],
            "reg_alpha": round(self.best_params["reg_alpha"], 6),
            "reg_lambda": round(self.best_params["reg_lambda"], 6),
        }


def main():
    """Run hyperparameter optimization."""
    optimizer = XGBoostOptimizer(n_splits=5)

    # Load data
    optimizer.load_data()

    # Run optimization
    results = optimizer.optimize(n_trials=50)

    # Print results
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"Baseline Score: {results['baseline_brier_score']}")
    print(f"Best Score:     {results['best_brier_score']}")
    print(f"Improvement:    {results['improvement_percent']}%")
    print(f"Is Better:      {results['is_better']}")
    print("-" * 60)
    print("Best Parameters for engine.py:")
    print("-" * 60)

    engine_params = optimizer.get_engine_params()
    for key, value in engine_params.items():
        print(f'    "{key}": {value},')

    print("=" * 60)

    return results


if __name__ == "__main__":
    main()
