#!/usr/bin/env python3
"""
Train model v1.0.1-league-only with league_only=True features.

FASE 1: Eliminates training-serving skew by training with the same
feature engineering used in serving (league_only=True).

Uses exactly 14 features (same as v1.0.0, without 3 competitiveness features).

Usage:
    python scripts/train_league_only_model.py --cutoff 2026-01-07

Output:
    models/xgb_v1.0.1-league-only_YYYYMMDD.json
"""

import argparse
import asyncio
import logging
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

from app.features.engineering import FeatureEngineer
from app.ml.metrics import calculate_brier_score

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# The original 14 features from v1.0.0 (without 3 competitiveness features)
# CRITICAL: Must match exactly what v1.0.0 was trained on
ORIGINAL_14_FEATURES = [
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

# Optimized hyperparameters from v1.0.0 (Optuna, 50 trials)
HYPERPARAMS = {
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


async def train_model(cutoff: str, min_date: str = None, draw_weight: float = 1.5):
    """
    Train v1.0.1-league-only model.

    Args:
        cutoff: Training cutoff date (ISO format). Only matches before this date.
        min_date: Minimum match date (ISO format). Use to limit dataset size.
        draw_weight: Weight multiplier for draws in training.
    """
    database_url = os.environ.get('DATABASE_URL')
    if not database_url:
        raise ValueError("DATABASE_URL environment variable required")

    if database_url.startswith('postgresql://'):
        database_url = database_url.replace('postgresql://', 'postgresql+asyncpg://')

    engine = create_async_engine(database_url)
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    cutoff_date = datetime.fromisoformat(cutoff)
    min_date_dt = datetime.fromisoformat(min_date) if min_date else None
    model_version = "v1.0.1-league-only"

    async with async_session() as session:
        engineer = FeatureEngineer(session=session)

        # CRITICAL: league_only=True eliminates training-serving skew
        date_range_str = f"min={min_date or 'None'}, max={cutoff}"
        logger.info(f"Building dataset with league_only=True, {date_range_str}")
        df = await engineer.build_training_dataset(
            min_date=min_date_dt,
            max_date=cutoff_date,
            league_only=True,  # <- ELIMINATES TRAINING-SERVING SKEW
        )

        logger.info(f"Dataset: {len(df)} rows")
        if len(df) > 0:
            logger.info(f"Date range: {df['date'].min()} to {df['date'].max()}")

        # Verify we have all 14 features
        missing = [f for f in ORIGINAL_14_FEATURES if f not in df.columns]
        if missing:
            raise ValueError(f"Missing features in dataset: {missing}")

        # Extract only the 14 original features (same as v1.0.0)
        X = df[ORIGINAL_14_FEATURES].fillna(0).values
        y = df['result'].values

        logger.info(f"Training with {X.shape[1]} features (expected: 14)")
        logger.info(f"Features: {ORIGINAL_14_FEATURES}")

        # Compute sample weights for draw balancing
        sample_weight = np.ones(len(y), dtype=np.float32)
        sample_weight[y == 1] = draw_weight  # Upweight draws

        n_home = (y == 0).sum()
        n_draw = (y == 1).sum()
        n_away = (y == 2).sum()
        logger.info(
            f"Class distribution: home={n_home} ({n_home/len(y):.1%}), "
            f"draw={n_draw} ({n_draw/len(y):.1%}), "
            f"away={n_away} ({n_away/len(y):.1%})"
        )
        logger.info(f"Sample weights: home/away=1.0, draw={draw_weight}")

        # TimeSeriesSplit cross-validation (3 folds, respecting temporal order)
        n_splits = 3
        tscv = TimeSeriesSplit(n_splits=n_splits)
        cv_scores = []

        logger.info(f"Cross-validation with TimeSeriesSplit (n_splits={n_splits})")

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            w_train = sample_weight[train_idx]

            fold_model = xgb.XGBClassifier(**HYPERPARAMS)
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
            logger.info(f"Fold {fold + 1}: Brier Score = {brier:.4f}, val_size={len(val_idx)}")

        avg_brier = np.mean(cv_scores)
        logger.info(f"Average Brier Score (CV): {avg_brier:.4f}")

        # Train final model on all data
        logger.info("Training final model on all data...")
        final_model = xgb.XGBClassifier(**HYPERPARAMS)
        final_model.fit(X, y, sample_weight=sample_weight, verbose=False)

        # Save model
        timestamp = datetime.now().strftime("%Y%m%d")
        model_path = f"models/xgb_{model_version}_{timestamp}.json"
        os.makedirs("models", exist_ok=True)
        final_model.save_model(model_path)
        logger.info(f"Model saved to {model_path}")

        # Feature importance
        importance = dict(zip(ORIGINAL_14_FEATURES, final_model.feature_importances_.tolist()))
        sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)

        print(f"\n{'='*60}")
        print(f"TRAINING COMPLETE: {model_version}")
        print(f"{'='*60}")
        print(f"  model_version: {model_version}")
        print(f"  samples_trained: {len(df)}")
        print(f"  brier_score (CV): {avg_brier:.4f}")
        print(f"  cv_scores: {[round(s, 4) for s in cv_scores]}")
        print(f"  n_features: {final_model.n_features_in_}")
        print(f"  expected_features (14): {ORIGINAL_14_FEATURES}")
        print(f"  league_only: True")
        print(f"  draw_weight: {draw_weight}")
        print(f"  date_range: {df['date'].min()} to {df['date'].max()}")
        print(f"  model_path: {model_path}")
        print(f"\nFeature Importance (top 5):")
        for feat, imp in sorted_importance[:5]:
            print(f"    {feat}: {imp:.4f}")

        result = {
            'model_version': model_version,
            'samples_trained': len(df),
            'brier_score': round(avg_brier, 4),
            'cv_scores': [round(s, 4) for s in cv_scores],
            'n_features': final_model.n_features_in_,
            'league_only': True,
            'model_path': model_path,
            'date_range': {
                'min': str(df['date'].min()),
                'max': str(df['date'].max()),
            },
            'feature_importance': {k: round(v, 4) for k, v in sorted_importance},
        }

        return result

    await engine.dispose()


def main():
    parser = argparse.ArgumentParser(
        description="Train v1.0.1-league-only model (FASE 1: eliminate training-serving skew)"
    )
    parser.add_argument(
        '--cutoff',
        required=True,
        help='Training cutoff date (ISO format, e.g., 2026-01-07). Only matches before this date.'
    )
    parser.add_argument(
        '--min-date',
        default=None,
        help='Minimum match date (ISO format, e.g., 2024-01-01). Use to limit dataset size.'
    )
    parser.add_argument(
        '--draw-weight',
        type=float,
        default=1.5,
        help='Weight multiplier for draws (default: 1.5)'
    )
    args = parser.parse_args()

    result = asyncio.run(train_model(args.cutoff, args.min_date, args.draw_weight))

    # Print JSON summary for ATI
    import json
    print(f"\n--- JSON Summary ---")
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
