#!/usr/bin/env python3
"""
Train model v1.0.1-league-only with OPTIMIZED feature calculation.

DIFERENCIA vs train_league_only_model.py:
- Usa UNA sola query SQL batch para calcular rolling averages
- Elimina las ~20,000 queries individuales del FeatureEngineer
- Mucho más rápido para ejecución remota (Railway)

Usage:
    python scripts/train_league_only_optimized.py --cutoff 2026-01-07 --min-date 2024-01-01

Output:
    models/xgb_v1.0.1-league-only_YYYYMMDD.json
"""

import argparse
import asyncio
import logging
import os
import sys
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

from app.ml.metrics import calculate_brier_score

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# The original 14 features from v1.0.0
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

# Rolling window for averages (10 matches)
ROLLING_WINDOW = 10


async def build_features_batch_sql(session: AsyncSession, min_date: datetime, max_date: datetime) -> pd.DataFrame:
    """
    Build training features using a single batch SQL approach.

    This calculates rolling averages for ALL matches in one pass using window functions,
    avoiding the N+1 query problem of the standard FeatureEngineer.
    """
    logger.info(f"Building features with batch SQL (min={min_date}, max={max_date})...")

    # Step 1: Get all league matches with stats (the source for rolling averages)
    # Only matches where admin_leagues.kind = 'league' (FASE 1 requirement)
    # Stats are stored as JSON in matches.stats column
    league_matches_query = text("""
        WITH league_matches AS (
            SELECT
                m.id as match_id,
                m.date,
                m.home_team_id,
                m.away_team_id,
                m.home_goals,
                m.away_goals,
                COALESCE((m.stats->'home'->>'total_shots')::int, 0) as home_shots,
                COALESCE((m.stats->'away'->>'total_shots')::int, 0) as away_shots,
                COALESCE((m.stats->'home'->>'corner_kicks')::int, 0) as home_corners,
                COALESCE((m.stats->'away'->>'corner_kicks')::int, 0) as away_corners,
                m.league_id
            FROM matches m
            JOIN admin_leagues al ON m.league_id = al.league_id
            WHERE m.status = 'FT'
              AND m.home_goals IS NOT NULL
              AND m.away_goals IS NOT NULL
              AND (m.tainted IS NULL OR m.tainted = false)
              AND al.kind = 'league'  -- CRITICAL: league_only=True
              AND m.league_id NOT IN (242, 250, 252, 268, 270, 299, 344)  -- Excluir ligas sin stats API (pilot STOP/NO_DATA)
              AND m.date >= :earliest_date
            ORDER BY m.date
        )
        SELECT * FROM league_matches
    """)

    # Need matches from before min_date for calculating rolling averages
    earliest_date = min_date - timedelta(days=365)  # 1 year lookback

    result = await session.execute(league_matches_query, {"earliest_date": earliest_date})
    all_league_matches = [dict(r._mapping) for r in result.fetchall()]

    logger.info(f"Fetched {len(all_league_matches)} league matches for rolling calculations")

    # Step 2: Get training matches (matches we want to predict on)
    training_matches_query = text("""
        SELECT
            m.id as match_id,
            m.date,
            m.home_team_id,
            m.away_team_id,
            m.home_goals,
            m.away_goals,
            m.league_id,
            CASE
                WHEN m.home_goals > m.away_goals THEN 0
                WHEN m.home_goals = m.away_goals THEN 1
                ELSE 2
            END as result
        FROM matches m
        JOIN admin_leagues al ON m.league_id = al.league_id
        WHERE m.status = 'FT'
          AND m.home_goals IS NOT NULL
          AND m.away_goals IS NOT NULL
          AND (m.tainted IS NULL OR m.tainted = false)
          AND al.kind = 'league'  -- CRITICAL: training targets league-only
          AND m.league_id NOT IN (242, 250, 252, 268, 270, 299, 344)  -- Excluir ligas sin stats API (pilot STOP/NO_DATA)
          AND m.date >= :min_date
          AND m.date < :max_date
        ORDER BY m.date
    """)

    result = await session.execute(training_matches_query, {
        "min_date": min_date,
        "max_date": max_date
    })
    training_matches = [dict(r._mapping) for r in result.fetchall()]

    logger.info(f"Found {len(training_matches)} training matches")

    if not training_matches:
        return pd.DataFrame()

    # Step 3: Build index of league matches by team for fast lookup
    # Structure: {team_id: [(date, match_data), ...]} sorted by date desc
    team_matches = {}
    for m in all_league_matches:
        for team_id in [m['home_team_id'], m['away_team_id']]:
            if team_id not in team_matches:
                team_matches[team_id] = []
            team_matches[team_id].append((m['date'], m))

    # Sort by date descending for each team
    for team_id in team_matches:
        team_matches[team_id].sort(key=lambda x: x[0], reverse=True)

    logger.info(f"Built match index for {len(team_matches)} teams")

    # Step 4: Calculate features for each training match
    rows = []
    for i, tm in enumerate(training_matches):
        if (i + 1) % 1000 == 0:
            logger.info(f"Processing match {i + 1}/{len(training_matches)}")

        match_date = tm['date']

        # Get home team's last N league matches BEFORE this match
        home_id = tm['home_team_id']
        home_history = []
        if home_id in team_matches:
            for dt, m in team_matches[home_id]:
                if dt < match_date:
                    home_history.append(m)
                    if len(home_history) >= ROLLING_WINDOW:
                        break

        # Get away team's last N league matches BEFORE this match
        away_id = tm['away_team_id']
        away_history = []
        if away_id in team_matches:
            for dt, m in team_matches[away_id]:
                if dt < match_date:
                    away_history.append(m)
                    if len(away_history) >= ROLLING_WINDOW:
                        break

        # Calculate features from history
        features = calculate_team_features(home_id, home_history, "home")
        features.update(calculate_team_features(away_id, away_history, "away"))

        # Derived features
        features["goal_diff_avg"] = features["home_goals_scored_avg"] - features["away_goals_scored_avg"]
        features["rest_diff"] = features["home_rest_days"] - features["away_rest_days"]

        # Rest days calculation (days since last match)
        features["home_rest_days"] = calculate_rest_days(home_id, home_history, match_date)
        features["away_rest_days"] = calculate_rest_days(away_id, away_history, match_date)
        features["rest_diff"] = features["home_rest_days"] - features["away_rest_days"]

        # Add target
        features["result"] = tm['result']
        features["match_id"] = tm['match_id']
        features["date"] = tm['date']

        rows.append(features)

    df = pd.DataFrame(rows)
    logger.info(f"Built dataset with {len(df)} samples")

    return df


def calculate_team_features(team_id: int, history: list, prefix: str) -> dict:
    """Calculate rolling average features for a team."""
    if not history:
        return {
            f"{prefix}_goals_scored_avg": 0.0,
            f"{prefix}_goals_conceded_avg": 0.0,
            f"{prefix}_shots_avg": 0.0,
            f"{prefix}_corners_avg": 0.0,
            f"{prefix}_rest_days": 7.0,  # Default
            f"{prefix}_matches_played": 0,
        }

    goals_scored = []
    goals_conceded = []
    shots = []
    corners = []

    for m in history:
        if m['home_team_id'] == team_id:
            goals_scored.append(m['home_goals'] or 0)
            goals_conceded.append(m['away_goals'] or 0)
            shots.append(m['home_shots'] or 0)
            corners.append(m['home_corners'] or 0)
        else:
            goals_scored.append(m['away_goals'] or 0)
            goals_conceded.append(m['home_goals'] or 0)
            shots.append(m['away_shots'] or 0)
            corners.append(m['away_corners'] or 0)

    return {
        f"{prefix}_goals_scored_avg": np.mean(goals_scored) if goals_scored else 0.0,
        f"{prefix}_goals_conceded_avg": np.mean(goals_conceded) if goals_conceded else 0.0,
        f"{prefix}_shots_avg": np.mean(shots) if shots else 0.0,
        f"{prefix}_corners_avg": np.mean(corners) if corners else 0.0,
        f"{prefix}_rest_days": 7.0,  # Will be overwritten
        f"{prefix}_matches_played": len(history),
    }


def calculate_rest_days(team_id: int, history: list, match_date: datetime) -> float:
    """Calculate days since last match."""
    if not history:
        return 7.0  # Default

    last_match_date = history[0]['date']
    if isinstance(last_match_date, str):
        last_match_date = datetime.fromisoformat(last_match_date.replace('Z', '+00:00'))

    if hasattr(match_date, 'tzinfo') and match_date.tzinfo is not None:
        if hasattr(last_match_date, 'tzinfo') and last_match_date.tzinfo is None:
            last_match_date = last_match_date.replace(tzinfo=match_date.tzinfo)

    delta = match_date - last_match_date
    return max(1.0, min(30.0, delta.days))  # Clamp to [1, 30]


async def train_model(cutoff: str, min_date: str = None, draw_weight: float = 1.5):
    """Train v1.0.1-league-only model using optimized batch SQL."""
    database_url = os.environ.get('DATABASE_URL')
    if not database_url:
        raise ValueError("DATABASE_URL environment variable required")

    if database_url.startswith('postgresql://'):
        database_url = database_url.replace('postgresql://', 'postgresql+asyncpg://')

    engine = create_async_engine(database_url)
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    cutoff_date = datetime.fromisoformat(cutoff)
    min_date_dt = datetime.fromisoformat(min_date) if min_date else datetime(2020, 1, 1)
    model_version = "v1.0.1-league-only"

    async with async_session() as session:
        # Build dataset using optimized batch SQL
        df = await build_features_batch_sql(session, min_date_dt, cutoff_date)

        logger.info(f"Dataset: {len(df)} rows")
        if len(df) > 0:
            logger.info(f"Date range: {df['date'].min()} to {df['date'].max()}")

        # Verify we have all 14 features
        missing = [f for f in ORIGINAL_14_FEATURES if f not in df.columns]
        if missing:
            raise ValueError(f"Missing features in dataset: {missing}")

        # Extract only the 14 original features
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

        # TimeSeriesSplit cross-validation
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
        print(f"  league_only: True (batch SQL)")
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
        description="Train v1.0.1-league-only model (OPTIMIZED batch SQL)"
    )
    parser.add_argument(
        '--cutoff',
        required=True,
        help='Training cutoff date (ISO format, e.g., 2026-01-07)'
    )
    parser.add_argument(
        '--min-date',
        default=None,
        help='Minimum match date (ISO format, e.g., 2024-01-01)'
    )
    parser.add_argument(
        '--draw-weight',
        type=float,
        default=1.5,
        help='Weight multiplier for draws (default: 1.5)'
    )
    args = parser.parse_args()

    result = asyncio.run(train_model(args.cutoff, args.min_date, args.draw_weight))

    # Print JSON summary
    import json
    print(f"\n--- JSON Summary ---")
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
