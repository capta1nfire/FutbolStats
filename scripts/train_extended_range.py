#!/usr/bin/env python3
"""
Train models with extended date range and kill-switch guardrails.

Experiment: Test if extending training to 2023 improves generalization.

Variants:
  A (baseline):       min_date=2024-01-01
  B (extended soft):  min_date=2023-11-01
  C (extended strong): min_date=2023-08-01

Guardrails (aligned with kill-switch):
  - Exclude training rows where home or away team has < 5 league matches in last 90 days
  - Optional: Exclude cold-start months (Aug-Sep 2023)

Usage:
    python scripts/train_extended_range.py --variant A --cutoff 2026-01-08
    python scripts/train_extended_range.py --variant B --cutoff 2026-01-08
    python scripts/train_extended_range.py --variant C --cutoff 2026-01-08 --exclude-coldstart

Output:
    models/xgb_v1.0.2-ext-{variant}_YYYYMMDD.json
"""

import argparse
import asyncio
import logging
import os
import sys
from datetime import datetime, timedelta
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

from app.ml.metrics import calculate_brier_score

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Variant configurations
VARIANTS = {
    "A": {
        "min_date": "2024-01-01",
        "description": "baseline (2024+)",
        "model_version": "v1.0.2-ext-A",
    },
    "B": {
        "min_date": "2023-11-01",
        "description": "extended soft (Nov 2023+)",
        "model_version": "v1.0.2-ext-B",
    },
    "C": {
        "min_date": "2023-08-01",
        "description": "extended strong (Aug 2023+)",
        "model_version": "v1.0.2-ext-C",
    },
}

# Kill-switch guardrail constants
MIN_LEAGUE_MATCHES = 5
LOOKBACK_DAYS = 90

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

ROLLING_WINDOW = 10


def count_league_matches_in_window(team_id: int, match_date: datetime,
                                    team_matches_index: dict) -> int:
    """
    Count how many league matches a team has in the 90-day window before match_date.

    PIT-safe: Only counts matches with dt < match_date AND dt >= match_date - 90d.
    """
    if team_id not in team_matches_index:
        return 0

    cutoff_start = match_date - timedelta(days=LOOKBACK_DAYS)
    count = 0

    for dt, _ in team_matches_index[team_id]:
        if cutoff_start <= dt < match_date:
            count += 1

    return count


async def build_features_with_guardrails(
    session: AsyncSession,
    min_date: datetime,
    max_date: datetime,
    exclude_coldstart: bool = False
) -> tuple[pd.DataFrame, dict]:
    """
    Build training features with kill-switch guardrails.

    Returns:
        (DataFrame with features, metadata dict with guardrail stats)
    """
    logger.info(f"Building features (min={min_date}, max={max_date}, exclude_coldstart={exclude_coldstart})...")

    # Step 1: Get all league matches for rolling calculations
    earliest_date = min_date - timedelta(days=365)

    league_matches_query = text("""
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
          AND al.kind = 'league'
          AND m.date >= :earliest_date
        ORDER BY m.date
    """)

    result = await session.execute(league_matches_query, {"earliest_date": earliest_date})
    all_league_matches = [dict(r._mapping) for r in result.fetchall()]
    logger.info(f"Fetched {len(all_league_matches)} league matches for rolling calculations")

    # Step 2: Get training matches
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
        WHERE m.status = 'FT'
          AND m.home_goals IS NOT NULL
          AND m.away_goals IS NOT NULL
          AND (m.tainted IS NULL OR m.tainted = false)
          AND m.date >= :min_date
          AND m.date < :max_date
        ORDER BY m.date
    """)

    result = await session.execute(training_matches_query, {
        "min_date": min_date,
        "max_date": max_date
    })
    training_matches = [dict(r._mapping) for r in result.fetchall()]
    logger.info(f"Found {len(training_matches)} candidate training matches")

    if not training_matches:
        return pd.DataFrame(), {"error": "No training matches found"}

    # Step 3: Build team match index for fast lookup
    team_matches_index = defaultdict(list)
    for m in all_league_matches:
        for team_id in [m['home_team_id'], m['away_team_id']]:
            team_matches_index[team_id].append((m['date'], m))

    # Sort by date for each team
    for team_id in team_matches_index:
        team_matches_index[team_id].sort(key=lambda x: x[0])

    logger.info(f"Built match index for {len(team_matches_index)} teams")

    # Step 4: Apply guardrails and calculate features
    rows = []
    guardrail_stats = {
        "total_candidates": len(training_matches),
        "excluded_home_insufficient": 0,
        "excluded_away_insufficient": 0,
        "excluded_both_insufficient": 0,
        "excluded_coldstart": 0,
        "included": 0,
    }

    for i, tm in enumerate(training_matches):
        if (i + 1) % 2000 == 0:
            logger.info(f"Processing match {i + 1}/{len(training_matches)}")

        match_date = tm['date']
        home_id = tm['home_team_id']
        away_id = tm['away_team_id']

        # GUARDRAIL 1: Cold-start filter (optional)
        if exclude_coldstart:
            if isinstance(match_date, datetime):
                if match_date.year == 2023 and match_date.month in [8, 9]:
                    guardrail_stats["excluded_coldstart"] += 1
                    continue

        # GUARDRAIL 2: Kill-switch alignment
        # Count league matches in 90-day window BEFORE this match (PIT-safe)
        home_league_count = count_league_matches_in_window(home_id, match_date, team_matches_index)
        away_league_count = count_league_matches_in_window(away_id, match_date, team_matches_index)

        home_ok = home_league_count >= MIN_LEAGUE_MATCHES
        away_ok = away_league_count >= MIN_LEAGUE_MATCHES

        if not home_ok and not away_ok:
            guardrail_stats["excluded_both_insufficient"] += 1
            continue
        elif not home_ok:
            guardrail_stats["excluded_home_insufficient"] += 1
            continue
        elif not away_ok:
            guardrail_stats["excluded_away_insufficient"] += 1
            continue

        # Match passes guardrails - calculate features
        guardrail_stats["included"] += 1

        # Get home team history (last N matches BEFORE this match)
        home_history = []
        for dt, m in reversed(team_matches_index.get(home_id, [])):
            if dt < match_date:
                home_history.append(m)
                if len(home_history) >= ROLLING_WINDOW:
                    break

        # Get away team history
        away_history = []
        for dt, m in reversed(team_matches_index.get(away_id, [])):
            if dt < match_date:
                away_history.append(m)
                if len(away_history) >= ROLLING_WINDOW:
                    break

        # Calculate features
        features = calculate_team_features(home_id, home_history, "home")
        features.update(calculate_team_features(away_id, away_history, "away"))

        # Rest days
        features["home_rest_days"] = calculate_rest_days(home_history, match_date)
        features["away_rest_days"] = calculate_rest_days(away_history, match_date)

        # Derived features
        features["goal_diff_avg"] = features["home_goals_scored_avg"] - features["away_goals_scored_avg"]
        features["rest_diff"] = features["home_rest_days"] - features["away_rest_days"]

        # Target and metadata
        features["result"] = tm['result']
        features["match_id"] = tm['match_id']
        features["date"] = tm['date']
        features["home_league_matches_90d"] = home_league_count
        features["away_league_matches_90d"] = away_league_count

        rows.append(features)

    df = pd.DataFrame(rows)

    # Log guardrail summary
    total_excluded = (guardrail_stats["excluded_home_insufficient"] +
                      guardrail_stats["excluded_away_insufficient"] +
                      guardrail_stats["excluded_both_insufficient"] +
                      guardrail_stats["excluded_coldstart"])

    logger.info(f"Guardrail summary:")
    logger.info(f"  Total candidates: {guardrail_stats['total_candidates']}")
    logger.info(f"  Excluded (home insufficient): {guardrail_stats['excluded_home_insufficient']}")
    logger.info(f"  Excluded (away insufficient): {guardrail_stats['excluded_away_insufficient']}")
    logger.info(f"  Excluded (both insufficient): {guardrail_stats['excluded_both_insufficient']}")
    logger.info(f"  Excluded (cold-start): {guardrail_stats['excluded_coldstart']}")
    logger.info(f"  Included: {guardrail_stats['included']}")
    logger.info(f"  Exclusion rate: {total_excluded/guardrail_stats['total_candidates']*100:.1f}%")

    return df, guardrail_stats


def calculate_team_features(team_id: int, history: list, prefix: str) -> dict:
    """Calculate rolling average features for a team."""
    if not history:
        return {
            f"{prefix}_goals_scored_avg": 0.0,
            f"{prefix}_goals_conceded_avg": 0.0,
            f"{prefix}_shots_avg": 0.0,
            f"{prefix}_corners_avg": 0.0,
            f"{prefix}_rest_days": 7.0,
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
        f"{prefix}_rest_days": 7.0,
        f"{prefix}_matches_played": len(history),
    }


def calculate_rest_days(history: list, match_date: datetime) -> float:
    """Calculate days since last match."""
    if not history:
        return 7.0

    last_match_date = history[0]['date']
    if isinstance(last_match_date, str):
        last_match_date = datetime.fromisoformat(last_match_date.replace('Z', '+00:00'))

    if hasattr(match_date, 'tzinfo') and match_date.tzinfo is not None:
        if hasattr(last_match_date, 'tzinfo') and last_match_date.tzinfo is None:
            last_match_date = last_match_date.replace(tzinfo=match_date.tzinfo)

    delta = match_date - last_match_date
    return max(1.0, min(30.0, delta.days))


async def train_variant(variant: str, cutoff: str, exclude_coldstart: bool = False,
                        draw_weight: float = 1.5) -> dict:
    """Train a single variant."""
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant: {variant}. Choose from {list(VARIANTS.keys())}")

    config = VARIANTS[variant]
    min_date = config["min_date"]
    model_version = config["model_version"]

    logger.info(f"=" * 60)
    logger.info(f"Training variant {variant}: {config['description']}")
    logger.info(f"  min_date: {min_date}")
    logger.info(f"  cutoff: {cutoff}")
    logger.info(f"  model_version: {model_version}")
    logger.info(f"  exclude_coldstart: {exclude_coldstart}")
    logger.info(f"=" * 60)

    database_url = os.environ.get('DATABASE_URL')
    if not database_url:
        raise ValueError("DATABASE_URL required")

    if database_url.startswith('postgresql://'):
        database_url = database_url.replace('postgresql://', 'postgresql+asyncpg://')

    engine = create_async_engine(database_url)
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    cutoff_date = datetime.fromisoformat(cutoff)
    min_date_dt = datetime.fromisoformat(min_date)

    async with async_session() as session:
        # Build dataset with guardrails
        df, guardrail_stats = await build_features_with_guardrails(
            session, min_date_dt, cutoff_date, exclude_coldstart
        )

        if len(df) == 0:
            return {"error": "No training data after guardrails", "guardrail_stats": guardrail_stats}

        logger.info(f"Dataset: {len(df)} samples")
        logger.info(f"Date range: {df['date'].min()} to {df['date'].max()}")

        # Verify features
        missing = [f for f in ORIGINAL_14_FEATURES if f not in df.columns]
        if missing:
            raise ValueError(f"Missing features: {missing}")

        # Prepare training data
        X = df[ORIGINAL_14_FEATURES].fillna(0).values
        y = df['result'].values

        # Class distribution
        n_home = (y == 0).sum()
        n_draw = (y == 1).sum()
        n_away = (y == 2).sum()
        logger.info(f"Class distribution: H={n_home} ({n_home/len(y):.1%}), "
                    f"D={n_draw} ({n_draw/len(y):.1%}), A={n_away} ({n_away/len(y):.1%})")

        # Feature statistics (for drift analysis)
        feature_stats = {}
        for f in ORIGINAL_14_FEATURES:
            feature_stats[f] = {
                "mean": float(df[f].mean()),
                "std": float(df[f].std()),
                "min": float(df[f].min()),
                "max": float(df[f].max()),
            }

        # Sample weights
        sample_weight = np.ones(len(y), dtype=np.float32)
        sample_weight[y == 1] = draw_weight

        # Cross-validation
        n_splits = 3
        tscv = TimeSeriesSplit(n_splits=n_splits)
        cv_scores = []

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            sw_train = sample_weight[train_idx]

            model = xgb.XGBClassifier(**HYPERPARAMS)
            model.fit(X_train, y_train, sample_weight=sw_train, verbose=False)

            y_proba = model.predict_proba(X_val)
            brier = calculate_brier_score(y_val, y_proba)
            cv_scores.append(brier)
            logger.info(f"  Fold {fold+1}: Brier={brier:.4f}")

        avg_brier = np.mean(cv_scores)
        logger.info(f"CV Brier: {avg_brier:.4f} (+/- {np.std(cv_scores):.4f})")

        # Train final model on all data
        final_model = xgb.XGBClassifier(**HYPERPARAMS)
        final_model.fit(X, y, sample_weight=sample_weight, verbose=False)

        # Feature importance
        importance = dict(zip(ORIGINAL_14_FEATURES,
                              [float(x) for x in final_model.feature_importances_]))

        # Save model
        timestamp = datetime.now().strftime("%Y%m%d")
        model_path = f"models/xgb_{model_version}_{timestamp}.json"
        final_model.save_model(model_path)
        logger.info(f"Model saved: {model_path}")

        result = {
            "variant": variant,
            "model_version": model_version,
            "model_path": model_path,
            "min_date": min_date,
            "cutoff": cutoff,
            "exclude_coldstart": exclude_coldstart,
            "samples_trained": len(df),
            "cv_brier": round(avg_brier, 4),
            "cv_scores": [round(s, 4) for s in cv_scores],
            "class_distribution": {
                "home": int(n_home),
                "draw": int(n_draw),
                "away": int(n_away),
            },
            "guardrail_stats": guardrail_stats,
            "feature_stats": feature_stats,
            "feature_importance": importance,
        }

        return result

    await engine.dispose()


def main():
    parser = argparse.ArgumentParser(description="Train extended range models")
    parser.add_argument('--variant', required=True, choices=['A', 'B', 'C', 'all'],
                        help='Variant to train (A/B/C or all)')
    parser.add_argument('--cutoff', required=True, help='Training cutoff date (ISO)')
    parser.add_argument('--exclude-coldstart', action='store_true',
                        help='Exclude Aug-Sep 2023 to avoid cold-start')
    parser.add_argument('--draw-weight', type=float, default=1.5,
                        help='Sample weight for draws')
    args = parser.parse_args()

    if args.variant == 'all':
        variants = ['A', 'B', 'C']
    else:
        variants = [args.variant]

    results = []
    for v in variants:
        result = asyncio.run(train_variant(
            variant=v,
            cutoff=args.cutoff,
            exclude_coldstart=args.exclude_coldstart,
            draw_weight=args.draw_weight
        ))
        results.append(result)

        # Print summary
        print(f"\n{'='*60}")
        print(f"VARIANT {v} COMPLETE")
        print(f"{'='*60}")
        print(f"  Model: {result.get('model_path', 'N/A')}")
        print(f"  Samples: {result.get('samples_trained', 'N/A')}")
        print(f"  CV Brier: {result.get('cv_brier', 'N/A')}")
        print(f"  Guardrails excluded: {result.get('guardrail_stats', {}).get('total_candidates', 0) - result.get('samples_trained', 0)}")

    # Save results to JSON
    import json
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"logs/train_extended_range_{timestamp}.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
