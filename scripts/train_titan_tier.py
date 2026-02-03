#!/usr/bin/env python3
"""
Train XGBoost model for a specific TITAN tier.

Usage:
    python scripts/train_titan_tier.py --tier baseline --cutoff 2026-01-06
    python scripts/train_titan_tier.py --tier T1b --cutoff 2026-01-06

Input:
    data/train_{tier}_{cutoff}.parquet

Output:
    models/xgb_{tier}_{cutoff}.json
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import brier_score_loss, log_loss, accuracy_score

from app.ml.engine import XGBoostEngine

# Feature definitions (same as build_titan_dataset.py)
BASELINE_FEATURES = XGBoostEngine.FEATURE_COLUMNS

# PRODUCCIÓN v1.0.0: Exactamente 14 features (sin los 3 de FASE 1)
BASELINE_V1_FEATURES = [
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

# BUG #3 FIX: Added missing flags to distinguish real 0 vs imputed 0
TITAN_T1B_FEATURES = [
    "xg_home_last5", "xg_away_last5", "xga_home_last5", "xga_away_last5",
    "npxg_home_last5", "npxg_away_last5",
    "xg_home_missing", "xg_away_missing"  # 1 if <5 matches available
]
# T1b_v2: Differential features
TITAN_T1B_V2_FEATURES = [
    "xg_diff_last5",      # xg_home_last5 - xg_away_last5
    "xga_diff_last5",     # xga_home_last5 - xga_away_last5
    "npxg_diff_last5",    # npxg_home_last5 - npxg_away_last5
    "xg_net_home_last5",  # xg_home_last5 - xga_home_last5
    "xg_net_away_last5",  # xg_away_last5 - xga_away_last5
    "net_diff_last5",     # xg_net_home_last5 - xg_net_away_last5
    "xg_home_missing", "xg_away_missing"  # 1 if <5 matches available
]
TITAN_T1C_FEATURES = [
    "sofascore_lineup_integrity_score",
    "lineup_home_starters_count", "lineup_away_starters_count"
]
TITAN_T1D_FEATURES = [
    "xi_home_def_count", "xi_home_mid_count", "xi_home_fwd_count",
    "xi_away_def_count", "xi_away_mid_count", "xi_away_fwd_count",
    "xi_formation_mismatch_flag"
]

TIER_FEATURES = {
    "baseline": list(BASELINE_FEATURES),
    "T1b": list(BASELINE_FEATURES) + TITAN_T1B_FEATURES,
    "T1b_v2": list(BASELINE_FEATURES) + TITAN_T1B_V2_FEATURES,
    "T1c": list(BASELINE_FEATURES) + TITAN_T1C_FEATURES,
    "T1d": list(BASELINE_FEATURES) + TITAN_T1D_FEATURES,
    # v1 tiers: usan exactamente los 14 features de producción v1.0.0
    "baseline_v1": list(BASELINE_V1_FEATURES),
    "T1b_v2_on_v1": list(BASELINE_V1_FEATURES) + TITAN_T1B_V2_FEATURES,
}

# Label encoding: H=0, D=1, A=2
LABEL_MAP = {'H': 0, 'D': 1, 'A': 2}


def multiclass_brier(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """Calculate multiclass Brier score."""
    n_classes = y_proba.shape[1]
    y_onehot = np.zeros((len(y_true), n_classes))
    for i, label in enumerate(y_true):
        y_onehot[i, label] = 1
    return np.mean(np.sum((y_proba - y_onehot) ** 2, axis=1))


def train_model(tier: str, cutoff: str) -> dict:
    """Train XGBoost model for the specified tier."""

    # Load dataset
    input_path = Path(f"data/train_{tier}_{cutoff.replace('-', '')}.parquet")
    if not input_path.exists():
        raise FileNotFoundError(f"Dataset not found: {input_path}. Run build_titan_dataset.py first.")

    df = pd.read_parquet(input_path)
    print(f"Loaded dataset: {input_path}")
    print(f"Shape: {df.shape}")

    # Get features for this tier
    feature_cols = TIER_FEATURES[tier]
    print(f"Using {len(feature_cols)} features for tier {tier}")

    # Check all features exist
    missing = [col for col in feature_cols if col not in df.columns]
    if missing:
        print(f"Warning: Missing features (will use 0): {missing}")
        for col in missing:
            df[col] = 0.0

    # Prepare X, y
    X = df[feature_cols].fillna(0).values
    y = df['outcome'].map(LABEL_MAP).values

    print(f"X shape: {X.shape}, y shape: {y.shape}")
    print(f"Class distribution: H={sum(y==0)}, D={sum(y==1)}, A={sum(y==2)}")

    # TimeSeriesSplit for cross-validation (3 folds)
    tscv = TimeSeriesSplit(n_splits=3)
    cv_metrics = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Train XGBoost
        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            objective='multi:softprob',
            num_class=3,
            eval_metric='mlogloss',
            use_label_encoder=False,
            verbosity=0,
            random_state=42,
        )

        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        # Evaluate
        y_proba = model.predict_proba(X_val)
        y_pred = model.predict(X_val)

        brier = multiclass_brier(y_val, y_proba)
        logloss = log_loss(y_val, y_proba)
        acc = accuracy_score(y_val, y_pred)

        cv_metrics.append({
            'fold': fold + 1,
            'brier': brier,
            'logloss': logloss,
            'accuracy': acc,
            'n_train': len(train_idx),
            'n_val': len(val_idx),
        })

        print(f"Fold {fold+1}: brier={brier:.4f}, logloss={logloss:.4f}, acc={acc:.4f}")

    # Final model: train on all data
    print("\nTraining final model on all data...")
    final_model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        objective='multi:softprob',
        num_class=3,
        eval_metric='mlogloss',
        use_label_encoder=False,
        verbosity=0,
        random_state=42,
    )
    final_model.fit(X, y, verbose=False)

    # Save model (XGBoost JSON format for consistency with production)
    output_dir = Path('models')
    output_dir.mkdir(exist_ok=True)
    model_path = output_dir / f"xgb_{tier}_{cutoff.replace('-', '')}.json"
    final_model.save_model(str(model_path))

    # Also save as "latest" for easy reference
    latest_path = output_dir / f"xgb_{tier}_latest.json"
    final_model.save_model(str(latest_path))

    print(f"\nModel saved: {model_path}")
    print(f"Latest symlink: {latest_path}")

    # Save training metadata
    metadata = {
        'tier': tier,
        'cutoff': cutoff,
        'n_samples': len(df),
        'n_features': len(feature_cols),
        'features': feature_cols,
        'cv_metrics': cv_metrics,
        'cv_mean': {
            'brier': np.mean([m['brier'] for m in cv_metrics]),
            'logloss': np.mean([m['logloss'] for m in cv_metrics]),
            'accuracy': np.mean([m['accuracy'] for m in cv_metrics]),
        },
        'trained_at': datetime.utcnow().isoformat(),
    }

    metadata_path = output_dir / f"xgb_{tier}_{cutoff.replace('-', '')}_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"Metadata saved: {metadata_path}")

    return metadata


def main():
    parser = argparse.ArgumentParser(description="Train TITAN tier model")
    parser.add_argument('--tier', required=True, choices=list(TIER_FEATURES.keys()),
                        help='Tier to train: baseline, T1b, T1c, T1d')
    parser.add_argument('--cutoff', required=True,
                        help='Training cutoff date (ISO format, e.g., 2026-01-06)')
    args = parser.parse_args()

    print(f"Training model for tier={args.tier}, cutoff={args.cutoff}")
    print("=" * 60)

    metadata = train_model(args.tier, args.cutoff)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Tier: {metadata['tier']}")
    print(f"Samples: {metadata['n_samples']}")
    print(f"Features: {metadata['n_features']}")
    print(f"CV Mean Brier: {metadata['cv_mean']['brier']:.4f}")
    print(f"CV Mean LogLoss: {metadata['cv_mean']['logloss']:.4f}")
    print(f"CV Mean Accuracy: {metadata['cv_mean']['accuracy']:.4f}")


if __name__ == "__main__":
    main()
