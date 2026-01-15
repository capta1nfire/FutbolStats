#!/usr/bin/env python3
"""
FASE 1 Complete Evaluation - Before/After Comparison

Compares v1.0.0 (baseline) vs v1.1.0 (FASE 1) with:
- Accuracy global
- Brier score global and per-class
- Log loss
- Confusion matrix 3x3
- Precision/Recall/F1 per class
- Temporal validation (last 30/90 days)

Run with: python scripts/eval_fase1_complete.py
"""

import asyncio
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Force production database
os.environ["DATABASE_URL"] = "postgresql://postgres:hzvozcXijUpblVrQshuowYcEGwZnMrfO@maglev.proxy.rlwy.net:24997/railway"

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    log_loss,
    precision_recall_fscore_support,
)
from sklearn.model_selection import TimeSeriesSplit

from app.database import get_async_session
from app.features.engineering import FeatureEngineer

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Feature columns for each version
FEATURES_V100 = [
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

FEATURES_V110 = FEATURES_V100 + [
    "abs_attack_diff",
    "abs_defense_diff",
    "abs_strength_gap",
]

# XGBoost params (same for both)
PARAMS = {
    "objective": "multi:softprob",
    "num_class": 3,
    "max_depth": 3,
    "learning_rate": 0.0283,
    "n_estimators": 114,
    "min_child_weight": 7,
    "subsample": 0.72,
    "colsample_bytree": 0.71,
    "random_state": 42,
    "use_label_encoder": False,
}


def prepare_features(df: pd.DataFrame, feature_cols: list) -> np.ndarray:
    """Extract features from DataFrame."""
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0
    return df[feature_cols].fillna(0).values


def calculate_brier_multiclass(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """Calculate multiclass Brier score."""
    n_classes = y_proba.shape[1]
    brier = 0.0
    for c in range(n_classes):
        y_true_binary = (y_true == c).astype(int)
        brier += brier_score_loss(y_true_binary, y_proba[:, c])
    return brier / n_classes


def calculate_brier_per_class(y_true: np.ndarray, y_proba: np.ndarray) -> dict:
    """Calculate Brier score per class."""
    classes = ["home", "draw", "away"]
    result = {}
    for c in range(3):
        y_true_binary = (y_true == c).astype(int)
        result[classes[c]] = brier_score_loss(y_true_binary, y_proba[:, c])
    return result


def evaluate_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    version: str,
    draw_weight: float = 1.0,
) -> dict:
    """Train and evaluate a model version."""

    # Create sample weights
    sample_weight = np.ones(len(y_train), dtype=np.float32)
    if draw_weight != 1.0:
        sample_weight[y_train == 1] = draw_weight

    # Train
    model = xgb.XGBClassifier(**PARAMS)
    model.fit(X_train, y_train, sample_weight=sample_weight, verbose=False)

    # Predict
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    logloss = log_loss(y_test, y_proba)
    brier_global = calculate_brier_multiclass(y_test, y_proba)
    brier_per_class = calculate_brier_per_class(y_test, y_proba)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, y_pred, labels=[0, 1, 2], zero_division=0
    )

    # Draw-specific
    draw_predictions = (y_pred == 1).sum()
    draw_pct = draw_predictions / len(y_pred)

    return {
        "version": version,
        "accuracy": acc,
        "log_loss": logloss,
        "brier_global": brier_global,
        "brier_per_class": brier_per_class,
        "confusion_matrix": cm,
        "precision": {"home": precision[0], "draw": precision[1], "away": precision[2]},
        "recall": {"home": recall[0], "draw": recall[1], "away": recall[2]},
        "f1": {"home": f1[0], "draw": f1[1], "away": f1[2]},
        "support": {"home": int(support[0]), "draw": int(support[1]), "away": int(support[2])},
        "draw_predictions": int(draw_predictions),
        "draw_pct": draw_pct,
        "y_proba": y_proba,
        "y_pred": y_pred,
    }


def print_comparison(v100: dict, v110: dict, period: str):
    """Print comparison table."""
    print(f"\n{'='*70}")
    print(f"COMPARACIÓN {period}")
    print(f"{'='*70}")

    print(f"\n{'Métrica':<25} {'v1.0.0':>15} {'v1.1.0':>15} {'Delta':>15}")
    print("-" * 70)

    # Global metrics
    delta_acc = v110["accuracy"] - v100["accuracy"]
    print(f"{'Accuracy':<25} {v100['accuracy']:>15.4f} {v110['accuracy']:>15.4f} {delta_acc:>+15.4f}")

    delta_ll = v110["log_loss"] - v100["log_loss"]
    print(f"{'Log Loss':<25} {v100['log_loss']:>15.4f} {v110['log_loss']:>15.4f} {delta_ll:>+15.4f}")

    delta_brier = v110["brier_global"] - v100["brier_global"]
    print(f"{'Brier (global)':<25} {v100['brier_global']:>15.4f} {v110['brier_global']:>15.4f} {delta_brier:>+15.4f}")

    # Brier per class
    print(f"\n{'Brier por clase:':<25}")
    for cls in ["home", "draw", "away"]:
        delta = v110["brier_per_class"][cls] - v100["brier_per_class"][cls]
        print(f"  {cls:<23} {v100['brier_per_class'][cls]:>15.4f} {v110['brier_per_class'][cls]:>15.4f} {delta:>+15.4f}")

    # Precision per class
    print(f"\n{'Precision por clase:':<25}")
    for cls in ["home", "draw", "away"]:
        delta = v110["precision"][cls] - v100["precision"][cls]
        print(f"  {cls:<23} {v100['precision'][cls]:>15.4f} {v110['precision'][cls]:>15.4f} {delta:>+15.4f}")

    # Recall per class
    print(f"\n{'Recall por clase:':<25}")
    for cls in ["home", "draw", "away"]:
        delta = v110["recall"][cls] - v100["recall"][cls]
        print(f"  {cls:<23} {v100['recall'][cls]:>15.4f} {v110['recall'][cls]:>15.4f} {delta:>+15.4f}")

    # F1 per class
    print(f"\n{'F1 por clase:':<25}")
    for cls in ["home", "draw", "away"]:
        delta = v110["f1"][cls] - v100["f1"][cls]
        print(f"  {cls:<23} {v100['f1'][cls]:>15.4f} {v110['f1'][cls]:>15.4f} {delta:>+15.4f}")

    # Draw predictions
    print(f"\n{'Draw predictions:':<25}")
    print(f"  {'count':<23} {v100['draw_predictions']:>15} {v110['draw_predictions']:>15} {v110['draw_predictions']-v100['draw_predictions']:>+15}")
    print(f"  {'pct':<23} {v100['draw_pct']:>15.1%} {v110['draw_pct']:>15.1%} {(v110['draw_pct']-v100['draw_pct'])*100:>+15.1f}pp")

    # Confusion matrices
    print(f"\n{'Confusion Matrix v1.0.0:':<25}")
    print(f"  {'':>10} {'pred_H':>10} {'pred_D':>10} {'pred_A':>10}")
    labels = ["true_H", "true_D", "true_A"]
    for i, label in enumerate(labels):
        print(f"  {label:>10} {v100['confusion_matrix'][i][0]:>10} {v100['confusion_matrix'][i][1]:>10} {v100['confusion_matrix'][i][2]:>10}")

    print(f"\n{'Confusion Matrix v1.1.0:':<25}")
    print(f"  {'':>10} {'pred_H':>10} {'pred_D':>10} {'pred_A':>10}")
    for i, label in enumerate(labels):
        print(f"  {label:>10} {v110['confusion_matrix'][i][0]:>10} {v110['confusion_matrix'][i][1]:>10} {v110['confusion_matrix'][i][2]:>10}")


async def main():
    print("=" * 70)
    print("FASE 1 COMPLETE EVALUATION: v1.0.0 vs v1.1.0")
    print("=" * 70)
    print(f"Database: PostgreSQL (producción)")
    print(f"Timestamp: {datetime.utcnow().isoformat()}")

    # Get data
    async for session in get_async_session():
        fe = FeatureEngineer(session)
        print("\nBuilding training dataset...")
        df = await fe.build_training_dataset()
        break

    print(f"Total samples: {len(df)}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Class distribution: {df['result'].value_counts().to_dict()}")

    # Sort by date for temporal split
    df = df.sort_values("date").reset_index(drop=True)

    # Prepare features
    X_v100 = prepare_features(df.copy(), FEATURES_V100)
    X_v110 = prepare_features(df.copy(), FEATURES_V110)
    y = df["result"].values
    dates = df["date"].values

    # === EVALUATION 1: TimeSeriesSplit (3 folds) ===
    print("\n" + "=" * 70)
    print("EVALUACIÓN 1: TimeSeriesSplit (3 folds)")
    print("=" * 70)

    tscv = TimeSeriesSplit(n_splits=3)

    all_results_v100 = []
    all_results_v110 = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X_v100)):
        print(f"\nFold {fold + 1}:")
        print(f"  Train: {len(train_idx)} samples")
        print(f"  Test: {len(test_idx)} samples")

        # v1.0.0
        r100 = evaluate_model(
            X_v100[train_idx], y[train_idx],
            X_v100[test_idx], y[test_idx],
            "v1.0.0", draw_weight=1.0
        )
        all_results_v100.append(r100)

        # v1.1.0
        r110 = evaluate_model(
            X_v110[train_idx], y[train_idx],
            X_v110[test_idx], y[test_idx],
            "v1.1.0", draw_weight=1.5
        )
        all_results_v110.append(r110)

        print(f"  v1.0.0: acc={r100['accuracy']:.4f}, draw_pred={r100['draw_predictions']}")
        print(f"  v1.1.0: acc={r110['accuracy']:.4f}, draw_pred={r110['draw_predictions']}")

    # === EVALUATION 2: Temporal Holdout (last 90 days) ===
    print("\n" + "=" * 70)
    print("EVALUACIÓN 2: Holdout Temporal (últimos 90 días)")
    print("=" * 70)

    cutoff_90 = datetime.utcnow() - timedelta(days=90)
    cutoff_30 = datetime.utcnow() - timedelta(days=30)

    # Convert dates for comparison
    df["date_dt"] = pd.to_datetime(df["date"])

    train_mask = df["date_dt"] < cutoff_90
    test_90_mask = df["date_dt"] >= cutoff_90
    test_30_mask = df["date_dt"] >= cutoff_30

    train_idx = df[train_mask].index.values
    test_90_idx = df[test_90_mask].index.values
    test_30_idx = df[test_30_mask].index.values

    print(f"\nTrain: {len(train_idx)} samples (hasta {cutoff_90.date()})")
    print(f"Test 90d: {len(test_90_idx)} samples")
    print(f"Test 30d: {len(test_30_idx)} samples")

    if len(test_90_idx) > 0:
        # Last 90 days
        v100_90 = evaluate_model(
            X_v100[train_idx], y[train_idx],
            X_v100[test_90_idx], y[test_90_idx],
            "v1.0.0", draw_weight=1.0
        )
        v110_90 = evaluate_model(
            X_v110[train_idx], y[train_idx],
            X_v110[test_90_idx], y[test_90_idx],
            "v1.1.0", draw_weight=1.5
        )
        print_comparison(v100_90, v110_90, "ÚLTIMOS 90 DÍAS")

    if len(test_30_idx) > 0:
        # Last 30 days
        v100_30 = evaluate_model(
            X_v100[train_idx], y[train_idx],
            X_v100[test_30_idx], y[test_30_idx],
            "v1.0.0", draw_weight=1.0
        )
        v110_30 = evaluate_model(
            X_v110[train_idx], y[train_idx],
            X_v110[test_30_idx], y[test_30_idx],
            "v1.1.0", draw_weight=1.5
        )
        print_comparison(v100_30, v110_30, "ÚLTIMOS 30 DÍAS")

    # === SUMMARY ===
    print("\n" + "=" * 70)
    print("RESUMEN EJECUTIVO PARA AUDITOR")
    print("=" * 70)

    print(f"""
Draw Label Canónico: DEFINIDO
- Draw = home_goals == away_goals donde status = 'FT'
- Draw rate histórico: {(y == 1).sum()}/{len(y)} = {(y == 1).mean():.1%}

Evaluación en últimos 90 días (holdout temporal):
""")

    if len(test_90_idx) > 0:
        print(f"  v1.0.0 → v1.1.0:")
        print(f"  - Accuracy: {v100_90['accuracy']:.4f} → {v110_90['accuracy']:.4f} ({v110_90['accuracy']-v100_90['accuracy']:+.4f})")
        print(f"  - Brier global: {v100_90['brier_global']:.4f} → {v110_90['brier_global']:.4f} ({v110_90['brier_global']-v100_90['brier_global']:+.4f})")
        print(f"  - Draw predictions: {v100_90['draw_pct']:.1%} → {v110_90['draw_pct']:.1%}")
        print(f"  - Draw precision: {v100_90['precision']['draw']:.4f} → {v110_90['precision']['draw']:.4f}")
        print(f"  - Home precision: {v100_90['precision']['home']:.4f} → {v110_90['precision']['home']:.4f}")
        print(f"  - Away precision: {v100_90['precision']['away']:.4f} → {v110_90['precision']['away']:.4f}")

        # Verdict
        print(f"\nVEREDICTO:")
        acc_ok = v110_90['accuracy'] >= v100_90['accuracy'] - 0.01  # Allow 1% drop
        brier_ok = v110_90['brier_global'] <= v100_90['brier_global'] + 0.005
        home_ok = v110_90['precision']['home'] >= v100_90['precision']['home'] - 0.02
        draw_improved = v110_90['draw_pct'] > 0.05  # At least 5% draw predictions

        if acc_ok and brier_ok and home_ok and draw_improved:
            print("  ✅ GO - Modelo v1.1.0 listo para producción")
        else:
            issues = []
            if not acc_ok:
                issues.append("accuracy degradada >1%")
            if not brier_ok:
                issues.append("brier degradado >0.005")
            if not home_ok:
                issues.append("home precision degradada >2%")
            if not draw_improved:
                issues.append("draw predictions <5%")
            print(f"  ⚠️ REVISAR - Issues: {', '.join(issues)}")


if __name__ == "__main__":
    asyncio.run(main())
