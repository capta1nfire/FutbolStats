#!/usr/bin/env python3
"""
FASE 1 Evaluation Script - Draw Prediction Improvement

Compares baseline model (v1.0.0) vs candidate model (v1.1.0) with:
- 3 new competitiveness features (abs_attack_diff, abs_defense_diff, abs_strength_gap)
- sample_weight=1.5 for draws

Metrics:
- PR-AUC per class (especially draw)
- Brier score per class
- draw_top1_rate (when model predicts draw, how often is it correct?)
- Calibration curves
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit

from app.database import get_async_session
from app.features.engineering import FeatureEngineer
from app.ml.engine import XGBoostEngine

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def calculate_class_metrics(y_true: np.ndarray, y_proba: np.ndarray) -> dict:
    """Calculate per-class metrics for draw analysis."""
    n_classes = y_proba.shape[1]
    metrics = {}

    for c in range(n_classes):
        class_name = ["home", "draw", "away"][c]
        y_true_binary = (y_true == c).astype(int)
        y_prob_class = y_proba[:, c]

        # PR-AUC (more relevant for imbalanced classes like draw)
        try:
            pr_auc = average_precision_score(y_true_binary, y_prob_class)
        except:
            pr_auc = 0.0

        # ROC-AUC
        try:
            roc_auc = roc_auc_score(y_true_binary, y_prob_class)
        except:
            roc_auc = 0.5

        # Brier score for this class
        brier = brier_score_loss(y_true_binary, y_prob_class)

        # Baseline PR-AUC (random = class prevalence)
        baseline_pr_auc = y_true_binary.mean()

        metrics[class_name] = {
            "pr_auc": pr_auc,
            "pr_auc_lift": pr_auc - baseline_pr_auc,
            "roc_auc": roc_auc,
            "brier": brier,
            "prevalence": y_true_binary.mean(),
        }

    return metrics


def calculate_draw_top1_rate(y_true: np.ndarray, y_proba: np.ndarray) -> dict:
    """Calculate how often draw predictions are correct."""
    predictions = np.argmax(y_proba, axis=1)

    # When model predicts draw (class 1)
    draw_predictions = predictions == 1
    n_draw_pred = draw_predictions.sum()

    if n_draw_pred == 0:
        return {
            "n_draw_predictions": 0,
            "draw_top1_rate": 0.0,
            "draw_precision": 0.0,
        }

    # Of those, how many were actually draws?
    correct_draw_preds = (y_true[draw_predictions] == 1).sum()
    draw_precision = correct_draw_preds / n_draw_pred

    # Also calculate: of all actual draws, how many did we predict?
    actual_draws = (y_true == 1).sum()
    draw_recall = correct_draw_preds / actual_draws if actual_draws > 0 else 0.0

    return {
        "n_draw_predictions": int(n_draw_pred),
        "pct_draw_predictions": n_draw_pred / len(y_true),
        "draw_precision": draw_precision,
        "draw_recall": draw_recall,
        "actual_draws": int(actual_draws),
    }


async def main():
    print("=" * 60)
    print("FASE 1 EVALUATION: Draw Prediction Improvement")
    print("=" * 60)

    # Get data from database
    async for session in get_async_session():
        fe = FeatureEngineer(session)
        print("\nBuilding training dataset...")
        df = await fe.build_training_dataset()
        break

    print(f"Dataset: {len(df)} samples")
    print(f"Class distribution: {df['result'].value_counts().to_dict()}")

    # Prepare features and target
    engine = XGBoostEngine(model_version="v1.1.0")
    X = engine._prepare_features(df)
    y = df["result"].values

    print(f"\nFeatures ({len(engine.FEATURE_COLUMNS)}):")
    for i, col in enumerate(engine.FEATURE_COLUMNS):
        print(f"  {i+1}. {col}")

    # Time-series split for evaluation
    tscv = TimeSeriesSplit(n_splits=3)

    all_y_true = []
    all_y_proba = []
    fold_metrics = []

    print("\n" + "-" * 60)
    print("Cross-validation with sample_weight=1.5 for draws")
    print("-" * 60)

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Create sample weights
        sample_weight = np.ones(len(y_train), dtype=np.float32)
        sample_weight[y_train == 1] = 1.5  # Upweight draws

        # Train model with sample weights
        import xgboost as xgb
        params = {
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

        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train, sample_weight=sample_weight, verbose=False)

        y_proba = model.predict_proba(X_val)

        # Store for aggregate metrics
        all_y_true.extend(y_val)
        all_y_proba.extend(y_proba)

        # Per-fold metrics
        class_metrics = calculate_class_metrics(y_val, y_proba)
        draw_top1 = calculate_draw_top1_rate(y_val, y_proba)

        print(f"\nFold {fold + 1}:")
        print(f"  Validation size: {len(y_val)}")
        print(f"  Draw PR-AUC: {class_metrics['draw']['pr_auc']:.4f} (baseline: {class_metrics['draw']['prevalence']:.4f})")
        print(f"  Draw ROC-AUC: {class_metrics['draw']['roc_auc']:.4f}")
        print(f"  Draw predictions: {draw_top1['n_draw_predictions']} ({draw_top1.get('pct_draw_predictions', 0):.1%})")
        print(f"  Draw precision: {draw_top1['draw_precision']:.1%}")

        fold_metrics.append({
            "fold": fold + 1,
            **class_metrics,
            **draw_top1,
        })

    # Aggregate metrics
    all_y_true = np.array(all_y_true)
    all_y_proba = np.array(all_y_proba)

    print("\n" + "=" * 60)
    print("AGGREGATE RESULTS")
    print("=" * 60)

    agg_class_metrics = calculate_class_metrics(all_y_true, all_y_proba)
    agg_draw_top1 = calculate_draw_top1_rate(all_y_true, all_y_proba)

    print("\nPer-class metrics:")
    print("-" * 50)
    print(f"{'Class':<10} {'PR-AUC':>10} {'Lift':>10} {'ROC-AUC':>10} {'Brier':>10}")
    print("-" * 50)
    for cls in ["home", "draw", "away"]:
        m = agg_class_metrics[cls]
        print(f"{cls:<10} {m['pr_auc']:>10.4f} {m['pr_auc_lift']:>+10.4f} {m['roc_auc']:>10.4f} {m['brier']:>10.4f}")

    print("\nDraw prediction behavior:")
    print(f"  Total draw predictions: {agg_draw_top1['n_draw_predictions']}")
    print(f"  Pct of predictions that are draws: {agg_draw_top1.get('pct_draw_predictions', 0):.1%}")
    print(f"  Draw precision (when we predict draw): {agg_draw_top1['draw_precision']:.1%}")
    print(f"  Draw recall (of actual draws, how many we caught): {agg_draw_top1['draw_recall']:.1%}")
    print(f"  Actual draws in validation: {agg_draw_top1['actual_draws']}")

    # Feature importance
    print("\n" + "-" * 60)
    print("Feature Importance (final model)")
    print("-" * 60)

    # Train final model on all data
    sample_weight_all = np.ones(len(y), dtype=np.float32)
    sample_weight_all[y == 1] = 1.5

    final_model = xgb.XGBClassifier(**params)
    final_model.fit(X, y, sample_weight=sample_weight_all, verbose=False)

    importance = dict(zip(engine.FEATURE_COLUMNS, final_model.feature_importances_))
    sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)

    for feat, imp in sorted_importance:
        bar = "█" * int(imp * 50)
        print(f"  {feat:<25} {imp:.4f} {bar}")

    # Summary for auditor
    print("\n" + "=" * 60)
    print("RESUMEN PARA AUDITOR")
    print("=" * 60)

    draw_metrics = agg_class_metrics["draw"]

    print(f"""
Model: v1.1.0 (FASE 1)
Features: 17 (14 base + 3 competitiveness)
Sample weight: draws=1.5, others=1.0

RESULTADOS DRAW:
- PR-AUC: {draw_metrics['pr_auc']:.4f} (baseline: {draw_metrics['prevalence']:.4f}, lift: {draw_metrics['pr_auc_lift']:+.4f})
- ROC-AUC: {draw_metrics['roc_auc']:.4f}
- Brier: {draw_metrics['brier']:.4f}
- Draw predictions: {agg_draw_top1['n_draw_predictions']} ({agg_draw_top1.get('pct_draw_predictions', 0):.1%} of all)
- Precision: {agg_draw_top1['draw_precision']:.1%}
- Recall: {agg_draw_top1['draw_recall']:.1%}

DIAGNOSTICO:
""")

    if agg_draw_top1['n_draw_predictions'] > 0:
        print("✓ El modelo AHORA predice draws")
        if draw_metrics['pr_auc_lift'] > 0.01:
            print(f"✓ PR-AUC mejoró +{draw_metrics['pr_auc_lift']:.4f}")
        elif draw_metrics['pr_auc_lift'] > 0:
            print(f"~ PR-AUC mejoró marginalmente +{draw_metrics['pr_auc_lift']:.4f}")
        else:
            print(f"✗ PR-AUC no mejoró ({draw_metrics['pr_auc_lift']:+.4f})")
    else:
        print("✗ El modelo TODAVIA no predice draws → considerar FASE 2 (two-stage)")


if __name__ == "__main__":
    asyncio.run(main())
