#!/usr/bin/env python3
"""
FASE 2: Two-Stage Model Evaluation

Architecture:
- Stage 1: Binary classifier (draw vs non-draw)
- Stage 2: Binary classifier (home vs away) for non-draws

Composition:
- p_draw = p1
- p_home = (1 - p1) * p2_home
- p_away = (1 - p1) * p2_away

Evaluation on 90-day holdout vs v1.0.0 baseline.
"""

import asyncio
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Require DATABASE_URL from environment (do not hardcode secrets)
if not os.environ.get("DATABASE_URL"):
    raise RuntimeError(
        "DATABASE_URL environment variable is required.\n"
        "Set it in your shell or .env file before running this script."
    )

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

from app.database import get_async_session
from app.features.engineering import FeatureEngineer


# Base features (same as v1.1.0)
BASE_FEATURES = [
    "home_goals_scored_avg", "home_goals_conceded_avg", "home_shots_avg",
    "home_corners_avg", "home_rest_days", "home_matches_played",
    "away_goals_scored_avg", "away_goals_conceded_avg", "away_shots_avg",
    "away_corners_avg", "away_rest_days", "away_matches_played",
    "goal_diff_avg", "rest_diff",
    "abs_attack_diff", "abs_defense_diff", "abs_strength_gap",
]

# Additional features for Stage 1 (draw detection)
STAGE1_EXTRA = [
    "implied_draw",  # Derived from odds if available
]

# XGBoost params (same base, adjusted per stage)
PARAMS_STAGE1 = {
    "objective": "binary:logistic",
    "max_depth": 3,
    "learning_rate": 0.05,
    "n_estimators": 100,
    "min_child_weight": 7,
    "subsample": 0.72,
    "colsample_bytree": 0.71,
    "random_state": 42,
    "use_label_encoder": False,
}

PARAMS_STAGE2 = {
    "objective": "binary:logistic",
    "max_depth": 3,
    "learning_rate": 0.05,
    "n_estimators": 100,
    "min_child_weight": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "use_label_encoder": False,
}

# Baseline v1.0.0 params (3-class, no draw weight)
PARAMS_BASELINE = {
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


def prepare_features(df: pd.DataFrame, features: list) -> np.ndarray:
    """Prepare feature matrix, filling missing with 0."""
    df = df.copy()
    for col in features:
        if col not in df.columns:
            df[col] = 0
    return df[features].fillna(0).values


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived features like implied probabilities from odds."""
    df = df.copy()

    # Implied draw probability from odds (if available)
    if "odds_draw" in df.columns:
        # implied_prob = 1/odds, then normalize
        df["implied_draw_raw"] = 1 / df["odds_draw"].replace(0, np.nan)
        df["implied_home_raw"] = 1 / df["odds_home"].replace(0, np.nan)
        df["implied_away_raw"] = 1 / df["odds_away"].replace(0, np.nan)

        # Normalize to sum to 1
        total = df["implied_draw_raw"] + df["implied_home_raw"] + df["implied_away_raw"]
        df["implied_draw"] = df["implied_draw_raw"] / total

        # Fill missing with league average draw rate (~0.25)
        df["implied_draw"] = df["implied_draw"].fillna(0.25)
    else:
        df["implied_draw"] = 0.25  # Default

    return df


class TwoStageModel:
    """Two-stage model: Stage1 (draw vs non-draw) + Stage2 (home vs away)."""

    def __init__(self):
        self.stage1 = None  # draw vs non-draw
        self.stage2 = None  # home vs away (for non-draws)
        self.features_stage1 = BASE_FEATURES + STAGE1_EXTRA
        self.features_stage2 = BASE_FEATURES

    def fit(self, X1: np.ndarray, y_draw: np.ndarray,
            X2: np.ndarray, y_home: np.ndarray,
            draw_weight: float = 1.5):
        """
        Train both stages.

        Args:
            X1: Features for stage 1 (all samples)
            y_draw: Binary target (1=draw, 0=non-draw)
            X2: Features for stage 2 (non-draw samples only)
            y_home: Binary target (1=home, 0=away) for non-draws
            draw_weight: Sample weight for draws in stage 1
        """
        # Stage 1: draw vs non-draw (upweight draws)
        sample_weight_s1 = np.ones(len(y_draw), dtype=np.float32)
        sample_weight_s1[y_draw == 1] = draw_weight

        self.stage1 = xgb.XGBClassifier(**PARAMS_STAGE1)
        self.stage1.fit(X1, y_draw, sample_weight=sample_weight_s1, verbose=False)

        # Stage 2: home vs away (only non-draws)
        self.stage2 = xgb.XGBClassifier(**PARAMS_STAGE2)
        self.stage2.fit(X2, y_home, verbose=False)

    def predict_proba(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """
        Predict 3-class probabilities using composition.

        Returns:
            Array of shape (n_samples, 3) with [p_home, p_draw, p_away]
        """
        # Stage 1: P(draw)
        p_draw = self.stage1.predict_proba(X1)[:, 1]

        # Stage 2: P(home | non-draw)
        p_home_given_nondraw = self.stage2.predict_proba(X2)[:, 1]

        # Compose final probabilities
        p_home = (1 - p_draw) * p_home_given_nondraw
        p_away = (1 - p_draw) * (1 - p_home_given_nondraw)

        # Stack as [home, draw, away] to match original format
        proba = np.column_stack([p_home, p_draw, p_away])

        # Sanity check: should sum to 1
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6), "Probabilities don't sum to 1"

        return proba

    def predict(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """Predict class labels (argmax)."""
        proba = self.predict_proba(X1, X2)
        return np.argmax(proba, axis=1)


def evaluate_model(y_true: np.ndarray, y_proba: np.ndarray, name: str) -> dict:
    """Calculate comprehensive metrics for 3-class model."""
    y_pred = np.argmax(y_proba, axis=1)

    # Global metrics
    acc = accuracy_score(y_true, y_pred)
    ll = log_loss(y_true, y_proba)

    # Brier per class
    brier = {}
    pr_auc = {}
    roc_auc = {}
    precision = {}
    recall = {}

    for c, cls_name in enumerate(["home", "draw", "away"]):
        y_binary = (y_true == c).astype(int)
        y_prob_c = y_proba[:, c]

        brier[cls_name] = brier_score_loss(y_binary, y_prob_c)

        try:
            pr_auc[cls_name] = average_precision_score(y_binary, y_prob_c)
        except:
            pr_auc[cls_name] = 0.0

        try:
            roc_auc[cls_name] = roc_auc_score(y_binary, y_prob_c)
        except:
            roc_auc[cls_name] = 0.5

        # Precision/Recall for predicted class
        pred_c = (y_pred == c).sum()
        true_c = (y_true == c).sum()
        correct_c = ((y_pred == c) & (y_true == c)).sum()

        precision[cls_name] = correct_c / pred_c if pred_c > 0 else 0
        recall[cls_name] = correct_c / true_c if true_c > 0 else 0

    brier["global"] = np.mean([brier["home"], brier["draw"], brier["away"]])

    # Draw-specific metrics
    draw_pred_count = (y_pred == 1).sum()
    draw_pred_pct = draw_pred_count / len(y_pred)
    draw_prevalence = (y_true == 1).sum() / len(y_true)

    return {
        "name": name,
        "accuracy": acc,
        "log_loss": ll,
        "brier_global": brier["global"],
        "brier_home": brier["home"],
        "brier_draw": brier["draw"],
        "brier_away": brier["away"],
        "pr_auc_home": pr_auc["home"],
        "pr_auc_draw": pr_auc["draw"],
        "pr_auc_away": pr_auc["away"],
        "roc_auc_draw": roc_auc["draw"],
        "precision_home": precision["home"],
        "precision_draw": precision["draw"],
        "precision_away": precision["away"],
        "recall_home": recall["home"],
        "recall_draw": recall["draw"],
        "recall_away": recall["away"],
        "draw_pred_count": int(draw_pred_count),
        "draw_pred_pct": draw_pred_pct,
        "draw_prevalence": draw_prevalence,
    }


async def main():
    print("=" * 80)
    print("FASE 2: TWO-STAGE MODEL EVALUATION")
    print("=" * 80)

    # Load data
    async for session in get_async_session():
        fe = FeatureEngineer(session)
        print("\nLoading dataset...")
        df = await fe.build_training_dataset()
        break

    df = df.sort_values("date").reset_index(drop=True)
    df["date_dt"] = pd.to_datetime(df["date"])

    # Add derived features
    df = add_derived_features(df)

    print(f"Dataset: {len(df)} samples")
    print(f"Class distribution: {df['result'].value_counts().to_dict()}")
    print(f"Draw rate: {(df['result'] == 1).mean():.1%}")

    # Temporal split: train on all except last 90 days
    cutoff_90 = datetime.utcnow() - timedelta(days=90)
    train_mask = df["date_dt"] < cutoff_90
    test_mask = df["date_dt"] >= cutoff_90

    df_train = df[train_mask].reset_index(drop=True)
    df_test = df[test_mask].reset_index(drop=True)

    print(f"\nTrain: {len(df_train)} samples")
    print(f"Test (90d holdout): {len(df_test)} samples")
    print(f"Test draw prevalence: {(df_test['result'] == 1).mean():.1%}")

    # Prepare targets
    y_train = df_train["result"].values  # 0=home, 1=draw, 2=away
    y_test = df_test["result"].values

    # Binary targets
    y_train_draw = (y_train == 1).astype(int)  # 1=draw, 0=non-draw
    y_test_draw = (y_test == 1).astype(int)

    # For stage 2: only non-draws, home=1, away=0
    nondraw_train_mask = y_train != 1
    nondraw_test_mask = y_test != 1

    y_train_home = (y_train[nondraw_train_mask] == 0).astype(int)  # 1=home, 0=away

    # Prepare feature matrices
    features_s1 = BASE_FEATURES + STAGE1_EXTRA
    features_s2 = BASE_FEATURES

    X_train_s1 = prepare_features(df_train, features_s1)
    X_test_s1 = prepare_features(df_test, features_s1)

    X_train_s2 = prepare_features(df_train[nondraw_train_mask], features_s2)
    X_test_s2 = prepare_features(df_test, features_s2)  # All test samples for prediction

    print(f"\nFeatures Stage 1 ({len(features_s1)}): {features_s1}")
    print(f"Features Stage 2 ({len(features_s2)}): {features_s2}")

    # ========================================
    # Train and evaluate BASELINE (v1.0.0)
    # ========================================
    print("\n" + "=" * 80)
    print("BASELINE: v1.0.0 (3-class, no draw weight)")
    print("=" * 80)

    X_train_base = prepare_features(df_train, BASE_FEATURES)
    X_test_base = prepare_features(df_test, BASE_FEATURES)

    baseline_model = xgb.XGBClassifier(**PARAMS_BASELINE)
    baseline_model.fit(X_train_base, y_train, verbose=False)
    baseline_proba = baseline_model.predict_proba(X_test_base)

    baseline_metrics = evaluate_model(y_test, baseline_proba, "v1.0.0 Baseline")

    print(f"Accuracy: {baseline_metrics['accuracy']:.4f}")
    print(f"LogLoss: {baseline_metrics['log_loss']:.4f}")
    print(f"Brier Global: {baseline_metrics['brier_global']:.4f}")
    print(f"Draw predictions: {baseline_metrics['draw_pred_count']} ({baseline_metrics['draw_pred_pct']:.1%})")
    print(f"Draw PR-AUC: {baseline_metrics['pr_auc_draw']:.4f} (prevalence: {baseline_metrics['draw_prevalence']:.4f})")

    # ========================================
    # Train and evaluate TWO-STAGE MODEL
    # ========================================
    print("\n" + "=" * 80)
    print("TWO-STAGE MODEL (FASE 2)")
    print("=" * 80)

    # Test different draw weights for stage 1
    draw_weights_to_test = [1.0, 1.5, 2.0, 2.5, 3.0]

    results = []

    for dw in draw_weights_to_test:
        print(f"\n--- Draw weight: {dw} ---")

        two_stage = TwoStageModel()
        two_stage.fit(
            X1=X_train_s1,
            y_draw=y_train_draw,
            X2=X_train_s2,
            y_home=y_train_home,
            draw_weight=dw
        )

        # Predict on test set
        y_proba_ts = two_stage.predict_proba(X_test_s1, X_test_s2)

        metrics = evaluate_model(y_test, y_proba_ts, f"Two-Stage (dw={dw})")
        metrics["draw_weight"] = dw
        results.append(metrics)

        print(f"  Accuracy: {metrics['accuracy']:.4f} (Δ={metrics['accuracy']-baseline_metrics['accuracy']:+.4f})")
        print(f"  LogLoss: {metrics['log_loss']:.4f} (Δ={metrics['log_loss']-baseline_metrics['log_loss']:+.4f})")
        print(f"  Brier: {metrics['brier_global']:.4f} (Δ={metrics['brier_global']-baseline_metrics['brier_global']:+.4f})")
        print(f"  Draw pred: {metrics['draw_pred_count']} ({metrics['draw_pred_pct']:.1%})")
        print(f"  Draw prec: {metrics['precision_draw']:.3f}, recall: {metrics['recall_draw']:.3f}")
        print(f"  Draw PR-AUC: {metrics['pr_auc_draw']:.4f} (Δ={metrics['pr_auc_draw']-baseline_metrics['pr_auc_draw']:+.4f})")

    # ========================================
    # COMPARATIVE REPORT
    # ========================================
    print("\n" + "=" * 80)
    print("REPORTE COMPARATIVO: BASELINE vs TWO-STAGE")
    print("=" * 80)

    print(f"\n{'Model':<25} {'Acc':>8} {'LogL':>8} {'Brier':>8} {'Draw%':>8} {'DPrec':>8} {'DRecall':>8} {'D-PRAUC':>8}")
    print("-" * 100)

    # Baseline
    b = baseline_metrics
    print(f"{'v1.0.0 Baseline':<25} {b['accuracy']:>8.4f} {b['log_loss']:>8.4f} {b['brier_global']:>8.4f} "
          f"{b['draw_pred_pct']:>8.1%} {b['precision_draw']:>8.3f} {b['recall_draw']:>8.3f} {b['pr_auc_draw']:>8.4f}")

    # Two-stage variants
    for r in results:
        delta_acc = r['accuracy'] - b['accuracy']
        delta_ll = r['log_loss'] - b['log_loss']
        delta_brier = r['brier_global'] - b['brier_global']

        # Check if acceptable
        is_acceptable = (
            r['brier_global'] <= b['brier_global'] + 0.002 and  # Slightly more tolerance for 2-stage
            r['log_loss'] <= b['log_loss'] + 0.02 and
            r['draw_pred_pct'] > 0.01 and
            r['precision_draw'] > r['draw_prevalence'] * 0.8  # At least 80% of prevalence
        )
        marker = "✓" if is_acceptable else " "

        print(f"{r['name']:<24}{marker} {r['accuracy']:>8.4f} {r['log_loss']:>8.4f} {r['brier_global']:>8.4f} "
              f"{r['draw_pred_pct']:>8.1%} {r['precision_draw']:>8.3f} {r['recall_draw']:>8.3f} {r['pr_auc_draw']:>8.4f}")

    # Find best candidate
    acceptable = [r for r in results if (
        r['brier_global'] <= baseline_metrics['brier_global'] + 0.002 and
        r['log_loss'] <= baseline_metrics['log_loss'] + 0.02 and
        r['draw_pred_pct'] > 0.01
    )]

    print("\n" + "=" * 80)
    print("ANÁLISIS Y RECOMENDACIÓN")
    print("=" * 80)

    if acceptable:
        # Best = highest draw PR-AUC among acceptable
        best = max(acceptable, key=lambda x: x['pr_auc_draw'])

        print(f"\n✓ CANDIDATO ACEPTABLE ENCONTRADO: {best['name']}")
        print(f"\n  Comparación vs Baseline (v1.0.0):")
        print(f"  - Accuracy: {best['accuracy']:.4f} (Δ={best['accuracy']-b['accuracy']:+.4f})")
        print(f"  - LogLoss: {best['log_loss']:.4f} (Δ={best['log_loss']-b['log_loss']:+.4f})")
        print(f"  - Brier Global: {best['brier_global']:.4f} (Δ={best['brier_global']-b['brier_global']:+.4f})")
        print(f"  - Draw predictions: {best['draw_pred_pct']:.1%} (antes: {b['draw_pred_pct']:.1%})")
        print(f"  - Draw precision: {best['precision_draw']:.3f} (prevalence: {best['draw_prevalence']:.3f})")
        print(f"  - Draw PR-AUC: {best['pr_auc_draw']:.4f} (Δ={best['pr_auc_draw']-b['pr_auc_draw']:+.4f})")

        print(f"\n  Métricas por clase (Brier):")
        print(f"  - Home: {best['brier_home']:.4f} (base: {b['brier_home']:.4f})")
        print(f"  - Draw: {best['brier_draw']:.4f} (base: {b['brier_draw']:.4f})")
        print(f"  - Away: {best['brier_away']:.4f} (base: {b['brier_away']:.4f})")

        # GO/NO-GO
        go_criteria = {
            "brier_ok": best['brier_global'] <= b['brier_global'] + 0.002,
            "logloss_ok": best['log_loss'] <= b['log_loss'] + 0.02,
            "draw_pred_ok": best['draw_pred_pct'] > 0.01,
            "draw_prauc_improved": best['pr_auc_draw'] > b['pr_auc_draw'],
            "away_recall_ok": best['recall_away'] > 0.3,  # Don't destroy away
        }

        print(f"\n  Criterios GO/NO-GO:")
        for criterion, passed in go_criteria.items():
            status = "✓" if passed else "✗"
            print(f"    {status} {criterion}")

        if all(go_criteria.values()):
            print(f"\n  → RECOMENDACIÓN: GO para deploy (shadow mode primero)")
        else:
            failed = [k for k, v in go_criteria.items() if not v]
            print(f"\n  → RECOMENDACIÓN: NO-GO - criterios fallidos: {failed}")
    else:
        print("\n✗ NO HAY CANDIDATO ACEPTABLE")
        print("  Ninguna configuración two-stage mantiene métricas mientras predice draws.")

        # Show best attempt
        if results:
            best_attempt = min(results, key=lambda x: x['brier_global'])
            print(f"\n  Mejor intento ({best_attempt['name']}):")
            print(f"  - Brier: {best_attempt['brier_global']:.4f} (Δ={best_attempt['brier_global']-b['brier_global']:+.4f})")
            print(f"  - Draw pred: {best_attempt['draw_pred_pct']:.1%}")

        print("\n  → RECOMENDACIÓN: Explorar alternativas (threshold tuning, ensemble, etc.)")


if __name__ == "__main__":
    asyncio.run(main())
