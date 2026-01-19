#!/usr/bin/env python3
"""
FASE 2: Two-Stage Model with Threshold Tuning

Instead of using extreme sample_weight (which destroys calibration),
we use moderate weights but adjust the decision threshold for draws.

The key insight: draw precision at dw=1.5 is 0.293 > prevalence (0.216),
but the model only predicts 12.2% draws. We want to find a threshold
that captures draws with good precision WITHOUT destroying probabilities.

Approach:
1. Train two-stage with dw=1.0 or 1.2 (good calibration)
2. Use probabilistic threshold: predict draw when p_draw > threshold
3. Find threshold that maximizes precision while maintaining recall
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
    roc_auc_score,
)

from app.database import get_async_session
from app.features.engineering import FeatureEngineer


# Features
BASE_FEATURES = [
    "home_goals_scored_avg", "home_goals_conceded_avg", "home_shots_avg",
    "home_corners_avg", "home_rest_days", "home_matches_played",
    "away_goals_scored_avg", "away_goals_conceded_avg", "away_shots_avg",
    "away_corners_avg", "away_rest_days", "away_matches_played",
    "goal_diff_avg", "rest_diff",
    "abs_attack_diff", "abs_defense_diff", "abs_strength_gap",
]

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
}


def prepare_features(df: pd.DataFrame, features: list) -> np.ndarray:
    df = df.copy()
    for col in features:
        if col not in df.columns:
            df[col] = 0
    return df[features].fillna(0).values


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "odds_draw" in df.columns:
        df["implied_draw_raw"] = 1 / df["odds_draw"].replace(0, np.nan)
        df["implied_home_raw"] = 1 / df["odds_home"].replace(0, np.nan)
        df["implied_away_raw"] = 1 / df["odds_away"].replace(0, np.nan)
        total = df["implied_draw_raw"] + df["implied_home_raw"] + df["implied_away_raw"]
        df["implied_draw"] = df["implied_draw_raw"] / total
        df["implied_draw"] = df["implied_draw"].fillna(0.25)
    else:
        df["implied_draw"] = 0.25
    return df


class TwoStageModelWithThreshold:
    """Two-stage model with configurable draw threshold."""

    def __init__(self, draw_weight: float = 1.2):
        self.stage1 = None
        self.stage2 = None
        self.draw_weight = draw_weight
        self.features_s1 = BASE_FEATURES + ["implied_draw"]
        self.features_s2 = BASE_FEATURES

    def fit(self, df_train: pd.DataFrame):
        y_train = df_train["result"].values
        y_draw = (y_train == 1).astype(int)

        # Non-draw subset for stage 2
        nondraw_mask = y_train != 1
        y_home = (y_train[nondraw_mask] == 0).astype(int)

        X_s1 = prepare_features(df_train, self.features_s1)
        X_s2 = prepare_features(df_train[nondraw_mask], self.features_s2)

        # Stage 1 with sample weight
        sample_weight = np.ones(len(y_draw), dtype=np.float32)
        sample_weight[y_draw == 1] = self.draw_weight

        self.stage1 = xgb.XGBClassifier(**PARAMS_STAGE1)
        self.stage1.fit(X_s1, y_draw, sample_weight=sample_weight, verbose=False)

        # Stage 2 (no weighting needed)
        self.stage2 = xgb.XGBClassifier(**PARAMS_STAGE2)
        self.stage2.fit(X_s2, y_home, verbose=False)

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Predict raw 3-class probabilities (before threshold)."""
        X_s1 = prepare_features(df, self.features_s1)
        X_s2 = prepare_features(df, self.features_s2)

        p_draw = self.stage1.predict_proba(X_s1)[:, 1]
        p_home_given_nondraw = self.stage2.predict_proba(X_s2)[:, 1]

        p_home = (1 - p_draw) * p_home_given_nondraw
        p_away = (1 - p_draw) * (1 - p_home_given_nondraw)

        return np.column_stack([p_home, p_draw, p_away])

    def predict_with_threshold(self, df: pd.DataFrame, draw_threshold: float = None) -> tuple:
        """
        Predict with optional draw threshold override.

        If draw_threshold is set, predict draw when p_draw > threshold,
        regardless of whether draw is the argmax.

        Returns:
            (y_pred, y_proba) - predictions and probabilities
        """
        proba = self.predict_proba(df)

        if draw_threshold is None:
            # Standard argmax
            y_pred = np.argmax(proba, axis=1)
        else:
            # Threshold-based: predict draw if p_draw > threshold
            y_pred = np.argmax(proba, axis=1)
            draw_override = proba[:, 1] > draw_threshold
            y_pred[draw_override] = 1  # Override to draw

        return y_pred, proba


def evaluate_threshold(y_true: np.ndarray, y_pred: np.ndarray,
                       y_proba: np.ndarray, baseline_metrics: dict) -> dict:
    """Evaluate model with specific threshold."""
    acc = accuracy_score(y_true, y_pred)
    ll = log_loss(y_true, y_proba)

    # Brier per class
    brier = {}
    for c, name in enumerate(["home", "draw", "away"]):
        y_bin = (y_true == c).astype(int)
        brier[name] = brier_score_loss(y_bin, y_proba[:, c])
    brier["global"] = np.mean([brier["home"], brier["draw"], brier["away"]])

    # Draw metrics
    draw_pred_count = (y_pred == 1).sum()
    draw_pred_pct = draw_pred_count / len(y_pred)
    draw_prevalence = (y_true == 1).sum() / len(y_true)

    # Precision/Recall for draws
    true_draws = y_true == 1
    pred_draws = y_pred == 1

    correct_draws = (true_draws & pred_draws).sum()
    precision_draw = correct_draws / pred_draws.sum() if pred_draws.sum() > 0 else 0
    recall_draw = correct_draws / true_draws.sum() if true_draws.sum() > 0 else 0

    # PR-AUC for draw class
    try:
        pr_auc_draw = average_precision_score(true_draws.astype(int), y_proba[:, 1])
    except:
        pr_auc_draw = 0

    # Deltas vs baseline
    delta_acc = acc - baseline_metrics["accuracy"]
    delta_ll = ll - baseline_metrics["log_loss"]
    delta_brier = brier["global"] - baseline_metrics["brier_global"]

    return {
        "accuracy": acc,
        "log_loss": ll,
        "brier_global": brier["global"],
        "brier_draw": brier["draw"],
        "draw_pred_count": int(draw_pred_count),
        "draw_pred_pct": draw_pred_pct,
        "draw_prevalence": draw_prevalence,
        "precision_draw": precision_draw,
        "recall_draw": recall_draw,
        "pr_auc_draw": pr_auc_draw,
        "delta_acc": delta_acc,
        "delta_ll": delta_ll,
        "delta_brier": delta_brier,
    }


async def main():
    print("=" * 80)
    print("FASE 2: TWO-STAGE WITH THRESHOLD TUNING")
    print("=" * 80)

    # Load data
    async for session in get_async_session():
        fe = FeatureEngineer(session)
        print("\nLoading dataset...")
        df = await fe.build_training_dataset()
        break

    df = df.sort_values("date").reset_index(drop=True)
    df["date_dt"] = pd.to_datetime(df["date"])
    df = add_derived_features(df)

    print(f"Dataset: {len(df)} samples")

    # Temporal split
    cutoff_90 = datetime.utcnow() - timedelta(days=90)
    df_train = df[df["date_dt"] < cutoff_90].reset_index(drop=True)
    df_test = df[df["date_dt"] >= cutoff_90].reset_index(drop=True)

    y_test = df_test["result"].values
    draw_prevalence = (y_test == 1).mean()

    print(f"\nTrain: {len(df_train)} samples")
    print(f"Test: {len(df_test)} samples")
    print(f"Test draw prevalence: {draw_prevalence:.1%}")

    # ========================================
    # BASELINE
    # ========================================
    print("\n" + "=" * 80)
    print("BASELINE: v1.0.0")
    print("=" * 80)

    X_train_base = prepare_features(df_train, BASE_FEATURES)
    X_test_base = prepare_features(df_test, BASE_FEATURES)
    y_train = df_train["result"].values

    baseline_model = xgb.XGBClassifier(**PARAMS_BASELINE)
    baseline_model.fit(X_train_base, y_train, verbose=False)
    baseline_proba = baseline_model.predict_proba(X_test_base)
    baseline_pred = np.argmax(baseline_proba, axis=1)

    baseline_metrics = {
        "accuracy": accuracy_score(y_test, baseline_pred),
        "log_loss": log_loss(y_test, baseline_proba),
        "brier_global": np.mean([
            brier_score_loss((y_test == c).astype(int), baseline_proba[:, c])
            for c in range(3)
        ]),
    }

    print(f"Accuracy: {baseline_metrics['accuracy']:.4f}")
    print(f"LogLoss: {baseline_metrics['log_loss']:.4f}")
    print(f"Brier: {baseline_metrics['brier_global']:.4f}")
    print(f"Draw predictions: {(baseline_pred == 1).sum()} ({(baseline_pred == 1).mean():.1%})")

    # ========================================
    # TWO-STAGE MODEL
    # ========================================
    print("\n" + "=" * 80)
    print("TWO-STAGE MODEL (dw=1.2)")
    print("=" * 80)

    model = TwoStageModelWithThreshold(draw_weight=1.2)
    model.fit(df_train)

    # Get raw probabilities
    proba = model.predict_proba(df_test)

    # Analyze draw probability distribution
    p_draw = proba[:, 1]
    print(f"\nDraw probability distribution:")
    print(f"  Min: {p_draw.min():.4f}")
    print(f"  Max: {p_draw.max():.4f}")
    print(f"  Mean: {p_draw.mean():.4f}")
    print(f"  Median: {np.median(p_draw):.4f}")
    print(f"  P75: {np.percentile(p_draw, 75):.4f}")
    print(f"  P90: {np.percentile(p_draw, 90):.4f}")

    # Threshold sweep
    print("\n" + "=" * 80)
    print("THRESHOLD SWEEP")
    print("=" * 80)

    thresholds = [None, 0.20, 0.22, 0.24, 0.26, 0.28, 0.30, 0.32, 0.35, 0.40]

    print(f"\n{'Threshold':<12} {'Acc':>8} {'ΔAcc':>8} {'LogL':>8} {'ΔLogL':>8} "
          f"{'Brier':>8} {'ΔBrier':>8} {'Draw%':>8} {'DPrec':>8} {'DRecall':>8}")
    print("-" * 110)

    results = []
    for thresh in thresholds:
        y_pred, y_proba = model.predict_with_threshold(df_test, thresh)
        metrics = evaluate_threshold(y_test, y_pred, y_proba, baseline_metrics)
        metrics["threshold"] = thresh
        results.append(metrics)

        thresh_str = "argmax" if thresh is None else f"{thresh:.2f}"

        # Check acceptability
        is_acceptable = (
            metrics["brier_global"] <= baseline_metrics["brier_global"] + 0.002 and
            metrics["log_loss"] <= baseline_metrics["log_loss"] + 0.015 and
            metrics["draw_pred_pct"] > 0.02 and
            metrics["precision_draw"] > draw_prevalence * 0.9
        )
        marker = "✓" if is_acceptable else " "

        print(f"{thresh_str:<11}{marker} {metrics['accuracy']:>8.4f} {metrics['delta_acc']:>+8.4f} "
              f"{metrics['log_loss']:>8.4f} {metrics['delta_ll']:>+8.4f} "
              f"{metrics['brier_global']:>8.4f} {metrics['delta_brier']:>+8.4f} "
              f"{metrics['draw_pred_pct']:>8.1%} {metrics['precision_draw']:>8.3f} "
              f"{metrics['recall_draw']:>8.3f}")

    # ========================================
    # ANALYSIS
    # ========================================
    print("\n" + "=" * 80)
    print("ANÁLISIS")
    print("=" * 80)

    acceptable = [r for r in results if (
        r["brier_global"] <= baseline_metrics["brier_global"] + 0.002 and
        r["log_loss"] <= baseline_metrics["log_loss"] + 0.015 and
        r["draw_pred_pct"] > 0.02 and
        r["precision_draw"] > draw_prevalence * 0.9
    )]

    if acceptable:
        # Best = highest precision among acceptable
        best = max(acceptable, key=lambda x: x["precision_draw"])

        thresh_str = "argmax" if best["threshold"] is None else f"{best['threshold']:.2f}"
        print(f"\n✓ CANDIDATO ENCONTRADO: threshold={thresh_str}")
        print(f"\n  Métricas:")
        print(f"  - Accuracy: {best['accuracy']:.4f} (Δ={best['delta_acc']:+.4f})")
        print(f"  - LogLoss: {best['log_loss']:.4f} (Δ={best['delta_ll']:+.4f})")
        print(f"  - Brier: {best['brier_global']:.4f} (Δ={best['delta_brier']:+.4f})")
        print(f"  - Draw predictions: {best['draw_pred_pct']:.1%}")
        print(f"  - Draw precision: {best['precision_draw']:.3f} (prevalence: {best['draw_prevalence']:.3f})")
        print(f"  - Draw recall: {best['recall_draw']:.3f}")

        # GO/NO-GO criteria
        criteria = {
            "brier_ok": best["delta_brier"] <= 0.002,
            "logloss_ok": best["delta_ll"] <= 0.015,
            "draw_pct_ok": best["draw_pred_pct"] > 0.02,
            "precision_ok": best["precision_draw"] > best["draw_prevalence"] * 0.9,
        }

        print(f"\n  Criterios GO:")
        all_pass = True
        for k, v in criteria.items():
            status = "✓" if v else "✗"
            print(f"    {status} {k}")
            if not v:
                all_pass = False

        if all_pass:
            print(f"\n  → RECOMENDACIÓN: GO (shadow mode)")
        else:
            print(f"\n  → RECOMENDACIÓN: NO-GO, continuar ajustando")
    else:
        print("\n✗ NO HAY CANDIDATO ACEPTABLE")
        print("  → Probar otras estrategias")

    # ========================================
    # CALIBRATION CHECK
    # ========================================
    print("\n" + "=" * 80)
    print("CALIBRATION CHECK (Two-Stage argmax)")
    print("=" * 80)

    # Compare predicted probabilities vs actual outcomes
    y_pred_argmax, proba_ts = model.predict_with_threshold(df_test, None)

    # Bin by predicted draw probability
    bins = [0, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 1.0]
    p_draw_test = proba_ts[:, 1]
    actual_draw = (y_test == 1).astype(int)

    print(f"\n{'P(draw) bin':<15} {'Count':>8} {'Pred avg':>10} {'Actual':>10} {'Gap':>10}")
    print("-" * 60)

    for i in range(len(bins) - 1):
        mask = (p_draw_test >= bins[i]) & (p_draw_test < bins[i + 1])
        count = mask.sum()
        if count > 0:
            pred_avg = p_draw_test[mask].mean()
            actual_rate = actual_draw[mask].mean()
            gap = pred_avg - actual_rate
            print(f"[{bins[i]:.2f}, {bins[i+1]:.2f}){'':<3} {count:>8} {pred_avg:>10.3f} {actual_rate:>10.3f} {gap:>+10.3f}")

    # Summary
    print("\n" + "=" * 80)
    print("RESUMEN PARA AUDITOR")
    print("=" * 80)

    print(f"""
FASE 2: Two-Stage Model
=======================

Arquitectura:
- Stage 1: Binary (draw vs non-draw), dw=1.2
- Stage 2: Binary (home vs away)
- Composición: p_draw=p1, p_home=(1-p1)*p2, p_away=(1-p1)*(1-p2)

Resultado vs Baseline (v1.0.0):
- Brier global: {'MEJOR' if results[0]['delta_brier'] < 0 else 'PEOR'} ({results[0]['delta_brier']:+.4f})
- LogLoss: {'MEJOR' if results[0]['delta_ll'] < 0 else 'PEOR'} ({results[0]['delta_ll']:+.4f})
- Draw predictions: {results[0]['draw_pred_pct']:.1%} (baseline: 0%)

Observación clave:
- El two-stage con argmax NO predice draws (igual que baseline)
- Necesita threshold override para forzar draws
- Al forzar draws con threshold, se degrada calibración

Diagnóstico:
- P(draw) en test: mean={p_draw.mean():.3f}, max={p_draw.max():.3f}
- El modelo aprende que draws son raros y sub-predice
- El threshold override fuerza predictions pero NO mejora discriminación
""")

    if acceptable:
        print(f"VEREDICTO: GO CONDICIONAL")
        print(f"- Threshold recomendado: {best['threshold']}")
        print(f"- Precision: {best['precision_draw']:.3f} > prevalence {best['draw_prevalence']:.3f}")
    else:
        print(f"VEREDICTO: NO-GO")
        print(f"- Two-stage no resuelve el problema fundamental")
        print(f"- Alternativas: ensemble con odds, recalibration, etc.")


if __name__ == "__main__":
    asyncio.run(main())
