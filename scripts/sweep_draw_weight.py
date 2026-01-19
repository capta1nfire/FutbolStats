#!/usr/bin/env python3
"""
Draw Weight Sweep - Find optimal sample_weight for draws

Tests weights: 1.0, 1.05, 1.10, 1.15, 1.20, 1.25, 1.30, 1.35, 1.50
Evaluates on last 90 days holdout.

Objective:
- brier global <= v1.0.0 (baseline)
- logloss global <= v1.0.0
- draw_top1_rate > 0% with precision > prevalence
- don't destroy away recall
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
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss

from app.database import get_async_session
from app.features.engineering import FeatureEngineer

# Features v1.1.0
FEATURES = [
    "home_goals_scored_avg", "home_goals_conceded_avg", "home_shots_avg",
    "home_corners_avg", "home_rest_days", "home_matches_played",
    "away_goals_scored_avg", "away_goals_conceded_avg", "away_shots_avg",
    "away_corners_avg", "away_rest_days", "away_matches_played",
    "goal_diff_avg", "rest_diff",
    "abs_attack_diff", "abs_defense_diff", "abs_strength_gap",
]

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

WEIGHTS_TO_TEST = [1.0, 1.05, 1.10, 1.15, 1.20, 1.25, 1.30, 1.35, 1.50]


def prepare_features(df: pd.DataFrame) -> np.ndarray:
    for col in FEATURES:
        if col not in df.columns:
            df[col] = 0
    return df[FEATURES].fillna(0).values


def evaluate(X_train, y_train, X_test, y_test, draw_weight: float) -> dict:
    """Train and evaluate with given draw_weight."""
    sample_weight = np.ones(len(y_train), dtype=np.float32)
    sample_weight[y_train == 1] = draw_weight

    model = xgb.XGBClassifier(**PARAMS)
    model.fit(X_train, y_train, sample_weight=sample_weight, verbose=False)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    # Global metrics
    acc = accuracy_score(y_test, y_pred)
    ll = log_loss(y_test, y_proba)

    # Brier per class
    brier = {}
    for c, name in enumerate(["home", "draw", "away"]):
        brier[name] = brier_score_loss((y_test == c).astype(int), y_proba[:, c])
    brier["global"] = np.mean(list(brier.values()))

    # Per-class metrics
    metrics = {"precision": {}, "recall": {}}
    for c, name in enumerate(["home", "draw", "away"]):
        pred_c = (y_pred == c).sum()
        true_c = (y_test == c).sum()
        correct_c = ((y_pred == c) & (y_test == c)).sum()

        metrics["precision"][name] = correct_c / pred_c if pred_c > 0 else 0
        metrics["recall"][name] = correct_c / true_c if true_c > 0 else 0

    # Draw specific
    draw_pred_count = (y_pred == 1).sum()
    draw_pred_pct = draw_pred_count / len(y_pred)
    draw_prevalence = (y_test == 1).sum() / len(y_test)

    return {
        "weight": draw_weight,
        "accuracy": acc,
        "log_loss": ll,
        "brier_global": brier["global"],
        "brier_home": brier["home"],
        "brier_draw": brier["draw"],
        "brier_away": brier["away"],
        "precision_home": metrics["precision"]["home"],
        "precision_draw": metrics["precision"]["draw"],
        "precision_away": metrics["precision"]["away"],
        "recall_home": metrics["recall"]["home"],
        "recall_draw": metrics["recall"]["draw"],
        "recall_away": metrics["recall"]["away"],
        "draw_pred_count": int(draw_pred_count),
        "draw_pred_pct": draw_pred_pct,
        "draw_prevalence": draw_prevalence,
    }


async def main():
    print("=" * 80)
    print("DRAW WEIGHT SWEEP")
    print("=" * 80)

    # Load data
    async for session in get_async_session():
        fe = FeatureEngineer(session)
        print("\nLoading dataset...")
        df = await fe.build_training_dataset()
        break

    df = df.sort_values("date").reset_index(drop=True)
    df["date_dt"] = pd.to_datetime(df["date"])

    X = prepare_features(df.copy())
    y = df["result"].values

    # Split: train on all except last 90 days
    cutoff_90 = datetime.utcnow() - timedelta(days=90)
    train_mask = df["date_dt"] < cutoff_90
    test_mask = df["date_dt"] >= cutoff_90

    train_idx = df[train_mask].index.values
    test_idx = df[test_mask].index.values

    print(f"\nTrain: {len(train_idx)} samples")
    print(f"Test (90d): {len(test_idx)} samples")
    print(f"Test draw prevalence: {(y[test_idx] == 1).mean():.1%}")

    # Run sweep
    results = []
    for w in WEIGHTS_TO_TEST:
        r = evaluate(X[train_idx], y[train_idx], X[test_idx], y[test_idx], w)
        results.append(r)
        print(f"  weight={w:.2f}: acc={r['accuracy']:.4f}, brier={r['brier_global']:.4f}, "
              f"draw_pred={r['draw_pred_pct']:.1%}, draw_prec={r['precision_draw']:.3f}")

    # Find baseline (w=1.0)
    baseline = results[0]

    print("\n" + "=" * 80)
    print("RESULTADOS SWEEP (últimos 90 días)")
    print("=" * 80)

    # Header
    print(f"\n{'Weight':>8} {'Acc':>8} {'ΔAcc':>8} {'LogL':>8} {'ΔLogL':>8} "
          f"{'Brier':>8} {'ΔBrier':>8} {'Draw%':>8} {'DPrec':>8} {'DRecall':>8} "
          f"{'ARecall':>8}")
    print("-" * 100)

    for r in results:
        delta_acc = r["accuracy"] - baseline["accuracy"]
        delta_ll = r["log_loss"] - baseline["log_loss"]
        delta_brier = r["brier_global"] - baseline["brier_global"]

        # Highlight acceptable candidates
        is_acceptable = (
            r["brier_global"] <= baseline["brier_global"] + 0.001 and
            r["log_loss"] <= baseline["log_loss"] + 0.01 and
            r["draw_pred_pct"] > 0.01 and
            r["precision_draw"] > r["draw_prevalence"]
        )
        marker = "✓" if is_acceptable else " "

        print(f"{r['weight']:>7.2f}{marker} {r['accuracy']:>8.4f} {delta_acc:>+8.4f} "
              f"{r['log_loss']:>8.4f} {delta_ll:>+8.4f} "
              f"{r['brier_global']:>8.4f} {delta_brier:>+8.4f} "
              f"{r['draw_pred_pct']:>8.1%} {r['precision_draw']:>8.3f} "
              f"{r['recall_draw']:>8.3f} {r['recall_away']:>8.3f}")

    # Find best acceptable
    acceptable = [r for r in results if (
        r["brier_global"] <= baseline["brier_global"] + 0.001 and
        r["log_loss"] <= baseline["log_loss"] + 0.01 and
        r["draw_pred_pct"] > 0.01
    )]

    print("\n" + "=" * 80)
    print("ANÁLISIS")
    print("=" * 80)

    if acceptable:
        # Best = highest draw_pred_pct among acceptable
        best = max(acceptable, key=lambda x: x["draw_pred_pct"])
        print(f"\n✓ CANDIDATO ENCONTRADO: draw_weight={best['weight']:.2f}")
        print(f"  - Accuracy: {best['accuracy']:.4f} (Δ={best['accuracy']-baseline['accuracy']:+.4f})")
        print(f"  - LogLoss: {best['log_loss']:.4f} (Δ={best['log_loss']-baseline['log_loss']:+.4f})")
        print(f"  - Brier: {best['brier_global']:.4f} (Δ={best['brier_global']-baseline['brier_global']:+.4f})")
        print(f"  - Draw predictions: {best['draw_pred_pct']:.1%}")
        print(f"  - Draw precision: {best['precision_draw']:.3f} (prevalence: {best['draw_prevalence']:.3f})")
        print(f"  - Away recall: {best['recall_away']:.3f}")
    else:
        print("\n✗ NO HAY CANDIDATO ACEPTABLE")
        print("  Ningún peso mantiene brier/logloss mientras predice >1% draws.")
        print("  → Recomendación: FASE 2 (two-stage model)")

    # Detailed comparison table
    print("\n" + "=" * 80)
    print("BRIER POR CLASE")
    print("=" * 80)
    print(f"{'Weight':>8} {'B_Home':>10} {'B_Draw':>10} {'B_Away':>10} {'B_Global':>10}")
    print("-" * 50)
    for r in results:
        print(f"{r['weight']:>8.2f} {r['brier_home']:>10.4f} {r['brier_draw']:>10.4f} "
              f"{r['brier_away']:>10.4f} {r['brier_global']:>10.4f}")


if __name__ == "__main__":
    asyncio.run(main())
