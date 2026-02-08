#!/usr/bin/env python3
"""
Experiment 1: Temporal Drift Analysis — Argentina (league_id=128)
=================================================================
ABE-directed experiment to determine if Argentina's poor Brier (0.6617)
is caused by regime change (format/phase) or inherent unpredictability.

Splits:
  - Train ≤2022, test 2023
  - Train ≤2023, test 2024
  - Train ≤2024, test 2025
  - Train ≤2025, test 2026

Also runs "expanding window" within Argentina-only data to isolate
the effect from the global model.

Usage:
  source .env
  python scripts/experiment_drift_argentina.py
  python scripts/experiment_drift_argentina.py --extract   # fresh DB extraction
"""

import json
import sys
import argparse
import warnings
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import log_loss

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ─── Reuse config from feature_diagnostic ────────────────────

FEATURES_V101 = [
    "home_goals_scored_avg", "home_goals_conceded_avg",
    "home_shots_avg", "home_corners_avg",
    "home_rest_days", "home_matches_played",
    "away_goals_scored_avg", "away_goals_conceded_avg",
    "away_shots_avg", "away_corners_avg",
    "away_rest_days", "away_matches_played",
    "goal_diff_avg", "rest_diff",
    "abs_attack_diff", "abs_defense_diff", "abs_strength_gap",
]

PROD_HYPERPARAMS = {
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
    "use_label_encoder": False,
    "eval_metric": "mlogloss",
    "verbosity": 0,
}

N_SEEDS = 3
DRAW_WEIGHT = 1.5
ARGENTINA_LEAGUE_ID = 128


def multiclass_brier(y_true, y_prob):
    n_classes = y_prob.shape[1]
    y_onehot = np.eye(n_classes)[y_true.astype(int)]
    return float(np.mean(np.sum((y_prob - y_onehot) ** 2, axis=1)))


def train_and_evaluate(X_train, y_train, X_test, y_test):
    """Train XGBoost with N_SEEDS and return mean metrics."""
    briers, loglosses, accs = [], [], []

    for seed in range(N_SEEDS):
        params = {**PROD_HYPERPARAMS, "random_state": seed * 42}
        model = xgb.XGBClassifier(**params)

        # Draw weighting (production parity)
        sample_weight = np.ones(len(y_train), dtype=np.float32)
        sample_weight[y_train == 1] = DRAW_WEIGHT
        model.fit(X_train, y_train, sample_weight=sample_weight)

        y_prob = model.predict_proba(X_test)
        briers.append(multiclass_brier(y_test, y_prob))
        loglosses.append(log_loss(y_test, y_prob, labels=[0, 1, 2]))
        accs.append(float(np.mean(model.predict(X_test) == y_test)))

    return {
        "brier_mean": round(np.mean(briers), 6),
        "brier_std": round(np.std(briers), 6),
        "logloss_mean": round(np.mean(loglosses), 6),
        "accuracy_mean": round(np.mean(accs), 4),
        "n_seeds": N_SEEDS,
    }


def label_distribution(y):
    """Return H/D/A distribution as percentages."""
    total = len(y)
    if total == 0:
        return {"H": 0, "D": 0, "A": 0}
    return {
        "H": round(100 * np.sum(y == 0) / total, 1),
        "D": round(100 * np.sum(y == 1) / total, 1),
        "A": round(100 * np.sum(y == 2) / total, 1),
    }


def naive_baseline_brier(y_train, y_test):
    """Brier score of a model that always predicts training class distribution."""
    total = len(y_train)
    probs = np.array([
        np.sum(y_train == 0) / total,
        np.sum(y_train == 1) / total,
        np.sum(y_train == 2) / total,
    ])
    y_prob = np.tile(probs, (len(y_test), 1))
    return multiclass_brier(y_test, y_prob)


def run_drift_experiment(df_full, df_argentina):
    """Run the 4-split drift experiment."""

    features = FEATURES_V101
    results = {"experiment": "drift_by_season_argentina", "timestamp": datetime.now().isoformat()}

    # ─── Part A: Argentina-only splits (train on ARG, test on ARG) ────
    print(f"\n{'=' * 70}")
    print(f"  PART A: ARGENTINA-ONLY DRIFT (train ARG → test ARG)")
    print(f"{'=' * 70}")

    splits_a = []
    season_boundaries = [
        ("≤2022 → 2023", lambda d: d.year <= 2022, lambda d: d.year == 2023),
        ("≤2023 → 2024", lambda d: d.year <= 2023, lambda d: d.year == 2024),
        ("≤2024 → 2025", lambda d: d.year <= 2024, lambda d: d.year == 2025),
        ("≤2025 → 2026", lambda d: d.year <= 2025, lambda d: d.year == 2026),
    ]

    for label, train_filter, test_filter in season_boundaries:
        df_train = df_argentina[df_argentina["date"].apply(train_filter)].copy()
        df_test = df_argentina[df_argentina["date"].apply(test_filter)].copy()

        # Drop NaN features
        df_train = df_train.dropna(subset=features)
        df_test = df_test.dropna(subset=features)

        if len(df_test) < 20:
            print(f"\n  {label}: SKIP (test={len(df_test)} < 20)")
            splits_a.append({"split": label, "status": "SKIPPED", "n_test": len(df_test)})
            continue

        X_train = df_train[features].values.astype(np.float32)
        y_train = df_train["result"].values.astype(int)
        X_test = df_test[features].values.astype(np.float32)
        y_test = df_test["result"].values.astype(int)

        metrics = train_and_evaluate(X_train, y_train, X_test, y_test)
        naive_brier = naive_baseline_brier(y_train, y_test)

        split_result = {
            "split": label,
            "n_train": len(df_train),
            "n_test": len(df_test),
            "train_date_range": [str(df_train["date"].min()), str(df_train["date"].max())],
            "test_date_range": [str(df_test["date"].min()), str(df_test["date"].max())],
            "train_labels": label_distribution(y_train),
            "test_labels": label_distribution(y_test),
            "naive_brier": round(naive_brier, 6),
            **metrics,
            "skill_vs_naive": round(naive_brier - metrics["brier_mean"], 6),
        }
        splits_a.append(split_result)

        print(f"\n  {label}")
        print(f"    Train: {len(df_train):,} | Test: {len(df_test):,}")
        print(f"    Test labels: H={split_result['test_labels']['H']}% D={split_result['test_labels']['D']}% A={split_result['test_labels']['A']}%")
        print(f"    Naive Brier:  {naive_brier:.4f}")
        print(f"    Model Brier:  {metrics['brier_mean']:.4f} ± {metrics['brier_std']:.4f}")
        print(f"    Skill (naive - model): {split_result['skill_vs_naive']:+.4f}")
        print(f"    Accuracy: {metrics['accuracy_mean']:.1%}")

    results["part_a_argentina_only"] = splits_a

    # ─── Part B: Global train → Argentina test (production-like) ────
    print(f"\n{'=' * 70}")
    print(f"  PART B: GLOBAL TRAIN → ARGENTINA TEST (production-like)")
    print(f"{'=' * 70}")

    splits_b = []
    for label, train_filter, test_filter in season_boundaries:
        # Train on ALL leagues up to cutoff
        df_train = df_full[df_full["date"].apply(train_filter)].copy()
        # Test on Argentina only
        df_test = df_argentina[df_argentina["date"].apply(test_filter)].copy()

        df_train = df_train.dropna(subset=features)
        df_test = df_test.dropna(subset=features)

        if len(df_test) < 20:
            print(f"\n  {label}: SKIP (test={len(df_test)} < 20)")
            splits_b.append({"split": label, "status": "SKIPPED", "n_test": len(df_test)})
            continue

        X_train = df_train[features].values.astype(np.float32)
        y_train = df_train["result"].values.astype(int)
        X_test = df_test[features].values.astype(np.float32)
        y_test = df_test["result"].values.astype(int)

        metrics = train_and_evaluate(X_train, y_train, X_test, y_test)
        naive_brier = naive_baseline_brier(y_train, y_test)

        split_result = {
            "split": label,
            "n_train": len(df_train),
            "n_test": len(df_test),
            "train_leagues": int(df_train["league_id"].nunique()),
            "test_date_range": [str(df_test["date"].min()), str(df_test["date"].max())],
            "test_labels": label_distribution(y_test),
            "naive_brier": round(naive_brier, 6),
            **metrics,
            "skill_vs_naive": round(naive_brier - metrics["brier_mean"], 6),
        }
        splits_b.append(split_result)

        print(f"\n  {label}")
        print(f"    Train: {len(df_train):,} ({split_result['train_leagues']} leagues) | Test ARG: {len(df_test):,}")
        print(f"    Test labels: H={split_result['test_labels']['H']}% D={split_result['test_labels']['D']}% A={split_result['test_labels']['A']}%")
        print(f"    Naive Brier:  {naive_brier:.4f}")
        print(f"    Model Brier:  {metrics['brier_mean']:.4f} ± {metrics['brier_std']:.4f}")
        print(f"    Skill (naive - model): {split_result['skill_vs_naive']:+.4f}")

    results["part_b_global_to_argentina"] = splits_b

    # ─── Part C: Global train → Global test (benchmark) ────
    print(f"\n{'=' * 70}")
    print(f"  PART C: GLOBAL TRAIN → GLOBAL TEST (benchmark)")
    print(f"{'=' * 70}")

    splits_c = []
    for label, train_filter, test_filter in season_boundaries:
        df_train = df_full[df_full["date"].apply(train_filter)].copy()
        df_test = df_full[df_full["date"].apply(test_filter)].copy()

        df_train = df_train.dropna(subset=features)
        df_test = df_test.dropna(subset=features)

        if len(df_test) < 20:
            print(f"\n  {label}: SKIP (test={len(df_test)} < 20)")
            splits_c.append({"split": label, "status": "SKIPPED", "n_test": len(df_test)})
            continue

        X_train = df_train[features].values.astype(np.float32)
        y_train = df_train["result"].values.astype(int)
        X_test = df_test[features].values.astype(np.float32)
        y_test = df_test["result"].values.astype(int)

        metrics = train_and_evaluate(X_train, y_train, X_test, y_test)
        naive_brier = naive_baseline_brier(y_train, y_test)

        split_result = {
            "split": label,
            "n_train": len(df_train),
            "n_test": len(df_test),
            "naive_brier": round(naive_brier, 6),
            **metrics,
            "skill_vs_naive": round(naive_brier - metrics["brier_mean"], 6),
        }
        splits_c.append(split_result)

        print(f"\n  {label}")
        print(f"    Train: {len(df_train):,} | Test: {len(df_test):,}")
        print(f"    Naive Brier:  {naive_brier:.4f}")
        print(f"    Model Brier:  {metrics['brier_mean']:.4f} ± {metrics['brier_std']:.4f}")
        print(f"    Skill (naive - model): {split_result['skill_vs_naive']:+.4f}")

    results["part_c_global_benchmark"] = splits_c

    # ─── Summary table ────
    print(f"\n{'=' * 70}")
    print(f"  SUMMARY: BRIER BY SPLIT")
    print(f"{'=' * 70}")
    print(f"\n  {'Split':<18} {'ARG→ARG':>10} {'Global→ARG':>12} {'Global→Global':>14}  {'ARG Δ naive':>12}")
    print(f"  {'-'*18} {'-'*10} {'-'*12} {'-'*14}  {'-'*12}")

    for sa, sb, sc in zip(splits_a, splits_b, splits_c):
        label = sa["split"]
        a = f"{sa['brier_mean']:.4f}" if "brier_mean" in sa else "SKIP"
        b = f"{sb['brier_mean']:.4f}" if "brier_mean" in sb else "SKIP"
        c = f"{sc['brier_mean']:.4f}" if "brier_mean" in sc else "SKIP"
        skill = f"{sa['skill_vs_naive']:+.4f}" if "skill_vs_naive" in sa else "N/A"
        print(f"  {label:<18} {a:>10} {b:>12} {c:>14}  {skill:>12}")

    # Save
    output_file = Path("scripts/output/experiment_drift_argentina.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved to: {output_file}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Drift experiment: Argentina by season")
    parser.add_argument("--extract", action="store_true", help="Extract fresh dataset from DB")
    parser.add_argument("--dataset", type=str, default="scripts/output/training_dataset.csv")
    args = parser.parse_args()

    dataset_path = Path(args.dataset)

    print(f"\n  EXPERIMENT 1: TEMPORAL DRIFT — ARGENTINA")
    print(f"  {'=' * 50}")

    if args.extract or not dataset_path.exists():
        print(f"\n  Extracting from DB...")
        from scripts.feature_diagnostic import _extract_via_sql
        df = _extract_via_sql(league_only=True, output_path=str(dataset_path))
    else:
        print(f"\n  Loading cached dataset: {dataset_path}")
        df = pd.read_csv(dataset_path, parse_dates=["date"])
        print(f"  Loaded {len(df)} rows")

    # Separate Argentina from full dataset
    df_argentina = df[df["league_id"] == ARGENTINA_LEAGUE_ID].copy()
    print(f"\n  Full dataset: {len(df):,} matches ({df['league_id'].nunique()} leagues)")
    print(f"  Argentina:    {len(df_argentina):,} matches")
    print(f"  Date range:   {df_argentina['date'].min()} → {df_argentina['date'].max()}")

    # Season distribution
    df_argentina["year"] = df_argentina["date"].dt.year
    for year in sorted(df_argentina["year"].unique()):
        n = len(df_argentina[df_argentina["year"] == year])
        print(f"    {year}: {n} matches")

    run_drift_experiment(df, df_argentina)


if __name__ == "__main__":
    main()
