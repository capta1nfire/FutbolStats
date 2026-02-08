#!/usr/bin/env python3
"""
ABE Task A: Market Anchor Backtest — Argentina (league_id=128)
===============================================================
Calibrate α* for: p_final = (1-α)*p_model_global + α*p_market

Problem: Argentina has 0 historical odds pre-2026. Only ~50 matches in 2026
with frozen_odds (Bet365). Consensus covers only 9 matches (not yet FT).

Strategy:
  Part 1: ARG 2026 with N≈50 (small but directionally useful)
  Part 2: Global proxy — same analysis on ALL leagues with odds (robust N)
  Part 3: Low-signal league proxy — same analysis on LATAM leagues with odds

Usage:
  source .env
  python scripts/experiment_market_anchor.py
"""

import json
import sys
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
N_BOOTSTRAP = 1000
ARGENTINA = 128

# LATAM leagues (low-signal cluster from diagnostic)
LATAM_LEAGUES = [128, 71, 239, 242, 253]  # ARG, BRA, COL, ECU, PAR


def multiclass_brier(y_true, y_prob):
    n_classes = y_prob.shape[1]
    y_onehot = np.eye(n_classes)[y_true.astype(int)]
    return float(np.mean(np.sum((y_prob - y_onehot) ** 2, axis=1)))


def odds_to_probs(odds_h, odds_d, odds_a):
    """Convert decimal odds to de-vigged probabilities (proportional method)."""
    raw_h = 1.0 / odds_h
    raw_d = 1.0 / odds_d
    raw_a = 1.0 / odds_a
    total = raw_h + raw_d + raw_a
    return np.column_stack([raw_h / total, raw_d / total, raw_a / total])


def naive_probs(y_train, n_test):
    total = len(y_train)
    probs = np.array([np.sum(y_train == c) / total for c in range(3)])
    return np.tile(probs, (n_test, 1))


def get_ensemble_probs(X_train, y_train, X_test, n_seeds=N_SEEDS):
    all_probs = []
    for seed in range(n_seeds):
        params = {**PROD_HYPERPARAMS, "random_state": seed * 42}
        model = xgb.XGBClassifier(**params)
        sw = np.ones(len(y_train), dtype=np.float32)
        sw[y_train == 1] = DRAW_WEIGHT
        model.fit(X_train, y_train, sample_weight=sw)
        all_probs.append(model.predict_proba(X_test))
    return np.mean(all_probs, axis=0)


def shrinkage_grid(y_true, p_model, p_anchor, alphas=None):
    """Find optimal α for p_final = (1-α)*p_model + α*p_anchor."""
    if alphas is None:
        alphas = np.arange(0.0, 1.01, 0.05)
    best_alpha, best_brier = 0.0, float("inf")
    grid = []
    for alpha in alphas:
        p_blend = (1 - alpha) * p_model + alpha * p_anchor
        b = multiclass_brier(y_true, p_blend)
        grid.append({"alpha": round(float(alpha), 2), "brier": round(b, 6)})
        if b < best_brier:
            best_brier = b
            best_alpha = alpha
    return round(float(best_alpha), 2), round(float(best_brier), 6), grid


def bootstrap_brier_ci(y_true, y_prob, n_bootstrap=N_BOOTSTRAP):
    n = len(y_true)
    rng = np.random.RandomState(42)
    briers = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, n, replace=True)
        briers.append(multiclass_brier(y_true[idx], y_prob[idx]))
    briers = np.array(briers)
    return {
        "mean": round(float(np.mean(briers)), 6),
        "p05": round(float(np.percentile(briers, 5)), 6),
        "p95": round(float(np.percentile(briers, 95)), 6),
    }


def bootstrap_skill_ci(y_true, y_prob_model, y_prob_baseline, n_bootstrap=N_BOOTSTRAP):
    n = len(y_true)
    rng = np.random.RandomState(42)
    skills = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, n, replace=True)
        skills.append(multiclass_brier(y_true[idx], y_prob_baseline[idx]) -
                      multiclass_brier(y_true[idx], y_prob_model[idx]))
    skills = np.array(skills)
    return {
        "mean": round(float(np.mean(skills)), 6),
        "p05": round(float(np.percentile(skills, 5)), 6),
        "p95": round(float(np.percentile(skills, 95)), 6),
        "crosses_zero": bool(np.percentile(skills, 5) <= 0 <= np.percentile(skills, 95)),
    }


def run_analysis(label, y_test, p_naive, p_model, p_market):
    """Run full analysis for a given split."""
    brier_naive = multiclass_brier(y_test, p_naive)
    brier_model = multiclass_brier(y_test, p_model)
    brier_market = multiclass_brier(y_test, p_market)

    # α* model+market
    alpha_mm, brier_mm, grid_mm = shrinkage_grid(y_test, p_model, p_market)
    # α* model+naive
    alpha_mn, brier_mn, _ = shrinkage_grid(y_test, p_model, p_naive)

    # Bootstrap CIs
    ci_naive = bootstrap_brier_ci(y_test, p_naive)
    ci_model = bootstrap_brier_ci(y_test, p_model)
    ci_market = bootstrap_brier_ci(y_test, p_market)
    ci_blend = bootstrap_brier_ci(y_test,
        (1 - alpha_mm) * p_model + alpha_mm * p_market)
    skill_model_vs_market = bootstrap_skill_ci(y_test, p_model, p_market)
    skill_blend_vs_market = bootstrap_skill_ci(y_test,
        (1 - alpha_mm) * p_model + alpha_mm * p_market, p_market)

    result = {
        "label": label,
        "n_test": len(y_test),
        "brier_naive": round(brier_naive, 6),
        "brier_model": round(brier_model, 6),
        "brier_market": round(brier_market, 6),
        "brier_blend_best": brier_mm,
        "alpha_model_market": alpha_mm,
        "alpha_model_naive": alpha_mn,
        "ci_naive": ci_naive,
        "ci_model": ci_model,
        "ci_market": ci_market,
        "ci_blend": ci_blend,
        "skill_model_vs_market": skill_model_vs_market,
        "skill_blend_vs_market": skill_blend_vs_market,
        "model_vs_market": round(brier_market - brier_model, 6),
        "blend_vs_market": round(brier_market - brier_mm, 6),
    }

    print(f"\n  {label} (N={len(y_test)})")
    print(f"    Naive:                {brier_naive:.4f} [{ci_naive['p05']:.4f}, {ci_naive['p95']:.4f}]")
    print(f"    Market (devigged):    {brier_market:.4f} [{ci_market['p05']:.4f}, {ci_market['p95']:.4f}]")
    print(f"    Model Global:         {brier_model:.4f} [{ci_model['p05']:.4f}, {ci_model['p95']:.4f}]")
    print(f"    Model vs Market:      {brier_market - brier_model:+.4f} (>0 = model better)")
    print(f"    α*(model+market):     {alpha_mm:.2f} → blend={brier_mm:.4f}")
    print(f"    α*(model+naive):      {alpha_mn:.2f}")
    print(f"    Blend vs Market:      {brier_market - brier_mm:+.4f} [{skill_blend_vs_market['p05']:+.4f}, {skill_blend_vs_market['p95']:+.4f}]")

    return result


def main():
    print(f"\n  ABE TASK A: MARKET ANCHOR BACKTEST")
    print(f"  {'='*50}")

    # Load dataset
    df = pd.read_csv("scripts/output/training_dataset.csv", parse_dates=["date"])
    features = FEATURES_V101

    results = {
        "experiment": "market_anchor_backtest",
        "timestamp": datetime.now().isoformat(),
        "caveat": "Argentina has 0 odds pre-2026. Part 1 uses N≈50 (2026 only). Parts 2-3 use global/LATAM data as proxy.",
    }

    # ─── PART 1: Argentina 2026 (N≈50) ──────────────────────
    print(f"\n{'='*70}")
    print(f"  PART 1: ARGENTINA 2026 (direct, small N)")
    print(f"{'='*70}")

    df_arg = df[df["league_id"] == ARGENTINA].copy()
    df_arg_odds = df_arg[
        (df_arg["odds_home"].notna()) &
        (df_arg["odds_home"] > 1) &
        (df_arg["date"].dt.year == 2026)
    ].dropna(subset=features)

    if len(df_arg_odds) >= 10:
        # Train: all global data ≤2025
        df_train = df[df["date"].dt.year <= 2025].dropna(subset=features)
        X_train = df_train[features].values.astype(np.float32)
        y_train = df_train["result"].values.astype(int)

        X_test = df_arg_odds[features].values.astype(np.float32)
        y_test = df_arg_odds["result"].values.astype(int)

        p_model = get_ensemble_probs(X_train, y_train, X_test)
        p_market = odds_to_probs(
            df_arg_odds["odds_home"].values,
            df_arg_odds["odds_draw"].values,
            df_arg_odds["odds_away"].values,
        )
        p_naive = naive_probs(y_train, len(y_test))

        r1 = run_analysis("ARG 2026 (Global→ARG)", y_test, p_naive, p_model, p_market)
        results["part1_argentina_2026"] = r1
    else:
        print(f"  SKIP: Only {len(df_arg_odds)} ARG matches with odds in 2026")
        results["part1_argentina_2026"] = {"status": "SKIPPED", "n": len(df_arg_odds)}

    # ─── PART 2: Global proxy (all leagues with odds) ────────
    print(f"\n{'='*70}")
    print(f"  PART 2: GLOBAL PROXY (all leagues with odds)")
    print(f"{'='*70}")

    part2_results = []
    season_splits = [
        ("≤2024→2025", lambda d: d.year <= 2024, lambda d: d.year == 2025),
        ("≤2025→2026", lambda d: d.year <= 2025, lambda d: d.year == 2026),
    ]

    for label, train_f, test_f in season_splits:
        df_train = df[df["date"].apply(train_f)].dropna(subset=features)
        df_test = df[df["date"].apply(test_f)].dropna(subset=features)
        df_test_odds = df_test[
            (df_test["odds_home"].notna()) &
            (df_test["odds_home"] > 1)
        ]

        if len(df_test_odds) < 50:
            print(f"\n  {label}: SKIP (N={len(df_test_odds)} with odds)")
            part2_results.append({"split": label, "status": "SKIPPED", "n": len(df_test_odds)})
            continue

        X_tr = df_train[features].values.astype(np.float32)
        y_tr = df_train["result"].values.astype(int)
        X_te = df_test_odds[features].values.astype(np.float32)
        y_te = df_test_odds["result"].values.astype(int)

        p_model = get_ensemble_probs(X_tr, y_tr, X_te)
        p_market = odds_to_probs(
            df_test_odds["odds_home"].values,
            df_test_odds["odds_draw"].values,
            df_test_odds["odds_away"].values,
        )
        p_naive = naive_probs(y_tr, len(y_te))

        n_leagues = df_test_odds["league_id"].nunique()
        r = run_analysis(f"Global {label} ({n_leagues} leagues)", y_te, p_naive, p_model, p_market)
        r["n_leagues"] = n_leagues
        part2_results.append(r)

    results["part2_global_proxy"] = part2_results

    # ─── PART 3: LATAM proxy ─────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  PART 3: LATAM PROXY (low-signal leagues with odds)")
    print(f"{'='*70}")

    part3_results = []
    for label, train_f, test_f in season_splits:
        df_train = df[df["date"].apply(train_f)].dropna(subset=features)
        df_test_latam = df[
            (df["date"].apply(test_f)) &
            (df["league_id"].isin(LATAM_LEAGUES))
        ].dropna(subset=features)
        df_test_odds = df_test_latam[
            (df_test_latam["odds_home"].notna()) &
            (df_test_latam["odds_home"] > 1)
        ]

        if len(df_test_odds) < 20:
            print(f"\n  {label}: SKIP (N={len(df_test_odds)} LATAM with odds)")
            part3_results.append({"split": label, "status": "SKIPPED", "n": len(df_test_odds)})
            continue

        X_tr = df_train[features].values.astype(np.float32)
        y_tr = df_train["result"].values.astype(int)
        X_te = df_test_odds[features].values.astype(np.float32)
        y_te = df_test_odds["result"].values.astype(int)

        p_model = get_ensemble_probs(X_tr, y_tr, X_te)
        p_market = odds_to_probs(
            df_test_odds["odds_home"].values,
            df_test_odds["odds_draw"].values,
            df_test_odds["odds_away"].values,
        )
        p_naive = naive_probs(y_tr, len(y_te))

        leagues_present = df_test_odds["league_id"].unique().tolist()
        r = run_analysis(f"LATAM {label}", y_te, p_naive, p_model, p_market)
        r["leagues"] = leagues_present
        part3_results.append(r)

    results["part3_latam_proxy"] = part3_results

    # ─── PART 4: Per-league α* (where we have odds) ─────────
    print(f"\n{'='*70}")
    print(f"  PART 4: PER-LEAGUE α* (where odds available, ≤2025→2026)")
    print(f"{'='*70}")

    df_train = df[df["date"].dt.year <= 2025].dropna(subset=features)
    df_test_2026 = df[
        (df["date"].dt.year == 2026) &
        (df["odds_home"].notna()) & (df["odds_home"] > 1)
    ].dropna(subset=features)

    X_tr = df_train[features].values.astype(np.float32)
    y_tr = df_train["result"].values.astype(int)

    part4 = []
    for lid in sorted(df_test_2026["league_id"].unique()):
        subset = df_test_2026[df_test_2026["league_id"] == lid]
        if len(subset) < 10:
            continue

        X_te = subset[features].values.astype(np.float32)
        y_te = subset["result"].values.astype(int)

        p_model = get_ensemble_probs(X_tr, y_tr, X_te)
        p_market = odds_to_probs(subset["odds_home"].values, subset["odds_draw"].values, subset["odds_away"].values)
        p_naive = naive_probs(y_tr, len(y_te))

        brier_model = multiclass_brier(y_te, p_model)
        brier_market = multiclass_brier(y_te, p_market)
        brier_naive = multiclass_brier(y_te, p_naive)
        alpha, brier_blend, _ = shrinkage_grid(y_te, p_model, p_market)

        entry = {
            "league_id": int(lid),
            "n": len(subset),
            "brier_naive": round(brier_naive, 4),
            "brier_model": round(brier_model, 4),
            "brier_market": round(brier_market, 4),
            "brier_blend": round(brier_blend, 4),
            "alpha_star": alpha,
            "model_vs_market": round(brier_market - brier_model, 4),
        }
        part4.append(entry)

        marker = "★" if brier_model < brier_market else "⚠"
        print(f"  {marker} League {lid:>3}: N={len(subset):>3} | Model={brier_model:.4f} Market={brier_market:.4f} | α*={alpha:.2f} Blend={brier_blend:.4f} | M-vs-Mkt={brier_market-brier_model:+.4f}")

    results["part4_per_league"] = part4

    # ─── SUMMARY ─────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  SUMMARY & RECOMMENDATION")
    print(f"{'='*70}")

    print(f"\n  CAVEAT: Argentina has 0 historical odds pre-2026.")
    print(f"  Part 1 (N≈50) is directionally useful but not statistically robust.")
    print(f"  Parts 2-4 use global/per-league data as calibration proxy.")

    if "part1_argentina_2026" in results and "alpha_model_market" in results["part1_argentina_2026"]:
        r1 = results["part1_argentina_2026"]
        print(f"\n  Argentina 2026 direct:")
        print(f"    α*(model+market) = {r1['alpha_model_market']:.2f}")
        print(f"    Market alone:      {r1['brier_market']:.4f}")
        print(f"    Model alone:       {r1['brier_model']:.4f}")
        print(f"    Best blend:        {r1['brier_blend_best']:.4f}")

    # Save
    output_file = Path("scripts/output/experiment_market_anchor.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved to: {output_file}")


if __name__ == "__main__":
    main()
