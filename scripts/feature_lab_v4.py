#!/usr/bin/env python3
"""Feature Lab V4 — Section Y Grid: V1.2.0 Candidate Search

GDT Directive: 3 Tracks with strict architecture separation.
  Track A (Fundamental): TwoStage dw=1.0, 14f baseline + decomposed features
  Track B (Market):      OneStage dw=1.0 (PARAMS_OS_FROM_TS), 14f + market_div
  Track C (Family S):    OneStage Optuna, best-of + odds + market_div + MTV

Doctrina GDT: Market-derived features ONLY on OneStage.
              TwoStage Baseline = 100% odds-blind.

Usage:
    source .env
    python scripts/feature_lab_v4.py
"""

import gc
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import brier_score_loss

sys.path.insert(0, str(Path(__file__).parent.parent))

# ─── Constants ─────────────────────────────────────────────────────────────────
MIN_DATE = "2023-01-01"
LAB_DATA_DIR = Path(__file__).parent / "output" / "lab"
N_SEEDS = 5
N_BOOTSTRAP = 1000
TEST_FRACTION = 0.2
DRAW_WEIGHT = 1.0

ALL_LEAGUES = {
    39, 40, 61, 71, 78, 88, 94, 128, 135, 140, 144,
    203, 239, 242, 250, 253, 262, 265, 268, 281, 299, 307, 344,
}

# ─── Feature Sets ──────────────────────────────────────────────────────────────
FEATURES_14 = [
    "home_goals_scored_avg", "home_goals_conceded_avg",
    "home_shots_avg", "home_corners_avg",
    "home_rest_days", "home_matches_played",
    "away_goals_scored_avg", "away_goals_conceded_avg",
    "away_shots_avg", "away_corners_avg",
    "away_rest_days", "away_matches_played",
    "goal_diff_avg", "rest_diff",
]

FEATURES_3_ODDS = ["odds_home", "odds_draw", "odds_away"]

MTV_FEATURES = [
    "home_talent_delta", "away_talent_delta",
    "talent_delta_diff", "shock_magnitude",
]

SOTA_FEATURES = [
    "xg_luck_residual_home", "xg_luck_residual_away", "xg_luck_residual_diff",
    "syndicate_steam_mtv_home", "syndicate_steam_mtv_away",
    "allostatic_load_home", "allostatic_load_away", "allostatic_load_diff"
]

# ─── Hyperparameters (exact production values) ─────────────────────────────────

# TwoStage Stage 1 (production v1.0.3)
PARAMS_TS_S1 = {
    "objective": "binary:logistic",
    "max_depth": 3,
    "learning_rate": 0.05,
    "n_estimators": 100,
    "min_child_weight": 7,
    "subsample": 0.72,
    "colsample_bytree": 0.71,
    "verbosity": 0,
}

# TwoStage Stage 2 (production v1.0.3)
PARAMS_TS_S2 = {
    "objective": "binary:logistic",
    "max_depth": 3,
    "learning_rate": 0.05,
    "n_estimators": 100,
    "min_child_weight": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "verbosity": 0,
}

# OneStage adapted from TS (for Track B: Y3/Y4)
PARAMS_OS_FROM_TS = {
    "objective": "multi:softprob",
    "num_class": 3,
    "max_depth": 3,
    "learning_rate": 0.05,
    "n_estimators": 100,
    "min_child_weight": 7,
    "subsample": 0.72,
    "colsample_bytree": 0.71,
    "eval_metric": "mlogloss",
    "verbosity": 0,
}

# Production Optuna-optimized (for Track C: Y9 Family S)
PARAMS_OPTUNA = {
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
    "eval_metric": "mlogloss",
    "verbosity": 0,
}


# ─── Y-Section Test Grid ──────────────────────────────────────────────────────

Y_GRID = [
    # Track A: Fundamental (TwoStage dw=1.0, home_away semantic, no odds)
    {"id": "Y0", "name": "Baseline 14f (control)", "track": "A",
     "features": FEATURES_14, "arch": "twostage"},
    {"id": "Y1", "name": "+elo_att_diff, elo_def_diff", "track": "A",
     "features": FEATURES_14 + ["elo_att_diff", "elo_def_diff"],
     "arch": "twostage"},
    {"id": "Y2", "name": "+4x elo decomposed", "track": "A",
     "features": FEATURES_14 + ["elo_att_diff", "elo_def_diff",
                                 "elo_att_vs_def", "elo_def_vs_att"],
     "arch": "twostage"},
    {"id": "Y5", "name": "+league_draw_rate, draw_propensity", "track": "A",
     "features": FEATURES_14 + ["league_draw_rate", "draw_propensity"],
     "arch": "twostage"},
    {"id": "Y6", "name": "+elo_k10_diff, elo_momentum_diff", "track": "A",
     "features": FEATURES_14 + ["elo_k10_diff", "elo_momentum_diff"],
     "arch": "twostage"},
    {"id": "Y7", "name": "+matchup attack vs defense", "track": "A",
     "features": FEATURES_14 + ["matchup_h_attack_v_a_defense",
                                 "matchup_a_attack_v_h_defense"],
     "arch": "twostage"},

    # Track B: Market (OneStage dw=1.0, PARAMS_OS_FROM_TS)
    {"id": "Y3", "name": "+market_div_home, market_div_away", "track": "B",
     "features": FEATURES_14 + ["market_div_home", "market_div_away"],
     "arch": "onestage"},
    {"id": "Y4", "name": "+4x market_div", "track": "B",
     "features": FEATURES_14 + ["market_div_home", "market_div_away",
                                 "market_div_abs", "market_div_sign"],
     "arch": "onestage"},
]
# Y8 (best fundamental) and Y9 (Family S) are composed dynamically.


# ─── Utilities ─────────────────────────────────────────────────────────────────

def calculate_brier(y_true, y_proba):
    """Multiclass Brier score (average of per-class brier_score_loss)."""
    scores = []
    for cls in range(y_proba.shape[1]):
        y_bin = (y_true == cls).astype(int)
        scores.append(brier_score_loss(y_bin, y_proba[:, cls]))
    return float(np.mean(scores))


def bootstrap_paired_delta(y_true, probs_ctrl, probs_chal,
                           n_boot=N_BOOTSTRAP, seed=42):
    """Bootstrap 95% CI for Δ(challenger - control) using paired samples.

    Per-match Brier is averaged across classes (not summed) to match
    the scale of calculate_brier().
    """
    rng = np.random.RandomState(seed)
    n = len(y_true)
    n_classes = probs_ctrl.shape[1]
    y_oh = np.eye(n_classes)[y_true.astype(int)]

    ctrl_per = np.mean((probs_ctrl - y_oh) ** 2, axis=1)
    chal_per = np.mean((probs_chal - y_oh) ** 2, axis=1)
    delta_per = chal_per - ctrl_per

    deltas = []
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        deltas.append(float(np.mean(delta_per[idx])))

    return float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))


def train_twostage(X_train, y_train, X_test, seed=42):
    """Train TwoStage (home_away semantic, no odds in features).

    Returns (N, 3) probs [p_home, p_draw, p_away].
    """
    # Stage 1: draw (1) vs non-draw (0)
    y_s1 = (y_train == 1).astype(int)
    sw_s1 = np.ones(len(y_s1), dtype=np.float32)
    sw_s1[y_s1 == 1] = DRAW_WEIGHT

    s1 = xgb.XGBClassifier(**{**PARAMS_TS_S1, "random_state": seed})
    s1.fit(X_train, y_s1, sample_weight=sw_s1, verbose=False)

    # Stage 2: home won (1) vs away won (0), non-draw only
    nd_mask = y_train != 1
    X_s2 = X_train[nd_mask]
    y_s2 = (y_train[nd_mask] == 0).astype(int)

    s2 = xgb.XGBClassifier(**{**PARAMS_TS_S2, "random_state": seed})
    s2.fit(X_s2, y_s2, verbose=False)

    # Predict
    p_draw = s1.predict_proba(X_test)[:, 1]
    p_home_nd = s2.predict_proba(X_test)[:, 1]
    p_home = (1 - p_draw) * p_home_nd
    p_away = (1 - p_draw) * (1 - p_home_nd)

    return np.column_stack([p_home, p_draw, p_away])


def train_onestage(X_train, y_train, X_test, params, seed=42):
    """Train OneStage. Returns (N, 3) probs."""
    sw = np.ones(len(y_train), dtype=np.float32)
    sw[y_train == 1] = DRAW_WEIGHT

    model = xgb.XGBClassifier(**{**params, "random_state": seed})
    model.fit(X_train, y_train, sample_weight=sw, verbose=False)
    return model.predict_proba(X_test)


def load_pooled_data():
    """Load all 23 leagues from Lab CSVs and pool."""
    frames = []
    for lid in sorted(ALL_LEAGUES):
        csv_path = LAB_DATA_DIR / f"lab_data_{lid}.csv"
        if not csv_path.exists():
            print(f"  WARNING: Missing CSV for league {lid}")
            continue

        df = pd.read_csv(csv_path)
        df = df[df["date"] >= MIN_DATE].copy()
        df = df[df["result"].notna()].copy()
        frames.append(df)
        print(f"  League {lid}: {len(df)} matches")

    if not frames:
        raise RuntimeError("No data loaded!")

    pooled = pd.concat(frames, ignore_index=True)
    pooled = pooled.sort_values("date").reset_index(drop=True)
    return pooled


def evaluate_test(test_def, X_train, y_train, X_test, y_test, feature_cols):
    """Evaluate a single Y-test with multi-seed ensemble."""
    features = test_def["features"]
    arch = test_def["arch"]

    feat_idx = [feature_cols.index(f) for f in features]
    X_tr = X_train[:, feat_idx]
    X_te = X_test[:, feat_idx]

    all_probs = []
    all_briers = []

    for seed_i in range(N_SEEDS):
        seed = seed_i * 42 + 7

        if arch == "twostage":
            probs = train_twostage(X_tr, y_train, X_te, seed=seed)
        elif arch == "onestage":
            probs = train_onestage(X_tr, y_train, X_te, PARAMS_OS_FROM_TS,
                                   seed=seed)
        elif arch == "onestage_optuna":
            probs = train_onestage(X_tr, y_train, X_te, PARAMS_OPTUNA,
                                   seed=seed)
        else:
            raise ValueError(f"Unknown arch: {arch}")

        all_probs.append(probs)
        all_briers.append(calculate_brier(y_test, probs))

    ensemble = np.mean(all_probs, axis=0)
    brier = calculate_brier(y_test, ensemble)

    return {
        "brier": brier,
        "brier_mean": float(np.mean(all_briers)),
        "brier_std": float(np.std(all_briers)),
        "ensemble_probs": ensemble,
        "n_features": len(features),
        "features": features,
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  FEATURE LAB V4 — SECTION Y GRID")
    print("  Track A: Fundamental (TwoStage dw=1.0, home_away)")
    print("  Track B: Market (OneStage dw=1.0, PARAMS_OS_FROM_TS)")
    print("  Track C: Family S V1.2 (OneStage Optuna)")
    print("=" * 70)

    # ── 1. Load data ──────────────────────────────────────────
    print("\n[1] Loading Lab CSVs...")
    df = load_pooled_data()
    print(f"  Pooled: {len(df)} matches")

    # Filter to odds-available (needed for market_div features + apples-to-apples)
    df = df[
        (df["odds_home"].notna()) & (df["odds_home"] > 0) &
        (df["odds_draw"].notna()) & (df["odds_draw"] > 0) &
        (df["odds_away"].notna()) & (df["odds_away"] > 0)
    ].copy()
    df = df.sort_values("date").reset_index(drop=True)
    print(f"  After odds filter: {len(df)} matches")

    # ── 2. Merge MTV features ─────────────────────────────────
    mtv_path = Path("data/historical_mtv_features.parquet")
    has_mtv = False
    if mtv_path.exists():
        mtv_df = pd.read_parquet(mtv_path, columns=[
            "match_id", "home_talent_delta", "away_talent_delta",
            "talent_delta_diff", "shock_magnitude",
        ])
        df["match_id"] = df["match_id"].astype("Int64")
        mtv_df["match_id"] = mtv_df["match_id"].astype("Int64")
        df = df.merge(mtv_df, on="match_id", how="left")
        n_mtv = int(df["home_talent_delta"].notna().sum())
        has_mtv = n_mtv > 0
        print(f"  MTV merged: {n_mtv}/{len(df)} ({100*n_mtv/len(df):.1f}%)")
    else:
        print(f"  MTV: parquet not found, Y9 will be skipped")
        for col in MTV_FEATURES:
            df[col] = np.nan

    # ── 2.5 Compute SOTA Features ─────────────────────────────
    print(f"\n[2.5] Computing SOTA Brier-Squeeze Features...")
    
    # A. xG Luck Residuals (The Theorem of Regression)
    df["xg_luck_residual_home"] = (df["home_goals_scored_avg"] - df["home_xg_for_avg"]) - (df["home_goals_conceded_avg"] - df["home_xg_against_avg"])
    df["xg_luck_residual_away"] = (df["away_goals_scored_avg"] - df["away_xg_for_avg"]) - (df["away_goals_conceded_avg"] - df["away_xg_against_avg"])
    df["xg_luck_residual_diff"] = df["xg_luck_residual_home"] - df["xg_luck_residual_away"]
    
    # B. Syndicate Steam vs MTV (Smart Money Cross)
    # Steam: opening probability - closing probability. Positive means money flowed IN (prob went up).
    # Need to handle potential division by zero or missing open odds
    steam_home = np.where(df["odds_home_open"].notna() & (df["odds_home_open"] > 0),
                          (1.0 / df["odds_home_close"].fillna(df["odds_home"])) - (1.0 / df["odds_home_open"]), 0.0)
    steam_away = np.where(df["odds_away_open"].notna() & (df["odds_away_open"] > 0),
                          (1.0 / df["odds_away_close"].fillna(df["odds_away"])) - (1.0 / df["odds_away_open"]), 0.0)
    
    df["syndicate_steam_mtv_home"] = steam_home * df["home_talent_delta"].fillna(0)
    df["syndicate_steam_mtv_away"] = steam_away * df["away_talent_delta"].fillna(0)
    
    # C. Non-Linear Allostatic Load (Compound Fatigue Tensor)
    # Avoid exp overflow by capping rest days at a reasonable number (e.g., 14)
    capped_rest_home = np.clip(df["home_rest_days"], 0, 14)
    capped_rest_away = np.clip(df["away_rest_days"], 0, 14)
    alt_home = df["altitude_home_m"].fillna(0)
    travel = df["travel_distance_km"].fillna(0)
    
    # Away team bears the travel and altitude change
    df["allostatic_load_away"] = (travel * (1 + (alt_home / 1000.0))) / np.exp(capped_rest_away)
    # Home team bears no travel, but lack of rest still hurts
    df["allostatic_load_home"] = 1.0 / np.exp(capped_rest_home)
    df["allostatic_load_diff"] = df["allostatic_load_home"] - df["allostatic_load_away"]

    # ── 3. Build feature matrix ───────────────────────────────
    all_features = set()
    for t in Y_GRID:
        all_features.update(t["features"])
    all_features.update(FEATURES_3_ODDS)
    all_features.update(MTV_FEATURES)
    all_features.update(SOTA_FEATURES)
    all_features.update(["market_div_abs", "market_div_sign",
                         "market_div_home", "market_div_away"])

    # Verify columns exist
    missing = [f for f in all_features if f not in df.columns]
    if missing:
        print(f"  ERROR: Missing columns in CSVs: {missing}")
        sys.exit(1)

    feature_cols = sorted(all_features & set(df.columns))

    # ── 4. Temporal split ─────────────────────────────────────
    n = len(df)
    split_idx = int(n * (1 - TEST_FRACTION))

    X_all = df[feature_cols].fillna(0).values
    y_all = df["result"].values.astype(int)

    X_train, X_test = X_all[:split_idx], X_all[split_idx:]
    y_train, y_test = y_all[:split_idx], y_all[split_idx:]

    print(f"\n[2] Data split:")
    print(f"  Train: {split_idx} ({df['date'].iloc[0]} → {df['date'].iloc[split_idx-1]})")
    print(f"  Test:  {n - split_idx} ({df['date'].iloc[split_idx]} → {df['date'].iloc[-1]})")

    # ── 5. Run Y0-Y7 ─────────────────────────────────────────
    print(f"\n[3] Running Y-Grid (Y0-Y7)...")
    results = {}

    for test_def in Y_GRID:
        tid = test_def["id"]
        nf = len(test_def["features"])
        print(f"  {tid}: {test_def['name']} "
              f"({test_def['arch']}, {nf}f)...", end="", flush=True)

        res = evaluate_test(test_def, X_train, y_train, X_test, y_test,
                            feature_cols)
        results[tid] = res
        print(f" Brier={res['brier']:.5f}")

    # ── 6. Paired delta CI vs Y0 ─────────────────────────────
    print(f"\n[4] Computing paired delta CIs vs Y0...")
    y0_probs = results["Y0"]["ensemble_probs"]

    for tid, res in results.items():
        if tid == "Y0":
            res["delta"] = 0.0
            res["delta_ci95"] = [0.0, 0.0]
        else:
            delta = res["brier"] - results["Y0"]["brier"]
            ci_lo, ci_hi = bootstrap_paired_delta(
                y_test, y0_probs, res["ensemble_probs"])
            res["delta"] = delta
            res["delta_ci95"] = [ci_lo, ci_hi]

    # ── 7. Y8: Best of Track A challengers ────────────────────
    track_a_ids = ["Y1", "Y2", "Y5", "Y6", "Y7"]
    best_a_id = min(track_a_ids, key=lambda t: results[t]["brier"])
    best_a_def = next(t for t in Y_GRID if t["id"] == best_a_id)
    y8_features = best_a_def["features"]

    print(f"\n[5] Y8 = Best fundamental: {best_a_id} ({best_a_def['name']})")
    # Y8 inherits results from best_a
    results["Y8"] = {
        **{k: v for k, v in results[best_a_id].items()},
        "source": best_a_id,
    }

    # ── 8. Track B winner ─────────────────────────────────────
    track_b_ids = ["Y3", "Y4"]
    best_b_id = min(track_b_ids, key=lambda t: results[t]["brier"])
    best_b_def = next(t for t in Y_GRID if t["id"] == best_b_id)
    best_market_features = [f for f in best_b_def["features"]
                            if f.startswith("market_div")]

    # ── 9. Y9: Family S V1.2 ─────────────────────────────────
    if has_mtv:
        # Compose: Y8 fundamentals + odds + best market_div + MTV
        y9_raw = (y8_features + FEATURES_3_ODDS
                  + best_market_features + MTV_FEATURES)
        seen = set()
        y9_features = []
        for f in y9_raw:
            if f not in seen:
                y9_features.append(f)
                seen.add(f)

        print(f"\n[6] Y9: Family S V1.2 ({len(y9_features)}f, onestage_optuna)")
        print(f"    Composition: {best_a_id} ({len(y8_features)}f) "
              f"+ odds(3) + {best_b_id}_market({len(best_market_features)}) "
              f"+ MTV(4)")

        # Filter to MTV-available subset
        mtv_mask_all = df["home_talent_delta"].notna().values
        mtv_mask_train = mtv_mask_all[:split_idx]
        mtv_mask_test = mtv_mask_all[split_idx:]

        X_tr_y9 = X_train[mtv_mask_train]
        y_tr_y9 = y_train[mtv_mask_train]
        X_te_y9 = X_test[mtv_mask_test]
        y_te_y9 = y_test[mtv_mask_test]

        print(f"    MTV subset: train={len(X_tr_y9)}, test={len(X_te_y9)}")

        if len(X_te_y9) < 100:
            print(f"    SKIP Y9: insufficient MTV test data ({len(X_te_y9)})")
        else:
            # Y0 baseline on same MTV subset (for fair delta)
            feat_idx_y0 = [feature_cols.index(f) for f in FEATURES_14]
            probs_y0_mtv = []
            for seed_i in range(N_SEEDS):
                seed = seed_i * 42 + 7
                p = train_twostage(X_tr_y9[:, feat_idx_y0], y_tr_y9,
                                   X_te_y9[:, feat_idx_y0], seed=seed)
                probs_y0_mtv.append(p)
            y0_mtv_ensemble = np.mean(probs_y0_mtv, axis=0)
            y0_mtv_brier = calculate_brier(y_te_y9, y0_mtv_ensemble)

            # Run Y9
            feat_idx_y9 = [feature_cols.index(f) for f in y9_features]
            probs_y9 = []
            briers_y9 = []
            for seed_i in range(N_SEEDS):
                seed = seed_i * 42 + 7
                p = train_onestage(X_tr_y9[:, feat_idx_y9], y_tr_y9,
                                   X_te_y9[:, feat_idx_y9], PARAMS_OPTUNA,
                                   seed=seed)
                probs_y9.append(p)
                briers_y9.append(calculate_brier(y_te_y9, p))

            y9_ensemble = np.mean(probs_y9, axis=0)
            y9_brier = calculate_brier(y_te_y9, y9_ensemble)
            y9_delta = y9_brier - y0_mtv_brier
            y9_ci = bootstrap_paired_delta(y_te_y9, y0_mtv_ensemble,
                                           y9_ensemble)

            results["Y9"] = {
                "brier": y9_brier,
                "brier_mean": float(np.mean(briers_y9)),
                "brier_std": float(np.std(briers_y9)),
                "delta": y9_delta,
                "delta_ci95": list(y9_ci),
                "n_features": len(y9_features),
                "features": y9_features,
                "n_train_mtv": len(X_tr_y9),
                "n_test_mtv": len(X_te_y9),
                "y0_mtv_brier": y0_mtv_brier,
                "source_fundamental": best_a_id,
                "source_market": best_b_id,
            }
            print(f"    Brier={y9_brier:.5f}, Y0_mtv={y0_mtv_brier:.5f}, "
                  f"Δ={y9_delta:+.5f}")
            
            # ── 9.5 Y10: SOTA Experimento ─────────────────────────────
            y10_raw = y9_raw + SOTA_FEATURES
            y10_features = []
            seen_y10 = set()
            for f in y10_raw:
                if f not in seen_y10:
                    y10_features.append(f)
                    seen_y10.add(f)
                    
            print(f"\n[6.5] Y10: SOTA Brier Squeeze ({len(y10_features)}f, onestage_optuna)")
            print(f"      Composition: Y9 ({len(y9_features)}f) + SOTA({len(SOTA_FEATURES)})")
            
            feat_idx_y10 = [feature_cols.index(f) for f in y10_features]
            probs_y10 = []
            briers_y10 = []
            for seed_i in range(N_SEEDS):
                seed = seed_i * 42 + 7
                p = train_onestage(X_tr_y9[:, feat_idx_y10], y_tr_y9,
                                   X_te_y9[:, feat_idx_y10], PARAMS_OPTUNA,
                                   seed=seed)
                probs_y10.append(p)
                briers_y10.append(calculate_brier(y_te_y9, p))
                
            y10_ensemble = np.mean(probs_y10, axis=0)
            y10_brier = calculate_brier(y_te_y9, y10_ensemble)
            y10_delta = y10_brier - y0_mtv_brier
            y10_ci = bootstrap_paired_delta(y_te_y9, y0_mtv_ensemble, y10_ensemble)
            
            results["Y10"] = {
                "brier": y10_brier,
                "brier_mean": float(np.mean(briers_y10)),
                "brier_std": float(np.std(briers_y10)),
                "delta": y10_delta,
                "delta_ci95": list(y10_ci),
                "n_features": len(y10_features),
                "features": y10_features,
                "n_train_mtv": len(X_tr_y9),
                "n_test_mtv": len(X_te_y9),
                "y0_mtv_brier": y0_mtv_brier,
                "source_fundamental": best_a_id,
                "source_market": best_b_id,
            }
            print(f"      Brier={y10_brier:.5f}, Y0_mtv={y0_mtv_brier:.5f}, Δ={y10_delta:+.5f}")
    else:
        print(f"\n[6] Y9: SKIPPED (no MTV data)")

    # ── 10. Results table ─────────────────────────────────────
    print(f"\n{'=' * 90}")
    print(f"  SECTION Y — RESULTS TABLE (N_train={split_idx}, N_test={n - split_idx})")
    print(f"{'=' * 90}")
    print(f"  {'Test':<6} {'Track':<3} {'Arch':<12} {'#f':>3} "
          f"{'Brier':>8} {'Δ vs Y0':>8} {'CI95_lo':>8} {'CI95_hi':>8} {'Sig':>4}")
    print(f"  {'-'*6} {'-'*3} {'-'*12} {'-'*3} "
          f"{'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*4}")

    ordered = ["Y0", "Y1", "Y2", "Y5", "Y6", "Y7", "Y8", "Y3", "Y4"]
    if "Y9" in results:
        ordered.append("Y9")
    if "Y10" in results:
        ordered.append("Y10")

    for tid in ordered:
        res = results[tid]
        test_def = next((t for t in Y_GRID if t["id"] == tid), None)

        if test_def:
            arch = test_def["arch"]
            track = test_def["track"]
            nf = len(test_def["features"])
        elif tid == "Y8":
            arch = "twostage"
            track = "A"
            nf = len(y8_features)
        elif tid == "Y9":
            arch = "os_optuna"
            track = "C"
            nf = res["n_features"]
        elif tid == "Y10":
            arch = "os_optuna"
            track = "C"
            nf = res["n_features"]
        else:
            arch = "?"
            track = "?"
            nf = 0

        delta = res["delta"]
        ci = res["delta_ci95"]

        if tid == "Y0":
            sig = "CTRL"
        elif ci[1] < 0:
            sig = "**"
        elif ci[0] > 0:
            sig = "!!"
        else:
            sig = "ns"

        # Special annotation for Y8, Y9, Y10
        suffix = ""
        if tid == "Y8":
            suffix = f" ← {best_a_id}"
        elif tid == "Y9":
            suffix = f" (mtv N={res.get('n_test_mtv', '?')})"
        elif tid == "Y10":
            suffix = f" (mtv N={res.get('n_test_mtv', '?')}) [SOTA]"

        print(f"  {tid:<6} {track:<3} {arch:<12} {nf:>3} "
              f"{res['brier']:>8.5f} {delta:>+8.5f} "
              f"{ci[0]:>+8.5f} {ci[1]:>+8.5f} {sig:>4}{suffix}")

    print(f"\n  Legend:")
    print(f"    ** = CI95 entirely <0 (challenger BEATS Y0)")
    print(f"    !! = CI95 entirely >0 (Y0 BEATS challenger)")
    print(f"    ns = not significant (CI95 crosses 0)")
    print(f"    Y8 = best Track A fundamental (source: {best_a_id})")
    if "Y9" in results:
        print(f"    Y9 = {best_a_id} + odds + {best_b_id} market_div + MTV "
              f"(delta vs Y0 on MTV subset)")

    # ── 11. Save JSON ─────────────────────────────────────────
    output = {
        "meta": {
            "timestamp": datetime.now().isoformat(),
            "n_seeds": N_SEEDS,
            "n_bootstrap": N_BOOTSTRAP,
            "test_fraction": TEST_FRACTION,
            "draw_weight": DRAW_WEIGHT,
            "min_date": MIN_DATE,
            "n_leagues": len(ALL_LEAGUES),
            "n_matches": n,
            "n_train": split_idx,
            "n_test": n - split_idx,
            "params_ts_s1": PARAMS_TS_S1,
            "params_ts_s2": PARAMS_TS_S2,
            "params_os_from_ts": PARAMS_OS_FROM_TS,
            "params_optuna": PARAMS_OPTUNA,
        },
        "results": {
            tid: {k: v for k, v in res.items() if k != "ensemble_probs"}
            for tid, res in results.items()
        },
        "y8_source": best_a_id,
        "y9_market_source": best_b_id if "Y9" in results else None,
    }

    output_path = LAB_DATA_DIR / "feature_lab_v4_section_y.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Results saved to: {output_path}")

    gc.collect()


if __name__ == "__main__":
    main()
