#!/usr/bin/env python3
"""
GDT Mega-Pool Analysis — Súper-Tensor de Ligas Periféricas

Mandato 1: Pooled ΔBrier (control vs MTV vs market) con bootstrap CI95
Mandato 2: Brier condicionado a shock_magnitude >= P90 global
Mandato 4: Injury era stratification (pre vs post Jul 2025) — ABE P1-1

Concatenates per-match OOT predictions from all peripheral leagues into a
"Super-Tensor", then computes pooled metrics.

Usage:
    source .env
    python3 scripts/mega_pool_analysis.py
    python3 scripts/mega_pool_analysis.py --include-conference  # include Conference League
"""
import argparse
import json
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import xgboost as xgb

warnings.filterwarnings("ignore", category=FutureWarning)

# ─── Constants (replicated from feature_lab.py) ──────────────

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

DRAW_WEIGHT = 1.5
N_SEEDS = 5
TEST_FRACTION = 0.2
N_BOOTSTRAP = 2000
INJURY_ERA_CUTOFF = "2025-07-01"

# ─── League filters ──────────────────────────────────────────

BIG5 = {39, 140, 78, 135, 61}
INTL = {2, 3, 4, 5, 6, 7, 10, 11, 13, 22, 28, 29, 30, 31, 32, 33, 34}
UEFA_CLUB = {848}

# GDT Tier 3: 10 leagues where MTV HELPS in Pair B (Elo+Odds)
TIER3_WINNERS = {265, 94, 268, 88, 203, 281, 299, 344, 262, 242}

LEAGUE_NAMES = {
    40: "Championship", 71: "Brazil Serie A", 88: "Eredivisie",
    94: "Primeira Liga", 128: "Argentina", 144: "Belgian Pro League",
    203: "Süper Lig", 239: "Colombia", 242: "Ecuador",
    250: "Paraguay Ap.", 252: "Paraguay Cl.", 253: "MLS",
    262: "Mexico Liga MX", 265: "Chile", 268: "Uruguay Ap.",
    270: "Uruguay Cl.", 281: "Peru", 299: "Venezuela",
    307: "Saudi Pro League", 344: "Bolivia", 848: "Conference League",
}

# ─── Feature definitions ─────────────────────────────────────

ELO_FEATURES = ["elo_home", "elo_away", "elo_diff"]
ODDS_FEATURES = ["odds_home", "odds_draw", "odds_away"]
FORM_CORE = ["home_win_rate5", "away_win_rate5", "form_diff"]
DEFENSE_PAIR = ["home_goals_conceded_avg", "away_goals_conceded_avg"]
MTV_FEATURES = [
    "home_talent_delta", "away_talent_delta",
    "talent_delta_diff", "shock_magnitude",
]

# Test pairs: control (without MTV) vs treatment (with MTV)
TEST_PAIRS = {
    "A_elo": {
        "name": "Elo",
        "control": ELO_FEATURES,
        "mtv": ELO_FEATURES + MTV_FEATURES,
        "requires_odds": False,
    },
    "B_elo_odds": {
        "name": "Elo+Odds",
        "control": ELO_FEATURES + ODDS_FEATURES,
        "mtv": ELO_FEATURES + ODDS_FEATURES + MTV_FEATURES,
        "requires_odds": True,
    },
    "C_full": {
        "name": "Full (Elo+Def+Form+Odds)",
        "control": ELO_FEATURES + DEFENSE_PAIR + FORM_CORE + ODDS_FEATURES,
        "mtv": ELO_FEATURES + DEFENSE_PAIR + FORM_CORE + ODDS_FEATURES + MTV_FEATURES,
        "requires_odds": True,
    },
}


# ─── Core functions ──────────────────────────────────────────

def multiclass_brier(y_true, y_prob):
    """Multiclass Brier score (lower = better)."""
    n_classes = y_prob.shape[1]
    y_onehot = np.eye(n_classes)[y_true.astype(int)]
    return float(np.mean(np.sum((y_prob - y_onehot) ** 2, axis=1)))


def train_xgb(X_train, y_train, seed=42):
    """Train XGBoost with production hyperparams."""
    params = {**PROD_HYPERPARAMS, "random_state": seed}
    model = xgb.XGBClassifier(**params)
    sw = np.ones(len(y_train), dtype=np.float32)
    sw[y_train == 1] = DRAW_WEIGHT
    model.fit(X_train, y_train, sample_weight=sw, verbose=False)
    return model


def ensemble_predict(X_train, y_train, X_test):
    """Train N_SEEDS models and return ensemble (averaged) probabilities."""
    all_probs = []
    for seed_i in range(N_SEEDS):
        seed = seed_i * 42 + 7
        model = train_xgb(X_train, y_train, seed=seed)
        all_probs.append(model.predict_proba(X_test))
    return np.mean(all_probs, axis=0)


def devig_market_probs(df):
    """De-vig closing odds to probabilities."""
    inv_h = 1.0 / df["odds_home"].values
    inv_d = 1.0 / df["odds_draw"].values
    inv_a = 1.0 / df["odds_away"].values
    total = inv_h + inv_d + inv_a
    return np.column_stack([inv_h / total, inv_d / total, inv_a / total])


def per_match_brier(y_true, y_prob):
    """Per-match Brier contributions (vector, not scalar)."""
    n_classes = y_prob.shape[1]
    y_onehot = np.eye(n_classes)[y_true.astype(int)]
    return np.sum((y_prob - y_onehot) ** 2, axis=1)


def bootstrap_paired_delta_ci(brier_a, brier_b, n_bootstrap=N_BOOTSTRAP, seed=42):
    """Bootstrap CI95 for Δ = mean(brier_a) - mean(brier_b).

    Positive Δ means A is worse (higher Brier) than B.
    """
    rng = np.random.RandomState(seed)
    n = len(brier_a)
    delta = brier_a - brier_b
    deltas = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        deltas.append(float(np.mean(delta[idx])))
    return {
        "delta_mean": float(np.mean(delta)),
        "ci95_lo": float(np.percentile(deltas, 2.5)),
        "ci95_hi": float(np.percentile(deltas, 97.5)),
        "significant": not (np.percentile(deltas, 2.5) <= 0 <= np.percentile(deltas, 97.5)),
    }


def bootstrap_brier_ci(brier_per_match, n_bootstrap=N_BOOTSTRAP, seed=42):
    """Bootstrap CI95 for a Brier score."""
    rng = np.random.RandomState(seed)
    n = len(brier_per_match)
    means = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        means.append(float(np.mean(brier_per_match[idx])))
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


# ─── Per-league OOT extraction ───────────────────────────────

def process_league(league_id, lab_dir, mtv_df):
    """Process one league: load data, merge MTV, split, train, collect OOT predictions."""
    csv_path = Path(lab_dir) / f"lab_data_{league_id}.csv"
    if not csv_path.exists():
        print(f"  SKIP {LEAGUE_NAMES.get(league_id, league_id)}: no lab_data CSV")
        return None

    df = pd.read_csv(csv_path, parse_dates=["date"])
    df["match_id"] = df["match_id"].astype("Int64")

    # Merge MTV features
    df = df.merge(
        mtv_df[["match_id"] + MTV_FEATURES],
        on="match_id", how="left",
    )

    # Strict-MTV filter: drop matches without BOTH talent_deltas + shock
    initial_len = len(df)
    df = df.dropna(subset=["home_talent_delta", "away_talent_delta",
                            "shock_magnitude"]).reset_index(drop=True)
    n_mtv = len(df)
    if n_mtv < 50:
        print(f"  SKIP {LEAGUE_NAMES.get(league_id, league_id)}: "
              f"only {n_mtv} matches with MTV (need >=50)")
        return None

    # Sort by date, temporal split
    df = df.sort_values("date").reset_index(drop=True)
    split_idx = int(len(df) * (1 - TEST_FRACTION))
    df_train = df.iloc[:split_idx]
    df_test = df.iloc[split_idx:]

    if len(df_test) < 10:
        print(f"  SKIP {LEAGUE_NAMES.get(league_id, league_id)}: "
              f"only {len(df_test)} test matches")
        return None

    y_train = df_train["result"].values.astype(int)
    y_test = df_test["result"].values.astype(int)

    # Check odds availability per match in test set
    odds_valid = (df_test["odds_home"].notna() & (df_test["odds_home"] > 1.0) &
                  df_test["odds_draw"].notna() & (df_test["odds_draw"] > 1.0) &
                  df_test["odds_away"].notna() & (df_test["odds_away"] > 1.0))
    n_odds = int(odds_valid.sum())
    odds_pct = n_odds / len(df_test) * 100

    # Collect per-match OOT predictions
    records = []
    for match_idx in range(len(df_test)):
        rec = {
            "match_id": int(df_test.iloc[match_idx]["match_id"]),
            "league_id": league_id,
            "date": df_test.iloc[match_idx]["date"],
            "result": int(y_test[match_idx]),
            "shock_magnitude": float(df_test.iloc[match_idx]["shock_magnitude"]),
        }
        # Market probabilities per-match (only if this match has valid odds)
        if odds_valid.iloc[match_idx]:
            inv_h = 1.0 / df_test.iloc[match_idx]["odds_home"]
            inv_d = 1.0 / df_test.iloc[match_idx]["odds_draw"]
            inv_a = 1.0 / df_test.iloc[match_idx]["odds_away"]
            total = inv_h + inv_d + inv_a
            rec["market_H"] = float(inv_h / total)
            rec["market_D"] = float(inv_d / total)
            rec["market_A"] = float(inv_a / total)
        records.append(rec)

    # For Pairs B/C: need odds in TRAIN set too (for features).
    # Use train matches that have valid odds; require >= 50% coverage for training.
    train_odds_valid = (df_train["odds_home"].notna() & (df_train["odds_home"] > 1.0) &
                        df_train["odds_draw"].notna() & (df_train["odds_draw"] > 1.0) &
                        df_train["odds_away"].notna() & (df_train["odds_away"] > 1.0))
    n_train_odds = int(train_odds_valid.sum())
    has_enough_odds = n_train_odds >= 100 and n_odds >= 20

    # Train each test pair
    for pair_id, pair in TEST_PAIRS.items():
        if pair.get("requires_odds") and not has_enough_odds:
            continue

        # Check all features exist
        control_feats = pair["control"]
        mtv_feats = pair["mtv"]
        missing_ctrl = [f for f in control_feats if f not in df.columns]
        missing_mtv = [f for f in mtv_feats if f not in df.columns]
        if missing_ctrl or missing_mtv:
            continue

        # For odds-requiring pairs: filter to matches with valid odds
        if pair.get("requires_odds"):
            df_tr_pair = df_train[train_odds_valid].reset_index(drop=True)
            df_te_pair = df_test  # predict on ALL test matches, NaN odds handled in features
            # For test set: fill missing odds with median (so XGBoost can still predict)
            for oc in ODDS_FEATURES:
                median_val = df_tr_pair[oc].median()
                df_te_pair = df_te_pair.copy()
                df_te_pair[oc] = df_te_pair[oc].fillna(median_val)
            y_tr_pair = df_tr_pair["result"].values.astype(int)
        else:
            df_tr_pair = df_train
            df_te_pair = df_test
            y_tr_pair = y_train

        # Control model
        X_tr_ctrl = df_tr_pair[control_feats].values.astype(np.float32)
        X_te_ctrl = df_te_pair[control_feats].values.astype(np.float32)
        ctrl_prob = ensemble_predict(X_tr_ctrl, y_tr_pair, X_te_ctrl)

        # MTV model
        X_tr_mtv = df_tr_pair[mtv_feats].values.astype(np.float32)
        X_te_mtv = df_te_pair[mtv_feats].values.astype(np.float32)
        mtv_prob = ensemble_predict(X_tr_mtv, y_tr_pair, X_te_mtv)

        for i, rec in enumerate(records):
            rec[f"{pair_id}_ctrl_H"] = float(ctrl_prob[i, 0])
            rec[f"{pair_id}_ctrl_D"] = float(ctrl_prob[i, 1])
            rec[f"{pair_id}_ctrl_A"] = float(ctrl_prob[i, 2])
            rec[f"{pair_id}_mtv_H"] = float(mtv_prob[i, 0])
            rec[f"{pair_id}_mtv_D"] = float(mtv_prob[i, 1])
            rec[f"{pair_id}_mtv_A"] = float(mtv_prob[i, 2])

    n_drop = initial_len - n_mtv
    odds_label = f"{n_odds}/{len(df_test)} ({odds_pct:.0f}%)" if n_odds > 0 else "NONE"
    print(f"  {LEAGUE_NAMES.get(league_id, league_id):25s} "
          f"total={initial_len:4d} mtv={n_mtv:4d} (-{n_drop:3d}) "
          f"train={len(df_train):4d} test={len(df_test):3d} "
          f"split={df_test['date'].min().strftime('%Y-%m-%d')} "
          f"odds_test={odds_label}")

    return pd.DataFrame(records)


# ─── Analysis functions ──────────────────────────────────────

def analyze_pooled(tensor, pair_id, pair_name, has_market=True):
    """Compute pooled Brier analysis for one test pair."""
    ctrl_cols = [f"{pair_id}_ctrl_H", f"{pair_id}_ctrl_D", f"{pair_id}_ctrl_A"]
    mtv_cols = [f"{pair_id}_mtv_H", f"{pair_id}_mtv_D", f"{pair_id}_mtv_A"]

    # Only include matches that have this pair's predictions
    mask = tensor[ctrl_cols[0]].notna()
    t = tensor[mask].copy()
    if len(t) < 30:
        return None

    y = t["result"].values.astype(int)
    ctrl_probs = t[ctrl_cols].values
    mtv_probs = t[mtv_cols].values

    brier_ctrl_vec = per_match_brier(y, ctrl_probs)
    brier_mtv_vec = per_match_brier(y, mtv_probs)

    brier_ctrl = float(np.mean(brier_ctrl_vec))
    brier_mtv = float(np.mean(brier_mtv_vec))
    ci_ctrl = bootstrap_brier_ci(brier_ctrl_vec)
    ci_mtv = bootstrap_brier_ci(brier_mtv_vec)

    delta_mtv_ctrl = bootstrap_paired_delta_ci(brier_mtv_vec, brier_ctrl_vec)

    result = {
        "pair": pair_name,
        "n_test": len(t),
        "n_leagues": t["league_id"].nunique(),
        "brier_control": round(brier_ctrl, 5),
        "brier_control_ci95": [round(ci_ctrl[0], 5), round(ci_ctrl[1], 5)],
        "brier_mtv": round(brier_mtv, 5),
        "brier_mtv_ci95": [round(ci_mtv[0], 5), round(ci_mtv[1], 5)],
        "delta_mtv_minus_ctrl": delta_mtv_ctrl,
    }

    # Market comparison: subset with valid odds
    if has_market and "market_H" in t.columns:
        mkt_mask = t["market_H"].notna()
        t_mkt = t[mkt_mask]
        if len(t_mkt) >= 30:
            y_mkt = t_mkt["result"].values.astype(int)
            idx = t_mkt.index
            mkt_probs = t_mkt[["market_H", "market_D", "market_A"]].values
            brier_mkt_vec = per_match_brier(y_mkt, mkt_probs)
            # Recompute ctrl/mtv on same subset for fair comparison
            ctrl_mkt_vec = per_match_brier(y_mkt, t_mkt[ctrl_cols].values)
            mtv_mkt_vec = per_match_brier(y_mkt, t_mkt[mtv_cols].values)
            delta_mtv_mkt = bootstrap_paired_delta_ci(mtv_mkt_vec, brier_mkt_vec)
            delta_ctrl_mkt = bootstrap_paired_delta_ci(ctrl_mkt_vec, brier_mkt_vec)
            result.update({
                "n_with_market": len(t_mkt),
                "brier_market": round(float(np.mean(brier_mkt_vec)), 5),
                "brier_market_ci95": list(map(lambda x: round(x, 5),
                                              bootstrap_brier_ci(brier_mkt_vec))),
                "brier_control_mkt_subset": round(float(np.mean(ctrl_mkt_vec)), 5),
                "brier_mtv_mkt_subset": round(float(np.mean(mtv_mkt_vec)), 5),
                "delta_mtv_minus_market": delta_mtv_mkt,
                "delta_ctrl_minus_market": delta_ctrl_mkt,
            })

    return result


def analyze_p90_shock(tensor, pair_id, pair_name, percentile=90):
    """M2: Brier conditioned on shock_magnitude >= global P(percentile)."""
    ctrl_cols = [f"{pair_id}_ctrl_H", f"{pair_id}_ctrl_D", f"{pair_id}_ctrl_A"]
    mtv_cols = [f"{pair_id}_mtv_H", f"{pair_id}_mtv_D", f"{pair_id}_mtv_A"]

    mask = tensor[ctrl_cols[0]].notna()
    t = tensor[mask].copy()
    if len(t) < 30:
        return None

    threshold = np.percentile(t["shock_magnitude"].values, percentile)
    shock_mask = t["shock_magnitude"] >= threshold
    t_shock = t[shock_mask]

    if len(t_shock) < 10:
        return None

    y = t_shock["result"].values.astype(int)
    ctrl_probs = t_shock[ctrl_cols].values
    mtv_probs = t_shock[mtv_cols].values

    brier_ctrl_vec = per_match_brier(y, ctrl_probs)
    brier_mtv_vec = per_match_brier(y, mtv_probs)

    delta = bootstrap_paired_delta_ci(brier_mtv_vec, brier_ctrl_vec)

    result = {
        "pair": pair_name,
        "percentile": percentile,
        "threshold": round(float(threshold), 4),
        "n_shock": len(t_shock),
        "n_total": len(t),
        "brier_control_shock": round(float(np.mean(brier_ctrl_vec)), 5),
        "brier_mtv_shock": round(float(np.mean(brier_mtv_vec)), 5),
        "delta_mtv_minus_ctrl": delta,
    }

    # Market comparison in shock zone (subset with valid odds)
    if "market_H" in t_shock.columns:
        mkt_mask = t_shock["market_H"].notna()
        t_shock_mkt = t_shock[mkt_mask]
        if len(t_shock_mkt) >= 10:
            y_mkt = t_shock_mkt["result"].values.astype(int)
            mkt_probs = t_shock_mkt[["market_H", "market_D", "market_A"]].values
            brier_mkt_vec = per_match_brier(y_mkt, mkt_probs)
            mtv_mkt_vec = per_match_brier(y_mkt, t_shock_mkt[mtv_cols].values)
            ctrl_mkt_vec = per_match_brier(y_mkt, t_shock_mkt[ctrl_cols].values)
            delta_mtv_mkt = bootstrap_paired_delta_ci(mtv_mkt_vec, brier_mkt_vec)
            delta_ctrl_mkt = bootstrap_paired_delta_ci(ctrl_mkt_vec, brier_mkt_vec)
            result.update({
                "n_shock_with_market": len(t_shock_mkt),
                "brier_market_shock": round(float(np.mean(brier_mkt_vec)), 5),
                "delta_mtv_minus_market": delta_mtv_mkt,
                "delta_ctrl_minus_market": delta_ctrl_mkt,
            })

    return result


def analyze_injury_eras(tensor, pair_id, pair_name):
    """M4 (ABE P1-1): Stratify by injury era."""
    ctrl_cols = [f"{pair_id}_ctrl_H", f"{pair_id}_ctrl_D", f"{pair_id}_ctrl_A"]
    mtv_cols = [f"{pair_id}_mtv_H", f"{pair_id}_mtv_D", f"{pair_id}_mtv_A"]

    mask = tensor[ctrl_cols[0]].notna()
    t = tensor[mask].copy()
    if len(t) < 30:
        return None

    cutoff = pd.Timestamp(INJURY_ERA_CUTOFF)
    era1 = t[t["date"] < cutoff]
    era2 = t[t["date"] >= cutoff]

    results = {}
    for era_name, era_df in [("era1_injury_blind", era1), ("era2_injury_aware", era2)]:
        if len(era_df) < 10:
            results[era_name] = {"n": len(era_df), "status": "INSUFFICIENT_DATA"}
            continue

        y = era_df["result"].values.astype(int)
        ctrl_probs = era_df[ctrl_cols].values
        mtv_probs = era_df[mtv_cols].values

        brier_ctrl_vec = per_match_brier(y, ctrl_probs)
        brier_mtv_vec = per_match_brier(y, mtv_probs)
        delta = bootstrap_paired_delta_ci(brier_mtv_vec, brier_ctrl_vec)

        era_result = {
            "n": len(era_df),
            "date_range": [str(era_df["date"].min().date()),
                           str(era_df["date"].max().date())],
            "brier_control": round(float(np.mean(brier_ctrl_vec)), 5),
            "brier_mtv": round(float(np.mean(brier_mtv_vec)), 5),
            "delta_mtv_minus_ctrl": delta,
            "shock_magnitude_mean": round(float(era_df["shock_magnitude"].mean()), 4),
            "shock_magnitude_p90": round(float(np.percentile(era_df["shock_magnitude"], 90)), 4),
        }

        if "market_H" in era_df.columns:
            mkt_mask = era_df["market_H"].notna()
            era_mkt = era_df[mkt_mask]
            if len(era_mkt) >= 10:
                y_mkt = era_mkt["result"].values.astype(int)
                mkt_probs = era_mkt[["market_H", "market_D", "market_A"]].values
                brier_mkt_vec = per_match_brier(y_mkt, mkt_probs)
                era_result["n_with_market"] = len(era_mkt)
                era_result["brier_market"] = round(float(np.mean(brier_mkt_vec)), 5)

        results[era_name] = era_result

    return {"pair": pair_name, "cutoff": INJURY_ERA_CUTOFF, **results}


# ─── Tier 3 Purified Sweep ────────────────────────────────────

def run_tier3_sweep():
    """Mandato Quant Final: Pair B shock sweep P80-P95 on Tier 3 winners only."""
    tensor_path = Path("scripts/output/lab/mega_pool_tensor.parquet")
    if not tensor_path.exists():
        print("ERROR: mega_pool_tensor.parquet not found. Run full analysis first.")
        sys.exit(1)

    tensor = pd.read_parquet(tensor_path)
    tensor["date"] = pd.to_datetime(tensor["date"])

    # Filter to Tier 3 winners only
    t3 = tensor[tensor["league_id"].isin(TIER3_WINNERS)].copy()

    print("=" * 70)
    print("  MANDATO QUANT FINAL — Tier 3 Purified Shock Sweep")
    print("=" * 70)
    print(f"\n  Tier 3 winners: {sorted(TIER3_WINNERS)}")
    print(f"  Leagues found in tensor: {sorted(t3['league_id'].unique().tolist())}")
    print(f"  N matches (Tier 3): {len(t3)}")

    pair_id = "B_elo_odds"
    ctrl_cols = [f"{pair_id}_ctrl_H", f"{pair_id}_ctrl_D", f"{pair_id}_ctrl_A"]
    mtv_cols = [f"{pair_id}_mtv_H", f"{pair_id}_mtv_D", f"{pair_id}_mtv_A"]

    # Filter to matches with Pair B predictions
    mask = t3[ctrl_cols[0]].notna()
    t3b = t3[mask].copy()
    n_leagues = t3b["league_id"].nunique()

    print(f"  N with Pair B predictions: {len(t3b)} ({n_leagues} leagues)")
    print(f"  Date range: {t3b['date'].min().date()} → {t3b['date'].max().date()}")
    print(f"  shock_magnitude: mean={t3b['shock_magnitude'].mean():.4f} "
          f"P90={np.percentile(t3b['shock_magnitude'], 90):.4f}")

    # Per-league summary
    print(f"\n  {'League':22s} {'N':>4} {'Brier_Ctrl':>11} {'Brier_MTV':>10} "
          f"{'Δ(MTV-Ctrl)':>12} {'Direction':>12}")
    print(f"  {'─' * 72}")

    for lid in sorted(t3b["league_id"].unique()):
        lt = t3b[t3b["league_id"] == lid]
        y = lt["result"].values.astype(int)
        bc = float(np.mean(per_match_brier(y, lt[ctrl_cols].values)))
        bm = float(np.mean(per_match_brier(y, lt[mtv_cols].values)))
        delta = bm - bc
        direction = "MTV HELPS" if delta < 0 else "MTV HURTS"
        name = LEAGUE_NAMES.get(lid, str(lid))
        print(f"  {name:22s} {len(lt):>4} {bc:>11.5f} {bm:>10.5f} "
              f"{delta:>+12.5f} {direction:>12}")

    # Pooled Pair B on Tier 3
    y_all = t3b["result"].values.astype(int)
    ctrl_probs = t3b[ctrl_cols].values
    mtv_probs = t3b[mtv_cols].values
    brier_ctrl_vec = per_match_brier(y_all, ctrl_probs)
    brier_mtv_vec = per_match_brier(y_all, mtv_probs)

    pooled_delta = bootstrap_paired_delta_ci(brier_mtv_vec, brier_ctrl_vec)
    sig = "***" if pooled_delta["significant"] else "n.s."
    direction = "MTV HELPS" if pooled_delta["delta_mean"] < 0 else "MTV HURTS"

    print(f"\n  POOLED Tier 3 (Pair B, N={len(t3b)}, {n_leagues} leagues):")
    print(f"    Brier Control = {float(np.mean(brier_ctrl_vec)):.5f}")
    print(f"    Brier MTV     = {float(np.mean(brier_mtv_vec)):.5f}")
    print(f"    Δ(MTV-Ctrl)   = {pooled_delta['delta_mean']:+.5f} "
          f"[{pooled_delta['ci95_lo']:+.5f}, {pooled_delta['ci95_hi']:+.5f}] {sig} → {direction}")

    # Market comparison on Tier 3
    mkt_mask = t3b["market_H"].notna()
    t3b_mkt = t3b[mkt_mask]
    if len(t3b_mkt) >= 30:
        y_mkt = t3b_mkt["result"].values.astype(int)
        mkt_probs = t3b_mkt[["market_H", "market_D", "market_A"]].values
        brier_mkt_vec = per_match_brier(y_mkt, mkt_probs)
        mtv_mkt_vec = per_match_brier(y_mkt, t3b_mkt[mtv_cols].values)
        ctrl_mkt_vec = per_match_brier(y_mkt, t3b_mkt[ctrl_cols].values)
        delta_mtv_mkt = bootstrap_paired_delta_ci(mtv_mkt_vec, brier_mkt_vec)
        delta_ctrl_mkt = bootstrap_paired_delta_ci(ctrl_mkt_vec, brier_mkt_vec)
        print(f"\n    --- Market subset (N={len(t3b_mkt)}) ---")
        print(f"    Brier Market   = {float(np.mean(brier_mkt_vec)):.5f}")
        print(f"    Brier Ctrl*    = {float(np.mean(ctrl_mkt_vec)):.5f}")
        print(f"    Brier MTV*     = {float(np.mean(mtv_mkt_vec)):.5f}")
        sig_mc = "***" if delta_mtv_mkt["significant"] else "n.s."
        sig_cc = "***" if delta_ctrl_mkt["significant"] else "n.s."
        print(f"    Δ(MTV-Mkt)    = {delta_mtv_mkt['delta_mean']:+.5f} "
              f"[{delta_mtv_mkt['ci95_lo']:+.5f}, {delta_mtv_mkt['ci95_hi']:+.5f}] {sig_mc}")
        print(f"    Δ(Ctrl-Mkt)   = {delta_ctrl_mkt['delta_mean']:+.5f} "
              f"[{delta_ctrl_mkt['ci95_lo']:+.5f}, {delta_ctrl_mkt['ci95_hi']:+.5f}] {sig_cc}")

    # Shock sweep P80, P85, P90, P95
    print(f"\n{'─' * 70}")
    print("  SHOCK SWEEP — Pair B on Tier 3 Purified")
    print(f"{'─' * 70}")
    print(f"\n  {'P%':>4} {'Thresh':>7} {'N':>5} {'Brier_Ctrl':>11} {'Brier_MTV':>10} "
          f"{'Δ(MTV-Ctrl)':>12} {'CI95_lo':>9} {'CI95_hi':>9} {'Sig':>5}")
    print(f"  {'─' * 74}")

    sweep_results = []
    for pct in [80, 85, 90, 95]:
        threshold = np.percentile(t3b["shock_magnitude"].values, pct)
        shock_mask = t3b["shock_magnitude"] >= threshold
        t_shock = t3b[shock_mask]
        if len(t_shock) < 10:
            print(f"  P{pct:<3} {threshold:>7.4f} {len(t_shock):>5} — INSUFFICIENT")
            continue

        y_s = t_shock["result"].values.astype(int)
        ctrl_s = per_match_brier(y_s, t_shock[ctrl_cols].values)
        mtv_s = per_match_brier(y_s, t_shock[mtv_cols].values)
        delta_s = bootstrap_paired_delta_ci(mtv_s, ctrl_s)
        sig_s = "***" if delta_s["significant"] else "n.s."

        print(f"  P{pct:<3} {threshold:>7.4f} {len(t_shock):>5} "
              f"{float(np.mean(ctrl_s)):>11.5f} {float(np.mean(mtv_s)):>10.5f} "
              f"{delta_s['delta_mean']:>+12.5f} {delta_s['ci95_lo']:>+9.5f} "
              f"{delta_s['ci95_hi']:>+9.5f} {sig_s:>5}")

        # Market comparison in shock zone
        mkt_shock_mask = t_shock["market_H"].notna()
        t_shock_mkt = t_shock[mkt_shock_mask]
        mkt_result = None
        if len(t_shock_mkt) >= 10:
            y_sm = t_shock_mkt["result"].values.astype(int)
            mkt_sm = per_match_brier(y_sm, t_shock_mkt[["market_H", "market_D", "market_A"]].values)
            mtv_sm = per_match_brier(y_sm, t_shock_mkt[mtv_cols].values)
            delta_sm = bootstrap_paired_delta_ci(mtv_sm, mkt_sm)
            mkt_result = {
                "n_with_market": len(t_shock_mkt),
                "brier_market": round(float(np.mean(mkt_sm)), 5),
                "brier_mtv_mkt_subset": round(float(np.mean(mtv_sm)), 5),
                "delta_mtv_minus_market": delta_sm,
            }

        sweep_results.append({
            "percentile": pct,
            "threshold": round(float(threshold), 4),
            "n_shock": len(t_shock),
            "brier_control": round(float(np.mean(ctrl_s)), 5),
            "brier_mtv": round(float(np.mean(mtv_s)), 5),
            "delta_mtv_minus_ctrl": delta_s,
            "market": mkt_result,
        })

    # Market sweep table
    if any(r.get("market") for r in sweep_results):
        print(f"\n  SHOCK SWEEP vs MARKET — Pair B on Tier 3")
        print(f"  {'P%':>4} {'N_mkt':>6} {'Brier_Mkt':>10} {'Brier_MTV':>10} "
              f"{'Δ(MTV-Mkt)':>12} {'CI95_lo':>9} {'CI95_hi':>9} {'Sig':>5}")
        print(f"  {'─' * 68}")
        for r in sweep_results:
            m = r.get("market")
            if not m:
                continue
            d = m["delta_mtv_minus_market"]
            sig_m = "***" if d["significant"] else "n.s."
            print(f"  P{r['percentile']:<3} {m['n_with_market']:>6} "
                  f"{m['brier_market']:>10.5f} {m['brier_mtv_mkt_subset']:>10.5f} "
                  f"{d['delta_mean']:>+12.5f} {d['ci95_lo']:>+9.5f} "
                  f"{d['ci95_hi']:>+9.5f} {sig_m:>5}")

    # Save Tier 3 results
    output = {
        "meta": {
            "timestamp": datetime.now().isoformat(),
            "analysis": "tier3_purified_shock_sweep",
            "tier3_leagues": sorted(list(TIER3_WINNERS)),
            "tier3_league_names": {lid: LEAGUE_NAMES.get(lid, str(lid))
                                   for lid in sorted(TIER3_WINNERS)},
            "n_total": len(t3b),
            "n_leagues": n_leagues,
        },
        "pooled": {
            "brier_control": round(float(np.mean(brier_ctrl_vec)), 5),
            "brier_mtv": round(float(np.mean(brier_mtv_vec)), 5),
            "delta_mtv_minus_ctrl": pooled_delta,
        },
        "shock_sweep": sweep_results,
    }

    out_path = Path("scripts/output/lab/tier3_sweep_results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Tier 3 sweep results saved to: {out_path}")

    print(f"\n{'=' * 70}")
    print("  MANDATO QUANT FINAL COMPLETE")
    print(f"{'=' * 70}")


# ─── Main ────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="GDT Mega-Pool Analysis")
    parser.add_argument("--include-conference", action="store_true",
                        help="Include Conference League (848)")
    parser.add_argument("--lab-dir", default="scripts/output/lab",
                        help="Directory with lab_data CSVs")
    parser.add_argument("--output", default="scripts/output/lab/mega_pool_results.json",
                        help="Output JSON path")
    parser.add_argument("--tier3-sweep", action="store_true",
                        help="Run Tier 3 purified Pair B shock sweep from existing tensor")
    args = parser.parse_args()

    if args.tier3_sweep:
        run_tier3_sweep()
        return

    print("=" * 70)
    print("  GDT MEGA-POOL ANALYSIS — Súper-Tensor de Ligas Periféricas")
    print("  Mandatos: M1 (Pooled ΔBrier), M2 (P90 Shock), M4 (Injury Eras)")
    print("=" * 70)

    # Load MTV parquet
    mtv_path = Path("data/historical_mtv_features.parquet")
    if not mtv_path.exists():
        print("ERROR: data/historical_mtv_features.parquet not found")
        sys.exit(1)

    mtv_df = pd.read_parquet(mtv_path, columns=[
        "match_id", "home_talent_delta", "away_talent_delta",
        "talent_delta_diff", "shock_magnitude",
    ])
    mtv_df["match_id"] = mtv_df["match_id"].astype("Int64")
    print(f"\n  MTV parquet loaded: {len(mtv_df)} rows")

    # Identify eligible leagues
    exclude = BIG5 | INTL
    if not args.include_conference:
        exclude |= UEFA_CLUB

    lab_dir = Path(args.lab_dir)
    available = []
    for csv_file in sorted(lab_dir.glob("lab_data_*.csv")):
        lid = int(csv_file.stem.split("_")[-1])
        if lid not in exclude:
            available.append(lid)

    print(f"  Eligible peripheral leagues: {len(available)}")
    print(f"  IDs: {available}\n")

    # Process each league
    print(f"  {'League':25s} {'total':>5} {'mtv':>5} {'drop':>5} "
          f"{'train':>5} {'test':>4} {'split':>11} {'odds':>4}")
    print("  " + "-" * 68)

    all_frames = []
    for lid in available:
        result = process_league(lid, lab_dir, mtv_df)
        if result is not None:
            all_frames.append(result)

    if not all_frames:
        print("\nERROR: No leagues processed successfully")
        sys.exit(1)

    # Build Super-Tensor
    tensor = pd.concat(all_frames, ignore_index=True)
    tensor["date"] = pd.to_datetime(tensor["date"])

    print(f"\n{'=' * 70}")
    print(f"  SUPER-TENSOR: {len(tensor)} OOT matches across "
          f"{tensor['league_id'].nunique()} leagues")
    print(f"  Date range: {tensor['date'].min().date()} → {tensor['date'].max().date()}")
    print(f"  shock_magnitude: mean={tensor['shock_magnitude'].mean():.4f} "
          f"P90={np.percentile(tensor['shock_magnitude'], 90):.4f}")
    print(f"{'=' * 70}")

    # ─── M1: Pooled ΔBrier Analysis ─────────────────────────

    print(f"\n{'─' * 70}")
    print("  M1: POOLED ΔBrier (Control vs MTV vs Market)")
    print(f"{'─' * 70}")

    pooled_results = []
    for pair_id, pair in TEST_PAIRS.items():
        r = analyze_pooled(tensor, pair_id, pair["name"])
        if r:
            pooled_results.append(r)
            # Print summary
            delta = r["delta_mtv_minus_ctrl"]
            sig = "***" if delta["significant"] else "n.s."
            direction = "MTV HELPS" if delta["delta_mean"] < 0 else "MTV HURTS"
            print(f"\n  Pair {pair['name']}:")
            print(f"    N_test = {r['n_test']} ({r['n_leagues']} leagues)")
            print(f"    Brier Control = {r['brier_control']:.5f} "
                  f"[{r['brier_control_ci95'][0]:.5f}, {r['brier_control_ci95'][1]:.5f}]")
            print(f"    Brier MTV     = {r['brier_mtv']:.5f} "
                  f"[{r['brier_mtv_ci95'][0]:.5f}, {r['brier_mtv_ci95'][1]:.5f}]")
            print(f"    Δ(MTV-Ctrl)   = {delta['delta_mean']:+.5f} "
                  f"[{delta['ci95_lo']:+.5f}, {delta['ci95_hi']:+.5f}] {sig} → {direction}")
            if "brier_market" in r:
                d_mkt = r["delta_mtv_minus_market"]
                d_ctrl_mkt = r["delta_ctrl_minus_market"]
                sig_mkt = "***" if d_mkt["significant"] else "n.s."
                n_mkt = r.get("n_with_market", "?")
                print(f"    --- Market subset (N={n_mkt}) ---")
                print(f"    Brier Market   = {r['brier_market']:.5f}")
                print(f"    Brier Ctrl*    = {r.get('brier_control_mkt_subset', 0):.5f}")
                print(f"    Brier MTV*     = {r.get('brier_mtv_mkt_subset', 0):.5f}")
                print(f"    Δ(MTV-Mkt)    = {d_mkt['delta_mean']:+.5f} "
                      f"[{d_mkt['ci95_lo']:+.5f}, {d_mkt['ci95_hi']:+.5f}] {sig_mkt}")
                print(f"    Δ(Ctrl-Mkt)   = {d_ctrl_mkt['delta_mean']:+.5f} "
                      f"[{d_ctrl_mkt['ci95_lo']:+.5f}, {d_ctrl_mkt['ci95_hi']:+.5f}]")

    # ─── M2: P90 Shock Analysis ─────────────────────────────

    print(f"\n{'─' * 70}")
    print("  M2: BRIER CONDICIONADO A SHOCK_MAGNITUDE >= P90")
    print(f"{'─' * 70}")

    shock_results = []
    for percentile in [80, 85, 90, 95]:
        for pair_id, pair in TEST_PAIRS.items():
            r = analyze_p90_shock(tensor, pair_id, pair["name"], percentile=percentile)
            if r:
                shock_results.append(r)
                if percentile == 90:  # Only print P90 in detail
                    delta = r["delta_mtv_minus_ctrl"]
                    sig = "***" if delta["significant"] else "n.s."
                    direction = "MTV HELPS" if delta["delta_mean"] < 0 else "MTV HURTS"
                    print(f"\n  Pair {pair['name']} (P{percentile}, "
                          f"shock >= {r['threshold']:.4f}, N={r['n_shock']}):")
                    print(f"    Brier Control = {r['brier_control_shock']:.5f}")
                    print(f"    Brier MTV     = {r['brier_mtv_shock']:.5f}")
                    print(f"    Δ(MTV-Ctrl)   = {delta['delta_mean']:+.5f} "
                          f"[{delta['ci95_lo']:+.5f}, {delta['ci95_hi']:+.5f}] {sig} → {direction}")
                    if "brier_market_shock" in r:
                        d_mkt = r["delta_mtv_minus_market"]
                        sig_mkt = "***" if d_mkt["significant"] else "n.s."
                        print(f"    Brier Market   = {r['brier_market_shock']:.5f}")
                        print(f"    Δ(MTV-Mkt)    = {d_mkt['delta_mean']:+.5f} "
                              f"[{d_mkt['ci95_lo']:+.5f}, {d_mkt['ci95_hi']:+.5f}] {sig_mkt}")

    # P80-P95 summary tables
    for sweep_pair, sweep_label in [("Elo", "A: Elo (all leagues)"),
                                     ("Elo+Odds", "B: Elo+Odds (odds leagues)")]:
        pair_rows = [r for r in shock_results if r["pair"] == sweep_pair]
        if not pair_rows:
            continue
        print(f"\n  SHOCK THRESHOLD SWEEP — Pair {sweep_label}:")
        print(f"  {'P%':>4} {'Thresh':>7} {'N':>5} {'Brier_Ctrl':>11} {'Brier_MTV':>10} "
              f"{'Δ(MTV-Ctrl)':>12} {'Sig':>5}")
        print(f"  {'─' * 56}")
        for r in pair_rows:
            delta = r["delta_mtv_minus_ctrl"]
            sig = "***" if delta["significant"] else "n.s."
            print(f"  P{r['percentile']:<3} {r['threshold']:>7.4f} {r['n_shock']:>5} "
                  f"{r['brier_control_shock']:>11.5f} {r['brier_mtv_shock']:>10.5f} "
                  f"{delta['delta_mean']:>+12.5f} {sig:>5}")

    # ─── M4: Injury Era Stratification ──────────────────────

    print(f"\n{'─' * 70}")
    print(f"  M4: INJURY ERA STRATIFICATION (cutoff: {INJURY_ERA_CUTOFF})")
    print(f"{'─' * 70}")

    injury_results = []
    for pair_id, pair in TEST_PAIRS.items():
        r = analyze_injury_eras(tensor, pair_id, pair["name"])
        if r:
            injury_results.append(r)
            print(f"\n  Pair {pair['name']}:")
            for era_key in ["era1_injury_blind", "era2_injury_aware"]:
                era = r[era_key]
                era_label = "ERA 1 (Injury-Blind)" if "era1" in era_key else "ERA 2 (Injury-Aware)"
                if era.get("status") == "INSUFFICIENT_DATA":
                    print(f"    {era_label}: N={era['n']} — INSUFFICIENT_DATA")
                    continue
                delta = era["delta_mtv_minus_ctrl"]
                sig = "***" if delta["significant"] else "n.s."
                direction = "MTV HELPS" if delta["delta_mean"] < 0 else "MTV HURTS"
                print(f"    {era_label}: N={era['n']} "
                      f"({era['date_range'][0]} → {era['date_range'][1]})")
                print(f"      Brier Ctrl={era['brier_control']:.5f} "
                      f"MTV={era['brier_mtv']:.5f} "
                      f"Δ={delta['delta_mean']:+.5f} [{delta['ci95_lo']:+.5f}, "
                      f"{delta['ci95_hi']:+.5f}] {sig} → {direction}")
                print(f"      shock_mean={era['shock_magnitude_mean']:.4f} "
                      f"shock_P90={era['shock_magnitude_p90']:.4f}")
                if "brier_market" in era:
                    print(f"      Brier Market={era['brier_market']:.5f}")

    # ─── Per-league breakdown ────────────────────────────────

    league_breakdown = []
    for breakdown_pair_id, breakdown_label in [("A_elo", "A: Elo (all 18 leagues)"),
                                                ("B_elo_odds", "B: Elo+Odds (3 odds leagues)")]:
        print(f"\n{'─' * 70}")
        print(f"  PER-LEAGUE BREAKDOWN — Pair {breakdown_label}")
        print(f"{'─' * 70}")

        bc_cols = [f"{breakdown_pair_id}_ctrl_H", f"{breakdown_pair_id}_ctrl_D",
                   f"{breakdown_pair_id}_ctrl_A"]
        bm_cols = [f"{breakdown_pair_id}_mtv_H", f"{breakdown_pair_id}_mtv_D",
                   f"{breakdown_pair_id}_mtv_A"]

        mask = tensor[bc_cols[0]].notna()
        t = tensor[mask]

        print(f"\n  {'League':22s} {'N':>4} {'Brier_Ctrl':>11} {'Brier_MTV':>10} "
              f"{'Δ(MTV-Ctrl)':>12} {'Direction':>12}")
        print(f"  {'─' * 72}")

        for lid in sorted(t["league_id"].unique()):
            lt = t[t["league_id"] == lid]
            if len(lt) < 10:
                continue
            y = lt["result"].values.astype(int)
            ctrl_p = lt[bc_cols].values
            mtv_p = lt[bm_cols].values
            bc = float(np.mean(per_match_brier(y, ctrl_p)))
            bm = float(np.mean(per_match_brier(y, mtv_p)))
            delta = bm - bc
            direction = "MTV HELPS" if delta < 0 else "MTV HURTS"

            name = LEAGUE_NAMES.get(lid, str(lid))
            print(f"  {name:22s} {len(lt):>4} {bc:>11.5f} {bm:>10.5f} "
                  f"{delta:>+12.5f} {direction:>12}")

            if breakdown_pair_id == "A_elo":
                league_breakdown.append({
                    "league_id": int(lid),
                    "league_name": name,
                    "n_test": len(lt),
                    "brier_control": round(bc, 5),
                    "brier_mtv": round(bm, 5),
                    "delta_mtv_ctrl": round(delta, 5),
                })

    # ─── Save results ────────────────────────────────────────

    output = {
        "meta": {
            "timestamp": datetime.now().isoformat(),
            "script": "mega_pool_analysis.py",
            "n_leagues": tensor["league_id"].nunique(),
            "n_oot_total": len(tensor),
            "date_range": [str(tensor["date"].min().date()),
                           str(tensor["date"].max().date())],
            "shock_magnitude_global_stats": {
                "mean": round(float(tensor["shock_magnitude"].mean()), 4),
                "p50": round(float(np.percentile(tensor["shock_magnitude"], 50)), 4),
                "p80": round(float(np.percentile(tensor["shock_magnitude"], 80)), 4),
                "p85": round(float(np.percentile(tensor["shock_magnitude"], 85)), 4),
                "p90": round(float(np.percentile(tensor["shock_magnitude"], 90)), 4),
                "p95": round(float(np.percentile(tensor["shock_magnitude"], 95)), 4),
            },
            "prod_hyperparams": PROD_HYPERPARAMS,
            "n_seeds": N_SEEDS,
            "n_bootstrap": N_BOOTSTRAP,
            "test_fraction": TEST_FRACTION,
            "draw_weight": DRAW_WEIGHT,
            "injury_era_cutoff": INJURY_ERA_CUTOFF,
        },
        "m1_pooled": pooled_results,
        "m2_shock": shock_results,
        "m4_injury_eras": injury_results,
        "league_breakdown": league_breakdown,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Results saved to: {output_path}")

    # Save Super-Tensor as parquet for future analysis
    tensor_path = output_path.parent / "mega_pool_tensor.parquet"
    tensor.to_parquet(tensor_path, index=False)
    print(f"  Super-Tensor saved to: {tensor_path}")

    print(f"\n{'=' * 70}")
    print("  MEGA-POOL ANALYSIS COMPLETE")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
