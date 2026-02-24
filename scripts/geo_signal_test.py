#!/usr/bin/env python3
"""
Geo Signal Test — A/B/C comparison: 16f vs 18f-old vs 18f-fresh (100% coverage)

Measures the delta in Brier score and flat ROI from the geo backfill.
Does NOT modify any production code. Pure evaluation.

Usage:
    python scripts/geo_signal_test.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

# ═══════════════════════════════════════════════════════════════════════════
# Config (mirrors latam_sniper_backtest.py exactly)
# ═══════════════════════════════════════════════════════════════════════════

WHITELIST = {128, 242, 250, 265}
ALL_LATAM = {71, 128, 239, 242, 250, 262, 265, 268, 281, 299, 344}
# Evaluate 10 CONMEBOL leagues (Liga MX 262 excluded — CONCACAF)
CONMEBOL = {71, 128, 239, 242, 250, 265, 268, 281, 299, 344}
EVAL_LEAGUES = CONMEBOL  # 10 CONMEBOL leagues
LEAGUE_NAMES = {
    71: "Brazil", 128: "Argentina", 239: "Colombia", 242: "Ecuador",
    250: "Paraguay", 265: "Chile", 268: "Uruguay",
    281: "Peru", 299: "Venezuela", 344: "Bolivia",
}
MIN_DATE = "2023-01-01"
OOS_FRACTION = 0.20

FEATURES_16 = [
    "home_goals_scored_avg", "home_goals_conceded_avg",
    "home_shots_avg", "home_corners_avg",
    "home_rest_days", "home_matches_played",
    "away_goals_scored_avg", "away_goals_conceded_avg",
    "away_shots_avg", "away_corners_avg",
    "away_rest_days", "away_matches_played",
    "goal_diff_avg", "rest_diff",
    "elo_k10_diff", "elo_momentum_diff",
]

GEO_FEATURES = ["altitude_diff_m", "travel_distance_km"]
FEATURES_18 = FEATURES_16 + GEO_FEATURES

PARAMS_S1 = {
    "objective": "binary:logistic", "max_depth": 3, "learning_rate": 0.05,
    "n_estimators": 100, "min_child_weight": 7, "subsample": 0.72,
    "colsample_bytree": 0.71, "verbosity": 0, "random_state": 42,
}
PARAMS_S2 = {
    "objective": "binary:logistic", "max_depth": 3, "learning_rate": 0.05,
    "n_estimators": 100, "min_child_weight": 5, "subsample": 0.8,
    "colsample_bytree": 0.8, "verbosity": 0, "random_state": 42,
}

# Kelly / trading (same as production)
MIN_EV = 0.05
KELLY_FRACTION = 0.125

LAB_DIR = Path(__file__).parent / "output" / "lab"
GEO_CACHE_PATH = Path(__file__).parent.parent / "data" / "geo_cache_fresh.json"


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def haversine_km(lat1, lon1, lat2, lon2):
    """Haversine distance in km."""
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2))
         * math.sin(dlon / 2) ** 2)
    return R * 2 * math.asin(math.sqrt(min(a, 1.0)))


def load_geo_cache() -> dict:
    """Load fresh geo cache from JSON."""
    with open(GEO_CACHE_PATH) as f:
        return json.load(f)


def compute_fresh_geo(df: pd.DataFrame, geo_cache: dict) -> pd.DataFrame:
    """Recompute altitude_diff_m and travel_distance_km from fresh geo cache."""
    alt_diff = []
    travel_km = []

    for _, row in df.iterrows():
        h_id = str(int(row["home_team_id"])) if pd.notna(row.get("home_team_id")) else None
        a_id = str(int(row["away_team_id"])) if pd.notna(row.get("away_team_id")) else None

        h_geo = geo_cache.get(h_id) if h_id else None
        a_geo = geo_cache.get(a_id) if a_id else None

        if h_geo and a_geo:
            dist = haversine_km(h_geo["lat"], h_geo["lon"], a_geo["lat"], a_geo["lon"])
            travel_km.append(dist)

            h_alt = h_geo.get("altitude") or h_geo.get("alt")
            a_alt = a_geo.get("altitude") or a_geo.get("alt")
            if h_alt is not None and a_alt is not None:
                alt_diff.append(float(h_alt - a_alt))
            else:
                alt_diff.append(float("nan"))
        else:
            alt_diff.append(float("nan"))
            travel_km.append(float("nan"))

    df = df.copy()
    df["altitude_diff_m_fresh"] = alt_diff
    df["travel_distance_km_fresh"] = travel_km
    return df


def load_lab_data() -> pd.DataFrame:
    """Load and merge lab data CSVs for all LATAM leagues."""
    frames = []
    for lid in sorted(ALL_LATAM):
        path = LAB_DIR / f"lab_data_{lid}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        df = df[df["result"].notna()].copy()
        df["league_id"] = lid
        frames.append(df)

    df = pd.concat(frames, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"] >= MIN_DATE].copy()
    df = df[
        (df["odds_home"].notna()) & (df["odds_home"] > 1.0) &
        (df["odds_draw"].notna()) & (df["odds_draw"] > 1.0) &
        (df["odds_away"].notna()) & (df["odds_away"] > 1.0)
    ].copy()
    df = df.sort_values("date").reset_index(drop=True)
    return df


# ═══════════════════════════════════════════════════════════════════════════
# Model
# ═══════════════════════════════════════════════════════════════════════════

def train_twostage(X_train, y_train):
    """TwoStage: Draw vs Non-Draw → Home vs Away."""
    y_s1 = (y_train == 1).astype(int)
    m1 = xgb.XGBClassifier(**PARAMS_S1)
    m1.fit(X_train, y_s1, verbose=False)

    mask_nd = (y_train != 1)
    y_s2 = (y_train[mask_nd] == 0).astype(int)
    m2 = xgb.XGBClassifier(**PARAMS_S2)
    m2.fit(X_train[mask_nd], y_s2, verbose=False)
    return m1, m2


def predict_twostage(m1, m2, X):
    """Predict 1X2 probabilities."""
    p_draw = m1.predict_proba(X)[:, 1]
    p_home_nd = m2.predict_proba(X)[:, 1]
    p_nd = 1.0 - p_draw
    return np.column_stack([p_nd * p_home_nd, p_draw, p_nd * (1.0 - p_home_nd)])


def brier_score(y_true, y_prob):
    """Multi-class Brier score."""
    n = len(y_true)
    bs = 0.0
    for i in range(n):
        actual = np.zeros(3)
        actual[int(y_true[i])] = 1.0
        bs += np.sum((y_prob[i] - actual) ** 2)
    return bs / n


def flat_roi(y_true, y_prob, odds):
    """Flat ROI: bet on best EV per match, only if EV > MIN_EV."""
    total_bets = 0
    total_profit = 0.0

    for i in range(len(y_true)):
        evs = y_prob[i] * odds[i] - 1.0
        best_idx = np.argmax(evs)
        best_ev = evs[best_idx]

        if best_ev < MIN_EV:
            continue

        total_bets += 1
        if int(y_true[i]) == best_idx:
            total_profit += odds[i][best_idx] - 1.0
        else:
            total_profit -= 1.0

    if total_bets == 0:
        return 0.0, 0
    return (total_profit / total_bets) * 100, total_bets


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def run_variant(label: str, features: list[str], df_train: pd.DataFrame,
                df_oos: pd.DataFrame) -> dict:
    """Train and evaluate one variant. Returns metrics dict."""
    for col in features:
        if col not in df_train.columns:
            df_train[col] = 0.0
        if col not in df_oos.columns:
            df_oos[col] = 0.0

    X_tr = df_train[features].fillna(0.0).values
    y_tr = df_train["result"].values.astype(int)
    X_oos = df_oos[features].fillna(0.0).values
    y_oos = df_oos["result"].values.astype(int)

    m1, m2 = train_twostage(X_tr, y_tr)
    probs = predict_twostage(m1, m2, X_oos)

    bs = brier_score(y_oos, probs)

    odds_arr = df_oos[["odds_home", "odds_draw", "odds_away"]].values
    roi, n_bets = flat_roi(y_oos, probs, odds_arr)

    # Per-league Brier
    league_brier = {}
    for lid in sorted(EVAL_LEAGUES):
        mask = df_oos["league_id"].values == lid
        if mask.sum() > 0:
            league_brier[lid] = brier_score(y_oos[mask], probs[mask])

    # Geo coverage stats
    geo_nans = 0
    if "altitude_diff_m" in features:
        col_idx = features.index("altitude_diff_m")
        geo_nans = np.isnan(df_oos[features].values[:, col_idx]).sum()
    elif "altitude_diff_m_fresh" in features:
        col_idx = features.index("altitude_diff_m_fresh")
        geo_nans = np.isnan(df_oos[features].values[:, col_idx]).sum()

    return {
        "label": label,
        "features": len(features),
        "brier": bs,
        "roi": roi,
        "n_bets": n_bets,
        "league_brier": league_brier,
        "geo_nans": geo_nans,
        "n_oos": len(y_oos),
    }


def main():
    print("=" * 78)
    print("  GEO SIGNAL TEST — A/B/C Comparison")
    print("  ¿Mejora la señal con 100% cobertura geo?")
    print("=" * 78)

    # Load data
    print("\n[1/5] Loading lab data...")
    df_all = load_lab_data()
    print(f"  {len(df_all):,} matches loaded")

    print("\n[2/5] Loading fresh geo cache...")
    geo_cache = load_geo_cache()
    print(f"  {len(geo_cache)} teams in cache")

    # Recompute geo features from fresh cache
    print("\n[3/5] Computing fresh geo features...")
    df_all = compute_fresh_geo(df_all, geo_cache)

    # Coverage check
    for lid in sorted(EVAL_LEAGUES):
        mask = df_all["league_id"] == lid
        total = mask.sum()
        old_ok = df_all.loc[mask, "altitude_diff_m"].notna().sum()
        new_ok = df_all.loc[mask, "altitude_diff_m_fresh"].notna().sum()
        print(f"  {LEAGUE_NAMES[lid]:<12}: old={old_ok}/{total} ({100*old_ok/total:.0f}%)  →  fresh={new_ok}/{total} ({100*new_ok/total:.0f}%)")

    # Split
    print("\n[4/5] Chronological split...")
    n = len(df_all)
    split_idx = int(n * (1 - OOS_FRACTION))
    split_date = df_all.iloc[split_idx]["date"]

    df_train = df_all.iloc[:split_idx].copy()
    df_test_all = df_all.iloc[split_idx:].copy()
    df_oos = df_test_all[df_test_all["league_id"].isin(EVAL_LEAGUES)].copy()

    print(f"  Train: {len(df_train):,} | OOS CONMEBOL (10 ligas): {len(df_oos):,}")
    print(f"  Split date: {split_date.date()}")

    # Prepare variant C features (fresh geo instead of old)
    FEATURES_18_FRESH = FEATURES_16 + ["altitude_diff_m_fresh", "travel_distance_km_fresh"]

    # Run 3 variants
    print("\n[5/5] Running A/B/C variants...")

    print("\n  Training A (16f, no geo)...")
    res_a = run_variant("A: 16f (no geo)", FEATURES_16, df_train.copy(), df_oos.copy())

    print("  Training B (18f, old geo ~50% coverage)...")
    res_b = run_variant("B: 18f (old geo)", FEATURES_18, df_train.copy(), df_oos.copy())

    print("  Training C (18f, fresh geo 100% coverage)...")
    res_c = run_variant("C: 18f (fresh geo)", FEATURES_18_FRESH, df_train.copy(), df_oos.copy())

    # ═══════════════════════════════════════════════════════════════════════
    # Report
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 78)
    print("  RESULTADOS")
    print("=" * 78)

    print(f"\n  {'Variante':<25} {'Brier':>8} {'Δ vs A':>8} {'FlatROI':>9} {'Bets':>6} {'GeoNaN':>8}")
    print("  " + "─" * 70)
    for res in [res_a, res_b, res_c]:
        delta = res["brier"] - res_a["brier"]
        delta_str = f"{delta:+.5f}" if res != res_a else "   base"
        nan_str = f"{res['geo_nans']}/{res['n_oos']}" if res["geo_nans"] > 0 else "0"
        print(f"  {res['label']:<25} {res['brier']:.5f} {delta_str:>8} {res['roi']:>+8.2f}% {res['n_bets']:>5}  {nan_str:>8}")

    # Per-league breakdown
    print(f"\n  {'Liga':<12}", end="")
    for res in [res_a, res_b, res_c]:
        lbl = res["label"].split("(")[0].strip()
        print(f" {lbl:>14}", end="")
    print(f"  {'Δ(C-A)':>10}  {'Δ(C-B)':>10}")
    print("  " + "─" * 74)

    for lid in sorted(EVAL_LEAGUES):
        name = LEAGUE_NAMES[lid]
        vals = [res["league_brier"].get(lid, 0) for res in [res_a, res_b, res_c]]
        delta_ca = vals[2] - vals[0]
        delta_cb = vals[2] - vals[1]
        print(f"  {name:<12}", end="")
        for v in vals:
            print(f"  {v:.5f}     ", end="")
        print(f"  {delta_ca:+.5f}   {delta_cb:+.5f}")

    # Verdict
    delta_ca_total = res_c["brier"] - res_a["brier"]
    delta_cb_total = res_c["brier"] - res_b["brier"]

    print(f"\n  {'─' * 70}")
    print(f"  DELTA TOTAL Brier:")
    print(f"    C vs A (fresh geo vs no geo):   {delta_ca_total:+.5f}  {'MEJORA' if delta_ca_total < 0 else 'EMPEORA'}")
    print(f"    C vs B (fresh geo vs old geo):  {delta_cb_total:+.5f}  {'MEJORA' if delta_cb_total < 0 else 'EMPEORA'}")
    print(f"  DELTA ROI:")
    print(f"    C vs A: {res_c['roi'] - res_a['roi']:+.2f} pp")
    print(f"    C vs B: {res_c['roi'] - res_b['roi']:+.2f} pp")
    print("=" * 78)


if __name__ == "__main__":
    main()
