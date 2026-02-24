#!/usr/bin/env python3
"""
LATAM Sniper Backtest — V1.4.0 GEO-ROUTER Financial Simulator

Simulates paper trading using the bifurcated GEO-ROUTER architecture:
  - Tier GEO (18f): Bolivia, Paraguay, Peru, Venezuela, Chile
  - Tier FLAT (16f): Argentina, Brasil, Colombia, Ecuador, Uruguay
  - VORP lineup shock (beta=1.4206) from historical talent_delta_diff
  - Kelly Engine (Eighth-Kelly, 5% match cap, 5% min EV, high-odds penalty >5.0)
  - Daily compounding (morning bankroll, settle at EOD)

Each tier model is trained ONLY on its own tier's leagues (same as production).
OOS matches are routed to the correct model by league_id.

Whitelist (current): Chile(265), Argentina(128), Ecuador(242), Paraguay(250)
Candidates (GDT): Bolivia(344), Peru(281) — evaluate for whitelist addition.

Usage:
    source .env
    python scripts/latam_sniper_backtest.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import xgboost as xgb

# ═══════════════════════════════════════════════════════════════════════════
# Configuration — must match production exactly
# ═══════════════════════════════════════════════════════════════════════════

# GEO-ROUTER tiers (exact from latam_serving.py)
TIER_GEO_LEAGUES = {344, 250, 281, 299, 265}
TIER_FLAT_LEAGUES = {128, 71, 239, 242, 268}

LEAGUE_NAMES = {
    128: "Argentina", 71: "Brasil", 239: "Colombia", 242: "Ecuador",
    250: "Paraguay", 265: "Chile", 268: "Uruguay",
    281: "Peru", 299: "Venezuela", 344: "Bolivia",
}

# Betting whitelist: current + candidates for evaluation
WHITELIST_CURRENT = {128, 242, 250, 265}
WHITELIST_CANDIDATES = {344, 281}  # Bolivia, Peru — GDT evaluation
WHITELIST = WHITELIST_CURRENT | WHITELIST_CANDIDATES

# All 10 LATAM leagues (training pool, Mexico excluded per GDT)
ALL_LATAM = TIER_GEO_LEAGUES | TIER_FLAT_LEAGUES

MIN_DATE = "2023-01-01"
OOS_FRACTION = 0.20

# Feature sets (exact from training_config)
FEATURES_14 = [
    "home_goals_scored_avg", "home_goals_conceded_avg",
    "home_shots_avg", "home_corners_avg",
    "home_rest_days", "home_matches_played",
    "away_goals_scored_avg", "away_goals_conceded_avg",
    "away_shots_avg", "away_corners_avg",
    "away_rest_days", "away_matches_played",
    "goal_diff_avg", "rest_diff",
]
MOMENTUM_2 = ["elo_k10_diff", "elo_momentum_diff"]
GEO_2 = ["altitude_diff_m", "travel_distance_km"]

FEATURES_16 = FEATURES_14 + MOMENTUM_2  # Tier Flat
FEATURES_18 = FEATURES_16 + GEO_2       # Tier Geo

# Hyperparams (exact from training_config)
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

# VORP
VORP_BETA = 1.4206
EPS = 1e-7

# Kelly Engine defaults (production)
KELLY_FRACTION = 0.125      # Eighth-Kelly (GDT risk adjustment 2026-02-23)
BANKROLL_INIT = 1000.0
MAX_STAKE_PCT = 0.05        # 5% of bankroll per match
MIN_EV = 0.05              # 5% minimum EV (GDT 2026-02-23)
HIGH_ODDS_THRESHOLD = 5.0
HIGH_ODDS_FACTOR = 0.5

# Paths
LAB_DIR = Path(__file__).parent / "output" / "lab"
MTV_PATH = Path(__file__).parent.parent / "data" / "historical_mtv_features.parquet"
OUTPUT_DIR = Path(__file__).parent / "output"


# ═══════════════════════════════════════════════════════════════════════════
# Data Loading
# ═══════════════════════════════════════════════════════════════════════════

def load_lab_data(league_ids: set[int], min_date: str) -> pd.DataFrame:
    """Load and merge lab data CSVs for specified leagues."""
    frames = []
    for lid in sorted(league_ids):
        path = LAB_DIR / f"lab_data_{lid}.csv"
        if not path.exists():
            print(f"  WARNING: {path} not found, skipping league {lid}")
            continue
        df = pd.read_csv(path)
        df = df[df["result"].notna()].copy()
        df["league_id"] = lid
        frames.append(df)

    df = pd.concat(frames, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"] >= min_date].copy()

    # Require valid odds
    df = df[
        (df["odds_home"].notna()) & (df["odds_home"] > 1.0) &
        (df["odds_draw"].notna()) & (df["odds_draw"] > 1.0) &
        (df["odds_away"].notna()) & (df["odds_away"] > 1.0)
    ].copy()

    df = df.sort_values("date").reset_index(drop=True)
    return df


def load_mtv_data() -> pd.DataFrame:
    """Load historical talent_delta_diff from parquet."""
    if not MTV_PATH.exists():
        print(f"  WARNING: {MTV_PATH} not found. All VORP will be zero.")
        return pd.DataFrame(columns=["match_id", "talent_delta_diff"])

    df = pd.read_parquet(MTV_PATH, columns=["match_id", "talent_delta_diff"])
    return df


# ═══════════════════════════════════════════════════════════════════════════
# Model — TwoStage XGBoost (exact production architecture)
# ═══════════════════════════════════════════════════════════════════════════

def train_twostage(X_train: np.ndarray, y_train: np.ndarray):
    """Train TwoStage model. Returns (model_s1, model_s2)."""
    y_s1 = (y_train == 1).astype(int)
    m1 = xgb.XGBClassifier(**PARAMS_S1)
    m1.fit(X_train, y_s1, verbose=False)

    mask_not_draw = (y_train != 1)
    X_s2 = X_train[mask_not_draw]
    y_s2 = (y_train[mask_not_draw] == 0).astype(int)
    m2 = xgb.XGBClassifier(**PARAMS_S2)
    m2.fit(X_s2, y_s2, verbose=False)

    return m1, m2


def predict_twostage(m1, m2, X: np.ndarray) -> np.ndarray:
    """Predict 1X2 probabilities. Returns (N, 3) [home, draw, away]."""
    p_draw = m1.predict_proba(X)[:, 1]
    p_home_given_nd = m2.predict_proba(X)[:, 1]
    p_not_draw = 1.0 - p_draw
    p_home = p_not_draw * p_home_given_nd
    p_away = p_not_draw * (1.0 - p_home_given_nd)
    return np.column_stack([p_home, p_draw, p_away])


def get_tier(league_id: int) -> str:
    """Get GEO-ROUTER tier for a league."""
    if league_id in TIER_GEO_LEAGUES:
        return "geo"
    if league_id in TIER_FLAT_LEAGUES:
        return "flat"
    return "none"


# ═══════════════════════════════════════════════════════════════════════════
# VORP — Lineup Shock (exact production formula)
# ═══════════════════════════════════════════════════════════════════════════

def apply_vorp_shock(probs: np.ndarray, td_diff: float, beta: float = VORP_BETA) -> np.ndarray:
    """Apply log-shift-softmax VORP adjustment. probs = [home, draw, away]."""
    if beta == 0 or td_diff == 0 or np.isnan(td_diff):
        return probs.copy()

    p = np.clip(probs, EPS, 1.0 - EPS)
    z = np.log(p)
    z[0] += beta * td_diff
    z[2] -= beta * td_diff

    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


# ═══════════════════════════════════════════════════════════════════════════
# Kelly Engine (exact production functions)
# ═══════════════════════════════════════════════════════════════════════════

def kelly_stake(prob: float, odds: float) -> float:
    """Full Kelly f* = (bp - q) / b. No-shorting rule."""
    if not (0 < prob < 1) or odds <= 1.0:
        return 0.0
    b = odds - 1.0
    q = 1.0 - prob
    f_star = (b * prob - q) / b
    return max(0.0, f_star)


def find_value_bets(probs: np.ndarray, odds: np.ndarray) -> list[dict]:
    """Find value bets (EV > 0) and size with Kelly + risk overrides."""
    outcomes = ["home", "draw", "away"]
    bets = []

    for i, outcome in enumerate(outcomes):
        p = float(probs[i])
        o = float(odds[i])
        if o <= 1.0 or p <= 0:
            continue

        ev = p * o - 1.0
        if ev <= 0:
            continue

        raw = kelly_stake(p, o)
        frac = raw * KELLY_FRACTION

        flags = []
        if ev < MIN_EV:
            frac = 0.0
            flags.append("MIN_EV_REJECTED")
        else:
            if o > HIGH_ODDS_THRESHOLD:
                frac *= HIGH_ODDS_FACTOR
                flags.append("HIGH_ODDS_PENALTY")
            if frac > MAX_STAKE_PCT:
                frac = MAX_STAKE_PCT
                flags.append("MAX_MATCH_CAP_APPLIED")

        if frac > 0:
            bets.append({
                "outcome": outcome,
                "outcome_idx": i,
                "prob": p,
                "odds": o,
                "ev": ev,
                "kelly_raw": raw,
                "kelly_frac": frac,
                "flags": flags if flags else None,
            })

    return bets


# ═══════════════════════════════════════════════════════════════════════════
# Daily Compounding Simulation
# ═══════════════════════════════════════════════════════════════════════════

def simulate_daily(df_oos: pd.DataFrame) -> dict:
    """Simulate day-by-day paper trading with daily compounding."""
    bankroll = BANKROLL_INIT
    peak_bankroll = bankroll

    all_bets = []
    daily_log = []
    equity_curve = [(df_oos["date"].min() - pd.Timedelta(days=1), bankroll)]

    for date, day_matches in df_oos.groupby("date"):
        morning_bankroll = bankroll
        day_pnl = 0.0
        day_bets = 0
        day_staked = 0.0

        for _, match in day_matches.iterrows():
            probs = np.array([match["p_home"], match["p_draw"], match["p_away"]])
            odds = np.array([match["odds_home"], match["odds_draw"], match["odds_away"]])
            result = int(match["result"])

            bets = find_value_bets(probs, odds)
            if not bets:
                continue

            total_kelly = sum(b["kelly_frac"] for b in bets)

            for bet in bets:
                if total_kelly > MAX_STAKE_PCT:
                    stake_frac = bet["kelly_frac"] / total_kelly * MAX_STAKE_PCT
                else:
                    stake_frac = bet["kelly_frac"]

                stake_units = stake_frac * morning_bankroll
                won = (result == bet["outcome_idx"])

                if won:
                    pnl = stake_units * (bet["odds"] - 1.0)
                else:
                    pnl = -stake_units

                day_pnl += pnl
                day_staked += stake_units
                day_bets += 1

                all_bets.append({
                    "date": date,
                    "match_id": match.get("match_id"),
                    "league_id": match["league_id"],
                    "tier": match.get("tier", "?"),
                    "outcome": bet["outcome"],
                    "prob": bet["prob"],
                    "odds": bet["odds"],
                    "ev": bet["ev"],
                    "stake_frac": stake_frac,
                    "stake_units": round(stake_units, 2),
                    "won": won,
                    "pnl": round(pnl, 2),
                    "bankroll_pre": round(morning_bankroll, 2),
                    "flags": bet["flags"],
                    "vorp_applied": bool(match.get("vorp_applied", False)),
                })

        bankroll += day_pnl
        bankroll = max(bankroll, 0.0)

        if bankroll > peak_bankroll:
            peak_bankroll = bankroll

        equity_curve.append((date, round(bankroll, 2)))

        if day_bets > 0:
            daily_log.append({
                "date": date,
                "matches": len(day_matches),
                "bets": day_bets,
                "staked": round(day_staked, 2),
                "pnl": round(day_pnl, 2),
                "bankroll": round(bankroll, 2),
            })

    return {
        "all_bets": all_bets,
        "daily_log": daily_log,
        "equity_curve": equity_curve,
        "final_bankroll": bankroll,
        "peak_bankroll": peak_bankroll,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Reporting
# ═══════════════════════════════════════════════════════════════════════════

def print_report(df_oos, sim, split_date):
    """Print executive report."""
    bets = sim["all_bets"]
    if not bets:
        print("\n!! NO BETS GENERATED — check EV thresholds and odds quality !!")
        return

    df_bets = pd.DataFrame(bets)

    total_evaluated = len(df_oos)
    total_bets = len(df_bets)
    selectivity = total_bets / total_evaluated * 100 if total_evaluated > 0 else 0

    wins = df_bets["won"].sum()
    win_rate = wins / total_bets * 100
    avg_odds = df_bets["odds"].mean()

    flat_profits = df_bets.apply(
        lambda r: (r["odds"] - 1.0) if r["won"] else -1.0, axis=1
    )
    flat_roi = flat_profits.sum() / total_bets * 100

    equity = [e[1] for e in sim["equity_curve"]]
    peak = equity[0]
    max_dd = 0.0
    for val in equity:
        if val > peak:
            peak = val
        dd = (peak - val) / peak * 100 if peak > 0 else 0
        if dd > max_dd:
            max_dd = dd

    final_br = sim["final_bankroll"]

    first_date = sim["equity_curve"][0][0]
    last_date = sim["equity_curve"][-1][0]
    days = (last_date - first_date).days
    years = days / 365.25 if days > 0 else 1
    if final_br > 0 and BANKROLL_INIT > 0:
        cagr = ((final_br / BANKROLL_INIT) ** (1 / years) - 1) * 100
    else:
        cagr = -100.0

    vorp_bets = df_bets[df_bets["vorp_applied"]]
    vorp_count = len(vorp_bets)

    print("=" * 78)
    print("  LATAM SNIPER BACKTEST — V1.4.0 GEO-ROUTER")
    print("  Bifurcated: Tier GEO (18f) + Tier FLAT (16f)")
    print("=" * 78)
    print()
    print(f"  OOS Period:   {split_date.date()} -> {df_oos['date'].max().date()}")
    print(f"  GEO training: {sorted(TIER_GEO_LEAGUES)} (18f)")
    print(f"  FLAT training:{sorted(TIER_FLAT_LEAGUES)} (16f)")
    print(f"  Whitelist:    {', '.join(LEAGUE_NAMES[k] for k in sorted(WHITELIST))}")
    print(f"  VORP beta:    {VORP_BETA}")
    print(f"  Kelly:        {KELLY_FRACTION:.1%} fraction, {MAX_STAKE_PCT:.0%} match cap, {MIN_EV:.0%} min EV")
    print()
    print("-" * 78)
    print("  VOLUMEN & SELECTIVIDAD")
    print("-" * 78)
    print(f"  Partidos Evaluados:      {total_evaluated:,}")
    print(f"  Apuestas Ejecutadas:     {total_bets:,}")
    print(f"  Selectividad:            {selectivity:.1f}%")
    print(f"  VORP-adjusted bets:      {vorp_count:,}")
    print()
    print("-" * 78)
    print("  PERFORMANCE")
    print("-" * 78)
    print(f"  Win Rate:                {win_rate:.1f}%  ({int(wins)}/{total_bets})")
    print(f"  Average Odds:            {avg_odds:.3f}")
    print(f"  Flat ROI / Yield:        {flat_roi:+.2f}%")
    print()
    print("-" * 78)
    print("  KELLY COMPOUNDING")
    print("-" * 78)
    print(f"  Initial Bankroll:        {BANKROLL_INIT:,.0f} U")
    print(f"  Final Bankroll:          {final_br:,.2f} U")
    print(f"  Peak Bankroll:           {sim['peak_bankroll']:,.2f} U")
    print(f"  Net P&L:                 {final_br - BANKROLL_INIT:+,.2f} U")
    print(f"  Max Drawdown:            {max_dd:.2f}%")
    print(f"  CAGR:                    {cagr:+.2f}%")
    print(f"  Period:                  {days} days ({years:.2f} years)")
    print()

    # Per-league breakdown
    print("-" * 78)
    print("  DESGLOSE POR LIGA")
    print("-" * 78)

    league_stats = df_bets.groupby("league_id").agg(
        bets=("won", "count"),
        wins=("won", "sum"),
        avg_odds=("odds", "mean"),
        total_pnl=("pnl", "sum"),
    ).reset_index()
    league_stats["win_rate"] = league_stats["wins"] / league_stats["bets"] * 100
    league_stats["flat_roi"] = league_stats.apply(
        lambda r: (
            df_bets[df_bets["league_id"] == r["league_id"]].apply(
                lambda b: (b["odds"] - 1.0) if b["won"] else -1.0, axis=1
            ).sum() / r["bets"] * 100
        ), axis=1
    )

    print(f"  {'Liga':<15} {'Tier':<5} {'Bets':>5}  {'WR%':>6}  {'AvgOdds':>7}  {'FlatROI%':>9}  {'Kelly P&L':>10}")
    for _, row in league_stats.sort_values("flat_roi", ascending=False).iterrows():
        lid = int(row["league_id"])
        name = LEAGUE_NAMES.get(lid, f"L{lid}")
        tier = get_tier(lid)
        candidate = " *" if lid in WHITELIST_CANDIDATES else ""
        print(f"  {name:<15} {tier:<5} {int(row['bets']):>5}  {row['win_rate']:>5.1f}%  {row['avg_odds']:>7.3f}  {row['flat_roi']:>+8.2f}%  {row['total_pnl']:>+9.2f} U{candidate}")

    # Per-tier summary
    print()
    print("-" * 78)
    print("  DESGLOSE POR TIER")
    print("-" * 78)

    for tier_name, tier_leagues in [("GEO", TIER_GEO_LEAGUES), ("FLAT", TIER_FLAT_LEAGUES)]:
        tier_bets = df_bets[df_bets["league_id"].isin(tier_leagues)]
        if len(tier_bets) == 0:
            print(f"  {tier_name}: No bets")
            continue
        t_wins = tier_bets["won"].sum()
        t_total = len(tier_bets)
        t_wr = t_wins / t_total * 100
        t_flat = tier_bets.apply(
            lambda r: (r["odds"] - 1.0) if r["won"] else -1.0, axis=1
        ).sum() / t_total * 100
        t_pnl = tier_bets["pnl"].sum()
        print(f"  {tier_name:<5}: {t_total:>4} bets, WR {t_wr:.1f}%, Flat ROI {t_flat:+.2f}%, Kelly P&L {t_pnl:+.2f} U")

    print()

    # Candidate evaluation (GDT: if Bolivia/Peru profitable → whitelist)
    print("-" * 78)
    print("  EVALUACION CANDIDATOS KELLY WHITELIST")
    print("-" * 78)
    for lid in sorted(WHITELIST_CANDIDATES):
        cand_bets = df_bets[df_bets["league_id"] == lid]
        name = LEAGUE_NAMES.get(lid, f"L{lid}")
        if len(cand_bets) == 0:
            print(f"  {name} ({lid}): No bets — BLOCK")
            continue
        c_wins = cand_bets["won"].sum()
        c_total = len(cand_bets)
        c_flat = cand_bets.apply(
            lambda r: (r["odds"] - 1.0) if r["won"] else -1.0, axis=1
        ).sum() / c_total * 100
        c_pnl = cand_bets["pnl"].sum()
        verdict = "APPROVE" if c_flat > 0 and c_pnl > 0 else "BLOCK"
        print(f"  {name} ({lid}): {c_total} bets, Flat ROI {c_flat:+.2f}%, "
              f"Kelly P&L {c_pnl:+.2f} U -> {verdict}")

    print()
    print("=" * 78)

    # Overall verdict
    if flat_roi > 0 and final_br > BANKROLL_INIT:
        print("  VEREDICTO: GEO-ROUTER IMPRIME")
    elif flat_roi > 0:
        print("  VEREDICTO: EDGE POSITIVO (flat), Kelly sizing suboptimo")
    elif final_br > BANKROLL_INIT:
        print("  VEREDICTO: Kelly compounding compensa, flat ROI negativo")
    else:
        print("  VEREDICTO: SIN EDGE. El mercado gana.")
    print("=" * 78)


def plot_equity_curve(sim: dict, output_path: Path):
    """Generate equity curve PNG."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates

        dates = [e[0] for e in sim["equity_curve"]]
        values = [e[1] for e in sim["equity_curve"]]

        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(dates, values, linewidth=1.5, color="#7c3aed")
        ax.axhline(y=BANKROLL_INIT, color="#64748b", linestyle="--", alpha=0.5, label="Initial Bankroll")
        ax.fill_between(dates, values, BANKROLL_INIT, alpha=0.1,
                         where=[v >= BANKROLL_INIT for v in values], color="#22c55e")
        ax.fill_between(dates, values, BANKROLL_INIT, alpha=0.1,
                         where=[v < BANKROLL_INIT for v in values], color="#ef4444")

        ax.set_title("LATAM Sniper V1.4.0 GEO-ROUTER — Equity Curve", fontsize=14, fontweight="bold")
        ax.set_xlabel("Date")
        ax.set_ylabel("Bankroll (Units)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        plt.xticks(rotation=45)
        plt.tight_layout()

        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150)
        plt.close()
        print(f"\n  Equity curve saved to: {output_path}")
    except ImportError:
        print("\n  (matplotlib not installed — skipping equity curve plot)")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("\n[1/7] Loading lab data for 10 LATAM leagues (GEO + FLAT tiers)...")
    df_all = load_lab_data(ALL_LATAM, MIN_DATE)
    df_all["tier"] = df_all["league_id"].map(get_tier)
    print(f"  Loaded {len(df_all):,} matches")
    print(f"  GEO tier: {df_all[df_all['tier']=='geo'].shape[0]:,} matches")
    print(f"  FLAT tier: {df_all[df_all['tier']=='flat'].shape[0]:,} matches")

    print("\n[2/7] Chronological train/test split...")
    n = len(df_all)
    split_idx = int(n * (1 - OOS_FRACTION))
    split_date = df_all.iloc[split_idx]["date"]

    df_train = df_all.iloc[:split_idx].copy()
    df_test_all = df_all.iloc[split_idx:].copy()
    df_oos = df_test_all[df_test_all["league_id"].isin(WHITELIST)].copy()

    print(f"  Train: {len(df_train):,} (-> {split_date.date()})")
    print(f"  OOS total: {len(df_test_all):,}")
    print(f"  OOS whitelist: {len(df_oos):,}")
    for lid in sorted(WHITELIST):
        c = len(df_oos[df_oos["league_id"] == lid])
        tier = get_tier(lid)
        cand = " (CANDIDATE)" if lid in WHITELIST_CANDIDATES else ""
        print(f"    {LEAGUE_NAMES[lid]:<12}: {c:>4} [{tier}]{cand}")

    # ── Train GEO model (18f on Tier Geo leagues only) ──
    print("\n[3/7] Training GEO model (18f, Tier Geo leagues)...")
    df_train_geo = df_train[df_train["tier"] == "geo"].copy()
    for col in FEATURES_18:
        if col not in df_train_geo.columns:
            df_train_geo[col] = 0.0
    X_train_geo = df_train_geo[FEATURES_18].fillna(0.0).values
    y_train_geo = df_train_geo["result"].values.astype(int)
    m1_geo, m2_geo = train_twostage(X_train_geo, y_train_geo)
    print(f"  GEO: {len(X_train_geo):,} training samples, 18 features")

    # ── Train FLAT model (16f on Tier Flat leagues only) ──
    print("\n[4/7] Training FLAT model (16f, Tier Flat leagues)...")
    df_train_flat = df_train[df_train["tier"] == "flat"].copy()
    for col in FEATURES_16:
        if col not in df_train_flat.columns:
            df_train_flat[col] = 0.0
    X_train_flat = df_train_flat[FEATURES_16].fillna(0.0).values
    y_train_flat = df_train_flat["result"].values.astype(int)
    m1_flat, m2_flat = train_twostage(X_train_flat, y_train_flat)
    print(f"  FLAT: {len(X_train_flat):,} training samples, 16 features")

    # ── Generate OOS predictions with tier routing ──
    print("\n[5/7] Generating OOS predictions (tier-routed) + VORP injection...")

    # Prepare feature columns
    for col in FEATURES_18:
        if col not in df_oos.columns:
            df_oos[col] = 0.0

    # Route each match to the correct model
    df_oos_geo = df_oos[df_oos["tier"] == "geo"].copy()
    df_oos_flat = df_oos[df_oos["tier"] == "flat"].copy()

    # GEO predictions (18f)
    if len(df_oos_geo) > 0:
        X_geo = df_oos_geo[FEATURES_18].fillna(0.0).values
        probs_geo = predict_twostage(m1_geo, m2_geo, X_geo)
        df_oos_geo["p_home"] = probs_geo[:, 0]
        df_oos_geo["p_draw"] = probs_geo[:, 1]
        df_oos_geo["p_away"] = probs_geo[:, 2]
        print(f"  GEO OOS: {len(df_oos_geo):,} matches")

    # FLAT predictions (16f)
    if len(df_oos_flat) > 0:
        X_flat = df_oos_flat[FEATURES_16].fillna(0.0).values
        probs_flat = predict_twostage(m1_flat, m2_flat, X_flat)
        df_oos_flat["p_home"] = probs_flat[:, 0]
        df_oos_flat["p_draw"] = probs_flat[:, 1]
        df_oos_flat["p_away"] = probs_flat[:, 2]
        print(f"  FLAT OOS: {len(df_oos_flat):,} matches")

    # Recombine
    df_oos = pd.concat([df_oos_geo, df_oos_flat], ignore_index=True)
    df_oos = df_oos.sort_values("date").reset_index(drop=True)

    # VORP injection
    mtv = load_mtv_data()
    df_oos = df_oos.merge(mtv, on="match_id", how="left", suffixes=("", "_mtv"))
    td_col = "talent_delta_diff_mtv" if "talent_delta_diff_mtv" in df_oos.columns else "talent_delta_diff"
    df_oos["td_diff"] = df_oos[td_col].fillna(0.0) if td_col in df_oos.columns else 0.0

    vorp_count = 0
    for idx in df_oos.index:
        td = df_oos.at[idx, "td_diff"]
        if td != 0 and not np.isnan(td):
            base = np.array([
                df_oos.at[idx, "p_home"],
                df_oos.at[idx, "p_draw"],
                df_oos.at[idx, "p_away"],
            ])
            adj = apply_vorp_shock(base, td)
            df_oos.at[idx, "p_home"] = adj[0]
            df_oos.at[idx, "p_draw"] = adj[1]
            df_oos.at[idx, "p_away"] = adj[2]
            df_oos.at[idx, "vorp_applied"] = True
            vorp_count += 1
        else:
            df_oos.at[idx, "vorp_applied"] = False

    print(f"  Total OOS: {len(df_oos):,}")
    print(f"  VORP applied: {vorp_count:,} ({vorp_count/len(df_oos)*100:.1f}%)")

    print("\n[6/7] Running daily compounding simulation...")
    sim = simulate_daily(df_oos)
    print(f"  Trading days: {len(sim['daily_log']):,}")
    print(f"  Total bets: {len(sim['all_bets']):,}")

    print("\n[7/7] Executive Report\n")
    print_report(df_oos, sim, split_date)

    plot_equity_curve(sim, OUTPUT_DIR / "latam_sniper_v140_equity.png")

    if sim["all_bets"]:
        bet_log_path = OUTPUT_DIR / "latam_sniper_v140_bets.csv"
        pd.DataFrame(sim["all_bets"]).to_csv(bet_log_path, index=False)
        print(f"  Bet log saved to: {bet_log_path}")


if __name__ == "__main__":
    main()
