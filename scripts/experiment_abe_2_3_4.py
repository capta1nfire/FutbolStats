#!/usr/bin/env python3
"""
ABE Experiments 2, 3, 4 — Argentina (league_id=128)
=====================================================
Experiment 2: Rolling scopeado por season (reset history at breaks)
Experiment 3: Cap matches_played (remove time-proxy)
Experiment 4: Phase feature (regular vs knockout, 2025+ only)

Uses the same production hyperparams and evaluation as experiment 1.

Usage:
  source .env
  python scripts/experiment_abe_2_3_4.py
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

# ─── Config (same as feature_diagnostic.py) ──────────────────

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

ROLLING_WINDOW = 10
TIME_DECAY_LAMBDA = 0.01
N_SEEDS = 3
DRAW_WEIGHT = 1.5
ARGENTINA_LEAGUE_ID = 128
BREAK_THRESHOLD_DAYS = 45  # gap > 45 days = season/phase break


def multiclass_brier(y_true, y_prob):
    n_classes = y_prob.shape[1]
    y_onehot = np.eye(n_classes)[y_true.astype(int)]
    return float(np.mean(np.sum((y_prob - y_onehot) ** 2, axis=1)))


def train_and_evaluate(X_train, y_train, X_test, y_test):
    briers, loglosses, accs = [], [], []
    for seed in range(N_SEEDS):
        params = {**PROD_HYPERPARAMS, "random_state": seed * 42}
        model = xgb.XGBClassifier(**params)
        sw = np.ones(len(y_train), dtype=np.float32)
        sw[y_train == 1] = DRAW_WEIGHT
        model.fit(X_train, y_train, sample_weight=sw)
        y_prob = model.predict_proba(X_test)
        briers.append(multiclass_brier(y_test, y_prob))
        loglosses.append(log_loss(y_test, y_prob, labels=[0, 1, 2]))
        accs.append(float(np.mean(model.predict(X_test) == y_test)))
    return {
        "brier_mean": round(np.mean(briers), 6),
        "brier_std": round(np.std(briers), 6),
        "logloss_mean": round(np.mean(loglosses), 6),
        "accuracy_mean": round(np.mean(accs), 4),
    }


def naive_baseline_brier(y_train, y_test):
    total = len(y_train)
    probs = np.array([np.sum(y_train == c) / total for c in range(3)])
    y_prob = np.tile(probs, (len(y_test), 1))
    return multiclass_brier(y_test, y_prob)


# ─── Feature recomputation functions ─────────────────────────

def compute_rolling_features(team_matches_df, reset_on_break=False, cap_matches=None):
    """Recompute rolling features per team.

    Args:
        team_matches_df: DataFrame with match_id, date, team_id, goals_scored,
                         goals_conceded, shots, corners, match_weight
        reset_on_break: If True, reset history when gap > BREAK_THRESHOLD_DAYS
        cap_matches: If set, clamp matches_played to this value
    """
    def _compute_one_team(group):
        group = group.sort_values("date")
        results = []
        history = []

        for _, row in group.iterrows():
            # Check for break reset
            if reset_on_break and len(history) > 0:
                gap = (row["date"] - history[-1]["date"]).days
                if gap > BREAK_THRESHOLD_DAYS:
                    history = []  # RESET

            if len(history) > 0:
                window = history[-ROLLING_WINDOW:]
                ref_date = row["date"]
                total_w = 0
                sum_gs, sum_gc, sum_sh, sum_co = 0.0, 0.0, 0.0, 0.0
                for h in window:
                    days = (ref_date - h["date"]).days
                    decay = np.exp(-TIME_DECAY_LAMBDA * days)
                    w = h["match_weight"] * decay
                    total_w += w
                    sum_gs += h["goals_scored"] * w
                    sum_gc += h["goals_conceded"] * w
                    sum_sh += h["shots"] * w
                    sum_co += h["corners"] * w

                if total_w > 0:
                    gsa = sum_gs / total_w
                    gca = sum_gc / total_w
                    sha = sum_sh / total_w
                    coa = sum_co / total_w
                else:
                    gsa, gca, sha, coa = 1.0, 1.0, 10.0, 4.0

                rest_days = (ref_date - history[-1]["date"]).days
                mp = len(history)
                if cap_matches is not None:
                    mp = min(mp, cap_matches)
            else:
                gsa, gca, sha, coa = 1.0, 1.0, 10.0, 4.0
                rest_days = 30
                mp = 0

            results.append({
                "match_id": row["match_id"],
                "team_id": row["team_id"],
                "goals_scored_avg": round(gsa, 3),
                "goals_conceded_avg": round(gca, 3),
                "shots_avg": round(sha, 3),
                "corners_avg": round(coa, 3),
                "rest_days": rest_days,
                "matches_played": mp,
            })

            history.append({
                "date": row["date"],
                "goals_scored": row["goals_scored"],
                "goals_conceded": row["goals_conceded"],
                "shots": row["shots"],
                "corners": row["corners"],
                "match_weight": row["match_weight"],
            })

        return pd.DataFrame(results)

    return team_matches_df.groupby("team_id", group_keys=False).apply(
        _compute_one_team
    ).reset_index(drop=True)


def build_dataset_from_raw(matches_df, team_features_df):
    """Merge team features back to match level and compute derived features."""
    home_feats = team_features_df.merge(
        matches_df[["match_id", "home_team_id"]],
        left_on=["match_id", "team_id"],
        right_on=["match_id", "home_team_id"],
    ).drop(columns=["team_id", "home_team_id"])
    home_feats = home_feats.rename(columns={
        c: f"home_{c}" for c in ["goals_scored_avg", "goals_conceded_avg",
                                   "shots_avg", "corners_avg", "rest_days", "matches_played"]
    })

    away_feats = team_features_df.merge(
        matches_df[["match_id", "away_team_id"]],
        left_on=["match_id", "team_id"],
        right_on=["match_id", "away_team_id"],
    ).drop(columns=["team_id", "away_team_id"])
    away_feats = away_feats.rename(columns={
        c: f"away_{c}" for c in ["goals_scored_avg", "goals_conceded_avg",
                                   "shots_avg", "corners_avg", "rest_days", "matches_played"]
    })

    df = matches_df[["match_id", "date", "league_id", "home_team_id", "away_team_id",
                       "home_goals", "away_goals"]].copy()
    df = df.merge(home_feats, on="match_id", how="left")
    df = df.merge(away_feats, on="match_id", how="left")

    # Derived features
    df["goal_diff_avg"] = df["home_goals_scored_avg"] - df["away_goals_scored_avg"]
    df["rest_diff"] = df["home_rest_days"] - df["away_rest_days"]
    df["abs_attack_diff"] = (df["home_goals_scored_avg"] - df["away_goals_scored_avg"]).abs()
    df["abs_defense_diff"] = (df["home_goals_conceded_avg"] - df["away_goals_conceded_avg"]).abs()
    home_net = df["home_goals_scored_avg"] - df["home_goals_conceded_avg"]
    away_net = df["away_goals_scored_avg"] - df["away_goals_conceded_avg"]
    df["abs_strength_gap"] = (home_net - away_net).abs()

    # Label
    df["result"] = np.where(
        df["home_goals"] > df["away_goals"], 0,
        np.where(df["home_goals"] == df["away_goals"], 1, 2)
    )

    return df


def extract_argentina_raw():
    """Extract raw Argentina match data + round info from DB."""
    import psycopg2
    from app.config import get_settings
    settings = get_settings()
    db_url = settings.DATABASE_URL
    if db_url.startswith("postgres://"):
        db_url = db_url.replace("postgres://", "postgresql://", 1)

    conn = psycopg2.connect(db_url)

    query = """
        SELECT m.id AS match_id, m.date, m.league_id,
               m.home_team_id, m.away_team_id,
               m.home_goals, m.away_goals,
               m.stats, m.match_weight, m.round,
               m.odds_home, m.odds_draw, m.odds_away
        FROM matches m
        WHERE m.status = 'FT'
          AND m.home_goals IS NOT NULL
          AND m.away_goals IS NOT NULL
          AND m.tainted = false
          AND m.league_id = %s
        ORDER BY m.date
    """
    print("  Querying Argentina matches from DB...")
    matches = pd.read_sql(query, conn, params=(ARGENTINA_LEAGUE_ID,))
    conn.close()
    print(f"  Raw Argentina matches: {len(matches)}")

    # Flatten stats JSON
    def extract_side_stats(stats, side):
        if not stats or not isinstance(stats, dict):
            return 0, 0
        s = stats.get(side, {})
        shots = s.get("total_shots", s.get("shots_on_goal", 0)) or 0
        corners = s.get("corner_kicks", 0) or 0
        return int(shots), int(corners)

    matches["home_shots"] = matches["stats"].apply(lambda s: extract_side_stats(s, "home")[0])
    matches["home_corners"] = matches["stats"].apply(lambda s: extract_side_stats(s, "home")[1])
    matches["away_shots"] = matches["stats"].apply(lambda s: extract_side_stats(s, "away")[0])
    matches["away_corners"] = matches["stats"].apply(lambda s: extract_side_stats(s, "away")[1])
    matches["match_weight"] = matches["match_weight"].fillna(1.0)

    return matches


def make_team_rows(matches):
    """Create team-centric rows from match data."""
    home = matches[["match_id", "date", "home_team_id", "home_goals", "away_goals",
                     "home_shots", "home_corners", "match_weight"]].copy()
    home.columns = ["match_id", "date", "team_id", "goals_scored", "goals_conceded",
                     "shots", "corners", "match_weight"]

    away = matches[["match_id", "date", "away_team_id", "away_goals", "home_goals",
                     "away_shots", "away_corners", "match_weight"]].copy()
    away.columns = ["match_id", "date", "team_id", "goals_scored", "goals_conceded",
                     "shots", "corners", "match_weight"]

    return pd.concat([home, away]).sort_values(["team_id", "date"])


def evaluate_variant(label, df, features, split_year=2025):
    """Train ≤split_year-1, test split_year. Return metrics."""
    df_clean = df.dropna(subset=features)
    df_train = df_clean[df_clean["date"].dt.year < split_year]
    df_test = df_clean[df_clean["date"].dt.year == split_year]

    if len(df_test) < 20:
        return {"variant": label, "status": "SKIPPED", "n_test": len(df_test)}

    X_train = df_train[features].values.astype(np.float32)
    y_train = df_train["result"].values.astype(int)
    X_test = df_test[features].values.astype(np.float32)
    y_test = df_test["result"].values.astype(int)

    metrics = train_and_evaluate(X_train, y_train, X_test, y_test)
    naive = naive_baseline_brier(y_train, y_test)

    return {
        "variant": label,
        "n_train": len(df_train),
        "n_test": len(df_test),
        "naive_brier": round(naive, 6),
        **metrics,
        "skill_vs_naive": round(naive - metrics["brier_mean"], 6),
    }


def main():
    print(f"\n  ABE EXPERIMENTS 2, 3, 4 — ARGENTINA")
    print(f"  {'=' * 50}")

    # ─── Extract raw data ─────────────────────────────────────
    matches = extract_argentina_raw()
    team_rows = make_team_rows(matches)

    # ─── BASELINE: Production features (no changes) ──────────
    print(f"\n{'=' * 70}")
    print(f"  BASELINE: Production features (same as v1.0.1)")
    print(f"{'=' * 70}")

    baseline_feats = compute_rolling_features(team_rows, reset_on_break=False, cap_matches=None)
    df_baseline = build_dataset_from_raw(matches, baseline_feats)
    print(f"  Dataset: {len(df_baseline)} rows")

    results_all = {}

    # Evaluate baseline for 2025 (main comparison year)
    for test_year in [2024, 2025]:
        r = evaluate_variant(f"baseline_test{test_year}", df_baseline, FEATURES_V101, split_year=test_year)
        results_all[f"baseline_test{test_year}"] = r
        print(f"\n  Baseline → test {test_year}")
        print(f"    N_train={r.get('n_train','?')} N_test={r.get('n_test','?')}")
        print(f"    Naive:  {r.get('naive_brier','?'):.4f}" if 'naive_brier' in r else "    SKIPPED")
        print(f"    Model:  {r.get('brier_mean','?'):.4f} ± {r.get('brier_std','?'):.4f}" if 'brier_mean' in r else "")
        print(f"    Skill:  {r.get('skill_vs_naive','?'):+.4f}" if 'skill_vs_naive' in r else "")

    # ─── EXPERIMENT 2: Rolling scopeado (reset on break) ─────
    print(f"\n{'=' * 70}")
    print(f"  EXPERIMENT 2: ROLLING SCOPEADO (reset on {BREAK_THRESHOLD_DAYS}-day gap)")
    print(f"{'=' * 70}")

    scoped_feats = compute_rolling_features(team_rows, reset_on_break=True, cap_matches=None)
    df_scoped = build_dataset_from_raw(matches, scoped_feats)

    # Count resets
    team_rows_sorted = team_rows.sort_values(["team_id", "date"])
    n_resets = 0
    for _, grp in team_rows_sorted.groupby("team_id"):
        dates = grp["date"].sort_values().values
        for i in range(1, len(dates)):
            gap = (pd.Timestamp(dates[i]) - pd.Timestamp(dates[i-1])).days
            if gap > BREAK_THRESHOLD_DAYS:
                n_resets += 1
    print(f"  History resets triggered: {n_resets}")

    for test_year in [2024, 2025]:
        r = evaluate_variant(f"scoped_test{test_year}", df_scoped, FEATURES_V101, split_year=test_year)
        results_all[f"exp2_scoped_test{test_year}"] = r
        print(f"\n  Scoped → test {test_year}")
        print(f"    N_train={r.get('n_train','?')} N_test={r.get('n_test','?')}")
        print(f"    Naive:  {r.get('naive_brier','?'):.4f}" if 'naive_brier' in r else "    SKIPPED")
        print(f"    Model:  {r.get('brier_mean','?'):.4f} ± {r.get('brier_std','?'):.4f}" if 'brier_mean' in r else "")
        print(f"    Skill:  {r.get('skill_vs_naive','?'):+.4f}" if 'skill_vs_naive' in r else "")

    # ─── EXPERIMENT 3: Cap matches_played ─────────────────────
    print(f"\n{'=' * 70}")
    print(f"  EXPERIMENT 3: CAP MATCHES_PLAYED")
    print(f"{'=' * 70}")

    for cap in [10, 20, 5]:
        capped_feats = compute_rolling_features(team_rows, reset_on_break=False, cap_matches=cap)
        df_capped = build_dataset_from_raw(matches, capped_feats)

        for test_year in [2024, 2025]:
            r = evaluate_variant(f"cap{cap}_test{test_year}", df_capped, FEATURES_V101, split_year=test_year)
            results_all[f"exp3_cap{cap}_test{test_year}"] = r
            print(f"\n  Cap={cap} → test {test_year}")
            print(f"    Model:  {r.get('brier_mean','?'):.4f} ± {r.get('brier_std','?'):.4f}" if 'brier_mean' in r else "    SKIPPED")
            print(f"    Skill:  {r.get('skill_vs_naive','?'):+.4f}" if 'skill_vs_naive' in r else "")

    # ─── EXPERIMENT 3b: Remove matches_played entirely ────────
    print(f"\n  --- Variant: Remove matches_played entirely ---")
    features_no_mp = [f for f in FEATURES_V101 if "matches_played" not in f]
    for test_year in [2024, 2025]:
        r = evaluate_variant(f"no_mp_test{test_year}", df_baseline, features_no_mp, split_year=test_year)
        results_all[f"exp3_no_mp_test{test_year}"] = r
        print(f"\n  No matches_played → test {test_year}")
        print(f"    Model:  {r.get('brier_mean','?'):.4f} ± {r.get('brier_std','?'):.4f}" if 'brier_mean' in r else "    SKIPPED")
        print(f"    Skill:  {r.get('skill_vs_naive','?'):+.4f}" if 'skill_vs_naive' in r else "")

    # ─── EXPERIMENT 4: Phase feature (2025+ only) ────────────
    print(f"\n{'=' * 70}")
    print(f"  EXPERIMENT 4: PHASE FEATURE (regular vs knockout, 2025+ only)")
    print(f"{'=' * 70}")

    # Derive phase_type from round column
    def classify_phase(round_val):
        if pd.isna(round_val) or round_val is None:
            return 0  # unknown (2020-2024)
        r = str(round_val).lower()
        if any(kw in r for kw in ["quarter", "semi", "final", "round of 16", "8th"]):
            return 2  # knockout
        elif "2nd phase" in r:
            return 1  # phase 2
        else:
            return 0  # phase 1 / unknown

    df_phase = df_baseline.copy()
    # Need round info → merge from matches
    round_map = matches[["match_id", "round"]].copy()
    df_phase = df_phase.merge(round_map, on="match_id", how="left")
    df_phase["is_knockout"] = df_phase["round"].apply(
        lambda r: 1 if r and any(kw in str(r).lower()
                                  for kw in ["quarter", "semi", "final", "round of 16", "8th"])
        else 0
    )
    df_phase["phase_num"] = df_phase["round"].apply(classify_phase)

    # 4a: Add is_knockout as feature
    features_ko = FEATURES_V101 + ["is_knockout"]
    r = evaluate_variant("phase_is_ko_test2025", df_phase, features_ko, split_year=2025)
    results_all["exp4_is_knockout_test2025"] = r
    print(f"\n  +is_knockout → test 2025 (note: 2024 train has all 0s)")
    print(f"    Model:  {r.get('brier_mean','?'):.4f}" if 'brier_mean' in r else "    SKIPPED")
    print(f"    Skill:  {r.get('skill_vs_naive','?'):+.4f}" if 'skill_vs_naive' in r else "")

    # 4b: Filter out knockout matches entirely (only evaluate on regular)
    df_regular_only = df_phase[df_phase["is_knockout"] == 0].copy()
    r = evaluate_variant("regular_only_test2025", df_regular_only, FEATURES_V101, split_year=2025)
    results_all["exp4_regular_only_test2025"] = r
    n_ko_2025 = len(df_phase[(df_phase["date"].dt.year == 2025) & (df_phase["is_knockout"] == 1)])
    print(f"\n  Regular-only (exclude {n_ko_2025} KO matches from 2025) → test 2025")
    print(f"    N_test={r.get('n_test','?')} (was 503 with KO)")
    print(f"    Model:  {r.get('brier_mean','?'):.4f}" if 'brier_mean' in r else "    SKIPPED")
    print(f"    Skill:  {r.get('skill_vs_naive','?'):+.4f}" if 'skill_vs_naive' in r else "")

    # 4c: Combined: scoped rolling + exclude KO
    scoped_phase = build_dataset_from_raw(matches, scoped_feats)
    scoped_phase = scoped_phase.merge(round_map, on="match_id", how="left")
    scoped_phase["is_knockout"] = scoped_phase["round"].apply(
        lambda r: 1 if r and any(kw in str(r).lower()
                                  for kw in ["quarter", "semi", "final", "round of 16", "8th"])
        else 0
    )
    df_combined = scoped_phase[scoped_phase["is_knockout"] == 0].copy()
    r = evaluate_variant("scoped+regular_test2025", df_combined, FEATURES_V101, split_year=2025)
    results_all["exp4_combined_test2025"] = r
    print(f"\n  Combined (scoped rolling + exclude KO) → test 2025")
    print(f"    Model:  {r.get('brier_mean','?'):.4f}" if 'brier_mean' in r else "    SKIPPED")
    print(f"    Skill:  {r.get('skill_vs_naive','?'):+.4f}" if 'skill_vs_naive' in r else "")

    # ─── SUMMARY TABLE ────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"  SUMMARY: ALL EXPERIMENTS (test 2025)")
    print(f"{'=' * 70}")
    print(f"\n  {'Variant':<40} {'Brier':>8} {'Skill':>8} {'N_test':>7}")
    print(f"  {'-'*40} {'-'*8} {'-'*8} {'-'*7}")

    summary_keys_2025 = [
        ("Baseline (production)", "baseline_test2025"),
        ("Exp2: Scoped rolling (45d reset)", "exp2_scoped_test2025"),
        ("Exp3: Cap MP=10", "exp3_cap10_test2025"),
        ("Exp3: Cap MP=20", "exp3_cap20_test2025"),
        ("Exp3: Cap MP=5", "exp3_cap5_test2025"),
        ("Exp3: Remove matches_played", "exp3_no_mp_test2025"),
        ("Exp4: +is_knockout feature", "exp4_is_knockout_test2025"),
        ("Exp4: Regular-only (excl KO)", "exp4_regular_only_test2025"),
        ("Exp4: Scoped + Regular-only", "exp4_combined_test2025"),
    ]

    for label, key in summary_keys_2025:
        r = results_all.get(key, {})
        if "brier_mean" in r:
            print(f"  {label:<40} {r['brier_mean']:>8.4f} {r['skill_vs_naive']:>+8.4f} {r['n_test']:>7}")
        else:
            print(f"  {label:<40} {'SKIP':>8}")

    # Also show 2024 comparison
    print(f"\n  {'Variant':<40} {'Brier':>8} {'Skill':>8} {'N_test':>7}  (test 2024)")
    print(f"  {'-'*40} {'-'*8} {'-'*8} {'-'*7}")
    summary_keys_2024 = [
        ("Baseline (production)", "baseline_test2024"),
        ("Exp2: Scoped rolling (45d reset)", "exp2_scoped_test2024"),
        ("Exp3: Cap MP=10", "exp3_cap10_test2024"),
        ("Exp3: Cap MP=5", "exp3_cap5_test2024"),
        ("Exp3: Remove matches_played", "exp3_no_mp_test2024"),
    ]
    for label, key in summary_keys_2024:
        r = results_all.get(key, {})
        if "brier_mean" in r:
            print(f"  {label:<40} {r['brier_mean']:>8.4f} {r['skill_vs_naive']:>+8.4f} {r['n_test']:>7}")
        else:
            print(f"  {label:<40} {'SKIP':>8}")

    # Save
    output_file = Path("scripts/output/experiment_abe_2_3_4.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(results_all, f, indent=2, default=str)
    print(f"\n  Results saved to: {output_file}")


if __name__ == "__main__":
    main()
