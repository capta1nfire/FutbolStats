#!/usr/bin/env python3
"""
xG Signal Ablation Test (ABE P0 approved 2026-02-09)
=====================================================
A/B test: Model A (14 baseline features) vs Model B (14 + 4 xG features).
Same training/test sets, same hyperparameters, temporal split.

ABE Directives:
- Same-set A/B: identical match sets for A and B
- Rolling xG: primary window=5, sensitivity 3 and 10
- Metrics: Brier + LogLoss + bootstrap CI of delta
- Iteration 1: only xG (4 features), no xGOT
- PIT strict: rolling uses only m.date < t0
- Only 2023-2026 data (avoid missingness confound)
- Only confidence >= 0.90 FotMob refs
- Success threshold: ΔBrier ≤ -0.005

Usage:
    set -a && source .env && set +a
    python3.12 scripts/xg_signal_test.py                          # Argentina (default)
    python3.12 scripts/xg_signal_test.py --league colombia        # Colombia
    python3.12 scripts/xg_signal_test.py --extract --league colombia  # re-extract
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

# ─── Configuration ───────────────────────────────────────────

BASELINE_FEATURES = [
    "home_goals_scored_avg", "home_goals_conceded_avg",
    "home_shots_avg", "home_corners_avg",
    "home_rest_days", "home_matches_played",
    "away_goals_scored_avg", "away_goals_conceded_avg",
    "away_shots_avg", "away_corners_avg",
    "away_rest_days", "away_matches_played",
    "goal_diff_avg", "rest_diff",
]

# Baseline WITHOUT shots — for redundancy test (ABE task 2)
BASELINE_NO_SHOTS = [f for f in BASELINE_FEATURES
                     if "shots_avg" not in f]

XG_FEATURES = [
    "xg_for_home", "xg_against_home",
    "xg_for_away", "xg_against_away",
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

N_SEEDS = 5
DRAW_WEIGHT = 1.5
N_BOOTSTRAP = 1000
ROLLING_WINDOW = 10
TIME_DECAY_LAMBDA = 0.01
XG_WINDOWS = [3, 5, 10]  # ABE: primary=5, sensitivity 3 and 10
MIN_TEST_DEFAULT = 100  # Argentina
MIN_TEST_SMALL = 30     # Small leagues (Colombia etc.)
SUCCESS_THRESHOLD = -0.005  # ΔBrier ≤ -0.005 = SIGNAL

LEAGUE_CONFIGS = {
    "argentina": {"league_id": 128, "name": "Argentina (128)", "min_test": MIN_TEST_DEFAULT,
                  "season_filter": "m.season >= 2023", "source": "fotmob"},
    "colombia":  {"league_id": 239, "name": "Colombia (239)", "min_test": MIN_TEST_SMALL,
                  "season_filter": "m.date >= '2025-07-01'", "source": "fotmob"},
    # Control positivo Top5 — Understat (ABE task 1)
    "laliga":    {"league_id": 140, "name": "LaLiga (140)", "min_test": MIN_TEST_SMALL,
                  "season_filter": "m.season >= 2024", "source": "understat"},
    "epl":       {"league_id": 39, "name": "EPL (39)", "min_test": MIN_TEST_SMALL,
                  "season_filter": "m.season >= 2024", "source": "understat"},
}


# ─── Data Extraction ─────────────────────────────────────────

def extract_league_data(league_config: dict, output_path: str = None) -> pd.DataFrame:
    """Extract matches with baseline features + FotMob xG for a given league.

    PIT-safe: rolling features use only matches before kickoff.
    Only FotMob refs with confidence >= 0.90.
    """
    import psycopg2
    from app.config import get_settings
    settings = get_settings()

    db_url = settings.DATABASE_URL
    if db_url.startswith("postgres://"):
        db_url = db_url.replace("postgres://", "postgresql://", 1)

    conn = psycopg2.connect(db_url)

    league_id = league_config["league_id"]
    season_filter = league_config["season_filter"]
    source = league_config.get("source", "fotmob")

    if source == "understat":
        # Understat: direct join on match_id, no refs needed
        query = f"""
            SELECT m.id AS match_id, m.date, m.league_id,
                   m.home_team_id, m.away_team_id,
                   m.home_goals, m.away_goals,
                   m.stats, m.match_weight,
                   m.odds_home, m.odds_draw, m.odds_away,
                   ut.xg_home AS fotmob_xg_home,
                   ut.xg_away AS fotmob_xg_away
            FROM matches m
            JOIN match_understat_team ut ON m.id = ut.match_id
            WHERE m.league_id = {league_id}
              AND m.status = 'FT'
              AND m.home_goals IS NOT NULL
              AND m.away_goals IS NOT NULL
              AND m.tainted = false
              AND {season_filter}
              AND ut.xg_home IS NOT NULL
              AND ut.xg_away IS NOT NULL
            ORDER BY m.date
        """
    else:
        # FotMob: requires match_external_refs linkage
        query = f"""
            SELECT m.id AS match_id, m.date, m.league_id,
                   m.home_team_id, m.away_team_id,
                   m.home_goals, m.away_goals,
                   m.stats, m.match_weight,
                   m.odds_home, m.odds_draw, m.odds_away,
                   mfs.xg_home AS fotmob_xg_home,
                   mfs.xg_away AS fotmob_xg_away
            FROM matches m
            JOIN match_external_refs mer
                ON m.id = mer.match_id AND mer.source = 'fotmob'
                AND mer.confidence >= 0.90
            JOIN match_fotmob_stats mfs
                ON m.id = mfs.match_id
            WHERE m.league_id = {league_id}
              AND m.status = 'FT'
              AND m.home_goals IS NOT NULL
              AND m.away_goals IS NOT NULL
              AND m.tainted = false
              AND {season_filter}
              AND mfs.xg_home IS NOT NULL
              AND mfs.xg_away IS NOT NULL
            ORDER BY m.date
        """
    print(f"  Querying {league_config['name']} matches with {source} xG...")
    matches = pd.read_sql(query, conn)
    conn.close()
    print(f"  Raw matches: {len(matches)}")

    if matches.empty:
        print("  [ERROR] No matches found")
        return pd.DataFrame()

    # 2. Flatten stats JSON → shots, corners per side
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

    # 3. Build team-level rolling baseline features (PIT-safe)
    print("  Computing rolling baseline features...")
    home_rows = matches[["match_id", "date", "home_team_id", "home_goals", "away_goals",
                          "home_shots", "home_corners", "match_weight"]].copy()
    home_rows.columns = ["match_id", "date", "team_id", "goals_scored", "goals_conceded",
                          "shots", "corners", "match_weight"]

    away_rows = matches[["match_id", "date", "away_team_id", "away_goals", "home_goals",
                          "away_shots", "away_corners", "match_weight"]].copy()
    away_rows.columns = ["match_id", "date", "team_id", "goals_scored", "goals_conceded",
                          "shots", "corners", "match_weight"]

    team_matches = pd.concat([home_rows, away_rows]).sort_values(["team_id", "date"])

    def compute_team_rolling(group):
        group = group.sort_values("date")
        results = []
        history = []

        for _, row in group.iterrows():
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
                    goals_scored_avg = sum_gs / total_w
                    goals_conceded_avg = sum_gc / total_w
                    shots_avg = sum_sh / total_w
                    corners_avg = sum_co / total_w
                else:
                    goals_scored_avg, goals_conceded_avg = 1.0, 1.0
                    shots_avg, corners_avg = 10.0, 4.0

                rest_days = (ref_date - history[-1]["date"]).days
                matches_played = len(history)
            else:
                goals_scored_avg, goals_conceded_avg = 1.0, 1.0
                shots_avg, corners_avg = 10.0, 4.0
                rest_days, matches_played = 30, 0

            results.append({
                "match_id": row["match_id"],
                "team_id": row["team_id"],
                "goals_scored_avg": round(goals_scored_avg, 3),
                "goals_conceded_avg": round(goals_conceded_avg, 3),
                "shots_avg": round(shots_avg, 3),
                "corners_avg": round(corners_avg, 3),
                "rest_days": rest_days,
                "matches_played": matches_played,
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

    team_features = team_matches.groupby("team_id", group_keys=False).apply(
        compute_team_rolling
    ).reset_index(drop=True)

    # 4. Merge baseline features
    home_feats = team_features.merge(
        matches[["match_id", "home_team_id"]],
        left_on=["match_id", "team_id"],
        right_on=["match_id", "home_team_id"],
    ).drop(columns=["team_id", "home_team_id"])
    home_feats = home_feats.rename(columns={
        c: f"home_{c}" for c in ["goals_scored_avg", "goals_conceded_avg",
                                   "shots_avg", "corners_avg", "rest_days", "matches_played"]
    })

    away_feats = team_features.merge(
        matches[["match_id", "away_team_id"]],
        left_on=["match_id", "team_id"],
        right_on=["match_id", "away_team_id"],
    ).drop(columns=["team_id", "away_team_id"])
    away_feats = away_feats.rename(columns={
        c: f"away_{c}" for c in ["goals_scored_avg", "goals_conceded_avg",
                                   "shots_avg", "corners_avg", "rest_days", "matches_played"]
    })

    df = matches[["match_id", "date", "league_id", "home_team_id", "away_team_id",
                   "home_goals", "away_goals", "odds_home", "odds_draw", "odds_away",
                   "fotmob_xg_home", "fotmob_xg_away"]].copy()
    df = df.merge(home_feats, on="match_id", how="left")
    df = df.merge(away_feats, on="match_id", how="left")

    # Derived baseline features
    df["goal_diff_avg"] = df["home_goals_scored_avg"] - df["away_goals_scored_avg"]
    df["rest_diff"] = df["home_rest_days"] - df["away_rest_days"]

    # Label: 0=home, 1=draw, 2=away
    df["result"] = np.where(
        df["home_goals"] > df["away_goals"], 0,
        np.where(df["home_goals"] == df["away_goals"], 1, 2)
    )

    # 5. Compute rolling xG features (PIT-safe, multiple windows)
    print("  Computing rolling xG features...")
    df = df.sort_values("date").reset_index(drop=True)

    # Build team-level xG history from FotMob
    # For each match: home team's xG = fotmob_xg_home, away team's xG = fotmob_xg_away
    # xG_for = team's own xG, xG_against = opponent's xG
    xg_rows_home = df[["match_id", "date", "home_team_id",
                         "fotmob_xg_home", "fotmob_xg_away"]].copy()
    xg_rows_home.columns = ["match_id", "date", "team_id", "xg_for", "xg_against"]

    xg_rows_away = df[["match_id", "date", "away_team_id",
                         "fotmob_xg_away", "fotmob_xg_home"]].copy()
    xg_rows_away.columns = ["match_id", "date", "team_id", "xg_for", "xg_against"]

    xg_team = pd.concat([xg_rows_home, xg_rows_away]).sort_values(["team_id", "date"])

    def compute_rolling_xg(group, window_size):
        """Compute rolling avg xG for one team. PIT-safe: only prior matches."""
        group = group.sort_values("date")
        results = []
        history_for = []
        history_against = []
        history_dates = []

        for _, row in group.iterrows():
            if len(history_for) > 0:
                w = history_for[-window_size:]
                wa = history_against[-window_size:]
                xg_for_avg = float(np.mean(w))
                xg_against_avg = float(np.mean(wa))
            else:
                xg_for_avg = None
                xg_against_avg = None

            results.append({
                "match_id": row["match_id"],
                "team_id": row["team_id"],
                "xg_for_avg": xg_for_avg,
                "xg_against_avg": xg_against_avg,
            })

            history_for.append(row["xg_for"])
            history_against.append(row["xg_against"])
            history_dates.append(row["date"])

        return pd.DataFrame(results)

    # Compute for each window size
    for w in XG_WINDOWS:
        print(f"    Window={w}...")
        xg_rolling = xg_team.groupby("team_id", group_keys=False).apply(
            lambda g: compute_rolling_xg(g, w)
        ).reset_index(drop=True)

        # Merge home team xG
        xg_home = xg_rolling.merge(
            df[["match_id", "home_team_id"]],
            left_on=["match_id", "team_id"],
            right_on=["match_id", "home_team_id"],
        ).drop(columns=["team_id", "home_team_id"])
        xg_home = xg_home.rename(columns={
            "xg_for_avg": f"xg_for_home_w{w}",
            "xg_against_avg": f"xg_against_home_w{w}",
        })

        # Merge away team xG
        xg_away = xg_rolling.merge(
            df[["match_id", "away_team_id"]],
            left_on=["match_id", "team_id"],
            right_on=["match_id", "away_team_id"],
        ).drop(columns=["team_id", "away_team_id"])
        xg_away = xg_away.rename(columns={
            "xg_for_avg": f"xg_for_away_w{w}",
            "xg_against_avg": f"xg_against_away_w{w}",
        })

        df = df.merge(xg_home, on="match_id", how="left")
        df = df.merge(xg_away, on="match_id", how="left")

    print(f"  Final dataset: {len(df)} rows")

    # Drop rows where any baseline feature is NaN
    baseline_cols = BASELINE_FEATURES
    before = len(df)
    df = df.dropna(subset=baseline_cols)
    print(f"  After dropping NaN baseline: {len(df)} rows (dropped {before - len(df)})")

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"  Saved to {output_path}")

    return df


# ─── Metrics ─────────────────────────────────────────────────

def multiclass_brier(y_true, y_prob):
    """Multi-class Brier score (lower = better)."""
    n_classes = y_prob.shape[1]
    y_onehot = np.eye(n_classes)[y_true.astype(int)]
    return float(np.mean(np.sum((y_prob - y_onehot) ** 2, axis=1)))


def compute_metrics(model, X, y):
    """Compute Brier + LogLoss."""
    y_prob = model.predict_proba(X)
    return {
        "brier": multiclass_brier(y, y_prob),
        "logloss": log_loss(y, y_prob, labels=[0, 1, 2]),
    }


# ─── Train ───────────────────────────────────────────────────

def train_xgb(X_train, y_train, seed=42):
    params = {**PROD_HYPERPARAMS, "random_state": seed}
    model = xgb.XGBClassifier(**params)
    sample_weight = np.ones(len(y_train), dtype=np.float32)
    sample_weight[y_train == 1] = DRAW_WEIGHT
    model.fit(X_train, y_train, sample_weight=sample_weight, verbose=False)
    return model


# ─── Temporal Split ──────────────────────────────────────────

def temporal_split(df, test_fraction=0.2):
    """Split by date (not random). PIT anti-leakage."""
    df_sorted = df.sort_values("date").reset_index(drop=True)
    split_idx = int(len(df_sorted) * (1 - test_fraction))
    return df_sorted.iloc[:split_idx].copy(), df_sorted.iloc[split_idx:].copy()


# ─── Bootstrap CI ────────────────────────────────────────────

def bootstrap_delta_ci(brier_a_samples, brier_b_samples, n_bootstrap=N_BOOTSTRAP, seed=42):
    """Bootstrap CI for Brier delta (B - A). Negative = B is better.

    Each sample is a per-match squared error array.
    """
    rng = np.random.RandomState(seed)
    n = len(brier_a_samples)
    deltas = []

    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        delta = float(np.mean(brier_b_samples[idx]) - np.mean(brier_a_samples[idx]))
        deltas.append(delta)

    deltas = sorted(deltas)
    ci_lo = deltas[int(0.025 * n_bootstrap)]
    ci_hi = deltas[int(0.975 * n_bootstrap)]
    mean_delta = float(np.mean(deltas))

    return {"mean": mean_delta, "ci95_lo": ci_lo, "ci95_hi": ci_hi}


# ─── A/B Test ────────────────────────────────────────────────

def run_ab_test(df, xg_window: int, features_a: list, xg_feature_names: list,
                min_test: int = MIN_TEST_DEFAULT):
    """Run A/B test for one xG rolling window.

    Model A: baseline features only.
    Model B: baseline + xG features.
    Same train/test split, same seeds.
    """
    features_b = features_a + xg_feature_names

    # Drop rows where xG features are NaN (need xG history)
    df_clean = df.dropna(subset=xg_feature_names).copy()
    n_total = len(df_clean)

    if n_total < min_test * 2:
        return {"status": "INSUFFICIENT_DATA", "n_total": n_total,
                "xg_window": xg_window}

    df_train, df_test = temporal_split(df_clean)
    n_train, n_test = len(df_train), len(df_test)

    print(f"\n  === Window={xg_window}: {n_total} matches (train={n_train}, test={n_test}) ===")

    if n_test < min_test // 2:  # Relaxed: at least half of min_test
        return {"status": "INSUFFICIENT_DATA", "n_total": n_total,
                "n_train": n_train, "n_test": n_test, "xg_window": xg_window}

    y_train = df_train["result"].values.astype(int)
    y_test = df_test["result"].values.astype(int)

    results_a = {"brier": [], "logloss": []}
    results_b = {"brier": [], "logloss": []}
    importances_b = []
    per_match_errors_a = []
    per_match_errors_b = []

    for seed_i in range(N_SEEDS):
        seed = seed_i * 42 + 7

        # Model A: baseline only
        X_tr_a = df_train[features_a].values.astype(np.float32)
        X_te_a = df_test[features_a].values.astype(np.float32)
        model_a = train_xgb(X_tr_a, y_train, seed=seed)
        metrics_a = compute_metrics(model_a, X_te_a, y_test)
        results_a["brier"].append(metrics_a["brier"])
        results_a["logloss"].append(metrics_a["logloss"])

        # Model B: baseline + xG
        X_tr_b = df_train[features_b].values.astype(np.float32)
        X_te_b = df_test[features_b].values.astype(np.float32)
        model_b = train_xgb(X_tr_b, y_train, seed=seed)
        metrics_b = compute_metrics(model_b, X_te_b, y_test)
        results_b["brier"].append(metrics_b["brier"])
        results_b["logloss"].append(metrics_b["logloss"])

        # Feature importance for Model B (gain)
        imp = dict(zip(features_b, model_b.feature_importances_.tolist()))
        importances_b.append(imp)

        # Per-match Brier for bootstrap (last seed only for CI)
        if seed_i == N_SEEDS - 1:
            y_onehot = np.eye(3)[y_test]
            prob_a = model_a.predict_proba(X_te_a)
            prob_b = model_b.predict_proba(X_te_b)
            per_match_errors_a = np.sum((prob_a - y_onehot) ** 2, axis=1)
            per_match_errors_b = np.sum((prob_b - y_onehot) ** 2, axis=1)

        print(f"    Seed {seed}: A={metrics_a['brier']:.4f}, B={metrics_b['brier']:.4f}, "
              f"Δ={metrics_b['brier'] - metrics_a['brier']:.4f}")

    # Aggregate
    mean_brier_a = float(np.mean(results_a["brier"]))
    mean_brier_b = float(np.mean(results_b["brier"]))
    mean_logloss_a = float(np.mean(results_a["logloss"]))
    mean_logloss_b = float(np.mean(results_b["logloss"]))
    delta_brier = mean_brier_b - mean_brier_a
    delta_logloss = mean_logloss_b - mean_logloss_a

    # Bootstrap CI
    bootstrap = bootstrap_delta_ci(per_match_errors_a, per_match_errors_b)

    # Average feature importance across seeds
    avg_importance = {}
    for feat in features_b:
        vals = [imp[feat] for imp in importances_b]
        avg_importance[feat] = round(float(np.mean(vals)), 4)

    # xG features ranked by importance
    xg_importance = {f: avg_importance[f] for f in xg_feature_names}
    xg_rank = sorted(xg_importance.items(), key=lambda x: -x[1])

    # Verdict
    ci_excludes_zero = bootstrap["ci95_hi"] < 0 or bootstrap["ci95_lo"] > 0
    if delta_brier <= SUCCESS_THRESHOLD and bootstrap["ci95_hi"] < 0:
        verdict = "SIGNAL"
    elif delta_brier > 0.005:
        verdict = "NOISE"
    elif not ci_excludes_zero:
        verdict = "NEUTRAL"
    else:
        verdict = "MARGINAL"

    return {
        "status": "OK",
        "xg_window": xg_window,
        "n_total": n_total,
        "n_train": n_train,
        "n_test": n_test,
        "model_a": {
            "features": len(features_a),
            "brier_mean": round(mean_brier_a, 5),
            "brier_std": round(float(np.std(results_a["brier"])), 5),
            "logloss_mean": round(mean_logloss_a, 5),
        },
        "model_b": {
            "features": len(features_b),
            "brier_mean": round(mean_brier_b, 5),
            "brier_std": round(float(np.std(results_b["brier"])), 5),
            "logloss_mean": round(mean_logloss_b, 5),
        },
        "delta_brier": round(delta_brier, 5),
        "delta_logloss": round(delta_logloss, 5),
        "bootstrap_ci95": {
            "mean": round(bootstrap["mean"], 5),
            "lo": round(bootstrap["ci95_lo"], 5),
            "hi": round(bootstrap["ci95_hi"], 5),
        },
        "ci_excludes_zero": ci_excludes_zero,
        "xg_feature_importance": dict(xg_rank),
        "all_feature_importance": avg_importance,
        "verdict": verdict,
    }


# ─── Coverage & Effective-N Report ────────────────────────────

def coverage_report(df, league_name: str) -> dict:
    """ABE task 3: Effective N and missingness report per window."""
    report = {"league": league_name, "total_matches": len(df)}

    # Temporal split to show test set coverage
    _, df_test = temporal_split(df)
    report["test_set_size"] = len(df_test)

    for w in XG_WINDOWS:
        xg_cols = [f"xg_for_home_w{w}", f"xg_against_home_w{w}",
                   f"xg_for_away_w{w}", f"xg_against_away_w{w}"]
        n_total_xg = df.dropna(subset=xg_cols).shape[0]
        n_test_xg = df_test.dropna(subset=xg_cols).shape[0]
        report[f"w{w}_total"] = n_total_xg
        report[f"w{w}_total_pct"] = round(100 * n_total_xg / len(df), 1) if len(df) > 0 else 0
        report[f"w{w}_test"] = n_test_xg
        report[f"w{w}_test_pct"] = round(100 * n_test_xg / len(df_test), 1) if len(df_test) > 0 else 0

    # Date range
    report["date_min"] = str(df["date"].min())[:10]
    report["date_max"] = str(df["date"].max())[:10]

    return report


# ─── Main ────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="xG Signal Ablation Test (ABE P0)")
    parser.add_argument("--extract", action="store_true",
                        help="Re-extract data from DB (otherwise use cached CSV)")
    parser.add_argument("--league", default="argentina", choices=list(LEAGUE_CONFIGS.keys()),
                        help="League to test (default: argentina)")
    parser.add_argument("--no-shots", action="store_true",
                        help="ABE task 2: redundancy test — baseline WITHOUT shots + xG")
    parser.add_argument("--all-leagues", action="store_true",
                        help="Run all leagues and produce combined ABE report")
    args = parser.parse_args()

    output_dir = Path("scripts/output")
    output_dir.mkdir(parents=True, exist_ok=True)

    leagues_to_run = list(LEAGUE_CONFIGS.keys()) if args.all_leagues else [args.league]

    combined_results = {}

    for league_key in leagues_to_run:
        league_config = LEAGUE_CONFIGS[league_key]
        min_test = league_config["min_test"]

        dataset_path = output_dir / f"xg_signal_test_dataset_{league_key}.csv"
        results_path = output_dir / f"xg_signal_test_results_{league_key}.json"

        print(f"\n{'='*60}")
        print(f"=== xG Signal Test: {league_config['name']} ===")
        print(f"{'='*60}")

        # 1. Extract or load data
        print("\n--- Phase 1: Data ---")
        if args.extract or not dataset_path.exists():
            print("  Extracting from DB...")
            df = extract_league_data(league_config, str(dataset_path))
        else:
            print(f"  Loading cached dataset: {dataset_path}")
            df = pd.read_csv(dataset_path, parse_dates=["date"])

        if df.empty or len(df) < 30:
            print(f"  [SKIP] Insufficient data: {len(df)} rows")
            combined_results[league_key] = {"status": "INSUFFICIENT_DATA", "n": len(df)}
            continue

        # Data summary
        print(f"\n  Dataset summary:")
        print(f"    Matches: {len(df)}")
        print(f"    Date range: {df['date'].min()} to {df['date'].max()}")
        dist = df["result"].value_counts().sort_index()
        print(f"    Results: Home={dist.get(0,0)}, Draw={dist.get(1,0)}, Away={dist.get(2,0)}")

        # ABE task 3: Coverage report
        cov = coverage_report(df, league_config["name"])
        print(f"\n  Coverage (effective N):")
        for w in XG_WINDOWS:
            print(f"    Window={w}: total={cov[f'w{w}_total']}/{cov['total_matches']} "
                  f"({cov[f'w{w}_total_pct']}%), "
                  f"test={cov[f'w{w}_test']}/{cov['test_set_size']} "
                  f"({cov[f'w{w}_test_pct']}%)")

        # 2. Standard A/B tests (baseline vs baseline+xG)
        print("\n--- Phase 2: Standard A/B Tests (baseline vs baseline+xG) ---")
        league_results = {
            "meta": {
                "script": "xg_signal_test.py",
                "timestamp": datetime.utcnow().isoformat(),
                "league": league_config["name"],
                "source": league_config.get("source", "fotmob"),
                "data_range": f"{df['date'].min()} to {df['date'].max()}",
                "n_matches": len(df),
                "n_seeds": N_SEEDS,
                "n_bootstrap": N_BOOTSTRAP,
                "success_threshold": SUCCESS_THRESHOLD,
            },
            "coverage": cov,
            "standard_tests": {},
            "no_shots_tests": {},
        }

        for w in XG_WINDOWS:
            xg_cols = [f"xg_for_home_w{w}", f"xg_against_home_w{w}",
                       f"xg_for_away_w{w}", f"xg_against_away_w{w}"]
            result = run_ab_test(df, xg_window=w,
                                 features_a=BASELINE_FEATURES,
                                 xg_feature_names=xg_cols,
                                 min_test=min_test)
            league_results["standard_tests"][f"window_{w}"] = result

        # 3. ABE task 2: Redundancy test (no shots baseline vs no shots + xG)
        if args.no_shots or args.all_leagues:
            print("\n--- Phase 3: Redundancy Test (no-shots baseline vs no-shots+xG) ---")
            for w in XG_WINDOWS:
                xg_cols = [f"xg_for_home_w{w}", f"xg_against_home_w{w}",
                           f"xg_for_away_w{w}", f"xg_against_away_w{w}"]
                result = run_ab_test(df, xg_window=w,
                                     features_a=BASELINE_NO_SHOTS,
                                     xg_feature_names=xg_cols,
                                     min_test=min_test)
                league_results["no_shots_tests"][f"window_{w}"] = result

        # 4. Summary for this league
        print(f"\n{'='*60}")
        print(f"RESULTS: {league_config['name']}")
        print(f"{'='*60}")

        std_verdicts = []
        for w in XG_WINDOWS:
            key = f"window_{w}"
            r = league_results["standard_tests"][key]
            if r["status"] != "OK":
                print(f"\n  [Standard] Window={w}: {r['status']} (n={r.get('n_total', 0)})")
                std_verdicts.append(r["status"])
                continue

            print(f"\n  [Standard] Window={w} (n_test={r['n_test']}):")
            print(f"    A (14 baseline):  Brier={r['model_a']['brier_mean']:.5f}")
            print(f"    B (14+4 xG):      Brier={r['model_b']['brier_mean']:.5f}")
            ci = r["bootstrap_ci95"]
            print(f"    ΔBrier={r['delta_brier']:+.5f}  CI95=[{ci['lo']:+.5f}, {ci['hi']:+.5f}]  "
                  f"→ {r['verdict']}")
            std_verdicts.append(r["verdict"])

        ns_verdicts = []
        if league_results["no_shots_tests"]:
            for w in XG_WINDOWS:
                key = f"window_{w}"
                r = league_results["no_shots_tests"][key]
                if r["status"] != "OK":
                    print(f"\n  [No-shots] Window={w}: {r['status']} (n={r.get('n_total', 0)})")
                    ns_verdicts.append(r["status"])
                    continue

                print(f"\n  [No-shots] Window={w} (n_test={r['n_test']}):")
                print(f"    A (12 no-shots):  Brier={r['model_a']['brier_mean']:.5f}")
                print(f"    B (12+4 xG):      Brier={r['model_b']['brier_mean']:.5f}")
                ci = r["bootstrap_ci95"]
                print(f"    ΔBrier={r['delta_brier']:+.5f}  CI95=[{ci['lo']:+.5f}, {ci['hi']:+.5f}]  "
                      f"→ {r['verdict']}")
                ns_verdicts.append(r["verdict"])

        # Global verdict for this league
        all_v = std_verdicts + ns_verdicts
        if "SIGNAL" in all_v:
            gv = "SIGNAL"
        elif all(v in ("NOISE", "INSUFFICIENT_DATA") for v in all_v):
            gv = "NOISE"
        elif "MARGINAL" in all_v:
            gv = "MARGINAL"
        else:
            gv = "NEUTRAL"

        league_results["global_verdict"] = gv
        league_results["std_verdicts"] = dict(zip([f"w{w}" for w in XG_WINDOWS], std_verdicts))
        league_results["ns_verdicts"] = dict(zip([f"w{w}" for w in XG_WINDOWS], ns_verdicts)) if ns_verdicts else {}

        print(f"\n  GLOBAL VERDICT: {gv}")
        print(f"  Standard: {league_results['std_verdicts']}")
        if ns_verdicts:
            print(f"  No-shots: {league_results['ns_verdicts']}")

        # Save per-league
        with open(results_path, "w") as f:
            json.dump(league_results, f, indent=2, default=str)
        print(f"  Saved to {results_path}")

        combined_results[league_key] = league_results

    # ─── Combined ABE Report ─────────────────────────────────
    if len(leagues_to_run) > 1:
        print(f"\n\n{'='*70}")
        print("ABE COMBINED REPORT: xG Signal Diagnostic")
        print(f"{'='*70}")

        # Header
        print(f"\n{'Liga':<20} {'Source':<10} {'N':<6} {'n_test':<7} "
              f"{'Cov w5':<8} {'Cov w10':<8} "
              f"{'ΔBrier w5':<12} {'CI95 w5':<22} {'Verd w5':<10} "
              f"{'ΔBr ns w5':<12} {'CI95 ns w5':<22} {'Verd ns':<10}")
        print("-" * 155)

        for lk in leagues_to_run:
            lr = combined_results.get(lk, {})
            if isinstance(lr, dict) and lr.get("status") == "INSUFFICIENT_DATA":
                print(f"{LEAGUE_CONFIGS[lk]['name']:<20} {'—':<10} {lr.get('n',0):<6} {'—':<7} "
                      f"{'INSUFFICIENT DATA'}")
                continue
            if not isinstance(lr, dict) or "coverage" not in lr:
                continue

            cov = lr["coverage"]
            src = lr["meta"]["source"]
            n = cov["total_matches"]

            # Standard w5
            sw5 = lr["standard_tests"].get("window_5", {})
            if sw5.get("status") == "OK":
                sw5_ntest = sw5["n_test"]
                sw5_delta = f"{sw5['delta_brier']:+.5f}"
                sw5_ci = f"[{sw5['bootstrap_ci95']['lo']:+.5f}, {sw5['bootstrap_ci95']['hi']:+.5f}]"
                sw5_v = sw5["verdict"]
            else:
                sw5_ntest = sw5.get("n_test", "—")
                sw5_delta = "—"
                sw5_ci = "—"
                sw5_v = sw5.get("status", "—")

            # No-shots w5
            ns5 = lr.get("no_shots_tests", {}).get("window_5", {})
            if ns5.get("status") == "OK":
                ns5_delta = f"{ns5['delta_brier']:+.5f}"
                ns5_ci = f"[{ns5['bootstrap_ci95']['lo']:+.5f}, {ns5['bootstrap_ci95']['hi']:+.5f}]"
                ns5_v = ns5["verdict"]
            else:
                ns5_delta = "—"
                ns5_ci = "—"
                ns5_v = ns5.get("status", "—") if ns5 else "—"

            cov5 = f"{cov.get('w5_test_pct', '—')}%"
            cov10 = f"{cov.get('w10_test_pct', '—')}%"

            print(f"{LEAGUE_CONFIGS[lk]['name']:<20} {src:<10} {n:<6} {sw5_ntest:<7} "
                  f"{cov5:<8} {cov10:<8} "
                  f"{sw5_delta:<12} {sw5_ci:<22} {sw5_v:<10} "
                  f"{ns5_delta:<12} {ns5_ci:<22} {ns5_v:<10}")

        # Save combined
        combined_path = output_dir / "xg_signal_test_combined.json"
        with open(combined_path, "w") as f:
            json.dump(combined_results, f, indent=2, default=str)
        print(f"\n  Combined results saved to {combined_path}")


if __name__ == "__main__":
    main()
