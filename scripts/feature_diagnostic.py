#!/usr/bin/env python3
"""
Feature Signal/Noise Diagnostic
================================
ATI-approved diagnostic for FutbolStats ML features.
Determines which features add signal vs noise to the XGBoost model.

P0 Guardrails (ATI):
- Metrics: Brier/LogLoss primary (not accuracy)
- Split: Temporal (anti-leakage)
- Uncertainty: Bootstrap CI, >=3 seeds
- Groups: Correlated feature families
- PIT-safe: Features computed pre-KO via FeatureEngineer
- Canary: Random noise features as baseline
- Gates: N_test >= 500 for global verdict

Usage:
  source .env
  python scripts/feature_diagnostic.py                      # uses cached dataset
  python scripts/feature_diagnostic.py --extract            # extract fresh from DB
  python scripts/feature_diagnostic.py --model-version v1.0.0
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
from sklearn.inspection import permutation_importance
from sklearn.metrics import log_loss

# Project imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ─── Configuration ───────────────────────────────────────────

# SSOT: Features per model version (from metadata JSON + engine.py)
FEATURES_V100 = [
    "home_goals_scored_avg", "home_goals_conceded_avg",
    "home_shots_avg", "home_corners_avg",
    "home_rest_days", "home_matches_played",
    "away_goals_scored_avg", "away_goals_conceded_avg",
    "away_shots_avg", "away_corners_avg",
    "away_rest_days", "away_matches_played",
    "goal_diff_avg", "rest_diff",
]

FEATURES_V101 = FEATURES_V100 + [
    "abs_attack_diff", "abs_defense_diff", "abs_strength_gap",
]

# Feature families for group permutation (ATI P0: correlated features)
FEATURE_FAMILIES = {
    "home_attack": ["home_goals_scored_avg", "home_shots_avg"],
    "home_defense": ["home_goals_conceded_avg"],
    "home_set_pieces": ["home_corners_avg"],
    "home_fitness": ["home_rest_days", "home_matches_played"],
    "away_attack": ["away_goals_scored_avg", "away_shots_avg"],
    "away_defense": ["away_goals_conceded_avg"],
    "away_set_pieces": ["away_corners_avg"],
    "away_fitness": ["away_rest_days", "away_matches_played"],
    "derived_diff": ["goal_diff_avg", "rest_diff"],
    "competitiveness": ["abs_attack_diff", "abs_defense_diff", "abs_strength_gap"],
}

# Production hyperparameters (from engine.py:466)
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
N_PERMUTATION_REPEATS = 30
DRAW_WEIGHT = 1.5
MIN_TEST_GLOBAL = 500
MIN_TEST_SUBSET = 200


# ─── Data Extraction (SQL + pandas, no ORM) ─────────────────

ROLLING_WINDOW = 10
TIME_DECAY_LAMBDA = 0.01


def _extract_via_sql(league_only: bool = True, output_path: str = None):
    """Extract PIT-safe training data using raw SQL + pandas rolling averages.

    Replicates FeatureEngineer logic without async ORM issues.
    """
    import psycopg2
    from app.config import get_settings
    settings = get_settings()

    db_url = settings.DATABASE_URL
    # psycopg2 needs postgresql:// not postgres://
    if db_url.startswith("postgres://"):
        db_url = db_url.replace("postgres://", "postgresql://", 1)

    conn = psycopg2.connect(db_url)

    # 1. Get league IDs (league-only filter)
    league_filter = ""
    if league_only:
        with conn.cursor() as cur:
            cur.execute("SELECT league_id FROM admin_leagues WHERE kind = 'league' AND is_active = true")
            league_ids = [r[0] for r in cur.fetchall()]
            print(f"  League-only mode: {len(league_ids)} leagues")
            league_filter = f"AND m.league_id IN ({','.join(str(x) for x in league_ids)})"

    # 2. Query all FT matches with stats
    query = f"""
        SELECT m.id AS match_id, m.date, m.league_id,
               m.home_team_id, m.away_team_id,
               m.home_goals, m.away_goals,
               m.stats, m.match_weight,
               m.odds_home, m.odds_draw, m.odds_away
        FROM matches m
        WHERE m.status = 'FT'
          AND m.home_goals IS NOT NULL
          AND m.away_goals IS NOT NULL
          AND m.tainted = false
          {league_filter}
        ORDER BY m.date
    """
    print("  Querying matches...")
    matches = pd.read_sql(query, conn)
    conn.close()
    print(f"  Raw matches: {len(matches)}")

    if matches.empty:
        print("  [ERROR] No matches found")
        return pd.DataFrame()

    # 3. Flatten stats JSON → shots, corners per side
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

    # 4. Build team-level rolling features
    # Create team-centric rows: each match generates 2 rows (home perspective, away perspective)
    print("  Computing rolling features...")
    home_rows = matches[["match_id", "date", "home_team_id", "home_goals", "away_goals",
                          "home_shots", "home_corners", "match_weight"]].copy()
    home_rows.columns = ["match_id", "date", "team_id", "goals_scored", "goals_conceded",
                          "shots", "corners", "match_weight"]

    away_rows = matches[["match_id", "date", "away_team_id", "away_goals", "home_goals",
                          "away_shots", "away_corners", "match_weight"]].copy()
    away_rows.columns = ["match_id", "date", "team_id", "goals_scored", "goals_conceded",
                          "shots", "corners", "match_weight"]

    team_matches = pd.concat([home_rows, away_rows]).sort_values(["team_id", "date"])

    # Compute exponential time decay per-match within each team
    def compute_team_rolling(group):
        """Compute weighted rolling averages for one team."""
        group = group.sort_values("date")
        results = []
        history = []  # list of dicts

        for _, row in group.iterrows():
            # Features use ONLY previous matches (PIT-safe)
            if len(history) > 0:
                window = history[-ROLLING_WINDOW:]  # last N
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
                    goals_scored_avg = 1.0
                    goals_conceded_avg = 1.0
                    shots_avg = 10.0
                    corners_avg = 4.0

                rest_days = (ref_date - history[-1]["date"]).days
                matches_played = len(history)
            else:
                goals_scored_avg = 1.0
                goals_conceded_avg = 1.0
                shots_avg = 10.0
                corners_avg = 4.0
                rest_days = 30
                matches_played = 0

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

    # 5. Merge back to matches: home features + away features
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

    # 6. Build final dataset
    df = matches[["match_id", "date", "league_id", "home_team_id", "away_team_id",
                   "home_goals", "away_goals", "odds_home", "odds_draw", "odds_away"]].copy()
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

    print(f"  Final dataset: {len(df)} rows, {len(df.columns)} columns")

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
    """Compute Brier + LogLoss + Accuracy on data."""
    y_prob = model.predict_proba(X)
    return {
        "brier": multiclass_brier(y, y_prob),
        "logloss": log_loss(y, y_prob, labels=[0, 1, 2]),
        "accuracy": float(np.mean(model.predict(X) == y)),
    }


# ─── Temporal Split ──────────────────────────────────────────

def temporal_split(df, test_fraction=0.2):
    """Split by date (not random). ATI P0: anti-leakage."""
    df_sorted = df.sort_values("date").reset_index(drop=True)
    split_idx = int(len(df_sorted) * (1 - test_fraction))
    return df_sorted.iloc[:split_idx].copy(), df_sorted.iloc[split_idx:].copy()


# ─── Train XGBoost ───────────────────────────────────────────

def train_xgb(X_train, y_train, seed=42, draw_weight=DRAW_WEIGHT):
    """Train XGBoost with production-aligned hyperparams."""
    params = {**PROD_HYPERPARAMS, "random_state": seed}
    model = xgb.XGBClassifier(**params)

    # Sample weighting for draws (production parity)
    sample_weight = np.ones(len(y_train), dtype=np.float32)
    sample_weight[y_train == 1] = draw_weight

    model.fit(X_train, y_train, sample_weight=sample_weight, verbose=False)
    return model


# ─── 1. Native XGBoost Gain ─────────────────────────────────

def xgb_gain_importance(model, feature_names):
    """Extract native XGBoost gain importance."""
    return dict(zip(feature_names, model.feature_importances_.tolist()))


# ─── 2. Permutation Importance (Brier) ──────────────────────

def run_permutation_importance(model, X_test, y_test, feature_names,
                                n_repeats=30, seed=42):
    """Permutation importance using Brier as scoring metric (ATI P0)."""
    def brier_scorer(estimator, X, y):
        y_prob = estimator.predict_proba(X)
        return -multiclass_brier(y, y_prob)  # Negative: sklearn maximizes

    result = permutation_importance(
        model, X_test, y_test,
        scoring=brier_scorer,
        n_repeats=n_repeats,
        random_state=seed,
        n_jobs=-1,
    )

    return {
        feat: {
            "mean": float(result.importances_mean[i]),
            "std": float(result.importances_std[i]),
        }
        for i, feat in enumerate(feature_names)
    }


# ─── 3. Group Permutation ───────────────────────────────────

def group_permutation_importance(model, X_test, y_test, feature_names,
                                  families, n_repeats=10, seed=42):
    """Permute entire feature families together (ATI P0: correlated features)."""
    rng = np.random.RandomState(seed)
    baseline_brier = multiclass_brier(y_test, model.predict_proba(X_test))

    results = {}
    for family_name, family_features in families.items():
        col_indices = [feature_names.index(f) for f in family_features
                       if f in feature_names]
        if not col_indices:
            continue

        deltas = []
        for _ in range(n_repeats):
            X_perm = X_test.copy()
            perm_idx = rng.permutation(len(X_perm))
            X_perm[:, col_indices] = X_perm[perm_idx][:, col_indices]
            perm_brier = multiclass_brier(y_test, model.predict_proba(X_perm))
            deltas.append(perm_brier - baseline_brier)

        results[family_name] = {
            "mean_delta_brier": float(np.mean(deltas)),
            "std": float(np.std(deltas)),
            "features": family_features,
        }

    return results


# ─── 4. Leave-One-Out Ablation ───────────────────────────────

def ablation_analysis(df_train, df_test, feature_names, n_seeds=3):
    """Train N models each missing one feature. Compare Brier to baseline.

    ATI P0: >=3 seeds for stability, Brier as metric.
    Positive delta = removing feature worsens model = SIGNAL.
    Negative delta = removing feature improves model = NOISE.
    """
    results = {}

    # Baseline (all features)
    baseline_briers = []
    for seed_i in range(n_seeds):
        seed = seed_i * 42 + 7
        X_tr = df_train[feature_names].values
        y_tr = df_train["result"].values
        X_te = df_test[feature_names].values
        y_te = df_test["result"].values

        model = train_xgb(X_tr, y_tr, seed=seed)
        metrics = compute_metrics(model, X_te, y_te)
        baseline_briers.append(metrics["brier"])

    baseline_mean = float(np.mean(baseline_briers))
    baseline_std = float(np.std(baseline_briers))
    results["__baseline__"] = {
        "brier_mean": baseline_mean,
        "brier_std": baseline_std,
    }

    # Ablation per feature
    for feat in feature_names:
        remaining = [f for f in feature_names if f != feat]
        feat_briers = []

        for seed_i in range(n_seeds):
            seed = seed_i * 42 + 7
            X_tr = df_train[remaining].values
            y_tr = df_train["result"].values
            X_te = df_test[remaining].values
            y_te = df_test["result"].values

            model = train_xgb(X_tr, y_tr, seed=seed)
            metrics = compute_metrics(model, X_te, y_te)
            feat_briers.append(metrics["brier"])

        delta = float(np.mean(feat_briers) - baseline_mean)
        results[feat] = {
            "brier_mean": float(np.mean(feat_briers)),
            "brier_std": float(np.std(feat_briers)),
            "delta_brier": delta,
        }

    return results


# ─── 5. Multi-seed Stability ────────────────────────────────

def stability_check(X_train, y_train, X_test, y_test, feature_names, n_seeds=3):
    """Check permutation importance stability across seeds (ATI P0)."""
    results = {}
    for feat in feature_names:
        perm_means = []
        for seed_i in range(n_seeds):
            seed = seed_i * 42 + 7
            model = train_xgb(X_train, y_train, seed=seed)
            perm = run_permutation_importance(
                model, X_test, y_test, feature_names,
                n_repeats=10, seed=seed,
            )
            perm_means.append(perm[feat]["mean"])

        results[feat] = {
            "mean": float(np.mean(perm_means)),
            "std_across_seeds": float(np.std(perm_means)),
        }

    return results


# ─── Canary Features ─────────────────────────────────────────

def add_canary_features(df, n_canaries=3, seed=42):
    """Add random noise features as baseline (ATI P1)."""
    rng = np.random.RandomState(seed)
    canary_names = []
    for i in range(n_canaries):
        name = f"__canary_{i}__"
        df[name] = rng.randn(len(df))
        canary_names.append(name)
    return df, canary_names


# ─── Verdict Logic ───────────────────────────────────────────

def assign_verdict(row, max_canary_perm):
    """Assign SIGNAL/NEUTRAL/NOISE/UNSTABLE verdict.

    Logic:
    - NOISE: ablation delta < -0.001 (model improves without feature)
    - UNSTABLE: stability std > 0.005 (inconsistent across seeds)
    - SIGNAL: perm importance > canary AND ablation delta > 0
    - NEUTRAL: everything else
    """
    if row.get("is_canary"):
        return "CANARY"

    ablation_delta = row.get("ablation_delta", 0)
    perm_mean = row.get("perm_mean", 0)
    stability_std = row.get("stability_std", 0)

    if ablation_delta < -0.001:
        return "NOISE"
    if stability_std > 0.005:
        return "UNSTABLE"
    if perm_mean > max_canary_perm and ablation_delta > 0:
        return "SIGNAL"
    return "NEUTRAL"


# ─── Main Diagnostic ─────────────────────────────────────────

def run_diagnostic(df, model_version="v1.0.1"):
    """Run full ATI-approved feature diagnostic."""

    # 1. SSOT: Select features for model version
    if model_version == "v1.0.0":
        features = FEATURES_V100.copy()
    else:
        features = FEATURES_V101.copy()

    print(f"\n{'=' * 70}")
    print(f"  FEATURE SIGNAL/NOISE DIAGNOSTIC")
    print(f"  Model: {model_version} | Features: {len(features)}")
    print(f"{'=' * 70}")

    # 2. Prepare data
    df = df.copy()
    if "result" not in df.columns:
        print("  [ERROR] Column 'result' not found in dataset")
        return None

    # Verify all features exist
    missing = [f for f in features if f not in df.columns]
    if missing:
        print(f"  [ERROR] Missing features in dataset: {missing}")
        return None

    # 3. Add canary features (ATI P1)
    df, canary_names = add_canary_features(df)
    all_features = features + canary_names

    # 4. Drop rows with missing feature values
    before = len(df)
    df = df.dropna(subset=features)
    if len(df) < before:
        print(f"  [WARN] Dropped {before - len(df)} rows with NaN features")

    # 5. Temporal split (ATI P0: anti-leakage)
    df_train, df_test = temporal_split(df, test_fraction=0.2)

    print(f"\n  Dataset:    {len(df):,} total ({len(df_train):,} train, {len(df_test):,} test)")
    print(f"  Date range: {df['date'].min()} -> {df['date'].max()}")
    print(f"  Split at:   {df_test['date'].min()}")

    # Gate check (ATI P0)
    if len(df_test) < MIN_TEST_GLOBAL:
        confidence = "LOW"
        print(f"\n  WARNING: N_test={len(df_test)} < {MIN_TEST_GLOBAL} -> LOW CONFIDENCE")
    else:
        confidence = "HIGH"

    # Label distribution
    for name, split in [("Train", df_train), ("Test", df_test)]:
        dist = split["result"].value_counts().sort_index()
        pct = (dist / len(split) * 100).round(1)
        print(f"  {name} labels: H={pct.get(0, 0):.1f}% D={pct.get(1, 0):.1f}% A={pct.get(2, 0):.1f}%")

    # Prepare arrays
    X_train = df_train[all_features].values.astype(np.float32)
    y_train = df_train["result"].values.astype(int)
    X_test = df_test[all_features].values.astype(np.float32)
    y_test = df_test["result"].values.astype(int)

    # 6. Train baseline model
    print(f"\n  Training baseline model...")
    model = train_xgb(X_train, y_train, seed=42)
    baseline = compute_metrics(model, X_test, y_test)
    print(f"  Baseline: Brier={baseline['brier']:.4f} "
          f"LogLoss={baseline['logloss']:.4f} "
          f"Acc={baseline['accuracy']:.3f}")

    # ─── DIAGNOSTIC 1: XGBoost Gain ─────────────────────────
    print(f"\n  [1/5] XGBoost native gain importance...")
    gain = xgb_gain_importance(model, all_features)

    # ─── DIAGNOSTIC 2: Permutation Importance ────────────────
    print(f"  [2/5] Permutation importance (Brier, {N_PERMUTATION_REPEATS} repeats)...")
    perm = run_permutation_importance(
        model, X_test, y_test, all_features,
        n_repeats=N_PERMUTATION_REPEATS, seed=42,
    )

    # ─── DIAGNOSTIC 3: Group Permutation ─────────────────────
    print(f"  [3/5] Group permutation (families)...")
    active_families = {
        k: [f for f in v if f in features]
        for k, v in FEATURE_FAMILIES.items()
    }
    active_families = {k: v for k, v in active_families.items() if v}
    group_perm = group_permutation_importance(
        model, X_test, y_test, all_features, active_families,
    )

    # ─── DIAGNOSTIC 4: Ablation ──────────────────────────────
    print(f"  [4/5] Leave-One-Out ablation ({N_SEEDS} seeds x {len(features)} features)...")
    ablation = ablation_analysis(df_train, df_test, features, n_seeds=N_SEEDS)

    # ─── DIAGNOSTIC 5: Multi-seed Stability ──────────────────
    print(f"  [5/5] Multi-seed stability ({N_SEEDS} seeds)...")
    stability = stability_check(
        X_train, y_train, X_test, y_test, all_features, n_seeds=N_SEEDS,
    )

    # ─── Compile Results ─────────────────────────────────────
    baseline_brier = ablation["__baseline__"]["brier_mean"]

    # Max canary permutation importance (ATI P1: noise floor)
    canary_perm_vals = [perm[c]["mean"] for c in canary_names if c in perm]
    max_canary_perm = max(canary_perm_vals) if canary_perm_vals else 0

    rows = []
    for feat in all_features:
        is_canary = feat.startswith("__canary_")
        row = {
            "feature": feat,
            "is_canary": is_canary,
            "xgb_gain": gain.get(feat, 0),
            "perm_mean": perm[feat]["mean"] if feat in perm else 0,
            "perm_std": perm[feat]["std"] if feat in perm else 0,
            "ablation_delta": ablation[feat]["delta_brier"] if feat in ablation else None,
            "ablation_std": ablation[feat]["brier_std"] if feat in ablation else None,
            "stability_std": stability[feat]["std_across_seeds"] if feat in stability else None,
        }
        row["verdict"] = assign_verdict(row, max_canary_perm)
        rows.append(row)

    results_df = pd.DataFrame(rows)
    results_df = results_df.sort_values("perm_mean", ascending=False)

    # ─── Print Feature Table ─────────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"  RESULTS (sorted by permutation importance)")
    print(f"{'=' * 70}")
    print(f"\n  {'Feature':<28} {'Gain':>7} {'Perm':>9} {'Ablat':>10} "
          f"{'Stab':>7} {'Verdict':>10}")
    print(f"  {'─' * 28} {'─' * 7} {'─' * 9} {'─' * 10} {'─' * 7} {'─' * 10}")

    for _, r in results_df.iterrows():
        name = r["feature"][:28]
        g = f"{r['xgb_gain']:.4f}"
        p = f"{r['perm_mean']:+.5f}"
        a = f"{r['ablation_delta']:+.5f}" if r["ablation_delta"] is not None else "      n/a"
        s = f"{r['stability_std']:.5f}" if r["stability_std"] is not None else "    n/a"
        v = r["verdict"]
        print(f"  {name:<28} {g:>7} {p:>9} {a:>10} {s:>7} {v:>10}")

    # ─── Print Group Permutation ─────────────────────────────
    print(f"\n  GROUP PERMUTATION (Feature Families)")
    print(f"  {'Family':<24} {'Delta Brier':>12} {'+-':>8} {'Members'}")
    print(f"  {'─' * 24} {'─' * 12} {'─' * 8} {'─' * 40}")

    for family, data in sorted(group_perm.items(),
                                 key=lambda x: -x[1]["mean_delta_brier"]):
        delta = f"{data['mean_delta_brier']:+.6f}"
        std = f"{data['std']:.6f}"
        members = ", ".join(data["features"])
        print(f"  {family:<24} {delta:>12} {std:>8} {members}")

    # ─── Print Canary Check ──────────────────────────────────
    print(f"\n  CANARY CHECK")
    print(f"  Max canary perm importance: {max_canary_perm:+.6f}")
    real = results_df[~results_df["is_canary"]]
    below = real[real["perm_mean"] <= max_canary_perm]
    if len(below) > 0:
        print(f"  WARNING: {len(below)} features BELOW canary noise floor:")
        for _, r in below.iterrows():
            print(f"    - {r['feature']} (perm={r['perm_mean']:+.6f})")
    else:
        print(f"  OK: All real features above canary noise floor")

    # ─── Print Summary ───────────────────────────────────────
    verdicts = results_df[~results_df["is_canary"]]["verdict"].value_counts()
    print(f"\n  SUMMARY")
    print(f"  Confidence: {confidence} (N_test={len(df_test)})")
    print(f"  Baseline Brier: {baseline_brier:.4f}")
    for v in ["SIGNAL", "NEUTRAL", "NOISE", "UNSTABLE"]:
        print(f"  {v}: {verdicts.get(v, 0)}")

    # ─── Save Results ────────────────────────────────────────
    output_dir = Path("scripts/output")
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"feature_diagnostic_{model_version}_{timestamp}.json"

    results_json = {
        "metadata": {
            "model_version": model_version,
            "n_features": len(features),
            "n_canaries": len(canary_names),
            "n_train": len(df_train),
            "n_test": len(df_test),
            "confidence": confidence,
            "baseline_brier": baseline_brier,
            "baseline_logloss": baseline["logloss"],
            "baseline_accuracy": baseline["accuracy"],
            "date_range": [str(df["date"].min()), str(df["date"].max())],
            "split_date": str(df_test["date"].min()),
            "timestamp": timestamp,
            "max_canary_perm": max_canary_perm,
            "n_seeds": N_SEEDS,
            "n_perm_repeats": N_PERMUTATION_REPEATS,
            "draw_weight": DRAW_WEIGHT,
        },
        "features": json.loads(
            results_df.drop(columns=["is_canary"]).to_json(orient="records")
        ),
        "group_permutation": group_perm,
        "ablation": {k: v for k, v in ablation.items() if k != "__baseline__"},
        "ablation_baseline": ablation["__baseline__"],
    }

    with open(output_file, "w") as f:
        json.dump(results_json, f, indent=2, default=str)
    print(f"\n  Results saved to: {output_file}")

    return results_json


# ─── CLI ─────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Feature Signal/Noise Diagnostic (ATI-approved)"
    )
    parser.add_argument(
        "--extract", action="store_true",
        help="Extract fresh dataset from DB (slow, ~2 min)",
    )
    parser.add_argument(
        "--dataset", type=str,
        default="scripts/output/training_dataset.csv",
        help="Path to cached parquet dataset",
    )
    parser.add_argument(
        "--model-version", type=str, default="v1.0.1",
        choices=["v1.0.0", "v1.0.1"],
        help="Model version to diagnose",
    )
    parser.add_argument(
        "--league-only", action="store_true", default=True,
        help="Use only league matches (production parity)",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset)

    print(f"\n  FEATURE DIAGNOSTIC")
    print(f"  {'=' * 50}")

    if args.extract or not dataset_path.exists():
        print(f"\n  [PHASE 1] Extracting training data from DB...")
        df = _extract_via_sql(
            league_only=args.league_only,
            output_path=str(dataset_path),
        )
    else:
        print(f"\n  [PHASE 1] Loading cached dataset: {dataset_path}")
        df = pd.read_csv(dataset_path, parse_dates=["date"])
        print(f"  Loaded {len(df)} rows, {len(df.columns)} columns")

    print(f"\n  [PHASE 2] Running diagnostic...")
    run_diagnostic(df, model_version=args.model_version)


if __name__ == "__main__":
    main()
