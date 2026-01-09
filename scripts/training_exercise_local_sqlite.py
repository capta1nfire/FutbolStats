#!/usr/bin/env python3
"""
Training Exercise - Local SQLite (Offline)

Ejercicio de entrenamiento XGBoost usando SQLite local.
NO toca Railway/Postgres. Solo lectura de SQLite, output en logs/.

Usage:
    python3 scripts/training_exercise_local_sqlite.py \
        --db /Users/inseqio/FutbolStats/futbolstat.db \
        --rolling-window 5 \
        --lambda-decay 0.01 \
        --min-date 2019-01-01
"""

import argparse
import json
import math
import sqlite3
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit

# Feature columns (compatible with XGBoostEngine.FEATURE_COLUMNS)
FEATURE_COLUMNS = [
    "home_goals_scored_avg",
    "home_goals_conceded_avg",
    "home_shots_avg",
    "home_corners_avg",
    "home_rest_days",
    "home_matches_played",
    "away_goals_scored_avg",
    "away_goals_conceded_avg",
    "away_shots_avg",
    "away_corners_avg",
    "away_rest_days",
    "away_matches_played",
    "goal_diff_avg",
    "rest_diff",
]

# Defaults when no history available
DEFAULTS = {
    "goals_scored_avg": 1.0,
    "goals_conceded_avg": 1.0,
    "shots_avg": 10.0,
    "corners_avg": 4.0,
    "rest_days": 30,
    "matches_played": 0,
}


def load_matches(db_path: str, min_date: str = None, max_date: str = None, limit: int = None) -> list[dict]:
    """Load FT matches from SQLite, ordered by date ascending."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    query = """
        SELECT
            id, date, league_id, home_team_id, away_team_id,
            home_goals, away_goals, stats, match_weight, status
        FROM matches
        WHERE status = 'FT'
          AND home_goals IS NOT NULL
          AND away_goals IS NOT NULL
    """
    params = []

    if min_date:
        query += " AND date >= ?"
        params.append(min_date)
    if max_date:
        query += " AND date <= ?"
        params.append(max_date)

    query += " ORDER BY date ASC"

    if limit:
        query += f" LIMIT {int(limit)}"

    cursor.execute(query, params)
    rows = cursor.fetchall()
    conn.close()

    matches = []
    for row in rows:
        match = dict(row)
        # Parse date
        if match["date"]:
            try:
                match["date_parsed"] = datetime.fromisoformat(match["date"].replace("Z", "+00:00").split("+")[0])
            except:
                match["date_parsed"] = None
        else:
            match["date_parsed"] = None

        # Parse stats JSON
        match["stats_parsed"] = parse_stats(match.get("stats"))

        # Ensure match_weight has a value
        if match["match_weight"] is None:
            match["match_weight"] = 1.0

        matches.append(match)

    return matches


def parse_stats(stats_str) -> dict:
    """Parse stats JSON string, handling NULL/'null'/empty cases."""
    if stats_str is None:
        return None
    if isinstance(stats_str, str):
        stats_str = stats_str.strip()
        if stats_str == "" or stats_str.lower() == "null":
            return None
        try:
            return json.loads(stats_str)
        except json.JSONDecodeError:
            return None
    return None


def extract_shots(stats: dict, side: str) -> float:
    """Extract shots from stats dict for home/away side."""
    if not stats:
        return 0.0

    side_stats = stats.get(side, {})
    if not side_stats:
        return 0.0

    # Try different keys
    for key in ["total_shots", "shots_on_goal", "shots"]:
        val = side_stats.get(key)
        if val is not None:
            try:
                return float(val)
            except (ValueError, TypeError):
                pass
    return 0.0


def extract_corners(stats: dict, side: str) -> float:
    """Extract corners from stats dict for home/away side."""
    if not stats:
        return 0.0

    side_stats = stats.get(side, {})
    if not side_stats:
        return 0.0

    for key in ["corner_kicks", "corners"]:
        val = side_stats.get(key)
        if val is not None:
            try:
                return float(val)
            except (ValueError, TypeError):
                pass
    return 0.0


def build_team_history_index(matches: list[dict]) -> dict:
    """
    Build index of matches by team_id for efficient lookup.
    Returns dict: team_id -> list of (date_parsed, match_dict) sorted by date.
    """
    team_history = defaultdict(list)

    for match in matches:
        date_parsed = match["date_parsed"]
        if date_parsed is None:
            continue

        home_id = match["home_team_id"]
        away_id = match["away_team_id"]

        # Add to both teams' history
        team_history[home_id].append((date_parsed, match, "home"))
        team_history[away_id].append((date_parsed, match, "away"))

    # Sort each team's history by date
    for team_id in team_history:
        team_history[team_id].sort(key=lambda x: x[0])

    return team_history


def compute_team_features(
    team_id: int,
    match_date: datetime,
    team_history: dict,
    rolling_window: int,
    lambda_decay: float,
) -> dict:
    """
    Compute rolling features for a team using only matches BEFORE match_date.
    Returns dict with goals_scored_avg, goals_conceded_avg, shots_avg, corners_avg, rest_days, matches_played.
    """
    history = team_history.get(team_id, [])

    # Filter to matches strictly before match_date
    prior_matches = [(d, m, side) for d, m, side in history if d < match_date]

    if not prior_matches:
        return {
            "goals_scored_avg": DEFAULTS["goals_scored_avg"],
            "goals_conceded_avg": DEFAULTS["goals_conceded_avg"],
            "shots_avg": DEFAULTS["shots_avg"],
            "corners_avg": DEFAULTS["corners_avg"],
            "rest_days": DEFAULTS["rest_days"],
            "matches_played": DEFAULTS["matches_played"],
        }

    # Take last N matches (rolling window)
    recent = prior_matches[-rolling_window:]

    # Compute rest_days from most recent match
    last_date = prior_matches[-1][0]
    rest_days = (match_date - last_date).days
    rest_days = min(rest_days, 90)  # Cap at 90

    # Weighted averages
    total_weight = 0.0
    weighted_goals_scored = 0.0
    weighted_goals_conceded = 0.0
    weighted_shots = 0.0
    weighted_corners = 0.0

    for d, m, side in recent:
        days_since = (match_date - d).days
        time_weight = math.exp(-lambda_decay * days_since)
        match_weight = m.get("match_weight", 1.0) or 1.0
        weight = match_weight * time_weight

        if side == "home":
            goals_scored = m["home_goals"]
            goals_conceded = m["away_goals"]
            shots = extract_shots(m["stats_parsed"], "home")
            corners = extract_corners(m["stats_parsed"], "home")
        else:
            goals_scored = m["away_goals"]
            goals_conceded = m["home_goals"]
            shots = extract_shots(m["stats_parsed"], "away")
            corners = extract_corners(m["stats_parsed"], "away")

        weighted_goals_scored += goals_scored * weight
        weighted_goals_conceded += goals_conceded * weight
        weighted_shots += shots * weight
        weighted_corners += corners * weight
        total_weight += weight

    if total_weight > 0:
        goals_scored_avg = weighted_goals_scored / total_weight
        goals_conceded_avg = weighted_goals_conceded / total_weight
        shots_avg = weighted_shots / total_weight
        corners_avg = weighted_corners / total_weight
    else:
        goals_scored_avg = DEFAULTS["goals_scored_avg"]
        goals_conceded_avg = DEFAULTS["goals_conceded_avg"]
        shots_avg = DEFAULTS["shots_avg"]
        corners_avg = DEFAULTS["corners_avg"]

    return {
        "goals_scored_avg": goals_scored_avg,
        "goals_conceded_avg": goals_conceded_avg,
        "shots_avg": shots_avg,
        "corners_avg": corners_avg,
        "rest_days": rest_days,
        "matches_played": len(prior_matches),
    }


def build_dataset(
    matches: list[dict],
    team_history: dict,
    rolling_window: int,
    lambda_decay: float,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """
    Build feature matrix X and target vector y.
    Returns (X, y, metadata) where metadata contains match info for each row.
    """
    X_rows = []
    y_rows = []
    metadata = []

    for match in matches:
        if match["date_parsed"] is None:
            continue

        home_id = match["home_team_id"]
        away_id = match["away_team_id"]
        match_date = match["date_parsed"]

        # Compute features for both teams
        home_feats = compute_team_features(home_id, match_date, team_history, rolling_window, lambda_decay)
        away_feats = compute_team_features(away_id, match_date, team_history, rolling_window, lambda_decay)

        # Build feature vector
        row = [
            home_feats["goals_scored_avg"],
            home_feats["goals_conceded_avg"],
            home_feats["shots_avg"],
            home_feats["corners_avg"],
            home_feats["rest_days"],
            home_feats["matches_played"],
            away_feats["goals_scored_avg"],
            away_feats["goals_conceded_avg"],
            away_feats["shots_avg"],
            away_feats["corners_avg"],
            away_feats["rest_days"],
            away_feats["matches_played"],
            # Derived features
            (home_feats["goals_scored_avg"] - home_feats["goals_conceded_avg"]) -
            (away_feats["goals_scored_avg"] - away_feats["goals_conceded_avg"]),  # goal_diff_avg
            home_feats["rest_days"] - away_feats["rest_days"],  # rest_diff
        ]

        # Target: 0=home win, 1=draw, 2=away win
        home_goals = match["home_goals"]
        away_goals = match["away_goals"]
        if home_goals > away_goals:
            target = 0
        elif home_goals == away_goals:
            target = 1
        else:
            target = 2

        X_rows.append(row)
        y_rows.append(target)
        metadata.append({
            "match_id": match["id"],
            "date": match["date"],
            "league_id": match["league_id"],
            "home_team_id": home_id,
            "away_team_id": away_id,
        })

    return np.array(X_rows), np.array(y_rows), metadata


def multiclass_brier_score(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """
    Compute multiclass Brier score (mean squared error of probabilities).
    Lower is better. Range [0, 2] for 3 classes.
    """
    n_samples = len(y_true)
    n_classes = y_proba.shape[1]

    # One-hot encode y_true
    y_onehot = np.zeros((n_samples, n_classes))
    for i, label in enumerate(y_true):
        y_onehot[i, label] = 1.0

    # Brier = mean of sum of squared differences
    brier = np.mean(np.sum((y_proba - y_onehot) ** 2, axis=1))
    return brier


def train_and_evaluate(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 3,
    holdout_fraction: float = 0.2,
) -> dict:
    """
    Train XGBoost with TimeSeriesSplit CV and holdout evaluation.
    Returns metrics dict.
    """
    n_samples = len(y)
    holdout_size = int(n_samples * holdout_fraction)
    train_size = n_samples - holdout_size

    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # XGBoost parameters (simplified version of production)
    params = {
        "objective": "multi:softprob",
        "num_class": 3,
        "max_depth": 4,
        "learning_rate": 0.05,
        "n_estimators": 100,
        "min_child_weight": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "eval_metric": "mlogloss",
        "verbosity": 0,
    }

    # Cross-validation with TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=n_splits)
    cv_scores = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train)):
        X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
        y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]

        model = xgb.XGBClassifier(**params)
        model.fit(X_fold_train, y_fold_train, verbose=False)

        y_proba = model.predict_proba(X_fold_val)
        brier = multiclass_brier_score(y_fold_val, y_proba)
        cv_scores.append(brier)

    # Train final model on all training data
    final_model = xgb.XGBClassifier(**params)
    final_model.fit(X_train, y_train, verbose=False)

    # Holdout evaluation
    y_test_proba = final_model.predict_proba(X_test)
    brier_test = multiclass_brier_score(y_test, y_test_proba)

    # Baseline: uniform (1/3, 1/3, 1/3)
    uniform_proba = np.ones((len(y_test), 3)) / 3
    brier_uniform = multiclass_brier_score(y_test, uniform_proba)

    # Baseline: historical frequency
    train_counts = np.bincount(y_train, minlength=3)
    train_freq = train_counts / train_counts.sum()
    freq_proba = np.tile(train_freq, (len(y_test), 1))
    brier_freq = multiclass_brier_score(y_test, freq_proba)

    return {
        "cv_brier_scores": cv_scores,
        "cv_brier_avg": float(np.mean(cv_scores)),
        "holdout_brier_test": float(brier_test),
        "baseline_uniform_brier_test": float(brier_uniform),
        "baseline_freq_brier_test": float(brier_freq),
        "train_size": train_size,
        "test_size": holdout_size,
        "train_outcome_distribution": {
            "home_win": int(train_counts[0]),
            "draw": int(train_counts[1]),
            "away_win": int(train_counts[2]),
        },
        "model_params": params,
    }


def compute_data_quality(matches: list[dict]) -> dict:
    """Compute data quality metrics."""
    total = len(matches)
    if total == 0:
        return {"pct_stats_null": 100.0, "pct_with_odds": 0.0}

    stats_null = sum(1 for m in matches if m["stats_parsed"] is None)

    return {
        "pct_stats_null": round(100 * stats_null / total, 2),
    }


def main():
    parser = argparse.ArgumentParser(description="Training Exercise - Local SQLite")
    parser.add_argument("--db", default="/Users/inseqio/FutbolStats/futbolstat.db", help="Path to SQLite database")
    parser.add_argument("--rolling-window", type=int, default=5, help="Rolling window for features")
    parser.add_argument("--lambda-decay", type=float, default=0.01, help="Temporal decay factor")
    parser.add_argument("--min-date", default=None, help="Minimum date (YYYY-MM-DD)")
    parser.add_argument("--max-date", default=None, help="Maximum date (YYYY-MM-DD)")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of matches (for testing)")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"Loading matches from {args.db}...")
    matches = load_matches(args.db, args.min_date, args.max_date, args.limit)
    print(f"Loaded {len(matches)} FT matches")

    if len(matches) < 50:
        print("ERROR: Not enough matches for training (need at least 50)")
        return

    # Date range
    dates = [m["date_parsed"] for m in matches if m["date_parsed"]]
    date_min = min(dates).isoformat() if dates else None
    date_max = max(dates).isoformat() if dates else None
    print(f"Date range: {date_min} to {date_max}")

    # League distribution
    league_counts = defaultdict(int)
    for m in matches:
        league_counts[m["league_id"]] += 1
    top_leagues = sorted(league_counts.items(), key=lambda x: -x[1])[:10]
    print(f"Top leagues: {top_leagues}")

    # Build team history index
    print("Building team history index...")
    team_history = build_team_history_index(matches)
    print(f"Teams indexed: {len(team_history)}")

    # Build dataset
    print("Building feature dataset...")
    X, y, metadata = build_dataset(matches, team_history, args.rolling_window, args.lambda_decay)
    print(f"Dataset shape: X={X.shape}, y={y.shape}")

    # Data quality
    data_quality = compute_data_quality(matches)
    print(f"Data quality: {data_quality}")

    # Train and evaluate
    print("Training and evaluating...")
    metrics = train_and_evaluate(X, y, n_splits=3, holdout_fraction=0.2)

    # Print summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"N matches:        {len(matches)}")
    print(f"Date range:       {date_min} to {date_max}")
    print(f"CV Brier scores:  {[round(s, 4) for s in metrics['cv_brier_scores']]}")
    print(f"CV Brier avg:     {metrics['cv_brier_avg']:.4f}")
    print(f"Holdout Brier:    {metrics['holdout_brier_test']:.4f}")
    print(f"Baseline uniform: {metrics['baseline_uniform_brier_test']:.4f}")
    print(f"Baseline freq:    {metrics['baseline_freq_brier_test']:.4f}")
    print("=" * 60)

    # Skill score (improvement over baseline)
    skill_vs_uniform = 1 - (metrics['holdout_brier_test'] / metrics['baseline_uniform_brier_test'])
    skill_vs_freq = 1 - (metrics['holdout_brier_test'] / metrics['baseline_freq_brier_test'])
    print(f"Skill vs uniform: {skill_vs_uniform:.2%}")
    print(f"Skill vs freq:    {skill_vs_freq:.2%}")

    # Build report
    report = {
        "generated_at": datetime.now().isoformat(),
        "db_path": args.db,
        "n_matches_total": len(matches),
        "date_min": date_min,
        "date_max": date_max,
        "small_sample": len(matches) < 300,
        "league_counts_top10": [{"league_id": lid, "n": n} for lid, n in top_leagues],
        "feature_columns": FEATURE_COLUMNS,
        "training_config": {
            "rolling_window": args.rolling_window,
            "lambda_decay": args.lambda_decay,
            "min_date_filter": args.min_date,
            "max_date_filter": args.max_date,
            "limit": args.limit,
            "cv_splits": 3,
            "holdout_fraction": 0.2,
            "model_params": metrics["model_params"],
        },
        "metrics": {
            "cv_brier_scores": metrics["cv_brier_scores"],
            "cv_brier_avg": metrics["cv_brier_avg"],
            "holdout_brier_test": metrics["holdout_brier_test"],
            "baseline_uniform_brier_test": metrics["baseline_uniform_brier_test"],
            "baseline_freq_brier_test": metrics["baseline_freq_brier_test"],
            "skill_vs_uniform": skill_vs_uniform,
            "skill_vs_freq": skill_vs_freq,
            "train_size": metrics["train_size"],
            "test_size": metrics["test_size"],
            "train_outcome_distribution": metrics["train_outcome_distribution"],
        },
        "data_quality": data_quality,
        "notes": "offline/local; no PIT; no Railway; sqlite3 only",
    }

    # Save JSON report
    logs_dir = Path(__file__).parent.parent / "logs"
    logs_dir.mkdir(exist_ok=True)

    json_path = logs_dir / f"training_exercise_local_sqlite_{timestamp}.json"
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nJSON report saved: {json_path}")

    # Save markdown summary
    md_path = logs_dir / f"training_exercise_local_sqlite_{timestamp}.md"
    md_content = f"""# Training Exercise - Local SQLite

Generated: {report['generated_at']}

## Dataset
- **N matches**: {report['n_matches_total']}
- **Date range**: {report['date_min']} to {report['date_max']}
- **Small sample**: {report['small_sample']}

## Top Leagues
| League ID | N |
|-----------|---|
""" + "\n".join([f"| {l['league_id']} | {l['n']} |" for l in report['league_counts_top10']]) + f"""

## Configuration
- Rolling window: {args.rolling_window}
- Lambda decay: {args.lambda_decay}
- CV splits: 3
- Holdout: 20%

## Results

| Metric | Value |
|--------|-------|
| CV Brier avg | {metrics['cv_brier_avg']:.4f} |
| Holdout Brier | {metrics['holdout_brier_test']:.4f} |
| Baseline uniform | {metrics['baseline_uniform_brier_test']:.4f} |
| Baseline freq | {metrics['baseline_freq_brier_test']:.4f} |
| Skill vs uniform | {skill_vs_uniform:.2%} |
| Skill vs freq | {skill_vs_freq:.2%} |

## Data Quality
- Stats NULL: {data_quality['pct_stats_null']}%

## Notes
{report['notes']}
"""
    with open(md_path, "w") as f:
        f.write(md_content)
    print(f"Markdown report saved: {md_path}")


if __name__ == "__main__":
    main()
