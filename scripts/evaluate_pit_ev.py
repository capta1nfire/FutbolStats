#!/usr/bin/env python3
"""
Evaluate PIT Dataset - ROI/EV Analysis with Confidence Intervals

Analyzes the PIT dataset to calculate:
- Expected Value (EV) per bet type
- Return on Investment (ROI) with confidence intervals
- Kelly criterion sizing recommendations
- Calibration analysis

Usage:
    python3 scripts/evaluate_pit_ev.py
    python3 scripts/evaluate_pit_ev.py --input data/pit_dataset.duckdb
    python3 scripts/evaluate_pit_ev.py --ev-threshold 0.05 --delta-ko-min 10 --delta-ko-max 90
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from typing import Optional

import duckdb
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_pit_dataset(input_path: str) -> duckdb.DuckDBPyConnection:
    """Load PIT dataset from DuckDB file."""
    if not os.path.exists(input_path):
        logger.error(f"Input file not found: {input_path}")
        sys.exit(1)

    return duckdb.connect(input_path, read_only=True)


def calculate_confidence_interval(
    successes: int,
    trials: int,
    confidence: float = 0.95
) -> tuple[float, float]:
    """
    Calculate Wilson score confidence interval for proportions.
    More accurate than normal approximation for small samples.
    """
    if trials == 0:
        return (0.0, 0.0)

    from scipy import stats

    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    p_hat = successes / trials

    denominator = 1 + z**2 / trials
    center = (p_hat + z**2 / (2 * trials)) / denominator
    margin = z * np.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * trials)) / trials) / denominator

    return (max(0, center - margin), min(1, center + margin))


def bootstrap_roi_ci(
    returns: list[float],
    n_bootstrap: int = 10000,
    confidence: float = 0.95
) -> tuple[float, float, float]:
    """
    Bootstrap confidence interval for ROI.
    Returns: (mean_roi, ci_lower, ci_upper)
    """
    if len(returns) == 0:
        return (0.0, 0.0, 0.0)

    returns_arr = np.array(returns)
    mean_roi = np.mean(returns_arr)

    if len(returns) < 2:
        return (mean_roi, mean_roi, mean_roi)

    # Bootstrap resampling
    bootstrap_means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(returns_arr, size=len(returns_arr), replace=True)
        bootstrap_means.append(np.mean(sample))

    alpha = 1 - confidence
    ci_lower = np.percentile(bootstrap_means, alpha / 2 * 100)
    ci_upper = np.percentile(bootstrap_means, (1 - alpha / 2) * 100)

    return (mean_roi, ci_lower, ci_upper)


def evaluate_betting_strategy(
    con: duckdb.DuckDBPyConnection,
    ev_threshold: float = 0.05,
    delta_ko_min: float = 10.0,
    delta_ko_max: float = 90.0,
    min_odds: float = 1.20,
    max_odds: float = 5.00,
) -> dict:
    """
    Evaluate betting strategy on PIT dataset.

    Bet selection rules:
    - EV > ev_threshold (default 5%)
    - Delta to kickoff between delta_ko_min and delta_ko_max minutes
    - Odds within min_odds and max_odds range
    - Valid prediction exists (anti-leakage)
    """

    # Fetch valid PIT records with predictions
    query = f"""
        SELECT
            snapshot_id,
            match_id,
            home_team,
            away_team,
            match_date,
            delta_ko_minutes,
            pit_odds_home,
            pit_odds_draw,
            pit_odds_away,
            home_prob,
            draw_prob,
            away_prob,
            ev_home,
            ev_draw,
            ev_away,
            actual_result,
            home_goals,
            away_goals
        FROM pit_dataset
        WHERE home_prob IS NOT NULL
          AND pit_odds_home IS NOT NULL
          AND delta_ko_minutes BETWEEN {delta_ko_min} AND {delta_ko_max}
    """

    df = con.execute(query).fetchdf()

    if len(df) == 0:
        logger.warning("No valid records found for evaluation")
        return {"error": "No valid records"}

    logger.info(f"Evaluating {len(df)} PIT records with valid predictions")

    results = {
        "parameters": {
            "ev_threshold": ev_threshold,
            "delta_ko_min": delta_ko_min,
            "delta_ko_max": delta_ko_max,
            "min_odds": min_odds,
            "max_odds": max_odds,
        },
        "total_matches": len(df),
        "bets": {
            "home": {"count": 0, "wins": 0, "returns": []},
            "draw": {"count": 0, "wins": 0, "returns": []},
            "away": {"count": 0, "wins": 0, "returns": []},
            "total": {"count": 0, "wins": 0, "returns": []},
        },
        "no_bet_count": 0,
    }

    for _, row in df.iterrows():
        # Calculate EV for each outcome
        evs = {
            "home": (row["home_prob"] * row["pit_odds_home"]) - 1 if row["pit_odds_home"] else -1,
            "draw": (row["draw_prob"] * row["pit_odds_draw"]) - 1 if row["pit_odds_draw"] else -1,
            "away": (row["away_prob"] * row["pit_odds_away"]) - 1 if row["pit_odds_away"] else -1,
        }

        odds = {
            "home": row["pit_odds_home"],
            "draw": row["pit_odds_draw"],
            "away": row["pit_odds_away"],
        }

        # Find best bet (highest EV above threshold)
        best_bet = None
        best_ev = ev_threshold

        for outcome in ["home", "draw", "away"]:
            if evs[outcome] > best_ev:
                # Check odds range
                if odds[outcome] and min_odds <= odds[outcome] <= max_odds:
                    best_ev = evs[outcome]
                    best_bet = outcome

        if best_bet is None:
            results["no_bet_count"] += 1
            continue

        # Determine if bet won
        actual = row["actual_result"]
        bet_won = (
            (best_bet == "home" and actual == "H") or
            (best_bet == "draw" and actual == "D") or
            (best_bet == "away" and actual == "A")
        )

        # Calculate return (1 unit stake)
        if bet_won:
            ret = odds[best_bet] - 1  # Net profit
        else:
            ret = -1  # Lost stake

        # Record bet
        results["bets"][best_bet]["count"] += 1
        results["bets"][best_bet]["returns"].append(ret)
        if bet_won:
            results["bets"][best_bet]["wins"] += 1

        results["bets"]["total"]["count"] += 1
        results["bets"]["total"]["returns"].append(ret)
        if bet_won:
            results["bets"]["total"]["wins"] += 1

    # Calculate statistics for each bet type
    for bet_type in ["home", "draw", "away", "total"]:
        bet_data = results["bets"][bet_type]
        returns = bet_data["returns"]

        if bet_data["count"] > 0:
            # Win rate with CI
            win_rate = bet_data["wins"] / bet_data["count"]
            win_ci_low, win_ci_high = calculate_confidence_interval(
                bet_data["wins"], bet_data["count"]
            )

            # ROI with bootstrap CI
            roi_mean, roi_ci_low, roi_ci_high = bootstrap_roi_ci(returns)

            # Total P&L
            total_pnl = sum(returns)

            bet_data["win_rate"] = round(win_rate, 4)
            bet_data["win_rate_ci"] = [round(win_ci_low, 4), round(win_ci_high, 4)]
            bet_data["roi"] = round(roi_mean, 4)
            bet_data["roi_ci"] = [round(roi_ci_low, 4), round(roi_ci_high, 4)]
            bet_data["total_pnl"] = round(total_pnl, 2)
            bet_data["avg_odds"] = round(np.mean([odds["home"] if bet_type == "home" else
                                                   odds["draw"] if bet_type == "draw" else
                                                   odds["away"] for odds in [{}]]) if bet_type != "total" else 0, 2)
        else:
            bet_data["win_rate"] = 0
            bet_data["win_rate_ci"] = [0, 0]
            bet_data["roi"] = 0
            bet_data["roi_ci"] = [0, 0]
            bet_data["total_pnl"] = 0

        # Remove raw returns from output (too verbose)
        del bet_data["returns"]

    return results


def calibration_analysis(
    con: duckdb.DuckDBPyConnection,
    n_bins: int = 10,
) -> dict:
    """
    Analyze model calibration by comparing predicted probabilities to actual outcomes.
    """
    query = """
        SELECT
            home_prob,
            draw_prob,
            away_prob,
            actual_result
        FROM pit_dataset
        WHERE home_prob IS NOT NULL
    """

    df = con.execute(query).fetchdf()

    if len(df) == 0:
        return {"error": "No data for calibration"}

    calibration = {"n_samples": len(df), "bins": []}

    # Analyze each outcome type
    for outcome, prob_col, result_val in [
        ("home", "home_prob", "H"),
        ("draw", "draw_prob", "D"),
        ("away", "away_prob", "A"),
    ]:
        bin_edges = np.linspace(0, 1, n_bins + 1)
        outcome_bins = []

        for i in range(n_bins):
            low, high = bin_edges[i], bin_edges[i + 1]
            mask = (df[prob_col] >= low) & (df[prob_col] < high)
            subset = df[mask]

            if len(subset) > 0:
                predicted_prob = subset[prob_col].mean()
                actual_freq = (subset["actual_result"] == result_val).mean()
                count = len(subset)

                outcome_bins.append({
                    "bin": f"{low:.1f}-{high:.1f}",
                    "predicted": round(predicted_prob, 3),
                    "actual": round(actual_freq, 3),
                    "count": count,
                    "gap": round(actual_freq - predicted_prob, 3),
                })

        calibration[outcome] = outcome_bins

    return calibration


def generate_report(
    results: dict,
    calibration: dict,
    output_path: str,
) -> None:
    """Generate evaluation report."""

    report = {
        "generated_at": datetime.utcnow().isoformat(),
        "evaluation": results,
        "calibration": calibration,
    }

    # Save JSON report
    json_path = output_path.replace(".duckdb", "_evaluation.json")
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    logger.info(f"Report saved to: {json_path}")

    # Print summary
    print("\n" + "=" * 70)
    print("PIT DATASET EVALUATION REPORT")
    print("=" * 70)

    params = results["parameters"]
    print(f"\nParameters:")
    print(f"  EV threshold: {params['ev_threshold']:.1%}")
    print(f"  Delta KO window: {params['delta_ko_min']:.0f}-{params['delta_ko_max']:.0f} min")
    print(f"  Odds range: {params['min_odds']:.2f}-{params['max_odds']:.2f}")

    print(f"\nDataset:")
    print(f"  Total matches: {results['total_matches']}")
    print(f"  No-bet (no edge): {results['no_bet_count']}")

    print(f"\nBetting Results:")
    print("-" * 70)
    print(f"{'Type':<10} {'Bets':>6} {'Wins':>6} {'Win%':>8} {'Win% CI':>16} {'ROI':>8} {'ROI CI':>16} {'P&L':>8}")
    print("-" * 70)

    for bet_type in ["home", "draw", "away", "total"]:
        b = results["bets"][bet_type]
        if b["count"] > 0:
            win_ci = f"[{b['win_rate_ci'][0]:.1%}, {b['win_rate_ci'][1]:.1%}]"
            roi_ci = f"[{b['roi_ci'][0]:.1%}, {b['roi_ci'][1]:.1%}]"
            print(f"{bet_type.upper():<10} {b['count']:>6} {b['wins']:>6} {b['win_rate']:>7.1%} {win_ci:>16} {b['roi']:>7.1%} {roi_ci:>16} {b['total_pnl']:>+8.2f}")

    print("-" * 70)

    # Statistical significance
    total = results["bets"]["total"]
    if total["count"] > 0:
        roi_significant = total["roi_ci"][0] > 0
        print(f"\nStatistical Significance (95% CI):")
        if roi_significant:
            print(f"  ✓ ROI is significantly positive (CI does not include 0)")
        else:
            print(f"  ✗ ROI is NOT significantly positive (CI includes 0)")
        print(f"  Sample size: {total['count']} bets")

    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Evaluate PIT Dataset (ROI/EV + CI)")
    parser.add_argument(
        "--input", "-i",
        default="data/pit_dataset.duckdb",
        help="Input DuckDB file path (default: data/pit_dataset.duckdb)"
    )
    parser.add_argument(
        "--ev-threshold",
        type=float,
        default=0.05,
        help="Minimum EV to place bet (default: 0.05 = 5%%)"
    )
    parser.add_argument(
        "--delta-ko-min",
        type=float,
        default=10.0,
        help="Minimum minutes before kickoff (default: 10)"
    )
    parser.add_argument(
        "--delta-ko-max",
        type=float,
        default=90.0,
        help="Maximum minutes before kickoff (default: 90)"
    )
    parser.add_argument(
        "--min-odds",
        type=float,
        default=1.20,
        help="Minimum odds to consider (default: 1.20)"
    )
    parser.add_argument(
        "--max-odds",
        type=float,
        default=5.00,
        help="Maximum odds to consider (default: 5.00)"
    )
    args = parser.parse_args()

    logger.info(f"Loading PIT dataset from: {args.input}")
    con = load_pit_dataset(args.input)

    # Run evaluation
    results = evaluate_betting_strategy(
        con,
        ev_threshold=args.ev_threshold,
        delta_ko_min=args.delta_ko_min,
        delta_ko_max=args.delta_ko_max,
        min_odds=args.min_odds,
        max_odds=args.max_odds,
    )

    # Run calibration analysis
    calibration = calibration_analysis(con)

    con.close()

    # Generate report
    generate_report(results, calibration, args.input)


if __name__ == "__main__":
    main()
