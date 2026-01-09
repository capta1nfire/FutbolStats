#!/usr/bin/env python3
"""
Evaluate PIT Dataset - ROI/EV Analysis with Confidence Intervals

Analyzes the PIT dataset to calculate:
- Expected Value (EV) per bet type
- Return on Investment (ROI) with confidence intervals
- Calibration analysis
- Market benchmark comparison (Brier score, de-vig probabilities)

BENCHMARK METHODOLOGY:
----------------------
Market probabilities are derived from PIT odds using de-vig normalization:
  qH = 1/pit_odds_home, qD = 1/pit_odds_draw, qA = 1/pit_odds_away
  overround = qH + qD + qA  (typically 1.03-1.10)
  mH = qH/overround, mD = qD/overround, mA = qA/overround

Bet selection rule (applies to both model and market):
  - Bet on outcome with highest EV above threshold
  - Same ev_threshold, odds range, and delta_ko filters for fair comparison

Brier Score (multi-class):
  Brier = sum_{k in {H,D,A}} (p_k - y_k)^2
  where y_k is one-hot encoded actual result

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

    # Avoid SciPy dependency (not in requirements.txt).
    # statistics.NormalDist is available in Python 3.8+
    from statistics import NormalDist

    z = NormalDist().inv_cdf(1 - (1 - confidence) / 2)
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
            "home": {"count": 0, "wins": 0, "returns": [], "odds": []},
            "draw": {"count": 0, "wins": 0, "returns": [], "odds": []},
            "away": {"count": 0, "wins": 0, "returns": [], "odds": []},
            "total": {"count": 0, "wins": 0, "returns": [], "odds": []},
        },
        "no_bet_count": 0,
    }

    for _, row in df.iterrows():
        # Normalize model probabilities defensively (should already sum ~1, but avoid drift)
        ph = float(row["home_prob"]) if row["home_prob"] is not None else 0.0
        pd = float(row["draw_prob"]) if row["draw_prob"] is not None else 0.0
        pa = float(row["away_prob"]) if row["away_prob"] is not None else 0.0
        s = ph + pd + pa
        if s > 0:
            ph, pd, pa = ph / s, pd / s, pa / s

        # Calculate EV for each outcome
        evs = {
            "home": (ph * row["pit_odds_home"]) - 1 if row["pit_odds_home"] else -1,
            "draw": (pd * row["pit_odds_draw"]) - 1 if row["pit_odds_draw"] else -1,
            "away": (pa * row["pit_odds_away"]) - 1 if row["pit_odds_away"] else -1,
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
        results["bets"][best_bet]["odds"].append(float(odds[best_bet]) if odds[best_bet] is not None else None)
        if bet_won:
            results["bets"][best_bet]["wins"] += 1

        results["bets"]["total"]["count"] += 1
        results["bets"]["total"]["returns"].append(ret)
        results["bets"]["total"]["odds"].append(float(odds[best_bet]) if odds[best_bet] is not None else None)
        if bet_won:
            results["bets"]["total"]["wins"] += 1

    # Calculate statistics for each bet type
    for bet_type in ["home", "draw", "away", "total"]:
        bet_data = results["bets"][bet_type]
        returns = bet_data["returns"]
        odds_used = [o for o in (bet_data.get("odds") or []) if o is not None]

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
            bet_data["avg_odds"] = round(float(np.mean(np.array(odds_used))), 2) if odds_used else None
        else:
            bet_data["win_rate"] = 0
            bet_data["win_rate_ci"] = [0, 0]
            bet_data["roi"] = 0
            bet_data["roi_ci"] = [0, 0]
            bet_data["total_pnl"] = 0
            bet_data["avg_odds"] = None

        # Remove raw returns from output (too verbose)
        del bet_data["returns"]
        del bet_data["odds"]

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


def calculate_brier_score(probs: dict, actual_result: str) -> float:
    """
    Calculate multi-class Brier score for a single prediction.

    Brier = sum_{k in {H,D,A}} (p_k - y_k)^2
    where y_k is 1 if k == actual_result, else 0
    """
    one_hot = {"H": 0.0, "D": 0.0, "A": 0.0}
    if actual_result in one_hot:
        one_hot[actual_result] = 1.0

    brier = 0.0
    for k in ["H", "D", "A"]:
        p_k = probs.get(k, 0.0)
        y_k = one_hot[k]
        brier += (p_k - y_k) ** 2

    return brier


def benchmark_model_vs_market(
    con: duckdb.DuckDBPyConnection,
    ev_threshold: float = 0.05,
    delta_ko_min: float = 10.0,
    delta_ko_max: float = 90.0,
    min_odds: float = 1.20,
    max_odds: float = 5.00,
) -> dict:
    """
    Benchmark model vs market (de-vig implied probabilities).

    Compares:
    - Brier scores (model vs market)
    - ROI/EV using same bet selection rule for both
    - Win rates with confidence intervals

    Market probabilities are derived via de-vig:
      qH = 1/odds_home, qD = 1/odds_draw, qA = 1/odds_away
      overround = qH + qD + qA
      mH = qH/overround, mD = qD/overround, mA = qA/overround
    """

    # Fetch records where ALL odds are present (required for de-vig)
    query = f"""
        SELECT
            snapshot_id,
            match_id,
            home_team,
            away_team,
            delta_ko_minutes,
            pit_odds_home,
            pit_odds_draw,
            pit_odds_away,
            home_prob,
            draw_prob,
            away_prob,
            actual_result
        FROM pit_dataset
        WHERE home_prob IS NOT NULL
          AND pit_odds_home IS NOT NULL
          AND pit_odds_draw IS NOT NULL
          AND pit_odds_away IS NOT NULL
          AND delta_ko_minutes BETWEEN {delta_ko_min} AND {delta_ko_max}
    """

    df = con.execute(query).fetchdf()

    if len(df) == 0:
        logger.warning("No valid records for benchmark (need all three odds)")
        return {"error": "No valid records for benchmark"}

    logger.info(f"Benchmarking {len(df)} records with complete odds")

    # Accumulators
    brier_model_sum = 0.0
    brier_market_sum = 0.0
    overrounds = []

    model_stats = {"count": 0, "wins": 0, "returns": []}
    market_stats = {"count": 0, "wins": 0, "returns": []}

    valid_count = 0

    for _, row in df.iterrows():
        # --- Model probabilities (normalized) ---
        ph = float(row["home_prob"]) if row["home_prob"] is not None else 0.0
        pd = float(row["draw_prob"]) if row["draw_prob"] is not None else 0.0
        pa = float(row["away_prob"]) if row["away_prob"] is not None else 0.0
        s = ph + pd + pa
        if s > 0:
            ph, pd, pa = ph / s, pd / s, pa / s

        model_probs = {"H": ph, "D": pd, "A": pa}

        # --- Market probabilities (de-vig) ---
        odds_h = float(row["pit_odds_home"])
        odds_d = float(row["pit_odds_draw"])
        odds_a = float(row["pit_odds_away"])

        if odds_h <= 0 or odds_d <= 0 or odds_a <= 0:
            continue  # Invalid odds

        q_h = 1.0 / odds_h
        q_d = 1.0 / odds_d
        q_a = 1.0 / odds_a
        overround = q_h + q_d + q_a

        # Validate overround is in expected range
        if overround < 1.0 or overround > 1.30:
            logger.debug(f"Unusual overround {overround:.3f} for match {row['match_id']}")

        overrounds.append(overround)

        m_h = q_h / overround
        m_d = q_d / overround
        m_a = q_a / overround

        # Validate market probs sum to ~1
        market_sum = m_h + m_d + m_a
        if abs(market_sum - 1.0) > 1e-6:
            logger.warning(f"Market probs sum to {market_sum:.6f}, expected 1.0")

        market_probs = {"H": m_h, "D": m_d, "A": m_a}

        actual = row["actual_result"]
        if actual not in ["H", "D", "A"]:
            continue  # Skip if no result yet

        valid_count += 1

        # --- Brier scores ---
        brier_model_sum += calculate_brier_score(model_probs, actual)
        brier_market_sum += calculate_brier_score(market_probs, actual)

        # --- EV-based betting (model) ---
        model_evs = {
            "home": (ph * odds_h) - 1,
            "draw": (pd * odds_d) - 1,
            "away": (pa * odds_a) - 1,
        }

        # --- EV-based betting (market) ---
        # Market EV = market_prob * odds - 1
        # Note: For market probs, EV will always be ~0 after de-vig (by construction)
        # But we use the raw odds for returns, just different prob estimation
        market_evs = {
            "home": (m_h * odds_h) - 1,
            "draw": (m_d * odds_d) - 1,
            "away": (m_a * odds_a) - 1,
        }

        odds = {"home": odds_h, "draw": odds_d, "away": odds_a}
        result_map = {"home": "H", "draw": "D", "away": "A"}

        # Model bet selection
        model_best_bet = None
        model_best_ev = ev_threshold
        for outcome in ["home", "draw", "away"]:
            if model_evs[outcome] > model_best_ev:
                if min_odds <= odds[outcome] <= max_odds:
                    model_best_ev = model_evs[outcome]
                    model_best_bet = outcome

        if model_best_bet:
            model_stats["count"] += 1
            won = (actual == result_map[model_best_bet])
            if won:
                model_stats["wins"] += 1
                model_stats["returns"].append(odds[model_best_bet] - 1)
            else:
                model_stats["returns"].append(-1)

        # Market bet selection (same rule for fair comparison)
        market_best_bet = None
        market_best_ev = ev_threshold
        for outcome in ["home", "draw", "away"]:
            if market_evs[outcome] > market_best_ev:
                if min_odds <= odds[outcome] <= max_odds:
                    market_best_ev = market_evs[outcome]
                    market_best_bet = outcome

        if market_best_bet:
            market_stats["count"] += 1
            won = (actual == result_map[market_best_bet])
            if won:
                market_stats["wins"] += 1
                market_stats["returns"].append(odds[market_best_bet] - 1)
            else:
                market_stats["returns"].append(-1)

    if valid_count == 0:
        return {"error": "No valid records with results"}

    # Calculate average Brier scores
    brier_model = brier_model_sum / valid_count
    brier_market = brier_market_sum / valid_count
    delta_brier = brier_market - brier_model  # Positive = model is better

    # Overround statistics
    overround_stats = {
        "mean": round(float(np.mean(overrounds)), 4),
        "min": round(float(np.min(overrounds)), 4),
        "max": round(float(np.max(overrounds)), 4),
    }

    # Model ROI/win rate with CIs
    model_result = {
        "n_bets": model_stats["count"],
        "wins": model_stats["wins"],
        "brier": round(brier_model, 4),
    }

    if model_stats["count"] > 0:
        win_rate = model_stats["wins"] / model_stats["count"]
        win_ci_low, win_ci_high = calculate_confidence_interval(
            model_stats["wins"], model_stats["count"]
        )
        roi_mean, roi_ci_low, roi_ci_high = bootstrap_roi_ci(model_stats["returns"])

        model_result.update({
            "win_rate": round(win_rate, 4),
            "win_rate_ci": [round(win_ci_low, 4), round(win_ci_high, 4)],
            "roi": round(roi_mean, 4),
            "roi_ci": [round(roi_ci_low, 4), round(roi_ci_high, 4)],
            "total_pnl": round(sum(model_stats["returns"]), 2),
        })
    else:
        model_result.update({
            "win_rate": 0, "win_rate_ci": [0, 0],
            "roi": 0, "roi_ci": [0, 0], "total_pnl": 0,
        })

    # Market ROI/win rate with CIs
    market_result = {
        "n_bets": market_stats["count"],
        "wins": market_stats["wins"],
        "brier": round(brier_market, 4),
    }

    if market_stats["count"] > 0:
        win_rate = market_stats["wins"] / market_stats["count"]
        win_ci_low, win_ci_high = calculate_confidence_interval(
            market_stats["wins"], market_stats["count"]
        )
        roi_mean, roi_ci_low, roi_ci_high = bootstrap_roi_ci(market_stats["returns"])

        market_result.update({
            "win_rate": round(win_rate, 4),
            "win_rate_ci": [round(win_ci_low, 4), round(win_ci_high, 4)],
            "roi": round(roi_mean, 4),
            "roi_ci": [round(roi_ci_low, 4), round(roi_ci_high, 4)],
            "total_pnl": round(sum(market_stats["returns"]), 2),
        })
    else:
        market_result.update({
            "win_rate": 0, "win_rate_ci": [0, 0],
            "roi": 0, "roi_ci": [0, 0], "total_pnl": 0,
        })

    # Calculate delta ROI
    delta_roi = model_result["roi"] - market_result["roi"]

    return {
        "n_matches": valid_count,
        "overround": overround_stats,
        "model": model_result,
        "market": market_result,
        "delta": {
            "brier": round(delta_brier, 4),  # Positive = model better
            "roi": round(delta_roi, 4),      # Positive = model better
        },
    }


def generate_report(
    results: dict,
    calibration: dict,
    benchmark: dict,
    output_path: str,
) -> None:
    """Generate evaluation report."""

    report = {
        "generated_at": datetime.utcnow().isoformat(),
        "evaluation": results,
        "calibration": calibration,
        "benchmark": benchmark,
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

    # Benchmark section
    if benchmark and "error" not in benchmark:
        print("\n" + "=" * 70)
        print("MODEL vs MARKET BENCHMARK")
        print("=" * 70)

        print(f"\nDataset: {benchmark['n_matches']} matches with complete odds")
        ovr = benchmark["overround"]
        print(f"Overround: mean={ovr['mean']:.2%}, range=[{ovr['min']:.2%}, {ovr['max']:.2%}]")

        print(f"\nBrier Score (lower is better):")
        print(f"  Model:  {benchmark['model']['brier']:.4f}")
        print(f"  Market: {benchmark['market']['brier']:.4f}")
        delta_b = benchmark["delta"]["brier"]
        if delta_b > 0:
            print(f"  Delta:  {delta_b:+.4f} (model is better)")
        else:
            print(f"  Delta:  {delta_b:+.4f} (market is better)")

        print(f"\nROI Comparison (EV threshold applied):")
        print("-" * 70)
        print(f"{'Source':<10} {'Bets':>6} {'Wins':>6} {'Win%':>8} {'Win% CI':>16} {'ROI':>8} {'ROI CI':>16} {'P&L':>8}")
        print("-" * 70)

        for src_name, src_data in [("MODEL", benchmark["model"]), ("MARKET", benchmark["market"])]:
            if src_data.get("n_bets", 0) > 0:
                win_ci = f"[{src_data['win_rate_ci'][0]:.1%}, {src_data['win_rate_ci'][1]:.1%}]"
                roi_ci = f"[{src_data['roi_ci'][0]:.1%}, {src_data['roi_ci'][1]:.1%}]"
                print(f"{src_name:<10} {src_data['n_bets']:>6} {src_data['wins']:>6} {src_data['win_rate']:>7.1%} {win_ci:>16} {src_data['roi']:>7.1%} {roi_ci:>16} {src_data['total_pnl']:>+8.2f}")
            else:
                print(f"{src_name:<10} {'N/A':>6} (no bets above threshold)")

        print("-" * 70)

        delta_roi = benchmark["delta"]["roi"]
        print(f"\nDelta ROI: {delta_roi:+.2%}", end="")
        if delta_roi > 0:
            print(" (model outperforms market)")
        elif delta_roi < 0:
            print(" (market outperforms model)")
        else:
            print(" (no difference)")

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

    # Run benchmark (model vs market)
    benchmark = benchmark_model_vs_market(
        con,
        ev_threshold=args.ev_threshold,
        delta_ko_min=args.delta_ko_min,
        delta_ko_max=args.delta_ko_max,
        min_odds=args.min_odds,
        max_odds=args.max_odds,
    )

    con.close()

    # Generate report
    generate_report(results, calibration, benchmark, args.input)


if __name__ == "__main__":
    main()
