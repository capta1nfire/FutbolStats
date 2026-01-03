"""Evaluation metrics for prediction model."""

import logging
from typing import Optional

import numpy as np
from sklearn.metrics import brier_score_loss, log_loss

logger = logging.getLogger(__name__)


def calculate_brier_score(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """
    Calculate multi-class Brier score.

    Lower is better. Perfect predictions = 0, random = 0.67.

    Args:
        y_true: True labels (0, 1, or 2).
        y_proba: Predicted probabilities, shape (n_samples, 3).

    Returns:
        Average Brier score across all classes.
    """
    n_classes = y_proba.shape[1]
    brier_scores = []

    for cls in range(n_classes):
        y_true_binary = (y_true == cls).astype(int)
        y_proba_cls = y_proba[:, cls]
        score = brier_score_loss(y_true_binary, y_proba_cls)
        brier_scores.append(score)

    return np.mean(brier_scores)


def calculate_log_loss(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """
    Calculate multi-class log loss.

    Args:
        y_true: True labels (0, 1, or 2).
        y_proba: Predicted probabilities, shape (n_samples, 3).

    Returns:
        Log loss value.
    """
    return log_loss(y_true, y_proba)


def calculate_accuracy(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """
    Calculate prediction accuracy.

    Args:
        y_true: True labels (0, 1, or 2).
        y_proba: Predicted probabilities, shape (n_samples, 3).

    Returns:
        Accuracy (0-1).
    """
    y_pred = np.argmax(y_proba, axis=1)
    return np.mean(y_true == y_pred)


def simulate_roi(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    market_odds: np.ndarray,
    threshold: float = 0.05,
    stake: float = 1.0,
) -> dict:
    """
    Simulate ROI from value betting strategy.

    Places bets when our probability exceeds implied probability by threshold.

    Args:
        y_true: True labels (0, 1, or 2).
        y_proba: Predicted probabilities, shape (n_samples, 3).
        market_odds: Market odds, shape (n_samples, 3) for [home, draw, away].
        threshold: Minimum edge required to place bet (default 5%).
        stake: Amount to stake per bet (default 1.0).

    Returns:
        Dictionary with ROI simulation results.
    """
    total_staked = 0.0
    total_returned = 0.0
    bets_placed = 0
    bets_won = 0

    bet_details = []

    for i in range(len(y_true)):
        actual_result = y_true[i]

        for outcome in range(3):  # home, draw, away
            if market_odds[i, outcome] <= 0:
                continue

            our_prob = y_proba[i, outcome]
            implied_prob = 1 / market_odds[i, outcome]
            edge = our_prob - implied_prob

            if edge > threshold:
                # Place bet
                total_staked += stake
                bets_placed += 1

                won = actual_result == outcome
                payout = stake * market_odds[i, outcome] if won else 0

                if won:
                    bets_won += 1
                    total_returned += payout

                bet_details.append({
                    "match_idx": i,
                    "outcome": ["home", "draw", "away"][outcome],
                    "our_prob": round(our_prob, 4),
                    "implied_prob": round(implied_prob, 4),
                    "edge": round(edge, 4),
                    "odds": market_odds[i, outcome],
                    "won": won,
                    "stake": stake,
                    "payout": round(payout, 2),
                })

    # Calculate ROI
    profit = total_returned - total_staked
    roi = (profit / total_staked * 100) if total_staked > 0 else 0.0
    win_rate = (bets_won / bets_placed * 100) if bets_placed > 0 else 0.0

    return {
        "total_staked": round(total_staked, 2),
        "total_returned": round(total_returned, 2),
        "profit": round(profit, 2),
        "roi_percent": round(roi, 2),
        "bets_placed": bets_placed,
        "bets_won": bets_won,
        "win_rate_percent": round(win_rate, 2),
        "threshold_used": threshold,
        "bet_details": bet_details,
    }


def evaluate_model(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    market_odds: Optional[np.ndarray] = None,
) -> dict:
    """
    Comprehensive model evaluation.

    Args:
        y_true: True labels (0, 1, or 2).
        y_proba: Predicted probabilities, shape (n_samples, 3).
        market_odds: Optional market odds for ROI simulation.

    Returns:
        Dictionary with all evaluation metrics.
    """
    results = {
        "n_samples": len(y_true),
        "brier_score": round(calculate_brier_score(y_true, y_proba), 4),
        "log_loss": round(calculate_log_loss(y_true, y_proba), 4),
        "accuracy": round(calculate_accuracy(y_true, y_proba), 4),
    }

    # Class distribution
    unique, counts = np.unique(y_true, return_counts=True)
    results["class_distribution"] = {
        "home_wins": int(counts[0]) if 0 in unique else 0,
        "draws": int(counts[1]) if 1 in unique else 0,
        "away_wins": int(counts[2]) if 2 in unique else 0,
    }

    # ROI simulation if odds available
    if market_odds is not None:
        # Filter out samples without valid odds
        valid_mask = (market_odds > 0).all(axis=1)
        if valid_mask.sum() > 0:
            roi_results = simulate_roi(
                y_true[valid_mask],
                y_proba[valid_mask],
                market_odds[valid_mask],
            )
            # Remove detailed bet info for summary
            del roi_results["bet_details"]
            results["roi_simulation"] = roi_results

    return results


def calibration_analysis(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_bins: int = 10,
) -> dict:
    """
    Analyze probability calibration.

    Compares predicted probabilities to actual frequencies.

    Args:
        y_true: True labels (0, 1, or 2).
        y_proba: Predicted probabilities, shape (n_samples, 3).
        n_bins: Number of probability bins.

    Returns:
        Calibration analysis for each outcome.
    """
    outcomes = ["home", "draw", "away"]
    calibration = {}

    for cls, outcome in enumerate(outcomes):
        y_true_binary = (y_true == cls).astype(int)
        y_proba_cls = y_proba[:, cls]

        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        actual_freqs = []
        pred_probs = []
        counts = []

        for j in range(n_bins):
            mask = (y_proba_cls >= bin_edges[j]) & (y_proba_cls < bin_edges[j + 1])
            if mask.sum() > 0:
                actual_freqs.append(y_true_binary[mask].mean())
                pred_probs.append(y_proba_cls[mask].mean())
                counts.append(int(mask.sum()))
            else:
                actual_freqs.append(None)
                pred_probs.append(None)
                counts.append(0)

        calibration[outcome] = {
            "bin_centers": bin_centers.tolist(),
            "actual_frequencies": actual_freqs,
            "predicted_probabilities": pred_probs,
            "sample_counts": counts,
        }

    return calibration
