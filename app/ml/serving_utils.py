"""
Serving utilities for post-prediction pipeline.

ABE P0 (2026-02-19): Centralized EV recalculation + optional calibration.

Functions:
- finalize_ev_and_value_bets: Recalculate value_bets from SERVED probabilities
- apply_calibration: Optional temperature scaling (baseline only, NS only)

Both functions are pure transforms on prediction dicts (no DB, no state).
"""

import numpy as np
from typing import Tuple

from app.ml.calibration import TemperatureScaling


def apply_calibration(
    predictions,
    enabled,
    method,
    temperature,
):
    """
    Apply post-hoc temperature calibration to probabilities.

    ABE P0-3: Only baseline predictions (skip_market_anchor=False).
    ABE P0-6: Frozen/FT intocable.

    Args:
        predictions: List of prediction dicts.
        enabled: PROBA_CALIBRATION_ENABLED flag.
        method: "temperature" or "none".
        temperature: Temperature value (1.0 = identity/no-op).

    Returns:
        (predictions, metadata) tuple.
    """
    meta = {"calibration_applied": False, "method": method, "temperature": temperature}

    if not enabled or method == "none" or abs(temperature - 1.0) < 1e-6:
        meta["skip_reason"] = "disabled" if not enabled else "identity"
        return predictions, meta

    # Build calibrator with pre-set temperature (no fit needed for serving)
    cal = TemperatureScaling()
    cal.temperature = temperature
    cal._is_fitted = True

    n_calibrated = 0
    for pred in predictions:
        # P0-6: Frozen/FT intocable
        if pred.get("is_frozen") or pred.get("status") != "NS":
            continue

        # P0-3: Only baseline — skip TS/Family S predictions
        if pred.get("skip_market_anchor"):
            continue

        probs = pred.get("probabilities")
        if not probs:
            continue

        raw = np.array([[probs["home"], probs["draw"], probs["away"]]])
        calibrated = cal.transform(raw)[0]

        pred["probabilities"] = {
            "home": round(float(calibrated[0]), 4),
            "draw": round(float(calibrated[1]), 4),
            "away": round(float(calibrated[2]), 4),
        }

        # Recalculate fair_odds from calibrated probs
        pred["fair_odds"] = {
            "home": round(1 / calibrated[0], 2) if calibrated[0] > 0.001 else None,
            "draw": round(1 / calibrated[1], 2) if calibrated[1] > 0.001 else None,
            "away": round(1 / calibrated[2], 2) if calibrated[2] > 0.001 else None,
        }

        if "policy_metadata" not in pred:
            pred["policy_metadata"] = {}
        pred["policy_metadata"]["calibration"] = {
            "applied": True,
            "method": method,
            "temperature": temperature,
        }
        n_calibrated += 1

    meta["calibration_applied"] = n_calibrated > 0
    meta["n_calibrated"] = n_calibrated
    return predictions, meta


def finalize_ev_and_value_bets(predictions, ml_engine):
    """
    Recalculate value_bets from SERVED probabilities (after all overlays + anchor).

    ABE P0-4: Reuses ml_engine._find_value_bets() — zero logic duplication.
    ABE P0-5: Ghost-edge clear for alpha >= 0.80 (don't reintroduce value_bets).
    ABE P0-6: Frozen/FT intocable.

    Args:
        predictions: List of prediction dicts.
        ml_engine: XGBoostEngine instance (for _find_value_bets).

    Returns:
        (predictions, metadata) tuple.
    """
    n_recalculated = 0
    n_ghost_cleared = 0
    n_skipped = 0

    for pred in predictions:
        # P0-6: Frozen/FT intocable
        if pred.get("is_frozen") or pred.get("status") != "NS":
            n_skipped += 1
            continue

        probs = pred.get("probabilities")
        market = pred.get("market_odds")
        if not probs or not market:
            n_skipped += 1
            continue

        # P0-5: Ghost-edge guard — if alpha >= 0.80, probs ≈ market, no edge
        anchor_meta = (pred.get("policy_metadata") or {}).get("market_anchor", {})
        if anchor_meta.get("applied") and anchor_meta.get("alpha", 0) >= 0.80:
            pred["value_bets"] = None
            pred["has_value_bet"] = False
            pred["best_value_bet"] = None
            n_ghost_cleared += 1
            continue

        # P0-4: Reuse engine method (threshold=None → reads POLICY_EDGE_THRESHOLD)
        prob_array = np.array([probs["home"], probs["draw"], probs["away"]])
        odds_list = [
            market.get("home", 0),
            market.get("draw", 0),
            market.get("away", 0),
        ]

        value_bets = ml_engine._find_value_bets(prob_array, odds_list)

        pred["value_bets"] = value_bets if value_bets else None
        pred["has_value_bet"] = bool(value_bets)
        pred["best_value_bet"] = (
            max(value_bets, key=lambda x: x["expected_value"]) if value_bets else None
        )

        # Update fair_odds from served probs
        pred["fair_odds"] = {
            "home": round(1 / probs["home"], 2) if probs["home"] > 0.001 else None,
            "draw": round(1 / probs["draw"], 2) if probs["draw"] > 0.001 else None,
            "away": round(1 / probs["away"], 2) if probs["away"] > 0.001 else None,
        }

        n_recalculated += 1

    meta = {
        "finalize_ev_applied": True,
        "n_recalculated": n_recalculated,
        "n_ghost_cleared": n_ghost_cleared,
        "n_skipped": n_skipped,
    }
    return predictions, meta
