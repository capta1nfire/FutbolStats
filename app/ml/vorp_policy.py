"""
VORP Policy — lineup shock adjustment (Sprint 2, Camino B).

Pure function, no side effects, no DB dependency.
Applies the log-shift-softmax adjustment calibrated in calibrate_vorp_lab.py.

Formula (GDT exact):
  Z = log(P_base)              # logit transform (clipped)
  Z_home += β * d              # shift home by talent advantage
  Z_away -= β * d              # shift away inversely
  Z_draw  unchanged            # draw absorbs via softmax renorm
  P_adj = softmax(Z_adj)       # back to probability space

Invariants:
  - probs ∈ [0, 1], sum to 1.0
  - Monotonicity: if β >= 0 and d > 0, home goes up, away goes down
  - No change if d == 0 or β == 0
  - No NaN/Inf in output (eps clipping + stable softmax)

Config: VORP_BETA env var or default from lab calibration.
"""

from __future__ import annotations

import logging
import os

import numpy as np

logger = logging.getLogger("futbolstats.vorp_policy")

# ─── Constants (must match calibrate_vorp_lab.py exactly) ────────────────
EPS = 1e-7
_DEFAULT_BETA = 1.4206  # β* from lab OOS optimization (2026-02-23)


def _get_vorp_beta() -> float:
    """Get β from env var or default."""
    raw = os.environ.get("VORP_BETA", "")
    if raw.strip():
        try:
            beta = float(raw.strip())
            if beta < 0:
                logger.warning("VORP_BETA=%s is negative, using 0", raw)
                return 0.0
            return beta
        except ValueError:
            logger.warning("Invalid VORP_BETA='%s', using default %.4f", raw, _DEFAULT_BETA)
    return _DEFAULT_BETA


VORP_BETA: float = _get_vorp_beta()


def _stable_softmax_row(z: np.ndarray) -> np.ndarray:
    """Numerically stable softmax for a single (3,) array."""
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def apply_lineup_shock(
    probabilities: dict,
    talent_delta_diff: float,
    beta: float | None = None,
) -> dict:
    """
    Apply VORP lineup shock adjustment to a prediction's probabilities.

    Pure function — takes a probabilities dict, returns a new dict.
    No side effects, no DB access.

    Args:
        probabilities: dict with keys "home", "draw", "away" (floats 0-1)
        talent_delta_diff: home_talent_delta - away_talent_delta
            Positive = home lineup stronger than expected (relative to away)
        beta: override for VORP_BETA (default: lab-calibrated value)

    Returns:
        dict with keys "home", "draw", "away" — adjusted probabilities
        summing to 1.0. Returns original probabilities unchanged if:
        - talent_delta_diff is None/NaN/0
        - beta is 0
        - input probabilities are invalid
    """
    if beta is None:
        beta = VORP_BETA

    # Guard: no adjustment needed
    if (beta == 0
            or talent_delta_diff is None
            or (isinstance(talent_delta_diff, float) and np.isnan(talent_delta_diff))
            or talent_delta_diff == 0):
        return dict(probabilities)

    # Extract base probs
    h = probabilities.get("home", 0)
    d = probabilities.get("draw", 0)
    a = probabilities.get("away", 0)

    # Guard: invalid input
    if not (isinstance(h, (int, float)) and isinstance(d, (int, float))
            and isinstance(a, (int, float))):
        return dict(probabilities)
    if h <= 0 or d <= 0 or a <= 0:
        return dict(probabilities)

    # Log transform (clipped to avoid -inf)
    p = np.array([float(h), float(d), float(a)], dtype=np.float64)
    p_clipped = np.clip(p, EPS, 1.0 - EPS)
    z = np.log(p_clipped)

    # Shift (GDT exact formula)
    dd = float(talent_delta_diff)
    z[0] += beta * dd    # home boost
    z[2] -= beta * dd    # away penalty
    # z[1] unchanged     # draw absorbs via softmax

    # Softmax back to probability space
    p_adj = _stable_softmax_row(z)

    return {
        "home": round(float(p_adj[0]), 4),
        "draw": round(float(p_adj[1]), 4),
        "away": round(float(p_adj[2]), 4),
    }
