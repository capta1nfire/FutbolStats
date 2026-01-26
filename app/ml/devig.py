"""
De-vig methods for removing bookmaker margin from odds.

FASE 3C.1 - ABE approved (2026-01-25)

Two methods available:
- devig_proportional: Baseline (normalize 1/odds). Default.
- devig_power: Alternative (multiplicative method). Use --devig power flag.

Implementation: numpy-only (no SciPy dependency per ABE requirement).
"""

from typing import Tuple


def devig_proportional(
    odds_home: float, odds_draw: float, odds_away: float
) -> Tuple[float, float, float]:
    """
    Proportional/additive de-vig (BASELINE method).

    Simply normalizes 1/odds to sum to 1.
    This is the method used in baseline evaluation.

    Args:
        odds_home: Decimal odds for home win
        odds_draw: Decimal odds for draw
        odds_away: Decimal odds for away win

    Returns:
        Tuple of (prob_home, prob_draw, prob_away) summing to 1.0
    """
    if odds_home <= 1 or odds_draw <= 1 or odds_away <= 1:
        # Invalid odds, return uniform
        return (1/3, 1/3, 1/3)

    implied = [1/odds_home, 1/odds_draw, 1/odds_away]
    total = sum(implied)

    if total < 0.001:
        return (1/3, 1/3, 1/3)

    return (implied[0] / total, implied[1] / total, implied[2] / total)


def devig_power(
    odds_home: float, odds_draw: float, odds_away: float
) -> Tuple[float, float, float]:
    """
    Power method (multiplicative) de-vig.

    More accurate theoretically for 3-way markets.
    Solves for k such that sum((1/o_i)^k) = 1 using bisection (numpy-only).

    The power method assumes the bookmaker applies margin multiplicatively,
    which is more realistic for football 1X2 markets.

    Args:
        odds_home: Decimal odds for home win
        odds_draw: Decimal odds for draw
        odds_away: Decimal odds for away win

    Returns:
        Tuple of (prob_home, prob_draw, prob_away) summing to 1.0
    """
    if odds_home <= 1 or odds_draw <= 1 or odds_away <= 1:
        # Invalid odds, return uniform
        return (1/3, 1/3, 1/3)

    implied = [1/odds_home, 1/odds_draw, 1/odds_away]
    overround = sum(implied)

    # If already fair odds (no margin), return as-is
    if abs(overround - 1.0) < 0.001:
        return tuple(implied)

    # Bisection to find k where sum(p^k) = 1
    def f(k: float) -> float:
        return sum(p**k for p in implied) - 1.0

    # k > 1 if overround > 1 (typical), k < 1 if underround
    k_low, k_high = 0.1, 3.0

    # 50 iterations gives precision < 1e-15
    for _ in range(50):
        k_mid = (k_low + k_high) / 2
        if f(k_mid) > 0:
            k_low = k_mid
        else:
            k_high = k_mid

    k = (k_low + k_high) / 2
    true_probs = [p**k for p in implied]

    # Ensure sum to 1 exactly (numerical safety)
    total = sum(true_probs)
    if total < 0.001:
        return (1/3, 1/3, 1/3)

    return (true_probs[0] / total, true_probs[1] / total, true_probs[2] / total)


def get_devig_function(method: str = "proportional"):
    """
    Get the de-vig function by name.

    Args:
        method: "proportional" (default/baseline) or "power"

    Returns:
        De-vig function
    """
    if method == "power":
        return devig_power
    else:
        return devig_proportional
