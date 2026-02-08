"""
Consensus odds calculation from multiple bookmakers.

Uses median of de-vigged probabilities (proportional method).
ABE-approved formula (2026-02-07).

Algorithm:
  1. Filter each bookmaker through validate_odds_1x2 (P0-2: same rules as capture)
  2. De-vig valid lines using devig_proportional
  3. Take median per outcome across bookmakers
  4. Renormalize and convert to fair odds (1/prob)
"""

from statistics import median
from typing import Optional

from app.ml.devig import devig_proportional
from app.telemetry.validators import validate_odds_1x2

MIN_BOOKS_DEFAULT = 5


def calculate_consensus(
    all_odds: list[dict],
    min_books: int = MIN_BOOKS_DEFAULT,
) -> Optional[dict]:
    """
    Calculate consensus line from multiple bookmakers.

    Args:
        all_odds: List of dicts with bookmaker/odds_home/odds_draw/odds_away
        min_books: Minimum valid bookmakers required (default 5)

    Returns:
        Dict with bookmaker='consensus', fair odds, probs, and n_books.
        None if fewer than min_books valid bookmakers.
    """
    valid_probs = []

    for odds in all_odds:
        h = odds.get("odds_home")
        d = odds.get("odds_draw")
        a = odds.get("odds_away")

        # P0-2: use the same validator as capture pipeline
        validation = validate_odds_1x2(
            odds_home=h, odds_draw=d, odds_away=a,
            book=odds.get("bookmaker", "unknown"),
            record_metrics=False,  # don't pollute Prometheus with consensus internals
        )
        if not validation.is_usable:
            continue

        ph, pd, pa = devig_proportional(h, d, a)
        valid_probs.append((ph, pd, pa))

    if len(valid_probs) < min_books:
        return None

    # Median per outcome
    med_h = median([p[0] for p in valid_probs])
    med_d = median([p[1] for p in valid_probs])
    med_a = median([p[2] for p in valid_probs])

    # Renormalize (numerical safety)
    total = med_h + med_d + med_a
    if total < 0.001:
        return None

    prob_h = med_h / total
    prob_d = med_d / total
    prob_a = med_a / total

    return {
        "bookmaker": "consensus",
        "odds_home": round(1 / prob_h, 3) if prob_h > 0 else None,
        "odds_draw": round(1 / prob_d, 3) if prob_d > 0 else None,
        "odds_away": round(1 / prob_a, 3) if prob_a > 0 else None,
        "consensus_prob_home": round(prob_h, 5),
        "consensus_prob_draw": round(prob_d, 5),
        "consensus_prob_away": round(prob_a, 5),
        "n_books": len(valid_probs),
    }
