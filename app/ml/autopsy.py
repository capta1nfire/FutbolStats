"""
Financial Autopsy — Post-match prediction classification.

Module 2 of the Post-Match Auditor (GDT directive).

Classifies each prediction into one of 6 mutually exclusive tags
combining: prediction correctness, CLV sign, and xG alignment.

GDT Override 3: XG_DIFF_THRESHOLD = 0.35 (was 0.15).
    xG model noise is ±0.20. With 0.15, technical draws were
    misclassified as "deserved wins", producing false positives.
"""

from enum import Enum
from typing import Optional

# CLV threshold: |clv| < 0.03 is "near zero" (< ~3% line movement)
CLV_THRESHOLD = 0.03

# GDT Override 3: raised from 0.15 to 0.35
# A team must exceed the rival by more than 1/3 expected goal
# for us to statistically consider they "deserved" to win.
XG_DIFF_THRESHOLD = 0.35


class AutopsyTag(str, Enum):
    """Mutually exclusive post-match prediction classification.

    | Tag           | Correct | CLV       | xG aligned | Meaning               |
    |---------------|---------|-----------|------------|-----------------------|
    | SHARP_WIN     | ✓       | > +0.03   | —          | Edge + timing         |
    | ROUTINE_WIN   | ✓       | ≈ 0       | ✓          | Standard correct      |
    | LUCKY_WIN     | ✓       | —         | ✗          | Got lucky             |
    | SHARP_LOSS    | ✗       | > +0.03   | —          | Timing edge, variance |
    | VARIANCE_LOSS | ✗       | ≈ 0       | ✓          | xG backed us          |
    | BLIND_SPOT    | ✗       | < -0.03   | ✗          | Systematic miss       |
    """

    SHARP_WIN = "sharp_win"
    ROUTINE_WIN = "routine_win"
    LUCKY_WIN = "lucky_win"
    SHARP_LOSS = "sharp_loss"
    VARIANCE_LOSS = "variance_loss"
    BLIND_SPOT = "blind_spot"


def classify_autopsy(
    prediction_correct: bool,
    predicted_result: str,
    clv_selected: Optional[float],
    xg_home: Optional[float],
    xg_away: Optional[float],
) -> AutopsyTag:
    """Classify a prediction into an autopsy tag.

    Args:
        prediction_correct: Whether the model predicted the right outcome.
        predicted_result: "home", "draw", or "away".
        clv_selected: CLV on the predicted outcome (from prediction_clv).
        xg_home: Post-match xG for home team.
        xg_away: Post-match xG for away team.

    Returns:
        AutopsyTag enum value.
    """
    # Determine xG-implied winner (with 0.35 threshold to overcome noise)
    xg_supports = None
    if xg_home is not None and xg_away is not None:
        xg_diff = xg_home - xg_away
        if xg_diff > XG_DIFF_THRESHOLD:
            xg_winner = "home"
        elif xg_diff < -XG_DIFF_THRESHOLD:
            xg_winner = "away"
        else:
            xg_winner = "draw"
        xg_supports = xg_winner == predicted_result

    # CLV classification
    has_pos_clv = clv_selected is not None and clv_selected > CLV_THRESHOLD
    has_neg_clv = clv_selected is not None and clv_selected < -CLV_THRESHOLD

    if prediction_correct:
        if has_pos_clv:
            return AutopsyTag.SHARP_WIN
        if xg_supports is False:
            return AutopsyTag.LUCKY_WIN
        return AutopsyTag.ROUTINE_WIN
    else:
        if has_pos_clv:
            return AutopsyTag.SHARP_LOSS
        if xg_supports is True:
            return AutopsyTag.VARIANCE_LOSS
        if has_neg_clv or xg_supports is False:
            return AutopsyTag.BLIND_SPOT
        return AutopsyTag.VARIANCE_LOSS  # conservative default
