"""
Odds validation for market integrity (P0.3).

Validates 1X2 odds for:
- Sanity (within bounds, not NaN/inf)
- Overround (within expected margin range)
"""

import math
import logging
from dataclasses import dataclass
from typing import Optional

from app.telemetry.config import get_telemetry_config
from app.telemetry.metrics import (
    record_odds_violation,
    record_odds_quarantined,
    record_overround,
)

logger = logging.getLogger(__name__)


@dataclass
class OddsValidationResult:
    """Result of odds validation."""

    is_valid: bool
    quarantined: bool
    overround: Optional[float]
    violations: list[str]
    warnings: list[str]

    @property
    def is_usable(self) -> bool:
        """Whether these odds can be used for training/backtesting."""
        return self.is_valid and not self.quarantined


def validate_odds_1x2(
    odds_home: Optional[float],
    odds_draw: Optional[float],
    odds_away: Optional[float],
    provider: str = "api_football",
    book: str = "unknown",
    record_metrics: bool = True,
) -> OddsValidationResult:
    """
    Validate 1X2 (home/draw/away) odds.

    Checks:
    1. Sanity: values are numeric, > 1.01, < 1000, not NaN/inf
    2. Overround: sum of implied probabilities within expected range

    Args:
        odds_home: Home win odds
        odds_draw: Draw odds
        odds_away: Away win odds
        provider: Data provider name (for metrics)
        book: Bookmaker name (for metrics)
        record_metrics: Whether to record Prometheus metrics

    Returns:
        OddsValidationResult with validation status and details
    """
    config = get_telemetry_config()
    violations = []
    warnings = []
    quarantined = False
    overround = None

    # Check if all odds are present
    if odds_home is None or odds_draw is None or odds_away is None:
        violations.append("missing_odds")
        if record_metrics:
            record_odds_violation(provider, book, "1x2", "missing_odds")
        return OddsValidationResult(
            is_valid=False,
            quarantined=True,
            overround=None,
            violations=violations,
            warnings=warnings,
        )

    # Sanity check: numeric, not NaN/inf
    for name, value in [("home", odds_home), ("draw", odds_draw), ("away", odds_away)]:
        if not isinstance(value, (int, float)):
            violations.append(f"{name}_not_numeric")
            quarantined = True
            continue

        if math.isnan(value) or math.isinf(value):
            violations.append(f"{name}_nan_or_inf")
            quarantined = True
            continue

        if value < config.DQ_ODDS_MIN:
            violations.append(f"{name}_too_low")
            quarantined = True

        if value > config.DQ_ODDS_MAX:
            violations.append(f"{name}_too_high")
            quarantined = True

    # If any sanity violation, record and return early
    if violations:
        if record_metrics:
            for v in violations:
                record_odds_violation(provider, book, "1x2", v)
            if quarantined:
                record_odds_quarantined(provider, book, "sanity_failure")
        return OddsValidationResult(
            is_valid=False,
            quarantined=quarantined,
            overround=None,
            violations=violations,
            warnings=warnings,
        )

    # Calculate overround (margin)
    try:
        implied_home = 1 / odds_home
        implied_draw = 1 / odds_draw
        implied_away = 1 / odds_away
        overround = implied_home + implied_draw + implied_away
    except (ZeroDivisionError, TypeError):
        violations.append("overround_calc_error")
        if record_metrics:
            record_odds_violation(provider, book, "1x2", "overround_calc_error")
        return OddsValidationResult(
            is_valid=False,
            quarantined=True,
            overround=None,
            violations=violations,
            warnings=warnings,
        )

    # Record overround metric
    if record_metrics:
        record_overround(provider, book, overround)

    # Check overround bounds
    if overround < config.DQ_OVERROUND_1X2_MIN:
        violations.append("overround_too_low")
        quarantined = True
        if record_metrics:
            record_odds_violation(provider, book, "1x2", "overround_too_low")
            record_odds_quarantined(provider, book, "overround_below_min")

    if overround > config.DQ_OVERROUND_1X2_MAX:
        violations.append("overround_too_high")
        quarantined = True
        if record_metrics:
            record_odds_violation(provider, book, "1x2", "overround_too_high")
            record_odds_quarantined(provider, book, "overround_above_max")

    # Warn if close to bounds
    if not violations:
        if overround < config.DQ_OVERROUND_1X2_MIN + 0.01:
            warnings.append("overround_near_min")
        if overround > config.DQ_OVERROUND_1X2_MAX - 0.02:
            warnings.append("overround_near_max")

    return OddsValidationResult(
        is_valid=len(violations) == 0,
        quarantined=quarantined,
        overround=overround,
        violations=violations,
        warnings=warnings,
    )


def calculate_overround_1x2(
    odds_home: float,
    odds_draw: float,
    odds_away: float,
) -> Optional[float]:
    """
    Calculate overround for 1X2 odds.

    Returns:
        Overround value (e.g., 1.05 = 5% margin) or None if invalid
    """
    try:
        if odds_home <= 0 or odds_draw <= 0 or odds_away <= 0:
            return None
        return (1 / odds_home) + (1 / odds_draw) + (1 / odds_away)
    except (TypeError, ZeroDivisionError):
        return None
