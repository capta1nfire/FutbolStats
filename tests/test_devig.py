"""
Tests for de-vig methods (FASE 3C.1).

ABE requirement: numpy-only implementation (no SciPy).
"""

import pytest
from app.ml.devig import devig_proportional, devig_power, get_devig_function


class TestDevigProportional:
    """Test baseline de-vig method."""

    def test_fair_odds_unchanged(self):
        """Fair odds (sum to 1) should be unchanged."""
        # Fair odds: 2.0, 4.0, 4.0 -> 0.5, 0.25, 0.25
        result = devig_proportional(2.0, 4.0, 4.0)
        assert abs(sum(result) - 1.0) < 1e-10
        assert abs(result[0] - 0.5) < 1e-10
        assert abs(result[1] - 0.25) < 1e-10
        assert abs(result[2] - 0.25) < 1e-10

    def test_typical_market_odds(self):
        """Typical market odds with ~5% overround."""
        # Bet365-style odds: 2.10, 3.50, 3.40
        result = devig_proportional(2.10, 3.50, 3.40)
        assert abs(sum(result) - 1.0) < 1e-10
        # Should be approximately normalized
        assert 0.4 < result[0] < 0.5  # home
        assert 0.25 < result[1] < 0.30  # draw
        assert 0.25 < result[2] < 0.30  # away

    def test_invalid_odds_returns_uniform(self):
        """Invalid odds should return uniform distribution."""
        result = devig_proportional(0.5, 1.0, 1.0)
        assert result == (1/3, 1/3, 1/3)

    def test_sums_to_one(self):
        """Result should always sum to 1."""
        test_cases = [
            (1.80, 3.60, 4.50),
            (2.50, 3.20, 2.90),
            (1.20, 6.00, 12.00),
        ]
        for odds in test_cases:
            result = devig_proportional(*odds)
            assert abs(sum(result) - 1.0) < 1e-10


class TestDevigPower:
    """Test power/multiplicative de-vig method."""

    def test_fair_odds_unchanged(self):
        """Fair odds should be unchanged."""
        result = devig_power(2.0, 4.0, 4.0)
        assert abs(sum(result) - 1.0) < 1e-10
        assert abs(result[0] - 0.5) < 1e-10

    def test_typical_market_odds(self):
        """Typical market odds with overround."""
        result = devig_power(2.10, 3.50, 3.40)
        assert abs(sum(result) - 1.0) < 1e-10
        # Power method should give similar but not identical to proportional
        assert 0.4 < result[0] < 0.5

    def test_sums_to_one(self):
        """Result should always sum to 1."""
        test_cases = [
            (1.80, 3.60, 4.50),
            (2.50, 3.20, 2.90),
            (1.20, 6.00, 12.00),
        ]
        for odds in test_cases:
            result = devig_power(*odds)
            assert abs(sum(result) - 1.0) < 1e-10

    def test_invalid_odds_returns_uniform(self):
        """Invalid odds should return uniform distribution."""
        result = devig_power(0.5, 1.0, 1.0)
        assert result == (1/3, 1/3, 1/3)

    def test_power_differs_from_proportional_on_high_overround(self):
        """Power method should differ from proportional on high overround."""
        # High overround case
        odds = (1.50, 4.00, 6.00)
        prop_result = devig_proportional(*odds)
        power_result = devig_power(*odds)

        # Both should sum to 1
        assert abs(sum(prop_result) - 1.0) < 1e-10
        assert abs(sum(power_result) - 1.0) < 1e-10

        # But they should differ (power adjusts more for favorites)
        diff = sum(abs(p - q) for p, q in zip(prop_result, power_result))
        assert diff > 0.001  # Should have measurable difference


class TestGetDevigFunction:
    """Test function selector."""

    def test_default_is_proportional(self):
        """Default should be proportional."""
        fn = get_devig_function()
        assert fn == devig_proportional

    def test_proportional_selection(self):
        """Explicit proportional selection."""
        fn = get_devig_function("proportional")
        assert fn == devig_proportional

    def test_power_selection(self):
        """Power selection."""
        fn = get_devig_function("power")
        assert fn == devig_power

    def test_unknown_defaults_to_proportional(self):
        """Unknown method defaults to proportional."""
        fn = get_devig_function("unknown")
        assert fn == devig_proportional
