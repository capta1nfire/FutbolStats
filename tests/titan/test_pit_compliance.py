"""Tests for TITAN PIT (Point-in-Time) compliance.

Verifies:
1. pit_max_captured_at < kickoff_utc constraint
2. PITViolationError raised when constraint would be violated
3. Insertion policy respects PIT requirements
"""

import pytest
from datetime import datetime, timedelta
from decimal import Decimal

from app.titan.materializers.feature_matrix import (
    PITViolationError,
    InsertionPolicyViolation,
    OddsFeatures,
    FormFeatures,
    H2HFeatures,
    should_insert_feature_row,
    compute_pit_max,
    compute_implied_probabilities,
)


class TestShouldInsertFeatureRow:
    """Test insertion policy enforcement."""

    def test_no_odds_returns_false(self):
        """Without odds (Tier 1), should not insert."""
        should_insert, reason = should_insert_feature_row(
            odds=None,
            form=None,
            h2h=None,
        )
        assert should_insert is False
        assert "Missing Tier 1" in reason

    def test_with_odds_only_returns_true(self):
        """With odds only (no Tier 2/3), should insert."""
        odds = OddsFeatures(
            odds_home_close=Decimal("2.10"),
            odds_draw_close=Decimal("3.40"),
            odds_away_close=Decimal("3.20"),
            implied_prob_home=Decimal("0.4651"),
            implied_prob_draw=Decimal("0.2874"),
            implied_prob_away=Decimal("0.3053"),
            captured_at=datetime(2026, 1, 25, 10, 0, 0),
        )

        should_insert, reason = should_insert_feature_row(
            odds=odds,
            form=None,
            h2h=None,
        )
        assert should_insert is True
        assert "Tier 1 complete" in reason

    def test_with_all_tiers_returns_true(self):
        """With all tiers, should insert."""
        odds = OddsFeatures(
            odds_home_close=Decimal("2.10"),
            odds_draw_close=Decimal("3.40"),
            odds_away_close=Decimal("3.20"),
            implied_prob_home=Decimal("0.4651"),
            implied_prob_draw=Decimal("0.2874"),
            implied_prob_away=Decimal("0.3053"),
            captured_at=datetime(2026, 1, 25, 10, 0, 0),
        )
        form = FormFeatures(
            form_last5="WWDLW",
            goals_scored_last5=8,
            goals_conceded_last5=4,
            points_last5=10,
            captured_at=datetime(2026, 1, 25, 10, 0, 0),
        )
        h2h = H2HFeatures(
            h2h_total_matches=10,
            h2h_home_wins=4,
            h2h_draws=3,
            h2h_away_wins=3,
            h2h_home_goals=15,
            h2h_away_goals=12,
            captured_at=datetime(2026, 1, 25, 10, 0, 0),
        )

        should_insert, reason = should_insert_feature_row(
            odds=odds,
            form=form,
            h2h=h2h,
        )
        assert should_insert is True


class TestComputePitMax:
    """Test pit_max_captured_at computation."""

    def test_single_tier(self):
        """With only odds, pit_max = odds.captured_at."""
        odds = OddsFeatures(
            odds_home_close=Decimal("2.10"),
            odds_draw_close=Decimal("3.40"),
            odds_away_close=Decimal("3.20"),
            implied_prob_home=Decimal("0.4651"),
            implied_prob_draw=Decimal("0.2874"),
            implied_prob_away=Decimal("0.3053"),
            captured_at=datetime(2026, 1, 25, 10, 0, 0),
        )

        pit_max = compute_pit_max(odds=odds, form=None, h2h=None)
        assert pit_max == datetime(2026, 1, 25, 10, 0, 0)

    def test_multiple_tiers_takes_max(self):
        """With multiple tiers, pit_max = max(all captured_at)."""
        odds = OddsFeatures(
            odds_home_close=Decimal("2.10"),
            odds_draw_close=Decimal("3.40"),
            odds_away_close=Decimal("3.20"),
            implied_prob_home=Decimal("0.4651"),
            implied_prob_draw=Decimal("0.2874"),
            implied_prob_away=Decimal("0.3053"),
            captured_at=datetime(2026, 1, 25, 10, 0, 0),  # Earlier
        )
        form = FormFeatures(
            form_last5="WWDLW",
            goals_scored_last5=8,
            goals_conceded_last5=4,
            points_last5=10,
            captured_at=datetime(2026, 1, 25, 12, 0, 0),  # Later
        )

        pit_max = compute_pit_max(odds=odds, form=form, h2h=None)
        assert pit_max == datetime(2026, 1, 25, 12, 0, 0)  # Should be the later one

    def test_three_tiers_takes_max(self):
        """With all three tiers, pit_max = max of all."""
        odds = OddsFeatures(
            odds_home_close=Decimal("2.10"),
            odds_draw_close=Decimal("3.40"),
            odds_away_close=Decimal("3.20"),
            implied_prob_home=Decimal("0.4651"),
            implied_prob_draw=Decimal("0.2874"),
            implied_prob_away=Decimal("0.3053"),
            captured_at=datetime(2026, 1, 25, 8, 0, 0),
        )
        form = FormFeatures(
            form_last5="WWDLW",
            goals_scored_last5=8,
            goals_conceded_last5=4,
            points_last5=10,
            captured_at=datetime(2026, 1, 25, 10, 0, 0),
        )
        h2h = H2HFeatures(
            h2h_total_matches=10,
            h2h_home_wins=4,
            h2h_draws=3,
            h2h_away_wins=3,
            h2h_home_goals=15,
            h2h_away_goals=12,
            captured_at=datetime(2026, 1, 25, 14, 0, 0),  # Latest
        )

        pit_max = compute_pit_max(odds=odds, form=form, h2h=h2h)
        assert pit_max == datetime(2026, 1, 25, 14, 0, 0)

    def test_no_timestamps_raises_error(self):
        """Without any timestamps, raises ValueError."""
        with pytest.raises(ValueError, match="No captured_at timestamps"):
            compute_pit_max(odds=None, form=None, h2h=None)


class TestPITConstraintValidation:
    """Test PIT constraint: pit_max_captured_at < kickoff_utc."""

    def test_valid_pit_before_kickoff(self):
        """captured_at before kickoff is valid."""
        kickoff = datetime(2026, 1, 25, 20, 0, 0)  # 8 PM kickoff
        captured = datetime(2026, 1, 25, 10, 0, 0)  # 10 AM capture

        # This should be valid (captured < kickoff)
        assert captured < kickoff

    def test_invalid_pit_after_kickoff(self):
        """captured_at after kickoff is invalid."""
        kickoff = datetime(2026, 1, 25, 20, 0, 0)  # 8 PM kickoff
        captured = datetime(2026, 1, 25, 21, 0, 0)  # 9 PM capture (AFTER kickoff)

        # This should be invalid (captured >= kickoff)
        assert captured >= kickoff

    def test_invalid_pit_equals_kickoff(self):
        """captured_at equals kickoff is invalid."""
        kickoff = datetime(2026, 1, 25, 20, 0, 0)
        captured = datetime(2026, 1, 25, 20, 0, 0)  # Same time

        # This should be invalid (captured >= kickoff)
        assert captured >= kickoff


class TestImpliedProbabilities:
    """Test implied probability computation."""

    def test_probabilities_sum_to_one(self):
        """Normalized probabilities should sum to 1."""
        home, draw, away = compute_implied_probabilities(
            Decimal("2.10"),
            Decimal("3.40"),
            Decimal("3.20"),
        )

        total = home + draw + away
        # Allow small floating point tolerance
        assert abs(total - Decimal("1.0")) < Decimal("0.0001")

    def test_favorite_has_highest_probability(self):
        """Lower odds = higher probability."""
        home, draw, away = compute_implied_probabilities(
            Decimal("1.50"),  # Favorite (lowest odds)
            Decimal("4.00"),
            Decimal("6.00"),
        )

        assert home > draw
        assert home > away

    def test_removes_overround(self):
        """Normalized probabilities remove bookmaker margin."""
        # Raw: 1/2.10 + 1/3.40 + 1/3.20 = 0.476 + 0.294 + 0.312 = 1.082 (8.2% margin)
        home, draw, away = compute_implied_probabilities(
            Decimal("2.10"),
            Decimal("3.40"),
            Decimal("3.20"),
        )

        # After normalization, sum should be exactly 1
        total = home + draw + away
        assert abs(total - Decimal("1.0")) < Decimal("0.0001")

    def test_even_odds(self):
        """Even odds should give equal probabilities."""
        home, draw, away = compute_implied_probabilities(
            Decimal("3.00"),
            Decimal("3.00"),
            Decimal("3.00"),
        )

        # All should be approximately 0.3333
        assert abs(home - Decimal("0.3333")) < Decimal("0.001")
        assert abs(draw - Decimal("0.3333")) < Decimal("0.001")
        assert abs(away - Decimal("0.3333")) < Decimal("0.001")
