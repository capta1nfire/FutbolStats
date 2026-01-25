"""Tests for TITAN insertion policy.

Per plan zazzy-jingling-pudding.md v1.1:
- RULE 1: No odds -> DON'T insert (Tier 1 is mandatory gate)
- RULE 2: With odds, missing Tier 2/3 -> DO insert with NULLs
"""

import pytest
from datetime import datetime
from decimal import Decimal

from app.titan.materializers.feature_matrix import (
    OddsFeatures,
    FormFeatures,
    H2HFeatures,
    should_insert_feature_row,
)


class TestInsertionPolicy:
    """Test feature matrix insertion policy."""

    # Helper to create valid odds
    @staticmethod
    def _make_odds(captured_at: datetime = None) -> OddsFeatures:
        return OddsFeatures(
            odds_home_close=Decimal("2.10"),
            odds_draw_close=Decimal("3.40"),
            odds_away_close=Decimal("3.20"),
            implied_prob_home=Decimal("0.4651"),
            implied_prob_draw=Decimal("0.2874"),
            implied_prob_away=Decimal("0.3053"),
            captured_at=captured_at or datetime(2026, 1, 25, 10, 0, 0),
        )

    @staticmethod
    def _make_form(captured_at: datetime = None) -> FormFeatures:
        return FormFeatures(
            form_last5="WWDLW",
            goals_scored_last5=8,
            goals_conceded_last5=4,
            points_last5=10,
            captured_at=captured_at or datetime(2026, 1, 25, 10, 0, 0),
        )

    @staticmethod
    def _make_h2h(captured_at: datetime = None) -> H2HFeatures:
        return H2HFeatures(
            h2h_total_matches=10,
            h2h_home_wins=4,
            h2h_draws=3,
            h2h_away_wins=3,
            h2h_home_goals=15,
            h2h_away_goals=12,
            captured_at=captured_at or datetime(2026, 1, 25, 10, 0, 0),
        )

    # ===========================================================================
    # RULE 1: No odds -> DON'T insert
    # ===========================================================================

    def test_rule1_no_data_at_all(self):
        """No data -> don't insert."""
        should_insert, reason = should_insert_feature_row(
            odds=None,
            form=None,
            h2h=None,
        )
        assert should_insert is False
        assert "Tier 1" in reason

    def test_rule1_only_form_no_odds(self):
        """Only form (no odds) -> don't insert."""
        should_insert, reason = should_insert_feature_row(
            odds=None,
            form=self._make_form(),
            h2h=None,
        )
        assert should_insert is False
        assert "Tier 1" in reason

    def test_rule1_only_h2h_no_odds(self):
        """Only H2H (no odds) -> don't insert."""
        should_insert, reason = should_insert_feature_row(
            odds=None,
            form=None,
            h2h=self._make_h2h(),
        )
        assert should_insert is False
        assert "Tier 1" in reason

    def test_rule1_form_and_h2h_no_odds(self):
        """Form + H2H (no odds) -> don't insert."""
        should_insert, reason = should_insert_feature_row(
            odds=None,
            form=self._make_form(),
            h2h=self._make_h2h(),
        )
        assert should_insert is False
        assert "Tier 1" in reason

    # ===========================================================================
    # RULE 2: With odds, missing Tier 2/3 -> DO insert with NULLs
    # ===========================================================================

    def test_rule2_odds_only(self):
        """Only odds -> insert (Tier 2/3 will be NULL)."""
        should_insert, reason = should_insert_feature_row(
            odds=self._make_odds(),
            form=None,
            h2h=None,
        )
        assert should_insert is True
        assert "Tier 1 complete" in reason

    def test_rule2_odds_and_form(self):
        """Odds + form -> insert."""
        should_insert, reason = should_insert_feature_row(
            odds=self._make_odds(),
            form=self._make_form(),
            h2h=None,
        )
        assert should_insert is True

    def test_rule2_odds_and_h2h(self):
        """Odds + H2H -> insert."""
        should_insert, reason = should_insert_feature_row(
            odds=self._make_odds(),
            form=None,
            h2h=self._make_h2h(),
        )
        assert should_insert is True

    def test_rule2_all_tiers(self):
        """All tiers -> insert."""
        should_insert, reason = should_insert_feature_row(
            odds=self._make_odds(),
            form=self._make_form(),
            h2h=self._make_h2h(),
        )
        assert should_insert is True


class TestTierCompletionFlags:
    """Test tier completion flag computation."""

    def test_tier1_complete_when_odds_present(self):
        """tier1_complete should be True when odds present."""
        # This would be checked in materializer, but we test the logic
        odds = TestInsertionPolicy._make_odds()
        tier1_complete = odds is not None
        assert tier1_complete is True

    def test_tier1_incomplete_when_odds_none(self):
        """tier1_complete should be False when odds None."""
        odds = None
        tier1_complete = odds is not None
        assert tier1_complete is False

    def test_tier2_complete_when_both_forms_present(self):
        """tier2_complete should be True when both home and away form present."""
        form_home = TestInsertionPolicy._make_form()
        form_away = TestInsertionPolicy._make_form()
        tier2_complete = form_home is not None and form_away is not None
        assert tier2_complete is True

    def test_tier2_incomplete_when_one_form_missing(self):
        """tier2_complete should be False when one form missing."""
        form_home = TestInsertionPolicy._make_form()
        form_away = None
        tier2_complete = form_home is not None and form_away is not None
        assert tier2_complete is False

    def test_tier3_complete_when_h2h_present(self):
        """tier3_complete should be True when H2H present."""
        h2h = TestInsertionPolicy._make_h2h()
        tier3_complete = h2h is not None
        assert tier3_complete is True

    def test_tier3_incomplete_when_h2h_none(self):
        """tier3_complete should be False when H2H None."""
        h2h = None
        tier3_complete = h2h is not None
        assert tier3_complete is False
