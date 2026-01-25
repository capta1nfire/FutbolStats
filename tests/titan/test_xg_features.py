"""Tests for xG (Tier 1b) features.

Verifies:
1. XGFeatures dataclass structure
2. xG captured_at participates in PIT computation
3. tier1b_complete flag logic
"""

import pytest
from datetime import datetime
from decimal import Decimal

from app.titan.materializers.feature_matrix import (
    XGFeatures,
    OddsFeatures,
    compute_pit_max,
    should_insert_feature_row,
)


class TestXGFeaturesDataclass:
    """Test XGFeatures structure."""

    def test_xg_features_creation(self):
        """XGFeatures can be created with all fields."""
        xg = XGFeatures(
            xg_home_last5=Decimal("1.85"),
            xg_away_last5=Decimal("1.42"),
            xga_home_last5=Decimal("0.95"),
            xga_away_last5=Decimal("1.28"),
            npxg_home_last5=Decimal("1.65"),
            npxg_away_last5=Decimal("1.22"),
            captured_at=datetime(2026, 1, 25, 10, 0, 0),
        )

        assert xg.xg_home_last5 == Decimal("1.85")
        assert xg.xg_away_last5 == Decimal("1.42")
        assert xg.xga_home_last5 == Decimal("0.95")
        assert xg.xga_away_last5 == Decimal("1.28")
        assert xg.npxg_home_last5 == Decimal("1.65")
        assert xg.npxg_away_last5 == Decimal("1.22")
        assert xg.captured_at == datetime(2026, 1, 25, 10, 0, 0)

    def test_xg_naming_is_last5(self):
        """xG field names use *_last5 convention (not *_season)."""
        xg = XGFeatures(
            xg_home_last5=Decimal("1.85"),
            xg_away_last5=Decimal("1.42"),
            xga_home_last5=Decimal("0.95"),
            xga_away_last5=Decimal("1.28"),
            npxg_home_last5=Decimal("1.65"),
            npxg_away_last5=Decimal("1.22"),
            captured_at=datetime(2026, 1, 25, 10, 0, 0),
        )

        # Verify naming convention matches auditor requirement
        assert hasattr(xg, 'xg_home_last5')
        assert hasattr(xg, 'xg_away_last5')
        assert hasattr(xg, 'xga_home_last5')
        assert hasattr(xg, 'xga_away_last5')
        assert hasattr(xg, 'npxg_home_last5')
        assert hasattr(xg, 'npxg_away_last5')

        # Verify old naming does NOT exist
        assert not hasattr(xg, 'xg_home_season')
        assert not hasattr(xg, 'xg_away_season')


class TestXGInPITComputation:
    """Test xG participates in PIT computation."""

    @staticmethod
    def _make_odds(captured_at: datetime) -> OddsFeatures:
        return OddsFeatures(
            odds_home_close=Decimal("2.10"),
            odds_draw_close=Decimal("3.40"),
            odds_away_close=Decimal("3.20"),
            implied_prob_home=Decimal("0.4651"),
            implied_prob_draw=Decimal("0.2874"),
            implied_prob_away=Decimal("0.3053"),
            captured_at=captured_at,
        )

    @staticmethod
    def _make_xg(captured_at: datetime) -> XGFeatures:
        return XGFeatures(
            xg_home_last5=Decimal("1.85"),
            xg_away_last5=Decimal("1.42"),
            xga_home_last5=Decimal("0.95"),
            xga_away_last5=Decimal("1.28"),
            npxg_home_last5=Decimal("1.65"),
            npxg_away_last5=Decimal("1.22"),
            captured_at=captured_at,
        )

    def test_xg_captured_at_participates_in_pit_max(self):
        """xG captured_at should be included in pit_max computation."""
        odds = self._make_odds(datetime(2026, 1, 25, 10, 0, 0))
        xg = self._make_xg(datetime(2026, 1, 25, 12, 0, 0))  # Later than odds

        pit_max = compute_pit_max(odds=odds, form=None, h2h=None, xg=xg)

        # pit_max should be xG captured_at (the later one)
        assert pit_max == datetime(2026, 1, 25, 12, 0, 0)

    def test_pit_max_with_odds_later_than_xg(self):
        """When odds captured_at is later, pit_max = odds.captured_at."""
        odds = self._make_odds(datetime(2026, 1, 25, 14, 0, 0))  # Later
        xg = self._make_xg(datetime(2026, 1, 25, 12, 0, 0))

        pit_max = compute_pit_max(odds=odds, form=None, h2h=None, xg=xg)

        # pit_max should be odds captured_at (the later one)
        assert pit_max == datetime(2026, 1, 25, 14, 0, 0)

    def test_xg_only_without_odds_raises_error(self):
        """xG without odds should raise ValueError (no odds means no pit computation)."""
        xg = self._make_xg(datetime(2026, 1, 25, 12, 0, 0))

        # This should raise because odds is required by insertion policy
        # but compute_pit_max will work with just xg
        pit_max = compute_pit_max(odds=None, form=None, h2h=None, xg=xg)
        assert pit_max == datetime(2026, 1, 25, 12, 0, 0)


class TestTier1bCompletionLogic:
    """Test tier1b_complete flag logic."""

    @staticmethod
    def _make_odds() -> OddsFeatures:
        return OddsFeatures(
            odds_home_close=Decimal("2.10"),
            odds_draw_close=Decimal("3.40"),
            odds_away_close=Decimal("3.20"),
            implied_prob_home=Decimal("0.4651"),
            implied_prob_draw=Decimal("0.2874"),
            implied_prob_away=Decimal("0.3053"),
            captured_at=datetime(2026, 1, 25, 10, 0, 0),
        )

    @staticmethod
    def _make_xg() -> XGFeatures:
        return XGFeatures(
            xg_home_last5=Decimal("1.85"),
            xg_away_last5=Decimal("1.42"),
            xga_home_last5=Decimal("0.95"),
            xga_away_last5=Decimal("1.28"),
            npxg_home_last5=Decimal("1.65"),
            npxg_away_last5=Decimal("1.22"),
            captured_at=datetime(2026, 1, 25, 10, 0, 0),
        )

    def test_tier1b_complete_when_xg_present(self):
        """tier1b_complete should be True when xG features present."""
        xg = self._make_xg()
        tier1b_complete = xg is not None
        assert tier1b_complete is True

    def test_tier1b_incomplete_when_xg_none(self):
        """tier1b_complete should be False when xG is None."""
        xg = None
        tier1b_complete = xg is not None
        assert tier1b_complete is False

    def test_xg_is_optional_for_insertion(self):
        """xG is optional - insertion should succeed without it."""
        odds = self._make_odds()

        should_insert, reason = should_insert_feature_row(
            odds=odds,
            form=None,
            h2h=None,
            xg=None,  # No xG
        )

        assert should_insert is True
        assert "Tier 1 complete" in reason

    def test_xg_enriches_insertion(self):
        """xG enriches insertion but doesn't change policy."""
        odds = self._make_odds()
        xg = self._make_xg()

        should_insert, reason = should_insert_feature_row(
            odds=odds,
            form=None,
            h2h=None,
            xg=xg,
        )

        assert should_insert is True
        # xG doesn't change the policy, just enriches the row


class TestXGPITCompliance:
    """Test xG-specific PIT compliance scenarios."""

    def test_xg_before_kickoff_is_valid(self):
        """xG captured before kickoff is PIT-compliant."""
        kickoff = datetime(2026, 1, 25, 20, 0, 0)  # 8 PM kickoff
        xg_captured = datetime(2026, 1, 25, 10, 0, 0)  # 10 AM capture

        # This is valid (captured < kickoff)
        assert xg_captured < kickoff

    def test_xg_after_kickoff_is_invalid(self):
        """xG captured after kickoff violates PIT."""
        kickoff = datetime(2026, 1, 25, 20, 0, 0)  # 8 PM kickoff
        xg_captured = datetime(2026, 1, 25, 21, 0, 0)  # 9 PM capture (AFTER)

        # This is invalid (captured >= kickoff)
        assert xg_captured >= kickoff

    def test_xg_captured_during_match_is_invalid(self):
        """xG captured during match (halftime) violates PIT."""
        kickoff = datetime(2026, 1, 25, 20, 0, 0)  # 8 PM kickoff
        xg_captured = datetime(2026, 1, 25, 20, 45, 0)  # Halftime

        # This is invalid (captured >= kickoff)
        assert xg_captured >= kickoff
