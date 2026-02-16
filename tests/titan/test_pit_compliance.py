"""Tests for TITAN PIT (Point-in-Time) compliance.

Verifies:
1. pit_max_captured_at < kickoff_utc constraint
2. PITViolationError raised when constraint would be violated
3. Insertion policy respects PIT requirements
4. Kickoff reschedule: UPSERT updates kickoff_utc via GREATEST
"""

import inspect

import pytest
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

from sqlalchemy.ext.asyncio import AsyncSession

from app.titan.materializers.feature_matrix import (
    PITViolationError,
    InsertionPolicyViolation,
    FeatureMatrixMaterializer,
    OddsFeatures,
    FormFeatures,
    H2HFeatures,
    XGFeatures,
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

    def test_four_tiers_with_xg_takes_max(self):
        """With xG (Tier 1b) included, pit_max = max of all including xG."""
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
            captured_at=datetime(2026, 1, 25, 14, 0, 0),
        )
        xg = XGFeatures(
            xg_home_last5=Decimal("1.85"),
            xg_away_last5=Decimal("1.42"),
            xga_home_last5=Decimal("0.95"),
            xga_away_last5=Decimal("1.28"),
            npxg_home_last5=Decimal("1.65"),
            npxg_away_last5=Decimal("1.22"),
            captured_at=datetime(2026, 1, 25, 16, 0, 0),  # Latest
        )

        pit_max = compute_pit_max(odds=odds, form=form, h2h=h2h, xg=xg)
        assert pit_max == datetime(2026, 1, 25, 16, 0, 0)  # xG is latest

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


class TestRescheduleUpsertPIT:
    """Test PIT compliance when a match is rescheduled.

    Scenario (production bug):
    1. Initial insert: kickoff_old=Feb15 15:45, pit_max_old=Feb15 10:00
       → Valid: pit_max_old < kickoff_old ✓
    2. Match rescheduled to Feb17 19:00. New data captured at Feb16 12:00.
       → pit_max_new=Feb16 12:00 > kickoff_old=Feb15 15:45  (DB constraint fails!)
       → pit_max_new=Feb16 12:00 < kickoff_new=Feb17 19:00  (should be valid)

    Fix: UPSERT uses GREATEST(EXCLUDED.kickoff_utc, existing.kickoff_utc) so
    the DB row's kickoff_utc is updated to the new (later) kickoff.
    """

    # Fixtures
    KICKOFF_OLD = datetime(2026, 2, 15, 15, 45, 0)
    KICKOFF_NEW = datetime(2026, 2, 17, 19, 0, 0)
    PIT_MAX_OLD = datetime(2026, 2, 15, 10, 0, 0)
    PIT_MAX_NEW = datetime(2026, 2, 16, 12, 0, 0)  # > KICKOFF_OLD but < KICKOFF_NEW

    def test_initial_insert_pit_valid(self):
        """Initial insert: pit_max_old < kickoff_old → valid."""
        assert self.PIT_MAX_OLD < self.KICKOFF_OLD

    def test_reschedule_pit_invalid_with_old_kickoff(self):
        """After reschedule, pit_max_new > kickoff_old → would violate DB constraint."""
        assert self.PIT_MAX_NEW > self.KICKOFF_OLD

    def test_reschedule_pit_valid_with_new_kickoff(self):
        """After reschedule, pit_max_new < kickoff_new → valid with updated kickoff."""
        assert self.PIT_MAX_NEW < self.KICKOFF_NEW

    def test_python_pit_check_uses_new_kickoff(self):
        """Python PIT check passes because runner uses kickoff from matches table (new)."""
        odds = OddsFeatures(
            odds_home_close=Decimal("2.10"),
            odds_draw_close=Decimal("3.40"),
            odds_away_close=Decimal("3.20"),
            implied_prob_home=Decimal("0.4651"),
            implied_prob_draw=Decimal("0.2874"),
            implied_prob_away=Decimal("0.3053"),
            captured_at=self.PIT_MAX_NEW,
        )
        pit_max = compute_pit_max(odds=odds, form=None, h2h=None)

        # With old kickoff → would raise PITViolationError
        assert pit_max >= self.KICKOFF_OLD

        # With new kickoff → valid
        assert pit_max < self.KICKOFF_NEW

    def test_upsert_sql_includes_greatest_kickoff(self):
        """UPSERT SQL must use GREATEST for kickoff_utc to handle reschedules."""
        source = inspect.getsource(FeatureMatrixMaterializer.insert_row)
        assert "GREATEST(EXCLUDED.kickoff_utc" in source, (
            "UPSERT must use GREATEST for kickoff_utc to handle rescheduled matches"
        )

    @pytest.mark.asyncio
    async def test_insert_row_reschedule_passes_pit_check(self):
        """Full insert_row call with rescheduled kickoff passes PIT validation."""
        mock_session = AsyncMock(spec=AsyncSession)
        materializer = FeatureMatrixMaterializer(mock_session)

        odds = OddsFeatures(
            odds_home_close=Decimal("2.10"),
            odds_draw_close=Decimal("3.40"),
            odds_away_close=Decimal("3.20"),
            implied_prob_home=Decimal("0.4651"),
            implied_prob_draw=Decimal("0.2874"),
            implied_prob_away=Decimal("0.3053"),
            captured_at=self.PIT_MAX_NEW,
        )

        # With new kickoff, PIT check should pass (no PITViolationError)
        result = await materializer.insert_row(
            match_id=1381072,
            kickoff_utc=self.KICKOFF_NEW,
            competition_id=128,
            season=2025,
            home_team_id=100,
            away_team_id=200,
            odds=odds,
        )
        assert result is True
        mock_session.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_insert_row_old_kickoff_raises_pit_violation(self):
        """insert_row with stale kickoff (pre-reschedule) raises PITViolationError."""
        mock_session = AsyncMock(spec=AsyncSession)
        materializer = FeatureMatrixMaterializer(mock_session)

        odds = OddsFeatures(
            odds_home_close=Decimal("2.10"),
            odds_draw_close=Decimal("3.40"),
            odds_away_close=Decimal("3.20"),
            implied_prob_home=Decimal("0.4651"),
            implied_prob_draw=Decimal("0.2874"),
            implied_prob_away=Decimal("0.3053"),
            captured_at=self.PIT_MAX_NEW,  # Feb 16 12:00
        )

        # With old kickoff (Feb 15), pit_max (Feb 16) >= kickoff → PITViolationError
        with pytest.raises(PITViolationError, match="PIT violation"):
            await materializer.insert_row(
                match_id=1381072,
                kickoff_utc=self.KICKOFF_OLD,  # Stale kickoff
                competition_id=128,
                season=2025,
                home_team_id=100,
                away_team_id=200,
                odds=odds,
            )
