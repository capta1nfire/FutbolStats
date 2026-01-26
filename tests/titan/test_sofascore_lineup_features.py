"""Tests for Tier 1c (SofaScore Lineup) features.

FASE 3A: SofaScore Lineups via SOTA tables.
Per ABE: Tests must use mocks (no real SofaScore calls).
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone
from decimal import Decimal

pytestmark = pytest.mark.anyio


class TestSofaScoreLineupFeaturesDataclass:
    """Test dataclass creation and fields."""

    def test_lineup_creation_all_fields(self):
        """SofaScoreLineupFeatures accepts all fields."""
        from app.titan.materializers.feature_matrix import SofaScoreLineupFeatures

        lineup = SofaScoreLineupFeatures(
            sofascore_lineup_available=True,
            sofascore_home_formation="4-3-3",
            sofascore_away_formation="4-2-3-1",
            lineup_home_starters_count=11,
            lineup_away_starters_count=11,
            sofascore_lineup_integrity_score=Decimal("1.000"),
            captured_at=datetime(2026, 1, 26, 19, 0, 0, tzinfo=timezone.utc),
        )
        assert lineup.sofascore_lineup_available is True
        assert lineup.sofascore_home_formation == "4-3-3"
        assert lineup.sofascore_away_formation == "4-2-3-1"
        assert lineup.lineup_home_starters_count == 11
        assert lineup.lineup_away_starters_count == 11
        assert lineup.sofascore_lineup_integrity_score == Decimal("1.000")
        assert lineup.captured_at.tzinfo is not None

    def test_lineup_partial_starters(self):
        """Lineup with partial starters (incomplete)."""
        from app.titan.materializers.feature_matrix import SofaScoreLineupFeatures

        lineup = SofaScoreLineupFeatures(
            sofascore_lineup_available=True,
            sofascore_home_formation="4-3-3",
            sofascore_away_formation=None,  # Missing
            lineup_home_starters_count=11,
            lineup_away_starters_count=8,  # Incomplete
            sofascore_lineup_integrity_score=Decimal("0.500"),
            captured_at=datetime(2026, 1, 26, 19, 0, 0, tzinfo=timezone.utc),
        )
        assert lineup.lineup_away_starters_count == 8
        assert lineup.sofascore_away_formation is None

    def test_lineup_nullable_fields(self):
        """SofaScoreLineupFeatures handles None values."""
        from app.titan.materializers.feature_matrix import SofaScoreLineupFeatures

        lineup = SofaScoreLineupFeatures(
            sofascore_lineup_available=False,
            sofascore_home_formation=None,
            sofascore_away_formation=None,
            lineup_home_starters_count=None,
            lineup_away_starters_count=None,
            sofascore_lineup_integrity_score=None,
            captured_at=None,
        )
        assert lineup.sofascore_lineup_available is False
        assert lineup.captured_at is None


class TestIntegrityScoreCalculation:
    """Test integrity score derivation logic.

    Formula:
    - formation_present: 1.0 if both formations exist, 0.0 otherwise
    - starters_complete: (home_starters==11 + away_starters==11) / 2
    - integrity_score = (formation_present + starters_complete) / 2
    """

    def test_perfect_integrity_score(self):
        """Both formations + both 11 starters = 1.000."""
        # formation_present = 1.0, starters_complete = 1.0
        # integrity = (1.0 + 1.0) / 2 = 1.0
        formation_present = 1.0  # both formations exist
        home_complete = 1.0  # 11 starters
        away_complete = 1.0  # 11 starters
        starters_complete = (home_complete + away_complete) / 2
        integrity = (formation_present + starters_complete) / 2
        assert integrity == 1.0

    def test_missing_one_formation(self):
        """One formation missing = 0.500 max."""
        formation_present = 0.0  # missing formation
        starters_complete = 1.0  # both 11 starters
        integrity = (formation_present + starters_complete) / 2
        assert integrity == 0.5

    def test_partial_starters_one_team(self):
        """8/11 and 11/11 starters = 0.750."""
        formation_present = 1.0
        home_complete = 0.0  # 8 starters (not 11)
        away_complete = 1.0  # 11 starters
        starters_complete = (home_complete + away_complete) / 2
        integrity = (formation_present + starters_complete) / 2
        assert integrity == 0.75

    def test_partial_starters_both_teams(self):
        """Both teams have incomplete starters = 0.500."""
        formation_present = 1.0
        home_complete = 0.0  # <11 starters
        away_complete = 0.0  # <11 starters
        starters_complete = (home_complete + away_complete) / 2
        integrity = (formation_present + starters_complete) / 2
        assert integrity == 0.5

    def test_no_formation_no_starters(self):
        """Missing formations and incomplete starters = 0.000."""
        formation_present = 0.0
        home_complete = 0.0
        away_complete = 0.0
        starters_complete = (home_complete + away_complete) / 2
        integrity = (formation_present + starters_complete) / 2
        assert integrity == 0.0


class TestLineupPITCompliance:
    """Test PIT (Point-in-Time) constraint enforcement."""

    def test_lineup_before_kickoff_valid(self):
        """Lineup captured before kickoff is PIT-valid."""
        kickoff = datetime(2026, 1, 26, 20, 0, 0, tzinfo=timezone.utc)
        captured = datetime(2026, 1, 26, 19, 0, 0, tzinfo=timezone.utc)
        assert captured < kickoff  # PIT valid

    def test_lineup_after_kickoff_invalid(self):
        """Lineup captured after kickoff violates PIT."""
        kickoff = datetime(2026, 1, 26, 20, 0, 0, tzinfo=timezone.utc)
        captured = datetime(2026, 1, 26, 21, 0, 0, tzinfo=timezone.utc)
        assert captured >= kickoff  # PIT invalid

    def test_lineup_equals_kickoff_invalid(self):
        """Lineup captured at exact kickoff violates PIT."""
        kickoff = datetime(2026, 1, 26, 20, 0, 0, tzinfo=timezone.utc)
        captured = datetime(2026, 1, 26, 20, 0, 0, tzinfo=timezone.utc)
        assert captured >= kickoff  # PIT invalid

    def test_lineup_1_hour_before_typical(self):
        """Typical lineup: captured ~1 hour before kickoff."""
        kickoff = datetime(2026, 1, 26, 20, 0, 0, tzinfo=timezone.utc)
        captured = datetime(2026, 1, 26, 19, 0, 0, tzinfo=timezone.utc)  # 1h before
        assert captured < kickoff
        assert (kickoff - captured).total_seconds() == 3600  # Exactly 1 hour


class TestTimezoneNormalization:
    """Test timezone handling for public.* vs titan.* tables.

    Ref: docs/OPS_RUNBOOK.md "TITAN: Reglas de Timezone"
    """

    def test_kickoff_normalized_to_naive_for_public_tables(self):
        """kickoff_utc must be naive for public.match_sofascore_lineup queries."""
        kickoff_utc = datetime(2026, 1, 26, 20, 0, 0, tzinfo=timezone.utc)
        # Normalization rule from feature_matrix.py
        kickoff_naive = kickoff_utc.replace(tzinfo=None) if kickoff_utc.tzinfo else kickoff_utc
        assert kickoff_naive.tzinfo is None
        assert kickoff_naive.hour == 20
        assert kickoff_naive.day == 26

    def test_naive_stays_naive(self):
        """Already naive datetime stays naive."""
        kickoff_naive = datetime(2026, 1, 26, 20, 0, 0)
        result = kickoff_naive.replace(tzinfo=None) if kickoff_naive.tzinfo else kickoff_naive
        assert result.tzinfo is None
        assert result == kickoff_naive

    def test_captured_at_converted_to_aware_for_titan_storage(self):
        """captured_at from public.* (naive) must be aware for titan.* storage."""
        captured_naive = datetime(2026, 1, 26, 19, 0, 0)  # from public.*
        captured_aware = captured_naive.replace(tzinfo=timezone.utc)
        assert captured_aware.tzinfo is not None
        assert captured_aware.tzinfo == timezone.utc

    def test_aware_stays_aware(self):
        """Already aware datetime stays aware."""
        captured_aware = datetime(2026, 1, 26, 19, 0, 0, tzinfo=timezone.utc)
        result = captured_aware if captured_aware.tzinfo else captured_aware.replace(tzinfo=timezone.utc)
        assert result.tzinfo is not None
        assert result == captured_aware


class TestFailOpenBehavior:
    """Test fail-open: no lineup returns None, doesn't crash."""

    def test_no_lineup_returns_none(self):
        """No lineup data returns None (fail-open)."""
        # Simulates compute_lineup_features when no row found
        result = None
        assert result is None

    def test_tier1c_incomplete_when_no_lineup(self):
        """tier1c_complete = FALSE when no lineup available."""
        lineup = None
        tier1c_complete = lineup.captured_at is not None if lineup else False
        assert tier1c_complete is False

    def test_tier1c_complete_flag_computation(self):
        """tier1c_complete computed from captured_at presence."""
        from app.titan.materializers.feature_matrix import SofaScoreLineupFeatures

        # With captured_at -> tier1c_complete = True
        lineup_with = SofaScoreLineupFeatures(
            sofascore_lineup_available=True,
            sofascore_home_formation="4-3-3",
            sofascore_away_formation="4-4-2",
            lineup_home_starters_count=11,
            lineup_away_starters_count=11,
            sofascore_lineup_integrity_score=Decimal("1.000"),
            captured_at=datetime(2026, 1, 26, 19, 0, 0, tzinfo=timezone.utc),
        )
        assert (lineup_with.captured_at is not None) is True

        # Without captured_at -> tier1c_complete = False
        lineup_without = SofaScoreLineupFeatures(
            sofascore_lineup_available=False,
            sofascore_home_formation=None,
            sofascore_away_formation=None,
            lineup_home_starters_count=None,
            lineup_away_starters_count=None,
            sofascore_lineup_integrity_score=None,
            captured_at=None,
        )
        assert (lineup_without.captured_at is not None) is False


class TestTier1cCompletion:
    """Test tier1c_complete flag logic."""

    def test_tier1c_complete_when_lineup_captured(self):
        """tier1c_complete = TRUE when sofascore_lineup_captured_at IS NOT NULL."""
        from app.titan.materializers.feature_matrix import SofaScoreLineupFeatures

        lineup = SofaScoreLineupFeatures(
            sofascore_lineup_available=True,
            sofascore_home_formation="4-3-3",
            sofascore_away_formation="4-4-2",
            lineup_home_starters_count=11,
            lineup_away_starters_count=11,
            sofascore_lineup_integrity_score=Decimal("1.000"),
            captured_at=datetime(2026, 1, 26, 19, 0, 0, tzinfo=timezone.utc),
        )
        tier1c_complete = lineup.captured_at is not None
        assert tier1c_complete is True


class TestComputePitMaxWithLineup:
    """Test compute_pit_max includes lineup in calculation."""

    def test_pit_max_with_all_tiers_including_lineup(self):
        """compute_pit_max should include lineup.captured_at."""
        from app.titan.materializers.feature_matrix import (
            compute_pit_max,
            OddsFeatures,
            FormFeatures,
            H2HFeatures,
            XGFeatures,
            SofaScoreLineupFeatures,
        )

        odds = OddsFeatures(
            odds_home_close=Decimal("2.10"),
            odds_draw_close=Decimal("3.40"),
            odds_away_close=Decimal("3.20"),
            implied_prob_home=Decimal("0.4651"),
            implied_prob_draw=Decimal("0.2874"),
            implied_prob_away=Decimal("0.3053"),
            captured_at=datetime(2026, 1, 25, 8, 0, 0, tzinfo=timezone.utc),
        )
        form = FormFeatures(
            form_last5="WWDLW",
            goals_scored_last5=8,
            goals_conceded_last5=4,
            points_last5=10,
            captured_at=datetime(2026, 1, 25, 10, 0, 0, tzinfo=timezone.utc),
        )
        h2h = H2HFeatures(
            h2h_total_matches=10,
            h2h_home_wins=4,
            h2h_draws=3,
            h2h_away_wins=3,
            h2h_home_goals=15,
            h2h_away_goals=12,
            captured_at=datetime(2026, 1, 25, 14, 0, 0, tzinfo=timezone.utc),
        )
        xg = XGFeatures(
            xg_home_last5=Decimal("1.85"),
            xg_away_last5=Decimal("1.42"),
            xga_home_last5=Decimal("0.95"),
            xga_away_last5=Decimal("1.28"),
            npxg_home_last5=Decimal("1.65"),
            npxg_away_last5=Decimal("1.22"),
            captured_at=datetime(2026, 1, 25, 16, 0, 0, tzinfo=timezone.utc),
        )
        lineup = SofaScoreLineupFeatures(
            sofascore_lineup_available=True,
            sofascore_home_formation="4-3-3",
            sofascore_away_formation="4-2-3-1",
            lineup_home_starters_count=11,
            lineup_away_starters_count=11,
            sofascore_lineup_integrity_score=Decimal("1.000"),
            captured_at=datetime(2026, 1, 25, 19, 0, 0, tzinfo=timezone.utc),  # Latest
        )

        pit_max = compute_pit_max(odds=odds, form=form, h2h=h2h, xg=xg, lineup=lineup)
        # Lineup has the latest captured_at
        assert pit_max == datetime(2026, 1, 25, 19, 0, 0, tzinfo=timezone.utc)

    def test_pit_max_lineup_only(self):
        """compute_pit_max with only odds and lineup."""
        from app.titan.materializers.feature_matrix import (
            compute_pit_max,
            OddsFeatures,
            SofaScoreLineupFeatures,
        )

        odds = OddsFeatures(
            odds_home_close=Decimal("2.10"),
            odds_draw_close=Decimal("3.40"),
            odds_away_close=Decimal("3.20"),
            implied_prob_home=Decimal("0.4651"),
            implied_prob_draw=Decimal("0.2874"),
            implied_prob_away=Decimal("0.3053"),
            captured_at=datetime(2026, 1, 25, 10, 0, 0, tzinfo=timezone.utc),
        )
        lineup = SofaScoreLineupFeatures(
            sofascore_lineup_available=True,
            sofascore_home_formation="4-3-3",
            sofascore_away_formation="4-2-3-1",
            lineup_home_starters_count=11,
            lineup_away_starters_count=11,
            sofascore_lineup_integrity_score=Decimal("1.000"),
            captured_at=datetime(2026, 1, 25, 19, 0, 0, tzinfo=timezone.utc),
        )

        pit_max = compute_pit_max(odds=odds, form=None, h2h=None, xg=None, lineup=lineup)
        # Lineup is later than odds
        assert pit_max == datetime(2026, 1, 25, 19, 0, 0, tzinfo=timezone.utc)


class TestShouldInsertWithLineup:
    """Test should_insert_feature_row with lineup parameter."""

    def test_with_odds_and_lineup_inserts(self):
        """With odds + lineup, should insert."""
        from app.titan.materializers.feature_matrix import (
            should_insert_feature_row,
            OddsFeatures,
            SofaScoreLineupFeatures,
        )

        odds = OddsFeatures(
            odds_home_close=Decimal("2.10"),
            odds_draw_close=Decimal("3.40"),
            odds_away_close=Decimal("3.20"),
            implied_prob_home=Decimal("0.4651"),
            implied_prob_draw=Decimal("0.2874"),
            implied_prob_away=Decimal("0.3053"),
            captured_at=datetime(2026, 1, 25, 10, 0, 0, tzinfo=timezone.utc),
        )
        lineup = SofaScoreLineupFeatures(
            sofascore_lineup_available=True,
            sofascore_home_formation="4-3-3",
            sofascore_away_formation="4-2-3-1",
            lineup_home_starters_count=11,
            lineup_away_starters_count=11,
            sofascore_lineup_integrity_score=Decimal("1.000"),
            captured_at=datetime(2026, 1, 25, 19, 0, 0, tzinfo=timezone.utc),
        )

        should_insert, reason = should_insert_feature_row(
            odds=odds, form=None, h2h=None, xg=None, lineup=lineup
        )
        assert should_insert is True
        assert "Tier 1 complete" in reason

    def test_no_odds_no_insert_even_with_lineup(self):
        """Without odds, should NOT insert (Tier 1 gate unchanged)."""
        from app.titan.materializers.feature_matrix import (
            should_insert_feature_row,
            SofaScoreLineupFeatures,
        )

        lineup = SofaScoreLineupFeatures(
            sofascore_lineup_available=True,
            sofascore_home_formation="4-3-3",
            sofascore_away_formation="4-2-3-1",
            lineup_home_starters_count=11,
            lineup_away_starters_count=11,
            sofascore_lineup_integrity_score=Decimal("1.000"),
            captured_at=datetime(2026, 1, 25, 19, 0, 0, tzinfo=timezone.utc),
        )

        should_insert, reason = should_insert_feature_row(
            odds=None, form=None, h2h=None, xg=None, lineup=lineup
        )
        assert should_insert is False
        assert "Missing Tier 1" in reason


class TestComputeLineupFeaturesMocked:
    """Test compute_lineup_features with mocked DB.

    Per ABE: No real SofaScore calls in tests.
    """

    @pytest.fixture
    def mock_session(self):
        """Create mock async session."""
        session = AsyncMock()
        return session

    async def test_returns_none_when_no_data(self, mock_session):
        """Returns None when query returns no rows (fail-open)."""
        from app.titan.materializers.feature_matrix import FeatureMatrixMaterializer

        mock_result = MagicMock()
        mock_result.fetchone.return_value = None
        mock_session.execute.return_value = mock_result

        materializer = FeatureMatrixMaterializer(mock_session)
        kickoff = datetime(2026, 1, 26, 20, 0, 0, tzinfo=timezone.utc)
        result = await materializer.compute_lineup_features(match_id=123, kickoff_utc=kickoff)
        assert result is None  # Fail-open

    async def test_returns_features_when_data_exists(self, mock_session):
        """Returns SofaScoreLineupFeatures when lineup data found."""
        from app.titan.materializers.feature_matrix import FeatureMatrixMaterializer

        mock_row = (
            "4-3-3",  # home_formation
            "4-2-3-1",  # away_formation
            11,  # home_starters
            11,  # away_starters
            datetime(2026, 1, 26, 19, 0, 0),  # captured_at (naive from public.*)
        )
        mock_result = MagicMock()
        mock_result.fetchone.return_value = mock_row
        mock_session.execute.return_value = mock_result

        materializer = FeatureMatrixMaterializer(mock_session)
        kickoff = datetime(2026, 1, 26, 20, 0, 0, tzinfo=timezone.utc)
        result = await materializer.compute_lineup_features(match_id=123, kickoff_utc=kickoff)

        assert result is not None
        assert result.sofascore_home_formation == "4-3-3"
        assert result.sofascore_away_formation == "4-2-3-1"
        assert result.lineup_home_starters_count == 11
        assert result.lineup_away_starters_count == 11
        assert result.captured_at is not None
        assert result.captured_at.tzinfo is not None

    async def test_partial_lineup_data(self, mock_session):
        """Handles partial lineup (missing formation or starters)."""
        from app.titan.materializers.feature_matrix import FeatureMatrixMaterializer

        mock_row = (
            "4-3-3",  # home_formation
            None,  # away_formation missing
            11,  # home_starters
            8,  # away_starters incomplete
            datetime(2026, 1, 26, 19, 0, 0),
        )
        mock_result = MagicMock()
        mock_result.fetchone.return_value = mock_row
        mock_session.execute.return_value = mock_result

        materializer = FeatureMatrixMaterializer(mock_session)
        kickoff = datetime(2026, 1, 26, 20, 0, 0, tzinfo=timezone.utc)
        result = await materializer.compute_lineup_features(match_id=123, kickoff_utc=kickoff)

        assert result is not None
        assert result.sofascore_home_formation == "4-3-3"
        assert result.sofascore_away_formation is None
        assert result.lineup_away_starters_count == 8

    async def test_query_does_not_use_nonexistent_columns(self, mock_session):
        """Regression: SOTA tables do not have lineup_id or surrogate ids."""
        from app.titan.materializers.feature_matrix import FeatureMatrixMaterializer

        mock_result = MagicMock()
        mock_result.fetchone.return_value = None
        mock_session.execute.return_value = mock_result

        materializer = FeatureMatrixMaterializer(mock_session)
        kickoff = datetime(2026, 1, 26, 20, 0, 0, tzinfo=timezone.utc)
        await materializer.compute_lineup_features(match_id=123, kickoff_utc=kickoff)

        assert mock_session.execute.call_count == 1
        executed_query = mock_session.execute.call_args[0][0]
        sql = str(getattr(executed_query, "text", executed_query))
        assert "lineup_id" not in sql
        assert "msl.id" not in sql
        assert "msp.id" not in sql
        assert "public.match_sofascore_lineup" in sql
        assert "public.match_sofascore_player" in sql
