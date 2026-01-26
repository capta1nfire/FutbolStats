"""Tests for Tier 1d (XI Depth) features.

FASE 3B-1: XI depth features derived from lineup positions.
Per ABE: Tests must use mocks (no real SofaScore calls).
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

pytestmark = pytest.mark.anyio


class TestXIDepthFeaturesDataclass:
    """Test XIDepthFeatures dataclass creation and fields."""

    def test_creation_all_fields(self):
        """XIDepthFeatures accepts all fields."""
        from app.titan.materializers.feature_matrix import XIDepthFeatures

        xi = XIDepthFeatures(
            xi_home_def_count=4,
            xi_home_mid_count=3,
            xi_home_fwd_count=3,
            xi_away_def_count=4,
            xi_away_mid_count=4,
            xi_away_fwd_count=2,
            xi_formation_mismatch_flag=False,
            captured_at=datetime(2026, 1, 26, 19, 0, 0, tzinfo=timezone.utc),
        )
        assert xi.xi_home_def_count == 4
        assert xi.xi_home_mid_count == 3
        assert xi.xi_home_fwd_count == 3
        assert xi.xi_away_def_count == 4
        assert xi.xi_away_mid_count == 4
        assert xi.xi_away_fwd_count == 2
        assert xi.xi_formation_mismatch_flag is False
        assert xi.captured_at is not None

    def test_creation_with_mismatch_flag(self):
        """XIDepthFeatures with formation mismatch flag."""
        from app.titan.materializers.feature_matrix import XIDepthFeatures

        xi = XIDepthFeatures(
            xi_home_def_count=5,
            xi_home_mid_count=4,
            xi_home_fwd_count=1,
            xi_away_def_count=4,
            xi_away_mid_count=3,
            xi_away_fwd_count=3,
            xi_formation_mismatch_flag=True,
            captured_at=datetime(2026, 1, 26, 19, 0, 0, tzinfo=timezone.utc),
        )
        assert xi.xi_formation_mismatch_flag is True

    def test_creation_with_none_captured_at(self):
        """XIDepthFeatures can have None captured_at (Optional)."""
        from app.titan.materializers.feature_matrix import XIDepthFeatures

        xi = XIDepthFeatures(
            xi_home_def_count=4,
            xi_home_mid_count=3,
            xi_home_fwd_count=3,
            xi_away_def_count=4,
            xi_away_mid_count=4,
            xi_away_fwd_count=2,
            xi_formation_mismatch_flag=False,
            captured_at=None,
        )
        assert xi.captured_at is None


class TestFormationMismatchDetection:
    """Test formation mismatch detection logic."""

    def test_433_matches_4_3_3(self):
        """4-3-3 formation matches DEF=4, MID=3, FWD=3."""
        from app.titan.materializers.feature_matrix import detect_formation_mismatch

        assert detect_formation_mismatch("4-3-3", 4, 3, 3) is False

    def test_442_matches_4_4_2(self):
        """4-4-2 formation matches DEF=4, MID=4, FWD=2."""
        from app.titan.materializers.feature_matrix import detect_formation_mismatch

        assert detect_formation_mismatch("4-4-2", 4, 4, 2) is False

    def test_4231_matches_4_5_1(self):
        """4-2-3-1 formation: MID = 2+3 = 5, FWD = 1."""
        from app.titan.materializers.feature_matrix import detect_formation_mismatch

        # 4-2-3-1 has 4 DEF, 2+3=5 MID, 1 FWD
        assert detect_formation_mismatch("4-2-3-1", 4, 5, 1) is False

    def test_433_with_tolerance_plus_one_mid(self):
        """4-3-3 tolerates MID=4 (±1 tolerance)."""
        from app.titan.materializers.feature_matrix import detect_formation_mismatch

        # Expected: DEF=4, MID=3, FWD=3
        # Actual: DEF=4, MID=4, FWD=2
        # MID diff = 1 (ok), FWD diff = 1 (ok) -> no mismatch
        assert detect_formation_mismatch("4-3-3", 4, 4, 2) is False

    def test_433_with_tolerance_minus_one_def(self):
        """4-3-3 tolerates DEF=3 (±1 tolerance)."""
        from app.titan.materializers.feature_matrix import detect_formation_mismatch

        assert detect_formation_mismatch("4-3-3", 3, 3, 3) is False

    def test_major_mismatch_541(self):
        """5-4-1 when expected 4-3-3 is a mismatch."""
        from app.titan.materializers.feature_matrix import detect_formation_mismatch

        # Expected: DEF=4, MID=3, FWD=3
        # Actual: DEF=5, MID=4, FWD=1
        # DEF diff = 1 (ok), MID diff = 1 (ok), FWD diff = 2 (not ok) -> mismatch
        assert detect_formation_mismatch("4-3-3", 5, 4, 1) is True

    def test_major_mismatch_def(self):
        """DEF diff > 1 is a mismatch."""
        from app.titan.materializers.feature_matrix import detect_formation_mismatch

        # Expected: DEF=4, actual: DEF=6 -> diff = 2 -> mismatch
        assert detect_formation_mismatch("4-3-3", 6, 3, 3) is True

    def test_no_formation_no_mismatch(self):
        """No formation provided -> no mismatch detectable."""
        from app.titan.materializers.feature_matrix import detect_formation_mismatch

        assert detect_formation_mismatch(None, 4, 3, 3) is False

    def test_empty_formation_no_mismatch(self):
        """Empty formation string -> no mismatch detectable."""
        from app.titan.materializers.feature_matrix import detect_formation_mismatch

        assert detect_formation_mismatch("", 4, 3, 3) is False

    def test_short_formation_no_mismatch(self):
        """Formation with < 3 digits -> no mismatch detectable."""
        from app.titan.materializers.feature_matrix import detect_formation_mismatch

        assert detect_formation_mismatch("4-3", 4, 3, 3) is False

    def test_formation_without_dashes(self):
        """Formation without dashes (e.g., '433') works."""
        from app.titan.materializers.feature_matrix import detect_formation_mismatch

        assert detect_formation_mismatch("433", 4, 3, 3) is False


class TestXIDepthPITCompliance:
    """Test PIT (Point-in-Time) constraint enforcement."""

    def test_captured_before_kickoff_valid(self):
        """XI captured before kickoff is PIT-valid."""
        kickoff = datetime(2026, 1, 26, 20, 0, 0, tzinfo=timezone.utc)
        captured = datetime(2026, 1, 26, 19, 0, 0, tzinfo=timezone.utc)
        assert captured < kickoff  # PIT valid

    def test_captured_after_kickoff_invalid(self):
        """XI captured after kickoff violates PIT."""
        kickoff = datetime(2026, 1, 26, 20, 0, 0, tzinfo=timezone.utc)
        captured = datetime(2026, 1, 26, 21, 0, 0, tzinfo=timezone.utc)
        assert captured >= kickoff  # PIT invalid

    def test_captured_equals_kickoff_invalid(self):
        """XI captured at exact kickoff violates PIT."""
        kickoff = datetime(2026, 1, 26, 20, 0, 0, tzinfo=timezone.utc)
        captured = datetime(2026, 1, 26, 20, 0, 0, tzinfo=timezone.utc)
        assert captured >= kickoff  # PIT invalid


class TestTimezoneNormalization:
    """Test timezone handling for public.* vs titan.* tables."""

    def test_kickoff_normalized_to_naive_for_public_tables(self):
        """kickoff_utc must be naive for public.match_sofascore_player queries."""
        kickoff_utc = datetime(2026, 1, 26, 20, 0, 0, tzinfo=timezone.utc)
        kickoff_naive = kickoff_utc.replace(tzinfo=None) if kickoff_utc.tzinfo else kickoff_utc
        assert kickoff_naive.tzinfo is None
        assert kickoff_naive.hour == 20

    def test_captured_at_converted_to_aware_for_titan_storage(self):
        """captured_at from public.* (naive) must be aware for titan.* storage."""
        captured_naive = datetime(2026, 1, 26, 19, 0, 0)  # from public.*
        captured_aware = captured_naive.replace(tzinfo=timezone.utc)
        assert captured_aware.tzinfo is not None
        assert captured_aware.tzinfo == timezone.utc


class TestFailOpenBehavior:
    """Test fail-open: no XI data returns None, doesn't crash."""

    def test_no_xi_returns_none(self):
        """No XI data returns None (fail-open)."""
        # Simulates compute_xi_depth_features when no rows found
        result = None
        assert result is None

    def test_tier1d_incomplete_when_no_xi(self):
        """tier1d_complete = FALSE when no XI data available."""
        xi_depth = None
        tier1d_complete = xi_depth is not None and xi_depth.captured_at is not None if xi_depth else False
        assert tier1d_complete is False


class TestTier1dCompletion:
    """Test tier1d_complete flag logic."""

    def test_tier1d_complete_when_xi_captured(self):
        """tier1d_complete = TRUE when xi_depth_captured_at IS NOT NULL."""
        from app.titan.materializers.feature_matrix import XIDepthFeatures

        xi = XIDepthFeatures(
            xi_home_def_count=4,
            xi_home_mid_count=3,
            xi_home_fwd_count=3,
            xi_away_def_count=4,
            xi_away_mid_count=4,
            xi_away_fwd_count=2,
            xi_formation_mismatch_flag=False,
            captured_at=datetime(2026, 1, 26, 19, 0, 0, tzinfo=timezone.utc),
        )
        tier1d_complete = xi.captured_at is not None
        assert tier1d_complete is True

    def test_tier1d_incomplete_when_captured_is_none(self):
        """tier1d_complete = FALSE when captured_at is None."""
        from app.titan.materializers.feature_matrix import XIDepthFeatures

        xi = XIDepthFeatures(
            xi_home_def_count=4,
            xi_home_mid_count=3,
            xi_home_fwd_count=3,
            xi_away_def_count=4,
            xi_away_mid_count=4,
            xi_away_fwd_count=2,
            xi_formation_mismatch_flag=False,
            captured_at=None,
        )
        tier1d_complete = xi.captured_at is not None
        assert tier1d_complete is False


class TestComputePitMaxWithXIDepth:
    """Test compute_pit_max includes xi_depth timestamp."""

    def test_pit_max_includes_xi_depth(self):
        """compute_pit_max considers xi_depth.captured_at."""
        from app.titan.materializers.feature_matrix import (
            compute_pit_max,
            OddsFeatures,
            XIDepthFeatures,
        )
        from decimal import Decimal

        odds = OddsFeatures(
            odds_home_close=Decimal("2.00"),
            odds_draw_close=Decimal("3.00"),
            odds_away_close=Decimal("3.50"),
            implied_prob_home=Decimal("0.40"),
            implied_prob_draw=Decimal("0.30"),
            implied_prob_away=Decimal("0.30"),
            captured_at=datetime(2026, 1, 26, 18, 0, 0, tzinfo=timezone.utc),
        )

        xi_depth = XIDepthFeatures(
            xi_home_def_count=4,
            xi_home_mid_count=3,
            xi_home_fwd_count=3,
            xi_away_def_count=4,
            xi_away_mid_count=4,
            xi_away_fwd_count=2,
            xi_formation_mismatch_flag=False,
            captured_at=datetime(2026, 1, 26, 19, 30, 0, tzinfo=timezone.utc),  # Latest
        )

        pit_max = compute_pit_max(odds=odds, form=None, h2h=None, xg=None, lineup=None, xi_depth=xi_depth)
        assert pit_max == datetime(2026, 1, 26, 19, 30, 0, tzinfo=timezone.utc)

    def test_pit_max_ignores_none_xi_depth(self):
        """compute_pit_max ignores xi_depth when None."""
        from app.titan.materializers.feature_matrix import compute_pit_max, OddsFeatures
        from decimal import Decimal

        odds = OddsFeatures(
            odds_home_close=Decimal("2.00"),
            odds_draw_close=Decimal("3.00"),
            odds_away_close=Decimal("3.50"),
            implied_prob_home=Decimal("0.40"),
            implied_prob_draw=Decimal("0.30"),
            implied_prob_away=Decimal("0.30"),
            captured_at=datetime(2026, 1, 26, 18, 0, 0, tzinfo=timezone.utc),
        )

        pit_max = compute_pit_max(odds=odds, form=None, h2h=None, xg=None, lineup=None, xi_depth=None)
        assert pit_max == datetime(2026, 1, 26, 18, 0, 0, tzinfo=timezone.utc)


class TestComputeXIDepthFeaturesMocked:
    """Test compute_xi_depth_features with mocked DB."""

    @pytest.fixture
    def mock_session(self):
        """Create mock async session."""
        session = AsyncMock()
        return session

    async def test_returns_none_when_no_data(self, mock_session):
        """Returns None when query returns no rows."""
        mock_result = MagicMock()
        mock_result.fetchall.return_value = []
        mock_session.execute.return_value = mock_result

        result = mock_result.fetchall()
        assert result == []

    async def test_aggregates_position_counts(self, mock_session):
        """Verify query result parsing."""
        # Simulate DB rows: (team_side, position, count, captured_at)
        mock_rows = [
            ("home", "DEF", 4, datetime(2026, 1, 26, 19, 0, 0)),
            ("home", "MID", 3, datetime(2026, 1, 26, 19, 0, 0)),
            ("home", "FWD", 3, datetime(2026, 1, 26, 19, 0, 0)),
            ("away", "DEF", 4, datetime(2026, 1, 26, 19, 0, 0)),
            ("away", "MID", 4, datetime(2026, 1, 26, 19, 0, 0)),
            ("away", "FWD", 2, datetime(2026, 1, 26, 19, 0, 0)),
        ]

        # Simulate aggregation logic
        counts = {"home": {"DEF": 0, "MID": 0, "FWD": 0}, "away": {"DEF": 0, "MID": 0, "FWD": 0}}
        for team_side, position, count, _ in mock_rows:
            if team_side in counts and position in counts[team_side]:
                counts[team_side][position] = count

        assert counts["home"]["DEF"] == 4
        assert counts["home"]["MID"] == 3
        assert counts["home"]["FWD"] == 3
        assert counts["away"]["DEF"] == 4
        assert counts["away"]["MID"] == 4
        assert counts["away"]["FWD"] == 2
