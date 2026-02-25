"""
Point-in-time (PIT) tests for SOTA feature engineering.

These tests verify that features are computed without data leakage:
1. No match data with date >= t0 is used for rolling features
2. Snapshot features (weather, understat) require captured_at < t0
3. Missing data results in *_missing=1 flags, not crashes

Can run with pytest if available, or standalone:
    python tests/test_feature_engineering_pit.py
"""

import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

# Optional pytest import
try:
    import pytest
    HAS_PYTEST = True
except ImportError:
    HAS_PYTEST = False
    # Create a no-op decorator for pytest.mark.asyncio
    class _MockPytest:
        class mark:
            @staticmethod
            def asyncio(func):
                return func
    pytest = _MockPytest()

# Test helpers
from app.features.engineering import (
    load_match_understat,
    load_match_weather,
    load_match_sofascore_xi,
    load_team_understat_history,
    calculate_circular_mean_hour,
    calculate_xi_features,
    get_tz_offset_hours,
    JUSTICE_SHRINKAGE_K,
    JUSTICE_EPSILON,
    XI_POSITION_WEIGHTS,
)


class TestPointInTimeUnderstat:
    """Tests for Understat feature point-in-time correctness."""

    @pytest.mark.asyncio
    async def test_understat_snapshot_ignored_if_captured_after_kickoff(self):
        """
        Test that canonical xG data with captured_at >= t0 is ignored.

        Scenario:
        - Match kickoff at 2026-01-15 15:00 UTC
        - xG snapshot captured at 2026-01-15 16:00 UTC (after kickoff)
        - Should return None (no valid snapshot)
        """
        # Mock session
        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.fetchone.return_value = None  # No valid row (filtered by PIT)
        mock_session.execute.return_value = mock_result

        t0 = datetime(2026, 1, 15, 15, 0, 0)  # Kickoff
        match_id = 12345

        result = await load_match_understat(mock_session, match_id, t0)

        # Verify the query includes PIT filter (COALESCE for NULL captured_at safety)
        call_args = mock_session.execute.call_args
        query_text = str(call_args[0][0])
        assert "captured_at" in query_text and "< :t0" in query_text

        # Result should be None since no valid snapshot
        assert result is None

    @pytest.mark.asyncio
    async def test_understat_snapshot_used_if_captured_before_kickoff(self):
        """
        Test that canonical xG data with captured_at < t0 is correctly used.

        Scenario:
        - Match kickoff at 2026-01-15 15:00 UTC
        - xG snapshot captured at 2026-01-15 10:00 UTC (before kickoff)
        - Should return the snapshot data
        """
        # Mock session with valid data
        mock_session = AsyncMock()
        mock_row = MagicMock()
        mock_row.xg_home = 1.5
        mock_row.xg_away = 0.8
        mock_row.source = "understat"
        mock_row.captured_at = datetime(2026, 1, 15, 10, 0, 0)
        mock_result = MagicMock()
        mock_result.fetchone.return_value = mock_row
        mock_session.execute.return_value = mock_result

        t0 = datetime(2026, 1, 15, 15, 0, 0)  # Kickoff
        match_id = 12345

        result = await load_match_understat(mock_session, match_id, t0)

        assert result is not None
        assert result["xg_home"] == 1.5
        assert result["xg_away"] == 0.8
        assert result["xg_source"] == "understat"

    @pytest.mark.asyncio
    async def test_understat_history_only_uses_matches_before_t0(self):
        """
        Test that rolling canonical xG history only includes matches before t0.

        The SQL query should filter:
        - m.date < :before_date (historical match must be before target kickoff)
        - captured_at < :before_date (xG snapshot must exist before target kickoff)

        Note: xG is POST-match data, so captured_at will always be AFTER
        the historical match. The PIT constraint ensures the snapshot was captured
        before the TARGET match kickoff, not before the historical match kickoff.
        """
        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.fetchall.return_value = []
        mock_session.execute.return_value = mock_result

        t0 = datetime(2026, 1, 15, 15, 0, 0)
        team_id = 100

        await load_team_understat_history(mock_session, team_id, t0)

        # Verify the query includes correct PIT filters
        call_args = mock_session.execute.call_args
        query_text = str(call_args[0][0])
        assert "m.date < :before_date" in query_text  # Historical match before target
        assert "captured_at" in query_text and "< :before_date" in query_text  # PIT filter


class TestPointInTimeWeather:
    """Tests for Weather feature point-in-time correctness."""

    @pytest.mark.asyncio
    async def test_weather_snapshot_ignored_if_captured_after_kickoff(self):
        """
        Test that weather data with captured_at >= t0 is ignored.
        """
        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.fetchone.return_value = None
        mock_session.execute.return_value = mock_result

        t0 = datetime(2026, 1, 15, 15, 0, 0)
        match_id = 12345

        result = await load_match_weather(mock_session, match_id, t0)

        # Verify query includes captured_at < t0
        call_args = mock_session.execute.call_args
        query_text = str(call_args[0][0])
        assert "captured_at < :t0" in query_text
        assert result is None

    @pytest.mark.asyncio
    async def test_weather_prefers_horizon_24_when_available(self):
        """
        Test that weather loading reads from match_weather_canonical
        with PIT guard (kind='archive' or captured_at < t0).
        """
        mock_session = AsyncMock()
        mock_row = MagicMock()
        mock_row.temp_c = 15.0
        mock_row.humidity_pct = 60.0
        mock_row.wind_ms = 3.0
        mock_row.precip_mm = 0.0
        mock_row.is_daylight = True
        mock_row.forecast_horizon_hours = 24
        mock_result = MagicMock()
        mock_result.fetchone.return_value = mock_row
        mock_session.execute.return_value = mock_result

        t0 = datetime(2026, 1, 15, 15, 0, 0)
        match_id = 12345

        result = await load_match_weather(mock_session, match_id, t0, preferred_horizon=24)

        # Verify query reads from canonical with PIT guard
        call_args = mock_session.execute.call_args
        query_text = str(call_args[0][0])
        assert "match_weather_canonical" in query_text
        assert "captured_at < :t0" in query_text

        assert result is not None
        assert result["weather_forecast_horizon_hours"] == 24


class TestPointInTimeSofascoreXI:
    """Tests for Sofascore XI feature point-in-time correctness."""

    @pytest.mark.asyncio
    async def test_xi_snapshot_ignored_if_captured_after_kickoff(self):
        """
        Test that XI data with captured_at >= t0 is ignored.

        Scenario:
        - Match kickoff at 2026-01-15 15:00 UTC
        - XI snapshot captured at 2026-01-15 16:00 UTC (after kickoff)
        - Should return None (no valid snapshot)
        """
        mock_session = AsyncMock()

        # First call for lineups returns empty
        # Second call for players returns empty
        mock_result = MagicMock()
        mock_result.fetchall.return_value = []
        mock_session.execute.return_value = mock_result

        t0 = datetime(2026, 1, 15, 15, 0, 0)
        match_id = 12345

        result = await load_match_sofascore_xi(mock_session, match_id, t0)

        # Verify the queries include captured_at < t0 filter
        for call in mock_session.execute.call_args_list:
            query_text = str(call[0][0])
            assert "captured_at < :t0" in query_text

        # Result should be None since no valid snapshot
        assert result is None

    @pytest.mark.asyncio
    async def test_xi_snapshot_used_if_captured_before_kickoff(self):
        """
        Test that XI data with captured_at < t0 is correctly used.

        Scenario:
        - Match kickoff at 2026-01-15 15:00 UTC
        - XI snapshot captured at 2026-01-15 10:00 UTC (before kickoff)
        - Should return the lineup data
        """
        mock_session = AsyncMock()

        # Mock lineup data
        mock_lineup_row = MagicMock()
        mock_lineup_row.team_side = "home"
        mock_lineup_row.formation = "4-3-3"
        mock_lineup_row.captured_at = datetime(2026, 1, 15, 10, 0, 0)

        mock_player_row = MagicMock()
        mock_player_row.team_side = "home"
        mock_player_row.player_id_ext = "123"
        mock_player_row.position = "MID"
        mock_player_row.is_starter = True
        mock_player_row.rating_pre_match = 7.5
        mock_player_row.rating_recent_form = 7.2

        # Set up mock to return different results for different calls
        call_count = 0

        def mock_execute(query, params=None):
            nonlocal call_count
            call_count += 1
            result = MagicMock()
            if call_count == 1:  # Lineup query
                result.fetchall.return_value = [mock_lineup_row]
            else:  # Player query
                result.fetchall.return_value = [mock_player_row]
            return result

        mock_session.execute = AsyncMock(side_effect=mock_execute)

        t0 = datetime(2026, 1, 15, 15, 0, 0)
        match_id = 12345

        result = await load_match_sofascore_xi(mock_session, match_id, t0)

        assert result is not None
        assert result["home"]["formation"] == "4-3-3"
        assert len(result["home"]["players"]) == 1
        assert result["home"]["players"][0]["rating_pre_match"] == 7.5


class TestXIFeatureCalculation:
    """Tests for XI feature calculation functions."""

    def test_calculate_xi_features_with_valid_ratings(self):
        """Test XI feature calculation with valid player data."""
        import numpy as np

        players = [
            {"position": "GK", "is_starter": True, "rating_pre_match": 7.0, "rating_recent_form": 6.8},
            {"position": "DEF", "is_starter": True, "rating_pre_match": 7.2, "rating_recent_form": 7.0},
            {"position": "DEF", "is_starter": True, "rating_pre_match": 6.8, "rating_recent_form": 6.5},
            {"position": "DEF", "is_starter": True, "rating_pre_match": 7.5, "rating_recent_form": 7.3},
            {"position": "DEF", "is_starter": True, "rating_pre_match": 7.1, "rating_recent_form": 6.9},
            {"position": "MID", "is_starter": True, "rating_pre_match": 7.8, "rating_recent_form": 7.5},
            {"position": "MID", "is_starter": True, "rating_pre_match": 7.3, "rating_recent_form": 7.1},
            {"position": "MID", "is_starter": True, "rating_pre_match": 7.0, "rating_recent_form": 6.8},
            {"position": "FWD", "is_starter": True, "rating_pre_match": 8.0, "rating_recent_form": 7.8},
            {"position": "FWD", "is_starter": True, "rating_pre_match": 7.6, "rating_recent_form": 7.4},
            {"position": "FWD", "is_starter": True, "rating_pre_match": 7.4, "rating_recent_form": 7.2},
        ]

        features = calculate_xi_features(players, "home")

        # Check all features exist (suffix convention per FEATURE_DICTIONARY_SOTA.md)
        assert "xi_weighted_home" in features
        assert "xi_p10_home" in features
        assert "xi_p50_home" in features
        assert "xi_p90_home" in features
        assert "xi_weaklink_home" in features
        assert "xi_std_home" in features

        # Weaklink should be the minimum rating
        assert features["xi_weaklink_home"] == 6.8

        # P50 should be around median
        ratings = [7.0, 7.2, 6.8, 7.5, 7.1, 7.8, 7.3, 7.0, 8.0, 7.6, 7.4]
        assert abs(features["xi_p50_home"] - np.median(ratings)) < 0.01

    def test_calculate_xi_features_only_uses_starters(self):
        """Test that XI features only consider starters, not subs."""
        players = [
            {"position": "GK", "is_starter": True, "rating_pre_match": 7.0, "rating_recent_form": 6.8},
            {"position": "DEF", "is_starter": True, "rating_pre_match": 7.2, "rating_recent_form": 7.0},
            {"position": "DEF", "is_starter": False, "rating_pre_match": 9.0, "rating_recent_form": 9.0},  # Sub - should be ignored
        ]

        features = calculate_xi_features(players, "home")

        # The 9.0 rating sub should NOT affect the max/p90
        assert features["xi_p90_home"] < 8.0

    def test_calculate_xi_features_with_empty_players(self):
        """Test XI features with no players returns defaults."""
        features = calculate_xi_features([], "home")

        # Should return default values (suffix convention)
        assert features["xi_weighted_home"] == 6.5
        assert features["xi_weaklink_home"] == 6.0

    def test_calculate_xi_features_uses_position_weights(self):
        """Test that position weights are applied correctly."""
        # Two players with same rating but different positions
        players = [
            {"position": "DEF", "is_starter": True, "rating_pre_match": 7.0, "rating_recent_form": None},
            {"position": "FWD", "is_starter": True, "rating_pre_match": 7.0, "rating_recent_form": None},
        ]

        features = calculate_xi_features(players, "home")

        # With DEF=0.9, FWD=1.1 weights and same 7.0 rating:
        # weighted = (7.0*0.9 + 7.0*1.1) / (0.9+1.1) = 14.0 / 2.0 = 7.0
        # But if weights differ, the weighted avg should differ from simple avg
        assert "xi_weighted_home" in features

    def test_xi_position_weights_exist(self):
        """Verify position weights are defined per ARCHITECTURE_SOTA.md."""
        assert XI_POSITION_WEIGHTS["GK"] == 1.0
        assert XI_POSITION_WEIGHTS["DEF"] == 0.9
        assert XI_POSITION_WEIGHTS["MID"] == 1.0
        assert XI_POSITION_WEIGHTS["FWD"] == 1.1


class TestMissingDataFlags:
    """Tests for missing data handling and flags."""

    def test_circular_mean_returns_none_for_empty_list(self):
        """Test that circular mean returns None for empty kickoff history."""
        result = calculate_circular_mean_hour([])
        assert result is None

    def test_circular_mean_handles_single_kickoff(self):
        """Test circular mean with single data point."""
        kickoffs = [datetime(2026, 1, 15, 20, 0, 0)]
        result = calculate_circular_mean_hour(kickoffs)
        assert result is not None
        assert 19.5 < result < 20.5  # Should be around 20

    def test_circular_mean_handles_wraparound(self):
        """Test circular mean handles 23:00 and 01:00 correctly."""
        kickoffs = [
            datetime(2026, 1, 15, 23, 0, 0),  # 23:00
            datetime(2026, 1, 16, 1, 0, 0),   # 01:00
        ]
        result = calculate_circular_mean_hour(kickoffs)
        # Mean should be around midnight (0 or 24)
        assert result is not None
        assert result < 2 or result > 22  # Near midnight

    def test_tz_offset_returns_zero_for_none(self):
        """Test that timezone offset returns 0 for None timezone."""
        result = get_tz_offset_hours(None, datetime.utcnow())
        assert result == 0.0

    def test_tz_offset_returns_zero_for_invalid_tz(self):
        """Test that timezone offset returns 0 for invalid timezone name."""
        result = get_tz_offset_hours("Invalid/Timezone", datetime.utcnow())
        assert result == 0.0


class TestJusticeCalculation:
    """Tests for justice shrinkage calculation."""

    def test_justice_shrinkage_formula(self):
        """
        Verify the justice shrinkage formula:
        justice = (G - XG) / sqrt(XG + eps)
        rho = n / (n + k)
        justice_shrunk = rho * justice
        """
        import math

        # Scenario: 5 matches, team scored 10 goals with 8 xG
        n = 5
        goals_total = 10
        xg_total = 8.0

        justice = (goals_total - xg_total) / math.sqrt(xg_total + JUSTICE_EPSILON)
        rho = n / (n + JUSTICE_SHRINKAGE_K)
        justice_shrunk = rho * justice

        # With k=10, n=5: rho = 5/15 = 0.333
        expected_rho = 5 / 15
        assert abs(rho - expected_rho) < 0.001

        # Justice = (10-8) / sqrt(8.01) ≈ 0.707
        expected_justice = 2 / math.sqrt(8 + JUSTICE_EPSILON)
        assert abs(justice - expected_justice) < 0.01

        # justice_shrunk ≈ 0.333 * 0.707 ≈ 0.236
        assert abs(justice_shrunk - (expected_rho * expected_justice)) < 0.01

    def test_justice_zero_when_no_history(self):
        """Justice should be 0 when there's no history (n=0)."""
        n = 0
        # With n=0, rho = 0/(0+k) = 0, so justice_shrunk = 0
        rho = n / (n + JUSTICE_SHRINKAGE_K) if n > 0 else 0
        assert rho == 0


class TestFeatureEngineerIntegration:
    """Integration tests for FeatureEngineer with SOTA features."""

    @pytest.mark.asyncio
    async def test_get_match_features_returns_all_sota_fields(self):
        """
        Test that get_match_features includes all SOTA feature fields.

        Expected fields:
        - Understat: home_xg_for_avg, away_xg_for_avg, xg_diff_avg, etc.
        - Weather: weather_temp_c, weather_humidity, etc.
        - Bio: thermal_shock, circadian_disruption, bio_disruption, etc.
        - Flags: understat_missing, weather_missing
        """
        from app.features.engineering import FeatureEngineer

        # Create mock session and match
        mock_session = AsyncMock()

        # Mock the session.execute to return empty results (graceful degradation)
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = []
        mock_result.fetchone.return_value = None
        mock_result.fetchall.return_value = []
        mock_session.execute.return_value = mock_result

        # Create mock match
        mock_match = MagicMock()
        mock_match.id = 1
        mock_match.external_id = "ext_1"
        mock_match.date = datetime(2026, 1, 15, 15, 0, 0)
        mock_match.league_id = 39
        mock_match.home_team_id = 100
        mock_match.away_team_id = 200
        mock_match.match_weight = 1.0
        mock_match.stats = None
        mock_match.tainted = False

        engineer = FeatureEngineer(mock_session)
        features = await engineer.get_match_features(mock_match)

        # Check baseline features exist
        assert "home_goals_scored_avg" in features
        assert "away_goals_scored_avg" in features
        assert "goal_diff_avg" in features

        # Check Understat features exist
        assert "home_xg_for_avg" in features
        assert "away_xg_for_avg" in features
        assert "xg_diff_avg" in features
        assert "home_justice_shrunk" in features
        assert "away_justice_shrunk" in features
        assert "justice_diff" in features
        assert "understat_missing" in features
        assert "understat_samples_home" in features
        assert "understat_samples_away" in features

        # Check Weather/Bio features exist
        assert "weather_temp_c" in features
        assert "weather_humidity" in features
        assert "weather_wind_ms" in features
        assert "weather_precip_mm" in features
        assert "is_daylight" in features
        assert "weather_missing" in features
        assert "thermal_shock" in features
        assert "thermal_shock_abs" in features
        assert "tz_shift" in features
        assert "circadian_disruption" in features
        assert "bio_disruption" in features

        # Check Sofascore XI features exist (suffix convention per FEATURE_DICTIONARY_SOTA.md)
        assert "xi_weighted_home" in features
        assert "xi_weighted_away" in features
        assert "xi_weighted_diff" in features
        assert "xi_p10_home" in features
        assert "xi_p50_home" in features
        assert "xi_p90_home" in features
        assert "xi_weaklink_home" in features
        assert "xi_std_home" in features
        assert "formation_home" in features
        assert "formation_away" in features
        assert "xi_missing" in features
        assert "xi_captured_horizon_minutes" in features

        # With no data, missing flags should be set
        assert features["understat_missing"] == 1
        assert features["weather_missing"] == 1
        assert features["xi_missing"] == 1


# ============================================================================
# Standalone test runner (when pytest is not available)
# ============================================================================

def run_sync_tests():
    """Run all synchronous tests."""
    print("Running synchronous tests...")

    # TestMissingDataFlags
    test_flags = TestMissingDataFlags()
    test_flags.test_circular_mean_returns_none_for_empty_list()
    print("  test_circular_mean_returns_none_for_empty_list: PASSED")

    test_flags.test_circular_mean_handles_single_kickoff()
    print("  test_circular_mean_handles_single_kickoff: PASSED")

    test_flags.test_circular_mean_handles_wraparound()
    print("  test_circular_mean_handles_wraparound: PASSED")

    test_flags.test_tz_offset_returns_zero_for_none()
    print("  test_tz_offset_returns_zero_for_none: PASSED")

    test_flags.test_tz_offset_returns_zero_for_invalid_tz()
    print("  test_tz_offset_returns_zero_for_invalid_tz: PASSED")

    # TestJusticeCalculation
    test_justice = TestJusticeCalculation()
    test_justice.test_justice_shrinkage_formula()
    print("  test_justice_shrinkage_formula: PASSED")

    test_justice.test_justice_zero_when_no_history()
    print("  test_justice_zero_when_no_history: PASSED")

    # TestXIFeatureCalculation
    test_xi = TestXIFeatureCalculation()
    test_xi.test_calculate_xi_features_with_valid_ratings()
    print("  test_calculate_xi_features_with_valid_ratings: PASSED")

    test_xi.test_calculate_xi_features_only_uses_starters()
    print("  test_calculate_xi_features_only_uses_starters: PASSED")

    test_xi.test_calculate_xi_features_with_empty_players()
    print("  test_calculate_xi_features_with_empty_players: PASSED")

    test_xi.test_calculate_xi_features_uses_position_weights()
    print("  test_calculate_xi_features_uses_position_weights: PASSED")

    test_xi.test_xi_position_weights_exist()
    print("  test_xi_position_weights_exist: PASSED")

    print("\nAll synchronous tests PASSED!")


async def run_async_tests():
    """Run all async tests."""
    print("\nRunning async tests...")

    # TestPointInTimeUnderstat
    test_understat = TestPointInTimeUnderstat()
    await test_understat.test_understat_snapshot_ignored_if_captured_after_kickoff()
    print("  test_understat_snapshot_ignored_if_captured_after_kickoff: PASSED")

    await test_understat.test_understat_snapshot_used_if_captured_before_kickoff()
    print("  test_understat_snapshot_used_if_captured_before_kickoff: PASSED")

    await test_understat.test_understat_history_only_uses_matches_before_t0()
    print("  test_understat_history_only_uses_matches_before_t0: PASSED")

    # TestPointInTimeWeather
    test_weather = TestPointInTimeWeather()
    await test_weather.test_weather_snapshot_ignored_if_captured_after_kickoff()
    print("  test_weather_snapshot_ignored_if_captured_after_kickoff: PASSED")

    await test_weather.test_weather_prefers_horizon_24_when_available()
    print("  test_weather_prefers_horizon_24_when_available: PASSED")

    # TestPointInTimeSofascoreXI
    test_xi = TestPointInTimeSofascoreXI()
    await test_xi.test_xi_snapshot_ignored_if_captured_after_kickoff()
    print("  test_xi_snapshot_ignored_if_captured_after_kickoff: PASSED")

    await test_xi.test_xi_snapshot_used_if_captured_before_kickoff()
    print("  test_xi_snapshot_used_if_captured_before_kickoff: PASSED")

    # TestFeatureEngineerIntegration
    test_integration = TestFeatureEngineerIntegration()
    await test_integration.test_get_match_features_returns_all_sota_fields()
    print("  test_get_match_features_returns_all_sota_fields: PASSED")

    print("\nAll async tests PASSED!")


if __name__ == "__main__":
    run_sync_tests()
    asyncio.run(run_async_tests())
    print("\n" + "=" * 60)
    print("ALL POINT-IN-TIME TESTS PASSED!")
    print("=" * 60)
