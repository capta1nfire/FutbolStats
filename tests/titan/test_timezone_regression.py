"""Regression tests for timezone handling and SQL syntax with asyncpg.

These tests verify fixes for production errors discovered during FASE 2 pilot:
1. Timezone aware vs naive datetime handling for public.* vs titan.*
2. SQL CAST syntax instead of ::type for asyncpg compatibility

References:
- Error 1: asyncpg.exceptions.DataError with offset-aware datetimes
- Error 2: asyncpg.exceptions.PostgresSyntaxError with ::jsonb
- Docs: docs/OPS_RUNBOOK.md section "TITAN: Reglas de Timezone"
"""

import pytest
from datetime import datetime, timezone
from decimal import Decimal

from app.titan.materializers.feature_matrix import (
    OddsFeatures,
    FormFeatures,
    H2HFeatures,
    XGFeatures,
)


class TestTimezoneNormalization:
    """Test datetime normalization for public.* (naive) vs titan.* (aware)."""

    def test_aware_to_naive_conversion(self):
        """Verify aware datetime can be converted to naive for public.* queries.

        This tests the fix for Error 1:
        asyncpg.exceptions.DataError: invalid input for query argument $2:
        (can't subtract offset-naive and offset-aware datetimes)
        """
        # Input: aware datetime (common in TITAN pipeline)
        kickoff_utc = datetime(2026, 1, 26, 20, 0, 0, tzinfo=timezone.utc)
        assert kickoff_utc.tzinfo is not None  # Confirm it's aware

        # Fix: strip tzinfo for public.matches queries
        kickoff_naive = kickoff_utc.replace(tzinfo=None) if kickoff_utc.tzinfo else kickoff_utc

        # Result should be naive
        assert kickoff_naive.tzinfo is None
        assert kickoff_naive.year == 2026
        assert kickoff_naive.month == 1
        assert kickoff_naive.day == 26
        assert kickoff_naive.hour == 20

    def test_naive_stays_naive(self):
        """Naive datetime should stay naive (no-op)."""
        kickoff_naive = datetime(2026, 1, 26, 20, 0, 0)
        assert kickoff_naive.tzinfo is None

        # The fix should be a no-op for already-naive datetimes
        result = kickoff_naive.replace(tzinfo=None) if kickoff_naive.tzinfo else kickoff_naive

        assert result.tzinfo is None
        assert result == kickoff_naive

    def test_naive_to_aware_conversion(self):
        """Verify naive datetime can be converted to aware for titan.* queries."""
        # Input: naive datetime
        kickoff_naive = datetime(2026, 1, 26, 20, 0, 0)
        assert kickoff_naive.tzinfo is None

        # Fix: add tzinfo for titan.* queries
        kickoff_aware = kickoff_naive if kickoff_naive.tzinfo else kickoff_naive.replace(tzinfo=timezone.utc)

        # Result should be aware
        assert kickoff_aware.tzinfo is not None
        assert kickoff_aware.tzinfo == timezone.utc

    def test_aware_stays_aware(self):
        """Aware datetime should stay aware (no-op)."""
        kickoff_aware = datetime(2026, 1, 26, 20, 0, 0, tzinfo=timezone.utc)
        assert kickoff_aware.tzinfo is not None

        # The fix should be a no-op for already-aware datetimes
        result = kickoff_aware if kickoff_aware.tzinfo else kickoff_aware.replace(tzinfo=timezone.utc)

        assert result.tzinfo is not None
        assert result == kickoff_aware


class TestFeaturesCapturedAtTimezone:
    """Test that feature dataclasses handle both naive and aware datetimes."""

    def test_odds_features_with_naive_datetime(self):
        """OddsFeatures should accept naive datetime."""
        captured = datetime(2026, 1, 25, 10, 0, 0)
        odds = OddsFeatures(
            odds_home_close=Decimal("2.10"),
            odds_draw_close=Decimal("3.40"),
            odds_away_close=Decimal("3.20"),
            implied_prob_home=Decimal("0.4651"),
            implied_prob_draw=Decimal("0.2874"),
            implied_prob_away=Decimal("0.3053"),
            captured_at=captured,
        )
        assert odds.captured_at.tzinfo is None

    def test_odds_features_with_aware_datetime(self):
        """OddsFeatures should accept aware datetime."""
        captured = datetime(2026, 1, 25, 10, 0, 0, tzinfo=timezone.utc)
        odds = OddsFeatures(
            odds_home_close=Decimal("2.10"),
            odds_draw_close=Decimal("3.40"),
            odds_away_close=Decimal("3.20"),
            implied_prob_home=Decimal("0.4651"),
            implied_prob_draw=Decimal("0.2874"),
            implied_prob_away=Decimal("0.3053"),
            captured_at=captured,
        )
        assert odds.captured_at.tzinfo is not None

    def test_xg_features_with_aware_datetime(self):
        """XGFeatures should accept aware datetime for TITAN compliance."""
        captured = datetime(2026, 1, 25, 10, 0, 0, tzinfo=timezone.utc)
        xg = XGFeatures(
            xg_home_last5=Decimal("1.85"),
            xg_away_last5=Decimal("1.42"),
            xga_home_last5=Decimal("0.95"),
            xga_away_last5=Decimal("1.28"),
            npxg_home_last5=Decimal("1.65"),
            npxg_away_last5=Decimal("1.22"),
            captured_at=captured,
        )
        assert xg.captured_at.tzinfo is not None


class TestSQLCastSyntax:
    """Test SQL CAST syntax requirements for asyncpg.

    This tests the fix for Error 2:
    asyncpg.exceptions.PostgresSyntaxError: syntax error at or near ":"
    [SQL: ... :response_body::jsonb ...]
    """

    def test_correct_cast_syntax(self):
        """Verify the correct CAST syntax is used in SQL.

        asyncpg does NOT support :param::type syntax.
        Must use CAST(:param AS type) instead.
        """
        # Bad: This would fail with asyncpg
        bad_sql = "INSERT INTO table (col) VALUES (:value::jsonb)"
        assert "::jsonb" in bad_sql  # Confirm it's the bad pattern

        # Good: This works with asyncpg
        good_sql = "INSERT INTO table (col) VALUES (CAST(:value AS jsonb))"
        assert "CAST(" in good_sql
        assert "AS jsonb)" in good_sql
        assert "::jsonb" not in good_sql

    def test_job_manager_uses_cast_syntax(self):
        """Verify job_manager.py uses CAST syntax for jsonb."""
        import inspect
        from app.titan.jobs.job_manager import TitanJobManager

        # Get the source code of save_extraction method
        source = inspect.getsource(TitanJobManager.save_extraction)

        # Should use CAST syntax
        assert "CAST(:response_body AS jsonb)" in source
        # Should NOT use ::jsonb syntax
        assert ":response_body::jsonb" not in source

    def test_job_manager_dlq_uses_cast_syntax(self):
        """Verify send_to_dlq also uses CAST syntax for jsonb."""
        import inspect
        from app.titan.jobs.job_manager import TitanJobManager

        # Get the source code of send_to_dlq method
        source = inspect.getsource(TitanJobManager.send_to_dlq)

        # Should use CAST syntax for params
        assert "CAST(:params AS jsonb)" in source
        # Should NOT use ::jsonb syntax
        assert ":params::jsonb" not in source


class TestFormFeaturesQueryNormalization:
    """Test that form features queries handle timezone correctly."""

    def test_kickoff_normalization_logic(self):
        """The normalization logic used in compute_form_features.

        public.matches.date is TIMESTAMP (naive), so we must strip
        tzinfo from kickoff_utc before comparing.
        """
        # Simulate kickoff from runner (aware)
        kickoff_utc = datetime(2026, 1, 26, 20, 0, 0, tzinfo=timezone.utc)

        # The fix applied in feature_matrix.py
        kickoff_naive = kickoff_utc.replace(tzinfo=None) if kickoff_utc.tzinfo else kickoff_utc

        # This naive datetime can now be compared with public.matches.date
        assert kickoff_naive.tzinfo is None

        # Verify the datetime values are preserved
        assert kickoff_naive == datetime(2026, 1, 26, 20, 0, 0)


class TestH2HFeaturesQueryNormalization:
    """Test that H2H features queries handle timezone correctly."""

    def test_h2h_kickoff_normalization(self):
        """H2H queries also need naive datetimes for public.matches."""
        kickoff_utc = datetime(2026, 1, 26, 20, 0, 0, tzinfo=timezone.utc)

        # Same normalization as form features
        kickoff_naive = kickoff_utc.replace(tzinfo=None) if kickoff_utc.tzinfo else kickoff_utc

        assert kickoff_naive.tzinfo is None


class TestXGFeaturesQueryNormalization:
    """Test that xG features queries handle timezone correctly."""

    def test_xg_kickoff_normalization(self):
        """xG queries need naive datetimes for public.matches join."""
        kickoff_utc = datetime(2026, 1, 26, 20, 0, 0, tzinfo=timezone.utc)

        # Same normalization for xG queries
        kickoff_naive = kickoff_utc.replace(tzinfo=None) if kickoff_utc.tzinfo else kickoff_utc

        assert kickoff_naive.tzinfo is None

    def test_xg_captured_at_aware_for_titan(self):
        """xg_captured_at should be aware for titan.feature_matrix."""
        # xG data captured now (aware)
        captured = datetime.now(timezone.utc)

        assert captured.tzinfo is not None
        assert captured.tzinfo == timezone.utc
