"""
Tests for calibration methods (FASE 3C.2).

ABE requirement: numpy-only implementation (no SciPy/sklearn).
"""

import pytest
import numpy as np
from app.ml.calibration import (
    IsotonicCalibrator,
    TemperatureScaling,
    get_calibrator,
)


class TestIsotonicCalibrator:
    """Test isotonic regression calibration."""

    def test_fit_requires_data(self):
        """Fitting with empty data should raise."""
        cal = IsotonicCalibrator()
        with pytest.raises(ValueError, match="empty"):
            cal.fit(np.array([]), np.array([]))

    def test_transform_requires_fit(self):
        """Transform before fit should raise."""
        cal = IsotonicCalibrator()
        probs = np.array([[0.5, 0.25, 0.25]])
        with pytest.raises(ValueError, match="not fitted"):
            cal.transform(probs)

    def test_output_sums_to_one(self):
        """Calibrated probabilities should sum to 1."""
        cal = IsotonicCalibrator()
        # Create synthetic data
        np.random.seed(42)
        n = 100
        probs = np.random.dirichlet([2, 1, 1], n)
        outcomes = np.random.choice([0, 1, 2], n, p=[0.5, 0.25, 0.25])

        cal.fit(probs, outcomes)
        calibrated = cal.transform(probs)

        # All rows should sum to 1
        row_sums = calibrated.sum(axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-10)

    def test_preserves_shape(self):
        """Output should have same shape as input."""
        cal = IsotonicCalibrator()
        np.random.seed(42)
        n = 50
        probs = np.random.dirichlet([1, 1, 1], n)
        outcomes = np.random.choice([0, 1, 2], n)

        cal.fit(probs, outcomes)
        calibrated = cal.transform(probs)

        assert calibrated.shape == probs.shape

    def test_is_fitted_property(self):
        """is_fitted should reflect actual state."""
        cal = IsotonicCalibrator()
        assert cal.is_fitted is False

        probs = np.array([[0.5, 0.25, 0.25], [0.4, 0.3, 0.3]])
        outcomes = np.array([0, 1])
        cal.fit(probs, outcomes)

        assert cal.is_fitted is True

    def test_n_train_tracked(self):
        """n_train should track training samples."""
        cal = IsotonicCalibrator()
        probs = np.array([[0.5, 0.25, 0.25], [0.4, 0.3, 0.3], [0.3, 0.4, 0.3]])
        outcomes = np.array([0, 1, 2])
        cal.fit(probs, outcomes)

        assert cal.n_train == 3

    def test_pava_monotonic(self):
        """PAVA should produce monotonically increasing sequence."""
        cal = IsotonicCalibrator()
        # Test internal PAVA
        y = np.array([0.5, 0.2, 0.3, 0.8, 0.6, 0.9])
        y_iso = cal._pava(y)

        # Check monotonicity
        for i in range(len(y_iso) - 1):
            assert y_iso[i] <= y_iso[i + 1]


class TestTemperatureScaling:
    """Test temperature scaling calibration."""

    def test_fit_requires_data(self):
        """Fitting with empty data should raise."""
        cal = TemperatureScaling()
        with pytest.raises(ValueError, match="empty"):
            cal.fit(np.array([]), np.array([]))

    def test_transform_requires_fit(self):
        """Transform before fit should raise."""
        cal = TemperatureScaling()
        probs = np.array([[0.5, 0.25, 0.25]])
        with pytest.raises(ValueError, match="not fitted"):
            cal.transform(probs)

    def test_output_sums_to_one(self):
        """Calibrated probabilities should sum to 1."""
        cal = TemperatureScaling()
        np.random.seed(42)
        n = 100
        probs = np.random.dirichlet([2, 1, 1], n)
        outcomes = np.random.choice([0, 1, 2], n, p=[0.5, 0.25, 0.25])

        cal.fit(probs, outcomes)
        calibrated = cal.transform(probs)

        row_sums = calibrated.sum(axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-10)

    def test_preserves_shape(self):
        """Output should have same shape as input."""
        cal = TemperatureScaling()
        np.random.seed(42)
        n = 50
        probs = np.random.dirichlet([1, 1, 1], n)
        outcomes = np.random.choice([0, 1, 2], n)

        cal.fit(probs, outcomes)
        calibrated = cal.transform(probs)

        assert calibrated.shape == probs.shape

    def test_temperature_in_range(self):
        """Fitted temperature should be in expected range."""
        cal = TemperatureScaling(t_min=0.1, t_max=5.0)
        np.random.seed(42)
        n = 100
        probs = np.random.dirichlet([2, 1, 1], n)
        outcomes = np.random.choice([0, 1, 2], n)

        cal.fit(probs, outcomes)

        assert 0.1 <= cal.temperature <= 5.0

    def test_is_fitted_property(self):
        """is_fitted should reflect actual state."""
        cal = TemperatureScaling()
        assert cal.is_fitted is False

        probs = np.array([[0.5, 0.25, 0.25], [0.4, 0.3, 0.3]])
        outcomes = np.array([0, 1])
        cal.fit(probs, outcomes)

        assert cal.is_fitted is True

    def test_high_temperature_smooths_distribution(self):
        """T > 1 should make distribution more uniform."""
        cal = TemperatureScaling()
        # Peaked distribution
        probs = np.array([[0.8, 0.1, 0.1]])

        # Apply high temperature manually
        calibrated = cal._apply_temperature(probs, T=2.0)

        # Should be less peaked
        assert calibrated[0, 0] < probs[0, 0]
        # Still sums to 1
        assert np.isclose(calibrated.sum(), 1.0)

    def test_low_temperature_sharpens_distribution(self):
        """T < 1 should make distribution more peaked."""
        cal = TemperatureScaling()
        # Moderate distribution
        probs = np.array([[0.5, 0.3, 0.2]])

        # Apply low temperature manually
        calibrated = cal._apply_temperature(probs, T=0.5)

        # Should be more peaked
        assert calibrated[0, 0] > probs[0, 0]
        # Still sums to 1
        assert np.isclose(calibrated.sum(), 1.0)


class TestGetCalibrator:
    """Test calibrator factory function."""

    def test_none_returns_none(self):
        """'none' should return None."""
        result = get_calibrator("none")
        assert result is None

    def test_default_is_none(self):
        """Default should be None."""
        result = get_calibrator()
        assert result is None

    def test_isotonic_returns_calibrator(self):
        """'isotonic' should return IsotonicCalibrator."""
        result = get_calibrator("isotonic")
        assert isinstance(result, IsotonicCalibrator)

    def test_temperature_returns_calibrator(self):
        """'temperature' should return TemperatureScaling."""
        result = get_calibrator("temperature")
        assert isinstance(result, TemperatureScaling)

    def test_unknown_returns_none(self):
        """Unknown method should return None."""
        result = get_calibrator("unknown_method")
        assert result is None


class TestCalibrationIntegration:
    """Integration tests for calibration workflow."""

    def test_isotonic_train_test_split(self):
        """Isotonic should work with train/test split."""
        np.random.seed(42)
        # Train data
        n_train = 80
        train_probs = np.random.dirichlet([2, 1, 1], n_train)
        train_outcomes = np.random.choice([0, 1, 2], n_train, p=[0.45, 0.25, 0.30])

        # Test data
        n_test = 20
        test_probs = np.random.dirichlet([2, 1, 1], n_test)

        # Fit on train, transform on test
        cal = IsotonicCalibrator()
        cal.fit(train_probs, train_outcomes)
        calibrated = cal.transform(test_probs)

        # Should work and sum to 1
        assert calibrated.shape == test_probs.shape
        assert np.allclose(calibrated.sum(axis=1), 1.0)

    def test_temperature_train_test_split(self):
        """Temperature should work with train/test split."""
        np.random.seed(42)
        # Train data
        n_train = 80
        train_probs = np.random.dirichlet([2, 1, 1], n_train)
        train_outcomes = np.random.choice([0, 1, 2], n_train, p=[0.45, 0.25, 0.30])

        # Test data
        n_test = 20
        test_probs = np.random.dirichlet([2, 1, 1], n_test)

        # Fit on train, transform on test
        cal = TemperatureScaling()
        cal.fit(train_probs, train_outcomes)
        calibrated = cal.transform(test_probs)

        # Should work and sum to 1
        assert calibrated.shape == test_probs.shape
        assert np.allclose(calibrated.sum(axis=1), 1.0)

