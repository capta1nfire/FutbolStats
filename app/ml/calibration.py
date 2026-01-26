"""
Post-hoc probability calibration methods.

FASE 3C.2 - ABE approved (2026-01-25)

Methods available:
- IsotonicCalibrator: Pool Adjacent Violators Algorithm (PAVA)
- TemperatureScaling: Grid search for optimal temperature

Implementation: numpy-only (no sklearn/scipy dependency per ABE requirement).

CRITICAL: Train/test split must be respected to avoid data leakage.
- Calibrator trains ONLY on data before --calib-train-end
- Evaluation is on data >= --min-snapshot-date (independent test set)
"""

import numpy as np
from typing import Optional, Tuple


class IsotonicCalibrator:
    """
    Isotonic regression calibration using Pool Adjacent Violators Algorithm (PAVA).

    Numpy-only implementation (no sklearn dependency).

    Per-class calibration: fits separate isotonic regression for each outcome
    (home, draw, away), then renormalizes to sum to 1.
    """

    def __init__(self):
        self.calibrators = {}  # {cls: (x_sorted, y_isotonic)}
        self._is_fitted = False
        self.n_train = 0

    def fit(self, probs: np.ndarray, outcomes: np.ndarray) -> 'IsotonicCalibrator':
        """
        Fit calibrators on historical predictions and outcomes.

        Args:
            probs: Shape (n, 3) - model probabilities [p_home, p_draw, p_away]
            outcomes: Shape (n,) - actual outcomes (0=home, 1=draw, 2=away)

        Returns:
            self (fitted calibrator)
        """
        probs = np.asarray(probs)
        outcomes = np.asarray(outcomes)

        if len(probs) == 0:
            raise ValueError("Cannot fit calibrator with empty data")

        self.n_train = len(probs)

        for cls in [0, 1, 2]:
            y_binary = (outcomes == cls).astype(float)

            # Sort by predicted probability
            order = np.argsort(probs[:, cls])
            x_sorted = probs[order, cls]
            y_sorted = y_binary[order]

            # Apply PAVA
            y_isotonic = self._pava(y_sorted)

            self.calibrators[cls] = (x_sorted.copy(), y_isotonic.copy())

        self._is_fitted = True
        return self

    def _pava(self, y: np.ndarray) -> np.ndarray:
        """
        Pool Adjacent Violators Algorithm.

        Ensures monotonically increasing sequence.

        Args:
            y: Array of values (sorted by x)

        Returns:
            Isotonic (monotonically increasing) version of y
        """
        n = len(y)
        if n == 0:
            return y

        y_iso = y.copy().astype(float)

        # Block structure: list of (start_idx, end_idx, mean_value)
        blocks = [(i, i, y_iso[i]) for i in range(n)]

        i = 0
        while i < len(blocks) - 1:
            # Check if current block violates monotonicity with next
            if blocks[i][2] > blocks[i + 1][2]:
                # Merge blocks
                start = blocks[i][0]
                end = blocks[i + 1][1]
                # Calculate mean of merged block
                merged_sum = sum(y_iso[j] for j in range(start, end + 1))
                merged_mean = merged_sum / (end - start + 1)

                # Update y_iso with merged mean
                for j in range(start, end + 1):
                    y_iso[j] = merged_mean

                # Replace two blocks with one
                blocks[i] = (start, end, merged_mean)
                blocks.pop(i + 1)

                # Go back to check previous block
                if i > 0:
                    i -= 1
            else:
                i += 1

        return y_iso

    def transform(self, probs: np.ndarray) -> np.ndarray:
        """
        Calibrate probabilities using fitted isotonic regression.

        Uses linear interpolation for values not seen during training.

        Args:
            probs: Shape (n, 3) - raw model probabilities

        Returns:
            Calibrated probabilities, renormalized to sum to 1
        """
        if not self._is_fitted:
            raise ValueError("Calibrator not fitted. Call fit() first.")

        probs = np.asarray(probs)
        calibrated = np.zeros_like(probs, dtype=float)

        for cls in [0, 1, 2]:
            x_sorted, y_isotonic = self.calibrators[cls]

            # Linear interpolation
            calibrated[:, cls] = np.interp(
                probs[:, cls],
                x_sorted,
                y_isotonic,
                left=y_isotonic[0],   # Extrapolate left
                right=y_isotonic[-1]  # Extrapolate right
            )

        # Renormalize to sum to 1
        row_sums = calibrated.sum(axis=1, keepdims=True)
        row_sums = np.maximum(row_sums, 1e-10)  # Avoid division by zero
        calibrated = calibrated / row_sums

        return calibrated

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted


class TemperatureScaling:
    """
    Temperature scaling calibration.

    Divides logits by temperature T before softmax.
    - T > 1: Makes distribution more uniform (less confident)
    - T < 1: Makes distribution more peaked (more confident)

    Numpy-only implementation using grid search (no scipy.optimize).
    """

    def __init__(self, t_min: float = 0.1, t_max: float = 5.0, n_grid: int = 200):
        """
        Initialize temperature scaling.

        Args:
            t_min: Minimum temperature to search
            t_max: Maximum temperature to search
            n_grid: Number of grid points for search
        """
        self.temperature: float = 1.0
        self.t_min = t_min
        self.t_max = t_max
        self.n_grid = n_grid
        self._is_fitted = False
        self.n_train = 0

    def fit(self, probs: np.ndarray, outcomes: np.ndarray) -> 'TemperatureScaling':
        """
        Find optimal temperature using NLL minimization via grid search.

        Args:
            probs: Shape (n, 3) - model probabilities
            outcomes: Shape (n,) - actual outcomes (0, 1, or 2)

        Returns:
            self (fitted calibrator)
        """
        probs = np.asarray(probs)
        outcomes = np.asarray(outcomes).astype(int)

        if len(probs) == 0:
            raise ValueError("Cannot fit calibrator with empty data")

        self.n_train = len(probs)

        best_nll = float('inf')
        best_T = 1.0

        # Grid search
        for T in np.linspace(self.t_min, self.t_max, self.n_grid):
            calibrated = self._apply_temperature(probs, T)
            nll = self._compute_nll(calibrated, outcomes)

            if nll < best_nll:
                best_nll = nll
                best_T = T

        self.temperature = best_T
        self._is_fitted = True
        return self

    def _apply_temperature(self, probs: np.ndarray, T: float) -> np.ndarray:
        """Apply temperature scaling to probabilities."""
        # Convert to logits (log odds)
        logits = np.log(np.maximum(probs, 1e-10))

        # Scale by temperature
        scaled_logits = logits / T

        # Softmax (with numerical stability)
        scaled_logits = scaled_logits - scaled_logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(scaled_logits)
        calibrated = exp_logits / exp_logits.sum(axis=1, keepdims=True)

        return calibrated

    def _compute_nll(self, probs: np.ndarray, outcomes: np.ndarray) -> float:
        """Compute negative log-likelihood."""
        # Get probability of true outcome for each sample
        true_probs = probs[np.arange(len(outcomes)), outcomes]

        # NLL = -mean(log(p_true))
        nll = -np.log(np.maximum(true_probs, 1e-10)).mean()
        return nll

    def transform(self, probs: np.ndarray) -> np.ndarray:
        """
        Apply temperature scaling to probabilities.

        Args:
            probs: Shape (n, 3) - raw model probabilities

        Returns:
            Calibrated probabilities
        """
        if not self._is_fitted:
            raise ValueError("Calibrator not fitted. Call fit() first.")

        return self._apply_temperature(probs, self.temperature)

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted


def get_calibrator(method: str = "none") -> Optional[object]:
    """
    Get calibrator instance by name.

    Args:
        method: "none" (default), "isotonic", or "temperature"

    Returns:
        Calibrator instance or None
    """
    if method == "isotonic":
        return IsotonicCalibrator()
    elif method == "temperature":
        return TemperatureScaling()
    else:
        return None
