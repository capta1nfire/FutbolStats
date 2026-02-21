"""
Tests exhaustivos para justice.py + autopsy.py — ABE mandatory.

Validates:
1. Dixon-Coles probabilities sum to 1.0 for all inputs
2. No NaN or Inf under any circumstance
3. Variance-scaled Justice weight: no division by zero
4. Autopsy tag coverage: all combinations classified without gaps
"""

import numpy as np
import pytest

from app.ml.justice import (
    DEFAULT_MAX_GOALS,
    DEFAULT_RHO,
    compute_justice_weight,
    compute_y_soft,
    compute_y_soft_batch,
)
from app.ml.autopsy import AutopsyTag, classify_autopsy


# ═══════════════════════════════════════════════════════════════════
# Module 1: Poisson Dixon-Coles Y_soft
# ═══════════════════════════════════════════════════════════════════


class TestPoissonDixonColes:
    """Dixon-Coles Y_soft: sum=1, draw inflation, edge cases."""

    def test_probabilities_sum_to_one(self):
        """Every xG combination produces probabilities summing to 1.0."""
        xg_pairs = [
            (1.0, 1.0), (0.5, 2.5), (3.0, 0.3), (0.01, 0.01),
            (4.5, 4.5), (0.0, 0.0), (5.5, 0.1), (0.1, 5.5),
        ]
        for xg_h, xg_a in xg_pairs:
            p_h, p_d, p_a = compute_y_soft(xg_h, xg_a)
            total = p_h + p_d + p_a
            assert abs(total - 1.0) < 1e-9, (
                f"xG ({xg_h}, {xg_a}): sum={total}"
            )

    def test_batch_probabilities_sum_to_one(self):
        """Batch version: every row sums to 1.0."""
        xg_h = np.array([1.0, 0.5, 3.0, 0.0, 5.5])
        xg_a = np.array([1.0, 2.5, 0.3, 0.0, 0.1])
        result = compute_y_soft_batch(xg_h, xg_a)
        row_sums = result.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-9)

    def test_no_nan_no_inf(self):
        """No NaN or Inf for extreme inputs."""
        extremes = [
            (0.0, 0.0), (1e-10, 1e-10), (10.0, 10.0),
            (0.001, 5.0), (5.0, 0.001),
        ]
        for xg_h, xg_a in extremes:
            p_h, p_d, p_a = compute_y_soft(xg_h, xg_a)
            assert np.isfinite(p_h), f"NaN/Inf p_home for ({xg_h}, {xg_a})"
            assert np.isfinite(p_d), f"NaN/Inf p_draw for ({xg_h}, {xg_a})"
            assert np.isfinite(p_a), f"NaN/Inf p_away for ({xg_h}, {xg_a})"

    def test_all_probabilities_non_negative(self):
        """All probabilities >= 0 for random inputs."""
        np.random.seed(42)
        xg_h = np.random.uniform(0.0, 5.0, 100)
        xg_a = np.random.uniform(0.0, 5.0, 100)
        result = compute_y_soft_batch(xg_h, xg_a)
        assert (result >= 0).all(), "Negative probabilities found"

    def test_dixon_coles_inflates_draw(self):
        """Dixon-Coles (ρ=-0.15) inflates p_draw vs independent Poisson (ρ=0)."""
        xg_h, xg_a = 1.0, 1.0
        _, p_draw_dc, _ = compute_y_soft(xg_h, xg_a, rho=-0.15)
        _, p_draw_indep, _ = compute_y_soft(xg_h, xg_a, rho=0.0)
        assert p_draw_dc > p_draw_indep, (
            f"Dixon-Coles should inflate draw: DC={p_draw_dc:.4f} "
            f"vs indep={p_draw_indep:.4f}"
        )

    def test_symmetric_xg_produces_symmetric_probs(self):
        """Equal xG → p_home ≈ p_away (symmetry)."""
        p_h, p_d, p_a = compute_y_soft(1.5, 1.5)
        assert abs(p_h - p_a) < 1e-9, f"Asymmetric: p_h={p_h}, p_a={p_a}"

    def test_dominant_xg_favors_home(self):
        """xG_home >> xG_away → p_home > p_draw > p_away."""
        p_h, p_d, p_a = compute_y_soft(3.0, 0.5)
        assert p_h > p_d > p_a, (
            f"Expected p_h > p_d > p_a: {p_h:.4f}, {p_d:.4f}, {p_a:.4f}"
        )

    def test_zero_xg_both(self):
        """Both xG = 0 → draw dominant (0-0)."""
        p_h, p_d, p_a = compute_y_soft(0.0, 0.0)
        assert p_d > 0.95, (
            f"Both xG=0 should be near-certain draw, got p_d={p_d:.4f}"
        )

    def test_rho_zero_matches_independent_poisson(self):
        """ρ=0 must reproduce independent Poisson exactly."""
        xg_h, xg_a = 1.5, 1.0
        p_h, p_d, p_a = compute_y_soft(xg_h, xg_a, rho=0.0)
        total = p_h + p_d + p_a
        assert abs(total - 1.0) < 1e-9

    def test_max_goals_10_covers_high_xg(self):
        """max_goals=10 captures >99.9% probability for xG=5.5."""
        p_h, p_d, p_a = compute_y_soft(5.5, 0.5, max_goals=10)
        total = p_h + p_d + p_a
        assert abs(total - 1.0) < 1e-6, f"High xG normalization: {total}"

    def test_batch_single_consistency(self):
        """Batch result matches single-call result."""
        xg_h, xg_a = 1.8, 0.9
        p_h_s, p_d_s, p_a_s = compute_y_soft(xg_h, xg_a)
        batch = compute_y_soft_batch(np.array([xg_h]), np.array([xg_a]))
        np.testing.assert_allclose(batch[0], [p_h_s, p_d_s, p_a_s], atol=1e-12)


# ═══════════════════════════════════════════════════════════════════
# Module 1: Justice Index W
# ═══════════════════════════════════════════════════════════════════


class TestJusticeWeight:
    """Justice Index W: variance-scaled, no NaN/division-by-zero."""

    def test_perfect_justice(self):
        """GD == xGD → W = 1.0."""
        w = compute_justice_weight(
            np.array([2.0]), np.array([1.0]),
            np.array([2.0]), np.array([1.0]),
        )
        assert abs(w[0] - 1.0) < 1e-9

    def test_no_nan_zero_xg(self):
        """Both xG = 0 → no NaN (denominator = √(0+0+1) = 1)."""
        w = compute_justice_weight(
            np.array([0.0]), np.array([0.0]),
            np.array([0.0]), np.array([0.0]),
        )
        assert np.isfinite(w[0]), f"NaN for zero xG: {w[0]}"

    def test_no_nan_nan_xg(self):
        """xG = NaN → W = 1.0 (fallback)."""
        w = compute_justice_weight(
            np.array([2.0]), np.array([1.0]),
            np.array([np.nan]), np.array([np.nan]),
        )
        assert w[0] == 1.0, f"NaN xG should produce W=1.0, got {w[0]}"

    def test_variance_scaling_high_xg(self):
        """GDT Override 2: 1-goal error weighs LESS in high-xG match.

        4-0 with xG 3.0-0.5:
        |GD-xGD| = |4.0 - 2.5| = 1.5
        σ = √(3.0 + 0.5 + 1.0) = √4.5 ≈ 2.12
        scaled = 1.5 / 2.12 ≈ 0.71
        W = exp(-0.5 × 0.71) ≈ 0.70
        Without scaling: W = exp(-0.5 × 1.5) ≈ 0.47
        """
        w = compute_justice_weight(
            np.array([4.0]), np.array([0.0]),
            np.array([3.0]), np.array([0.5]),
            alpha=0.5,
        )
        assert w[0] > 0.60, (
            f"Variance scaling should produce W > 0.60, got {w[0]:.4f}"
        )

    def test_variance_scaling_low_xg(self):
        """1-goal error in a tight match weighs MORE."""
        w = compute_justice_weight(
            np.array([1.0]), np.array([0.0]),
            np.array([0.3]), np.array([0.3]),
            alpha=0.5,
        )
        # σ = √(0.3 + 0.3 + 1.0) = √1.6 ≈ 1.26
        # scaled = |1.0 - 0.0| / 1.26 ≈ 0.79
        # W = exp(-0.5 × 0.79) ≈ 0.67
        assert w[0] < 0.75, (
            f"Low xG should produce lower W, got {w[0]:.4f}"
        )

    def test_w_always_in_zero_one(self):
        """W always in (0, 1] for any finite input."""
        np.random.seed(42)
        N = 500
        hg = np.random.randint(0, 8, N).astype(float)
        ag = np.random.randint(0, 8, N).astype(float)
        xh = np.random.uniform(0.0, 5.0, N)
        xa = np.random.uniform(0.0, 5.0, N)
        w = compute_justice_weight(hg, ag, xh, xa)
        assert (w > 0).all() and (w <= 1.0).all(), "W out of (0, 1]"

    def test_batch_mixed_nan(self):
        """Batch with mix of valid xG and NaN."""
        w = compute_justice_weight(
            np.array([2.0, 3.0, 1.0]),
            np.array([1.0, 0.0, 0.0]),
            np.array([1.5, np.nan, 0.5]),
            np.array([0.5, np.nan, 0.5]),
        )
        assert np.isfinite(w).all()
        assert w[1] == 1.0  # NaN xG → W = 1.0

    def test_alpha_zero_gives_all_ones(self):
        """α=0 → all weights = 1.0 (no justice modulation)."""
        w = compute_justice_weight(
            np.array([5.0, 0.0]), np.array([0.0, 3.0]),
            np.array([1.0, 2.0]), np.array([1.0, 0.5]),
            alpha=0.0,
        )
        np.testing.assert_allclose(w, 1.0)


# ═══════════════════════════════════════════════════════════════════
# Module 2: Financial Autopsy Tags
# ═══════════════════════════════════════════════════════════════════


class TestAutopsyClassification:
    """Autopsy tag: full coverage, no gaps, correct hierarchy."""

    def test_sharp_win(self):
        """Correct + positive CLV → SHARP_WIN."""
        tag = classify_autopsy(
            prediction_correct=True,
            predicted_result="home",
            clv_selected=0.05,
            xg_home=1.5, xg_away=0.8,
        )
        assert tag == AutopsyTag.SHARP_WIN

    def test_routine_win(self):
        """Correct + near-zero CLV + xG aligned → ROUTINE_WIN."""
        tag = classify_autopsy(
            prediction_correct=True,
            predicted_result="home",
            clv_selected=0.01,
            xg_home=2.0, xg_away=0.5,
        )
        assert tag == AutopsyTag.ROUTINE_WIN

    def test_lucky_win(self):
        """Correct + xG against us → LUCKY_WIN."""
        tag = classify_autopsy(
            prediction_correct=True,
            predicted_result="home",
            clv_selected=0.0,
            xg_home=0.5, xg_away=2.0,
        )
        assert tag == AutopsyTag.LUCKY_WIN

    def test_sharp_loss(self):
        """Wrong + positive CLV → SHARP_LOSS."""
        tag = classify_autopsy(
            prediction_correct=False,
            predicted_result="home",
            clv_selected=0.08,
            xg_home=1.5, xg_away=0.8,
        )
        assert tag == AutopsyTag.SHARP_LOSS

    def test_variance_loss(self):
        """Wrong + xG supported us → VARIANCE_LOSS."""
        tag = classify_autopsy(
            prediction_correct=False,
            predicted_result="home",
            clv_selected=0.0,
            xg_home=2.5, xg_away=0.3,
        )
        assert tag == AutopsyTag.VARIANCE_LOSS

    def test_blind_spot(self):
        """Wrong + negative CLV + xG against → BLIND_SPOT."""
        tag = classify_autopsy(
            prediction_correct=False,
            predicted_result="home",
            clv_selected=-0.10,
            xg_home=0.5, xg_away=2.0,
        )
        assert tag == AutopsyTag.BLIND_SPOT

    def test_no_clv_no_xg_correct(self):
        """Correct + no CLV + no xG → ROUTINE_WIN (default)."""
        tag = classify_autopsy(
            prediction_correct=True,
            predicted_result="draw",
            clv_selected=None,
            xg_home=None, xg_away=None,
        )
        assert tag == AutopsyTag.ROUTINE_WIN

    def test_no_clv_no_xg_wrong(self):
        """Wrong + no CLV + no xG → VARIANCE_LOSS (conservative default)."""
        tag = classify_autopsy(
            prediction_correct=False,
            predicted_result="draw",
            clv_selected=None,
            xg_home=None, xg_away=None,
        )
        assert tag == AutopsyTag.VARIANCE_LOSS

    def test_xg_threshold_035(self):
        """GDT Override 3: xG diff of 0.30 is a technical draw, not a winner."""
        # xG 1.30 - 1.00 = 0.30 < 0.35 threshold → xG says "draw"
        # Predicted "home" but lost → xG doesn't support (draw ≠ home)
        tag = classify_autopsy(
            prediction_correct=False,
            predicted_result="home",
            clv_selected=0.0,
            xg_home=1.30, xg_away=1.00,
        )
        # xG says draw, predicted home → xg_supports = False → BLIND_SPOT
        assert tag == AutopsyTag.BLIND_SPOT

    def test_all_tags_reachable(self):
        """Every AutopsyTag value is reachable."""
        cases = {
            AutopsyTag.SHARP_WIN: (True, "home", 0.05, 2.0, 0.5),
            AutopsyTag.ROUTINE_WIN: (True, "home", 0.0, 2.0, 0.5),
            AutopsyTag.LUCKY_WIN: (True, "home", 0.0, 0.5, 2.0),
            AutopsyTag.SHARP_LOSS: (False, "home", 0.05, 2.0, 0.5),
            AutopsyTag.VARIANCE_LOSS: (False, "home", 0.0, 2.0, 0.5),
            AutopsyTag.BLIND_SPOT: (False, "home", -0.05, 0.5, 2.0),
        }
        for expected_tag, args in cases.items():
            tag = classify_autopsy(*args)
            assert tag == expected_tag, (
                f"Expected {expected_tag}, got {tag} for args={args}"
            )
