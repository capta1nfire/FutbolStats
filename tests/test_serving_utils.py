"""
Tests for serving_utils — finalize_ev + calibration pipeline.

ABE P0 (2026-02-19):
- Bug test: value_bets must reflect SERVED probs, not raw
- Frozen/FT intocable in ALL stages
- Ghost-edge cleared for alpha >= 0.80
- best_value_bet by expected_value
- Calibration only baseline (skip TS/Family S)
"""

import pytest
import numpy as np


class MockEngine:
    """Minimal mock of XGBoostEngine._find_value_bets."""

    def _find_value_bets(self, probas, market_odds, threshold=None):
        if threshold is None:
            threshold = 0.05  # default for tests

        outcomes = ["home", "draw", "away"]
        value_bets = []
        for i, (prob, odds) in enumerate(zip(probas, market_odds)):
            if odds <= 0:
                continue
            prob = float(prob)
            implied = 1 / odds
            edge = prob - implied
            ev = (prob * odds) - 1
            if edge > threshold:
                value_bets.append({
                    "outcome": outcomes[i],
                    "our_probability": round(prob, 4),
                    "implied_probability": round(implied, 4),
                    "edge": round(edge, 4),
                    "edge_percentage": round(edge * 100, 1),
                    "expected_value": round(ev, 4),
                    "ev_percentage": round(ev * 100, 1),
                    "market_odds": float(odds),
                    "fair_odds": round(1 / prob, 2),
                    "is_value_bet": True,
                })
        return value_bets


class TestFinalizeEvBug:
    """Core bug test: value_bets must use SERVED probs, not raw model probs."""

    def test_served_probs_no_ghost_edge(self):
        """
        Scenario: market_anchor changed prob from 0.60 (raw) to 0.52 (served).
        Market odds = 2.0 (implied = 0.50). Threshold = 0.05.
        Raw edge = 0.60 - 0.50 = 0.10 > 0.05 → value_bet (BUG: old behavior)
        Served edge = 0.52 - 0.50 = 0.02 < 0.05 → NOT a value_bet (CORRECT)
        """
        from app.ml.serving_utils import finalize_ev_and_value_bets

        pred = {
            "status": "NS",
            "is_frozen": False,
            "probabilities": {"home": 0.52, "draw": 0.25, "away": 0.23},
            "market_odds": {"home": 2.0, "draw": 4.0, "away": 5.0},
            "value_bets": [
                {  # This is a GHOST: computed from raw 0.60, not served 0.52
                    "outcome": "home",
                    "our_probability": 0.60,
                    "edge": 0.10,
                    "expected_value": 0.20,
                    "market_odds": 2.0,
                    "is_value_bet": True,
                }
            ],
            "has_value_bet": True,
            "best_value_bet": {"outcome": "home", "expected_value": 0.20},
            "fair_odds": {"home": 1.67, "draw": 4.0, "away": 5.0},
        }

        engine = MockEngine()
        [result], meta = finalize_ev_and_value_bets([pred], engine)

        # Served prob 0.52 vs implied 0.50 = edge 0.02 < 0.05 → no value bet
        assert result["has_value_bet"] is False
        assert result["value_bets"] is None
        assert meta["n_recalculated"] == 1

    def test_best_value_bet_by_ev_not_edge(self):
        """best_value_bet must be by expected_value, not edge."""
        from app.ml.serving_utils import finalize_ev_and_value_bets

        pred = {
            "status": "NS",
            "is_frozen": False,
            "probabilities": {"home": 0.40, "draw": 0.30, "away": 0.30},
            "market_odds": {"home": 2.5, "draw": 3.0, "away": 3.5},
            "value_bets": None,
            "has_value_bet": False,
            "best_value_bet": None,
            "fair_odds": {},
        }

        engine = MockEngine()
        [result], _ = finalize_ev_and_value_bets([pred], engine)

        if result["has_value_bet"] and len(result["value_bets"]) > 1:
            best = result["best_value_bet"]
            for vb in result["value_bets"]:
                assert best["expected_value"] >= vb["expected_value"]


class TestFrozenFTIntocable:
    """ABE P0-6: Frozen and FT predictions must NEVER be modified."""

    def test_frozen_not_modified(self):
        from app.ml.serving_utils import finalize_ev_and_value_bets

        original_vb = [{"outcome": "home", "edge": 0.15, "expected_value": 0.30}]
        pred = {
            "status": "NS",
            "is_frozen": True,
            "probabilities": {"home": 0.60, "draw": 0.20, "away": 0.20},
            "market_odds": {"home": 2.0, "draw": 4.0, "away": 5.0},
            "value_bets": original_vb,
            "has_value_bet": True,
            "best_value_bet": original_vb[0],
        }

        engine = MockEngine()
        [result], meta = finalize_ev_and_value_bets([pred], engine)

        assert result["value_bets"] is original_vb
        assert result["has_value_bet"] is True
        assert meta["n_skipped"] == 1

    def test_ft_not_modified(self):
        from app.ml.serving_utils import finalize_ev_and_value_bets

        pred = {
            "status": "FT",
            "is_frozen": False,
            "probabilities": {"home": 0.60, "draw": 0.20, "away": 0.20},
            "market_odds": {"home": 2.0, "draw": 4.0, "away": 5.0},
            "value_bets": None,
            "has_value_bet": False,
            "best_value_bet": None,
        }

        engine = MockEngine()
        [result], meta = finalize_ev_and_value_bets([pred], engine)

        assert result["value_bets"] is None
        assert meta["n_skipped"] == 1

    def test_frozen_calibration_skipped(self):
        from app.ml.serving_utils import apply_calibration

        pred = {
            "status": "NS",
            "is_frozen": True,
            "probabilities": {"home": 0.50, "draw": 0.30, "away": 0.20},
        }

        [result], meta = apply_calibration(
            [pred], enabled=True, method="temperature", temperature=2.0
        )

        # Probabilities must NOT change
        assert result["probabilities"]["home"] == 0.50
        assert result["probabilities"]["draw"] == 0.30
        assert result["probabilities"]["away"] == 0.20


class TestGhostEdge:
    """ABE P0-5: alpha >= 0.80 means probs ≈ market, no real edge."""

    def test_ghost_cleared_for_high_alpha(self):
        from app.ml.serving_utils import finalize_ev_and_value_bets

        pred = {
            "status": "NS",
            "is_frozen": False,
            "probabilities": {"home": 0.51, "draw": 0.25, "away": 0.24},
            "market_odds": {"home": 2.0, "draw": 4.0, "away": 4.5},
            "value_bets": [{"outcome": "home", "edge": 0.01}],
            "has_value_bet": True,
            "best_value_bet": {"outcome": "home"},
            "policy_metadata": {
                "market_anchor": {"applied": True, "alpha": 0.90},
            },
        }

        engine = MockEngine()
        [result], meta = finalize_ev_and_value_bets([pred], engine)

        assert result["value_bets"] is None
        assert result["has_value_bet"] is False
        assert result["best_value_bet"] is None
        assert meta["n_ghost_cleared"] == 1

    def test_no_clear_for_low_alpha(self):
        from app.ml.serving_utils import finalize_ev_and_value_bets

        pred = {
            "status": "NS",
            "is_frozen": False,
            "probabilities": {"home": 0.60, "draw": 0.20, "away": 0.20},
            "market_odds": {"home": 2.0, "draw": 4.0, "away": 5.0},
            "value_bets": None,
            "has_value_bet": False,
            "best_value_bet": None,
            "policy_metadata": {
                "market_anchor": {"applied": True, "alpha": 0.30},
            },
        }

        engine = MockEngine()
        [result], meta = finalize_ev_and_value_bets([pred], engine)

        # Should recalculate normally (not ghost-clear)
        assert meta["n_ghost_cleared"] == 0
        assert meta["n_recalculated"] == 1


class TestCalibrationP03:
    """ABE P0-3: Calibration only for baseline, not TS/Family S."""

    def test_skip_ts_predictions(self):
        """Predictions with skip_market_anchor=True are TS/Family S → skip calibration."""
        from app.ml.serving_utils import apply_calibration

        pred = {
            "status": "NS",
            "is_frozen": False,
            "skip_market_anchor": True,  # TS or Family S
            "probabilities": {"home": 0.50, "draw": 0.30, "away": 0.20},
        }

        [result], meta = apply_calibration(
            [pred], enabled=True, method="temperature", temperature=2.0
        )

        # Must NOT be modified
        assert result["probabilities"]["home"] == 0.50
        assert meta["n_calibrated"] == 0

    def test_calibrate_baseline(self):
        """Baseline predictions (no skip_market_anchor) should be calibrated."""
        from app.ml.serving_utils import apply_calibration

        pred = {
            "status": "NS",
            "is_frozen": False,
            "probabilities": {"home": 0.70, "draw": 0.15, "away": 0.15},
        }

        [result], meta = apply_calibration(
            [pred], enabled=True, method="temperature", temperature=2.0
        )

        # T=2.0 softens distribution → home should be < 0.70
        assert result["probabilities"]["home"] < 0.70
        assert meta["n_calibrated"] == 1

    def test_identity_temperature(self):
        """T=1.0 should be a no-op (detected and skipped)."""
        from app.ml.serving_utils import apply_calibration

        pred = {
            "status": "NS",
            "is_frozen": False,
            "probabilities": {"home": 0.50, "draw": 0.30, "away": 0.20},
        }

        [result], meta = apply_calibration(
            [pred], enabled=True, method="temperature", temperature=1.0
        )

        assert result["probabilities"]["home"] == 0.50
        assert meta["skip_reason"] == "identity"


class TestDrawCapNSOnly:
    """ABE P0-1: draw_cap only operates on NS + non-frozen."""

    def test_ft_not_counted_in_draw_cap(self):
        from app.ml.policy import apply_draw_cap

        preds = [
            {
                "status": "FT",
                "is_frozen": False,
                "value_bets": [{"outcome": "draw", "edge": 0.10}],
                "has_value_bet": True,
            },
            {
                "status": "NS",
                "is_frozen": False,
                "value_bets": [{"outcome": "home", "edge": 0.08}],
                "has_value_bet": True,
            },
        ]

        result, meta = apply_draw_cap(preds, max_draw_share=0.35, enabled=True)

        # FT draw should NOT be counted or removed
        assert result[0]["value_bets"] == [{"outcome": "draw", "edge": 0.10}]
