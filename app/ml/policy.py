"""
Betting policy utilities for prediction serving.

FASE 1: Draw cap to prevent over-concentration in draw bets.
FASE 2: Market anchor for low-signal leagues (ABE P0 2026-02-08).

Feature flags (from config):
- POLICY_DRAW_CAP_ENABLED: Enable/disable draw cap
- POLICY_MAX_DRAW_SHARE: Maximum fraction of value bets that can be draws (0.35)
- POLICY_EDGE_THRESHOLD: Minimum edge for value bet (0.05)
- MARKET_ANCHOR_ENABLED: Enable/disable market anchor blend
- MARKET_ANCHOR_ALPHA_DEFAULT: Default blend weight (0.0 = model pure)
- MARKET_ANCHOR_LEAGUE_OVERRIDES: Per-league alpha overrides ("128:1.0")
"""

from __future__ import annotations

from typing import Optional
import logging

logger = logging.getLogger(__name__)


def apply_draw_cap(
    predictions: list[dict],
    max_draw_share: float = 0.35,
    enabled: bool = True,
) -> tuple[list[dict], dict]:
    """
    Apply draw cap to a list of predictions with value bets.

    For each prediction that has a draw value bet, we track it.
    If draws exceed max_draw_share of total value bets, we remove
    the draw value bet from predictions with lowest edge, keeping
    the top draws by edge.

    Args:
        predictions: List of prediction dicts with 'value_bets' field
        max_draw_share: Maximum fraction of value bets that can be draws (default 0.35)
        enabled: Whether to apply the cap

    Returns:
        Tuple of (modified predictions, metadata dict with stats)
    """
    if not enabled:
        return predictions, {"cap_applied": False, "reason": "disabled"}

    if not predictions:
        return predictions, {"cap_applied": False, "reason": "no_predictions"}

    # Collect all value bets with their prediction index
    # ABE P0-1: Only NS + non-frozen predictions participate in draw cap
    all_value_bets = []
    for i, pred in enumerate(predictions):
        if pred.get("is_frozen") or pred.get("status") != "NS":
            continue
        value_bets = pred.get("value_bets", []) or []
        for vb in value_bets:
            all_value_bets.append({
                "pred_idx": i,
                "match_id": pred.get("match_id"),
                "outcome": vb.get("outcome"),
                "edge": vb.get("edge", 0),
                "value_bet": vb,
            })

    if not all_value_bets:
        return predictions, {"cap_applied": False, "reason": "no_value_bets"}

    # Separate draws and non-draws
    draws = [vb for vb in all_value_bets if vb["outcome"] == "draw"]
    others = [vb for vb in all_value_bets if vb["outcome"] != "draw"]

    n_draws = len(draws)
    n_others = len(others)
    n_total = n_draws + n_others

    # Calculate max draws so that final concentration = max_draw_share
    if n_others == 0:
        # All value bets are draws - keep top by edge
        max_draws = max(1, int(n_draws * max_draw_share))
    else:
        # max_draws / (max_draws + n_others) = max_draw_share
        # max_draws = (max_draw_share * n_others) / (1 - max_draw_share)
        max_draws = int((max_draw_share * n_others) / (1 - max_draw_share))

    if n_draws <= max_draws:
        return predictions, {
            "cap_applied": False,
            "reason": "within_limit",
            "n_draws": n_draws,
            "n_others": n_others,
            "n_total": n_total,
            "draw_share": round(n_draws / n_total * 100, 1) if n_total > 0 else 0,
            "max_draws_allowed": max_draws,
        }

    # Sort draws by edge descending, keep top max_draws
    draws_sorted = sorted(draws, key=lambda x: x["edge"], reverse=True)
    draws_to_keep = set(id(d) for d in draws_sorted[:max_draws])
    draws_to_remove = [d for d in draws if id(d) not in draws_to_keep]

    n_removed = len(draws_to_remove)

    # Remove draw value bets from predictions
    # Group by prediction index
    pred_draw_removals = {}
    for d in draws_to_remove:
        pred_idx = d["pred_idx"]
        if pred_idx not in pred_draw_removals:
            pred_draw_removals[pred_idx] = []
        pred_draw_removals[pred_idx].append(d["value_bet"])

    # Create modified predictions
    modified_predictions = []
    for i, pred in enumerate(predictions):
        if i in pred_draw_removals:
            # Filter out the draw value bets that should be removed
            vb_to_remove = pred_draw_removals[i]
            current_vbs = pred.get("value_bets", []) or []
            filtered_vbs = [vb for vb in current_vbs if vb not in vb_to_remove]

            # Create a copy with modified value_bets
            modified_pred = dict(pred)
            modified_pred["value_bets"] = filtered_vbs if filtered_vbs else None
            modified_pred["has_value_bet"] = len(filtered_vbs) > 0 if filtered_vbs else False

            # Update best_value_bet if needed
            if filtered_vbs:
                best = max(filtered_vbs, key=lambda x: x.get("expected_value", x.get("ev", 0)))
                modified_pred["best_value_bet"] = best
            else:
                modified_pred["best_value_bet"] = None

            # Add metadata about cap application
            if "policy_metadata" not in modified_pred:
                modified_pred["policy_metadata"] = {}
            modified_pred["policy_metadata"]["draw_cap_applied"] = True
            modified_pred["policy_metadata"]["draw_removed"] = True

            modified_predictions.append(modified_pred)
        else:
            modified_predictions.append(pred)

    # Calculate final stats
    final_draws = max_draws
    final_total = final_draws + n_others
    final_draw_share = round(final_draws / final_total * 100, 1) if final_total > 0 else 0

    metadata = {
        "cap_applied": True,
        "n_draws_original": n_draws,
        "n_draws_after": final_draws,
        "n_draws_removed": n_removed,
        "n_others": n_others,
        "n_total_original": n_total,
        "n_total_after": final_total,
        "draw_share_original": round(n_draws / n_total * 100, 1) if n_total > 0 else 0,
        "draw_share_after": final_draw_share,
        "max_draw_share_config": max_draw_share,
    }

    logger.info(
        f"[POLICY] Draw cap applied: {n_draws}→{final_draws} draws "
        f"({metadata['draw_share_original']}%→{final_draw_share}%), "
        f"removed {n_removed} low-edge draws"
    )

    return modified_predictions, metadata


def apply_market_anchor(
    predictions: list[dict],
    alpha_default: float = 0.0,
    league_overrides: dict[int, float] | None = None,
    enabled: bool = False,
) -> tuple[list[dict], dict]:
    """
    Blend model probabilities with de-vigged market odds for low-signal leagues.

    ABE P0 2026-02-08: Only applies to leagues explicitly in league_overrides.
    Only NS (not-started) predictions are eligible. Frozen/LIVE/FT are skipped.

    Args:
        predictions: List of prediction dicts from engine.predict()
        alpha_default: Default blend weight (0.0 = model pure, 1.0 = market pure)
        league_overrides: Per-league alpha overrides {128: 1.0, ...}
        enabled: Feature flag

    Returns:
        Tuple of (modified predictions, metadata dict)
    """
    if not enabled:
        return predictions, {"anchor_applied": False, "reason": "disabled"}

    if not predictions:
        return predictions, {"anchor_applied": False, "reason": "no_predictions"}

    from app.ml.devig import devig_proportional

    if league_overrides is None:
        league_overrides = {}

    modified_predictions = []
    n_anchored = 0
    n_skipped_not_ns = 0
    n_skipped_no_override = 0
    n_skipped_no_market = 0
    n_skipped_invalid_odds = 0
    leagues_affected: dict[int, int] = {}

    for pred in predictions:
        # P0-2: Only anchor NS predictions that are not frozen
        status = pred.get("status")
        if status != "NS" or pred.get("is_frozen"):
            modified_predictions.append(pred)
            n_skipped_not_ns += 1
            continue

        # Skip Family S predictions (already final from cascade, avoid double-anchor)
        if pred.get("skip_market_anchor"):
            modified_predictions.append(pred)
            continue

        # Determine alpha for this league
        league_id = pred.get("league_id")
        alpha = league_overrides.get(league_id, alpha_default) if league_id else alpha_default

        # P0-1: Skip if alpha is 0 (not in overrides and default is 0)
        if alpha == 0.0:
            modified_predictions.append(pred)
            n_skipped_no_override += 1
            continue

        # Check market odds availability
        market_odds = pred.get("market_odds")
        if not market_odds:
            modified_predictions.append(pred)
            n_skipped_no_market += 1
            continue

        h_odds = market_odds.get("home")
        d_odds = market_odds.get("draw")
        a_odds = market_odds.get("away")

        # P0-3: Strict validation — all odds must be > 1.0
        if not (h_odds and d_odds and a_odds and h_odds > 1.0 and d_odds > 1.0 and a_odds > 1.0):
            modified_predictions.append(pred)
            n_skipped_invalid_odds += 1
            continue

        # De-vig market odds to fair probabilities
        p_mkt_h, p_mkt_d, p_mkt_a = devig_proportional(h_odds, d_odds, a_odds)

        # Get current model probabilities
        probs = pred.get("probabilities", {})
        p_mod_h = probs.get("home", 1 / 3)
        p_mod_d = probs.get("draw", 1 / 3)
        p_mod_a = probs.get("away", 1 / 3)

        # Create modified copy
        modified_pred = dict(pred)

        # P1-1: Preserve model probabilities in dedicated field
        modified_pred["model_probabilities"] = dict(probs)
        if not pred.get("raw_probabilities"):
            modified_pred["raw_probabilities"] = dict(probs)

        # Blend: p_served = (1 - alpha) * p_model + alpha * p_market
        h_blend = (1 - alpha) * p_mod_h + alpha * p_mkt_h
        d_blend = (1 - alpha) * p_mod_d + alpha * p_mkt_d
        a_blend = (1 - alpha) * p_mod_a + alpha * p_mkt_a

        # Renormalize (safety net)
        total = h_blend + d_blend + a_blend
        if total > 0.001:
            h_blend /= total
            d_blend /= total
            a_blend /= total

        modified_pred["probabilities"] = {
            "home": round(h_blend, 4),
            "draw": round(d_blend, 4),
            "away": round(a_blend, 4),
        }

        # Recalculate fair_odds from blended probs
        modified_pred["fair_odds"] = {
            "home": round(1 / h_blend, 2) if h_blend > 0 else None,
            "draw": round(1 / d_blend, 2) if d_blend > 0 else None,
            "away": round(1 / a_blend, 2) if a_blend > 0 else None,
        }

        # P0-4: When alpha >= 0.80, clear value bets (probs ≈ market, no edge)
        if alpha >= 0.80:
            modified_pred["value_bets"] = None
            modified_pred["has_value_bet"] = False
            modified_pred["best_value_bet"] = None
            warnings = modified_pred.get("warnings") or []
            warnings.append("MARKET_ANCHORED")
            modified_pred["warnings"] = warnings

        # Metadata per prediction
        if "policy_metadata" not in modified_pred:
            modified_pred["policy_metadata"] = {}
        modified_pred["policy_metadata"]["market_anchor"] = {
            "applied": True,
            "alpha": alpha,
            "market_source": "match_odds_bet365",  # P1-2
            "league_id": league_id,
        }

        modified_predictions.append(modified_pred)
        n_anchored += 1
        leagues_affected[league_id] = leagues_affected.get(league_id, 0) + 1

    metadata = {
        "anchor_applied": n_anchored > 0,
        "n_anchored": n_anchored,
        "n_skipped_not_ns": n_skipped_not_ns,
        "n_skipped_no_override": n_skipped_no_override,
        "n_skipped_no_market": n_skipped_no_market,
        "n_skipped_invalid_odds": n_skipped_invalid_odds,
        "n_total": len(predictions),
        "leagues_affected": leagues_affected,
    }

    if n_anchored > 0:
        logger.info(
            f"[MARKET-ANCHOR] Blended {n_anchored} predictions | "
            f"not_ns={n_skipped_not_ns} no_override={n_skipped_no_override} "
            f"no_market={n_skipped_no_market} invalid_odds={n_skipped_invalid_odds} | "
            f"leagues={leagues_affected}"
        )

    return modified_predictions, metadata


def get_policy_config() -> dict:
    """Get policy configuration from settings."""
    from app.config import get_settings

    settings = get_settings()

    # Parse league overrides: "128:1.0,239:0.8" → {128: 1.0, 239: 0.8}
    league_overrides = {}
    if settings.MARKET_ANCHOR_LEAGUE_OVERRIDES:
        for pair in settings.MARKET_ANCHOR_LEAGUE_OVERRIDES.split(","):
            pair = pair.strip()
            if ":" in pair:
                try:
                    lid, alpha = pair.split(":", 1)
                    league_overrides[int(lid.strip())] = float(alpha.strip())
                except (ValueError, TypeError):
                    pass

    return {
        "draw_cap_enabled": settings.POLICY_DRAW_CAP_ENABLED,
        "max_draw_share": settings.POLICY_MAX_DRAW_SHARE,
        "edge_threshold": settings.POLICY_EDGE_THRESHOLD,
        # Market anchor
        "market_anchor_enabled": settings.MARKET_ANCHOR_ENABLED,
        "market_anchor_alpha_default": settings.MARKET_ANCHOR_ALPHA_DEFAULT,
        "market_anchor_league_overrides": league_overrides,
    }
