"""
Betting policy utilities for value bet selection.

FASE 1: Draw cap to prevent over-concentration in draw bets.

Feature flags (from config):
- POLICY_DRAW_CAP_ENABLED: Enable/disable draw cap
- POLICY_MAX_DRAW_SHARE: Maximum fraction of value bets that can be draws (0.35)
- POLICY_EDGE_THRESHOLD: Minimum edge for value bet (0.05)
"""

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
    all_value_bets = []
    for i, pred in enumerate(predictions):
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


def get_policy_config() -> dict:
    """Get policy configuration from settings."""
    from app.config import get_settings

    settings = get_settings()
    return {
        "draw_cap_enabled": settings.POLICY_DRAW_CAP_ENABLED,
        "max_draw_share": settings.POLICY_MAX_DRAW_SHARE,
        "edge_threshold": settings.POLICY_EDGE_THRESHOLD,
    }
