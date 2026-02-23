"""
Serving Telemetry — stamps each prediction with metadata about what was served.

Pure function, no side effects. Called after all overlays (TS, Family S, market anchor)
and before PredictionItem construction.
"""

import logging

logger = logging.getLogger("futbolstats.serving_telemetry")


def stamp_serving_metadata(predictions, baseline_version):
    """
    Add serving_metadata dict to each prediction in the list.

    Args:
        predictions: list of prediction dicts (mutated in place)
        baseline_version: str, version of the baseline model (e.g. "v1.0.0")
    """
    from app.ml.league_router import get_serving_config, _serving_configs

    config_source = "db" if _serving_configs else "hardcoded_fallback"

    for pred in predictions:
        league_id = pred.get("league_id")
        if not league_id:
            continue

        cfg = get_serving_config(league_id)
        preferred = cfg["preferred_strategy"]
        fallback = cfg["fallback_strategy"]
        alpha = cfg["anchor_alpha"]

        # Determine what was actually served
        served_strategy = preferred
        served_version = cfg.get("model_version") or baseline_version
        fallback_reason = None

        # Check if a model overlay was applied
        if pred.get("served_from_family_s"):
            served_strategy = "family_s"
            served_version = (pred.get("model_version_served")
                              or pred.get("family_s_model_version")
                              or served_version)
        elif pred.get("latam_overlay"):
            # LATAM overlay (v1.3.0, 18f with geo) — treated as baseline variant
            served_strategy = "latam_baseline"
            served_version = pred.get("model_version_served", served_version)
        elif pred.get("model_version_served"):
            # TS overlay sets model_version_served
            served_strategy = "twostage"
            served_version = pred["model_version_served"]
        elif preferred != "baseline":
            # Preferred was TS or Family S but no overlay was applied → fallback
            served_strategy = fallback
            served_version = baseline_version
            fallback_reason = "model_not_loaded_or_no_prediction"

        # Actual alpha applied (TS/FS skip anchor, so alpha=0 for them)
        # policy.py stores anchor info in policy_metadata.market_anchor
        # LATAM baseline also receives market anchor downstream
        alpha_applied = 0.0
        if served_strategy in ("baseline", "latam_baseline"):
            anchor_meta = (pred.get("policy_metadata") or {}).get("market_anchor")
            if anchor_meta and anchor_meta.get("applied"):
                alpha_applied = anchor_meta.get("alpha", alpha)

        pred["serving_metadata"] = {
            "preferred_strategy": preferred,
            "served_strategy": served_strategy,
            "served_model_version": served_version,
            "alpha_applied": alpha_applied,
            "fallback_reason": fallback_reason,
            "config_source": config_source,
        }
