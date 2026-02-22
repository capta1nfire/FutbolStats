"""
Kelly sizing pipeline — transforms predictions list, same contract as policy.py.

GDT Epoch 2: Trading Core (2026-02-22).
Pipeline order (GDT VETO): finalize → draw_cap → kelly_sizing → serve/freeze.
Kelly is the logically FINAL layer before DB/serve.
"""

from __future__ import annotations

import logging

from app.trading.kelly import enrich_value_bet_with_kelly

logger = logging.getLogger(__name__)


def apply_kelly_sizing(
    predictions: list[dict],
    *,
    enabled: bool = False,
    fraction: float = 0.25,
    bankroll_units: float = 1000.0,
    min_ev: float = 0.03,
    high_odds_threshold: float = 5.0,
    high_odds_factor: float = 0.5,
    max_stake_pct: float = 0.05,
) -> tuple[list[dict], dict]:
    """
    Apply Kelly stake sizing to all value_bets in predictions.

    ABE: if enabled=False, return list untouched (zero CPU).
    Respects PIT: frozen/FT predictions are untouchable.

    GDT Guardrail #2 (Match Exposure Cap): after enriching all value_bets
    for a single match, if sum(suggested_stake) > max_stake_pct, scale ALL
    stakes proportionally. The 5% cap is per-match, not per-bet.

    Returns:
        (predictions, metadata) tuple — same contract as apply_draw_cap().
    """
    if not enabled:
        return predictions, {"kelly_applied": False, "reason": "disabled"}

    if not predictions:
        return predictions, {"kelly_applied": False, "reason": "no_predictions"}

    n_enriched = 0
    n_skipped_frozen = 0
    n_match_capped = 0
    total_stake_units = 0.0

    kelly_kwargs = dict(
        fraction=fraction,
        bankroll_units=bankroll_units,
        min_ev=min_ev,
        high_odds_threshold=high_odds_threshold,
        high_odds_factor=high_odds_factor,
        max_stake_pct=max_stake_pct,
    )

    modified_predictions = []
    for pred in predictions:
        # PIT: frozen/FT untouchable
        if pred.get("is_frozen") or pred.get("status") != "NS":
            modified_predictions.append(pred)
            n_skipped_frozen += 1
            continue

        value_bets = pred.get("value_bets")
        if not value_bets:
            modified_predictions.append(pred)
            continue

        # Enrich each value_bet with Kelly fields
        enriched_vbs = [
            enrich_value_bet_with_kelly(vb, **kelly_kwargs)
            for vb in value_bets
        ]

        # GDT Guardrail #2: Match Exposure Cap
        sum_stakes = sum(vb["suggested_stake"] for vb in enriched_vbs)
        if sum_stakes > max_stake_pct:
            # Proportional reduction — safe denominator (GDT)
            factor = max_stake_pct / max(sum_stakes, 1e-9)
            for vb in enriched_vbs:
                vb["suggested_stake"] = round(vb["suggested_stake"] * factor, 4)
                vb["stake_units"] = round(vb["suggested_stake"] * bankroll_units, 2)
                # Add match cap flag
                existing = vb.get("stake_flags") or []
                if "MAX_MATCH_CAP_APPLIED" not in existing:
                    existing.append("MAX_MATCH_CAP_APPLIED")
                vb["stake_flags"] = existing
            n_match_capped += 1

        # Build modified prediction
        modified_pred = dict(pred)
        modified_pred["value_bets"] = enriched_vbs

        # Update best_value_bet if present
        active_vbs = [vb for vb in enriched_vbs if vb.get("suggested_stake", 0) > 0]
        if active_vbs:
            best = max(active_vbs, key=lambda x: x.get("suggested_stake", 0))
            modified_pred["best_value_bet"] = best

        n_enriched += 1
        total_stake_units += sum(vb.get("stake_units", 0) for vb in enriched_vbs)
        modified_predictions.append(modified_pred)

    metadata = {
        "kelly_applied": n_enriched > 0,
        "n_enriched": n_enriched,
        "n_skipped_frozen": n_skipped_frozen,
        "n_match_capped": n_match_capped,
        "total_stake_units": round(total_stake_units, 2),
        "fraction": fraction,
        "bankroll_units": bankroll_units,
    }

    if n_enriched > 0:
        logger.info(
            f"[KELLY] Sized {n_enriched} predictions | "
            f"match_capped={n_match_capped} | "
            f"total_units={total_stake_units:.1f} | "
            f"fraction={fraction}"
        )

    return modified_predictions, metadata
