"""
Kelly Criterion stake sizing — pure math functions.

GDT Epoch 2: Trading Core (2026-02-22).

Functions:
- kelly_stake: Full Kelly f* = (bp - q) / b
- fractional_kelly: Conservative sizing (default Quarter-Kelly)
- apply_risk_overrides: EV floor, high-odds penalty, max-stake cap
- enrich_value_bet_with_kelly: Entry point — enriches a value_bet dict
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def kelly_stake(prob: float, odds: float) -> float:
    """
    Full Kelly stake fraction: f* = (bp - q) / b

    GDT Guardrail #1:
    - b = odds - 1.0. If odds <= 1.0 → return 0.0 (ZeroDivisionError protection)
    - No-Shorting Rule: max(0.0, f*) — only long positions
    - ABE: validate prob ∈ (0, 1) and odds > 1.0 defensively
    """
    if not (0 < prob < 1) or odds <= 1.0:
        return 0.0

    b = odds - 1.0
    q = 1.0 - prob
    f_star = (b * prob - q) / b
    return max(0.0, f_star)


def fractional_kelly(prob: float, odds: float, fraction: float = 0.25) -> float:
    """Quarter-Kelly by default. Returns fraction × kelly_stake."""
    return kelly_stake(prob, odds) * fraction


def apply_risk_overrides(
    stake: float,
    ev: float,
    odds: float,
    *,
    min_ev: float = 0.03,
    high_odds_threshold: float = 5.0,
    high_odds_factor: float = 0.5,
    max_stake_pct: float = 0.05,
) -> tuple[float, list[str] | None]:
    """
    Apply risk filters to a Kelly stake. Returns (adjusted_stake, flags).

    GDT Guardrail #3 (audit traceability):
    - EV < min_ev → stake = 0, flag MIN_EV_REJECTED
    - Odds > threshold → stake × factor, flag HIGH_ODDS_PENALTY
    - Stake > max_stake_pct → cap, flag MAX_MATCH_CAP_APPLIED

    Flags are cumulative. Returns None if no flags applied.
    """
    flags: list[str] = []

    # Filter 1: EV floor
    if ev < min_ev:
        return 0.0, ["MIN_EV_REJECTED"]

    # Filter 2: High-odds penalty
    if odds > high_odds_threshold:
        stake *= high_odds_factor
        flags.append("HIGH_ODDS_PENALTY")

    # Filter 3: Single-bet cap (individual, before match-level cap)
    if stake > max_stake_pct:
        stake = max_stake_pct
        flags.append("MAX_MATCH_CAP_APPLIED")

    return stake, flags if flags else None


def enrich_value_bet_with_kelly(
    value_bet: dict,
    *,
    fraction: float = 0.25,
    bankroll_units: float = 1000.0,
    min_ev: float = 0.03,
    high_odds_threshold: float = 5.0,
    high_odds_factor: float = 0.5,
    max_stake_pct: float = 0.05,
) -> dict:
    """
    Enrich a value_bet dict with Kelly fields. Returns a NEW dict (never mutates input).

    Added fields: kelly_raw, kelly_fraction, suggested_stake, stake_units, stake_flags.
    Rounding (GDT): kelly_raw/kelly_fraction/suggested_stake → 4 decimals, stake_units → 2.
    """
    vb = dict(value_bet)

    prob = vb.get("our_probability", 0)
    odds = vb.get("market_odds", 0)
    ev = vb.get("expected_value", 0)

    raw = kelly_stake(prob, odds)
    frac = raw * fraction

    adjusted, flags = apply_risk_overrides(
        frac, ev, odds,
        min_ev=min_ev,
        high_odds_threshold=high_odds_threshold,
        high_odds_factor=high_odds_factor,
        max_stake_pct=max_stake_pct,
    )

    vb["kelly_raw"] = round(raw, 4)
    vb["kelly_fraction"] = round(frac, 4)
    vb["suggested_stake"] = round(adjusted, 4)
    vb["stake_units"] = round(adjusted * bankroll_units, 2)
    vb["stake_flags"] = flags

    return vb
