"""
Asymmetric League-Router — GDT Mandato M3 / Mandato D.

Routes predictions through different strategies based on league tier:
  - TIER 1 (Big 5 European): Baseline model, high market anchor alpha
  - TIER 2 (Peripheral, MTV-neutral): Baseline model, moderate market anchor
  - TIER 3 (Peripheral, MTV-positive): Family S model (MTV features injected)

Tier 3 pruned from 10 to 5 leagues (Mandato D, 2026-02-17) based on
PIT-padded Feature Lab results: strongest MTV signal + best data coverage.

SteamChaser: For T1/T2, talent_delta is computed and logged (forward data
collection) but NOT injected into predictions. For T3, talent_delta is
injected into the feature set when available.

Status: INFRASTRUCTURE READY.
  - Router classification: ACTIVE
  - MTV feature injection: GATED behind LEAGUE_ROUTER_MTV_ENABLED flag
  - Family S model: v2.0-tier3-family_s (loaded via app.ml.family_s)
"""

import logging
from typing import Optional

logger = logging.getLogger("futbolstats.league_router")

# ═══════════════════════════════════════════════════════════════════════════
# TIER CLASSIFICATION (GDT Mandato M3, 2026-02-15)
# ═══════════════════════════════════════════════════════════════════════════

# Tier 1: Big 5 European — most efficient markets, model loses to market
TIER_1 = {
    39,   # Premier League (England)
    61,   # Ligue 1 (France)
    78,   # Bundesliga (Germany)
    135,  # Serie A (Italy)
    140,  # La Liga (Spain)
}

# Tier 3: Family S — 5 leagues with strongest MTV signal (Mandato D, 2026-02-17)
# Pruned from 10 to 5 based on PIT-padded Feature Lab + odds/MTV coverage
TIER_3 = {
    88,   # Eredivisie (Netherlands)     Δ=-0.00987
    94,   # Primeira Liga (Portugal)     Δ=-0.01085
    144,  # Belgian Pro League           (strong odds+MTV coverage)
    203,  # Süper Lig (Turkey)           Δ=-0.00624
    265,  # Primera División (Chile)     Δ=-0.01719 (best)
}

# Tier 2: Everything else (peripheral leagues where MTV was neutral/negative,
# international competitions, cups, etc.)
# Not enumerated — it's the default tier for any league_id not in T1 or T3.


def get_league_tier(league_id: int) -> int:
    """
    Classify a league into its prediction tier.

    Returns:
        1 for Tier 1 (Big 5), 3 for Tier 3 (MTV winners), 2 for everything else.
    """
    if league_id in TIER_1:
        return 1
    if league_id in TIER_3:
        return 3
    return 2


def should_inject_mtv(league_id: int, mtv_enabled: bool = False) -> bool:
    """
    Whether MTV features should be injected into predictions for this league.

    Only Tier 3 leagues get MTV injection, and only when the feature flag
    is enabled (requires trained Family S model).

    Args:
        league_id: The league ID.
        mtv_enabled: Global feature flag (from LEAGUE_ROUTER_MTV_ENABLED).

    Returns:
        True if MTV features should be used for prediction.
    """
    if not mtv_enabled:
        return False
    return league_id in TIER_3


def should_compute_talent_delta(league_id: int) -> bool:
    """
    Whether talent_delta should be computed for this match.

    Returns True for ALL leagues — Tier 3 for injection, Tier 1/2 for
    SteamChaser data collection (forward logging).
    """
    return True


def get_prediction_strategy(league_id: int, mtv_enabled: bool = False) -> dict:
    """
    Get the full prediction strategy for a league.

    Returns a dict describing how to handle predictions for this league.
    """
    tier = get_league_tier(league_id)
    inject_mtv = should_inject_mtv(league_id, mtv_enabled)

    strategy = {
        "league_id": league_id,
        "tier": tier,
        "inject_mtv": inject_mtv,
        "compute_talent_delta": True,  # Always compute for data collection
        "talent_delta_purpose": "injection" if inject_mtv else "steamchaser_logging",
    }

    if tier == 1:
        strategy["label"] = "ODDS_CENTRIC"
        strategy["description"] = "Big 5: Baseline + high market anchor"
    elif tier == 3 and inject_mtv:
        strategy["label"] = "FAMILY_S"
        strategy["description"] = "Tier 3: Family S model with MTV features"
    elif tier == 3:
        strategy["label"] = "FAMILY_S_PENDING"
        strategy["description"] = "Tier 3: MTV identified but model not yet active"
    else:
        strategy["label"] = "BASELINE"
        strategy["description"] = "Tier 2: Baseline model"

    return strategy


def get_tier_summary() -> dict:
    """Return summary of tier classification for ops/diagnostics."""
    return {
        "tier_1_big5": sorted(TIER_1),
        "tier_1_count": len(TIER_1),
        "tier_3_mtv_winners": sorted(TIER_3),
        "tier_3_count": len(TIER_3),
        "tier_2_note": "All other leagues (default tier)",
    }
