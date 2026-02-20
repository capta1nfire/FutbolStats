"""
Asymmetric League-Router — GDT Mandato M3 / Mandato D / SSOT Serving.

Routes predictions through different strategies based on league tier.
Config loaded from `league_serving_config` DB table at startup, with
hardcoded fallbacks if DB is unavailable.

Tier classification:
  - TIER 1 (Big 5 European): Baseline model, high market anchor alpha
  - TIER 2 (Peripheral, default): Baseline model
  - TIER 3 (MTV winners): Family S model (MTV features injected)

Two-Stage / One-Stage routing:
  - TS_LEAGUES: leagues with guardrail-PASS W3 paired bootstrap
  - OS_LEAGUES: leagues that stay on baseline (guardrail-FAIL or OS winners)

SSOT: league_serving_config table is the single source of truth.
Sets (TIER_1, TIER_3, TS_LEAGUES, OS_LEAGUES) are mutated in-place on reload
to preserve references held by importers [P0-E].

Status: ACTIVE — DB-backed with TTL auto-refresh [P0-F].
"""

import logging
import time
from typing import Optional

logger = logging.getLogger("futbolstats.league_router")

# Strategy constants
STRATEGY_BASELINE = "baseline"
STRATEGY_TWOSTAGE = "twostage"
STRATEGY_FAMILY_S = "family_s"

# ═══════════════════════════════════════════════════════════════════════════
# TIER CLASSIFICATION — Hardcoded fallbacks (used if DB unavailable)
# ═══════════════════════════════════════════════════════════════════════════

# Tier 1: Big 5 European
TIER_1 = {
    39,   # Premier League (England)
    61,   # Ligue 1 (France)
    78,   # Bundesliga (Germany)
    135,  # Serie A (Italy)
    140,  # La Liga (Spain)
}

# Tier 3: Family S — 10 leagues where MTV HELPS per Mega-Pool V2
TIER_3 = {
    88,   # Eredivisie (Netherlands)
    94,   # Primeira Liga (Portugal)
    203,  # Süper Lig (Turkey)
    242,  # LDU (Ecuador)
    262,  # Liga MX (Mexico)
    265,  # Primera División (Chile)
    268,  # Primera División (Uruguay)
    281,  # Liga 1 (Peru)
    299,  # Primera División (Venezuela)
    344,  # División Profesional (Bolivia)
}

# Two-Stage W3 guardrail-PASS leagues (15 leagues)
TS_LEAGUES = {
    39, 61, 78, 88, 94, 140, 203, 239, 253, 262, 265, 268, 299, 307, 344,
}

# One-Stage baseline leagues (8 leagues: 4 OS-winners + 4 guardrail-FAIL)
OS_LEAGUES = {
    128, 135, 242, 250,  # OS winners
    40, 71, 144, 281,    # guardrail FAIL
}

# ═══════════════════════════════════════════════════════════════════════════
# SERVING CONFIG CACHE — Loaded from league_serving_config table
# ═══════════════════════════════════════════════════════════════════════════

_serving_configs = {}       # league_id -> dict row from DB
_configs_loaded_at = 0.0    # monotonic timestamp of last load [P0-F]
_CONFIG_TTL_SECONDS = 60    # TTL for lazy reload [P0-F]


async def load_serving_configs(session) -> int:
    """
    Load league_serving_config from DB into memory cache.

    [P0-E] Mutates existing sets in-place to preserve references held by importers.
    [P0-F] Updates _configs_loaded_at for TTL-based refresh.

    Returns number of configs loaded.
    """
    global _configs_loaded_at
    from sqlalchemy import text

    rows = (await session.execute(
        text("SELECT league_id, preferred_strategy, anchor_alpha, model_version, "
             "prerequisites, fallback_strategy, notes, updated_at, updated_by "
             "FROM league_serving_config")
    )).fetchall()

    if not rows:
        logger.warning("league_serving_config table is empty, keeping current sets")
        _configs_loaded_at = time.monotonic()
        return 0

    # Build config dict
    new_configs = {}
    for r in rows:
        new_configs[r[0]] = {
            "league_id": r[0],
            "preferred_strategy": r[1],
            "anchor_alpha": r[2],
            "model_version": r[3],
            "prerequisites": r[4] or {},
            "fallback_strategy": r[5],
            "notes": r[6],
            "updated_at": r[7],
            "updated_by": r[8],
        }

    _serving_configs.clear()
    _serving_configs.update(new_configs)

    # [P0-E] Derive sets from DB config and mutate in-place
    new_tier3 = set()
    new_ts = set()
    new_os = set()

    for lid, cfg in _serving_configs.items():
        strategy = cfg["preferred_strategy"]
        fallback = cfg["fallback_strategy"]

        if strategy == STRATEGY_FAMILY_S:
            new_tier3.add(lid)
            # Family S leagues may also be TS-eligible via fallback
            if fallback == STRATEGY_TWOSTAGE:
                new_ts.add(lid)
        elif strategy == STRATEGY_TWOSTAGE:
            new_ts.add(lid)
        elif strategy == STRATEGY_BASELINE:
            new_os.add(lid)

    # Mutate in-place (importers keep same reference)
    TIER_3.clear()
    TIER_3.update(new_tier3)
    TS_LEAGUES.clear()
    TS_LEAGUES.update(new_ts)
    OS_LEAGUES.clear()
    OS_LEAGUES.update(new_os)
    # TIER_1 stays hardcoded — Big 5 never changes via DB

    _configs_loaded_at = time.monotonic()

    logger.info(
        "Serving configs loaded: %d leagues | TIER_3=%s TS=%d OS=%d",
        len(_serving_configs), sorted(TIER_3), len(TS_LEAGUES), len(OS_LEAGUES)
    )
    return len(_serving_configs)


async def maybe_refresh_configs(session) -> bool:
    """
    [P0-F] Lazy reload: if cache is older than TTL, reload from DB.

    Called before prediction batches. Idempotent, ~1ms if no reload needed.
    Returns True if configs were reloaded.
    """
    if _configs_loaded_at == 0.0:
        # Never loaded — skip (startup handles initial load)
        return False

    elapsed = time.monotonic() - _configs_loaded_at
    if elapsed < _CONFIG_TTL_SECONDS:
        return False

    try:
        n = await load_serving_configs(session)
        logger.debug("TTL refresh: reloaded %d serving configs", n)
        return True
    except Exception as e:
        logger.warning("TTL refresh failed, keeping stale cache: %s", e)
        return False


def get_serving_config(league_id):
    """
    Get serving config for a league.

    Returns dict with strategy/alpha/model_version/fallback or defaults.
    """
    if league_id in _serving_configs:
        return dict(_serving_configs[league_id])  # Return copy

    # Default for unconfigured leagues
    return {
        "league_id": league_id,
        "preferred_strategy": STRATEGY_BASELINE,
        "anchor_alpha": 0.0,
        "model_version": None,
        "prerequisites": {},
        "fallback_strategy": STRATEGY_BASELINE,
        "notes": None,
        "updated_at": None,
        "updated_by": "default",
    }


def get_league_overrides_from_cache():
    """
    Return {league_id: alpha} for market anchor policy.

    Called by policy.py to replace MARKET_ANCHOR_LEAGUE_OVERRIDES env var.
    Only includes leagues with anchor_alpha > 0.
    """
    if not _serving_configs:
        return {}  # Cache empty — caller should fallback to env var

    return {
        lid: cfg["anchor_alpha"]
        for lid, cfg in _serving_configs.items()
        if cfg["anchor_alpha"] > 0.0
    }


# ═══════════════════════════════════════════════════════════════════════════
# BACKWARD-COMPATIBLE API — Same functions, now backed by cache
# ═══════════════════════════════════════════════════════════════════════════

def get_league_tier(league_id):
    """
    Classify a league into its prediction tier.

    Returns: 1 for Tier 1 (Big 5), 3 for Tier 3 (Family S), 2 for everything else.
    """
    if league_id in TIER_1:
        return 1
    if league_id in TIER_3:
        return 3
    return 2


def is_ts_league(league_id):
    """Whether this league should use Two-Stage W3 predictions."""
    return league_id in TS_LEAGUES


def should_inject_mtv(league_id, mtv_enabled=False):
    """
    Whether MTV features should be injected into predictions for this league.
    Only Tier 3 + flag enabled.
    """
    if not mtv_enabled:
        return False
    return league_id in TIER_3


def should_compute_talent_delta(league_id):
    """Whether talent_delta should be computed (always True — data collection)."""
    return True


def get_prediction_strategy(league_id, mtv_enabled=False):
    """
    Get the full prediction strategy for a league.

    Returns a dict describing how to handle predictions.
    """
    tier = get_league_tier(league_id)
    inject_mtv = should_inject_mtv(league_id, mtv_enabled)

    strategy = {
        "league_id": league_id,
        "tier": tier,
        "inject_mtv": inject_mtv,
        "compute_talent_delta": True,
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


def get_tier_summary():
    """Return summary of tier classification for ops/diagnostics."""
    config_source = "db" if _serving_configs else "hardcoded_fallback"
    return {
        "tier_1_big5": sorted(TIER_1),
        "tier_1_count": len(TIER_1),
        "tier_3_family_s": sorted(TIER_3),
        "tier_3_count": len(TIER_3),
        "ts_leagues": sorted(TS_LEAGUES),
        "ts_count": len(TS_LEAGUES),
        "os_leagues": sorted(OS_LEAGUES),
        "os_count": len(OS_LEAGUES),
        "config_source": config_source,
        "configs_loaded": len(_serving_configs),
        "tier_2_note": "All other leagues (default tier)",
    }
