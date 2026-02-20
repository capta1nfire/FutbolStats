"""Family S prediction engine for Tier 3 leagues (Mandato D).

Loads the Family S model from model_snapshots.
Uses FamilySEngine, a subclass of XGBoostEngine with expanded features:
  - 14 baseline (same as XGBoostEngine.FEATURE_COLUMNS)
  - 3 odds features (odds_home, odds_draw, odds_away)
  - 4 MTV features (home_talent_delta, away_talent_delta, talent_delta_diff, shock_magnitude)

v2.1: Removed 3 redundant competitiveness features (abs_attack_diff,
abs_defense_diff, abs_strength_gap) per ablation (Δ ≈ 0). Expanded
from 5 to 10 Tier 3 leagues per Mega-Pool V2 revalidation.

FamilySEngine overrides FEATURE_COLUMNS (21 total) so _prepare_features() and
_get_model_expected_features() use the correct feature list in both
training and serving.

Status: GATED behind LEAGUE_ROUTER_MTV_ENABLED flag.
"""

import logging
from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.ml.engine import XGBoostEngine
from app.models import ModelSnapshot

logger = logging.getLogger("futbolstats.family_s")

# ═════════════════════════════════════════════════════════════════════════════
# FamilySEngine — XGBoostEngine with expanded feature set (P0-2)
# ═════════════════════════════════════════════════════════════════════════════


class FamilySEngine(XGBoostEngine):
    """XGBoostEngine subclass with 21-feature set for Tier 3 MTV model.

    Inherits all training, prediction, and serialization logic from
    XGBoostEngine. Only overrides FEATURE_COLUMNS to include odds + MTV.

    _prepare_features() and _get_model_expected_features() are inherited
    and use this FEATURE_COLUMNS, ensuring train/serve feature parity.
    """

    FEATURE_COLUMNS = [
        # ── 14 baseline (same as XGBoostEngine) ──
        "home_goals_scored_avg",
        "home_goals_conceded_avg",
        "home_shots_avg",
        "home_corners_avg",
        "home_rest_days",
        "home_matches_played",
        "away_goals_scored_avg",
        "away_goals_conceded_avg",
        "away_shots_avg",
        "away_corners_avg",
        "away_rest_days",
        "away_matches_played",
        "goal_diff_avg",
        "rest_diff",
        # ── 3 odds ──
        "odds_home",
        "odds_draw",
        "odds_away",
        # ── 4 MTV ──
        "home_talent_delta",
        "away_talent_delta",
        "talent_delta_diff",
        "shock_magnitude",
    ]

    def __init__(self, model_version=None):
        super().__init__(model_version=model_version or "v2.1-tier3-family_s")


# ═════════════════════════════════════════════════════════════════════════════
# Loader — global state (mirrors app/ml/shadow.py pattern)
# ═════════════════════════════════════════════════════════════════════════════

_family_s_engine: Optional[FamilySEngine] = None
_family_s_loaded: bool = False

FAMILY_S_VERSION_PATTERN = "%family_s%"


async def init_family_s_engine(session: AsyncSession) -> bool:
    """Initialize Family S engine from DB snapshot (called at startup).

    P1: Called regardless of LEAGUE_ROUTER_MTV_ENABLED so that flipping
    the flag doesn't require a redeploy.

    Returns True if engine loaded successfully.
    """
    global _family_s_engine, _family_s_loaded

    result = await session.execute(
        select(ModelSnapshot)
        .where(ModelSnapshot.model_version.like(FAMILY_S_VERSION_PATTERN))
        .order_by(ModelSnapshot.created_at.desc())
        .limit(1)
    )
    snapshot = result.scalar_one_or_none()

    if not snapshot or not snapshot.model_blob:
        logger.info(
            "Family S model not found in DB. "
            "Tier 3 will use baseline fallback."
        )
        _family_s_loaded = False
        return False

    engine = FamilySEngine(model_version=snapshot.model_version)
    if engine.load_from_bytes(snapshot.model_blob):
        _family_s_engine = engine
        _family_s_loaded = True
        logger.info(
            "Family S engine loaded: version=%s, brier=%.4f, features=%d",
            snapshot.model_version,
            snapshot.brier_score or 0.0,
            len(FamilySEngine.FEATURE_COLUMNS),
        )
        return True

    logger.error("Failed to deserialize Family S model blob")
    _family_s_loaded = False
    return False


def is_family_s_loaded() -> bool:
    """Check if Family S model is loaded and ready."""
    return (
        _family_s_loaded
        and _family_s_engine is not None
        and _family_s_engine.is_loaded
    )


def get_family_s_engine() -> Optional[FamilySEngine]:
    """Get the Family S engine instance (or None if not loaded)."""
    return _family_s_engine if _family_s_loaded else None


def reload_family_s_engine(blob: bytes) -> bool:
    """Hot-reload Family S engine from new model blob without restart.

    Safe fallback: if load fails, previous engine is preserved.
    """
    global _family_s_engine, _family_s_loaded
    old_engine = _family_s_engine

    new_engine = FamilySEngine()
    if new_engine.load_from_bytes(blob):
        _family_s_engine = new_engine
        _family_s_loaded = True
        logger.info(
            "Family S engine hot-reloaded: %s",
            new_engine.model_version,
        )
        return True

    _family_s_engine = old_engine
    logger.error("Family S hot-reload failed, keeping previous engine")
    return False
