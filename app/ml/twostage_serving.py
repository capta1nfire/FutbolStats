"""Overlay serving layer (V1.1.0 cross-wire).

Loads the overlay model (v1.1.0-twostage, 3f one-stage XGBoostEngine) from
model_snapshots and provides prediction overlay for TS_LEAGUES.

V1.1.0 cross-wire: Overlay is now a one-stage XGBoostEngine (3 odds features,
multi:softprob) — the inverse of the pre-V1.1.0 architecture where overlay
was a TwoStageEngine.

Pattern: identical to app/ml/family_s.py (global engine, startup init).
Routing: TS predictions for TS_LEAGUES (15), baseline for OS_LEAGUES (8).
Fallback: if odds triplet invalid → keep baseline (no crash).
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.ml.engine import XGBoostEngine
from app.ml.league_router import TS_LEAGUES, is_ts_league
from app.models import ModelSnapshot

logger = logging.getLogger("futbolstats.twostage_serving")

# ═══════════════════════════════════════════════════════════════════════════
# Global engine (loaded at startup)
# ═══════════════════════════════════════════════════════════════════════════

_ts_engine: Optional[XGBoostEngine] = None
_ts_loaded: bool = False

TS_VERSION_PATTERN = "%twostage%"


async def init_ts_engine(session: AsyncSession) -> bool:
    """Load the Two-Stage W3 engine from DB at startup.

    Called regardless of any feature flag so flipping routing on/off
    doesn't require a redeploy.
    """
    global _ts_engine, _ts_loaded

    result = await session.execute(
        select(ModelSnapshot)
        .where(ModelSnapshot.model_version.like(TS_VERSION_PATTERN))
        .order_by(ModelSnapshot.created_at.desc())
        .limit(1)
    )
    snapshot = result.scalar_one_or_none()

    if not snapshot or not snapshot.model_blob:
        logger.info(
            "Two-Stage W3 model not found in DB. "
            "All leagues will use baseline."
        )
        _ts_loaded = False
        return False

    engine = XGBoostEngine(model_version=snapshot.model_version)
    if engine.load_from_bytes(snapshot.model_blob):
        # V1.1.0: overlay uses 3f one-stage XGBoostEngine (odds only)
        engine.FEATURE_COLUMNS = ["odds_home", "odds_draw", "odds_away"]
        _ts_engine = engine
        _ts_loaded = True
        logger.info(
            "Overlay engine loaded: version=%s, brier=%.4f, "
            "features=%s, snapshot_id=%d",
            snapshot.model_version,
            snapshot.brier_score or 0.0,
            engine.FEATURE_COLUMNS,
            snapshot.id,
        )
        return True

    logger.error("Failed to deserialize Two-Stage W3 model blob")
    _ts_loaded = False
    return False


def is_ts_loaded() -> bool:
    """Check if Two-Stage W3 engine is loaded and ready."""
    return _ts_loaded and _ts_engine is not None and _ts_engine.is_loaded


def get_ts_engine() -> Optional[XGBoostEngine]:
    """Get the Two-Stage W3 engine (or None if not loaded)."""
    return _ts_engine if _ts_loaded else None


# ═══════════════════════════════════════════════════════════════════════════
# Overlay function for /predictions/upcoming
# ═══════════════════════════════════════════════════════════════════════════

def overlay_ts_predictions(predictions, ml_engine=None):
    """Overlay Two-Stage W3 predictions for TS league matches.

    For each NS, non-frozen prediction in a TS league with valid odds,
    re-predicts using the Two-Stage engine and replaces probabilities.

    Args:
        predictions: List of prediction dicts from ml_engine.predict()
        ml_engine: The baseline XGBoostEngine (for _find_value_bets)

    Returns:
        (predictions, stats) — modified in-place + overlay stats dict
    """
    stats = {"ts_hits": 0, "ts_no_odds": 0, "ts_eligible": 0, "os_kept": 0}

    if not is_ts_loaded():
        return predictions, stats

    engine = get_ts_engine()
    if not engine:
        return predictions, stats

    for pred in predictions:
        league_id = pred.get("league_id")
        if not league_id or not is_ts_league(league_id):
            continue

        # Only NS, not frozen
        if pred.get("status") != "NS" or pred.get("is_frozen"):
            continue

        stats["ts_eligible"] += 1

        # Check valid odds triplet
        market = pred.get("market_odds") or {}
        odds_h = market.get("home")
        odds_d = market.get("draw")
        odds_a = market.get("away")

        if not odds_h or not odds_d or not odds_a or odds_h <= 0 or odds_d <= 0 or odds_a <= 0:
            stats["ts_no_odds"] += 1
            stats["os_kept"] += 1
            continue

        # Build DataFrame from full prediction dict so overlay engine
        # gets all available features, not just odds.
        # Engine handles missing features (fills with 0 + logs warning).
        row_data = {
            "odds_home": [float(odds_h)],
            "odds_draw": [float(odds_d)],
            "odds_away": [float(odds_a)],
        }
        for k, v in pred.items():
            if isinstance(v, (int, float)) and k not in row_data:
                row_data[k] = [v]
        df_row = pd.DataFrame(row_data)

        try:
            proba = engine.predict_proba(df_row)
            h, d, a = float(proba[0][0]), float(proba[0][1]), float(proba[0][2])

            # Replace probabilities
            pred["probabilities"] = {
                "home": round(h, 4),
                "draw": round(d, 4),
                "away": round(a, 4),
            }

            # Recalculate fair odds
            pred["fair_odds"] = {
                "home": round(1.0 / h, 2) if h > 0.001 else None,
                "draw": round(1.0 / d, 2) if d > 0.001 else None,
                "away": round(1.0 / a, 2) if a > 0.001 else None,
            }

            # Recompute value_bets
            if ml_engine:
                vb = ml_engine._find_value_bets(
                    np.array([h, d, a]),
                    [float(odds_h), float(odds_d), float(odds_a)],
                )
                pred["value_bets"] = vb if vb else []
                pred["has_value_bet"] = bool(vb)
                pred["best_value_bet"] = max(vb, key=lambda x: x["edge"]) if vb else None

            # Mark as Two-Stage (skip market anchor downstream)
            pred["skip_market_anchor"] = True
            pred["model_version_served"] = engine.model_version
            stats["ts_hits"] += 1

        except Exception as e:
            logger.warning(
                "TS overlay failed for match %s: %s. Keeping baseline.",
                pred.get("match_id"), e,
            )
            stats["os_kept"] += 1

    return predictions, stats


def predict_single_ts(match_odds, pred_dict=None):
    """Predict for a single match using TS engine. Returns (h, d, a) or None.

    Args:
        match_odds: dict with keys 'home', 'draw', 'away' (float odds)
        pred_dict: optional full prediction dict — scalar fields are passed
                   to the engine so it has access to any available features.
    """
    if not is_ts_loaded():
        return None

    engine = get_ts_engine()
    if not engine:
        return None

    odds_h = match_odds.get("home")
    odds_d = match_odds.get("draw")
    odds_a = match_odds.get("away")

    if not odds_h or not odds_d or not odds_a:
        return None
    if odds_h <= 0 or odds_d <= 0 or odds_a <= 0:
        return None

    row_data = {
        "odds_home": [float(odds_h)],
        "odds_draw": [float(odds_d)],
        "odds_away": [float(odds_a)],
    }
    if pred_dict:
        for k, v in pred_dict.items():
            if isinstance(v, (int, float)) and k not in row_data:
                row_data[k] = [v]
    df = pd.DataFrame(row_data)

    try:
        proba = engine.predict_proba(df)
        return float(proba[0][0]), float(proba[0][1]), float(proba[0][2])
    except Exception as e:
        logger.warning("TS single predict failed: %s", e)
        return None
