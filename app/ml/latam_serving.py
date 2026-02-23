"""LATAM baseline serving layer (V1.3.0).

Loads the v1.3.0-latam-first model (18f TwoStageEngine: 16f baseline + 2 geo)
from model_snapshots and provides prediction overlay for LATAM leagues.

Alpha Lab Sprint 1 result: T0 (Global LATAM 18f) beat Tprod (23-league 16f)
with Weighted Skill -1.8% vs -2.2%. Chile +4.0%, Argentina +1.0%.

Pattern: identical to app/ml/twostage_serving.py (global engine, startup init).
Routing: LATAM league_ids → latam engine, others → keep baseline.
Fallback: if geo data missing → NaN (XGBoost handles natively via fillna(0)).
"""

import logging
import math
from typing import Optional

import numpy as np
import pandas as pd
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.ml.engine import TwoStageEngine
from app.models import ModelSnapshot

logger = logging.getLogger("futbolstats.latam_serving")

# ═══════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════

LATAM_LEAGUE_IDS = {128, 71, 239, 242, 262, 265, 268, 281, 299, 344, 250}

LATAM_VERSION_PATTERN = "v1.3.0-latam%"

# ═══════════════════════════════════════════════════════════════════════════
# Global engine + geo cache (loaded at startup)
# ═══════════════════════════════════════════════════════════════════════════

_latam_engine: Optional[TwoStageEngine] = None
_latam_loaded: bool = False
_geo_cache: dict = {}  # team_id -> {"lat": float, "lon": float, "altitude_m": int}


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Haversine distance in kilometers between two lat/lon points."""
    R = 6371.0  # Earth radius in km
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2))
         * math.sin(dlon / 2) ** 2)
    return R * 2 * math.asin(math.sqrt(min(a, 1.0)))


async def init_latam_engine(session: AsyncSession) -> bool:
    """Load the LATAM engine + geo cache at startup.

    Called regardless of any feature flag so toggling routing on/off
    doesn't require a redeploy.
    """
    global _latam_engine, _latam_loaded, _geo_cache

    # ── Load model snapshot ──
    result = await session.execute(
        select(ModelSnapshot)
        .where(ModelSnapshot.model_version.like(LATAM_VERSION_PATTERN))
        .order_by(ModelSnapshot.created_at.desc())
        .limit(1)
    )
    snapshot = result.scalar_one_or_none()

    if not snapshot or not snapshot.model_blob:
        logger.info(
            "LATAM model not found in DB. "
            "All leagues will use global baseline."
        )
        _latam_loaded = False
        return False

    engine = TwoStageEngine(model_version=snapshot.model_version)
    if not engine.load_from_bytes(snapshot.model_blob):
        logger.error("Failed to deserialize LATAM model blob")
        _latam_loaded = False
        return False

    _latam_engine = engine
    _latam_loaded = True
    logger.info(
        "LATAM engine loaded: version=%s, brier=%.4f, "
        "features_s1=%d, features_s2=%d, snapshot_id=%d",
        snapshot.model_version,
        snapshot.brier_score or 0.0,
        len(engine.active_stage1_features),
        len(engine.active_stage2_features),
        snapshot.id,
    )

    # ── Load geo cache from team_wikidata_enrichment ──
    try:
        geo_result = await session.execute(text("""
            SELECT team_id, lat, lon, stadium_altitude_m
            FROM team_wikidata_enrichment
            WHERE lat IS NOT NULL AND lon IS NOT NULL
        """))
        rows = geo_result.fetchall()
        _geo_cache = {
            row[0]: {
                "lat": float(row[1]),
                "lon": float(row[2]),
                "altitude_m": int(row[3]) if row[3] is not None else None,
            }
            for row in rows
        }
        n_with_alt = sum(1 for v in _geo_cache.values() if v["altitude_m"] is not None)
        logger.info(
            "Geo cache loaded: %d teams with lat/lon, %d with altitude",
            len(_geo_cache), n_with_alt,
        )
    except Exception as e:
        logger.warning("Failed to load geo cache: %s. Geo features will be NaN.", e)
        _geo_cache = {}

    return True


def is_latam_loaded() -> bool:
    """Check if LATAM engine is loaded and ready."""
    return _latam_loaded and _latam_engine is not None


def get_latam_engine() -> Optional[TwoStageEngine]:
    """Get the LATAM engine (or None if not loaded)."""
    return _latam_engine if _latam_loaded else None


def is_latam_league(league_id: int) -> bool:
    """Check if league_id belongs to LATAM routing."""
    return league_id in LATAM_LEAGUE_IDS


def compute_geo_features(home_team_id: int, away_team_id: int) -> dict:
    """Compute altitude_diff_m and travel_distance_km from geo cache.

    Returns dict with keys altitude_diff_m, travel_distance_km.
    Values are float or NaN if data is missing.
    """
    home_geo = _geo_cache.get(home_team_id)
    away_geo = _geo_cache.get(away_team_id)

    altitude_diff = float("nan")
    travel_dist = float("nan")

    if home_geo and away_geo:
        # Travel distance (always computable if lat/lon exist)
        travel_dist = _haversine_km(
            home_geo["lat"], home_geo["lon"],
            away_geo["lat"], away_geo["lon"],
        )

        # Altitude difference (only if both have altitude data)
        if home_geo["altitude_m"] is not None and away_geo["altitude_m"] is not None:
            altitude_diff = float(home_geo["altitude_m"] - away_geo["altitude_m"])

    return {
        "altitude_diff_m": altitude_diff,
        "travel_distance_km": travel_dist,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Single-match prediction (for /predictions/match/{id}, /matches/{id}/details)
# ═══════════════════════════════════════════════════════════════════════════

def predict_single_latam(feature_df, home_team_id, away_team_id):
    """Predict for a single match using LATAM engine. Returns (h, d, a) or None.

    Args:
        feature_df: Single-row DataFrame with baseline features
        home_team_id: int — for geo feature lookup
        away_team_id: int — for geo feature lookup

    Returns:
        (h, d, a) tuple or None if engine not loaded
    """
    if not is_latam_loaded():
        return None

    engine = get_latam_engine()
    if engine is None:
        return None

    try:
        df = feature_df.copy()

        # Add geo features from cache
        geo = compute_geo_features(int(home_team_id), int(away_team_id))
        df["altitude_diff_m"] = geo["altitude_diff_m"]
        df["travel_distance_km"] = geo["travel_distance_km"]

        proba = engine.predict_proba(df)
        return float(proba[0][0]), float(proba[0][1]), float(proba[0][2])
    except Exception as e:
        logger.warning("LATAM single predict failed: %s", e)
        return None


# ═══════════════════════════════════════════════════════════════════════════
# Overlay function for daily_save_predictions / /predictions/upcoming
# ═══════════════════════════════════════════════════════════════════════════

def overlay_latam_predictions(predictions, feature_df, ml_engine=None):
    """Overlay LATAM engine predictions for LATAM league matches.

    For each NS prediction in a LATAM league, re-predicts using the LATAM
    engine (18f with geo features) and replaces baseline probabilities.

    Args:
        predictions: List of prediction dicts from engine.predict()
        feature_df: Original feature DataFrame with all columns
        ml_engine: The baseline engine (for _find_value_bets recalculation)

    Returns:
        (predictions, stats) — modified in-place + overlay stats dict
    """
    stats = {
        "latam_hits": 0,
        "latam_no_match": 0,
        "latam_eligible": 0,
        "latam_errors": 0,
        "global_kept": 0,
    }

    if not is_latam_loaded():
        return predictions, stats

    engine = get_latam_engine()
    if engine is None:
        return predictions, stats

    # Build match_id index for fast lookup
    if feature_df is not None and "match_id" in feature_df.columns:
        df_indexed = feature_df.set_index("match_id", drop=False)
    else:
        df_indexed = None

    for pred in predictions:
        league_id = pred.get("league_id")
        if not league_id or not is_latam_league(league_id):
            continue

        # Only NS, not frozen
        if pred.get("status") != "NS" or pred.get("is_frozen"):
            continue

        stats["latam_eligible"] += 1

        match_id = pred.get("match_id")

        # Get original feature row
        if df_indexed is None or match_id not in df_indexed.index:
            stats["latam_no_match"] += 1
            stats["global_kept"] += 1
            continue

        try:
            row = df_indexed.loc[match_id]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]  # Deduplicate if multiple rows

            # Build single-row DataFrame with all features
            row_data = {}
            for col in feature_df.columns:
                val = row.get(col) if hasattr(row, "get") else row[col] if col in row.index else None
                if val is not None and not (isinstance(val, float) and math.isnan(val)):
                    row_data[col] = [val]
                else:
                    row_data[col] = [float("nan")]

            # Add geo features from cache
            home_team_id = pred.get("home_team_id") or row.get("home_team_id")
            away_team_id = pred.get("away_team_id") or row.get("away_team_id")

            if home_team_id and away_team_id:
                geo = compute_geo_features(int(home_team_id), int(away_team_id))
                row_data["altitude_diff_m"] = [geo["altitude_diff_m"]]
                row_data["travel_distance_km"] = [geo["travel_distance_km"]]
            else:
                row_data["altitude_diff_m"] = [float("nan")]
                row_data["travel_distance_km"] = [float("nan")]

            df_row = pd.DataFrame(row_data)

            # Predict with LATAM engine
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
                market = pred.get("market_odds") or {}
                odds_h = market.get("home")
                odds_d = market.get("draw")
                odds_a = market.get("away")
                if odds_h and odds_d and odds_a:
                    vb = ml_engine._find_value_bets(
                        np.array([h, d, a]),
                        [float(odds_h), float(odds_d), float(odds_a)],
                    )
                    pred["value_bets"] = vb if vb else []
                    pred["has_value_bet"] = bool(vb)
                    pred["best_value_bet"] = max(vb, key=lambda x: x["edge"]) if vb else None

            pred["model_version_served"] = engine.model_version
            pred["latam_overlay"] = True
            stats["latam_hits"] += 1

        except Exception as e:
            logger.warning(
                "LATAM overlay failed for match %s (league %s): %s. Keeping baseline.",
                match_id, league_id, e,
            )
            stats["latam_errors"] += 1
            stats["global_kept"] += 1

    return predictions, stats
