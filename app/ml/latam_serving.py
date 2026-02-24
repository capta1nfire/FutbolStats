"""LATAM serving layer (V1.4.1 GEO-ROUTER).

Two specialist models routed by league tier:
  - v1.4.1-latam-geo  (18f): Bolivia, Paraguay, Peru, Venezuela, Chile, Uruguay
  - v1.4.1-latam-flat (16f): Argentina, Brasil, Colombia, Ecuador

Mexico (262) stays in LATAM_LEAGUE_IDS but tier=None → baseline global,
no overlay, no VORP. Deterministic behavior (ABE P0).

Backward compat: if only v1.3.0 exists, loads as single universal engine.

Pattern: identical to app/ml/twostage_serving.py (global engine, startup init).
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

# Tier routing (v1.4.1, Uruguay moved to GEO per Geo Signal Test re-run 2026-02-24)
TIER_GEO_LEAGUES = {344, 250, 281, 299, 265, 268}  # Bolivia, Paraguay, Peru, Venezuela, Chile, Uruguay
TIER_FLAT_LEAGUES = {128, 71, 239, 242}             # Argentina, Brasil, Colombia, Ecuador
# Mexico (262): NOT in either tier → tier=None → baseline global

# Version patterns for model_snapshots lookup
GEO_VERSION_PATTERN = "v1.4.%-latam-geo%"
FLAT_VERSION_PATTERN = "v1.4.%-latam-flat%"
LEGACY_VERSION_PATTERN = "v1.3.0-latam%"  # backward compat

# ═══════════════════════════════════════════════════════════════════════════
# Global engines + geo cache (loaded at startup)
# ═══════════════════════════════════════════════════════════════════════════

_latam_geo_engine: Optional[TwoStageEngine] = None
_latam_flat_engine: Optional[TwoStageEngine] = None
_latam_legacy_engine: Optional[TwoStageEngine] = None  # v1.3.0 fallback
_geo_loaded: bool = False
_flat_loaded: bool = False
_legacy_loaded: bool = False
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


async def _load_engine(session: AsyncSession, pattern: str, label: str
                       ) -> Optional[tuple[TwoStageEngine, int]]:
    """Load a single engine by version pattern. Returns (engine, snapshot_id) or None."""
    result = await session.execute(
        select(ModelSnapshot)
        .where(ModelSnapshot.model_version.like(pattern))
        .order_by(ModelSnapshot.created_at.desc())
        .limit(1)
    )
    snapshot = result.scalar_one_or_none()

    if not snapshot or not snapshot.model_blob:
        logger.info("LATAM %s model not found (pattern=%s)", label, pattern)
        return None

    engine = TwoStageEngine(model_version=snapshot.model_version)
    if not engine.load_from_bytes(snapshot.model_blob):
        logger.error("Failed to deserialize LATAM %s model blob", label)
        return None

    logger.info(
        "LATAM %s engine loaded: version=%s, brier=%.4f, "
        "features_s1=%d, features_s2=%d, snapshot_id=%d",
        label,
        snapshot.model_version,
        snapshot.brier_score or 0.0,
        len(engine.active_stage1_features),
        len(engine.active_stage2_features),
        snapshot.id,
    )
    return engine, snapshot.id


async def init_latam_engines(session: AsyncSession) -> bool:
    """Load LATAM engines + geo cache at startup.

    Tries v1.4.0 geo/flat first. Falls back to v1.3.0 universal if needed.
    """
    global _latam_geo_engine, _latam_flat_engine, _latam_legacy_engine
    global _geo_loaded, _flat_loaded, _legacy_loaded, _geo_cache

    # ── Try v1.4.0 GEO-ROUTER engines ──
    geo_result = await _load_engine(session, GEO_VERSION_PATTERN, "GEO")
    if geo_result:
        _latam_geo_engine, _ = geo_result
        _geo_loaded = True

    flat_result = await _load_engine(session, FLAT_VERSION_PATTERN, "FLAT")
    if flat_result:
        _latam_flat_engine, _ = flat_result
        _flat_loaded = True

    # ── Fallback: v1.3.0 universal (if v1.4.0 not available) ──
    if not _geo_loaded or not _flat_loaded:
        legacy_result = await _load_engine(session, LEGACY_VERSION_PATTERN, "LEGACY")
        if legacy_result:
            _latam_legacy_engine, _ = legacy_result
            _legacy_loaded = True
            logger.info(
                "LATAM LEGACY engine loaded as fallback (geo=%s, flat=%s)",
                _geo_loaded, _flat_loaded,
            )

    any_loaded = _geo_loaded or _flat_loaded or _legacy_loaded

    if not any_loaded:
        logger.info("No LATAM models found. All leagues use global baseline.")

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

    return any_loaded


# Backward compat alias
async def init_latam_engine(session: AsyncSession) -> bool:
    """Backward compat: delegates to init_latam_engines."""
    return await init_latam_engines(session)


# ═══════════════════════════════════════════════════════════════════════════
# Tier routing
# ═══════════════════════════════════════════════════════════════════════════

def get_latam_tier(league_id: int) -> Optional[str]:
    """Get the GEO-ROUTER tier for a league.

    Returns "geo", "flat", or None (for Mexico/unrouted).
    """
    if league_id in TIER_GEO_LEAGUES:
        return "geo"
    if league_id in TIER_FLAT_LEAGUES:
        return "flat"
    return None


def is_latam_league(league_id: int) -> bool:
    """Check if league_id belongs to LATAM routing."""
    return league_id in LATAM_LEAGUE_IDS


def is_latam_loaded() -> bool:
    """Check if any LATAM engine is loaded and ready."""
    return _geo_loaded or _flat_loaded or _legacy_loaded


def is_latam_geo_loaded() -> bool:
    """Check if GEO tier engine is loaded."""
    return _geo_loaded and _latam_geo_engine is not None


def is_latam_flat_loaded() -> bool:
    """Check if FLAT tier engine is loaded."""
    return _flat_loaded and _latam_flat_engine is not None


def get_latam_geo_engine() -> Optional[TwoStageEngine]:
    """Get the GEO tier engine (or None)."""
    return _latam_geo_engine if _geo_loaded else None


def get_latam_flat_engine() -> Optional[TwoStageEngine]:
    """Get the FLAT tier engine (or None)."""
    return _latam_flat_engine if _flat_loaded else None


def get_latam_engine() -> Optional[TwoStageEngine]:
    """Backward compat: get any LATAM engine (prefers v1.4.0 geo, then legacy)."""
    if _geo_loaded:
        return _latam_geo_engine
    if _legacy_loaded:
        return _latam_legacy_engine
    return None


def _get_engine_for_tier(tier: Optional[str]) -> Optional[TwoStageEngine]:
    """Get the correct engine for a tier, with legacy fallback."""
    if tier == "geo":
        if _geo_loaded:
            return _latam_geo_engine
        if _legacy_loaded:
            return _latam_legacy_engine  # legacy is 18f, works for geo
        return None
    elif tier == "flat":
        if _flat_loaded:
            return _latam_flat_engine
        if _legacy_loaded:
            return _latam_legacy_engine  # legacy 18f works but suboptimal
        return None
    return None  # tier=None (Mexico) → no engine → baseline


# ═══════════════════════════════════════════════════════════════════════════
# Geo features
# ═══════════════════════════════════════════════════════════════════════════

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

def predict_single_latam(feature_df, home_team_id, away_team_id, league_id=None):
    """Predict for a single match using the correct LATAM engine.

    Args:
        feature_df: Single-row DataFrame with baseline features
        home_team_id: int — for geo feature lookup
        away_team_id: int — for geo feature lookup
        league_id: int — for tier routing (optional, enables GEO-ROUTER)

    Returns:
        (h, d, a) tuple or None if no engine for this tier
    """
    tier = get_latam_tier(league_id) if league_id else None
    engine = _get_engine_for_tier(tier)

    # tier=None (Mexico) or no engine loaded → return None → baseline handles it
    if engine is None:
        return None

    try:
        df = feature_df.copy()

        # Add geo features only for geo tier
        if tier == "geo":
            geo = compute_geo_features(int(home_team_id), int(away_team_id))
            df["altitude_diff_m"] = geo["altitude_diff_m"]
            df["travel_distance_km"] = geo["travel_distance_km"]

        proba = engine.predict_proba(df)
        return float(proba[0][0]), float(proba[0][1]), float(proba[0][2])
    except Exception as e:
        logger.warning("LATAM single predict failed (tier=%s): %s", tier, e)
        return None


# ═══════════════════════════════════════════════════════════════════════════
# Overlay function for daily_save_predictions / /predictions/upcoming
# ═══════════════════════════════════════════════════════════════════════════

def overlay_latam_predictions(predictions, feature_df, ml_engine=None):
    """Overlay LATAM engine predictions for LATAM league matches.

    Routes each match to the correct tier engine:
      - geo tier: 18f engine + geo features
      - flat tier: 16f engine, no geo features
      - tier=None (Mexico): skip overlay, keep baseline

    Args:
        predictions: List of prediction dicts from engine.predict()
        feature_df: Original feature DataFrame with all columns
        ml_engine: The baseline engine (for _find_value_bets recalculation)

    Returns:
        (predictions, stats) — modified in-place + overlay stats dict
    """
    stats = {
        "latam_hits": 0,
        "latam_geo_hits": 0,
        "latam_flat_hits": 0,
        "latam_no_match": 0,
        "latam_eligible": 0,
        "latam_errors": 0,
        "latam_tier_none": 0,
        "global_kept": 0,
    }

    if not is_latam_loaded():
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

        # Tier routing
        tier = get_latam_tier(league_id)
        if tier is None:
            # Mexico (262) or unrouted → keep baseline
            stats["latam_tier_none"] += 1
            stats["global_kept"] += 1
            continue

        engine = _get_engine_for_tier(tier)
        if engine is None:
            stats["global_kept"] += 1
            continue

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

            # Add geo features only for geo tier
            if tier == "geo":
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

            # Predict with tier engine
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
            pred["latam_tier"] = tier
            stats["latam_hits"] += 1
            if tier == "geo":
                stats["latam_geo_hits"] += 1
            else:
                stats["latam_flat_hits"] += 1

        except Exception as e:
            logger.warning(
                "LATAM overlay failed for match %s (league %s, tier=%s): %s. Keeping baseline.",
                match_id, league_id, tier, e,
            )
            stats["latam_errors"] += 1
            stats["global_kept"] += 1

    return predictions, stats
