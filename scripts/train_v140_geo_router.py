#!/usr/bin/env python3
"""V1.4.1 — GEO-ROUTER: Train 2 LATAM specialist models.

GDT FULL CLEARANCE (2026-02-23). ML FREEZE lifted.

Two models bifurcated by geo signal strength:
  - v1.4.1-latam-geo  (18f): 16 baseline + altitude_diff_m + travel_distance_km
    Tier Geo: Bolivia(344), Paraguay(250), Peru(281), Venezuela(299), Chile(265), Uruguay(268)

  - v1.4.1-latam-flat (16f): 14 baseline + elo_k10_diff + elo_momentum_diff
    Tier Flat: Argentina(128), Brasil(71), Colombia(239), Ecuador(242)

Architecture: TwoStage, home_away semantic, dw=1.0 (identical to production v1.2.0).
Hyperparams: Exact production PARAMS_TS_S1 + PARAMS_TS_S2.
Persistence: Both saved as is_active=False snapshots (non-destructive).

Usage:
    source .env
    python scripts/train_v140_geo_router.py [--dry-run]
"""

import asyncio
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import TimeSeriesSplit

sys.path.insert(0, str(Path(__file__).parent.parent))

if not os.environ.get("DATABASE_URL"):
    raise RuntimeError(
        "DATABASE_URL required.\n"
        "Run: source .env && python scripts/train_v140_geo_router.py"
    )

# ─── Constants ────────────────────────────────────────────────────────────────
MIN_DATE = "2023-01-01"
N_CV_SPLITS = 3
LAB_DATA_DIR = Path(__file__).parent / "output" / "lab"

# ─── Tier Definitions ─────────────────────────────────────────────────────────

TIER_GEO_LEAGUES = {
    344: "Bolivia",
    250: "Paraguay",
    281: "Peru",
    299: "Venezuela",
    265: "Chile",
    268: "Uruguay",
}

TIER_FLAT_LEAGUES = {
    128: "Argentina",
    71: "Brasil",
    239: "Colombia",
    242: "Ecuador",
}

# ─── Feature Sets ─────────────────────────────────────────────────────────────

FEATURES_14 = [
    "home_goals_scored_avg", "home_goals_conceded_avg",
    "home_shots_avg", "home_corners_avg",
    "home_rest_days", "home_matches_played",
    "away_goals_scored_avg", "away_goals_conceded_avg",
    "away_shots_avg", "away_corners_avg",
    "away_rest_days", "away_matches_played",
    "goal_diff_avg", "rest_diff",
]

MOMENTUM_2 = ["elo_k10_diff", "elo_momentum_diff"]
GEO_2 = ["altitude_diff_m", "travel_distance_km"]

FEATURES_16 = FEATURES_14 + MOMENTUM_2       # Tier Flat
FEATURES_18 = FEATURES_16 + GEO_2            # Tier Geo

# ─── Hyperparameters (exact production TwoStage v1.2.0) ──────────────────────

PARAMS_TS_S1 = {
    "objective": "binary:logistic",
    "max_depth": 3,
    "learning_rate": 0.05,
    "n_estimators": 100,
    "min_child_weight": 7,
    "subsample": 0.72,
    "colsample_bytree": 0.71,
    "random_state": 42,
}
PARAMS_TS_S2 = {
    "objective": "binary:logistic",
    "max_depth": 3,
    "learning_rate": 0.05,
    "n_estimators": 100,
    "min_child_weight": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
}


# ─── Utilities ────────────────────────────────────────────────────────────────

def calculate_brier(y_true, y_proba):
    """Multiclass Brier score (average per-class)."""
    scores = []
    for cls in range(y_proba.shape[1]):
        y_bin = (y_true == cls).astype(int)
        scores.append(brier_score_loss(y_bin, y_proba[:, cls]))
    return float(np.mean(scores))


def load_lab_csvs(leagues: dict) -> pd.DataFrame:
    """Load Lab CSVs for specified leagues."""
    frames = []
    for lid in sorted(leagues):
        csv_path = LAB_DATA_DIR / f"lab_data_{lid}.csv"
        if not csv_path.exists():
            print(f"  WARNING: Missing CSV for league {lid} ({leagues[lid]})")
            continue

        df = pd.read_csv(csv_path)
        df = df[df["date"] >= MIN_DATE].copy()
        df = df[df["result"].notna()].copy()
        frames.append(df)
        print(f"  League {lid} ({leagues[lid]}): {len(df)} samples")

    if not frames:
        raise RuntimeError("No data loaded!")

    pooled = pd.concat(frames, ignore_index=True)
    pooled = pooled.sort_values(["date", "match_id"]).reset_index(drop=True)
    return pooled


def train_twostage_cv(X, y, n_splits=N_CV_SPLITS):
    """CV evaluation with TimeSeriesSplit. Returns (fold_scores, cv_avg)."""
    y_draw = (y == 1).astype(int)
    sample_weight = np.ones(len(y), dtype=np.float32)
    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_scores = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        s1 = xgb.XGBClassifier(**PARAMS_TS_S1)
        s1.fit(X[train_idx], y_draw[train_idx],
               sample_weight=sample_weight[train_idx], verbose=False)

        nd_train = y[train_idx] != 1
        idx_nd = train_idx[nd_train]
        y_home_train = (y[idx_nd] == 0).astype(int)
        s2 = xgb.XGBClassifier(**PARAMS_TS_S2)
        s2.fit(X[idx_nd], y_home_train, verbose=False)

        p_draw = s1.predict_proba(X[val_idx])[:, 1]
        p_home_nd = s2.predict_proba(X[val_idx])[:, 1]
        p_home = (1 - p_draw) * p_home_nd
        p_away = (1 - p_draw) * (1 - p_home_nd)
        y_proba = np.column_stack([p_home, p_draw, p_away])

        brier = calculate_brier(y[val_idx], y_proba)
        fold_scores.append(brier)
        print(f"    Fold {fold + 1}: Brier = {brier:.6f}")

    cv_avg = float(np.mean(fold_scores))
    print(f"    CV Average: {cv_avg:.6f}")
    return fold_scores, cv_avg


def train_twostage_final(X, y):
    """Train final TwoStage on 100% data. Returns (s1, s2)."""
    y_draw = (y == 1).astype(int)
    sample_weight = np.ones(len(y), dtype=np.float32)

    s1 = xgb.XGBClassifier(**PARAMS_TS_S1)
    s1.fit(X, y_draw, sample_weight=sample_weight, verbose=False)

    nondraw_mask = y != 1
    y_stage2 = (y[nondraw_mask] == 0).astype(int)
    s2 = xgb.XGBClassifier(**PARAMS_TS_S2)
    s2.fit(X[nondraw_mask], y_stage2, verbose=False)

    return s1, s2


async def persist_snapshot(engine, brier_score, cv_scores, samples_trained,
                           training_config, dry_run=False):
    """Persist model as inactive snapshot to DB."""
    from app.database import get_async_session
    from app.models import ModelSnapshot

    model_blob = engine.save_to_bytes()
    print(f"  Model blob: {len(model_blob):,} bytes")

    if dry_run:
        print(f"  [DRY RUN] Would persist {engine.model_version}, is_active=False")
        return None

    async for session in get_async_session():
        snapshot = ModelSnapshot(
            model_version=engine.model_version,
            model_blob=model_blob,
            model_path="db_stored",
            brier_score=brier_score,
            cv_brier_scores=cv_scores,
            samples_trained=samples_trained,
            training_config=training_config,
            is_active=False,
            is_baseline=False,
        )
        session.add(snapshot)
        await session.commit()
        await session.refresh(snapshot)

        print(f"  Snapshot persisted: id={snapshot.id}, "
              f"version={engine.model_version}, is_active=False")
        return snapshot.id


# ═══════════════════════════════════════════════════════════════════════════════
# Model Training Functions
# ═══════════════════════════════════════════════════════════════════════════════

async def train_model(model_version, features, leagues, dry_run=False):
    """Train a single TwoStage model with given features and league pool."""
    from app.ml.engine import TwoStageEngine

    n_features = len(features)
    print(f"\n{'=' * 70}")
    print(f"  {model_version} ({n_features}f TwoStage, home_away, dw=1.0)")
    print(f"  Features: {features}")
    print(f"  Leagues: {sorted(leagues.keys())} ({len(leagues)} leagues)")
    print(f"{'=' * 70}")

    # Load data
    print(f"\nLoading Lab CSVs ({len(leagues)} leagues)...")
    df = load_lab_csvs(leagues)
    print(f"  Total samples: {len(df)}")

    # Verify features exist
    missing = [f for f in features if f not in df.columns]
    if missing:
        raise RuntimeError(
            f"Lab CSVs missing features: {missing}. "
            f"Re-run feature extraction."
        )

    # NaN report
    for f in features:
        n_nan = df[f].isna().sum()
        if n_nan > 0:
            pct = 100 * n_nan / len(df)
            print(f"  {f}: {n_nan} NaN ({pct:.1f}%)")

    df = df.sort_values(["date", "match_id"]).reset_index(drop=True)
    y = df["result"].values
    X = df[features].fillna(0).values

    # CV evaluation
    print(f"\nCV evaluation (TimeSeriesSplit, n={N_CV_SPLITS})...")
    fold_scores, cv_avg = train_twostage_cv(X, y)

    # Train final on 100% data
    print(f"\nTraining final model on 100% data...")
    s1, s2 = train_twostage_final(X, y)

    # Package into TwoStageEngine
    engine = TwoStageEngine(
        model_version=model_version,
        draw_weight=1.0,
        stage1_features=features,
        stage2_features=features,
    )
    engine.stage1 = s1
    engine.stage2 = s2
    engine.stage2_semantic = "home_away"

    print(f"  Stage 1: {s1.n_features_in_} features")
    print(f"  Stage 2: {s2.n_features_in_} features")
    print(f"  Semantic: {engine.stage2_semantic}")

    # Verify round-trip
    print(f"\nVerify round-trip (load from bytes)...")
    verify = TwoStageEngine()
    blob = engine.save_to_bytes()
    assert verify.load_from_bytes(blob), "Round-trip load failed!"
    assert verify._custom_stage1_features == features, \
        f"Stage1 features mismatch: {verify._custom_stage1_features}"
    assert verify._custom_stage2_features == features, \
        f"Stage2 features mismatch: {verify._custom_stage2_features}"
    assert verify.stage2_semantic == "home_away", \
        f"Semantic mismatch: {verify.stage2_semantic}"

    # Quick predict sanity
    test_row = pd.DataFrame({f: [0.0] for f in features})
    test_proba = verify.predict_proba(test_row)
    assert test_proba.shape == (1, 3), f"Bad proba shape: {test_proba.shape}"
    assert abs(test_proba.sum() - 1.0) < 0.001, f"Probs don't sum to 1: {test_proba.sum()}"
    print(f"  Round-trip OK. Test proba: {test_proba[0]}")

    # Persist
    training_config = {
        "architecture": "two_stage",
        "stage2_semantic": "home_away",
        "draw_weight": 1.0,
        "features_s1": features,
        "features_s2": features,
        "n_features": n_features,
        "min_date": MIN_DATE,
        "n_cv_splits": N_CV_SPLITS,
        "leagues": sorted(leagues.keys()),
        "league_names": {str(k): v for k, v in leagues.items()},
        "hyperparams_s1": {k: v for k, v in PARAMS_TS_S1.items() if k != "random_state"},
        "hyperparams_s2": {k: v for k, v in PARAMS_TS_S2.items() if k != "random_state"},
        "geo_router_tier": "geo" if n_features == 18 else "flat",
    }

    snapshot_id = await persist_snapshot(
        engine, cv_avg, fold_scores, len(df), training_config, dry_run
    )

    return cv_avg, snapshot_id, len(df)


# ═══════════════════════════════════════════════════════════════════════════════
# Per-League Skill Score Matrix
# ═══════════════════════════════════════════════════════════════════════════════

def compute_skill_scores(leagues, features):
    """Compute per-league Brier + Skill Score vs implied market probs."""
    print(f"\n{'─' * 70}")
    print(f"  SKILL SCORE MATRIX ({len(leagues)} leagues, {len(features)}f)")
    print(f"{'─' * 70}")

    results = []
    for lid in sorted(leagues):
        csv_path = LAB_DATA_DIR / f"lab_data_{lid}.csv"
        if not csv_path.exists():
            continue

        df = pd.read_csv(csv_path)
        df = df[df["date"] >= MIN_DATE].copy()
        df = df[df["result"].notna()].copy()

        # Need odds for market comparison
        has_odds = (
            df["odds_home"].notna() & (df["odds_home"] > 0) &
            df["odds_draw"].notna() & (df["odds_draw"] > 0) &
            df["odds_away"].notna() & (df["odds_away"] > 0)
        )
        df_odds = df[has_odds].copy()

        if len(df_odds) < 100:
            print(f"  League {lid} ({leagues[lid]}): <100 with odds, skipping")
            continue

        df_odds = df_odds.sort_values("date").reset_index(drop=True)
        y = df_odds["result"].values

        # Market implied probs (naive 1/odds normalized)
        imp_h = 1.0 / df_odds["odds_home"].values
        imp_d = 1.0 / df_odds["odds_draw"].values
        imp_a = 1.0 / df_odds["odds_away"].values
        imp_sum = imp_h + imp_d + imp_a
        market_probs = np.column_stack([imp_h / imp_sum, imp_d / imp_sum, imp_a / imp_sum])
        market_brier = calculate_brier(y, market_probs)

        # Model: 80/20 chronological split
        X = df_odds[features].fillna(0).values
        split_idx = int(len(X) * 0.8)
        X_tr, X_te = X[:split_idx], X[split_idx:]
        y_tr, y_te = y[:split_idx], y[split_idx:]

        if len(y_te) < 30:
            continue

        # Train TwoStage
        y_draw_tr = (y_tr == 1).astype(int)
        sw = np.ones(len(y_tr), dtype=np.float32)

        s1 = xgb.XGBClassifier(**PARAMS_TS_S1)
        s1.fit(X_tr, y_draw_tr, sample_weight=sw, verbose=False)

        nd_mask = y_tr != 1
        s2 = xgb.XGBClassifier(**PARAMS_TS_S2)
        s2.fit(X_tr[nd_mask], (y_tr[nd_mask] == 0).astype(int), verbose=False)

        p_draw = s1.predict_proba(X_te)[:, 1]
        p_home_nd = s2.predict_proba(X_te)[:, 1]
        p_home = (1 - p_draw) * p_home_nd
        p_away = (1 - p_draw) * (1 - p_home_nd)
        model_probs = np.column_stack([p_home, p_draw, p_away])

        model_brier = calculate_brier(y_te, model_probs)

        # Market Brier on test set only
        market_probs_te = market_probs[split_idx:]
        market_brier_te = calculate_brier(y_te, market_probs_te)

        skill = 1.0 - (model_brier / market_brier_te) if market_brier_te > 0 else 0.0
        delta = model_brier - market_brier_te

        beat = "YES" if delta < 0 else "no"
        results.append({
            "league_id": lid,
            "league_name": leagues[lid],
            "n_test": len(y_te),
            "model_brier": model_brier,
            "market_brier": market_brier_te,
            "delta": delta,
            "skill_pct": skill * 100,
            "beats_market": beat,
        })

        print(f"  {lid:>3d} {leagues[lid]:>12s}: "
              f"Model={model_brier:.4f} Market={market_brier_te:.4f} "
              f"Δ={delta:+.4f} Skill={skill*100:+.1f}% [{beat}] (n={len(y_te)})")

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

async def main():
    dry_run = "--dry-run" in sys.argv

    if dry_run:
        print("*** DRY RUN MODE — no DB writes ***\n")

    print("V1.4.1 — GEO-ROUTER Training")
    print("=" * 70)
    print(f"MIN_DATE: {MIN_DATE}")
    print(f"Tier Geo: {sorted(TIER_GEO_LEAGUES.keys())} → 18f (16 + geo)")
    print(f"Tier Flat: {sorted(TIER_FLAT_LEAGUES.keys())} → 16f (14 + elo)")
    print(f"LAB_DATA_DIR: {LAB_DATA_DIR}")
    print(f"Dry run: {dry_run}")

    # ── Train GEO model ──
    geo_brier, geo_id, geo_n = await train_model(
        model_version="v1.4.1-latam-geo",
        features=FEATURES_18,
        leagues=TIER_GEO_LEAGUES,
        dry_run=dry_run,
    )

    # ── Train FLAT model ──
    flat_brier, flat_id, flat_n = await train_model(
        model_version="v1.4.1-latam-flat",
        features=FEATURES_16,
        leagues=TIER_FLAT_LEAGUES,
        dry_run=dry_run,
    )

    # ── Skill Score Matrix ──
    print("\n\n" + "=" * 70)
    print("  SKILL SCORE MATRICES")
    print("=" * 70)

    geo_skills = compute_skill_scores(TIER_GEO_LEAGUES, FEATURES_18)
    flat_skills = compute_skill_scores(TIER_FLAT_LEAGUES, FEATURES_16)

    # ── Summary ──
    print("\n\n" + "=" * 70)
    print("  SUMMARY — V1.4.0 GEO-ROUTER")
    print("=" * 70)
    print(f"\n  GEO Model (v1.4.1-latam-geo):")
    print(f"    Features:     18f = FEATURES_16 + altitude_diff_m + travel_distance_km")
    print(f"    Leagues:      {sorted(TIER_GEO_LEAGUES.keys())}")
    print(f"    Samples:      {geo_n:,}")
    print(f"    CV Brier:     {geo_brier:.6f}")
    print(f"    Snapshot ID:  {geo_id}")

    print(f"\n  FLAT Model (v1.4.1-latam-flat):")
    print(f"    Features:     16f = FEATURES_14 + elo_k10_diff + elo_momentum_diff")
    print(f"    Leagues:      {sorted(TIER_FLAT_LEAGUES.keys())}")
    print(f"    Samples:      {flat_n:,}")
    print(f"    CV Brier:     {flat_brier:.6f}")
    print(f"    Snapshot ID:  {flat_id}")

    if not dry_run and geo_id and flat_id:
        print(f"\n  Next steps:")
        print(f"    1. Update app/ml/latam_serving.py for dual engine routing")
        print(f"    2. Update app/events/handlers.py for tier-based cascade")
        print(f"    3. Run latam_sniper_backtest.py with bifurcated models")
        print(f"    4. Deploy to Railway")


if __name__ == "__main__":
    asyncio.run(main())
