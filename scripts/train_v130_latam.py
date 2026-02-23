#!/usr/bin/env python3
"""V1.3.0 — LATAM-First Golden Snapshot Training & Persistence.

Trains the V1.3.0 LATAM Baseline: 18f TwoStageEngine with home_away semantic,
dw=1.0, on 100% of Lab data (11 LATAM leagues). Persists to model_snapshots
with is_active=False.

Alpha Lab Sprint 1 result: T0 (Global LATAM 18f) beat Tprod (23-league 16f)
with Weighted Skill -1.8% vs -2.2%. Chile +1.8%, Argentina +0.7%.

Usage:
    source .env
    python scripts/train_v130_latam.py [--dry-run]
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
        "Run: source .env && python scripts/train_v130_latam.py"
    )

# ─── Constants ────────────────────────────────────────────────────────────────
MIN_DATE = "2023-01-01"
N_CV_SPLITS = 3
MODEL_VERSION = "v1.3.0-latam-first"
LAB_DATA_DIR = Path(__file__).parent / "output" / "lab"

LATAM_LEAGUES = {
    128: "Argentina", 71: "Brazil", 239: "Colombia", 242: "Ecuador",
    262: "Mexico", 265: "Chile", 268: "Uruguay", 281: "Peru",
    299: "Venezuela", 344: "Bolivia", 250: "Paraguay",
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
FEATURES_16 = FEATURES_14 + MOMENTUM_2

GEO_2 = ["altitude_diff_m", "travel_distance_km"]
FEATURES_LATAM = FEATURES_16 + GEO_2  # 18f

# ─── Hyperparameters (exact production TwoStage params) ──────────────────────

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


def load_lab_csvs(leagues):
    """Load Lab CSVs for specified leagues."""
    frames = []
    for lid in sorted(leagues):
        csv_path = LAB_DATA_DIR / f"lab_data_{lid}.csv"
        if not csv_path.exists():
            print(f"  WARNING: Missing CSV for league {lid}")
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
# V1.3.0 LATAM-First: 18f TwoStageEngine, home_away, dw=1.0
# ═══════════════════════════════════════════════════════════════════════════════

async def train_v130_latam(dry_run=False):
    """V1.3.0 LATAM-First: 18f TwoStageEngine with home_away semantic."""
    from app.ml.engine import TwoStageEngine

    print("\n" + "=" * 70)
    print(f"  {MODEL_VERSION} (18f TwoStage, home_away, dw=1.0)")
    print(f"  New features: +altitude_diff_m, +travel_distance_km")
    print(f"  Training pool: 11 LATAM leagues")
    print("=" * 70)

    # Load data (NO odds filter — 18f doesn't use odds as features)
    print("\nLoading Lab CSVs (11 LATAM leagues)...")
    df = load_lab_csvs(LATAM_LEAGUES)
    print(f"  Total samples: {len(df)}")

    # Verify 18 features exist
    missing = [f for f in FEATURES_LATAM if f not in df.columns]
    if missing:
        raise RuntimeError(
            f"Lab CSVs missing features: {missing}. "
            f"Re-run feature extraction with geo features."
        )

    # NaN report for geo features (XGBoost handles NaN natively)
    for f in GEO_2:
        n_nan = df[f].isna().sum()
        pct = 100 * n_nan / len(df)
        print(f"  {f}: {n_nan} NaN ({pct:.1f}%)")

    df = df.sort_values(["date", "match_id"]).reset_index(drop=True)
    y = df["result"].values
    # fillna(0) aligns training with serving (_prepare_features does fillna(0))
    X = df[FEATURES_LATAM].fillna(0).values

    # Sample weights (dw=1.0 = uniform)
    sample_weight = np.ones(len(y), dtype=np.float32)

    # ── CV evaluation (for metadata, not for training) ──
    print("\nCV evaluation (TimeSeriesSplit, n=3)...")

    y_draw = (y == 1).astype(int)
    tscv = TimeSeriesSplit(n_splits=N_CV_SPLITS)
    fold_scores = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        # Stage 1: draw vs non-draw
        s1 = xgb.XGBClassifier(**PARAMS_TS_S1)
        s1.fit(X[train_idx], y_draw[train_idx],
               sample_weight=sample_weight[train_idx], verbose=False)

        # Stage 2: home vs away (non-draws only), home_away semantic
        nd_train = y[train_idx] != 1
        idx_nd = train_idx[nd_train]
        y_home_train = (y[idx_nd] == 0).astype(int)
        s2 = xgb.XGBClassifier(**PARAMS_TS_S2)
        s2.fit(X[idx_nd], y_home_train, verbose=False)

        # Predict validation
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

    # ── Train final on 100% data ──
    print("\nTraining final model on 100% data...")

    s1_final = xgb.XGBClassifier(**PARAMS_TS_S1)
    s1_final.fit(X, y_draw, sample_weight=sample_weight, verbose=False)

    nondraw_mask = y != 1
    y_stage2_full = (y[nondraw_mask] == 0).astype(int)
    s2_final = xgb.XGBClassifier(**PARAMS_TS_S2)
    s2_final.fit(X[nondraw_mask], y_stage2_full, verbose=False)

    # Package into TwoStageEngine with custom 18f feature lists
    engine = TwoStageEngine(
        model_version=MODEL_VERSION,
        draw_weight=1.0,
        stage1_features=FEATURES_LATAM,
        stage2_features=FEATURES_LATAM,
    )
    engine.stage1 = s1_final
    engine.stage2 = s2_final
    engine.stage2_semantic = "home_away"

    print(f"  Stage 1: {s1_final.n_features_in_} features")
    print(f"  Stage 2: {s2_final.n_features_in_} features")
    print(f"  Semantic: {engine.stage2_semantic}")
    print(f"  Features: {FEATURES_LATAM}")

    # Verify round-trip
    print("\nVerify round-trip (load from bytes)...")
    verify = TwoStageEngine()
    blob = engine.save_to_bytes()
    assert verify.load_from_bytes(blob), "Round-trip load failed!"
    assert verify._custom_stage1_features == FEATURES_LATAM, \
        f"Stage1 features mismatch: {verify._custom_stage1_features}"
    assert verify._custom_stage2_features == FEATURES_LATAM, \
        f"Stage2 features mismatch: {verify._custom_stage2_features}"
    assert verify.stage2_semantic == "home_away", \
        f"Semantic mismatch: {verify.stage2_semantic}"

    # Quick predict sanity
    test_row = pd.DataFrame({f: [0.0] for f in FEATURES_LATAM})
    test_proba = verify.predict_proba(test_row)
    assert test_proba.shape == (1, 3), f"Bad proba shape: {test_proba.shape}"
    assert abs(test_proba.sum() - 1.0) < 0.001, f"Probs don't sum to 1: {test_proba.sum()}"
    print(f"  Round-trip OK. Test proba: {test_proba[0]}")

    # Persist
    training_config = {
        "architecture": "two_stage",
        "stage2_semantic": "home_away",
        "draw_weight": 1.0,
        "features_s1": FEATURES_LATAM,
        "features_s2": FEATURES_LATAM,
        "n_features": len(FEATURES_LATAM),
        "min_date": MIN_DATE,
        "n_cv_splits": N_CV_SPLITS,
        "leagues": sorted(LATAM_LEAGUES.keys()),
        "league_names": {str(k): v for k, v in LATAM_LEAGUES.items()},
        "hyperparams_s1": {k: v for k, v in PARAMS_TS_S1.items() if k != "random_state"},
        "hyperparams_s2": {k: v for k, v in PARAMS_TS_S2.items() if k != "random_state"},
        "alpha_lab_ref": "T0 Global LATAM 18f (Sprint 1 winner)",
        "alpha_lab_weighted_skill": -0.018,
    }

    snapshot_id = await persist_snapshot(
        engine, cv_avg, fold_scores, len(df), training_config, dry_run
    )

    return cv_avg, snapshot_id


# ═══════════════════════════════════════════════════════════════════════════════

async def main():
    dry_run = "--dry-run" in sys.argv

    if dry_run:
        print("*** DRY RUN MODE — no DB writes ***\n")

    print("V1.3.0 — LATAM-First Golden Snapshot Training")
    print("=" * 70)
    print(f"MIN_DATE: {MIN_DATE}")
    print(f"MODEL_VERSION: {MODEL_VERSION}")
    print(f"FEATURES: {len(FEATURES_LATAM)}f = 14 baseline + 2 momentum + 2 geo")
    print(f"LEAGUES: {len(LATAM_LEAGUES)} LATAM")
    print(f"LAB_DATA_DIR: {LAB_DATA_DIR}")
    print(f"Dry run: {dry_run}")

    cv_avg, snapshot_id = await train_v130_latam(dry_run)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY — Golden Snapshot V1.3.0 LATAM-First")
    print("=" * 70)
    print(f"  Model:        {MODEL_VERSION}")
    print(f"  Architecture: TwoStage (home_away, dw=1.0)")
    print(f"  Features:     {len(FEATURES_LATAM)}f = {FEATURES_LATAM}")
    print(f"  Leagues:      {len(LATAM_LEAGUES)} LATAM")
    print(f"  CV Brier:     {cv_avg:.6f}")
    print(f"  Snapshot ID:  {snapshot_id}")
    print()

    if not dry_run and snapshot_id:
        print("Next steps:")
        print("  1. Implement dual serving (latam_baseline_serving.py)")
        print("  2. Route LATAM league_ids to this engine at inference time")
        print("  3. Do NOT activate globally — this is LATAM-only baseline")
        print(f"  4. Snapshot ID for reference: {snapshot_id}")


if __name__ == "__main__":
    asyncio.run(main())
