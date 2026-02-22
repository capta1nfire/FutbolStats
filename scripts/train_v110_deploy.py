#!/usr/bin/env python3
"""V1.1.0 Cross-Wire — Golden Snapshot Training & Persistence.

Trains 3 models on 100% data (no CV holdout) with dw=1.0 and persists
them to model_snapshots with is_active=False.

Model A: v1.1.0-league-only  — TwoStageEngine, 14f, home_away semantic
Model B: v1.1.0-twostage     — XGBoostEngine,  3f, multi:softprob (overlay slot)
Model C: v2.1.1-family_s     — XGBoostEngine, 21f, multi:softprob (Family S slot)

All models: draw_weight=1.0, all 23 leagues (A/B) or 10 Tier 3 (C).

Usage:
    source .env
    python scripts/train_v110_deploy.py [--dry-run]
"""

import asyncio
import gc
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
        "DATABASE_URL required.\nRun: source .env && python scripts/train_v110_deploy.py"
    )

# ─── Constants ────────────────────────────────────────────────────────────────
MIN_DATE = "2023-01-01"
N_CV_SPLITS = 3
LAB_DATA_DIR = Path(__file__).parent / "output" / "lab"

ALL_LEAGUES = {
    39, 40, 61, 71, 78, 88, 94, 128, 135, 140, 144,
    203, 239, 242, 250, 253, 262, 265, 268, 281, 299, 307, 344,
}
TIER3_LEAGUES = {88, 94, 203, 242, 262, 265, 268, 281, 299, 344}

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

FEATURES_3 = ["odds_home", "odds_draw", "odds_away"]

FEATURES_21 = FEATURES_14 + [
    "odds_home", "odds_draw", "odds_away",
    "home_talent_delta", "away_talent_delta",
    "talent_delta_diff", "shock_magnitude",
]

# ─── Hyperparameters (from crosswire experiment) ──────────────────────────────

# TS-native params (used for Model A: 14f TwoStage)
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

# One-Stage adapted from TS S1 (used for Model B: 3f overlay)
PARAMS_OS_FROM_TS = {
    "objective": "multi:softprob",
    "num_class": 3,
    "max_depth": 3,
    "learning_rate": 0.05,
    "n_estimators": 100,
    "min_child_weight": 7,
    "subsample": 0.72,
    "colsample_bytree": 0.71,
    "eval_metric": "mlogloss",
    "random_state": 42,
}

# Optuna-optimized (used for Model C: 21f Family S)
PARAMS_OPTUNA = {
    "objective": "multi:softprob",
    "num_class": 3,
    "max_depth": 3,
    "learning_rate": 0.0283,
    "n_estimators": 114,
    "min_child_weight": 7,
    "subsample": 0.72,
    "colsample_bytree": 0.71,
    "reg_alpha": 2.8e-05,
    "reg_lambda": 0.000904,
    "eval_metric": "mlogloss",
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


def load_lab_csvs(leagues, require_odds=False):
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

        if require_odds:
            df = df[
                (df["odds_home"] > 0)
                & (df["odds_draw"] > 0)
                & (df["odds_away"] > 0)
            ].copy()

        frames.append(df)
        print(f"  League {lid}: {len(df)} samples")

    if not frames:
        raise RuntimeError("No data loaded!")

    pooled = pd.concat(frames, ignore_index=True)
    pooled = pooled.sort_values("date").reset_index(drop=True)
    return pooled


def cv_evaluate(X, y, model_fn, sample_weight=None):
    """Run TimeSeriesSplit CV and return (avg_brier, per_fold_scores).

    model_fn(X_train, y_train, w_train) -> model that has predict_proba().
    """
    tscv = TimeSeriesSplit(n_splits=N_CV_SPLITS)
    fold_scores = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        w_train = sample_weight[train_idx] if sample_weight is not None else None

        model = model_fn(X_train, y_train, w_train)
        y_proba = model.predict_proba(X_val)
        brier = calculate_brier(y_val, y_proba)
        fold_scores.append(brier)
        print(f"    Fold {fold + 1}: Brier = {brier:.6f}")

    avg = float(np.mean(fold_scores))
    print(f"    CV Average: {avg:.6f}")
    return avg, fold_scores


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

        print(f"  ✓ Snapshot persisted: id={snapshot.id}, "
              f"version={engine.model_version}, is_active=False")
        return snapshot.id


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL A: v1.1.0-league-only (TwoStageEngine, 14f, home_away, dw=1.0)
# ═══════════════════════════════════════════════════════════════════════════════

async def train_model_a(dry_run=False):
    """New baseline: 14f TwoStageEngine with home_away semantic."""
    from app.ml.engine import TwoStageEngine

    print("\n" + "=" * 70)
    print("MODEL A: v1.1.0-league-only (14f TwoStage, home_away, dw=1.0)")
    print("=" * 70)

    # Load data (NO odds filter — 14f doesn't use odds as features)
    print("\nLoading Lab CSVs (23 leagues, NO odds filter)...")
    df = load_lab_csvs(ALL_LEAGUES, require_odds=False)
    print(f"  Total samples: {len(df)}")

    # Ensure 14 features exist
    for col in FEATURES_14:
        if col not in df.columns:
            print(f"  WARNING: Missing feature {col}, filling with 0")
            df[col] = 0.0

    df = df.sort_values("date").reset_index(drop=True)
    y = df["result"].values
    X_s1 = df[FEATURES_14].fillna(0).values
    X_s2 = df[FEATURES_14].fillna(0).values  # Same features for both stages

    # Sample weights for Stage 1 (dw=1.0 = uniform)
    sample_weight_s1 = np.ones(len(y), dtype=np.float32)
    # draw_weight=1.0 → no upweighting

    # ── CV evaluation ──
    print("\nCV evaluation (TimeSeriesSplit, n=3)...")

    y_draw = (y == 1).astype(int)
    nondraw_mask = y != 1
    y_stage2_full = (y[nondraw_mask] == 0).astype(int)  # home_away semantic

    tscv = TimeSeriesSplit(n_splits=N_CV_SPLITS)
    fold_scores = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_s1)):
        # Stage 1
        s1 = xgb.XGBClassifier(**PARAMS_TS_S1)
        s1.fit(X_s1[train_idx], y_draw[train_idx],
               sample_weight=sample_weight_s1[train_idx], verbose=False)

        # Stage 2 (non-draws in train only)
        nd_train = y[train_idx] != 1
        idx_nd = train_idx[nd_train]
        y_home_train = (y[idx_nd] == 0).astype(int)
        s2 = xgb.XGBClassifier(**PARAMS_TS_S2)
        s2.fit(X_s2[idx_nd], y_home_train, verbose=False)

        # Predict validation
        p_draw = s1.predict_proba(X_s1[val_idx])[:, 1]
        p_home_nd = s2.predict_proba(X_s2[val_idx])[:, 1]
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
    s1_final.fit(X_s1, y_draw, sample_weight=sample_weight_s1, verbose=False)

    s2_final = xgb.XGBClassifier(**PARAMS_TS_S2)
    s2_final.fit(X_s2[nondraw_mask], y_stage2_full, verbose=False)

    # Package into TwoStageEngine
    engine = TwoStageEngine(
        model_version="v1.1.0-league-only",
        draw_weight=1.0,
        stage1_features=FEATURES_14,
        stage2_features=FEATURES_14,
    )
    engine.stage1 = s1_final
    engine.stage2 = s2_final
    engine.stage2_semantic = "home_away"

    print(f"  Stage 1: {s1_final.n_features_in_} features")
    print(f"  Stage 2: {s2_final.n_features_in_} features")
    print(f"  Semantic: {engine.stage2_semantic}")

    # Persist
    training_config = {
        "architecture": "two_stage",
        "stage2_semantic": "home_away",
        "draw_weight": 1.0,
        "features_s1": FEATURES_14,
        "features_s2": FEATURES_14,
        "n_features": 14,
        "min_date": MIN_DATE,
        "n_cv_splits": N_CV_SPLITS,
        "leagues": sorted(ALL_LEAGUES),
        "hyperparams_s1": {k: v for k, v in PARAMS_TS_S1.items() if k != "random_state"},
        "hyperparams_s2": {k: v for k, v in PARAMS_TS_S2.items() if k != "random_state"},
        "crosswire_ref_brier": 0.203664,
    }

    snapshot_id = await persist_snapshot(
        engine, cv_avg, fold_scores, len(df), training_config, dry_run
    )

    return cv_avg, snapshot_id


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL B: v1.1.0-twostage (XGBoostEngine, 3f odds, multi:softprob, dw=1.0)
# ═══════════════════════════════════════════════════════════════════════════════

async def train_model_b(dry_run=False):
    """New overlay: 3f one-stage XGBoostEngine (odds only)."""
    from app.ml.engine import XGBoostEngine

    print("\n" + "=" * 70)
    print("MODEL B: v1.1.0-twostage (3f One-Stage XGBoost, dw=1.0)")
    print("=" * 70)

    # Load data (WITH odds filter — 3f requires odds)
    print("\nLoading Lab CSVs (23 leagues, WITH odds filter)...")
    df = load_lab_csvs(ALL_LEAGUES, require_odds=True)
    print(f"  Total samples: {len(df)}")

    df = df.sort_values("date").reset_index(drop=True)
    y = df["result"].values
    X = df[FEATURES_3].fillna(0).values

    # Sample weights (dw=1.0 = uniform)
    sample_weight = np.ones(len(y), dtype=np.float32)

    # ── CV evaluation ──
    print("\nCV evaluation (TimeSeriesSplit, n=3)...")

    def train_os(X_tr, y_tr, w_tr):
        m = xgb.XGBClassifier(**PARAMS_OS_FROM_TS)
        m.fit(X_tr, y_tr, sample_weight=w_tr, verbose=False)
        return m

    cv_avg, fold_scores = cv_evaluate(X, y, train_os, sample_weight)

    # ── Train final on 100% data ──
    print("\nTraining final model on 100% data...")
    final_model = xgb.XGBClassifier(**PARAMS_OS_FROM_TS)
    final_model.fit(X, y, sample_weight=sample_weight, verbose=False)

    # Package into XGBoostEngine
    engine = XGBoostEngine(model_version="v1.1.0-twostage")
    engine.model = final_model
    engine.FEATURE_COLUMNS = FEATURES_3

    print(f"  Features: {engine.FEATURE_COLUMNS}")
    print(f"  n_features: {final_model.n_features_in_}")

    # Persist
    training_config = {
        "architecture": "one_stage",
        "draw_weight": 1.0,
        "features": FEATURES_3,
        "n_features": 3,
        "min_date": MIN_DATE,
        "n_cv_splits": N_CV_SPLITS,
        "leagues": sorted(ALL_LEAGUES),
        "hyperparams": {k: v for k, v in PARAMS_OS_FROM_TS.items() if k != "random_state"},
        "crosswire_ref_brier": 0.194890,
    }

    snapshot_id = await persist_snapshot(
        engine, cv_avg, fold_scores, len(df), training_config, dry_run
    )

    return cv_avg, snapshot_id


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL C: v2.1.1-family_s (XGBoostEngine, 21f, Optuna, dw=1.0)
# ═══════════════════════════════════════════════════════════════════════════════

async def train_model_c(dry_run=False):
    """Family S with dw=1.0 (purged from dw=1.5)."""
    from app.ml.engine import XGBoostEngine

    print("\n" + "=" * 70)
    print("MODEL C: v2.1.1-family_s (21f One-Stage XGBoost, dw=1.0)")
    print("=" * 70)

    # Load Lab CSVs for Tier 3 only (WITH odds filter)
    print("\nLoading Lab CSVs (10 Tier 3 leagues, WITH odds filter)...")
    df = load_lab_csvs(TIER3_LEAGUES, require_odds=True)
    print(f"  Lab CSV rows (after odds filter): {len(df)}")

    # Merge MTV features from parquet
    mtv_path = Path("data/historical_mtv_features_tm_hiconf_padded.parquet")
    mtv_cols = [
        "home_talent_delta", "away_talent_delta",
        "talent_delta_diff", "shock_magnitude",
    ]

    if mtv_path.exists():
        mtv_df = pd.read_parquet(mtv_path)
        if "match_id" in df.columns and "match_id" in mtv_df.columns:
            merge_cols = ["match_id"] + [c for c in mtv_cols if c in mtv_df.columns]
            df = df.merge(mtv_df[merge_cols], on="match_id", how="left")
            n_mtv = df[mtv_cols[0]].notna().sum()
            print(f"  MTV merged: {n_mtv}/{len(df)} matches with MTV data")
        else:
            print("  WARNING: Cannot merge MTV (no match_id)")
            for col in mtv_cols:
                df[col] = np.nan
    else:
        print(f"  WARNING: MTV parquet not found: {mtv_path}")
        for col in mtv_cols:
            df[col] = np.nan

    # Ensure all 21 feature columns exist
    for col in FEATURES_21:
        if col not in df.columns:
            print(f"  WARNING: Missing feature {col}, filling with NaN")
            df[col] = np.nan

    df = df.sort_values("date").reset_index(drop=True)
    print(f"  Total samples: {len(df)}")

    y = df["result"].values
    X = df[FEATURES_21].fillna(0).values

    # Sample weights (dw=1.0 = uniform)
    sample_weight = np.ones(len(y), dtype=np.float32)

    # ── CV evaluation ──
    print("\nCV evaluation (TimeSeriesSplit, n=3)...")

    def train_os(X_tr, y_tr, w_tr):
        m = xgb.XGBClassifier(**PARAMS_OPTUNA)
        m.fit(X_tr, y_tr, sample_weight=w_tr, verbose=False)
        return m

    cv_avg, fold_scores = cv_evaluate(X, y, train_os, sample_weight)

    # ── Train final on 100% data ──
    print("\nTraining final model on 100% data...")
    final_model = xgb.XGBClassifier(**PARAMS_OPTUNA)
    final_model.fit(X, y, sample_weight=sample_weight, verbose=False)

    # Package into XGBoostEngine
    engine = XGBoostEngine(model_version="v2.1.1-family_s")
    engine.model = final_model
    engine.FEATURE_COLUMNS = FEATURES_21

    print(f"  Features: {len(engine.FEATURE_COLUMNS)} columns")
    print(f"  n_features: {final_model.n_features_in_}")

    # Persist
    training_config = {
        "architecture": "one_stage",
        "draw_weight": 1.0,
        "features": FEATURES_21,
        "n_features": 21,
        "min_date": MIN_DATE,
        "n_cv_splits": N_CV_SPLITS,
        "leagues": sorted(TIER3_LEAGUES),
        "hyperparams": {k: v for k, v in PARAMS_OPTUNA.items() if k != "random_state"},
        "crosswire_ref_brier": 0.190617,
    }

    snapshot_id = await persist_snapshot(
        engine, cv_avg, fold_scores, len(df), training_config, dry_run
    )

    return cv_avg, snapshot_id


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

async def main():
    dry_run = "--dry-run" in sys.argv

    if dry_run:
        print("*** DRY RUN MODE — no DB writes ***\n")

    print("V1.1.0 Cross-Wire — Golden Snapshot Training")
    print("=" * 70)
    print(f"MIN_DATE: {MIN_DATE}")
    print(f"N_CV_SPLITS: {N_CV_SPLITS}")
    print(f"LAB_DATA_DIR: {LAB_DATA_DIR}")
    print(f"Dry run: {dry_run}")

    results = {}

    # Model A
    brier_a, id_a = await train_model_a(dry_run)
    results["A"] = {"brier": brier_a, "snapshot_id": id_a}
    gc.collect()  # ABE directive: prevent OOM

    # Model B
    brier_b, id_b = await train_model_b(dry_run)
    results["B"] = {"brier": brier_b, "snapshot_id": id_b}
    gc.collect()

    # Model C
    brier_c, id_c = await train_model_c(dry_run)
    results["C"] = {"brier": brier_c, "snapshot_id": id_c}
    gc.collect()

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY — Golden Snapshots")
    print("=" * 70)
    print(f"  Model A (v1.1.0-league-only): Brier={brier_a:.6f}, snapshot_id={id_a}")
    print(f"  Model B (v1.1.0-twostage):    Brier={brier_b:.6f}, snapshot_id={id_b}")
    print(f"  Model C (v2.1.1-family_s):    Brier={brier_c:.6f}, snapshot_id={id_c}")
    print()

    if not dry_run:
        print("Next steps:")
        print("  1. Push code changes to main (auto-deploy)")
        print("  2. Wait for deploy to complete")
        print("  3. Activate baseline:")
        print(f"     UPDATE model_snapshots SET is_active=False WHERE is_active=True;")
        print(f"     UPDATE model_snapshots SET is_active=True WHERE id={id_a};")
        print("  4. Restart Railway or wait for next prediction cycle")
        print("  5. Verify: railway logs -n 20 | grep 'model loaded'")


if __name__ == "__main__":
    asyncio.run(main())
