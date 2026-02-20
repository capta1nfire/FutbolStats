#!/usr/bin/env python3
"""Train v1.0.3 Two-Stage W3 model (fav/underdog) and persist as INACTIVE snapshot.

ABE Mandato: Routing TS/OS per league.
- Config: W3_ts_odds (3 features: odds_home, odds_draw, odds_away)
- Architecture: Two-Stage (S1: draw vs non-draw, S2: fav vs underdog)
- Fav = team with lower odds. Swap-back at inference to home/away.
- Trained on all 23 leagues pooled, served only for 15 TS leagues
- min_date=2023-01-01, league_only features (pre-computed in cached CSVs)
- Persisted as inactive snapshot (does NOT touch SSOT baseline)

Usage:
    source .env
    python scripts/train_v102_ts.py [--dry-run]
"""

import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).parent.parent))

if not os.environ.get("DATABASE_URL"):
    raise RuntimeError(
        "DATABASE_URL environment variable is required.\n"
        "Run: source .env && python scripts/train_v102_ts.py"
    )

# ─── Constants ────────────────────────────────────────────────
MODEL_VERSION = "v1.0.3-twostage-w3-fav"
MIN_DATE = "2023-01-01"
W3_FEATURES = ["odds_home", "odds_draw", "odds_away"]
DRAW_WEIGHT = 1.2
N_CV_SPLITS = 3

# ABE-approved routing (2026-02-18)
TS_LEAGUES = {39, 61, 78, 88, 94, 140, 203, 239, 253, 262, 265, 268, 299, 307, 344}
OS_LEAGUES = {128, 135, 242, 250, 40, 71, 144, 281}
ALL_LEAGUES = TS_LEAGUES | OS_LEAGUES

LAB_DATA_DIR = Path(__file__).parent / "output" / "lab"

# Hyperparameters (match engine.py TwoStageEngine exactly)
PARAMS_S1 = {
    "objective": "binary:logistic",
    "max_depth": 3,
    "learning_rate": 0.05,
    "n_estimators": 100,
    "min_child_weight": 7,
    "subsample": 0.72,
    "colsample_bytree": 0.71,
    "random_state": 42,
}
PARAMS_S2 = {
    "objective": "binary:logistic",
    "max_depth": 3,
    "learning_rate": 0.05,
    "n_estimators": 100,
    "min_child_weight": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
}


def load_all_league_data():
    """Load and concatenate all 23 league CSVs from Feature Lab cache."""
    frames = []
    per_league = {}

    for lid in sorted(ALL_LEAGUES):
        csv_path = LAB_DATA_DIR / f"lab_data_{lid}.csv"
        if not csv_path.exists():
            print(f"  WARNING: Missing CSV for league {lid}")
            continue

        df = pd.read_csv(csv_path)
        # Filter date >= MIN_DATE
        df = df[df["date"] >= MIN_DATE].copy()

        # Filter valid odds triplet (all three > 0)
        valid_odds = (
            (df["odds_home"] > 0) &
            (df["odds_draw"] > 0) &
            (df["odds_away"] > 0)
        )
        n_before = len(df)
        df = df[valid_odds].copy()
        n_no_odds = n_before - len(df)

        # Filter valid result
        df = df[df["result"].notna()].copy()

        per_league[lid] = {
            "n_total": len(df),
            "n_no_odds": n_no_odds,
            "draw_rate": float((df["result"] == 1).mean()) if len(df) > 0 else 0,
            "date_min": str(df["date"].min()) if len(df) > 0 else None,
            "date_max": str(df["date"].max()) if len(df) > 0 else None,
        }

        frames.append(df)
        print(f"  League {lid}: {len(df)} samples (no_odds={n_no_odds}, "
              f"draw={per_league[lid]['draw_rate']:.1%})")

    if not frames:
        raise RuntimeError("No data loaded!")

    pooled = pd.concat(frames, ignore_index=True)
    pooled = pooled.sort_values("date").reset_index(drop=True)
    return pooled, per_league


def calculate_brier(y_true, y_proba):
    """Multiclass Brier score matching production (app/ml/metrics.py).

    Average of per-class sklearn brier_score_loss. Lower = better.
    """
    from sklearn.metrics import brier_score_loss
    scores = []
    for cls in range(y_proba.shape[1]):
        y_bin = (y_true == cls).astype(int)
        scores.append(brier_score_loss(y_bin, y_proba[:, cls]))
    return float(np.mean(scores))


def train_and_evaluate(df):
    """Train TS W3 with TimeSeriesSplit CV (fav/underdog), then final on all data.

    Stage 2 target: P(fav wins | non-draw).
    Fav = team with lower odds (W3 indices: 0=odds_home, 2=odds_away).
    Swap-back at validation to compute home/away probabilities for Brier.
    """
    from sklearn.model_selection import TimeSeriesSplit

    X = df[W3_FEATURES].fillna(0).values
    y = df["result"].values.astype(int)

    # TimeSeriesSplit CV
    tscv = TimeSeriesSplit(n_splits=N_CV_SPLITS)
    cv_scores = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Stage 1: draw (1) vs non-draw (0)
        y_s1_train = (y_train == 1).astype(int)
        sw_s1 = np.ones(len(y_s1_train), dtype=np.float32)
        sw_s1[y_s1_train == 1] = DRAW_WEIGHT
        s1 = xgb.XGBClassifier(**PARAMS_S1)
        s1.fit(X_train, y_s1_train, sample_weight=sw_s1, verbose=False)

        # Stage 2: fav wins (1) vs underdog wins (0), non-draw only
        nd_mask = y_train != 1
        X_s2_train = X_train[nd_mask]
        odds_h = X_s2_train[:, 0]  # odds_home
        odds_a = X_s2_train[:, 2]  # odds_away
        is_home_fav = odds_h <= odds_a
        home_won = (y_train[nd_mask] == 0)
        y_s2_train = np.where(is_home_fav, home_won, ~home_won).astype(int)

        s2 = xgb.XGBClassifier(**PARAMS_S2)
        s2.fit(X_s2_train, y_s2_train, verbose=False)

        # Predict validation + swap-back
        p_draw = s1.predict_proba(X_val)[:, 1]
        p_s2_raw = s2.predict_proba(X_val)[:, 1]  # P(fav wins | non-draw)
        is_hf_val = X_val[:, 0] <= X_val[:, 2]    # home is fav?
        p_home_nd = np.where(is_hf_val, p_s2_raw, 1 - p_s2_raw)
        p_home = (1 - p_draw) * p_home_nd
        p_away = (1 - p_draw) * (1 - p_home_nd)
        y_proba = np.column_stack([p_home, p_draw, p_away])

        brier = calculate_brier(y_val, y_proba)
        cv_scores.append(brier)
        print(f"  Fold {fold + 1}: Brier = {brier:.4f} "
              f"(n_train={len(train_idx)}, n_val={len(val_idx)})")

    avg_brier = float(np.mean(cv_scores))
    print(f"  Average CV Brier: {avg_brier:.4f}")

    # Train final models on ALL data
    print("\n  Training final models on all data...")
    y_s1 = (y == 1).astype(int)
    sw_s1 = np.ones(len(y_s1), dtype=np.float32)
    sw_s1[y_s1 == 1] = DRAW_WEIGHT

    final_s1 = xgb.XGBClassifier(**PARAMS_S1)
    final_s1.fit(X, y_s1, sample_weight=sw_s1, verbose=False)

    # Final Stage 2: fav/underdog on all non-draws
    nd_mask = y != 1
    X_s2_all = X[nd_mask]
    is_home_fav_all = X_s2_all[:, 0] <= X_s2_all[:, 2]
    home_won_all = (y[nd_mask] == 0)
    y_s2_all = np.where(is_home_fav_all, home_won_all, ~home_won_all).astype(int)

    final_s2 = xgb.XGBClassifier(**PARAMS_S2)
    final_s2.fit(X_s2_all, y_s2_all, verbose=False)

    # Stats for logging
    fav_win_rate = y_s2_all.mean()
    print(f"  Fav win rate (all data): {fav_win_rate:.1%}")

    return final_s1, final_s2, avg_brier, cv_scores


async def persist_snapshot(engine, brier_score, cv_scores, samples_trained,
                           per_league_stats, dry_run=False):
    """Persist model as inactive snapshot to DB."""
    from app.database import get_async_session
    from app.models import ModelSnapshot

    model_blob = engine.save_to_bytes()
    print(f"  Model blob: {len(model_blob)} bytes")

    training_config = {
        "architecture": "two_stage_w3",
        "stage2_semantic": "fav_underdog",
        "draw_weight": DRAW_WEIGHT,
        "features": W3_FEATURES,
        "n_features": len(W3_FEATURES),
        "min_date": MIN_DATE,
        "n_cv_splits": N_CV_SPLITS,
        "ts_leagues": sorted(TS_LEAGUES),
        "os_leagues": sorted(OS_LEAGUES),
        "per_league": per_league_stats,
        "hyperparams_s1": {k: v for k, v in PARAMS_S1.items() if k != "random_state"},
        "hyperparams_s2": {k: v for k, v in PARAMS_S2.items() if k != "random_state"},
    }

    if dry_run:
        print("  [DRY RUN] Would persist snapshot:")
        print(f"    model_version: {engine.model_version}")
        print(f"    brier_score: {brier_score:.4f}")
        print(f"    samples_trained: {samples_trained}")
        print(f"    is_active: False")
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

        print(f"  Snapshot persisted: id={snapshot.id}, is_active=False")

        # Verify load
        from app.ml.engine import TwoStageEngine as TSE
        verify_engine = TSE()
        if verify_engine.load_from_bytes(model_blob):
            test_df = pd.DataFrame({
                "odds_home": [2.10],
                "odds_draw": [3.40],
                "odds_away": [3.50],
            })
            proba = verify_engine.predict_proba(test_df)
            print(f"  Verify: proba={proba[0]}, sum={proba[0].sum():.6f}")
            print(f"  Verify: semantic={verify_engine.stage2_semantic}")
            print(f"  Verify: features_s1={verify_engine.active_stage1_features}")
        else:
            print("  ERROR: Model load verification failed!")

        return snapshot.id


async def main():
    dry_run = "--dry-run" in sys.argv

    print("=" * 70)
    print(f"TRAINING {MODEL_VERSION} — Two-Stage W3 (3 odds features)")
    print(f"min_date={MIN_DATE}, draw_weight={DRAW_WEIGHT}")
    if dry_run:
        print("[DRY RUN MODE — will NOT persist to DB]")
    print("=" * 70)

    # Step 1: Load data
    print("\n--- Step 1: Loading data from Feature Lab cache ---")
    df, per_league = load_all_league_data()
    print(f"\nPooled dataset: {len(df)} samples")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Class distribution: {dict(pd.Series(df['result'].values).value_counts().sort_index())}")
    print(f"Draw rate: {(df['result'] == 1).mean():.1%}")

    # Step 2: Train
    print("\n--- Step 2: Training Two-Stage W3 ---")
    s1, s2, brier, cv_scores = train_and_evaluate(df)

    # Step 3: Package into TwoStageEngine
    print("\n--- Step 3: Packaging into TwoStageEngine ---")
    from app.ml.engine import TwoStageEngine
    engine = TwoStageEngine(
        model_version=MODEL_VERSION,
        draw_weight=DRAW_WEIGHT,
        stage1_features=W3_FEATURES,
        stage2_features=W3_FEATURES,
    )
    engine.stage1 = s1
    engine.stage2 = s2
    engine.stage2_semantic = "fav_underdog"

    # SANITY CHECK: 4 test cases verify fav > underdog directional coherence
    print("\n  Running sanity check (4 test cases)...")
    test_cases = [
        {"odds_home": 1.80, "odds_draw": 3.50, "odds_away": 4.50, "label": "Home fav"},
        {"odds_home": 4.20, "odds_draw": 3.40, "odds_away": 1.90, "label": "Away fav"},
        {"odds_home": 2.90, "odds_draw": 3.30, "odds_away": 2.50, "label": "Close"},
        {"odds_home": 3.00, "odds_draw": 3.00, "odds_away": 3.00, "label": "Equal"},
    ]
    for tc in test_cases:
        label = tc.pop("label")
        proba = engine.predict_proba(pd.DataFrame([tc]))
        p = proba[0]
        assert abs(p.sum() - 1.0) < 1e-6, f"Sanity FAIL: sum={p.sum()}"
        if tc["odds_home"] < tc["odds_away"]:
            assert p[0] > p[2], f"Sanity FAIL ({label}): p_home={p[0]:.4f} <= p_away={p[2]:.4f}"
        elif tc["odds_away"] < tc["odds_home"]:
            assert p[2] > p[0], f"Sanity FAIL ({label}): p_away={p[2]:.4f} <= p_home={p[0]:.4f}"
        print(f"    {label}: H={p[0]:.4f} D={p[1]:.4f} A={p[2]:.4f} ✓")
    print("  Sanity check PASSED")

    # Step 4: Persist
    print("\n--- Step 4: Persisting as inactive snapshot ---")
    snapshot_id = await persist_snapshot(
        engine, brier, cv_scores, len(df), per_league, dry_run=dry_run,
    )

    # Summary
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"  Model version: {MODEL_VERSION}")
    print(f"  Stage 2 semantic: fav_underdog")
    print(f"  Features: {W3_FEATURES}")
    print(f"  Samples: {len(df)}")
    print(f"  CV Brier: {brier:.4f}")
    print(f"  CV Scores: {[round(float(s), 4) for s in cv_scores]}")
    print(f"  TS leagues (15): {sorted(TS_LEAGUES)}")
    print(f"  OS leagues (8):  {sorted(OS_LEAGUES)}")
    if snapshot_id:
        print(f"  Snapshot ID: {snapshot_id}")
        print(f"  is_active: False (SSOT baseline untouched)")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
