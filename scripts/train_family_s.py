#!/usr/bin/env python3
"""Train Family S model (v2.0-tier3-family_s) for Tier 3 leagues.

P0-1: Uses FeatureEngineer.build_training_dataset() for feature fidelity
(1:1 with serving/cascade). Merges MTV from canonical parquet.

Feature set (24): 17 core + 3 odds + 4 MTV

Usage:
    source .env
    python scripts/train_family_s.py
    python scripts/train_family_s.py --min-date 2023-01-01 --persist
    python scripts/train_family_s.py --persist --dry-run  # show stats only
"""

import argparse
import asyncio
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("train_family_s")

# ── Constants ────────────────────────────────────────────────────────────────

MODEL_VERSION = "v2.0-tier3-family_s"
TIER_3_LEAGUES = [94, 265, 88, 203, 144]  # Primeira, Chile, Eredivisie, Turkey, Belgium
MTV_PARQUET = "data/historical_mtv_features_tm_hiconf_padded.parquet"
DEFAULT_MIN_DATE = "2023-01-01"

MTV_COLS = [
    "home_talent_delta",
    "away_talent_delta",
    "talent_delta_diff",
    "shock_magnitude",
]


async def build_dataset(min_date: str, max_date: str = None) -> pd.DataFrame:
    """Build training dataset using FeatureEngineer (P0-1 fidelity)."""
    from app.database import AsyncSessionLocal
    from app.features.engineering import FeatureEngineer

    min_dt = datetime.strptime(min_date, "%Y-%m-%d")
    max_dt = datetime.strptime(max_date, "%Y-%m-%d") if max_date else None

    logger.info(
        "Building features via FeatureEngineer "
        "(leagues=%s, min_date=%s, max_date=%s)",
        TIER_3_LEAGUES, min_date, max_date or "None",
    )

    async with AsyncSessionLocal() as session:
        fe = FeatureEngineer(session=session)
        df = await fe.build_training_dataset(
            league_only=True,
            league_ids=TIER_3_LEAGUES,
            min_date=min_dt,
            max_date=max_dt,
        )

    logger.info("FeatureEngineer returned %d rows, %d columns", len(df), len(df.columns))
    return df


def merge_mtv(df: pd.DataFrame) -> pd.DataFrame:
    """Merge MTV features from canonical parquet by match_id."""
    parquet_path = Path(MTV_PARQUET)
    if not parquet_path.exists():
        logger.error("MTV parquet not found: %s", MTV_PARQUET)
        sys.exit(1)

    mtv = pd.read_parquet(parquet_path)
    mtv = mtv[mtv["league_id"].isin(TIER_3_LEAGUES)]
    logger.info(
        "MTV parquet: %d rows for Tier 3 leagues (total %d in file)",
        len(mtv), len(pd.read_parquet(parquet_path)),
    )

    before = len(df)
    df = df.merge(
        mtv[["match_id"] + MTV_COLS],
        on="match_id",
        how="left",
    )
    logger.info("After MTV merge: %d rows (was %d)", len(df), before)

    # Preserve NaN for MTV — XGBoost sparsity-aware split handles missing natively.
    # 0.0 has semantic meaning (talent matched expectation), NaN = no data.
    for col in MTV_COLS:
        n_missing = df[col].isna().sum()
        if n_missing > 0:
            logger.info("  %s: %d missing (NaN preserved for XGBoost sparsity)", col, n_missing)

    return df


def filter_odds(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows without complete odds triplet."""
    before = len(df)
    mask = (
        df["odds_home"].notna()
        & df["odds_draw"].notna()
        & df["odds_away"].notna()
    )
    df = df[mask].reset_index(drop=True)
    dropped = before - len(df)
    logger.info(
        "Odds filter: dropped %d rows without complete odds (%d remain, %.1f%%)",
        dropped, len(df), 100 * len(df) / before if before else 0,
    )
    return df


def train_model(df: pd.DataFrame, dry_run: bool = False) -> dict:
    """Train FamilySEngine on the dataset."""
    from app.ml.family_s import FamilySEngine

    engine = FamilySEngine(model_version=MODEL_VERSION)

    # Verify all 24 features are present
    missing = [c for c in engine.FEATURE_COLUMNS if c not in df.columns]
    if missing:
        logger.error("Missing features in df: %s", missing)
        sys.exit(1)

    logger.info(
        "Training %s with %d samples, %d features",
        MODEL_VERSION, len(df), len(engine.FEATURE_COLUMNS),
    )

    if dry_run:
        logger.info("[DRY-RUN] Skipping actual training")
        return {"dry_run": True, "samples": len(df)}

    # Train uses inherited XGBoostEngine.train() which calls _prepare_features()
    # → uses FamilySEngine.FEATURE_COLUMNS (24 features)
    result = engine.train(df, n_splits=3)

    logger.info(
        "Training complete: brier=%.4f, cv_scores=%s, samples=%d",
        result.get("brier_score", 0),
        result.get("cv_brier_scores", []),
        len(df),
    )

    # Save local artifact
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    date_str = datetime.now().strftime("%Y%m%d")
    local_path = models_dir / f"xgb_{MODEL_VERSION}_{date_str}.json"
    engine.model.save_model(str(local_path))
    logger.info("Local artifact saved: %s", local_path)

    return {
        "engine": engine,
        "brier_score": result.get("brier_score", 0),
        "cv_brier_scores": result.get("cv_brier_scores", []),
        "samples": len(df),
        "local_path": str(local_path),
    }


async def persist_to_db(train_result: dict, min_date: str):
    """Persist Family S snapshot to DB (is_active=False, P0-3)."""
    from app.database import AsyncSessionLocal
    from app.ml.persistence import persist_family_s_snapshot

    engine = train_result["engine"]
    training_config = {
        "league_ids": TIER_3_LEAGUES,
        "league_only": True,
        "min_date": min_date,
        "feature_set_name": "core17+odds3+mtv4",
        "feature_columns": engine.FEATURE_COLUMNS,
        "mtv_source": "tm_hiconf_padded",
        "n_features": len(engine.FEATURE_COLUMNS),
        "hyperparams": "baseline_v1_optuna",
    }

    async with AsyncSessionLocal() as session:
        snapshot_id = await persist_family_s_snapshot(
            session=session,
            engine=engine,
            brier_score=train_result["brier_score"],
            cv_scores=train_result["cv_brier_scores"],
            samples_trained=train_result["samples"],
            training_config=training_config,
        )

    logger.info("Snapshot persisted to DB: id=%d, is_active=False", snapshot_id)
    return snapshot_id


async def main():
    parser = argparse.ArgumentParser(description="Train Family S model (Mandato D)")
    parser.add_argument("--min-date", default=DEFAULT_MIN_DATE, help="Min training date (YYYY-MM-DD)")
    parser.add_argument("--max-date", default=None, help="Max training date (YYYY-MM-DD)")
    parser.add_argument("--persist", action="store_true", help="Save snapshot to DB (is_active=False)")
    parser.add_argument("--dry-run", action="store_true", help="Build dataset but skip training")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Family S Training — Mandato D")
    logger.info("Model: %s", MODEL_VERSION)
    logger.info("Leagues: %s", TIER_3_LEAGUES)
    logger.info("Min date: %s", args.min_date)
    logger.info("Persist: %s", args.persist)
    logger.info("=" * 60)

    # Step 1: Build features via FeatureEngineer (P0-1)
    df = await build_dataset(args.min_date, args.max_date)
    if len(df) == 0:
        logger.error("Empty dataset, aborting")
        sys.exit(1)

    # Step 2: Merge MTV from parquet
    df = merge_mtv(df)

    # Step 3: Filter rows without odds
    df = filter_odds(df)

    # Step 4: Print dataset summary
    logger.info("--- Dataset Summary ---")
    logger.info("Total samples: %d", len(df))
    logger.info("Leagues: %s", sorted(df["league_id"].unique().tolist()) if "league_id" in df.columns else "N/A")
    logger.info("Date range: %s to %s",
                df["match_date"].min() if "match_date" in df.columns else "?",
                df["match_date"].max() if "match_date" in df.columns else "?")
    for col in MTV_COLS:
        nonzero = (df[col] != 0.0).sum()
        logger.info("  %s: %.1f%% non-zero", col, 100 * nonzero / len(df) if len(df) else 0)

    # Step 5: Train
    result = train_model(df, dry_run=args.dry_run)

    if args.dry_run:
        logger.info("[DRY-RUN] Done. No model trained or persisted.")
        return

    # Step 6: Persist to DB
    if args.persist:
        snapshot_id = await persist_to_db(result, args.min_date)
        logger.info("Done. Snapshot #%d saved (is_active=False). Baseline untouched.", snapshot_id)
    else:
        logger.info("Done. Local artifact: %s (use --persist to save to DB)", result["local_path"])


if __name__ == "__main__":
    asyncio.run(main())
