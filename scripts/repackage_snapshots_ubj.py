"""Re-package model_snapshots from pickle to UBJ format.

One-off migration script (Paso A of XGBoost upgrade plan).
Loads existing pickle blobs, re-serializes as UBJ, and updates the DB.

Targets:
  - id=4: v1.0.1-league-only (active baseline, XGBoostEngine)
  - id=7: v2.0-tier3-family_s (Family S, FamilySEngine → XGBoostEngine)
  - id=8: v1.0.2-twostage-w3 (Two-Stage W3, TwoStageEngine)

Usage:
    source .env
    python scripts/repackage_snapshots_ubj.py --dry-run   # verify
    python scripts/repackage_snapshots_ubj.py              # execute
"""

import argparse
import asyncio
import logging
import os
import sys
import zlib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

from app.ml.engine import XGBoostEngine, TwoStageEngine, _UBJ_MAGIC
from app.ml.family_s import FamilySEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger("repackage")

# Snapshots to migrate: (id, engine_type, description)
# engine_type: "baseline" → XGBoostEngine, "family_s" → FamilySEngine, "twostage" → TwoStageEngine
TARGETS = [
    (4, "baseline", "v1.0.1-league-only (active baseline)"),
    (7, "family_s", "v2.0-tier3-family_s (Family S)"),
    (8, "twostage", "v1.0.2-twostage-w3 (Two-Stage W3)"),
]


async def main(dry_run: bool):
    db_url = os.environ.get("DATABASE_URL_ASYNC") or os.environ.get("DATABASE_URL")
    if not db_url:
        logger.error("DATABASE_URL_ASYNC or DATABASE_URL not set")
        sys.exit(1)

    if db_url.startswith("postgresql://"):
        db_url = db_url.replace("postgresql://", "postgresql+asyncpg://", 1)

    engine = create_async_engine(db_url, pool_size=2)
    Session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async with Session() as session:
        for snapshot_id, engine_type, desc in TARGETS:
            logger.info("─── Snapshot id=%d: %s ───", snapshot_id, desc)

            # Read current blob
            result = await session.execute(
                text("SELECT model_version, model_blob, LENGTH(model_blob) as size FROM model_snapshots WHERE id = :id"),
                {"id": snapshot_id},
            )
            row = result.first()
            if not row or not row.model_blob:
                logger.warning("  SKIP: no blob for id=%d", snapshot_id)
                continue

            old_blob = row.model_blob
            old_size = row.size
            model_version = row.model_version

            # Check if already UBJ
            try:
                decompressed = zlib.decompress(old_blob)
                if decompressed[:len(_UBJ_MAGIC)] == _UBJ_MAGIC:
                    logger.info("  ALREADY UBJ — skipping (size=%d)", old_size)
                    continue
                logger.info("  Current: pickle format, size=%d bytes", old_size)
            except Exception as e:
                logger.error("  ERROR decompressing: %s", e)
                continue

            # Load with legacy pickle path
            import numpy as np
            import pandas as pd

            if engine_type == "baseline":
                eng = XGBoostEngine(model_version=model_version)
                ok = eng.load_from_bytes(old_blob)
                if not ok:
                    logger.error("  FAILED to load pickle blob")
                    continue
                X_test = pd.DataFrame(np.zeros((1, 14)), columns=XGBoostEngine.FEATURE_COLUMNS)
                proba_before = eng.predict_proba(X_test)
                logger.info("  Loaded OK. Test predict: %s", [round(float(p), 4) for p in proba_before[0]])
                new_blob = eng.save_to_bytes()

            elif engine_type == "family_s":
                eng = FamilySEngine(model_version=model_version)
                ok = eng.load_from_bytes(old_blob)
                if not ok:
                    logger.error("  FAILED to load pickle blob")
                    continue
                X_test = pd.DataFrame(np.zeros((1, 24)), columns=FamilySEngine.FEATURE_COLUMNS)
                proba_before = eng.predict_proba(X_test)
                logger.info("  Loaded OK. Test predict: %s", [round(float(p), 4) for p in proba_before[0]])
                new_blob = eng.save_to_bytes()

            else:  # twostage
                eng = TwoStageEngine(model_version=model_version)
                ok = eng.load_from_bytes(old_blob)
                if not ok:
                    logger.error("  FAILED to load pickle blob")
                    continue
                feats = eng.active_stage1_features
                X_test = pd.DataFrame(np.zeros((1, len(feats))), columns=feats)
                proba_before = eng.stage1.predict_proba(X_test.values)
                logger.info("  Loaded OK. Test S1 predict: %s", [round(float(p), 4) for p in proba_before[0]])
                new_blob = eng.save_to_bytes()

            new_size = len(new_blob)

            # Verify UBJ magic
            new_decompressed = zlib.decompress(new_blob)
            assert new_decompressed[:len(_UBJ_MAGIC)] == _UBJ_MAGIC, "UBJ magic missing!"

            # Verify roundtrip: load NEW blob and check predictions match
            if engine_type == "baseline":
                eng2 = XGBoostEngine(model_version=model_version)
                ok2 = eng2.load_from_bytes(new_blob)
                assert ok2, "Failed to load UBJ blob"
                proba_after = eng2.predict_proba(X_test)
                max_diff = float(np.max(np.abs(proba_before - proba_after)))
                assert max_diff < 1e-6, f"Prediction mismatch! max_diff={max_diff}"

            elif engine_type == "family_s":
                eng2 = FamilySEngine(model_version=model_version)
                ok2 = eng2.load_from_bytes(new_blob)
                assert ok2, "Failed to load UBJ blob"
                proba_after = eng2.predict_proba(X_test)
                max_diff = float(np.max(np.abs(proba_before - proba_after)))
                assert max_diff < 1e-6, f"Prediction mismatch! max_diff={max_diff}"

            else:  # twostage
                eng2 = TwoStageEngine(model_version=model_version)
                ok2 = eng2.load_from_bytes(new_blob)
                assert ok2, "Failed to load UBJ blob"
                proba_after = eng2.stage1.predict_proba(X_test.values)
                max_diff = float(np.max(np.abs(proba_before - proba_after)))
                assert max_diff < 1e-6, f"S1 prediction mismatch! max_diff={max_diff}"

            logger.info("  Roundtrip verified: max_diff=%.2e", max_diff)
            logger.info("  Size: %d → %d bytes (%.1f%%)", old_size, new_size,
                        (1 - new_size / old_size) * 100 if old_size else 0)

            if dry_run:
                logger.info("  DRY-RUN: would update model_blob for id=%d", snapshot_id)
            else:
                await session.execute(
                    text("UPDATE model_snapshots SET model_blob = :blob WHERE id = :id"),
                    {"blob": new_blob, "id": snapshot_id},
                )
                await session.commit()
                logger.info("  UPDATED model_blob for id=%d ✓", snapshot_id)

    await engine.dispose()
    logger.info("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Re-package model snapshots from pickle to UBJ")
    parser.add_argument("--dry-run", action="store_true", help="Verify without writing to DB")
    args = parser.parse_args()

    asyncio.run(main(dry_run=args.dry_run))
