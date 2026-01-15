#!/usr/bin/env python3
"""
Train and persist the two-stage model for shadow evaluation.

Usage:
    python scripts/train_twostage.py

This script:
1. Loads training data from production DB
2. Trains the TwoStageEngine
3. Persists the model to model_snapshots table
4. Reports metrics

Run this BEFORE enabling shadow mode in production.
"""

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
os.environ["DATABASE_URL"] = "postgresql://postgres:hzvozcXijUpblVrQshuowYcEGwZnMrfO@maglev.proxy.rlwy.net:24997/railway"

from sqlalchemy import update

from app.database import get_async_session
from app.features.engineering import FeatureEngineer
from app.ml.engine import TwoStageEngine
from app.models import ModelSnapshot


async def main():
    print("=" * 80)
    print("TRAINING TWO-STAGE MODEL FOR SHADOW MODE")
    print("=" * 80)

    # Load data
    async for session in get_async_session():
        fe = FeatureEngineer(session)
        print("\nLoading training dataset...")
        df = await fe.build_training_dataset()

        print(f"Dataset: {len(df)} samples")
        print(f"Class distribution: {df['result'].value_counts().to_dict()}")
        print(f"Draw rate: {(df['result'] == 1).mean():.1%}")

        # Sort by date for proper time series split
        df = df.sort_values("date").reset_index(drop=True)

        # Train two-stage model
        print("\n" + "-" * 80)
        print("Training TwoStageEngine...")
        print("-" * 80)

        engine = TwoStageEngine(model_version="v1.1.0-twostage", draw_weight=1.2)
        metrics = engine.train(df, n_splits=3)

        print(f"\nTraining complete:")
        print(f"  Version: {metrics['model_version']}")
        print(f"  Architecture: {metrics['architecture']}")
        print(f"  Samples: {metrics['samples_trained']}")
        print(f"  Brier Score: {metrics['brier_score']:.4f}")
        print(f"  CV Scores: {metrics['cv_scores']}")

        # Persist to database
        print("\n" + "-" * 80)
        print("Persisting to database...")
        print("-" * 80)

        model_blob = engine.save_to_bytes()
        print(f"Model blob size: {len(model_blob)} bytes")

        # Create snapshot (NOT active - this is shadow model)
        snapshot = ModelSnapshot(
            model_version=engine.model_version,
            model_blob=model_blob,
            model_path="db_stored",
            brier_score=metrics["brier_score"],
            cv_brier_scores=metrics["cv_scores"],
            samples_trained=metrics["samples_trained"],
            training_config={
                "architecture": "two_stage",
                "draw_weight": engine.draw_weight,
                "stage1_features": len(engine.STAGE1_FEATURES),
                "stage2_features": len(engine.STAGE2_FEATURES),
            },
            is_active=False,  # Shadow model, not active
            is_baseline=False,
        )
        session.add(snapshot)
        await session.commit()
        await session.refresh(snapshot)

        print(f"Snapshot saved: id={snapshot.id}")

        # Verify load
        print("\n" + "-" * 80)
        print("Verifying model load...")
        print("-" * 80)

        engine2 = TwoStageEngine()
        if engine2.load_from_bytes(model_blob):
            print("✓ Model loads successfully from bytes")

            # Quick sanity check
            import pandas as pd
            test_row = df.iloc[-1:].copy()
            proba = engine2.predict_proba(test_row)
            print(f"✓ Prediction test: {proba[0]}")
            print(f"  Sum: {proba[0].sum():.6f} (should be 1.0)")
        else:
            print("✗ Model load failed!")

        print("\n" + "=" * 80)
        print("DONE")
        print("=" * 80)
        print(f"""
To enable shadow mode in production:
1. Set MODEL_SHADOW_ARCHITECTURE=two_stage in Railway
2. The shadow_predictions table will be auto-created
3. Shadow predictions will be logged alongside baseline
4. Use /admin/shadow-report endpoint to monitor

Model snapshot ID: {snapshot.id}
Version: {snapshot.model_version}
Brier: {snapshot.brier_score:.4f}
""")

        break


if __name__ == "__main__":
    asyncio.run(main())
