#!/usr/bin/env python3
"""
Upload a trained model file to PostgreSQL model_snapshots.

Usage:
    python scripts/upload_model_to_db.py --model-path models/xgb_v1.0.1-league-only_20260201.json

This will:
1. Load the model from the JSON file
2. Compress it with pickle+zlib
3. Insert into model_snapshots with is_active=true
4. Deactivate previous models
"""

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

from app.ml.engine import XGBoostEngine
from app.ml.persistence import persist_model_snapshot

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def upload_model(model_path: str, brier_score: float = 0.21):
    """Upload model file to database."""

    # Setup database connection
    database_url = os.environ.get('DATABASE_URL')
    if not database_url:
        raise ValueError("DATABASE_URL not set")

    if database_url.startswith('postgresql://'):
        database_url = database_url.replace('postgresql://', 'postgresql+asyncpg://')

    engine = create_async_engine(database_url, echo=False)
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    # Load model from file
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # Extract version from filename (e.g., xgb_v1.0.1-league-only_20260201.json)
    filename = path.stem  # Remove .json
    parts = filename.split('_')
    if len(parts) >= 2:
        model_version = parts[1]  # v1.0.1-league-only
    else:
        model_version = "unknown"

    logger.info(f"Loading model from {model_path}")
    logger.info(f"Detected version: {model_version}")

    # Create engine and load model
    ml_engine = XGBoostEngine(model_version=model_version)
    if not ml_engine.load_model(str(path)):
        raise RuntimeError(f"Failed to load model from {model_path}")

    logger.info(f"Model loaded: {ml_engine.model.n_features_in_} features")

    # Save to database
    async with async_session() as session:
        snapshot_id = await persist_model_snapshot(
            session=session,
            engine=ml_engine,
            brier_score=brier_score,
            cv_scores=[brier_score],  # Placeholder
            samples_trained=20233,  # From training logs
            training_config={
                "source": "file_upload",
                "original_path": str(path),
                "league_only": True,
            },
            is_baseline=False,
        )

        logger.info(f"Model uploaded to database: snapshot_id={snapshot_id}")
        logger.info(f"Version: {model_version}")
        logger.info(f"Now active in production!")

    await engine.dispose()
    return snapshot_id


def main():
    parser = argparse.ArgumentParser(description="Upload model to PostgreSQL")
    parser.add_argument('--model-path', required=True, help='Path to model JSON file')
    parser.add_argument('--brier-score', type=float, default=0.21, help='Brier score for metadata')
    args = parser.parse_args()

    asyncio.run(upload_model(args.model_path, args.brier_score))


if __name__ == "__main__":
    main()
