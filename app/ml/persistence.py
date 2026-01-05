"""Model persistence functions for PostgreSQL storage."""

import logging
from typing import Optional

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import ModelSnapshot
from app.ml.engine import XGBoostEngine

logger = logging.getLogger(__name__)


async def persist_model_snapshot(
    session: AsyncSession,
    engine: XGBoostEngine,
    brier_score: float,
    cv_scores: list[float],
    samples_trained: int,
    training_config: Optional[dict] = None,
    is_baseline: bool = False,
) -> int:
    """
    Save a trained model to the database.

    Deactivates any previously active snapshots and saves the new one.

    Args:
        session: Async database session.
        engine: The XGBoostEngine with the trained model.
        brier_score: Cross-validation Brier score.
        cv_scores: Per-fold CV scores.
        samples_trained: Number of training samples.
        training_config: Optional hyperparameters used.
        is_baseline: Whether this is the baseline reference model.

    Returns:
        The ID of the created snapshot.
    """
    # Get model bytes
    model_blob = engine.save_to_bytes()

    # Deactivate all currently active snapshots
    await session.execute(
        update(ModelSnapshot).where(ModelSnapshot.is_active == True).values(is_active=False)
    )

    # Create new snapshot
    snapshot = ModelSnapshot(
        model_version=engine.model_version,
        model_blob=model_blob,
        model_path="db_stored",  # Legacy field placeholder
        brier_score=brier_score,
        cv_brier_scores=cv_scores,
        samples_trained=samples_trained,
        training_config=training_config,
        is_active=True,
        is_baseline=is_baseline,
    )
    session.add(snapshot)
    await session.commit()
    await session.refresh(snapshot)

    logger.info(
        f"Model snapshot saved: version={engine.model_version}, "
        f"brier={brier_score:.4f}, size={len(model_blob)} bytes, id={snapshot.id}"
    )

    return snapshot.id


async def load_active_model(session: AsyncSession, engine: XGBoostEngine) -> bool:
    """
    Load the active model from database into the engine.

    Args:
        session: Async database session.
        engine: The XGBoostEngine to load the model into.

    Returns:
        True if a model was found and loaded, False otherwise.
    """
    result = await session.execute(
        select(ModelSnapshot)
        .where(ModelSnapshot.is_active == True)
        .order_by(ModelSnapshot.created_at.desc())
        .limit(1)
    )
    snapshot = result.scalar_one_or_none()

    if not snapshot:
        logger.warning("No active model snapshot found in database")
        return False

    if not snapshot.model_blob:
        logger.warning(f"Snapshot {snapshot.id} has no model_blob data")
        return False

    success = engine.load_from_bytes(snapshot.model_blob)

    if success:
        engine.model_version = snapshot.model_version
        logger.info(
            f"Loaded model from DB: version={snapshot.model_version}, "
            f"brier={snapshot.brier_score:.4f}, created={snapshot.created_at}"
        )
    else:
        logger.error(f"Failed to load model from snapshot {snapshot.id}")

    return success


async def activate_snapshot(session: AsyncSession, snapshot_id: int) -> bool:
    """
    Activate a specific model snapshot (for rollback).

    Deactivates all other snapshots and activates the specified one.

    Args:
        session: Async database session.
        snapshot_id: ID of the snapshot to activate.

    Returns:
        True if the snapshot was found and activated, False otherwise.
    """
    # Deactivate all snapshots
    await session.execute(
        update(ModelSnapshot).values(is_active=False)
    )

    # Activate the specified snapshot
    result = await session.execute(
        update(ModelSnapshot)
        .where(ModelSnapshot.id == snapshot_id)
        .values(is_active=True)
        .returning(ModelSnapshot.id)
    )
    activated_id = result.scalar_one_or_none()

    if activated_id is not None:
        await session.commit()
        logger.info(f"Activated model snapshot {snapshot_id}")
        return True
    else:
        await session.rollback()
        logger.error(f"Snapshot {snapshot_id} not found")
        return False


async def get_latest_snapshot(session: AsyncSession) -> Optional[ModelSnapshot]:
    """
    Get the most recent model snapshot (regardless of active status).

    Args:
        session: Async database session.

    Returns:
        The latest ModelSnapshot or None.
    """
    result = await session.execute(
        select(ModelSnapshot)
        .order_by(ModelSnapshot.created_at.desc())
        .limit(1)
    )
    return result.scalar_one_or_none()
