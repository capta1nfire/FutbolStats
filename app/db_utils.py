"""Database utility functions for cross-database compatibility."""

import logging
from typing import Any, TypeVar

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)

T = TypeVar("T")


async def upsert(
    session: AsyncSession,
    model: type[T],
    values: dict[str, Any],
    conflict_columns: list[str],
    update_columns: list[str] | None = None,
) -> T | None:
    """
    Database-agnostic upsert operation.

    Attempts PostgreSQL dialect first (optimal for production),
    falls back to SELECT + INSERT/UPDATE for compatibility with SQLite/MySQL.

    Args:
        session: AsyncSession instance
        model: SQLAlchemy model class
        values: Dictionary of column values to insert/update
        conflict_columns: Columns that define uniqueness (for conflict detection)
        update_columns: Columns to update on conflict (defaults to all non-conflict columns)

    Returns:
        The upserted model instance, or None if failed

    Example:
        await upsert(
            session,
            Prediction,
            {"match_id": 123, "model_version": "v1", "home_prob": 0.5},
            conflict_columns=["match_id", "model_version"],
            update_columns=["home_prob"]
        )
    """
    if update_columns is None:
        update_columns = [k for k in values.keys() if k not in conflict_columns]

    try:
        # Try PostgreSQL-specific upsert first (most efficient)
        from sqlalchemy.dialects.postgresql import insert as pg_insert

        stmt = pg_insert(model).values(**values)

        if update_columns:
            update_dict = {col: getattr(stmt.excluded, col) for col in update_columns}
            stmt = stmt.on_conflict_do_update(
                index_elements=conflict_columns,
                set_=update_dict,
            )
        else:
            stmt = stmt.on_conflict_do_nothing(index_elements=conflict_columns)

        await session.execute(stmt)
        return None  # PostgreSQL upsert doesn't return the object easily

    except Exception as pg_error:
        # Fall back to generic SELECT + INSERT/UPDATE
        logger.debug(f"PostgreSQL upsert failed, using fallback: {pg_error}")

        # Build filter for conflict columns
        filters = [getattr(model, col) == values[col] for col in conflict_columns]
        query = select(model).where(*filters)
        result = await session.execute(query)
        existing = result.scalar_one_or_none()

        if existing:
            # Update existing record
            for col in update_columns:
                if col in values:
                    setattr(existing, col, values[col])
            return existing
        else:
            # Insert new record
            new_instance = model(**values)
            session.add(new_instance)
            return new_instance


async def bulk_upsert(
    session: AsyncSession,
    model: type[T],
    values_list: list[dict[str, Any]],
    conflict_columns: list[str],
    update_columns: list[str] | None = None,
) -> int:
    """
    Bulk upsert operation for multiple records.

    Args:
        session: AsyncSession instance
        model: SQLAlchemy model class
        values_list: List of dictionaries with column values
        conflict_columns: Columns that define uniqueness
        update_columns: Columns to update on conflict

    Returns:
        Number of records processed
    """
    count = 0
    for values in values_list:
        try:
            await upsert(session, model, values, conflict_columns, update_columns)
            count += 1
        except Exception as e:
            logger.warning(f"Failed to upsert record: {e}")

    return count
