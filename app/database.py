"""Async database connection using SQLAlchemy (supports SQLite and PostgreSQL)."""

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from sqlalchemy.exc import InterfaceError, InvalidRequestError, OperationalError
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from sqlmodel import SQLModel

from app.config import get_settings

logger = logging.getLogger(__name__)

settings = get_settings()


def get_database_url() -> str:
    """Convert database URL to async format."""
    url = settings.DATABASE_URL

    # SQLite
    if url.startswith("sqlite"):
        return url.replace("sqlite://", "sqlite+aiosqlite://", 1)

    # PostgreSQL
    if url.startswith("postgres://"):
        return url.replace("postgres://", "postgresql+asyncpg://", 1)
    if url.startswith("postgresql://"):
        return url.replace("postgresql://", "postgresql+asyncpg://", 1)

    return url


DATABASE_URL = get_database_url()
is_sqlite = DATABASE_URL.startswith("sqlite")

# Engine configuration
engine_kwargs = {
    "echo": False,
}

if is_sqlite:
    # SQLite-specific settings
    engine_kwargs["connect_args"] = {"check_same_thread": False}
    engine_kwargs["poolclass"] = StaticPool
else:
    # PostgreSQL-specific settings
    engine_kwargs["pool_pre_ping"] = True  # Verify connection before checkout
    engine_kwargs["pool_size"] = 10  # Increased from 5 - base connections kept open
    engine_kwargs["max_overflow"] = 20  # Increased from 10 - temporary connections when pool is full
    engine_kwargs["pool_recycle"] = 300  # Recycle connections every 5 min (Railway can drop idle connections)
    engine_kwargs["pool_timeout"] = 30  # Timeout waiting for connection from pool
    # Return connections to pool in clean state (prevents leaks)
    engine_kwargs["pool_reset_on_return"] = "rollback"
    # Statement timeout: kill queries that run longer than 60s (prevents connection hogging)
    engine_kwargs["connect_args"] = {
        "server_settings": {"statement_timeout": "60000"}  # 60 seconds in milliseconds
    }

async_engine = create_async_engine(DATABASE_URL, **engine_kwargs)

AsyncSessionLocal = sessionmaker(
    bind=async_engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)

# Alias for background tasks that need to create their own sessions
async_session_maker = AsyncSessionLocal


async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    """Dependency for getting async database sessions."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()


async def init_db() -> None:
    """Initialize database tables."""
    logger.info("Initializing database tables...")
    async with async_engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
    logger.info("Database tables created successfully.")


async def close_db() -> None:
    """Close database connections."""
    logger.info("Closing database connections...")
    await async_engine.dispose()
    logger.info("Database connections closed.")


def get_pool_status() -> dict:
    """Get current connection pool statistics for monitoring."""
    if is_sqlite:
        return {"type": "sqlite", "pooled": False}

    pool = async_engine.pool
    checked_out = pool.checkedout()
    total_capacity = pool.size() + pool.overflow()
    return {
        "type": "postgresql",
        "pool_size": pool.size(),
        "checked_in": pool.checkedin(),
        "checked_out": checked_out,
        "overflow": pool.overflow(),
        # Utilization percentage
        "utilization_pct": round(
            (checked_out / total_capacity) * 100, 1
        ) if total_capacity > 0 else 0,
    }


@asynccontextmanager
async def get_session_with_retry(max_retries: int = 3, retry_delay: float = 1.0):
    """
    Context manager that provides a session with automatic retry on connection errors.

    Use this for scheduled jobs that may encounter stale connections after Railway
    restarts or network interruptions.

    Example:
        async with get_session_with_retry() as session:
            result = await session.execute(...)
            await session.commit()

    Note: Uses @asynccontextmanager decorator to properly handle generator cleanup
    and avoid "generator didn't stop after athrow()" errors when connections drop.

    IMPORTANT: Retries only happen on session CREATION failure. If a connection drops
    DURING execution, the exception propagates to the caller. For mid-execution failures,
    the caller should wrap their logic in a retry loop.
    """
    last_error = None
    current_delay = retry_delay
    session = None

    for attempt in range(max_retries):
        try:
            session = AsyncSessionLocal()
            # Test the connection is alive before yielding
            await session.connection()
            break  # Connection successful, exit retry loop
        except (InterfaceError, OperationalError, InvalidRequestError) as e:
            last_error = e
            error_msg = str(e)

            if session is not None:
                try:
                    await session.close()
                except Exception:
                    pass
                session = None

            # MissingGreenlet during pool_pre_ping on stale Railway connections
            is_greenlet = "greenlet" in error_msg.lower()
            # Check if this is a connection-closed error worth retrying
            if is_greenlet or "closed" in error_msg.lower() or "connection" in error_msg.lower() or "terminated" in error_msg.lower():
                if attempt < max_retries - 1:
                    logger.warning(
                        f"Database connection error (attempt {attempt + 1}/{max_retries}): {e}. "
                        f"Retrying in {current_delay}s..."
                    )
                    await asyncio.sleep(current_delay)
                    current_delay *= 2  # Exponential backoff
                    continue

            # Non-retryable error or max retries reached
            raise

    # All retries exhausted without success
    if session is None:
        if last_error:
            raise last_error
        raise RuntimeError("Failed to create database session after retries")

    # Now yield the session - this is a single yield point
    try:
        yield session
    finally:
        # Ensure session cleanup even if connection dropped mid-execution
        if session is not None:
            try:
                await session.close()
            except Exception as close_error:
                # Log but don't raise - we want cleanup to be best-effort
                logger.debug(f"Error closing session during cleanup: {close_error}")
