"""Job run tracking for ops dashboard.

Provides functions to record job executions in DB for fallback
when Prometheus metrics are unavailable (cold-start after deploy).

Usage:
    from app.jobs.tracking import record_job_run

    # In scheduler job:
    start = datetime.now(timezone.utc)
    try:
        # ... job logic ...
        await record_job_run(session, "stats_backfill", "ok", start, metrics={"rows": 5})
    except Exception as e:
        await record_job_run(session, "stats_backfill", "error", start, error=str(e))
        raise
"""

import logging
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import JobRun

logger = logging.getLogger(__name__)


async def record_job_run(
    session: AsyncSession,
    job_name: str,
    status: str,
    started_at: datetime,
    error: Optional[str] = None,
    metrics: Optional[dict] = None,
) -> None:
    """
    Record a job execution in the database.

    Args:
        session: Database session.
        job_name: Job identifier (stats_backfill, odds_sync, fastpath, etc.).
        status: Execution status (ok, error, rate_limited, budget_exceeded).
        started_at: When the job started.
        error: Error message if failed.
        metrics: Optional job-specific metrics dict.
    """
    finished_at = datetime.now(timezone.utc)
    duration_ms = int((finished_at - started_at).total_seconds() * 1000)

    job_run = JobRun(
        job_name=job_name,
        status=status,
        started_at=started_at,
        finished_at=finished_at,
        duration_ms=duration_ms,
        error_message=error,
        metrics=metrics,
    )

    session.add(job_run)
    await session.commit()

    logger.debug(f"[JOB_TRACKING] Recorded {job_name} run: {status} in {duration_ms}ms")


async def get_last_success_at(
    session: AsyncSession,
    job_name: str,
) -> Optional[datetime]:
    """
    Get the last successful run timestamp for a job from DB.

    Args:
        session: Database session.
        job_name: Job identifier.

    Returns:
        Datetime of last successful run, or None if no successful runs.
    """
    result = await session.execute(
        select(JobRun.finished_at)
        .where(JobRun.job_name == job_name)
        .where(JobRun.status == "ok")
        .order_by(JobRun.finished_at.desc())
        .limit(1)
    )
    row = result.first()
    return row[0] if row else None


async def get_jobs_health_from_db(session: AsyncSession) -> dict:
    """
    Get jobs health data from DB for all tracked jobs.

    Returns dict mapping job_name -> {last_success_at, last_status, ...}.
    Used as fallback when Prometheus metrics are unavailable.
    """
    # Get last run for each job (regardless of status)
    result = await session.execute(text("""
        SELECT DISTINCT ON (job_name)
            job_name,
            status,
            finished_at,
            duration_ms,
            error_message
        FROM job_runs
        ORDER BY job_name, finished_at DESC
    """))
    rows = result.fetchall()

    # Get last SUCCESS for each job
    result_success = await session.execute(text("""
        SELECT DISTINCT ON (job_name)
            job_name,
            finished_at as last_success_at
        FROM job_runs
        WHERE status = 'ok'
        ORDER BY job_name, finished_at DESC
    """))
    success_rows = result_success.fetchall()
    success_map = {row[0]: row[1] for row in success_rows}

    jobs_data = {}
    for row in rows:
        job_name = row[0]
        jobs_data[job_name] = {
            "last_run_status": row[1],
            "last_run_at": row[2].isoformat() if row[2] else None,
            "last_success_at": success_map.get(job_name).isoformat() if success_map.get(job_name) else None,
            "duration_ms": row[3],
            "last_error": row[4] if row[1] == "error" else None,
        }

    return jobs_data


async def cleanup_old_runs(session: AsyncSession, days_to_keep: int = 7) -> int:
    """
    Delete job runs older than specified days.

    Args:
        session: Database session.
        days_to_keep: Keep runs from the last N days.

    Returns:
        Number of rows deleted.
    """
    result = await session.execute(text(f"""
        DELETE FROM job_runs
        WHERE created_at < NOW() - INTERVAL '{days_to_keep} days'
    """))
    await session.commit()
    deleted = result.rowcount or 0
    if deleted > 0:
        logger.info(f"[JOB_TRACKING] Cleaned up {deleted} old job runs")
    return deleted
