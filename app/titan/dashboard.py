"""TITAN Dashboard - Operational status endpoint.

Provides /dashboard/titan.json with:
- Extraction counts and coverage
- DLQ status
- PIT compliance metrics
- Feature matrix stats

Protected by X-Dashboard-Token (same auth as /dashboard/ops.json).
"""

import logging
from datetime import datetime, timedelta, timezone

from sqlalchemy import text


def _utc_now() -> datetime:
    """Get current UTC timestamp (timezone-aware) for TIMESTAMPTZ compatibility."""
    return datetime.now(timezone.utc)
from sqlalchemy.ext.asyncio import AsyncSession

from app.titan.config import get_titan_settings
from app.titan.jobs.job_manager import TitanJobManager
from app.titan.materializers.feature_matrix import FeatureMatrixMaterializer

logger = logging.getLogger(__name__)
titan_settings = get_titan_settings()


async def get_titan_status(session: AsyncSession) -> dict:
    """Get comprehensive TITAN operational status.

    Args:
        session: Database session

    Returns:
        Dict with all TITAN metrics for dashboard
    """
    schema = titan_settings.TITAN_SCHEMA
    now = _utc_now()

    # Initialize response
    status = {
        "timestamp": now.isoformat(),
        "schema": schema,
        "schema_exists": False,
        "tables": {},
        "extractions": {},
        "dlq": {},
        "feature_matrix": {},
        "pit_compliance": {},
    }

    try:
        # Check schema exists
        schema_check = await session.execute(text("""
            SELECT 1 FROM information_schema.schemata
            WHERE schema_name = :schema
        """), {"schema": schema})
        status["schema_exists"] = schema_check.scalar() is not None

        if not status["schema_exists"]:
            status["error"] = f"Schema '{schema}' does not exist. Run migrations first."
            return status

        # Check tables exist
        tables_check = await session.execute(text("""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = :schema
        """), {"schema": schema})
        existing_tables = [row[0] for row in tables_check.fetchall()]
        status["tables"] = {
            "raw_extractions": "raw_extractions" in existing_tables,
            "job_dlq": "job_dlq" in existing_tables,
            "feature_matrix": "feature_matrix" in existing_tables,
        }

        # Extraction stats (last 24h and total)
        if status["tables"]["raw_extractions"]:
            cutoff_24h = now - timedelta(hours=24)

            extractions_query = await session.execute(text(f"""
                SELECT
                    COUNT(*) as total,
                    COUNT(*) FILTER (WHERE created_at >= :cutoff) as last_24h,
                    COUNT(DISTINCT source_id) as sources,
                    MAX(captured_at) as last_captured,
                    COUNT(*) FILTER (WHERE http_status >= 200 AND http_status < 300) as success_count,
                    COUNT(*) FILTER (WHERE http_status >= 400) as error_count
                FROM {schema}.raw_extractions
            """), {"cutoff": cutoff_24h})
            row = extractions_query.fetchone()

            status["extractions"] = {
                "total": row[0] or 0,
                "last_24h": row[1] or 0,
                "sources": row[2] or 0,
                "last_captured_at": row[3].isoformat() if row[3] else None,
                "success_count": row[4] or 0,
                "error_count": row[5] or 0,
                "success_rate_pct": round((row[4] or 0) / row[0] * 100, 1) if row[0] else 0,
            }

            # Breakdown by source
            source_breakdown = await session.execute(text(f"""
                SELECT source_id, COUNT(*) as count
                FROM {schema}.raw_extractions
                WHERE created_at >= :cutoff
                GROUP BY source_id
                ORDER BY count DESC
            """), {"cutoff": cutoff_24h})
            status["extractions"]["by_source_24h"] = {
                row[0]: row[1] for row in source_breakdown.fetchall()
            }

        # DLQ stats
        if status["tables"]["job_dlq"]:
            job_manager = TitanJobManager(session)
            status["dlq"] = await job_manager.get_dlq_stats()

        # Feature matrix stats
        if status["tables"]["feature_matrix"]:
            materializer = FeatureMatrixMaterializer(session)
            fm_stats = await materializer.get_pit_stats()
            status["feature_matrix"] = fm_stats

            # PIT compliance summary
            status["pit_compliance"] = {
                "violations": fm_stats["pit_violations"],
                "compliant": fm_stats["pit_violations"] == 0,
                "total_rows": fm_stats["total_rows"],
            }

        # Overall health
        status["health"] = _compute_health(status)

    except Exception as e:
        logger.error(f"Error getting TITAN status: {e}")
        status["error"] = str(e)
        status["health"] = "error"

    return status


def _compute_health(status: dict) -> str:
    """Compute overall TITAN health status.

    Args:
        status: Full status dict

    Returns:
        'healthy', 'degraded', or 'unhealthy'
    """
    if not status.get("schema_exists"):
        return "unhealthy"

    # Check all tables exist
    tables = status.get("tables", {})
    if not all(tables.values()):
        return "unhealthy"

    # Check for PIT violations
    pit = status.get("pit_compliance", {})
    if pit.get("violations", 0) > 0:
        return "unhealthy"

    # Check DLQ health
    dlq = status.get("dlq", {})
    pending = dlq.get("pending_count", 0)
    exhausted = dlq.get("exhausted_count", 0)

    if exhausted > 0:
        return "degraded"
    if pending > 10:
        return "degraded"

    return "healthy"


async def get_titan_summary(session: AsyncSession) -> dict:
    """Get minimal TITAN summary for ops.json integration.

    Args:
        session: Database session

    Returns:
        Minimal dict with key health indicators
    """
    schema = titan_settings.TITAN_SCHEMA

    try:
        # Quick existence check
        schema_check = await session.execute(text("""
            SELECT 1 FROM information_schema.schemata
            WHERE schema_name = :schema
        """), {"schema": schema})

        if not schema_check.scalar():
            return {
                "status": "not_deployed",
                "schema": schema,
            }

        # Quick stats
        stats = await session.execute(text(f"""
            SELECT
                (SELECT COUNT(*) FROM {schema}.raw_extractions) as extractions,
                (SELECT COUNT(*) FROM {schema}.job_dlq WHERE resolved_at IS NULL) as dlq_pending,
                (SELECT COUNT(*) FROM {schema}.feature_matrix) as feature_rows,
                (SELECT COUNT(*) FROM {schema}.feature_matrix WHERE pit_max_captured_at >= kickoff_utc) as pit_violations
        """))
        row = stats.fetchone()

        health = "healthy"
        if row[3] > 0:  # PIT violations
            health = "unhealthy"
        elif row[1] > 10:  # DLQ pending
            health = "degraded"

        return {
            "status": health,
            "schema": schema,
            "extractions_total": row[0] or 0,
            "dlq_pending": row[1] or 0,
            "feature_rows": row[2] or 0,
            "pit_violations": row[3] or 0,
        }

    except Exception as e:
        return {
            "status": "error",
            "schema": schema,
            "error": str(e),
        }
