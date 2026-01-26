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
        "r2_storage": {},
        "dlq": {},
        "feature_matrix": {},
        "pit_compliance": {},
        "sofascore": {},
        "xi_depth": {},
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

        # DLQ stats and R2 storage stats
        if status["tables"]["job_dlq"]:
            job_manager = TitanJobManager(session)
            status["dlq"] = await job_manager.get_dlq_stats()

        # R2 storage stats
        if status["tables"]["raw_extractions"]:
            job_manager = TitanJobManager(session)
            status["r2_storage"] = await job_manager.get_r2_stats()

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

            # SofaScore lineup metrics (Tier 1c)
            status["sofascore"] = await _get_sofascore_metrics(session, schema, now)

            # XI Depth metrics (Tier 1d)
            status["xi_depth"] = await _get_xi_depth_metrics(session, schema, now)

        # Overall health
        status["health"] = _compute_health(status)

    except Exception as e:
        logger.error(f"Error getting TITAN status: {e}")
        status["error"] = str(e)
        status["health"] = "error"

    return status


async def _get_sofascore_metrics(session: AsyncSession, schema: str, now: datetime) -> dict:
    """Get SofaScore lineup metrics for Tier 1c.

    Args:
        session: Database session
        schema: TITAN schema name
        now: Current UTC timestamp

    Returns:
        Dict with SofaScore lineup metrics
    """
    metrics = {
        "ref_coverage_pct": 0.0,
        "lineup_coverage_pct_given_ref": 0.0,
        "lineup_freshness_hours": None,
    }

    try:
        # 1. ref_coverage_pct: refs / matches in target window (MVP leagues)
        ref_coverage_query = await session.execute(text("""
            SELECT
                COUNT(DISTINCT m.id) as total_matches,
                COUNT(DISTINCT mer.match_id) as matches_with_ref
            FROM matches m
            LEFT JOIN match_external_refs mer
                ON m.id = mer.match_id AND mer.source = 'sofascore'
            WHERE m.date > NOW() - INTERVAL '7 days'
              AND m.date < NOW() + INTERVAL '7 days'
              AND m.league_id IN (140, 39, 135)
        """))
        ref_row = ref_coverage_query.fetchone()
        total_matches = ref_row[0] or 0
        matches_with_ref = ref_row[1] or 0
        metrics["ref_coverage_pct"] = round(
            (matches_with_ref / total_matches * 100) if total_matches > 0 else 0, 1
        )

        # 2. lineup_coverage_pct_given_ref: lineups / refs (last 7 days)
        lineup_given_ref_query = await session.execute(text("""
            SELECT
                COUNT(DISTINCT mer.match_id) as matches_with_ref,
                COUNT(DISTINCT msl.match_id) as matches_with_lineup
            FROM match_external_refs mer
            LEFT JOIN match_sofascore_lineup msl
                ON mer.match_id = msl.match_id
            WHERE mer.source = 'sofascore'
              AND mer.created_at > NOW() - INTERVAL '7 days'
        """))
        lineup_row = lineup_given_ref_query.fetchone()
        refs_count = lineup_row[0] or 0
        lineups_count = lineup_row[1] or 0
        metrics["lineup_coverage_pct_given_ref"] = round(
            (lineups_count / refs_count * 100) if refs_count > 0 else 0, 1
        )

        # 3. lineup_freshness_hours: time since last lineup capture in titan.feature_matrix
        freshness_query = await session.execute(text(f"""
            SELECT MAX(sofascore_lineup_captured_at) as latest
            FROM {schema}.feature_matrix
            WHERE tier1c_complete = TRUE
        """))
        freshness_row = freshness_query.fetchone()
        if freshness_row and freshness_row[0]:
            latest = freshness_row[0]
            # Ensure latest is timezone-aware
            if latest.tzinfo is None:
                latest = latest.replace(tzinfo=timezone.utc)
            age_hours = (now - latest).total_seconds() / 3600
            metrics["lineup_freshness_hours"] = round(age_hours, 1)

    except Exception as e:
        logger.warning(f"Error getting SofaScore metrics: {e}")
        metrics["error"] = str(e)

    return metrics


async def _get_xi_depth_metrics(session: AsyncSession, schema: str, now: datetime) -> dict:
    """Get XI Depth metrics for Tier 1d.

    Args:
        session: Database session
        schema: TITAN schema name
        now: Current UTC timestamp

    Returns:
        Dict with XI depth metrics
    """
    metrics = {
        "formation_mismatch_rate_pct": 0.0,
    }

    try:
        # Formation mismatch rate (mismatches / tier1d_complete rows)
        mismatch_query = await session.execute(text(f"""
            SELECT
                COUNT(*) FILTER (WHERE xi_formation_mismatch_flag = TRUE) as mismatches,
                COUNT(*) FILTER (WHERE tier1d_complete = TRUE) as total
            FROM {schema}.feature_matrix
            WHERE kickoff_utc > NOW() - INTERVAL '7 days'
        """))
        row = mismatch_query.fetchone()
        mismatches = row[0] or 0
        total = row[1] or 0
        metrics["formation_mismatch_rate_pct"] = round(
            (mismatches / total * 100) if total > 0 else 0, 1
        )

    except Exception as e:
        logger.warning(f"Error getting XI depth metrics: {e}")
        metrics["error"] = str(e)

    return metrics


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
