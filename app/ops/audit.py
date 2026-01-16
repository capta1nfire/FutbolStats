"""OPS Audit logging utilities.

Provides functions to log dashboard actions for accountability and debugging.
"""

import hashlib
import logging
import time
import uuid
from datetime import datetime
from typing import Optional

from fastapi import Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import OpsAuditLog

logger = logging.getLogger(__name__)


def _get_actor_id(token: str) -> str:
    """Generate a short hash of a token for actor identification."""
    if not token:
        return "unknown"
    # Take first 8 chars of MD5 hash for readability
    return hashlib.md5(token.encode()).hexdigest()[:8]


def _extract_token(request: Request) -> Optional[str]:
    """Extract authentication token from request."""
    # Try header first
    token = request.headers.get("x-dashboard-token")
    if token:
        return token

    # Try query param
    token = request.query_params.get("token")
    if token:
        return token

    # Try API key
    token = request.headers.get("x-api-key")
    if token:
        return token

    return None


def _get_client_ip(request: Request) -> str:
    """Extract client IP, handling proxies."""
    # Check X-Forwarded-For header (Railway/reverse proxy)
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        # Take first IP in chain
        return forwarded.split(",")[0].strip()

    # Check X-Real-IP
    real_ip = request.headers.get("x-real-ip")
    if real_ip:
        return real_ip

    # Fallback to client host
    if request.client:
        return request.client.host

    return "unknown"


async def log_ops_action(
    session: AsyncSession,
    request: Request,
    action: str,
    params: Optional[dict] = None,
    result: str = "ok",
    result_detail: Optional[dict] = None,
    error_message: Optional[str] = None,
    duration_ms: Optional[int] = None,
) -> OpsAuditLog:
    """
    Log an OPS dashboard action to the audit table.

    Args:
        session: Database session
        request: FastAPI request object
        action: Action type (e.g., 'predictions_trigger', 'odds_sync')
        params: Action parameters (optional)
        result: Result status ('ok', 'error', 'rejected')
        result_detail: Detailed result data (optional)
        error_message: Error message if result='error' (optional)
        duration_ms: Action duration in milliseconds (optional)

    Returns:
        The created OpsAuditLog record
    """
    token = _extract_token(request)

    # Determine actor type
    if request.headers.get("x-dashboard-token"):
        actor = "dashboard_token"
    elif request.headers.get("x-api-key"):
        actor = "api_key"
    else:
        actor = "unknown"

    audit = OpsAuditLog(
        action=action,
        request_id=str(uuid.uuid4()),
        actor=actor,
        actor_id=_get_actor_id(token),
        ip_address=_get_client_ip(request),
        user_agent=request.headers.get("user-agent", "")[:500],
        params=params,
        result=result,
        result_detail=result_detail,
        error_message=error_message[:500] if error_message else None,
        duration_ms=duration_ms,
    )

    session.add(audit)
    await session.commit()

    logger.info(
        f"[OPS_AUDIT] action={action} actor_id={audit.actor_id} "
        f"result={result} duration_ms={duration_ms}"
    )

    return audit


class OpsActionTimer:
    """Context manager for timing OPS actions."""

    def __init__(self):
        self.start_time = None
        self.duration_ms = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.duration_ms = int((time.time() - self.start_time) * 1000)
        return False


async def get_recent_audit_logs(
    session: AsyncSession,
    limit: int = 20
) -> list[dict]:
    """
    Get recent OPS audit log entries for dashboard display.

    Args:
        session: Database session
        limit: Maximum number of entries to return

    Returns:
        List of audit log entries as dicts
    """
    from sqlalchemy import select, desc

    stmt = (
        select(OpsAuditLog)
        .order_by(desc(OpsAuditLog.created_at))
        .limit(limit)
    )

    result = await session.execute(stmt)
    logs = result.scalars().all()

    return [
        {
            "id": log.id,
            "action": log.action,
            "actor": log.actor,
            "actor_id": log.actor_id,
            "params": log.params,
            "result": log.result,
            "result_summary": _summarize_result(log.result_detail),
            "error_message": log.error_message,
            "created_at": log.created_at.isoformat() if log.created_at else None,
            "duration_ms": log.duration_ms,
        }
        for log in logs
    ]


def _summarize_result(result_detail: Optional[dict]) -> str:
    """Create a short summary of result_detail for display."""
    if not result_detail:
        return ""

    # Common patterns
    if "predictions_saved" in result_detail:
        return f"saved={result_detail['predictions_saved']}"
    if "updated" in result_detail:
        return f"updated={result_detail['updated']}"
    if "matches_synced" in result_detail:
        return f"synced={result_detail['matches_synced']}"
    if "scanned" in result_detail:
        return f"scanned={result_detail['scanned']}, updated={result_detail.get('updated', 0)}"

    # Fallback: first key-value
    for k, v in result_detail.items():
        if isinstance(v, (int, str, bool)):
            return f"{k}={v}"

    return ""
