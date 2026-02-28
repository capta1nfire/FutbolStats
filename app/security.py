"""Security middleware: Rate limiting, API key, and dashboard token authentication."""

import logging
import os
from typing import Optional

from fastapi import Header, HTTPException, Request, Security
from fastapi.security import APIKeyHeader
from slowapi import Limiter
from slowapi.util import get_remote_address

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# Detect production environment (Railway sets RAILWAY_ENVIRONMENT)
IS_PRODUCTION = os.getenv("RAILWAY_ENVIRONMENT") == "production" or os.getenv("RAILWAY_PROJECT_ID") is not None

# Rate limiter using client IP
limiter = Limiter(key_func=get_remote_address)

# API Key header for admin endpoints
api_key_header = APIKeyHeader(name=settings.API_KEY_HEADER, auto_error=False)


def _has_valid_ops_session(request: Request) -> bool:
    """
    Check if request has a valid OPS session cookie.

    This enables browser UX for INTERNAL endpoints (e.g., /dashboard/ops.json)
    without requiring manual X-API-Key header injection.
    """
    try:
        session = getattr(request, "session", None)
        if not session or not session.get("ops_authenticated"):
            return False

        issued_at = session.get("issued_at")
        if issued_at:
            from datetime import datetime, timedelta, timezone

            issued = datetime.fromisoformat(issued_at)
            ttl_hours = settings.OPS_SESSION_TTL_HOURS
            if datetime.now(timezone.utc) - issued > timedelta(hours=ttl_hours):
                return False

        return True
    except Exception:
        return False


async def verify_api_key_or_ops_session(
    request: Request,
    api_key: Optional[str] = Security(api_key_header),
) -> bool:
    """
    Verify internal access via either:
    - Valid OPS session cookie (browser)
    - X-API-Key header (automation/scripts)
    """
    if _has_valid_ops_session(request):
        return True
    return await verify_api_key(api_key)


async def verify_api_key(
    api_key: Optional[str] = Security(api_key_header),
) -> bool:
    """
    Verify API key for protected endpoints.

    SECURITY: In production, API_KEY must be configured. Empty API_KEY
    blocks all admin requests (fail-closed). In development, empty API_KEY
    allows all requests for convenience.
    """
    # FAIL-CLOSED: In production, require API_KEY to be configured
    if not settings.API_KEY:
        if IS_PRODUCTION:
            logger.error("API_KEY not configured in production - blocking admin access")
            raise HTTPException(
                status_code=503,
                detail="Service misconfigured. Admin access disabled.",
            )
        # Dev only: allow all if API_KEY not set
        return True

    if not api_key:
        raise HTTPException(
            status_code=401,
            detail="Missing API key. Provide it via X-API-Key header.",
        )

    if api_key != settings.API_KEY:
        logger.warning(f"Invalid API key attempt from request")
        raise HTTPException(
            status_code=403,
            detail="Invalid API key",
        )

    return True


async def optional_api_key(
    api_key: Optional[str] = Security(api_key_header),
) -> Optional[str]:
    """
    Optional API key check - doesn't fail if missing.
    Useful for endpoints with different behavior based on auth status.
    """
    if not settings.API_KEY:
        return None
    return api_key if api_key == settings.API_KEY else None


def get_rate_limit_key(request: Request) -> str:
    """
    Generate rate limit key based on IP and optional API key.

    Authenticated requests get higher limits.
    """
    api_key = request.headers.get(settings.API_KEY_HEADER)

    # If valid API key, use a separate bucket with higher limits
    if settings.API_KEY and api_key == settings.API_KEY:
        return f"authenticated:{api_key[:8]}"

    # Default: limit by IP
    return get_remote_address(request)


# =============================================================================
# Dashboard Token Authentication
# =============================================================================


def verify_dashboard_token_bool(request: Request) -> bool:
    """
    Verify dashboard access via token OR session. Returns bool (no exception).

    Auth methods (in order of preference):
    1. X-Dashboard-Token header (for services/automation)
    2. Valid session cookie (for web browser access)
    3. Query param token (dev only, disabled in prod)
    """
    # Method 1: Check header token
    token = settings.DASHBOARD_TOKEN
    if token:
        provided = request.headers.get("X-Dashboard-Token")
        if provided == token:
            return True

    # Method 2: Check valid session (reuses _has_valid_ops_session, P0-2)
    if _has_valid_ops_session(request):
        return True

    # Method 3: Query param fallback ONLY in development
    if token and not IS_PRODUCTION:
        provided = request.query_params.get("token")
        if provided == token:
            return True

    return False


async def verify_dashboard_token(
    x_dashboard_token: str = Header(None, alias="X-Dashboard-Token"),
) -> str:
    """
    FastAPI Dependency version for Depends() in routers.

    Raises HTTPException on failure (suitable for router-level dependencies).
    """
    expected = settings.DASHBOARD_TOKEN

    # If no token configured, allow all (dev mode)
    if not expected:
        return "dev-mode"

    if not x_dashboard_token:
        raise HTTPException(
            status_code=401,
            detail="Missing X-Dashboard-Token header",
        )

    if x_dashboard_token != expected:
        raise HTTPException(
            status_code=403,
            detail="Invalid X-Dashboard-Token",
        )

    return x_dashboard_token


def _get_dashboard_token_from_request(request: Request) -> str | None:
    """
    Extract dashboard token from request headers (prod) or query (dev).

    SECURITY: In production, only accepts token via X-Dashboard-Token header.
    Query params are only allowed in development (token leaks in logs/browser history).
    """
    token = request.headers.get("X-Dashboard-Token")

    # Query param fallback ONLY in development
    if not token and not IS_PRODUCTION:
        token = request.query_params.get("token")

    return token


def verify_debug_token(request: Request) -> None:
    """
    Verify dashboard token for debug endpoints. Raises HTTPException if invalid.

    Accepts either:
    - X-Dashboard-Token header
    - Valid session cookie

    SECURITY: Query params disabled in prod.
    """
    # Check session first (for browser access)
    if _has_valid_ops_session(request):
        return

    # Then check header token
    expected = settings.DASHBOARD_TOKEN
    if not expected:
        raise HTTPException(status_code=503, detail="Dashboard token not configured")

    provided = _get_dashboard_token_from_request(request)
    if provided and provided == expected:
        return

    raise HTTPException(status_code=401, detail="Invalid token")
