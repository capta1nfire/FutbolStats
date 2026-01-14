"""Security middleware: Rate limiting and API key authentication."""

import logging
import os
from typing import Optional

from fastapi import HTTPException, Request, Security
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
