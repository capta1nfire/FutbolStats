"""Security middleware: Rate limiting and API key authentication."""

import logging
from typing import Optional

from fastapi import HTTPException, Request, Security
from fastapi.security import APIKeyHeader
from slowapi import Limiter
from slowapi.util import get_remote_address

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# Rate limiter using client IP
limiter = Limiter(key_func=get_remote_address)

# API Key header for admin endpoints
api_key_header = APIKeyHeader(name=settings.API_KEY_HEADER, auto_error=False)


async def verify_api_key(
    api_key: Optional[str] = Security(api_key_header),
) -> bool:
    """
    Verify API key for protected endpoints.

    If API_KEY is not configured (empty), all requests are allowed.
    This enables gradual rollout of authentication.
    """
    # If no API key is configured, allow all requests
    if not settings.API_KEY:
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
