"""Authentication for Logos Dashboard endpoints.

Provides X-Dashboard-Token verification as a FastAPI dependency.
Uses the same DASHBOARD_TOKEN from config as other dashboard endpoints.
"""

from fastapi import Depends, HTTPException, Header
from functools import lru_cache

from app.config import get_settings


@lru_cache()
def get_dashboard_token() -> str:
    """Get dashboard token from settings (cached)."""
    settings = get_settings()
    return settings.DASHBOARD_TOKEN


async def verify_dashboard_token(
    x_dashboard_token: str = Header(None, alias="X-Dashboard-Token"),
) -> str:
    """Verify X-Dashboard-Token header.

    FastAPI dependency that checks the X-Dashboard-Token header
    against the configured DASHBOARD_TOKEN.

    Args:
        x_dashboard_token: Token from request header

    Returns:
        The verified token

    Raises:
        HTTPException 401: If token is missing
        HTTPException 403: If token is invalid
    """
    expected_token = get_dashboard_token()

    # If no token configured, allow all (dev mode warning)
    if not expected_token:
        return "dev-mode"

    if not x_dashboard_token:
        raise HTTPException(
            status_code=401,
            detail="Missing X-Dashboard-Token header",
        )

    if x_dashboard_token != expected_token:
        raise HTTPException(
            status_code=403,
            detail="Invalid X-Dashboard-Token",
        )

    return x_dashboard_token
