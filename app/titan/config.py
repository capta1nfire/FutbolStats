"""TITAN OMNISCIENCE Configuration.

Minimal configuration for FASE 1. R2/aioboto3 deferred to FASE 2.
"""

from functools import lru_cache
from typing import Optional

from pydantic_settings import BaseSettings


class TitanSettings(BaseSettings):
    """TITAN-specific settings (supplements main app Settings)."""

    # Schema
    TITAN_SCHEMA: str = "titan"

    # Idempotency
    TITAN_IDEMPOTENCY_KEY_LENGTH: int = 32  # CHAR(32) = SHA256[:32]

    # DLQ Retry Policy
    TITAN_DLQ_MAX_ATTEMPTS: int = 3
    TITAN_DLQ_RETRY_BASE_SECONDS: int = 60  # Exponential backoff base
    TITAN_DLQ_RETRY_MAX_SECONDS: int = 3600  # Max 1 hour between retries

    # Feature Matrix
    TITAN_FEATURE_MATRIX_REQUIRE_TIER1: bool = True  # Tier 1 (odds) required for insert

    # Sources (FASE 1: only API-Football)
    TITAN_SOURCE_API_FOOTBALL: str = "api_football"

    # Dashboard
    TITAN_DASHBOARD_ENABLED: bool = True

    class Config:
        env_prefix = "TITAN_"
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache
def get_titan_settings() -> TitanSettings:
    """Get cached TITAN settings instance."""
    return TitanSettings()
