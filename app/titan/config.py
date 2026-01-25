"""TITAN OMNISCIENCE Configuration.

Configuration for TITAN infrastructure including R2 storage (FASE 2).
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

    # Sources
    TITAN_SOURCE_API_FOOTBALL: str = "api_football"
    TITAN_SOURCE_UNDERSTAT: str = "understat"

    # Dashboard
    TITAN_DASHBOARD_ENABLED: bool = True

    # ==========================================================================
    # R2 Storage Configuration (FASE 2)
    # ==========================================================================
    # Cloudflare R2 / S3-compatible storage for large response offload

    R2_ENABLED: bool = False
    R2_ENDPOINT_URL: str = ""  # https://<account_id>.r2.cloudflarestorage.com
    R2_ACCESS_KEY_ID: str = ""
    R2_SECRET_ACCESS_KEY: str = ""
    R2_BUCKET: str = "titan-extractions"

    # Offload threshold: responses larger than this go to R2
    R2_OFFLOAD_THRESHOLD_BYTES: int = 100 * 1024  # 100KB

    # Retention policy
    RETENTION_DAYS_DB: int = 90   # Keep response_body in DB for 90 days
    RETENTION_DAYS_R2: int = 365  # Keep in R2 for 1 year

    # ==========================================================================
    # xG Features Configuration (FASE 2)
    # ==========================================================================

    XG_ROLLING_WINDOW: int = 5  # Number of past matches for xG average

    class Config:
        env_prefix = "TITAN_"
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache
def get_titan_settings() -> TitanSettings:
    """Get cached TITAN settings instance."""
    return TitanSettings()
