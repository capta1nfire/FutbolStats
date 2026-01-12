"""Application configuration using Pydantic Settings."""

from functools import lru_cache
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Database
    DATABASE_URL: str

    # API-Football (RapidAPI)
    RAPIDAPI_KEY: str
    RAPIDAPI_HOST: str = "api-football-v1.p.rapidapi.com"

    # ML Config
    MODEL_PATH: str = "./models"
    ROLLING_WINDOW: int = 5
    TIME_DECAY_LAMBDA: float = 0.01

    # Rate Limiting (Ultra Plan: 450 r/m, 75K/day)
    API_REQUESTS_PER_MINUTE: int = 450
    API_DAILY_BUDGET: int = 75000
    RATE_LIMIT_PER_MINUTE: str = "60/minute"

    # API Security
    API_KEY: str = ""  # Optional API key for admin endpoints
    API_KEY_HEADER: str = "X-API-Key"

    # Model versioning
    MODEL_VERSION: str = "v1.0.0"

    # Development settings
    SKIP_AUTO_TRAIN: bool = False  # Skip auto-training on startup (useful during dev)

    # Dashboard security
    DASHBOARD_TOKEN: str = ""  # Token for /dashboard/pit access (empty = disabled)

    # Stats Backfill Job Configuration
    STATS_BACKFILL_ENABLED: bool = True
    STATS_BACKFILL_LOOKBACK_HOURS: int = 72
    STATS_BACKFILL_MAX_CALLS_PER_RUN: int = 200

    # RunPod LLM Narrative Configuration
    RUNPOD_API_KEY: str = ""
    RUNPOD_ENDPOINT_ID: str = "a49n0iddpgsv7r"
    RUNPOD_BASE_URL: str = "https://api.runpod.ai/v2"
    NARRATIVE_LLM_ENABLED: bool = False
    NARRATIVE_LLM_MAX_TOKENS: int = 1500
    NARRATIVE_LLM_TIMEOUT_SECONDS: int = 60
    NARRATIVE_LLM_POLL_INTERVAL_SECONDS: float = 1.5
    NARRATIVE_LLM_TEMPERATURE: float = 0.15
    NARRATIVE_LLM_TOP_P: float = 0.9

    # Fast-Path Narrative Configuration
    FASTPATH_ENABLED: bool = False
    FASTPATH_INTERVAL_SECONDS: int = 120
    FASTPATH_LOOKBACK_MINUTES: int = 90
    FASTPATH_MAX_CONCURRENT_JOBS: int = 10

    # Telemetry / Observability
    METRICS_BEARER_TOKEN: str = ""  # Bearer token for /metrics endpoint (Grafana Cloud scraping)

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
