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

    # Model architecture (baseline | two_stage)
    MODEL_ARCHITECTURE: str = "baseline"
    # Shadow architecture for A/B comparison (runs in parallel, logged but not served)
    MODEL_SHADOW_ARCHITECTURE: str = ""  # Empty = disabled
    # Draw threshold for two-stage (None = use argmax, 0.30 = threshold override)
    MODEL_DRAW_THRESHOLD: float = 0.0  # 0.0 = disabled (argmax), >0 = threshold

    # Prediction Rerun Serving Preference
    # When True, serve two-stage predictions from active reruns for NS matches
    # When False (or no active rerun), serve baseline predictions
    # This allows A/B testing and rollback via flag toggle without deleting data
    PREFER_RERUN_PREDICTIONS: bool = False  # Default: serve baseline
    RERUN_FRESHNESS_HOURS: int = 6  # Max age of DB prediction before falling back to live

    # Shadow Mode Evaluation Thresholds
    SHADOW_MIN_SAMPLES: int = 50  # Minimum evaluated predictions for GO/NO_GO decision
    SHADOW_BRIER_IMPROVEMENT_MIN: float = 0.005  # Shadow must improve brier by this much for GO
    SHADOW_ACCURACY_DROP_MAX: float = 0.01  # Max accuracy drop allowed (1%)
    SHADOW_WINDOW_DAYS: int = 14  # Window for shadow evaluation metrics

    # Sensor B (Calibration Diagnostics) - Internal only, not for production
    SENSOR_ENABLED: bool = True
    SENSOR_WINDOW_SIZE: int = 50  # Number of recent FT matches for training
    SENSOR_MIN_SAMPLES: int = 50  # Minimum evaluated predictions for reporting
    SENSOR_RETRAIN_INTERVAL_HOURS: int = 6  # How often to retrain sensor
    SENSOR_SIGNAL_SCORE_GO: float = 1.1  # Signal score above this = A may be stale
    SENSOR_SIGNAL_SCORE_NOISE: float = 0.9  # Signal score below this = B is overfitting
    SENSOR_EVAL_WINDOW_DAYS: int = 14  # Window for sensor evaluation metrics

    # Telemetry: Stale evaluation thresholds (minutes)
    SHADOW_EVAL_STALE_MINUTES: int = 120  # Alert if oldest pending > this
    SENSOR_EVAL_STALE_MINUTES: int = 120  # Alert if oldest pending > this

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
    NARRATIVE_LLM_TEMPERATURE: float = 0.28
    NARRATIVE_LLM_TOP_P: float = 0.9

    # Gemini LLM Configuration
    GEMINI_API_KEY: str = ""
    GEMINI_MODEL: str = "gemini-2.0-flash"
    GEMINI_MAX_TOKENS: int = 1000
    # Gemini pricing (per 1M tokens) - adjust when changing models
    # 2.0-flash: $0.075/$0.30, 2.5-flash: $0.15/$0.60
    GEMINI_PRICE_INPUT: float = 0.075
    GEMINI_PRICE_OUTPUT: float = 0.30

    # Narrative Provider Selection (runpod | gemini)
    NARRATIVE_PROVIDER: str = "runpod"

    # Fast-Path Narrative Configuration
    FASTPATH_ENABLED: bool = False
    FASTPATH_INTERVAL_SECONDS: int = 120
    FASTPATH_LOOKBACK_MINUTES: int = 90
    FASTPATH_MAX_CONCURRENT_JOBS: int = 10

    # Odds Sync Job Configuration
    ODDS_SYNC_ENABLED: bool = True  # Kill-switch for odds sync job
    ODDS_SYNC_WINDOW_HOURS: int = 48  # Sync odds for matches in next N hours
    ODDS_SYNC_MAX_FIXTURES: int = 100  # Max fixtures per run (API budget control)
    ODDS_SYNC_FRESHNESS_HOURS: int = 6  # Skip if odds updated within N hours
    ODDS_SYNC_INTERVAL_HOURS: int = 6  # How often to run the job

    # Telemetry / Observability
    METRICS_BEARER_TOKEN: str = ""  # Bearer token for /metrics endpoint (Grafana Cloud scraping)

    # Email Alerting (SMTP)
    SMTP_ENABLED: bool = False
    SMTP_HOST: str = "smtp.gmail.com"
    SMTP_PORT: int = 587
    SMTP_USERNAME: str = ""
    SMTP_PASSWORD: str = ""  # App password for Gmail
    SMTP_FROM_EMAIL: str = ""
    SMTP_TO_EMAIL: str = "capta1nfire@me.com"
    ALERT_COOLDOWN_MINUTES: int = 60  # Min time between same alert type

    # OPS Console Session Auth (web login)
    OPS_ADMIN_PASSWORD: str = ""  # Password for /ops/login (empty = login disabled)
    OPS_SESSION_SECRET: str = ""  # Secret key for session signing (required in prod)
    OPS_SESSION_TTL_HOURS: int = 8  # Session duration

    # OPS Dashboard External Links (for drill-down)
    # Sentry: org slug and project ID (avoids URL encoding issues with ?)
    SENTRY_ORG: str = ""  # e.g. devseqio
    SENTRY_PROJECT_ID: str = ""  # e.g. 4510721108869120
    GRAFANA_BASE_URL: str = ""  # e.g. https://capta1nfire.grafana.net
    GITHUB_REPO_URL: str = "https://github.com/capta1nfire/FutbolStats"  # For runbook links

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
