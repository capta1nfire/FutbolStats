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

    # Rate Limiting (Mega Plan: 600 r/m, 150K/day)
    API_REQUESTS_PER_MINUTE: int = 600
    API_DAILY_BUDGET: int = 150000
    RATE_LIMIT_PER_MINUTE: str = "60/minute"

    # API Security
    API_KEY: str = ""  # Optional API key for admin endpoints
    API_KEY_HEADER: str = "X-API-Key"

    # Model versioning
    MODEL_VERSION: str = "v1.0.1-league-only"

    # ═══════════════════════════════════════════════════════════════
    # FASE 1: League-Only + Kill-Switch + Policy Draw Cap
    # ═══════════════════════════════════════════════════════════════

    # Kill-switch: Filter matches where teams lack league history
    KILLSWITCH_ENABLED: bool = True
    KILLSWITCH_MIN_LEAGUE_MATCHES: int = 5
    KILLSWITCH_LOOKBACK_DAYS: int = 90

    # Policy: Draw cap to prevent over-concentration
    POLICY_DRAW_CAP_ENABLED: bool = True
    POLICY_MAX_DRAW_SHARE: float = 0.35
    POLICY_EDGE_THRESHOLD: float = 0.05

    # Policy: Market anchor for low-signal leagues (ABE P0 2026-02-08)
    MARKET_ANCHOR_ENABLED: bool = False  # Feature flag (default OFF until validated)
    MARKET_ANCHOR_ALPHA_DEFAULT: float = 0.0  # 0.0 = only overrides apply (no global blend)
    MARKET_ANCHOR_LEAGUE_OVERRIDES: str = "128:1.0"  # "league_id:alpha" — explicit low-signal leagues
    MARKET_ANCHOR_MIN_SAMPLES: int = 200  # Min FT with odds before accepting per-league α

    # Trading Core: Kelly Criterion stake sizing (GDT Epoch 2 2026-02-22)
    TRADING_KELLY_ENABLED: bool = False            # Feature flag (default OFF — shadow mode)
    TRADING_KELLY_FRACTION: float = 0.125          # Eighth-Kelly (GDT risk adjustment 2026-02-23)
    TRADING_BANKROLL_UNITS: float = 1000.0         # Abstract units for stake_units
    TRADING_MIN_EV: float = 0.05                   # Min 5% EV to size a bet (GDT 2026-02-23)
    TRADING_MAX_ODDS_PENALTY_THRESHOLD: float = 5.0  # Odds above this get penalized
    TRADING_MAX_ODDS_PENALTY_FACTOR: float = 0.5   # 50% penalty for high-odds bets
    TRADING_MAX_STAKE_PCT: float = 0.05            # Max 5% bankroll per match

    # Calibration: optional post-hoc temperature scaling (ABE P0 2026-02-19)
    PROBA_CALIBRATION_ENABLED: bool = False       # Feature flag (default OFF)
    PROBA_CALIBRATION_METHOD: str = "temperature"  # "temperature" | "none"
    PROBA_CALIBRATION_TEMPERATURE: float = 1.0     # T=1.0 = identity (no-op)

    # League Router: Asymmetric prediction routing (GDT M3 2026-02-15)
    LEAGUE_ROUTER_ENABLED: bool = True        # Enable tier classification + logging
    LEAGUE_ROUTER_MTV_ENABLED: bool = False   # Gate MTV feature injection for Tier 3
    # Requires trained Family S model. Keep False until model deployed.

    # SOTA: FotMob xG provider (ABE P0 2026-02-08)
    FOTMOB_REFS_ENABLED: bool = False         # Flag refs sync (Phase A)
    FOTMOB_XG_ENABLED: bool = False           # Flag xG backfill (Phase B) — requires refs
    FOTMOB_XG_LEAGUES: str = "128"            # P0-8: default SOLO Argentina
    FOTMOB_PROXY_URL: str = ""                # IPRoyal proxy (shared or separate from SofaScore)
    FOTMOB_RATE_LIMIT_SECONDS: float = 1.5    # Slightly more conservative than SofaScore
    FOTMOB_CIRCUIT_BREAKER_THRESHOLD: int = 5 # Consecutive errors before skip

    # ═══════════════════════════════════════════════════════════════

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
    # Sensor B numerical stability / calibration controls
    # Temperature > 1.0 softens overconfident probabilities (diagnostic still valid)
    SENSOR_TEMPERATURE: float = 2.0
    # Floor/ceiling for probabilities used by Sensor B (prevents extreme ~0/1 outputs)
    SENSOR_PROB_EPS: float = 1e-12

    # Telemetry: Stale evaluation thresholds (minutes)
    SHADOW_EVAL_STALE_MINUTES: int = 120  # Alert if oldest pending > this
    SENSOR_EVAL_STALE_MINUTES: int = 120  # Alert if oldest pending > this

    # ═══════════════════════════════════════════════════════════════
    # SHADOW ext-A/B/C (experimental model evaluation)
    # ATI: Un solo job genérico procesa todas las variantes habilitadas
    # ═══════════════════════════════════════════════════════════════
    # ext-A: min_date 2024-01-01
    EXTA_SHADOW_ENABLED: bool = False  # ATI: default OFF
    EXTA_SHADOW_MODEL_VERSION: str = "v1.0.2-ext-A"
    EXTA_SHADOW_MODEL_PATH: str = "models/xgb_v1.0.2-ext-A_20260201.json"

    # ext-B: min_date 2023-11-01
    EXTB_SHADOW_ENABLED: bool = False  # ATI: default OFF
    EXTB_SHADOW_MODEL_VERSION: str = "v1.0.2-ext-B"
    EXTB_SHADOW_MODEL_PATH: str = "models/xgb_v1.0.2-ext-B_20260201.json"

    # ext-C: min_date 2023-08-01 (ya existente)
    EXTC_SHADOW_ENABLED: bool = False  # ATI: default OFF, activar explícitamente en Railway
    EXTC_SHADOW_MODEL_VERSION: str = "v1.0.2-ext-C"
    EXTC_SHADOW_MODEL_PATH: str = "models/xgb_v1.0.2-ext-C_20260201.json"

    # ext-D: league-only candidato (v1.0.1-league-only trained 2026-02-02)
    # ATI: Shadow paralelo para evaluación extendida sin tocar ext-A/B/C
    EXTD_SHADOW_ENABLED: bool = False  # ATI: default OFF
    EXTD_SHADOW_MODEL_VERSION: str = "v1.0.1-league-only-20260202"
    EXTD_SHADOW_MODEL_PATH: str = "models/xgb_v1.0.1-league-only_20260202.json"

    # Shared settings for all ext variants
    EXT_SHADOW_BATCH_SIZE: int = 200  # Max snapshots per run per variant
    EXT_SHADOW_INTERVAL_MINUTES: int = 30  # Run every 30 min
    # ATI: Controlar si backfill o solo OOS forward
    EXT_SHADOW_START_AT: str = "2026-02-01"  # ISO date, cambiar para backfill
    EXT_SHADOW_OOS_ONLY: bool = True  # True = solo snapshots recientes (desde deploy)

    # Legacy aliases (for backwards compatibility)
    EXTC_SHADOW_BATCH_SIZE: int = 200
    EXTC_SHADOW_INTERVAL_MINUTES: int = 30
    EXTC_SHADOW_START_AT: str = "2026-02-01"
    EXTC_SHADOW_OOS_ONLY: bool = True

    # Development settings
    SKIP_AUTO_TRAIN: bool = False  # Skip auto-training on startup (useful during dev)

    # Dashboard security
    DASHBOARD_TOKEN: str = ""  # Token for /dashboard/* access (empty = disabled)

    # Alerts webhook security (Grafana → ops_alerts)
    ALERTS_WEBHOOK_SECRET: str = ""  # Secret for POST /dashboard/ops/alerts/webhook

    # Stats Backfill Job Configuration
    STATS_BACKFILL_ENABLED: bool = True
    STATS_BACKFILL_LOOKBACK_HOURS: int = 72
    STATS_BACKFILL_MAX_CALLS_PER_RUN: int = 200

    # Player Stats Sync (going-forward job for match_player_stats)
    PLAYER_STATS_SYNC_ENABLED: bool = False  # OFF during incubation
    PLAYER_STATS_SYNC_DELAY_HOURS: int = 4   # T+4h after FT (provider audit lag)
    PLAYER_STATS_SYNC_MAX_CALLS: int = 50    # Max calls per run

    # Wikidata Team Enrichment
    WIKIDATA_ENRICH_ENABLED: bool = False  # Feature flag (off by default)
    WIKIDATA_ENRICH_BATCH_SIZE: int = 100  # Teams per run (736 teams / 100 = ~8 days catch-up)
    WIKIDATA_SPARQL_ENDPOINT: str = "https://query.wikidata.org/sparql"
    WIKIDATA_RATE_LIMIT_DELAY: float = 0.2  # 5 req/sec (conservative, Wikidata allows 500/min)
    WIKIPEDIA_RATE_LIMIT_DELAY: float = 0.3  # ~3 req/sec (Wikipedia allows 200/min = 3.3 req/sec)

    # Auto-Lab Online (PASO 2: advisory feature evaluation per league)
    AUTO_LAB_ENABLED: bool = False               # Kill-switch (default OFF)
    AUTO_LAB_MAX_PER_DAY: int = 1                # Max leagues per day
    AUTO_LAB_TIMEOUT_MIN: int = 30               # Hard timeout per league (minutes)
    AUTO_LAB_CADENCE_HIGH_DAYS: int = 14         # Re-run if >=180 FT in 90d
    AUTO_LAB_CADENCE_MID_DAYS: int = 21          # Re-run if >=120 FT in 90d
    AUTO_LAB_CADENCE_LOW_DAYS: int = 35          # Re-run if <120 FT in 90d
    AUTO_LAB_MIN_MATCHES_TOTAL: int = 200        # Min FT matches to run

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

    # Gemini pricing (per 1M tokens) - SINGLE SOURCE OF TRUTH
    # Update this dict when adding/changing models
    # Pricing as of Jan 2026 (https://ai.google.dev/pricing)
    GEMINI_PRICING: dict = {
        "gemini-2.0-flash": {"input": 0.10, "output": 0.40},
        "gemini-2.0-flash-exp": {"input": 0.10, "output": 0.40},
        "gemini-2.5-flash": {"input": 0.10, "output": 0.40},
        "gemini-2.5-pro": {"input": 1.25, "output": 10.00},
        "gemini-1.5-flash": {"input": 0.075, "output": 0.30},
        "gemini-1.5-pro": {"input": 1.25, "output": 5.00},
    }
    # Default pricing fallback (used when model not in GEMINI_PRICING)
    GEMINI_PRICE_INPUT: float = 0.10
    GEMINI_PRICE_OUTPUT: float = 0.40

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

    # Players & Managers (Phase 1 MVP)
    INJURIES_SYNC_ENABLED: bool = False  # Kill-switch for injuries sync job
    MANAGER_SYNC_ENABLED: bool = False   # Kill-switch for manager sync job

    # Sofascore Refs Matching
    SOFASCORE_REFS_THRESHOLD: float = 0.75  # Default threshold for match score
    SOFASCORE_REFS_THRESHOLD_OVERRIDES: str = ""  # Per-league: "128:0.70,307:0.70"

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

    # Sentry API (for ops dashboard health aggregation - server-side only)
    SENTRY_AUTH_TOKEN: str = ""  # Read-only auth token for Sentry API
    SENTRY_PROJECT_SLUG: str = ""  # e.g. futbolstats (project slug, not ID)
    SENTRY_ENV: str = "production"  # Environment filter (default: production)

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# ============================================================================
# LLM Models Catalog (Single Source of Truth for pricing and capabilities)
# ============================================================================

LLM_MODELS: dict = {
    "gemini-2.5-flash-lite": {
        "provider": "gemini",
        "display_name": "Gemini 2.5 Flash-Lite",
        "model_id": "gemini-2.5-flash-lite",
        "input_price_per_1m": 0.10,
        "output_price_per_1m": 0.40,
        "max_tokens": 8192,
        "default_temperature": 0.7,
    },
    "gemini-2.5-flash": {
        "provider": "gemini",
        "display_name": "Gemini 2.5 Flash",
        "model_id": "gemini-2.5-flash",
        "input_price_per_1m": 0.30,
        "output_price_per_1m": 2.50,
        "max_tokens": 8192,
        "default_temperature": 0.7,
    },
    "gemini-2.0-flash": {
        "provider": "gemini",
        "display_name": "Gemini 2.0 Flash",
        "model_id": "gemini-2.0-flash",
        "input_price_per_1m": 0.10,
        "output_price_per_1m": 0.40,
        "max_tokens": 8192,
        "default_temperature": 0.7,
    },
    "kimi-k2.5": {
        "provider": "moonshot",
        "display_name": "Kimi K2.5",
        "model_id": "kimi-k2.5",
        "input_price_per_1m": 0.60,
        "output_price_per_1m": 2.50,
        "max_tokens": 131072,
        "default_temperature": 0.6,
    },
    "qwen-vllm": {
        "provider": "runpod",
        "display_name": "Qwen vLLM (RunPod)",
        "model_id": "qwen-vllm",
        "input_price_per_1m": 0.00,  # RunPod is per-second, not per-token
        "output_price_per_1m": 0.00,
        "max_tokens": 4096,
        "default_temperature": 0.7,
    },
}


async def get_ia_features_config(session) -> dict:
    """
    Get IA Features configuration from ops_settings.

    Priority: ops_settings > env vars > defaults

    Returns dict with:
      - narratives_enabled: bool | None (None = inherit from env)
      - narrative_feedback_enabled: bool
      - primary_model: str (model key from LLM_MODELS)
      - temperature: float
      - max_tokens: int
    """
    from sqlalchemy import text

    # Defaults (fallback if no DB entry)
    defaults = {
        "narratives_enabled": None,  # Will inherit from FASTPATH_ENABLED
        "narrative_feedback_enabled": False,
        "primary_model": "gemini-2.5-flash-lite",
        "temperature": 0.7,
        "max_tokens": 4096,
    }

    try:
        result = await session.execute(
            text("SELECT value FROM ops_settings WHERE key = 'ia_features'")
        )
        row = result.fetchone()
        if row and row[0]:
            config = row[0]
            # Merge with defaults (in case new fields added)
            return {**defaults, **config}
    except Exception:
        pass

    return defaults


def should_generate_narratives(ia_config: dict, env_settings: Settings) -> bool:
    """
    Determine if narratives should be generated.

    3-state logic:
      - ia_config['narratives_enabled'] = True  → always generate (override)
      - ia_config['narratives_enabled'] = False → never generate (override)
      - ia_config['narratives_enabled'] = None  → inherit from env (FASTPATH_ENABLED)
    """
    if ia_config.get("narratives_enabled") is None:
        return env_settings.FASTPATH_ENABLED
    return ia_config.get("narratives_enabled", False)
