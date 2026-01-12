"""
Telemetry configuration with environment variable overrides.
"""

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class TelemetryConfig:
    """Configuration for Data Quality Telemetry thresholds."""

    # Feature flags
    DQ_ENABLED: bool = True
    DQ_MARKET_VALIDATION_ENABLED: bool = True
    DQ_ANTI_LOOKAHEAD_ENABLED: bool = True
    DQ_MAPPING_TRACKING_ENABLED: bool = True

    # Provider availability thresholds
    DQ_SUCCESS_RATE_WARN: float = 0.98
    DQ_SUCCESS_RATE_RED: float = 0.95

    # Overround bounds for 1X2 market
    DQ_OVERROUND_1X2_MIN: float = 1.01
    DQ_OVERROUND_1X2_MAX: float = 1.20

    # Odds sanity bounds
    DQ_ODDS_MIN: float = 1.01
    DQ_ODDS_MAX: float = 1000.0

    # Event latency thresholds (anti-lookahead)
    DQ_EVENT_LAG_P95_WARN_SECONDS: float = 30.0
    DQ_EVENT_LAG_P95_RED_SECONDS: float = 90.0

    # Entity mapping coverage thresholds
    DQ_MAPPING_COVERAGE_WARN: float = 0.995
    DQ_MAPPING_COVERAGE_RED: float = 0.99

    # Frozen market detection (if applicable)
    DQ_FROZEN_MINUTES_WARN: int = 8
    DQ_FROZEN_MINUTES_RED: int = 15

    @classmethod
    def from_env(cls) -> "TelemetryConfig":
        """Load configuration from environment variables."""
        def get_bool(key: str, default: bool) -> bool:
            val = os.environ.get(key, "").lower()
            if val in ("true", "1", "yes"):
                return True
            if val in ("false", "0", "no"):
                return False
            return default

        def get_float(key: str, default: float) -> float:
            try:
                return float(os.environ.get(key, default))
            except (ValueError, TypeError):
                return default

        def get_int(key: str, default: int) -> int:
            try:
                return int(os.environ.get(key, default))
            except (ValueError, TypeError):
                return default

        return cls(
            DQ_ENABLED=get_bool("DQ_ENABLED", True),
            DQ_MARKET_VALIDATION_ENABLED=get_bool("DQ_MARKET_VALIDATION_ENABLED", True),
            DQ_ANTI_LOOKAHEAD_ENABLED=get_bool("DQ_ANTI_LOOKAHEAD_ENABLED", True),
            DQ_MAPPING_TRACKING_ENABLED=get_bool("DQ_MAPPING_TRACKING_ENABLED", True),
            DQ_SUCCESS_RATE_WARN=get_float("DQ_SUCCESS_RATE_WARN", 0.98),
            DQ_SUCCESS_RATE_RED=get_float("DQ_SUCCESS_RATE_RED", 0.95),
            DQ_OVERROUND_1X2_MIN=get_float("DQ_OVERROUND_1X2_MIN", 1.01),
            DQ_OVERROUND_1X2_MAX=get_float("DQ_OVERROUND_1X2_MAX", 1.20),
            DQ_ODDS_MIN=get_float("DQ_ODDS_MIN", 1.01),
            DQ_ODDS_MAX=get_float("DQ_ODDS_MAX", 1000.0),
            DQ_EVENT_LAG_P95_WARN_SECONDS=get_float("DQ_EVENT_LAG_P95_WARN_SECONDS", 30.0),
            DQ_EVENT_LAG_P95_RED_SECONDS=get_float("DQ_EVENT_LAG_P95_RED_SECONDS", 90.0),
            DQ_MAPPING_COVERAGE_WARN=get_float("DQ_MAPPING_COVERAGE_WARN", 0.995),
            DQ_MAPPING_COVERAGE_RED=get_float("DQ_MAPPING_COVERAGE_RED", 0.99),
            DQ_FROZEN_MINUTES_WARN=get_int("DQ_FROZEN_MINUTES_WARN", 8),
            DQ_FROZEN_MINUTES_RED=get_int("DQ_FROZEN_MINUTES_RED", 15),
        )


# Global config instance (loaded once at import)
_config: Optional[TelemetryConfig] = None


def get_telemetry_config() -> TelemetryConfig:
    """Get the global telemetry configuration."""
    global _config
    if _config is None:
        _config = TelemetryConfig.from_env()
    return _config
