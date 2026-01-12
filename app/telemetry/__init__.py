"""
Data Quality Telemetry Module (P0)

Provides Prometheus metrics for:
- Provider ingestion (requests, errors, latency)
- Anti-lookahead (event latency, tainted records)
- Market integrity (odds validation, overround, frozen markets)
- Entity mapping coverage

Design: multi-provider ready with provider label.
"""

from app.telemetry.metrics import (
    # Ingestión
    dq_provider_requests_total,
    dq_provider_errors_total,
    dq_provider_latency_ms,
    dq_provider_rate_limited_total,
    dq_provider_timeouts_total,
    # Anti-lookahead
    dq_event_latency_seconds,
    dq_tainted_records_total,
    # Market integrity
    dq_odds_invariant_violations_total,
    dq_odds_quarantined_total,
    dq_frozen_market_suspects_total,
    dq_odds_overround,
    # Entity mapping
    dq_entity_mapping_unmapped_total,
    dq_entity_mapping_coverage_pct,
    # Helpers
    record_provider_request,
    record_provider_error,
    record_provider_latency,
    record_tainted_record,
    record_odds_violation,
    record_unmapped_entity,
    get_metrics_text,
)

from app.telemetry.validators import (
    validate_odds_1x2,
    OddsValidationResult,
)

from app.telemetry.config import TelemetryConfig

__all__ = [
    # Metrics
    "dq_provider_requests_total",
    "dq_provider_errors_total",
    "dq_provider_latency_ms",
    "dq_provider_rate_limited_total",
    "dq_provider_timeouts_total",
    "dq_event_latency_seconds",
    "dq_tainted_records_total",
    "dq_odds_invariant_violations_total",
    "dq_odds_quarantined_total",
    "dq_frozen_market_suspects_total",
    "dq_odds_overround",
    "dq_entity_mapping_unmapped_total",
    "dq_entity_mapping_coverage_pct",
    # Helpers
    "record_provider_request",
    "record_provider_error",
    "record_provider_latency",
    "record_tainted_record",
    "record_odds_violation",
    "record_unmapped_entity",
    "get_metrics_text",
    # Validators
    "validate_odds_1x2",
    "OddsValidationResult",
    # Config
    "TelemetryConfig",
]
