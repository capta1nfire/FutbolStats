"""
Prometheus metrics for Data Quality Telemetry.

Design principles:
- Multi-provider ready (provider label on all metrics)
- Low cardinality (controlled labels)
- Best-effort (never block main flow)

=============================================================================
CARDINALITY CONTROL (CRITICAL)
=============================================================================

Labels are restricted to LOW-CARDINALITY values only. High-cardinality
identifiers will cause metric explosion and must NEVER be used as labels.

ALLOWED LABELS (bounded sets):
- provider:     "api_football", "football_data_uk", "betfair" (max ~10)
- entity:       "fixture", "odds", "lineup", "stats", "events" (max ~10)
- endpoint:     "fixtures", "fixtures/statistics", "odds", etc. (max ~20)
- status_code:  "200", "400", "404", "429", "500", "0" (max ~10)
- error_code:   "timeout", "rate_limit", "api_error", "http_4xx" (max ~15)
- book:         "bet365", "pinnacle", "betfair", "api_football" (max ~20)
- market:       "1x2", "over_under", "asian_handicap", "btts" (max ~10)
- rule:         "overround_low", "overround_high", "sanity" (max ~10)
- entity_type:  "team", "league", "match", "player" (max ~5)
- league:       Use league_id buckets or "top5", "other" (max ~50)
- reason:       "lag_exceeded", "stale_data", "validation_fail" (max ~15)

FORBIDDEN AS LABELS (will cause cardinality explosion):
- match_id, fixture_id, external_id
- team names, player names
- URLs, full paths
- Timestamps, dates
- Raw payloads, error messages
- Any unbounded identifier

For debugging specific matches/teams, use logs with structured fields,
NOT metric labels. Metrics are for aggregated observability.

Example of CORRECT instrumentation:
    record_provider_request(
        provider="api_football",
        entity="fixture",
        endpoint="fixtures",
        status_code=200,
        ...
    )

Example of WRONG instrumentation (DO NOT DO THIS):
    record_provider_request(
        provider="api_football",
        entity=f"match_{match_id}",  # WRONG: unbounded
        endpoint=f"/fixtures/{fixture_id}",  # WRONG: unbounded
        ...
    )
=============================================================================
"""

import time
import logging
from typing import Optional

from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    generate_latest,
    CONTENT_TYPE_LATEST,
    REGISTRY,
    CollectorRegistry,
)

logger = logging.getLogger(__name__)

# =============================================================================
# INGESTIÓN METRICS
# =============================================================================

dq_provider_requests_total = Counter(
    "dq_provider_requests_total",
    "Total requests to data providers",
    ["provider", "entity", "endpoint", "status_code"],
)

dq_provider_errors_total = Counter(
    "dq_provider_errors_total",
    "Total errors from data providers",
    ["provider", "entity", "error_code"],
)

dq_provider_rate_limited_total = Counter(
    "dq_provider_rate_limited_total",
    "Total rate-limited responses (429) from providers",
    ["provider", "entity"],
)

dq_provider_timeouts_total = Counter(
    "dq_provider_timeouts_total",
    "Total timeout errors from providers",
    ["provider", "entity"],
)

dq_provider_latency_ms = Histogram(
    "dq_provider_latency_ms",
    "Request latency in milliseconds",
    ["provider", "entity", "endpoint"],
    buckets=[10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000],
)

# =============================================================================
# ANTI-LOOKAHEAD METRICS
# =============================================================================

dq_event_latency_seconds = Histogram(
    "dq_event_latency_seconds",
    "Latency between event time and ingestion time (anti-lookahead)",
    ["provider", "entity"],
    buckets=[1, 5, 10, 30, 60, 120, 300, 600, 1800, 3600],
)

dq_tainted_records_total = Counter(
    "dq_tainted_records_total",
    "Records marked as tainted (potentially contaminated by lookahead)",
    ["provider", "entity", "reason"],
)

# =============================================================================
# MARKET INTEGRITY METRICS
# =============================================================================

dq_odds_invariant_violations_total = Counter(
    "dq_odds_invariant_violations_total",
    "Odds invariant violations detected",
    ["provider", "book", "market", "rule"],
)

dq_odds_quarantined_total = Counter(
    "dq_odds_quarantined_total",
    "Odds records quarantined due to validation failures",
    ["provider", "book", "reason"],
)

dq_frozen_market_suspects_total = Counter(
    "dq_frozen_market_suspects_total",
    "Suspected frozen markets (no updates for extended period)",
    ["provider", "book"],
)

# =============================================================================
# LLM NARRATIVE METRICS
# =============================================================================

llm_unsupported_claims_total = Counter(
    "llm_unsupported_claims_total",
    "LLM narratives rejected due to unsupported claims (hallucinations)",
    ["claim_type"],  # red_card, penalty, goal_minute
)

llm_narratives_validated_total = Counter(
    "llm_narratives_validated_total",
    "LLM narratives that passed claim validation",
    ["status"],  # ok, rejected
)

llm_requests_total = Counter(
    "llm_requests_total",
    "Total LLM requests by provider and status",
    ["provider", "status"],  # provider: runpod/gemini, status: ok/error/rejected
)

llm_latency_ms = Histogram(
    "llm_latency_ms",
    "LLM request latency in milliseconds by provider",
    ["provider"],
    buckets=[500, 1000, 2000, 3000, 5000, 10000, 20000, 30000, 60000, 120000],
)

llm_tokens_total = Counter(
    "llm_tokens_total",
    "Total LLM tokens by provider and direction",
    ["provider", "direction"],  # direction: input/output
)

llm_cost_usd = Counter(
    "llm_cost_usd",
    "Estimated LLM cost in USD by provider",
    ["provider"],
)

# Circuit breaker state
llm_circuit_open = Gauge(
    "llm_circuit_open",
    "Circuit breaker state (1=open/tripped, 0=closed/healthy)",
    ["provider"],
)

llm_consecutive_failures = Gauge(
    "llm_consecutive_failures",
    "Current consecutive failure count for circuit breaker",
    ["provider"],
)

# =============================================================================
# AGGREGATES METRICS (P1)
# =============================================================================

aggregates_refresh_runs_total = Counter(
    "aggregates_refresh_runs_total",
    "Total aggregates refresh job runs",
    ["status"],  # ok, error
)

aggregates_refresh_duration_ms = Histogram(
    "aggregates_refresh_duration_ms",
    "Aggregates refresh job duration in milliseconds",
    [],
    buckets=[1000, 5000, 10000, 30000, 60000, 120000, 300000],
)

aggregates_baselines_rows = Gauge(
    "aggregates_baselines_rows",
    "Current number of league baseline rows",
    [],
)

aggregates_profiles_rows = Gauge(
    "aggregates_profiles_rows",
    "Current number of team profile rows",
    [],
)

aggregates_leagues_distinct = Gauge(
    "aggregates_leagues_distinct",
    "Number of distinct leagues with baselines",
    [],
)

aggregates_profiles_min_sample_ok_pct = Gauge(
    "aggregates_profiles_min_sample_ok_pct",
    "Percentage of team profiles with min_sample_ok=true",
    [],
)

dq_odds_overround = Histogram(
    "dq_odds_overround",
    "Overround (margin) distribution for 1X2 markets",
    ["provider", "book"],
    buckets=[1.0, 1.02, 1.04, 1.06, 1.08, 1.10, 1.12, 1.15, 1.20, 1.30],
)

# =============================================================================
# ENTITY MAPPING METRICS
# =============================================================================

dq_entity_mapping_unmapped_total = Counter(
    "dq_entity_mapping_unmapped_total",
    "Unmapped entities encountered",
    ["provider", "entity_type", "league"],
)

dq_entity_mapping_coverage_pct = Gauge(
    "dq_entity_mapping_coverage_pct",
    "Entity mapping coverage percentage",
    ["provider", "entity_type", "league"],
)

# =============================================================================
# HELPER FUNCTIONS (for instrumentation)
# =============================================================================


def record_provider_request(
    provider: str,
    entity: str,
    endpoint: str,
    status_code: int,
    latency_ms: float,
    is_rate_limited: bool = False,
    is_timeout: bool = False,
) -> None:
    """Record a provider request with all associated metrics."""
    try:
        # Always record request count and latency
        dq_provider_requests_total.labels(
            provider=provider,
            entity=entity,
            endpoint=endpoint,
            status_code=str(status_code),
        ).inc()

        dq_provider_latency_ms.labels(
            provider=provider,
            entity=entity,
            endpoint=endpoint,
        ).observe(latency_ms)

        # Track rate limits and timeouts separately
        if is_rate_limited:
            dq_provider_rate_limited_total.labels(
                provider=provider,
                entity=entity,
            ).inc()

        if is_timeout:
            dq_provider_timeouts_total.labels(
                provider=provider,
                entity=entity,
            ).inc()

    except Exception as e:
        logger.warning(f"Failed to record provider request metric: {e}")


def record_provider_error(
    provider: str,
    entity: str,
    error_code: str,
) -> None:
    """Record a provider error."""
    try:
        dq_provider_errors_total.labels(
            provider=provider,
            entity=entity,
            error_code=error_code,
        ).inc()
    except Exception as e:
        logger.warning(f"Failed to record provider error metric: {e}")


def record_provider_latency(
    provider: str,
    entity: str,
    endpoint: str,
    latency_ms: float,
) -> None:
    """Record just the latency (when request count handled separately)."""
    try:
        dq_provider_latency_ms.labels(
            provider=provider,
            entity=entity,
            endpoint=endpoint,
        ).observe(latency_ms)
    except Exception as e:
        logger.warning(f"Failed to record latency metric: {e}")


def record_event_latency(
    provider: str,
    entity: str,
    latency_seconds: float,
) -> None:
    """Record event latency (for anti-lookahead tracking)."""
    try:
        dq_event_latency_seconds.labels(
            provider=provider,
            entity=entity,
        ).observe(latency_seconds)
    except Exception as e:
        logger.warning(f"Failed to record event latency metric: {e}")


def record_tainted_record(
    provider: str,
    entity: str,
    reason: str,
) -> None:
    """Record a tainted record."""
    try:
        dq_tainted_records_total.labels(
            provider=provider,
            entity=entity,
            reason=reason,
        ).inc()
    except Exception as e:
        logger.warning(f"Failed to record tainted record metric: {e}")


def record_odds_violation(
    provider: str,
    book: str,
    market: str,
    rule: str,
) -> None:
    """Record an odds invariant violation."""
    try:
        dq_odds_invariant_violations_total.labels(
            provider=provider,
            book=book,
            market=market,
            rule=rule,
        ).inc()
    except Exception as e:
        logger.warning(f"Failed to record odds violation metric: {e}")


def record_odds_quarantined(
    provider: str,
    book: str,
    reason: str,
) -> None:
    """Record a quarantined odds record."""
    try:
        dq_odds_quarantined_total.labels(
            provider=provider,
            book=book,
            reason=reason,
        ).inc()
    except Exception as e:
        logger.warning(f"Failed to record quarantined odds metric: {e}")


def record_overround(
    provider: str,
    book: str,
    overround: float,
) -> None:
    """Record overround observation."""
    try:
        dq_odds_overround.labels(
            provider=provider,
            book=book,
        ).observe(overround)
    except Exception as e:
        logger.warning(f"Failed to record overround metric: {e}")


def record_frozen_market_suspect(
    provider: str,
    book: str,
) -> None:
    """Record a frozen market suspect."""
    try:
        dq_frozen_market_suspects_total.labels(
            provider=provider,
            book=book,
        ).inc()
    except Exception as e:
        logger.warning(f"Failed to record frozen market metric: {e}")


def record_unmapped_entity(
    provider: str,
    entity_type: str,
    league: str = "unknown",
) -> None:
    """Record an unmapped entity."""
    try:
        dq_entity_mapping_unmapped_total.labels(
            provider=provider,
            entity_type=entity_type,
            league=league,
        ).inc()
    except Exception as e:
        logger.warning(f"Failed to record unmapped entity metric: {e}")


def set_mapping_coverage(
    provider: str,
    entity_type: str,
    league: str,
    coverage_pct: float,
) -> None:
    """Set the mapping coverage gauge."""
    try:
        dq_entity_mapping_coverage_pct.labels(
            provider=provider,
            entity_type=entity_type,
            league=league,
        ).set(coverage_pct)
    except Exception as e:
        logger.warning(f"Failed to set mapping coverage metric: {e}")


# =============================================================================
# LLM TELEMETRY HELPERS
# =============================================================================

# Pricing per 1M tokens (as of 2026-01)
LLM_PRICING = {
    "gemini": {"input": 0.075, "output": 0.30},  # Gemini 1.5 Flash
    "runpod": {"input": 0.20, "output": 0.20},   # Qwen 32B estimate
}


def record_llm_request(
    provider: str,
    status: str,
    latency_ms: float,
    input_tokens: int = 0,
    output_tokens: int = 0,
) -> None:
    """
    Record a complete LLM request with all metrics.

    Args:
        provider: "gemini" or "runpod"
        status: "ok", "error", "rejected", "skipped"
        latency_ms: End-to-end latency in milliseconds
        input_tokens: Number of input tokens (0 if unknown)
        output_tokens: Number of output tokens (0 if unknown)
    """
    try:
        # Request count
        llm_requests_total.labels(provider=provider, status=status).inc()

        # Latency (only for completed requests)
        if status in ("ok", "rejected") and latency_ms > 0:
            llm_latency_ms.labels(provider=provider).observe(latency_ms)

        # Tokens
        if input_tokens > 0:
            llm_tokens_total.labels(provider=provider, direction="input").inc(input_tokens)
        if output_tokens > 0:
            llm_tokens_total.labels(provider=provider, direction="output").inc(output_tokens)

        # Estimated cost
        if input_tokens > 0 or output_tokens > 0:
            pricing = LLM_PRICING.get(provider, {"input": 0.10, "output": 0.10})
            cost = (input_tokens * pricing["input"] + output_tokens * pricing["output"]) / 1_000_000
            if cost > 0:
                llm_cost_usd.labels(provider=provider).inc(cost)

    except Exception as e:
        logger.warning(f"Failed to record LLM request metric: {e}")


def set_circuit_breaker_state(
    provider: str,
    is_open: bool,
    consecutive_failures: int,
) -> None:
    """
    Update circuit breaker state gauges.

    Args:
        provider: "gemini" or "runpod"
        is_open: True if circuit is open (tripped), False if closed
        consecutive_failures: Current consecutive failure count
    """
    try:
        llm_circuit_open.labels(provider=provider).set(1 if is_open else 0)
        llm_consecutive_failures.labels(provider=provider).set(consecutive_failures)
    except Exception as e:
        logger.warning(f"Failed to set circuit breaker metric: {e}")


# =============================================================================
# AGGREGATES TELEMETRY HELPERS
# =============================================================================


def record_aggregates_refresh(
    status: str,
    duration_ms: float,
    baselines_count: int = 0,
    profiles_count: int = 0,
    leagues_count: int = 0,
    min_sample_ok_pct: float = 0.0,
) -> None:
    """
    Record aggregates refresh job metrics.

    Args:
        status: "ok" or "error"
        duration_ms: Job duration in milliseconds
        baselines_count: Number of baseline rows after refresh
        profiles_count: Number of profile rows after refresh
        leagues_count: Number of distinct leagues
        min_sample_ok_pct: Percentage of profiles with min_sample_ok=true
    """
    try:
        # Job run counter
        aggregates_refresh_runs_total.labels(status=status).inc()

        # Duration histogram
        if duration_ms > 0:
            aggregates_refresh_duration_ms.observe(duration_ms)

        # State gauges (only update on success)
        if status == "ok":
            aggregates_baselines_rows.set(baselines_count)
            aggregates_profiles_rows.set(profiles_count)
            aggregates_leagues_distinct.set(leagues_count)
            aggregates_profiles_min_sample_ok_pct.set(min_sample_ok_pct)

    except Exception as e:
        logger.warning(f"Failed to record aggregates refresh metric: {e}")


def get_metrics_text() -> tuple[str, str]:
    """
    Generate Prometheus metrics text output.

    Returns:
        Tuple of (content, content_type)
    """
    return generate_latest(REGISTRY).decode("utf-8"), CONTENT_TYPE_LATEST
