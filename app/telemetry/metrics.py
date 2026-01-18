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
    "Estimated LLM cost in USD by provider. Formula: (in_tokens * in_rate + out_tokens * out_rate) / 1M",
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
# PREDICTIONS HEALTH METRICS (P1 Alerting)
# =============================================================================

predictions_hours_since_last_saved = Gauge(
    "predictions_hours_since_last_saved",
    "Hours since last prediction was saved to database",
    [],
)

predictions_ns_next_48h = Gauge(
    "predictions_ns_next_48h",
    "Number of NS (not started) matches in next 48 hours",
    [],
)

predictions_ns_missing_next_48h = Gauge(
    "predictions_ns_missing_next_48h",
    "NS matches in next 48h missing predictions",
    [],
)

predictions_coverage_ns_pct = Gauge(
    "predictions_coverage_ns_pct",
    "Prediction coverage percentage for upcoming NS matches (0-100)",
    [],
)

predictions_health_status = Gauge(
    "predictions_health_status",
    "Predictions health status code: 0=ok, 1=warn, 2=red",
    [],
)

predictions_ft_missing_last_48h = Gauge(
    "predictions_ft_missing_last_48h",
    "FT (finished) matches in last 48h that never received a prediction (impact metric)",
    [],
)

predictions_ft_coverage_pct = Gauge(
    "predictions_ft_coverage_pct",
    "Prediction coverage percentage for FT matches in last 48h (0-100)",
    [],
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
# Formula: cost_usd = (input_tokens * input_rate + output_tokens * output_rate) / 1,000,000
# Sources:
#   - Gemini 2.0 Flash: https://ai.google.dev/pricing (Jan 2026)
#   - RunPod/Qwen: Estimate based on serverless GPU pricing
LLM_PRICING = {
    "gemini": {"input": 0.075, "output": 0.30},  # Gemini 2.0 Flash: $0.075/1M in, $0.30/1M out
    "runpod": {"input": 0.20, "output": 0.20},   # Qwen 32B estimate: ~$0.20/1M tokens
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


# =============================================================================
# PREDICTIONS HEALTH TELEMETRY HELPER
# =============================================================================


def set_predictions_health_metrics(
    hours_since_last: Optional[float],
    ns_next_48h: int,
    ns_missing_next_48h: int,
    coverage_ns_pct: float,
    status: str,
    ft_missing_48h: int = 0,
    ft_coverage_pct: float = 100.0,
) -> None:
    """
    Update predictions health gauges for Prometheus/Grafana alerting.

    Args:
        hours_since_last: Hours since last prediction saved (None if never)
        ns_next_48h: NS matches in next 48 hours
        ns_missing_next_48h: NS matches missing predictions
        coverage_ns_pct: Coverage percentage for NS (0-100)
        status: Health status string: "ok", "warn", "red"
        ft_missing_48h: FT matches in last 48h missing predictions (impact metric)
        ft_coverage_pct: Coverage percentage for FT (0-100)
    """
    try:
        # Hours since last (use -1 if None/never to indicate missing data)
        if hours_since_last is not None:
            predictions_hours_since_last_saved.set(hours_since_last)
        else:
            predictions_hours_since_last_saved.set(-1)

        predictions_ns_next_48h.set(ns_next_48h)
        predictions_ns_missing_next_48h.set(ns_missing_next_48h)
        predictions_coverage_ns_pct.set(coverage_ns_pct)

        # FT missing metrics (impact - matches that finished without prediction)
        predictions_ft_missing_last_48h.set(ft_missing_48h)
        predictions_ft_coverage_pct.set(ft_coverage_pct)

        # Status code: ok=0, warn=1, red=2
        status_code = {"ok": 0, "warn": 1, "red": 2}.get(status, 1)
        predictions_health_status.set(status_code)

    except Exception as e:
        logger.warning(f"Failed to set predictions health metrics: {e}")


# =============================================================================
# SHADOW MODE METRICS (A/B Testing)
# =============================================================================

shadow_predictions_logged_total = Counter(
    "shadow_predictions_logged_total",
    "Shadow mode: predictions logged per run",
    [],
)

shadow_predictions_evaluated_total = Counter(
    "shadow_predictions_evaluated_total",
    "Shadow mode: predictions evaluated (matched with FT outcomes)",
    [],
)

shadow_predictions_errors_total = Counter(
    "shadow_predictions_errors_total",
    "Shadow mode: prediction logging errors",
    [],
)

shadow_eval_lag_minutes = Gauge(
    "shadow_eval_lag_minutes",
    "Shadow mode: minutes since oldest pending prediction (0 if none pending)",
    [],
)

shadow_pending_ft_to_evaluate = Gauge(
    "shadow_pending_ft_to_evaluate",
    "Shadow mode: FT matches with pending shadow evaluations",
    [],
)

# =============================================================================
# SENSOR B METRICS (Calibration Diagnostics)
# =============================================================================

sensor_predictions_logged_total = Counter(
    "sensor_predictions_logged_total",
    "Sensor B: predictions logged per run",
    [],
)

sensor_predictions_evaluated_total = Counter(
    "sensor_predictions_evaluated_total",
    "Sensor B: predictions evaluated (matched with FT outcomes)",
    [],
)

sensor_predictions_errors_total = Counter(
    "sensor_predictions_errors_total",
    "Sensor B: prediction logging errors",
    [],
)

sensor_retrain_runs_total = Counter(
    "sensor_retrain_runs_total",
    "Sensor B: retrain job runs",
    ["status"],  # ok, learning, error
)

sensor_eval_lag_minutes = Gauge(
    "sensor_eval_lag_minutes",
    "Sensor B: minutes since oldest pending prediction (0 if none pending)",
    [],
)

sensor_pending_ft_to_evaluate = Gauge(
    "sensor_pending_ft_to_evaluate",
    "Sensor B: FT matches with pending sensor evaluations",
    [],
)

sensor_state = Gauge(
    "sensor_state",
    "Sensor B: current state (0=disabled, 1=learning, 2=ready, 3=error)",
    [],
)


# =============================================================================
# SHADOW MODE TELEMETRY HELPERS
# =============================================================================


def record_shadow_predictions_batch(
    logged: int,
    errors: int,
) -> None:
    """
    Record shadow predictions batch metrics.

    Args:
        logged: Number of predictions successfully logged
        errors: Number of prediction errors
    """
    try:
        if logged > 0:
            shadow_predictions_logged_total.inc(logged)
        if errors > 0:
            shadow_predictions_errors_total.inc(errors)
    except Exception as e:
        logger.warning(f"Failed to record shadow batch metrics: {e}")


def record_shadow_evaluation_batch(
    evaluated: int,
) -> None:
    """
    Record shadow evaluation batch metrics.

    Args:
        evaluated: Number of predictions evaluated
    """
    try:
        if evaluated > 0:
            shadow_predictions_evaluated_total.inc(evaluated)
    except Exception as e:
        logger.warning(f"Failed to record shadow evaluation metrics: {e}")


def set_shadow_health_metrics(
    eval_lag_minutes: float,
    pending_ft: int,
) -> None:
    """
    Update shadow mode health gauges.

    Args:
        eval_lag_minutes: Minutes since oldest pending prediction
        pending_ft: FT matches with pending evaluations
    """
    try:
        shadow_eval_lag_minutes.set(eval_lag_minutes)
        shadow_pending_ft_to_evaluate.set(pending_ft)
    except Exception as e:
        logger.warning(f"Failed to set shadow health metrics: {e}")


# =============================================================================
# SENSOR B TELEMETRY HELPERS
# =============================================================================


def record_sensor_predictions_batch(
    logged: int,
    errors: int,
) -> None:
    """
    Record sensor predictions batch metrics.

    Args:
        logged: Number of predictions successfully logged
        errors: Number of prediction errors
    """
    try:
        if logged > 0:
            sensor_predictions_logged_total.inc(logged)
        if errors > 0:
            sensor_predictions_errors_total.inc(errors)
    except Exception as e:
        logger.warning(f"Failed to record sensor batch metrics: {e}")


def record_sensor_evaluation_batch(
    evaluated: int,
) -> None:
    """
    Record sensor evaluation batch metrics.

    Args:
        evaluated: Number of predictions evaluated
    """
    try:
        if evaluated > 0:
            sensor_predictions_evaluated_total.inc(evaluated)
    except Exception as e:
        logger.warning(f"Failed to record sensor evaluation metrics: {e}")


def record_sensor_retrain(
    status: str,
) -> None:
    """
    Record sensor retrain job run.

    Args:
        status: "ok", "learning", or "error"
    """
    try:
        sensor_retrain_runs_total.labels(status=status).inc()
    except Exception as e:
        logger.warning(f"Failed to record sensor retrain metrics: {e}")


def set_sensor_health_metrics(
    eval_lag_minutes: float,
    pending_ft: int,
    state: str,
) -> None:
    """
    Update sensor B health gauges.

    Args:
        eval_lag_minutes: Minutes since oldest pending prediction
        pending_ft: FT matches with pending evaluations
        state: "disabled", "learning", "ready", "error"
    """
    try:
        sensor_eval_lag_minutes.set(eval_lag_minutes)
        sensor_pending_ft_to_evaluate.set(pending_ft)

        # State code: disabled=0, learning=1, ready=2, error=3
        state_code = {"disabled": 0, "learning": 1, "ready": 2, "error": 3}.get(state, 1)
        sensor_state.set(state_code)
    except Exception as e:
        logger.warning(f"Failed to set sensor health metrics: {e}")


# =============================================================================
# ODDS SYNC METRICS
# =============================================================================

odds_sync_requests_total = Counter(
    "odds_sync_requests_total",
    "Odds sync: API requests by status",
    ["status"],  # ok, empty, error, rate_limited
)

odds_sync_fixtures_scanned_total = Counter(
    "odds_sync_fixtures_scanned_total",
    "Odds sync: total fixtures scanned",
    [],
)

odds_sync_fixtures_updated_total = Counter(
    "odds_sync_fixtures_updated_total",
    "Odds sync: fixtures successfully updated with odds",
    [],
)

odds_sync_payload_bytes_total = Counter(
    "odds_sync_payload_bytes_total",
    "Odds sync: total bytes received from API (for monitoring large responses)",
    [],
)

odds_sync_runs_total = Counter(
    "odds_sync_runs_total",
    "Odds sync: job runs by status",
    ["status"],  # ok, error, disabled
)

odds_sync_duration_ms = Histogram(
    "odds_sync_duration_ms",
    "Odds sync: job duration in milliseconds",
    [],
    buckets=[1000, 5000, 10000, 30000, 60000, 120000, 300000],
)

odds_coverage_ns_pct = Gauge(
    "odds_coverage_ns_pct",
    "Odds coverage: percentage of NS matches in 48h window with odds",
    ["region"],  # latam, europe, other
)


# =============================================================================
# ODDS SYNC TELEMETRY HELPERS
# =============================================================================


def record_odds_sync_request(
    status: str,
    payload_bytes: int = 0,
) -> None:
    """
    Record an odds sync API request.

    Args:
        status: "ok", "empty", "error", "rate_limited"
        payload_bytes: Response size in bytes (for monitoring)
    """
    try:
        odds_sync_requests_total.labels(status=status).inc()
        if payload_bytes > 0:
            odds_sync_payload_bytes_total.inc(payload_bytes)
    except Exception as e:
        logger.warning(f"Failed to record odds sync request metric: {e}")


def record_odds_sync_batch(
    scanned: int,
    updated: int,
) -> None:
    """
    Record odds sync batch metrics.

    Args:
        scanned: Number of fixtures scanned
        updated: Number of fixtures updated with odds
    """
    try:
        if scanned > 0:
            odds_sync_fixtures_scanned_total.inc(scanned)
        if updated > 0:
            odds_sync_fixtures_updated_total.inc(updated)
    except Exception as e:
        logger.warning(f"Failed to record odds sync batch metrics: {e}")


def record_odds_sync_run(
    status: str,
    duration_ms: float,
) -> None:
    """
    Record odds sync job run.

    Args:
        status: "ok", "error", "disabled"
        duration_ms: Job duration in milliseconds
    """
    try:
        odds_sync_runs_total.labels(status=status).inc()
        if duration_ms > 0:
            odds_sync_duration_ms.observe(duration_ms)
    except Exception as e:
        logger.warning(f"Failed to record odds sync run metrics: {e}")


def set_odds_coverage_metrics(
    latam_pct: float,
    europe_pct: float,
    other_pct: float,
) -> None:
    """
    Update odds coverage gauges by region.

    Args:
        latam_pct: Coverage percentage for LATAM leagues (0-100)
        europe_pct: Coverage percentage for European leagues (0-100)
        other_pct: Coverage percentage for other leagues (0-100)
    """
    try:
        odds_coverage_ns_pct.labels(region="latam").set(latam_pct)
        odds_coverage_ns_pct.labels(region="europe").set(europe_pct)
        odds_coverage_ns_pct.labels(region="other").set(other_pct)
    except Exception as e:
        logger.warning(f"Failed to set odds coverage metrics: {e}")


# =============================================================================
# RERUN SERVING METRICS
# =============================================================================

rerun_serving_db_hits_total = Counter(
    "rerun_serving_db_hits_total",
    "Rerun serving: predictions served from DB (two-stage)",
    [],
)

rerun_serving_db_stale_total = Counter(
    "rerun_serving_db_stale_total",
    "Rerun serving: DB predictions rejected as stale",
    [],
)

rerun_serving_live_fallback_total = Counter(
    "rerun_serving_live_fallback_total",
    "Rerun serving: fell back to live baseline",
    [],
)

rerun_serving_ns_total = Counter(
    "rerun_serving_ns_total",
    "Rerun serving: total NS matches processed",
    [],
)

rerun_serving_enabled = Gauge(
    "rerun_serving_enabled",
    "Rerun serving: 1 if PREFER_RERUN_PREDICTIONS is enabled, 0 otherwise",
    [],
)


def record_rerun_serving_batch(
    db_hits: int,
    db_stale: int,
    live_fallback: int,
    total_ns: int,
) -> None:
    """
    Record rerun serving metrics for a batch of predictions.

    Args:
        db_hits: Predictions served from DB (two-stage)
        db_stale: DB predictions rejected as stale
        live_fallback: Predictions served from live baseline
        total_ns: Total NS matches in batch
    """
    try:
        if db_hits > 0:
            rerun_serving_db_hits_total.inc(db_hits)
        if db_stale > 0:
            rerun_serving_db_stale_total.inc(db_stale)
        if live_fallback > 0:
            rerun_serving_live_fallback_total.inc(live_fallback)
        if total_ns > 0:
            rerun_serving_ns_total.inc(total_ns)
    except Exception as e:
        logger.warning(f"Failed to record rerun serving metrics: {e}")


def set_rerun_serving_enabled(enabled: bool) -> None:
    """Set the rerun serving enabled gauge."""
    try:
        rerun_serving_enabled.set(1 if enabled else 0)
    except Exception as e:
        logger.warning(f"Failed to set rerun serving enabled metric: {e}")


# =============================================================================
# GENERIC JOB HEALTH METRICS (P0 Jobs)
# =============================================================================

job_runs_total = Counter(
    "job_runs_total",
    "Total job runs by job and status",
    ["job", "status"],  # job: stats_backfill, odds_sync, fastpath; status: ok, error, rate_limited, budget_exceeded
)

job_last_success_timestamp = Gauge(
    "job_last_success_timestamp",
    "Unix timestamp of last successful job run",
    ["job"],
)

job_duration_ms = Histogram(
    "job_duration_ms",
    "Job duration in milliseconds",
    ["job"],
    buckets=[100, 500, 1000, 5000, 10000, 30000, 60000, 120000, 300000, 600000],
)


# =============================================================================
# STATS BACKFILL METRICS
# =============================================================================

stats_backfill_rows_updated_total = Counter(
    "stats_backfill_rows_updated_total",
    "Stats backfill: matches updated with stats",
    [],
)

stats_backfill_ft_pending_gauge = Gauge(
    "stats_backfill_ft_pending_gauge",
    "Stats backfill: FT/AET/PEN matches in lookback window without stats",
    [],
)

stats_last_captured_at_timestamp = Gauge(
    "stats_last_captured_at_timestamp",
    "Unix timestamp of last stats capture (freshness indicator)",
    [],
)


# =============================================================================
# FASTPATH METRICS
# =============================================================================

fastpath_ticks_total = Counter(
    "fastpath_ticks_total",
    "FastPath: tick executions by status",
    ["status"],  # ok, error
)

fastpath_completed_total = Counter(
    "fastpath_completed_total",
    "FastPath: narratives completed by status",
    ["status"],  # ok, rejected, error
)

fastpath_backlog_ready_gauge = Gauge(
    "fastpath_backlog_ready_gauge",
    "FastPath: audits ready for narrative generation",
    [],
)

fastpath_last_success_timestamp = Gauge(
    "fastpath_last_success_timestamp",
    "Unix timestamp of last successful fastpath tick",
    [],
)


# =============================================================================
# JOB HEALTH TELEMETRY HELPERS
# =============================================================================


def record_job_run(
    job: str,
    status: str,
    duration_ms: float,
    error: str = None,
    metrics: dict = None,
) -> None:
    """
    Record a job run with status and duration.

    Args:
        job: Job identifier (stats_backfill, odds_sync, fastpath)
        status: "ok", "error", "rate_limited", "budget_exceeded"
        duration_ms: Job duration in milliseconds
        error: Optional error message for failed runs
        metrics: Optional job-specific metrics dict
    """
    # 1) Record to Prometheus (always)
    try:
        job_runs_total.labels(job=job, status=status).inc()
        if duration_ms > 0:
            job_duration_ms.labels(job=job).observe(duration_ms)
        if status == "ok":
            job_last_success_timestamp.labels(job=job).set(time.time())
    except Exception as e:
        logger.warning(f"Failed to record job run metric: {e}")

    # 2) P1-B: Also persist to DB (for cold-start fallback)
    # Run in background to avoid blocking
    try:
        import asyncio
        from datetime import datetime, timedelta

        # Calculate started_at from duration
        finished_at = datetime.utcnow()
        started_at = finished_at - timedelta(milliseconds=duration_ms)

        async def _persist_to_db():
            try:
                from app.database import AsyncSessionLocal
                from app.models import JobRun

                async with AsyncSessionLocal() as session:
                    job_run = JobRun(
                        job_name=job,
                        status=status,
                        started_at=started_at,
                        finished_at=finished_at,
                        duration_ms=int(duration_ms),
                        error_message=error,
                        metrics=metrics,
                    )
                    session.add(job_run)
                    await session.commit()
            except Exception as db_err:
                logger.debug(f"[JOB_TRACKING] DB persist failed (non-blocking): {db_err}")

        # Try to get running loop, if exists schedule in background
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(_persist_to_db())
        except RuntimeError:
            # No running loop - skip DB persist (startup/tests)
            pass
    except Exception as e:
        logger.debug(f"[JOB_TRACKING] Failed to schedule DB persist: {e}")


def set_job_last_success(job: str) -> None:
    """Set the last success timestamp for a job."""
    try:
        job_last_success_timestamp.labels(job=job).set(time.time())
    except Exception as e:
        logger.warning(f"Failed to set job last success metric: {e}")


# =============================================================================
# STATS BACKFILL TELEMETRY HELPERS
# =============================================================================


def record_stats_backfill_result(
    rows_updated: int,
    ft_pending: int,
) -> None:
    """
    Record stats backfill result metrics.

    Args:
        rows_updated: Number of matches updated with stats
        ft_pending: Number of FT/AET/PEN matches still without stats
    """
    try:
        if rows_updated > 0:
            stats_backfill_rows_updated_total.inc(rows_updated)
            stats_last_captured_at_timestamp.set(time.time())
        stats_backfill_ft_pending_gauge.set(ft_pending)
    except Exception as e:
        logger.warning(f"Failed to record stats backfill metrics: {e}")


def set_stats_backfill_pending(ft_pending: int) -> None:
    """Set the pending FT matches gauge."""
    try:
        stats_backfill_ft_pending_gauge.set(ft_pending)
    except Exception as e:
        logger.warning(f"Failed to set stats backfill pending metric: {e}")


# =============================================================================
# FASTPATH TELEMETRY HELPERS
# =============================================================================


def record_fastpath_tick(
    status: str,
    completed_ok: int = 0,
    completed_rejected: int = 0,
    completed_error: int = 0,
    backlog_ready: int = 0,
) -> None:
    """
    Record fastpath tick metrics.

    Args:
        status: "ok" or "error"
        completed_ok: Narratives completed successfully
        completed_rejected: Narratives rejected by validation
        completed_error: Narratives failed with error
        backlog_ready: Audits ready for narrative generation
    """
    try:
        fastpath_ticks_total.labels(status=status).inc()
        if status == "ok":
            fastpath_last_success_timestamp.set(time.time())

        if completed_ok > 0:
            fastpath_completed_total.labels(status="ok").inc(completed_ok)
        if completed_rejected > 0:
            fastpath_completed_total.labels(status="rejected").inc(completed_rejected)
        if completed_error > 0:
            fastpath_completed_total.labels(status="error").inc(completed_error)

        fastpath_backlog_ready_gauge.set(backlog_ready)
    except Exception as e:
        logger.warning(f"Failed to record fastpath tick metrics: {e}")


def set_fastpath_backlog(backlog_ready: int) -> None:
    """Set the fastpath backlog gauge."""
    try:
        fastpath_backlog_ready_gauge.set(backlog_ready)
    except Exception as e:
        logger.warning(f"Failed to set fastpath backlog metric: {e}")


# =============================================================================
# LIVE SUMMARY METRICS (iOS Live Score Polling)
# =============================================================================

live_summary_requests_total = Counter(
    "live_summary_requests_total",
    "Live summary: requests by status",
    ["status"],  # ok, error
)

live_summary_latency_ms = Histogram(
    "live_summary_latency_ms",
    "Live summary: endpoint latency in milliseconds",
    [],
    buckets=[1, 5, 10, 25, 50, 100, 250, 500],
)

live_summary_matches_count = Gauge(
    "live_summary_matches_count",
    "Live summary: number of live matches in response",
    [],
)


# =============================================================================
# LIVE SUMMARY TELEMETRY HELPERS
# =============================================================================


def record_live_summary_request(
    status: str,
    latency_ms: float,
    matches_count: int = 0,
) -> None:
    """
    Record a live summary request.

    Args:
        status: "ok" or "error"
        latency_ms: Request latency in milliseconds
        matches_count: Number of live matches returned
    """
    try:
        live_summary_requests_total.labels(status=status).inc()
        if latency_ms > 0:
            live_summary_latency_ms.observe(latency_ms)
        live_summary_matches_count.set(matches_count)
    except Exception as e:
        logger.warning(f"Failed to record live summary metrics: {e}")


def get_metrics_text() -> tuple[str, str]:
    """
    Generate Prometheus metrics text output.

    Returns:
        Tuple of (content, content_type)
    """
    return generate_latest(REGISTRY).decode("utf-8"), CONTENT_TYPE_LATEST
