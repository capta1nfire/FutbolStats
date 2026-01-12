# Grafana Alerts Checklist - Data Quality P0

Alertas mínimas recomendadas basadas en los thresholds de `TelemetryConfig`.

## Provider Availability (RED/WARN)

```yaml
# Alert: Provider Success Rate WARN
expr: |
  (
    sum(rate(dq_provider_requests_total{status_code="200"}[5m])) by (provider)
    /
    sum(rate(dq_provider_requests_total[5m])) by (provider)
  ) < 0.98
for: 5m
severity: warning
summary: "Provider {{ $labels.provider }} success rate below 98%"

# Alert: Provider Success Rate RED
expr: |
  (
    sum(rate(dq_provider_requests_total{status_code="200"}[5m])) by (provider)
    /
    sum(rate(dq_provider_requests_total[5m])) by (provider)
  ) < 0.95
for: 5m
severity: critical
summary: "Provider {{ $labels.provider }} success rate below 95%"
```

## Rate Limiting

```yaml
# Alert: Rate Limited Requests
expr: |
  sum(rate(dq_provider_rate_limited_total[5m])) by (provider) > 0
for: 2m
severity: warning
summary: "Provider {{ $labels.provider }} being rate limited"
```

## Timeouts

```yaml
# Alert: Provider Timeouts
expr: |
  sum(rate(dq_provider_timeouts_total[5m])) by (provider) > 0.1
for: 5m
severity: warning
summary: "Provider {{ $labels.provider }} experiencing timeouts"
```

## Latency

```yaml
# Alert: High Latency P95
expr: |
  histogram_quantile(0.95,
    sum(rate(dq_provider_latency_ms_bucket[5m])) by (provider, le)
  ) > 5000
for: 5m
severity: warning
summary: "Provider {{ $labels.provider }} P95 latency > 5s"
```

## Market Integrity

```yaml
# Alert: Odds Violations Spike
expr: |
  sum(rate(dq_odds_invariant_violations_total[5m])) by (provider, rule) > 1
for: 5m
severity: warning
summary: "High odds violations rate: {{ $labels.rule }} from {{ $labels.provider }}"

# Alert: High Quarantine Rate
expr: |
  sum(rate(dq_odds_quarantined_total[5m])) by (provider)
  /
  sum(rate(dq_provider_requests_total{entity="odds"}[5m])) by (provider)
  > 0.05
for: 10m
severity: warning
summary: "Provider {{ $labels.provider }} quarantine rate > 5%"
```

## Anti-Lookahead

```yaml
# Alert: Event Latency P95 WARN
expr: |
  histogram_quantile(0.95,
    sum(rate(dq_event_latency_seconds_bucket[5m])) by (provider, le)
  ) > 30
for: 5m
severity: warning
summary: "Event latency P95 > 30s for {{ $labels.provider }}"

# Alert: Event Latency P95 RED
expr: |
  histogram_quantile(0.95,
    sum(rate(dq_event_latency_seconds_bucket[5m])) by (provider, le)
  ) > 90
for: 5m
severity: critical
summary: "Event latency P95 > 90s for {{ $labels.provider }}"

# Alert: Tainted Records
expr: |
  sum(rate(dq_tainted_records_total[5m])) by (provider, reason) > 0
for: 5m
severity: warning
summary: "Tainted records detected: {{ $labels.reason }}"
```

## Entity Mapping

```yaml
# Alert: Unmapped Entities
expr: |
  sum(rate(dq_entity_mapping_unmapped_total[1h])) by (provider, entity_type) > 10
for: 30m
severity: warning
summary: "High unmapped {{ $labels.entity_type }} rate from {{ $labels.provider }}"

# Alert: Low Mapping Coverage
expr: |
  dq_entity_mapping_coverage_pct < 0.99
for: 1h
severity: warning
summary: "Mapping coverage below 99% for {{ $labels.entity_type }}"
```

## Dashboard Panels Sugeridos

1. **Provider Health Overview**
   - Success rate by provider (gauge)
   - Request rate by entity (time series)
   - Error breakdown by error_code (pie)

2. **Latency Distribution**
   - P50/P95/P99 by provider (time series)
   - Latency heatmap by endpoint

3. **Market Integrity**
   - Overround distribution (histogram)
   - Quarantine rate trend (time series)
   - Violation breakdown by rule (bar)

4. **Anti-Lookahead**
   - Event latency distribution (histogram)
   - Tainted records by reason (time series)

## Thresholds Reference (from TelemetryConfig)

| Metric | WARN | RED |
|--------|------|-----|
| Success Rate | < 98% | < 95% |
| Event Lag P95 | > 30s | > 90s |
| Overround 1X2 | < 1.01 or > 1.20 | - |
| Mapping Coverage | < 99.5% | < 99% |
| Frozen Market | > 8 min | > 15 min |
