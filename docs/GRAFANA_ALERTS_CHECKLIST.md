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

## P0 Jobs Health (Scheduler Jobs)

### Stats Backfill Job

```yaml
# Alert: Stats Backfill Not Running
# The job runs hourly. Missing 2+ runs indicates failure.
expr: |
  (time() - job_last_success_timestamp{job="stats_backfill"}) / 60 > 120
for: 10m
severity: warning
summary: "Stats backfill job not running for 2+ hours"

# Alert: Stats Backfill Critical
expr: |
  (time() - job_last_success_timestamp{job="stats_backfill"}) / 60 > 180
for: 10m
severity: critical
summary: "Stats backfill job not running for 3+ hours"

# Alert: High FT Pending Without Stats
expr: |
  stats_backfill_ft_pending_gauge > 5
for: 2h
severity: warning
summary: "{{ $value }} finished matches without stats for 2+ hours"

# Alert: Critical FT Pending Without Stats
expr: |
  stats_backfill_ft_pending_gauge > 10
for: 3h
severity: critical
summary: "{{ $value }} finished matches without stats for 3+ hours"
```

### Odds Sync Job

```yaml
# Alert: Odds Sync Not Running
# The job runs every 6 hours. Missing run indicates failure.
expr: |
  (time() - job_last_success_timestamp{job="odds_sync"}) / 60 > 720
for: 10m
severity: warning
summary: "Odds sync job not running for 12+ hours"

# Alert: Odds Sync Critical
expr: |
  (time() - job_last_success_timestamp{job="odds_sync"}) / 60 > 1080
for: 10m
severity: critical
summary: "Odds sync job not running for 18+ hours"

# Alert: Odds Sync Rate Limited
expr: |
  increase(job_runs_total{job="odds_sync", status="rate_limited"}[1h]) > 2
for: 5m
severity: warning
summary: "Odds sync hitting rate limits"

# Alert: Odds Sync Budget Exceeded
expr: |
  increase(job_runs_total{job="odds_sync", status="budget_exceeded"}[6h]) > 0
for: 5m
severity: critical
summary: "Odds sync stopped due to API budget exceeded"
```

### FastPath LLM Narratives Job

```yaml
# Alert: FastPath Not Running
# The job runs every 2 minutes. Missing 5+ minutes indicates failure.
expr: |
  (time() - job_last_success_timestamp{job="fastpath"}) / 60 > 5
for: 5m
severity: warning
summary: "FastPath job not running for 5+ minutes"

# Alert: FastPath Critical
expr: |
  (time() - job_last_success_timestamp{job="fastpath"}) / 60 > 10
for: 5m
severity: critical
summary: "FastPath job not running for 10+ minutes"

# Alert: FastPath Backlog Growing
# Audits ready for narrative generation but not being processed.
expr: |
  fastpath_backlog_ready_gauge > 3
for: 15m
severity: warning
summary: "{{ $value }} audits waiting for FastPath narratives"

# Alert: FastPath Backlog Critical
expr: |
  fastpath_backlog_ready_gauge > 5
for: 30m
severity: critical
summary: "{{ $value }} audits stuck in FastPath backlog"

# Alert: FastPath Errors Spike
expr: |
  rate(fastpath_ticks_total{status="error"}[5m]) > 0.1
for: 10m
severity: warning
summary: "FastPath experiencing errors"
```

### Generic Job Monitoring

```yaml
# Alert: Any P0 Job Failing
expr: |
  increase(job_runs_total{job=~"stats_backfill|odds_sync|fastpath", status="error"}[1h]) > 3
for: 5m
severity: warning
summary: "Job {{ $labels.job }} failing repeatedly"
```

## Thresholds Reference (from TelemetryConfig)

| Metric | WARN | RED |
|--------|------|-----|
| Success Rate | < 98% | < 95% |
| Event Lag P95 | > 30s | > 90s |
| Overround 1X2 | < 1.01 or > 1.20 | - |
| Mapping Coverage | < 99.5% | < 99% |
| Frozen Market | > 8 min | > 15 min |

## P0 Jobs Thresholds Reference

| Job | Metric | WARN | RED |
|-----|--------|------|-----|
| stats_backfill | minutes_since_success | > 120 | > 180 |
| stats_backfill | ft_pending | > 5 (2h) | > 10 (3h) |
| odds_sync | minutes_since_success | > 720 | > 1080 |
| fastpath | minutes_since_success | > 5 | > 10 |
| fastpath | backlog_ready | > 3 (15m) | > 5 (30m) |
