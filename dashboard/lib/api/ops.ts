/**
 * OPS API Adapter
 *
 * Safe extraction and adaptation of data from /dashboard/ops.json
 * Designed to be resilient to partial or malformed responses.
 */

import {
  ApiBudget,
  ApiBudgetStatus,
  HealthSummary,
  HealthCard,
  HealthStatus,
  OverviewCounts,
} from "@/lib/types";

/**
 * Raw response type - intentionally loose since backend schema may evolve
 */
export type OpsResponse = unknown;

/**
 * Safely check if value is a non-null object
 */
function isObject(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

/**
 * Safely get nested property
 */
function getNestedValue(obj: unknown, ...keys: string[]): unknown {
  let current = obj;
  for (const key of keys) {
    if (!isObject(current)) return undefined;
    current = current[key];
  }
  return current;
}

/**
 * Extract budget object from ops response
 *
 * Expected structure: { data: { budget: {...} } }
 */
export function extractBudget(ops: OpsResponse): unknown | null {
  if (!isObject(ops)) return null;

  const budget = getNestedValue(ops, "data", "budget");
  if (!isObject(budget)) return null;

  return budget;
}

/**
 * Normalize backend status to ApiBudgetStatus
 *
 * Backend uses "warn" but frontend expects "warning".
 * Also handles "red" -> "critical" mapping.
 */
function normalizeStatus(status: unknown): ApiBudgetStatus | null {
  if (status === "ok") return "ok";
  if (status === "warning" || status === "warn") return "warning";
  if (status === "critical" || status === "red") return "critical";
  if (status === "degraded") return "degraded";
  return null;
}


/**
 * Adapt raw budget object to ApiBudget type
 *
 * Only assumes these fields are stable (confirmed in spec):
 * - status, plan, plan_end, active
 * - requests_today, requests_limit, requests_remaining
 * - cached, cache_age_seconds
 * - tokens_reset_tz, tokens_reset_local_time, tokens_reset_at_la, tokens_reset_at_utc, tokens_reset_note
 *
 * Returns null if critical fields are missing or have wrong types.
 */
export function adaptApiBudget(raw: unknown): ApiBudget | null {
  if (!isObject(raw)) return null;

  // Required fields with type validation (normalize "warn" -> "warning")
  const status = normalizeStatus(raw.status);
  if (status === null) return null;

  const plan = raw.plan;
  if (typeof plan !== "string") return null;

  const active = raw.active;
  if (typeof active !== "boolean") return null;

  const requests_today = raw.requests_today;
  if (typeof requests_today !== "number") return null;

  const requests_limit = raw.requests_limit;
  if (typeof requests_limit !== "number") return null;

  const requests_remaining = raw.requests_remaining;
  if (typeof requests_remaining !== "number") return null;

  // Optional but expected fields with defaults
  const cached = typeof raw.cached === "boolean" ? raw.cached : false;
  const cache_age_seconds =
    typeof raw.cache_age_seconds === "number" ? raw.cache_age_seconds : 0;

  // Optional string fields
  const plan_end =
    typeof raw.plan_end === "string" ? raw.plan_end : undefined;
  const tokens_reset_at_la =
    typeof raw.tokens_reset_at_la === "string"
      ? raw.tokens_reset_at_la
      : undefined;
  const tokens_reset_note =
    typeof raw.tokens_reset_note === "string"
      ? raw.tokens_reset_note
      : undefined;

  return {
    status,
    plan,
    plan_end,
    active,
    requests_today,
    requests_limit,
    requests_remaining,
    cached,
    cache_age_seconds,
    tokens_reset_at_la,
    tokens_reset_note,
  };
}

/**
 * Combined extraction and adaptation
 *
 * Returns ApiBudget if successful, null otherwise
 */
export function parseOpsBudget(ops: OpsResponse): ApiBudget | null {
  const rawBudget = extractBudget(ops);
  if (!rawBudget) return null;
  return adaptApiBudget(rawBudget);
}

// ============================================================================
// Health Summary Extraction (best-effort)
// ============================================================================

/**
 * Map backend status to frontend HealthStatus
 */
function mapHealthStatus(status: unknown): HealthStatus {
  if (status === "ok") return "healthy";
  if (status === "warning") return "warning";
  if (status === "critical" || status === "red") return "critical";
  return "healthy"; // default fallback
}

/**
 * Extract health-related fields from ops response
 *
 * Expected structure:
 * {
 *   data: {
 *     predictions_health: { status, ns_coverage_pct, ns_matches_next_48h_missing_prediction, ... },
 *     jobs_health: { status, stats_backfill, odds_sync, fastpath, ... },
 *     live_summary: { cached_live_matches },
 *     generated_at: "ISO string"
 *   }
 * }
 */
export function extractHealth(ops: OpsResponse): {
  predictionsHealth: Record<string, unknown> | null;
  jobsHealth: Record<string, unknown> | null;
  liveSummary: Record<string, unknown> | null;
  generatedAt: string | null;
} {
  const predictionsHealth = getNestedValue(ops, "data", "predictions_health");
  const jobsHealth = getNestedValue(ops, "data", "jobs_health");
  const liveSummary = getNestedValue(ops, "data", "live_summary");
  const generatedAt = getNestedValue(ops, "data", "generated_at");

  return {
    predictionsHealth: isObject(predictionsHealth) ? predictionsHealth : null,
    jobsHealth: isObject(jobsHealth) ? jobsHealth : null,
    liveSummary: isObject(liveSummary) ? liveSummary : null,
    generatedAt: typeof generatedAt === "string" ? generatedAt : null,
  };
}

/**
 * Build System health card
 *
 * Backend doesn't have explicit "system" status, so we derive:
 * - healthy: all jobs healthy
 * - warning: any job has warning
 * - critical: any job has critical status
 */
function buildSystemCard(jobsHealth: Record<string, unknown> | null): HealthCard {
  const status = jobsHealth?.status;
  const healthStatus = mapHealthStatus(status);

  return {
    id: "system",
    title: "System",
    status: healthStatus,
    value: healthStatus === "healthy" ? "OK" : healthStatus === "warning" ? "Warn" : "Error",
    subtitle: healthStatus === "healthy" ? "All services operational" : "Check jobs health",
    trend: "stable",
  };
}

/**
 * Build Predictions health card
 */
function buildPredictionsCard(
  predictionsHealth: Record<string, unknown> | null
): HealthCard {
  const status = predictionsHealth?.status;
  const healthStatus = mapHealthStatus(status);

  const coveragePct = predictionsHealth?.ns_coverage_pct;
  const missing = predictionsHealth?.ns_matches_next_48h_missing_prediction;

  const value =
    typeof coveragePct === "number" ? `${Math.round(coveragePct)}%` : "—";

  let subtitle = "Coverage OK";
  if (typeof missing === "number" && missing > 0) {
    subtitle = `${missing} match${missing === 1 ? "" : "es"} missing`;
  }

  return {
    id: "predictions",
    title: "Predictions",
    status: healthStatus,
    value,
    subtitle,
    trend: healthStatus === "healthy" ? "stable" : "down",
  };
}

/**
 * Build Jobs health card
 */
function buildJobsCard(jobsHealth: Record<string, unknown> | null): HealthCard {
  const status = jobsHealth?.status;
  const healthStatus = mapHealthStatus(status);

  // Count running jobs (we don't have real-time running count, so show job statuses)
  let runningCount = 0;
  const statsBackfill = jobsHealth?.stats_backfill;
  const oddsSync = jobsHealth?.odds_sync;
  const fastpath = jobsHealth?.fastpath;

  // Count healthy jobs
  if (isObject(statsBackfill) && statsBackfill.status === "ok") runningCount++;
  if (isObject(oddsSync) && oddsSync.status === "ok") runningCount++;
  if (isObject(fastpath) && fastpath.status === "ok") runningCount++;

  return {
    id: "jobs",
    title: "Jobs",
    status: healthStatus,
    value: `${runningCount}/3`,
    subtitle: healthStatus === "healthy" ? "All jobs healthy" : "Check job status",
    trend: "stable",
  };
}

/**
 * Build Live health card
 */
function buildLiveCard(liveSummary: Record<string, unknown> | null): HealthCard {
  const liveCount = liveSummary?.cached_live_matches;
  const value = typeof liveCount === "number" ? liveCount : 0;

  return {
    id: "live",
    title: "Live",
    status: "healthy",
    value,
    subtitle: value === 0 ? "No live matches" : `Match${value === 1 ? "" : "es"} in progress`,
    trend: value > 0 ? "up" : "stable",
  };
}

/**
 * Build OverviewCounts from backend data (best-effort)
 */
function buildOverviewCounts(
  predictionsHealth: Record<string, unknown> | null,
  liveSummary: Record<string, unknown> | null
): OverviewCounts {
  const liveMatches = liveSummary?.cached_live_matches;
  const predMissing = predictionsHealth?.ns_matches_next_48h_missing_prediction;
  const predTotal = predictionsHealth?.ns_matches_next_48h;

  return {
    matchesLive: typeof liveMatches === "number" ? liveMatches : 0,
    matchesScheduledToday: 0, // Not available from backend
    incidentsActive: 0, // Not available from this endpoint
    incidentsCritical: 0,
    jobsRunning: 0, // Not real-time
    jobsFailedLast24h: 0,
    predictionsMissing: typeof predMissing === "number" ? predMissing : 0,
    predictionsTotal: typeof predTotal === "number" ? predTotal : 0,
  };
}

/**
 * Adapt extracted health data to HealthSummary type
 *
 * Returns null only if all critical data is missing.
 * Individual missing pieces get defaults (best-effort).
 */
export function adaptHealthSummary(ops: OpsResponse): HealthSummary | null {
  const { predictionsHealth, jobsHealth, liveSummary, generatedAt } =
    extractHealth(ops);

  // If we have nothing, return null
  if (!predictionsHealth && !jobsHealth && !liveSummary && !generatedAt) {
    return null;
  }

  // Build health cards
  const cards: HealthCard[] = [
    buildSystemCard(jobsHealth),
    buildPredictionsCard(predictionsHealth),
    buildJobsCard(jobsHealth),
    buildLiveCard(liveSummary),
  ];

  // Coverage percentage from predictions
  const coveragePct =
    typeof predictionsHealth?.ns_coverage_pct === "number"
      ? predictionsHealth.ns_coverage_pct
      : 100;

  // Build counts
  const counts = buildOverviewCounts(predictionsHealth, liveSummary);

  return {
    coveragePct,
    counts,
    cards,
    lastUpdated: generatedAt || new Date().toISOString(),
  };
}

/**
 * Combined extraction and adaptation for health
 *
 * Returns HealthSummary if successful, null otherwise
 */
export function parseOpsHealth(ops: OpsResponse): HealthSummary | null {
  return adaptHealthSummary(ops);
}

// ============================================================================
// Sentry Summary Extraction (best-effort)
// ============================================================================

/**
 * Sentry issue level
 */
export type SentryIssueLevel = "error" | "warning" | "info";

/**
 * Sentry top issue (normalized)
 * Accepts both old shape (count_24h, level) and new shape (count)
 */
export interface SentryTopIssue {
  title: string;
  count_24h: number;
  level?: SentryIssueLevel;
}

/**
 * Sentry summary from ops.json
 * Flexible to support both old and new backend contracts
 */
export interface OpsSentrySummary {
  status: ApiBudgetStatus; // ok | warning | critical | degraded
  cached?: boolean;
  cache_age_seconds?: number;
  generated_at?: string;
  project: {
    org_slug: string;
    project_slug: string;
    env: string;
  };
  counts: {
    new_issues_1h: number;
    new_issues_24h: number;
    active_issues_1h?: number;
    active_issues_24h?: number;
    open_issues: number;
  };
  last_event_at?: string;
  top_issues?: SentryTopIssue[];
  note?: string;
}

/**
 * Validate SentryIssueLevel
 */
function isValidSentryLevel(level: unknown): level is SentryIssueLevel {
  return level === "error" || level === "warning" || level === "info";
}

/**
 * Extract sentry object from ops response
 *
 * Expected structure: { data: { sentry: {...} } }
 */
export function extractSentry(ops: OpsResponse): unknown | null {
  if (!isObject(ops)) return null;

  const sentry = getNestedValue(ops, "data", "sentry");
  if (!isObject(sentry)) return null;

  return sentry;
}

/**
 * Adapt raw sentry object to OpsSentrySummary type
 *
 * Returns null only if critical fields (status, project, counts.open_issues) are missing.
 * Other fields default to 0 or undefined (best-effort).
 */
export function adaptOpsSentry(raw: unknown): OpsSentrySummary | null {
  if (!isObject(raw)) return null;

  // Required: status (normalize "warn" -> "warning")
  const status = normalizeStatus(raw.status);
  if (status === null) return null;

  // Required: project
  const project = raw.project;
  if (!isObject(project)) return null;
  const org_slug = project.org_slug;
  const project_slug = project.project_slug;
  const env = project.env;
  if (typeof org_slug !== "string" || typeof project_slug !== "string" || typeof env !== "string") {
    return null;
  }

  // Required: counts with open_issues minimum
  const counts = raw.counts;
  if (!isObject(counts)) return null;
  const open_issues = counts.open_issues;
  if (typeof open_issues !== "number") return null;

  // Optional counts with defaults
  const new_issues_1h = typeof counts.new_issues_1h === "number" ? counts.new_issues_1h : 0;
  const new_issues_24h = typeof counts.new_issues_24h === "number" ? counts.new_issues_24h : 0;
  const active_issues_1h = typeof counts.active_issues_1h === "number" ? counts.active_issues_1h : undefined;
  const active_issues_24h = typeof counts.active_issues_24h === "number" ? counts.active_issues_24h : undefined;

  // Optional: generated_at (no longer required)
  const generated_at = typeof raw.generated_at === "string" ? raw.generated_at : undefined;

  // Optional: cached/cache_age_seconds
  const cached = typeof raw.cached === "boolean" ? raw.cached : undefined;
  const cache_age_seconds = typeof raw.cache_age_seconds === "number" ? raw.cache_age_seconds : undefined;

  // Optional: last_event_at
  const last_event_at = typeof raw.last_event_at === "string" ? raw.last_event_at : undefined;

  // Optional: note
  const note = typeof raw.note === "string" ? raw.note : undefined;

  // Optional: top_issues (array) - supports both old {count_24h, level} and new {count} shapes
  let top_issues: SentryTopIssue[] | undefined;
  if (Array.isArray(raw.top_issues)) {
    top_issues = [];
    for (const issue of raw.top_issues) {
      if (isObject(issue) && typeof issue.title === "string") {
        // Normalize count: prefer count_24h, fallback to count
        const count_24h = typeof issue.count_24h === "number"
          ? issue.count_24h
          : typeof issue.count === "number"
            ? issue.count
            : 0;

        // Level is optional now
        const level = isValidSentryLevel(issue.level) ? issue.level : undefined;

        top_issues.push({ title: issue.title, count_24h, level });
      }
    }
    if (top_issues.length === 0) {
      top_issues = undefined;
    }
  }

  return {
    status,
    cached,
    cache_age_seconds,
    generated_at,
    project: { org_slug, project_slug, env },
    counts: { new_issues_1h, new_issues_24h, active_issues_1h, active_issues_24h, open_issues },
    last_event_at,
    top_issues,
    note,
  };
}

/**
 * Combined extraction and adaptation for sentry
 *
 * Returns OpsSentrySummary if successful, null otherwise
 */
export function parseOpsSentry(ops: OpsResponse): OpsSentrySummary | null {
  const rawSentry = extractSentry(ops);
  if (!rawSentry) return null;
  return adaptOpsSentry(rawSentry);
}

// ============================================================================
// Jobs Health Extraction (best-effort)
// ============================================================================

/**
 * Individual job status from backend
 */
export interface OpsJobItem {
  status: ApiBudgetStatus;
  last_success_at?: string;
  minutes_since_success?: number;
  source?: string;
  help_url?: string;
  // Job-specific fields
  ft_pending?: number; // stats_backfill
  backlog_ready?: number; // fastpath
}

/**
 * Top alert from jobs_health (Auditor Dashboard enhancement)
 * Only present when jobs_health.status is warn or red
 */
export interface OpsJobTopAlert {
  job_key: string;
  label: string;
  severity: "warn" | "red";
  reason: string;
  minutes_since_success: number | null;
  runbook_url?: string;
}

/**
 * Jobs health summary from ops.json
 */
export interface OpsJobsHealth {
  status: ApiBudgetStatus;
  runbook_url?: string;
  stats_backfill: OpsJobItem | null;
  odds_sync: OpsJobItem | null;
  fastpath: OpsJobItem | null;
  /** Top alert - only present when status is warn or red */
  top_alert?: OpsJobTopAlert;
  /** Count of jobs with alerts */
  alerts_count?: number;
}

/**
 * Parse individual job item
 */
function parseJobItem(raw: unknown): OpsJobItem | null {
  if (!isObject(raw)) return null;

  // Normalize status (backend may use "warn" instead of "warning")
  const status = normalizeStatus(raw.status);
  if (status === null) return null;

  return {
    status,
    last_success_at: typeof raw.last_success_at === "string" ? raw.last_success_at : undefined,
    minutes_since_success: typeof raw.minutes_since_success === "number" ? raw.minutes_since_success : undefined,
    source: typeof raw.source === "string" ? raw.source : undefined,
    help_url: typeof raw.help_url === "string" ? raw.help_url : undefined,
    ft_pending: typeof raw.ft_pending === "number" ? raw.ft_pending : undefined,
    backlog_ready: typeof raw.backlog_ready === "number" ? raw.backlog_ready : undefined,
  };
}

/**
 * Parse top_alert from jobs_health
 */
function parseJobTopAlert(raw: unknown): OpsJobTopAlert | undefined {
  if (!isObject(raw)) return undefined;

  const job_key = raw.job_key;
  const label = raw.label;
  const severity = raw.severity;
  const reason = raw.reason;

  // Validate required fields
  if (typeof job_key !== "string") return undefined;
  if (typeof label !== "string") return undefined;
  if (severity !== "warn" && severity !== "red") return undefined;
  if (typeof reason !== "string") return undefined;

  return {
    job_key,
    label,
    severity,
    reason,
    minutes_since_success: typeof raw.minutes_since_success === "number" ? raw.minutes_since_success : null,
    runbook_url: typeof raw.runbook_url === "string" ? raw.runbook_url : undefined,
  };
}

/**
 * Parse jobs health from ops response
 */
export function parseOpsJobsHealth(ops: OpsResponse): OpsJobsHealth | null {
  const jobsHealth = getNestedValue(ops, "data", "jobs_health");
  if (!isObject(jobsHealth)) return null;

  // Normalize status (backend may use "warn" instead of "warning")
  const status = normalizeStatus(jobsHealth.status);
  if (status === null) return null;

  // Parse top_alert (only present when status is warn/red)
  const top_alert = parseJobTopAlert(jobsHealth.top_alert);
  const alerts_count = typeof jobsHealth.alerts_count === "number" ? jobsHealth.alerts_count : undefined;

  return {
    status,
    runbook_url: typeof jobsHealth.runbook_url === "string" ? jobsHealth.runbook_url : undefined,
    stats_backfill: parseJobItem(jobsHealth.stats_backfill),
    odds_sync: parseJobItem(jobsHealth.odds_sync),
    fastpath: parseJobItem(jobsHealth.fastpath),
    top_alert,
    alerts_count,
  };
}

// ============================================================================
// Fastpath Health Extraction (detailed)
// ============================================================================

/**
 * Fastpath last tick result
 */
export interface FastpathTickResult {
  selected: number;
  refreshed: number;
  stats_ready: number;
  enqueued: number;
  completed: number;
  errors: number;
  skipped: number;
}

/**
 * Fastpath 60m stats
 */
export interface FastpathLast60m {
  ok: number;
  ok_retry: number;
  error: number;
  skipped: number;
  in_queue: number;
  running: number;
  total_processed: number;
  error_rate_pct: number;
}

/**
 * Fastpath health summary from ops.json
 */
export interface OpsFastpathHealth {
  status: ApiBudgetStatus;
  status_reason?: string;
  enabled: boolean;
  last_tick_at?: string;
  minutes_since_tick?: number;
  last_tick_result?: FastpathTickResult;
  last_60m?: FastpathLast60m;
  top_error_codes_60m?: Record<string, number>;
  pending_ready: number;
  config?: {
    interval_seconds: number;
    lookback_minutes: number;
    max_concurrent_jobs: number;
  };
}

/**
 * Parse fastpath health from ops response
 */
export function parseOpsFastpathHealth(ops: OpsResponse): OpsFastpathHealth | null {
  const fastpath = getNestedValue(ops, "data", "fastpath_health");
  if (!isObject(fastpath)) return null;

  // Normalize status (backend may use "warn" instead of "warning")
  const status = normalizeStatus(fastpath.status);
  if (status === null) return null;

  const enabled = typeof fastpath.enabled === "boolean" ? fastpath.enabled : false;
  const pending_ready = typeof fastpath.pending_ready === "number" ? fastpath.pending_ready : 0;

  // Parse last_tick_result
  let last_tick_result: FastpathTickResult | undefined;
  const tickResult = fastpath.last_tick_result;
  if (isObject(tickResult)) {
    last_tick_result = {
      selected: typeof tickResult.selected === "number" ? tickResult.selected : 0,
      refreshed: typeof tickResult.refreshed === "number" ? tickResult.refreshed : 0,
      stats_ready: typeof tickResult.stats_ready === "number" ? tickResult.stats_ready : 0,
      enqueued: typeof tickResult.enqueued === "number" ? tickResult.enqueued : 0,
      completed: typeof tickResult.completed === "number" ? tickResult.completed : 0,
      errors: typeof tickResult.errors === "number" ? tickResult.errors : 0,
      skipped: typeof tickResult.skipped === "number" ? tickResult.skipped : 0,
    };
  }

  // Parse last_60m
  let last_60m: FastpathLast60m | undefined;
  const l60m = fastpath.last_60m;
  if (isObject(l60m)) {
    last_60m = {
      ok: typeof l60m.ok === "number" ? l60m.ok : 0,
      ok_retry: typeof l60m.ok_retry === "number" ? l60m.ok_retry : 0,
      error: typeof l60m.error === "number" ? l60m.error : 0,
      skipped: typeof l60m.skipped === "number" ? l60m.skipped : 0,
      in_queue: typeof l60m.in_queue === "number" ? l60m.in_queue : 0,
      running: typeof l60m.running === "number" ? l60m.running : 0,
      total_processed: typeof l60m.total_processed === "number" ? l60m.total_processed : 0,
      error_rate_pct: typeof l60m.error_rate_pct === "number" ? l60m.error_rate_pct : 0,
    };
  }

  // Parse config
  let config: OpsFastpathHealth["config"];
  const cfg = fastpath.config;
  if (isObject(cfg)) {
    config = {
      interval_seconds: typeof cfg.interval_seconds === "number" ? cfg.interval_seconds : 120,
      lookback_minutes: typeof cfg.lookback_minutes === "number" ? cfg.lookback_minutes : 180,
      max_concurrent_jobs: typeof cfg.max_concurrent_jobs === "number" ? cfg.max_concurrent_jobs : 10,
    };
  }

  return {
    status,
    status_reason: typeof fastpath.status_reason === "string" ? fastpath.status_reason : undefined,
    enabled,
    last_tick_at: typeof fastpath.last_tick_at === "string" ? fastpath.last_tick_at : undefined,
    minutes_since_tick: typeof fastpath.minutes_since_tick === "number" ? fastpath.minutes_since_tick : undefined,
    last_tick_result,
    last_60m,
    top_error_codes_60m: isObject(fastpath.top_error_codes_60m)
      ? (fastpath.top_error_codes_60m as Record<string, number>)
      : undefined,
    pending_ready,
    config,
  };
}

// ============================================================================
// Predictions Health Extraction
// ============================================================================

/**
 * Predictions health summary from ops.json
 */
export interface OpsPredictionsHealth {
  status: ApiBudgetStatus;
  status_reason?: string;
  ns_matches_next_48h: number;
  ns_matches_next_48h_missing_prediction: number;
  ns_coverage_pct: number;
  next_ns_match_utc?: string;
  ft_matches_last_48h: number;
  ft_matches_last_48h_missing_prediction: number;
  ft_coverage_pct: number;
  last_prediction_saved_at?: string;
  hours_since_last_prediction?: number;
  predictions_saved_last_24h: number;
  predictions_saved_today_utc: number;
  thresholds?: {
    ns_coverage_warn_pct: number;
    ns_coverage_red_pct: number;
    ft_coverage_warn_pct: number;
    ft_coverage_red_pct: number;
  };
}

/**
 * Parse predictions health from ops response
 */
export function parseOpsPredictionsHealth(ops: OpsResponse): OpsPredictionsHealth | null {
  const pred = getNestedValue(ops, "data", "predictions_health");
  if (!isObject(pred)) return null;

  // Backend uses "warn" but frontend expects "warning"
  const status = normalizeStatus(pred.status);
  if (status === null) return null;

  return {
    status,
    status_reason: typeof pred.status_reason === "string" ? pred.status_reason : undefined,
    ns_matches_next_48h: typeof pred.ns_matches_next_48h === "number" ? pred.ns_matches_next_48h : 0,
    ns_matches_next_48h_missing_prediction: typeof pred.ns_matches_next_48h_missing_prediction === "number" ? pred.ns_matches_next_48h_missing_prediction : 0,
    ns_coverage_pct: typeof pred.ns_coverage_pct === "number" ? pred.ns_coverage_pct : 0,
    next_ns_match_utc: typeof pred.next_ns_match_utc === "string" ? pred.next_ns_match_utc : undefined,
    ft_matches_last_48h: typeof pred.ft_matches_last_48h === "number" ? pred.ft_matches_last_48h : 0,
    ft_matches_last_48h_missing_prediction: typeof pred.ft_matches_last_48h_missing_prediction === "number" ? pred.ft_matches_last_48h_missing_prediction : 0,
    ft_coverage_pct: typeof pred.ft_coverage_pct === "number" ? pred.ft_coverage_pct : 100,
    last_prediction_saved_at: typeof pred.last_prediction_saved_at === "string" ? pred.last_prediction_saved_at : undefined,
    hours_since_last_prediction: typeof pred.hours_since_last_prediction === "number" ? pred.hours_since_last_prediction : undefined,
    predictions_saved_last_24h: typeof pred.predictions_saved_last_24h === "number" ? pred.predictions_saved_last_24h : 0,
    predictions_saved_today_utc: typeof pred.predictions_saved_today_utc === "number" ? pred.predictions_saved_today_utc : 0,
    thresholds: isObject(pred.thresholds) ? {
      ns_coverage_warn_pct: typeof pred.thresholds.ns_coverage_warn_pct === "number" ? pred.thresholds.ns_coverage_warn_pct : 80,
      ns_coverage_red_pct: typeof pred.thresholds.ns_coverage_red_pct === "number" ? pred.thresholds.ns_coverage_red_pct : 50,
      ft_coverage_warn_pct: typeof pred.thresholds.ft_coverage_warn_pct === "number" ? pred.thresholds.ft_coverage_warn_pct : 80,
      ft_coverage_red_pct: typeof pred.thresholds.ft_coverage_red_pct === "number" ? pred.thresholds.ft_coverage_red_pct : 50,
    } : undefined,
  };
}

// ============================================================================
// Shadow Mode Extraction
// ============================================================================

/**
 * Shadow mode summary from ops.json
 */
export interface OpsShadowMode {
  state: {
    enabled: boolean;
    shadow_architecture?: string;
    shadow_model_version?: string;
    baseline_model_version?: string;
    last_evaluation_at?: string;
    evaluation_job_interval_minutes?: number;
  };
  counts: {
    shadow_predictions_total: number;
    shadow_predictions_evaluated: number;
    shadow_predictions_pending: number;
    shadow_predictions_last_24h: number;
    shadow_evaluations_last_24h: number;
    shadow_errors_last_24h: number;
  };
  metrics: {
    baseline_accuracy: number;
    shadow_accuracy: number;
    baseline_brier: number;
    shadow_brier: number;
    delta_accuracy: number;
    delta_brier: number;
  };
  recommendation: {
    status: string; // "GO" | "NO_GO" | "PENDING"
    reason?: string;
  };
  health: {
    pending_ft_to_evaluate: number;
    eval_lag_minutes: number;
    stale_threshold_minutes: number;
    is_stale: boolean;
  };
}

/**
 * Parse shadow mode from ops response
 */
export function parseOpsShadowMode(ops: OpsResponse): OpsShadowMode | null {
  const shadow = getNestedValue(ops, "data", "shadow_mode");
  if (!isObject(shadow)) return null;

  const state = shadow.state;
  const counts = shadow.counts;
  const metrics = shadow.metrics;
  const recommendation = shadow.recommendation;
  const health = shadow.health;

  if (!isObject(state) || !isObject(counts) || !isObject(metrics) || !isObject(recommendation) || !isObject(health)) {
    return null;
  }

  return {
    state: {
      enabled: typeof state.enabled === "boolean" ? state.enabled : false,
      shadow_architecture: typeof state.shadow_architecture === "string" ? state.shadow_architecture : undefined,
      shadow_model_version: typeof state.shadow_model_version === "string" ? state.shadow_model_version : undefined,
      baseline_model_version: typeof state.baseline_model_version === "string" ? state.baseline_model_version : undefined,
      last_evaluation_at: typeof state.last_evaluation_at === "string" ? state.last_evaluation_at : undefined,
      evaluation_job_interval_minutes: typeof state.evaluation_job_interval_minutes === "number" ? state.evaluation_job_interval_minutes : undefined,
    },
    counts: {
      shadow_predictions_total: typeof counts.shadow_predictions_total === "number" ? counts.shadow_predictions_total : 0,
      shadow_predictions_evaluated: typeof counts.shadow_predictions_evaluated === "number" ? counts.shadow_predictions_evaluated : 0,
      shadow_predictions_pending: typeof counts.shadow_predictions_pending === "number" ? counts.shadow_predictions_pending : 0,
      shadow_predictions_last_24h: typeof counts.shadow_predictions_last_24h === "number" ? counts.shadow_predictions_last_24h : 0,
      shadow_evaluations_last_24h: typeof counts.shadow_evaluations_last_24h === "number" ? counts.shadow_evaluations_last_24h : 0,
      shadow_errors_last_24h: typeof counts.shadow_errors_last_24h === "number" ? counts.shadow_errors_last_24h : 0,
    },
    metrics: {
      baseline_accuracy: typeof metrics.baseline_accuracy === "number" ? metrics.baseline_accuracy : 0,
      shadow_accuracy: typeof metrics.shadow_accuracy === "number" ? metrics.shadow_accuracy : 0,
      baseline_brier: typeof metrics.baseline_brier === "number" ? metrics.baseline_brier : 0,
      shadow_brier: typeof metrics.shadow_brier === "number" ? metrics.shadow_brier : 0,
      delta_accuracy: typeof metrics.delta_accuracy === "number" ? metrics.delta_accuracy : 0,
      delta_brier: typeof metrics.delta_brier === "number" ? metrics.delta_brier : 0,
    },
    recommendation: {
      status: typeof recommendation.status === "string" ? recommendation.status : "PENDING",
      reason: typeof recommendation.reason === "string" ? recommendation.reason : undefined,
    },
    health: {
      pending_ft_to_evaluate: typeof health.pending_ft_to_evaluate === "number" ? health.pending_ft_to_evaluate : 0,
      eval_lag_minutes: typeof health.eval_lag_minutes === "number" ? health.eval_lag_minutes : 0,
      stale_threshold_minutes: typeof health.stale_threshold_minutes === "number" ? health.stale_threshold_minutes : 120,
      is_stale: typeof health.is_stale === "boolean" ? health.is_stale : false,
    },
  };
}

// ============================================================================
// Sensor B Extraction
// ============================================================================

/**
 * Sensor B summary from ops.json
 */
export interface OpsSensorB {
  state: string; // "CALIBRATING" | "NOMINAL" | "OVERFITTING_SUSPECTED" | etc.
  reason?: string;
  note?: string;
  signal_score?: number;
  brier_a?: number;
  brier_b?: number;
  delta_brier?: number;
  accuracy_a?: number;
  accuracy_b?: number;
  window_size?: number;
  is_ready: boolean;
  health: {
    pending_ft_to_evaluate: number;
    eval_lag_minutes: number;
    stale_threshold_minutes: number;
    is_stale: boolean;
  };
}

/**
 * Parse sensor B from ops response
 */
export function parseOpsSensorB(ops: OpsResponse): OpsSensorB | null {
  const sensor = getNestedValue(ops, "data", "sensor_b");
  if (!isObject(sensor)) return null;

  const health = sensor.health;
  if (!isObject(health)) return null;

  const state = sensor.state;
  if (typeof state !== "string") return null;

  return {
    state,
    reason: typeof sensor.reason === "string" ? sensor.reason : undefined,
    note: typeof sensor.note === "string" ? sensor.note : undefined,
    signal_score: typeof sensor.signal_score === "number" ? sensor.signal_score : undefined,
    brier_a: typeof sensor.brier_a === "number" ? sensor.brier_a : undefined,
    brier_b: typeof sensor.brier_b === "number" ? sensor.brier_b : undefined,
    delta_brier: typeof sensor.delta_brier === "number" ? sensor.delta_brier : undefined,
    accuracy_a: typeof sensor.accuracy_a === "number" ? sensor.accuracy_a : undefined,
    accuracy_b: typeof sensor.accuracy_b === "number" ? sensor.accuracy_b : undefined,
    window_size: typeof sensor.window_size === "number" ? sensor.window_size : undefined,
    is_ready: typeof sensor.is_ready === "boolean" ? sensor.is_ready : false,
    health: {
      pending_ft_to_evaluate: typeof health.pending_ft_to_evaluate === "number" ? health.pending_ft_to_evaluate : 0,
      eval_lag_minutes: typeof health.eval_lag_minutes === "number" ? health.eval_lag_minutes : 0,
      stale_threshold_minutes: typeof health.stale_threshold_minutes === "number" ? health.stale_threshold_minutes : 120,
      is_stale: typeof health.is_stale === "boolean" ? health.is_stale : false,
    },
  };
}

// ============================================================================
// LLM Cost Extraction
// ============================================================================

/**
 * Model pricing entry
 */
export interface OpsLlmModelPricing {
  input: number;
  output: number;
}

/**
 * Model usage stats for a time window
 */
export interface OpsLlmModelUsage {
  model: string;
  requests: number;
  tokens_in: number;
  tokens_out: number;
  cost_usd: number;
}

/**
 * LLM cost summary from ops.json
 */
export interface OpsLlmCost {
  status: ApiBudgetStatus;
  provider: string;
  /** Source of pricing config (e.g., "config.GEMINI_PRICING") */
  pricing_source?: string;
  /** Current model in use */
  model?: string;
  /** Pricing for current model (per 1M tokens) */
  pricing_input_per_1m?: number;
  pricing_output_per_1m?: number;
  /** Full pricing table for all models */
  model_pricing?: Record<string, OpsLlmModelPricing>;
  /** Model usage breakdown by time window (when available) */
  model_usage_24h?: OpsLlmModelUsage[];
  model_usage_7d?: OpsLlmModelUsage[];
  model_usage_28d?: OpsLlmModelUsage[];
  cost_24h_usd: number;
  cost_7d_usd: number;
  cost_28d_usd: number;
  cost_total_usd: number;
  requests_24h: number;
  requests_7d: number;
  requests_28d: number;
  requests_total: number;
  // Legacy fields (mapped from new field names for backwards compat)
  requests_ok_24h: number;
  requests_ok_7d: number;
  requests_ok_total: number;
  avg_cost_per_request_24h?: number;
  tokens_in_24h: number;
  tokens_out_24h: number;
  tokens_in_7d: number;
  tokens_out_7d: number;
  tokens_in_28d: number;
  tokens_out_28d: number;
  note?: string;
}

/**
 * Parse model_pricing dict from backend
 */
function parseModelPricing(raw: unknown): Record<string, OpsLlmModelPricing> | undefined {
  if (!isObject(raw)) return undefined;

  const result: Record<string, OpsLlmModelPricing> = {};
  for (const [model, pricing] of Object.entries(raw)) {
    if (isObject(pricing) && typeof pricing.input === "number" && typeof pricing.output === "number") {
      result[model] = { input: pricing.input, output: pricing.output };
    }
  }

  return Object.keys(result).length > 0 ? result : undefined;
}

/**
 * Parse model_usage array from backend
 */
function parseModelUsage(raw: unknown): OpsLlmModelUsage[] | undefined {
  if (!Array.isArray(raw)) return undefined;

  const result: OpsLlmModelUsage[] = [];
  for (const item of raw) {
    if (
      isObject(item) &&
      typeof item.model === "string" &&
      typeof item.requests === "number" &&
      typeof item.tokens_in === "number" &&
      typeof item.tokens_out === "number" &&
      typeof item.cost_usd === "number"
    ) {
      result.push({
        model: item.model,
        requests: item.requests,
        tokens_in: item.tokens_in,
        tokens_out: item.tokens_out,
        cost_usd: item.cost_usd,
      });
    }
  }

  return result.length > 0 ? result : undefined;
}

/**
 * Parse LLM cost from ops response
 */
export function parseOpsLlmCost(ops: OpsResponse): OpsLlmCost | null {
  const llm = getNestedValue(ops, "data", "llm_cost");
  if (!isObject(llm)) return null;

  // Normalize status (backend may use "warn" instead of "warning")
  const status = normalizeStatus(llm.status);
  if (status === null) return null;

  const provider = llm.provider;
  if (typeof provider !== "string") return null;

  // Parse new field names with fallback to legacy names
  const requests_24h = typeof llm.requests_24h === "number" ? llm.requests_24h
    : typeof llm.requests_ok_24h === "number" ? llm.requests_ok_24h : 0;
  const requests_7d = typeof llm.requests_7d === "number" ? llm.requests_7d
    : typeof llm.requests_ok_7d === "number" ? llm.requests_ok_7d : 0;
  const requests_28d = typeof llm.requests_28d === "number" ? llm.requests_28d : 0;
  const requests_total = typeof llm.requests_total === "number" ? llm.requests_total
    : typeof llm.requests_ok_total === "number" ? llm.requests_ok_total : 0;

  // Parse pricing fields
  const pricing_source = typeof llm.pricing_source === "string" ? llm.pricing_source : undefined;
  const model = typeof llm.model === "string" ? llm.model : undefined;
  const pricing_input_per_1m = typeof llm.pricing_input_per_1m === "number" ? llm.pricing_input_per_1m : undefined;
  const pricing_output_per_1m = typeof llm.pricing_output_per_1m === "number" ? llm.pricing_output_per_1m : undefined;
  const model_pricing = parseModelPricing(llm.model_pricing);

  // Parse model usage breakdowns (when available)
  const model_usage_24h = parseModelUsage(llm.model_usage_24h);
  const model_usage_7d = parseModelUsage(llm.model_usage_7d);
  const model_usage_28d = parseModelUsage(llm.model_usage_28d);

  return {
    status,
    provider,
    pricing_source,
    model,
    pricing_input_per_1m,
    pricing_output_per_1m,
    model_pricing,
    model_usage_24h,
    model_usage_7d,
    model_usage_28d,
    cost_24h_usd: typeof llm.cost_24h_usd === "number" ? llm.cost_24h_usd : 0,
    cost_7d_usd: typeof llm.cost_7d_usd === "number" ? llm.cost_7d_usd : 0,
    cost_28d_usd: typeof llm.cost_28d_usd === "number" ? llm.cost_28d_usd : 0,
    cost_total_usd: typeof llm.cost_total_usd === "number" ? llm.cost_total_usd : 0,
    requests_24h,
    requests_7d,
    requests_28d,
    requests_total,
    // Legacy field mappings for backwards compat
    requests_ok_24h: requests_24h,
    requests_ok_7d: requests_7d,
    requests_ok_total: requests_total,
    avg_cost_per_request_24h: typeof llm.avg_cost_per_request_24h === "number" ? llm.avg_cost_per_request_24h : undefined,
    tokens_in_24h: typeof llm.tokens_in_24h === "number" ? llm.tokens_in_24h : 0,
    tokens_out_24h: typeof llm.tokens_out_24h === "number" ? llm.tokens_out_24h : 0,
    tokens_in_7d: typeof llm.tokens_in_7d === "number" ? llm.tokens_in_7d : 0,
    tokens_out_7d: typeof llm.tokens_out_7d === "number" ? llm.tokens_out_7d : 0,
    tokens_in_28d: typeof llm.tokens_in_28d === "number" ? llm.tokens_in_28d : 0,
    tokens_out_28d: typeof llm.tokens_out_28d === "number" ? llm.tokens_out_28d : 0,
    note: typeof llm.note === "string" ? llm.note : undefined,
  };
}

// ============================================================================
// Ops Freshness Extraction
// ============================================================================

/**
 * Ops freshness metadata
 */
export interface OpsFreshness {
  generated_at: string;
  cache_age_seconds: number;
  is_stale: boolean;
}

/**
 * Parse ops freshness from response
 */
export function parseOpsFreshness(ops: OpsResponse): OpsFreshness | null {
  if (!isObject(ops)) return null;

  const generated_at = getNestedValue(ops, "data", "generated_at");
  const cache_age_seconds = ops.cache_age_seconds;

  if (typeof generated_at !== "string") return null;
  const cacheAge = typeof cache_age_seconds === "number" ? cache_age_seconds : 0;

  return {
    generated_at,
    cache_age_seconds: cacheAge,
    is_stale: cacheAge > 120, // stale if > 2 minutes
  };
}

// ============================================================================
// Progress Extraction (PIT evaluation progress)
// ============================================================================

/**
 * Progress summary from ops.json (PIT evaluation progress)
 */
export interface OpsProgress {
  pit_snapshots_30d: number;
  target_pit_snapshots_30d: number;
  pit_bets_30d: number;
  target_pit_bets_30d: number;
  baseline_coverage_pct: number;
  pit_with_baseline: number;
  pit_total_for_baseline: number;
  target_baseline_coverage_pct: number;
  ready_for_retest: boolean;
}

/**
 * Parse progress from ops response
 *
 * Expected structure: { data: { progress: {...} } }
 */
export function parseOpsProgress(ops: OpsResponse): OpsProgress | null {
  const progress = getNestedValue(ops, "data", "progress");
  if (!isObject(progress)) return null;

  return {
    pit_snapshots_30d: typeof progress.pit_snapshots_30d === "number" ? progress.pit_snapshots_30d : 0,
    target_pit_snapshots_30d: typeof progress.target_pit_snapshots_30d === "number" ? progress.target_pit_snapshots_30d : 0,
    pit_bets_30d: typeof progress.pit_bets_30d === "number" ? progress.pit_bets_30d : 0,
    target_pit_bets_30d: typeof progress.target_pit_bets_30d === "number" ? progress.target_pit_bets_30d : 0,
    baseline_coverage_pct: typeof progress.baseline_coverage_pct === "number" ? progress.baseline_coverage_pct : 0,
    pit_with_baseline: typeof progress.pit_with_baseline === "number" ? progress.pit_with_baseline : 0,
    pit_total_for_baseline: typeof progress.pit_total_for_baseline === "number" ? progress.pit_total_for_baseline : 0,
    target_baseline_coverage_pct: typeof progress.target_baseline_coverage_pct === "number" ? progress.target_baseline_coverage_pct : 0,
    ready_for_retest: typeof progress.ready_for_retest === "boolean" ? progress.ready_for_retest : false,
  };
}

// ============================================================================
// PIT Activity Extraction (live odds snapshots)
// ============================================================================

/**
 * PIT activity summary from ops.json
 */
export interface OpsPitActivity {
  live_60m: number;
  live_24h: number;
}

/**
 * Parse PIT activity from ops response
 *
 * Expected structure: { data: { pit: { live_60m, live_24h, ... } } }
 */
export function parseOpsPitActivity(ops: OpsResponse): OpsPitActivity | null {
  const pit = getNestedValue(ops, "data", "pit");
  if (!isObject(pit)) return null;

  return {
    live_60m: typeof pit.live_60m === "number" ? pit.live_60m : 0,
    live_24h: typeof pit.live_24h === "number" ? pit.live_24h : 0,
  };
}

// ============================================================================
// Movement Extraction (lineup & market movement)
// ============================================================================

/**
 * Movement summary from ops.json
 */
export interface OpsMovement {
  lineup_movement_24h: number;
  market_movement_24h: number;
}

/**
 * Parse movement from ops response
 *
 * Expected structure: { data: { movement: {...} } }
 */
export function parseOpsMovement(ops: OpsResponse): OpsMovement | null {
  const movement = getNestedValue(ops, "data", "movement");
  if (!isObject(movement)) return null;

  return {
    lineup_movement_24h: typeof movement.lineup_movement_24h === "number" ? movement.lineup_movement_24h : 0,
    market_movement_24h: typeof movement.market_movement_24h === "number" ? movement.market_movement_24h : 0,
  };
}

// ============================================================================
// Telemetry (Data Quality) Extraction
// ============================================================================

/**
 * Telemetry status type
 *
 * Separate from ApiBudgetStatus to handle backend's different status format.
 * Backend sends: "OK" | "WARN" | "RED"
 * We normalize to: "ok" | "warning" | "critical" | "degraded"
 */
export type TelemetryStatus = "ok" | "warning" | "critical" | "degraded";

/**
 * Normalize telemetry status from backend to frontend format
 *
 * Backend can send various formats:
 * - "OK", "ok", "green" -> "ok"
 * - "WARN", "warning", "yellow" -> "warning"
 * - "RED", "critical", "error" -> "critical"
 * - fallback -> "degraded"
 */
function normalizeTelemetryStatus(status: unknown): TelemetryStatus {
  if (typeof status !== "string") return "degraded";

  const normalized = status.toLowerCase();

  if (normalized === "ok" || normalized === "green") return "ok";
  if (normalized === "warn" || normalized === "warning" || normalized === "yellow") return "warning";
  if (normalized === "red" || normalized === "critical" || normalized === "error") return "critical";

  return "degraded";
}

/**
 * Telemetry summary from ops.json (data quality metrics)
 */
export interface OpsTelemetry {
  status: TelemetryStatus;
  updated_at?: string;
  summary: {
    quarantined_odds_24h: number;
    tainted_matches_24h: number;
    unmapped_entities_24h: number;
    odds_desync_6h: number;
    odds_desync_90m: number;
  };
  links?: Array<{ title: string; url: string }>;
}

/**
 * Parse telemetry (data quality) from ops response
 *
 * Expected structure: { data: { telemetry: {...} } }
 * Backend status is normalized (OK -> ok, WARN -> warning, RED -> critical)
 */
export function parseOpsTelemetry(ops: OpsResponse): OpsTelemetry | null {
  const telemetry = getNestedValue(ops, "data", "telemetry");
  if (!isObject(telemetry)) return null;

  const summary = telemetry.summary;
  if (!isObject(summary)) return null;

  // Normalize status (accepts OK, WARN, RED from backend)
  const status = normalizeTelemetryStatus(telemetry.status);

  return {
    status,
    updated_at: typeof telemetry.updated_at === "string" ? telemetry.updated_at : undefined,
    summary: {
      quarantined_odds_24h: typeof summary.quarantined_odds_24h === "number" ? summary.quarantined_odds_24h : 0,
      tainted_matches_24h: typeof summary.tainted_matches_24h === "number" ? summary.tainted_matches_24h : 0,
      unmapped_entities_24h: typeof summary.unmapped_entities_24h === "number" ? summary.unmapped_entities_24h : 0,
      odds_desync_6h: typeof summary.odds_desync_6h === "number" ? summary.odds_desync_6h : 0,
      odds_desync_90m: typeof summary.odds_desync_90m === "number" ? summary.odds_desync_90m : 0,
    },
    links: Array.isArray(telemetry.links)
      ? telemetry.links
          .filter((l): l is { title: string; url: string } =>
            isObject(l) && typeof l.title === "string" && typeof l.url === "string"
          )
      : undefined,
  };
}

// ============================================================================
// SOTA Enrichment Extraction (best-effort)
// ============================================================================

/**
 * SOTA enrichment component status
 */
export type SotaEnrichmentStatus = "ok" | "warn" | "red" | "unavailable" | "pending";

/**
 * SOTA enrichment component keys
 */
export type SotaEnrichmentKey = "understat" | "weather" | "venue_geo" | "team_profiles" | "sofascore_xi";

/**
 * Normalized SOTA enrichment item (per component)
 */
export interface SotaEnrichmentNormalizedItem {
  key: SotaEnrichmentKey;
  status: SotaEnrichmentStatus;
  coverage_pct: number;
  with_data: number;
  total: number;
  staleness_hours: number | null;
  latest_capture_at: string | null;
  note: string | null;
  error: string | null;
  // Optional secondary KPIs for sofascore_xi
  total_lineups?: number;
  total_players?: number;
}

/**
 * Normalized SOTA enrichment summary
 */
export interface SotaEnrichmentNormalized {
  status: "ok" | "warn" | "red" | "unavailable";
  generated_at: string | null;
  items: SotaEnrichmentNormalizedItem[];
}

/**
 * Normalize SOTA status from backend
 */
function normalizeSotaStatus(status: unknown): SotaEnrichmentStatus {
  if (typeof status !== "string") return "unavailable";

  const normalized = status.toLowerCase();
  if (normalized === "ok" || normalized === "green") return "ok";
  if (normalized === "warn" || normalized === "warning" || normalized === "yellow") return "warn";
  if (normalized === "red" || normalized === "critical" || normalized === "error") return "red";
  if (normalized === "pending") return "pending";

  return "unavailable";
}

/**
 * Normalize overall SOTA status
 */
function normalizeSotaOverallStatus(status: unknown): "ok" | "warn" | "red" | "unavailable" {
  if (typeof status !== "string") return "unavailable";

  const normalized = status.toLowerCase();
  if (normalized === "ok" || normalized === "green") return "ok";
  if (normalized === "warn" || normalized === "warning" || normalized === "yellow") return "warn";
  if (normalized === "red" || normalized === "critical" || normalized === "error") return "red";

  return "unavailable";
}

/**
 * Extract sota_enrichment object from ops response
 *
 * Expected structure: { data: { sota_enrichment: {...} } }
 */
export function extractSotaEnrichment(ops: OpsResponse): unknown | null {
  if (!isObject(ops)) return null;

  const sota = getNestedValue(ops, "data", "sota_enrichment");
  if (!isObject(sota)) return null;

  return sota;
}

/**
 * Normalize a single SOTA component
 *
 * Best-effort parsing with safe defaults
 */
function normalizeSotaComponent(
  key: SotaEnrichmentKey,
  raw: unknown
): SotaEnrichmentNormalizedItem {
  // Default safe item
  const defaultItem: SotaEnrichmentNormalizedItem = {
    key,
    status: "unavailable",
    coverage_pct: 0,
    with_data: 0,
    total: 0,
    staleness_hours: null,
    latest_capture_at: null,
    note: null,
    error: null,
  };

  if (!isObject(raw)) {
    return { ...defaultItem, note: "No data available" };
  }

  // Check for pending status (e.g., tables not deployed yet)
  const status = normalizeSotaStatus(raw.status);

  // Extract common fields
  const coverage_pct = typeof raw.coverage_pct === "number" ? raw.coverage_pct : 0;
  const total = typeof raw.total === "number" ? raw.total : 0;
  const with_data = typeof raw.with_data === "number" ? raw.with_data : 0;
  const staleness_hours = typeof raw.staleness_hours === "number" ? raw.staleness_hours : null;
  const latest_capture_at = typeof raw.latest_capture_at === "string" ? raw.latest_capture_at : null;
  const note = typeof raw.note === "string" ? raw.note : null;
  const error = typeof raw.error === "string" ? raw.error : null;

  const item: SotaEnrichmentNormalizedItem = {
    key,
    status,
    coverage_pct,
    with_data,
    total,
    staleness_hours,
    latest_capture_at,
    note,
    error,
  };

  // Handle sofascore_xi specific fields
  if (key === "sofascore_xi") {
    if (typeof raw.total_lineups === "number") {
      item.total_lineups = raw.total_lineups;
    }
    if (typeof raw.total_players === "number") {
      item.total_players = raw.total_players;
    }
  }

  return item;
}

/**
 * Normalize SOTA enrichment from raw backend data
 *
 * Handles:
 * - Missing components (status = unavailable)
 * - Pending components (tables not deployed)
 * - All expected fields with safe defaults
 */
export function normalizeSotaEnrichment(raw: unknown): SotaEnrichmentNormalized | null {
  if (!isObject(raw)) return null;

  // Extract overall status and generated_at
  const status = normalizeSotaOverallStatus(raw.status);
  const generated_at = typeof raw.generated_at === "string" ? raw.generated_at : null;

  // All component keys we expect
  const componentKeys: SotaEnrichmentKey[] = [
    "understat",
    "weather",
    "venue_geo",
    "team_profiles",
    "sofascore_xi",
  ];

  // Normalize each component
  const items: SotaEnrichmentNormalizedItem[] = componentKeys.map((key) => {
    const component = raw[key];
    return normalizeSotaComponent(key, component);
  });

  return {
    status,
    generated_at,
    items,
  };
}

/**
 * Combined extraction and normalization for SOTA enrichment
 *
 * Returns SotaEnrichmentNormalized if successful, null otherwise
 */
export function parseOpsSotaEnrichment(ops: OpsResponse): SotaEnrichmentNormalized | null {
  const rawSota = extractSotaEnrichment(ops);
  if (!rawSota) return null;
  return normalizeSotaEnrichment(rawSota);
}

// ============================================================================
// TITAN OMNISCIENCE Extraction
// ============================================================================

/**
 * TITAN status type
 */
export type TitanStatus = "ok" | "building" | "warn" | "error" | "unavailable";

/**
 * TITAN job status type
 */
export type TitanJobStatus = "success" | "failed" | "never_run";

/**
 * TITAN feature matrix
 */
export interface OpsTitanFeatureMatrix {
  total_rows: number;
  tier1_complete: number;
  tier1b_complete: number;
  tier1c_complete: number;
  tier1d_complete: number;
  with_outcome: number;
  tier1b_pct: number;
  tier1c_pct: number;
}

/**
 * TITAN gate (evaluation readiness)
 */
export interface OpsTitanGate {
  n_current: number;
  n_target_pilot: number;
  n_target_prelim: number;
  n_target_formal: number;
  ready_for_pilot: boolean;
  ready_for_prelim: boolean;
  ready_for_formal: boolean;
  pct_to_pilot: number;
  pct_to_prelim: number;
  pct_to_formal: number;
}

/**
 * TITAN job status
 */
export interface OpsTitanJob {
  last_status: TitanJobStatus;
  last_run_at: string | null;
  last_metrics: Record<string, unknown>;
  note: string | null;
}

/**
 * TITAN OMNISCIENCE summary from ops.json
 */
export interface OpsTitan {
  status: TitanStatus;
  generated_at: string | null;
  note: string | null;
  feature_matrix: OpsTitanFeatureMatrix;
  gate: OpsTitanGate;
  job: OpsTitanJob;
}

/**
 * Normalize TITAN status from backend
 */
function normalizeTitanStatus(status: unknown): TitanStatus {
  if (typeof status !== "string") return "unavailable";

  const normalized = status.toLowerCase();
  if (normalized === "ok" || normalized === "green") return "ok";
  if (normalized === "building") return "building";
  if (normalized === "warn" || normalized === "warning" || normalized === "yellow") return "warn";
  if (normalized === "error" || normalized === "red" || normalized === "critical") return "error";

  return "unavailable";
}

/**
 * Normalize TITAN job status from backend
 */
function normalizeTitanJobStatus(status: unknown): TitanJobStatus {
  if (typeof status !== "string") return "never_run";

  const normalized = status.toLowerCase();
  if (normalized === "success" || normalized === "ok") return "success";
  if (normalized === "failed" || normalized === "error") return "failed";

  return "never_run";
}

/**
 * Parse TITAN from ops response
 *
 * Expected structure: { data: { titan: {...} } }
 */
export function parseOpsTitan(ops: OpsResponse): OpsTitan | null {
  const titan = getNestedValue(ops, "data", "titan");
  if (!isObject(titan)) return null;

  const status = normalizeTitanStatus(titan.status);
  const generated_at = typeof titan.generated_at === "string" ? titan.generated_at : null;
  const note = typeof titan.note === "string" ? titan.note : null;

  // Parse feature_matrix
  const fm = titan.feature_matrix;
  const feature_matrix: OpsTitanFeatureMatrix = isObject(fm) ? {
    total_rows: typeof fm.total_rows === "number" ? fm.total_rows : 0,
    tier1_complete: typeof fm.tier1_complete === "number" ? fm.tier1_complete : 0,
    tier1b_complete: typeof fm.tier1b_complete === "number" ? fm.tier1b_complete : 0,
    tier1c_complete: typeof fm.tier1c_complete === "number" ? fm.tier1c_complete : 0,
    tier1d_complete: typeof fm.tier1d_complete === "number" ? fm.tier1d_complete : 0,
    with_outcome: typeof fm.with_outcome === "number" ? fm.with_outcome : 0,
    tier1b_pct: typeof fm.tier1b_pct === "number" ? fm.tier1b_pct : 0,
    tier1c_pct: typeof fm.tier1c_pct === "number" ? fm.tier1c_pct : 0,
  } : {
    total_rows: 0,
    tier1_complete: 0,
    tier1b_complete: 0,
    tier1c_complete: 0,
    tier1d_complete: 0,
    with_outcome: 0,
    tier1b_pct: 0,
    tier1c_pct: 0,
  };

  // Parse gate
  const g = titan.gate;
  const gate: OpsTitanGate = isObject(g) ? {
    n_current: typeof g.n_current === "number" ? g.n_current : 0,
    n_target_pilot: typeof g.n_target_pilot === "number" ? g.n_target_pilot : 50,
    n_target_prelim: typeof g.n_target_prelim === "number" ? g.n_target_prelim : 200,
    n_target_formal: typeof g.n_target_formal === "number" ? g.n_target_formal : 500,
    ready_for_pilot: typeof g.ready_for_pilot === "boolean" ? g.ready_for_pilot : false,
    ready_for_prelim: typeof g.ready_for_prelim === "boolean" ? g.ready_for_prelim : false,
    ready_for_formal: typeof g.ready_for_formal === "boolean" ? g.ready_for_formal : false,
    pct_to_pilot: typeof g.pct_to_pilot === "number" ? g.pct_to_pilot : 0,
    pct_to_prelim: typeof g.pct_to_prelim === "number" ? g.pct_to_prelim : 0,
    pct_to_formal: typeof g.pct_to_formal === "number" ? g.pct_to_formal : 0,
  } : {
    n_current: 0,
    n_target_pilot: 50,
    n_target_prelim: 200,
    n_target_formal: 500,
    ready_for_pilot: false,
    ready_for_prelim: false,
    ready_for_formal: false,
    pct_to_pilot: 0,
    pct_to_prelim: 0,
    pct_to_formal: 0,
  };

  // Parse job
  const j = titan.job;
  const job: OpsTitanJob = isObject(j) ? {
    last_status: normalizeTitanJobStatus(j.last_status),
    last_run_at: typeof j.last_run_at === "string" ? j.last_run_at : null,
    last_metrics: isObject(j.last_metrics) ? j.last_metrics as Record<string, unknown> : {},
    note: typeof j.note === "string" ? j.note : null,
  } : {
    last_status: "never_run",
    last_run_at: null,
    last_metrics: {},
    note: null,
  };

  return {
    status,
    generated_at,
    note,
    feature_matrix,
    gate,
    job,
  };
}
