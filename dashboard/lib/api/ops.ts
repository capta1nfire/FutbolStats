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
 * Validate ApiBudgetStatus
 */
function isValidStatus(status: unknown): status is ApiBudgetStatus {
  return (
    status === "ok" ||
    status === "warning" ||
    status === "critical" ||
    status === "degraded"
  );
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

  // Required fields with type validation
  const status = raw.status;
  if (!isValidStatus(status)) return null;

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
