/**
 * Jobs API Adapter
 *
 * Safe extraction and adaptation of data from /dashboard/jobs.json
 * Designed to be resilient to partial or malformed responses.
 */

import { JobRun, JobStatus, JobDefinition } from "@/lib/types";

/**
 * Expected response structure from backend
 */
export interface JobsResponse {
  generated_at?: string;
  cached?: boolean;
  cache_age_seconds?: number;
  data: {
    runs: unknown[];
    total: number;
    page: number;
    limit: number;
    pages: number;
    jobs_summary: Record<string, unknown>;
  };
}

/**
 * Pagination metadata from response
 */
export interface JobsPagination {
  total: number;
  page: number;
  limit: number;
  pages: number;
}

/**
 * Jobs summary (per-job health)
 */
export interface JobsSummary {
  [jobName: string]: {
    lastRunStatus: string | null;
    lastRunAt: string | null;
    lastSuccessAt: string | null;
    durationMs: number | null;
    lastError: string | null;
  };
}

/**
 * Safely check if value is a non-null object
 */
function isObject(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

/**
 * Map backend status to frontend JobStatus
 *
 * Backend uses: ok, error, rate_limited, budget_exceeded
 * Frontend uses: running, success, failed, pending
 */
function mapJobStatus(backendStatus: string): JobStatus {
  const status = backendStatus?.toLowerCase() || "pending";

  if (status === "ok") return "success";
  if (status === "error" || status === "rate_limited" || status === "budget_exceeded") return "failed";
  if (status === "running") return "running";
  return "pending";
}

/**
 * Adapt a single raw job run object to JobRun type
 *
 * Returns null if critical fields are missing or have wrong types.
 */
export function adaptJobRun(raw: unknown): JobRun | null {
  if (!isObject(raw)) return null;

  // Required fields with type validation
  const id = raw.id;
  if (typeof id !== "number") return null;

  const jobName = raw.job_name;
  if (typeof jobName !== "string" || jobName.length === 0) return null;

  const backendStatus = typeof raw.status === "string" ? raw.status : "pending";
  const status = mapJobStatus(backendStatus);

  const startedAt = raw.started_at;
  if (typeof startedAt !== "string") return null;

  // Build result
  const result: JobRun = {
    id,
    jobName,
    status,
    startedAt,
    triggeredBy: "scheduler", // Default, backend doesn't track this yet
  };

  // Optional: finishedAt
  if (typeof raw.finished_at === "string") {
    result.finishedAt = raw.finished_at;
  }

  // Optional: durationMs
  if (typeof raw.duration_ms === "number") {
    result.durationMs = raw.duration_ms;
  }

  // Optional: error
  if (typeof raw.error === "string" && raw.error.length > 0) {
    result.error = raw.error;
  }

  // Optional: metrics
  if (isObject(raw.metrics)) {
    result.metadata = raw.metrics;
  }

  return result;
}

/**
 * Extract runs array from response
 *
 * Expected structure: { data: { runs: [...] } }
 */
export function extractRuns(response: unknown): unknown[] | null {
  if (!isObject(response)) {
    if (Array.isArray(response)) return response;
    return null;
  }

  // Try data.runs first (spec format)
  if (isObject(response.data) && Array.isArray(response.data.runs)) {
    return response.data.runs;
  }

  // Try root runs (alternative format)
  if (Array.isArray(response.runs)) {
    return response.runs;
  }

  return null;
}

/**
 * Extract pagination metadata from response
 */
export function extractPagination(response: unknown): JobsPagination {
  const defaults: JobsPagination = {
    total: 0,
    page: 1,
    limit: 50,
    pages: 1,
  };

  if (!isObject(response)) return defaults;

  const data = isObject(response.data) ? response.data : response;

  return {
    total: typeof data.total === "number" ? data.total : defaults.total,
    page: typeof data.page === "number" ? data.page : defaults.page,
    limit: typeof data.limit === "number" ? data.limit : defaults.limit,
    pages: typeof data.pages === "number" ? data.pages : defaults.pages,
  };
}

/**
 * Extract jobs summary from response
 */
export function extractJobsSummary(response: unknown): JobsSummary {
  if (!isObject(response)) return {};

  const summary = isObject(response.data)
    ? response.data.jobs_summary
    : response.jobs_summary;

  if (!isObject(summary)) return {};

  const result: JobsSummary = {};

  for (const [jobName, rawData] of Object.entries(summary)) {
    if (!isObject(rawData)) continue;

    result[jobName] = {
      lastRunStatus: typeof rawData.last_run_status === "string" ? rawData.last_run_status : null,
      lastRunAt: typeof rawData.last_run_at === "string" ? rawData.last_run_at : null,
      lastSuccessAt: typeof rawData.last_success_at === "string" ? rawData.last_success_at : null,
      durationMs: typeof rawData.duration_ms === "number" ? rawData.duration_ms : null,
      lastError: typeof rawData.last_error === "string" ? rawData.last_error : null,
    };
  }

  return result;
}

/**
 * Extract cache metadata from response
 */
export function extractMetadata(response: unknown): {
  generatedAt: string | null;
  cached: boolean;
  cacheAgeSeconds: number;
} {
  if (!isObject(response)) {
    return { generatedAt: null, cached: false, cacheAgeSeconds: 0 };
  }

  const generatedAt =
    typeof response.generated_at === "string" ? response.generated_at : null;
  const cached =
    typeof response.cached === "boolean" ? response.cached : false;
  const cacheAgeSeconds =
    typeof response.cache_age_seconds === "number"
      ? response.cache_age_seconds
      : 0;

  return { generatedAt, cached, cacheAgeSeconds };
}

/**
 * Parse full response to array of JobRun
 *
 * Returns null if extraction fails completely.
 * Individual invalid runs are skipped (best-effort).
 */
export function parseJobRuns(response: unknown): JobRun[] | null {
  const rawRuns = extractRuns(response);

  if (!rawRuns) {
    return null;
  }

  // Empty array is valid (no runs in time window)
  if (rawRuns.length === 0) {
    return [];
  }

  // Adapt each run, filtering out invalid ones
  const runs: JobRun[] = [];
  for (const raw of rawRuns) {
    const run = adaptJobRun(raw);
    if (run) {
      runs.push(run);
    }
  }

  return runs;
}

/**
 * Build JobDefinitions from jobs_summary
 *
 * Converts the per-job health summary into JobDefinition objects
 * for the dashboard UI.
 */
export function buildJobDefinitions(summary: JobsSummary): JobDefinition[] {
  const definitions: JobDefinition[] = [];

  // Known jobs with their schedules
  const jobSchedules: Record<string, { description: string; schedule: string }> = {
    global_sync: { description: "Sync matches from API-Football", schedule: "Every 1 min" },
    live_tick: { description: "Update live match data", schedule: "Every 10 sec" },
    stats_backfill: { description: "Capture stats for finished matches", schedule: "Every 60 min" },
    odds_sync: { description: "Sync odds for upcoming matches", schedule: "Every 6 hours" },
    fastpath: { description: "Generate LLM narratives", schedule: "Every 2 min" },
  };

  for (const [jobName, data] of Object.entries(summary)) {
    const config = jobSchedules[jobName] || {
      description: `Job: ${jobName}`,
      schedule: "Unknown",
    };

    const lastRun: JobRun | undefined = data.lastRunAt ? {
      id: 0, // Placeholder
      jobName,
      status: mapJobStatus(data.lastRunStatus || "pending"),
      startedAt: data.lastRunAt,
      finishedAt: data.lastRunAt,
      durationMs: data.durationMs || undefined,
      triggeredBy: "scheduler",
      error: data.lastError || undefined,
    } : undefined;

    definitions.push({
      name: jobName,
      description: config.description,
      schedule: config.schedule,
      lastRun,
      enabled: true, // Assume enabled if we have data
    });
  }

  return definitions;
}

/**
 * Map frontend JobStatus filter to backend status param
 */
export function mapStatusFilter(statuses: JobStatus[]): string | undefined {
  if (statuses.length === 0) return undefined;

  // Map frontend status to backend
  const backendStatuses: string[] = [];

  for (const status of statuses) {
    switch (status) {
      case "success":
        backendStatuses.push("ok");
        break;
      case "failed":
        backendStatuses.push("error");
        break;
      case "running":
        backendStatuses.push("running");
        break;
      // pending not directly mappable
    }
  }

  // Backend only supports single status filter currently
  return backendStatuses[0] || undefined;
}
