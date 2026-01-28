/**
 * Incidents API Adapter
 *
 * Safe extraction and adaptation of data from /dashboard/incidents.json
 * Designed to be resilient to partial or malformed responses.
 */

import {
  Incident,
  IncidentSeverity,
  IncidentStatus,
  IncidentType,
} from "@/lib/types";

/**
 * Pagination metadata from response
 */
export interface IncidentsPagination {
  total: number;
  page: number;
  limit: number;
  pages: number;
}

/**
 * Safely check if value is a non-null object
 */
function isObject(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

/**
 * Valid severity values from backend
 */
const VALID_SEVERITIES: Set<string> = new Set(["critical", "warning", "info"]);

/**
 * Valid status values from backend
 */
const VALID_STATUSES: Set<string> = new Set([
  "active",
  "acknowledged",
  "resolved",
]);

/**
 * Map backend type to frontend IncidentType
 * Backend uses: sentry|predictions|scheduler|llm|api_budget
 * Frontend uses: missing_prediction|job_failure|api_error|data_inconsistency|high_latency|other
 */
function mapIncidentType(backendType: string): IncidentType {
  switch (backendType) {
    case "sentry":
      return "api_error";
    case "predictions":
      return "missing_prediction";
    case "scheduler":
      return "job_failure";
    case "llm":
      return "high_latency";
    case "api_budget":
      return "api_error";
    default:
      return "other";
  }
}

/**
 * Adapt a single raw incident object to Incident type
 *
 * Returns null if critical fields are missing or have wrong types.
 */
export function adaptIncident(raw: unknown): Incident | null {
  if (!isObject(raw)) return null;

  // Required: id
  const id = raw.id;
  if (typeof id !== "number") return null;

  // Required: title
  const title = raw.title;
  if (typeof title !== "string" || title.length === 0) return null;

  // Required: severity (validated)
  const severity = raw.severity;
  if (typeof severity !== "string" || !VALID_SEVERITIES.has(severity)) {
    return null;
  }

  // Required: status (validated)
  const status = raw.status;
  if (typeof status !== "string" || !VALID_STATUSES.has(status)) {
    return null;
  }

  // Required: created_at (backend format)
  const createdAt = raw.created_at;
  if (typeof createdAt !== "string") return null;

  // Map backend type to frontend type
  const backendType = typeof raw.type === "string" ? raw.type : "other";
  const type = mapIncidentType(backendType);

  // Build result
  const result: Incident = {
    id,
    type,
    severity: severity as IncidentSeverity,
    status: status as IncidentStatus,
    createdAt,
    title,
  };

  // Optional: description
  if (typeof raw.description === "string" && raw.description.length > 0) {
    result.description = raw.description;
  }

  // Optional: acknowledged_at (persisted from PATCH endpoint)
  if (typeof raw.acknowledged_at === "string") {
    result.acknowledgedAt = raw.acknowledged_at;
  } else {
    // Legacy fallback: infer from updated_at
    const updatedAt = raw.updated_at;
    if (typeof updatedAt === "string") {
      if (status === "acknowledged" || status === "resolved") {
        result.acknowledgedAt = updatedAt;
      }
    }
  }

  // Optional: resolved_at (persisted from PATCH or auto-resolve)
  if (typeof raw.resolved_at === "string") {
    result.resolvedAt = raw.resolved_at;
  } else {
    const updatedAt = raw.updated_at;
    if (typeof updatedAt === "string" && status === "resolved") {
      result.resolvedAt = updatedAt;
    }
  }

  // Optional: timeline (persisted history events)
  if (Array.isArray(raw.timeline) && raw.timeline.length > 0) {
    result.timeline = raw.timeline
      .filter(
        (e: unknown) =>
          isObject(e) && typeof e.ts === "string" && typeof e.message === "string"
      )
      .map((e: Record<string, unknown>) => ({
        ts: e.ts as string,
        message: e.message as string,
        ...(typeof e.actor === "string" ? { actor: e.actor as "system" | "user" } : {}),
        ...(typeof e.action === "string" ? { action: e.action } : {}),
      }));
  }

  // Optional: runbook_url → create simple runbook
  const runbookUrl = raw.runbook_url;
  if (typeof runbookUrl === "string" && runbookUrl.length > 0) {
    result.runbook = {
      steps: [
        { id: "1", text: `See runbook: ${runbookUrl}`, done: false },
      ],
    };
  }

  // Optional: details (operational context dict)
  if (isObject(raw.details)) {
    result.details = raw.details as Record<string, unknown>;
  }

  return result;
}

/**
 * Extract incidents array from response
 *
 * Expected structure: { data: { incidents: [...] } }
 */
export function extractIncidents(response: unknown): unknown[] | null {
  if (!isObject(response)) {
    if (Array.isArray(response)) return response;
    return null;
  }

  // Try data.incidents first (spec format)
  if (isObject(response.data) && Array.isArray(response.data.incidents)) {
    return response.data.incidents;
  }

  // Try root incidents (alternative format)
  if (Array.isArray(response.incidents)) {
    return response.incidents;
  }

  return null;
}

/**
 * Extract pagination metadata from response
 */
export function extractPagination(response: unknown): IncidentsPagination {
  const defaults: IncidentsPagination = {
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
 * Parse full response to array of Incident
 *
 * Returns null if extraction fails completely.
 * Individual invalid incidents are skipped (best-effort).
 */
export function parseIncidents(response: unknown): Incident[] | null {
  const rawIncidents = extractIncidents(response);

  if (!rawIncidents) {
    return null;
  }

  // Empty array is valid (no incidents)
  if (rawIncidents.length === 0) {
    return [];
  }

  // Adapt each incident, filtering out invalid ones
  const incidents: Incident[] = [];
  for (const raw of rawIncidents) {
    const incident = adaptIncident(raw);
    if (incident) {
      incidents.push(incident);
    }
  }

  return incidents;
}
