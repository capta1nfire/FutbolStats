/**
 * Audit Logs API Adapter
 *
 * Transforms backend /dashboard/ops/logs.json response to dashboard AuditEventRow types.
 * Best-effort parsing with null-safety.
 *
 * Note: This is "Ops Logs" from backend, not a true audit trail.
 * The UI should reflect this distinction.
 */

import {
  AuditEventRow,
  AuditEventDetail,
  AuditSeverity,
  AuditEventType,
  AuditTimeRange,
} from "@/lib/types";

/**
 * Backend log entry from /dashboard/ops/logs.json
 */
interface BackendLogEntry {
  ts_utc: string;
  level: string; // INFO, WARNING, ERROR, DEBUG, CRITICAL
  logger: string; // app.main, app.scheduler, app.etl.api_football, etc.
  message: string;
}

/**
 * Backend logs response
 */
interface BackendLogsResponse {
  generated_at: string;
  limit: number;
  since_minutes: number;
  level: string | null;
  mode: string | null;
  entries: BackendLogEntry[];
}

/**
 * Parsed ops logs metadata
 */
export interface OpsLogsMetadata {
  generatedAt: string;
  limit: number;
  sinceMinutes: number;
  levelFilter: string | null;
}

/**
 * Map backend level to AuditSeverity
 */
function mapLevelToSeverity(level: string): AuditSeverity {
  const normalized = level.toUpperCase();
  if (normalized === "ERROR" || normalized === "CRITICAL") {
    return "error";
  }
  if (normalized === "WARNING" || normalized === "WARN") {
    return "warning";
  }
  return "info";
}

/**
 * Derive AuditEventType from logger name
 */
function deriveEventType(logger: string): AuditEventType {
  const normalized = logger.toLowerCase();

  if (normalized.includes("scheduler")) {
    return "job_run";
  }
  if (normalized.includes("etl")) {
    return "job_run";
  }
  if (normalized.includes("prediction") || normalized.includes("ml")) {
    return "prediction_generated";
  }
  if (normalized.includes("incident")) {
    return "incident_ack";
  }
  if (normalized.includes("config")) {
    return "config_changed";
  }
  if (normalized.includes("quality") || normalized.includes("check")) {
    return "data_quality_check";
  }

  return "system";
}

/**
 * Generate stable ID from log entry
 * Uses timestamp + logger + first 50 chars of message for uniqueness
 */
function generateStableId(entry: BackendLogEntry, index: number): number {
  // Simple hash from string - not cryptographic, just stable
  const str = `${entry.ts_utc}|${entry.logger}|${entry.message.slice(0, 50)}`;
  let hash = 0;
  for (let i = 0; i < str.length; i++) {
    const char = str.charCodeAt(i);
    hash = ((hash << 5) - hash) + char;
    hash = hash & hash; // Convert to 32bit integer
  }
  // Ensure positive and add index for collision safety
  return Math.abs(hash) + index;
}

/**
 * Parse a single backend log entry to AuditEventRow
 */
function parseLogEntry(entry: BackendLogEntry, index: number): AuditEventRow {
  return {
    id: generateStableId(entry, index),
    timestamp: entry.ts_utc || new Date().toISOString(),
    type: deriveEventType(entry.logger || ""),
    severity: mapLevelToSeverity(entry.level || "INFO"),
    actor: {
      kind: "system",
      name: entry.logger || "unknown",
    },
    message: entry.message || "",
  };
}

/**
 * Parse backend logs response to AuditEventRow[]
 */
export function parseOpsLogs(data: unknown): AuditEventRow[] {
  if (!data || typeof data !== "object") {
    return [];
  }

  const response = data as BackendLogsResponse;
  const entries = response.entries;

  if (!Array.isArray(entries)) {
    return [];
  }

  return entries.map((entry, index) => parseLogEntry(entry, index));
}

/**
 * Extract metadata from logs response
 */
export function extractLogsMetadata(data: unknown): OpsLogsMetadata | null {
  if (!data || typeof data !== "object") {
    return null;
  }

  const response = data as BackendLogsResponse;

  return {
    generatedAt: response.generated_at || new Date().toISOString(),
    limit: response.limit || 0,
    sinceMinutes: response.since_minutes || 0,
    levelFilter: response.level,
  };
}

/**
 * Convert AuditTimeRange to since_minutes for API
 */
export function timeRangeToMinutes(timeRange: AuditTimeRange): number {
  switch (timeRange) {
    case "1h":
      return 60;
    case "24h":
      return 1440;
    case "7d":
      return 10080;
    case "30d":
      return 43200;
    default:
      return 1440; // Default to 24h
  }
}

/**
 * Map AuditSeverity to backend level param
 */
export function severityToLevel(severity: AuditSeverity): string {
  switch (severity) {
    case "error":
      return "ERROR";
    case "warning":
      return "WARNING";
    case "info":
      return "INFO";
    default:
      return "INFO";
  }
}

/**
 * Create AuditEventDetail from AuditEventRow
 * Used for drawer display - adds context from the log data
 */
export function createEventDetail(row: AuditEventRow): AuditEventDetail {
  return {
    ...row,
    context: {
      env: "prod",
    },
    payload: undefined,
    related: undefined,
  };
}
