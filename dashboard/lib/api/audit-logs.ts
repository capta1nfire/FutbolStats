/**
 * Audit Logs API Adapter
 *
 * Transforms backend /dashboard/audit_logs.json response to dashboard types.
 * Best-effort parsing with null-safety.
 */

import {
  AuditEventRow,
  AuditEventDetail,
  AuditSeverity,
  AuditEventType,
  AuditActorKind,
  AuditTimeRange,
} from "@/lib/types";

// ============================================================================
// Backend Types
// ============================================================================

/**
 * Backend audit event from /dashboard/audit_logs.json
 */
interface BackendAuditEvent {
  id: string;
  type: string;
  severity: string;
  actor_kind: string;
  actor_display: string;
  message: string;
  created_at: string;
  correlation_id: string | null;
  runbook_url: string | null;
}

/**
 * Backend audit logs response wrapper
 */
interface BackendAuditLogsResponse {
  generated_at: string;
  cached: boolean;
  cache_age_seconds: number;
  data: {
    events: BackendAuditEvent[];
    total: number;
    page: number;
    limit: number;
    pages: number;
  };
}

// ============================================================================
// Parsed Types (exported)
// ============================================================================

/**
 * Pagination metadata from audit logs response
 */
export interface AuditEventsPagination {
  total: number;
  page: number;
  limit: number;
  pages: number;
}

/**
 * Metadata from audit logs response
 */
export interface AuditEventsMetadata {
  generatedAt: string | null;
  cached: boolean;
  cacheAgeSeconds: number;
}

/**
 * Full parsed audit events response
 */
export interface AuditEventsApiResponse {
  events: AuditEventRow[];
  pagination: AuditEventsPagination;
  metadata: AuditEventsMetadata;
}

// ============================================================================
// Mappers
// ============================================================================

/**
 * Helper to check if value is object
 */
function isObject(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

/**
 * Map backend severity to frontend AuditSeverity
 * Normalizes: warn → warning, etc.
 */
function mapSeverity(severity: string): AuditSeverity {
  const normalized = severity.toLowerCase();
  if (normalized === "error") return "error";
  if (normalized === "warning" || normalized === "warn") return "warning";
  return "info";
}

/**
 * Map backend actor_kind to frontend AuditActorKind
 */
function mapActorKind(actorKind: string): AuditActorKind {
  const normalized = actorKind.toLowerCase();
  if (normalized === "user") return "user";
  return "system";
}

/**
 * Map backend type to frontend AuditEventType
 * Backend types: job_live_tick, job_fastpath, job_global_sync, job_odds_sync, etc.
 */
function mapEventType(type: string): AuditEventType {
  const normalized = type.toLowerCase();

  // Job types
  if (normalized.startsWith("job_")) {
    return "job_run";
  }

  // Prediction types
  if (normalized.includes("prediction") || normalized.includes("forecast")) {
    if (normalized.includes("freeze") || normalized.includes("frozen")) {
      return "prediction_frozen";
    }
    return "prediction_generated";
  }

  // Incident types
  if (normalized.includes("incident")) {
    if (normalized.includes("resolve") || normalized.includes("closed")) {
      return "incident_resolve";
    }
    return "incident_ack";
  }

  // Config types
  if (normalized.includes("config") || normalized.includes("setting")) {
    return "config_changed";
  }

  // Quality types
  if (normalized.includes("quality") || normalized.includes("check")) {
    return "data_quality_check";
  }

  // User actions
  if (normalized.includes("user") || normalized.includes("action")) {
    return "user_action";
  }

  // Default to system
  return "system";
}

/**
 * Generate numeric ID from string ID for backwards compatibility
 * Uses simple hash for deterministic conversion
 */
function stringIdToNumeric(id: string): number {
  let hash = 0;
  for (let i = 0; i < id.length; i++) {
    const char = id.charCodeAt(i);
    hash = ((hash << 5) - hash) + char;
    hash = hash & hash; // Convert to 32bit integer
  }
  return Math.abs(hash);
}

/**
 * Extended row type with internal fields for detail view
 */
export interface ExtendedAuditEventRow extends AuditEventRow {
  _correlationId?: string;
  _runbookUrl?: string;
  _originalId?: string;
  _originalType?: string;
}

/**
 * Parse a single backend event to AuditEventRow
 */
function parseEvent(raw: unknown): ExtendedAuditEventRow | null {
  if (!isObject(raw)) return null;

  const id = raw.id;
  if (typeof id !== "string" && typeof id !== "number") return null;

  const type = typeof raw.type === "string" ? raw.type : "system";
  const severity = typeof raw.severity === "string" ? raw.severity : "info";
  const actorKind = typeof raw.actor_kind === "string" ? raw.actor_kind : "system";
  const actorDisplay = typeof raw.actor_display === "string" ? raw.actor_display : "unknown";
  const message = typeof raw.message === "string" ? raw.message : "";
  const createdAt = typeof raw.created_at === "string" ? raw.created_at : new Date().toISOString();
  const correlationId = typeof raw.correlation_id === "string" ? raw.correlation_id : undefined;
  const runbookUrl = typeof raw.runbook_url === "string" ? raw.runbook_url : undefined;

  // Convert string ID to numeric for backwards compatibility with existing UI
  const numericId = typeof id === "number" ? id : stringIdToNumeric(id);

  return {
    id: numericId,
    timestamp: createdAt,
    type: mapEventType(type),
    severity: mapSeverity(severity),
    actor: mapActorKind(actorKind) === "user"
      ? { kind: "user" as const, id: 0, name: actorDisplay }
      : { kind: "system" as const, name: actorDisplay },
    message,
    entity: undefined,
    // Extended fields for detail view
    _correlationId: correlationId,
    _runbookUrl: runbookUrl,
    _originalId: typeof id === "string" ? id : undefined,
    _originalType: type,
  };
}

// ============================================================================
// Main Parser
// ============================================================================

/**
 * Parse audit events response from backend
 *
 * Expected format:
 * {
 *   generated_at: string,
 *   cached: boolean,
 *   cache_age_seconds: number,
 *   data: {
 *     events: [...],
 *     total: number,
 *     page: number,
 *     limit: number,
 *     pages: number
 *   }
 * }
 */
export function parseAuditEventsResponse(response: unknown): AuditEventsApiResponse | null {
  if (!isObject(response)) {
    return null;
  }

  // Extract metadata from root
  const generatedAt = typeof response.generated_at === "string" ? response.generated_at : null;
  const cached = typeof response.cached === "boolean" ? response.cached : false;
  const cacheAgeSeconds = typeof response.cache_age_seconds === "number" ? response.cache_age_seconds : 0;

  // Extract data object
  const data = response.data;
  if (!isObject(data)) {
    return null;
  }

  // Extract events array
  const rawEvents = data.events;
  if (!Array.isArray(rawEvents)) {
    return null;
  }

  // Parse events with best-effort (skip invalid items)
  const events: AuditEventRow[] = [];
  for (const item of rawEvents) {
    const event = parseEvent(item);
    if (event) {
      events.push(event);
    }
  }

  // Extract pagination
  const pagination: AuditEventsPagination = {
    total: typeof data.total === "number" ? data.total : events.length,
    page: typeof data.page === "number" ? data.page : 1,
    limit: typeof data.limit === "number" ? data.limit : 50,
    pages: typeof data.pages === "number" ? data.pages : 1,
  };

  return {
    events,
    pagination,
    metadata: {
      generatedAt,
      cached,
      cacheAgeSeconds,
    },
  };
}

// ============================================================================
// Detail Creator
// ============================================================================

/**
 * Create AuditEventDetail from AuditEventRow
 * Used for drawer display - extracts context from extended row data
 */
export function createEventDetail(row: AuditEventRow): AuditEventDetail {
  const extendedRow = row as ExtendedAuditEventRow;

  return {
    ...row,
    context: {
      correlationId: extendedRow._correlationId,
      env: "prod",
    },
    payload: extendedRow._originalType
      ? {
          originalType: extendedRow._originalType,
          originalId: extendedRow._originalId,
          runbookUrl: extendedRow._runbookUrl,
        }
      : undefined,
    related: undefined,
  };
}

// ============================================================================
// Legacy Exports (for backwards compatibility with old hooks)
// ============================================================================

/**
 * @deprecated Use AuditEventsMetadata instead
 */
export interface OpsLogsMetadata {
  generatedAt: string;
  limit: number;
  sinceMinutes: number;
  levelFilter: string | null;
}

/**
 * @deprecated Legacy function - use parseAuditEventsResponse
 */
export function parseOpsLogs(data: unknown): AuditEventRow[] {
  const result = parseAuditEventsResponse(data);
  return result?.events ?? [];
}

/**
 * @deprecated Legacy function - use parseAuditEventsResponse
 */
export function extractLogsMetadata(data: unknown): OpsLogsMetadata | null {
  if (!isObject(data)) return null;

  const response = data as unknown as BackendAuditLogsResponse;
  return {
    generatedAt: response.generated_at || new Date().toISOString(),
    limit: response.data?.limit || 0,
    sinceMinutes: 0, // Not available in new format
    levelFilter: null,
  };
}

/**
 * Convert AuditTimeRange to since_minutes (for legacy compatibility)
 */
export function timeRangeToMinutes(timeRange: AuditTimeRange): number {
  switch (timeRange) {
    case "1h": return 60;
    case "24h": return 1440;
    case "7d": return 10080;
    case "30d": return 43200;
    default: return 1440;
  }
}

/**
 * Map AuditSeverity to backend level param (for legacy compatibility)
 */
export function severityToLevel(severity: AuditSeverity): string {
  switch (severity) {
    case "error": return "ERROR";
    case "warning": return "WARNING";
    case "info": return "INFO";
    default: return "INFO";
  }
}
