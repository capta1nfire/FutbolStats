/**
 * Audit Types
 *
 * Types for audit trail events and logging
 */

export type AuditEventType =
  | "job_run"
  | "prediction_generated"
  | "prediction_frozen"
  | "incident_ack"
  | "incident_resolve"
  | "config_changed"
  | "data_quality_check"
  | "system"
  | "user_action";

export type AuditSeverity = "info" | "warning" | "error";

export type AuditActorKind = "user" | "system";

export type AuditActor =
  | { kind: "user"; id: number; name: string }
  | { kind: "system"; name: string };

export type AuditEntityKind = "match" | "job" | "prediction" | "incident" | "check";

export type AuditTimeRange = "1h" | "24h" | "7d" | "30d";

/**
 * Audit event row for table display
 */
export interface AuditEventRow {
  id: number;
  timestamp: string; // ISO
  type: AuditEventType;
  severity?: AuditSeverity;
  actor: AuditActor;
  message: string;
  entity?: { kind: AuditEntityKind; id: number };
}

/**
 * Context information for audit event
 */
export interface AuditEventContext {
  requestId?: string;
  correlationId?: string;
  ip?: string;
  userAgent?: string;
  env?: "prod" | "staging" | "local";
}

/**
 * Related event for context
 */
export interface AuditRelatedEvent {
  id: number;
  timestamp: string;
  message: string;
}

/**
 * Full audit event with details
 */
export interface AuditEventDetail extends AuditEventRow {
  context?: AuditEventContext;
  payload?: Record<string, unknown>;
  related?: AuditRelatedEvent[];
}

/**
 * Filters for audit events
 */
export interface AuditFilters {
  type?: AuditEventType[];
  severity?: AuditSeverity[];
  actorKind?: AuditActorKind[];
  timeRange?: AuditTimeRange;
  search?: string;
}

/**
 * Event type labels for display
 */
export const AUDIT_EVENT_TYPE_LABELS: Record<AuditEventType, string> = {
  job_run: "Job Run",
  prediction_generated: "Prediction Generated",
  prediction_frozen: "Prediction Frozen",
  incident_ack: "Incident Acknowledged",
  incident_resolve: "Incident Resolved",
  config_changed: "Config Changed",
  data_quality_check: "Data Quality Check",
  system: "System",
  user_action: "User Action",
};

/**
 * All event types for filtering
 */
export const AUDIT_EVENT_TYPES: AuditEventType[] = [
  "job_run",
  "prediction_generated",
  "prediction_frozen",
  "incident_ack",
  "incident_resolve",
  "config_changed",
  "data_quality_check",
  "system",
  "user_action",
];

/**
 * Severity labels
 */
export const AUDIT_SEVERITY_LABELS: Record<AuditSeverity, string> = {
  info: "Info",
  warning: "Warning",
  error: "Error",
};

/**
 * All severities
 */
export const AUDIT_SEVERITIES: AuditSeverity[] = ["info", "warning", "error"];

/**
 * Time range labels
 */
export const AUDIT_TIME_RANGE_LABELS: Record<AuditTimeRange, string> = {
  "1h": "Last hour",
  "24h": "Last 24 hours",
  "7d": "Last 7 days",
  "30d": "Last 30 days",
};

/**
 * All time ranges
 */
export const AUDIT_TIME_RANGES: AuditTimeRange[] = ["1h", "24h", "7d", "30d"];

/**
 * Actor kind labels
 */
export const AUDIT_ACTOR_KIND_LABELS: Record<AuditActorKind, string> = {
  user: "User",
  system: "System",
};

/**
 * All actor kinds
 */
export const AUDIT_ACTOR_KINDS: AuditActorKind[] = ["user", "system"];
