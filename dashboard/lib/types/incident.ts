/**
 * Incident Types
 *
 * Represents system incidents for monitoring and alerting
 */

export type IncidentSeverity = "critical" | "warning" | "info";

export type IncidentStatus = "active" | "acknowledged" | "resolved";

export type IncidentType =
  | "missing_prediction"
  | "job_failure"
  | "api_error"
  | "data_inconsistency"
  | "high_latency"
  | "other";

export interface RunbookStep {
  id: string;
  text: string;
  done?: boolean;
}

export interface TimelineEvent {
  ts: string; // ISO timestamp
  message: string;
  actor?: "system" | "user"; // who performed the action
  action?: string; // created|acknowledged|resolved|reopened|auto_resolved|updated
}

export interface Incident {
  id: number;
  type: IncidentType;
  severity: IncidentSeverity;
  status: IncidentStatus;
  createdAt: string; // ISO timestamp
  title: string;
  description?: string;
  entity?: {
    kind: "match" | "job" | "prediction";
    id: number;
  };
  runbook?: {
    steps: RunbookStep[];
  };
  timeline?: TimelineEvent[];
  acknowledgedAt?: string; // ISO timestamp
  resolvedAt?: string; // ISO timestamp
  details?: Record<string, unknown>; // Operational context from backend
}

export interface IncidentFilters {
  status?: IncidentStatus[];
  severity?: IncidentSeverity[];
  type?: IncidentType[];
  search?: string;
}

/**
 * Incident type labels for display
 */
export const INCIDENT_TYPE_LABELS: Record<IncidentType, string> = {
  missing_prediction: "Missing Prediction",
  job_failure: "Job Failure",
  api_error: "API Error",
  data_inconsistency: "Data Inconsistency",
  high_latency: "High Latency",
  other: "Other",
};

/**
 * All incident types for filtering
 */
export const INCIDENT_TYPES: IncidentType[] = [
  "missing_prediction",
  "job_failure",
  "api_error",
  "data_inconsistency",
  "high_latency",
  "other",
];

/**
 * All incident severities
 */
export const INCIDENT_SEVERITIES: IncidentSeverity[] = [
  "critical",
  "warning",
  "info",
];

/**
 * All incident statuses
 */
export const INCIDENT_STATUSES: IncidentStatus[] = [
  "active",
  "acknowledged",
  "resolved",
];
