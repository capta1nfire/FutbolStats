/**
 * Incident mock data
 * Provides deterministic factories and datasets for testing
 * All data is static to avoid hydration mismatches
 */

import {
  Incident,
  IncidentSeverity,
  IncidentStatus,
  IncidentType,
  IncidentFilters,
  RunbookStep,
  TimelineEvent,
} from "@/lib/types";
import { mockConfig, simulateDelay, checkMockError } from "./config";

/**
 * Static base timestamp for deterministic mock data
 */
const BASE_TIMESTAMP = new Date("2026-01-20T10:00:00Z").getTime();

// Sample data for variety
const incidentTypes: IncidentType[] = [
  "missing_prediction",
  "job_failure",
  "api_error",
  "data_inconsistency",
  "high_latency",
  "other",
];

const severities: IncidentSeverity[] = ["critical", "warning", "info"];
const statuses: IncidentStatus[] = ["active", "acknowledged", "resolved"];

const incidentTitles: Record<IncidentType, string[]> = {
  missing_prediction: [
    "No prediction for Real Madrid vs Barcelona",
    "Missing prediction: Premier League match",
    "Prediction gap: Serie A fixture",
    "Coverage gap: Bundesliga match",
  ],
  job_failure: [
    "global_sync job failed: API timeout",
    "live_tick job failed: Connection refused",
    "stats_backfill job failed: Rate limit exceeded",
    "fastpath job failed: LLM service unavailable",
  ],
  api_error: [
    "API-Football 503: Service unavailable",
    "API-Football rate limit exceeded",
    "API-Football invalid response format",
    "RunPod endpoint timeout",
  ],
  data_inconsistency: [
    "Score mismatch detected: match 12345",
    "Duplicate fixture entries found",
    "Invalid kickoff time: past date",
    "Missing team data for fixture",
  ],
  high_latency: [
    "API response time >5s: /predictions/upcoming",
    "Database query slow: match lookup",
    "LLM generation timeout: narrative",
    "Scheduler job queue backlog",
  ],
  other: [
    "Unexpected error in prediction pipeline",
    "Configuration mismatch detected",
    "Memory usage above threshold",
    "Disk space warning",
  ],
};

const runbookSteps: Record<IncidentType, RunbookStep[]> = {
  missing_prediction: [
    { id: "1", text: "Check if match exists in database" },
    { id: "2", text: "Verify odds data is available" },
    { id: "3", text: "Run manual prediction generation" },
    { id: "4", text: "Verify prediction appears in /predictions/upcoming" },
  ],
  job_failure: [
    { id: "1", text: "Check Railway logs for error details" },
    { id: "2", text: "Verify external API status" },
    { id: "3", text: "Restart job manually if needed" },
    { id: "4", text: "Monitor for recurrence" },
  ],
  api_error: [
    { id: "1", text: "Check API-Football status page" },
    { id: "2", text: "Verify API key is valid" },
    { id: "3", text: "Check rate limit usage" },
    { id: "4", text: "Wait for service recovery or contact support" },
  ],
  data_inconsistency: [
    { id: "1", text: "Identify affected records" },
    { id: "2", text: "Compare with source API data" },
    { id: "3", text: "Run data correction script" },
    { id: "4", text: "Verify data integrity" },
  ],
  high_latency: [
    { id: "1", text: "Check system metrics (CPU, memory, DB)" },
    { id: "2", text: "Review recent query patterns" },
    { id: "3", text: "Check for blocking queries" },
    { id: "4", text: "Consider scaling or optimization" },
  ],
  other: [
    { id: "1", text: "Review error logs" },
    { id: "2", text: "Identify root cause" },
    { id: "3", text: "Apply fix or workaround" },
    { id: "4", text: "Document resolution" },
  ],
};

/**
 * Create a deterministic mock incident based on index
 */
function createDeterministicIncident(index: number): Incident {
  const typeIndex = index % incidentTypes.length;
  const severityIndex = index % severities.length;
  const statusIndex = index % statuses.length;

  const type = incidentTypes[typeIndex];
  const severity = severities[severityIndex];
  const status = statuses[statusIndex];

  // Deterministic timestamps
  const hoursAgo = index * 2 + (index % 5);
  const createdAt = new Date(BASE_TIMESTAMP - hoursAgo * 3600000).toISOString();

  // Title from pool based on type and index
  const titlePool = incidentTitles[type];
  const title = titlePool[index % titlePool.length];

  // Create deterministic runbook with some steps marked done for resolved incidents
  const steps = runbookSteps[type].map((step, stepIndex) => ({
    ...step,
    done: status === "resolved" ? true : status === "acknowledged" ? stepIndex < 2 : false,
  }));

  // Create deterministic timeline
  const timeline: TimelineEvent[] = [
    { ts: createdAt, message: "Incident created automatically by monitoring system" },
  ];

  if (status === "acknowledged" || status === "resolved") {
    const ackTime = new Date(BASE_TIMESTAMP - hoursAgo * 3600000 + 1800000).toISOString();
    timeline.push({ ts: ackTime, message: "Incident acknowledged by operator" });
  }

  if (status === "resolved") {
    const resolveTime = new Date(BASE_TIMESTAMP - hoursAgo * 3600000 + 7200000).toISOString();
    timeline.push({ ts: resolveTime, message: "Incident resolved - root cause addressed" });
  }

  const incident: Incident = {
    id: 10000 - index,
    type,
    severity,
    status,
    createdAt,
    title,
    description: `${title}. This incident was detected by the automated monitoring system and requires attention.`,
    runbook: { steps },
    timeline,
  };

  // Add entity reference for some incidents
  if (type === "missing_prediction" && index % 2 === 0) {
    incident.entity = { kind: "match", id: 1000 + index };
  } else if (type === "job_failure" && index % 3 === 0) {
    incident.entity = { kind: "job", id: 99900 + index };
  }

  // Add timestamps for acknowledged/resolved
  if (status === "acknowledged" || status === "resolved") {
    incident.acknowledgedAt = new Date(BASE_TIMESTAMP - hoursAgo * 3600000 + 1800000).toISOString();
  }
  if (status === "resolved") {
    incident.resolvedAt = new Date(BASE_TIMESTAMP - hoursAgo * 3600000 + 7200000).toISOString();
  }

  return incident;
}

/**
 * Create multiple deterministic mock incidents
 */
function createDeterministicIncidents(count: number): Incident[] {
  return Array.from({ length: count }, (_, i) => createDeterministicIncident(i));
}

// Pre-generated static datasets
const normalDataset: Incident[] = createDeterministicIncidents(25);
const largeDataset: Incident[] = createDeterministicIncidents(120);

/**
 * Get incidents based on current mock scenario
 */
export async function getIncidentsMock(
  filters?: IncidentFilters
): Promise<Incident[]> {
  await simulateDelay();
  checkMockError();

  let data: Incident[];

  switch (mockConfig.scenario) {
    case "empty":
      data = [];
      break;
    case "large":
      data = [...largeDataset];
      break;
    default:
      data = [...normalDataset];
  }

  // Apply filters
  if (filters) {
    if (filters.status && filters.status.length > 0) {
      data = data.filter((i) => filters.status!.includes(i.status));
    }
    if (filters.severity && filters.severity.length > 0) {
      data = data.filter((i) => filters.severity!.includes(i.severity));
    }
    if (filters.type && filters.type.length > 0) {
      data = data.filter((i) => filters.type!.includes(i.type));
    }
    if (filters.search) {
      const search = filters.search.toLowerCase();
      data = data.filter(
        (i) =>
          i.title.toLowerCase().includes(search) ||
          i.description?.toLowerCase().includes(search) ||
          i.type.toLowerCase().includes(search)
      );
    }
  }

  return data;
}

/**
 * Get a single incident by ID
 */
export async function getIncidentByIdMock(
  id: number
): Promise<Incident | null> {
  await simulateDelay(300);
  checkMockError();

  const allIncidents =
    mockConfig.scenario === "large" ? largeDataset : normalDataset;
  return allIncidents.find((i) => i.id === id) ?? null;
}

/**
 * Get counts per status
 */
export function getIncidentStatusCountsMock(): Record<IncidentStatus, number> {
  const allIncidents =
    mockConfig.scenario === "large" ? largeDataset : normalDataset;
  return {
    active: allIncidents.filter((i) => i.status === "active").length,
    acknowledged: allIncidents.filter((i) => i.status === "acknowledged").length,
    resolved: allIncidents.filter((i) => i.status === "resolved").length,
  };
}

/**
 * Get counts per severity
 */
export function getSeverityCountsMock(): Record<IncidentSeverity, number> {
  const allIncidents =
    mockConfig.scenario === "large" ? largeDataset : normalDataset;
  return {
    critical: allIncidents.filter((i) => i.severity === "critical").length,
    warning: allIncidents.filter((i) => i.severity === "warning").length,
    info: allIncidents.filter((i) => i.severity === "info").length,
  };
}

// Legacy exports for backwards compatibility
export function createMockIncident(overrides?: Partial<Incident>): Incident {
  return { ...createDeterministicIncident(0), ...overrides };
}

export function createMockIncidents(count: number): Incident[] {
  return createDeterministicIncidents(count);
}
