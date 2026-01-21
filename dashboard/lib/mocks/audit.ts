/**
 * Audit mock data
 * Provides deterministic data for audit trail events
 */

import {
  AuditEventRow,
  AuditEventDetail,
  AuditFilters,
  AuditEventType,
  AuditSeverity,
  AuditActor,
  AuditActorKind,
  AuditEntityKind,
  AuditEventContext,
  AuditRelatedEvent,
  AuditTimeRange,
} from "@/lib/types";
import { mockConfig, simulateDelay, checkMockError } from "./config";

/**
 * Static base timestamp for deterministic mock data
 */
const BASE_TIMESTAMP = new Date("2026-01-20T12:00:00Z").getTime();

/**
 * System actors
 */
const systemActors: AuditActor[] = [
  { kind: "system", name: "scheduler" },
  { kind: "system", name: "api-football-sync" },
  { kind: "system", name: "prediction-engine" },
  { kind: "system", name: "data-quality-monitor" },
  { kind: "system", name: "incident-detector" },
];

/**
 * User actors
 */
const userActors: AuditActor[] = [
  { kind: "user", id: 1, name: "david@futbolstats.io" },
  { kind: "user", id: 2, name: "ops@futbolstats.io" },
  { kind: "user", id: 3, name: "admin@futbolstats.io" },
];

/**
 * Event definitions with realistic messages
 */
interface EventDefinition {
  type: AuditEventType;
  severity: AuditSeverity;
  actorKind: AuditActorKind;
  message: string;
  entityKind?: AuditEntityKind;
}

const eventDefinitions: EventDefinition[] = [
  // Job events
  { type: "job_run", severity: "info", actorKind: "system", message: "global_sync completed successfully (156 matches synced)", entityKind: "job" },
  { type: "job_run", severity: "info", actorKind: "system", message: "live_tick processed 4 live matches", entityKind: "job" },
  { type: "job_run", severity: "warning", actorKind: "system", message: "stats_backfill took longer than expected (45s)", entityKind: "job" },
  { type: "job_run", severity: "error", actorKind: "system", message: "odds_sync failed: API-Football rate limit exceeded", entityKind: "job" },

  // Prediction events
  { type: "prediction_generated", severity: "info", actorKind: "system", message: "Generated prediction for Real Madrid vs Barcelona (home: 42%, draw: 28%, away: 30%)", entityKind: "prediction" },
  { type: "prediction_generated", severity: "info", actorKind: "system", message: "Generated prediction for Liverpool vs Man United (home: 55%, draw: 25%, away: 20%)", entityKind: "prediction" },
  { type: "prediction_frozen", severity: "info", actorKind: "system", message: "Frozen prediction for match #73245 (kickoff in 1h)", entityKind: "prediction" },

  // Incident events
  { type: "incident_ack", severity: "info", actorKind: "user", message: "Acknowledged incident #10001: API rate limit warning", entityKind: "incident" },
  { type: "incident_resolve", severity: "info", actorKind: "user", message: "Resolved incident #10002: Missing prediction for match #73200", entityKind: "incident" },

  // Config events
  { type: "config_changed", severity: "warning", actorKind: "user", message: "Updated NARRATIVE_PROVIDER from 'runpod' to 'gemini'" },
  { type: "config_changed", severity: "info", actorKind: "user", message: "Enabled shadow model evaluation for La Liga" },

  // Data quality events
  { type: "data_quality_check", severity: "info", actorKind: "system", message: "Prediction Coverage check passed (98%)", entityKind: "check" },
  { type: "data_quality_check", severity: "warning", actorKind: "system", message: "Live Score Latency check warning: 45s avg (threshold: 30s)", entityKind: "check" },
  { type: "data_quality_check", severity: "error", actorKind: "system", message: "Duplicate Fixtures check failed: 3 duplicates detected", entityKind: "check" },

  // System events
  { type: "system", severity: "info", actorKind: "system", message: "Application started successfully" },
  { type: "system", severity: "warning", actorKind: "system", message: "High memory usage detected (85%)" },
  { type: "system", severity: "error", actorKind: "system", message: "Database connection pool exhausted, scaling up" },

  // User actions
  { type: "user_action", severity: "info", actorKind: "user", message: "Triggered manual sync for La Liga fixtures" },
  { type: "user_action", severity: "info", actorKind: "user", message: "Viewed prediction details for match #73238" },
  { type: "user_action", severity: "info", actorKind: "user", message: "Exported analytics report: Model Performance 7d" },
];

/**
 * Create deterministic events
 */
function createEvents(count: number): AuditEventRow[] {
  return Array.from({ length: count }, (_, i) => {
    const def = eventDefinitions[i % eventDefinitions.length];
    const actor =
      def.actorKind === "system"
        ? systemActors[i % systemActors.length]
        : userActors[i % userActors.length];

    return {
      id: 5000 + i,
      timestamp: new Date(BASE_TIMESTAMP - i * 180000).toISOString(), // 3 min apart
      type: def.type,
      severity: def.severity,
      actor,
      message: def.message,
      entity: def.entityKind
        ? { kind: def.entityKind, id: 10000 + (i % 50) }
        : undefined,
    };
  });
}

/**
 * Create context for an event
 */
function createContext(eventId: number): AuditEventContext {
  return {
    requestId: `req_${eventId}_${Math.random().toString(36).substr(2, 9)}`,
    correlationId: `corr_${Math.floor(eventId / 10)}_batch`,
    ip: eventId % 3 === 0 ? "10.0.0.1" : eventId % 3 === 1 ? "10.0.0.2" : undefined,
    userAgent: eventId % 2 === 0 ? "FutbolStats-Scheduler/1.0" : "Mozilla/5.0 (Macintosh)",
    env: "prod",
  };
}

/**
 * Create payload for an event
 */
function createPayload(event: AuditEventRow): Record<string, unknown> {
  switch (event.type) {
    case "job_run":
      return {
        jobName: "global_sync",
        duration_ms: 2340,
        items_processed: 156,
        errors: 0,
        started_at: event.timestamp,
      };
    case "prediction_generated":
      return {
        match_id: event.entity?.id ?? 73245,
        model: "xgb_v1.0.0",
        probabilities: { home: 0.42, draw: 0.28, away: 0.30 },
        confidence: 0.72,
        features_used: 14,
      };
    case "incident_ack":
    case "incident_resolve":
      return {
        incident_id: event.entity?.id ?? 10001,
        action: event.type === "incident_ack" ? "acknowledge" : "resolve",
        notes: "Checked and resolved manually",
      };
    case "config_changed":
      return {
        key: "NARRATIVE_PROVIDER",
        old_value: "runpod",
        new_value: "gemini",
        changed_by: (event.actor as { name: string }).name,
      };
    case "data_quality_check":
      return {
        check_name: "Prediction Coverage",
        status: event.severity === "error" ? "failing" : event.severity === "warning" ? "warning" : "passing",
        current_value: "98%",
        threshold: "95%",
      };
    default:
      return {
        event_type: event.type,
        timestamp: event.timestamp,
      };
  }
}

/**
 * Create related events
 */
function createRelatedEvents(eventId: number, count: number): AuditRelatedEvent[] {
  return Array.from({ length: count }, (_, i) => ({
    id: eventId - (i + 1),
    timestamp: new Date(BASE_TIMESTAMP - (eventId - 5000 + i + 1) * 180000).toISOString(),
    message: `Related event ${i + 1}: Previous action in sequence`,
  }));
}

// Pre-generated datasets
const normalDataset = createEvents(40);
const largeDataset = createEvents(200);

/**
 * Get time range in milliseconds
 */
function getTimeRangeMs(range: AuditTimeRange): number {
  switch (range) {
    case "1h":
      return 60 * 60 * 1000;
    case "24h":
      return 24 * 60 * 60 * 1000;
    case "7d":
      return 7 * 24 * 60 * 60 * 1000;
    case "30d":
      return 30 * 24 * 60 * 60 * 1000;
  }
}

/**
 * Get audit events based on scenario
 */
export async function getAuditEventsMock(
  filters?: AuditFilters
): Promise<AuditEventRow[]> {
  await simulateDelay();
  checkMockError();

  let data: AuditEventRow[];

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
    if (filters.type && filters.type.length > 0) {
      data = data.filter((e) => filters.type!.includes(e.type));
    }
    if (filters.severity && filters.severity.length > 0) {
      data = data.filter((e) => e.severity && filters.severity!.includes(e.severity));
    }
    if (filters.actorKind && filters.actorKind.length > 0) {
      data = data.filter((e) => filters.actorKind!.includes(e.actor.kind));
    }
    if (filters.timeRange) {
      const rangeMs = getTimeRangeMs(filters.timeRange);
      const cutoff = BASE_TIMESTAMP - rangeMs;
      data = data.filter((e) => new Date(e.timestamp).getTime() >= cutoff);
    }
    if (filters.search) {
      const search = filters.search.toLowerCase();
      data = data.filter(
        (e) =>
          e.message.toLowerCase().includes(search) ||
          e.type.toLowerCase().includes(search) ||
          (e.actor.kind === "user" && e.actor.name.toLowerCase().includes(search)) ||
          (e.actor.kind === "system" && e.actor.name.toLowerCase().includes(search))
      );
    }
  }

  return data;
}

/**
 * Get a single event by ID with full details
 */
export async function getAuditEventMock(
  id: number
): Promise<AuditEventDetail | null> {
  await simulateDelay(300);
  checkMockError();

  const allEvents = mockConfig.scenario === "large" ? largeDataset : normalDataset;
  const event = allEvents.find((e) => e.id === id);

  if (!event) return null;

  return {
    ...event,
    context: createContext(id),
    payload: createPayload(event),
    related: createRelatedEvents(id, 3),
  };
}

/**
 * Get counts per event type
 */
export function getAuditTypeCountsMock(): Record<AuditEventType, number> {
  const allEvents = mockConfig.scenario === "large" ? largeDataset : normalDataset;
  const counts: Record<AuditEventType, number> = {
    job_run: 0,
    prediction_generated: 0,
    prediction_frozen: 0,
    incident_ack: 0,
    incident_resolve: 0,
    config_changed: 0,
    data_quality_check: 0,
    system: 0,
    user_action: 0,
  };
  allEvents.forEach((e) => counts[e.type]++);
  return counts;
}

/**
 * Get counts per severity
 */
export function getAuditSeverityCountsMock(): Record<AuditSeverity, number> {
  const allEvents = mockConfig.scenario === "large" ? largeDataset : normalDataset;
  const counts: Record<AuditSeverity, number> = { info: 0, warning: 0, error: 0 };
  allEvents.forEach((e) => {
    if (e.severity) counts[e.severity]++;
  });
  return counts;
}
