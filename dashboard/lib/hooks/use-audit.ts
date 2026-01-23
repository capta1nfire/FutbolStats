/**
 * Audit hooks using TanStack Query
 *
 * Supports both mock data and real API with fallback.
 */

import { useQuery } from "@tanstack/react-query";
import {
  AuditEventRow,
  AuditEventDetail,
  AuditFilters,
  AuditEventType,
  AuditSeverity,
  AuditTimeRange,
} from "@/lib/types";
import {
  getAuditEventsMock,
  getAuditEventMock,
  getAuditTypeCountsMock,
  getAuditSeverityCountsMock,
} from "@/lib/mocks";
import {
  parseOpsLogs,
  extractLogsMetadata,
  timeRangeToMinutes,
  severityToLevel,
  createEventDetail,
  OpsLogsMetadata,
} from "@/lib/api/audit-logs";

/**
 * Fetch all audit events with optional filters
 */
export function useAuditEvents(filters?: AuditFilters) {
  return useQuery<AuditEventRow[]>({
    queryKey: ["audit", "events", filters],
    queryFn: () => getAuditEventsMock(filters),
  });
}

/**
 * Fetch a single audit event with full details
 */
export function useAuditEvent(id: number | null) {
  return useQuery<AuditEventDetail | null>({
    queryKey: ["audit", "event", id],
    queryFn: () => (id ? getAuditEventMock(id) : Promise.resolve(null)),
    enabled: id !== null,
  });
}

/**
 * Get type counts (synchronous, for filter badges)
 */
export function useAuditTypeCounts(): Record<AuditEventType, number> {
  return getAuditTypeCountsMock();
}

/**
 * Get severity counts (synchronous, for filter badges)
 */
export function useAuditSeverityCounts(): Record<AuditSeverity, number> {
  return getAuditSeverityCountsMock();
}

// ============================================================================
// Real API Hooks (with mock fallback)
// ============================================================================

/**
 * API filter params for ops logs
 */
interface OpsLogsApiParams {
  timeRange?: AuditTimeRange;
  severity?: AuditSeverity; // Single severity for backend filter
  limit?: number;
}

/**
 * Fetch ops logs from API
 */
async function fetchOpsLogsApi(
  params: OpsLogsApiParams
): Promise<{ events: AuditEventRow[]; metadata: OpsLogsMetadata | null }> {
  const queryParams = new URLSearchParams();

  // Map timeRange to since_minutes
  if (params.timeRange) {
    queryParams.set("since_minutes", String(timeRangeToMinutes(params.timeRange)));
  } else {
    queryParams.set("since_minutes", "1440"); // Default 24h
  }

  // Set limit
  queryParams.set("limit", String(params.limit || 100));

  // Map severity to level
  if (params.severity) {
    queryParams.set("level", severityToLevel(params.severity));
  }

  const response = await fetch(`/api/audit-logs?${queryParams.toString()}`);

  if (!response.ok) {
    throw new Error(`API error: ${response.status}`);
  }

  const data = await response.json();
  const events = parseOpsLogs(data);
  const metadata = extractLogsMetadata(data);

  return { events, metadata };
}

/**
 * Fetch ops logs from real API with mock fallback
 *
 * Note: This fetches "Ops Logs" from backend, not a true audit trail.
 * Filters are applied as follows:
 * - timeRange -> since_minutes
 * - severity (single) -> level param
 * - type/actorKind/search -> client-side filtering (backend doesn't support)
 */
export function useOpsLogsApi(filters?: AuditFilters, limit: number = 100) {
  // Extract API-supported params
  const apiParams: OpsLogsApiParams = {
    timeRange: filters?.timeRange,
    // Only pass single severity to API if exactly one is selected
    severity: filters?.severity?.length === 1 ? filters.severity[0] : undefined,
    limit,
  };

  const apiQuery = useQuery<
    { events: AuditEventRow[]; metadata: OpsLogsMetadata | null },
    Error
  >({
    queryKey: ["audit", "ops-logs", "api", apiParams],
    queryFn: () => fetchOpsLogsApi(apiParams),
    staleTime: 30 * 1000, // 30 seconds
    retry: 1,
  });

  // Fallback to mock on error
  const mockQuery = useQuery<AuditEventRow[]>({
    queryKey: ["audit", "events", "mock", filters],
    queryFn: () => getAuditEventsMock(filters),
    enabled: apiQuery.isError,
  });

  const isApiDegraded = apiQuery.isError;

  // Get base data
  let events = isApiDegraded
    ? (mockQuery.data ?? [])
    : (apiQuery.data?.events ?? []);

  // Apply client-side filters that backend doesn't support
  if (!isApiDegraded && filters) {
    // Filter by type (client-side)
    if (filters.type && filters.type.length > 0) {
      events = events.filter((e) => filters.type!.includes(e.type));
    }

    // Filter by severity (client-side if multiple selected)
    if (filters.severity && filters.severity.length > 1) {
      events = events.filter((e) => e.severity && filters.severity!.includes(e.severity));
    }

    // Filter by actorKind (client-side - all ops logs are "system")
    if (filters.actorKind && filters.actorKind.length > 0) {
      events = events.filter((e) => filters.actorKind!.includes(e.actor.kind));
    }

    // Search filter (client-side)
    if (filters.search) {
      const searchLower = filters.search.toLowerCase();
      events = events.filter(
        (e) =>
          e.message.toLowerCase().includes(searchLower) ||
          e.actor.name.toLowerCase().includes(searchLower)
      );
    }
  }

  const metadata = isApiDegraded ? null : (apiQuery.data?.metadata ?? null);
  const isLoading = apiQuery.isLoading || (isApiDegraded && mockQuery.isLoading);

  return {
    data: events,
    metadata,
    isLoading,
    error: isApiDegraded ? null : apiQuery.error,
    isApiDegraded,
    refetch: apiQuery.refetch,
  };
}

/**
 * Get event detail from row (for drawer)
 * Since ops logs don't have a detail endpoint, we create detail from row
 */
export function useOpsLogDetail(event: AuditEventRow | null) {
  // No API call needed - create detail from row data
  const detail = event ? createEventDetail(event) : null;

  return {
    data: detail,
    isLoading: false,
    error: null,
    isApiDegraded: false,
  };
}

// Re-export types for convenience
export type { OpsLogsMetadata };
