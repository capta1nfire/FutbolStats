/**
 * Audit hooks using TanStack Query
 *
 * Supports real API data from /dashboard/audit_logs.json with mock fallback.
 */

import { useQuery } from "@tanstack/react-query";
import {
  AuditEventRow,
  AuditEventDetail,
  AuditFilters,
  AuditEventType,
  AuditSeverity,
  AuditTimeRange,
  AuditActorKind,
} from "@/lib/types";
import {
  getAuditEventsMock,
  getAuditEventMock,
  getAuditTypeCountsMock,
  getAuditSeverityCountsMock,
} from "@/lib/mocks";
import {
  parseAuditEventsResponse,
  createEventDetail,
  AuditEventsPagination,
  AuditEventsMetadata,
} from "@/lib/api/audit-logs";

// ============================================================================
// Legacy Mock Hooks (kept for backwards compatibility)
// ============================================================================

/**
 * Fetch all audit events with optional filters (mock only)
 * @deprecated Use useAuditEventsApi instead
 */
export function useAuditEvents(filters?: AuditFilters) {
  return useQuery<AuditEventRow[]>({
    queryKey: ["audit", "events", filters],
    queryFn: () => getAuditEventsMock(filters),
  });
}

/**
 * Fetch a single audit event with full details (mock only)
 * @deprecated Use useAuditEventDetail instead
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
// Real API Hooks
// ============================================================================

/**
 * Query params for audit events API
 */
interface AuditEventsQueryParams {
  type?: AuditEventType[];
  severity?: AuditSeverity[];
  actorKind?: AuditActorKind[];
  q?: string;
  range?: AuditTimeRange;
  page?: number;
  limit?: number;
}

/**
 * Internal response from fetch
 */
interface AuditEventsData {
  events: AuditEventRow[];
  pagination: AuditEventsPagination;
  metadata: AuditEventsMetadata;
}

/**
 * Result from useAuditEventsApi hook
 */
export interface UseAuditEventsApiResult {
  events: AuditEventRow[];
  pagination: AuditEventsPagination;
  metadata: AuditEventsMetadata;
  isLoading: boolean;
  error: Error | null;
  isDegraded: boolean;
  refetch: () => void;
}

/**
 * Build query string from params
 */
function buildAuditQueryString(params: AuditEventsQueryParams): string {
  const searchParams = new URLSearchParams();

  // Multi-value params
  if (params.type && params.type.length > 0) {
    for (const t of params.type) {
      searchParams.append("type", t);
    }
  }
  if (params.severity && params.severity.length > 0) {
    for (const s of params.severity) {
      searchParams.append("severity", s);
    }
  }
  if (params.actorKind && params.actorKind.length > 0) {
    for (const a of params.actorKind) {
      searchParams.append("actor_kind", a);
    }
  }

  // Single value params
  if (params.q) searchParams.set("q", params.q);
  if (params.range) searchParams.set("range", params.range);
  if (params.page) searchParams.set("page", params.page.toString());
  if (params.limit) searchParams.set("limit", params.limit.toString());

  const qs = searchParams.toString();
  return qs ? `?${qs}` : "";
}

/**
 * Fetch audit events from proxy endpoint
 */
async function fetchAuditEvents(params: AuditEventsQueryParams): Promise<AuditEventsData> {
  const queryString = buildAuditQueryString(params);
  const response = await fetch(`/api/audit-logs${queryString}`, {
    method: "GET",
    headers: {
      Accept: "application/json",
    },
  });

  if (!response.ok) {
    throw new Error(`API error: ${response.status}`);
  }

  const data = await response.json();
  const parsed = parseAuditEventsResponse(data);

  if (!parsed) {
    throw new Error("Failed to parse audit events response");
  }

  return {
    events: parsed.events,
    pagination: parsed.pagination,
    metadata: parsed.metadata,
  };
}

/**
 * Fetch audit events from real API with mock fallback
 *
 * Features:
 * - Server-side filtering (type, severity, actor_kind, q, range)
 * - Real pagination (page, limit, total, pages)
 * - isDegraded indicator for fallback state
 * - Automatic mock fallback on API error
 *
 * Usage:
 * ```tsx
 * const {
 *   events,
 *   pagination,
 *   isDegraded,
 *   isLoading,
 *   refetch,
 * } = useAuditEventsApi({
 *   type: ["job_run"],
 *   severity: ["error"],
 *   range: "24h",
 *   page: 1,
 *   limit: 50,
 * });
 * ```
 */
export function useAuditEventsApi(options?: {
  type?: AuditEventType[];
  severity?: AuditSeverity[];
  actorKind?: AuditActorKind[];
  q?: string;
  range?: AuditTimeRange;
  page?: number;
  limit?: number;
  enabled?: boolean;
}): UseAuditEventsApiResult {
  const {
    type,
    severity,
    actorKind,
    q,
    range,
    page = 1,
    limit = 50,
    enabled = true,
  } = options || {};

  const queryParams: AuditEventsQueryParams = {
    type: type && type.length > 0 ? type : undefined,
    severity: severity && severity.length > 0 ? severity : undefined,
    actorKind: actorKind && actorKind.length > 0 ? actorKind : undefined,
    q: q || undefined,
    range: range || undefined,
    page,
    limit,
  };

  // Fetch from API
  const apiQuery = useQuery<AuditEventsData, Error>({
    queryKey: ["audit", "events", "api", queryParams],
    queryFn: () => fetchAuditEvents(queryParams),
    staleTime: 90_000, // 90 seconds (aligned with backend TTL)
    retry: 1,
    enabled,
  });

  // Build filters for mock fallback (client-side filtering)
  const mockFilters: AuditFilters = {
    type: type,
    severity: severity,
    actorKind: actorKind,
    timeRange: range,
    search: q,
  };

  // Fallback to mock on error
  const mockQuery = useQuery<AuditEventRow[], Error>({
    queryKey: ["audit", "events", "mock", mockFilters],
    queryFn: () => getAuditEventsMock(mockFilters),
    enabled: apiQuery.isError && enabled,
  });

  // Determine which data to use
  const isDegraded = apiQuery.isError;
  const events = isDegraded
    ? (mockQuery.data ?? [])
    : (apiQuery.data?.events ?? []);
  const pagination: AuditEventsPagination = isDegraded
    ? { total: mockQuery.data?.length ?? 0, page: 1, limit: 50, pages: 1 }
    : (apiQuery.data?.pagination ?? { total: 0, page: 1, limit: 50, pages: 1 });
  const metadata: AuditEventsMetadata = isDegraded
    ? { generatedAt: null, cached: false, cacheAgeSeconds: 0 }
    : (apiQuery.data?.metadata ?? { generatedAt: null, cached: false, cacheAgeSeconds: 0 });
  const isLoading = apiQuery.isLoading || (isDegraded && mockQuery.isLoading);
  const error = isDegraded ? null : apiQuery.error; // Suppress if mock fallback works

  return {
    events,
    pagination,
    metadata,
    isLoading,
    error,
    isDegraded,
    refetch: () => apiQuery.refetch(),
  };
}

/**
 * Get event detail from row (for drawer)
 * Since the list already contains all needed data, we create detail from row
 */
export function useAuditEventDetail(event: AuditEventRow | null) {
  const detail = event ? createEventDetail(event) : null;

  return {
    data: detail,
    isLoading: false,
    error: null,
    isDegraded: false,
  };
}

// ============================================================================
// Legacy Exports (for backwards compatibility)
// ============================================================================

/**
 * @deprecated Use useAuditEventsApi instead
 */
export function useOpsLogsApi(filters?: AuditFilters, limit: number = 100) {
  const result = useAuditEventsApi({
    type: filters?.type,
    severity: filters?.severity,
    actorKind: filters?.actorKind,
    q: filters?.search,
    range: filters?.timeRange,
    limit,
  });

  return {
    data: result.events,
    metadata: result.metadata ? {
      generatedAt: result.metadata.generatedAt || new Date().toISOString(),
      limit: result.pagination.limit,
      sinceMinutes: 0,
      levelFilter: null,
    } : null,
    isLoading: result.isLoading,
    error: result.error,
    isApiDegraded: result.isDegraded,
    refetch: result.refetch,
  };
}

/**
 * @deprecated Use useAuditEventDetail instead
 */
export function useOpsLogDetail(event: AuditEventRow | null) {
  return useAuditEventDetail(event);
}

// Re-export types for convenience
export type { AuditEventsPagination, AuditEventsMetadata };
