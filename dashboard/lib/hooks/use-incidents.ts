"use client";

/**
 * Incident data hooks using TanStack Query
 */

import { useQuery } from "@tanstack/react-query";
import {
  Incident,
  IncidentFilters,
  IncidentStatus,
  IncidentSeverity,
} from "@/lib/types";
import { getIncidentsMock, getIncidentByIdMock } from "@/lib/mocks";
import {
  parseIncidents,
  extractPagination,
  extractMetadata,
  IncidentsPagination,
} from "@/lib/api/incidents";

/**
 * Response from useIncidentsApi hook
 */
export interface UseIncidentsApiResult {
  /** Parsed incidents, null if unavailable */
  incidents: Incident[] | null;
  /** Pagination info */
  pagination: IncidentsPagination;
  /** True if data fetch failed or parsing failed */
  isDegraded: boolean;
  /** Request ID for debugging */
  requestId?: string;
  /** When backend generated this data */
  generatedAt: string | null;
  /** Whether data is from backend cache */
  cached: boolean;
  /** Age of backend cache in seconds */
  cacheAgeSeconds: number;
  /** Loading state */
  isLoading: boolean;
  /** Error object if fetch failed */
  error: Error | null;
  /** Refetch function */
  refetch: () => void;
}

/**
 * Query params for the API
 */
interface IncidentsQueryParams {
  status?: string[];
  severity?: string[];
  type?: string;
  q?: string;
  page?: number;
  limit?: number;
}

/**
 * Internal response type
 */
interface IncidentsData {
  incidents: Incident[] | null;
  pagination: IncidentsPagination;
  requestId?: string;
  generatedAt: string | null;
  cached: boolean;
  cacheAgeSeconds: number;
}

/**
 * Build query string from params
 */
function buildQueryString(params: IncidentsQueryParams): string {
  const searchParams = new URLSearchParams();

  // Multi-select params: status and severity
  if (params.status && params.status.length > 0) {
    params.status.forEach((s) => searchParams.append("status", s));
  }
  if (params.severity && params.severity.length > 0) {
    params.severity.forEach((s) => searchParams.append("severity", s));
  }
  if (params.type) searchParams.set("type", params.type);
  if (params.q) searchParams.set("q", params.q);
  if (params.page) searchParams.set("page", params.page.toString());
  if (params.limit) searchParams.set("limit", params.limit.toString());

  const qs = searchParams.toString();
  return qs ? `?${qs}` : "";
}

/**
 * Fetch incidents from proxy endpoint
 */
async function fetchIncidents(params: IncidentsQueryParams): Promise<IncidentsData> {
  const queryString = buildQueryString(params);
  const response = await fetch(`/api/incidents${queryString}`, {
    method: "GET",
    headers: {
      Accept: "application/json",
    },
  });

  const requestId = response.headers.get("x-request-id") || undefined;

  if (!response.ok) {
    return {
      incidents: null,
      pagination: { total: 0, page: 1, limit: 50, pages: 1 },
      requestId,
      generatedAt: null,
      cached: false,
      cacheAgeSeconds: 0,
    };
  }

  const data = await response.json();
  const incidents = parseIncidents(data);
  const pagination = extractPagination(data);
  const metadata = extractMetadata(data);

  return {
    incidents,
    pagination,
    requestId,
    generatedAt: metadata.generatedAt,
    cached: metadata.cached,
    cacheAgeSeconds: metadata.cacheAgeSeconds,
  };
}

/**
 * Hook to fetch incidents from backend via /api/incidents proxy
 *
 * Features:
 * - Enterprise-safe: uses same-origin proxy, no secrets exposed
 * - Graceful degradation: returns isDegraded=true if fetch/parse fails
 * - Pagination support
 * - Status/severity/type filtering
 *
 * Usage:
 * ```tsx
 * const { incidents, isDegraded, pagination, isLoading } = useIncidentsApi({
 *   status: ["active"],
 *   severity: ["critical", "warning"],
 *   page: 1,
 *   limit: 50,
 * });
 *
 * if (isLoading) return <Loader />;
 *
 * const displayIncidents = incidents ?? mockIncidents;
 * ```
 */
export function useIncidentsApi(options?: {
  status?: IncidentStatus[];
  severity?: IncidentSeverity[];
  type?: string;
  q?: string;
  page?: number;
  limit?: number;
  enabled?: boolean;
}): UseIncidentsApiResult {
  const {
    status = [],
    severity = [],
    type,
    q,
    page = 1,
    limit = 50,
    enabled = true,
  } = options || {};

  const queryParams: IncidentsQueryParams = {
    status: status.length > 0 ? status : undefined,
    severity: severity.length > 0 ? severity : undefined,
    type,
    q: q || undefined,
    page,
    limit,
  };

  // Cache timing aligned with backend TTL (30s)
  const staleTime = 30_000;
  const refetchInterval = 30_000;

  const { data, isLoading, error, refetch } = useQuery({
    queryKey: ["incidents-api", queryParams],
    queryFn: () => fetchIncidents(queryParams),
    retry: 1,
    staleTime,
    refetchInterval,
    refetchOnWindowFocus: false,
    throwOnError: false,
    enabled,
  });

  const incidents = data?.incidents ?? null;
  const pagination = data?.pagination ?? { total: 0, page: 1, limit: 50, pages: 1 };
  const requestId = data?.requestId;
  const generatedAt = data?.generatedAt ?? null;
  const cached = data?.cached ?? false;
  const cacheAgeSeconds = data?.cacheAgeSeconds ?? 0;
  const isDegraded = !!error || incidents === null;

  return {
    incidents,
    pagination,
    isDegraded,
    requestId,
    generatedAt,
    cached,
    cacheAgeSeconds,
    isLoading,
    error: error as Error | null,
    refetch: () => refetch(),
  };
}

/**
 * Hook to fetch incidents with optional filters (mock fallback)
 * @deprecated Use useIncidentsApi for real data with mock fallback
 */
export function useIncidents(filters?: IncidentFilters) {
  return useQuery<Incident[]>({
    queryKey: ["incidents", filters],
    queryFn: () => getIncidentsMock(filters),
  });
}

/**
 * Synchronous mock getter for fallback (no delay)
 */
export function getIncidentsMockSync(filters?: IncidentFilters): Incident[] {
  // Import the mock data synchronously for fallback
  // This mirrors the pattern from matches
  const { getIncidentsMock } = require("@/lib/mocks");

  // Since we need sync data, we'll use the underlying data directly
  // The mock function is async, so we need to access the raw data
  const normalDataset: Incident[] = [];

  // Generate deterministic incidents (same as mocks but sync)
  const BASE_TIMESTAMP = new Date("2026-01-20T10:00:00Z").getTime();
  const incidentTypes = [
    "missing_prediction",
    "job_failure",
    "api_error",
    "data_inconsistency",
    "high_latency",
    "other",
  ] as const;
  const severities = ["critical", "warning", "info"] as const;
  const statuses = ["active", "acknowledged", "resolved"] as const;

  for (let i = 0; i < 25; i++) {
    const hoursAgo = i * 2 + (i % 5);
    const createdAt = new Date(BASE_TIMESTAMP - hoursAgo * 3600000).toISOString();

    normalDataset.push({
      id: 10000 - i,
      type: incidentTypes[i % incidentTypes.length],
      severity: severities[i % severities.length],
      status: statuses[i % statuses.length],
      createdAt,
      title: `Mock incident ${i + 1}`,
      description: `This is a mock incident for fallback display.`,
    });
  }

  let data = [...normalDataset];

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
          i.description?.toLowerCase().includes(search)
      );
    }
  }

  return data;
}

/**
 * Fetch a single incident by ID (mock only, used as fallback)
 */
export function useIncident(id: number | null) {
  return useQuery<Incident | null>({
    queryKey: ["incident", id],
    queryFn: () => (id !== null ? getIncidentByIdMock(id) : Promise.resolve(null)),
    enabled: id !== null,
  });
}
