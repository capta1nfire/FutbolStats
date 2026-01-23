/**
 * Overview data hooks using TanStack Query
 */

"use client";

import { useQuery } from "@tanstack/react-query";
import {
  OverviewData,
  HealthSummary,
  ActiveIncident,
} from "@/lib/types";
import {
  getOverviewDataMock,
  getHealthSummaryMock,
  getActiveIncidentsMock,
} from "@/lib/mocks";
import { parseIncidents, extractMetadata } from "@/lib/api/incidents";

/**
 * Fetch all overview data in one query
 */
export function useOverviewData() {
  return useQuery<OverviewData>({
    queryKey: ["overview"],
    queryFn: getOverviewDataMock,
  });
}

/**
 * Fetch health summary only
 */
export function useHealthSummary() {
  return useQuery<HealthSummary>({
    queryKey: ["overview", "health"],
    queryFn: getHealthSummaryMock,
  });
}

/**
 * Fetch active incidents only (mock)
 */
export function useActiveIncidents() {
  return useQuery<ActiveIncident[]>({
    queryKey: ["overview", "incidents"],
    queryFn: getActiveIncidentsMock,
  });
}

/**
 * Response from useActiveIncidentsApi hook
 */
export interface UseActiveIncidentsApiResult {
  /** Active incidents, null if unavailable */
  incidents: ActiveIncident[] | null;
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
 * Internal response type
 */
interface ActiveIncidentsData {
  incidents: ActiveIncident[] | null;
  requestId?: string;
  generatedAt: string | null;
  cached: boolean;
  cacheAgeSeconds: number;
}

/**
 * Map full Incident to minimal ActiveIncident for overview
 */
function toActiveIncident(inc: {
  id: number;
  title: string;
  severity: string;
  createdAt: string;
  type: string;
}): ActiveIncident {
  // Normalize severity to valid ActiveIncident severity
  const normalizedSeverity =
    inc.severity === "critical" || inc.severity === "warning"
      ? inc.severity
      : "info";

  return {
    id: inc.id,
    title: inc.title,
    severity: normalizedSeverity as "critical" | "warning" | "info",
    createdAt: inc.createdAt,
    type: inc.type,
  };
}

/**
 * Fetch active incidents from proxy endpoint
 */
async function fetchActiveIncidentsApi(): Promise<ActiveIncidentsData> {
  const response = await fetch("/api/incidents?status=active&limit=5", {
    method: "GET",
    headers: {
      Accept: "application/json",
    },
  });

  // Extract request ID from header
  const requestId = response.headers.get("x-request-id") || undefined;

  if (!response.ok) {
    return {
      incidents: null,
      requestId,
      generatedAt: null,
      cached: false,
      cacheAgeSeconds: 0,
    };
  }

  const data = await response.json();
  const fullIncidents = parseIncidents(data);
  const metadata = extractMetadata(data);

  // Map to ActiveIncident (minimal shape for overview)
  const incidents = fullIncidents
    ? fullIncidents.map(toActiveIncident)
    : null;

  return {
    incidents,
    requestId,
    generatedAt: metadata.generatedAt,
    cached: metadata.cached,
    cacheAgeSeconds: metadata.cacheAgeSeconds,
  };
}

/**
 * Hook to fetch active incidents from backend via /api/incidents proxy
 *
 * Features:
 * - Enterprise-safe: uses same-origin proxy, no secrets exposed
 * - Graceful degradation: returns isDegraded=true if fetch/parse fails
 * - Auto-refetch: refreshes every 60s
 * - Single retry on failure
 *
 * Usage:
 * ```tsx
 * const { incidents, isDegraded, isLoading } = useActiveIncidentsApi();
 *
 * if (isLoading) return <Loader />;
 *
 * const displayIncidents = incidents ?? mockActiveIncidents;
 * ```
 */
export function useActiveIncidentsApi(): UseActiveIncidentsApiResult {
  const { data, isLoading, error, refetch } = useQuery({
    queryKey: ["overview", "incidents", "api"],
    queryFn: fetchActiveIncidentsApi,
    retry: 1,
    staleTime: 30_000, // Consider data fresh for 30s
    refetchInterval: 60_000, // Refetch every 60s
    refetchOnWindowFocus: false,
    throwOnError: false,
  });

  const incidents = data?.incidents ?? null;
  const requestId = data?.requestId;
  const generatedAt = data?.generatedAt ?? null;
  const cached = data?.cached ?? false;
  const cacheAgeSeconds = data?.cacheAgeSeconds ?? 0;
  const isDegraded = !!error || incidents === null;

  return {
    incidents,
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
