/**
 * Analytics hooks using TanStack Query
 *
 * Supports both mock data and real API with fallback.
 */

import { useQuery } from "@tanstack/react-query";
import {
  AnalyticsReportRow,
  AnalyticsReportDetail,
  AnalyticsFilters,
  AnalyticsReportType,
  AnalyticsReportStatus,
} from "@/lib/types";
import {
  getAnalyticsReportsMock,
  getAnalyticsReportMock,
  getAnalyticsTypeCountsMock,
  getAnalyticsStatusCountsMock,
} from "@/lib/mocks";
import {
  parseOpsHistory,
  parsePredictionsPerformance,
  parseAnalyticsReportsResponse,
  OpsHistoryRollup,
  PredictionsPerformance,
  AnalyticsReportsPagination,
  AnalyticsReportsMetadata,
} from "@/lib/api/analytics";

/**
 * Fetch all analytics reports with optional filters
 */
export function useAnalyticsReports(filters?: AnalyticsFilters) {
  return useQuery<AnalyticsReportRow[]>({
    queryKey: ["analytics", "reports", filters],
    queryFn: () => getAnalyticsReportsMock(filters),
  });
}

/**
 * Fetch a single analytics report with full details
 * Note: id is number | string to support both mock (number) and real backend (string)
 */
export function useAnalyticsReport(id: number | string | null) {
  // For now, this still uses mock - convert string ID to number for mock compatibility
  const numericId = typeof id === "string" ? parseInt(id, 10) || null : id;
  return useQuery<AnalyticsReportDetail | null>({
    queryKey: ["analytics", "report", id],
    queryFn: () => (numericId ? getAnalyticsReportMock(numericId) : Promise.resolve(null)),
    enabled: id !== null,
  });
}

/**
 * Get type counts (synchronous, for filter badges)
 */
export function useAnalyticsTypeCounts(): Record<AnalyticsReportType, number> {
  return getAnalyticsTypeCountsMock();
}

/**
 * Get status counts (synchronous, for filter badges)
 */
export function useAnalyticsStatusCounts(): Record<AnalyticsReportStatus, number> {
  return getAnalyticsStatusCountsMock();
}

// ============================================================================
// Real API Hooks (with mock fallback)
// ============================================================================

/**
 * Fetch ops history from API
 */
async function fetchOpsHistoryApi(days: number): Promise<OpsHistoryRollup[]> {
  const response = await fetch(`/api/analytics/history?days=${days}`);

  if (!response.ok) {
    throw new Error(`API error: ${response.status}`);
  }

  const data = await response.json();
  return parseOpsHistory(data);
}

/**
 * Fetch predictions performance from API
 */
async function fetchPredictionsPerformanceApi(
  windowDays: number
): Promise<PredictionsPerformance | null> {
  const response = await fetch(
    `/api/analytics/performance?window_days=${windowDays}`
  );

  if (!response.ok) {
    throw new Error(`API error: ${response.status}`);
  }

  const data = await response.json();
  return parsePredictionsPerformance(data);
}

/**
 * Mock fallback for ops history
 * Returns empty array when API fails
 */
function getOpsHistoryMock(): OpsHistoryRollup[] {
  return [];
}

/**
 * Mock fallback for predictions performance
 * Returns null when API fails
 */
function getPredictionsPerformanceMock(): PredictionsPerformance | null {
  return null;
}

/**
 * Fetch ops history from real API with mock fallback
 *
 * @param days - Number of days of history to fetch (default: 30)
 */
export function useOpsHistoryApi(days: number = 30) {
  const apiQuery = useQuery<OpsHistoryRollup[], Error>({
    queryKey: ["analytics", "history", "api", days],
    queryFn: () => fetchOpsHistoryApi(days),
    staleTime: 60 * 1000, // 1 minute
    retry: 1,
  });

  // Fallback to mock on error
  const mockQuery = useQuery<OpsHistoryRollup[]>({
    queryKey: ["analytics", "history", "mock"],
    queryFn: () => Promise.resolve(getOpsHistoryMock()),
    enabled: apiQuery.isError,
  });

  const isApiDegraded = apiQuery.isError;
  const data = isApiDegraded
    ? (mockQuery.data ?? [])
    : (apiQuery.data ?? []);
  const isLoading = apiQuery.isLoading || (isApiDegraded && mockQuery.isLoading);

  return {
    data,
    isLoading,
    error: isApiDegraded ? null : apiQuery.error,
    isApiDegraded,
    refetch: apiQuery.refetch,
  };
}

/**
 * Fetch predictions performance from real API with mock fallback
 *
 * @param windowDays - Number of days for performance window (default: 7)
 */
export function usePredictionsPerformanceApi(windowDays: number = 7) {
  const apiQuery = useQuery<PredictionsPerformance | null, Error>({
    queryKey: ["analytics", "performance", "api", windowDays],
    queryFn: () => fetchPredictionsPerformanceApi(windowDays),
    staleTime: 60 * 1000, // 1 minute
    retry: 1,
  });

  // Fallback to mock on error
  const mockQuery = useQuery<PredictionsPerformance | null>({
    queryKey: ["analytics", "performance", "mock"],
    queryFn: () => Promise.resolve(getPredictionsPerformanceMock()),
    enabled: apiQuery.isError,
  });

  const isApiDegraded = apiQuery.isError;
  const data = isApiDegraded ? mockQuery.data : apiQuery.data;
  const isLoading = apiQuery.isLoading || (isApiDegraded && mockQuery.isLoading);

  return {
    data: data ?? null,
    isLoading,
    error: isApiDegraded ? null : apiQuery.error,
    isApiDegraded,
    refetch: apiQuery.refetch,
  };
}

// ============================================================================
// Analytics Reports API Hook
// ============================================================================

/**
 * Query params for reports API
 */
interface AnalyticsReportsQueryParams {
  type?: AnalyticsReportType;
  q?: string;
  page?: number;
  limit?: number;
}

/**
 * Internal response from fetch
 */
interface AnalyticsReportsData {
  reports: AnalyticsReportRow[];
  pagination: AnalyticsReportsPagination;
  metadata: AnalyticsReportsMetadata;
}

/**
 * Result from useAnalyticsReportsApi hook
 */
export interface UseAnalyticsReportsApiResult {
  reports: AnalyticsReportRow[];
  pagination: AnalyticsReportsPagination;
  metadata: AnalyticsReportsMetadata;
  isLoading: boolean;
  error: Error | null;
  isDegraded: boolean;
  refetch: () => void;
}

/**
 * Build query string from params
 */
function buildReportsQueryString(params: AnalyticsReportsQueryParams): string {
  const searchParams = new URLSearchParams();

  if (params.type) searchParams.set("type", params.type);
  if (params.q) searchParams.set("q", params.q);
  if (params.page) searchParams.set("page", params.page.toString());
  if (params.limit) searchParams.set("limit", params.limit.toString());

  const qs = searchParams.toString();
  return qs ? `?${qs}` : "";
}

/**
 * Fetch reports from proxy endpoint
 */
async function fetchAnalyticsReports(params: AnalyticsReportsQueryParams): Promise<AnalyticsReportsData> {
  const queryString = buildReportsQueryString(params);
  const response = await fetch(`/api/analytics/reports${queryString}`, {
    method: "GET",
    headers: {
      Accept: "application/json",
    },
  });

  if (!response.ok) {
    throw new Error(`API error: ${response.status}`);
  }

  const data = await response.json();
  const parsed = parseAnalyticsReportsResponse(data);

  if (!parsed) {
    throw new Error("Failed to parse analytics reports response");
  }

  return {
    reports: parsed.reports,
    pagination: parsed.pagination,
    metadata: parsed.metadata,
  };
}

/**
 * Fetch analytics reports from real API with mock fallback
 *
 * Features:
 * - Server-side filtering (type, q)
 * - Real pagination (page, limit, total, pages)
 * - isDegraded indicator for fallback state
 * - Automatic mock fallback on API error
 *
 * Usage:
 * ```tsx
 * const {
 *   reports,
 *   pagination,
 *   isDegraded,
 *   isLoading,
 *   refetch,
 * } = useAnalyticsReportsApi({
 *   type: "model_performance",
 *   q: "accuracy",
 *   page: 1,
 *   limit: 50,
 * });
 * ```
 */
export function useAnalyticsReportsApi(options?: {
  type?: AnalyticsReportType;
  q?: string;
  page?: number;
  limit?: number;
  enabled?: boolean;
}): UseAnalyticsReportsApiResult {
  const {
    type,
    q,
    page = 1,
    limit = 50,
    enabled = true,
  } = options || {};

  const queryParams: AnalyticsReportsQueryParams = {
    type: type || undefined,
    q: q || undefined,
    page,
    limit,
  };

  // Fetch from API
  const apiQuery = useQuery<AnalyticsReportsData, Error>({
    queryKey: ["analytics", "reports", "api", queryParams],
    queryFn: () => fetchAnalyticsReports(queryParams),
    staleTime: 30_000, // 30 seconds
    retry: 1,
    enabled,
  });

  // Build filters for mock fallback (client-side filtering)
  const mockFilters: AnalyticsFilters = {
    type: type ? [type] : undefined,
    search: q,
  };

  // Fallback to mock on error
  const mockQuery = useQuery<AnalyticsReportRow[], Error>({
    queryKey: ["analytics", "reports", "mock", mockFilters],
    queryFn: () => getAnalyticsReportsMock(mockFilters),
    enabled: apiQuery.isError && enabled,
  });

  // Determine which data to use
  const isDegraded = apiQuery.isError;
  const reports = isDegraded
    ? (mockQuery.data ?? [])
    : (apiQuery.data?.reports ?? []);
  const pagination: AnalyticsReportsPagination = isDegraded
    ? { total: mockQuery.data?.length ?? 0, page: 1, limit: 50, pages: 1 }
    : (apiQuery.data?.pagination ?? { total: 0, page: 1, limit: 50, pages: 1 });
  const metadata: AnalyticsReportsMetadata = isDegraded
    ? { generatedAt: null, cached: false, cacheAgeSeconds: 0 }
    : (apiQuery.data?.metadata ?? { generatedAt: null, cached: false, cacheAgeSeconds: 0 });
  const isLoading = apiQuery.isLoading || (isDegraded && mockQuery.isLoading);
  const error = isDegraded ? null : apiQuery.error; // Suppress if mock fallback works

  return {
    reports,
    pagination,
    metadata,
    isLoading,
    error,
    isDegraded,
    refetch: () => apiQuery.refetch(),
  };
}

// Re-export types for convenience
export type { OpsHistoryRollup, PredictionsPerformance, AnalyticsReportsPagination, AnalyticsReportsMetadata };
