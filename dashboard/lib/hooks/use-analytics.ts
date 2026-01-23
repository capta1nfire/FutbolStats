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
  OpsHistoryRollup,
  PredictionsPerformance,
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
 */
export function useAnalyticsReport(id: number | null) {
  return useQuery<AnalyticsReportDetail | null>({
    queryKey: ["analytics", "report", id],
    queryFn: () => (id ? getAnalyticsReportMock(id) : Promise.resolve(null)),
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

// Re-export types for convenience
export type { OpsHistoryRollup, PredictionsPerformance };
