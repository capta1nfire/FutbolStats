/**
 * Data Quality hooks using TanStack Query
 *
 * Supports both mock data and real API with fallback.
 */

import { useQuery } from "@tanstack/react-query";
import {
  DataQualityCheck,
  DataQualityCheckDetail,
  DataQualityAffectedItem,
  DataQualityFilters,
  DataQualityStatus,
  DataQualityCategory,
} from "@/lib/types";
import {
  getDataQualityChecksMock,
  getDataQualityCheckMock,
  getDataQualityAffectedItemsMock,
  getDataQualityStatusCountsMock,
  getDataQualityCategoryCountsMock,
} from "@/lib/mocks";
import {
  parseDataQualityChecks,
  parseDataQualityDetail,
  extractPagination,
  DataQualityPagination,
  calculateStatusCounts,
  calculateCategoryCounts,
} from "@/lib/api/data-quality";

/**
 * Fetch checks from API
 */
async function fetchChecksApi(
  filters?: DataQualityFilters,
  page?: number,
  limit?: number
): Promise<{ checks: DataQualityCheck[]; pagination: DataQualityPagination | null }> {
  const params = new URLSearchParams();

  // Add filter params
  if (filters?.status && filters.status.length > 0) {
    filters.status.forEach((s) => params.append("status", s));
  }
  if (filters?.category && filters.category.length > 0) {
    filters.category.forEach((c) => params.append("category", c));
  }
  if (filters?.search) {
    params.set("q", filters.search);
  }
  if (page !== undefined) {
    params.set("page", String(page));
  }
  if (limit !== undefined) {
    params.set("limit", String(limit));
  }

  const queryString = params.toString();
  const url = `/api/data-quality${queryString ? `?${queryString}` : ""}`;

  const response = await fetch(url);

  if (!response.ok) {
    throw new Error(`API error: ${response.status}`);
  }

  const data = await response.json();
  const checks = parseDataQualityChecks(data);
  const pagination = extractPagination(data);

  return { checks, pagination };
}

/**
 * Fetch check detail from API
 * Note: Backend uses string IDs like "dq_quarantined_odds_24h"
 */
async function fetchCheckDetailApi(id: string): Promise<DataQualityCheckDetail | null> {
  const response = await fetch(`/api/data-quality/${encodeURIComponent(id)}`);

  if (!response.ok) {
    throw new Error(`API error: ${response.status}`);
  }

  const data = await response.json();
  return parseDataQualityDetail(data);
}

/**
 * Fetch all data quality checks with optional filters (mock only)
 */
export function useDataQualityChecks(filters?: DataQualityFilters) {
  return useQuery<DataQualityCheck[]>({
    queryKey: ["data-quality", "checks", filters],
    queryFn: () => getDataQualityChecksMock(filters),
  });
}

/**
 * Fetch data quality checks from real API with mock fallback
 *
 * Returns:
 * - data: checks from API (or mock if degraded)
 * - isApiDegraded: true if using mock fallback
 * - pagination: server pagination info (null if using mock)
 */
export function useDataQualityChecksApi(
  filters?: DataQualityFilters,
  page?: number,
  limit?: number
) {
  const apiQuery = useQuery<
    { checks: DataQualityCheck[]; pagination: DataQualityPagination | null },
    Error
  >({
    queryKey: ["data-quality", "checks", "api", filters, page, limit],
    queryFn: () => fetchChecksApi(filters, page, limit),
    staleTime: 30 * 1000, // 30 seconds
    retry: 1,
  });

  // Fallback to mock on error
  const mockQuery = useQuery<DataQualityCheck[]>({
    queryKey: ["data-quality", "checks", "mock", filters],
    queryFn: () => getDataQualityChecksMock(filters),
    enabled: apiQuery.isError,
  });

  // Determine which data to use
  const isApiDegraded = apiQuery.isError;
  const data = isApiDegraded
    ? (mockQuery.data ?? [])
    : (apiQuery.data?.checks ?? []);
  const pagination = isApiDegraded ? null : (apiQuery.data?.pagination ?? null);
  const isLoading = apiQuery.isLoading || (isApiDegraded && mockQuery.isLoading);
  const error = isApiDegraded ? null : apiQuery.error; // Suppress error if mock fallback works

  return {
    data,
    pagination,
    isLoading,
    error,
    isApiDegraded,
    refetch: apiQuery.refetch,
  };
}

/**
 * Fetch a single data quality check with full details (mock only)
 * Note: Uses string ID for compatibility with backend
 */
export function useDataQualityCheck(id: string | null) {
  return useQuery<DataQualityCheckDetail | null>({
    queryKey: ["data-quality", "check", id],
    queryFn: () => (id ? getDataQualityCheckMock(id) : Promise.resolve(null)),
    enabled: id !== null,
  });
}

/**
 * Fetch a single data quality check from real API with mock fallback
 * Note: Uses string ID (e.g., "dq_quarantined_odds_24h")
 */
export function useDataQualityCheckApi(id: string | null) {
  const apiQuery = useQuery<DataQualityCheckDetail | null, Error>({
    queryKey: ["data-quality", "check", "api", id],
    queryFn: () => (id ? fetchCheckDetailApi(id) : Promise.resolve(null)),
    enabled: id !== null,
    staleTime: 30 * 1000,
    retry: 1,
  });

  // Fallback to mock on error
  const mockQuery = useQuery<DataQualityCheckDetail | null>({
    queryKey: ["data-quality", "check", "mock", id],
    queryFn: () => (id ? getDataQualityCheckMock(id) : Promise.resolve(null)),
    enabled: id !== null && apiQuery.isError,
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

/**
 * Fetch affected items for a specific check
 * Note: Uses string ID for compatibility with backend
 */
export function useDataQualityAffectedItems(checkId: string | null) {
  return useQuery<DataQualityAffectedItem[]>({
    queryKey: ["data-quality", "affected", checkId],
    queryFn: () =>
      checkId
        ? getDataQualityAffectedItemsMock(checkId)
        : Promise.resolve([]),
    enabled: checkId !== null,
  });
}

/**
 * Get status counts (synchronous, for filter badges)
 */
export function useDataQualityStatusCounts(): Record<DataQualityStatus, number> {
  return getDataQualityStatusCountsMock();
}

/**
 * Get category counts (synchronous, for filter badges)
 */
export function useDataQualityCategoryCounts(): Record<DataQualityCategory, number> {
  return getDataQualityCategoryCountsMock();
}

/**
 * Calculate status counts from checks data
 */
export function useStatusCountsFromData(
  checks: DataQualityCheck[]
): Record<DataQualityStatus, number> {
  return calculateStatusCounts(checks);
}

/**
 * Calculate category counts from checks data
 */
export function useCategoryCountsFromData(
  checks: DataQualityCheck[]
): Record<DataQualityCategory, number> {
  return calculateCategoryCounts(checks);
}

// Re-export pagination type for convenience
export type { DataQualityPagination };
