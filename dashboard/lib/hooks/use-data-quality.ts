/**
 * Data Quality hooks using TanStack Query
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

/**
 * Fetch all data quality checks with optional filters
 */
export function useDataQualityChecks(filters?: DataQualityFilters) {
  return useQuery<DataQualityCheck[]>({
    queryKey: ["data-quality", "checks", filters],
    queryFn: () => getDataQualityChecksMock(filters),
  });
}

/**
 * Fetch a single data quality check with full details
 */
export function useDataQualityCheck(id: number | null) {
  return useQuery<DataQualityCheckDetail | null>({
    queryKey: ["data-quality", "check", id],
    queryFn: () => (id ? getDataQualityCheckMock(id) : Promise.resolve(null)),
    enabled: id !== null,
  });
}

/**
 * Fetch affected items for a specific check
 */
export function useDataQualityAffectedItems(checkId: number | null) {
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
