/**
 * Analytics hooks using TanStack Query
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
