/**
 * Overview data hooks using TanStack Query
 */

import { useQuery } from "@tanstack/react-query";
import {
  OverviewData,
  HealthSummary,
  UpcomingMatch,
  ActiveIncident,
} from "@/lib/types";
import {
  getOverviewDataMock,
  getHealthSummaryMock,
  getUpcomingMatchesMock,
  getActiveIncidentsMock,
} from "@/lib/mocks";

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
 * Fetch upcoming matches only
 */
export function useUpcomingMatches() {
  return useQuery<UpcomingMatch[]>({
    queryKey: ["overview", "upcoming"],
    queryFn: getUpcomingMatchesMock,
  });
}

/**
 * Fetch active incidents only
 */
export function useActiveIncidents() {
  return useQuery<ActiveIncident[]>({
    queryKey: ["overview", "incidents"],
    queryFn: getActiveIncidentsMock,
  });
}
