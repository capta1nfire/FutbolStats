/**
 * Incident data hooks using TanStack Query
 */

import { useQuery } from "@tanstack/react-query";
import { Incident, IncidentFilters } from "@/lib/types";
import { getIncidentsMock, getIncidentByIdMock } from "@/lib/mocks";

/**
 * Fetch incidents with optional filters
 */
export function useIncidents(filters?: IncidentFilters) {
  return useQuery<Incident[]>({
    queryKey: ["incidents", filters],
    queryFn: () => getIncidentsMock(filters),
  });
}

/**
 * Fetch a single incident by ID
 */
export function useIncident(id: number | null) {
  return useQuery<Incident | null>({
    queryKey: ["incident", id],
    queryFn: () => (id !== null ? getIncidentByIdMock(id) : Promise.resolve(null)),
    enabled: id !== null,
  });
}
