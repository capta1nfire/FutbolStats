"use client";

import { useQuery } from "@tanstack/react-query";
import { MatchFilters, MatchSummary } from "@/lib/types";
import { getMatchesMock, getMatchByIdMock } from "@/lib/mocks";

/**
 * Hook to fetch matches with optional filters
 */
export function useMatches(filters?: MatchFilters) {
  return useQuery<MatchSummary[], Error>({
    queryKey: ["matches", filters],
    queryFn: () => getMatchesMock(filters),
  });
}

/**
 * Hook to fetch a single match by ID
 */
export function useMatch(id: number | null) {
  return useQuery<MatchSummary | null, Error>({
    queryKey: ["match", id],
    queryFn: () => (id ? getMatchByIdMock(id) : null),
    enabled: id !== null,
  });
}
