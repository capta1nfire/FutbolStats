"use client";

import { useQuery } from "@tanstack/react-query";
import type { AdminLeagueDetailCore, AdminLeagueDetailData, AdminLeagueDetailResponse } from "@/lib/types";

async function fetchAdminLeague(id: number): Promise<AdminLeagueDetailResponse | null> {
  const response = await fetch(`/api/admin/league/${id}`, {
    method: "GET",
    headers: { Accept: "application/json" },
  });
  if (!response.ok) return null;
  return response.json();
}

export function useAdminLeague(leagueId: number | null) {
  const { data, isLoading, error, refetch } = useQuery({
    queryKey: ["admin-league", leagueId],
    queryFn: () => fetchAdminLeague(leagueId!),
    enabled: leagueId !== null,
    retry: 1,
    staleTime: 30_000,
    refetchOnWindowFocus: false,
    throwOnError: false,
  });

  return {
    data: (data?.data?.league ?? null) as AdminLeagueDetailCore | null,
    isLoading,
    error: error as Error | null,
    refetch: () => refetch(),
  };
}

/**
 * Returns the full league detail payload (stats_by_season, teams, titan_coverage, recent_matches).
 * Shares the same query key as useAdminLeague — no extra fetch.
 */
export function useAdminLeagueDetail(leagueId: number | null) {
  const { data, isLoading, error, refetch } = useQuery({
    queryKey: ["admin-league", leagueId],
    queryFn: () => fetchAdminLeague(leagueId!),
    enabled: leagueId !== null,
    retry: 1,
    staleTime: 30_000,
    refetchOnWindowFocus: false,
    throwOnError: false,
  });

  return {
    data: (data?.data ?? null) as AdminLeagueDetailData | null,
    isLoading,
    error: error as Error | null,
    refetch: () => refetch(),
  };
}
