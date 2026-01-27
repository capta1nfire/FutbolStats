"use client";

import { useQuery } from "@tanstack/react-query";
import type { AdminLeagueDetailCore, AdminLeagueDetailResponse } from "@/lib/types";

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
