"use client";

import { useQuery } from "@tanstack/react-query";
import type { AdminLeagueGroupsList, AdminLeagueGroupsListResponse } from "@/lib/types";

async function fetchAdminLeagueGroups(): Promise<AdminLeagueGroupsListResponse | null> {
  const response = await fetch("/api/admin/league-groups", {
    method: "GET",
    headers: { Accept: "application/json" },
  });
  if (!response.ok) return null;
  return response.json();
}

export function useAdminLeagueGroups() {
  const { data, isLoading, error, refetch } = useQuery({
    queryKey: ["admin-league-groups"],
    queryFn: fetchAdminLeagueGroups,
    retry: 1,
    staleTime: 60_000,
    refetchOnWindowFocus: false,
    throwOnError: false,
  });

  return {
    data: (data?.data ?? null) as AdminLeagueGroupsList | null,
    isLoading,
    error: error as Error | null,
    refetch: () => refetch(),
  };
}
