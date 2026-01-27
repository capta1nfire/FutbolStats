"use client";

import { useQuery } from "@tanstack/react-query";
import type { AdminLeagueGroupDetail, AdminLeagueGroupDetailResponse } from "@/lib/types";

async function fetchAdminLeagueGroup(id: number): Promise<AdminLeagueGroupDetailResponse | null> {
  const response = await fetch(`/api/admin/league-group/${id}`, {
    method: "GET",
    headers: { Accept: "application/json" },
  });
  if (!response.ok) return null;
  return response.json();
}

export function useAdminLeagueGroup(groupId: number | null) {
  const { data, isLoading, error, refetch } = useQuery({
    queryKey: ["admin-league-group", groupId],
    queryFn: () => fetchAdminLeagueGroup(groupId!),
    enabled: groupId !== null,
    retry: 1,
    staleTime: 30_000,
    refetchOnWindowFocus: false,
    throwOnError: false,
  });

  return {
    data: (data?.data ?? null) as AdminLeagueGroupDetail | null,
    isLoading,
    error: error as Error | null,
    refetch: () => refetch(),
  };
}
