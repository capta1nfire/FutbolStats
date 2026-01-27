"use client";

import { useQuery } from "@tanstack/react-query";
import type { TeamDetail, TeamDetailResponse } from "@/lib/types";

export interface UseFootballTeamResult {
  data: TeamDetail | null;
  generatedAt: string | null;
  cached: boolean;
  cacheAgeSeconds: number | null;
  isLoading: boolean;
  error: Error | null;
  refetch: () => void;
}

async function fetchFootballTeam(id: number): Promise<TeamDetailResponse | null> {
  const response = await fetch(`/api/football/team/${id}`, {
    method: "GET",
    headers: { Accept: "application/json" },
  });

  if (!response.ok) {
    return null;
  }

  return response.json();
}

export function useFootballTeam(teamId: number | null): UseFootballTeamResult {
  const { data, isLoading, error, refetch } = useQuery({
    queryKey: ["football-team", teamId],
    queryFn: () => fetchFootballTeam(teamId!),
    enabled: teamId !== null,
    retry: 1,
    staleTime: 30_000,
    refetchOnWindowFocus: false,
    throwOnError: false,
  });

  return {
    data: data?.data ?? null,
    generatedAt: data?.generated_at ?? null,
    cached: data?.cached ?? false,
    cacheAgeSeconds: data?.cache_age_seconds ?? null,
    isLoading,
    error: error as Error | null,
    refetch: () => refetch(),
  };
}
