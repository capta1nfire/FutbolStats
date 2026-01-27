"use client";

import { useQuery } from "@tanstack/react-query";
import type { LeagueDetail, LeagueDetailResponse } from "@/lib/types";

export interface UseFootballLeagueResult {
  data: LeagueDetail | null;
  generatedAt: string | null;
  cached: boolean;
  cacheAgeSeconds: number | null;
  isLoading: boolean;
  error: Error | null;
  refetch: () => void;
}

async function fetchFootballLeague(id: number): Promise<LeagueDetailResponse | null> {
  const response = await fetch(`/api/football/league/${id}`, {
    method: "GET",
    headers: { Accept: "application/json" },
  });

  if (!response.ok) {
    return null;
  }

  return response.json();
}

export function useFootballLeague(leagueId: number | null): UseFootballLeagueResult {
  const { data, isLoading, error, refetch } = useQuery({
    queryKey: ["football-league", leagueId],
    queryFn: () => fetchFootballLeague(leagueId!),
    enabled: leagueId !== null,
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
