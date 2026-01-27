"use client";

import { useQuery } from "@tanstack/react-query";
import type { FootballOverview, FootballOverviewResponse } from "@/lib/types";

export interface UseFootballOverviewResult {
  data: FootballOverview | null;
  generatedAt: string | null;
  cached: boolean;
  cacheAgeSeconds: number | null;
  isLoading: boolean;
  error: Error | null;
  refetch: () => void;
}

async function fetchFootballOverview(): Promise<FootballOverviewResponse | null> {
  const response = await fetch("/api/football/overview", {
    method: "GET",
    headers: { Accept: "application/json" },
  });

  if (!response.ok) {
    return null;
  }

  return response.json();
}

export function useFootballOverview(): UseFootballOverviewResult {
  const { data, isLoading, error, refetch } = useQuery({
    queryKey: ["football-overview"],
    queryFn: fetchFootballOverview,
    retry: 1,
    staleTime: 30_000,
    refetchInterval: 60_000,
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
