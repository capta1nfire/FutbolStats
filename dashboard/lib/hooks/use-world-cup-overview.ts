"use client";

import { useQuery } from "@tanstack/react-query";
import type { WorldCupOverview, WorldCupOverviewResponse } from "@/lib/types";

export interface UseWorldCupOverviewResult {
  data: WorldCupOverview | null;
  generatedAt: string | null;
  cached: boolean;
  cacheAgeSeconds: number | null;
  isLoading: boolean;
  error: Error | null;
  refetch: () => void;
}

async function fetchWorldCupOverview(): Promise<WorldCupOverviewResponse | null> {
  const response = await fetch("/api/football/world-cup-2026/overview", {
    method: "GET",
    headers: { Accept: "application/json" },
  });

  if (!response.ok) {
    return null;
  }

  return response.json();
}

export function useWorldCupOverview(): UseWorldCupOverviewResult {
  const { data, isLoading, error, refetch } = useQuery({
    queryKey: ["world-cup-2026-overview"],
    queryFn: fetchWorldCupOverview,
    retry: 1,
    staleTime: 60_000,
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
