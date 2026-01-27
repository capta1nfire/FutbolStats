"use client";

import { useQuery } from "@tanstack/react-query";
import type { WorldCupGroups, WorldCupGroupsResponse } from "@/lib/types";

export interface UseWorldCupGroupsResult {
  data: WorldCupGroups | null;
  generatedAt: string | null;
  cached: boolean;
  cacheAgeSeconds: number | null;
  isLoading: boolean;
  error: Error | null;
  refetch: () => void;
}

async function fetchWorldCupGroups(): Promise<WorldCupGroupsResponse | null> {
  const response = await fetch("/api/football/world-cup-2026/groups", {
    method: "GET",
    headers: { Accept: "application/json" },
  });

  if (!response.ok) {
    return null;
  }

  return response.json();
}

export function useWorldCupGroups(): UseWorldCupGroupsResult {
  const { data, isLoading, error, refetch } = useQuery({
    queryKey: ["world-cup-2026-groups"],
    queryFn: fetchWorldCupGroups,
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
