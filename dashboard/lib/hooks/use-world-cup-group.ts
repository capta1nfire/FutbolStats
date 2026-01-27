"use client";

import { useQuery } from "@tanstack/react-query";
import type { WorldCupGroupDetail, WorldCupGroupDetailResponse } from "@/lib/types";

export interface UseWorldCupGroupResult {
  data: WorldCupGroupDetail | null;
  generatedAt: string | null;
  cached: boolean;
  cacheAgeSeconds: number | null;
  isLoading: boolean;
  error: Error | null;
  refetch: () => void;
}

async function fetchWorldCupGroup(group: string): Promise<WorldCupGroupDetailResponse | null> {
  const response = await fetch(`/api/football/world-cup-2026/group/${encodeURIComponent(group)}`, {
    method: "GET",
    headers: { Accept: "application/json" },
  });

  if (!response.ok) {
    return null;
  }

  return response.json();
}

export function useWorldCupGroup(group: string | null): UseWorldCupGroupResult {
  const { data, isLoading, error, refetch } = useQuery({
    queryKey: ["world-cup-2026-group", group],
    queryFn: () => (group ? fetchWorldCupGroup(group) : Promise.resolve(null)),
    retry: 1,
    staleTime: 60_000,
    refetchOnWindowFocus: false,
    throwOnError: false,
    enabled: !!group,
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
