"use client";

import { useQuery } from "@tanstack/react-query";
import type { GroupDetail, GroupDetailResponse } from "@/lib/types";

export interface UseFootballGroupResult {
  data: GroupDetail | null;
  generatedAt: string | null;
  cached: boolean;
  cacheAgeSeconds: number | null;
  isLoading: boolean;
  error: Error | null;
  refetch: () => void;
}

async function fetchFootballGroup(id: number): Promise<GroupDetailResponse | null> {
  const response = await fetch(`/api/football/group/${id}`, {
    method: "GET",
    headers: { Accept: "application/json" },
  });

  if (!response.ok) {
    return null;
  }

  return response.json();
}

export function useFootballGroup(groupId: number | null): UseFootballGroupResult {
  const { data, isLoading, error, refetch } = useQuery({
    queryKey: ["football-group", groupId],
    queryFn: () => fetchFootballGroup(groupId!),
    enabled: groupId !== null,
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
