"use client";

import { useQuery } from "@tanstack/react-query";
import { OverviewRollupResponse } from "@/lib/types/drawer-v2";

export interface UseOverviewRollupResult {
  data: OverviewRollupResponse | null;
  isLoading: boolean;
  isError: boolean;
  isDegraded: boolean;
  error: Error | null;
  refetch: () => void;
}

/**
 * Fetch overview rollup from v2 endpoint
 */
async function fetchOverviewRollup(): Promise<OverviewRollupResponse> {
  const response = await fetch("/api/overview/rollup");

  if (!response.ok) {
    throw new Error(`Failed to fetch overview rollup: ${response.status}`);
  }

  return response.json();
}

/**
 * Hook for fetching overview rollup data
 *
 * Features:
 * - Aggregated metrics for overview page
 * - isDegraded flag for fallback UI
 * - Auto-refetch on window focus
 */
export function useOverviewRollup(): UseOverviewRollupResult {
  const query = useQuery({
    queryKey: ["overview-rollup"],
    queryFn: fetchOverviewRollup,
    staleTime: 30_000, // 30 seconds
    refetchOnWindowFocus: true,
    retry: 1,
  });

  // Determine if degraded (error or no data)
  const isDegraded = query.isError || (!query.isLoading && !query.data);

  return {
    data: query.data ?? null,
    isLoading: query.isLoading,
    isError: query.isError,
    isDegraded,
    error: query.error as Error | null,
    refetch: query.refetch,
  };
}
