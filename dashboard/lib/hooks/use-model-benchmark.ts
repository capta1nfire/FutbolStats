/**
 * Model Benchmark hook
 *
 * Fetches model benchmark data from the backend via proxy.
 * Compares Market, Model A, and Shadow accuracy over time.
 */

"use client";

import { useQuery } from "@tanstack/react-query";
import { ModelBenchmarkResponse } from "@/lib/types/model-benchmark";

/**
 * Result type for useModelBenchmark hook
 */
export interface UseModelBenchmarkResult {
  /** Benchmark data, null if unavailable */
  data: ModelBenchmarkResponse | null;
  /** True while initial fetch is in progress */
  isLoading: boolean;
  /** True if fetch failed or data is null */
  isDegraded: boolean;
  /** Error object if fetch failed */
  error: Error | null;
  /** Refetch function */
  refetch: () => void;
}

/**
 * Fetch model benchmark data from proxy endpoint
 */
async function fetchModelBenchmark(): Promise<ModelBenchmarkResponse | null> {
  const response = await fetch("/api/model-benchmark", {
    method: "GET",
    headers: {
      Accept: "application/json",
    },
  });

  if (!response.ok) {
    console.error("Failed to fetch model benchmark:", response.status);
    return null;
  }

  return response.json();
}

/**
 * Hook to fetch model benchmark data
 *
 * Features:
 * - Enterprise-safe: uses same-origin proxy, no secrets exposed
 * - Graceful degradation: returns isDegraded=true if fetch fails
 * - 5-minute stale time (data doesn't change frequently)
 * - Auto-refetch every 5 minutes
 * - Single retry on failure
 *
 * Usage:
 * ```tsx
 * const { data, isLoading, isDegraded } = useModelBenchmark();
 *
 * if (isLoading) return <Loader />;
 * if (isDegraded) return <ErrorState />;
 *
 * // Use data.daily_data for chart, data.models for summary
 * ```
 */
export function useModelBenchmark(): UseModelBenchmarkResult {
  const { data, isLoading, error, refetch } = useQuery({
    queryKey: ["model-benchmark"],
    queryFn: fetchModelBenchmark,
    retry: 1,
    staleTime: 5 * 60 * 1000, // 5 minutes
    refetchInterval: 5 * 60 * 1000, // Refetch every 5 minutes
    refetchOnWindowFocus: false,
    throwOnError: false,
  });

  return {
    data: data ?? null,
    isLoading,
    isDegraded: !!error || data === null,
    error: error as Error | null,
    refetch: () => refetch(),
  };
}
