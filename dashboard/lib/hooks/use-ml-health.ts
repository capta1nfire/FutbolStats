/**
 * ML Health data hook using TanStack Query
 *
 * Fetches ML Health dashboard data from backend via proxy.
 * Provides graceful degradation if fetch fails.
 */

"use client";

import { useQuery } from "@tanstack/react-query";
import { MLHealthResponse, MLHealthData, MLHealthStatus } from "@/lib/types";

/**
 * Hook result interface
 */
export interface UseMLHealthResult {
  /** Full ML Health data, null if unavailable */
  data: MLHealthData | null;
  /** Root health status */
  health: MLHealthStatus | null;
  /** True if data fetch failed */
  isDegraded: boolean;
  /** Request ID for debugging */
  requestId?: string;
  /** When backend generated this data */
  generatedAt: string | null;
  /** Whether data is from backend cache */
  cached: boolean;
  /** Age of backend cache in seconds */
  cacheAgeSeconds: number | null;
  /** Loading state */
  isLoading: boolean;
  /** Error object if fetch failed */
  error: Error | null;
  /** Refetch function */
  refetch: () => void;
}

/**
 * Internal parsed data type
 */
interface MLHealthParsedData {
  data: MLHealthData | null;
  health: MLHealthStatus | null;
  requestId?: string;
  generatedAt: string | null;
  cached: boolean;
  cacheAgeSeconds: number | null;
}

/**
 * Fetch ML Health data from proxy endpoint
 */
async function fetchMLHealth(): Promise<MLHealthParsedData> {
  const response = await fetch("/api/ml-health", {
    method: "GET",
    headers: {
      Accept: "application/json",
    },
  });

  const requestId = response.headers.get("x-request-id") || undefined;

  if (!response.ok) {
    return {
      data: null,
      health: null,
      requestId,
      generatedAt: null,
      cached: false,
      cacheAgeSeconds: null,
    };
  }

  const json: MLHealthResponse = await response.json();

  return {
    data: json.data,
    health: json.health,
    requestId,
    generatedAt: json.generated_at,
    cached: json.cached,
    cacheAgeSeconds: json.cache_age_seconds,
  };
}

/**
 * Hook to fetch ML Health data from backend via /api/ml-health proxy
 *
 * Features:
 * - Enterprise-safe: uses same-origin proxy, no secrets exposed
 * - Graceful degradation: returns isDegraded=true if fetch fails
 * - Auto-refetch: refreshes every 60s
 * - Single retry on failure
 *
 * Usage:
 * ```tsx
 * const { data, health, isDegraded, isLoading } = useMLHealth();
 *
 * if (isLoading) return <Loader />;
 * if (isDegraded) return <DegradedBanner />;
 *
 * return <MLHealthCards data={data} />;
 * ```
 */
export function useMLHealth(): UseMLHealthResult {
  const { data, isLoading, error, refetch } = useQuery({
    queryKey: ["ml-health"],
    queryFn: fetchMLHealth,
    retry: 1,
    staleTime: 30_000, // Consider data fresh for 30s
    refetchInterval: 60_000, // Refetch every 60s
    refetchOnWindowFocus: false,
    throwOnError: false,
  });

  const mlData = data?.data ?? null;
  const health = data?.health ?? null;
  const requestId = data?.requestId;
  const generatedAt = data?.generatedAt ?? null;
  const cached = data?.cached ?? false;
  const cacheAgeSeconds = data?.cacheAgeSeconds ?? null;
  const isDegraded = !!error || mlData === null;

  return {
    data: mlData,
    health,
    isDegraded,
    requestId,
    generatedAt,
    cached,
    cacheAgeSeconds,
    isLoading,
    error: error as Error | null,
    refetch: () => refetch(),
  };
}
