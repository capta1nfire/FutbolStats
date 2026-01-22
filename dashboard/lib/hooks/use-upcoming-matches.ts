"use client";

import { useQuery } from "@tanstack/react-query";
import { UpcomingMatch } from "@/lib/types";
import {
  parseUpcomingMatches,
  extractMetadata,
} from "@/lib/api/upcoming-matches";

/**
 * Response from useUpcomingMatches hook
 */
export interface UseUpcomingMatchesResult {
  /** Parsed upcoming matches, null if unavailable */
  matches: UpcomingMatch[] | null;
  /** True if data fetch failed or parsing failed */
  isDegraded: boolean;
  /** Request ID for debugging (from response header) */
  requestId?: string;
  /** When backend generated this data */
  generatedAt: string | null;
  /** Whether data is from backend cache */
  cached: boolean;
  /** Age of backend cache in seconds */
  cacheAgeSeconds: number;
  /** Loading state */
  isLoading: boolean;
  /** Error object if fetch failed */
  error: Error | null;
  /** Refetch function */
  refetch: () => void;
}

/**
 * Internal response type
 */
interface UpcomingMatchesData {
  matches: UpcomingMatch[] | null;
  requestId?: string;
  generatedAt: string | null;
  cached: boolean;
  cacheAgeSeconds: number;
}

/**
 * Fetch upcoming matches from proxy endpoint
 */
async function fetchUpcomingMatches(): Promise<UpcomingMatchesData> {
  const response = await fetch("/api/upcoming-matches", {
    method: "GET",
    headers: {
      Accept: "application/json",
    },
  });

  // Extract request ID from header
  const requestId = response.headers.get("x-request-id") || undefined;

  if (!response.ok) {
    // Return degraded state instead of throwing
    // This allows fallback to mocks without breaking the UI
    return {
      matches: null,
      requestId,
      generatedAt: null,
      cached: false,
      cacheAgeSeconds: 0,
    };
  }

  const data = await response.json();
  const matches = parseUpcomingMatches(data);
  const metadata = extractMetadata(data);

  return {
    matches,
    requestId,
    generatedAt: metadata.generatedAt,
    cached: metadata.cached,
    cacheAgeSeconds: metadata.cacheAgeSeconds,
  };
}

/**
 * Hook to fetch upcoming matches from backend via /api/upcoming-matches proxy
 *
 * Features:
 * - Enterprise-safe: uses same-origin proxy, no secrets exposed
 * - Graceful degradation: returns isDegraded=true if fetch/parse fails
 * - Auto-refetch: refreshes every 60s
 * - Single retry on failure
 *
 * Usage:
 * ```tsx
 * const { matches, isDegraded, isLoading } = useUpcomingMatches();
 *
 * if (isLoading) return <Loader />;
 *
 * const displayMatches = matches ?? mockUpcomingMatches;
 * ```
 */
export function useUpcomingMatches(): UseUpcomingMatchesResult {
  const { data, isLoading, error, refetch } = useQuery({
    queryKey: ["upcoming-matches"],
    queryFn: fetchUpcomingMatches,
    retry: 1,
    staleTime: 30_000, // Consider data fresh for 30s
    refetchInterval: 60_000, // Refetch every 60s
    refetchOnWindowFocus: false, // Avoid noise on tab switch
    // Don't throw on error - we handle degradation gracefully
    throwOnError: false,
  });

  const matches = data?.matches ?? null;
  const requestId = data?.requestId;
  const generatedAt = data?.generatedAt ?? null;
  const cached = data?.cached ?? false;
  const cacheAgeSeconds = data?.cacheAgeSeconds ?? 0;
  const isDegraded = !!error || matches === null;

  return {
    matches,
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
