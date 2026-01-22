"use client";

import { useQuery } from "@tanstack/react-query";
import { ApiBudget } from "@/lib/types";
import { parseOpsBudget } from "@/lib/api/ops";

/**
 * Response from useOpsBudget hook
 */
export interface UseOpsBudgetResult {
  /** Parsed budget data, null if unavailable */
  budget: ApiBudget | null;
  /** True if data fetch failed or parsing failed */
  isDegraded: boolean;
  /** Request ID for debugging (from response header) */
  requestId?: string;
  /** Loading state */
  isLoading: boolean;
  /** Error object if fetch failed */
  error: Error | null;
  /** Refetch function */
  refetch: () => void;
}

/**
 * Fetch ops data from proxy endpoint
 */
async function fetchOpsBudget(): Promise<{
  budget: ApiBudget | null;
  requestId?: string;
}> {
  const response = await fetch("/api/ops", {
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
    return { budget: null, requestId };
  }

  const data = await response.json();
  const budget = parseOpsBudget(data);

  return { budget, requestId };
}

/**
 * Hook to fetch real API budget from backend via /api/ops proxy
 *
 * Features:
 * - Enterprise-safe: uses same-origin proxy, no secrets exposed
 * - Graceful degradation: returns isDegraded=true if fetch/parse fails
 * - Auto-refetch: refreshes every 45s (aligned with backend cache)
 * - Single retry on failure
 *
 * Usage:
 * ```tsx
 * const { budget, isDegraded, isLoading } = useOpsBudget();
 *
 * if (isLoading) return <Loader />;
 * if (isDegraded) return <ApiBudgetCard budget={mockBudget} isDegraded />;
 * return <ApiBudgetCard budget={budget} />;
 * ```
 */
export function useOpsBudget(): UseOpsBudgetResult {
  const {
    data,
    isLoading,
    error,
    refetch,
  } = useQuery({
    queryKey: ["ops-budget"],
    queryFn: fetchOpsBudget,
    retry: 1,
    staleTime: 30_000, // Consider data fresh for 30s
    refetchInterval: 45_000, // Refetch every 45s (aligned with backend cache)
    refetchOnWindowFocus: false, // Avoid noise on tab switch
    // Don't throw on error - we handle degradation gracefully
    throwOnError: false,
  });

  const budget = data?.budget ?? null;
  const requestId = data?.requestId;
  const isDegraded = !!error || budget === null;

  return {
    budget,
    isDegraded,
    requestId,
    isLoading,
    error: error as Error | null,
    refetch: () => refetch(),
  };
}
