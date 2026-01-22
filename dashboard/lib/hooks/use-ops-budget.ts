"use client";

import { useQuery } from "@tanstack/react-query";
import { ApiBudget, HealthSummary } from "@/lib/types";
import { parseOpsBudget, parseOpsHealth } from "@/lib/api/ops";

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
 * Combined ops data response
 */
interface OpsData {
  budget: ApiBudget | null;
  health: HealthSummary | null;
  requestId?: string;
}

/**
 * Fetch ops data from proxy endpoint
 *
 * Parses both budget and health from the same response.
 */
async function fetchOpsData(): Promise<OpsData> {
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
    return { budget: null, health: null, requestId };
  }

  const data = await response.json();
  const budget = parseOpsBudget(data);
  const health = parseOpsHealth(data);

  return { budget, health, requestId };
}

/**
 * Shared query options for ops data
 */
const opsQueryOptions = {
  queryKey: ["ops-data"],
  queryFn: fetchOpsData,
  retry: 1,
  staleTime: 30_000, // Consider data fresh for 30s
  refetchInterval: 45_000, // Refetch every 45s (aligned with backend cache)
  refetchOnWindowFocus: false, // Avoid noise on tab switch
  throwOnError: false as const,
};

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
  const { data, isLoading, error, refetch } = useQuery(opsQueryOptions);

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

// ============================================================================
// Overview Hook (Budget + Health)
// ============================================================================

/**
 * Response from useOpsOverview hook
 */
export interface UseOpsOverviewResult {
  /** Parsed budget data, null if unavailable */
  budget: ApiBudget | null;
  /** Parsed health summary, null if unavailable */
  health: HealthSummary | null;
  /** True if data fetch failed or all parsing failed */
  isDegraded: boolean;
  /** True if budget specifically is degraded */
  isBudgetDegraded: boolean;
  /** True if health specifically is degraded */
  isHealthDegraded: boolean;
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
 * Hook to fetch real ops data (budget + health) from backend via /api/ops proxy
 *
 * Features:
 * - Enterprise-safe: uses same-origin proxy, no secrets exposed
 * - Graceful degradation: individual fields can degrade independently
 * - Auto-refetch: refreshes every 45s (aligned with backend cache)
 * - Single retry on failure
 * - Shares cache with useOpsBudget (same queryKey)
 *
 * Usage:
 * ```tsx
 * const { budget, health, isDegraded, isLoading } = useOpsOverview();
 *
 * if (isLoading) return <Loader />;
 *
 * // Use real data with mock fallback
 * const displayBudget = budget ?? mockBudget;
 * const displayHealth = health ?? mockHealth;
 * ```
 */
export function useOpsOverview(): UseOpsOverviewResult {
  const { data, isLoading, error, refetch } = useQuery(opsQueryOptions);

  const budget = data?.budget ?? null;
  const health = data?.health ?? null;
  const requestId = data?.requestId;

  const isBudgetDegraded = !!error || budget === null;
  const isHealthDegraded = !!error || health === null;
  const isDegraded = isBudgetDegraded && isHealthDegraded;

  return {
    budget,
    health,
    isDegraded,
    isBudgetDegraded,
    isHealthDegraded,
    requestId,
    isLoading,
    error: error as Error | null,
    refetch: () => refetch(),
  };
}
