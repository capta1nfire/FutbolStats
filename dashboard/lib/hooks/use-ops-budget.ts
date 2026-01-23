"use client";

import { useQuery } from "@tanstack/react-query";
import { ApiBudget, HealthSummary } from "@/lib/types";
import {
  parseOpsBudget,
  parseOpsHealth,
  parseOpsSentry,
  parseOpsJobsHealth,
  parseOpsFastpathHealth,
  parseOpsPredictionsHealth,
  parseOpsShadowMode,
  parseOpsSensorB,
  parseOpsLlmCost,
  parseOpsFreshness,
  parseOpsTelemetry,
  parseOpsProgress,
  parseOpsPitActivity,
  parseOpsMovement,
  parseOpsSotaEnrichment,
  OpsSentrySummary,
  OpsJobsHealth,
  OpsFastpathHealth,
  OpsPredictionsHealth,
  OpsShadowMode,
  OpsSensorB,
  OpsLlmCost,
  OpsFreshness,
  OpsTelemetry,
  OpsProgress,
  OpsPitActivity,
  OpsMovement,
  SotaEnrichmentNormalized,
} from "@/lib/api/ops";

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
  sentry: OpsSentrySummary | null;
  jobs: OpsJobsHealth | null;
  fastpath: OpsFastpathHealth | null;
  predictions: OpsPredictionsHealth | null;
  shadowMode: OpsShadowMode | null;
  sensorB: OpsSensorB | null;
  llmCost: OpsLlmCost | null;
  freshness: OpsFreshness | null;
  telemetry: OpsTelemetry | null;
  progress: OpsProgress | null;
  pitActivity: OpsPitActivity | null;
  movement: OpsMovement | null;
  sotaEnrichment: SotaEnrichmentNormalized | null;
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
    return {
      budget: null,
      health: null,
      sentry: null,
      jobs: null,
      fastpath: null,
      predictions: null,
      shadowMode: null,
      sensorB: null,
      llmCost: null,
      freshness: null,
      telemetry: null,
      progress: null,
      pitActivity: null,
      movement: null,
      sotaEnrichment: null,
      requestId,
    };
  }

  const data = await response.json();
  const budget = parseOpsBudget(data);
  const health = parseOpsHealth(data);
  const sentry = parseOpsSentry(data);
  const jobs = parseOpsJobsHealth(data);
  const fastpath = parseOpsFastpathHealth(data);
  const predictions = parseOpsPredictionsHealth(data);
  const shadowMode = parseOpsShadowMode(data);
  const sensorB = parseOpsSensorB(data);
  const llmCost = parseOpsLlmCost(data);
  const freshness = parseOpsFreshness(data);
  const telemetry = parseOpsTelemetry(data);
  const progress = parseOpsProgress(data);
  const pitActivity = parseOpsPitActivity(data);
  const movement = parseOpsMovement(data);
  const sotaEnrichment = parseOpsSotaEnrichment(data);

  return {
    budget,
    health,
    sentry,
    jobs,
    fastpath,
    predictions,
    shadowMode,
    sensorB,
    llmCost,
    freshness,
    telemetry,
    progress,
    pitActivity,
    movement,
    sotaEnrichment,
    requestId,
  };
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
  /** Parsed sentry summary, null if unavailable */
  sentry: OpsSentrySummary | null;
  /** Parsed jobs health, null if unavailable */
  jobs: OpsJobsHealth | null;
  /** Parsed fastpath health, null if unavailable */
  fastpath: OpsFastpathHealth | null;
  /** Parsed predictions health, null if unavailable */
  predictions: OpsPredictionsHealth | null;
  /** Parsed shadow mode, null if unavailable */
  shadowMode: OpsShadowMode | null;
  /** Parsed sensor B, null if unavailable */
  sensorB: OpsSensorB | null;
  /** Parsed LLM cost, null if unavailable */
  llmCost: OpsLlmCost | null;
  /** Parsed freshness metadata, null if unavailable */
  freshness: OpsFreshness | null;
  /** Parsed telemetry (data quality), null if unavailable */
  telemetry: OpsTelemetry | null;
  /** Parsed progress (PIT evaluation), null if unavailable */
  progress: OpsProgress | null;
  /** Parsed PIT activity, null if unavailable */
  pitActivity: OpsPitActivity | null;
  /** Parsed movement data, null if unavailable */
  movement: OpsMovement | null;
  /** Parsed SOTA enrichment, null if unavailable */
  sotaEnrichment: SotaEnrichmentNormalized | null;
  /** True if data fetch failed or all parsing failed */
  isDegraded: boolean;
  /** True if budget specifically is degraded */
  isBudgetDegraded: boolean;
  /** True if health specifically is degraded */
  isHealthDegraded: boolean;
  /** True if sentry specifically is degraded */
  isSentryDegraded: boolean;
  /** True if jobs specifically is degraded */
  isJobsDegraded: boolean;
  /** True if fastpath specifically is degraded */
  isFastpathDegraded: boolean;
  /** True if predictions specifically is degraded */
  isPredictionsDegraded: boolean;
  /** True if shadow mode specifically is degraded */
  isShadowModeDegraded: boolean;
  /** True if sensor B specifically is degraded */
  isSensorBDegraded: boolean;
  /** True if LLM cost specifically is degraded */
  isLlmCostDegraded: boolean;
  /** True if telemetry specifically is degraded */
  isTelemetryDegraded: boolean;
  /** True if progress specifically is degraded */
  isProgressDegraded: boolean;
  /** True if PIT activity specifically is degraded */
  isPitActivityDegraded: boolean;
  /** True if movement specifically is degraded */
  isMovementDegraded: boolean;
  /** True if SOTA enrichment specifically is degraded */
  isSotaEnrichmentDegraded: boolean;
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
  const sentry = data?.sentry ?? null;
  const jobs = data?.jobs ?? null;
  const fastpath = data?.fastpath ?? null;
  const predictions = data?.predictions ?? null;
  const shadowMode = data?.shadowMode ?? null;
  const sensorB = data?.sensorB ?? null;
  const llmCost = data?.llmCost ?? null;
  const freshness = data?.freshness ?? null;
  const telemetry = data?.telemetry ?? null;
  const progress = data?.progress ?? null;
  const pitActivity = data?.pitActivity ?? null;
  const movement = data?.movement ?? null;
  const sotaEnrichment = data?.sotaEnrichment ?? null;
  const requestId = data?.requestId;

  const isBudgetDegraded = !!error || budget === null;
  const isHealthDegraded = !!error || health === null;
  const isSentryDegraded = !!error || sentry === null;
  const isJobsDegraded = !!error || jobs === null;
  const isFastpathDegraded = !!error || fastpath === null;
  const isPredictionsDegraded = !!error || predictions === null;
  const isShadowModeDegraded = !!error || shadowMode === null;
  const isSensorBDegraded = !!error || sensorB === null;
  const isLlmCostDegraded = !!error || llmCost === null;
  const isTelemetryDegraded = !!error || telemetry === null;
  const isProgressDegraded = !!error || progress === null;
  const isPitActivityDegraded = !!error || pitActivity === null;
  const isMovementDegraded = !!error || movement === null;
  const isSotaEnrichmentDegraded = !!error || sotaEnrichment === null;
  // isDegraded = ALL core blocks failed (not diagnostics)
  const isDegraded = isBudgetDegraded && isHealthDegraded && isJobsDegraded && isPredictionsDegraded;

  return {
    budget,
    health,
    sentry,
    jobs,
    fastpath,
    predictions,
    shadowMode,
    sensorB,
    llmCost,
    freshness,
    telemetry,
    progress,
    pitActivity,
    movement,
    sotaEnrichment,
    isDegraded,
    isBudgetDegraded,
    isHealthDegraded,
    isSentryDegraded,
    isJobsDegraded,
    isFastpathDegraded,
    isPredictionsDegraded,
    isShadowModeDegraded,
    isSensorBDegraded,
    isLlmCostDegraded,
    isTelemetryDegraded,
    isProgressDegraded,
    isPitActivityDegraded,
    isMovementDegraded,
    isSotaEnrichmentDegraded,
    requestId,
    isLoading,
    error: error as Error | null,
    refetch: () => refetch(),
  };
}
