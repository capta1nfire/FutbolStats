/**
 * Model Benchmark hook
 *
 * Fetches model benchmark data from the backend via proxy.
 * Supports dynamic model selection with minimum 2 models required.
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
  /** Validation error message (e.g., "less than 2 models") */
  validationError: string | null;
  /** Refetch function */
  refetch: () => void;
}

/**
 * Fetch model benchmark data from proxy endpoint
 */
async function fetchModelBenchmark(
  selectedModels: string[]
): Promise<ModelBenchmarkResponse | null> {
  const params = new URLSearchParams();
  params.set("models", selectedModels.join(","));

  const response = await fetch(`/api/model-benchmark?${params.toString()}`, {
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
 * Hook to fetch model benchmark data with dynamic model selection
 *
 * Features:
 * - Dynamic model selection via query param
 * - Minimum 2 models required for comparison
 * - Enterprise-safe: uses same-origin proxy, no secrets exposed
 * - Graceful degradation: returns isDegraded=true if fetch fails
 * - 5-minute stale time (data doesn't change frequently)
 * - Single retry on failure
 *
 * Usage:
 * ```tsx
 * const [selectedModels, setSelectedModels] = useState(["Market", "Model A", "Shadow", "Sensor B"]);
 * const { data, isLoading, isDegraded, validationError } = useModelBenchmark(selectedModels);
 *
 * if (validationError) return <Alert>{validationError}</Alert>;
 * if (isLoading) return <Loader />;
 * if (isDegraded) return <ErrorState />;
 *
 * // Use data.daily_data for chart, data.models for summary
 * ```
 */
export function useModelBenchmark(
  selectedModels: string[]
): UseModelBenchmarkResult {
  const enabled = selectedModels.length >= 2;

  const { data, isLoading, error, refetch } = useQuery({
    queryKey: ["model-benchmark", selectedModels.sort().join(",")],
    queryFn: () => fetchModelBenchmark(selectedModels),
    enabled,
    retry: 1,
    staleTime: 5 * 60 * 1000, // 5 minutes
    refetchInterval: 5 * 60 * 1000, // Refetch every 5 minutes
    refetchOnWindowFocus: false,
    throwOnError: false,
  });

  const validationError =
    !enabled && selectedModels.length > 0
      ? "Selecciona al menos 2 modelos para comparar"
      : null;

  return {
    data: data ?? null,
    isLoading: enabled ? isLoading : false,
    isDegraded: enabled ? !!error || data === null : true,
    error: error as Error | null,
    validationError,
    refetch: () => refetch(),
  };
}
