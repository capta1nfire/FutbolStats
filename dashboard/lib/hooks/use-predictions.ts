/**
 * Predictions Hooks
 *
 * TanStack Query hooks for predictions data
 */

import { useQuery } from "@tanstack/react-query";
import {
  PredictionRow,
  PredictionDetail,
  PredictionFilters,
  PredictionCoverage,
  PredictionTimeRange,
} from "@/lib/types";
import {
  getPredictionsMock,
  getPredictionMock,
  getPredictionCoverageMock,
} from "@/lib/mocks";

/**
 * Fetch predictions with optional filters
 */
export function usePredictions(filters?: PredictionFilters) {
  return useQuery<PredictionRow[], Error>({
    queryKey: ["predictions", filters],
    queryFn: () => getPredictionsMock(filters),
  });
}

/**
 * Fetch a single prediction by ID
 */
export function usePrediction(id: number | null) {
  return useQuery<PredictionDetail | null, Error>({
    queryKey: ["prediction", id],
    queryFn: () => (id !== null ? getPredictionMock(id) : Promise.resolve(null)),
    enabled: id !== null,
  });
}

/**
 * Fetch prediction coverage summary
 */
export function usePredictionCoverage(timeRange: PredictionTimeRange = "24h") {
  return useQuery<PredictionCoverage, Error>({
    queryKey: ["predictions", "coverage", timeRange],
    queryFn: () => getPredictionCoverageMock(timeRange),
    staleTime: 60 * 1000, // 1 minute
  });
}
