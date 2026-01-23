/**
 * Predictions Hooks
 *
 * TanStack Query hooks for predictions data.
 * Supports both mock data and real API with fallback.
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
import {
  parsePredictions,
  buildCoverageFromPredictions,
} from "@/lib/api/predictions";

/**
 * Map time range to days_back/days_ahead params
 */
function timeRangeToDays(timeRange?: PredictionTimeRange): {
  days_back: number;
  days_ahead: number;
} {
  switch (timeRange) {
    case "24h":
      return { days_back: 1, days_ahead: 1 };
    case "48h":
      return { days_back: 2, days_ahead: 2 };
    case "7d":
      return { days_back: 7, days_ahead: 7 };
    case "30d":
      return { days_back: 30, days_ahead: 30 };
    default:
      return { days_back: 7, days_ahead: 7 };
  }
}

/**
 * Fetch predictions from API
 */
async function fetchPredictionsApi(
  filters?: PredictionFilters
): Promise<PredictionRow[]> {
  const { days_back, days_ahead } = timeRangeToDays(filters?.timeRange);

  const params = new URLSearchParams();
  params.set("days_back", String(days_back));
  params.set("days_ahead", String(days_ahead));

  // Add league filter if specified
  if (filters?.league && filters.league.length > 0) {
    // Note: Backend expects league_ids as comma-separated IDs
    // For now, we'll filter client-side since we have league names
  }

  const response = await fetch(`/api/predictions?${params.toString()}`);

  if (!response.ok) {
    throw new Error(`API error: ${response.status}`);
  }

  const data = await response.json();
  let predictions = parsePredictions(data);

  // Client-side filtering for fields backend doesn't support
  if (filters?.status && filters.status.length > 0) {
    predictions = predictions.filter((p) => filters.status!.includes(p.status));
  }

  if (filters?.model && filters.model.length > 0) {
    predictions = predictions.filter((p) => filters.model!.includes(p.model));
  }

  if (filters?.league && filters.league.length > 0) {
    predictions = predictions.filter((p) =>
      filters.league!.some((l) =>
        p.leagueName.toLowerCase().includes(l.toLowerCase())
      )
    );
  }

  if (filters?.search) {
    const search = filters.search.toLowerCase();
    predictions = predictions.filter(
      (p) =>
        p.matchLabel.toLowerCase().includes(search) ||
        p.leagueName.toLowerCase().includes(search)
    );
  }

  return predictions;
}

/**
 * Fetch predictions with optional filters - uses mock data
 */
export function usePredictions(filters?: PredictionFilters) {
  return useQuery<PredictionRow[], Error>({
    queryKey: ["predictions", filters],
    queryFn: () => getPredictionsMock(filters),
  });
}

/**
 * Fetch predictions from real API with mock fallback
 */
export function usePredictionsApi(filters?: PredictionFilters) {
  return useQuery<PredictionRow[], Error>({
    queryKey: ["predictions", "api", filters],
    queryFn: () => fetchPredictionsApi(filters),
    staleTime: 30 * 1000, // 30 seconds
    retry: 1,
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
 * Fetch prediction coverage summary (mock)
 */
export function usePredictionCoverage(timeRange: PredictionTimeRange = "24h") {
  return useQuery<PredictionCoverage, Error>({
    queryKey: ["predictions", "coverage", timeRange],
    queryFn: () => getPredictionCoverageMock(timeRange),
    staleTime: 60 * 1000, // 1 minute
  });
}

/**
 * Calculate coverage from predictions data
 */
export function usePredictionCoverageFromData(
  predictions: PredictionRow[],
  periodLabel: string = "Current view"
): PredictionCoverage {
  return buildCoverageFromPredictions(predictions, periodLabel);
}
