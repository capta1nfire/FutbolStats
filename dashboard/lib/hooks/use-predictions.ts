/**
 * Predictions Hooks
 *
 * TanStack Query hooks for predictions data.
 * Supports real API with mock fallback and isDegraded indicator.
 */

"use client";

import { useQuery } from "@tanstack/react-query";
import {
  PredictionRow,
  PredictionDetail,
  PredictionFilters,
  PredictionCoverage,
  PredictionTimeRange,
  PredictionStatus,
  ModelType,
} from "@/lib/types";
import {
  getPredictionsMock,
  getPredictionMock,
  getPredictionCoverageMock,
} from "@/lib/mocks";
import {
  parsePredictionsResponse,
  buildCoverageFromPredictions,
  PredictionsPagination,
  PredictionsMetadata,
} from "@/lib/api/predictions";

// ============================================================================
// Types
// ============================================================================

/**
 * Query params for the API
 */
interface PredictionsQueryParams {
  status?: PredictionStatus[];
  model?: ModelType[];
  league_ids?: number[];
  q?: string;
  days_back?: number;
  days_ahead?: number;
  page?: number;
  limit?: number;
}

/**
 * Internal response from fetch
 */
interface PredictionsData {
  predictions: PredictionRow[];
  pagination: PredictionsPagination;
  metadata: PredictionsMetadata;
}

/**
 * Result from usePredictionsApi hook
 */
export interface UsePredictionsApiResult {
  predictions: PredictionRow[];
  pagination: PredictionsPagination;
  metadata: PredictionsMetadata;
  isLoading: boolean;
  error: Error | null;
  isDegraded: boolean;
  refetch: () => void;
}

// ============================================================================
// Helpers
// ============================================================================

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
      return { days_back: 0, days_ahead: 3 }; // Default: 3 days ahead
  }
}

/**
 * Map frontend ModelType to backend model string
 */
function mapModelToBackend(model: ModelType): string {
  return model === "Shadow" ? "shadow" : "baseline";
}

/**
 * Build query string from params
 */
function buildQueryString(params: PredictionsQueryParams): string {
  const searchParams = new URLSearchParams();

  // Multi-select params
  if (params.status && params.status.length > 0) {
    // Map frontend status to backend status if needed
    params.status.forEach((s) => {
      // Backend uses same status names, just forward them
      searchParams.append("status", s);
    });
  }
  if (params.model && params.model.length > 0) {
    params.model.forEach((m) => searchParams.append("model", mapModelToBackend(m)));
  }
  if (params.league_ids && params.league_ids.length > 0) {
    params.league_ids.forEach((id) => searchParams.append("league_ids", id.toString()));
  }

  // Single value params
  if (params.q) searchParams.set("q", params.q);
  if (params.days_back !== undefined) searchParams.set("days_back", params.days_back.toString());
  if (params.days_ahead !== undefined) searchParams.set("days_ahead", params.days_ahead.toString());
  if (params.page) searchParams.set("page", params.page.toString());
  if (params.limit) searchParams.set("limit", params.limit.toString());

  const qs = searchParams.toString();
  return qs ? `?${qs}` : "";
}

/**
 * Fetch predictions from proxy endpoint
 */
async function fetchPredictions(params: PredictionsQueryParams): Promise<PredictionsData> {
  const queryString = buildQueryString(params);
  const response = await fetch(`/api/predictions${queryString}`, {
    method: "GET",
    headers: {
      Accept: "application/json",
    },
  });

  if (!response.ok) {
    throw new Error(`API error: ${response.status}`);
  }

  const data = await response.json();
  const parsed = parsePredictionsResponse(data);

  if (!parsed) {
    throw new Error("Failed to parse predictions response");
  }

  return {
    predictions: parsed.predictions,
    pagination: parsed.pagination,
    metadata: parsed.metadata,
  };
}

// ============================================================================
// Hooks
// ============================================================================

/**
 * Fetch predictions from real API with mock fallback
 *
 * Features:
 * - Server-side filtering (status, model, league_ids, q, days_back, days_ahead)
 * - Real pagination (page, limit, total, pages)
 * - isDegraded indicator for fallback state
 * - Automatic mock fallback on API error
 *
 * Usage:
 * ```tsx
 * const {
 *   predictions,
 *   pagination,
 *   isDegraded,
 *   isLoading,
 *   refetch,
 * } = usePredictionsApi({
 *   status: ["generated"],
 *   model: ["A"],
 *   timeRange: "48h",
 *   page: 1,
 *   limit: 50,
 * });
 * ```
 */
export function usePredictionsApi(options?: {
  status?: PredictionStatus[];
  model?: ModelType[];
  league_ids?: number[];
  q?: string;
  timeRange?: PredictionTimeRange;
  page?: number;
  limit?: number;
  enabled?: boolean;
}): UsePredictionsApiResult {
  const {
    status,
    model,
    league_ids,
    q,
    timeRange,
    page = 1,
    limit = 50,
    enabled = true,
  } = options || {};

  // Convert timeRange to days_back/days_ahead
  const { days_back, days_ahead } = timeRangeToDays(timeRange);

  const queryParams: PredictionsQueryParams = {
    status: status && status.length > 0 ? status : undefined,
    model: model && model.length > 0 ? model : undefined,
    league_ids: league_ids && league_ids.length > 0 ? league_ids : undefined,
    q: q || undefined,
    days_back,
    days_ahead,
    page,
    limit,
  };

  // Fetch from API
  const apiQuery = useQuery<PredictionsData, Error>({
    queryKey: ["predictions", "api", queryParams],
    queryFn: () => fetchPredictions(queryParams),
    staleTime: 30_000, // 30 seconds
    retry: 1,
    enabled,
  });

  // Build filters for mock fallback (client-side filtering)
  const mockFilters: PredictionFilters = {
    status: status,
    model: model,
    timeRange: timeRange,
    search: q,
  };

  // Fallback to mock on error
  const mockQuery = useQuery<PredictionRow[], Error>({
    queryKey: ["predictions", "mock", mockFilters],
    queryFn: () => getPredictionsMock(mockFilters),
    enabled: apiQuery.isError && enabled,
  });

  // Determine which data to use
  const isDegraded = apiQuery.isError;
  const predictions = isDegraded
    ? (mockQuery.data ?? [])
    : (apiQuery.data?.predictions ?? []);
  const pagination: PredictionsPagination = isDegraded
    ? { total: mockQuery.data?.length ?? 0, page: 1, limit: 50, pages: 1 }
    : (apiQuery.data?.pagination ?? { total: 0, page: 1, limit: 50, pages: 1 });
  const metadata: PredictionsMetadata = isDegraded
    ? { generatedAt: null, cached: false, cacheAgeSeconds: 0 }
    : (apiQuery.data?.metadata ?? { generatedAt: null, cached: false, cacheAgeSeconds: 0 });
  const isLoading = apiQuery.isLoading || (isDegraded && mockQuery.isLoading);
  const error = isDegraded ? null : apiQuery.error; // Suppress if mock fallback works

  return {
    predictions,
    pagination,
    metadata,
    isLoading,
    error,
    isDegraded,
    refetch: () => apiQuery.refetch(),
  };
}

/**
 * Fetch predictions with optional filters - uses mock data only
 * @deprecated Use usePredictionsApi for real data with mock fallback
 */
export function usePredictions(filters?: PredictionFilters) {
  return useQuery<PredictionRow[], Error>({
    queryKey: ["predictions", filters],
    queryFn: () => getPredictionsMock(filters),
  });
}

/**
 * Fetch a single prediction by ID (mock only, used for drawer detail)
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

// Re-export types for convenience
export type { PredictionsPagination, PredictionsMetadata };
