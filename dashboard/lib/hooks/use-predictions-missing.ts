"use client";

import { useQuery } from "@tanstack/react-query";
import {
  MissingPredictionMatch,
  MissingPredictionsResponse,
  MissingPredictionsRawResponse,
} from "@/lib/types/drawer-v2";

export interface UsePredictionsMissingParams {
  hours?: number;
  leagueIds?: number[];
  page?: number;
  limit?: number;
  enabled?: boolean;
}

export interface UsePredictionsMissingResult {
  matches: MissingPredictionMatch[];
  total: number;
  page: number;
  limit: number;
  hasMore: boolean;
  hours: number;
  isLoading: boolean;
  isError: boolean;
  isDegraded: boolean;
  error: Error | null;
  refetch: () => void;
}

/**
 * Fetch and normalize missing predictions from v2 endpoint
 */
async function fetchMissingPredictions(
  hours: number,
  leagueIds: number[] | undefined,
  page: number,
  limit: number
): Promise<MissingPredictionsResponse> {
  const params = new URLSearchParams({
    hours: hours.toString(),
    page: page.toString(),
    limit: limit.toString(),
  });

  if (leagueIds && leagueIds.length > 0) {
    params.set("league_ids", leagueIds.join(","));
  }

  const response = await fetch(`/api/predictions/missing?${params.toString()}`);

  if (!response.ok) {
    throw new Error(`Failed to fetch missing predictions: ${response.status}`);
  }

  const raw: MissingPredictionsRawResponse = await response.json();

  // Normalize backend response to frontend shape
  return {
    matches: raw.data.missing,
    total: raw.data.total,
    page: raw.data.page,
    limit: raw.data.limit,
    hasMore: raw.data.page < raw.data.pages,
    hours,
  };
}

/**
 * Hook for fetching paginated missing predictions
 *
 * Features:
 * - Filter by hours window (default 48h)
 * - Optional league_ids filter
 * - Paginated results
 * - isDegraded flag for fallback UI
 */
export function usePredictionsMissing(
  params: UsePredictionsMissingParams = {}
): UsePredictionsMissingResult {
  const {
    hours = 48,
    leagueIds,
    page = 1,
    limit = 20,
    enabled = true,
  } = params;

  const query = useQuery({
    queryKey: ["predictions-missing", hours, leagueIds, page, limit],
    queryFn: () => fetchMissingPredictions(hours, leagueIds, page, limit),
    enabled,
    staleTime: 60_000, // 1 minute
    refetchOnWindowFocus: true,
    retry: 1,
  });

  // Determine if degraded (error or no data)
  const isDegraded = query.isError || (!query.isLoading && !query.data);

  return {
    matches: query.data?.matches ?? [],
    total: query.data?.total ?? 0,
    page: query.data?.page ?? page,
    limit: query.data?.limit ?? limit,
    hasMore: query.data?.hasMore ?? false,
    hours: query.data?.hours ?? hours,
    isLoading: query.isLoading,
    isError: query.isError,
    isDegraded,
    error: query.error as Error | null,
    refetch: query.refetch,
  };
}
