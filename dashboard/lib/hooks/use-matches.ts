"use client";

import { useQuery } from "@tanstack/react-query";
import { MatchFilters, MatchSummary, MatchStatus } from "@/lib/types";
import { getMatchesMock, getMatchByIdMock } from "@/lib/mocks";
import {
  parseMatches,
  extractPagination,
  extractMetadata,
  mapStatusFilter,
  MatchesPagination,
} from "@/lib/api/matches";

/**
 * Response from useMatchesApi hook
 */
export interface UseMatchesApiResult {
  /** Parsed matches, null if unavailable */
  matches: MatchSummary[] | null;
  /** Pagination info */
  pagination: MatchesPagination;
  /** True if data fetch failed or parsing failed */
  isDegraded: boolean;
  /** Request ID for debugging */
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
 * Query params for the API
 */
interface MatchesQueryParams {
  status?: string;
  hours?: number;
  league_id?: number;
  match_id?: number;
  page?: number;
  limit?: number;
}

/**
 * Internal response type
 */
interface MatchesData {
  matches: MatchSummary[] | null;
  pagination: MatchesPagination;
  requestId?: string;
  generatedAt: string | null;
  cached: boolean;
  cacheAgeSeconds: number;
}

/**
 * Build query string from params
 */
function buildQueryString(params: MatchesQueryParams): string {
  const searchParams = new URLSearchParams();

  if (params.status) searchParams.set("status", params.status);
  if (params.hours) searchParams.set("hours", params.hours.toString());
  if (params.league_id) searchParams.set("league_id", params.league_id.toString());
  if (params.match_id) searchParams.set("match_id", params.match_id.toString());
  if (params.page) searchParams.set("page", params.page.toString());
  if (params.limit) searchParams.set("limit", params.limit.toString());

  const qs = searchParams.toString();
  return qs ? `?${qs}` : "";
}

/**
 * Fetch matches from proxy endpoint
 */
async function fetchMatches(params: MatchesQueryParams): Promise<MatchesData> {
  const queryString = buildQueryString(params);
  const response = await fetch(`/api/matches${queryString}`, {
    method: "GET",
    headers: {
      Accept: "application/json",
    },
  });

  const requestId = response.headers.get("x-request-id") || undefined;

  if (!response.ok) {
    return {
      matches: null,
      pagination: { total: 0, page: 1, limit: 50, pages: 1 },
      requestId,
      generatedAt: null,
      cached: false,
      cacheAgeSeconds: 0,
    };
  }

  const data = await response.json();
  const matches = parseMatches(data);
  const pagination = extractPagination(data);
  const metadata = extractMetadata(data);

  return {
    matches,
    pagination,
    requestId,
    generatedAt: metadata.generatedAt,
    cached: metadata.cached,
    cacheAgeSeconds: metadata.cacheAgeSeconds,
  };
}

/**
 * Hook to fetch matches from backend via /api/matches proxy
 *
 * Features:
 * - Enterprise-safe: uses same-origin proxy, no secrets exposed
 * - Graceful degradation: returns isDegraded=true if fetch/parse fails
 * - Pagination support
 * - Status/league filtering
 *
 * Usage:
 * ```tsx
 * const { matches, isDegraded, pagination, isLoading } = useMatchesApi({
 *   status: ["scheduled"],
 *   page: 1,
 *   limit: 50,
 * });
 *
 * if (isLoading) return <Loader />;
 *
 * const displayMatches = matches ?? mockMatches;
 * ```
 */
export function useMatchesApi(options?: {
  status?: MatchStatus[];
  hours?: number;
  leagueId?: number;
  page?: number;
  limit?: number;
  enabled?: boolean;
}): UseMatchesApiResult {
  const {
    status = [],
    hours = 168, // 7 days default
    leagueId,
    page = 1,
    limit = 50,
    enabled = true,
  } = options || {};

  // Map frontend status filter to backend
  const backendStatus = mapStatusFilter(status);

  const queryParams: MatchesQueryParams = {
    status: backendStatus,
    hours,
    league_id: leagueId,
    page,
    limit,
  };

  // Cache timing aligned with backend TTL (P1 auditor)
  const isLive = backendStatus === "LIVE";
  const staleTime = isLive ? 15_000 : 60_000;
  const refetchInterval = isLive ? 15_000 : 60_000;

  const { data, isLoading, error, refetch } = useQuery({
    queryKey: ["matches-api", queryParams],
    queryFn: () => fetchMatches(queryParams),
    retry: 1,
    staleTime,
    refetchInterval,
    refetchOnWindowFocus: false,
    throwOnError: false,
    enabled,
  });

  const matches = data?.matches ?? null;
  const pagination = data?.pagination ?? { total: 0, page: 1, limit: 50, pages: 1 };
  const requestId = data?.requestId;
  const generatedAt = data?.generatedAt ?? null;
  const cached = data?.cached ?? false;
  const cacheAgeSeconds = data?.cacheAgeSeconds ?? 0;
  const isDegraded = !!error || matches === null;

  return {
    matches,
    pagination,
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

/**
 * Hook to fetch matches with optional filters (mock fallback)
 * @deprecated Use useMatchesApi for real data with mock fallback
 */
export function useMatches(filters?: MatchFilters) {
  return useQuery<MatchSummary[], Error>({
    queryKey: ["matches", filters],
    queryFn: () => getMatchesMock(filters),
  });
}

/**
 * Hook to fetch a single match by ID (mock only, used as fallback)
 */
export function useMatch(id: number | null) {
  return useQuery<MatchSummary | null, Error>({
    queryKey: ["match", id],
    queryFn: () => (id ? getMatchByIdMock(id) : null),
    enabled: id !== null,
  });
}

/**
 * Hook to fetch a single match by ID from backend via /api/matches proxy.
 * Uses backend support for match_id query param (deep-link support).
 */
export function useMatchApi(id: number | null): {
  match: MatchSummary | null;
  isDegraded: boolean;
  requestId?: string;
  isLoading: boolean;
  error: Error | null;
  refetch: () => void;
} {
  const { data, isLoading, error, refetch } = useQuery({
    queryKey: ["match-api", id],
    queryFn: async () => {
      if (id === null) {
        return { match: null as MatchSummary | null, requestId: undefined as string | undefined };
      }

      // Fetch only by match_id. Backend ignores other filters when match_id is provided.
      const response = await fetch(`/api/matches?match_id=${id}&limit=1`, {
        method: "GET",
        headers: { Accept: "application/json" },
      });

      const requestId = response.headers.get("x-request-id") || undefined;

      if (!response.ok) {
        return { match: null as MatchSummary | null, requestId };
      }

      const raw = await response.json();
      const matches = parseMatches(raw);
      const match = matches && matches.length > 0 ? matches[0] : null;
      return { match, requestId };
    },
    retry: 1,
    staleTime: 15_000,
    refetchInterval: 30_000,
    refetchOnWindowFocus: false,
    throwOnError: false,
    enabled: id !== null,
  });

  const match = data?.match ?? null;
  const requestId = data?.requestId;
  const isDegraded = !!error || (id !== null && match === null);

  return {
    match,
    isDegraded,
    requestId,
    isLoading,
    error: error as Error | null,
    refetch: () => refetch(),
  };
}
