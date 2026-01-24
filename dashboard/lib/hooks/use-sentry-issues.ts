"use client";

import { useQuery } from "@tanstack/react-query";
import {
  SentryIssue,
  SentryIssuesResponse,
  SentryIssuesRawResponse,
  normalizeSentryIssue,
} from "@/lib/types/drawer-v2";

export type SentryIssuesRange = "1h" | "24h" | "7d";

export interface UseSentryIssuesParams {
  range?: SentryIssuesRange;
  page?: number;
  limit?: number;
  enabled?: boolean;
}

export interface UseSentryIssuesResult {
  issues: SentryIssue[];
  total: number;
  page: number;
  limit: number;
  hasMore: boolean;
  range: SentryIssuesRange;
  isLoading: boolean;
  isError: boolean;
  isDegraded: boolean;
  error: Error | null;
  refetch: () => void;
}

/**
 * Fetch and normalize Sentry issues from v2 endpoint
 */
async function fetchSentryIssues(
  range: SentryIssuesRange,
  page: number,
  limit: number
): Promise<SentryIssuesResponse> {
  const params = new URLSearchParams({
    range,
    page: page.toString(),
    limit: limit.toString(),
  });

  const response = await fetch(`/api/sentry/issues?${params.toString()}`);

  if (!response.ok) {
    throw new Error(`Failed to fetch sentry issues: ${response.status}`);
  }

  const raw: SentryIssuesRawResponse = await response.json();

  // Normalize backend response to frontend shape
  return {
    issues: raw.data.issues.map(normalizeSentryIssue),
    total: raw.data.total,
    page: raw.data.page,
    limit: raw.data.limit,
    hasMore: raw.data.page < raw.data.pages,
    range,
  };
}

/**
 * Hook for fetching paginated Sentry issues
 *
 * Features:
 * - Paginated with range filter (1h, 24h, 7d)
 * - isDegraded flag for fallback UI
 * - Auto-refetch on window focus
 */
export function useSentryIssues(
  params: UseSentryIssuesParams = {}
): UseSentryIssuesResult {
  const {
    range = "24h",
    page = 1,
    limit = 20,
    enabled = true,
  } = params;

  const query = useQuery({
    queryKey: ["sentry-issues", range, page, limit],
    queryFn: () => fetchSentryIssues(range, page, limit),
    enabled,
    staleTime: 30_000, // 30 seconds
    refetchOnWindowFocus: true,
    retry: 1,
  });

  // Determine if degraded (error or no data)
  const isDegraded = query.isError || (!query.isLoading && !query.data);

  return {
    issues: query.data?.issues ?? [],
    total: query.data?.total ?? 0,
    page: query.data?.page ?? page,
    limit: query.data?.limit ?? limit,
    hasMore: query.data?.hasMore ?? false,
    range: query.data?.range ?? range,
    isLoading: query.isLoading,
    isError: query.isError,
    isDegraded,
    error: query.error as Error | null,
    refetch: query.refetch,
  };
}
