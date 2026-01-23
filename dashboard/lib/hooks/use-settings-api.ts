"use client";

/**
 * Settings API Hooks
 *
 * TanStack Query hooks for fetching settings data from backend.
 * Graceful degradation: returns isDegraded=true if fetch/parse fails.
 */

import { useQuery } from "@tanstack/react-query";
import {
  parseSettingsSummary,
  parseFeatureFlags,
  parseModelVersions,
  SettingsSummary,
  FeatureFlagsResponse,
  ModelVersionsResponse,
} from "@/lib/api/settings";

// ============================================================================
// Settings Summary Hook
// ============================================================================

export interface UseSettingsSummaryResult {
  summary: SettingsSummary | null;
  isDegraded: boolean;
  isLoading: boolean;
  error: Error | null;
  refetch: () => void;
}

async function fetchSettingsSummary(): Promise<SettingsSummary | null> {
  const response = await fetch("/api/settings/summary", {
    method: "GET",
    headers: { Accept: "application/json" },
  });

  if (!response.ok) {
    return null;
  }

  const data = await response.json();
  return parseSettingsSummary(data);
}

/**
 * Hook to fetch settings summary from backend
 *
 * Features:
 * - Enterprise-safe: uses same-origin proxy
 * - Graceful degradation: isDegraded=true if fetch/parse fails
 * - Stale time: 5 minutes (summary rarely changes)
 */
export function useSettingsSummaryApi(): UseSettingsSummaryResult {
  const { data, isLoading, error, refetch } = useQuery({
    queryKey: ["settings", "summary", "api"],
    queryFn: fetchSettingsSummary,
    retry: 1,
    staleTime: 5 * 60 * 1000, // 5 minutes
    refetchOnWindowFocus: false,
    throwOnError: false,
  });

  return {
    summary: data ?? null,
    isDegraded: !!error || data === null,
    isLoading,
    error: error as Error | null,
    refetch: () => refetch(),
  };
}

// ============================================================================
// Feature Flags Hook
// ============================================================================

export interface UseFeatureFlagsParams {
  q?: string;
  enabled?: boolean;
  scope?: string;
  page?: number;
  limit?: number;
}

export interface UseFeatureFlagsResult {
  flags: FeatureFlagsResponse | null;
  isDegraded: boolean;
  isLoading: boolean;
  error: Error | null;
  refetch: () => void;
}

async function fetchFeatureFlags(params: UseFeatureFlagsParams): Promise<FeatureFlagsResponse | null> {
  const searchParams = new URLSearchParams();

  if (params.q) searchParams.set("q", params.q);
  if (params.enabled !== undefined) searchParams.set("enabled", String(params.enabled));
  if (params.scope) searchParams.set("scope", params.scope);
  if (params.page) searchParams.set("page", String(params.page));
  if (params.limit) searchParams.set("limit", String(params.limit));

  const queryString = searchParams.toString();
  const url = `/api/settings/feature-flags${queryString ? `?${queryString}` : ""}`;

  const response = await fetch(url, {
    method: "GET",
    headers: { Accept: "application/json" },
  });

  if (!response.ok) {
    return null;
  }

  const data = await response.json();
  return parseFeatureFlags(data);
}

/**
 * Hook to fetch feature flags from backend
 *
 * Features:
 * - Enterprise-safe: uses same-origin proxy
 * - Graceful degradation: isDegraded=true if fetch/parse fails
 * - Stale time: 1 minute (flags can change more frequently)
 * - Supports filtering/pagination via params
 */
export function useFeatureFlagsApi(params: UseFeatureFlagsParams = {}): UseFeatureFlagsResult {
  const { data, isLoading, error, refetch } = useQuery({
    queryKey: ["settings", "feature-flags", "api", params],
    queryFn: () => fetchFeatureFlags(params),
    retry: 1,
    staleTime: 60 * 1000, // 1 minute
    refetchOnWindowFocus: false,
    throwOnError: false,
  });

  return {
    flags: data ?? null,
    isDegraded: !!error || data === null,
    isLoading,
    error: error as Error | null,
    refetch: () => refetch(),
  };
}

// ============================================================================
// Model Versions Hook
// ============================================================================

export interface UseModelVersionsResult {
  models: ModelVersionsResponse | null;
  isDegraded: boolean;
  isLoading: boolean;
  error: Error | null;
  refetch: () => void;
}

async function fetchModelVersions(): Promise<ModelVersionsResponse | null> {
  const response = await fetch("/api/settings/model-versions", {
    method: "GET",
    headers: { Accept: "application/json" },
  });

  if (!response.ok) {
    return null;
  }

  const data = await response.json();
  return parseModelVersions(data);
}

/**
 * Hook to fetch model versions from backend
 *
 * Features:
 * - Enterprise-safe: uses same-origin proxy
 * - Graceful degradation: isDegraded=true if fetch/parse fails
 * - Stale time: 5 minutes (model versions rarely change)
 */
export function useModelVersionsApi(): UseModelVersionsResult {
  const { data, isLoading, error, refetch } = useQuery({
    queryKey: ["settings", "model-versions", "api"],
    queryFn: fetchModelVersions,
    retry: 1,
    staleTime: 5 * 60 * 1000, // 5 minutes
    refetchOnWindowFocus: false,
    throwOnError: false,
  });

  return {
    models: data ?? null,
    isDegraded: !!error || data === null,
    isLoading,
    error: error as Error | null,
    refetch: () => refetch(),
  };
}

// ============================================================================
// Combined Settings Hook (convenience)
// ============================================================================

export interface UseSettingsApiResult {
  summary: SettingsSummary | null;
  flags: FeatureFlagsResponse | null;
  models: ModelVersionsResponse | null;
  isSummaryDegraded: boolean;
  isFlagsDegraded: boolean;
  isModelsDegraded: boolean;
  isDegraded: boolean; // Any section degraded
  isLoading: boolean;
  refetchAll: () => void;
}

/**
 * Combined hook to fetch all settings data
 *
 * Convenience hook that combines summary, flags, and models.
 */
export function useSettingsApi(): UseSettingsApiResult {
  const summaryResult = useSettingsSummaryApi();
  const flagsResult = useFeatureFlagsApi();
  const modelsResult = useModelVersionsApi();

  const isLoading = summaryResult.isLoading || flagsResult.isLoading || modelsResult.isLoading;
  const isDegraded = summaryResult.isDegraded && flagsResult.isDegraded && modelsResult.isDegraded;

  return {
    summary: summaryResult.summary,
    flags: flagsResult.flags,
    models: modelsResult.models,
    isSummaryDegraded: summaryResult.isDegraded,
    isFlagsDegraded: flagsResult.isDegraded,
    isModelsDegraded: modelsResult.isDegraded,
    isDegraded,
    isLoading,
    refetchAll: () => {
      summaryResult.refetch();
      flagsResult.refetch();
      modelsResult.refetch();
    },
  };
}
