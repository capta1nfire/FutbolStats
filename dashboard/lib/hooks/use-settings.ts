/**
 * Settings Hooks
 *
 * TanStack Query hooks for settings data (read-only in Phase 0)
 * Now with real backend data support + mock fallback
 */

"use client";

import { useQuery } from "@tanstack/react-query";
import {
  SettingsSummary,
  FeatureFlag,
  SettingsUser,
  FeatureFlagsFilters,
  UsersFilters,
} from "@/lib/types";
import {
  getSettingsSummaryMock,
  getFeatureFlagsMock,
  getUsersMock,
} from "@/lib/mocks";
import {
  parseSettingsSummary,
  parseFeatureFlags,
  parseModelVersions,
  SettingsSummary as ApiSettingsSummary,
  FeatureFlagsResponse,
  ModelVersionsResponse,
} from "@/lib/api/settings";

// ============================================================================
// Backend API fetchers
// ============================================================================

async function fetchSettingsSummaryApi(): Promise<ApiSettingsSummary | null> {
  try {
    const response = await fetch("/api/settings/summary", {
      method: "GET",
      headers: { Accept: "application/json" },
    });
    if (!response.ok) return null;
    const data = await response.json();
    return parseSettingsSummary(data);
  } catch {
    return null;
  }
}

async function fetchFeatureFlagsApi(): Promise<FeatureFlagsResponse | null> {
  try {
    const response = await fetch("/api/settings/feature-flags", {
      method: "GET",
      headers: { Accept: "application/json" },
    });
    if (!response.ok) return null;
    const data = await response.json();
    return parseFeatureFlags(data);
  } catch {
    return null;
  }
}

async function fetchModelVersionsApi(): Promise<ModelVersionsResponse | null> {
  try {
    const response = await fetch("/api/settings/model-versions", {
      method: "GET",
      headers: { Accept: "application/json" },
    });
    if (!response.ok) return null;
    const data = await response.json();
    return parseModelVersions(data);
  } catch {
    return null;
  }
}

// ============================================================================
// Mappers: API data -> Mock structure (for component compatibility)
// ============================================================================

/**
 * Map backend feature flags to component-expected format
 */
function mapApiFeatureFlags(apiFlags: FeatureFlagsResponse): FeatureFlag[] {
  return apiFlags.flags.map((f) => ({
    id: f.key,
    name: f.key
      .split("_")
      .map((w) => w.charAt(0).toUpperCase() + w.slice(1).toLowerCase())
      .join(" "),
    description: f.description,
    enabled: f.enabled ?? false,
    updatedAt: undefined, // Backend doesn't provide per-flag update times
  }));
}

/**
 * Build SettingsSummary from API data with mock fallback
 */
async function fetchSettingsSummaryWithFallback(): Promise<{
  summary: SettingsSummary;
  isApiDegraded: boolean;
}> {
  // Fetch all three endpoints in parallel
  const [apiSummary, apiFlags, apiModels] = await Promise.all([
    fetchSettingsSummaryApi(),
    fetchFeatureFlagsApi(),
    fetchModelVersionsApi(),
  ]);

  // If all API calls failed, use mock
  if (!apiSummary && !apiFlags && !apiModels) {
    const mockData = await getSettingsSummaryMock();
    return { summary: mockData, isApiDegraded: true };
  }

  // Get mock as base for fields not available from API
  const mockBase = await getSettingsSummaryMock();

  // Build summary from API data + mock fallback
  const summary: SettingsSummary = {
    lastUpdated: apiSummary?.generatedAt ?? mockBase.lastUpdated,
    environment: mockBase.environment, // Not exposed by API
    timezoneDisplay: mockBase.timezoneDisplay, // Not exposed by API
    narrativeProvider: mockBase.narrativeProvider, // Not exposed by API
    apiFootballKeyStatus: apiSummary?.integrations.find((i) => i.key === "rapidapi")?.configured
      ? "configured"
      : mockBase.apiFootballKeyStatus,
    modelVersions: apiModels
      ? {
          modelA:
            apiModels.models.find((m) => m.name === "baseline")?.version ??
            mockBase.modelVersions.modelA,
          shadow:
            apiModels.models.find((m) => m.name === "shadow_version")?.version ??
            mockBase.modelVersions.shadow,
          updatedAt:
            apiModels.models.find((m) => m.name === "baseline")?.updatedAt ??
            mockBase.modelVersions.updatedAt,
        }
      : mockBase.modelVersions,
    featureFlags: apiFlags ? mapApiFeatureFlags(apiFlags) : mockBase.featureFlags,
    users: mockBase.users, // Users not exposed by API
  };

  return {
    summary,
    isApiDegraded: !apiSummary && !apiFlags && !apiModels,
  };
}

// ============================================================================
// Result types with degraded indicator
// ============================================================================

export interface UseSettingsSummaryResult {
  data: SettingsSummary | undefined;
  isLoading: boolean;
  error: Error | null;
  isApiDegraded: boolean;
  refetch: () => void;
}

export interface UseFeatureFlagsResult {
  data: FeatureFlag[];
  isLoading: boolean;
  error: Error | null;
  isApiDegraded: boolean;
  refetch: () => void;
}

// ============================================================================
// Hooks
// ============================================================================

/**
 * Fetch settings summary - tries API first, falls back to mock
 */
export function useSettingsSummary(): UseSettingsSummaryResult {
  const query = useQuery({
    queryKey: ["settings", "summary", "hybrid"],
    queryFn: fetchSettingsSummaryWithFallback,
    staleTime: 5 * 60 * 1000, // 5 minutes
    throwOnError: false,
  });

  return {
    data: query.data?.summary,
    isLoading: query.isLoading,
    error: query.error as Error | null,
    isApiDegraded: query.data?.isApiDegraded ?? true,
    refetch: () => query.refetch(),
  };
}

/**
 * Fetch feature flags - tries API first, falls back to mock
 *
 * TODO: When backend implements server-side filtering for q/enabled params,
 * move filtering to server-side to avoid fetching full list for large datasets.
 * Current client-side filtering is acceptable for <100 flags.
 */
export function useFeatureFlags(filters?: FeatureFlagsFilters): UseFeatureFlagsResult {
  const query = useQuery({
    queryKey: ["settings", "featureFlags", "hybrid", filters],
    queryFn: async () => {
      const apiFlags = await fetchFeatureFlagsApi();
      if (apiFlags) {
        let flags = mapApiFeatureFlags(apiFlags);

        // Apply client-side filtering (see TODO above)
        if (filters?.search) {
          const search = filters.search.toLowerCase();
          flags = flags.filter(
            (f) =>
              f.name.toLowerCase().includes(search) ||
              f.description?.toLowerCase().includes(search)
          );
        }
        if (filters?.enabled !== undefined) {
          flags = flags.filter((f) => f.enabled === filters.enabled);
        }

        return { flags, isApiDegraded: false };
      }

      // Fallback to mock
      const mockFlags = await getFeatureFlagsMock(filters);
      return { flags: mockFlags, isApiDegraded: true };
    },
    staleTime: 60 * 1000, // 1 minute
    throwOnError: false,
  });

  return {
    data: query.data?.flags ?? [],
    isLoading: query.isLoading,
    error: query.error as Error | null,
    isApiDegraded: query.data?.isApiDegraded ?? true,
    refetch: () => query.refetch(),
  };
}

/**
 * Fetch users with optional filters
 * Note: Users are not exposed by API, always uses mock
 */
export function useUsers(filters?: UsersFilters) {
  return useQuery<SettingsUser[], Error>({
    queryKey: ["settings", "users", filters],
    queryFn: () => getUsersMock(filters),
  });
}
