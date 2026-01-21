/**
 * Settings Hooks
 *
 * TanStack Query hooks for settings data (read-only in Phase 0)
 */

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

/**
 * Fetch settings summary
 */
export function useSettingsSummary() {
  return useQuery<SettingsSummary, Error>({
    queryKey: ["settings", "summary"],
    queryFn: getSettingsSummaryMock,
    staleTime: 5 * 60 * 1000, // 5 minutes - settings don't change often
  });
}

/**
 * Fetch feature flags with optional filters
 */
export function useFeatureFlags(filters?: FeatureFlagsFilters) {
  return useQuery<FeatureFlag[], Error>({
    queryKey: ["settings", "featureFlags", filters],
    queryFn: () => getFeatureFlagsMock(filters),
  });
}

/**
 * Fetch users with optional filters
 */
export function useUsers(filters?: UsersFilters) {
  return useQuery<SettingsUser[], Error>({
    queryKey: ["settings", "users", filters],
    queryFn: () => getUsersMock(filters),
  });
}
