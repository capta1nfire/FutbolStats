/**
 * Feature Coverage hook using TanStack Query
 *
 * Fetches feature coverage matrix data from backend.
 * Data changes infrequently - uses longer stale time.
 */

import { useQuery } from "@tanstack/react-query";

/**
 * Window definition
 */
export interface FeatureCoverageWindow {
  key: string; // "23/24", "24/25"
  from: string; // ISO date
  to: string; // ISO date
}

/**
 * Tier definition
 */
export interface FeatureCoverageTier {
  id: string; // "tier1", "tier1b", etc.
  label: string; // "[PROD] Tier 1 - Core"
  badge: string; // "PROD", "TITAN"
}

/**
 * Feature definition
 */
export interface FeatureCoverageFeature {
  key: string; // "home_goals_scored_avg"
  tier_id: string; // "tier1"
  badge: string; // "PROD"
  source: string; // "public.matches"
}

/**
 * League definition
 */
export interface FeatureCoverageLeague {
  league_id: number;
  name: string;
}

/**
 * Coverage cell data
 */
export interface FeatureCoverageCell {
  pct: number; // 0-100
  n: number; // non-null count
}

/**
 * League summary per window
 *
 * matches_total_ft: count of FT matches (for tier1 PROD features)
 * matches_total_titan: count of feature_matrix rows (for tier1b/1c/1d TITAN features)
 */
export interface FeatureCoverageLeagueSummary {
  matches_total_ft: number;
  matches_total_titan: number;
  avg_pct: number;
}

/**
 * Full response from backend
 */
export interface FeatureCoverageResponse {
  generated_at: string;
  cached: boolean;
  cache_age_seconds: number | null;
  data: {
    windows: FeatureCoverageWindow[];
    tiers: FeatureCoverageTier[];
    features: FeatureCoverageFeature[];
    leagues: FeatureCoverageLeague[];
    league_summaries: Record<
      string,
      Record<string, FeatureCoverageLeagueSummary>
    >;
    coverage: Record<
      string,
      Record<string, Record<string, FeatureCoverageCell>>
    >;
  };
}

/**
 * Fetch feature coverage from API
 */
async function fetchFeatureCoverage(): Promise<FeatureCoverageResponse> {
  const response = await fetch("/api/feature-coverage");

  if (!response.ok) {
    throw new Error(`API error: ${response.status}`);
  }

  return response.json();
}

/**
 * Hook to fetch feature coverage data
 *
 * Uses longer stale time since this data changes infrequently.
 */
export function useFeatureCoverage() {
  return useQuery({
    queryKey: ["feature-coverage"],
    queryFn: fetchFeatureCoverage,
    staleTime: 15 * 60 * 1000, // 15 minutes
    gcTime: 30 * 60 * 1000, // 30 minutes (formerly cacheTime)
    retry: 1,
    refetchOnWindowFocus: false,
  });
}
