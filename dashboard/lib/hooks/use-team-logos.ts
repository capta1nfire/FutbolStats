/**
 * Team Logos Hook
 *
 * Fetches and caches team logos map for rendering shields.
 * Uses TanStack Query with long stale time since logos rarely change.
 */

import { useQuery } from "@tanstack/react-query";

/**
 * Response from /api/team-logos
 */
interface TeamLogosResponse {
  generated_at?: string;
  cached?: boolean;
  cache_age_seconds?: number;
  teams: Record<string, string>;
  count?: number;
  error?: string;
}

/**
 * Team logos map: team_name -> logo_url
 */
export type TeamLogosMap = Map<string, string>;

/**
 * Fetch team logos from proxy endpoint
 */
async function fetchTeamLogos(): Promise<TeamLogosMap> {
  const response = await fetch("/api/team-logos");

  if (!response.ok) {
    throw new Error(`Failed to fetch team logos: ${response.status}`);
  }

  const data: TeamLogosResponse = await response.json();

  // Convert object to Map for O(1) lookups
  const map = new Map<string, string>();
  if (data.teams && typeof data.teams === "object") {
    for (const [name, url] of Object.entries(data.teams)) {
      if (typeof url === "string" && url.length > 0) {
        map.set(name, url);
      }
    }
  }

  return map;
}

/**
 * Hook result
 */
export interface UseTeamLogosResult {
  /** Map of team_name -> logo_url */
  logos: TeamLogosMap;
  /** True while loading */
  isLoading: boolean;
  /** True if fetch failed */
  isDegraded: boolean;
  /** Get logo URL for a team name */
  getLogoUrl: (teamName: string) => string | null;
}

/**
 * Hook to fetch and use team logos
 *
 * @example
 * const { getLogoUrl } = useTeamLogos();
 * const logoUrl = getLogoUrl("River Plate");
 */
export function useTeamLogos(): UseTeamLogosResult {
  const { data, isLoading, isError } = useQuery<TeamLogosMap>({
    queryKey: ["team-logos"],
    queryFn: fetchTeamLogos,
    staleTime: 60 * 60 * 1000, // 1 hour
    gcTime: 24 * 60 * 60 * 1000, // 24 hours (formerly cacheTime)
    retry: 2,
    refetchOnWindowFocus: false,
  });

  const logos = data ?? new Map<string, string>();

  // Lookup function with case-insensitive fallback
  const getLogoUrl = (teamName: string): string | null => {
    // Exact match first
    const exact = logos.get(teamName);
    if (exact) return exact;

    // Case-insensitive fallback (slower, but handles edge cases)
    const lowerName = teamName.toLowerCase();
    for (const [name, url] of logos) {
      if (name.toLowerCase() === lowerName) {
        return url;
      }
    }

    return null;
  };

  return {
    logos,
    isLoading,
    isDegraded: isError,
    getLogoUrl,
  };
}
