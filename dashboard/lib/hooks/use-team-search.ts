"use client";

import { useQuery } from "@tanstack/react-query";

/**
 * Team search result
 */
export interface TeamSearchResult {
  team_id: number;
  external_id: number | null;
  name: string;
  display_name: string;  // COALESCE(override.short_name, wikidata.short_name, name) - always has value
  country: string;
  team_type: string;
  logo_url: string | null;
  stats: {
    total_matches: number;
    matches_25_26: number;
    leagues_played: number;
  };
}

interface TeamsSearchResponse {
  teams: TeamSearchResult[];
  pagination: {
    total: number;
    limit: number;
    offset: number;
    has_more: boolean;
  };
}

/**
 * Fetch teams with optional search
 */
async function fetchTeamsSearch(
  search?: string,
  limit: number = 10
): Promise<TeamsSearchResponse> {
  const params = new URLSearchParams();
  if (search?.trim()) {
    params.set("search", search.trim());
  }
  params.set("limit", limit.toString());

  const url = `/api/admin/teams${params.toString() ? `?${params}` : ""}`;
  const res = await fetch(url);

  if (!res.ok) {
    throw new Error(`Search failed: ${res.status}`);
  }

  const json = await res.json();
  return json.data || { teams: [], pagination: { total: 0, limit, offset: 0, has_more: false } };
}

/**
 * Hook for team search
 *
 * Usage:
 * const { data, isLoading } = useTeamSearch(searchQuery);
 */
export function useTeamSearch(search: string | undefined, enabled: boolean = true) {
  return useQuery({
    queryKey: ["teams", "search", search],
    queryFn: () => fetchTeamsSearch(search, 10),
    enabled: enabled && !!search?.trim() && search.trim().length >= 2,
    staleTime: 30 * 1000, // 30 seconds
    gcTime: 5 * 60 * 1000, // 5 minutes
  });
}
