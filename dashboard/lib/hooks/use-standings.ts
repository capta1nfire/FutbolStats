"use client";

import { useQuery } from "@tanstack/react-query";
import { StandingEntry, StandingsResponse } from "@/lib/types";

/**
 * Adapt raw backend response to typed StandingEntry
 */
function adaptStanding(raw: unknown): StandingEntry | null {
  if (!raw || typeof raw !== "object") return null;
  const r = raw as Record<string, unknown>;

  const teamName = typeof r.team_name === "string" ? r.team_name : "Unknown";

  return {
    position: typeof r.position === "number" ? r.position : 0,
    teamId: typeof r.team_id === "number" ? r.team_id : 0,
    teamName,
    displayName: typeof r.display_name === "string" ? r.display_name : teamName,
    teamLogo: typeof r.team_logo === "string" ? r.team_logo : null,
    points: typeof r.points === "number" ? r.points : 0,
    played: typeof r.played === "number" ? r.played : 0,
    won: typeof r.won === "number" ? r.won : 0,
    drawn: typeof r.drawn === "number" ? r.drawn : 0,
    lost: typeof r.lost === "number" ? r.lost : 0,
    goalsFor: typeof r.goals_for === "number" ? r.goals_for : 0,
    goalsAgainst: typeof r.goals_against === "number" ? r.goals_against : 0,
    goalDiff: typeof r.goal_diff === "number" ? r.goal_diff : 0,
    form: typeof r.form === "string" ? r.form : undefined,
    description: typeof r.description === "string" ? r.description : null,
  };
}

/**
 * Fetch standings from API
 */
async function fetchStandings(leagueId: number, group?: string, season?: number): Promise<StandingsResponse> {
  const params = new URLSearchParams();
  if (group) params.set("group", group);
  if (season) params.set("season", String(season));
  const qs = params.toString();
  const url = `/api/standings/${leagueId}${qs ? `?${qs}` : ""}`;
  const response = await fetch(url);

  if (!response.ok) {
    if (response.status === 404) {
      throw new Error("Standings not found for this league");
    }
    throw new Error(`Failed to fetch standings: ${response.status}`);
  }

  const data = await response.json();

  // Adapt response
  const standings: StandingEntry[] = [];
  if (Array.isArray(data.standings)) {
    for (const raw of data.standings) {
      const adapted = adaptStanding(raw);
      if (adapted) standings.push(adapted);
    }
  }

  return {
    leagueId: data.league_id ?? leagueId,
    season: data.season ?? new Date().getFullYear(),
    standings,
    source: data.source ?? "unknown",
    isPlaceholder: data.is_placeholder ?? false,
    isCalculated: data.is_calculated ?? false,
    meta: data.meta ?? undefined,
    reclasificacion: data.reclasificacion ?? null,
    descenso: data.descenso ?? null,
  };
}

/**
 * Hook to fetch league standings
 *
 * @param leagueId - League ID to fetch standings for
 * @param options.enabled - Whether to enable the query (default: true)
 * @param options.group - Specific group to fetch (for multi-group leagues)
 */
export function useStandings(
  leagueId: number | null,
  options: { enabled?: boolean; group?: string; season?: number } = {},
) {
  const { enabled = true, group, season } = options;
  return useQuery({
    queryKey: ["standings", leagueId, group ?? null, season ?? null],
    queryFn: () => fetchStandings(leagueId!, group, season),
    enabled: enabled && leagueId !== null && leagueId > 0,
    staleTime: 5 * 60 * 1000, // 5 minutes
    gcTime: 30 * 60 * 1000, // 30 minutes
  });
}
