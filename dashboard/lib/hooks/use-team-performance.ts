"use client";

import { useQuery } from "@tanstack/react-query";
import type { TeamPerformanceData } from "@/lib/types/performance";

interface TeamPerformanceResponse {
  generated_at: string;
  cached: boolean;
  cache_age_seconds: number | null;
  data: TeamPerformanceData;
}

async function fetchTeamPerformance(
  teamId: number,
  leagueId: number,
  season: number | null
): Promise<TeamPerformanceData> {
  const qs = new URLSearchParams({ league_id: String(leagueId) });
  if (season) qs.set("season", String(season));

  const url = `/api/football/team/${teamId}/performance?${qs.toString()}`;
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Failed to fetch team performance: ${res.status}`);

  const json: TeamPerformanceResponse = await res.json();
  return json.data;
}

export function useTeamPerformance(
  teamId: number | null,
  leagueId: number | null,
  season: number | null,
  enabled = true
) {
  return useQuery({
    queryKey: ["team-performance", teamId, leagueId, season],
    queryFn: () => fetchTeamPerformance(teamId!, leagueId!, season),
    enabled: enabled && teamId !== null && teamId > 0 && leagueId !== null && leagueId > 0,
    staleTime: 2 * 60 * 1000,
    gcTime: 10 * 60 * 1000,
  });
}
