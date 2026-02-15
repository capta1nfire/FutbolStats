"use client";

import { useQuery } from "@tanstack/react-query";
import type { TeamSquadStatsData } from "@/lib/types/squad";

interface TeamSquadStatsResponse {
  generated_at: string;
  cached: boolean;
  cache_age_seconds: number | null;
  data: TeamSquadStatsData;
}

async function fetchTeamSquadStats(
  teamId: number,
  season: number | null
): Promise<TeamSquadStatsData> {
  const qs = new URLSearchParams();
  if (season) qs.set("season", String(season));

  const url = `/api/football/team/${teamId}/squad-stats${
    qs.toString() ? `?${qs.toString()}` : ""
  }`;
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Failed to fetch team squad stats: ${res.status}`);

  const json: TeamSquadStatsResponse = await res.json();
  return json.data;
}

export function useTeamSquadStats(teamId: number | null, season: number | null, enabled = true) {
  return useQuery({
    queryKey: ["team-squad-stats", teamId, season],
    queryFn: () => fetchTeamSquadStats(teamId!, season),
    enabled: enabled && teamId !== null && teamId > 0,
    staleTime: 2 * 60 * 1000,
    gcTime: 10 * 60 * 1000,
  });
}

