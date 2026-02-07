"use client";

import { useQuery } from "@tanstack/react-query";
import type { TeamSquadData } from "@/lib/types/squad";

interface TeamSquadResponse {
  generated_at: string;
  cached: boolean;
  cache_age_seconds: number | null;
  data: TeamSquadData;
}

async function fetchTeamSquad(teamId: number): Promise<TeamSquadData> {
  const res = await fetch(`/api/football/team/${teamId}/squad`);
  if (!res.ok) throw new Error(`Failed to fetch team squad: ${res.status}`);
  const json: TeamSquadResponse = await res.json();
  return json.data;
}

export function useTeamSquad(teamId: number | null, enabled = true) {
  return useQuery({
    queryKey: ["team-squad", teamId],
    queryFn: () => fetchTeamSquad(teamId!),
    enabled: enabled && teamId !== null && teamId > 0,
    staleTime: 2 * 60 * 1000,
    gcTime: 10 * 60 * 1000,
  });
}
