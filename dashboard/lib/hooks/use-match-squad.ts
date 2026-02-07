"use client";

import { useQuery } from "@tanstack/react-query";
import type { MatchSquadData } from "@/lib/types/squad";

interface MatchSquadResponse {
  generated_at: string;
  cached: boolean;
  cache_age_seconds: number | null;
  data: MatchSquadData;
}

async function fetchMatchSquad(matchId: number): Promise<MatchSquadData> {
  const res = await fetch(`/api/football/match/${matchId}/squad`);
  if (!res.ok) throw new Error(`Failed to fetch match squad: ${res.status}`);
  const json: MatchSquadResponse = await res.json();
  return json.data;
}

export function useMatchSquad(matchId: number | null, enabled = true) {
  return useQuery({
    queryKey: ["match-squad", matchId],
    queryFn: () => fetchMatchSquad(matchId!),
    enabled: enabled && matchId !== null && matchId > 0,
    staleTime: 2 * 60 * 1000,
    gcTime: 10 * 60 * 1000,
  });
}
