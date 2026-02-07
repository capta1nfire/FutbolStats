"use client";

import { useQuery } from "@tanstack/react-query";
import type { PlayersViewData } from "@/lib/types/squad";

interface PlayersViewResponse {
  generated_at: string;
  cached: boolean;
  cache_age_seconds: number | null;
  data: PlayersViewData;
}

async function fetchPlayersView(leagueId?: number): Promise<PlayersViewData> {
  const params = new URLSearchParams({ view: "injuries", limit: "1000" });
  if (leagueId) params.set("league_id", String(leagueId));
  const res = await fetch(`/api/football/players-managers?${params.toString()}`);
  if (!res.ok) throw new Error(`Failed to fetch players view: ${res.status}`);
  const json: PlayersViewResponse = await res.json();
  return json.data;
}

export function usePlayersView(leagueId?: number, enabled = true) {
  return useQuery({
    queryKey: ["players-view", leagueId ?? null],
    queryFn: () => fetchPlayersView(leagueId),
    enabled,
    staleTime: 2 * 60 * 1000,
    gcTime: 10 * 60 * 1000,
  });
}
