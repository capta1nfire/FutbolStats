"use client";

import { useQuery } from "@tanstack/react-query";
import type { ManagersViewData } from "@/lib/types/squad";

interface ManagersViewResponse {
  generated_at: string;
  cached: boolean;
  cache_age_seconds: number | null;
  data: ManagersViewData;
}

async function fetchManagersView(leagueId?: number): Promise<ManagersViewData> {
  const params = new URLSearchParams({ view: "managers" });
  if (leagueId) params.set("league_id", String(leagueId));
  const res = await fetch(`/api/football/players-managers?${params.toString()}`);
  if (!res.ok) throw new Error(`Failed to fetch managers view: ${res.status}`);
  const json: ManagersViewResponse = await res.json();
  return json.data;
}

export function useManagersView(leagueId?: number, enabled = true) {
  return useQuery({
    queryKey: ["managers-view", leagueId ?? null],
    queryFn: () => fetchManagersView(leagueId),
    enabled,
    staleTime: 2 * 60 * 1000,
    gcTime: 10 * 60 * 1000,
  });
}
