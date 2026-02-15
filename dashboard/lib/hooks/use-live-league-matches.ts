"use client";

import { useQuery } from "@tanstack/react-query";

export interface LiveMatch {
  id: number;
  home: string;
  away: string;
  homeScore: number;
  awayScore: number;
  status: string;
  elapsed: number | null;
  elapsedExtra: number | null;
  kickoffISO: string;
}

async function fetchLiveMatches(leagueId: number): Promise<LiveMatch[]> {
  const params = new URLSearchParams({
    status: "LIVE",
    league_id: String(leagueId),
    limit: "50",
  });
  const res = await fetch(`/api/matches?${params}`);
  if (!res.ok) return [];
  const json = await res.json();

  const data = json.data ?? json;
  const matches = data?.matches;
  if (!Array.isArray(matches)) return [];

  return matches.map((m: Record<string, unknown>) => {
    const score = m.score as Record<string, unknown> | undefined;
    return {
      id: Number(m.id) || 0,
      home: String(m.home || ""),
      away: String(m.away || ""),
      homeScore: score?.home != null ? Number(score.home) : (m.home_goals != null ? Number(m.home_goals) : 0),
      awayScore: score?.away != null ? Number(score.away) : (m.away_goals != null ? Number(m.away_goals) : 0),
      status: String(m.status || "LIVE"),
      elapsed: m.elapsed != null ? Number(m.elapsed) : null,
      elapsedExtra: m.elapsed_extra != null ? Number(m.elapsed_extra) : null,
      kickoffISO: String(m.kickoff_iso || m.kickoffISO || ""),
    };
  });
}

/**
 * Fetch live (in-play) matches for a specific league.
 * Polls every 30s. Returns empty array when no live matches.
 */
export function useLiveLeagueMatches(leagueId: number | null) {
  return useQuery({
    queryKey: ["live-league-matches", leagueId],
    queryFn: () => fetchLiveMatches(leagueId!),
    enabled: leagueId != null,
    staleTime: 15_000,
    refetchInterval: 30_000,
    refetchOnWindowFocus: true,
    retry: 1,
  });
}
