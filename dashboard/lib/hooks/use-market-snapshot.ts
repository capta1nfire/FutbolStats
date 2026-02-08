"use client";

import { useQuery } from "@tanstack/react-query";

export interface BookOdds {
  bookmaker: string;
  odds_home: number;
  odds_draw: number;
  odds_away: number;
  prob_home: number | null;
  prob_draw: number | null;
  prob_away: number | null;
  margin_pct: number | null;
  is_sharp: boolean;
  recorded_at: string;
}

export interface DispersionStats {
  home: { min: number; max: number; std: number };
  draw: { min: number; max: number; std: number };
  away: { min: number; max: number; std: number };
}

export interface MarketSnapshot {
  match_id: number;
  kickoff: string;
  n_books: number;
  consensus: {
    odds_home: number;
    odds_draw: number;
    odds_away: number;
    prob_home: number | null;
    prob_draw: number | null;
    prob_away: number | null;
    recorded_at: string;
  } | null;
  books: BookOdds[];
  dispersion: DispersionStats | null;
}

async function fetchMarketSnapshot(matchId: number): Promise<MarketSnapshot> {
  const response = await fetch(`/api/matches/${matchId}/market-snapshot`, {
    method: "GET",
    headers: { Accept: "application/json" },
  });

  if (!response.ok) {
    throw new Error(`Market snapshot fetch failed: ${response.status}`);
  }

  return response.json();
}

/**
 * Hook to fetch market snapshot for a match (lazy-loaded in Market tab).
 */
export function useMarketSnapshot(matchId: number | null) {
  return useQuery<MarketSnapshot>({
    queryKey: ["market-snapshot", matchId],
    queryFn: () => fetchMarketSnapshot(matchId!),
    enabled: matchId !== null,
    staleTime: 60_000,
    retry: 1,
  });
}
