"use client";

import { useQuery } from "@tanstack/react-query";
import type { TournamentsList, TournamentsListResponse } from "@/lib/types";

export interface UseFootballTournamentsResult {
  data: TournamentsList | null;
  generatedAt: string | null;
  cached: boolean;
  cacheAgeSeconds: number | null;
  isLoading: boolean;
  error: Error | null;
  refetch: () => void;
}

async function fetchFootballTournaments(): Promise<TournamentsListResponse | null> {
  const response = await fetch("/api/football/tournaments", {
    method: "GET",
    headers: { Accept: "application/json" },
  });

  if (!response.ok) {
    return null;
  }

  return response.json();
}

export function useFootballTournaments(): UseFootballTournamentsResult {
  const { data, isLoading, error, refetch } = useQuery({
    queryKey: ["football-tournaments"],
    queryFn: fetchFootballTournaments,
    retry: 1,
    staleTime: 60_000,
    refetchOnWindowFocus: false,
    throwOnError: false,
  });

  return {
    data: data?.data ?? null,
    generatedAt: data?.generated_at ?? null,
    cached: data?.cached ?? false,
    cacheAgeSeconds: data?.cache_age_seconds ?? null,
    isLoading,
    error: error as Error | null,
    refetch: () => refetch(),
  };
}
