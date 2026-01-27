"use client";

import { useQuery } from "@tanstack/react-query";
import type { CountriesList, CountriesListResponse } from "@/lib/types";

export interface UseFootballCountriesResult {
  data: CountriesList | null;
  generatedAt: string | null;
  cached: boolean;
  cacheAgeSeconds: number | null;
  isLoading: boolean;
  error: Error | null;
  refetch: () => void;
}

async function fetchFootballCountries(): Promise<CountriesListResponse | null> {
  const response = await fetch("/api/football/countries", {
    method: "GET",
    headers: { Accept: "application/json" },
  });

  if (!response.ok) {
    return null;
  }

  return response.json();
}

export function useFootballCountries(): UseFootballCountriesResult {
  const { data, isLoading, error, refetch } = useQuery({
    queryKey: ["football-countries"],
    queryFn: fetchFootballCountries,
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
