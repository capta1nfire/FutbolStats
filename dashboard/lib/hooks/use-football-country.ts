"use client";

import { useQuery } from "@tanstack/react-query";
import type { CountryDetail, CountryDetailResponse } from "@/lib/types";

export interface UseFootballCountryResult {
  data: CountryDetail | null;
  generatedAt: string | null;
  cached: boolean;
  cacheAgeSeconds: number | null;
  isLoading: boolean;
  error: Error | null;
  refetch: () => void;
}

async function fetchFootballCountry(country: string): Promise<CountryDetailResponse | null> {
  const response = await fetch(`/api/football/country/${encodeURIComponent(country)}`, {
    method: "GET",
    headers: { Accept: "application/json" },
  });

  if (!response.ok) {
    return null;
  }

  return response.json();
}

export function useFootballCountry(country: string | null): UseFootballCountryResult {
  const { data, isLoading, error, refetch } = useQuery({
    queryKey: ["football-country", country],
    queryFn: () => fetchFootballCountry(country!),
    enabled: !!country,
    retry: 1,
    staleTime: 30_000,
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
