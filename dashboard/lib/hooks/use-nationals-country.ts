"use client";

import { useQuery } from "@tanstack/react-query";
import type { NationalsCountryDetail, NationalsCountryDetailResponse } from "@/lib/types";

export interface UseNationalsCountryResult {
  data: NationalsCountryDetail | null;
  isLoading: boolean;
  error: Error | null;
  refetch: () => void;
}

async function fetchNationalsCountry(country: string): Promise<NationalsCountryDetailResponse | null> {
  const response = await fetch(`/api/football/nationals/country/${encodeURIComponent(country)}`, {
    method: "GET",
    headers: { Accept: "application/json" },
  });

  if (!response.ok) {
    return null;
  }

  return response.json();
}

export function useNationalsCountry(country: string): UseNationalsCountryResult {
  const { data, isLoading, error, refetch } = useQuery({
    queryKey: ["nationals-country", country],
    queryFn: () => fetchNationalsCountry(country),
    enabled: !!country,
    retry: 1,
    staleTime: 60_000,
    refetchOnWindowFocus: false,
    throwOnError: false,
  });

  return {
    data: data?.data ?? null,
    isLoading,
    error: error as Error | null,
    refetch: () => refetch(),
  };
}
