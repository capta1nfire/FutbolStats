"use client";

import { useQuery } from "@tanstack/react-query";
import type { NationalsCountriesList, NationalsCountriesListResponse } from "@/lib/types";

export interface UseNationalsCountriesResult {
  data: NationalsCountriesList | null;
  isLoading: boolean;
  error: Error | null;
  refetch: () => void;
}

async function fetchNationalsCountries(): Promise<NationalsCountriesListResponse | null> {
  const response = await fetch("/api/football/nationals/countries", {
    method: "GET",
    headers: { Accept: "application/json" },
  });

  if (!response.ok) {
    return null;
  }

  return response.json();
}

export function useNationalsCountries(): UseNationalsCountriesResult {
  const { data, isLoading, error, refetch } = useQuery({
    queryKey: ["nationals-countries"],
    queryFn: fetchNationalsCountries,
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
