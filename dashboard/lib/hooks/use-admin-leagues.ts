"use client";

import { useQuery } from "@tanstack/react-query";
import type { AdminLeaguesList, AdminLeaguesListResponse, AdminLeaguesFilters } from "@/lib/types";

async function fetchAdminLeagues(filters: AdminLeaguesFilters): Promise<AdminLeaguesListResponse | null> {
  const params = new URLSearchParams();
  if (filters.search) params.set("search", filters.search);
  if (filters.country) params.set("country", filters.country);
  if (filters.kind) params.set("kind", filters.kind);
  if (filters.is_active) params.set("is_active", filters.is_active);
  if (filters.source) params.set("source", filters.source);

  const qs = params.toString();
  const url = `/api/admin/leagues${qs ? `?${qs}` : ""}`;

  const response = await fetch(url, {
    method: "GET",
    headers: { Accept: "application/json" },
  });
  if (!response.ok) return null;
  return response.json();
}

export function useAdminLeagues(filters: AdminLeaguesFilters = {}) {
  const { data, isLoading, error, refetch } = useQuery({
    queryKey: ["admin-leagues", filters],
    queryFn: () => fetchAdminLeagues(filters),
    retry: 1,
    staleTime: 30_000,
    refetchOnWindowFocus: false,
    throwOnError: false,
  });

  return {
    data: (data?.data ?? null) as AdminLeaguesList | null,
    isLoading,
    error: error as Error | null,
    refetch: () => refetch(),
  };
}
