"use client";

import { useQuery } from "@tanstack/react-query";
import type { AdminOverviewData, AdminOverviewResponse } from "@/lib/types";

async function fetchAdminOverview(): Promise<AdminOverviewResponse | null> {
  const response = await fetch("/api/admin/overview", {
    method: "GET",
    headers: { Accept: "application/json" },
  });
  if (!response.ok) return null;
  return response.json();
}

export function useAdminOverview() {
  const { data, isLoading, error, refetch } = useQuery({
    queryKey: ["admin-overview"],
    queryFn: fetchAdminOverview,
    retry: 1,
    staleTime: 60_000,
    refetchOnWindowFocus: false,
    throwOnError: false,
  });

  return {
    data: (data?.data ?? null) as AdminOverviewData | null,
    isLoading,
    error: error as Error | null,
    refetch: () => refetch(),
  };
}
