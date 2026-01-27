"use client";

import { useQuery } from "@tanstack/react-query";
import type { AdminAuditList, AdminAuditListResponse, AdminAuditFilters } from "@/lib/types";

async function fetchAdminAudit(filters: AdminAuditFilters): Promise<AdminAuditListResponse | null> {
  const params = new URLSearchParams();
  if (filters.entity_type) params.set("entity_type", filters.entity_type);
  if (filters.entity_id) params.set("entity_id", filters.entity_id);
  if (filters.limit) params.set("limit", filters.limit.toString());
  if (filters.offset) params.set("offset", filters.offset.toString());

  const qs = params.toString();
  const url = `/api/admin/audit${qs ? `?${qs}` : ""}`;

  const response = await fetch(url, {
    method: "GET",
    headers: { Accept: "application/json" },
  });
  if (!response.ok) return null;
  return response.json();
}

export function useAdminAudit(filters: AdminAuditFilters = {}) {
  const { data, isLoading, error, refetch } = useQuery({
    queryKey: ["admin-audit", filters],
    queryFn: () => fetchAdminAudit(filters),
    retry: 1,
    staleTime: 15_000,
    refetchOnWindowFocus: false,
    throwOnError: false,
  });

  return {
    data: (data?.data ?? null) as AdminAuditList | null,
    isLoading,
    error: error as Error | null,
    refetch: () => refetch(),
  };
}
