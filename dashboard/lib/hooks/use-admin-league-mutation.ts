"use client";

import { useMutation, useQueryClient } from "@tanstack/react-query";
import type { AdminLeaguePatchResponse } from "@/lib/types";

interface MutationParams {
  id: number;
  body: Record<string, unknown>;
}

async function patchAdminLeague({ id, body }: MutationParams): Promise<AdminLeaguePatchResponse> {
  const response = await fetch(`/api/admin/leagues/${id}`, {
    method: "PATCH",
    headers: {
      "Content-Type": "application/json",
      Accept: "application/json",
    },
    body: JSON.stringify(body),
  });

  const data = await response.json();

  if (!response.ok) {
    throw new Error(data?.error || `PATCH failed: ${response.status}`);
  }

  return data;
}

export function useAdminLeagueMutation() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: patchAdminLeague,
    onSuccess: (_data, variables) => {
      queryClient.invalidateQueries({ queryKey: ["admin-league", variables.id] });
      queryClient.invalidateQueries({ queryKey: ["admin-leagues"] });
      queryClient.invalidateQueries({ queryKey: ["admin-audit"] });
      queryClient.invalidateQueries({ queryKey: ["admin-overview"] });
    },
  });
}
