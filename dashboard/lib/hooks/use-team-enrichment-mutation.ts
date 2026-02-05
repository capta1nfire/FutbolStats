"use client";

import { useMutation, useQueryClient } from "@tanstack/react-query";
import {
  putTeamEnrichment,
  deleteTeamEnrichment,
  type TeamEnrichmentPutRequest,
  type TeamEnrichmentPutResponse,
  type TeamEnrichmentDeleteResponse,
} from "@/lib/api/team-enrichment";

interface PutMutationParams {
  teamId: number;
  data: TeamEnrichmentPutRequest;
}

/**
 * Hook for PUT team enrichment override
 *
 * Upserts manual override values. Invalidates team query on success.
 */
export function useTeamEnrichmentPutMutation() {
  const queryClient = useQueryClient();

  return useMutation<TeamEnrichmentPutResponse, Error, PutMutationParams>({
    mutationFn: ({ teamId, data }) => putTeamEnrichment(teamId, data),
    onSuccess: (_data, variables) => {
      // Invalidate team detail to refresh enrichment fields
      queryClient.invalidateQueries({ queryKey: ["football-team", variables.teamId] });
      // Invalidate standings to reflect updated display_name
      queryClient.invalidateQueries({ queryKey: ["standings"] });
    },
  });
}

interface DeleteMutationParams {
  teamId: number;
}

/**
 * Hook for DELETE team enrichment override
 *
 * Removes all override values. Invalidates team query on success.
 */
export function useTeamEnrichmentDeleteMutation() {
  const queryClient = useQueryClient();

  return useMutation<TeamEnrichmentDeleteResponse, Error, DeleteMutationParams>({
    mutationFn: ({ teamId }) => deleteTeamEnrichment(teamId),
    onSuccess: (_data, variables) => {
      // Invalidate team detail to refresh enrichment fields
      queryClient.invalidateQueries({ queryKey: ["football-team", variables.teamId] });
      // Invalidate standings to reflect updated display_name
      queryClient.invalidateQueries({ queryKey: ["standings"] });
    },
  });
}
