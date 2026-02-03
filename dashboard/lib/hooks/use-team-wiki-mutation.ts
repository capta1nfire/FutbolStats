"use client";

import { useMutation, useQueryClient } from "@tanstack/react-query";
import { patchTeam, type TeamPatchRequest, type TeamPatchResponse } from "@/lib/api/teams";

interface MutationParams {
  teamId: number;
  data: TeamPatchRequest;
}

export function useTeamWikiMutation() {
  const queryClient = useQueryClient();

  return useMutation<TeamPatchResponse, Error & { isNotSupported?: boolean }, MutationParams>({
    mutationFn: ({ teamId, data }) => patchTeam(teamId, data),
    onSuccess: (_data, variables) => {
      // Invalidate team detail to refresh derived wiki fields
      queryClient.invalidateQueries({ queryKey: ["football-team", variables.teamId] });
    },
  });
}
