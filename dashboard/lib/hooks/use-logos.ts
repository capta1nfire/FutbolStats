/**
 * Logos 3D Hooks
 *
 * TanStack Query hooks for logos generation and review.
 * Includes polling for batch progress and mutations for actions.
 */

"use client";

import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import {
  fetchLeaguesForGeneration,
  fetchBatchStatus,
  fetchLeagueReview,
  fetchTeamLogoStatus,
  startBatchJob,
  pauseBatch,
  resumeBatch,
  cancelBatch,
  processBatchTeams,
  reviewTeamLogo,
  approveLeagueLogos,
} from "@/lib/api/logos";
import type {
  GenerateBatchRequest,
  ReviewTeamRequest,
  ReviewLeagueRequest,
} from "@/lib/types/logos";

// =============================================================================
// Query Keys
// =============================================================================

export const logosKeys = {
  all: ["logos"] as const,
  leagues: () => [...logosKeys.all, "leagues"] as const,
  batch: (batchId: string) => [...logosKeys.all, "batch", batchId] as const,
  review: (leagueId: number, filter?: string) =>
    [...logosKeys.all, "review", leagueId, filter] as const,
  teamStatus: (teamId: number) =>
    [...logosKeys.all, "team", teamId] as const,
};

// =============================================================================
// Leagues Hook
// =============================================================================

/**
 * Fetch leagues available for logo generation
 */
export function useLogosLeagues() {
  return useQuery({
    queryKey: logosKeys.leagues(),
    queryFn: fetchLeaguesForGeneration,
    staleTime: 60 * 1000, // 1 minute
    refetchOnWindowFocus: false,
  });
}

// =============================================================================
// Batch Status Hook (with polling)
// =============================================================================

/**
 * Fetch and poll batch job status
 *
 * Polls every 5 seconds while the batch is running.
 * Stops polling when completed, cancelled, or errored.
 */
export function useLogosBatchStatus(batchId: string | null) {
  return useQuery({
    queryKey: logosKeys.batch(batchId || ""),
    queryFn: () => fetchBatchStatus(batchId!),
    enabled: !!batchId,
    staleTime: 0, // Always refetch
    refetchInterval: (query) => {
      const status = query.state.data?.status;
      // Poll every 5s while running
      if (status === "running") return 5000;
      // Stop polling when done
      return false;
    },
  });
}

// =============================================================================
// Review Hook
// =============================================================================

/**
 * Fetch teams for review in a league
 */
export function useLogosReview(
  leagueId: number | null,
  statusFilter?: string
) {
  return useQuery({
    queryKey: logosKeys.review(leagueId || 0, statusFilter),
    queryFn: () => fetchLeagueReview(leagueId!, statusFilter),
    enabled: !!leagueId,
    staleTime: 30 * 1000, // 30 seconds
  });
}

// =============================================================================
// Team Status Hook
// =============================================================================

/**
 * Fetch detailed status for a single team
 */
export function useTeamLogoStatus(teamId: number | null) {
  return useQuery({
    queryKey: logosKeys.teamStatus(teamId || 0),
    queryFn: () => fetchTeamLogoStatus(teamId!),
    enabled: !!teamId,
    staleTime: 30 * 1000,
  });
}

// =============================================================================
// Mutations
// =============================================================================

/**
 * Start a new batch generation job
 */
export function useStartBatch() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({
      leagueId,
      request,
    }: {
      leagueId: number;
      request: GenerateBatchRequest;
    }) => startBatchJob(leagueId, request),
    onSuccess: () => {
      // Invalidate leagues to refresh counts
      queryClient.invalidateQueries({ queryKey: logosKeys.leagues() });
    },
  });
}

/**
 * Pause a running batch job
 */
export function usePauseBatch() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: pauseBatch,
    onSuccess: (_, batchId) => {
      queryClient.invalidateQueries({ queryKey: logosKeys.batch(batchId) });
    },
  });
}

/**
 * Resume a paused batch job
 */
export function useResumeBatch() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: resumeBatch,
    onSuccess: (_, batchId) => {
      queryClient.invalidateQueries({ queryKey: logosKeys.batch(batchId) });
    },
  });
}

/**
 * Cancel a batch job
 */
export function useCancelBatch() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: cancelBatch,
    onSuccess: (_, batchId) => {
      queryClient.invalidateQueries({ queryKey: logosKeys.batch(batchId) });
      queryClient.invalidateQueries({ queryKey: logosKeys.leagues() });
    },
  });
}

/**
 * Process next batch of teams
 */
export function useProcessBatch() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({
      batchId,
      batchSize = 5,
    }: {
      batchId: string;
      batchSize?: number;
    }) => processBatchTeams(batchId, batchSize),
    onSuccess: (_, { batchId }) => {
      queryClient.invalidateQueries({ queryKey: logosKeys.batch(batchId) });
    },
  });
}

/**
 * Review a single team's logos
 */
export function useReviewTeamLogo() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({
      teamId,
      request,
    }: {
      teamId: number;
      request: ReviewTeamRequest;
    }) => reviewTeamLogo(teamId, request),
    onSuccess: (_, { teamId }) => {
      // Invalidate team status
      queryClient.invalidateQueries({ queryKey: logosKeys.teamStatus(teamId) });
      // Invalidate all review queries (league context unknown)
      queryClient.invalidateQueries({
        queryKey: [...logosKeys.all, "review"],
      });
    },
  });
}

/**
 * Bulk approve or reject all logos in a league
 */
export function useApproveLeague() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({
      leagueId,
      request,
    }: {
      leagueId: number;
      request: ReviewLeagueRequest;
    }) => approveLeagueLogos(leagueId, request),
    onSuccess: (_, { leagueId }) => {
      // Invalidate review for this league
      queryClient.invalidateQueries({
        queryKey: [...logosKeys.all, "review", leagueId],
      });
      // Invalidate leagues to refresh counts
      queryClient.invalidateQueries({ queryKey: logosKeys.leagues() });
    },
  });
}

// =============================================================================
// localStorage Helper for Active Batch (Kimi recommendation)
// =============================================================================

const ACTIVE_BATCH_KEY = "logos_active_batch";

/**
 * Get active batch ID from localStorage
 */
export function getStoredActiveBatch(): string | null {
  if (typeof window === "undefined") return null;
  return localStorage.getItem(ACTIVE_BATCH_KEY);
}

/**
 * Store active batch ID in localStorage
 */
export function setStoredActiveBatch(batchId: string | null): void {
  if (typeof window === "undefined") return;
  if (batchId) {
    localStorage.setItem(ACTIVE_BATCH_KEY, batchId);
  } else {
    localStorage.removeItem(ACTIVE_BATCH_KEY);
  }
}

/**
 * Clear active batch from localStorage
 */
export function clearStoredActiveBatch(): void {
  if (typeof window === "undefined") return;
  localStorage.removeItem(ACTIVE_BATCH_KEY);
}
