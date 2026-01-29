/**
 * Logos 3D API Client
 *
 * Proxy calls to backend /dashboard/logos/* endpoints.
 * Uses Next.js API routes for auth handling and CORS.
 */

import type {
  LeagueForGeneration,
  LogoBatchJob,
  TeamLogoReview,
  GenerateBatchRequest,
  GenerateBatchResponse,
  ReviewTeamRequest,
  ReviewLeagueRequest,
  CostEstimate,
  GenerationMode,
  IAModel,
} from "@/lib/types/logos";

const API_BASE = "/api/logos";

// =============================================================================
// Leagues
// =============================================================================

/**
 * Fetch leagues available for logo generation
 */
export async function fetchLeaguesForGeneration(): Promise<LeagueForGeneration[]> {
  const res = await fetch(`${API_BASE}/leagues`);

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `Failed to fetch leagues: ${res.status}`);
  }

  const data = await res.json();
  return (data.leagues || []).map(parseLeague);
}

// =============================================================================
// Batch Jobs
// =============================================================================

/**
 * Start a batch generation job for a league
 */
export async function startBatchJob(
  leagueId: number,
  request: GenerateBatchRequest
): Promise<GenerateBatchResponse> {
  const res = await fetch(`${API_BASE}/generate/league/${leagueId}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(request),
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `Failed to start batch: ${res.status}`);
  }

  return res.json();
}

/**
 * Fetch status of a batch job
 */
export async function fetchBatchStatus(batchId: string): Promise<LogoBatchJob> {
  const res = await fetch(`${API_BASE}/batch/${batchId}`);

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `Failed to fetch batch status: ${res.status}`);
  }

  return parseBatchJob(await res.json());
}

/**
 * Pause a running batch job
 */
export async function pauseBatch(batchId: string): Promise<void> {
  const res = await fetch(`${API_BASE}/batch/${batchId}/pause`, {
    method: "POST",
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || "Failed to pause batch");
  }
}

/**
 * Resume a paused batch job
 */
export async function resumeBatch(batchId: string): Promise<void> {
  const res = await fetch(`${API_BASE}/batch/${batchId}/resume`, {
    method: "POST",
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || "Failed to resume batch");
  }
}

/**
 * Cancel a batch job (cannot be resumed)
 */
export async function cancelBatch(batchId: string): Promise<void> {
  const res = await fetch(`${API_BASE}/batch/${batchId}/cancel`, {
    method: "POST",
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || "Failed to cancel batch");
  }
}

/**
 * Process next batch of teams (called repeatedly)
 */
export async function processBatchTeams(
  batchId: string,
  batchSize: number = 5
): Promise<{ processed: number; remaining: number; status: string }> {
  const res = await fetch(
    `${API_BASE}/generate/batch/${batchId}/process?batch_size=${batchSize}`,
    { method: "POST" }
  );

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || "Failed to process batch");
  }

  return res.json();
}

// =============================================================================
// Review
// =============================================================================

/**
 * Fetch teams for review in a league
 */
export async function fetchLeagueReview(
  leagueId: number,
  statusFilter?: string
): Promise<{ total: number; teams: TeamLogoReview[] }> {
  const params = new URLSearchParams();
  if (statusFilter) params.set("status_filter", statusFilter);

  const url = `${API_BASE}/review/league/${leagueId}${params.toString() ? `?${params}` : ""}`;
  const res = await fetch(url);

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `Failed to fetch review: ${res.status}`);
  }

  const data = await res.json();
  return {
    total: data.total || 0,
    teams: (data.teams || []).map(parseTeamLogo),
  };
}

/**
 * Review a single team's logos
 */
export async function reviewTeamLogo(
  teamId: number,
  request: ReviewTeamRequest
): Promise<void> {
  const res = await fetch(`${API_BASE}/review/team/${teamId}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(request),
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || "Failed to review team");
  }
}

/**
 * Bulk approve or reject all logos in a league
 */
export async function approveLeagueLogos(
  leagueId: number,
  request: ReviewLeagueRequest
): Promise<{ updated_count: number }> {
  const res = await fetch(`${API_BASE}/review/league/${leagueId}/approve`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(request),
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || "Failed to approve league");
  }

  return res.json();
}

// =============================================================================
// Team Status
// =============================================================================

/**
 * Get detailed status for a single team
 */
export async function fetchTeamLogoStatus(
  teamId: number
): Promise<TeamLogoReview> {
  const res = await fetch(`${API_BASE}/teams/${teamId}/status`);

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `Failed to fetch team status: ${res.status}`);
  }

  return parseTeamLogo(await res.json());
}

// =============================================================================
// Cost Estimation (Client-side)
// =============================================================================

/**
 * Estimate cost for a batch job
 */
export function estimateCost(
  teamCount: number,
  generationMode: GenerationMode,
  iaModel: IAModel
): CostEstimate {
  const imagesPerTeam: Record<GenerationMode, number> = {
    full_3d: 3,
    facing_only: 2,
    front_only: 1,
    manual: 0,
  };

  const costPerImage: Record<IAModel, number> = {
    "imagen-3": 0,
    "imagen-4": 0,
    gemini: 0,
    "dall-e-3": 0.04,
    sdxl: 0.006,
  };

  const freeModels: IAModel[] = ["imagen-3", "imagen-4", "gemini"];

  const images = imagesPerTeam[generationMode];
  const cost = costPerImage[iaModel];

  return {
    teamCount,
    imagesPerTeam: images,
    costPerImage: cost,
    totalCost: teamCount * images * cost,
    isFree: freeModels.includes(iaModel),
  };
}

// =============================================================================
// Parsers
// =============================================================================

function parseLeague(raw: Record<string, unknown>): LeagueForGeneration {
  return {
    leagueId: Number(raw.league_id) || 0,
    name: String(raw.name || ""),
    country: String(raw.country || ""),
    teamCount: Number(raw.team_count) || 0,
    pendingCount: Number(raw.pending_count) || 0,
    readyCount: Number(raw.ready_count) || 0,
    errorCount: Number(raw.error_count) || 0,
  };
}

function parseBatchJob(raw: Record<string, unknown>): LogoBatchJob {
  const progress = (raw.progress as Record<string, unknown>) || {};
  const cost = (raw.cost as Record<string, unknown>) || {};
  const timestamps = (raw.timestamps as Record<string, unknown>) || {};

  return {
    id: String(raw.batch_id || raw.id || ""),
    leagueId: Number(raw.league_id) || 0,
    leagueName: String(raw.league_name || ""),
    iaModel: (raw.ia_model as IAModel) || "imagen-3",
    generationMode: (raw.generation_mode as GenerationMode) || "full_3d",
    status: (raw.status as LogoBatchJob["status"]) || "running",
    totalTeams: Number(raw.total_teams) || 0,
    processedTeams: Number(progress.processed_teams) || 0,
    failedTeams: Number(progress.failed_teams) || 0,
    processedImages: Number(progress.processed_images) || 0,
    estimatedCostUsd: Number(cost.estimated_usd) || 0,
    actualCostUsd: Number(cost.actual_usd) || 0,
    progress: Number(progress.percentage) || 0,
    startedAt: String(timestamps.started_at || ""),
    pausedAt: timestamps.paused_at ? String(timestamps.paused_at) : undefined,
    completedAt: timestamps.completed_at
      ? String(timestamps.completed_at)
      : undefined,
    startedBy: String(raw.started_by || "unknown"),
  };
}

function parseTeamLogo(raw: Record<string, unknown>): TeamLogoReview {
  const urls = (raw.urls as Record<string, string>) || {};
  const thumbnails = raw.thumbnails as TeamLogoReview["thumbnails"];

  return {
    teamId: Number(raw.team_id) || 0,
    teamName: String(raw.team_name || raw.name || ""),
    status: (raw.status as TeamLogoReview["status"]) || "pending",
    reviewStatus: (raw.review_status as TeamLogoReview["reviewStatus"]) || "pending",
    urls: {
      original: urls.original || undefined,
      front: urls.front || undefined,
      right: urls.right || undefined,
      left: urls.left || undefined,
    },
    thumbnails,
    fallbackUrl: raw.fallback_url ? String(raw.fallback_url) : undefined,
    errorMessage: raw.error_message ? String(raw.error_message) : undefined,
    iaCostUsd: raw.ia_cost_usd ? Number(raw.ia_cost_usd) : undefined,
  };
}
