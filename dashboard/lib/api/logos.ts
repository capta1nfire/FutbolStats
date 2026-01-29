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
    name: String(raw.league_name || raw.name || ""),
    country: String(raw.country || ""),
    teamCount: Number(raw.total_teams || raw.team_count) || 0,
    pendingCount: Number(raw.pending || raw.pending_count) || 0,
    readyCount: Number(raw.ready || raw.ready_count) || 0,
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

// =============================================================================
// Team Logo Upload & Status (Individual)
// =============================================================================

/**
 * Upload logo response
 */
export interface UploadLogoResponse {
  teamId: number;
  status: string;
  r2Key: string;
  validation: {
    width: number;
    height: number;
    format: string;
  };
}

/**
 * Team logo detailed status
 */
export interface TeamLogoStatus {
  teamId: number;
  teamName: string;
  status: string;
  reviewStatus: string;
  urls: {
    original?: string;
    front?: string;
    right?: string;
    left?: string;
  };
  fallbackUrl?: string;
  r2Keys: {
    original?: string;
    front?: string;
    right?: string;
    left?: string;
  };
  generation: {
    mode?: string;
    iaModel?: string;
    promptVersion?: string;
    costUsd?: number;
  };
  error?: {
    message?: string;
    phase?: string;
    retryCount?: number;
  };
  timestamps: {
    uploadedAt?: string;
    processingStartedAt?: string;
    processingCompletedAt?: string;
    resizeCompletedAt?: string;
  };
}

/**
 * Upload a logo for a specific team
 */
export async function uploadTeamLogo(
  teamId: number,
  file: File
): Promise<UploadLogoResponse> {
  const formData = new FormData();
  formData.append("file", file);

  const res = await fetch(`/api/logos/teams/${teamId}/upload`, {
    method: "POST",
    body: formData,
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || err.error || `Upload failed: ${res.status}`);
  }

  const data = await res.json();
  return {
    teamId: Number(data.team_id) || teamId,
    status: String(data.status || "pending"),
    r2Key: String(data.r2_key || ""),
    validation: {
      width: Number(data.validation?.width) || 0,
      height: Number(data.validation?.height) || 0,
      format: String(data.validation?.format || "unknown"),
    },
  };
}

/**
 * Fetch detailed status for a team's logo
 */
export async function fetchTeamLogoStatus(
  teamId: number
): Promise<TeamLogoStatus | null> {
  const res = await fetch(`${API_BASE}/teams/${teamId}/status`);

  if (res.status === 404) {
    return null; // No logo record exists
  }

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `Failed to fetch status: ${res.status}`);
  }

  const data = await res.json();
  return parseTeamLogoStatus(data);
}

function parseTeamLogoStatus(raw: Record<string, unknown>): TeamLogoStatus {
  const urls = (raw.urls as Record<string, string>) || {};
  const r2Keys = (raw.r2_keys as Record<string, string>) || {};
  const generation = (raw.generation as Record<string, unknown>) || {};
  const error = (raw.error as Record<string, unknown>) || {};
  const timestamps = (raw.timestamps as Record<string, string>) || {};

  return {
    teamId: Number(raw.team_id) || 0,
    teamName: String(raw.team_name || ""),
    status: String(raw.status || "pending"),
    reviewStatus: String(raw.review_status || "pending"),
    urls: {
      original: urls.original || undefined,
      front: urls.front || undefined,
      right: urls.right || undefined,
      left: urls.left || undefined,
    },
    fallbackUrl: raw.fallback_url ? String(raw.fallback_url) : undefined,
    r2Keys: {
      original: r2Keys.original || undefined,
      front: r2Keys.front || undefined,
      right: r2Keys.right || undefined,
      left: r2Keys.left || undefined,
    },
    generation: {
      mode: generation.mode ? String(generation.mode) : undefined,
      iaModel: generation.ia_model ? String(generation.ia_model) : undefined,
      promptVersion: generation.prompt_version
        ? String(generation.prompt_version)
        : undefined,
      costUsd: generation.cost_usd ? Number(generation.cost_usd) : undefined,
    },
    error: {
      message: error.message ? String(error.message) : undefined,
      phase: error.phase ? String(error.phase) : undefined,
      retryCount: error.retry_count ? Number(error.retry_count) : undefined,
    },
    timestamps: {
      uploadedAt: timestamps.uploaded_at || undefined,
      processingStartedAt: timestamps.processing_started_at || undefined,
      processingCompletedAt: timestamps.processing_completed_at || undefined,
      resizeCompletedAt: timestamps.resize_completed_at || undefined,
    },
  };
}

// =============================================================================
// Prompt Templates
// =============================================================================

/**
 * Prompt template for logo generation
 */
export interface PromptTemplate {
  id: number;
  version: string;
  variant: "front" | "right" | "left" | "main";
  promptTemplate: string;
  iaModel?: string;
  isActive: boolean;
  successRate?: number;
  usageCount?: number;
  notes?: string;
  createdAt?: string;
}

/**
 * Fetch all prompt templates
 */
export async function fetchPromptTemplates(
  version?: string,
  includeFull: boolean = true
): Promise<PromptTemplate[]> {
  const params = new URLSearchParams();
  if (version) params.set("version", version);
  if (includeFull) params.set("include_full", "true");

  const url = `${API_BASE}/prompts${params.toString() ? `?${params}` : ""}`;
  const res = await fetch(url);

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `Failed to fetch prompts: ${res.status}`);
  }

  const data = await res.json();
  return (data.prompts || []).map(parsePromptTemplate);
}

/**
 * Fetch a single prompt template by ID
 */
export async function fetchPromptTemplate(promptId: number): Promise<PromptTemplate> {
  const res = await fetch(`${API_BASE}/prompts/${promptId}`);

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `Failed to fetch prompt: ${res.status}`);
  }

  return parsePromptTemplate(await res.json());
}

/**
 * Update a prompt template
 */
export async function updatePromptTemplate(
  promptId: number,
  data: {
    promptTemplate?: string;
    isActive?: boolean;
    notes?: string;
    iaModel?: string;
  }
): Promise<PromptTemplate> {
  const res = await fetch(`${API_BASE}/prompts/${promptId}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      prompt_template: data.promptTemplate,
      is_active: data.isActive,
      notes: data.notes,
      ia_model: data.iaModel,
    }),
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `Failed to update prompt: ${res.status}`);
  }

  return parsePromptTemplate(await res.json());
}

function parsePromptTemplate(raw: Record<string, unknown>): PromptTemplate {
  return {
    id: Number(raw.id) || 0,
    version: String(raw.version || "v1"),
    variant: (raw.variant as PromptTemplate["variant"]) || "front",
    promptTemplate: String(raw.prompt_template || ""),
    iaModel: raw.ia_model ? String(raw.ia_model) : undefined,
    isActive: Boolean(raw.is_active),
    successRate: raw.success_rate ? Number(raw.success_rate) : undefined,
    usageCount: raw.usage_count ? Number(raw.usage_count) : undefined,
    notes: raw.notes ? String(raw.notes) : undefined,
    createdAt: raw.created_at ? String(raw.created_at) : undefined,
  };
}
