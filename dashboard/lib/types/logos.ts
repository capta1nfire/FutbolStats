/**
 * Logos 3D Types
 *
 * Types for the 3D logo generation and review system.
 * Spec: docs/TEAM_LOGOS_3D_SPEC.md
 */

// =============================================================================
// Status Types
// =============================================================================

/** Batch job status */
export type LogoBatchStatus =
  | "running"
  | "paused"
  | "completed"
  | "cancelled"
  | "error"
  | "pending_review";

/** Team logo pipeline status */
export type TeamLogoStatus =
  | "pending"
  | "queued"
  | "processing"
  | "pending_resize"
  | "ready"
  | "error"
  | "paused";

/** Review status for approval workflow */
export type ReviewStatus =
  | "pending"
  | "approved"
  | "rejected"
  | "needs_regeneration";

/** Generation mode */
export type GenerationMode = "full_3d" | "facing_only" | "front_only" | "manual";

/** IA Model for generation */
export type IAModel = "imagen-3" | "imagen-4" | "gemini" | "dall-e-3" | "sdxl";

// =============================================================================
// Data Types
// =============================================================================

/**
 * League available for logo generation
 */
export interface LeagueForGeneration {
  leagueId: number;
  name: string;
  country: string;
  teamCount: number;
  pendingCount: number;
  readyCount: number;
  errorCount: number;
}

/**
 * Batch job status and progress
 */
export interface LogoBatchJob {
  id: string;
  leagueId: number;
  leagueName: string;
  iaModel: IAModel;
  generationMode: GenerationMode;
  status: LogoBatchStatus;
  totalTeams: number;
  processedTeams: number;
  failedTeams: number;
  processedImages: number;
  estimatedCostUsd: number;
  actualCostUsd: number;
  progress: number; // 0-100
  startedAt: string;
  pausedAt?: string;
  completedAt?: string;
  startedBy: string;
}

/**
 * Team logo for review grid
 */
export interface TeamLogoReview {
  teamId: number;
  teamName: string;
  status: TeamLogoStatus;
  reviewStatus: ReviewStatus;
  urls: {
    original?: string;
    front?: string;
    right?: string;
    left?: string;
  };
  thumbnails?: {
    front?: Record<number, string>;
    right?: Record<number, string>;
    left?: Record<number, string>;
  };
  fallbackUrl?: string;
  errorMessage?: string;
  iaCostUsd?: number;
}

// =============================================================================
// Request/Response Types
// =============================================================================

/**
 * Request to start batch generation
 */
export interface GenerateBatchRequest {
  generation_mode: GenerationMode;
  ia_model: IAModel;
  prompt_version?: string;
}

/**
 * Response from starting batch
 */
export interface GenerateBatchResponse {
  batch_id: string;
  status: string;
  total_teams: number;
  estimated_cost_usd: number;
  message: string;
}

/**
 * Request to review a team's logos
 */
export interface ReviewTeamRequest {
  action: "approve" | "reject" | "regenerate";
  notes?: string;
}

/**
 * Request to bulk approve/reject league
 */
export interface ReviewLeagueRequest {
  action: "approve_all" | "reject_all";
}

// =============================================================================
// Cost Estimation
// =============================================================================

/**
 * Cost estimate for a batch
 */
export interface CostEstimate {
  teamCount: number;
  imagesPerTeam: number;
  costPerImage: number;
  totalCost: number;
  isFree: boolean;
}

// =============================================================================
// Error Types
// =============================================================================

/**
 * Generation error with specific codes
 */
export interface GenerationError {
  code:
    | "FREE_TIER_EXCEEDED"
    | "RATE_LIMIT"
    | "INVALID_PROMPT"
    | "API_ERROR"
    | "VALIDATION_FAILED";
  message: string;
  retryAfter?: number;
}

// =============================================================================
// Constants
// =============================================================================

/** IA model display names */
export const IA_MODEL_LABELS: Record<IAModel, string> = {
  "imagen-3": "Imagen 3 (FREE)",
  "imagen-4": "Imagen 4 (FREE)",
  gemini: "Gemini (FREE)",
  "dall-e-3": "DALL-E 3 ($0.04/img)",
  sdxl: "SDXL ($0.006/img)",
};

/** Generation mode display names */
export const GENERATION_MODE_LABELS: Record<GenerationMode, string> = {
  full_3d: "Full 3D (3 variants)",
  facing_only: "Facing Only (2 variants)",
  front_only: "Front Only (1 variant)",
  manual: "Manual Upload",
};

/** Images per team for each mode */
export const IMAGES_PER_MODE: Record<GenerationMode, number> = {
  full_3d: 3,
  facing_only: 2,
  front_only: 1,
  manual: 0,
};

/** Cost per image for each model (USD) */
export const COST_PER_IMAGE: Record<IAModel, number> = {
  "imagen-3": 0,
  "imagen-4": 0,
  gemini: 0,
  "dall-e-3": 0.04,
  sdxl: 0.006,
};

/** Free tier models */
export const FREE_TIER_MODELS: IAModel[] = ["imagen-3", "imagen-4", "gemini"];

/** Free tier daily limit */
export const FREE_TIER_DAILY_LIMIT = 50;

/** Batch status display labels */
export const BATCH_STATUS_LABELS: Record<LogoBatchStatus, string> = {
  running: "Running",
  paused: "Paused",
  completed: "Completed",
  cancelled: "Cancelled",
  error: "Error",
  pending_review: "Pending Review",
};

/** Review status display labels */
export const REVIEW_STATUS_LABELS: Record<ReviewStatus, string> = {
  pending: "Pending",
  approved: "Approved",
  rejected: "Rejected",
  needs_regeneration: "Needs Regen",
};
