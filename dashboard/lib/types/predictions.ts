/**
 * Predictions Types
 *
 * Types for the Predictions section
 */

import { ModelType } from "./match";

export type PredictionStatus = "generated" | "missing" | "frozen" | "evaluated";

// Re-export ModelType for convenience
export type { ModelType };

export type PickOutcome = "home" | "draw" | "away";

export type MatchResult = "home" | "draw" | "away" | "unknown";

export type PredictionTimeRange = "24h" | "48h" | "7d" | "30d";

/**
 * Probability distribution for 1X2 prediction
 */
export interface PredictionProbs {
  home: number;
  draw: number;
  away: number;
}

/**
 * Prediction row for table display
 */
export interface PredictionRow {
  id: number;
  matchId: number;
  matchLabel: string; // e.g., "Real Madrid vs Barcelona"
  leagueName: string;
  kickoffISO: string;
  model: ModelType;
  status: PredictionStatus;
  generatedAt?: string; // ISO timestamp
  probs?: PredictionProbs;
  pick?: PickOutcome;
  result?: MatchResult; // if evaluated
}

/**
 * Feature importance for model explanation
 */
export interface PredictionFeature {
  name: string;
  value: string | number;
}

/**
 * Evaluation metrics for a prediction
 */
export interface PredictionEvaluation {
  accuracy?: number; // 0-1
  brier?: number; // Brier score
  notes?: string;
}

/**
 * History entry for prediction changes
 */
export interface PredictionHistoryEntry {
  ts: string; // ISO timestamp
  status: PredictionStatus;
  model: ModelType;
}

/**
 * Full prediction detail with extended information
 */
export interface PredictionDetail extends PredictionRow {
  featuresTop?: PredictionFeature[];
  evaluation?: PredictionEvaluation;
  history?: PredictionHistoryEntry[];
}

/**
 * Filters for predictions table
 */
export interface PredictionFilters {
  status?: PredictionStatus[];
  model?: ModelType[];
  league?: string[];
  timeRange?: PredictionTimeRange;
  search?: string;
}

/**
 * Coverage summary for predictions
 */
export interface PredictionCoverage {
  totalMatches: number;
  withPrediction: number;
  missingCount: number;
  coveragePct: number;
  periodLabel: string; // e.g., "Next 24 hours"
}

/**
 * Prediction status labels
 */
export const PREDICTION_STATUS_LABELS: Record<PredictionStatus, string> = {
  generated: "Generated",
  missing: "Missing",
  frozen: "Frozen",
  evaluated: "Evaluated",
};

/**
 * All prediction statuses
 */
export const PREDICTION_STATUSES: PredictionStatus[] = [
  "generated",
  "missing",
  "frozen",
  "evaluated",
];

/**
 * Model type labels
 */
export const MODEL_TYPE_LABELS: Record<ModelType, string> = {
  A: "Model A",
  Shadow: "Shadow",
};

/**
 * All model types
 */
export const MODEL_TYPES: ModelType[] = ["A", "Shadow"];

/**
 * Time range labels
 */
export const PREDICTION_TIME_RANGE_LABELS: Record<PredictionTimeRange, string> = {
  "24h": "Next 24 hours",
  "48h": "Next 48 hours",
  "7d": "Next 7 days",
  "30d": "Next 30 days",
};

/**
 * All time ranges
 */
export const PREDICTION_TIME_RANGES: PredictionTimeRange[] = ["24h", "48h", "7d", "30d"];

/**
 * Pick outcome labels
 */
export const PICK_OUTCOME_LABELS: Record<PickOutcome, string> = {
  home: "Home",
  draw: "Draw",
  away: "Away",
};

/**
 * Match result labels
 */
export const MATCH_RESULT_LABELS: Record<MatchResult, string> = {
  home: "Home Win",
  draw: "Draw",
  away: "Away Win",
  unknown: "Unknown",
};
