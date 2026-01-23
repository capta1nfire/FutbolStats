/**
 * Predictions API Adapter
 *
 * Transforms backend /predictions/upcoming response to dashboard types.
 * Best-effort parsing with null-safety.
 */

import {
  PredictionRow,
  PredictionStatus,
  ModelType,
  PredictionProbs,
  PickOutcome,
  MatchResult,
  PredictionCoverage,
} from "@/lib/types";

/**
 * Backend prediction item from /predictions/upcoming
 */
interface BackendPrediction {
  match_id: number;
  match_external_id?: number;
  home_team: string;
  away_team: string;
  date: string; // ISO
  status: string; // "NS", "FT", "1H", etc.
  home_goals?: number | null;
  away_goals?: number | null;
  league_id: number;
  pick?: string;
  probabilities?: {
    home: number;
    draw: number;
    away: number;
  };
  is_frozen?: boolean;
  frozen_at?: string;
  served_from_rerun?: boolean;
}

/**
 * Backend response from /predictions/upcoming
 */
interface BackendPredictionsResponse {
  predictions: BackendPrediction[];
  model_version: string;
  context_applied: boolean;
}

/**
 * Determine prediction status from backend data
 */
function mapStatus(item: BackendPrediction): PredictionStatus {
  // If match is finished (FT), check if we have a prediction
  if (item.status === "FT") {
    if (item.probabilities) {
      return "evaluated";
    }
    return "missing";
  }

  // If frozen, return frozen status
  if (item.is_frozen) {
    return "frozen";
  }

  // If we have probabilities, it's generated
  if (item.probabilities) {
    return "generated";
  }

  // No prediction
  return "missing";
}

/**
 * Determine match result from score
 */
function mapResult(item: BackendPrediction): MatchResult {
  if (item.status !== "FT") {
    return "unknown";
  }

  const homeGoals = item.home_goals ?? 0;
  const awayGoals = item.away_goals ?? 0;

  if (homeGoals > awayGoals) return "home";
  if (awayGoals > homeGoals) return "away";
  return "draw";
}

/**
 * Parse pick outcome
 */
function mapPick(pick?: string): PickOutcome | undefined {
  if (!pick) return undefined;
  const normalized = pick.toLowerCase();
  if (normalized === "home" || normalized === "draw" || normalized === "away") {
    return normalized as PickOutcome;
  }
  return undefined;
}

/**
 * Parse probabilities
 */
function mapProbs(
  probs?: { home: number; draw: number; away: number }
): PredictionProbs | undefined {
  if (!probs) return undefined;
  return {
    home: probs.home,
    draw: probs.draw,
    away: probs.away,
  };
}

/**
 * Parse a single backend prediction to PredictionRow
 */
function parsePrediction(item: BackendPrediction): PredictionRow {
  return {
    id: item.match_id, // Use match_id as the row ID
    matchId: item.match_id,
    matchLabel: `${item.home_team} vs ${item.away_team}`,
    leagueName: `League ${item.league_id}`, // Will be enriched by UI if needed
    kickoffISO: item.date,
    model: (item.served_from_rerun ? "Shadow" : "A") as ModelType,
    status: mapStatus(item),
    generatedAt: item.frozen_at,
    probs: mapProbs(item.probabilities),
    pick: mapPick(item.pick),
    result: mapResult(item),
  };
}

/**
 * Parse backend predictions response to PredictionRow[]
 *
 * Supports both response shapes:
 * 1) { predictions: [...] }
 * 2) { data: { predictions: [...] } }
 */
export function parsePredictions(data: unknown): PredictionRow[] {
  if (!data || typeof data !== "object") {
    return [];
  }

  const obj = data as Record<string, unknown>;

  // Try shape 1: { predictions: [...] }
  if (Array.isArray(obj.predictions)) {
    return obj.predictions.map((item) =>
      parsePrediction(item as BackendPrediction)
    );
  }

  // Try shape 2: { data: { predictions: [...] } }
  if (obj.data && typeof obj.data === "object") {
    const nested = obj.data as Record<string, unknown>;
    if (Array.isArray(nested.predictions)) {
      return nested.predictions.map((item) =>
        parsePrediction(item as BackendPrediction)
      );
    }
  }

  // Best-effort: no crash, return empty
  return [];
}

/**
 * Extract model version from response
 */
export function extractModelVersion(data: unknown): string | null {
  if (!data || typeof data !== "object") {
    return null;
  }

  const response = data as BackendPredictionsResponse;
  return response.model_version || null;
}

/**
 * Build coverage from predictions data
 * Since backend doesn't have a separate coverage endpoint,
 * we calculate it from the predictions list
 */
export function buildCoverageFromPredictions(
  predictions: PredictionRow[],
  periodLabel: string = "Next 24 hours"
): PredictionCoverage {
  const totalMatches = predictions.length;
  const withPrediction = predictions.filter(
    (p) => p.status !== "missing"
  ).length;
  const missingCount = totalMatches - withPrediction;
  const coveragePct = totalMatches > 0 ? (withPrediction / totalMatches) * 100 : 100;

  return {
    totalMatches,
    withPrediction,
    missingCount,
    coveragePct: Math.round(coveragePct * 10) / 10,
    periodLabel,
  };
}

/**
 * Extract coverage from ops.json predictions_health
 * This is more accurate as it comes from the ops dashboard
 */
export function extractCoverageFromOps(
  opsData: unknown,
  periodLabel: string = "Next 48 hours"
): PredictionCoverage | null {
  if (!opsData || typeof opsData !== "object") {
    return null;
  }

  const data = opsData as Record<string, unknown>;
  const predictionsHealth = data.data as Record<string, unknown> | undefined;

  if (!predictionsHealth) {
    return null;
  }

  const health = predictionsHealth.predictions_health as Record<string, unknown> | undefined;

  if (!health) {
    return null;
  }

  const nsNext48h = (health.ns_matches_next_48h as number) ?? 0;
  const nsMissing = (health.ns_matches_next_48h_missing_prediction as number) ?? 0;
  const coveragePct = (health.ns_coverage_pct as number) ?? 100;

  return {
    totalMatches: nsNext48h,
    withPrediction: nsNext48h - nsMissing,
    missingCount: nsMissing,
    coveragePct,
    periodLabel,
  };
}
