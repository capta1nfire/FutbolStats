/**
 * Predictions API Adapter
 *
 * Transforms backend /dashboard/predictions.json response to dashboard types.
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

// ============================================================================
// Backend Types
// ============================================================================

/**
 * Backend prediction item from /dashboard/predictions.json
 */
interface BackendPrediction {
  id: number;
  match_id: number;
  league_id: number;
  league_name: string;
  kickoff_utc: string; // ISO
  home_team: string;
  away_team: string;
  status: string; // "NS", "FT", "1H", etc.
  score?: string | null; // "2-1" format
  model: string; // "baseline" or "shadow"
  model_version: string;
  pick?: string;
  probs?: {
    home: number;
    draw: number;
    away: number;
  };
  is_frozen?: boolean;
  frozen_at?: string | null;
  confidence_tier?: string;
  created_at?: string;
}

/**
 * Pagination info from backend
 */
export interface PredictionsPagination {
  total: number;
  page: number;
  limit: number;
  pages: number;
}

/**
 * Metadata from backend response
 */
export interface PredictionsMetadata {
  generatedAt: string | null;
  cached: boolean;
  cacheAgeSeconds: number;
  filtersApplied?: Record<string, unknown>;
}

/**
 * Full parsed response
 */
export interface PredictionsApiResponse {
  predictions: PredictionRow[];
  pagination: PredictionsPagination;
  metadata: PredictionsMetadata;
}

// ============================================================================
// Helpers
// ============================================================================

function isObject(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

/**
 * Map backend status to frontend PredictionStatus
 */
function mapStatus(backendStatus: string, hasPick: boolean): PredictionStatus {
  const status = backendStatus.toUpperCase();

  // Finished match
  if (status === "FT" || status === "AET" || status === "PEN") {
    return hasPick ? "evaluated" : "missing";
  }

  // Not started - has prediction
  if (status === "NS" || status === "TBD") {
    return hasPick ? "generated" : "missing";
  }

  // Live matches
  if (["1H", "2H", "HT", "ET", "BT", "P", "LIVE"].includes(status)) {
    return hasPick ? "frozen" : "missing";
  }

  // Postponed/cancelled
  if (["PST", "CANC", "ABD", "AWD", "WO"].includes(status)) {
    return "missing";
  }

  // Default: generated if has pick, missing otherwise
  return hasPick ? "generated" : "missing";
}

/**
 * Map backend model string to ModelType
 */
function mapModel(model: string): ModelType {
  const normalized = model.toLowerCase();
  if (normalized === "shadow" || normalized.includes("shadow")) {
    return "Shadow";
  }
  return "A"; // baseline maps to A
}

/**
 * Parse score string "2-1" to result
 */
function parseScoreToResult(score: string | null | undefined): MatchResult {
  if (!score) return "unknown";

  const parts = score.split("-");
  if (parts.length !== 2) return "unknown";

  const home = parseInt(parts[0], 10);
  const away = parseInt(parts[1], 10);

  if (isNaN(home) || isNaN(away)) return "unknown";

  if (home > away) return "home";
  if (away > home) return "away";
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

// ============================================================================
// Parsers
// ============================================================================

/**
 * Parse a single backend prediction to PredictionRow
 */
function parsePrediction(item: BackendPrediction): PredictionRow {
  const hasPick = !!item.pick && !!item.probs;

  return {
    id: item.id,
    matchId: item.match_id,
    matchLabel: `${item.home_team} vs ${item.away_team}`,
    home: item.home_team,
    away: item.away_team,
    leagueName: item.league_name || `League ${item.league_id}`,
    kickoffISO: item.kickoff_utc,
    model: mapModel(item.model),
    status: mapStatus(item.status, hasPick),
    generatedAt: item.frozen_at || item.created_at,
    probs: mapProbs(item.probs),
    pick: mapPick(item.pick),
    result: parseScoreToResult(item.score),
  };
}

/**
 * Parse backend response to PredictionsApiResponse
 *
 * Expected wrapper:
 * {
 *   generated_at: string,
 *   cached: boolean,
 *   cache_age_seconds: number,
 *   data: {
 *     predictions: [...],
 *     total: number,
 *     page: number,
 *     limit: number,
 *     pages: number,
 *     filters_applied: {...}
 *   }
 * }
 */
export function parsePredictionsResponse(response: unknown): PredictionsApiResponse | null {
  if (!isObject(response)) {
    return null;
  }

  // Extract metadata from root
  const generatedAt = typeof response.generated_at === "string" ? response.generated_at : null;
  const cached = typeof response.cached === "boolean" ? response.cached : false;
  const cacheAgeSeconds = typeof response.cache_age_seconds === "number" ? response.cache_age_seconds : 0;

  // Extract data object
  const data = response.data;
  if (!isObject(data)) {
    return null;
  }

  // Extract predictions array
  const rawPredictions = data.predictions;
  if (!Array.isArray(rawPredictions)) {
    return null;
  }

  // Parse predictions with best-effort (skip invalid items)
  const predictions: PredictionRow[] = [];
  for (const item of rawPredictions) {
    if (isObject(item) && typeof item.id === "number") {
      try {
        predictions.push(parsePrediction(item as unknown as BackendPrediction));
      } catch {
        // Skip invalid items
      }
    }
  }

  // Extract pagination
  const pagination: PredictionsPagination = {
    total: typeof data.total === "number" ? data.total : predictions.length,
    page: typeof data.page === "number" ? data.page : 1,
    limit: typeof data.limit === "number" ? data.limit : 50,
    pages: typeof data.pages === "number" ? data.pages : 1,
  };

  // Extract filters applied (for debugging)
  const filtersApplied = isObject(data.filters_applied) ? data.filters_applied : undefined;

  return {
    predictions,
    pagination,
    metadata: {
      generatedAt,
      cached,
      cacheAgeSeconds,
      filtersApplied,
    },
  };
}

/**
 * Legacy parser for backwards compatibility
 * @deprecated Use parsePredictionsResponse instead
 */
export function parsePredictions(data: unknown): PredictionRow[] {
  const response = parsePredictionsResponse(data);
  return response?.predictions ?? [];
}

/**
 * Extract pagination from response
 */
export function extractPagination(data: unknown): PredictionsPagination | null {
  const response = parsePredictionsResponse(data);
  return response?.pagination ?? null;
}

/**
 * Extract metadata from response
 */
export function extractMetadata(data: unknown): PredictionsMetadata {
  const response = parsePredictionsResponse(data);
  return response?.metadata ?? {
    generatedAt: null,
    cached: false,
    cacheAgeSeconds: 0,
  };
}

// ============================================================================
// Coverage Helpers
// ============================================================================

/**
 * Build coverage from predictions data
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

/**
 * Extract model version from response
 */
export function extractModelVersion(data: unknown): string | null {
  if (!data || typeof data !== "object") {
    return null;
  }

  const obj = data as Record<string, unknown>;

  // Try new format: data.predictions[0].model_version
  if (isObject(obj.data)) {
    const dataObj = obj.data as Record<string, unknown>;
    if (Array.isArray(dataObj.predictions) && dataObj.predictions.length > 0) {
      const first = dataObj.predictions[0] as Record<string, unknown>;
      if (typeof first.model_version === "string") {
        return first.model_version;
      }
    }
  }

  // Try legacy format
  if (typeof obj.model_version === "string") {
    return obj.model_version;
  }

  return null;
}
