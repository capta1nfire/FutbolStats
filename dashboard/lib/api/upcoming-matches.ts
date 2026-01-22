/**
 * Upcoming Matches API Adapter
 *
 * Safe extraction and adaptation of data from /dashboard/upcoming_matches.json
 * Designed to be resilient to partial or malformed responses.
 */

import { UpcomingMatch } from "@/lib/types";

/**
 * Expected response structure from backend
 */
export interface UpcomingMatchesResponse {
  generated_at?: string;
  cached?: boolean;
  cache_age_seconds?: number;
  data: {
    upcoming: unknown[];
  };
}

/**
 * Safely check if value is a non-null object
 */
function isObject(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

/**
 * Adapt a single raw match object to UpcomingMatch type
 *
 * Required fields:
 * - id: number
 * - home: string
 * - away: string
 * - kickoffISO: string (ISO date)
 * - leagueName: string
 * - hasPrediction: boolean
 *
 * Returns null if critical fields are missing or have wrong types.
 */
export function adaptUpcomingMatch(raw: unknown): UpcomingMatch | null {
  if (!isObject(raw)) return null;

  // Required fields with type validation
  const id = raw.id;
  if (typeof id !== "number") return null;

  const home = raw.home;
  if (typeof home !== "string" || home.length === 0) return null;

  const away = raw.away;
  if (typeof away !== "string" || away.length === 0) return null;

  // Accept both kickoffISO and kickoff_iso from backend
  const kickoffISO = raw.kickoffISO ?? raw.kickoff_iso ?? raw.kickoff;
  if (typeof kickoffISO !== "string") return null;

  // Accept both leagueName and league_name from backend
  const leagueName = raw.leagueName ?? raw.league_name ?? raw.league;
  if (typeof leagueName !== "string") return null;

  // Accept both hasPrediction and has_prediction from backend
  const hasPrediction = raw.hasPrediction ?? raw.has_prediction;
  // Default to false if not present or not boolean
  const hasPredictionBool =
    typeof hasPrediction === "boolean" ? hasPrediction : false;

  return {
    id,
    home,
    away,
    kickoffISO,
    leagueName,
    hasPrediction: hasPredictionBool,
  };
}

/**
 * Extract upcoming matches array from response
 *
 * Expected structure: { data: { upcoming: [...] } }
 * Also accepts: { upcoming: [...] } or just [...]
 */
export function extractUpcomingMatches(response: unknown): unknown[] | null {
  if (!isObject(response)) {
    // Maybe it's directly an array
    if (Array.isArray(response)) return response;
    return null;
  }

  // Try data.upcoming first (spec format)
  if (isObject(response.data) && Array.isArray(response.data.upcoming)) {
    return response.data.upcoming;
  }

  // Try root upcoming (alternative format)
  if (Array.isArray(response.upcoming)) {
    return response.upcoming;
  }

  // Try root data as array
  if (Array.isArray(response.data)) {
    return response.data;
  }

  return null;
}

/**
 * Extract metadata from response
 */
export function extractMetadata(response: unknown): {
  generatedAt: string | null;
  cached: boolean;
  cacheAgeSeconds: number;
} {
  if (!isObject(response)) {
    return { generatedAt: null, cached: false, cacheAgeSeconds: 0 };
  }

  const generatedAt =
    typeof response.generated_at === "string" ? response.generated_at : null;
  const cached =
    typeof response.cached === "boolean" ? response.cached : false;
  const cacheAgeSeconds =
    typeof response.cache_age_seconds === "number"
      ? response.cache_age_seconds
      : 0;

  return { generatedAt, cached, cacheAgeSeconds };
}

/**
 * Parse full response to array of UpcomingMatch
 *
 * Returns null if extraction fails completely.
 * Individual invalid matches are skipped (best-effort).
 */
export function parseUpcomingMatches(response: unknown): UpcomingMatch[] | null {
  const rawMatches = extractUpcomingMatches(response);

  if (!rawMatches || rawMatches.length === 0) {
    return null;
  }

  // Adapt each match, filtering out invalid ones
  const matches: UpcomingMatch[] = [];
  for (const raw of rawMatches) {
    const match = adaptUpcomingMatch(raw);
    if (match) {
      matches.push(match);
    }
  }

  // Return null if no valid matches (not empty array)
  // This allows fallback to mocks
  if (matches.length === 0) {
    return null;
  }

  return matches;
}
