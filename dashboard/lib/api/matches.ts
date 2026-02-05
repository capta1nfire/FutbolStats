/**
 * Matches API Adapter
 *
 * Safe extraction and adaptation of data from /dashboard/matches.json
 * Designed to be resilient to partial or malformed responses.
 */

import { MatchSummary, MatchStatus, ProbabilitySet, MatchVenue, MatchWeather } from "@/lib/types";

/**
 * Expected response structure from backend
 */
export interface MatchesResponse {
  generated_at?: string;
  cached?: boolean;
  cache_age_seconds?: number;
  data: {
    matches: unknown[];
    total: number;
    page: number;
    limit: number;
    pages: number;
  };
}

/**
 * Pagination metadata from response
 */
export interface MatchesPagination {
  total: number;
  page: number;
  limit: number;
  pages: number;
}

/**
 * Safely check if value is a non-null object
 */
function isObject(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

/**
 * Map backend status codes to frontend MatchStatus
 *
 * Backend uses API-Football status codes:
 * - NS = Not Started (scheduled)
 * - 1H, 2H, ET, BT, P, SUSP, INT, LIVE = Live statuses
 * - HT = Half Time
 * - FT, AET, PEN = Finished
 * - PST, CANC, ABD, AWD, WO = Cancelled/Postponed
 */
function mapStatus(backendStatus: string): MatchStatus {
  const status = backendStatus?.toUpperCase() || "NS";

  // Scheduled
  if (status === "NS" || status === "TBD") {
    return "scheduled";
  }

  // Live statuses
  if (["1H", "2H", "ET", "BT", "P", "SUSP", "INT", "LIVE"].includes(status)) {
    return "live";
  }

  // Half time
  if (status === "HT") {
    return "ht";
  }

  // Finished
  if (["FT", "AET", "PEN"].includes(status)) {
    return "ft";
  }

  // Postponed
  if (status === "PST") {
    return "postponed";
  }

  // Cancelled
  if (["CANC", "ABD", "AWD", "WO"].includes(status)) {
    return "cancelled";
  }

  // Default to scheduled for unknown
  return "scheduled";
}

/**
 * Adapt a single raw match object to MatchSummary type
 *
 * Returns null if critical fields are missing or have wrong types.
 */
export function adaptMatch(raw: unknown): MatchSummary | null {
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

  // Accept both leagueId and league_id from backend
  const leagueId = raw.leagueId ?? raw.league_id ?? 0;
  const leagueIdNum = typeof leagueId === "number" ? leagueId : 0;

  // Accept both leagueName and league_name from backend
  const leagueName = raw.leagueName ?? raw.league_name ?? "";
  const leagueNameStr = typeof leagueName === "string" ? leagueName : "Unknown";

  // Accept both leagueCountry and league_country from backend
  const leagueCountry = raw.leagueCountry ?? raw.league_country ?? "";
  const leagueCountryStr = typeof leagueCountry === "string" ? leagueCountry : "";

  // Status mapping
  const backendStatus = typeof raw.status === "string" ? raw.status : "NS";
  const status = mapStatus(backendStatus);

  // Display names for use_short_names toggle (accept both snake_case and camelCase)
  const homeDisplayName = raw.home_display_name ?? raw.homeDisplayName;
  const awayDisplayName = raw.away_display_name ?? raw.awayDisplayName;

  // Build result
  const result: MatchSummary = {
    id,
    status,
    leagueId: leagueIdNum,
    leagueName: leagueNameStr,
    leagueCountry: leagueCountryStr,
    home,
    away,
    homeDisplayName: typeof homeDisplayName === "string" ? homeDisplayName : home,
    awayDisplayName: typeof awayDisplayName === "string" ? awayDisplayName : away,
    kickoffISO,
  };

  // Optional: score
  if (isObject(raw.score)) {
    const homeGoals = raw.score.home;
    const awayGoals = raw.score.away;
    if (typeof homeGoals === "number" && typeof awayGoals === "number") {
      result.score = { home: homeGoals, away: awayGoals };
    }
  }

  // Optional: elapsed (for live matches)
  if (typeof raw.elapsed === "number") {
    result.elapsed = {
      min: raw.elapsed,
      extra: typeof raw.elapsed_extra === "number" ? raw.elapsed_extra : undefined,
    };
  }

  // Optional: Venue
  const venue = parseVenue(raw.venue);
  if (venue) result.venue = venue;

  // Optional: Weather
  const weather = parseWeather(raw.weather);
  if (weather) result.weather = weather;

  // Optional: Market implied probabilities
  const market = parseProbabilitySet(raw.market);
  if (market) result.market = market;

  // Optional: Model A prediction
  const modelA = parseProbabilitySet(raw.model_a);
  if (modelA) result.modelA = modelA;

  // Optional: Shadow/Two-Stage prediction
  const shadow = parseProbabilitySet(raw.shadow);
  if (shadow) result.shadow = shadow;

  // Optional: Sensor B prediction
  const sensorB = parseProbabilitySet(raw.sensor_b);
  if (sensorB) result.sensorB = sensorB;

  // Optional: Ext-A experimental prediction
  const extA = parseProbabilitySet(raw.extA ?? raw.ext_a);
  if (extA) result.extA = extA;

  // Optional: Ext-B experimental prediction
  const extB = parseProbabilitySet(raw.extB ?? raw.ext_b);
  if (extB) result.extB = extB;

  // Optional: Ext-C experimental prediction
  const extC = parseProbabilitySet(raw.extC ?? raw.ext_c);
  if (extC) result.extC = extC;

  // Optional: Ext-D experimental prediction (league-only retrained)
  const extD = parseProbabilitySet(raw.extD ?? raw.ext_d);
  if (extD) result.extD = extD;

  return result;
}

/**
 * Parse probability set from raw object
 */
function parseProbabilitySet(raw: unknown): ProbabilitySet | null {
  if (!isObject(raw)) return null;

  const home = raw.home;
  const draw = raw.draw;
  const away = raw.away;

  if (typeof home !== "number" || typeof draw !== "number" || typeof away !== "number") {
    return null;
  }

  return { home, draw, away };
}

/**
 * Parse venue from raw object
 */
function parseVenue(raw: unknown): MatchVenue | null {
  if (!isObject(raw)) return null;

  const name = raw.name;
  const city = raw.city;

  // At least one field should be present
  if (typeof name !== "string" && typeof city !== "string") {
    return null;
  }

  return {
    name: typeof name === "string" ? name : null,
    city: typeof city === "string" ? city : null,
  };
}

/**
 * Parse weather from raw object
 */
function parseWeather(raw: unknown): MatchWeather | null {
  if (!isObject(raw)) return null;

  const temp_c = raw.temp_c;
  if (typeof temp_c !== "number") return null;

  return {
    temp_c,
    humidity: typeof raw.humidity === "number" ? raw.humidity : null,
    wind_ms: typeof raw.wind_ms === "number" ? raw.wind_ms : null,
    precip_mm: typeof raw.precip_mm === "number" ? raw.precip_mm : null,
    precip_prob: typeof raw.precip_prob === "number" ? raw.precip_prob : null,
    cloudcover: typeof raw.cloudcover === "number" ? raw.cloudcover : null,
    is_daylight: typeof raw.is_daylight === "boolean" ? raw.is_daylight : null,
  };
}

/**
 * Extract matches array from response
 *
 * Expected structure: { data: { matches: [...] } }
 */
export function extractMatches(response: unknown): unknown[] | null {
  if (!isObject(response)) {
    if (Array.isArray(response)) return response;
    return null;
  }

  // Try data.matches first (spec format)
  if (isObject(response.data) && Array.isArray(response.data.matches)) {
    return response.data.matches;
  }

  // Try root matches (alternative format)
  if (Array.isArray(response.matches)) {
    return response.matches;
  }

  return null;
}

/**
 * Extract pagination metadata from response
 */
export function extractPagination(response: unknown): MatchesPagination {
  const defaults: MatchesPagination = {
    total: 0,
    page: 1,
    limit: 50,
    pages: 1,
  };

  if (!isObject(response)) return defaults;

  const data = isObject(response.data) ? response.data : response;

  return {
    total: typeof data.total === "number" ? data.total : defaults.total,
    page: typeof data.page === "number" ? data.page : defaults.page,
    limit: typeof data.limit === "number" ? data.limit : defaults.limit,
    pages: typeof data.pages === "number" ? data.pages : defaults.pages,
  };
}

/**
 * Extract cache metadata from response
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
 * Parse full response to array of MatchSummary
 *
 * Returns null if extraction fails completely.
 * Individual invalid matches are skipped (best-effort).
 */
export function parseMatches(response: unknown): MatchSummary[] | null {
  const rawMatches = extractMatches(response);

  if (!rawMatches) {
    return null;
  }

  // Empty array is valid (no matches in time window)
  if (rawMatches.length === 0) {
    return [];
  }

  // Adapt each match, filtering out invalid ones
  const matches: MatchSummary[] = [];
  for (const raw of rawMatches) {
    const match = adaptMatch(raw);
    if (match) {
      matches.push(match);
    }
  }

  return matches;
}

/**
 * Map frontend MatchStatus filter to backend status param
 *
 * When no statuses provided, returns "ALL" to get all matches.
 */
export function mapStatusFilter(statuses: MatchStatus[]): string | undefined {
  if (statuses.length === 0) return "ALL";

  // Map frontend status to backend
  const backendStatuses: string[] = [];

  for (const status of statuses) {
    switch (status) {
      case "scheduled":
        backendStatuses.push("NS");
        break;
      case "live":
      case "ht":
        backendStatuses.push("LIVE");
        break;
      case "ft":
        backendStatuses.push("FT");
        break;
      // postponed/cancelled not directly supported by backend filter
    }
  }

  // If multiple, prefer ALL
  if (backendStatuses.length > 1) {
    return "ALL";
  }

  return backendStatuses[0] || undefined;
}
