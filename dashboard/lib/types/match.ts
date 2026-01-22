/**
 * Match types - Stable contracts for Phase 0
 * SSOT: dashboard/DASHBOARD_PRO_BY_DAVID.md Section 8
 */

export type MatchStatus =
  | "scheduled"
  | "live"
  | "ht"
  | "ft"
  | "postponed"
  | "cancelled";

/**
 * All match statuses (for validation and iteration)
 */
export const MATCH_STATUSES: readonly MatchStatus[] = [
  "scheduled",
  "live",
  "ht",
  "ft",
  "postponed",
  "cancelled",
] as const;

export type PredictionPick = "home" | "draw" | "away";

export interface MatchScore {
  home: number;
  away: number;
}

export interface MatchElapsed {
  min: number;
  extra?: number;
}

/**
 * Probability distribution for 1X2 predictions
 */
export interface ProbabilitySet {
  home: number;
  draw: number;
  away: number;
}

export interface MatchSummary {
  id: number;
  status: MatchStatus;
  leagueName: string;
  leagueCountry: string;
  home: string;
  away: string;
  kickoffISO: string;
  score?: MatchScore;
  elapsed?: MatchElapsed;
  /** Market implied probabilities (from frozen odds) */
  market?: ProbabilitySet;
  /** Model A prediction (production model) */
  modelA?: ProbabilitySet;
  /** Shadow/Two-Stage prediction (experimental) */
  shadow?: ProbabilitySet;
  /** Sensor B prediction (calibration diagnostic) */
  sensorB?: ProbabilitySet;
}

/**
 * Filters for match queries
 */
export interface MatchFilters {
  status?: MatchStatus[];
  leagues?: string[];
  dateFrom?: string;
  dateTo?: string;
  search?: string;
}
