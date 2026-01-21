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

export type PredictionPick = "home" | "draw" | "away";
export type ModelType = "A" | "Shadow";

export interface MatchScore {
  home: number;
  away: number;
}

export interface MatchElapsed {
  min: number;
  extra?: number;
}

export interface MatchPrediction {
  model: ModelType;
  pick: PredictionPick;
  probs?: {
    home: number;
    draw: number;
    away: number;
  };
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
  prediction?: MatchPrediction;
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
