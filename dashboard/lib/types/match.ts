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

/**
 * Venue information (stadium)
 */
export interface MatchVenue {
  name: string | null;
  city: string | null;
}

/**
 * Weather forecast data
 */
export interface MatchWeather {
  temp_c: number;
  humidity: number | null;
  wind_ms: number | null;
  precip_mm: number | null;
  precip_prob: number | null;
  cloudcover: number | null;
  is_daylight: boolean | null;
}

export interface MatchSummary {
  id: number;
  status: MatchStatus;
  leagueId: number;
  leagueName: string;
  leagueCountry: string;
  /** Round / matchday label (e.g., "Regular Season - 21") */
  round?: string;
  home: string;
  away: string;
  /** Team ID for home team (for navigation) */
  homeTeamId: number;
  /** Team ID for away team (for navigation) */
  awayTeamId: number;
  /** Short display name for home team (COALESCE: override > wikidata > name) */
  homeDisplayName: string;
  /** Short display name for away team (COALESCE: override > wikidata > name) */
  awayDisplayName: string;
  kickoffISO: string;
  score?: MatchScore;
  elapsed?: MatchElapsed;
  /** Venue/stadium information */
  venue?: MatchVenue;
  /** Weather forecast at kickoff */
  weather?: MatchWeather;
  /** Market implied probabilities (from frozen odds) */
  market?: ProbabilitySet;
  /** Model A prediction (production model) */
  modelA?: ProbabilitySet;
  /** Shadow/Two-Stage prediction (experimental) */
  shadow?: ProbabilitySet;
  /** Sensor B prediction (calibration diagnostic) */
  sensorB?: ProbabilitySet;
  /** Experimental ext-A prediction */
  extA?: ProbabilitySet;
  /** Experimental ext-B prediction */
  extB?: ProbabilitySet;
  /** Experimental ext-C prediction */
  extC?: ProbabilitySet;
  /** Experimental ext-D prediction (league-only retrained) */
  extD?: ProbabilitySet;
  /** Consensus market (median of de-vigged bookmakers, fair probs) */
  consensus?: ProbabilitySet;
  /** Pinnacle sharp benchmark (implied probs) */
  pinnacle?: ProbabilitySet;
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

/**
 * Standing entry for a team in a league
 */
export interface StandingEntry {
  position: number;
  teamId: number;
  teamName: string;
  displayName: string; // COALESCE(override.short_name, wikidata.short_name, name)
  teamLogo: string | null;
  points: number;
  played: number;
  won: number;
  drawn: number;
  lost: number;
  goalsFor: number;
  goalsAgainst: number;
  goalDiff: number;
  form?: string;
  description?: string | null; // Promotion/Relegation status
}

/**
 * Available table entry in standings metadata (Phase 5)
 */
export interface AvailableTable {
  group: string;
  team_count: number;
  type: "regular" | "playoff" | "reclasificacion" | "descenso" | "group_stage";
  is_current: boolean;
}

/**
 * Standings metadata (Phase 5)
 */
export interface StandingsMeta {
  available_groups: string[];
  available_tables: AvailableTable[];
  selected_group: string;
  selection_reason: string;
  tie_warning: string[] | null;
  zones_source?: string | null;
  is_group_stage: boolean;
}

/**
 * Reclasificación entry (Phase 3 — accumulated Apertura + Clausura)
 */
export interface ReclasificacionEntry {
  position: number;
  team_id: number;
  team_name: string;
  team_logo: string | null;
  points: number;
  played: number;
  won: number;
  drawn: number;
  lost: number;
  goals_for: number;
  goals_against: number;
  goal_diff: number;
  zone: { type: string; style: string } | null;
}

/**
 * Reclasificación data (Phase 3)
 */
export interface ReclasificacionData {
  data: ReclasificacionEntry[];
  source: string;
}

/**
 * Descenso entry (Phase 4 — relegation by 3-year average)
 */
export interface DescensoEntry {
  position: number;
  team_id: number;
  team_name: string;
  display_name?: string;
  team_logo: string | null;
  points: number;
  played: number;
  average: number;
  goals_for: number;
  goals_against: number;
  goal_diff: number;
  zone: { type: string; style: string } | null;
}

/**
 * Descenso data (Phase 4)
 */
export interface DescensoData {
  data: DescensoEntry[];
  method: string;
  source: string;
  relegation_count: number;
  seasons?: number[];
}

/**
 * League standings response
 */
export interface StandingsResponse {
  leagueId: number;
  season: number;
  standings: StandingEntry[];
  source: string;
  isPlaceholder?: boolean;
  isCalculated?: boolean;
  meta?: StandingsMeta;
  reclasificacion?: ReclasificacionData | null;
  descenso?: DescensoData | null;
}
