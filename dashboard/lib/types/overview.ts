/**
 * Overview Types
 *
 * Types for the dashboard overview/home page
 */

export type HealthStatus = "healthy" | "warning" | "critical";

export interface HealthCard {
  id: string;
  title: string;
  status: HealthStatus;
  value: string | number;
  subtitle?: string;
  trend?: "up" | "down" | "stable";
}

export interface OverviewCounts {
  matchesLive: number;
  matchesScheduledToday: number;
  incidentsActive: number;
  incidentsCritical: number;
  jobsRunning: number;
  jobsFailedLast24h: number;
  predictionsMissing: number;
  predictionsTotal: number;
}

export interface HealthSummary {
  coveragePct: number;
  counts: OverviewCounts;
  cards: HealthCard[];
  lastUpdated: string; // ISO timestamp
}

/**
 * Minimal match info for upcoming list
 */
export interface UpcomingMatch {
  id: number;
  home: string;
  away: string;
  kickoffISO: string;
  leagueName: string;
  hasPrediction: boolean;
}

/**
 * Minimal incident info for active list
 */
export interface ActiveIncident {
  id: number;
  title: string;
  severity: "critical" | "warning" | "info";
  createdAt: string;
  type: string;
}

/**
 * API Budget status
 */
export type ApiBudgetStatus = "ok" | "warning" | "critical" | "degraded";

/**
 * API Budget information
 *
 * Future source: GET /dashboard/ops.json → data.budget
 */
export interface ApiBudget {
  /** Current status of the API budget */
  status: ApiBudgetStatus;
  /** Plan name (e.g., "Ultra", "Pro") */
  plan: string;
  /** Plan expiration date (ISO string, optional) */
  plan_end?: string;
  /** Whether the API is currently active */
  active: boolean;
  /** Number of requests made today */
  requests_today: number;
  /** Daily request limit */
  requests_limit: number;
  /** Remaining requests for today */
  requests_remaining: number;
  /** Whether this data is from cache */
  cached: boolean;
  /** Age of the cache in seconds */
  cache_age_seconds: number;
  /** Timestamp when tokens reset (ISO string, LA timezone) */
  tokens_reset_at_la?: string;
  /** Human-readable note about reset timing */
  tokens_reset_note?: string;
}

/**
 * Combined overview data
 */
export interface OverviewData {
  health: HealthSummary;
  upcomingMatches: UpcomingMatch[];
  activeIncidents: ActiveIncident[];
  apiBudget?: ApiBudget;
}
