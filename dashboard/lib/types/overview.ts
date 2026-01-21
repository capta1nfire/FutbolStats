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
 * Combined overview data
 */
export interface OverviewData {
  health: HealthSummary;
  upcomingMatches: UpcomingMatch[];
  activeIncidents: ActiveIncident[];
}
