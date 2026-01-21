/**
 * Overview mock data
 * Provides deterministic data for the dashboard overview page
 */

import {
  HealthSummary,
  HealthCard,
  OverviewCounts,
  UpcomingMatch,
  ActiveIncident,
  OverviewData,
  ApiBudget,
} from "@/lib/types";
import { mockConfig, simulateDelay, checkMockError } from "./config";

/**
 * Static base timestamp for deterministic mock data
 */
const BASE_TIMESTAMP = new Date("2026-01-20T14:00:00Z").getTime();

/**
 * Create deterministic health cards
 */
function createHealthCards(scenario: string): HealthCard[] {
  if (scenario === "empty") {
    return [
      { id: "system", title: "System", status: "healthy", value: "OK", subtitle: "All services running" },
      { id: "predictions", title: "Predictions", status: "healthy", value: "0%", subtitle: "No matches today" },
      { id: "jobs", title: "Jobs", status: "healthy", value: "0", subtitle: "No jobs running" },
      { id: "live", title: "Live", status: "healthy", value: "0", subtitle: "No live matches" },
    ];
  }

  // Normal/large scenarios
  return [
    {
      id: "system",
      title: "System",
      status: "healthy",
      value: "OK",
      subtitle: "All services operational",
      trend: "stable",
    },
    {
      id: "predictions",
      title: "Predictions",
      status: "warning",
      value: "87%",
      subtitle: "3 matches missing",
      trend: "down",
    },
    {
      id: "jobs",
      title: "Jobs",
      status: "healthy",
      value: "2",
      subtitle: "Running now",
      trend: "stable",
    },
    {
      id: "live",
      title: "Live",
      status: "healthy",
      value: "4",
      subtitle: "Matches in progress",
      trend: "up",
    },
  ];
}

/**
 * Create deterministic counts
 */
function createCounts(scenario: string): OverviewCounts {
  if (scenario === "empty") {
    return {
      matchesLive: 0,
      matchesScheduledToday: 0,
      incidentsActive: 0,
      incidentsCritical: 0,
      jobsRunning: 0,
      jobsFailedLast24h: 0,
      predictionsMissing: 0,
      predictionsTotal: 0,
    };
  }

  return {
    matchesLive: 4,
    matchesScheduledToday: 12,
    incidentsActive: 3,
    incidentsCritical: 1,
    jobsRunning: 2,
    jobsFailedLast24h: 1,
    predictionsMissing: 3,
    predictionsTotal: 23,
  };
}

/**
 * Create deterministic health summary
 */
function createHealthSummary(scenario: string): HealthSummary {
  return {
    coveragePct: scenario === "empty" ? 100 : 87,
    counts: createCounts(scenario),
    cards: createHealthCards(scenario),
    lastUpdated: new Date(BASE_TIMESTAMP).toISOString(),
  };
}

/**
 * Create deterministic upcoming matches
 */
function createUpcomingMatches(count: number): UpcomingMatch[] {
  const teams = [
    ["Manchester United", "Liverpool"],
    ["Real Madrid", "Barcelona"],
    ["Bayern Munich", "Dortmund"],
    ["PSG", "Lyon"],
    ["Juventus", "AC Milan"],
    ["Arsenal", "Chelsea"],
  ];

  const leagues = [
    "Premier League",
    "La Liga",
    "Bundesliga",
    "Ligue 1",
    "Serie A",
    "Premier League",
  ];

  return Array.from({ length: count }, (_, i) => {
    const teamPair = teams[i % teams.length];
    const hoursFromNow = 1 + i * 2;
    return {
      id: 1000 + i,
      home: teamPair[0],
      away: teamPair[1],
      kickoffISO: new Date(BASE_TIMESTAMP + hoursFromNow * 3600000).toISOString(),
      leagueName: leagues[i % leagues.length],
      hasPrediction: i % 4 !== 0, // 75% have predictions
    };
  });
}

/**
 * Create deterministic active incidents
 */
function createActiveIncidents(count: number): ActiveIncident[] {
  const titles = [
    "API-Football rate limit warning",
    "Missing prediction: Real Madrid vs Barcelona",
    "global_sync job timeout",
    "High latency on /predictions endpoint",
    "Data inconsistency: duplicate fixture",
  ];

  const types = [
    "api_error",
    "missing_prediction",
    "job_failure",
    "high_latency",
    "data_inconsistency",
  ];

  const severities: ("critical" | "warning" | "info")[] = [
    "warning",
    "critical",
    "warning",
    "info",
    "warning",
  ];

  return Array.from({ length: count }, (_, i) => ({
    id: 10000 - i,
    title: titles[i % titles.length],
    severity: severities[i % severities.length],
    createdAt: new Date(BASE_TIMESTAMP - i * 3600000).toISOString(),
    type: types[i % types.length],
  }));
}

/**
 * Create deterministic API Budget data
 */
function createApiBudget(scenario: string): ApiBudget {
  // Calculate reset time: today at 4pm LA (24:00 UTC or 00:00 UTC next day depending on DST)
  const resetHour = 16; // 4pm LA
  const laOffset = -8; // PST (simplification, ignores DST)
  const resetDate = new Date(BASE_TIMESTAMP);
  resetDate.setUTCHours(resetHour - laOffset, 0, 0, 0);

  // If reset time has passed, add 24 hours
  if (resetDate.getTime() < BASE_TIMESTAMP) {
    resetDate.setTime(resetDate.getTime() + 24 * 3600000);
  }

  if (scenario === "empty") {
    return {
      status: "ok",
      plan: "Ultra",
      plan_end: "2026-06-15",
      active: true,
      requests_today: 0,
      requests_limit: 75000,
      requests_remaining: 75000,
      cached: false,
      cache_age_seconds: 0,
      tokens_reset_at_la: resetDate.toISOString(),
      tokens_reset_note: "Observed daily refresh around 4:00pm LA",
    };
  }

  // Normal scenario: moderate usage
  const requestsToday = 2847;
  const requestsLimit = 75000;

  return {
    status: "ok",
    plan: "Ultra",
    plan_end: "2026-06-15",
    active: true,
    requests_today: requestsToday,
    requests_limit: requestsLimit,
    requests_remaining: requestsLimit - requestsToday,
    cached: true,
    cache_age_seconds: 45,
    tokens_reset_at_la: resetDate.toISOString(),
    tokens_reset_note: "Observed daily refresh around 4:00pm LA",
  };
}

// Pre-generated datasets
const normalUpcoming = createUpcomingMatches(6);
const normalIncidents = createActiveIncidents(3);
const largeUpcoming = createUpcomingMatches(12);
const largeIncidents = createActiveIncidents(8);
const normalApiBudget = createApiBudget("normal");
const emptyApiBudget = createApiBudget("empty");

/**
 * Get overview data based on scenario
 */
export async function getOverviewDataMock(): Promise<OverviewData> {
  await simulateDelay();
  checkMockError();

  const scenario = mockConfig.scenario;

  if (scenario === "empty") {
    return {
      health: createHealthSummary("empty"),
      upcomingMatches: [],
      activeIncidents: [],
      apiBudget: emptyApiBudget,
    };
  }

  const isLarge = scenario === "large";

  return {
    health: createHealthSummary(scenario),
    upcomingMatches: isLarge ? largeUpcoming : normalUpcoming,
    activeIncidents: isLarge ? largeIncidents : normalIncidents,
    apiBudget: normalApiBudget,
  };
}

/**
 * Get health summary only
 */
export async function getHealthSummaryMock(): Promise<HealthSummary> {
  await simulateDelay(300);
  checkMockError();
  return createHealthSummary(mockConfig.scenario);
}

/**
 * Get upcoming matches only
 */
export async function getUpcomingMatchesMock(): Promise<UpcomingMatch[]> {
  await simulateDelay(300);
  checkMockError();

  if (mockConfig.scenario === "empty") return [];
  return mockConfig.scenario === "large" ? largeUpcoming : normalUpcoming;
}

/**
 * Get active incidents only
 */
export async function getActiveIncidentsMock(): Promise<ActiveIncident[]> {
  await simulateDelay(300);
  checkMockError();

  if (mockConfig.scenario === "empty") return [];
  return mockConfig.scenario === "large" ? largeIncidents : normalIncidents;
}
