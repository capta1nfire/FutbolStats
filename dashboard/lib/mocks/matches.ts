/**
 * Match mock data
 * Provides deterministic factories and datasets for testing
 * All data is static to avoid hydration mismatches
 */

import {
  MatchSummary,
  MatchStatus,
  MatchFilters,
  ProbabilitySet,
} from "@/lib/types";
import { mockConfig, simulateDelay, checkMockError } from "./config";

/**
 * Static base timestamp for deterministic mock data
 */
const BASE_TIMESTAMP = new Date("2026-01-20T15:00:00Z").getTime();

// Sample data for variety
const leagues = [
  { id: 39, name: "Premier League", country: "England" },
  { id: 140, name: "La Liga", country: "Spain" },
  { id: 135, name: "Serie A", country: "Italy" },
  { id: 78, name: "Bundesliga", country: "Germany" },
  { id: 61, name: "Ligue 1", country: "France" },
  { id: 239, name: "Liga BetPlay", country: "Colombia" },
  { id: 262, name: "Liga MX", country: "Mexico" },
];

const teams = [
  ["Manchester United", "Liverpool"],
  ["Real Madrid", "Barcelona"],
  ["Juventus", "AC Milan"],
  ["Bayern Munich", "Borussia Dortmund"],
  ["PSG", "Lyon"],
  ["América de Cali", "Deportivo Cali"],
  ["Club América", "Guadalajara"],
  ["Arsenal", "Chelsea"],
  ["Atlético Madrid", "Sevilla"],
  ["Inter Milan", "Roma"],
  ["RB Leipzig", "Bayer Leverkusen"],
  ["Marseille", "Monaco"],
  ["Millonarios", "Santa Fe"],
  ["Tigres", "Monterrey"],
];

const statuses: MatchStatus[] = ["scheduled", "live", "ht", "ft", "ft", "ft", "scheduled", "scheduled"];

/**
 * Create a deterministic mock match based on index
 */
function createDeterministicMatch(index: number): MatchSummary {
  const leagueIndex = index % leagues.length;
  const teamIndex = index % teams.length;
  const statusIndex = index % statuses.length;

  const league = leagues[leagueIndex];
  const teamPair = teams[teamIndex];
  const status = statuses[statusIndex];

  // Deterministic kickoff time
  const dayOffset = (index % 7) - 3; // -3 to +3 days
  const hourOffset = 12 + (index % 10); // 12:00 to 21:00
  const kickoffTimestamp = BASE_TIMESTAMP + dayOffset * 86400000 + hourOffset * 3600000;

  const match: MatchSummary = {
    id: index + 1,
    status,
    leagueId: league.id,
    leagueName: league.name,
    leagueCountry: league.country,
    home: teamPair[0],
    away: teamPair[1],
    kickoffISO: new Date(kickoffTimestamp).toISOString(),
  };

  // Add score for live/ht/ft matches
  if (status === "live" || status === "ht" || status === "ft") {
    match.score = {
      home: (index * 7) % 4,
      away: (index * 3) % 3,
    };
  }

  // Add elapsed for live matches
  if (status === "live") {
    match.elapsed = {
      min: 1 + ((index * 13) % 89),
      extra: index % 5 === 0 ? 1 + (index % 4) : undefined,
    };
  } else if (status === "ht") {
    match.elapsed = { min: 45 };
  }

  // Add predictions (most matches have them)
  if (index % 6 !== 0) {
    // Base probabilities that vary by index
    const homeBase = 25 + ((index * 11) % 35);
    const drawBase = 15 + ((index * 7) % 20);
    const awayBase = 100 - homeBase - drawBase;

    const probs: ProbabilitySet = {
      home: homeBase / 100,
      draw: drawBase / 100,
      away: awayBase / 100,
    };

    // Model A (always present if has prediction)
    match.modelA = probs;

    // Market (slightly different, simulating odds)
    match.market = {
      home: Math.max(0.1, probs.home + ((index % 10) - 5) / 100),
      draw: Math.max(0.1, probs.draw + ((index % 8) - 4) / 100),
      away: Math.max(0.1, probs.away + ((index % 6) - 3) / 100),
    };

    // Shadow (50% of matches)
    if (index % 2 === 0) {
      match.shadow = {
        home: Math.max(0.1, probs.home + ((index % 12) - 6) / 100),
        draw: Math.max(0.1, probs.draw + ((index % 9) - 4) / 100),
        away: Math.max(0.1, probs.away + ((index % 7) - 3) / 100),
      };
    }

    // Sensor B (33% of matches)
    if (index % 3 === 0) {
      match.sensorB = {
        home: Math.max(0.1, probs.home + ((index % 14) - 7) / 100),
        draw: Math.max(0.1, probs.draw + ((index % 11) - 5) / 100),
        away: Math.max(0.1, probs.away + ((index % 8) - 4) / 100),
      };
    }
  }

  return match;
}

/**
 * Create multiple deterministic mock matches
 */
function createDeterministicMatches(count: number): MatchSummary[] {
  return Array.from({ length: count }, (_, i) => createDeterministicMatch(i));
}

// Pre-generated static datasets
const normalDataset: MatchSummary[] = createDeterministicMatches(25);
const largeDataset: MatchSummary[] = createDeterministicMatches(120);

/**
 * Get matches based on current mock scenario
 */
export async function getMatchesMock(
  filters?: MatchFilters
): Promise<MatchSummary[]> {
  await simulateDelay();
  checkMockError();

  let data: MatchSummary[];

  switch (mockConfig.scenario) {
    case "empty":
      data = [];
      break;
    case "large":
      data = [...largeDataset];
      break;
    default:
      data = [...normalDataset];
  }

  // Apply filters (basic implementation)
  if (filters) {
    if (filters.status && filters.status.length > 0) {
      data = data.filter((m) => filters.status!.includes(m.status));
    }
    if (filters.leagues && filters.leagues.length > 0) {
      data = data.filter((m) => filters.leagues!.includes(m.leagueName));
    }
    if (filters.search) {
      const search = filters.search.toLowerCase();
      data = data.filter(
        (m) =>
          m.home.toLowerCase().includes(search) ||
          m.away.toLowerCase().includes(search) ||
          m.leagueName.toLowerCase().includes(search)
      );
    }
  }

  return data;
}

/**
 * Get matches synchronously (for fallback when API fails)
 * Used by components for immediate fallback without async
 */
export function getMatchesMockSync(filters?: MatchFilters): MatchSummary[] {
  let data: MatchSummary[];

  switch (mockConfig.scenario) {
    case "empty":
      data = [];
      break;
    case "large":
      data = [...largeDataset];
      break;
    default:
      data = [...normalDataset];
  }

  // Apply filters
  if (filters) {
    if (filters.status && filters.status.length > 0) {
      data = data.filter((m) => filters.status!.includes(m.status));
    }
    if (filters.leagues && filters.leagues.length > 0) {
      data = data.filter((m) => filters.leagues!.includes(m.leagueName));
    }
    if (filters.search) {
      const search = filters.search.toLowerCase();
      data = data.filter(
        (m) =>
          m.home.toLowerCase().includes(search) ||
          m.away.toLowerCase().includes(search) ||
          m.leagueName.toLowerCase().includes(search)
      );
    }
  }

  return data;
}

/**
 * Get a single match by ID
 */
export async function getMatchByIdMock(
  id: number
): Promise<MatchSummary | null> {
  await simulateDelay(300);
  checkMockError();

  const allMatches =
    mockConfig.scenario === "large" ? largeDataset : normalDataset;
  return allMatches.find((m) => m.id === id) ?? null;
}

/**
 * Get unique leagues from dataset
 */
export function getLeaguesMock(): string[] {
  const allMatches =
    mockConfig.scenario === "large" ? largeDataset : normalDataset;
  return [...new Set(allMatches.map((m) => m.leagueName))];
}

/**
 * Get counts per status
 */
export function getStatusCountsMock(): Record<MatchStatus, number> {
  const allMatches =
    mockConfig.scenario === "large" ? largeDataset : normalDataset;
  return {
    scheduled: allMatches.filter((m) => m.status === "scheduled").length,
    live: allMatches.filter((m) => m.status === "live").length,
    ht: allMatches.filter((m) => m.status === "ht").length,
    ft: allMatches.filter((m) => m.status === "ft").length,
    postponed: allMatches.filter((m) => m.status === "postponed").length,
    cancelled: allMatches.filter((m) => m.status === "cancelled").length,
  };
}

// Legacy exports for backwards compatibility
export function createMockMatch(overrides?: Partial<MatchSummary>): MatchSummary {
  return { ...createDeterministicMatch(0), ...overrides };
}

export function createMockMatches(count: number): MatchSummary[] {
  return createDeterministicMatches(count);
}
