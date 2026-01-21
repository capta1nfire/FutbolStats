/**
 * Match mock data
 * Provides factories and datasets for testing
 */

import {
  MatchSummary,
  MatchStatus,
  MatchFilters,
  ModelType,
  PredictionPick,
} from "@/lib/types";
import { mockConfig, simulateDelay, checkMockError } from "./config";

// Sample data for variety
const leagues = [
  { name: "Premier League", country: "England" },
  { name: "La Liga", country: "Spain" },
  { name: "Serie A", country: "Italy" },
  { name: "Bundesliga", country: "Germany" },
  { name: "Ligue 1", country: "France" },
  { name: "Liga BetPlay", country: "Colombia" },
  { name: "Liga MX", country: "Mexico" },
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

const statuses: MatchStatus[] = ["scheduled", "live", "ht", "ft"];
const models: ModelType[] = ["A", "Shadow"];

/**
 * Generate a random integer between min and max (inclusive)
 */
function randomInt(min: number, max: number): number {
  return Math.floor(Math.random() * (max - min + 1)) + min;
}

/**
 * Generate a random date within the next N days
 */
function randomFutureDate(days: number): string {
  const now = new Date();
  const future = new Date(now.getTime() + randomInt(0, days) * 24 * 60 * 60 * 1000);
  future.setHours(randomInt(12, 22), randomInt(0, 59), 0, 0);
  return future.toISOString();
}

/**
 * Generate a random past date within the last N days
 */
function randomPastDate(days: number): string {
  const now = new Date();
  const past = new Date(now.getTime() - randomInt(0, days) * 24 * 60 * 60 * 1000);
  past.setHours(randomInt(12, 22), randomInt(0, 59), 0, 0);
  return past.toISOString();
}

let idCounter = 1;

/**
 * Create a single mock match
 */
export function createMockMatch(overrides?: Partial<MatchSummary>): MatchSummary {
  const league = leagues[randomInt(0, leagues.length - 1)];
  const teamPair = teams[randomInt(0, teams.length - 1)];
  const status = statuses[randomInt(0, statuses.length - 1)];

  const match: MatchSummary = {
    id: idCounter++,
    status,
    leagueName: league.name,
    leagueCountry: league.country,
    home: teamPair[0],
    away: teamPair[1],
    kickoffISO:
      status === "ft" || status === "live" || status === "ht"
        ? randomPastDate(7)
        : randomFutureDate(14),
    ...overrides,
  };

  // Add score for live/ht/ft matches
  if (status === "live" || status === "ht" || status === "ft") {
    match.score = {
      home: randomInt(0, 4),
      away: randomInt(0, 3),
    };
  }

  // Add elapsed for live matches
  if (status === "live") {
    match.elapsed = {
      min: randomInt(1, 90),
      extra: randomInt(0, 100) > 80 ? randomInt(1, 5) : undefined,
    };
  } else if (status === "ht") {
    match.elapsed = { min: 45 };
  }

  // Add prediction (most matches have one)
  if (randomInt(0, 100) > 15) {
    const home = randomInt(20, 60) / 100;
    const draw = randomInt(15, 35) / 100;
    const away = Math.round((1 - home - draw) * 100) / 100;
    const maxProb = Math.max(home, draw, away);
    const pick: PredictionPick =
      maxProb === home ? "home" : maxProb === draw ? "draw" : "away";

    match.prediction = {
      model: models[randomInt(0, models.length - 1)],
      pick,
      probs: { home, draw, away },
    };
  }

  return match;
}

/**
 * Create multiple mock matches
 */
export function createMockMatches(count: number): MatchSummary[] {
  idCounter = 1; // Reset counter for consistent IDs
  return Array.from({ length: count }, () => createMockMatch());
}

// Pre-generated datasets
const normalDataset = createMockMatches(25);
const largeDataset = createMockMatches(120);

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
      data = largeDataset;
      break;
    default:
      data = normalDataset;
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
