/**
 * Predictions mock data
 * Provides deterministic data for predictions
 */

import {
  PredictionRow,
  PredictionDetail,
  PredictionFilters,
  PredictionCoverage,
  PredictionStatus,
  ModelType,
  PredictionTimeRange,
  PredictionFeature,
  PredictionHistoryEntry,
} from "@/lib/types";
import { mockConfig, simulateDelay, checkMockError } from "./config";

/**
 * Static base timestamp for deterministic mock data
 */
const BASE_TIMESTAMP = new Date("2026-01-20T12:00:00Z").getTime();

/**
 * Mock leagues
 */
const leagues = [
  "La Liga",
  "Premier League",
  "Serie A",
  "Bundesliga",
  "Ligue 1",
  "Liga MX",
  "MLS",
  "Copa Libertadores",
];

/**
 * Mock match pairs [home, away]
 */
const matchPairs: [string, string][] = [
  ["Real Madrid", "Barcelona"],
  ["Liverpool", "Man United"],
  ["Juventus", "Inter Milan"],
  ["Bayern Munich", "Dortmund"],
  ["PSG", "Lyon"],
  ["América", "Guadalajara"],
  ["LA Galaxy", "LAFC"],
  ["Boca Juniors", "River Plate"],
  ["Atlético Madrid", "Sevilla"],
  ["Arsenal", "Chelsea"],
  ["AC Milan", "Napoli"],
  ["RB Leipzig", "Leverkusen"],
  ["Marseille", "Monaco"],
  ["Cruz Azul", "Monterrey"],
  ["Inter Miami", "Atlanta United"],
  ["Flamengo", "Palmeiras"],
];

/**
 * Create deterministic predictions
 */
function createPredictions(count: number): PredictionRow[] {
  return Array.from({ length: count }, (_, i) => {
    const statusIndex = i % 10;
    let status: PredictionStatus;
    if (statusIndex < 5) status = "generated";
    else if (statusIndex < 6) status = "missing";
    else if (statusIndex < 8) status = "frozen";
    else status = "evaluated";

    const model: ModelType = i % 3 === 0 ? "Shadow" : "A";
    const [home, away] = matchPairs[i % matchPairs.length];
    const matchLabel = `${home} vs ${away}`;
    const league = leagues[i % leagues.length];
    const kickoffOffset = (i % 20) * 3 * 60 * 60 * 1000; // 0-60 hours ahead

    const homeProb = 0.3 + (((i * 7) % 30) / 100);
    const drawProb = 0.2 + (((i * 3) % 20) / 100);
    const awayProb = 1 - homeProb - drawProb;

    const pick = homeProb > awayProb && homeProb > drawProb
      ? "home"
      : awayProb > drawProb
        ? "away"
        : "draw";

    const result = status === "evaluated"
      ? (["home", "draw", "away"] as const)[i % 3]
      : undefined;

    return {
      id: 2000 + i,
      matchId: 70000 + i,
      matchLabel,
      home,
      away,
      leagueName: league,
      kickoffISO: new Date(BASE_TIMESTAMP + kickoffOffset).toISOString(),
      model,
      status,
      generatedAt: status !== "missing"
        ? new Date(BASE_TIMESTAMP - (i + 1) * 30 * 60 * 1000).toISOString()
        : undefined,
      probs: status !== "missing"
        ? { home: homeProb, draw: drawProb, away: awayProb }
        : undefined,
      pick: status !== "missing" ? pick : undefined,
      result,
    };
  });
}

/**
 * Create top features for a prediction
 */
function createFeatures(predictionId: number): PredictionFeature[] {
  const seed = predictionId * 17;
  return [
    { name: "Home Form (5 games)", value: `${60 + (seed % 30)}%` },
    { name: "Away Form (5 games)", value: `${50 + ((seed + 3) % 35)}%` },
    { name: "H2H Last 5", value: `${2 + (seed % 3)}W-${1 + (seed % 2)}D-${seed % 2}L` },
    { name: "Home Goals Avg", value: (1.5 + (seed % 10) / 10).toFixed(2) },
    { name: "Away Goals Avg", value: (1.2 + ((seed + 5) % 10) / 10).toFixed(2) },
    { name: "xG Difference", value: `+${(0.2 + (seed % 5) / 10).toFixed(2)}` },
    { name: "League Position Diff", value: seed % 10 - 5 },
    { name: "Days Since Last Match", value: 3 + (seed % 4) },
  ];
}

/**
 * Create history for a prediction
 */
function createHistory(prediction: PredictionRow): PredictionHistoryEntry[] {
  const entries: PredictionHistoryEntry[] = [];
  const baseTs = prediction.generatedAt
    ? new Date(prediction.generatedAt).getTime()
    : BASE_TIMESTAMP;

  if (prediction.status === "evaluated") {
    entries.push({
      ts: new Date(baseTs + 4 * 60 * 60 * 1000).toISOString(),
      status: "evaluated",
      model: prediction.model,
    });
  }

  if (prediction.status === "frozen" || prediction.status === "evaluated") {
    entries.push({
      ts: new Date(baseTs + 60 * 60 * 1000).toISOString(),
      status: "frozen",
      model: prediction.model,
    });
  }

  if (prediction.status !== "missing") {
    entries.push({
      ts: new Date(baseTs).toISOString(),
      status: "generated",
      model: prediction.model,
    });
  }

  return entries.reverse();
}

// Pre-generated datasets
const normalDataset = createPredictions(50);
const largeDataset = createPredictions(200);

/**
 * Get time range in milliseconds
 */
function getTimeRangeMs(range: PredictionTimeRange): number {
  switch (range) {
    case "24h":
      return 24 * 60 * 60 * 1000;
    case "48h":
      return 48 * 60 * 60 * 1000;
    case "7d":
      return 7 * 24 * 60 * 60 * 1000;
    case "30d":
      return 30 * 24 * 60 * 60 * 1000;
  }
}

/**
 * Get predictions based on scenario and filters
 */
export async function getPredictionsMock(
  filters?: PredictionFilters
): Promise<PredictionRow[]> {
  await simulateDelay();
  checkMockError();

  let data: PredictionRow[];

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
      data = data.filter((p) => filters.status!.includes(p.status));
    }
    if (filters.model && filters.model.length > 0) {
      data = data.filter((p) => filters.model!.includes(p.model));
    }
    if (filters.league && filters.league.length > 0) {
      data = data.filter((p) => filters.league!.includes(p.leagueName));
    }
    if (filters.timeRange) {
      const rangeMs = getTimeRangeMs(filters.timeRange);
      const cutoff = BASE_TIMESTAMP + rangeMs;
      data = data.filter((p) => new Date(p.kickoffISO).getTime() <= cutoff);
    }
    if (filters.search) {
      const search = filters.search.toLowerCase();
      data = data.filter(
        (p) =>
          p.matchLabel.toLowerCase().includes(search) ||
          p.leagueName.toLowerCase().includes(search)
      );
    }
  }

  return data;
}

/**
 * Get a single prediction by ID with full details
 */
export async function getPredictionMock(
  id: number
): Promise<PredictionDetail | null> {
  await simulateDelay(300);
  checkMockError();

  const allPredictions = mockConfig.scenario === "large" ? largeDataset : normalDataset;
  const prediction = allPredictions.find((p) => p.id === id);

  if (!prediction) return null;

  return {
    ...prediction,
    featuresTop: createFeatures(id),
    evaluation: prediction.status === "evaluated"
      ? {
          accuracy: prediction.result === prediction.pick ? 1 : 0,
          brier: 0.15 + (id % 20) / 100,
          notes: prediction.result === prediction.pick
            ? "Prediction was correct"
            : "Prediction was incorrect",
        }
      : undefined,
    history: createHistory(prediction),
  };
}

/**
 * Get prediction coverage summary
 */
export async function getPredictionCoverageMock(
  timeRange: PredictionTimeRange = "24h"
): Promise<PredictionCoverage> {
  await simulateDelay(200);
  checkMockError();

  const allPredictions = mockConfig.scenario === "large" ? largeDataset : normalDataset;
  const rangeMs = getTimeRangeMs(timeRange);
  const cutoff = BASE_TIMESTAMP + rangeMs;

  const inRange = allPredictions.filter(
    (p) => new Date(p.kickoffISO).getTime() <= cutoff
  );

  const withPrediction = inRange.filter((p) => p.status !== "missing").length;
  const missingCount = inRange.filter((p) => p.status === "missing").length;
  const totalMatches = inRange.length;

  return {
    totalMatches,
    withPrediction,
    missingCount,
    coveragePct: totalMatches > 0 ? Math.round((withPrediction / totalMatches) * 100) : 100,
    periodLabel: PREDICTION_TIME_RANGE_LABELS[timeRange],
  };
}

// Import labels for coverage mock
import { PREDICTION_TIME_RANGE_LABELS } from "@/lib/types";

/**
 * Get unique leagues from predictions
 */
export function getPredictionLeaguesMock(): string[] {
  return leagues;
}

/**
 * Get status counts
 */
export function getPredictionStatusCountsMock(): Record<PredictionStatus, number> {
  const allPredictions = mockConfig.scenario === "large" ? largeDataset : normalDataset;
  const counts: Record<PredictionStatus, number> = {
    generated: 0,
    missing: 0,
    frozen: 0,
    evaluated: 0,
  };
  allPredictions.forEach((p) => counts[p.status]++);
  return counts;
}

/**
 * Get model counts
 */
export function getPredictionModelCountsMock(): Record<ModelType, number> {
  const allPredictions = mockConfig.scenario === "large" ? largeDataset : normalDataset;
  const counts: Record<ModelType, number> = { A: 0, Shadow: 0 };
  allPredictions.forEach((p) => counts[p.model]++);
  return counts;
}
