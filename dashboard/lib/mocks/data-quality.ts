/**
 * Data Quality mock data
 * Provides deterministic data for testing data quality checks
 */

import {
  DataQualityCheck,
  DataQualityCheckDetail,
  DataQualityAffectedItem,
  DataQualityHistoryEntry,
  DataQualityStatus,
  DataQualityCategory,
  DataQualityFilters,
} from "@/lib/types";
import { mockConfig, simulateDelay, checkMockError } from "./config";

/**
 * Static base timestamp for deterministic mock data
 */
const BASE_TIMESTAMP = new Date("2026-01-20T12:00:00Z").getTime();

/**
 * Predefined checks with realistic data
 */
const checkDefinitions: Omit<DataQualityCheck, "id" | "lastRunAt">[] = [
  {
    name: "Prediction Coverage",
    category: "coverage",
    status: "warning",
    currentValue: "87%",
    threshold: "95%",
    affectedCount: 3,
    description: "Percentage of scheduled matches with predictions",
  },
  {
    name: "Odds Data Availability",
    category: "odds",
    status: "passing",
    currentValue: "98%",
    threshold: "90%",
    affectedCount: 0,
    description: "Percentage of matches with odds data from bookmakers",
  },
  {
    name: "Match Score Consistency",
    category: "consistency",
    status: "passing",
    currentValue: "100%",
    threshold: "99%",
    affectedCount: 0,
    description: "Scores match between API source and database",
  },
  {
    name: "Team Data Completeness",
    category: "completeness",
    status: "passing",
    currentValue: "99.2%",
    threshold: "98%",
    affectedCount: 2,
    description: "Teams have all required metadata (logo, country, etc)",
  },
  {
    name: "Fixture Freshness",
    category: "freshness",
    status: "passing",
    currentValue: "2m",
    threshold: "5m",
    affectedCount: 0,
    description: "Time since last fixture sync from API",
  },
  {
    name: "Live Score Latency",
    category: "freshness",
    status: "warning",
    currentValue: "45s",
    threshold: "30s",
    affectedCount: 4,
    description: "Average delay in live score updates",
  },
  {
    name: "Duplicate Fixtures",
    category: "consistency",
    status: "failing",
    currentValue: "3",
    threshold: "0",
    affectedCount: 3,
    description: "Number of duplicate fixture entries detected",
  },
  {
    name: "Missing Kickoff Times",
    category: "completeness",
    status: "passing",
    currentValue: "0",
    threshold: "0",
    affectedCount: 0,
    description: "Scheduled matches without kickoff time",
  },
  {
    name: "Stale Predictions",
    category: "freshness",
    status: "passing",
    currentValue: "0",
    threshold: "5",
    affectedCount: 0,
    description: "Predictions older than 24h for upcoming matches",
  },
  {
    name: "Odds Movement Anomalies",
    category: "odds",
    status: "warning",
    currentValue: "2",
    threshold: "0",
    affectedCount: 2,
    description: "Unusual odds movements detected (>20% change)",
  },
  {
    name: "League Coverage",
    category: "coverage",
    status: "passing",
    currentValue: "12/12",
    threshold: "12/12",
    affectedCount: 0,
    description: "Active leagues with prediction coverage",
  },
  {
    name: "Historical Stats Completeness",
    category: "completeness",
    status: "warning",
    currentValue: "94%",
    threshold: "98%",
    affectedCount: 8,
    description: "Finished matches with complete statistics",
  },
];

/**
 * Create affected items for a check
 */
function createAffectedItems(
  checkId: number,
  count: number
): DataQualityAffectedItem[] {
  const kinds: DataQualityAffectedItem["kind"][] = ["match", "team", "league", "job"];

  return Array.from({ length: count }, (_, i) => {
    const kind = kinds[i % kinds.length];
    const labels: Record<typeof kind, string[]> = {
      match: ["Real Madrid vs Barcelona", "Liverpool vs Man United", "Bayern vs Dortmund"],
      team: ["Deportivo Cali", "América de Cali", "Millonarios"],
      league: ["Copa Libertadores", "MLS", "J-League"],
      job: ["global_sync #12345", "odds_sync #12344", "stats_backfill #12343"],
    };

    return {
      id: `${checkId}-${i}`,
      label: labels[kind][i % labels[kind].length],
      kind,
      details: {
        timestamp: new Date(BASE_TIMESTAMP - i * 3600000).toISOString(),
        reason: "Value below threshold",
      },
    };
  });
}

/**
 * Create history entries for a check
 */
function createHistoryEntries(
  status: DataQualityStatus,
  count: number
): DataQualityHistoryEntry[] {
  const statuses: DataQualityStatus[] = ["passing", "warning", "failing"];

  return Array.from({ length: count }, (_, i) => ({
    timestamp: new Date(BASE_TIMESTAMP - i * 3600000).toISOString(),
    status: i === 0 ? status : statuses[(i + 1) % statuses.length],
    value: `${85 + (i % 15)}%`,
  }));
}

/**
 * Create deterministic checks
 */
function createChecks(count: number): DataQualityCheck[] {
  return Array.from({ length: count }, (_, i) => {
    const def = checkDefinitions[i % checkDefinitions.length];
    return {
      ...def,
      id: 1000 + i,
      lastRunAt: new Date(BASE_TIMESTAMP - (i * 5) * 60000).toISOString(),
    };
  });
}

// Pre-generated datasets
const normalDataset = createChecks(12);
const largeDataset = createChecks(50);

/**
 * Get data quality checks based on scenario
 */
export async function getDataQualityChecksMock(
  filters?: DataQualityFilters
): Promise<DataQualityCheck[]> {
  await simulateDelay();
  checkMockError();

  let data: DataQualityCheck[];

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
      data = data.filter((c) => filters.status!.includes(c.status));
    }
    if (filters.category && filters.category.length > 0) {
      data = data.filter((c) => filters.category!.includes(c.category));
    }
    if (filters.search) {
      const search = filters.search.toLowerCase();
      data = data.filter(
        (c) =>
          c.name.toLowerCase().includes(search) ||
          c.description?.toLowerCase().includes(search) ||
          c.category.toLowerCase().includes(search)
      );
    }
  }

  return data;
}

/**
 * Get a single check by ID with full details
 */
export async function getDataQualityCheckMock(
  id: number
): Promise<DataQualityCheckDetail | null> {
  await simulateDelay(300);
  checkMockError();

  const allChecks = mockConfig.scenario === "large" ? largeDataset : normalDataset;
  const check = allChecks.find((c) => c.id === id);

  if (!check) return null;

  return {
    ...check,
    affectedItems: createAffectedItems(id, check.affectedCount),
    history: createHistoryEntries(check.status, 24),
  };
}

/**
 * Get affected items for a check
 */
export async function getDataQualityAffectedItemsMock(
  checkId: number
): Promise<DataQualityAffectedItem[]> {
  await simulateDelay(200);
  checkMockError();

  const allChecks = mockConfig.scenario === "large" ? largeDataset : normalDataset;
  const check = allChecks.find((c) => c.id === checkId);

  if (!check) return [];

  return createAffectedItems(checkId, check.affectedCount);
}

/**
 * Get counts per status
 */
export function getDataQualityStatusCountsMock(): Record<DataQualityStatus, number> {
  const allChecks = mockConfig.scenario === "large" ? largeDataset : normalDataset;
  return {
    passing: allChecks.filter((c) => c.status === "passing").length,
    warning: allChecks.filter((c) => c.status === "warning").length,
    failing: allChecks.filter((c) => c.status === "failing").length,
  };
}

/**
 * Get counts per category
 */
export function getDataQualityCategoryCountsMock(): Record<DataQualityCategory, number> {
  const allChecks = mockConfig.scenario === "large" ? largeDataset : normalDataset;
  return {
    coverage: allChecks.filter((c) => c.category === "coverage").length,
    consistency: allChecks.filter((c) => c.category === "consistency").length,
    completeness: allChecks.filter((c) => c.category === "completeness").length,
    freshness: allChecks.filter((c) => c.category === "freshness").length,
    odds: allChecks.filter((c) => c.category === "odds").length,
  };
}
