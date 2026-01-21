import {
  JobRun,
  JobDefinition,
  JobStatus,
  JobFilters,
  JOB_NAMES,
  JobName,
} from "@/lib/types";

/**
 * Static base timestamp for deterministic mock data
 * Using a fixed date avoids hydration mismatches
 */
const BASE_TIMESTAMP = new Date("2026-01-20T12:00:00Z").getTime();

/**
 * Create a deterministic mock job run
 */
function createDeterministicJobRun(index: number): JobRun {
  const statuses: JobStatus[] = ["success", "success", "success", "failed", "running", "success", "success", "pending"];
  const jobNames: JobName[] = [...JOB_NAMES];

  // Use index as seed for deterministic selection
  const statusIndex = index % statuses.length;
  const jobIndex = index % jobNames.length;
  const status = statuses[statusIndex];
  const jobName = jobNames[jobIndex];

  // Deterministic timestamps based on index
  const startedAt = new Date(BASE_TIMESTAMP - index * 120000).toISOString();
  const durationMs = status === "running" || status === "pending" ? undefined : 500 + (index * 137) % 29500;
  const finishedAt =
    status === "running" || status === "pending"
      ? undefined
      : new Date(BASE_TIMESTAMP - index * 120000 + (durationMs || 0)).toISOString();

  const triggeredBy = index % 5 === 0 ? "manual" as const : "scheduler" as const;
  const isRetry = index % 7 === 0;

  return {
    id: 100000 - index,
    jobName,
    status,
    startedAt,
    finishedAt,
    durationMs,
    triggeredBy: isRetry ? "retry" : triggeredBy,
    error: status === "failed" ? "Connection timeout to API-Football" : undefined,
  };
}

/**
 * Create deterministic mock job runs
 */
function createDeterministicJobRuns(count: number): JobRun[] {
  const runs: JobRun[] = [];

  for (let i = 0; i < count; i++) {
    runs.push(createDeterministicJobRun(i));
  }

  return runs;
}

/**
 * Job definitions with schedules
 */
export const jobDefinitions: JobDefinition[] = [
  {
    name: "global_sync",
    description: "Sync match fixtures and results from API-Football",
    schedule: "Every 1 minute",
    enabled: true,
  },
  {
    name: "live_tick",
    description: "Update live match data (scores, events)",
    schedule: "Every 10 seconds",
    enabled: true,
  },
  {
    name: "stats_backfill",
    description: "Capture statistics for finished matches",
    schedule: "Every 60 minutes",
    enabled: true,
  },
  {
    name: "odds_sync",
    description: "Sync betting odds for upcoming matches",
    schedule: "Every 6 hours",
    enabled: true,
  },
  {
    name: "fastpath",
    description: "Generate LLM narratives for live matches",
    schedule: "Every 2 minutes",
    enabled: true,
  },
  {
    name: "narrative_generator",
    description: "Generate post-match narratives with Gemini",
    schedule: "On match finish",
    enabled: false,
  },
];

/**
 * Static mock datasets - created once, deterministic
 */
const normalDataset: JobRun[] = createDeterministicJobRuns(30);
const largeDataset: JobRun[] = createDeterministicJobRuns(150);
const emptyDataset: JobRun[] = [];

/**
 * Get mock job runs based on scenario
 */
export function getJobRunsMock(
  filters?: JobFilters,
  scenario: "normal" | "empty" | "error" | "large" = "normal"
): JobRun[] {
  if (scenario === "error") {
    throw new Error("Failed to fetch job runs");
  }

  if (scenario === "empty") {
    return emptyDataset;
  }

  const dataset = scenario === "large" ? largeDataset : normalDataset;

  let filtered = [...dataset];

  // Apply filters
  if (filters?.status && filters.status.length > 0) {
    filtered = filtered.filter((run) => filters.status!.includes(run.status));
  }

  if (filters?.jobName && filters.jobName.length > 0) {
    filtered = filtered.filter((run) => filters.jobName!.includes(run.jobName));
  }

  if (filters?.search) {
    const searchLower = filters.search.toLowerCase();
    filtered = filtered.filter(
      (run) =>
        run.jobName.toLowerCase().includes(searchLower) ||
        (run.error && run.error.toLowerCase().includes(searchLower))
    );
  }

  return filtered;
}

/**
 * Get a single job run by ID
 */
export function getJobRunMock(id: number): JobRun | undefined {
  return normalDataset.find((run) => run.id === id);
}

/**
 * Get job definitions with deterministic last run info
 */
export function getJobDefinitionsMock(): JobDefinition[] {
  return jobDefinitions.map((def, index) => {
    const lastRun = normalDataset.find((run) => run.jobName === def.name);
    // Deterministic next run time based on index
    const nextRunAt = def.enabled
      ? new Date(BASE_TIMESTAMP + (index + 1) * 60000).toISOString()
      : undefined;

    return {
      ...def,
      lastRun,
      nextRunAt,
    };
  });
}

// Legacy exports for backwards compatibility
export function createMockJobRun(overrides?: Partial<JobRun>): JobRun {
  return { ...createDeterministicJobRun(0), ...overrides };
}

export function createMockJobRuns(count: number): JobRun[] {
  return createDeterministicJobRuns(count);
}
