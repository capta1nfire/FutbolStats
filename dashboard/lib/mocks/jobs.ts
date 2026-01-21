import {
  JobRun,
  JobDefinition,
  JobStatus,
  JobFilters,
  JOB_NAMES,
} from "@/lib/types";

/**
 * Create a mock job run
 */
export function createMockJobRun(overrides?: Partial<JobRun>): JobRun {
  const statuses: JobStatus[] = ["success", "success", "success", "failed", "running"];
  const status = statuses[Math.floor(Math.random() * statuses.length)];
  const jobNames = [...JOB_NAMES];
  const jobName = jobNames[Math.floor(Math.random() * jobNames.length)];

  const now = new Date();
  const startedAt = new Date(now.getTime() - Math.random() * 3600000); // Within last hour
  const durationMs = status === "running" ? undefined : Math.floor(Math.random() * 30000) + 500;
  const finishedAt =
    status === "running"
      ? undefined
      : new Date(startedAt.getTime() + (durationMs || 0)).toISOString();

  return {
    id: Math.floor(Math.random() * 100000),
    jobName,
    status,
    startedAt: startedAt.toISOString(),
    finishedAt,
    durationMs,
    triggeredBy: Math.random() > 0.8 ? "manual" : "scheduler",
    error: status === "failed" ? "Connection timeout to API-Football" : undefined,
    ...overrides,
  };
}

/**
 * Create multiple mock job runs
 */
export function createMockJobRuns(count: number): JobRun[] {
  const runs: JobRun[] = [];
  const now = new Date();

  for (let i = 0; i < count; i++) {
    const startedAt = new Date(now.getTime() - i * 120000 - Math.random() * 60000);
    runs.push(
      createMockJobRun({
        id: 100000 - i,
        startedAt: startedAt.toISOString(),
      })
    );
  }

  return runs.sort(
    (a, b) => new Date(b.startedAt).getTime() - new Date(a.startedAt).getTime()
  );
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
 * Mock datasets
 */
const normalDataset = createMockJobRuns(30);

const emptyDataset: JobRun[] = [];

const largeDataset = createMockJobRuns(150);

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
 * Get job definitions
 */
export function getJobDefinitionsMock(): JobDefinition[] {
  // Add last run info to definitions
  return jobDefinitions.map((def) => {
    const lastRun = normalDataset.find((run) => run.jobName === def.name);
    const nextRunAt = def.enabled
      ? new Date(Date.now() + Math.random() * 300000).toISOString()
      : undefined;

    return {
      ...def,
      lastRun,
      nextRunAt,
    };
  });
}
