/**
 * Job Types
 *
 * Represents scheduler jobs and their run history
 */

export type JobStatus = "running" | "success" | "failed" | "pending";

export interface JobRun {
  id: number;
  jobName: string;
  status: JobStatus;
  startedAt: string; // ISO timestamp
  finishedAt?: string; // ISO timestamp (null if running)
  durationMs?: number;
  triggeredBy: "scheduler" | "manual" | "retry";
  error?: string; // Error message if failed
  metadata?: Record<string, unknown>;
}

export interface JobDefinition {
  name: string;
  description: string;
  schedule: string; // Cron expression or human-readable
  lastRun?: JobRun;
  nextRunAt?: string; // ISO timestamp
  enabled: boolean;
}

export interface JobFilters {
  status?: JobStatus[];
  jobName?: string[];
  search?: string;
  dateRange?: {
    start: string;
    end: string;
  };
}

/**
 * Job names matching FutbolStats scheduler
 */
export const JOB_NAMES = [
  "global_sync",
  "live_tick",
  "stats_backfill",
  "odds_sync",
  "fastpath",
  "narrative_generator",
] as const;

export type JobName = (typeof JOB_NAMES)[number];
