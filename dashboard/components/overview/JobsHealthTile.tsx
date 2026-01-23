"use client";

import { OpsJobsHealth, OpsJobItem } from "@/lib/api/ops";
import { ApiBudgetStatus } from "@/lib/types";
import { cn } from "@/lib/utils";
import { Clock, ExternalLink } from "lucide-react";

interface JobsHealthTileProps {
  jobs: OpsJobsHealth | null;
  className?: string;
  isMockFallback?: boolean;
}

const statusColors: Record<ApiBudgetStatus, string> = {
  ok: "text-green-400",
  warning: "text-yellow-400",
  critical: "text-red-400",
  degraded: "text-orange-400",
};

const statusDot: Record<ApiBudgetStatus, string> = {
  ok: "bg-green-500",
  warning: "bg-yellow-500",
  critical: "bg-red-500",
  degraded: "bg-orange-500",
};

/**
 * Format minutes since success
 */
function formatMinutes(minutes: number | undefined): string {
  if (minutes === undefined) return "—";
  if (minutes < 1) return "<1m";
  if (minutes < 60) return `${Math.round(minutes)}m`;
  const hours = Math.floor(minutes / 60);
  const mins = Math.round(minutes % 60);
  if (hours < 24) return `${hours}h ${mins}m`;
  const days = Math.floor(hours / 24);
  return `${days}d ${hours % 24}h`;
}

/**
 * Individual job row
 */
function JobRow({ name, job }: { name: string; job: OpsJobItem | null }) {
  if (!job) {
    return (
      <div className="flex items-center justify-between py-1.5">
        <span className="text-xs text-muted-foreground">{name}</span>
        <span className="text-xs text-muted-foreground/50">—</span>
      </div>
    );
  }

  return (
    <div className="flex items-center justify-between py-1.5">
      <div className="flex items-center gap-2">
        <span className={cn("h-2 w-2 rounded-full", statusDot[job.status])} />
        <span className="text-xs text-foreground">{name}</span>
      </div>
      <div className="flex items-center gap-2">
        <span className={cn("text-xs tabular-nums", statusColors[job.status])}>
          {formatMinutes(job.minutes_since_success)}
        </span>
        {job.help_url && (
          <a
            href={job.help_url}
            target="_blank"
            rel="noopener noreferrer"
            className="text-muted-foreground hover:text-primary transition-colors"
            aria-label={`Help for ${name}`}
          >
            <ExternalLink className="h-3 w-3" />
          </a>
        )}
      </div>
    </div>
  );
}

/**
 * Jobs Health Tile
 *
 * Displays scheduler jobs health using backend-calculated status.
 * NO false positives - uses status from backend, not inferred lateness.
 */
export function JobsHealthTile({
  jobs,
  className,
  isMockFallback = false,
}: JobsHealthTileProps) {
  const isDegraded = !jobs || isMockFallback;
  const displayStatus: ApiBudgetStatus = jobs?.status ?? "degraded";

  // Count healthy jobs
  const healthyCount = [jobs?.stats_backfill, jobs?.odds_sync, jobs?.fastpath]
    .filter((j) => j?.status === "ok")
    .length;

  return (
    <div
      className={cn(
        "bg-surface border border-border rounded-lg p-4",
        className
      )}
    >
      {/* Header */}
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <Clock className="h-4 w-4 text-primary" />
          <h3 className="text-sm font-semibold text-foreground">Jobs</h3>
          {jobs?.runbook_url && (
            <a
              href={jobs.runbook_url}
              target="_blank"
              rel="noopener noreferrer"
              className="text-muted-foreground hover:text-primary transition-colors"
              aria-label="Jobs runbook"
            >
              <ExternalLink className="h-3 w-3" />
            </a>
          )}
        </div>
        <div className="flex items-center gap-2">
          <span className="text-xs text-muted-foreground">
            {healthyCount}/3 OK
          </span>
          {displayStatus === "ok" ? (
            <span className="h-2.5 w-2.5 rounded-full bg-green-500" title="OK" />
          ) : (
            <span className={cn("h-2.5 w-2.5 rounded-full", statusDot[displayStatus])} />
          )}
        </div>
      </div>

      {/* Jobs list */}
      <div className="divide-y divide-border">
        <JobRow name="Stats Backfill" job={jobs?.stats_backfill ?? null} />
        <JobRow name="Odds Sync" job={jobs?.odds_sync ?? null} />
        <JobRow name="Fastpath" job={jobs?.fastpath ?? null} />
      </div>

      {/* Degraded indicator */}
      {isDegraded && (
        <div className="mt-3 pt-3 border-t border-border">
          <span className="px-1.5 py-0.5 text-[10px] font-medium rounded bg-muted text-muted-foreground border border-border">
            Degraded (mock)
          </span>
        </div>
      )}
    </div>
  );
}
