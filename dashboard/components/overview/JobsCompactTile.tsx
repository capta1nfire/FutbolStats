"use client";

import { OpsJobsHealth, OpsJobItem } from "@/lib/api/ops";
import { ApiBudgetStatus } from "@/lib/types";
import { cn } from "@/lib/utils";
import { Clock } from "lucide-react";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";

interface JobsCompactTileProps {
  jobs: OpsJobsHealth | null;
  className?: string;
  isMockFallback?: boolean;
}

const statusDot: Record<ApiBudgetStatus, string> = {
  ok: "bg-[var(--status-success-text)]",
  warning: "bg-[var(--status-warning-text)]",
  critical: "bg-[var(--status-error-text)]",
  degraded: "bg-orange-500",
};

function formatMinutes(minutes: number | undefined): string {
  if (minutes === undefined) return "—";
  if (minutes < 1) return "<1m";
  if (minutes < 60) return `${Math.round(minutes)}m`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours}h`;
  return `${Math.floor(hours / 24)}d`;
}

function JobDot({ job, name }: { job: OpsJobItem | null; name: string }) {
  const status = job?.status ?? "degraded";
  return (
    <TooltipProvider>
      <Tooltip>
        <TooltipTrigger asChild>
          <div className="flex items-center gap-1.5 cursor-default">
            <span className={cn("h-2 w-2 rounded-full", statusDot[status])} />
            <span className="text-xs text-foreground">{name}</span>
            <span className="text-[10px] text-muted-foreground tabular-nums">
              {formatMinutes(job?.minutes_since_success)}
            </span>
          </div>
        </TooltipTrigger>
        <TooltipContent side="bottom">
          <p>{name}: {status} - {formatMinutes(job?.minutes_since_success)} since success</p>
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  );
}

/**
 * Jobs Compact Tile
 *
 * Compact version for 2x2 grid. Shows job status dots.
 */
export function JobsCompactTile({
  jobs,
  className,
  isMockFallback = false,
}: JobsCompactTileProps) {
  const isDegraded = !jobs || isMockFallback;
  const displayStatus: ApiBudgetStatus = jobs?.status ?? "degraded";

  const healthyCount = [jobs?.stats_backfill, jobs?.odds_sync, jobs?.fastpath]
    .filter((j) => j?.status === "ok")
    .length;

  return (
    <div
      className={cn(
        "bg-surface border border-border rounded-lg p-3 h-full flex flex-col",
        className
      )}
    >
      {/* Header */}
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-1.5">
          <Clock className="h-3.5 w-3.5 text-primary" />
          <h3 className="text-xs font-semibold text-foreground">Jobs</h3>
        </div>
        <div className="flex items-center gap-1.5">
          <span className="text-[10px] text-muted-foreground">{healthyCount}/3</span>
          <span className={cn("h-2 w-2 rounded-full", statusDot[displayStatus])} />
        </div>
      </div>

      {/* Jobs list - compact */}
      <div className="space-y-1.5 flex-1">
        <JobDot name="Backfill" job={jobs?.stats_backfill ?? null} />
        <JobDot name="Odds" job={jobs?.odds_sync ?? null} />
        <JobDot name="Fastpath" job={jobs?.fastpath ?? null} />
      </div>

      {/* Bottom */}
      {isDegraded && (
        <div className="text-[10px] text-orange-400 mt-2 pt-2 border-t border-border">
          mock
        </div>
      )}
    </div>
  );
}
