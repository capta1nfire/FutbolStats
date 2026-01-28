"use client";

import { OpsProgress, OpsPitActivity } from "@/lib/api/ops";
import { cn } from "@/lib/utils";
import { TrendingUp, CheckCircle2 } from "lucide-react";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";

interface PitProgressCompactTileProps {
  progress: OpsProgress | null;
  pitActivity: OpsPitActivity | null;
  className?: string;
  isProgressDegraded?: boolean;
  isPitActivityDegraded?: boolean;
}

function getProgressStatus(actual: number, target: number): "ok" | "warning" | "critical" {
  const pct = target > 0 ? (actual / target) * 100 : 0;
  if (pct >= 100) return "ok";
  if (pct >= 60) return "warning";
  return "critical";
}

const statusColors = {
  ok: "text-[var(--status-success-text)]",
  warning: "text-[var(--status-warning-text)]",
  critical: "text-[var(--status-error-text)]",
};

/**
 * PIT Progress Compact Tile
 *
 * Compact version for 2x2 grid. Shows key progress metrics.
 */
export function PitProgressCompactTile({
  progress,
  pitActivity,
  className,
  isProgressDegraded = false,
  isPitActivityDegraded = false,
}: PitProgressCompactTileProps) {
  const isDegraded = isProgressDegraded && isPitActivityDegraded;

  const snapshotsStatus = progress
    ? getProgressStatus(progress.pit_snapshots_30d, progress.target_pit_snapshots_30d)
    : "critical";
  const betsStatus = progress
    ? getProgressStatus(progress.pit_bets_30d, progress.target_pit_bets_30d)
    : "critical";

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
          <TrendingUp className="h-3.5 w-3.5 text-primary" />
          <h3 className="text-xs font-semibold text-foreground">PIT Progress</h3>
        </div>
        {progress?.ready_for_retest && (
          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger asChild>
                <span className="flex items-center gap-0.5 px-1.5 py-0.5 text-[9px] font-medium rounded-full bg-[var(--status-success-bg)] text-[var(--status-success-text)]">
                  <CheckCircle2 className="h-2.5 w-2.5" />
                  Ready
                </span>
              </TooltipTrigger>
              <TooltipContent side="bottom">
                <p>Ready for retest evaluation</p>
              </TooltipContent>
            </Tooltip>
          </TooltipProvider>
        )}
      </div>

      {/* Progress metrics - compact */}
      {!isProgressDegraded && progress ? (
        <div className="space-y-1.5 flex-1">
          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger asChild>
                <div className="flex items-center justify-between text-xs cursor-default">
                  <span className="text-muted-foreground">Snapshots</span>
                  <span className={cn("font-medium tabular-nums", statusColors[snapshotsStatus])}>
                    {progress.pit_snapshots_30d}/{progress.target_pit_snapshots_30d}
                  </span>
                </div>
              </TooltipTrigger>
              <TooltipContent side="bottom">
                <p>PIT snapshots (30d)</p>
              </TooltipContent>
            </Tooltip>
          </TooltipProvider>

          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger asChild>
                <div className="flex items-center justify-between text-xs cursor-default">
                  <span className="text-muted-foreground">Bets</span>
                  <span className={cn("font-medium tabular-nums", statusColors[betsStatus])}>
                    {progress.pit_bets_30d}/{progress.target_pit_bets_30d}
                  </span>
                </div>
              </TooltipTrigger>
              <TooltipContent side="bottom">
                <p>PIT bets (30d)</p>
              </TooltipContent>
            </Tooltip>
          </TooltipProvider>

          <div className="flex items-center justify-between text-xs">
            <span className="text-muted-foreground">Coverage</span>
            <span className="font-medium tabular-nums text-foreground">
              {progress.baseline_coverage_pct.toFixed(1)}%
            </span>
          </div>
        </div>
      ) : (
        <div className="text-xs text-muted-foreground flex-1">Progress unavailable</div>
      )}

      {/* Activity */}
      {!isPitActivityDegraded && pitActivity && (
        <div className="text-[10px] text-muted-foreground mt-2 pt-2 border-t border-border flex justify-between">
          <span>Live 60m: {pitActivity.live_60m}</span>
          <span>24h: {pitActivity.live_24h}</span>
        </div>
      )}

      {isDegraded && (
        <div className="text-[10px] text-orange-400 mt-2 pt-2 border-t border-border">
          mock
        </div>
      )}
    </div>
  );
}
