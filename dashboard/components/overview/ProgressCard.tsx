"use client";

import { OpsProgress, OpsPitActivity, OpsMovement } from "@/lib/api/ops";
import { cn } from "@/lib/utils";
import { TrendingUp, Activity, ArrowUpDown, CheckCircle2 } from "lucide-react";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";

interface ProgressCardProps {
  progress: OpsProgress | null;
  pitActivity: OpsPitActivity | null;
  movement: OpsMovement | null;
  className?: string;
  isProgressDegraded?: boolean;
  isPitActivityDegraded?: boolean;
  isMovementDegraded?: boolean;
}

/**
 * Format percentage with appropriate precision
 */
function formatPct(value: number): string {
  return `${value.toFixed(1)}%`;
}

/**
 * Get status color based on percentage of target
 */
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
 * Progress Card (compact)
 *
 * Displays PIT evaluation progress, activity, and movement metrics.
 * Designed for Overview left rail or main section.
 */
export function ProgressCard({
  progress,
  pitActivity,
  movement,
  className,
  isProgressDegraded = false,
  isPitActivityDegraded = false,
  isMovementDegraded = false,
}: ProgressCardProps) {
  const allDegraded = isProgressDegraded && isPitActivityDegraded && isMovementDegraded;

  // Progress metrics
  const snapshotsStatus = progress
    ? getProgressStatus(progress.pit_snapshots_30d, progress.target_pit_snapshots_30d)
    : "critical";
  const betsStatus = progress
    ? getProgressStatus(progress.pit_bets_30d, progress.target_pit_bets_30d)
    : "critical";
  const coverageStatus = progress
    ? getProgressStatus(progress.baseline_coverage_pct, progress.target_baseline_coverage_pct)
    : "critical";

  return (
    <div
      className={cn(
        "bg-surface border border-border rounded-lg p-4",
        className
      )}
    >
      {/* Header */}
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-semibold text-foreground flex items-center gap-1.5">
          <TrendingUp className="h-4 w-4 text-primary" />
          PIT Progress
        </h3>
        {progress?.ready_for_retest && (
          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger asChild>
                <span className="flex items-center gap-1 px-2 py-0.5 text-[10px] font-medium rounded-full bg-[var(--status-success-bg)] text-[var(--status-success-text)] border border-[var(--status-success-border)]">
                  <CheckCircle2 className="h-3 w-3" />
                  Ready
                </span>
              </TooltipTrigger>
              <TooltipContent side="top">
                <p>Ready for retest evaluation</p>
              </TooltipContent>
            </Tooltip>
          </TooltipProvider>
        )}
      </div>

      {/* Progress metrics */}
      {!isProgressDegraded && progress ? (
        <div className="space-y-2 mb-3">
          {/* Snapshots */}
          <div className="flex items-center justify-between text-xs">
            <span className="text-muted-foreground">Snapshots (30d)</span>
            <span className={cn("font-medium tabular-nums", statusColors[snapshotsStatus])}>
              {progress.pit_snapshots_30d}/{progress.target_pit_snapshots_30d}
            </span>
          </div>
          {/* Bets */}
          <div className="flex items-center justify-between text-xs">
            <span className="text-muted-foreground">Bets (30d)</span>
            <span className={cn("font-medium tabular-nums", statusColors[betsStatus])}>
              {progress.pit_bets_30d}/{progress.target_pit_bets_30d}
            </span>
          </div>
          {/* Coverage */}
          <div className="flex items-center justify-between text-xs">
            <span className="text-muted-foreground">Baseline Coverage</span>
            <span className={cn("font-medium tabular-nums", statusColors[coverageStatus])}>
              {formatPct(progress.baseline_coverage_pct)} / {formatPct(progress.target_baseline_coverage_pct)}
            </span>
          </div>
        </div>
      ) : (
        <div className="text-xs text-muted-foreground mb-3">Progress data unavailable</div>
      )}

      {/* PIT Activity */}
      {!isPitActivityDegraded && pitActivity && (
        <div className="pt-2 border-t border-border">
          <div className="flex items-center gap-1.5 text-xs text-muted-foreground mb-1.5">
            <Activity className="h-3 w-3" />
            <span>Live Activity</span>
          </div>
          <div className="grid grid-cols-2 gap-2 text-xs">
            <div className="flex justify-between">
              <span className="text-muted-foreground">60m:</span>
              <span className="text-foreground font-medium tabular-nums">{pitActivity.live_60m}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-muted-foreground">24h:</span>
              <span className="text-foreground font-medium tabular-nums">{pitActivity.live_24h}</span>
            </div>
          </div>
        </div>
      )}

      {/* Movement */}
      {!isMovementDegraded && movement && (
        <div className="pt-2 border-t border-border mt-2">
          <div className="flex items-center gap-1.5 text-xs text-muted-foreground mb-1.5">
            <ArrowUpDown className="h-3 w-3" />
            <span>Movement (24h)</span>
          </div>
          <div className="grid grid-cols-2 gap-2 text-xs">
            <div className="flex justify-between">
              <span className="text-muted-foreground">Lineup:</span>
              <span className="text-foreground font-medium tabular-nums">{movement.lineup_movement_24h}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-muted-foreground">Market:</span>
              <span className="text-foreground font-medium tabular-nums">{movement.market_movement_24h}</span>
            </div>
          </div>
        </div>
      )}

      {/* Degraded indicator */}
      {allDegraded && (
        <TooltipProvider>
          <Tooltip>
            <TooltipTrigger asChild>
              <div className="flex items-center gap-2 mt-2 pt-2 border-t border-border cursor-help">
                <span className="px-1.5 py-0.5 text-[10px] font-medium rounded bg-muted text-muted-foreground border border-border">
                  Degraded
                </span>
              </div>
            </TooltipTrigger>
            <TooltipContent side="top">
              <p>Progress data unavailable from backend.</p>
            </TooltipContent>
          </Tooltip>
        </TooltipProvider>
      )}
    </div>
  );
}
