"use client";

import { OpsPredictionsHealth } from "@/lib/api/ops";
import { ApiBudgetStatus } from "@/lib/types";
import { cn } from "@/lib/utils";
import { Target } from "lucide-react";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";

interface PredictionsCompactTileProps {
  predictions: OpsPredictionsHealth | null;
  className?: string;
  isMockFallback?: boolean;
}

const statusDot: Record<ApiBudgetStatus, string> = {
  ok: "bg-[var(--status-success-text)]",
  warning: "bg-[var(--status-warning-text)]",
  critical: "bg-[var(--status-error-text)]",
  degraded: "bg-orange-500",
};

function getCoverageColor(pct: number, warnThreshold: number, redThreshold: number): string {
  if (pct >= warnThreshold) return "text-[var(--status-success-text)]";
  if (pct >= redThreshold) return "text-[var(--status-warning-text)]";
  return "text-[var(--status-error-text)]";
}

/**
 * Predictions Compact Tile
 *
 * Compact version for 2x2 grid. Shows key coverage metrics only.
 */
export function PredictionsCompactTile({
  predictions,
  className,
  isMockFallback = false,
}: PredictionsCompactTileProps) {
  const isDegraded = !predictions || isMockFallback;
  const displayStatus: ApiBudgetStatus = predictions?.status ?? "degraded";

  const thresholds = predictions?.thresholds ?? {
    ns_coverage_warn_pct: 80,
    ns_coverage_red_pct: 50,
    ft_coverage_warn_pct: 80,
    ft_coverage_red_pct: 50,
  };

  const nsCoverage = predictions?.ns_coverage_pct ?? 0;
  const ftCoverage = predictions?.ft_coverage_pct ?? 0;

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
          <Target className="h-3.5 w-3.5 text-primary" />
          <h3 className="text-xs font-semibold text-foreground">Predictions</h3>
        </div>
        <span className={cn("h-2 w-2 rounded-full", statusDot[displayStatus])} />
      </div>

      {/* Coverage metrics - compact grid */}
      <div className="grid grid-cols-2 gap-3 flex-1">
        {/* NS (Next 48h) */}
        <TooltipProvider>
          <Tooltip>
            <TooltipTrigger asChild>
              <div className="cursor-default">
                <div className="text-[10px] uppercase tracking-wider text-muted-foreground mb-0.5">
                  Next 48h
                </div>
                <div className={cn(
                  "text-xl font-bold tabular-nums",
                  getCoverageColor(nsCoverage, thresholds.ns_coverage_warn_pct, thresholds.ns_coverage_red_pct)
                )}>
                  {Math.round(nsCoverage)}%
                </div>
              </div>
            </TooltipTrigger>
            <TooltipContent side="bottom">
              <p>{predictions?.ns_matches_next_48h_missing_prediction ?? 0} missing of {predictions?.ns_matches_next_48h ?? 0}</p>
            </TooltipContent>
          </Tooltip>
        </TooltipProvider>

        {/* FT (Last 48h) */}
        <TooltipProvider>
          <Tooltip>
            <TooltipTrigger asChild>
              <div className="cursor-default">
                <div className="text-[10px] uppercase tracking-wider text-muted-foreground mb-0.5">
                  Last 48h
                </div>
                <div className={cn(
                  "text-xl font-bold tabular-nums",
                  getCoverageColor(ftCoverage, thresholds.ft_coverage_warn_pct, thresholds.ft_coverage_red_pct)
                )}>
                  {Math.round(ftCoverage)}%
                </div>
              </div>
            </TooltipTrigger>
            <TooltipContent side="bottom">
              <p>{predictions?.ft_matches_last_48h_missing_prediction ?? 0} missing of {predictions?.ft_matches_last_48h ?? 0}</p>
            </TooltipContent>
          </Tooltip>
        </TooltipProvider>
      </div>

      {/* Bottom stats */}
      <div className="text-[10px] text-muted-foreground mt-2 pt-2 border-t border-border flex justify-between">
        <span>24h: {predictions?.predictions_saved_last_24h ?? 0} saved</span>
        {isDegraded && (
          <span className="text-orange-400">mock</span>
        )}
      </div>
    </div>
  );
}
