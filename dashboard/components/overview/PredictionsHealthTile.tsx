"use client";

import { OpsPredictionsHealth } from "@/lib/api/ops";
import { ApiBudgetStatus } from "@/lib/types";
import { cn } from "@/lib/utils";
import { Target } from "lucide-react";

interface PredictionsHealthTileProps {
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

/**
 * Format coverage percentage with color
 */
function getCoverageColor(pct: number, warnThreshold: number, redThreshold: number): string {
  if (pct >= warnThreshold) return "text-[var(--status-success-text)]";
  if (pct >= redThreshold) return "text-[var(--status-warning-text)]";
  return "text-[var(--status-error-text)]";
}

/**
 * Predictions Health Tile
 *
 * Displays ML prediction coverage for NS (next scheduled) and FT (finished) matches.
 */
export function PredictionsHealthTile({
  predictions,
  className,
  isMockFallback = false,
}: PredictionsHealthTileProps) {
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
        "bg-surface border border-border rounded-lg p-4",
        className
      )}
    >
      {/* Header */}
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <Target className="h-4 w-4 text-primary" />
          <h3 className="text-sm font-semibold text-foreground">Predictions</h3>
        </div>
        {displayStatus === "ok" ? (
          <span className="h-2.5 w-2.5 rounded-full bg-[var(--status-success-text)]" title="OK" />
        ) : (
          <span className={cn("h-2.5 w-2.5 rounded-full", statusDot[displayStatus])} />
        )}
      </div>

      {/* Status reason */}
      {predictions?.status_reason && (
        <div className="text-xs text-muted-foreground mb-3">
          {predictions.status_reason}
        </div>
      )}

      {/* Coverage grid */}
      <div className="grid grid-cols-2 gap-4 mb-3">
        {/* NS (Next Scheduled) */}
        <div>
          <div className="text-[10px] uppercase tracking-wider text-muted-foreground mb-1">
            Next 48h
          </div>
          <div className={cn(
            "text-2xl font-bold tabular-nums",
            getCoverageColor(nsCoverage, thresholds.ns_coverage_warn_pct, thresholds.ns_coverage_red_pct)
          )}>
            {Math.round(nsCoverage)}%
          </div>
          <div className="text-xs text-muted-foreground">
            {predictions?.ns_matches_next_48h_missing_prediction ?? 0} missing of {predictions?.ns_matches_next_48h ?? 0}
          </div>
        </div>

        {/* FT (Finished) */}
        <div>
          <div className="text-[10px] uppercase tracking-wider text-muted-foreground mb-1">
            Last 48h
          </div>
          <div className={cn(
            "text-2xl font-bold tabular-nums",
            getCoverageColor(ftCoverage, thresholds.ft_coverage_warn_pct, thresholds.ft_coverage_red_pct)
          )}>
            {Math.round(ftCoverage)}%
          </div>
          <div className="text-xs text-muted-foreground">
            {predictions?.ft_matches_last_48h_missing_prediction ?? 0} missing of {predictions?.ft_matches_last_48h ?? 0}
          </div>
        </div>
      </div>

      {/* Stats */}
      <div className="text-xs text-muted-foreground space-y-0.5 pt-3 border-t border-border">
        <div className="flex justify-between">
          <span>Saved today (UTC):</span>
          <span className="text-foreground tabular-nums">{predictions?.predictions_saved_today_utc ?? 0}</span>
        </div>
        <div className="flex justify-between">
          <span>Saved last 24h:</span>
          <span className="text-foreground tabular-nums">{predictions?.predictions_saved_last_24h ?? 0}</span>
        </div>
        {predictions?.hours_since_last_prediction !== undefined && (
          <div className="flex justify-between">
            <span>Last saved:</span>
            <span className="text-foreground tabular-nums">
              {predictions.hours_since_last_prediction < 1
                ? "<1h ago"
                : `${Math.round(predictions.hours_since_last_prediction)}h ago`}
            </span>
          </div>
        )}
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
