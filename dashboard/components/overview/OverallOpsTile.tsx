"use client";

import { ApiBudgetStatus } from "@/lib/types";
import { OpsFreshness } from "@/lib/api/ops";
import { cn } from "@/lib/utils";
import { Activity, RefreshCw } from "lucide-react";

interface OverallOpsTileProps {
  /** Status of each domain for rollup */
  statuses: {
    jobs: ApiBudgetStatus | null;
    predictions: ApiBudgetStatus | null;
    fastpath: ApiBudgetStatus | null;
    budget: ApiBudgetStatus | null;
    sentry: ApiBudgetStatus | null;
    llmCost: ApiBudgetStatus | null;
  };
  freshness: OpsFreshness | null;
  className?: string;
  onRefresh?: () => void;
}

const statusPriority: Record<ApiBudgetStatus, number> = {
  critical: 3,
  warning: 2,
  degraded: 1,
  ok: 0,
};

/**
 * Status colors for domain cards
 *
 * Note: We use higher opacity backgrounds (25-30%) because the parent container
 * has a dark surface background (#1c1e21). Lower opacities result in nearly
 * invisible colors that blend with the background.
 */
const statusColors: Record<ApiBudgetStatus | "unknown", { bg: string; text: string; border: string }> = {
  ok: { bg: "bg-green-900/50", text: "text-green-400", border: "border-green-500/60" },
  warning: { bg: "bg-yellow-900/50", text: "text-yellow-300", border: "border-yellow-500/60" },
  critical: { bg: "bg-red-900/50", text: "text-red-400", border: "border-red-500/60" },
  degraded: { bg: "bg-orange-900/50", text: "text-orange-300", border: "border-orange-500/60" },
  unknown: { bg: "bg-muted", text: "text-muted-foreground", border: "border-border" },
};

const statusLabels: Record<ApiBudgetStatus | "unknown", string> = {
  ok: "All Systems OK",
  warning: "Warning",
  critical: "Critical",
  degraded: "Degraded",
  unknown: "Unknown",
};

/**
 * Calculate worst-of status from multiple domains
 */
function getOverallStatus(statuses: Record<string, ApiBudgetStatus | null>): ApiBudgetStatus | "unknown" {
  let worst: ApiBudgetStatus | "unknown" = "unknown";
  let worstPriority = -1;

  for (const status of Object.values(statuses)) {
    if (status !== null) {
      const priority = statusPriority[status];
      if (worst === "unknown" || priority > worstPriority) {
        worst = status;
        worstPriority = priority;
      }
    }
  }

  return worst;
}

/**
 * Format UTC timestamp for display
 */
function formatUtcTime(isoString: string): string {
  try {
    const date = new Date(isoString);
    return date.toISOString().replace("T", " ").slice(0, 19) + " UTC";
  } catch {
    return isoString;
  }
}

/**
 * Overall Ops Tile
 *
 * Shows aggregated system status with freshness indicator.
 * Rollup = worst-of across production domains (excludes diagnostics).
 */
export function OverallOpsTile({
  statuses,
  freshness,
  className,
  onRefresh,
}: OverallOpsTileProps) {
  const overallStatus = getOverallStatus(statuses);
  const colors = statusColors[overallStatus];

  // Count domains by status
  const statusCounts = {
    ok: 0,
    warning: 0,
    critical: 0,
    degraded: 0,
    unknown: 0,
  };

  for (const status of Object.values(statuses)) {
    if (status === null) {
      statusCounts.unknown++;
    } else {
      statusCounts[status]++;
    }
  }

  const totalDomains = Object.keys(statuses).length;
  const healthyDomains = statusCounts.ok;

  return (
    <div
      className={cn(
        "bg-surface border border-border rounded-lg p-4",
        className
      )}
    >
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <Activity className="h-4 w-4 text-primary" />
          <h3 className="text-sm font-semibold text-foreground">Overall Ops</h3>
        </div>
        {onRefresh && (
          <button
            onClick={onRefresh}
            className="text-muted-foreground hover:text-primary transition-colors p-1"
            aria-label="Refresh"
          >
            <RefreshCw className="h-3.5 w-3.5" />
          </button>
        )}
      </div>

      {/* Main status */}
      <div className="flex items-center gap-3 mb-4">
        <div className={cn(
          "h-12 w-12 rounded-full flex items-center justify-center",
          colors.bg,
          "border",
          colors.border
        )}>
          {overallStatus === "ok" ? (
            <span className="text-lg">✓</span>
          ) : overallStatus === "critical" ? (
            <span className="text-lg">!</span>
          ) : overallStatus === "warning" ? (
            <span className="text-lg">⚠</span>
          ) : (
            <span className="text-lg">?</span>
          )}
        </div>
        <div>
          <div className={cn("text-lg font-semibold", colors.text)}>
            {statusLabels[overallStatus]}
          </div>
          <div className="text-xs text-muted-foreground">
            {healthyDomains}/{totalDomains} domains healthy
          </div>
        </div>
      </div>

      {/* Domain breakdown */}
      <div className="grid grid-cols-3 gap-2 mb-4">
        {Object.entries(statuses).map(([domain, status]) => {
          const domainColors = status ? statusColors[status] : statusColors.unknown;
          return (
            <div
              key={domain}
              className={cn(
                "px-2 py-1 rounded text-center text-[10px] border",
                domainColors.bg,
                domainColors.border
              )}
            >
              <span className={domainColors.text}>
                {domain.charAt(0).toUpperCase() + domain.slice(1)}
              </span>
            </div>
          );
        })}
      </div>

      {/* Freshness */}
      {freshness && (
        <div className="pt-3 border-t border-border space-y-1">
          <div className="flex items-center justify-between text-xs">
            <span className="text-muted-foreground">Updated:</span>
            <span className="text-foreground font-mono text-[11px]">
              {formatUtcTime(freshness.generated_at)}
            </span>
          </div>
          <div className="flex items-center justify-between text-xs">
            <span className="text-muted-foreground">Cache age:</span>
            <div className="flex items-center gap-1.5">
              <span className="text-foreground tabular-nums">
                {freshness.cache_age_seconds.toFixed(1)}s
              </span>
              {freshness.is_stale && (
                <span className="px-1 py-0.5 text-[9px] font-medium rounded bg-yellow-900/50 text-yellow-300 border border-yellow-500/60">
                  stale
                </span>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
