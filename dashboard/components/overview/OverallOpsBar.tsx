"use client";

import { useRouter } from "next/navigation";
import { ApiBudgetStatus } from "@/lib/types";
import { OpsFreshness, OpsJobsHealth } from "@/lib/api/ops";
import { cn } from "@/lib/utils";
import { Activity, RefreshCw, AlertTriangle } from "lucide-react";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";

interface OverallOpsBarProps {
  /** Status of each domain for rollup */
  statuses: {
    jobs: ApiBudgetStatus | null;
    predictions: ApiBudgetStatus | null;
    fastpath: ApiBudgetStatus | null;
    budget: ApiBudgetStatus | null;
    sentry: ApiBudgetStatus | null;
    llmCost: ApiBudgetStatus | null;
  };
  /** Full jobs health for top_alert display */
  jobs?: OpsJobsHealth | null;
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

const statusColors: Record<ApiBudgetStatus | "unknown", { bg: string; text: string; dot: string }> = {
  ok: { bg: "bg-green-500/15", text: "text-green-400", dot: "bg-green-500" },
  warning: { bg: "bg-yellow-500/15", text: "text-yellow-400", dot: "bg-yellow-500" },
  critical: { bg: "bg-red-500/15", text: "text-red-400", dot: "bg-red-500" },
  degraded: { bg: "bg-orange-500/15", text: "text-orange-400", dot: "bg-orange-500" },
  unknown: { bg: "bg-muted", text: "text-muted-foreground", dot: "bg-muted-foreground" },
};

const statusLabels: Record<ApiBudgetStatus | "unknown", string> = {
  ok: "All OK",
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
 * Format relative time for freshness
 */
function formatFreshness(seconds: number): string {
  if (seconds < 60) return `${Math.round(seconds)}s ago`;
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.floor(minutes / 60);
  return `${hours}h ${minutes % 60}m ago`;
}

/**
 * Format minutes since success
 */
function formatMinutes(minutes: number | null): string {
  if (minutes === null) return "unknown";
  if (minutes < 60) return `${Math.round(minutes)}m`;
  const hours = Math.floor(minutes / 60);
  const remainingMinutes = Math.round(minutes % 60);
  if (hours < 24) return `${hours}h ${remainingMinutes}m`;
  const days = Math.floor(hours / 24);
  return `${days}d ${hours % 24}h`;
}

/**
 * Overall Ops Bar (Compact)
 *
 * Horizontal bar showing aggregated system status with domain chips.
 * Height: ≤48px for above-the-fold optimization.
 */
export function OverallOpsBar({
  statuses,
  jobs,
  freshness,
  className,
  onRefresh,
}: OverallOpsBarProps) {
  const router = useRouter();
  const overallStatus = getOverallStatus(statuses);
  const colors = statusColors[overallStatus];

  // Count healthy domains
  const healthyCount = Object.values(statuses).filter((s) => s === "ok").length;
  const totalDomains = Object.keys(statuses).length;

  return (
    <div
      className={cn(
        "h-12 bg-surface border border-border rounded-lg px-4 flex items-center gap-4",
        className
      )}
    >
      {/* Status indicator + label */}
      <div className="flex items-center gap-2 shrink-0">
        <Activity className="h-4 w-4 text-primary" />
        <span className={cn("h-2.5 w-2.5 rounded-full", colors.dot)} />
        <span className={cn("text-sm font-medium", colors.text)}>
          {statusLabels[overallStatus]}
        </span>
        <span className="text-xs text-muted-foreground">
          ({healthyCount}/{totalDomains})
        </span>
      </div>

      {/* Divider */}
      <div className="h-5 w-px bg-border shrink-0" />

      {/* Domain chips */}
      <div className="flex items-center gap-1.5 flex-1 overflow-x-auto">
        <TooltipProvider delayDuration={200}>
          {Object.entries(statuses).map(([domain, status]) => {
            const chipColors = status ? statusColors[status] : statusColors.unknown;
            const displayName = domain.charAt(0).toUpperCase() + domain.slice(1);

            // Special handling for Jobs with top_alert
            if (domain === "jobs" && jobs?.top_alert) {
              const alert = jobs.top_alert;
              const alertColors = alert.severity === "red" ? statusColors.critical : statusColors.warning;
              const additionalAlerts = (jobs.alerts_count ?? 1) - 1;

              {
                const chipEl = (
                  <span
                    className={cn(
                      "inline-flex items-center gap-1.5 px-2 py-1 rounded-full text-[11px] font-medium border shrink-0",
                      alertColors.bg,
                      alert.severity === "red" ? "border-red-500/30" : "border-yellow-500/30",
                    )}
                  >
                    <AlertTriangle className={cn("h-3 w-3", alertColors.text)} />
                    <span className={alertColors.text}>
                      Jobs: {alert.label}
                      {additionalAlerts > 0 && ` +${additionalAlerts}`}
                    </span>
                  </span>
                );

                const wrappedChip = alert.incident_id ? (
                  <button
                    key={domain}
                    type="button"
                    className="cursor-pointer hover:opacity-80 transition-opacity shrink-0"
                    onClick={() => router.push(`/incidents?id=${alert.incident_id}`)}
                  >
                    <Tooltip>
                      <TooltipTrigger asChild>{chipEl}</TooltipTrigger>
                      <TooltipContent side="bottom" className="max-w-xs">
                        <div className="space-y-1">
                          <p className="font-medium">{alert.label}</p>
                          <p className="text-muted-foreground">{alert.reason}</p>
                          {alert.minutes_since_success !== null && (
                            <p className="text-muted-foreground">
                              Last success: {formatMinutes(alert.minutes_since_success)} ago
                            </p>
                          )}
                          <p className="text-xs text-primary">Click to view incident details</p>
                        </div>
                      </TooltipContent>
                    </Tooltip>
                  </button>
                ) : (
                  <Tooltip key={domain}>
                    <TooltipTrigger asChild>{chipEl}</TooltipTrigger>
                    <TooltipContent side="bottom" className="max-w-xs">
                      <div className="space-y-1">
                        <p className="font-medium">{alert.label}</p>
                        <p className="text-muted-foreground">{alert.reason}</p>
                        {alert.minutes_since_success !== null && (
                          <p className="text-muted-foreground">
                            Last success: {formatMinutes(alert.minutes_since_success)} ago
                          </p>
                        )}
                      </div>
                    </TooltipContent>
                  </Tooltip>
                );

                return wrappedChip;
              }
            }

            // Default chip for other domains or Jobs OK
            // For Jobs when OK, show explicit "Jobs OK" label
            const chipLabel = domain === "jobs" && status === "ok" ? "Jobs OK" : displayName;

            return (
              <Tooltip key={domain}>
                <TooltipTrigger asChild>
                  <div
                    className={cn(
                      "flex items-center gap-1.5 px-2 py-1 rounded-full text-[11px] font-medium border border-transparent cursor-default shrink-0",
                      chipColors.bg
                    )}
                  >
                    <span className={cn("h-1.5 w-1.5 rounded-full", chipColors.dot)} />
                    <span className={chipColors.text}>{chipLabel}</span>
                  </div>
                </TooltipTrigger>
                <TooltipContent side="bottom">
                  <p>{displayName}: {status ?? "unknown"}</p>
                </TooltipContent>
              </Tooltip>
            );
          })}
        </TooltipProvider>
      </div>

      {/* Freshness + Refresh */}
      <div className="flex items-center gap-3 shrink-0">
        {freshness && (
          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger asChild>
                <span className={cn(
                  "text-xs tabular-nums cursor-default",
                  freshness.is_stale ? "text-yellow-400" : "text-muted-foreground"
                )}>
                  {formatFreshness(freshness.cache_age_seconds)}
                  {freshness.is_stale && " (stale)"}
                </span>
              </TooltipTrigger>
              <TooltipContent side="bottom">
                <p>Updated: {freshness.generated_at}</p>
              </TooltipContent>
            </Tooltip>
          </TooltipProvider>
        )}
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
    </div>
  );
}
