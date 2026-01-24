"use client";

import { OverviewTab } from "@/lib/overview-drawer";
import { Activity, Clock, RefreshCw } from "lucide-react";
import { useOpsOverview } from "@/lib/hooks";
import { Loader } from "@/components/ui/loader";
import { ApiBudgetStatus } from "@/lib/types";

interface OverviewDrawerOverallProps {
  tab: OverviewTab;
}

/**
 * Overall panel content for overview drawer
 *
 * Tabs:
 * - summary: Rollup status from ops/rollup
 */
// eslint-disable-next-line @typescript-eslint/no-unused-vars -- tab prop reserved for future use
export function OverviewDrawerOverall({ tab }: OverviewDrawerOverallProps) {
  // Only summary tab for now
  return <OverallSummaryTab />;
}

const statusColors: Record<ApiBudgetStatus | "unknown", string> = {
  ok: "text-green-400 bg-green-500/10",
  warning: "text-yellow-400 bg-yellow-500/10",
  critical: "text-red-400 bg-red-500/10",
  degraded: "text-orange-400 bg-orange-500/10",
  unknown: "text-muted-foreground bg-muted",
};

const statusLabels: Record<ApiBudgetStatus | "unknown", string> = {
  ok: "All Systems OK",
  warning: "Warning",
  critical: "Critical",
  degraded: "Degraded",
  unknown: "Unknown",
};

/**
 * Summary tab - overall system status
 */
function OverallSummaryTab() {
  const {
    budget,
    sentry,
    llmCost,
    jobs,
    fastpath,
    predictions,
    freshness,
    isLoading,
    refetch,
  } = useOpsOverview();

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-32">
        <Loader size="sm" />
      </div>
    );
  }

  // Build domain statuses
  const domainStatuses: { name: string; status: ApiBudgetStatus | "unknown" }[] = [
    { name: "Jobs", status: jobs?.status ?? "unknown" },
    { name: "Predictions", status: predictions?.status ?? "unknown" },
    { name: "Fastpath", status: fastpath?.status ?? "unknown" },
    { name: "Budget", status: budget?.status ?? "unknown" },
    { name: "Sentry", status: sentry?.status ?? "unknown" },
    { name: "LLM Cost", status: llmCost?.status ?? "unknown" },
  ];

  // Calculate overall status (worst of all)
  const statusPriority: Record<ApiBudgetStatus | "unknown", number> = {
    critical: 4,
    warning: 3,
    degraded: 2,
    unknown: 1,
    ok: 0,
  };

  const overallStatus = domainStatuses.reduce<ApiBudgetStatus | "unknown">(
    (worst, { status }) => {
      if (statusPriority[status] > statusPriority[worst]) {
        return status;
      }
      return worst;
    },
    "ok"
  );

  const healthyCount = domainStatuses.filter((d) => d.status === "ok").length;

  return (
    <div className="p-4 space-y-4">
      {/* Overall Status */}
      <div className={`rounded-lg p-4 ${statusColors[overallStatus]}`}>
        <div className="flex items-center gap-2 mb-1">
          <Activity className="h-5 w-5" />
          <span className="font-semibold">{statusLabels[overallStatus]}</span>
        </div>
        <p className="text-sm opacity-80">
          {healthyCount}/{domainStatuses.length} systems healthy
        </p>
      </div>

      {/* Domain Statuses */}
      <div className="space-y-2">
        <h3 className="text-xs font-medium text-muted-foreground uppercase tracking-wider">
          Domain Status
        </h3>
        {domainStatuses.map(({ name, status }) => (
          <div key={name} className="flex items-center justify-between py-1.5">
            <span className="text-sm text-foreground">{name}</span>
            <span
              className={`text-xs font-medium px-2 py-0.5 rounded-full ${statusColors[status]}`}
            >
              {status.charAt(0).toUpperCase() + status.slice(1)}
            </span>
          </div>
        ))}
      </div>

      {/* Freshness */}
      {freshness && (
        <div className="pt-3 border-t border-border">
          <div className="flex items-center justify-between">
            <span className="text-sm text-muted-foreground flex items-center gap-1.5">
              <Clock className="h-3.5 w-3.5" />
              Last Updated
            </span>
            <span className="text-sm text-foreground">
              {freshness.cache_age_seconds < 60
                ? `${Math.round(freshness.cache_age_seconds)}s ago`
                : `${Math.floor(freshness.cache_age_seconds / 60)}m ago`}
            </span>
          </div>
          {freshness.is_stale && (
            <p className="text-xs text-yellow-400 mt-1">Data may be stale</p>
          )}
        </div>
      )}

      {/* Refresh Button */}
      <button
        onClick={() => refetch()}
        className="w-full flex items-center justify-center gap-2 py-2 text-sm text-primary hover:bg-muted rounded-lg transition-colors"
      >
        <RefreshCw className="h-4 w-4" />
        Refresh Data
      </button>
    </div>
  );
}
