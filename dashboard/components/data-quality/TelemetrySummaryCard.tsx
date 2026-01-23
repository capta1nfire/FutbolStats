"use client";

import { cn } from "@/lib/utils";
import { OpsTelemetry } from "@/lib/api/ops";
import {
  CheckCircle2,
  AlertTriangle,
  XCircle,
  Clock,
  Database,
  AlertOctagon,
  Link2Off,
} from "lucide-react";

interface TelemetrySummaryCardProps {
  telemetry: OpsTelemetry | null;
  isLoading?: boolean;
  isDegraded?: boolean;
  className?: string;
}

/**
 * Map telemetry status to display config
 */
function getStatusConfig(status: string) {
  switch (status) {
    case "ok":
      return {
        label: "OK",
        icon: CheckCircle2,
        bgClass: "bg-green-500/10",
        borderClass: "border-green-500/30",
        textClass: "text-green-400",
      };
    case "warning":
      return {
        label: "Warning",
        icon: AlertTriangle,
        bgClass: "bg-yellow-500/10",
        borderClass: "border-yellow-500/30",
        textClass: "text-yellow-400",
      };
    case "critical":
    case "red":
      return {
        label: "Critical",
        icon: XCircle,
        bgClass: "bg-red-500/10",
        borderClass: "border-red-500/30",
        textClass: "text-red-400",
      };
    default:
      return {
        label: "Unknown",
        icon: AlertOctagon,
        bgClass: "bg-muted/50",
        borderClass: "border-border",
        textClass: "text-muted-foreground",
      };
  }
}

/**
 * Format relative time
 */
function formatRelativeTime(isoString: string): string {
  const date = new Date(isoString);
  const now = new Date();
  const diffMs = now.getTime() - date.getTime();
  const diffMins = Math.floor(diffMs / 60000);

  if (diffMins < 1) return "just now";
  if (diffMins < 60) return `${diffMins}m ago`;
  const diffHours = Math.floor(diffMins / 60);
  if (diffHours < 24) return `${diffHours}h ago`;
  const diffDays = Math.floor(diffHours / 24);
  return `${diffDays}d ago`;
}

/**
 * Telemetry Summary Card
 *
 * Displays real-time data quality metrics from ops.json telemetry.
 * Shows quarantined odds, tainted matches, unmapped entities, and odds desync.
 */
export function TelemetrySummaryCard({
  telemetry,
  isLoading = false,
  isDegraded = false,
  className,
}: TelemetrySummaryCardProps) {
  if (isLoading) {
    return (
      <div className={cn("rounded-lg border border-border bg-surface p-4", className)}>
        <div className="flex items-center gap-2 mb-3">
          <div className="h-4 w-4 bg-muted animate-pulse rounded" />
          <div className="h-4 w-32 bg-muted animate-pulse rounded" />
        </div>
        <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
          {[...Array(5)].map((_, i) => (
            <div key={i} className="h-16 bg-muted animate-pulse rounded" />
          ))}
        </div>
      </div>
    );
  }

  if (isDegraded || !telemetry) {
    return (
      <div className={cn("rounded-lg border border-border bg-surface p-4", className)}>
        <div className="flex items-center gap-2 text-muted-foreground">
          <AlertOctagon className="h-4 w-4" />
          <span className="text-sm">Telemetry unavailable</span>
        </div>
      </div>
    );
  }

  const statusConfig = getStatusConfig(telemetry.status);
  const StatusIcon = statusConfig.icon;
  const { summary } = telemetry;

  // Determine if any metric is non-zero (indicates potential issues)
  const hasIssues =
    summary.quarantined_odds_24h > 0 ||
    summary.tainted_matches_24h > 0 ||
    summary.unmapped_entities_24h > 0;

  return (
    <div
      className={cn(
        "rounded-lg border p-4",
        statusConfig.bgClass,
        statusConfig.borderClass,
        className
      )}
    >
      {/* Header */}
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <StatusIcon className={cn("h-4 w-4", statusConfig.textClass)} />
          <span className="text-sm font-medium">Data Quality Telemetry</span>
          <span className={cn("text-xs font-medium", statusConfig.textClass)}>
            {statusConfig.label}
          </span>
        </div>
        {telemetry.updated_at && (
          <div className="flex items-center gap-1 text-xs text-muted-foreground">
            <Clock className="h-3 w-3" />
            <span>{formatRelativeTime(telemetry.updated_at)}</span>
          </div>
        )}
      </div>

      {/* Metrics Grid */}
      <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
        {/* Quarantined Odds */}
        <MetricItem
          icon={Database}
          label="Quarantined Odds"
          value={summary.quarantined_odds_24h}
          subtitle="24h"
          isWarning={summary.quarantined_odds_24h > 0}
        />

        {/* Tainted Matches */}
        <MetricItem
          icon={AlertOctagon}
          label="Tainted Matches"
          value={summary.tainted_matches_24h}
          subtitle="24h"
          isWarning={summary.tainted_matches_24h > 0}
        />

        {/* Unmapped Entities */}
        <MetricItem
          icon={Link2Off}
          label="Unmapped"
          value={summary.unmapped_entities_24h}
          subtitle="24h"
          isWarning={summary.unmapped_entities_24h > 0}
        />

        {/* Odds Desync 6h */}
        <MetricItem
          icon={AlertTriangle}
          label="Odds Desync"
          value={summary.odds_desync_6h}
          subtitle="6h"
          isWarning={summary.odds_desync_6h > 0}
        />

        {/* Odds Desync 90m */}
        <MetricItem
          icon={AlertTriangle}
          label="Odds Desync"
          value={summary.odds_desync_90m}
          subtitle="90m"
          isWarning={summary.odds_desync_90m > 0}
        />
      </div>

      {/* Summary message */}
      {!hasIssues && (
        <p className="mt-3 text-xs text-muted-foreground">
          All data quality checks passing. No quarantined or tainted data.
        </p>
      )}
    </div>
  );
}

/**
 * Individual metric item
 */
function MetricItem({
  icon: Icon,
  label,
  value,
  subtitle,
  isWarning = false,
}: {
  icon: typeof Database;
  label: string;
  value: number;
  subtitle: string;
  isWarning?: boolean;
}) {
  return (
    <div className="flex flex-col gap-1 p-2 rounded bg-background/50">
      <div className="flex items-center gap-1.5">
        <Icon
          className={cn(
            "h-3.5 w-3.5",
            isWarning ? "text-yellow-400" : "text-muted-foreground"
          )}
        />
        <span className="text-xs text-muted-foreground truncate">{label}</span>
      </div>
      <div className="flex items-baseline gap-1">
        <span
          className={cn(
            "text-lg font-semibold",
            isWarning ? "text-yellow-400" : "text-foreground"
          )}
        >
          {value}
        </span>
        <span className="text-xs text-muted-foreground">{subtitle}</span>
      </div>
    </div>
  );
}
