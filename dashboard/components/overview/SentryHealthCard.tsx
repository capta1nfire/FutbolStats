"use client";

import { useMemo } from "react";
import { OpsSentrySummary, SentryIssueLevel } from "@/lib/api/ops";
import { ApiBudgetStatus } from "@/lib/types";
import { cn } from "@/lib/utils";
import { Bug, AlertTriangle, Info, Clock } from "lucide-react";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";

interface SentryHealthCardProps {
  sentry: OpsSentrySummary | null;
  className?: string;
  /** True when showing mock/degraded state */
  isMockFallback?: boolean;
}

const statusColors: Record<ApiBudgetStatus, string> = {
  ok: "bg-green-500/20 text-green-400 border-green-500/30",
  warning: "bg-yellow-500/20 text-yellow-400 border-yellow-500/30",
  critical: "bg-red-500/20 text-red-400 border-red-500/30",
  degraded: "bg-orange-500/20 text-orange-400 border-orange-500/30",
};

const statusLabels: Record<ApiBudgetStatus, string> = {
  ok: "OK",
  warning: "Warning",
  critical: "Critical",
  degraded: "Degraded",
};

const levelIcons: Record<SentryIssueLevel, React.ReactNode> = {
  error: <AlertTriangle className="h-3 w-3 text-red-400" />,
  warning: <AlertTriangle className="h-3 w-3 text-yellow-400" />,
  info: <Info className="h-3 w-3 text-blue-400" />,
};

// Default icon when level is not provided
const defaultLevelIcon = <Bug className="h-3 w-3 text-muted-foreground" />;

/**
 * Format relative time from ISO string
 */
function formatRelativeTime(isoString: string): string {
  const now = Date.now();
  const then = new Date(isoString).getTime();
  const diffMs = now - then;

  if (diffMs < 0) return "just now";

  const minutes = Math.floor(diffMs / 60000);
  if (minutes < 1) return "just now";
  if (minutes < 60) return `${minutes}m ago`;

  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours}h ago`;

  const days = Math.floor(hours / 24);
  return `${days}d ago`;
}

/**
 * Format cache age
 */
function formatCacheAge(seconds: number): string {
  if (seconds < 60) {
    return `${seconds}s ago`;
  }
  const minutes = Math.floor(seconds / 60);
  return `${minutes}m ago`;
}

/**
 * Sentry Health Card
 *
 * Displays Sentry error monitoring summary:
 * - Status pill (ok/warning/critical/degraded)
 * - Issue counts (1h, 24h, open)
 * - Last event relative time
 * - Top issues (2-3)
 * - Cache freshness
 */
export function SentryHealthCard({
  sentry,
  className,
  isMockFallback = false,
}: SentryHealthCardProps) {
  // If no sentry data, show degraded state
  const isDegraded = !sentry || isMockFallback;
  const displayStatus: ApiBudgetStatus = sentry?.status ?? "degraded";
  const isStale = (sentry?.cache_age_seconds ?? 0) > 600; // 10 minutes

  // Format last event time
  const lastEventAt = sentry?.last_event_at;
  const lastEventFormatted = useMemo(() => {
    if (!lastEventAt) return null;
    return formatRelativeTime(lastEventAt);
  }, [lastEventAt]);

  return (
    <div
      className={cn(
        "bg-surface border border-border rounded-lg p-4",
        className
      )}
    >
      {/* Header: Title + Status */}
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-semibold text-foreground flex items-center gap-1.5">
          <Bug className="h-4 w-4 text-primary" />
          Sentry
        </h3>
        {displayStatus === "ok" ? (
          <span className="h-2.5 w-2.5 rounded-full bg-green-500" title="OK" />
        ) : (
          <span
            className={cn(
              "px-2 py-0.5 text-xs font-medium rounded-full border",
              statusColors[displayStatus]
            )}
          >
            {statusLabels[displayStatus]}
          </span>
        )}
      </div>

      {/* Subtitle: Project info */}
      {sentry?.project && (
        <div className="text-xs text-muted-foreground mb-4">
          {sentry.project.project_slug}
          <span className="ml-1 text-muted-foreground/70">
            ({sentry.project.env})
          </span>
        </div>
      )}

      {/* Issue Counts: Active 24h, New 24h, Open */}
      <div className="grid grid-cols-3 gap-2 mb-4">
        <div className="text-center">
          <div className="text-lg font-bold text-foreground tabular-nums">
            {sentry?.counts.active_issues_24h ?? sentry?.counts.new_issues_24h ?? 0}
          </div>
          <div className="text-[10px] text-muted-foreground">Active 24h</div>
        </div>
        <div className="text-center">
          <div className="text-lg font-bold text-foreground tabular-nums">
            {sentry?.counts.new_issues_24h ?? 0}
          </div>
          <div className="text-[10px] text-muted-foreground">New 24h</div>
        </div>
        <div className="text-center">
          <div className="text-lg font-bold text-foreground tabular-nums">
            {sentry?.counts.open_issues ?? 0}
          </div>
          <div className="text-[10px] text-muted-foreground">Open</div>
        </div>
      </div>

      {/* Last Event */}
      {lastEventFormatted && (
        <div className="flex items-center gap-1.5 text-xs text-muted-foreground mb-3">
          <Clock className="h-3 w-3" />
          Last event: <span className="text-foreground">{lastEventFormatted}</span>
        </div>
      )}

      {/* Top Issues */}
      {sentry?.top_issues && sentry.top_issues.length > 0 && (
        <div className="mb-3">
          <div className="text-[10px] uppercase tracking-wider text-muted-foreground mb-1.5">
            Top Issues
          </div>
          <div className="space-y-1">
            {sentry.top_issues.slice(0, 3).map((issue, idx) => (
              <div
                key={idx}
                className="flex items-center gap-1.5 text-xs"
              >
                {issue.level ? levelIcons[issue.level] : defaultLevelIcon}
                <span className="text-muted-foreground truncate flex-1" title={issue.title}>
                  {issue.title.length > 30
                    ? `${issue.title.slice(0, 30)}...`
                    : issue.title}
                </span>
                <span className="text-muted-foreground/70 tabular-nums shrink-0">
                  {issue.count_24h}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Note */}
      {sentry?.note && (
        <div className="text-[10px] text-muted-foreground/70 italic mb-3">
          {sentry.note}
        </div>
      )}

      {/* Cache freshness - only render if cache_age_seconds is available */}
      {sentry && !isDegraded && typeof sentry.cache_age_seconds === "number" && (
        <div className="flex items-center gap-2 pt-3 border-t border-border">
          <span className="text-xs text-muted-foreground">
            Cached: {formatCacheAge(sentry.cache_age_seconds)}
          </span>
          {isStale && (
            <span className="px-1.5 py-0.5 text-[10px] font-medium rounded bg-yellow-500/20 text-yellow-400 border border-yellow-500/30">
              stale
            </span>
          )}
        </div>
      )}

      {/* Degraded/Mock fallback indicator */}
      {isDegraded && (
        <TooltipProvider>
          <Tooltip>
            <TooltipTrigger asChild>
              <div className="flex items-center gap-2 pt-3 border-t border-border cursor-help">
                <span className="px-1.5 py-0.5 text-[10px] font-medium rounded bg-muted text-muted-foreground border border-border">
                  Degraded (mock)
                </span>
              </div>
            </TooltipTrigger>
            <TooltipContent side="top">
              <p>Sentry data unavailable. Backend may not expose this yet.</p>
            </TooltipContent>
          </Tooltip>
        </TooltipProvider>
      )}
    </div>
  );
}
