"use client";

import { useState } from "react";
import { OverviewTab } from "@/lib/overview-drawer";
import { AlertCircle, ExternalLink, Clock, ChevronLeft, ChevronRight } from "lucide-react";
import { useOpsOverview, useSentryIssues, SentryIssuesRange } from "@/lib/hooks";
import { Loader } from "@/components/ui/loader";
import { cn } from "@/lib/utils";

interface OverviewDrawerSentryProps {
  tab: OverviewTab;
}

/**
 * Sentry panel content for overview drawer
 *
 * Tabs:
 * - summary: Status from ops/rollup
 * - issues: Paginated issues list (from /api/sentry/issues)
 */
export function OverviewDrawerSentry({ tab }: OverviewDrawerSentryProps) {
  if (tab === "issues") {
    return <SentryIssuesTab />;
  }

  return <SentrySummaryTab />;
}

/**
 * Summary tab - uses existing ops data
 */
function SentrySummaryTab() {
  const { sentry, isSentryDegraded, isLoading } = useOpsOverview();

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-32">
        <Loader size="sm" />
      </div>
    );
  }

  if (!sentry || isSentryDegraded) {
    return (
      <div className="p-4 text-sm text-muted-foreground">
        Sentry data unavailable
      </div>
    );
  }

  const statusColors = {
    ok: "text-green-400",
    warning: "text-yellow-400",
    critical: "text-red-400",
    degraded: "text-orange-400",
  };

  return (
    <div className="p-4 space-y-4">
      {/* Status */}
      <div className="flex items-center justify-between">
        <span className="text-sm text-muted-foreground">Status</span>
        <span className={`text-sm font-medium ${statusColors[sentry.status]}`}>
          {sentry.status.charAt(0).toUpperCase() + sentry.status.slice(1)}
        </span>
      </div>

      {/* New Issues 1h */}
      <div className="flex items-center justify-between">
        <span className="text-sm text-muted-foreground">New Issues (1h)</span>
        <span className="text-sm font-medium text-foreground tabular-nums">
          {sentry.counts.new_issues_1h}
        </span>
      </div>

      {/* New Issues 24h */}
      <div className="flex items-center justify-between">
        <span className="text-sm text-muted-foreground">New Issues (24h)</span>
        <span className="text-sm font-medium text-foreground tabular-nums">
          {sentry.counts.new_issues_24h}
        </span>
      </div>

      {/* Open Issues */}
      <div className="flex items-center justify-between">
        <span className="text-sm text-muted-foreground">Open Issues</span>
        <span className="text-sm font-medium text-foreground tabular-nums">
          {sentry.counts.open_issues}
        </span>
      </div>

      {/* Last event */}
      {sentry.last_event_at && (
        <div className="flex items-center justify-between">
          <span className="text-sm text-muted-foreground">Last Event</span>
          <span className="text-xs text-muted-foreground flex items-center gap-1">
            <Clock className="h-3 w-3" />
            {new Date(sentry.last_event_at).toLocaleString()}
          </span>
        </div>
      )}

      {/* Project info */}
      {sentry.project && (
        <div className="text-xs text-muted-foreground pt-2 border-t border-border">
          <span>{sentry.project.org_slug}/{sentry.project.project_slug}</span>
          {sentry.project.env && (
            <span className="ml-2 px-1.5 py-0.5 bg-muted rounded text-[10px]">
              {sentry.project.env}
            </span>
          )}
        </div>
      )}
    </div>
  );
}

const rangeOptions: { value: SentryIssuesRange; label: string }[] = [
  { value: "1h", label: "1h" },
  { value: "24h", label: "24h" },
  { value: "7d", label: "7d" },
];

const levelColors = {
  error: "bg-red-500/20 text-red-400 border-red-500/30",
  warning: "bg-yellow-500/20 text-yellow-400 border-yellow-500/30",
  info: "bg-blue-500/20 text-blue-400 border-blue-500/30",
  debug: "bg-muted text-muted-foreground border-border",
};

/**
 * Issues tab - paginated list from /api/sentry/issues
 */
function SentryIssuesTab() {
  const [range, setRange] = useState<SentryIssuesRange>("24h");
  const [page, setPage] = useState(1);
  const limit = 10;

  const { issues, total, hasMore, isLoading, isDegraded } = useSentryIssues({
    range,
    page,
    limit,
  });

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-32">
        <Loader size="sm" />
      </div>
    );
  }

  if (isDegraded) {
    return (
      <div className="p-4">
        <div className="flex flex-col items-center justify-center py-8 text-center">
          <AlertCircle className="h-8 w-8 text-muted-foreground mb-2" />
          <p className="text-sm text-muted-foreground">
            Unable to fetch Sentry issues
          </p>
          <p className="text-xs text-muted-foreground mt-1">
            Backend endpoint unavailable
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="p-4 space-y-4">
      {/* Range filter */}
      <div className="flex items-center justify-between">
        <span className="text-xs text-muted-foreground">Time range</span>
        <div className="flex gap-1">
          {rangeOptions.map((opt) => (
            <button
              key={opt.value}
              onClick={() => {
                setRange(opt.value);
                setPage(1);
              }}
              className={cn(
                "px-2 py-1 text-xs rounded-full transition-colors",
                range === opt.value
                  ? "bg-primary text-primary-foreground"
                  : "text-muted-foreground hover:text-foreground hover:bg-muted"
              )}
            >
              {opt.label}
            </button>
          ))}
        </div>
      </div>

      {/* Issues list */}
      {issues.length === 0 ? (
        <div className="bg-green-500/10 rounded-lg p-4 text-center">
          <p className="text-sm text-green-400">No issues in this time range</p>
        </div>
      ) : (
        <div className="space-y-2">
          {issues.map((issue) => (
            <div
              key={issue.id}
              className={cn(
                "rounded-lg p-3 border",
                levelColors[issue.level] ?? levelColors.info
              )}
            >
              <div className="flex items-start justify-between gap-2">
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-medium text-foreground truncate">
                    {issue.title}
                  </p>
                  <p className="text-xs text-muted-foreground mt-0.5 truncate">
                    {issue.culprit}
                  </p>
                </div>
                <div className="flex items-center gap-2 shrink-0">
                  <span className="text-xs tabular-nums">{issue.count}x</span>
                  {issue.permalink && (
                    <a
                      href={issue.permalink}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-muted-foreground hover:text-foreground"
                    >
                      <ExternalLink className="h-3.5 w-3.5" />
                    </a>
                  )}
                </div>
              </div>
              <div className="flex items-center gap-2 mt-1.5 text-[10px] text-muted-foreground">
                <span className="flex items-center gap-1">
                  <Clock className="h-3 w-3" />
                  {new Date(issue.lastSeen).toLocaleString()}
                </span>
                <span className="px-1.5 py-0.5 rounded bg-muted/50">
                  {issue.shortId}
                </span>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Pagination */}
      {total > limit && (
        <div className="flex items-center justify-between pt-2 border-t border-border">
          <span className="text-xs text-muted-foreground">
            {(page - 1) * limit + 1}-{Math.min(page * limit, total)} of {total}
          </span>
          <div className="flex gap-1">
            <button
              onClick={() => setPage((p) => Math.max(1, p - 1))}
              disabled={page === 1}
              className="p-1 rounded text-muted-foreground hover:text-foreground disabled:opacity-50"
            >
              <ChevronLeft className="h-4 w-4" />
            </button>
            <button
              onClick={() => setPage((p) => p + 1)}
              disabled={!hasMore}
              className="p-1 rounded text-muted-foreground hover:text-foreground disabled:opacity-50"
            >
              <ChevronRight className="h-4 w-4" />
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
