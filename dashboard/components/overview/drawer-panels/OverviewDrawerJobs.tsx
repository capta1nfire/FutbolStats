"use client";

import { OverviewTab } from "@/lib/overview-drawer";
import { AlertCircle, Clock, ExternalLink, Briefcase, CheckCircle2, XCircle } from "lucide-react";
import { useOpsOverview } from "@/lib/hooks";
import { Loader } from "@/components/ui/loader";
import { cn } from "@/lib/utils";

interface OverviewDrawerJobsProps {
  tab: OverviewTab;
}

/**
 * Jobs panel content for overview drawer
 *
 * Tabs:
 * - summary: Jobs health from ops
 * - runs: Job runs history (placeholder)
 * - links: Runbook links
 */
export function OverviewDrawerJobs({ tab }: OverviewDrawerJobsProps) {
  if (tab === "runs") {
    return <JobsRunsTab />;
  }
  if (tab === "links") {
    return <JobsLinksTab />;
  }
  return <JobsSummaryTab />;
}

/**
 * Summary tab - uses existing ops data
 */
function JobsSummaryTab() {
  const { jobs, isJobsDegraded, isLoading } = useOpsOverview();

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-32">
        <Loader size="sm" />
      </div>
    );
  }

  if (!jobs || isJobsDegraded) {
    return (
      <div className="p-4 text-sm text-muted-foreground">
        Jobs data unavailable
      </div>
    );
  }

  const statusColors = {
    ok: "text-[var(--status-success-text)]",
    warning: "text-[var(--status-warning-text)]",
    critical: "text-[var(--status-error-text)]",
    degraded: "text-orange-400",
  };

  const jobItems = [
    { key: "stats_backfill", label: "Stats Backfill", data: jobs.stats_backfill },
    { key: "odds_sync", label: "Odds Sync", data: jobs.odds_sync },
    { key: "fastpath", label: "Fastpath", data: jobs.fastpath },
  ];

  return (
    <div className="p-4 space-y-4">
      {/* Overall Status */}
      <div className="flex items-center justify-between">
        <span className="text-sm text-muted-foreground">Status</span>
        <span className={`text-sm font-medium ${statusColors[jobs.status]}`}>
          {jobs.status.charAt(0).toUpperCase() + jobs.status.slice(1)}
        </span>
      </div>

      {/* Top Alert */}
      {jobs.top_alert && (
        <div className={cn(
          "rounded-lg p-3 border",
          jobs.top_alert.severity === "red"
            ? "bg-[var(--status-error-bg)] border-[var(--status-error-border)]"
            : "bg-[var(--status-warning-bg)] border-[var(--status-warning-border)]"
        )}>
          <div className="flex items-start gap-2">
            <AlertCircle className={cn(
              "h-4 w-4 mt-0.5",
              jobs.top_alert.severity === "red" ? "text-[var(--status-error-text)]" : "text-[var(--status-warning-text)]"
            )} />
            <div>
              <p className="text-sm font-medium text-foreground">
                {jobs.top_alert.label}
              </p>
              <p className="text-xs text-muted-foreground mt-0.5">
                {jobs.top_alert.reason}
              </p>
              {jobs.top_alert.minutes_since_success !== null && (
                <p className="text-xs text-muted-foreground mt-1 flex items-center gap-1">
                  <Clock className="h-3 w-3" />
                  Last success: {Math.round(jobs.top_alert.minutes_since_success)}m ago
                </p>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Job Items */}
      <div className="space-y-2">
        <h3 className="text-xs font-medium text-muted-foreground uppercase tracking-wider">
          Jobs
        </h3>
        {jobItems.map(({ key, label, data }) => (
          <div key={key} className="flex items-center justify-between py-2 border-b border-border last:border-0">
            <div className="flex items-center gap-2">
              <Briefcase className="h-4 w-4 text-muted-foreground" />
              <span className="text-sm text-foreground">{label}</span>
            </div>
            {data ? (
              <div className="flex items-center gap-2">
                {data.minutes_since_success !== undefined && (
                  <span className="text-xs text-muted-foreground">
                    {data.minutes_since_success < 60
                      ? `${Math.round(data.minutes_since_success)}m ago`
                      : `${Math.floor(data.minutes_since_success / 60)}h ago`}
                  </span>
                )}
                {data.status === "ok" ? (
                  <CheckCircle2 className="h-4 w-4 text-[var(--status-success-text)]" />
                ) : (
                  <XCircle className="h-4 w-4 text-[var(--status-warning-text)]" />
                )}
              </div>
            ) : (
              <span className="text-xs text-muted-foreground">N/A</span>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}

/**
 * Runs tab - job runs history
 */
function JobsRunsTab() {
  return (
    <div className="p-4">
      <div className="flex flex-col items-center justify-center py-8 text-center">
        <AlertCircle className="h-8 w-8 text-muted-foreground mb-2" />
        <p className="text-sm text-muted-foreground">
          Job runs history coming soon
        </p>
      </div>
    </div>
  );
}

/**
 * Links tab - runbook and documentation
 */
function JobsLinksTab() {
  const { jobs } = useOpsOverview();

  const links = [
    { label: "Jobs Runbook", url: jobs?.runbook_url },
    { label: "Scheduler Documentation", url: "/docs/OPS_RUNBOOK.md" },
  ].filter((l) => l.url);

  return (
    <div className="p-4 space-y-2">
      <h3 className="text-xs font-medium text-muted-foreground uppercase tracking-wider mb-3">
        Documentation
      </h3>
      {links.length > 0 ? (
        links.map(({ label, url }) => (
          <a
            key={url}
            href={url}
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center justify-between py-2 px-3 rounded-lg hover:bg-muted transition-colors"
          >
            <span className="text-sm text-foreground">{label}</span>
            <ExternalLink className="h-4 w-4 text-muted-foreground" />
          </a>
        ))
      ) : (
        <p className="text-sm text-muted-foreground">No links available</p>
      )}
    </div>
  );
}
