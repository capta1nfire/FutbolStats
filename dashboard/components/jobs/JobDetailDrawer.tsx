"use client";

import { useState } from "react";
import { JobRun } from "@/lib/types";
import { useIsDesktop } from "@/lib/hooks";
import { DetailDrawer } from "@/components/shell";
import {
  Sheet,
  SheetContent,
  SheetHeader,
  SheetTitle,
} from "@/components/ui/sheet";
import { IconTabs } from "@/components/ui/icon-tabs";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import { JobStatusBadge } from "./JobStatusBadge";
import { formatDistanceToNow } from "@/lib/utils";
import { Clock, Terminal, AlertCircle, Info } from "lucide-react";

/** Tab definitions for job detail drawer */
const JOB_TABS = [
  { id: "overview", icon: <Info />, label: "Overview" },
  { id: "logs", icon: <Terminal />, label: "Logs" },
];

interface JobDetailDrawerProps {
  job: JobRun | null;
  open: boolean;
  onClose: () => void;
}

/**
 * Format job name for display
 */
function formatJobName(name: string): string {
  return name
    .split("_")
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(" ");
}

/**
 * Format duration in ms to human-readable
 */
function formatDuration(ms?: number): string {
  if (!ms) return "-";
  if (ms < 1000) return `${ms}ms`;
  if (ms < 60000) return `${(ms / 1000).toFixed(1)}s`;
  return `${(ms / 60000).toFixed(1)}m`;
}

/**
 * Job Tab Content - content only, without tabs (for desktop drawer with fixedContent)
 */
function JobTabContent({ job, activeTab }: { job: JobRun; activeTab: string }) {
  const startedAt = new Date(job.startedAt);
  const formattedStarted = startedAt.toLocaleString("en-US", {
    weekday: "short",
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });

  return (
    <div className="w-full">
      {/* Overview Tab */}
      {activeTab === "overview" && (
        <div className="bg-surface rounded-lg p-4 space-y-3">
          <div className="flex items-center justify-between">
            <JobStatusBadge status={job.status} />
            <Badge variant="outline" className="text-xs">
              {job.triggeredBy}
            </Badge>
          </div>

          {/* Timing info */}
          <div className="space-y-3 pt-3 border-t border-border">
            <div className="flex items-center gap-2 text-sm">
              <Clock className="h-4 w-4 text-muted-foreground" />
              <span className="text-muted-foreground">Started:</span>
              <span className="text-foreground">{formattedStarted}</span>
            </div>

            {job.finishedAt && (
              <div className="flex items-center gap-2 text-sm">
                <Clock className="h-4 w-4 text-muted-foreground" />
                <span className="text-muted-foreground">Duration:</span>
                <span className="text-foreground font-mono">
                  {formatDuration(job.durationMs)}
                </span>
              </div>
            )}

            {!job.finishedAt && (
              <div className="flex items-center gap-2 text-sm">
                <Clock className="h-4 w-4 text-muted-foreground animate-pulse" />
                <span className="text-muted-foreground">Running for:</span>
                <span className="text-foreground">
                  {formatDistanceToNow(job.startedAt)}
                </span>
              </div>
            )}
          </div>

          {/* Error info */}
          {job.error && (
            <div className="bg-error/10 border border-error/20 rounded-lg p-4">
              <div className="flex items-start gap-2">
                <AlertCircle className="h-4 w-4 text-error shrink-0 mt-0.5" />
                <div>
                  <div className="text-sm font-medium text-error mb-1">
                    Error
                  </div>
                  <div className="text-sm text-error/80">{job.error}</div>
                </div>
              </div>
            </div>
          )}

          {/* Metadata */}
          <div className="space-y-2 pt-3 border-t border-border">
            <div className="text-sm">
              <span className="text-muted-foreground">Job ID:</span>{" "}
              <span className="text-foreground font-mono">{job.id}</span>
            </div>
            <div className="text-sm">
              <span className="text-muted-foreground">Job Name:</span>{" "}
              <span className="text-foreground font-mono">{job.jobName}</span>
            </div>
          </div>
        </div>
      )}

      {/* Logs Tab */}
      {activeTab === "logs" && (
        <div className="bg-surface rounded-lg p-4">
          <div className="flex items-center gap-2 mb-4">
            <Terminal className="h-4 w-4 text-muted-foreground" />
            <span className="text-sm font-medium">Execution Logs</span>
          </div>

          <div className="font-mono text-xs text-muted-foreground space-y-1">
            <p>[{formattedStarted}] Starting {job.jobName}...</p>
            {job.status === "running" && (
              <p className="text-primary">[...] Job in progress</p>
            )}
            {job.status === "success" && (
              <>
                <p>[...] Processing completed</p>
                <p className="text-success">
                  [{job.finishedAt ? new Date(job.finishedAt).toLocaleTimeString() : ""}] Job finished successfully
                </p>
              </>
            )}
            {job.status === "failed" && (
              <>
                <p>[...] Processing started</p>
                <p className="text-error">[ERROR] {job.error}</p>
                <p className="text-error">
                  [{job.finishedAt ? new Date(job.finishedAt).toLocaleTimeString() : ""}] Job failed
                </p>
              </>
            )}
          </div>
          <p className="text-xs text-muted-foreground mt-4 pt-2 border-t border-border">
            Full logs coming soon
          </p>
        </div>
      )}
    </div>
  );
}

/**
 * Job Detail Content - used for mobile sheet (tabs + content together)
 */
function JobDetailContentMobile({ job }: { job: JobRun }) {
  const [activeTab, setActiveTab] = useState("overview");

  return (
    <div className="w-full space-y-3">
      <IconTabs
        tabs={JOB_TABS}
        value={activeTab}
        onValueChange={setActiveTab}
        className="w-full"
      />
      <JobTabContent job={job} activeTab={activeTab} />
    </div>
  );
}

/**
 * Responsive Job Detail Drawer
 *
 * Desktop (>=1280px): Overlay drawer (no reflow, ~400px)
 * Mobile/Tablet (<1280px): Sheet overlay
 */
export function JobDetailDrawer({
  job,
  open,
  onClose,
}: JobDetailDrawerProps) {
  const isDesktop = useIsDesktop();
  const [activeTab, setActiveTab] = useState("overview");
  const jobTitle = job ? formatJobName(job.jobName) : "Job Details";

  // Desktop: overlay drawer with tabs in fixedContent
  if (isDesktop) {
    return (
      <DetailDrawer
        open={open}
        onClose={onClose}
        title={jobTitle}
        fixedContent={
          job && (
            <IconTabs
              tabs={JOB_TABS}
              value={activeTab}
              onValueChange={setActiveTab}
              className="w-full"
            />
          )
        }
      >
        {job ? (
          <JobTabContent job={job} activeTab={activeTab} />
        ) : (
          <p className="text-muted-foreground text-sm">Select a job to view details</p>
        )}
      </DetailDrawer>
    );
  }

  // Mobile/Tablet: Sheet overlay
  return (
    <Sheet open={open} onOpenChange={(isOpen) => !isOpen && onClose()}>
      <SheetContent side="right" className="w-full sm:max-w-md p-0" data-dev-ref="JobDetailDrawer">
        <SheetHeader className="px-4 py-3 border-b border-border">
          <SheetTitle className="text-sm font-semibold truncate">
            {jobTitle}
          </SheetTitle>
        </SheetHeader>
        <ScrollArea className="h-[calc(100vh-60px)]">
          <div className="p-4">
            {job ? (
              <JobDetailContentMobile job={job} />
            ) : (
              <p className="text-muted-foreground text-sm">Select a job to view details</p>
            )}
          </div>
        </ScrollArea>
      </SheetContent>
    </Sheet>
  );
}
