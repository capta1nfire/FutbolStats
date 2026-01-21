"use client";

import { JobRun } from "@/lib/types";
import { useIsDesktop } from "@/lib/hooks";
import { DetailDrawer } from "@/components/shell";
import {
  Sheet,
  SheetContent,
  SheetHeader,
  SheetTitle,
} from "@/components/ui/sheet";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import { JobStatusBadge } from "./JobStatusBadge";
import { formatDistanceToNow } from "@/lib/utils";
import { Clock, Terminal, AlertCircle } from "lucide-react";

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
 * Job Detail Content - shared between desktop drawer and mobile sheet
 */
function JobDetailContent({ job }: { job: JobRun }) {
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
    <Tabs defaultValue="overview" className="w-full">
      <TabsList className="w-full grid grid-cols-2 mb-4">
        <TabsTrigger value="overview" className="rounded-full text-xs">
          Overview
        </TabsTrigger>
        <TabsTrigger value="logs" className="rounded-full text-xs">
          Logs
        </TabsTrigger>
      </TabsList>

      {/* Overview Tab */}
      <TabsContent value="overview" className="space-y-4">
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <JobStatusBadge status={job.status} />
            <Badge variant="outline" className="text-xs">
              {job.triggeredBy}
            </Badge>
          </div>

          {/* Timing info */}
          <div className="bg-background rounded-lg p-4 space-y-3">
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
          <div className="space-y-2">
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
      </TabsContent>

      {/* Logs Tab */}
      <TabsContent value="logs" className="space-y-4">
        <div className="flex items-center gap-2 mb-4">
          <Terminal className="h-4 w-4 text-muted-foreground" />
          <span className="text-sm font-medium">Execution Logs</span>
        </div>

        <div className="bg-background rounded-lg p-4">
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
      </TabsContent>
    </Tabs>
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
  const jobTitle = job ? formatJobName(job.jobName) : "Job Details";

  // Desktop: overlay drawer
  if (isDesktop) {
    return (
      <DetailDrawer open={open} onClose={onClose} title={jobTitle}>
        {job ? (
          <JobDetailContent job={job} />
        ) : (
          <p className="text-muted-foreground text-sm">Select a job to view details</p>
        )}
      </DetailDrawer>
    );
  }

  // Mobile/Tablet: Sheet overlay
  return (
    <Sheet open={open} onOpenChange={(isOpen) => !isOpen && onClose()}>
      <SheetContent side="right" className="w-full sm:max-w-md p-0">
        <SheetHeader className="px-4 py-3 border-b border-border">
          <SheetTitle className="text-sm font-semibold truncate">
            {jobTitle}
          </SheetTitle>
        </SheetHeader>
        <ScrollArea className="h-[calc(100vh-60px)]">
          <div className="p-4">
            {job ? (
              <JobDetailContent job={job} />
            ) : (
              <p className="text-muted-foreground text-sm">Select a job to view details</p>
            )}
          </div>
        </ScrollArea>
      </SheetContent>
    </Sheet>
  );
}
