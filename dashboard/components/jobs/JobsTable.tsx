"use client";

import { useMemo } from "react";
import { ColumnDef } from "@tanstack/react-table";
import { JobRun } from "@/lib/types";
import { DataTable } from "@/components/tables";
import { JobStatusBadge } from "./JobStatusBadge";
import { formatDistanceToNow } from "@/lib/utils";
import { Badge } from "@/components/ui/badge";

interface JobsTableProps {
  data: JobRun[];
  isLoading?: boolean;
  error?: Error | null;
  onRetry?: () => void;
  selectedJobId?: number | null;
  onRowClick?: (job: JobRun) => void;
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
 * Format job name for display
 */
function formatJobName(name: string): string {
  return name
    .split("_")
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(" ");
}

export function JobsTable({
  data,
  isLoading,
  error,
  onRetry,
  selectedJobId,
  onRowClick,
}: JobsTableProps) {
  const columns = useMemo<ColumnDef<JobRun>[]>(
    () => [
      {
        accessorKey: "status",
        header: "Status",
        cell: ({ row }) => <JobStatusBadge status={row.original.status} />,
        enableSorting: true,
      },
      {
        accessorKey: "jobName",
        header: "Job",
        cell: ({ row }) => (
          <div className="font-medium text-foreground">
            {formatJobName(row.original.jobName)}
          </div>
        ),
        enableSorting: true,
      },
      {
        accessorKey: "startedAt",
        header: "Started",
        cell: ({ row }) => (
          <span className="text-muted-foreground">
            {formatDistanceToNow(row.original.startedAt)}
          </span>
        ),
        enableSorting: true,
      },
      {
        accessorKey: "durationMs",
        header: "Duration",
        cell: ({ row }) => (
          <span className="text-muted-foreground font-mono text-xs">
            {formatDuration(row.original.durationMs)}
          </span>
        ),
        enableSorting: true,
      },
      {
        accessorKey: "triggeredBy",
        header: "Trigger",
        cell: ({ row }) => (
          <Badge variant="outline" className="text-xs font-normal">
            {row.original.triggeredBy}
          </Badge>
        ),
        enableSorting: true,
      },
      {
        id: "error",
        header: "Error",
        cell: ({ row }) =>
          row.original.error ? (
            <span className="text-error text-xs truncate max-w-[200px] block">
              {row.original.error}
            </span>
          ) : (
            <span className="text-muted-foreground">-</span>
          ),
        enableSorting: false,
      },
    ],
    []
  );

  return (
    <DataTable
      columns={columns}
      data={data}
      isLoading={isLoading}
      error={error}
      onRetry={onRetry}
      selectedRowId={selectedJobId}
      onRowClick={onRowClick}
      getRowId={(row) => row.id}
      emptyMessage="No job runs found"
    />
  );
}
