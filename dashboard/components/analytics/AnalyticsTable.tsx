"use client";

import { useMemo } from "react";
import { ColumnDef } from "@tanstack/react-table";
import { AnalyticsReportRow } from "@/lib/types";
import { DataTable } from "@/components/tables";
import { ReportStatusBadge } from "./ReportStatusBadge";
import { ReportTypeBadge } from "./ReportTypeBadge";
import { formatDistanceToNow } from "@/lib/utils";

interface AnalyticsTableProps {
  data: AnalyticsReportRow[];
  isLoading?: boolean;
  error?: Error | null;
  onRetry?: () => void;
  selectedReportId?: number | null;
  onRowClick?: (report: AnalyticsReportRow) => void;
}

export function AnalyticsTable({
  data,
  isLoading,
  error,
  onRetry,
  selectedReportId,
  onRowClick,
}: AnalyticsTableProps) {
  const columns = useMemo<ColumnDef<AnalyticsReportRow>[]>(
    () => [
      {
        accessorKey: "status",
        header: "Status",
        cell: ({ row }) =>
          row.original.status ? (
            <ReportStatusBadge status={row.original.status} />
          ) : (
            <span className="text-muted-foreground">-</span>
          ),
        enableSorting: true,
      },
      {
        accessorKey: "title",
        header: "Report",
        cell: ({ row }) => (
          <div className="font-medium text-foreground max-w-[250px] truncate">
            {row.original.title}
          </div>
        ),
        enableSorting: true,
      },
      {
        accessorKey: "type",
        header: "Type",
        cell: ({ row }) => <ReportTypeBadge type={row.original.type} />,
        enableSorting: true,
      },
      {
        accessorKey: "periodLabel",
        header: "Period",
        cell: ({ row }) => (
          <span className="text-muted-foreground text-sm">
            {row.original.periodLabel}
          </span>
        ),
        enableSorting: false,
      },
      {
        id: "summary",
        header: "Summary",
        cell: ({ row }) => {
          const summary = row.original.summary;
          const entries = Object.entries(summary).slice(0, 2);
          return (
            <div className="flex gap-3 text-xs">
              {entries.map(([key, value]) => (
                <span key={key} className="text-muted-foreground">
                  <span className="capitalize">{key}:</span>{" "}
                  <span className="text-foreground font-mono">{value}</span>
                </span>
              ))}
            </div>
          );
        },
        enableSorting: false,
      },
      {
        accessorKey: "lastUpdated",
        header: "Updated",
        cell: ({ row }) => (
          <span className="text-muted-foreground">
            {formatDistanceToNow(row.original.lastUpdated)}
          </span>
        ),
        enableSorting: true,
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
      selectedRowId={selectedReportId}
      onRowClick={onRowClick}
      getRowId={(row) => row.id}
      emptyMessage="No analytics reports found"
    />
  );
}
