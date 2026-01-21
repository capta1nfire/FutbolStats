"use client";

import { useMemo } from "react";
import { ColumnDef } from "@tanstack/react-table";
import { DataQualityCheck } from "@/lib/types";
import { DataTable } from "@/components/tables";
import { DataQualityStatusBadge } from "./DataQualityStatusBadge";
import { DataQualityCategoryBadge } from "./DataQualityCategoryBadge";
import { formatDistanceToNow } from "@/lib/utils";
import { AlertCircle } from "lucide-react";

interface DataQualityTableProps {
  data: DataQualityCheck[];
  isLoading?: boolean;
  error?: Error | null;
  onRetry?: () => void;
  selectedCheckId?: number | null;
  onRowClick?: (check: DataQualityCheck) => void;
}

export function DataQualityTable({
  data,
  isLoading,
  error,
  onRetry,
  selectedCheckId,
  onRowClick,
}: DataQualityTableProps) {
  const columns = useMemo<ColumnDef<DataQualityCheck>[]>(
    () => [
      {
        accessorKey: "status",
        header: "Status",
        cell: ({ row }) => <DataQualityStatusBadge status={row.original.status} />,
        enableSorting: true,
      },
      {
        accessorKey: "name",
        header: "Check",
        cell: ({ row }) => (
          <div className="font-medium text-foreground max-w-[250px] truncate">
            {row.original.name}
          </div>
        ),
        enableSorting: true,
      },
      {
        accessorKey: "category",
        header: "Category",
        cell: ({ row }) => (
          <DataQualityCategoryBadge category={row.original.category} />
        ),
        enableSorting: true,
      },
      {
        accessorKey: "currentValue",
        header: "Value",
        cell: ({ row }) => (
          <span className="font-mono text-sm text-foreground">
            {row.original.currentValue ?? "-"}
          </span>
        ),
        enableSorting: false,
      },
      {
        accessorKey: "threshold",
        header: "Threshold",
        cell: ({ row }) => (
          <span className="font-mono text-sm text-muted-foreground">
            {row.original.threshold ?? "-"}
          </span>
        ),
        enableSorting: false,
      },
      {
        accessorKey: "affectedCount",
        header: "Affected",
        cell: ({ row }) =>
          row.original.affectedCount > 0 ? (
            <span className="flex items-center gap-1 text-yellow-400">
              <AlertCircle className="h-3 w-3" />
              {row.original.affectedCount}
            </span>
          ) : (
            <span className="text-muted-foreground">0</span>
          ),
        enableSorting: true,
      },
      {
        accessorKey: "lastRunAt",
        header: "Last Run",
        cell: ({ row }) => (
          <span className="text-muted-foreground">
            {formatDistanceToNow(row.original.lastRunAt)}
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
      selectedRowId={selectedCheckId}
      onRowClick={onRowClick}
      getRowId={(row) => row.id}
      emptyMessage="No data quality checks found"
    />
  );
}
