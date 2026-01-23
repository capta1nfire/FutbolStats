"use client";

import { useMemo } from "react";
import { ColumnDef, VisibilityState } from "@tanstack/react-table";
import { DataQualityCheck } from "@/lib/types";
import { DataTable, ColumnOption } from "@/components/tables";
import { DataQualityStatusBadge } from "./DataQualityStatusBadge";
import { DataQualityCategoryBadge } from "./DataQualityCategoryBadge";
import { formatDistanceToNow } from "@/lib/utils";
import { AlertCircle } from "lucide-react";

interface DataQualityTableProps {
  data: DataQualityCheck[];
  isLoading?: boolean;
  error?: Error | null;
  onRetry?: () => void;
  selectedCheckId?: string | null; // String ID for backend compatibility
  onRowClick?: (check: DataQualityCheck) => void;
  columnVisibility?: VisibilityState;
  onColumnVisibilityChange?: (visibility: VisibilityState) => void;
}

/**
 * Column options for Customize Columns panel
 */
export const DATA_QUALITY_COLUMN_OPTIONS: ColumnOption[] = [
  { id: "status", label: "Status", enableHiding: false },
  { id: "name", label: "Check", enableHiding: false },
  { id: "category", label: "Category", enableHiding: true },
  { id: "currentValue", label: "Value", enableHiding: true },
  { id: "threshold", label: "Threshold", enableHiding: true },
  { id: "affectedCount", label: "Affected", enableHiding: true },
  { id: "lastRunAt", label: "Last Run", enableHiding: true },
];

/**
 * Default column visibility for Data Quality table
 */
export const DATA_QUALITY_DEFAULT_VISIBILITY: VisibilityState = {};

export function DataQualityTable({
  data,
  isLoading,
  error,
  onRetry,
  selectedCheckId,
  onRowClick,
  columnVisibility,
  onColumnVisibilityChange,
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
      columnVisibility={columnVisibility}
      onColumnVisibilityChange={onColumnVisibilityChange}
    />
  );
}
