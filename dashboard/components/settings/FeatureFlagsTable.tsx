"use client";

import { useMemo } from "react";
import { ColumnDef } from "@tanstack/react-table";
import { FeatureFlag } from "@/lib/types";
import { DataTable } from "@/components/tables";
import { Badge } from "@/components/ui/badge";
import { formatDistanceToNow } from "@/lib/utils";
import { ToggleLeft, ToggleRight } from "lucide-react";

interface FeatureFlagsTableProps {
  data: FeatureFlag[];
  isLoading?: boolean;
  error?: Error | null;
  onRetry?: () => void;
}

export function FeatureFlagsTable({
  data,
  isLoading,
  error,
  onRetry,
}: FeatureFlagsTableProps) {
  const columns = useMemo<ColumnDef<FeatureFlag>[]>(
    () => [
      {
        accessorKey: "enabled",
        header: "Status",
        cell: ({ row }) =>
          row.original.enabled ? (
            <div className="flex items-center gap-1.5">
              <ToggleRight className="h-4 w-4 text-success" />
              <Badge
                variant="outline"
                className="bg-success/10 text-success border-success/20"
              >
                Enabled
              </Badge>
            </div>
          ) : (
            <div className="flex items-center gap-1.5">
              <ToggleLeft className="h-4 w-4 text-muted-foreground" />
              <Badge variant="outline" className="text-muted-foreground">
                Disabled
              </Badge>
            </div>
          ),
        enableSorting: true,
      },
      {
        accessorKey: "name",
        header: "Name",
        cell: ({ row }) => (
          <div>
            <div className="text-sm font-medium text-foreground">
              {row.original.name}
            </div>
            <div className="text-xs text-muted-foreground font-mono">
              {row.original.id}
            </div>
          </div>
        ),
        enableSorting: true,
      },
      {
        accessorKey: "description",
        header: "Description",
        cell: ({ row }) => (
          <span className="text-sm text-muted-foreground max-w-[300px] truncate block">
            {row.original.description || "-"}
          </span>
        ),
        enableSorting: false,
      },
      {
        accessorKey: "updatedAt",
        header: "Last Updated",
        cell: ({ row }) =>
          row.original.updatedAt ? (
            <span className="text-sm text-muted-foreground">
              {formatDistanceToNow(row.original.updatedAt)}
            </span>
          ) : (
            <span className="text-muted-foreground">-</span>
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
      getRowId={(row) => row.id}
      emptyMessage="No feature flags configured"
    />
  );
}
