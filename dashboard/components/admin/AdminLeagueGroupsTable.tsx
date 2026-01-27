"use client";

import { useMemo } from "react";
import type { ColumnDef } from "@tanstack/react-table";
import { DataTable } from "@/components/tables/DataTable";
import { Badge } from "@/components/ui/badge";
import type { AdminLeagueGroupListItem } from "@/lib/types";
import { useAdminLeagueGroups } from "@/lib/hooks";

interface AdminLeagueGroupsTableProps {
  selectedGroupId: number | null;
  onRowClick: (group: AdminLeagueGroupListItem) => void;
}

export function AdminLeagueGroupsTable({
  selectedGroupId,
  onRowClick,
}: AdminLeagueGroupsTableProps) {
  const { data, isLoading, error, refetch } = useAdminLeagueGroups();

  const columns = useMemo<ColumnDef<AdminLeagueGroupListItem>[]>(
    () => [
      {
        accessorKey: "group_id",
        header: "ID",
        cell: ({ row }) => (
          <span className="font-mono text-xs text-muted-foreground">
            {row.original.group_id}
          </span>
        ),
        size: 60,
        enableSorting: true,
      },
      {
        accessorKey: "name",
        header: "Name",
        cell: ({ row }) => (
          <span className="font-medium text-foreground">{row.original.name}</span>
        ),
        enableSorting: true,
      },
      {
        accessorKey: "group_key",
        header: "Key",
        cell: ({ row }) => (
          <span className="font-mono text-xs text-muted-foreground">
            {row.original.group_key}
          </span>
        ),
        enableSorting: true,
      },
      {
        accessorKey: "country",
        header: "Country",
        cell: ({ row }) => (
          <span className="text-sm">{row.original.country}</span>
        ),
        enableSorting: true,
      },
      {
        id: "members",
        header: "Members",
        cell: ({ row }) => (
          <span className="font-mono text-xs">{row.original.leagues?.length ?? 0}</span>
        ),
        size: 80,
        enableSorting: false,
      },
      {
        accessorKey: "is_active_any",
        header: "Active",
        cell: ({ row }) => (
          <Badge
            variant={row.original.is_active_any ? "default" : "secondary"}
            className="text-xs"
          >
            {row.original.is_active_all ? "All" : row.original.is_active_any ? "Partial" : "No"}
          </Badge>
        ),
        size: 80,
        enableSorting: true,
      },
      {
        accessorKey: "stats.matches_25_26",
        header: "25/26",
        cell: ({ row }) => (
          <span className="font-mono text-xs">
            {row.original.stats?.matches_25_26 ?? "—"}
          </span>
        ),
        size: 70,
        enableSorting: true,
      },
    ],
    []
  );

  const groups = data?.groups ?? [];

  return (
    <DataTable
      columns={columns}
      data={groups}
      isLoading={isLoading}
      error={error}
      onRetry={refetch}
      selectedRowId={selectedGroupId}
      onRowClick={onRowClick}
      getRowId={(row) => row.group_id}
    />
  );
}
