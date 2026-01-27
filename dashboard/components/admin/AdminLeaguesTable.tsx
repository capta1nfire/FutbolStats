"use client";

import { useMemo } from "react";
import type { ColumnDef } from "@tanstack/react-table";
import { DataTable } from "@/components/tables/DataTable";
import { Badge } from "@/components/ui/badge";
import type { AdminLeagueListItem, AdminLeaguesFilters } from "@/lib/types";
import { useAdminLeagues } from "@/lib/hooks";

interface AdminLeaguesTableProps {
  filters: AdminLeaguesFilters;
  selectedLeagueId: number | null;
  onRowClick: (league: AdminLeagueListItem) => void;
}

export function AdminLeaguesTable({
  filters,
  selectedLeagueId,
  onRowClick,
}: AdminLeaguesTableProps) {
  const { data, isLoading, error, refetch } = useAdminLeagues(filters);

  const columns = useMemo<ColumnDef<AdminLeagueListItem>[]>(
    () => [
      {
        accessorKey: "league_id",
        header: "ID",
        cell: ({ row }) => (
          <span className="font-mono text-xs text-muted-foreground">
            {row.original.league_id}
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
        accessorKey: "country",
        header: "Country",
        cell: ({ row }) => (
          <span className="text-sm">{row.original.country}</span>
        ),
        enableSorting: true,
      },
      {
        accessorKey: "kind",
        header: "Kind",
        cell: ({ row }) => (
          <Badge variant="outline" className="text-xs font-normal">
            {row.original.kind}
          </Badge>
        ),
        size: 90,
        enableSorting: true,
      },
      {
        accessorKey: "is_active",
        header: "Active",
        cell: ({ row }) => (
          <Badge
            variant={row.original.is_active ? "default" : "secondary"}
            className="text-xs"
          >
            {row.original.is_active ? "Yes" : "No"}
          </Badge>
        ),
        size: 70,
        enableSorting: true,
      },
      {
        accessorKey: "source",
        header: "Source",
        cell: ({ row }) => (
          <span className="text-xs text-muted-foreground">{row.original.source}</span>
        ),
        size: 90,
        enableSorting: true,
      },
      {
        accessorKey: "priority",
        header: "Priority",
        cell: ({ row }) => (
          <span className="text-xs">{row.original.priority}</span>
        ),
        size: 80,
        enableSorting: true,
      },
      {
        accessorKey: "match_weight",
        header: "Weight",
        cell: ({ row }) => (
          <span className="font-mono text-xs">
            {row.original.match_weight != null ? row.original.match_weight : "—"}
          </span>
        ),
        size: 70,
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

  const leagues = data?.leagues ?? [];

  return (
    <DataTable
      columns={columns}
      data={leagues}
      isLoading={isLoading}
      error={error}
      onRetry={refetch}
      selectedRowId={selectedLeagueId}
      onRowClick={onRowClick}
      getRowId={(row) => row.league_id}
    />
  );
}
