"use client";

import { useMemo } from "react";
import { ColumnDef } from "@tanstack/react-table";
import { Incident, INCIDENT_TYPE_LABELS } from "@/lib/types";
import { DataTable } from "@/components/tables";
import { SeverityBadge } from "./SeverityBadge";
import { IncidentStatusChip } from "./IncidentStatusChip";
import { formatDistanceToNow } from "@/lib/utils";
import { ExternalLink } from "lucide-react";

interface IncidentsTableProps {
  data: Incident[];
  isLoading?: boolean;
  error?: Error | null;
  onRetry?: () => void;
  selectedIncidentId?: number | null;
  onRowClick?: (incident: Incident) => void;
}

export function IncidentsTable({
  data,
  isLoading,
  error,
  onRetry,
  selectedIncidentId,
  onRowClick,
}: IncidentsTableProps) {
  const columns = useMemo<ColumnDef<Incident>[]>(
    () => [
      {
        accessorKey: "severity",
        header: "Severity",
        cell: ({ row }) => <SeverityBadge severity={row.original.severity} />,
        enableSorting: true,
      },
      {
        accessorKey: "status",
        header: "Status",
        cell: ({ row }) => <IncidentStatusChip status={row.original.status} />,
        enableSorting: true,
      },
      {
        accessorKey: "title",
        header: "Title",
        cell: ({ row }) => (
          <div className="font-medium text-foreground max-w-[300px] truncate">
            {row.original.title}
          </div>
        ),
        enableSorting: true,
      },
      {
        accessorKey: "type",
        header: "Type",
        cell: ({ row }) => (
          <span className="text-muted-foreground text-sm">
            {INCIDENT_TYPE_LABELS[row.original.type]}
          </span>
        ),
        enableSorting: true,
      },
      {
        accessorKey: "createdAt",
        header: "Created",
        cell: ({ row }) => (
          <span className="text-muted-foreground">
            {formatDistanceToNow(row.original.createdAt)}
          </span>
        ),
        enableSorting: true,
      },
      {
        id: "entity",
        header: "Related",
        cell: ({ row }) =>
          row.original.entity ? (
            <span className="text-accent text-xs flex items-center gap-1">
              {row.original.entity.kind} #{row.original.entity.id}
              <ExternalLink className="h-3 w-3" />
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
      selectedRowId={selectedIncidentId}
      onRowClick={onRowClick}
      getRowId={(row) => row.id}
      emptyMessage="No incidents found"
    />
  );
}
