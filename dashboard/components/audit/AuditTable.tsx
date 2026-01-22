"use client";

import { useMemo } from "react";
import { ColumnDef, VisibilityState } from "@tanstack/react-table";
import { AuditEventRow } from "@/lib/types";
import { DataTable, ColumnOption } from "@/components/tables";
import { AuditSeverityBadge } from "./AuditSeverityBadge";
import { AuditTypeBadge } from "./AuditTypeBadge";
import { formatDistanceToNow } from "@/lib/utils";
import { ExternalLink, User, Server } from "lucide-react";

interface AuditTableProps {
  data: AuditEventRow[];
  isLoading?: boolean;
  error?: Error | null;
  onRetry?: () => void;
  selectedEventId?: number | null;
  onRowClick?: (event: AuditEventRow) => void;
  columnVisibility?: VisibilityState;
  onColumnVisibilityChange?: (visibility: VisibilityState) => void;
}

/**
 * Column options for Customize Columns panel
 */
export const AUDIT_COLUMN_OPTIONS: ColumnOption[] = [
  { id: "severity", label: "Severity", enableHiding: true },
  { id: "timestamp", label: "Time", enableHiding: false },
  { id: "type", label: "Type", enableHiding: true },
  { id: "actor", label: "Actor", enableHiding: true },
  { id: "message", label: "Message", enableHiding: false },
  { id: "entity", label: "Entity", enableHiding: true },
];

/**
 * Default column visibility for Audit table
 */
export const AUDIT_DEFAULT_VISIBILITY: VisibilityState = {};

export function AuditTable({
  data,
  isLoading,
  error,
  onRetry,
  selectedEventId,
  onRowClick,
  columnVisibility,
  onColumnVisibilityChange,
}: AuditTableProps) {
  const columns = useMemo<ColumnDef<AuditEventRow>[]>(
    () => [
      {
        accessorKey: "severity",
        header: "Severity",
        cell: ({ row }) =>
          row.original.severity ? (
            <AuditSeverityBadge severity={row.original.severity} />
          ) : (
            <span className="text-muted-foreground">-</span>
          ),
        enableSorting: true,
      },
      {
        accessorKey: "timestamp",
        header: "Time",
        cell: ({ row }) => (
          <span className="text-muted-foreground text-sm">
            {formatDistanceToNow(row.original.timestamp)}
          </span>
        ),
        enableSorting: true,
      },
      {
        accessorKey: "type",
        header: "Type",
        cell: ({ row }) => <AuditTypeBadge type={row.original.type} />,
        enableSorting: true,
      },
      {
        id: "actor",
        header: "Actor",
        cell: ({ row }) => {
          const actor = row.original.actor;
          const isUser = actor.kind === "user";
          return (
            <span className="flex items-center gap-1 text-sm">
              {isUser ? (
                <User className="h-3 w-3 text-muted-foreground" />
              ) : (
                <Server className="h-3 w-3 text-muted-foreground" />
              )}
              <span className={isUser ? "text-primary" : "text-muted-foreground"}>
                {actor.name}
              </span>
            </span>
          );
        },
        enableSorting: false,
      },
      {
        accessorKey: "message",
        header: "Message",
        cell: ({ row }) => (
          <div className="text-sm text-foreground max-w-[300px] truncate">
            {row.original.message}
          </div>
        ),
        enableSorting: false,
      },
      {
        id: "entity",
        header: "Entity",
        cell: ({ row }) =>
          row.original.entity ? (
            <span className="text-primary text-xs flex items-center gap-1">
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
      selectedRowId={selectedEventId}
      onRowClick={onRowClick}
      getRowId={(row) => row.id}
      emptyMessage="No audit events found"
      columnVisibility={columnVisibility}
      onColumnVisibilityChange={onColumnVisibilityChange}
    />
  );
}
