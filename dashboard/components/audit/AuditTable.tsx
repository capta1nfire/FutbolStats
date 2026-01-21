"use client";

import { useMemo } from "react";
import { ColumnDef } from "@tanstack/react-table";
import { AuditEventRow } from "@/lib/types";
import { DataTable } from "@/components/tables";
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
}

export function AuditTable({
  data,
  isLoading,
  error,
  onRetry,
  selectedEventId,
  onRowClick,
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
              <span className={isUser ? "text-accent" : "text-muted-foreground"}>
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
      selectedRowId={selectedEventId}
      onRowClick={onRowClick}
      getRowId={(row) => row.id}
      emptyMessage="No audit events found"
    />
  );
}
