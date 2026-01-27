"use client";

import { useMemo, useState } from "react";
import type { ColumnDef } from "@tanstack/react-table";
import { DataTable } from "@/components/tables/DataTable";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { DetailDrawer } from "@/components/shell/DetailDrawer";
import type { AdminAuditEntry, AdminAuditFilters } from "@/lib/types";
import { useAdminAudit } from "@/lib/hooks";

interface AdminAuditTableProps {
  filters: AdminAuditFilters;
  onOffsetChange?: (offset: number) => void;
}

export function AdminAuditTable({ filters, onOffsetChange }: AdminAuditTableProps) {
  const { data, isLoading, error, refetch } = useAdminAudit(filters);
  const [selectedEntry, setSelectedEntry] = useState<AdminAuditEntry | null>(null);

  const columns = useMemo<ColumnDef<AdminAuditEntry>[]>(
    () => [
      {
        accessorKey: "id",
        header: "ID",
        cell: ({ row }) => (
          <span className="font-mono text-xs text-muted-foreground">
            {row.original.id}
          </span>
        ),
        size: 60,
        enableSorting: true,
      },
      {
        accessorKey: "created_at",
        header: "Date",
        cell: ({ row }) => (
          <span className="text-xs text-muted-foreground">
            {new Date(row.original.created_at).toLocaleString()}
          </span>
        ),
        enableSorting: true,
      },
      {
        accessorKey: "actor",
        header: "Actor",
        cell: ({ row }) => (
          <span className="text-sm">{row.original.actor}</span>
        ),
        enableSorting: true,
      },
      {
        accessorKey: "action",
        header: "Action",
        cell: ({ row }) => (
          <Badge variant="outline" className="text-xs font-normal">
            {row.original.action}
          </Badge>
        ),
        size: 100,
        enableSorting: true,
      },
      {
        accessorKey: "entity_type",
        header: "Entity Type",
        cell: ({ row }) => (
          <span className="text-xs">{row.original.entity_type}</span>
        ),
        size: 120,
        enableSorting: true,
      },
      {
        accessorKey: "entity_id",
        header: "Entity ID",
        cell: ({ row }) => (
          <span className="font-mono text-xs">{row.original.entity_id}</span>
        ),
        size: 80,
        enableSorting: true,
      },
    ],
    []
  );

  const entries = data?.entries ?? [];
  const pagination = data?.pagination;
  const limit = pagination?.limit ?? 50;
  const offset = pagination?.offset ?? 0;
  const total = pagination?.total ?? 0;
  const hasMore = pagination?.has_more ?? false;
  const hasPrev = offset > 0;
  const showingFrom = total > 0 ? offset + 1 : 0;
  const showingTo = Math.min(offset + limit, total);

  return (
    <div className="relative h-full flex flex-col overflow-hidden">
      <DataTable
        columns={columns}
        data={entries}
        isLoading={isLoading}
        error={error}
        onRetry={refetch}
        selectedRowId={selectedEntry?.id ?? null}
        onRowClick={(entry) => setSelectedEntry(entry)}
        getRowId={(row) => row.id}
      />

      {/* Pagination bar */}
      {total > 0 && (
        <div className="flex items-center justify-between border-t border-border px-4 py-2 text-xs text-muted-foreground shrink-0">
          <span>
            Showing {showingFrom}–{showingTo} of {total}
          </span>
          <div className="flex items-center gap-2">
            <Button
              variant="outline"
              size="sm"
              disabled={!hasPrev || isLoading}
              onClick={() => onOffsetChange?.(Math.max(0, offset - limit))}
              className="h-7 text-xs"
            >
              Prev
            </Button>
            <Button
              variant="outline"
              size="sm"
              disabled={!hasMore || isLoading}
              onClick={() => onOffsetChange?.(offset + limit)}
              className="h-7 text-xs"
            >
              Next
            </Button>
          </div>
        </div>
      )}

      <DetailDrawer
        open={selectedEntry !== null}
        onClose={() => setSelectedEntry(null)}
        title={`Audit #${selectedEntry?.id ?? ""}`}
        variant="overlay"
      >
        {selectedEntry && <AuditDetail entry={selectedEntry} />}
      </DetailDrawer>
    </div>
  );
}

function AuditDetail({ entry }: { entry: AdminAuditEntry }) {
  return (
    <div className="space-y-4 text-sm">
      <div className="grid grid-cols-2 gap-2">
        <InfoRow label="ID" value={String(entry.id)} />
        <InfoRow label="Date" value={new Date(entry.created_at).toLocaleString()} />
        <InfoRow label="Actor" value={entry.actor} />
        <InfoRow label="Action" value={entry.action} />
        <InfoRow label="Entity Type" value={entry.entity_type} />
        <InfoRow label="Entity ID" value={entry.entity_id} />
      </div>

      {entry.before && (
        <JsonBlock label="Before" data={entry.before} />
      )}
      {entry.after && (
        <JsonBlock label="After" data={entry.after} />
      )}
    </div>
  );
}

function InfoRow({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <p className="text-xs text-muted-foreground">{label}</p>
      <p className="font-medium">{value}</p>
    </div>
  );
}

function JsonBlock({ label, data }: { label: string; data: Record<string, unknown> }) {
  return (
    <div className="space-y-1">
      <p className="text-xs font-medium text-muted-foreground uppercase tracking-wider">
        {label}
      </p>
      <pre className="whitespace-pre-wrap break-all bg-muted/50 rounded-md p-3 font-mono text-[11px] max-h-64 overflow-auto">
        {JSON.stringify(data, null, 2)}
      </pre>
    </div>
  );
}
