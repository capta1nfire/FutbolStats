"use client";

import { useState } from "react";
import {
  ColumnDef,
  flexRender,
  getCoreRowModel,
  getSortedRowModel,
  SortingState,
  useReactTable,
} from "@tanstack/react-table";
import { ArrowUpDown, ArrowUp, ArrowDown, Loader2 } from "lucide-react";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";

interface DataTableProps<TData> {
  columns: ColumnDef<TData>[];
  data: TData[];
  isLoading?: boolean;
  error?: Error | null;
  onRetry?: () => void;
  selectedRowId?: string | number | null;
  onRowClick?: (row: TData) => void;
  getRowId?: (row: TData) => string | number;
  emptyMessage?: string;
}

/**
 * DataTable wrapper for TanStack Table v8
 *
 * Uses ColumnDef<T> directly from TanStack Table - no custom API.
 * Handles: sticky header, sorting, row selection, loading/empty/error states.
 */
export function DataTable<TData>({
  columns,
  data,
  isLoading = false,
  error = null,
  onRetry,
  selectedRowId,
  onRowClick,
  getRowId,
  emptyMessage = "No data found",
}: DataTableProps<TData>) {
  const [sorting, setSorting] = useState<SortingState>([]);

  // eslint-disable-next-line react-hooks/incompatible-library -- TanStack Table is compatible
  const table = useReactTable({
    data,
    columns,
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: getSortedRowModel(),
    onSortingChange: setSorting,
    state: {
      sorting,
    },
    getRowId: getRowId
      ? (row) => String(getRowId(row))
      : (row, index) => String(index),
  });

  // Loading state
  if (isLoading) {
    return (
      <div className="flex-1 flex items-center justify-center">
        <div className="flex flex-col items-center gap-2">
          <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
          <p className="text-sm text-muted-foreground">Loading...</p>
        </div>
      </div>
    );
  }

  // Error state
  if (error) {
    return (
      <div className="flex-1 flex items-center justify-center">
        <div className="flex flex-col items-center gap-3 text-center">
          <p className="text-sm text-error">{error.message}</p>
          {onRetry && (
            <Button variant="secondary" size="sm" onClick={onRetry}>
              Retry
            </Button>
          )}
        </div>
      </div>
    );
  }

  // Empty state
  if (data.length === 0) {
    return (
      <div className="flex-1 flex items-center justify-center">
        <p className="text-sm text-muted-foreground">{emptyMessage}</p>
      </div>
    );
  }

  return (
    <div className="flex-1 overflow-auto">
      <table className="w-full border-collapse text-sm">
        {/* Sticky header */}
        <thead className="sticky top-0 z-10 bg-surface border-b border-border">
          {table.getHeaderGroups().map((headerGroup) => (
            <tr key={headerGroup.id}>
              {headerGroup.headers.map((header) => {
                const canSort = header.column.getCanSort();
                const sorted = header.column.getIsSorted();
                const headerContent = header.isPlaceholder
                  ? null
                  : flexRender(header.column.columnDef.header, header.getContext());

                return (
                  <th
                    key={header.id}
                    className="px-3 py-2 text-left font-medium text-muted-foreground text-xs uppercase tracking-wider"
                  >
                    {canSort ? (
                      <button
                        type="button"
                        onClick={header.column.getToggleSortingHandler()}
                        className="flex items-center gap-1 cursor-pointer select-none hover:text-foreground focus:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-1 rounded"
                        aria-label={`Sort by ${typeof headerContent === "string" ? headerContent : header.id}`}
                      >
                        {headerContent}
                        <span className="text-muted-foreground">
                          {sorted === "asc" ? (
                            <ArrowUp className="h-3.5 w-3.5" />
                          ) : sorted === "desc" ? (
                            <ArrowDown className="h-3.5 w-3.5" />
                          ) : (
                            <ArrowUpDown className="h-3.5 w-3.5 opacity-50" />
                          )}
                        </span>
                      </button>
                    ) : (
                      <div className="flex items-center gap-1">{headerContent}</div>
                    )}
                  </th>
                );
              })}
            </tr>
          ))}
        </thead>

        {/* Body */}
        <tbody>
          {table.getRowModel().rows.map((row) => {
            const isSelected =
              selectedRowId !== null &&
              selectedRowId !== undefined &&
              row.id === String(selectedRowId);

            return (
              <tr
                key={row.id}
                onClick={() => onRowClick?.(row.original)}
                className={cn(
                  "border-b border-border transition-colors cursor-pointer",
                  isSelected
                    ? "bg-primary/10 border-l-2 border-l-primary"
                    : "hover:bg-accent/50"
                )}
              >
                {row.getVisibleCells().map((cell) => (
                  <td key={cell.id} className="px-3 py-2.5">
                    {flexRender(cell.column.columnDef.cell, cell.getContext())}
                  </td>
                ))}
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

/**
 * Helper component for sortable header
 */
export function SortableHeader({
  column,
  children,
}: {
  column: { getCanSort: () => boolean; getIsSorted: () => false | "asc" | "desc" };
  children: React.ReactNode;
}) {
  const sorted = column.getIsSorted();

  return (
    <div className="flex items-center gap-1">
      {children}
      {column.getCanSort() && (
        <span>
          {sorted === "asc" ? (
            <ArrowUp className="h-3.5 w-3.5" />
          ) : sorted === "desc" ? (
            <ArrowDown className="h-3.5 w-3.5" />
          ) : (
            <ArrowUpDown className="h-3.5 w-3.5 opacity-50" />
          )}
        </span>
      )}
    </div>
  );
}
