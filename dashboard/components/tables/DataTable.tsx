"use client";

import { useState, useRef, useCallback, useImperativeHandle, forwardRef } from "react";
import {
  ColumnDef,
  flexRender,
  getCoreRowModel,
  getSortedRowModel,
  SortingState,
  VisibilityState,
  Updater,
  useReactTable,
} from "@tanstack/react-table";
import { ChevronUp, ChevronDown } from "lucide-react";
import { Loader } from "@/components/ui/loader";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";

export interface DataTableHandle {
  /** Focus the selected row or first row */
  focusSelectedRow: () => void;
}

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
  /** Column visibility state (from useColumnVisibility hook) */
  columnVisibility?: VisibilityState;
  /** Called when column visibility changes (for controlled state) */
  onColumnVisibilityChange?: (visibility: VisibilityState) => void;
}

/**
 * DataTable wrapper for TanStack Table v8
 *
 * Uses ColumnDef<T> directly from TanStack Table - no custom API.
 * Handles: sticky header, sorting, row selection, loading/empty/error states.
 *
 * Density: All tables share the same spacing (header px-3 pt-3 pb-2, cell px-3 py-2.5).
 * Visual differences (e.g. Matches 2-line rows vs Jobs 1-line) are achieved via cell
 * content layout (flex-col), not via density presets. See ADS spec §3.4 for rationale.
 *
 * UX Features:
 * - Keyboard navigation: Arrow keys move between rows
 * - Enter key triggers onRowClick for focused row
 * - Tab to navigate into table, then arrows to move
 */
function DataTableInner<TData>(
  {
    columns,
    data,
    isLoading = false,
    error = null,
    onRetry,
    selectedRowId,
    onRowClick,
    getRowId,
    emptyMessage = "No data found",
    columnVisibility,
    onColumnVisibilityChange,
  }: DataTableProps<TData>,
  ref: React.ForwardedRef<DataTableHandle>
) {
  const [sorting, setSorting] = useState<SortingState>([]);
  const [focusedRowIndex, setFocusedRowIndex] = useState<number>(-1);
  const tableBodyRef = useRef<HTMLTableSectionElement>(null);
  const rowRefs = useRef<Map<number, HTMLTableRowElement>>(new Map());

  // Handle TanStack's Updater<VisibilityState> and convert to plain VisibilityState
  const handleColumnVisibilityChange = useCallback(
    (updaterOrValue: Updater<VisibilityState>) => {
      if (!onColumnVisibilityChange) return;

      if (typeof updaterOrValue === "function") {
        // It's an updater function, call it with current state
        const newValue = updaterOrValue(columnVisibility ?? {});
        onColumnVisibilityChange(newValue);
      } else {
        // It's a direct value
        onColumnVisibilityChange(updaterOrValue);
      }
    },
    [onColumnVisibilityChange, columnVisibility]
  );

  // eslint-disable-next-line react-hooks/incompatible-library -- TanStack Table is compatible
  const table = useReactTable({
    data,
    columns,
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: getSortedRowModel(),
    onSortingChange: setSorting,
    onColumnVisibilityChange: handleColumnVisibilityChange,
    sortDescFirst: false, // First click sorts ascending (arrow up)
    state: {
      sorting,
      columnVisibility: columnVisibility ?? {},
    },
    getRowId: getRowId
      ? (row) => String(getRowId(row))
      : (row, index) => String(index),
  });

  const rows = table.getRowModel().rows;

  // Expose method to focus selected row (for returning focus from drawer)
  useImperativeHandle(ref, () => ({
    focusSelectedRow: () => {
      // Find selected row index
      const selectedIndex = rows.findIndex(
        (row) => selectedRowId !== null && selectedRowId !== undefined && row.id === String(selectedRowId)
      );
      const targetIndex = selectedIndex >= 0 ? selectedIndex : 0;

      if (rows.length > 0) {
        setFocusedRowIndex(targetIndex);
        rowRefs.current.get(targetIndex)?.focus();
      }
    },
  }));

  // Handle keyboard navigation
  const handleRowKeyDown = useCallback(
    (event: React.KeyboardEvent<HTMLTableRowElement>, rowIndex: number, rowData: TData) => {
      switch (event.key) {
        case "Enter":
        case " ":
          event.preventDefault();
          onRowClick?.(rowData);
          break;
        case "ArrowDown":
          event.preventDefault();
          if (rowIndex < rows.length - 1) {
            setFocusedRowIndex(rowIndex + 1);
            rowRefs.current.get(rowIndex + 1)?.focus();
          }
          break;
        case "ArrowUp":
          event.preventDefault();
          if (rowIndex > 0) {
            setFocusedRowIndex(rowIndex - 1);
            rowRefs.current.get(rowIndex - 1)?.focus();
          }
          break;
        case "Home":
          event.preventDefault();
          setFocusedRowIndex(0);
          rowRefs.current.get(0)?.focus();
          break;
        case "End":
          event.preventDefault();
          setFocusedRowIndex(rows.length - 1);
          rowRefs.current.get(rows.length - 1)?.focus();
          break;
      }
    },
    [onRowClick, rows.length]
  );

  // Loading state
  if (isLoading) {
    return (
      <div className="flex-1 flex items-center justify-center">
        <Loader size="md" />
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
    <div className="flex-1 flex flex-col overflow-hidden">
      {/* Fixed header - outside scroll area */}
      <div className="flex-shrink-0 bg-background border-b border-border">
        <table className="w-full border-collapse text-sm table-fixed">
          <thead>
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
                      style={{ width: header.getSize() !== 150 ? header.getSize() : undefined }}
                      className="px-3 pt-3 pb-2 text-left font-semibold text-muted-foreground text-sm align-bottom"
                    >
                      {canSort ? (
                        <button
                          type="button"
                          onClick={header.column.getToggleSortingHandler()}
                          className={cn(
                            "group flex items-end gap-0.5 cursor-pointer select-none focus:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-1 rounded",
                            sorted ? "text-primary" : "hover:text-foreground"
                          )}
                          aria-label={`Sort by ${typeof headerContent === "string" ? headerContent : header.id}`}
                        >
                          {headerContent}
                          <span className={cn(
                            "opacity-0 group-hover:opacity-100 transition-opacity",
                            sorted && "text-primary"
                          )}>
                            {sorted === "asc" ? (
                              <ChevronUp className="h-4 w-4" />
                            ) : (
                              <ChevronDown className="h-4 w-4" />
                            )}
                          </span>
                        </button>
                      ) : (
                        <div className="flex items-end gap-1">{headerContent}</div>
                      )}
                    </th>
                  );
                })}
              </tr>
            ))}
          </thead>
        </table>
      </div>

      {/* Scrollable body */}
      <div className="flex-1 overflow-auto">
        <table className="w-full border-collapse text-sm table-fixed">
          <tbody ref={tableBodyRef} role="rowgroup">
            {rows.map((row, rowIndex) => {
              const isSelected =
                selectedRowId !== null &&
                selectedRowId !== undefined &&
                row.id === String(selectedRowId);
              const isFocused = focusedRowIndex === rowIndex;

              return (
                <tr
                  key={row.id}
                  ref={(el) => {
                    if (el) {
                      rowRefs.current.set(rowIndex, el);
                    } else {
                      rowRefs.current.delete(rowIndex);
                    }
                  }}
                  tabIndex={isFocused || (focusedRowIndex === -1 && rowIndex === 0) ? 0 : -1}
                  role="row"
                  aria-selected={isSelected}
                  onClick={() => onRowClick?.(row.original)}
                  onKeyDown={(e) => handleRowKeyDown(e, rowIndex, row.original)}
                  onFocus={() => setFocusedRowIndex(rowIndex)}
                  className={cn(
                    "border-b border-border transition-colors cursor-pointer",
                    "focus:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-inset",
                    isSelected
                      ? "bg-primary/10 border-l-2 border-l-primary"
                      : "hover:bg-accent/50"
                  )}
                >
                  {row.getVisibleCells().map((cell) => (
                    <td
                      key={cell.id}
                      style={{ width: cell.column.getSize() !== 150 ? cell.column.getSize() : undefined }}
                      className="px-3 py-2.5"
                    >
                      {flexRender(cell.column.columnDef.cell, cell.getContext())}
                    </td>
                  ))}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

/**
 * DataTable with forwardRef for imperative handle
 * This allows parent components to call focusSelectedRow() when drawer closes
 */
export const DataTable = forwardRef(DataTableInner) as <TData>(
  props: DataTableProps<TData> & { ref?: React.ForwardedRef<DataTableHandle> }
) => ReturnType<typeof DataTableInner>;

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
    <div className={cn(
      "group flex items-center gap-0.5",
      sorted && "text-primary"
    )}>
      {children}
      {column.getCanSort() && (
        <span className="opacity-0 group-hover:opacity-100 transition-opacity">
          {sorted === "asc" ? (
            <ChevronUp className="h-4 w-4" />
          ) : (
            <ChevronDown className="h-4 w-4" />
          )}
        </span>
      )}
    </div>
  );
}
