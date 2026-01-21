"use client";

import { useMemo, useCallback } from "react";
import { VisibilityState } from "@tanstack/react-table";
import { cn } from "@/lib/utils";
import { Checkbox } from "@/components/ui/checkbox";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";

/**
 * Column definition for the customize panel
 */
export interface ColumnOption {
  /** Column ID (matches TanStack column id) */
  id: string;
  /** Display label for the column */
  label: string;
  /** Whether this column can be hidden (false = always visible) */
  enableHiding?: boolean;
}

interface CustomizeColumnsPanelProps {
  /** Available columns to customize */
  columns: ColumnOption[];
  /** Current visibility state */
  columnVisibility: VisibilityState;
  /** Called when a column's visibility changes */
  onColumnVisibilityChange: (columnId: string, visible: boolean) => void;
  /** Called when "Restore" is clicked */
  onRestore: () => void;
  /** Called when "Done" is clicked (collapses Left Rail) */
  onDone: () => void;
  /** Additional class names */
  className?: string;
}

/**
 * Customize Columns Panel (UniFi style)
 *
 * Lives inside the Left Rail (column 2) below FilterPanel.
 * - Header "Customize Columns"
 * - "All" checkbox with indeterminate state
 * - List of column checkboxes with scroll
 * - Footer: "Restore" (defaults) + "Done" (collapses rail)
 *
 * Changes apply immediately (no confirmation).
 */
export function CustomizeColumnsPanel({
  columns,
  columnVisibility,
  onColumnVisibilityChange,
  onRestore,
  onDone,
  className,
}: CustomizeColumnsPanelProps) {
  // Filter to only hideable columns
  const hideableColumns = useMemo(
    () => columns.filter((col) => col.enableHiding !== false),
    [columns]
  );

  // Calculate "All" checkbox state
  const { allChecked, allIndeterminate } = useMemo(() => {
    const visibleCount = hideableColumns.filter(
      (col) => columnVisibility[col.id] !== false
    ).length;
    const total = hideableColumns.length;

    return {
      allChecked: visibleCount === total,
      allIndeterminate: visibleCount > 0 && visibleCount < total,
    };
  }, [hideableColumns, columnVisibility]);

  // Handle "All" checkbox
  const handleAllChange = useCallback(
    (checked: boolean) => {
      hideableColumns.forEach((col) => {
        onColumnVisibilityChange(col.id, checked);
      });
    },
    [hideableColumns, onColumnVisibilityChange]
  );

  // Handle individual column checkbox
  const handleColumnChange = useCallback(
    (columnId: string, checked: boolean) => {
      onColumnVisibilityChange(columnId, checked);
    },
    [onColumnVisibilityChange]
  );

  return (
    <div
      className={cn(
        "flex flex-col border-t border-border bg-surface",
        className
      )}
    >
      {/* Header */}
      <div className="px-3 py-2 border-b border-border">
        <h3 className="text-xs font-semibold text-foreground uppercase tracking-wider">
          Customize Columns
        </h3>
      </div>

      {/* "All" checkbox */}
      <div className="px-3 py-2 border-b border-border">
        <label className="flex items-center gap-2 cursor-pointer">
          <Checkbox
            checked={allIndeterminate ? "indeterminate" : allChecked}
            onCheckedChange={(checked) => handleAllChange(checked === true)}
            aria-label="Toggle all columns"
          />
          <span className="text-sm text-foreground">All</span>
        </label>
      </div>

      {/* Column list with scroll */}
      <ScrollArea className="flex-1 max-h-[280px]">
        <div className="px-3 py-2 space-y-1">
          {hideableColumns.map((column) => {
            const isVisible = columnVisibility[column.id] !== false;

            return (
              <label
                key={column.id}
                className="flex items-center gap-2 py-1 cursor-pointer hover:bg-accent/30 rounded px-1 -mx-1"
              >
                <Checkbox
                  checked={isVisible}
                  onCheckedChange={(checked) =>
                    handleColumnChange(column.id, checked === true)
                  }
                  aria-label={`Show ${column.label} column`}
                />
                <span className="text-sm text-foreground truncate">
                  {column.label}
                </span>
              </label>
            );
          })}
        </div>
      </ScrollArea>

      {/* Footer: Restore + Done */}
      <div className="px-3 py-2 border-t border-border flex items-center gap-2">
        <Button
          variant="ghost"
          size="sm"
          onClick={onRestore}
          className="flex-1 text-xs"
        >
          Restore
        </Button>
        <Button
          variant="secondary"
          size="sm"
          onClick={onDone}
          className="flex-1 text-xs"
        >
          Done
        </Button>
      </div>
    </div>
  );
}
