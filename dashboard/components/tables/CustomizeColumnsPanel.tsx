"use client";

import { useMemo, useCallback } from "react";
import { VisibilityState } from "@tanstack/react-table";
import { ChevronLeft } from "lucide-react";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
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
  /** Whether the panel is visible */
  open: boolean;
  /** Available columns to customize */
  columns: ColumnOption[];
  /** Current visibility state */
  columnVisibility: VisibilityState;
  /** Called when a column's visibility changes */
  onColumnVisibilityChange: (columnId: string, visible: boolean) => void;
  /** Called when "Restore" is clicked */
  onRestore: () => void;
  /** Called when "Done" is clicked (closes panel) */
  onDone: () => void;
  /** Called when collapse button is clicked (collapses entire Left Rail) */
  onCollapse?: () => void;
  /** Additional class names */
  className?: string;
}

/**
 * Customize Columns Panel (UniFi style)
 *
 * A separate column that appears between Filters and the table.
 * Triggered by "Customize Columns" link in FilterPanel footer.
 *
 * Layout (UniFi reference):
 * - Col 1: Filters (Status, League, etc.)
 * - Col 2: Customize Columns (this component) - appears when open=true
 * - Col 3: Table
 *
 * UI:
 * - Header "Customize Columns"
 * - "All" checkbox with indeterminate state
 * - List of column checkboxes with scroll
 * - Footer: "Restore" (defaults) + "Done" (closes panel)
 *
 * Changes apply immediately (no confirmation).
 */
export function CustomizeColumnsPanel({
  open,
  columns,
  columnVisibility,
  onColumnVisibilityChange,
  onRestore,
  onDone,
  onCollapse,
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

  // Don't render if not open
  if (!open) {
    return null;
  }

  return (
    <aside
      className={cn(
        "w-[200px] border-r border-border bg-sidebar flex flex-col shrink-0",
        className
      )}
    >
      {/* Header - same height as FilterPanel header, with collapse button */}
      <div className="h-12 flex items-center justify-between px-3">
        <h3 className="text-sm font-medium text-foreground">
          Customize Columns
        </h3>
        {onCollapse && (
          <Button
            variant="ghost"
            size="icon"
            onClick={onCollapse}
            className="h-8 w-8"
            aria-label="Collapse filters"
          >
            <ChevronLeft className="h-4 w-4" />
          </Button>
        )}
      </div>

      {/* "All" checkbox */}
      <div className="px-3 py-2">
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
      <ScrollArea className="flex-1">
        <div className="px-3 py-2 space-y-0.5">
          {hideableColumns.map((column) => {
            const isVisible = columnVisibility[column.id] !== false;

            return (
              <label
                key={column.id}
                className="flex items-center gap-2 py-1.5 cursor-pointer hover:bg-accent/30 rounded px-1 -mx-1"
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

      {/* Footer: Restore + Done (UniFi style - text links, same height as FilterPanel footer) */}
      <div className="px-3 py-3 flex items-center justify-between">
        <button
          onClick={onRestore}
          className="text-sm font-medium text-primary hover:text-primary-hover transition-colors"
        >
          Restore
        </button>
        <button
          onClick={onDone}
          className="text-sm font-semibold text-primary hover:text-primary-hover transition-colors"
        >
          Done
        </button>
      </div>
    </aside>
  );
}
