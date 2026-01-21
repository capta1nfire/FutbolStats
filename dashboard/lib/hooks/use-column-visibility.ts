"use client";

import { useState, useCallback, useEffect } from "react";
import { VisibilityState } from "@tanstack/react-table";

const STORAGE_KEY_PREFIX = "columns:";

/**
 * Hook for managing column visibility state with localStorage persistence
 *
 * @param tableId - Unique identifier for the table (e.g., "matches", "jobs")
 * @param defaultVisibility - Default visibility state for columns
 * @returns Object with visibility state and handlers
 */
export function useColumnVisibility(
  tableId: string,
  defaultVisibility: VisibilityState = {}
) {
  const storageKey = `${STORAGE_KEY_PREFIX}${tableId}`;

  // Initialize state from localStorage or defaults
  const [columnVisibility, setColumnVisibility] = useState<VisibilityState>(() => {
    if (typeof window === "undefined") {
      return defaultVisibility;
    }
    try {
      const stored = localStorage.getItem(storageKey);
      if (stored) {
        return JSON.parse(stored) as VisibilityState;
      }
    } catch {
      // Ignore parsing errors
    }
    return defaultVisibility;
  });

  // Persist to localStorage when visibility changes
  useEffect(() => {
    if (typeof window === "undefined") return;
    try {
      localStorage.setItem(storageKey, JSON.stringify(columnVisibility));
    } catch {
      // Ignore storage errors
    }
  }, [storageKey, columnVisibility]);

  // Toggle a single column's visibility
  const toggleColumn = useCallback((columnId: string) => {
    setColumnVisibility((prev) => ({
      ...prev,
      [columnId]: prev[columnId] === false ? true : false,
    }));
  }, []);

  // Set visibility for a specific column
  const setColumnVisible = useCallback((columnId: string, visible: boolean) => {
    setColumnVisibility((prev) => ({
      ...prev,
      [columnId]: visible,
    }));
  }, []);

  // Set all columns visible/hidden (for "All" checkbox)
  const setAllColumnsVisible = useCallback(
    (columnIds: string[], visible: boolean) => {
      setColumnVisibility((prev) => {
        const newState = { ...prev };
        columnIds.forEach((id) => {
          newState[id] = visible;
        });
        return newState;
      });
    },
    []
  );

  // Reset to default visibility
  const resetToDefault = useCallback(() => {
    setColumnVisibility(defaultVisibility);
  }, [defaultVisibility]);

  // Check if a column is visible (undefined means visible by default)
  const isColumnVisible = useCallback(
    (columnId: string): boolean => {
      return columnVisibility[columnId] !== false;
    },
    [columnVisibility]
  );

  // Get count of visible columns from a list
  const getVisibleCount = useCallback(
    (columnIds: string[]): number => {
      return columnIds.filter((id) => columnVisibility[id] !== false).length;
    },
    [columnVisibility]
  );

  return {
    columnVisibility,
    setColumnVisibility,
    toggleColumn,
    setColumnVisible,
    setAllColumnsVisible,
    resetToDefault,
    isColumnVisible,
    getVisibleCount,
  };
}
