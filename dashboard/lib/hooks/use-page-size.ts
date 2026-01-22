"use client";

import { useState, useCallback, useEffect } from "react";

const STORAGE_KEY_PREFIX = "pageSize:";
const DEFAULT_PAGE_SIZE = 50;

/**
 * Hook for managing page size state with localStorage persistence
 *
 * @param tableId - Unique identifier for the table (e.g., "matches", "jobs")
 * @param defaultSize - Default page size (defaults to 50)
 * @returns Object with pageSize state and setter
 */
export function usePageSize(
  tableId: string,
  defaultSize: number = DEFAULT_PAGE_SIZE
) {
  const storageKey = `${STORAGE_KEY_PREFIX}${tableId}`;

  // Initialize state from localStorage or defaults
  const [pageSize, setPageSizeState] = useState<number>(() => {
    if (typeof window === "undefined") {
      return defaultSize;
    }
    try {
      const stored = localStorage.getItem(storageKey);
      if (stored) {
        const parsed = parseInt(stored, 10);
        if (!isNaN(parsed) && parsed > 0) {
          return parsed;
        }
      }
    } catch {
      // Ignore parsing errors
    }
    return defaultSize;
  });

  // Persist to localStorage when pageSize changes
  useEffect(() => {
    if (typeof window === "undefined") return;
    try {
      localStorage.setItem(storageKey, String(pageSize));
    } catch {
      // Ignore storage errors
    }
  }, [storageKey, pageSize]);

  // Setter that also resets to page 1 (returned separately for flexibility)
  const setPageSize = useCallback((size: number) => {
    setPageSizeState(size);
  }, []);

  return {
    pageSize,
    setPageSize,
  };
}
