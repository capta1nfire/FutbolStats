"use client";

/**
 * URL State Utilities
 *
 * Shared utilities for persisting filter state in URL query parameters.
 * Convention (SSOT Section 13):
 * - selected id: ?id=123
 * - search: ?q=term
 * - multiselect filters: repeated params (e.g., ?status=live&status=ft)
 * - single select: ?range=24h
 */

import { useSearchParams, useRouter, usePathname } from "next/navigation";
import { useCallback, useMemo } from "react";

/**
 * Parse a numeric ID from URL parameter
 * Returns null if invalid (non-numeric, NaN, negative)
 */
export function parseNumericId(param: string | null): number | null {
  if (!param) return null;
  const parsed = parseInt(param, 10);
  if (isNaN(parsed) || parsed < 0) return null;
  return parsed;
}

/**
 * Parse an array of strings from repeated URL parameters
 * e.g., ?status=live&status=ft → ["live", "ft"]
 */
export function parseArrayParam<T extends string>(
  searchParams: URLSearchParams,
  key: string,
  validValues?: readonly T[]
): T[] {
  const values = searchParams.getAll(key);
  if (!validValues) return values as T[];
  return values.filter((v): v is T => validValues.includes(v as T));
}

/**
 * Parse a single string param with validation against allowed values
 */
export function parseSingleParam<T extends string>(
  param: string | null,
  validValues: readonly T[]
): T | null {
  if (!param) return null;
  if (validValues.includes(param as T)) return param as T;
  return null;
}

/**
 * Build URLSearchParams from filter state
 * Handles arrays (repeated params), single values, and the id param
 */
export function buildSearchParams(
  filters: Record<string, string | string[] | number | null | undefined>,
  options?: { preserveEmpty?: boolean }
): URLSearchParams {
  const params = new URLSearchParams();

  for (const [key, value] of Object.entries(filters)) {
    if (value === null || value === undefined) continue;
    if (value === "" && !options?.preserveEmpty) continue;

    if (Array.isArray(value)) {
      // Repeated params for arrays
      for (const v of value) {
        if (v) params.append(key, v);
      }
    } else if (typeof value === "number") {
      params.set(key, String(value));
    } else if (value) {
      params.set(key, value);
    }
  }

  return params;
}

/**
 * Serialize filters to URL search string
 * Returns empty string if no filters active
 */
export function serializeFilters(
  filters: Record<string, string | string[] | number | null | undefined>
): string {
  const params = buildSearchParams(filters);
  const str = params.toString();
  return str ? `?${str}` : "";
}

/**
 * Hook to manage URL state for a page with filters
 * Returns parsed state and update functions
 */
export function useUrlFilters<TFilters extends Record<string, unknown>>(
  basePath: string,
  parseFilters: (searchParams: URLSearchParams) => TFilters
) {
  const router = useRouter();
  usePathname(); // keep for potential future use
  const searchParams = useSearchParams();

  // Parse current filters from URL
  const filters = useMemo(
    () => parseFilters(searchParams),
    [searchParams, parseFilters]
  );

  // Parse selected ID
  const selectedId = useMemo(
    () => parseNumericId(searchParams.get("id")),
    [searchParams]
  );

  // Parse search query
  const searchQuery = useMemo(
    () => searchParams.get("q") ?? "",
    [searchParams]
  );

  // Update URL with new filters (preserves id if present)
  const updateFilters = useCallback(
    (
      newFilters: Record<string, string | string[] | number | null | undefined>,
      options?: { preserveId?: boolean }
    ) => {
      const params = buildSearchParams(newFilters);

      // Preserve id if requested and present
      if (options?.preserveId && selectedId !== null) {
        params.set("id", String(selectedId));
      }

      const search = params.toString();
      router.replace(`${basePath}${search ? `?${search}` : ""}`, {
        scroll: false,
      });
    },
    [router, basePath, selectedId]
  );

  // Select an item (updates id, preserves filters)
  const selectItem = useCallback(
    (id: number) => {
      const params = new URLSearchParams(searchParams.toString());
      params.set("id", String(id));
      router.replace(`${basePath}?${params.toString()}`, { scroll: false });
    },
    [router, basePath, searchParams]
  );

  // Close selection (removes id, preserves filters)
  const clearSelection = useCallback(() => {
    const params = new URLSearchParams(searchParams.toString());
    params.delete("id");
    const search = params.toString();
    router.replace(`${basePath}${search ? `?${search}` : ""}`, {
      scroll: false,
    });
  }, [router, basePath, searchParams]);

  // Clear all filters (preserves id if drawer open)
  const clearFilters = useCallback(
    (options?: { preserveId?: boolean }) => {
      if (options?.preserveId && selectedId !== null) {
        router.replace(`${basePath}?id=${selectedId}`, { scroll: false });
      } else {
        router.replace(basePath, { scroll: false });
      }
    },
    [router, basePath, selectedId]
  );

  return {
    filters,
    selectedId,
    searchQuery,
    updateFilters,
    selectItem,
    clearSelection,
    clearFilters,
    searchParams,
  };
}

/**
 * Check if any filters are active (excluding id and q)
 */
export function hasActiveFilters(
  searchParams: URLSearchParams,
  filterKeys: string[]
): boolean {
  for (const key of filterKeys) {
    if (searchParams.has(key)) return true;
  }
  return false;
}

/**
 * Toggle a value in an array filter
 */
export function toggleArrayValue<T extends string>(
  current: T[],
  value: T,
  checked: boolean
): T[] {
  if (checked) {
    return current.includes(value) ? current : [...current, value];
  }
  return current.filter((v) => v !== value);
}
