/**
 * Data Quality API Adapter
 *
 * Transforms backend /dashboard/data_quality.json response to dashboard types.
 * Best-effort parsing with null-safety.
 */

import {
  DataQualityCheck,
  DataQualityCheckDetail,
  DataQualityAffectedItem,
  DataQualityStatus,
  DataQualityCategory,
  DataQualityHistoryEntry,
} from "@/lib/types";

/**
 * Backend check item from /dashboard/data_quality.json
 * Note: Backend uses string IDs like "dq_quarantined_odds_24h"
 */
interface BackendCheck {
  id: string;
  name: string;
  category: string;
  status: string;
  last_run_at: string; // ISO timestamp
  current_value?: number | string;
  threshold?: number | string;
  affected_count: number;
  description?: string;
}

/**
 * Backend check detail from /dashboard/data_quality/{id}.json
 */
interface BackendCheckDetail extends BackendCheck {
  affected_items?: BackendAffectedItem[];
  history?: BackendHistoryEntry[];
}

/**
 * Backend affected item
 */
interface BackendAffectedItem {
  id: string | number;
  label: string;
  kind: string; // "match" | "team" | "league" | "job"
  details?: Record<string, string | number>;
}

/**
 * Backend history entry
 */
interface BackendHistoryEntry {
  timestamp: string; // ISO
  status: string;
  value?: number | string;
}

/**
 * Pagination info extracted from response
 */
export interface DataQualityPagination {
  page: number;
  limit: number;
  total: number;
  totalPages: number;
}

/**
 * Normalize status from backend format to frontend type
 * Backend may send: "passing", "warning", "failing", "PASSING", "WARN", etc.
 */
function normalizeStatus(status: unknown): DataQualityStatus {
  if (typeof status !== "string") return "warning";

  const normalized = status.toLowerCase();
  if (normalized === "passing" || normalized === "pass" || normalized === "ok") {
    return "passing";
  }
  if (normalized === "failing" || normalized === "fail" || normalized === "error" || normalized === "critical") {
    return "failing";
  }
  // Default to warning for unknown statuses
  return "warning";
}

/**
 * Normalize category from backend format to frontend type
 */
function normalizeCategory(category: unknown): DataQualityCategory {
  if (typeof category !== "string") return "coverage";

  const normalized = category.toLowerCase();
  if (normalized === "coverage") return "coverage";
  if (normalized === "consistency") return "consistency";
  if (normalized === "completeness") return "completeness";
  if (normalized === "freshness") return "freshness";
  if (normalized === "odds") return "odds";

  // Default to coverage for unknown
  return "coverage";
}

/**
 * Normalize affected item kind
 */
function normalizeKind(kind: unknown): DataQualityAffectedItem["kind"] {
  if (typeof kind !== "string") return "match";

  const normalized = kind.toLowerCase();
  if (normalized === "match") return "match";
  if (normalized === "team") return "team";
  if (normalized === "league") return "league";
  if (normalized === "job") return "job";

  return "match";
}

/**
 * Parse a single backend check to DataQualityCheck
 */
function parseCheck(item: BackendCheck): DataQualityCheck {
  // Ensure id is string
  const id = typeof item.id === "string" ? item.id : String(item.id);

  return {
    id,
    name: item.name || `Check ${id}`,
    category: normalizeCategory(item.category),
    status: normalizeStatus(item.status),
    lastRunAt: item.last_run_at || new Date().toISOString(),
    currentValue: item.current_value,
    threshold: item.threshold,
    affectedCount: typeof item.affected_count === "number" ? item.affected_count : 0,
    description: item.description,
  };
}

/**
 * Parse affected items
 */
function parseAffectedItems(items: unknown): DataQualityAffectedItem[] {
  if (!Array.isArray(items)) return [];

  return items.map((item) => {
    const i = item as BackendAffectedItem;
    return {
      id: i.id ?? 0,
      label: i.label || "Unknown",
      kind: normalizeKind(i.kind),
      details: i.details,
    };
  });
}

/**
 * Parse history entries
 */
function parseHistory(entries: unknown): DataQualityHistoryEntry[] {
  if (!Array.isArray(entries)) return [];

  return entries.map((entry) => {
    const e = entry as BackendHistoryEntry;
    return {
      timestamp: e.timestamp || new Date().toISOString(),
      status: normalizeStatus(e.status),
      value: e.value,
    };
  });
}

/**
 * Parse backend list response to DataQualityCheck[]
 *
 * Supports response shapes:
 * 1) { checks: [...] }
 * 2) { data: [...] } (array directly)
 * 3) { data: { checks: [...] } } (nested object with checks)
 * 4) Direct array [...]
 */
export function parseDataQualityChecks(data: unknown): DataQualityCheck[] {
  if (!data) {
    return [];
  }

  // Shape 4: Direct array
  if (Array.isArray(data)) {
    return data.map((item) => parseCheck(item as BackendCheck));
  }

  if (typeof data !== "object") {
    return [];
  }

  const obj = data as Record<string, unknown>;

  // Shape 1: { checks: [...] }
  if (Array.isArray(obj.checks)) {
    return obj.checks.map((item) => parseCheck(item as BackendCheck));
  }

  // Shape 2 or 3: { data: [...] } or { data: { checks: [...] } }
  if (obj.data) {
    // Shape 2: { data: [...] } (array directly)
    if (Array.isArray(obj.data)) {
      return obj.data.map((item) => parseCheck(item as BackendCheck));
    }

    // Shape 3: { data: { checks: [...] } } (nested object)
    if (typeof obj.data === "object") {
      const nested = obj.data as Record<string, unknown>;
      if (Array.isArray(nested.checks)) {
        return nested.checks.map((item) => parseCheck(item as BackendCheck));
      }
    }
  }

  // Best-effort: no crash, return empty
  return [];
}

/**
 * Extract pagination info from response
 *
 * Supports shapes:
 * 1) { pagination: { page, limit, total, total_pages } }
 * 2) { data: { total, page, limit, pages } }
 */
export function extractPagination(data: unknown): DataQualityPagination | null {
  if (!data || typeof data !== "object") {
    return null;
  }

  const obj = data as Record<string, unknown>;

  // Shape 1: { pagination: {...} }
  if (obj.pagination && typeof obj.pagination === "object") {
    const pagination = obj.pagination as Record<string, unknown>;
    return {
      page: typeof pagination.page === "number" ? pagination.page : 1,
      limit: typeof pagination.limit === "number" ? pagination.limit : 25,
      total: typeof pagination.total === "number" ? pagination.total : 0,
      totalPages: typeof pagination.total_pages === "number" ? pagination.total_pages : 1,
    };
  }

  // Shape 2: { data: { total, page, limit, pages } }
  if (obj.data && typeof obj.data === "object" && !Array.isArray(obj.data)) {
    const nested = obj.data as Record<string, unknown>;
    // Only treat as pagination if it has total/page/limit fields
    if (typeof nested.total === "number" || typeof nested.page === "number") {
      return {
        page: typeof nested.page === "number" ? nested.page : 1,
        limit: typeof nested.limit === "number" ? nested.limit : 25,
        total: typeof nested.total === "number" ? nested.total : 0,
        totalPages: typeof nested.pages === "number" ? nested.pages : 1,
      };
    }
  }

  return null;
}

/**
 * Parse backend check detail response to DataQualityCheckDetail
 *
 * Supports response shapes:
 * 1) Direct object { id, name, ... }
 * 2) Wrapped { data: { id, name, ... } }
 * 3) Wrapped { check: { id, name, ... } }
 * 4) Wrapped { data: { check: {...}, affected_items: [], history: [] } }
 */
export function parseDataQualityDetail(data: unknown): DataQualityCheckDetail | null {
  if (!data || typeof data !== "object") {
    return null;
  }

  const container = data as Record<string, unknown>;
  let checkData: Record<string, unknown>;
  let affectedItems: unknown = undefined;
  let history: unknown = undefined;

  // Shape 4: { data: { check: {...}, affected_items: [], history: [] } }
  if (container.data && typeof container.data === "object" && !Array.isArray(container.data)) {
    const dataObj = container.data as Record<string, unknown>;
    if (dataObj.check && typeof dataObj.check === "object") {
      checkData = dataObj.check as Record<string, unknown>;
      affectedItems = dataObj.affected_items;
      history = dataObj.history;
    } else {
      // Shape 2: { data: { id, name, ... } }
      checkData = dataObj;
    }
  } else if (container.check && typeof container.check === "object") {
    // Shape 3: { check: { id, name, ... } }
    checkData = container.check as Record<string, unknown>;
  } else {
    // Shape 1: Direct object { id, name, ... }
    checkData = container;
  }

  // Must have an id to be valid (string or number that can be converted)
  if (checkData.id === undefined || checkData.id === null) {
    return null;
  }

  const baseCheck = parseCheck(checkData as unknown as BackendCheck);

  // Get affected_items and history from check data if not already extracted
  if (affectedItems === undefined) {
    affectedItems = (checkData as unknown as BackendCheckDetail).affected_items;
  }
  if (history === undefined) {
    history = (checkData as unknown as BackendCheckDetail).history;
  }

  return {
    ...baseCheck,
    affectedItems: parseAffectedItems(affectedItems),
    history: parseHistory(history),
  };
}

/**
 * Calculate status counts from checks array
 */
export function calculateStatusCounts(
  checks: DataQualityCheck[]
): Record<DataQualityStatus, number> {
  const counts: Record<DataQualityStatus, number> = {
    passing: 0,
    warning: 0,
    failing: 0,
  };

  checks.forEach((check) => {
    counts[check.status]++;
  });

  return counts;
}

/**
 * Calculate category counts from checks array
 */
export function calculateCategoryCounts(
  checks: DataQualityCheck[]
): Record<DataQualityCategory, number> {
  const counts: Record<DataQualityCategory, number> = {
    coverage: 0,
    consistency: 0,
    completeness: 0,
    freshness: 0,
    odds: 0,
  };

  checks.forEach((check) => {
    counts[check.category]++;
  });

  return counts;
}
