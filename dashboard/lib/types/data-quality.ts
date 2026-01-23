/**
 * Data Quality Types
 *
 * Types for data quality checks and monitoring
 */

export type DataQualityStatus = "passing" | "warning" | "failing";

export type DataQualityCategory =
  | "coverage"
  | "consistency"
  | "completeness"
  | "freshness"
  | "odds";

export interface DataQualityCheck {
  id: string; // Backend uses string IDs like "dq_quarantined_odds_24h"
  name: string;
  category: DataQualityCategory;
  status: DataQualityStatus;
  lastRunAt: string; // ISO timestamp
  currentValue?: number | string;
  threshold?: number | string;
  affectedCount: number;
  description?: string;
}

export interface DataQualityAffectedItem {
  id: string | number;
  label: string;
  kind: "match" | "team" | "league" | "job";
  details?: Record<string, string | number>;
}

export interface DataQualityFilters {
  status?: DataQualityStatus[];
  category?: DataQualityCategory[];
  search?: string;
}

/**
 * Data quality check with affected items (for detail view)
 */
export interface DataQualityCheckDetail extends DataQualityCheck {
  affectedItems: DataQualityAffectedItem[];
  history: DataQualityHistoryEntry[];
}

export interface DataQualityHistoryEntry {
  timestamp: string; // ISO
  status: DataQualityStatus;
  value?: number | string;
}

/**
 * Category labels for display
 */
export const DATA_QUALITY_CATEGORY_LABELS: Record<DataQualityCategory, string> = {
  coverage: "Coverage",
  consistency: "Consistency",
  completeness: "Completeness",
  freshness: "Freshness",
  odds: "Odds",
};

/**
 * All categories for filtering
 */
export const DATA_QUALITY_CATEGORIES: DataQualityCategory[] = [
  "coverage",
  "consistency",
  "completeness",
  "freshness",
  "odds",
];

/**
 * All statuses for filtering
 */
export const DATA_QUALITY_STATUSES: DataQualityStatus[] = [
  "passing",
  "warning",
  "failing",
];
