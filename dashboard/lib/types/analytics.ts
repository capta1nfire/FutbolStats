/**
 * Analytics Types
 *
 * Types for analytics reports and metrics
 */

export type AnalyticsReportType =
  | "model_performance"
  | "prediction_accuracy"
  | "system_metrics"
  | "api_usage";

export type AnalyticsReportStatus = "ok" | "warning" | "stale";

export type AnalyticsTimeRange = "7d" | "30d" | "90d";

/**
 * Analytics report row for table display
 */
export interface AnalyticsReportRow {
  id: number;
  type: AnalyticsReportType;
  title: string;
  periodLabel: string; // e.g. "Last 7 days"
  lastUpdated: string; // ISO timestamp
  status?: AnalyticsReportStatus;
  summary: Record<string, string | number>; // e.g. accuracy, brier, p95, errorRate
}

/**
 * Breakdown table for detail view
 */
export interface AnalyticsBreakdownTable {
  columns: string[];
  rows: (string | number)[][];
}

/**
 * Series placeholder for trends (no charts yet)
 */
export interface AnalyticsSeriesPlaceholder {
  label: string;
  points: number;
}

/**
 * Full analytics report with details
 */
export interface AnalyticsReportDetail {
  row: AnalyticsReportRow;
  breakdownTable?: AnalyticsBreakdownTable;
  seriesPlaceholder?: AnalyticsSeriesPlaceholder[];
}

/**
 * Filters for analytics reports
 */
export interface AnalyticsFilters {
  type?: AnalyticsReportType[];
  timeRange?: AnalyticsTimeRange;
  league?: string[];
  search?: string;
}

/**
 * Report type labels for display
 */
export const ANALYTICS_REPORT_TYPE_LABELS: Record<AnalyticsReportType, string> = {
  model_performance: "Model Performance",
  prediction_accuracy: "Prediction Accuracy",
  system_metrics: "System Metrics",
  api_usage: "API Usage",
};

/**
 * All report types for filtering
 */
export const ANALYTICS_REPORT_TYPES: AnalyticsReportType[] = [
  "model_performance",
  "prediction_accuracy",
  "system_metrics",
  "api_usage",
];

/**
 * Time range labels
 */
export const ANALYTICS_TIME_RANGE_LABELS: Record<AnalyticsTimeRange, string> = {
  "7d": "Last 7 days",
  "30d": "Last 30 days",
  "90d": "Last 90 days",
};

/**
 * All time ranges
 */
export const ANALYTICS_TIME_RANGES: AnalyticsTimeRange[] = ["7d", "30d", "90d"];

/**
 * Status labels
 */
export const ANALYTICS_STATUS_LABELS: Record<AnalyticsReportStatus, string> = {
  ok: "OK",
  warning: "Warning",
  stale: "Stale",
};

/**
 * All statuses
 */
export const ANALYTICS_STATUSES: AnalyticsReportStatus[] = ["ok", "warning", "stale"];
