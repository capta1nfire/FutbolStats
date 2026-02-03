/**
 * Analytics mock data
 * Provides deterministic data for analytics reports
 */

import {
  AnalyticsReportRow,
  AnalyticsReportDetail,
  AnalyticsFilters,
  AnalyticsReportType,
  AnalyticsReportStatus,
  AnalyticsBreakdownTable,
  AnalyticsSeriesPlaceholder,
} from "@/lib/types";
import { mockConfig, simulateDelay, checkMockError } from "./config";

/**
 * Static base timestamp for deterministic mock data
 */
const BASE_TIMESTAMP = new Date("2026-01-20T12:00:00Z").getTime();

/**
 * Predefined reports with realistic data
 */
const reportDefinitions: Omit<AnalyticsReportRow, "id" | "lastUpdated">[] = [
  // Model Performance reports
  {
    type: "model_performance",
    title: "XGBoost v1.0.0 - Overall",
    periodLabel: "Last 7 days",
    status: "ok",
    summary: { accuracy: "68.2%", brier: 0.198, logLoss: 0.542, samples: 156 },
  },
  {
    type: "model_performance",
    title: "XGBoost v1.0.0 - La Liga",
    periodLabel: "Last 7 days",
    status: "ok",
    summary: { accuracy: "71.4%", brier: 0.185, logLoss: 0.521, samples: 28 },
  },
  {
    type: "model_performance",
    title: "XGBoost v1.0.0 - Premier League",
    periodLabel: "Last 7 days",
    status: "warning",
    summary: { accuracy: "62.5%", brier: 0.221, logLoss: 0.589, samples: 32 },
  },
  {
    type: "model_performance",
    title: "Shadow Model - Two-Stage",
    periodLabel: "Last 30 days",
    status: "stale",
    summary: { accuracy: "65.8%", brier: 0.205, logLoss: 0.558, samples: 412 },
  },
  // Prediction Accuracy reports
  {
    type: "prediction_accuracy",
    title: "1X2 Predictions - All Leagues",
    periodLabel: "Last 30 days",
    status: "ok",
    summary: { homeWinRate: "42.3%", drawRate: "28.1%", awayWinRate: "29.6%", total: 523 },
  },
  {
    type: "prediction_accuracy",
    title: "High Confidence Picks (>70%)",
    periodLabel: "Last 7 days",
    status: "ok",
    summary: { accuracy: "74.2%", roi: "+8.3%", picks: 31, avgOdds: 1.52 },
  },
  {
    type: "prediction_accuracy",
    title: "Upset Predictions Analysis",
    periodLabel: "Last 30 days",
    status: "warning",
    summary: { correctUpsets: 12, missedUpsets: 8, falseUpsets: 15, precision: "44.4%" },
  },
  // System Metrics reports
  {
    type: "system_metrics",
    title: "API Response Times",
    periodLabel: "Last 24 hours",
    status: "ok",
    summary: { p50: "45ms", p95: "120ms", p99: "280ms", errorRate: "0.02%" },
  },
  {
    type: "system_metrics",
    title: "Job Execution Stats",
    periodLabel: "Last 7 days",
    status: "ok",
    summary: { totalRuns: 2156, successRate: "99.2%", avgDuration: "2.3s", failures: 17 },
  },
  {
    type: "system_metrics",
    title: "Database Performance",
    periodLabel: "Last 24 hours",
    status: "warning",
    summary: { queryP95: "85ms", connections: 12, poolUsage: "78%", slowQueries: 3 },
  },
  // API Usage reports
  {
    type: "api_usage",
    title: "API-Football Consumption",
    periodLabel: "Last 30 days",
    status: "ok",
    summary: { requests: 45230, daily: 1507, remaining: 104770, plan: "Mega" },
  },
  {
    type: "api_usage",
    title: "iOS App API Calls",
    periodLabel: "Last 7 days",
    status: "ok",
    summary: { requests: 8420, uniqueUsers: 156, avgPerUser: 54, peakHour: "20:00" },
  },
];

/**
 * Create breakdown table for a report type
 */
function createBreakdownTable(type: AnalyticsReportType): AnalyticsBreakdownTable {
  switch (type) {
    case "model_performance":
      return {
        columns: ["League", "Matches", "Accuracy", "Brier", "ROI"],
        rows: [
          ["La Liga", 28, "71.4%", 0.185, "+5.2%"],
          ["Premier League", 32, "62.5%", 0.221, "-3.1%"],
          ["Bundesliga", 24, "70.8%", 0.189, "+4.8%"],
          ["Serie A", 30, "66.7%", 0.201, "+1.2%"],
          ["Ligue 1", 22, "68.2%", 0.195, "+2.5%"],
        ],
      };
    case "prediction_accuracy":
      return {
        columns: ["Prediction", "Count", "Correct", "Accuracy", "Avg Odds"],
        rows: [
          ["Home Win", 221, 148, "67.0%", 1.85],
          ["Draw", 147, 89, "60.5%", 3.42],
          ["Away Win", 155, 102, "65.8%", 2.65],
        ],
      };
    case "system_metrics":
      return {
        columns: ["Endpoint", "Requests", "P50", "P95", "Errors"],
        rows: [
          ["/predictions/upcoming", 12450, "32ms", "85ms", 2],
          ["/live-summary", 8920, "45ms", "120ms", 0],
          ["/health", 24560, "5ms", "12ms", 0],
          ["/dashboard/ops.json", 3240, "65ms", "180ms", 1],
        ],
      };
    case "api_usage":
      return {
        columns: ["Day", "Requests", "Fixtures", "Live", "Stats"],
        rows: [
          ["Mon", 1520, 450, 820, 250],
          ["Tue", 1380, 420, 710, 250],
          ["Wed", 1650, 480, 890, 280],
          ["Thu", 1420, 430, 750, 240],
          ["Fri", 1580, 460, 860, 260],
          ["Sat", 2100, 580, 1200, 320],
          ["Sun", 1980, 550, 1150, 280],
        ],
      };
  }
}

/**
 * Create series placeholder for trends
 */
function createSeriesPlaceholder(type: AnalyticsReportType): AnalyticsSeriesPlaceholder[] {
  switch (type) {
    case "model_performance":
      return [
        { label: "Accuracy (7d avg)", points: 7 },
        { label: "Brier Score (7d avg)", points: 7 },
      ];
    case "prediction_accuracy":
      return [
        { label: "Daily Accuracy", points: 30 },
        { label: "Cumulative ROI", points: 30 },
      ];
    case "system_metrics":
      return [
        { label: "P95 Latency (hourly)", points: 24 },
        { label: "Error Rate (hourly)", points: 24 },
      ];
    case "api_usage":
      return [
        { label: "Daily Requests", points: 30 },
        { label: "Remaining Quota", points: 30 },
      ];
  }
}

/**
 * Create deterministic reports
 */
function createReports(count: number): AnalyticsReportRow[] {
  return Array.from({ length: count }, (_, i) => {
    const def = reportDefinitions[i % reportDefinitions.length];
    return {
      ...def,
      id: 2000 + i,
      lastUpdated: new Date(BASE_TIMESTAMP - i * 3600000).toISOString(),
    };
  });
}

// Pre-generated datasets
const normalDataset = createReports(12);
const largeDataset = createReports(50);

/**
 * Get analytics reports based on scenario
 */
export async function getAnalyticsReportsMock(
  filters?: AnalyticsFilters
): Promise<AnalyticsReportRow[]> {
  await simulateDelay();
  checkMockError();

  let data: AnalyticsReportRow[];

  switch (mockConfig.scenario) {
    case "empty":
      data = [];
      break;
    case "large":
      data = [...largeDataset];
      break;
    default:
      data = [...normalDataset];
  }

  // Apply filters
  if (filters) {
    if (filters.type && filters.type.length > 0) {
      data = data.filter((r) => filters.type!.includes(r.type));
    }
    if (filters.search) {
      const search = filters.search.toLowerCase();
      data = data.filter(
        (r) =>
          r.title.toLowerCase().includes(search) ||
          r.type.toLowerCase().includes(search) ||
          r.periodLabel.toLowerCase().includes(search)
      );
    }
  }

  return data;
}

/**
 * Get a single report by ID with full details
 */
export async function getAnalyticsReportMock(
  id: number
): Promise<AnalyticsReportDetail | null> {
  await simulateDelay(300);
  checkMockError();

  const allReports = mockConfig.scenario === "large" ? largeDataset : normalDataset;
  const report = allReports.find((r) => r.id === id);

  if (!report) return null;

  return {
    row: report,
    breakdownTable: createBreakdownTable(report.type),
    seriesPlaceholder: createSeriesPlaceholder(report.type),
  };
}

/**
 * Get counts per report type
 */
export function getAnalyticsTypeCountsMock(): Record<AnalyticsReportType, number> {
  const allReports = mockConfig.scenario === "large" ? largeDataset : normalDataset;
  return {
    model_performance: allReports.filter((r) => r.type === "model_performance").length,
    prediction_accuracy: allReports.filter((r) => r.type === "prediction_accuracy").length,
    system_metrics: allReports.filter((r) => r.type === "system_metrics").length,
    api_usage: allReports.filter((r) => r.type === "api_usage").length,
  };
}

/**
 * Get counts per status
 */
export function getAnalyticsStatusCountsMock(): Record<AnalyticsReportStatus, number> {
  const allReports = mockConfig.scenario === "large" ? largeDataset : normalDataset;
  return {
    ok: allReports.filter((r) => r.status === "ok").length,
    warning: allReports.filter((r) => r.status === "warning").length,
    stale: allReports.filter((r) => r.status === "stale").length,
  };
}
