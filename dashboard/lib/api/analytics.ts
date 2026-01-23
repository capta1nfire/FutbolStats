/**
 * Analytics API Adapter
 *
 * Transforms backend /dashboard/ops/* responses to dashboard types.
 * Best-effort parsing with null-safety.
 */

/**
 * Backend history rollup entry
 */
interface BackendHistoryEntry {
  day: string; // YYYY-MM-DD
  payload: {
    day: string;
    note?: string;
    pit_snapshots_total: number;
    pit_snapshots_live: number;
    pit_bets_evaluable: number;
    baseline_coverage: {
      pit_total: number;
      pit_with_market_baseline: number;
      baseline_pct: number;
    };
    market_movement: {
      total: number;
      by_type: Record<string, number>;
    };
    errors_summary: {
      api_429_full: number;
      api_429_critical: number;
      timeouts_full: number;
      timeouts_critical: number;
      budget_used?: number | null;
      budget_limit?: number | null;
      budget_pct?: number | null;
    };
    by_league: Record<string, unknown>;
  };
  updated_at: string;
}

/**
 * Backend history response
 */
interface BackendHistoryResponse {
  days_requested: number;
  days_available: number;
  history: BackendHistoryEntry[];
}

/**
 * Backend predictions performance response
 */
interface BackendPerformanceResponse {
  global: {
    n: number;
    n_with_odds?: number;
    coverage_pct?: number;
    metrics: {
      accuracy: {
        pct: number;
        total: number;
        correct: number;
      };
      log_loss: number;
      brier_score: number;
      calibration: Array<{
        bin: string;
        count: number;
        avg_confidence: number;
        empirical_accuracy: number;
        calibration_error: number | null;
      }>;
      confusion_matrix?: {
        labels: string[];
        matrix: number[][];
        description: string;
      };
      market_comparison?: {
        market_brier: number;
        skill_vs_market: number;
        interpretation: string;
      };
    };
  };
  by_league: Record<string, {
    n: number;
    metrics: {
      accuracy: {
        pct: number;
        total: number;
        correct: number;
      };
      log_loss: number;
      brier_score: number;
    };
  }>;
}

/**
 * Parsed history rollup for display
 */
export interface OpsHistoryRollup {
  day: string;
  pitSnapshots: number;
  pitLive: number;
  evaluableBets: number;
  baselineCoverage: number; // percentage
  marketMovements: number;
  errors429: number;
  timeouts: number;
  note?: string;
  updatedAt: string;
}

/**
 * Parsed predictions performance
 */
export interface PredictionsPerformance {
  totalPredictions: number;
  accuracy: number; // percentage
  brierScore: number;
  logLoss: number;
  marketBrier?: number;
  skillVsMarket?: number;
  coveragePct?: number;
  byLeague: Array<{
    leagueId: string;
    n: number;
    accuracy: number;
    brierScore: number;
  }>;
  calibration: Array<{
    bin: string;
    count: number;
    avgConfidence: number;
    empiricalAccuracy: number;
    calibrationError: number | null;
  }>;
}

/**
 * Parse history response
 */
export function parseOpsHistory(data: unknown): OpsHistoryRollup[] {
  if (!data || typeof data !== "object") {
    return [];
  }

  const response = data as BackendHistoryResponse;
  const history = response.history;

  if (!Array.isArray(history)) {
    return [];
  }

  return history.map((entry) => {
    const payload = entry.payload || {};
    const baseline = payload.baseline_coverage || {};
    const errors = payload.errors_summary || {};
    const market = payload.market_movement || {};

    return {
      day: entry.day || payload.day || "unknown",
      pitSnapshots: typeof payload.pit_snapshots_total === "number" ? payload.pit_snapshots_total : 0,
      pitLive: typeof payload.pit_snapshots_live === "number" ? payload.pit_snapshots_live : 0,
      evaluableBets: typeof payload.pit_bets_evaluable === "number" ? payload.pit_bets_evaluable : 0,
      baselineCoverage: typeof baseline.baseline_pct === "number" ? baseline.baseline_pct : 0,
      marketMovements: typeof market.total === "number" ? market.total : 0,
      errors429: (errors.api_429_full || 0) + (errors.api_429_critical || 0),
      timeouts: (errors.timeouts_full || 0) + (errors.timeouts_critical || 0),
      note: payload.note,
      updatedAt: entry.updated_at || new Date().toISOString(),
    };
  });
}

/**
 * Parse predictions performance response
 */
export function parsePredictionsPerformance(data: unknown): PredictionsPerformance | null {
  if (!data || typeof data !== "object") {
    return null;
  }

  const response = data as BackendPerformanceResponse;
  const global = response.global;

  if (!global || !global.metrics) {
    return null;
  }

  const metrics = global.metrics;
  const byLeague = response.by_league || {};

  return {
    totalPredictions: global.n || 0,
    accuracy: metrics.accuracy?.pct || 0,
    brierScore: metrics.brier_score || 0,
    logLoss: metrics.log_loss || 0,
    marketBrier: metrics.market_comparison?.market_brier,
    skillVsMarket: metrics.market_comparison?.skill_vs_market,
    coveragePct: global.coverage_pct,
    byLeague: Object.entries(byLeague).map(([leagueId, league]) => ({
      leagueId,
      n: league.n || 0,
      accuracy: league.metrics?.accuracy?.pct || 0,
      brierScore: league.metrics?.brier_score || 0,
    })),
    calibration: (metrics.calibration || []).map((cal) => ({
      bin: cal.bin,
      count: cal.count,
      avgConfidence: cal.avg_confidence,
      empiricalAccuracy: cal.empirical_accuracy,
      calibrationError: cal.calibration_error,
    })),
  };
}

// ============================================================================
// Analytics Reports API
// ============================================================================

import {
  AnalyticsReportRow,
  AnalyticsReportType,
  AnalyticsReportStatus,
} from "@/lib/types";

/**
 * Helper to check if value is object
 */
function isObject(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

/**
 * Pagination metadata from reports response
 */
export interface AnalyticsReportsPagination {
  total: number;
  page: number;
  limit: number;
  pages: number;
}

/**
 * Metadata from reports response
 */
export interface AnalyticsReportsMetadata {
  generatedAt: string | null;
  cached: boolean;
  cacheAgeSeconds: number;
}

/**
 * Full parsed reports response
 */
export interface AnalyticsReportsApiResponse {
  reports: AnalyticsReportRow[];
  pagination: AnalyticsReportsPagination;
  metadata: AnalyticsReportsMetadata;
}

/**
 * Map backend type to frontend AnalyticsReportType
 */
function mapReportType(type: string): AnalyticsReportType {
  const normalized = type.toLowerCase();
  if (normalized === "model_performance") return "model_performance";
  if (normalized === "prediction_accuracy") return "prediction_accuracy";
  if (normalized === "system_metrics") return "system_metrics";
  if (normalized === "api_usage") return "api_usage";
  return "system_metrics"; // default
}

/**
 * Map backend status to frontend AnalyticsReportStatus
 */
function mapReportStatus(status: string | undefined): AnalyticsReportStatus | undefined {
  if (!status) return undefined;
  const normalized = status.toLowerCase();
  if (normalized === "ok") return "ok";
  if (normalized === "warning") return "warning";
  if (normalized === "stale") return "stale";
  return undefined;
}

/**
 * Parse a single report from backend
 */
function parseReport(raw: unknown): AnalyticsReportRow | null {
  if (!isObject(raw)) return null;

  // ID can be number or string (backend uses composite string IDs like "model_perf_14d_20")
  const id = raw.id;
  if (typeof id !== "number" && typeof id !== "string") return null;

  const type = typeof raw.type === "string" ? raw.type : "system_metrics";
  const title = typeof raw.title === "string" ? raw.title : `Report ${id}`;
  // Backend uses "subtitle" instead of "period_label"
  const periodLabel = typeof raw.period_label === "string"
    ? raw.period_label
    : typeof raw.subtitle === "string"
      ? raw.subtitle
      : "Unknown period";
  // Backend uses "updated_at" instead of "last_updated"
  const lastUpdated = typeof raw.last_updated === "string"
    ? raw.last_updated
    : typeof raw.updated_at === "string"
      ? raw.updated_at
      : new Date().toISOString();
  const status = typeof raw.status === "string" ? raw.status : undefined;

  // Parse summary object
  const summary: Record<string, string | number> = {};
  if (isObject(raw.summary)) {
    for (const [key, value] of Object.entries(raw.summary)) {
      if (typeof value === "string" || typeof value === "number") {
        summary[key] = value;
      }
    }
  }

  return {
    id,
    type: mapReportType(type),
    title,
    periodLabel,
    lastUpdated,
    status: mapReportStatus(status),
    summary,
  };
}

/**
 * Parse analytics reports response
 *
 * Expected wrapper:
 * {
 *   generated_at: string,
 *   cached: boolean,
 *   cache_age_seconds: number,
 *   data: {
 *     reports: [...],
 *     total: number,
 *     page: number,
 *     limit: number,
 *     pages: number
 *   }
 * }
 */
export function parseAnalyticsReportsResponse(response: unknown): AnalyticsReportsApiResponse | null {
  if (!isObject(response)) {
    return null;
  }

  // Extract metadata from root
  const generatedAt = typeof response.generated_at === "string" ? response.generated_at : null;
  const cached = typeof response.cached === "boolean" ? response.cached : false;
  const cacheAgeSeconds = typeof response.cache_age_seconds === "number" ? response.cache_age_seconds : 0;

  // Extract data object
  const data = response.data;
  if (!isObject(data)) {
    return null;
  }

  // Extract reports array
  const rawReports = data.reports;
  if (!Array.isArray(rawReports)) {
    return null;
  }

  // Parse reports with best-effort (skip invalid items)
  const reports: AnalyticsReportRow[] = [];
  for (const item of rawReports) {
    const report = parseReport(item);
    if (report) {
      reports.push(report);
    }
  }

  // Extract pagination
  const pagination: AnalyticsReportsPagination = {
    total: typeof data.total === "number" ? data.total : reports.length,
    page: typeof data.page === "number" ? data.page : 1,
    limit: typeof data.limit === "number" ? data.limit : 50,
    pages: typeof data.pages === "number" ? data.pages : 1,
  };

  return {
    reports,
    pagination,
    metadata: {
      generatedAt,
      cached,
      cacheAgeSeconds,
    },
  };
}

// ============================================================================
// History Summary
// ============================================================================

/**
 * Extract summary stats from history for display
 */
export function summarizeHistory(rollups: OpsHistoryRollup[]): {
  totalPitSnapshots: number;
  totalEvaluableBets: number;
  avgBaselineCoverage: number;
  totalErrors: number;
  daysWithData: number;
} {
  if (rollups.length === 0) {
    return {
      totalPitSnapshots: 0,
      totalEvaluableBets: 0,
      avgBaselineCoverage: 0,
      totalErrors: 0,
      daysWithData: 0,
    };
  }

  const totals = rollups.reduce(
    (acc, r) => ({
      pitSnapshots: acc.pitSnapshots + r.pitSnapshots,
      evaluableBets: acc.evaluableBets + r.evaluableBets,
      baselineCoverage: acc.baselineCoverage + r.baselineCoverage,
      errors: acc.errors + r.errors429 + r.timeouts,
    }),
    { pitSnapshots: 0, evaluableBets: 0, baselineCoverage: 0, errors: 0 }
  );

  const daysWithData = rollups.filter((r) => r.pitSnapshots > 0).length;

  return {
    totalPitSnapshots: totals.pitSnapshots,
    totalEvaluableBets: totals.evaluableBets,
    avgBaselineCoverage: daysWithData > 0 ? totals.baselineCoverage / rollups.length : 0,
    totalErrors: totals.errors,
    daysWithData,
  };
}
