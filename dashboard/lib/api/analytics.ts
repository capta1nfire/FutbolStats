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
