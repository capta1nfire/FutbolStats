/**
 * ML Health Dashboard Types
 *
 * These types match the backend contract from /dashboard/ml_health.json
 * IMPORTANT: Do not invent thresholds or logic here - backend is source of truth
 */

/**
 * Status values used throughout ML Health
 */
export type MLHealthStatus = "ok" | "warn" | "error" | "partial" | "not_ready";

/**
 * Fuel Gauge - Top-level ML pipeline health indicator
 */
export interface FuelGauge {
  status: MLHealthStatus;
  reasons: string[];
  as_of_utc: string;
}

/**
 * SOTA Stats Coverage - Season stats
 */
export interface SotaSeasonStats {
  total_matches_ft: number;
  with_stats_pct: number;
  marked_no_stats_pct: number;
  shots_present_pct: number;
}

/**
 * SOTA Stats Coverage - League stats
 */
export interface SotaLeagueStats {
  league_id: number;
  name: string;
  with_stats_pct: number;
}

/**
 * SOTA Stats Coverage section
 */
export interface SotaStatsCoverage {
  by_season: Record<string, SotaSeasonStats>;
  by_league: SotaLeagueStats[];
  status: MLHealthStatus;
  /** True if this section failed to load (fail-soft) */
  _degraded?: boolean;
  /** Error message if degraded */
  _error?: string;
}

/**
 * TITAN Coverage - Tier coverage data
 */
export interface TitanTierCoverage {
  complete: number;
  total: number;
  pct: number;
}

/**
 * TITAN Coverage - Season data with all tiers
 */
export interface TitanSeasonCoverage {
  tier1?: TitanTierCoverage;
  tier1b?: TitanTierCoverage;
  tier1c?: TitanTierCoverage;
  tier1d?: TitanTierCoverage;
}

/**
 * TITAN Coverage - League stats
 */
export interface TitanLeagueStats {
  league_id: number;
  name: string;
  tier1_pct: number;
  tier1b_pct: number;
}

/**
 * TITAN Coverage section
 */
export interface TitanCoverage {
  by_season: Record<string, TitanSeasonCoverage>;
  by_league: TitanLeagueStats[];
  status: MLHealthStatus;
  _degraded?: boolean;
  _error?: string;
}

/**
 * PIT Compliance section
 */
export interface PitCompliance {
  total_rows: number;
  violations: number;
  violation_pct: number;
  status: MLHealthStatus;
  _degraded?: boolean;
  _error?: string;
}

/**
 * Percentile stats (p50, p95, max)
 */
export interface PercentileStats {
  p50: number;
  p95: number;
  max: number;
}

/**
 * Freshness section
 */
export interface Freshness {
  age_hours_now?: {
    odds?: PercentileStats;
    xg?: PercentileStats;
  };
  lead_time_hours?: {
    odds?: PercentileStats;
    xg?: PercentileStats;
  };
  status: MLHealthStatus;
  _degraded?: boolean;
  _error?: string;
}

/**
 * Entropy stats for prediction confidence
 */
export interface EntropyStats {
  avg: number;
  p25: number;
  p50: number;
  p75: number;
  p95: number;
}

/**
 * Tier distribution for prediction confidence
 */
export interface TierDistribution {
  gold: number;
  silver: number;
  copper: number;
}

/**
 * Prediction Confidence section
 */
export interface PredictionConfidence {
  entropy?: EntropyStats;
  tier_distribution?: TierDistribution;
  sample_n: number;
  window_days: number;
  _degraded?: boolean;
  _error?: string;
}

/**
 * Top Regressions section (placeholder until ready)
 */
export interface TopRegressions {
  status: "not_ready" | "ok" | "warn";
  note?: string;
  _degraded?: boolean;
  _error?: string;
  // Future: regressions list when status != "not_ready"
}

/**
 * Full ML Health data payload
 */
export interface MLHealthData {
  fuel_gauge: FuelGauge;
  sota_stats_coverage: SotaStatsCoverage;
  titan_coverage: TitanCoverage;
  pit_compliance: PitCompliance;
  freshness: Freshness;
  prediction_confidence: PredictionConfidence;
  top_regressions: TopRegressions;
}

/**
 * Full ML Health API response
 */
export interface MLHealthResponse {
  generated_at: string;
  cached: boolean;
  cache_age_seconds: number | null;
  health: MLHealthStatus;
  data: MLHealthData;
}
