/**
 * Types for the Benchmark Matrix endpoint.
 *
 * Matrix: leagues (rows) × sources (columns) with Brier Skill % vs Pinnacle.
 */

export interface BenchmarkCell {
  skill_pct: number | null;
  delta_brier: number | null;
  delta_logloss: number | null;
  brier_abs: number | null;
  pinnacle_brier: number | null;
  logloss_abs: number | null;
  pinnacle_logloss: number | null;
  n: number;
  confidence_tier: "insufficient" | "low" | "normal" | "confident";
  ci_lo: number | null;
  ci_hi: number | null;
}

export interface BenchmarkLeague {
  league_id: number;
  name: string;
  country: string;
  total_resolved: number;
}

export interface BenchmarkSource {
  key: string;
  label: string;
  kind: "bookmaker" | "model";
  total_matches: number;
}

export interface BenchmarkMatrixResponse {
  generated_at: string;
  period: string;
  anchor: string;
  leagues: BenchmarkLeague[];
  sources: BenchmarkSource[];
  cells: Record<string, BenchmarkCell>;
  global_row: Record<string, BenchmarkCell>;
}

export type BenchmarkMetric = "skill_pct" | "delta_brier" | "delta_logloss";

export type BenchmarkPeriod = "30d" | "60d" | "90d" | "all";

export const BENCHMARK_PERIODS: { value: BenchmarkPeriod; label: string }[] = [
  { value: "30d", label: "30 days" },
  { value: "60d", label: "60 days" },
  { value: "90d", label: "90 days" },
  { value: "all", label: "All" },
];

export const BENCHMARK_METRICS: { value: BenchmarkMetric; label: string }[] = [
  { value: "skill_pct", label: "Skill %" },
  { value: "delta_brier", label: "ΔBrier" },
  { value: "delta_logloss", label: "ΔLogLoss" },
];
