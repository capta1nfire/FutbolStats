/**
 * Model Benchmark types
 *
 * Types for the model benchmark comparison feature.
 * Compares Market (bet365 odds), Model A (v1.0.0), and Shadow (v1.1.0-two_stage).
 */

/**
 * Daily aggregated stats for each model
 */
export interface DailyModelStats {
  /** ISO date format: "2026-01-17" */
  date: string;
  /** Number of matches evaluated that day */
  matches: number;
  /** Correct predictions by Market (from odds) */
  market_correct: number;
  /** Correct predictions by Model A (v1.0.0) */
  model_a_correct: number;
  /** Correct predictions by Shadow (v1.1.0-two_stage) */
  shadow_correct: number;
  /** Correct predictions by Sensor B (from sensor_predictions) */
  sensor_b_correct: number;
  /** Correct predictions by Family S (Tier 3 MTV model) */
  family_s_correct: number;
  /** Predominant Model A baseline version that day (e.g. "v1.2.0-league-only") */
  model_a_version?: string | null;
}

/**
 * Summary stats for a single model
 */
export interface ModelSummary {
  /** Display name: "Market", "Model A", "Shadow" */
  name: string;
  /** Accuracy percentage (0-100) */
  accuracy: number;
  /** Total correct predictions */
  correct: number;
  /** Total matches evaluated */
  total: number;
  /** Days won (fractional: ties split the day - 2 tied = 0.5 each, 4 tied = 0.25 each) */
  days_won: number;
}

/**
 * Response from /dashboard/model-benchmark endpoint
 */
export interface ModelBenchmarkResponse {
  /** ISO timestamp when response was generated */
  generated_at: string;
  /** Start date for benchmark (dynamic based on selected models) */
  start_date: string;
  /** Models included in this comparison */
  selected_models: string[];
  /** Total matches evaluated across all days */
  total_matches: number;
  /** Daily breakdown of results */
  daily_data: DailyModelStats[];
  /** Summary stats per model (only selected models) */
  models: ModelSummary[];
}
