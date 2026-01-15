-- Migration: Create shadow_predictions table for A/B model comparison
-- Date: 2026-01-15
-- Purpose: FASE 2 - Two-stage model shadow evaluation

-- Create table if not exists
CREATE TABLE IF NOT EXISTS shadow_predictions (
    id SERIAL PRIMARY KEY,
    match_id INTEGER NOT NULL REFERENCES matches(id),

    -- Baseline model predictions (currently served)
    baseline_version VARCHAR(50) NOT NULL,
    baseline_home_prob FLOAT NOT NULL,
    baseline_draw_prob FLOAT NOT NULL,
    baseline_away_prob FLOAT NOT NULL,
    baseline_predicted VARCHAR(10) NOT NULL,

    -- Shadow model predictions (experimental, not served)
    shadow_version VARCHAR(50) NOT NULL,
    shadow_architecture VARCHAR(50) NOT NULL,
    shadow_home_prob FLOAT NOT NULL,
    shadow_draw_prob FLOAT NOT NULL,
    shadow_away_prob FLOAT NOT NULL,
    shadow_predicted VARCHAR(10) NOT NULL,

    -- Outcome (filled after match completes)
    actual_result VARCHAR(10),
    baseline_correct BOOLEAN,
    shadow_correct BOOLEAN,

    -- Metrics (computed after outcome)
    baseline_brier FLOAT,
    shadow_brier FLOAT,

    -- Error tracking (if shadow fails)
    error_code VARCHAR(50),
    error_detail VARCHAR(500),

    -- Metadata
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    evaluated_at TIMESTAMP
);

-- Indexes for efficient queries
CREATE INDEX IF NOT EXISTS idx_shadow_match_id ON shadow_predictions(match_id);
CREATE INDEX IF NOT EXISTS idx_shadow_created_at ON shadow_predictions(created_at);
CREATE INDEX IF NOT EXISTS idx_shadow_evaluated ON shadow_predictions(evaluated_at) WHERE evaluated_at IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_shadow_pending ON shadow_predictions(actual_result) WHERE actual_result IS NULL;

-- Unique constraint: one shadow prediction per match per timestamp
-- (allows re-runs but prevents duplicates in same batch)
CREATE UNIQUE INDEX IF NOT EXISTS uq_shadow_match_created ON shadow_predictions(match_id, created_at);

-- Comments for documentation
COMMENT ON TABLE shadow_predictions IS 'FASE 2: A/B comparison between baseline and two-stage model';
COMMENT ON COLUMN shadow_predictions.baseline_predicted IS 'home, draw, or away';
COMMENT ON COLUMN shadow_predictions.shadow_predicted IS 'home, draw, or away';
COMMENT ON COLUMN shadow_predictions.actual_result IS 'Filled when match finishes (FT/AET/PEN)';
COMMENT ON COLUMN shadow_predictions.baseline_brier IS 'Brier score contribution for this prediction';
COMMENT ON COLUMN shadow_predictions.shadow_brier IS 'Brier score contribution for this prediction';
COMMENT ON COLUMN shadow_predictions.error_code IS 'If shadow prediction failed: shadow_load_error, shadow_predict_error, etc.';
