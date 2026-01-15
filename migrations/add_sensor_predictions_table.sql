-- Migration: Create sensor_predictions table for Sensor B calibration diagnostics
-- Date: 2026-01-15
-- Purpose: LogReg L2 sliding-window model for detecting if Model A is stale/rigid

-- Create table if not exists
CREATE TABLE IF NOT EXISTS sensor_predictions (
    id SERIAL PRIMARY KEY,
    match_id INTEGER NOT NULL REFERENCES matches(id) UNIQUE,

    -- Metadata
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    evaluated_at TIMESTAMP,
    window_size INTEGER NOT NULL,

    -- Model versions
    model_a_version VARCHAR(50) NOT NULL,
    model_b_version VARCHAR(50),  -- NULL if sensor not ready (LEARNING state)

    -- Model A predictions (baseline/production)
    a_home_prob FLOAT NOT NULL,
    a_draw_prob FLOAT NOT NULL,
    a_away_prob FLOAT NOT NULL,
    a_pick VARCHAR(10) NOT NULL,  -- 'home', 'draw', 'away'

    -- Model B predictions (sensor LogReg L2) - NULL if not ready
    b_home_prob FLOAT,
    b_draw_prob FLOAT,
    b_away_prob FLOAT,
    b_pick VARCHAR(10),

    -- Outcome (filled after match completes)
    actual_outcome VARCHAR(10),  -- 'home', 'draw', 'away'
    a_correct BOOLEAN,
    b_correct BOOLEAN,

    -- Brier scores (computed after outcome)
    a_brier FLOAT,
    b_brier FLOAT,

    -- Sensor state at prediction time
    sensor_state VARCHAR(20) DEFAULT 'LEARNING'  -- LEARNING, READY, ERROR
);

-- Indexes for efficient queries
CREATE INDEX IF NOT EXISTS idx_sensor_created_at ON sensor_predictions(created_at);
CREATE INDEX IF NOT EXISTS idx_sensor_evaluated_at ON sensor_predictions(evaluated_at) WHERE evaluated_at IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_sensor_pending ON sensor_predictions(evaluated_at) WHERE evaluated_at IS NULL;

-- Comments for documentation
COMMENT ON TABLE sensor_predictions IS 'Sensor B: LogReg L2 calibration diagnostics (internal only, not for production picks)';
COMMENT ON COLUMN sensor_predictions.window_size IS 'Number of recent FT matches used to train sensor at prediction time';
COMMENT ON COLUMN sensor_predictions.model_b_version IS 'e.g. logreg_l2_w50_v1 - NULL if sensor in LEARNING state';
COMMENT ON COLUMN sensor_predictions.sensor_state IS 'LEARNING: insufficient samples, READY: sensor trained, ERROR: training failed';
COMMENT ON COLUMN sensor_predictions.a_brier IS 'Brier score for Model A prediction';
COMMENT ON COLUMN sensor_predictions.b_brier IS 'Brier score for Model B prediction (NULL if sensor not ready)';
