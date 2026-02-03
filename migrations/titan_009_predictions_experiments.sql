-- Migration: Create predictions_experiments table
-- Purpose: Store experimental predictions by tier for TITAN A/B comparison
-- PIT-safe by design: created_at <= snapshot_at enforced by constraint
-- ATI-approved: 2026-01-29

CREATE TABLE IF NOT EXISTS predictions_experiments (
    id SERIAL PRIMARY KEY,
    snapshot_id BIGINT NOT NULL REFERENCES odds_snapshots(id),
    match_id BIGINT NOT NULL REFERENCES matches(id),
    snapshot_at TIMESTAMPTZ NOT NULL,
    model_version VARCHAR(50) NOT NULL,
    home_prob NUMERIC(5,4) NOT NULL,
    draw_prob NUMERIC(5,4) NOT NULL,
    away_prob NUMERIC(5,4) NOT NULL,
    feature_set JSONB,
    created_at TIMESTAMPTZ NOT NULL,

    UNIQUE(snapshot_id, model_version)
);

CREATE INDEX IF NOT EXISTS idx_pred_exp_model ON predictions_experiments(model_version);
CREATE INDEX IF NOT EXISTS idx_pred_exp_snapshot ON predictions_experiments(snapshot_at);
CREATE INDEX IF NOT EXISTS idx_pred_exp_match ON predictions_experiments(match_id);

ALTER TABLE predictions_experiments
ADD CONSTRAINT chk_pit_safe CHECK (created_at <= snapshot_at);

COMMENT ON TABLE predictions_experiments IS 'Experimental predictions by tier for TITAN comparison. PIT-safe by design.';
COMMENT ON COLUMN predictions_experiments.snapshot_id IS 'FK to odds_snapshots. Primary key with model_version.';
COMMENT ON COLUMN predictions_experiments.created_at IS 'Must be <= snapshot_at. Set to snapshot_at - 1s during generation.';
