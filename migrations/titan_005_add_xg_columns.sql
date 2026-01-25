-- Migration: Add Tier 1b (xG) columns to feature_matrix
-- Date: 2026-01-25
-- Purpose: FASE 2 - Understat xG integration
-- Plan: zazzy-jingling-pudding.md v2.0

-- =============================================================================
-- Tier 1b: xG Features (Understat)
-- =============================================================================

-- Rolling xG aggregates (last N matches per team, default N=5)
-- Note: "*_last5" naming clarifies these are rolling windows, not full season
ALTER TABLE titan.feature_matrix
ADD COLUMN IF NOT EXISTS xg_home_last5 DECIMAL(5,2),
ADD COLUMN IF NOT EXISTS xg_away_last5 DECIMAL(5,2),
ADD COLUMN IF NOT EXISTS xga_home_last5 DECIMAL(5,2),  -- xG Against
ADD COLUMN IF NOT EXISTS xga_away_last5 DECIMAL(5,2),
ADD COLUMN IF NOT EXISTS npxg_home_last5 DECIMAL(5,2), -- Non-penalty xG
ADD COLUMN IF NOT EXISTS npxg_away_last5 DECIMAL(5,2),
ADD COLUMN IF NOT EXISTS xg_captured_at TIMESTAMPTZ,
ADD COLUMN IF NOT EXISTS tier1b_complete BOOLEAN NOT NULL DEFAULT FALSE;

-- Index for tier1b queries
CREATE INDEX IF NOT EXISTS idx_fm_tier1b
ON titan.feature_matrix(tier1b_complete)
WHERE tier1b_complete = TRUE;

-- Comments
COMMENT ON COLUMN titan.feature_matrix.xg_home_last5 IS 'Home team avg xG over last 5 matches (Understat)';
COMMENT ON COLUMN titan.feature_matrix.xg_away_last5 IS 'Away team avg xG over last 5 matches (Understat)';
COMMENT ON COLUMN titan.feature_matrix.xga_home_last5 IS 'Home team avg xG Against over last 5 matches';
COMMENT ON COLUMN titan.feature_matrix.xga_away_last5 IS 'Away team avg xG Against over last 5 matches';
COMMENT ON COLUMN titan.feature_matrix.npxg_home_last5 IS 'Home team avg non-penalty xG over last 5 matches';
COMMENT ON COLUMN titan.feature_matrix.npxg_away_last5 IS 'Away team avg non-penalty xG over last 5 matches';
COMMENT ON COLUMN titan.feature_matrix.xg_captured_at IS 'Timestamp when xG data was captured/computed (PIT)';
COMMENT ON COLUMN titan.feature_matrix.tier1b_complete IS 'TRUE if xG features are populated';
