-- Migration: Add Tier 1d (XI Depth) columns to feature_matrix
-- Date: 2026-01-26
-- Purpose: FASE 3B-1 - XI lineup-derived features
-- Architecture: Read from public.match_sofascore_player (SOTA), materialize to TITAN
-- Approved by: ABE

-- =============================================================================
-- Tier 1d: XI Depth Features (via SOTA)
-- =============================================================================

ALTER TABLE titan.feature_matrix
ADD COLUMN IF NOT EXISTS xi_home_def_count SMALLINT,
ADD COLUMN IF NOT EXISTS xi_home_mid_count SMALLINT,
ADD COLUMN IF NOT EXISTS xi_home_fwd_count SMALLINT,
ADD COLUMN IF NOT EXISTS xi_away_def_count SMALLINT,
ADD COLUMN IF NOT EXISTS xi_away_mid_count SMALLINT,
ADD COLUMN IF NOT EXISTS xi_away_fwd_count SMALLINT,
ADD COLUMN IF NOT EXISTS xi_formation_mismatch_flag BOOLEAN DEFAULT FALSE,
ADD COLUMN IF NOT EXISTS xi_depth_captured_at TIMESTAMPTZ,
ADD COLUMN IF NOT EXISTS tier1d_complete BOOLEAN NOT NULL DEFAULT FALSE;

-- =============================================================================
-- Indexes
-- =============================================================================

CREATE INDEX IF NOT EXISTS idx_fm_tier1d_complete
ON titan.feature_matrix(tier1d_complete)
WHERE tier1d_complete = TRUE;

-- =============================================================================
-- Comments
-- =============================================================================

COMMENT ON COLUMN titan.feature_matrix.xi_home_def_count IS 'Home team starting defenders count (from SofaScore XI)';
COMMENT ON COLUMN titan.feature_matrix.xi_home_mid_count IS 'Home team starting midfielders count';
COMMENT ON COLUMN titan.feature_matrix.xi_home_fwd_count IS 'Home team starting forwards count';
COMMENT ON COLUMN titan.feature_matrix.xi_away_def_count IS 'Away team starting defenders count';
COMMENT ON COLUMN titan.feature_matrix.xi_away_mid_count IS 'Away team starting midfielders count';
COMMENT ON COLUMN titan.feature_matrix.xi_away_fwd_count IS 'Away team starting forwards count';
COMMENT ON COLUMN titan.feature_matrix.xi_formation_mismatch_flag IS 'TRUE if XI position counts differ from declared formation';
COMMENT ON COLUMN titan.feature_matrix.xi_depth_captured_at IS 'Timestamp when XI depth data was captured (PIT-critical)';
COMMENT ON COLUMN titan.feature_matrix.tier1d_complete IS 'TRUE if XI depth features are populated (xi_depth_captured_at IS NOT NULL)';
