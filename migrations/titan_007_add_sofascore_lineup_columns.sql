-- Migration: Add Tier 1c (SofaScore Lineup) columns to feature_matrix
-- Date: 2026-01-26
-- Purpose: FASE 3A - SofaScore lineup integration
-- Architecture: Read from public.match_sofascore_lineup (SOTA), materialize to TITAN
-- Approved by: ABE

-- =============================================================================
-- Tier 1c: SofaScore Lineup Features (via SOTA)
-- =============================================================================

ALTER TABLE titan.feature_matrix
ADD COLUMN IF NOT EXISTS sofascore_lineup_available BOOLEAN NOT NULL DEFAULT FALSE,
ADD COLUMN IF NOT EXISTS sofascore_home_formation VARCHAR(20),
ADD COLUMN IF NOT EXISTS sofascore_away_formation VARCHAR(20),
ADD COLUMN IF NOT EXISTS sofascore_lineup_captured_at TIMESTAMPTZ,
ADD COLUMN IF NOT EXISTS lineup_home_starters_count SMALLINT,
ADD COLUMN IF NOT EXISTS lineup_away_starters_count SMALLINT,
ADD COLUMN IF NOT EXISTS sofascore_lineup_integrity_score DECIMAL(4,3),
ADD COLUMN IF NOT EXISTS tier1c_complete BOOLEAN NOT NULL DEFAULT FALSE;

-- =============================================================================
-- Indexes
-- =============================================================================

-- Index for tier1c queries
CREATE INDEX IF NOT EXISTS idx_fm_tier1c_complete
ON titan.feature_matrix(tier1c_complete)
WHERE tier1c_complete = TRUE;

-- Index for lineup freshness queries
CREATE INDEX IF NOT EXISTS idx_fm_sofascore_lineup_captured
ON titan.feature_matrix(sofascore_lineup_captured_at)
WHERE sofascore_lineup_captured_at IS NOT NULL;

-- =============================================================================
-- Comments
-- =============================================================================

COMMENT ON COLUMN titan.feature_matrix.sofascore_lineup_available IS 'TRUE if SofaScore lineup was found for this match';
COMMENT ON COLUMN titan.feature_matrix.sofascore_home_formation IS 'Home team formation (e.g., 4-3-3, 4-2-3-1)';
COMMENT ON COLUMN titan.feature_matrix.sofascore_away_formation IS 'Away team formation';
COMMENT ON COLUMN titan.feature_matrix.sofascore_lineup_captured_at IS 'Timestamp when lineup was captured (PIT)';
COMMENT ON COLUMN titan.feature_matrix.lineup_home_starters_count IS 'Number of home starters (should be 11 for complete lineup)';
COMMENT ON COLUMN titan.feature_matrix.lineup_away_starters_count IS 'Number of away starters (should be 11 for complete lineup)';
COMMENT ON COLUMN titan.feature_matrix.sofascore_lineup_integrity_score IS 'Lineup completeness score (0.000-1.000): avg of formation_present + starters==11';
COMMENT ON COLUMN titan.feature_matrix.tier1c_complete IS 'TRUE if SofaScore lineup features are populated';
