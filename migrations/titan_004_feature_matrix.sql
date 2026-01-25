-- Migration: Create titan.feature_matrix table
-- Date: 2026-01-25
-- Purpose: FASE 1 - Feature matrix with PIT constraint and tier system
-- Depends: titan_001_create_schema.sql
-- Plan: zazzy-jingling-pudding.md v1.1

-- =============================================================================
-- feature_matrix: Pre-computed features for ML with PIT compliance
-- =============================================================================
-- Key design decisions:
-- 1. PIT constraint: pit_max_captured_at < kickoff_utc (CRITICAL)
-- 2. Tier system: Tier 1 (odds) required, Tier 2/3 optional with NULLs
-- 3. All *_captured_at timestamps for audit trail
-- 4. Trigger for automatic updated_at

CREATE TABLE IF NOT EXISTS titan.feature_matrix (
    -- Primary key: external match ID from API-Football
    match_id            BIGINT PRIMARY KEY,

    -- Match identification
    kickoff_utc         TIMESTAMPTZ NOT NULL,
    competition_id      INT NOT NULL,
    season              INT NOT NULL,
    home_team_id        BIGINT NOT NULL,
    away_team_id        BIGINT NOT NULL,

    -- ==========================================================================
    -- TIER 1: Odds (REQUIRED for insertion - Fase 1 gate)
    -- Source: API-Football odds endpoint
    -- ==========================================================================
    odds_home_close     DECIMAL(6,3),       -- e.g., 2.150
    odds_draw_close     DECIMAL(6,3),
    odds_away_close     DECIMAL(6,3),
    implied_prob_home   DECIMAL(5,4),       -- e.g., 0.4651 (normalized)
    implied_prob_draw   DECIMAL(5,4),
    implied_prob_away   DECIMAL(5,4),
    odds_captured_at    TIMESTAMPTZ,        -- When odds were captured

    -- ==========================================================================
    -- TIER 2: Form (optional - computed from public.matches)
    -- ==========================================================================
    form_home_last5     VARCHAR(5),         -- e.g., 'WWDLW'
    form_away_last5     VARCHAR(5),
    goals_home_last5    SMALLINT,           -- Goals scored in last 5
    goals_away_last5    SMALLINT,
    goals_against_home_last5 SMALLINT,      -- Goals conceded in last 5
    goals_against_away_last5 SMALLINT,
    points_home_last5   SMALLINT,           -- Points in last 5 (W=3, D=1, L=0)
    points_away_last5   SMALLINT,
    form_captured_at    TIMESTAMPTZ,        -- When form was computed

    -- ==========================================================================
    -- TIER 3: H2H (optional - computed from public.matches)
    -- ==========================================================================
    h2h_total_matches   SMALLINT,           -- Total H2H matches
    h2h_home_wins       SMALLINT,           -- Wins by home team in H2H
    h2h_draws           SMALLINT,
    h2h_away_wins       SMALLINT,
    h2h_home_goals      SMALLINT,           -- Total goals by home team in H2H
    h2h_away_goals      SMALLINT,
    h2h_captured_at     TIMESTAMPTZ,        -- When H2H was computed

    -- ==========================================================================
    -- PIT COMPLIANCE (CRITICAL)
    -- ==========================================================================
    -- Maximum captured_at across all tiers. MUST be < kickoff_utc.
    pit_max_captured_at TIMESTAMPTZ NOT NULL,

    -- Outcome (filled after match completes)
    outcome             VARCHAR(10),        -- 'home', 'draw', 'away', or NULL if not finished

    -- Tier completion flags
    tier1_complete      BOOLEAN NOT NULL DEFAULT FALSE,
    tier2_complete      BOOLEAN NOT NULL DEFAULT FALSE,
    tier3_complete      BOOLEAN NOT NULL DEFAULT FALSE,

    -- Metadata
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- ==========================================================================
    -- PIT CONSTRAINT: Data must be captured BEFORE kickoff
    -- ==========================================================================
    CONSTRAINT pit_valid CHECK (pit_max_captured_at < kickoff_utc)
);

-- Trigger for automatic updated_at
CREATE TRIGGER update_feature_matrix_updated_at
    BEFORE UPDATE ON titan.feature_matrix
    FOR EACH ROW
    EXECUTE FUNCTION titan.update_updated_at_column();

-- ==========================================================================
-- INDEXES
-- ==========================================================================

-- Primary query patterns
CREATE INDEX IF NOT EXISTS idx_fm_kickoff
    ON titan.feature_matrix(kickoff_utc);

CREATE INDEX IF NOT EXISTS idx_fm_competition
    ON titan.feature_matrix(competition_id, season);

-- Tier completion queries
CREATE INDEX IF NOT EXISTS idx_fm_tier1
    ON titan.feature_matrix(tier1_complete)
    WHERE tier1_complete = TRUE;

CREATE INDEX IF NOT EXISTS idx_fm_incomplete
    ON titan.feature_matrix(created_at)
    WHERE tier1_complete = FALSE;

-- PIT audit queries
CREATE INDEX IF NOT EXISTS idx_fm_pit
    ON titan.feature_matrix(pit_max_captured_at);

-- Outcome backfill queries
CREATE INDEX IF NOT EXISTS idx_fm_pending_outcome
    ON titan.feature_matrix(kickoff_utc)
    WHERE outcome IS NULL;

-- ==========================================================================
-- DOCUMENTATION
-- ==========================================================================

COMMENT ON TABLE titan.feature_matrix IS 'Pre-computed features for ML with strict PIT (Point-in-Time) compliance';

-- PIT columns
COMMENT ON COLUMN titan.feature_matrix.pit_max_captured_at IS 'CRITICAL: Max captured_at across all tiers. Must be < kickoff_utc.';
COMMENT ON COLUMN titan.feature_matrix.odds_captured_at IS 'When odds were captured (must be before kickoff)';
COMMENT ON COLUMN titan.feature_matrix.form_captured_at IS 'When form was computed from historical matches';
COMMENT ON COLUMN titan.feature_matrix.h2h_captured_at IS 'When H2H was computed from historical matches';

-- Tier columns
COMMENT ON COLUMN titan.feature_matrix.tier1_complete IS 'Tier 1 (odds) is complete - REQUIRED for valid row';
COMMENT ON COLUMN titan.feature_matrix.tier2_complete IS 'Tier 2 (form) is complete - optional';
COMMENT ON COLUMN titan.feature_matrix.tier3_complete IS 'Tier 3 (H2H) is complete - optional';

-- Odds (Tier 1)
COMMENT ON COLUMN titan.feature_matrix.implied_prob_home IS 'Normalized probability: 1/odds_home / (1/odds_home + 1/odds_draw + 1/odds_away)';

-- Form (Tier 2)
COMMENT ON COLUMN titan.feature_matrix.form_home_last5 IS 'Last 5 results: W=Win, D=Draw, L=Loss (most recent first)';
COMMENT ON COLUMN titan.feature_matrix.points_home_last5 IS 'Points in last 5 games: W=3, D=1, L=0';

-- Outcome
COMMENT ON COLUMN titan.feature_matrix.outcome IS 'Match result: home, draw, away (filled after FT/AET/PEN)';
