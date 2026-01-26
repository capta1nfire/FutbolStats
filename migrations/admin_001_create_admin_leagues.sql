-- Migration: Create admin_leagues table
-- Date: 2026-01-26
-- Purpose: P2A - Persistent league configuration (replaces COMPETITIONS dict)
-- Plan: zazzy-jingling-pudding.md v2.0

-- =============================================================================
-- admin_leagues: Source of truth for league configuration
-- =============================================================================
-- Key design decisions:
-- 1. is_active = TRUE means league is SERVED to end-users (product decision, platform-agnostic)
-- 2. source indicates origin: 'seed' (from COMPETITIONS), 'override' (manual), 'observed' (auto-discovered)
-- 3. Fields match COMPETITIONS dataclass: name, priority, match_type, match_weight
-- 4. kind categorizes: 'league', 'cup', 'international', 'friendly'
-- 5. group_id for paired leagues (Apertura/Clausura) - FK to admin_league_groups
-- 6. tags.channels (optional): ["ios","android","web"] for per-platform control; if absent, all platforms when is_active=true

-- =============================================================================
-- admin_league_groups: Groups for paired leagues (Apertura/Clausura)
-- =============================================================================
CREATE TABLE IF NOT EXISTS admin_league_groups (
    group_id        SERIAL PRIMARY KEY,
    group_key       TEXT UNIQUE NOT NULL,           -- e.g., 'URY_PRIMERA', 'PAR_PRIMERA'
    name            TEXT NOT NULL,                  -- e.g., 'Uruguay Primera División'
    country         TEXT,                           -- e.g., 'Uruguay'
    tags            JSONB NOT NULL DEFAULT '{}',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- =============================================================================
-- admin_leagues: Main configuration table
-- =============================================================================
CREATE TABLE IF NOT EXISTS admin_leagues (
    -- Primary key: API-Football league_id
    league_id       INT PRIMARY KEY,

    -- Basic info
    sport           TEXT NOT NULL DEFAULT 'football',
    name            TEXT NOT NULL,
    country         TEXT,                           -- NULL for international competitions

    -- Classification
    kind            TEXT NOT NULL DEFAULT 'league'
        CHECK (kind IN ('league', 'cup', 'international', 'friendly')),

    -- Activation (CRITICAL: is_active = TRUE means SERVED to end-users across all platforms)
    -- For per-platform control, use tags.channels (e.g., {"channels": ["ios","android","web"]})
    is_active       BOOLEAN NOT NULL DEFAULT TRUE,

    -- Configuration (from COMPETITIONS dataclass)
    priority        TEXT CHECK (priority IN ('high', 'medium', 'low')),
    match_type      TEXT CHECK (match_type IN ('official', 'friendly')),
    match_weight    DOUBLE PRECISION CHECK (match_weight >= 0 AND match_weight <= 1.0),

    -- UI/Display
    display_order   INT,

    -- Paired leagues (Apertura/Clausura)
    group_id        INT REFERENCES admin_league_groups(group_id),

    -- Extensible metadata
    tags            JSONB NOT NULL DEFAULT '{}',
    rules_json      JSONB NOT NULL DEFAULT '{}',

    -- Source tracking
    source          TEXT NOT NULL DEFAULT 'seed'
        CHECK (source IN ('seed', 'override', 'observed')),

    -- Timestamps
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- =============================================================================
-- TRIGGER: Automatic updated_at
-- =============================================================================
CREATE OR REPLACE FUNCTION update_admin_leagues_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_admin_leagues_updated_at
    BEFORE UPDATE ON admin_leagues
    FOR EACH ROW
    EXECUTE FUNCTION update_admin_leagues_updated_at();

CREATE TRIGGER update_admin_league_groups_updated_at
    BEFORE UPDATE ON admin_league_groups
    FOR EACH ROW
    EXECUTE FUNCTION update_admin_leagues_updated_at();

-- =============================================================================
-- INDEXES
-- =============================================================================

-- Active leagues for end-users (all platforms)
CREATE INDEX IF NOT EXISTS idx_admin_leagues_active
    ON admin_leagues(is_active)
    WHERE is_active = TRUE;

-- By country for grouping
CREATE INDEX IF NOT EXISTS idx_admin_leagues_country
    ON admin_leagues(country)
    WHERE country IS NOT NULL;

-- By source for filtering
CREATE INDEX IF NOT EXISTS idx_admin_leagues_source
    ON admin_leagues(source);

-- By group for paired leagues
CREATE INDEX IF NOT EXISTS idx_admin_leagues_group
    ON admin_leagues(group_id)
    WHERE group_id IS NOT NULL;

-- =============================================================================
-- DOCUMENTATION
-- =============================================================================

COMMENT ON TABLE admin_leagues IS 'Source of truth for league configuration. is_active=TRUE means served to end-users (all platforms). Use tags.channels for per-platform control.';
COMMENT ON TABLE admin_league_groups IS 'Groups for paired leagues (Apertura/Clausura, etc.)';

-- admin_leagues columns
COMMENT ON COLUMN admin_leagues.league_id IS 'API-Football league ID (primary key)';
COMMENT ON COLUMN admin_leagues.is_active IS 'CRITICAL: TRUE = league served to end-users (product decision, not temporal). For per-platform: use tags.channels';
COMMENT ON COLUMN admin_leagues.source IS 'Origin: seed (from COMPETITIONS), override (manual edit), observed (auto-discovered from matches)';
COMMENT ON COLUMN admin_leagues.kind IS 'Classification: league, cup, international, friendly';
COMMENT ON COLUMN admin_leagues.priority IS 'Sync priority: high, medium, low';
COMMENT ON COLUMN admin_leagues.match_type IS 'Match classification: official, friendly';
COMMENT ON COLUMN admin_leagues.match_weight IS 'ML weight factor (0.0 to 1.0)';
COMMENT ON COLUMN admin_leagues.group_id IS 'FK to admin_league_groups for paired leagues (Apertura/Clausura)';
COMMENT ON COLUMN admin_leagues.tags IS 'Extensible metadata (JSONB). Supports: channels (["ios","android","web"]) for per-platform control';
COMMENT ON COLUMN admin_leagues.rules_json IS 'League-specific rules (team_count, season_model, promotion_relegation, etc.)';

-- admin_league_groups columns
COMMENT ON COLUMN admin_league_groups.group_key IS 'Unique key: e.g., URY_PRIMERA, PAR_PRIMERA';
COMMENT ON COLUMN admin_league_groups.name IS 'Display name: e.g., Uruguay Primera División';
