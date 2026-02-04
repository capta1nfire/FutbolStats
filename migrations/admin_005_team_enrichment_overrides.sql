-- Migration: admin_005_team_enrichment_overrides.sql
-- Description: Create team_enrichment_overrides table for manual corrections
-- Author: Master (via Kimi recommendation)
-- Date: 2026-02-04

-- Table: team_enrichment_overrides
-- Purpose: Manual corrections that take priority over Wikidata/Wikipedia
-- Cascade: override -> wikidata -> wikipedia -> basic name

CREATE TABLE IF NOT EXISTS team_enrichment_overrides (
    -- PK/FK
    team_id INTEGER NOT NULL PRIMARY KEY REFERENCES teams(id) ON DELETE CASCADE,

    -- Override fields (null = use upstream source)
    full_name VARCHAR(500),           -- Official full name override
    short_name VARCHAR(255),          -- Short name override
    stadium_name VARCHAR(255),        -- Stadium name override
    admin_location_label VARCHAR(255),-- City/location override
    lat DOUBLE PRECISION,             -- Coords override
    lon DOUBLE PRECISION,
    website VARCHAR(500),
    twitter_handle VARCHAR(100),
    instagram_handle VARCHAR(100),

    -- Provenance
    source VARCHAR(100) NOT NULL,     -- e.g., "manual", "transfermarkt", "official_site"
    notes TEXT,                       -- Why this override exists
    created_at TIMESTAMP NOT NULL DEFAULT (NOW() AT TIME ZONE 'UTC'),
    updated_at TIMESTAMP NOT NULL DEFAULT (NOW() AT TIME ZONE 'UTC')
);

-- Index for audit queries
CREATE INDEX IF NOT EXISTS idx_teo_source ON team_enrichment_overrides(source);
CREATE INDEX IF NOT EXISTS idx_teo_updated_at ON team_enrichment_overrides(updated_at);

-- Table comments
COMMENT ON TABLE team_enrichment_overrides IS 'Manual corrections with priority over Wikidata/Wikipedia. Use sparingly for data gaps.';
COMMENT ON COLUMN team_enrichment_overrides.source IS 'Where this override came from: manual, transfermarkt, official_site, etc.';
COMMENT ON COLUMN team_enrichment_overrides.notes IS 'Reason for override, e.g., "P1448 missing in Wikidata"';
