-- Migration: admin_004_team_wikidata_enrichment.sql
-- Description: Create team_wikidata_enrichment table for Wikidata SPARQL enrichment
-- Author: Master (via ABE-approved plan)
-- Date: 2026-02-04

-- Table: team_wikidata_enrichment
-- Purpose: Store enriched team data from Wikidata SPARQL API
-- Note: Separate from teams table to avoid column explosion (ABE directive)

CREATE TABLE IF NOT EXISTS team_wikidata_enrichment (
    -- PK/FK
    team_id INTEGER NOT NULL PRIMARY KEY REFERENCES teams(id) ON DELETE CASCADE,

    -- Provenance (non-negotiable per ABE)
    wikidata_id VARCHAR(20) NOT NULL,  -- Q-number, e.g., Q6150984
    fetched_at TIMESTAMP NOT NULL DEFAULT (NOW() AT TIME ZONE 'UTC'),  -- UTC naive (repo consistency)
    raw_jsonb JSONB,  -- Reproducible snapshot of Wikidata response

    -- Derived: Geo/Stadium (high ROI for weather/venue_geo)
    -- ABE: Coords come from STADIUM (P115->P625), not the club
    stadium_name VARCHAR(255),
    stadium_wikidata_id VARCHAR(20),  -- Q-number of stadium
    stadium_capacity INTEGER,
    stadium_altitude_m INTEGER,  -- Home advantage factor in LATAM
    lat DOUBLE PRECISION,  -- From stadium (P115->P625)
    lon DOUBLE PRECISION,  -- From stadium (P115->P625)

    -- ABE: "city" from P131 is noisy. Store as informational label,
    -- NOT as absolute truth. Timezone resolves downstream via _resolve_timezone_for_result()
    admin_location_label VARCHAR(255),  -- P131 label (informational, not normalized)

    -- Derived: Identity (product)
    full_name VARCHAR(500),
    short_name VARCHAR(255),

    -- Derived: Social/Web (JSONB, flexible structure)
    social_handles JSONB,  -- {"twitter": "AmericadeCali", "instagram": "americadecali", ...}
    website VARCHAR(500),

    -- Derived: Colors (JSONB of QIDs/labels, not hex)
    colors JSONB,  -- [{"qid": "Q3142", "label": "red"}, {"qid": "Q1088", "label": "blue"}]

    -- Metadata
    enrichment_version INTEGER NOT NULL DEFAULT 1,  -- For future schema migrations

    -- Constraints
    CONSTRAINT chk_wikidata_id_format CHECK (wikidata_id ~ '^Q[0-9]+$')
);

-- Indexes for frequent queries
CREATE INDEX IF NOT EXISTS idx_twe_wikidata_id ON team_wikidata_enrichment(wikidata_id);
CREATE INDEX IF NOT EXISTS idx_twe_has_geo ON team_wikidata_enrichment(lat, lon) WHERE lat IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_twe_fetched_at ON team_wikidata_enrichment(fetched_at);

-- Table comment
COMMENT ON TABLE team_wikidata_enrichment IS 'Enriched team data from Wikidata SPARQL. MVP: geo+stadium (stadium coords), identity, social.';
COMMENT ON COLUMN team_wikidata_enrichment.admin_location_label IS 'P131 label - informational only, not normalized city name';
COMMENT ON COLUMN team_wikidata_enrichment.lat IS 'Stadium latitude from P115->P625, not club location';
COMMENT ON COLUMN team_wikidata_enrichment.lon IS 'Stadium longitude from P115->P625, not club location';
