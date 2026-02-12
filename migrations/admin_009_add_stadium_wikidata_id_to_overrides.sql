-- Migration: admin_009_add_stadium_wikidata_id_to_overrides.sql
-- Description: Add stadium_wikidata_id column to team_enrichment_overrides
-- Author: Claude (Owner request)
-- Date: 2026-02-11

-- Add stadium_wikidata_id column (idempotent)
ALTER TABLE team_enrichment_overrides
ADD COLUMN IF NOT EXISTS stadium_wikidata_id VARCHAR(20);

COMMENT ON COLUMN team_enrichment_overrides.stadium_wikidata_id IS 'Wikidata Q-number of stadium (e.g. Q12345)';
