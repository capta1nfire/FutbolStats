-- Admin 003: Add Wikipedia/Wikidata fields to teams
--
-- Purpose:
-- - Allow ops/dashboard to link a team to Wikipedia/Wikidata
-- - Store user input (wiki_url, wikidata_id) and backend-derived fields
--
-- Notes:
-- - All columns are nullable and optional
-- - wikidata_id gets a partial unique index when present

ALTER TABLE teams ADD COLUMN IF NOT EXISTS wiki_url TEXT;
ALTER TABLE teams ADD COLUMN IF NOT EXISTS wikidata_id VARCHAR(20);
ALTER TABLE teams ADD COLUMN IF NOT EXISTS wiki_title VARCHAR(255);
ALTER TABLE teams ADD COLUMN IF NOT EXISTS wiki_lang VARCHAR(32);
ALTER TABLE teams ADD COLUMN IF NOT EXISTS wiki_url_cached TEXT;
ALTER TABLE teams ADD COLUMN IF NOT EXISTS wiki_source VARCHAR(32);
ALTER TABLE teams ADD COLUMN IF NOT EXISTS wiki_confidence DOUBLE PRECISION;
ALTER TABLE teams ADD COLUMN IF NOT EXISTS wiki_matched_at TIMESTAMP;

CREATE UNIQUE INDEX IF NOT EXISTS idx_teams_wikidata_id
  ON teams (wikidata_id)
  WHERE wikidata_id IS NOT NULL;

