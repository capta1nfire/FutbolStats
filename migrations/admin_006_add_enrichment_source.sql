-- Migration: admin_006_add_enrichment_source.sql
-- Description: Add enrichment_source column to track data origin
-- Author: Master (per Kimi recommendation)
-- Date: 2026-02-04

-- Add enrichment_source column to track where data came from
-- Values: "wikidata", "wikipedia", "wikidata+wikipedia", "override:manual", etc.
ALTER TABLE team_wikidata_enrichment
ADD COLUMN IF NOT EXISTS enrichment_source VARCHAR(50) DEFAULT 'wikidata';

-- Index for filtering by source
CREATE INDEX IF NOT EXISTS idx_twe_enrichment_source ON team_wikidata_enrichment(enrichment_source);

COMMENT ON COLUMN team_wikidata_enrichment.enrichment_source IS 'Data origin: wikidata, wikipedia, wikidata+wikipedia, override:{source}';
