-- =============================================================================
-- Migration: logos_004_add_original_svg.sql
-- Description: Add r2_key_original_svg to preserve vector logos for future use
-- Date: 2026-01-28
-- =============================================================================
--
-- Context:
-- When users upload SVG logos, we convert to PNG for IA processing (Gemini doesn't
-- support SVG input). However, we now also preserve the original SVG in R2 for:
-- - Future re-rendering at different resolutions
-- - Backup of vector source
-- - Potential use if IA models support SVG in the future
--
-- Storage format: teams/{internal_id}/{apifb_id}-{slug}_original_svg_v{rev}.svg
-- =============================================================================

-- Add r2_key_original_svg column to team_logos
ALTER TABLE team_logos
ADD COLUMN IF NOT EXISTS r2_key_original_svg VARCHAR(255) DEFAULT NULL;

COMMENT ON COLUMN team_logos.r2_key_original_svg IS 'R2 key for original SVG file (if uploaded as SVG)';

-- =============================================================================
-- Verification query (run after migration)
-- =============================================================================
-- SELECT column_name, data_type, is_nullable
-- FROM information_schema.columns
-- WHERE table_name = 'team_logos' AND column_name = 'r2_key_original_svg';
