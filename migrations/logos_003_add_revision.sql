-- =============================================================================
-- Migration: logos_003_add_revision.sql
-- Description: Add revision column for immutable asset versioning (ABE recommendation)
-- Date: 2026-01-28
-- =============================================================================
--
-- Context:
-- R2 keys now include revision number for cache busting without CDN purges.
-- Format: teams/{internal_id}/{apifb_id}-{slug}_{variant}_v{rev}.{ext}
-- Example: teams/1/529-america-de-cali_original_v1.png
--
-- Cache strategy: URLs are immutable, Cache-Control: public, max-age=31536000, immutable
-- When regenerating, increment revision → new URL → automatic cache invalidation.
-- =============================================================================

-- Add revision column to team_logos
ALTER TABLE team_logos
ADD COLUMN IF NOT EXISTS revision INTEGER NOT NULL DEFAULT 1;

COMMENT ON COLUMN team_logos.revision IS 'Asset revision number, increments on regeneration for cache busting';

-- Add revision column to competition_logos
ALTER TABLE competition_logos
ADD COLUMN IF NOT EXISTS revision INTEGER NOT NULL DEFAULT 1;

COMMENT ON COLUMN competition_logos.revision IS 'Asset revision number, increments on regeneration for cache busting';

-- =============================================================================
-- Verification query (run after migration)
-- =============================================================================
-- SELECT 'team_logos' as table_name, column_name, data_type, column_default
-- FROM information_schema.columns
-- WHERE table_name = 'team_logos' AND column_name = 'revision'
-- UNION ALL
-- SELECT 'competition_logos', column_name, data_type, column_default
-- FROM information_schema.columns
-- WHERE table_name = 'competition_logos' AND column_name = 'revision';
