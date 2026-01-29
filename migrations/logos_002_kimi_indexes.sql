-- =============================================================================
-- Migration: logos_002_kimi_indexes.sql
-- Purpose: Add indexes recommended by ATI (Kimi) audit
-- Date: 2026-01-28
-- =============================================================================

-- Index for review grid queries (league + review_status)
-- Used by: GET /dashboard/logos/review/league/{id}
CREATE INDEX IF NOT EXISTS idx_team_logos_league_review
    ON team_logos(batch_job_id, review_status)
    WHERE review_status IN ('pending', 'needs_regeneration');

-- Index for batch job queries by league and status
-- Used by: Checking for existing running jobs before starting new one
CREATE INDEX IF NOT EXISTS idx_logo_batch_jobs_league_status
    ON logo_batch_jobs(league_id, status, created_at);

-- Note: Run this migration manually in production:
-- psql $DATABASE_URL -f migrations/logos_002_kimi_indexes.sql
