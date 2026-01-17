-- Migration: Add elapsed_extra field to matches for injury/added time
-- Date: 2026-01-17
-- Purpose: Store added/injury time minutes (e.g., 3 for 90+3) for live match display

-- Added/injury time minutes (null when not in stoppage time)
ALTER TABLE matches ADD COLUMN IF NOT EXISTS elapsed_extra INTEGER;

COMMENT ON COLUMN matches.elapsed_extra IS 'Added/injury time minutes (e.g., 3 for 90+3 display)';
