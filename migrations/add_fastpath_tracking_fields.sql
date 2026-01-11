-- Migration: Add Fast-Path narrative tracking fields to matches
-- Date: 2026-01-11
-- Purpose: Track match completion and stats readiness for fast-path LLM narrative generation

-- Timestamp when match transitioned to FT/AET/PEN status
ALTER TABLE matches ADD COLUMN IF NOT EXISTS finished_at TIMESTAMP;

-- Timestamp when stats passed gating requirements (possession, shots, shots_on_goal)
ALTER TABLE matches ADD COLUMN IF NOT EXISTS stats_ready_at TIMESTAMP;

-- Last time we attempted to refresh stats for this match
ALTER TABLE matches ADD COLUMN IF NOT EXISTS stats_last_checked_at TIMESTAMP;

-- Index for fast-path candidate selection query
CREATE INDEX IF NOT EXISTS idx_matches_fastpath_candidates
ON matches(finished_at, stats_ready_at)
WHERE finished_at IS NOT NULL AND stats_ready_at IS NULL;

-- Index for recent finished matches lookback
CREATE INDEX IF NOT EXISTS idx_matches_finished_at
ON matches(finished_at)
WHERE finished_at IS NOT NULL;

COMMENT ON COLUMN matches.finished_at IS 'When match finished (FT/AET/PEN detected) - triggers fast-path';
COMMENT ON COLUMN matches.stats_ready_at IS 'When stats passed gating requirements for LLM';
COMMENT ON COLUMN matches.stats_last_checked_at IS 'Last stats refresh attempt (for backoff)';
