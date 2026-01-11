-- Migration: Add fastpath_ticks table for persistent tick tracking
-- Date: 2026-01-11
-- Purpose: Store fast-path tick results for ops dashboard (not in-memory)

CREATE TABLE IF NOT EXISTS fastpath_ticks (
    id SERIAL PRIMARY KEY,
    tick_at TIMESTAMP NOT NULL DEFAULT NOW(),
    selected INTEGER NOT NULL DEFAULT 0,
    refreshed INTEGER NOT NULL DEFAULT 0,
    ready INTEGER NOT NULL DEFAULT 0,
    enqueued INTEGER NOT NULL DEFAULT 0,
    completed INTEGER NOT NULL DEFAULT 0,
    errors INTEGER NOT NULL DEFAULT 0,
    skipped INTEGER NOT NULL DEFAULT 0,
    duration_ms INTEGER,
    error_detail TEXT
);

-- Index for recent tick lookups
CREATE INDEX IF NOT EXISTS idx_fastpath_ticks_at ON fastpath_ticks(tick_at DESC);

-- Keep only last 7 days of ticks (cleanup via scheduled job or manual)
COMMENT ON TABLE fastpath_ticks IS 'Fast-path LLM narrative tick history for ops monitoring';
