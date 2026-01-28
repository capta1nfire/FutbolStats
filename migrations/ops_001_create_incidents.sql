-- Migration: Create ops_incidents table for persistent incident tracking
-- Date: 2026-01-27
-- Purpose: Persist incidents from _aggregate_incidents() so created_at is stable,
--          acknowledge/resolve actions survive refresh, and timeline/history is populated.
-- Approved by: ABE (with guardrails: last_seen_at, grace window, reopen logic, actor in timeline)

CREATE TABLE IF NOT EXISTS ops_incidents (
    -- Stable ID from make_id() hash (MD5 first 8 hex → int)
    id BIGINT PRIMARY KEY,

    -- Source identification (UNIQUE pair for upsert)
    source VARCHAR(30) NOT NULL,         -- sentry|predictions|jobs|fastpath|budget
    source_key VARCHAR(100) NOT NULL,    -- key within source (e.g. "stats_backfill", "health", issue title)

    -- Severity and status
    severity VARCHAR(20) NOT NULL,       -- critical|warning|info
    status VARCHAR(20) NOT NULL DEFAULT 'active',  -- active|acknowledged|resolved
    type VARCHAR(30) NOT NULL,           -- backend type: sentry|predictions|scheduler|llm|api_budget

    -- Content
    title VARCHAR(200) NOT NULL,
    description TEXT,
    details JSONB,                       -- operational context dict (max ~16KB enforced in app)
    runbook_url VARCHAR(500),

    -- Timeline: array of {ts, message, actor, action}
    -- actor: "system"|"user"
    -- action: "created"|"acknowledged"|"resolved"|"reopened"|"updated"|"auto_resolved"
    timeline JSONB NOT NULL DEFAULT '[]'::jsonb,

    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),       -- first time incident was detected (never overwritten)
    last_seen_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),     -- last time source reported this incident
    acknowledged_at TIMESTAMPTZ,                         -- when user acknowledged
    resolved_at TIMESTAMPTZ,                             -- when resolved (user or auto)
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),       -- last modification of any kind

    -- Unique constraint for upsert
    CONSTRAINT uq_ops_incidents_source_key UNIQUE (source, source_key)
);

-- Indexes for efficient queries
CREATE INDEX IF NOT EXISTS idx_ops_incidents_status ON ops_incidents(status);
CREATE INDEX IF NOT EXISTS idx_ops_incidents_source ON ops_incidents(source);
CREATE INDEX IF NOT EXISTS idx_ops_incidents_created_at ON ops_incidents(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_ops_incidents_status_resolved ON ops_incidents(status, resolved_at)
    WHERE status = 'resolved';  -- for purge job

-- Comments
COMMENT ON TABLE ops_incidents IS 'Persistent incidents from _aggregate_incidents(): Sentry, Jobs, Predictions, FastPath, Budget';
COMMENT ON COLUMN ops_incidents.id IS 'Stable hash from make_id(source, source_key) — MD5 first 8 hex as int';
COMMENT ON COLUMN ops_incidents.source IS 'Incident source: sentry, predictions, jobs, fastpath, budget';
COMMENT ON COLUMN ops_incidents.source_key IS 'Key within source for dedup (e.g. job name, issue title truncated)';
COMMENT ON COLUMN ops_incidents.last_seen_at IS 'Updated on every aggregation cycle where this incident is still active';
COMMENT ON COLUMN ops_incidents.timeline IS 'JSONB array of {ts, message, actor, action} events for History tab';
COMMENT ON COLUMN ops_incidents.details IS 'Operational context dict (truncated to ~16KB in app layer)';
