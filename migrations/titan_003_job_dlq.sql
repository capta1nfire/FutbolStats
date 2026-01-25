-- Migration: Create titan.job_dlq (Dead Letter Queue)
-- Date: 2026-01-25
-- Purpose: FASE 1 - Track failed extractions for retry and debugging
-- Depends: titan_001_create_schema.sql
-- Plan: zazzy-jingling-pudding.md v1.1

-- =============================================================================
-- job_dlq: Dead Letter Queue for failed extraction jobs
-- =============================================================================
-- Key design decisions:
-- 1. Captures all error info for debugging and replay
-- 2. attempts counter for exponential backoff
-- 3. resolution tracking for manual fixes
-- 4. Trigger for automatic updated_at

CREATE TABLE IF NOT EXISTS titan.job_dlq (
    dlq_id          BIGSERIAL PRIMARY KEY,

    -- Job identification (correlates with raw_extractions.job_id)
    job_id          UUID NOT NULL,
    source_id       VARCHAR(50) NOT NULL,       -- 'api_football', 'understat', etc.
    idempotency_key CHAR(32) NOT NULL,          -- Same key that would have been used

    -- Error details
    error_type      VARCHAR(50) NOT NULL,       -- 'http_error', 'timeout', 'rate_limit', 'parse_error'
    error_message   TEXT,                       -- Full error message/traceback
    http_status     SMALLINT,                   -- HTTP status if applicable (429, 500, etc.)

    -- Retry info
    attempts        SMALLINT NOT NULL DEFAULT 1,-- Number of attempts made
    max_attempts    SMALLINT NOT NULL DEFAULT 3,-- Max attempts before giving up

    -- Request info for replay
    endpoint        TEXT NOT NULL,              -- Full endpoint path
    params          JSONB,                      -- Request parameters
    date_bucket     DATE NOT NULL,              -- Target date

    -- Timestamps
    first_attempt   TIMESTAMPTZ NOT NULL,       -- When first attempted
    last_attempt    TIMESTAMPTZ NOT NULL,       -- When last attempted
    next_retry_at   TIMESTAMPTZ,                -- Scheduled retry time (NULL if giving up)
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Resolution (manual fix)
    resolved_at     TIMESTAMPTZ,                -- When resolved (NULL if pending)
    resolution      VARCHAR(50),                -- 'retried_success', 'manual_skip', 'source_unavailable'
    resolved_by     VARCHAR(100)                -- Who/what resolved it
);

-- Trigger for automatic updated_at
CREATE TRIGGER update_job_dlq_updated_at
    BEFORE UPDATE ON titan.job_dlq
    FOR EACH ROW
    EXECUTE FUNCTION titan.update_updated_at_column();

-- Indexes for common query patterns
CREATE INDEX IF NOT EXISTS idx_dlq_pending
    ON titan.job_dlq(source_id, created_at)
    WHERE resolved_at IS NULL;

CREATE INDEX IF NOT EXISTS idx_dlq_retry
    ON titan.job_dlq(next_retry_at)
    WHERE resolved_at IS NULL AND next_retry_at IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_dlq_job
    ON titan.job_dlq(job_id);

CREATE INDEX IF NOT EXISTS idx_dlq_source_date
    ON titan.job_dlq(source_id, date_bucket);

-- Documentation
COMMENT ON TABLE titan.job_dlq IS 'Dead Letter Queue: failed extraction jobs for debugging and replay';
COMMENT ON COLUMN titan.job_dlq.idempotency_key IS 'Same key that would have been used in raw_extractions';
COMMENT ON COLUMN titan.job_dlq.error_type IS 'Category: http_error, timeout, rate_limit, parse_error, validation_error';
COMMENT ON COLUMN titan.job_dlq.attempts IS 'Number of attempts made (incremented on each retry)';
COMMENT ON COLUMN titan.job_dlq.next_retry_at IS 'Scheduled retry time with exponential backoff (NULL if max attempts reached)';
COMMENT ON COLUMN titan.job_dlq.resolution IS 'How resolved: retried_success, manual_skip, source_unavailable, data_recovered';
