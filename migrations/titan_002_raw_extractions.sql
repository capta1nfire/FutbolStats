-- Migration: Create titan.raw_extractions staging table
-- Date: 2026-01-25
-- Purpose: FASE 1 - Immutable staging for all API extractions with idempotency
-- Depends: titan_001_create_schema.sql
-- Plan: zazzy-jingling-pudding.md v1.1

-- =============================================================================
-- raw_extractions: Staging table for ALL external API responses
-- =============================================================================
-- Key design decisions:
-- 1. idempotency_key CHAR(32) = SHA256(source|endpoint|params|date)[:32]
-- 2. captured_at = momento exacto de captura (para PIT compliance)
-- 3. UNIQUE constraint previene duplicados
-- 4. response_body JSONB preserva respuesta original sin transformación

CREATE TABLE IF NOT EXISTS titan.raw_extractions (
    extraction_id   BIGSERIAL PRIMARY KEY,

    -- Source identification
    source_id       VARCHAR(50) NOT NULL,       -- 'api_football', 'understat', 'sofascore', etc.
    job_id          UUID NOT NULL,              -- Correlación con job_dlq si falla

    -- Request info
    url             TEXT NOT NULL,              -- Full URL called
    endpoint        VARCHAR(100) NOT NULL,      -- Endpoint name: 'fixtures', 'odds', 'statistics'
    params_hash     CHAR(32) NOT NULL,          -- MD5 of normalized params (for debugging)
    date_bucket     DATE NOT NULL,              -- Partition key (YYYY-MM-DD of target date)

    -- Response
    response_type   VARCHAR(20) NOT NULL,       -- 'json', 'html', 'error'
    response_body   JSONB,                      -- Raw response (NULL if error)
    http_status     SMALLINT NOT NULL,          -- HTTP status code (200, 429, 500, etc.)
    response_time_ms INT,                       -- Response time in milliseconds

    -- PIT Compliance (CRÍTICO)
    captured_at     TIMESTAMPTZ NOT NULL,       -- Momento exacto de captura

    -- Idempotency (CRÍTICO)
    idempotency_key CHAR(32) NOT NULL,          -- SHA256[:32] determinístico

    -- Metadata
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Prevent duplicate extractions
    CONSTRAINT uq_extraction_idempotency UNIQUE (idempotency_key)
);

-- Indexes for common query patterns
CREATE INDEX IF NOT EXISTS idx_raw_source_date
    ON titan.raw_extractions(source_id, date_bucket);

CREATE INDEX IF NOT EXISTS idx_raw_captured
    ON titan.raw_extractions(captured_at);

CREATE INDEX IF NOT EXISTS idx_raw_job
    ON titan.raw_extractions(job_id);

CREATE INDEX IF NOT EXISTS idx_raw_endpoint
    ON titan.raw_extractions(source_id, endpoint);

-- Documentation
COMMENT ON TABLE titan.raw_extractions IS 'Immutable staging for all external API responses. Never UPDATE, only INSERT.';
COMMENT ON COLUMN titan.raw_extractions.idempotency_key IS 'SHA256(source_id|endpoint|params|date_bucket)[:32] - prevents duplicate extractions';
COMMENT ON COLUMN titan.raw_extractions.captured_at IS 'PIT timestamp: exact moment when data was captured from external source';
COMMENT ON COLUMN titan.raw_extractions.params_hash IS 'MD5 of JSON-serialized params for debugging (not for idempotency)';
COMMENT ON COLUMN titan.raw_extractions.date_bucket IS 'Logical date partition (match date or extraction date)';
