-- Migration: Add R2 offload support to raw_extractions
-- Date: 2026-01-25
-- Purpose: FASE 2 - Offload large JSONB responses to Cloudflare R2
-- Plan: zazzy-jingling-pudding.md v2.0

-- =============================================================================
-- R2 Offload Columns
-- =============================================================================

ALTER TABLE titan.raw_extractions
ADD COLUMN IF NOT EXISTS r2_bucket VARCHAR(100),
ADD COLUMN IF NOT EXISTS r2_key VARCHAR(500),
ADD COLUMN IF NOT EXISTS response_size_bytes INT;

-- =============================================================================
-- Integrity Constraint (Auditor Condition #4, strengthened per audit)
-- =============================================================================
-- Rule: If r2_key IS NOT NULL, then:
--   1. r2_bucket MUST also be NOT NULL
--   2. response_body MUST be NULL (offloaded to R2)
-- This ensures we don't store data in both places and have complete R2 refs

ALTER TABLE titan.raw_extractions
ADD CONSTRAINT chk_r2_offload_integrity
CHECK (
    (r2_key IS NULL) OR (r2_key IS NOT NULL AND r2_bucket IS NOT NULL AND response_body IS NULL)
);

-- =============================================================================
-- Indexes
-- =============================================================================

-- Index for R2 cleanup/audit queries
CREATE INDEX IF NOT EXISTS idx_raw_r2_offloaded
ON titan.raw_extractions(r2_key)
WHERE r2_key IS NOT NULL;

-- Index for retention policy (find old rows not yet offloaded)
CREATE INDEX IF NOT EXISTS idx_raw_retention
ON titan.raw_extractions(created_at)
WHERE r2_key IS NULL AND response_body IS NOT NULL;

-- =============================================================================
-- Comments
-- =============================================================================

COMMENT ON COLUMN titan.raw_extractions.r2_bucket IS 'R2 bucket name if response offloaded (NULL if stored in DB)';
COMMENT ON COLUMN titan.raw_extractions.r2_key IS 'R2 object key if response offloaded (NULL if stored in DB)';
COMMENT ON COLUMN titan.raw_extractions.response_size_bytes IS 'Original response size in bytes (for metrics/billing)';
COMMENT ON CONSTRAINT chk_r2_offload_integrity ON titan.raw_extractions IS 'Ensures response_body is NULL when offloaded to R2';
