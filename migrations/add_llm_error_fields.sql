-- Migration: Add LLM error observability fields to post_match_audits
-- Date: 2026-01-11
-- Purpose: Store error details for RCA and monitoring

ALTER TABLE post_match_audits ADD COLUMN IF NOT EXISTS llm_narrative_error_code VARCHAR(50);
ALTER TABLE post_match_audits ADD COLUMN IF NOT EXISTS llm_narrative_error_detail VARCHAR(500);
ALTER TABLE post_match_audits ADD COLUMN IF NOT EXISTS llm_narrative_request_id VARCHAR(100);
ALTER TABLE post_match_audits ADD COLUMN IF NOT EXISTS llm_narrative_attempts INTEGER;

-- Add index for error analysis
CREATE INDEX IF NOT EXISTS idx_pma_llm_error_code ON post_match_audits(llm_narrative_error_code) WHERE llm_narrative_error_code IS NOT NULL;

COMMENT ON COLUMN post_match_audits.llm_narrative_error_code IS 'Error code: runpod_http_error, runpod_timeout, schema_invalid, json_parse_error, gating_skipped, empty_output, unknown';
COMMENT ON COLUMN post_match_audits.llm_narrative_error_detail IS 'Error detail/exception message (truncated to 500 chars)';
COMMENT ON COLUMN post_match_audits.llm_narrative_request_id IS 'RunPod job ID for correlation';
COMMENT ON COLUMN post_match_audits.llm_narrative_attempts IS 'Number of generation attempts (1 or 2)';
