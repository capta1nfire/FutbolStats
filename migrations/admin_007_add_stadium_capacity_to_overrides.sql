-- Migration: admin_007_add_stadium_capacity_to_overrides.sql
-- Description: Add stadium_capacity column to team_enrichment_overrides
-- Author: Master (ABE approval)
-- Date: 2026-02-04
-- P0 ABE: Made idempotent with DO block

-- Add stadium_capacity column (idempotent)
ALTER TABLE team_enrichment_overrides
ADD COLUMN IF NOT EXISTS stadium_capacity INTEGER;

-- Add constraint for reasonable capacity values (0-200000)
-- P0 ABE: Idempotent - check if constraint exists before adding
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'chk_stadium_capacity'
          AND conrelid = 'team_enrichment_overrides'::regclass
    ) THEN
        ALTER TABLE team_enrichment_overrides
        ADD CONSTRAINT chk_stadium_capacity
        CHECK (stadium_capacity IS NULL OR (stadium_capacity >= 0 AND stadium_capacity < 200000));
    END IF;
END $$;

-- Update table comment
COMMENT ON COLUMN team_enrichment_overrides.stadium_capacity IS 'Stadium capacity override (0-200000)';
