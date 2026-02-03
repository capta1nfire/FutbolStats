-- Migration: Archive shadow/two_stage legacy predictions
-- Author: Claude (per ABE FASE 3 directive)
-- Date: 2026-01-30
--
-- Purpose:
--   - Archive 137 legacy shadow predictions from `predictions` table
--   - Mark them as LEGACY-ARCHIVED to prevent accidental use
--   - DO NOT DELETE (FK constraints in prediction_outcomes)
--   - shadow_predictions remains the single source of truth
--
-- Pre-checks confirmed:
--   - 137 rows with model_version = 'v1.1.0-two_stage'
--   - 137 FK references in prediction_outcomes (cannot delete)
--   - Archive table does not exist yet

BEGIN;

-- 1) Create archive table with same schema (includes constraints/indexes/defaults)
CREATE TABLE IF NOT EXISTS predictions_shadow_legacy (LIKE predictions INCLUDING ALL);

-- 2) Add audit columns to archive table (idempotent)
ALTER TABLE predictions_shadow_legacy
  ADD COLUMN IF NOT EXISTS archived_at TIMESTAMP WITHOUT TIME ZONE;

ALTER TABLE predictions_shadow_legacy
  ADD COLUMN IF NOT EXISTS archived_reason TEXT;

-- 3) Insert into archive (idempotent: don't duplicate by id)
INSERT INTO predictions_shadow_legacy
SELECT p.*, NOW() AS archived_at, 'shadow/two_stage legacy rows found in predictions (ABE FASE 3)' AS archived_reason
FROM predictions p
WHERE (p.model_version ILIKE '%two_stage%' OR p.model_version ILIKE '%shadow%')
  AND NOT EXISTS (
    SELECT 1 FROM predictions_shadow_legacy a WHERE a.id = p.id
  );

-- 4) "Disable" in predictions (no delete): mark model_version
--    Prevents anyone from filtering by v1.1.0-two_stage and using it accidentally
UPDATE predictions
SET model_version = model_version || '-LEGACY-ARCHIVED'
WHERE (model_version ILIKE '%two_stage%' OR model_version ILIKE '%shadow%')
  AND model_version NOT ILIKE '%LEGACY-ARCHIVED%';

COMMIT;

-- Post-check queries (run after migration):
--
-- A) Verify archive count:
--    SELECT COUNT(*) AS archived FROM predictions_shadow_legacy WHERE archived_reason ILIKE '%ABE FASE 3%';
--
-- B) Verify no active shadow model_versions remain:
--    SELECT COUNT(*) AS remaining_unarchived FROM predictions
--    WHERE (model_version ILIKE '%two_stage%' OR model_version ILIKE '%shadow%')
--      AND model_version NOT ILIKE '%LEGACY-ARCHIVED%';
--
-- Rollback (if needed - only reverts the suffix, keeps archive):
--    UPDATE predictions
--    SET model_version = REPLACE(model_version, '-LEGACY-ARCHIVED', '')
--    WHERE model_version ILIKE '%LEGACY-ARCHIVED%';
