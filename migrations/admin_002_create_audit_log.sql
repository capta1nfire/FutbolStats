-- Migration: Create admin_audit_log table
-- Date: 2026-01-26
-- Purpose: P2B - Audit trail for admin_leagues mutations
-- Plan: zazzy-jingling-pudding.md v2.0

-- =============================================================================
-- admin_audit_log: Audit trail for admin panel mutations
-- =============================================================================
-- Design:
-- 1. Generic entity_type/entity_id for future expansion (teams, rules, etc.)
-- 2. before_json/after_json capture full state diff
-- 3. actor is optional (null = system/unknown)
-- 4. Immutable: no UPDATE/DELETE on this table

CREATE TABLE IF NOT EXISTS admin_audit_log (
    id              SERIAL PRIMARY KEY,
    entity_type     TEXT NOT NULL,                  -- e.g., 'admin_leagues', 'admin_league_groups'
    entity_id       TEXT NOT NULL,                  -- e.g., '39' (league_id as string for flexibility)
    action          TEXT NOT NULL,                  -- e.g., 'update', 'create', 'delete'
    actor           TEXT,                           -- e.g., 'dashboard', 'api', null for system
    before_json     JSONB,                          -- state before change (null for create)
    after_json      JSONB,                          -- state after change (null for delete)
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- =============================================================================
-- INDEXES
-- =============================================================================

-- Query by entity
CREATE INDEX IF NOT EXISTS idx_audit_log_entity
    ON admin_audit_log(entity_type, entity_id);

-- Query by time (recent first)
CREATE INDEX IF NOT EXISTS idx_audit_log_created
    ON admin_audit_log(created_at DESC);

-- Query by action
CREATE INDEX IF NOT EXISTS idx_audit_log_action
    ON admin_audit_log(action);

-- =============================================================================
-- DOCUMENTATION
-- =============================================================================

COMMENT ON TABLE admin_audit_log IS 'Immutable audit trail for admin panel mutations. Do not UPDATE or DELETE rows.';
COMMENT ON COLUMN admin_audit_log.entity_type IS 'Type of entity modified: admin_leagues, admin_league_groups, etc.';
COMMENT ON COLUMN admin_audit_log.entity_id IS 'ID of entity as string (league_id, group_id, etc.)';
COMMENT ON COLUMN admin_audit_log.action IS 'Action performed: update, create, delete';
COMMENT ON COLUMN admin_audit_log.actor IS 'Who performed the action: dashboard, api, system, or null';
COMMENT ON COLUMN admin_audit_log.before_json IS 'Full entity state before change (null for create)';
COMMENT ON COLUMN admin_audit_log.after_json IS 'Full entity state after change (null for delete)';
