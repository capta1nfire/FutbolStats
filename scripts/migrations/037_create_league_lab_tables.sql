-- Migration 037: Create league_lab_runs + league_lab_results tables
-- Part of: Auto-Lab Online (PASO 2 of SSOT + Auto-Lab initiative)
-- Date: 2026-02-19
--
-- Advisory: Auto-Lab results are informational only.
-- They do NOT modify league_serving_config automatically.

BEGIN;

CREATE TABLE IF NOT EXISTS league_lab_runs (
    id              SERIAL PRIMARY KEY,
    league_id       INTEGER NOT NULL REFERENCES admin_leagues(league_id),
    started_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    finished_at     TIMESTAMPTZ,
    status          TEXT NOT NULL DEFAULT 'running'
                    CHECK (status IN ('running', 'completed', 'error', 'timeout')),
    mode            TEXT NOT NULL DEFAULT 'fast',
    trigger_reason  TEXT,
    n_matches_used  INTEGER,
    n_tests_run     INTEGER,
    duration_ms     INTEGER,
    error_message   TEXT,
    best_test_name  TEXT,
    best_brier      FLOAT,
    market_brier    FLOAT,
    delta_vs_market FLOAT,
    config_snapshot JSONB,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_lab_runs_league
    ON league_lab_runs(league_id, started_at DESC);

CREATE TABLE IF NOT EXISTS league_lab_results (
    id              SERIAL PRIMARY KEY,
    run_id          INTEGER NOT NULL REFERENCES league_lab_runs(id) ON DELETE CASCADE,
    league_id       INTEGER NOT NULL,
    test_name       TEXT NOT NULL,
    test_type       TEXT NOT NULL DEFAULT 'feature_set',
    brier_ensemble  FLOAT,
    brier_market    FLOAT,
    delta_vs_market FLOAT,
    accuracy        FLOAT,
    n_train         INTEGER,
    n_test          INTEGER,
    n_features      INTEGER,
    result_json     JSONB,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_lab_results_run
    ON league_lab_results(run_id);

COMMIT;
