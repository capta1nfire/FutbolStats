-- Migration: Phase 2 Sprint 1 — Player ID Mapping, Prediction CLV, and column additions
-- Date: 2026-02-14
-- Sprint: Phase 2 Sprint 1
--
-- Purpose:
--   Documents schema changes already applied to production for Phase 2 Sprint 1:
--   1. player_id_mapping  — Maps API-Football player IDs to Sofascore player IDs
--   2. prediction_clv     — Closing Line Value tracking per prediction/bookmaker
--   3. predictions.asof_timestamp — Timestamp when prediction odds were captured
--   4. match_lineups.lineup_detected_at — Timestamp when lineup was first detected
--
-- Idempotent: All statements use IF NOT EXISTS / IF EXISTS guards.
-- Safe to re-run without side effects.

BEGIN;

-- ============================================================================
-- 1. player_id_mapping
-- ============================================================================
-- Maps player IDs between API-Football and Sofascore for lineup matching.
-- Used by the squad sync and match lineups backfill pipeline.

CREATE TABLE IF NOT EXISTS player_id_mapping (
    id              SERIAL          PRIMARY KEY,
    api_football_id INTEGER         NOT NULL,
    sofascore_id    VARCHAR         NOT NULL,
    player_name     VARCHAR,
    team_id         INTEGER,
    position        VARCHAR,
    confidence      DOUBLE PRECISION NOT NULL,
    method          VARCHAR         NOT NULL,
    status          VARCHAR         DEFAULT 'active',
    source_match_id INTEGER,
    created_at      TIMESTAMPTZ     DEFAULT now(),
    updated_at      TIMESTAMPTZ     DEFAULT now(),

    UNIQUE (api_football_id, sofascore_id)
);

-- Lookup by API-Football ID (used during lineup resolution)
CREATE INDEX IF NOT EXISTS idx_pim_api
    ON player_id_mapping (api_football_id);

-- Lookup by Sofascore ID (used during reverse mapping)
CREATE INDEX IF NOT EXISTS idx_pim_sofascore
    ON player_id_mapping (sofascore_id);

COMMENT ON TABLE player_id_mapping
    IS 'Cross-provider player ID mapping: API-Football <-> Sofascore. Used for lineup matching.';

COMMENT ON COLUMN player_id_mapping.confidence
    IS 'Match confidence score (0.0-1.0). Higher = more reliable mapping.';

COMMENT ON COLUMN player_id_mapping.method
    IS 'How the mapping was established (e.g., name_match, lineup_correlation).';

COMMENT ON COLUMN player_id_mapping.status
    IS 'Mapping status: active, deprecated, manual_override.';

COMMENT ON COLUMN player_id_mapping.source_match_id
    IS 'Match ID where this mapping was first discovered/confirmed.';


-- ============================================================================
-- 2. prediction_clv
-- ============================================================================
-- Tracks Closing Line Value (CLV) for each prediction.
-- Compares odds at prediction time (asof) vs closing odds to measure edge.

CREATE TABLE IF NOT EXISTS prediction_clv (
    id                  SERIAL      PRIMARY KEY,
    prediction_id       INTEGER     REFERENCES predictions(id),
    match_id            INTEGER     NOT NULL,
    asof_timestamp      TIMESTAMPTZ NOT NULL,
    canonical_bookmaker VARCHAR     NOT NULL,

    -- Odds at prediction time
    odds_asof_home      NUMERIC,
    odds_asof_draw      NUMERIC,
    odds_asof_away      NUMERIC,

    -- Implied probabilities at prediction time (de-vigged)
    prob_asof_home      NUMERIC,
    prob_asof_draw      NUMERIC,
    prob_asof_away      NUMERIC,

    -- Implied probabilities at closing time (de-vigged)
    prob_close_home     NUMERIC,
    prob_close_draw     NUMERIC,
    prob_close_away     NUMERIC,

    -- CLV per outcome (prob_asof - prob_close; positive = favorable)
    clv_home            NUMERIC,
    clv_draw            NUMERIC,
    clv_away            NUMERIC,

    -- Selected outcome and its CLV
    selected_outcome    VARCHAR,
    clv_selected        NUMERIC,

    -- Source of closing odds (e.g., pinnacle, bet365)
    close_source        VARCHAR,

    created_at          TIMESTAMPTZ DEFAULT now()
);

-- FK to predictions for joining prediction details (idempotent guard)
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint WHERE conname = 'prediction_clv_prediction_id_fkey'
    ) THEN
        ALTER TABLE prediction_clv
            ADD CONSTRAINT prediction_clv_prediction_id_fkey
            FOREIGN KEY (prediction_id) REFERENCES predictions(id);
    END IF;
END $$;

-- Lookup by match (for match-level CLV analysis)
CREATE INDEX IF NOT EXISTS idx_prediction_clv_match
    ON prediction_clv (match_id);

-- Lookup by prediction (for per-prediction CLV retrieval)
CREATE INDEX IF NOT EXISTS idx_prediction_clv_prediction
    ON prediction_clv (prediction_id);

-- One CLV record per prediction+bookmaker combination
CREATE UNIQUE INDEX IF NOT EXISTS idx_prediction_clv_unique
    ON prediction_clv (prediction_id, canonical_bookmaker);

COMMENT ON TABLE prediction_clv
    IS 'Closing Line Value tracking. Measures whether prediction captured favorable odds vs closing line.';

COMMENT ON COLUMN prediction_clv.asof_timestamp
    IS 'Timestamp when the prediction odds (odds_asof_*) were captured.';

COMMENT ON COLUMN prediction_clv.canonical_bookmaker
    IS 'Normalized bookmaker name (e.g., pinnacle, bet365).';

COMMENT ON COLUMN prediction_clv.clv_selected
    IS 'CLV for the selected outcome. Positive = model captured value before close.';

COMMENT ON COLUMN prediction_clv.close_source
    IS 'Compound audit field: "asof_source|close_method". asof_source: pit_aligned or opening_proxy. close_method: is_closing, latest_pre_kickoff, or single_snapshot.';


-- ============================================================================
-- 3. predictions.asof_timestamp
-- ============================================================================
-- Records the point-in-time when odds were snapshot for this prediction.
-- Critical for PIT (Point-In-Time) correctness in evaluation.

ALTER TABLE predictions
    ADD COLUMN IF NOT EXISTS asof_timestamp TIMESTAMPTZ;

COMMENT ON COLUMN predictions.asof_timestamp
    IS 'Point-in-time timestamp when odds were captured for this prediction. PIT-critical.';


-- ============================================================================
-- 4. match_lineups.lineup_detected_at
-- ============================================================================
-- Records when the official lineup was first detected/ingested.
-- Used to track lineup availability lead time before kickoff.

ALTER TABLE match_lineups
    ADD COLUMN IF NOT EXISTS lineup_detected_at TIMESTAMPTZ;

COMMENT ON COLUMN match_lineups.lineup_detected_at
    IS 'Timestamp when lineup was first detected. Used to measure lead time vs kickoff.';


COMMIT;

-- ============================================================================
-- Post-migration verification queries (run manually, do NOT include in migration):
-- ============================================================================
--
-- 1) Verify player_id_mapping exists with correct columns:
--    SELECT column_name, data_type FROM information_schema.columns
--    WHERE table_name = 'player_id_mapping' ORDER BY ordinal_position;
--
-- 2) Verify prediction_clv exists with correct columns:
--    SELECT column_name, data_type FROM information_schema.columns
--    WHERE table_name = 'prediction_clv' ORDER BY ordinal_position;
--
-- 3) Verify predictions.asof_timestamp:
--    SELECT column_name, data_type FROM information_schema.columns
--    WHERE table_name = 'predictions' AND column_name = 'asof_timestamp';
--
-- 4) Verify match_lineups.lineup_detected_at:
--    SELECT column_name, data_type FROM information_schema.columns
--    WHERE table_name = 'match_lineups' AND column_name = 'lineup_detected_at';
--
-- 5) Verify row counts (should be non-zero if data was already loaded):
--    SELECT 'player_id_mapping' AS tbl, COUNT(*) FROM player_id_mapping
--    UNION ALL
--    SELECT 'prediction_clv', COUNT(*) FROM prediction_clv;
