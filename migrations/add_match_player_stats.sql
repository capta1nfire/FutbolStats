-- Migration: Create match_player_stats table
-- Date: 2026-02-14
-- Purpose: Per-player per-match stats from API-Football /fixtures/players
-- Source of truth for PTS (Player Talent Score) and future xT features.
-- Idempotent: All statements use IF NOT EXISTS guards.

BEGIN;

CREATE TABLE IF NOT EXISTS match_player_stats (
    id                  SERIAL          PRIMARY KEY,
    match_id            INTEGER         NOT NULL REFERENCES matches(id),
    player_external_id  INTEGER         NOT NULL,   -- API-Football player ID (no FK to players)
    player_name         TEXT,                        -- For audit/debugging
    team_external_id    INTEGER,                     -- API-Football team ID
    team_id             INTEGER,                     -- Internal team ID (resolved at ingestion, nullable)
    match_date          DATE,                        -- Denormalized from matches.date for PTS lookups

    -- Key columns for direct queries (PTS, talent delta)
    rating              NUMERIC(4,2),                -- "7.3" -> 7.30, NULL if bench/unrated
    minutes             SMALLINT,                    -- 0-120+, NULL if bench
    position            VARCHAR(5),                  -- G/D/M/F
    is_substitute       BOOLEAN,
    is_captain          BOOLEAN,

    -- Granular stats (for Feature Lab, future xT)
    goals               SMALLINT,
    assists             SMALLINT,
    saves               SMALLINT,
    shots_total         SMALLINT,
    shots_on_target     SMALLINT,
    passes_total        SMALLINT,
    passes_key          SMALLINT,
    passes_accuracy     SMALLINT,                    -- Percentage 0-100
    tackles             SMALLINT,
    interceptions       SMALLINT,
    blocks              SMALLINT,
    duels_total         SMALLINT,
    duels_won           SMALLINT,
    dribbles_attempts   SMALLINT,
    dribbles_success    SMALLINT,
    fouls_drawn         SMALLINT,
    fouls_committed     SMALLINT,
    yellow_cards        SMALLINT,
    red_cards           SMALLINT,

    -- Per-player statistics block (not full payload)
    raw_json            JSONB,

    -- Metadata
    captured_at         TIMESTAMPTZ     DEFAULT NOW(),

    UNIQUE(match_id, player_external_id)
);

-- Critical index for PTS: "last 10 matches for player X" ordered by date
CREATE INDEX IF NOT EXISTS idx_mps_player_date
    ON match_player_stats (player_external_id, match_date DESC);

-- For per-match queries
CREATE INDEX IF NOT EXISTS idx_mps_match
    ON match_player_stats (match_id);

-- For per-team queries
CREATE INDEX IF NOT EXISTS idx_mps_team
    ON match_player_stats (team_external_id);

COMMENT ON TABLE match_player_stats IS 'Per-player per-match stats from API-Football /fixtures/players. PIT: always filter WHERE match_date < asof_timestamp.';
COMMENT ON COLUMN match_player_stats.rating IS 'API-Football post-match rating. NULL for unplayed bench. PIT-critical: never use same-match rating for pre-match PTS.';
COMMENT ON COLUMN match_player_stats.player_external_id IS 'API-Football player ID. No FK to players table to allow ingestion without prior catalog sync.';
COMMENT ON COLUMN match_player_stats.match_date IS 'Denormalized from matches.date. Avoids JOIN for PTS last-N lookups.';
COMMENT ON COLUMN match_player_stats.raw_json IS 'Per-player statistics block from API response (not full payload). For future feature extraction without schema migration.';

COMMIT;
