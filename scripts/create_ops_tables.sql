-- ============================================================
-- Script: create_ops_tables.sql
-- Descripción: Crear tablas OPS para comparativa diaria
-- Ejecutar con: railway run psql < scripts/create_ops_tables.sql
-- ============================================================

-- 1) Tabla match_odds_snapshot
CREATE TABLE IF NOT EXISTS match_odds_snapshot (
    id SERIAL PRIMARY KEY,
    match_id INT NOT NULL REFERENCES matches(id) ON DELETE CASCADE,
    bookmaker VARCHAR(50) NOT NULL,
    odds_home FLOAT,
    odds_draw FLOAT,
    odds_away FLOAT,
    implied_home FLOAT,
    implied_draw FLOAT,
    implied_away FLOAT,
    market_pick VARCHAR(10),
    snapshot_at TIMESTAMPTZ,
    is_primary BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(match_id, bookmaker)
);

-- Índices para match_odds_snapshot
CREATE INDEX IF NOT EXISTS idx_match_odds_snapshot_match_id ON match_odds_snapshot(match_id);
CREATE INDEX IF NOT EXISTS idx_match_odds_snapshot_primary ON match_odds_snapshot(match_id) WHERE is_primary = TRUE;

-- 2) Tabla league_bookmaker_config
CREATE TABLE IF NOT EXISTS league_bookmaker_config (
    id SERIAL PRIMARY KEY,
    league_id INT NOT NULL,
    bookmaker VARCHAR(50) NOT NULL,
    priority INT DEFAULT 1,
    is_primary BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(league_id, bookmaker)
);

-- MVP: Bet365 como primary para las ligas activas
INSERT INTO league_bookmaker_config (league_id, bookmaker, priority, is_primary) VALUES
(39, 'bet365', 1, TRUE),   -- Premier League
(140, 'bet365', 1, TRUE),  -- La Liga
(135, 'bet365', 1, TRUE),  -- Serie A
(78, 'bet365', 1, TRUE),   -- Bundesliga
(61, 'bet365', 1, TRUE),   -- Ligue 1
(239, 'bet365', 1, TRUE),  -- Liga BetPlay
(253, 'bet365', 1, TRUE)   -- MLS
ON CONFLICT (league_id, bookmaker) DO NOTHING;

-- 3) Índices adicionales en tablas existentes (IF NOT EXISTS)
CREATE INDEX IF NOT EXISTS idx_matches_date_status ON matches(date, status);
CREATE INDEX IF NOT EXISTS idx_predictions_match_id ON predictions(match_id);
CREATE INDEX IF NOT EXISTS idx_shadow_predictions_match_id ON shadow_predictions(match_id);
CREATE INDEX IF NOT EXISTS idx_sensor_predictions_match_id ON sensor_predictions(match_id);

-- 4) VIEW v_daily_match_comparison
CREATE OR REPLACE VIEW v_daily_match_comparison AS
SELECT
    -- Match info
    m.id AS match_id,
    m.date AS kickoff_utc,
    (m.date AT TIME ZONE 'UTC' AT TIME ZONE 'America/Los_Angeles')::date AS match_day_la,
    m.league_id,
    m.status,
    ht.name AS home_team,
    at.name AS away_team,

    -- Actual outcome
    m.home_goals,
    m.away_goals,
    CASE
        WHEN m.home_goals > m.away_goals THEN 'home'
        WHEN m.home_goals = m.away_goals THEN 'draw'
        WHEN m.home_goals < m.away_goals THEN 'away'
    END AS actual_outcome,

    -- Model A (frozen prediction)
    p.home_prob AS a_home_prob,
    p.draw_prob AS a_draw_prob,
    p.away_prob AS a_away_prob,
    CASE
        WHEN p.home_prob >= p.draw_prob AND p.home_prob >= p.away_prob THEN 'home'
        WHEN p.draw_prob >= p.home_prob AND p.draw_prob >= p.away_prob THEN 'draw'
        ELSE 'away'
    END AS a_pick,
    p.model_version AS a_version,
    p.is_frozen AS a_is_frozen,

    -- Shadow (two-stage)
    sp.shadow_home_prob,
    sp.shadow_draw_prob,
    sp.shadow_away_prob,
    sp.shadow_predicted AS shadow_pick,
    sp.shadow_version,
    sp.shadow_architecture,

    -- Sensor B
    sen.b_home_prob AS sensor_home_prob,
    sen.b_draw_prob AS sensor_draw_prob,
    sen.b_away_prob AS sensor_away_prob,
    sen.b_pick AS sensor_pick,
    sen.model_b_version AS sensor_version,
    sen.sensor_state,

    -- Market (primary bookmaker)
    mos.bookmaker AS market_bookmaker,
    mos.odds_home AS market_odds_home,
    mos.odds_draw AS market_odds_draw,
    mos.odds_away AS market_odds_away,
    mos.implied_home AS market_implied_home,
    mos.implied_draw AS market_implied_draw,
    mos.implied_away AS market_implied_away,
    mos.market_pick

FROM matches m
JOIN teams ht ON ht.id = m.home_team_id
JOIN teams at ON at.id = m.away_team_id
LEFT JOIN predictions p ON p.match_id = m.id
LEFT JOIN shadow_predictions sp ON sp.match_id = m.id
LEFT JOIN sensor_predictions sen ON sen.match_id = m.id
LEFT JOIN match_odds_snapshot mos ON mos.match_id = m.id AND mos.is_primary = TRUE

WHERE m.status IN ('FT', 'AET', 'PEN');

-- 5) Migrar datos históricos de predictions.frozen_odds_* a match_odds_snapshot
INSERT INTO match_odds_snapshot (match_id, bookmaker, odds_home, odds_draw, odds_away,
                                  implied_home, implied_draw, implied_away, market_pick,
                                  snapshot_at, is_primary)
SELECT
    p.match_id,
    'bet365' AS bookmaker,
    p.frozen_odds_home,
    p.frozen_odds_draw,
    p.frozen_odds_away,
    -- Normalizar implied probs (remover vig)
    CASE WHEN (1/p.frozen_odds_home + 1/p.frozen_odds_draw + 1/p.frozen_odds_away) > 0
         THEN (1/p.frozen_odds_home) / (1/p.frozen_odds_home + 1/p.frozen_odds_draw + 1/p.frozen_odds_away)
         ELSE NULL END,
    CASE WHEN (1/p.frozen_odds_home + 1/p.frozen_odds_draw + 1/p.frozen_odds_away) > 0
         THEN (1/p.frozen_odds_draw) / (1/p.frozen_odds_home + 1/p.frozen_odds_draw + 1/p.frozen_odds_away)
         ELSE NULL END,
    CASE WHEN (1/p.frozen_odds_home + 1/p.frozen_odds_draw + 1/p.frozen_odds_away) > 0
         THEN (1/p.frozen_odds_away) / (1/p.frozen_odds_home + 1/p.frozen_odds_draw + 1/p.frozen_odds_away)
         ELSE NULL END,
    CASE
        WHEN 1/p.frozen_odds_home >= 1/p.frozen_odds_draw AND 1/p.frozen_odds_home >= 1/p.frozen_odds_away THEN 'home'
        WHEN 1/p.frozen_odds_draw >= 1/p.frozen_odds_home AND 1/p.frozen_odds_draw >= 1/p.frozen_odds_away THEN 'draw'
        ELSE 'away'
    END,
    p.frozen_at,
    TRUE
FROM predictions p
WHERE p.frozen_odds_home IS NOT NULL
  AND p.frozen_odds_draw IS NOT NULL
  AND p.frozen_odds_away IS NOT NULL
  AND p.frozen_odds_home > 0
  AND p.frozen_odds_draw > 0
  AND p.frozen_odds_away > 0
ON CONFLICT (match_id, bookmaker) DO NOTHING;

-- Verificación
SELECT 'Tablas creadas' AS status;
SELECT COUNT(*) AS match_odds_snapshot_rows FROM match_odds_snapshot;
SELECT COUNT(*) AS league_bookmaker_config_rows FROM league_bookmaker_config;
