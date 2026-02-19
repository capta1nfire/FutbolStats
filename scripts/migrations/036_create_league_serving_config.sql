-- Migration 036: Create league_serving_config table (SSOT de Serving por Liga)
-- Part of: SSOT + Auto-Lab Online initiative
-- Date: 2026-02-19
--
-- This table is the Single Source of Truth for per-league prediction serving strategy.
-- The prediction pipeline reads this config instead of using hardcoded constants.
--
-- NOTE: Two-Stage (twostage) NEUTRALIZED on 2026-02-19 due to semantic swap
-- (home/away inverted in v1.0.2-twostage-w3). All TS leagues set to baseline
-- until model is re-trained. Seed data below reflects neutralized state.

BEGIN;

CREATE TABLE IF NOT EXISTS league_serving_config (
    league_id           INTEGER PRIMARY KEY REFERENCES admin_leagues(league_id),
    preferred_strategy  TEXT NOT NULL DEFAULT 'baseline'
                        CHECK (preferred_strategy IN ('baseline', 'twostage', 'family_s')),
    anchor_alpha        FLOAT NOT NULL DEFAULT 0.0
                        CHECK (anchor_alpha >= 0.0 AND anchor_alpha <= 1.0),
    model_version       TEXT,
    prerequisites       JSONB DEFAULT '{}',
    fallback_strategy   TEXT NOT NULL DEFAULT 'baseline'
                        CHECK (fallback_strategy IN ('baseline', 'twostage', 'family_s')),
    notes               TEXT,
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_by          TEXT NOT NULL DEFAULT 'manual'
);

-- Seed data: neutralized state (no twostage until semantic swap fixed)
-- family_s: TIER_3 leagues (5) — all fallback=baseline (TS neutralized)
-- baseline: all other leagues

INSERT INTO league_serving_config (league_id, preferred_strategy, anchor_alpha, model_version, fallback_strategy, notes, updated_by)
VALUES
    -- Tier 1 (Big 5) — baseline (TS neutralized)
    (39,  'baseline',  0.0, NULL, 'baseline', 'W3 PASS, T1 EPL [NEUTRALIZED: TS semantic swap]', 'migration'),
    (61,  'baseline',  0.0, NULL, 'baseline', 'W3 PASS, T1 Ligue 1 [NEUTRALIZED: TS semantic swap]', 'migration'),
    (78,  'baseline',  0.0, NULL, 'baseline', 'W3 PASS, T1 Bundesliga [NEUTRALIZED: TS semantic swap]', 'migration'),
    (140, 'baseline',  0.0, NULL, 'baseline', 'W3 PASS, T1 La Liga [NEUTRALIZED: TS semantic swap]', 'migration'),

    -- Tier 1 but OS_LEAGUE (Serie A)
    (135, 'baseline',  0.0, NULL, 'baseline', 'OS_LEAGUE (OS winner), T1 Serie A', 'migration'),

    -- TIER_3 (Family S primary) — all fallback=baseline (TS neutralized)
    (88,  'family_s',  0.0, NULL, 'baseline', 'TIER_3, Family S primary; fallback=baseline (TS neutralized), Eredivisie', 'migration'),
    (94,  'family_s',  0.0, NULL, 'baseline', 'TIER_3, Family S primary; fallback=baseline (TS neutralized), Primeira Liga', 'migration'),
    (203, 'family_s',  0.0, NULL, 'baseline', 'TIER_3, Family S primary; fallback=baseline (TS neutralized), Süper Lig', 'migration'),
    (265, 'family_s',  0.0, NULL, 'baseline', 'TIER_3, Family S primary; fallback=baseline (TS neutralized), Chile', 'migration'),
    (144, 'family_s',  0.0, NULL, 'baseline', 'TIER_3, Family S primary; fallback=baseline (W3 FAIL), Belgium', 'migration'),

    -- W3-PASS leagues — baseline (TS neutralized)
    (239, 'baseline',  0.0, NULL, 'baseline', 'W3 PASS, Colombia [NEUTRALIZED: TS semantic swap]', 'migration'),
    (253, 'baseline',  0.0, NULL, 'baseline', 'W3 PASS, MLS [NEUTRALIZED: TS semantic swap]', 'migration'),
    (262, 'baseline',  0.0, NULL, 'baseline', 'W3 PASS, México [NEUTRALIZED: TS semantic swap]', 'migration'),
    (268, 'baseline',  0.0, NULL, 'baseline', 'W3 PASS, Uruguay [NEUTRALIZED: TS semantic swap]', 'migration'),
    (299, 'baseline',  0.0, NULL, 'baseline', 'W3 PASS, Venezuela [NEUTRALIZED: TS semantic swap]', 'migration'),
    (307, 'baseline',  0.0, NULL, 'baseline', 'W3 PASS, Saudi Pro [NEUTRALIZED: TS semantic swap]', 'migration'),
    (344, 'baseline',  0.0, NULL, 'baseline', 'W3 PASS, Bolivia [NEUTRALIZED: TS semantic swap]', 'migration'),

    -- Baseline leagues (W3 FAIL or OS winners)
    (128, 'baseline',  1.0, NULL, 'baseline', '[P0-D] Market eficiente, α=1.0 safety net, Argentina', 'migration'),
    (40,  'baseline',  0.0, NULL, 'baseline', 'W3 FAIL (Δ=+0.002), Championship', 'migration'),
    (71,  'baseline',  0.0, NULL, 'baseline', 'W3 FAIL (CI=+0.006), Brazil', 'migration'),
    (242, 'baseline',  0.0, NULL, 'baseline', 'OS_LEAGUE (OS winner), Ecuador', 'migration'),
    (250, 'baseline',  0.0, NULL, 'baseline', 'OS_LEAGUE, Paraguay', 'migration'),
    (281, 'baseline',  0.0, NULL, 'baseline', 'W3 FAIL (CI=+0.006), Perú', 'migration')
ON CONFLICT (league_id) DO NOTHING;

COMMIT;
