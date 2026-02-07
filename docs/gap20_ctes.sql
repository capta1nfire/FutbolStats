-- ============================================================
-- GAP20 Analysis CTEs — Reproducible Queries
-- ============================================================
-- READ-ONLY / NO PRODUCCIÓN
-- Date: 2026-02-07
-- Verdict: NO-GO (see scripts/sensitivity_gap20.py for details)
-- Gates for re-run: v1.0.1-league-only N(FT)>=300, N(SFAV)>=30
-- ============================================================
--
-- CONTEXTO
-- --------
-- Origen: Investigación del partido Genoa vs Napoli (match_id=6354,
--   2025-12-21). El modelo asignó 47.4% a Genoa (Home) mientras el
--   mercado favorecía a Napoli (Away) con 48.0%. Gap=26.6pp.
--   Napoli ganó 3-2. Caso textbook de STRONG_FAV_DISAGREE (SFAV).
--
-- Pregunta: ¿Conviene suprimir o penalizar predicciones donde el
--   modelo diverge fuertemente del mercado (GAP>=20pp)?
--
-- Análisis: Sobre 740 predicciones v1.0.0 con frozen_odds válidos:
--   - 69 casos SFAV identificados
--   - ROI aparente +10.86% pero CI95% bootstrap [-39%, +66%]
--   - CLV(SFAV) = -0.728% (proporcional), -0.862% (power devig)
--   - Leave-one-out: 1 match mueve ROI de +10.86% a -3.95%
--
-- Decisión ATI (2026-02-07): NO-GO para guardrail de supresión.
--   Implementar metadata diagnóstica vía VIEW (no ALTER TABLE):
--   public.predictions_gap20_diag_v1 (mismas definiciones que CTE 2).
--
-- Artefactos relacionados:
--   scripts/sensitivity_gap20.py  — Sensitivity analysis (power devig + allow-ev-neg)
--   VIEW predictions_gap20_diag_v1 — Metadata diagnóstica persistente (creada 2026-02-07)
-- ============================================================


-- ============================================================
-- CTE 1: Full dataset (input for sensitivity_gap20.py)
-- ============================================================
-- Returns: 740 rows (v1.0.0, FT, valid frozen_odds)
-- Includes T5 closing odds for CLV calculation
-- Usage: Export to JSON, feed to scripts/sensitivity_gap20.py

WITH base AS (
  SELECT
    p.match_id,
    m.home_goals, m.away_goals,
    p.frozen_odds_home AS oh, p.frozen_odds_draw AS od, p.frozen_odds_away AS oa,
    -- Model probs (renormalized to sum=1)
    p.home_prob/(p.home_prob+p.draw_prob+p.away_prob) AS m_h,
    p.draw_prob/(p.home_prob+p.draw_prob+p.away_prob) AS m_d,
    p.away_prob/(p.home_prob+p.draw_prob+p.away_prob) AS m_a,
    -- Market probs (proportional devig: implied/Σimplied)
    (1.0/p.frozen_odds_home)/((1.0/p.frozen_odds_home)+(1.0/p.frozen_odds_draw)+(1.0/p.frozen_odds_away)) AS mkt_h,
    (1.0/p.frozen_odds_draw)/((1.0/p.frozen_odds_home)+(1.0/p.frozen_odds_draw)+(1.0/p.frozen_odds_away)) AS mkt_d,
    (1.0/p.frozen_odds_away)/((1.0/p.frozen_odds_home)+(1.0/p.frozen_odds_draw)+(1.0/p.frozen_odds_away)) AS mkt_a,
    -- Actual result (0=Home, 1=Draw, 2=Away)
    CASE WHEN m.home_goals > m.away_goals THEN 0
         WHEN m.home_goals = m.away_goals THEN 1
         ELSE 2 END AS result,
    -- T5 closing odds (for CLV)
    mms.odds_home AS t5_oh, mms.odds_draw AS t5_od, mms.odds_away AS t5_oa
  FROM predictions p
  JOIN matches m ON p.match_id = m.id
  LEFT JOIN market_movement_snapshots mms
    ON p.match_id = mms.match_id AND mms.snapshot_type = 'T5'
  WHERE p.model_version = 'v1.0.0'  -- Change to 'v1.0.1-league-only' for re-run
    AND m.status = 'FT'
    AND p.frozen_odds_home IS NOT NULL
    AND p.frozen_odds_home > 1.0
    AND p.frozen_odds_draw > 1.0
    AND p.frozen_odds_away > 1.0
)
SELECT
  match_id, home_goals, away_goals, oh, od, oa,
  m_h, m_d, m_a, mkt_h, mkt_d, mkt_a, result,
  t5_oh, t5_od, t5_oa
FROM base
ORDER BY match_id;


-- ============================================================
-- CTE 2: SFAV subset (GAP20_DISAGREE_STRONGFAV)
-- ============================================================
-- Definitions:
--   model_fav  = argmax(p_model_H, p_model_D, p_model_A)
--   market_fav = argmax(p_mkt_devigged_H, p_mkt_devigged_D, p_mkt_devigged_A)
--   gap_on_model_fav = p_model[model_fav] - p_mkt[model_fav]
--   SFAV = model_fav != market_fav AND gap >= 0.20 AND max(p_mkt) >= 0.45
-- Returns: ~69 rows (proportional devig)

WITH base AS (
  SELECT
    p.match_id,
    m.home_goals, m.away_goals,
    p.frozen_odds_home AS oh, p.frozen_odds_draw AS od, p.frozen_odds_away AS oa,
    p.home_prob/(p.home_prob+p.draw_prob+p.away_prob) AS m_h,
    p.draw_prob/(p.home_prob+p.draw_prob+p.away_prob) AS m_d,
    p.away_prob/(p.home_prob+p.draw_prob+p.away_prob) AS m_a,
    (1.0/p.frozen_odds_home)/((1.0/p.frozen_odds_home)+(1.0/p.frozen_odds_draw)+(1.0/p.frozen_odds_away)) AS mkt_h,
    (1.0/p.frozen_odds_draw)/((1.0/p.frozen_odds_home)+(1.0/p.frozen_odds_draw)+(1.0/p.frozen_odds_away)) AS mkt_d,
    (1.0/p.frozen_odds_away)/((1.0/p.frozen_odds_home)+(1.0/p.frozen_odds_draw)+(1.0/p.frozen_odds_away)) AS mkt_a,
    CASE WHEN m.home_goals > m.away_goals THEN 0
         WHEN m.home_goals = m.away_goals THEN 1
         ELSE 2 END AS result
  FROM predictions p
  JOIN matches m ON p.match_id = m.id
  WHERE p.model_version = 'v1.0.0'
    AND m.status = 'FT'
    AND p.frozen_odds_home IS NOT NULL
    AND p.frozen_odds_home > 1.0 AND p.frozen_odds_draw > 1.0 AND p.frozen_odds_away > 1.0
),
classified AS (
  SELECT *,
    CASE WHEN m_h >= m_d AND m_h >= m_a THEN 0
         WHEN m_d >= m_h AND m_d >= m_a THEN 1
         ELSE 2 END AS model_fav,
    CASE WHEN mkt_h >= mkt_d AND mkt_h >= mkt_a THEN 0
         WHEN mkt_d >= mkt_h AND mkt_d >= mkt_a THEN 1
         ELSE 2 END AS market_fav,
    GREATEST(mkt_h, mkt_d, mkt_a) AS market_fav_prob
  FROM base
),
with_gap AS (
  SELECT *,
    CASE WHEN model_fav = 0 THEN m_h - mkt_h
         WHEN model_fav = 1 THEN m_d - mkt_d
         ELSE m_a - mkt_a END AS gap_on_mf
  FROM classified
)
SELECT *
FROM with_gap
WHERE model_fav != market_fav          -- DISAGREE
  AND gap_on_mf >= 0.20                -- GAP >= 20pp
  AND market_fav_prob >= 0.45;         -- STRONGFAV (market confident)


-- ============================================================
-- CTE 3: CLV calculation for SFAV bets vs T5 closing odds
-- ============================================================
-- CLV = (close_devigged_prob[best_idx] / open_devigged_prob[best_idx]) - 1
-- Bet policy: argmax(edge), edge >= 0.05, EV > 0
-- T5 source: market_movement_snapshots WHERE snapshot_type = 'T5'
-- Returns: ~53 rows (SFAV bets with T5 data available)

WITH base AS (
  SELECT
    p.match_id,
    m.home_goals, m.away_goals,
    p.frozen_odds_home AS oh, p.frozen_odds_draw AS od, p.frozen_odds_away AS oa,
    p.home_prob/(p.home_prob+p.draw_prob+p.away_prob) AS m_h,
    p.draw_prob/(p.home_prob+p.draw_prob+p.away_prob) AS m_d,
    p.away_prob/(p.home_prob+p.draw_prob+p.away_prob) AS m_a,
    (1.0/p.frozen_odds_home)/((1.0/p.frozen_odds_home)+(1.0/p.frozen_odds_draw)+(1.0/p.frozen_odds_away)) AS mkt_h,
    (1.0/p.frozen_odds_draw)/((1.0/p.frozen_odds_home)+(1.0/p.frozen_odds_draw)+(1.0/p.frozen_odds_away)) AS mkt_d,
    (1.0/p.frozen_odds_away)/((1.0/p.frozen_odds_home)+(1.0/p.frozen_odds_draw)+(1.0/p.frozen_odds_away)) AS mkt_a,
    CASE WHEN m.home_goals > m.away_goals THEN 0
         WHEN m.home_goals = m.away_goals THEN 1
         ELSE 2 END AS result
  FROM predictions p
  JOIN matches m ON p.match_id = m.id
  WHERE p.model_version = 'v1.0.0'
    AND m.status = 'FT'
    AND p.frozen_odds_home IS NOT NULL
    AND p.frozen_odds_home > 1.0 AND p.frozen_odds_draw > 1.0 AND p.frozen_odds_away > 1.0
),
classified AS (
  SELECT *,
    CASE WHEN m_h >= m_d AND m_h >= m_a THEN 0
         WHEN m_d >= m_h AND m_d >= m_a THEN 1
         ELSE 2 END AS model_fav,
    CASE WHEN mkt_h >= mkt_d AND mkt_h >= mkt_a THEN 0
         WHEN mkt_d >= mkt_h AND mkt_d >= mkt_a THEN 1
         ELSE 2 END AS market_fav,
    GREATEST(mkt_h, mkt_d, mkt_a) AS market_fav_prob
  FROM base
),
with_gap AS (
  SELECT *,
    CASE WHEN model_fav = 0 THEN m_h - mkt_h
         WHEN model_fav = 1 THEN m_d - mkt_d
         ELSE m_a - mkt_a END AS gap_on_mf
  FROM classified
),
sfav AS (
  SELECT * FROM with_gap
  WHERE model_fav != market_fav AND gap_on_mf >= 0.20 AND market_fav_prob >= 0.45
),
with_bet AS (
  SELECT *,
    -- Edges per outcome
    m_h - mkt_h AS edge_h,
    m_d - mkt_d AS edge_d,
    m_a - mkt_a AS edge_a,
    -- Best outcome = argmax(edge) with edge >= 0.05
    CASE
      WHEN GREATEST(m_h - mkt_h, m_d - mkt_d, m_a - mkt_a) < 0.05 THEN -1
      WHEN m_h - mkt_h >= m_d - mkt_d AND m_h - mkt_h >= m_a - mkt_a THEN 0
      WHEN m_d - mkt_d >= m_h - mkt_h AND m_d - mkt_d >= m_a - mkt_a THEN 1
      ELSE 2
    END AS best_idx
  FROM sfav
),
bettable AS (
  SELECT * FROM with_bet
  WHERE best_idx >= 0
    AND CASE WHEN best_idx = 0 THEN m_h * oh
             WHEN best_idx = 1 THEN m_d * od
             ELSE m_a * oa END > 1.0  -- EV > 0
),
clv_calc AS (
  SELECT
    b.match_id,
    b.best_idx,
    -- Open devigged prob (at bet time, proportional)
    CASE WHEN b.best_idx = 0 THEN b.mkt_h
         WHEN b.best_idx = 1 THEN b.mkt_d
         ELSE b.mkt_a END AS open_prob,
    -- Close devigged prob (T5, proportional)
    CASE WHEN b.best_idx = 0 THEN
      (1.0/mms.odds_home)/((1.0/mms.odds_home)+(1.0/mms.odds_draw)+(1.0/mms.odds_away))
         WHEN b.best_idx = 1 THEN
      (1.0/mms.odds_draw)/((1.0/mms.odds_home)+(1.0/mms.odds_draw)+(1.0/mms.odds_away))
         ELSE
      (1.0/mms.odds_away)/((1.0/mms.odds_home)+(1.0/mms.odds_draw)+(1.0/mms.odds_away))
    END AS close_prob
  FROM bettable b
  JOIN market_movement_snapshots mms
    ON b.match_id = mms.match_id AND mms.snapshot_type = 'T5'
  WHERE mms.odds_home > 1.0 AND mms.odds_draw > 1.0 AND mms.odds_away > 1.0
)
SELECT
  match_id,
  best_idx,
  open_prob,
  close_prob,
  (close_prob / open_prob) - 1.0 AS clv
FROM clv_calc;
