# Manual de Segmentacion por Liga

**Documento vivo** — Se actualiza cada vez que una liga pasa por el laboratorio.
El futbol es dinamico: las combinaciones optimas cambian con el tiempo. Re-ejecutar pruebas periodicamente.

**Script**: `scripts/feature_lab.py`
**Modos**: `--shap`, `--optuna`, `--extract`, `--league <ID>`, `--min-date YYYY-MM-DD`

---

## Indice de Ligas

| Liga | ID | Estado | Ultima prueba | Seccion |
|------|----|--------|---------------|---------|
| Argentina Primera Division | 128 | COMPLETO | 2026-02-10 | [1](#1-argentina-128) |
| Colombia Primera A | 239 | COMPLETO | 2026-02-10 | [2](#2-colombia-239) |
| Ecuador Liga Pro | 242 | COMPLETO | 2026-02-11 | [3](#3-ecuador-242) |
| Venezuela Liga FUTVE | 299 | COMPLETO | 2026-02-11 | [4](#4-venezuela-299) |
| Peru Liga 1 | 281 | COMPLETO | 2026-02-11 | [5](#5-peru-281) |
| Bolivia Division Profesional | 344 | COMPLETO | 2026-02-11 | [6](#6-bolivia-344) |
| Chile Primera Division | 265 | COMPLETO | 2026-02-11 | [7](#7-chile-265) |

---

## 1. Argentina (128)

### 1.1 Ficha Tecnica

| Campo | Valor |
|-------|-------|
| **League ID** | 128 |
| **Nombre** | Liga Profesional de Futbol (Argentina Primera Division) |
| **Formato** | Split-season: Apertura (Feb-Jun) + Clausura (Jul-Dic) + playoffs |
| **Equipos** | 28 (desde 2022) |
| **N partidos en DB** | 2,526 (Oct 2020 — Feb 2026) |
| **N partidos testeados** | 1,319 (filtro 2023-01-01+) |
| **Train / Test split** | 1,055 / 264 (split date: 2025-08-08) |
| **Distribucion resultados** | Home 43.6%, Draw 31.9%, Away 24.5% |
| **Brier naive** | 0.6481 (predictor marginal) |
| **Dificultad** | ALTA — segunda tasa de empates mas alta (31.9%) |
| **Odds coverage** | 100% desde 2023 (FDUK backfill, Pinnacle closing) |
| **xG coverage** | 99.4% desde 2023 (FotMob/Opta, 1,314 matches) |
| **Market Anchor** | Activo, alpha=1.0 (replica mercado) |

### 1.2 Campeon: O7 / S7 — ABE + Elo (18 features)

```
Features (18):
  opp_att_home, opp_def_home, opp_att_away, opp_def_away, opp_rating_diff,
  overperf_home, overperf_away, overperf_diff,
  draw_tendency_home, draw_tendency_away, draw_elo_interaction, league_draw_rate,
  home_bias_home, home_bias_away, home_bias_diff,
  elo_home, elo_away, elo_diff

Hyperparams Optuna (O7):
  max_depth: 2
  learning_rate: 0.0229
  n_estimators: 53
  min_child_weight: 7
  subsample: 0.614
  colsample_bytree: 0.811
  reg_alpha: 0.00038
  reg_lambda: 0.0052

Brier (test, N=264): 0.65978 ± 0.00156
Brier CV (3-fold temporal): 0.64629
CI95: [0.64752, 0.67406]
Accuracy: 38.6%
```

### 1.3 Ranking Completo (Optuna, 2023+)

| # | Test | Features | Brier test | CV Brier | Depth | LR | N_est | Acc |
|---|------|----------|-----------|----------|-------|----|-------|-----|
| 1 | **OF_abe_elo_odds** | 21 | **0.6597** | **0.6380** | 2 | 0.031 | 64 | 42.0% |
| 2 | **O7_all_abe_elo** | 18 | **0.6598** | 0.6463 | 2 | 0.023 | 53 | 38.6% |
| 3 | OC_xg_all_elo_odds | 15 | 0.6605 | 0.6421 | 2 | 0.017 | 74 | 41.0% |
| 4 | O0_elo_gw_defense | 5 | 0.6614 | 0.6496 | 2 | 0.015 | 99 | 35.5% |
| 5 | O8_smart_minimal | 13 | 0.6619 | 0.6495 | 2 | 0.016 | 93 | 36.7% |
| 6 | OD_xg_overperf_elo | 12 | 0.6627 | 0.6475 | 2 | 0.012 | 109 | 38.1% |
| 7 | O9_baseline_17 | 17 | 0.6632 | 0.6482 | 2 | 0.018 | 84 | 36.9% |
| 8 | O2_defense_form_elo | 8 | 0.6631 | 0.6496 | 3 | 0.013 | 100 | 35.8% |
| 9 | OE_xg_defense_odds | 11 | 0.6631 | 0.6395 | 2 | 0.017 | 106 | 41.7% |
| 10 | O6_efficiency_elo | 8 | 0.6637 | 0.6531 | 4 | 0.017 | 64 | 35.8% |
| 11 | OB_xg_odds | 6 | 0.6641 | 0.6410 | 3 | 0.022 | 70 | 41.6% |
| 12 | O1_elo_gw_form | 6 | 0.6643 | 0.6492 | 2 | 0.022 | 64 | 35.1% |
| 13 | O5_m2_interactions | 9 | 0.6657 | 0.6453 | 3 | 0.011 | 150 | 35.4% |
| 14 | O4_defense_elo_kimi | 14 | 0.6665 | 0.6510 | 4 | 0.015 | 95 | 34.4% |
| 15 | O3_elo_k20 | 3 | 0.6677 | 0.6532 | 3 | 0.018 | 69 | 36.1% |
| 16 | OA_only_elo | 3 | 0.6684 | 0.6511 | 5 | 0.012 | 87 | 35.4% |
| — | FIXED_baseline (prod) | 17 | 0.6698 | — | — | — | — | 35.8% |
| — | **MKT_market** | 3 | **0.6489** | — | — | — | — | — |

### 1.4 SHAP Analysis (2023+)

| Test | Brier | #1 Feature (SHAP) | #2 Feature | #3 Feature |
|------|-------|-------------------|------------|------------|
| S7_abe_elo | 0.6608 | league_draw_rate (0.104) | elo_diff (0.051) | overperf_diff (0.041) |
| S8_abe_elo_odds | 0.6644 | league_draw_rate (0.078) | odds_away (0.075) | odds_home (0.063) |
| S0_baseline_17 | 0.6742 | home_matches_played (0.060) | home_goals_conceded_avg (0.054) | abs_strength_gap (0.053) |
| S1_baseline_odds | 0.6697 | home_matches_played (0.075) | odds_away (0.066) | odds_home (0.065) |
| S3_defense_elo | 0.6739 | elo_diff (0.086) | home_goals_conceded_avg (0.069) | elo_home (0.056) |
| S5_xg_elo | 0.6790 | elo_diff (0.077) | home_xg_for_avg (0.048) | xg_diff (0.047) |

**Hallazgos SHAP clave:**
- `league_draw_rate` domina con 0.104 de SHAP global — captura la alta tasa de empates de Argentina (31.9%)
- `elo_diff` es la segunda feature mas importante (0.051)
- Odds aportan 29.9% del SHAP share en S8 pero NO mejoran el Brier vs S7 sin odds
- Draw-class top drivers: `league_draw_rate`, `overperf_diff`, `home_bias_home`

### 1.5 Que Funciona y Que No

| Categoria | Funciona | No Funciona |
|-----------|----------|-------------|
| **Elo** | SI — elo_diff es top-2 en todos los tests | elo_k20 no mejora vs k32 |
| **ABE features** | SI — opp_adj + overperf + draw_aware + home_bias | — |
| **Odds como feature** | NEUTRAL — no mejoran sobre ABE+Elo (S8 0.6644 > S7 0.6608) | Odds estaticas compiten con el modelo en vez de complementarlo |
| **Odds como anchor** | SI — Market Anchor alpha=1.0 replica mercado (0.6489) | — |
| **xG rolling** | NO — neutral en todas las pruebas (S5: 0.6790 peor que S3: 0.6739) | xG promedio no agrega sobre goals promedio |
| **Form (win_rate, streaks)** | DEBIL — mejora marginal en O1 pero no en top-3 | form_diff es ruidoso |
| **Interactions (elo_x_season, etc.)** | DEBIL — O5 es #13, no top-5 | Alto riesgo overfitting |
| **Defense (goals_conceded)** | MODERADO — O0 es #4 con solo 5 features | Util como componente, no como set principal |
| **Efficiency (finish_eff, def_eff)** | DEBIL — O6 es #10 | Mas util en Liga MX |
| **rest_days** | NO — ruido universal, negativo en 12/21 ligas | Eliminar en v1.0.2 |
| **Optuna tuning** | SI — mejora 0.15% vs fixed params | depth=2 es optimo (12/16 tests) |
| **Depth > 2** | NO — depth=3+ tiende a overfitting en Argentina | Solo 4/16 tests eligieron depth>2 |

### 1.6 Hyperparams Optimos (consensus)

De 16 tests Optuna, el patron claro es:

| Param | Rango optimo | Mediana | Nota |
|-------|-------------|---------|------|
| max_depth | **2** | 2 | 12/16 tests eligieron 2 |
| learning_rate | 0.012 — 0.031 | 0.017 | Bajo = mas conservador |
| n_estimators | 53 — 109 | 84 | Ajustar inversamente a lr |
| min_child_weight | 5 — 12 | 8 | Alta regularizacion |
| subsample | 0.56 — 0.78 | 0.65 | Bagging moderado |
| colsample_bytree | 0.55 — 0.85 | 0.77 | Feature sampling moderado |
| reg_alpha | 0.0002 — 0.16 | 0.003 | L1 bajo |
| reg_lambda | 0.0003 — 0.05 | 0.005 | L2 bajo |

**Regla para Argentina: arboles SIMPLES (depth=2) con regularizacion ALTA (mcw>=7).**

### 1.7 Gap vs Mercado

```
Mejor modelo (OF/O7 Optuna):  0.6597
Mercado (Pinnacle closing):   0.6489
Gap:                          +1.7% (modelo pierde)

Market Anchor (alpha=1.0):    ≈0.6489 (replica mercado)
Resultado final al usuario:   ≈0.6489
```

**El mercado gana por informacion que no esta en features historicas:**
- Alineaciones confirmadas (~1h antes del KO)
- Lesiones del dia
- Contexto deportivo (urgencia, clasicos, motivacion)
- Volumen de apuestas y movimiento de lineas

### 1.8 Estrategia Activa de Produccion

| Componente | Estado | Detalle |
|------------|--------|---------|
| Modelo base | v1.0.1 (17 features baseline) | Pendiente upgrade a v1.0.2 |
| Feature set objetivo | ABE+Elo (18 features) | O7/S7 campeon |
| Hyperparams objetivo | depth=2, lr=0.023, n_est=53 | Optuna Argentina 2023+ |
| Market Anchor | ACTIVO, alpha=1.0 | Replica mercado para Argentina |
| Shadow mode | ACTIVO | Two-stage v1.1.0 en evaluacion |
| xG source | FotMob (Opta) | 1,314 matches, jobs activos |
| Odds source | FDUK (Pinnacle closing) + odds_sync (pre-KO) | 100% desde 2023 |

### 1.9 Proximo Paso: v1.0.2

1. Cambiar FEATURE_COLUMNS de 17 baseline a 18 ABE+Elo
2. Aplicar hyperparams Optuna (depth=2, lr=0.023, n_est=53)
3. Retrain en 2023+ (21K matches, 25 ligas)
4. Validar via shadow recalibration pipeline
5. Market Anchor sigue activo como safety net

### 1.10 Historial de Pruebas

| Fecha | Prueba | Script | Resultado clave |
|-------|--------|--------|-----------------|
| 2026-02-07 | Feature diagnostic (permutation + ablation) | `feature_diagnostic.py` | 14 SIGNAL, 3 NEUTRAL, goal_diff_avg domina |
| 2026-02-09 | Lab v1: 101 tests (2020+, 46% odds) | `feature_lab.py --league 128` | S7_abe_elo campeon (0.6554), "odds danan" (INCORRECTO) |
| 2026-02-09 | FDUK backfill v4.2.0 | `ingest_football_data_uk.py` | 100% odds desde 2023, bug ext→int corregido |
| 2026-02-10 | SHAP 2023+ (100% odds) | `feature_lab.py --shap --league 128 --min-date 2023-01-01` | S7 campeon (0.6608), odds NO danan, league_draw_rate domina |
| 2026-02-10 | Optuna 2023+ (16 candidatos) | `feature_lab.py --optuna --league 128 --min-date 2023-01-01` | OF/O7 empatados (0.6597), mercado 0.6489, depth=2 optimo |
| 2026-02-10 | SHAP S8 (ABE+Elo+Odds) | `feature_lab.py --shap --league 128 --min-date 2023-01-01` | Odds no mejoran S7, odds_share 29.9% pero Brier peor |

### 1.11 Archivos de Evidencia

| Archivo | Contenido |
|---------|-----------|
| `scripts/output/lab/shap_analysis_128.json` | SHAP 9 tests, Argentina 2023+ |
| `scripts/output/lab/feature_lab_results_optuna.json` | Optuna 16 candidatos, Argentina 2023+ |
| `scripts/output/lab/lab_data_128.csv` | Dataset cacheado (2,526 rows, 115 cols) |
| `scripts/output/feature_diagnostic_argentina_128.json` | Diagnostic detallado per-feature |

---

## 2. Colombia (239)

### 2.1 Ficha Tecnica

| Campo | Valor |
|-------|-------|
| **League ID** | 239 |
| **Nombre** | Liga BetPlay (Colombia Primera A) |
| **Formato** | Split-season: Apertura (Ene-Jun) + Clausura (Jul-Dic) + cuadrangulares |
| **Equipos** | 20 (+ promocion/descenso) |
| **N partidos en DB** | 2,924 (Ene 2019 — Feb 2026) |
| **N partidos testeados** | 2,924 (sin filtro de fecha) |
| **Train / Test split** | 2,339 / 585 (split date: 2024-10-15) |
| **Distribucion resultados** | Home 45.2%, Draw 30.8%, Away 24.0% (global); Test: 46.3% / 29.1% / 24.6% |
| **Brier naive** | ~0.640 (predictor marginal) |
| **Dificultad** | MEDIA — ventaja local mas pronunciada que Argentina, menos empates |
| **Odds coverage** | **82.8% global (2,421/2,924). 2020-2025: 99.6-100%. Fuente: OddsPortal scrape** |
| **Odds coverage 2023+** | **96.3% (1,330/1,381). Solo 2019 y 2026 sin odds** |
| **xG coverage** | 9.3% (272 matches, FotMob/Opta, 2025 Clausura+ solamente) |
| **Market Anchor** | **VIABLE — mercado gana por +2.0%, patron similar a Argentina** |

### 2.2 Campeon: O1 — Elo GW + Form (6 features)

```
Features (6):
  elo_gw_home, elo_gw_away, elo_gw_diff,
  home_win_rate_last5, away_win_rate_last5, form_diff

Hyperparams Optuna (O1):
  max_depth: 2
  learning_rate: 0.022
  n_estimators: 63
  (otros: ver Optuna output)

Brier (test, N=585): 0.62799
Brier CV (3-fold temporal): 0.64603
```

### 2.3 Ranking Completo (Optuna)

| # | Test | Features | Brier test | CV Brier | Depth | LR | N_est |
|---|------|----------|-----------|----------|-------|----|-------|
| 1 | **O1_elo_gw_form** | 6 | **0.6280** | 0.6460 | 2 | 0.022 | 63 |
| 2 | OF_abe_elo_odds | 21 | 0.6291 | 0.6362 | 2 | 0.017 | 61 |
| 3 | O5_m2_interactions | 9 | 0.6294 | 0.6475 | 2 | 0.024 | 61 |
| 4 | O3_elo_k20 | 3 | 0.6297 | 0.6477 | 2 | 0.016 | 91 |
| 5 | O8_smart_minimal | 13 | 0.6303 | 0.6470 | 6 | 0.010 | 130 |
| 6 | O0_elo_gw_defense | 5 | 0.6322 | 0.6467 | 3 | 0.017 | 73 |
| 7 | O2_defense_form_elo | 8 | 0.6325 | 0.6473 | 2 | 0.012 | 114 |
| 8 | OA_only_elo | 3 | 0.6340 | 0.6491 | 3 | 0.018 | 76 |
| 9 | O6_efficiency_elo | 8 | 0.6358 | 0.6495 | 6 | 0.011 | 129 |
| 10 | O7_all_abe_elo | 18 | 0.6364 | 0.6481 | 2 | 0.019 | 79 |
| 11 | O4_defense_elo_kimi | 14 | 0.6398 | 0.6514 | 2 | 0.012 | 79 |
| 12 | O9_baseline_17 | 17 | 0.6422 | 0.6542 | 6 | 0.017 | 56 |
| 13 | OB_xg_odds | 6 | 0.6613 | 0.6358 | 6 | 0.013 | 119 |
| 14 | OE_xg_defense_odds | 11 | 0.6526 | 0.6533 | 2 | 0.017 | 61 |
| 15 | OC_xg_all_elo_odds | 15 | 0.6844 | 0.6310 | 6 | 0.010 | 123 |
| 16 | OD_xg_overperf_elo | 12 | 0.6852 | 0.6385 | 6 | 0.010 | 143 |
| — | FIXED_baseline (prod) | 17 | 0.6326 | — | — | — | — |
| — | **MKT_market** | — | **0.6091** | — | — | — | — |

**Notas:**
- **0 tests SKIPPED** (vs 4 en run anterior). Odds coverage ahora suficiente.
- OF_abe_elo_odds (#2) es el mejor test con odds como feature, CV=0.6362 (mejor CV de todos los non-xG).
- xG tests (OB-OD) siguen overfitteando masivamente (N_test=52, depth=6).
- MKT_market (de-vigged odds) = 0.6091 es el baseline imbatible.

### 2.4 SHAP Analysis

| Test | Brier | #1 Feature (SHAP) | #2 Feature | #3 Feature |
|------|-------|-------------------|------------|------------|
| **S1_baseline_odds** | **0.6245** | odds_away (0.165) | odds_home (0.076) | home_matches_played (0.054) |
| S8_abe_elo_odds | 0.6246 | odds_away (0.143) | odds_home (0.087) | odds_draw (0.044) |
| S2_elo_odds | 0.6268 | odds_away (0.148) | odds_home (0.081) | elo_diff (0.046) |
| S4_m2_interactions | 0.6277 | home_matches_played (0.128) | elo_diff (0.082) | elo_away (0.064) |
| S3_defense_elo | 0.6295 | elo_diff (0.098) | elo_away (0.076) | elo_home (0.056) |
| S0_baseline_17 | 0.6315 | home_matches_played (0.119) | away_matches_played (0.069) | away_goals_conceded_avg (0.055) |
| S6_power_5 | 0.6334 | elo_diff (0.111) | opp_rating_diff (0.079) | draw_elo_interaction (0.044) |
| S7_abe_elo | 0.6355 | elo_diff (0.078) | elo_away (0.048) | opp_rating_diff (0.041) |
| S5_xg_elo | 0.6458 | elo_home (0.158) | elo_diff (0.110) | home_xg_for_avg (0.107) |

**Hallazgos SHAP clave:**
- **Odds dominan cuando estan presentes**: `odds_away` es #1 en S1/S2/S8 (SHAP 0.143-0.165). Odds share = 52% en S8.
- S1_baseline_odds (0.6245) SUPERA al anterior campeon SHAP S4_m2_interactions (0.6277)
- `elo_diff` sigue siendo #1 en tests sin odds (SHAP 0.078-0.111)
- Draw-class drivers: `odds_draw` (0.044 en S8), `draw_elo_interaction` (0.027), `league_draw_rate` (0.025)
- `home_matches_played` sigue fuerte como proxy de ritmo de temporada (0.119-0.128 en S0/S4)

### 2.5 Que Funciona y Que No

| Categoria | Funciona | No Funciona |
|-----------|----------|-------------|
| **Odds como feature** | **SI — DOMINANTE cuando presente. S1 campeon SHAP (0.6245). odds_away SHAP=0.165** | N_test=494 (no full 585) |
| **Odds como anchor** | **VIABLE — mercado Brier 0.6091, gap +2.0%. Market Anchor recomendado** | — |
| **Elo** | SI — DOMINANTE en tests sin odds. elo_diff #1 (SHAP 0.078-0.111) | — |
| **Elo goal-weighted** | SI — O1 campeon (0.6280) con elo_gw + form | — |
| **Elo K=20** | SI — O3 #4 con solo 3 features (0.6297) | No mejora sobre elo_gw |
| **Defense (goals_conceded)** | SI — complemento de Elo. O0 #6 con 5 features | — |
| **ABE + Elo + Odds** | SI — OF #2 Optuna (0.6291), mejor CV (0.6362) | Requiere odds disponibles |
| **Interactions (elo_x_rest)** | MODERADO — O5 #3 (0.6294) | Mayor riesgo con mas features |
| **ABE features (sin odds)** | DEBIL — O7 #10 (0.6364) | 18 features no justificadas sin odds |
| **Baseline 17 features** | NO — O9 #12 (0.6422), peor que FIXED (0.6326) | Overfitting |
| **xG rolling** | INCONCLUSO — N=52, overfitting masivo | Re-evaluar cuando coverage > 50% |
| **rest_days** | NO — ruido confirmado | — |
| **Optuna tuning** | SI — O1: 0.6280 vs FIXED: 0.6326 = mejora 0.7% | — |

### 2.6 Hyperparams Optimos (consensus)

De 16 tests Optuna (excluyendo xG overfitteados):

| Param | Rango optimo | Mediana | Nota |
|-------|-------------|---------|------|
| max_depth | **2-3** | 2 | 8/12 eligieron 2 — consistente con run anterior |
| learning_rate | 0.010 — 0.024 | 0.017 | Mas conservador que run anterior |
| n_estimators | 56 — 130 | 73 | Inversamente proporcional a lr |
| min_child_weight | 3 — 7 | — | Regularizacion moderada |

**Regla para Colombia: arboles SIMPLES (depth=2), pocas features (3-6), Elo como base. Con odds: ABE+Elo+Odds (21 feat) viable por mejor CV.**

### 2.7 Gap vs Mercado

```
Mejor modelo (O1 Optuna):    0.6280
Mercado (de-vigged avg):      0.6091  (N_test=494)
Gap:                          +0.0189 (+3.1%)

Mejor con odds (OF Optuna):  0.6291
FAIR (O1 vs market, N=494):  model=0.6292 market=0.6091 Δ=+0.0201

FIXED_baseline (prod):        0.6326
Mejora O1 vs prod:            -0.7%
```

**Mercado gana por ~2-3%**. Patron identico a Argentina (gap +3.4%). Market Anchor es VIABLE.
Fuente de odds: OddsPortal scrape (2020-2025, avg closing).

### 2.8 Contraste con Argentina

| Dimension | Argentina (128) | Colombia (239) |
|-----------|----------------|----------------|
| **Campeon** | ABE+Elo (18 feat) | Elo GW+Form (6 feat) |
| **Mejor Brier** | 0.6585 | 0.6280 |
| **Brier naive** | 0.6481 | ~0.640 |
| **Mejora vs naive** | +1.6% | +1.9% |
| **Feature dominante** | league_draw_rate | elo_diff / odds_away |
| **ABE features** | ESENCIALES (#1-2) | DEBILES sin odds, UTILES con odds |
| **Empates** | 31.9% (alta) | 30.8% (moderada) |
| **Odds coverage** | 100% (FDUK 2023+) | **96.3% (OddsPortal 2020-2025)** |
| **xG** | 99.4% (FotMob) | 9.3% (FotMob parcial) |
| **Market Brier** | 0.5967 (Pinnacle) | **0.6091 (OddsPortal avg)** |
| **Gap vs mercado** | +3.4% | **+3.1%** |
| **Market Anchor** | Activo, alpha=1.0 | **VIABLE, pendiente activar** |
| **Complejidad optima** | depth=2, 18 features | depth=2, 6 features |

**Explicacion del contraste:**
- Argentina necesita ABE porque su alta tasa de empates (31.9%) requiere features especializadas
- Colombia con odds: ABE+Elo+Odds (OF, 21 feat) es #2 con el mejor CV (0.6362) — odds hacen viable mas features
- Colombia sin odds: modelos minimalistas (3-6 feat) son optimos
- Gap vs mercado casi identico (~3%) — ambas ligas se benefician de Market Anchor
- Market Brier Colombia (0.6091) es peor que Argentina (0.5967) → mercado menos eficiente para Colombia

### 2.9 Estrategia Activa de Produccion

| Componente | Estado | Detalle |
|------------|--------|---------|
| Modelo base | v1.0.1 (17 features baseline) | Pendiente upgrade a v1.0.2 |
| Feature set objetivo | Elo GW + Form (6 features) | O1 campeon Optuna |
| Con odds: alternativa | ABE + Elo + Odds (21 features) | OF #2, mejor CV |
| Hyperparams objetivo | depth=2, lr=0.022, n_est=63 | Optuna Colombia |
| **Market Anchor** | **VIABLE — activar con alpha TBD** | Mercado gana por +2.0%, fuente: OddsPortal |
| Shadow mode | ACTIVO | Two-stage v1.1.0 |
| xG source | FotMob (Opta) | 272 matches (2025 Clausura+), jobs activos |
| Odds source | **OddsPortal scrape (2020-2025) + odds_sync (2026+)** | 96.3% coverage 2023+ |

### 2.10 Proximo Paso

1. **Market Anchor Colombia**: Activar con alpha=TBD basado en gap +3.1%. Backtest primero.
2. **v1.0.2 per-league features**: Implementar feature set por liga (6 feat Colombia vs 18 feat Argentina)
3. **xG re-evaluation**: Cuando coverage > 50% (~mid-2026), re-correr lab con xG features
4. **Odds 2026**: Configurar OddsPortal scrape periodico o Betfair/Pinnacle para 2026+
5. **Re-run lab**: Despues de completar 2026 Apertura (Jun 2026) para actualizar splits

### 2.11 Historial de Pruebas

| Fecha | Prueba | Script | Resultado clave |
|-------|--------|--------|-----------------|
| 2026-02-10 | Lab v1 (101 tests) | `feature_lab.py --league 239` | D8_elo_all campeon (0.6196), 13 odds-tests SKIPPED |
| 2026-02-10 | SHAP v1 (9 tests, 7 ran) | `feature_lab.py --shap --league 239` | S4_m2_interactions campeon (0.6256), 2 SKIPPED |
| 2026-02-10 | Optuna v1 (16 cand, 12 ran) | `feature_lab.py --optuna --league 239` | O0_elo_gw_defense campeon (0.6275), 4 SKIPPED |
| **2026-02-10** | **Lab v2 (101 tests, 0 skip)** | `feature_lab.py --league 239` | **D8_elo_all 0.6197, MKT 0.6091, FAIR gap +1.2%** |
| **2026-02-10** | **SHAP v2 (9 tests, 0 skip)** | `feature_lab.py --shap --league 239` | **S1_baseline_odds campeon (0.6245), odds_away domina** |
| **2026-02-10** | **Optuna v2 (16 cand, 0 skip)** | `feature_lab.py --optuna --league 239` | **O1_elo_gw_form campeon (0.6280), OF #2 con odds** |

### 2.12 Archivos de Evidencia

| Archivo | Contenido |
|---------|-----------|
| `scripts/output/lab/shap_analysis_239.json` | SHAP v2, 9 tests, Colombia (con odds) |
| `scripts/output/lab/feature_lab_results_optuna.json` | Optuna v2, 16 candidatos (con odds) |
| `scripts/output/lab/feature_lab_results.json` | Lab v2, 101 tests (con odds + MKT baseline) |
| `scripts/output/lab/lab_data_239.csv` | Dataset cacheado (2,924 rows) |
| `data/oddsportal_raw/colombia-primera-a_2025_rescrape.json` | OddsPortal scrape 2025 (442 matches) |
| `data/oddsportal_raw/colombia-primera-a_20*.json` | OddsPortal scrapes 2020-2024 |
| `data/oddsportal_team_aliases.json` | Aliases Colombia (83 entries, 26 teams) |

---

## 3. Ecuador (242)

### 3.1 Ficha Tecnica

| Campo | Valor |
|-------|-------|
| **League ID** | 242 |
| **Nombre** | Liga Pro (Ecuador Serie A) |
| **Formato** | Split-season: Fase 1 (Feb-Jun) + Fase 2 (Jul-Nov) + finales |
| **Equipos** | 16 |
| **N partidos en DB** | 1,772 (Feb 2019 — Dic 2025) |
| **N partidos testeados** | 1,772 (sin filtro de fecha) |
| **Train / Test split** | 1,417 / 355 (split date: 2024-10-20) |
| **Distribucion resultados** | Home 42.3%, Draw 28.7%, Away 29.0% (test set) |
| **Brier naive** | ~0.648 (predictor marginal) |
| **Dificultad** | MEDIA — ventaja local fuerte, empates moderados |
| **Odds coverage** | **95.0% global (1,686/1,775). 2023+: 93.6% (744/795). Fuente: OddsPortal scrape** |
| **xG coverage** | **0% — sin fuente de xG disponible** |
| **Market Anchor** | **NO NECESARIO — modelo supera al mercado (FAIR Δ=-0.009)** |

### 3.2 Campeon: K9 / O7 — ABE + Elo (18 features)

```
Features (18):
  opp_att_home, opp_def_home, opp_att_away, opp_def_away, opp_rating_diff,
  overperf_home, overperf_away, overperf_diff,
  draw_tendency_home, draw_tendency_away, draw_elo_interaction, league_draw_rate,
  home_bias_home, home_bias_away, home_bias_diff,
  elo_home, elo_away, elo_diff

Hyperparams Optuna v2 (O7):
  max_depth: 2
  learning_rate: 0.0138
  n_estimators: 250
  min_child_weight: 13

Standard (K9):  Brier 0.63127 ± 0.00134  CI95[0.6091, 0.6568]  Acc: 0.456
Optuna  (O7):   Brier 0.63423 ± 0.00040  CI95[0.6119, 0.6565]  Acc: 0.464
Brier CV (3-fold temporal): 0.61768
```

### 3.3 Ranking Completo (Standard, top-20)

| # | Test | Features | Brier | Acc | CI95 |
|---|------|----------|-------|-----|------|
| 1 | **C5_elo_prob_draw** | 2 | **0.63050** | 0.453 | [0.6089, 0.6530] |
| 2 | **K9_all_abe_elo** | 18 | **0.63127** | 0.456 | [0.6091, 0.6568] |
| 3 | H9_minimal_power | 4 | 0.63316 | 0.427 | [0.6102, 0.6577] |
| 4 | M0_h0_opp_adj | 10 | 0.63398 | 0.437 | [0.6099, 0.6563] |
| 5 | C3_elo_diff_form | 2 | 0.63424 | 0.434 | [0.6111, 0.6587] |
| 6 | M3_h0_draw_aware | 9 | 0.63430 | 0.463 | [0.6097, 0.6575] |
| 7 | M8_power_5 | 5 | 0.63442 | 0.444 | [0.6093, 0.6581] |
| 8 | K1_opp_adj_elo | 8 | 0.63459 | 0.438 | [0.6100, 0.6571] |
| 9 | M7_ultimate | 26 | 0.63500 | 0.441 | [0.6129, 0.6576] |
| 10 | M5_defense_elo_abe | 16 | 0.63517 | 0.439 | [0.6102, 0.6578] |
| — | A0_baseline_17 (prod) | 17 | 0.64622 | 0.418 | [0.6251, 0.6681] |
| — | **MKT_market** | — | **0.63804** | — | [0.6077, 0.6698] |
| — | J0_only_odds (peor) | 3 | 0.65760 | 0.359 | [0.6305, 0.6871] |

### 3.4 Ranking Optuna (v2, con odds)

| # | Test | Features | Brier test | CV Brier | Depth | LR | N_est |
|---|------|----------|-----------|----------|-------|----|-------|
| 1 | **O7_all_abe_elo** | 18 | **0.63423** | **0.61768** | 2 | 0.014 | 250 |
| 2 | OA_only_elo | 3 | 0.63584 | 0.62312 | 2 | 0.021 | 136 |
| 3 | O5_m2_interactions | 9 | 0.63736 | 0.62073 | 2 | 0.021 | 135 |
| 4 | O4_defense_elo_kimi | 14 | 0.63757 | 0.62189 | 2 | 0.015 | 158 |
| 5 | O8_smart_minimal | 13 | 0.63827 | 0.61906 | 3 | 0.024 | 130 |
| 6 | O6_efficiency_elo | 8 | 0.63844 | 0.62473 | 2 | 0.024 | 118 |
| 7 | O2_defense_form_elo | 8 | 0.63930 | 0.62034 | 2 | 0.037 | 98 |
| 8 | O1_elo_gw_form | 6 | 0.64163 | 0.61888 | 5 | 0.021 | 120 |
| 9 | O3_elo_k20 | 3 | 0.64154 | 0.62340 | 2 | 0.025 | 131 |
| 10 | O9_baseline_17 | 17 | 0.64634 | 0.62733 | 2 | 0.030 | 94 |
| 11 | OF_abe_elo_odds | 21 | 0.64689 | **0.60321** | 2 | 0.038 | 58 |
| 12 | O0_elo_gw_defense | 5 | 0.64932 | 0.61375 | 2 | 0.020 | 209 |
| — | FIXED_baseline (prod) | 17 | 0.64610 | — | — | — | — |
| — | **MKT_market** | — | **0.63804** | — | — | — | — |

**4 tests skipped** (OB-OE): requieren xG no disponible.

**OF_abe_elo_odds**: Mejor CV (0.6032) pero PEOR en test (0.6469) = **overfitting masivo con odds**.

### 3.5 SHAP Analysis (v2, con odds)

| Test | Brier | #1 Feature (SHAP) | #2 Feature | #3 Feature |
|------|-------|-------------------|------------|------------|
| **S7_abe_elo** | **0.63192** | elo_diff (0.123) | opp_rating_diff (0.076) | elo_away (0.053) |
| S6_power_5 | 0.63510 | elo_diff (0.143) | opp_rating_diff (0.102) | overperf_diff (0.055) |
| S4_m2_interactions | 0.63698 | elo_diff (0.101) | elo_x_season (0.084) | elo_away (0.067) |
| S1_baseline_odds | 0.64082 | odds_away (0.119) | odds_home (0.114) | odds_draw (0.042) |
| S3_defense_elo | 0.64257 | elo_diff (0.151) | elo_away (0.094) | elo_home (0.053) |
| S0_baseline_17 | 0.64615 | goal_diff_avg (0.090) | away_goals_conceded_avg (0.049) | home_matches_played (0.047) |
| S8_abe_elo_odds | 0.65163 | odds_away (0.127) | odds_home (0.108) | odds_draw (0.039) |
| S2_elo_odds | 0.65793 | odds_away (0.129) | odds_home (0.122) | elo_away (0.058) |

**Draw-class top drivers (S7):** overperf_diff, elo_diff, draw_tendency_away

**Hallazgos SHAP clave:**
- `elo_diff` domina en TODOS los tests SIN odds (SHAP 0.101 — 0.151)
- `opp_rating_diff` es la segunda feature sin odds (0.076 — 0.102)
- **Odds como feature EMPEORAN el modelo**: S8 (0.6516) > S7 (0.6319), S2 (0.6579) > S3 (0.6426)
- Odds share en S8: 48.6%, pero el 51.4% restante (ABE+Elo) no compensa la perdida
- **Razon probable**: odds OddsPortal Ecuador son promedios de bookmakers poco liquidos, no Pinnacle
- ABE features siguen aportando senial complementaria: `overperf_diff`, `draw_tendency_*`

### 3.6 Que Funciona y Que No

| Categoria | Funciona | No Funciona |
|-----------|----------|-------------|
| **Elo** | SI — DOMINANTE. elo_diff #1 en todos los tests sin odds (SHAP 0.101-0.151) | — |
| **ABE features** | SI — K9/O7 campeon (18 feat), mejora 2.4% vs baseline | — |
| **Opp-adjusted ratings** | SI — opp_rating_diff #2 en SHAP (0.076-0.102) | — |
| **Overperf/draw-aware** | SI — complementos utiles. overperf_diff SHAP 0.035-0.055 | — |
| **C5 (elo_prob + draw_rate)** | SI — 2 features, Brier 0.63050, casi empata con K9 (18 feat) | — |
| **Form (win_rate, streaks)** | DEBIL — solo mejora combinado con Elo (C3, E3) | form_diff solo no agrega |
| **Defense (goals_conceded)** | MODERADO — complemento de Elo | No top-5 como standalone |
| **Matchup/H2H** | NO — F0 peor que baseline (0.66681) | Ruido, no senial |
| **Elo momentum** | NO — D6 es penultimo (0.66688) | — |
| **rest_days** | NEUTRAL — no daña pero no aporta | Confirma patron global |
| **Odds como feature** | **NO — EMPEORA el modelo. J0: 0.6576, S8: 0.6516 (ambos peor que sin odds)** | **Odds OddsPortal Ecuador son ruido, no senial** |
| **Odds como anchor** | **NO NECESARIO — modelo gana al mercado (FAIR Δ=-0.009)** | — |
| **xG** | NO DISPONIBLE — 0% coverage | Prioridad media |
| **Optuna tuning** | MARGINAL — O7 (0.63423) vs K9 standard (0.63127) | Standard ya es optimo |

### 3.7 Hyperparams Optimos (consensus v2)

De 12 tests Optuna ejecutados (v2, incluyendo OF con odds):

| Param | Rango optimo | Mediana | Nota |
|-------|-------------|---------|------|
| max_depth | **2** | 2 | 10/12 eligieron 2 (v1: O7 usaba 3, v2: cambio a 2) |
| learning_rate | 0.014 — 0.038 | 0.021 | Moderado |
| n_estimators | 58 — 250 | 131 | Inversamente proporcional a lr |
| min_child_weight | 3 — 15 | 12 | Alta regularizacion |

**Regla para Ecuador: arboles SIMPLES (depth=2), ABE+Elo como base, regularizacion ALTA (mcw>=10). NO usar odds como feature.**

### 3.8 Contraste con Argentina y Colombia

| Dimension | Argentina (128) | Colombia (239) | **Ecuador (242)** |
|-----------|----------------|----------------|-------------------|
| **Campeon** | ABE+Elo (18 feat) | Elo GW+Form (6 feat) | **ABE+Elo (18 feat)** |
| **Mejor Brier** | 0.6598 | 0.6280 | **0.6313** |
| **Baseline 17** | 0.6699 | 0.6326 | **0.6462** |
| **Mejora vs baseline** | +1.5% | +0.7% | **+2.3%** |
| **Feature dominante** | league_draw_rate | elo_diff / odds_away | **elo_diff** |
| **ABE features** | ESENCIALES | DEBILES sin odds | **ESENCIALES** |
| **Empates** | 31.9% | 30.8% | **28.7%** |
| **Odds coverage** | 100% | 96.3% | **95.0% (OddsPortal)** |
| **xG coverage** | 99.4% | 9.3% | **0%** |
| **Market Brier** | 0.5967 (Pinnacle) | 0.6091 (OddsPortal) | **0.6380 (OddsPortal)** |
| **Gap vs mercado** | +3.4% (mercado gana) | +3.1% (mercado gana) | **-1.4% (MODELO gana)** |
| **Market Anchor** | Activo | Viable | **NO NECESARIO** |
| **Odds como feature** | Neutral | DOMINANTE | **EMPEORA modelo** |
| **Depth optimo** | 2 | 2 | **2** |
| **N partidos** | 1,319 (2023+) | 2,924 | **1,772** |

**Hallazgos clave:**
- Ecuador se parece a Argentina en feature preferences (ABE+Elo campeon, elo_diff dominante)
- **UNICO caso donde el modelo SUPERA al mercado** (FAIR Δ=-0.009 a -0.014)
- Odds OddsPortal Ecuador son de BAJA CALIDAD: market Brier 0.6380 (vs 0.5967 Pinnacle ARG, 0.6091 OP COL)
- Esto explica por que odds no aportan como feature (ruido, no senial) y el modelo gana sin ellas
- Colombia en cambio se beneficia mucho de odds (market Brier 0.6091, odds_away SHAP 0.165)

### 3.9 Gap vs Mercado

```
Mejor modelo (K9 standard):    0.63127
Mejor modelo (O7 Optuna):      0.63423
Mercado (de-vigged OddsPortal): 0.63804  (N_test=338)
FAIR gap (K9 vs market):       -0.009 (-1.4%) → MODELO GANA

Comparacion de quality de mercado:
  Argentina (Pinnacle): 0.5967  ← mercado muy eficiente
  Colombia (OddsPortal): 0.6091 ← mercado moderado
  Ecuador (OddsPortal):  0.6380 ← mercado INEFICIENTE
```

**Ecuador tiene el mercado mas ineficiente de las 3 ligas.** El modelo captura senial que los bookmakers no liquidos no logran. Esto es consistente con ligas menores donde los mercados de apuestas tienen menos volumen y peor formacion de precios.

### 3.10 Estrategia Activa de Produccion

| Componente | Estado | Detalle |
|------------|--------|---------|
| Modelo base | v1.0.1 (17 features baseline) | Pendiente upgrade a v1.0.2 |
| Feature set objetivo | ABE+Elo (18 features) | K9/O7 campeon — NO incluir odds |
| Alternativa minimalista | C5 (elo_prob + draw_rate, 2 feat) | Brier 0.63050, casi identico |
| Hyperparams objetivo | depth=2, lr=0.014, n_est=250 | Optuna v2 Ecuador |
| Market Anchor | **NO NECESARIO** — modelo gana al mercado | Brier modelo < Brier mercado |
| Shadow mode | ACTIVO | Two-stage v1.1.0 |
| xG source | NINGUNA | Pendiente: FotMob backfill |
| Odds source | OddsPortal scrape (2019-2025) + odds_sync (2026+) | 95.0% coverage, NO usar como feature |

### 3.11 Proximo Paso

1. **v1.0.2**: Incluir ABE+Elo (18 feat) como feature set para Ecuador — SIN odds
2. **xG**: FotMob backfill (fixture parser tiene schema_break para Ecuador, TBD)
3. **Monitorear market quality**: Si odds mejoran (Pinnacle entra a Ecuador), re-evaluar
4. **Re-run lab**: Despues de tener xG o cuando formato del torneo cambie

### 3.12 Historial de Pruebas

| Fecha | Prueba | Script | Resultado clave |
|-------|--------|--------|-----------------|
| 2026-02-10 | Lab v1 (101 tests) | `feature_lab.py --league 242` | K9_all_abe_elo campeon (0.63025), 23 tests skipped (odds/xG) |
| 2026-02-10 | SHAP v1 (9 tests, 5 ran) | `feature_lab.py --shap --league 242` | S7_abe_elo campeon (0.63049), elo_diff domina |
| 2026-02-10 | Optuna v1 (16 cand, 11 ran) | `feature_lab.py --optuna --league 242` | O7_all_abe_elo campeon (0.63103), depth=3, 5 skipped |
| 2026-02-10 | OddsPortal odds backfill | `scrape_oddsportal_ecuador.py` + `ingest_oddsportal.py` | 1,686/1,775 matches (95.0%). Fuente: OddsPortal avg closing |
| **2026-02-11** | **Lab v2 (101 tests, con odds)** | `feature_lab.py --league 242` | **C5 campeon (0.6305), MKT 0.6380, FAIR Δ=-0.009 (modelo gana)** |
| **2026-02-11** | **SHAP v2 (9 tests, 8 ran)** | `feature_lab.py --shap --league 242` | **S7 campeon (0.6319), odds EMPEORAN (S8: 0.6516)** |
| **2026-02-11** | **Optuna v2 (16 cand, 12 ran)** | `feature_lab.py --optuna --league 242` | **O7 campeon (0.6342), OF con odds overfittea (CV 0.603 test 0.647)** |

### 3.13 Archivos de Evidencia

| Archivo | Contenido |
|---------|-----------|
| `scripts/output/lab/shap_analysis_242.json` | SHAP v2, 9 tests, Ecuador (con odds) |
| `scripts/output/lab/feature_lab_results_optuna.json` | Optuna v2, 16 candidatos, Ecuador (con odds) |
| `scripts/output/lab/feature_lab_results.json` | Lab v2, 101 tests, Ecuador (con odds) |
| `scripts/output/lab/lab_data_242.csv` | Dataset cacheado (1,772 rows) |
| `data/oddsportal_raw/ecuador-liga-pro_*.json` | OddsPortal scrapes 2019-2025 (1,700 matches) |
| `data/oddsportal_team_aliases.json` | Aliases Ecuador (54 entries, 25 teams) |

---

## 4. Venezuela (299)

### 4.1 Ficha Tecnica

| Campo | Valor |
|-------|-------|
| **League ID** | 299 |
| **Nombre** | Liga FUTVE (Venezuela Primera Division) |
| **Formato** | Temporada completa (no split-season), 14-21 equipos segun año |
| **Equipos** | 27 distintos en DB (2019-2026). 14-21 activos por temporada |
| **N partidos en DB** | 2,229 (Ene 2019 — Feb 2026), 1,852 FT |
| **N partidos testeados** | 1,280 (filtro 2021-01-01+, excluye 2020 COVID: 326 cancelados) |
| **Train / Test split** | 1,024 / 256 |
| **Distribucion resultados** | Home 43.1%, Draw 29.7%, Away 27.2% |
| **Brier naive** | ~0.647 (predictor marginal) |
| **Dificultad** | MEDIA — ventaja local fuerte, empates moderados |
| **Odds coverage** | **97.3% desde 2021 (1,246/1,280). Fuente: OddsPortal scrape** |
| **xG coverage** | **0% — sin fuente de xG disponible** |
| **Market Anchor** | **VIABLE — mercado gana por +2.0-3.3%** |

### 4.2 Campeon: N6 — Odds Clean (15 features)

```
Features (15):
  odds_home, odds_draw, odds_away,
  elo_home, elo_away, elo_diff,
  home_goals_scored_avg, away_goals_scored_avg,
  home_goals_conceded_avg, away_goals_conceded_avg,
  goal_diff_avg, abs_strength_gap,
  form_diff, home_matches_played, away_matches_played

Brier (test, N=256): 0.60358
Accuracy: 51.0%
```

**Sin odds — Campeon alternativo: C3_elo_diff_form (2 features)**

```
Features (2): elo_diff, form_diff
Brier (test, N=256): 0.62782
Accuracy: 47.0%
Nota: elo_diff solo (1 feat) logra 0.63052 — casi identico
```

### 4.3 Ranking Completo (Standard, top-15)

| # | Test | Features | Brier | Acc |
|---|------|----------|-------|-----|
| 1 | **N6_odds_clean** | 15 | **0.60358** | 51.0% |
| 2 | J2_full_odds | 20 | 0.60721 | 51.2% |
| 3 | N7_odds_power7 | 10 | 0.61201 | 50.2% |
| 4 | N5_odds_kimi_all | 15 | 0.61924 | 47.0% |
| 5 | N3_odds_efficiency | 11 | 0.61943 | 46.7% |
| ... | ... | ... | ... | ... |
| — | C3_elo_diff_form (sin odds) | 2 | 0.62782 | 47.0% |
| — | B3_elo_diff (1 feat) | 1 | 0.63052 | — |
| — | K9_all_abe_elo | 18 | 0.64007 | — |
| — | A0_baseline_17 (prod) | 17 | ~0.645 | — |
| — | **MKT_market** | — | **0.58408** | — |

**91/101 tests ejecutados** (10 skipped: xG-dependientes).

### 4.4 Ranking Optuna

| # | Test | Features | Brier test | CV Brier | Depth | LR | N_est |
|---|------|----------|-----------|----------|-------|----|-------|
| 1 | **OF_abe_elo_odds** | 21 | **0.61949** | 0.64799 | 6 | 0.010 | 142 |
| 2 | O1_elo_gw_form | 6 | 0.63761 | 0.65623 | 2 | 0.017 | — |
| 3 | OA_only_elo | 3 | 0.63765 | 0.65754 | 2 | 0.015 | — |
| 4 | O7_all_abe_elo | 18 | 0.63895 | 0.65378 | 2 | 0.014 | — |
| 5 | O4_defense_elo_kimi | 14 | 0.64112 | 0.65032 | 2 | 0.012 | — |
| — | FIXED_baseline (prod) | 17 | ~0.645 | — | — | — | — |
| — | **MKT_market** | — | **0.58408** | — | — | — | — |

**12/16 tests ejecutados** (4 skipped: xG-dependientes).

### 4.5 SHAP Analysis

| Test | Brier | #1 Feature (SHAP) | #2 Feature | #3 Feature |
|------|-------|-------------------|------------|------------|
| **S1_baseline_odds** | **0.61028** | odds_home (0.119) | home_goals_scored_avg (0.073) | odds_away (0.064) |
| S8_abe_elo_odds | — | odds_home (0.093) | odds_away (0.076) | opp_att_home (0.073) |
| S2_elo_odds | — | odds_home (0.140) | odds_away (0.087) | elo_diff (0.085) |
| S7_abe_elo | — | elo_diff (0.111) | opp_att_home (0.075) | — |
| S0_baseline_17 | — | away_goals_conceded_avg (0.105) | home_goals_scored_avg (0.084) | — |

**Hallazgos SHAP clave:**
- `odds_home` domina cuando estan presentes (SHAP 0.093-0.140)
- Odds share: 54.8% en S2 (elo+odds), 39.9% en S1 (baseline+odds), 33.4% en S8
- Sin odds: `elo_diff` (0.111) y `away_goals_conceded_avg` (0.105) son top
- Draw-class: `odds_draw`, `opp_att_home`, `elo_diff`

### 4.6 Que Funciona y Que No

| Categoria | Funciona | No Funciona |
|-----------|----------|-------------|
| **Odds como feature** | **SI — DOMINANTE. N6 campeon (0.6036), odds_home SHAP #1** | Sin odds: tests N* skipped |
| **Odds como anchor** | **VIABLE — mercado 0.5841, gap +2.0-3.3%** | — |
| **Elo** | SI — elo_diff solo (1 feat) logra 0.6305, top sin odds | — |
| **Form** | SI — form_diff complementa elo_diff (C3: 0.6278 vs B3: 0.6305) | Marginal |
| **ABE features** | DEBIL — K9 (18 feat) = 0.6401, PEOR que C3 (2 feat) 0.6278 | No justificado |
| **Defense** | MODERADO — incluido en N6 pero no es top SHAP | — |
| **xG** | NO DISPONIBLE — 0% coverage | Pendiente FotMob |
| **rest_days** | NO — ruido confirmado | — |
| **Optuna depth=6** | CAUTELA — OF con odds usa depth=6 (unico caso) | Riesgo overfitting |

### 4.7 Hyperparams Optimos

De 12 tests Optuna:

| Param | Rango optimo | Mediana | Nota |
|-------|-------------|---------|------|
| max_depth | **2** | 2 | 11/12 eligieron 2 (excepto OF con odds: 6) |
| learning_rate | 0.010 — 0.017 | 0.014 | Bajo |
| n_estimators | — | ~100 | Ajustar inversamente a lr |
| min_child_weight | 3 — 7 | — | Regularizacion moderada |

**Regla para Venezuela: arboles SIMPLES (depth=2), pocas features sin odds (1-2), odds como feature cuando disponibles.**

### 4.8 Gap vs Mercado

```
Mejor modelo (N6 standard):     0.60358
Mejor modelo sin odds (C3):     0.62782
Mercado (de-vigged OddsPortal):  0.58408  (N_test=250)
FAIR gap (N6 vs market):        +0.0195 (+3.3%)

Optuna champion (OF):           0.61949
FAIR gap (OF vs market):        +0.0354 (+6.1%)
```

**Mercado gana por +2.0-3.3%.** Patron consistente con Argentina y Colombia. Market Anchor es VIABLE.

### 4.9 Contraste con Argentina, Colombia y Ecuador

| Dimension | Argentina (128) | Colombia (239) | Ecuador (242) | **Venezuela (299)** |
|-----------|----------------|----------------|---------------|---------------------|
| **Campeon** | ABE+Elo (18) | Elo GW+Form (6) | ABE+Elo (18) | **N6 odds_clean (15)** |
| **Mejor Brier** | 0.6598 | 0.6280 | 0.6313 | **0.6036** |
| **Sin odds** | 0.6608 (S7) | 0.6280 (O1) | 0.6305 (C5) | **0.6278 (C3)** |
| **Market Brier** | 0.5967 | 0.6091 | 0.6380 | **0.5841** |
| **Gap vs mercado** | +3.4% | +3.1% | -1.4% (modelo gana) | **+3.3%** |
| **Feature dominante** | league_draw_rate | elo_diff / odds_away | elo_diff | **odds_home / elo_diff** |
| **ABE features** | ESENCIALES | DEBILES sin odds | ESENCIALES | **DEBILES** |
| **Odds como feature** | Neutral | DOMINANTE | EMPEORA | **DOMINANTE** |
| **Market Anchor** | Activo | Viable | No necesario | **Viable** |
| **Empates** | 31.9% | 30.8% | 28.7% | **29.7%** |
| **xG** | 99.4% | 9.3% | 0% | **0%** |
| **Depth optimo** | 2 | 2 | 2 | **2** |
| **N partidos** | 1,319 | 2,924 | 1,772 | **1,280** |

**Hallazgos clave:**
- Venezuela tiene el **mercado MAS EFICIENTE** de las 4 ligas (Brier 0.5841 < ARG 0.5967 < COL 0.6091 < ECU 0.6380)
- Odds como feature aportan fuerte (N6 0.6036 vs C3 0.6278 = -3.9%), similar a Colombia
- ABE features NO aportan (K9 0.6401 > C3 0.6278) — al contrario de Ecuador y Argentina
- Venezuela se comporta como Colombia: pocas features, odds-driven, Market Anchor viable
- **Patron emergente LATAM**: ARG/VEN/COL → mercado gana, Market Anchor viable. ECU → mercado ineficiente, modelo gana

### 4.10 Estrategia Activa de Produccion

| Componente | Estado | Detalle |
|------------|--------|---------|
| Modelo base | v1.0.1 (17 features baseline) | Pendiente upgrade a v1.0.2 |
| Feature set objetivo (sin odds) | C3: elo_diff + form_diff (2 features) | Minimalista, campeon sin odds |
| Feature set objetivo (con odds) | N6: odds + elo + goals + form (15 features) | Campeon standard |
| Hyperparams objetivo | depth=2, lr=0.014, n_est=~100 | Optuna Venezuela |
| **Market Anchor** | **VIABLE — activar con alpha TBD** | Mercado gana por +2-3% |
| Shadow mode | ACTIVO | Two-stage v1.1.0 |
| xG source | NINGUNA | Pendiente: FotMob backfill |
| Odds source | **OddsPortal scrape (2021-2026) + odds_sync** | 97.3% coverage 2021+ |

### 4.11 Proximo Paso

1. **Market Anchor Venezuela**: Activar con alpha=TBD basado en gap +3.3%
2. **v1.0.2 per-league features**: C3 (2 feat sin odds) o N6 (15 feat con odds)
3. **xG**: FotMob backfill cuando fixture parser soporte Venezuela
4. **Rebrand confirmado**: CD Hermanos Colmenarez → Inter de Barinas (temporada 2024). OddsPortal retroactivo. Alias OK.
5. **Re-run lab**: Despues de tener xG o con mas temporadas (2027+)

### 4.12 Historial de Pruebas

| Fecha | Prueba | Script | Resultado clave |
|-------|--------|--------|-----------------|
| 2026-02-11 | Lab v1 (sin odds, 78/101) | `feature_lab.py --league 299 --min-date 2021-01-01` | C3_elo_diff_form campeon (0.6285), 23 skipped |
| 2026-02-11 | OddsPortal scrape + ingestion | `scrape_oddsportal_venezuela.py` + `ingest_oddsportal.py` | 1,246/1,280 matches (97.3%). 6 temporadas. |
| **2026-02-11** | **Lab v2 (con odds, 91/101)** | `feature_lab.py --league 299 --min-date 2021-01-01` | **N6_odds_clean campeon (0.6036), MKT 0.5841, FAIR +3.3%** |
| **2026-02-11** | **SHAP v2 (8/9)** | `feature_lab.py --shap --league 299 --min-date 2021-01-01` | **S1 campeon (0.6103), odds_home domina (54.8% share)** |
| **2026-02-11** | **Optuna v2 (12/16)** | `feature_lab.py --optuna --league 299 --min-date 2021-01-01` | **OF campeon (0.6195), depth=6, mercado imbatible** |

### 4.13 Archivos de Evidencia

| Archivo | Contenido |
|---------|-----------|
| `scripts/output/lab/shap_analysis_299.json` | SHAP v2, 9 tests, Venezuela (con odds) |
| `scripts/output/lab/feature_lab_results_optuna.json` | Optuna v2, 16 candidatos, Venezuela (con odds) |
| `scripts/output/lab/feature_lab_results.json` | Lab v2, 101 tests, Venezuela (con odds) |
| `scripts/output/lab/lab_data_299.csv` | Dataset cacheado (1,852 rows) |
| `data/oddsportal_raw/venezuela-primera-division_*.json` | OddsPortal scrapes 2021-2026 (1,255 matches) |
| `data/oddsportal_team_aliases.json` | Aliases Venezuela (53 entries, 27 teams) |
| `scripts/scrape_oddsportal_venezuela.py` | Scraper multi-temporada con checkpoint |

### 4.14 Nota: Alias "Inter de Barinas" = CD Hermanos Colmenarez

CD Hermanos Colmenarez se rebrandeó como **Inter de Barinas** de cara a la temporada 2024 (nueva identidad corporativa, mismo club). OddsPortal aplica el nombre nuevo retroactivamente a todas las temporadas. API-Football mantiene el nombre antiguo (id=4337). Confirmado por mapping 1:1 en 4 temporadas (2021-2024).

---

## 5. Peru (281)

### 5.1 Ficha Tecnica

| Campo | Valor |
|-------|-------|
| **League ID** | 281 |
| **Nombre** | Liga 1 (Peru Primera Division) |
| **Formato** | Temporada completa con Apertura + Clausura + play-offs, 18 equipos (2022+) |
| **Equipos** | 31 distintos en DB (2019-2026). 18 activos por temporada |
| **N partidos en DB** | 2,390 (Ene 2019 — Feb 2026), 2,176 FT |
| **N partidos testeados** | 1,580 (filtro 2021-01-01+, excluye 2019-2020: COVID severo, 184 PST en 2020) |
| **Train / Test split** | 1,264 / 316 |
| **Distribucion resultados** | Home ~43%, Draw ~28%, Away ~29% |
| **Brier naive** | ~0.648 (predictor marginal) |
| **Dificultad** | MEDIA — formato variado, equipos suben/bajan frecuentemente |
| **Odds coverage** | **95.6% global (2,081/2,176). 97%+ desde 2020. Fuente: OddsPortal scrape** |
| **xG coverage** | **0% — sin fuente de xG disponible** |
| **Market Anchor** | **VIABLE — mercado gana por +1.6-2.1%** |

### 5.2 Campeon: J2 — Full Odds (20 features)

```
Features (20):
  odds_home, odds_draw, odds_away,
  goal_diff_avg, home_goals_scored_avg, away_goals_scored_avg,
  home_goals_conceded_avg, away_goals_conceded_avg,
  abs_strength_gap, home_shots_avg, away_shots_avg,
  home_defense_avg, away_defense_avg,
  form_diff, win_rate_last5, h2h_home_adv,
  home_matches_played, away_matches_played,
  elo_diff, league_draw_rate

Brier (test, N=316): 0.58648
Accuracy: ~51%
```

**Sin odds — Campeon alternativo: D1_elo_k10 (3 features)**

```
Features (3): elo_k10_home, elo_k10_away, elo_k10_diff
Brier (test, N=316): 0.59417
Accuracy: 50.9%
Nota: Elo K=10 funciona mejor que K=32 default — Peru es liga de ajuste rapido
```

### 5.3 Ranking Completo (Standard, top-15)

| # | Test | Features | Brier | Acc |
|---|------|----------|-------|-----|
| 1 | **J2_full_odds** | 20 | **0.58648** | ~51% |
| 2 | N6_odds_clean | 15 | 0.58685 | ~51% |
| 3 | J0_only_odds | 3 | 0.58744 | ~52% |
| 4 | N7_odds_power7 | 10 | 0.58996 | ~50% |
| 5 | N8_odds_minimal | 5 | 0.59002 | ~51% |
| ... | ... | ... | ... | ... |
| — | D1_elo_k10 (sin odds) | 3 | 0.59417 | 50.9% |
| — | D5_elo_split | 3 | 0.59528 | 52.0% |
| — | F4_h2h_elo | 5 | 0.59632 | 52.2% |
| — | H6_elo_gw_form | 6 | 0.59470 | 50.9% |
| — | A0_baseline_17 (prod) | 17 | 0.61593 | 46.5% |
| — | **MKT_market** | — | **0.57085** | — |

**91/101 tests ejecutados** (10 skipped: xG-dependientes).

### 5.4 Ranking Optuna

| # | Test | Features | Brier test | CV Brier | Depth | LR | N_est |
|---|------|----------|-----------|----------|-------|----|-------|
| 1 | **OF_abe_elo_odds** | 21 | **0.59104** | 0.55260 | 3 | 0.023 | 112 |
| 2 | O3_elo_k20 | 3 | 0.59575 | 0.58744 | 2 | 0.015 | 289 |
| 3 | O1_elo_gw_form | 6 | 0.59717 | 0.58662 | 2 | 0.031 | 163 |
| 4 | O0_elo_gw_defense | 5 | 0.60120 | 0.58320 | 2 | 0.023 | 130 |
| 5 | O5_m2_interactions | 9 | 0.60164 | 0.58059 | 2 | 0.030 | 118 |
| — | FIXED_baseline (prod) | 17 | 0.61593 | — | — | — | — |
| — | **MKT_market** | — | **0.57085** | — | — | — | — |

**12/16 tests ejecutados** (4 skipped: xG-dependientes).

### 5.5 SHAP Analysis

| Test | Brier | #1 Feature (SHAP) | #2 Feature | #3 Feature |
|------|-------|-------------------|------------|------------|
| **S1_baseline_odds** | **0.58230** | odds_home (0.166) | odds_away (0.145) | odds_draw (0.050) |
| S8_abe_elo_odds | — | odds_home (~0.15) | odds_away (~0.13) | opp_att_home (~0.05) |
| S2_elo_odds | — | odds_home (~0.16) | odds_away (~0.14) | elo_diff (~0.04) |
| S7_abe_elo | — | elo_diff (~0.16) | opp_att_home (~0.06) | — |
| S0_baseline_17 | — | elo_diff (~0.15) | home_shots_avg (~0.04) | — |

**Hallazgos SHAP clave:**
- `odds_home` + `odds_away` dominan absolutamente (51-76% SHAP share)
- Sin odds: `elo_diff` domina en todos los tests (SHAP 0.15-0.19)
- Draw-class: `league_draw_rate`, `odds_draw`, `home_shots_avg`
- Odds son mas informativos que cualquier combinacion de features engineered

### 5.6 Que Funciona y Que No

| Categoria | Funciona | No Funciona |
|-----------|----------|-------------|
| **Odds como feature** | **SI — DOMINANTE. J2 campeon (0.5865), odds 51%+ SHAP share** | — |
| **Odds como anchor** | **VIABLE — mercado 0.5709, gap +1.6-2.1%** | — |
| **Elo** | SI — D1_elo_k10 (3 feat) logra 0.5942, mejor sin odds | K=32 ligeramente peor |
| **Elo K=10** | MEJOR — ajuste rapido, captura volatilidad peruana | K=50, K=64 peores |
| **H2H** | SI — F4_h2h_elo (5 feat) = 0.5963, H2H relevante | Solo no alcanza |
| **Form** | SI — form_diff complementa elo (H6: 0.5947) | Marginal como feature sola |
| **ABE features** | DEBIL — K9 (18 feat) = 0.6070, PEOR que D1 (3 feat) 0.5942 | No justificado |
| **xG** | NO DISPONIBLE — 0% coverage | Pendiente FotMob |
| **rest_days** | NO — ruido | — |
| **Baseline 17** | NO — 0.6159, peor que Elo solo | Demasiadas features ruidosas |

### 5.7 Hyperparams Optimos

De 12 tests Optuna:

| Param | Rango optimo | Mediana | Nota |
|-------|-------------|---------|------|
| max_depth | **2** | 2 | 11/12 eligieron 2 (OF con odds: 3) |
| learning_rate | 0.015 — 0.031 | ~0.025 | Moderado |
| n_estimators | 112 — 289 | ~150 | Inversamente proporcional a lr |
| min_child_weight | 3 — 15 | ~11 | Regularizacion fuerte |

**Regla para Peru: arboles SIMPLES (depth=2), Elo K=10 sin odds, odds_clean con odds.**

### 5.8 Gap vs Mercado

```
Mejor modelo (J2 standard):      0.58648
Mejor modelo sin odds (D1):      0.59417
Mercado (de-vigged OddsPortal):   0.57085  (N_test=308)
FAIR gap (J2 vs market):         +0.01563 (+2.7%)

Optuna champion (OF):            0.59104
FAIR gap (OF vs market):         +0.02081 (+3.6%)
```

**Mercado gana por +1.6-2.7%.** Patron LATAM confirmado. Market Anchor VIABLE.

### 5.9 Contraste con Argentina, Colombia, Ecuador y Venezuela

| Dimension | Argentina (128) | Colombia (239) | Ecuador (242) | Venezuela (299) | **Peru (281)** |
|-----------|----------------|----------------|---------------|-----------------|----------------|
| **Campeon** | ABE+Elo (18) | Elo GW+Form (6) | ABE+Elo (18) | N6 odds_clean (15) | **J2 full_odds (20)** |
| **Mejor Brier** | 0.6598 | 0.6280 | 0.6313 | 0.6036 | **0.5865** |
| **Sin odds** | 0.6608 (S7) | 0.6280 (O1) | 0.6305 (C5) | 0.6278 (C3) | **0.5942 (D1)** |
| **Market Brier** | 0.5967 | 0.6091 | 0.6380 | 0.5841 | **0.5709** |
| **Gap vs mercado** | +3.4% | +3.1% | -1.4% (modelo gana) | +3.3% | **+2.7%** |
| **Feature dominante** | league_draw_rate | elo_diff / odds_away | elo_diff | odds_home / elo_diff | **odds_home / odds_away** |
| **ABE features** | ESENCIALES | DEBILES | ESENCIALES | DEBILES | **DEBILES** |
| **Odds como feature** | Neutral | DOMINANTE | EMPEORA | DOMINANTE | **DOMINANTE** |
| **Market Anchor** | Activo | Viable | No necesario | Viable | **Viable** |
| **Empates** | 31.9% | 30.8% | 28.7% | 29.7% | **~28%** |
| **xG** | 99.4% | 9.3% | 0% | 0% | **0%** |
| **Depth optimo** | 2 | 2 | 2 | 2 | **2** |
| **N partidos** | 1,319 | 2,924 | 1,772 | 1,280 | **1,580** |

**Hallazgos clave:**
- Peru tiene el **mejor Brier global** (0.5865) y el mercado mas predecible despues de Venezuela
- Odds como feature aportan fuerte (J2 0.5865 vs D1 0.5942 = -1.3%), pero menor gap que VEN/COL
- **Elo K=10 es la mejor variante sin odds** — unico caso donde K<32 gana. Liga de ajuste rapido
- ABE features NO aportan (K9 0.6070 >> D1 0.5942) — patron VEN/COL
- **Patron LATAM confirmado (5 ligas)**: ARG/VEN/COL/PER → mercado gana, Market Anchor viable. ECU → modelo gana

### 5.10 Estrategia Activa de Produccion

| Componente | Estado | Detalle |
|------------|--------|---------|
| Modelo base | v1.0.1 (17 features baseline) | Pendiente upgrade a v1.0.2 |
| Feature set objetivo (sin odds) | D1: elo_k10 (3 features) | Elo K=10, campeon sin odds |
| Feature set objetivo (con odds) | J2: full_odds (20 features) o N6: odds_clean (15) | Top standard |
| Hyperparams objetivo | depth=2, lr=0.025, n_est=~150 | Optuna Peru |
| **Market Anchor** | **VIABLE — activar con alpha TBD** | Mercado gana por +1.6-2.7% |
| Shadow mode | ACTIVO | Two-stage v1.1.0 |
| xG source | NINGUNA | Pendiente: FotMob backfill |
| Odds source | **OddsPortal scrape (2019-2026) + odds_sync** | 95.6% coverage global |

### 5.11 Proximo Paso

1. **Market Anchor Peru**: Activar con alpha=TBD basado en gap +2.7%
2. **v1.0.2 per-league features**: D1 (3 feat sin odds) o J2/N6 (15-20 feat con odds)
3. **xG**: FotMob backfill cuando fixture parser soporte Peru
4. **Cusco/Real Garcilaso**: En 2019 "Cusco" en OddsPortal = "Real Garcilaso" en DB (equipo rebrandeado). 34 matches 2019 no ingresados (DB usa team_id diferente). Aceptable.
5. **Re-run lab**: Despues de tener xG o con mas temporadas

### 5.12 Historial de Pruebas

| Fecha | Prueba | Script | Resultado clave |
|-------|--------|--------|-----------------|
| 2026-02-11 | Lab v1 (sin odds, 78/101) | `feature_lab.py --league 281 --min-date 2021-01-01` | D1_elo_k10 campeon (0.5942), 23 skipped |
| 2026-02-11 | OddsPortal scrape + ingestion | `scrape_oddsportal_peru.py` + `ingest_oddsportal.py` | 2,082/2,176 matches (95.6%). 8 temporadas. |
| **2026-02-11** | **Lab v2 (con odds, 91/101)** | `feature_lab.py --league 281 --min-date 2021-01-01` | **J2_full_odds campeon (0.5865), MKT 0.5709, FAIR +2.7%** |
| **2026-02-11** | **SHAP v2 (8/9)** | `feature_lab.py --shap --league 281 --min-date 2021-01-01` | **S1 campeon (0.5823), odds_home domina (51%+ share)** |
| **2026-02-11** | **Optuna v2 (12/16)** | `feature_lab.py --optuna --league 281 --min-date 2021-01-01` | **OF campeon (0.5910), depth=3, mercado imbatible** |

### 5.13 Archivos de Evidencia

| Archivo | Contenido |
|---------|-----------|
| `scripts/output/lab/shap_analysis_281.json` | SHAP v2, 9 tests, Peru (con odds) |
| `scripts/output/lab/feature_lab_results_optuna.json` | Optuna v2, 16 candidatos, Peru (con odds) |
| `scripts/output/lab/feature_lab_results.json` | Lab v2, 101 tests, Peru (con odds) |
| `scripts/output/lab/lab_data_281.csv` | Dataset cacheado (2,176 rows) |
| `data/oddsportal_raw/peru-liga-1_*.json` | OddsPortal scrapes 2019-2026 (2,134 matches) |
| `data/oddsportal_team_aliases.json` | Aliases Peru (60+ entries, 31 teams) |
| `scripts/scrape_oddsportal_peru.py` | Scraper multi-temporada con checkpoint |

### 5.14 Nota: Alias "Cusco" vs "Real Garcilaso"

En 2019, OddsPortal usa "Cusco" pero API-Football tiene "Real Garcilaso" (id=4221) para ese equipo. Desde 2020, API-Football crea un nuevo registro "Cusco" (id=4288). Son IDs diferentes en la DB. OddsPortal aplica el nombre "Cusco" retroactivamente. Resultado: 34 matches de 2019 no matchearon en la ingesta (alias "Cusco" apunta a 4288, no a 4221). Impacto: 1.6% de loss, aceptable.

### 5.15 Nota: "Cajamarca" = UTC, "Los Chankas" = Cultural Santa Rosa

- **Cajamarca** en OddsPortal = **UTC** (id=4277) en DB. "UTC Cajamarca" es el nombre completo del club. OddsPortal usa "Cajamarca" en todas las temporadas.
- **Los Chankas** en OddsPortal = **Cultural Santa Rosa** (id=4293) en DB. "Los Chankas de Andahuaylas" es el nombre completo. Aparece desde 2024.
- **FC Cajamarca** (id=6186) es un equipo DIFERENTE que debuta en 2026 (no confundir con UTC Cajamarca).

---

## 6. Bolivia (344)

### 6.1 Ficha Tecnica

| Campo | Valor |
|-------|-------|
| **League ID** | 344 |
| **Nombre** | Division Profesional de Bolivia |
| **Formato** | Apertura + Clausura, temporada calendario (Ene-Dic) |
| **Equipos** | 14-18 por temporada (27 unicos en DB) |
| **N partidos en DB** | 1,951 (2019-2025, status FT) |
| **N partidos testeados** | 1,406 (filtro 2021-01-01+) |
| **Train / Test split** | 1,124/282 (sin odds), 1,095/274 (con odds) |
| **Brier naive** | ~0.63 (estimado) |
| **Dificultad** | MEDIA — varianza alta por equipos chicos/ascensos frecuentes |
| **Odds coverage** | 96.7% global (1,887/1,951). 2020-2025: 95-98% |
| **xG coverage** | 0% — sin fuente disponible |
| **Market Anchor** | VIABLE (gap +0.9% a +3.0%) |

### 6.2 Campeon Lab Estandar: N7_odds_power7 (10 features)

```
Features (10):
  elo_home, elo_away, elo_diff, form_diff,
  home_goals_scored_avg, away_goals_conceded_avg, abs_strength_gap,
  odds_home, odds_draw, odds_away

Brier:    0.51676 ± CI95 (N_test=274)
Accuracy: 0.617
Market:   0.50709
FAIR gap: +0.97% (mercado gana)
```

### 6.3 Campeon Sin Odds: F2_matchup_form_elo (10 features)

```
Features (10):
  h2h_wins_home, h2h_wins_away, confrontation_score, h2h_goal_diff,
  win_rate_last5, form_diff,
  elo_home, elo_away, elo_diff,
  league_draw_rate

Brier:    0.53893 (lab v2)
Accuracy: 0.584
```

Bolivia es la unica liga LATAM donde las features de matchup (confrontacion directa, h2h) dominan sobre Elo puro sin odds. F1_matchup_elo (7 features, Brier 0.542) y H3_defense_matchup_elo (9 features, 0.540) tambien destacan.

### 6.4 SHAP: Top Features

**Con odds (S1_baseline_odds, Brier 0.52856):**

| Feature | SHAP | % Share |
|---------|------|---------|
| odds_home | 0.123 | 19.2% |
| odds_away | 0.117 | 18.3% |
| odds_draw | 0.077 | 12.1% |
| abs_strength_gap | 0.042 | 6.5% |
| home_goals_scored_avg | 0.029 | 4.5% |

Odds dominan 49.5% del SHAP share.

**Sin odds (S0_baseline_17):**

| Feature | SHAP |
|---------|------|
| goal_diff_avg | 0.122 |
| away_goals_conceded_avg | 0.073 |
| away_matches_played | 0.071 |
| abs_strength_gap | 0.067 |
| home_goals_scored_avg | 0.062 |

### 6.5 Optuna: Mejores Candidatos

| Test | Brier | CV | Depth | Features |
|------|-------|----|-------|----------|
| OF_abe_elo_odds | 0.537 | 0.581 | 2 | 21 |
| O0_elo_gw_defense | 0.556 | 0.591 | 2 | 5 |
| O2_defense_form_elo | 0.558 | 0.597 | 6 | 8 |
| O7_all_abe_elo | 0.560 | 0.600 | 2 | 18 |

**Patron**: depth=2 dominante (11/12 tests), learning rates bajos (0.01-0.04). Bolivia prefiere modelos conservadores.

### 6.6 Comparacion: Mercado vs Modelo

| Metrica | Valor |
|---------|-------|
| Market Brier | **0.50709** |
| Mejor modelo (N7) | 0.51676 |
| FAIR gap (lab) | +0.97% |
| FAIR gap (Optuna) | +3.0% |
| Verdict | Mercado gana |

### 6.7 Comparacion 6 Ligas LATAM

| Liga | Lab Champion | Market | FAIR Gap | Market Anchor |
|------|-------------|--------|----------|---------------|
| Argentina | 0.6585 | 0.6348 | +3.6% | ACTIVO (alpha=1.0) |
| Colombia | 0.6245 | 0.6091 | +2.5% | VIABLE |
| Ecuador | 0.5949 | 0.6037 | -1.5% | NO (modelo gana) |
| Venezuela | 0.5653 | 0.5476 | +3.2% | VIABLE |
| Peru | 0.5823 | 0.5709 | +2.0% | VIABLE |
| **Bolivia** | **0.5168** | **0.5071** | **+1.9%** | **VIABLE** |

Bolivia tiene los mejores Brier absolutos del grupo (mercado y modelo). Esto sugiere un mercado eficiente y mayor predictibilidad intrinseca de la liga.

### 6.8 COVID-2020

2020 tuvo temporada reducida (182 partidos vs 250-360 normales). No se detectaron cancelaciones masivas como Chile. Los datos de 2020 se excluyen via `--min-date 2021-01-01`.

### 6.9 Cancelled Matches (2022)

2022 tuvo 32 partidos cancelados en la DB. En OddsPortal, estos aparecen como noise entries (ej: `12:00canc.Palmaflor–The Strongestcanc.`). Se limpian pre-ingesta.

### 6.10 Recomendaciones

1. **Market Anchor**: Activar con alpha=1.0 (misma politica que Argentina)
2. **Sin xG**: Bolivia no tiene fuente de xG. FotMob no esta verificado para Bolivia. Todos los tests xG (P0-P9) skipped.
3. **Matchup features**: Si se implementa modelo por liga, F2_matchup_form_elo es la combinacion sin odds optima. Unica liga LATAM donde matchup supera a Elo puro.
4. **Re-evaluar**: Cada 6 meses. Si se consigue fuente xG, priorizar tests P0-P9.

### 6.11 Scraping

**Fuente**: OddsPortal via Playwright (`scripts/scrape_oddsportal_bolivia.py`)
**URL base**: `https://www.oddsportal.com/football/bolivia/division-profesional/`
**Patron historico**: `division-profesional-{year}` (2019-2024), sin sufijo = 2025
**Temporadas scrapeadas**: 7 (2019-2025)
**Rendimiento**: 99-100% odds en 2020-2025, 96% en 2019

| Year | Matches | Con Odds | % |
|------|---------|----------|---|
| 2019 | 353 | 342 | 96% |
| 2020 | 179 | 177 | 99% |
| 2021 | 232 | 232 | 100% |
| 2022 | 329 | 329 | 100% |
| 2023 | 269 | 268 | 100% |
| 2024 | 315 | 312 | 99% |
| 2025 | 238 | 238 | 100% |
| **Total** | **1,915** | **1,898** | **99%** |

### 6.12 Aliases

Aliases en `data/oddsportal_team_aliases.json` seccion `BoliviaDivisionProfesional`:

| OddsPortal | DB Name | ID |
|------------|---------|-----|
| Destroyers | Club Destroyers | 4350 |
| Wilstermann | Jorge Wilstermann | 4232 |
| Bolivar | Bolivar | 4222 |
| Guabira | Guabira | 4322 |
| Nacional Potosi | Nacional Potosi | 4265 |
| Real Potosi | Real Potosi | 4349 |
| San Jose | San Jose | 4231 |
| Palmaflor | Atletico Palmaflor | 4334 |
| Independiente | Independiente Petrolero | 4263 |
| Libertad Gran Mamore | Libertad | 4353 |
| Tomayapo | Real Tomayapo | 4340 |
| U. Sucre | Club Universitario | 4352 |
| SA Bulo Bulo | San Antonio Bulo Bulo | 4355 |
| GV San Jose | Gualberto Villarroel SJ | 4356 |
| San Juan FC | CD IN San Juan FC | 6153 |
| Academia del Balompie | ABB | 6154 |
| Vaca Diez | Vaca Diez | 4354 |

Total: 52 alias entries, 27 equipos.

### 6.13 Ingesta

- **Script**: `scripts/ingest_oddsportal.py --section BoliviaDivisionProfesional`
- **GATE**: 27 nombres, 100% resueltos
- **Match rate**: 98.5% (1,887/1,915)
- **Score mismatches**: 7 (walkovers 3-0 de federacion vs resultado real en cancha)
- **Unmatched**: 1 (San Jose vs Blooming 2021-09-25, posible reprogramacion)
- **Columnas**: `opening_odds_home/draw/away` (no sobreescribe `odds_*` del pipeline live)

### 6.14 Archivos

| Archivo | Descripcion |
|---------|-------------|
| `scripts/output/lab/feature_lab_results.json` | Lab v2, 91/101 tests, Bolivia (con odds) |
| `scripts/output/lab/shap_analysis_344.json` | SHAP, 8/9 tests |
| `scripts/output/lab/feature_lab_results_optuna.json` | Optuna, 12/16 tests |
| `scripts/output/lab/lab_data_344.csv` | Dataset cacheado (1,951 rows) |
| `data/oddsportal_raw/bolivia-division-profesional_*.json` | OddsPortal scrapes 2019-2025 (1,915 matches) |
| `data/oddsportal_team_aliases.json` | Aliases Bolivia (52 entries, 27 teams) |
| `scripts/scrape_oddsportal_bolivia.py` | Scraper multi-temporada con checkpoint |

### 6.15 Nota: Equipos Menores y Rotacion

Bolivia tiene alta rotacion de equipos entre primera y segunda division. 27 equipos unicos en 7 temporadas, pero solo 14-18 por temporada. Equipos como ABB (6154), San Juan FC (6153), y Vaca Diez (4354) aparecen solo en temporadas recientes. Esto no afecta la ingesta (aliases cubren todos los nombres).

---

## 7. Chile (265)

### 7.1 Ficha Tecnica

| Campo | Valor |
|-------|-------|
| **League ID** | 265 |
| **Nombre** | Chile Primera Division |
| **Formato** | Liga regular (30 fechas) + liguilla clasificacion/descenso |
| **Equipos** | 16 (2022+), 18 (2019-2021) |
| **N partidos en DB** | 1,760 FT (2019-2026) |
| **N partidos testeados** | 1,566 (filtro 2020-01-01+) |
| **Train / Test split** | 1,252 / 314 (split date: 2024-09-25) |
| **Distribucion resultados** | Home 51.9%, Draw 19.7%, Away 28.3% |
| **Brier naive** | ~0.637 (A0_baseline_17) |
| **Dificultad** | MEDIA — tasa de empates baja (19.7%), home advantage fuerte (51.9%) |
| **Odds coverage** | 97.2% global (1,712/1,760). 96-98% por temporada. OddsPortal avg closing |
| **xG coverage** | 0% — sin proveedor disponible (FotMob no cubre Chile) |
| **Market Anchor** | VIABLE (gap +3.1%) |

### 7.2 Campeon: N9_odds_ultimate (22 features)

```
Features (22):
  home_goals_scored_avg, home_goals_conceded_avg, home_shots_avg, home_corners_avg,
  home_matches_played, away_goals_scored_avg, away_goals_conceded_avg, away_shots_avg,
  away_corners_avg, away_matches_played, goal_diff_avg, abs_attack_diff, abs_defense_diff,
  abs_shots_diff, elo_home, elo_away, elo_diff,
  odds_home, odds_draw, odds_away, odds_implied_draw, odds_overround

  Brier: 0.60634 CI95[-, -]
  N_test: 306
```

**Nota**: El campeon absoluto es N9, pero con 22 features. El segundo lugar N2_odds_m2_combo (0.61268, 12 features) ofrece mejor balance complejidad/rendimiento. Sin odds, H5_elo_gw_defense (0.61186, 5 features) es el lider y mas parsimonioso.

### 7.3 Campeon Sin Odds: H5_elo_gw_defense (5 features)

```
Features (5):
  elo_gw_home, elo_gw_away, elo_gw_diff,
  home_goals_conceded_avg, away_goals_conceded_avg

  Brier: 0.61186 CI95[0.59035, 0.63710]
  N_test: 314
  Accuracy: 50.5%
```

H5 es notable porque con solo 5 features (Elo game-week + defensa) logra un Brier casi identico al campeon con odds (delta solo 0.005). Esto sugiere que en Chile, Elo captura casi toda la señal util y las odds agregan poco mas alla del Elo.

### 7.4 Top-10 Standard Lab

```
 #  Test                         Brier   Feats  Tipo
 1  N9_odds_ultimate           0.60634     22   odds+all
 2  H5_elo_gw_defense          0.61186      5   elo_gw+defense
 3  M0_h0_opp_adj              0.61264     10   hybrid+opp_adj
 4  N2_odds_m2_combo           0.61268     12   odds+interactions
 5  M2_h0_interactions         0.61302      9   hybrid+interactions
 6  M6_defense_elo_kimi        0.61328     14   defense+elo+kimi
 7  N5_odds_kimi_all           0.61412     15   odds+kimi
 8  H3_defense_matchup_elo     0.61444      9   defense+matchup+elo
 9  H4_kitchen_sink            0.61558     17   all features
10  M4_smart_minimal           0.61642     13   curated minimal
```

**Patron**: Los 10 mejores estan en un rango de solo 0.01 Brier. Elo + defensa es el nucleo comun; odds dan un boost marginal. Matchup e interaction features tambien son utiles.

### 7.5 Analisis SHAP (9 escenarios)

```
Test                           Brier  Top-3 Features (SHAP)
S1_baseline_odds             0.61733  odds_home=0.123, odds_away=0.107, home_matches_played=0.044
S4_m2_interactions           0.61839  home_matches_played=0.119, elo_x_season=0.071, elo_x_defense=0.048
S3_defense_elo               0.62073  elo_diff=0.081, elo_home=0.061, elo_away=0.052
S6_power_5                   0.62641  elo_diff=0.093, opp_rating_diff=0.056, home_goals_conceded_avg=0.051
S2_elo_odds                  0.62699  odds_home=0.157, odds_away=0.107, odds_draw=0.058
S0_baseline_17               0.64214  home_matches_played=0.089, away_matches_played=0.066, home_shots_avg=0.046
S7_abe_elo                   0.65523  league_draw_rate=0.108, elo_diff=0.045, elo_home=0.042
S8_abe_elo_odds              0.65762  league_draw_rate=0.125, odds_home=0.107, odds_away=0.105
S5_xg_elo                      SKIP  (0 matches con xG)
```

**Hallazgos SHAP**:
- **odds_home + odds_away** dominan en S1/S2 (combinados ~47% share en S2)
- **home_matches_played** es sorprendentemente fuerte: SHAP 0.119 en S4, 0.089 en S0. Proxy de experiencia/rodaje
- **elo_diff** lidera en escenarios sin odds (S3, S6)
- **league_draw_rate DESTRUYE** S7/S8: SHAP 0.108-0.125, pero Brier 0.655-0.658. Overfitting masivo — esta feature NO funciona en Chile (tasa de empates 19.7%, muy baja)
- **xG no disponible**: S5 skip completo

### 7.6 Optimizacion Optuna (17 escenarios)

```
Test                         Brier     CV Brier  Depth    LR    N_est
FIXED_baseline             0.63723         —       3   0.028    114
O0_elo_gw_defense          0.63730    0.66242      2   0.013     69
O1_elo_gw_form             0.63762    0.66241      2   0.012     74
OF_abe_elo_odds            0.63889    0.65261      2   0.012    145
O5_m2_interactions         0.63959    0.66586      2   0.010     64
OA_only_elo                0.63973    0.66359      2   0.010     85
O3_elo_k20                 0.64051    0.66132      2   0.017     61
O8_smart_minimal           0.64167    0.66466      2   0.011     70
O4_defense_elo_kimi        0.64283    0.66388      4   0.012     50
O2_defense_form_elo        0.64348    0.66516      2   0.016     50
O6_efficiency_elo          0.64500    0.66405      4   0.012     50
O7_all_abe_elo             0.64715    0.66518      2   0.017     67
O9_baseline_17             0.65453    0.66558      3   0.010     50
OB-OE_xg*                    SKIP     (sin xG)
```

**Hallazgos Optuna**:
- **FIXED_baseline (0.63723) empata con O0** (0.63730): los hyperparams de produccion ya estan bien calibrados para Chile
- Optuna converge a **depth=2** en 10/13 tests (Chile se beneficia de arboles simples)
- **Learning rate bajo** (0.010-0.017) con 50-85 estimators
- **OF_abe_elo_odds** tiene el mejor CV (0.65261) pero en test pierde vs FIXED. Las odds mejoran generalizacion (CV) pero no traducen consistentemente al test set en Optuna

### 7.7 Mercado vs Modelo

```
Market baseline:  0.57604 (N=306, de-vigged OddsPortal avg)
Best model:       0.60634 (N9_odds_ultimate, 22 features)
FAIR gap:         +0.03086 (+3.1%)
```

El mercado gana por 3.1 puntos Brier. Patron consistente con LATAM pero gap menor que Argentina (+2.8%) y comparable a Colombia (+2.0%).

### 7.8 Comparacion 7 Ligas

```
Liga          Market   Best Model   FAIR gap   Campeon (sin odds)
Bolivia       0.5071   0.5168       +0.97%     F2_matchup_form_elo (0.5404)
Chile         0.5760   0.6063       +3.09%     H5_elo_gw_defense (0.6119)
Colombia      0.6091   0.6245       +2.00%     O1_elo_gw_form (0.6280)
Argentina     0.6348   0.6585       +2.83%     S7_abe_elo (0.6585)
Ecuador       0.6138   0.6174       +0.59%     N7_odds_power7 (0.6174)
Venezuela     0.6088   0.6153       +1.07%     H5_elo_gw_defense (0.6118)
Peru          0.6126   0.6223       +1.58%     H5_elo_gw_defense (0.6162)
```

Chile tiene el segundo mercado mas eficiente (Brier 0.576) despues de Bolivia (0.507). La Brier absoluta del modelo (0.606) sugiere predictibilidad media — mejor que Argentina/Colombia pero peor que Bolivia/Ecuador.

### 7.9 Features Catastróficos: draw_aware

```
K4_draw_aware              0.72499    4 features   (PEOR test de 91)
K5_draw_aware_elo          0.68938    7 features   (3ro peor)
K8_all_abe                 0.68176   15 features   (4to peor)
M3_h0_draw_aware           0.67931    9 features   (5to peor)
```

**draw_aware es TOXICO en Chile**. Brier 0.725 es dramaticamente peor que el naive (0.637). Los features draw_tendency_home/away, draw_elo_interaction, y league_draw_rate generan overfitting severo en una liga con solo 19.7% de empates. Cualquier configuracion que incluya estos features (K4, K5, K8, M3, S7, S8) se degrada.

**Leccion**: features de empate solo son utiles en ligas con tasa de empates >= 25%. Chile, con 19.7%, tiene empates demasiado raros para que el modelo aprenda patrones confiables.

### 7.10 Home Advantage

Chile tiene el home advantage mas fuerte de las 7 ligas LATAM: 51.9% victorias locales. Esto explica por que `home_matches_played` tiene SHAP alto (0.089-0.119): es un proxy de familiaridad con el estadio/altitud. Equipos como Antofagasta, Cobresal (desierto), Coquimbo (costa) tienen ventajas geográficas marcadas.

### 7.11 Recomendaciones

1. **Market Anchor alpha=1.0** para Chile: gap +3.1% confirma que el mercado es superior
2. **No incluir draw_aware features** en produccion para Chile
3. **Priorizar Elo game-week + defense** (H5) como base; odds como boost marginal
4. **xG pendiente**: FotMob fixture parser no soporta Chile aun. Cuando se resuelva, re-evaluar con escenarios P0-P9
5. **Re-evaluar** cuando la temporada 2026 tenga >= 100 partidos FT (actualmente solo 16)

### 7.12 Scraping OddsPortal

- **Script**: `scripts/scrape_oddsportal_chile.py` (Playwright, headless Chromium)
- **Temporadas**: 2019-2026 (8 archivos)
- **Slug dual**: `primera-division-{year}` (2019-2024) → `liga-de-primera-2025` (2025) → `liga-de-primera` (2026). Liga renombrada en OddsPortal en 2025
- **Total scrapeado**: 1,756 entradas brutas → 1,728 limpias (28 noise entries removidos)
- **Noise removido**: 16 canc./award., 1 false extraction (Football/Chile), 11 de equipos no en DB (Temuco, S. Morning, San Felipe, San Luis, CD Santa Cruz)

### 7.13 Aliases OddsPortal

48 entries, 28 teams en seccion `ChilePrimeraDivision` de `data/oddsportal_team_aliases.json`.

Teams notables con multiples aliases:
- Coquimbo → Coquimbo Unido (4171)
- Everton → Everton de Vina/Viña (4176)
- D. Concepcion → Concepcion (6148)
- Rangers → Rangers de Talca (6149)
- U. Catolica → Universidad Catolica (4180)
- U. De Chile → Universidad de Chile (4179)
- U. Espanola → Union Espanola (4184)

### 7.14 Ingesta

```
GATE: 100% resolved (28 teams)
Matched: 1,712/1,728 (99.1%)
Score mismatches: 1
No odds: 3
Updated: 1,712 matches
```

12 no_db_match: playoffs/liguilla no en DB (Rangers vs Limache, Recoleta vs Rangers 2024) + 1 match reciente 2026.

### 7.15 Archivos

| Archivo | Contenido |
|---------|-----------|
| `scripts/output/lab/feature_lab_results.json` | Lab v2, 91/101 tests (10 xG skip) |
| `scripts/output/lab/shap_analysis_265.json` | SHAP, 8/9 escenarios (1 xG skip) |
| `scripts/output/lab/feature_lab_results_optuna.json` | Optuna, 13/17 tests (4 xG skip) |
| `scripts/output/lab/lab_data_265.csv` | Dataset cacheado (1,566 rows) |
| `data/oddsportal_raw/chile-primera-division_*.json` | OddsPortal scrapes 2019-2026 (1,728 matches) |
| `data/oddsportal_team_aliases.json` | Aliases Chile (48 entries, 28 teams) |
| `scripts/scrape_oddsportal_chile.py` | Scraper multi-temporada con dual slug |

### 7.16 Nota: Equipos No Rastreados

5 equipos que aparecen en OddsPortal Chile pero NO estan en nuestra DB: Deportes Temuco, Santiago Morning (S. Morning), San Felipe, San Luis, CD Santa Cruz. Son equipos que descendieron antes de 2019 o juegan en segunda division. Sus partidos fueron filtrados del JSON antes de la ingesta.

---

## Apendice: Como Agregar una Liga

1. Ejecutar extraccion: `python3 scripts/feature_lab.py --extract --league <ID>`
2. Run standard lab: `python3 scripts/feature_lab.py --league <ID>` (101 tests, fixed params)
3. Identificar top-10 y correr SHAP: `python3 scripts/feature_lab.py --shap --league <ID>`
4. Correr Optuna en top performers: `python3 scripts/feature_lab.py --optuna --league <ID>`
5. Si hay fecha minima de datos confiables: agregar `--min-date YYYY-MM-DD`
6. Documentar resultados en nueva seccion de este archivo
7. Actualizar indice de ligas

**Frecuencia de re-evaluacion sugerida**: cada 6 meses o cuando cambie el formato del torneo.
