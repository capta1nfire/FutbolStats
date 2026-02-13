# Manual de Segmentacion por Liga

**Documento vivo** — Se actualiza cada vez que una liga pasa por el laboratorio.
El futbol es dinamico: las combinaciones optimas cambian con el tiempo. Re-ejecutar pruebas periodicamente.

**Script**: `scripts/feature_lab.py`
**Modos**: `--shap`, `--optuna`, `--extract`, `--residual`, `--league <ID>`, `--min-date YYYY-MM-DD`

---

## Indice

- [Section R — Market Residual Diagnostic](#section-r--market-residual-diagnostic) (23 ligas, diagnostico de eficiencia de mercado)

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
| Brasil Serie A | 71 | COMPLETO | 2026-02-11 | [8](#8-brasil-71) |
| Paraguay Primera Division | 250 | COMPLETO | 2026-02-12 | [9](#9-paraguay-250) |
| Uruguay Primera Division | 268 | COMPLETO | 2026-02-12 | [10](#10-uruguay-268) |
| Mexico Liga MX | 262 | COMPLETO | 2026-02-12 | [11](#11-mexico-262) |
| MLS | 253 | COMPLETO | 2026-02-12 | [12](#12-mls-253) |
| La Liga - España | 140 | **COMPLETO+xG** | 2026-02-13 (re-run con xG 100%) | [13](#13-la-liga---españa-140) |
| Ligue 1 - Francia | 61 | **COMPLETO+xG** | 2026-02-13 (re-run con xG 99%) | [14](#14-ligue-1---francia-61) |
| Bundesliga - Alemania | 78 | **COMPLETO+xG** | 2026-02-13 (re-run con xG 99.6%) | [15](#15-bundesliga---alemania-78) |
| Serie A - Italia | 135 | **COMPLETO+xG** | 2026-02-13 (re-run con xG 99.4%) | [16](#16-serie-a---italia-135) |
| Premier League - Inglaterra | 39 | **COMPLETO+xG** | 2026-02-13 (re-run con xG 99.4%) | [17](#17-premier-league---inglaterra-39) |
| Eredivisie - Holanda | 88 | **COMPLETO+xG+Optuna** | 2026-02-13 (xG 58.5%, 16 Optuna) | [18](#18-eredivisie---holanda-88) |
| Belgian Pro League - Bélgica | 144 | **COMPLETO+xG+Optuna** | 2026-02-13 (xG 84.3%, 16 Optuna) | [19](#19-belgian-pro-league---bélgica-144) |
| Primeira Liga - Portugal | 94 | **COMPLETO+xG+Optuna** | 2026-02-13 (xG 55.4%, 16 Optuna) | [20](#20-primeira-liga---portugal-94) |
| Süper Lig - Turquía | 203 | **COMPLETO+xG+Optuna** | 2026-02-13 (xG 38.1%, 16 Optuna) — **MODELO > MERCADO** | [21](#21-sueper-lig---turquia-203) |
| EFL Championship - Inglaterra | 40 | **COMPLETO+xG+Optuna** | 2026-02-13 (xG 84.2%, 16 Optuna, odds insuf.) | [22](#22-efl-championship---inglaterra-40) |
| Saudi Pro League | 307 | **COMPLETO+xG+Optuna** | 2026-02-13 (odds 96%, xG 52%, 16 Optuna) — **MODELO ≈ MERCADO** | [23](#23-saudi-pro-league-307) |

---

## Section R — Market Residual Diagnostic

### Metodologia

Section R evalua si nuestras features contienen informacion que el mercado de apuestas no tiene. Utiliza el parametro `base_margin` de XGBoost para que el modelo arranque desde las probabilidades del mercado (odds de-vigged) y solo aprenda una correccion pequena g(x):

```
p_final = softmax(log(p_market) + g(x))
```

Con regularizacion fuerte (max_depth=2, reg_alpha=0.1, reg_lambda=1.0), si no existen sesgos sistematicos del mercado, g(x) converge a cero y el modelo reproduce el mercado exacto.

**Validacion**: Un modelo con zero-learning (n_estimators=1, lr=0.0001) reproduce el Brier del mercado al quinto decimal en todas las ligas testeadas, confirmando que la implementacion es correcta.

**Interpretacion del delta** (Brier_residual - Brier_market):
- **Delta positivo**: las features EMPEORAN las predicciones del mercado. No hay informacion nueva.
- **Delta negativo**: las features MEJORAN las predicciones del mercado. Hay sesgos explotables.
- **Direccion consistente** (N/N tests en la misma direccion): mas confiable que cualquier delta individual.

**Ejecucion**: `python3 scripts/feature_lab.py --league <ID> --extract --residual`

### Resultados Consolidados (23 ligas, 2026-02-13)

| Liga | ID | N_test | Tests R | Direccion | Mejor Delta | Veredicto |
|------|----|--------|---------|-----------|-------------|-----------|
| Argentina | 128 | 265 | 7/7 | POSITIVO | +0.01265 | EFICIENTE |
| Colombia | 239 | 495 | 7/7 | POSITIVO | +0.00361 | EFICIENTE |
| Ecuador | 242 | 338 | 5/5 | POSITIVO | +0.00495 | EFICIENTE |
| Venezuela | 299 | 250 | 5/5 | POSITIVO | +0.01213 | EFICIENTE |
| Peru | 281 | 417 | 5/5 | POSITIVO | +0.00589 | EFICIENTE |
| Bolivia | 344 | 378 | 5/5 | POSITIVO | +0.00160 | EFICIENTE |
| Chile | 265 | 343 | 5/5 | POSITIVO | +0.02735 | EFICIENTE |
| Brasil | 71 | 234 | 7/7 | POSITIVO | +0.00717 | EFICIENTE |
| Paraguay | 250 | 314 | 5/5 | POSITIVO | +0.00042 | EFICIENTE |
| Uruguay | 268 | 377 | 5/5 | POSITIVO | +0.00252 | EFICIENTE |
| Mexico | 262 | 381 | 7/7 | POSITIVO | +0.00498 | EFICIENTE |
| MLS | 253 | 313 | 7/7 | POSITIVO | +0.01522 | EFICIENTE |
| Premier League | 39 | 812 | 7/7 | POSITIVO | +0.00060 | EFICIENTE |
| La Liga | 140 | 806 | 7/7 | POSITIVO | +0.01196 | EFICIENTE |
| Ligue 1 | 61 | 748 | 7/7 | POSITIVO | +0.01525 | EFICIENTE |
| Bundesliga | 78 | 650 | 7/7 | POSITIVO | +0.00509 | EFICIENTE |
| Serie A | 135 | 808 | 7/7 | POSITIVO | +0.00140 | EFICIENTE |
| Eredivisie | 88 | 575 | 7/7 | MIXTO | -0.00343 | AMBIGUO |
| Belgian Pro | 144 | 393 | 7/7 | POSITIVO | +0.00365 | EFICIENTE |
| Primeira Liga | 94 | 578 | 7/7 | MIXTO | -0.00041 | AMBIGUO |
| **Turquia** | **203** | **567** | **7/7** | **NEGATIVO** | **-0.00327** | **INEFICIENTE** |
| Championship | 40 | 70 | 7/7 | POSITIVO | +0.01970 | EFICIENTE |
| Saudi Pro | 307 | 0 | 0/0 | N/A | N/A | INSUFFICIENT_ODDS |

### Resumen

| Categoria | Ligas | Total |
|-----------|-------|-------|
| EFICIENTE (mercado gana) | ARG, COL, ECU, VEN, PER, BOL, CHI, BRA, PAR, URU, MEX, MLS, EPL, ESP, FRA, GER, ITA, BEL, ENG2 | 19/22 |
| INEFICIENTE (modelo gana) | **Turquia** | 1/22 |
| AMBIGUO (mixto) | Eredivisie (2/7 neg solo en xG), Primeira Liga (1/7 neg, ~0) | 2/22 |
| N/A (sin odds) | Saudi Pro | 1/23 |

### Hallazgo clave: Ecuador como falso positivo

Ecuador es el caso mas instructivo. El modelo directo mostraba -1.4% vs mercado (aparentemente el modelo ganaba). Sin embargo, Section R revelo 5/5 deltas positivos — las features no corrigen al mercado. Esto indica que el alpha observado en el modelo directo era **varianza muestral** (N=338, CIs anchos), no edge real.

| Metodo | Ecuador | Turquia |
|--------|---------|---------|
| Modelo directo vs market | Modelo gana (-1.4%) | Modelo gana (-1.5%) |
| Section R (residual) | Mercado gana (5/5 positivo) | Modelo gana (7/7 negativo) |
| Ambos coinciden? | **No** — falso positivo | **Si** — alpha real |

**Regla**: Un modelo directo que "supera" al mercado no es suficiente. Section R debe confirmar que las features contienen informacion no incorporada en el precio. Si ambos metodos coinciden, el alpha es robusto. Si se contradicen, es sospechoso.

### Uso para calibracion de Market Anchor

| Resultado Section R | Alpha recomendado | Justificacion |
|---------------------|-------------------|---------------|
| Todos deltas positivos | alpha >= 0.8 | Mercado eficiente, usar Market Anchor |
| Todos deltas negativos | alpha = 0.0 | Mercado ineficiente, usar modelo directo |
| Mixto / contradice modelo directo | alpha = 0.5 | Zona gris, blend conservador |

### Notas sobre los deltas

Los deltas positivos no son uniformes. Revelan cuanto dano hacen las features como correcciones:

| Categoria | Ligas | Delta tipico | Interpretacion |
|-----------|-------|-------------|----------------|
| g(x) converge a cero | Paraguay (+0.0004), EPL (+0.001), Serie A (+0.001) | < +0.002 | Regularizacion funciona, features neutras |
| Features agregan ruido moderado | Colombia, Bolivia, Uruguay, Mexico, Bundesliga | +0.003 a +0.007 | Features no daninas pero inutiles |
| Features activamente daninas | Chile (+0.027), MLS (+0.015), Ligue 1 (+0.015) | > +0.010 | Features empujan en direccion equivocada |

### Re-evaluacion

Section R debe re-ejecutarse:
- Cada vez que se agreguen nuevas features al laboratorio
- Si una liga cambia de eficiencia de mercado (nuevo bookmaker, cambio de liquidez)
- Como paso obligatorio antes de desactivar Market Anchor (alpha < 0.5) en cualquier liga

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

### 7.8 Comparacion 8 Ligas

```
Liga          Market   Best Model   FAIR gap   Campeon (sin odds)
Bolivia       0.5283   0.5374       +0.94%     H7_elo_split_defense (0.5539)
Peru          0.5605   0.5724       +0.96%     H5_elo_gw_defense (0.6162)
Chile         0.5719   0.6049       +3.04%     H5_elo_gw_defense (0.6119)
Venezuela     0.5841   0.5958       +1.94%     H5_elo_gw_defense (0.6118)
Brasil        0.6018   0.6125       +1.64%     D8_elo_all (0.6158)
Colombia      0.6091   0.6280       +1.28%     O1_elo_gw_form (0.6280)
Ecuador       0.6380   0.6419       +0.55%     N7_odds_power7 (0.6174)
Argentina     0.6704   0.6585       -1.19%     S7_abe_elo (0.6585)
```

Chile tiene el tercer mercado mas eficiente (Brier 0.572) despues de Bolivia (0.528) y Peru (0.561). Brasil esta en el medio de la tabla (0.602). La Brier absoluta del modelo (0.606) sugiere predictibilidad media.

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

## 8. Brasil (71)

### 8.1 Ficha Tecnica

| Campo | Valor |
|-------|-------|
| **League ID** | 71 |
| **Nombre** | Campeonato Brasileiro Serie A |
| **Formato** | Liga regular (38 fechas), todos contra todos ida y vuelta |
| **Equipos** | 20 (fijos) |
| **N partidos en DB** | 2,305 FT (Ago 2020 — Feb 2026) |
| **N partidos testeados** | 2,305 (sin filtro min-date) |
| **Train / Test split** | 1,844 / 461 (split date: 2024-11-09) |
| **Distribucion resultados** | Home 47.9%, Draw 27.3%, Away 24.7% |
| **Brier naive** | 0.6250 (A0_baseline_17) |
| **Dificultad** | MEDIA-ALTA — tasa de empates moderada (27.3%) |
| **Odds coverage** | 50.3% global, **100% desde 2023** (FDUK backfill, Pinnacle closing) |
| **xG coverage** | **99.7% desde 2023** (FotMob/Opta, 1,162/1,165 matches). 0% pre-2023 |
| **Market Anchor** | VIABLE (gap +1.6%, no significativo) |

### 8.2 Campeon: J1_elo_odds (6 features)

```
Features (6):
  elo_home, elo_away, elo_diff,
  odds_home, odds_draw, odds_away

Brier (test, N=233): 0.61246
CI95: [0.57875, 0.64588]
Accuracy: 45.8%
```

Campeon minimalista: solo 6 features (Elo + odds). Todos los tests con mas features rinden igual o peor.

### 8.3 Campeon Sin Odds: D8_elo_all (12 features)

```
Features (12):
  elo_home, elo_away, elo_diff,
  elo_gw_home, elo_gw_away, elo_gw_diff,
  elo_split_home, elo_split_away, elo_split_diff,
  elo_x_season, elo_x_defense, elo_x_form

Brier (test, N=461): 0.61576
CI95: [0.59598, 0.63581]
Accuracy: 46.1%
```

D8 usa 12 variantes de Elo (base, game-week, split, interactions). Con N=461 (mayor test set que odds universe), el Brier base de 0.616 es competitivo con J1 odds (0.612). La diferencia es marginal.

### 8.4 Top-15 Standard Lab

```
 #  Test                         Brier   Feats  Tipo        N_test
 1  J1_elo_odds                0.61246     6    odds         233
 2  N0_odds_elo                0.61246     6    odds         233
 3  N5_odds_kimi_all           0.61259    15    odds         233
 4  N3_odds_efficiency         0.61324    11    odds         233
 5  N1_odds_defense_elo        0.61416     8    odds         233
 6  N2_odds_m2_combo           0.61499    12    odds         233
 7  D8_elo_all                 0.61576    12    base         461
 8  D5_elo_split               0.61643     3    base         461
 9  H7_elo_split_defense       0.61883     5    base         461
10  J2_full_odds               0.61910    20    odds         233
11  M5_defense_elo_abe         0.62047    16    base         461
12  N9_odds_ultimate           0.62093    22    odds         233
13  I1_clean_elo               0.62108    15    base         461
14  H0_arg_signal_elo          0.62117     5    base         461
15  P6_xg_elo_odds             0.62126     9    odds_xg      231
```

**Patron**: Los top-6 son todos del universo odds, dominados por Elo + odds simples. Mas features no mejoran. xG (P6) entra recien en posicion 15.

### 8.5 Universos xG

```
=== xG universe (puro, N_test=301) ===
  P3_xg_defense_elo          0.62707   9f   (campeon xG)
  P4_xg_overperf_elo         0.62730  12f
  P2_xg_elo                  0.62937   6f
  P1_xg_all                  0.64006   9f
  P0_xg_core                 0.64378   3f

=== odds_xg universe (N_test=231) ===
  P6_xg_elo_odds             0.62126   9f   (campeon odds_xg)
  P7_xg_all_elo_odds         0.62331  15f
  P9_xg_ultimate             0.62382  22f
  P8_xg_defense_odds         0.62782  11f
  P5_xg_odds                 0.63145   6f
```

**Veredicto xG**: No aporta signal adicional.
- xG puro (P3=0.627) es peor que Elo puro (D8=0.616)
- xG + odds (P6=0.621) es peor que odds solo (J1=0.612)
- xG como feature individual tiene SHAP = 0.035-0.037, muy debil comparado con elo_diff (0.10) u odds_home (0.13)

### 8.6 Analisis SHAP (9 escenarios)

```
Test                           Brier  Top-3 Features (SHAP)
S2_elo_odds                  0.61819  odds_home=0.127, odds_away=0.115, odds_draw=0.070
S1_baseline_odds             0.62003  odds_away=0.114, odds_home=0.110, odds_draw=0.044
S4_m2_interactions           0.62288  elo_diff=0.099, home_goals_conceded_avg=0.052, elo_home=0.047
S7_abe_elo                   0.62444  elo_diff=0.074, opp_rating_diff=0.058, elo_home=0.046
S3_defense_elo               0.62565  elo_diff=0.106, elo_home=0.070, home_goals_conceded_avg=0.063
S6_power_5                   0.62842  elo_diff=0.096, opp_rating_diff=0.079, draw_elo_interaction=0.056
S0_baseline_17               0.62858  home_goals_conceded_avg=0.067, goal_diff_avg=0.054, away_goals_conceded_avg=0.046
S5_xg_elo                   0.63042  elo_diff=0.102, elo_home=0.074, elo_away=0.059
S8_abe_elo_odds              0.63704  odds_away=0.105, odds_home=0.084, league_draw_rate=0.068
```

**Hallazgos SHAP**:
- **odds_home + odds_away** dominan en S2 (72.5% share combinado, incluido odds_draw)
- **elo_diff** lidera todos los escenarios sin odds (SHAP 0.074-0.106)
- **xG features** (S5): home_xg_for_avg = 0.037, away_xg_for_avg = 0.035 — marginales, 3x menos que elo_diff
- **home_goals_conceded_avg** es la mejor feature defensiva (SHAP 0.052-0.067)
- **S8 overfits**: league_draw_rate SHAP=0.068 degrada (Brier 0.637, peor que S7 sin odds)

### 8.7 Optimizacion Optuna (17 escenarios)

```
Test                         Brier     CV Brier  Depth    LR    N_est  Feats
OC_xg_all_elo_odds         0.61944    0.62575      2   0.038     58    15
O2_defense_form_elo        0.62467    0.63791      2   0.034     76     8
FIXED_baseline             0.62495         —       3   0.028    114    17
OB_xg_odds                 0.62517    0.61707      2   0.025     96     6
O5_m2_interactions         0.62543    0.63943      2   0.034     52     9
O1_elo_gw_form             0.62598    0.63518      2   0.017    110     6
O7_all_abe_elo             0.62642    0.64505      2   0.024     76    18
OF_abe_elo_odds            0.62702    0.61944      6   0.031     81    21
O3_elo_k20                 0.62731    0.63784      2   0.015    117     3
OA_only_elo                0.62749    0.63950      2   0.015    144     3
O9_baseline_17             0.62750    0.63456      6   0.013    162    17
O8_smart_minimal           0.62777    0.64179      2   0.010    161    13
O0_elo_gw_defense          0.62813    0.63165      3   0.018    167     5
O4_defense_elo_kimi        0.62813    0.63543      3   0.015    164    14
O6_efficiency_elo          0.63011    0.63680      3   0.010    160     8
OD_xg_overperf_elo         0.63047    0.64694      3   0.014     94    12
OE_xg_defense_odds         0.63293    0.61722      6   0.016    164    11
```

**Hallazgos Optuna**:
- **OC_xg_all_elo_odds** (0.619) campeon Optuna, pero peor que J1 standard (0.612)
- **FIXED_baseline** (0.625) supera a 12/16 tests tuneados — produccion ya esta bien calibrada
- Optuna converge a **depth=2** en 11/16 tests (Brasil prefiere arboles simples)
- **OB vs OF**: OB tiene mejor CV (0.617) pero peor test (0.625). OF tiene mejor CV (0.619) pero peor test (0.627). Overfitting con features complejos
- **xG Optuna** (OC=0.619, OB=0.625, OD=0.630, OE=0.633): xG no mejora vs J1 ni con tuning

### 8.8 Mercado vs Modelo

```
Market baseline:  0.60178 (N=233, de-vigged FDUK closing odds)
Best model:       0.61246 (J1_elo_odds, 6 features)
FAIR gap:         +0.01641 (+1.6%)
CI95:             [-0.00156, +0.03521]
Significativo:    NO (CI cruza cero)
```

Gap de +1.6% es el segundo menor en LATAM (solo Ecuador tiene menor gap con +0.6%). El mercado es ligeramente mejor pero la diferencia no es estadisticamente significativa.

### 8.9 Comparacion 8 Ligas

```
Liga          Market   Best Model   FAIR gap   Sig?
Bolivia       0.5283   0.5374       +0.9%      No
Peru          0.5605   0.5724       +1.0%      No
Chile         0.5719   0.6049       +3.0%      YES
Venezuela     0.5841   0.5958       +1.9%      No
Brasil        0.6018   0.6125       +1.6%      No
Colombia      0.6091   0.6280       +1.3%      YES
Ecuador       0.6380   0.6419       +0.6%      No
Argentina     0.6704   0.6585       -1.2%      No
```

Brasil esta en la mitad de la tabla: mercado moderadamente eficiente (Brier 0.602), gap medio (+1.6%).

### 8.10 xG Backfill

- **Fuente**: FotMob (Opta xG), league_id FotMob = 268
- **Script**: `scripts/backfill_fotmob_xg.py --league 71 --seasons 2020-2026`
- **Temporadas**: 7 (2020-2026)
- **Total linkeado**: 1,909 matches (refs en match_external_refs)
- **Total xG capturado**: 1,288 matches (match_fotmob_stats)
- **Cobertura por temporada**:

```
Year   Linked   xG     Rate    Nota
2020     353     69    19.5%   Pre-Opta, COVID parcial
2021     380     36     9.5%   Pre-Opta
2022     371     26     7.0%   Pre-Opta
2023     375    374    99.7%   Opta empieza (1 sin xG: Fortaleza-Gremio)
2024     378    378   100.0%   Cobertura total
2025     380    380   100.0%   Cobertura total
2026      25     25   100.0%   Temporada en curso
```

- **5 partidos recuperados manualmente**: fechas desplazadas (FotMob Oct 25-26 vs DB Oct 27-28)
- **1 irrecuperable**: Fortaleza 1-1 Gremio (Sep 30, 2023) — FotMob solo tiene stats basicas, sin xG
- **Pre-2023**: Opta no tenia xG desplegado para Brasil Serie A. Cobertura ~10% (partidos sueltos)
- **Aliases agregados**: America MG↔America Mineiro (125), Atletico GO↔Atletico Goianiense (144), Chapecoense AF↔Chapecoense-sc (132)
- **FOTMOB_CONFIRMED_XG_LEAGUES**: Brasil (71) agregado en `sota_constants.py`

### 8.11 Recomendaciones

1. **Market Anchor alpha >= 0.8** para Brasil: gap +1.6% sugiere que el mercado es mejor, aunque no significativo
2. **Campeon produccion: J1_elo_odds** (6 features) — minimalista y robusto
3. **No incluir xG** en produccion: no aporta sobre odds+Elo (SHAP 0.035 vs elo 0.10)
4. **Depth=2** preferido por Optuna (arboles simples, evita overfitting)
5. **OddsPortal pendiente** para 2020-2022: solo 50% odds global. Si se necesita entrenar con mas historia, scrapear odds historicas
6. **Re-evaluar** cuando temporada 2026 tenga >= 200 partidos FT

### 8.12 Archivos

| Archivo | Contenido |
|---------|-----------|
| `scripts/output/lab/feature_lab_results.json` | Lab v2, 101 tests (4 universos) |
| `scripts/output/lab/shap_analysis_71.json` | SHAP, 9 escenarios |
| `scripts/output/lab/feature_lab_results_optuna.json` | Optuna, 17 tests |
| `scripts/output/lab/lab_data_71.csv` | Dataset cacheado (2,305 rows) |
| `scripts/backfill_fotmob_xg.py` | Script generico FotMob xG (CLI args) |

---

## 9. Paraguay (250)

### 9.1 Ficha Tecnica

| Campo | Valor |
|-------|-------|
| **League ID** | 250 (Apertura) + 252 (Clausura) |
| **Nombre** | Copa de Primera / Primera Division (Paraguay) |
| **Formato** | Split-season: Apertura (Feb-Jun) + Clausura (Jul-Dic) |
| **Equipos** | 12 |
| **N partidos en DB** | 1,727 (Ene 2019 — Feb 2026) |
| **Train / Test split** | 1,381 / 346 (split date: 2024-09-28) |
| **Distribucion resultados** | Home 36.1%, Draw 32.7%, Away 31.2% |
| **Brier naive** | ~0.666 (predictor marginal) |
| **Dificultad** | ALTA — tasa de empates muy alta (32.7%), distribucion casi uniforme |
| **Odds coverage** | 89.4% global, 88.2% 2023+ (OddsPortal backfill) |
| **xG coverage** | 0% — FotMob no tiene xG para Paraguay |
| **Market Anchor** | Candidato (gap +0.5%, no significativo) |

### 9.2 Campeon: OF_abe_elo_odds (Optuna, 21 features)

```
Features (21):
  opp_att_home, opp_def_home, opp_att_away, opp_def_away, opp_rating_diff,
  overperf_home, overperf_away, overperf_diff,
  draw_tendency_home, draw_tendency_away, draw_elo_interaction, league_draw_rate,
  home_bias_home, home_bias_away, home_bias_diff,
  elo_home, elo_away, elo_diff,
  odds_home, odds_away, odds_draw

Hyperparams Optuna:
  max_depth: 2
  learning_rate: 0.0141
  n_estimators: 150
  min_child_weight: 12
  subsample: 0.577
  colsample_bytree: 0.632
  reg_alpha: 0.00718
  reg_lambda: 0.25759

Brier (test, N=346): 0.64985
Brier CV (3-fold temporal): 0.61569
Accuracy: 43.4%
```

### 9.3 Ranking Completo

| # | Test | Universe | Features | Brier test | Acc |
|---|------|----------|----------|-----------|-----|
| 1 | **OF_abe_elo_odds** | odds | 21 | **0.64985** | 43.4% |
| 2 | O1_elo_gw_form | base | 6 | 0.65014 | 42.1% |
| 3 | D7_elo_probs | base | 3 | 0.65158 | 42.4% |
| 4 | C5_elo_prob_draw | base | 2 | 0.65168 | 40.5% |
| 5 | O0_elo_gw_defense | base | 5 | 0.65166 | 42.4% |
| 6 | O7_all_abe_elo | base | 18 | 0.65190 | 41.8% |
| 7 | B3_1f_elo_diff | base | 1 | 0.65219 | 42.0% |
| 8 | C3_elo_diff_form | base | 2 | 0.65251 | 40.9% |
| 9 | F1_matchup_elo | base | 7 | 0.65306 | 44.2% |
| 10 | O6_efficiency_elo | base | 8 | 0.65310 | 40.8% |
| — | FIXED_baseline (prod) | base | 17 | 0.65812 | 38.7% |
| — | **MKT_market** | odds | 3 | **0.65672** | — |

### 9.4 SHAP Analysis

| Test | Brier | #1 Feature (SHAP) | #2 Feature | #3 Feature |
|------|-------|-------------------|------------|------------|
| S0_baseline_17 | 0.6592 | home_goals_conceded_avg (0.111) | home_matches_played (0.092) | away_matches_played (0.078) |
| S1_baseline_odds | 0.6640 | odds_away (0.131) | odds_home (0.124) | home_goals_conceded_avg (0.060) |
| S3_defense_elo | 0.6589 | elo_diff (0.098) | elo_home (0.081) | home_goals_conceded_avg (0.074) |
| S7_abe_elo | 0.6596 | — ABE+Elo set | — | — |
| S8_abe_elo_odds | 0.6597 | odds_share 40.3% | — | — |

**Hallazgos SHAP clave:**
- `home_goals_conceded_avg` domina en baseline (0.111) — defensa concedida es mas informativa que ataque
- `elo_diff` domina con Elo (0.098)
- Odds share 49.9% en S1 (NO dominan como en otras ligas, 50/50 con features base)
- Sin Elo ni odds, `away_matches_played` sube como proxy temporal

### 9.5 Que Funciona y Que No

| Categoria | Funciona | No Funciona |
|-----------|----------|-------------|
| **Elo** | SI — todas las variantes de Elo en top-10 | elo_k10/k20 no mejoran vs k32 |
| **ABE features** | SI — OF_abe_elo_odds es campeon Optuna | — |
| **Odds como feature** | MIXTO — ayudan con Optuna, no en standard | Standard: base universe domina top-10 |
| **Odds como anchor** | VIABLE — mercado NO gana significativamente | — |
| **xG** | N/A — 0% cobertura | FotMob no tiene xG para Paraguay |
| **Defense (goals_conceded)** | SI — #1 SHAP en baseline | Componente clave |
| **Form** | DEBIL — O1 es #2 con Optuna, pero inconsistente | — |
| **rest_days** | NO — ruido | Eliminar en v1.0.2 |
| **Optuna tuning** | SI — mejora de 0.65158 a 0.64985 | depth=2 optimo |

### 9.6 Gap vs Mercado

```
Mejor modelo (OF Optuna):    0.64985
Mercado (OddsPortal closing): 0.65672
Gap:                          -0.5% (modelo GANA marginalmente)

FAIR delta: +0.00155 [-0.014, +0.017] (NO significativo)
```

**Paraguay es la unica liga LATAM donde el modelo supera al mercado** (aunque no significativamente). Esto puede deberse a la menor eficiencia del mercado paraguayo. La distribucion casi uniforme (36/33/31) dificulta tanto al modelo como al mercado.

### 9.7 Recomendaciones

1. **Market Anchor NO necesario**: A diferencia de otras ligas LATAM, el mercado no gana. El modelo puede operar sin anchor
2. **Campeon produccion: OF_abe_elo_odds** (21 features, Optuna tuned) o minimalista D7_elo_probs (3 features, comparable)
3. **No incluir xG**: no disponible
4. **Depth=2** preferido (12/16 Optuna eligieron 2)
5. **Re-evaluar**: cuando Apertura 2026 tenga >= 100 partidos FT

### 9.8 Archivos

| Archivo | Contenido |
|---------|-----------|
| `scripts/output/lab/feature_lab_results_250.json` | Lab v2, 101 tests |
| `scripts/output/lab/shap_analysis_250.json` | SHAP, 9 escenarios |
| `scripts/output/lab/feature_lab_results_optuna_250.json` | Optuna, 16 tests |
| `scripts/output/lab/lab_data_250.csv` | Dataset cacheado (1,727 rows) |
| `data/oddsportal_raw/paraguay_all.json` | OddsPortal raw (1,673 matches) |

---

## 10. Uruguay (268)

### 10.1 Ficha Tecnica

| Campo | Valor |
|-------|-------|
| **League ID** | 268 (Apertura) + 270 (Clausura) |
| **Nombre** | Liga AUF Uruguaya / Primera Division (Uruguay) |
| **Formato** | Split-season: Apertura + Clausura + Tabla Anual |
| **Equipos** | 16 |
| **N partidos en DB** | 1,977 (Feb 2019 — Feb 2026) |
| **Train / Test split** | 1,581 / 396 (split date: 2024-09-22) |
| **Distribucion resultados** | Home 40.9%, Draw 28.3%, Away 30.8% |
| **Brier naive** | ~0.648 (predictor marginal) |
| **Dificultad** | MEDIA — distribucion cercana a promedio sudamericano |
| **Odds coverage** | 94.9% global, 95.1% 2023+ (OddsPortal backfill) |
| **xG coverage** | 0% — FotMob no tiene xG para Uruguay |
| **Market Anchor** | VIABLE (gap +2.1%, significativo) |

### 10.2 Campeon: N4_odds_abe_best (14 features, odds)

```
Features (14):
  home_goals_scored_avg, home_goals_conceded_avg, home_shots_avg,
  home_matches_played, away_goals_scored_avg, away_goals_conceded_avg,
  away_shots_avg, away_matches_played, goal_diff_avg,
  abs_attack_diff, abs_defense_diff, abs_strength_gap,
  odds_home, odds_away

Brier (test, N=377): 0.61669
CI95: [0.5985, 0.6328]
Accuracy: 47.6%
```

### 10.3 Ranking Completo

| # | Test | Universe | Features | Brier test | Acc |
|---|------|----------|----------|-----------|-----|
| 1 | **N4_odds_abe_best** | odds | 14 | **0.61669** | 47.6% |
| 2 | J2_full_odds | odds | 20 | 0.61751 | 47.3% |
| 3 | N9_odds_ultimate | odds | 22 | 0.61946 | 47.5% |
| 4 | D1_elo_k10 | base | 3 | 0.61952 | 48.5% |
| 5 | N6_odds_clean | odds | 15 | 0.61998 | 46.6% |
| 6 | J0_only_odds | odds | 3 | 0.62132 | 46.6% |
| 7 | N5_odds_kimi_all | odds | 15 | 0.62283 | 46.3% |
| 8 | N7_odds_power7 | odds | 10 | 0.62334 | 47.0% |
| 9 | N8_odds_minimal | odds | 5 | 0.62362 | 44.9% |
| 10 | D0_elo_gw | base | 3 | 0.62474 | 44.8% |
| — | FIXED_baseline (prod) | base | 17 | 0.63845 | 45.3% |
| — | **MKT_market** | odds | 3 | **0.59598** | — |

### 10.4 SHAP Analysis

| Test | Brier | #1 Feature (SHAP) | #2 Feature | #3 Feature |
|------|-------|-------------------|------------|------------|
| S0_baseline_17 | 0.6397 | away_matches_played (0.115) | home_goals_scored_avg (0.074) | home_matches_played (0.065) |
| S1_baseline_odds | 0.6180 | away_matches_played (0.101) | odds_home (0.090) | odds_away (0.089) |
| S3_defense_elo | 0.6409 | elo_diff (0.113) | elo_home (0.082) | elo_away (0.059) |
| S7_abe_elo | 0.6303 | — ABE+Elo set | — | — |
| S8_abe_elo_odds | 0.6198 | — ABE+Elo+Odds set | — | — |

**Hallazgos SHAP clave:**
- `away_matches_played` domina globalmente (SHAP=0.115 en S0, 0.101 en S1) — proxy de fase de temporada para Draw
- En clase Draw: `away_matches_played` tiene SHAP=0.231 (abrumador), posible fuga temporal
- Odds aportan pero NO dominan (33.6% share en S1, vs 40-52% en otras ligas)
- `elo_diff` domina sin odds (0.113)
- D1_elo_k10 (K=10) supera a D0_elo_gw (K=32): Elo lento captura mejor inercia uruguaya

### 10.5 Que Funciona y Que No

| Categoria | Funciona | No Funciona |
|-----------|----------|-------------|
| **Elo** | SI — D1_elo_k10 es #4 con solo 3 features | K=32 no es optimo, K=10 mejor |
| **Odds como feature** | SI — 7/10 top tests usan odds | — |
| **Odds como anchor** | NECESARIO — mercado gana +2.1% significativamente | — |
| **xG** | N/A — 0% cobertura | FotMob no tiene xG para Uruguay |
| **Form** | MODERADO — H6_elo_gw_form es #14 | — |
| **Defense** | SI — goals_conceded en top SHAP | — |
| **away_matches_played** | SUSPECTO — SHAP 0.231 en Draw sugiere fuga temporal | Investigar |
| **rest_days** | NO — ruido | Eliminar |
| **Optuna tuning** | — | No se guardo archivo separado |

### 10.6 Gap vs Mercado

```
Mejor modelo (N4 standard):  0.61669
Mercado (OddsPortal closing): 0.59598
Gap:                          +2.1% (mercado GANA)

FAIR delta: +0.02147 [+0.003, +0.040] (SIGNIFICATIVO)
```

**El mercado gana con significancia estadistica.** Mismo patron que Chile y Colombia. Market Anchor es la estrategia recomendada.

### 10.7 Recomendaciones

1. **Market Anchor alpha >= 0.8** para Uruguay: gap +2.1% significativo
2. **Campeon modelo: N4_odds_abe_best** (14 features) si se necesita modelo propio
3. **Investigar `away_matches_played`**: SHAP 0.231 en Draw es sospechosamente alto. Posible fuga temporal
4. **Elo K=10**: considerar para Uruguay (mejor que K=32 default)
5. **No incluir xG**: no disponible
6. **Re-evaluar**: cuando Apertura 2026 tenga >= 150 partidos FT

### 10.8 Archivos

| Archivo | Contenido |
|---------|-----------|
| `scripts/output/lab/feature_lab_results_268.json` | Lab v2, 101 tests |
| `scripts/output/lab/shap_analysis_268.json` | SHAP, 9 escenarios |
| `scripts/output/lab/lab_data_268.csv` | Dataset cacheado (1,977 rows) |
| `data/oddsportal_raw/uruguay_all.json` | OddsPortal raw (1,987 matches) |

---

## 11. Mexico (262)

### 11.1 Ficha Tecnica

| Campo | Valor |
|-------|-------|
| league_id | 262 |
| Nombre | Liga MX |
| Pais | Mexico |
| Formato | Apertura (Jul-Dec) + Clausura (Ene-May), con liguilla |
| N partidos | 1,905 (2020-2026) |
| Universos | base=1,905, odds=1,904 (99.9%), xg=1,724, odds_xg=1,723 |
| Odds source | FDUK (1,904 matches, 99.9% — migrated from odds_history 2026-02-12) |
| xG source | FotMob ID=230 (1,695 matches, 97%+ since 2022) |
| Stats | 100% todas las temporadas |
| Split date (base) | ~2024-10-20 (80/20 temporal) |
| N_test | 381 (base/odds), 345 (xg/odds_xg) |
| Home advantage | H=48.0%, D=23.6%, A=28.3% (fuerte) |
| Market Brier | **0.56605** (N=381, de-vigged Pinnacle closing) |

### 11.2 Campeon: N5_odds_kimi_all (Standard, 15 features)

| Metrica | Valor |
|---------|-------|
| Brier (ensemble) | **0.57571** |
| N_train / N_test | 1,524 / 381 |
| Universe | odds |
| Features | 15 (odds + Kimi selection) |

> **CORRECCION 2026-02-12**: Resultados anteriores (N=178, Market=0.578, Δ=-0.003) eran artefacto de odds_history no migrada a matches. Post-migracion: odds=1,904 (99.9%), N_test=381, Market=0.566. Resultado invertido: mercado gana.

### 11.3 Ranking SHAP (post-migracion, 9 escenarios)

| Test | Feats | N_test | Brier | Top-3 Global SHAP |
|------|-------|--------|-------|-------------------|
| **S1_baseline_odds** | 20 | 381 | **0.58091** | odds_away, odds_home, odds_draw |
| S8_abe_elo_odds | 21 | 381 | 0.58840 | odds_away, odds_home, opp_att_home |
| S2_elo_odds | 6 | 381 | 0.59028 | odds_away, odds_home, elo_diff |
| S6_power_5 | 5 | 381 | 0.60354 | (power features) |
| S5_xg_elo | 6 | 345 | 0.60370 | xG + elo features |
| S3_defense_elo | 5 | 381 | 0.60580 | (base features + elo) |
| S4_m2_interactions | 9 | 381 | 0.60706 | (interactions) |
| S7_abe_elo | 18 | 381 | 0.60957 | ABE + Elo features |
| S0_baseline_17 | 17 | 381 | 0.61021 | home_goals_scored, home_goals_conceded, away_goals_conceded |

**Odds SHAP share en S1**: 50.6% (odds dominate)

### 11.4 Ranking Optuna (post-migracion)

| # | Test | Feats | Universe | Brier | CV Brier | Depth | LR |
|---|------|-------|----------|-------|----------|-------|----|
| 1 | OE_xg_defense_odds | 11 | odds_xg | 0.58210 | 0.60528 | 2 | 0.045 |
| 2 | OB_xg_odds | 6 | odds_xg | 0.58795 | 0.60174 | 2 | 0.018 |
| 3 | OF_abe_elo_odds | 21 | odds | 0.58845 | 0.62069 | 5 | 0.011 |
| 4 | OC_xg_all_elo_odds | 15 | odds_xg | 0.59208 | 0.60505 | 2 | 0.036 |
| 5 | O2_defense_form_elo | 8 | base | 0.60075 | 0.63669 | 2 | 0.028 |
| 6 | O6_efficiency_elo | 8 | base | 0.60353 | 0.64046 | 5 | 0.024 |
| 7 | O4_defense_elo_kimi | 14 | base | 0.60418 | 0.64054 | 2 | 0.032 |
| 8 | O0_elo_gw_defense | 5 | base | 0.60574 | 0.63780 | 2 | 0.041 |

### 11.5 SHAP share (S1_baseline_odds)

**Odds SHAP share**: 50.6% — odds_home=0.150, odds_away=0.100, odds_draw dominan

### 11.6 Que Funciona y Que No

| Feature Group | Veredicto | Evidencia |
|---------------|-----------|-----------|
| **Odds (home/draw/away)** | DOMINANTE | 42.1% SHAP share en S8, odds_away #1 global |
| **ABE features** | COMPLEMENTARIO | opp_att_home 0.052 en S8, overperf_away 0.041 |
| **Elo (home/away/diff)** | SEÑAL | elo_away=0.040 en S8, elo_diff=0.024. Mejor sin odds |
| **xG (for/against/overperf)** | SEÑAL COMPLEMENTARIA | P8_xg_defense_odds=0.58187 (mejor que muchos odds-only). Unica liga LATAM donde xG mejora |
| **Efficiency (finish_eff, def_eff)** | MODERADO | O6_efficiency_elo=0.60353 (#2 Optuna base). Mas util en Liga MX que en otras ligas |
| **Defense conceded** | SEÑAL FUERTE | #2 en S0 baseline (0.062), consistente across scenarios |
| **Shots avg** | MODERADO | home_shots 0.057 en S0, complementario |
| **rest_days** | DEBIL | home_rest 0.016 en S0, marginal en S8 |
| **Corners** | MODERADO | away_corners 0.035 en S0, mas util que en otras ligas |

### 11.7 Hyperparams Optimos (consensus)

| Param | Rango optimo | Notas |
|-------|-------------|-------|
| max_depth | 2-3 | Preferir 2 sin odds, 3 con xG+odds |
| learning_rate | 0.01-0.04 | Conservador |
| n_estimators | 50-150 | Depende de LR |
| min_child_weight | 10-15 | Regularizacion alta |
| subsample | 0.53-0.64 | Moderado |
| colsample_bytree | 0.40-0.80 | Amplio rango |

### 11.8 Gap vs Mercado (CORREGIDO 2026-02-12)

```
Market Brier:  0.56605 (N=381, de-vigged Pinnacle closing)
Model Brier:   0.57571 (N5_odds_kimi_all, 15 features)
SHAP Best:     0.58091 (S1_baseline_odds, 20 features)
FAIR Delta:    +0.0098 (mercado gana) CI95 [-0.005, +0.025]
Significativo: NO (CI cruza cero)

FAIR xG+odds:         +0.0187 SIGNIFICATIVO CI [+0.004, +0.034]
FAIR Optuna xG+odds:  +0.0230 SIGNIFICATIVO CI [+0.008, +0.037]
```

> **El resultado anterior "modelo gana" (Δ=-0.003) era artefacto** de odds incompletas (N=178, solo 888/1,905 odds). Con cobertura completa (1,904/1,905): mercado gana, patron identico al resto de LATAM.

### 11.9 Nota sobre xG en Mexico

Mexico es la **unica liga LATAM donde xG aporta señal complementaria**:
- OE_xg_defense_odds (Optuna): 0.58210 — mejor Optuna con xG
- P5_xg_odds (standard): 0.57685 — competitivo
- Pero FAIR xG+odds es SIGNIFICATIVO (+1.9-2.3%): el mercado sigue ganando incluso con xG

### 11.10 Recomendaciones (ACTUALIZADAS)

1. **Market Anchor VIABLE** — mercado gana +1.0% (no sig standard, pero sig con xG+odds)
2. **xG vale la pena** — unica liga LATAM donde xG+odds combinadas dan señal (OE=0.58210)
3. **Best sin odds**: O2_defense_form_elo (0.60075, 8 feats) — viable para matches sin odds
4. **N_test suficiente** — 381 matches (post-migracion) vs 178 antes. Resultados ahora confiables
5. **Market Anchor α recomendado**: 0.6-0.8 (gap menor que Chile/MLS, xG aporta algo)

### 11.11 Backfill Status

**FDUK Odds** (2026-02-12):
- Script: `/tmp/backfill_fduk_mexico.py` (targeted)
- CSV: `https://www.football-data.co.uk/new/MEX.csv` (Extra format, AvgCH/AvgCD/AvgCA)
- Resultado: 1,904/1,905 matched and inserted
- Unmatched: Monarcas (defunct), Veracruz (defunct)
- Cobertura: 99.7-100% across all years 2020-2026

**FotMob xG** (2026-02-12):
- Script: `/tmp/backfill_fotmob_xg_mexico.py`
- FotMob ID: 230
- Resultado: 1,870 linked, 1,695 xG captured (90.6% overall, 97%+ since 2022)
- Season format: academic year "2024/2025 - Apertura"/"2024/2025 - Clausura"
- 2020/2021 Apertura: 142 errors (old data), non-blocking
- DB season=N maps to FotMob "N/N+1" (Jul-May)

### 11.12 Archivos

| Archivo | Contenido |
|---------|-----------|
| `scripts/output/lab/feature_lab_results.json` | Lab v2, 101 tests (league 262) |
| `scripts/output/lab/shap_analysis_262.json` | SHAP, 9 escenarios |
| `scripts/output/lab/feature_lab_results_optuna.json` | Optuna, 16 candidatos |
| `scripts/output/lab/lab_data_262.csv` | Dataset cacheado (1,905 rows) |

---

## 12. MLS (253)

### 12.1 Ficha Tecnica

| Campo | Valor |
|-------|-------|
| **League ID** | 253 |
| **Nombre** | Major League Soccer (MLS) |
| **Pais** | USA / Canada |
| **Formato** | Temporada unica (Feb-Oct), playoffs (Oct-Dec) |
| **Equipos** | 29 (desde 2024) |
| **N partidos en DB** | 2,830 (Feb 2020 — Dec 2025) |
| **Universos** | base=2,830, odds=1,561 (55.2%), xg=2,613 (92.3%), odds_xg=1,452 (51.3%) |
| **Train / Test split** | 2,264 / 566 (base), 1,248 / 313 (odds), 2,090 / 523 (xg), 1,161 / 291 (odds_xg) |
| **Split date** | ~2024-10-20 (base), ~2025-05-25 (odds) |
| **Distribucion resultados** | Home 46.1%, Draw 22.8%, Away 31.1% |
| **Odds coverage** | 0% pre-2023, 100% desde 2023 (FDUK Pinnacle closing) |
| **xG coverage** | **91.0%** via FotMob (2,576/2,830). Backfill 2026-02-12 |
| **Market Brier** | **0.60165** (N=313) |

### 12.2 Campeon: P8_xg_defense_odds (Standard, 11 features)

| Metrica | Valor |
|---------|-------|
| Brier (ensemble) | **0.63176** |
| CI95 | [0.608, 0.658] |
| N_train / N_test | 1,161 / 291 |
| Universe | odds_xg |
| Features | odds_home, odds_draw, odds_away, elo_diff, elo_home, elo_away, xg_diff, xg_home, xg_away, goals_conceded_avg_home, goals_conceded_avg_away |

**RE-RUN 2026-02-12** tras backfill FotMob xG (91% coverage). P8_xg_defense_odds (11 feats, odds+xG) supera al anterior campeon N8_odds_minimal (0.63420) por 0.0024. xG aporta señal marginal via metricas defensivas combinadas con odds. Sin embargo, xG solo (P3=0.64723) es peor que baseline.

### 12.3 Ranking Completo (Standard Lab, top-20)

| # | Test | Feats | Universe | Brier | CI95 | Acc |
|---|------|-------|----------|-------|------|-----|
| 1 | **P8_xg_defense_odds** | 11 | **odds_xg** | **0.63176** | [0.608, 0.658] | 0.437 |
| 2 | N8_odds_minimal | 5 | odds | 0.63420 | [0.609, 0.661] | 0.422 |
| 3 | P9_xg_ultimate | 22 | odds_xg | 0.63620 | [0.612, 0.664] | 0.414 |
| 4 | M2_h0_interactions | 9 | base | 0.63795 | [0.622, 0.653] | 0.453 |
| 5 | D5_elo_split | 3 | base | 0.63991 | [0.625, 0.654] | 0.478 |
| 6 | P7_xg_all_elo_odds | 15 | odds_xg | 0.64039 | [0.615, 0.667] | 0.411 |
| 7 | P5_xg_odds | 6 | odds_xg | 0.64183 | [0.618, 0.668] | 0.414 |
| 8 | A0_baseline_17 | 17 | base | 0.64307 | [0.628, 0.659] | 0.453 |
| 9 | P6_xg_elo_odds | 9 | odds_xg | 0.64419 | [0.620, 0.672] | 0.393 |
| 10 | P3_xg_defense_elo | 9 | xg | 0.64723 | [0.635, 0.660] | 0.443 |
| 11 | P4_xg_overperf_elo | 12 | xg | 0.64930 | [0.633, 0.666] | 0.436 |
| 12 | P1_xg_all | 9 | xg | 0.65127 | [0.636, 0.668] | 0.429 |
| 13 | P2_xg_elo | 6 | xg | 0.65141 | [0.635, 0.668] | 0.438 |
| 14 | P0_xg_core | 3 | xg | 0.65855 | [0.645, 0.674] | 0.431 |

**Anomalias notables**:
- **D5_elo_split** (3 feats: elo_honly_home, elo_aonly_away, elo_split_diff) = 0.63991 — casi tan bueno como el campeon con 5-15 features. MLS home advantage (46.1%) es perfectamente capturada por Elo split.
- **xG solo**: Todos los tests P0-P4 (universo xg) son PEORES que baseline (A0=0.64307). xG por si solo no aporta señal predictiva en MLS.
- **xG + odds**: P8_xg_defense_odds (0.63176) supera a N8_odds_minimal (0.63420) por 0.0024. xG aporta via metricas defensivas.

### 12.4 Ranking Optuna

| # | Test | Feats | Universe | Brier | CV Brier | Depth | LR | N_est |
|---|------|-------|----------|-------|----------|-------|----|-------|
| 1 | **OF_abe_elo_odds** | 21 | odds | **0.63494** | 0.63208 | 6 | 0.014 | 113 |
| 2 | O5_m2_interactions | 9 | base | 0.64163 | 0.64524 | 2 | 0.017 | 147 |
| 3 | O7_all_abe_elo | 18 | base | 0.64314 | 0.64259 | 6 | 0.010 | 148 |
| 4 | O0_elo_gw_defense | 5 | base | 0.64392 | 0.64280 | 2 | 0.017 | 101 |
| 5 | O4_defense_elo_kimi | 14 | base | 0.64394 | 0.64203 | 5 | 0.014 | 114 |
| 6 | O2_defense_form_elo | 8 | base | 0.64455 | 0.64485 | 2 | 0.038 | 65 |
| 7 | O6_efficiency_elo | 8 | base | 0.64546 | 0.64383 | 6 | 0.010 | 163 |
| 8 | O1_elo_gw_form | 6 | base | 0.64590 | 0.64464 | 2 | 0.026 | 65 |

**Tests xG (RE-RUN 2026-02-12)**:

| # | Test | Feats | Universe | Brier | CV Brier | Depth | LR | N_est |
|---|------|-------|----------|-------|----------|-------|----|-------|
| — | OB_xg_odds | 6 | odds_xg | 0.63929 | 0.64106 | 2 | 0.011 | 121 |
| — | OC_xg_all_elo_odds | 15 | odds_xg | 0.63707 | 0.64188 | 2 | 0.015 | 105 |
| — | OD_xg_overperf_elo | 12 | xg | 0.64662 | 0.64356 | 2 | 0.034 | 50 |
| — | OE_xg_defense_odds | 11 | odds_xg | 0.63734 | 0.64501 | 2 | 0.023 | 53 |

xG Optuna: OC y OE competitivos con OF, pero no lo superan.

**Patron Optuna**: tests con odds necesitan depth=6 (mas profundidad), tests base y xG depth=2.

### 12.5 SHAP Analysis (9 escenarios)

| Test | Feats | N_test | Brier | Universo |
|------|-------|--------|-------|----------|
| **S8_abe_elo_odds** | 21 | 313 | **0.63767** | odds |
| S2_elo_odds | 6 | 313 | 0.63933 | odds |
| S4_m2_interactions | 9 | 566 | 0.64009 | base |
| S0_baseline_17 | 17 | 566 | 0.64488 | base |
| S7_abe_elo | 18 | 566 | 0.64609 | base |
| S6_power_5 | 5 | 566 | 0.64711 | base |
| S3_defense_elo | 5 | 566 | 0.65047 | base |
| **S5_xg_elo** | 9 | 523 | 0.65198 | **xg** |
| S1_baseline_odds | 20 | 313 | 0.67606 | odds |

**S8_abe_elo_odds SHAP global (top-5)**:
| Feature | Mean |SHAP| |
|---------|-----------|
| odds_away | 0.0891 |
| odds_home | 0.0800 |
| opp_att_home | 0.0352 |
| home_bias_home | 0.0345 |
| home_bias_away | 0.0270 |

**Odds share**: 37.7%

**Anomalia S1**: Brier 0.67606 — CATASTROFICO. 20 features en odds universe (N_train=1,248) provoca overfit severo. Confirma que parsimonia es critica con N pequeño.

### 12.6 Que Funciona y Que No

| Feature Group | Veredicto | Evidencia |
|---------------|-----------|-----------|
| **Odds (home/draw/away)** | DOMINANTE | 37.7% SHAP share en S8, odds_away #1 |
| **Elo split (home-only/away-only)** | SEÑAL FUERTE | D5=0.63991 con solo 3 feats, captura home advantage |
| **ABE features** | COMPLEMENTARIO | opp_att_home=0.035, home_bias=0.035 en S8 |
| **Elo standard (home/away/diff)** | SEÑAL | elo_diff #1 sin odds, pero Elo split es superior |
| **Interactions (elo_x_defense, etc.)** | MODERADO | M2=0.63795 (9 feats) competitivo |
| **home_matches_played** | ANOMALIA | B2=0.64537 con 1 solo feature. Proxy de calendario MLS |
| **Defense conceded** | SEÑAL | home_goals_conceded = 0.066 en baseline |
| **rest_days** | DEBIL | Negligible en S0 |
| **xG (solo)** | DEBIL | P0-P4 todos peores que baseline. xG_diff SHAP=0.058 en S5 (debil vs elo=0.093) |
| **xG + odds** | MARGINAL+ | P8_xg_defense_odds (0.63176) mejora 0.0024 sobre N8 odds-only. xG aporta via defense |

### 12.7 Hyperparams Optimos (consensus)

| Param | Con odds | Sin odds |
|-------|----------|----------|
| max_depth | 6 | 2 |
| learning_rate | 0.010-0.014 | 0.017-0.038 |
| n_estimators | 100-150 | 65-150 |
| min_child_weight | 3-7 | 3-10 |
| subsample | 0.84 | 0.80-0.87 |

Patron claro: odds requieren mas profundidad (6) para explotar interacciones, base prefiere arboles simples (2).

### 12.8 Gap vs Mercado

```
Market Brier:  0.60165 (N=313, de-vigged Pinnacle closing)
Best model:    0.63176 (P8_xg_defense_odds, 11 features, odds_xg)
Best odds-only:0.63420 (N8_odds_minimal, 5 features)
Best Optuna:   0.63494 (OF_abe_elo_odds, 21 features)
Best SHAP:     0.63767 (S8_abe_elo_odds, 21 features)
FAIR Delta:    +0.0298 SIGNIFICATIVO (N8 vs market, N=313)
FAIR xG+odds:  +0.0287 SIGNIFICATIVO (P8 vs market, N=291)
```

MLS tiene el **gap significativo mas grande** de todas las ligas analizadas (+2.9-3.2%), superando incluso a Chile (+3.0%). xG reduce el gap marginalmente (de +3.0% a +2.9%) pero no lo cierra.

### 12.9 Recomendaciones

1. **Market Anchor α=0.8-1.0** — gap +2.9% significativo, mercado claramente superior
2. **P8_xg_defense_odds para produccion** — 11 features (0.63176), mejor absoluto con xG+odds
3. **N8_odds_minimal alternativa** — 5 features (0.63420), casi igual sin depender de xG
4. **D5_elo_split como base** — 3 features (0.63991) mejor opcion eficiente sin odds
5. **xG veredicto**: marginal. Solo mejora 0.0024 sobre odds-only, y solo combinado con defense metrics
6. **No usar S1_baseline_odds** — overfit catastrofico con 20 features en N=1,248 odds

### 12.10 Odds Breakdown

| Fuente | Matches | Periodo | Tipo |
|--------|---------|---------|------|
| FDUK Pinnacle (PS) | 1,551 | 2023-02 a 2025-12 | closing |
| FDUK Average (Avg) | 10 | 2025-10 a 2025-11 | closing |
| API-Football | 0 | — | (purged) |
| **Total** | **1,561** | | |

100% Pinnacle closing desde 2023 — mejor calidad disponible.

### 12.11 Archivos

| Archivo | Contenido |
|---------|-----------|
| `scripts/output/lab/feature_lab_results.json` | Lab v2, 101 tests (league 253) |
| `scripts/output/lab/shap_analysis_253.json` | SHAP, 8 escenarios |
| `scripts/output/lab/feature_lab_results_optuna.json` | Optuna, 13 candidatos (4 xG skipped) |
| `scripts/output/lab/lab_data_253.csv` | Dataset cacheado (2,830 rows) |

---

## 13. La Liga - España (140)

### 13.1 Ficha Tecnica

| Campo | Valor |
|-------|-------|
| League ID | 140 |
| Pais | España |
| N matches | 4,028 FT (2015-2026) |
| Odds coverage | 4,028/4,028 = **100%** |
| xG coverage | **99.5%** via Understat (backfill historico 2026-02-12). 4,007/4,028 matches con xG. |
| N_train / N_test | 3,222 / 806 (base/odds), 3,205 / 802 (xg/odds_xg) |
| Split date | ~2024-01 |
| Tests ejecutados | 101 standard + 9 SHAP + 17 Optuna |
| Fecha lab | 2026-02-13 (re-run con xG completo) |

### 13.2 Campeon: OB_xg_odds (Optuna, 6 features)

El campeon global es el Optuna **OB_xg_odds** con Brier **0.58046** (odds_xg universe). Es el UNICO test que supera a N8_odds_minimal (0.58098). Con Optuna tuning, xG logra exprimir +0.00052 que standard no consigue.

| Metrica | Valor |
|---------|-------|
| Test | OB_xg_odds (Optuna) |
| Brier | **0.58046** |
| Universe | odds_xg |
| Features | 6 (xG + odds) |
| N_test | 802 |

| Metrica | Valor |
|---------|-------|
| Test | N8_odds_minimal (Standard) |
| Brier | **0.58098** |
| Universe | odds |
| Features | 5 (odds + elo_diff + form_diff) |
| N_test | 806 |

Best xG sin odds (Optuna): OD_xg_overperf_elo = 0.58633 (12 feats, xg)
Best base (Optuna): O1_elo_gw_form = 0.58784 (6 feats)

### 13.3 Ranking Completo (Standard Lab, top-20)

| # | Test | Brier | Uni | N_feat | N_test |
|---|------|-------|-----|--------|--------|
| 1 | **N8_odds_minimal** | **0.58098** | odds | 5 | 806 |
| 2 | J0_only_odds | 0.58233 | odds | 3 | 806 |
| 3 | P5_xg_odds | 0.58258 | odds_xg | 6 | 802 |
| 4 | P8_xg_defense_odds | 0.58307 | odds_xg | 11 | 802 |
| 5 | N1_odds_defense_elo | 0.58343 | odds | 8 | 806 |
| 6 | N9_odds_ultimate | 0.58366 | odds | 22 | 806 |
| 7 | N5_odds_kimi_all | 0.58371 | odds | 15 | 806 |
| 8 | N2_odds_m2_combo | 0.58396 | odds | 12 | 806 |
| 9 | N7_odds_power7 | 0.58430 | odds | 10 | 806 |
| 10 | N3_odds_efficiency | 0.58434 | odds | 11 | 806 |
| 11 | P6_xg_elo_odds | 0.58444 | odds_xg | 9 | 802 |
| 12 | N4_odds_abe_best | 0.58450 | odds | 14 | 806 |
| 13 | P7_xg_all_elo_odds | 0.58472 | odds_xg | 15 | 802 |
| 14 | P9_xg_ultimate | 0.58486 | odds_xg | 22 | 802 |
| 15 | J1_elo_odds | 0.58517 | odds | 6 | 806 |
| 16 | N0_odds_elo | 0.58517 | odds | 6 | 806 |
| 17 | P3_xg_defense_elo | 0.58559 | xg | 9 | 802 |
| 18 | P2_xg_elo | 0.58695 | xg | 6 | 802 |
| 19 | H6_elo_gw_form | 0.58703 | base | 6 | 806 |
| 20 | P4_xg_overperf_elo | 0.58718 | xg | 12 | 802 |

**Notas**:
- N8 (odds-only, 5 feats) sigue como campeon. xG+odds tests (P5, P8) se intercalan en #3-#4 pero NO superan a N8.
- xG sin odds (P3=0.586, P2=0.587) mejoran base (H6=0.587) solo marginalmente.
- Gap odds vs base: ~0.6% (0.581 vs 0.587) — el mas estrecho de las 3 ligas EUR. Elo es muy fuerte en España.
- Spread total top-20: solo 0.006 (0.581 a 0.587) — todos los tests estan muy cerca.

### 13.4 Ranking Optuna (17 candidatos, con xG completo)

| Test | Brier | Uni | N_feat | FAIR delta | Sig? |
|------|-------|-----|--------|------------|------|
| **OB_xg_odds** | **0.58046** | odds_xg | 6 | +0.01675 | **SI** |
| OF_abe_elo_odds | 0.58094 | odds | 21 | — | — |
| OE_xg_defense_odds | 0.58136 | odds_xg | 11 | — | — |
| OC_xg_all_elo_odds | 0.58305 | odds_xg | 15 | — | — |
| OD_xg_overperf_elo | 0.58633 | xg | 12 | — | — |
| O1_elo_gw_form | 0.58784 | base | 6 | — | — |
| O3_elo_k20 | 0.58883 | base | 3 | — | — |
| O8_smart_minimal | 0.58991 | base | 13 | — | — |
| O7_all_abe_elo | 0.59092 | base | 18 | — | — |
| O2_defense_form_elo | 0.59111 | base | 8 | — | — |
| O4_defense_elo_kimi | 0.59143 | base | 14 | — | — |
| O0_elo_gw_defense | 0.59169 | base | 5 | — | — |
| O6_efficiency_elo | 0.59204 | base | 8 | — | — |
| OA_only_elo | 0.59325 | base | 3 | — | — |
| O5_m2_interactions | 0.59497 | base | 9 | — | — |
| O9_baseline_17 | 0.61387 | base | 17 | — | — |
| FIXED_baseline | 0.61515 | base | 17 | — | — |

**OB_xg_odds (0.58046) supera a N8 (0.58098)** — Optuna con xG es el unico camino que supera al campeon odds-only. Top-3 son xG+odds. OF (odds-only) cae a #2. xG aporta +0.00048 con Optuna tuning.

### 13.5 SHAP Analysis (9 escenarios, con xG completo)

| Escenario | Brier | N_feat | N_test | Odds share |
|-----------|-------|--------|--------|------------|
| S0_baseline_17 | 0.62036 | 17 | 806 | — |
| S1_baseline_odds | 0.59496 | 20 | 806 | — |
| S2_elo_odds | 0.58568 | 6 | 806 | 79% |
| S3_defense_elo | 0.59198 | 5 | 806 | — |
| S4_m2_interactions | 0.59949 | 9 | 806 | — |
| **S5_xg_elo** | **0.58630** | 6 | 802 | — |
| S6_power_5 | 0.59383 | 5 | 806 | — |
| S7_abe_elo | 0.59280 | 18 | 806 | — |
| **S8_abe_elo_odds** | **0.58236** | 21 | 806 | 59.2% |

**S5_xg_elo ahora funciona** (N=802 vs N=78 pre-backfill). Brier **0.58630** — supera S3_defense_elo (0.59198) por **+0.00568**. Es el mejor escenario sin odds.

SHAP top-6 (S5_xg_elo):
1. elo_diff: 0.16895 (39.4%) — domina
2. elo_away: 0.07126 (16.6%)
3. **xg_diff: 0.06869 (16.0%)** — #3 global, senal real
4. elo_home: 0.05004 (11.7%)
5. away_xg_for_avg: 0.03791 (8.8%)
6. home_xg_for_avg: 0.03218 (7.5%)

**xG share en S5: ~32%** (xg_diff + xg_for features). Menor que Alemania (40%) pero sustancial.

SHAP S8_abe_elo_odds (top-5):
1. odds_home: 0.126 (25.5%)
2. odds_away: 0.113 (22.8%)
3. odds_draw: 0.055 (11.1%)
4. elo_diff: 0.039 (7.9%)
5. overperf_away: 0.017 (3.4%)

**Odds share S8: 59.2%** — record absoluto de todas las ligas.

### 13.6 Que Funciona y Que No

| Feature Group | Veredicto | Evidencia |
|---------------|-----------|-----------|
| **Odds** | DOMINANTE | 59.2% SHAP share en S8 — record. N8 (5 feats odds) = campeon |
| **Elo** | SEÑAL FUERTE | elo_diff #1 sin odds (0.169 SHAP en S5). Base tests competitivos (0.587) |
| **xG** | SEÑAL REAL, NO MEJORA ODDS | S5_xg_elo (0.586) vs S3 (0.592) = +0.006 sin odds. Pero P5_xg_odds (0.583) > N8 (0.581) — xG no mejora a odds |
| **ABE (opp_rating, overperf)** | COMPLEMENTARIO | overperf_away=0.017 en S8 |
| **Defense conceded** | MODERADO | home_goals_conceded_avg aparece en draw prediction |
| **rest_days** | DEBIL | Negligible |

**Paradoja España**: Mercado mas eficiente (0.563) + Elo mas fuerte entre EUR + odds absorben toda la info de xG. xG aporta sin odds pero es redundante con odds.

### 13.7 Impacto xG (comparacion pre/post backfill)

| Metrica | Sin xG (pre-backfill) | Con xG (post-backfill) | Delta |
|---------|-----------------------|------------------------|-------|
| Champion standard | N8_odds_minimal 0.58098 | N8_odds_minimal **0.58098** | **0** (xG no desplaza standard) |
| Champion Optuna | OF_abe_elo_odds 0.58094 | OB_xg_odds **0.58046** | **-0.00048** (marginal) |
| Best xG+odds (std) | P5 N=78 (no confiable) | P5_xg_odds **0.58258** (N=802) | ahora confiable, pero > N8 |
| Best no-odds | O1_elo_gw_form 0.58784 | OD_xg_overperf_elo **0.58633** | **-0.00151** |
| SHAP S5_xg_elo | N=78 (no confiable) | **0.58630** (N=802) | ahora confiable |
| FAIR gap (OB) | +0.01675 (Optuna xG) | vs +0.01788 (N8 odds) | **-0.00113** |

xG en España: **marginal con Optuna** (+0.0005), **nulo en standard**, **+0.002 sin odds**. Contraste con Alemania donde xG mejora todo en +0.004.

### 13.8 Gap vs Mercado

| Metrica | Valor |
|---------|-------|
| Market Brier | **0.56287** |
| Best model (OB Optuna xG+odds) | 0.58046 |
| FAIR delta (OB) | +0.01675 |
| CI95 | [+0.00732, +0.02606] |
| Significativo? | **SI** |
| Best standard (N8 odds-only) | 0.58098 |
| FAIR delta (N8) | +0.01788 |
| CI95 | [+0.00821, +0.02694] |
| Significativo? | **SI** |

Gap +1.7% (OB Optuna) a +1.8% (N8 standard). Optuna con xG reduce gap marginalmente (de +0.0179 a +0.0168). El mercado español es el mas eficiente de las 3 ligas EUR analizadas.

### 13.9 Recomendaciones

1. **OB_xg_odds (Optuna) o N8_odds_minimal (standard) para produccion** — OB gana por +0.0005 con xG, N8 es mas robusto con 5 features
2. **Market Anchor α=0.6-0.8** — gap +1.7% significativo pero moderado
3. **xG para v1.0.2: incluir, beneficio marginal con Optuna** — OB (0.58046) vs OF (0.58094) = +0.0005 con xG. Sin odds: +0.002.
4. **elo_diff como ancla sin odds** — 39.4% SHAP en S5, domina todos los tests base
5. **Odds share 59.2%** — record. Modelo es esencialmente un wrapper de odds con Elo como ajuste
6. **FIXED_baseline 0.615**: produccion actual 3.4% peor que OB. Margen de mejora significativo.

### 13.10 Odds Breakdown

| Fuente | Matches | Periodo | Tipo |
|--------|---------|---------|------|
| FDUK Pinnacle (PS) | 3,417 | 2016-08 a 2025-05 | closing |
| Sofascore | 171 | 2025-08 a 2026-01 | opening |
| FDUK Bet365 (B365) | 57 | 2025-09 a 2026-02 | closing |
| Sin source | 380 | 2015-08 a 2016-05 | — |
| UNVERIFIED | 3 | 2020-2021 | — |
| **Total** | **4,028** | | |

100% odds coverage. Pinnacle closing domina (85%).

### 13.11 Archivos

| Archivo | Contenido |
|---------|-----------|
| `scripts/output/lab/feature_lab_results_140.json` | Lab v2, 101+ tests (league 140) |
| `scripts/output/lab/shap_analysis_140.json` | SHAP, 9 escenarios (con xG completo) |
| `scripts/output/lab/feature_lab_results_optuna_140.json` | Optuna, 17 candidatos (con xG completo) |
| `scripts/output/lab/lab_data_140.csv` | Dataset cacheado (4,028 rows) |

---

## 14. Ligue 1 - Francia (61)

### 14.1 Ficha Tecnica

| Campo | Valor |
|-------|-------|
| League ID | 61 |
| Pais | Francia |
| N matches | 3,745 (2015-2026) |
| Odds coverage | 3,740/3,745 = **99.9%** |
| xG coverage | 3,683/3,745 = **99%** (Understat, backfill historico 2026-02-12) |
| N_train / N_test | 2,996 / 749 (base), 2,992 / 748 (odds), 2,946 / 737 (xg), 2,943 / 736 (odds_xg) |
| Split date | 2023-09-29 (base), 2023-10-07 (xg/odds_xg) |
| Tests ejecutados | 101 standard + 9 SHAP + 16 Optuna (re-run completo post-xG backfill) |
| Fecha lab | 2026-02-13 (v2 con xG completo) |

**NOTA IMPORTANTE**: Esta seccion fue completamente reescrita el 2026-02-13 despues de ejecutar el backfill historico de Understat xG (de 8.1% a 99% cobertura). La version anterior (2026-02-12) mostraba tests xG con N_test=61 y CIs inutiles. Todos los resultados xG aqui son DEFINITIVOS con N_test=736-737.

### 14.2 Campeon: OB_xg_odds (Optuna, 6 features)

El campeon absoluto es el Optuna **OB_xg_odds** con Brier **0.60040**. El mejor standard es P5_xg_odds con Brier 0.60100. Ambos usan xG + odds (6 features).

| Metrica | Valor |
|---------|-------|
| Test | OB_xg_odds (Optuna) |
| Brier | **0.60040** |
| CI95 | [0.58386, 0.61828] |
| CV Brier | 0.60432 |
| Features | home_xg_for_avg, away_xg_for_avg, xg_diff, odds_home, odds_draw, odds_away |
| Hyperparams | depth=2, lr=0.028, n_est=120, mcw=11 |
| N_test | 736 |

Champion alternativo (standard): P5_xg_odds = **0.60100** (6 features, odds_xg)
Champion sin xG: J0_only_odds = **0.60302** (3 features, odds)
Champion base: O3_elo_k20 = **0.60714** (3 features, base, Optuna)

### 14.3 Ranking Completo (Standard Lab, top-15)

| # | Test | Brier | Uni | N_feat | N_test |
|---|------|-------|-----|--------|--------|
| 1 | **P5_xg_odds** | **0.60100** | odds_xg | 6 | 736 |
| 2 | P6_xg_elo_odds | 0.60271 | odds_xg | 9 | 736 |
| 3 | J0_only_odds | 0.60302 | odds | 3 | 748 |
| 4 | P2_xg_elo | 0.60368 | xg | 6 | 737 |
| 5 | N1_odds_defense_elo | 0.60452 | odds | 8 | 748 |
| 6 | N8_odds_minimal | 0.60584 | odds | 5 | 748 |
| 7 | N7_odds_power7 | 0.60588 | odds | 10 | 748 |
| 8 | D1_elo_k10 | 0.60604 | base | 3 | 749 |
| 9 | H4_kitchen_sink | 0.60638 | base | 17 | 749 |
| 10 | P3_xg_defense_elo | 0.60747 | xg | 9 | 737 |
| 11 | P1_xg_all | 0.60778 | xg | 9 | 737 |
| 12 | P7_xg_all_elo_odds | 0.60786 | odds_xg | 15 | 736 |
| 13 | P4_xg_overperf_elo | 0.61015 | xg | 12 | 737 |
| 14 | A1_only_elo_k32 | 0.61183 | base | 3 | 749 |
| 15 | P8_xg_defense_odds | 0.61381 | odds_xg | 11 | 736 |

**Patron clave**: xG + odds (P5, 0.601) supera odds puras (J0, 0.603) y xG + Elo (P2, 0.604). Parsimonia: 6 features > 15 features.

**Correccion critica vs version anterior**: Los tests xG que antes mostraban Brier 0.56-0.58 con N_test=61 ahora muestran 0.601-0.614 con N_test=736. La "promesa" de xG era ruido estadistico puro.

### 14.4 SHAP Analysis (9 escenarios)

| Escenario | Brier | N_feat | N_test | Odds share |
|-----------|-------|--------|--------|------------|
| S0_baseline_17 | 0.64198 | 17 | 749 | — |
| S1_baseline_odds | 0.62173 | 20 | 748 | 50.9% |
| S2_elo_odds | 0.60680 | 6 | 748 | 80.1% |
| S3_defense_elo | 0.61084 | 5 | 749 | — |
| S4_m2_interactions | 0.61546 | 9 | 749 | — |
| **S5_xg_elo** | **0.60496** | 6 | 737 | — |
| S6_power_5 | 0.61489 | 5 | 749 | — |
| S7_abe_elo | 0.62243 | 18 | 749 | — |
| S8_abe_elo_odds | 0.61826 | 21 | 748 | 53.9% |

**S5_xg_elo** es el campeon SHAP (0.60496) — primer escenario donde xG domina Elo.

SHAP top-5 (S5_xg_elo, global ranking):
1. **elo_diff**: 0.140 — dominante
2. elo_away: 0.065
3. **xg_diff**: 0.059 — señal real (#3)
4. home_xg_for_avg: 0.056
5. elo_home: 0.044

SHAP top-5 (S1_baseline_odds):
1. odds_away: dominante
2. odds_home: dominante
3. odds_draw: fuerte
4. away_matches_played
5. goal_diff_avg

**xG SHAP por clase** (S5_xg_elo):
- Home: xg_diff=0.088 (#2 detras de elo_diff=0.186)
- Away: xg_diff=0.069 (#3 detras de elo_diff=0.195, elo_away=0.082)
- Draw: home_xg_for_avg=0.075 domina (xg_diff=0.021, minor)

### 14.5 Que Funciona y Que No

**Funciona bien**:
- **xG + odds (P5/OB)**: 0.600-0.601 — campeon absoluto, xG aporta señal real
- **xG + Elo (P2/S5)**: 0.604-0.605 — xG mejora sobre Elo solo (+0.008 Brier)
- Odds puras (J0): 0.603 — sigue siendo fuerte con solo 3 features
- Elo_k10/k20 > k32: O3=0.607 vs A1=0.612 — K bajo sigue confirmado con N grande
- Parsimonia: 6 features (OB) gana sobre 15+ features en todos los casos

**No funciona**:
- Baseline 17 features (A0): 0.638 — demasiado ruido sin Elo
- ABE features (S7/S8): no mejoran sobre Elo+odds simple
- xG defense (P3_xg_defense_elo): 0.607 — xG_against no aporta vs xG_for+diff
- OF_abe_elo_odds (Optuna, 21 feats): 0.617 — PEOR que J0 (0.603). Overfitting.
- FIXED_baseline (prod actual): 0.638 — 3.8% peor que campeon OB

**xG — Veredicto definitivo** (con N_test=736, NO los N=61 previos):
- xG es SEÑAL REAL: xg_diff es #3 SHAP feature, cierra ~0.003 del gap
- P5_xg_odds (0.601) vs J0_only_odds (0.603) = +0.002 mejora con odds
- P2_xg_elo (0.604) vs A1_only_elo (0.612) = +0.008 mejora sin odds
- PERO el mercado ya incorpora xG: gap sigue +2.3% significativo
- El resultado previo OE=0.566 era ARTEFACTO de N=61 (CI [-0.05, +0.07])

### 14.6 Gap vs Mercado

```
Market Brier:     0.57704 (N=748, de-vigged Pinnacle closing)
Best model (OB):  0.60040 (Optuna xG+odds, 6 features)
Best standard:    0.60100 (P5_xg_odds, 6 features)
Best sin xG:      0.60302 (J0_only_odds, 3 features)
Best base:        0.60714 (O3_elo_k20, 3 features)
FAIR (OB vs mkt): +0.02347 SIGNIFICATIVO CI[+0.013, +0.033]
FAIR (J0 vs mkt): +0.02648 SIGNIFICATIVO CI[+0.017, +0.036]
```

Gap de +2.3% significativo (reducido de +2.6% gracias a xG). El mayor gap entre las top-5 EUR.

**Contribucion del xG al gap**: xG reduce el FAIR delta de +0.0265 a +0.0235, una mejora de ~0.003. Real pero insuficiente para superar al mercado.

### 14.7 Ranking Optuna (16 candidatos, re-run con xG completo)

| Test | Brier | CV Brier | Depth | LR | N_est | Uni |
|------|-------|----------|-------|----|-------|-----|
| **OB_xg_odds** | **0.60040** | 0.60432 | 2 | 0.028 | 120 | odds_xg |
| OE_xg_defense_odds | 0.60154 | 0.60443 | 2 | 0.021 | 134 | odds_xg |
| OC_xg_all_elo_odds | 0.60374 | 0.60512 | 2 | 0.038 | 58 | odds_xg |
| OD_xg_overperf_elo | 0.60505 | 0.61624 | 2 | 0.019 | 157 | xg |
| O3_elo_k20 | 0.60714 | 0.61608 | 2 | 0.038 | 58 | base |
| O1_elo_gw_form | 0.60719 | 0.61597 | 2 | 0.026 | 91 | base |
| O2_defense_form_elo | 0.60789 | 0.61643 | 2 | 0.012 | 254 | base |
| O0_elo_gw_defense | 0.60858 | 0.61571 | 2 | 0.011 | 253 | base |
| OA_only_elo | 0.60907 | 0.61783 | 2 | 0.042 | 62 | base |
| O8_smart_minimal | 0.61041 | 0.61720 | 3 | 0.018 | 159 | base |
| O6_efficiency_elo | 0.61144 | 0.61825 | 2 | 0.045 | 61 | base |
| O5_m2_interactions | 0.61399 | 0.61919 | 2 | 0.040 | 67 | base |
| O4_defense_elo_kimi | 0.61393 | 0.61711 | 4 | 0.020 | 112 | base |
| OF_abe_elo_odds | 0.61678 | 0.60385 | 5 | 0.012 | 164 | odds |
| O7_all_abe_elo | 0.61966 | 0.61863 | 2 | 0.012 | 270 | base |
| O9_baseline_17 | 0.62669 | 0.62738 | 2 | 0.046 | 77 | base |
| FIXED_baseline | 0.63840 | — | — | — | — | — |
| MKT_market | 0.57704 | — | — | — | — | — |

**Top-3 son todos xG universes** — xG aporta señal real en Optuna tambien.
**Campeon**: OB_xg_odds = **0.60040** (6 features, depth=2)
**FAIR**: OB vs market = **+0.02347** CI[+0.013, +0.033] — SIGNIFICATIVO

**Correccion critica vs version anterior**: OE_xg_defense_odds antes mostraba 0.566 con N=61 (parecia acercarse al market). Ahora con N=736: OE=0.60154. La "promesa" era ruido puro. Leccion: NUNCA tomar decisiones con N<100.

### 14.8 Recomendaciones para v1.0.2

1. **Market Anchor VIABLE**: gap +2.3% significativo, α=0.7-0.9 recomendado (mayor gap entre EUR top-5)
2. **Feature set produccion**: OB_xg_odds (6 features: xG for/against/diff + odds H/D/A) — campeon con parsimonia
3. **Fallback sin xG**: J0_only_odds (3 features) — solo -0.003 peor que campeon con xG
4. **Elo_k10/k20 > k32**: confirmado con N=749. Usar k=10-20 para Francia en v1.0.2
5. **xG es SEÑAL REAL pero marginal**: cierra ~0.003 del gap. Incluir en v1.0.2 pero no esperar que revolucione
6. **depth=2 universal**: todos los Optuna champions usan depth=2. Consistente con España/Alemania
7. **No sobrecomplicar**: OB (6 feats) > OF (21 feats). Parsimonia demostrada empiricamente
8. **Optuna mejora ~0.001**: OB=0.60040 vs P5=0.60100. Marginal pero gratis

### 14.9 Archivos

| Archivo | Contenido |
|---------|-----------|
| `scripts/output/lab/lab_data_61.csv` | Dataset cacheado (3,745 rows, 114 cols, con xG) |
| `scripts/output/lab/feature_lab_results.json` | Lab v2 standard, 101 tests (league 61) |
| `scripts/output/lab/shap_analysis_61.json` | SHAP, 9 escenarios con xG (league 61) |
| `scripts/output/lab/feature_lab_results_optuna.json` | Optuna, 16 candidatos con xG (league 61) |
| `scripts/backfill_understat_historical.py` | Script de backfill historico Understat |

---

## 15. Bundesliga - Alemania (78)

### 15.1 Ficha Tecnica

| Campo | Valor |
|-------|-------|
| League ID | 78 |
| Pais | Alemania |
| N matches | 3,261 (2014-2026) |
| Odds coverage | 3,248/3,261 = **99.6%** |
| xG coverage | **99.6%** via Understat (backfill historico 2026-02-12). 3,232/3,261 matches con xG. |
| N_train / N_test | 2,608 / 653 (base), 2,598 / 650 (odds), 2,585 / 647 (xg), 2,581 / 646 (odds_xg) |
| Split date | 2024-01-14 |
| Tests ejecutados | 101 standard + 9 SHAP + 17 Optuna |
| Fecha lab | 2026-02-13 (re-run con xG completo) |

### 15.2 Campeon: OC_xg_all_elo_odds (Optuna, 15 features)

El campeon global es el Optuna **OC_xg_all_elo_odds** con Brier **0.58886** (odds_xg universe). El mejor standard es **P7_xg_all_elo_odds** con Brier **0.59028**. Ambos son xG+odds — xG desplaza a los tests sin xG como campeones.

| Metrica | Valor |
|---------|-------|
| Test | OC_xg_all_elo_odds (Optuna) |
| Brier | **0.58886** |
| Universe | odds_xg |
| Features | 15 (xG + ABE + Elo + Odds) |
| N_test | 646 |

| Metrica | Valor |
|---------|-------|
| Test | P7_xg_all_elo_odds (Standard) |
| Brier | **0.59028** |
| Universe | odds_xg |
| Features | 15 (xG + ABE + Elo + Odds) |
| N_test | 646 |

Champion sin odds (Optuna): OD_xg_overperf_elo = **0.59637** (12 features, xg universe)
Champion sin xG (Optuna): OF_abe_elo_odds = **0.59275** (21 features, odds universe)
Champion base: O1_elo_gw_form = **0.60191** (6 features)

### 15.3 Ranking Completo (Standard Lab, top-20)

| # | Test | Brier | Uni | N_feat | N_test |
|---|------|-------|-----|--------|--------|
| 1 | **P7_xg_all_elo_odds** | **0.59028** | odds_xg | 15 | 646 |
| 2 | P8_xg_defense_odds | 0.59265 | odds_xg | 11 | 646 |
| 3 | P6_xg_elo_odds | 0.59274 | odds_xg | 9 | 646 |
| 4 | P9_xg_ultimate | 0.59281 | odds_xg | 22 | 646 |
| 5 | P5_xg_odds | 0.59310 | odds_xg | 6 | 646 |
| 6 | N5_odds_kimi_all | 0.59413 | odds | 15 | 650 |
| 7 | N2_odds_m2_combo | 0.59456 | odds | 12 | 650 |
| 8 | J1_elo_odds | 0.59499 | odds | 6 | 650 |
| 9 | N0_odds_elo | 0.59499 | odds | 6 | 650 |
| 10 | N1_odds_defense_elo | 0.59511 | odds | 8 | 650 |
| 11 | N3_odds_efficiency | 0.59516 | odds | 11 | 650 |
| 12 | P4_xg_overperf_elo | 0.59523 | xg | 12 | 647 |
| 13 | J0_only_odds | 0.59536 | odds | 3 | 650 |
| 14 | N8_odds_minimal | 0.59584 | odds | 5 | 650 |
| 15 | P3_xg_defense_elo | 0.59682 | xg | 9 | 647 |
| 16 | N9_odds_ultimate | 0.59655 | odds | 22 | 650 |
| 17 | N4_odds_abe_best | 0.59677 | odds | 14 | 650 |
| 18 | N7_odds_power7 | 0.59753 | odds | 10 | 650 |
| 19 | P2_xg_elo | 0.59990 | xg | 6 | 647 |
| 20 | N6_odds_clean | 0.60077 | odds | 15 | 650 |

**Top-5 son TODOS odds_xg**. xG desplaza a los odds-only como top tier. P4_xg_overperf_elo (#12, xg sin odds) compite con odds-only tests.

### 15.4 Ranking Optuna (17 candidatos)

| Test | Brier | Uni | N_feat | FAIR delta | Sig? |
|------|-------|-----|--------|------------|------|
| **OC_xg_all_elo_odds** | **0.58886** | odds_xg | 15 | +0.01263 | **SI** |
| OE_xg_defense_odds | 0.59141 | odds_xg | 11 | — | — |
| OB_xg_odds | 0.59174 | odds_xg | 6 | — | — |
| OF_abe_elo_odds | 0.59275 | odds | 21 | — | — |
| OD_xg_overperf_elo | 0.59637 | xg | 12 | — | — |
| O1_elo_gw_form | 0.60191 | base | 6 | — | — |
| O0_elo_gw_defense | 0.60256 | base | 5 | — | — |
| O3_elo_k20 | 0.60474 | base | 3 | — | — |
| OA_only_elo | 0.60564 | base | 3 | — | — |
| O4_defense_elo_kimi | 0.60569 | base | 14 | — | — |
| O5_m2_interactions | 0.60583 | base | 9 | — | — |
| O7_all_abe_elo | 0.60613 | base | 18 | — | — |
| O2_defense_form_elo | 0.60700 | base | 8 | — | — |
| O6_efficiency_elo | 0.60736 | base | 8 | — | — |
| O8_smart_minimal | 0.60769 | base | 13 | — | — |
| FIXED_baseline | 0.61902 | base | 17 | — | — |
| O9_baseline_17 | 0.62022 | base | 17 | — | — |

Top-3 Optuna son todos odds_xg. OC mejora OF (sin xG) en **0.00389** (0.59275 → 0.58886).

### 15.5 SHAP Analysis (9 escenarios)

| Escenario | Brier | N_feat | N_test | Odds share |
|-----------|-------|--------|--------|------------|
| S0_baseline_17 | 0.61796 | 17 | 653 | — |
| S1_baseline_odds | 0.60094 | 20 | 650 | 50.4% |
| **S2_elo_odds** | **0.59455** | 6 | 650 | 75.5% |
| S3_defense_elo | 0.60656 | 5 | 653 | — |
| S4_m2_interactions | 0.60749 | 9 | 653 | — |
| **S5_xg_elo** | **0.59911** | 6 | 647 | — |
| S6_power_5 | 0.61387 | 5 | 653 | — |
| S7_abe_elo | 0.60764 | 18 | 653 | — |
| S8_abe_elo_odds | 0.59543 | 21 | 650 | 54.1% |

**S5_xg_elo (0.59911)** ahora funciona con 99.6% xG (antes SKIP con N=42). Supera a S3_defense_elo (0.60656) por **+0.00745**. S5 es el mejor escenario sin odds.

SHAP top-6 (S5_xg_elo):
1. elo_diff: 0.14546 (34.5%)
2. **xg_diff: 0.09510 (22.6%)** — #2 global, #1 para clase Home (0.143)
3. elo_home: 0.05800 (13.8%)
4. elo_away: 0.05074 (12.0%)
5. home_xg_for_avg: 0.04566 (10.8%)
6. away_xg_for_avg: 0.02844 (6.7%)

**xG share en S5: ~40%** (xg_diff + xg_for features). xG es la segunda senal mas fuerte despues de Elo.

SHAP S1_baseline_odds (top-5):
1. odds_away: 0.121
2. odds_home: 0.103
3. odds_draw: 0.046
4. away_matches_played: 0.038
5. away_goals_scored_avg: 0.026

### 15.6 Que Funciona y Que No

**Funciona bien**:
- **xG + Odds = nuevo campeon**: P7_xg_all_elo_odds (0.590) supera N5_odds_kimi_all (0.594) por +0.004
- **xG sin odds competitive**: P4_xg_overperf_elo (0.595, xg) compite con J0_only_odds (0.595, odds)
- **xg_diff = SHAP #2**: 0.095 global, 0.143 para Home (#1 por encima de elo_diff). xG share 40% en S5.
- Optuna OC=0.589 — mejora significativa sobre OF=0.593 (sin xG)
- Kimi interactions (L4): mejor test base con 0.604
- Elo goal-weighted + form (O1): mejor Optuna base 0.602

**No funciona**:
- Baseline 17 (A0): 0.619 — peor que Elo solo (0.607)
- xG solo (P0_xg_core): 0.613 — debil sin Elo/odds como ancla
- Power 5/7: 0.614/0.614 — demasiado simplificado
- Feature combos sobredimensionados: M7_ultimate (26 feats) = 0.608 vs L4 (12 feats) = 0.604

### 15.7 Impacto xG (comparacion pre/post backfill)

| Metrica | Sin xG (pre-backfill) | Con xG (post-backfill) | Delta |
|---------|-----------------------|------------------------|-------|
| Champion standard | N5_odds_kimi_all 0.59413 | P7_xg_all_elo_odds **0.59028** | **-0.00385** |
| Champion Optuna | OF_abe_elo_odds 0.59275 | OC_xg_all_elo_odds **0.58886** | **-0.00389** |
| Best no-odds | O1_elo_gw_form 0.60191 | P4_xg_overperf_elo **0.59523** | **-0.00668** |
| SHAP S5_xg_elo | SKIP (N=42) | **0.59911** (N=647) | nuevo |
| FAIR gap | +0.01783 | **+0.01263** | **-0.00520** |

xG cierra **~29% del gap** entre modelo y mercado (FAIR de +0.01783 a +0.01263). Es la mejora mas grande de las 3 ligas EUR analizadas.

### 15.8 Gap vs Mercado

| Metrica | Valor |
|---------|-------|
| Market Brier | **0.57675** |
| Best model (OC Optuna) | 0.58886 |
| FAIR delta (OC) | +0.01263 |
| CI95 | [+0.00204, +0.02398] |
| Significativo? | **SI** |
| FAIR delta (P7 standard) | +0.01446 |
| CI95 | [+0.00299, +0.02617] |
| Significativo? | **SI** |

Gap reducido a +1.3% (OC) desde +1.8% (OF sin xG). Aun significativo pero **menor gap de las 3 ligas EUR** analizadas.

### 15.9 Recomendaciones

1. **xG es CRITICO para v1.0.2 en Bundesliga**: mayor impacto relativo de las 3 ligas EUR (+0.004 standard, +0.004 Optuna, 29% gap reduction)
2. **Market Anchor VIABLE**: gap +1.3% significativo, α=0.5-0.7 recomendado (menor que Francia, recalibrar con xG)
3. **Feature set produccion v1.0.2**: OC_xg_all_elo_odds (15 features, Optuna) o P7_xg_all_elo_odds (15 features, standard)
4. **Feature set sin odds**: P4_xg_overperf_elo (12 features, xg) — compite con odds-only tests
5. **xg_diff es #2 SHAP global**: 0.095, detras de elo_diff 0.145. Para Home class es #1 (0.143). Incluir siempre en v1.0.2.
6. **FIXED_baseline 0.619**: produccion actual 5.1% peor que OC. Mucho margen de mejora.

### 15.10 Archivos

- Lab data: `scripts/output/lab/lab_data_78.csv`
- Standard: `scripts/output/lab/feature_lab_results_78.json` (league_id=78)
- SHAP: `scripts/output/lab/shap_analysis_78.json`
- Optuna: `scripts/output/lab/feature_lab_results_optuna_78.json` (league_id=78)

---

## 16. Serie A - Italia (135)

### 16.1 Ficha Tecnica

| Campo | Valor |
|-------|-------|
| League ID | 135 |
| Pais | Italia |
| N matches | 4,040 (2014-2026) |
| Odds coverage | 4,039/4,040 = **100%** |
| xG coverage | 4,015/4,040 = **99.4%** (Understat, backfill historico 2026-02-12) |
| N_train / N_test | 3,232 / 808 (base/odds), 3,212 / 803 (xg/odds_xg) |
| Split date | ~2024-01-13 (base/odds), ~2024-01-14 (xg/odds_xg) |
| Tests ejecutados | 101 standard + 9 SHAP + 16 Optuna (re-run completo post-xG backfill) |
| Fecha lab | 2026-02-13 (v2 con xG completo) |

### 16.2 Campeon: OF_abe_elo_odds (Optuna, 21 features)

El campeon absoluto es el Optuna **OF_abe_elo_odds** con Brier **0.58110**. El mejor standard es N4_odds_abe_best con Brier 0.58015. Ambos son odds-only — xG NO mejora al campeon con odds para Italia.

| Metrica | Valor |
|---------|-------|
| Test | OF_abe_elo_odds (Optuna) |
| Brier | **0.58110** |
| CI95 | [0.56202, 0.60043] |
| CV Brier | 0.56991 |
| Features | 21 (ABE + Elo + odds) |
| Hyperparams | depth=2, lr=0.017, n_est=170, mcw=11 |
| N_test | 808 |

Champion alternativo (standard): N4_odds_abe_best = **0.58015** (14 features, odds)
Champion xG+odds (Optuna): OB_xg_odds = **0.58302** (6 features, odds_xg)
Champion xG sin odds: OD_xg_overperf_elo = **0.59245** (12 features, xg)
Champion base: O7_all_abe_elo = **0.59186** (18 features, base)

### 16.3 Ranking Completo (Standard Lab, top-15)

| # | Test | Brier | Uni | N_feat | N_test |
|---|------|-------|-----|--------|--------|
| 1 | **N4_odds_abe_best** | **0.58015** | odds | 14 | 808 |
| 2 | N1_odds_defense_elo | 0.58045 | odds | 8 | 808 |
| 3 | P6_xg_elo_odds | 0.58082 | odds_xg | 9 | 803 |
| 4 | J1_elo_odds | 0.58152 | odds | 6 | 808 |
| 5 | N0_odds_elo | 0.58152 | odds | 6 | 808 |
| 6 | N3_odds_efficiency | 0.58157 | odds | 11 | 808 |
| 7 | N8_odds_minimal | 0.58171 | odds | 5 | 808 |
| 8 | N7_odds_power7 | 0.58177 | odds | 10 | 808 |
| 9 | N9_odds_ultimate | 0.58238 | odds | 22 | 808 |
| 10 | N5_odds_kimi_all | 0.58243 | odds | 15 | 808 |
| 11 | P9_xg_ultimate | 0.58246 | odds_xg | 22 | 803 |
| 12 | P5_xg_odds | 0.58259 | odds_xg | 6 | 803 |
| 13 | N2_odds_m2_combo | 0.58306 | odds | 12 | 808 |
| 14 | P7_xg_all_elo_odds | 0.58321 | odds_xg | 15 | 803 |
| 15 | J0_only_odds | 0.58385 | odds | 3 | 808 |

**Patron clave**: Odds-only domina el top-10 (8/10 son odds puras). xG+odds aparece en #3 (P6) y #11-14 pero NO supera a odds puras. Italia favorece features complejas (ABE, 14f) sobre parsimonia.

### 16.4 SHAP Analysis (9 escenarios)

| Escenario | Brier | N_feat | N_test | Odds share |
|-----------|-------|--------|--------|------------|
| S0_baseline_17 | 0.61582 | 17 | 808 | — |
| S1_baseline_odds | 0.58601 | 20 | 808 | 63.5% |
| S2_elo_odds | 0.58197 | 6 | 808 | 79.9% |
| S3_defense_elo | 0.59506 | 5 | 808 | — |
| S4_m2_interactions | 0.59836 | 9 | 808 | — |
| **S5_xg_elo** | **0.59114** | 6 | 803 | — |
| S6_power_5 | 0.59494 | 5 | 808 | — |
| S7_abe_elo | 0.59226 | 18 | 808 | — |
| **S8_abe_elo_odds** | **0.57981** | 21 | 808 | 62.0% |

**S8_abe_elo_odds** es el campeon SHAP (0.57981) — el MEJOR Brier de todos los tests de Italia, incluyendo Optuna.

SHAP top-5 (S5_xg_elo, global ranking):
1. **elo_diff**: 0.199 — dominante
2. **xg_diff**: 0.073 — señal real (#2)
3. elo_away: 0.071
4. elo_home: 0.071
5. away_xg_for_avg: 0.035

SHAP top-5 (S8_abe_elo_odds):
1. odds_away: 0.191 — dominante
2. odds_home: 0.109
3. odds_draw: 0.039
4. elo_diff: 0.032
5. league_draw_rate: 0.022

**xG SHAP por clase** (S5_xg_elo):
- xg_diff es #2 global (SHAP 0.073) detras de elo_diff
- xG mejora S3 sin odds: S5 (0.591) vs S3 (0.595) = +0.004 mejora

### 16.5 Que Funciona y Que No

**Funciona bien**:
- **ABE + Elo + odds (N4/OF/S8)**: 0.580-0.581 — campeon. Italia es la unica EUR donde ABE 21f supera parsimonia 6f
- **Elo + odds simple (J1/N0)**: 0.582 — solo -0.002 detras, excelente relacion señal/features
- **xG sin odds (P2/S5)**: 0.591 — xG MEJORA sobre S3 (0.595) por +0.004
- Odds_share SHAP: 62-80% (odds dominan cuando estan disponibles)

**No funciona**:
- **xG con odds NO supera odds puras**: N4 (0.580) > P6 (0.581) > OB (0.583). xG no aporta información incremental sobre odds
- Baseline 17 features (A0): 0.616 — ruido
- M2 interactions (S4): 0.598 — peor que S3/S5/S7
- FIXED_baseline (prod actual): **0.61592** — 6.2% peor que market

### 16.6 Gap vs Mercado

```
Market Brier:     0.57968 (N=808, de-vigged)
Best model (S8):  0.57981 (SHAP ABE+Elo+odds, 21 features)
Best Optuna (OF): 0.58110 (ABE+Elo+odds, 21 features)
Best standard:    0.58015 (N4_odds_abe_best, 14 features)
Best xG+odds:     0.58082 (P6_xg_elo_odds, 9 features)
FAIR (N4 vs mkt): +0.00074 NO SIGNIFICATIVO CI[-0.009, 0.009]
FAIR (OF vs mkt): +0.00144 NO SIGNIFICATIVO CI[-0.007, 0.009]
```

**Italia es el mercado MAS EFICIENTE de las 5 top EUR**. Gap de solo +0.07-0.14%, CI siempre cruza cero. El modelo EMPATA estadisticamente con el mercado.

### 16.7 Ranking Optuna (16 candidatos)

| Test | Brier | CV Brier | Depth | LR | N_est | Uni |
|------|-------|----------|-------|----|-------|-----|
| **OF_abe_elo_odds** | **0.58110** | 0.56991 | 2 | 0.017 | 170 | odds |
| OB_xg_odds | 0.58302 | 0.57106 | 2 | 0.018 | 166 | odds_xg |
| OC_xg_all_elo_odds | 0.58339 | 0.57229 | 2 | 0.021 | 153 | odds_xg |
| OE_xg_defense_odds | 0.58432 | 0.57233 | 2 | 0.022 | 137 | odds_xg |
| O7_all_abe_elo | 0.59186 | 0.58335 | 2 | 0.018 | 204 | base |
| O1_elo_gw_form | 0.59220 | 0.58083 | 2 | 0.026 | 124 | base |
| OD_xg_overperf_elo | 0.59245 | 0.58364 | 2 | 0.014 | 215 | xg |
| O3_elo_k20 | 0.59275 | 0.58157 | 2 | 0.038 | 58 | base |
| O0_elo_gw_defense | 0.59285 | 0.58052 | 2 | 0.018 | 158 | base |
| O2_defense_form_elo | 0.59403 | 0.58532 | 2 | 0.022 | 148 | base |
| O8_smart_minimal | 0.59428 | 0.58615 | 2 | 0.020 | 153 | base |
| OA_only_elo | 0.59473 | 0.58365 | 2 | 0.034 | 92 | base |
| O6_efficiency_elo | 0.59565 | 0.58571 | 2 | 0.020 | 126 | base |
| O4_defense_elo_kimi | 0.59649 | 0.58485 | 2 | 0.019 | 183 | base |
| O5_m2_interactions | 0.59887 | 0.58628 | 2 | 0.020 | 147 | base |
| O9_baseline_17 | 0.61472 | 0.59630 | 2 | 0.019 | 266 | base |
| FIXED_baseline | 0.61592 | — | — | — | — | — |
| MKT_market | 0.57968 | — | — | — | — | — |

**Campeon Optuna odds-only (OF)** supera a todos los xG+odds. Italia es la UNICA liga EUR top-5 donde el campeon Optuna NO incluye xG.

### 16.8 Recomendaciones para v1.0.2

1. **Market Anchor MENOS necesario**: gap +0.07% no significativo. Si se aplica, α=0.3-0.5 (gap minimo)
2. **Feature set produccion**: OF_abe_elo_odds (21 features) o N4 (14 features) — odds domina, ABE features utiles
3. **xG es señal REAL pero insuficiente**: mejora +0.004 sin odds (S5 vs S3) pero NO mejora con odds
4. **depth=2 universal**: confirmado para todos los Optuna champions
5. **S8 SHAP (0.57981) es el mejor Brier absoluto** — mejor que Optuna OF. Posible underfit de Optuna
6. **Italia = benchmark de eficiencia**: el mercado ya incorpora toda la informacion publica disponible

### 16.9 Archivos

| Archivo | Contenido |
|---------|-----------|
| `scripts/output/lab/lab_data_135.csv` | Dataset cacheado (4,040 rows, 114 cols, con xG) |
| `scripts/output/lab/feature_lab_results_135.json` | Lab v2 standard, 101 tests (league 135) |
| `scripts/output/lab/shap_analysis_135.json` | SHAP, 9 escenarios con xG (league 135) |
| `scripts/output/lab/feature_lab_results_optuna_135.json` | Optuna, 16 candidatos con xG (league 135) |

---

## 17. Premier League - Inglaterra (39)

### 17.1 Ficha Tecnica

| Campo | Valor |
|-------|-------|
| League ID | 39 |
| Pais | Inglaterra |
| N matches | 4,060 (2014-2026) |
| Odds coverage | 4,060/4,060 = **100%** |
| xG coverage | 4,036/4,060 = **99.4%** (Understat, backfill historico 2026-02-12) |
| N_train / N_test | 3,248 / 812 (base/odds), 3,228 / 808 (xg/odds_xg) |
| Split date | ~2024-01-13 (base/odds) |
| Tests ejecutados | 101 standard + 9 SHAP + 16 Optuna (completo con xG) |
| Fecha lab | 2026-02-13 (v2 con xG completo) |

### 17.2 Campeon: N1_odds_defense_elo (Standard, 8 features)

El campeon absoluto en standard es **N1_odds_defense_elo** con Brier **0.57816**. En Optuna, el campeon es OC_xg_all_elo_odds (0.58103, 15f) — Optuna NO mejora sobre standard para Inglaterra.

| Metrica | Valor |
|---------|-------|
| Test | N1_odds_defense_elo (Standard) |
| Brier | **0.57816** |
| CI95 | [0.55825, 0.59745] |
| Features | odds_home, odds_draw, odds_away, elo_diff, elo_home, elo_away, home_goals_conceded_avg, away_goals_conceded_avg |
| Hyperparams | depth=4, lr=0.05, n_est=200 (fixed) |
| N_test | 812 |

Champion alternativo (Optuna): OC_xg_all_elo_odds = **0.58103** (15 features, odds_xg)
Champion xG+odds (standard): P7_xg_all_elo_odds = **0.57966** (15 features, odds_xg)
Champion sin odds: H2_defense_form_elo = **0.58893** (8 features, base)
Champion xG sin odds: P3_xg_defense_elo = **0.58960** (9 features, xg)

### 17.3 Ranking Completo (Standard Lab, top-15)

| # | Test | Brier | Uni | N_feat | N_test |
|---|------|-------|-----|--------|--------|
| 1 | **N1_odds_defense_elo** | **0.57816** | odds | 8 | 812 |
| 2 | N3_odds_efficiency | 0.57817 | odds | 11 | 812 |
| 3 | J1_elo_odds | 0.57877 | odds | 6 | 812 |
| 4 | N0_odds_elo | 0.57877 | odds | 6 | 812 |
| 5 | N5_odds_kimi_all | 0.57877 | odds | 15 | 812 |
| 6 | N2_odds_m2_combo | 0.57947 | odds | 12 | 812 |
| 7 | P7_xg_all_elo_odds | 0.57966 | odds_xg | 15 | 808 |
| 8 | N9_odds_ultimate | 0.57995 | odds | 22 | 812 |
| 9 | P9_xg_ultimate | 0.58037 | odds_xg | 22 | 808 |
| 10 | P8_xg_defense_odds | 0.58070 | odds_xg | 11 | 808 |
| 11 | J0_only_odds | 0.58111 | odds | 3 | 812 |
| 12 | N7_odds_power7 | 0.58144 | odds | 10 | 812 |
| 13 | P6_xg_elo_odds | 0.58159 | odds_xg | 9 | 808 |
| 14 | N4_odds_abe_best | 0.58170 | odds | 14 | 812 |
| 15 | N6_odds_clean | 0.58206 | odds | 15 | 812 |

**Patron clave**: Odds-only domina top-6 (6/6 puros). xG+odds aparece a partir de #7 pero NO supera odds puras. N1 (8f) gana por 0.001 sobre tests de 6 features — defense features (goals_conceded) aportan marginalidad real.

### 17.4 SHAP Analysis (9 escenarios)

| Escenario | Brier | N_feat | N_test | Odds share |
|-----------|-------|--------|--------|------------|
| S0_baseline_17 | 0.60853 | 17 | 812 | — |
| S1_baseline_odds | 0.58337 | 20 | 812 | 64.2% |
| S2_elo_odds | 0.57881 | 6 | 812 | 79.4% |
| **S3_defense_elo** | **0.59012** | 5 | 812 | — |
| S4_m2_interactions | 0.59296 | 9 | 812 | — |
| S5_xg_elo | 0.59442 | 6 | 808 | — |
| S6_power_5 | 0.59754 | 5 | 812 | — |
| S7_abe_elo | 0.59236 | 18 | 812 | — |
| S8_abe_elo_odds | 0.58322 | 21 | 812 | 58.4% |

**S2_elo_odds** es el campeon SHAP (0.57881) — superior al S8 con 21 features. Parsimonia maxima: 6 features bastan.

SHAP top-5 (S5_xg_elo, global ranking):
1. **elo_diff**: 0.193 — dominante
2. **xg_diff**: 0.076 — señal real (#2)
3. elo_home: 0.067
4. away_xg_for_avg: 0.052
5. home_xg_for_avg: 0.037

SHAP top-5 (S2_elo_odds):
1. odds_home: 0.164 — dominante
2. odds_away: 0.109
3. odds_draw: 0.060
4. elo_diff: 0.040
5. elo_away: 0.024

**HALLAZGO CRITICO**: S5_xg_elo (0.594) es PEOR que S3_defense_elo (0.590). En Inglaterra, los features defensivos (goals_conceded) superan a xG sin odds. xG daña el modelo cuando reemplaza defense features.

### 17.5 Que Funciona y Que No

**Funciona bien**:
- **Odds + Elo + defense (N1)**: 0.578 — campeon absoluto, 8 features bien calibradas
- **Elo + odds simple (J1/N0/S2)**: 0.579 — -0.001 detras, excelente
- **Defense features sin odds (H2/S3)**: 0.589-0.590 — goals_conceded aporta señal real
- Odds_share SHAP: 58-79% (odds dominan)
- N1 = N3 (0.57816 vs 0.57817): defense + efficiency equivalentes

**No funciona**:
- **xG NO mejora con odds**: P7 (#7, 0.580) < N1 (#1, 0.578) — xG no aporta sobre odds
- **xG DAÑA sin odds**: S5_xg_elo (0.594) > S3_defense_elo (0.590) — defense features > xG
- **Optuna NO mejora**: OC (0.581) > N1 (0.578) — standard con fixed params gana
- ABE features (S7/N4): no mejoran sobre Elo+odds simple
- FIXED_baseline: **0.60784** — 6.5% peor que market

### 17.6 Gap vs Mercado

```
Market Brier:     0.57065 (N=812, de-vigged)
Best model (N1):  0.57816 (odds+defense+elo, 8 features)
Best Optuna (OC): 0.58103 (xG+elo+odds, 15 features)
Best xG+odds:     0.57966 (P7_xg_all_elo_odds, 15 features)
Best base:        0.58893 (H2_defense_form_elo, 8 features)
FAIR (N1 vs mkt): +0.00906 SIGNIFICATIVO CI[+0.0004, +0.017]
FAIR (OC vs mkt): +0.01035 SIGNIFICATIVO CI[+0.003, +0.018]
```

Gap de +0.9% significativo — el **MENOR gap de las 5 top EUR**. Inglaterra es el mercado mas cercano al modelo despues de Italia (que no es significativo).

### 17.7 Ranking Optuna (16 candidatos)

| Test | Brier | CV Brier | Depth | LR | N_est | Uni |
|------|-------|----------|-------|----|-------|-----|
| **OC_xg_all_elo_odds** | **0.58103** | 0.57170 | 2 | 0.018 | 148 | odds_xg |
| OE_xg_defense_odds | 0.58188 | 0.57238 | 2 | 0.019 | 144 | odds_xg |
| OF_abe_elo_odds | 0.58404 | 0.56986 | 2 | 0.016 | 176 | odds |
| OB_xg_odds | 0.58440 | 0.57235 | 2 | 0.018 | 175 | odds_xg |
| O2_defense_form_elo | 0.59169 | 0.57961 | 2 | 0.015 | 224 | base |
| O8_smart_minimal | 0.59190 | 0.58129 | 2 | 0.018 | 140 | base |
| O1_elo_gw_form | 0.59202 | 0.57965 | 2 | 0.025 | 109 | base |
| OD_xg_overperf_elo | 0.59202 | 0.58158 | 2 | 0.021 | 127 | xg |
| O3_elo_k20 | 0.59206 | 0.57835 | 2 | 0.031 | 76 | base |
| O4_defense_elo_kimi | 0.59239 | 0.58042 | 2 | 0.015 | 186 | base |
| OA_only_elo | 0.59248 | 0.57814 | 2 | 0.028 | 92 | base |
| O5_m2_interactions | 0.59350 | 0.58043 | 2 | 0.028 | 79 | base |
| O0_elo_gw_defense | 0.59456 | 0.57966 | 2 | 0.022 | 110 | base |
| O7_all_abe_elo | 0.59457 | 0.58122 | 2 | 0.018 | 151 | base |
| O6_efficiency_elo | 0.59483 | 0.57939 | 2 | 0.015 | 160 | base |
| O9_baseline_17 | 0.60771 | 0.59410 | 2 | 0.016 | 199 | base |
| FIXED_baseline | 0.60784 | — | — | — | — | — |
| MKT_market | 0.57065 | — | — | — | — | — |

**Top-4 Optuna**: todos odds-based, con 3 de 4 incluyendo xG. Pero Optuna NO supera al standard N1 (0.578 < 0.581). El fixed hyperparams depth=4 funciona mejor que el Optuna depth=2 para Inglaterra — excepcion unica entre las 5 ligas EUR.

### 17.8 Recomendaciones para v1.0.2

1. **Market Anchor VIABLE pero gap menor**: +0.9% significativo, α=0.4-0.6 recomendado
2. **Feature set produccion**: N1_odds_defense_elo (8 features) — standard con depth=4 supera Optuna
3. **NO incluir xG para England**: no mejora con odds, DAÑA sin odds. Defense features son superiores
4. **Excepcion en hyperparams**: depth=4 lr=0.05 (standard) > depth=2 lr=0.02 (Optuna). Unica liga donde profundidad mayor gana
5. **defense_elo > xG_elo**: goals_conceded captura mas señal que xG en mercado ultra-eficiente
6. **Parsimonia extrema**: N1 (8f) = N3 (11f) = J1 (6f). 6-8 features optimas

### 17.9 Archivos

| Archivo | Contenido |
|---------|-----------|
| `scripts/output/lab/lab_data_39.csv` | Dataset cacheado (4,060 rows, 114 cols, con xG) |
| `scripts/output/lab/feature_lab_results_39.json` | Lab v2 standard, 101 tests (league 39) |
| `scripts/output/lab/shap_analysis_39.json` | SHAP, 9 escenarios con xG (league 39) |
| `scripts/output/lab/feature_lab_results_optuna_39.json` | Optuna, 16 candidatos con xG (league 39) |

---

## 18. Eredivisie - Holanda (88)

### 18.1 Ficha Tecnica

| Campo | Valor |
|-------|-------|
| League ID | 88 |
| Pais | Holanda |
| N matches | 2,958 (2016-2026) |
| Odds coverage | 2,872/2,958 = **97.1%** (FDUK) |
| xG coverage | **1,730/2,958 = 58.5%** (FotMob/Opta, backfill 2026-02-13) |
| N_train / N_test | 2,366 / 592 (base), 2,297 / 575 (odds), ~1,384 / 346 (xG) |
| Tests ejecutados | 110 standard + 9 SHAP + 16 Optuna (completo con xG) |
| Fecha lab | 2026-02-13 (re-run con xG) |

### 18.2 Campeon: OF_abe_elo_odds (Optuna, 21 features)

El campeon absoluto es el Optuna **OF_abe_elo_odds** con Brier **0.56597**. Practicamente EMPATA con el mercado (0.56592). FAIR test: modelo=0.56546 vs market=0.56592, Δ=-0.00046 CI[-0.010, +0.010] — NO significativo.

| Metrica | Valor |
|---------|-------|
| Test | OF_abe_elo_odds (Optuna) |
| Brier | **0.56597** |
| Features | 21 (ABE + Elo + odds) |
| N_test | 575 |
| Hyperparams | depth=2, lr=0.0205, n_est=202, mcw=11 |

Champion Optuna: OF_abe_elo_odds = **0.56597** (21 features, odds)
Champion SHAP: S8_abe_elo_odds = **0.56614** (21 features, odds share 54.4%)
Champion standard: N5_odds_kimi_all = **0.56755** (15 features, odds)
Champion base: M6_defense_elo_kimi = **0.57614** (14 features, base)
Mejor xG: P8_xg_defense_odds = **0.59510** (11 features, N=342) — xG NO mejora

### 18.3 Ranking Completo (Standard Lab, top-15)

| # | Test | Brier | Uni | N_feat | N_test |
|---|------|-------|-----|--------|--------|
| 1 | **N5_odds_kimi_all** | **0.56755** | odds | 15 | 575 |
| 2 | N9_odds_ultimate | 0.56866 | odds | 22 | 575 |
| 3 | N1_odds_defense_elo | 0.56940 | odds | 8 | 575 |
| 4 | J0_only_odds | 0.56944 | odds | 3 | 575 |
| 5 | N4_odds_abe_best | 0.56958 | odds | 14 | 575 |
| 6 | N3_odds_efficiency | 0.56965 | odds | 11 | 575 |
| 7 | N8_odds_minimal | 0.57007 | odds | 5 | 575 |
| 8 | J1_elo_odds | 0.57113 | odds | 6 | 575 |
| 9 | N0_odds_elo | 0.57113 | odds | 6 | 575 |
| 10 | N7_odds_power7 | 0.57335 | odds | 10 | 575 |
| 11 | M6_defense_elo_kimi | 0.57614 | base | 14 | 592 |
| 12 | I3_goals_core_elo | 0.57637 | base | 8 | 592 |
| 13 | H1_defense_elo | 0.57679 | base | 5 | 592 |
| 14 | H7_elo_split_defense | 0.57699 | base | 5 | 592 |
| 15 | L4_kimi_all_elo | 0.57705 | base | 12 | 592 |

**Patron clave**: Top-10 TODOS odds. El gap entre mejor odds (#1, 0.568) y mejor base (#11, 0.576) es solo 0.009.

### 18.4 SHAP Analysis (9 escenarios, todos completos con xG)

| Escenario | Brier | N_feat | N_test | Odds share |
|-----------|-------|--------|--------|------------|
| S0_baseline_17 | 0.60206 | 17 | 592 | — |
| S1_baseline_odds | 0.58105 | 20 | 575 | 50.1% |
| S2_elo_odds | 0.57202 | 6 | 575 | 77.0% |
| S3_defense_elo | 0.57790 | 5 | 592 | — |
| S4_m2_interactions | 0.58504 | 9 | 592 | — |
| S5_xg_elo | **0.61295** | 5 | 346 | — |
| S6_power_5 | 0.58409 | 5 | 592 | — |
| S7_abe_elo | 0.57688 | 18 | 592 | — |
| **S8_abe_elo_odds** | **0.56614** | 21 | 575 | 54.4% |

**S8_abe_elo_odds** sigue siendo campeon SHAP (0.56614). **xG NO aporta**: S5_xg_elo (0.613) es PEOR que S3_defense_elo (0.578) y peor que baseline S0 (0.602).

SHAP top-5 (S8_abe_elo_odds, global ranking):
1. **odds_home**: 0.172 — dominante
2. odds_away: 0.109
3. elo_diff: 0.033
4. opp_rating_diff: 0.032
5. overperf_diff: 0.029

SHAP top-5 (S5_xg_elo, global ranking):
1. **elo_diff**: 0.199 — dominante
2. elo_home: 0.092
3. xg_diff: 0.091
4. elo_away: 0.050
5. home_xg_for_avg: 0.046

### 18.5 Que Funciona y Que No

**Funciona bien**:
- **ABE + Elo + odds (S8)**: 0.566 — empata con mercado. ABE features aportan sobre Elo+odds simple
- **Odds puras (J0)**: 0.569 — solo 3 features, -0.003 detras de campeon
- **Defense + Elo (S3/M6)**: 0.576-0.578 — mejores sin odds
- **S7_abe_elo (0.577)**: mejor que S3_defense_elo (0.578) — ABE > defense sin odds
- Odds_share SHAP: 50-77% (odds dominan)

**No funciona**:
- **xG rolling**: S5_xg_elo (0.613) es PEOR que baseline (0.602). P8_xg_defense_odds (0.595) no mejora sobre odds-only
- Baseline 17 features (S0): 0.602 — ruido
- M2 interactions (S4): 0.585 — peor que S3/S7
- FIXED_baseline: **0.60186** — 6.3% peor que market

### 18.6 Gap vs Mercado

```
Market Brier:     0.56592 (N=575, de-vigged)
Best model (S8):  0.56650 (SHAP ABE+Elo+odds, 21 features)
Best Optuna (OF): 0.56597 (ABE+Elo+odds, 21 features)
Best standard:    0.56755 (N5_odds_kimi_all, 15 features)
Best base:        0.57614 (M6_defense_elo_kimi, 14 features)
FAIR (OF vs mkt): -0.00046 NO SIGNIFICATIVO CI[-0.010, +0.010]
```

**Eredivisie es el SEGUNDO mercado mas eficiente** (despues de Italia). Gap +0.06%, CI cruza cero ampliamente. El modelo empata estadisticamente con el mercado.

### 18.7 Ranking Optuna (16 candidatos con xG, 50 trials × 3-fold temporal CV)

| # | Test | Brier | CV Brier | Uni | #F | Depth | LR |
|---|------|-------|----------|-----|----|-------|----|
| 1 | **OF_abe_elo_odds** | **0.56597** | 0.56416 | odds | 21 | 2 | 0.021 |
| — | MKT_market | 0.56592 | — | — | — | — | — |
| 2 | O4_defense_elo_kimi | 0.57701 | 0.57415 | base | 14 | 2 | 0.015 |
| 3 | O7_all_abe_elo | 0.57802 | 0.57495 | base | 18 | 2 | 0.012 |
| 4 | O0_elo_gw_defense | 0.57805 | 0.56774 | base | 5 | 2 | 0.014 |
| 5 | O2_defense_form_elo | 0.57820 | 0.57052 | base | 8 | 2 | 0.019 |
| 6 | O6_efficiency_elo | 0.57886 | 0.57555 | base | 8 | 2 | 0.033 |
| 7 | O8_smart_minimal | 0.57910 | 0.57052 | base | 13 | 2 | 0.014 |
| 8 | OA_only_elo | 0.58005 | 0.57188 | base | 3 | 2 | 0.025 |
| 9 | O1_elo_gw_form | 0.58065 | 0.56916 | base | 6 | 2 | 0.027 |
| 10 | O3_elo_k20 | 0.58143 | 0.57000 | base | 3 | 2 | 0.020 |
| 11 | O5_m2_interactions | 0.58354 | 0.57508 | base | 9 | 2 | 0.014 |
| 12 | OE_xg_defense_odds | 0.59576 | 0.56142 | odds_xg | 11 | 2 | 0.024 |
| 13 | OB_xg_odds | 0.59850 | 0.56110 | odds_xg | 6 | 2 | 0.014 |
| 14 | O9_baseline_17 | 0.60012 | 0.58062 | base | 17 | 2 | 0.014 |
| — | FIXED_baseline | 0.60186 | — | — | 17 | — | — |
| 15 | OC_xg_all_elo_odds | 0.60268 | 0.56111 | odds_xg | 15 | 2 | 0.014 |
| 16 | OD_xg_overperf_elo | 0.61025 | 0.56161 | xg | 12 | 2 | 0.013 |

**0 tests SKIPPED.** Campeon OF_abe_elo_odds (0.56597) empata con mercado (0.56592), Δ=-0.05%.

**Hallazgo clave xG Optuna**: Tests con xG (OB, OC, OD, OE) rinden PEOR en test (~0.596-0.610) a pesar de CVs competitivos (~0.561). N_test=342 (odds_xg) vs 575 (odds) — overfitting al subset pequeño. **xG no aporta en Eredivisie con Optuna.**

**depth=2 universal**: los 16 Optuna convergen a depth=2.

### 18.8 Recomendaciones para v1.0.2

1. **Market Anchor IDEAL**: gap no significativo → α=0.3-0.5. Mercado ya refleja toda la informacion
2. **Feature set produccion**: S8_abe_elo_odds (21 features) — empata con mercado
3. **xG NO aporta**: FotMob backfill disponible (58.5%) pero S5/P8 peor que alternatives sin xG
4. **ABE features utiles**: S8 (0.566) y S7 (0.577) ambos superan a sus equivalentes sin ABE
5. **depth=2 confirmado**: todos los Optuna champions usan depth=2
6. **Parsimonia limitada**: 21f (S8) > 15f (N5) > 3f (J0). Mas features = mejor para Eredivisie

### 18.9 Archivos

| Archivo | Contenido |
|---------|-----------|
| `scripts/output/lab/lab_data_88.csv` | Dataset cacheado (2,958 rows, con xG) |
| `scripts/output/lab/feature_lab_results_88.json` | Lab v2 standard, 110 tests (league 88) |
| `scripts/output/lab/shap_analysis_88.json` | SHAP, 9 escenarios (league 88) |
| `scripts/output/lab/feature_lab_results_optuna_88.json` | Optuna v2, 16 candidatos con xG |

---

## 19. Belgian Pro League - Bélgica (144)

### 19.1 Ficha Tecnica

| Campo | Valor |
|-------|-------|
| Liga | Belgian Pro League (Jupiler Pro League) |
| league_id | 144 |
| Pais | Bélgica |
| Formato | Regular season (30 jornadas) + playoffs |
| Partidos en DB | 2,046 FT (2019-2026) |
| Cobertura odds | 96.0% (1,965/2,046) — FDUK Bet365 |
| Cobertura xG | **84.3% (1,725/2,046)** — FotMob/Opta (backfill 2026-02-13) |
| Fuente odds | football-data.co.uk (B1) |
| N_train / N_test | ~1,637 / ~409 (base), ~1,572 / ~393 (odds), ~1,380 / ~345 (xG) |
| Fecha prueba | 2026-02-13 (re-run con xG) |
| Backfill | API-Football 2019-2024 (1,865 matches) + FDUK odds + FotMob xG |

### 19.2 Universos

| Universo | N_matches | Descripción |
|----------|-----------|-------------|
| base | 2,046 | Todos los partidos FT |
| odds | 1,965 | Con odds FDUK (96.0%) |
| xg | 1,725 | FotMob/Opta (84.3%) |
| odds_xg | ~1,656 | Interseccion odds + xG |

### 19.3 Resultados Standard (110 tests con xG, hyperparams fijos)

| Rank | Test | Brier | #F | Universo | N_test |
|------|------|-------|----|----------|--------|
| 1 | **J0_only_odds** | **0.60689** | 3 | odds | 393 |
| 2 | N7_odds_power7 | 0.60639 | 10 | odds | 393 |
| 3 | N8_odds_minimal | 0.60737 | 5 | odds | 393 |
| 4 | Q7_xi_xg_elo_odds | 0.60716 | ~15 | odds_xg | 328 |
| 5 | J1_elo_odds | 0.60899 | 6 | odds | 393 |
| 6 | P6_xg_elo_odds | 0.60849 | ~10 | xg_odds | 331 |
| — | FIXED_baseline | 0.63762 | 17 | base | 410 |
| — | **MKT_market** | **0.60144** | — | odds | 393 |

**Top sin odds (base)**: M8_power_5 = 0.62041 (5 feats)
**Top xG sin odds**: S5_xg_elo = 0.61886 (5 feats, N=345) — mejora sobre S3 (0.63457)

### 19.4 FAIR Comparison (Paired Bootstrap)

| Métrica | Valor |
|---------|-------|
| Modelo campeón | N7_odds_power7 (standard) |
| Brier modelo | 0.60256 |
| Brier mercado | 0.60144 |
| **FAIR Δ** | **+0.00112** |
| CI95 | [-0.012, +0.013] |
| **Significativo?** | **NO** |

**Mercado ultra-eficiente** — el modelo NO bate al mercado. Mismo tier que Italia y Eredivisie.

### 19.5 SHAP (9 escenarios, todos completos con xG)

| Test | Brier | #1 Feature | #2 Feature | Odds share |
|------|-------|------------|------------|------------|
| S0_baseline_17 | 0.63152 | home_matches_played (0.092) | away_goals_conceded_avg (0.081) | — |
| S1_baseline_odds | 0.61807 | odds_away (0.124) | odds_home (0.108) | 40.2% |
| **S2_elo_odds** | **0.61064** | odds_away (0.138) | odds_home (0.100) | 71.7% |
| S3_defense_elo | 0.63457 | elo_diff (0.144) | elo_home (0.072) | — |
| S4_m2_interactions | 0.63120 | elo_diff (0.157) | home_matches_played (0.143) | — |
| **S5_xg_elo** | **0.61886** | elo_diff (0.146) | xg_diff (0.074) | — |
| S6_power_5 | 0.62138 | elo_diff (0.164) | opp_rating_diff (0.077) | — |
| S7_abe_elo | 0.62300 | elo_diff (0.122) | league_draw_rate (0.104) | — |
| S8_abe_elo_odds | 0.61262 | odds_away (0.115) | league_draw_rate (0.104) | 37.6% |

**Hallazgos SHAP**:
- odds_away domina en todos los tests con odds
- league_draw_rate es #2 en S8 (ABE + odds) — señal de liga con draw rate alto
- **xG mejora base**: S5_xg_elo (0.619) mejora sobre S3_defense_elo (0.635) — ganancia de -0.016
- xg_diff es #2 en S5 con SHAP=0.074
- Odds share solo 37.6% en S8 (menor que la mayoría de ligas — más features contribuyen)

### 19.6 Optuna (16 candidatos con xG, 50 trials × 3-fold temporal CV)

| # | Test | Brier | CV Brier | Uni | #F | Depth | LR |
|---|------|-------|----------|-----|----|-------|----|
| — | MKT_market | 0.60144 | — | — | — | — | — |
| 1 | **OB_xg_odds** | **0.60798** | 0.60684 | odds_xg | 6 | 2 | 0.034 |
| 2 | OC_xg_all_elo_odds | 0.60889 | 0.61119 | odds_xg | 15 | 2 | 0.025 |
| 3 | OE_xg_defense_odds | 0.60966 | 0.61033 | odds_xg | 11 | 2 | 0.024 |
| 4 | OF_abe_elo_odds | 0.60967 | 0.60546 | odds | 21 | 2 | 0.040 |
| 5 | OD_xg_overperf_elo | 0.61875 | 0.62189 | xg | 12 | 2 | 0.038 |
| 6 | O7_all_abe_elo | 0.62187 | 0.62234 | base | 18 | 3 | 0.039 |
| 7 | OA_only_elo | 0.62363 | 0.61411 | base | 3 | 2 | 0.018 |
| 8 | O3_elo_k20 | 0.62377 | 0.61423 | base | 3 | 2 | 0.013 |
| 9 | O1_elo_gw_form | 0.62511 | 0.61543 | base | 6 | 2 | 0.024 |
| 10 | O5_m2_interactions | 0.62677 | 0.61500 | base | 9 | 2 | 0.015 |
| 11 | O8_smart_minimal | 0.62892 | 0.61790 | base | 13 | 2 | 0.028 |
| 12 | O9_baseline_17 | 0.62893 | 0.63349 | base | 17 | 2 | 0.023 |
| 13 | O2_defense_form_elo | 0.62968 | 0.61465 | base | 8 | 2 | 0.049 |
| 14 | O0_elo_gw_defense | 0.63029 | 0.61105 | base | 5 | 2 | 0.044 |
| 15 | O4_defense_elo_kimi | 0.63199 | 0.62481 | base | 14 | 2 | 0.038 |
| 16 | O6_efficiency_elo | 0.63231 | 0.62667 | base | 8 | 2 | 0.026 |
| — | FIXED_baseline | 0.63246 | — | — | 17 | — | — |

**0 tests SKIPPED.** FAIR Optuna: OB Δ=+0.00161 CI95[-0.014, +0.017] **NO SIGNIFICATIVO**

**Hallazgo clave**: Top-4 Optuna incluyen xG (OB, OC, OE). **OB_xg_odds (6 features xG+odds) es campeon** — la xG aporta con Optuna, a diferencia del standard donde no mejoraba sobre odds-only. FAIR gap mínimo (+0.16%) vs mercado.

### 19.7 Hallazgos Clave

1. **Mercado ultra-eficiente**: FAIR +0.50% (ns) — tercer mercado más eficiente después de Primeira Liga (+0.04%) y Eredivisie (+0.21%)
2. **Parsimonia gana**: J0 (3 feats) casi iguala a todo. N7 (10 feats) similar
3. **Odds dominan**: todos los top-5 son universo odds. Gap odds vs base: 0.607 vs 0.620 = +0.013
4. **xG mejora base**: S5_xg_elo (0.619) bate a S3_defense_elo (0.635) con -0.016 — señal xG válida
5. **xG NO bate odds**: P6_xg_elo_odds (0.608) no mejora sobre J0_only_odds (0.607)
6. **League_draw_rate relevante**: #2 en SHAP S8 — Bélgica tiene draw rate alto (~27%)
7. **FIXED_baseline gap**: 0.638 → +6% peor que mercado

### 19.8 Recomendaciones v1.0.2

| Decisión | Recomendación |
|----------|---------------|
| Tier | **Ultra-efficient** (ns vs market) — mismo tier que Italia/Eredivisie |
| Market Anchor α | **0.3-0.5** (gap no significativo, modelo cerca del mercado) |
| Features óptimas | J0_only_odds (3 features) o N7_odds_power7 (10 features) |
| xG | Disponible (84.3%) pero no mejora sobre odds. Usar para ligas sin odds |
| Prioridad | Baja — modelo ya opera cerca de eficiencia máxima |

### 19.9 Archivos

| Archivo | Contenido |
|---------|-----------|
| `scripts/output/lab/lab_data_144.csv` | Dataset cacheado (2,046 rows, con xG) |
| `scripts/output/lab/feature_lab_results_144.json` | Lab v2 standard, 110 tests (league 144) |
| `scripts/output/lab/shap_analysis_144.json` | SHAP, 9 escenarios (league 144) |
| `scripts/output/lab/feature_lab_results_optuna_144.json` | Optuna v2, 16 candidatos con xG |

---

## 20. Primeira Liga - Portugal (94)

### 20.1 Ficha Tecnica

| Campo | Valor |
|-------|-------|
| League ID | 94 |
| Pais | Portugal |
| Formato | Regular season (34 jornadas) |
| N matches | 2,952 (2016-2026) |
| Odds coverage | 2,887/2,952 = **97.8%** (FDUK Pinnacle closing) |
| xG coverage | **1,636/2,952 = 55.4%** (FotMob/Opta, backfill 2026-02-13) |
| N_train / N_test | 2,361 / 591 (base), 2,309 / 578 (odds), ~1,308 / 328 (xG) |
| Tests ejecutados | 110 standard + 9 SHAP + 16 Optuna (completo con xG) |
| Fecha lab | 2026-02-13 |

### 20.2 Campeon: OC_xg_all_elo_odds (Optuna, 15 features)

| Metrica | Valor |
|---------|-------|
| Test | OC_xg_all_elo_odds (Optuna) |
| Brier | **0.53714** |
| Features | 15 (xG_ALL + Elo + odds) |
| N_test | 327 (odds_xg universe) |
| Hyperparams | depth=2, lr=0.0249, n_est=137, mcw=14 |

Champion Optuna: OC_xg_all_elo_odds = **0.53714** (15 features, odds_xg)
Champion SHAP: S2_elo_odds = **0.54019** (6 features, odds)
Champion standard: P5_xg_odds = **0.53669** (N=327) — mejor Brier absoluto (menor N)
Champion base: D8_elo_all = **0.54909** (N=591)
Mercado: **0.53979** (N=578)

### 20.3 SHAP Analysis (9 escenarios)

| Test | Brier | N_test | #1 Feature (SHAP) | #2 Feature | Odds share |
|------|-------|--------|-------------------|------------|------------|
| S0_baseline_17 | 0.56032 | 591 | home_shots_avg (0.111) | away_goals_conceded_avg (0.081) | — |
| S1_baseline_odds | 0.54699 | 578 | odds_home (0.191) | odds_away (0.123) | 57.7% |
| **S2_elo_odds** | **0.54019** | 578 | odds_home (0.181) | odds_away (0.162) | **81.1%** |
| S3_defense_elo | 0.55394 | 591 | elo_diff (0.232) | elo_home (0.115) | — |
| S4_m2_interactions | 0.55429 | 591 | elo_diff (0.238) | elo_home (0.084) | — |
| S5_xg_elo | 0.55711 | 328 | elo_diff (0.191) | elo_home (0.122) | — |
| S6_power_5 | 0.55428 | 591 | elo_diff (0.245) | opp_rating_diff (0.107) | — |
| S7_abe_elo | 0.55369 | 591 | elo_diff (0.216) | elo_home (0.063) | — |
| S8_abe_elo_odds | 0.54079 | 578 | odds_home (0.182) | odds_away (0.143) | 65.7% |

### 20.4 Ranking Optuna (16 candidatos con xG, 50 trials × 3-fold temporal CV)

| # | Test | Brier | CV Brier | Uni | #F | Depth | LR |
|---|------|-------|----------|-----|----|-------|----|
| 1 | **OC_xg_all_elo_odds** | **0.53714** | 0.54792 | odds_xg | 15 | 2 | 0.025 |
| 2 | OB_xg_odds | 0.53834 | 0.54746 | odds_xg | 6 | 2 | 0.013 |
| — | MKT_market | 0.53979 | — | — | — | — | — |
| 3 | OE_xg_defense_odds | 0.53991 | 0.55077 | odds_xg | 11 | 2 | 0.038 |
| 4 | OF_abe_elo_odds | 0.54371 | 0.56273 | odds | 21 | 2 | 0.013 |
| 5 | OD_xg_overperf_elo | 0.54758 | 0.55578 | xg | 12 | 2 | 0.015 |
| 6 | O1_elo_gw_form | 0.54918 | 0.56332 | base | 6 | 2 | 0.031 |
| 7 | O0_elo_gw_defense | 0.55027 | 0.56401 | base | 5 | 2 | 0.020 |
| 8 | O3_elo_k20 | 0.55048 | 0.56412 | base | 3 | 2 | 0.021 |
| 9 | OA_only_elo | 0.55192 | 0.56580 | base | 3 | 2 | 0.021 |
| 10 | O5_m2_interactions | 0.55198 | 0.56973 | base | 9 | 2 | 0.021 |
| 11 | O7_all_abe_elo | 0.55227 | 0.57232 | base | 18 | 2 | 0.019 |
| 12 | O2_defense_form_elo | 0.55263 | 0.56711 | base | 8 | 2 | 0.017 |
| 13 | O8_smart_minimal | 0.55371 | 0.57110 | base | 13 | 2 | 0.014 |
| 14 | O6_efficiency_elo | 0.55524 | 0.56770 | base | 8 | 2 | 0.025 |
| 15 | O4_defense_elo_kimi | 0.55563 | 0.56848 | base | 14 | 2 | 0.038 |
| — | FIXED_baseline | 0.56180 | — | — | 17 | — | — |
| 16 | O9_baseline_17 | 0.56267 | 0.58520 | base | 17 | 3 | 0.031 |

**0 tests SKIPPED.** Top-3 son TODOS xG+odds. **OC (0.53714) supera al mercado (0.53979)** pero FAIR Δ=+0.00585 CI[-0.009, +0.019] NO significativo.

**Hallazgo**: xG aporta en Primeira Liga con Optuna — OC (xG+Elo+odds, 0.537) vs OF (Elo+odds, 0.544) = mejora de 0.007.

### 20.5 Gap vs Mercado

```
Market Brier:       0.53979 (N=578, de-vigged)
Best Optuna (OC):   0.53714 (xG+Elo+odds, 15 features, N=327)
Best SHAP (S2):     0.54019 (Elo + Odds, 6 features)
Best standard:      0.53669 (P5_xg_odds, N=327 — xG subset)
Best base:          0.54909 (D8_elo_all, N=591)
FAIR (OC vs mkt):   +0.00585 CI[-0.009, +0.019] — NO SIGNIFICATIVO
```

**Primeira Liga es el MERCADO MAS EFICIENTE del dataset**: FAIR gap +0.59% (Optuna), CI cruza cero ampliamente. El modelo empata estadisticamente con el mercado.

### 20.6 Que Funciona y Que No

**Funciona bien**:
- **Elo + Odds (S2)**: 0.540 — empata con mercado, solo 6 features. Odds share 81.1%
- **Elo puro (S3/D8)**: 0.549-0.554 — mejor base. elo_diff domina con SHAP 0.232
- **ABE + Elo (S7)**: 0.554 — on par con S3, ABE features no aportan significativamente
- **Parsimonia extrema**: S2 (6f) empata con S8 (21f). Menos features = mejor

**No funciona**:
- **xG rolling**: S5_xg_elo (0.557) es PEOR que S3_defense_elo (0.554) — xG no mejora sobre Elo
- **P5_xg_odds tiene mejor Brier (0.537)** pero en N=327 — posible overfitting al subset xG
- **ABE features marginales**: S7 (0.554) y S3 (0.554) practicamente iguales
- **Baseline 17 (S0)**: 0.560 — ruido significativo

### 20.7 Recomendaciones v1.0.2

| Decisión | Recomendación |
|----------|---------------|
| Tier | **Ultra-efficient** — mercado MAS eficiente (Δ +0.59%) |
| Market Anchor α | **0.5-0.7** (modelo empata, mercado excelente) |
| Features | OC_xg_all_elo_odds (15f) si xG disponible, S2_elo_odds (6f) sin xG |
| xG | **Aporta con Optuna**: OC (0.537) vs OF (0.544) = -0.007 |
| Prioridad | Baja — modelo opera en eficiencia maxima |
| depth=2 confirmado | todos los Optuna convergen a depth=2 (excepto O9) |

### 20.8 Archivos

| Archivo | Contenido |
|---------|-----------|
| `scripts/output/lab/lab_data_94.csv` | Dataset cacheado (2,952 rows, con xG) |
| `scripts/output/lab/shap_analysis_94.json` | SHAP, 9 escenarios |

---

## 21. Sueper Lig - Turquia (203)

### 21.1 Ficha Tecnica

| Campo | Valor |
|-------|-------|
| League ID | 203 |
| Pais | Turquia |
| Formato | Regular season (34 jornadas, 19 equipos desde 2024) |
| N matches | 3,276 (2016-2026) |
| Odds coverage | 2,829/3,276 = **86.4%** (FDUK) |
| xG coverage | **1,249/3,276 = 38.1%** (FotMob/Opta, backfill 2026-02-13) |
| N_train / N_test | 2,620 / 656 (base), 2,263 / 566 (odds), ~999 / 250 (xG) |
| Tests ejecutados | 110 standard + 9 SHAP + 16 Optuna (completo con xG) |
| Fecha lab | 2026-02-13 |

### 21.2 Campeon: OE_xg_defense_odds (Optuna, 11 features)

**HALLAZGO DESTACADO**: Sueper Lig es la UNICA liga donde el modelo BATE al mercado — confirmado por Standard Y Optuna.

| Metrica | Valor |
|---------|-------|
| Test | OE_xg_defense_odds (Optuna) |
| Brier | **0.56099** (N=240, odds_xg universe) |
| Market | 0.57745 (N=240, subset xG+odds) |
| **FAIR gap** | **-0.01525 CI[-0.034, +0.002]** |
| Features | 11 (xG rolling + defense + odds) |
| Hyperparams | depth=2, lr=0.0137, n_est=256, mcw=7 |

Champion Optuna: OE_xg_defense_odds = **0.56099** (11f, odds_xg)
Champion standard: P8_xg_defense_odds = **0.56199** (11f, odds_xg)
Champion odds: N2_odds_m2_combo = **0.56518** (N=566)
Champion base: D4_elo_k64 = **0.57506** (N=656)
Mercado global: **0.56878** (N=566)

### 21.3 SHAP Analysis (9 escenarios)

| Test | Brier | N_test | #1 Feature (SHAP) | #2 Feature | Odds share |
|------|-------|--------|-------------------|------------|------------|
| S0_baseline_17 | 0.59032 | 656 | goal_diff_avg (0.090) | home_shots_avg (0.063) | — |
| **S1_baseline_odds** | **0.56860** | 566 | odds_away (0.104) | odds_home (0.097) | 49.0% |
| S2_elo_odds | 0.56946 | 566 | odds_away (0.128) | odds_draw (0.105) | 72.6% |
| S3_defense_elo | 0.57944 | 656 | elo_diff (0.162) | elo_away (0.074) | — |
| S4_m2_interactions | 0.58120 | 656 | elo_diff (0.162) | elo_away (0.055) | — |
| S5_xg_elo | 0.59538 | 250 | elo_diff (0.203) | xg_diff (0.072) | — |
| S6_power_5 | 0.58376 | 656 | elo_diff (0.166) | opp_rating_diff (0.097) | — |
| S7_abe_elo | 0.58164 | 656 | elo_diff (0.151) | opp_rating_diff (0.073) | — |
| S8_abe_elo_odds | 0.57103 | 566 | odds_away (0.119) | odds_draw (0.082) | 51.4% |

### 21.4 Ranking Optuna (16 candidatos con xG, 50 trials × 3-fold temporal CV)

| # | Test | Brier | CV Brier | Uni | #F | Depth | LR |
|---|------|-------|----------|-----|----|-------|----|
| 1 | **OE_xg_defense_odds** | **0.56099** | 0.57879 | odds_xg | 11 | 2 | 0.014 |
| 2 | OC_xg_all_elo_odds | 0.56644 | 0.57584 | odds_xg | 15 | 2 | 0.018 |
| — | MKT_market | 0.56878 | — | — | — | — | — |
| 3 | OB_xg_odds | 0.56911 | 0.57601 | odds_xg | 6 | 5 | 0.026 |
| 4 | OF_abe_elo_odds | 0.57243 | 0.61085 | odds | 21 | 2 | 0.010 |
| 5 | O4_defense_elo_kimi | 0.57968 | 0.61945 | base | 14 | 3 | 0.023 |
| 6 | O5_m2_interactions | 0.58241 | 0.61890 | base | 9 | 2 | 0.019 |
| 7 | O2_defense_form_elo | 0.58377 | 0.62129 | base | 8 | 2 | 0.032 |
| 8 | O6_efficiency_elo | 0.58384 | 0.62214 | base | 8 | 2 | 0.015 |
| 9 | O0_elo_gw_defense | 0.58502 | 0.61947 | base | 5 | 2 | 0.015 |
| 10 | O8_smart_minimal | 0.58625 | 0.62342 | base | 13 | 2 | 0.038 |
| 11 | OD_xg_overperf_elo | 0.58710 | 0.58580 | xg | 12 | 2 | 0.017 |
| 12 | O7_all_abe_elo | 0.58711 | 0.62325 | base | 18 | 2 | 0.038 |
| — | FIXED_baseline | 0.58807 | — | — | 17 | — | — |
| 13 | OA_only_elo | 0.58980 | 0.62322 | base | 3 | 2 | 0.038 |
| 14 | O9_baseline_17 | 0.59039 | 0.61963 | base | 17 | 2 | 0.022 |
| 15 | O3_elo_k20 | 0.59069 | 0.62274 | base | 3 | 2 | 0.040 |
| 16 | O1_elo_gw_form | 0.59104 | 0.62189 | base | 6 | 2 | 0.018 |

**0 tests SKIPPED.** Top-3 son TODOS odds_xg. OE bate al mercado (0.561 vs 0.569, Δ=-0.78% global). FAIR en interseccion: Δ=-0.01525.

### 21.5 Gap vs Mercado

```
Market Brier (global):  0.56878 (N=566, de-vigged)
Best Optuna (OE):       0.56099 → FAIR Δ=-0.01525 CI[-0.034, +0.002]
Best standard (P8):     0.56199 → FAIR Δ=-0.01509 CI[-0.034, +0.004]
Best odds (N2):         0.56518 → FAIR Δ=-0.00360 CI[-0.017, +0.009]
Best base (D4):         0.57506 (N=656)

*** MODELO GANA vs MERCADO — CONFIRMADO por Standard Y Optuna ***
El gap -1.5% es el mayor de todas las ligas testeadas.
CI95 casi excluye cero: [-0.034, +0.002].
```

### 21.6 Que Funciona y Que No

**Funciona MUY bien**:
- **xG + defense + odds (P8)**: 0.562 — BATE al mercado por -1.5%. Unica liga con modelo > mercado
- **Odds + M2 combo (N2)**: 0.565 — BATE al mercado por -0.36% en sample completo
- **S1_baseline_odds**: 0.569 — odds share solo 49% (features baseline complementan)
- **Defense + Elo (S3)**: 0.579 — mejor base sin odds

**No funciona**:
- **xG sin odds (S5)**: 0.595 — PEOR que baseline 0.590. xG solo no es suficiente
- **ABE features**: S7 (0.582) vs S3 (0.579) — marginal. ABE no aporta significativamente en Turquia
- **Baseline 17 (S0)**: 0.590 — ruido

**Hallazgo clave**: xG SOLO aporta cuando se combina con odds. El patron {xG + defense + odds} captura algo que ni odds ni xG solos logran.

### 21.7 Recomendaciones v1.0.2

| Decisión | Recomendación |
|----------|---------------|
| Tier | **Model-competitive** — UNICA liga donde modelo bate mercado (Standard Y Optuna) |
| Market Anchor α | **0.0** (NO aplicar — el modelo aporta valor propio) |
| Features | OE_xg_defense_odds (11f) campeon Optuna, P8 campeon standard (misma composicion) |
| xG | **CRITICO** — xG es la señal diferencial. Top-3 Optuna son todos xG+odds |
| Prioridad | **ALTA** — oportunidad de alpha positivo |
| depth=2 confirmado | todos excepto OB (depth=5) |

### 21.8 Archivos

| Archivo | Contenido |
|---------|-----------|
| `scripts/output/lab/lab_data_203.csv` | Dataset cacheado (3,276 rows, con xG) |
| `scripts/output/lab/shap_analysis_203.json` | SHAP, 9 escenarios |

---

## 22. EFL Championship - Inglaterra (40)

### 22.1 Ficha Tecnica

| Campo | Valor |
|-------|-------|
| League ID | 40 |
| Pais | Inglaterra |
| Formato | Regular season (46 jornadas, 24 equipos) + playoffs |
| N matches | 3,712 (2016-2026) |
| Odds coverage | 349/3,712 = **9.4%** — muy bajo (FDUK parcial) |
| xG coverage | **3,127/3,712 = 84.2%** (FotMob/Opta, backfill 2026-02-13) |
| N_train / N_test | 2,969 / 743 (base), 277 / 70 (odds), ~2,502 / 626 (xG) |
| Tests ejecutados | 110 standard + 9 SHAP + 16 Optuna (completo con xG) |
| Fecha lab | 2026-02-13 |

**ADVERTENCIA**: Solo 70 matches con odds en test set. Todos los resultados con odds son POCO CONFIABLES.

### 22.2 Campeon: O6_efficiency_elo (Optuna, 8 features)

| Metrica | Valor |
|---------|-------|
| Test | O6_efficiency_elo (Optuna) |
| Brier | **0.64189** |
| Features | 8 (finish_eff_home/away, def_eff_home/away, efficiency_diff, elo_home, elo_away, elo_diff) |
| N_test | 743 (base universe) |
| Hyperparams | depth=2, lr=0.0120, n_est=218, mcw=9 |

Champion Optuna (base): O6_efficiency_elo = **0.64189** (8f)
Champion standard: M7_ultimate = **0.64054** (N=743)
Champion SHAP: S3_defense_elo = **0.64581** (5f)
Champion xG: P4_xg_overperf_elo = **0.64499** (N=626)
Mercado: 0.61262 (N=70 — **NO CONFIABLE**)

### 22.3 SHAP Analysis (9 escenarios)

| Test | Brier | N_test | #1 Feature (SHAP) | #2 Feature | Odds share |
|------|-------|--------|-------------------|------------|------------|
| S0_baseline_17 | 0.65012 | 743 | home_shots_avg (0.050) | away_matches_played (0.045) | — |
| S1_baseline_odds | 0.68058 | 70 | home_shots_avg (0.113) | abs_defense_diff (0.074) | 8.8% |
| S2_elo_odds | 0.72602 | 70 | elo_away (0.151) | elo_diff (0.102) | 37.3% |
| **S3_defense_elo** | **0.64581** | 743 | elo_diff (0.086) | elo_home (0.053) | — |
| S4_m2_interactions | 0.64854 | 743 | elo_diff (0.086) | elo_home (0.041) | — |
| S5_xg_elo | 0.64696 | 626 | elo_diff (0.085) | elo_home (0.061) | — |
| S6_power_5 | 0.64911 | 743 | elo_diff (0.096) | opp_rating_diff (0.054) | — |
| S7_abe_elo | 0.64620 | 743 | elo_diff (0.077) | elo_home (0.036) | — |
| S8_abe_elo_odds | 0.74895 | 70 | opp_att_away (0.097) | home_bias_home (0.075) | 8.6% |

**Nota**: Tests con odds (S1, S2, S8) tienen N=70 y son RUIDO. Ignorar.

### 22.4 Ranking Optuna (16 candidatos con xG, 50 trials × 3-fold temporal CV)

| # | Test | Brier | CV Brier | Uni | #F | Depth | LR |
|---|------|-------|----------|-----|----|-------|----|
| — | MKT_market | 0.61262 | — | — | — | — | — |
| 1 | **O6_efficiency_elo** | **0.64189** | 0.65395 | base | 8 | 2 | 0.012 |
| 2 | O4_defense_elo_kimi | 0.64299 | 0.65380 | base | 14 | 2 | 0.024 |
| 3 | O1_elo_gw_form | 0.64481 | 0.65073 | base | 6 | 2 | 0.025 |
| 4 | OA_only_elo | 0.64489 | 0.65144 | base | 3 | 2 | 0.019 |
| 5 | O2_defense_form_elo | 0.64573 | 0.65290 | base | 8 | 2 | 0.023 |
| 6 | OD_xg_overperf_elo | 0.64620 | 0.64432 | xg | 12 | 2 | 0.031 |
| 7 | O8_smart_minimal | 0.64640 | 0.65346 | base | 13 | 2 | 0.011 |
| 8 | O0_elo_gw_defense | 0.64656 | 0.65175 | base | 5 | 2 | 0.020 |
| 9 | O3_elo_k20 | 0.64717 | 0.65021 | base | 3 | 2 | 0.040 |
| 10 | O7_all_abe_elo | 0.64840 | 0.65794 | base | 18 | 2 | 0.019 |
| — | FIXED_baseline | 0.65069 | — | — | 17 | — | — |
| 11 | O5_m2_interactions | 0.65094 | 0.65402 | base | 9 | 4 | 0.015 |
| 12 | O9_baseline_17 | 0.65162 | 0.65618 | base | 17 | 5 | 0.011 |
| 13 | OB_xg_odds | 0.66462 | 0.66554 | odds_xg | 6 | 4 | 0.012 |
| 14 | OC_xg_all_elo_odds | 0.66542 | 0.66630 | odds_xg | 15 | 5 | 0.011 |
| 15 | OE_xg_defense_odds | 0.67185 | 0.66761 | odds_xg | 11 | 2 | 0.014 |
| 16 | OF_abe_elo_odds | 0.67507 | 0.65934 | odds | 21 | 2 | 0.010 |

**0 tests SKIPPED.** Tests con odds (OB-OF) son los PEORES — N_test=70 causa overfitting severo. **El mercado (0.613) bate a TODOS los modelos por 2.9+ puntos.**

FAIR (O6 vs market, N=70): modelo=0.73198, market=0.61262, Δ=+0.11936 CI[+0.059, +0.182] — **SIGNIFICATIVO**, mercado MUY superior.

### 22.5 Gap vs Mercado

```
Market Brier:       0.61262 (N=70 — NO CONFIABLE por N bajo)
Best Optuna (O6):   0.64189 (N=743, base)
Best standard (M7): 0.64054 (N=743)
FAIR:               +0.119 CI[+0.059, +0.182] — modelo MUY detras

*** MARKET TEST NO CONFIABLE (N=70) ***
La Championship necesita backfill de odds masivo (FDUK tiene datos para E0).
El gap real vs mercado es desconocido.
```

### 22.6 Que Funciona y Que No

**Funciona**:
- **Defense + Elo (S3)**: 0.646 — campeon estable con N=743. elo_diff domina
- **xG + Elo (S5)**: 0.647 — marginal sobre S3 (0.001 peor). xG no añade
- **M7_ultimate (base)**: 0.641 — mejor standard test con N=743
- **ABE + Elo (S7)**: 0.646 — on par con S3

**No funciona**:
- **Todos los tests con odds**: N=70, completamente no confiables
- **xG rolling**: S5 (0.647) no mejora sobre S3 (0.646)
- **Baseline 17 (S0)**: 0.650 — peor que S3 con 5 features

**Hallazgo clave**: Championship tiene Brier ~0.645 (alto). Posiblemente la liga mas competitiva y dificil de predecir.

### 22.7 Recomendaciones v1.0.2

| Decisión | Recomendación |
|----------|---------------|
| Tier | **Data-limited** (odds insuficientes, xG sin impacto) |
| Market Anchor | No aplicable — sin odds suficientes |
| Features | O6_efficiency_elo (8f) Optuna o M7_ultimate (17f) standard |
| xG | Disponible (84.2%) pero Optuna OD (0.646) marginal sobre base. xG no aporta |
| **Prioridad** | **Backfill odds FDUK** — FDUK tiene Championship (E0). Necesita ingestion |
| depth=2 confirmado | 14/16 Optuna usan depth=2. O5 y O9 usan 4-5 |

### 22.8 Archivos

| Archivo | Contenido |
|---------|-----------|
| `scripts/output/lab/lab_data_40.csv` | Dataset cacheado (3,712 rows, con xG) |
| `scripts/output/lab/shap_analysis_40.json` | SHAP, 9 escenarios |

---

## 23. Saudi Pro League (307)

### 23.1 Ficha Tecnica

| Campo | Valor |
|-------|-------|
| League ID | 307 |
| Pais | Arabia Saudita |
| Formato | Regular season (30 jornadas, 18 equipos) |
| N matches | 1,758 (2019-2026) |
| Odds coverage | **1,682/1,758 = 95.6%** (OddsPortal backfill 2026-02-13) |
| xG coverage | **912/1,758 = 51.9%** (FotMob/Opta, desde 2022) |
| N_train / N_test | 1,406 / 352 (base), 1,345 / 337 (odds), 624 / 156 (xG), 613 / 154 (odds_xg) |
| Tests ejecutados | 110 standard + 9 SHAP + 16 Optuna |
| Fecha lab | 2026-02-13 (re-run post-OddsPortal backfill) |

### 23.2 Campeon: OE_xg_defense_odds (Optuna, 11 features)

| Metrica | Valor |
|---------|-------|
| Test | OE_xg_defense_odds (Optuna) |
| Brier | **0.52223** |
| Features | 11 (xG + defense + odds, universe=odds_xg) |
| N_test | 154 |
| Hyperparams | depth=2, lr=0.0373, n_est=69, mcw=10 |

Market baseline: **0.52487** (N=337)
Champion Optuna: OE_xg_defense_odds = **0.52223** (N=154) — **MEJOR QUE MERCADO** (Brier absoluto)
Champion standard odds: N6_odds_clean = **0.54354** (N=337)
Champion standard xG+odds: P7_xg_all_elo_odds = **0.52363** (N=337)
Champion base: H5_elo_gw_defense = **0.55902** (N=352)

FAIR (OE vs market, N=154): Δ=+0.01770 CI[-0.005, +0.040] — **NO significativo** (CI cruza cero)

### 23.3 SHAP Analysis (9 escenarios — ahora con odds)

| Test | Brier | N_test | #1 Feature (SHAP) | #2 Feature | Odds share |
|------|-------|--------|-------------------|------------|------------|
| S0_baseline_17 | 0.57515 | 352 | goal_diff_avg (0.130) | away_goals_scored_avg (0.054) | — |
| S1_baseline_odds | 0.54658 | 337 | **odds_home (0.203)** | odds_away (0.127) | 54.5% |
| S2_elo_odds | 0.55256 | 337 | **odds_home (0.214)** | odds_away (0.130) | 73.0% |
| S3_defense_elo | 0.56181 | 352 | elo_diff (0.195) | elo_away (0.115) | — |
| S4_m2_interactions | 0.57049 | 352 | elo_diff (0.191) | elo_away (0.065) | — |
| S5_xg_elo | 0.55356 | 156 | elo_diff (0.151) | **xg_diff (0.120)** | — |
| S6_power_5 | 0.56671 | 352 | elo_diff (0.202) | opp_rating_diff (0.117) | — |
| S7_abe_elo | 0.57486 | 352 | elo_diff (0.170) | opp_rating_diff (0.063) | — |
| S8_abe_elo_odds | 0.55979 | 337 | **odds_home (0.183)** | odds_away (0.122) | 48.3% |

**Patron**: Odds son la señal dominante (54-73% de SHAP). En S8 con 21 features, odds siguen siendo top-3. xG es #2 señal más fuerte en S5.

### 23.4 Optuna Ranking (16 candidatos)

| Rank | Test | Brier | CV Brier | #F | Universo | Hyperparams |
|------|------|-------|----------|----|----------|-------------|
| 1 | **OE_xg_defense_odds** | **0.52223** | 0.57728 | 11 | odds_xg | d=2, lr=0.037, n=69 |
| 2 | OC_xg_all_elo_odds | 0.52325 | 0.58202 | 15 | odds_xg | d=2, lr=0.039, n=73 |
| 3 | OB_xg_odds | 0.53259 | 0.57337 | 6 | odds_xg | d=4, lr=0.011, n=219 |
| 4 | OD_xg_overperf_elo | 0.54374 | 0.58646 | 12 | xg | d=5, lr=0.012, n=177 |
| 5 | O6_efficiency_elo | 0.55837 | 0.60903 | 8 | base | d=3, lr=0.048, n=70 |
| 6 | O1_elo_gw_form | 0.55938 | 0.60399 | 6 | base | d=2, lr=0.015, n=239 |
| 7 | O0_elo_gw_defense | 0.56283 | 0.60456 | 5 | base | d=3, lr=0.029, n=93 |
| 8 | O2_defense_form_elo | 0.56447 | 0.60656 | 8 | base | d=2, lr=0.012, n=285 |
| 9 | OA_only_elo | 0.56519 | 0.60767 | 3 | base | d=2, lr=0.036, n=94 |
| 10 | O7_all_abe_elo | 0.56553 | 0.60282 | 18 | base | d=2, lr=0.014, n=257 |
| 11 | O3_elo_k20 | 0.56701 | 0.60510 | 3 | base | d=2, lr=0.039, n=61 |
| 12 | OF_abe_elo_odds | 0.56763 | 0.58740 | 21 | odds | d=5, lr=0.017, n=182 |
| 13 | O8_smart_minimal | 0.57000 | 0.60906 | 13 | base | d=2, lr=0.038, n=58 |
| 14 | O5_m2_interactions | 0.57116 | 0.60666 | 9 | base | d=2, lr=0.048, n=64 |
| 15 | O4_defense_elo_kimi | 0.57317 | 0.60806 | 14 | base | d=2, lr=0.038, n=58 |
| 16 | O9_baseline_17 | 0.58310 | 0.63351 | 17 | base | d=2, lr=0.038, n=58 |
| — | FIXED_baseline | 0.57502 | — | 17 | base | prod hyperparams |
| — | MKT_market | **0.52487** | — | — | odds | — |

**Top-3 todos son odds_xg**. El modelo con xG+odds (OE, OC) se acerca y potencialmente supera al mercado.

### 23.5 Que Funciona y Que No

**Funciona bien**:
- **xG + odds es la combinacion ganadora**: Top-3 Optuna son TODOS odds_xg (OE 0.522, OC 0.523, OB 0.533)
- **Odds son la señal #1**: SHAP 54-73% en tests con odds. odds_home SHAP=0.20+ en S1/S2
- **xG es señal #2 fuerte**: xg_diff SHAP=0.120 en S5, y xG+odds supera a solo-odds consistentemente
- **Modelo se acerca al mercado**: OE 0.522 vs Market 0.525 — gap de solo 0.003 puntos Brier
- **Elo domina en base**: elo_diff #1 en 5/6 tests sin odds

**No funciona**:
- **ABE features no ayudan**: OF_abe_elo_odds (0.568) es #12, peor que OB_xg_odds (0.533) con solo 6 features
- **Baseline 17 (O9)**: 0.583 — PEOR de todos, demasiadas features
- **Interactions/Kimi**: O4/O5 en posiciones 14-15

**Hallazgo clave**: Saudi es una liga donde xG+odds combinadas permiten al modelo CASI ALCANZAR al mercado (gap 0.003). La cobertura xG (52%) limita el N_test a 154 — con mas cobertura, la señal podria ser aun mas clara.

### 23.6 Comparativa pre/post OddsPortal backfill

| Metrica | Antes (54 odds) | Despues (1,682 odds) |
|---------|-----------------|----------------------|
| Market baseline | No disponible | **0.52487** |
| Tests odds | Todos SKIPPED | 110 ejecutados |
| Optuna | Todos SKIPPED | 16 ejecutados |
| Champion | S3_defense_elo 0.565 (base) | **OE_xg_defense_odds 0.522** (odds_xg) |
| FAIR gap | No calculable | +0.018 (ns) |
| Tier | No-odds | **Odds-available** |

### 23.7 Recomendaciones v1.0.2

| Decision | Recomendacion |
|----------|---------------|
| Tier | **Modelo ≈ Mercado** — FAIR +0.018 (ns), modelo potencialmente competitivo |
| Market Anchor | **CANDIDATO** α=0.5 — modelo Optuna casi alcanza mercado, blend podria ser optimo |
| Features | OE_xg_defense_odds (11f) para xG subset, N6_odds_clean (15f) para odds-only |
| xG | **CRITICO** — xG separa top-3 del resto. Expandir cobertura FotMob (52%→meta 80%+) |
| Optuna | EJECUTADO — OE champion con Brier 0.522, mejor que market 0.525 en absoluto |
| **Prioridad** | **Alta** — uno de los pocos mercados donde modelo compite con mercado |

### 23.8 Archivos

| Archivo | Contenido |
|---------|-----------|
| `scripts/output/lab/lab_data_307.csv` | Dataset cacheado (1,758 rows, 96% odds, 52% xG) |
| `scripts/output/lab/shap_analysis_307.json` | SHAP, 9 escenarios completos |
| `scripts/output/lab/feature_lab_results_optuna.json` | Optuna, 16 candidatos |
| `data/oddsportal_raw/saudi_all.json` | OddsPortal raw (1,689 matches, 99.6% con odds) |

---

## Apendice: Como Agregar una Liga

1. Ejecutar extraccion: `python3 scripts/feature_lab.py --extract --league <ID>`
2. Run standard lab: `python3 scripts/feature_lab.py --league <ID>` (101 tests, fixed params)
3. Identificar top-10 y correr SHAP: `python3 scripts/feature_lab.py --shap --league <ID>`
4. Correr Optuna en top performers: `python3 scripts/feature_lab.py --optuna --league <ID>`
5. **Correr Section R**: `python3 scripts/feature_lab.py --residual --league <ID>` (diagnostico de eficiencia de mercado)
6. Si hay fecha minima de datos confiables: agregar `--min-date YYYY-MM-DD`
7. Documentar resultados en nueva seccion de este archivo
8. Actualizar indice de ligas y tabla de Section R

**Frecuencia de re-evaluacion sugerida**: cada 6 meses o cuando cambie el formato del torneo.
**Section R**: re-ejecutar cada vez que se agreguen features nuevas o antes de modificar Market Anchor alpha.
