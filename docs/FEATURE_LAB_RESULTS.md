# Feature Lab — Resultados Completos

**Fecha**: 2026-02-09 (actualizado 2026-02-10: FDUK backfill + SHAP 2023+)
**Autor**: David (dirección) + Master (ejecución)
**Scripts**: `scripts/feature_lab.py`, `scripts/feature_diagnostic.py`, `scripts/xg_signal_test.py`

---

## 1. Resumen Ejecutivo

El Feature Lab probó **101 combinaciones de features** + **16 candidatos Optuna** en **7 ligas** (3 LATAM + 4 EUR) con ~22,000 partidos totales. Los hallazgos principales:

1. **Elo-goals es la feature más importante del sistema** — aparece en el top-3 de todas las ligas
2. **El mercado (odds de-vigorizadas) es imbatible en Europa** — supera al mejor modelo por 1.7-3.6%
3. **Odds LATAM ahora disponibles (FDUK backfill)** — Argentina 100% desde 2023, odds mejoran el modelo cuando la cobertura es completa
4. **Optuna per-liga mejora 1.1-4.2%** vs hyperparams fijos de producción
5. **Las 17 features originales (v1.0.1) son subóptimas** — modelos con 3-9 features seleccionadas ganan consistentemente
6. **`rest_days` es ruido universal** — negativa en 12/21 ligas
7. **Gap EUR vs LATAM**: Brier 0.61 vs 0.66 (Argentina)
8. **xG (FotMob) es NEUTRAL como señal** — no mejora ni empeora el baseline de forma significativa

---

## 2. Metodología

### 2.1 Configuración del Lab

| Parámetro | Valor |
|-----------|-------|
| Split temporal | 80/20 cronológico (nunca random) |
| Seeds | 5 (multi-seed evaluation) |
| Bootstrap CI95 | 1,000 iteraciones (sobre último seed; brier_mean es promedio de N_SEEDS) |
| DRAW_WEIGHT | 1.5 (sample weighting para empates) |
| Rolling window | 10 partidos |
| Time decay λ | 0.01 |
| Modelo | XGBoost `multi:softprob` (3 clases: H/D/A) |
| Métrica principal | Multiclass Brier Score (lower = better) |

### 2.2 Hyperparams de Producción (v1.0.1)

```
max_depth=3, learning_rate=0.0283, n_estimators=114,
min_child_weight=7, subsample=0.72, colsample_bytree=0.71,
reg_alpha=2.8e-05, reg_lambda=0.000904
```

### 2.3 Optuna (Wave 10)

| Parámetro | Valor |
|-----------|-------|
| Trials | 50 |
| CV Folds | 3 (forward-chaining temporal) |
| Search space | depth[2-6], lr[0.01-0.15], n_est[50-300], mcw[3-15], subsample[0.5-0.9], colsample[0.4-1.0], reg_alpha/lambda[1e-6, 1.0] |

### 2.4 Elo-Goals Rating System

Sistema Elo secuencial PIT-safe calculado en el lab:
- K=32, home_advantage=100, initial=1500
- Goal-weighted: margen de goles como factor multiplicador
- Variantes: K=20, K=40, split (home/away separados), momentum (last 5)

### 2.5 Ligas Evaluadas

| Liga | ID | N matches | Fecha rango | Odds disponibles |
|------|----|-----------|-------------|-----------------|
| Argentina | 128 | 2,526 | Oct 2020 — Feb 2026 | SI — 100% desde 2023 (FDUK backfill) |
| Liga MX | 262 | 1,905 | Jul 2020 — Feb 2026 | SI — 46% global, ~100% desde 2023 (FDUK) |
| Primeira Liga | 94 | 3,043 | Ago 2015 — Feb 2026 | SI — 99.0% (FDUK) |
| Serie A | 135 | 4,040 | Ago 2015 — Feb 2026 | SI — 99.9% (FDUK) |
| Premier League | 39 | 4,050 | Ago 2015 — Feb 2026 | SI — 100% (FDUK) |
| La Liga | 140 | 4,028 | Ago 2015 — Feb 2026 | SI — 100% (FDUK) |
| Bundesliga | 78 | 3,261 | Ago 2015 — Feb 2026 | SI — 99.4% (FDUK) |

---

## 3. Catálogo de Features Probadas

### 3.1 Sections A-I (Waves 1-6: Baseline, Elo, Form, Matchup, Surprise, Calendar)

| Section | Tests | Descripción |
|---------|-------|-------------|
| A | A0-A2 | Baseline 17, No rest_days (14), Baseline + Elo (20) |
| B | B0-B2 | Only Elo (3), Baseline + Odds (20), Elo + Odds (6) |
| C | C0-C4 | Elo variants: goal-weighted, K=20, K=40, split, momentum |
| D | D0-D9 | Elo combos: defense+elo, form+elo, all_gw, etc. |
| E | E0-E4 | Form: streaks, volatility, clean sheets, form+elo |
| F | F0-F3 | Matchup: confrontation, H2H, elo_delta, etc. |
| G | G0-G3 | Surprise & meta: upset rate, surprise features |
| H | H0-H2 | Calendar: day of week, month, season progress |

### 3.2 Sections I-K (Waves 7-8: ABE)

| Section | Tests | Descripción |
|---------|-------|-------------|
| I | I0-I4 | Opponent-adjusted (opp_att, opp_def, opp_rating_diff) |
| K | K0-K5 | Overperf, draw-aware, home-bias, combos ABE |

### 3.3 Section L-M (Wave 9: Kimi)

| Section | Tests | Descripción |
|---------|-------|-------------|
| L | L0-L5 | Interactions (elo_x_rest, elo_x_season, form_x_defense) + Efficiency (finish_eff, def_eff) |
| M | M0-M4 | Argentina signal features (top diagnostic) + interaction combos |

### 3.4 Section N (Wave Odds)

| Test | Features | Descripción |
|------|----------|-------------|
| N0_odds_elo | 6 | odds_home/draw/away + elo_home/away/diff |
| N1_odds_defense_elo | 8 | N0 + defense pair |
| N2_odds_m2_combo | 12 | odds + Argentina champion features |
| N3_odds_efficiency | 11 | odds + efficiency + elo |
| N4_odds_abe_best | 14 | odds + opp_adj + overperf + elo |
| N5_odds_kimi_all | 15 | odds + interactions + efficiency + elo |
| N6_odds_clean | 15 | odds + defense + form + elo + overperf |
| N7_odds_power7 | 10 | odds + defense + elo_gw |
| N8_odds_minimal | 5 | odds (3) + defense pair (2) |
| N9_odds_ultimate | 22 | kitchen sink: todo con odds |

### 3.5 Section O (Wave Optuna)

| Test | Features | Origen |
|------|----------|--------|
| O0_elo_gw_defense | 5 | defense + elo_gw (top ITA, GER, ESP) |
| O1_elo_gw_form | 6 | form + elo_gw (top La Liga) |
| O2_defense_form_elo | 8 | defense + form + elo (top Premier League) |
| O3_elo_k20 | 3 | solo elo K=20 (top ITA, ESP) |
| O4_defense_elo_kimi | 14 | defense + elo + interactions + efficiency (top ENG) |
| O5_m2_interactions | 9 | Argentina signal + elo + interactions (top Argentina) |
| O6_efficiency_elo | 8 | efficiency + elo (top Liga MX) |
| O7_all_abe_elo | 18 | opp_adj + overperf + draw_aware + home_bias + elo |
| O8_smart_minimal | 13 | defense + elo + opp_adj + overperf |
| O9_baseline_17 | 17 | baseline original con Optuna tuning |
| OA_only_elo | 3 | solo elo estándar |

---

## 4. Resultados: Lab Estándar (Hyperparams Fijos de Producción)

### 4.1 Ligas Europeas — Top-5 por Liga

#### Serie A (N=4,040)

| Rank | Test | Brier | CI95 | Features |
|------|------|-------|------|----------|
| MKT | Market (de-vigged) | **0.57040** | [0.5499, 0.5937] | — |
| 1 | N4_odds_abe_best | 0.57276 | [0.5548, 0.5938] | 14 |
| 2 | J1_elo_odds / N0_odds_elo | 0.57318 | [0.5548, 0.5937] | 6 |
| 3 | J0_only_odds | 0.57406 | [0.5556, 0.5938] | 3 |
| 4 | N3_odds_efficiency | 0.57437 | [0.5558, 0.5942] | 11 |
| 5 | N8_odds_minimal | 0.57478 | [0.5568, 0.5963] | 5 |
| ref | A0_baseline_17 | 0.61720 | — | 17 |
| | **Best vs baseline** | **-0.04444 (-7.2%)** | | |

#### Premier League (N=4,050)

| Rank | Test | Brier | CI95 | Features |
|------|------|-------|------|----------|
| MKT | Market (de-vigged) | **0.56306** | [0.5382, 0.5865] | — |
| 1 | N3_odds_efficiency | 0.57280 | [0.5506, 0.5940] | 11 |
| 2 | N1_odds_defense_elo | 0.57300 | [0.5515, 0.5936] | 8 |
| 3 | N5_odds_kimi_all | 0.57402 | [0.5529, 0.5962] | 15 |
| 4 | N6_odds_clean | 0.57438 | [0.5543, 0.5950] | 15 |
| 5 | N9_odds_ultimate | 0.57445 | [0.5536, 0.5967] | 22 |
| ref | A0_baseline_17 | 0.60733 | — | 17 |
| | **Best vs baseline** | **-0.03453 (-5.7%)** | | |

#### La Liga (N=4,028)

| Rank | Test | Brier | CI95 | Features |
|------|------|-------|------|----------|
| MKT | Market (de-vigged) | **0.56624** | [0.5434, 0.5896] | — |
| 1 | N6_odds_clean | 0.57601 | [0.5576, 0.5955] | 15 |
| 2 | N2_odds_m2_combo | 0.57869 | [0.5594, 0.5981] | 12 |
| 3 | N8_odds_minimal | 0.57972 | [0.5610, 0.5999] | 5 |
| 4 | N4_odds_abe_best | 0.57987 | [0.5596, 0.5984] | 14 |
| 5 | J0_only_odds | 0.58023 | [0.5610, 0.6014] | 3 |
| ref | A0_baseline_17 | 0.61602 | — | 17 |
| | **Best vs baseline** | **-0.04001 (-6.5%)** | | |

#### Bundesliga (N=3,261)

| Rank | Test | Brier | CI95 | Features |
|------|------|-------|------|----------|
| MKT | Market (de-vigged) | **0.57571** | [0.5492, 0.6035] | — |
| 1 | N8_odds_minimal | 0.59166 | [0.5726, 0.6131] | 5 |
| 2 | J0_only_odds | 0.59180 | [0.5699, 0.6119] | 3 |
| 3 | N3_odds_efficiency | 0.59207 | [0.5712, 0.6128] | 11 |
| 4 | N6_odds_clean | 0.59213 | [0.5700, 0.6115] | 15 |
| 5 | N5_odds_kimi_all | 0.59224 | [0.5720, 0.6121] | 15 |
| ref | A0_baseline_17 | 0.61941 | — | 17 |
| | **Best vs baseline** | **-0.02775 (-4.5%)** | | |

### 4.2 Hallazgo clave EUR: Market imbatible

En las 4 ligas europeas, el market (de-vigged odds convertidas a probabilidades) supera a **todos** los 101 modelos XGBoost:

| Liga | Market | Mejor modelo | Gap |
|------|--------|--------------|-----|
| Serie A | 0.57040 | 0.57276 | +0.4% |
| Premier League | 0.56306 | 0.57280 | +1.7% |
| La Liga | 0.56624 | 0.57601 | +1.7% |
| Bundesliga | 0.57571 | 0.59166 | +2.8% |

**Conclusión**: No es posible superar al mercado con XGBoost y features rolling. El mercado incorpora información que el modelo no tiene (lesiones, mercado de fichajes, rumores, lineups probables, etc.).

---

## 5. Resultados: Optuna (Hyperparams Per-Liga, 50 Trials)

### 5.1 Argentina (N=2,523, sin odds)

| Rank | Test | Brier | CV Brier | Δ vs FIXED | Features |
|------|------|-------|----------|------------|----------|
| **1** | **O5_m2_interactions** | **0.64972** | 0.65217 | **-0.01378** | 9 |
| 2 | O0_elo_gw_defense | 0.65610 | 0.65461 | -0.00740 | 5 |
| 3 | O1_elo_gw_form | 0.65694 | 0.65398 | -0.00656 | 6 |
| 4 | O7_all_abe_elo | 0.65696 | 0.65095 | -0.00654 | 18 |
| 5 | O4_defense_elo_kimi | 0.65727 | 0.65348 | -0.00623 | 14 |
| ctrl | FIXED_baseline | 0.66350 | — | — | 17 |

**Champion features** (O5_m2_interactions, 9 features):
```
home_matches_played, home_goals_conceded_avg,
elo_home, elo_away, elo_diff,
elo_x_rest, elo_x_season, elo_x_defense, form_x_defense
```

**Optuna params**: depth=2, lr=0.038, n_est=58, mcw=14

### 5.2 Liga MX (N=1,905, sin odds)

| Rank | Test | Brier | CV Brier | Δ vs FIXED | Features |
|------|------|-------|----------|------------|----------|
| **1** | **O6_efficiency_elo** | **0.60041** | 0.64005 | **-0.01144** | 8 |
| 2 | O4_defense_elo_kimi | 0.60069 | 0.63958 | -0.01116 | 14 |
| 3 | O2_defense_form_elo | 0.60329 | 0.63696 | -0.00856 | 8 |
| 4 | O0_elo_gw_defense | 0.60483 | 0.63889 | -0.00702 | 5 |
| 5 | O1_elo_gw_form | 0.60523 | 0.63853 | -0.00662 | 6 |
| ctrl | FIXED_baseline | 0.61185 | — | — | 17 |

**Champion features** (O6_efficiency_elo, 8 features):
```
finish_eff_home, finish_eff_away, def_eff_home, def_eff_away,
efficiency_diff, elo_home, elo_away, elo_diff
```

**Optuna params**: depth=2, lr=0.038, n_est=58, mcw=14

### 5.3 Serie A (N=4,040, control EUR)

| Rank | Test | Brier | CV Brier | Δ vs FIXED | Features |
|------|------|-------|----------|------------|----------|
| **1** | **O0_elo_gw_defense** | **0.59125** | 0.58135 | **-0.02568** | 5 |
| 2 | O1_elo_gw_form | 0.59171 | 0.58062 | -0.02522 | 6 |
| 3 | O3_elo_k20 | 0.59197 | 0.58147 | -0.02496 | 3 |
| 4 | O7_all_abe_elo | 0.59238 | 0.58480 | -0.02455 | 18 |
| 5 | O2_defense_form_elo | 0.59375 | 0.58486 | -0.02318 | 8 |
| ctrl | FIXED_baseline | 0.61693 | — | — | 17 |
| MKT | Market | 0.57040 | — | — | — |

**Champion features** (O0_elo_gw_defense, 5 features):
```
home_goals_conceded_avg, away_goals_conceded_avg,
elo_gw_home, elo_gw_away, elo_gw_diff
```

**Optuna params**: depth=2, lr=0.016, n_est=173, mcw=14

### 5.4 Resumen Optuna: Delta vs FIXED

| Liga | FIXED baseline | Best Optuna | Δ | % mejora |
|------|---------------|-------------|---|----------|
| **Argentina** | 0.66350 | **0.64972** | -0.01378 | **-2.1%** |
| **Liga MX** | 0.61185 | **0.60041** | -0.01144 | **-1.9%** |
| **Serie A** | 0.61693 | **0.59125** | -0.02568 | **-4.2%** |

### 5.5 Hyperparams Convergentes (Optuna)

Optuna consistentemente converge a:

| Param | Producción (v1.0.1) | Optuna óptimo | Dirección |
|-------|--------------------:|:--------------|-----------|
| max_depth | 3 | **2** | Más simple |
| learning_rate | 0.0283 | **0.01-0.05** | Similar o más bajo |
| n_estimators | 114 | **50-173** | Variable |
| min_child_weight | 7 | **8-15** | Más regularización |
| subsample | 0.72 | **0.50-0.68** | Más regularización |
| colsample_bytree | 0.71 | **0.50-0.80** | Similar |

**Conclusión**: El modelo de producción está ligeramente overfitteado. `max_depth=2` y `min_child_weight` más alto son universalmente mejores.

---

## 6. Feature Diagnostic (Argentina, v1.0.1)

Script: `scripts/feature_diagnostic.py` | Output: `scripts/output/feature_diagnostic_argentina_128.json`

Evaluación de las 17 features de producción para Argentina usando permutation importance + ablation + canary features.

### 6.1 Resultados por Feature

| Feature | Perm Mean | Ablation Δ | Verdict |
|---------|-----------|------------|---------|
| home_matches_played | +0.00268 | +0.00738 | **SIGNAL** |
| home_goals_conceded_avg | +0.00174 | +0.00222 | **SIGNAL** |
| away_corners_avg | +0.00094 | +0.00248 | NEUTRAL |
| away_goals_scored_avg | +0.00065 | -0.00058 | NEUTRAL |
| goal_diff_avg | +0.00061 | -0.00003 | NEUTRAL |
| abs_defense_diff | +0.00029 | +0.00088 | NEUTRAL |
| away_goals_conceded_avg | +0.00024 | +0.00075 | NEUTRAL |
| home_goals_scored_avg | +0.00010 | +0.00187 | NEUTRAL |
| home_rest_days | +0.00001 | +0.00066 | NEUTRAL |
| home_corners_avg | -0.00000 | +0.00030 | NEUTRAL |
| rest_diff | -0.00018 | -0.00020 | NEUTRAL |
| away_shots_avg | -0.00019 | +0.00139 | NEUTRAL |
| abs_strength_gap | -0.00046 | +0.00020 | NEUTRAL |
| away_rest_days | -0.00059 | -0.00040 | NEUTRAL |
| away_matches_played | -0.00092 | -0.00150 | **NOISE** |
| abs_attack_diff | -0.00094 | -0.00046 | NEUTRAL |
| home_shots_avg | -0.00209 | -0.00215 | **NOISE** |

### 6.2 Conclusiones del Diagnostic

- **Solo 2 features con señal clara** en Argentina: `home_matches_played` y `home_goals_conceded_avg`
- **2 features NOISE**: `away_matches_played` y `home_shots_avg` (empeoran el modelo)
- **13 features NEUTRAL**: no ayudan ni perjudican — ruido que Optuna regulariza mejor
- Canaries: max perm = 0.00137 (bajo, validando que el test funciona)

### 6.3 Diagnostic Global (25 ligas, 54,667 matches)

| Hallazgo | Detalle |
|----------|---------|
| Features SIGNAL | 14 de 17 (globalmente) |
| Features NOISE | 0 (globalmente) |
| Features NEUTRAL | 3 (rest_days group) |
| Feature dominante | `goal_diff_avg` (6x #2 en importancia) |
| rest_days | Negativa en 12/21 ligas → universal noise |
| Gap EUR vs LATAM | Brier 0.6108 vs 0.6365 (+0.026 peor LATAM) |
| Mejor liga | Primeira Liga (0.58) |
| Peor liga | Paraguay (0.66), Argentina (0.65) |

---

## 7. xG Signal Test

Script: `scripts/xg_signal_test.py` | Source: FotMob (LATAM) + Understat (EUR)

A/B test: baseline 14 features vs baseline+4 xG features (rolling windows w3/w5/w10).

### 7.1 Resultados por Liga

| Liga | Source | N | Window | ΔBrier | CI95 | Verdict |
|------|--------|---|--------|--------|------|---------|
| Argentina | FotMob | 1,160 | w5 | -0.00174 | [-0.011, +0.009] | NEUTRAL |
| Argentina | FotMob | 1,160 | w10 | +0.00174 | [-0.004, +0.016] | NEUTRAL |
| Colombia | FotMob | 270 | w5 | +0.00543 | [-0.028, +0.028] | NOISE |
| Colombia | FotMob | 270 | w10 | -0.01037 | [-0.046, +0.004] | NEUTRAL |
| La Liga | Understat | 403 | w5 | -0.00970 | [-0.032, +0.022] | NEUTRAL |
| EPL | Understat | 366 | w5 | +0.01244 | [-0.015, +0.070] | NOISE |

### 7.2 No-Shots Redundancy Test

| Liga | Window | ΔBrier | CI95 | Verdict |
|------|--------|--------|------|---------|
| Argentina | w5 | +0.00206 | [-0.005, +0.015] | NEUTRAL |
| Colombia | w5 | +0.00593 | [-0.023, +0.051] | NOISE |
| La Liga | w5 | -0.00988 | [-0.045, +0.008] | NEUTRAL |
| EPL | w5 | -0.00047 | [-0.055, +0.038] | NEUTRAL |

### 7.3 Veredicto Global

**xG es NEUTRAL** en todas las ligas probadas. No hay señal estadísticamente significativa (CI95 cruza cero en todos los casos). Posibles razones:
1. Las features rolling (goals_scored_avg, shots_avg) ya capturan la señal que xG aporta
2. El rolling window sobre xG diluye la señal (xG de un partido es más informativo que el promedio)
3. N insuficiente para Colombia/EPL/LaLiga (300-400 test)

---

## 8. Odds LATAM: FDUK Backfill (resuelto 2026-02-09)

### 8.1 Problema Original

API-Football purga odds ~7-10 días después de FT. Solo capturábamos odds pre-KO para partidos futuros desde `odds_sync` (27-Ene-2026), resultando en <2% cobertura LATAM.

### 8.2 Solución: Football-Data UK Backfill

Se implementó ingesta de odds históricas desde Football-Data UK (`scripts/ingest_football_data_uk.py`):
- **Fuente**: CSVs públicos de football-data.co.uk (Pinnacle closing odds para LATAM, B365/PS para EUR)
- **Aliases**: `data/fduk_team_aliases.json` v4.2.0 (619 aliases, 16 ligas, smoke-tested)
- **Bug crítico corregido**: aliases LATAM tenían `external_id` (API-Football) en lugar de `internal_id` (DB). El mapping fallaba silenciosamente. Fix: detección inteligente ext→int con verificación por liga.

### 8.3 Cobertura Post-Backfill

| Liga | ID | FT con odds | Total FT | Cobertura | Rango con odds |
|------|----|-------------|----------|-----------|----------------|
| EPL | 39 | 2,331 | 2,331 | **100%** | 2020+ |
| LaLiga | 140 | 2,327 | 2,328 | **100%** | 2020+ |
| SerieA | 135 | 2,348 | 2,351 | **99.9%** | 2020+ |
| Ligue1 | 61 | 2,033 | 2,038 | **99.8%** | 2020+ |
| Bundesliga | 78 | 1,871 | 1,882 | **99.4%** | 2020+ |
| Belgium | 144 | 192 | 192 | **100%** | 2020+ |
| Primeira | 94 | 1,889 | 1,908 | **99.0%** | 2020+ |
| Eredivisie | 88 | 1,795 | 1,859 | **96.6%** | 2020+ |
| SuperLig | 203 | 2,101 | 2,205 | **95.3%** | 2020+ |
| Championship | 40 | 338 | 370 | **91.4%** | 2020+ |
| MLS | 253 | 1,561 | 2,830 | 55.2% | 2023+ |
| **Argentina** | **128** | **1,301** | **2,526** | **51.5% global** | **2023+: 100%** |
| Brasil | 71 | 1,159 | 2,299 | 50.4% | 2023+ |
| Liga MX | 262 | 879 | 1,905 | 46.1% | 2023+ |

**Nota**: LATAM muestra ~50% global porque FDUK no cubre antes de 2023. Para el rango de training (2023+, `TRAINING_MIN_DATE`), Argentina tiene **100% cobertura**.

### 8.4 SHAP con Odds Completas (Argentina 2023+, N=1,319)

Con datos filtrados a 2023+ (100% odds coverage), los resultados cambiaron significativamente vs el run original (2020+, 46% odds):

| Test | Antes (46%) | 2023+ (100%) | Δ | Hallazgo |
|------|-------------|--------------|---|----------|
| S0_baseline_17 | 0.6621 | 0.6722 | +0.010 | Baseline |
| S1_baseline_odds | 0.6770 | **0.6696** | **-0.007** | Odds MEJORAN (antes dañaban) |
| S2_elo_odds | 0.7055 | **0.6814** | **-0.024** | Mejora masiva |
| S7_abe_elo | 0.6554 | **0.6585** | +0.003 | Campeón estable |

**Hallazgo clave**: La conclusión anterior "odds dañan Argentina" era un **artefacto de cobertura parcial** (46%). Con 100%, odds son una feature útil que mejora el baseline.

**Campeón SHAP 2023+**: S7_abe_elo (0.6585) — `league_draw_rate` domina (SHAP 0.098), seguida de `elo_diff` (0.052).

**SHAP top features por test (2023+)**:
| Test | #1 Feature (SHAP) | #2 Feature |
|------|-------------------|------------|
| S0_baseline_17 | home_matches_played (0.066) | abs_strength_gap (0.051) |
| S1_baseline_odds | home_matches_played (0.077) | odds_home (0.062) |
| S7_abe_elo | league_draw_rate (0.098) | elo_diff (0.052) |

---

## 9. Campeones por Liga (Feature Sets Óptimos)

### 9.1 Sin Odds (LATAM)

| Liga | Champion | Brier | Features | Hyperparams |
|------|----------|-------|----------|-------------|
| **Argentina** | O5_m2_interactions | **0.64972** | 9: matches_played, defense, elo, interactions | depth=2, lr=0.038, n_est=58, mcw=14 |
| **Liga MX** | O6_efficiency_elo | **0.60041** | 8: efficiency (4), elo (3), efficiency_diff | depth=2, lr=0.038, n_est=58, mcw=14 |

### 9.2 Con Odds (EUR)

| Liga | Champion (con odds) | Brier | Champion (sin odds, Optuna) | Brier | Market |
|------|---------------------|-------|-----------------------------|-------|--------|
| **Serie A** | N4_odds_abe_best | 0.57276 | O0_elo_gw_defense | 0.59125 | **0.57040** |
| **Premier League** | N3_odds_efficiency | 0.57280 | O2_defense_form_elo | — | **0.56306** |
| **La Liga** | N6_odds_clean | 0.57601 | O1_elo_gw_form | — | **0.56624** |
| **Bundesliga** | N8_odds_minimal | 0.59166 | — | — | **0.57571** |

### 9.3 "Universales" (top-5 en 3+ ligas)

1. **Elo + Defense** (elo_home/away/diff + goals_conceded_avg): top-5 en 6/7 ligas
2. **Elo goal-weighted** (elo_gw_home/away/diff): top-3 en 5/7 ligas
3. **Efficiency + Elo** (finish_eff, def_eff + elo): top-5 en 4/7 ligas
4. **Form + Elo** (win_rate5 + form_diff + elo): top-5 en 5/7 ligas

---

## 10. Recomendaciones para v1.0.2

### 10.1 Acciones Completadas

1. **Market Anchor activado** (`MARKET_ANCHOR_ENABLED=true`) para Argentina (α=1.0) — implementado 2026-02-08
2. **FDUK backfill completado** — odds históricas para 16 ligas, Argentina 100% desde 2023

### 10.2 v1.0.2: Nuevo Feature Set

Basado en los hallazgos del lab + SHAP 2023+:

1. **Eliminar**: rest_days (3 features), away_matches_played, home_shots_avg → de 17 a 12
2. **Agregar**: elo_home, elo_away, elo_diff (3) → 15 features
3. **Agregar odds como features**: ahora viable para LATAM (100% cobertura 2023+). Odds mejoran el modelo cuando la cobertura es completa.
4. **Considerar per-liga**: interactions para ARG, efficiency para MX, league_draw_rate (SHAP dominante en ARG)
5. **Re-tunear hyperparams**: depth=2, mcw=10-14 (más regularización)

### 10.3 v1.0.3: Market-Aware Model

- Entrenar modelo donde odds de-vigorizadas son la feature principal
- Features adicionales capturan "edges" sobre el mercado: forma reciente, xG, fatiga
- **Ya viable**: FDUK backfill provee odds históricas suficientes (1,300+ matches ARG desde 2023)

---

## 11. Archivos de Referencia

### Scripts (conservar en repo)

| Script | Descripción | Modo |
|--------|-------------|------|
| `scripts/feature_lab.py` | Lab principal: 101 tests + Optuna + SHAP | `--optuna`, `--shap`, `--league`, `--extract`, `--min-date` |
| `scripts/feature_diagnostic.py` | Permutation + ablation + canary | ATI-approved, PIT-safe |
| `scripts/xg_signal_test.py` | A/B test xG signal | `--all-leagues`, `--no-shots` |
| `scripts/shadow_eval_abe.py` | Shadow model evaluation | READ-ONLY, psycopg2 |

### Outputs (en `scripts/output/`)

| Archivo | Contenido |
|---------|-----------|
| `lab/feature_lab_results.json` | Resultados 101 tests × 4 EUR leagues |
| `lab/feature_lab_results_optuna.json` | Resultados 16 Optuna × 3 leagues |
| `lab/shap_analysis_128.json` | SHAP Argentina 2023+ (8 tests, 100% odds) |
| `lab/lab_data_{id}.csv` | Datasets cacheados por liga |
| `feature_diagnostic_argentina_128.json` | Diagnostic detallado Argentina |
| `feature_diagnostic_v1.0.1_*.json` | Diagnostics globales |
| `xg_signal_test_combined.json` | xG A/B test 4 ligas |
| `shadow_eval_abe.json` | Shadow eval (N=151, CONTINUE-SHADOW) |

---

## 12. Lecciones Aprendidas

1. **El mercado es el rey** — En Europa, incluso el mejor modelo XGBoost con 22 features + Optuna no puede superar odds de-vigorizadas convertidas a probabilidades simples.

2. **Menos es más** — Los modelos con 3-9 features seleccionadas ganan consistentemente vs los de 17+. La regularización de Optuna (depth=2, mcw alto) confirma que el modelo necesita ser más simple.

3. **Elo es la feature más valiosa** — Un modelo con solo 3 features Elo (home/away/diff) compite con modelos de 17 features en todas las ligas.

4. **Las features importan más que los hyperparams** — Seleccionar las features correctas (elo + defense) mejora más que tunear hyperparams (Optuna da +2-4%, features correctas dan +5-7%).

5. **Cada liga tiene su receta** — Argentina responde a interactions, Liga MX a efficiency, Europa a elo_gw + defense. No existe un "feature set universal óptimo".

6. **xG no es la bala de plata** — En forma rolling, xG no agrega señal significativa sobre las features rolling existentes. Puede ser más útil como feature puntual (xG del último partido) que como promedio.

7. **Las odds históricas SÍ son recuperables** — API-Football purga odds post-FT, pero Football-Data UK (football-data.co.uk) provee CSVs con odds históricas (Pinnacle closing para LATAM, B365/PS para EUR). El backfill FDUK resolvió el bottleneck de cobertura LATAM desde 2023.

8. **Verificar cobertura de datos ANTES de concluir** — La conclusión original "odds dañan Argentina" era un artefacto de 46% cobertura. Con 100% (2023+), odds son una feature útil. Siempre verificar que el dataset es representativo antes de sacar conclusiones.

9. **Smoke test obligatorio para mappings** — El bug de aliases ext_id→int_id causó meses de datos parciales. Implementar siempre un smoke test que verifique cada mapping contra la DB antes de ejecutar backfills.
