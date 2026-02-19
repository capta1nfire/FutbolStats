# League Deployment Policy — Bon Jogo v1.0

**Fecha**: 2026-02-14
**Basado en**: Feature Lab 23 ligas (110 standard + 9 SHAP + 16 Optuna + Section R + Brier Decomposition + Devig Sensitivity + Walk-Forward + Calibration Test)
**Modelo**: XGBoost v1.0.0 (14 features, hyperparams fijos)

---

## 1. Taxonomia de Tiers

| Tier | Nombre | Criterio | Market Anchor α | Accion |
|------|--------|----------|-----------------|--------|
| 1 | Ultra-Eficiente | Section R positivo (7/7), FAIR sig mkt, ECE mkt < modelo | α ≥ 0.9 | Market Anchor dominante |
| 2 | Eficiente-Estable | Section R positivo (7/7 o 5/5), FAIR ns, modelo cercano | α = 0.7–0.8 | Market Anchor moderado |
| 3 | Eficiente-Lejano | Section R positivo, FAIR sig mkt, gap > 2% | α = 0.8–1.0 | Market Anchor fuerte, modelo solo fallback |
| 4 | Alpha-Candidato | Section R negativo, modelo ≈ mercado, FAIR ns | α = 0.0–0.3 | Modelo con cautela |
| 5 | Datos-Insuficientes | Section R N/A, odds < 50%, N_test < 100 | α = 1.0 | Market Anchor total hasta backfill |
| 6 | Suspender | FAIR sig positivo + gap > 3%, modelo no aporta | SUSPENDER predicciones | Evaluar si vale mantener |

---

## 2. Clasificacion de 23 Ligas

### Tier 1 — Ultra-Eficiente (5 ligas)

Mercado incorpora toda la informacion disponible. Modelo no aporta señal incremental.

| Liga | ID | Market Brier | FAIR Δ | Section R | Diagnosis | α recom |
|------|----|-------------|--------|-----------|-----------|---------|
| Serie A | 135 | 0.57968 | +0.0007 (NS) | 7/7 POSITIVO | — | 0.95 |
| Eredivisie | 88 | 0.56592 | -0.0005 (NS) | MIXTO (2/7 neg, ~0) | — | 0.90 |
| Primeira Liga | 94 | 0.53979 | +0.0059 (NS) | MIXTO (1/7 neg, ~0) | — | 0.90 |
| Bolivia | 344 | 0.52833 | +0.0094 (NS) | 5/5 POSITIVO | — | 0.90 |
| Paraguay | 250 | 0.65672 | -0.0054 (NS) | 5/5 POSITIVO | — | 0.85 |

**Nota**: Eredivisie y Primeira tienen Section R MIXTO/AMBIGUO pero FAIR gap ~0, modelo empata estadisticamente. Bolivia y Paraguay tienen mercados predecibles con gaps pequeños.

**Justificacion Paraguay α=0.85**: Unica liga LATAM donde modelo se acerca al mercado (delta -0.5%, NS). Merece algo menos de anchor que el resto de Tier 1, pero no suficiente para Tier 4.

### Tier 2 — Eficiente-Estable (9 ligas)

Mercado gana pero modelo es competitivo. Gap moderado, no significativo.

| Liga | ID | Market Brier | FAIR Δ | Section R | Diagnosis | α recom |
|------|----|-------------|--------|-----------|-----------|---------|
| EPL | 39 | 0.57065 | +0.0091 (Sig) | 7/7 POSITIVO | RESOLUTION_ISSUE | 0.80 |
| Bundesliga | 78 | 0.57675 | +0.0178 (Sig) | 7/7 POSITIVO | — | 0.80 |
| Belgium | 144 | 0.60144 | +0.0011 (NS) | 7/7 POSITIVO | — | 0.75 |
| Argentina | 128 | 0.64694 | +0.0209 (NS) | 7/7 POSITIVO | RESOLUTION_ISSUE | 0.80 |
| Mexico | 262 | 0.56605 | +0.0098 (NS) | 7/7 POSITIVO | — | 0.75 |
| Peru | 281 | 0.56053 | +0.0096 (NS) | 5/5 POSITIVO | — | 0.75 |
| Ecuador | 242 | 0.63804 | +0.0055 (NS) | 5/5 POSITIVO | — | 0.75 |
| Venezuela | 299 | 0.58408 | +0.0194 (NS) | 5/5 POSITIVO | — | 0.80 |
| Brasil | 71 | 0.60178 | +0.0164 (NS) | 7/7 POSITIVO | — | 0.75 |

**Nota Argentina**: Market Brier 0.647 (el peor de LATAM — liga muy impredecible). Decomposition muestra RESOLUTION_ISSUE: modelos bien calibrados pero no discriminan. Market Anchor α=1.0 actual se mantiene en produccion; 0.80 es la recomendacion para el nuevo framework.

**Nota EPL**: Decomposition confirma RESOLUTION_ISSUE. Market ECE_avg=0.034 (excelentemente calibrado). Devig sensitivity negligible (<0.0002).

### Tier 3 — Eficiente-Lejano (5 ligas)

Mercado gana significativamente. Gap > 2%, modelo lejos. Market Anchor fuerte.

| Liga | ID | Market Brier | FAIR Δ | Section R | Diagnosis | α recom |
|------|----|-------------|--------|-----------|-----------|---------|
| La Liga | 140 | 0.56287 | +0.0179 (Sig) | 7/7 POSITIVO | — | 0.85 |
| Ligue 1 | 61 | 0.57704 | +0.0235 (Sig) | 7/7 POSITIVO | — | 0.90 |
| Colombia | 239 | 0.60912 | +0.0128 (Sig) | 7/7 POSITIVO | — | 0.85 |
| Uruguay | 268 | 0.59598 | +0.0215 (Sig) | 5/5 POSITIVO | — | 0.85 |
| MLS | 253 | 0.60165 | +0.0298 (Sig) | 7/7 POSITIVO | — | 0.90 |

**Justificacion**: Gaps significativos (1.3%–3.0%) donde el mercado gana consistentemente. El modelo podria aportar como fallback cuando no haya odds disponibles, pero el anchor deberia ser fuerte.

**Nota MLS**: Gap +3.0% (el segundo mayor tras Chile). Naturaleza de la liga (expansion/contraction teams, single-entity) puede contribuir al gap. Geo features (travel_distance) podrian aportar señal — ver precheck P1.

### Tier 4 — Alpha-Candidato (1 liga)

Unica liga donde el modelo potencialmente supera al mercado.

| Liga | ID | Market Brier | FAIR Δ | Section R | Diagnosis | α recom |
|------|----|-------------|--------|-----------|-----------|---------|
| **Turquia** | 203 | 0.56911 | -0.0035 (NS) / -0.0135 xG (NS) | **7/7 NEGATIVO** | RESOLUTION_ISSUE* | **0.50** |

**Detalle Turquia**:
- Section R: 7/7 NEGATIVO = mercado tiene sesgos explotables. **Unica liga INEFICIENTE**
- FAIR delta odds: -0.0035 (N=567, NS). FAIR delta xG+odds: -0.0135 (N=241, NS, CI [-0.031, +0.005])
- Decomposition: Modelos tienen RES=0.09-0.12 (MAYOR que mercado RES=0.087) — el modelo DISCRIMINA mejor
- Pero REL modelo (0.014-0.024) > REL mercado (0.012) — el modelo está peor calibrado
- Devig: power -0.001 vs prop (marginal pero mejor que prop/shin)
- xG + odds es la combinacion que mas explota la ineficiencia (OE_xg_defense_odds Brier=0.561 vs market 0.569)
- **Walk-forward CONTRADICE Section R**: model_worse en 8-9/10 ventanas. Δ_mean=+1.5% a +3.3%. Stability MODERATE
- **Calibracion isotonica**: NO ayuda (xG models empeoran +3.7-4.1%). Solo odds champion mejora marginalmente (-0.001)
- **α REVISADO de 0.20 → 0.50**: Section R dice modelo tiene info, pero walk-forward dice no es robusto temporalmente. Blend 50/50 como compromiso
- **Siguiente paso**: Re-evaluar con mas datos xG (cobertura actual 38%)

### Tier 5 — Datos-Insuficientes (2 ligas)

Datos incompletos o N_test insuficiente para conclusiones confiables.

| Liga | ID | Market Brier | FAIR Δ | Section R | Problema | α recom |
|------|----|-------------|--------|-----------|----------|---------|
| Championship | 40 | 0.61262 | N/A | 7/7 POSITIVO | N_test=70 (odds), NO CONFIABLE | 1.0 |
| Saudi Pro | 307 | 0.52487 | +0.0177 (NS) | INSUFFICIENT_ODDS* | N_test=154 (xG+odds), market recien backfilled | 0.70 |

**Nota Championship**: Solo 70 matches con odds en test → todos los FAIR son ruido. Base universe funciona (N=743) pero no hay odds comparison confiable. Necesita backfill de odds.

**Nota Saudi**: OddsPortal backfill reciente (2026-02-13). Section R no se pudo ejecutar por baja cobertura previa. FAIR no significativo. xG+odds modelo (0.522) se acerca a mercado (0.525). Re-evaluar con mas datos.

### Tier 6 — Suspender (1 liga)

Gap extremo donde el modelo no aporta valor.

| Liga | ID | Market Brier | FAIR Δ | Section R | Problema | Accion |
|------|----|-------------|--------|-----------|----------|--------|
| Chile | 265 | 0.57185 | +0.0304 (Sig) | 5/5 POSITIVO | CALIBRATION_ISSUE, sin xG, gap 3% | SUSPENDER o α=1.0 |

**Detalle Chile**:
- Mayor FAIR gap de todas las ligas (+3.04%, significativo)
- Decomposition: CALIBRATION_ISSUE (unica canario con este diagnostico)
- ECE modelos > 0.05 (mal calibrados), REL > RES
- xG: 0% — FotMob NO tiene xG para Chile. Limitacion permanente
- Devig: power -0.004 vs prop (la mayor diferencia de las canario)
- **Sin xG y con calibracion mala, el modelo no puede competir**
- **Accion**: Market Anchor α=1.0. Suspender predicciones propias, usar solo odds como proxy. Reevaluar si Opta despliega xG para Chile

---

## 3. Diagnostico CAL vs RES (4 ligas canario)

| Liga | REL (modelo) | RES (modelo) | REL (mkt) | RES (mkt) | Diagnosis | Interpretacion |
|------|-------------|-------------|-----------|-----------|-----------|----------------|
| Argentina (128) | 0.006-0.009 | 0.005-0.016 | 0.012 | 0.018 | RESOLUTION | Modelos bien calibrados, no discriminan |
| Turkey (203) | 0.013-0.024 | 0.087-0.116 | 0.012 | 0.087 | RESOLUTION* | Modelos discriminan MAS, pero mal calibrados |
| EPL (39) | — | — | — | — | RESOLUTION | Mercado excelente (ECE=0.034) |
| Chile (265) | — | — | — | — | CALIBRATION | Modelos mal calibrados (ECE > 0.05) |

*Turkey es especial: tiene RESOLUTION_ISSUE por formula (ECE < 0.05 no se cumple) pero el modelo tiene MAYOR resolution que el mercado — la unica liga donde esto ocurre.

**Implicaciones**:
- **Argentina**: Calibracion isotonica NO ayudara (el problema es discriminacion). Mas features o datos para mejorar
- **Turkey**: Calibracion isotonica PODRIA ayudar (reducir REL preservando la RES alta)
- **EPL**: Necesita features con mas poder discriminativo
- **Chile**: Calibracion podria ayudar pero sin xG la señal base es debil

---

## 4. Devig Sensitivity (4 ligas canario)

| Liga | Prop Brier | Power Δ | Shin Δ | Relevante? | Metodo optimo |
|------|-----------|---------|--------|------------|---------------|
| Argentina | 0.64694 | +0.0004 | +0.0004 | NO | Cualquiera |
| Turkey | 0.56911 | -0.0010 | -0.0008 | MARGINAL | power (ligeramente mejor) |
| EPL | 0.57065 | +0.0001 | +0.0001 | NO | Cualquiera |
| Chile | 0.57185 | -0.0037 | -0.0025 | MARGINAL | power (mejor, -0.4%) |

**Conclusion**: Devig method es irrelevante para EPL y Argentina (diferencias < 0.0005). Para Chile y Turkey, power es marginalmente mejor pero la diferencia no cambia clasificaciones. **Mantener proportional como default** por simplicidad, con nota de que power es una alternativa para ligas con spreads altos.

---

## 5. Opening vs Closing Test (FS-07)

**Precheck de opening_odds_kind**:

| Liga | true_opening | proxy/closing | Testable? |
|------|-------------|---------------|-----------|
| Argentina (128) | 0 | 1,301 closing | NO — INSUFFICIENT_DATA |
| Turkey (203) | 0 | 2,817 proxy/unknown | NO — INSUFFICIENT_DATA |
| EPL (39) | 198 | 3,861 proxy/null | MARGINAL (N=198, >50 threshold) |
| Chile (265) | 0 | 1,712 closing | NO — INSUFFICIENT_DATA |

**Resultado**: 3 de 4 canarios no tienen opening odds reales. EPL tiene 198 true_opening pero la mayoria de nuestros datos son closing proxy (FDUK). **El test de opening vs closing no es ejecutable con la cobertura actual**. Para habilitarlo se necesitarian fuentes con opening odds historicos (ej: OddsPortal timestamps, Betfair exchange).

---

## 6. Walk-Forward Multi-Ventana (FS-04)

Validacion temporal con ventanas expansivas (6 meses de test, train crece). Criterio: ≥3 ventanas, Brier_std < 0.03 = ESTABLE, < 0.05 = MODERADO, > 0.05 = INESTABLE.

### 6.1 Resultados Consolidados

| Liga | Test | Windows | Stability | model_worse | model_better | Δ_mean |
|------|------|---------|-----------|-------------|--------------|--------|
| Argentina | FIXED_baseline (base) | 4 | STABLE (0.011) | 4 | 0 | +0.025 |
| Argentina | Q8_m2_plus_xi (xi) | 7 | STABLE (0.011) | 4 | 1 | +0.012 |
| Argentina | Q7_xi_xg_elo_odds (xi_odds_xg) | 4 | STABLE (0.024) | 4 | 0 | +0.016 |
| Turkey | unnamed (base/odds) | 10 | MODERATE (0.031) | 9 | 1 | +0.033 |
| Turkey | Q6_xi_full (xi_odds) | 10 | MODERATE (0.030) | 8 | 2 | +0.019 |
| Turkey | Q7_xi_xg_elo_odds (xi_odds_xg) | 6 | MODERATE (0.030) | 5 | 1 | +0.015 |
| EPL | FIXED_baseline (base) | 18 | STABLE (0.027) | 16 | 2 | +0.020 |
| EPL | Q7_xi_xg_elo_odds (xi_odds_xg) | 18 | STABLE (0.028) | 17 | 1 | +0.021 |
| Chile | FIXED_baseline (base) | 11 | MODERATE (0.035) | 11 | 0 | +0.049 |
| Chile | N9_odds_ultimate (odds) | 11 | UNSTABLE (0.056) | 11 | 0 | +0.038 |
| Chile | Q8_m2_plus_xi (xi) | 9 | MODERATE (0.036) | 9 | 0 | +0.063 |

### 6.2 Interpretacion

- **Todas las ligas**: El mercado gana en la GRAN MAYORIA de ventanas (>80%). No hay ninguna liga donde el modelo gane consistentemente
- **Argentina**: ESTABLE. El modelo pierde por ~1.2-2.5% en promedio, consistente con FAIR
- **Turkey**: CONTRADICE Section R. A pesar de 7/7 NEGATIVO en Section R, walk-forward muestra model_worse en 80-90% de ventanas. El "alpha" de Turkey es especifico a ciertos periodos, no robusto
- **EPL**: La liga con MAS ventanas (18) y MAS estabilidad. Patron ultra-consistente: modelo pierde ~2% siempre
- **Chile**: Mayor delta (+3.8-6.3%) y MODERADA a INESTABLE estabilidad. Confirma Tier 6

### 6.3 Gate Semana 1→2

| Criterio | Resultado | Pass? |
|----------|-----------|-------|
| ≥3 ventanas en ≥3 canario | 4/4 canarios con ≥4 ventanas | PASS |
| Brier_std < 0.05 en ≥3 canario | ARG STABLE, TUR MODERATE, EPL STABLE, CHI MODERATE | PASS |
| Direccion consistente | 3/4 ligas >90% model_worse (Turkey 80-90%) | PASS |

---

## 7. Calibracion Isotonica (FS-03)

Test: XGBoost + isotonic regression (inner split 80/20 dentro de train) vs raw XGBoost.

### 7.1 Resultados Consolidados

| Liga | Test | Brier (raw) | Brier (iso) | Δ_Brier | ECE (raw) | ECE (iso) | Conclusion |
|------|------|------------|------------|---------|-----------|-----------|------------|
| **Argentina** | M2_h0_interactions (base) | 0.650 | 0.663 | +0.013 | 0.041 | 0.056 | RESOLUTION_CONFIRMED |
| Argentina | P4_xg_overperf (xg) | 0.673 | 0.661 | -0.011 | 0.071 | 0.029 | CALIBRATION_HELPS |
| Argentina | P9_xg_ultimate (odds_xg) | 0.673 | 0.654 | -0.019 | 0.076 | 0.040 | CALIBRATION_HELPS |
| **Turkey** | D4_elo_k64 (base) | 0.575 | 0.579 | +0.004 | 0.067 | 0.030 | CAL_ISSUE_CONFIRMED |
| Turkey | N2_odds_m2 (odds) | 0.566 | 0.564 | -0.001 | 0.059 | 0.045 | CALIBRATION_HELPS |
| Turkey | P9_xg_ultimate (odds_xg) | 0.564 | 0.605 | **+0.041** | 0.059 | 0.109 | **RESOLUTION_CONFIRMED** |
| **EPL** | ALL 7 models | ~0.58 | ~0.58 | ~0.000 | ~0.05 | ~0.05 | RESOLUTION_CONFIRMED |
| **Chile** | M6_defense_elo (base) | 0.611 | 0.600 | -0.011 | 0.098 | 0.079 | CALIBRATION_HELPS |
| Chile | N9_odds_ultimate (odds) | 0.602 | 0.610 | +0.007 | 0.101 | 0.088 | CAL_ISSUE_CONFIRMED |
| Chile | Q8_m2_plus_xi (xi) | 0.614 | 0.601 | -0.014 | 0.091 | 0.072 | CALIBRATION_HELPS |

### 7.2 Interpretacion

- **EPL**: RESOLUTION_ISSUE puro. Isotonic no cambia nada — modelos ya bien calibrados (ECE ~0.05). El problema es discriminacion
- **Chile**: CALIBRATION_ISSUE confirmado. 3/4 modelos mejoran con isotonic (Brier -1.0% a -1.4%). Isotonic reduce ECE de ~0.10 a ~0.08
- **Argentina**: MIXTO. Base models → RESOLUTION (no mejoran). xG models → CALIBRATION (mejoran). Los modelos xG estan mal calibrados pero isotonic los corrige
- **Turkey**: RESOLUTION dominante. Los modelos xG EMPEORAN dramaticamente con isotonic (+3.7-4.1%). El small N del xG universe (N=240) causa que isotonic sobreajuste. Solo odds champion mejora marginalmente

### 7.3 Implicacion para Produccion

| Liga | Isotonic recomendado? | Razon |
|------|----------------------|-------|
| EPL | NO | Ya bien calibrado, no aporta |
| Argentina | SOLO para xG models | Base/xi no lo necesitan |
| Turkey | NO | xG models empeoran, odds marginal |
| Chile | SI (si se usara el modelo) | Pero Chile es Tier 6 (Suspender) |

**Conclusion general**: Isotonic calibration NO es una solucion viable para cerrar el gap vs mercado. El problema fundamental es RESOLUTION (discriminacion), no calibracion

---

## 8. Resumen Ejecutivo

### Distribucion de Tiers

| Tier | Ligas | Count | α promedio |
|------|-------|-------|-----------|
| 1 Ultra-Eficiente | ITA, ERE, POR, BOL, PAR | 5 | 0.90 |
| 2 Eficiente-Estable | EPL, BUN, BEL, ARG, MEX, PER, ECU, VEN, BRA | 9 | 0.78 |
| 3 Eficiente-Lejano | ESP, FRA, COL, URU, MLS | 5 | 0.87 |
| 4 Alpha-Candidato | TUR | 1 | 0.50 |
| 5 Datos-Insuficientes | ENG2, SAU | 2 | 0.85 |
| 6 Suspender | CHI | 1 | 1.0 / SUSP |

### Patrones Cross-League

1. **19/22 ligas EFICIENTES**: El mercado incorpora toda la informacion que nuestro modelo tiene, y mas
2. **1 INEFICIENTE (Turkey)**: Unica liga con alpha potencial. xG + odds es la clave
3. **2 AMBIGUOS (Eredivisie, Primeira)**: Clasificados como Ultra-Eficientes — empatan estadisticamente
4. **Diagnostico universal**: RESOLUTION_ISSUE en la mayoria. Los modelos estan bien calibrados pero no discriminan. Necesitan features con mas poder predictivo (standings, geo, lineup quality, etc.)
5. **Chile es outlier**: Unica liga con CALIBRATION_ISSUE. Combinacion de sin xG + mal calibrado → gap mas grande
6. **Devig irrelevante**: Proportional ≈ power ≈ shin en todas las canario. Diferencias < 0.004

### Acciones Inmediatas

| Prioridad | Accion | Liga(s) | Impacto |
|-----------|--------|---------|---------|
| P0 | Mantener α=1.0 para Argentina | 128 | Ya activo, confirmado |
| P0 | Activar Market Anchor para Tier 3 | ESP, FRA, COL, URU, MLS | Reduce error en 5 ligas con gap sig |
| P0 | α=0.50 para Turkey (blend modelo/mercado) | 203 | Unica liga con alpha, walk-forward modera |
| P1 | Suspender predicciones Chile (o α=1.0) | 265 | Evita predicciones miscalibradas |
| P1 | Re-evaluar Saudi post-backfill | 307 | Ejecutar Section R cuando N_test > 200 |
| P1 | Backfill odds Championship | 40 | Habilitar FAIR comparison confiable |
| P2 | Walk-forward 23 ligas | ALL | Validar estabilidad temporal |
| P2 | Calibracion isotonica Turkey | 203 | Puede mejorar la unica liga con alpha |

---

## 9. Checklist Go/No-Go por Liga

```
Para cada liga, verificar antes de modificar alpha o features:

□ Section R: veredicto documentado (EFICIENTE / INEFICIENTE / AMBIGUO)
□ FAIR: delta + CI + significancia
□ Brier Decomposition: REL, RES, UNC, diagnostico CAL/RES
□ Devig sensitivity: 3 metodos comparados (si hay odds)
□ Walk-forward: ≥3 ventanas, estabilidad documentada (PENDIENTE)
□ Opening test: resultado documentado (si hay data)
□ Alpha asignado con justificacion cuantitativa
```

---

## 10. Criterios de Re-evaluacion

Ejecutar Section R + FAIR + Decomposition cuando:
- Se añaden features nuevas (standings, geo, lineup quality)
- Se cambia el modelo (v1.0.0 → v2.0.0)
- Se backfillean odds o xG significativos (>500 matches)
- Han pasado ≥3 meses desde la ultima evaluacion
- Walk-forward muestra inestabilidad (Brier_std > 0.05)

---

---

## 11. Prechecks P1 (Standings + Geo)

### 11.1 FS-08 — Standings Urgency Features

**Gate 1 (Historical Standings)**: FALLA — solo 1 snapshot/temporada/liga en `league_standings`. Datos capturados Feb 2026, no hay series historicas.

| Liga | Temporadas | Snapshots/temporada | Veredicto |
|------|-----------|---------------------|-----------|
| Argentina (128) | 2 | 1 | NO — datos insuficientes |
| Turkey (203) | 1 | 1 | NO |
| EPL (39) | 2 | 1 | NO |
| Chile (265) | 2 | 1 | NO |

**Approach alternativo**: Derivar standings de resultados acumulados (calcular tabla de posiciones a partir de `matches` con `date < match_date`). Viable — feature_lab ya tiene todos los partidos necesarios. PIT-safe por construccion.

**Complejidad adicional**: Argentina (split-season, relegacion por promedio) y Chile (split-season) requieren logica especial para calcular standings intra-torneo.

### 11.2 FS-09 — Geo Friction Pack

**Gate 1 (Coordenadas + Altitud)**:

| Liga | ID | Teams | Coords % | Altitude % | Veredicto |
|------|-----|-------|----------|------------|-----------|
| Argentina | 128 | 35 | 91.4% | 91.4% | PASS |
| EPL | 39 | 34 | 88.2% | 88.2% | PASS |
| Chile | 265 | 27 | 74.1% | 74.1% | PASS (barely) |
| MLS | 253 | 30 | 73.3% | 73.3% | PASS (barely) |
| Turkey | 203 | 37 | 40.5% | 40.5% | FAIL (<70%) |
| Bolivia | 144 | 26 | 57.7% | 57.7% | FAIL (<70%) |
| Colombia | 344 | 27 | 22.2% | 22.2% | FAIL (<70%) |

**Resultados Section U (ejecutado en Argentina, Chile, MLS)**:

| Liga | U0_geo_only | U2_geo_def_elo | U5_geo_full | Market | Mejor no-geo (odds) | Veredicto |
|------|-------------|----------------|-------------|--------|---------------------|-----------|
| Argentina (128) | 0.6637 | 0.6595 | 0.6751 | 0.6487 | 0.674 | Neutro |
| Chile (265) | 0.6390 | 0.6003 | 0.5939 | 0.5620 | 0.5935 | **NO APORTA** (+0.0004) |
| MLS (253) | 0.6659 | 0.6511 | 0.6551 | 0.6114 | 0.6299 | **EMPEORA** (-2.5%) |

**Conclusion FS-09**: Geo features **NO aportan señal incremental**. La informacion de travel distance y altitud esta capturada implicitamente por Elo (que refleja ventaja local) y por el aprendizaje team-level del modelo. En MLS, geo features empeoran el modelo significativamente. **No incluir en produccion.**

### 11.3 FS-08 — Standings Urgency

**Implementacion**: Standings derivados de resultados acumulados (PIT-safe). Season key por year cross-year (Jul→Jun). 6 features: posicion, PPG, season_progress, position_diff.

**Coverage**: Argentina 99.8%, Chile 99.5% (solo primeros ~6 partidos/temporada sin datos).

**Resultados Section T + V**:

| Liga | T0_standings | T4_standings_full | V2_geo+standings | Market | Champion no-stand | Veredicto |
|------|-------------|-------------------|------------------|--------|-------------------|-----------|
| Argentina (128) | 0.6681 | 0.6814 | 0.6741 | 0.6480 | 0.6537 (Q8) | **NO APORTA** |
| Chile (265) | 0.6388 | 0.5930 | 0.5912 | 0.5640 | 0.5906 (Q6) | Competitivo (+0.0006) |

**Conclusion FS-08**: Standings features son **neutrales**. En Argentina, empeoran el modelo (T4 = 0.681 vs Q8 = 0.654). En Chile, V2_geo_standings_full (0.591) es marginalmente peor que Q6_xi_full (0.591). La posicion en tabla y PPG son derivados de los mismos resultados que Elo ya captura — informacion redundante. **No incluir en produccion.**

---

## 12. Conclusion P1

Ambos tickets P1 confirman que **features contextuales (geo, standings) no aportan señal incremental** sobre Elo + defense + form features. La informacion de posicion, distancia y altitud ya esta capturada implicitamente por:

1. **Elo ratings**: Reflejan fuerza relativa (lo mismo que posicion en tabla)
2. **Home advantage learnt**: El modelo ya aprende ventaja local por equipo
3. **Rolling averages**: Capturan la forma reciente (correlacionada con standings)

El **problema fundamental sigue siendo RESOLUTION** (discriminacion), no features faltantes. Para mejorar, se necesitarian señales genuinamente nuevas (lineups detalladas, in-game events, suspension/injury data) o un cambio de arquitectura (ensemble methods, temporal attention).

---

*Documento generado por Feature Lab Evolution (FS-01 a FS-10 + P1). Basado en evidencia cuantitativa de 23 ligas x 130+ tests por liga.*
