# Consulta: Futuro de Shadow B (Two-Stage Model)

## Contexto del Proyecto

Bon Jogo es un sistema de predicción de fútbol 1X2 (Home/Draw/Away) con XGBoost, cubriendo 25 ligas domésticas. El proyecto tiene 6 semanas de vida y una inversión de ~$500.

### Modelos activos

| Modelo | Arquitectura | Training Data | Estado |
|--------|-------------|---------------|--------|
| Model A (v1.0.1) | XGBoost multi-class (14 features) | ~22K matches, league-only, desde 2023 | Producción (baseline) |
| Shadow B (v1.1.0) | Two-Stage XGBoost (18+17 features) | Originalmente ~66K cross-league, reentrenado a 22K league-only | Evaluación paralela |
| Family S (v2.0) | XGBoost multi-class (24 features, incluye MTV) | ~22K matches, league-only, 5 ligas Tier 3 | Producción (5 ligas) |
| Market (Pinnacle) | Bookmaker odds de-vigged | N/A | Referencia |

### Arquitectura Two-Stage de Shadow

Shadow no es un XGBoost estándar. Es un modelo de dos etapas:

- **Stage 1**: Clasificador binario (empate vs no-empate). Usa 18 features, incluyendo `implied_draw` derivado de las odds del mercado. Draws ponderados con weight=1.2.
- **Stage 2**: Clasificador binario (home vs away) para los no-empates. Usa 17 features.
- **Composición soft**: p_draw = P(draw|S1), p_home = (1-p_draw) × P(home|nondraw, S2), p_away = (1-p_draw) × P(away|nondraw, S2). Suma a 1.0 por construcción.

Hiperparámetros conservadores: max_depth=3, learning_rate=0.05, n_estimators=100 en ambas etapas.

---

## Historia de Shadow

### Shadow v1 — Snapshot 3 (15 de enero)
- Entrenado sobre dataset **cross-league completo** (~66K+ matches, sin restricción de fecha ni de liga)
- Config registrada: `{draw_weight: 1.2, architecture: "two_stage", stage1_features: 18, stage2_features: 17}`
- No se registró `samples_trained`, `league_only`, ni `min_date` (campos no existían)
- Brier CV: 0.2094

### Descubrimiento del 1 de febrero
Se descubrió que de las 14 features del XGBoost, **solo 4 tenían datos reales** — el resto era NaN. Esto llevó a:
- Model A reentrenado como v1.0.1 con `league_only=true` (solo matches de la misma liga) y `min_date=2023-01-01`
- Dataset reducido de ~66K a ~22K matches
- Shadow NO fue reentrenado en ese momento — siguió corriendo con su snapshot v1 (66K)

### Shadow v2 — Snapshot 5 (10 de febrero, rebaseline)
- Reentrenado automáticamente con el MISMO régimen de Model A: `league_only=true`, `min_date=2023-01-01`
- 21,806 matches (vs ~66K+ del original)
- Brier CV mejoró ligeramente: 0.2094 → 0.2078
- Flag `is_rebaseline: true` — el sistema detectó que el snapshot anterior era de otro régimen

### Killswitch (9 de febrero)
Commit `bfa7a02` aplicó killswitch universal (mínimo 5 partidos de liga en 90 días). Shadow perdió cobertura en ligas LATAM en temporada temprana: Perú 18→0, Uruguay 8→0, Venezuela 14→0 predicciones. Cobertura total bajó de 100% a ~80%.

---

## Datos de Performance

### Head-to-head global (N=948 matches evaluados, ambos con predicción)

| Modelo | Correctos | Accuracy | Brier Score |
|--------|-----------|----------|-------------|
| Market (Pinnacle) | — | — | **0.5795** |
| Shadow B | 458 | **48.3%** | **0.6145** |
| Model A | 433 | 45.7% | 0.6343 |

Shadow cierra el 36% del gap entre Model A y el mercado en Brier Score.

### Exclusivos (cuando difieren en pick)

| Métrica | Shadow | Model A |
|---------|--------|---------|
| Aciertos exclusivos (el otro falló) | **84** | 59 |
| Ratio | **1.42x** | 1x |

### Accuracy por liga (Shadow edge, top 10)

| Liga | N | Shadow | Model A | Edge |
|------|---|--------|---------|------|
| Nations League | 36 | 66.7% | 50.0% | +16.7 |
| Liga MX | 36 | 47.2% | 38.9% | +8.3 |
| Süper Lig | 45 | 53.3% | 46.7% | +6.6 |
| Eredivisie | 47 | 36.2% | 29.8% | +6.4 |
| Ligue 1 | 45 | 55.6% | 51.1% | +4.5 |
| Chile | 23 | 56.5% | 52.2% | +4.3 |
| Premier League | 50 | 34.0% | 30.0% | +4.0 |
| Argentina | 75 | 49.3% | 45.3% | +4.0 |
| Colombia | 63 | 49.2% | 46.0% | +3.2 |
| La Liga | 49 | 51.0% | 49.0% | +2.0 |

Shadow gana en **20 de 25 ligas**. Model A solo gana en Brasil (-11.6%), Uruguay (-12.5%, N=8) y Paraguay (-3.6%).

### Alineación con el mercado (cuando Shadow y Model A difieren)

| Escenario | N | Shadow acierta | Model A acierta |
|-----------|---|----------------|-----------------|
| Shadow = Model A (acuerdan) | 724 | 51.4% | 51.4% |
| **Shadow = Market, difiere de Model A** | **120** | **45.8%** | **20.8%** |
| Shadow solo (difiere de ambos) | 96 | 29.2% | 33.3% |
| Model A = Market, difiere de Shadow | 48 | 29.2% | 41.7% |

En los 120 partidos donde Shadow se alinea con el mercado y Model A no, Shadow acierta 45.8% vs 20.8%. Shadow "descubrió" patrones del mercado sin ver las odds directamente (las usa solo para `implied_draw` en Stage 1).

### El problema del empate

| Modelo | Veces que predice empate | Acierta | Accuracy |
|--------|--------------------------|---------|----------|
| Model A | 101 | 25 | **24.8%** |
| Shadow | 54 | 19 | **35.2%** |

Model A predice empate el doble de veces y acierta 10pp menos. La Stage 2 de Shadow reclasifica empates dudosos a home/away con éxito.

### Calibración (probabilidades vs realidad)

| Confianza Shadow | N | Shadow acierta | Model A acierta | Shadow Brier | Model A Brier |
|-----------------|---|----------------|-----------------|--------------|---------------|
| < 40% | 307 | 35.5% | 31.9% | 0.6709 | 0.6904 |
| 40-50% | 401 | 48.6% | 46.9% | 0.6307 | 0.6392 |
| 50-60% | 178 | 59.0% | 55.1% | 0.5713 | 0.6055 |
| 60-70% | 52 | 76.9% | 76.9% | 0.3927 | 0.4274 |
| 70%+ | 10 | 90.0% | 90.0% | 0.2271 | 0.2949 |

Shadow tiene mejor Brier en **todos** los buckets. Incluso con accuracy igual (60-70%), Shadow asigna mejores probabilidades.

### Evolución temporal

| Semana | Snapshot | N | Shadow | Model A | Edge |
|--------|----------|---|--------|---------|------|
| Ene 12 | v1 (66K) | 114 | 42.1% | 46.5% | -4.4 |
| Ene 19 | v1 (66K) | 193 | **48.2%** | 38.9% | **+9.3** |
| Ene 26 | v1 (66K) | 309 | 49.5% | 49.5% | 0.0 |
| Feb 2 | v1 (66K) | 147 | 46.3% | 45.6% | +0.7 |
| Feb 9 | v1 (66K) | 167 | **53.3%** | 46.7% | **+6.6** |
| Feb 16 | v2 (22K) | 18 | 38.9% | 38.9% | 0.0 |

La mejor semana de Shadow fue Feb 9 (53.3%) — justo ANTES del rebaseline a 22K. Shadow v2 (22K league-only) solo tiene N=18, pero la ventaja desapareció.

---

## Hipótesis sobre la ventaja de Shadow

1. **Volumen cross-league (66K vs 22K)**: Shadow v1 se entrenó con todo el dataset cross-league. Aunque solo 4 features tenían datos, XGBoost maneja NaN nativamente. 66K partidos dan más poder estadístico para las features que SÍ existen. Patrones universales del fútbol (ventaja local, momentum, ritmo de empates) cruzan ligas.

2. **Arquitectura Two-Stage**: La descomposición draw/no-draw + home/away permite calibración más fina. Stage 1 aprende CUÁNDO hay empate (usando `implied_draw` del mercado). Stage 2 se especializa en la dirección. Model A intenta resolver las 3 clases simultáneamente.

3. **Ambos factores combinados**: El volumen cross-league alimenta mejor a la Two-Stage porque la Stage 1 (draw detection) se beneficia más de ver empates de TODAS las ligas — los empates son el evento más difícil de predecir y tener más ejemplos importa más aquí.

---

## Decisión solicitada

### Opción A: Restaurar Shadow con entrenamiento cross-league
- Reentrenar Shadow con dataset completo (sin `league_only`, posiblemente sin `min_date` o con min_date más antiguo)
- Restaurar cobertura eliminando o relajando killswitch para Shadow
- Evaluar durante 4 semanas adicionales (N~300+)
- Riesgo: puede haber contaminación cross-league que infle métricas

### Opción B: Mantener Shadow con régimen actual (league-only) y observar
- Shadow v2 (22K) solo tiene N=18. Puede necesitar más volumen para concluir
- Si Shadow v2 resulta comparable a Model A, la ventaja era del volumen cross-league
- Si Shadow v2 resulta mejor, la ventaja es de la arquitectura Two-Stage
- Riesgo: si la ventaja ERA del cross-league, la perdimos sin saberlo

### Opción C: A/B controlado — dos Shadows en paralelo
- Shadow-CL: entrenado cross-league (~66K)
- Shadow-LO: entrenado league-only (~22K)
- Ambos corriendo en shadow mode simultáneamente durante 4 semanas
- Esto aísla el factor: arquitectura vs volumen
- Riesgo: complejidad operacional

### Opción D: Dejar morir Shadow y absorber aprendizajes en Model A
- Aplicar Two-Stage como nueva arquitectura de Model A (reemplazando single-stage XGBoost)
- Mantener league-only training
- Riesgo: perdemos la señal cross-league sin haberla evaluado

---

## Preguntas específicas para los auditores

1. **¿El entrenamiento cross-league (66K) es señal legítima o contaminación?** Los datos muestran que Shadow superó a Model A en 20/25 ligas, incluyendo ligas donde Shadow no tenía datos league-only suficientes. ¿Esto indica señal universal, o hay un riesgo de overfitting cross-league que no estamos midiendo?

2. **¿La mejora de Shadow viene principalmente de la arquitectura Two-Stage o del volumen de datos?** Shadow v1 (66K, Two-Stage) fue claramente superior. Shadow v2 (22K, Two-Stage) parece haber perdido la ventaja (N=18, inconcluso). ¿Qué diseño experimental recomiendan para aislar estos factores?

3. **¿La Feature `implied_draw` (derivada de las odds) en Stage 1 constituye data leakage?** Stage 1 usa la probabilidad implícita de empate del mercado como feature. Esto le da a Shadow acceso indirecto a la sabiduría del mercado. ¿Es esto un uso legítimo de información disponible pre-kickoff, o invalida la comparación con Model A que no usa esa feature?

4. **¿Qué recomiendan como siguiente paso?** Dado el estado actual del proyecto (6 semanas, $500, un solo desarrollador, 25 ligas), ¿cuál de las opciones (A/B/C/D) o qué variante recomiendan?

5. **¿El Brier Score gap de 0.0198 (Shadow 0.6145 vs Model A 0.6343) es estadísticamente significativo con N=948?** ¿O necesitamos más volumen para concluir?

---

*Datos extraídos de producción el 17 de febrero de 2026. Todas las métricas usan predicciones frozen pre-kickoff evaluadas contra resultados FT (Full Time, excluye AET/PEN).*
