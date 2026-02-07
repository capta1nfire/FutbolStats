# Propuesta: Skill `ml-diagnostic`

**Fecha**: 2026-02-06
**Estado**: v2 — Corregido post-revisión ABE + Kimi
**Tipo**: Skill read-only (SELECT only, sin mutations)

---

## Problema

Model A (v1.0.1-league-only, XGBoost 14 features) emite predicciones en producción pero sus pronósticos son mediocres vs el mercado. Existen múltiples fuentes de diagnóstico dispersas (Sensor B, shadow reports, performance_metrics, ML health) pero no hay un flujo unificado que responda:

1. **Calibración**: ¿Las probabilidades emitidas reflejan la realidad? (si digo 40% home, ¿gana home ~40% de las veces?)
2. **Aprendizaje**: ¿El modelo mejora con más datos o está estancado?
3. **Features**: ¿Qué features aportan valor y cuáles son ruido?
4. **Cobertura**: ¿Hay features con datos faltantes que degradan predicciones?
5. **Mercado**: ¿Dónde divergimos del mercado y quién tiene razón?

---

## Alcance

### Qué ES el skill
- Herramienta de diagnóstico **read-only** invocable bajo demanda
- Ejecuta queries SQL sobre tablas existentes + interpreta resultados
- Output estructurado para que ABE/Kimi puedan evaluar y actuar
- Usa solo datos ya capturados (no entrena, no muta, no calcula features nuevas)

### Qué NO es
- No reemplaza PIT evaluation protocol (complementa)
- No entrena modelos ni ajusta hiperparámetros
- No propone features nuevas (detecta problemas con las actuales)
- No es un dashboard (es un reporte puntual)

---

## Paso 0: Verificación de Schema

Antes de ejecutar cualquier diagnóstico, el skill DEBE verificar que las columnas referenciadas existen:

```sql
-- Verificar columnas críticas de predictions
SELECT column_name FROM information_schema.columns
WHERE table_name = 'predictions' AND table_schema = 'public'
  AND column_name IN ('home_prob', 'draw_prob', 'away_prob', 'is_frozen',
    'frozen_odds_home', 'frozen_odds_draw', 'frozen_odds_away',
    'frozen_confidence_tier', 'model_version', 'match_id');

-- Verificar columnas críticas de matches
SELECT column_name FROM information_schema.columns
WHERE table_name = 'matches' AND table_schema = 'public'
  AND column_name IN ('home_goals', 'away_goals', 'status', 'date',
    'league_id', 'stats');

-- Verificar admin_leagues existe (para nombres de liga)
SELECT column_name FROM information_schema.columns
WHERE table_name = 'admin_leagues' AND table_schema = 'public'
  AND column_name IN ('league_id', 'name');
```

Si falta alguna columna, reportar error y NO continuar con queries que la referencien.

---

## 6 Áreas de Diagnóstico

**Parámetros globales** (configurables por invocación):
- `WINDOW_DAYS`: Ventana temporal (default 90 días). Aplica a áreas 1, 2, 3.
- `MIN_SAMPLES`: Mínimo de predicciones para calcular métricas confiables (default 100).
- `STATUS_FILTER`: Resultados terminados (default `('FT', 'AET', 'PEN')`). AET/PEN usan score a 90' para label de draw.

---

### 1. Calibración por Clase (ECE)

**Pregunta**: ¿Las probabilidades emitidas son fieles a la realidad, por cada outcome (home/draw/away)?

**Método**: Expected Calibration Error (ECE) **por clase**, no solo por top-prob. Para cada clase c ∈ {home, draw, away}: agrupar predicciones en bins por la prob predicha de esa clase, comparar avg(prob_predicha) vs freq(resultado=c).

```sql
-- ECE por clase: calibración de home_prob
-- (repetir para draw_prob y away_prob cambiando la columna y el indicator)
WITH resolved AS (
  SELECT
    p.home_prob AS predicted,
    CASE WHEN m.home_goals > m.away_goals THEN 1 ELSE 0 END AS actual
  FROM predictions p
  JOIN matches m ON m.id = p.match_id
  WHERE m.status IN ('FT', 'AET', 'PEN')
    AND p.model_version = 'v1.0.1-league-only'
    AND p.is_frozen = true
    AND m.date >= NOW() - INTERVAL '90 days'
)
SELECT
  WIDTH_BUCKET(predicted, 0.0, 1.0, 10) AS bin,
  COUNT(*) AS n,
  ROUND(AVG(predicted)::numeric, 3) AS avg_predicted,
  ROUND(AVG(actual)::numeric, 3) AS avg_actual,
  ROUND(ABS(AVG(predicted) - AVG(actual))::numeric, 4) AS calibration_error
FROM resolved
GROUP BY WIDTH_BUCKET(predicted, 0.0, 1.0, 10)
HAVING COUNT(*) >= 5
ORDER BY bin;
```

El skill ejecuta 3 veces (home, draw, away) y calcula:
- **ECE global** = Σ (n_bin / n_total) × |avg_predicted - avg_actual| para cada bin
- **ECE por clase** = ECE de home + ECE de draw + ECE de away (promediado)

**Output**: 3 tablas (una por clase) + ECE agregado.

**Interpretación**:
- ECE < 0.03: Calibración excelente
- ECE 0.03-0.08: Aceptable, margen de mejora
- ECE > 0.08: Problema de calibración, investigar
- **N < MIN_SAMPLES**: Reportar "INSUFFICIENT_DATA", no calcular ECE

---

### 2. Brier Score vs Mercado (Skill Score)

**Pregunta**: ¿Cuánto valor agrega el modelo sobre simplemente usar las odds del mercado?

**Método**: Brier skill score = `1 - (brier_model / brier_market)`.

**Odds de mercado**: Usar `predictions.frozen_odds_*` (odds congeladas al momento PIT-safe de la predicción), NO `matches.odds_*` que pueden cambiar post-predicción.

```sql
WITH scored AS (
  SELECT
    p.match_id,
    m.league_id,
    p.frozen_confidence_tier,
    -- Model Brier (3-class)
    POWER(p.home_prob - CASE WHEN m.home_goals > m.away_goals THEN 1 ELSE 0 END, 2)
    + POWER(p.draw_prob - CASE WHEN m.home_goals = m.away_goals THEN 1 ELSE 0 END, 2)
    + POWER(p.away_prob - CASE WHEN m.home_goals < m.away_goals THEN 1 ELSE 0 END, 2)
    AS brier_model,
    -- Market Brier (implied probs from frozen odds, normalized)
    POWER(
      (1.0 / p.frozen_odds_home) / (1.0/p.frozen_odds_home + 1.0/p.frozen_odds_draw + 1.0/p.frozen_odds_away)
      - CASE WHEN m.home_goals > m.away_goals THEN 1 ELSE 0 END, 2)
    + POWER(
      (1.0 / p.frozen_odds_draw) / (1.0/p.frozen_odds_home + 1.0/p.frozen_odds_draw + 1.0/p.frozen_odds_away)
      - CASE WHEN m.home_goals = m.away_goals THEN 1 ELSE 0 END, 2)
    + POWER(
      (1.0 / p.frozen_odds_away) / (1.0/p.frozen_odds_home + 1.0/p.frozen_odds_draw + 1.0/p.frozen_odds_away)
      - CASE WHEN m.home_goals < m.away_goals THEN 1 ELSE 0 END, 2)
    AS brier_market
  FROM predictions p
  JOIN matches m ON m.id = p.match_id
  WHERE m.status IN ('FT', 'AET', 'PEN')
    AND p.model_version = 'v1.0.1-league-only'
    AND p.is_frozen = true
    AND p.frozen_odds_home > 1.0
    AND p.frozen_odds_draw > 1.0
    AND p.frozen_odds_away > 1.0
    AND m.date >= NOW() - INTERVAL '90 days'
)
SELECT
  COUNT(*) AS n_matches,
  ROUND(AVG(brier_model)::numeric, 4) AS avg_brier_model,
  ROUND(AVG(brier_market)::numeric, 4) AS avg_brier_market,
  ROUND((1.0 - AVG(brier_model) / NULLIF(AVG(brier_market), 0))::numeric, 4) AS skill_score
FROM scored;
```

**Segmentación** (queries adicionales): misma query con `GROUP BY league_id` o `GROUP BY frozen_confidence_tier`.

**Interpretación**:
- Skill > 0: Modelo supera al mercado
- Skill = 0: Modelo = mercado (no agrega valor)
- Skill < 0: Mercado supera al modelo (situación actual esperada)
- **N < MIN_SAMPLES**: Reportar "INSUFFICIENT_DATA"

---

### 3. Evolución Temporal (¿Está Aprendiendo?)

**Pregunta**: ¿El Brier score mejora con el tiempo o está estancado?

**Método**: Brier score en ventanas semanales sobre los últimos 90 días.

```sql
WITH windowed AS (
  SELECT
    DATE_TRUNC('week', m.date) AS week,
    POWER(p.home_prob - CASE WHEN m.home_goals > m.away_goals THEN 1 ELSE 0 END, 2)
    + POWER(p.draw_prob - CASE WHEN m.home_goals = m.away_goals THEN 1 ELSE 0 END, 2)
    + POWER(p.away_prob - CASE WHEN m.home_goals < m.away_goals THEN 1 ELSE 0 END, 2)
    AS brier
  FROM predictions p
  JOIN matches m ON m.id = p.match_id
  WHERE m.status IN ('FT', 'AET', 'PEN')
    AND p.model_version = 'v1.0.1-league-only'
    AND p.is_frozen = true
    AND m.date >= NOW() - INTERVAL '90 days'
)
SELECT
  week,
  COUNT(*) AS n,
  ROUND(AVG(brier)::numeric, 4) AS brier_avg,
  ROUND(STDDEV(brier)::numeric, 4) AS brier_std
FROM windowed
GROUP BY week
HAVING COUNT(*) >= 5
ORDER BY week;
```

**Interpretación**:
- Tendencia descendente: Modelo mejora (posiblemente por mejor cobertura stats)
- Plano: Estancado — esperable dado que Model A NO se reentrena automáticamente
- Ascendente: Degradación (posible data drift, stats coverage degradada, o cambio en ligas)

**Nota clave**: Model A no se reentrena. La pregunta real es: ¿cuándo ext-D (reentrenado) supera a Model A? Eso se mide con:

```sql
-- Comparativa ext-D vs Model A (solo sobre matches donde AMBOS predicen)
WITH model_a AS (
  SELECT p.match_id,
    POWER(p.home_prob - CASE WHEN m.home_goals > m.away_goals THEN 1 ELSE 0 END, 2)
    + POWER(p.draw_prob - CASE WHEN m.home_goals = m.away_goals THEN 1 ELSE 0 END, 2)
    + POWER(p.away_prob - CASE WHEN m.home_goals < m.away_goals THEN 1 ELSE 0 END, 2) AS brier
  FROM predictions p
  JOIN matches m ON m.id = p.match_id
  WHERE m.status IN ('FT', 'AET', 'PEN') AND p.is_frozen = true
    AND p.model_version = 'v1.0.1-league-only'
),
ext_d AS (
  SELECT pe.match_id,
    POWER(pe.home_prob - CASE WHEN m.home_goals > m.away_goals THEN 1 ELSE 0 END, 2)
    + POWER(pe.draw_prob - CASE WHEN m.home_goals = m.away_goals THEN 1 ELSE 0 END, 2)
    + POWER(pe.away_prob - CASE WHEN m.home_goals < m.away_goals THEN 1 ELSE 0 END, 2) AS brier
  FROM predictions_experiments pe
  JOIN matches m ON m.id = pe.match_id
  WHERE m.status IN ('FT', 'AET', 'PEN')
)
SELECT
  COUNT(*) AS n_common,
  ROUND(AVG(a.brier)::numeric, 4) AS brier_model_a,
  ROUND(AVG(d.brier)::numeric, 4) AS brier_ext_d,
  ROUND((AVG(a.brier) - AVG(d.brier))::numeric, 4) AS improvement,
  SUM(CASE WHEN d.brier < a.brier THEN 1 ELSE 0 END) AS ext_d_wins
FROM model_a a
JOIN ext_d d ON d.match_id = a.match_id;
```

---

### 4. Feature Coverage y Data Gaps

**Pregunta**: ¿Hay features con datos faltantes que fuerzan al modelo a usar defaults?

**Método**: Auditar cobertura de `matches.stats` (shots, corners) por liga y temporada.

```sql
-- Cobertura de stats por liga (últimos 30 días)
SELECT
  m.league_id,
  al.name AS league_name,
  COUNT(*) AS total_ft,
  COUNT(CASE WHEN m.stats IS NOT NULL AND m.stats::text != '{}' AND m.stats::text != 'null' THEN 1 END) AS with_stats,
  ROUND(
    100.0 * COUNT(CASE WHEN m.stats IS NOT NULL AND m.stats::text != '{}' AND m.stats::text != 'null' THEN 1 END)
    / NULLIF(COUNT(*), 0), 1
  ) AS coverage_pct
FROM matches m
LEFT JOIN admin_leagues al ON al.league_id = m.league_id
WHERE m.status IN ('FT', 'AET', 'PEN')
  AND m.date >= NOW() - INTERVAL '30 days'
GROUP BY m.league_id, al.name
ORDER BY coverage_pct ASC;
```

**Impacto**: Sin stats, `home_shots_avg` y `home_corners_avg` (4 de 14 features = 29%) usan valor 0 o COALESCE genérico. Esto degrada predicciones significativamente para esas ligas.

**Output**: Tabla de ligas ordenada por cobertura ascendente. Ligas con < 80% coverage son candidatas a exclusión o features alternativas.

---

### 5. Sensor B: ¿Hay Señal de Mejora?

**Pregunta**: ¿El Sensor B (LogReg recalibrado cada 6h) supera a Model A? Si sí, Model A está stale.

**Método**: Consultar directamente `sensor_predictions` (últimos 90 días).

```sql
SELECT
  sensor_state,
  COUNT(*) AS n,
  ROUND(AVG(a_brier)::numeric, 4) AS model_a_brier,
  ROUND(AVG(b_brier)::numeric, 4) AS sensor_b_brier,
  SUM(CASE WHEN a_correct THEN 1 ELSE 0 END) AS a_wins,
  SUM(CASE WHEN b_correct THEN 1 ELSE 0 END) AS b_wins,
  ROUND(
    (0.667 - AVG(b_brier)) / NULLIF(0.667 - AVG(a_brier), 0)
  ::numeric, 3) AS signal_score
FROM sensor_predictions sp
JOIN matches m ON m.id = sp.match_id
WHERE sp.a_brier IS NOT NULL AND sp.b_brier IS NOT NULL
  AND m.date >= NOW() - INTERVAL '90 days'
GROUP BY sensor_state;
```

**Interpretación del signal_score** (fórmula: `(brier_uniform - brier_B) / (brier_uniform - brier_A)`, uniform=0.667):
- `> 1.10`: Sensor B supera significativamente → Model A puede estar stale, considerar reentrenamiento
- `0.90 - 1.10`: Empate técnico → Model A sigue vigente
- `< 0.90`: Sensor B peor (ruido) → LogReg no captura patrones, Model A OK
- **N < MIN_SAMPLES (con b_brier != NULL)**: Reportar "INSUFFICIENT_DATA"

---

### 6. Prediction Coverage Rate

**Pregunta**: ¿Cuántos matches elegibles reciben predicción? ¿Hay ligas con baja cobertura?

**Nota**: No se instrumenta el motivo del rechazo (kill-switch, falta de odds, etc.). Esta métrica mide cobertura end-to-end sin distinguir causa.

**Método**: Comparar matches recientes vs predicciones emitidas.

```sql
-- Matches con resultado vs predicciones emitidas (últimos 30 días)
WITH eligible AS (
  SELECT m.id, m.league_id
  FROM matches m
  WHERE m.status IN ('FT', 'AET', 'PEN')
    AND m.date >= NOW() - INTERVAL '30 days'
),
predicted AS (
  SELECT DISTINCT match_id FROM predictions
  WHERE model_version = 'v1.0.1-league-only'
)
SELECT
  e.league_id,
  al.name AS league_name,
  COUNT(*) AS total_matches,
  COUNT(p.match_id) AS with_prediction,
  COUNT(*) - COUNT(p.match_id) AS without_prediction,
  ROUND(100.0 * COUNT(p.match_id) / NULLIF(COUNT(*), 0), 1) AS prediction_rate_pct
FROM eligible e
LEFT JOIN predicted p ON p.match_id = e.id
LEFT JOIN admin_leagues al ON al.league_id = e.league_id
GROUP BY e.league_id, al.name
ORDER BY prediction_rate_pct ASC;
```

**Interpretación**:
- `prediction_rate < 50%`: Liga con baja cobertura. Posible causa: kill-switch (historial insuficiente), liga nueva, o sin odds.
- `prediction_rate > 90%`: Liga saludable.

---

## Reglas Determinísticas de Estado Global

El diagnóstico general se calcula con precedencia (de mayor a menor gravedad):

```
IF n_resolved < MIN_SAMPLES:
  Estado = INSUFFICIENT_DATA
  Acción = "Acumular al menos {MIN_SAMPLES} predicciones resueltas"

ELIF ece_avg > 0.15:
  Estado = MISCALIBRATED (CRITICAL)
  Acción = "Evaluar recalibración isotónica o reentrenamiento urgente"

ELIF ece_avg > 0.08:
  Estado = MISCALIBRATED
  Acción = "Investigar bins con mayor error; considerar recalibración"

ELIF skill_score < -0.10 AND brier_trend_slope > 0 (ascendente):
  Estado = DEGRADING
  Acción = "Modelo se degrada vs mercado con tendencia negativa; priorizar ext-D"

ELIF sensor_signal > 1.10 AND sensor_n >= MIN_SAMPLES:
  Estado = STALE
  Acción = "Sensor B supera a Model A; evaluar reentrenamiento o promoción ext-D"

ELIF skill_score < -0.05:
  Estado = UNDERPERFORMING
  Acción = "Modelo ligeramente por debajo del mercado; monitorear"

ELSE:
  Estado = HEALTHY
  Acción = "Continuar monitoreo regular"
```

**Umbrales** (configurables):

| Umbral | Valor Default | Descripción |
|--------|---------------|-------------|
| `ECE_CRITICAL` | 0.15 | ECE promedio para MISCALIBRATED critical |
| `ECE_WARN` | 0.08 | ECE promedio para MISCALIBRATED |
| `SKILL_DEGRADING` | -0.10 | Skill score para DEGRADING |
| `SKILL_UNDERPERFORM` | -0.05 | Skill score para UNDERPERFORMING |
| `SENSOR_SIGNAL_GO` | 1.10 | Signal score para STALE |
| `MIN_SAMPLES` | 100 | Mínimo para cálculos confiables |

---

## Formato de Output del Skill

```
# ML Diagnostic Report
Fecha: {fecha} | Modelo: {model_version} | Ventana: {WINDOW_DAYS}d | N resueltos: {n}

## 0. Schema Check
predictions: OK (17 cols) | matches: OK | admin_leagues: OK | sensor_predictions: OK

## 1. Calibración (ECE por clase)
ECE promedio: {valor} ({interpretación})
| Clase | ECE | Peor Bin | Pred Avg | Actual Avg | N |
|-------|-----|----------|----------|------------|---|
| home  | ... | ........ | ........ | .......... | . |
| draw  | ... | ........ | ........ | .......... | . |
| away  | ... | ........ | ........ | .......... | . |

## 2. Skill vs Mercado
Brier Model: {valor} | Brier Market: {valor} | Skill Score: {valor}
Odds usadas: predictions.frozen_odds_* (PIT-safe)
{interpretación}
Segmentación:
| Liga | N | Skill | Modelo vs Mercado |
| Tier | N | Skill | Modelo vs Mercado |

## 3. Evolución Temporal (últimos {WINDOW_DAYS}d)
| Semana | N | Brier Avg | Brier Std |
|--------|---|-----------|-----------|
| ...... | . | ......... | ......... |
Tendencia: {plana|descendente|ascendente} (slope: {valor})
ext-D vs Model A: N={n_common}, improvement={valor}

## 4. Feature Coverage
| Liga | FT+AET+PEN | Stats Coverage | Riesgo |
|------|-----------|---------------|--------|
| .... | ......... | ............. | ...... |
Features afectadas: home_shots_avg, home_corners_avg, away_shots_avg, away_corners_avg

## 5. Sensor B Signal
Signal Score: {valor} ({interpretación})
Model A Brier: {brier} | Sensor B Brier: {brier} | Head-to-head: {a_wins}-{b_wins}
N evaluados (con B): {n}

## 6. Prediction Coverage
| Liga | Matches | Con Predicción | Rate |
|------|---------|---------------|------|
| .... | ....... | ............. | .... |

## Diagnóstico General
Estado: {HEALTHY|UNDERPERFORMING|STALE|DEGRADING|MISCALIBRATED|INSUFFICIENT_DATA}
Umbrales aplicados: ECE={ece_avg} vs {ECE_WARN}, Skill={skill} vs {SKILL_UNDERPERFORM}, Signal={signal} vs {SENSOR_SIGNAL_GO}
Evidencia: {2-3 bullets}
Acción recomendada: {1 acción concreta}
```

---

## Herramientas del Skill

| Herramienta | Uso |
|-------------|-----|
| `mcp__railway-postgres__query` | Todas las queries SQL (SELECT only) |
| `Read` | Leer configs de modelo, features, archivos ML |
| `Grep` | Buscar patrones en código ML |
| `Glob` | Encontrar archivos de modelo/features |

**Restricciones**:
- NUNCA INSERT/UPDATE/DELETE
- NUNCA ejecutar training/predictions
- NUNCA mostrar API keys o tokens
- Queries con `LIMIT` (max 500 rows) y ventana temporal obligatoria
- `SET statement_timeout = '30s'` antes de queries pesadas (Kimi P1)

---

## Correcciones aplicadas (ABE P0 + Kimi P1)

### ABE P0-1: ECE incompleta/incorrecta (CORREGIDO)
- Cambio de top-prob a **calibración por clase** (3 queries: home, draw, away)
- Eliminado `predicted_prob` sin definir, `actual_hit` dummy, `WIDTH_BUCKET(max_prob, ...)` con alias no disponible
- Ahora: `WIDTH_BUCKET(predicted, 0.0, 1.0, 10)` sobre la prob de la clase específica

### ABE P0-2: Join incorrecto para league_name (CORREGIDO)
- Antes: `LEFT JOIN teams t ON t.id = m.league_id` (semánticamente incorrecto)
- Ahora: `LEFT JOIN admin_leagues al ON al.league_id = m.league_id`

### ABE P0-3: Queries sin ventana temporal (CORREGIDO)
- Todas las queries ahora incluyen `m.date >= NOW() - INTERVAL '{WINDOW_DAYS} days'`
- Default: 90 días para calibración/Brier/evolución, 30 días para coverage
- Parametrizable por invocación

### ABE P0-4: Odds de mercado no definidas (CORREGIDO)
- Antes: `matches.odds_*` (pueden cambiar post-predicción, no PIT-safe)
- Ahora: `predictions.frozen_odds_*` (odds congeladas al momento de la predicción)
- 98.8% de predictions congeladas tienen frozen_odds (verificado en prod)

### ABE P1-1: Reglas determinísticas para estado global (AGREGADO)
- Precedencia explícita: INSUFFICIENT_DATA > MISCALIBRATED > DEGRADING > STALE > UNDERPERFORMING > HEALTHY
- Umbrales numéricos configurables con defaults

### ABE P1-2: Status FT/AET/PEN (CORREGIDO)
- Antes: solo `WHERE m.status = 'FT'`
- Ahora: `WHERE m.status IN ('FT', 'AET', 'PEN')` en todas las queries

### ABE P1-3: Renombrar "kill-switch" (CORREGIDO)
- Antes: "Kill-Switch & Cobertura de Predicciones" (implica razón instrumentada)
- Ahora: "Prediction Coverage Rate" (métrica end-to-end sin asumir causa)

### Kimi: Umbrales numéricos en output (AGREGADO)
- Output ahora incluye umbrales aplicados junto a valores actuales

### Kimi: statement_timeout (AGREGADO)
- Documentado en restricciones del skill

---

## Preguntas resueltas

1. **Scope**: 6 áreas — aprobado por ambos auditores
2. **ECE vs Brier**: ECE por clase (ABE) con Brier como complemento — aprobado
3. **Temporal**: Medir estabilidad de Model A + comparativa ext-D — aprobado
4. **Feature importance**: Diferido. SHAP requiere cargar modelo → fuera de scope read-only SQL
5. **Frecuencia**: Bajo demanda. Cron diario opcional (Kimi: 05:00 UTC)
6. **Granularidad por liga**: Aggregate + segmentación en áreas 2 y 6

---

## Implementación Estimada

- **Archivo**: `.cursor/skills/ml-diagnostic/SKILL.md` (~250 líneas)
- **Queries**: 6 principales + 3 variantes (por liga, por tier, ext-D vs A)
- **Dependencias**: Solo tablas existentes (predictions, matches, admin_leagues, sensor_predictions, predictions_experiments)
- **Sin código nuevo en backend**: El skill es un prompt con queries predefinidas
- **Checklist pre-implementación** (Kimi):
  - [ ] Verificar índices en `predictions(match_id, model_version, is_frozen)`
  - [ ] Verificar índices en `matches(date, status)`
  - [ ] MIN_SAMPLES=100 para evitar varianza alta en ECE
