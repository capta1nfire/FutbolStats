---
name: ml-diagnostic
description: Ejecuta un diagnóstico ML read-only en Bon Jogo (calibración ECE por clase, Brier vs mercado con frozen_odds PIT-safe, tendencia temporal, coverage de stats, señal Sensor B, coverage de predicciones) y entrega un reporte determinístico (HEALTHY/UNDERPERFORMING/STALE/DEGRADING/MISCALIBRATED/INSUFFICIENT_DATA). Usar cuando el usuario pida "ml diagnostic", "calibración", "ECE", "brier", "skill score", "vs mercado", "Sensor B", "stale", "degrading" o auditorías de performance del modelo.
---

# ML Diagnostic (Bon Jogo)

## Objetivo

Generar un **reporte unificado** de salud del modelo 1X2 (Model A u otro) usando **solo SELECT** sobre PostgreSQL (Railway) + interpretación determinística.

## Restricciones (obligatorio)

- **READ-ONLY**: ejecutar **solo `SELECT`** (sin `INSERT/UPDATE/DELETE/DDL`).
- **No secretos**: nunca imprimir tokens, headers, API keys, DSNs.
- **Ventana temporal obligatoria**: todas las métricas deben filtrar por `m.date >= NOW() - INTERVAL '{WINDOW_DAYS} days'` (default 90d).
- **Timeout operativo**: antes de queries pesadas, fijar `statement_timeout` a 30s usando un `SELECT` (ver Paso 0.5).
- **PIT-safe market baseline**: para “mercado”, usar `predictions.frozen_odds_*` (no `matches.odds_*`).
- **MIN_SAMPLES**: si `n_resueltos < MIN_SAMPLES` ⇒ estado `INSUFFICIENT_DATA` y evitar conclusiones fuertes.

## Parámetros (defaults)

- **MODEL_VERSION**: `'v1.0.1-league-only'`
- **WINDOW_DAYS**: `90` (áreas 1–3, 5)
- **COVERAGE_DAYS**: `30` (áreas 4, 6)
- **MIN_SAMPLES**: `100`
- **N_BINS**: `10` (ECE)
- **MIN_BIN_N**: `5` (ECE)

## Paso 0: Schema check (fail-closed)

Ejecutar primero. Si falta una tabla/columna crítica, reportar y **no** ejecutar queries que la referencien.

```sql
-- 0A) tablas (public)
SELECT table_name
FROM information_schema.tables
WHERE table_schema = 'public'
  AND table_name IN ('predictions', 'matches', 'admin_leagues', 'sensor_predictions', 'predictions_experiments')
ORDER BY table_name;

-- 0B) predictions: columnas críticas
SELECT column_name
FROM information_schema.columns
WHERE table_schema = 'public' AND table_name = 'predictions'
  AND column_name IN (
    'match_id','model_version','created_at',
    'home_prob','draw_prob','away_prob',
    'is_frozen',
    'frozen_odds_home','frozen_odds_draw','frozen_odds_away',
    'frozen_confidence_tier'
  )
ORDER BY column_name;

-- 0C) matches: columnas críticas
SELECT column_name
FROM information_schema.columns
WHERE table_schema = 'public' AND table_name = 'matches'
  AND column_name IN ('id','date','status','league_id','home_goals','away_goals','stats')
ORDER BY column_name;

-- 0D) admin_leagues (opcional; solo para nombres)
SELECT column_name
FROM information_schema.columns
WHERE table_schema = 'public' AND table_name = 'admin_leagues'
  AND column_name IN ('league_id','name')
ORDER BY column_name;

-- 0E) sensor_predictions (opcional; solo si existe)
SELECT column_name
FROM information_schema.columns
WHERE table_schema = 'public' AND table_name = 'sensor_predictions'
  AND column_name IN ('match_id','sensor_state','a_brier','b_brier','a_correct','b_correct','created_at')
ORDER BY column_name;

-- 0F) predictions_experiments (opcional; para ext-D vs A)
SELECT column_name
FROM information_schema.columns
WHERE table_schema = 'public' AND table_name = 'predictions_experiments'
  AND column_name IN ('match_id','model_version','home_prob','draw_prob','away_prob','created_at')
ORDER BY column_name;
```

## Paso 0.5: statement_timeout (solo vía SELECT)

```sql
SELECT set_config('statement_timeout', '30000', true) AS statement_timeout_ms;
```

## Etiquetado del outcome (CRÍTICO para 1X2 a 90')

Para evaluación 1X2 (90’):
- **`PEN` y `AET` implican empate al 90’** ⇒ el label debe ser **DRAW**.
- En `FT`, el label se deriva de `home_goals` vs `away_goals`.

Usar esta lógica en las áreas 1–3:

```sql
CASE
  WHEN m.status IN ('AET','PEN') THEN 'draw'
  WHEN m.home_goals > m.away_goals THEN 'home'
  WHEN m.home_goals < m.away_goals THEN 'away'
  ELSE 'draw'
END
```

## 1) Calibración (ECE por clase)

Ejecutar 3 veces (home/draw/away) cambiando `PRED_COL` y `ACTUAL_INDICATOR`.

### 1A. ECE (plantilla)

```sql
WITH resolved AS (
  SELECT
    /* PRED_COL */ p.home_prob AS predicted,
    /* ACTUAL_INDICATOR (home) */
    CASE
      WHEN m.status IN ('AET','PEN') THEN 0
      WHEN m.home_goals > m.away_goals THEN 1 ELSE 0
    END AS actual
  FROM predictions p
  JOIN matches m ON m.id = p.match_id
  WHERE m.status IN ('FT','AET','PEN')
    AND p.model_version = 'v1.0.1-league-only'
    AND p.is_frozen = true
    AND m.date >= NOW() - INTERVAL '90 days'
),
binned AS (
  SELECT
    WIDTH_BUCKET(predicted, 0.0, 1.0, 10) AS bin,
    COUNT(*) AS n,
    AVG(predicted) AS avg_predicted,
    AVG(actual) AS avg_actual
  FROM resolved
  GROUP BY 1
),
filtered AS (
  SELECT * FROM binned WHERE n >= 5
),
totals AS (
  SELECT SUM(n)::float AS n_total FROM filtered
)
SELECT
  f.bin,
  f.n,
  ROUND(f.avg_predicted::numeric, 3) AS avg_predicted,
  ROUND(f.avg_actual::numeric, 3) AS avg_actual,
  ROUND(ABS(f.avg_predicted - f.avg_actual)::numeric, 4) AS calibration_error,
  ROUND(((f.n / t.n_total) * ABS(f.avg_predicted - f.avg_actual))::numeric, 6) AS ece_contribution,
  ROUND(SUM((f.n / t.n_total) * ABS(f.avg_predicted - f.avg_actual)) OVER ()::numeric, 6) AS ece_total
FROM filtered f
CROSS JOIN totals t
ORDER BY f.bin;
```

### 1B. Notas de interpretación

- Si `n_total < MIN_SAMPLES` ⇒ `INSUFFICIENT_DATA`.
- Tomar `ece_total` de cada clase y promediar \(home/draw/away\) ⇒ `ece_avg`.
- Guardar “peor bin” = el bin con `calibration_error` más alto (por clase).

## 2) Brier vs Mercado (Skill Score)
### 2A.0 Cobertura de `frozen_odds_*` (sesgo de selección)

Antes de calcular skill vs mercado, reportar cuántas predicciones **resueltas** tienen odds congeladas válidas. Si esta cobertura baja, el skill score puede quedar sesgado (solo evalúa el subconjunto con odds).

```sql
SELECT
  COUNT(*) AS n_total_resolved,
  COUNT(*) FILTER (
    WHERE p.frozen_odds_home > 1.0
      AND p.frozen_odds_draw > 1.0
      AND p.frozen_odds_away > 1.0
  ) AS n_with_frozen_odds,
  ROUND(
    100.0 * COUNT(*) FILTER (
      WHERE p.frozen_odds_home > 1.0
        AND p.frozen_odds_draw > 1.0
        AND p.frozen_odds_away > 1.0
    ) / NULLIF(COUNT(*), 0),
    1
  ) AS frozen_odds_coverage_pct
FROM predictions p
JOIN matches m ON m.id = p.match_id
WHERE m.status IN ('FT','AET','PEN')
  AND p.model_version = 'v1.0.1-league-only'
  AND p.is_frozen = true
  AND m.date >= NOW() - INTERVAL '90 days';
```

### 2A. Agregado

```sql
WITH scored AS (
  SELECT
    p.match_id,
    m.league_id,
    p.frozen_confidence_tier,
    -- Outcome one-hot (1X2 a 90')
    CASE
      WHEN m.status IN ('AET','PEN') THEN 0
      WHEN m.home_goals > m.away_goals THEN 1 ELSE 0
    END AS y_home,
    CASE
      WHEN m.status IN ('AET','PEN') THEN 1
      WHEN m.home_goals = m.away_goals THEN 1 ELSE 0
    END AS y_draw,
    CASE
      WHEN m.status IN ('AET','PEN') THEN 0
      WHEN m.home_goals < m.away_goals THEN 1 ELSE 0
    END AS y_away,
    -- Model Brier
    POWER(p.home_prob - (CASE
      WHEN m.status IN ('AET','PEN') THEN 0
      WHEN m.home_goals > m.away_goals THEN 1 ELSE 0
    END), 2)
    + POWER(p.draw_prob - (CASE
      WHEN m.status IN ('AET','PEN') THEN 1
      WHEN m.home_goals = m.away_goals THEN 1 ELSE 0
    END), 2)
    + POWER(p.away_prob - (CASE
      WHEN m.status IN ('AET','PEN') THEN 0
      WHEN m.home_goals < m.away_goals THEN 1 ELSE 0
    END), 2) AS brier_model,
    -- Market probs from frozen odds (normalized)
    (1.0 / p.frozen_odds_home) / (1.0/p.frozen_odds_home + 1.0/p.frozen_odds_draw + 1.0/p.frozen_odds_away) AS m_home,
    (1.0 / p.frozen_odds_draw) / (1.0/p.frozen_odds_home + 1.0/p.frozen_odds_draw + 1.0/p.frozen_odds_away) AS m_draw,
    (1.0 / p.frozen_odds_away) / (1.0/p.frozen_odds_home + 1.0/p.frozen_odds_draw + 1.0/p.frozen_odds_away) AS m_away
  FROM predictions p
  JOIN matches m ON m.id = p.match_id
  WHERE m.status IN ('FT','AET','PEN')
    AND p.model_version = 'v1.0.1-league-only'
    AND p.is_frozen = true
    AND p.frozen_odds_home > 1.0 AND p.frozen_odds_draw > 1.0 AND p.frozen_odds_away > 1.0
    AND m.date >= NOW() - INTERVAL '90 days'
),
scored2 AS (
  SELECT
    *,
    POWER(m_home - y_home, 2) + POWER(m_draw - y_draw, 2) + POWER(m_away - y_away, 2) AS brier_market
  FROM scored
)
SELECT
  COUNT(*) AS n_matches,
  ROUND(AVG(brier_model)::numeric, 4) AS avg_brier_model,
  ROUND(AVG(brier_market)::numeric, 4) AS avg_brier_market,
  ROUND((1.0 - AVG(brier_model) / NULLIF(AVG(brier_market), 0))::numeric, 4) AS skill_score
FROM scored2;
```

### 2B. Segmentación (opcional)
Para breakdown por **liga** o **tier**, reutilizar la query **2A** y en el `SELECT` final:
- Agregar `GROUP BY league_id` (y/o `GROUP BY frozen_confidence_tier`)
- Agregar `HAVING COUNT(*) >= 20`
- Ordenar `ORDER BY skill_score ASC` y limitar (`LIMIT 50`)

## 3) Evolución temporal (Brier semanal + slope)

### 3A. Serie semanal

```sql
WITH windowed AS (
  SELECT
    DATE_TRUNC('week', m.date) AS week,
    POWER(p.home_prob - (CASE WHEN m.status IN ('AET','PEN') THEN 0 WHEN m.home_goals > m.away_goals THEN 1 ELSE 0 END), 2)
    + POWER(p.draw_prob - (CASE WHEN m.status IN ('AET','PEN') THEN 1 WHEN m.home_goals = m.away_goals THEN 1 ELSE 0 END), 2)
    + POWER(p.away_prob - (CASE WHEN m.status IN ('AET','PEN') THEN 0 WHEN m.home_goals < m.away_goals THEN 1 ELSE 0 END), 2) AS brier
  FROM predictions p
  JOIN matches m ON m.id = p.match_id
  WHERE m.status IN ('FT','AET','PEN')
    AND p.model_version = 'v1.0.1-league-only'
    AND p.is_frozen = true
    AND m.date >= NOW() - INTERVAL '90 days'
),
weekly AS (
  SELECT
    week,
    COUNT(*) AS n,
    AVG(brier) AS brier_avg,
    STDDEV(brier) AS brier_std
  FROM windowed
  GROUP BY week
  HAVING COUNT(*) >= 5
)
SELECT
  week,
  n,
  ROUND(brier_avg::numeric, 4) AS brier_avg,
  ROUND(brier_std::numeric, 4) AS brier_std
FROM weekly
ORDER BY week;
```

### 3B. slope (degrading si > 0)

```sql
WITH weekly AS (
  SELECT
    DATE_TRUNC('week', m.date) AS week,
    AVG(
      POWER(p.home_prob - (CASE WHEN m.status IN ('AET','PEN') THEN 0 WHEN m.home_goals > m.away_goals THEN 1 ELSE 0 END), 2)
      + POWER(p.draw_prob - (CASE WHEN m.status IN ('AET','PEN') THEN 1 WHEN m.home_goals = m.away_goals THEN 1 ELSE 0 END), 2)
      + POWER(p.away_prob - (CASE WHEN m.status IN ('AET','PEN') THEN 0 WHEN m.home_goals < m.away_goals THEN 1 ELSE 0 END), 2)
    ) AS brier_avg
  FROM predictions p
  JOIN matches m ON m.id = p.match_id
  WHERE m.status IN ('FT','AET','PEN')
    AND p.model_version = 'v1.0.1-league-only'
    AND p.is_frozen = true
    AND m.date >= NOW() - INTERVAL '90 days'
  GROUP BY 1
),
indexed AS (
  SELECT week, brier_avg, ROW_NUMBER() OVER (ORDER BY week) AS idx
  FROM weekly
)
SELECT
  ROUND(regr_slope(brier_avg, idx)::numeric, 6) AS brier_trend_slope
FROM indexed;
```

### 3C. ext-D vs Model A (opcional)

Primero, identificar qué `model_version` existe en `predictions_experiments` (si aplica):

```sql
SELECT
  pe.model_version,
  COUNT(*) AS n
FROM predictions_experiments pe
JOIN matches m ON m.id = pe.match_id
WHERE m.status IN ('FT','AET','PEN')
  AND m.date >= NOW() - INTERVAL '90 days'
GROUP BY pe.model_version
ORDER BY n DESC;
```

Luego correr la comparativa directa (reemplazar `'ext-D'` por el `model_version` correcto):

```sql
WITH model_a AS (
  SELECT
    p.match_id,
    POWER(p.home_prob - (CASE WHEN m.status IN ('AET','PEN') THEN 0 WHEN m.home_goals > m.away_goals THEN 1 ELSE 0 END), 2)
    + POWER(p.draw_prob - (CASE WHEN m.status IN ('AET','PEN') THEN 1 WHEN m.home_goals = m.away_goals THEN 1 ELSE 0 END), 2)
    + POWER(p.away_prob - (CASE WHEN m.status IN ('AET','PEN') THEN 0 WHEN m.home_goals < m.away_goals THEN 1 ELSE 0 END), 2) AS brier
  FROM predictions p
  JOIN matches m ON m.id = p.match_id
  WHERE m.status IN ('FT','AET','PEN')
    AND p.model_version = 'v1.0.1-league-only'
    AND p.is_frozen = true
    AND p.home_prob IS NOT NULL AND p.draw_prob IS NOT NULL AND p.away_prob IS NOT NULL
    AND m.date >= NOW() - INTERVAL '90 days'
),
ext_d AS (
  SELECT
    pe.match_id,
    POWER(pe.home_prob - (CASE WHEN m.status IN ('AET','PEN') THEN 0 WHEN m.home_goals > m.away_goals THEN 1 ELSE 0 END), 2)
    + POWER(pe.draw_prob - (CASE WHEN m.status IN ('AET','PEN') THEN 1 WHEN m.home_goals = m.away_goals THEN 1 ELSE 0 END), 2)
    + POWER(pe.away_prob - (CASE WHEN m.status IN ('AET','PEN') THEN 0 WHEN m.home_goals < m.away_goals THEN 1 ELSE 0 END), 2) AS brier
  FROM predictions_experiments pe
  JOIN matches m ON m.id = pe.match_id
  WHERE m.status IN ('FT','AET','PEN')
    AND pe.model_version = 'ext-D'
    AND pe.home_prob IS NOT NULL AND pe.draw_prob IS NOT NULL AND pe.away_prob IS NOT NULL
    AND m.date >= NOW() - INTERVAL '90 days'
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

## 4) Feature coverage (stats)

```sql
SELECT
  m.league_id,
  al.name AS league_name,
  COUNT(*) AS total_finished,
  COUNT(CASE WHEN m.stats IS NOT NULL AND m.stats::text != '{}' AND m.stats::text != 'null' THEN 1 END) AS with_stats,
  ROUND(
    100.0 * COUNT(CASE WHEN m.stats IS NOT NULL AND m.stats::text != '{}' AND m.stats::text != 'null' THEN 1 END)
    / NULLIF(COUNT(*), 0), 1
  ) AS coverage_pct
FROM matches m
LEFT JOIN admin_leagues al ON al.league_id = m.league_id
WHERE m.status IN ('FT','AET','PEN')
  AND m.date >= NOW() - INTERVAL '30 days'
GROUP BY m.league_id, al.name
ORDER BY coverage_pct ASC
LIMIT 500;
```

## 5) Sensor B signal (si existe)

```sql
SELECT
  sp.sensor_state,
  COUNT(*) AS n,
  ROUND(AVG(sp.a_brier)::numeric, 4) AS model_a_brier,
  ROUND(AVG(sp.b_brier)::numeric, 4) AS sensor_b_brier,
  SUM(CASE WHEN sp.a_correct THEN 1 ELSE 0 END) AS a_wins,
  SUM(CASE WHEN sp.b_correct THEN 1 ELSE 0 END) AS b_wins,
  ROUND(((0.667 - AVG(sp.b_brier)) / NULLIF(0.667 - AVG(sp.a_brier), 0))::numeric, 3) AS signal_score
FROM sensor_predictions sp
JOIN matches m ON m.id = sp.match_id
WHERE sp.a_brier IS NOT NULL AND sp.b_brier IS NOT NULL
  AND m.status IN ('FT','AET','PEN')
  AND m.date >= NOW() - INTERVAL '90 days'
GROUP BY sp.sensor_state
ORDER BY n DESC;
```

## 6) Prediction coverage rate (no asume causa)

```sql
WITH eligible AS (
  SELECT m.id, m.league_id
  FROM matches m
  WHERE m.status IN ('FT','AET','PEN')
    AND m.date >= NOW() - INTERVAL '30 days'
),
predicted AS (
  SELECT DISTINCT match_id
  FROM predictions
  WHERE model_version = 'v1.0.1-league-only'
)
SELECT
  e.league_id,
  al.name AS league_name,
  COUNT(*) AS total_matches,
  COUNT(p.match_id) AS with_prediction,
  ROUND(100.0 * COUNT(p.match_id) / NULLIF(COUNT(*), 0), 1) AS prediction_rate_pct
FROM eligible e
LEFT JOIN predicted p ON p.match_id = e.id
LEFT JOIN admin_leagues al ON al.league_id = e.league_id
GROUP BY e.league_id, al.name
ORDER BY prediction_rate_pct ASC
LIMIT 500;
```

## Estado global (determinístico)

Aplicar precedencia:

1. `n_resueltos < MIN_SAMPLES` ⇒ `INSUFFICIENT_DATA`
2. `ece_avg > 0.15` ⇒ `MISCALIBRATED (CRITICAL)`
3. `ece_avg > 0.08` ⇒ `MISCALIBRATED`
4. `skill_score < -0.10` **y** `brier_trend_slope > 0` ⇒ `DEGRADING`
5. `sensor_signal > 1.10` **y** `sensor_n >= MIN_SAMPLES` ⇒ `STALE`
6. `skill_score < -0.05` ⇒ `UNDERPERFORMING`
7. else ⇒ `HEALTHY`

## Output (plantilla)

```markdown
# ML Diagnostic Report
Fecha: {utc_now} | Modelo: {MODEL_VERSION} | Ventana: {WINDOW_DAYS}d | N resueltos: {n_resueltos}

## 0. Schema check
{tabla resumen OK/missing + skips}

## 1. Calibración (ECE por clase)
ECE promedio: {ece_avg}
| Clase | ECE | Peor bin | Pred avg | Actual avg | N |
|------|-----|----------|----------|------------|---|
| home | ... | ... | ... | ... | ... |
| draw | ... | ... | ... | ... | ... |
| away | ... | ... | ... | ... | ... |

## 2. Skill vs Mercado (frozen_odds_*)
Brier model: {brier_model} | Brier market: {brier_market} | Skill: {skill_score}
Cobertura frozen_odds: {n_with_frozen_odds}/{n_total_resolved} ({frozen_odds_coverage_pct}%)

## 3. Evolución temporal
slope: {brier_trend_slope} | tendencia: {ascendente|plana|descendente}

## 4. Feature coverage (stats)
{top ligas con coverage baja}

## 5. Sensor B
signal: {signal_score} | n: {sensor_n}

## 6. Prediction coverage
{top ligas con rate bajo}

## Diagnóstico general
Estado: {INSUFFICIENT_DATA|MISCALIBRATED|DEGRADING|STALE|UNDERPERFORMING|HEALTHY}
Evidencia:
- ...
- ...
Acción recomendada: ...
```
