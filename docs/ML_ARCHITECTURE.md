# ML Architecture - FutbolStats

Documentación técnica del sistema de predicciones ML, sus guardrails y mecanismos de evaluación/shadow.

## Estado actual (Feb 2026)

- **Producción**: Model A `v1.0.1-league-only` (XGBoost baseline, 14 features league-only) + kill-switch + policy draw cap.
- **Shadow two-stage**: `v1.1.0-twostage` **ACTIVO** con reentrenamiento automático bi-semanal (ATI-aprobado 2026-02-09). Mejor accuracy interna (Brier 0.2094). No sirve predicciones públicas.
- **Sensor B**: diagnóstico de calibración (LogReg L2) con guardrails de estabilidad numérica + sanity en OPS.
- **ext-C (Automatic Shadow)**: job automático que genera predicciones `v1.0.2-ext-C` en `predictions_experiments` (no servido).

---

## Model A (producción) - v1.0.1-league-only

### Configuración

- **Algoritmo**: XGBoost (baseline)
- **Output**: Probabilidades 1X2 (home/draw/away)
- **Artefacto**: `models/xgb_v1.0.1-league-only_*.json`
- **Flags**:
  - `MODEL_ARCHITECTURE=baseline`
  - `MODEL_VERSION=v1.0.1-league-only`

### Features (14)

Features baseline (misma lista usada por Sensor B para comparación justa):

- `home_goals_scored_avg`
- `home_goals_conceded_avg`
- `home_shots_avg`
- `home_corners_avg`
- `home_rest_days`
- `home_matches_played`
- `away_goals_scored_avg`
- `away_goals_conceded_avg`
- `away_shots_avg`
- `away_corners_avg`
- `away_rest_days`
- `away_matches_played`
- `goal_diff_avg`
- `rest_diff`

### Draw label canónico

**Draw = home_goals == away_goals** donde el score almacenado representa:

- **FT (Full Time)**: resultado a los 90 minutos
- **PEN (Penalties)**: score ANTES de penales (empate a 90'/120')
- **AET (After Extra Time)**: score DESPUÉS de prórroga

### Dataset de entrenamiento (league-only + PIT-safe)

- **Solo** partidos con `status = 'FT'` (excluye AET/PEN por dinámica distinta).
- **League-only para features**: rolling averages calculadas usando únicamente matches donde `admin_leagues.kind = 'league'`.
- **PIT-safe**: para un match objetivo con kickoff \(t\), el history usa solo partidos con `match.date < t` (sin mirar el futuro).
- **Rolling window**: últimos 10 partidos de liga (por equipo) previos al match (ventana configurable en scripts).

---

## Guardrails Fase 0/1 (anti-Exeter + robustez operativa)

### 1) Feature engineering league-only (anti-Exeter)

Motivación: evitar el “Modo Exeter” (stats infladas por copas vs rivales amateurs) contaminando rolling averages.

Implementación: el `FeatureEngineer` calcula features para upcoming matches con `league_only=True` (history solo de liga), aunque el match a predecir sea copa/internacional.

### 2) Kill-switch router (historial mínimo)

Objetivo: no predecir cuando las features son inestables por falta de historial.

Criterio: ambos equipos deben tener al menos **N partidos de liga** en los últimos **LOOKBACK_DAYS** previos al kickoff:

- `KILLSWITCH_ENABLED` (default true)
- `KILLSWITCH_MIN_LEAGUE_MATCHES` (default 5)
- `KILLSWITCH_LOOKBACK_DAYS` (default 90)

Telemetría:
- `predictions_killswitch_filtered_total{reason}` (`home_insufficient|away_insufficient|both_insufficient`)
- `predictions_killswitch_eligible`

---

## Betting policy (value bets) - Draw cap (portfolio guardrail)

Motivación: el modelo mejoró en detección de empates, pero el selector “best_edge” puede concentrar demasiado el portfolio en draws.

El cap **no cambia** probabilidades del modelo; solo filtra/remueve apuestas “draw” marginales en la selección de value bets.

Flags:

```
POLICY_DRAW_CAP_ENABLED=true
POLICY_MAX_DRAW_SHARE=0.35
POLICY_EDGE_THRESHOLD=0.05
```

Regla: si los value bets de outcome=draw superan `POLICY_MAX_DRAW_SHARE`, se mantienen solo los draws con mayor edge.

---

## Betting policy - Market Anchor (low-signal leagues)

### Motivación

El diagnóstico ABE de Argentina (league_id=128, Feb 2026) demostró que el modelo XGBoost v1.0.1 no tiene señal predictiva para esa liga:

| Métrica | Modelo | Mercado | Naive |
|---------|--------|---------|-------|
| Brier   | 0.6625 | 0.6348  | 0.6545|

El modelo es **peor que naive** en Argentina (skill negativo). El backtest confirmó que en 14/19 ligas evaluadas, el mercado domina al modelo (α* ≥ 0.60).

### Mecanismo

Para ligas explícitamente listadas en `LEAGUE_OVERRIDES`, las probabilidades servidas se blendean con probabilidades de mercado de-viggeadas:

```
p_served = (1 - α) × p_model + α × p_market
```

Donde `p_market` se obtiene aplicando de-vig proporcional (normalizar 1/odds) a las cuotas Bet365.

Con α=1.0 (Argentina), se sirven **probabilidades de mercado puras** en lugar del modelo.

### Ligas afectadas

| Liga | league_id | α | Efecto | Evidencia |
|------|-----------|---|--------|-----------|
| Argentina Primera División | 128 | 1.0 | Mercado puro | Brier 0.6348 vs 0.6625 (modelo) |

**Ninguna otra liga se ve afectada.** El `ALPHA_DEFAULT=0.0` garantiza que sin override explícito, el modelo se sirve sin modificar.

### Configuración (env vars)

```
MARKET_ANCHOR_ENABLED=false          # Feature flag (activar con true)
MARKET_ANCHOR_ALPHA_DEFAULT=0.0      # Solo overrides aplican (no global)
MARKET_ANCHOR_LEAGUE_OVERRIDES=128:1.0  # "league_id:alpha" separado por comas
MARKET_ANCHOR_MIN_SAMPLES=200        # Min FT con odds antes de aceptar α por liga
```

Para agregar otra liga: `MARKET_ANCHOR_LEAGUE_OVERRIDES=128:1.0,239:0.8`

### Guardrails (ABE P0)

1. **Scope (P0-1)**: Solo ligas en `LEAGUE_OVERRIDES` se anclan. α_default=0.0 previene anclaje global accidental.
2. **NS-only (P0-2)**: Solo predicciones `status=NS` no-frozen. LIVE/FT/frozen nunca se tocan.
3. **Odds validation (P0-3)**: Todas las cuotas (home/draw/away) deben ser > 1.0 antes del de-vig.
4. **Value bets (P0-4)**: Cuando α ≥ 0.80, se limpian value_bets (probs ≈ mercado → sin edge) y se agrega warning `MARKET_ANCHORED`.

### Pipeline (orden de ejecución)

1. `engine.predict()` → probabilidades crudas del modelo
2. `_overlay_frozen_predictions()` → partidos FT
3. `_overlay_rerun_predictions()` → rerun serving
4. `_apply_team_overrides()` → rebranding
5. `apply_draw_cap()` → filtro de value bets
6. **`apply_market_anchor()`** → blend para ligas low-signal
7. `_save_predictions_to_db()` → guardar

### Campos preservados

- `probabilities`: contiene las probs blended (o mercado puro si α=1.0)
- `model_probabilities`: preserva las probs originales del modelo XGBoost
- `raw_probabilities`: preserva probs pre-anchor si no existían previamente
- `fair_odds`: recalculadas desde probs blended
- `policy_metadata.market_anchor`: `{applied, alpha, market_source, league_id}`

### Criterio para agregar/quitar ligas

**Para agregar** una liga al override:
1. Backtest con `scripts/experiment_market_anchor.py` (o script equivalente)
2. Si `brier_market < brier_model` con significancia estadística (bootstrap CI no cruza cero), la liga es candidata
3. Usar α* del grid search como valor del override
4. Requiere aprobación ABE

**Para quitar** una liga:
- Si el modelo mejora (v1.0.2+) y `brier_model <= brier_market`, reducir α gradualmente o eliminar override

### Limitaciones actuales

- **Odds históricas**: API-Football NO almacena odds después de FT (purge). Solo datos forward desde odds_sync (activo desde 27-Ene-2026).
- **Coverage Argentina**: ~12 días de odds acumuladas al momento de la implementación. Partidos sin odds se sirven con modelo puro (graceful skip).
- **Es una medida interina**: v1.0.2 debería entrenar el modelo como "ajuste sobre mercado" (incluir market probs como feature) + agregar Elo/xG, reduciendo la necesidad de anclaje externo.

---

## Shadow Mode (Two-Stage Architecture) - v1.1.0-twostage

### Propósito
Arquitectura two-stage para mejorar predicción de empates. Actualmente el modelo con mejor accuracy interna (Brier 0.2094), superando a Model A. **No sirve predicciones públicas** — solo inferencia interna vía `shadow_predictions`.

### Arquitectura
**Stage 1**: Binary classifier (draw vs non-draw)
- 18 features (17 base + implied_draw de odds)
- sample_weight: draws=1.2, otros=1.0

**Stage 2**: Binary classifier (home vs away) para non-draws
- 17 features (sin implied_draw)

### Composición de Probabilidades
```
p_draw = P(draw | Stage1)
p_home = (1 - p_draw) × P(home | non-draw, Stage2)
p_away = (1 - p_draw) × P(away | non-draw, Stage2)
```

### Config (env vars)
```
MODEL_ARCHITECTURE=baseline              # Activo en producción
MODEL_SHADOW_ARCHITECTURE=two_stage      # Shadow mode
MODEL_DRAW_THRESHOLD=0.0                 # Deshabilitado, usar argmax
```

### Reentrenamiento automático (ATI-aprobado 2026-02-09)

Job: `shadow_recalibration` — separado de `weekly_recalibration` (Model A). No afecta producción.

**Cadencia**: Martes 5AM UTC, con check interno de ≥14 días desde último retrain aprobado.

**Triggers**:
- **Intervalo**: ≥14 días desde último `validation_verdict='approved'`
- **Volumen**: ≥1,500 nuevos FT matches en cohort (ligas activas domésticas, tainted=false)

**Cohort** (idéntico a Model A):
- `league_only=True` (features calculadas solo con partidos de liga)
- Ligas: `admin_leagues(is_active=true, kind='league')` — 25 ligas activas
- `min_date=2023-01-01`
- `status='FT'` only

**Validación (retrain gate)**:
- `new_brier < last_shadow_brier` → aprobado
- Si snapshot previo no tiene `dataset_mode` (régimen viejo 69K) → rebaseline automático (primer run con nuevo cohort siempre aprueba)

**Deploy (shadow-only)**:
- Snapshot en `model_snapshots` con `is_active=False`, `is_baseline=False`
- Model blob serializado en DB (`save_to_bytes`)
- Hot-reload de `_shadow_engine` global sin restart (safe fallback: si falla, conserva engine anterior)

**Guardrails**:
- Cooldown: 3 rechazos consecutivos → skip 14 días
- Scheduler: `max_instances=1, coalesce=True, misfire_grace_time=300`
- `is_active=False` siempre — no usa `create_snapshot()` (que desactivaría Model A)
- Patrón 3-phase DB→CPU→DB (evita idle connection timeout Railway)
- Trazabilidad: métricas JSONB en `job_runs` + `training_config` en snapshot

**Criterios de promoción a producción** (P1 futuro, no implementado):
- N≥300 evaluaciones shadow vs market
- ΔBrier ≤ -0.005 (shadow mejor que Model A)
- CI95 < 0 (significativo)
- Guardrail market: skill vs Pinnacle ≤ 0.010

### Estado Actual
**ACTIVO** — shadow mode habilitado (`MODEL_SHADOW_ARCHITECTURE=two_stage`), reentrenamiento automático registrado. Snapshot actual: v1.1.0-twostage (Brier 0.2094, 69K muestras, régimen pre-cohort — será rebaselined en primer retrain automático).

Ver `/dashboard/ops.json` → `shadow_mode` para el estado/métricas actuales.

---

## ext-C (Automatic Shadow) - v1.0.2-ext-C

### Propósito

Evaluar si una variante entrenada con rango histórico distinto / filtros de cold-start mejora calibración/skill de forma OOS **sin tocar serving**.

### Entrenamiento

- **Algoritmo**: XGBoost baseline (mismas 14 features)
- **Artefacto**: `models/xgb_v1.0.2-ext-C_*.json`
- **Notas**:
  - ext-C se compara OOS contra `v1.0.1-league-only` antes de considerar un switch.
  - Es un candidato “shadow”: genera predicciones, pero **no se sirve** al usuario.

### Shadow automático (scheduler)

Job: `extc_shadow` (cada `EXTC_SHADOW_INTERVAL_MINUTES`), con guardrails:

- **Write target**: solo `predictions_experiments` (nunca `predictions`)
- **Idempotencia**: `ON CONFLICT (snapshot_id, model_version) DO NOTHING`
- **PIT-safe**: `created_at = snapshot_at - 1s`
- **Gating**: `snapshot_type='lineup_confirmed'` + ventana 10–90 min pre-kickoff
- **Anti-join**: selecciona snapshots pendientes con `LEFT JOIN ... IS NULL` (evita gaps)

Flags:

```
EXTC_SHADOW_ENABLED=false
EXTC_SHADOW_MODEL_VERSION=v1.0.2-ext-C
EXTC_SHADOW_MODEL_PATH=models/xgb_v1.0.2-ext-C_*.json
EXTC_SHADOW_BATCH_SIZE=200
EXTC_SHADOW_INTERVAL_MINUTES=30
EXTC_SHADOW_START_AT=2026-02-01
EXTC_SHADOW_OOS_ONLY=true
```

### Telemetría

**Prometheus (genérico ext-A/B/C/D, recomendado)**

- `ext_shadow_predictions_inserted_total{variant}` (`variant` = A|B|C|D)
- `ext_shadow_predictions_skipped_total{variant}`
- `ext_shadow_errors_total{variant}`
- `ext_shadow_last_success_timestamp{variant}`
- `ext_shadow_rejections_total{variant,reason}`
  - `reason` (observabilidad mínima, sin sobrecontar): `no_pending_snapshots | model_not_found | insert_error`
  - **Nota**: `snapshot_before_start_at` y `outside_window_*` NO se miden en el job (porque el query ya filtra).

**Prometheus (legacy ext-C)**

- `extc_shadow_predictions_inserted_total`
- `extc_shadow_predictions_skipped_total`
- `extc_shadow_errors_total`
- `extc_shadow_last_success_timestamp`

**Logs (Railway)**

- `"[EXT_SHADOW] ext_shadow_no_snapshots variant=... model_version=... start_at=..."`  
  Indica que el job corrió pero no encontró snapshots pendientes (dentro del `LIMIT` y gating PIT).
- `"[EXT_SHADOW] ext-<X> run_summary: batch_size=..., processed=..., inserted=..., skipped=..., errors=..."`  
  Resumen por variante (A/B/C/D) del batch procesado.

**Debug on-demand (dashboard, read-only)**

- `GET /dashboard/debug/experiment-gating/{match_id}?variant=A|B|C|D` (requiere `X-Dashboard-Token`)  
  Devuelve checks y `failure_reason` usando la misma lógica estricta del job:
  - `lineup_confirmed_exists`
  - `snapshot_after_start_at` (vs `EXT_SHADOW_START_AT`)
  - `window_10_90_min_strict` (estricto: \(delta > 10\) y \(delta < 90\))
  - `model_exists`
  - `has_pit_safe_prediction` (por `match_id + model_version` y `snapshot_at <= kickoff`)

**ops.json**

- `extc_shadow` (state, counts, last_success_at, predictions_count)  
  Nota: actualmente el resumen en ops.json está orientado a ext-C; para A/B/D usar Prometheus + logs.

---

## ext-D (Interim Evaluation Shadow) - v1.0.1-league-only retrained (2026-02-02)

### Propósito

Evaluar en producción (solo logging) el **modelo reentrenado** con contrato **league-only end-to-end** habilitado por el backfill de `matches.stats` (shots/corners).  
Este modelo **NO se sirve** al usuario y funciona como **modelo interino de evaluación** para decidir promoción a producción.

### Entrenamiento (Opción A aprobada ATI)

- **Algoritmo**: XGBoost baseline (mismas 14 features)
- **Artefacto**: `models/xgb_v1.0.1-league-only_20260202.json`
- **Rango temporal**:
  - `min_date = 2020-01-01`
  - `cutoff = 2026-01-15` (dataset usa `m.date < cutoff`)
  - `date_range entrenado`: 2020-01-01 → 2026-01-14
- **Filtros del dataset**:
  - `matches.status = 'FT'`
  - `home_goals/away_goals NOT NULL`
  - `tainted = false` (o NULL)
  - **Targets league-only**: `admin_leagues.kind = 'league'`
  - **History league-only**: rolling averages usan solo `admin_leagues.kind = 'league'`
- **Exclusiones por calidad de stats (ligas sin datos API-Football)**:
  - `league_id NOT IN (242, 250, 252, 268, 270, 299, 344)`
  - Motivación: estas ligas tienen alto % de stats faltantes; `COALESCE(..., 0)` introduciría ceros sistemáticos en shots/corners y degradaría el training.

### Shadow automático (scheduler)

Se ejecuta en paralelo como una variante adicional del job shadow (no reemplaza ext-A/B/C).  
Write target: solo `predictions_experiments` con `model_version` propio (p.ej. `v1.0.1-league-only-20260202`).

Guardrails:
- **Idempotencia**: `ON CONFLICT (snapshot_id, model_version) DO NOTHING`
- **PIT-safe**: `created_at = snapshot_at - 1s`
- **Gating**: `snapshot_type='lineup_confirmed'` + ventana 10–90 min pre-kickoff
- **Fail-closed por variante**: si el artefacto no existe o falla, no afecta serving

### Criterio de promoción

No se promueve con N pequeño. Requiere evaluación extendida:
- **≥ 2,000** predicciones PIT-safe **o** 2–4 semanas OOS
- Mejoras sostenidas en: **Brier, ECE, skill_vs_market**, sin degradación en ligas top (estratificado)

## Sensor B - Calibration Diagnostics

### Propósito
LogReg L2 sliding-window para detectar si el modelo de producción se volvió stale/rígido.
**SOLO DIAGNÓSTICO** - No afecta predicciones de producción.

### Arquitectura
- **Model A**: Modelo de producción (baseline)
- **Model B**: LogReg L2 simple (C=0.1, balanced classes)
- **Window**: Últimos N partidos FT (default 50)
- **Retrain**: Cada 6 horas

### Config (env vars)
```
SENSOR_ENABLED=true
SENSOR_WINDOW_SIZE=50
SENSOR_MIN_SAMPLES=50
SENSOR_RETRAIN_INTERVAL_HOURS=6
SENSOR_SIGNAL_SCORE_GO=1.1
SENSOR_SIGNAL_SCORE_NOISE=0.9
SENSOR_EVAL_WINDOW_DAYS=14
SENSOR_TEMPERATURE=2.0
SENSOR_PROB_EPS=1e-12
```

### Signal Score
```
signal = (brier_uniform - brier_B) / (brier_uniform - brier_A)
```
- **signal >= 1.1**: Sensor B mejora sobre Model A → revisar Model A
- **signal < 0.9**: Sensor B es ruido → Model A está bien calibrado
- **0.9 <= signal < 1.1**: Comparable → seguir monitoreando

### Estados
- **DISABLED**: SENSOR_ENABLED=false
- **LEARNING**: < min_samples, no reporta métricas
- **READY**: >= min_samples, reportando
- **ERROR**: Fallo en entrenamiento

### Governance
1. Sensor B es SOLO diagnóstico interno
2. NUNCA afecta predicciones de producción
3. Si signal > 1.1 consistentemente: revisar Model A manualmente
4. No tomar decisiones con < 100 samples evaluados

### Estabilidad numérica (fix 2026-02)

Sensor B aplica guardrails para evitar overconfidence extremo (probs pegadas a 0/1) y mejorar la estabilidad de métricas:

- `StandardScaler + LogisticRegression` (reduce logits extremos por escala)
- sanitización + clipping de features (anti-outliers)
- temperature scaling (`SENSOR_TEMPERATURE`)
- clipping de probabilidades (`SENSOR_PROB_EPS`)

### Sanity check (OPS)

En `ops.json` se expone `sensor_b.sanity` calculado sobre las últimas 24h de `sensor_predictions`:

- `overconfident_ratio`: % con `max_prob > 0.9999`
- `mean_entropy` y `low_entropy_ratio` (entropía < 0.25)
- `min_prob`

Estado: `HEALTHY` / `OVERCONFIDENT` (solo alerta; no bloquea).

### Semántica de b_* = NULL

Cuando un registro en `sensor_predictions` tiene `b_home_prob IS NULL`:

**Causa raíz**: El sensor estaba en estado LEARNING cuando se logueó la predicción:
- El sensor no tenía suficientes samples para entrenar (< `SENSOR_WINDOW_SIZE`)
- El sensor no estaba inicializado (cold start)
- Hubo un error en el entrenamiento del sensor

**Implicaciones**:
- `a_*` campos siempre tienen valores (Model A siempre predice)
- `b_*` campos son NULL → no hay predicción B que comparar
- `sensor_state = 'LEARNING'` indica esta condición
- Estos registros se **excluyen** del cálculo A vs B (`evaluated_with_b`)

**Métricas en ops.json**:
- `samples_evaluated`: Solo registros con B predictions (`evaluated_with_b`)
- `missing_b_evaluated`: Registros evaluados sin B predictions
- `missing_b_pending`: Registros pending sin B predictions

**Guardrails implementados (2026-01-25)**:
1. `ON CONFLICT` usa `COALESCE` para preservar B existente si nueva inserción tiene NULL
2. `retry_missing_b_predictions` solo procesa partidos NS (nunca FT/AET/PEN)
3. No se hace backfill de B para partidos terminados (evita time-travel leakage)

---

## Telemetría ML

### Métricas Prometheus (`/metrics`)
```
# Shadow Mode
shadow_predictions_logged_total
shadow_predictions_evaluated_total
shadow_predictions_errors_total
shadow_eval_lag_minutes
shadow_pending_ft_to_evaluate

# Sensor B
sensor_predictions_logged_total
sensor_predictions_evaluated_total
sensor_predictions_errors_total
sensor_retrain_runs_total{status}
sensor_state  # 0=disabled, 1=learning, 2=ready, 3=error

# Kill-switch router
predictions_killswitch_filtered_total{reason}
predictions_killswitch_eligible

# ext shadow (A/B/C/D)
ext_shadow_predictions_inserted_total{variant}
ext_shadow_predictions_skipped_total{variant}
ext_shadow_errors_total{variant}
ext_shadow_last_success_timestamp{variant}
ext_shadow_rejections_total{variant,reason}

# ext-C shadow (legacy)
extc_shadow_predictions_inserted_total
extc_shadow_predictions_skipped_total
extc_shadow_errors_total
extc_shadow_last_success_timestamp
```

### Health en ops.json
```json
{
  "shadow_mode": {
    "health": {
      "pending_ft_to_evaluate": 0,
      "eval_lag_minutes": 0.0,
      "is_stale": false
    }
  },
  "sensor_b": {
    "health": { /* misma estructura */ },
    "sanity": { /* HEALTHY|OVERCONFIDENT + ratios */ }
  },
  "extc_shadow": {
    "state": "DISABLED|ACTIVE|ERROR",
    "predictions_count": 0,
    "last_success_at": "2026-02-01T..."
  }
}
```

### Umbrales de Alerta
- `pending_ft_to_evaluate > 0` por más de 120 min → Investigar
- `eval_lag_minutes > 120` → Job de evaluación puede estar fallando

---

## Data quality: `matches.stats` (shots/corners) y backtests PIT

### Cobertura de stats (shots/corners)

Las features `*_shots_avg` y `*_corners_avg` dependen de `matches.stats`. Históricamente, muchos partidos antiguos no tienen stats y el pipeline puede imputar 0 (vía `COALESCE`), lo cual contamina el training si se entrena muy atrás en el tiempo.

Regla práctica:

- **Sin backfill**: entrenar desde el primer período con coverage consistente (p.ej. `min_date >= 2024-01-01`).
- **Con backfill**: ampliar `min_date` solo cuando el % de stats faltantes esté dentro de tolerancia.

### Backtests PIT-safe (evitar leakage temporal)

Para cualquier evaluación con `min_snapshot_date = D`:

- el modelo debe entrenarse con **cutoff ≤ D** (solo FT anteriores a D),
- y las features deben respetar **as-of** (`match.date < snapshot_at` / `match.date < kickoff`).

Esto evita “mejoras falsas” por entrenar con resultados incluidos luego en el período evaluado.

---

## Histórico de Experimentos

### FASE 0/1 (2026-02) - Exeter fix + league-only
- **v1.0.1-league-only**: Model A en producción (features league-only + kill-switch).
- **Draw cap**: guardrail de portfolio para value bets (evita sobreconcentración en draws).
- **ext-C**: shadow automático en paralelo (no servido) para evaluación OOS.
- **Two-stage shadow**: pausado mientras se evalúa ext-C.

### FASE 1 (v1.1.0) - NO-GO
- 17 features (14 base + 3 competitividad)
- sample_weight: draws=1.5, otros=1.0
- Draw predictions: 16.1% pero degradó Brier/LogLoss
- Resultado: Sweep de pesos no encontró punto aceptable

### FASE 2 (v1.1.0-twostage) - ACTIVO (retrain automático 2026-02-09)
- Arquitectura two-stage descrita arriba
- Snapshot original: 15-Ene-2026, Brier 0.2094, 69K muestras (régimen pre-cohort)
- **Reentrenamiento automático** aprobado ATI 2026-02-09: job `shadow_recalibration`, bi-semanal, cohort-matched (21K, league_only, 2023+)
- Primer retrain será rebaseline (cambio de cohort 69K→21K)
- Criterio de promoción a producción: N≥300, ΔBrier≤-0.005, CI95<0 (P1 futuro)

### Market Anchor Argentina (2026-02-08) - ACTIVO
- **Diagnóstico**: modelo sin señal en Argentina (Brier 0.6625 > naive 0.6545)
- **Backtest**: `scripts/experiment_market_anchor.py` — α*=1.0 para ARG, mercado Brier 0.6348
- **Implementación**: `apply_market_anchor()` en `app/ml/policy.py`, feature-flagged
- **Resultados**: `scripts/output/experiment_market_anchor.json`
- **Próximo paso**: v1.0.2 con market probs como feature (modelo aprende "ajuste sobre mercado")

---

## Phase 2: Asymmetry & Microstructure (2026-02-14) — SEALED

### Objetivo
Explotar asimetrías de timing entre lineup confirmation (~T-60m) y kickoff.
Infraestructura para capturar, medir y eventualmente explotar ventajas informacionales.

**Status**: Code Freeze. Shadow Mode / Data Accumulation. Ver `docs/PHASE2_ARCHITECTURE.md` y `docs/PHASE2_EVALUATION.md`.

### CLV (Closing Line Value)
Métrica post-hoc que mide si la predicción capturó valor vs el cierre de línea.

```
CLV_k = ln(odds_asof_k / odds_close_k)   para k ∈ {home, draw, away}
Positivo = obtuvimos mejor precio que el cierre
```

- **Tabla**: `prediction_clv` (prediction_id, canonical_bookmaker, prob_asof_*, prob_close_*, clv_*)
- **Baseline (N=849)**: Home -0.00522, Draw -0.00169, Away +0.00009 — modelo sangra CLV
- **Bolsillos positivos**: Serie A home +0.0176, Süper Lig away +0.0275, EPL away +0.0190
- **NO es feature** (sería leakage). Solo métrica de evaluación.

### SteamChaser (Modelo Secundario — Shadow Mode)
XGBoost binario que predice si la línea colapsará post-lineup.

```
Target: y = 1 si max(|prob_close_k - prob_T60_k|) > overround_T60 / 2
```

- **Archivo**: `app/ml/steamchaser.py`
- **Threshold**: `VIG_DIVISOR = 2` (sagrado — mandato ATI, nunca relajar)
- **Estado (2026-02-14)**: 644 pares, 10 positivos (1.55%), ACCUMULATING
- **Metrics**: PR-AUC + LogLoss (ATI mandate: NO ROC-AUC, NO accuracy para imbalanceo severo)
- **Gate**: MIN_TRAINING_SAMPLES=500, MIN_POSITIVE_SAMPLES=30
- **Evaluación**: `run_oot_evaluation()` — chronological 70/30 split, baseline = prevalence

### Event-Driven Cascade
Re-predicción post-lineup con odds frescos.

```
LINEUP_CONFIRMED(match_id)
  → validate NS + not frozen
  → get features + predict (Phase 1 model)
  → compute_talent_delta (5s timeout — steel degradation)
  → compute_line_movement (PIT-safe: captured_at ≤ asof_timestamp)
  → apply_market_anchor (fresh odds)
  → UPSERT prediction (asof = lineup_detected_at)
```

- **Event Bus**: `app/events/bus.py` — asyncio.Queue + DB source of truth
- **Handler**: `app/events/handlers.py` — cascade_handler with steel degradation
- **Sweeper**: Every 2min, FOR UPDATE SKIP LOCKED, reconciles missed lineups
- **Idempotent**: Skips if pred_asof >= lineup_detected_at

### MTV (Missing Talent Value)
Forward data collection — NOT in model yet.

- **Player ID Mapping**: 4,613 (Hungarian bipartite), avg confidence 0.926
- **PTS + VORP**: Player Talent Score with P25 bayesian prior (zero-division impossible)
- **Expected XI**: Injury-aware (filters player_injuries)
- **Talent Delta**: `mean(PTS(XI_real)) - mean(PTS(XI_expected))`
- **Xi Continuity**: Historical XI overlap (58,924 matches for backtest)

### Phase 2 Compliance
- ATI #1-4: SteamChaser, VORP P25, steel degradation 5s, Sweeper Queue ✓
- GDT #1-7: asof_timestamp, lineup_detected_at, bipartite matching, injury-aware XI, CLV 3-way, DB-backed bus, cascade optimized ✓
