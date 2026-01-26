# Ops Runbook (FutbolStats) — Monitoreo Diario + Guardrails

Este documento es una guía **operativa** (5 min/día) para verificar que el backend está sano, capturando PIT correctamente y que las ligas críticas (incl. CONMEBOL) están en el radar.

## Accesos (Dashboards)

Los dashboards están protegidos por `DASHBOARD_TOKEN`:

- **HTML (recomendado)**:
  - `/dashboard/ops`
  - `/dashboard/pit`
- **JSON**:
  - `/dashboard/ops.json`
  - `/dashboard/pit.json`

**Auth**:
- Header: `X-Dashboard-Token: <token>` (OBLIGATORIO en producción)

> **IMPORTANTE**: En producción, los query params `?token=` están **deshabilitados** por seguridad.

## Autenticación de Endpoints (Resumen)

| Endpoint | Header | Env Var |
|----------|--------|---------|
| `/dashboard/*` | `X-Dashboard-Token` | `DASHBOARD_TOKEN` |
| `/metrics` (Prometheus) | `Authorization: Bearer <token>` | `METRICS_BEARER_TOKEN` |
| `/predictions`, `/model/*` | `X-API-Key` | `API_KEY` |

### Ejemplos curl (producción)
```bash
# Dashboard ops.json
curl -s -H "X-Dashboard-Token: $DASHBOARD_TOKEN" "https://web-production-f2de9.up.railway.app/dashboard/ops.json"

# Prometheus metrics (Grafana Cloud)
curl -s -H "Authorization: Bearer $METRICS_BEARER_TOKEN" "https://web-production-f2de9.up.railway.app/metrics"

# API predictions
curl -s -H "X-API-Key: $API_KEY" "https://web-production-f2de9.up.railway.app/predictions"
```

## Config clave (Railway)

### `LEAGUE_MODE`

Define qué ligas se consideran en `global_sync_today` + `lineup_monitoring_*` + movement tracking:

- `tracked` (**default**): todas las ligas presentes en DB (`SELECT DISTINCT league_id FROM matches`)
- `extended`: lista extendida en `app/scheduler.py` (**incluye 11 y 13**)
- `top5`: solo Top5

**Recomendación operativa (temporaria):**
- Mantener `LEAGUE_MODE=extended` hasta que haya **upcoming fixtures > 0** en `11/13` para temporada actual.
- Luego volver a `tracked` cuando ya estés cómodo con cobertura y volumen.

## Checklist diario (2–5 min)

### 1) Salud general (Ops Dashboard)

En `/dashboard/ops` verificar:
- **PIT live (60 min)**: >0 cuando hay fútbol con lineups disponibles
- **ΔKO** (min_to_ko) razonable: típicamente 10–90 min; ideal 45–75
- **API Budget**: sin señales de “exhausted”
- **Movimiento 24h**: contadores no nulos en días con partidos
- **Stats FT 72h**: `missing` no debería crecer sin razón

### 2) PIT (lo crítico del negocio)

En `/dashboard/ops` o SQL:
- Capturas `odds_snapshots.snapshot_type='lineup_confirmed'`
- `odds_freshness='live'`
- Distribución de `delta_to_kickoff_seconds`

### 3) CONMEBOL (11/13) — cuando empiece temporada

Esperable:
- `upcoming fixtures 24h` para ligas 11 y 13 > 0 (cuando haya calendario del día)
- PIT comenzará a aparecer en esas ligas cuando haya lineups confirmados

## Queries SQL (read-only) — auditoría rápida

> Ejecutar en Railway Postgres (read-only). Guardar output si se requiere evidencia.

### A) PIT live última hora (conteo)

```sql
SELECT COUNT(*)
FROM odds_snapshots
WHERE snapshot_type='lineup_confirmed'
  AND odds_freshness='live'
  AND snapshot_at > NOW() - INTERVAL '60 minutes';
```

### B) ΔKO PIT live (última hora)

```sql
SELECT ROUND(delta_to_kickoff_seconds/60.0) AS min_to_ko, COUNT(*)
FROM odds_snapshots
WHERE snapshot_type='lineup_confirmed'
  AND odds_freshness='live'
  AND snapshot_at > NOW() - INTERVAL '60 minutes'
  AND delta_to_kickoff_seconds IS NOT NULL
GROUP BY 1
ORDER BY 1;
```

### C) Upcoming fixtures (24h) por liga (incluye 11/13)

```sql
SELECT league_id, COUNT(*) AS upcoming_24h
FROM matches
WHERE date >= NOW()
  AND date < NOW() + INTERVAL '24 hours'
  AND league_id IN (11,13)
GROUP BY league_id
ORDER BY league_id;
```

### D) Health stats backfill (FT 72h)

```sql
SELECT
  COUNT(*) FILTER (WHERE stats IS NOT NULL AND stats::text != '{}' AND stats::text != 'null') AS with_stats,
  COUNT(*) FILTER (WHERE stats IS NULL OR stats::text = '{}' OR stats::text = 'null') AS missing_stats
FROM matches
WHERE status IN ('FT','AET','PEN')
  AND date > NOW() - INTERVAL '72 hours';
```

### E) Guardrail anti-contaminación (Pack2 ejemplo)

```sql
-- Pack2 league_ids: 265,239,13,281,242,268,270,11,250,252,344,299
SELECT
  (SELECT COUNT(*) FROM odds_snapshots WHERE match_id IN (
     SELECT id FROM matches WHERE league_id IN (265,239,13,281,242,268,270,11,250,252,344,299)
  )) AS odds_snapshots_pack2,
  (SELECT COUNT(*) FROM market_movement_snapshots WHERE match_id IN (
     SELECT id FROM matches WHERE league_id IN (265,239,13,281,242,268,270,11,250,252,344,299)
  )) AS market_movement_pack2,
  (SELECT COUNT(*) FROM lineup_movement_snapshots WHERE match_id IN (
     SELECT id FROM matches WHERE league_id IN (265,239,13,281,242,268,270,11,250,252,344,299)
  )) AS lineup_movement_pack2;
```

## “Qué hacer si algo falla”

- **PIT=0 en días con partidos**:
  - revisar logs Railway: jobs `lineup_monitoring_*` y `global_sync_today`
  - revisar si hay `429` o `budget exceeded`
  - confirmar que hay `matches` con `status='NS'` y kickoff próximo
- **429 / budget exceeded**:
  - bajar volumen (throttles env) o dejar `LEAGUE_MODE=tracked/top5` temporalmente
- **Upcoming 11/13 sigue en 0**:
  - puede ser normal si aún no hay fixtures del día para temporada vigente; revisar API/fixtures del día

## iOS App: Configuración del Token

La app iOS requiere el `DASHBOARD_TOKEN` para acceder a los endpoints protegidos.

### Para desarrollo local (Xcode)

**Opción 1 - UserDefaults (recomendada para testing rápido):**
```bash
# En terminal con el simulador corriendo:
xcrun simctl spawn booted defaults write com.futbolstats.app dashboard_token 'tu_token_aquí'

# Para iPhone físico conectado, usar Xcode:
# Product > Scheme > Edit Scheme > Run > Arguments > Environment Variables
# Agregar: dashboard_token = tu_token_aquí
```

**Opción 2 - Info.plist (recomendada para builds persistentes):**
1. En Xcode, abrir `Info.plist`
2. Agregar key: `DASHBOARD_TOKEN`
3. Tipo: String
4. Valor: tu token

**Prioridad de lectura** (en `AppConfiguration.swift`):
1. UserDefaults `dashboard_token` (override para dev)
2. Info.plist `DASHBOARD_TOKEN`

**Importante**: Nunca commitear tokens al repositorio. Usar `.gitignore` para configs locales.

## Telemetría Shadow Mode + Sensor B

### Métricas Prometheus (`/metrics`)

```
# Shadow Mode (A/B Testing)
shadow_predictions_logged_total      # Counter: predicciones shadow registradas
shadow_predictions_evaluated_total   # Counter: predicciones evaluadas vs FT
shadow_predictions_errors_total      # Counter: errores en logging
shadow_eval_lag_minutes              # Gauge: minutos desde oldest pending FT
shadow_pending_ft_to_evaluate        # Gauge: partidos FT con evaluaciones pendientes

# Sensor B (Calibration Diagnostics)
sensor_predictions_logged_total      # Counter: predicciones sensor registradas
sensor_predictions_evaluated_total   # Counter: predicciones evaluadas vs FT
sensor_predictions_errors_total      # Counter: errores en logging
sensor_retrain_runs_total{status}    # Counter: retrains (ok/learning/error)
sensor_eval_lag_minutes              # Gauge: minutos desde oldest pending FT
sensor_pending_ft_to_evaluate        # Gauge: partidos FT con evaluaciones pendientes
sensor_state                         # Gauge: 0=disabled, 1=learning, 2=ready, 3=error
```

### Health Blocks en ops.json

```json
{
  "shadow_mode": {
    "health": {
      "pending_ft_to_evaluate": 0,
      "eval_lag_minutes": 0.0,
      "stale_threshold_minutes": 120,
      "is_stale": false
    }
  },
  "sensor_b": {
    "health": { /* misma estructura */ }
  }
}
```

### Semántica de `pending_ft_to_evaluate`

**IMPORTANTE**: Esta métrica cuenta **partidos FT/AET/PEN** que tienen predicciones shadow/sensor sin evaluar.
- NO cuenta predicciones para partidos NS (futuros)
- Detecta "silent failures": cuando el job de evaluación no procesa partidos terminados
- Alerta cuando `is_stale=true` (eval_lag > threshold)

### Config env vars

```
SHADOW_EVAL_STALE_MINUTES=120  # Alerta si oldest pending > esto (default 120)
SENSOR_EVAL_STALE_MINUTES=120  # Alerta si oldest pending > esto (default 120)
```

### Troubleshooting

**`pending_ft_to_evaluate > 0` y `is_stale=true`**:
1. Verificar logs Railway: `railway logs -n 50 --filter "shadow"` o `--filter "sensor"`
2. Buscar errores en jobs de evaluación (`evaluate_shadow_predictions`, `evaluate_sensor_predictions_job`)
3. Verificar que los partidos FT tienen `home_goals` y `away_goals` no-null

**Counters en 0 después de mucho tiempo**:
- Normal si no hay partidos FT recientes con predicciones shadow/sensor
- Verificar que `MODEL_SHADOW_ARCHITECTURE=two_stage` está configurado
- Verificar que `SENSOR_ENABLED=true`

---

## Release Readiness Checklist

Verificar antes de cada deploy a producción:

### 1. ML Model
```bash
# Verificar modelo blessed está commiteado
ls -la models/xgb_v*.json

# Post-deploy: verificar en OPS
curl -s -H "X-Dashboard-Token: $DASHBOARD_TOKEN" \
  "https://web-production-f2de9.up.railway.app/dashboard/ops.json" | jq '.data.ml_model'
```
- [ ] `loaded: true`
- [ ] `version` coincide con `MODEL_VERSION` en config
- [ ] `n_features` coincide con features esperados (17 para v1.0.0)

### 2. Jobs Health
```bash
curl -s -H "X-Dashboard-Token: $DASHBOARD_TOKEN" \
  "https://web-production-f2de9.up.railway.app/dashboard/ops.json" | jq '.data.jobs_health'
```
- [ ] `status: "ok"` (no warn/red)
- [ ] `stats_backfill.status: "ok"`
- [ ] `odds_sync.status: "ok"`
- [ ] `fastpath.status: "ok"`

### 3. Sentry
```bash
sentry-cli issues list --org devseqio --project python-fastapi -s unresolved
```
- [ ] 0 issues unresolved nuevos post-deploy
- [ ] Si hay issues, evaluar si son críticos o transitorios

### 4. API Budget
```bash
curl -s -H "X-Dashboard-Token: $DASHBOARD_TOKEN" \
  "https://web-production-f2de9.up.railway.app/dashboard/ops.json" | jq '.data.budget'
```
- [ ] `status: "ok"`
- [ ] `requests_remaining` > 1000 (buffer seguro)

### 5. Quick Smoke Test
```bash
# Health endpoint
curl -s "https://web-production-f2de9.up.railway.app/health" | jq

# Predictions (requiere API key)
curl -s -H "X-API-Key: $API_KEY" \
  "https://web-production-f2de9.up.railway.app/predictions?days_ahead=1" | jq '.total'
```
- [ ] `/health` retorna 200
- [ ] `/predictions` retorna datos

### Notas sobre Modelos

**Solo commitear modelos "blessed":**
- Nomenclatura: `xgb_v{VERSION}_{YYYYMMDD}.json`
- Ignorados en `.gitignore`: `*_experimental*.json`, `*_temp*.json`
- Un solo modelo por versión major (el más reciente)

**Feature compatibility:**
- El engine selecciona features dinámicamente según `model.n_features_in_`
- Si el modelo espera 14 y runtime produce 17, se truncan automáticamente
- OPS muestra `n_features` para diagnóstico rápido

---

## Alerts Bell (Grafana Webhook → Dashboard)

El dashboard `/dashboard/ops` incluye un icono de campana (🔔) que muestra alertas provenientes de Grafana Alerting.

### Arquitectura
```
Grafana Alerting Rules
        ↓
  Contact Point (Webhook)
        ↓
POST /dashboard/ops/alerts/webhook
        ↓
      ops_alerts (PostgreSQL)
        ↓
GET /dashboard/ops/alerts.json
        ↓
   Bell + Toast (UI polling 20s)
```

### Configuración Grafana

1. **Crear Contact Point (Webhook)**:
   - Type: `webhook`
   - URL: `https://web-production-f2de9.up.railway.app/dashboard/ops/alerts/webhook`
   - HTTP Method: `POST`
   - HTTP Headers:
     - `X-Alerts-Secret: <ALERTS_WEBHOOK_SECRET>`

2. **Crear Alert Rules** (ejemplos):
   ```
   # Shadow stale
   shadow_eval_lag_minutes > 120
   severity: critical

   # Sensor stale
   sensor_eval_lag_minutes > 120
   severity: critical

   # Jobs failing
   job_last_success_minutes{job="fastpath"} > 10
   severity: warning
   ```

3. **Asignar Contact Point a Notification Policy**

### Env Vars (Railway)
```bash
# Secreto para webhook (NO reutilizar DASHBOARD_TOKEN)
ALERTS_WEBHOOK_SECRET=<generar-secret-aleatorio>
```

### Endpoints

| Endpoint | Method | Auth | Descripción |
|----------|--------|------|-------------|
| `/dashboard/ops/alerts/webhook` | POST | `X-Alerts-Secret` | Ingesta desde Grafana |
| `/dashboard/ops/alerts.json` | GET | `X-Dashboard-Token` | Lista alertas (bell dropdown) |
| `/dashboard/ops/alerts/ack` | POST | `X-Dashboard-Token` | Marcar leídas |

### Comportamiento UI

- **Badge**: Muestra conteo de alertas firing + unread
- **Dropdown**: Lista alertas recientes con link a Grafana
- **Toast**: Solo para alertas `severity: critical` + `status: firing` (una vez por alerta)
- **Polling**: Cada 20 segundos

### Test Manual (curl)
```bash
# Simular alerta desde Grafana
curl -X POST "https://web-production-f2de9.up.railway.app/dashboard/ops/alerts/webhook" \
  -H "X-Alerts-Secret: $ALERTS_WEBHOOK_SECRET" \
  -H "Content-Type: application/json" \
  -d '{
    "alerts": [{
      "status": "firing",
      "labels": {"alertname": "ShadowStale", "severity": "critical"},
      "annotations": {"summary": "Shadow eval lag > 2h", "description": "Shadow predictions pending > 120 min"},
      "startsAt": "2026-01-24T12:00:00Z",
      "fingerprint": "abc123"
    }]
  }'

# Verificar
curl -s -H "X-Dashboard-Token: $DASHBOARD_TOKEN" \
  "https://web-production-f2de9.up.railway.app/dashboard/ops/alerts.json" | jq
```

---

## TITAN: Reglas de Timezone (PostgreSQL + asyncpg)

El sistema TITAN usa dos esquemas con diferentes tipos de timestamp:

| Schema | Tipo Columna | Ejemplo | Formato Python |
|--------|--------------|---------|----------------|
| `public.*` | `TIMESTAMP` (naive) | `matches.date` | `datetime` sin tzinfo |
| `titan.*` | `TIMESTAMPTZ` (aware) | `feature_matrix.kickoff_utc` | `datetime` con `tzinfo=timezone.utc` |

### Regla General

```python
# Para queries contra public.* (TIMESTAMP naive)
kickoff_naive = kickoff_utc.replace(tzinfo=None) if kickoff_utc.tzinfo else kickoff_utc

# Para queries contra titan.* (TIMESTAMPTZ aware)
kickoff_aware = kickoff_utc if kickoff_utc.tzinfo else kickoff_utc.replace(tzinfo=timezone.utc)
```

### Error Típico (asyncpg)

```
asyncpg.exceptions.DataError: invalid input for query argument $2:
datetime.datetime(2026, 01, 26, 15, 20, tzinfo=datetime.timezone.utc)
(can't subtract offset-naive and offset-aware datetimes)
```

**Causa**: Pasar `datetime` aware a una columna `TIMESTAMP` naive (o viceversa).

**Fix**: Normalizar el datetime según el tipo de columna destino.

### Archivos Afectados

| Archivo | Columna | Tipo | Acción |
|---------|---------|------|--------|
| `app/titan/materializers/feature_matrix.py` | `public.matches.date` | TIMESTAMP | Strip tzinfo |
| `app/titan/runner.py` | `public.matches.date` | TIMESTAMP | Usar naive |
| `app/titan/jobs/job_manager.py` | `titan.raw_extractions.*` | TIMESTAMPTZ | Usar aware UTC |

### SQL Syntax (asyncpg)

asyncpg no soporta `::type` después de placeholders. Usar `CAST()`:

```sql
-- ❌ Error con asyncpg
INSERT INTO tabla (col) VALUES (:value::jsonb)

-- ✅ Correcto
INSERT INTO tabla (col) VALUES (CAST(:value AS jsonb))
```

### Verificación

```sql
-- Columnas TIMESTAMP (naive) en public
SELECT column_name, data_type
FROM information_schema.columns
WHERE table_schema = 'public' AND data_type = 'timestamp without time zone';

-- Columnas TIMESTAMPTZ (aware) en titan
SELECT column_name, data_type
FROM information_schema.columns
WHERE table_schema = 'titan' AND data_type = 'timestamp with time zone';
```
