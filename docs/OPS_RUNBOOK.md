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
- Header: `X-Dashboard-Token: <token>` (preferido)
- o query param: `?token=<token>` (menos seguro)

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

