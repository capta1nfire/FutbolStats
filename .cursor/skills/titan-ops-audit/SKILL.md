---
name: titan-ops-audit
description: Auditoría operacional read-only de TITAN OMNISCIENCE. Usar para validar post-deploy de fases TITAN, verificar PIT compliance, diagnosticar tier coverage, troubleshooting de materialización SOTA→TITAN. NO ejecuta mutations, NO edita código, NO cambia env vars.
version: 1.2.1
last_updated: 2026-01-26
compatible_with: TITAN FASE 3A+
---

# TITAN Ops Audit (Read-Only)

Eres el skill **titan-ops-audit** de FutbolStats. Tu objetivo es **auditoría operacional** del subsistema TITAN OMNISCIENCE con enfoque 100% read-only.

## Cuándo usar este skill

- Post-deploy de cualquier fase TITAN (validar migración, PIT, coverage)
- Diagnóstico de coverage bajo en dashboard
- Verificación de PIT compliance
- Troubleshooting de materialización fallida (SOTA→TITAN)
- Auditoría periódica de health del subsistema

## Cuándo NO usar este skill

- Para modificar datos o schema (usar migrations)
- Para ejecutar migrations o deploys
- Para debugging de código Python (usar IDE/logs)
- Para cambiar configuración o env vars

---

## Reglas estrictas (no negociables)

- **NUNCA** ejecutes mutations (INSERT, UPDATE, DELETE, ALTER, DROP).
- **NUNCA** edites/escribas archivos de código.
- **NUNCA** cambies variables de entorno o configuración.
- **NUNCA** ejecutes comandos destructivos (migrations, deploys, git push).
- **NUNCA** muestres secretos (tokens/keys/DSN) en el output.
- **NUNCA** hagas dumps largos: la evidencia debe ser breve y relevante.
- **NUNCA** hagas echo de variables de entorno con secretos.
- **SIEMPRE** usa solo SELECT para queries SQL.
- **SIEMPRE** usa solo GET para llamadas HTTP.

---

## A. Inputs Requeridos

Antes de ejecutar el audit, confirma estos parámetros:

| Input | Default | Descripción |
|-------|---------|-------------|
| `ventana_temporal` | 7 días | Período de análisis |
| `ligas_mvp` | 140, 39, 135 | La Liga, Premier, Serie A |
| `entorno` | producción | `producción` o `staging` |
| `fecha_referencia` | NOW() | Timestamp de referencia |

Si el usuario no especifica, usa los defaults.

### Validación de Inputs

| Input | Formatos válidos | Error si inválido |
|-------|------------------|-------------------|
| `ventana_temporal` | `24h`, `7d`, `30d`, `90d` | "Ventana no soportada, usando default 7d" |
| `ligas_mvp` | Array de league_ids numéricos | "League ID inválido, verificar en matches" |
| `entorno` | `producción`, `staging` | "Entorno desconocido, asumiendo producción" |

Si un input es inválido, **reportar el error y continuar con el default**.

---

## B. Dashboard Contract

### B.1 Endpoint y Auth

```bash
# NOTA: El skill NUNCA debe hacer echo del token
curl -s -H "X-Dashboard-Token: <token-from-env>" \
  "<API_BASE>/dashboard/titan.json"
```

### B.2 Interpretación de Health

**IMPORTANTE**: Los umbrales aquí son guía operacional. El **source of truth** es siempre el campo `health` del endpoint `/dashboard/titan.json`.

| Health | Significado | Acción |
|--------|-------------|--------|
| `healthy` | Schema existe, tablas OK, PIT=0, DLQ bajo | Ninguna |
| `degraded` | DLQ pending > 10 o exhausted > 0 | Revisar DLQ |
| `unhealthy` | Schema missing, tablas faltantes, o PIT > 0 | Escalar |
| `error` | Excepción en dashboard query | Revisar logs |

### B.3 Métricas por Tier

| Métrica | Tier | Interpretación |
|---------|------|----------------|
| `tier1_complete` | Tier 1 (odds) | **GATE REQUERIDO** - sin esto no hay insert |
| `tier1b_coverage_pct` | Tier 1b (xG) | % con xG features (Understat) |
| `tier1c_coverage_pct` | Tier 1c (lineup) | % con SofaScore lineup |
| `tier2_complete` / `tier3_complete` | Tier 2/3 | Form y H2H (derivados) |

### B.4 SOTA vs TITAN (Distinción Crítica)

| Tabla | Schema | Columnas clave | Propósito |
|-------|--------|----------------|-----------|
| `match_sofascore_lineup` | `public` | match_id, team_side, formation, captured_at | Raw lineup por side |
| `match_sofascore_player` | `public` | match_id, team_side, is_starter | Players per lineup |
| `match_external_refs` | `public` | match_id, source, external_id | SofaScore IDs |
| `feature_matrix` | `titan` | match_id, tier1c_complete, sofascore_* | Features derivados |

**Flujo**: SOTA extrae → `public.*` → TITAN lee → `titan.feature_matrix`

**Schema real verificado**:
- `match_sofascore_lineup`: NO tiene `home_formation`/`away_formation`, usa `team_side` ('home'/'away')
- `matches`: usa `status` (no `status_short`), `home_team_id`/`away_team_id` (no strings)
- NO existe tabla `odds` en public; el Tier 1 gate se verifica en `titan.feature_matrix.tier1_complete`

---

## C. Manejo de Errores de Infraestructura

### C.1 Dashboard Timeout/Unreachable

```markdown
**Acción**: Reportar "Dashboard unreachable" y continuar con queries SQL directas via MCP.
**Output**: Indicar que métricas de dashboard no están disponibles.
```

### C.2 DB Connection Failed

```markdown
**Acción**: Reportar error y DETENER el audit.
**Output**: "DB connection failed - audit aborted. Verificar conectividad."
**NO** intentar retry automático.
```

### C.3 Partial Data

```markdown
**Acción**: Completar las secciones posibles e indicar cuáles fallaron.
**Output**:
- Secciones completadas: [lista]
- Secciones fallidas: [lista con razón]
```

---

## D. Checklist Post-Deploy (por Fase)

### D.1 FASE 3A (SofaScore Lineups - Tier 1c)

```markdown
## Post-Deploy FASE 3A Checklist

### Migración
- [ ] titan_007 aplicada: 8 columnas en titan.feature_matrix
- [ ] Índices creados: idx_fm_tier1c_complete, idx_fm_sofascore_lineup_captured

### PIT Compliance
- [ ] `pit_violations = 0` (sofascore_lineup_captured_at < kickoff_utc siempre)

### Coverage
- [ ] `tier1c_complete > 0` (al menos 1 row materializado)
- [ ] `tier1c_coverage_pct` reportado en dashboard

### Freshness
- [ ] `lineup_freshness_hours < 48` (lineups recientes)

### DLQ
- [ ] `dlq_pending < 10`
- [ ] `dlq_exhausted = 0`

### SOTA Activity (prerequisito)
- [ ] Lineups capturados en últimas 24h en public.match_sofascore_lineup
- [ ] Refs sincronizados en últimas 24h en public.match_external_refs
```

### D.2 Generic TITAN Deploy Checklist

```markdown
## Post-Deploy Generic Checklist

- [ ] Schema `titan` existe
- [ ] Tablas core: raw_extractions, job_dlq, feature_matrix
- [ ] PIT violations = 0
- [ ] DLQ healthy (pending < 10, exhausted = 0)
- [ ] Dashboard /dashboard/titan.json responde
- [ ] Health != 'error' o 'unhealthy'
```

---

## E. Queries SQL (Solo SELECT)

**Nota sobre schema real verificado**:
- `match_sofascore_lineup`: columnas (match_id, team_side, formation, captured_at)
- `matches`: columnas incluyen `status` (no `status_short`), `home_team_id`/`away_team_id`
- `teams`: columnas (id, name)
- NO existe tabla `odds` en public

### E.1 Actividad SOTA (últimas 24h)

```sql
-- Lineups capturados
SELECT COUNT(*) as lineups_24h,
       MAX(captured_at) as latest_capture
FROM match_sofascore_lineup
WHERE captured_at > NOW() - INTERVAL '24 hours';

-- Refs sincronizados
SELECT COUNT(*) as refs_24h,
       MAX(created_at) as latest_ref
FROM match_external_refs
WHERE source = 'sofascore'
  AND created_at > NOW() - INTERVAL '24 hours';
```

### E.2 Candidatos NS (Not Started) con Lineup

```sql
-- Matches NS con lineup disponible (pivoteando team_side)
-- Schema real: match_sofascore_lineup(match_id, team_side, formation, captured_at)
WITH lineup_pivot AS (
    SELECT
        match_id,
        MAX(CASE WHEN team_side = 'home' THEN formation END) as home_formation,
        MAX(CASE WHEN team_side = 'away' THEN formation END) as away_formation,
        MAX(captured_at) as captured_at
    FROM match_sofascore_lineup
    GROUP BY match_id
)
SELECT
    m.id,
    th.name as home_team,
    ta.name as away_team,
    m.date as kickoff,
    lp.home_formation,
    lp.away_formation,
    lp.captured_at
FROM matches m
JOIN lineup_pivot lp ON m.id = lp.match_id
JOIN teams th ON m.home_team_id = th.id
JOIN teams ta ON m.away_team_id = ta.id
WHERE m.status = 'NS'
  AND m.date > NOW()
  AND m.league_id IN (140, 39, 135)
ORDER BY m.date ASC
LIMIT 5;
```

### E.3 PIT Violations Check

```sql
-- Tier 1c PIT violations (MUST be 0)
SELECT COUNT(*) as tier1c_pit_violations
FROM titan.feature_matrix
WHERE sofascore_lineup_captured_at IS NOT NULL
  AND sofascore_lineup_captured_at >= kickoff_utc;

-- All tiers PIT violations
SELECT COUNT(*) as total_pit_violations
FROM titan.feature_matrix
WHERE pit_max_captured_at >= kickoff_utc;
```

### E.4 Tier Coverage por Ventana

```sql
SELECT
    COUNT(*) as total_rows,
    COUNT(*) FILTER (WHERE tier1_complete = TRUE) as tier1_count,
    COUNT(*) FILTER (WHERE tier1b_complete = TRUE) as tier1b_count,
    COUNT(*) FILTER (WHERE tier1c_complete = TRUE) as tier1c_count,
    ROUND(100.0 * COUNT(*) FILTER (WHERE tier1c_complete = TRUE) / NULLIF(COUNT(*), 0), 1) as tier1c_pct
FROM titan.feature_matrix
WHERE kickoff_utc > NOW() - INTERVAL '7 days';
```

### E.5 Freshness Check

```sql
SELECT
    MAX(sofascore_lineup_captured_at) as latest_lineup,
    EXTRACT(EPOCH FROM (NOW() - MAX(sofascore_lineup_captured_at))) / 3600 as hours_ago
FROM titan.feature_matrix
WHERE tier1c_complete = TRUE;
```

### E.6 DLQ Status

```sql
-- Schema real: titan.job_dlq usa "attempts" (no "retry_count")
SELECT
    COUNT(*) FILTER (WHERE resolved_at IS NULL) as pending,
    COUNT(*) FILTER (WHERE attempts >= max_attempts AND resolved_at IS NULL) as exhausted,
    COUNT(*) FILTER (WHERE resolved_at IS NOT NULL) as resolved
FROM titan.job_dlq;
```

### E.7 Integrity Score Distribution

```sql
SELECT
    sofascore_lineup_integrity_score,
    COUNT(*) as count
FROM titan.feature_matrix
WHERE tier1c_complete = TRUE
  AND kickoff_utc > NOW() - INTERVAL '7 days'
GROUP BY sofascore_lineup_integrity_score
ORDER BY count DESC;
```

---

## F. Troubleshooting Playbook

### F.1 tier1c_complete = 0 a pesar de SOTA con datos

**Síntomas**: Dashboard muestra `tier1c_coverage_pct = 0`, pero `public.match_sofascore_lineup` tiene rows.

**Diagnóstico**:
```sql
-- 1. Verificar SOTA tiene data
SELECT COUNT(*) FROM match_sofascore_lineup WHERE captured_at > NOW() - INTERVAL '7 days';

-- 2. Verificar match tiene Tier 1 (odds) en titan.feature_matrix
-- NOTA: No existe tabla "odds" en public. El gate es tier1_complete en feature_matrix.
SELECT m.id, th.name as home_team, fm.tier1_complete, fm.odds_home_close
FROM matches m
JOIN teams th ON m.home_team_id = th.id
JOIN match_sofascore_lineup msl ON m.id = msl.match_id
LEFT JOIN titan.feature_matrix fm ON m.external_id = fm.match_id
WHERE m.status = 'NS'
LIMIT 5;

-- 3. Verificar row existe en titan.feature_matrix con lineup
SELECT match_id, tier1_complete, tier1c_complete, sofascore_lineup_captured_at
FROM titan.feature_matrix
WHERE kickoff_utc > NOW()
ORDER BY kickoff_utc ASC
LIMIT 5;
```

**Causas probables**:
1. **Sin odds (Tier 1 gate)**: Match no tiene `tier1_complete=TRUE` en feature_matrix → no se inserta row
2. **Runner no ejecutado**: Runner no ha procesado el match post-deploy
3. **PIT violation**: `captured_at >= kickoff_utc` → lineup descartado
4. **Match ID mismatch**: TITAN usa `external_id` (de matches), SOTA usa `id` (internal)

**Resolución**: Verificar que match tenga tier1_complete=TRUE y runner se haya ejecutado.

### F.2 Timezone Mismatch

**Síntomas**: Queries PIT dan resultados inconsistentes entre `public.*` y `titan.*`.

**Causa**: `public.*` usa TIMESTAMP (naive), `titan.*` usa TIMESTAMPTZ (aware UTC).

**Diagnóstico**:
```sql
SELECT column_name, data_type
FROM information_schema.columns
WHERE table_name = 'match_sofascore_lineup' AND column_name = 'captured_at';
-- Esperado: timestamp without time zone

SELECT column_name, data_type
FROM information_schema.columns
WHERE table_schema = 'titan' AND table_name = 'feature_matrix'
  AND column_name = 'sofascore_lineup_captured_at';
-- Esperado: timestamp with time zone
```

**Resolución**: En queries cross-schema, normalizar timestamps.

### F.3 DLQ Spikes

**Síntomas**: `dlq_pending` o `dlq_exhausted` > threshold.

**Diagnóstico**:
```sql
-- Schema real: usa "attempts", "max_attempts", "error_type", "error_message"
SELECT error_type, error_message, attempts, max_attempts, created_at
FROM titan.job_dlq
WHERE resolved_at IS NULL
ORDER BY created_at DESC
LIMIT 10;
```

**Causas probables**:
1. **Upstream down**: API-Football, Understat, SofaScore con 429/503
2. **Rate limiting**: Demasiadas requests en ventana corta
3. **Data malformada**: Response no parseable

### F.4 Ref Coverage Bajo

**Síntomas**: `ref_coverage_pct` < 50% en dashboard.

**Diagnóstico**:
```sql
-- Matches sin ref de SofaScore (usando schema real)
SELECT m.id, th.name as home_team, ta.name as away_team, m.league_id
FROM matches m
JOIN teams th ON m.home_team_id = th.id
JOIN teams ta ON m.away_team_id = ta.id
LEFT JOIN match_external_refs mer
    ON m.id = mer.match_id AND mer.source = 'sofascore'
WHERE m.status = 'NS'
  AND m.date > NOW()
  AND m.league_id IN (140, 39, 135)
  AND mer.match_id IS NULL
LIMIT 10;
```

**Causas**: Job SOTA deshabilitado o matching fallido.

---

## G. Output Estructurado (Siempre)

Al reportar un audit, usa este formato:

```markdown
## TITAN Ops Audit Report

**Fecha**: {timestamp}
**Entorno**: {producción/staging}
**Ventana**: {período analizado}
**Skill version**: 1.2.1

### Summary
- **Health**: {valor del endpoint, NO calculado por skill}
- **PIT Violations**: {0/N}
- **Tier Coverage**: T1={X}%, T1b={Y}%, T1c={Z}%

### Evidence
{2-6 líneas de queries/dashboard relevantes}

### Diagnosis
- {Bullet 1: observación}
- {Bullet 2: observación}

### Sections Status
- Completadas: {lista}
- Fallidas: {lista o "ninguna"}

### Next Actions
1. {Acción concreta si hay issues, o "Ninguna requerida"}
```

---

## H. Ejemplos

### Ejemplo A: Audit OK

```markdown
## TITAN Ops Audit Report

**Fecha**: 2026-01-26T15:00:00Z
**Entorno**: producción
**Ventana**: 7 días
**Skill version**: 1.2.1

### Summary
- **Health**: healthy (from /dashboard/titan.json)
- **PIT Violations**: 0
- **Tier Coverage**: T1=100%, T1b=85%, T1c=45%

### Evidence
- Dashboard: health=healthy, pit_violations=0
- tier1c_complete: 12 rows, coverage_pct=45.0%
- lineup_freshness_hours: 3.2

### Diagnosis
- Sistema operando normalmente
- Tier 1c coverage esperado (lineups solo disponibles pre-KO)

### Sections Status
- Completadas: Dashboard, PIT, Coverage, Freshness, DLQ
- Fallidas: ninguna

### Next Actions
- Ninguna requerida
```

### Ejemplo B: PIT Violation Detectada

```markdown
## TITAN Ops Audit Report

**Fecha**: 2026-01-26T15:00:00Z
**Entorno**: producción
**Ventana**: 7 días
**Skill version**: 1.2.1

### Summary
- **Health**: unhealthy (from /dashboard/titan.json)
- **PIT Violations**: 3
- **Tier Coverage**: T1=100%, T1b=85%, T1c=40%

### Evidence
- pit_violations query: 3 rows con sofascore_lineup_captured_at >= kickoff_utc
- match_ids afectados: 1391045, 1391067, 1391089

### Diagnosis
- Bug en compute_lineup_features(): no está filtrando por PIT
- O timezone mismatch entre public.* (naive) y titan.* (aware)

### Sections Status
- Completadas: Dashboard, PIT, Coverage
- Fallidas: ninguna

### Next Actions
1. Escalar a ABE/Master para fix en materializer
2. Investigar con query E.3 los 3 match_ids específicos
```

---

## Changelog

### v1.2.1 (2026-01-26)
- **E.6 corregido**: DLQ usa `attempts`/`max_attempts` (no `retry_count`)
- **F.3 corregido**: DLQ troubleshooting usa columnas reales del schema

### v1.2.0 (2026-01-26)
- **E.2 corregido**: Query usa CTE pivot para `team_side` + JOIN a `teams` para nombres
- **F.1 corregido**: Troubleshooting usa `titan.feature_matrix.tier1_complete` (no tabla `odds`)
- **Columnas matches**: Queries usan `status` (no `status_short`), JOINs a `teams` para nombres
- **Health disclaimer**: Añadido que `/dashboard/titan.json.health` es source of truth
- **Versionamiento**: Añadido frontmatter con version, last_updated, compatible_with
- **Cuándo usar/NO usar**: Nueva sección de scope
- **Validación inputs**: Tabla de errores y defaults
- **Manejo errores infra**: Nueva sección C
- **Sections Status**: Output incluye completitud parcial
