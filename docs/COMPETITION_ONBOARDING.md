# Competition Onboarding Protocol

> Checklist auditable para agregar una nueva competición al sistema FutbolStats.

## Quick Reference

| Paso | P0 (Obligatorio) | P1 (Analítica/ML) |
|------|------------------|-------------------|
| 1. Código | `competitions.py` + `scheduler.py` | - |
| 2. Fixtures/Teams | Verificar sync | - |
| 3. Odds | Verificar odds_sync | Backfill histórico |
| 4. Stats | Verificar stats_backfill | Backfill 30-90 días |
| 5. Lineups | Forward-only | - |
| 6. Monitoring | jobs_health + Sentry | - |

---

## P0: Obligatorio (UX-ready)

### 1. Agregar Competition al código

```python
# app/etl/competitions.py

# Definir la competición
COPA_DEL_REY = Competition(
    league_id=143,
    name="Copa del Rey",
    match_type="official",      # "official" | "friendly"
    priority=Priority.MEDIUM,   # HIGH | MEDIUM | LOW
    match_weight=0.85,          # Liga=1.0, Copa=0.85, Friendly=0.6
)

# Agregar al dict COMPETITIONS
COMPETITIONS: dict[int, Competition] = {
    ...
    COPA_DEL_REY,
}
```

```python
# app/scheduler.py - verificar que league_id está en EXTENDED_LEAGUES
EXTENDED_LEAGUES = [
    ...
    143,  # Copa del Rey
]
```

### 2. Verificar Fixtures/Teams (post-deploy)

```sql
-- Partidos de la nueva competición
SELECT m.id, m.external_id, ht.name as home, at.name as away,
       m.date, m.status, m.league_id
FROM matches m
JOIN teams ht ON ht.id = m.home_team_id
JOIN teams at ON at.id = m.away_team_id
WHERE m.league_id = 143
ORDER BY m.date DESC
LIMIT 10;

-- Confirmar equipos tienen logo
SELECT t.id, t.name, t.logo_url
FROM teams t
JOIN matches m ON t.id IN (m.home_team_id, m.away_team_id)
WHERE m.league_id = 143
GROUP BY t.id
HAVING t.logo_url IS NULL;
```

### 3. Verificar Odds Sync

```sql
-- Partidos próximos con/sin odds
SELECT m.id, ht.name || ' vs ' || at.name as match,
       m.date, m.odds_home, m.odds_draw, m.odds_away,
       m.odds_recorded_at
FROM matches m
JOIN teams ht ON ht.id = m.home_team_id
JOIN teams at ON at.id = m.away_team_id
WHERE m.league_id = 143
  AND m.status = 'NS'
  AND m.date <= NOW() + INTERVAL '48 hours'
ORDER BY m.date;
```

### 4. Verificar Stats Backfill

```sql
-- Partidos FT sin stats
SELECT m.id, ht.name || ' vs ' || at.name as match,
       m.date, m.status,
       CASE WHEN m.stats IS NULL OR m.stats::text = '{}'
            THEN 'MISSING' ELSE 'OK' END as stats_status
FROM matches m
JOIN teams ht ON ht.id = m.home_team_id
JOIN teams at ON at.id = m.away_team_id
WHERE m.league_id = 143
  AND m.status IN ('FT', 'AET', 'PEN')
  AND m.date > NOW() - INTERVAL '72 hours'
ORDER BY m.date DESC;
```

### 5. Monitoring Check

```bash
# Verificar jobs_health sin errores nuevos
curl -s -H "X-Dashboard-Token: $TOKEN" \
  "https://web-production-f2de9.up.railway.app/dashboard/ops.json" \
  | jq '.data.jobs_health'

# Verificar Sentry sin nuevos issues relacionados
# https://futbolstats.sentry.io/issues/?query=league_id%3A143
```

---

## P0 Exit Checklist

> **Criterio de cierre**: Todos los checks deben ser OK o tener justificación documentada.

| # | Check | Verificación | Expected Output | Status |
|---|-------|--------------|-----------------|--------|
| 1 | **Código desplegado** | `git log -1 --oneline` en prod | Commit con `competitions.py` | OK / FAIL |
| 2 | **Fixtures en DB** | Query SQL abajo | >= 1 partido con `league_id=X` | OK / FAIL |
| 3 | **Visible en iOS** | `GET /predictions?days_ahead=30` | Partidos de la competición aparecen | OK / FAIL |
| 4 | **Odds sync** | Query SQL abajo | Partidos 48h tienen odds, o documentar "N/A - API no provee" | OK / N/A |
| 5 | **Stats backfill** | Query SQL abajo | FT en últimas 72h tienen stats, o "N/A - sin FT recientes" | OK / N/A |
| 6 | **jobs_health** | `curl .../ops.json \| jq '.data.jobs_health.status'` | `"ok"` | OK / FAIL |
| 7 | **Sentry** | Revisar issues últimas 24h | Sin errores nuevos con `league_id:X` | OK / FAIL |

### Queries de verificación

```sql
-- Check 2: Fixtures en DB
SELECT COUNT(*) as fixtures_count
FROM matches
WHERE league_id = {LEAGUE_ID}
  AND date >= NOW() - INTERVAL '30 days'
  AND date <= NOW() + INTERVAL '30 days';
-- Expected: >= 1

-- Check 4: Odds coverage (partidos próximos 48h)
SELECT
  COUNT(*) as total_upcoming,
  COUNT(odds_home) as with_odds,
  COUNT(*) - COUNT(odds_home) as missing_odds
FROM matches
WHERE league_id = {LEAGUE_ID}
  AND status = 'NS'
  AND date <= NOW() + INTERVAL '48 hours';
-- Expected: missing_odds = 0, o documentar razón

-- Check 5: Stats backfill (FT últimas 72h)
SELECT
  COUNT(*) as total_ft,
  SUM(CASE WHEN stats IS NOT NULL AND stats::text != '{}' THEN 1 ELSE 0 END) as with_stats
FROM matches
WHERE league_id = {LEAGUE_ID}
  AND status IN ('FT', 'AET', 'PEN')
  AND date > NOW() - INTERVAL '72 hours';
-- Expected: with_stats = total_ft, o "N/A" si total_ft = 0
```

### Comandos de verificación

```bash
# Check 3: Visible en iOS
curl -s -H "X-API-Key: $API_KEY" \
  "https://web-production-f2de9.up.railway.app/predictions?days_ahead=30" \
  | jq '[.predictions[] | select(.league_id == {LEAGUE_ID})] | length'
# Expected: >= 1 (si hay partidos programados)

# Check 6: jobs_health
curl -s -H "X-Dashboard-Token: $TOKEN" \
  "https://web-production-f2de9.up.railway.app/dashboard/ops.json" \
  | jq '.data.jobs_health.status'
# Expected: "ok"

# Check 7: Sentry (manual)
# Visitar: https://futbolstats.sentry.io/issues/?query=league_id:{LEAGUE_ID}
# Expected: 0 issues en últimas 24h
```

---

## P1: Opcional (Analítica/ML)

> Solo ejecutar si la competición requiere:
> - Entrenar modelo con historial específico
> - Market skill analysis (odds closing)
> - Feature engineering con stats históricas

### Heurísticas de Decisión: ¿UX-only vs Analítica?

> **IMPORTANTE**: Estas son heurísticas orientativas, NO reglas absolutas.
> Si el torneo es estratégicamente importante (alto tráfico, mercado clave, sponsor)
> o tiene características únicas que justifiquen análisis profundo, se puede elevar
> a Analítica aunque cumpla criterios de UX-only.

| Señal | Sugiere UX-only | Sugiere Analítica/ML |
|-------|-----------------|----------------------|
| match_weight configurado | < 0.9 | >= 0.9 |
| Volumen partidos/año | < 50 | > 100 |
| Equipos también en ligas top | Sí (datos redundantes) | No (datos únicos) |
| Requiere predicciones altamente calibradas | No | Sí |
| Tráfico/engagement esperado | Bajo/medio | Alto |
| Mercado estratégico | No | Sí |

**Ejemplo Copa del Rey (2026-01-16)**:
- Heurísticas: weight=0.85, equipos en La Liga, <100 partidos/año → sugieren UX-only
- Decisión final: **UX-only** (sin factores estratégicos que justifiquen Analítica)

**Contraejemplo hipotético - Liga Saudí**:
- Heurísticas: weight=0.9, equipos únicos, ~300 partidos/año → sugieren Analítica
- Factor adicional: Alto tráfico esperado por fichajes mediáticos
- Decisión final: **Analítica** (backfill histórico justificado)

### Stats Backfill Histórico

```sql
-- Identificar partidos FT sin stats (últimos 90 días)
SELECT COUNT(*) as pending,
       MIN(date) as oldest,
       MAX(date) as newest
FROM matches
WHERE league_id = {LEAGUE_ID}
  AND status IN ('FT', 'AET', 'PEN')
  AND date > NOW() - INTERVAL '90 days'
  AND (stats IS NULL OR stats::text = '{}');
```

```bash
# Ejecutar backfill manual (si >10 pendientes)
# Opción 1: Esperar que job stats_backfill procese gradualmente
# Opción 2: Script one-off con rate limiting

# Ver progreso
railway logs -n 20 --filter "Stats backfill"
```

### Odds Backfill Histórico

> Solo necesario para market skill analysis o calibración de implied probabilities.

```sql
-- Verificar cobertura histórica de odds
SELECT
  DATE_TRUNC('month', m.date) as month,
  COUNT(*) as total,
  COUNT(m.odds_home) as with_odds,
  ROUND(100.0 * COUNT(m.odds_home) / COUNT(*), 1) as coverage_pct
FROM matches m
WHERE m.league_id = {LEAGUE_ID}
  AND m.status IN ('FT', 'AET', 'PEN')
  AND m.date > NOW() - INTERVAL '12 months'
GROUP BY 1
ORDER BY 1 DESC;
```

**Decisión**: Si coverage < 50% y necesitas odds históricos, evaluar fetch manual.

---

## Riesgos Típicos y Mitigaciones

### 1. Cups/Knockouts sin Standings (404 esperado)

**Síntoma**: Warmup de standings falla con 404 para la competición.

**Causa**: Copas y torneos knockout no tienen tabla de posiciones tradicional.

**Mitigación**:
- El código actual maneja esto gracefully (try/except por liga, logging warning)
- No requiere acción inmediata
- Opcional P2: Agregar flag `has_standings=False` en Competition para suprimir warnings

```sql
-- Verificar si es cup/knockout
SELECT league_id, name, match_type
FROM competitions
WHERE league_id = {LEAGUE_ID};
-- Si es copa, 404 en standings es esperado
```

### 2. Ligas con Múltiples Grupos/Fases

**Síntoma**: Standings devuelve múltiples grupos, duplicados en UI.

**Causa**: Competiciones con fase de grupos (Champions League, Libertadores, etc.)

**Mitigación**:
- Verificar cómo API-Football estructura los datos
- Si hay múltiples grupos, el código actual toma el primero
- Para torneos con fases, puede requerir lógica específica

```sql
-- Ver estructura de standings guardada
SELECT league_id, season,
       jsonb_typeof(standings) as type,
       jsonb_array_length(standings) as groups_count
FROM league_standings
WHERE league_id = {LEAGUE_ID}
ORDER BY captured_at DESC
LIMIT 1;
```

### 3. Season Calendar-Year vs European

**Síntoma**: Partidos no aparecen porque season está mal calculado.

**Causa**: Ligas LATAM usan año calendario (2026), europeas usan temporada (2025-26 = "2025").

**Mitigación**:
- Verificar `_season_for_league()` en main.py
- Ligas LATAM (league_id en lista específica) usan año actual
- Europeas usan año anterior si estamos en primera mitad del año

```python
# Ver lógica actual
def _season_for_league(league_id: int, now: datetime) -> int:
    # LATAM leagues use calendar year
    latam_leagues = [71, 128, 239, 242, 250, 262, 265, 268, 281, 299, 344]
    if league_id in latam_leagues:
        return now.year
    # European leagues: if Jan-Jul, use previous year
    return now.year if now.month >= 8 else now.year - 1
```

**Acción si falla**: Agregar league_id a `latam_leagues` si es competición calendario-year.

### 4. Rebrandings y TeamOverride

**Síntoma**: Equipo aparece con nombre/logo incorrecto o duplicado.

**Causa**: API-Football cambió ID o nombre del equipo (rebrand, fusión, etc.)

**Mitigación**:
- Tabla `team_overrides` permite mapear IDs y nombres
- Verificar si el equipo tiene override configurado

```sql
-- Buscar overrides existentes
SELECT * FROM team_overrides
WHERE old_team_id IN (
  SELECT DISTINCT home_team_id FROM matches WHERE league_id = {LEAGUE_ID}
  UNION
  SELECT DISTINCT away_team_id FROM matches WHERE league_id = {LEAGUE_ID}
);

-- Si necesitas agregar override
INSERT INTO team_overrides (old_team_id, new_team_id, old_name, new_name, reason)
VALUES (123, 456, 'Nombre Viejo', 'Nombre Nuevo', 'Rebranding 2026');
```

### 5. Odds Coverage Bajo o Nulo

**Síntoma**: Partidos próximos no tienen odds.

**Causas posibles**:
1. API-Football no cubre odds para esa liga
2. Job odds_sync no incluye la liga en su ventana
3. Partidos muy lejanos (>48h) aún no tienen odds publicados

**Diagnóstico**:
```sql
-- Verificar cobertura general de la liga
SELECT
  COUNT(*) as total_matches,
  COUNT(odds_home) as with_odds,
  ROUND(100.0 * COUNT(odds_home) / NULLIF(COUNT(*), 0), 1) as coverage_pct
FROM matches
WHERE league_id = {LEAGUE_ID}
  AND status IN ('FT', 'AET', 'PEN')
  AND date > NOW() - INTERVAL '30 days';

-- Si coverage = 0%, probablemente API no cubre
-- Si coverage > 0% pero upcoming no tiene, verificar timing
```

**Mitigación**:
- Si API no cubre: Documentar "N/A - odds no disponibles" en checklist
- Si es timing: Los odds aparecerán más cerca del partido
- Si es bug en job: Revisar logs `railway logs -n 30 --filter "odds_sync"`

---

## Checklist Final (Copy-Paste)

```markdown
## Competition Onboarding: [NOMBRE] (league_id=[ID])

**Fecha**: YYYY-MM-DD
**Responsable**: [nombre]
**Decisión**: UX-only / Analítica (justificación: ...)

### P0 Exit Checklist

| # | Check | Status | Notas |
|---|-------|--------|-------|
| 1 | Código desplegado | OK/FAIL | commit: abc123 |
| 2 | Fixtures en DB | OK/FAIL | N partidos encontrados |
| 3 | Visible en iOS | OK/FAIL | verificado en /predictions |
| 4 | Odds sync | OK/N/A | cobertura X% o "API no provee" |
| 5 | Stats backfill | OK/N/A | X/Y FT con stats o "sin FT recientes" |
| 6 | jobs_health | OK/FAIL | status="ok" |
| 7 | Sentry | OK/FAIL | 0 errores nuevos |

**P0 Resultado**: PASS / FAIL (si FAIL, documentar blockers)

### P1 Analítica (si aplica)

| Check | Status | Notas |
|-------|--------|-------|
| Stats backfill histórico | OK/SKIP | N partidos procesados |
| Odds backfill histórico | OK/SKIP | cobertura X% |

### Riesgos Identificados

- [ ] Cup sin standings: esperado / no aplica
- [ ] Múltiples grupos: no aplica / requiere atención
- [ ] Season calendar-year: verificado / agregado a latam_leagues
- [ ] Team overrides: no requeridos / agregados (IDs: ...)
- [ ] Odds coverage: OK / bajo (documentado)
```

---

## Registro de Onboardings

| Fecha | Competition | league_id | Tipo | P0 Status | Notas |
|-------|-------------|-----------|------|-----------|-------|
| 2026-01-16 | Copa del Rey | 143 | UX-only | PASS | weight=0.85, priority=MEDIUM, cup sin standings (esperado) |
