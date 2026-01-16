# Competition Onboarding Protocol

> Checklist para agregar una nueva competición al sistema FutbolStats.

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

**Criterio OK**: Partidos aparecen en DB y visibles en iOS.

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

**Criterio OK**: Partidos en ventana 48h tienen odds (o API no los provee).

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

**Criterio OK**: Jobs captura stats de partidos FT dentro del lookback (72h).

### 5. Monitoring Check

```bash
# Verificar jobs_health sin errores nuevos
curl -s -H "X-Dashboard-Token: $TOKEN" \
  "https://web-production-f2de9.up.railway.app/dashboard/ops.json" \
  | jq '.data.jobs_health'

# Verificar Sentry sin nuevos issues relacionados
# https://futbolstats.sentry.io/issues/?query=league_id%3A143
```

**Criterio OK**:
- `jobs_health.status = "ok"`
- Sin nuevos errores en Sentry con el league_id

---

## P1: Opcional (Analítica/ML)

> Solo ejecutar si la competición requiere:
> - Entrenar modelo con historial
> - Market skill analysis (odds closing)
> - Feature engineering con stats históricas

### Criterio de Decisión: ¿UX-only vs Analítica?

| Señal | UX-only | Analítica/ML |
|-------|---------|--------------|
| match_weight | < 0.9 | >= 0.9 |
| Volumen partidos/año | < 50 | > 100 |
| Equipos también en ligas top | Sí | No |
| Requiere predicciones calibradas | No | Sí |

**Ejemplo Copa del Rey**: UX-only (weight=0.85, equipos ya en La Liga, <100 partidos/año)

### Stats Backfill Histórico

```sql
-- Identificar partidos FT sin stats (últimos 90 días)
SELECT COUNT(*) as pending,
       MIN(date) as oldest,
       MAX(date) as newest
FROM matches
WHERE league_id = 143
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
WHERE m.league_id = 143
  AND m.status IN ('FT', 'AET', 'PEN')
  AND m.date > NOW() - INTERVAL '12 months'
GROUP BY 1
ORDER BY 1 DESC;
```

**Decisión**: Si coverage < 50% y necesitas odds históricos, evaluar fetch manual.

---

## Checklist Final (Copy-Paste)

```markdown
## Competition Onboarding: [NOMBRE] (league_id=[ID])

### P0 Obligatorio
- [ ] `competitions.py`: Competition definida con priority/weight
- [ ] `scheduler.py`: league_id en EXTENDED_LEAGUES
- [ ] Deploy completado
- [ ] Fixtures visibles en DB (`SELECT ... WHERE league_id=X`)
- [ ] Partidos aparecen en iOS
- [ ] Odds sync: partidos 48h tienen odds (o N/A)
- [ ] Stats backfill: FT recientes tienen stats
- [ ] jobs_health: status=ok
- [ ] Sentry: sin errores nuevos

### P1 Analítica (si aplica)
- [ ] Decisión: UX-only / Analítica
- [ ] Stats backfill histórico: N partidos procesados
- [ ] Odds backfill histórico: cobertura X%
```

---

## Registro de Onboardings

| Fecha | Competition | league_id | Tipo | Notas |
|-------|-------------|-----------|------|-------|
| 2026-01-16 | Copa del Rey | 143 | UX-only | weight=0.85, priority=MEDIUM |

