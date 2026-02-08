# Team Enrichment System

Sistema de enriquecimiento de datos de equipos para FutbolStats.

## Arquitectura General

```
┌─────────────────────────────────────────────────────────────────────┐
│                           DATA SOURCES                              │
├──────────┬──────────┬──────────┬──────────────┬─────────────────────┤
│ Wikidata │ Wikipedia│  Venue   │   Website    │  Manual Override    │
│ (SPARQL) │ (Scrape) │ (matches)│  (Scraping)  │  (Dashboard)        │
└────┬─────┴────┬─────┴────┬─────┴──────┬───────┴──────────┬──────────┘
     │          │          │            │                  │
     ▼          ▼          ▼            ▼                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│              merge_enrichment_data() — CASCADE PRIORITY             │
│   override > wikidata > wikipedia > venue (lowest fills gaps only)  │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    team_wikidata_enrichment                         │
│  (stadium, city, coords, social, full_name, short_name)            │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              │ COALESCE priority
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                  team_enrichment_overrides                          │
│  (short_name manual, social overrides)                              │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    display_name en endpoints
```

---

## 1. Tablas de Datos

### `team_wikidata_enrichment`
Datos estructurados desde Wikidata SPARQL + fallbacks.

| Campo | Tipo | Descripción |
|-------|------|-------------|
| `team_id` | INT PK | FK a teams.id |
| `wikidata_id` | VARCHAR(20) NOT NULL | Q-number (ej: Q6150984). CHECK `'^Q[0-9]+$'` |
| `stadium_name` | VARCHAR(255) | Nombre del estadio |
| `stadium_wikidata_id` | VARCHAR(20) | QID del estadio |
| `stadium_capacity` | INT | Capacidad |
| `stadium_altitude_m` | INT | Altitud en metros |
| `lat`, `lon` | DOUBLE | Coords del estadio (P115→P625) |
| `admin_location_label` | VARCHAR | Ciudad del equipo (P131/P159) |
| `full_name` | VARCHAR(500) | Nombre oficial (P1448) |
| `short_name` | VARCHAR(255) | Nombre corto (P1813) |
| `social_handles` | JSONB | `{"twitter": "x", "instagram": "y"}` |
| `website` | VARCHAR(500) | Sitio oficial |
| `colors` | JSONB | Colores del equipo |
| `fetched_at` | TIMESTAMP | Última actualización |
| `enrichment_source` | VARCHAR(50) | Origen: ver sección 2.3 |
| `enrichment_version` | INT | Versión del schema de enrichment |
| `raw_jsonb` | JSONB | Payloads originales (provenance) |

**Restricción P0**: `wikidata_id` es NOT NULL con CHECK regex. No se puede INSERT sin QID válido.

### `team_enrichment_overrides`
Overrides manuales desde el dashboard (prioridad máxima).

| Campo | Tipo | Descripción |
|-------|------|-------------|
| `team_id` | INT PK | FK a teams.id |
| `short_name` | VARCHAR(255) | Override manual del nombre corto |
| `twitter` | VARCHAR(100) | Override de Twitter handle |
| `instagram` | VARCHAR(100) | Override de Instagram handle |
| `updated_at` | TIMESTAMP | Última modificación |

### `team_manager_history`
Historial de técnicos detectados por el sync diario.

| Campo | Tipo | Descripción |
|-------|------|-------------|
| `id` | SERIAL PK | ID auto |
| `team_id` | INT | FK a teams.id |
| `coach_name` | VARCHAR | Nombre del DT |
| `start_date` | DATE | Inicio del stint |
| `end_date` | DATE NULL | Fin del stint (NULL = actual) |
| `detected_at` | TIMESTAMP | Cuándo se detectó |

---

## 2. Enrichment Cascade

### 2.1 Prioridad (mayor a menor)

```
1. Override manual (team_enrichment_overrides)
2. Wikidata SPARQL (P131, P159, P115, P1448, etc.)
3. Wikipedia scraping (infobox parsing)
4. Venue harvest (matches.venue_name / venue_city)
```

Cada nivel solo llena campos que los niveles superiores dejaron vacíos. `merge_enrichment_data()` en `wikidata_enrich.py` implementa esta lógica.

### 2.2 Venue Harvest Fallback

Cuando Wikidata/Wikipedia no proveen stadium o city, se usa el venue más frecuente de los home matches de liga del equipo:

```sql
SELECT venue_name, venue_city, COUNT(*) AS freq
FROM matches m
JOIN admin_leagues al ON al.league_id = m.league_id
WHERE m.home_team_id = :team_id
  AND m.venue_name IS NOT NULL
  AND al.kind = 'league'
  AND m.status IN ('FT','AET','PEN')
  AND m.date >= NOW() - INTERVAL '365 days'
GROUP BY venue_name, venue_city
ORDER BY freq DESC
LIMIT 1
```

**Heurística**: Usa el venue más frecuente (no el más reciente) para filtrar sedes neutrales de copas. Solo considera matches de liga (`al.kind = 'league'`).

Implementado en `_get_venue_fallback()` en `app/etl/wikidata_enrich.py`.

### 2.3 Valores de `enrichment_source`

| Valor | Significado |
|-------|-------------|
| `wikidata` | Solo datos de Wikidata SPARQL |
| `wikipedia` | Solo datos de Wikipedia scraping |
| `wikidata+wikipedia` | Merge de ambas fuentes |
| `venue_harvest` | Solo datos de matches.venue_name/city |
| `manual` | Override manual desde dashboard |

---

## 3. Jobs Programados

### `sota_wikidata_team_enrich`
**Frecuencia**: Diario 04:30 UTC (catch-up) / Semanal domingos (refresh)
**Feature flag**: `WIKIDATA_ENRICH_ENABLED`

**Modos**:
- **Catch-up**: Procesa equipos sin enrichment (batch=100)
- **Refresh**: Actualiza equipos con enrichment >30 días (batch=50)

**Lógica**:
```python
# Catch-up: equipos con wikidata_id pero sin enrichment
SELECT t.id, t.wikidata_id FROM teams t
LEFT JOIN team_wikidata_enrichment twe ON t.id = twe.team_id
WHERE t.wikidata_id IS NOT NULL AND twe.team_id IS NULL
LIMIT 100

# Refresh: equipos con enrichment viejo
SELECT t.id, t.wikidata_id FROM teams t
JOIN team_wikidata_enrichment twe ON t.id = twe.team_id
WHERE twe.fetched_at < NOW() - INTERVAL '30 days'
LIMIT 50
```

**Guardrails**:
- Rate limit: 0.2s entre requests (5 req/sec)
- 429 backoff: Respeta Retry-After header
- Fail-open: Errores no crashean el job
- Venue fallback automático cuando SPARQL+Wikipedia no proveen stadium/city

### `player_manager_sync`
**Frecuencia**: Diario 02:00 UTC
**Feature flag**: `MANAGER_SYNC_ENABLED`
**API calls**: ~560/día (0.4% del budget de 150K)

Detecta cambios de DT consultando API-Football `/coachs?team={id}`. Si el coach actual difiere del registrado, cierra el stint anterior (`end_date = hoy`) e inserta el nuevo.

**Bug fix (commit 28bb758)**: `_find_current_coach()` ahora selecciona el coach con `start` más reciente entre los que tienen `end=null`, en vez del primero encontrado. API-Football puede retornar stints viejos con `end=null` que nunca se cerraron.

---

## 4. Website Scraping (Social Media)

Script manual para extraer redes sociales de sitios oficiales.

**Archivo**: `scripts/extract_social_from_websites.py`

**Fuentes de extracción** (orden de prioridad):
1. JSON-LD `sameAs` (más confiable)
2. Meta tag `twitter:site`
3. Links en footer/header
4. Links en documento completo

**Protecciones**:
- SSRF: Bloquea IPs privadas y localhost
- Robots.txt: Best-effort check
- Payload limit: 3MB máximo
- Rate limit: 2s entre requests

**Ejecución**:
```bash
# Dry-run (ver candidatos)
python scripts/extract_social_from_websites.py --dry-run

# Aplicar (batches de 50)
python scripts/extract_social_from_websites.py --apply --batch-size 50
```

---

## 5. Display Name (Short Names)

### Patrón COALESCE
```sql
COALESCE(
    team_enrichment_overrides.short_name,  -- Prioridad 1: Override manual
    team_wikidata_enrichment.short_name,   -- Prioridad 2: Wikidata
    teams.name                              -- Fallback: Nombre original
) AS display_name
```

### Endpoints que incluyen display_name

| Endpoint | Campos | Uso |
|----------|--------|-----|
| `GET /standings/{league_id}` | `display_name` | Standings con toggle |
| `GET /football/league/{id}` | `home_display_name`, `away_display_name` | Recent/Next matches |
| `GET /matches/{id}/details` | `display_name` (home/away) | Match detail popup |
| `GET /dashboard/matches.json` | `home_display_name`, `away_display_name` | Matches page |

### Toggle `use_short_names`

Almacenado en `admin_leagues.tags`:
```json
{"use_short_names": true}
```

**Comportamiento**:
- **Football page**: Respeta el toggle por liga
- **Matches page**: Siempre usa displayName (consistencia visual)

---

## 6. API de Enrichment (Dashboard)

### PUT `/dashboard/admin/teams/{team_id}/enrichment`
Upsert de override manual.

```json
{
  "short_name": "América",
  "twitter": "AmericadeCali",
  "instagram": "americadecali"
}
```

### DELETE `/dashboard/admin/teams/{team_id}/enrichment`
Elimina override manual (vuelve a usar Wikidata).

### Cache Invalidation
Al modificar enrichment:
- `football-team` (detalle del equipo)
- `standings` (tabla de posiciones)

---

## 7. Monitoring (OPS Dashboard)

Sección `sota_enrichment` en `/dashboard/ops.json` incluye dos componentes:

### `wikidata_enrichment`
Cobertura de equipos activos de liga (30d) con stadium y city.

| Métrica | Descripción |
|---------|-------------|
| `coverage_pct` | MIN(stadium_pct, city_pct) |
| `stadium` | % equipos con `stadium_name` |
| `city` | % equipos con `admin_location_label` |
| `qid` | % equipos con `wikidata_id` (cuello de botella) |
| `staleness_hours` | Edad del `fetched_at` más viejo |

**Umbrales**: ok (>=95%) | warn (>=80%) | red (<80%)

### `managers`
Cobertura de equipos activos con DT actual (`end_date IS NULL`).

| Métrica | Descripción |
|---------|-------------|
| `coverage_pct` | % equipos con manager actual |
| `staleness_hours` | Tiempo desde última detección |

**Umbrales**: ok (>=90%) | warn (>=70%) | red (<70%)

---

## 8. Cobertura Actual (2026-02-06)

| Métrica | Equipos activos liga | Cobertura | Notas |
|---------|---------------------|-----------|-------|
| `wikidata_id` | 372 | **100%** | Todos los activos tienen QID |
| Stadium | 372 | **100%** | Wikidata + venue harvest |
| Ciudad | 372 | **100%** | Wikidata + venue harvest |
| Manager actual | 560 | **100%** | Re-synced post fix _find_current_coach |
| Twitter (Wikidata) | ~67% | parcial | De equipos con enrichment |
| Instagram (Wikidata) | ~55% | parcial | De equipos con enrichment |

### Historial de mejoras

| Fecha | Stadium | City | Acción |
|-------|---------|------|--------|
| Pre-2026-02 | 83% | 88% | Solo Wikidata SPARQL |
| 2026-02-06 | 99.5% | 99.2% | + Venue harvest desde matches (50 stadiums, 33 cities) |
| 2026-02-06 | 100% | 100% | + Fix manual 3 residuales (Al Khaleej, Concepción, FC Cajamarca) |

---

## 9. Scripts de Backfill

### `scripts/backfill_venue_from_matches.py`
Backfill one-off de stadium/city desde `matches.venue_name/venue_city`.

**Heurística**: Venue más frecuente en home matches de liga (filtra sedes neutrales).
**Restricción**: Solo UPDATE filas existentes en `team_wikidata_enrichment` (respeta NOT NULL de `wikidata_id`).

```bash
source .env
python scripts/backfill_venue_from_matches.py --batch 100 --dry-run
python scripts/backfill_venue_from_matches.py --batch 100
```

### `scripts/backfill_city.py`
Backfill de `admin_location_label` vía Wikidata SPARQL (P131 + P159 fallback).

```bash
source .env
python scripts/backfill_city.py --batch 50 --dry-run
python scripts/backfill_city.py --batch 50
```

---

## 10. Archivos Clave

```
app/
├── etl/wikidata_enrich.py       # Job de enriquecimiento (cascade + venue fallback)
├── etl/player_jobs.py           # Manager sync (_find_current_coach)
├── teams/overrides.py           # Lógica de display_name y overrides
├── dashboard/ops_routes.py      # Monitoring (wikidata_enrichment + managers)
└── scheduler.py                 # Job scheduling

scripts/
├── backfill_venue_from_matches.py  # Venue harvest one-off
├── backfill_city.py                # City backfill via SPARQL
├── extract_social_from_websites.py # Scraping de social media
└── wikidata_catchup.py             # Utilidad para catch-up manual

migrations/
├── 044_team_wikidata_enrichment.sql
└── 045_team_enrichment_overrides.sql

dashboard/
├── lib/hooks/use-team-enrichment-mutation.ts
└── components/football/LeagueSettingsDrawer.tsx
```

---

## 11. Troubleshooting

### Equipo sin datos en Team 360
1. Verificar `teams.wikidata_id` — si es NULL, buscar QID en Wikidata manualmente
2. Verificar fila en `team_wikidata_enrichment` — si no existe, el pipeline la creará si tiene QID
3. Si no existe QID en Wikidata, usar override manual vía dashboard

### Manager incorrecto
- API-Football puede retornar stints viejos con `end=null` (nunca cerrados)
- `_find_current_coach()` selecciona el stint con `start` más reciente
- Si persiste, verificar en API-Football directamente: `/coachs?team={external_id}`

### Venue harvest no encuentra datos
- Requiere home matches de liga en los últimos 365 días
- Equipos recién ascendidos pueden no tener matches suficientes
- Fallback: override manual desde dashboard
