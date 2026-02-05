# Team Enrichment System

Sistema de enriquecimiento de datos de equipos para FutbolStats.

## Arquitectura General

```
┌─────────────────────────────────────────────────────────────────┐
│                         DATA SOURCES                            │
├─────────────────┬─────────────────┬─────────────────────────────┤
│   Wikidata      │   Website       │   Manual Override           │
│   (SPARQL)      │   (Scraping)    │   (Dashboard)               │
└────────┬────────┴────────┬────────┴──────────────┬──────────────┘
         │                 │                       │
         ▼                 ▼                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                    team_wikidata_enrichment                     │
│  (stadium, coords, social, full_name, short_name)               │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ COALESCE priority
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                  team_enrichment_overrides                      │
│  (short_name manual, social overrides)                          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    display_name en endpoints
```

---

## 1. Tablas de Datos

### `team_wikidata_enrichment`
Datos estructurados desde Wikidata SPARQL.

| Campo | Tipo | Descripción |
|-------|------|-------------|
| `team_id` | INT PK | FK a teams.id |
| `wikidata_id` | VARCHAR(20) | Q-number (ej: Q6150984) |
| `stadium_name` | VARCHAR(255) | Nombre del estadio |
| `stadium_capacity` | INT | Capacidad |
| `lat`, `lon` | DOUBLE | Coords del estadio (P115→P625) |
| `full_name` | VARCHAR(500) | Nombre oficial (P1448) |
| `short_name` | VARCHAR(255) | Nombre corto (P1813) |
| `social_handles` | JSONB | `{"twitter": "x", "instagram": "y"}` |
| `website` | VARCHAR(500) | Sitio oficial |
| `fetched_at` | TIMESTAMP | Última actualización |
| `enrichment_source` | VARCHAR(50) | 'wikidata', 'wikidata+website' |

### `team_enrichment_overrides`
Overrides manuales desde el dashboard (prioridad máxima).

| Campo | Tipo | Descripción |
|-------|------|-------------|
| `team_id` | INT PK | FK a teams.id |
| `short_name` | VARCHAR(255) | Override manual del nombre corto |
| `twitter` | VARCHAR(100) | Override de Twitter handle |
| `instagram` | VARCHAR(100) | Override de Instagram handle |
| `updated_at` | TIMESTAMP | Última modificación |

---

## 2. Jobs Programados

### `wikidata_team_enrich`
**Frecuencia**: Diario 04:30 UTC (catch-up) / Semanal domingos (refresh)

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
- Feature flag: `WIKIDATA_ENRICH_ENABLED`

---

## 3. Website Scraping (Social Media)

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

## 4. Display Name (Short Names)

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

## 5. API de Enrichment (Dashboard)

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

## 6. Cobertura Actual

| Métrica | Valor | Notas |
|---------|-------|-------|
| Equipos con `wikidata_id` | ~736 (29%) | Prerequisito para enrichment |
| Equipos con enrichment | Variable | Crece con job diario |
| Twitter coverage (Wikidata) | ~67% | De equipos con enrichment |
| Instagram coverage (Wikidata) | ~55% | De equipos con enrichment |

**Limitación**: El enriquecimiento solo puede cubrir equipos que tengan `wikidata_id` válido en la tabla `teams`.

---

## 7. Archivos Clave

```
app/
├── etl/wikidata_enrich.py      # Job de enriquecimiento Wikidata
├── teams/overrides.py          # Lógica de display_name y overrides
├── main.py                     # Endpoints con display_name
└── scheduler.py                # Job scheduling

scripts/
├── extract_social_from_websites.py  # Scraping de social media
└── wikidata_catchup.py              # Utilidad para catch-up manual

migrations/
├── 044_team_wikidata_enrichment.sql
└── 045_team_enrichment_overrides.sql

dashboard/
├── lib/hooks/use-team-enrichment-mutation.ts
└── components/football/LeagueSettingsDrawer.tsx
```
