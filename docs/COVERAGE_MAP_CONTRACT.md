# Coverage Map Contract v1

Contrato formal para `GET /dashboard/coverage-map.json` (choropleth mundial de cobertura por liga/pais).

## 1) Endpoint

- **Metodo**: `GET`
- **Path**: `/dashboard/coverage-map.json`
- **Auth**: header `X-Dashboard-Token` (mismo mecanismo que `/dashboard/ops.json`)
- **Cache recomendado backend**: TTL `1800s` (30 min)
- **Timezone**: UTC

## 2) Query Params (exactos)

| Param | Tipo | Requerido | Default | Reglas |
|---|---|---:|---|---|
| `window` | enum | no | `since_2023` | `since_2023 \| season_to_date \| last_365d \| custom \| current_season \| prev_season \| prev_season_2` |
| `season` | int | condicional | `null` | Requerido cuando `window=season_to_date` |
| `from` | date (`YYYY-MM-DD`) | condicional | `null` | Requerido cuando `window=custom` |
| `to` | date (`YYYY-MM-DD`) | condicional | `null` | Requerido cuando `window=custom` y `to > from` |
| `league_ids` | csv<int> | no | `null` | Filtro opcional de ligas |
| `country_iso3` | csv<string> | no | `null` | Filtro opcional de paises ISO3 |
| `group_by` | enum | no | `country` | `country \| league` |
| `min_matches` | int | no | `30` | rango `1..500`; debajo de umbral => `insufficient_data` |
| `include_leagues` | bool | no | `true` | Incluye bloque `data.leagues` |
| `include_quality_flags` | bool | no | `true` | Incluye dimension `data_quality_flags` en tooltip (no en score total) |

### Resolucion de ventana temporal

**Ventanas globales** (rango identico para todas las ligas):

- `since_2023` -> `from=2023-01-01`, `to=now_utc`
- `season_to_date` -> `from=<season>-07-01`, `to=now_utc` (legacy, hardcoded julio)
- `last_365d` -> `from=now_utc-365d`, `to=now_utc`
- `custom` -> usa `from/to` del request

**Ventanas per-league** (rango depende de `admin_leagues.season_start_month`):

- `current_season` -> cada liga usa `make_date(base_year, ssm, 1)` a `make_date(base_year+1, ssm, 1)` donde `base_year = year si month >= ssm, else year-1`
- `prev_season` -> offset 1 (temporada anterior)
- `prev_season_2` -> offset 2 (hace 2 temporadas)

No requieren parametros adicionales. El SQL computa el rango per-league usando la columna `season_start_month` (INT, 1-12).

**Ejemplo** (2026-02-14): Premier League (ssm=8) current_season = 2025-08-01..2026-08-01; Argentina (ssm=1) = 2026-01-01..2027-01-01. Ambas filtran `status IN ('FT','AET','PEN')` (no incluye futuros).

## 3) Universo base y PIT guardrails

### Denominador comun por liga

`eligible_matches`:

- tabla base: `matches m`
- `m.status IN ('FT','AET','PEN')`
- `m.date >= from_utc AND m.date < to_utc`
- filtros opcionales por `league_ids` / `country_iso3`

### Formula de cobertura por dimension

`coverage_pct = round(100.0 * numerator / denominator, 1)`

- `denominator = eligible_matches` de la liga
- `numerator = cantidad de matches con dato valido segun regla de dimension`

## 4) Catalogo de dimensiones (19 + 1 diagnostico)

Cada dimension expone: `key`, `label`, `priority`, `source_tables`, `contributes_to_score`, `pit_guardrail`.

### Nota sobre timezone en PIT guardrails SQL

`matches.date` es `timestamp without time zone` (UTC implicito). `odds_recorded_at` y `opening_odds_recorded_at` son `timestamp with time zone`. PostgreSQL coerce automaticamente en comparaciones, pero todas las fechas se asumen UTC.

### P0 (critico ML)

1. `xg`
   - valido si existe xG PIT-safe:
     - `match_understat_team` con `captured_at < m.date`, **o**
     - `match_fotmob_stats` con `xg_home IS NOT NULL AND xg_away IS NOT NULL` y `captured_at < m.date`
2. `odds_closing`
   - `m.odds_home > 1.0 AND m.odds_draw > 1.0 AND m.odds_away > 1.0`
   - `m.odds_recorded_at < m.date`
3. `odds_opening`
   - `m.opening_odds_home > 1.0 AND m.opening_odds_draw > 1.0 AND m.opening_odds_away > 1.0`
   - `m.opening_odds_recorded_at < m.date`
4. `lineups`
   - existen 2 filas en `match_lineups` (home/away) para el `match_id`
   - `array_length(starting_xi_ids, 1) >= 7` en ambos lados (columna es PostgreSQL `ARRAY`, no JSON)
   - si `lineup_confirmed_at IS NOT NULL`, debe cumplir `lineup_confirmed_at < m.date`

### P1 (SOTA opcional)

5. `weather`
   - existe fila en `match_weather` con `captured_at < m.date`
6. `bio_adaptability`
   - existen perfiles en `team_home_city_profile` para ambos equipos (`team_id` match home y away)
   - `timezone IS NOT NULL` para ambos
   - `climate_normals_by_month IS NOT NULL` para away
   - Nota: tabla es perfil estatico (sin `captured_at`); tiene `last_updated_at` (timestamptz) pero no aplica PIT per-match
7. `sofascore_xi_ratings`
   - `match_sofascore_player`: >= 11 filas con `is_starter=true` por lado
   - al menos `rating_pre_match` o `rating_recent_form` no nulo por jugador
   - `captured_at < m.date`
8. `external_refs`
   - existe al menos 1 fila en `match_external_refs` para el `match_id`
   - `source` en (`understat`, `fotmob`, `sofascore`)
   - `confidence >= 0.90` (auto-link auditable)
9. `freshness`
   - mide si los datos criticos (odds, xG, lineups) llegaron a tiempo pre-kickoff
   - valido si: para el match, al menos odds_closing Y (xg O lineups) tienen `captured_at`/`odds_recorded_at` < `m.date`
   - ratio: matches con datos frescos pre-kickoff / eligible_matches
   - Nota: dimension meta — refleja timeliness operacional, no presencia de dato
10. `join_health`
    - mide tasa de matching exitoso entre providers via `match_external_refs`
    - valido si: para el match, existe al menos 1 ref con `confidence >= 0.90`
    - por liga, reportar breakdown por `source` (understat/fotmob/sofascore match rates)
    - Nota: evita "cobertura falsa" donde xG/ratings existen en provider pero no se linkean al match

### P2 (operacional/enrichment)

11. `match_stats`
    - `m.stats IS NOT NULL` y `m.stats != '{}'::jsonb`
    - contiene claves minimas de tiros/corners por ambos lados
12. `match_events`
    - `m.events IS NOT NULL` y `m.events != '[]'::jsonb`
13. `venue`
    - `m.venue_name IS NOT NULL AND m.venue_city IS NOT NULL`
14. `referees`
    - datos de arbitro extraidos de `match_fotmob_stats.raw_stats` (JSONB, campo `referee` dentro del JSON)
    - Nota: no existen columnas `referee_*` dedicadas; el dato vive en `raw_stats` JSONB
    - fallback: `match_lineups.coach_id IS NOT NULL` (coach, no referee — cobertura aproximada)
15. `player_injuries`
    - snapshot utilizable pre-match en `player_injuries` para al menos un equipo
    - PIT: registros con fecha relevante anterior a `m.date`
16. `managers`
    - manager activo en `team_manager_history` para ambos equipos
    - valido si: existe fila con `team_id` = home/away, `start_date <= m.date` y (`end_date IS NULL` o `end_date >= m.date`)
    - Nota: tabla `team_managers` NO existe; usar solo `team_manager_history`
17. `squad_catalog`
    - cobertura de plantilla en `players` para ambos equipos (al menos 11 jugadores por `team_id`)
18. `standings`
    - fila en `league_standings` para (`league_id`, `season`) con `captured_at < m.date`

### Diagnostico (no contribuye al score)

19. `data_quality_flags`
    - metrica de calidad (no de presencia):
      - match `tainted = false` (o `tainted IS NULL`)
      - sin filas en `odds_history` con `quarantined = true` para ese `match_id`
    - `contributes_to_score = false` (solo diagnostico/tooltip)

## 5) Score agregado y Universe Tier

### Pesos exactos

- `P0 = 0.60` (4 dimensiones: xg, odds_closing, odds_opening, lineups)
- `P1 = 0.25` (6 dimensiones: weather, bio_adaptability, sofascore_xi_ratings, external_refs, freshness, join_health)
- `P2 = 0.15` (8 dimensiones: match_stats, match_events, venue, referees, player_injuries, managers, squad_catalog, standings)
- `data_quality_flags` excluido del score (solo diagnostico)

### Calculo

- `p0_pct = promedio simple de las 4 dimensiones P0`
- `p1_pct = promedio simple de las 6 dimensiones P1`
- `p2_pct = promedio simple de las 8 dimensiones P2`
- `coverage_total_pct = round(0.60*p0_pct + 0.25*p1_pct + 0.15*p2_pct, 1)`

### Universe coverage (liga)

Sobre `eligible_matches`:

- `base_pct = 100`
- `odds_pct = % matches con odds_closing valido`
- `xg_pct = % matches con xg valido`
- `odds_xg_pct = % matches con odds_closing AND xg`
- `xi_odds_xg_pct = % matches con lineups AND odds_closing AND xg`

### Universe tier (clasificacion)

- si `eligible_matches < min_matches` -> `insufficient_data`
- si `xi_odds_xg_pct >= 70` -> `xi_odds_xg`
- else si `odds_xg_pct >= 70` -> `odds_xg`
- else si `xg_pct >= 70` -> `xg`
- else si `odds_pct >= 70` -> `odds`
- else -> `base`

## 6) Response JSON (exacto)

```json
{
  "generated_at": "2026-02-13T23:10:12Z",
  "cached": false,
  "cache_age_seconds": 0,
  "data": {
    "contract_version": "coverage-map.v1",
    "request": {
      "window": "since_2023",
      "from": "2023-01-01",
      "to": "2026-02-13",
      "group_by": "country",
      "league_ids": [],
      "country_iso3": [],
      "min_matches": 30,
      "include_leagues": true,
      "include_quality_flags": true
    },
    "weights": {
      "p0": 0.6,
      "p1": 0.25,
      "p2": 0.15
    },
    "dimensions": [
      {
        "key": "xg",
        "label": "xG",
        "priority": "P0",
        "contributes_to_score": true,
        "source_tables": ["match_understat_team", "match_fotmob_stats"],
        "pit_guardrail": "captured_at < match.date"
      }
    ],
    "color_scale": [
      {"min": 0, "max": 24.9, "color": "#7f1d1d"},
      {"min": 25, "max": 49.9, "color": "#b45309"},
      {"min": 50, "max": 69.9, "color": "#0369a1"},
      {"min": 70, "max": 84.9, "color": "#15803d"},
      {"min": 85, "max": 100, "color": "#22c55e"}
    ],
    "countries": [
      {
        "country_iso3": "ARG",
        "country_name": "Argentina",
        "league_count": 2,
        "eligible_matches": 1240,
        "coverage_total_pct": 78.4,
        "p0_pct": 81.2,
        "p1_pct": 52.8,
        "p2_pct": 86.4,
        "universe_tier": "odds_xg",
        "universe_coverage": {
          "base_pct": 100.0,
          "odds_pct": 93.4,
          "xg_pct": 71.8,
          "odds_xg_pct": 69.1,
          "xi_odds_xg_pct": 55.2
        },
        "dimensions": {
          "xg": {"pct": 71.8, "numerator": 890, "denominator": 1240},
          "odds_closing": {"pct": 93.4, "numerator": 1158, "denominator": 1240},
          "odds_opening": {"pct": 89.7, "numerator": 1112, "denominator": 1240},
          "lineups": {"pct": 77.9, "numerator": 966, "denominator": 1240}
        }
      }
    ],
    "leagues": [
      {
        "league_id": 128,
        "league_name": "Liga Profesional Argentina",
        "country_iso3": "ARG",
        "country_name": "Argentina",
        "eligible_matches": 980,
        "coverage_total_pct": 79.1,
        "p0_pct": 82.0,
        "p1_pct": 53.4,
        "p2_pct": 86.7,
        "universe_tier": "odds_xg",
        "universe_coverage": {
          "base_pct": 100.0,
          "odds_pct": 94.1,
          "xg_pct": 73.0,
          "odds_xg_pct": 70.6,
          "xi_odds_xg_pct": 56.0
        },
        "dimensions": {
          "xg": {"pct": 73.0, "numerator": 715, "denominator": 980},
          "odds_closing": {"pct": 94.1, "numerator": 922, "denominator": 980},
          "odds_opening": {"pct": 90.4, "numerator": 886, "denominator": 980},
          "lineups": {"pct": 78.3, "numerator": 767, "denominator": 980},
          "weather": {"pct": 1.2, "numerator": 12, "denominator": 980},
          "bio_adaptability": {"pct": 61.0, "numerator": 598, "denominator": 980},
          "sofascore_xi_ratings": {"pct": 0.0, "numerator": 0, "denominator": 980},
          "external_refs": {"pct": 96.5, "numerator": 946, "denominator": 980},
          "match_stats": {"pct": 86.2, "numerator": 845, "denominator": 980},
          "match_events": {"pct": 84.9, "numerator": 832, "denominator": 980},
          "venue": {"pct": 95.1, "numerator": 932, "denominator": 980},
          "freshness": {"pct": 88.7, "numerator": 869, "denominator": 980},
          "join_health": {"pct": 94.2, "numerator": 923, "denominator": 980},
          "referees": {"pct": 68.0, "numerator": 666, "denominator": 980},
          "player_injuries": {"pct": 38.3, "numerator": 375, "denominator": 980},
          "managers": {"pct": 71.1, "numerator": 697, "denominator": 980},
          "squad_catalog": {"pct": 77.4, "numerator": 758, "denominator": 980},
          "standings": {"pct": 100.0, "numerator": 980, "denominator": 980},
          "data_quality_flags": {"pct": 97.2, "numerator": 953, "denominator": 980}
        }
      }
    ],
    "summary": {
      "countries": 18,
      "leagues": 25,
      "eligible_matches": 21456,
      "coverage_total_pct_mean": 74.9
    }
  }
}
```

## 7) Codigos de error

- `401 Unauthorized`: falta token o token invalido.
- `400 Bad Request`: combinacion de parametros invalida (`window` desconocido, `from/to` inconsistentes).
- `422 Unprocessable Entity`: tipo/formato invalido (`season`, `min_matches`, csv malformed).
- `500 Internal Server Error`: fallo inesperado de calculo.

## 8) Reglas de compatibilidad (frontend)

- El envelope `generated_at/cached/cache_age_seconds/data` es obligatorio.
- `data.dimensions[*].key` es la fuente de verdad para tooltips (no hardcodear labels).
- El mapa choropleth debe colorear por `countries[*].coverage_total_pct`.
- Tooltip pais:
  - `coverage_total_pct`, `p0_pct`, `p1_pct`, `p2_pct`, `universe_tier`
  - lista de ligas del pais leyendo `leagues[*]` por `country_iso3`.

## 9) Nota de implementacion SQL

Para evitar doble conteo por joins 1:N (lineups, players, injuries), usar CTE por `match_id` y luego agregar por liga:

- `eligible_matches` (base)
- `dim_xg_match`, `dim_odds_match`, ..., `dim_standings_match`
- join final por `match_id` + agregacion por `league_id`

Esto garantiza `denominator` consistente y evita inflar `numerator`.
