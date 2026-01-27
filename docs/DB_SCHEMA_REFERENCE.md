# Database Schema Reference

> **IMPORTANTE**: Consultar este documento ANTES de escribir queries SQL para evitar errores de columnas inexistentes.
>
> Generado: 2026-01-26 (actualizado con tablas P2B + SOTA Sofascore + standings)

---

## public.matches (51 columnas)

| Column | Type | Nullable | Default |
|--------|------|----------|---------|
| `id` | integer | NO | nextval('matches_id_seq') |
| `external_id` | integer | NO | - |
| `date` | timestamp without time zone | NO | - |
| `league_id` | integer | NO | - |
| `season` | integer | NO | - |
| `home_team_id` | integer | NO | - |
| `away_team_id` | integer | NO | - |
| `home_goals` | integer | YES | - |
| `away_goals` | integer | YES | - |
| `stats` | json | YES | - |
| `status` | varchar | NO | - |
| `match_type` | varchar | NO | - |
| `match_weight` | double precision | NO | - |
| `odds_home` | double precision | YES | - |
| `odds_draw` | double precision | YES | - |
| `odds_away` | double precision | YES | - |
| `odds_recorded_at` | timestamp with time zone | YES | - |
| `opening_odds_home` | double precision | YES | - |
| `opening_odds_draw` | double precision | YES | - |
| `opening_odds_away` | double precision | YES | - |
| `opening_odds_recorded_at` | timestamp with time zone | YES | - |
| `xg_home` | double precision | YES | - |
| `xg_away` | double precision | YES | - |
| `xg_source` | varchar | YES | - |
| `sofascore_id` | integer | YES | - |
| `opening_odds_source` | varchar | YES | - |
| `lineup_confirmed` | boolean | YES | - |
| `home_formation` | varchar | YES | - |
| `away_formation` | varchar | YES | - |
| `home_lineup_surprise_index` | numeric | YES | - |
| `away_lineup_surprise_index` | numeric | YES | - |
| `home_missing_starter_count` | integer | YES | - |
| `away_missing_starter_count` | integer | YES | - |
| `home_minutes_share_missing` | numeric | YES | - |
| `away_minutes_share_missing` | numeric | YES | - |
| `lineup_features_computed_at` | timestamp without time zone | YES | - |
| `opening_odds_recorded_at_type` | varchar | YES | - |
| `opening_odds_kind` | varchar | YES | 'unknown' |
| `opening_odds_column` | varchar | YES | - |
| `market_movement_complete` | boolean | YES | false |
| `lineup_movement_tracked` | boolean | YES | false |
| `finished_at` | timestamp without time zone | YES | - |
| `stats_ready_at` | timestamp without time zone | YES | - |
| `stats_last_checked_at` | timestamp without time zone | YES | - |
| `events` | json | YES | - |
| `tainted` | boolean | YES | false |
| `tainted_reason` | varchar | YES | NULL |
| `venue_name` | varchar | YES | NULL |
| `venue_city` | varchar | YES | NULL |
| `elapsed` | integer | YES | - |
| `elapsed_extra` | integer | YES | - |

### Notas importantes - matches

- **Goles**: Usar `home_goals` y `away_goals` (NO `home_score`/`away_score`)
- **Status terminado**: `status IN ('FT', 'AET', 'PEN')`
- **Stats disponibles**: `stats IS NOT NULL AND stats::text != '{}' AND (stats->>'_no_stats') IS NULL`
- **Temporada 25/26**: `date >= '2025-08-01'`

### Definición de Odds Coverage

El campo `with_odds_pct` en Admin Panel representa el porcentaje de partidos con odds capturadas:

```sql
-- Definición: partido tiene odds si ANY de las 3 columnas tiene valor
COUNT(*) FILTER (WHERE odds_home IS NOT NULL) / COUNT(*)
```

**Columnas de odds en matches:**

| Columna | Descripción |
|---------|-------------|
| `odds_home`, `odds_draw`, `odds_away` | Odds más recientes (close) |
| `opening_odds_home/draw/away` | Odds de apertura |
| `odds_recorded_at` | Timestamp de última captura |
| `opening_odds_recorded_at` | Timestamp de odds apertura |
| `opening_odds_kind` | `'true_opening'`, `'earliest_available'`, `'unknown'` |

**Nota**: Para histórico detallado de movimientos, consultar tabla `odds_history` (no en este documento).

---

## public.teams (6 columnas)

| Column | Type | Nullable | Default |
|--------|------|----------|---------|
| `id` | integer | NO | nextval('teams_id_seq') |
| `external_id` | integer | NO | - |
| `name` | varchar | NO | - |
| `country` | varchar | YES | - |
| `team_type` | varchar | NO | - |
| `logo_url` | varchar | YES | - |

### Notas importantes - teams

- **team_type**: Valores posibles: `'club'`, `'national'`
- **external_id**: ID de API-Football
- **id**: ID interno (usado en `matches.home_team_id`, `matches.away_team_id`)

---

## public.predictions (18 columnas)

| Column | Type | Nullable | Default |
|--------|------|----------|---------|
| `id` | integer | NO | nextval('predictions_id_seq') |
| `match_id` | integer | NO | - |
| `model_version` | varchar | NO | - |
| `home_prob` | double precision | NO | - |
| `draw_prob` | double precision | NO | - |
| `away_prob` | double precision | NO | - |
| `created_at` | timestamp without time zone | NO | - |
| `is_frozen` | boolean | YES | false |
| `frozen_at` | timestamp with time zone | YES | - |
| `frozen_odds_home` | double precision | YES | - |
| `frozen_odds_draw` | double precision | YES | - |
| `frozen_odds_away` | double precision | YES | - |
| `frozen_ev_home` | double precision | YES | - |
| `frozen_ev_draw` | double precision | YES | - |
| `frozen_ev_away` | double precision | YES | - |
| `frozen_confidence_tier` | varchar | YES | - |
| `frozen_value_bets` | jsonb | YES | - |
| `run_id` | uuid | YES | - |

### Notas importantes - predictions

- **Relación**: `predictions.match_id` → `matches.id`
- **Frozen**: Predicción congelada antes del kickoff para evaluación
- **Probabilidades**: `home_prob + draw_prob + away_prob = 1.0`

---

## titan.feature_matrix (63 columnas)

| Column | Type | Nullable | Default |
|--------|------|----------|---------|
| `match_id` | bigint | NO | - |
| `kickoff_utc` | timestamp with time zone | NO | - |
| `competition_id` | integer | NO | - |
| `season` | integer | NO | - |
| `home_team_id` | bigint | NO | - |
| `away_team_id` | bigint | NO | - |
| `odds_home_close` | numeric | YES | - |
| `odds_draw_close` | numeric | YES | - |
| `odds_away_close` | numeric | YES | - |
| `implied_prob_home` | numeric | YES | - |
| `implied_prob_draw` | numeric | YES | - |
| `implied_prob_away` | numeric | YES | - |
| `odds_captured_at` | timestamp with time zone | YES | - |
| `form_home_last5` | varchar | YES | - |
| `form_away_last5` | varchar | YES | - |
| `goals_home_last5` | smallint | YES | - |
| `goals_away_last5` | smallint | YES | - |
| `goals_against_home_last5` | smallint | YES | - |
| `goals_against_away_last5` | smallint | YES | - |
| `points_home_last5` | smallint | YES | - |
| `points_away_last5` | smallint | YES | - |
| `form_captured_at` | timestamp with time zone | YES | - |
| `h2h_total_matches` | smallint | YES | - |
| `h2h_home_wins` | smallint | YES | - |
| `h2h_draws` | smallint | YES | - |
| `h2h_away_wins` | smallint | YES | - |
| `h2h_home_goals` | smallint | YES | - |
| `h2h_away_goals` | smallint | YES | - |
| `h2h_captured_at` | timestamp with time zone | YES | - |
| `pit_max_captured_at` | timestamp with time zone | NO | - |
| `outcome` | varchar | YES | - |
| `tier1_complete` | boolean | NO | false |
| `tier2_complete` | boolean | NO | false |
| `tier3_complete` | boolean | NO | false |
| `created_at` | timestamp with time zone | NO | now() |
| `updated_at` | timestamp with time zone | NO | now() |
| `xg_home_last5` | numeric | YES | - |
| `xg_away_last5` | numeric | YES | - |
| `xga_home_last5` | numeric | YES | - |
| `xga_away_last5` | numeric | YES | - |
| `npxg_home_last5` | numeric | YES | - |
| `npxg_away_last5` | numeric | YES | - |
| `xg_captured_at` | timestamp with time zone | YES | - |
| `tier1b_complete` | boolean | NO | false |
| `sofascore_lineup_available` | boolean | NO | false |
| `sofascore_home_formation` | varchar | YES | - |
| `sofascore_away_formation` | varchar | YES | - |
| `sofascore_lineup_captured_at` | timestamp with time zone | YES | - |
| `lineup_home_starters_count` | smallint | YES | - |
| `lineup_away_starters_count` | smallint | YES | - |
| `sofascore_lineup_integrity_score` | numeric | YES | - |
| `tier1c_complete` | boolean | NO | false |
| `xi_home_def_count` | smallint | YES | - |
| `xi_home_mid_count` | smallint | YES | - |
| `xi_home_fwd_count` | smallint | YES | - |
| `xi_away_def_count` | smallint | YES | - |
| `xi_away_mid_count` | smallint | YES | - |
| `xi_away_fwd_count` | smallint | YES | - |
| `xi_formation_mismatch_flag` | boolean | YES | false |
| `xi_depth_captured_at` | timestamp with time zone | YES | - |
| `tier1d_complete` | boolean | NO | false |

### Notas importantes - titan.feature_matrix

- **Schema**: `titan` (no `public`)
- **PK**: `match_id` (referencia `matches.id`)
- **Tiers**:
  - `tier1_complete`: Odds + Form + H2H
  - `tier1b_complete`: xG rolling
  - `tier1c_complete`: Lineups (Sofascore)
  - `tier1d_complete`: XI depth (posiciones)
  - `tier2_complete`: Features avanzados
  - `tier3_complete`: Features experimentales
- **Uso para cobertura**: `competition_id` = `matches.league_id`

---

## Queries de ejemplo

### Verificar schema antes de escribir SQL

```sql
SELECT column_name, data_type
FROM information_schema.columns
WHERE table_name = 'matches' AND table_schema = 'public'
ORDER BY ordinal_position;
```

### Partidos con goles (forma correcta)

```sql
SELECT home_goals, away_goals  -- NO home_score/away_score
FROM matches
WHERE status = 'FT'
```

### Join matches con teams

```sql
SELECT m.id, ht.name as home, at.name as away
FROM matches m
JOIN teams ht ON m.home_team_id = ht.id
JOIN teams at ON m.away_team_id = at.id
```

### TITAN coverage por liga

```sql
SELECT
    competition_id,
    COUNT(*) as total,
    SUM(CASE WHEN tier1_complete THEN 1 ELSE 0 END) as tier1
FROM titan.feature_matrix
WHERE season = 2025
GROUP BY competition_id
```

---

## public.admin_leagues (15 columnas)

| Column | Type | Nullable | Default |
|--------|------|----------|---------|
| `league_id` | integer | NO | - |
| `sport` | text | NO | 'football' |
| `name` | text | NO | - |
| `country` | text | YES | - |
| `kind` | text | NO | 'league' |
| `is_active` | boolean | NO | true |
| `priority` | text | YES | - |
| `match_type` | text | YES | - |
| `match_weight` | double precision | YES | - |
| `display_order` | integer | YES | - |
| `group_id` | integer | YES | - |
| `tags` | jsonb | NO | '{}' |
| `rules_json` | jsonb | NO | '{}' |
| `source` | text | NO | 'seed' |
| `created_at` | timestamptz | NO | now() |
| `updated_at` | timestamptz | NO | now() |

### Notas importantes - admin_leagues

- **is_active**: `TRUE` = liga servida a end-users (decisión de producto, todas las plataformas)
- **source**: `'seed'` (from COMPETITIONS), `'override'` (manual), `'observed'` (auto-discovered)
- **kind**: `'league'`, `'cup'`, `'international'`, `'friendly'`
- **configured**: `source IN ('seed', 'override')` en queries
- **tags.channels**: Opcional. Array de plataformas: `["ios","android","web"]`. Si no existe y is_active=true, se asume todas.
- **group_id**: FK a `admin_league_groups` para ligas pareadas (Apertura/Clausura)

### rules_json v1 Schema

El campo `rules_json` almacena reglas específicas de la liga. Schema v1:

| Campo | Tipo | Descripción |
|-------|------|-------------|
| `team_count_expected` | int (>0) | Equipos esperados, ej: 18, 20 |
| `season_model` | string | `"aug_jul"` o `"calendar"` |
| `promotion_relegation.promote` | int (>=0) | Equipos que ascienden |
| `promotion_relegation.relegate` | int (>=0) | Equipos que descienden |
| `promotion_relegation.playoffs` | bool | Si hay playoffs |
| `qualification.targets[]` | array | Lista de clasificaciones |
| `qualification.targets[].target_league_id` | int | Liga destino |
| `qualification.targets[].slots` | int (>=0) | Cupos |
| `qualification.targets[].note` | string | Nota opcional |
| `paired_handling` | string | `"grouped"` (default) o `"separate"` |

**Ejemplo completo:**

```json
{
  "team_count_expected": 20,
  "season_model": "aug_jul",
  "promotion_relegation": {
    "promote": 0,
    "relegate": 3,
    "playoffs": false
  },
  "qualification": {
    "targets": [
      {"target_league_id": 2, "slots": 4, "note": "Champions League"},
      {"target_league_id": 3, "slots": 2, "note": "Europa League"}
    ]
  },
  "paired_handling": "grouped"
}
```

**Notas:**
- Campos desconocidos se ignoran (forward compatibility)
- `{}` es válido (sin reglas)
- `paired_handling`: cómo tratar ligas pareadas en UI (`"grouped"` = mostrar como entidad única)

### Ejemplo de control por plataforma

```sql
-- Liga activa solo en iOS y Android (no web)
UPDATE admin_leagues
SET tags = jsonb_set(tags, '{channels}', '["ios","android"]')
WHERE league_id = 39;

-- Liga activa en todas las plataformas (default)
UPDATE admin_leagues
SET tags = tags - 'channels'  -- remove channels key = all platforms
WHERE league_id = 39;
```

---

## public.admin_league_groups (6 columnas)

| Column | Type | Nullable | Default |
|--------|------|----------|---------|
| `group_id` | serial | NO | nextval |
| `group_key` | text | NO | - |
| `name` | text | NO | - |
| `country` | text | YES | - |
| `tags` | jsonb | NO | '{}' |
| `created_at` | timestamptz | NO | now() |
| `updated_at` | timestamptz | NO | now() |

### Notas importantes - admin_league_groups

- **Uso**: Agrupar ligas pareadas (Apertura/Clausura)
- **group_key**: Identificador único, ej: `'URY_PRIMERA'`, `'PAR_PRIMERA'`
- **Relación**: `admin_leagues.group_id` → `admin_league_groups.group_id`

---

## public.admin_audit_log (8 columnas)

| Column | Type | Nullable | Default |
|--------|------|----------|---------|
| `id` | integer | NO | nextval('admin_audit_log_id_seq') |
| `entity_type` | text | NO | - |
| `entity_id` | text | NO | - |
| `action` | text | NO | - |
| `actor` | text | YES | - |
| `before_json` | jsonb | YES | - |
| `after_json` | jsonb | YES | - |
| `created_at` | timestamptz | NO | now() |

### Notas importantes - admin_audit_log

- **Inmutable**: No UPDATE ni DELETE en esta tabla
- **entity_type**: `'admin_leagues'`, `'admin_league_groups'`
- **entity_id**: ID como string (ej: `'39'` para league_id)
- **action**: `'update'`, `'create'`, `'delete'`
- **actor**: `'dashboard'`, `'api'`, `null` (sistema)
- **before_json/after_json**: Estado completo antes/después del cambio
- **Migración**: `migrations/admin_002_create_audit_log.sql`

---

## public.match_external_refs (6 columnas)

| Column | Type | Nullable | Default |
|--------|------|----------|---------|
| `match_id` | integer | NO | - |
| `source` | varchar | NO | - |
| `source_match_id` | varchar | NO | - |
| `confidence` | double precision | NO | - |
| `matched_by` | varchar | NO | - |
| `created_at` | timestamp without time zone | NO | now() |

### Notas importantes - match_external_refs

- **Uso**: Mapeo de partidos internos a IDs externos (Sofascore, Understat, etc.)
- **PK compuesta**: `(match_id, source)` — un ref por fuente por partido
- **source**: `'sofascore'`, `'understat'`, etc.
- **source_match_id**: ID del partido en la fuente externa (string)
- **confidence**: Score de confianza del matching (0.0 a 1.0)
- **matched_by**: Método de matching: `'auto'` (automático), `'review'` (manual/low-confidence)
- **Relación**: `match_external_refs.match_id` → `matches.id`
- **NOTA**: Reemplaza el campo legacy `matches.sofascore_id` para Sofascore refs

---

## public.match_sofascore_lineup (4 columnas)

| Column | Type | Nullable | Default |
|--------|------|----------|---------|
| `match_id` | integer | NO | - |
| `team_side` | varchar | NO | - |
| `formation` | varchar | NO | - |
| `captured_at` | timestamp without time zone | NO | now() |

### Notas importantes - match_sofascore_lineup

- **Uso**: Formación capturada de Sofascore pre-kickoff
- **team_side**: `'home'` o `'away'`
- **formation**: Ej: `'4-3-3'`, `'4-4-2'`
- **Timing**: Se captura ~1-2h antes del kickoff (job `sota_sofascore_xi_capture`)

---

## public.match_sofascore_player (9 columnas)

| Column | Type | Nullable | Default |
|--------|------|----------|---------|
| `match_id` | integer | NO | - |
| `team_side` | varchar | NO | - |
| `player_id_ext` | varchar | NO | - |
| `position` | varchar | NO | - |
| `is_starter` | boolean | NO | - |
| `rating_pre_match` | double precision | YES | - |
| `rating_recent_form` | double precision | YES | - |
| `minutes_expected` | integer | YES | - |
| `captured_at` | timestamp without time zone | NO | now() |

### Notas importantes - match_sofascore_player

- **Uso**: Jugadores individuales del XI capturado de Sofascore
- **player_id_ext**: ID del jugador en Sofascore (string)
- **position**: Ej: `'GK'`, `'CB'`, `'CM'`, `'ST'`
- **is_starter**: `true` = titular, `false` = suplente
- **Relación**: `match_sofascore_player.match_id` → `matches.id`

---

## public.league_standings (7 columnas)

| Column | Type | Nullable | Default |
|--------|------|----------|---------|
| `id` | integer | NO | nextval('league_standings_id_seq') |
| `league_id` | integer | NO | - |
| `season` | integer | NO | - |
| `standings` | json | YES | - |
| `captured_at` | timestamp without time zone | NO | - |
| `source` | varchar | NO | - |
| `expires_at` | timestamp without time zone | YES | - |

### Notas importantes - league_standings

- **Uso**: Tablas de posiciones por liga y temporada
- **standings**: JSON array con objetos por equipo (position, team_id, points, played, won, drawn, lost, goals_for, goals_against, goal_diff, form, group, description)
- **source**: `'api_football'`, `'no_table'` (sin datos disponibles)
- **team_id en standings JSON**: Es `external_id` de API-Football, NO `teams.id`
- **Usado por**: World Cup 2026 endpoints (`/dashboard/football/world-cup-2026/*`)

---

## public.job_runs (9 columnas)

| Column | Type | Nullable | Default |
|--------|------|----------|---------|
| `id` | integer | NO | nextval('job_runs_id_seq') |
| `job_name` | varchar | NO | - |
| `status` | varchar | NO | 'ok' |
| `started_at` | timestamp without time zone | NO | now() |
| `finished_at` | timestamp without time zone | YES | - |
| `duration_ms` | integer | YES | - |
| `error_message` | text | YES | - |
| `metrics` | jsonb | YES | - |
| `created_at` | timestamp without time zone | NO | now() |

### Notas importantes - job_runs

- **Uso**: Log de ejecución de jobs del scheduler
- **job_name**: Ej: `'global_sync'`, `'live_tick'`, `'stats_backfill'`, `'sota_sofascore_refs_sync'`, `'sota_sofascore_xi_capture'`
- **status**: `'ok'`, `'error'`, `'partial'`
- **metrics**: JSONB con métricas específicas del job (ej: `{"scanned": 84, "linked_auto": 49}`)
- **Usado por**: Dashboard ops (`/dashboard/ops.json`)

---

## public.odds_history (17 columnas)

| Column | Type | Nullable | Default |
|--------|------|----------|---------|
| `id` | integer | NO | nextval('odds_history_id_seq') |
| `match_id` | integer | NO | - |
| `recorded_at` | timestamptz | NO | now() |
| `odds_home` | double precision | YES | - |
| `odds_draw` | double precision | YES | - |
| `odds_away` | double precision | YES | - |
| `source` | varchar | YES | 'api_football' |
| `is_opening` | boolean | YES | false |
| `is_closing` | boolean | YES | false |
| `implied_home` | double precision | YES | - |
| `implied_draw` | double precision | YES | - |
| `implied_away` | double precision | YES | - |
| `overround` | double precision | YES | - |
| `quarantined` | boolean | YES | false |
| `quarantine_reason` | varchar | YES | NULL |
| `tainted` | boolean | YES | false |
| `taint_reason` | varchar | YES | NULL |

### Notas importantes - odds_history

- **Uso**: Historial detallado de movimientos de odds por partido
- **Relación**: `odds_history.match_id` → `matches.id`
- **is_opening/is_closing**: Marca si es la primera/última captura
- **implied_***: Probabilidades implícitas calculadas (1/odds)
- **overround**: Margen del bookmaker (suma de implied > 1.0)
- **quarantined/tainted**: Flags de calidad de datos
- **Usado por**: Endpoint `/matches/{id}/odds-history`

---

## Otras tablas (no documentadas en detalle)

| Tabla | Schema | Descripción |
|-------|--------|-------------|
| `alpha_progress_snapshots` | public | Snapshots de progreso alpha |
| `fastpath_ticks` | public | Ticks del pipeline fastpath (narrativas LLM) |
| `league_bookmaker_config` | public | Configuración de bookmakers por liga |
| `league_season_baselines` | public | Baselines por temporada/liga |
| `league_team_profiles` | public | Perfiles de equipos por liga |
| `lineup_movement_snapshots` | public | Snapshots de movimiento de lineups |
| `market_movement_snapshots` | public | Snapshots de movimiento de mercado |
| `match_lineups` | public | Lineups de API-Football |
| `match_odds_snapshot` | public | Snapshots de odds |
| `match_understat_team` | public | Stats xG de Understat |
| `match_weather` | public | Datos meteorológicos |
| `model_performance_logs` | public | Logs de performance del modelo ML |
| `model_snapshots` | public | Snapshots de modelos ML |
| `odds_snapshots` | public | Snapshots de odds (bulk) |
| `ops_alerts` | public | Alertas operacionales |
| `ops_audit_log` | public | Log de auditoría operacional |
| `ops_daily_rollups` | public | Rollups diarios de operaciones |
| `pit_reports` | public | Reportes PIT (Point-in-Time) |
| `player_stats_rolling` | public | Stats rolling de jugadores |
| `post_match_audits` | public | Auditorías post-partido |
| `prediction_outcomes` | public | Outcomes de predicciones |
| `prediction_performance_reports` | public | Reportes de performance |
| `prediction_reruns` | public | Re-ejecuciones de predicciones |
| `sensor_predictions` | public | Predicciones de sensores |
| `shadow_predictions` | public | Predicciones shadow model |
| `team_adjustments` | public | Ajustes por equipo |
| `team_home_city_profile` | public | Perfiles de ciudades (venue) |
| `team_overrides` | public | Overrides manuales de equipos |
| `team_values` | public | Valores de mercado de equipos |
| `unmapped_entities_backlog` | public | Backlog de entidades sin mapear |
| `venue_geo` | public | Geolocalización de venues |
| `titan.job_dlq` | titan | Dead letter queue de jobs TITAN |
| `titan.raw_extractions` | titan | Extracciones crudas TITAN |

Para consultar schema de estas tablas:

```sql
SELECT column_name, data_type, is_nullable
FROM information_schema.columns
WHERE table_name = '<tabla>' AND table_schema = 'public'
ORDER BY ordinal_position;
```
