# Database Schema Reference

> **IMPORTANTE**: Consultar este documento ANTES de escribir queries SQL para evitar errores de columnas inexistentes.
>
> Generado: 2026-01-26

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

## Otras tablas (menos usadas)

| Tabla | Schema | Descripción |
|-------|--------|-------------|
| `narratives` | public | Narrativas LLM post-partido |
| `odds_history` | public | Historial de odds por partido |
| `job_runs` | public | Log de jobs scheduler |

Para consultar schema de estas tablas:

```sql
SELECT column_name, data_type, is_nullable
FROM information_schema.columns
WHERE table_name = '<tabla>' AND table_schema = 'public'
ORDER BY ordinal_position;
```
