# Propuesta: Integración de Players & Managers

**Versión**: 2.1 (P0 fixes ABE re-auditoría + P1 Kimi observabilidad)
**Fecha**: 2026-02-07
**Autor**: Master (codificador backend)
**Estado**: APROBADO (ABE + Kimi) — LISTO PARA IMPLEMENTAR
**Auditorías incorporadas**: ABE (5 respuestas + re-auditoría P0), Kimi/ATI (5 puntos + validación v2.0), Master (análisis + contra-propuesta)

---

## Changelog v1.0 → v2.0

| Cambio | Origen | Motivo |
|--------|--------|--------|
| 7 tablas → 3 tablas MVP | Kimi + ABE + Master | Reducir scope Phase 1, ROI inmediato |
| Jobs en `player_jobs.py` (no `sota_jobs.py`) | Kimi + ABE | `sota_jobs.py` ya tiene 1,641 líneas |
| Manager sync diario (no semanal) | Kimi + Master | Reducir latencia de detección → PIT compliance |
| Features directo a Model A (no TITAN) | ABE | Quick win, menor fricción. TITAN cuando haya uplift demostrado |
| `key_player_missing` binario + counts | ABE | Binario robusto con poca data + `n_missing` para gradiente |
| Injuries como prioridad #1 | ABE + Kimi | Señal más directa, managers en paralelo |
| Cross-ref Sofascore↔AF diferido a Phase 2 | ABE + Master | No requisito MVP, multi-señal con tabla `player_identity_map` |
| `player_injuries` sin FK a `players` | ABE | Ingesta no depende del catálogo |
| `team_manager_history` con `team_id` interno | ABE | FK relacional interna, ext_id como debug column |
| Feature gate: 30d + gain threshold | Kimi + Master | Si gain < 0.05 en XGBoost tras 30d, reconsiderar Phase 2 |
| Tablas diferidas: `players`, `team_squads`, `player_season_stats`, `player_transfers` | Consenso | Phase 2+, cuando MVP demuestre valor |
| PIT policy explícita para backfills | ABE | Documentar reglas de compliance temporal |

### Changelog v2.0 → v2.1 (P0 fixes)

| Fix | Origen | Cambio |
|-----|--------|--------|
| PIT injuries: `captured_at < kickoff` (no `fixture_date`) | ABE P0-1 | Backfills NO PIT-safe para training. Solo sync recurrente es PIT-safe |
| `fixture_external_id NOT NULL` | ABE P0-2 | UNIQUE constraint requiere NOT NULL |
| `key_player_missing` diferido a Phase 2 | ABE P1 | events no da "starter", sesga contra defensas. MVP solo `n_missing`/`n_doubtful` |
| Índice parcial `WHERE end_date IS NULL` | ABE P1 | Mejor que `NULLS FIRST` para query DT actual |
| Job stagger documentado | Kimi P1 | Evitar contención de conexiones |
| Métricas de observabilidad | Kimi P1 | Calidad de datos para Feature Gate |
| Verificación SQL corregida | ABE P1 | Quitar referencia a `prediction_features` inexistente |

---

## 1. Motivación

FutbolStats opera a nivel de **equipo**. La granularidad más fina son los lineups de Sofascore (XI + ratings), pero sin modelo de datos para jugadores ni managers.

**Problema**: No podemos saber si el goleador está lesionado ni si cambió el DT esta semana — dos de los factores más predictivos en fútbol que **no están en nuestro feature set**.

**Hipótesis ML**: Bajas de jugadores clave y cambios de manager son señales de alto impacto predictivo con ROI inmediato.

---

## 2. Estado Actual

### Datos que YA tenemos

| Tabla | Filas | Contenido | Cobertura |
|-------|-------|-----------|-----------|
| `match_sofascore_player` | 10,694 | XI pre-kickoff (posición, is_starter) | 28 ligas, Ene 2026+ |
| `sofascore_player_rating_history` | ~35K (backfill 74%) | Rating post-match por jugador | 28 ligas, Nov 2025+ |
| `match_lineups` | 4,210 | XI + suplentes + coach_id | **Solo PL 2015-2021** (legacy) |
| `matches.events` (JSON) | ~100K+ | Goles, tarjetas, subs con player_id | Todas las ligas |

### Gaps que resuelve esta propuesta

1. **No hay datos de lesiones** — No sabemos si un jugador está lesionado/suspendido
2. **No hay tracking de managers** — No detectamos cambios de DT
3. **No hay catálogo de managers** — No tenemos metadata ni historial de carrera

---

## 3. Presupuesto API

| Concepto | Requests |
|----------|----------|
| Límite diario (Mega Plan) | 150,000 |
| Uso actual típico | ~5,000-9,000 |
| **Disponible** | **~141,000** |

### Costo MVP

| Concepto | Cálculo | Requests |
|----------|---------|----------|
| **Backfill inicial** | | |
| Injuries (28 ligas, temporada actual) | 28 × 1 | 28 |
| Coaches (28 ligas, ~384 equipos) | 384 × 1 | 384 |
| **Total backfill** | | **412** |
| | | |
| **Recurrente diario** | | |
| Injuries sync cada 6h | 28 × 4 | 112 |
| Manager sync diario | 384 × 1 | 384 |
| **Total diario** | | **~500** |

**0.3% del presupuesto diario**. Impacto negligible.

---

## 4. Arquitectura MVP: 3 Capas

### Principio rector
> **Ingestar amplio (raw_json), indexar lo necesario, computar métricas propias PIT-safe.**
> Nunca depender de métricas pre-calculadas de APIs como features ML primarios.

```
Capa 1: INGESTA (3 tablas, raw JSONB + columnas tipadas)
  └── player_injuries, team_manager_history, managers

Capa 2: MÉTRICAS PROPIAS (funciones PIT-safe)
  └── get_team_absences(), get_manager_context()

Capa 3: FEATURES ML (directo a Model A)
  └── n_missing, n_doubtful, is_new_manager, manager_tenure_days
```

---

## 5. Schema SQL — Phase 1 MVP (3 tablas)

### 5.1 Tabla `managers` (catálogo ligero)

```sql
CREATE TABLE managers (
    id SERIAL PRIMARY KEY,
    external_id INTEGER NOT NULL UNIQUE,          -- API-Football coach ID
    name VARCHAR(200) NOT NULL,
    first_name VARCHAR(100),
    last_name VARCHAR(100),
    birth_date DATE,
    nationality VARCHAR(100),
    photo_url TEXT,
    career JSONB,                                 -- [{team_id, team_name, start, end}, ...]
    raw_json JSONB,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE INDEX ix_managers_ext_id ON managers (external_id);
```

**Nota sobre `career`**: JSONB array porque el historial es variable-length. Las queries frecuentes (DT actual) usan `team_manager_history`, no este campo.

### 5.2 Tabla `team_manager_history` (stints DT-equipo)

```sql
CREATE TABLE team_manager_history (
    id SERIAL PRIMARY KEY,
    team_id INTEGER NOT NULL REFERENCES teams(id),
    manager_external_id INTEGER NOT NULL,
    manager_name VARCHAR(200) NOT NULL,
    start_date DATE NOT NULL,
    end_date DATE,                                -- NULL = actualmente en cargo
    -- Debug/denormalized
    team_external_id INTEGER,                     -- API-Football team ID (debug only)
    source VARCHAR(20) NOT NULL DEFAULT 'api-football',
    detected_at TIMESTAMP NOT NULL DEFAULT NOW(),
    UNIQUE (team_id, manager_external_id, start_date)
);

CREATE INDEX ix_tmh_team_current ON team_manager_history (team_id) WHERE end_date IS NULL;
CREATE INDEX ix_tmh_manager ON team_manager_history (manager_external_id);
```

**Decisión ABE**: `team_id` es FK interna (como descenso colombiano). `team_external_id` como columna denormalizada para debug.

**Uso ML**: `is_new_manager` se computa dinámicamente:
```sql
SELECT *, (match_date - start_date) AS tenure_days
FROM team_manager_history
WHERE team_id = :team_id AND end_date IS NULL;
-- is_new_manager = tenure_days < 60
```

### 5.3 Tabla `player_injuries` (lesiones por fixture)

```sql
CREATE TABLE player_injuries (
    id SERIAL PRIMARY KEY,
    player_external_id INTEGER NOT NULL,          -- API-Football player ID (sin FK a players)
    player_name VARCHAR(200) NOT NULL,
    team_id INTEGER REFERENCES teams(id),
    league_id INTEGER NOT NULL,
    season INTEGER NOT NULL,
    fixture_external_id INTEGER NOT NULL,           -- API-Football fixture ID (NOT NULL: UNIQUE constraint)
    match_id INTEGER REFERENCES matches(id),      -- Interno (resuelto post-sync)
    injury_type VARCHAR(50) NOT NULL,             -- 'Missing Fixture', 'Questionable', 'Doubtful'
    injury_reason VARCHAR(200),                   -- 'Ankle Injury', 'Muscle Injury', 'Suspended', etc.
    fixture_date TIMESTAMP,
    raw_json JSONB,
    captured_at TIMESTAMP NOT NULL DEFAULT NOW(),
    UNIQUE (player_external_id, fixture_external_id)
);

CREATE INDEX ix_injuries_match ON player_injuries (match_id) WHERE match_id IS NOT NULL;
CREATE INDEX ix_injuries_team_date ON player_injuries (team_id, fixture_date DESC);
CREATE INDEX ix_injuries_league_season ON player_injuries (league_id, season);
```

**Decisión ABE**: `player_external_id` sin FK a `players` — la ingesta no depende del catálogo. FK opcional en Phase 2 cuando el catálogo esté maduro.

**Nota**: `fixture_external_id` es NOT NULL porque participa en el UNIQUE constraint (ABE P0-2).

---

## 6. PIT Policy (ABE P0 — corregida v2.1)

### Regla general
> Un feature para el partido P debe usar SOLO información que **existía en nuestra DB** antes del kickoff de P.

### Modo PIT-estricto (default del proyecto)

El criterio PIT es `captured_at < kickoff`, NO `fixture_date`. Esto porque:
- El sync recurrente (cada 6h) captura injuries ANTES del partido → `captured_at` es naturalmente pre-kickoff → **PIT-safe**
- Un backfill histórico captura injuries DESPUÉS del partido → `captured_at` es posterior → **NO PIT-safe**
- Usar `fixture_date` como proxy PIT permitiría leakage: datos que no teníamos en el momento de la predicción

### Aplicación por tabla

| Tabla | Regla PIT-estricto | Enforcement |
|-------|-------------------|-------------|
| `player_injuries` | `captured_at < match_kickoff` | Query filter en `get_team_absences()` |
| `team_manager_history` | `start_date <= match_date` AND (`end_date IS NULL` OR `end_date > match_date`) AND `detected_at < match_kickoff` | Query filter en `get_manager_context()` |
| `managers.career` | No se usa como feature directo — solo referencia | N/A |

### PIT en backfills (CRÍTICO)

**Los backfills históricos de injuries NO son PIT-safe para training/evaluación.**

- Backfill: 1 call trae toda la temporada → `captured_at = NOW()` para todos los registros
- Estos datos son útiles para: cobertura operacional, observabilidad, dashboard, debug
- Estos datos **NO se usan** para: training de modelo, evaluación retrospectiva, feature engineering de partidos pasados
- Para features de partidos futuros (operación real), el sync recurrente cada 6h captura las injuries con `captured_at` pre-kickoff → PIT-safe

**Modo PIT-pragmático** (solo exploración offline):
- Habilitado explícitamente vía flag `pit_mode='pragmatic'`
- Permite usar `fixture_date` como proxy PIT para análisis exploratorios
- NUNCA en producción ni evaluación formal

**Invariante PIT-estricto**: `get_team_absences(match_id=X)` en modo estricto retorna SOLO injuries con `captured_at` anterior al kickoff de X.

---

## 7. Jobs de Ingesta — `app/etl/player_jobs.py` (nuevo módulo)

### 7.1 `sync_injuries` — Lesiones (PRIORIDAD #1)

- **Frecuencia**: Cada 6 horas
- **Endpoint**: `GET /injuries?league={id}&season={year}`
- **Lógica**:
  1. Para cada liga en `SOFASCORE_SUPPORTED_LEAGUES` (28 ligas):
     - GET todas las lesiones de la temporada actual (1 call = toda la liga)
     - Para cada injury: resolver `team_id` interno vía `teams.external_id`
     - Resolver `match_id` cruzando `fixture_external_id` con `matches.external_id`
     - UPSERT en `player_injuries` (ON CONFLICT DO UPDATE para correcciones)
  2. Métricas: injuries_inserted, injuries_updated, leagues_synced, errors
- **Costo**: 28 requests × 4/día = **112 req/día**
- **Config**: `INJURIES_SYNC_ENABLED=true`, `INJURIES_SYNC_INTERVAL_HOURS=6`
- **Rate limit**: Reutiliza `APIFootballProvider` (respeta 600 req/min del plan Mega)

### 7.2 `sync_managers` — DTs + detección de cambios

- **Frecuencia**: Diaria
- **Endpoint**: `GET /coachs?team={id}`
- **Lógica**:
  1. Para cada equipo activo (~560 equipos en 28 ligas, derivados de matches últimos 180d):
     - GET coach actual
     - UPSERT en `managers` (catálogo)
     - Comparar `manager_external_id` con último registro en `team_manager_history` para este `team_id`
     - Si cambió:
       a. Cerrar registro anterior: `UPDATE ... SET end_date = :today WHERE team_id = :tid AND end_date IS NULL`
       b. Insertar nuevo: `INSERT INTO team_manager_history (team_id, manager_external_id, manager_name, start_date, ...)`
       c. Log: `[MANAGER_CHANGE] {team_name}: {old_manager} → {new_manager}`
  2. Métricas: managers_synced, changes_detected, errors
- **Costo**: ~560 req/día (equipos activos con partidos en últimos 180d)
- **Config**: `MANAGER_SYNC_ENABLED=true`, `MANAGER_SYNC_INTERVAL_HOURS=24`
- **Rate limit**: Reutiliza `APIFootballProvider._rate_limited_request()` (respeta 1 req/s + backoff)

### 7.3 Schedule Stagger (Kimi P1)

Para evitar contención de conexiones Railway con otros jobs:

| Job | Horario UTC | Frecuencia |
|-----|-------------|------------|
| `sync_managers` | 02:00 | Diario |
| `sync_injuries` | 06:00, 12:00, 18:00, 00:00 | Cada 6h |

No colisiona con `global_sync` (cada 1 min), `stats_backfill` (cada 60 min), ni `odds_sync` (cada 6h offset).

---

## 8. Capa 2: Métricas Propias — `app/features/player_features.py` (nuevo módulo)

### 8.1 `get_team_absences()`

```python
async def get_team_absences(
    session, team_id: int, match_id: int, match_kickoff: datetime
) -> dict:
    """
    Ausencias conocidas para un equipo en un partido (PIT-estricto).

    Returns:
        n_missing: int       — injury_type = 'Missing Fixture'
        n_doubtful: int      — 'Doubtful' + 'Questionable'
        injured_names: list[str]  — Para debug/logs, no como feature
    """
```

**Nota**: `key_player_missing` diferido a Phase 2 (ABE P1: `events` no indica "starter" y sesga contra defensas/GK). MVP arranca solo con counts (`n_missing`, `n_doubtful`).

**Query PIT-estricto**:
```sql
SELECT player_external_id, player_name, injury_type, injury_reason
FROM player_injuries
WHERE team_id = :team_id
  AND fixture_external_id = :fixture_ext_id   -- Exacto al fixture
  AND captured_at < :match_kickoff            -- PIT-estricto (ABE P0-1)
```

### 8.2 `get_manager_context()`

```python
async def get_manager_context(
    session, team_id: int, match_kickoff: datetime
) -> dict:
    """
    Contexto del DT para un equipo en un partido (PIT-estricto).

    Args:
        match_kickoff: timestamp del kickoff (no date, para evitar leakage intra-día)

    Returns:
        manager_name: str
        manager_tenure_days: int   — días desde start_date
        is_new_manager: bool       — tenure < 60 días
        manager_games: int         — partidos del equipo desde start_date (calculado de matches)
    """
```

**Query PIT-estricto**:
```sql
SELECT manager_external_id, manager_name, start_date,
       (:match_kickoff::date - start_date) AS tenure_days
FROM team_manager_history
WHERE team_id = :team_id
  AND start_date <= :match_kickoff::date
  AND (end_date IS NULL OR end_date > :match_kickoff::date)
  AND detected_at < :match_kickoff           -- PIT-estricto (ABE checklist)
ORDER BY start_date DESC
LIMIT 1;
```

---

## 9. Capa 3: Features ML — Directo a Model A

### Phase 1 Features (MVP) — 9 features (4 injuries × home/away + 4 managers × home/away + 1 flag)

| Feature | Tipo | Función fuente | Imputación si falta |
|---------|------|---------------|-------------------|
| `home_n_missing` | int | `get_team_absences()` | 0 (asume sin bajas reportadas) |
| `away_n_missing` | int | `get_team_absences()` | 0 |
| `home_n_doubtful` | int | `get_team_absences()` | 0 |
| `away_n_doubtful` | int | `get_team_absences()` | 0 |
| `home_manager_tenure_days` | int | `get_manager_context()` | 365 (asume manager estable) |
| `away_manager_tenure_days` | int | `get_manager_context()` | 365 |
| `home_is_new_manager` | bool→int(0/1) | `get_manager_context()` | 0 |
| `away_is_new_manager` | bool→int(0/1) | `get_manager_context()` | 0 |
| `player_manager_missing` | int(0/1) | `get_player_manager_features()` | 1 (sin datos = missing) |

**Nota**: `n_doubtful` cuenta tanto `'Questionable'` como `'Doubtful'` de la API. `player_manager_missing` = 1 cuando no hay **ninguna** fila de injuries ni manager para el partido (derivado de presencia de filas en DB, no de valores imputados).

**Diferido a Phase 2**: `key_player_missing` (requiere cross-ref AF↔Sofascore o `player_season_stats` para definir "titular habitual" sin sesgo).

**Destino**: Directo al pipeline de Model A (no TITAN). Se agregan al feature vector en `engineering.py` junto a los features existentes (decisión ABE).

**Integración a TITAN**: Cuando haya uplift demostrado y histórico suficiente, materializar en `feature_matrix`. No antes.

### Feature Gate (30 días)

Después de 30 días con datos de injuries + managers:
1. Entrenar modelo con y sin los 6 nuevos features
2. Medir feature importance (Gain) en XGBoost para `n_missing` y `is_new_manager`
3. **Si ninguno de los 4 features base tiene gain ≥ 0.05**: Reconsiderar inversión en Phase 2
4. **Si gain ≥ 0.05**: Proceder con Phase 2 (cross-reference, `key_player_missing`, stats, transfers)

---

## 10. Phase 2+ (Diferido — NO implementar ahora)

Estas tablas y features se implementan SOLO si Phase 1 demuestra uplift.

| Tabla | Propósito | Requisito |
|-------|-----------|-----------|
| `players` | Catálogo maestro (edad, nacionalidad, posición) | Feature gate passed |
| `player_identity_map` | Cross-ref Sofascore↔API-Football (confidence, method, overrides) | ABE: tabla separada, no columna |
| `team_squads` | Plantilla actual por equipo | Feature gate passed |
| `player_season_stats` | Stats por temporada (goles, asistencias, minutos) | Feature gate passed |
| `player_transfers` | Historial de transferencias | Feature gate passed |

| Feature | Tipo | Dependencia |
|---------|------|-------------|
| `key_player_missing` | bool→int(0/1) | `player_identity_map` + `player_season_stats` (define "titular" sin sesgo) |
| `missing_minutes_pct` | float | `player_season_stats` para minutos esperados |
| `squad_stability_index` | float | `match_sofascore_player` (ya existe) |
| `xi_avg_rating` | float | `sofascore_player_rating_history` (ya existe, no consumido) |
| `manager_win_rate` | float | Acumulación de matches >30 por DT |

---

## 11. Secuencia de Implementación — Phase 1

| Paso | Qué | Archivos | Costo |
|------|-----|----------|-------|
| **1** | Migración SQL: 3 tablas | `scripts/migrate_players_managers.sql` | 0 req |
| **2** | `sync_injuries` job | `app/etl/player_jobs.py` (nuevo) | 28 req/run |
| **3** | `sync_managers` job + change detection | `app/etl/player_jobs.py` | ~560 req/run |
| **4** | Registrar 2 jobs en scheduler | `app/scheduler.py` | 0 req |
| **5** | Config vars en `app/config.py` | `app/config.py` | 0 req |
| **6** | `get_team_absences()` + `get_manager_context()` | `app/features/player_features.py` (nuevo) | 0 req |
| **7** | Integrar features a Model A pipeline | `app/features/engineering.py` | 0 req |
| **8** | Backfill injuries (temporada actual, 28 ligas) | Manual o script | 28 req |
| **9** | Backfill managers (~560 equipos activos) | Manual o script | ~560 req |

**Timeline estimado**: 3-5 días de desarrollo.

---

## 12. Verificación Post-Implementación

### 12.1 Cobertura de datos

```sql
-- 1. Injuries coverage
SELECT COUNT(DISTINCT league_id) AS leagues_covered,
       COUNT(*) AS total_injuries
FROM player_injuries;
-- Expected: 28 leagues, >500 injuries

-- 2. Manager history
SELECT COUNT(*) AS total_stints,
       COUNT(*) FILTER (WHERE end_date IS NOT NULL) AS completed_stints
FROM team_manager_history;
-- Expected: >384 stints (1 per team minimum)

-- 3. Manager changes detected
SELECT team_id, manager_name, start_date, end_date
FROM team_manager_history
WHERE end_date IS NOT NULL
ORDER BY end_date DESC LIMIT 10;
-- Expected: Recent manager changes visible
```

### 12.2 PIT invariance test

```sql
-- Verificar que NO hay injuries backfilleadas usadas como PIT-safe:
-- injuries con captured_at DESPUÉS del kickoff del fixture asociado
SELECT COUNT(*) AS backfilled_injuries_post_kickoff
FROM player_injuries pi
JOIN matches m ON m.id = pi.match_id
WHERE pi.captured_at > m.date
  AND m.status IN ('FT', 'AET', 'PEN');
-- Estos registros son válidos para operacional/coverage pero NO para training
```

### 12.3 Manager PIT integrity (Kimi P0)

```sql
-- end_date nunca debe ser posterior a detected_at
SELECT COUNT(*) AS integrity_violations
FROM team_manager_history
WHERE end_date > detected_at;
-- Expected: 0
```

### 12.4 Observabilidad — Métricas de calidad (Kimi P1)

```sql
-- 1. Coverage de injuries por liga (¿todas reportan igual?)
SELECT pi.league_id, al.name,
       COUNT(*) AS total_injuries,
       COUNT(DISTINCT pi.fixture_external_id) AS fixtures_covered
FROM player_injuries pi
LEFT JOIN admin_leagues al ON al.league_id = pi.league_id
GROUP BY pi.league_id, al.name
ORDER BY total_injuries ASC;

-- 2. Lag de detección de cambios de manager
SELECT AVG(detected_at::date - start_date) AS avg_detection_lag_days,
       MAX(detected_at::date - start_date) AS max_detection_lag_days
FROM team_manager_history
WHERE start_date IS NOT NULL;
-- Expected: avg < 2 days para sync diario

-- 3. Feature integration (log del scheduler o debug endpoint)
-- Verificar en logs de producción:
-- [INJURIES_SYNC] leagues=28, inserted=X, updated=Y, errors=0
-- [MANAGER_SYNC] teams=384, changes=Z, errors=0
```

---

## 13. Resumen de Decisiones Auditadas

| Decisión | Respondido por | Respuesta |
|----------|---------------|-----------|
| `player_injuries` FK a `players`? | ABE | **No en MVP.** Solo `player_external_id`. FK opcional Phase 2. |
| Cross-ref Sofascore↔AF: columna o tabla? | ABE | **Tabla separada** `player_identity_map` (Phase 2). Multi-señal con confidence. |
| `team_manager_history` ID interno o externo? | ABE | **`team_id` interno** (FK). `team_external_id` como columna debug. |
| Jobs en `sota_jobs.py` o módulo nuevo? | ABE + Kimi | **Módulo nuevo** `player_jobs.py`. |
| Features a TITAN o Model A? | ABE | **Model A directo.** TITAN cuando haya uplift demostrado. |
| `key_player_missing` binario o cuantitativo? | ABE | **Diferido a Phase 2.** `events` no da "starter", sesga contra defensas. MVP solo `n_missing`/`n_doubtful`. |
| Prioridad injuries vs managers? | ABE + Kimi | **Injuries primero.** Managers en paralelo si no retrasa. |
| `squad_stability_index`? | ABE + Kimi | **Phase 2.** Probar injuries/manager primero. |
| Fuzzy matching suficiente para cross-ref? | ABE | **No como fundamento.** Multi-señal (equipo+nombre+DOB). Phase 2. |
| Scope: 7 tablas o MVP reducido? | ABE + Kimi + Master | **3 tablas MVP.** 4 diferidas a Phase 2. |
| Manager sync frecuencia? | Kimi + Master | **Diaria** (384 req/día, PIT compliance). |
| Feature gate? | Kimi + Master | **30 días, gain ≥ 0.05** para proceder a Phase 2. |
| PIT injuries: `fixture_date` o `captured_at`? | ABE re-auditoría P0-1 | **`captured_at < kickoff`** (PIT-estricto). Backfills NO PIT-safe para training. |
| `fixture_external_id` nullable? | ABE re-auditoría P0-2 | **NOT NULL** (requerido por UNIQUE constraint). |
| Job stagger? | Kimi validación P1 | **Managers 02:00 UTC, Injuries cada 6h offset.** Evitar contención. |
| Métricas de observabilidad? | Kimi validación P1 | **Sí.** Coverage por liga, lag de detección, integridad PIT. |
