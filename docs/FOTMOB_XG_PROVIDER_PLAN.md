# Plan: FotMob xG Provider (TITAN Ingestion)

> **Estado**: Aprobado por ABE (2026-02-08) con correcciones P0 incorporadas + P1 PIT guard aceptado.
> **Split**: PR1 (Fases 1-5, ingesta) + PR2 (Fase 6, feature matrix).

---

## Contexto

El modelo v1.0.1-league-only NO tiene señal xG para ligas LATAM (Argentina, Colombia, Ecuador, Paraguay, etc.) porque Understat solo cubre 5 ligas europeas y SofaScore tiene 0 xG para Argentina (confirmado: key `expectedGoals` ausente en raw_stats). FBref está muerto (Opta revocó datos Jan-2026).

**Hallazgo clave**: FotMob tiene xG completo (Opta) para TODAS las ligas, incluidas LATAM. Verificado con 3 partidos reales de Argentina — datos a nivel equipo, jugador y tiro con 17 decimales de precisión.

**ABE aprobó Option B**: FotMob SOLO para ligas SIN Understat. Fuente fija por liga (Understat=top-5 EUR, FotMob=resto). Sin mezclar dentro de una liga.

**Objetivo**: Construir el provider FotMob siguiendo la arquitectura de SofaScore (resiliencia, proxy, rate-limit, best-effort), almacenar xG en nueva tabla, y extender feature_matrix para consumir ambas fuentes de xG.

---

## Guardrails ABE P0

| # | Guardrail | Implementación |
|---|-----------|----------------|
| P0-1 | Scheduler-only, no request-path | Jobs en scheduler.py, nunca llamado desde endpoints |
| P0-2 | Match score >= 0.90 auto-link | `calculate_match_score()` reutilizado de sofascore_provider |
| P0-3 | Parse por key name, no index | Buscar `"expected_goals"` en stats array por `title`/`key` |
| P0-4 | PIT: `captured_at = match_date + 6h` (backfill) | En upsert, `captured_at` explícito |
| P0-5 | 1 req/s + circuit breaker | Rate limiter + skip on 5 errores consecutivos |
| P0-6 | Solo team-level xG/xGOT + raw_stats JSONB | No shotmap, no player xG en esta fase |
| P0-7 | Feature flags per-phase + league allowlist | `FOTMOB_REFS_ENABLED` + `FOTMOB_XG_ENABLED` + `FOTMOB_XG_LEAGUES` |
| P0-8 | Solo ligas confirmadas ejecutan (ABE review P0-1) | Jobs: `eligible = parsed_config ∩ FOTMOB_CONFIRMED_XG_LEAGUES` |
| P0-9 | Fase 6 sin columnas inexistentes (ABE review P0-2) | xga derivada del rival, npxg=NULL con guard Decimal-safe, league_id determinista |

---

## Fase 1: Configuración y Constantes

### 1A. Settings en `app/config.py`

Después de línea 52 (tras `MARKET_ANCHOR_MIN_SAMPLES`), agregar:

```python
# SOTA: FotMob xG provider (ABE P0 2026-02-08)
FOTMOB_REFS_ENABLED: bool = False         # Flag refs sync (Phase A)
FOTMOB_XG_ENABLED: bool = False           # Flag xG backfill (Phase B) — requires refs
FOTMOB_XG_LEAGUES: str = "128"            # P0-8: default SOLO Argentina; expandir post-verificación
FOTMOB_PROXY_URL: str = ""                # IPRoyal proxy (shared or separate from SofaScore)
FOTMOB_RATE_LIMIT_SECONDS: float = 1.5    # Slightly more conservative than SofaScore
FOTMOB_CIRCUIT_BREAKER_THRESHOLD: int = 5 # Consecutive errors before skip
```

**ABE P1**: Flags separados (`REFS_ENABLED` / `XG_ENABLED`) permiten activar ingesta por fases sin acoplamiento.
**ABE P0-8**: `FOTMOB_XG_LEAGUES` default "128" (solo Argentina). Expandir a más ligas requiere verificar FotMob ID + confirmar xG availability.

~6 líneas nuevas.

### 1B. Constantes en `app/etl/sota_constants.py`

Después de línea 95 (tras `LEAGUE_PROXY_COUNTRY`), agregar:

```python
# FotMob league ID mapping (API-Football ID -> FotMob league ID)
LEAGUE_ID_TO_FOTMOB: dict[int, int] = {
    128: 112,   # Argentina Primera División (CONFIRMED)
    71: 268,    # Brazil Serie A  (TBD: verify)
    239: 11535, # Colombia Primera A (TBD: verify)
    250: 14056, # Paraguay Apertura (TBD: verify)
    252: 14056, # Paraguay Clausura (TBD: verify)
    265: 11653, # Chile Primera División (TBD: verify)
    242: 14064, # Ecuador Liga Pro (TBD: verify)
    268: 13475, # Uruguay Apertura (TBD: verify)
    270: 13475, # Uruguay Clausura (TBD: verify)
    281: 14070, # Perú Liga 1 (TBD: verify)
    299: 17015, # Venezuela Primera División (TBD: verify)
    344: 15736, # Bolivia Primera División (TBD: verify)
    253: 130,   # MLS (TBD: verify)
    262: 230,   # Mexico Liga MX (TBD: verify)
    307: 955,   # Saudi Pro League (TBD: verify)
    88: 57,     # Eredivisie (TBD: verify)
    94: 61,     # Primeira Liga (TBD: verify)
    144: 54,    # Belgian Pro League (TBD: verify)
    203: 71,    # Süper Lig (TBD: verify)
    40: 47,     # EFL Championship (TBD: verify)
    2: 42,      # Champions League (TBD: verify)
    3: 73,      # Europa League (TBD: verify)
    848: 10216, # Conference League (TBD: verify)
}

# Start conservative: only confirmed leagues
FOTMOB_CONFIRMED_XG_LEAGUES: set[int] = {
    128,  # Argentina Primera División (CONFIRMED 2026-02-08)
}
```

**ABE P0-8 enforcement**: Los mappings TBD quedan en el dict para referencia futura, pero los jobs NUNCA procesan una liga que no esté en `FOTMOB_CONFIRMED_XG_LEAGUES`:

```python
# En cada job: eligible = config ∩ confirmed
parsed_leagues = {int(x) for x in settings.FOTMOB_XG_LEAGUES.split(",") if x.strip()}
eligible_leagues = parsed_leagues & FOTMOB_CONFIRMED_XG_LEAGUES
# Solo 'eligible_leagues' se procesan. TBD leagues se ignoran silenciosamente.
```

~35 líneas nuevas.

---

## Fase 2: DB Migration

### Nuevo archivo: `scripts/migrations/0XX_add_fotmob_stats_table.py`

Tabla modelada como `match_sofascore_stats` pero enfocada en xG:

```sql
CREATE TABLE IF NOT EXISTS match_fotmob_stats (
    match_id INTEGER NOT NULL PRIMARY KEY
        REFERENCES matches(id) ON DELETE CASCADE,
    xg_home DOUBLE PRECISION,
    xg_away DOUBLE PRECISION,
    xgot_home DOUBLE PRECISION,
    xgot_away DOUBLE PRECISION,
    xg_open_play_home DOUBLE PRECISION,
    xg_open_play_away DOUBLE PRECISION,
    xg_set_play_home DOUBLE PRECISION,
    xg_set_play_away DOUBLE PRECISION,
    raw_stats JSONB,
    captured_at TIMESTAMP NOT NULL DEFAULT NOW(),
    source_version VARCHAR(50) DEFAULT 'fotmob_opta_v1'
);

CREATE INDEX IF NOT EXISTS ix_match_fotmob_stats_captured_at
ON match_fotmob_stats (captured_at);
```

~20 líneas. Sigue patrón de migration 032.

---

## Fase 3: Provider Class

### Nuevo archivo: `app/etl/fotmob_provider.py` (~250 líneas)

Sigue la arquitectura de `app/etl/sofascore_provider.py`:

### Estructura

```python
class FotmobProvider:
    """Provider for FotMob xG data (team-level).
    ABE P0: scheduler-only, 1 req/s, circuit breaker, parse by key name.
    """
    SCHEMA_VERSION = "fotmob.xg.v1"

    def __init__(self):
        self._client: httpx.AsyncClient | None = None
        self._last_request_time: float = 0
        self._consecutive_errors: int = 0
        self._proxy_url = os.environ.get("FOTMOB_PROXY_URL") or os.environ.get("SOFASCORE_PROXY_URL")
```

### Métodos principales

| Método | Descripción | Patrón base |
|--------|-------------|-------------|
| `_get_client(cc)` | httpx client con proxy geo | `SofascoreProvider._get_client()` |
| `_build_geo_proxy_url(cc)` | IPRoyal `_country-{cc}` suffix | `SofascoreProvider._build_geo_proxy_url()` |
| `_rate_limit()` | Enforce 1.5s entre requests | `SofascoreProvider._rate_limit()` |
| `_fetch_json(url, match_id, cc)` | GET con retry/backoff + circuit breaker (P0-5) | `SofascoreProvider._fetch_json()` + circuit breaker |
| `_build_headers()` | Headers FotMob + x-fm-req signature | **NUEVO** |
| `get_match_xg(fotmob_id, cc)` | Fetch + parse xG de matchDetails | **NUEVO** |
| `get_league_fixtures(fotmob_league_id, cc)` | Fetch fixtures for match linking | **NUEVO** |
| `_parse_xg_stats(data)` | Extraer xG por key name (P0-3) | **NUEVO** |
| `close()` | Cerrar clientes | `SofascoreProvider.close()` |

### Header x-fm-req (anti-scraping)

FotMob requiere un header `x-fm-req` generado con MD5+Base64. La librería `soccerdata` (PR#745) resolvió esto. Implementaremos la misma lógica.

### Parsing xG (P0-3: por key name, NUNCA por índice)

```python
def _parse_xg_stats(self, data: dict) -> FotmobXGData | None:
    """Parse xG from matchDetails response.
    P0-3: NEVER parse by index, always find by title/key name.
    """
    stats_section = data.get("content", {}).get("stats", {})
    all_stats = stats_section.get("Ede", {}).get("stats", [])

    xg_map = {}
    for group in all_stats:
        for stat in group.get("stats", []):
            key = (stat.get("title") or stat.get("key") or "").lower()
            values = stat.get("stats", [])
            if len(values) >= 2:
                xg_map[key] = (values[0], values[1])

    return FotmobXGData(
        xg_home=_safe_float(xg_map.get("expected_goals", (None,None))[0]),
        xg_away=_safe_float(xg_map.get("expected_goals", (None,None))[1]),
        # ... xgot, open_play, set_play similarly
    )
```

### Dataclass

```python
@dataclass
class FotmobXGData:
    xg_home: float | None = None
    xg_away: float | None = None
    xgot_home: float | None = None
    xgot_away: float | None = None
    xg_open_play_home: float | None = None
    xg_open_play_away: float | None = None
    xg_set_play_home: float | None = None
    xg_set_play_away: float | None = None
    raw_stats: dict | None = None
```

### Circuit Breaker (P0-5)

```python
if self._consecutive_errors >= CIRCUIT_BREAKER_THRESHOLD:
    logger.warning("[FOTMOB] Circuit breaker OPEN, skipping")
    return None, "circuit_breaker_open"
# Reset on success: self._consecutive_errors = 0
```

---

## Fase 4: Jobs en `app/etl/sota_jobs.py`

Dos funciones nuevas al final del archivo:

### 4A. `sync_fotmob_refs()` — Match Linking

Reutiliza `calculate_match_score()` de `app/etl/sofascore_provider.py`:

```python
async def sync_fotmob_refs(session, days=7, limit=200) -> dict:
    """Link matches to FotMob match IDs for xG-eligible leagues.
    P0-2: Auto-link at score >= 0.90, needs_review at 0.75-0.90.
    """
```

**Lógica:**
1. **P0-8**: `eligible_leagues = parsed_config ∩ FOTMOB_CONFIRMED_XG_LEAGUES` — solo ligas verificadas
2. Para cada liga eligible: validar que `LEAGUE_ID_TO_FOTMOB[league_id]` existe, sino skip
3. Query matches FT sin ref `source='fotmob'` en `match_external_refs`, filtrar por eligible
4. Para cada liga: fetch FotMob fixtures (`GET /api/leagues?id={fm_id}`)
5. Para cada match FotMob FT: `calculate_match_score()` contra matches internos
6. Auto-link si score >= 0.90 → INSERT en `match_external_refs(source='fotmob')`
7. Return metrics dict: `{scanned, linked_auto, linked_review, skipped_no_match, skipped_no_mapping, errors}`

~100 líneas.

### 4B. `backfill_fotmob_xg_ft()` — xG Capture

```python
async def backfill_fotmob_xg_ft(session, days=7, limit=100) -> dict:
    """Fetch xG from FotMob for linked FT matches.
    P0-4: captured_at = match_date + 6h for backfill.
    P0-6: Only team-level xG/xGOT.
    """
```

**Lógica:**
1. Query matches FT con ref `source='fotmob'` pero SIN entry en `match_fotmob_stats`
2. Para cada: `provider.get_match_xg(fotmob_id, country_code)`
3. Upsert en `match_fotmob_stats` con `ON CONFLICT (match_id) DO UPDATE`
4. PIT: `captured_at = match.date + timedelta(hours=6)` (P0-4)
5. Return metrics dict: `{scanned, captured, skipped_no_xg, errors}`

~100 líneas.

### Upsert Pattern (sigue Sofascore Stats — ON CONFLICT + ABE P1 CAST)

```python
await session.execute(text("""
    INSERT INTO match_fotmob_stats (
        match_id, xg_home, xg_away, xgot_home, xgot_away,
        xg_open_play_home, xg_open_play_away,
        xg_set_play_home, xg_set_play_away,
        raw_stats, captured_at, source_version
    ) VALUES (
        :match_id, :xg_home, :xg_away, :xgot_home, :xgot_away,
        :xg_open_play_home, :xg_open_play_away,
        :xg_set_play_home, :xg_set_play_away,
        CAST(:raw_stats AS jsonb), :captured_at, :source_version
    )
    ON CONFLICT (match_id) DO UPDATE SET
        xg_home = EXCLUDED.xg_home, xg_away = EXCLUDED.xg_away,
        xgot_home = EXCLUDED.xgot_home, xgot_away = EXCLUDED.xgot_away,
        xg_open_play_home = EXCLUDED.xg_open_play_home,
        xg_open_play_away = EXCLUDED.xg_open_play_away,
        xg_set_play_home = EXCLUDED.xg_set_play_home,
        xg_set_play_away = EXCLUDED.xg_set_play_away,
        raw_stats = EXCLUDED.raw_stats,
        captured_at = EXCLUDED.captured_at
"""), params)
```

**ABE P1**: `CAST(:raw_stats AS jsonb)` explícito para evitar errores de tipo como en otros pipelines.

---

## Fase 5: Scheduler en `app/scheduler.py`

Después de los jobs Sofascore (~línea 7419), agregar gated por flags separados:

```python
# SOTA: FotMob refs sync - every 12h (ABE P1: flag independiente)
if settings.FOTMOB_REFS_ENABLED:
    scheduler.add_job(
        sota_fotmob_refs_sync,
        trigger=IntervalTrigger(hours=12),
        id="sota_fotmob_refs_sync",
        name="SOTA FotMob Refs Sync (every 12h)",
        replace_existing=True,
        next_run_time=datetime.utcnow() + timedelta(seconds=75),
        max_instances=1, coalesce=True, misfire_grace_time=12*3600,
    )

# SOTA: FotMob xG backfill - every 6h (requires refs to exist)
if settings.FOTMOB_XG_ENABLED:
    scheduler.add_job(
        sota_fotmob_xg_backfill,
        trigger=IntervalTrigger(hours=6),
        id="sota_fotmob_xg_backfill",
        name="SOTA FotMob xG Backfill (every 6h)",
        replace_existing=True,
        next_run_time=datetime.utcnow() + timedelta(seconds=85),
        max_instances=1, coalesce=True, misfire_grace_time=6*3600,
    )
```

**ABE P1**: Flags separados permiten: (1) activar solo refs para verificar linking, (2) activar xG una vez confirmado que refs funcionan.
**P0-1**: Solo scheduler, nunca desde endpoints.

~25 líneas.

---

## Fase 6: Feature Matrix — Dual xG Source (PR2)

> **ABE P0-2 review**: Esta fase va en PR separado (PR2), una vez verificada ingesta de PR1.

### Modificar `app/titan/materializers/feature_matrix.py`

En `compute_xg_last5_features()` (línea 530), extender para soportar ambas fuentes.

### Problemas del plan original (corregidos)

1. **Columnas inexistentes**: `match_fotmob_stats` NO tiene `xga_home/xga_away` ni `npxg_home/npxg_away`. Usar `NULL as xga_home` en SQL funciona, pero luego `Decimal(str(round(None, 2)))` **explota con TypeError**.
2. **Heurística `_get_match_league()`**: Buscar liga por "último match del equipo" es no-determinista y frágil (equipos en copas internacionales cambian de liga). ABE requiere determinismo.
3. **xGA se puede derivar**: xGA de un equipo = xG del rival. FotMob tiene xg_home/xg_away, suficiente para derivar.

### Estrategia corregida (ABE P0-2 compliant)

**Cambio 1**: La función recibe `league_id` como parámetro explícito (determinista, sin heurística):

```python
async def compute_xg_last5_features(
    self,
    home_team_id: int,
    away_team_id: int,
    kickoff_utc: datetime,
    league_id: int | None = None,  # NUEVO: determinista
    limit: int = None,
) -> Optional[XGFeatures]:
```

El caller (materializer principal) YA tiene `league_id` del match target — solo necesita pasarlo.

**Cambio 2**: Join dinámico según source, con xGA derivada para FotMob:

```python
if league_id and league_id in UNDERSTAT_SUPPORTED_LEAGUES:
    # Understat: tiene xga y npxg nativamente
    xg_join = "JOIN match_understat_team xg ON m.id = xg.match_id"
    xg_select = """
        CASE WHEN m.home_team_id = :team_id THEN xg.xg_home ELSE xg.xg_away END as xg,
        CASE WHEN m.home_team_id = :team_id THEN xg.xga_home ELSE xg.xga_away END as xga,
        CASE WHEN m.home_team_id = :team_id THEN xg.npxg_home ELSE xg.npxg_away END as npxg
    """
    has_npxg = True
else:
    # FotMob: derivar xGA como "xG del rival", npxg no disponible
    xg_join = "JOIN match_fotmob_stats xg ON m.id = xg.match_id"
    xg_select = """
        CASE WHEN m.home_team_id = :team_id THEN xg.xg_home ELSE xg.xg_away END as xg,
        CASE WHEN m.home_team_id = :team_id THEN xg.xg_away ELSE xg.xg_home END as xga,
        NULL::double precision as npxg
    """
    has_npxg = False
```

**Clave**: `xga = xG del rival` es correcto semánticamente (xG Against = lo que generó el oponente). Esto permite poblar `xga_home_last5` y `xga_away_last5` sin columnas adicionales en FotMob.

**Cambio 3**: Guard Decimal-safe para campos NULL (npxg en path FotMob):

```python
# Construir XGFeatures con guards para None
xg_home_last5 = Decimal(str(round(home_row[0], 2))) if home_row[0] is not None else None
xga_home_last5 = Decimal(str(round(home_row[1], 2))) if home_row[1] is not None else None
npxg_home_last5 = Decimal(str(round(home_row[2], 2))) if home_row[2] is not None else None
# ... ídem away

return XGFeatures(
    xg_home_last5=xg_home_last5,
    xg_away_last5=xg_away_last5,
    xga_home_last5=xga_home_last5,
    xga_away_last5=xga_away_last5,
    npxg_home_last5=npxg_home_last5,   # None para FotMob leagues
    npxg_away_last5=npxg_away_last5,   # None para FotMob leagues
    captured_at=_utc_now(),
)
```

**ABE P1 (incluido en PR2)**: Agregar `AND xg.captured_at < :kickoff` al query para PIT extra-safe. Eliminará cualquier duda de leakage si hay capturas tardías o re-capturas.

**Nota operativa**: Cuando `npxg_*_last5` sea NULL (FotMob leagues), revisar cómo se decide `tier1b_complete`. Puede requerir que npxg sea opcional para FotMob-leagues, o que el completeness check se base solo en xg/xga.

### Resumen de campos por source

| Campo | Understat | FotMob |
|-------|-----------|--------|
| `xg_home_last5` | nativo | nativo |
| `xg_away_last5` | nativo | nativo |
| `xga_home_last5` | nativo (xga_home) | **derivado** (xg_away del rival) |
| `xga_away_last5` | nativo (xga_away) | **derivado** (xg_home del rival) |
| `npxg_home_last5` | nativo | **NULL** (no disponible) |
| `npxg_away_last5` | nativo | **NULL** (no disponible) |

~40 líneas modificadas.

---

## Archivos a modificar/crear

### PR1: Tabla + Provider + Jobs + Scheduler (solo 128, flags OFF)

| Archivo | Acción | Fase | ~Líneas |
|---------|--------|------|---------|
| `app/config.py` | Modificar: 6 settings (split flags) | 1A | +6 |
| `app/etl/sota_constants.py` | Modificar: league mappings + confirmed set | 1B | +35 |
| `scripts/migrations/0XX_add_fotmob_stats.py` | **Crear**: migration | 2 | +25 |
| `app/etl/fotmob_provider.py` | **Crear**: provider class | 3 | +250 |
| `app/etl/sota_jobs.py` | Modificar: 2 jobs + upsert | 4 | +200 |
| `app/scheduler.py` | Modificar: 2 scheduler entries (split flags) | 5 | +30 |

### PR2: Feature Matrix (post-verificación ingesta)

| Archivo | Acción | Fase | ~Líneas |
|---------|--------|------|---------|
| `app/titan/materializers/feature_matrix.py` | Modificar: dual xG source + Decimal guards | 6 | +40 |

**Total**: 2 archivos nuevos, 6 modificados, ~586 líneas nuevas.

---

## Orden de implementación (ABE: split en 2 PRs)

### PR1 (Fases 1-5): Ingesta FotMob
1. **Fase 1** (Config + Constants) — base, no-op hasta flags ON
2. **Fase 2** (Migration) — crear tabla en DB
3. **Fase 3** (Provider) — clase FotMob completa
4. **Fase 4** (Jobs) — sync_refs + backfill_xg (P0-8: solo ligas confirmed)
5. **Fase 5** (Scheduler) — registrar jobs (split flags)
6. **Deploy PR1** con flags OFF, activar `FOTMOB_REFS_ENABLED=true` primero, verificar linking, luego `FOTMOB_XG_ENABLED=true`

### PR2 (Fase 6): Feature Matrix
7. **Fase 6** (Feature Matrix) — solo después de verificar que `match_fotmob_stats` tiene datos válidos
8. **Verificación end-to-end**

---

## Verificación

### Pre-deploy (ambos flags false)
- Pipeline idéntico al actual — early return en jobs
- Feature matrix: sin datos en `match_fotmob_stats`, retorna None (igual que hoy)

### PR1 Fase A: Activar `FOTMOB_REFS_ENABLED=true` (solo linking)

```sql
-- Verificar refs creados
SELECT source, COUNT(*), AVG(confidence)
FROM match_external_refs WHERE source = 'fotmob'
GROUP BY source;
-- Expected: N > 0 para Argentina, avg confidence >= 0.90
```

Logs: `[FOTMOB-REFS] scanned=N linked_auto=M skipped_no_mapping=0`

### PR1 Fase B: Activar `FOTMOB_XG_ENABLED=true` (xG capture)

```sql
-- Verificar xG capturado
SELECT m.league_id, COUNT(*) as xg_count,
       ROUND(AVG(mfs.xg_home + mfs.xg_away)::numeric, 2) as avg_total_xg,
       MIN(mfs.captured_at), MAX(mfs.captured_at)
FROM match_fotmob_stats mfs
JOIN matches m ON m.id = mfs.match_id
GROUP BY m.league_id ORDER BY xg_count DESC;
-- Sanity: avg_total_xg entre 1.5 y 3.5
-- Solo league_id=128 (Argentina) debe aparecer
```

Logs: `[FOTMOB-XG] captured=N skipped_no_xg=M errors=0`

### PR2 Fase C: Feature Matrix (post-ingesta verificada)

```sql
-- Verificar que feature_matrix tiene xG para Argentina
SELECT fm.match_id, fm.xg_home_last5, fm.xg_away_last5,
       fm.xga_home_last5, fm.xga_away_last5,
       fm.npxg_home_last5, fm.tier1b_complete
FROM titan.feature_matrix fm
JOIN matches m ON m.id = fm.match_id
WHERE m.league_id = 128
ORDER BY m.date DESC LIMIT 10;
-- xg_*_last5: NOT NULL (datos FotMob)
-- xga_*_last5: NOT NULL (derivado de xG rival)
-- npxg_*_last5: NULL (no disponible en FotMob, esperado)
-- tier1b_complete: depende de regla (ajustar si npxg es opcional)
```

### Rollback
- `FOTMOB_REFS_ENABLED=false` + `FOTMOB_XG_ENABLED=false` → jobs no corren
- Feature matrix ignora tabla vacía (fallback a None, como hoy)
- No requiere delete de datos — tabla queda para reactivación

---

## Resultados Post-Deploy (2026-02-09)

### PR1 Deploy & Activación

| Commit | Descripción |
|--------|-------------|
| `42a49a2` | feat: FotMob xG provider (7 archivos, 1,786 líneas) |
| `f94ba15` | fix: `mer.match_id IS NULL` (ABE P0 blocker — PK compuesto) |
| `bd8d419` | fix: fixtures parser (`data["fixtures"]["allMatches"]`, no `data["matches"]`) |
| `1e5bb65` | fix: xG parser (`content.stats.Periods.All.stats`, no `content.stats.Ede`) |
| `b2fe102` | feat: aliases FotMob + backfill histórico 2023-2025 |

**Bugs de schema corregidos (3)**: Los JSON paths del provider se escribieron basándose en documentación de soccerdata, no contra la API real. Los 3 errores son del mismo tipo: estructura asumida ≠ estructura real de FotMob. Metodología de fix: curl directo → inspeccionar keys reales → corregir parser → verificar con datos reales antes de push.

### Cobertura xG Argentina (league_id=128)

| Season | xG capturados | Total FT | Cobertura |
|--------|--------------|----------|-----------|
| 2023 | 371 | 378 | 98.1% |
| 2024 | 377 | 378 | 99.7% |
| 2025 | 510 | 510 | 100.0% |
| 2026 | 56 | 56 | 100.0% |
| **Total** | **1,314** | **1,322** | **99.4%** |

- **Backfill histórico**: 2 pasadas (~35 min total, rate limit 1.5s/req, 0 bloqueos)
- **Pasada 1**: Sin alias → 1,050 capturados (79.4%)
- **Pasada 2**: Con alias → +264 capturados → 1,314 (99.4%)
- **7 faltantes**: Jornada 1 de 2023 (desfase timezone) — no vale la pena perseguir
- **Alias agregados**: Argentinos JRS↔Argentinos Juniors, Newell's, Defensa Y/y, Platense↔Club Atletico Platense, Estudiantes (sin LP), Union (sin Santa Fe), San Martin San Juan

### Jobs Automáticos (en producción)

| Job | Frecuencia | Flag | Estado |
|-----|------------|------|--------|
| `sota_fotmob_refs_sync` | 12h (+75s offset) | `FOTMOB_REFS_ENABLED=true` | Activo |
| `sota_fotmob_xg_backfill` | 6h (+85s offset) | `FOTMOB_XG_ENABLED=true` | Activo |

### Pendiente

- **PR2 (Fase 6)**: Integración xG en `titan.feature_matrix` (dual source Understat/FotMob)
- **Verificar IDs TBD**: 22 league mappings en `LEAGUE_ID_TO_FOTMOB` marcados como TBD
- **Alias para sync_fotmob_refs en scheduler**: El job de producción no pasa `alias_index` a `calculate_match_score` — solo el script de backfill lo hace. P1: agregar en próximo PR.
