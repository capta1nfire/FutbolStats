# AUDITORÍA CRÍTICA - Bon Jogo
**Fecha**: 2026-01-09  
**Auditor**: Senior  
**Scope**: Cambios últimos ~2 días  
**Enfoque**: Bugs críticos, omisiones, riesgos operativos

---

## RESUMEN EJECUTIVO

Se identificaron **5 riesgos críticos/altos** que requieren corrección inmediata antes de producción. Los scripts de backfill están correctamente guardados (no escriben a PIT/odds tables), pero hay problemas de sincronización de budget, anti-leakage edge cases, y métricas engañosas en el dashboard.

---

## TOP 5 RIESGOS

### 🔴 **RIESGO #1: Desincronización Budget Interno vs API Externa**
**Severidad**: **CRITICAL**

**Problema**:  
Existen dos sistemas de budget que pueden desincronizarse:

1. **Budget interno** (`app/etl/api_football.py` líneas 26-55): `_budget_used` se incrementa en cada request
2. **Budget externo** (`get_api_account_status()` líneas 77-142): Consulta real a `/status` endpoint (cache 10 min)

**Evidencia**:
```31:55:app/etl/api_football.py
async def _budget_check_and_increment(cost: int = 1) -> None:
    """
    Enforce a global daily request budget across ALL APIFootballProvider instances.
    Budget is controlled via env var API_DAILY_BUDGET (default 75000).
    """
    global _budget_day, _budget_used
    
    daily_budget = int(getattr(settings, "API_DAILY_BUDGET", 0) or 0)
    if daily_budget <= 0:
        # Backward compatible: if not configured, do not enforce budget.
        return
    
    today = datetime.utcnow().date()
    async with _budget_lock:
        if _budget_day != today:
            _budget_day = today
            _budget_used = 0
        
        if _budget_used + cost > daily_budget:
            raise APIBudgetExceeded(...)
        
        _budget_used += cost
```

**Riesgo**:  
- Si el proceso se reinicia, `_budget_used` se resetea a 0 pero el API real sigue contando
- Si hay múltiples procesos/workers, cada uno tiene su propio contador interno
- El dashboard muestra `requests_today` del API real pero el guardrail usa `_budget_used` interno

**Corrección mínima**:
```python
# En app/etl/api_football.py, modificar _budget_check_and_increment():
async def _budget_check_and_increment(cost: int = 1) -> None:
    global _budget_day, _budget_used
    
    daily_budget = int(getattr(settings, "API_DAILY_BUDGET", 0) or 0)
    if daily_budget <= 0:
        return
    
    today = datetime.utcnow().date()
    async with _budget_lock:
        if _budget_day != today:
            _budget_day = today
            _budget_used = 0
        
        # SYNC: Verificar con API real si estamos cerca del límite
        if _budget_used > daily_budget * 0.8:  # Solo check si >80% usado
            try:
                real_status = await get_api_account_status()
                real_used = real_status.get("requests_today", 0)
                if real_used > 0:
                    _budget_used = max(_budget_used, real_used)  # Usar el mayor
            except Exception:
                pass  # Fallback a contador interno si API falla
        
        if _budget_used + cost > daily_budget:
            raise APIBudgetExceeded(...)
        
        _budget_used += cost
```

**Verificación faltante**:
```sql
-- Query para detectar desincronización:
SELECT 
    DATE_TRUNC('day', NOW()) as today,
    COUNT(*) FILTER (WHERE created_at >= DATE_TRUNC('day', NOW())) as internal_count
FROM odds_snapshots
WHERE created_at >= DATE_TRUNC('day', NOW());
-- Comparar con requests_today del dashboard
```

---

### 🔴 **RIESGO #2: Anti-Leakage Edge Case - Predictions Creadas en Mismo Segundo**
**Severidad**: **HIGH**

**Problema**:  
En `build_pit_dataset.py` línea 110, la condición anti-leakage es:
```sql
WHERE p.created_at <= ps.snapshot_at
```

Si una predicción se crea en el mismo segundo (o microsegundo) que el snapshot, puede pasar el filtro pero técnicamente es leakage si el snapshot se capturó primero.

**Evidencia**:
```97:112:scripts/build_pit_dataset.py
predictions_asof AS (
    -- Latest prediction BEFORE snapshot (ANTI-LEAKAGE constraint)
    SELECT DISTINCT ON (p.match_id, ps.snapshot_id)
        ps.snapshot_id,
        p.id as prediction_id,
        p.match_id,
        p.home_prob,
        p.draw_prob,
        p.away_prob,
        p.model_version,
        p.created_at as prediction_at
    FROM predictions p
    JOIN pit_snapshots ps ON p.match_id = ps.match_id
    WHERE p.created_at <= ps.snapshot_at  -- ANTI-LEAKAGE: prediction must exist BEFORE snapshot
    ORDER BY p.match_id, ps.snapshot_id, p.created_at DESC
)
```

**Riesgo**:  
- En sistemas de alta frecuencia, predicciones y snapshots pueden tener timestamps idénticos
- PostgreSQL `TIMESTAMP` tiene precisión de microsegundos, pero si se usa `NOW()` en ambos, pueden coincidir
- La validación en línea 395 solo verifica `prediction_at > snapshot_at`, no igualdad

**Corrección mínima**:
```python
# En scripts/build_pit_dataset.py, cambiar línea 110:
WHERE p.created_at < ps.snapshot_at  -- Cambiar <= a < (strict before)

# Y agregar validación más estricta en línea 395:
result = con.execute("""
    SELECT COUNT(*)
    FROM pit_dataset
    WHERE prediction_at >= snapshot_at  -- Cambiar > a >= (incluye igualdad como violación)
""").fetchone()
```

**Verificación faltante**:
```sql
-- Query para detectar timestamps idénticos:
SELECT 
    snapshot_id,
    prediction_id,
    snapshot_at,
    prediction_at,
    EXTRACT(EPOCH FROM (snapshot_at - prediction_at)) as diff_seconds
FROM pit_dataset
WHERE prediction_at IS NOT NULL
  AND ABS(EXTRACT(EPOCH FROM (snapshot_at - prediction_at))) < 1.0  -- <1 segundo de diferencia
ORDER BY diff_seconds;
```

---

### 🟠 **RIESGO #3: MARKET_MOVEMENT_REQUIRE_LINEUP=0 Puede Contaminar Baseline**
**Severidad**: **HIGH**

**Problema**:  
Cuando `MARKET_MOVEMENT_REQUIRE_LINEUP=0`, el scheduler captura snapshots de market movement **sin** lineup confirmado. Estos snapshots se usan como baseline en `build_pit_dataset.py` (líneas 66-95), pero pueden no ser comparables si el lineup aún no se anunció.

**Evidencia**:
```832:846:app/scheduler.py
# Guardrail: require lineup_confirmed by default to avoid extra API calls
require_lineup = os.environ.get("MARKET_MOVEMENT_REQUIRE_LINEUP", "1") == "1"

try:
    async with AsyncSessionLocal() as session:
        now = datetime.utcnow()
        league_ids = await resolve_lineup_monitoring_leagues(session)
        
        # Get matches that need market movement data
        # Focus on matches 5-65 minutes from now (covers all buckets)
        window_start = now + timedelta(minutes=3)
        window_end = now + timedelta(minutes=65)
        
        # Build query conditionally based on require_lineup setting
        lineup_condition = "AND m.lineup_confirmed = TRUE" if require_lineup else ""
```

Y en `build_pit_dataset.py`:
```66:95:scripts/build_pit_dataset.py
baseline_market_same AS (
    -- Baseline odds from market_movement_snapshots for the SAME bookmaker as PIT (best-effort).
    -- We pick the earliest available pre-kickoff snapshot (max minutes_to_kickoff).
    SELECT DISTINCT ON (mms.match_id, mms.bookmaker)
        mms.match_id,
        mms.bookmaker,
        mms.snapshot_type as baseline_snapshot_type,
        mms.captured_at as baseline_captured_at,
        mms.minutes_to_kickoff as baseline_minutes_to_kickoff,
        mms.odds_home as baseline_odds_home,
        mms.odds_draw as baseline_odds_draw,
        mms.odds_away as baseline_odds_away
    FROM market_movement_snapshots mms
    WHERE mms.captured_at < mms.kickoff_time
    ORDER BY mms.match_id, mms.bookmaker, mms.minutes_to_kickoff DESC, mms.captured_at ASC
),
```

**Riesgo**:  
- Si `MARKET_MOVEMENT_REQUIRE_LINEUP=0`, se capturan snapshots T-60/T-30 **antes** del lineup
- Estos se usan como baseline para calcular CLV proxy
- Pero el PIT snapshot es **después** del lineup, entonces estamos comparando "pre-lineup" vs "post-lineup"
- Esto infla artificialmente el edge (el mercado se mueve por el lineup, no por nuestro modelo)

**Corrección mínima**:
```python
# En scripts/build_pit_dataset.py, modificar baseline_market_same y baseline_market_any:
baseline_market_same AS (
    SELECT DISTINCT ON (mms.match_id, mms.bookmaker)
        mms.match_id,
        mms.bookmaker,
        mms.snapshot_type as baseline_snapshot_type,
        mms.captured_at as baseline_captured_at,
        mms.minutes_to_kickoff as baseline_minutes_to_kickoff,
        mms.odds_home as baseline_odds_home,
        mms.odds_draw as baseline_odds_draw,
        mms.odds_away as baseline_odds_away
    FROM market_movement_snapshots mms
    JOIN matches m ON m.id = mms.match_id
    WHERE mms.captured_at < mms.kickoff_time
      AND m.lineup_confirmed = TRUE  -- AGREGAR: Solo usar baseline si lineup ya estaba confirmado
      AND mms.captured_at >= (
          SELECT MIN(os.snapshot_at) 
          FROM odds_snapshots os 
          WHERE os.match_id = mms.match_id 
            AND os.snapshot_type = 'lineup_confirmed'
      )  -- AGREGAR: Baseline debe ser POST-lineup detection
    ORDER BY mms.match_id, mms.bookmaker, mms.minutes_to_kickoff DESC, mms.captured_at ASC
),
```

**Verificación faltante**:
```sql
-- Query para detectar baseline pre-lineup:
SELECT 
    mms.match_id,
    mms.captured_at as baseline_at,
    m.lineup_confirmed,
    MIN(os.snapshot_at) as lineup_detected_at,
    CASE 
        WHEN mms.captured_at < MIN(os.snapshot_at) THEN 'PRE-LINEUP (CONTAMINADO)'
        ELSE 'POST-LINEUP (OK)'
    END as contamination_check
FROM market_movement_snapshots mms
JOIN matches m ON m.id = mms.match_id
LEFT JOIN odds_snapshots os ON os.match_id = mms.match_id AND os.snapshot_type = 'lineup_confirmed'
WHERE mms.captured_at < mms.kickoff_time
GROUP BY mms.match_id, mms.captured_at, m.lineup_confirmed
HAVING mms.captured_at < MIN(os.snapshot_at)
LIMIT 100;
```

---

### 🟠 **RIESGO #4: Dashboard Budget Hardcodeado vs Real**
**Severidad**: **MEDIUM**

**Problema**:  
El endpoint `/sync/status` (línea 236-252) devuelve valores hardcodeados que no coinciden con el budget real:

```236:252:app/main.py
@app.get("/sync/status")
async def get_sync_status():
    """
    Get current sync status for iOS display.
    Returns last sync timestamp and API budget info.
    Used by mobile app to show data freshness.
    """
    last_sync = get_last_sync_time()
    return {
        "last_sync_at": last_sync.isoformat() if last_sync else None,
        "sync_interval_seconds": 60,
        "daily_api_calls": 1440,
        "daily_budget": 7500,  # HARDCODED - no coincide con API_DAILY_BUDGET
        "budget_remaining_percent": 80,  # HARDCODED
        "leagues": SYNC_LEAGUES,
    }
```

**Riesgo**:  
- iOS app muestra información incorrecta al usuario
- Puede mostrar "80% remaining" cuando en realidad está al 95%
- Valores hardcodeados (`7500`, `1440`) no reflejan la realidad

**Corrección mínima**:
```python
# En app/main.py, modificar get_sync_status():
@app.get("/sync/status")
async def get_sync_status():
    last_sync = get_last_sync_time()
    
    # Fetch real budget status
    budget_status = {"status": "unavailable"}
    try:
        from app.etl.api_football import get_api_account_status, get_api_budget_status
        budget_status = await get_api_account_status()
        budget_internal = get_api_budget_status()
    except Exception:
        pass
    
    daily_budget = budget_status.get("requests_limit") or budget_internal.get("budget_total") or 7500
    daily_used = budget_status.get("requests_today") or budget_internal.get("budget_used") or 0
    remaining_pct = round((1 - daily_used / daily_budget) * 100, 1) if daily_budget > 0 else 80
    
    return {
        "last_sync_at": last_sync.isoformat() if last_sync else None,
        "sync_interval_seconds": 60,
        "daily_api_calls": daily_used,
        "daily_budget": daily_budget,
        "budget_remaining_percent": remaining_pct,
        "leagues": SYNC_LEAGUES,
    }
```

**Verificación faltante**:  
- Comparar `daily_budget` del endpoint con `API_DAILY_BUDGET` env var
- Verificar que `budget_remaining_percent` se calcula correctamente

---

### 🟡 **RIESGO #5: Overround Calculation Inconsistente**
**Severidad**: **MEDIUM**

**Problema**:  
Hay dos fórmulas diferentes para calcular overround:

1. **En `scheduler.py` línea 922**: `overround = total - 1` (donde `total = raw_home + raw_draw + raw_away`)
2. **En `evaluate_pit_ev.py` línea 529**: `overround = q_h + q_d + q_a` (sin restar 1)

**Evidencia**:
```916:925:app/scheduler.py
# Calculate implied probabilities
if odds_home > 1 and odds_draw > 1 and odds_away > 1:
    raw_home = 1 / odds_home
    raw_draw = 1 / odds_draw
    raw_away = 1 / odds_away
    total = raw_home + raw_draw + raw_away
    overround = total - 1  # <-- RESTA 1
    prob_home = raw_home / total
```

```526:529:scripts/evaluate_pit_ev.py
q_h = 1.0 / odds_h
q_d = 1.0 / odds_draw
q_a = 1.0 / odds_away
overround = q_h + q_d + q_a  # <-- NO RESTA 1
```

**Riesgo**:  
- Métricas inconsistentes: scheduler guarda `overround = 0.05` (5%) pero evaluación muestra `overround = 1.05` (105%)
- Dashboard puede mostrar valores incorrectos
- Comparaciones entre sistemas fallan

**Corrección mínima**:
```python
# Estandarizar: overround = total - 1 (exceso sobre 1.0)
# En scripts/evaluate_pit_ev.py línea 529, cambiar a:
overround = (q_h + q_d + q_a) - 1.0  # Exceso sobre fair odds

# O documentar que scheduler usa "excess" y evaluación usa "total"
# Pero mejor estandarizar a "excess" (más común en betting)
```

**Verificación faltante**:
```sql
-- Query para detectar inconsistencias:
SELECT 
    snapshot_id,
    overround as stored_overround,
    (1.0/pit_odds_home + 1.0/pit_odds_draw + 1.0/pit_odds_away) as calculated_total,
    (1.0/pit_odds_home + 1.0/pit_odds_draw + 1.0/pit_odds_away) - 1.0 as calculated_excess
FROM pit_dataset
WHERE pit_odds_home IS NOT NULL
LIMIT 100;
-- Comparar stored_overround con calculated_excess
```

---

## CONFIRMACIÓN: WRITES NO DESEADOS

✅ **CONFIRMADO**: Los scripts de backfill **NO escriben** a PIT/odds tables:

1. **`scripts/ingest_football_data_uk.py`**:
   - Solo escribe a `matches.opening_odds_*` (líneas 624-628)
   - Comentario explícito línea 19: "Does NOT touch: odds_snapshots, market_movement_snapshots..."
   - Guardrail línea 628: `WHERE opening_odds_home IS NULL` (solo actualiza NULLs)

2. **`scripts/backfill_fixtures_latam_pack2.py`**:
   - Solo escribe a `matches` y `teams` (líneas 312-392)
   - Comentario explícito líneas 9-14: "NO escribe: opening_odds_*, odds_snapshots, market_movement_snapshots..."
   - Confirmación línea 561: "CONFIRMACION: NO se escribieron odds"

3. **`scripts/build_pit_dataset.py`**:
   - Solo escribe a DuckDB local (`data/pit_dataset.duckdb`)
   - No toca PostgreSQL

4. **`scripts/evaluate_pit_ev.py`**:
   - Solo lee DuckDB, no escribe nada

✅ **SEGURO**: No hay contaminación de datos PIT/odds desde scripts de backfill.

---

## VERIFICACIONES FALTANTES (Queries/Logs)

### 1. **Budget Desincronización**
```sql
-- Detectar desincronización budget interno vs API
SELECT 
    DATE_TRUNC('day', NOW()) as today,
    COUNT(*) as snapshots_today,
    COUNT(DISTINCT match_id) as matches_today
FROM odds_snapshots
WHERE created_at >= DATE_TRUNC('day', NOW());
-- Comparar con requests_today del dashboard
```

### 2. **Anti-Leakage Timestamps Idénticos**
```sql
-- Detectar predicciones con timestamp <= snapshot (edge case)
SELECT 
    snapshot_id,
    prediction_id,
    snapshot_at,
    prediction_at,
    EXTRACT(EPOCH FROM (snapshot_at - prediction_at)) as diff_seconds
FROM pit_dataset
WHERE prediction_at IS NOT NULL
  AND prediction_at >= snapshot_at  -- Violación (debe ser <)
ORDER BY diff_seconds;
```

### 3. **Baseline Pre-Lineup Contamination**
```sql
-- Detectar baseline capturado antes del lineup
SELECT 
    mms.match_id,
    mms.captured_at as baseline_at,
    MIN(os.snapshot_at) as lineup_detected_at,
    CASE 
        WHEN mms.captured_at < MIN(os.snapshot_at) THEN 'CONTAMINADO'
        ELSE 'OK'
    END as status
FROM market_movement_snapshots mms
JOIN matches m ON m.id = mms.match_id
LEFT JOIN odds_snapshots os ON os.match_id = mms.match_id 
    AND os.snapshot_type = 'lineup_confirmed'
WHERE mms.captured_at < mms.kickoff_time
GROUP BY mms.match_id, mms.captured_at
HAVING mms.captured_at < MIN(os.snapshot_at)
LIMIT 100;
```

### 4. **Overround Inconsistencia**
```sql
-- Verificar overround almacenado vs calculado
SELECT 
    snapshot_id,
    overround as stored,
    (1.0/odds_home + 1.0/odds_draw + 1.0/odds_away) - 1.0 as calculated_excess,
    ABS(overround - ((1.0/odds_home + 1.0/odds_draw + 1.0/odds_away) - 1.0)) as diff
FROM odds_snapshots
WHERE odds_home > 1 AND odds_draw > 1 AND odds_away > 1
  AND ABS(overround - ((1.0/odds_home + 1.0/odds_draw + 1.0/odds_away) - 1.0)) > 0.01
LIMIT 50;
```

### 5. **Fallos Silenciosos - Budget Exceeded Sin Log**
```python
# Agregar alerta cuando budget > 90%
# En app/scheduler.py, después de cada job:
budget_status = get_api_budget_status()
if budget_status.get("budget_remaining", 999999) < budget_status.get("budget_total", 1) * 0.1:
    logger.critical(f"BUDGET CRITICAL: {budget_status}")
    # Enviar alerta (email/Slack/webhook)
```

---

## MÉTRICAS ENGAÑOSAS EN DASHBOARD

### 1. **Budget Hardcodeado** (`/sync/status`)
- **Problema**: Valores fijos `7500`, `1440`, `80%`
- **Impacto**: iOS app muestra información incorrecta
- **Fix**: Usar `get_api_account_status()` real (ver Riesgo #4)

### 2. **Delta KO Minutes Sign**
- **Verificar**: En `build_pit_dataset.py` línea 123, `delta_ko_minutes` se calcula como:
  ```sql
  ROUND(ps.delta_to_kickoff_seconds / 60.0, 1) as delta_ko_minutes
  ```
- **Pregunta**: ¿`delta_to_kickoff_seconds` es positivo (minutos antes) o negativo (minutos después)?
- **Riesgo**: Si es negativo, `delta_ko_minutes` será negativo y los filtros `BETWEEN 10 AND 90` fallarán
- **Verificación**: Revisar cómo se calcula `delta_to_kickoff_seconds` en `scheduler.py`

### 3. **EV Calculation Sign**
- **Verificar**: En `build_pit_dataset.py` línea 312, `ev_home = (home_prob * pit_odds_home) - 1`
- **Correcto**: EV positivo = valor esperado positivo (correcto)
- **Confirmar**: En `evaluate_pit_ev.py` línea 204, misma fórmula (consistente ✅)

---

## RESUMEN DE CORRECCIONES MÍNIMAS

1. **Budget Sync** (CRITICAL): Agregar sync periódico con API real cuando `_budget_used > 80%`
2. **Anti-Leakage** (HIGH): Cambiar `<=` a `<` en línea 110 de `build_pit_dataset.py`
3. **Baseline Contamination** (HIGH): Agregar filtro `lineup_confirmed = TRUE` y `captured_at >= lineup_detected_at` en baseline queries
4. **Dashboard Budget** (MEDIUM): Reemplazar valores hardcodeados con `get_api_account_status()` real
5. **Overround** (MEDIUM): Estandarizar a `overround = total - 1` en `evaluate_pit_ev.py`

---

## PRÓXIMOS PASOS RECOMENDADOS

1. **Inmediato**: Aplicar fixes #1, #2, #3 (riesgos críticos/altos)
2. **Esta semana**: Aplicar fixes #4, #5 (métricas)
3. **Siguiente sprint**: Implementar verificaciones faltantes (queries de monitoreo)
4. **Fase A**: Continuar con métricas objetivo para captura + N + CLV proxy

---

**FIN DEL REPORTE**

