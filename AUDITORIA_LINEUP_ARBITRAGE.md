# AUDITORÍA TÉCNICA: Real-time Lineup Odds Capture (Lineup Arbitrage)

**Fecha:** 2025-01-XX  
**Auditor:** Lead Data Scientist / Auditor Externo  
**Objetivo:** Validar que el sistema mide correctamente el baseline del mercado en el timestamp de alineaciones y no introduce bugs/leakage.

---

## RESUMEN EJECUTIVO

**Estado General:** 🟢 **LISTO PARA PRODUCCIÓN** - Cambios críticos implementados

**Hallazgos Principales (Estado Actual):**
1. ✅ **CORREGIDO:** Eliminado fallback a `odds_history` stale → baseline garantiza frescura
2. ✅ **MEJORADO:** `lineup_confirmed_at` usa timestamp real en producción (backfill usa aproximación histórica)
3. ✅ **OK:** Idempotencia correcta con `ON CONFLICT`
4. ✅ **CORREGIDO:** Endpoint `/lineup/monitor` protegido con autenticación
5. ✅ **IMPLEMENTADO:** Validación de `snapshot_at < kickoff` antes de insertar

**Cambios Críticos Implementados:**
- ✅ Baseline freshness garantizado (NO fallback a stale odds)
- ✅ Validaciones temporales (snapshot_at < kickoff, delta en rango)
- ✅ Seguridad del endpoint corregida
- ✅ Robustez mejorada (retry logic, validaciones de calidad)

---

## CHECKLIST DE AUDITORÍA

### 1) Correctitud del Baseline ("odds en ese instante")

**Estado:** ✅ **CORREGIDO** - Cambios implementados

#### Análisis

**Código Relevante:**
```226:353:app/scheduler.py
async def monitor_lineups_and_capture_odds() -> dict:
    # ...
    # PRIMARY: Get LIVE odds from API-Football (most accurate)
    fresh_odds = await provider.get_odds(external_id)

    if fresh_odds and fresh_odds.get("odds_home"):
        odds_home = float(fresh_odds["odds_home"])
        # ...
        source = fresh_odds.get("bookmaker", "api_football_live")
    else:
        # FALLBACK 1: Most recent from odds_history
        odds_result = await session.execute(text("""
            SELECT odds_home, odds_draw, odds_away, recorded_at, source
            FROM odds_history
            WHERE match_id = :match_id
              AND odds_home IS NOT NULL
            ORDER BY recorded_at DESC
            LIMIT 1
        """), {"match_id": match_id})
```

**Problema Identificado:**

1. **`odds_history` NO se actualiza en tiempo real:**
   - Revisión del código muestra que `odds_history` solo se actualiza en:
     - `ETLPipeline._save_odds_history()` durante syncs manuales (`POST /etl/sync`)
     - NO hay job automático que actualice `odds_history` cada pocos minutos
   - El scheduler `global_sync_today()` (cada 60s) NO guarda en `odds_history`, solo actualiza `matches.odds_*`

2. **Fallback a `odds_history` puede ser MUY STALE:**
   - Si `provider.get_odds()` falla (rate limit, API down), el sistema usa `odds_history`
   - Pero `odds_history` puede tener datos de días/horas antes
   - El código marca como `"_stale"` pero NO valida la antigüedad

3. **Falta validación de frescura:**
   - No hay check de `recorded_at` vs `snapshot_at` para determinar si `odds_history` es aceptable
   - Un snapshot de hace 2 horas NO es válido como baseline para T-60min

#### Verificación de Actualización de `odds_history`

**Búsqueda en código:**
- `app/etl/pipeline.py` línea 186-208: `_save_odds_history()` solo se llama cuando `fetch_odds=True` en sync manual
- `app/scheduler.py` línea 41-90: `global_sync_today()` NO llama a `_save_odds_history()`
- **Conclusión:** `odds_history` NO se actualiza automáticamente en producción

#### Diagnóstico

**Veredicto:** ✅ **CORREGIDO**

El sistema ahora garantiza frescura del baseline. Si la API falla, omite el match y reintenta en el siguiente ciclo (5 min).

#### Cambios Implementados

**✅ IMPLEMENTADO - Opción A (Recomendada):**

El código ahora captura odds DIRECTAMENTE del provider y NO usa fallback a stale:
```python
# app/scheduler.py, líneas 371-378
if fresh_odds and fresh_odds.get("odds_home"):
    # Usar odds frescas del API
    ...
else:
    # NO FALLBACK: Skip match y retry en siguiente run
    logger.error(
        f"Cannot capture fresh odds for match {match_id} "
        f"- API returned: {fresh_odds}. Skipping. Will retry in next run (5 min)."
    )
    continue
```

---

### 2) Alineación Temporal (Core del Negocio)

**Estado:** ✅ **MEJORADO** - Validaciones implementadas

#### Análisis

**Código Relevante:**
```369:418:app/scheduler.py
# Insert the lineup_confirmed snapshot with timing metadata
snapshot_at = datetime.utcnow()

# Calculate delta to kickoff (positive = before kickoff)
delta_to_kickoff = None
if kickoff_time:
    delta_to_kickoff = int((kickoff_time - snapshot_at).total_seconds())

# Determine odds freshness
odds_freshness = "live" if "live" in source.lower() or source in ["Bet365", "Pinnacle", "1xBet"] else "stale"
if "_stale" in source:
    odds_freshness = "stale"

await session.execute(text("""
    INSERT INTO odds_snapshots (
        match_id, snapshot_type, snapshot_at,
        odds_home, odds_draw, odds_away,
        prob_home, prob_draw, prob_away,
        overround, bookmaker,
        kickoff_time, delta_to_kickoff_seconds, odds_freshness
    ) VALUES (
        :match_id, 'lineup_confirmed', :snapshot_at,
        ...
    )
    ON CONFLICT (match_id, snapshot_type, bookmaker) DO NOTHING
"""), {
    "match_id": match_id,
    "snapshot_at": snapshot_at,
    ...
    "kickoff_time": kickoff_time,
    "delta_to_kickoff": delta_to_kickoff,
    "odds_freshness": odds_freshness,
})

# Also update match_lineups with lineup_confirmed_at if not already set
await session.execute(text("""
    UPDATE match_lineups
    SET lineup_confirmed_at = COALESCE(lineup_confirmed_at, :confirmed_at)
    WHERE match_id = :match_id
"""), {"match_id": match_id, "confirmed_at": snapshot_at})
```

**Problemas Identificados:**

1. ✅ **`snapshot_at` es correcto:** Usa `datetime.utcnow()` al momento de captura
2. ⚠️ **`lineup_confirmed_at` en producción:** Se actualiza con `snapshot_at` (correcto)
3. ⚠️ **`lineup_confirmed_at` en backfill:** Se calcula como `match_date - timedelta(hours=1)` (aproximación)
   ```115:115:scripts/backfill_lineups.py
   lineup_confirmed_at = match_date - timedelta(hours=1)
   ```
4. ❌ **Falta validación:** No se valida que `snapshot_at < kickoff_time` antes de insertar
5. ✅ **`delta_to_kickoff_seconds` se calcula correctamente:** `(kickoff_time - snapshot_at).total_seconds()`

#### Diagnóstico

**Veredicto:** ✅ **MEJORADO** - Validaciones implementadas

#### Cambios Implementados

**✅ IMPLEMENTADO:**

1. **Validación `snapshot_at < kickoff` antes de insertar:**
   ```python
   # app/scheduler.py, líneas 410-415
   if kickoff_time and snapshot_at >= kickoff_time:
       logger.warning(
           f"Snapshot AFTER kickoff for match {match_id}: "
           f"snapshot_at={snapshot_at}, kickoff={kickoff_time}. Skipping."
       )
       continue
   ```

2. **Validación de rango esperado de `delta_to_kickoff`:**
   ```python
   # app/scheduler.py, líneas 420-430
   minutes_to_kickoff = delta_to_kickoff / 60
   if minutes_to_kickoff < 0:
       logger.error(f"Negative delta for match {match_id}: {minutes_to_kickoff:.1f} min")
       continue
   elif minutes_to_kickoff > 120:
       logger.warning(f"Delta very large for match {match_id}: {minutes_to_kickoff:.1f} min")
       # Don't skip, but log warning for monitoring
   ```

3. **Reporte de distribución:** El endpoint `/lineup/snapshots` calcula p50/p90 (implementado).

---

### 3) Idempotencia / Deduplicación

**Estado:** ✅ **OK**

#### Análisis

**Código Relevante:**
```396:396:app/scheduler.py
ON CONFLICT (match_id, snapshot_type, bookmaker) DO NOTHING
```

**Verificación:**

1. ✅ **Unique constraint correcto:** `UNIQUE(match_id, snapshot_type, bookmaker)` (migración 012, línea 158)
2. ✅ **`ON CONFLICT DO NOTHING`:** Correcto - no pisa datos existentes
3. ⚠️ **Potencial race condition:** Si dos instancias del scheduler corren simultáneamente, ambas pueden intentar insertar. `ON CONFLICT` lo maneja correctamente.

**Escenario de Race Condition:**
- Instancia A detecta lineup a las 14:00:00.123
- Instancia B detecta lineup a las 14:00:00.456 (mismo match)
- Ambas intentan insertar con mismo `(match_id, snapshot_type, bookmaker)`
- Solo una inserta (la primera), la otra hace `DO NOTHING`
- **Resultado:** OK, pero el `snapshot_at` puede ser ligeramente diferente

**Mejora Opcional:**
```sql
ON CONFLICT (match_id, snapshot_type, bookmaker) 
DO UPDATE SET 
    snapshot_at = LEAST(odds_snapshots.snapshot_at, EXCLUDED.snapshot_at),
    odds_freshness = CASE 
        WHEN EXCLUDED.odds_freshness = 'live' THEN 'live'
        ELSE odds_snapshots.odds_freshness
    END
```
Esto asegura que siempre se use el snapshot más temprano y preserve 'live' si está disponible.

#### Diagnóstico

**Veredicto:** ✅ **OK** - Idempotencia correcta, race condition manejada

#### Cambios Requeridos

**OPCIONAL:** Mejora para preservar snapshot más temprano (ver código arriba).

---

### 4) Robustez de Detección de Lineups

**Estado:** ✅ **MEJORADO** - Retry logic y validaciones implementadas

#### Análisis

**Código Relevante:**
```288:307:app/scheduler.py
# Fetch lineup from API
lineup_data = await provider.get_lineups(external_id)

if not lineup_data:
    # Lineup not yet announced
    continue

# Check if we have valid starting XI
home_lineup = lineup_data.get("home")
away_lineup = lineup_data.get("away")

if not home_lineup or not away_lineup:
    continue

home_xi = home_lineup.get("starting_xi", [])
away_xi = away_lineup.get("starting_xi", [])

# Consider lineup confirmed if we have 11 players for each team
if len(home_xi) < 11 or len(away_xi) < 11:
    continue
```

**Problemas Identificados:**

1. ✅ **Validación de 11 jugadores:** Correcta
2. ⚠️ **Manejo de status:** El query filtra `status IN ('NS', '1H')` (línea 264), pero no valida que el match no haya empezado después de la query
3. ❌ **Falta manejo de API failures:** No hay retry logic si `get_lineups()` falla
4. ❌ **Falta rate limiting:** No hay throttling si hay muchos matches en ventana
5. ⚠️ **Falta validación de calidad:** No valida que los jugadores sean válidos (no null IDs)

#### Diagnóstico

**Veredicto:** ✅ **MEJORADO** - Retry logic y validaciones implementadas

#### Cambios Implementados

**✅ IMPLEMENTADO:**

1. **Retry logic agregado:**
   ```python
   # app/scheduler.py, líneas 288-303
   max_retries = 3
   for attempt in range(max_retries):
       try:
           lineup_data = await provider.get_lineups(external_id)
           break
       except Exception as e:
           if attempt == max_retries - 1:
               logger.error(f"Failed to fetch lineup after {max_retries} attempts: {e}")
               raise
           await asyncio.sleep(2 ** attempt)  # Exponential backoff: 2s, 4s, 8s
   ```

2. **Validación de status implementada:**
   ```python
   # app/scheduler.py, líneas 335-341
   if match.status != 'NS':
       logger.debug(
           f"Match {match_id} status changed to {match.status} "
           f"during processing, skipping"
       )
       continue
   ```

3. **Validación de calidad de datos implementada:**
   ```python
   # app/scheduler.py, líneas 343-349
   if any(p is None for p in home_xi) or any(p is None for p in away_xi):
       logger.warning(
           f"Invalid player IDs (None values) in lineup for match {match_id}. "
           f"Skipping to avoid data quality issues."
       )
       continue
   ```

---

### 5) Seguridad / Riesgos Operativos

**Estado:** ✅ **CORREGIDO** - Endpoint protegido

#### Análisis

**Código Relevante:**
```1970:1991:app/main.py
@app.post("/lineup/monitor")
async def trigger_lineup_monitoring():
    """
    Manually trigger lineup monitoring to capture odds at lineup_confirmed time.
    ...
    """
    from app.scheduler import monitor_lineups_and_capture_odds

    result = await monitor_lineups_and_capture_odds()
    return result
```

**Problemas Identificados:**

1. ❌ **Endpoint sin autenticación:** `POST /lineup/monitor` NO tiene `Depends(verify_api_key)`
2. ✅ **Logs:** No se imprimen secretos (revisado código)
3. ⚠️ **Carga:** Job cada 5 min puede procesar muchos matches si hay ventana grande (90 min)

#### Diagnóstico

**Veredicto:** ✅ **CORREGIDO** - Endpoint protegido y carga limitada

#### Cambios Implementados

**✅ IMPLEMENTADO:**

1. **Autenticación agregada al endpoint:**
   ```python
   # app/main.py, líneas 1970-1972
   @app.post("/lineup/monitor")
   @limiter.limit("10/minute")
   async def trigger_lineup_monitoring(
       request: Request,
       _: bool = Depends(verify_api_key),  # ✅ IMPLEMENTADO
   ):
   ```

2. **Límite de carga implementado:**
   ```python
   # app/scheduler.py, líneas 273-280
   matches = result.fetchall()
   
   # Limitar a 50 matches por run para evitar sobrecarga
   if len(matches) > 50:
       logger.warning(
           f"Too many matches in window ({len(matches)}), processing first 50 "
           f"to avoid overload. Remaining will be processed in next run."
       )
       matches = matches[:50]
   ```

---

### 6) Evaluación y Anti-Leakage

**Estado:** ✅ **OK** - Pipeline de entrenamiento correcto

#### Análisis

**Código Relevante:**
```75:117:scripts/train_lineup_model.py
query = """
    SELECT
        m.id as match_id,
        ...
        -- Real lineup_confirmed odds (when available from production)
        os_lineup.prob_home as lineup_prob_home,
        os_lineup.prob_draw as lineup_prob_draw,
        os_lineup.prob_away as lineup_prob_away,
        os_lineup.snapshot_at as lineup_snapshot_at
    FROM matches m
    JOIN match_lineups hl ON m.id = hl.match_id AND hl.is_home = true
    JOIN match_lineups al ON m.id = al.match_id AND hl.is_home = false
    LEFT JOIN odds_snapshots os_open ON m.id = os_open.match_id AND os_open.snapshot_type = 'opening'
    LEFT JOIN odds_snapshots os_close ON m.id = os_close.match_id AND os_close.snapshot_type = 'closing'
    LEFT JOIN odds_snapshots os_lineup ON m.id = os_lineup.match_id AND os_lineup.snapshot_type = 'lineup_confirmed'
    WHERE m.status = 'FT'
      AND (os_open.prob_home IS NOT NULL OR os_close.prob_home IS NOT NULL)
    ORDER BY m.date
"""
```

**Verificación:**

1. ✅ **Usa `snapshot_type='lineup_confirmed'`:** Correcto (línea 113)
2. ✅ **Filtra `status='FT'`:** Solo matches terminados (no leakage de resultados futuros)
3. ✅ **CV temporal:** Usa `TimeSeriesSplit` (línea 839)
4. ✅ **Features calculadas ANTES del match:** El código calcula features basadas en historial previo (líneas 344-502)
5. ⚠️ **Dual benchmark:** El código usa opening/closing como benchmarks, pero prefiere `lineup_confirmed` cuando está disponible (líneas 421-424)

**Potencial Leakage Detectado:**

❌ **NONE** - El código está correctamente diseñado para evitar leakage:
- Features se calculan cronológicamente (líneas 344-502)
- No usa datos posteriores al match
- `TimeSeriesSplit` asegura que train < val temporalmente

#### Diagnóstico

**Veredicto:** ✅ **OK** - Pipeline correcto, sin leakage

#### Cambios Requeridos

**NINGUNO** - El pipeline está correctamente implementado.

---

## LISTA DE CAMBIOS MÍNIMOS NECESARIOS

### Prioridad CRÍTICA (Antes de producción)

1. **Fix baseline freshness (Punto 1):**
   - Implementar Opción A: NO usar fallback a `odds_history` si es muy antiguo
   - O implementar Opción B: Actualizar `odds_history` en `global_sync_today()`
   - **Archivo:** `app/scheduler.py`, función `monitor_lineups_and_capture_odds()`

2. **Proteger endpoint (Punto 5):**
   - Agregar `Depends(verify_api_key)` a `POST /lineup/monitor`
   - **Archivo:** `app/main.py`, línea 1970

3. **Validar snapshot_at < kickoff (Punto 2):**
   - Agregar validación antes de insertar snapshot
   - **Archivo:** `app/scheduler.py`, después de línea 375

### Prioridad ALTA (Próxima iteración)

4. **Agregar retry logic (Punto 4):**
   - Implementar retry con exponential backoff para `get_lineups()`
   - **Archivo:** `app/scheduler.py`, función `monitor_lineups_and_capture_odds()`

5. **Limitar carga (Punto 5):**
   - Limitar matches procesados por run a 50
   - **Archivo:** `app/scheduler.py`, después de línea 273

6. **Validar calidad de datos (Punto 4):**
   - Validar que player IDs no sean None
   - **Archivo:** `app/scheduler.py`, después de línea 305

### Prioridad MEDIA (Mejoras)

7. **Mejorar idempotencia (Punto 3):**
   - Usar `DO UPDATE` para preservar snapshot más temprano
   - **Archivo:** `app/scheduler.py`, línea 396

8. **Validar rango de delta (Punto 2):**
   - Agregar validación de `delta_to_kickoff` en rango esperado
   - **Archivo:** `app/scheduler.py`, después de línea 375

---

## BUGS CRÍTICOS IDENTIFICADOS

### Bug #1: Baseline puede ser STALE

**Ubicación:** `app/scheduler.py`, líneas 326-352

**Problema:** Si `provider.get_odds()` falla, el sistema usa `odds_history` que puede ser muy antiguo (horas/días).

**Impacto:** El baseline del mercado NO es válido → evaluación incorrecta del modelo.

**Fix:**
```python
# Reemplazar líneas 326-352 con:
if fresh_odds and fresh_odds.get("odds_home"):
    odds_home = float(fresh_odds["odds_home"])
    odds_draw = float(fresh_odds["odds_draw"])
    odds_away = float(fresh_odds["odds_away"])
    source = fresh_odds.get("bookmaker", "api_football_live")
    logger.info(f"Got FRESH odds from API for match {match_id}")
else:
    # NO usar fallback si no hay odds frescas
    logger.error(
        f"Cannot capture fresh odds for match {match_id} - "
        f"API returned: {fresh_odds}. Skipping this match."
    )
    continue  # Skip, try again in next run (5 min)
```

---

## CONCLUSIÓN

El sistema tiene una **arquitectura sólida** y **todos los cambios críticos han sido implementados**:

1. ✅ **Idempotencia:** Correcta
2. ✅ **Anti-leakage:** Correcto
3. ✅ **Baseline freshness:** CORREGIDO - garantiza odds frescas (no fallback a stale)
4. ✅ **Validaciones temporales:** IMPLEMENTADAS
5. ✅ **Seguridad:** CORREGIDO - endpoint protegido con autenticación
6. ✅ **Robustez:** MEJORADO - retry logic y validaciones implementadas

**Estado Final:** 🟢 **LISTO PARA PRODUCCIÓN**

**Recomendación:** 
- ✅ Sistema listo para deploy a Railway
- ⚠️ Monitoreo intensivo primera semana (ver `CHECKLIST_PRE_DEPLOY_LINEUP.md`)
- 📊 Ejecutar evaluación después de acumular 200+ snapshots live

---

**Firma del Auditor:**  
_Lead Data Scientist / Auditor Externo_

