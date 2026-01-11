# AUDITORÍA DE CÓDIGO - Cambios Recientes Sesión
**Fecha**: 2026-01-09  
**Auditor**: Senior (Python/FastAPI/SQLAlchemy/Postgres/DuckDB/APScheduler)  
**Scope**: Cambios recientes de esta sesión  
**Enfoque**: Robustez PIT/anti-leakage, dashboards DB-backed, backward compatibility

---

## RESUMEN EJECUTIVO

Se identificaron **6 hallazgos** (1 P0 bloqueante, 2 P1 importantes, 3 P2 mejoras). El código está bien estructurado con buen manejo de backward compatibility y anti-leakage, pero hay un import incorrecto que causará fallo en runtime y algunos edge cases menores.

---

## HALLAZGOS PRIORIZADOS

### 🔴 **P0: Import Incorrecto en `weekly_pit_report`**
**Archivo**: `app/scheduler.py`  
**Líneas**: 1937  
**Función**: `weekly_pit_report()`

**Problema**:  
```python
from app.db import async_engine  # ❌ INCORRECTO - app.db no existe
```

El módulo `app.db` no existe en el codebase. Esto causará `ImportError` cuando `weekly_pit_report()` se ejecute.

**Impacto**:  
- Job semanal falla completamente
- No se generan reportes semanales consolidados
- Dashboard no puede mostrar datos semanales

**Fix recomendado**:
```python
# Opción 1: Usar AsyncSessionLocal directamente (más común en el codebase)
from app.database import AsyncSessionLocal

# En lugar de:
# async with async_engine.connect() as conn:
#     result = await conn.execute(text("..."))

# Usar:
async with AsyncSessionLocal() as session:
    result = await session.execute(text("..."))

# Opción 2: Si realmente necesitas async_engine, verificar dónde se define
# (no existe en app.database según el codebase revisado)
```

**Evidencia**:
```1937:1938:app/scheduler.py
from app.db import async_engine
from app.database import AsyncSessionLocal
```

**Severidad**: **P0 - Bloqueante** (falla en runtime)

---

### 🟠 **P1: Anti-Leakage Edge Case - Timestamps Idénticos**
**Archivo**: `scripts/evaluate_pit_live_only.py`  
**Líneas**: 199  
**Función**: `get_pit_safe_prediction()`

**Problema**:  
```python
if pred_created <= snapshot_at_naive:  # Permite igualdad
    return pred
```

Si `pred_created == snapshot_at_naive` (mismo segundo/microsegundo), técnicamente hay leakage si el snapshot se capturó primero. Aunque el diseño permite `<=` según la nota en línea 664, es un edge case que puede causar falsos positivos en validaciones.

**Impacto**:  
- En sistemas de alta frecuencia, predicciones y snapshots pueden tener timestamps idénticos
- La validación anti-leakage puede pasar cuando debería fallar
- Métricas pueden estar ligeramente infladas

**Fix recomendado**:
```python
# Opción 1: Cambiar a < estricto (más seguro)
if pred_created < snapshot_at_naive:
    return pred

# Opción 2: Si el diseño requiere <=, documentar explícitamente y agregar validación
if pred_created <= snapshot_at_naive:
    if pred_created == snapshot_at_naive:
        # Log warning para monitoreo
        logger.debug(f"Edge case: pred_created == snapshot_at for match {match_id}")
    return pred
```

**Evidencia**:
```199:200:scripts/evaluate_pit_live_only.py
if pred_created <= snapshot_at_naive:
    return pred
```

**Nota**: El código en línea 664 dice `"PIT integrity enforced (pred.created_at <= snapshot_at)"`, lo cual sugiere que `<=` es intencional. Pero debería documentarse mejor o cambiarse a `<` para mayor seguridad.

**Severidad**: **P1 - Importante** (puede causar leakage sutil)

---

### 🟠 **P1: Falta Manejo de Errores en `_save_pit_report_to_db`**
**Archivo**: `app/scheduler.py`  
**Líneas**: 1877-1904  
**Función**: `_save_pit_report_to_db()`

**Problema**:  
Si `session.commit()` falla (línea 1903), no hay rollback explícito. Aunque SQLAlchemy hace rollback automático en excepciones, no hay logging del error antes de que la excepción se propague.

**Impacto**:  
- Si hay error de DB (ej: constraint violation, conexión perdida), el error se propaga sin contexto
- No hay diferenciación entre errores transitorios vs permanentes
- El job `daily_pit_evaluation()` captura el error genérico pero pierde detalles

**Fix recomendado**:
```python
async def _save_pit_report_to_db(report_type: str, payload: dict, source: str = "scheduler"):
    """..."""
    from sqlalchemy import text
    from app.database import AsyncSessionLocal
    import json

    today = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)

    async with AsyncSessionLocal() as session:
        try:
            await session.execute(text("""
                INSERT INTO pit_reports (report_type, report_date, payload, source, created_at, updated_at)
                VALUES (:report_type, :report_date, CAST(:payload AS JSON), :source, NOW(), NOW())
                ON CONFLICT (report_type, report_date) DO UPDATE SET
                    payload = CAST(EXCLUDED.payload AS JSON),
                    source = EXCLUDED.source,
                    updated_at = NOW()
            """), {
                "report_type": report_type,
                "report_date": today,
                "payload": json.dumps(payload),
                "source": source,
            })
            await session.commit()
            logger.info(f"Saved {report_type} PIT report to DB for {today.date()}")
        except Exception as e:
            await session.rollback()  # Explícito (aunque SQLAlchemy lo hace automáticamente)
            logger.error(f"Failed to save {report_type} PIT report to DB: {e}", exc_info=True)
            raise  # Re-raise para que el caller maneje
```

**Evidencia**:
```1888:1904:app/scheduler.py
async with AsyncSessionLocal() as session:
    # UPSERT: insert or update if exists for same type+date
    await session.execute(text("""
        INSERT INTO pit_reports (report_type, report_date, payload, source, created_at, updated_at)
        VALUES (:report_type, :report_date, CAST(:payload AS JSON), :source, NOW(), NOW())
        ON CONFLICT (report_type, report_date) DO UPDATE SET
            payload = CAST(EXCLUDED.payload AS JSON),
            source = EXCLUDED.source,
            updated_at = NOW()
    """), {
        "report_type": report_type,
        "report_date": today,
        "payload": json.dumps(payload),
        "source": source,
    })
    await session.commit()
    logger.info(f"Saved {report_type} PIT report to DB for {today.date()}")
```

**Severidad**: **P1 - Importante** (mejora robustez y debugging)

---

### 🟡 **P2: Token en Query Params Puede Aparecer en Logs**
**Archivo**: `app/main.py`  
**Líneas**: 2544, 3088-3107, 4237-4256  
**Función**: `_verify_dashboard_token()`, JavaScript en dashboards

**Problema**:  
Aunque el token se preserva correctamente en URLs (JavaScript líneas 3088-3107), no hay filtrado explícito de tokens en logs de acceso. Si FastAPI/uvicorn loguea requests con query params, el token puede aparecer en logs.

**Impacto**:  
- Tokens pueden filtrarse en logs de acceso HTTP
- Riesgo de seguridad si logs son accesibles

**Fix recomendado**:
```python
# Opción 1: Filtrar token de logs en middleware
@app.middleware("http")
async def filter_token_from_logs(request: Request, call_next):
    # Remover token de query params antes de logging
    if "token" in request.query_params:
        # Crear nueva URL sin token para logging
        filtered_url = str(request.url).replace(f"token={request.query_params['token']}", "")
        # Log filtered_url en lugar de request.url
    response = await call_next(request)
    return response

# Opción 2: Usar header en lugar de query param (más seguro)
# Ya existe soporte para X-Dashboard-Token header (línea 2544)
# Documentar que query param es solo para desarrollo/conveniencia
```

**Evidencia**:
```2544:2545:app/main.py
provided = request.headers.get("X-Dashboard-Token") or request.query_params.get("token")
return provided == token
```

**Nota**: El código ya soporta header `X-Dashboard-Token` (más seguro). El query param es para conveniencia. Debería documentarse que en producción se use header.

**Severidad**: **P2 - Mejora** (riesgo bajo, pero buena práctica)

---

### 🟡 **P2: Validación de N Bajo en Interpretation Podría Ser Más Estricta**
**Archivo**: `scripts/evaluate_pit_live_only.py`  
**Líneas**: 350-406, 625-634  
**Función**: `generate_interpretation()`

**Problema**:  
Aunque `generate_interpretation()` valida `phase == 'insufficient'` y `roi_ci_status == 'insufficient_n'`, no hay validación explícita de que `interpretation` solo se genere si `n_bets >= MIN_BETS_FOR_CI` (30). La función siempre retorna un dict, incluso con N=0.

**Impacto**:  
- Interpretation puede mostrar "HOLD" con `confidence="low"` incluso cuando N=0
- No hay diferenciación clara entre "no data" vs "insufficient data"

**Fix recomendado**:
```python
def generate_interpretation(phase: str, brier: dict, betting: dict) -> dict:
    """..."""
    n_bets = betting.get('n_bets', 0)
    
    # Early return si no hay datos
    if n_bets == 0:
        return {
            "confidence": "none",
            "verdict": "NO_DATA",
            "bullet_notes": ["no_bets: no predictions met edge threshold"],
        }
    
    # Resto de la lógica...
```

**Evidencia**:
```367:406:scripts/evaluate_pit_live_only.py
n_bets = betting.get('n_bets', 0)
roi_ci_status = betting.get('roi_ci_status', 'no_bets')
# ... resto de lógica siempre ejecuta, incluso si n_bets == 0
```

**Nota**: El código actual funciona pero podría ser más explícito sobre el caso N=0.

**Severidad**: **P2 - Mejora** (funcionalidad correcta, mejor UX)

---

### 🟡 **P2: Falta Validación de Payload JSON en `_save_pit_report_to_db`**
**Archivo**: `app/scheduler.py`  
**Líneas**: 1900  
**Función**: `_save_pit_report_to_db()`

**Problema**:  
`json.dumps(payload)` puede fallar si `payload` contiene objetos no serializables (ej: `datetime`, `Decimal`). No hay validación previa.

**Impacto**:  
- Si el payload tiene objetos no serializables, el job falla silenciosamente
- Error solo se ve en logs, no hay fallback

**Fix recomendado**:
```python
import json
from datetime import datetime
from decimal import Decimal

def json_serializer(obj):
    """Custom JSON serializer for datetime/Decimal."""
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, Decimal):
        return float(obj)
    raise TypeError(f"Type {type(obj)} not serializable")

# En _save_pit_report_to_db:
try:
    payload_json = json.dumps(payload, default=json_serializer)
except TypeError as e:
    logger.error(f"Payload not JSON serializable: {e}")
    raise
```

**Evidencia**:
```1900:1900:app/scheduler.py
"payload": json.dumps(payload),
```

**Nota**: Los payloads actuales probablemente son serializables (vienen de `evaluate_pit_live_only.py` que ya serializa), pero es buena práctica validar.

**Severidad**: **P2 - Mejora** (defensive programming)

---

## CHECKLIST DE INVARIANTES

### ✅ Anti-Leakage
- **Estado**: **CORRECTO** (con nota menor)
- **Verificación**:
  - `scripts/evaluate_pit_live_only.py` línea 199: `pred_created <= snapshot_at_naive` ✅
  - `scripts/build_pit_dataset.py` línea 110: `p.created_at <= ps.snapshot_at` ✅
  - `app/scheduler.py` línea 2495: `p.created_at < os.snapshot_at` ✅ (usa `<` estricto)
- **Nota**: Hay inconsistencia menor (`<=` vs `<`), pero ambos son válidos según diseño. El edge case de igualdad está documentado.

### ✅ Backward Compatibility
- **Estado**: **EXCELENTE**
- **Verificación**:
  - `scripts/evaluate_pit_ev.py` líneas 400-415: Detecta columnas `baseline_odds_*` y maneja ausencia ✅
  - `scripts/evaluate_pit_ev.py` líneas 674-680: Usa `.get()` seguro para acceso a columnas ✅
  - `app/main.py` líneas 2372-2378: Fallback a filesystem si DB falla ✅
  - `app/main.py` líneas 2400-2414: Soporta múltiples formatos de archivos legacy ✅

### ✅ Persistencia DB
- **Estado**: **CORRECTO** (con mejora recomendada en manejo de errores)
- **Verificación**:
  - `app/scheduler.py` líneas 1890-1903: UPSERT con `ON CONFLICT` ✅
  - `app/scheduler.py` línea 1903: `await session.commit()` ✅
  - `app/main.py` líneas 2441-2473: Lectura desde DB con fallback ✅
  - `app/models.py` líneas 509-534: Modelo `PITReport` con constraint único ✅

### ✅ Seguridad
- **Estado**: **BUENO** (con mejora recomendada)
- **Verificación**:
  - `app/main.py` línea 2544: Verifica header `X-Dashboard-Token` primero ✅
  - `app/main.py` líneas 3088-3107: Preserva token en URLs (JavaScript) ✅
  - `app/main.py` línea 3163: Endpoint `/dashboard/pit.json` requiere auth ✅
- **Nota**: Query param `?token=` es conveniente pero menos seguro. Debería documentarse uso de header en producción.

### ✅ Idempotencia
- **Estado**: **CORRECTO**
- **Verificación**:
  - `app/scheduler.py` líneas 1890-1896: `ON CONFLICT (report_type, report_date) DO UPDATE` ✅
  - `app/scheduler.py` líneas 2694-2700: `ON CONFLICT (day) DO UPDATE` para ops_rollups ✅
  - `app/models.py` línea 517: `UniqueConstraint("report_type", "report_date")` ✅

### ✅ Manejo de N Bajo
- **Estado**: **BUENO** (con mejora recomendada)
- **Verificación**:
  - `scripts/evaluate_pit_live_only.py` línea 64: `MIN_BETS_FOR_CI = 30` ✅
  - `scripts/evaluate_pit_live_only.py` línea 347: `MIN_PREDICTIONS_FOR_STABLE_METRICS = 30` ✅
  - `scripts/evaluate_pit_live_only.py` líneas 374-377: Valida `phase == 'insufficient'` ✅
  - `scripts/evaluate_pit_live_only.py` líneas 391-392: Valida `n_with_predictions < MIN_PREDICTIONS_FOR_STABLE_METRICS` ✅
- **Nota**: Podría ser más explícito sobre el caso N=0 (ver P2 arriba).

---

## VERIFICACIONES ADICIONALES RECOMENDADAS

### 1. **Validar Import de `async_engine`**
```python
# Verificar si existe app/db.py o si async_engine está en app/database.py
# Si no existe, usar AsyncSessionLocal directamente
```

### 2. **Query Anti-Leakage Timestamps Idénticos**
```sql
-- Detectar predicciones con timestamp igual al snapshot
SELECT 
    os.snapshot_id,
    p.id as prediction_id,
    os.snapshot_at,
    p.created_at,
    EXTRACT(EPOCH FROM (os.snapshot_at - p.created_at)) as diff_seconds
FROM odds_snapshots os
JOIN predictions p ON p.match_id = os.match_id
WHERE os.snapshot_type = 'lineup_confirmed'
  AND EXTRACT(EPOCH FROM (os.snapshot_at - p.created_at)) = 0  -- Exactamente igual
LIMIT 100;
```

### 3. **Validar Payloads JSON en DB**
```sql
-- Verificar que todos los payloads son JSON válidos
SELECT 
    id,
    report_type,
    report_date,
    jsonb_typeof(payload) as payload_type
FROM pit_reports
WHERE jsonb_typeof(payload) != 'object'
LIMIT 10;
```

### 4. **Monitorear Errores de Persistencia**
```python
# Agregar métrica en daily_ops_rollup para contar errores de persistencia
# Ej: "pit_reports_persist_errors": count
```

---

## CONCLUSIÓN

**Hallazgos críticos**: 1 (P0)  
**Hallazgos importantes**: 2 (P1)  
**Mejoras opcionales**: 3 (P2)

**Estado general**: **BUENO** con un bug bloqueante que debe corregirse antes de deploy.

**Recomendación inmediata**: 
1. Corregir import en `weekly_pit_report()` (P0)
2. Agregar manejo de errores explícito en `_save_pit_report_to_db()` (P1)
3. Considerar cambiar `<=` a `<` en anti-leakage si el diseño lo permite (P1)

**Mejoras opcionales** (P2) pueden implementarse en siguiente sprint.

---

**FIN DEL REPORTE**


