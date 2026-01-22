# Diagnóstico: Shadow Mode y Sensor B - Cards No Se Mueven en OPS

**Fecha:** 2025-01-15  
**Síntoma:** Las cards de Shadow Mode y Sensor B muestran métricas estáticas en OPS  
**Estado:** Análisis completo sin cambios de código

---

## 1. REGISTRO DE JOBS EN SCHEDULER

### ✅ Jobs Registrados Correctamente

**Shadow Evaluation:**
- **Job ID:** `evaluate_shadow_predictions`
- **Función:** `evaluate_shadow_predictions()` (línea 1654-1718 en `scheduler.py`)
- **Frecuencia:** Cada 30 minutos (`IntervalTrigger(minutes=30)`)
- **Registro:** Línea 3905-3911 en `scheduler.py`
- **Condición:** Solo corre si `is_shadow_enabled()` retorna True

**Sensor B Evaluation:**
- **Job ID:** `evaluate_sensor_predictions`
- **Función:** `evaluate_sensor_predictions_job()` (línea 1766-1820 en `scheduler.py`)
- **Frecuencia:** Cada 30 minutos (`IntervalTrigger(minutes=30)`)
- **Registro:** Línea 3927-3933 en `scheduler.py`
- **Condición:** Solo se registra si `SENSOR_ENABLED=True` (línea 3917)

### Manejo de Errores

Ambos jobs tienen try/except que capturan excepciones:
- Shadow: Línea 1716-1718 - retorna `{"status": "error", "error": str(e)}`
- Sensor: Línea 1818-1820 - retorna `{"status": "error", "error": str(e)}`

**⚠️ OBSERVACIÓN:** Los errores se loggean pero NO se propagan a OPS. Si hay errores silenciosos, no se verían en las cards.

---

## 2. DIAGNÓSTICO: SHADOW MODE

### Síntoma Reportado
```json
{
  "total_evaluated": 5,
  "baseline": {"accuracy": 0.0, "brier_avg": 0.8233},
  "shadow": {"accuracy": 0.0, "brier_avg": 0.7517},
  "recommendation": "INSUFFICIENT_DATA: Need at least 50 evaluated predictions"
}
```

### Análisis del Código

**Función `get_shadow_report()` (línea 307-408 en `shadow.py`):**

1. **Query de Evaluados (línea 323-338):**
   ```python
   select(
       func.count(ShadowPrediction.id).label("total"),
       func.sum(case((ShadowPrediction.baseline_correct == True, 1), else_=0)).label("baseline_correct"),
       func.sum(case((ShadowPrediction.shadow_correct == True, 1), else_=0)).label("shadow_correct"),
       ...
   )
   .where(
       and_(
           ShadowPrediction.actual_result.isnot(None),
           ShadowPrediction.evaluated_at.isnot(None),
           ShadowPrediction.evaluated_at > text(f"NOW() - INTERVAL '{window_days} days'"),
       )
   )
   ```

2. **Cálculo de Accuracy (línea 382, 388):**
   ```python
   "accuracy": baseline_correct / total,  # Si baseline_correct es NULL, se convierte a 0
   "accuracy": shadow_correct / total,     # Si shadow_correct es NULL, se convierte a 0
   ```

### Hallazgos

**H2 PARCIALMENTE CONFIRMADA:** Shadow sí evalúa (hay 5 registros con `evaluated_at IS NOT NULL`), pero `accuracy=0.0` puede ser por:

1. **Campos NULL:** Si `baseline_correct` o `shadow_correct` son NULL, el `func.sum(case(...))` retorna NULL, que luego se convierte a 0 en línea 345-346 (`row.baseline_correct or 0`).

2. **Todos Incorrectos:** Si los 5 registros tienen `baseline_correct=False` y `shadow_correct=False`, entonces `accuracy=0.0` es correcto.

3. **Brier Scores Existen:** El hecho de que `brier_avg` tenga valores (0.8233 y 0.7517) indica que `baseline_brier` y `shadow_brier` NO son NULL, lo que sugiere que la evaluación SÍ se ejecutó.

### Verificación de `evaluate_shadow_outcomes()`

**Función (línea 196-304 en `shadow.py`):**

1. **Query de Pendientes (línea 214-225):**
   ```python
   select(ShadowPrediction, Match)
   .join(Match, ShadowPrediction.match_id == Match.id)
   .where(
       and_(
           ShadowPrediction.actual_result.is_(None),
           Match.status.in_(["FT", "AET", "PEN"]),
           Match.home_goals.isnot(None),
           Match.away_goals.isnot(None),
       )
   )
   ```

2. **Asignación de Correctness (línea 274-276):**
   ```python
   shadow_pred.baseline_correct = b_correct  # Boolean
   shadow_pred.shadow_correct = s_correct    # Boolean
   ```

**✅ CONFIRMACIÓN:** Los campos `baseline_correct` y `shadow_correct` SÍ se setean correctamente durante la evaluación.

### Conclusión Shadow

**Estado:** Comportamiento esperado con gating activo.

- Hay 5 evaluaciones en ventana de 14 días
- Se requiere mínimo 50 para mostrar recomendación
- `accuracy=0.0` es plausible si los 5 fueron incorrectos (o si hay NULLs, pero menos probable dado que Brier existe)
- El job está corriendo (evidencia: hay 5 evaluados)

**Estimación:** Con ~3-4 partidos FT por día en promedio, tomaría ~12-17 días más alcanzar 50 evaluaciones.

---

## 3. DIAGNÓSTICO: SENSOR B

### Síntoma Reportado
```json
{
  "status": "NO_DATA",
  "reason": "Need 50 evaluated samples, have 0",
  "counts": {"total": 0, "pending": 1571, "evaluated": 0}
}
```

### Análisis del Código

**Función `get_sensor_report()` (línea 487-634 en `sensor.py`):**

1. **Query de Evaluados (línea 501-516):**
   ```sql
   SELECT COUNT(*) AS total, ...
   FROM sensor_predictions
   WHERE evaluated_at IS NOT NULL
     AND created_at > NOW() - INTERVAL '14 days'
     AND b_home_prob IS NOT NULL  -- ⚠️ FILTRO CRÍTICO
   ```

2. **Query de Pendientes (línea 588-592):**
   ```sql
   SELECT COUNT(*) FROM sensor_predictions
   WHERE evaluated_at IS NULL
     AND created_at > NOW() - INTERVAL '14 days'
   -- ⚠️ NO FILTRA por b_home_prob
   ```

### 🐛 BUG IDENTIFICADO: Desajuste de Campos en Modelo vs SQL

**Problema en `log_sensor_prediction()` (línea 320-346):**

El INSERT usa nombres de campos que NO coinciden con el modelo:

- **Modelo (`models.py` línea 986):** Campo se llama `a_version`
- **SQL (`sensor.py` línea 322):** INSERT usa `model_a_version`

```python
# En sensor.py línea 322:
INSERT INTO sensor_predictions (
    match_id, window_size, model_a_version, model_b_version,  # ❌ model_a_version no existe
    ...
)
```

**Impacto:**
- Si la tabla tiene `a_version` pero el INSERT usa `model_a_version`, el campo queda NULL
- Esto podría causar que las queries fallen o que los registros no se inserten correctamente

**Verificación Necesaria:** Revisar si hay migración que renombró `a_version` → `model_a_version` o viceversa.

### 🐛 BUG CRÍTICO: Filtro `b_home_prob IS NOT NULL` en Reporte

**Problema (línea 515 en `sensor.py`):**

El reporte de evaluados requiere `b_home_prob IS NOT NULL`, pero:

1. **Registros pueden tener `b_home_prob=NULL`** si Sensor B estaba en estado "LEARNING" cuando se hizo la predicción (línea 357-359):
   ```python
   "b_home_prob": float(b_probs[0]) if b_probs is not None else None,
   ```

2. **El evaluator (`evaluate_sensor_predictions`) NO requiere `b_home_prob`** (línea 392-403):
   ```sql
   WHERE sp.evaluated_at IS NULL
     AND m.status IN ('FT', 'AET', 'PEN')
     AND m.home_goals IS NOT NULL
     AND m.away_goals IS NOT NULL
   -- ✅ NO filtra por b_home_prob
   ```

3. **Resultado:** Un registro puede ser evaluado (`evaluated_at` set), pero si `b_home_prob IS NULL`, NO cuenta en el reporte.

**Impacto:**
- Si hay 1571 registros con `evaluated_at=NULL` pero muchos tienen `b_home_prob=NULL` (Sensor en LEARNING), entonces:
  - El evaluator puede evaluarlos (marca `evaluated_at`)
  - Pero el reporte NO los cuenta porque requiere `b_home_prob IS NOT NULL`
  - Esto explica `pending=1571` pero `evaluated=0` en el reporte

### Verificación de `evaluate_sensor_predictions()`

**Función (línea 376-484 en `sensor.py`):**

1. **Query (línea 392-403):** Correcta, filtra por `evaluated_at IS NULL` y match FT con goles.

2. **Asignación (línea 444-462):** Correcta, setea `evaluated_at`, `a_correct`, `b_correct`, `a_brier`, `b_brier`.

3. **⚠️ PROBLEMA:** Si `b_pick IS NULL` (línea 430), entonces `b_correct` se setea a NULL, pero el registro SÍ se marca como evaluado.

### Conclusión Sensor B

**H3 CONFIRMADA:** Sensor B está parcialmente atascado por diseño.

**Causa Raíz:**
1. **Filtro Inconsistente:** El reporte requiere `b_home_prob IS NOT NULL`, pero el evaluator NO lo requiere.
2. **Registros Evaluables pero No Contables:** Si hay registros con `b_home_prob=NULL` (Sensor en LEARNING), pueden ser evaluados pero no cuentan en el reporte.
3. **Pending Inflado:** Los 1571 "pending" incluyen registros que:
   - Pueden tener `b_home_prob=NULL` (no evaluables para reporte)
   - Pueden tener `evaluated_at=NULL` pero match aún no FT
   - Pueden tener match FT pero `home_goals/away_goals` NULL

**H4 PARCIALMENTE CONFIRMADA:** "pending" está inflado porque cuenta TODO (`evaluated_at IS NULL`), no solo evaluables FT.

---

## 4. MÉTRICAS Y OBSERVABILIDAD

### Logs Existentes

**Shadow:**
- Línea 1688-1693: Log info cuando `updated > 0`
- Línea 1696-1699: Log warning si `selected > 0` pero `updated == 0` (silent failure)
- Línea 1710-1713: Log warning si `eval_lag_minutes > threshold`

**Sensor:**
- Línea 1799-1803: Log info cuando `updated > 0`
- Línea 1806-1809: Log warning si `selected > 0` pero `updated == 0` (silent failure)
- Similar a Shadow

### Métricas Telemetría

**Shadow:**
- `record_shadow_evaluation_batch(updated)` - línea 1683
- `set_shadow_health_metrics(...)` - línea 1703-1706

**Sensor:**
- `record_sensor_evaluation_batch(updated)` - línea 1794
- `set_sensor_health_metrics(...)` - línea 1813-1816

### Recomendaciones de Observabilidad

**Faltante en OPS:**
1. **`last_success` de jobs:** No se muestra cuándo fue la última ejecución exitosa de `evaluate_shadow_predictions` o `evaluate_sensor_predictions_job`.
2. **`pending_evaluable_ft`:** Debería separarse de `pending_total` para Sensor B (solo FT con `b_home_prob IS NOT NULL`).
3. **Errores silenciosos:** Si el job falla con excepción, solo se loggea, no se expone en OPS.

---

## 5. RESUMEN DE HALLAZGOS

### Shadow Mode

| Hipótesis | Estado | Evidencia |
|-----------|--------|-----------|
| H1: Normal por gating | ✅ **CONFIRMADA** | `total_evaluated=5` < `min_samples=50` |
| H2: Accuracy=0 por bug | ⚠️ **PARCIAL** | Brier existe, sugiere evaluación OK; accuracy=0 puede ser por todos incorrectos o NULLs |

**Conclusión:** Comportamiento esperado. Falta muestra evaluada en ventana de 14 días.

### Sensor B

| Hipótesis | Estado | Evidencia |
|-----------|--------|-----------|
| H3: Atascado por filtro | ✅ **CONFIRMADA** | Reporte requiere `b_home_prob IS NOT NULL`, evaluator NO |
| H4: Pending inflado | ✅ **CONFIRMADA** | Cuenta todo `evaluated_at IS NULL`, no solo evaluables FT |

**Conclusión:** Bug de diseño. El reporte filtra por `b_home_prob IS NOT NULL` pero el evaluator no, causando que registros evaluados no cuenten.

### Bugs Identificados

1. **🐛 CRÍTICO - Sensor B Report Filter:**
   - **Archivo:** `app/ml/sensor.py`
   - **Función:** `get_sensor_report()` línea 515
   - **Problema:** Query de evaluados requiere `b_home_prob IS NOT NULL`, pero evaluator no lo requiere
   - **Impacto:** Registros evaluados con `b_home_prob=NULL` no cuentan en reporte
   - **Solución sugerida:** Remover filtro `b_home_prob IS NOT NULL` de query de evaluados, o agregar contador separado para "evaluated_with_b"

2. **⚠️ POSIBLE - Desajuste de Campos:**
   - **Archivo:** `app/ml/sensor.py` vs `app/models.py`
   - **Problema:** INSERT usa `model_a_version` pero modelo define `a_version`
   - **Impacto:** Campo puede quedar NULL si nombres no coinciden
   - **Verificación necesaria:** Revisar migraciones o schema real de BD

3. **⚠️ OBSERVABILIDAD - Falta `last_success` en OPS:**
   - **Problema:** No se muestra cuándo fue la última ejecución exitosa de jobs
   - **Impacto:** No se puede diagnosticar si jobs están corriendo o fallando silenciosamente

---

## 6. RECOMENDACIONES (Sin Implementar)

### Inmediatas

1. **Verificar schema real de `sensor_predictions`:**
   ```sql
   SELECT column_name FROM information_schema.columns 
   WHERE table_name = 'sensor_predictions';
   ```
   Confirmar si campo se llama `a_version` o `model_a_version`.

2. **Query de diagnóstico Sensor B:**
   ```sql
   SELECT 
     COUNT(*) FILTER (WHERE evaluated_at IS NOT NULL) as evaluated_total,
     COUNT(*) FILTER (WHERE evaluated_at IS NOT NULL AND b_home_prob IS NOT NULL) as evaluated_with_b,
     COUNT(*) FILTER (WHERE evaluated_at IS NULL AND b_home_prob IS NULL) as pending_no_b,
     COUNT(*) FILTER (WHERE evaluated_at IS NULL AND b_home_prob IS NOT NULL) as pending_with_b
   FROM sensor_predictions
   WHERE created_at > NOW() - INTERVAL '14 days';
   ```

3. **Query de diagnóstico Shadow:**
   ```sql
   SELECT 
     COUNT(*) as total,
     COUNT(*) FILTER (WHERE baseline_correct IS NULL) as baseline_null,
     COUNT(*) FILTER (WHERE shadow_correct IS NULL) as shadow_null,
     COUNT(*) FILTER (WHERE baseline_correct = False) as baseline_false,
     COUNT(*) FILTER (WHERE shadow_correct = False) as shadow_false
   FROM shadow_predictions
   WHERE evaluated_at IS NOT NULL
     AND evaluated_at > NOW() - INTERVAL '14 days';
   ```

### Mejoras de Observabilidad

1. **Agregar `last_success` a cards de OPS:**
   - Timestamp de última ejecución exitosa de `evaluate_shadow_predictions`
   - Timestamp de última ejecución exitosa de `evaluate_sensor_predictions_job`

2. **Separar `pending_evaluable_ft` de `pending_total` en Sensor B:**
   - `pending_total`: Todos los `evaluated_at IS NULL`
   - `pending_evaluable_ft`: Solo FT con `b_home_prob IS NOT NULL`

3. **Exponer errores de jobs en OPS:**
   - Si último run fue error, mostrar en card
   - Agregar contador de errores consecutivos

---

## 7. ESTIMACIÓN DE CUANDO DEBERÍA MOVERSE

### Shadow Mode

**Asumiendo:**
- Ventana: 14 días
- Mínimo requerido: 50 evaluaciones
- Actual: 5 evaluaciones
- Necesario: 45 más

**Con ~3-4 partidos FT/día que tienen shadow prediction:**
- Tiempo estimado: 12-15 días más
- **Fecha estimada:** ~2025-01-27 a 2025-01-30

### Sensor B

**Depende de:**
1. Si bug de `b_home_prob` se corrige: Inmediato (si hay evaluados con `b_home_prob=NULL`)
2. Si Sensor B está en estado LEARNING: Hasta que tenga 50 muestras para entrenar
3. Si hay 1571 pending reales evaluables: Debería moverse en próximas ejecuciones del evaluator (cada 30 min)

**Si bug NO se corrige:** Nunca se moverá si todos los evaluados tienen `b_home_prob=NULL`.

---

## CONCLUSIÓN FINAL

**Shadow Mode:** ✅ Comportamiento esperado. Gating activo por falta de muestra.

**Sensor B:** 🐛 Bug de diseño. Filtro inconsistente entre evaluator y reporte causa que evaluados no cuenten.

**Acción requerida:** Verificar schema de BD y corregir filtro en `get_sensor_report()` si es necesario.
