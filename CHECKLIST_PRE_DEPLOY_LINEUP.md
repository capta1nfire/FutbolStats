# ✅ CHECKLIST PRE-DEPLOY: Lineup Arbitrage System

**Fecha:** 2025-01-XX  
**Estado:** 🟢 LISTO PARA DEPLOY

---

## 🔒 SEGURIDAD Y ROBUSTEZ

### ✅ Implementado
- [x] Endpoint `/lineup/monitor` protegido con `verify_api_key`
- [x] Rate limiting aplicado (`@limiter.limit("10/minute")`)
- [x] No se imprimen secretos en logs
- [x] Retry logic con exponential backoff (3 intentos: 2s, 4s, 8s)
- [x] Límite de 50 matches por run para evitar sobrecarga

---

## 📊 CORRECTITUD DEL BASELINE

### ✅ Implementado
- [x] **NO HAY FALLBACK a odds stale** - Si API no devuelve odds frescas, skip y retry en 5 min
- [x] Priorización de bookmakers: Bet365 > Pinnacle > 1xBet (sharp bookmakers)
- [x] Source guardado como `{bookmaker}_live` para tracking
- [x] Campo `odds_freshness` marcado como `'live'` cuando viene de API

### ⚠️ Verificar en Producción
- [ ] Monitorear tasa de éxito de captura de odds frescas (target: >95%)
- [ ] Alertar si tasa de fallos de API > 10% por hora
- [ ] Verificar que `odds_freshness='live'` en snapshots capturados

---

## ⏰ ALINEACIÓN TEMPORAL

### ✅ Implementado
- [x] Validación `snapshot_at < kickoff_time` antes de insertar
- [x] Validación de `delta_to_kickoff` en rango esperado (0-120 minutos)
- [x] Cálculo correcto de `delta_to_kickoff_seconds` = `(kickoff_time - snapshot_at).total_seconds()`
- [x] `lineup_confirmed_at` actualizado con timestamp real (no aproximación)

### ⚠️ Verificar en Producción
- [ ] Distribución de `delta_to_kickoff`: p50 ~60 min, p90 < 90 min
- [ ] No hay snapshots con `delta_to_kickoff < 0` (después de kickoff)
- [ ] Endpoint `/lineup/snapshots` muestra timing_stats correctamente

---

## 🔄 IDEMPOTENCIA Y DEDUPLICACIÓN

### ✅ Implementado
- [x] Unique constraint: `(match_id, snapshot_type, bookmaker)`
- [x] `ON CONFLICT DO NOTHING` - no pisa datos existentes
- [x] Race condition manejada correctamente (múltiples instancias)

### ⚠️ Verificar en Producción
- [ ] No hay duplicados en `odds_snapshots` con mismo `(match_id, snapshot_type, bookmaker)`
- [ ] Si dos instancias detectan lineup simultáneamente, solo una inserta

---

## 🎯 DETECCIÓN DE LINEUPS

### ✅ Implementado
- [x] Validación de 11 jugadores por equipo
- [x] Validación de calidad: player IDs no None
- [x] Validación doble de status: query inicial + después de API call
- [x] Retry logic para `get_lineups()` con exponential backoff

### ⚠️ Verificar en Producción
- [ ] Tasa de detección de lineups: >80% de matches con lineups disponibles
- [ ] No hay snapshots con `home_xi_count < 11` o `away_xi_count < 11`
- [ ] Logs muestran retries cuando API falla temporalmente

---

## 📈 EVALUACIÓN Y ANTI-LEAKAGE

### ✅ Implementado
- [x] Script de evaluación: `scripts/evaluate_lineup_arbitrage.py`
- [x] Filtro por `odds_freshness='live'` disponible (`--live-only`)
- [x] Bootstrap CI 95% para validación estadística
- [x] Check de sesgo live vs stale (alerta si diferencia > 0.01)

### ⚠️ Verificar en Producción
- [ ] Ejecutar evaluación después de acumular 200+ snapshots live
- [ ] Verificar que CI 95% excluye cero antes de decidir CONTINUE/CLOSE
- [ ] Comparar Brier Score entre grupos live/stale para detectar sesgo

---

## 🗄️ DATOS Y BACKFILL

### ✅ Estado Actual
- [x] Backfill de opening/closing odds en progreso (27k+ snapshots)
- [x] Tabla `odds_snapshots` con campos de timing (`delta_to_kickoff_seconds`, `odds_freshness`)
- [x] Tabla `match_lineups` con `lineup_confirmed_at` timestamp

### ⚠️ Verificar en Producción
- [ ] Backfill completo de opening/closing para matches históricos
- [ ] `match_lineups.lineup_confirmed_at` poblado para matches con lineups disponibles

---

## 🚀 DEPLOYMENT STEPS

### 1. Pre-Deploy Verification
```bash
# Verificar que no hay errores de linting
python -m pylint app/scheduler.py app/main.py

# Verificar que tests pasan (si existen)
pytest tests/  # Si hay tests

# Verificar configuración de API keys
echo $API_KEY  # Debe estar configurado en Railway
```

### 2. Deploy a Railway
```bash
# Railway debería detectar cambios automáticamente
# Verificar que scheduler se inicia correctamente
# Verificar logs: "Scheduler started: ... Lineup monitoring: Every 5 minutes"
```

### 3. Post-Deploy Monitoring (Primeras 24h)
```bash
# Verificar que el job corre cada 5 minutos
# Revisar logs para:
# - "Lineup confirmed for match X"
# - "Got FRESH odds from {bookmaker} for match X"
# - NO debería haber: "Using STALE odds" (ya no existe fallback)

# Verificar snapshots capturados
psql $DATABASE_URL -c "
  SELECT COUNT(*), odds_freshness, 
         AVG(delta_to_kickoff_seconds/60) as avg_minutes_before_kickoff
  FROM odds_snapshots
  WHERE snapshot_type = 'lineup_confirmed'
    AND snapshot_at > NOW() - INTERVAL '24 hours'
  GROUP BY odds_freshness;
"
```

### 4. Acumulación de Datos (2-4 semanas)
- [ ] Esperar acumulación de 200+ snapshots con `odds_freshness='live'`
- [ ] Monitorear distribución de timing (p50/p90 de `delta_to_kickoff`)
- [ ] Verificar tasa de éxito de captura (>95% target)

### 5. Evaluación Final
```bash
# Ejecutar evaluación con CI 95%
python scripts/evaluate_lineup_arbitrage.py \
  --min-snapshots 200 \
  --live-only \
  --bootstrap-n 1000

# Decisión basada en:
# - CI 95% excluye cero → CONTINUE proyecto
# - CI 95% incluye cero → CLOSE proyecto (no hay alpha)
```

---

## 📋 MÉTRICAS DE ÉXITO

### Semana 1-2
- [ ] Tasa de captura de odds frescas: >90%
- [ ] Snapshots capturados: >50 con `odds_freshness='live'`
- [ ] No hay errores críticos en logs

### Semana 3-4
- [ ] Snapshots acumulados: >200 con `odds_freshness='live'`
- [ ] Distribución de timing: p50 ~60 min, p90 < 90 min
- [ ] Tasa de detección de lineups: >80%

### Evaluación Final
- [ ] CI 95% de delta Brier Score calculado
- [ ] Decisión CONTINUE/CLOSE basada en CI
- [ ] Reporte de sesgo live vs stale (si aplica)

---

## 🐛 MONITOREO Y ALERTAS

### Alertas Críticas
- [ ] Tasa de fallos de API > 10% por hora
- [ ] Snapshots con `delta_to_kickoff < 0` (después de kickoff)
- [ ] Duplicados en `odds_snapshots` con mismo `(match_id, snapshot_type, bookmaker)`

### Alertas de Advertencia
- [ ] Tasa de captura de odds frescas < 90%
- [ ] Distribución de timing fuera de rango esperado (p50 < 45 min o > 75 min)
- [ ] Sesgo detectado entre grupos live/stale (diferencia > 0.01)

---

## 📝 DOCUMENTACIÓN

### ✅ Completado
- [x] `AUDITORIA_LINEUP_ARBITRAGE.md` - Auditoría técnica completa
- [x] `CHECKLIST_PRE_DEPLOY_LINEUP.md` - Este documento
- [x] Scripts de evaluación documentados

### ⚠️ Pendiente (Opcional)
- [ ] Documentar proceso de evaluación en README
- [ ] Crear dashboard de métricas en Railway (opcional)

---

## ✅ FIRMA DE APROBACIÓN

**Auditor:** Lead Data Scientist  
**Fecha:** 2025-01-XX  
**Estado:** 🟢 **APROBADO PARA DEPLOY**

**Cambios Críticos Implementados:**
- ✅ Baseline freshness garantizado (no fallback a stale)
- ✅ Validaciones temporales implementadas
- ✅ Seguridad del endpoint corregida
- ✅ Robustez mejorada (retry, validaciones)

**Riesgos Residuales Mitigados:**
- ✅ Calidad de odds: Priorización de sharp bookmakers
- ✅ Zona horaria: Verificado UTC consistente
- ✅ Sesgo live/stale: Check en evaluación implementado

**Próximo Paso:** Deploy a Railway y monitoreo intensivo primera semana.

