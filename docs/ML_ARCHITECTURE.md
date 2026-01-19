# ML Architecture - FutbolStats

Documentación técnica del sistema de predicciones ML.

## Modelo de Producción (v1.0.0)

### Configuración
- **Algoritmo**: XGBoost
- **Features**: 14 (form, head-to-head, odds implícitas)
- **Output**: Probabilidades 1X2 (home, draw, away)
- **Archivo**: `models/xgb_v1.0.0_*.json`

### Draw Label Canónico
**Draw = home_goals == away_goals** donde el score almacenado representa:
- **FT (Full Time)**: Resultado a los 90 minutos
- **PEN (Penalties)**: Score ANTES de penales (empate a 90'/120')
- **AET (After Extra Time)**: Score DESPUÉS de prórroga

### Training Dataset
- Solo partidos con `status = 'FT'` (~69K muestras)
- Draw rate: ~24.5%
- Excluye AET/PEN (dinámicas de copa knockout diferentes)

---

## Shadow Mode (Two-Stage Architecture)

### Propósito
Evaluar arquitectura two-stage para mejorar predicción de empates sin degradar métricas generales.

### Arquitectura
**Stage 1**: Binary classifier (draw vs non-draw)
- 18 features (17 base + implied_draw de odds)
- sample_weight: draws=1.2, otros=1.0

**Stage 2**: Binary classifier (home vs away) para non-draws
- 17 features (sin implied_draw)

### Composición de Probabilidades
```
p_draw = P(draw | Stage1)
p_home = (1 - p_draw) × P(home | non-draw, Stage2)
p_away = (1 - p_draw) × P(away | non-draw, Stage2)
```

### Config (env vars)
```
MODEL_ARCHITECTURE=baseline              # Activo en producción
MODEL_SHADOW_ARCHITECTURE=two_stage      # Shadow mode
MODEL_DRAW_THRESHOLD=0.0                 # Deshabilitado, usar argmax
```

### Criterios de GO/NO-GO
- **GO**: brier_shadow <= brier_baseline + 0.002 AND accuracy_drop < 2%
- **NO-GO**: Cualquier degradación fuera de tolerancia

### Estado Actual
Ver `/dashboard/ops.json` → `shadow_mode` para métricas actuales.

---

## Sensor B - Calibration Diagnostics

### Propósito
LogReg L2 sliding-window para detectar si el modelo de producción se volvió stale/rígido.
**SOLO DIAGNÓSTICO** - No afecta predicciones de producción.

### Arquitectura
- **Model A**: Modelo de producción (baseline)
- **Model B**: LogReg L2 simple (C=0.1, balanced classes)
- **Window**: Últimos N partidos FT (default 50)
- **Retrain**: Cada 6 horas

### Config (env vars)
```
SENSOR_ENABLED=true
SENSOR_WINDOW_SIZE=50
SENSOR_MIN_SAMPLES=50
SENSOR_RETRAIN_INTERVAL_HOURS=6
SENSOR_SIGNAL_SCORE_GO=1.1
SENSOR_SIGNAL_SCORE_NOISE=0.9
SENSOR_EVAL_WINDOW_DAYS=14
```

### Signal Score
```
signal = (brier_uniform - brier_B) / (brier_uniform - brier_A)
```
- **signal >= 1.1**: Sensor B mejora sobre Model A → revisar Model A
- **signal < 0.9**: Sensor B es ruido → Model A está bien calibrado
- **0.9 <= signal < 1.1**: Comparable → seguir monitoreando

### Estados
- **DISABLED**: SENSOR_ENABLED=false
- **LEARNING**: < min_samples, no reporta métricas
- **READY**: >= min_samples, reportando
- **ERROR**: Fallo en entrenamiento

### Governance
1. Sensor B es SOLO diagnóstico interno
2. NUNCA afecta predicciones de producción
3. Si signal > 1.1 consistentemente: revisar Model A manualmente
4. No tomar decisiones con < 100 samples evaluados

---

## Telemetría ML

### Métricas Prometheus (`/metrics`)
```
# Shadow Mode
shadow_predictions_logged_total
shadow_predictions_evaluated_total
shadow_predictions_errors_total
shadow_eval_lag_minutes
shadow_pending_ft_to_evaluate

# Sensor B
sensor_predictions_logged_total
sensor_predictions_evaluated_total
sensor_predictions_errors_total
sensor_retrain_runs_total{status}
sensor_state  # 0=disabled, 1=learning, 2=ready, 3=error
```

### Health en ops.json
```json
{
  "shadow_mode": {
    "health": {
      "pending_ft_to_evaluate": 0,
      "eval_lag_minutes": 0.0,
      "is_stale": false
    }
  },
  "sensor_b": {
    "health": { /* misma estructura */ }
  }
}
```

### Umbrales de Alerta
- `pending_ft_to_evaluate > 0` por más de 120 min → Investigar
- `eval_lag_minutes > 120` → Job de evaluación puede estar fallando

---

## Histórico de Experimentos

### FASE 1 (v1.1.0) - NO-GO
- 17 features (14 base + 3 competitividad)
- sample_weight: draws=1.5, otros=1.0
- Draw predictions: 16.1% pero degradó Brier/LogLoss
- Resultado: Sweep de pesos no encontró punto aceptable

### FASE 2 (v1.1.0-twostage) - EN EVALUACIÓN
- Arquitectura two-stage descrita arriba
- Shadow mode activo, acumulando samples
- Decisión pendiente tras >= 200 evaluaciones
