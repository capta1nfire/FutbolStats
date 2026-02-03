# Plan: Retrain v1.0.1-league-only con Training Set League-Only

**Fecha**: 2026-02-02
**Estado**: ✅ Entrenado → En evaluación extendida (ATI)
**Autor**: Master (instrucciones ATI)

---

## Objetivo

Agregar filtro `admin_leagues.kind='league'` a `training_matches_query` para evitar training-serving skew (ATI Opción A aprobada).

---

## Resultado del Entrenamiento (2026-02-02)

```
model_version: v1.0.1-league-only
samples_trained: 32,486
brier_score (CV): 0.2080
cv_scores: [0.2099, 0.2064, 0.2078]
n_features: 14
league_only: True
date_range: 2020-01-01 → 2026-01-14
model_path: models/xgb_v1.0.1-league-only_20260202.json
```

---

## PIT Preliminar (2026-02-02, N=462)

| Métrica | v1.0.1-league-only-trained | v1.0.0-control | Delta |
|---------|---------------------------|----------------|-------|
| Skill vs Market | -5.23% | -9.55% | **+4.32pp** |
| Brier (model) | 0.6169 | 0.6483 | **-0.031** |
| Brier (market) | 0.5862 | 0.5918 | -0.006 |
| ROI | 1.83% | 2.21% | -0.38pp |
| ROI CI95 | [-16%, +18%] | [-17%, +22%] | ~igual |
| N bets | 402 | 376 | +26 |
| N pit_safe | 462 | 421 | +41 |

**Nota ATI**: No comparar brier_score(CV)=0.208 con brier_model(PIT)=0.6169 - son escalas distintas.

---

## Instrucciones ATI: Evaluación Extendida

**Veredicto preliminar**: OK para continuar como candidato, NO activar en producción aún.

### Proceso
1. **Artefacto candidato**: `models/xgb_v1.0.1-league-only_20260202.json`
2. **Repetir evaluación semanalmente** hasta:
   - ≥2,000 predicciones PIT-safe, O
   - 2-4 semanas de datos
3. **Métricas a reportar**:
   - Brier, logloss, ECE, skill_vs_market (primarias)
   - ROI solo como señal débil (CI ancho con N bajo)

### Criterio GO para Producción
- Mejora sostenida en Brier/skill_vs_market
- Sin degradación en ligas top (estratificado)

### Próxima Evaluación
- **Fecha**: ~2026-02-09
- **Comando**:
  ```bash
  python scripts/evaluate_pit_v3.py \
    --min-snapshot-date 2026-01-15 \
    --model-version v1.0.1-league-only-trained \
    --source experiments
  ```

---

## Cambios Requeridos

### Archivo: `scripts/train_league_only_optimized.py`

**Cambio 1**: `league_matches_query` (fuente para rolling averages)
**Cambio 2**: `training_matches_query` (targets)

**Diff aplicado en ambas queries**:
```diff
           AND al.kind = 'league'
+          AND m.league_id NOT IN (242, 250, 252, 268, 270, 299, 344)  -- Excluir ligas sin stats API
           AND m.date >= :min_date
```

---

## Ligas Excluidas del Training (pilot STOP/NO_DATA)

| league_id | Liga | País | Razón |
|-----------|------|------|-------|
| 242 | Ecuador Liga Pro | Ecuador | 55.9% missing stats |
| 250 | Paraguay Apertura | Paraguay | 100% missing (pilot STOP) |
| 252 | Paraguay Clausura | Paraguay | 100% missing (pilot STOP) |
| 268 | Uruguay Apertura | Uruguay | 65.9% missing stats |
| 270 | Uruguay Clausura | Uruguay | 100% missing (pilot STOP) |
| 299 | Venezuela Primera | Venezuela | 100% missing (pilot STOP) |
| 344 | Bolivia Primera | Bolivia | 68.7% missing stats |

**Nota**: Perú (281) NO está excluida porque tuvo pilot GO (94% coverage).

---

## Pasos de Ejecución

### 1. Aplicar cambio al script
Editar `scripts/train_league_only_optimized.py` líneas 144-151.

### 2. Ejecutar reentrenamiento
```bash
# Usar DATABASE_URL ya configurada en Railway (no pegar secretos aquí)
python scripts/train_league_only_optimized.py \
  --cutoff 2026-01-15 \
  --min-date 2020-01-01
```

### 3. Verificar output del entrenamiento
Confirmar en log:
- `league_only: True`
- `date_range` empieza ~2020 y termina antes del cutoff
- `samples_trained` razonable (~25,000-30,000, no caída drástica)

### 4. Ejecutar PIT/eval comparativo
```bash
# Usar DATABASE_URL ya configurada
python scripts/evaluate_pit_v3.py \
  --min-snapshot-date 2026-01-15 \
  --model-version v1.0.1-league-only
```
Comparar vs modelo actual: Brier/logloss/ECE + accuracy.

---

## Criterio GO/NO-GO (ATI)

| Criterio | GO | NO-GO |
|----------|-----|-------|
| Brier Score | Mejora o igual | Degradación significativa |
| ECE (calibración) | Mejora o igual | Se dispara |
| Top ligas (estratificado) | Sin degradación | Degradación fuerte |

---

## Verificación

- [x] Diff del cambio para revisión ATI
- [x] Output completo del entrenamiento (32,486 samples, CV=0.208)
- [x] Resultados PIT comparativo preliminar (N=462, skill +4.32pp)
- [ ] Evaluación extendida (N≥2,000)
- [ ] Veredicto GO/NO-GO final

---

## Contexto: Backfill Completado

El backfill de stats históricas se completó exitosamente:

| Grupo | Ligas | Coverage |
|-------|-------|----------|
| Top 5 Europeas | 5 | 99.84% |
| Secundarias EU | 3 | 99.57% |
| LATAM Principal | 5 | 97.12% |
| **Total** | ~45,000 partidos | **98.39%** |

Esto habilita un retrain con `--min-date 2020-01-01` sin contaminar features por stats faltantes.
