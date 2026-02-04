# TITAN — Tier Comparison Framework (PIT-safe)

Este documento define el **framework operativo** para evaluar si los tiers de TITAN (**T1b/T1c/T1d**) aportan *lift* real vs baseline y vs mercado, **sin leakage** y con trazabilidad reproducible.

## Objetivo

- **Generar predicciones comparables** sobre la **misma cohorte PIT** para:
  - `v1.0.0-exp` (baseline experimental)
  - `v1.0.0+T1b`, `v1.0.0+T1c`, `v1.0.0+T1d`
- **Evaluar** con PIT integrity (created_at <= snapshot_at, features as-of).
- **Decidir** GO/HOLD/NO-GO antes de continuar con fases costosas (p. ej. Entity Resolution).

## Principios / Guardrails (no negociables)

1. **Cohorte por snapshot**: una predicción por `odds_snapshots.id` (snapshot), no por match.
2. **PIT-safe timestamps**:
   - En experiments, `created_at = snapshot_at - interval '1 second'` (o equivalente).
   - Constraint DB garantiza: `created_at <= snapshot_at`.
3. **No leakage por entrenamiento**:
   - Definir `cutoff_train` y `eval_start` tal que **`cutoff_train < eval_start`**.
   - Entrenar únicamente con partidos `FT` con `match.date < cutoff_train`.
   - Evaluar snapshots con `snapshot_at >= eval_start`.
4. **As-of TITAN**:
   - Cualquier feature de `titan.feature_matrix` usada para un snapshot debe cumplir: `pit_max_captured_at <= snapshot_at` (y por diseño, pre-kickoff).
5. **Tabla separada**:
   - `predictions_experiments` no contamina `predictions` productiva.

## Schema (experiments)

Aplicar migración:

```bash
psql "$DATABASE_URL" -f migrations/titan_009_predictions_experiments.sql
```

Puntos clave del schema:

- `snapshot_id` (FK a `odds_snapshots.id`)
- `UNIQUE(snapshot_id, model_version)`
- `snapshot_at TIMESTAMPTZ`, `created_at TIMESTAMPTZ`
- `CHECK (created_at <= snapshot_at)`

## Componentes (scripts)

- **Dataset builder**: `scripts/build_titan_dataset.py`
- **Trainer**: `scripts/train_titan_tier.py`
- **Prediction generator**: `scripts/generate_tier_preds.py`
- **Evaluator**: `scripts/evaluate_pit_v3.py` (con `--source experiments`)
- **Comparator**: `scripts/compare_tiers.py`

## Convención de `model_version`

- Baseline experimental: `v1.0.0-exp`
- Tiers:
  - `v1.0.0+T1b`
  - `v1.0.0+T1c`
  - `v1.0.0+T1d`

## Flujo end-to-end (recomendado)

Definir fechas:

```text
cutoff_train = 2026-01-06
eval_start   = 2026-01-07
REGLA: cutoff_train < eval_start
```

### 1) Build datasets

```bash
python scripts/build_titan_dataset.py --tier baseline --cutoff 2026-01-06
python scripts/build_titan_dataset.py --tier T1b     --cutoff 2026-01-06
python scripts/build_titan_dataset.py --tier T1c     --cutoff 2026-01-06
python scripts/build_titan_dataset.py --tier T1d     --cutoff 2026-01-06
```

### 2) Train models (XGBoost JSON)

```bash
python scripts/train_titan_tier.py --tier baseline --cutoff 2026-01-06
python scripts/train_titan_tier.py --tier T1b     --cutoff 2026-01-06
python scripts/train_titan_tier.py --tier T1c     --cutoff 2026-01-06
python scripts/train_titan_tier.py --tier T1d     --cutoff 2026-01-06
```

### 3) Generate predicciones por snapshot (PIT-safe)

```bash
python scripts/generate_tier_preds.py --tier baseline --since 2026-01-07
python scripts/generate_tier_preds.py --tier T1b     --since 2026-01-07
python scripts/generate_tier_preds.py --tier T1c     --since 2026-01-07
python scripts/generate_tier_preds.py --tier T1d     --since 2026-01-07
```

### 4) Evaluar cada tier (misma cohorte)

```bash
python scripts/evaluate_pit_v3.py --source experiments --min-snapshot-date 2026-01-07 --model-version v1.0.0-exp
python scripts/evaluate_pit_v3.py --source experiments --min-snapshot-date 2026-01-07 --model-version v1.0.0+T1b
python scripts/evaluate_pit_v3.py --source experiments --min-snapshot-date 2026-01-07 --model-version v1.0.0+T1c
python scripts/evaluate_pit_v3.py --source experiments --min-snapshot-date 2026-01-07 --model-version v1.0.0+T1d
```

### 5) Comparar resultados

```bash
python scripts/compare_tiers.py logs/pit_evaluation_v3_*_from_2026-01-07_*.json
```

## Verificación (SQL)

Conteos por versión:

```sql
SELECT model_version, COUNT(*) AS n
FROM predictions_experiments
GROUP BY 1
ORDER BY 2 DESC;
```

PIT safety (debe ser 100%):

```sql
SELECT
  model_version,
  COUNT(*) AS total,
  SUM(CASE WHEN created_at <= snapshot_at THEN 1 ELSE 0 END) AS pit_safe
FROM predictions_experiments
GROUP BY 1;
```

## Criterios de aceptación (framework)

- **Funcional**:
  - Cada `model_version` evaluado produce `brier.n_with_predictions > 0`.
  - `prediction_integrity.source == "predictions_experiments"` y `keyed_by == "snapshot_id"` en los JSON.
- **Científico**:
  - La decisión se toma con `skill_vs_market` y `paired_differential` (CI95) vs mercado.
  - No se avanza a ER / más fuentes sin evidencia de lift.

## Nota operativa (entorno)

Si el entorno crashea al importar XGBoost (`python -c "import xgboost"`), **no ejecutar** training/generation hasta corregir dependencias nativas del runtime.

## Runlog / Evidencia de Corridas

La bitácora de implementación, auditorías, bugs y paths de los JSON de evaluación está en:

- `docs/TITAN_TIER_COMPARISON_RUNLOG.md`

