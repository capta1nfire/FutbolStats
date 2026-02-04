# TITAN — Tier Comparison Framework (Runlog)

Bitácora de implementación, auditorías, bugs encontrados, y estado actual del experimento de tiers.

> **Fuente de verdad**: los artefactos de cada corrida están en `logs/pit_evaluation_v3_*.json` y el código en `scripts/` + `migrations/`.

## Alcance

- Tabla: `predictions_experiments` (1 predicción por `snapshot_id` y `model_version`)
- Evaluador: `scripts/evaluate_pit_v3.py` con `--source experiments`
- Modelos: baseline experimental (`v1.0.0-exp`) y variantes T1b / T1b_v2 (y el framework para T1c/T1d)

## Implementación (artefactos creados/modificados)

- **Migración**: `migrations/titan_009_predictions_experiments.sql`
- **Scripts nuevos**:
  - `scripts/build_titan_dataset.py`
  - `scripts/train_titan_tier.py`
  - `scripts/generate_tier_preds.py`
  - `scripts/compare_tiers.py`
- **Script modificado**:
  - `scripts/evaluate_pit_v3.py` (agrega soporte `--source experiments`, `--model-version`, `--league-ids`, trazabilidad)

## Hallazgos / Bugs (resumen)

- **P1 — Baseline no “as-of snapshot_at”**: `rest_days`/`rest_diff` cambiaban entre `snapshot_at` y `kickoff`.  
  Mitigación/implementación: `FeatureEngineer.get_match_features_asof(match, asof_dt)` y uso de `snapshot_at` en generación experimental.

- **P1/P0 — Rolling xG y cobertura**:
  - Se detectaron issues de evaluación por cobertura (xG ausente fuera de Top5 → imputación a 0).
  - Se iteró hacia `T1b_v2` y se añadieron missing flags (`xg_home_missing`, `xg_away_missing`).
  - Se recalculó bootstrap con el mismo filtro de timing (10–90 min) para paridad exacta.

> Nota: Parte del experimento quedó etiquetado como **PIT‑RELAXED** para xG histórico (ver estado actual).

## Corridas (artefactos relevantes)

### 2026-01-30 — Segmentación baseline vs T1b (raw) (Top5 / Global / 17 leagues)

- `logs/pit_evaluation_v3_20260130_084159_from_2026-01-07_v1_0_0-exp.json`
- `logs/pit_evaluation_v3_20260130_084212_from_2026-01-07_v1_0_0+T1b.json`
- `logs/pit_evaluation_v3_20260130_084225_from_2026-01-07_filtered_5leagues_v1_0_0-exp.json`
- `logs/pit_evaluation_v3_20260130_084240_from_2026-01-07_filtered_5leagues_v1_0_0+T1b.json`
- `logs/pit_evaluation_v3_20260130_084300_from_2026-01-07_filtered_17leagues_v1_0_0-exp.json`
- `logs/pit_evaluation_v3_20260130_084312_from_2026-01-07_filtered_17leagues_v1_0_0+T1b.json`

### 2026-01-30 — T1b_v2 (diferenciales) + missing flags (Top5 / Global)

- Global:
  - `logs/pit_evaluation_v3_20260130_093922_from_2026-01-07_v1_0_0-exp.json`
  - `logs/pit_evaluation_v3_20260130_111335_from_2026-01-07_v1_0_0+T1b_v2.json`
- Top5:
  - `logs/pit_evaluation_v3_20260130_111349_from_2026-01-07_filtered_5leagues_v1_0_0-exp.json`
  - `logs/pit_evaluation_v3_20260130_110611_from_2026-01-07_filtered_5leagues_v1_0_0+T1b_v2.json`

> Nota: Existen otros artefactos intermedios de la misma fecha (p. ej. `*_092552_*`, `*_092607_*`, `*_093934_*`). Mantenerlos para auditoría; el reporte debe referenciar siempre los “finales” por timestamp.

## Estado actual (decisión)

- **Top5**: **HOLD preliminar positivo** (señal leve, bootstrap CI95 cruza 0 con N≈144).
- **Global**: **NO‑GO** (regime mismatch por cobertura xG fuera de Top5).
- **Gate**: acumular hacia **N≥200** (checkpoint) y **N≥500** (formal).

### Contrato PIT (xG)

- **Baseline / odds / snapshots**: PIT-safe (por snapshot_id + created_at <= snapshot_at).
- **xG (Understat) histórico**: marcado como **PIT‑RELAXED** con proxy operacional `kickoff + 2h < snapshot_at` (cuando se use).

## Tracker (recomendado)

Mantener un JSON (o tabla) que se actualice en cada checkpoint con:

- fecha
- N Top5 acumulado (10–90 min)
- skill_vs_market baseline vs T1b_v2
- bootstrap CI95 (delta Brier) con filtro timing 10–90 min
- estado del gate (HOLD / GO / NO-GO)

