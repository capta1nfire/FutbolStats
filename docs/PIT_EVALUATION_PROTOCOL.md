# PIT Evaluation Protocol - Bon Jogo

**Version**: 2.0
**Date**: 2026-01-09
**Status**: Active (v2 - clarified)

---

## 1. Definiciones

### 1.1 Captura PIT Válida

Una captura PIT es **válida** si cumple TODOS estos criterios:

| Criterio | Requisito | Verificación |
|----------|-----------|--------------|
| Lineup confirmado | `matches.lineup_confirmed = TRUE` | Flag set by lineup job |
| Odds capturadas | `odds_snapshots.snapshot_type = 'lineup_confirmed'` | Snapshot exists |
| Timing (explícito) | Definir \(\Delta_{KO} = (kickoff\_time - snapshot\_at)\) en minutos. **Válida** si \(10 \le \Delta_{KO} \le 90\). | `odds_snapshots.delta_to_kickoff_seconds` > 0 y en rango |
| Odds válidas | `odds_home > 1.0 AND odds_draw > 1.0 AND odds_away > 1.0` | Non-null, valid range |
| Resultado conocido | `matches.status = 'FT'` | Para evaluación |

**Ventanas de análisis (dos niveles):**
- **Principal (validez)**: \([10, 90]\) min pre-kickoff
- **Ideal (calidad)**: \([45, 75]\) min pre-kickoff

**Nota importante (no mezclar conceptos):**
- La ventana “+5 min” **NO** pertenece a “captura PIT baseline”. Lo post-lineup se mide con `lineup_movement_snapshots` (L+5/L+10) y es otro análisis.

### 1.2 Exclusiones

- Partidos cancelados/pospuestos (status != 'FT')
- Odds capturadas post-kickoff (\(\Delta_{KO} \le 0\))
- Odds “muy tempranas” (>\!90 min) para PIT baseline (se pueden guardar, pero **no** cuentan como PIT válido principal)
- Odds de bookmakers secundarios sin liquidez

---

## 2. Métricas (y decisión)

### 2.1 Métrica primaria para monetización: ROI / EV (con intervalos)

La monetización depende de **rentabilidad**, no solo de calibración. Por eso:
- **Primaria (GO/NO-GO)**: ROI (y EV) con **IC95%**
- **Secundaria**: Brier (calibración) para diagnóstico

#### ROI (Return on Investment)

```
ROI = (Total_Returns - Total_Staked) / Total_Staked
```

Simulación recomendada (para comparabilidad):
- **Stake (principal)**: Flat 1 unit por apuesta
- (Opcional, exploratorio) **Kelly fraccionado**: 0.25 Kelly

**Intervalos (IC95%)**:
- Usar bootstrap sobre apuestas (idealmente block bootstrap por fecha).
- **Criterio formal**: \(IC95\%_{ROI,lower} > 0\).

#### Expected Value (EV)

```
EV = mean(p_model * odds - 1) over placed bets
```

Para apuestas donde `p_model > 1/odds + margin`:

- **Margin threshold**: 5% (solo apostar si edge > 5%)
- **Target EV (formal)**: \(IC95\%_{EV,lower} > 0\)

### 2.2 Métrica secundaria: Brier Score (calibración)

```
Brier = mean(sum((p_predicted - p_actual)²))
```

- **Rango**: [0, 2] para 3 clases
- **Baseline uniform**: 0.6667
- **Target orientativo**: < 0.62 (≈ >7% skill)

**Importante**: Brier bueno **no implica** ROI>0. Se usa para diagnóstico.

### 2.3 Reglas de decisión por fase

| Fase | Qué se permite concluir | Criterio mínimo recomendado |
|------|--------------------------|-----------------------------|
| Piloto (N≥50) | **Sanity check operativo** (pipeline, logs, captura) | No tomar decisiones de negocio |
| Preliminar (N≥200) | Señal **indicativa** | Reportar ROI/EV/Brier con IC95% (puede cruzar 0) |
| Formal (N≥500) | Decisión monetizable | \(IC95\%_{ROI,lower} > 0\) y sin alertas críticas |

---

## 3. Umbrales y Guardrails

### 3.1 Mínimos para Evaluación

| Fase | N mínimo | Propósito |
|------|----------|-----------|
| Piloto | N ≥ 50 | Sanity check |
| Preliminar | N ≥ 200 | Señal estadística (ruidosa) |
| Formal | N ≥ 500 | Decisión monetizable con IC |
| Producción | N ≥ 1000 | Confianza alta |

### 3.2 Criterios de Alarma (Stop)

| Señal | Umbral | Acción |
|-------|--------|--------|
| Brier > baseline | > 0.70 sostenido | Pausar y auditar |
| ROI claramente negativo | \(IC95\\%_{ROI,upper} < 0\\) y N≥200 | Pausar monetización / revisar |
| Drawdown | > 30% del bankroll | Reducir stake |
| Drift detectado | p-value < 0.05 en test | Reentrenar |

### 3.3 Criterios de Éxito

| Nivel | Brier Skill | ROI | Acción |
|-------|-------------|-----|--------|
| Marginal | 5-8% | 0-3% | Continuar observando |
| Positivo | 8-12% | 3-7% | Escalar gradualmente |
| Fuerte | >12% | >7% | Full deployment |

---

## 4. Protocolo de Evaluación

### 4.1 Evaluación Continua (Daily)

**Nota de implementación**: hoy la evaluación PIT debe basarse en `odds_snapshots` + resultados en `matches` (y/o predicciones persistidas). Una tabla `pit_evaluations` puede ser una mejora futura, pero no es requisito del protocolo.

```sql
-- Conteo de PIT válidos (principal window) + live odds
SELECT COUNT(*) AS n_pit_valid
FROM odds_snapshots os
JOIN matches m ON m.id = os.match_id
WHERE os.snapshot_type = 'lineup_confirmed'
  AND os.odds_freshness = 'live'
  AND os.delta_to_kickoff_seconds BETWEEN 600 AND 5400  -- 10..90 min
  AND m.status = 'FT';
```

### 4.2 Evaluación Semanal

1. Calcular métricas acumuladas
2. Comparar vs baselines (uniform, freq, closing odds implied)
3. Test de calibración (reliability diagram)
4. Detectar drift vs período anterior

### 4.3 Evaluación Mensual

1. Backtest rolling (train en mes N-1, test en mes N)
2. Feature importance analysis
3. Decisión: mantener/ajustar/pausar

---

## 5. Implementación

### 5.1 Tablas Requeridas

```sql
-- pit_evaluations (para tracking)
CREATE TABLE IF NOT EXISTS pit_evaluations (
    id SERIAL PRIMARY KEY,
    match_id INTEGER REFERENCES matches(id),
    snapshot_id INTEGER REFERENCES odds_snapshots(id),
    eval_date DATE,

    -- Predicciones del modelo
    p_home REAL,
    p_draw REAL,
    p_away REAL,

    -- Odds capturadas (PIT)
    odds_home REAL,
    odds_draw REAL,
    odds_away REAL,

    -- Resultado real
    result INTEGER, -- 0=H, 1=D, 2=A

    -- Métricas
    brier_score REAL,
    log_loss REAL,

    -- Betting simulation
    bet_placed BOOLEAN,
    bet_outcome INTEGER, -- 0=H, 1=D, 2=A, NULL=no bet
    bet_odds REAL,
    bet_ev REAL,
    bet_won BOOLEAN,

    created_at TIMESTAMP DEFAULT NOW()
);
```

### 5.2 Script de Evaluación

```bash
# Evaluar PIT acumulados
python3 scripts/evaluate_pit.py --min-n 50 --output logs/pit_eval_YYYYMMDD.json
```

---

## 6. Estado Actual

| Métrica | Valor | Status |
|---------|-------|--------|
| PIT válidos totales | 14 | Acumulando |
| PIT con resultado FT | 14 | Evaluables |
| Fecha inicio captura | 2026-01-07 | Reciente |
| Ligas activas | 39,140,135,78,61,71,262,128,11,13 | Pack1+CONMEBOL |

**Próximo milestone**: N ≥ 50 para evaluación piloto.

---

## 7. Changelog

| Fecha | Versión | Cambio |
|-------|---------|--------|
| 2026-01-08 | 1.0 | Protocolo inicial |
| 2026-01-09 | 2.0 | Timing explícito (ΔKO), ROI/EV primaria con IC95%, N=50 solo sanity operativo |

