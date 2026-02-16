# League Router — Arquitectura de Prediccion Asimetrica

> **Version**: v1.0
> **Fecha**: 2026-02-15
> **Origen**: GDT Mandato M3 (Mega-Pool Analysis)
> **Estado**: INFRAESTRUCTURA DESPLEGADA. Family S model pendiente.

---

## 1. Contexto y Hallazgos

### 1.1 Feature Lab V4: MTV Backtest Historico

Se materializo `talent_delta` para 33,598 matches (2023+) usando las funciones EXACTAS de
produccion (`compute_match_talent_delta_features`). Coverage: 89.7% con talent_delta, 0 errores.

**Parametros PIT-safe**: `xi_window=15` (prediccion de XI), `pts_limit=10` (PTS rolling).
Ambos valores son defaults de produccion.

### 1.2 Mega-Pool V2: Super-Tensor

18 ligas perifericas (excluido Big 5 europeo), 3,771 matches OOT. Tres pares de test:

| Par | Features Control | Features MTV | Requiere Odds |
|-----|-----------------|-------------|---------------|
| A: Elo | elo_home/away/diff | + 4 MTV features | No |
| B: Elo+Odds | Elo + odds_home/draw/away | + 4 MTV features | Si |
| C: Full | Elo + Defense + Form + Odds | + 4 MTV features | Si |

**Resultados pooled (18 ligas, V2 corregido)**:

| Par | N_test | Brier Ctrl | Brier MTV | Delta(MTV-Ctrl) | Significancia |
|-----|--------|-----------|-----------|-----------------|---------------|
| A: Elo | 3,771 | 0.62372 | 0.62098 | **-0.00274** | n.s. |
| B: Elo+Odds | 3,422 | 0.61509 | 0.61371 | **-0.00138** | n.s. |
| C: Full | 3,422 | 0.61339 | 0.61204 | **-0.00135** | n.s. |

MTV mejora en los tres pares pero no alcanza significancia en el pool completo.

### 1.3 Bug V1 → V2: Survivorship Bias en Odds

**V1** usaba `df["odds_home"].notna().all()` que requeria 100% coverage de odds, excluyendo
15 ligas con 78-100% coverage parcial. Pair B paso de 3 ligas (N=802) a 17 ligas (N=3,422).
**Conclusion invertida**: V1 decia MTV HURTS con odds; V2 dice MTV HELPS.

### 1.4 Mandato Quant Final: Tier 3 Purificado

Filtrado a las 10 ligas donde MTV ayuda en Pair B (10/10 consistencia):

| Liga | ID | N | Delta(MTV-Ctrl) |
|------|----|---|-----------------|
| Chile | 265 | 146 | **-0.01719** (mejor) |
| Primeira Liga | 94 | 196 | -0.01085 |
| Uruguay | 268 | 175 | -0.01074 |
| Eredivisie | 88 | 204 | -0.00987 |
| Super Lig | 203 | 214 | -0.00624 |
| Peru | 281 | 195 | -0.00496 |
| Venezuela | 299 | 139 | -0.00306 |
| Bolivia | 344 | 164 | -0.00178 |
| Mexico | 262 | 212 | -0.00160 |
| Ecuador | 242 | 157 | -0.00094 |

**Pooled Tier 3** (N=1,802, 10 ligas):
- Delta(MTV-Ctrl) = **-0.00668 [-0.01116, -0.00214] *** SIGNIFICATIVO**
- 10/10 ligas MTV HELPS (100% consistencia)

### 1.5 Shock Sweep (Pair B, solo Tier 3)

| Percentil | Threshold | N | Brier Ctrl | Brier MTV | Delta | Sig |
|-----------|----------|---|-----------|-----------|-------|-----|
| P80 | 0.1133 | 361 | 0.60555 | 0.59288 | **-0.01267** | *** |
| P85 | 0.1279 | 271 | 0.60086 | 0.58593 | -0.01493 | n.s. |
| P90 | 0.1469 | 181 | 0.58625 | 0.57466 | -0.01159 | n.s. |
| P95 | 0.1844 | 91 | 0.59354 | 0.58135 | -0.01219 | n.s. |

**vs Mercado en zonas de shock**: El mercado mejora mas que el modelo a mayor shock
(P95: Brier_Mkt=0.515 vs Brier_MTV=0.560). El mercado anticipa shocks mejor, pero MTV
cierra parte del gap vs control.

---

## 2. Clasificacion de Tiers

### 2.1 Tier 1 — Big 5 Europeo (ODDS_CENTRIC)

Mercados mas eficientes. El modelo pierde vs mercado en todas. Estrategia: baseline model
con market anchor de alpha alto.

| Liga | ID |
|------|----|
| Premier League | 39 |
| Ligue 1 | 61 |
| Bundesliga | 78 |
| Serie A | 135 |
| La Liga | 140 |

### 2.2 Tier 2 — Default (BASELINE)

Ligas perifericas donde MTV fue neutral o negativo. Incluye ligas con buenos datos
pero sin señal MTV. Estrategia: baseline model, market anchor moderado.

Ligas explicitas en Tier 2 por resultado: Championship (40), Brazil (71), Argentina (128),
Belgium (144), Colombia (239), MLS (253), Saudi (307), Paraguay (250/252).

**Nota**: Cualquier liga no clasificada en T1 o T3 cae a T2 por defecto.

### 2.3 Tier 3 — MTV Winners (FAMILY_S)

10 ligas donde MTV mejora estadisticamente las predicciones. Estrategia: modelo Family S
con features MTV inyectados.

| Liga | ID | Delta Pair B |
|------|----|-------------|
| Chile | 265 | -0.01719 |
| Primeira Liga | 94 | -0.01085 |
| Uruguay | 268 | -0.01074 |
| Eredivisie | 88 | -0.00987 |
| Super Lig | 203 | -0.00624 |
| Peru | 281 | -0.00496 |
| Venezuela | 299 | -0.00306 |
| Bolivia | 344 | -0.00178 |
| Mexico | 262 | -0.00160 |
| Ecuador | 242 | -0.00094 |

---

## 3. Arquitectura del Router

### 3.1 Modulo Principal

**Archivo**: `app/ml/league_router.py`

```
get_league_tier(league_id) → 1 | 2 | 3
should_inject_mtv(league_id, mtv_enabled) → bool
should_compute_talent_delta(league_id) → True  (siempre, para SteamChaser)
get_prediction_strategy(league_id, mtv_enabled) → dict
get_tier_summary() → dict  (para ops/diagnostico)
```

### 3.2 Feature Flags

```env
LEAGUE_ROUTER_ENABLED=true       # Clasificacion + logging (activo)
LEAGUE_ROUTER_MTV_ENABLED=false  # Inyeccion MTV para Tier 3 (requiere Family S)
```

### 3.3 Flow de Prediccion

#### Cascade Handler (LINEUP_CONFIRMED)

```
Match NS + not frozen
    │
    ├── get_prediction_strategy(league_id) → strategy
    │
    ├── IF strategy.inject_mtv (T3 + flag ON):
    │   ├── compute_talent_delta (5s timeout)
    │   ├── Inject features en df_match
    │   └── Predict con Family S model
    │
    ├── ELSE (T1/T2, o T3 con flag OFF):
    │   ├── Predict con baseline model
    │   └── compute_talent_delta DESPUES (SteamChaser log)
    │
    ├── Apply Market Anchor
    └── UPSERT prediction
```

#### Daily Save Predictions (batch)

```
Fetch features → Filter NS → Kill-switch
    │
    ├── Log tier distribution (observabilidad)
    ├── Predict ALL con baseline model (batch no tiene talent_delta live)
    ├── Apply Market Anchor
    └── Save en batches
```

**Nota**: daily_save_predictions NO computa talent_delta en tiempo real (es batch).
La inyeccion MTV solo ocurre en el cascade handler (event-driven, post-lineup).

### 3.4 Archivos Modificados

| Archivo | Cambio |
|---------|--------|
| `app/ml/league_router.py` | **NUEVO** — Tier classification + routing |
| `app/config.py` | +2 settings (LEAGUE_ROUTER_*) |
| `app/events/handlers.py` | Cascade tier-aware (T3 inject vs T1/T2 SteamChaser) |
| `app/scheduler.py` | +Log tier distribution en daily_save_predictions |
| `app/ml/__init__.py` | Exports del router |

---

## 4. Datos y Artefactos

| Archivo | Descripcion | Rows/Size |
|---------|-------------|-----------|
| `data/historical_mtv_features.parquet` | talent_delta para 33,598 matches 2023+ | 33,598 rows |
| `scripts/build_historical_mtv.py` | Script de materializacion MTV | ~250 LOC |
| `scripts/mega_pool_analysis.py` | Mega-Pool + Tier 3 sweep | ~780 LOC |
| `scripts/output/lab/mega_pool_tensor.parquet` | Super-Tensor con predicciones OOT | 3,771 rows |
| `scripts/output/lab/mega_pool_results.json` | Resultados V2 completos | M1+M2+M4 |
| `scripts/output/lab/tier3_sweep_results.json` | Mandato Quant Final | Shock sweep |

---

## 5. Roadmap: Activacion de Family S

### Paso 1: Entrenar modelo Family S

Entrenar XGBoost con FEATURE_COLUMNS + MTV_FEATURES para ligas Tier 3.
Usar `scripts/feature_lab.py --extract --league <T3_IDs>` como base de datos.

**Features Family S** (21 features = 17 baseline + 4 MTV):
```
home_goals_scored_avg, home_goals_conceded_avg, home_shots_avg, home_corners_avg,
home_rest_days, home_matches_played, away_goals_scored_avg, away_goals_conceded_avg,
away_shots_avg, away_corners_avg, away_rest_days, away_matches_played,
goal_diff_avg, rest_diff, abs_attack_diff, abs_defense_diff, abs_strength_gap,
home_talent_delta, away_talent_delta, talent_delta_diff, shock_magnitude
```

### Paso 2: Validar OOT

Correr Feature Lab Section S completa en las 10 ligas Tier 3.
Criterio GO: Brier Family S < Brier Baseline en >=7/10 ligas.

### Paso 3: Deploy modelo

1. Guardar modelo en `models/xgb_family_s_v1.0.0_YYYYMMDD.json`
2. Actualizar `XGBoostEngine` para cargar modelo segun tier (o crear `FamilySEngine`)
3. Flip flag: `LEAGUE_ROUTER_MTV_ENABLED=true` en Railway

### Paso 4: Monitoreo

- Shadow mode primeras 2 semanas (log Family S pero servir baseline)
- Comparar Brier live Family S vs baseline vs mercado
- Si Family S degrada: flip flag a false (rollback instantaneo)

---

## 6. Resoluciones Vinculantes

| ID | Origen | Resolucion |
|----|--------|-----------|
| GDT-M1 | Mega-Pool | Excluir Big 5, poolear perifericas, N_test > 1,000 |
| GDT-M2 | Shock Sweep | Brier condicionado P80-P95 en shock_magnitude |
| GDT-M3 | League Router | Codificar tiers, bifurcar prediccion, SteamChaser |
| GDT-MQF | Quant Final | Filtrar a 10 T3 winners, sweep purificado |
| ABE-P0-1 | Window | xi_window=15, pts_limit=10 (metadata registrada) |
| ABE-P0-2 | Canary-First | EPL canary completado antes de full-run |
| ABE-P1-1 | Injury Eras | Pre-Jul-2025 (blind) vs post (aware) — ambas MTV HELPS |

---

## 7. Nomenclatura V2 (GDT Mandato 4)

### Convencion obligatoria para Phase 4+

| Patron | Ejemplo | Uso |
|--------|---------|-----|
| `vX.Y.Z-descriptor` | `v1.0.1-league-only` | Legacy (grandfathered) |
| `v2.X-tierN-name` | `v2.0-tier1-aegis` | Phase 4+ (obligatorio) |
| `v2.X-tierN-name+ext` | `v2.0-tier3-mtv+microstructure` | Compound |

### Nombres rechazados
- `v2.0` (sin tier/descriptor)
- `shadow`, `twostage` (sueltos, sin version)
- Cualquier v2+ sin prefijo `-tierN-`

### Guardrail
`app/ml/persistence.py:validate_model_version()` — valida en `persist_model_snapshot`.
Para v2+, requiere prefijo `v2.X-tierN-`. Log warning si falla (no abort en serving).

### SSOT de cohorte
`model_snapshots.is_active=true` es la UNICA fuente canonica para resolver el modelo activo.
`performance_metrics.py` y `evaluate_pit_v3.py` lo consultan como default.
Fallback a "prediccion mas reciente" solo si no hay snapshot activo (integrity=degraded).

---

## 8. Limitaciones Conocidas

1. **Family S no entrenado**: El router clasifica pero no bifurca predicciones hasta que
   exista un modelo entrenado con features MTV.
2. **Daily batch sin MTV live**: `daily_save_predictions` no computa talent_delta en
   tiempo real. Solo el cascade handler (post-lineup) puede inyectar MTV.
3. **Injury era pre-2025**: El 90%+ de los datos MTV son injury-blind (player_injuries
   solo desde Jul 2025). El talent_delta puede tener ruido por ausencias no filtradas.
4. **Coverage asimetrica**: Lineups 93%, MPS ratings 72.2%. Ligas con baja cobertura
   (Uruguay ~0.6% MPS) tendran talent_delta poco confiable.
5. **Mercado sigue ganando**: Incluso con MTV, Brier_MTV (0.593) > Brier_Market (0.568)
   en Tier 3. MTV reduce el gap pero no lo cierra.
