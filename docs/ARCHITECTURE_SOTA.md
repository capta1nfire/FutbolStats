# Bon Jogo — Arquitectura SOTA (Source of Truth)

Documento técnico rector para evolucionar Bon Jogo desde el baseline XGBoost hacia un sistema SOTA orientado a **Alpha** (ventaja estadística contra el mercado), manteniendo **causalidad temporal** y operabilidad en producción.

## TL;DR
- **Data Fuel**: Feature engineering 2ª generación con *point-in-time correctness* + multi-fuente (API-Football spine + Understat xG/xPTS + Sofascore XI ratings + Open-Meteo).
- **Engine**: Stacking ensemble 2 niveles: (XGBoost + LightGBM + NN tabular) → meta-learner (Ridge/LogReg multinomial + calibración).
- **Safety**: Validación estricta temporal (TimeSeriesSplit/Sliding Window + OOF manual). Prohibido KFold aleatorio.
- **Roadmap**: 3 fases (ETL/Features → modularización ML → stacking/calibración + shadow).

---

## 0) Estado actual (baseline) y constraints que NO debemos romper

### 0.1 Pipeline actual (resumen)
- **ETL**: arquitectura extensible vía `app/etl/base.py::DataProvider` y orquestación `app/etl/pipeline.py::ETLPipeline` con upserts e historial de odds.
- **Features**: `app/features/engineering.py::FeatureEngineer` calcula rolling averages con decaimiento exponencial \(w = e^{-\lambda \Delta t}\) y usa únicamente partidos anteriores a `match.date`.
- **Modelo**: `app/ml/engine.py` implementa baseline XGBoost y arquitectura two-stage (draw vs no-draw + home vs away) con `TimeSeriesSplit`.
- **DB**: `app/models.py` ya incluye `matches`, `odds_history`, `predictions`, `prediction_outcomes`, `post_match_audits` (incluye algunos campos xG en outcomes).

### 0.2 Principios no negociables
1. **Point-in-time**: toda feature usada para un partido debe existir *as_of* kickoff (o un snapshot predefinido pre-kickoff). Nada post-partido.
2. **Idempotencia**: ETL y backfills deben ser re-ejecutables (upserts + unique constraints).
3. **Degradación controlada**: si falla Understat/Sofascore/Open-Meteo, el sistema debe imputar + marcar flags, nunca “crashear”.
4. **Observabilidad**: cada fuente y cada grupo de features debe reportar cobertura, missingness y staleness.

---

## 1) ESTRATEGIA DE DATOS (The Fuel): Features de 2ª Generación

### 1.1 Brechas vs SOTA
Hoy el sistema se apoya en promedios ponderados (goles/tiros/corners) + descanso + features de competitividad y, en two-stage, odds implícitas. Esto es buen “baseline”, pero faltan señales SOTA de:
- **Proceso vs resultado** (xG/xPTS), para modelar regresión a la media y “justicia”.
- **Calidad granular de XI** (ratings/forma) y persistencia de alineaciones.
- **Contexto ambiental y bio-adaptabilidad** (clima, horario, adaptación).
- **Market microstructure** (movimientos de odds, no solo snapshot).
- **Normalización por liga/temporada** y drift.

### 1.2 Arquitectura de ingesta multi-fuente (spine + enrichers)

#### Spine: `matches` (API-Football)
`matches.external_id` (API-Football) permanece como **ID canónico operativo** para scheduler y UI.

#### Enrichers: Understat / Sofascore / Open-Meteo
Se integran como “enriquecimientos” vinculados a `match_id` canónico mediante una capa de referenciación y matching auditable.

**Nueva tabla: `match_external_refs`**
- Clave: `(match_id, source)`
- Campos:
  - `source`: `'api_football' | 'understat' | 'sofascore'`
  - `source_match_id`: string/int (según proveedor)
  - `confidence`: float [0,1]
  - `matched_by`: string (heurística)
  - `created_at`

**Matching determinista (auditable)**
Score \(S\) basado en:
- kickoff UTC (tolerancia ±2h)
- liga/temporada (si aplica)
- nombres normalizados de equipos (con alias)
- venue/city (si existe)

Regla:
- si \(S \ge 0.90\): auto-link
- si \(0.75 \le S < 0.90\): link con “needs_review”
- si \(S < 0.75\): no link

> Nota: este layer es crítico para no contaminar datos con matchs mal linkeados (leakage “semántico”).

### 1.3 Modelo de datos mínimo (tablas nuevas)

#### Understat (xG / xPTS)
**Tabla**: `match_understat_team`
- `match_id` (FK)
- `xg_home`, `xg_away`
- `xpts_home`, `xpts_away`
- opcional: `npxg_home/away`, `xga_home/away` si se dispone
- `captured_at`, `source_version`

#### Sofascore (XI, ratings, formación)
**Tabla**: `match_sofascore_player`
- `match_id` (FK)
- `team_side`: `'home'|'away'`
- `player_id_ext`
- `position` (GK/DEF/MID/FWD + sub-roles si hay)
- `is_starter` bool
- `rating_pre_match` (si disponible) o `rating_recent_form` (ventana)
- `minutes_expected` (si disponible)
- `captured_at`

**Tabla** (opcional, recomendado): `match_sofascore_lineup`
- `match_id`, `team_side`, `formation`, `captured_at`

#### Open-Meteo (clima)
**Tabla**: `match_weather`
- `match_id` (FK)
- `temp_c`, `humidity`, `wind_ms`, `precip_mm`, `pressure_hpa` (si existe), `cloudcover`
- `is_daylight` (derivado de sunrise/sunset)
- `forecast_horizon_hours` (ej: 24h, 1h)
- `captured_at`

#### Geo / perfiles de clima base (para bio-adaptabilidad)
**Tabla**: `venue_geo`
- `venue_city`, `country`, `lat`, `lon`, `source`, `confidence`

**Tabla**: `team_home_city_profile`
- `team_id` (FK)
- `home_city`, `timezone`
- `climate_normals_by_month` (JSON) con medias (temp/humidity) históricas

### 1.4 Lógica de Jugadores (XI) → vector granular (WeightedXI)

Definiciones:
- XI local: jugadores \(i=1..11\) con rating \(r_i\) (Sofascore).
- Transformación robusta (si ratings varían por liga/temporada):
  - \(z_i = \text{clip}((r_i-\mu)/\sigma, -3, 3)\)
- Pesos por posición:
  - \(w_{GK}=1.0,\; w_{DEF}=0.9,\; w_{MID}=1.0,\; w_{FWD}=1.1\) (tuneable)

Features:
- `xi_weighted_home = Σ_i w_pos(i) * z_i`
- `xi_weighted_away = Σ_j w_pos(j) * z_j`
- `xi_weighted_diff = xi_weighted_home - xi_weighted_away`

Features de distribución:
- `xi_p10_*`, `xi_p50_*`, `xi_p90_*`
- `xi_weaklink_* = min(z_i)` (captura “tail risk”)
- `xi_std_*` (cohesión/heterogeneidad del XI)

Imputación:
- Si no hay XI confirmado: usar estimación por “squad baseline” (últimos K partidos) + flag `xi_missing=1`.

### 1.5 Feature “Justicia” (Delta real vs esperado) + Regresión a la media

Objetivo: capturar divergencia entre **proceso** (xG) y **resultado** (goles), que tiende a revertir.

Por equipo \(T\) en ventana \(\mathcal{W}\) (últimos N partidos):
- \(G_T = \sum_{m \in \mathcal{W}} goals_T(m)\)
- \(XG_T = \sum_{m \in \mathcal{W}} xg_T(m)\)

Z-score aproximado:
- \(\text{justice}_T = \frac{G_T - XG_T}{\sqrt{XG_T + \epsilon}}\)

Shrinkage por tamaño muestral (evita sobre-reaccionar con pocos partidos):
- \(\rho = \frac{n}{n+k}\) con \(k \approx 10\) (tuneable)
- \(\text{justice\_shrunk}_T = \rho \cdot \text{justice}_T\)

Para el partido:
- `justice_diff = justice_shrunk_home - justice_shrunk_away`

Interpretación:
- `justice_diff < 0`: home subconvirtió relativo a away → potencial “rebote” (mejora futura).
- `justice_diff > 0`: home sobreconvirtió → riesgo de reversión.

### 1.6 Features Ambientales & Contextuales (Open-Meteo) — bio-adaptabilidad

#### Thermal_Shock (aclimatación)
Sea:
- \(T_{stad}\): temperatura prevista en estadio a kickoff.
- \(\bar T_{away,month}\): media histórica de temperatura del “home city” del visitante en ese mes.

Feature:
- `thermal_shock = T_stad - T_away_month_mean`
- `thermal_shock_abs = abs(thermal_shock)`

Notas:
- modelar no-linealidad con bins/splines (0–5, 5–10, 10+ °C).

#### Circadian_Disruption (horario vs costumbre)
Sea:
- \(h_{match}\): hora local del partido (visitante) en [0,24).
- \(\mu_{away}\): hora típica de kickoff del visitante (media circular) últimos K partidos.

Distancia circular:
- \(d = \min(|h_{match}-\mu|, 24-|h_{match}-\mu|)\)

Feature:
- `circadian_disruption = d / 12` (normaliza a [0,1])

Extensión (viaje y zona horaria):
- `tz_shift = abs(tz_match - tz_away_base)`
- `bio_disruption = a*circadian_disruption + b*min(tz_shift,6)/6`

### 1.7 Carta blanca: señales exógenas con potencial Alpha

#### Market microstructure (odds_history)
Ya existe `odds_history`. Se agregan features de movimiento:
- `odds_log_move_open_to_close = log(odds_close) - log(odds_open)`
- `odds_log_move_lineup_to_close` (si hay snapshot al confirmar lineup)
- `overround_*` (si se calcula/almacena por snapshot)
- `steam_move_flag` (umbral + confirmación multi-book, si existe)

#### Congestión y fatiga avanzada
- `matches_last_7d`, `matches_last_14d`, `minutes_last_14d` (si minutos disponibles)
- `travel_km` (Haversine entre ciudades) + interacción con descanso

#### Importancia del partido (proxy)
- “pressure” por standings:
  - distancia a puestos europeos/descenso
  - cercanía a fin de temporada

---

## 2) ARQUITECTURA DEL NÚCLEO (The Engine): Stacking Ensemble Robusto

### 2.1 Objetivo de optimización
No buscamos solo accuracy; buscamos:
- **Brier / LogLoss** (probabilístico)
- **Calibración** (para EV)
- **Estabilidad** (bajo drift y missingness)

### 2.2 Nivel 0 — Base Learners
1. **XGBoost (GBDT)**: fuerte baseline tabular; robusto a missing; captura interacciones.
2. **LightGBM (leaf-wise)**: sesgo inductivo diferente; a veces superior en generalización.
3. **Red neuronal tabular** (MLP + embeddings / FT-Transformer):
   - útil para interacciones suaves y señales densas (XI distributions, clima, market movement).

Justificación (reducción de varianza):
- Si los errores de los modelos son parcialmente no correlacionados, un ensamblado aprende a “promediar” y reduce la varianza efectiva.

### 2.3 Nivel 1 — Meta-Learner
Meta recomendado:
- **Logistic Regression multinomial (L2)** o **Ridge + softmax** sobre logits.

Entrada al meta:
- concatenación de probabilidades OOF o logits OOF:
  - \( \ell_k = \log \frac{p_k}{p_{ref}} \)

Calibración:
- calibración global (y opcional por liga/segmento) mediante:
  - Dirichlet calibration multinomial, o
  - calibración post-meta (sigmoid/isotonic por clase con cuidado temporal).

### 2.4 Por qué stacking optimiza mejor la pérdida global
- Cada base learner minimiza \(\mathcal{L}\) dentro de su familia.
- El meta aprende a corregir sesgos sistemáticos (p.ej., overconfidence del GBDT en ciertos regímenes) usando predicciones OOF que aproximan la generalización real.

---

## 3) PROTOCOLO DE VALIDACIÓN ANTI-LEAKAGE (Critical Safety)

### 3.1 Regla de oro (prohibición)
- **PROHIBIDO**: KFold aleatorio estándar, shuffle splits, holdouts aleatorios.
- **OBLIGATORIO**: `TimeSeriesSplit` o Sliding Window respetando orden temporal por `match.date`.

### 3.2 OOF manual para entrenar el meta-learner (requisito)
El meta **solo** se entrena con predicciones generadas por modelos que no vieron esos ejemplos.

Plantilla oficial:
```python
df = df.sort_values("date").reset_index(drop=True)
tscv = TimeSeriesSplit(n_splits=N_SPLITS)

oof = {name: np.zeros((len(df), 3)) for name in base_learners}

for train_idx, val_idx in tscv.split(df):
    train = df.iloc[train_idx]
    val = df.iloc[val_idx]

    for name, model in base_learners.items():
        model.fit(train[X], train[y])
        oof[name][val_idx] = model.predict_proba(val[X])

meta_X = np.concatenate([oof[n] for n in base_learners], axis=1)
meta.fit(meta_X, df[y])

for name, model in base_learners.items():
    model.fit(df[X], df[y])  # modelos finales para producción
```

### 3.3 Gap y “as_of” enforcement
Si hay fuentes con latencia (xG publicado tarde, ratings retroactivos), imponer:
- **gap temporal** (ej: 1–7 días) entre train y val, o
- usar únicamente snapshots capturados pre-kickoff y almacenar `captured_at`.

### 3.4 Segmentación de evaluación (obligatoria para Alpha)
Reportar métricas por:
- liga/competición
- favoritos vs underdogs (por odds)
- matches con/ sin lineup confirmado
- buckets de thermal_shock y bio_disruption

---

## 4) HOJA DE RUTA DE IMPLEMENTACIÓN (Step-by-Step)

### Fase 1 — ETL + nuevas features (Understat / Sofascore / Open-Meteo)
1. Implementar providers (o servicios) por fuente:
   - `UnderstatProvider` (xG/xPTS)
   - `SofascoreProvider` (XI, ratings, formations)
   - `OpenMeteoProvider` (forecast por venue_geo)
2. Crear tablas nuevas y migraciones.
3. Agregar scheduler jobs:
   - backfill FT (Understat)
   - captura pre-kickoff (Sofascore/Open-Meteo)
4. Extender feature building por “feature groups” + flags de missingness.
5. Añadir cobertura y staleness metrics (observabilidad).

### Fase 2 — Refactor de entrenamiento (modularización)
1. Separar responsabilidades:
   - dataset + point-in-time enforcement
   - modelos base (xgb/lgbm/nn) con interfaz común
   - stacking + calibración
2. Unificar persistencia de artefactos (DB snapshots como SoT) con:
   - versión de features (schema)
   - cutoff de entrenamiento
   - métricas y segmentos
3. Tests anti-leakage y “time travel”.

### Fase 3 — Stacking + calibración + shadow rollout
1. Entrenar base learners con el mismo protocolo temporal.
2. Generar OOF y entrenar meta.
3. Calibración (y evaluación por segmentos).
4. Shadow mode (paralelo) y gating GO/NO-GO.

---

## Criterios GO/NO-GO (SOTA)
- **GO**:
  - Brier mejora globalmente y no degrada por segmentos críticos (odds buckets).
  - Calibración mejora (reliability, ECE/MCE) y disminuye overconfidence.
  - Lift reproducible en ablations por feature group (Understat / XI / Weather / Market).
  - Robustez: >95% cobertura features críticas, con degradación controlada.
- **NO-GO**:
  - Evidencia de leakage (cualquier mejora “demasiado buena” fuera de tiempo).
  - Degradación sistemática en underdogs o draws (regímenes clave para EV).
  - Missingness o staleness frecuentes sin fallback.

