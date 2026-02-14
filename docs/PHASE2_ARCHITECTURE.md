# Fase 2 — Arquitectura de Asimetria de Informacion y Microestructura

> **Version**: v3 (post-ATI + GDT review)
> **Fecha**: 2026-02-14
> **Estado**: APROBADO SIN RESERVAS (ATI DEFCON 1 / FULL GO)
> **Prerequisito completado**: Feature Lab v3, Market Anchor desplegado 23 ligas

---

## 1. Contexto y Motivacion

### 1.1 Hallazgo de Feature Lab v3

Feature Lab v3 evaluo 110 configuraciones de features + 9 SHAP + 16 Optuna + Section R
across 23 ligas. Hallazgo universal: **RESOLUTION_ISSUE**.

- Los modelos estan **bien calibrados** (ECE aceptable)
- Pero **no discriminan** mejor que el mercado (Brier modelo >= Brier mercado en 19/23 ligas)
- Solo Turquia (203) es INEFICIENTE con alpha confirmado
- Market Anchor (alpha blending con mercado) ya desplegado como mitigacion

### 1.2 Tesis de Fase 2

El mercado de closing odds es casi inexpugnable con features estaticos (form, Elo, H2H, xG).
La ventana de oportunidad esta en la **asimetria temporal**: entre T-60m (confirmacion de
alineacion) y T-0m (kickoff), el mercado absorbe informacion nueva. Si podemos cuantificar
el impacto de la alineacion ANTES que el mercado lo incorpore completamente, tenemos edge.

### 1.3 Tres Pilares

1. **MTV (Missing Talent Value)**: Cuantificar talent delta al confirmar lineup
2. **CLV (Closing Line Value)**: Medir si nuestras predicciones "cierran bien"
3. **Event-Driven Infrastructure**: Pipeline de re-prediccion con SLA <2s

### 1.4 Enmiendas Vinculantes Integradas

| ID | Origen | Enmienda |
|----|--------|----------|
| ATI #1 | Sancion | SteamChaserModel: modelo secundario XGBoost binario predice line movement |
| ATI #2 | Sancion | VORP Prior: prior bayesiano P25 para jugadores sin historial |
| ATI #3 | Sancion | Degradacion de acero: timeout 5s, fallback Phase 1 + live odds |
| ATI #4 | Sancion | Sweeper Queue: reconciliacion cada 2min post-crash |
| GDT #1 | Ajuste | asof_timestamp: ancla temporal explicita por prediccion |
| GDT #2 | Ajuste | lineup_detected_at: separacion semantica vs lineup_confirmed_at |
| GDT #3 | Ajuste | Bipartite matching: Hungarian algorithm 11v11, precision >98% |
| GDT #4 | Ajuste | Injury-aware Expected XI: filtrar lesionados/suspendidos |
| GDT #5 | Ajuste | CLV 3-way formalizado: log-odds, canonical bookmaker, is_closing real |
| GDT #6 | Ajuste | DB-backed event bus: fuente de verdad en DB, no solo memoria |
| GDT #7 | Ajuste | Cascade optimizada: solo features lineup-dependent |

---

## 2. Inventario de Infraestructura Existente

### 2.1 Tablas de Jugadores

| Tabla | Rows | IDs | Notas |
|-------|------|-----|-------|
| `players` | 1,012 | API-Football (int) | Sparse. Solo nombre, posicion, equipo. |
| `match_lineups` | 118,844 | API-Football (int[]) | `starting_xi_ids`, `starting_xi_positions`. `lineup_confirmed_at` 3.5% coverage (stale 2015-2021). |
| `match_sofascore_player` | 16,289 | Sofascore (varchar) | `rating_pre_match`/`rating_recent_form` ALL NULL. |
| `sofascore_player_rating_history` | 49,319 | Sofascore (varchar) | 9,992 jugadores, 1,634 matches. Solo Nov 2025+. Rating avg=6.80, std=0.60. |
| `player_injuries` | 18,847 | API-Football (int) | Historial lesiones con fechas inicio/fin. |

### 2.2 Tablas de Odds/Mercado

| Tabla | Rows | Cobertura | Notas |
|-------|------|-----------|-------|
| `odds_history` | 18,868 | Multi-bookmaker | Timestamped, `is_opening`/`is_closing`, `source`, quarantine system. |
| `market_movement_snapshots` | 2,841 | Ene 2026+ | T5/T15/T30/T60 (~700/tipo). `bookmaker`, `overround`. |
| `lineup_movement_snapshots` | 1,682 | Ene 2026+ | L0(849)/L+5(814)/L+10(19). Post-lineup market reaction. |

### 2.3 Sofascore Ratings por Liga (top 15)

| Liga | ID | Matches | Jugadores est. | Desde |
|------|----|---------|---------------|-------|
| Championship | 40 | 171 | ~500 | Nov 2025 |
| EPL | 39 | 140 | ~400 | Nov 2025 |
| Serie A | 135 | 120 | ~360 | Nov 2025 |
| Saudi | 307 | 105 | ~300 | Dic 2025 |
| La Liga | 140 | 99 | ~300 | Nov 2025 |
| Turkey | 203 | 74 | ~220 | Nov 2025 |
| Argentina | 128 | 72 | ~220 | Nov 2025 |
| Colombia | 239 | 70 | ~210 | Nov 2025 |
| Ligue 1 | 61 | 65 | ~190 | Nov 2025 |
| Eredivisie | 71 | 56 | ~170 | Nov 2025 |
| Bolivia | 144 | 53 | ~160 | Nov 2025 |
| Belgium | 88 | 82 | ~250 | Nov 2025 |
| Bundesliga | 78 | 90 | ~270 | Nov 2025 |
| Primeira Liga | 94 | 92 | ~280 | Nov 2025 |
| Chile | 265 | 29 | ~90 | Nov 2025 |

### 2.4 TITAN Feature Matrix

| Schema | Columnas | Tiers |
|--------|----------|-------|
| `titan.feature_matrix` | 61 | 1a(odds), 1b(xG), 1c(sofascore_lineup), 1d(xi_depth), form, H2H |

### 2.5 Brechas Criticas

1. **Player ID resolution**: Sofascore IDs (varchar) vs API-Football IDs (int) — NO existe tabla de mapeo
2. **Talent quantification**: Sofascore ratings solo desde Nov 2025 — sin profundidad historica
3. **lineup_confirmed_at**: 3.5% coverage, datos stale. Semantica mixta (provider vs observed)
4. **CLV**: No existe computo. No hay canonical bookmaker definido
5. **Event bus**: Todo APScheduler time-based. No cascada, no reconciliacion

---

## 3. Concepto Transversal: asof_timestamp

### 3.1 Principio (GDT #1)

Cada prediccion tiene un `asof_timestamp` explicito que define el punto temporal de
**TODA** la informacion utilizada. Es la maxima garantia de PIT (Point-in-Time) Safety.

### 3.2 Reglas de Consistencia

```
prediction.asof_timestamp = momento exacto en que se genero la prediccion

Restricciones PIT:
- Odds:     solo odds con recorded_at     <= asof_timestamp
- Lineup:   solo si lineup_detected_at    <= asof_timestamp
- Ratings:  solo de partidos con match_date < asof_timestamp
- Market:   comparar modelo vs mercado en el MISMO asof_timestamp
- CLV:      comparar prob_at_asof vs prob_at_close del MISMO canonical bookmaker
```

### 3.3 Implementacion

- Nueva columna: `predictions.asof_timestamp` (TIMESTAMPTZ)
- Predicciones diarias (05:00 UTC): `asof = datetime.utcnow()` al momento de ejecucion
- Predicciones cascade (post-lineup): `asof = datetime.utcnow()` al iniciar handler
- Toda query de odds, lineup y ratings debe filtrar por `<= asof_timestamp`

### 3.4 Canonical Bookmaker

Para garantizar comparabilidad CLV, definir un **canonical bookmaker** unico por prediccion:

- Prioridad: Bet365 > Pinnacle > 1xBet (misma jerarquia que `PRIORITY_BOOKMAKERS` en `app/etl/base.py`)
- Closing odds: preferir `is_closing=true` de `odds_history`. Si no existe, usar `T5` de `market_movement_snapshots`
- Siempre registrar `canonical_bookmaker` junto con CLV para auditabilidad

---

## 4. Pilar 1: MTV (Missing Talent Value)

### 4.1 Objetivo

Cuantificar el delta de talento cuando se confirma la alineacion a T-60m, antes de que
el mercado incorpore completamente la informacion. Si el XI titular difiere del esperado
(estrella ausente, debutante sorpresa), el mercado necesita minutos para ajustar las cuotas.

### 4.2 Player Entity Resolution (GDT #3)

#### 4.2.1 Tabla

```sql
CREATE TABLE player_id_mapping (
    id                SERIAL PRIMARY KEY,
    api_football_id   INT NOT NULL,
    sofascore_id      VARCHAR NOT NULL,
    player_name       VARCHAR,
    team_id           INT,
    position          VARCHAR,
    confidence        FLOAT NOT NULL,         -- 0.0-1.0
    method            VARCHAR NOT NULL,        -- 'exact_name', 'bipartite', 'fuzzy', 'manual'
    status            VARCHAR DEFAULT 'active', -- 'active', 'blocked', 'pending_review'
    source_match_id   INT,                     -- match donde se detecto el par
    created_at        TIMESTAMPTZ DEFAULT NOW(),
    updated_at        TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(api_football_id, sofascore_id)
);
```

#### 4.2.2 Algoritmo de Matching (Bipartite Hungarian)

**NO usar join posicional naive** (alta tasa de falsos positivos por posiciones ambiguas,
formaciones diferentes, orden de jugadores distinto entre proveedores).

**Algoritmo**:

1. Para cada `match_id` con datos en AMBAS fuentes (`match_lineups` + `match_sofascore_player`):
   - Extraer set de 11 jugadores home de API-Football (desde `starting_xi_ids` + `players`)
   - Extraer set de 11 jugadores home de Sofascore (desde `match_sofascore_player`)
   - Construir **matriz de costos 11x11**:
     ```
     cost(i,j) = 1.0 - score(i,j)
     score(i,j) = 0.5 * name_similarity(normalize(name_i), normalize(name_j))
                + 0.3 * position_compatibility(pos_i, pos_j)
                + 0.2 * shirt_number_match(num_i, num_j)
     ```
   - Resolver con `scipy.optimize.linear_sum_assignment` (Hungarian algorithm O(n^3))
   - Repetir para equipo away

2. **Clasificacion de pares**:
   - Score > 0.7 → `status='active'`, `method='bipartite'`
   - Score 0.5-0.7 → `status='pending_review'` para auditoria manual
   - Score < 0.5 → Descartado

3. **Consolidacion cross-match**: Si el mismo par (api_id, sofascore_id) aparece en multiples
   matches con score > 0.7 → incrementar confidence. Priorizar pares confirmados en >=3 matches.

4. **Overrides manuales**: Soporte para `method='manual'` y `status='blocked'` para corregir
   falsos positivos de jugadores de alto perfil.

#### 4.2.3 Normalizacion de Nombres

```
normalize(name):
  - Lowercase
  - Remove diacritics (unidecode)
  - Remove suffixes: Jr., Sr., III, etc.
  - Normalize whitespace
  - Handle: "J. Rodriguez" <-> "James Rodriguez" (initial expansion from players table)
```

#### 4.2.4 Metrica de Exito Semana 1

- **Precision >98%** en sample auditado manualmente (50 jugadores de alto perfil)
- **Coverage**: Reportar % sobre universo con doble fuente (matches donde ambos proveedores
  tienen datos), NO "70% en 23 ligas" (irrealista dado coverage desigual)

### 4.3 Player Talent Score (PTS) + VORP Prior (ATI #2)

#### 4.3.1 Formula Base

```
PTS_i = SUM(rating_j * minutes_j) / SUM(minutes_j)
        para j en ultimos 10 matches del jugador i
        desde sofascore_player_rating_history
```

#### 4.3.2 VORP Prior (Value Over Replacement Player)

Cuando `SUM(minutes_j) == 0` (debutante, regreso de lesion larga, jugador sin historial):

```
PTS_i = P25(PTS del equipo en posicion del jugador)

Donde:
  P25 = percentil 25 de PTS entre jugadores del mismo equipo
        en la misma posicion (GK/DEF/MID/FWD)
        con >= 3 partidos en la ventana de 10 matches

  Si no hay peers suficientes en posicion:
    PTS_i = P25(PTS global del equipo)

  Si no hay peers en el equipo (equipo nuevo/sin datos):
    PTS_i = 6.30 (P25 de la distribucion global: avg 6.80 - 0.75*std)
```

**Garantia matematica**: La formula NUNCA produce NaN ni ZeroDivisionError. Todo jugador
tiene un PTS computable en todos los escenarios posibles.

#### 4.3.3 Almacenamiento

On-the-fly desde `sofascore_player_rating_history` (49K rows). Sin materializar — el join
por `player_id_ext` + order by `match_date DESC` + limit 10 es eficiente con indice.

#### 4.3.4 Backtest Proxy (pre-Nov 2025)

Para backtesting historico donde no hay Sofascore ratings:
```
PTS_proxy_i = start_frequency_i * 7.0 + (1 - start_frequency_i) * 6.3

Donde:
  start_frequency_i = titularidades / total_matches del equipo en ventana
  7.0 = PTS promedio de titulares regulares
  6.3 = PTS promedio de suplentes (VORP)
```

### 4.4 Expected XI Prediction (GDT #4: Injury-Aware)

#### 4.4.1 Metodo

1. **Pool de candidatos**: Jugadores con >= 1 titularidad en ultimos N matches del equipo
   (desde `match_lineups.starting_xi_ids`)

2. **Filtro de disponibilidad** (GDT #4): Excluir jugadores con lesion/suspension activa:
   ```sql
   SELECT player_external_id FROM player_injuries
   WHERE team_id = :team_id
     AND injury_date <= :match_date
     AND (recovery_date IS NULL OR recovery_date > :match_date)
   ```

3. Para cada candidato disponible:
   ```
   xi_probability_i = starts_in_window / matches_in_window
   ```

4. Seleccionar top-11 por posicion (respetar estructura tactica):
   - 1 GK (mayor xi_probability entre GKs disponibles)
   - 3-5 DEF, 2-5 MID, 1-3 FWD (segun formacion mas frecuente del equipo)

5. **Output**: Lista ordenada de ~18 jugadores con xi_probability, filtrados por disponibilidad

#### 4.4.2 Funcion Existente a Reutilizar

`compute_xi_continuity()` en `app/features/engineering.py:490-559`:
- Ya ejecuta query de historial de XI por equipo (ultimos N matches, PIT-safe)
- Reutilizar la query base, extender con filtro de lesiones y calculo de xi_probability

#### 4.4.3 Efecto del Filtro de Lesiones

| Escenario | Sin filtro | Con filtro |
|-----------|-----------|------------|
| Mbappe lesionado 3 semanas | Expected XI lo incluye (alta start_freq) → su ausencia = shock falso | Expected XI lo excluye → su ausencia = esperada (no shock) |
| Mbappe regresa de lesion | Expected XI lo incluye → su presencia = no shock | Expected XI lo excluye → su regreso = shock positivo real |
| Debutante canterano | Ni aparece (0 start_freq) → usa VORP | Igual → usa VORP |

**Sin filtro, el SteamChaserModel detectaria "falso talento faltante" y apostaria
ciegamente hacia una linea que nunca se iba a mover. El filtro purifica la senal.**

### 4.5 Talent Delta Computation

#### 4.5.1 Formula

```
talent_delta_team = mean(PTS(XI_real)) - mean(PTS(XI_esperado))

  Negativo = equipo debilitado vs expectativa ajustada
  Positivo = equipo reforzado (regreso de estrella, sorpresa tactica)
  ~0        = alineacion esperada, sin informacion nueva
```

#### 4.5.2 Features para Modelo

| Feature | Definicion | Tipo |
|---------|-----------|------|
| `home_talent_delta` | talent_delta del equipo local | Continuo |
| `away_talent_delta` | talent_delta del equipo visitante | Continuo |
| `talent_delta_diff` | home_delta - away_delta | Continuo |
| `home_talent_delta_def` | Delta solo posiciones DEF | Continuo |
| `home_talent_delta_mid` | Delta solo posiciones MID | Continuo |
| `home_talent_delta_fwd` | Delta solo posiciones FWD | Continuo |
| `away_talent_delta_def/mid/fwd` | Idem para visitante | Continuo |
| `shock_magnitude` | max(\|home_delta\|, \|away_delta\|) | Continuo |

### 4.6 Estrategia de Backtest Dual

| Approach | Datos | Horizonte | Limitacion |
|----------|-------|-----------|------------|
| **A) Forward shadow** (Sofascore real) | 49K ratings, Nov 2025+ | 3-6 meses acumulacion | No backtestable pre-Nov 2025 |
| **B) Full backtest** (proxy xi_freq) | 118K match_lineups, 2023+ | Inmediato | Proxy imperfecto de talento |

**MTV full con Sofascore NO es gate de Semana 4**. Es pipeline forward shadow que madura
en meses. El proxy xi_freq permite evaluacion inmediata.

---

## 5. Pilar 2: CLV (Closing Line Value) + SteamChaserModel

### 5.1 CLV Scoring Formalizado (GDT #5)

#### 5.1.1 Definicion Formal

Para outcome k IN {home, draw, away}:

```
CLV_k = log(odds_asof_k / odds_close_k)

  Positivo = obtuvimos mejor precio que el cierre (edge de timing)
  Negativo = mercado se movio contra nosotros
  ~0       = nuestra prediccion no tenia ventaja temporal
```

La forma log-odds garantiza simetria (un movimiento de 2.0→2.5 tiene el mismo peso
que 2.5→2.0 en valor absoluto).

#### 5.1.2 Canonical Bookmaker

| Prioridad | Bookmaker | Razon |
|-----------|-----------|-------|
| 1 | Bet365 | Mayor volumen, mejor proxy de mercado retail |
| 2 | Pinnacle | Sharp bookmaker, mejor para analisis de eficiencia |
| 3 | 1xBet | Alta cobertura en ligas menores |

Usar `odds_history.is_closing=true` del canonical bookmaker. Si no existe closing real,
usar `market_movement_snapshots.T5` como proxy (documentar cuando se use proxy).

#### 5.1.3 Tabla

```sql
CREATE TABLE prediction_clv (
    id                   SERIAL PRIMARY KEY,
    prediction_id        INT REFERENCES predictions(id),
    match_id             INT NOT NULL,
    asof_timestamp       TIMESTAMPTZ NOT NULL,
    canonical_bookmaker  VARCHAR NOT NULL,
    -- Odds crudas al momento asof (antes de de-vig)
    odds_asof_home       NUMERIC,
    odds_asof_draw       NUMERIC,
    odds_asof_away       NUMERIC,
    -- Vector de probabilidades al momento de prediccion (de-vigged)
    prob_asof_home       NUMERIC,
    prob_asof_draw       NUMERIC,
    prob_asof_away       NUMERIC,
    -- Vector de probabilidades al cierre (de-vigged)
    prob_close_home      NUMERIC,
    prob_close_draw      NUMERIC,
    prob_close_away      NUMERIC,
    -- CLV por outcome (log-odds)
    clv_home             NUMERIC,
    clv_draw             NUMERIC,
    clv_away             NUMERIC,
    -- CLV del outcome seleccionado (si hay value bet)
    selected_outcome     VARCHAR,   -- 'home'/'draw'/'away'/NULL
    clv_selected         NUMERIC,
    -- Metadata: "asof_source|close_method"
    -- asof_source: 'pit_aligned' (odds at asof_ts) | 'opening_proxy' (earliest recorded)
    -- close_method: 'is_closing' | 'latest_pre_kickoff' | 'single_snapshot'
    close_source         VARCHAR,
    created_at           TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (prediction_id, canonical_bookmaker)
);
-- Indices
CREATE INDEX idx_prediction_clv_match ON prediction_clv (match_id);
CREATE INDEX idx_prediction_clv_prediction ON prediction_clv (prediction_id);
```

> **Nota Sprint 1 (implementado)**: `close_source` contiene un valor compuesto
> `"asof_source|close_method"` para auditabilidad. Filtrar por
> `close_source LIKE 'pit_aligned%'` para obtener solo scores PIT-correctos.
> Predicciones historicas usan `opening_proxy` como fallback.

#### 5.1.4 Uso

CLV es **metrica de evaluacion post-hoc**. NUNCA feature de input (seria leakage de futuro).

Metricas de reporte por liga y ventana temporal:
- CLV mean / median / %>0
- CLV por outcome class (home/draw/away)
- CLV distribution (histograma)

### 5.2 SteamChaserModel (ATI #1)

#### 5.2.1 Objetivo

Modelo secundario XGBoost de **clasificacion binaria** que predice **line movement**,
no outcome del partido. El target es:

```
y = 1  si  prob_close_k > prob_T60_k + threshold
y = 0  otherwise

Donde:
  k = outcome con mayor |talent_delta_signal|
  threshold = 2% (calibrar por liga)
```

#### 5.2.2 Features

| Feature | Fuente | Disponibilidad |
|---------|--------|----------------|
| `home_talent_delta` | MTV computation | T-60m (post lineup) |
| `away_talent_delta` | MTV computation | T-60m |
| `shock_magnitude` | MTV computation | T-60m |
| `overround_T60` | market_movement_snapshots | T-60m |
| `prob_T60_home/draw/away` | market_movement_snapshots | T-60m |
| `xi_continuity_home/away` | Pre-computado batch diario | T-60m |
| `injury_count_change` | player_injuries delta semanal | T-60m |

#### 5.2.3 Pipeline

- XGBoost separado del modelo 1X2 principal
- Entrenado sobre `market_movement_snapshots` (T60 vs T5/closing)
- Walk-forward con ventanas de 2 meses (dado datos limitados)
- **NO productivo en Semana 4**: Solo pipeline + data collection. Requiere >=2,000 snapshots (~3 meses)

#### 5.2.4 Uso Comercial

```
Si SteamChaser.predict(features_T60) = 1 (linea va a colapsar para Team A):
  → Apostar Team A AHORA (a T-60m) antes de que la casa ajuste
  → El value bet no es "Team A gana", sino "el precio de Team A va a bajar"
```

### 5.3 Line Movement Features (PIT-safe)

Features derivados de `market_movement_snapshots`, usables como input del modelo 1X2:

| Feature | Calculo | PIT |
|---------|---------|-----|
| `line_drift_T60_T30` | prob_T30 - prob_T60 | Disponible a T-30 |
| `line_drift_magnitude` | \|prob_T60 - prob_T30\| | Disponible a T-30 |
| `overround_T60` | sum(implied_probs) at T60 | Disponible a T-60 |
| `overround_delta` | overround_T30 - overround_T60 | Disponible a T-30 |

**Limitacion**: Solo 2,841 snapshots (Ene 2026+). Acumular 3-6 meses antes de usar como features.

---

## 6. Pilar 3: Event-Driven Infrastructure

### 6.1 lineup_detected_at (GDT #2)

**Separacion semantica** — NO reutilizar `lineup_confirmed_at`:

| Columna | Semantica | Quien la pone | Uso PIT |
|---------|-----------|---------------|---------|
| `lineup_confirmed_at` | Timestamp del provider (API-Football) | Sync job historico | Legacy, no usar para PIT |
| `lineup_detected_at` | Timestamp de NUESTRA deteccion | Lineup monitoring job | **SI — es "cuando lo supimos"** |

```sql
ALTER TABLE match_lineups ADD COLUMN lineup_detected_at TIMESTAMPTZ;
CREATE INDEX idx_ml_detected ON match_lineups(lineup_detected_at) WHERE lineup_detected_at IS NOT NULL;
```

Going-forward: lineup monitoring job escribe `lineup_detected_at = NOW()` al detectar
cambio de `confirmed=false` a `confirmed=true`.

### 6.2 DB-Backed Event Bus (GDT #6)

#### 6.2.1 Diseno

`asyncio.Queue` para dispatch inmediato **pero DB como fuente de verdad**.

```
Fuente de verdad: match_lineups.lineup_detected_at (DB persistente)
Dispatch rapido:  asyncio.Queue (in-memory, efimero)
Reconciliacion:   Sweeper Queue cada 2 min (DB scan)
```

Si el contenedor crashea, la Queue se pierde. El Sweeper reconstruye automaticamente
los eventos pendientes desde DB.

#### 6.2.2 Eventos

| Evento | Trigger | Payload |
|--------|---------|---------|
| `LINEUP_CONFIRMED` | Lineup monitoring detecta confirmacion | match_id, team_id, xi_ids, lineup_detected_at |
| `ODDS_UPDATED` | Odds sync detecta cambio >2% implied | match_id, odds_home/draw/away, bookmaker, recorded_at |
| `PREDICTION_STALE` | Cualquier evento que invalida prediccion vigente | match_id, reason, trigger_event |

#### 6.2.3 Idempotencia y Dedupe (GDT #6)

Handlers son **idempotentes**: si se ejecutan 2 veces para el mismo match_id + asof_timestamp
(truncado a minuto), la segunda ejecucion detecta prediccion ya existente y skip.

```sql
-- Dedupe: verificar antes de insertar nueva prediccion
SELECT 1 FROM predictions
WHERE match_id = :match_id
  AND model_version = :version
  AND asof_timestamp >= :asof_truncated_to_minute
```

### 6.3 Sweeper Queue (ATI #4)

Job APScheduler cada 2 minutos:

```sql
-- Partidos que necesitan re-prediccion pero no la tienen
SELECT m.id, ml.lineup_detected_at
FROM matches m
JOIN match_lineups ml ON ml.match_id = m.id
WHERE m.date BETWEEN NOW() AND NOW() + INTERVAL '65 minutes'
  AND ml.lineup_detected_at IS NOT NULL
  AND NOT EXISTS (
    SELECT 1 FROM predictions p
    WHERE p.match_id = m.id
      AND p.asof_timestamp > ml.lineup_detected_at
  )
```

Si encuentra matches → auto-inyecta en `asyncio.Queue` para re-prediction.

**Escenario**: Contenedor crashea sabado 15:30 (hora pico futbol europeo).
Se reinicia a 15:31. A 15:32, Sweeper detecta 5 partidos con lineup confirmada
pero sin prediccion post-lineup → los reencola → cascade los procesa en ~10s total.

### 6.4 Cascade Optimizada (GDT #7 + ATI #3)

#### 6.4.1 Happy Path (~1.9s)

```
LINEUP_CONFIRMED(match_id)
  → fetch_latest_odds(match_id)                    [~300ms]
  → compute_talent_delta(match_id, timeout=5s)     [~500ms]
  → re_predict(match_id, with_lineup_features)     [~1s]
  → apply_market_anchor(match_id)                  [~10ms]
  → save_prediction(match_id, asof=NOW())          [~100ms]
Total: ~1.9s
```

**Eliminado de cascade** (GDT #7): `compute_xi_continuity()` NO se ejecuta en la cascade.
Es casi estatico (depende de ultimos 15 partidos, no cambia al confirmar lineup de hoy).
Pre-computado en batch diario.

#### 6.4.2 Degradacion de Acero (ATI #3)

```python
async def cascade_handler(match_id: int) -> None:
    """Re-predict after lineup confirmation. NEVER fails silently."""
    try:
        # Step 1: Live odds (obligatorio)
        odds = await fetch_latest_odds(match_id)
        if not odds or not all(o > 1.0 for o in [odds.home, odds.draw, odds.away]):
            raise OddsUnavailable(f"No valid live odds for {match_id}")

        # Step 2: Talent delta (con timeout de 5s)
        try:
            talent_delta = await asyncio.wait_for(
                compute_talent_delta(match_id),
                timeout=5.0
            )
            prediction = re_predict(match_id, talent_delta=talent_delta)
        except (asyncio.TimeoutError, Exception) as e:
            # MTV ABORTADO: fallback a modelo Phase 1 estandar
            logger.warning(f"MTV failed for match {match_id}: {e}. "
                          f"Falling back to Phase 1 model.")
            prediction = re_predict_phase1(match_id)  # modelo tabular sin MTV

        # Step 3: Market Anchor (siempre, con live odds)
        prediction = apply_market_anchor(prediction, odds)

        # Step 4: Persistir con asof_timestamp
        await save_prediction(prediction, asof=datetime.utcnow())
        logger.info(f"Cascade OK for match {match_id}: "
                    f"{'MTV' if talent_delta else 'Phase1'} + anchor")

    except Exception as e:
        # FALLBACK TOTAL: la prediccion diaria de 05:00 UTC sigue activa
        logger.error(f"Cascade FAILED for match {match_id}: {e}. "
                     f"Daily prediction remains active.")
```

**Regla inviolable**: La operacion comercial NO se detiene jamas.
- Si Sofascore cae → Phase 1 model + Market Anchor + live odds
- Si API-Football cae → Phase 1 model + cached odds + Market Anchor
- Si TODO cae → prediccion diaria de 05:00 UTC sigue vigente

#### 6.4.3 Backward Compatibility

- `daily_save_predictions` (05:00 UTC) sigue como baseline y fallback
- Event cascade es un UPGRADE, no un reemplazo
- Feature flag: `EVENT_CASCADE_ENABLED=false` (default off)
- Activar solo despues de validar en staging con >=50 matches

---

## 7. Anti-Leakage Backtesting

### 7.1 Principios PIT (actualizados con asof_timestamp)

1. `asof_timestamp` es el ancla de TODA comparacion y evaluacion
2. Odds: `recorded_at <= asof_timestamp`, mismo canonical bookmaker para modelo y mercado
3. Lineup: solo si `lineup_detected_at <= asof_timestamp`
4. Talent scores: solo ratings de partidos con `match_date < asof_timestamp`
5. CLV: comparar `prob_at_asof` vs `prob_at_close` del MISMO canonical bookmaker
6. Expected XI: solo usar start_frequency de partidos con `date < asof_timestamp`
7. Injuries: solo lesiones conocidas con `injury_date <= asof_timestamp`

### 7.2 Backtest Strategy por Pilar

| Pilar | Metodo | Datos | Gate |
|-------|--------|-------|------|
| MTV proxy (xi_freq_delta) | Full backtest walk-forward | match_lineups 118K, 2023+ | Semana 4 |
| MTV full (Sofascore PTS) | Forward shadow | 49K ratings, Nov 2025+ | Mes 3+ (NO gate S4) |
| CLV scoring | Forward, post-hoc | odds_history + snapshots, Ene 2026+ | Mes 2-3 |
| SteamChaserModel | Forward shadow | T60 snapshots, Ene 2026+ | Mes 3+ (>=2K snapshots) |
| Event cascade | A/B test live | Predicciones con/sin cascade | Inmediato |

### 7.3 Metricas de Evaluacion (GDT revisadas)

| Metrica | Que mide | Mejor que |
|---------|----------|-----------|
| Delta-logloss vs baseline | Poder predictivo incremental de MTV | "Top-10 SHAP" |
| AUC de "big move" | Capacidad de detectar movimientos grandes | Binary accuracy |
| CLV mean/median/%>0 | Consistencia del edge temporal | CLV > threshold simple |
| Correlation(talent_delta, line_movement) | Validez de la tesis MTV | P-value solo |

---

## 8. Epic Backlog — 4 Semanas

### Semana 1: Canon de Datos + Entity Resolution

**Gate S1→S2**: P2-02 mergeado + P2-01 con precision >98% auditada

| ID | Ticket | Est. | Deps | Entregable |
|----|--------|------|------|------------|
| P2-02 | asof_timestamp + canonical bookmaker | 4h | — | Columna predictions.asof_timestamp, query canonical |
| P2-01 | Player entity resolution (Hungarian) | 8h | — | Tabla player_id_mapping, script, audit 50 jugadores |
| P2-03 | lineup_detected_at + monitoring update | 3h | — | Columna nueva, going-forward population |
| P2-04 | CLV table + computation job | 5h | P2-02 | Tabla prediction_clv, job post-match, log-odds |

### Semana 2: Features MTV + VORP

**Gate S2→S3**: P2-07 computable en 4 ligas canario sin errores

| ID | Ticket | Est. | Deps | Entregable |
|----|--------|------|------|------------|
| P2-05 | PTS + VORP prior (P25 bayesiano) | 6h | P2-01 | compute_pts() con VORP, zero-div impossible |
| P2-06 | Expected XI injury-aware | 6h | P2-01 | predict_expected_xi() con filtro lesiones |
| P2-07 | Talent delta features | 5h | P2-05, P2-06 | talent_delta total + posicional + shock_magnitude |
| P2-08 | XI frequency proxy (backtest) | 3h | — | xi_freq_delta_home/away desde match_lineups |

### Semana 3: Event-Driven + Cascade

**Gate S3→S4**: Cascade completa happy + degradation path validados

| ID | Ticket | Est. | Deps | Entregable |
|----|--------|------|------|------------|
| P2-09 | DB-backed event bus + Sweeper Queue | 6h | P2-03 | Dispatcher + sweeper 2min + idempotent handlers |
| P2-10 | LINEUP_CONFIRMED cascade + degradacion | 7h | P2-07, P2-09 | Handler con timeout 5s, fallback Phase 1 |
| P2-11 | Line movement features PIT-safe | 4h | P2-02 | Features market_movement_snapshots |
| P2-12 | CLV dashboard metric | 3h | P2-04 | CLV rolling por liga (mean/median/%>0) |

### Semana 4: SteamChaser + Evaluacion

| ID | Ticket | Est. | Deps | Entregable |
|----|--------|------|------|------------|
| P2-13 | SteamChaserModel pipeline + data collection | 6h | P2-07, P2-11 | Pipeline XGBoost binario, acumulacion datos |
| P2-14 | Feature Lab integration (MTV proxy) | 5h | P2-08 | Tests xi_freq_delta, delta-logloss, AUC big move |
| P2-15 | A/B test cascade vs daily | 3h | P2-10 | Metricas frescura + accuracy |
| P2-16 | Phase 2 evaluation report | 4h | Todos | Documento CLV distribution, MTV results, Phase 3 rec |

**Total**: ~78h (4 semanas, ~19.5h/semana)

---

## 9. Orden de Dependencias

```
P2-02 (asof_timestamp + canonical)  ← FUNDAMENTO, bloqueante
  |
  +--→ P2-04 (CLV scoring)
  +--→ P2-11 (line movement features)

P2-03 (lineup_detected_at)  ← Independiente, parallel con P2-02
  |
  +--→ P2-09 (event bus + sweeper)

P2-01 (player ID mapping)  ← Independiente, parallel con P2-02
  |
  +--→ P2-05 (PTS + VORP)
  +--→ P2-06 (Expected XI + injuries)
       |
       +--→ P2-07 (talent delta)
            |
            +--→ P2-10 (cascade)
            +--→ P2-13 (SteamChaser)

P2-08 (xi_freq proxy)  ← Independiente (no necesita Sofascore)
  |
  +--→ P2-14 (Feature Lab MTV proxy)

P2-10 (cascade) → P2-15 (A/B test)
P2-04 (CLV) → P2-12 (CLV dashboard)
Todos → P2-16 (evaluation report)
```

---

## 10. Archivos a Crear/Modificar

| Archivo | Accion | LOC est. |
|---------|--------|----------|
| `scripts/build_player_id_mapping.py` | CREAR | ~300 |
| `app/features/engineering.py` | MODIFICAR | +250 |
| `app/events/bus.py` | CREAR | ~200 |
| `app/events/handlers.py` | CREAR | ~280 |
| `app/scheduler.py` | MODIFICAR | +100 |
| `app/ml/steamchaser.py` | CREAR | ~180 |
| `app/ml/devig.py` | MODIFICAR | +20 (canonical bookmaker helper) |
| `app/routes/api.py` | MODIFICAR | +30 (asof_timestamp en serving) |
| `scripts/feature_lab.py` | MODIFICAR | +150 |
| DB migrations (3-4) | CREAR | ~100 |

---

## 11. Metricas de Exito

| Metrica | Target | Horizonte |
|---------|--------|-----------|
| Player ID mapping precision | >98% en 50 jugadores auditados | Semana 1 |
| Player ID mapping coverage | Reportar % sobre universo doble fuente | Semana 1 |
| lineup_detected_at going forward | >95% lineups nuevos | Semana 1+ |
| CLV distribution por liga | Reportar mean/median/%>0 | Mes 2-3 |
| Cascade latency happy path | <2s per match | Semana 3 |
| Cascade degradation test | 100% fallback exitoso en error sim | Semana 3 |
| Sweeper Queue auto-heal | 100% matches recuperados post-restart | Semana 3 |
| MTV proxy delta-logloss | Reportar vs baseline en OOT | Semana 4 |
| MTV proxy AUC "big move" | Reportar en 4 ligas canario | Semana 4 |
| SteamChaser data accumulated | >=500 snapshots T60→close | Semana 4 |

---

## 12. Riesgos y Mitigaciones

| Riesgo | Prob | Impacto | Mitigacion |
|--------|------|---------|-----------|
| Bipartite matching precision <90% | Baja | Alto | Audit manual top-50, override system, pending_review |
| Sofascore insuficiente para MTV full | Alta | Alto | Proxy xi_freq para backtest, forward shadow para real |
| Expected XI shocks falsos | Alta sin filtro | Alto | Filtro player_injuries obligatorio (GDT #4) |
| SteamChaser sin datos suficientes S4 | Alta | Medio | Solo pipeline + collection, training mes 3+ |
| Event bus pierde eventos en crash | Media | Alto | Sweeper Queue cada 2min reconcilia (ATI #4) |
| Cascade timeout cascading | Baja | Alto | Hard timeout 5s + fallback Phase 1 (ATI #3) |
| CLV no auditable por mezcla bookmakers | Media | Alto | Canonical bookmaker strict + close_source auditable (GDT #5) |
| asof_timestamp no implementado | — | Critico | P2-02 bloqueante S1, sin esto NADA es auditable |
| VORP division por cero | — | Critico | P25 prior + global fallback 6.30 (ATI #2) |

---

## 13. Glosario

| Termino | Definicion |
|---------|-----------|
| **MTV** | Missing Talent Value: delta de talento entre XI real y XI esperado |
| **CLV** | Closing Line Value: diferencia entre odds at prediction vs odds at close |
| **PTS** | Player Talent Score: rating Sofascore ponderado por minutos |
| **VORP** | Value Over Replacement Player: prior bayesiano P25 para jugadores sin datos |
| **SteamChaser** | Modelo secundario que predice line movement, no outcome |
| **asof_timestamp** | Ancla temporal que define el punto de toda informacion usada |
| **Canonical bookmaker** | Bookmaker de referencia unico para comparaciones CLV |
| **Sweeper Queue** | Job de reconciliacion que detecta eventos perdidos post-crash |
| **Bipartite matching** | Algoritmo Hungaro para resolver asignacion optima 11v11 |
| **Cascade** | Pipeline event-driven de re-prediccion post-lineup |
