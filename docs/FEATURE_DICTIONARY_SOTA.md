# FutbolStats — Feature Dictionary SOTA (Point-in-Time)

Diccionario oficial de features para el stack SOTA. Cada feature debe cumplir:
- **as_of**: disponible antes del kickoff (o snapshot predefinido), nunca post-partido.
- **fuente**: trazable (API-Football / Understat / Sofascore / Open-Meteo / derivadas).
- **imputación + flags**: degradación controlada cuando falta la señal.

## Convenciones
- \(t_0\): kickoff UTC (`matches.date`)
- Ventana rolling: últimos \(N\) partidos (default actual `ROLLING_WINDOW`)
- Decaimiento temporal: \(w(\Delta d)=e^{-\lambda \Delta d}\) (default actual `TIME_DECAY_LAMBDA`)
- Toda columna derivada debe especificar dependencias.

---

## 1) Features existentes (baseline / compatibilidad)

> Estas ya existen en `app/features/engineering.py` y son el backbone. Se mantienen para backward compatibility.

| Feature | Grupo | Fuente | Fórmula / Definición | as_of | Imputación | Tests anti-leakage |
|---|---|---|---|---|---|---|
| `home_goals_scored_avg` | form_base | API-Football | Promedio ponderado (match_weight × decay) de goles anotados por home en últimos N partidos previos | \(< t_0\) | default 1.0 si no hay historia | asegurar que solo usa `Match.date < t0` |
| `home_goals_conceded_avg` | form_base | API-Football | idem concedidos | \(< t_0\) | default 1.0 | idem |
| `home_shots_avg` | form_base | API-Football | promedio ponderado de tiros (stats JSON) | \(< t_0\) | default 10.0 | excluir stats vacíos post-match |
| `home_corners_avg` | form_base | API-Football | promedio ponderado corners | \(< t_0\) | default 4.0 | idem |
| `home_rest_days` | schedule | API-Football | días desde último partido previo | \(< t_0\) | default 30 | usar historial previo |
| `home_matches_played` | schedule | API-Football | tamaño de historial disponible | \(< t_0\) | 0 | - |
| `away_goals_scored_avg` | form_base | API-Football | análogo away | \(< t_0\) | default 1.0 | idem |
| `away_goals_conceded_avg` | form_base | API-Football | análogo away | \(< t_0\) | default 1.0 | idem |
| `away_shots_avg` | form_base | API-Football | análogo away | \(< t_0\) | default 10.0 | idem |
| `away_corners_avg` | form_base | API-Football | análogo away | \(< t_0\) | default 4.0 | idem |
| `away_rest_days` | schedule | API-Football | análogo away | \(< t_0\) | default 30 | idem |
| `away_matches_played` | schedule | API-Football | análogo away | \(< t_0\) | 0 | - |
| `goal_diff_avg` | derived | derivada | `home_goals_scored_avg - away_goals_scored_avg` | \(< t_0\) | 0 | - |
| `rest_diff` | derived | derivada | `home_rest_days - away_rest_days` | \(< t_0\) | 0 | - |
| `abs_attack_diff` | draw_competitiveness | derivada | `abs(home_goals_scored_avg - away_goals_scored_avg)` | \(< t_0\) | 0 | - |
| `abs_defense_diff` | draw_competitiveness | derivada | `abs(home_goals_conceded_avg - away_goals_conceded_avg)` | \(< t_0\) | 0 | - |
| `abs_strength_gap` | draw_competitiveness | derivada | `abs((home_scored-home_conceded) - (away_scored-away_conceded))` | \(< t_0\) | 0 | - |
| `implied_draw` | market | odds | ` (1/odds_draw) / ((1/odds_home)+(1/odds_draw)+(1/odds_away)) ` | \(< t_0\) | 0.25 | odds must be captured pre-kickoff |

---

## 2) Understat — “Proceso” (xG/xPTS) + Regresión a la media (Justicia)

### 2.1 Features de proceso (team-level)
| Feature | Grupo | Fuente | Fórmula / Definición | as_of | Imputación | Tests anti-leakage |
|---|---|---|---|---|---|---|
| `home_xg_for_avg` | understat_form | Understat | rolling ponderado de xG a favor (home) en últimos N partidos | \(< t_0\) | usar media liga/temporada + flag | verificar `captured_at < t0` si se usa snapshot |
| `home_xg_against_avg` | understat_form | Understat | rolling ponderado de xG en contra (home) | \(< t_0\) | idem | idem |
| `away_xg_for_avg` | understat_form | Understat | análogo away | \(< t_0\) | idem | idem |
| `away_xg_against_avg` | understat_form | Understat | análogo away | \(< t_0\) | idem | idem |
| `xg_diff_avg` | understat_derived | derivada | `home_xg_for_avg - away_xg_for_avg` | \(< t_0\) | 0 | - |
| `xpts_diff_avg` | understat_derived | Understat | rolling ponderado de xPTS diff (si disponible) | \(< t_0\) | 0 | idem |

### 2.2 Feature “Justicia” (Delta real vs esperado)
Definiciones (ventana \(\mathcal{W}\), últimos N partidos):
- \(G_T = \sum goals_T\), \(XG_T = \sum xg_T\)
- \(\text{justice}_T = \frac{G_T - XG_T}{\sqrt{XG_T+\epsilon}}\)
- \(\rho=\frac{n}{n+k}\), \(\text{justice\_shrunk}_T=\rho \cdot \text{justice}_T\)

| Feature | Grupo | Fuente | Fórmula / Definición | as_of | Imputación | Tests anti-leakage |
|---|---|---|---|---|---|---|
| `home_justice_shrunk` | understat_justice | Understat+AF | \(\rho \cdot (G-XG)/sqrt(XG+\epsilon)\) | \(< t_0\) | 0 + `justice_missing=1` | asegurar que \(G\) y \(XG\) solo usan partidos anteriores |
| `away_justice_shrunk` | understat_justice | Understat+AF | análogo | \(< t_0\) | 0 | idem |
| `justice_diff` | understat_justice | derivada | `home_justice_shrunk - away_justice_shrunk` | \(< t_0\) | 0 | - |

Flags recomendados:
- `understat_missing` (0/1)
- `understat_samples_home`, `understat_samples_away` (n)

---

## 3) Sofascore — XI / ratings / formación

### 3.1 Vectorización XI (WeightedXI + distribución)
Normalización:
- Si ratings están en [0,10], opcionalmente centrar/estandarizar por liga/temporada.

Pesos por posición (default):
- GK=1.0, DEF=0.9, MID=1.0, FWD=1.1

| Feature | Grupo | Fuente | Fórmula / Definición | as_of | Imputación | Tests anti-leakage |
|---|---|---|---|---|---|---|
| `xi_weighted_home` | xi | Sofascore | \(\sum_{i \in XI} w_{pos(i)} z_i\) | snapshot pre-kickoff | baseline por squad + `xi_missing=1` | `captured_at < t0` |
| `xi_weighted_away` | xi | Sofascore | análogo | snapshot pre-kickoff | idem | idem |
| `xi_weighted_diff` | xi | derivada | `xi_weighted_home - xi_weighted_away` | \(< t_0\) | 0 | - |
| `xi_p10_home` | xi_dist | Sofascore | percentil 10 de \(z_i\) en XI | \(< t_0\) | media histórica | `captured_at < t0` |
| `xi_p50_home` | xi_dist | Sofascore | percentil 50 | \(< t_0\) | media histórica | idem |
| `xi_p90_home` | xi_dist | Sofascore | percentil 90 | \(< t_0\) | media histórica | idem |
| `xi_weaklink_home` | xi_dist | Sofascore | \(\min(z_i)\) | \(< t_0\) | media histórica | idem |
| `xi_std_home` | xi_dist | Sofascore | std dev de \(z_i\) | \(< t_0\) | media histórica | idem |
| `xi_p10_away` | xi_dist | Sofascore | análogo away | \(< t_0\) | media histórica | idem |
| `xi_p50_away` | xi_dist | Sofascore | análogo | \(< t_0\) | media histórica | idem |
| `xi_p90_away` | xi_dist | Sofascore | análogo | \(< t_0\) | media histórica | idem |
| `xi_weaklink_away` | xi_dist | Sofascore | análogo | \(< t_0\) | media histórica | idem |
| `xi_std_away` | xi_dist | Sofascore | análogo | \(< t_0\) | media histórica | idem |

### 3.2 Formación (si disponible)
| Feature | Grupo | Fuente | Definición | as_of | Imputación | Tests |
|---|---|---|---|---|---|---|
| `formation_home` | xi_formation | Sofascore | encoding (one-hot / ordinal) de formación (4-3-3, 3-5-2, etc.) | \(< t_0\) | “unknown” + flag | `captured_at < t0` |
| `formation_away` | xi_formation | Sofascore | análogo | \(< t_0\) | idem | idem |

Flags recomendados:
- `xi_missing` (0/1)
- `xi_captured_horizon_minutes` (cuántos minutos antes de kickoff se capturó)

---

## 4) Open-Meteo — clima + bio-adaptabilidad

### 4.1 Clima base a kickoff
| Feature | Grupo | Fuente | Definición | as_of | Imputación | Tests |
|---|---|---|---|---|---|---|
| `weather_temp_c` | weather | Open-Meteo | temperatura forecast en kickoff | snapshot (T-24h y/o T-1h) | climatología mensual + flag | `captured_at < t0` |
| `weather_humidity` | weather | Open-Meteo | humedad % | \(< t_0\) | climatología + flag | idem |
| `weather_wind_ms` | weather | Open-Meteo | viento m/s | \(< t_0\) | 0 + flag | idem |
| `weather_precip_mm` | weather | Open-Meteo | precipitación mm | \(< t_0\) | 0 + flag | idem |
| `is_daylight` | weather | derivada | 1 si kickoff entre sunrise/sunset local | \(< t_0\) | derivable | - |

Flags recomendados:
- `weather_missing` (0/1)
- `weather_forecast_horizon_hours` (24, 1)

### 4.2 Thermal_Shock (aclimatación)
| Feature | Grupo | Fuente | Definición | as_of | Imputación | Tests |
|---|---|---|---|---|---|---|
| `thermal_shock` | bio | Open-Meteo + team_profile | \(T_{stad} - \bar T_{away,month}\) | \(< t_0\) | 0 + flag | inputs must be pre-kickoff |
| `thermal_shock_abs` | bio | derivada | `abs(thermal_shock)` | \(< t_0\) | 0 | - |

### 4.3 Circadian_Disruption
| Feature | Grupo | Fuente | Definición | as_of | Imputación | Tests |
|---|---|---|---|---|---|---|
| `circadian_disruption` | bio | schedule + team_profile | \(d/12\) donde \(d\) es distancia circular a hora típica | \(< t_0\) | 0 | solo usa horarios previos |
| `tz_shift` | bio | team_profile | \(|tz_{match} - tz_{away\_base}|\) | \(< t_0\) | 0 | - |
| `bio_disruption` | bio | derivada | `a*circadian_disruption + b*min(tz_shift,6)/6` | \(< t_0\) | 0 | - |

---

## 5) Market microstructure (odds_history)

> Requiere snapshots consistentes (opening / lineup / close). Si no existen, se computa con el snapshot más cercano pre-kickoff.

| Feature | Grupo | Fuente | Definición | as_of | Imputación | Tests |
|---|---|---|---|---|---|---|
| `odds_log_move_open_to_close_home` | market_move | odds_history | `log(odds_close_home) - log(odds_open_home)` | \(< t_0\) | 0 + flag | snapshots deben ser pre-kickoff |
| `odds_log_move_open_to_close_draw` | market_move | odds_history | análogo draw | \(< t_0\) | 0 | idem |
| `odds_log_move_open_to_close_away` | market_move | odds_history | análogo away | \(< t_0\) | 0 | idem |
| `steam_move_flag` | market_move | odds_history | 1 si \(|Δlog(odds)| > τ\) y confirmado en ≥2 books | \(< t_0\) | 0 | idem |

Flags recomendados:
- `odds_open_missing`, `odds_close_missing`, `odds_lineup_missing`

---

## 6) Campos “control” obligatorios (para robustez y auditoría)

Estos NO son señales predictivas per se; son guardrails para producción:
- `*_missing` flags por fuente/grupo
- `*_samples_*` (tamaño muestral real usado por rolling)
- `feature_snapshot_age_minutes` (staleness)
- `tainted` (excluir en train/backtests)

---

## 7) Tests anti-leakage (suite mínima)

1. **Point-in-time assertion**: para cada partido, ninguna consulta puede leer eventos con `Match.date >= t0` del mismo equipo.
2. **Captured_at assertion**: cualquier feature derivada de snapshots (weather/XI/odds movement) debe cumplir `captured_at < t0`.
3. **Time travel test**: elegir una fecha histórica, entrenar hasta esa fecha, predecir los próximos K días y comparar con lo que habría estado disponible en ese momento.
4. **Leakage canary**: introducir un feature “prohibido” (p.ej. goles FT) en un experimento local debe disparar alertas (métrica irreal + test rojo).

