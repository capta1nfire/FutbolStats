# API-Football Payload Inventory (Post-Match)

**Fecha:** 2026-01-14
**Objetivo:** Documentar qué datos entrega API-Football cuando un partido termina (FT/AET/PEN)

---

## 1. Resumen de Endpoints Disponibles

| Endpoint | Descripción | Disponibilidad | Uso en Bon Jogo |
|----------|-------------|----------------|---------------------|
| `/fixtures?id=X` | Datos básicos del partido | Siempre | ✅ Activo |
| `/fixtures/statistics?fixture=X` | Estadísticas del partido | Post-FT | ✅ Activo |
| `/fixtures/events?fixture=X` | Goles, tarjetas, VAR, subs | Post-FT | ✅ Activo |
| `/fixtures/lineups?fixture=X` | Alineaciones y formaciones | ~60min pre-KO | ✅ Activo |
| `/fixtures/players?fixture=X` | Stats por jugador + rating | Post-FT | ⚠️ No usado |

---

## 2. Detalle por Endpoint

### 2.1 `/fixtures` - Datos Básicos

**Estructura Response:**
```
response[].fixture
  ├── id                    # int - ID único del partido
  ├── referee               # str | null - Nombre del árbitro
  ├── timezone              # str - "UTC"
  ├── date                  # str - ISO8601
  ├── timestamp             # int - Unix timestamp
  ├── venue
  │   ├── id               # int | null
  │   ├── name             # str - "Signal Iduna Park"
  │   └── city             # str - "Dortmund"
  └── status
      ├── long             # str - "Match Finished"
      ├── short            # str - "FT" | "AET" | "PEN"
      └── elapsed          # int | null - minutos

response[].teams
  ├── home
  │   ├── id              # int - Team external ID
  │   ├── name            # str
  │   ├── logo            # str - URL
  │   └── winner          # bool | null
  └── away (misma estructura)

response[].goals
  ├── home                # int | null
  └── away                # int | null

response[].score
  ├── halftime {home, away}
  ├── fulltime {home, away}
  ├── extratime {home, away}  # Solo si AET
  └── penalty {home, away}    # Solo si PEN
```

**Campos Faltantes (NO disponibles):**
- ❌ `weather` - No incluido
- ❌ `pitch_condition` - No incluido
- ❌ `attendance` - No incluido (a veces en `venue`)

---

### 2.2 `/fixtures/statistics` - Estadísticas del Partido

**Estructura Response:**
```
response[0] = Home Team Stats
response[1] = Away Team Stats

Cada uno:
  ├── team {id, name, logo}
  └── statistics[]
      └── {type: str, value: int|str|null}
```

**Campos Disponibles (type):**

| Campo API | Nuestro key | Tipo | Confiabilidad |
|-----------|-------------|------|---------------|
| "Ball Possession" | ball_possession | "55%" → 55.0 | ✅ Alta |
| "Total Shots" | total_shots | int | ✅ Alta |
| "Shots on Goal" | shots_on_goal | int | ✅ Alta |
| "Shots off Goal" | shots_off_goal | int | ✅ Alta |
| "Blocked Shots" | blocked_shots | int | ✅ Alta |
| "Shots insidebox" | shots_insidebox | int | ⚠️ Media |
| "Shots outsidebox" | shots_outsidebox | int | ⚠️ Media |
| "Fouls" | fouls | int | ✅ Alta |
| "Corner Kicks" | corner_kicks | int | ✅ Alta |
| "Offsides" | offsides | int | ⚠️ Media |
| "Yellow Cards" | yellow_cards | int | ✅ Alta |
| "Red Cards" | red_cards | int | ✅ Alta |
| "Goalkeeper Saves" | goalkeeper_saves | int | ✅ Alta |
| "Total passes" | total_passes | int | ⚠️ Media |
| "Passes accurate" | passes_accurate | int | ⚠️ Media |
| "Passes %" | passes_pct | "80%" | ⚠️ Media |
| "expected_goals" | expected_goals | float | ⚠️ No siempre |

**Nota:** `expected_goals` (xG) no siempre está disponible. Depende de la liga.

---

### 2.3 `/fixtures/events` - Timeline del Partido

**Estructura Response:**
```
response[]
  ├── time
  │   ├── elapsed         # int - Minuto
  │   └── extra           # int | null - Tiempo añadido
  ├── team {id, name, logo}
  ├── player {id, name}
  ├── assist {id, name}   # null si no hay
  ├── type                # str - Tipo de evento
  ├── detail              # str - Detalle específico
  └── comments            # str | null
```

**Tipos de Eventos (type):**

| type | detail posibles | Uso en narrativa |
|------|-----------------|------------------|
| "Goal" | "Normal Goal", "Own Goal", "Penalty", "Missed Penalty" | ✅ Crítico |
| "Card" | "Yellow Card", "Red Card", "Second Yellow card" | ✅ Importante |
| "Var" | "Goal cancelled", "Penalty confirmed", etc. | ✅ Disponible |
| "subst" | "Substitution 1", etc. | ⚠️ Bajo valor |

**Ejemplo de evento VAR:**
```json
{
  "type": "Var",
  "detail": "Goal cancelled",
  "comments": "Offside"
}
```

---

### 2.4 `/fixtures/lineups` - Alineaciones

**Estructura Response:**
```
response[0] = Home Team Lineup
response[1] = Away Team Lineup

Cada uno:
  ├── team {id, name, logo, colors}
  ├── formation          # str - "4-3-3"
  ├── coach {id, name, photo}
  ├── startXI[]
  │   └── player {id, name, number, pos, grid}
  └── substitutes[]
      └── player {id, name, number, pos}
```

**Campos por Jugador:**
- `id` - Player external ID
- `name` - Nombre completo
- `number` - Dorsal
- `pos` - Posición ("G", "D", "M", "F")
- `grid` - Posición en cuadrícula ("1:1" para portero)

**Disponibilidad:** ~60 minutos antes del kickoff. Post-partido sigue disponible.

---

### 2.5 `/fixtures/players` - Stats por Jugador (NO USADO)

**Estructura Response:**
```
response[]
  ├── team {id, name, logo}
  └── players[]
      ├── player {id, name, photo}
      └── statistics[]
          └── {games, shots, passes, dribbles, duels, cards, etc.}
```

**Campos Destacados:**
```
statistics[0].games
  ├── minutes          # int
  ├── number           # int - Dorsal
  ├── position         # str - "M", "D", etc.
  ├── rating           # str - "7.3" (0-10 scale)
  └── captain          # bool

statistics[0].shots
  ├── total            # int
  └── on               # int (on target)

statistics[0].passes
  ├── total            # int
  ├── key              # int
  └── accuracy         # str - "85"
```

**⚠️ IMPORTANTE:** Este endpoint contiene `rating` por jugador, pero NO tiene "MVP" o "Player of the Match" explícito. El MVP tendría que inferirse del rating más alto.

---

## 3. Preguntas del Auditor

### ¿Incluye MVP / Player of the Match?
**NO explícito.** API-Football no tiene un campo `mvp` o `player_of_match`.

**Workaround:** Se podría inferir del endpoint `/fixtures/players` tomando el jugador con `rating` más alto, pero esto no es oficial.

### ¿Incluye flags de VAR?
**SÍ.** En `/fixtures/events` con `type: "Var"`.

Ejemplos de `detail`:
- "Goal cancelled"
- "Penalty confirmed"
- "Goal Disallowed - offside"
- "Card upgrade"

### ¿Incluye clima/estado de cancha?
**NO.** API-Football no provee:
- `weather` / `temperature`
- `pitch_condition` / `grass_quality`
- `attendance` (esporádicamente en algunos fixtures)

---

## 4. Recomendaciones para derived_facts

### ✅ Campos Confiables (usar sin reservas)

| Campo | Fuente | Notas |
|-------|--------|-------|
| `home_goals`, `away_goals` | `/fixtures` | Siempre presente |
| `status` (FT/AET/PEN) | `/fixtures` | Siempre presente |
| `ball_possession` | `/statistics` | Formato "55%" |
| `total_shots`, `shots_on_goal` | `/statistics` | Siempre presente |
| `corner_kicks` | `/statistics` | Siempre presente |
| `yellow_cards`, `red_cards` | `/statistics` | Siempre presente |
| Goles (minuto, jugador) | `/events` | Alta calidad |
| Tarjetas (minuto, jugador) | `/events` | Alta calidad |
| `formation` | `/lineups` | Siempre presente |

### ⚠️ Campos Variables (validar presencia)

| Campo | Fuente | Problema |
|-------|--------|----------|
| `expected_goals` (xG) | `/statistics` | No en todas las ligas |
| `passes_accurate` | `/statistics` | A veces null |
| `offsides` | `/statistics` | A veces null |
| VAR events | `/events` | Solo si hubo revisión |
| Player ratings | `/players` | No siempre disponible |

### ❌ Campos NO Disponibles

| Campo | Alternativa |
|-------|-------------|
| MVP / Player of Match | Inferir de rating más alto (no oficial) |
| Weather | No disponible |
| Pitch condition | No disponible |
| Attendance | Esporádico, no confiable |

---

## 5. Ejemplos de Payloads Almacenados

Los payloads ya procesados están en:
- `logs/payloads/payload_70509.json` - Copa del Rey (AET, roja, penales)
- `logs/payloads/payload_6648.json` - Bundesliga (FT normal, con xG)

Estos muestran la estructura POST-procesada que usamos internamente.

---

## 6. Consistencia de Nomenclatura

| API-Football | Nuestro código | Notas |
|--------------|----------------|-------|
| "Ball Possession" | `ball_possession` | Convertimos "55%" → 55.0 |
| "Total Shots" | `total_shots` | - |
| "Shots on Goal" | `shots_on_goal` | - |
| "Corner Kicks" | `corner_kicks` | - |
| "Yellow Cards" | `yellow_cards` | - |
| `time.elapsed` | `minute` | En events |
| `time.extra` | `extra_minute` | Tiempo añadido |

La API usa formato "Title Case" inconsistente. Nuestro código normaliza a `snake_case`.

---

## 7. Conclusión

**API-Football provee datos robustos para:**
- Timeline de eventos (goles, tarjetas, VAR)
- Estadísticas básicas del partido
- Alineaciones y formaciones

**NO provee (y debemos documentar como limitación):**
- MVP / Player of Match oficial
- Clima / Condiciones de cancha
- Asistencia al estadio (inconsistente)
- xG en todas las ligas

**Recomendación:** Para derived_facts, usar solo campos de la sección "Confiables". Los campos "Variables" deben tener fallbacks en el código.
