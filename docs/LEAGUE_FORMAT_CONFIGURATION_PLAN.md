# Plan: League Format Configuration System

**Autor**: David (Owner)
**Revisor**: ABE (Auditor Backend)
**Fecha**: 2026-02-05
**Estado**: Pendiente aprobación ABE

---

## Resumen Ejecutivo

Sistema de configuración declarativa para manejar la variabilidad de formatos de liga (~50 ligas activas) con mínima intervención manual. Resuelve problemas inmediatos (Ecuador 32 equipos) y sienta bases para funcionalidades futuras (reclasificación Colombia, zonas/badges).

**Principios de diseño:**
- API-Football como **hint**, no como source of truth
- Configuración declarativa en `rules_json` JSONB
- Heurística automática con override manual
- Patrón SofaScore: standings = temporada regular, playoffs = eventos de calendario

---

## Tabla de Contenidos

1. [Contexto y Problema](#1-contexto-y-problema)
2. [Fase 1: Filtrado de Standings](#2-fase-1-filtrado-de-standings)
3. [Fase 2: Sistema de Zonas/Badges](#3-fase-2-sistema-de-zonasbadges)
4. [Fase 3: Reclasificación Colombia](#4-fase-3-reclasificación-colombia)
5. [Fase 4: Tabla de Descenso](#5-fase-4-tabla-de-descenso)
6. [Fase 5: Extensiones SofaScore-style](#6-fase-5-extensiones-sofascore-style)
7. [Schema rules_json Completo](#7-schema-rules_json-completo)
8. [Criterios de Aceptación](#8-criterios-de-aceptación)
9. [Cronograma Sugerido](#9-cronograma-sugerido)

---

## 1. Contexto y Problema

### 1.1 Problema Principal: Ecuador 32 Equipos

El endpoint `GET /standings/{league_id}` devuelve 32 equipos para Ecuador cuando deberían ser 16.

**Causa raíz**: La tabla `league_standings.standings` (JSON array) contiene múltiples grupos/fases:
- "Serie A 2025" (16 equipos) ← **Este es el correcto**
- "Championship Round" (6 equipos)
- "Qualifying Round" (6 equipos)
- "Relegation Round" (4 equipos)

El backend actual (`_get_standings_from_db()` línea ~331) retorna el array completo sin filtrar.

### 1.2 Variabilidad de Formatos de Liga

| Liga | Grupos en standings | Problema |
|------|---------------------|----------|
| Ecuador (242) | 4 grupos (fases mezcladas) | Duplicados |
| MLS (253) | 2 grupos ("Eastern/Western Conference") | TIE: ambos con 15 equipos |
| Argentina (128) | 3 grupos ("Group A", "Group B", "Promedios 2026") | 30+30+30 |
| Colombia (239) | Solo "Primera Division: Apertura" | OK pero falta reclasificación |
| Premier League (39) | 1 grupo | OK con zonas de API |

### 1.3 Sistema de Zonas en API-Football

Campo `description` presente en algunos standings:
- **Premier League**: "Promotion - Champions League", "Promotion - Europa League", "Relegation - Championship"
- **Colombia**: "Promotion - Primera A (Apertura - Play Offs)" para 8 equipos, **SIN zona de descenso**
- **Ecuador**: NULL (sin zonas definidas)

**Problema**: Inconsistente entre ligas. Requiere override manual para completar.

### 1.4 Colombia: Sistema Especial (Investigación Dimayor)

Según [Reglamento Dimayor 2026](https://dimayor.com.co/wp-content/uploads/2026/01/REGLAMENTO-LIGA-2026-V12-1.pdf):

**3 Tablas Independientes:**

| Tabla | Fuente | Qué Determina |
|-------|--------|---------------|
| **Standings de Fase** | API-Football | Top 8 → Cuadrangulares |
| **Reclasificación** | Calculada (Apertura + Clausura) | Libertadores (#3, #4), Sudamericana (#2-4) |
| **Descenso** | Calculada (promedio 3 años) | 2 equipos bajan |

**Clasificación a torneos internacionales:**
- **Copa Libertadores (4 cupos)**: Campeón Apertura + Campeón Clausura + Top 2 reclasificación (no campeones)
- **Copa Sudamericana (4 cupos)**: Campeón Copa BetPlay + Top 3 reclasificación (no clasificados a Libertadores)

**Fuentes**: [365scores](https://www.365scores.com/es/news/reclasificacion-liga-betplay-2026/), [colombia.com](https://www.colombia.com/futbol/liga-colombiana/reclasificacion)

### 1.5 Insights de SofaScore (via ABE)

SofaScore maneja estos casos con:
- `team.name` (largo), `team.shortName` (corto), `team.nameCode` (3 letras)
- `standings: [...]` donde cada entrada es una tabla independiente
- Metadata: `tournament.isGroup`, `tournament.groupName`
- UI con selector de fase/grupo - NO mezclan tablas

---

## 2. Fase 1: Filtrado de Standings

**Objetivo**: Resolver Ecuador y casos similares con heurística automática.

**Prioridad**: P0 (bloquea uso de standings)

### 2.1 Arquitectura: DB-First (ABE P0)

**Principio**: La base de datos almacena TODOS los grupos/fases aplanados. El filtrado ocurre en el endpoint, NO en la ingesta.

```
┌─────────────────────────────────────────────────────────────────┐
│                         FLUJO DE DATOS                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  API-Football ──► league_standings.standings (JSON array)       │
│                   [Serie A 2025, Championship Round, ...]       │
│                                                                 │
│                              │                                  │
│                              ▼                                  │
│                                                                 │
│  GET /standings/{id} ──► select_standings_view()                │
│                          ├─ Leer rules_json                     │
│                          ├─ Aplicar heurística/override         │
│                          └─ Retornar grupo seleccionado         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Beneficios**:
- Provenance completa (data original intacta)
- Frontend puede pedir cualquier grupo con `?group=`
- Heurística mejorable sin re-ingesta
- Debugging más fácil (ver todos los grupos disponibles)

### 2.2 Función Canónica: `select_standings_view()` (ABE P0)

Toda la lógica de selección en UNA función para evitar duplicación y garantizar consistencia.

```python
def select_standings_view(
    standings: list[dict],
    rules_json: dict,
    requested_group: Optional[str] = None,
) -> StandingsViewResult:
    """
    ÚNICA función canónica para seleccionar vista de standings.

    Args:
        standings: Array completo de league_standings.standings
        rules_json: Config de admin_leagues.rules_json
        requested_group: Query param ?group= (opcional)

    Returns:
        StandingsViewResult con datos filtrados y metadata
    """
    # 1. Agrupar entradas por nombre de grupo
    groups = group_standings_by_name(standings)
    available_groups = list(groups.keys())

    # 2. Si hay query param, validar y usar
    if requested_group:
        if requested_group not in groups:
            raise StandingsGroupNotFound(
                requested=requested_group,
                available=available_groups,
            )
        return StandingsViewResult(
            standings=groups[requested_group],
            selected_group=requested_group,
            selection_reason="query_param",
            available_groups=available_groups,
            tie_warning=None,
        )

    # 3. Aplicar heurística
    selected_group, reason = select_default_standings_group(
        groups=groups,
        rules_json=rules_json,
    )

    # 4. Detectar TIE para warning
    tie_warning = detect_standings_tie(groups)

    return StandingsViewResult(
        standings=groups[selected_group],
        selected_group=selected_group,
        selection_reason=reason,
        available_groups=available_groups,
        tie_warning=tie_warning,
    )


@dataclass
class StandingsViewResult:
    standings: list[dict]
    selected_group: str
    selection_reason: str  # "query_param", "config_override", "heuristic_*"
    available_groups: list[str]
    tie_warning: Optional[list[str]]


class StandingsGroupNotFound(Exception):
    """Raised when requested group doesn't exist."""
    def __init__(self, requested: str, available: list[str]):
        self.requested = requested
        self.available = available
        super().__init__(f"Group '{requested}' not found. Available: {available}")
```

### 2.3 Heurística de Selección de Grupo

**Ajustes ABE P0 + Kimi**:
- Usar `team_count` como hint (ABE): Si config tiene `team_count`, preferir grupo que coincida
- Whitelist de patrones válidos (Kimi): Evita confusión con "Promotion Playoff"
- Keywords adicionales (ABE P1): "promedios", "reclasificacion"

```python
def select_default_standings_group(
    groups: dict[str, list],
    rules_json: dict,
) -> tuple[str, str]:
    """
    Selecciona el grupo por defecto para mostrar en standings.

    Args:
        groups: Dict {group_name: [entries]}
        rules_json: Config de admin_leagues

    Returns:
        Tuple (selected_group_name, selection_reason)

    Prioridades:
    1. Override manual en rules_json.standings.default_group
    2. Whitelist de patrones válidos (si configurada)
    3. Preferir grupo con team_count == config.team_count (hint)
    4. Preferir "Overall" si existe
    5. MAX(team_count) excluyendo grupos con keywords de playoffs
    """
    standings_config = rules_json.get("standings", {})

    # 1. Override manual
    if standings_config.get("default_group"):
        return (standings_config["default_group"], "config_override")

    # 2. Whitelist de patrones válidos (Ajuste Kimi)
    valid_patterns = standings_config.get("valid_group_patterns")
    if valid_patterns:
        for name in groups:
            if any(pattern.lower() in name.lower() for pattern in valid_patterns):
                return (name, "heuristic_whitelist")

    # Filtrar grupos de playoffs/fases finales (Blacklist)
    # ABE P1: Agregados "promedios", "reclasificacion"
    PLAYOFF_KEYWORDS = [
        "playoff", "play-off", "final", "semifinal", "quarter",
        "championship round", "relegation round", "qualifying round",
        "cuadrangular", "octavos", "liguilla", "knockout",
        "promotion playoff", "relegation playoff",
        "promedios", "reclasificacion",  # ABE P1: Tablas auxiliares
    ]

    def is_playoff_group(name: str) -> bool:
        return any(kw in name.lower() for kw in PLAYOFF_KEYWORDS)

    # Candidatos = grupos que NO son playoffs
    candidates = {
        name: entries for name, entries in groups.items()
        if not is_playoff_group(name)
    }

    if not candidates:
        candidates = groups  # Fallback: todos

    # 3. Preferir grupo con team_count == config hint (ABE P0)
    expected_team_count = standings_config.get("team_count")
    if expected_team_count:
        for name, entries in candidates.items():
            if len(entries) == expected_team_count:
                return (name, "heuristic_team_count_match")

    # 4. Preferir "Overall" si existe
    for name in candidates:
        if "overall" in name.lower():
            return (name, "heuristic_overall")

    # 5. MAX team count
    max_group = max(candidates.items(), key=lambda x: len(x[1]))

    return (max_group[0], "heuristic_max_teams")
```

### 2.4 Detección de TIE

```python
def detect_standings_tie(groups: dict[str, list]) -> Optional[list[str]]:
    """
    Detecta si hay múltiples grupos con igual cantidad de equipos (MAX).
    Retorna lista de grupos en TIE o None si no hay TIE.

    Args:
        groups: Dict {group_name: [entries]} (ya agrupado)

    Usado para: Log warning + requerir config manual.
    """
    if not groups:
        return None

    counts = [(name, len(entries)) for name, entries in groups.items()]
    max_count = max(c[1] for c in counts)

    tied = [name for name, count in counts if count == max_count]

    if len(tied) > 1:
        return tied
    return None
```

### 2.3 Response Shape Actualizado (ABE P0: Backwards Compatible)

**CRÍTICO ABE**: No romper el shape actual. Agregar `meta` como campo adicional, manteniendo campos existentes.

```json
{
  "league_id": 242,
  "season": 2026,
  "standings": [...],
  "source": "db_cache",
  "is_placeholder": false,
  "is_calculated": false,
  "meta": {
    "available_groups": ["Serie A 2025", "Championship Round", "Qualifying Round", "Relegation Round"],
    "selected_group": "Serie A 2025",
    "selection_reason": "heuristic_max_teams",
    "tie_warning": null
  }
}
```

### 2.6 Query Param para Override (ABE P0: Validación)

```
GET /standings/242?group=Championship%20Round
```

Retorna solo el grupo especificado con `selection_reason: "query_param"`.

**ABE P0: Comportamiento de validación**:

| Caso | HTTP Status | Response |
|------|-------------|----------|
| `?group=` válido | 200 OK | Standings del grupo |
| `?group=` inválido | 404 Not Found | Error con `available_groups` |
| Sin `?group=` | 200 OK | Grupo por heurística |

**Response de error (404)**:
```json
{
  "detail": "Group 'Fase Final' not found",
  "available_groups": ["Serie A 2025", "Championship Round", "Qualifying Round", "Relegation Round"]
}
```

**Headers adicionales** (para clientes que prefieran headers):
```
X-Available-Groups: Serie A 2025,Championship Round,Qualifying Round,Relegation Round
```

**Implementación en endpoint**:
```python
@router.get("/standings/{league_id}")
async def get_standings(
    league_id: int,
    season: Optional[int] = None,
    group: Optional[str] = None,
):
    """Get league standings with optional group filter."""
    try:
        result = select_standings_view(
            standings=raw_standings,
            rules_json=rules_json,
            requested_group=group,
        )
    except StandingsGroupNotFound as e:
        # ABE nit: available_groups en BODY y HEADER para flexibilidad
        raise HTTPException(
            status_code=404,
            detail={
                "message": f"Group '{e.requested}' not found",
                "available_groups": e.available,
            },
            headers={"X-Available-Groups": ",".join(e.available)},
        )

    return {
        "league_id": league_id,
        "season": season,
        "standings": result.standings,
        "source": "db_cache",
        "is_placeholder": False,
        "is_calculated": False,
        "meta": {
            "available_groups": result.available_groups,
            "selected_group": result.selected_group,
            "selection_reason": result.selection_reason,
            "tie_warning": result.tie_warning,
        },
    }
```

### 2.7 Archivos a Modificar (Fase 1)

| Archivo | Acción | Descripción |
|---------|--------|-------------|
| `app/main.py` | Modificar | Actualizar endpoint standings con filtrado |
| `app/utils/standings.py` | Crear | Funciones canónicas `select_standings_view()`, `select_default_standings_group()`, `detect_standings_tie()` |

### 2.8 Criterios de Aceptación (Fase 1)

**Funcionalidad**:
- [ ] Ecuador devuelve 16 equipos (no 32)
- [ ] `meta.available_groups` lista todos los grupos
- [ ] `meta.selected_group` indica grupo mostrado
- [ ] `meta.selection_reason` indica razón ("heuristic_max_teams", "query_param", "config_override", etc.)
- [ ] MLS con TIE genera warning en logs
- [ ] Query param `?group=X` funciona para grupos válidos
- [ ] Ligas sin config funcionan con heurística por defecto

**ABE P0 (Obligatorios)**:
- [ ] Response mantiene backwards compatibility (campos existentes intactos)
- [ ] `?group=` inválido retorna 404 con `available_groups`
- [ ] `team_count` hint tiene prioridad en heurística
- [ ] Función canónica `select_standings_view()` usada en endpoint
- [ ] SQL de migración usa merge parcial (`||` o `jsonb_set`)

**Observabilidad (ABE P1)**:
- [ ] Log WARNING cuando se detecta TIE entre grupos
- [ ] Log INFO con grupo seleccionado y razón
- [ ] Log DEBUG con todos los grupos disponibles

---

## 3. Fase 2: Sistema de Zonas/Badges

**Objetivo**: Mostrar badges de clasificación/descenso en standings.

**Prioridad**: P1 (mejora UX significativa)

### 3.1 Fuentes de Zonas

1. **API-Football** (`description` field) - cuando disponible
2. **Override manual** (`rules_json.zones.overrides`) - cuando API incompleta/incorrecta

### 3.2 Función `apply_zones()`

```python
def apply_zones(standings: list, zones_config: dict) -> list:
    """
    Apply zone information to standings entries.

    Priority:
    1. Manual overrides from zones_config.overrides
    2. API-Football description field
    """
    if not zones_config.get("enabled", True):
        return standings

    overrides = zones_config.get("overrides", {})

    for entry in standings:
        pos = entry.get("position") or entry.get("rank")

        # Check manual override first
        zone = None
        for range_str, zone_config in overrides.items():
            if "-" in range_str:
                start, end = map(int, range_str.split("-"))
                if start <= pos <= end:
                    zone = zone_config
                    break
            elif int(range_str) == pos:
                zone = zone_config
                break

        # Fallback to API description
        if not zone and entry.get("description"):
            zone = parse_api_zone_description(entry["description"])

        entry["zone"] = zone

    return standings


def parse_api_zone_description(description: str) -> Optional[dict]:
    """Parse API-Football zone description to structured format."""
    desc_lower = description.lower()

    ZONE_MAPPINGS = [
        ("champions league", {"type": "promotion", "tournament": "Champions League", "style": "blue"}),
        ("europa league", {"type": "promotion", "tournament": "Europa League", "style": "orange"}),
        ("conference league", {"type": "promotion", "tournament": "Conference League", "style": "green"}),
        ("libertadores", {"type": "promotion", "tournament": "Copa Libertadores", "style": "blue"}),
        ("sudamericana", {"type": "promotion", "tournament": "Copa Sudamericana", "style": "orange"}),
        ("play", {"type": "playoff", "style": "cyan"}),  # play-off, playoff
        ("relegation", {"type": "relegation", "style": "red"}),
        ("descenso", {"type": "relegation", "style": "red"}),
    ]

    for keyword, zone in ZONE_MAPPINGS:
        if keyword in desc_lower:
            return zone

    return {"type": "other", "description": description, "style": "gray"}
```

### 3.3 Response Shape con Zonas

```json
{
  "meta": {
    "league_id": 239,
    "zones_source": "hybrid"
  },
  "standings": [
    {
      "position": 1,
      "team_id": 1234,
      "team_name": "Atlético Nacional",
      "zone": {
        "type": "playoff",
        "description": "Clasifica a cuadrangulares",
        "style": "cyan"
      }
    },
    {
      "position": 19,
      "team_id": 5678,
      "team_name": "Deportivo Cali",
      "zone": {
        "type": "relegation",
        "description": "Zona de descenso",
        "style": "red"
      }
    }
  ]
}
```

### 3.4 Configuración de Zonas por Liga

```json
{
  "zones": {
    "enabled": true,
    "source": "hybrid",
    "overrides": {
      "1-4": {"type": "promotion", "tournament": "Copa Libertadores", "style": "blue"},
      "5-8": {"type": "playoff", "description": "Clasifica a cuadrangulares", "style": "cyan"},
      "19-20": {"type": "relegation", "description": "Descenso por promedio", "style": "red"}
    }
  }
}
```

### 3.5 Archivos a Modificar (Fase 2)

| Archivo | Acción | Descripción |
|---------|--------|-------------|
| `app/utils/standings.py` | Modificar | Agregar `apply_zones()`, `parse_api_zone_description()` |
| `app/main.py` | Modificar | Integrar zonas en response |
| `migrations/045_standings_zones.sql` | Crear | Config zonas para ligas prioritarias |

### 3.6 Criterios de Aceptación (Fase 2)

- [ ] Colombia muestra zonas de playoffs (top 8)
- [ ] Premier League usa zonas de API-Football
- [ ] `rules_json.zones.overrides` tiene prioridad sobre API
- [ ] `meta.zones_source` indica origen ("api", "manual", "hybrid")
- [ ] Styles soportados: blue, orange, green, cyan, red, gray

---

## 4. Fase 3: Reclasificación Colombia

**Objetivo**: Calcular y mostrar tabla de reclasificación (acumulado anual).

**Prioridad**: P2 (feature específico Colombia)

### 4.1 Lógica de Cálculo

```python
async def calculate_reclasificacion(
    session: AsyncSession,
    league_id: int,
    year: int,
) -> list[dict]:
    """
    Calcula tabla de reclasificación sumando Apertura + Clausura.

    Requiere: standings de ambos torneos del año.
    """
    # Obtener standings de Apertura (Liga BetPlay I)
    apertura = await get_standings_for_tournament(session, league_id, year, phase="apertura")

    # Obtener standings de Clausura (Liga BetPlay II)
    clausura = await get_standings_for_tournament(session, league_id, year, phase="clausura")

    if not apertura or not clausura:
        return []  # No se puede calcular sin ambos

    # Combinar por team_id
    combined = {}
    for entry in apertura + clausura:
        team_id = entry["team_id"]
        if team_id not in combined:
            combined[team_id] = {
                "team_id": team_id,
                "team_name": entry["team_name"],
                "points": 0,
                "played": 0,
                "won": 0,
                "drawn": 0,
                "lost": 0,
                "goals_for": 0,
                "goals_against": 0,
            }

        for field in ["points", "played", "won", "drawn", "lost", "goals_for", "goals_against"]:
            combined[team_id][field] += entry.get(field, 0)

    # Calcular goal_diff y ordenar
    result = list(combined.values())
    for entry in result:
        entry["goal_diff"] = entry["goals_for"] - entry["goals_against"]

    # Ordenar: puntos DESC, goal_diff DESC, goals_for DESC
    result.sort(key=lambda x: (-x["points"], -x["goal_diff"], -x["goals_for"]))

    # Asignar posiciones
    for i, entry in enumerate(result, 1):
        entry["position"] = i

    return result
```

### 4.2 Response Shape

```json
{
  "meta": {
    "league_id": 239,
    "season": 2026
  },
  "standings": [...],
  "reclasificacion": {
    "enabled": true,
    "source": "calculated",
    "data": [
      {
        "position": 1,
        "team_id": 1234,
        "team_name": "Atlético Nacional",
        "points": 68,
        "played": 40,
        "zone": {
          "type": "promotion",
          "tournament": "Copa Libertadores",
          "style": "blue"
        }
      }
    ],
    "zones": {
      "1-2": {"type": "promotion", "tournament": "Copa Libertadores (si no campeón)", "style": "blue"},
      "3-4": {"type": "promotion", "tournament": "Copa Sudamericana", "style": "orange"}
    }
  }
}
```

### 4.3 Dependencias

- Requiere standings históricos de Apertura y Clausura del mismo año
- Requiere mapeo de `league_id` a fases (Apertura=I, Clausura=II)
- Solo aplica a ligas con formato `apertura_clausura`

### 4.4 Dependencia Crítica (Kimi)

**Verificar consistencia de `team_id`**: Reclasificación requiere que standings históricos usen el mismo `team_id` a lo largo del tiempo.

**Validación pre-implementación**:
```sql
-- Detectar posibles duplicados de equipos
SELECT name, COUNT(DISTINCT id) as ids
FROM teams
WHERE country = 'Colombia'
GROUP BY name
HAVING COUNT(DISTINCT id) > 1;

-- Verificar que los team_id en standings corresponden a equipos actuales
SELECT DISTINCT s.team_id, t.name, t.id
FROM league_standings ls,
     jsonb_array_elements(ls.standings::jsonb) s,
     teams t
WHERE ls.league_id = 239
  AND (s->>'team_id')::int = t.id
ORDER BY t.name;
```

Si hay duplicados (ej: "América de Cali" vs "América de Cali 2024"), resolver con merge/alias antes de implementar Fase 3.

### 4.4 Criterios de Aceptación (Fase 3)

- [ ] Reclasificación se calcula correctamente sumando Apertura + Clausura
- [ ] Solo se muestra para ligas con `rules_json.reclasificacion.enabled = true`
- [ ] Zonas de reclasificación muestran clasificación a Libertadores/Sudamericana
- [ ] Si falta un torneo (ej: Clausura no empezó), devuelve `null`

---

## 5. Fase 4: Tabla de Descenso

**Objetivo**: Calcular tabla de descenso por promedio (3 años).

**Prioridad**: P2 (feature específico Colombia y ligas similares)

### 5.1 Lógica de Cálculo

```python
async def calculate_descenso_promedio(
    session: AsyncSession,
    league_id: int,
    current_year: int,
    years_back: int = 3,
) -> list[dict]:
    """
    Calcula tabla de descenso por promedio.

    Fórmula: Puntos últimos N años ÷ Partidos jugados

    Criterios de desempate (Colombia Dimayor):
    1. Mayor puntos sumados de Fase I Liga I + Fase I Liga II del año actual
    2. Diferencia de goles superior
    3. Más goles anotados
    4. Más goles de visitante
    5. Menos goles recibidos de visitante
    6. Más partidos ganados
    7. Sorteo Dimayor
    """
    # Obtener standings de últimos N años
    all_standings = []
    for year in range(current_year - years_back + 1, current_year + 1):
        year_standings = await get_all_standings_for_year(session, league_id, year)
        all_standings.extend(year_standings)

    # Acumular por equipo
    teams = {}
    for entry in all_standings:
        team_id = entry["team_id"]
        if team_id not in teams:
            teams[team_id] = {
                "team_id": team_id,
                "team_name": entry["team_name"],
                "total_points": 0,
                "total_played": 0,
                "goal_diff": 0,
                "goals_for": 0,
                "goals_against": 0,
                "won": 0,
            }

        teams[team_id]["total_points"] += entry.get("points", 0)
        teams[team_id]["total_played"] += entry.get("played", 0)
        teams[team_id]["goal_diff"] += entry.get("goal_diff", 0)
        teams[team_id]["goals_for"] += entry.get("goals_for", 0)
        teams[team_id]["goals_against"] += entry.get("goals_against", 0)
        teams[team_id]["won"] += entry.get("won", 0)

    # Calcular promedio
    result = []
    for team in teams.values():
        played = team["total_played"]
        avg = team["total_points"] / played if played > 0 else 0
        result.append({
            **team,
            "average": round(avg, 4),
        })

    # Ordenar por promedio ASC (peor promedio = más abajo)
    result.sort(key=lambda x: (x["average"], x["goal_diff"], x["goals_for"]))

    # Asignar posiciones (1 = peor, último = mejor)
    for i, entry in enumerate(result, 1):
        entry["position"] = i

    return result
```

### 5.2 Response Shape

```json
{
  "meta": {
    "league_id": 239
  },
  "standings": [...],
  "reclasificacion": {...},
  "descenso": {
    "enabled": true,
    "method": "average_3y",
    "count": 2,
    "data": [
      {
        "position": 1,
        "team_id": 5678,
        "team_name": "Deportivo Cali",
        "average": 1.0641,
        "total_points": 98,
        "total_played": 92,
        "zone": {
          "type": "relegation",
          "description": "Desciende a Segunda División",
          "style": "red"
        }
      }
    ]
  }
}
```

### 5.3 Edge Case: Equipos Ascendidos (Kimi)

Equipos recién ascendidos tienen 0 puntos en años anteriores (no existen en historial). El cálculo debe manejar esto explícitamente:

```python
def calculate_average_with_new_teams(team_id: int, history: dict) -> float:
    """
    Calcula promedio manejando equipos nuevos/ascendidos.

    - Si equipo NO tiene historial: promedio = 0.0000 (peor posible)
    - Si equipo tiene historial parcial: solo cuenta años con data
    """
    total_points = history.get(team_id, {}).get("total_points", 0)
    total_played = history.get(team_id, {}).get("total_played", 0)

    if total_played == 0:
        # Equipo nuevo/ascendido: promedio 0 (fondo de la tabla)
        return 0.0

    return total_points / total_played
```

**Comportamiento esperado**:
- Cúcuta Deportivo (ascendido 2026): promedio = 0.0000
- Jaguares de Córdoba (ascendido 2026): promedio = 0.0000
- Deportivo Cali (3 años historial): promedio = 1.0641

### 5.4 Criterios de Aceptación (Fase 4)

- [ ] Promedio se calcula correctamente (puntos ÷ partidos)
- [ ] Considera 3 años de historial
- [ ] Equipos ascendidos sin historial = promedio 0.0 (Kimi)
- [ ] Equipos con historial parcial solo cuentan años con data
- [ ] Solo se muestra para ligas con `rules_json.relegation.method = "average_3y"`

---

## 6. Fase 5: Extensiones SofaScore-style

**Objetivo**: Mejoras de UX inspiradas en SofaScore.

**Prioridad**: P3 (nice-to-have)

### 6.1 `nameCode` (Código 3 letras)

```sql
ALTER TABLE team_wikidata_enrichment
ADD COLUMN name_code VARCHAR(5);

-- Derivar de short_name o manual
-- Ej: "América" → "AME", "Nacional" → "NAC"
```

Uso en UI: Scoreboards compactos, móvil, headers de tabla.

### 6.2 Metadata de Grupos Enriquecida

```json
{
  "meta": {
    "available_tables": [
      {
        "group": "Serie A 2025",
        "team_count": 16,
        "type": "regular",
        "is_current": true
      },
      {
        "group": "Championship Round",
        "team_count": 6,
        "type": "playoff",
        "is_current": false
      }
    ]
  }
}
```

### 6.3 Flag `is_group_stage`

Para ligas con grupos reales (Champions League, Libertadores fase grupos):

```json
{
  "rules_json": {
    "standings": {
      "is_group_stage": true,
      "group_names": ["Group A", "Group B", "Group C", "Group D"]
    }
  }
}
```

### 6.4 Criterios de Aceptación (Fase 5)

- [ ] `name_code` disponible en `team_wikidata_enrichment`
- [ ] `meta.available_tables` con metadata por grupo
- [ ] `is_group_stage` flag para Champions/Libertadores

---

## 7. Schema rules_json Completo

```json
{
  "version": 1,
  "standings": {
    "default_group": null,
    "team_count": 20,
    "format": "apertura_clausura",
    "is_group_stage": false,
    "group_names": null,
    "valid_group_patterns": null
  },
  "zones": {
    "enabled": true,
    "source": "hybrid",
    "overrides": {
      "1-4": {"type": "promotion", "tournament": "Copa Libertadores", "style": "blue"},
      "5-8": {"type": "playoff", "description": "Clasifica a cuadrangulares", "style": "cyan"},
      "19-20": {"type": "relegation", "description": "Zona de descenso", "style": "red"}
    }
  },
  "relegation": {
    "enabled": true,
    "count": 2,
    "method": "average_3y"
  },
  "reclasificacion": {
    "enabled": true,
    "source": "calculated",
    "libertadores_spots": 4,
    "sudamericana_spots": 4
  },
  "international": {
    "libertadores": {"direct": 2, "playoff": 2},
    "sudamericana": {"direct": 4},
    "copa_centroamericana": null
  }
}
```

---

## 8. Criterios de Aceptación

### 8.1 Por Fase

| Fase | Criterios Bloqueantes |
|------|----------------------|
| **Fase 1** | Ecuador 16 equipos, `?group=` funciona, heurística automática |
| **Fase 2** | Zonas visibles, override funciona, styles correctos |
| **Fase 3** | Reclasificación calculada, solo cuando aplica |
| **Fase 4** | Descenso por promedio, 3 años historial |
| **Fase 5** | nameCode, metadata enriquecida |

### 8.2 Observabilidad

- [ ] Log warning cuando se detecta TIE
- [ ] Log info con grupo seleccionado y razón
- [ ] Log error si falta data para reclasificación/descenso

### 8.3 Backwards Compatibility

- [ ] Response sin `meta` sigue funcionando (graceful degradation)
- [ ] iOS client ignora campos nuevos que no conoce
- [ ] Ligas sin config funcionan con defaults

---

## 9. Cronograma Sugerido

| Fase | Descripción | Dependencias | Estimación |
|------|-------------|--------------|------------|
| **Fase 1** | Filtrado de Standings | Ninguna | Sprint actual |
| **Fase 2** | Zonas/Badges | Fase 1 | Sprint +1 |
| **Fase 3** | Reclasificación | Fase 2, Standings históricos | Sprint +2 |
| **Fase 4** | Descenso por promedio | Fase 3, 3 años de data | Sprint +3 |
| **Fase 5** | Extensiones SofaScore | Fase 1-2 | Backlog |

---

## 10. Configuraciones Iniciales por Liga

### 10.0 SQL Merge Parcial (ABE P0)

**CRÍTICO**: Usar `jsonb_set()` para merge parcial, NO reemplazo completo. Preserva campos existentes.

```sql
-- CORRECTO: Merge parcial con jsonb_set
UPDATE admin_leagues
SET rules_json = jsonb_set(
    jsonb_set(
        COALESCE(rules_json, '{}'::jsonb),
        '{standings}',
        '{"format": "split", "team_count": 16}'::jsonb,
        true
    ),
    '{zones}',
    '{"enabled": true, "source": "manual"}'::jsonb,
    true
)
WHERE league_id = 242;

-- INCORRECTO: Reemplazo completo (pierde campos existentes)
-- UPDATE admin_leagues SET rules_json = '{"standings": {...}}'::jsonb WHERE ...
```

**Helper function recomendada** (para scripts de migración):
```sql
CREATE OR REPLACE FUNCTION merge_rules_json(
    league_id_param INT,
    new_config JSONB
) RETURNS VOID AS $$
BEGIN
    UPDATE admin_leagues
    SET rules_json = COALESCE(rules_json, '{}'::jsonb) || new_config
    WHERE league_id = league_id_param;
END;
$$ LANGUAGE plpgsql;

-- Uso:
SELECT merge_rules_json(242, '{"standings": {"team_count": 16}}'::jsonb);
```

### 10.1 Ecuador (242) - Heurística automática

```sql
-- Merge parcial (preserva campos existentes)
UPDATE admin_leagues
SET rules_json = COALESCE(rules_json, '{}'::jsonb) || '{
  "version": 1,
  "standings": {
    "format": "split",
    "team_count": 16,
    "valid_group_patterns": ["Serie A"]
  },
  "zones": {
    "enabled": true,
    "source": "manual",
    "overrides": {
      "1-6": {"type": "playoff", "description": "Championship Round", "style": "cyan"},
      "7-12": {"type": "playoff", "description": "Qualifying Round", "style": "gray"},
      "13-16": {"type": "relegation", "description": "Relegation Round", "style": "red"}
    }
  },
  "relegation": {
    "enabled": true,
    "count": 2,
    "method": "direct"
  }
}'::jsonb
WHERE league_id = 242;
```

### 10.2 Colombia (239) - Config completa

```sql
-- Merge parcial
UPDATE admin_leagues
SET rules_json = COALESCE(rules_json, '{}'::jsonb) || '{
  "version": 1,
  "standings": {
    "format": "apertura_clausura",
    "team_count": 20,
    "valid_group_patterns": ["Primera Division"]
  },
  "zones": {
    "enabled": true,
    "source": "hybrid",
    "overrides": {
      "1-8": {"type": "playoff", "description": "Clasifica a cuadrangulares", "style": "cyan"}
    }
  },
  "relegation": {
    "enabled": true,
    "count": 2,
    "method": "average_3y"
  },
  "reclasificacion": {
    "enabled": true,
    "source": "calculated"
  }
}'::jsonb
WHERE league_id = 239;
```

### 10.3 MLS (253) - Requiere default_group manual

```sql
-- Merge parcial
UPDATE admin_leagues
SET rules_json = COALESCE(rules_json, '{}'::jsonb) || '{
  "version": 1,
  "standings": {
    "default_group": "Overall",
    "format": "single",
    "team_count": 30
  },
  "zones": {
    "enabled": true,
    "source": "api"
  },
  "relegation": {
    "enabled": false
  }
}'::jsonb
WHERE league_id = 253;
```

### 10.4 Premier League (39) - Zonas de API

```sql
-- Merge parcial
UPDATE admin_leagues
SET rules_json = COALESCE(rules_json, '{}'::jsonb) || '{
  "version": 1,
  "standings": {
    "format": "single",
    "team_count": 20
  },
  "zones": {
    "enabled": true,
    "source": "api"
  },
  "relegation": {
    "enabled": true,
    "count": 3,
    "method": "direct"
  }
}'::jsonb
WHERE league_id = 39;
```

---

## 11. Fuera de Scope

1. **UI de configuración**: Dashboard para editar `rules_json` (backlog)
2. **Validación JSON schema**: Validar `rules_json` en PATCH (backlog)
3. **Multi-temporada**: Comparar standings entre temporadas (backlog)
4. **Scraping SofaScore**: API no oficial con anti-scraping (descartado)
5. **Playoffs como eventos**: Brackets/llaves (feature separado)

---

## 12. Apéndice: Archivos a Crear/Modificar

| Archivo | Fase | Acción | Descripción |
|---------|------|--------|-------------|
| `app/utils/standings.py` | 1 | Crear | Heurística, TIE detection |
| `app/main.py` | 1-2 | Modificar | Endpoint standings |
| `migrations/045_standings_zones.sql` | 2 | Crear | Config inicial ligas |
| `app/utils/reclasificacion.py` | 3 | Crear | Cálculo reclasificación |
| `app/utils/descenso.py` | 4 | Crear | Cálculo promedio descenso |
| `team_wikidata_enrichment` | 5 | Modificar | Agregar `name_code` |

---

## 13. Auditorías

### 13.1 Kimi (2026-02-05)

**Veredicto**: APROBADO con comentarios menores

**Ajustes incorporados**:
- [x] Fase 1: Whitelist de grupos válidos además de blacklist de playoffs
- [x] Fase 3: Validación pre-implementación de consistencia `team_id`
- [x] Fase 4: Edge case equipos ascendidos (promedio 0 explícito)
- [x] Schema: `version: 1` agregado al root de `rules_json`

**Prioridades confirmadas**: Fase 1 = P0 legítimo. Fases 3-4 = P2 (Colombia-specific).

**Nota operativa (Kimi)**: Asegurar que el job de standings históricos tenga idempotencia (mismo `team_id` a través de temporadas) antes de activar Fase 3 en producción.

**GO para Fase 1.**

### 13.2 ABE (2026-02-05)

**Veredicto**: APROBADO con P0 obligatorios

**P0 Incorporados (Obligatorios)**:
- [x] **Backwards compatible**: Response mantiene shape existente, `meta` como campo adicional
- [x] **DB-first approach**: Persistir todos los grupos, filtrar en endpoint (no en ingesta)
- [x] **Función canónica**: `select_standings_view()` como única fuente de verdad
- [x] **team_count como hint**: Heurística prioriza grupo que coincida con config
- [x] **SQL merge parcial**: Usar `jsonb_set()` o `||` en lugar de reemplazo completo
- [x] **Validación ?group=**: 404 Not Found con `available_groups` si grupo no existe
- [x] **Keywords adicionales**: "promedios", "reclasificacion" en blacklist

**P1 Incorporados (Recomendados)**:
- [x] **Observabilidad**: Log warning en TIE, log info con selección y razón

**Tests sugeridos por ABE** (para implementación):
```python
def test_ecuador_returns_16_teams():
    """Ecuador debe retornar 16 equipos (Serie A), no 32."""
    ...

def test_mls_tie_warning():
    """MLS con Eastern/Western debe generar tie_warning."""
    ...

def test_invalid_group_returns_404():
    """?group=Inexistente debe retornar 404 con available_groups."""
    ...

def test_team_count_hint_priority():
    """Grupo con team_count == config.team_count tiene prioridad."""
    ...
```

**GO para Fase 1.**

**Nits corregidos (ABE)**:
- [x] `detect_standings_tie()` recibe `groups: dict` (alineado con uso en `select_standings_view`)
- [x] 404 de `?group=` incluye `available_groups` en body Y header

**Nota de seguridad (ABE P0)**:
- ⚠️ `scripts/backfill_ecuador.py` tiene DATABASE_URL hardcodeado - mover a env var o eliminar antes de push

---

**Estado**: APROBADO por Kimi y ABE. Pendiente visto bueno del Owner para comenzar implementación.
