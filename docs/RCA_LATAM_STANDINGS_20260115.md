# Informe RCA: Standings LATAM (Colombia 239) + Temporadas nuevas

**Fecha**: 2026-01-15  
**Severidad**: P0  
**Síntoma**: Tabla de posiciones inconsistente al inicio de temporada (equipos duplicados / roster incorrecto / season no disponible).

## Contexto técnico (cómo funciona hoy)

- **Fuente**: API-Football endpoint `standings`.
- **Arquitectura DB-first**:
  - L1: cache en memoria (`_standings_cache`, TTL 30m)
  - L2: DB `league_standings` (TTL 6h)
  - L3: provider (solo en `/standings/{league_id}`; en `/matches/{id}/details` no se llama al provider en hot path)
- **Persistencia**: `league_standings.standings` guarda una lista de dicts (JSON/JSONB) con los campos parseados por `APIFootballProvider._parse_standing()`.

## Causas raíz

### 1) Duplicados por “múltiples grupos/fases” en API-Football

API-Football puede devolver **más de una tabla** dentro de `league.standings` (ej.: Clausura + cuadrangulares/playoffs).  
Si se aplana todo sin seleccionar un “grupo principal”, algunos equipos aparecen repetidos.

### 2) Temporada aún no publicada por el proveedor (season 2026)

Antes del kickoff, es normal que API-Football **no tenga standings** para `season=2026` y responda vacío.  
En ese caso el sistema debe:
- no caer en fallback silencioso a la temporada previa (evita confusión)
- responder con un mensaje explícito para que el front muestre “Temporada aún no inicializada”.

## Cambios implementados

### A) Selección determinística del “grupo principal” (anti-duplicados)

**Commit**: `a5ef602`  
**Archivo**: `app/etl/api_football.py`

- Se agregó `group` en `_parse_standing()` para propagar el contexto de fase/grupo.
- Se implementó `_select_primary_standings_group()` y se aplica en `get_standings()`:
  - agrupa por `group`
  - selecciona el grupo “principal” con scoring:
    1) más equipos
    2) mayor suma de partidos jugados
    3) preferencia por nombre (Regular Season > Apertura > Clausura)

**Efecto**: Para Colombia 2025, se elimina el caso de “28 equipos duplicados” (Clausura + cuadrangulares).

### B) `description` capturado y persistido en el JSON

**Archivo**: `app/etl/api_football.py`

El parser `_parse_standing()` incluye `description`, que en ligas europeas indica:
- Champions League / Europa / etc.
- Relegation / Promotion (cuando el proveedor lo expone)

**Nota**: No se creó columna dedicada; se guarda dentro de `league_standings.standings` (JSON) para mantener flexibilidad y evitar migraciones.

### C) Mensaje 404 explícito para temporada no inicializada

**Commit**: `d013c50`  
**Archivo**: `app/main.py`

Cuando `GET /standings/{league_id}?season=YYYY` no tiene datos (provider retorna vacío), se responde:
- `404` con detalle: `"Standings not available yet for season {season}"`

**Efecto**: El front puede mostrar “Temporada aún no inicializada” y no confundir con standings viejos.

### D) Placeholder inteligente con roster correcto

**Commits**: `d9fb2cd`, `fe856c0`, `a81cdad`
**Archivo**: `app/main.py`

Cuando API-Football no tiene standings para una temporada nueva, se genera un placeholder con:

1. **Prioridad 1**: Equipos de fixtures de la nueva temporada (más preciso - refleja roster oficial)
2. **Prioridad 2**: Standings anteriores menos equipos con `description: Relegation`
3. **Prioridad 3**: Matches recientes (fallback)

```python
async def _generate_placeholder_standings(session, league_id, season):
    # Estrategia 1: Fixtures de la nueva temporada
    SELECT DISTINCT t.external_id, t.name, t.logo_url
    FROM teams t
    JOIN matches m ON (t.id = m.home_team_id OR t.id = m.away_team_id)
    WHERE m.league_id = :league_id
      AND EXTRACT(YEAR FROM m.date) = :season
```

**Efecto**: Colombia 2026 muestra los 20 equipos correctos:
- ✅ Incluye Cúcuta, Jaguares (ascendidos)
- ✅ Excluye Envigado, Union Magdalena (descendidos)

## Verificación (ejecutada 2026-01-15)

### Test 1: Season Default
```bash
GET /standings/239
# season=2026, 20 equipos, source=placeholder ✅
```

### Test 2: Sin Duplicados
```bash
GET /standings/239?season=2025
# 20 equipos únicos (sin duplicados) ✅
```

### Test 3: Roster Correcto
```bash
GET /standings/239
# ✅ Cúcuta presente (ascendido)
# ✅ Jaguares presente (ascendido)
# ✅ Envigado ausente (descendido)
# ✅ Union Magdalena ausente (descendido)
```

### Test 4: Liga Europea
```bash
GET /standings/140  # La Liga
# season=2025 en enero 2026 ✅
```

## Recomendación sobre “description como columna”

Hoy `description` **ya está en DB** dentro del JSON. Esto suele ser suficiente porque:
- evita duplicar datos
- no requiere migración
- mantiene compatibilidad con variaciones del proveedor

Si en el futuro queremos queries/alertas a gran escala sin parsear JSON (ej.: “¿cuántos equipos tienen Relegation?” en cientos de ligas), alternativas mejores que una columna suelta:
- crear una tabla derivada/materialized view `league_standings_rows(league_id, season, team_id, position, description, ...)`
- o agregar índices de expresión sobre JSONB (si el tipo en DB es JSONB) para acelerar consultas.

## Commits relacionados

| Hash | Descripción |
|------|-------------|
| `a5ef602` | fix: filter duplicate teams from LATAM leagues with multiple groups |
| `d013c50` | fix: improve 404 message for unavailable seasons |
| `d9fb2cd` | feat: add placeholder fallback for pre-season leagues |
| `6c39aa1` | fix: use truthiness check for empty standings lists |
| `fe856c0` | fix: use previous season teams for placeholder |
| `26a9794` | feat: capture description field for promotion/relegation |
| `a81cdad` | fix: prioritize new season fixtures for placeholder roster |

## Nota de seguridad

Nunca incluir credenciales de DB en texto/commits/logs. Si alguna credencial se compartió fuera del canal seguro, **rotarla** inmediatamente.
