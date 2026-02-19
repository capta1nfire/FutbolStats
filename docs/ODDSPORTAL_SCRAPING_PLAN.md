# Plan de Implementación: OddsPortal Scraping Pipeline

> **Fecha**: 2026-02-10
> **Objetivo**: Llenar huecos de odds históricas (2020-2026) para 15+ ligas usando OddsPortal vía OddsHarvester
> **Prioridad**: P0 — odds son feature #2 en importancia (SHAP) después de goal_diff_avg
> **ABE Review**: 2026-02-10 — 5 gaps P0 cerrados (ver sección 13)

---

## 1. Estado Actual de Cobertura de Odds

### Ligas con 0% de opening_odds (TIER 3 — gap total)

| league_id | Liga | Matches (2020+) | Fuente FDUK | OddsPortal |
|-----------|------|-----------------|-------------|------------|
| 239 | Colombia Primera A | 2,506 | NO | SI |
| 265 | Chile Primera División | 1,577 | NO | SI |
| 281 | Perú Liga 1 | 2,077 | NO | SI |
| 299 | Venezuela Primera División | 1,812 | NO | SI |
| 242 | Ecuador Liga Pro | 1,522 | NO | SI |
| 250 | Paraguay Apertura | 786 | NO | SI |
| 252 | Paraguay Clausura | 691 | NO | SI |
| 268 | Uruguay Apertura | 964 | NO | SI |
| 270 | Uruguay Clausura | 727 | NO | SI |
| 344 | Bolivia Primera División | 1,630 | NO | SI |
| 2 | Champions League | 1,378 | NO | SI |
| 3 | Europa League | 1,453 | NO | SI |
| 848 | Conference League | 2,044 | NO | SI |

**Subtotal TIER 3**: ~19,167 matches sin odds

### Ligas con cobertura parcial (TIER 2 — gap 2020-2022)

| league_id | Liga | Total | Con odds | Gap |
|-----------|------|-------|----------|-----|
| 128 | Argentina | 2,962 | 1,301 (43.9%) | 1,661 |
| 71 | Brasil Serie A | 2,310 | 1,159 (50.2%) | 1,151 |
| 262 | México Liga MX | 2,029 | 879 (43.3%) | 1,150 |
| 253 | MLS | 2,886 | 1,561 (54.1%) | 1,325 |

**Subtotal TIER 2**: ~5,287 matches sin odds

### Total gap: ~24,454 matches

---

## 2. Herramienta: OddsHarvester

**Repo**: `jordantete/OddsHarvester` (GitHub, Python, Playwright)
**Licencia**: Open source
**Motor**: Playwright (browser headless) → scraping de OddsPortal.com

### Output JSON por partido

```json
{
  "match_date": "2025-06-10 20:00:00",
  "home_team": "Arsenal",
  "away_team": "Chelsea",
  "home_score": 2,
  "away_score": 1,
  "1x2_market": [
    {
      "bookmaker_name": "bet365",
      "1": "1.90",
      "X": "3.50",
      "2": "4.20",
      "odds_history_data": [...]
    },
    {
      "bookmaker_name": "Pinnacle",
      "1": "1.88",
      "X": "3.60",
      "2": "4.10"
    }
  ]
}
```

### CLI

```bash
# Histórico: una temporada, una liga, mercado 1X2
oddsharvester historic \
  -s football \
  -l colombia-primera-a \
  --season 2024 \
  -m 1x2 \
  --headless \
  -f json \
  -o /tmp/odds_colombia_2024.json
```

### Ligas ya disponibles en OddsHarvester

| Constante | OddsPortal URL |
|-----------|---------------|
| `argentina-liga-profesional` | /football/argentina/liga-profesional/ |
| `brazil-serie-a` | /football/brazil/serie-a |
| `colombia-primera-a` | /football/colombia/primera-a/ |
| `usa-mls` | /football/usa/mls |
| `mexico-liga-mx` | /football/mexico/liga-de-expansion-mx (**⚠️ URL incorrecta — verificar**) |
| `champions-league` | /football/europe/champions-league |
| `europa-league` | /football/europe/europa-league |
| `conference-league` | /football/europe/conference-league/ |

### Ligas que FALTAN en OddsHarvester (agregar manualmente)

| Liga | URL OddsPortal esperada | league_id |
|------|------------------------|-----------|
| Chile Primera División | /football/chile/primera-division/ | 265 |
| Ecuador Liga Pro | /football/ecuador/liga-pro/ | 242 |
| Uruguay Primera División | /football/uruguay/primera-division/ | 268, 270 |
| Paraguay Primera División | /football/paraguay/primera-division/ | 250, 252 |
| Perú Liga 1 | /football/peru/liga-1/ | 281 |
| Venezuela Liga FUTVE | /football/venezuela/primera-division/ | 299 |
| Bolivia División Profesional | /football/bolivia/division-profesional/ | 344 |

> **Acción**: No necesitamos modificar OddsHarvester. Podemos pasar URLs custom con `--league` o usar el flag `--match-link` para scrapear directamente.

---

## 3. Arquitectura del Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│  Fase 0: Smoke Test (1 liga, 1 temporada, ~20 matches)     │
│  → Validar: scraping OK, team names extraídos, JSON válido │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│  Fase 1: Build Alias Dictionary                             │
│  → Por cada liga: extraer team names únicos de OddsPortal   │
│  → Mapear a team_id interno (automático + manual review)    │
│  → Smoke test: 100% aliases resueltos antes de continuar    │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│  Fase 2: Scrape Histórico (por liga, por temporada)         │
│  → oddsharvester historic → JSON files por temporada        │
│  → Rate limit natural de Playwright (~2-5 seg/página)       │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│  Fase 3: Ingest to PostgreSQL                               │
│  → Match OddsPortal → matches table (team_names + date)     │
│  → Extraer odds: Pinnacle > bet365 > Average                │
│  → UPDATE matches SET opening_odds_* WHERE id = ? AND NULL  │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│  Fase 4: Verificación Post-Backfill                         │
│  → Coverage por liga (antes/después)                        │
│  → Sanity: odds entre 1.01 y 50.0                           │
│  → No cross-contamination entre ligas                       │
└─────────────────────────────────────────────────────────────┘
```

---

## 4. Fase 0: Smoke Test Inicial

### Objetivo
Validar que OddsHarvester funciona con nuestras ligas target antes de invertir tiempo en aliases.

### Pasos

```bash
# 1. Instalar OddsHarvester
pip install oddsharvester
playwright install chromium

# 2. Smoke: Colombia 2024 (liga ya soportada en OddsHarvester)
oddsharvester historic \
  -s football \
  -l colombia-primera-a \
  --season 2024 \
  -m 1x2 \
  --headless \
  -f json \
  -o /tmp/smoke_colombia_2024.json

# 3. Verificar output
python -c "
import json
data = json.load(open('/tmp/smoke_colombia_2024.json'))
print(f'Matches: {len(data)}')
teams = set()
for m in data:
    teams.add(m['home_team'])
    teams.add(m['away_team'])
print(f'Unique teams: {len(teams)}')
for t in sorted(teams):
    print(f'  {t}')
# Verificar que hay odds
with_odds = sum(1 for m in data if m.get('1x2_market'))
print(f'With 1x2 odds: {with_odds}/{len(data)}')
"
```

### Criterios de éxito
- [ ] JSON parseado correctamente
- [ ] N matches > 100 para temporada completa
- [ ] Unique teams ~ 18-20 para liga colombiana
- [ ] 90%+ matches con 1x2_market
- [ ] Team names legibles (no IDs, no basura)

### Smoke para ligas SIN constante en OddsHarvester

Para Chile, Ecuador, etc. que no están en las constantes:

```bash
# Opción A: Probar con URL directa
oddsharvester historic \
  -s football \
  -l chile-primera-division \
  --season 2024 \
  -m 1x2 \
  --headless \
  -o /tmp/smoke_chile_2024.json

# Si falla, verificar URL en OddsPortal manualmente y usar match-link
```

> **Decisión**: Si OddsHarvester no soporta pasar ligas custom por URL, clonaremos el repo y agregaremos las constantes faltantes (7 ligas).

---

## 5. Fase 1: Diccionario de Aliases (CRÍTICO)

### 5.1 Estrategia de mapeo

El mapeo de team names es el paso más crítico. Seguimos el patrón probado de FDUK (`data/fduk_team_aliases.json`):

```
OddsPortal team name → team_id interno (Bon Jogo DB)
```

### 5.2 Estructura del archivo de aliases

**Archivo**: `data/oddsportal_team_aliases.json`

```json
{
  "_meta": {
    "version": "1.0.0",
    "description": "OddsPortal team name → internal team_id mapping",
    "source": "oddsportal.com via OddsHarvester",
    "total_entries": 0,
    "leagues_covered": []
  },
  "ColombiaPrimeraA": {
    "Atletico Nacional": 1130,
    "Millonarios": 1131,
    "Junior": 1132,
    "Deportivo Cali": 1136,
    "America de Cali": 1137,
    "Santa Fe": 1134,
    "La Equidad": 4195,
    "Alianza Petrolera": 4193,
    "Alianza Valledupar": 4193,
    "Internacional de Bogota": 4195
  },
  "ChilePrimeraDivision": {
    "Colo Colo": 2315,
    "Universidad de Chile": 2317
  }
}
```

### 5.3 Proceso de construcción de aliases (semi-automático)

#### Paso 1: Extraer team names únicos de OddsPortal

Después del smoke test, por cada liga:

```python
# scripts/build_oddsportal_aliases.py (helper)
import json, sys
from collections import defaultdict

def extract_teams(json_files: list[str]) -> dict[str, set]:
    """Extract unique team names from OddsHarvester JSON output."""
    teams_by_season = defaultdict(set)
    all_teams = set()
    for f in json_files:
        data = json.load(open(f))
        season = f.split("_")[-1].replace(".json", "")
        for m in data:
            teams_by_season[season].add(m["home_team"])
            teams_by_season[season].add(m["away_team"])
            all_teams.add(m["home_team"])
            all_teams.add(m["away_team"])
    return all_teams, teams_by_season
```

#### Paso 2: Auto-match contra DB

```python
async def auto_match_teams(session, league_id: int, op_names: set[str]) -> dict:
    """Try to auto-match OddsPortal names to DB teams."""
    # Query all teams in this league
    db_teams = await session.execute(text("""
        SELECT DISTINCT t.id, t.name, t.code
        FROM teams t
        JOIN matches m ON (m.home_team_id = t.id OR m.away_team_id = t.id)
        WHERE m.league_id = :lid AND m.date >= '2020-01-01'
    """), {"lid": league_id})

    db_map = {row.name.lower().strip(): row.id for row in db_teams}
    # Also check existing FDUK aliases
    fduk = json.load(open("data/fduk_team_aliases.json"))
    fduk_flat = {}
    for section, aliases in fduk.items():
        if section.startswith("_"): continue
        for name, tid in aliases.items():
            fduk_flat[name.lower().strip()] = tid

    matched = {}
    unmatched = []
    for op_name in op_names:
        key = op_name.lower().strip()
        # 1. Exact match in DB
        if key in db_map:
            matched[op_name] = db_map[key]
        # 2. Check FDUK aliases (might already have this name)
        elif key in fduk_flat:
            matched[op_name] = fduk_flat[key]
        # 3. Fuzzy match (Jaccard > 0.85)
        else:
            best_score, best_id = 0, None
            for db_name, db_id in db_map.items():
                score = _jaccard_similarity(key, db_name)
                if score > best_score:
                    best_score, best_id = score, db_id
            if best_score >= 0.85:
                matched[op_name] = best_id  # mark as auto-fuzzy
            else:
                unmatched.append((op_name, best_score, best_id))

    return matched, unmatched
```

#### Paso 3: Revisión manual de unmatched

```
UNMATCHED TEAMS (Colombia Primera A):
  "Int. de Bogota"     → best fuzzy: "Internacional de Bogota" (0.72) → team_id=4195 ✓
  "Alianza FC"         → best fuzzy: "Alianza Petrolera" (0.68) → team_id=4193 ✓
  "Dep. Pereira"       → best fuzzy: "Deportivo Pereira" (0.71) → team_id=4196 ✓

ACTION: Add these manually to oddsportal_team_aliases.json
```

### 5.4 Manejo de equipos renombrados

**Problema**: Equipos como "La Equidad" → "Internacional de Bogotá" (2026) aparecen con distintos nombres según la temporada en OddsPortal.

**Solución**: El diccionario de aliases mapea TODOS los nombres históricos al mismo team_id:

```json
{
  "ColombiaPrimeraA": {
    "La Equidad": 4195,
    "Internacional de Bogota": 4195,
    "Int. de Bogota": 4195,

    "Alianza Petrolera": 4193,
    "Alianza Valledupar": 4193,
    "Alianza FC": 4193
  }
}
```

**Fuentes para detectar renombramientos**:
1. `team_overrides` table (2 registros actuales: La Equidad → Internacional de Bogotá, Alianza Petrolera → Alianza Valledupar)
2. Comparar `teams_by_season` del extractor — si un equipo desaparece y otro nuevo aparece en la misma temporada, probable rename

### 5.5 Smoke test de aliases (GATE — no continuar sin pasar)

```python
# scripts/smoke_test_oddsportal_aliases.py
"""
GATE: 100% de team names del scrape deben tener alias resuelto.
Verificar que:
1. Cada alias resuelve a un team_id que existe en la DB
2. Cada team_id pertenece a la liga esperada (league-scoped validation)
3. No hay team_ids duplicados mapeando a ligas distintas (collision check)
4. Cobertura: == 100% de los team names del scrape tienen alias (ABE P0)
"""

LEAGUE_ALIAS_SECTIONS = {
    "ColombiaPrimeraA": 239,
    "ChilePrimeraDivision": 265,
    "EcuadorLigaPro": 242,
    "UruguayPrimeraDivision": [268, 270],
    "ParaguayPrimeraDivision": [250, 252],
    "PeruLiga1": 281,
    "VenezuelaPrimeraDivision": 299,
    "BoliviaDivisionProfesional": 344,
    "ArgentinaLigaProfesional": 128,
    "BrasilSerieA": 71,
    "MexicoLigaMX": 262,
    "MLS": 253,
}

def smoke_test():
    aliases = json.load(open("data/oddsportal_team_aliases.json"))
    errors = []

    for section, expected_league_ids in LEAGUE_ALIAS_SECTIONS.items():
        if section not in aliases:
            errors.append(f"MISSING section: {section}")
            continue

        if isinstance(expected_league_ids, int):
            expected_league_ids = [expected_league_ids]

        for name, team_id in aliases[section].items():
            # 1. team_id exists in DB
            row = db.execute(f"SELECT id FROM teams WHERE id = {team_id}")
            if not row:
                errors.append(f"INVALID team_id={team_id} for '{name}' in {section}")
                continue

            # 2. team_id belongs to expected league
            league_check = db.execute(f"""
                SELECT DISTINCT league_id FROM matches
                WHERE (home_team_id = {team_id} OR away_team_id = {team_id})
                AND league_id IN ({','.join(map(str, expected_league_ids))})
                AND date >= '2020-01-01' LIMIT 1
            """)
            if not league_check:
                errors.append(f"team_id={team_id} ('{name}') NOT in leagues {expected_league_ids}")

    if errors:
        print(f"FAILED: {len(errors)} errors")
        for e in errors:
            print(f"  ✗ {e}")
        sys.exit(1)
    else:
        print(f"PASSED: All aliases valid")
```

---

## 6. Fase 2: Scraping Histórico

### 6.1 Orden de ejecución (por prioridad)

| Prioridad | Liga | Temporadas | Matches estimados | Razón |
|-----------|------|------------|-------------------|-------|
| P0 | Colombia (239) | 2020-2026 | ~2,500 | 0% odds, liga en producción |
| P0 | Chile (265) | 2020-2026 | ~1,500 | 0% odds |
| P0 | Ecuador (242) | 2020-2026 | ~1,500 | 0% odds |
| P0 | Perú (281) | 2020-2026 | ~2,000 | 0% odds |
| P1 | Uruguay (268/270) | 2020-2026 | ~1,700 | 0% odds |
| P1 | Paraguay (250/252) | 2020-2026 | ~1,500 | 0% odds |
| P1 | Venezuela (299) | 2020-2026 | ~1,800 | 0% odds |
| P1 | Bolivia (344) | 2020-2026 | ~1,600 | 0% odds |
| P2 | Argentina (128) | 2020-2022 | ~1,661 | Gap parcial (FDUK cubre 2023+) |
| P2 | Brasil (71) | 2020-2022 | ~1,151 | Gap parcial |
| P2 | México (262) | 2020-2022 | ~1,150 | Gap parcial |
| P2 | MLS (253) | 2020-2022 | ~1,325 | Gap parcial |
| P3 | Champions League (2) | 2020-2026 | ~1,378 | Copa, no liga (menor impacto ML) |
| P3 | Europa League (3) | 2020-2026 | ~1,453 | Copa |
| P3 | Conference League (848) | 2021-2026 | ~2,044 | Copa |

### 6.2 Script de scraping

```bash
#!/bin/bash
# scripts/scrape_oddsportal.sh

# Format: "oddsharvester-league-name:alias_section:league_ids"
# league_ids is comma-separated for split seasons (Apertura/Clausura)
LEAGUES=(
  "colombia-primera-a:ColombiaPrimeraA:239"
  "chile-primera-division:ChilePrimeraDivision:265"
  "ecuador-liga-pro:EcuadorLigaPro:242"
  "peru-liga-1:PeruLiga1:281"
  "uruguay-primera-division:UruguayPrimeraDivision:268,270"
  "paraguay-primera-division:ParaguayPrimeraDivision:250,252"
  "venezuela-primera-division:VenezuelaPrimeraDivision:299"
  "bolivia-division-profesional:BoliviaDivisionProfesional:344"
  "argentina-liga-profesional:ArgentinaLigaProfesional:128"
  "brazil-serie-a:BrasilSerieA:71"
  "mexico-liga-mx:MexicoLigaMX:262"
  "usa-mls:MLS:253"
)

SEASONS=("2020" "2021" "2022" "2023" "2024" "2025" "2026")
OUTPUT_DIR="data/oddsportal_raw"

mkdir -p "$OUTPUT_DIR"

for entry in "${LEAGUES[@]}"; do
  IFS=':' read -r league section lids <<< "$entry"
  for season in "${SEASONS[@]}"; do
    outfile="$OUTPUT_DIR/${league}_${season}.json"
    if [ -f "$outfile" ]; then
      echo "SKIP: $outfile already exists"
      continue
    fi
    echo "SCRAPING: $league season $season → $outfile"
    oddsharvester historic \
      -s football \
      -l "$league" \
      --season "$season" \
      -m 1x2 \
      --headless \
      -f json \
      -o "$outfile"
    # Rate limit between seasons (OddsPortal anti-bot)
    sleep 10
  done
done
```

### 6.3 Consideraciones de rate limiting

- **Playwright natural**: ~2-5 seg/página (50-100 matches por página)
- **Inter-season pause**: 10 segundos entre temporadas
- **Inter-league pause**: 30 segundos entre ligas
- **Proxy**: Usar IPRoyal si OddsPortal bloquea (misma infra que FotMob/SofaScore)
- **Headless**: Siempre `--headless` para no necesitar display
- **Estimación total**: ~70 temporadas × ~5 min/temporada = ~6 horas para todo

### 6.4 Manejo de temporadas split (Apertura/Clausura) — ABE P0

Para Paraguay (250/252) y Uruguay (268/270), OddsPortal probablemente agrupa ambas fases en una sola temporada ("Paraguay 2024" = Apertura + Clausura).

**Problema**: Nuestra DB tiene `league_id=250` (Apertura) y `league_id=252` (Clausura) como ligas separadas. Un query con `league_id = 250` no encontraría partidos del Clausura (252).

**Solución (ABE P0)**: El script de ingesta usa `league_id IN (...)` para ligas split:

```python
# En scrape_oddsportal.sh: scrapeamos una sola vez
"uruguay-primera-division:268,270"   # OddsPortal → una temporada con ambas fases
"paraguay-primera-division:250,252"  # ídem

# En ingest_oddsportal.py: match_op_to_db() recibe league_ids=[268, 270]
# → query: WHERE league_id = ANY(:league_ids) AND team_ids match AND date ±1d
# → la DB determina el league_id correcto del match encontrado
```

**Validación en Fase 0**: Verificar cuántos partidos por temporada trae OddsPortal vs nuestra DB:
- Si OddsPortal trae ~300 partidos para "Paraguay 2024" y nuestra DB tiene ~150 en lid=250 + ~150 en lid=252, confirma que agrupa ambas fases.
- Si OddsPortal tiene URLs separadas ("paraguay/apertura" vs "paraguay/clausura"), scrapeamos ambas pero con los mismos `league_ids=[250, 252]`.

---

## 7. Fase 3: Script de Ingesta a PostgreSQL

### 7.1 Archivo: `scripts/ingest_oddsportal.py`

Sigue el patrón de `scripts/ingest_football_data_uk.py` pero adaptado al JSON de OddsHarvester.

### 7.2 Lógica de extracción de odds

Prioridad de bookmakers (misma que FDUK):

```python
BOOKMAKER_PRIORITY = [
    "Pinnacle",       # Gold standard para closing odds
    "bet365",         # Mayor liquidez retail
    "1xBet",          # Amplia cobertura LATAM
    "Marathon Bet",   # Alternativa
    "William Hill",   # Legacy
]

def extract_best_odds(match_data: dict) -> tuple[float, float, float, str] | None:
    """Extract best available 1x2 odds following bookmaker priority.
    Returns (home_odds, draw_odds, away_odds, source_name) or None.
    """
    market = match_data.get("1x2_market", [])
    if not market:
        return None

    # Index by bookmaker name (case-insensitive)
    by_bookie = {m["bookmaker_name"].lower(): m for m in market}

    # Try priority order
    for bookie in BOOKMAKER_PRIORITY:
        entry = by_bookie.get(bookie.lower())
        if entry:
            try:
                h = float(entry["1"])
                d = float(entry["X"])
                a = float(entry["2"])
                if h > 1.0 and d > 1.0 and a > 1.0:
                    return h, d, a, f"OddsPortal ({bookie})"
            except (ValueError, KeyError):
                continue

    # Fallback: compute average across all bookmakers
    odds_h, odds_d, odds_a = [], [], []
    for entry in market:
        try:
            odds_h.append(float(entry["1"]))
            odds_d.append(float(entry["X"]))
            odds_a.append(float(entry["2"]))
        except (ValueError, KeyError):
            continue

    if odds_h:
        avg_h = sum(odds_h) / len(odds_h)
        avg_d = sum(odds_d) / len(odds_d)
        avg_a = sum(odds_a) / len(odds_a)
        return avg_h, avg_d, avg_a, f"OddsPortal (avg of {len(odds_h)})"

    return None
```

### 7.3 Lógica de matching OddsPortal → DB

**ABE P0: Alias lookup es league-scoped** para evitar colisiones (ej. "Nacional" existe en Colombia, Uruguay, Paraguay).

```python
# Constante: mapeo de secciones del alias dict a league_ids válidos
# Para ligas split, incluir AMBOS league_ids → el query busca en todos
SECTION_TO_LEAGUE_IDS: dict[str, list[int]] = {
    "ColombiaPrimeraA": [239],
    "ChilePrimeraDivision": [265],
    "EcuadorLigaPro": [242],
    "UruguayPrimeraDivision": [268, 270],    # Apertura + Clausura
    "ParaguayPrimeraDivision": [250, 252],    # Apertura + Clausura
    "PeruLiga1": [281],
    "VenezuelaPrimeraDivision": [299],
    "BoliviaDivisionProfesional": [344],
    "ArgentinaLigaProfesional": [128],
    "BrasilSerieA": [71],
    "MexicoLigaMX": [262],
    "MLS": [253],
}

def load_aliases_for_section(section: str) -> dict[str, int]:
    """Load aliases ONLY for a specific league section.
    Returns dict {normalized_name: team_id}.
    ABE P0: league-scoped to prevent cross-league collisions.
    """
    all_aliases = json.load(open("data/oddsportal_team_aliases.json"))
    section_data = all_aliases.get(section, {})
    return {name.lower().strip(): tid for name, tid in section_data.items()}


async def match_op_to_db(
    session,
    op_match: dict,
    league_ids: list[int],
    aliases: dict[str, int]
) -> int | None:
    """Match an OddsPortal match to our matches table.

    ABE P0 changes vs original:
    - aliases is league-scoped (loaded per-section, not global flat)
    - league_ids is a LIST to handle split seasons (Apertura/Clausura)
    - Query uses league_id IN (:league_ids) instead of = :league_id

    Strategy:
    1. Resolve home/away team names via league-scoped alias dictionary
    2. Find match in DB by team_ids + date (±1 day) + league_ids
    3. Validate score matches (if available)

    Returns match_id or None.
    """
    home_name = op_match["home_team"].strip()
    away_name = op_match["away_team"].strip()

    # Resolve via league-scoped aliases (NO global flat dict)
    home_id = aliases.get(home_name.lower())
    away_id = aliases.get(away_name.lower())

    if not home_id or not away_id:
        return None  # Unresolved team — needs alias

    # Parse match date
    match_date = datetime.strptime(op_match["match_date"], "%Y-%m-%d %H:%M:%S")

    # Query DB with ±1 day tolerance, league_id IN (...) for split seasons
    result = await session.execute(text("""
        SELECT id, league_id, home_goals, away_goals
        FROM matches
        WHERE home_team_id = :home_id
          AND away_team_id = :away_id
          AND league_id = ANY(:league_ids)
          AND ABS(EXTRACT(EPOCH FROM (date - :match_date))) < 86400
          AND status = 'FT'
        ORDER BY ABS(EXTRACT(EPOCH FROM (date - :match_date)))
        LIMIT 1
    """), {
        "home_id": home_id,
        "away_id": away_id,
        "league_ids": league_ids,
        "match_date": match_date,
    })

    row = result.first()
    if not row:
        return None

    # Score validation (hard reject on mismatch — safety first)
    if op_match.get("home_score") is not None:
        if row.home_goals != op_match["home_score"] or row.away_goals != op_match["away_score"]:
            logger.warning(
                f"Score mismatch: OP {home_name} {op_match['home_score']}-{op_match['away_score']} "
                f"vs DB {row.home_goals}-{row.away_goals} (match_id={row.id}, league={row.league_id})"
            )
            return None

    return row.id
```

### 7.4 Escritura a DB — Contrato exacto de columnas (ABE P0)

**Semántica**: OddsPortal muestra **closing odds** (último precio antes del KO). Las escribimos en `opening_odds_*` porque:
1. Son la mejor aproximación disponible para ligas sin otra fuente
2. FDUK ya usa el mismo patrón (`opening_odds_kind = 'closing'`) para ligas Extra (LATAM)
3. Para el modelo ML, closing ≈ opening para propósitos de entrenamiento (spread típico < 3%)

**Contrato de columnas**:

| Columna | Valor OddsPortal | Valor FDUK (referencia) |
|---------|-----------------|------------------------|
| `opening_odds_home` | float (1X2 home) | float |
| `opening_odds_draw` | float (1X2 draw) | float |
| `opening_odds_away` | float (1X2 away) | float |
| `opening_odds_source` | `"OddsPortal (Pinnacle)"` o `"OddsPortal (avg of N)"` | `"football-data.co.uk (B365)"` |
| `opening_odds_kind` | **`"closing"`** | `"proxy_pre_closing"` (EUR) / `"closing"` (LATAM) |
| `opening_odds_column` | `"1x2"` | `"B365H"` / `"PSCH"` |
| `opening_odds_recorded_at` | `match.date` (fecha del partido) | `match.date` |
| `opening_odds_recorded_at_type` | `"match_date"` | `"file_asof"` |

```python
async def backfill_match_odds(conn, match_id: int, odds: tuple):
    """Write odds to matches table. Only where opening_odds_home IS NULL.
    ABE P0: contrato de columnas documentado arriba.
    """
    home_odds, draw_odds, away_odds, source_name = odds

    await conn.execute("""
        UPDATE matches
        SET opening_odds_home = $1,
            opening_odds_draw = $2,
            opening_odds_away = $3,
            opening_odds_source = $4,
            opening_odds_kind = 'closing',
            opening_odds_column = '1x2',
            opening_odds_recorded_at = date,
            opening_odds_recorded_at_type = 'match_date'
        WHERE id = $5
          AND opening_odds_home IS NULL
    """, home_odds, draw_odds, away_odds, source_name, match_id)
```

**Guardias**:
- `AND opening_odds_home IS NULL` — nunca sobreescribe datos FDUK existentes
- `opening_odds_kind = 'closing'` — explícitamente marcado. Consumidores (feature_matrix, ML) tratan closing = proxy-opening
- Score validation upstream — rechaza matches con score distinto antes de llegar aquí
- Odds range check: `h > 1.0 AND d > 1.0 AND a > 1.0` en `extract_best_odds()`

### 7.5 Dry-run mode

```bash
# Dry run: solo reporta qué haría, sin escribir a DB
python scripts/ingest_oddsportal.py \
  --league 239 \
  --input data/oddsportal_raw/colombia-primera-a_*.json \
  --dry-run

# Output:
# DRY RUN: Would update 2,341 matches with odds
# Unmatched teams: 3
#   "Dep. Pereira" → no alias (appears 45 times)
#   "Int. de Bogota" → no alias (appears 38 times)
#   "Jaguares FC" → no alias (appears 22 times)
# Score mismatches: 7
# Already has odds (skip): 0
```

---

## 8. Fase 4: Verificación Post-Backfill

### 8.1 Queries de verificación

```sql
-- 1. Coverage antes vs después (por liga)
SELECT l.id, l.name,
  COUNT(*) as total,
  COUNT(m.opening_odds_home) as with_odds,
  ROUND(100.0 * COUNT(m.opening_odds_home) / COUNT(*), 1) as pct
FROM matches m
JOIN admin_leagues l ON l.id = m.league_id
WHERE m.date >= '2020-01-01'
  AND l.id IN (239, 265, 242, 281, 299, 250, 252, 268, 270, 344, 128, 71, 262, 253)
GROUP BY l.id, l.name
ORDER BY pct ASC;

-- 2. Sanity: odds en rango razonable
SELECT
  COUNT(*) as total,
  COUNT(*) FILTER (WHERE opening_odds_home < 1.01 OR opening_odds_home > 50) as invalid_home,
  COUNT(*) FILTER (WHERE opening_odds_draw < 1.01 OR opening_odds_draw > 50) as invalid_draw,
  COUNT(*) FILTER (WHERE opening_odds_away < 1.01 OR opening_odds_away > 50) as invalid_away,
  MIN(opening_odds_home) as min_home,
  MAX(opening_odds_home) as max_home
FROM matches
WHERE opening_odds_source LIKE 'OddsPortal%';

-- 3. Distribution de fuentes (no cross-contamination)
SELECT opening_odds_source, COUNT(*)
FROM matches
WHERE opening_odds_home IS NOT NULL
GROUP BY opening_odds_source
ORDER BY COUNT(*) DESC;

-- 4. Coverage temporal (por año, por liga)
SELECT
  m.league_id, l.name,
  EXTRACT(YEAR FROM m.date) as year,
  COUNT(*) as total,
  COUNT(m.opening_odds_home) as with_odds,
  ROUND(100.0 * COUNT(m.opening_odds_home) / COUNT(*), 1) as pct
FROM matches m
JOIN admin_leagues l ON l.id = m.league_id
WHERE m.league_id IN (239, 128, 71)
  AND m.date >= '2020-01-01'
GROUP BY m.league_id, l.name, EXTRACT(YEAR FROM m.date)
ORDER BY m.league_id, year;
```

### 8.2 Criterios de éxito por tier

| Tier | Ligas | Target coverage post-backfill |
|------|-------|------------------------------|
| TIER 3 → filled | Colombia, Chile, Ecuador, etc. | ≥ 80% (2020-2026) |
| TIER 2 → filled | Argentina, Brasil, México, MLS | ≥ 85% (2020-2026) |
| TIER 1 (ya OK) | EPL, La Liga, Serie A, etc. | Sin cambios (≥ 87%) |

---

## 9. Archivos a Crear/Modificar

| Archivo | Acción | Descripción |
|---------|--------|-------------|
| `data/oddsportal_team_aliases.json` | CREAR | Diccionario de aliases OddsPortal → team_id |
| `data/oddsportal_raw/` | CREAR (dir) | JSONs crudos del scraping (no commitear, .gitignore) |
| `scripts/build_oddsportal_aliases.py` | CREAR | Helper para construir aliases semi-automáticamente |
| `scripts/smoke_test_oddsportal_aliases.py` | CREAR | Smoke test de aliases (GATE antes de backfill) |
| `scripts/scrape_oddsportal.sh` | CREAR | Wrapper para scraping batch |
| `scripts/ingest_oddsportal.py` | CREAR | Script principal de ingesta a PostgreSQL |
| `.gitignore` | MODIFICAR | Agregar `data/oddsportal_raw/` |

---

## 10. Cronograma de Ejecución

| Paso | Descripción | Estimación | Dependencia |
|------|-------------|------------|-------------|
| 0 | Instalar OddsHarvester + smoke test | 30 min | — |
| 1a | Smoke ligas faltantes (Chile, Ecuador, etc.) | 1 hora | Paso 0 |
| 1b | Agregar constantes faltantes si necesario | 30 min | Paso 1a |
| 2 | Scraping Colombia (smoke + all seasons) | 30 min | Paso 0 |
| 3 | Build aliases Colombia (auto + manual) | 1 hora | Paso 2 |
| 4 | Smoke test aliases Colombia | 15 min | Paso 3 |
| 5 | Ingest Colombia (dry-run → real) | 30 min | Paso 4 |
| 6 | Verificación Colombia | 15 min | Paso 5 |
| 7 | Repetir pasos 2-6 para cada liga | ~4 horas | Paso 6 OK |
| 8 | Scraping TIER 2 (ARG/BRA/MEX/MLS 2020-2022) | 2 horas | Paso 7 |
| 9 | Verificación final global | 30 min | Paso 8 |

**Total estimado**: ~10 horas (distribuidas en 2-3 sesiones)

---

## 11. Riesgos y Mitigación

| Riesgo | Probabilidad | Impacto | Mitigación |
|--------|-------------|---------|------------|
| OddsPortal bloquea scraping | Media | Alto | Usar proxy IPRoyal, reducir velocidad, rotar User-Agent |
| Team names muy diferentes a DB | Baja | Media | Auto-match + fuzzy + review manual. FDUK aliases como base |
| Temporadas split (Apertura/Clausura) confunden matching | Media | Baja | Match por team_ids + fecha, no por liga. La DB sabe qué liga es |
| OddsHarvester no soporta liga custom | Media | Baja | Clonar repo y agregar constantes (7 líneas de código) |
| Playwright falla en Railway/CI | Alta | Baja | Scraping es LOCAL (laptop), solo ingesta corre contra DB |
| Closing odds ≠ opening odds (sesgo) | Baja | Baja | Para training ML esto es aceptable. FDUK ya usa closing para LATAM |

---

## 12. Decisiones Pendientes

1. **¿Incluir copas internacionales (UCL, UEL, UECL)?** — P3, menor impacto en ML league-only
2. **¿Scraping de odds_history (movimiento)?** — Opción `--scrape-odds-history` aumenta mucho el tiempo. Solo activar si necesitamos opening vs closing
3. **¿Mantener JSONs crudos?** — Recomendación: sí, en `data/oddsportal_raw/` (gitignored). Permite re-procesar sin re-scrapear
4. **¿Bookmaker preferido?** — Pinnacle > bet365 > promedio. Alineado con FDUK

---

## 13. ABE Review — Gaps P0 Cerrados (2026-02-10)

### Gap 1: Semántica "closing en opening_odds" — CERRADO

**Problema**: OddsPortal es "closing" pero escribimos en `opening_odds_*`. Podría confundir métricas futuras.

**Solución**: Contrato exacto documentado en §7.4. `opening_odds_kind = 'closing'` marcado explícitamente. `opening_odds_source = 'OddsPortal (Pinnacle)'` diferencia claramente de FDUK. Consumidores (feature_matrix, ML) ya tratan ambos tipos como equivalentes.

### Gap 2: Aliases league-scoped — CERRADO

**Problema**: Alias lookup plano (`dict[str, int]`) causa colisiones entre ligas (ej. "Nacional" existe en Colombia=1130, Uruguay=TBD, Paraguay=TBD).

**Solución**: Aliases se cargan POR SECCIÓN (`load_aliases_for_section("ColombiaPrimeraA")`). Nunca se aplana a un dict global. El matching recibe aliases ya filtrados por liga. Ver §7.3.

### Gap 3: Split seasons league_ids — CERRADO

**Problema**: Query filtraba `league_id = :league_id` pero OddsPortal agrupa Apertura+Clausura en una temporada.

**Solución**: `match_op_to_db()` recibe `league_ids: list[int]` y query usa `league_id = ANY(:league_ids)`. Para Paraguay=[250,252], Uruguay=[268,270]. Ver §6.4 y §7.3.

### Gap 4: GATE 100% aliases (no 95%) — CERRADO

**Problema**: Inconsistencia entre "100% resueltos" y ">=95%".

**Solución**: GATE es **100%** de team names del scrape objetivo. Si hay unmatched, se resuelven manualmente ANTES de continuar. Ver §5.5.

### Gap 5: SEASONS incluye 2026 — CERRADO

**Problema**: Lista de temporadas solo llegaba a 2025.

**Solución**: `SEASONS=("2020" ... "2026")`. Ver §6.2.

### ABE P0 adicional: Métricas de GATE para abort automático

El script de ingesta (`ingest_oddsportal.py`) DEBE imprimir estas métricas en dry-run y abortar si no cumple:

```
GATE METRICS (per-league):
  total_op_matches:     N    (matches en JSON scrapeado)
  resolved_teams:       N/N  (100% requerido — ABORT si < 100%)
  matched_to_db:        N    (matches enlazados a DB)
  match_rate:           X%   (matched/total — WARNING si < 70%)
  score_mismatches:     N    (rechazados por score distinto)
  already_has_odds:     N    (skipped, ya tienen opening_odds)
  would_update:         N    (filas que se actualizarían)

ABORT CONDITIONS:
  - resolved_teams < 100% → "ABORT: unresolved teams, update aliases first"
  - match_rate < 50% → "ABORT: too many unmatched, check league/date mapping"
  - score_mismatches > 5% of total → "WARNING: high score mismatch rate"
```
