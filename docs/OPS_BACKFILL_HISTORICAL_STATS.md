# Backfill Historical Stats - Plan Operativo

**Fecha**: 2026-02-01
**Estado**: Pendiente aprobación ATI para --dry-run piloto
**Objetivo**: Recuperar estadísticas históricas (shots, corners) de API-Football

---

## Contexto

### Problema Detectado (P0 "Data Fantasma")
- Las 4 features de shots/corners están en 0 para partidos históricos
- Impacto: ~30% del dataset de training tiene features contaminados
- Causa: `stats_backfill` job solo corre para partidos recientes (72h)

### Features Afectados
| # | Feature | Requiere |
|---|---------|----------|
| 3 | `home_shots_avg` | `stats->'home'->>'total_shots'` |
| 4 | `home_corners_avg` | `stats->'home'->>'corner_kicks'` |
| 9 | `away_shots_avg` | `stats->'away'->>'total_shots'` |
| 10 | `away_corners_avg` | `stats->'away'->>'corner_kicks'` |

Los otros 10 features (goals, rest_days, matches_played, goal_diff, rest_diff) se calculan de datos que ya tenemos.

---

## Arquitectura por Fases

### Fase 0: Pilot (OBLIGATORIO)
- **Objetivo**: Validar cobertura real de API-Football antes de backfill masivo
- **Scope**: 200 fixtures por (liga, temporada)
- **Output**: CSV con métricas de cobertura por segmento
- **Decisión**: `GO/HOLD/STOP` por segmento basado en evidencia

### Fase 1: Backfill Segmentos GO
- **Scope**: Solo segmentos que pasaron Fase 0 con umbral
- **Umbral GO**: ≥70% `API_OK` (ligas principales), ≥50% (secundarias)

### Fase 2: Marcado de Segmentos STOP
- **Scope**: Segmentos con `NO_DATA_RATIO > 0.50` después del piloto
- **Acción**: Marcar con `tainted=true` + `tainted_evidence` JSON

---

## Estados de Respuesta API

| Estado | Descripción | Acción |
|--------|-------------|--------|
| `API_OK` | Stats completos (shots + corners) | Merge a DB |
| `PARTIAL` | Solo algunos campos | Merge parcial, log warning |
| `NO_DATA` | API retorna `response: []` o vacío | Skip, registrar en evidencia |
| `ERROR` | HTTP error, timeout, rate limit | Retry con backoff |

---

## Hipótesis Iniciales (NO VINCULANTE)

> **IMPORTANTE**: Estas son hipótesis basadas en pruebas manuales.
> El piloto determinará la realidad. NO excluir ligas antes del piloto.

### Hipótesis de Cobertura API-Football

| Liga | Hipótesis | Verificar en Piloto |
|------|-----------|---------------------|
| Venezuela | Probablemente sin stats | Confirmar con sample |
| Bolivia | Stats parciales desde ~2023 | Confirmar desde cuándo |
| Uruguay | Stats parciales desde ~2020 | Confirmar desde cuándo |
| Paraguay | Stats parciales, fecha incierta | Confirmar desde cuándo |
| Turquía | API tiene stats desde 2015, DB vacía | Confirmar backfill viable |
| Top 5 Europa | Alta cobertura esperada | Confirmar % real |

### Decisión Post-Piloto

El piloto generará un CSV con:
```
league_id, league_name, season, sampled, api_ok, partial, no_data, error, coverage_pct, decision
```

Decisiones automáticas:
- `coverage_pct >= 70%` → `GO` (backfill)
- `coverage_pct >= 50% AND < 70%` → `HOLD` (revisar manualmente)
- `coverage_pct < 50%` → `STOP` (candidato a tainted)

---

## CLI del Script

### Argumentos

```
--mode pilot|backfill     # Fase 0 (pilot) o Fase 1+ (backfill)
--dry-run                 # Solo reporte, sin writes a DB
--league-id INT           # Filtrar por liga específica
--season INT              # Filtrar por temporada
--checkpoint FILE         # Archivo para resume (JSON)
--rate-limit INT          # Requests por minuto (default: 75)
--output-csv FILE         # Exportar resultados a CSV
--batch-size INT          # Fixtures por segmento (default: 200 para pilot)
```

### Ejemplos de Uso

```bash
# Fase 0: Pilot completo (dry-run primero)
python scripts/backfill_historical_stats.py \
  --mode pilot \
  --dry-run \
  --output-csv logs/pilot_dryrun.csv

# Fase 0: Pilot real (con llamadas API)
python scripts/backfill_historical_stats.py \
  --mode pilot \
  --output-csv logs/pilot_results.csv

# Fase 0: Pilot solo para una liga
python scripts/backfill_historical_stats.py \
  --mode pilot \
  --league-id 39 \
  --output-csv logs/pilot_epl.csv

# Fase 1: Backfill con checkpoint (después de aprobar piloto)
python scripts/backfill_historical_stats.py \
  --mode backfill \
  --checkpoint logs/backfill_cp.json \
  --rate-limit 75
```

---

## Estructura del Script

**Archivo**: `scripts/backfill_historical_stats.py`

```python
#!/usr/bin/env python3
"""
Backfill historical stats from API-Football.

Phases:
  - pilot: Sample fixtures per (league, season) to validate API coverage
  - backfill: Full backfill for segments that passed pilot

IMPORTANTE: No hay ligas hardcodeadas como excluidas.
            El piloto determina qué segmentos son GO/HOLD/STOP.
"""

import argparse
import asyncio
import csv
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import aiohttp
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

# ═══════════════════════════════════════════════════════════════
# CONFIGURACIÓN
# ═══════════════════════════════════════════════════════════════

# API key MUST come from environment (never hardcode secrets in repo)
# Accepts either API_FOOTBALL_KEY (preferred) or RAPIDAPI_KEY (legacy)
API_KEY = os.environ.get("API_FOOTBALL_KEY") or os.environ.get("RAPIDAPI_KEY")
if not API_KEY:
    raise RuntimeError("Missing API key: set API_FOOTBALL_KEY (preferred) or RAPIDAPI_KEY")
API_BASE = "https://v3.football.api-sports.io"
HEADERS = {"x-apisports-key": API_KEY}

# Umbrales de decisión (configurables)
THRESHOLD_GO = 70.0       # >= 70% coverage → GO
THRESHOLD_HOLD = 50.0     # >= 50% AND < 70% → HOLD
# < 50% → STOP

# Circuit breaker: parar segmento si NO_DATA streak muy alto
CIRCUIT_BREAKER_RATIO = 0.30  # 30% NO_DATA consecutivos → parar segmento

# Estados de respuesta
class ApiStatus:
    OK = "API_OK"
    PARTIAL = "PARTIAL"
    NO_DATA = "NO_DATA"
    ERROR = "ERROR"

# ═══════════════════════════════════════════════════════════════
# RATE LIMITER
# ═══════════════════════════════════════════════════════════════

class RateLimiter:
    def __init__(self, requests_per_minute: int = 75):
        self.rpm = requests_per_minute
        self.interval = 60.0 / requests_per_minute
        self.last_request = 0

    async def acquire(self):
        now = time.time()
        wait = self.interval - (now - self.last_request)
        if wait > 0:
            await asyncio.sleep(wait)
        self.last_request = time.time()

# ═══════════════════════════════════════════════════════════════
# API CLIENT
# ═══════════════════════════════════════════════════════════════

async def fetch_fixture_stats(
    session: aiohttp.ClientSession,
    fixture_id: int,
    rate_limiter: RateLimiter
) -> tuple[str, dict | None, float]:
    """
    Fetch stats for a single fixture from API-Football.

    Returns:
        (status, stats_dict, latency_ms)
    """
    await rate_limiter.acquire()

    url = f"{API_BASE}/fixtures/statistics?fixture={fixture_id}"
    start = time.time()

    try:
        async with session.get(url, headers=HEADERS, timeout=30) as resp:
            latency = (time.time() - start) * 1000

            if resp.status == 429:
                return ApiStatus.ERROR, {"error": "rate_limited"}, latency

            if resp.status != 200:
                return ApiStatus.ERROR, {"error": f"http_{resp.status}"}, latency

            data = await resp.json()

            if data.get("errors"):
                return ApiStatus.ERROR, {"error": str(data["errors"])}, latency

            if not data.get("response") or len(data["response"]) == 0:
                return ApiStatus.NO_DATA, None, latency

            stats = parse_stats(data["response"])

            if stats is None:
                return ApiStatus.NO_DATA, None, latency

            # Check completeness
            has_shots = any(
                t.get("total_shots") is not None
                for t in stats.values()
            )
            has_corners = any(
                t.get("corner_kicks") is not None
                for t in stats.values()
            )

            if has_shots and has_corners:
                return ApiStatus.OK, stats, latency
            elif has_shots or has_corners:
                return ApiStatus.PARTIAL, stats, latency
            else:
                return ApiStatus.NO_DATA, None, latency

    except asyncio.TimeoutError:
        return ApiStatus.ERROR, {"error": "timeout"}, 30000
    except Exception as e:
        return ApiStatus.ERROR, {"error": str(e)}, 0


def parse_stats(response: list) -> dict | None:
    """Parse API response into stats by team_id."""
    if not response or len(response) < 2:
        return None

    stats_by_team = {}
    for team_data in response:
        team_id = team_data.get("team", {}).get("id")
        if not team_id:
            continue

        team_stats = {}
        for stat in team_data.get("statistics", []):
            stat_type = stat.get("type", "")
            value = stat.get("value")

            if stat_type == "Total Shots":
                team_stats["total_shots"] = value
            elif stat_type == "Shots on Goal":
                team_stats["shots_on_goal"] = value
            elif stat_type == "Corner Kicks":
                team_stats["corner_kicks"] = value
            elif stat_type == "Ball Possession":
                if isinstance(value, str) and value.endswith("%"):
                    value = value.rstrip("%")
                team_stats["possession"] = int(value) if value else None
            elif stat_type == "Fouls":
                team_stats["fouls"] = value
            elif stat_type == "Yellow Cards":
                team_stats["yellow_cards"] = value
            elif stat_type == "Red Cards":
                team_stats["red_cards"] = value

        stats_by_team[team_id] = team_stats

    return stats_by_team

# ═══════════════════════════════════════════════════════════════
# MERGE LOGIC (IDEMPOTENTE)
# ═══════════════════════════════════════════════════════════════

def merge_stats(
    existing: dict | None,
    new: dict,
    home_team_id: int,
    away_team_id: int
) -> dict:
    """
    Merge new stats into existing WITHOUT overwriting.
    Only fills missing fields.
    """
    result = existing.copy() if existing else {}

    if "home" not in result:
        result["home"] = {}
    if "away" not in result:
        result["away"] = {}

    home_stats = new.get(home_team_id, {})
    away_stats = new.get(away_team_id, {})

    # Merge home (only fill missing)
    for key, value in home_stats.items():
        if value is not None and result["home"].get(key) is None:
            result["home"][key] = value

    # Merge away (only fill missing)
    for key, value in away_stats.items():
        if value is not None and result["away"].get(key) is None:
            result["away"][key] = value

    return result

# ═══════════════════════════════════════════════════════════════
# PILOT MODE
# ═══════════════════════════════════════════════════════════════

async def run_pilot(
    db_session: AsyncSession,
    http_session: aiohttp.ClientSession,
    rate_limiter: RateLimiter,
    league_id: int | None,
    season: int | None,
    batch_size: int,
    output_csv: str | None,
    dry_run: bool
) -> dict:
    """
    Run pilot to validate API coverage per (league, season) segment.

    NO hay ligas excluidas a priori. El piloto determina todo.
    """
    # Get all segments that need stats
    query = """
        SELECT DISTINCT
            al.league_id,
            al.name as league_name,
            al.country,
            EXTRACT(YEAR FROM m.date)::int as season,
            COUNT(*) as total_missing
        FROM matches m
        JOIN admin_leagues al ON m.league_id = al.league_id
        WHERE m.status = 'FT'
          AND m.api_football_id IS NOT NULL
          AND (m.stats IS NULL
               OR m.stats->'home'->>'total_shots' IS NULL
               OR m.stats->'home'->>'total_shots' = '')
          AND al.kind = 'league'
    """

    params = {}
    if league_id:
        query += " AND al.league_id = :league_id"
        params["league_id"] = league_id
    if season:
        query += " AND EXTRACT(YEAR FROM m.date) = :season"
        params["season"] = season

    query += """
        GROUP BY al.league_id, al.name, al.country, EXTRACT(YEAR FROM m.date)
        HAVING COUNT(*) >= 10
        ORDER BY COUNT(*) DESC
    """

    result = await db_session.execute(text(query), params)
    segments = [dict(r._mapping) for r in result.fetchall()]

    logging.info(f"Pilot: Found {len(segments)} segments to sample")

    pilot_results = []

    for seg in segments:
        lid = seg["league_id"]
        lname = seg["league_name"]
        country = seg["country"]
        syear = seg["season"]
        total_missing = seg["total_missing"]

        if dry_run:
            # Dry-run: solo reportar segmentos, no llamar API
            pilot_results.append({
                "league_id": lid,
                "league_name": lname,
                "country": country,
                "season": syear,
                "total_missing": total_missing,
                "sampled": 0,
                "api_ok": 0,
                "partial": 0,
                "no_data": 0,
                "error": 0,
                "coverage_pct": None,
                "avg_latency_ms": None,
                "decision": "DRY_RUN",
            })
            continue

        # Sample fixtures for this segment
        sample_query = text("""
            SELECT m.id, m.api_football_id, m.home_team_id, m.away_team_id
            FROM matches m
            WHERE m.league_id = :league_id
              AND EXTRACT(YEAR FROM m.date) = :season
              AND m.status = 'FT'
              AND m.api_football_id IS NOT NULL
              AND (m.stats IS NULL
                   OR m.stats->'home'->>'total_shots' IS NULL
                   OR m.stats->'home'->>'total_shots' = '')
            ORDER BY RANDOM()
            LIMIT :limit
        """)

        sample_result = await db_session.execute(sample_query, {
            "league_id": lid,
            "season": syear,
            "limit": batch_size
        })
        fixtures = [dict(r._mapping) for r in sample_result.fetchall()]

        if not fixtures:
            continue

        # Pilot this segment
        counts = {"API_OK": 0, "NO_DATA": 0, "PARTIAL": 0, "ERROR": 0}
        latencies = []
        no_data_streak = 0
        circuit_broken = False

        for fix in fixtures:
            status, stats, latency = await fetch_fixture_stats(
                http_session, fix["api_football_id"], rate_limiter
            )
            counts[status] += 1
            latencies.append(latency)

            # Circuit breaker
            if status == ApiStatus.NO_DATA:
                no_data_streak += 1
                if no_data_streak >= int(batch_size * CIRCUIT_BREAKER_RATIO):
                    logging.warning(
                        f"Circuit breaker: {lname} {syear} - "
                        f"{no_data_streak} consecutive NO_DATA"
                    )
                    circuit_broken = True
                    break
            else:
                no_data_streak = 0

        sampled = sum(counts.values())
        api_ok_count = counts["API_OK"] + counts["PARTIAL"]
        coverage = (api_ok_count / sampled * 100) if sampled > 0 else 0
        avg_latency = sum(latencies) / len(latencies) if latencies else 0

        # Determine decision
        if circuit_broken:
            decision = "STOP_CIRCUIT"
        elif coverage >= THRESHOLD_GO:
            decision = "GO"
        elif coverage >= THRESHOLD_HOLD:
            decision = "HOLD"
        else:
            decision = "STOP"

        pilot_results.append({
            "league_id": lid,
            "league_name": lname,
            "country": country,
            "season": syear,
            "total_missing": total_missing,
            "sampled": sampled,
            "api_ok": counts["API_OK"],
            "partial": counts["PARTIAL"],
            "no_data": counts["NO_DATA"],
            "error": counts["ERROR"],
            "coverage_pct": round(coverage, 1),
            "avg_latency_ms": round(avg_latency, 1),
            "decision": decision,
        })

        logging.info(
            f"Pilot {lname} {syear}: {coverage:.1f}% coverage, "
            f"decision={decision} (sampled {sampled})"
        )

    # Export CSV
    if output_csv and pilot_results:
        with open(output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=pilot_results[0].keys())
            writer.writeheader()
            writer.writerows(pilot_results)
        logging.info(f"Pilot results exported to {output_csv}")

    # Summary
    summary = {
        "total_segments": len(pilot_results),
        "go": sum(1 for r in pilot_results if r["decision"] == "GO"),
        "hold": sum(1 for r in pilot_results if r["decision"] == "HOLD"),
        "stop": sum(1 for r in pilot_results if r["decision"] in ("STOP", "STOP_CIRCUIT")),
        "dry_run": sum(1 for r in pilot_results if r["decision"] == "DRY_RUN"),
    }

    return {"segments": pilot_results, "summary": summary}

# ═══════════════════════════════════════════════════════════════
# BACKFILL MODE
# ═══════════════════════════════════════════════════════════════

async def run_backfill(
    db_session: AsyncSession,
    http_session: aiohttp.ClientSession,
    rate_limiter: RateLimiter,
    league_id: int | None,
    season: int | None,
    checkpoint_file: str | None,
    pilot_csv: str | None,
    force_all: bool,
    dry_run: bool,
    output_csv: str | None
) -> dict:
    """
    Run backfill for segments marked GO in pilot.

    Requires either pilot CSV (with GO segments) or --force-all flag.
    """
    # Load pilot results to know which segments are GO
    go_segments = set()
    if pilot_csv and Path(pilot_csv).exists():
        with open(pilot_csv) as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("decision") == "GO":
                    key = (int(row["league_id"]), int(row["season"]))
                    go_segments.add(key)
        logging.info(f"Loaded {len(go_segments)} GO segments from pilot")
    elif force_all:
        logging.warning("--force-all flag used - will process ALL segments (no pilot filtering)")
    # Note: if neither pilot_csv nor force_all, main() already exited with error

    # Load checkpoint
    checkpoint = {}
    if checkpoint_file and Path(checkpoint_file).exists():
        with open(checkpoint_file) as f:
            checkpoint = json.load(f)
        logging.info(f"Loaded checkpoint: last_id={checkpoint.get('last_id')}")

    # Query fixtures to backfill
    query = """
        SELECT
            m.id,
            m.api_football_id,
            m.home_team_id,
            m.away_team_id,
            m.stats,
            al.league_id,
            al.name as league_name,
            EXTRACT(YEAR FROM m.date)::int as season
        FROM matches m
        JOIN admin_leagues al ON m.league_id = al.league_id
        WHERE m.status = 'FT'
          AND m.api_football_id IS NOT NULL
          AND (m.stats IS NULL
               OR m.stats->'home'->>'total_shots' IS NULL
               OR m.stats->'home'->>'total_shots' = '')
          AND al.kind = 'league'
    """

    params = {}
    if league_id:
        query += " AND al.league_id = :league_id"
        params["league_id"] = league_id
    if season:
        query += " AND EXTRACT(YEAR FROM m.date) = :season"
        params["season"] = season
    if checkpoint.get("last_id"):
        query += " AND m.id > :last_id"
        params["last_id"] = checkpoint["last_id"]

    query += " ORDER BY m.id LIMIT 10000"

    result = await db_session.execute(text(query), params)
    fixtures = [dict(r._mapping) for r in result.fetchall()]

    logging.info(f"Backfill: Found {len(fixtures)} fixtures to process")

    # Process
    results = []
    updated = 0
    skipped_not_go = 0
    skipped_no_data = 0
    errors = 0

    for i, fix in enumerate(fixtures):
        lid = fix["league_id"]
        syear = fix["season"]

        # Skip if segment not in GO list (when pilot provided)
        if go_segments and (lid, syear) not in go_segments:
            skipped_not_go += 1
            continue

        # Fetch from API
        status, stats, latency = await fetch_fixture_stats(
            http_session, fix["api_football_id"], rate_limiter
        )

        result_row = {
            "match_id": fix["id"],
            "api_football_id": fix["api_football_id"],
            "league_id": lid,
            "league": fix["league_name"],
            "season": syear,
            "status": status,
            "latency_ms": round(latency, 1),
            "updated": False,
        }

        if status in (ApiStatus.OK, ApiStatus.PARTIAL) and stats:
            merged = merge_stats(
                fix["stats"],
                stats,
                fix["home_team_id"],
                fix["away_team_id"]
            )

            if not dry_run:
                await db_session.execute(
                    text("UPDATE matches SET stats = :stats WHERE id = :id"),
                    {"stats": json.dumps(merged), "id": fix["id"]}
                )

                if (i + 1) % 100 == 0:
                    await db_session.commit()
                    logging.info(f"Progress: {i+1}/{len(fixtures)} ({updated} updated)")

            updated += 1
            result_row["updated"] = True

        elif status == ApiStatus.NO_DATA:
            skipped_no_data += 1
        else:
            errors += 1

        results.append(result_row)

        # Save checkpoint
        if checkpoint_file and (i + 1) % 500 == 0:
            with open(checkpoint_file, "w") as f:
                json.dump({"last_id": fix["id"], "updated": updated}, f)

    # Final commit
    if not dry_run:
        await db_session.commit()

    # Export CSV
    if output_csv and results:
        with open(output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)

    return {
        "total_processed": len(fixtures),
        "updated": updated,
        "skipped_not_go": skipped_not_go,
        "skipped_no_data": skipped_no_data,
        "errors": errors,
        "dry_run": dry_run,
    }

# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

async def main():
    parser = argparse.ArgumentParser(
        description="Backfill historical stats from API-Football"
    )
    parser.add_argument(
        "--mode", required=True, choices=["pilot", "backfill"],
        help="pilot: sample to validate coverage; backfill: full update"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Report only, no DB writes or API calls (pilot) / no DB writes (backfill)"
    )
    parser.add_argument("--league-id", type=int, help="Filter by league ID")
    parser.add_argument("--season", type=int, help="Filter by season year")
    parser.add_argument("--checkpoint", help="Checkpoint file for resume")
    parser.add_argument("--pilot-csv", help="Pilot results CSV (for backfill mode)")
    parser.add_argument(
        "--force-all", action="store_true",
        help="Backfill ALL segments without pilot CSV (use with caution)"
    )
    parser.add_argument(
        "--rate-limit", type=int, default=75,
        help="Requests per minute (default: 75)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=200,
        help="Fixtures per segment in pilot (default: 200)"
    )
    parser.add_argument("--output-csv", help="Export results to CSV")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )

    # ATI Safety: backfill mode requires --pilot-csv or --force-all
    if args.mode == "backfill" and not args.pilot_csv and not args.force_all:
        logging.error(
            "Backfill mode requires --pilot-csv or --force-all flag. "
            "Run pilot first to validate coverage, then use --pilot-csv with GO segments."
        )
        sys.exit(1)

    # DB connection
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        logging.error("DATABASE_URL not set")
        sys.exit(1)

    if database_url.startswith("postgresql://"):
        database_url = database_url.replace("postgresql://", "postgresql+asyncpg://")

    engine = create_async_engine(database_url)
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    rate_limiter = RateLimiter(args.rate_limit)

    async with async_session() as db_session:
        async with aiohttp.ClientSession() as http_session:
            if args.mode == "pilot":
                result = await run_pilot(
                    db_session, http_session, rate_limiter,
                    args.league_id, args.season, args.batch_size,
                    args.output_csv, args.dry_run
                )
            else:
                result = await run_backfill(
                    db_session, http_session, rate_limiter,
                    args.league_id, args.season, args.checkpoint,
                    args.pilot_csv, args.force_all, args.dry_run, args.output_csv
                )

    await engine.dispose()

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
```

---

## Flujo de Ejecución

```
┌─────────────────────────────────────────────────────────────────┐
│ PASO 1: PILOT DRY-RUN (ver segmentos sin llamar API)            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  python scripts/backfill_historical_stats.py \                  │
│    --mode pilot \                                               │
│    --dry-run \                                                  │
│    --output-csv logs/pilot_dryrun.csv                           │
│                                                                  │
│  Output: Lista de segmentos (liga, temporada) con total_missing │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ PASO 2: PILOT REAL (sample 200 fixtures por segmento)           │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  python scripts/backfill_historical_stats.py \                  │
│    --mode pilot \                                               │
│    --output-csv logs/pilot_results.csv                          │
│                                                                  │
│  Output: CSV con decision=GO/HOLD/STOP por segmento             │
│                                                                  │
│  Revisar resultados:                                            │
│    - GO (>=70%): Proceder a backfill                            │
│    - HOLD (50-70%): Revisar manualmente                         │
│    - STOP (<50%): Candidato a tainted                           │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ PASO 3: BACKFILL SEGMENTOS GO                                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  python scripts/backfill_historical_stats.py \                  │
│    --mode backfill \                                            │
│    --pilot-csv logs/pilot_results.csv \                         │
│    --checkpoint logs/backfill_cp.json \                         │
│    --output-csv logs/backfill_results.csv                       │
│                                                                  │
│  Solo procesa segmentos marcados GO en el pilot                 │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ PASO 4: MARCAR TAINTED (solo con evidencia)                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Para segmentos STOP con evidencia del piloto:                  │
│                                                                  │
│  -- Ejemplo: Venezuela confirmado sin datos                     │
│  UPDATE matches                                                 │
│  SET tainted = true,                                            │
│      tainted_evidence = '{"reason": "api_no_data",              │
│                           "pilot_date": "2026-02-01",           │
│                           "sampled": 200,                       │
│                           "no_data_pct": 100}'::jsonb           │
│  WHERE league_id = 299                                          │
│    AND (stats IS NULL OR stats->'home'->>'total_shots' IS NULL);│
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Verificación Post-Backfill

```sql
-- Comparar cobertura antes/después
SELECT
    al.name,
    al.country,
    COUNT(*) as total_ft,
    COUNT(CASE WHEN m.stats->'home'->>'total_shots' IS NOT NULL THEN 1 END) as with_shots,
    ROUND(100.0 * COUNT(CASE WHEN m.stats->'home'->>'total_shots' IS NOT NULL THEN 1 END) / COUNT(*), 1) as pct
FROM matches m
JOIN admin_leagues al ON m.league_id = al.league_id
WHERE m.status = 'FT' AND al.kind = 'league'
GROUP BY al.name, al.country
ORDER BY pct;

-- Verificar home/away correctamente mapeados
SELECT
    m.id,
    th.name as home_team,
    ta.name as away_team,
    m.home_goals || '-' || m.away_goals as score,
    m.stats->'home'->>'total_shots' as home_shots,
    m.stats->'away'->>'total_shots' as away_shots
FROM matches m
JOIN teams th ON m.home_team_id = th.id
JOIN teams ta ON m.away_team_id = ta.id
WHERE m.stats->'home'->>'total_shots' IS NOT NULL
ORDER BY m.date DESC
LIMIT 20;
```

---

## Recursos

| Métrica | Valor |
|---------|-------|
| Budget API diario | 75,000 requests |
| Rate limit script | 75 req/min (configurable) |
| Pilot: fixtures por segmento | 200 |
| Backfill: tiempo estimado | Depende de segmentos GO |

---

## Referencias

- Dashboard SOTA/Features: Vista de cobertura actual por liga/temporada
- API-Football docs: https://www.api-football.com/documentation-v3
