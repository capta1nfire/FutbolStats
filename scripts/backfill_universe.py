#!/usr/bin/env python3
"""
Unified API-Football Backfill Runner for FutbolStats.

Single entry-point for all API-Football historical data ingestion.
Replaces 20+ fragmented backfill scripts with one controlled runner.

Subcommands:
  discover    Enumerate coverage gaps (0 API calls, DB reads only)
  fixtures    Upsert matches + teams per (league, season)
  details     Backfill stats/events/lineups/player_stats for FT matches
  standings   League standings per (league, season)
  injuries    Player injuries per (league, season)
  verify      Coverage queries + sanity checks (0 API calls)

Usage:
  source .env
  python scripts/backfill_universe.py discover --scope admin_leagues
  python scripts/backfill_universe.py fixtures --scope league_ids=39 --from-season 2024
  python scripts/backfill_universe.py details --what stats events --scope admin_leagues --from-season 2023
  python scripts/backfill_universe.py standings --scope admin_leagues
  python scripts/backfill_universe.py injuries --scope admin_leagues --from-season 2023
  python scripts/backfill_universe.py verify --scope admin_leagues --from-season 2023

Guardrails:
  - Uses APIFootballProvider for ALL HTTP calls (centralized rate limit + budget)
  - Idempotent: ON CONFLICT in all writes. Re-runnable without duplicates.
  - Checkpoints in data/backfill/ for resume after interruption.
  - BudgetGuard per run + Provider daily budget (150K).
  - SIGINT handler saves checkpoint on Ctrl+C.
  - Short transactions: commit per batch, not per league.

ABE-approved design: 2026-02-19. Consolidates legacy scripts.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import os
import re
import signal
import sys
import time
from collections import Counter
from datetime import datetime, timedelta
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("backfill_universe")
logging.getLogger("httpx").setLevel(logging.WARNING)

# ============================================================================
# Constants
# ============================================================================

# Leagues without standings tables (cups, tournaments, knockouts)
# Ported from backfill_standings.py
NO_TABLE_LEAGUES = {
    2, 3, 848,         # UEFA club competitions
    45, 143,            # Domestic cups
    13, 11,             # CONMEBOL club competitions
    29, 30, 31, 32, 33, 34, 37,  # World Cup qualifiers
    1, 4, 5, 7, 9, 10, 28,      # International tournaments/friendlies
}

NO_TABLE_TTL_DAYS = 30

# ============================================================================
# SQL Constants (ported from existing scripts)
# ============================================================================

LINEUP_UPSERT_SQL = """
    INSERT INTO match_lineups (
        match_id, team_id, is_home, formation,
        starting_xi_ids, starting_xi_names, starting_xi_positions,
        substitutes_ids, substitutes_names,
        coach_id, coach_name, source, created_at
    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, NOW())
    ON CONFLICT (match_id, team_id) DO NOTHING
"""

PLAYER_STATS_UPSERT_SQL = """
    INSERT INTO match_player_stats (
        match_id, player_external_id, player_name,
        team_external_id, team_id, match_date,
        rating, minutes, position, is_substitute, is_captain,
        goals, assists, saves,
        shots_total, shots_on_target,
        passes_total, passes_key, passes_accuracy,
        tackles, interceptions, blocks,
        duels_total, duels_won,
        dribbles_attempts, dribbles_success,
        fouls_drawn, fouls_committed,
        yellow_cards, red_cards,
        raw_json, captured_at
    ) VALUES (
        $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11,
        $12, $13, $14, $15, $16, $17, $18, $19, $20, $21,
        $22, $23, $24, $25, $26, $27, $28, $29, $30, $31, NOW()
    )
    ON CONFLICT (match_id, player_external_id) DO UPDATE SET
        rating = EXCLUDED.rating,
        minutes = EXCLUDED.minutes,
        position = EXCLUDED.position,
        is_substitute = EXCLUDED.is_substitute,
        is_captain = EXCLUDED.is_captain,
        goals = EXCLUDED.goals,
        assists = EXCLUDED.assists,
        saves = EXCLUDED.saves,
        shots_total = EXCLUDED.shots_total,
        shots_on_target = EXCLUDED.shots_on_target,
        passes_total = EXCLUDED.passes_total,
        passes_key = EXCLUDED.passes_key,
        passes_accuracy = EXCLUDED.passes_accuracy,
        tackles = EXCLUDED.tackles,
        interceptions = EXCLUDED.interceptions,
        blocks = EXCLUDED.blocks,
        duels_total = EXCLUDED.duels_total,
        duels_won = EXCLUDED.duels_won,
        dribbles_attempts = EXCLUDED.dribbles_attempts,
        dribbles_success = EXCLUDED.dribbles_success,
        fouls_drawn = EXCLUDED.fouls_drawn,
        fouls_committed = EXCLUDED.fouls_committed,
        yellow_cards = EXCLUDED.yellow_cards,
        red_cards = EXCLUDED.red_cards,
        raw_json = EXCLUDED.raw_json,
        captured_at = NOW()
"""

INJURY_UPSERT_SQL = """
    INSERT INTO player_injuries (
        player_external_id, player_name, team_id, league_id, season,
        fixture_external_id, match_id, injury_type, injury_reason,
        fixture_date, raw_json
    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
    ON CONFLICT (player_external_id, fixture_external_id)
    DO UPDATE SET
        injury_type = EXCLUDED.injury_type,
        injury_reason = EXCLUDED.injury_reason,
        team_id = COALESCE(EXCLUDED.team_id, player_injuries.team_id),
        match_id = COALESCE(EXCLUDED.match_id, player_injuries.match_id),
        raw_json = EXCLUDED.raw_json
"""

STANDINGS_UPSERT_SQL = """
    INSERT INTO league_standings (league_id, season, standings, captured_at, expires_at, source)
    VALUES ($1, $2, $3::json, NOW(), $4, 'backfill_universe')
    ON CONFLICT (league_id, season)
    DO UPDATE SET standings = EXCLUDED.standings, captured_at = NOW(),
                  expires_at = EXCLUDED.expires_at, source = 'backfill_universe'
"""

STANDINGS_NO_TABLE_SQL = """
    INSERT INTO league_standings (league_id, season, standings, captured_at, expires_at, source)
    VALUES ($1, $2, NULL, NOW(), $3, 'no_table')
    ON CONFLICT (league_id, season)
    DO UPDATE SET standings = NULL, captured_at = NOW(),
                  expires_at = EXCLUDED.expires_at, source = 'no_table'
"""

# ============================================================================
# Infrastructure: CheckpointManager
# ============================================================================

class CheckpointManager:
    """Persist progress to data/backfill/ckpt_{command}_{scope_hash}.json."""

    def __init__(self, command: str, scope: str, base_dir: str = "data/backfill"):
        self.base_dir = base_dir
        scope_hash = hashlib.md5(scope.encode()).hexdigest()[:8]
        self.path = os.path.join(base_dir, f"ckpt_{command}_{scope_hash}.json")
        self.data: dict = {}

    def load(self) -> dict:
        if os.path.exists(self.path):
            with open(self.path) as f:
                self.data = json.load(f)
            logger.info("Loaded checkpoint: %s (%d completed keys)",
                        self.path, len(self.data.get("completed_keys", [])))
        return self.data

    def save(self):
        os.makedirs(self.base_dir, exist_ok=True)
        self.data["updated_at"] = datetime.utcnow().isoformat()
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.data, f, indent=2)
        os.replace(tmp, self.path)

    def is_completed(self, key: str) -> bool:
        return key in self.data.get("completed_keys", [])

    def mark_completed(self, key: str):
        self.data.setdefault("completed_keys", []).append(key)

    def get_last_match_id(self) -> int:
        return self.data.get("last_match_id", 0)

    def set_last_match_id(self, match_id: int):
        self.data["last_match_id"] = match_id


# ============================================================================
# Infrastructure: BudgetGuard
# ============================================================================

class BudgetGuard:
    """Local per-run request counter enforcing --max-requests."""

    def __init__(self, max_requests: int):
        self.max_requests = max_requests
        self.used = 0
        self._lock = asyncio.Lock()

    async def consume(self, cost: int = 1) -> bool:
        async with self._lock:
            if self.used + cost > self.max_requests:
                return False
            self.used += cost
            return True

    @property
    def remaining(self) -> int:
        return max(0, self.max_requests - self.used)


# ============================================================================
# Infrastructure: GlobalRateLimiter (ported from backfill_player_stats.py)
# ============================================================================

class GlobalRateLimiter:
    """Token-bucket rate limiter — at most `rps` requests per second."""

    def __init__(self, rps: float):
        self.interval = 1.0 / rps
        self.last_time = 0.0
        self._lock = asyncio.Lock()

    async def acquire(self):
        async with self._lock:
            now = time.monotonic()
            wait = self.last_time + self.interval - now
            if wait > 0:
                await asyncio.sleep(wait)
            self.last_time = time.monotonic()


# ============================================================================
# Scope Resolver
# ============================================================================

async def resolve_leagues(conn, scope: str) -> list[int]:
    """Parse --scope flag and return list of league_ids."""
    if scope == "all":
        from app.etl.competitions import COMPETITIONS
        return sorted(COMPETITIONS.keys())

    if scope == "admin_leagues":
        rows = await conn.fetch(
            "SELECT league_id FROM admin_leagues WHERE kind = 'league' AND is_active = true ORDER BY league_id"
        )
        return [r["league_id"] for r in rows]

    if scope.startswith("league_ids="):
        ids_str = scope.split("=", 1)[1]
        return sorted(int(x.strip()) for x in ids_str.split(","))

    if scope.startswith("country="):
        country = scope.split("=", 1)[1]
        rows = await conn.fetch(
            "SELECT league_id FROM admin_leagues WHERE country = $1 ORDER BY league_id",
            country,
        )
        return [r["league_id"] for r in rows]

    raise ValueError(f"Invalid --scope: {scope!r}. Use: admin_leagues | all | league_ids=39,61 | country=England")


def resolve_seasons(from_season: int, to_season: int | None) -> list[int]:
    """Return inclusive list of season years."""
    if to_season is None:
        to_season = datetime.utcnow().year
    return list(range(from_season, to_season + 1))


# ============================================================================
# Parsing & Sanitization (ported from backfill_player_stats.py — GDT #2)
# ============================================================================

def safe_int(val):
    """Parse various int representations: 90, "90", "90+4", None → int or None."""
    if val is None:
        return None
    s = str(val).strip()
    if not s or s == "-":
        return None
    s = s.split("+")[0].strip()
    try:
        return int(s)
    except (ValueError, TypeError):
        return None


def safe_rating(val):
    """Parse rating: "7.3"→7.3, "-"→None, None→None."""
    if val is None:
        return None
    s = str(val).strip()
    if not s or s == "-":
        return None
    try:
        return round(float(s), 2)
    except (ValueError, TypeError):
        return None


def safe_accuracy(val):
    """Parse passes_accuracy: "68%"→68, "68"→68, None→None."""
    if val is None:
        return None
    s = str(val).strip().rstrip("%")
    if not s or s == "-":
        return None
    try:
        return int(round(float(s)))
    except (ValueError, TypeError):
        return None


def parse_player_row(player_data, match_id, match_date, team_ext_id, team_id):
    """Parse one player entry from API response into a tuple for UPSERT.

    Ghost filter (GDT #2): minutes=0/NULL → rating forced to NULL.
    raw_json stores per-player statistics[0] only (ABE P0 #3).
    """
    player = player_data.get("player") or {}
    stats_list = player_data.get("statistics") or []
    stats = stats_list[0] if stats_list else {}

    player_ext_id = player.get("id")
    if not player_ext_id:
        return None

    games = stats.get("games") or {}
    goals_data = stats.get("goals") or {}
    passes_data = stats.get("passes") or {}
    tackles_data = stats.get("tackles") or {}
    duels_data = stats.get("duels") or {}
    dribbles_data = stats.get("dribbles") or {}
    fouls_data = stats.get("fouls") or {}
    cards_data = stats.get("cards") or {}
    shots_data = stats.get("shots") or {}

    minutes = safe_int(games.get("minutes"))
    rating = safe_rating(games.get("rating"))

    # Ghost filter: bench warmers → force rating NULL
    if minutes is None or minutes == 0:
        rating = None

    raw = {"player_id": player_ext_id, "statistics": stats}

    return (
        match_id, player_ext_id, player.get("name"),
        team_ext_id, team_id, match_date,
        rating, minutes, games.get("position"),
        games.get("substitute"), games.get("captain"),
        safe_int(goals_data.get("total")), safe_int(goals_data.get("assists")),
        safe_int(goals_data.get("saves")),
        safe_int(shots_data.get("total")), safe_int(shots_data.get("on")),
        safe_int(passes_data.get("total")), safe_int(passes_data.get("key")),
        safe_accuracy(passes_data.get("accuracy")),
        safe_int(tackles_data.get("total")), safe_int(tackles_data.get("interceptions")),
        safe_int(tackles_data.get("blocks")),
        safe_int(duels_data.get("total")), safe_int(duels_data.get("won")),
        safe_int(dribbles_data.get("attempts")), safe_int(dribbles_data.get("success")),
        safe_int(fouls_data.get("drawn")), safe_int(fouls_data.get("committed")),
        safe_int(cards_data.get("yellow")), safe_int(cards_data.get("red")),
        json.dumps(raw),
    )


def deduplicate_players(rows):
    """Deduplicate player rows in memory (GDT #1).

    Keep the entry with non-null rating, or more minutes.
    """
    seen = {}
    for row in rows:
        key = row[1]  # player_external_id
        if key in seen:
            existing = seen[key]
            if existing[6] is None and row[6] is not None:
                seen[key] = row
            elif row[7] is not None and (existing[7] is None or row[7] > existing[7]):
                seen[key] = row
        else:
            seen[key] = row
    return list(seen.values())


def build_lineup_rows(lineups, match_id, home_team_id, away_team_id):
    """Build tuple rows for LINEUP_UPSERT_SQL.

    Ported from backfill_match_lineups.py.
    """
    rows = []
    for side, is_home in [("home", True), ("away", False)]:
        lineup = lineups.get(side)
        if not lineup:
            continue

        team_id = home_team_id if is_home else away_team_id
        xi = lineup.get("starting_xi", [])
        subs = lineup.get("substitutes", [])
        coach = lineup.get("coach") or {}

        rows.append((
            match_id, team_id, is_home, lineup.get("formation"),
            [p["id"] for p in xi], [p["name"] for p in xi],
            [p.get("pos", "") for p in xi],
            [p["id"] for p in subs], [p["name"] for p in subs],
            coach.get("id"), coach.get("name"), "api-football",
        ))
    return rows


def build_injury_rows(injuries, team_map, match_map, league_id, season):
    """Build tuple rows for INJURY_UPSERT_SQL.

    Ported from backfill_players_managers.py.
    """
    rows = []
    for entry in injuries:
        player = entry.get("player", {})
        team_data = entry.get("team", {})
        fixture = entry.get("fixture", {})

        player_ext_id = player.get("id")
        fixture_ext_id = fixture.get("id")
        if not player_ext_id or not fixture_ext_id:
            continue

        team_ext_id = team_data.get("id")
        team_internal_id = team_map.get(team_ext_id)
        match_internal_id = match_map.get(fixture_ext_id)

        fixture_date = None
        fixture_date_raw = fixture.get("date")
        if fixture_date_raw:
            try:
                dt = datetime.fromisoformat(fixture_date_raw.replace("Z", "+00:00"))
                fixture_date = dt.replace(tzinfo=None)
            except (ValueError, AttributeError):
                pass

        player_name = player.get("name") or f"Player#{player_ext_id}"

        rows.append((
            player_ext_id, player_name, team_internal_id,
            league_id, season, fixture_ext_id, match_internal_id,
            player.get("type") or "Unknown", player.get("reason"),
            fixture_date, json.dumps(entry),
        ))
    return rows


# ============================================================================
# DB Reconnection helper (IDEAM lesson: long runs kill connections)
# ============================================================================

async def ensure_connection(conn):
    """Test connection; reconnect if dead. Returns (conn, reconnected)."""
    try:
        await conn.fetchval("SELECT 1")
        return conn, False
    except Exception:
        logger.warning("DB connection lost, reconnecting...")
        try:
            await conn.close()
        except Exception:
            pass
        db_url = os.environ.get("DATABASE_URL", "")
        if "+asyncpg" in db_url:
            db_url = db_url.replace("postgresql+asyncpg://", "postgresql://")
        import asyncpg
        new_conn = await asyncpg.connect(db_url)
        logger.info("Reconnected to database")
        return new_conn, True


# ============================================================================
# Subcommand: discover
# ============================================================================

async def cmd_discover(conn, args, leagues, seasons):
    """Enumerate coverage gaps. 0 API calls."""
    manifest = []

    for lid in leagues:
        name_row = await conn.fetchrow(
            "SELECT name FROM admin_leagues WHERE league_id = $1", lid
        )
        league_name = name_row["name"] if name_row else f"League {lid}"

        for season in seasons:
            row = await conn.fetchrow("""
                SELECT
                    COUNT(*) as total,
                    COUNT(*) FILTER (WHERE status IN ('FT','AET','PEN')) as ft,
                    COUNT(*) FILTER (WHERE stats IS NOT NULL AND stats::text != '{}' AND (stats->>'_no_stats') IS NULL) as has_stats,
                    COUNT(*) FILTER (WHERE events IS NOT NULL AND events::text NOT IN ('[]','null')) as has_events,
                    COUNT(*) FILTER (WHERE venue_name IS NOT NULL) as has_venue,
                    COUNT(*) FILTER (WHERE odds_home IS NOT NULL) as has_odds,
                    MIN(date)::date as min_date,
                    MAX(date)::date as max_date
                FROM matches
                WHERE league_id = $1 AND season = $2
            """, lid, season)

            lineup_count = await conn.fetchval("""
                SELECT COUNT(DISTINCT ml.match_id)
                FROM match_lineups ml JOIN matches m ON ml.match_id = m.id
                WHERE m.league_id = $1 AND m.season = $2
            """, lid, season)

            ps_count = await conn.fetchval("""
                SELECT COUNT(DISTINCT mps.match_id)
                FROM match_player_stats mps JOIN matches m ON mps.match_id = m.id
                WHERE m.league_id = $1 AND m.season = $2
            """, lid, season)

            total = row["total"]
            ft = row["ft"]
            if total == 0:
                continue

            entry = {
                "league_id": lid,
                "name": league_name,
                "season": season,
                "total": total,
                "ft": ft,
                "min_date": str(row["min_date"]) if row["min_date"] else None,
                "max_date": str(row["max_date"]) if row["max_date"] else None,
                "has_stats": row["has_stats"],
                "has_events": row["has_events"],
                "has_lineups": lineup_count,
                "has_player_stats": ps_count,
                "has_venue": row["has_venue"],
                "has_odds": row["has_odds"],
                "gaps": {
                    "stats": ft - row["has_stats"],
                    "events": ft - row["has_events"],
                    "lineups": ft - lineup_count,
                    "player_stats": ft - ps_count,
                },
            }
            manifest.append(entry)

            # Log per-segment
            pct_s = f"{100*row['has_stats']/ft:.0f}%" if ft else "n/a"
            pct_e = f"{100*row['has_events']/ft:.0f}%" if ft else "n/a"
            pct_l = f"{100*lineup_count/ft:.0f}%" if ft else "n/a"
            pct_p = f"{100*ps_count/ft:.0f}%" if ft else "n/a"
            logger.info(
                "  %s (%d) season=%d: %d total, %d FT | stats=%s events=%s lineups=%s player_stats=%s",
                league_name, lid, season, total, ft, pct_s, pct_e, pct_l, pct_p,
            )

    # Summary
    total_gaps = Counter()
    for e in manifest:
        for k, v in e["gaps"].items():
            total_gaps[k] += v
    total_ft = sum(e["ft"] for e in manifest)

    logger.info(
        "\n=== DISCOVER SUMMARY ===\n"
        "  Leagues: %d | Seasons: %d segments | Total FT: %d\n"
        "  Gaps: stats=%d events=%d lineups=%d player_stats=%d\n"
        "  Estimated API calls:\n"
        "    stats+events: ~%d (2 per match with gap)\n"
        "    lineups+player_stats: ~%d (2 per match with gap)",
        len(leagues), len(manifest), total_ft,
        total_gaps["stats"], total_gaps["events"],
        total_gaps["lineups"], total_gaps["player_stats"],
        total_gaps["stats"] + total_gaps["events"],
        total_gaps["lineups"] + total_gaps["player_stats"],
    )

    # Save manifest
    out_path = os.path.join("data", "backfill", f"manifest_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info("Manifest saved to %s", out_path)

    return {"segments": len(manifest), "total_ft": total_ft, "gaps": dict(total_gaps)}


# ============================================================================
# Subcommand: fixtures
# ============================================================================

async def cmd_fixtures(conn, provider, args, leagues, seasons, ckpt, budget, limiter):
    """Upsert matches + teams per (league, season) via ETLPipeline."""
    from app.database import AsyncSessionLocal
    from app.etl.pipeline import ETLPipeline

    total_synced = 0
    total_teams = 0

    for lid in leagues:
        for season in seasons:
            key = f"{lid}_{season}"
            if args.resume and ckpt.is_completed(key):
                logger.info("SKIP (checkpoint): league=%d season=%d", lid, season)
                continue

            if not await budget.consume(1):
                logger.warning("Budget exhausted (%d used). Stopping.", budget.used)
                ckpt.save()
                return {"matches_synced": total_synced, "teams_synced": total_teams, "budget_exhausted": True}

            await limiter.acquire()

            if args.dry_run:
                logger.info("[DRY-RUN] fixtures league=%d season=%d", lid, season)
                ckpt.mark_completed(key)
                continue

            try:
                async with AsyncSessionLocal() as session:
                    pipeline = ETLPipeline(provider=provider, session=session)
                    result = await pipeline.sync_league(league_id=lid, season=season)
                    matches_n = result.get("matches_synced", result.get("matches", 0))
                    teams_n = result.get("teams_synced", result.get("teams", 0))
                    total_synced += matches_n
                    total_teams += teams_n
                    logger.info("league=%d season=%d: %d matches, %d teams", lid, season, matches_n, teams_n)
            except Exception as e:
                logger.error("league=%d season=%d: %s", lid, season, e)

            ckpt.mark_completed(key)
            ckpt.save()

    logger.info("=== FIXTURES COMPLETE: %d matches, %d teams ===", total_synced, total_teams)
    return {"matches_synced": total_synced, "teams_synced": total_teams}


# ============================================================================
# Subcommand: details
# ============================================================================

async def cmd_details(conn, provider, args, leagues, seasons, ckpt, budget, limiter):
    """Backfill stats/events/lineups/player_stats for FT matches."""
    what_set = set(args.what)
    CHUNK = 200

    # Build missing-data conditions (derived, not boolean columns)
    conditions = []
    if "stats" in what_set:
        conditions.append("(m.stats IS NULL OR m.stats::text = '{}' OR (m.stats->>'_no_stats') IS NOT NULL)")
    if "events" in what_set:
        conditions.append("(m.events IS NULL OR m.events::text IN ('[]','null'))")
    if "lineups" in what_set:
        conditions.append("NOT EXISTS (SELECT 1 FROM match_lineups ml WHERE ml.match_id = m.id)")
    if "player_stats" in what_set:
        conditions.append("NOT EXISTS (SELECT 1 FROM match_player_stats mps WHERE mps.match_id = m.id)")

    missing_filter = " OR ".join(conditions)

    # Build league filter
    league_placeholders = ", ".join(f"${i+3}" for i in range(len(leagues)))

    last_id = ckpt.get_last_match_id() if args.resume else 0
    min_date = datetime(args.from_season, 1, 1)

    query = f"""
        SELECT m.id as match_id, m.external_id, m.date::date as match_date,
               m.league_id, m.season, m.home_team_id, m.away_team_id,
               m.stats IS NOT NULL AND m.stats::text != '{{}}'
                   AND (m.stats->>'_no_stats') IS NULL as has_stats,
               m.events IS NOT NULL AND m.events::text NOT IN ('[]','null') as has_events,
               EXISTS (SELECT 1 FROM match_lineups ml WHERE ml.match_id = m.id) as has_lineups,
               EXISTS (SELECT 1 FROM match_player_stats mps WHERE mps.match_id = m.id) as has_player_stats
        FROM matches m
        WHERE m.status IN ('FT', 'AET', 'PEN')
          AND m.external_id IS NOT NULL
          AND m.date >= $1
          AND m.id > $2
          AND m.league_id IN ({league_placeholders})
          AND ({missing_filter})
        ORDER BY m.id
    """

    params = [min_date, last_id] + leagues
    matches = await conn.fetch(query, *params)

    if args.limit > 0:
        matches = matches[:args.limit]

    total = len(matches)
    logger.info("Found %d FT matches needing details (%s)", total, ", ".join(sorted(what_set)))

    if args.dry_run:
        # Estimate API calls
        calls_est = 0
        for m in matches:
            if "stats" in what_set and not m["has_stats"]:
                calls_est += 1
            if "events" in what_set and not m["has_events"]:
                calls_est += 1
            if "lineups" in what_set and not m["has_lineups"]:
                calls_est += 1
            if "player_stats" in what_set and not m["has_player_stats"]:
                calls_est += 1
        logger.info("[DRY-RUN] Would make ~%d API calls for %d matches. Exiting.", calls_est, total)
        return {"matches": total, "estimated_calls": calls_est}

    if total == 0:
        logger.info("Nothing to do.")
        return {"matches": 0}

    # Pre-load team map for player_stats
    team_map = {}
    if "player_stats" in what_set:
        team_rows = await conn.fetch("SELECT id, external_id FROM teams WHERE external_id IS NOT NULL")
        team_map = {r["external_id"]: r["id"] for r in team_rows}

    sem = asyncio.Semaphore(args.concurrency)
    counters = Counter()
    t0 = time.time()

    for chunk_start in range(0, total, CHUNK):
        chunk = matches[chunk_start:chunk_start + CHUNK]

        # Reconnect if needed on long runs
        conn, reconnected = await ensure_connection(conn)

        async def process_one(match):
            async with sem:
                ext_id = match["external_id"]
                mid = match["match_id"]

                # Count how many calls this match needs
                calls_needed = 0
                if "stats" in what_set and not match["has_stats"]:
                    calls_needed += 1
                if "events" in what_set and not match["has_events"]:
                    calls_needed += 1
                if "lineups" in what_set and not match["has_lineups"]:
                    calls_needed += 1
                if "player_stats" in what_set and not match["has_player_stats"]:
                    calls_needed += 1

                if not await budget.consume(calls_needed):
                    counters["budget_exhausted"] += 1
                    return

                try:
                    # Stats
                    if "stats" in what_set and not match["has_stats"]:
                        await limiter.acquire()
                        stats_data = await provider.get_fixture_statistics(ext_id)
                        if stats_data:
                            await conn.execute(
                                "UPDATE matches SET stats = $1::json WHERE id = $2",
                                json.dumps(stats_data), mid,
                            )
                            counters["stats_ok"] += 1
                        else:
                            counters["stats_no_data"] += 1

                    # Events
                    if "events" in what_set and not match["has_events"]:
                        await limiter.acquire()
                        events_data = await provider.get_fixture_events(ext_id)
                        if events_data is not None:
                            await conn.execute(
                                "UPDATE matches SET events = $1::json WHERE id = $2",
                                json.dumps(events_data), mid,
                            )
                            counters["events_ok"] += 1
                        else:
                            counters["events_no_data"] += 1

                    # Lineups
                    if "lineups" in what_set and not match["has_lineups"]:
                        await limiter.acquire()
                        lineups_data = await provider.get_lineups(ext_id)
                        if lineups_data and (lineups_data.get("home") or lineups_data.get("away")):
                            rows = build_lineup_rows(
                                lineups_data, mid,
                                match["home_team_id"], match["away_team_id"],
                            )
                            if rows:
                                await conn.executemany(LINEUP_UPSERT_SQL, rows)
                            counters["lineups_ok"] += 1
                        else:
                            counters["lineups_no_data"] += 1

                    # Player stats
                    if "player_stats" in what_set and not match["has_player_stats"]:
                        await limiter.acquire()
                        ps_data = await provider.get_fixture_players(ext_id)
                        if ps_data:
                            ps_rows = []
                            for team_block in ps_data:
                                team_info = team_block.get("team") or {}
                                t_ext_id = team_info.get("id")
                                t_id = team_map.get(t_ext_id)
                                for p in team_block.get("players") or []:
                                    row = parse_player_row(p, mid, match["match_date"], t_ext_id, t_id)
                                    if row:
                                        ps_rows.append(row)
                            ps_rows = deduplicate_players(ps_rows)
                            if ps_rows:
                                await conn.executemany(PLAYER_STATS_UPSERT_SQL, ps_rows)
                            counters["player_stats_ok"] += 1
                        else:
                            counters["player_stats_no_data"] += 1

                except Exception as e:
                    counters["errors"] += 1
                    if counters["errors"] <= 20:
                        logger.warning("Error match %d (ext=%s): %s", mid, ext_id, e)

        tasks = [process_one(m) for m in chunk]
        await asyncio.gather(*tasks, return_exceptions=True)

        # Checkpoint after each chunk
        last_processed_id = chunk[-1]["match_id"]
        ckpt.set_last_match_id(last_processed_id)
        ckpt.data["processed"] = chunk_start + len(chunk)
        ckpt.data["counters"] = dict(counters)
        ckpt.save()

        # Progress
        processed = chunk_start + len(chunk)
        elapsed = time.time() - t0
        rate = processed / elapsed if elapsed > 0 else 0
        eta_min = (total - processed) / rate / 60 if rate > 0 else 0
        logger.info(
            "[%d/%d] (%.1f%%) stats=%d events=%d lineups=%d ps=%d errors=%d "
            "budget=%d/%d [%.1f m/s ETA %.0fm]",
            processed, total, 100 * processed / total,
            counters["stats_ok"], counters["events_ok"],
            counters["lineups_ok"], counters["player_stats_ok"],
            counters["errors"], budget.used, budget.max_requests,
            rate, eta_min,
        )

        if counters["budget_exhausted"] > 0:
            logger.warning("Budget exhausted. Stopping.")
            break

    elapsed = time.time() - t0
    logger.info(
        "\n=== DETAILS COMPLETE ===\n"
        "  processed: %d matches in %.1fm\n"
        "  stats: %d ok, %d no_data\n"
        "  events: %d ok, %d no_data\n"
        "  lineups: %d ok, %d no_data\n"
        "  player_stats: %d ok, %d no_data\n"
        "  errors: %d",
        chunk_start + len(chunk) if matches else 0, elapsed / 60,
        counters["stats_ok"], counters["stats_no_data"],
        counters["events_ok"], counters["events_no_data"],
        counters["lineups_ok"], counters["lineups_no_data"],
        counters["player_stats_ok"], counters["player_stats_no_data"],
        counters["errors"],
    )
    return dict(counters)


# ============================================================================
# Subcommand: standings
# ============================================================================

async def cmd_standings(conn, provider, args, leagues, seasons, ckpt, budget, limiter):
    """Standings per (league, season). Excludes NO_TABLE_LEAGUES."""

    # Load DB-persisted no_table marks
    db_no_table = set()
    for season in seasons:
        rows = await conn.fetch(
            "SELECT league_id FROM league_standings WHERE season = $1 AND source = 'no_table' AND expires_at > NOW()",
            season,
        )
        db_no_table.update(r["league_id"] for r in rows)

    all_no_table = NO_TABLE_LEAGUES | db_no_table
    counters = Counter()

    for lid in leagues:
        if lid in all_no_table:
            logger.info("SKIP (no_table): league=%d", lid)
            counters["skipped_no_table"] += 1
            continue

        for season in seasons:
            key = f"{lid}_{season}"
            if args.resume and ckpt.is_completed(key):
                continue

            if not await budget.consume(1):
                logger.warning("Budget exhausted. Stopping.")
                ckpt.save()
                return dict(counters)

            await limiter.acquire()

            if args.dry_run:
                logger.info("[DRY-RUN] standings league=%d season=%d", lid, season)
                ckpt.mark_completed(key)
                continue

            try:
                standings = await provider.get_standings(lid, season)
                if standings:
                    expires_at = datetime.utcnow() + timedelta(hours=6)
                    await conn.execute(
                        STANDINGS_UPSERT_SQL,
                        lid, season, json.dumps(standings), expires_at,
                    )
                    counters["fetched"] += 1
                    logger.info("league=%d season=%d: %d teams", lid, season, len(standings))
                else:
                    # Mark as no_table (TTL 30 days)
                    expires_at = datetime.utcnow() + timedelta(days=NO_TABLE_TTL_DAYS)
                    await conn.execute(STANDINGS_NO_TABLE_SQL, lid, season, expires_at)
                    counters["marked_no_table"] += 1
                    logger.info("league=%d season=%d: no data, marked no_table", lid, season)
            except Exception as e:
                counters["errors"] += 1
                logger.error("league=%d season=%d: %s", lid, season, e)

            ckpt.mark_completed(key)
            ckpt.save()

    logger.info("=== STANDINGS COMPLETE: fetched=%d no_table=%d errors=%d ===",
                counters["fetched"], counters["marked_no_table"] + counters["skipped_no_table"],
                counters["errors"])
    return dict(counters)


# ============================================================================
# Subcommand: injuries
# ============================================================================

async def cmd_injuries(conn, provider, args, leagues, seasons, ckpt, budget, limiter):
    """Injuries per (league, season)."""

    # Pre-load maps
    team_rows = await conn.fetch("SELECT id, external_id FROM teams WHERE external_id IS NOT NULL")
    team_map = {r["external_id"]: r["id"] for r in team_rows}

    match_rows = await conn.fetch("SELECT id, external_id FROM matches WHERE external_id IS NOT NULL")
    match_map = {r["external_id"]: r["id"] for r in match_rows}
    logger.info("Loaded %d teams, %d matches for ID resolution", len(team_map), len(match_map))

    counters = Counter()

    for lid in leagues:
        for season in seasons:
            key = f"{lid}_{season}"
            if args.resume and ckpt.is_completed(key):
                continue

            if not await budget.consume(1):
                logger.warning("Budget exhausted. Stopping.")
                ckpt.save()
                return dict(counters)

            await limiter.acquire()

            if args.dry_run:
                logger.info("[DRY-RUN] injuries league=%d season=%d", lid, season)
                ckpt.mark_completed(key)
                continue

            try:
                injuries = await provider.get_injuries(lid, season)

                if not injuries:
                    logger.info("league=%d season=%d: 0 injuries", lid, season)
                    ckpt.mark_completed(key)
                    ckpt.save()
                    counters["empty"] += 1
                    continue

                rows = build_injury_rows(injuries, team_map, match_map, lid, season)
                if rows:
                    await conn.executemany(INJURY_UPSERT_SQL, rows)
                    counters["upserted"] += len(rows)
                logger.info("league=%d season=%d: %d injuries", lid, season, len(rows))
            except Exception as e:
                counters["errors"] += 1
                logger.error("league=%d season=%d: %s", lid, season, e)

            ckpt.mark_completed(key)
            ckpt.save()

    logger.info("=== INJURIES COMPLETE: upserted=%d empty=%d errors=%d ===",
                counters["upserted"], counters["empty"], counters["errors"])
    return dict(counters)


# ============================================================================
# Subcommand: verify
# ============================================================================

async def cmd_verify(conn, args, leagues, seasons):
    """Coverage queries + sanity checks. 0 API calls."""
    report = {"generated_at": datetime.utcnow().isoformat(), "leagues": []}

    for lid in leagues:
        name_row = await conn.fetchrow(
            "SELECT name FROM admin_leagues WHERE league_id = $1", lid
        )
        league_name = name_row["name"] if name_row else f"League {lid}"
        league_report = {"league_id": lid, "name": league_name, "seasons": {}}

        for season in seasons:
            row = await conn.fetchrow("""
                SELECT
                    COUNT(*) as total,
                    COUNT(*) FILTER (WHERE status IN ('FT','AET','PEN')) as ft,
                    COUNT(*) FILTER (WHERE status = 'NS') as ns,
                    COUNT(*) FILTER (WHERE stats IS NOT NULL AND stats::text != '{}' AND (stats->>'_no_stats') IS NULL) as has_stats,
                    COUNT(*) FILTER (WHERE events IS NOT NULL AND events::text NOT IN ('[]','null')) as has_events,
                    COUNT(*) FILTER (WHERE venue_name IS NOT NULL) as has_venue,
                    COUNT(*) FILTER (WHERE odds_home IS NOT NULL) as has_odds
                FROM matches
                WHERE league_id = $1 AND season = $2
            """, lid, season)

            lineup_count = await conn.fetchval("""
                SELECT COUNT(DISTINCT ml.match_id)
                FROM match_lineups ml JOIN matches m ON ml.match_id = m.id
                WHERE m.league_id = $1 AND m.season = $2
            """, lid, season)

            ps_count = await conn.fetchval("""
                SELECT COUNT(DISTINCT mps.match_id)
                FROM match_player_stats mps JOIN matches m ON mps.match_id = m.id
                WHERE m.league_id = $1 AND m.season = $2
            """, lid, season)

            total = row["total"]
            ft = row["ft"]
            if total == 0:
                continue

            season_data = {
                "total": total, "ft": ft, "ns": row["ns"],
                "stats_pct": round(100 * row["has_stats"] / ft, 1) if ft else 0,
                "events_pct": round(100 * row["has_events"] / ft, 1) if ft else 0,
                "lineups_pct": round(100 * lineup_count / ft, 1) if ft else 0,
                "player_stats_pct": round(100 * ps_count / ft, 1) if ft else 0,
                "venue_pct": round(100 * row["has_venue"] / total, 1) if total else 0,
                "odds_pct": round(100 * row["has_odds"] / total, 1) if total else 0,
            }
            league_report["seasons"][str(season)] = season_data

            logger.info(
                "  %s (%d) %d: %d FT | stats=%s%% events=%s%% lineups=%s%% ps=%s%% venue=%s%% odds=%s%%",
                league_name, lid, season, ft,
                season_data["stats_pct"], season_data["events_pct"],
                season_data["lineups_pct"], season_data["player_stats_pct"],
                season_data["venue_pct"], season_data["odds_pct"],
            )

        if league_report["seasons"]:
            report["leagues"].append(league_report)

    # Sanity checks
    checks = []

    orphans = await conn.fetchval("SELECT COUNT(*) FROM matches WHERE external_id IS NULL")
    checks.append({"check": "matches_without_external_id", "count": orphans, "status": "OK" if orphans == 0 else "WARN"})

    null_goals = await conn.fetchval("""
        SELECT COUNT(*) FROM matches
        WHERE status IN ('FT','AET','PEN') AND (home_goals IS NULL OR away_goals IS NULL)
    """)
    checks.append({"check": "ft_without_goals", "count": null_goals, "status": "OK" if null_goals == 0 else "CRITICAL"})

    dupes = await conn.fetchval("""
        SELECT COUNT(*) FROM (
            SELECT external_id FROM matches WHERE external_id IS NOT NULL
            GROUP BY external_id HAVING COUNT(*) > 1
        ) x
    """)
    checks.append({"check": "duplicate_external_ids", "count": dupes, "status": "OK" if dupes == 0 else "CRITICAL"})

    orphan_lineups = await conn.fetchval("""
        SELECT COUNT(*) FROM match_lineups ml
        WHERE NOT EXISTS (SELECT 1 FROM matches m WHERE m.id = ml.match_id)
    """)
    checks.append({"check": "orphan_lineups", "count": orphan_lineups, "status": "OK" if orphan_lineups == 0 else "WARN"})

    report["sanity"] = checks

    logger.info("\n=== SANITY CHECKS ===")
    for c in checks:
        logger.info("  %s: %s (%d)", c["check"], c["status"], c["count"])

    # Save report
    if args.output:
        out_path = args.output
    else:
        out_path = os.path.join("data", "backfill", f"verify_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Report saved to %s", out_path)

    return report


# ============================================================================
# Job Recording (job_runs table)
# ============================================================================

async def record_job_start(conn, job_name: str, args) -> int:
    """Insert into job_runs, return id."""
    if args.dry_run:
        return 0
    try:
        row = await conn.fetchrow("""
            INSERT INTO job_runs (job_name, status, started_at, created_at, metrics)
            VALUES ($1, 'running', NOW(), NOW(), $2)
            RETURNING id
        """, job_name, json.dumps({"scope": args.scope, "command": args.command}))
        return row["id"]
    except Exception as e:
        logger.warning("Failed to record job start: %s", e)
        return 0


async def record_job_finish(conn, job_id: int, status: str, duration_ms: int,
                            metrics=None, error=None):
    """Update job_runs with result."""
    if job_id == 0:
        return
    try:
        await conn.execute("""
            UPDATE job_runs SET
                status = $2, finished_at = NOW(), duration_ms = $3,
                metrics = $4, error_message = $5
            WHERE id = $1
        """, job_id, status, duration_ms,
            json.dumps(metrics) if metrics else None, error)
    except Exception as e:
        logger.warning("Failed to record job finish: %s", e)


# ============================================================================
# CLI
# ============================================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="backfill_universe",
        description="Unified API-Football backfill runner for FutbolStats",
    )

    # Global flags
    parser.add_argument("--scope", default="admin_leagues",
                        help="admin_leagues | all | league_ids=39,61 | country=England")
    parser.add_argument("--from-season", type=int, default=2023)
    parser.add_argument("--to-season", type=int, default=None,
                        help="Default: current year")
    parser.add_argument("--concurrency", type=int, default=5,
                        help="Max concurrent API calls (default: 5)")
    parser.add_argument("--rps", type=float, default=8.0,
                        help="Max requests per second (default: 8)")
    parser.add_argument("--max-requests", type=int, default=10000,
                        help="Budget cap per run (default: 10000)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Log only, no writes")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from checkpoint in data/backfill/")

    sub = parser.add_subparsers(dest="command", required=True)

    # discover
    sub.add_parser("discover", help="Enumerate coverage gaps (0 API calls)")

    # fixtures
    sub.add_parser("fixtures", help="Upsert matches + teams per (league, season)")

    # details
    p_details = sub.add_parser("details", help="Backfill stats/events/lineups/player_stats for FT matches")
    p_details.add_argument("--what", nargs="+",
                           choices=["stats", "events", "lineups", "player_stats"],
                           default=["stats", "events", "lineups", "player_stats"],
                           help="Detail types to backfill (default: all)")
    p_details.add_argument("--limit", type=int, default=0,
                           help="Max fixtures to process (0=all)")

    # standings
    sub.add_parser("standings", help="Backfill league standings per (league, season)")

    # injuries
    sub.add_parser("injuries", help="Backfill player injuries per (league, season)")

    # verify
    p_verify = sub.add_parser("verify", help="Coverage queries + sanity checks (0 API calls)")
    p_verify.add_argument("--output", default=None, help="Save report to this JSON path")

    return parser


# ============================================================================
# Main
# ============================================================================

async def main():
    import asyncpg
    from app.etl.api_football import APIFootballProvider

    parser = build_parser()
    args = parser.parse_args()

    seasons = resolve_seasons(args.from_season, args.to_season)

    # DB connection
    db_url = os.environ.get("DATABASE_URL", "")
    if not db_url:
        raise RuntimeError("DATABASE_URL must be set (use: source .env)")
    if "+asyncpg" in db_url:
        db_url = db_url.replace("postgresql+asyncpg://", "postgresql://")

    conn = await asyncpg.connect(db_url)
    logger.info("Connected to database")

    # SIGINT handler: save checkpoint
    ckpt = CheckpointManager(args.command, args.scope)
    interrupted = False

    def sigint_handler(signum, frame):
        nonlocal interrupted
        if not interrupted:
            interrupted = True
            logger.warning("SIGINT received — saving checkpoint and exiting...")
            ckpt.save()
            # Don't exit immediately; let the current batch finish
        else:
            # Second SIGINT: force exit
            sys.exit(1)

    signal.signal(signal.SIGINT, sigint_handler)

    try:
        leagues = await resolve_leagues(conn, args.scope)
        logger.info("Scope: %d leagues %s, seasons %s", len(leagues), leagues, seasons)

        # Checkpoint
        if args.resume:
            ckpt.load()

        # Provider (only for commands that need API calls)
        provider = None
        needs_provider = args.command in ("fixtures", "details", "standings", "injuries")
        if needs_provider:
            provider = APIFootballProvider()

        budget = BudgetGuard(args.max_requests)
        limiter = GlobalRateLimiter(args.rps)

        # Record job start
        t0 = time.time()
        job_id = await record_job_start(conn, f"backfill_{args.command}", args)

        result = None
        try:
            if args.command == "discover":
                result = await cmd_discover(conn, args, leagues, seasons)
            elif args.command == "fixtures":
                result = await cmd_fixtures(conn, provider, args, leagues, seasons, ckpt, budget, limiter)
            elif args.command == "details":
                result = await cmd_details(conn, provider, args, leagues, seasons, ckpt, budget, limiter)
            elif args.command == "standings":
                result = await cmd_standings(conn, provider, args, leagues, seasons, ckpt, budget, limiter)
            elif args.command == "injuries":
                result = await cmd_injuries(conn, provider, args, leagues, seasons, ckpt, budget, limiter)
            elif args.command == "verify":
                result = await cmd_verify(conn, args, leagues, seasons)

            elapsed_ms = int((time.time() - t0) * 1000)
            await record_job_finish(conn, job_id, "success", elapsed_ms, result)
            logger.info("Job completed in %.1fs. Budget used: %d/%d.",
                        (time.time() - t0), budget.used, budget.max_requests)

        except Exception as e:
            elapsed_ms = int((time.time() - t0) * 1000)
            await record_job_finish(conn, job_id, "error", elapsed_ms, error=str(e))
            logger.error("Job failed: %s", e, exc_info=True)
            raise

        finally:
            if provider:
                await provider.close()

    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(main())
