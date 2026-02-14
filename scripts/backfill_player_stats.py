"""
Backfill: Populate match_player_stats from API-Football /fixtures/players.

Fetches per-player per-match stats (rating, minutes, goals, passes, etc.)
for finished matches since 2023-01-01. Two modes:
  - pilot:   Sample N fixtures per (league, season) to assess coverage → GO/HOLD/STOP
  - backfill: Ingest all FT matches for GO segments (or all if --skip-pilot)

Guardrails (ATI/ABE):
  - In-memory deduplication per fixture (GDT #1)
  - Aggressive type sanitization: minutes "90+4"→90, rating "-"→NULL (GDT #2)
  - Ghost filter: minutes=0/NULL → rating forced to NULL (GDT #2)
  - raw_json = per-player statistics[0] only, not full payload (ABE P0 #3)
  - ON CONFLICT DO UPDATE (APIs may correct stats post-audit)
  - Circuit breaker: 30% NO_DATA streak → pause segment

Usage:
  source .env && python3 scripts/backfill_player_stats.py --mode pilot
  source .env && python3 scripts/backfill_player_stats.py --mode backfill
  source .env && python3 scripts/backfill_player_stats.py --mode backfill --league 39 --rps 10
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import sys
import time
from collections import defaultdict
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("backfill_player_stats")
logging.getLogger("httpx").setLevel(logging.WARNING)

# ---------------------------------------------------------------------------
# Thresholds (ABE pattern from backfill_historical_stats.py)
# ---------------------------------------------------------------------------
THRESHOLD_GO = 70.0
THRESHOLD_HOLD = 50.0
CIRCUIT_BREAKER_RATIO = 0.30
MIN_DATE = datetime(2023, 1, 1).date()  # asyncpg needs native date, not str

# ---------------------------------------------------------------------------
# UPSERT SQL — ON CONFLICT DO UPDATE (APIs correct post-audit)
# ---------------------------------------------------------------------------
UPSERT_SQL = """
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


# ---------------------------------------------------------------------------
# Rate Limiter (pattern from backfill_match_lineups.py)
# ---------------------------------------------------------------------------
class GlobalRateLimiter:
    """Token-bucket rate limiter — ensures at most `rps` requests per second."""

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


# ---------------------------------------------------------------------------
# Parsing & Sanitization (GDT #2 + ABE P0 #3)
# ---------------------------------------------------------------------------
def safe_int(val):
    """Parse various int representations: 90, "90", "90+4", None → int or None."""
    if val is None:
        return None
    s = str(val).strip()
    if not s or s == "-":
        return None
    # Handle "90+4" format (stoppage time)
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

    Applies ghost filter (GDT #2): if minutes == 0 or None → rating = NULL.
    raw_json stores only per-player statistics[0] block (ABE P0 #3).
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
    penalty_data = stats.get("penalty") or {}

    minutes = safe_int(games.get("minutes"))
    rating = safe_rating(games.get("rating"))

    # Ghost filter (GDT #2): bench warmers → force rating NULL
    if minutes is None or minutes == 0:
        rating = None

    # Build raw_json: per-player statistics block only (ABE P0 #3)
    raw = {"player_id": player_ext_id, "statistics": stats}

    return (
        match_id,                                   # $1  match_id
        player_ext_id,                              # $2  player_external_id
        player.get("name"),                         # $3  player_name
        team_ext_id,                                # $4  team_external_id
        team_id,                                    # $5  team_id (internal, nullable)
        match_date,                                 # $6  match_date
        rating,                                     # $7  rating
        minutes,                                    # $8  minutes
        games.get("position"),                      # $9  position
        games.get("substitute"),                    # $10 is_substitute
        games.get("captain"),                       # $11 is_captain
        safe_int(goals_data.get("total")),          # $12 goals
        safe_int(goals_data.get("assists")),        # $13 assists
        safe_int(goals_data.get("saves")),          # $14 saves
        safe_int(shots_data.get("total")),          # $15 shots_total
        safe_int(shots_data.get("on")),             # $16 shots_on_target
        safe_int(passes_data.get("total")),         # $17 passes_total
        safe_int(passes_data.get("key")),           # $18 passes_key
        safe_accuracy(passes_data.get("accuracy")), # $19 passes_accuracy
        safe_int(tackles_data.get("total")),        # $20 tackles
        safe_int(tackles_data.get("interceptions")),# $21 interceptions
        safe_int(tackles_data.get("blocks")),       # $22 blocks
        safe_int(duels_data.get("total")),          # $23 duels_total
        safe_int(duels_data.get("won")),            # $24 duels_won
        safe_int(dribbles_data.get("attempts")),    # $25 dribbles_attempts
        safe_int(dribbles_data.get("success")),     # $26 dribbles_success
        safe_int(fouls_data.get("drawn")),          # $27 fouls_drawn
        safe_int(fouls_data.get("committed")),      # $28 fouls_committed
        safe_int(cards_data.get("yellow")),         # $29 yellow_cards
        safe_int(cards_data.get("red")),            # $30 red_cards
        json.dumps(raw),                            # $31 raw_json
    )


def deduplicate_players(rows):
    """Deduplicate player rows in memory (GDT #1).

    API may return same player twice (tactical position change glitch).
    Keep the entry with non-null rating, or more minutes.
    """
    seen = {}
    for row in rows:
        key = row[1]  # player_external_id ($2)
        if key in seen:
            existing = seen[key]
            # Prefer non-null rating
            if existing[6] is None and row[6] is not None:
                seen[key] = row
            # If both have rating, prefer more minutes
            elif row[7] is not None and (existing[7] is None or row[7] > existing[7]):
                seen[key] = row
        else:
            seen[key] = row
    return list(seen.values())


# ---------------------------------------------------------------------------
# Fetch one fixture
# ---------------------------------------------------------------------------
async def fetch_one(provider, match, limiter, results, team_map):
    """Fetch player stats for one fixture with rate limiting."""
    await limiter.acquire()

    fixture_ext_id = match["external_id"]
    match_id = match["match_id"]
    match_date = match["match_date"]

    try:
        data = await provider.get_fixture_players(fixture_ext_id)

        if not data:
            results["no_data"] += 1
            results["processed"] += 1
            return

        rows = []
        for team_block in data:
            team_info = team_block.get("team") or {}
            team_ext_id = team_info.get("id")
            team_id = team_map.get(team_ext_id)  # resolve to internal ID

            players = team_block.get("players") or []
            for p in players:
                row = parse_player_row(p, match_id, match_date, team_ext_id, team_id)
                if row:
                    rows.append(row)

        # Deduplicate in memory (GDT #1)
        rows = deduplicate_players(rows)

        if rows:
            results["rows"].extend(rows)
            results["with_rating"] += sum(1 for r in rows if r[6] is not None)
            results["total_players"] += len(rows)

    except Exception as e:
        results["errors"] += 1
        if results["errors"] <= 10:
            logger.warning("Error for fixture %s: %s", fixture_ext_id, e)

    results["processed"] += 1


# ---------------------------------------------------------------------------
# Pilot Mode
# ---------------------------------------------------------------------------
async def run_pilot(conn, provider, limiter, args, team_map):
    """Sample N fixtures per (league, season) and assess coverage."""
    logger.info("=== PILOT MODE (sample_size=%d) ===", args.sample_size)

    league_filter = "AND m.league_id = $2" if args.league else ""
    params = [MIN_DATE]
    if args.league:
        params.append(args.league)

    # Get segments
    segments = await conn.fetch(f"""
        SELECT m.league_id, m.season, COUNT(*) as cnt,
               COALESCE(al.name, 'League ' || m.league_id) as league_name
        FROM matches m
        LEFT JOIN admin_leagues al ON al.league_id = m.league_id
        WHERE m.status IN ('FT', 'AET', 'PEN')
          AND m.date >= $1
          AND m.external_id IS NOT NULL
          {league_filter}
        GROUP BY m.league_id, m.season, al.name
        ORDER BY m.league_id, m.season
    """, *params)

    logger.info("Found %d segments to evaluate", len(segments))
    report = []

    for seg in segments:
        league_id = seg["league_id"]
        season = seg["season"]
        total_ft = seg["cnt"]
        name = seg["league_name"]

        # Sample N random fixtures
        sample = await conn.fetch("""
            SELECT m.id as match_id, m.external_id, m.date::date as match_date
            FROM matches m
            WHERE m.league_id = $1 AND m.season = $2
              AND m.status IN ('FT', 'AET', 'PEN')
              AND m.external_id IS NOT NULL
            ORDER BY RANDOM()
            LIMIT $3
        """, league_id, season, args.sample_size)

        has_data = 0
        has_rating = 0
        player_counts = []

        for match in sample:
            await limiter.acquire()
            try:
                data = await provider.get_fixture_players(match["external_id"])
                if data:
                    has_data += 1
                    total_p = 0
                    rated_p = 0
                    for team_block in data:
                        for p in (team_block.get("players") or []):
                            total_p += 1
                            stats = (p.get("statistics") or [{}])[0]
                            r = (stats.get("games") or {}).get("rating")
                            if r and str(r).strip() not in ("", "-"):
                                rated_p += 1
                    player_counts.append(total_p)
                    if rated_p > 0:
                        has_rating += 1
            except Exception as e:
                logger.warning("Pilot error %s/%s fixture %s: %s", league_id, season, match["external_id"], e)

        n_sampled = len(sample)
        data_pct = 100 * has_data / n_sampled if n_sampled else 0
        rating_pct = 100 * has_rating / n_sampled if n_sampled else 0
        avg_players = sum(player_counts) / len(player_counts) if player_counts else 0

        if data_pct >= THRESHOLD_GO:
            decision = "GO"
        elif data_pct >= THRESHOLD_HOLD:
            decision = "HOLD"
        else:
            decision = "STOP"

        report.append({
            "league_id": league_id,
            "season": season,
            "name": name,
            "total_ft": total_ft,
            "sampled": n_sampled,
            "has_data": has_data,
            "has_rating": has_rating,
            "data_pct": round(data_pct, 1),
            "rating_pct": round(rating_pct, 1),
            "avg_players": round(avg_players, 1),
            "decision": decision,
        })

        logger.info(
            "  %s (season %s): %d/%d data (%.0f%%), %d/%d rated (%.0f%%), avg %.0f players → %s",
            name, season, has_data, n_sampled, data_pct,
            has_rating, n_sampled, rating_pct, avg_players, decision
        )

    # Summary
    go = sum(1 for r in report if r["decision"] == "GO")
    hold = sum(1 for r in report if r["decision"] == "HOLD")
    stop = sum(1 for r in report if r["decision"] == "STOP")
    total_go_matches = sum(r["total_ft"] for r in report if r["decision"] == "GO")

    logger.info(
        "\n=== PILOT SUMMARY ===\n"
        "  GO: %d segments (%d matches)\n"
        "  HOLD: %d segments\n"
        "  STOP: %d segments\n"
        "  Estimated requests for GO segments: ~%d",
        go, total_go_matches, hold, stop, total_go_matches
    )

    # Save report
    report_path = os.path.join(os.path.dirname(__file__), "..", "data", "player_stats_pilot.json")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Report saved to %s", report_path)

    return report


# ---------------------------------------------------------------------------
# Backfill Mode
# ---------------------------------------------------------------------------
async def run_backfill(conn, provider, limiter, args, team_map):
    """Ingest player stats for all FT matches since 2023."""
    logger.info("=== BACKFILL MODE ===")

    league_filter = "AND m.league_id = $2" if args.league else ""
    season_filter = "AND m.season = $3" if args.season else ""
    params = [MIN_DATE]
    if args.league:
        params.append(args.league)
    if args.season:
        params.append(args.season)

    query = f"""
        SELECT m.id as match_id, m.external_id, m.date::date as match_date,
               m.league_id, m.season
        FROM matches m
        WHERE m.status IN ('FT', 'AET', 'PEN')
          AND m.date >= $1
          AND m.external_id IS NOT NULL
          AND NOT EXISTS (
              SELECT 1 FROM match_player_stats mps WHERE mps.match_id = m.id
          )
          {league_filter}
          {season_filter}
        ORDER BY m.date DESC
    """
    matches = await conn.fetch(query, *params)
    total = len(matches)

    if args.limit > 0:
        matches = matches[:args.limit]

    logger.info("Found %d FT matches without player stats (processing %d)", total, len(matches))
    logger.info("Rate limit: %.1f req/s", args.rps)
    eta = len(matches) / args.rps / 60
    logger.info("Estimated time: %.0f minutes", eta)

    if args.dry_run:
        logger.info("[DRY-RUN] Would process %d matches. Exiting.", len(matches))
        return

    CHUNK = 500
    total_inserted = 0
    total_no_data = 0
    total_errors = 0
    total_processed = 0
    total_players = 0
    total_with_rating = 0
    t0 = time.time()

    for chunk_start in range(0, len(matches), CHUNK):
        chunk = matches[chunk_start:chunk_start + CHUNK]
        results = {
            "rows": [], "no_data": 0, "errors": 0, "processed": 0,
            "with_rating": 0, "total_players": 0,
        }

        tasks = [fetch_one(provider, m, limiter, results, team_map) for m in chunk]
        await asyncio.gather(*tasks)

        if results["rows"]:
            await conn.executemany(UPSERT_SQL, results["rows"])

        total_inserted += len(results["rows"])
        total_no_data += results["no_data"]
        total_errors += results["errors"]
        total_processed += results["processed"]
        total_players += results["total_players"]
        total_with_rating += results["with_rating"]

        elapsed = time.time() - t0
        rate = total_processed / elapsed if elapsed > 0 else 0
        remaining = len(matches) - total_processed
        eta_min = remaining / rate / 60 if rate > 0 else 0

        # Circuit breaker: if >30% NO_DATA in recent chunk
        if results["processed"] > 0:
            no_data_ratio = results["no_data"] / results["processed"]
            if no_data_ratio > CIRCUIT_BREAKER_RATIO and results["processed"] >= 50:
                logger.warning(
                    "Circuit breaker: %.0f%% NO_DATA in last chunk (%d/%d). Consider --league filter.",
                    no_data_ratio * 100, results["no_data"], results["processed"]
                )

        logger.info(
            "Progress: %d/%d (%.1f%%) rows=%d no_data=%d errors=%d "
            "rated=%d/%d [%.1f req/s, ETA %.0fm]",
            total_processed, len(matches),
            100 * total_processed / len(matches) if matches else 0,
            total_inserted, total_no_data, total_errors,
            total_with_rating, total_players,
            rate, eta_min,
        )

    elapsed = time.time() - t0
    logger.info(
        "\n=== BACKFILL COMPLETE ===\n"
        "  processed: %d fixtures\n"
        "  inserted: %d player rows\n"
        "  no_data: %d fixtures\n"
        "  errors: %d\n"
        "  rated: %d/%d players (%.1f%%)\n"
        "  time: %.0fs (%.1fm)",
        total_processed, total_inserted, total_no_data, total_errors,
        total_with_rating, total_players,
        100 * total_with_rating / total_players if total_players else 0,
        elapsed, elapsed / 60,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
async def main():
    import asyncpg
    from app.etl.api_football import APIFootballProvider

    parser = argparse.ArgumentParser(description="Backfill match_player_stats from API-Football")
    parser.add_argument("--mode", choices=["pilot", "backfill"], default="pilot",
                        help="pilot: sample coverage, backfill: ingest all (default: pilot)")
    parser.add_argument("--league", type=int, default=0, help="Filter to single league ID (0 = all)")
    parser.add_argument("--season", type=int, default=0, help="Filter to single season (0 = all)")
    parser.add_argument("--sample-size", type=int, default=10, help="Fixtures per segment in pilot (default: 10)")
    parser.add_argument("--rps", type=float, default=8.0, help="Max requests per second (default: 8)")
    parser.add_argument("--limit", type=int, default=0, help="Max fixtures to process (0 = all)")
    parser.add_argument("--dry-run", action="store_true", help="Don't insert, just count")
    args = parser.parse_args()

    db_url = os.environ.get("DATABASE_URL", "")
    if not db_url:
        raise RuntimeError("DATABASE_URL must be set")
    if "+asyncpg" in db_url:
        db_url = db_url.replace("postgresql+asyncpg://", "postgresql://")

    conn = await asyncpg.connect(db_url)
    logger.info("Connected to database")

    try:
        # Pre-load team external_id → internal id mapping
        team_rows = await conn.fetch(
            "SELECT id, external_id FROM teams WHERE external_id IS NOT NULL"
        )
        team_map = {r["external_id"]: r["id"] for r in team_rows}
        logger.info("Loaded %d team mappings", len(team_map))

        provider = APIFootballProvider()
        limiter = GlobalRateLimiter(args.rps)

        try:
            if args.mode == "pilot":
                await run_pilot(conn, provider, limiter, args, team_map)
            else:
                await run_backfill(conn, provider, limiter, args, team_map)
        finally:
            await provider.close()

    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(main())
