"""
Backfill: Populate match_lineups for recent FT matches missing lineups.

Fetches lineups from API-Football per fixture, inserts into match_lineups
with ON CONFLICT DO NOTHING (safe to re-run).

Usage:
  source .env && python3 scripts/backfill_match_lineups.py [--months 6] [--league 128] [--dry-run] [--limit 100] [--rps 8]
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Reduce httpx noise for bulk runs
logging.getLogger("httpx").setLevel(logging.WARNING)

UPSERT_SQL = """
    INSERT INTO match_lineups (
        match_id, team_id, is_home, formation,
        starting_xi_ids, starting_xi_names, starting_xi_positions,
        substitutes_ids, substitutes_names,
        coach_id, coach_name, source, created_at
    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, NOW())
    ON CONFLICT (match_id, team_id) DO NOTHING
"""


class GlobalRateLimiter:
    """Token-bucket rate limiter — ensures at most `rps` requests per second globally."""

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


async def fetch_one(provider, match, limiter, results):
    """Fetch lineups for one fixture with global rate limiting."""
    await limiter.acquire()

    fixture_ext_id = match["external_id"]
    match_id = match["match_id"]
    home_team_id = match["home_team_id"]
    away_team_id = match["away_team_id"]

    try:
        lineups = await provider.get_lineups(fixture_ext_id)

        if not lineups or (not lineups.get("home") and not lineups.get("away")):
            results["no_data"] += 1
            return

        rows = []
        for side, is_home in [("home", True), ("away", False)]:
            lineup = lineups.get(side)
            if not lineup:
                continue

            team_id = home_team_id if is_home else away_team_id

            xi_ids = [p["id"] for p in lineup.get("starting_xi", [])]
            xi_names = [p["name"] for p in lineup.get("starting_xi", [])]
            xi_positions = [p.get("pos", "") for p in lineup.get("starting_xi", [])]
            sub_ids = [p["id"] for p in lineup.get("substitutes", [])]
            sub_names = [p["name"] for p in lineup.get("substitutes", [])]

            coach = lineup.get("coach") or {}
            coach_id = coach.get("id")
            coach_name = coach.get("name")

            rows.append((
                match_id, team_id, is_home,
                lineup.get("formation"),
                xi_ids, xi_names, xi_positions,
                sub_ids, sub_names,
                coach_id, coach_name,
                "api-football",
            ))

        if rows:
            results["rows"].extend(rows)

    except Exception as e:
        results["errors"] += 1
        if results["errors"] <= 10:
            logger.warning(f"Error for fixture {fixture_ext_id}: {e}")

    results["processed"] += 1


async def main():
    import asyncpg
    from app.etl.api_football import APIFootballProvider

    parser = argparse.ArgumentParser(description="Backfill match lineups")
    parser.add_argument("--months", type=int, default=6, help="Lookback months (default: 6)")
    parser.add_argument("--league", type=int, default=0, help="Filter to single league ID (0 = all)")
    parser.add_argument("--dry-run", action="store_true", help="Don't insert, just count")
    parser.add_argument("--limit", type=int, default=0, help="Max fixtures to process (0 = all)")
    parser.add_argument("--rps", type=float, default=8.0, help="Max requests per second (default: 8)")
    args = parser.parse_args()

    db_url = os.environ.get("DATABASE_URL", "")
    if not db_url:
        raise RuntimeError("DATABASE_URL must be set")
    if "+asyncpg" in db_url:
        db_url = db_url.replace("postgresql+asyncpg://", "postgresql://")

    conn = await asyncpg.connect(db_url)
    logger.info("Connected to database")

    try:
        team_rows = await conn.fetch(
            "SELECT id, external_id FROM teams WHERE external_id IS NOT NULL"
        )
        logger.info(f"Loaded {len(team_rows)} teams")

        league_filter = "AND m.league_id = $2" if args.league else ""
        params: list = [args.months]
        if args.league:
            params.append(args.league)

        query = f"""
            SELECT m.id as match_id, m.external_id, m.home_team_id, m.away_team_id,
                   m.date, m.league_id
            FROM matches m
            LEFT JOIN (SELECT DISTINCT match_id FROM match_lineups) ml ON ml.match_id = m.id
            WHERE m.status = 'FT'
              AND m.date >= NOW() - make_interval(months => $1)
              AND m.external_id IS NOT NULL
              AND ml.match_id IS NULL
              {league_filter}
            ORDER BY m.date DESC
        """
        matches = await conn.fetch(query, *params)
        total = len(matches)
        if args.limit > 0:
            matches = matches[:args.limit]
        logger.info(f"Found {total} FT matches without lineups (processing {len(matches)})")
        logger.info(f"Rate limit: {args.rps} req/s = {args.rps * 60:.0f} req/min")
        eta = len(matches) / args.rps / 60
        logger.info(f"Estimated time: {eta:.0f} minutes")

        if args.dry_run:
            logger.info("[DRY-RUN] Would process %d matches. Exiting.", len(matches))
            return

        provider = APIFootballProvider()
        limiter = GlobalRateLimiter(args.rps)

        try:
            CHUNK = 500
            total_inserted = 0
            total_no_data = 0
            total_errors = 0
            total_processed = 0
            t0 = time.time()

            for chunk_start in range(0, len(matches), CHUNK):
                chunk = matches[chunk_start:chunk_start + CHUNK]
                results: dict = {"rows": [], "no_data": 0, "errors": 0, "processed": 0}

                tasks = [fetch_one(provider, m, limiter, results) for m in chunk]
                await asyncio.gather(*tasks)

                if results["rows"]:
                    await conn.executemany(UPSERT_SQL, results["rows"])

                total_inserted += len(results["rows"])
                total_no_data += results["no_data"]
                total_errors += results["errors"]
                total_processed += results["processed"]

                elapsed = time.time() - t0
                rate = total_processed / elapsed if elapsed > 0 else 0
                remaining = len(matches) - total_processed
                eta_min = remaining / rate / 60 if rate > 0 else 0
                logger.info(
                    f"Progress: {total_processed}/{len(matches)} "
                    f"({100*total_processed/len(matches):.1f}%) "
                    f"inserted={total_inserted} no_data={total_no_data} "
                    f"errors={total_errors} "
                    f"[{rate:.1f} req/s, ETA {eta_min:.0f}m]"
                )

        finally:
            await provider.close()

        elapsed = time.time() - t0
        logger.info(
            f"=== BACKFILL COMPLETE ===\n"
            f"  processed={total_processed}\n"
            f"  inserted={total_inserted} lineup rows\n"
            f"  no_data={total_no_data}\n"
            f"  errors={total_errors}\n"
            f"  time={elapsed:.0f}s ({elapsed/60:.1f}m)"
        )

    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(main())
