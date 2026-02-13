"""
Backfill: Populate match_lineups for recent FT matches missing lineups.

Fetches lineups from API-Football per fixture, inserts into match_lineups
with ON CONFLICT DO NOTHING (safe to re-run).

Usage:
  source .env && python3 scripts/backfill_match_lineups.py [--months 6] [--league 128] [--dry-run] [--limit 100]
"""

import argparse
import asyncio
import json
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

UPSERT_SQL = """
    INSERT INTO match_lineups (
        match_id, team_id, is_home, formation,
        starting_xi_ids, starting_xi_names, starting_xi_positions,
        substitutes_ids, substitutes_names,
        coach_id, coach_name, source, created_at
    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, NOW())
    ON CONFLICT (match_id, team_id) DO NOTHING
"""


async def main():
    import asyncpg
    from app.etl.api_football import APIFootballProvider

    parser = argparse.ArgumentParser(description="Backfill match lineups")
    parser.add_argument("--months", type=int, default=6, help="Lookback months (default: 6)")
    parser.add_argument("--league", type=int, default=0, help="Filter to single league ID (0 = all)")
    parser.add_argument("--dry-run", action="store_true", help="Don't insert, just count")
    parser.add_argument("--limit", type=int, default=0, help="Max fixtures to process (0 = all)")
    args = parser.parse_args()

    db_url = os.environ.get("DATABASE_URL", "")
    if not db_url:
        raise RuntimeError("DATABASE_URL must be set")
    # asyncpg needs raw postgresql:// URL
    if "+asyncpg" in db_url:
        db_url = db_url.replace("postgresql+asyncpg://", "postgresql://")

    conn = await asyncpg.connect(db_url)
    logger.info("Connected to database")

    try:
        # Load team external_id -> internal_id map
        team_rows = await conn.fetch(
            "SELECT id, external_id FROM teams WHERE external_id IS NOT NULL"
        )
        team_map = {r["external_id"]: r["id"] for r in team_rows}
        logger.info(f"Loaded {len(team_map)} teams")

        # Find FT matches missing lineups
        league_filter = "AND m.league_id = $2" if args.league else ""
        params = [args.months]
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

        if args.dry_run:
            logger.info("[DRY-RUN] Would process %d matches. Exiting.", len(matches))
            return

        provider = APIFootballProvider()
        inserted = 0
        skipped = 0
        errors = 0
        no_data = 0
        batch_args = []

        try:
            for i, match in enumerate(matches):
                fixture_ext_id = match["external_id"]
                match_id = match["match_id"]
                home_team_id = match["home_team_id"]
                away_team_id = match["away_team_id"]

                try:
                    lineups = await provider.get_lineups(fixture_ext_id)

                    if not lineups or (not lineups.get("home") and not lineups.get("away")):
                        no_data += 1
                        if no_data <= 5:
                            logger.debug(f"No lineup data for fixture {fixture_ext_id}")
                        continue

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

                        batch_args.append((
                            match_id, team_id, is_home,
                            lineup.get("formation"),
                            xi_ids, xi_names, xi_positions,
                            sub_ids, sub_names,
                            coach_id, coach_name,
                            "api-football",
                        ))

                    # Flush batch every 200 fixtures
                    if len(batch_args) >= 400:
                        await conn.executemany(UPSERT_SQL, batch_args)
                        inserted += len(batch_args)
                        batch_args = []

                except Exception as e:
                    errors += 1
                    if errors <= 10:
                        logger.warning(f"Error for fixture {fixture_ext_id}: {e}")

                if (i + 1) % 100 == 0:
                    logger.info(
                        f"Progress: {i+1}/{len(matches)} "
                        f"(inserted={inserted}, no_data={no_data}, errors={errors})"
                    )

            # Flush remaining
            if batch_args:
                await conn.executemany(UPSERT_SQL, batch_args)
                inserted += len(batch_args)

        finally:
            await provider.close()

        logger.info(
            f"Backfill complete: processed={len(matches)}, "
            f"inserted={inserted}, no_data={no_data}, errors={errors}, skipped={skipped}"
        )

    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(main())
