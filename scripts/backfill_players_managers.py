"""
Backfill: Populate player_injuries and team_manager_history tables.

Uses raw asyncpg executemany for injuries (batch of all records per league)
to minimize round-trips to remote DB.

Usage:
  source .env && python3 scripts/backfill_players_managers.py [--injuries-only | --managers-only]
"""

import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


UPSERT_SQL = """
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


async def backfill_injuries_batch():
    """Batch-optimized injuries backfill using raw asyncpg executemany."""
    import asyncpg
    from app.config import get_settings
    from app.etl.api_football import APIFootballProvider
    from app.etl.player_jobs import TRACKED_LEAGUES, _current_season

    settings = get_settings()
    db_url = settings.DATABASE_URL
    if db_url.startswith("postgresql+asyncpg://"):
        db_url = db_url.replace("postgresql+asyncpg://", "postgresql://", 1)
    elif not db_url.startswith("postgresql://"):
        db_url = "postgresql://" + db_url.split("://", 1)[-1]

    # Use raw asyncpg for fast executemany
    conn = await asyncpg.connect(db_url)
    provider = APIFootballProvider()
    total_inserted = 0
    errors = 0

    # Pre-load maps
    team_rows = await conn.fetch("SELECT id, external_id FROM teams WHERE external_id IS NOT NULL")
    team_map = {r["external_id"]: r["id"] for r in team_rows}

    match_rows = await conn.fetch(
        "SELECT id, external_id FROM matches WHERE external_id IS NOT NULL AND date >= NOW() - INTERVAL '90 days'"
    )
    match_map = {r["external_id"]: r["id"] for r in match_rows}

    print(f"  Loaded {len(team_map)} teams, {len(match_map)} recent matches")

    try:
        for league_id in TRACKED_LEAGUES:
            season = _current_season(league_id)

            try:
                data = await provider._rate_limited_request(
                    "injuries",
                    {"league": league_id, "season": season},
                    entity="injury",
                )
                injuries = data.get("response", [])

                if not injuries:
                    print(f"  league={league_id}: 0 injuries")
                    continue

                # Build batch tuples
                args = []
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

                    fixture_date_raw = fixture.get("date")
                    fixture_date = None
                    if fixture_date_raw:
                        try:
                            dt = datetime.fromisoformat(fixture_date_raw.replace("Z", "+00:00"))
                            fixture_date = dt.replace(tzinfo=None)
                        except (ValueError, AttributeError):
                            pass

                    player_name = player.get("name") or f"Player#{player_ext_id}"

                    args.append((
                        player_ext_id,
                        player_name,
                        team_internal_id,
                        league_id,
                        season,
                        fixture_ext_id,
                        match_internal_id,
                        player.get("type") or "Unknown",
                        player.get("reason"),
                        fixture_date,
                        json.dumps(entry),
                    ))

                if not args:
                    print(f"  league={league_id}: 0 valid injuries")
                    continue

                # executemany: single protocol message, much faster than individual INSERTs
                await conn.executemany(UPSERT_SQL, args)
                total_inserted += len(args)
                print(f"  league={league_id}: {len(args)} injuries upserted")

            except Exception as e:
                errors += 1
                print(f"  league={league_id}: ERROR - {e}")
                logger.error(f"league={league_id}: {e}", exc_info=True)

    finally:
        await provider.close()
        await conn.close()

    return total_inserted, errors


async def main():
    injuries_only = "--injuries-only" in sys.argv
    managers_only = "--managers-only" in sys.argv

    if injuries_only and managers_only:
        print("ERROR: Cannot use both --injuries-only and --managers-only")
        sys.exit(1)

    # --- Injuries (batch optimized with raw asyncpg) ---
    if not managers_only:
        print("\n=== Backfill: Player Injuries (batch mode) ===")
        t0 = time.time()
        inserted, errors = await backfill_injuries_batch()
        elapsed = time.time() - t0
        print(f"\n  Total upserted: {inserted}")
        print(f"  Errors: {errors}")
        print(f"  Time: {elapsed:.1f}s")

    # --- Managers (regular sync) ---
    if not injuries_only:
        from app.database import AsyncSessionLocal
        from app.etl.player_jobs import sync_managers

        print("\n=== Backfill: Team Managers ===")
        t0 = time.time()
        async with AsyncSessionLocal() as session:
            metrics = await sync_managers(session)
        elapsed = time.time() - t0
        print(f"  Teams OK: {metrics['teams_ok']}/{metrics['teams_attempted']}")
        print(f"  Managers upserted: {metrics['managers_upserted']}")
        print(f"  Changes detected: {metrics['changes_detected']}")
        print(f"  Errors: {metrics['errors']}")
        if metrics["error_details"]:
            for err in metrics["error_details"][:5]:
                print(f"    - {err}")
        print(f"  Time: {elapsed:.1f}s")

    print("\nBackfill complete.")


if __name__ == "__main__":
    asyncio.run(main())
