#!/usr/bin/env python3
"""
One-off squad sync for leagues missing from regular sync.

ABE mandate: Reuse sync_squads() UPSERT logic but bypass the 180-day filter
in _get_active_teams(). Enumerate teams from matches table (historical).

Target leagues: 242 (Ecuador), 299 (Venezuela), 268 (Uruguay), 253 (MLS)
Priority: P0 = Uruguay, Ecuador, Venezuela; P1 = MLS

Usage:
    source .env
    python3 scripts/squad_sync_oneoff.py
    python3 scripts/squad_sync_oneoff.py --league 242   # single league
    python3 scripts/squad_sync_oneoff.py --dry-run       # count only
"""
import argparse
import asyncio
import logging
import os
import sys
from datetime import date

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.etl.api_football import APIFootballProvider

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

TARGET_LEAGUES = [242, 299, 268, 253]

LEAGUE_NAMES = {
    242: "Ecuador Liga Pro",
    299: "Venezuela Primera",
    268: "Uruguay Primera",
    253: "MLS",
}


async def get_historical_teams(session, league_ids):
    """Get ALL teams that ever played in these leagues (no date filter).

    Returns team info with latest season for API-Football call.
    """
    result = await session.execute(
        text("""
            SELECT DISTINCT ON (t.id)
                   t.id, t.external_id, t.name,
                   m.league_id,
                   m.season
            FROM teams t
            JOIN matches m ON (m.home_team_id = t.id OR m.away_team_id = t.id)
            WHERE m.league_id = ANY(CAST(:league_ids AS int[]))
              AND t.external_id IS NOT NULL
            ORDER BY t.id, m.date DESC
        """),
        {"league_ids": league_ids},
    )
    return [
        {
            "id": row.id,
            "external_id": row.external_id,
            "name": row.name,
            "league_id": row.league_id,
            "league_external_id": row.league_id,  # admin_leagues.league_id = external
            "season": row.season,
        }
        for row in result.fetchall()
    ]


async def upsert_player(session, p, team_id, team_ext_id):
    """Same UPSERT as player_jobs.py:sync_squads() L572-622."""
    await session.execute(
        text("""
            INSERT INTO players
                (external_id, name, firstname, lastname, position,
                 team_id, team_external_id, jersey_number, age, photo_url,
                 birth_date, birth_place, birth_country, nationality,
                 height, weight, last_synced_at)
            VALUES
                (:ext_id, :name, :firstname, :lastname, :position,
                 :team_id, :team_ext_id, :number, :age, :photo,
                 :birth_date, :birth_place, :birth_country,
                 :nationality, :height, :weight, NOW())
            ON CONFLICT (external_id) DO UPDATE SET
                name = COALESCE(EXCLUDED.name, players.name),
                firstname = COALESCE(EXCLUDED.firstname, players.firstname),
                lastname = COALESCE(EXCLUDED.lastname, players.lastname),
                position = COALESCE(EXCLUDED.position, players.position),
                team_id = COALESCE(EXCLUDED.team_id, players.team_id),
                team_external_id = COALESCE(EXCLUDED.team_external_id, players.team_external_id),
                jersey_number = COALESCE(EXCLUDED.jersey_number, players.jersey_number),
                age = COALESCE(EXCLUDED.age, players.age),
                photo_url = COALESCE(EXCLUDED.photo_url, players.photo_url),
                birth_date = COALESCE(EXCLUDED.birth_date, players.birth_date),
                birth_place = COALESCE(EXCLUDED.birth_place, players.birth_place),
                birth_country = COALESCE(EXCLUDED.birth_country, players.birth_country),
                nationality = COALESCE(EXCLUDED.nationality, players.nationality),
                height = COALESCE(EXCLUDED.height, players.height),
                weight = COALESCE(EXCLUDED.weight, players.weight),
                last_synced_at = NOW()
        """),
        {
            "ext_id": p["id"],
            "name": p.get("name", "Unknown"),
            "firstname": p.get("firstname"),
            "lastname": p.get("lastname"),
            "position": p.get("position"),
            "team_id": team_id,
            "team_ext_id": team_ext_id,
            "number": p.get("number"),
            "age": p.get("age"),
            "photo": p.get("photo"),
            "birth_date": date.fromisoformat(p["birth_date"]) if p.get("birth_date") else None,
            "birth_place": p.get("birth_place"),
            "birth_country": p.get("birth_country"),
            "nationality": p.get("nationality"),
            "height": p.get("height"),
            "weight": p.get("weight"),
        },
    )


async def sync_team(session, provider, team, dry_run=False):
    """Sync one team's squad. Returns (players_upserted, squad_complement)."""
    team_id = team["id"]
    team_ext_id = team["external_id"]
    team_name = team["name"]
    league_ext_id = team["league_external_id"]
    season = team["season"]

    upserted = 0
    complement = 0

    # Primary: get_players_full (has birth_date, nationality, etc.)
    # Try current season first, then season-1 (2026 may be too new for /players)
    players = []
    seasons_to_try = [season]
    if season and season >= 2026:
        seasons_to_try.append(season - 1)

    for s in seasons_to_try:
        try:
            players = await provider.get_players_full(
                team_ext_id, s, league_id=league_ext_id
            )
        except Exception as e:
            logger.warning(f"  get_players_full(season={s}) failed for {team_name}: {e}")
            players = []
        if players:
            if s != season:
                logger.info(f"  {team_name}: used season={s} (current {season} had 0 players)")
            break

    if not players:
        # Last resort: get_players_squad (basic info, no birth_date)
        try:
            players = await provider.get_players_squad(team_ext_id)
            if players:
                logger.info(f"  {team_name}: fallback to squad endpoint ({len(players)} players, NO birth_date)")
        except Exception as e:
            logger.warning(f"  get_players_squad also failed for {team_name}: {e}")
            return 0, 0

    if not players:
        logger.info(f"  {team_name}: no players from API-Football")
        return 0, 0

    if dry_run:
        bd_count = sum(1 for p in players if p.get("birth_date"))
        logger.info(f"  {team_name}: {len(players)} players ({bd_count} with birth_date) [DRY RUN]")
        return len(players), 0

    # UPSERT all players
    await session.execute(text("SAVEPOINT sp_squad"))
    try:
        for p in players:
            await upsert_player(session, p, team_id, team_ext_id)
            upserted += 1
        await session.execute(text("RELEASE SAVEPOINT sp_squad"))
    except Exception:
        await session.execute(text("ROLLBACK TO SAVEPOINT sp_squad"))
        raise

    # Complement with /players/squads for missing players
    if season:
        full_ids = {p["id"] for p in players if p.get("id")}
        try:
            squad_players = await provider.get_players_squad(team_ext_id)
        except Exception:
            squad_players = []

        squad_only = [sp for sp in squad_players if sp.get("id") and sp["id"] not in full_ids]
        if squad_only:
            await session.execute(text("SAVEPOINT sp_complement"))
            try:
                for p in squad_only:
                    await upsert_player(session, p, team_id, team_ext_id)
                    complement += 1
                await session.execute(text("RELEASE SAVEPOINT sp_complement"))
            except Exception:
                await session.execute(text("ROLLBACK TO SAVEPOINT sp_complement"))
                raise

    return upserted, complement


async def count_players(session, league_ids):
    """Count players per league (via matches join)."""
    result = await session.execute(
        text("""
            SELECT m.league_id,
                   COUNT(DISTINCT p.id) AS total,
                   COUNT(DISTINCT CASE WHEN p.birth_date IS NOT NULL THEN p.id END) AS with_bd
            FROM matches m
            JOIN teams t ON (t.id = m.home_team_id OR t.id = m.away_team_id)
            JOIN players p ON p.team_id = t.id
            WHERE m.league_id = ANY(CAST(:lids AS int[]))
              AND t.external_id IS NOT NULL
            GROUP BY m.league_id
            ORDER BY m.league_id
        """),
        {"lids": league_ids},
    )
    return {row.league_id: {"total": row.total, "with_bd": row.with_bd} for row in result.fetchall()}


async def main():
    parser = argparse.ArgumentParser(description="One-off squad sync for missing leagues")
    parser.add_argument("--league", type=int, action="append", help="Specific league(s)")
    parser.add_argument("--dry-run", action="store_true", help="Count only, no DB writes")
    args = parser.parse_args()

    league_ids = args.league or TARGET_LEAGUES

    db_url = os.environ.get("DATABASE_URL_ASYNC")
    if not db_url:
        db_url = os.environ.get("DATABASE_URL", "").replace(
            "postgresql://", "postgresql+asyncpg://", 1
        )
    if not db_url:
        logger.error("DATABASE_URL_ASYNC or DATABASE_URL required")
        sys.exit(1)

    engine = create_async_engine(db_url, pool_size=5)
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    provider = APIFootballProvider()

    async with async_session() as session:
        # Pre-check
        pre_counts = await count_players(session, league_ids)
        logger.info("=== PRE-CHECK ===")
        for lid in league_ids:
            c = pre_counts.get(lid, {"total": 0, "with_bd": 0})
            logger.info(f"  {LEAGUE_NAMES.get(lid, lid)}: {c['total']} players ({c['with_bd']} with BD)")

        # Get teams
        teams = await get_historical_teams(session, league_ids)
        logger.info(f"\n=== TEAMS: {len(teams)} across {len(league_ids)} leagues ===")

        by_league = {}
        for t in teams:
            by_league.setdefault(t["league_id"], []).append(t)

        total_upserted = 0
        total_complement = 0
        total_errors = 0

        for lid in league_ids:
            league_teams = by_league.get(lid, [])
            logger.info(f"\n--- {LEAGUE_NAMES.get(lid, lid)} ({lid}): {len(league_teams)} teams ---")

            for team in league_teams:
                try:
                    ups, comp = await sync_team(session, provider, team, dry_run=args.dry_run)
                    total_upserted += ups
                    total_complement += comp
                    if ups > 0 or comp > 0:
                        logger.info(f"  {team['name']}: {ups} upserted + {comp} complement")
                except Exception as e:
                    total_errors += 1
                    logger.error(f"  {team['name']}: ERROR {e}")

        if not args.dry_run:
            await session.commit()

        # Post-check
        post_counts = await count_players(session, league_ids)
        logger.info("\n=== POST-CHECK ===")
        for lid in league_ids:
            pre = pre_counts.get(lid, {"total": 0, "with_bd": 0})
            post = post_counts.get(lid, {"total": 0, "with_bd": 0})
            delta = post["total"] - pre["total"]
            bd_pct = round(post["with_bd"] / post["total"] * 100, 1) if post["total"] else 0
            logger.info(
                f"  {LEAGUE_NAMES.get(lid, lid)}: {pre['total']} → {post['total']} "
                f"(+{delta}), BD: {post['with_bd']}/{post['total']} ({bd_pct}%)"
            )

        logger.info(f"\n=== TOTALS: {total_upserted} upserted, {total_complement} complement, {total_errors} errors ===")

    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
