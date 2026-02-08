"""
One-off script: backfill stadium_name and admin_location_label from matches.venue_name/city.

Uses the MOST FREQUENT venue from home league matches (filters neutral venues from cups).
Only UPDATEs existing rows in team_wikidata_enrichment (respects wikidata_id NOT NULL constraint).

Usage:
  source .env
  python scripts/backfill_venue_from_matches.py [--dry-run]
"""
import asyncio
import argparse
import logging
import os

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy import text

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)


async def main(dry_run: bool):
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        log.error("DATABASE_URL not set. Run: source .env")
        return

    if db_url.startswith("postgres://"):
        db_url = db_url.replace("postgres://", "postgresql+asyncpg://", 1)
    elif db_url.startswith("postgresql://"):
        db_url = db_url.replace("postgresql://", "postgresql+asyncpg://", 1)

    engine = create_async_engine(db_url, echo=False)
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async with async_session() as session:
        # Find active league teams with twe row but missing stadium or city
        result = await session.execute(text("""
            SELECT DISTINCT t.id, t.name
            FROM teams t
            JOIN matches m ON (m.home_team_id = t.id OR m.away_team_id = t.id)
            JOIN admin_leagues al ON al.league_id = m.league_id
            JOIN team_wikidata_enrichment twe ON twe.team_id = t.id
            WHERE m.date >= NOW() - INTERVAL '30 days'
              AND al.kind = 'league' AND al.is_active = true
              AND t.team_type = 'club'
              AND (
                (twe.stadium_name IS NULL OR twe.stadium_name = '')
                OR (twe.admin_location_label IS NULL OR twe.admin_location_label = '')
              )
            ORDER BY t.name
        """))
        teams = result.fetchall()

    log.info(f"Found {len(teams)} teams with gaps (dry_run={dry_run})")

    stadium_updated = 0
    city_updated = 0
    skipped = 0

    for team_id, name in teams:
        async with async_session() as session:
            # Most frequent venue from home league matches
            result = await session.execute(text("""
                SELECT venue_name, venue_city, COUNT(*) AS freq
                FROM matches m
                JOIN admin_leagues al ON al.league_id = m.league_id
                WHERE m.home_team_id = :team_id
                  AND m.venue_name IS NOT NULL
                  AND al.kind = 'league'
                  AND m.status IN ('FT','AET','PEN')
                  AND m.date >= NOW() - INTERVAL '365 days'
                GROUP BY venue_name, venue_city
                ORDER BY freq DESC
                LIMIT 1
            """), {"team_id": team_id})
            row = result.fetchone()

            if not row or not row.venue_name:
                log.info(f"  {name:30s} | no venue data in matches — skipped")
                skipped += 1
                continue

            venue_name = row.venue_name
            venue_city = row.venue_city

            # Check current twe values
            twe_result = await session.execute(text("""
                SELECT stadium_name, admin_location_label
                FROM team_wikidata_enrichment
                WHERE team_id = :team_id
            """), {"team_id": team_id})
            twe = twe_result.fetchone()
            if not twe:
                log.info(f"  {name:30s} | no twe row — skipped (P0-1)")
                skipped += 1
                continue

            updates = []
            params = {"team_id": team_id}

            if not twe.stadium_name or twe.stadium_name == "":
                updates.append("stadium_name = :venue_name")
                params["venue_name"] = venue_name
                stadium_updated += 1

            if (not twe.admin_location_label or twe.admin_location_label == "") and venue_city:
                updates.append("admin_location_label = :venue_city")
                params["venue_city"] = venue_city
                city_updated += 1

            if updates:
                log.info(f"  {name:30s} | stadium={venue_name or '-':35s} | city={venue_city or '-'}")
                if not dry_run:
                    sql = f"UPDATE team_wikidata_enrichment SET {', '.join(updates)} WHERE team_id = :team_id"
                    await session.execute(text(sql), params)
                    await session.commit()
            else:
                log.info(f"  {name:30s} | nothing to update")

    log.info(f"Done: stadium={stadium_updated}, city={city_updated}, skipped={skipped}")
    await engine.dispose()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Don't write to DB")
    args = parser.parse_args()
    asyncio.run(main(args.dry_run))
