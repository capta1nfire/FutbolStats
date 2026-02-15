"""Backfill Sofascore lineups for matches missing in SOTA but present in TITAN.

Fetches /event/{id}/lineups from Sofascore API for matches that:
- Are in titan.feature_matrix with sofascore_lineup_available=false
- Have a sofascore ref in match_external_refs
- Don't have lineup data in match_sofascore_lineup

One-time backfill script. Uses existing SofascoreProvider + _upsert_sofascore_lineup.
"""
import asyncio
import logging
from datetime import timedelta

from sqlalchemy import text

from app.database import AsyncSessionLocal
from app.etl.sofascore_provider import SofascoreProvider
from app.etl.sota_jobs import _upsert_sofascore_lineup
from app.etl.sota_constants import LEAGUE_PROXY_COUNTRY

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


async def main():
    provider = SofascoreProvider(use_mock=False)
    stats = {"total": 0, "captured": 0, "no_data": 0, "errors": 0, "low_integrity": 0}

    try:
        async with AsyncSessionLocal() as session:
            result = await session.execute(text("""
                SELECT
                    m.id AS match_id,
                    m.date AS kickoff_utc,
                    m.league_id,
                    mer.source_match_id AS sofascore_id
                FROM titan.feature_matrix fm
                JOIN public.matches m ON m.external_id = CAST(fm.match_id AS INTEGER)
                JOIN match_external_refs mer ON mer.match_id = m.id AND mer.source = 'sofascore'
                WHERE fm.sofascore_lineup_available = false
                  AND NOT EXISTS (
                      SELECT 1 FROM match_sofascore_lineup msl
                      WHERE msl.match_id = m.id AND msl.team_side = 'home'
                  )
                ORDER BY m.date DESC
            """))
            matches = result.fetchall()
            stats["total"] = len(matches)
            logger.info(f"Found {len(matches)} matches to backfill")

            for i, match in enumerate(matches):
                match_id = match.match_id
                sofascore_id = match.sofascore_id
                kickoff_utc = match.kickoff_utc
                cc = LEAGUE_PROXY_COUNTRY.get(match.league_id)

                try:
                    lineup_data = await provider.get_match_lineup(
                        sofascore_id, country_code=cc
                    )

                    if lineup_data.error:
                        stats["no_data"] += 1
                        continue

                    if lineup_data.integrity_score < 0.3:
                        stats["low_integrity"] += 1
                        continue

                    # For historical backfill: set captured_at to 1 min before kickoff
                    # This is PIT-safe (captured_at < kickoff) and allows TITAN to pick it up
                    safe_captured_at = kickoff_utc - timedelta(minutes=1)
                    lineup_data.captured_at = safe_captured_at

                    await _upsert_sofascore_lineup(
                        session, match_id, lineup_data, kickoff_utc
                    )
                    stats["captured"] += 1

                    if (i + 1) % 20 == 0:
                        await session.commit()
                        logger.info(
                            f"Progress: {i+1}/{len(matches)}, captured={stats['captured']}"
                        )

                    # Rate limit: ~5 req/s
                    await asyncio.sleep(0.2)

                except Exception as e:
                    stats["errors"] += 1
                    logger.warning(f"Error match {match_id}: {e}")
                    continue

            await session.commit()
    finally:
        await provider.close()

    logger.info(f"DONE: {stats}")


if __name__ == "__main__":
    asyncio.run(main())
