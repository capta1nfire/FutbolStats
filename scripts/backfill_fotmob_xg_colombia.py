"""
Backfill FotMob xG for Colombia (2025 Clausura onwards).

Colombia uses split seasons (Apertura/Clausura) in FotMob.
Opta/FotMob only has xG data starting from 2025 Clausura.
Earlier seasons (2023-2025 Apertura) have NO xG coverage.

Usage:
    set -a && source .env && set +a
    python3.12 scripts/backfill_fotmob_xg_colombia.py
"""
import asyncio
import json
import logging
import os
import sys
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

FOTMOB_LEAGUE_ID = 274   # Colombia Primera A in FotMob
OUR_LEAGUE_ID = 239       # Colombia Primera A in API-Football

# FotMob split seasons for Colombia (only seasons with xG coverage)
# Opta xG starts from 2025 Clausura. Earlier seasons have NO xG.
SEASONS = [
    "2025 - Clausura",
    "2026 - Apertura",
]


async def backfill_season(session, provider, alias_index, season_str: str) -> dict:
    """Link + capture xG for one Colombia half-season."""
    from sqlalchemy import text
    from app.etl.sofascore_provider import calculate_match_score

    # Parse year from season string (e.g. "2024 - Clausura" -> 2024)
    year = int(season_str.split(" - ")[0])
    phase = season_str.split(" - ")[1].lower()  # "apertura" or "clausura"

    metrics = {"season": season_str, "our_matches": 0, "fm_fixtures": 0,
               "linked": 0, "skipped_low": 0, "xg_captured": 0,
               "xg_no_data": 0, "xg_errors": 0}

    # --- Phase A: Link ---
    logger.info("[%s] Phase A: fetching FotMob fixtures...", season_str)
    fm_fixtures, error = await provider.get_league_fixtures(FOTMOB_LEAGUE_ID, season=season_str)
    if error:
        logger.error("[%s] Failed to fetch fixtures: %s", season_str, error)
        return metrics
    fm_finished = [f for f in fm_fixtures if f.status == "finished"]
    metrics["fm_fixtures"] = len(fm_finished)
    logger.info("[%s] FotMob: %d finished fixtures", season_str, len(fm_finished))

    # Date range filter: Apertura = Jan-Jun, Clausura = Jul-Dec
    if phase == "apertura":
        date_start = datetime(year, 1, 1)
        date_end = datetime(year, 7, 1)
    else:
        date_start = datetime(year, 7, 1)
        date_end = datetime(year + 1, 1, 1)

    # Our unlinked FT matches for this half-season
    result = await session.execute(text("""
        SELECT m.id, m.date, t_home.name AS home_team, t_away.name AS away_team
        FROM matches m
        JOIN teams t_home ON m.home_team_id = t_home.id
        JOIN teams t_away ON m.away_team_id = t_away.id
        LEFT JOIN match_external_refs mer
            ON m.id = mer.match_id AND mer.source = 'fotmob'
        WHERE m.league_id = :league_id
          AND m.status IN ('FT', 'AET', 'PEN')
          AND m.date >= :date_start
          AND m.date < :date_end
          AND mer.match_id IS NULL
        ORDER BY m.date
    """), {"league_id": OUR_LEAGUE_ID, "date_start": date_start, "date_end": date_end})
    unlinked = result.fetchall()
    metrics["our_matches"] = len(unlinked)
    logger.info("[%s] Our DB: %d unlinked FT matches", season_str, len(unlinked))

    if not unlinked:
        logger.info("[%s] All matches already linked, skipping to Phase B", season_str)
    else:
        for match in unlinked:
            best_score = 0.0
            best_fixture = None
            best_matched_by = ""

            for fm in fm_finished:
                score, matched_by = calculate_match_score(
                    our_home=match.home_team,
                    our_away=match.away_team,
                    our_kickoff=match.date,
                    sf_home=fm.home_team,
                    sf_away=fm.away_team,
                    sf_kickoff=fm.kickoff_utc,
                    alias_index=alias_index,
                )
                if score > best_score:
                    best_score = score
                    best_fixture = fm
                    best_matched_by = matched_by

            if best_score >= 0.75 and best_fixture:
                await session.execute(text("""
                    INSERT INTO match_external_refs (match_id, source, source_match_id, confidence, matched_by)
                    VALUES (:match_id, 'fotmob', :source_match_id, :confidence, :matched_by)
                    ON CONFLICT (match_id, source) DO NOTHING
                """), {
                    "match_id": match.id,
                    "source_match_id": str(best_fixture.fotmob_id),
                    "confidence": best_score,
                    "matched_by": best_matched_by,
                })
                metrics["linked"] += 1
            else:
                metrics["skipped_low"] += 1

        await session.commit()
        logger.info("[%s] Phase A done: linked=%d, skipped_low=%d",
                    season_str, metrics["linked"], metrics["skipped_low"])

    # --- Phase B: Capture xG ---
    logger.info("[%s] Phase B: capturing xG...", season_str)
    result = await session.execute(text("""
        SELECT mer.match_id, mer.source_match_id, m.date
        FROM match_external_refs mer
        JOIN matches m ON m.id = mer.match_id
        LEFT JOIN match_fotmob_stats mfs ON mer.match_id = mfs.match_id
        WHERE mer.source = 'fotmob'
          AND m.league_id = :league_id
          AND m.date >= :date_start
          AND m.date < :date_end
          AND mfs.match_id IS NULL
        ORDER BY m.date
    """), {"league_id": OUR_LEAGUE_ID, "date_start": date_start, "date_end": date_end})
    to_capture = result.fetchall()
    logger.info("[%s] %d matches need xG capture", season_str, len(to_capture))

    batch_size = 20
    for i, row in enumerate(to_capture):
        fotmob_id = int(row.source_match_id)
        xg_data, error = await provider.get_match_xg(fotmob_id)

        if error:
            metrics["xg_errors"] += 1
            if (i + 1) % 50 == 0:
                logger.warning("[%s] xG error %d/%d: match=%d err=%s",
                               season_str, i + 1, len(to_capture), row.match_id, error)
            continue

        if xg_data is None:
            metrics["xg_no_data"] += 1
            continue

        captured_at = row.date + timedelta(hours=6) if row.date else datetime.utcnow()

        await session.execute(text("""
            INSERT INTO match_fotmob_stats
                (match_id, xg_home, xg_away, xgot_home, xgot_away,
                 xg_open_play_home, xg_open_play_away,
                 xg_set_play_home, xg_set_play_away,
                 raw_stats, captured_at, source_version)
            VALUES
                (:match_id, :xg_home, :xg_away, :xgot_home, :xgot_away,
                 :xg_oph, :xg_opa, :xg_sph, :xg_spa,
                 CAST(:raw_stats AS jsonb), :captured_at, 'fotmob_v1')
            ON CONFLICT (match_id) DO UPDATE SET
                xg_home = EXCLUDED.xg_home,
                xg_away = EXCLUDED.xg_away,
                xgot_home = EXCLUDED.xgot_home,
                xgot_away = EXCLUDED.xgot_away,
                xg_open_play_home = EXCLUDED.xg_open_play_home,
                xg_open_play_away = EXCLUDED.xg_open_play_away,
                xg_set_play_home = EXCLUDED.xg_set_play_home,
                xg_set_play_away = EXCLUDED.xg_set_play_away,
                raw_stats = EXCLUDED.raw_stats,
                captured_at = EXCLUDED.captured_at
        """), {
            "match_id": row.match_id,
            "xg_home": xg_data.xg_home,
            "xg_away": xg_data.xg_away,
            "xgot_home": xg_data.xgot_home,
            "xgot_away": xg_data.xgot_away,
            "xg_oph": xg_data.xg_open_play_home,
            "xg_opa": xg_data.xg_open_play_away,
            "xg_sph": xg_data.xg_set_play_home,
            "xg_spa": xg_data.xg_set_play_away,
            "raw_stats": json.dumps(xg_data.raw_stats) if xg_data.raw_stats else "{}",
            "captured_at": captured_at,
        })
        metrics["xg_captured"] += 1

        if (i + 1) % batch_size == 0:
            await session.commit()
            logger.info("[%s] Progress: %d/%d captured", season_str, i + 1, len(to_capture))

    await session.commit()
    logger.info("[%s] Phase B done: captured=%d, no_data=%d, errors=%d",
                season_str, metrics["xg_captured"], metrics["xg_no_data"], metrics["xg_errors"])
    return metrics


async def main():
    from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
    from sqlalchemy.orm import sessionmaker
    from app.etl.fotmob_provider import FotmobProvider
    from app.etl.sofascore_aliases import build_alias_index

    db_url = os.environ.get("DATABASE_URL_ASYNC") or os.environ.get("DATABASE_URL", "")
    if "postgresql://" in db_url and "+asyncpg" not in db_url:
        db_url = db_url.replace("postgresql://", "postgresql+asyncpg://")

    engine = create_async_engine(db_url, pool_size=5)
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    provider = FotmobProvider()
    alias_index = build_alias_index()
    logger.info("Alias index built: %d entries", len(alias_index))

    all_metrics = []
    try:
        for season_str in SEASONS:
            async with async_session() as session:
                metrics = await backfill_season(session, provider, alias_index, season_str)
                all_metrics.append(metrics)
                logger.info("[%s] === COMPLETE === %s", season_str, metrics)
    finally:
        await provider.close()
        await engine.dispose()

    # Summary
    total_linked = sum(m["linked"] for m in all_metrics)
    total_xg = sum(m["xg_captured"] for m in all_metrics)
    logger.info("=" * 60)
    logger.info("BACKFILL COMPLETE: %d half-seasons, %d linked, %d xG captured",
                len(SEASONS), total_linked, total_xg)
    for m in all_metrics:
        logger.info("  %s", m)


if __name__ == "__main__":
    asyncio.run(main())
