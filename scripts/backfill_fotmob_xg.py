"""
Backfill FotMob xG for historical seasons.

Usage:
    set -a && source .env && set +a
    python3 scripts/backfill_fotmob_xg.py --league 71 --seasons 2020-2026
    python3 scripts/backfill_fotmob_xg.py --league 128              # default: 2023-2025

Phases per season:
  A) Fetch FotMob fixtures, link to our matches via calculate_match_score
  B) Fetch matchDetails xG for each linked match

Rate limit: 1.5s per request. ~10-13 min per season.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# League mapping from sota_constants
from app.etl.sota_constants import LEAGUE_ID_TO_FOTMOB


def parse_args():
    parser = argparse.ArgumentParser(description="Backfill FotMob xG for a league")
    parser.add_argument("--league", type=int, required=True,
                        help="API-Football league_id (e.g. 71 for Brazil, 128 for Argentina)")
    parser.add_argument("--seasons", type=str, default=None,
                        help="Season range (e.g. '2020-2026') or comma-separated (e.g. '2023,2024,2025')")
    parser.add_argument("--dry-run", action="store_true",
                        help="Only run Phase A (linking), skip xG capture")
    return parser.parse_args()


def parse_seasons(seasons_str: str | None, league_id: int) -> list[int]:
    """Parse seasons argument. Default: 2023-2025."""
    if not seasons_str:
        return [2023, 2024, 2025]
    if "-" in seasons_str and "," not in seasons_str:
        start, end = seasons_str.split("-")
        return list(range(int(start), int(end) + 1))
    return [int(s.strip()) for s in seasons_str.split(",")]


async def backfill_season(session, provider, alias_index, season: int,
                          our_league_id: int, fotmob_league_id: int,
                          dry_run: bool = False) -> dict:
    """Link + capture xG for one historical season."""
    from sqlalchemy import text
    from app.etl.sofascore_provider import calculate_match_score

    metrics = {"season": season, "our_matches": 0, "fm_fixtures": 0,
               "linked": 0, "skipped_low": 0, "xg_captured": 0,
               "xg_no_data": 0, "xg_errors": 0}

    # --- Phase A: Link ---
    logger.info("[%d] Phase A: fetching FotMob fixtures...", season)
    fm_fixtures, error = await provider.get_league_fixtures(fotmob_league_id, season=season)
    if error:
        logger.error("[%d] Failed to fetch fixtures: %s", season, error)
        return metrics
    fm_finished = [f for f in fm_fixtures if f.status == "finished"]
    metrics["fm_fixtures"] = len(fm_finished)
    logger.info("[%d] FotMob: %d finished fixtures", season, len(fm_finished))

    # Our unlinked FT matches for this season
    result = await session.execute(text("""
        SELECT m.id, m.date, t_home.name AS home_team, t_away.name AS away_team
        FROM matches m
        JOIN teams t_home ON m.home_team_id = t_home.id
        JOIN teams t_away ON m.away_team_id = t_away.id
        LEFT JOIN match_external_refs mer
            ON m.id = mer.match_id AND mer.source = 'fotmob'
        WHERE m.league_id = :league_id
          AND m.status IN ('FT', 'AET', 'PEN')
          AND m.season = :season
          AND mer.match_id IS NULL
        ORDER BY m.date
    """), {"league_id": our_league_id, "season": season})
    unlinked = result.fetchall()
    metrics["our_matches"] = len(unlinked)
    logger.info("[%d] Our DB: %d unlinked FT matches", season, len(unlinked))

    if not unlinked:
        logger.info("[%d] All matches already linked, skipping to Phase B", season)
    else:
        # Match linking
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
        logger.info("[%d] Phase A done: linked=%d, skipped_low=%d",
                    season, metrics["linked"], metrics["skipped_low"])

    if dry_run:
        logger.info("[%d] Dry run — skipping Phase B (xG capture)", season)
        return metrics

    # --- Phase B: Capture xG ---
    logger.info("[%d] Phase B: capturing xG...", season)
    result = await session.execute(text("""
        SELECT mer.match_id, mer.source_match_id, m.date
        FROM match_external_refs mer
        JOIN matches m ON m.id = mer.match_id
        LEFT JOIN match_fotmob_stats mfs ON mer.match_id = mfs.match_id
        WHERE mer.source = 'fotmob'
          AND m.league_id = :league_id
          AND m.season = :season
          AND mfs.match_id IS NULL
        ORDER BY m.date
    """), {"league_id": our_league_id, "season": season})
    to_capture = result.fetchall()
    logger.info("[%d] %d matches need xG capture", season, len(to_capture))

    batch_size = 20
    for i, row in enumerate(to_capture):
        fotmob_id = int(row.source_match_id)
        xg_data, error = await provider.get_match_xg(fotmob_id)

        if error:
            metrics["xg_errors"] += 1
            if (i + 1) % 50 == 0:
                logger.warning("[%d] xG error %d/%d: match=%d err=%s",
                               season, i + 1, len(to_capture), row.match_id, error)
            continue

        if xg_data is None:
            metrics["xg_no_data"] += 1
            continue

        # PIT: captured_at = match kickoff + 6h (backfill convention)
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

        # Commit in batches
        if (i + 1) % batch_size == 0:
            await session.commit()
            logger.info("[%d] Progress: %d/%d captured", season, i + 1, len(to_capture))

    await session.commit()
    logger.info("[%d] Phase B done: captured=%d, no_data=%d, errors=%d",
                season, metrics["xg_captured"], metrics["xg_no_data"], metrics["xg_errors"])
    return metrics


async def main():
    from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
    from sqlalchemy.orm import sessionmaker
    from app.etl.fotmob_provider import FotmobProvider
    from app.etl.sofascore_aliases import build_alias_index

    args = parse_args()

    our_league_id = args.league
    fotmob_league_id = LEAGUE_ID_TO_FOTMOB.get(our_league_id)
    if not fotmob_league_id:
        logger.error("No FotMob mapping for league_id=%d. Check LEAGUE_ID_TO_FOTMOB in sota_constants.py",
                      our_league_id)
        sys.exit(1)

    seasons = parse_seasons(args.seasons, our_league_id)
    logger.info("League: %d → FotMob %d | Seasons: %s | Dry-run: %s",
                our_league_id, fotmob_league_id, seasons, args.dry_run)

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
        for season in seasons:
            async with async_session() as session:
                metrics = await backfill_season(
                    session, provider, alias_index, season,
                    our_league_id=our_league_id,
                    fotmob_league_id=fotmob_league_id,
                    dry_run=args.dry_run,
                )
                all_metrics.append(metrics)
                logger.info("[%d] === COMPLETE === %s", season, metrics)
    finally:
        await provider.close()
        await engine.dispose()

    # Summary
    total_linked = sum(m["linked"] for m in all_metrics)
    total_xg = sum(m["xg_captured"] for m in all_metrics)
    logger.info("=" * 60)
    logger.info("BACKFILL COMPLETE: league=%d, %d seasons, %d linked, %d xG captured",
                our_league_id, len(seasons), total_linked, total_xg)
    for m in all_metrics:
        logger.info("  %s", m)


if __name__ == "__main__":
    asyncio.run(main())
