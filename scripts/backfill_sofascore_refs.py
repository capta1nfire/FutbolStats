#!/usr/bin/env python3
"""
P0 Backfill: Populate match_external_refs (source='sofascore') for historical matches.

Unlike the operational job (sync_sofascore_refs, NS-only ±72h window), this script:
- Scans ALL statuses except CANC/PST (covers FT, AET, PEN, NS, etc.)
- Iterates day-by-day over a configurable date range (default: last 60 days)
- Batch commits every N days for resilience
- Fail-soft: if a day's API call fails, logs and continues to next day
- Produces a final report with per-league breakdown

Reuses 100% of the existing matching logic:
- SofascoreProvider.get_scheduled_events(date)
- calculate_match_score() with alias_index
- get_sofascore_threshold() per league

Usage:
    DATABASE_URL="postgresql://..." python3 scripts/backfill_sofascore_refs.py
    DATABASE_URL="..." python3 scripts/backfill_sofascore_refs.py --days 60 --dry-run
    DATABASE_URL="..." python3 scripts/backfill_sofascore_refs.py --start-date 2025-12-01 --end-date 2026-01-27
    DATABASE_URL="..." python3 scripts/backfill_sofascore_refs.py --league-id 39 --days 30

ATI authorization: GO formal 2026-01-27 (P0 Sofascore Refs — Cobertura Sistémica)
"""

import argparse
import asyncio
import logging
import os
import sys
from collections import defaultdict
from datetime import date, datetime, timedelta

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


async def run_backfill(args):
    """Main backfill logic."""
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        logger.error("DATABASE_URL environment variable not set")
        sys.exit(1)

    if database_url.startswith("postgresql://"):
        database_url = database_url.replace("postgresql://", "postgresql+asyncpg://", 1)

    from sqlalchemy import text
    from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
    from sqlalchemy.orm import sessionmaker

    from app.etl.sofascore_aliases import build_alias_index
    from app.etl.sofascore_provider import (
        SofascoreProvider,
        calculate_match_score,
        get_sofascore_threshold,
    )
    from app.etl.sota_constants import SOFASCORE_SUPPORTED_LEAGUES

    # --- Resolve date range ---
    if args.start_date:
        start_date = datetime.strptime(args.start_date, "%Y-%m-%d").date()
    else:
        start_date = date.today() - timedelta(days=args.days)

    if args.end_date:
        end_date = datetime.strptime(args.end_date, "%Y-%m-%d").date()
    else:
        end_date = date.today()

    # --- Resolve leagues ---
    target_leagues = set(SOFASCORE_SUPPORTED_LEAGUES)
    if args.league_id:
        if args.league_id not in target_leagues:
            logger.warning(
                f"League {args.league_id} is not in SOFASCORE_SUPPORTED_LEAGUES. "
                "Proceeding anyway (override)."
            )
        target_leagues = {args.league_id}

    league_ids_str = ",".join(str(lid) for lid in target_leagues)

    logger.info(f"Backfill range: {start_date} to {end_date} ({(end_date - start_date).days + 1} days)")
    logger.info(f"Target leagues: {len(target_leagues)} ({league_ids_str[:80]}...)")
    logger.info(f"Dry run: {args.dry_run}")
    logger.info(f"Batch size: {args.batch_size} days")

    # --- Build alias index once ---
    alias_index = build_alias_index()
    logger.info(f"Alias index built: {len(alias_index)} entries")

    # --- Initialize provider ---
    provider = SofascoreProvider()

    # --- Metrics ---
    totals = {
        "scanned": 0,
        "already_linked": 0,
        "linked_auto": 0,
        "linked_review": 0,
        "skipped_no_candidates": 0,
        "skipped_low_score": 0,
        "near_misses": 0,
        "errors": 0,
        "api_calls": 0,
        "days_processed": 0,
        "days_failed": 0,
    }
    per_league = defaultdict(lambda: {
        "scanned": 0, "linked_auto": 0, "linked_review": 0,
        "skipped_low_score": 0, "near_misses": 0,
    })

    # --- DB setup ---
    engine = create_async_engine(database_url, echo=False)
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    try:
        async with async_session() as session:
            current_date = start_date
            days_in_batch = 0

            while current_date <= end_date:
                day_start = datetime(current_date.year, current_date.month, current_date.day, 0, 0, 0)
                day_end = day_start + timedelta(days=1)

                try:
                    # 1. Fetch Sofascore events for this day
                    events = await provider.get_scheduled_events(day_start)
                    totals["api_calls"] += 1

                    if not events:
                        logger.debug(f"[{current_date}] No Sofascore events returned")
                        totals["days_processed"] += 1
                        current_date += timedelta(days=1)
                        days_in_batch += 1
                        continue

                    # 2. Query our matches for this day (ALL statuses except CANC/PST, without existing ref)
                    result = await session.execute(text(f"""
                        SELECT
                            m.id AS match_id,
                            m.external_id,
                            m.date AS kickoff_utc,
                            m.league_id,
                            t_home.name AS home_team,
                            t_away.name AS away_team
                        FROM matches m
                        JOIN teams t_home ON m.home_team_id = t_home.id
                        JOIN teams t_away ON m.away_team_id = t_away.id
                        LEFT JOIN match_external_refs mer
                            ON m.id = mer.match_id AND mer.source = 'sofascore'
                        WHERE m.date >= :day_start AND m.date < :day_end
                          AND m.status NOT IN ('PST', 'CANC')
                          AND m.league_id IN ({league_ids_str})
                          AND mer.source_match_id IS NULL
                        ORDER BY m.date ASC
                    """), {"day_start": day_start, "day_end": day_end})

                    matches = result.fetchall()

                    if not matches:
                        logger.debug(f"[{current_date}] No unlinked matches in SOTA leagues")
                        totals["days_processed"] += 1
                        current_date += timedelta(days=1)
                        days_in_batch += 1
                        continue

                    # 3. Match each of our matches against Sofascore candidates
                    day_linked = 0
                    for match in matches:
                        totals["scanned"] += 1
                        per_league[match.league_id]["scanned"] += 1

                        try:
                            best_score = 0.0
                            best_event = None
                            best_matched_by = ""
                            top_candidates = []

                            for event in events:
                                score, matched_by = calculate_match_score(
                                    our_home=match.home_team,
                                    our_away=match.away_team,
                                    our_kickoff=match.kickoff_utc,
                                    sf_home=event["home_team"],
                                    sf_away=event["away_team"],
                                    sf_kickoff=event["kickoff_utc"],
                                    alias_index=alias_index,
                                )

                                top_candidates.append((score, matched_by, event))
                                top_candidates.sort(key=lambda x: x[0], reverse=True)
                                top_candidates = top_candidates[:3]

                                if score > best_score:
                                    best_score = score
                                    best_event = event
                                    best_matched_by = matched_by

                            # Decision
                            threshold = get_sofascore_threshold(match.league_id)

                            if best_score < threshold:
                                totals["skipped_low_score"] += 1
                                per_league[match.league_id]["skipped_low_score"] += 1

                                if best_score >= 0.50:
                                    totals["near_misses"] += 1
                                    per_league[match.league_id]["near_misses"] += 1
                                    candidates_summary = " | ".join(
                                        f"#{i+1} sf={c[2]['home_team']} vs {c[2]['away_team']} "
                                        f"score={c[0]:.3f} ({c[1]})"
                                        for i, c in enumerate(top_candidates[:3])
                                    )
                                    logger.warning(
                                        "[BACKFILL] Near-miss: match=%d league=%d "
                                        "our=%s vs %s | best=%.3f | candidates: %s",
                                        match.match_id, match.league_id,
                                        match.home_team, match.away_team,
                                        best_score, candidates_summary,
                                    )
                                continue

                            # Link it
                            if best_score >= 0.90:
                                totals["linked_auto"] += 1
                                per_league[match.league_id]["linked_auto"] += 1
                            else:
                                best_matched_by += ";needs_review"
                                totals["linked_review"] += 1
                                per_league[match.league_id]["linked_review"] += 1

                            if not args.dry_run:
                                await session.execute(text("""
                                    INSERT INTO match_external_refs
                                        (match_id, source, source_match_id, confidence, matched_by, created_at)
                                    VALUES
                                        (:match_id, 'sofascore', :source_match_id, :confidence, :matched_by, NOW())
                                    ON CONFLICT (match_id, source) DO UPDATE SET
                                        source_match_id = EXCLUDED.source_match_id,
                                        confidence = EXCLUDED.confidence,
                                        matched_by = EXCLUDED.matched_by
                                """), {
                                    "match_id": match.match_id,
                                    "source_match_id": best_event["event_id"],
                                    "confidence": best_score,
                                    "matched_by": best_matched_by,
                                })

                            day_linked += 1

                        except Exception as e:
                            totals["errors"] += 1
                            logger.error(f"[BACKFILL] Error match {match.match_id}: {e}")

                    totals["days_processed"] += 1
                    days_in_batch += 1

                    if day_linked > 0 or len(matches) > 0:
                        logger.info(
                            f"[{current_date}] events={len(events)} matches={len(matches)} "
                            f"linked={day_linked}"
                        )

                    # Batch commit
                    if days_in_batch >= args.batch_size and not args.dry_run:
                        await session.commit()
                        logger.info(
                            f"[BATCH COMMIT] {days_in_batch} days committed. "
                            f"Running totals: linked_auto={totals['linked_auto']}, "
                            f"linked_review={totals['linked_review']}, "
                            f"near_misses={totals['near_misses']}"
                        )
                        days_in_batch = 0

                except Exception as e:
                    totals["days_failed"] += 1
                    logger.error(f"[BACKFILL] Day {current_date} failed: {e}")

                current_date += timedelta(days=1)

            # Final commit
            if not args.dry_run and days_in_batch > 0:
                await session.commit()
                logger.info(f"[FINAL COMMIT] {days_in_batch} remaining days committed")

    finally:
        await provider.close()
        await engine.dispose()

    # --- Print report ---
    print("\n" + "=" * 70)
    print("SOFASCORE REFS BACKFILL REPORT")
    print("=" * 70)
    print(f"  Date range:           {start_date} to {end_date}")
    print(f"  Target leagues:       {len(target_leagues)}")
    print(f"  Dry run:              {args.dry_run}")
    print(f"  Days processed:       {totals['days_processed']}")
    print(f"  Days failed:          {totals['days_failed']}")
    print(f"  API calls:            {totals['api_calls']}")
    print()
    print(f"  Matches scanned:      {totals['scanned']}")
    print(f"  Linked (auto ≥0.90):  {totals['linked_auto']}")
    print(f"  Linked (review):      {totals['linked_review']}")
    total_linked = totals["linked_auto"] + totals["linked_review"]
    print(f"  Total linked:         {total_linked}")
    print(f"  Skipped (low score):  {totals['skipped_low_score']}")
    print(f"  Near-misses (≥0.50):  {totals['near_misses']}")
    print(f"  Errors:               {totals['errors']}")
    print()

    if per_league:
        print("  Per-league breakdown:")
        print(f"  {'League':>8} | {'Scanned':>8} | {'Auto':>6} | {'Review':>6} | {'Low':>6} | {'Near':>6}")
        print("  " + "-" * 55)
        for lid in sorted(per_league.keys()):
            lg = per_league[lid]
            linked = lg["linked_auto"] + lg["linked_review"]
            print(
                f"  {lid:>8} | {lg['scanned']:>8} | {lg['linked_auto']:>6} | "
                f"{lg['linked_review']:>6} | {lg['skipped_low_score']:>6} | {lg['near_misses']:>6}"
            )
        print()

    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="P0 Backfill: Sofascore refs for historical matches (60d)"
    )
    parser.add_argument(
        "--days", type=int, default=60,
        help="Days back from today (default: 60). Overridden by --start-date.",
    )
    parser.add_argument(
        "--start-date", type=str, default=None,
        help="Start date (YYYY-MM-DD). Overrides --days.",
    )
    parser.add_argument(
        "--end-date", type=str, default=None,
        help="End date (YYYY-MM-DD, default: today).",
    )
    parser.add_argument(
        "--league-id", type=int, default=None,
        help="Filter to a specific league ID (default: all SOFASCORE_SUPPORTED_LEAGUES).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Report only, do not write to DB.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=5,
        help="Commit every N days (default: 5).",
    )

    args = parser.parse_args()
    asyncio.run(run_backfill(args))


if __name__ == "__main__":
    main()
