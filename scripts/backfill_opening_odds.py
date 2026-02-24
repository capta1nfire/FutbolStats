#!/usr/bin/env python3
"""
Retro-fill matches.opening_odds_* from odds_history earliest records.

ABE Approval: 2026-02-23
Context: sync_odds_for_upcoming_matches() never populated opening_odds_*
from API-Football live sync. This script backfills from odds_history using
the EARLIEST recorded_at per match + PRIORITY_BOOKMAKERS ordering.

Unlike migrate_odds_history_to_matches.py (which uses closing/latest),
this script explicitly picks the FIRST captured odds per bookmaker priority.

Guardrails:
- Only fills where opening_odds_home IS NULL (never overwrites FDUK/OddsPortal)
- Uses PRIORITY_BOOKMAKERS ordering (Bet365 > Pinnacle > 1xBet > ...)
- Dry-run by default
- Only considers is_opening=true records in odds_history

Usage:
    # Dry-run (default) - show what would be updated
    python3 scripts/backfill_opening_odds.py

    # Apply changes
    python3 scripts/backfill_opening_odds.py --apply

    # Filter by league
    python3 scripts/backfill_opening_odds.py --apply --league-id 242

    # Filter by date range
    python3 scripts/backfill_opening_odds.py --apply --since 2026-02-01
"""

import argparse
import logging
import os
import sys

import psycopg2

DATABASE_URL = os.environ.get("DATABASE_URL")
if not DATABASE_URL:
    print("ERROR: DATABASE_URL environment variable is required", file=sys.stderr)
    sys.exit(1)

# Normalize driver for psycopg2
if "+asyncpg" in DATABASE_URL:
    DATABASE_URL = DATABASE_URL.replace("postgresql+asyncpg://", "postgresql://", 1)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Same priority as app/etl/base.py PRIORITY_BOOKMAKERS
PRIORITY_BOOKMAKERS = [
    "Bet365", "Pinnacle", "1xBet", "Unibet",
    "William Hill", "Betfair", "Bwin", "888sport",
]


def run_backfill(apply: bool = False, league_id: int = None, since: str = None):
    logger.info("=" * 60)
    logger.info("Opening Odds Retro-Fill from odds_history")
    logger.info(f"Mode: {'APPLY' if apply else 'DRY-RUN'}")
    logger.info(f"League filter: {league_id or 'ALL'}")
    logger.info(f"Since: {since or 'ALL TIME'}")
    logger.info("=" * 60)

    conn = psycopg2.connect(DATABASE_URL)

    metrics = {
        "candidates": 0,
        "resolved": 0,
        "no_opening_odds": 0,
    }

    try:
        with conn.cursor() as cur:
            # Step 1: Find matches with opening_odds_home IS NULL
            # but that have odds_history records with is_opening=true
            conditions = [
                "m.opening_odds_home IS NULL",
                "m.status IN ('NS', 'FT', 'AET', 'PEN', 'PST')",
            ]
            params = []

            if league_id:
                conditions.append("m.league_id = %s")
                params.append(league_id)

            if since:
                conditions.append("m.date >= %s")
                params.append(since)

            query = f"""
                SELECT m.id, m.league_id, m.date, m.status
                FROM matches m
                WHERE {' AND '.join(conditions)}
                  AND EXISTS (
                      SELECT 1 FROM odds_history oh
                      WHERE oh.match_id = m.id AND oh.is_opening = true
                  )
                ORDER BY m.date ASC
            """

            cur.execute(query, params)
            candidates = cur.fetchall()
            metrics["candidates"] = len(candidates)

            logger.info(f"Found {len(candidates)} candidates with opening odds in odds_history")

            if not candidates:
                logger.info("No candidates to process")
                return metrics

            # Step 2: For each match, pick the best bookmaker from earliest opening odds
            # Build priority map (lower = better)
            priority_map = {b.lower(): i for i, b in enumerate(PRIORITY_BOOKMAKERS)}
            fallback_priority = len(PRIORITY_BOOKMAKERS)

            for match_id, m_league_id, m_date, m_status in candidates:
                # Get all opening odds for this match
                cur.execute("""
                    SELECT source, odds_home, odds_draw, odds_away, recorded_at
                    FROM odds_history
                    WHERE match_id = %s AND is_opening = true
                      AND odds_home > 1.0 AND odds_draw > 1.0 AND odds_away > 1.0
                    ORDER BY recorded_at ASC
                """, (match_id,))

                opening_rows = cur.fetchall()
                if not opening_rows:
                    metrics["no_opening_odds"] += 1
                    continue

                # Pick best bookmaker by priority (from earliest capture batch)
                # All opening rows for a match typically share the same recorded_at
                # (captured in one odds_sync batch)
                earliest_ts = opening_rows[0][4]  # recorded_at of first row

                # Filter to only the earliest batch (same recorded_at)
                earliest_batch = [r for r in opening_rows if r[4] == earliest_ts]

                # Pick by priority
                best = None
                best_priority = fallback_priority + 1
                for source, oh, od, oa, ts in earliest_batch:
                    p = priority_map.get(source.lower(), fallback_priority)
                    if p < best_priority:
                        best = (source, oh, od, oa, ts)
                        best_priority = p

                if not best:
                    # Fallback: first row from earliest batch
                    r = earliest_batch[0]
                    best = (r[0], r[1], r[2], r[3], r[4])

                source, oh, od, oa, ts = best
                metrics["resolved"] += 1

                if apply:
                    cur.execute("""
                        UPDATE matches
                        SET opening_odds_home = %s,
                            opening_odds_draw = %s,
                            opening_odds_away = %s,
                            opening_odds_source = %s,
                            opening_odds_kind = 'earliest_available',
                            opening_odds_column = '1x2',
                            opening_odds_recorded_at = %s,
                            opening_odds_recorded_at_type = 'first_sync'
                        WHERE id = %s AND opening_odds_home IS NULL
                    """, (oh, od, oa, source, ts, match_id))
                    logger.info(f"  OK: match={match_id} date={m_date} league={m_league_id} "
                                f"source={source} odds={oh}/{od}/{oa}")
                else:
                    logger.info(f"  [DRY] match={match_id} date={m_date} league={m_league_id} "
                                f"source={source} odds={oh}/{od}/{oa}")

            if apply:
                conn.commit()
                logger.info("Changes committed")

    finally:
        conn.close()

    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Candidates:       {metrics['candidates']}")
    logger.info(f"Resolved:         {metrics['resolved']}")
    logger.info(f"No opening odds:  {metrics['no_opening_odds']}")
    if not apply:
        logger.info("(DRY-RUN — no updates applied)")
    logger.info("=" * 60)

    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Retro-fill opening_odds from odds_history earliest records"
    )
    parser.add_argument("--apply", action="store_true", help="Apply changes (default is dry-run)")
    parser.add_argument("--league-id", type=int, default=None, help="Filter by league_id")
    parser.add_argument("--since", type=str, default=None, help="Filter by date (YYYY-MM-DD)")

    args = parser.parse_args()
    metrics = run_backfill(apply=args.apply, league_id=args.league_id, since=args.since)

    if metrics["resolved"] == 0 and metrics["candidates"] > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
