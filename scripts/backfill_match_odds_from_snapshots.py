#!/usr/bin/env python3
"""
Backfill matches.odds_* from odds_snapshots.

P0 Fix (2026-01-14): odds_snapshots was populated but matches.odds_* stayed NULL,
breaking /predictions/upcoming → market_odds null → iOS no Bookie/EV.

Usage:
    # Dry-run (default)
    python scripts/backfill_match_odds_from_snapshots.py

    # Execute backfill
    python scripts/backfill_match_odds_from_snapshots.py --execute

    # Limit to specific matches
    python scripts/backfill_match_odds_from_snapshots.py --match-ids 67561,67562

    # Only NS matches in next 48h
    python scripts/backfill_match_odds_from_snapshots.py --ns-only --hours-ahead 48
"""
import argparse
import os
import sys
from datetime import datetime, timedelta

import psycopg2
from psycopg2.extras import RealDictCursor


def get_connection():
    """Get database connection from environment or CLAUDE.md defaults."""
    db_url = os.environ.get("DATABASE_URL")
    if db_url and db_url.startswith("postgresql"):
        return psycopg2.connect(db_url)

    # Fallback to Railway production
    return psycopg2.connect(
        host="maglev.proxy.rlwy.net",
        port=24997,
        user="postgres",
        password="hzvozcXijUpblVrQshuowYcEGwZnMrfO",
        database="railway"
    )


def find_matches_needing_backfill(
    conn,
    ns_only: bool = True,
    hours_ahead: int = 48,
    hours_back: int = 24,
    match_ids: list = None,
) -> list:
    """
    Find matches with odds_snapshots but NULL matches.odds_*.

    Returns list of dicts with match info and best snapshot to use.
    """
    cur = conn.cursor(cursor_factory=RealDictCursor)

    # Build WHERE clause
    conditions = ["m.odds_home IS NULL"]
    params = {}

    if match_ids:
        conditions.append("m.id = ANY(%(match_ids)s)")
        params["match_ids"] = match_ids
    else:
        if ns_only:
            conditions.append("m.status = 'NS'")

        # Date range
        now = datetime.utcnow()
        range_start = now - timedelta(hours=hours_back)
        range_end = now + timedelta(hours=hours_ahead)
        conditions.append("m.date BETWEEN %(range_start)s AND %(range_end)s")
        params["range_start"] = range_start
        params["range_end"] = range_end

    where_clause = " AND ".join(conditions)

    query = f"""
        SELECT
            m.id as match_id,
            m.external_id,
            m.status,
            m.date,
            m.odds_home as current_odds_home,
            m.odds_recorded_at as current_recorded_at,
            ht.name as home_team,
            at.name as away_team,
            os.id as snapshot_id,
            os.odds_home as snapshot_odds_home,
            os.odds_draw as snapshot_odds_draw,
            os.odds_away as snapshot_odds_away,
            os.snapshot_at,
            os.bookmaker,
            os.odds_freshness
        FROM matches m
        JOIN teams ht ON m.home_team_id = ht.id
        JOIN teams at ON m.away_team_id = at.id
        JOIN odds_snapshots os ON m.id = os.match_id
        WHERE {where_clause}
          AND os.snapshot_type = 'lineup_confirmed'
          AND os.odds_freshness = 'live'
        ORDER BY m.date, os.snapshot_at DESC
    """

    cur.execute(query, params)
    rows = cur.fetchall()
    cur.close()

    # Dedupe: keep most recent snapshot per match
    matches = {}
    for row in rows:
        mid = row["match_id"]
        if mid not in matches:
            matches[mid] = dict(row)

    return list(matches.values())


def backfill_match_odds(conn, match: dict, dry_run: bool = True) -> bool:
    """
    Update matches.odds_* from snapshot.

    Returns True if updated (or would update in dry-run).
    """
    if dry_run:
        print(f"  [DRY-RUN] Would update match {match['match_id']}: "
              f"H={match['snapshot_odds_home']:.2f}, "
              f"D={match['snapshot_odds_draw']:.2f}, "
              f"A={match['snapshot_odds_away']:.2f} "
              f"from {match['bookmaker']}")
        return True

    cur = conn.cursor()
    cur.execute("""
        UPDATE matches
        SET odds_home = %(odds_home)s,
            odds_draw = %(odds_draw)s,
            odds_away = %(odds_away)s,
            odds_recorded_at = %(recorded_at)s
        WHERE id = %(match_id)s
          AND (odds_recorded_at IS NULL OR odds_recorded_at < %(recorded_at)s)
    """, {
        "match_id": match["match_id"],
        "odds_home": float(match["snapshot_odds_home"]),
        "odds_draw": float(match["snapshot_odds_draw"]),
        "odds_away": float(match["snapshot_odds_away"]),
        "recorded_at": match["snapshot_at"],
    })

    updated = cur.rowcount > 0
    cur.close()

    if updated:
        print(f"  [UPDATED] match {match['match_id']}: "
              f"H={match['snapshot_odds_home']:.2f}, "
              f"D={match['snapshot_odds_draw']:.2f}, "
              f"A={match['snapshot_odds_away']:.2f} "
              f"from {match['bookmaker']}")
    else:
        print(f"  [SKIPPED] match {match['match_id']}: already has newer odds")

    return updated


def main():
    parser = argparse.ArgumentParser(
        description="Backfill matches.odds_* from odds_snapshots"
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually execute updates (default: dry-run)"
    )
    parser.add_argument(
        "--match-ids",
        type=str,
        help="Comma-separated match IDs to backfill"
    )
    parser.add_argument(
        "--ns-only",
        action="store_true",
        default=True,
        help="Only process NS (not started) matches"
    )
    parser.add_argument(
        "--all-statuses",
        action="store_true",
        help="Process all statuses, not just NS"
    )
    parser.add_argument(
        "--hours-ahead",
        type=int,
        default=48,
        help="Hours ahead to look for matches (default: 48)"
    )
    parser.add_argument(
        "--hours-back",
        type=int,
        default=24,
        help="Hours back to look for matches (default: 24)"
    )

    args = parser.parse_args()

    dry_run = not args.execute
    ns_only = args.ns_only and not args.all_statuses
    match_ids = None
    if args.match_ids:
        match_ids = [int(x.strip()) for x in args.match_ids.split(",")]

    print("=" * 60)
    print("BACKFILL MATCH ODDS FROM SNAPSHOTS")
    print("=" * 60)
    print(f"Mode:        {'DRY-RUN' if dry_run else 'EXECUTE'}")
    print(f"NS only:     {ns_only}")
    print(f"Hours ahead: {args.hours_ahead}")
    print(f"Hours back:  {args.hours_back}")
    if match_ids:
        print(f"Match IDs:   {match_ids}")
    print("-" * 60)

    conn = get_connection()

    try:
        matches = find_matches_needing_backfill(
            conn,
            ns_only=ns_only,
            hours_ahead=args.hours_ahead,
            hours_back=args.hours_back,
            match_ids=match_ids,
        )

        print(f"\nFound {len(matches)} matches needing backfill:\n")

        if not matches:
            print("No matches found needing backfill.")
            return 0

        updated_count = 0
        for match in matches:
            print(f"\n{match['home_team']} vs {match['away_team']}")
            print(f"  Match ID: {match['match_id']}, Date: {match['date']}, Status: {match['status']}")
            print(f"  Snapshot: {match['snapshot_at']} ({match['bookmaker']}, {match['odds_freshness']})")

            if backfill_match_odds(conn, match, dry_run=dry_run):
                updated_count += 1

        if not dry_run:
            conn.commit()

        print("\n" + "=" * 60)
        print(f"SUMMARY: {'Would update' if dry_run else 'Updated'} {updated_count}/{len(matches)} matches")
        if dry_run:
            print("\nRun with --execute to apply changes.")
        print("=" * 60)

        return 0

    except Exception as e:
        print(f"\nERROR: {e}")
        conn.rollback()
        return 1
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
