#!/usr/bin/env python3
"""
Migrate orphaned odds from odds_history → matches.opening_odds_*.

Problem: FDUK backfill for Mexico wrote 1,904 odds to odds_history but only
879 reached matches.opening_odds_*. This script migrates the remaining 1,016.

Also renames "OddsPortal (avg of 1)" → "OddsPortal (single-book unknown)"
across all leagues (ABE P0 condition).

Usage:
    # Dry-run audit for all leagues
    python scripts/migrate_odds_history_to_matches.py --all --dry-run

    # Migrate Mexico only
    python scripts/migrate_odds_history_to_matches.py --league 262

    # Rename avg-of-1 sources
    python scripts/migrate_odds_history_to_matches.py --rename-sources

    # Full run: migrate Mexico + rename
    python scripts/migrate_odds_history_to_matches.py --league 262 --rename-sources
"""

from __future__ import annotations

import argparse
import os
import sys

import psycopg2

DATABASE_URL = os.environ.get("DATABASE_URL")
if not DATABASE_URL:
    print("ERROR: DATABASE_URL not set. Run: source .env")
    sys.exit(1)


def get_conn():
    return psycopg2.connect(DATABASE_URL)


def audit_all_leagues(conn) -> None:
    """Show orphaned odds in odds_history for every league."""
    cur = conn.cursor()
    cur.execute("""
        WITH leagues AS (
            SELECT league_id, name FROM admin_leagues WHERE kind = 'league'
        )
        SELECT
            m.league_id,
            l.name,
            oh.source,
            COUNT(DISTINCT oh.match_id) AS total_in_history,
            COUNT(DISTINCT oh.match_id) FILTER (
                WHERE m.opening_odds_home IS NULL AND m.odds_home IS NULL
            ) AS fully_orphaned,
            COUNT(DISTINCT oh.match_id) FILTER (
                WHERE m.opening_odds_home IS NULL AND m.odds_home IS NOT NULL
            ) AS has_closing_only
        FROM odds_history oh
        JOIN matches m ON oh.match_id = m.id
        JOIN leagues l ON m.league_id = l.league_id
        WHERE m.status = 'FT' AND m.date >= '2019-01-01'
          AND oh.odds_home IS NOT NULL
          AND oh.odds_draw IS NOT NULL
          AND oh.odds_away IS NOT NULL
          AND (oh.quarantined IS NOT TRUE)
          AND (oh.tainted IS NOT TRUE)
        GROUP BY m.league_id, l.name, oh.source
        HAVING COUNT(DISTINCT oh.match_id) FILTER (
            WHERE m.opening_odds_home IS NULL
        ) > 0
        ORDER BY COUNT(DISTINCT oh.match_id) FILTER (
            WHERE m.opening_odds_home IS NULL
        ) DESC
    """)

    rows = cur.fetchall()
    if not rows:
        print("No orphaned odds found in any league.")
        return

    print(f"\n{'League':<40} {'Source':<25} {'Total':>6} {'Orphaned':>9} {'ClosingOnly':>12}")
    print("-" * 95)
    for league_id, name, source, total, orphaned, closing_only in rows:
        label = f"({league_id}) {name}"
        print(f"{label:<40} {source:<25} {total:>6} {orphaned:>9} {closing_only:>12}")

    cur.close()


def migrate_league(conn, league_id: int, dry_run: bool = False) -> int:
    """Migrate orphaned odds_history → matches.opening_odds_* for one league."""
    cur = conn.cursor()

    # Pre-check: count orphaned matches
    cur.execute("""
        SELECT COUNT(DISTINCT oh.match_id)
        FROM odds_history oh
        JOIN matches m ON oh.match_id = m.id
        WHERE m.league_id = %s
          AND m.status = 'FT'
          AND m.opening_odds_home IS NULL
          AND oh.odds_home IS NOT NULL
          AND oh.odds_draw IS NOT NULL
          AND oh.odds_away IS NOT NULL
          AND (oh.quarantined IS NOT TRUE)
          AND (oh.tainted IS NOT TRUE)
    """, (league_id,))
    orphaned_count = cur.fetchone()[0]

    if orphaned_count == 0:
        print(f"League {league_id}: 0 orphaned matches. Nothing to migrate.")
        cur.close()
        return 0

    # Pre-migration state
    cur.execute("""
        SELECT
            COUNT(*) FILTER (WHERE odds_home IS NOT NULL OR opening_odds_home IS NOT NULL) AS with_odds,
            COUNT(*) FILTER (WHERE odds_home IS NULL AND opening_odds_home IS NULL) AS without_odds,
            COUNT(*) AS total
        FROM matches
        WHERE league_id = %s AND status = 'FT'
    """, (league_id,))
    pre_with, pre_without, pre_total = cur.fetchone()
    print(f"\nLeague {league_id}: {orphaned_count} orphaned matches to migrate")
    print(f"  PRE:  {pre_with}/{pre_total} with odds ({100*pre_with/pre_total:.1f}%), {pre_without} without")

    if dry_run:
        print("  DRY-RUN: no changes made.")
        cur.close()
        return orphaned_count

    # Migrate using DISTINCT ON (ABE guardrail: is_closing DESC, recorded_at DESC)
    cur.execute("""
        UPDATE matches m
        SET opening_odds_home = sub.odds_home,
            opening_odds_draw = sub.odds_draw,
            opening_odds_away = sub.odds_away,
            opening_odds_source = sub.source || ' (migrated from odds_history)',
            opening_odds_kind = 'closing',
            opening_odds_recorded_at = sub.recorded_at,
            opening_odds_recorded_at_type = 'odds_history_migration'
        FROM (
            SELECT DISTINCT ON (oh.match_id)
                oh.match_id, oh.odds_home, oh.odds_draw, oh.odds_away,
                oh.recorded_at, oh.source
            FROM odds_history oh
            JOIN matches m2 ON oh.match_id = m2.id
            WHERE m2.league_id = %s
              AND m2.status = 'FT'
              AND m2.opening_odds_home IS NULL
              AND oh.odds_home IS NOT NULL
              AND oh.odds_draw IS NOT NULL
              AND oh.odds_away IS NOT NULL
              AND (oh.quarantined IS NOT TRUE)
              AND (oh.tainted IS NOT TRUE)
            ORDER BY oh.match_id, oh.is_closing DESC NULLS LAST, oh.recorded_at DESC
        ) sub
        WHERE m.id = sub.match_id
          AND m.opening_odds_home IS NULL
    """, (league_id,))
    migrated = cur.rowcount
    conn.commit()

    # Post-migration state
    cur.execute("""
        SELECT
            COUNT(*) FILTER (WHERE odds_home IS NOT NULL OR opening_odds_home IS NOT NULL) AS with_odds,
            COUNT(*) FILTER (WHERE odds_home IS NULL AND opening_odds_home IS NULL) AS without_odds
        FROM matches
        WHERE league_id = %s AND status = 'FT'
    """, (league_id,))
    post_with, post_without = cur.fetchone()
    print(f"  MIGRATED: {migrated} matches")
    print(f"  POST: {post_with}/{pre_total} with odds ({100*post_with/pre_total:.1f}%), {post_without} without")

    cur.close()
    return migrated


def rename_sources(conn, dry_run: bool = False) -> int:
    """Rename 'OddsPortal (avg of 1)' → 'OddsPortal (single-book unknown)'."""
    cur = conn.cursor()

    cur.execute("""
        SELECT COUNT(*) FROM matches
        WHERE opening_odds_source = 'OddsPortal (avg of 1)'
    """)
    count = cur.fetchone()[0]

    if count == 0:
        print("\nNo 'OddsPortal (avg of 1)' sources found. Already renamed or absent.")
        cur.close()
        return 0

    print(f"\nRename: {count} matches with 'OddsPortal (avg of 1)'")

    if dry_run:
        # Show per-league breakdown
        cur.execute("""
            SELECT m.league_id, l.name, COUNT(*)
            FROM matches m
            JOIN admin_leagues l ON m.league_id = l.league_id AND l.kind = 'league'
            WHERE m.opening_odds_source = 'OddsPortal (avg of 1)'
            GROUP BY m.league_id, l.name
            ORDER BY COUNT(*) DESC
        """)
        for lid, name, cnt in cur.fetchall():
            print(f"  ({lid}) {name}: {cnt}")
        print("  DRY-RUN: no changes made.")
        cur.close()
        return count

    cur.execute("""
        UPDATE matches
        SET opening_odds_source = 'OddsPortal (single-book unknown)'
        WHERE opening_odds_source = 'OddsPortal (avg of 1)'
    """)
    renamed = cur.rowcount
    conn.commit()
    print(f"  RENAMED: {renamed} matches")

    cur.close()
    return renamed


def main():
    parser = argparse.ArgumentParser(description="Migrate odds_history → matches.opening_odds_*")
    parser.add_argument("--league", type=int, help="League ID to migrate")
    parser.add_argument("--all", action="store_true", help="Audit all leagues for orphaned odds")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without changes")
    parser.add_argument("--rename-sources", action="store_true", help="Rename 'avg of 1' → 'single-book unknown'")
    args = parser.parse_args()

    if not args.league and not args.all and not args.rename_sources:
        parser.print_help()
        sys.exit(1)

    conn = get_conn()

    try:
        if args.all:
            audit_all_leagues(conn)

        if args.league:
            migrate_league(conn, args.league, dry_run=args.dry_run)

        if args.rename_sources:
            rename_sources(conn, dry_run=args.dry_run)

    finally:
        conn.close()


if __name__ == "__main__":
    main()
