#!/usr/bin/env python3
"""
build_canonical_odds.py — Populate match_canonical_odds with priority cascade.

Cascade (Market Truth > System Myopia):
  P1: FDUK Pinnacle (raw_odds_1x2 where provider='fduk' AND bookmaker='Pinnacle')
  P2: FDUK B365 / OddsPortal closing (raw_odds_1x2 where provider IN ('fduk','oddsportal') excl. Pinnacle)
  P3: prediction_clv.odds_asof / match_odds_snapshot bet365 frozen at T-0
  P4: odds_snapshots Bet365_live (latest pre-kickoff snapshot)
  P5: predictions.frozen_odds
  P6: matches.odds_home (API-Football live, last resort)
  P7: odds_snapshots avg (API-Football average, absolute last resort)

Rules:
  - First source with all 3 odds non-null wins.
  - is_closing = TRUE only for FT/AET/PEN/AWD matches. FALSE otherwise.
  - P1 rows with is_closing=TRUE are IMMUTABLE (Pinnacle = gold standard).
  - Lower priority sources can be overwritten by higher priority ones.

Usage:
  python scripts/build_canonical_odds.py [--dry-run] [--priority N] [--limit N]
"""

import argparse
import asyncio
import os
import sys
import time
from datetime import datetime

import asyncpg

# Statuses that represent completed matches (closing odds exist)
CLOSING_STATUSES = {'FT', 'AET', 'PEN', 'AWD'}


async def get_conn() -> asyncpg.Connection:
    url = os.environ.get("DATABASE_URL_ASYNC") or os.environ.get("DATABASE_URL")
    if not url:
        print("ERROR: DATABASE_URL_ASYNC or DATABASE_URL not set")
        sys.exit(1)
    if url.startswith("postgresql://") and "asyncpg" not in url:
        url = url.replace("postgresql://", "postgresql+asyncpg://", 1)
    raw = url.replace("postgresql+asyncpg://", "postgresql://")
    return await asyncpg.connect(raw)


async def count_existing(conn: asyncpg.Connection) -> int:
    return await conn.fetchval("SELECT COUNT(*) FROM match_canonical_odds")


async def run_priority(
    conn: asyncpg.Connection,
    priority: int,
    sql: str,
    label: str,
    dry_run: bool = False,
) -> int:
    """Execute an INSERT for a given priority level. Returns rows inserted."""
    if dry_run:
        # Wrap in a CTE to count without inserting
        count_sql = f"SELECT COUNT(*) FROM ({sql.replace('INSERT INTO match_canonical_odds', 'SELECT *').split('ON CONFLICT')[0]}) sub"
        try:
            n = await conn.fetchval(count_sql)
        except Exception:
            n = "?"
        print(f"  P{priority} [{label}]: ~{n} candidates (dry-run)")
        return 0

    result = await conn.execute(sql)
    # Parse "INSERT 0 N" from asyncpg
    try:
        inserted = int(result.split()[-1])
    except (ValueError, IndexError):
        inserted = 0
    print(f"  P{priority} [{label}]: {inserted} rows inserted")
    return inserted


async def build_canonical(dry_run: bool = False, only_priority: int = None, limit: int = 0):
    conn = await get_conn()
    try:
        before = await count_existing(conn)
        total_matches = await conn.fetchval("SELECT COUNT(*) FROM matches")
        print(f"Matches total: {total_matches}")
        print(f"Canonical odds before: {before}")
        print(f"Mode: {'DRY-RUN' if dry_run else 'LIVE'}")
        print()

        total_inserted = 0
        limit_clause = f"LIMIT {limit}" if limit > 0 else ""

        # ---------------------------------------------------------------
        # P1: FDUK Pinnacle — Gold standard (from raw_odds_1x2)
        # ---------------------------------------------------------------
        if only_priority is None or only_priority == 1:
            sql_p1 = f"""
            INSERT INTO match_canonical_odds (match_id, odds_home, odds_draw, odds_away, source, priority, is_closing)
            SELECT DISTINCT ON (r.match_id)
                r.match_id,
                r.odds_home,
                r.odds_draw,
                r.odds_away,
                r.provider || ' (' || r.bookmaker || ')',
                1,
                m.status IN ('FT','AET','PEN','AWD')
            FROM raw_odds_1x2 r
            JOIN matches m ON m.id = r.match_id
            WHERE r.provider = 'fduk'
              AND r.bookmaker = 'Pinnacle'
            ORDER BY r.match_id,
              CASE r.odds_kind WHEN 'closing' THEN 1 ELSE 2 END
            {limit_clause}
            ON CONFLICT (match_id) DO NOTHING
            """
            total_inserted += await run_priority(conn, 1, sql_p1, "FDUK Pinnacle", dry_run)

        # ---------------------------------------------------------------
        # P2: FDUK B365 / OddsPortal / other (from raw_odds_1x2, excl. Pinnacle)
        # ---------------------------------------------------------------
        if only_priority is None or only_priority == 2:
            sql_p2 = f"""
            INSERT INTO match_canonical_odds (match_id, odds_home, odds_draw, odds_away, source, priority, is_closing)
            SELECT DISTINCT ON (r.match_id)
                r.match_id,
                r.odds_home,
                r.odds_draw,
                r.odds_away,
                r.provider || ' (' || r.bookmaker || ')',
                2,
                m.status IN ('FT','AET','PEN','AWD')
            FROM raw_odds_1x2 r
            JOIN matches m ON m.id = r.match_id
            WHERE r.match_id NOT IN (SELECT match_id FROM match_canonical_odds)
              AND NOT (r.provider = 'fduk' AND r.bookmaker = 'Pinnacle')
              AND r.provider IN ('fduk', 'oddsportal')
            ORDER BY r.match_id,
              CASE r.provider
                  WHEN 'fduk' THEN 1
                  WHEN 'oddsportal' THEN 2
                  ELSE 3
              END,
              CASE r.bookmaker
                  WHEN 'Bet365' THEN 1
                  WHEN 'unknown' THEN 2
                  WHEN 'bet-at-home' THEN 3
                  WHEN 'BetInAsia' THEN 4
                  WHEN 'avg' THEN 5
                  ELSE 6
              END,
              CASE r.odds_kind WHEN 'closing' THEN 1 ELSE 2 END
            {limit_clause}
            ON CONFLICT (match_id) DO NOTHING
            """
            total_inserted += await run_priority(conn, 2, sql_p2, "FDUK B365 / OddsPortal", dry_run)

        # ---------------------------------------------------------------
        # P3: prediction_clv odds_asof (bet365 frozen at T-0) + match_odds_snapshot
        # ---------------------------------------------------------------
        if only_priority is None or only_priority == 3:
            # P3a: prediction_clv (Bet365 canonical, closest to true closing)
            sql_p3a = f"""
            INSERT INTO match_canonical_odds (match_id, odds_home, odds_draw, odds_away, source, priority, is_closing)
            SELECT DISTINCT ON (pc.match_id)
                pc.match_id,
                pc.odds_asof_home::double precision,
                pc.odds_asof_draw::double precision,
                pc.odds_asof_away::double precision,
                'prediction_clv (' || pc.canonical_bookmaker || ')',
                3,
                TRUE
            FROM prediction_clv pc
            WHERE pc.odds_asof_home IS NOT NULL
              AND pc.odds_asof_draw IS NOT NULL
              AND pc.odds_asof_away IS NOT NULL
              AND pc.match_id NOT IN (SELECT match_id FROM match_canonical_odds)
            ORDER BY pc.match_id, pc.created_at DESC
            {limit_clause}
            ON CONFLICT (match_id) DO NOTHING
            """
            total_inserted += await run_priority(conn, 3, sql_p3a, "prediction_clv (Bet365)", dry_run)

            # P3b: match_odds_snapshot (bet365 frozen)
            sql_p3b = f"""
            INSERT INTO match_canonical_odds (match_id, odds_home, odds_draw, odds_away, source, priority, is_closing)
            SELECT DISTINCT ON (mos.match_id)
                mos.match_id,
                mos.odds_home,
                mos.odds_draw,
                mos.odds_away,
                'match_odds_snapshot (' || mos.bookmaker || ')',
                3,
                TRUE
            FROM match_odds_snapshot mos
            WHERE mos.odds_home IS NOT NULL
              AND mos.odds_draw IS NOT NULL
              AND mos.odds_away IS NOT NULL
              AND mos.match_id NOT IN (SELECT match_id FROM match_canonical_odds)
            ORDER BY mos.match_id, mos.snapshot_at DESC
            {limit_clause}
            ON CONFLICT (match_id) DO NOTHING
            """
            total_inserted += await run_priority(conn, 3, sql_p3b, "match_odds_snapshot (bet365)", dry_run)

        # ---------------------------------------------------------------
        # P4: odds_snapshots Bet365_live (latest pre-kickoff)
        # ---------------------------------------------------------------
        if only_priority is None or only_priority == 4:
            sql_p4 = f"""
            INSERT INTO match_canonical_odds (match_id, odds_home, odds_draw, odds_away, source, priority, is_closing)
            SELECT DISTINCT ON (os.match_id)
                os.match_id,
                os.odds_home::double precision,
                os.odds_draw::double precision,
                os.odds_away::double precision,
                'odds_snapshots (' || os.bookmaker || ')',
                4,
                (SELECT m.status IN ('FT','AET','PEN','AWD') FROM matches m WHERE m.id = os.match_id)
            FROM odds_snapshots os
            WHERE os.bookmaker = 'Bet365_live'
              AND os.odds_home IS NOT NULL
              AND os.odds_draw IS NOT NULL
              AND os.odds_away IS NOT NULL
              AND os.match_id NOT IN (SELECT match_id FROM match_canonical_odds)
            ORDER BY os.match_id, os.snapshot_at DESC
            {limit_clause}
            ON CONFLICT (match_id) DO NOTHING
            """
            total_inserted += await run_priority(conn, 4, sql_p4, "Bet365_live snapshot", dry_run)

        # ---------------------------------------------------------------
        # P5: predictions.frozen_odds (PIT safe but potentially stale)
        # ---------------------------------------------------------------
        if only_priority is None or only_priority == 5:
            sql_p5 = f"""
            INSERT INTO match_canonical_odds (match_id, odds_home, odds_draw, odds_away, source, priority, is_closing)
            SELECT DISTINCT ON (p.match_id)
                p.match_id,
                p.frozen_odds_home,
                p.frozen_odds_draw,
                p.frozen_odds_away,
                'predictions.frozen_odds',
                5,
                (SELECT m.status IN ('FT','AET','PEN','AWD') FROM matches m WHERE m.id = p.match_id)
            FROM predictions p
            WHERE p.frozen_odds_home IS NOT NULL
              AND p.frozen_odds_draw IS NOT NULL
              AND p.frozen_odds_away IS NOT NULL
              AND p.match_id NOT IN (SELECT match_id FROM match_canonical_odds)
            ORDER BY p.match_id, p.frozen_at DESC NULLS LAST
            {limit_clause}
            ON CONFLICT (match_id) DO NOTHING
            """
            total_inserted += await run_priority(conn, 5, sql_p5, "predictions.frozen_odds", dry_run)

        # ---------------------------------------------------------------
        # P6: matches.odds_home (API-Football live, last resort)
        # ---------------------------------------------------------------
        if only_priority is None or only_priority == 6:
            sql_p6 = f"""
            INSERT INTO match_canonical_odds (match_id, odds_home, odds_draw, odds_away, source, priority, is_closing)
            SELECT
                m.id,
                m.odds_home,
                m.odds_draw,
                m.odds_away,
                'matches.odds (API-Football)',
                6,
                m.status IN ('FT','AET','PEN','AWD')
            FROM matches m
            WHERE m.odds_home IS NOT NULL
              AND m.odds_draw IS NOT NULL
              AND m.odds_away IS NOT NULL
              AND m.id NOT IN (SELECT match_id FROM match_canonical_odds)
            {limit_clause}
            ON CONFLICT (match_id) DO NOTHING
            """
            total_inserted += await run_priority(conn, 6, sql_p6, "API-Football odds_home", dry_run)

        # ---------------------------------------------------------------
        # P7: odds_snapshots avg (API-Football average, absolute last resort)
        # ---------------------------------------------------------------
        if only_priority is None or only_priority == 7:
            sql_p7 = f"""
            INSERT INTO match_canonical_odds (match_id, odds_home, odds_draw, odds_away, source, priority, is_closing)
            SELECT DISTINCT ON (os.match_id)
                os.match_id,
                os.odds_home::double precision,
                os.odds_draw::double precision,
                os.odds_away::double precision,
                'odds_snapshots (avg)',
                7,
                (SELECT m.status IN ('FT','AET','PEN','AWD') FROM matches m WHERE m.id = os.match_id)
            FROM odds_snapshots os
            WHERE os.bookmaker = 'avg'
              AND os.odds_home IS NOT NULL
              AND os.odds_draw IS NOT NULL
              AND os.odds_away IS NOT NULL
              AND os.match_id NOT IN (SELECT match_id FROM match_canonical_odds)
            ORDER BY os.match_id, os.snapshot_at DESC
            {limit_clause}
            ON CONFLICT (match_id) DO NOTHING
            """
            total_inserted += await run_priority(conn, 7, sql_p7, "odds_snapshots avg (fallback)", dry_run)

        # ---------------------------------------------------------------
        # Summary
        # ---------------------------------------------------------------
        after = await count_existing(conn) if not dry_run else before
        print()
        print("=" * 60)
        print(f"Total inserted this run: {total_inserted}")
        print(f"Canonical odds after: {after}")
        print(f"Coverage: {after}/{total_matches} ({after/total_matches*100:.1f}%)")

        # Breakdown by priority
        if not dry_run:
            rows = await conn.fetch("""
                SELECT priority, source, COUNT(*) AS n,
                       COUNT(*) FILTER (WHERE is_closing) AS n_closing
                FROM match_canonical_odds
                GROUP BY priority, source
                ORDER BY priority, n DESC
            """)
            print()
            print("Breakdown by priority:")
            for r in rows:
                print(f"  P{r['priority']} [{r['source']}]: {r['n']} total, {r['n_closing']} closing")

        # Coverage by status
        if not dry_run:
            rows = await conn.fetch("""
                SELECT m.status,
                       COUNT(*) AS total,
                       COUNT(co.match_id) AS con_canonical,
                       ROUND(COUNT(co.match_id)::numeric / NULLIF(COUNT(*), 0) * 100, 1) AS pct
                FROM matches m
                LEFT JOIN match_canonical_odds co ON co.match_id = m.id
                GROUP BY m.status
                ORDER BY COUNT(*) DESC
            """)
            print()
            print("Coverage by match status:")
            for r in rows:
                print(f"  {r['status']:6s}: {r['con_canonical']}/{r['total']} ({r['pct']}%)")

    finally:
        await conn.close()


def main():
    parser = argparse.ArgumentParser(description="Build match_canonical_odds from multi-provider cascade")
    parser.add_argument("--dry-run", action="store_true", help="Count candidates without inserting")
    parser.add_argument("--priority", type=int, choices=[1, 2, 3, 4, 5, 6, 7], help="Run only a specific priority level")
    parser.add_argument("--limit", type=int, default=0, help="Limit rows per priority (0=all)")
    args = parser.parse_args()

    t0 = time.time()
    asyncio.run(build_canonical(
        dry_run=args.dry_run,
        only_priority=args.priority,
        limit=args.limit,
    ))
    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
