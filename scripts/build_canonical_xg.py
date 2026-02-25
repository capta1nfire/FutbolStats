#!/usr/bin/env python3
"""
build_canonical_xg.py — Populate match_canonical_xg with priority cascade.

SSOT for xG data. Replaces distributed COALESCE spaghetti across consumers.

Cascade (GDT + ABE dictamen, 2026-02-25):
  P1: Understat (Opta-grade, shot-level) — Big 5 only
  P2: FotMob (Opta feed, match-level) — 11 Tier-2/3 leagues
  P3: FBRef (StatsBomb/Opta historical) — Big 5 legacy
  P4: FootyStats (browser scraping) — 5 LATAM leagues
  P5: Sofascore (proprietary model, last resort) — mixed

Rules:
  - UPSERT with anti-downgrade: WHERE priority >= EXCLUDED.priority
    (allows same-priority refresh + higher-priority overwrite)
  - League-first preference: natural source for each league preferred.
  - xgot/npxg nullable: populated only where source provides them.

Usage:
  source .env
  python scripts/build_canonical_xg.py [--dry-run] [--priority N] [--limit N]
"""

import argparse
import asyncio
import os
import sys
import time

import asyncpg


# UPSERT clause shared by all priorities (Guardrail 1: auto-healing)
UPSERT_CLAUSE = """
ON CONFLICT (match_id) DO UPDATE SET
    xg_home = EXCLUDED.xg_home,
    xg_away = EXCLUDED.xg_away,
    xgot_home = EXCLUDED.xgot_home,
    xgot_away = EXCLUDED.xgot_away,
    npxg_home = EXCLUDED.npxg_home,
    npxg_away = EXCLUDED.npxg_away,
    source = EXCLUDED.source,
    priority = EXCLUDED.priority,
    captured_at = EXCLUDED.captured_at,
    updated_at = NOW()
WHERE match_canonical_xg.priority >= EXCLUDED.priority
"""


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
    return await conn.fetchval("SELECT COUNT(*) FROM match_canonical_xg")


async def run_priority(
    conn: asyncpg.Connection,
    priority: int,
    sql: str,
    label: str,
    dry_run: bool = False,
) -> int:
    if dry_run:
        # Extract the SELECT part for counting
        select_part = sql.split("ON CONFLICT")[0].replace(
            "INSERT INTO match_canonical_xg", "SELECT *", 1
        )
        # Remove the INSERT column list
        import re
        select_part = re.sub(
            r"SELECT \*\s*\([^)]+\)\s*SELECT",
            "SELECT",
            select_part,
        )
        count_sql = f"SELECT COUNT(*) FROM ({select_part.strip()}) sub"
        try:
            n = await conn.fetchval(count_sql)
        except Exception:
            n = "?"
        print(f"  P{priority} [{label}]: ~{n} candidates (dry-run)")
        return 0

    result = await conn.execute(sql)
    try:
        inserted = int(result.split()[-1])
    except (ValueError, IndexError):
        inserted = 0
    print(f"  P{priority} [{label}]: {inserted} rows upserted")
    return inserted


async def build_canonical(dry_run: bool = False, only_priority: int = None, limit: int = 0):
    conn = await get_conn()
    try:
        before = await count_existing(conn)
        total_ft = await conn.fetchval(
            "SELECT COUNT(*) FROM matches WHERE status = 'FT'"
        )
        print(f"Matches FT total: {total_ft}")
        print(f"Canonical xG before: {before}")
        print(f"Mode: {'DRY-RUN' if dry_run else 'LIVE'}")
        print()

        total_upserted = 0
        limit_clause = f"LIMIT {limit}" if limit > 0 else ""

        # ---------------------------------------------------------------
        # P1: Understat — Opta-grade, shot-level (Big 5 only)
        # Columns: xg_home, xg_away, npxg_home, npxg_away, captured_at
        # ---------------------------------------------------------------
        if only_priority is None or only_priority == 1:
            sql_p1 = f"""
            INSERT INTO match_canonical_xg
                (match_id, xg_home, xg_away, xgot_home, xgot_away,
                 npxg_home, npxg_away, source, priority, captured_at)
            SELECT
                u.match_id,
                u.xg_home,
                u.xg_away,
                NULL,
                NULL,
                u.npxg_home,
                u.npxg_away,
                'understat',
                1,
                u.captured_at
            FROM match_understat_team u
            WHERE u.xg_home IS NOT NULL
              AND u.xg_away IS NOT NULL
            {limit_clause}
            {UPSERT_CLAUSE}
            """
            total_upserted += await run_priority(conn, 1, sql_p1, "Understat (Big 5)", dry_run)

        # ---------------------------------------------------------------
        # P2: FotMob — Opta feed, match-level (11 Tier-2/3 leagues)
        # Columns: xg_home, xg_away, xgot_home, xgot_away, captured_at
        # ---------------------------------------------------------------
        if only_priority is None or only_priority == 2:
            sql_p2 = f"""
            INSERT INTO match_canonical_xg
                (match_id, xg_home, xg_away, xgot_home, xgot_away,
                 npxg_home, npxg_away, source, priority, captured_at)
            SELECT
                f.match_id,
                f.xg_home,
                f.xg_away,
                f.xgot_home,
                f.xgot_away,
                NULL,
                NULL,
                'fotmob',
                2,
                f.captured_at
            FROM match_fotmob_stats f
            WHERE f.xg_home IS NOT NULL
              AND f.xg_away IS NOT NULL
            {limit_clause}
            {UPSERT_CLAUSE}
            """
            total_upserted += await run_priority(conn, 2, sql_p2, "FotMob (Tier 2/3)", dry_run)

        # ---------------------------------------------------------------
        # P3: FBRef — StatsBomb/Opta historical (Big 5 legacy)
        # Source: matches.xg_home/away WHERE xg_source = 'fbref'
        # No xgot/npxg, no captured_at (use match.date + 6h as proxy)
        # ---------------------------------------------------------------
        if only_priority is None or only_priority == 3:
            sql_p3 = f"""
            INSERT INTO match_canonical_xg
                (match_id, xg_home, xg_away, xgot_home, xgot_away,
                 npxg_home, npxg_away, source, priority, captured_at)
            SELECT
                m.id,
                m.xg_home,
                m.xg_away,
                NULL,
                NULL,
                NULL,
                NULL,
                'fbref',
                3,
                m.date + interval '6 hours'
            FROM matches m
            WHERE m.xg_source = 'fbref'
              AND m.xg_home IS NOT NULL
              AND m.xg_away IS NOT NULL
            {limit_clause}
            {UPSERT_CLAUSE}
            """
            total_upserted += await run_priority(conn, 3, sql_p3, "FBRef (legacy Big 5)", dry_run)

        # ---------------------------------------------------------------
        # P4: FootyStats — Browser scraping (5 LATAM leagues)
        # Source: matches.xg_home/away WHERE xg_source = 'footystats'
        # No xgot/npxg, no captured_at (use match.date + 6h as proxy)
        # ---------------------------------------------------------------
        if only_priority is None or only_priority == 4:
            sql_p4 = f"""
            INSERT INTO match_canonical_xg
                (match_id, xg_home, xg_away, xgot_home, xgot_away,
                 npxg_home, npxg_away, source, priority, captured_at)
            SELECT
                m.id,
                m.xg_home,
                m.xg_away,
                NULL,
                NULL,
                NULL,
                NULL,
                'footystats',
                4,
                m.date + interval '6 hours'
            FROM matches m
            WHERE m.xg_source = 'footystats'
              AND m.xg_home IS NOT NULL
              AND m.xg_away IS NOT NULL
            {limit_clause}
            {UPSERT_CLAUSE}
            """
            total_upserted += await run_priority(conn, 4, sql_p4, "FootyStats (LATAM)", dry_run)

        # ---------------------------------------------------------------
        # P5: Sofascore — Proprietary model (last resort)
        # Source: match_sofascore_stats.xg_home/away
        # No xgot/npxg
        # ---------------------------------------------------------------
        if only_priority is None or only_priority == 5:
            sql_p5 = f"""
            INSERT INTO match_canonical_xg
                (match_id, xg_home, xg_away, xgot_home, xgot_away,
                 npxg_home, npxg_away, source, priority, captured_at)
            SELECT
                s.match_id,
                s.xg_home,
                s.xg_away,
                NULL,
                NULL,
                NULL,
                NULL,
                'sofascore',
                5,
                s.captured_at
            FROM match_sofascore_stats s
            WHERE s.xg_home IS NOT NULL
              AND s.xg_away IS NOT NULL
            {limit_clause}
            {UPSERT_CLAUSE}
            """
            total_upserted += await run_priority(conn, 5, sql_p5, "Sofascore (last resort)", dry_run)

        # ---------------------------------------------------------------
        # Summary
        # ---------------------------------------------------------------
        after = await count_existing(conn) if not dry_run else before
        print()
        print("=" * 60)
        print(f"Total upserted this run: {total_upserted}")
        print(f"Canonical xG after: {after}")
        print(f"Coverage: {after}/{total_ft} FT matches ({after/max(total_ft,1)*100:.1f}%)")

        if not dry_run:
            # Breakdown by priority + source
            rows = await conn.fetch("""
                SELECT priority, source, COUNT(*) AS n
                FROM match_canonical_xg
                GROUP BY priority, source
                ORDER BY priority, n DESC
            """)
            print()
            print("Breakdown by priority:")
            for r in rows:
                print(f"  P{r['priority']} [{r['source']}]: {r['n']} matches")

            # Coverage by league
            rows = await conn.fetch("""
                SELECT m.league_id, al.name AS league_name,
                       cxg.source,
                       COUNT(*) AS n
                FROM match_canonical_xg cxg
                JOIN matches m ON m.id = cxg.match_id
                LEFT JOIN admin_leagues al ON al.league_id = m.league_id
                GROUP BY m.league_id, al.name, cxg.source
                ORDER BY COUNT(*) DESC
            """)
            print()
            print("Coverage by league:")
            for r in rows:
                name = r['league_name'] or f"league_{r['league_id']}"
                print(f"  {name:30s} [{r['source']:12s}]: {r['n']} matches")

            # xgot/npxg availability
            xgot_n = await conn.fetchval(
                "SELECT COUNT(*) FROM match_canonical_xg WHERE xgot_home IS NOT NULL"
            )
            npxg_n = await conn.fetchval(
                "SELECT COUNT(*) FROM match_canonical_xg WHERE npxg_home IS NOT NULL"
            )
            print()
            print(f"xGOT populated: {xgot_n}/{after} ({xgot_n/max(after,1)*100:.1f}%)")
            print(f"npxG populated: {npxg_n}/{after} ({npxg_n/max(after,1)*100:.1f}%)")

    finally:
        await conn.close()


def main():
    parser = argparse.ArgumentParser(
        description="Build match_canonical_xg from multi-provider cascade"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Count candidates without inserting",
    )
    parser.add_argument(
        "--priority", type=int, choices=[1, 2, 3, 4, 5],
        help="Run only a specific priority level",
    )
    parser.add_argument(
        "--limit", type=int, default=0,
        help="Limit rows per priority (0=all)",
    )
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
