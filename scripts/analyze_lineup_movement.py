#!/usr/bin/env python3
"""
Analyze Lineup-Relative Market Movement

This script analyzes how the market moves around lineup announcement:
- Pre-lineup movement (L-30 to L0): Did odds drift before we detected the lineup?
- Post-lineup movement (L0 to L+10): Did market react to lineup?

Key metric: delta_p = max(|prob_H(t2) - prob_H(t1)|, |prob_D(t2) - prob_D(t1)|, |prob_A(t2) - prob_A(t1)|)
This is computed on NORMALIZED probabilities (after removing overround).

Usage:
    DATABASE_URL="..." python scripts/analyze_lineup_movement.py
"""

import asyncio
import os
import sys
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine


async def analyze_movement():
    """Analyze lineup-relative market movement."""

    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        print("ERROR: DATABASE_URL required")
        sys.exit(1)

    db_url = database_url.replace("postgresql://", "postgresql+asyncpg://")
    engine = create_async_engine(db_url)

    async with engine.connect() as conn:
        # Overall stats
        print("=" * 70)
        print("LINEUP-RELATIVE MARKET MOVEMENT ANALYSIS")
        print("=" * 70)
        print(f"Generated: {datetime.utcnow().isoformat()}")
        print()

        # Total snapshots
        result = await conn.execute(text("""
            SELECT
                snapshot_type,
                COUNT(*) as count,
                AVG(delta_p_vs_baseline) as avg_delta_p,
                MAX(delta_p_vs_baseline) as max_delta_p
            FROM lineup_movement_snapshots
            GROUP BY snapshot_type
            ORDER BY snapshot_type
        """))
        rows = result.fetchall()

        if not rows:
            print("No lineup movement data yet. Wait for captures to accumulate.")
            print()
            print("The system captures snapshots at:")
            print("  L-30: 30 min BEFORE lineup detection")
            print("  L-15: 15 min BEFORE lineup detection")
            print("  L-5:  5 min BEFORE lineup detection")
            print("  L0:   At lineup detection")
            print("  L+5:  5 min AFTER lineup detection")
            print("  L+10: 10 min AFTER lineup detection")
            await engine.dispose()
            return

        print("--- Snapshots by Type ---")
        print(f"{'Type':<10} {'Count':>8} {'Avg delta_p':>12} {'Max delta_p':>12}")
        print("-" * 44)
        for row in rows:
            avg_dp = f"{row.avg_delta_p:.4f}" if row.avg_delta_p else "N/A"
            max_dp = f"{row.max_delta_p:.4f}" if row.max_delta_p else "N/A"
            print(f"{row.snapshot_type:<10} {row.count:>8} {avg_dp:>12} {max_dp:>12}")
        print()

        # Matches with complete tracking
        result = await conn.execute(text("""
            SELECT COUNT(*) FROM matches WHERE lineup_movement_tracked = TRUE
        """))
        complete_count = result.scalar()
        print(f"Matches with complete tracking (L0 + post-lineup): {complete_count}")
        print()

        # Movement analysis using the view
        result = await conn.execute(text("""
            SELECT * FROM v_lineup_movement_analysis
            WHERE delta_p_post_lineup IS NOT NULL
            ORDER BY delta_p_post_lineup DESC
            LIMIT 20
        """))
        movement_rows = result.fetchall()

        if movement_rows:
            print("--- Top 20 Matches by Post-Lineup Movement ---")
            print(f"{'Match':<40} {'delta_p post':>12} {'delta_p pre':>12}")
            print("-" * 66)
            for row in movement_rows:
                match_name = f"{row.home_team} vs {row.away_team}"[:38]
                post = f"{row.delta_p_post_lineup:.4f}" if row.delta_p_post_lineup else "N/A"
                pre = f"{row.delta_p_pre_lineup:.4f}" if row.delta_p_pre_lineup else "N/A"
                print(f"{match_name:<40} {post:>12} {pre:>12}")
            print()

        # Summary statistics
        result = await conn.execute(text("""
            SELECT
                COUNT(*) as total,
                AVG(delta_p_post_lineup) as avg_post,
                AVG(delta_p_pre_lineup) as avg_pre,
                PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY delta_p_post_lineup) as p50_post,
                PERCENTILE_CONT(0.9) WITHIN GROUP (ORDER BY delta_p_post_lineup) as p90_post
            FROM v_lineup_movement_analysis
            WHERE delta_p_post_lineup IS NOT NULL
        """))
        summary = result.fetchone()

        if summary and summary.total > 0:
            print("--- Summary Statistics ---")
            print(f"Total matches analyzed: {summary.total}")
            print(f"Avg delta_p post-lineup (L0 to L+10): {summary.avg_post:.4f}")
            print(f"Avg delta_p pre-lineup (L-30 to L0):  {summary.avg_pre:.4f}" if summary.avg_pre else "")
            print(f"p50 post-lineup movement: {summary.p50_post:.4f}")
            print(f"p90 post-lineup movement: {summary.p90_post:.4f}")
            print()

            # Interpretation guide
            print("--- Interpretation Guide ---")
            print("delta_p < 0.01: Minimal movement (<1 percentage point)")
            print("delta_p 0.01-0.03: Minor movement (1-3 percentage points)")
            print("delta_p 0.03-0.05: Moderate movement (3-5 percentage points)")
            print("delta_p > 0.05: Significant movement (>5 percentage points)")
            print()
            print("If avg delta_p POST-lineup > avg delta_p PRE-lineup:")
            print("  -> Market is reacting TO the lineup announcement")
            print("  -> Our PIT odds capture is happening BEFORE the market adjusts")
            print("  -> This is GOOD for our value betting strategy")
            print()
            print("If avg delta_p POST-lineup < avg delta_p PRE-lineup:")
            print("  -> Market was drifting anyway, lineup had less impact")
            print("  -> Still valid if we're capturing true point-in-time odds")
            print()

        # Limitations
        print("--- IMPORTANT LIMITATIONS ---")
        print("1. Small N: Results are preliminary until N > 100 matches")
        print("2. Survivorship bias: Only includes matches where we detected lineup")
        print("3. Timing variance: L-30 captures may not be exactly 30 min before")
        print("4. Bookmaker variance: Bet365 may move differently than Pinnacle")
        print()

    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(analyze_movement())
