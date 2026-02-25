"""
Backfill match_justice_stats table for training data.

Populates Dixon-Coles Y_soft + Justice Weight W for ALL matches FT 2023+
that have xG data in match_canonical_xg (SSOT).

Reads from match_canonical_xg which unifies all xG sources with priority cascade:
  P1: Understat, P2: FotMob, P3: FBRef, P4: FootyStats, P5: Sofascore

This table is used by the training script (train_v104_justice.py) to get
per-match sample weights (W) without joining multiple tables at training time.

Usage:
    source .env
    python scripts/backfill_match_justice_stats.py [--dry-run] [--since 2023-01-01]
    python scripts/backfill_match_justice_stats.py --rewrite-existing  # recalc all
"""

import argparse
import asyncio
import logging
import os
import sys
from datetime import datetime

import numpy as np
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.ml.justice import compute_y_soft_batch, compute_justice_weight

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

JUSTICE_ALPHA = 0.5
DIXON_COLES_RHO = -0.15


async def main():
    parser = argparse.ArgumentParser(description="Backfill match_justice_stats")
    parser.add_argument("--dry-run", action="store_true", help="Don't write to DB")
    parser.add_argument("--since", default="2023-01-01", help="Minimum match date")
    parser.add_argument("--rewrite-existing", action="store_true",
                        help="Recalculate and overwrite existing rows (UPSERT)")
    args = parser.parse_args()

    db_url = os.environ.get("DATABASE_URL_ASYNC")
    if not db_url:
        db_url = os.environ.get("DATABASE_URL", "").replace(
            "postgresql://", "postgresql+asyncpg://"
        )
    if not db_url:
        logger.error("No DATABASE_URL_ASYNC or DATABASE_URL in env")
        sys.exit(1)

    engine = create_async_engine(db_url, pool_size=5)
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    since_date = datetime.strptime(args.since, "%Y-%m-%d")
    rewrite = args.rewrite_existing
    prefix = "[DRY-RUN] " if args.dry_run else ""
    logger.info(f"{prefix}Starting match_justice_stats backfill (since {args.since})"
                f"{' [REWRITE MODE]' if rewrite else ''}")

    async with async_session() as session:
        # Get already-populated match_ids to skip
        existing = await session.execute(text(
            "SELECT match_id FROM match_justice_stats"
        ))
        existing_ids = {r[0] for r in existing.fetchall()}
        logger.info(f"  Existing rows: {len(existing_ids)}")

        # ── Single query from canonical SSOT (replaces 4-source cascade) ──
        r_canonical = await session.execute(text("""
            SELECT m.id, m.home_goals, m.away_goals,
                   cxg.xg_home, cxg.xg_away, cxg.source as xg_source
            FROM matches m
            JOIN match_canonical_xg cxg ON cxg.match_id = m.id
            WHERE m.status IN ('FT', 'AET', 'PEN')
              AND m.date >= :since
        """), {"since": since_date})
        all_rows = r_canonical.fetchall()
        logger.info(f"  Canonical xG matches: {len(all_rows)}")

        # Filter rows depending on mode
        if rewrite:
            to_process = all_rows
            skipped = 0
        else:
            to_process = [r for r in all_rows if r[0] not in existing_ids]
            skipped = len(all_rows) - len(to_process)
        logger.info(f"  Total with xG: {len(all_rows)}")
        logger.info(f"  To process: {len(to_process)} (skipping {skipped}"
                     f"{'' if rewrite else ' existing'})")

        if not to_process:
            logger.info("  Nothing to process.")
            await engine.dispose()
            return

        # Vectorized computation
        match_ids = [r[0] for r in to_process]
        home_goals = np.array([float(r[1]) for r in to_process])
        away_goals = np.array([float(r[2]) for r in to_process])
        xg_home = np.array([float(r[3]) for r in to_process])
        xg_away = np.array([float(r[4]) for r in to_process])
        xg_sources = [r[5] for r in to_process]

        # Batch Y_soft computation (vectorized Dixon-Coles)
        y_soft = compute_y_soft_batch(xg_home, xg_away)

        # Batch Justice Weight computation (DRY — uses compute_justice_weight)
        justice_w = compute_justice_weight(
            home_goals, away_goals, xg_home, xg_away, alpha=JUSTICE_ALPHA,
        )

        logger.info(f"  Y_soft computed: shape={y_soft.shape}")
        logger.info(f"  Justice W: mean={justice_w.mean():.4f}, "
                     f"median={np.median(justice_w):.4f}, "
                     f"min={justice_w.min():.4f}, max={justice_w.max():.4f}")

        # Choose SQL template based on mode
        if rewrite:
            upsert_sql = text("""
                INSERT INTO match_justice_stats
                    (match_id, xg_home, xg_away, xg_source,
                     y_soft_home, y_soft_draw, y_soft_away,
                     justice_weight, justice_alpha, dixon_coles_rho)
                VALUES
                    (:mid, :xgh, :xga, :src,
                     :yh, :yd, :ya,
                     :w, :alpha, :rho)
                ON CONFLICT (match_id) DO UPDATE SET
                    xg_home = EXCLUDED.xg_home,
                    xg_away = EXCLUDED.xg_away,
                    xg_source = EXCLUDED.xg_source,
                    y_soft_home = EXCLUDED.y_soft_home,
                    y_soft_draw = EXCLUDED.y_soft_draw,
                    y_soft_away = EXCLUDED.y_soft_away,
                    justice_weight = EXCLUDED.justice_weight,
                    justice_alpha = EXCLUDED.justice_alpha,
                    dixon_coles_rho = EXCLUDED.dixon_coles_rho
            """)
        else:
            upsert_sql = text("""
                INSERT INTO match_justice_stats
                    (match_id, xg_home, xg_away, xg_source,
                     y_soft_home, y_soft_draw, y_soft_away,
                     justice_weight, justice_alpha, dixon_coles_rho)
                VALUES
                    (:mid, :xgh, :xga, :src,
                     :yh, :yd, :ya,
                     :w, :alpha, :rho)
                ON CONFLICT (match_id) DO NOTHING
            """)

        # Process in batches
        BATCH_SIZE = 500
        processed = 0
        for batch_start in range(0, len(to_process), BATCH_SIZE):
            batch_end = min(batch_start + BATCH_SIZE, len(to_process))

            if not args.dry_run:
                for j in range(batch_start, batch_end):
                    await session.execute(upsert_sql, {
                        "mid": match_ids[j],
                        "xgh": round(float(xg_home[j]), 4),
                        "xga": round(float(xg_away[j]), 4),
                        "src": xg_sources[j],
                        "yh": round(float(y_soft[j, 0]), 6),
                        "yd": round(float(y_soft[j, 1]), 6),
                        "ya": round(float(y_soft[j, 2]), 6),
                        "w": round(float(justice_w[j]), 6),
                        "alpha": JUSTICE_ALPHA,
                        "rho": DIXON_COLES_RHO,
                    })
                await session.commit()

            processed += (batch_end - batch_start)
            verb = "Upserted" if rewrite else "Inserted"
            logger.info(f"  {verb} batch {batch_start}-{batch_end} "
                         f"({processed}/{len(to_process)})")

    await engine.dispose()

    mode_label = "REWRITE (upsert)" if rewrite else "INSERT-ONLY"
    logger.info(f"""
{prefix}=== MATCH JUSTICE STATS COMPLETE ===
Mode:                              {mode_label}
Total matches with xG (canonical): {len(all_rows)}
Already in table:                  {len(existing_ids)}
Processed:                         {len(to_process)}
Justice W: mean={justice_w.mean():.4f}, median={np.median(justice_w):.4f}
""")


if __name__ == "__main__":
    asyncio.run(main())
