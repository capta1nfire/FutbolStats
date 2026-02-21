"""
Backfill Justice Statistics + Financial Autopsy for existing outcomes.

Three operations:
1. prediction_outcomes: compute Y_soft (Dixon-Coles) + Justice Weight W
2. prediction_clv: fill selected_outcome + clv_selected (1,255 rows)
3. post_match_audits: classify autopsy_tag using CLV + xG data

Usage:
    source .env
    python scripts/backfill_justice_autopsy.py [--dry-run]
"""

import argparse
import asyncio
import logging
import math
import os
import sys

import numpy as np
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.ml.justice import compute_y_soft
from app.ml.autopsy import classify_autopsy

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

JUSTICE_ALPHA = 0.5


async def resolve_xg(session: AsyncSession, match_id: int):
    """Resolve best xG: Understat > matches.xg_home (FotMob/FootyStats)."""
    r = await session.execute(text(
        "SELECT xg_home, xg_away FROM match_understat_team "
        "WHERE match_id = :mid AND xg_home IS NOT NULL LIMIT 1"
    ), {"mid": match_id})
    row = r.fetchone()
    if row and row[0] is not None:
        return float(row[0]), float(row[1]), "understat"

    r2 = await session.execute(text(
        "SELECT xg_home, xg_away, xg_source FROM matches "
        "WHERE id = :mid AND xg_home IS NOT NULL"
    ), {"mid": match_id})
    row2 = r2.fetchone()
    if row2 and row2[0] is not None:
        return float(row2[0]), float(row2[1]), row2[2] or "matches"

    return None, None, None


async def backfill_justice_stats(session: AsyncSession, dry_run: bool):
    """Step 1: Backfill Y_soft + W on prediction_outcomes."""
    logger.info("=== Step 1: Backfill Justice Stats on prediction_outcomes ===")

    result = await session.execute(text("""
        SELECT po.id, po.match_id, m.home_goals, m.away_goals
        FROM prediction_outcomes po
        JOIN matches m ON m.id = po.match_id
        WHERE po.y_soft_home IS NULL
          AND m.status IN ('FT', 'AET', 'PEN')
        ORDER BY po.id
    """))
    rows = result.fetchall()
    logger.info(f"  Found {len(rows)} outcomes without justice stats")

    updated = 0
    no_xg = 0
    errors = 0

    for i, (po_id, match_id, home_goals, away_goals) in enumerate(rows):
        try:
            xg_h, xg_a, xg_src = await resolve_xg(session, match_id)
            if xg_h is None or xg_a is None:
                no_xg += 1
                continue

            p_h, p_d, p_a = compute_y_soft(xg_h, xg_a)
            std_dev = float(np.sqrt(xg_h + xg_a + 1.0))
            gd = home_goals - away_goals
            xgd = xg_h - xg_a
            w = float(np.exp(-JUSTICE_ALPHA * abs(gd - xgd) / std_dev))

            if not dry_run:
                await session.execute(text("""
                    UPDATE prediction_outcomes
                    SET y_soft_home = :yh, y_soft_draw = :yd, y_soft_away = :ya,
                        justice_weight = :w, justice_alpha = :alpha,
                        xg_source_ysoft = :src
                    WHERE id = :po_id
                """), {
                    "yh": round(p_h, 6), "yd": round(p_d, 6), "ya": round(p_a, 6),
                    "w": round(w, 6), "alpha": JUSTICE_ALPHA, "src": xg_src,
                    "po_id": po_id,
                })
            updated += 1
        except Exception as e:
            errors += 1
            logger.error(f"  Error on outcome {po_id}: {e}")

        if (i + 1) % 200 == 0:
            logger.info(f"  Progress: {i+1}/{len(rows)} (updated={updated}, no_xg={no_xg})")

    if not dry_run and updated > 0:
        await session.commit()

    logger.info(f"  Done: updated={updated}, no_xg={no_xg}, errors={errors}")
    return {"updated": updated, "no_xg": no_xg, "errors": errors}


async def backfill_clv_selected(session: AsyncSession, dry_run: bool):
    """Step 2: Fill selected_outcome + clv_selected in prediction_clv."""
    logger.info("=== Step 2: Backfill CLV selected_outcome ===")

    result = await session.execute(text("""
        SELECT pc.prediction_id, pc.clv_home, pc.clv_draw, pc.clv_away,
               po.predicted_result
        FROM prediction_clv pc
        JOIN prediction_outcomes po ON po.prediction_id = pc.prediction_id
        WHERE pc.selected_outcome IS NULL
    """))
    rows = result.fetchall()
    logger.info(f"  Found {len(rows)} CLV rows without selected_outcome")

    updated = 0
    for pred_id, clv_h, clv_d, clv_a, predicted_result in rows:
        clv_map = {"home": clv_h, "draw": clv_d, "away": clv_a}
        sel_clv = clv_map.get(predicted_result)
        sel_clv_f = float(sel_clv) if sel_clv is not None else None

        if not dry_run:
            await session.execute(text("""
                UPDATE prediction_clv
                SET selected_outcome = :sel, clv_selected = :clv
                WHERE prediction_id = :pid
            """), {"sel": predicted_result, "clv": sel_clv_f, "pid": pred_id})
        updated += 1

    if not dry_run and updated > 0:
        await session.commit()

    logger.info(f"  Done: updated={updated}")
    return {"updated": updated}


async def backfill_autopsy_tags(session: AsyncSession, dry_run: bool):
    """Step 3: Classify autopsy_tag on post_match_audits."""
    logger.info("=== Step 3: Backfill autopsy_tag on post_match_audits ===")

    result = await session.execute(text("""
        SELECT pma.id as audit_id, po.prediction_id, po.prediction_correct,
               po.predicted_result, po.xg_home, po.xg_away, po.match_id,
               pc.clv_home, pc.clv_draw, pc.clv_away, pc.clv_selected
        FROM post_match_audits pma
        JOIN prediction_outcomes po ON po.id = pma.outcome_id
        LEFT JOIN prediction_clv pc ON pc.prediction_id = po.prediction_id
        WHERE pma.autopsy_tag IS NULL
    """))
    rows = result.fetchall()
    logger.info(f"  Found {len(rows)} audits without autopsy_tag")

    updated = 0
    errors = 0

    for audit_id, pred_id, correct, predicted, xg_h, xg_a, match_id, \
            clv_h, clv_d, clv_a, clv_selected in rows:
        try:
            # Resolve xG (prefer resolved over outcome's xG which may be incomplete)
            xg_h_res, xg_a_res, _ = await resolve_xg(session, match_id)
            # Fallback to outcome's xG if resolve fails
            if xg_h_res is None:
                xg_h_res = float(xg_h) if xg_h is not None else None
            if xg_a_res is None:
                xg_a_res = float(xg_a) if xg_a is not None else None

            # Determine CLV for selected outcome
            if clv_selected is not None:
                sel_clv = float(clv_selected)
            elif clv_h is not None:
                clv_map = {"home": clv_h, "draw": clv_d, "away": clv_a}
                raw = clv_map.get(predicted)
                sel_clv = float(raw) if raw is not None else None
            else:
                sel_clv = None

            tag = classify_autopsy(
                prediction_correct=correct,
                predicted_result=predicted,
                clv_selected=sel_clv,
                xg_home=xg_h_res,
                xg_away=xg_a_res,
            ).value

            if not dry_run:
                await session.execute(text(
                    "UPDATE post_match_audits SET autopsy_tag = :tag WHERE id = :aid"
                ), {"tag": tag, "aid": audit_id})
            updated += 1
        except Exception as e:
            errors += 1
            logger.error(f"  Error on audit {audit_id}: {e}")

        if (updated + errors) % 200 == 0 and (updated + errors) > 0:
            logger.info(f"  Progress: {updated + errors}/{len(rows)} (tagged={updated})")

    if not dry_run and updated > 0:
        await session.commit()

    logger.info(f"  Done: tagged={updated}, errors={errors}")
    return {"tagged": updated, "errors": errors}


async def main():
    parser = argparse.ArgumentParser(description="Backfill Justice Stats + Autopsy Tags")
    parser.add_argument("--dry-run", action="store_true", help="Don't write to DB")
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

    prefix = "[DRY-RUN] " if args.dry_run else ""
    logger.info(f"{prefix}Starting Justice + Autopsy backfill")

    async with async_session() as session:
        # Step 1: Justice Stats
        r1 = await backfill_justice_stats(session, args.dry_run)

        # Step 2: CLV selected_outcome
        r2 = await backfill_clv_selected(session, args.dry_run)

        # Step 3: Autopsy tags
        r3 = await backfill_autopsy_tags(session, args.dry_run)

    await engine.dispose()

    logger.info(f"""
{prefix}=== BACKFILL COMPLETE ===
Justice Stats: {r1['updated']} updated, {r1['no_xg']} no xG, {r1['errors']} errors
CLV Selected:  {r2['updated']} updated
Autopsy Tags:  {r3['tagged']} tagged, {r3['errors']} errors
""")


if __name__ == "__main__":
    asyncio.run(main())
