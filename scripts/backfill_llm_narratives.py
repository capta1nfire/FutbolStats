#!/usr/bin/env python3
"""
Backfill LLM Narratives for completed matches.

Usage:
    DATABASE_URL=<url> RUNPOD_API_KEY=<key> NARRATIVE_LLM_ENABLED=true \
    python3 scripts/backfill_llm_narratives.py --from-date 2026-01-09 --to-date 2026-01-10 --limit 100

This script:
1. Finds matches with PostMatchAudit that don't have LLM narratives yet
2. Builds match_data from PredictionOutcome + Match + Prediction
3. Runs NarrativeGenerator (same gating + prompt v3.2)
4. Updates PostMatchAudit with LLM results
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from typing import Optional

# Ensure app module is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def get_matches_for_backfill(
    db_url: str,
    from_date: str,
    to_date: str,
    limit: int
) -> list:
    """Get matches that need LLM narrative backfill."""
    from sqlalchemy import text
    from sqlalchemy.ext.asyncio import create_async_engine

    if db_url.startswith("postgres://"):
        db_url = db_url.replace("postgres://", "postgresql+asyncpg://", 1)
    elif db_url.startswith("postgresql://") and "+asyncpg" not in db_url:
        db_url = db_url.replace("postgresql://", "postgresql+asyncpg://", 1)

    engine = create_async_engine(db_url)
    async with engine.connect() as conn:
        # Build query with inline values (asyncpg has issues with some param styles)
        query = f"""
            SELECT
                pma.id as audit_id,
                po.match_id,
                t_home.name as home_team,
                t_away.name as away_team,
                m.home_goals,
                m.away_goals,
                po.prediction_correct,
                po.home_possession,
                po.total_shots_home,
                po.total_shots_away,
                po.shots_on_target_home,
                po.shots_on_target_away,
                po.xg_home,
                po.xg_away,
                p.home_prob,
                p.draw_prob,
                p.away_prob,
                m.date::text as match_date,
                'Unknown' as league_name,
                pma.llm_narrative_status
            FROM post_match_audits pma
            JOIN prediction_outcomes po ON pma.outcome_id = po.id
            JOIN matches m ON po.match_id = m.id
            JOIN teams t_home ON m.home_team_id = t_home.id
            JOIN teams t_away ON m.away_team_id = t_away.id
            JOIN predictions p ON po.prediction_id = p.id
            WHERE m.status IN ('FT', 'AET', 'PEN')
              AND m.date >= '{from_date}'::date
              AND m.date < ('{to_date}'::date + interval '1 day')
              AND (pma.llm_narrative_status IS NULL OR pma.llm_narrative_status != 'ok')
            ORDER BY m.date DESC
            LIMIT {limit}
        """
        result = await conn.execute(text(query))
        rows = result.fetchall()
        await engine.dispose()

        matches = []
        for r in rows:
            # Determine predicted result
            home_prob, draw_prob, away_prob = r[14], r[15], r[16]
            if home_prob >= draw_prob and home_prob >= away_prob:
                predicted = "HOME"
                confidence = home_prob
            elif away_prob >= home_prob and away_prob >= draw_prob:
                predicted = "AWAY"
                confidence = away_prob
            else:
                predicted = "DRAW"
                confidence = draw_prob

            matches.append({
                "audit_id": r[0],
                "match_id": r[1],
                "home_team": r[2],
                "away_team": r[3],
                "home_goals": r[4],
                "away_goals": r[5],
                "prediction_correct": r[6],
                "stats": {
                    "home": {
                        "ball_possession": r[7],
                        "total_shots": r[8],
                        "shots_on_goal": r[10],
                        "expected_goals": r[12],
                    },
                    "away": {
                        "ball_possession": 100 - r[7] if r[7] else None,
                        "total_shots": r[9],
                        "shots_on_goal": r[11],
                        "expected_goals": r[13],
                    },
                },
                "prediction": {
                    "predicted_result": predicted,
                    "confidence": confidence,
                    "home_prob": home_prob,
                    "draw_prob": draw_prob,
                    "away_prob": away_prob,
                    "correct": r[6],
                },
                "date": r[17],
                "league_name": r[18],
                "current_status": r[19],
            })

        return matches


async def update_audit_with_llm_result(
    db_url: str,
    audit_id: int,
    result
) -> None:
    """Update PostMatchAudit with LLM result."""
    from sqlalchemy import text
    from sqlalchemy.ext.asyncio import create_async_engine

    if db_url.startswith("postgres://"):
        db_url = db_url.replace("postgres://", "postgresql+asyncpg://", 1)
    elif db_url.startswith("postgresql://") and "+asyncpg" not in db_url:
        db_url = db_url.replace("postgresql://", "postgresql+asyncpg://", 1)

    engine = create_async_engine(db_url)
    async with engine.begin() as conn:
        await conn.execute(
            text("""
            UPDATE post_match_audits SET
                llm_narrative_json = :narrative_json,
                llm_narrative_status = :status,
                llm_narrative_generated_at = :generated_at,
                llm_narrative_model = :model,
                llm_narrative_delay_ms = :delay_ms,
                llm_narrative_exec_ms = :exec_ms,
                llm_narrative_tokens_in = :tokens_in,
                llm_narrative_tokens_out = :tokens_out,
                llm_narrative_worker_id = :worker_id
            WHERE id = :audit_id
        """),
            {
                "audit_id": audit_id,
                "narrative_json": json.dumps(result.narrative_json) if result.narrative_json else None,
                "status": result.status,
                "generated_at": datetime.utcnow(),
                "model": result.model if hasattr(result, 'model') else "qwen-vllm",
                "delay_ms": result.delay_ms,
                "exec_ms": result.exec_ms,
                "tokens_in": result.tokens_in,
                "tokens_out": result.tokens_out,
                "worker_id": result.worker_id,
            }
        )
    await engine.dispose()


async def run_backfill(
    from_date: str,
    to_date: str,
    limit: int,
    dry_run: bool = False
) -> dict:
    """Run the backfill process."""
    from app.llm.narrative_generator import NarrativeGenerator

    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        raise ValueError("DATABASE_URL environment variable required")

    # Get matches needing backfill
    logger.info(f"Finding matches from {from_date} to {to_date} (limit {limit})...")
    matches = await get_matches_for_backfill(db_url, from_date, to_date, limit)
    logger.info(f"Found {len(matches)} matches needing LLM narratives")

    if not matches:
        return {"total": 0, "ok": 0, "skipped": 0, "error": 0}

    # Print matches found
    print(f"\n{'='*60}")
    print(f"Matches to process: {len(matches)}")
    print(f"{'='*60}")
    for m in matches:
        status = m["current_status"] or "null"
        correct = "✓" if m["prediction_correct"] else "✗"
        print(f"  {m['match_id']}: {m['home_team']} vs {m['away_team']} "
              f"({m['home_goals']}-{m['away_goals']}) pred={correct} status={status}")

    if dry_run:
        logger.info("DRY RUN - not processing")
        return {"total": len(matches), "ok": 0, "skipped": 0, "error": 0, "dry_run": True}

    # Process matches
    generator = NarrativeGenerator()
    stats = {"total": len(matches), "ok": 0, "skipped": 0, "error": 0}

    try:
        for i, match in enumerate(matches, 1):
            match_id = match["match_id"]
            audit_id = match["audit_id"]

            print(f"\n[{i}/{len(matches)}] Processing match {match_id}: "
                  f"{match['home_team']} vs {match['away_team']}...")

            try:
                result = await generator.generate(match)

                if result.status == "ok":
                    stats["ok"] += 1
                    logger.info(f"  ✓ OK - tokens: {result.tokens_in}/{result.tokens_out}, "
                               f"exec: {result.exec_ms}ms")

                    # Show narrative preview
                    if result.narrative_json:
                        title = result.narrative_json.get("narrative", {}).get("title", "N/A")
                        tone = result.narrative_json.get("narrative", {}).get("tone", "N/A")
                        print(f"    Title: {title}")
                        print(f"    Tone: {tone}")

                elif result.status == "skipped":
                    stats["skipped"] += 1
                    logger.info(f"  ⊘ SKIPPED - {result.error}")
                else:
                    stats["error"] += 1
                    logger.error(f"  ✗ ERROR - {result.error}")

                # Update database
                await update_audit_with_llm_result(db_url, audit_id, result)

                # Small delay between requests to avoid overwhelming RunPod
                await asyncio.sleep(0.5)

            except Exception as e:
                stats["error"] += 1
                logger.error(f"  ✗ EXCEPTION - {e}")

    finally:
        await generator.close()

    return stats


async def main():
    parser = argparse.ArgumentParser(description="Backfill LLM narratives for matches")
    parser.add_argument("--from-date", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--to-date", required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--limit", type=int, default=100, help="Max matches to process")
    parser.add_argument("--dry-run", action="store_true", help="Just list matches, don't process")
    args = parser.parse_args()

    # Check environment
    if not os.environ.get("DATABASE_URL"):
        print("ERROR: DATABASE_URL environment variable required")
        sys.exit(1)

    llm_enabled = os.environ.get("NARRATIVE_LLM_ENABLED", "false").lower() == "true"
    print(f"NARRATIVE_LLM_ENABLED: {llm_enabled}")

    if not llm_enabled and not args.dry_run:
        print("ERROR: Set NARRATIVE_LLM_ENABLED=true to run backfill")
        sys.exit(1)

    runpod_key = os.environ.get("RUNPOD_API_KEY")
    if not runpod_key and not args.dry_run:
        print("ERROR: RUNPOD_API_KEY environment variable required")
        sys.exit(1)

    max_tokens = os.environ.get("NARRATIVE_LLM_MAX_TOKENS", "1200")
    print(f"NARRATIVE_LLM_MAX_TOKENS: {max_tokens}")

    # Run backfill
    stats = await run_backfill(
        from_date=args.from_date,
        to_date=args.to_date,
        limit=args.limit,
        dry_run=args.dry_run
    )

    # Print summary
    print(f"\n{'='*60}")
    print("BACKFILL SUMMARY")
    print(f"{'='*60}")
    print(f"  Total:   {stats['total']}")
    print(f"  OK:      {stats['ok']}")
    print(f"  Skipped: {stats['skipped']}")
    print(f"  Error:   {stats['error']}")
    if stats.get("dry_run"):
        print("  (DRY RUN - no changes made)")
    print(f"{'='*60}")

    sys.exit(0 if stats["error"] == 0 else 1)


if __name__ == "__main__":
    asyncio.run(main())
