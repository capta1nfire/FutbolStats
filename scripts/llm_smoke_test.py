#!/usr/bin/env python3
"""
LLM Narrative Smoke Test

Usage:
    DATABASE_URL=<postgres_url> RUNPOD_API_KEY=<key> NARRATIVE_LLM_ENABLED=true python3 scripts/llm_smoke_test.py [match_id]

If match_id not provided, finds a suitable match with complete stats.
"""

import asyncio
import json
import os
import sys
from typing import Optional

# Ensure app module is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


async def find_test_match(db_url: str) -> Optional[dict]:
    """Find a finished match with complete stats from prediction_outcomes."""
    from sqlalchemy import text
    from sqlalchemy.ext.asyncio import create_async_engine

    if db_url.startswith("postgres://"):
        db_url = db_url.replace("postgres://", "postgresql+asyncpg://", 1)
    elif db_url.startswith("postgresql://") and "+asyncpg" not in db_url:
        db_url = db_url.replace("postgresql://", "postgresql+asyncpg://", 1)

    engine = create_async_engine(db_url)
    async with engine.connect() as conn:
        # Get matches with stats from prediction_outcomes
        result = await conn.execute(
            text("""
            SELECT
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
                m.date::text as match_date
            FROM prediction_outcomes po
            JOIN matches m ON po.match_id = m.id
            JOIN teams t_home ON m.home_team_id = t_home.id
            JOIN teams t_away ON m.away_team_id = t_away.id
            JOIN predictions p ON po.prediction_id = p.id
            WHERE po.home_possession IS NOT NULL
              AND po.total_shots_home IS NOT NULL
              AND po.shots_on_target_home IS NOT NULL
            ORDER BY po.id DESC
            LIMIT 10
        """)
        )
        rows = result.fetchall()
        await engine.dispose()

        print("\n=== Available Matches with Complete Stats ===")
        for r in rows:
            correct = "✓" if r[5] else "✗"
            print(f"  {r[0]}: {r[1]} vs {r[2]} ({r[3]}-{r[4]}) pred={correct}")

        if rows:
            r = rows[0]
            return {
                "match_id": r[0],
                "home_team": r[1],
                "away_team": r[2],
                "home_goals": r[3],
                "away_goals": r[4],
                "prediction_correct": r[5],
                "stats": {
                    "home": {
                        "ball_possession": r[6],
                        "total_shots": r[7],
                        "shots_on_goal": r[9],
                        "expected_goals": r[11],
                    },
                    "away": {
                        "ball_possession": 100 - r[6] if r[6] else None,
                        "total_shots": r[8],
                        "shots_on_goal": r[10],
                        "expected_goals": r[12],
                    },
                },
                "prediction": {
                    "predicted_result": "HOME" if r[13] > r[14] and r[13] > r[15] else ("AWAY" if r[15] > r[13] and r[15] > r[14] else "DRAW"),
                    "confidence": max(r[13], r[14], r[15]),
                    "home_prob": r[13],
                    "draw_prob": r[14],
                    "away_prob": r[15],
                    "correct": r[5],
                },
                "date": r[16],
                "league_name": "Unknown",
            }
        return None


async def run_smoke_test(match_data: dict):
    """Run the LLM smoke test for a specific match."""
    from app.llm.narrative_generator import NarrativeGenerator

    match_id = match_data["match_id"]
    print(f"\n=== Running LLM Smoke Test for match_id={match_id} ===\n")
    print(f"Match: {match_data['home_team']} vs {match_data['away_team']}")
    print(f"Score: {match_data['home_goals']}-{match_data['away_goals']}")
    print(f"Prediction correct: {match_data['prediction_correct']}")
    print(f"Stats: {json.dumps(match_data['stats'], indent=2)}")

    generator = NarrativeGenerator()
    try:
        result = await generator.generate(match_data)

        print(f"\nLLM Status: {result.status}")

        if result.status == "skipped":
            print(f"⚠️  LLM skipped: {result.error}")
            return False

        if result.status == "disabled":
            print("⚠️  LLM disabled - set NARRATIVE_LLM_ENABLED=true")
            return False

        if result.status == "error":
            print(f"❌ LLM error: {result.error}")
            return False

        if result.status == "ok":
            print("\n✓ LLM generation successful!")
            print(f"  Tokens: {result.tokens_in} in / {result.tokens_out} out")
            print(f"  Exec time: {result.exec_ms}ms")
            print(f"  Worker: {result.worker_id}")

            narrative = result.narrative_json
            if narrative:
                print("\n=== Generated Narrative ===")
                print(json.dumps(narrative, indent=2, ensure_ascii=False))

                # Validate tone consistency
                if isinstance(narrative.get("narrative"), dict):
                    tone = narrative["narrative"].get("tone")
                    bet_won = narrative.get("result", {}).get("bet_won")
                    print(f"\n=== Tone Validation ===")
                    print(f"  bet_won: {bet_won}")
                    print(f"  tone: {tone}")

                    expected_tone = "reinforce_win" if bet_won else "mitigate_loss"
                    if tone == expected_tone:
                        print(f"  ✓ Tone matches expected ({expected_tone})")
                    else:
                        print(f"  ⚠️  Tone mismatch! Expected {expected_tone}, got {tone}")

                return True

        return False

    finally:
        await generator.close()


async def main():
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        print("ERROR: DATABASE_URL environment variable required")
        print("Usage: DATABASE_URL=<url> RUNPOD_API_KEY=<key> NARRATIVE_LLM_ENABLED=true python3 scripts/llm_smoke_test.py")
        sys.exit(1)

    runpod_key = os.environ.get("RUNPOD_API_KEY")
    if not runpod_key:
        print("WARNING: RUNPOD_API_KEY not set - LLM calls will fail")

    # Check if LLM is enabled
    llm_enabled = os.environ.get("NARRATIVE_LLM_ENABLED", "false").lower() == "true"
    print(f"NARRATIVE_LLM_ENABLED: {llm_enabled}")

    if not llm_enabled:
        print("\n⚠️  Set NARRATIVE_LLM_ENABLED=true to enable LLM generation")

    # Find a test match
    match_data = await find_test_match(db_url)
    if not match_data:
        print("ERROR: No suitable matches found with complete stats")
        sys.exit(1)

    print(f"\nSelected: {match_data['home_team']} vs {match_data['away_team']}")

    success = await run_smoke_test(match_data)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())
