#!/usr/bin/env python
"""
LLM Narrative Smoke Test

Usage:
    DATABASE_URL=<postgres_url> RUNPOD_API_KEY=<key> python scripts/llm_smoke_test.py [match_id]

If match_id not provided, finds a suitable match with complete stats.
"""

import asyncio
import json
import os
import sys

# Ensure app module is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


async def find_test_match(db_url: str) -> dict | None:
    """Find a finished match with complete stats."""
    from sqlalchemy import text
    from sqlalchemy.ext.asyncio import create_async_engine

    if db_url.startswith("postgres://"):
        db_url = db_url.replace("postgres://", "postgresql+asyncpg://", 1)
    elif db_url.startswith("postgresql://"):
        db_url = db_url.replace("postgresql://", "postgresql+asyncpg://", 1)

    engine = create_async_engine(db_url)
    async with engine.connect() as conn:
        result = await conn.execute(
            text("""
            SELECT m.id, m.home_team, m.away_team, m.home_goals, m.away_goals,
                   m.league_name, m.match_date::text,
                   fp.frozen_probabilities IS NOT NULL as has_prediction
            FROM matches m
            JOIN match_stats ms ON m.id = ms.match_id
            LEFT JOIN frozen_predictions fp ON m.id = fp.match_id
            WHERE m.status IN ('FT', 'AET', 'PEN')
              AND ms.ball_possession IS NOT NULL
              AND ms.total_shots IS NOT NULL
              AND ms.shots_on_goal IS NOT NULL
            ORDER BY m.match_date DESC
            LIMIT 10
        """)
        )
        rows = result.fetchall()
        await engine.dispose()

        print("\n=== Available Matches with Complete Stats ===")
        for r in rows:
            pred_marker = "✓" if r[7] else "✗"
            print(f"  {r[0]}: {r[1]} vs {r[2]} ({r[3]}-{r[4]}) [{r[5]}] pred={pred_marker}")

        # Prefer one with prediction, else first
        for r in rows:
            if r[7]:  # has_prediction
                return {"match_id": r[0], "home_team": r[1], "away_team": r[2]}
        if rows:
            return {"match_id": rows[0][0], "home_team": rows[0][1], "away_team": rows[0][2]}
        return None


async def run_smoke_test(match_id: int):
    """Run the LLM smoke test for a specific match."""
    from app.audit.service import AuditService

    print(f"\n=== Running LLM Smoke Test for match_id={match_id} ===\n")

    service = AuditService()
    try:
        # Run audit which includes LLM generation
        result = await service.audit_match(match_id)

        if result is None:
            print("❌ Audit returned None - match not found or not finished")
            return False

        print(f"Match: {result.home_team} vs {result.away_team}")
        print(f"Score: {result.home_goals}-{result.away_goals}")
        print(f"Audit Status: {result.audit_status}")
        print(f"LLM Status: {result.llm_narrative_status}")

        if result.llm_narrative_status == "skipped":
            print(f"⚠️  LLM skipped: {result.llm_narrative_json}")
            return False

        if result.llm_narrative_status == "disabled":
            print("⚠️  LLM disabled - set NARRATIVE_LLM_ENABLED=true in .env")
            return False

        if result.llm_narrative_status == "error":
            print(f"❌ LLM error: check logs")
            return False

        if result.llm_narrative_status == "ok":
            print("\n✓ LLM generation successful!")
            print(f"  Tokens: {result.llm_narrative_tokens_in} in / {result.llm_narrative_tokens_out} out")
            print(f"  Exec time: {result.llm_narrative_exec_ms}ms")
            print(f"  Worker: {result.llm_narrative_worker_id}")

            narrative = result.llm_narrative_json
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
        await service.close()


async def main():
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        print("ERROR: DATABASE_URL environment variable required")
        print("Usage: DATABASE_URL=<url> RUNPOD_API_KEY=<key> python scripts/llm_smoke_test.py")
        sys.exit(1)

    runpod_key = os.environ.get("RUNPOD_API_KEY")
    if not runpod_key:
        print("WARNING: RUNPOD_API_KEY not set - LLM calls will fail")

    # Check if LLM is enabled
    llm_enabled = os.environ.get("NARRATIVE_LLM_ENABLED", "false").lower() == "true"
    print(f"NARRATIVE_LLM_ENABLED: {llm_enabled}")

    if not llm_enabled:
        print("\n⚠️  Set NARRATIVE_LLM_ENABLED=true to enable LLM generation")

    # Get match_id from args or find one
    if len(sys.argv) > 1:
        match_id = int(sys.argv[1])
    else:
        match_info = await find_test_match(db_url)
        if not match_info:
            print("ERROR: No suitable matches found")
            sys.exit(1)
        match_id = match_info["match_id"]
        print(f"\nSelected: {match_info['home_team']} vs {match_info['away_team']}")

    success = await run_smoke_test(match_id)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())
