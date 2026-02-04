#!/usr/bin/env python3
"""Debug: test backfill with 3 fixtures using main script functions."""
import asyncio
import json
import os
import aiohttp
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

# Import from main script
import sys
sys.path.insert(0, os.path.dirname(__file__))
from backfill_historical_stats import (
    fetch_fixture_stats, merge_stats, parse_stats,
    RateLimiter, ApiStatus, API_KEY, HEADERS
)

DATABASE_URL = os.environ.get("DATABASE_URL", "").replace("postgresql://", "postgresql+asyncpg://")

async def debug_test():
    print("=== DEBUG BACKFILL TEST ===")

    engine = create_async_engine(DATABASE_URL)
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    rate_limiter = RateLimiter(75)

    async with async_session() as db:
        # Get 3 fixtures without stats
        result = await db.execute(text("""
            SELECT m.id, m.external_id, m.home_team_id, m.away_team_id, m.stats
            FROM matches m
            WHERE m.league_id = 39 AND m.status = 'FT'
              AND (m.stats IS NULL OR m.stats->'home'->>'total_shots' IS NULL)
            ORDER BY m.id LIMIT 3
        """))
        fixtures = [dict(r._mapping) for r in result.fetchall()]

        if not fixtures:
            print("No fixtures to process!")
            return

        print(f"Fixtures to test: {[f['id'] for f in fixtures]}")

        async with aiohttp.ClientSession() as http:
            updated_ids = []
            for fix in fixtures:
                status, stats, latency = await fetch_fixture_stats(
                    http, fix["external_id"], rate_limiter
                )
                print(f"  {fix['id']}: {status}, latency={latency:.0f}ms")

                if status in (ApiStatus.OK, ApiStatus.PARTIAL) and stats:
                    merged = merge_stats(
                        fix["stats"], stats,
                        fix["home_team_id"], fix["away_team_id"]
                    )

                    # This is EXACTLY what the main script does
                    await db.execute(
                        text("UPDATE matches SET stats = CAST(:stats AS json) WHERE id = :id"),
                        {"stats": json.dumps(merged), "id": fix["id"]}
                    )
                    updated_ids.append(fix["id"])
                    print(f"    -> Updated with shots={merged['home'].get('total_shots')}")

        # Commit (same as main script final commit)
        print("Committing...")
        await db.commit()
        print("Committed!")

        # Verify
        print("\n=== VERIFICATION ===")
        if updated_ids:
            ids_str = ",".join(str(i) for i in updated_ids)
            result2 = await db.execute(text(f"""
                SELECT id, stats->'home'->>'total_shots' as shots
                FROM matches WHERE id IN ({ids_str})
            """))
            verified = 0
            for row in result2.fetchall():
                has_shots = row[1] is not None
                print(f"  {row[0]}: shots={row[1]} {'✓' if has_shots else '✗'}")
                if has_shots:
                    verified += 1

            print(f"\nResult: {verified}/{len(updated_ids)} verified")
            print("✅ SUCCESS" if verified == len(updated_ids) else "❌ FAILED")

    await engine.dispose()

if __name__ == "__main__":
    asyncio.run(debug_test())
