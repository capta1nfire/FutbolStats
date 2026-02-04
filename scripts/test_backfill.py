#!/usr/bin/env python3
"""Quick test: backfill 5 fixtures and verify persistence."""
import asyncio
import aiohttp
import json
import os
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

API_KEY = os.environ.get("API_FOOTBALL_KEY")
DATABASE_URL = os.environ.get("DATABASE_URL", "").replace("postgresql://", "postgresql+asyncpg://")

async def test():
    engine = create_async_engine(DATABASE_URL)
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async with async_session() as db:
        # Get 5 fixtures sin stats
        result = await db.execute(text("""
            SELECT m.id, m.external_id, m.home_team_id, m.away_team_id
            FROM matches m
            WHERE m.league_id = 39 AND m.status = 'FT'
              AND (m.stats IS NULL OR m.stats->'home'->>'total_shots' IS NULL)
            ORDER BY m.id LIMIT 5
        """))
        fixtures = [dict(r._mapping) for r in result.fetchall()]
        print(f"Fixtures a procesar: {[f['id'] for f in fixtures]}")

        if not fixtures:
            print("No hay fixtures sin stats!")
            return

        async with aiohttp.ClientSession() as http:
            updated = 0
            for fix in fixtures:
                await asyncio.sleep(0.8)
                url = f"https://v3.football.api-sports.io/fixtures/statistics?fixture={fix['external_id']}"
                async with http.get(url, headers={"x-apisports-key": API_KEY}) as resp:
                    data = await resp.json()

                if not data.get("response"):
                    print(f"  {fix['id']}: NO_DATA")
                    continue

                stats = {"home": {}, "away": {}}
                for j, td in enumerate(data["response"][:2]):
                    side = "home" if j == 0 else "away"
                    for s in td.get("statistics", []):
                        if s["type"] == "Total Shots": stats[side]["total_shots"] = s["value"]
                        elif s["type"] == "Corner Kicks": stats[side]["corner_kicks"] = s["value"]

                await db.execute(
                    text("UPDATE matches SET stats = CAST(:stats AS json) WHERE id = :id"),
                    {"stats": json.dumps(stats), "id": fix["id"]}
                )
                updated += 1
                print(f"  {fix['id']}: shots={stats['home'].get('total_shots')}")

            await db.commit()
            print(f"Committed {updated} updates")

        # Verify
        ids = [f["id"] for f in fixtures]
        result2 = await db.execute(text(f"""
            SELECT id, stats->'home'->>'total_shots' as shots
            FROM matches WHERE id IN ({','.join(str(i) for i in ids)})
        """))
        rows = result2.fetchall()
        verified = sum(1 for r in rows if r[1] is not None)
        print(f"Verificado en DB: {verified}/{updated}")
        print("✅ FUNCIONA" if verified == updated and updated > 0 else "❌ FALLO")

    await engine.dispose()

if __name__ == "__main__":
    asyncio.run(test())
