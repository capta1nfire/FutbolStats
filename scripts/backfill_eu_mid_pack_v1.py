#!/usr/bin/env python3
"""
EU Mid Pack v1 Backfill - Fixtures and Teams ONLY (NO odds)

Leagues:
- 94: Primeira Liga (Portugal)
- 88: Eredivisie (Netherlands)
- 203: Super Lig (Turkey)

Seasons: 2019-2024
"""

import asyncio
import json
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

# EU Mid Pack v1 leagues
EU_MID_PACK_V1 = {
    94: "Primeira Liga",
    88: "Eredivisie", 
    203: "Super Lig",
}

SEASONS = [2019, 2020, 2021, 2022, 2023, 2024]

async def get_coverage_before(session, league_ids: list[int]) -> dict:
    """Get current coverage for leagues."""
    coverage = {}
    for lid in league_ids:
        # Matches
        res = await session.execute(text("""
            SELECT COUNT(*) FROM matches WHERE league_id = :lid
        """), {"lid": lid})
        matches = res.scalar() or 0
        
        # Teams
        res = await session.execute(text("""
            SELECT COUNT(DISTINCT home_team) + COUNT(DISTINCT away_team) 
            FROM matches WHERE league_id = :lid
        """), {"lid": lid})
        teams_approx = res.scalar() or 0
        
        coverage[lid] = {
            "name": EU_MID_PACK_V1.get(lid, f"League {lid}"),
            "matches": matches,
            "teams_approx": teams_approx,
        }
    return coverage


async def verify_anti_contamination(session, league_ids: list[int]) -> dict:
    """Verify no odds data exists for these leagues."""
    contamination = {}
    
    tables_to_check = [
        ("odds_snapshots", "match_id IN (SELECT id FROM matches WHERE league_id = :lid)"),
        ("market_movement_snapshots", "match_id IN (SELECT id FROM matches WHERE league_id = :lid)"),
        ("lineup_movement_snapshots", "match_id IN (SELECT id FROM matches WHERE league_id = :lid)"),
        ("odds_history", "match_id IN (SELECT id FROM matches WHERE league_id = :lid)"),
    ]
    
    for lid in league_ids:
        contamination[lid] = {}
        for table_name, condition in tables_to_check:
            try:
                res = await session.execute(text(f"""
                    SELECT COUNT(*) FROM {table_name} WHERE {condition}
                """), {"lid": lid})
                count = res.scalar() or 0
                contamination[lid][table_name] = count
            except Exception as e:
                contamination[lid][table_name] = f"error: {e}"
    
    return contamination


async def backfill_fixtures_teams(session, provider, league_id: int, season: int) -> dict:
    """Backfill fixtures and teams for a league/season. NO odds."""
    from app.etl.pipeline import ETLPipeline

    result = {
        "league_id": league_id,
        "season": season,
        "fixtures_fetched": 0,
        "fixtures_upserted": 0,
        "errors": [],
    }

    try:
        # Fetch fixtures for league/season (returns MatchData objects)
        fixtures = await provider.get_fixtures(
            league_id=league_id,
            season=season
        )
        result["fixtures_fetched"] = len(fixtures)
        
        # Upsert matches (NO odds fetching)
        pipeline = ETLPipeline(provider, session)
        for fixture in fixtures:
            try:
                await pipeline._upsert_match(fixture)
                result["fixtures_upserted"] += 1
            except Exception as e:
                result["errors"].append(f"Match {fixture.get('fixture', {}).get('id')}: {str(e)[:50]}")
        
        await session.commit()
        
    except Exception as e:
        result["errors"].append(str(e))
    
    return result


async def main():
    from app.etl.api_football import APIFootballProvider
    
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        print("ERROR: DATABASE_URL not set")
        return
    
    if database_url.startswith("postgresql://"):
        database_url = database_url.replace("postgresql://", "postgresql+asyncpg://", 1)
    
    engine = create_async_engine(database_url)
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    
    league_ids = list(EU_MID_PACK_V1.keys())
    
    log = {
        "started_at": datetime.utcnow().isoformat(),
        "leagues": EU_MID_PACK_V1,
        "seasons": SEASONS,
        "coverage_before": {},
        "backfill_results": [],
        "coverage_after": {},
        "anti_contamination": {},
    }
    
    async with async_session() as session:
        # Coverage BEFORE
        print("=" * 60)
        print("EU MID PACK v1 BACKFILL - Fixtures/Teams ONLY")
        print("=" * 60)
        print(f"Leagues: {EU_MID_PACK_V1}")
        print(f"Seasons: {SEASONS}")
        print()
        
        print("Coverage BEFORE:")
        log["coverage_before"] = await get_coverage_before(session, league_ids)
        for lid, data in log["coverage_before"].items():
            print(f"  {data['name']} ({lid}): {data['matches']} matches")
        print()
        
        # Anti-contamination check BEFORE
        print("Anti-contamination check BEFORE:")
        contamination_before = await verify_anti_contamination(session, league_ids)
        for lid, tables in contamination_before.items():
            name = EU_MID_PACK_V1.get(lid)
            total = sum(v for v in tables.values() if isinstance(v, int))
            print(f"  {name} ({lid}): {total} total odds-related rows")
            for table, count in tables.items():
                if count != 0:
                    print(f"    - {table}: {count}")
        print()
        
        # Backfill
        print("Starting backfill...")
        provider = APIFootballProvider()
        
        try:
            for lid in league_ids:
                name = EU_MID_PACK_V1[lid]
                for season in SEASONS:
                    print(f"  {name} ({lid}) season {season}...", end=" ", flush=True)
                    result = await backfill_fixtures_teams(session, provider, lid, season)
                    log["backfill_results"].append(result)
                    print(f"fetched={result['fixtures_fetched']}, upserted={result['fixtures_upserted']}")
                    if result["errors"]:
                        print(f"    Errors: {result['errors'][:3]}")
                    
                    # Small delay between API calls
                    await asyncio.sleep(0.5)
        finally:
            await provider.close()
        
        print()
        
        # Coverage AFTER
        print("Coverage AFTER:")
        log["coverage_after"] = await get_coverage_before(session, league_ids)
        for lid, data in log["coverage_after"].items():
            before = log["coverage_before"][lid]["matches"]
            after = data["matches"]
            delta = after - before
            print(f"  {data['name']} ({lid}): {after} matches (+{delta})")
        print()
        
        # Anti-contamination check AFTER
        print("Anti-contamination check AFTER:")
        log["anti_contamination"] = await verify_anti_contamination(session, league_ids)
        all_clean = True
        for lid, tables in log["anti_contamination"].items():
            name = EU_MID_PACK_V1.get(lid)
            total = sum(v for v in tables.values() if isinstance(v, int))
            status = "CLEAN" if total == 0 else f"CONTAMINATED ({total} rows)"
            if total != 0:
                all_clean = False
            print(f"  {name} ({lid}): {status}")
        print()
        
        print("=" * 60)
        if all_clean:
            print("SUCCESS: All leagues clean (0 odds contamination)")
        else:
            print("WARNING: Some leagues have odds contamination!")
        print("=" * 60)
    
    await engine.dispose()
    
    # Save log
    log["completed_at"] = datetime.utcnow().isoformat()
    log_path = f"logs/eu_mid_pack_v1_backfill_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
    os.makedirs("logs", exist_ok=True)
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2, default=str)
    print(f"\nLog saved: {log_path}")


if __name__ == "__main__":
    asyncio.run(main())
