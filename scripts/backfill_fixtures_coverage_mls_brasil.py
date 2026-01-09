#!/usr/bin/env python3
"""
Backfill Fixtures Coverage - MLS and Brazil Serie A ONLY.

Worker: AM
Scope: league_id 71 (Brazil Serie A) and 253 (MLS)
Seasons: 2016, 2017, 2018, 2019, 2024

PROHIBITED (handled by AF worker):
- league_id 61 (Ligue 1)
- league_id 94, 88, 203

This script:
1. Captures coverage BEFORE backfill
2. Syncs fixtures/teams (NO ODDS) via API-Football
3. Captures coverage AFTER backfill
4. Generates before/after JSON logs per league
5. Logs any unmatched teams

Protocols:
- NO odds: does not touch opening_odds_*, odds_history, odds_snapshots
- Idempotent: upsert by external_id
- Batch commits: every ~100 matches
- Rate limit: respects API_REQUESTS_PER_MINUTE (default 300)

Usage:
    export DATABASE_URL="postgresql://..."
    export RAPIDAPI_KEY="..."
    export API_REQUESTS_PER_MINUTE=300
    python scripts/backfill_fixtures_coverage_mls_brasil.py
"""

import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Add parent directory for app imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

# =============================================================================
# CONFIGURATION - MLS AND BRAZIL SERIE A ONLY
# =============================================================================
# PROHIBITED: 61, 94, 88, 203 (AF worker territory)

BACKFILL_CONFIG = {
    71: {  # Brazil Serie A
        "name": "Brazil Serie A",
        "seasons": [2016, 2017, 2018, 2019, 2024],
    },
    253: {  # MLS
        "name": "MLS",
        "seasons": [2016, 2017, 2018, 2019, 2024],
    },
}

# Forbidden leagues (AF worker)
FORBIDDEN_LEAGUES = {61, 94, 88, 203}

BATCH_SIZE = 100
API_REQUESTS_PER_MINUTE = int(os.environ.get("API_REQUESTS_PER_MINUTE", "300"))


def get_database_url() -> str:
    """Get async database URL."""
    url = os.environ.get("DATABASE_URL", "")
    if not url:
        raise ValueError("DATABASE_URL environment variable required")
    if url.startswith("postgres://"):
        return url.replace("postgres://", "postgresql+asyncpg://", 1)
    if url.startswith("postgresql://"):
        return url.replace("postgresql://", "postgresql+asyncpg://", 1)
    return url


async def get_coverage_stats(session: AsyncSession, league_id: int) -> dict:
    """Get current coverage stats for a league."""
    result = await session.execute(text("""
        SELECT
            league_id,
            season,
            COUNT(*) as matches_total,
            COUNT(*) FILTER (WHERE status = 'FT') as ft_count,
            COUNT(DISTINCT home_team_id) as home_teams,
            COUNT(DISTINCT away_team_id) as away_teams,
            MIN(date) as kickoff_min,
            MAX(date) as kickoff_max
        FROM matches
        WHERE league_id = :league_id
        GROUP BY league_id, season
        ORDER BY season
    """), {"league_id": league_id})

    rows = result.fetchall()
    stats = {
        "league_id": league_id,
        "captured_at": datetime.utcnow().isoformat(),
        "by_season": [],
        "totals": {"matches": 0, "ft": 0, "teams": set()},
    }

    for row in rows:
        entry = {
            "season": row.season,
            "matches_total": row.matches_total,
            "ft_count": row.ft_count,
            "ft_pct": round(row.ft_count / row.matches_total * 100, 2) if row.matches_total else 0,
            "teams_distinct": max(row.home_teams, row.away_teams),
            "kickoff_min": row.kickoff_min.isoformat() if row.kickoff_min else None,
            "kickoff_max": row.kickoff_max.isoformat() if row.kickoff_max else None,
        }
        stats["by_season"].append(entry)
        stats["totals"]["matches"] += row.matches_total
        stats["totals"]["ft"] += row.ft_count

    # Convert set to count for JSON
    stats["totals"]["teams"] = len(stats["totals"].get("teams", set()) or set())

    return stats


async def sync_fixtures_for_league(
    session: AsyncSession,
    provider,
    league_id: int,
    seasons: list[int],
    unmatched_teams: list,
) -> dict:
    """Sync fixtures for a league across multiple seasons. NO ODDS."""
    from app.etl.pipeline import ETLPipeline

    pipeline = ETLPipeline(provider, session)

    total_synced = 0
    total_teams = 0
    errors = []

    for season in seasons:
        try:
            print(f"  Syncing {league_id} season {season}...")

            # Fetch fixtures (NO ODDS - fetch_odds=False)
            fixtures = await provider.get_fixtures(
                league_id=league_id,
                season=season,
            )

            synced_this_season = 0
            for i, match_data in enumerate(fixtures):
                try:
                    await pipeline._upsert_match(match_data)
                    synced_this_season += 1

                    # Batch commit
                    if (i + 1) % BATCH_SIZE == 0:
                        await session.commit()
                        print(f"    Committed batch at {i + 1} matches")

                except Exception as e:
                    error_msg = f"Match {match_data.external_id}: {str(e)}"
                    errors.append(error_msg)
                    # Check for unmatched team
                    if "team" in str(e).lower():
                        unmatched_teams.append({
                            "league_id": league_id,
                            "season": season,
                            "match_external_id": match_data.external_id,
                            "error": str(e),
                        })
                    await session.rollback()
                    continue

            # Final commit for season
            await session.commit()
            total_synced += synced_this_season
            print(f"    Season {season}: {synced_this_season} matches synced")

        except Exception as e:
            errors.append(f"Season {season}: {str(e)}")
            print(f"    ERROR season {season}: {e}")

    return {
        "matches_synced": total_synced,
        "seasons_processed": len(seasons),
        "errors_count": len(errors),
        "errors_sample": errors[:5],
    }


async def run_backfill():
    """Main backfill execution."""
    print("=" * 70)
    print("BACKFILL FIXTURES - MLS & BRAZIL SERIE A")
    print("Worker: AM")
    print("=" * 70)

    # Validate no forbidden leagues
    for lid in BACKFILL_CONFIG.keys():
        if lid in FORBIDDEN_LEAGUES:
            raise ValueError(f"FORBIDDEN: league_id {lid} is AF territory!")

    database_url = get_database_url()
    engine = create_async_engine(database_url, echo=False)
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    # Import provider
    from app.etl.api_football import APIFootballProvider
    provider = APIFootballProvider()

    logs_dir = Path(__file__).parent.parent / "logs"
    logs_dir.mkdir(exist_ok=True)

    summary = {
        "started_at": datetime.utcnow().isoformat(),
        "leagues_processed": [],
        "total_matches_synced": 0,
        "total_errors": 0,
    }

    all_unmatched_teams = {}

    async with async_session() as session:
        for league_id, config in BACKFILL_CONFIG.items():
            print(f"\n{'='*50}")
            print(f"Processing: {config['name']} (league_id={league_id})")
            print(f"Seasons: {config['seasons']}")
            print(f"{'='*50}")

            # BEFORE coverage
            print("\nCapturing BEFORE coverage...")
            before_stats = await get_coverage_stats(session, league_id)

            # Sync fixtures
            print("\nSyncing fixtures (NO ODDS)...")
            unmatched_teams = []
            sync_result = await sync_fixtures_for_league(
                session=session,
                provider=provider,
                league_id=league_id,
                seasons=config["seasons"],
                unmatched_teams=unmatched_teams,
            )

            # AFTER coverage
            print("\nCapturing AFTER coverage...")
            after_stats = await get_coverage_stats(session, league_id)

            # Save before/after log
            coverage_log = {
                "league_id": league_id,
                "league_name": config["name"],
                "worker": "AM",
                "generated_at": datetime.utcnow().isoformat(),
                "before": before_stats,
                "after": after_stats,
                "sync_result": sync_result,
                "delta": {
                    "matches_added": after_stats["totals"]["matches"] - before_stats["totals"]["matches"],
                    "ft_added": after_stats["totals"]["ft"] - before_stats["totals"]["ft"],
                },
            }

            log_path = logs_dir / f"coverage_before_after_{league_id}.json"
            with open(log_path, "w") as f:
                json.dump(coverage_log, f, indent=2, default=str)
            print(f"\nSaved: {log_path}")

            # Save unmatched teams if any
            if unmatched_teams:
                all_unmatched_teams[league_id] = unmatched_teams
                unmatched_path = logs_dir / f"unmatched_teams_{league_id}.json"
                with open(unmatched_path, "w") as f:
                    json.dump(unmatched_teams, f, indent=2)
                print(f"Saved: {unmatched_path} ({len(unmatched_teams)} entries)")

            # Update summary
            summary["leagues_processed"].append({
                "league_id": league_id,
                "name": config["name"],
                "matches_synced": sync_result["matches_synced"],
                "errors": sync_result["errors_count"],
            })
            summary["total_matches_synced"] += sync_result["matches_synced"]
            summary["total_errors"] += sync_result["errors_count"]

    await provider.close()
    await engine.dispose()

    summary["completed_at"] = datetime.utcnow().isoformat()

    # Print summary
    print("\n" + "=" * 70)
    print("BACKFILL SUMMARY")
    print("=" * 70)
    print(f"Worker: AM")
    print(f"Started: {summary['started_at']}")
    print(f"Completed: {summary['completed_at']}")
    print(f"\nLeagues processed: {len(summary['leagues_processed'])}")
    for lg in summary["leagues_processed"]:
        print(f"  - {lg['name']} (id={lg['league_id']}): {lg['matches_synced']} matches, {lg['errors']} errors")
    print(f"\nTOTAL MATCHES SYNCED: {summary['total_matches_synced']}")
    print(f"TOTAL ERRORS: {summary['total_errors']}")

    if all_unmatched_teams:
        print(f"\nUNMATCHED TEAMS:")
        for lid, teams in all_unmatched_teams.items():
            print(f"  - League {lid}: {len(teams)} unmatched")
    else:
        print(f"\nNo unmatched teams.")

    print("\nLogs generated:")
    for league_id in BACKFILL_CONFIG.keys():
        print(f"  - logs/coverage_before_after_{league_id}.json")
        if league_id in all_unmatched_teams:
            print(f"  - logs/unmatched_teams_{league_id}.json")

    print("=" * 70)

    return summary


if __name__ == "__main__":
    asyncio.run(run_backfill())
