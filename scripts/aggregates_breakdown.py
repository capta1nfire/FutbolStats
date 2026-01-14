#!/usr/bin/env python3
"""Quick script to get aggregates breakdown metrics."""

import asyncio
import os
import sys

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

async def main():
    from sqlalchemy import select, func, distinct
    from app.database import AsyncSessionLocal
    from app.models import LeagueSeasonBaseline, LeagueTeamProfile

    async with AsyncSessionLocal() as session:
        # Baselines breakdown
        total_baselines = await session.execute(
            select(func.count(LeagueSeasonBaseline.id))
        )
        distinct_leagues = await session.execute(
            select(func.count(distinct(LeagueSeasonBaseline.league_id)))
        )
        distinct_league_seasons = await session.execute(
            select(func.count(distinct(
                func.concat(LeagueSeasonBaseline.league_id, '-', LeagueSeasonBaseline.season)
            )))
        )
        distinct_dates = await session.execute(
            select(func.count(distinct(LeagueSeasonBaseline.as_of_date)))
        )

        # Sample of seasons per league
        seasons_sample = await session.execute(
            select(
                LeagueSeasonBaseline.league_id,
                func.array_agg(distinct(LeagueSeasonBaseline.season))
            )
            .group_by(LeagueSeasonBaseline.league_id)
            .limit(10)
        )

        # Profiles breakdown
        total_profiles = await session.execute(
            select(func.count(LeagueTeamProfile.id))
        )
        distinct_teams = await session.execute(
            select(func.count(distinct(LeagueTeamProfile.team_id)))
        )

        print("=== BASELINES BREAKDOWN ===")
        print(f"Total rows: {total_baselines.scalar()}")
        print(f"Distinct league_id: {distinct_leagues.scalar()}")
        print(f"Distinct (league_id, season): {distinct_league_seasons.scalar()}")
        print(f"Distinct as_of_date: {distinct_dates.scalar()}")

        print("\n=== SEASONS PER LEAGUE (sample) ===")
        for row in seasons_sample:
            print(f"  League {row[0]}: seasons {row[1]}")

        print("\n=== PROFILES BREAKDOWN ===")
        print(f"Total rows: {total_profiles.scalar()}")
        print(f"Distinct team_id: {distinct_teams.scalar()}")


if __name__ == "__main__":
    asyncio.run(main())
