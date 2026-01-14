#!/usr/bin/env python3
"""
Verification script for league aggregates.

Tests:
1. Computes baselines and profiles for Liga MX (262)
2. Verifies coherence of calculated stats
3. Shows sample output for audit

Usage:
    python scripts/verify_aggregates.py
"""

import asyncio
import json
import os
import sys
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


async def main():
    """Run verification."""
    # Import here to avoid issues with environment
    from sqlalchemy import select, and_
    from app.database import async_session_maker
    from app.models import Match, Team, LeagueSeasonBaseline, LeagueTeamProfile
    from app.aggregates.service import AggregatesService
    from app.aggregates.refresh_job import refresh_single_league, get_aggregates_status

    print("=" * 80)
    print("AGGREGATES VERIFICATION")
    print("=" * 80)

    async with async_session_maker() as session:
        # 1. Check current status
        print("\n1. Current Status")
        print("-" * 40)
        status = await get_aggregates_status(session)
        print(json.dumps(status, indent=2))

        # 2. Test with Liga MX (262) or any available league
        print("\n2. Testing League Aggregates")
        print("-" * 40)

        # Find a league with enough matches
        result = await session.execute(
            select(Match.league_id, Match.season)
            .where(Match.status.in_(["FT", "AET", "PEN"]))
            .group_by(Match.league_id, Match.season)
            .order_by(Match.league_id)
            .limit(5)
        )
        available_leagues = result.all()

        if not available_leagues:
            print("No finished matches found. Cannot test aggregates.")
            return

        print(f"Available leagues: {available_leagues}")

        # Use first available league
        test_league_id, test_season = available_leagues[0]
        print(f"\nTesting with league_id={test_league_id}, season={test_season}")

        # Compute aggregates
        refresh_result = await refresh_single_league(session, test_league_id, test_season)
        print(f"\nRefresh result:")
        print(json.dumps(refresh_result, indent=2))

        # 3. Verify baseline coherence
        print("\n3. Baseline Coherence Check")
        print("-" * 40)

        baseline_result = await session.execute(
            select(LeagueSeasonBaseline)
            .where(
                and_(
                    LeagueSeasonBaseline.league_id == test_league_id,
                    LeagueSeasonBaseline.season == test_season,
                )
            )
            .order_by(LeagueSeasonBaseline.as_of_date.desc())
            .limit(1)
        )
        baseline = baseline_result.scalar_one_or_none()

        if baseline:
            print(f"League {test_league_id} Season {test_season} Baseline:")
            print(f"  Sample: {baseline.sample_n_matches} matches")
            print(f"  Goals/match: {baseline.goals_avg_per_match}")
            print(f"  Over 1.5: {baseline.over_1_5_pct}%")
            print(f"  Over 2.5: {baseline.over_2_5_pct}%")
            print(f"  Over 3.5: {baseline.over_3_5_pct}%")
            print(f"  BTTS: {baseline.btts_yes_pct}%")
            print(f"  Corners/match: {baseline.corners_avg_per_match}")
            print(f"  Yellow cards/match: {baseline.yellow_cards_avg_per_match}")

            # Coherence checks
            checks_passed = True

            # Over percentages should be monotonically decreasing
            if not (baseline.over_1_5_pct >= baseline.over_2_5_pct >= baseline.over_3_5_pct):
                print("  [FAIL] Over percentages not monotonically decreasing!")
                checks_passed = False
            else:
                print("  [OK] Over percentages are coherent")

            # Goals avg should be reasonable (0.5 - 5.0)
            if not (0.5 <= baseline.goals_avg_per_match <= 5.0):
                print(f"  [WARN] Goals avg seems unusual: {baseline.goals_avg_per_match}")
            else:
                print("  [OK] Goals avg is reasonable")

            # BTTS should be reasonable (10-90%)
            if not (10 <= baseline.btts_yes_pct <= 90):
                print(f"  [WARN] BTTS seems unusual: {baseline.btts_yes_pct}%")
            else:
                print("  [OK] BTTS is reasonable")

            if checks_passed:
                print("\n  ✓ All coherence checks passed")
        else:
            print("No baseline found")

        # 4. Verify team profiles
        print("\n4. Team Profiles Sample")
        print("-" * 40)

        profiles_result = await session.execute(
            select(LeagueTeamProfile)
            .where(
                and_(
                    LeagueTeamProfile.league_id == test_league_id,
                    LeagueTeamProfile.season == test_season,
                    LeagueTeamProfile.min_sample_ok == True,
                )
            )
            .order_by(LeagueTeamProfile.rank_best_attack)
            .limit(5)
        )
        profiles = profiles_result.scalars().all()

        if profiles:
            print(f"Top 5 teams by attack (league {test_league_id}):")
            for p in profiles:
                # Get team name
                team_result = await session.execute(
                    select(Team.name).where(Team.id == p.team_id)
                )
                team_name = team_result.scalar() or f"Team {p.team_id}"

                print(f"\n  {p.rank_best_attack}. {team_name}")
                print(f"     Matches: {p.matches_played}, Confidence: {p.rank_confidence}")
                print(f"     Goals for/match: {p.goals_for_per_match}")
                print(f"     Goals against/match: {p.goals_against_per_match}")
                print(f"     Clean sheet: {p.clean_sheet_pct}%")
                print(f"     Over 2.5: {p.over_2_5_pct}%")
                if p.rank_worst_defense:
                    print(f"     Defense rank: {p.rank_worst_defense}/{p.total_teams_in_league}")
        else:
            print("No profiles with sufficient sample found")

        # 5. Test context retrieval
        print("\n5. Context Retrieval Test")
        print("-" * 40)

        service = AggregatesService(session)

        league_ctx = await service.get_league_context(test_league_id, test_season)
        print(f"League context for {test_league_id}:")
        print(json.dumps(league_ctx, indent=2, default=str) if league_ctx else "None")

        if profiles:
            team_ctx = await service.get_team_context(test_league_id, test_season, profiles[0].team_id)
            print(f"\nTeam context for team {profiles[0].team_id}:")
            print(json.dumps(team_ctx, indent=2, default=str) if team_ctx else "None")

        print("\n" + "=" * 80)
        print("VERIFICATION COMPLETE")
        print("=" * 80)


if __name__ == "__main__":
    # Set DATABASE_URL for local testing if not set
    if "DATABASE_URL" not in os.environ:
        os.environ["DATABASE_URL"] = "sqlite+aiosqlite:///./futbolstats.db"

    asyncio.run(main())
