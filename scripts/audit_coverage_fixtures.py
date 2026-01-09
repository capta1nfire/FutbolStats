#!/usr/bin/env python3
"""
Audit Coverage Fixtures - PIT-first coverage analysis.

Audits fixture/result coverage in DB by league_id and season.
Generates logs/coverage_report.json with detailed metrics.

Read-only, idempotent, safe.

Usage:
    python scripts/audit_coverage_fixtures.py

Requires DATABASE_URL environment variable.
"""

import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Standalone - no app imports required
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv optional if DATABASE_URL already set

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker


def get_database_url() -> str:
    """Get async database URL from environment."""
    url = os.environ.get("DATABASE_URL", "")
    if not url:
        raise ValueError("DATABASE_URL environment variable required")

    if url.startswith("postgres://"):
        return url.replace("postgres://", "postgresql+asyncpg://", 1)
    if url.startswith("postgresql://"):
        return url.replace("postgresql://", "postgresql+asyncpg://", 1)
    return url


async def run_audit():
    """Run the coverage audit and generate report."""
    database_url = get_database_url()
    engine = create_async_engine(database_url, echo=False)
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    report = {
        "generated_at": datetime.utcnow().isoformat(),
        "summary": {},
        "by_league_season": [],
        "worst_by_ft_pct": [],
        "worst_by_gaps": [],
        "missing_or_sparse": [],
    }

    async with async_session() as session:
        # Get coverage stats by league_id and season
        result = await session.execute(text("""
            SELECT
                league_id,
                season,
                COUNT(*) as matches_total,
                COUNT(*) FILTER (WHERE status = 'FT') as ft_count,
                COUNT(*) FILTER (WHERE date IS NULL) as missing_kickoff_count,
                MIN(date) as kickoff_min,
                MAX(date) as kickoff_max,
                COUNT(DISTINCT home_team_id) + COUNT(DISTINCT away_team_id) as teams_raw,
                COUNT(DISTINCT LEAST(home_team_id, away_team_id), GREATEST(home_team_id, away_team_id)) as teams_distinct_approx
            FROM matches
            GROUP BY league_id, season
            ORDER BY league_id, season
        """))
        rows = result.fetchall()

        # Get duplicate external_id counts
        dup_result = await session.execute(text("""
            SELECT external_id, COUNT(*) as cnt
            FROM matches
            GROUP BY external_id
            HAVING COUNT(*) > 1
        """))
        duplicates = dup_result.fetchall()
        total_duplicates = len(duplicates)

        # Get distinct teams count per league/season (more accurate)
        teams_result = await session.execute(text("""
            SELECT
                league_id,
                season,
                COUNT(DISTINCT home_team_id) as home_teams,
                COUNT(DISTINCT away_team_id) as away_teams
            FROM matches
            GROUP BY league_id, season
        """))
        teams_map = {}
        for r in teams_result.fetchall():
            key = (r.league_id, r.season)
            # Union of home and away teams
            teams_map[key] = max(r.home_teams, r.away_teams)

        league_season_data = []
        total_matches = 0
        total_ft = 0

        for row in rows:
            league_id = row.league_id
            season = row.season
            matches_total = row.matches_total
            ft_count = row.ft_count
            missing_kickoff = row.missing_kickoff_count
            kickoff_min = row.kickoff_min
            kickoff_max = row.kickoff_max

            total_matches += matches_total
            total_ft += ft_count

            ft_pct = round((ft_count / matches_total * 100), 2) if matches_total > 0 else 0

            # Calculate days with zero matches (gaps)
            days_with_zero = 0
            if kickoff_min and kickoff_max:
                # Query for gaps in this league/season
                gap_result = await session.execute(text("""
                    WITH date_range AS (
                        SELECT generate_series(
                            DATE(:min_date),
                            DATE(:max_date),
                            '1 day'::interval
                        )::date as match_day
                    ),
                    match_days AS (
                        SELECT DISTINCT DATE(date) as match_day
                        FROM matches
                        WHERE league_id = :league_id
                          AND season = :season
                          AND date IS NOT NULL
                    )
                    SELECT COUNT(*) as gap_days
                    FROM date_range dr
                    LEFT JOIN match_days md ON dr.match_day = md.match_day
                    WHERE md.match_day IS NULL
                """), {
                    "min_date": kickoff_min,
                    "max_date": kickoff_max,
                    "league_id": league_id,
                    "season": season,
                })
                gap_row = gap_result.fetchone()
                days_with_zero = gap_row.gap_days if gap_row else 0

            teams_distinct = teams_map.get((league_id, season), 0)

            entry = {
                "league_id": league_id,
                "season": season,
                "matches_total": matches_total,
                "ft_count": ft_count,
                "ft_pct": ft_pct,
                "missing_kickoff_count": missing_kickoff,
                "kickoff_min": kickoff_min.isoformat() if kickoff_min else None,
                "kickoff_max": kickoff_max.isoformat() if kickoff_max else None,
                "teams_distinct_count": teams_distinct,
                "days_with_zero_matches": days_with_zero,
            }
            league_season_data.append(entry)

        report["by_league_season"] = league_season_data

        # Summary
        report["summary"] = {
            "total_leagues_seasons": len(league_season_data),
            "total_matches": total_matches,
            "total_ft": total_ft,
            "overall_ft_pct": round((total_ft / total_matches * 100), 2) if total_matches > 0 else 0,
            "duplicate_external_id_count": total_duplicates,
        }

        # Worst 10 by FT%
        sorted_by_ft = sorted(league_season_data, key=lambda x: x["ft_pct"])
        report["worst_by_ft_pct"] = sorted_by_ft[:10]

        # Worst 10 by gaps (days with zero matches)
        sorted_by_gaps = sorted(league_season_data, key=lambda x: -x["days_with_zero_matches"])
        report["worst_by_gaps"] = [x for x in sorted_by_gaps[:10] if x["days_with_zero_matches"] > 0]

        # Detect missing/sparse league-seasons (expected leagues with low coverage)
        # Top 5 leagues expected seasons (2018-2025)
        expected_leagues = [39, 140, 135, 78, 61]  # EPL, La Liga, Serie A, Bundesliga, Ligue 1
        expected_seasons = list(range(2018, 2026))

        existing_pairs = {(e["league_id"], e["season"]) for e in league_season_data}
        missing = []
        for lid in expected_leagues:
            for s in expected_seasons:
                if (lid, s) not in existing_pairs:
                    missing.append({"league_id": lid, "season": s, "status": "missing"})
                else:
                    # Check if sparse (< 100 matches for a full season)
                    entry = next((e for e in league_season_data if e["league_id"] == lid and e["season"] == s), None)
                    if entry and entry["matches_total"] < 100:
                        missing.append({
                            "league_id": lid,
                            "season": s,
                            "status": "sparse",
                            "matches_total": entry["matches_total"],
                        })

        report["missing_or_sparse"] = missing

    await engine.dispose()
    return report


def print_summary(report: dict):
    """Print a concise summary table to console."""
    print("\n" + "=" * 70)
    print("FIXTURES COVERAGE AUDIT REPORT")
    print("=" * 70)

    summary = report["summary"]
    print(f"\nGenerated: {report['generated_at']}")
    print(f"\nOVERALL SUMMARY:")
    print(f"  Total league-seasons: {summary['total_leagues_seasons']}")
    print(f"  Total matches:        {summary['total_matches']:,}")
    print(f"  Total FT (finished):  {summary['total_ft']:,}")
    print(f"  Overall FT%:          {summary['overall_ft_pct']}%")
    print(f"  Duplicate external_id:{summary['duplicate_external_id_count']}")

    print(f"\nWORST 10 BY FT% (lowest coverage):")
    print(f"  {'League':>8} {'Season':>6} {'Total':>6} {'FT':>6} {'FT%':>7}")
    print(f"  {'-'*8} {'-'*6} {'-'*6} {'-'*6} {'-'*7}")
    for entry in report["worst_by_ft_pct"][:10]:
        print(f"  {entry['league_id']:>8} {entry['season']:>6} {entry['matches_total']:>6} {entry['ft_count']:>6} {entry['ft_pct']:>6.1f}%")

    print(f"\nWORST 10 BY GAPS (days with zero matches in range):")
    print(f"  {'League':>8} {'Season':>6} {'Gaps':>6} {'Range':<25}")
    print(f"  {'-'*8} {'-'*6} {'-'*6} {'-'*25}")
    for entry in report["worst_by_gaps"][:10]:
        range_str = f"{entry['kickoff_min'][:10] if entry['kickoff_min'] else 'N/A'} to {entry['kickoff_max'][:10] if entry['kickoff_max'] else 'N/A'}"
        print(f"  {entry['league_id']:>8} {entry['season']:>6} {entry['days_with_zero_matches']:>6} {range_str:<25}")

    if report["missing_or_sparse"]:
        print(f"\nMISSING/SPARSE TOP-5 LEAGUE SEASONS (2018-2025):")
        print(f"  {'League':>8} {'Season':>6} {'Status':>10} {'Matches':>8}")
        print(f"  {'-'*8} {'-'*6} {'-'*10} {'-'*8}")
        for entry in report["missing_or_sparse"][:15]:
            matches = entry.get("matches_total", "-")
            print(f"  {entry['league_id']:>8} {entry['season']:>6} {entry['status']:>10} {str(matches):>8}")
    else:
        print(f"\nNo missing/sparse seasons detected for Top-5 leagues.")

    print("\n" + "=" * 70)
    print(f"Full report saved to: logs/coverage_report.json")
    print("=" * 70 + "\n")


async def main():
    """Main entry point."""
    print("Running fixtures coverage audit...")

    report = await run_audit()

    # Ensure logs directory exists
    logs_dir = Path(__file__).parent.parent / "logs"
    logs_dir.mkdir(exist_ok=True)

    # Save JSON report
    report_path = logs_dir / "coverage_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    print_summary(report)

    return report


if __name__ == "__main__":
    asyncio.run(main())
