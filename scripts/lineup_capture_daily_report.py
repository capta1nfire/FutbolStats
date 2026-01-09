#!/usr/bin/env python3
"""
Daily Lineup Capture Report

Generates a summary of lineup_confirmed captures for the last 24 hours.
Saves output to logs/lineup_capture_daily_YYYYMMDD.json

Usage:
    DATABASE_URL="..." python scripts/lineup_capture_daily_report.py

    # Custom time range (hours)
    DATABASE_URL="..." python scripts/lineup_capture_daily_report.py --hours 48
"""

import asyncio
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

# League ID to name mapping
LEAGUE_NAMES = {
    39: "Premier League",
    140: "La Liga",
    135: "Serie A",
    78: "Bundesliga",
    61: "Ligue 1",
    94: "Primeira Liga",
    88: "Eredivisie",
    203: "Super Lig",
    136: "Serie B",
    45: "FA Cup",
    253: "Major League Soccer",
    71: "Serie A (Brazil)",
}


async def generate_daily_report(hours: int = 24):
    """Generate lineup capture report for the last N hours."""

    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        print("ERROR: DATABASE_URL required")
        sys.exit(1)

    db_url = database_url.replace("postgresql://", "postgresql+asyncpg://")
    engine = create_async_engine(db_url)

    cutoff = datetime.utcnow() - timedelta(hours=hours)

    report = {
        "generated_at": datetime.utcnow().isoformat(),
        "period_hours": hours,
        "cutoff_utc": cutoff.isoformat(),
        "summary": {},
        "by_league": {},
        "timing_histogram": {
            "10-30min": 0,
            "30-45min": 0,
            "45-60min": 0,
            "60-75min": 0,
            "75-90min": 0,
            "other": 0
        },
        "timing_percentiles": {},
        "odds_freshness": {
            "live": 0,
            "cached": 0,
            "unknown": 0
        },
        "by_bookmaker": {},
        "captures": []
    }

    # Collect all timing values for percentile calculation
    all_minutes = []

    async with engine.connect() as conn:
        # Get all lineup_confirmed captures in period
        result = await conn.execute(text("""
            SELECT
                os.match_id,
                os.created_at as captured_at,
                EXTRACT(EPOCH FROM (m.date - os.created_at)) / 60 as minutes_to_kickoff,
                os.odds_freshness,
                os.bookmaker,
                os.odds_home,
                os.odds_draw,
                os.odds_away,
                m.date as kickoff,
                m.league_id,
                ht.name as home_team,
                at.name as away_team
            FROM odds_snapshots os
            JOIN matches m ON os.match_id = m.id
            JOIN teams ht ON m.home_team_id = ht.id
            JOIN teams at ON m.away_team_id = at.id
            WHERE os.snapshot_type = 'lineup_confirmed'
              AND os.created_at >= :cutoff
            ORDER BY os.created_at DESC
        """), {"cutoff": cutoff})

        rows = result.fetchall()

        # Process each capture
        for row in rows:
            minutes = row.minutes_to_kickoff or 0
            all_minutes.append(float(minutes))

            # Timing histogram (more granular)
            if 10 <= minutes < 30:
                report["timing_histogram"]["10-30min"] += 1
            elif 30 <= minutes < 45:
                report["timing_histogram"]["30-45min"] += 1
            elif 45 <= minutes < 60:
                report["timing_histogram"]["45-60min"] += 1
            elif 60 <= minutes < 75:
                report["timing_histogram"]["60-75min"] += 1
            elif 75 <= minutes <= 90:
                report["timing_histogram"]["75-90min"] += 1
            else:
                report["timing_histogram"]["other"] += 1

            # Odds freshness
            freshness = row.odds_freshness or "unknown"
            if freshness in report["odds_freshness"]:
                report["odds_freshness"][freshness] += 1
            else:
                report["odds_freshness"]["unknown"] += 1

            # By bookmaker
            bookmaker = row.bookmaker or "unknown"
            report["by_bookmaker"][bookmaker] = report["by_bookmaker"].get(bookmaker, 0) + 1

            # Get league name from mapping
            league_id = row.league_id
            league_name = LEAGUE_NAMES.get(league_id, f"League {league_id}")

            # By league (with timing stats)
            league_key = f"{league_id}_{league_name}"
            if league_key not in report["by_league"]:
                report["by_league"][league_key] = {
                    "league_id": league_id,
                    "league_name": league_name,
                    "count": 0,
                    "matches": [],
                    "timing_minutes": []  # For per-league timing stats
                }
            report["by_league"][league_key]["count"] += 1
            report["by_league"][league_key]["timing_minutes"].append(float(minutes))

            # Add match to list if not already there
            match_entry = f"{row.home_team} vs {row.away_team}"
            if match_entry not in report["by_league"][league_key]["matches"]:
                report["by_league"][league_key]["matches"].append(match_entry)

            # Capture detail
            report["captures"].append({
                "match_id": row.match_id,
                "match": f"{row.home_team} vs {row.away_team}",
                "league": league_name,
                "league_id": league_id,
                "kickoff_utc": row.kickoff.isoformat() if row.kickoff else None,
                "captured_at": row.captured_at.isoformat() if row.captured_at else None,
                "minutes_to_kickoff": float(minutes) if minutes else 0,
                "odds_freshness": freshness,
                "bookmaker": bookmaker,
                "odds": {
                    "home": float(row.odds_home) if row.odds_home else None,
                    "draw": float(row.odds_draw) if row.odds_draw else None,
                    "away": float(row.odds_away) if row.odds_away else None
                }
            })

        # Calculate percentiles
        if all_minutes:
            sorted_mins = sorted(all_minutes)
            n = len(sorted_mins)

            def percentile(data, p):
                k = (len(data) - 1) * p / 100
                f = int(k)
                c = f + 1 if f + 1 < len(data) else f
                return data[f] + (data[c] - data[f]) * (k - f) if c != f else data[f]

            report["timing_percentiles"] = {
                "p10": round(percentile(sorted_mins, 10), 1),
                "p25": round(percentile(sorted_mins, 25), 1),
                "p50": round(percentile(sorted_mins, 50), 1),
                "p75": round(percentile(sorted_mins, 75), 1),
                "p90": round(percentile(sorted_mins, 90), 1),
                "min": round(min(sorted_mins), 1),
                "max": round(max(sorted_mins), 1),
                "mean": round(sum(sorted_mins) / n, 1)
            }

        # Calculate bin percentages
        total = len(rows)
        if total > 0:
            report["timing_bin_pct"] = {
                k: round(v / total * 100, 1)
                for k, v in report["timing_histogram"].items()
            }
        else:
            report["timing_bin_pct"] = {}

        # Summary stats
        report["summary"] = {
            "total_captures": total,
            "unique_leagues": len(report["by_league"]),
            "live_pct": round(report["odds_freshness"]["live"] / total * 100, 1) if total > 0 else 0,
            "ideal_window_pct": round(
                (report["timing_histogram"]["45-60min"] + report["timing_histogram"]["60-75min"] + report["timing_histogram"]["75-90min"]) / total * 100, 1
            ) if total > 0 else 0,
            "p50_minutes": report["timing_percentiles"].get("p50", None),
            "p90_minutes": report["timing_percentiles"].get("p90", None)
        }

        # Get total historical for context
        hist_result = await conn.execute(text("""
            SELECT COUNT(*) FROM odds_snapshots WHERE snapshot_type = 'lineup_confirmed'
        """))
        report["summary"]["total_historical"] = hist_result.scalar()

        # Calculate per-league timing stats
        for league_key, league_data in report["by_league"].items():
            timings = league_data["timing_minutes"]
            if timings:
                sorted_t = sorted(timings)
                league_data["timing_stats"] = {
                    "p50": round(sorted_t[len(sorted_t) // 2], 1),
                    "min": round(min(sorted_t), 1),
                    "max": round(max(sorted_t), 1),
                    "mean": round(sum(sorted_t) / len(sorted_t), 1)
                }
            del league_data["timing_minutes"]  # Remove raw data

        # Simplify by_league for output
        report["by_league"] = {
            v["league_name"]: {
                "count": v["count"],
                "matches": v["matches"],
                "timing_stats": v.get("timing_stats", {})
            }
            for v in report["by_league"].values()
        }

        # Note: PARTIAL_LINEUP logs are in Railway logs, not DB
        # We document this for future reference
        report["diagnostic_notes"] = {
            "partial_lineup_logging": "Enabled in scheduler.py - check Railway logs for PARTIAL_LINEUP entries",
            "market_movement_tracking": "Enabled - check market_movement_snapshots table for T60/T30/T15/T5 data"
        }

    await engine.dispose()
    return report


def print_report(report: dict):
    """Print human-readable summary."""
    print("\n" + "=" * 60)
    print(f"LINEUP CAPTURE DAILY REPORT")
    print(f"Generated: {report['generated_at']}")
    print(f"Period: Last {report['period_hours']} hours")
    print("=" * 60)

    summary = report["summary"]
    print(f"\nTotal captures: {summary['total_captures']}")
    print(f"Unique leagues: {summary['unique_leagues']}")
    print(f"Live odds: {summary['live_pct']}%")
    print(f"Ideal window (45-90min): {summary['ideal_window_pct']}%")
    print(f"Historical total: {summary['total_historical']}")

    # Timing percentiles
    if report.get("timing_percentiles"):
        tp = report["timing_percentiles"]
        print(f"\n--- Timing Percentiles (minutes before kickoff) ---")
        print(f"  p50 (median): {tp['p50']} min")
        print(f"  p90: {tp['p90']} min")
        print(f"  Range: {tp['min']} - {tp['max']} min")
        print(f"  Mean: {tp['mean']} min")

    # Bin percentages
    if report.get("timing_bin_pct"):
        print(f"\n--- Timing Distribution (%) ---")
        for bucket, pct in report["timing_bin_pct"].items():
            bar = "#" * int(pct / 5)
            print(f"  {bucket:12s}: {pct:5.1f}% {bar}")

    print("\n--- By League (with timing) ---")
    for league, data in sorted(report["by_league"].items(), key=lambda x: -x[1]["count"]):
        timing = data.get("timing_stats", {})
        timing_str = f"p50={timing.get('p50', '?')}min" if timing else "no data"
        print(f"  {league}: {data['count']} captures ({timing_str})")
        for match in data["matches"][:3]:
            print(f"    - {match}")

    print("\n--- Timing Histogram ---")
    for bucket, count in report["timing_histogram"].items():
        bar = "#" * count
        print(f"  {bucket:12s}: {count:3d} {bar}")

    print("\n--- Odds Freshness ---")
    for freshness, count in report["odds_freshness"].items():
        print(f"  {freshness:10s}: {count}")

    print("\n--- By Bookmaker ---")
    for bookie, count in sorted(report["by_bookmaker"].items(), key=lambda x: -x[1]):
        print(f"  {bookie}: {count}")

    if report["captures"]:
        print("\n--- Recent Captures ---")
        for cap in report["captures"][:5]:
            print(f"  {cap['match']} ({cap['league']})")
            print(f"    Captured {cap['minutes_to_kickoff']:.0f}min before kickoff")
            print(f"    Odds: H={cap['odds']['home']} D={cap['odds']['draw']} A={cap['odds']['away']}")
            print(f"    Freshness: {cap['odds_freshness']}, Bookie: {cap['bookmaker']}")


async def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate lineup capture daily report")
    parser.add_argument("--hours", type=int, default=24, help="Hours to look back (default: 24)")
    parser.add_argument("--quiet", action="store_true", help="Only save JSON, no console output")
    args = parser.parse_args()

    # Ensure logs directory exists
    Path("logs").mkdir(exist_ok=True)

    report = await generate_daily_report(hours=args.hours)

    # Save to JSON
    date_str = datetime.utcnow().strftime("%Y%m%d")
    output_path = f"logs/lineup_capture_daily_{date_str}.json"
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    if not args.quiet:
        print_report(report)
        print(f"\nSaved to: {output_path}")
    else:
        print(f"Report saved to: {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
