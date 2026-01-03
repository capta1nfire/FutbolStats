#!/usr/bin/env python3
"""Export data from local SQLite to production PostgreSQL."""

import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import sqlite3
import asyncpg


def parse_date(date_str):
    """Parse date string from SQLite to datetime object."""
    if date_str is None:
        return None
    if isinstance(date_str, datetime):
        return date_str

    # Try common formats
    formats = [
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d",
    ]

    for fmt in formats:
        try:
            return datetime.strptime(str(date_str), fmt)
        except ValueError:
            continue

    # Last resort - try fromisoformat
    try:
        return datetime.fromisoformat(str(date_str).replace("Z", "+00:00"))
    except:
        return datetime.now()


async def export_data():
    """Export teams and matches from SQLite to PostgreSQL."""

    # Get production database URL from environment or Railway
    prod_db_url = os.environ.get("PROD_DATABASE_URL")
    if not prod_db_url:
        print("Error: Set PROD_DATABASE_URL environment variable")
        print("Example: postgresql://user:pass@host:port/database")
        sys.exit(1)

    # Connect to local SQLite
    sqlite_path = Path(__file__).parent.parent / "futbolstat.db"
    if not sqlite_path.exists():
        print(f"Error: SQLite database not found at {sqlite_path}")
        sys.exit(1)

    sqlite_conn = sqlite3.connect(str(sqlite_path))
    sqlite_conn.row_factory = sqlite3.Row

    # Connect to production PostgreSQL
    # Parse the URL for asyncpg
    pg_url = prod_db_url.replace("postgresql://", "").replace("postgres://", "")

    print(f"Connecting to production database...")
    pg_conn = await asyncpg.connect(prod_db_url)

    try:
        # Export teams
        print("\nExporting teams...")
        teams = sqlite_conn.execute("SELECT * FROM teams").fetchall()
        print(f"Found {len(teams)} teams in local database")

        teams_inserted = 0
        for team in teams:
            try:
                await pg_conn.execute("""
                    INSERT INTO teams (external_id, name, country, team_type, logo_url)
                    VALUES ($1, $2, $3, $4, $5)
                    ON CONFLICT (external_id) DO UPDATE SET
                        name = EXCLUDED.name,
                        country = EXCLUDED.country,
                        team_type = EXCLUDED.team_type,
                        logo_url = EXCLUDED.logo_url
                """, team['external_id'], team['name'], team['country'],
                    team['team_type'], team['logo_url'])
                teams_inserted += 1
            except Exception as e:
                print(f"  Error inserting team {team['name']}: {e}")

        print(f"Inserted/updated {teams_inserted} teams")

        # Create a mapping of external_id to PostgreSQL id for teams
        pg_teams = await pg_conn.fetch("SELECT id, external_id FROM teams")
        team_id_map = {row['external_id']: row['id'] for row in pg_teams}

        # Export matches
        print("\nExporting matches...")
        matches = sqlite_conn.execute("""
            SELECT m.*,
                   home.external_id as home_external_id,
                   away.external_id as away_external_id
            FROM matches m
            JOIN teams home ON m.home_team_id = home.id
            JOIN teams away ON m.away_team_id = away.id
        """).fetchall()
        print(f"Found {len(matches)} matches in local database")

        matches_inserted = 0
        for match in matches:
            try:
                # Get PostgreSQL team IDs
                home_id = team_id_map.get(match['home_external_id'])
                away_id = team_id_map.get(match['away_external_id'])

                if not home_id or not away_id:
                    print(f"  Skipping match {match['external_id']}: team not found")
                    continue

                # Parse stats JSON if present
                stats = None
                if match['stats']:
                    try:
                        stats = json.loads(match['stats']) if isinstance(match['stats'], str) else match['stats']
                    except:
                        pass

                # Parse date
                match_date = parse_date(match['date'])

                await pg_conn.execute("""
                    INSERT INTO matches (
                        external_id, date, league_id, season,
                        home_team_id, away_team_id,
                        home_goals, away_goals, stats,
                        status, match_type, match_weight,
                        odds_home, odds_draw, odds_away
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15)
                    ON CONFLICT (external_id) DO UPDATE SET
                        home_goals = EXCLUDED.home_goals,
                        away_goals = EXCLUDED.away_goals,
                        stats = EXCLUDED.stats,
                        status = EXCLUDED.status
                """,
                    match['external_id'], match_date, match['league_id'], match['season'],
                    home_id, away_id,
                    match['home_goals'], match['away_goals'],
                    json.dumps(stats) if stats else None,
                    match['status'], match['match_type'], match['match_weight'],
                    match['odds_home'], match['odds_draw'], match['odds_away']
                )
                matches_inserted += 1

                if matches_inserted % 500 == 0:
                    print(f"  Processed {matches_inserted} matches...")

            except Exception as e:
                print(f"  Error inserting match {match['external_id']}: {e}")

        print(f"Inserted/updated {matches_inserted} matches")

        # Verify counts
        pg_team_count = await pg_conn.fetchval("SELECT COUNT(*) FROM teams")
        pg_match_count = await pg_conn.fetchval("SELECT COUNT(*) FROM matches")

        print(f"\n✅ Export complete!")
        print(f"   Production database now has:")
        print(f"   - {pg_team_count} teams")
        print(f"   - {pg_match_count} matches")

    finally:
        await pg_conn.close()
        sqlite_conn.close()


if __name__ == "__main__":
    asyncio.run(export_data())
