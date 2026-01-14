#!/usr/bin/env python3
"""
Sync Railway PostgreSQL to Local SQLite (Read-Only Export)

Exports selected tables from Railway to local SQLite for offline training.
Only reads from Railway, writes to local SQLite.

Usage:
    python3 scripts/sync_railway_to_local_sqlite.py \
        --output /Users/inseqio/FutbolStats/futbolstat_railway.db \
        --tables matches,teams
"""

import argparse
import asyncio
import json
import os
import sqlite3
from datetime import datetime

import asyncpg

# Columns to export from matches (subset for training)
MATCHES_COLUMNS = [
    "id",
    "external_id",
    "date",
    "league_id",
    "season",
    "home_team_id",
    "away_team_id",
    "home_goals",
    "away_goals",
    "status",
    "match_type",
    "match_weight",
    "stats",
    "home_formation",
    "away_formation",
    "xg_home",
    "xg_away",
    "odds_home",
    "odds_draw",
    "odds_away",
    "opening_odds_home",
    "opening_odds_draw",
    "opening_odds_away",
]

TEAMS_COLUMNS = [
    "id",
    "external_id",
    "name",
    "country",
    "team_type",
]


async def fetch_matches(conn, status_filter: str = None, limit: int = None) -> list:
    """Fetch matches from Railway PostgreSQL."""
    columns_str = ", ".join(MATCHES_COLUMNS)
    query = f"SELECT {columns_str} FROM matches"

    conditions = []
    if status_filter:
        conditions.append(f"status = '{status_filter}'")

    if conditions:
        query += " WHERE " + " AND ".join(conditions)

    query += " ORDER BY date ASC"

    if limit:
        query += f" LIMIT {limit}"

    rows = await conn.fetch(query)
    return [dict(row) for row in rows]


async def fetch_teams(conn, limit: int = None) -> list:
    """Fetch teams from Railway PostgreSQL."""
    columns_str = ", ".join(TEAMS_COLUMNS)
    query = f"SELECT {columns_str} FROM teams ORDER BY id"

    if limit:
        query += f" LIMIT {limit}"

    rows = await conn.fetch(query)
    return [dict(row) for row in rows]


def create_sqlite_schema(cursor):
    """Create SQLite tables matching the export schema."""

    # Matches table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS matches (
            id INTEGER PRIMARY KEY,
            external_id INTEGER,
            date TEXT,
            league_id INTEGER,
            season INTEGER,
            home_team_id INTEGER,
            away_team_id INTEGER,
            home_goals INTEGER,
            away_goals INTEGER,
            status TEXT,
            match_type TEXT,
            match_weight REAL,
            stats TEXT,
            home_formation TEXT,
            away_formation TEXT,
            xg_home REAL,
            xg_away REAL,
            odds_home REAL,
            odds_draw REAL,
            odds_away REAL,
            opening_odds_home REAL,
            opening_odds_draw REAL,
            opening_odds_away REAL
        )
    """)

    # Teams table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS teams (
            id INTEGER PRIMARY KEY,
            external_id INTEGER,
            name TEXT,
            country TEXT,
            team_type TEXT
        )
    """)

    # Create indexes
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_matches_date ON matches(date)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_matches_league ON matches(league_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_matches_status ON matches(status)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_matches_home_team ON matches(home_team_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_matches_away_team ON matches(away_team_id)")


def insert_matches(cursor, matches: list):
    """Insert matches into SQLite."""
    columns = MATCHES_COLUMNS
    placeholders = ", ".join(["?" for _ in columns])
    columns_str = ", ".join(columns)

    cursor.execute("DELETE FROM matches")  # Clear existing

    for match in matches:
        values = []
        for col in columns:
            val = match.get(col)
            # Convert datetime to string
            if isinstance(val, datetime):
                val = val.isoformat()
            # Convert dict/list to JSON string
            elif isinstance(val, (dict, list)):
                val = json.dumps(val)
            values.append(val)

        cursor.execute(f"INSERT INTO matches ({columns_str}) VALUES ({placeholders})", values)


def insert_teams(cursor, teams: list):
    """Insert teams into SQLite."""
    columns = TEAMS_COLUMNS
    placeholders = ", ".join(["?" for _ in columns])
    columns_str = ", ".join(columns)

    cursor.execute("DELETE FROM teams")  # Clear existing

    for team in teams:
        values = [team.get(col) for col in columns]
        cursor.execute(f"INSERT INTO teams ({columns_str}) VALUES ({placeholders})", values)


async def main():
    parser = argparse.ArgumentParser(description="Sync Railway PostgreSQL to Local SQLite")
    parser.add_argument("--output", default="/Users/inseqio/FutbolStats/futbolstat_railway.db",
                        help="Output SQLite database path")
    parser.add_argument("--tables", default="matches,teams", help="Tables to sync (comma-separated)")
    parser.add_argument("--status", default=None, help="Filter matches by status (e.g., FT)")
    parser.add_argument("--limit", type=int, default=None, help="Limit rows per table")
    args = parser.parse_args()

    # Railway connection string from environment variable (required)
    pg_url = os.environ.get("RAILWAY_DATABASE_URL") or os.environ.get("DATABASE_URL")
    if not pg_url:
        print("ERROR: RAILWAY_DATABASE_URL or DATABASE_URL environment variable required.")
        print("Usage: export DATABASE_URL='postgresql://user:pass@host:port/db'")
        sys.exit(1)

    tables = [t.strip() for t in args.tables.split(",")]

    print(f"Connecting to Railway PostgreSQL...")
    conn = await asyncpg.connect(pg_url)

    print(f"Output SQLite: {args.output}")
    sqlite_conn = sqlite3.connect(args.output)
    cursor = sqlite_conn.cursor()

    # Create schema
    print("Creating SQLite schema...")
    create_sqlite_schema(cursor)

    # Sync tables
    if "matches" in tables:
        print(f"Fetching matches from Railway (status={args.status})...")
        matches = await fetch_matches(conn, status_filter=args.status, limit=args.limit)
        print(f"  Fetched {len(matches)} matches")

        print("Inserting matches into SQLite...")
        insert_matches(cursor, matches)
        print(f"  Inserted {len(matches)} matches")

    if "teams" in tables:
        print("Fetching teams from Railway...")
        teams = await fetch_teams(conn, limit=args.limit)
        print(f"  Fetched {len(teams)} teams")

        print("Inserting teams into SQLite...")
        insert_teams(cursor, teams)
        print(f"  Inserted {len(teams)} teams")

    # Commit and close
    sqlite_conn.commit()

    # Verify counts
    cursor.execute("SELECT COUNT(*) FROM matches")
    match_count = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM teams")
    team_count = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM matches WHERE status = 'FT'")
    ft_count = cursor.fetchone()[0]

    cursor.execute("SELECT MIN(date), MAX(date) FROM matches WHERE status = 'FT'")
    date_range = cursor.fetchone()

    sqlite_conn.close()
    await conn.close()

    # File size
    file_size = os.path.getsize(args.output) / (1024 * 1024)

    print("\n" + "=" * 60)
    print("SYNC COMPLETE")
    print("=" * 60)
    print(f"Output:       {args.output}")
    print(f"File size:    {file_size:.2f} MB")
    print(f"Matches:      {match_count:,}")
    print(f"  FT:         {ft_count:,}")
    print(f"Teams:        {team_count:,}")
    print(f"Date range:   {date_range[0]} to {date_range[1]}")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
