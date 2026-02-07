"""
Migration: Create tables for Players & Managers MVP (Phase 1).

Tables created:
  1. managers          — DT catalog (lightweight)
  2. team_manager_history — DT stints per team
  3. player_injuries    — Injuries per fixture

Reference: docs/PLAYERS_MANAGERS_PROPOSAL.md v2.1

Usage:
  DATABASE_URL=postgresql://... python scripts/migrate_players_managers.py
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine


# Each statement must be executed separately (asyncpg requirement)
STATEMENTS = [
    # --- Table 1: managers ---
    (
        "CREATE TABLE managers",
        """
        CREATE TABLE IF NOT EXISTS managers (
            id SERIAL PRIMARY KEY,
            external_id INTEGER NOT NULL UNIQUE,
            name VARCHAR(200) NOT NULL,
            first_name VARCHAR(100),
            last_name VARCHAR(100),
            birth_date DATE,
            nationality VARCHAR(100),
            photo_url TEXT,
            career JSONB,
            raw_json JSONB,
            created_at TIMESTAMP NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMP NOT NULL DEFAULT NOW()
        )
        """,
    ),
    (
        "CREATE INDEX ix_managers_ext_id",
        "CREATE INDEX IF NOT EXISTS ix_managers_ext_id ON managers (external_id)",
    ),
    # --- Table 2: team_manager_history ---
    (
        "CREATE TABLE team_manager_history",
        """
        CREATE TABLE IF NOT EXISTS team_manager_history (
            id SERIAL PRIMARY KEY,
            team_id INTEGER NOT NULL REFERENCES teams(id),
            manager_external_id INTEGER NOT NULL,
            manager_name VARCHAR(200) NOT NULL,
            start_date DATE NOT NULL,
            end_date DATE,
            team_external_id INTEGER,
            source VARCHAR(20) NOT NULL DEFAULT 'api-football',
            detected_at TIMESTAMP NOT NULL DEFAULT NOW(),
            UNIQUE (team_id, manager_external_id, start_date)
        )
        """,
    ),
    (
        "CREATE INDEX ix_tmh_team_current",
        "CREATE INDEX IF NOT EXISTS ix_tmh_team_current ON team_manager_history (team_id) WHERE end_date IS NULL",
    ),
    (
        "CREATE INDEX ix_tmh_manager",
        "CREATE INDEX IF NOT EXISTS ix_tmh_manager ON team_manager_history (manager_external_id)",
    ),
    # --- Table 3: player_injuries ---
    (
        "CREATE TABLE player_injuries",
        """
        CREATE TABLE IF NOT EXISTS player_injuries (
            id SERIAL PRIMARY KEY,
            player_external_id INTEGER NOT NULL,
            player_name VARCHAR(200) NOT NULL,
            team_id INTEGER REFERENCES teams(id),
            league_id INTEGER NOT NULL,
            season INTEGER NOT NULL,
            fixture_external_id INTEGER NOT NULL,
            match_id INTEGER REFERENCES matches(id),
            injury_type VARCHAR(50) NOT NULL,
            injury_reason VARCHAR(200),
            fixture_date TIMESTAMP,
            raw_json JSONB,
            captured_at TIMESTAMP NOT NULL DEFAULT NOW(),
            UNIQUE (player_external_id, fixture_external_id)
        )
        """,
    ),
    (
        "CREATE INDEX ix_injuries_match",
        "CREATE INDEX IF NOT EXISTS ix_injuries_match ON player_injuries (match_id) WHERE match_id IS NOT NULL",
    ),
    (
        "CREATE INDEX ix_injuries_team_date",
        "CREATE INDEX IF NOT EXISTS ix_injuries_team_date ON player_injuries (team_id, fixture_date DESC)",
    ),
    (
        "CREATE INDEX ix_injuries_league_season",
        "CREATE INDEX IF NOT EXISTS ix_injuries_league_season ON player_injuries (league_id, season)",
    ),
]


async def main():
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        print("ERROR: DATABASE_URL not set")
        sys.exit(1)

    if db_url.startswith("postgresql://"):
        db_url = db_url.replace("postgresql://", "postgresql+asyncpg://", 1)

    engine = create_async_engine(db_url, echo=False)

    async with engine.begin() as conn:
        for label, sql in STATEMENTS:
            await conn.execute(text(sql))
            print(f"  OK: {label}")

    await engine.dispose()
    print("\nMigration complete: 3 tables + 6 indexes created.")


if __name__ == "__main__":
    asyncio.run(main())
