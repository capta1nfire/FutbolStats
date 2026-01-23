"""Migration 029: SOTA Core Tables (Fase 1 - Datos/Features).

Creates core tables for SOTA feature engineering:
- match_external_refs: Multi-source match linking (Understat, Sofascore)
- match_understat_team: xG/xPTS data per match
- venue_geo: Venue geolocation for weather features
- team_home_city_profile: Team climate baseline for bio-adaptability
- match_weather: Weather forecast data per match

Reference: docs/ARCHITECTURE_SOTA.md sections 1.2, 1.3
Column names are EXACT as defined in architecture doc.
"""

import asyncio
import logging

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# Table 1: match_external_refs
# Links matches.id to external sources (Understat, Sofascore) with confidence
# =============================================================================
SQL_MATCH_EXTERNAL_REFS = """
CREATE TABLE IF NOT EXISTS match_external_refs (
    match_id INTEGER NOT NULL REFERENCES matches(id) ON DELETE CASCADE,
    source VARCHAR(50) NOT NULL,  -- 'api_football' | 'understat' | 'sofascore'
    source_match_id VARCHAR(100) NOT NULL,  -- External ID (string for flexibility)
    confidence FLOAT NOT NULL,  -- [0,1] matching confidence
    matched_by VARCHAR(100) NOT NULL,  -- Heuristic used (e.g., 'kickoff+teams')
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    PRIMARY KEY (match_id, source)
);

CREATE INDEX IF NOT EXISTS ix_match_external_refs_source_id
ON match_external_refs (source, source_match_id);

COMMENT ON TABLE match_external_refs IS 'Multi-source match linking for SOTA enrichment (ARCHITECTURE_SOTA.md 1.2)';
"""

# =============================================================================
# Table 2: match_understat_team
# xG/xPTS data from Understat per match
# =============================================================================
SQL_MATCH_UNDERSTAT_TEAM = """
CREATE TABLE IF NOT EXISTS match_understat_team (
    match_id INTEGER NOT NULL REFERENCES matches(id) ON DELETE CASCADE PRIMARY KEY,
    xg_home FLOAT NOT NULL,
    xg_away FLOAT NOT NULL,
    xpts_home FLOAT,  -- nullable: not always available
    xpts_away FLOAT,
    npxg_home FLOAT,  -- nullable: non-penalty xG
    npxg_away FLOAT,
    xga_home FLOAT,  -- nullable: xG against (if available)
    xga_away FLOAT,
    captured_at TIMESTAMP NOT NULL DEFAULT NOW(),
    source_version VARCHAR(50)  -- e.g., 'understat_2024'
);

CREATE INDEX IF NOT EXISTS ix_match_understat_team_captured_at
ON match_understat_team (captured_at);

COMMENT ON TABLE match_understat_team IS 'Understat xG/xPTS per match (ARCHITECTURE_SOTA.md 1.3)';
"""

# =============================================================================
# Table 3: venue_geo
# Geolocation for venues (weather lookup)
# =============================================================================
SQL_VENUE_GEO = """
CREATE TABLE IF NOT EXISTS venue_geo (
    venue_city VARCHAR(255) NOT NULL,
    country VARCHAR(100) NOT NULL,
    lat FLOAT NOT NULL,
    lon FLOAT NOT NULL,
    source VARCHAR(50) NOT NULL,  -- e.g., 'manual', 'geocoding_api'
    confidence FLOAT NOT NULL,  -- [0,1]
    PRIMARY KEY (venue_city, country)
);

COMMENT ON TABLE venue_geo IS 'Venue geolocation for weather features (ARCHITECTURE_SOTA.md 1.3)';
"""

# =============================================================================
# Table 4: team_home_city_profile
# Team climate baseline for bio-adaptability features
# =============================================================================
SQL_TEAM_HOME_CITY_PROFILE = """
CREATE TABLE IF NOT EXISTS team_home_city_profile (
    team_id INTEGER NOT NULL REFERENCES teams(id) ON DELETE CASCADE PRIMARY KEY,
    home_city VARCHAR(255) NOT NULL,
    timezone VARCHAR(50) NOT NULL,  -- e.g., 'Europe/Madrid'
    climate_normals_by_month JSONB  -- {"01": {"temp": 10, "humidity": 70}, ...}
);

COMMENT ON TABLE team_home_city_profile IS 'Team climate baseline for thermal_shock/bio features (ARCHITECTURE_SOTA.md 1.3)';
"""

# =============================================================================
# Table 5: match_weather
# Weather forecast data per match
# =============================================================================
SQL_MATCH_WEATHER = """
CREATE TABLE IF NOT EXISTS match_weather (
    match_id INTEGER NOT NULL REFERENCES matches(id) ON DELETE CASCADE,
    temp_c FLOAT NOT NULL,
    humidity FLOAT NOT NULL,  -- percentage
    wind_ms FLOAT NOT NULL,  -- m/s
    precip_mm FLOAT NOT NULL,  -- mm
    pressure_hpa FLOAT,  -- nullable
    cloudcover FLOAT,  -- nullable, percentage
    is_daylight BOOLEAN NOT NULL,
    forecast_horizon_hours INTEGER NOT NULL,  -- e.g., 24, 1
    captured_at TIMESTAMP NOT NULL DEFAULT NOW(),
    PRIMARY KEY (match_id, forecast_horizon_hours)
);

CREATE INDEX IF NOT EXISTS ix_match_weather_captured_at
ON match_weather (captured_at);

COMMENT ON TABLE match_weather IS 'Weather forecast per match (ARCHITECTURE_SOTA.md 1.3)';
"""


# =============================================================================
# All tables to create
# =============================================================================
TABLES = [
    ("match_external_refs", SQL_MATCH_EXTERNAL_REFS),
    ("match_understat_team", SQL_MATCH_UNDERSTAT_TEAM),
    ("venue_geo", SQL_VENUE_GEO),
    ("team_home_city_profile", SQL_TEAM_HOME_CITY_PROFILE),
    ("match_weather", SQL_MATCH_WEATHER),
]


async def table_exists(conn, table_name: str) -> bool:
    """Check if a table exists in the database."""
    result = await conn.execute(text("""
        SELECT EXISTS (
            SELECT FROM information_schema.tables
            WHERE table_schema = 'public' AND table_name = :table_name
        )
    """), {"table_name": table_name})
    return result.scalar()


async def main():
    """Run migration from command line."""
    import os

    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        raise ValueError("DATABASE_URL environment variable not set")

    # Convert to async URL if needed
    if database_url.startswith("postgresql://"):
        database_url = database_url.replace("postgresql://", "postgresql+asyncpg://", 1)

    engine = create_async_engine(database_url, echo=False)

    results = {"created": [], "already_exists": [], "errors": []}

    async with engine.begin() as conn:
        for table_name, sql in TABLES:
            try:
                exists = await table_exists(conn, table_name)
                if exists:
                    logger.info(f"Table {table_name} already exists, skipping...")
                    results["already_exists"].append(table_name)
                else:
                    logger.info(f"Creating table {table_name}...")
                    # Execute each statement separately
                    for statement in sql.strip().split(";"):
                        statement = statement.strip()
                        if statement and not statement.startswith("--"):
                            await conn.execute(text(statement))
                    results["created"].append(table_name)
                    logger.info(f"Table {table_name} created.")
            except Exception as e:
                logger.error(f"Error creating table {table_name}: {e}")
                results["errors"].append((table_name, str(e)))

        # Verification: count rows in each table
        logger.info("\n" + "=" * 60)
        logger.info("VERIFICATION - Tables status:")
        logger.info("=" * 60)
        for table_name, _ in TABLES:
            try:
                exists = await table_exists(conn, table_name)
                if exists:
                    result = await conn.execute(text(f"SELECT COUNT(*) FROM {table_name}"))
                    count = result.scalar() or 0
                    logger.info(f"  {table_name}: EXISTS ({count} rows)")
                else:
                    logger.info(f"  {table_name}: NOT EXISTS")
            except Exception as e:
                logger.info(f"  {table_name}: ERROR ({e})")

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("MIGRATION SUMMARY:")
    logger.info("=" * 60)
    logger.info(f"  Created: {results['created']}")
    logger.info(f"  Already existed: {results['already_exists']}")
    if results["errors"]:
        logger.error(f"  Errors: {results['errors']}")
        raise RuntimeError(f"Migration failed with errors: {results['errors']}")

    logger.info("\nMigration 029 complete.")


if __name__ == "__main__":
    asyncio.run(main())
