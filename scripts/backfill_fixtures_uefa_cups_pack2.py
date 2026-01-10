#!/usr/bin/env python3
"""
Worker Fixtures/Teams: EU Mid Pack2 - UEFA Cups Backfill

Importa desde API-Football SOLO:
- Fixtures (matches) con resultados FT
- Teams (si faltan)

NO escribe (anti-contaminacion):
- opening_odds_* columns
- odds_recorded_at column
- odds_history table
- odds_snapshots table
- market_movement_snapshots table
- lineup_movement_snapshots table

Idempotente: usa external_id para upsert sin duplicar.

Scope EU Mid Pack2 - UEFA Cups:
- 2: UEFA Champions League
- 3: UEFA Europa League
- 848: UEFA Europa Conference League
- Seasons: 2019-2024
"""

import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Optional

from dotenv import load_dotenv
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
import httpx

load_dotenv()

# Timestamp for logs
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"logs/eu_mid_pack2_uefa_cups_backfill_{TIMESTAMP}.log")
    ]
)
logger = logging.getLogger(__name__)

# EU Mid Pack2 - UEFA Cups Configuration
BACKFILL_CONFIG = {
    2: {"name": "UEFA Champions League", "country": "World", "tier": 1},
    3: {"name": "UEFA Europa League", "country": "World", "tier": 1},
    848: {"name": "UEFA Europa Conference League", "country": "World", "tier": 1},
}

# Authorized seasons
SEASONS = [2019, 2020, 2021, 2022, 2023, 2024]


@dataclass
class CoverageStats:
    league_id: int
    league_name: str
    country: str
    tier: int
    seasons_processed: list
    min_date: Optional[str]
    max_date: Optional[str]
    matches_before: int
    matches_after: int
    matches_new: int
    ft_matches: int
    ft_percentage: float
    teams_new: int
    errors: int


@dataclass
class MatchData:
    external_id: int
    date: datetime
    league_id: int
    season: int
    home_team_external_id: int
    away_team_external_id: int
    home_goals: Optional[int]
    away_goals: Optional[int]
    status: str
    match_type: str = "official"
    match_weight: float = 1.0


@dataclass
class TeamData:
    external_id: int
    name: str
    country: Optional[str]
    team_type: str
    logo_url: Optional[str]


class APIFootballClient:
    """Minimal API-Football client for backfill."""

    def __init__(self):
        host = os.environ.get("RAPIDAPI_HOST", "v3.football.api-sports.io")
        key = os.environ.get("RAPIDAPI_KEY")

        if not key:
            raise ValueError("RAPIDAPI_KEY required")

        if "api-sports.io" in host:
            self.base_url = f"https://{host}"
            headers = {"x-apisports-key": key}
        else:
            self.base_url = f"https://{host}/v3"
            headers = {
                "X-RapidAPI-Key": key,
                "X-RapidAPI-Host": host,
            }

        self.client = httpx.AsyncClient(headers=headers, timeout=30.0)
        self.requests_per_minute = int(os.environ.get("API_REQUESTS_PER_MINUTE", 250))
        self._delay = 60 / self.requests_per_minute

    async def _request(self, endpoint: str, params: dict) -> dict:
        url = f"{self.base_url}/{endpoint}"

        for attempt in range(3):
            try:
                response = await self.client.get(url, params=params)

                if response.status_code == 429:
                    wait_time = 5 * (2 ** attempt)
                    logger.warning(f"Rate limited. Waiting {wait_time}s...")
                    await asyncio.sleep(wait_time)
                    continue

                response.raise_for_status()
                await asyncio.sleep(self._delay)

                data = response.json()
                if data.get("errors"):
                    logger.error(f"API error: {data['errors']}")
                    return {"response": []}

                return data

            except httpx.HTTPStatusError as e:
                logger.error(f"HTTP error: {e}")
                if attempt < 2:
                    await asyncio.sleep(5 * (2 ** attempt))
                    continue
                raise

        return {"response": []}

    async def get_fixtures(self, league_id: int, season: int) -> list[MatchData]:
        logger.info(f"Fetching fixtures for league {league_id}, season {season}")

        data = await self._request("fixtures", {
            "league": league_id,
            "season": season,
        })

        fixtures = data.get("response", [])
        matches = []

        for fixture in fixtures:
            try:
                match = self._parse_fixture(fixture, league_id)
                if match:
                    matches.append(match)
            except Exception as e:
                logger.debug(f"Error parsing fixture: {e}")
                continue

        logger.info(f"Fetched {len(matches)} fixtures")
        return matches

    def _parse_fixture(self, fixture: dict, league_id: int) -> Optional[MatchData]:
        fixture_info = fixture.get("fixture", {})
        teams = fixture.get("teams", {})
        goals = fixture.get("goals", {})

        home_id = teams.get("home", {}).get("id")
        away_id = teams.get("away", {}).get("id")

        if not home_id or not away_id:
            return None

        date_str = fixture_info.get("date", "")
        try:
            match_date = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
            if match_date.tzinfo is not None:
                match_date = match_date.replace(tzinfo=None)
        except ValueError:
            match_date = datetime.utcnow()

        return MatchData(
            external_id=fixture_info.get("id"),
            date=match_date,
            league_id=league_id,
            season=fixture.get("league", {}).get("season", datetime.now().year),
            home_team_external_id=home_id,
            away_team_external_id=away_id,
            home_goals=goals.get("home"),
            away_goals=goals.get("away"),
            status=fixture_info.get("status", {}).get("short", "NS"),
        )

    async def get_team(self, team_id: int) -> Optional[TeamData]:
        data = await self._request("teams", {"id": team_id})
        teams = data.get("response", [])

        if not teams:
            return None

        team_info = teams[0].get("team", {})
        country = team_info.get("country")
        name = team_info.get("name", "")
        is_national = team_info.get("national", False) or name == country

        return TeamData(
            external_id=team_info.get("id"),
            name=name,
            country=country if not is_national else None,
            team_type="national" if is_national else "club",
            logo_url=team_info.get("logo"),
        )

    async def close(self):
        await self.client.aclose()


class FixturesBackfill:
    """Backfill fixtures and teams WITHOUT odds."""

    def __init__(self, session: AsyncSession, api: APIFootballClient):
        self.session = session
        self.api = api
        self._teams_cache: dict[int, int] = {}
        self._new_teams_count = 0
        self._errors = 0

    async def get_coverage_stats(self, league_id: int) -> dict:
        result = await self.session.execute(text("""
            SELECT
                COUNT(*) as total_matches,
                COUNT(*) FILTER (WHERE status = 'FT') as ft_matches,
                MIN(date) as min_date,
                MAX(date) as max_date,
                array_agg(DISTINCT season) as seasons
            FROM matches
            WHERE league_id = :league_id
        """), {"league_id": league_id})

        row = result.fetchone()

        return {
            "total_matches": row.total_matches or 0,
            "ft_matches": row.ft_matches or 0,
            "min_date": row.min_date.isoformat() if row.min_date else None,
            "max_date": row.max_date.isoformat() if row.max_date else None,
            "seasons": sorted([s for s in (row.seasons or []) if s]) if row.seasons else [],
        }

    async def _get_or_create_team(self, external_id: int) -> int:
        if external_id in self._teams_cache:
            return self._teams_cache[external_id]

        result = await self.session.execute(text(
            "SELECT id, name FROM teams WHERE external_id = :ext_id"
        ), {"ext_id": external_id})
        row = result.fetchone()

        if row:
            self._teams_cache[external_id] = row.id
            return row.id

        team_data = await self.api.get_team(external_id)

        if not team_data:
            logger.warning(f"Could not fetch team {external_id}")
            team_data = TeamData(
                external_id=external_id,
                name=f"Unknown Team {external_id}",
                country=None,
                team_type="club",
                logo_url=None,
            )

        # Upsert team - NO odds columns
        result = await self.session.execute(text("""
            INSERT INTO teams (external_id, name, country, team_type, logo_url)
            VALUES (:external_id, :name, :country, :team_type, :logo_url)
            ON CONFLICT (external_id) DO UPDATE SET
                name = EXCLUDED.name,
                country = EXCLUDED.country,
                logo_url = EXCLUDED.logo_url
            RETURNING id
        """), {
            "external_id": team_data.external_id,
            "name": team_data.name,
            "country": team_data.country,
            "team_type": team_data.team_type,
            "logo_url": team_data.logo_url,
        })

        team_id = result.scalar_one()
        self._teams_cache[external_id] = team_id
        self._new_teams_count += 1

        logger.info(f"Upserted team: {team_data.name} (ID: {team_id})")
        return team_id

    async def _upsert_match(self, match: MatchData) -> bool:
        """Upsert match WITHOUT odds columns. Returns True if new."""
        home_team_id = await self._get_or_create_team(match.home_team_external_id)
        away_team_id = await self._get_or_create_team(match.away_team_external_id)

        result = await self.session.execute(text(
            "SELECT id FROM matches WHERE external_id = :ext_id"
        ), {"ext_id": match.external_id})
        existing = result.scalar_one_or_none()

        if existing:
            # Update only core fields - NO odds
            await self.session.execute(text("""
                UPDATE matches SET
                    date = :date,
                    home_goals = :home_goals,
                    away_goals = :away_goals,
                    status = :status,
                    home_team_id = :home_team_id,
                    away_team_id = :away_team_id
                WHERE external_id = :external_id
            """), {
                "external_id": match.external_id,
                "date": match.date,
                "home_goals": match.home_goals,
                "away_goals": match.away_goals,
                "status": match.status,
                "home_team_id": home_team_id,
                "away_team_id": away_team_id,
            })
            return False

        # Insert new match - NO odds columns
        await self.session.execute(text("""
            INSERT INTO matches (
                external_id, date, league_id, season,
                home_team_id, away_team_id,
                home_goals, away_goals,
                status, match_type, match_weight
            ) VALUES (
                :external_id, :date, :league_id, :season,
                :home_team_id, :away_team_id,
                :home_goals, :away_goals,
                :status, :match_type, :match_weight
            )
        """), {
            "external_id": match.external_id,
            "date": match.date,
            "league_id": match.league_id,
            "season": match.season,
            "home_team_id": home_team_id,
            "away_team_id": away_team_id,
            "home_goals": match.home_goals,
            "away_goals": match.away_goals,
            "status": match.status,
            "match_type": match.match_type,
            "match_weight": match.match_weight,
        })
        return True

    async def backfill_league(self, league_id: int, seasons: list[int]) -> CoverageStats:
        config = BACKFILL_CONFIG.get(league_id, {})
        league_name = config.get("name", f"League {league_id}")
        country = config.get("country", "Unknown")
        tier = config.get("tier", 0)

        logger.info(f"\n{'='*60}")
        logger.info(f"BACKFILLING: {league_name} (ID: {league_id}, Tier {tier})")
        logger.info(f"Country: {country}, Seasons: {seasons}")
        logger.info(f"{'='*60}")

        before = await self.get_coverage_stats(league_id)
        logger.info(f"BEFORE: {before['total_matches']} matches, {before['ft_matches']} FT")

        self._new_teams_count = 0
        self._errors = 0
        matches_new = 0
        matches_processed = 0

        for season in seasons:
            logger.info(f"\n--- Season {season} ---")

            try:
                fixtures = await self.api.get_fixtures(league_id, season)
            except Exception as e:
                logger.error(f"Failed to fetch season {season}: {e}")
                self._errors += 1
                continue

            for match in fixtures:
                try:
                    is_new = await self._upsert_match(match)
                    if is_new:
                        matches_new += 1
                    matches_processed += 1

                    if matches_processed % 100 == 0:
                        await self.session.commit()
                        logger.info(f"Progress: {matches_processed} matches processed")

                except Exception as e:
                    logger.error(f"Error upserting match {match.external_id}: {e}")
                    self._errors += 1
                    await self.session.rollback()
                    continue

            await self.session.commit()
            logger.info(f"Season {season}: {len(fixtures)} fixtures processed")

        after = await self.get_coverage_stats(league_id)
        logger.info(f"\nAFTER: {after['total_matches']} matches, {after['ft_matches']} FT")

        ft_pct = (after['ft_matches'] / after['total_matches'] * 100) if after['total_matches'] > 0 else 0

        return CoverageStats(
            league_id=league_id,
            league_name=league_name,
            country=country,
            tier=tier,
            seasons_processed=seasons,
            min_date=after['min_date'],
            max_date=after['max_date'],
            matches_before=before['total_matches'],
            matches_after=after['total_matches'],
            matches_new=matches_new,
            ft_matches=after['ft_matches'],
            ft_percentage=round(ft_pct, 2),
            teams_new=self._new_teams_count,
            errors=self._errors,
        )


async def main():
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        raise ValueError("DATABASE_URL required")

    if database_url.startswith("postgresql://"):
        database_url = database_url.replace("postgresql://", "postgresql+asyncpg://", 1)
    elif database_url.startswith("postgres://"):
        database_url = database_url.replace("postgres://", "postgresql+asyncpg://", 1)

    engine = create_async_engine(database_url, pool_pre_ping=True)
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    api = APIFootballClient()

    logs_dir = os.path.join(os.path.dirname(__file__), "..", "logs")
    os.makedirs(logs_dir, exist_ok=True)

    all_stats = []
    start_time = datetime.now()

    try:
        async with async_session() as session:
            backfill = FixturesBackfill(session, api)

            # Process by league ID order
            sorted_leagues = sorted(BACKFILL_CONFIG.items(), key=lambda x: x[0])

            for league_id, config in sorted_leagues:
                try:
                    stats = await backfill.backfill_league(
                        league_id=league_id,
                        seasons=SEASONS
                    )
                    all_stats.append(asdict(stats))

                    # Save individual report with timestamp
                    report_path = os.path.join(logs_dir, f"coverage_before_after_{league_id}_{TIMESTAMP}.json")
                    with open(report_path, "w") as f:
                        json.dump(asdict(stats), f, indent=2, default=str)
                    logger.info(f"Saved: {report_path}")

                except Exception as e:
                    logger.error(f"Error backfilling league {league_id}: {e}")
                    continue

    finally:
        await api.close()
        await engine.dispose()

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    # Save combined report with timestamp
    combined_report = {
        "timestamp": TIMESTAMP,
        "duration_seconds": duration,
        "seasons": SEASONS,
        "leagues": all_stats,
        "totals": {
            "total_leagues": len(all_stats),
            "total_matches_new": sum(s["matches_new"] for s in all_stats),
            "total_teams_new": sum(s["teams_new"] for s in all_stats),
            "total_errors": sum(s["errors"] for s in all_stats),
        }
    }

    report_path = os.path.join(logs_dir, f"eu_mid_pack2_uefa_cups_coverage_report_{TIMESTAMP}.json")
    with open(report_path, "w") as f:
        json.dump(combined_report, f, indent=2, default=str)

    # Print summary
    print("\n" + "=" * 70)
    print("EU MID PACK2 - UEFA CUPS BACKFILL SUMMARY")
    print("=" * 70)
    print(f"Duration: {duration:.1f} seconds")
    print(f"Seasons: {SEASONS}")
    print()

    print(f"{'ID':<6} {'Liga':<40} {'New':<8} {'Total':<8} {'FT%':<8} {'Teams':<6} {'Err'}")
    print("-" * 90)

    for stats in all_stats:
        print(f"{stats['league_id']:<6} {stats['league_name']:<40} "
              f"{stats['matches_new']:<8} {stats['matches_after']:<8} "
              f"{stats['ft_percentage']:<8} {stats['teams_new']:<6} {stats['errors']}")

    print("-" * 90)
    totals = combined_report["totals"]
    print(f"{'TOTAL':<6} {'':<40} {totals['total_matches_new']:<8} {'':<8} "
          f"{'':<8} {totals['total_teams_new']:<6} {totals['total_errors']}")

    print("\n" + "=" * 70)
    print("Reports saved to logs/")
    print("CONFIRMACION: NO se escribieron odds - solo fixtures/teams")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
