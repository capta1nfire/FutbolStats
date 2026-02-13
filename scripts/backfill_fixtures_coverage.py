#!/usr/bin/env python3
"""
Worker Fixtures/Teams: Backfill de cobertura para subir match rate global.

Importa desde API-Football:
- Fixtures (matches) con resultados FT
- Teams (si faltan)

NO escribe:
- opening_odds_*
- odds_history
- odds_snapshots

Idempotente: usa external_id para upsert sin duplicar.

Entregables en logs/:
- coverage_before_after_{league_id}.json
- unmatched_teams_{league_id}.json (si hay)

Scope:
1. Ligue 1 (61): season 2024
2. Primeira Liga (94): seasons 2016-2019
3. Eredivisie (88): seasons 2016-2019
4. Super Lig (203): seasons 2016-2019
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
from sqlalchemy.dialects.postgresql import insert as pg_insert
import httpx

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# League configurations
BACKFILL_CONFIG = {
    40: {"name": "EFL Championship", "seasons": [2019, 2020, 2021, 2022, 2023, 2024]},
    307: {"name": "Saudi Pro League", "seasons": [2019, 2020, 2021, 2022, 2023, 2024]},
}


@dataclass
class CoverageStats:
    """Coverage statistics for a league."""
    league_id: int
    league_name: str
    seasons_processed: list
    min_date: Optional[str]
    max_date: Optional[str]
    matches_before: int
    matches_after: int
    matches_upserted: int
    ft_matches_before: int
    ft_matches_after: int
    ft_percentage_before: float
    ft_percentage_after: float
    teams_new: int
    teams_total: int


@dataclass
class MatchData:
    """Match data from API."""
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
    """Team data from API."""
    external_id: int
    name: str
    country: Optional[str]
    team_type: str
    logo_url: Optional[str]


class APIFootballClient:
    """Minimal API-Football client for backfill."""

    def __init__(self):
        host = os.environ.get("RAPIDAPI_HOST", "api-football-v1.p.rapidapi.com")
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
        self.requests_per_minute = int(os.environ.get("API_REQUESTS_PER_MINUTE", 450))
        self._delay = 60 / self.requests_per_minute

    async def _request(self, endpoint: str, params: dict) -> dict:
        """Make rate-limited request."""
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
        """Fetch fixtures for a league/season."""
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
                matches.append(match)
            except Exception as e:
                logger.debug(f"Error parsing fixture: {e}")
                continue

        logger.info(f"Fetched {len(matches)} fixtures")
        return matches

    def _parse_fixture(self, fixture: dict, league_id: int) -> MatchData:
        """Parse API fixture response."""
        fixture_info = fixture.get("fixture", {})
        teams = fixture.get("teams", {})
        goals = fixture.get("goals", {})

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
            home_team_external_id=teams.get("home", {}).get("id"),
            away_team_external_id=teams.get("away", {}).get("id"),
            home_goals=goals.get("home"),
            away_goals=goals.get("away"),
            status=fixture_info.get("status", {}).get("short", "NS"),
        )

    async def get_team(self, team_id: int) -> Optional[TeamData]:
        """Fetch team info."""
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
    """Backfill fixtures and teams without odds."""

    def __init__(self, session: AsyncSession, api: APIFootballClient):
        self.session = session
        self.api = api
        self._teams_cache: dict[int, int] = {}  # external_id -> internal_id
        self._new_teams: list[dict] = []
        self._unmatched_teams: list[dict] = []

    async def get_coverage_stats(self, league_id: int) -> dict:
        """Get current coverage stats for a league."""
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

    async def get_team_count(self, league_id: int) -> int:
        """Get team count for a league."""
        result = await self.session.execute(text("""
            SELECT COUNT(DISTINCT t.id)
            FROM teams t
            JOIN matches m ON t.id = m.home_team_id OR t.id = m.away_team_id
            WHERE m.league_id = :league_id
        """), {"league_id": league_id})
        return result.scalar() or 0

    async def _get_or_create_team(self, external_id: int) -> int:
        """Get or create team, return internal ID."""
        if external_id in self._teams_cache:
            return self._teams_cache[external_id]

        # Check DB
        result = await self.session.execute(text(
            "SELECT id, name FROM teams WHERE external_id = :ext_id"
        ), {"ext_id": external_id})
        row = result.fetchone()

        if row:
            self._teams_cache[external_id] = row.id
            return row.id

        # Fetch from API
        team_data = await self.api.get_team(external_id)

        if not team_data:
            logger.warning(f"Could not fetch team {external_id}")
            self._unmatched_teams.append({
                "external_id": external_id,
                "reason": "API returned no data"
            })
            # Create placeholder
            team_data = TeamData(
                external_id=external_id,
                name=f"Unknown Team {external_id}",
                country=None,
                team_type="club",
                logo_url=None,
            )

        # Upsert team (NO odds columns)
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
        self._new_teams.append({
            "external_id": team_data.external_id,
            "name": team_data.name,
            "internal_id": team_id,
        })

        logger.info(f"Upserted team: {team_data.name} (ID: {team_id})")
        return team_id

    async def _upsert_match(self, match: MatchData) -> bool:
        """Upsert match WITHOUT odds. Returns True if new."""
        # Get team IDs
        home_team_id = await self._get_or_create_team(match.home_team_external_id)
        away_team_id = await self._get_or_create_team(match.away_team_external_id)

        # Check if exists
        result = await self.session.execute(text(
            "SELECT id FROM matches WHERE external_id = :ext_id"
        ), {"ext_id": match.external_id})
        existing = result.scalar_one_or_none()

        if existing:
            # Update only results, NOT odds
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

        # Insert new match WITHOUT odds columns
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
        """Backfill a single league."""
        league_name = BACKFILL_CONFIG.get(league_id, {}).get("name", f"League {league_id}")
        logger.info(f"\n{'='*60}")
        logger.info(f"BACKFILLING: {league_name} (ID: {league_id})")
        logger.info(f"Seasons: {seasons}")
        logger.info(f"{'='*60}")

        # Get before stats
        before = await self.get_coverage_stats(league_id)
        teams_before = await self.get_team_count(league_id)

        logger.info(f"BEFORE: {before['total_matches']} matches, {before['ft_matches']} FT")

        # Reset counters
        self._new_teams = []
        self._unmatched_teams = []
        matches_upserted = 0

        for season in seasons:
            logger.info(f"\n--- Season {season} ---")

            fixtures = await self.api.get_fixtures(league_id, season)

            for match in fixtures:
                try:
                    is_new = await self._upsert_match(match)
                    matches_upserted += 1

                    # Commit every 100 matches
                    if matches_upserted % 100 == 0:
                        await self.session.commit()
                        logger.info(f"Progress: {matches_upserted} matches upserted")

                except Exception as e:
                    logger.error(f"Error upserting match {match.external_id}: {e}")
                    await self.session.rollback()
                    continue

            await self.session.commit()
            logger.info(f"Season {season}: {len(fixtures)} fixtures processed")

        # Get after stats
        after = await self.get_coverage_stats(league_id)
        teams_after = await self.get_team_count(league_id)

        logger.info(f"\nAFTER: {after['total_matches']} matches, {after['ft_matches']} FT")

        ft_pct_before = (before['ft_matches'] / before['total_matches'] * 100) if before['total_matches'] > 0 else 0
        ft_pct_after = (after['ft_matches'] / after['total_matches'] * 100) if after['total_matches'] > 0 else 0

        return CoverageStats(
            league_id=league_id,
            league_name=league_name,
            seasons_processed=seasons,
            min_date=after['min_date'],
            max_date=after['max_date'],
            matches_before=before['total_matches'],
            matches_after=after['total_matches'],
            matches_upserted=matches_upserted,
            ft_matches_before=before['ft_matches'],
            ft_matches_after=after['ft_matches'],
            ft_percentage_before=round(ft_pct_before, 2),
            ft_percentage_after=round(ft_pct_after, 2),
            teams_new=len(self._new_teams),
            teams_total=teams_after,
        )

    def get_new_teams(self) -> list[dict]:
        return self._new_teams

    def get_unmatched_teams(self) -> list[dict]:
        return self._unmatched_teams


async def main():
    """Run the backfill."""
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
    all_unmatched = {}

    try:
        async with async_session() as session:
            backfill = FixturesBackfill(session, api)

            for league_id, config in BACKFILL_CONFIG.items():
                try:
                    stats = await backfill.backfill_league(
                        league_id=league_id,
                        seasons=config["seasons"]
                    )
                    all_stats.append(asdict(stats))

                    # Save coverage report
                    report_path = os.path.join(logs_dir, f"coverage_before_after_{league_id}.json")
                    with open(report_path, "w") as f:
                        json.dump(asdict(stats), f, indent=2, default=str)
                    logger.info(f"Saved: {report_path}")

                    # Save unmatched teams if any
                    unmatched = backfill.get_unmatched_teams()
                    if unmatched:
                        all_unmatched[league_id] = unmatched
                        unmatched_path = os.path.join(logs_dir, f"unmatched_teams_{league_id}.json")
                        with open(unmatched_path, "w") as f:
                            json.dump(unmatched, f, indent=2)
                        logger.info(f"Saved: {unmatched_path}")

                except Exception as e:
                    logger.error(f"Error backfilling league {league_id}: {e}")
                    continue

    finally:
        await api.close()
        await engine.dispose()

    # Print summary
    print("\n" + "=" * 70)
    print("BACKFILL SUMMARY")
    print("=" * 70)

    for stats in all_stats:
        print(f"\n{stats['league_name']} (ID: {stats['league_id']})")
        print(f"  Seasons: {stats['seasons_processed']}")
        print(f"  Matches: {stats['matches_before']} -> {stats['matches_after']} (+{stats['matches_after'] - stats['matches_before']})")
        print(f"  FT%: {stats['ft_percentage_before']}% -> {stats['ft_percentage_after']}%")
        print(f"  New teams: {stats['teams_new']}")
        print(f"  Date range: {stats['min_date']} to {stats['max_date']}")

    print("\n" + "=" * 70)
    print("Reports saved to logs/")
    print("NO odds were written - ready for odds backfill if needed")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
