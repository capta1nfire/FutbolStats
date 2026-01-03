"""ETL pipeline orchestrator."""

import logging
from datetime import datetime
from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.etl.api_football import APIFootballProvider
from app.etl.base import DataProvider, MatchData, TeamData
from app.etl.competitions import COMPETITIONS, Competition
from app.models import Match, Team

logger = logging.getLogger(__name__)


class ETLPipeline:
    """Orchestrates the ETL process for football data."""

    def __init__(self, provider: DataProvider, session: AsyncSession):
        self.provider = provider
        self.session = session
        self._teams_cache: dict[int, int] = {}  # external_id -> internal_id

    async def _get_or_create_team(self, external_id: int) -> int:
        """Get or create a team, returning the internal ID."""
        # Check cache first
        if external_id in self._teams_cache:
            return self._teams_cache[external_id]

        # Check database
        result = await self.session.execute(
            select(Team).where(Team.external_id == external_id)
        )
        team = result.scalar_one_or_none()

        if team:
            self._teams_cache[external_id] = team.id
            return team.id

        # Fetch from API and create
        team_data = await self.provider.get_team(external_id)
        if not team_data:
            logger.warning(f"Could not fetch team {external_id}")
            # Create placeholder team
            team_data = TeamData(
                external_id=external_id,
                name=f"Unknown Team {external_id}",
                country=None,
                team_type="national",
                logo_url=None,
            )

        new_team = Team(
            external_id=team_data.external_id,
            name=team_data.name,
            country=team_data.country,
            team_type=team_data.team_type,
            logo_url=team_data.logo_url,
        )
        self.session.add(new_team)
        await self.session.flush()

        self._teams_cache[external_id] = new_team.id
        logger.info(f"Created team: {team_data.name} (ID: {new_team.id})")

        return new_team.id

    async def _upsert_match(self, match_data: MatchData) -> Match:
        """Create or update a match in the database."""
        # Check if match exists
        result = await self.session.execute(
            select(Match).where(Match.external_id == match_data.external_id)
        )
        existing_match = result.scalar_one_or_none()

        # Get or create teams
        home_team_id = await self._get_or_create_team(match_data.home_team_external_id)
        away_team_id = await self._get_or_create_team(match_data.away_team_external_id)

        if existing_match:
            # Update existing match
            existing_match.date = match_data.date
            existing_match.home_goals = match_data.home_goals
            existing_match.away_goals = match_data.away_goals
            existing_match.stats = match_data.stats
            existing_match.status = match_data.status
            existing_match.odds_home = match_data.odds_home
            existing_match.odds_draw = match_data.odds_draw
            existing_match.odds_away = match_data.odds_away
            return existing_match

        # Create new match
        new_match = Match(
            external_id=match_data.external_id,
            date=match_data.date,
            league_id=match_data.league_id,
            season=match_data.season,
            home_team_id=home_team_id,
            away_team_id=away_team_id,
            home_goals=match_data.home_goals,
            away_goals=match_data.away_goals,
            stats=match_data.stats,
            status=match_data.status,
            match_type=match_data.match_type,
            match_weight=match_data.match_weight,
            odds_home=match_data.odds_home,
            odds_draw=match_data.odds_draw,
            odds_away=match_data.odds_away,
        )
        self.session.add(new_match)
        return new_match

    async def sync_league(
        self,
        league_id: int,
        season: int,
        from_date: Optional[datetime] = None,
        to_date: Optional[datetime] = None,
        fetch_odds: bool = False,
    ) -> dict:
        """
        Sync fixtures for a specific league and season.

        Args:
            league_id: The competition ID.
            season: The season year.
            from_date: Optional start date filter.
            to_date: Optional end date filter.
            fetch_odds: Whether to also fetch odds for each fixture.

        Returns:
            Dictionary with sync statistics.
        """
        competition = COMPETITIONS.get(league_id)
        comp_name = competition.name if competition else f"League {league_id}"
        logger.info(f"Syncing {comp_name} - Season {season}")

        # Fetch fixtures
        fixtures = await self.provider.get_fixtures(
            league_id=league_id,
            season=season,
            from_date=from_date,
            to_date=to_date,
        )

        matches_synced = 0
        teams_synced = set()

        for match_data in fixtures:
            try:
                # Fetch odds if requested
                if fetch_odds and match_data.status == "NS":
                    odds = await self.provider.get_odds(match_data.external_id)
                    if odds:
                        match_data.odds_home = odds.get("odds_home")
                        match_data.odds_draw = odds.get("odds_draw")
                        match_data.odds_away = odds.get("odds_away")

                await self._upsert_match(match_data)
                matches_synced += 1
                teams_synced.add(match_data.home_team_external_id)
                teams_synced.add(match_data.away_team_external_id)

            except Exception as e:
                logger.error(f"Error syncing match {match_data.external_id}: {e}")
                continue

        await self.session.commit()

        result = {
            "league_id": league_id,
            "league_name": comp_name,
            "season": season,
            "matches_synced": matches_synced,
            "teams_synced": len(teams_synced),
        }
        logger.info(f"Sync complete: {result}")

        return result

    async def sync_multiple_leagues(
        self,
        league_ids: list[int],
        season: int,
        from_date: Optional[datetime] = None,
        to_date: Optional[datetime] = None,
        fetch_odds: bool = False,
    ) -> dict:
        """
        Sync fixtures for multiple leagues.

        Args:
            league_ids: List of competition IDs.
            season: The season year.
            from_date: Optional start date filter.
            to_date: Optional end date filter.
            fetch_odds: Whether to also fetch odds.

        Returns:
            Dictionary with aggregate sync statistics.
        """
        total_matches = 0
        total_teams = set()
        results = []

        for league_id in league_ids:
            try:
                result = await self.sync_league(
                    league_id=league_id,
                    season=season,
                    from_date=from_date,
                    to_date=to_date,
                    fetch_odds=fetch_odds,
                )
                total_matches += result["matches_synced"]
                results.append(result)
            except Exception as e:
                logger.error(f"Error syncing league {league_id}: {e}")
                continue

        return {
            "leagues_synced": len(results),
            "total_matches_synced": total_matches,
            "total_teams_synced": len(self._teams_cache),
            "details": results,
        }

    async def sync_historical_data(
        self,
        league_ids: list[int],
        start_year: int = 2018,
        end_year: Optional[int] = None,
    ) -> dict:
        """
        Sync historical data for multiple seasons.

        Args:
            league_ids: List of competition IDs.
            start_year: First season year (default 2018).
            end_year: Last season year (default current year).

        Returns:
            Dictionary with aggregate sync statistics.
        """
        if end_year is None:
            end_year = datetime.now().year

        logger.info(f"Starting historical sync from {start_year} to {end_year}")

        all_results = []
        for year in range(start_year, end_year + 1):
            logger.info(f"Syncing season {year}...")
            result = await self.sync_multiple_leagues(
                league_ids=league_ids,
                season=year,
            )
            all_results.append({"season": year, **result})

        total_matches = sum(r["total_matches_synced"] for r in all_results)

        return {
            "seasons_synced": len(all_results),
            "total_matches_synced": total_matches,
            "total_teams_synced": len(self._teams_cache),
            "by_season": all_results,
        }


async def create_etl_pipeline(session: AsyncSession) -> ETLPipeline:
    """Factory function to create an ETL pipeline with API-Football provider."""
    provider = APIFootballProvider()
    return ETLPipeline(provider=provider, session=session)
