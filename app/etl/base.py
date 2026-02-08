"""Abstract base class for data providers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class TeamData:
    """Data transfer object for team information."""

    external_id: int
    name: str
    country: Optional[str]
    team_type: str  # "national" or "club"
    logo_url: Optional[str]


@dataclass
class MatchData:
    """Data transfer object for match information."""

    external_id: int
    date: datetime
    league_id: int
    season: int
    home_team_external_id: int
    away_team_external_id: int
    home_goals: Optional[int]
    away_goals: Optional[int]
    stats: Optional[dict]
    status: str
    elapsed: Optional[int]  # Current minute for live matches
    # --- Fields with defaults must come after fields without defaults ---
    round: Optional[str] = None  # API-Football: fixture.league.round (e.g., "Regular Season - 21")
    elapsed_extra: Optional[int] = None  # Added/injury time (e.g., 3 for 90+3)
    match_type: str = "official"  # "official" or "friendly"
    match_weight: float = 1.0
    odds_home: Optional[float] = None
    odds_draw: Optional[float] = None
    odds_away: Optional[float] = None
    venue_name: Optional[str] = None
    venue_city: Optional[str] = None


class DataProvider(ABC):
    """Abstract base class for football data providers."""

    @abstractmethod
    async def get_fixtures(
        self,
        league_id: int,
        season: int,
        from_date: Optional[datetime] = None,
        to_date: Optional[datetime] = None,
    ) -> list[MatchData]:
        """
        Fetch fixtures for a given league and season.

        Args:
            league_id: The competition ID.
            season: The season year.
            from_date: Optional start date filter.
            to_date: Optional end date filter.

        Returns:
            List of MatchData objects.
        """
        pass

    @abstractmethod
    async def get_fixture_by_id(self, fixture_id: int) -> Optional[MatchData]:
        """
        Fetch a single fixture by its ID.

        Args:
            fixture_id: The fixture external ID.

        Returns:
            MatchData object or None if not found.
        """
        pass

    @abstractmethod
    async def get_team(self, team_id: int) -> Optional[TeamData]:
        """
        Fetch team information by ID.

        Args:
            team_id: The team external ID.

        Returns:
            TeamData object or None if not found.
        """
        pass

    @abstractmethod
    async def get_odds_all(self, fixture_id: int) -> Optional[list[dict]]:
        """
        Fetch all bookmaker odds for a fixture.

        Returns:
            List of dicts with bookmaker/odds_home/odds_draw/odds_away, or None.
        """
        pass

    # Priority bookmakers for selecting "primary" odds (best to worst)
    PRIORITY_BOOKMAKERS = [
        "Bet365", "Pinnacle", "1xBet", "Unibet",
        "William Hill", "Betfair", "Bwin", "888sport",
    ]

    async def get_odds(self, fixture_id: int) -> Optional[dict]:
        """
        Fetch best-priority bookmaker odds for a fixture.
        Calls get_odds_all() internally (single API call).

        Returns:
            Dictionary with bookmaker/odds_home/odds_draw/odds_away or None.
        """
        all_odds = await self.get_odds_all(fixture_id)
        if not all_odds:
            return None
        for priority_book in self.PRIORITY_BOOKMAKERS:
            for odds in all_odds:
                if odds["bookmaker"].lower() == priority_book.lower():
                    return odds
        return all_odds[0]

    @abstractmethod
    async def close(self) -> None:
        """Close any open connections."""
        pass
