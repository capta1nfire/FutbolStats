"""API-Football data provider implementation."""

import asyncio
import logging
from datetime import datetime
from typing import Optional

import httpx

from app.config import get_settings
from app.etl.base import DataProvider, MatchData, TeamData
from app.etl.competitions import COMPETITIONS

logger = logging.getLogger(__name__)

settings = get_settings()


class APIFootballProvider(DataProvider):
    """API-Football data provider with rate limiting (supports RapidAPI and API-Sports)."""

    def __init__(self):
        # Detect if using API-Sports directly or RapidAPI
        host = settings.RAPIDAPI_HOST
        if "api-sports.io" in host:
            # API-Sports direct
            self.BASE_URL = f"https://{host}"
            headers = {
                "x-apisports-key": settings.RAPIDAPI_KEY,
            }
        else:
            # RapidAPI
            self.BASE_URL = f"https://{host}/v3"
            headers = {
                "X-RapidAPI-Key": settings.RAPIDAPI_KEY,
                "X-RapidAPI-Host": host,
            }

        self.client = httpx.AsyncClient(
            headers=headers,
            timeout=30.0,
        )
        self.requests_per_minute = settings.API_REQUESTS_PER_MINUTE
        self._request_count = 0
        self._last_reset = datetime.now()

    async def _rate_limited_request(self, endpoint: str, params: dict = None) -> dict:
        """
        Make a rate-limited request to the API.

        Respects API rate limits by adding delays between requests.
        Implements exponential backoff on 429 errors.
        """
        # Calculate delay to respect rate limit
        delay = 60 / self.requests_per_minute

        url = f"{self.BASE_URL}/{endpoint}"
        max_retries = 3
        retry_delay = 5

        for attempt in range(max_retries):
            try:
                response = await self.client.get(url, params=params)

                if response.status_code == 429:
                    # Rate limited - exponential backoff
                    wait_time = retry_delay * (2**attempt)
                    logger.warning(f"Rate limited. Waiting {wait_time}s before retry...")
                    await asyncio.sleep(wait_time)
                    continue

                response.raise_for_status()
                await asyncio.sleep(delay)  # Respect rate limit

                data = response.json()
                if data.get("errors"):
                    logger.error(f"API error: {data['errors']}")
                    return {"response": []}

                return data

            except httpx.HTTPStatusError as e:
                logger.error(f"HTTP error: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay * (2**attempt))
                    continue
                raise

            except httpx.RequestError as e:
                logger.error(f"Request error: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay * (2**attempt))
                    continue
                raise

        return {"response": []}

    def _parse_fixture(self, fixture: dict, league_id: int) -> MatchData:
        """Parse API fixture response into MatchData."""
        fixture_info = fixture.get("fixture", {})
        teams = fixture.get("teams", {})
        goals = fixture.get("goals", {})
        score = fixture.get("score", {})

        # Get competition config for match type/weight
        competition = COMPETITIONS.get(league_id)
        match_type = competition.match_type if competition else "official"
        match_weight = competition.match_weight if competition else 1.0

        # Parse stats if available
        stats = None
        if "statistics" in fixture and fixture["statistics"]:
            stats = self._parse_stats(fixture["statistics"])

        # Parse date - convert to naive UTC for PostgreSQL
        date_str = fixture_info.get("date", "")
        try:
            match_date = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
            # Convert to naive datetime (remove timezone info) for PostgreSQL
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
            stats=stats,
            status=fixture_info.get("status", {}).get("short", "NS"),
            match_type=match_type,
            match_weight=match_weight,
        )

    def _parse_stats(self, statistics: list) -> dict:
        """Parse match statistics from API response.

        API returns stats in order: [home_team, away_team]
        Each item has team info and statistics array.
        """
        stats = {"home": {}, "away": {}}

        for i, team_stats in enumerate(statistics):
            # First team in response is home, second is away
            team_key = "home" if i == 0 else "away"

            for stat in team_stats.get("statistics", []):
                stat_type = stat.get("type", "").lower().replace(" ", "_")
                value = stat.get("value")
                if value is not None:
                    stats[team_key][stat_type] = value

        return stats

    async def get_fixtures(
        self,
        league_id: int,
        season: int,
        from_date: Optional[datetime] = None,
        to_date: Optional[datetime] = None,
    ) -> list[MatchData]:
        """Fetch fixtures for a given league and season."""
        params = {
            "league": league_id,
            "season": season,
        }

        if from_date:
            params["from"] = from_date.strftime("%Y-%m-%d")
        if to_date:
            params["to"] = to_date.strftime("%Y-%m-%d")

        logger.info(f"Fetching fixtures for league {league_id}, season {season}")

        data = await self._rate_limited_request("fixtures", params)
        fixtures = data.get("response", [])

        matches = []
        for fixture in fixtures:
            try:
                match_data = self._parse_fixture(fixture, league_id)
                matches.append(match_data)
            except Exception as e:
                logger.error(f"Error parsing fixture: {e}")
                continue

        logger.info(f"Fetched {len(matches)} fixtures for league {league_id}")
        return matches

    async def get_fixtures_by_date(
        self,
        date: datetime,
        league_ids: Optional[list[int]] = None,
    ) -> list[MatchData]:
        """
        Fetch ALL fixtures for a specific date globally.

        Uses: GET /fixtures?date=YYYY-MM-DD (1 single API call)
        Returns: All matches worldwide for that date.

        If league_ids provided, filters results in memory (no extra API calls).

        Args:
            date: The date to fetch fixtures for.
            league_ids: Optional list of league IDs to filter (in memory).

        Returns:
            List of MatchData for the specified date.
        """
        params = {"date": date.strftime("%Y-%m-%d")}

        logger.info(f"Global sync: Fetching all fixtures for {date.strftime('%Y-%m-%d')}")
        data = await self._rate_limited_request("fixtures", params)
        fixtures = data.get("response", [])

        logger.info(f"Global sync: Received {len(fixtures)} fixtures worldwide")

        # Parse all fixtures
        matches = []
        for fixture in fixtures:
            try:
                league_id = fixture.get("league", {}).get("id")
                match_data = self._parse_fixture(fixture, league_id)
                matches.append(match_data)
            except Exception as e:
                logger.debug(f"Skipping fixture: {e}")
                continue

        # Filter by league_ids if specified (in memory, no extra API calls)
        if league_ids:
            matches = [m for m in matches if m.league_id in league_ids]
            logger.info(f"Global sync: Filtered to {len(matches)} matches for leagues {league_ids}")

        return matches

    async def get_fixture_by_id(self, fixture_id: int) -> Optional[MatchData]:
        """Fetch a single fixture by its ID."""
        data = await self._rate_limited_request("fixtures", {"id": fixture_id})
        fixtures = data.get("response", [])

        if not fixtures:
            return None

        fixture = fixtures[0]
        league_id = fixture.get("league", {}).get("id")
        return self._parse_fixture(fixture, league_id)

    async def get_team(self, team_id: int) -> Optional[TeamData]:
        """Fetch team information by ID."""
        data = await self._rate_limited_request("teams", {"id": team_id})
        teams = data.get("response", [])

        if not teams:
            return None

        team_info = teams[0].get("team", {})

        # Determine if national team based on country
        country = team_info.get("country")
        name = team_info.get("name", "")

        # National teams typically have national=True or their name matches country
        is_national = team_info.get("national", False) or name == country

        return TeamData(
            external_id=team_info.get("id"),
            name=name,
            country=country if not is_national else None,
            team_type="national" if is_national else "club",
            logo_url=team_info.get("logo"),
        )

    # Priority bookmakers for reliable odds (best to worst)
    PRIORITY_BOOKMAKERS = [
        "Bet365",
        "Pinnacle",
        "1xBet",
        "Unibet",
        "William Hill",
        "Betfair",
        "Bwin",
        "888sport",
    ]

    async def get_odds(self, fixture_id: int) -> Optional[dict]:
        """
        Fetch betting odds for a fixture.

        Prioritizes major bookmakers (Bet365, Pinnacle) for reliable odds.
        Returns odds with bookmaker source for transparency.
        """
        data = await self._rate_limited_request("odds", {"fixture": fixture_id})
        odds_data = data.get("response", [])

        if not odds_data:
            return None

        # Collect all available odds by bookmaker
        all_odds = []

        for bookmaker_data in odds_data:
            for bookmaker in bookmaker_data.get("bookmakers", []):
                bookmaker_name = bookmaker.get("name", "Unknown")

                for bet in bookmaker.get("bets", []):
                    if bet.get("name") in ["Match Winner", "3Way Result", "1X2"]:
                        values = bet.get("values", [])
                        odds = {"bookmaker": bookmaker_name}

                        for v in values:
                            if v.get("value") == "Home":
                                odds["odds_home"] = float(v.get("odd", 0))
                            elif v.get("value") == "Draw":
                                odds["odds_draw"] = float(v.get("odd", 0))
                            elif v.get("value") == "Away":
                                odds["odds_away"] = float(v.get("odd", 0))

                        if len(odds) == 4:  # bookmaker + 3 odds
                            all_odds.append(odds)

        if not all_odds:
            return None

        # Find best bookmaker by priority
        for priority_book in self.PRIORITY_BOOKMAKERS:
            for odds in all_odds:
                if odds["bookmaker"].lower() == priority_book.lower():
                    logger.info(f"Using odds from {priority_book} for fixture {fixture_id}")
                    return odds

        # Fallback to first available
        logger.info(f"Using odds from {all_odds[0]['bookmaker']} for fixture {fixture_id}")
        return all_odds[0]

    async def get_fixture_statistics(self, fixture_id: int) -> Optional[dict]:
        """Fetch detailed statistics for a fixture."""
        data = await self._rate_limited_request(
            "fixtures/statistics", {"fixture": fixture_id}
        )
        stats_data = data.get("response", [])

        if not stats_data:
            return None

        return self._parse_stats(stats_data)

    async def get_standings(self, league_id: int, season: int) -> list[dict]:
        """
        Fetch league standings/table.

        Returns list of team standings with position, points, etc.
        """
        data = await self._rate_limited_request(
            "standings", {"league": league_id, "season": season}
        )
        standings_data = data.get("response", [])

        if not standings_data:
            return []

        results = []
        for league_data in standings_data:
            league_standings = league_data.get("league", {}).get("standings", [])
            # Standings can be nested (for groups) - flatten
            for group in league_standings:
                if isinstance(group, list):
                    for team_standing in group:
                        results.append(self._parse_standing(team_standing))
                else:
                    results.append(self._parse_standing(group))

        return results

    def _parse_standing(self, standing: dict) -> dict:
        """Parse a single standing entry."""
        team = standing.get("team", {})
        return {
            "position": standing.get("rank"),
            "team_id": team.get("id"),
            "team_name": team.get("name"),
            "team_logo": team.get("logo"),
            "points": standing.get("points", 0),
            "played": standing.get("all", {}).get("played", 0),
            "won": standing.get("all", {}).get("win", 0),
            "drawn": standing.get("all", {}).get("draw", 0),
            "lost": standing.get("all", {}).get("lose", 0),
            "goals_for": standing.get("all", {}).get("goals", {}).get("for", 0),
            "goals_against": standing.get("all", {}).get("goals", {}).get("against", 0),
            "goal_diff": standing.get("goalsDiff", 0),
            "form": standing.get("form", ""),
        }

    async def get_lineups(self, fixture_id: int) -> Optional[dict]:
        """
        Fetch lineup information for a fixture.

        Returns lineup data including starting XI and substitutes for both teams.
        Available approximately 60 minutes before kickoff.

        Returns:
            Dictionary with home and away lineups, each containing:
            - team_id: Team external ID
            - team_name: Team name
            - formation: e.g., "4-3-3"
            - starting_xi: List of player dicts with id, name, number, pos
            - substitutes: List of substitute player dicts
        """
        data = await self._rate_limited_request("fixtures/lineups", {"fixture": fixture_id})
        lineups_data = data.get("response", [])

        if not lineups_data:
            return None

        result = {"home": None, "away": None}

        for i, lineup in enumerate(lineups_data):
            team_info = lineup.get("team", {})
            coach = lineup.get("coach", {})

            # Parse starting XI
            starting_xi = []
            for player in lineup.get("startXI", []):
                p = player.get("player", {})
                starting_xi.append({
                    "id": p.get("id"),
                    "name": p.get("name"),
                    "number": p.get("number"),
                    "pos": p.get("pos"),
                    "grid": p.get("grid"),  # Position on pitch grid
                })

            # Parse substitutes
            substitutes = []
            for player in lineup.get("substitutes", []):
                p = player.get("player", {})
                substitutes.append({
                    "id": p.get("id"),
                    "name": p.get("name"),
                    "number": p.get("number"),
                    "pos": p.get("pos"),
                })

            lineup_data = {
                "team_id": team_info.get("id"),
                "team_name": team_info.get("name"),
                "team_logo": team_info.get("logo"),
                "formation": lineup.get("formation"),
                "coach": {
                    "id": coach.get("id"),
                    "name": coach.get("name"),
                } if coach else None,
                "starting_xi": starting_xi,
                "substitutes": substitutes,
            }

            # First team is home, second is away
            if i == 0:
                result["home"] = lineup_data
            else:
                result["away"] = lineup_data

        return result

    async def get_players_squad(self, team_id: int) -> list[dict]:
        """
        Fetch full squad for a team.

        Returns all registered players with their market value and position.
        Used to determine the "Equipo de Gala" (best XI).
        """
        data = await self._rate_limited_request("players/squads", {"team": team_id})
        squad_data = data.get("response", [])

        if not squad_data:
            return []

        players = []
        for team_data in squad_data:
            for player in team_data.get("players", []):
                players.append({
                    "id": player.get("id"),
                    "name": player.get("name"),
                    "age": player.get("age"),
                    "number": player.get("number"),
                    "position": player.get("position"),
                    "photo": player.get("photo"),
                })

        return players

    async def close(self) -> None:
        """Close the HTTP client."""
        await self.client.aclose()
