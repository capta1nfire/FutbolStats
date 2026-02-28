"""
Understat Provider for xG/xPTS data.

Provides xG (expected goals) and xPTS (expected points) data from Understat.
Uses Understat's internal API endpoints (discovered Jan 2026).

Usage:
    provider = UnderstatProvider()
    data = await provider.get_match_team_xg(source_match_id="12345")

    # Or fetch by Understat match ID directly
    data = await provider.fetch_match_xg_by_understat_id("12345")

Rate limiting: ~1 req/s to be respectful of Understat servers.

Reference: docs/ARCHITECTURE_SOTA.md section 1.3 (match_understat_team)

API Discovery (Jan 2026):
    Understat changed their HTML structure - they no longer embed JSON data inline
    (datesData, match_info, shotsData). The frontend now fetches data via AJAX.

    How we discovered the endpoints:
    1. Inspected js/league.min.js and js/match.min.js in browser DevTools
    2. Found AJAX calls to internal endpoints that return JSON directly

    Endpoints:
    - GET /getLeagueData/{league}/{season} -> {dates: [...matches], teams: {...}}
    - GET /getMatchData/{match_id} -> {shots: {h: [...], a: [...]}, rosters: {...}}

    Required headers (otherwise returns 404):
    - X-Requested-With: XMLHttpRequest
    - Referer: https://understat.com/league/{league}/{season} (or /match/{id})
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

# Understat configuration
UNDERSTAT_BASE_URL = "https://understat.com"
UNDERSTAT_MATCH_URL = f"{UNDERSTAT_BASE_URL}/match"

# Rate limiting
MIN_REQUEST_INTERVAL = 1.0  # seconds between requests
MAX_RETRIES = 3
RETRY_DELAY_BASE = 2.0  # exponential backoff base


@dataclass
class UnderstatMatchData:
    """
    xG/xPTS data for a match from Understat.

    Column names match EXACTLY the match_understat_team table schema.
    """
    xg_home: float
    xg_away: float
    xpts_home: Optional[float] = None
    xpts_away: Optional[float] = None
    npxg_home: Optional[float] = None  # Non-penalty xG
    npxg_away: Optional[float] = None
    xga_home: Optional[float] = None   # xG against (if available)
    xga_away: Optional[float] = None
    source_version: Optional[str] = None
    captured_at: Optional[datetime] = None


class UnderstatProvider:
    """
    Provider for Understat xG/xPTS data.

    Uses Understat's internal API endpoints to fetch xG data.
    - /getLeagueData/{league}/{season} - List of matches for a league
    - /getMatchData/{match_id} - Shots data for a match (xG calculated from shots)

    Rate limiting: ~1 req/s with exponential backoff on errors.
    """

    SOURCE_VERSION = "understat_scrape_v1"

    def __init__(self, use_mock: bool = False):
        """
        Initialize the Understat provider.

        Args:
            use_mock: If True, return mock data for testing. Default False.
        """
        self.use_mock = use_mock
        self._client: Optional[httpx.AsyncClient] = None
        self._last_request_time: float = 0
        logger.info(f"UnderstatProvider initialized (mock={use_mock})")

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=30.0,
                headers={
                    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                                  "Chrome/120.0.0.0 Safari/537.36",
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                    "Accept-Language": "en-US,en;q=0.5",
                },
                follow_redirects=True,
            )
        return self._client

    async def _rate_limit(self) -> None:
        """Enforce rate limiting between requests."""
        import time
        now = time.time()
        elapsed = now - self._last_request_time
        if elapsed < MIN_REQUEST_INTERVAL:
            await asyncio.sleep(MIN_REQUEST_INTERVAL - elapsed)
        self._last_request_time = time.time()

    async def get_match_team_xg(
        self,
        source_match_id: str,
    ) -> Optional[UnderstatMatchData]:
        """
        Get xG/xPTS data for a match from Understat.

        Args:
            source_match_id: The Understat match ID (from match_external_refs).

        Returns:
            UnderstatMatchData with xG metrics, or None if not available.
        """
        if self.use_mock:
            return self._get_mock_data(source_match_id)

        return await self.fetch_match_xg_by_understat_id(source_match_id)

    async def fetch_match_xg_by_understat_id(
        self,
        understat_match_id: str,
    ) -> Optional[UnderstatMatchData]:
        """
        Fetch xG/xPTS data directly from Understat by match ID.

        Uses Understat's internal API endpoint /getMatchData/{id} which returns
        shots data. xG is calculated by summing shot xG values.

        Args:
            understat_match_id: The Understat match ID.

        Returns:
            UnderstatMatchData with xG metrics, or None if failed.
        """
        # Use internal API endpoint (discovered Jan 2026)
        url = f"{UNDERSTAT_BASE_URL}/getMatchData/{understat_match_id}"

        for attempt in range(MAX_RETRIES):
            try:
                await self._rate_limit()
                client = await self._get_client()

                # Required headers for the internal API
                headers = {
                    "X-Requested-With": "XMLHttpRequest",
                    "Referer": f"{UNDERSTAT_MATCH_URL}/{understat_match_id}",
                }

                response = await client.get(url, headers=headers)

                # Handle rate limiting
                if response.status_code == 429:
                    delay = RETRY_DELAY_BASE * (2 ** attempt)
                    logger.warning(f"Rate limited by Understat, waiting {delay}s...")
                    await asyncio.sleep(delay)
                    continue

                # Handle server errors
                if response.status_code >= 500:
                    delay = RETRY_DELAY_BASE * (2 ** attempt)
                    logger.warning(f"Understat server error {response.status_code}, retrying in {delay}s...")
                    await asyncio.sleep(delay)
                    continue

                # Handle not found
                if response.status_code == 404:
                    logger.debug(f"Understat match {understat_match_id} not found")
                    return None

                response.raise_for_status()
                data = response.json()

                # Parse xG from shots data
                return self._parse_match_api_response(data, understat_match_id)

            except httpx.TimeoutException:
                logger.warning(f"Timeout fetching Understat match {understat_match_id}, attempt {attempt + 1}")
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(RETRY_DELAY_BASE * (2 ** attempt))
                continue

            except Exception as e:
                logger.error(f"Error fetching Understat match {understat_match_id}: {e}")
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(RETRY_DELAY_BASE * (2 ** attempt))
                continue

        logger.error(f"Failed to fetch Understat match {understat_match_id} after {MAX_RETRIES} attempts")
        return None

    def _parse_match_api_response(
        self,
        data: dict,
        understat_match_id: str,
    ) -> Optional[UnderstatMatchData]:
        """
        Parse xG data from Understat's getMatchData API response.

        The API returns shots data, from which we calculate xG by summing.

        Args:
            data: JSON response from /getMatchData/{id}
            understat_match_id: Match ID for logging.

        Returns:
            UnderstatMatchData or None if parsing failed.
        """
        try:
            shots = data.get("shots", {})
            home_shots = shots.get("h", [])
            away_shots = shots.get("a", [])

            if not home_shots and not away_shots:
                logger.debug(f"No shots data for Understat match {understat_match_id}")
                return None

            # Calculate xG by summing shot xG values
            xg_home = sum(float(s.get("xG", 0)) for s in home_shots)
            xg_away = sum(float(s.get("xG", 0)) for s in away_shots)

            # Calculate non-penalty xG
            npxg_home = sum(
                float(s.get("xG", 0))
                for s in home_shots
                if s.get("situation") != "Penalty"
            )
            npxg_away = sum(
                float(s.get("xG", 0))
                for s in away_shots
                if s.get("situation") != "Penalty"
            )

            return UnderstatMatchData(
                xg_home=round(xg_home, 2),
                xg_away=round(xg_away, 2),
                xpts_home=None,  # Not available from match API
                xpts_away=None,
                npxg_home=round(npxg_home, 2),
                npxg_away=round(npxg_away, 2),
                xga_home=round(xg_away, 2),  # xGA home = xG away
                xga_away=round(xg_home, 2),  # xGA away = xG home
                source_version=self.SOURCE_VERSION,
                captured_at=datetime.now(timezone.utc),
            )

        except Exception as e:
            logger.error(f"Error parsing Understat match API {understat_match_id}: {e}")
            return None

    def _get_mock_data(self, source_match_id: str) -> UnderstatMatchData:
        """
        Return mock data for testing purposes.

        Mock values are reasonable ranges for top-5 league matches.
        """
        import random

        # Generate plausible xG values (typically 0.5 - 2.5 per team)
        xg_home = round(random.uniform(0.8, 2.2), 2)
        xg_away = round(random.uniform(0.5, 1.8), 2)

        return UnderstatMatchData(
            xg_home=xg_home,
            xg_away=xg_away,
            xpts_home=round(random.uniform(0.8, 1.8), 2),
            xpts_away=round(random.uniform(0.6, 1.5), 2),
            npxg_home=round(xg_home * random.uniform(0.85, 0.98), 2),
            npxg_away=round(xg_away * random.uniform(0.85, 0.98), 2),
            xga_home=round(xg_away, 2),  # xGA home = xG away
            xga_away=round(xg_home, 2),  # xGA away = xG home
            source_version=f"{self.SOURCE_VERSION}_mock",
            captured_at=datetime.now(timezone.utc),
        )

    async def close(self) -> None:
        """Close any open connections."""
        if self._client:
            await self._client.aclose()
            self._client = None
        logger.debug("UnderstatProvider closed")

    async def get_league_matches(
        self,
        league: str,
        season: str,
    ) -> list[dict]:
        """
        Get all matches for a league/season from Understat.

        This can be used to find Understat match IDs for matching.

        Args:
            league: Understat league name (e.g., "EPL", "La_Liga", "Serie_A", "Bundesliga", "Ligue_1")
            season: Season year (e.g., "2025" for 2025-26 season)

        Returns:
            List of match dicts with id, datetime, home team, away team, score, xG.
        """
        # Use Understat's internal API endpoint (discovered Jan 2026)
        # The old HTML scraping method (datesData regex) no longer works
        url = f"{UNDERSTAT_BASE_URL}/getLeagueData/{league}/{season}"

        for attempt in range(MAX_RETRIES):
            try:
                await self._rate_limit()
                client = await self._get_client()

                # Required headers for the internal API
                headers = {
                    "X-Requested-With": "XMLHttpRequest",
                    "Referer": f"{UNDERSTAT_BASE_URL}/league/{league}/{season}",
                }

                response = await client.get(url, headers=headers)

                # Handle rate limiting
                if response.status_code == 429:
                    delay = RETRY_DELAY_BASE * (2 ** attempt)
                    logger.warning(f"Rate limited by Understat, waiting {delay}s...")
                    await asyncio.sleep(delay)
                    continue

                # Handle server errors
                if response.status_code >= 500:
                    delay = RETRY_DELAY_BASE * (2 ** attempt)
                    logger.warning(f"Understat server error {response.status_code}, retrying in {delay}s...")
                    await asyncio.sleep(delay)
                    continue

                # Handle not found
                if response.status_code == 404:
                    logger.warning(f"Understat league {league}/{season} not found (404)")
                    return []

                response.raise_for_status()
                data = response.json()

                # Extract matches from "dates" key
                dates_data = data.get("dates", [])
                if not dates_data:
                    logger.warning(f"No dates data in response for {league}/{season}")
                    return []

                # Parse matches from the API response
                matches = []
                for match in dates_data:
                    matches.append({
                        "id": match.get("id"),
                        "datetime": match.get("datetime"),
                        "home_team": match.get("h", {}).get("title"),
                        "away_team": match.get("a", {}).get("title"),
                        "home_goals": match.get("goals", {}).get("h"),
                        "away_goals": match.get("goals", {}).get("a"),
                        "xg_home": match.get("xG", {}).get("h"),
                        "xg_away": match.get("xG", {}).get("a"),
                        "is_result": match.get("isResult", False),
                    })

                logger.info(f"Fetched {len(matches)} matches from Understat {league}/{season}")
                return matches

            except httpx.TimeoutException:
                logger.warning(f"Timeout fetching Understat league {league}/{season}, attempt {attempt + 1}")
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(RETRY_DELAY_BASE * (2 ** attempt))
                continue

            except Exception as e:
                logger.error(f"Error fetching league matches for {league}/{season}: {e}")
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(RETRY_DELAY_BASE * (2 ** attempt))
                continue

        logger.error(f"Failed to fetch Understat league {league}/{season} after {MAX_RETRIES} attempts")
        return []
