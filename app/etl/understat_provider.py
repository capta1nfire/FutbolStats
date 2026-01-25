"""
Understat Provider for xG/xPTS data.

Provides xG (expected goals) and xPTS (expected points) data from Understat.
Understat embeds JSON data in their HTML pages, which we parse.

Usage:
    provider = UnderstatProvider()
    data = await provider.get_match_team_xg(source_match_id="12345")

    # Or fetch by Understat match ID directly
    data = await provider.fetch_match_xg_by_understat_id("12345")

Rate limiting: ~1 req/s to be respectful of Understat servers.

Reference: docs/ARCHITECTURE_SOTA.md section 1.3 (match_understat_team)
"""

import asyncio
import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime
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

# Regex to extract JSON from HTML script tags
# Understat embeds data like: var matchData = JSON.parse('...');
MATCH_DATA_PATTERN = re.compile(r"var\s+match_info\s*=\s*JSON\.parse\('(.+?)'\)")
SHOTS_DATA_PATTERN = re.compile(r"var\s+shotsData\s*=\s*JSON\.parse\('(.+?)'\)")


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


def _decode_understat_json(encoded: str) -> dict:
    """
    Decode Understat's encoded JSON string.

    Understat escapes characters in their JSON, we need to unescape them.
    """
    # Unescape the string (Understat uses \\x hex escapes)
    decoded = encoded.encode().decode('unicode_escape')
    return json.loads(decoded)


class UnderstatProvider:
    """
    Provider for Understat xG/xPTS data.

    Fetches data by scraping Understat match pages. The xG data is embedded
    as JSON in script tags within the HTML.

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

        This is the real implementation that scrapes Understat's match page.
        The xG data is embedded as JSON in script tags.

        Args:
            understat_match_id: The Understat match ID.

        Returns:
            UnderstatMatchData with xG metrics, or None if failed.
        """
        url = f"{UNDERSTAT_MATCH_URL}/{understat_match_id}"

        for attempt in range(MAX_RETRIES):
            try:
                await self._rate_limit()
                client = await self._get_client()

                response = await client.get(url)

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
                html = response.text

                # Parse the embedded JSON data
                return self._parse_match_page(html, understat_match_id)

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

    def _parse_match_page(
        self,
        html: str,
        understat_match_id: str,
    ) -> Optional[UnderstatMatchData]:
        """
        Parse Understat match page HTML to extract xG data.

        Understat embeds data in script tags like:
            var match_info = JSON.parse('{"h":{"title":"Liverpool",...,"xG":"1.45"},...}');
            var shotsData = JSON.parse('{"h":[...],"a":[...]}');

        Args:
            html: The HTML content of the match page.
            understat_match_id: Match ID for logging.

        Returns:
            UnderstatMatchData or None if parsing failed.
        """
        try:
            # Extract match_info JSON
            match_info_match = MATCH_DATA_PATTERN.search(html)
            if not match_info_match:
                logger.warning(f"Could not find match_info in Understat page {understat_match_id}")
                return None

            match_info = _decode_understat_json(match_info_match.group(1))

            # Extract xG from match_info
            # Structure: {"h": {"xG": "1.45", ...}, "a": {"xG": "0.87", ...}}
            home_data = match_info.get("h", {})
            away_data = match_info.get("a", {})

            xg_home = float(home_data.get("xG", 0))
            xg_away = float(away_data.get("xG", 0))

            # Try to extract shots data for npxG calculation
            npxg_home = None
            npxg_away = None
            shots_match = SHOTS_DATA_PATTERN.search(html)
            if shots_match:
                try:
                    shots_data = _decode_understat_json(shots_match.group(1))
                    # Calculate npxG by summing xG of non-penalty shots
                    home_shots = shots_data.get("h", [])
                    away_shots = shots_data.get("a", [])

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
                except Exception as e:
                    logger.debug(f"Could not parse shots data: {e}")

            # Note: xPTS is not directly available on match pages
            # It's calculated from match outcomes, so we leave it as None
            # (could be derived from league standings page if needed)

            return UnderstatMatchData(
                xg_home=round(xg_home, 2),
                xg_away=round(xg_away, 2),
                xpts_home=None,  # Not available on match page
                xpts_away=None,
                npxg_home=round(npxg_home, 2) if npxg_home is not None else None,
                npxg_away=round(npxg_away, 2) if npxg_away is not None else None,
                xga_home=round(xg_away, 2),  # xGA home = xG away
                xga_away=round(xg_home, 2),  # xGA away = xG home
                source_version=self.SOURCE_VERSION,
                captured_at=datetime.utcnow(),
            )

        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error for Understat match {understat_match_id}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error parsing Understat match {understat_match_id}: {e}")
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
            captured_at=datetime.utcnow(),
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
