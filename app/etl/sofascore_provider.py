"""
Sofascore Provider for XI/ratings/formation data.

Provides lineup and player rating data from Sofascore API endpoints.
Aligns with SCRAPING_ARCHITECTURE.md (resilient, best-effort, PIT-safe).

Usage:
    provider = SofascoreProvider()
    result = await provider.get_match_lineup(sofascore_event_id="12345678")

API Endpoints (semi-public JSON):
- https://api.sofascore.com/api/v1/event/{event_id}/lineups
- https://api.sofascore.com/api/v1/event/{event_id}

Rate limiting: ~1 req/s with exponential backoff on errors.

Reference:
- SCRAPING_ARCHITECTURE.md (resilient scraping)
- docs/ARCHITECTURE_SOTA.md section 1.3 (Sofascore tables)
- docs/FEATURE_DICTIONARY_SOTA.md section 3 (xi_* features)
"""

import asyncio
import logging
import random
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

import httpx

logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

SOFASCORE_API_BASE = "https://api.sofascore.com/api/v1"

# Rate limiting (be nice)
MIN_REQUEST_INTERVAL = 1.0  # seconds between requests
MAX_RETRIES = 4
RETRY_DELAY_BASE = 2.0  # exponential backoff base
JITTER_MAX = 1.0  # random jitter

# HTTP headers (mimic browser)
DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Origin": "https://www.sofascore.com",
    "Referer": "https://www.sofascore.com/",
}


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class SofascorePlayerData:
    """
    Player data from Sofascore lineup.

    Column names match match_sofascore_player table schema.
    """
    player_id_ext: str
    position: str  # GK/DEF/MID/FWD
    is_starter: bool
    rating_pre_match: Optional[float] = None
    rating_recent_form: Optional[float] = None
    minutes_expected: Optional[int] = None
    name: Optional[str] = None  # For debugging, not persisted


@dataclass
class SofascoreLineupData:
    """
    Lineup data for a single team side.
    """
    team_side: str  # 'home' or 'away'
    formation: Optional[str] = None
    players: list[SofascorePlayerData] = field(default_factory=list)


@dataclass
class SofascoreMatchLineup:
    """
    Complete lineup data for a match.

    Returned by get_match_lineup().
    """
    source_event_id: str
    home: Optional[SofascoreLineupData] = None
    away: Optional[SofascoreLineupData] = None
    captured_at: Optional[datetime] = None
    integrity_score: float = 0.0  # 0-1, based on completeness
    error: Optional[str] = None


# =============================================================================
# POSITION MAPPING
# =============================================================================

# Sofascore position codes to our simplified categories
POSITION_MAP = {
    "G": "GK",
    "GK": "GK",
    "D": "DEF",
    "CB": "DEF",
    "LB": "DEF",
    "RB": "DEF",
    "LWB": "DEF",
    "RWB": "DEF",
    "M": "MID",
    "CM": "MID",
    "CDM": "MID",
    "DM": "MID",
    "AM": "MID",
    "CAM": "MID",
    "LM": "MID",
    "RM": "MID",
    "F": "FWD",
    "FW": "FWD",
    "CF": "FWD",
    "ST": "FWD",
    "LW": "FWD",
    "RW": "FWD",
    "SS": "FWD",
}


def normalize_position(pos: Optional[str]) -> str:
    """Normalize position to GK/DEF/MID/FWD."""
    if not pos:
        return "MID"  # Default to MID if unknown
    pos_upper = pos.upper().strip()
    return POSITION_MAP.get(pos_upper, "MID")


# =============================================================================
# PROVIDER CLASS
# =============================================================================


class SofascoreProvider:
    """
    Provider for Sofascore lineup/ratings data.

    Uses Sofascore's semi-public JSON API endpoints.
    Implements resilient fetching with backoff and jitter.

    Error handling per SCRAPING_ARCHITECTURE.md:
    - Soft fails (timeout, 429, 5xx): retry with backoff
    - Hard fails (schema break, 404): skip, don't burn requests
    """

    SCHEMA_VERSION = "sofascore.lineup.v1"

    def __init__(self, use_mock: bool = False):
        """
        Initialize the Sofascore provider.

        Args:
            use_mock: If True, return mock data for testing.
        """
        self.use_mock = use_mock
        self._client: Optional[httpx.AsyncClient] = None
        self._last_request_time: float = 0
        logger.info(f"SofascoreProvider initialized (mock={use_mock})")

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=30.0,
                headers=DEFAULT_HEADERS,
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

    async def _fetch_json(
        self,
        url: str,
        event_id: str,
    ) -> tuple[Optional[dict], Optional[str]]:
        """
        Fetch JSON with retry logic.

        Returns:
            Tuple of (data, error_message).
            On success: (dict, None)
            On failure: (None, error_string)
        """
        for attempt in range(MAX_RETRIES):
            try:
                await self._rate_limit()
                client = await self._get_client()

                response = await client.get(url)

                # Handle soft fails (retryable)
                if response.status_code == 429:
                    delay = RETRY_DELAY_BASE * (2 ** attempt) + random.uniform(0, JITTER_MAX)
                    logger.warning(f"[SOFASCORE] Rate limited (429), waiting {delay:.1f}s...")
                    await asyncio.sleep(delay)
                    continue

                if response.status_code >= 500:
                    delay = RETRY_DELAY_BASE * (2 ** attempt) + random.uniform(0, JITTER_MAX)
                    logger.warning(f"[SOFASCORE] Server error {response.status_code}, retrying in {delay:.1f}s...")
                    await asyncio.sleep(delay)
                    continue

                # Handle hard fails (non-retryable)
                if response.status_code == 404:
                    logger.debug(f"[SOFASCORE] Event {event_id} not found (404)")
                    return None, "not_found"

                if response.status_code == 403:
                    # Could be geo-blocked or detected as bot
                    logger.warning(f"[SOFASCORE] Access denied (403) for event {event_id}")
                    return None, "access_denied"

                response.raise_for_status()

                # Parse JSON
                try:
                    data = response.json()
                    return data, None
                except Exception as e:
                    logger.error(f"[SOFASCORE] JSON parse error for event {event_id}: {e}")
                    return None, "json_parse_error"

            except httpx.TimeoutException:
                delay = RETRY_DELAY_BASE * (2 ** attempt) + random.uniform(0, JITTER_MAX)
                logger.warning(f"[SOFASCORE] Timeout for event {event_id}, attempt {attempt + 1}, waiting {delay:.1f}s")
                await asyncio.sleep(delay)
                continue

            except httpx.RequestError as e:
                delay = RETRY_DELAY_BASE * (2 ** attempt) + random.uniform(0, JITTER_MAX)
                logger.warning(f"[SOFASCORE] Request error for event {event_id}: {e}, waiting {delay:.1f}s")
                await asyncio.sleep(delay)
                continue

            except Exception as e:
                logger.error(f"[SOFASCORE] Unexpected error for event {event_id}: {e}")
                return None, f"unexpected_error: {str(e)[:100]}"

        logger.error(f"[SOFASCORE] Failed to fetch event {event_id} after {MAX_RETRIES} attempts")
        return None, "max_retries_exceeded"

    async def get_match_lineup(
        self,
        sofascore_event_id: str,
    ) -> SofascoreMatchLineup:
        """
        Get lineup/formation/ratings for a match.

        Args:
            sofascore_event_id: Sofascore event ID.

        Returns:
            SofascoreMatchLineup with home/away lineup data.
            On failure, returns result with error field set.
        """
        if self.use_mock:
            return self._get_mock_lineup(sofascore_event_id)

        url = f"{SOFASCORE_API_BASE}/event/{sofascore_event_id}/lineups"

        data, error = await self._fetch_json(url, sofascore_event_id)

        if error:
            return SofascoreMatchLineup(
                source_event_id=sofascore_event_id,
                captured_at=datetime.utcnow(),
                error=error,
            )

        try:
            return self._parse_lineup_response(data, sofascore_event_id)
        except Exception as e:
            logger.error(f"[SOFASCORE] Schema break parsing lineups for {sofascore_event_id}: {e}")
            return SofascoreMatchLineup(
                source_event_id=sofascore_event_id,
                captured_at=datetime.utcnow(),
                error=f"schema_break: {str(e)[:100]}",
            )

    def _parse_lineup_response(
        self,
        data: dict,
        event_id: str,
    ) -> SofascoreMatchLineup:
        """
        Parse Sofascore lineups API response.

        Expected structure:
        {
            "home": {
                "players": [
                    {
                        "player": {"id": ..., "name": ..., "position": ...},
                        "substitute": false,
                        "statistics": {"rating": ...}
                    },
                    ...
                ],
                "formation": "4-3-3"
            },
            "away": {...}
        }
        """
        captured_at = datetime.utcnow()

        home_data = data.get("home", {})
        away_data = data.get("away", {})

        home_lineup = self._parse_team_lineup(home_data, "home")
        away_lineup = self._parse_team_lineup(away_data, "away")

        # Calculate integrity score
        integrity = self._calculate_integrity(home_lineup, away_lineup)

        return SofascoreMatchLineup(
            source_event_id=event_id,
            home=home_lineup,
            away=away_lineup,
            captured_at=captured_at,
            integrity_score=integrity,
        )

    def _parse_team_lineup(
        self,
        team_data: dict,
        team_side: str,
    ) -> Optional[SofascoreLineupData]:
        """Parse lineup for a single team."""
        if not team_data:
            return None

        formation = team_data.get("formation")
        players_raw = team_data.get("players", [])

        players = []
        for player_entry in players_raw:
            player_info = player_entry.get("player", {})
            stats = player_entry.get("statistics", {})

            player_id = player_info.get("id")
            if not player_id:
                continue

            # Determine if starter (not a substitute)
            is_starter = not player_entry.get("substitute", False)

            # Extract position
            raw_position = player_info.get("position", "")
            position = normalize_position(raw_position)

            # Extract ratings
            rating = stats.get("rating")
            rating_pre_match = None
            rating_recent_form = None

            if rating:
                try:
                    rating_float = float(rating)
                    # Sofascore ratings are typically post-match
                    # For pre-match, we might get averageRating or similar
                    rating_recent_form = rating_float
                except (ValueError, TypeError):
                    pass

            # Try to get pre-match rating if available
            avg_rating = player_info.get("averageRating")
            if avg_rating:
                try:
                    rating_pre_match = float(avg_rating)
                except (ValueError, TypeError):
                    pass

            players.append(SofascorePlayerData(
                player_id_ext=str(player_id),
                position=position,
                is_starter=is_starter,
                rating_pre_match=rating_pre_match,
                rating_recent_form=rating_recent_form,
                name=player_info.get("name"),
            ))

        return SofascoreLineupData(
            team_side=team_side,
            formation=formation,
            players=players,
        )

    def _calculate_integrity(
        self,
        home: Optional[SofascoreLineupData],
        away: Optional[SofascoreLineupData],
    ) -> float:
        """
        Calculate data integrity score (0-1).

        Based on:
        - Presence of lineups (0.3 each)
        - Number of starters (0.15 each for 11)
        - Presence of formation (0.05 each)
        """
        score = 0.0

        if home:
            score += 0.3
            starters_home = sum(1 for p in home.players if p.is_starter)
            score += 0.15 * min(starters_home / 11.0, 1.0)
            if home.formation:
                score += 0.05

        if away:
            score += 0.3
            starters_away = sum(1 for p in away.players if p.is_starter)
            score += 0.15 * min(starters_away / 11.0, 1.0)
            if away.formation:
                score += 0.05

        return round(score, 2)

    async def get_scheduled_events(
        self,
        date: datetime,
    ) -> list[dict]:
        """
        Get all scheduled football events for a specific date.

        Uses Sofascore's scheduled-events endpoint.

        Args:
            date: Date to fetch events for (UTC).

        Returns:
            List of event dicts with keys: event_id, home_team, away_team,
            kickoff_utc, league_name.
        """
        if self.use_mock:
            return self._get_mock_scheduled_events(date)

        date_str = date.strftime("%Y-%m-%d")
        url = f"{SOFASCORE_API_BASE}/sport/football/scheduled-events/{date_str}"

        data, error = await self._fetch_json(url, f"scheduled_{date_str}")

        if error:
            logger.warning(f"[SOFASCORE] Failed to fetch scheduled events for {date_str}: {error}")
            return []

        events = []
        try:
            # Response structure: {"events": [...]}
            for event in data.get("events", []):
                event_id = event.get("id")
                if not event_id:
                    continue

                home_team_data = event.get("homeTeam", {})
                away_team_data = event.get("awayTeam", {})
                tournament = event.get("tournament", {})

                # Parse kickoff timestamp
                start_timestamp = event.get("startTimestamp")
                kickoff_utc = None
                if start_timestamp:
                    try:
                        kickoff_utc = datetime.utcfromtimestamp(start_timestamp)
                    except (ValueError, TypeError):
                        pass

                events.append({
                    "event_id": str(event_id),
                    "home_team": home_team_data.get("name", ""),
                    "away_team": away_team_data.get("name", ""),
                    "kickoff_utc": kickoff_utc,
                    "league_name": tournament.get("name", ""),
                    "league_slug": tournament.get("slug", ""),
                    "home_team_slug": home_team_data.get("slug", ""),
                    "away_team_slug": away_team_data.get("slug", ""),
                })

        except Exception as e:
            logger.error(f"[SOFASCORE] Error parsing scheduled events for {date_str}: {e}")

        logger.debug(f"[SOFASCORE] Found {len(events)} events for {date_str}")
        return events

    def _get_mock_scheduled_events(self, date: datetime) -> list[dict]:
        """Return mock scheduled events for testing."""
        return [
            {
                "event_id": "12345678",
                "home_team": "Manchester United",
                "away_team": "Liverpool",
                "kickoff_utc": date.replace(hour=15, minute=0, second=0),
                "league_name": "Premier League",
                "league_slug": "premier-league",
                "home_team_slug": "manchester-united",
                "away_team_slug": "liverpool",
            },
            {
                "event_id": "12345679",
                "home_team": "Real Madrid",
                "away_team": "Barcelona",
                "kickoff_utc": date.replace(hour=20, minute=0, second=0),
                "league_name": "LaLiga",
                "league_slug": "laliga",
                "home_team_slug": "real-madrid",
                "away_team_slug": "barcelona",
            },
        ]

    def _get_mock_lineup(
        self,
        event_id: str,
    ) -> SofascoreMatchLineup:
        """Return mock lineup data for testing."""
        import random

        def make_players(team_side: str) -> list[SofascorePlayerData]:
            """Generate 11 mock starters + 7 subs."""
            positions = ["GK"] + ["DEF"] * 4 + ["MID"] * 3 + ["FWD"] * 3
            players = []

            for i, pos in enumerate(positions):
                players.append(SofascorePlayerData(
                    player_id_ext=f"{team_side}_{i+1}",
                    position=pos,
                    is_starter=True,
                    rating_pre_match=round(random.uniform(6.0, 8.5), 2),
                    rating_recent_form=round(random.uniform(6.0, 8.5), 2),
                    name=f"Player {i+1}",
                ))

            # Add subs
            for i in range(7):
                pos = random.choice(["DEF", "MID", "FWD"])
                players.append(SofascorePlayerData(
                    player_id_ext=f"{team_side}_sub_{i+1}",
                    position=pos,
                    is_starter=False,
                    rating_pre_match=round(random.uniform(5.5, 7.5), 2),
                    rating_recent_form=round(random.uniform(5.5, 7.5), 2),
                    name=f"Sub {i+1}",
                ))

            return players

        formations = ["4-3-3", "4-2-3-1", "3-5-2", "4-4-2", "3-4-3"]

        return SofascoreMatchLineup(
            source_event_id=event_id,
            home=SofascoreLineupData(
                team_side="home",
                formation=random.choice(formations),
                players=make_players("home"),
            ),
            away=SofascoreLineupData(
                team_side="away",
                formation=random.choice(formations),
                players=make_players("away"),
            ),
            captured_at=datetime.utcnow(),
            integrity_score=1.0,
        )

    async def close(self) -> None:
        """Close any open connections."""
        if self._client:
            await self._client.aclose()
            self._client = None
        logger.debug("SofascoreProvider closed")


# =============================================================================
# TEAM NAME NORMALIZATION & MATCHING
# =============================================================================

import re
import unicodedata


def normalize_team_name(name: str) -> str:
    """
    Normalize team name for fuzzy matching.

    Steps:
    1. Lowercase
    2. Remove accents/diacritics
    3. Remove common suffixes (FC, CF, SC, etc.)
    4. Remove punctuation
    5. Collapse whitespace

    Examples:
        "Manchester United FC" -> "manchester united"
        "Atlético Madrid" -> "atletico madrid"
        "FC Barcelona" -> "barcelona"
    """
    if not name:
        return ""

    # Lowercase
    name = name.lower().strip()

    # Remove accents
    name = unicodedata.normalize("NFKD", name)
    name = "".join(c for c in name if not unicodedata.combining(c))

    # Remove common prefixes/suffixes
    suffixes = [
        r"\bfc\b", r"\bcf\b", r"\bsc\b", r"\bafc\b", r"\bssc\b",
        r"\bac\b", r"\bas\b", r"\bcd\b", r"\bud\b", r"\brc\b",
        r"\bsv\b", r"\bvfb\b", r"\btsv\b", r"\bfk\b", r"\bsk\b",
        r"\breal\b", r"\bunited\b", r"\bcity\b", r"\bclub\b",
    ]
    for suffix in suffixes:
        name = re.sub(suffix, "", name)

    # Remove punctuation
    name = re.sub(r"[^\w\s]", "", name)

    # Collapse whitespace
    name = " ".join(name.split())

    return name


def calculate_team_similarity(name1: str, name2: str) -> float:
    """
    Calculate similarity score between two team names.

    Uses normalized Levenshtein-like approach:
    - Exact match after normalization: 1.0
    - One contains the other: 0.85
    - Token overlap (Jaccard): 0.0-0.8

    Returns:
        Score from 0.0 to 1.0.
    """
    n1 = normalize_team_name(name1)
    n2 = normalize_team_name(name2)

    if not n1 or not n2:
        return 0.0

    # Exact match
    if n1 == n2:
        return 1.0

    # One contains the other
    if n1 in n2 or n2 in n1:
        return 0.85

    # Token-based Jaccard similarity
    tokens1 = set(n1.split())
    tokens2 = set(n2.split())

    if not tokens1 or not tokens2:
        return 0.0

    intersection = tokens1 & tokens2
    union = tokens1 | tokens2

    jaccard = len(intersection) / len(union)
    return jaccard * 0.8  # Max 0.8 for token overlap


def calculate_match_score(
    our_home: str,
    our_away: str,
    our_kickoff: datetime,
    sf_home: str,
    sf_away: str,
    sf_kickoff: Optional[datetime],
    kickoff_tolerance_hours: float = 2.0,
) -> tuple[float, str]:
    """
    Calculate matching score between our match and Sofascore event.

    Scoring:
    - Kickoff time (within tolerance): 0.3 weight
    - Home team name match: 0.35 weight
    - Away team name match: 0.35 weight

    Args:
        our_*: Our match data.
        sf_*: Sofascore event data.
        kickoff_tolerance_hours: Max hours difference for kickoff match.

    Returns:
        Tuple of (score, matched_by_description).
    """
    score = 0.0
    matched_by_parts = []

    # Kickoff time matching (0.3 weight)
    if sf_kickoff:
        time_diff_hours = abs((our_kickoff - sf_kickoff).total_seconds()) / 3600
        if time_diff_hours <= kickoff_tolerance_hours:
            # Linear decay: full score at 0h, zero at tolerance
            kickoff_score = 1.0 - (time_diff_hours / kickoff_tolerance_hours)
            score += 0.3 * kickoff_score
            if kickoff_score >= 0.5:
                matched_by_parts.append(f"kickoff(±{time_diff_hours:.1f}h)")

    # Home team matching (0.35 weight)
    home_sim = calculate_team_similarity(our_home, sf_home)
    score += 0.35 * home_sim
    if home_sim >= 0.7:
        matched_by_parts.append(f"home({home_sim:.2f})")

    # Away team matching (0.35 weight)
    away_sim = calculate_team_similarity(our_away, sf_away)
    score += 0.35 * away_sim
    if away_sim >= 0.7:
        matched_by_parts.append(f"away({away_sim:.2f})")

    matched_by = "+".join(matched_by_parts) if matched_by_parts else "low_confidence"

    return round(score, 3), matched_by
