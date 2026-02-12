from __future__ import annotations
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
import os
import random
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional
from urllib.parse import urlparse, urlunparse

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
        self._clients: dict[str, httpx.AsyncClient] = {}  # keyed by country code ("_base" for no geo)
        self._last_request_time: float = 0
        self._proxy_url: Optional[str] = os.environ.get("SOFASCORE_PROXY_URL")
        if self._proxy_url:
            logger.info("SofascoreProvider initialized with proxy (mock=%s)", use_mock)
        else:
            logger.info("SofascoreProvider initialized without proxy (mock=%s)", use_mock)

    def _build_geo_proxy_url(self, country_code: str) -> str:
        """Build proxy URL with IPRoyal geo-targeting suffix on the password.

        Format: http://USER:PASS_country-CC@HOST:PORT
        """
        parsed = urlparse(self._proxy_url)
        password = parsed.password or ""
        # Strip any existing _country-XX suffix to avoid duplication
        if "_country-" in password:
            password = password[: password.index("_country-")]
        geo_password = f"{password}_country-{country_code}"
        netloc = f"{parsed.username}:{geo_password}@{parsed.hostname}"
        if parsed.port:
            netloc += f":{parsed.port}"
        return urlunparse((parsed.scheme, netloc, parsed.path, "", "", ""))

    async def _get_client(self, country_code: Optional[str] = None) -> httpx.AsyncClient:
        """Get or create HTTP client, optionally geo-targeted by country."""
        cache_key = country_code or "_base"
        if cache_key not in self._clients:
            kwargs: dict[str, Any] = {
                "timeout": 30.0,
                "headers": DEFAULT_HEADERS,
                "follow_redirects": True,
            }
            if self._proxy_url:
                if country_code:
                    kwargs["proxy"] = self._build_geo_proxy_url(country_code)
                    logger.info("[SOFASCORE] Creating geo-proxy client for country=%s", country_code)
                else:
                    kwargs["proxy"] = self._proxy_url
            self._clients[cache_key] = httpx.AsyncClient(**kwargs)
        return self._clients[cache_key]

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
        country_code: Optional[str] = None,
    ) -> tuple[Optional[dict], Optional[str]]:
        """
        Fetch JSON with retry logic.

        Args:
            country_code: ISO 2-letter code for geo-proxy routing (e.g. "es", "gb").

        Returns:
            Tuple of (data, error_message).
            On success: (dict, None)
            On failure: (None, error_string)
        """
        for attempt in range(MAX_RETRIES):
            try:
                await self._rate_limit()
                client = await self._get_client(country_code)

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
        country_code: Optional[str] = None,
    ) -> SofascoreMatchLineup:
        """
        Get lineup/formation/ratings for a match.

        Args:
            sofascore_event_id: Sofascore event ID.
            country_code: ISO 2-letter code for geo-proxy routing.

        Returns:
            SofascoreMatchLineup with home/away lineup data.
            On failure, returns result with error field set.
        """
        if self.use_mock:
            return self._get_mock_lineup(sofascore_event_id)

        url = f"{SOFASCORE_API_BASE}/event/{sofascore_event_id}/lineups"

        data, error = await self._fetch_json(url, sofascore_event_id, country_code)

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

    # =================================================================
    # POST-MATCH STATISTICS
    # =================================================================

    async def get_match_statistics(
        self,
        sofascore_event_id: str,
        country_code: Optional[str] = None,
    ) -> tuple[Optional[dict], Optional[str]]:
        """
        Get post-match statistics for a finished match.

        Endpoint: /api/v1/event/{event_id}/statistics

        Args:
            country_code: ISO 2-letter code for geo-proxy routing.

        Returns:
            Tuple of (parsed_stats_dict, error_message).
            parsed_stats_dict keys: possession_home, xg_home, big_chances_home, etc.
            Plus raw_stats with the full response for future-proofing.
        """
        if self.use_mock:
            return self._get_mock_statistics(), None

        url = f"{SOFASCORE_API_BASE}/event/{sofascore_event_id}/statistics"
        data, error = await self._fetch_json(url, sofascore_event_id, country_code)

        if error:
            return None, error

        try:
            return self._parse_statistics_response(data), None
        except Exception as e:
            logger.error(f"[SOFASCORE] Schema break parsing stats for {sofascore_event_id}: {e}")
            return None, f"schema_break: {str(e)[:100]}"

    def _parse_statistics_response(self, data: dict) -> dict:
        """
        Parse Sofascore /event/{id}/statistics response.

        Response structure:
        {
          "statistics": [
            {
              "period": "ALL",
              "groups": [
                {
                  "groupName": "Possession",
                  "statisticsItems": [
                    {"name": "Ball possession", "home": "55%", "away": "45%", ...}
                  ]
                },
                {
                  "groupName": "Shots",
                  "statisticsItems": [
                    {"name": "Total shots", "home": "12", "away": "8", ...},
                    {"name": "Shots on target", "home": "5", "away": "3", ...}
                  ]
                },
                ...
              ]
            }
          ]
        }

        Mapping by stat name (robust: not positional).
        """
        result = {
            "raw_stats": data,
        }

        # Find the "ALL" period (full match stats)
        all_period = None
        for period_block in data.get("statistics", []):
            if period_block.get("period") == "ALL":
                all_period = period_block
                break

        if not all_period:
            # Fallback: use first period if "ALL" not found
            periods = data.get("statistics", [])
            if periods:
                all_period = periods[0]

        if not all_period:
            return result

        # Build flat lookup: stat_name_lower -> (home_val, away_val)
        stat_lookup: dict[str, tuple[str, str]] = {}
        for group in all_period.get("groups", []):
            for item in group.get("statisticsItems", []):
                name = (item.get("name") or "").strip().lower()
                home_val = str(item.get("home", ""))
                away_val = str(item.get("away", ""))
                stat_lookup[name] = (home_val, away_val)

        # Normalize known synonyms to canonical names
        _SYNONYMS = {
            "expected goals (xg)": "expected goals",
            "xg": "expected goals",
            "corners": "corner kicks",
            "big chances created": "big chances",
        }
        for alias, canonical in _SYNONYMS.items():
            if alias in stat_lookup and canonical not in stat_lookup:
                stat_lookup[canonical] = stat_lookup[alias]

        # Map stat names to our columns
        STAT_MAP = {
            "ball possession": ("possession_home", "possession_away", "pct"),
            "total shots": ("total_shots_home", "total_shots_away", "int"),
            "shots on target": ("shots_on_target_home", "shots_on_target_away", "int"),
            "expected goals": ("xg_home", "xg_away", "float"),
            "corner kicks": ("corners_home", "corners_away", "int"),
            "fouls": ("fouls_home", "fouls_away", "int"),
            "big chances": ("big_chances_home", "big_chances_away", "int"),
            "big chances missed": ("big_chances_missed_home", "big_chances_missed_away", "int"),
            "accurate passes": ("accurate_passes_home", "accurate_passes_away", "int"),
        }

        for stat_name, (col_home, col_away, dtype) in STAT_MAP.items():
            if stat_name not in stat_lookup:
                continue
            home_raw, away_raw = stat_lookup[stat_name]
            result[col_home] = self._parse_stat_value(home_raw, dtype)
            result[col_away] = self._parse_stat_value(away_raw, dtype)

        # Pass accuracy (special: sometimes "80%" or "354/443 (80%)")
        if "passes accurate" in stat_lookup:
            # Some responses use "Passes accurate" with value like "354/443 (80%)"
            home_raw, away_raw = stat_lookup["passes accurate"]
            result["accurate_passes_home"] = self._parse_stat_value(home_raw.split("/")[0].strip(), "int")
            result["accurate_passes_away"] = self._parse_stat_value(away_raw.split("/")[0].strip(), "int")

        # Pass accuracy percentage
        if "pass accuracy" in stat_lookup:
            home_raw, away_raw = stat_lookup["pass accuracy"]
            result["pass_accuracy_home"] = self._parse_stat_value(home_raw, "pct")
            result["pass_accuracy_away"] = self._parse_stat_value(away_raw, "pct")

        return result

    @staticmethod
    def _parse_stat_value(raw: str, dtype: str) -> Optional[int | float]:
        """Parse a stat value string to typed value."""
        if not raw or raw == "-":
            return None
        # Strip percentage sign
        cleaned = raw.replace("%", "").strip()
        try:
            if dtype == "float":
                return float(cleaned)
            elif dtype == "pct":
                return int(float(cleaned))
            else:  # "int"
                return int(float(cleaned))
        except (ValueError, TypeError):
            return None

    def _get_mock_statistics(self) -> dict:
        """Return mock statistics for testing."""
        return {
            "possession_home": 55,
            "possession_away": 45,
            "total_shots_home": 12,
            "total_shots_away": 8,
            "shots_on_target_home": 5,
            "shots_on_target_away": 3,
            "xg_home": 1.85,
            "xg_away": 0.92,
            "corners_home": 6,
            "corners_away": 3,
            "fouls_home": 14,
            "fouls_away": 11,
            "big_chances_home": 3,
            "big_chances_away": 1,
            "big_chances_missed_home": 1,
            "big_chances_missed_away": 0,
            "accurate_passes_home": 354,
            "accurate_passes_away": 280,
            "pass_accuracy_home": 80,
            "pass_accuracy_away": 72,
            "raw_stats": {},
        }

    async def close(self) -> None:
        """Close all open connections (including geo-proxy clients)."""
        for key, client in self._clients.items():
            await client.aclose()
        n = len(self._clients)
        self._clients.clear()
        logger.debug("SofascoreProvider closed (%d clients)", n)


# =============================================================================
# TEAM NAME NORMALIZATION & MATCHING
# =============================================================================

from app.etl.name_normalization import normalize_team_name  # noqa: F401 — used by calculate_team_similarity + re-exported


def calculate_team_similarity(
    name1: str,
    name2: str,
    alias_index: Optional[dict] = None,
) -> float:
    """
    Calculate similarity score between two team names.

    Scoring tiers:
    - Exact match after normalization: 1.0
    - Alias match (via cross-provider index): 0.95
    - One contains the other: 0.85
    - Token overlap (Jaccard): 0.0-0.8

    Args:
        name1: First team name (e.g. from API-Football).
        name2: Second team name (e.g. from Sofascore).
        alias_index: Optional alias index from build_alias_index().
            If provided, enables cross-provider alias matching.

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

    # Alias match (cross-provider: e.g. "Man City" ↔ "Manchester City")
    if alias_index is not None:
        if n1 in alias_index.get(n2, set()) or n2 in alias_index.get(n1, set()):
            return 0.95

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
    alias_index: Optional[dict] = None,
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
        alias_index: Optional cross-provider alias index for improved name matching.

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
    home_sim = calculate_team_similarity(our_home, sf_home, alias_index)
    score += 0.35 * home_sim
    if home_sim >= 0.7:
        matched_by_parts.append(f"home({home_sim:.2f})")

    # Away team matching (0.35 weight)
    away_sim = calculate_team_similarity(our_away, sf_away, alias_index)
    score += 0.35 * away_sim
    if away_sim >= 0.7:
        matched_by_parts.append(f"away({away_sim:.2f})")

    matched_by = "+".join(matched_by_parts) if matched_by_parts else "low_confidence"

    return round(score, 3), matched_by


def get_sofascore_threshold(league_id: int) -> float:
    """Get match score threshold for a league (env-overridable via config)."""
    from app.config import get_settings
    settings = get_settings()
    overrides = _parse_threshold_overrides(settings.SOFASCORE_REFS_THRESHOLD_OVERRIDES)
    return overrides.get(league_id, settings.SOFASCORE_REFS_THRESHOLD)


def _parse_threshold_overrides(raw: str) -> dict[int, float]:
    """Parse '128:0.70,307:0.70' -> {128: 0.70, 307: 0.70}."""
    if not raw:
        return {}
    result = {}
    for pair in raw.split(","):
        pair = pair.strip()
        if ":" not in pair:
            continue
        k, v = pair.split(":", 1)
        try:
            result[int(k.strip())] = float(v.strip())
        except (ValueError, TypeError):
            continue
    return result
