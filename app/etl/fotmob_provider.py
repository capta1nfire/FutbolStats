"""
FotMob Provider for team-level xG data (Opta source).

ABE P0 2026-02-08: Scheduler-only, 1.5 req/s, circuit breaker, parse by key name.
Used for leagues WITHOUT Understat coverage (LATAM, secondary EUR, etc.).
Fixed source per league: Understat=top-5 EUR, FotMob=rest.

API Endpoints:
- https://www.fotmob.com/api/matchDetails?matchId={id}  — match xG stats
- https://www.fotmob.com/api/leagues?id={id}             — league fixtures

Rate limiting: 1.5s between requests with circuit breaker (5 consecutive errors).
Anti-scraping: x-fm-req header (MD5+Base64 signature).
"""

import asyncio
import base64
import hashlib
import json
import logging
import os
import random
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional
from urllib.parse import urlparse, urlunparse

import httpx

logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

FOTMOB_API_BASE = "https://www.fotmob.com/api"

# Rate limiting
MIN_REQUEST_INTERVAL = 1.5  # P0-5: slightly more conservative than SofaScore
MAX_RETRIES = 3
RETRY_DELAY_BASE = 3.0
JITTER_MAX = 1.5

# Circuit breaker (P0-5)
DEFAULT_CIRCUIT_BREAKER_THRESHOLD = 5

# HTTP headers (browser-like)
DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/121.0.0.0 Safari/537.36",
    "Accept": "*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.fotmob.com/",
    "Origin": "https://www.fotmob.com",
}


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class FotmobXGData:
    """Team-level xG data from FotMob (P0-6: no shotmap, no player xG)."""
    xg_home: Optional[float] = None
    xg_away: Optional[float] = None
    xgot_home: Optional[float] = None
    xgot_away: Optional[float] = None
    xg_open_play_home: Optional[float] = None
    xg_open_play_away: Optional[float] = None
    xg_set_play_home: Optional[float] = None
    xg_set_play_away: Optional[float] = None
    raw_stats: Optional[dict] = None


@dataclass
class FotmobFixture:
    """A fixture from FotMob league endpoint, used for match linking."""
    fotmob_id: int
    home_team: str
    away_team: str
    kickoff_utc: Optional[datetime] = None
    status: Optional[str] = None  # "finished", "notstarted", etc.
    home_score: Optional[int] = None
    away_score: Optional[int] = None


# =============================================================================
# PROVIDER CLASS
# =============================================================================

class FotmobProvider:
    """
    Provider for FotMob xG data (team-level).

    ABE P0: scheduler-only, 1.5 req/s, circuit breaker, parse by key name.
    Follows SofascoreProvider architecture for resilience and proxy support.
    """

    SCHEMA_VERSION = "fotmob.xg.v1"

    def __init__(self):
        self._clients: dict[str, httpx.AsyncClient] = {}
        self._last_request_time: float = 0
        self._consecutive_errors: int = 0
        self._proxy_url: Optional[str] = (
            os.environ.get("FOTMOB_PROXY_URL")
            or os.environ.get("SOFASCORE_PROXY_URL")
        )

        # Load thresholds from config (with defaults)
        try:
            from app.config import get_settings
            settings = get_settings()
            self._rate_limit_seconds = settings.FOTMOB_RATE_LIMIT_SECONDS
            self._circuit_breaker_threshold = settings.FOTMOB_CIRCUIT_BREAKER_THRESHOLD
        except Exception:
            self._rate_limit_seconds = MIN_REQUEST_INTERVAL
            self._circuit_breaker_threshold = DEFAULT_CIRCUIT_BREAKER_THRESHOLD

        if self._proxy_url:
            logger.info("[FOTMOB] Provider initialized with proxy")
        else:
            logger.info("[FOTMOB] Provider initialized without proxy")

    # -----------------------------------------------------------------
    # HTTP infrastructure (follows SofascoreProvider patterns)
    # -----------------------------------------------------------------

    def _build_geo_proxy_url(self, country_code: str) -> str:
        """Build proxy URL with IPRoyal geo-targeting suffix."""
        parsed = urlparse(self._proxy_url)
        password = parsed.password or ""
        if "_country-" in password:
            password = password[: password.index("_country-")]
        geo_password = f"{password}_country-{country_code}"
        netloc = f"{parsed.username}:{geo_password}@{parsed.hostname}"
        if parsed.port:
            netloc += f":{parsed.port}"
        return urlunparse((parsed.scheme, netloc, parsed.path, "", "", ""))

    async def _get_client(self, country_code: Optional[str] = None) -> httpx.AsyncClient:
        """Get or create HTTP client, optionally geo-targeted."""
        cache_key = country_code or "_base"
        if cache_key not in self._clients:
            kwargs: dict[str, Any] = {
                "timeout": 30.0,
                "headers": {**DEFAULT_HEADERS, **self._build_headers()},
                "follow_redirects": True,
            }
            if self._proxy_url:
                if country_code:
                    kwargs["proxy"] = self._build_geo_proxy_url(country_code)
                    logger.info("[FOTMOB] Creating geo-proxy client for country=%s", country_code)
                else:
                    kwargs["proxy"] = self._proxy_url
            self._clients[cache_key] = httpx.AsyncClient(**kwargs)
        return self._clients[cache_key]

    @staticmethod
    def _build_headers() -> dict[str, str]:
        """
        Build FotMob-specific headers including x-fm-req signature.

        The x-fm-req header is an anti-scraping mechanism. It's computed as
        Base64(MD5(path + salt)) where salt is a known constant derived from
        the FotMob JS bundle (solved by soccerdata PR#745).
        """
        # The FotMob anti-scraping token — static salt from their JS bundle
        # This generates a valid x-fm-req for any request
        return {}  # Headers set per-request in _fetch_json

    @staticmethod
    def _compute_xfm_req(path: str) -> str:
        """
        Compute x-fm-req header value for a given API path.

        Algorithm (from soccerdata PR#745):
        1. Take the API path (e.g., "/api/matchDetails?matchId=4336691")
        2. Concatenate with known salt
        3. MD5 hash → Base64 encode
        """
        # Salt extracted from FotMob's JS bundle (public, rotated infrequently)
        salt = "d41d8cd98f00b204e9800998ecf8427e"
        raw = path + salt
        md5_hash = hashlib.md5(raw.encode()).digest()
        return base64.b64encode(md5_hash).decode()

    async def _rate_limit(self) -> None:
        """Enforce rate limiting between requests."""
        now = time.time()
        elapsed = now - self._last_request_time
        if elapsed < self._rate_limit_seconds:
            await asyncio.sleep(self._rate_limit_seconds - elapsed)
        self._last_request_time = time.time()

    async def _fetch_json(
        self,
        url: str,
        identifier: str,
        country_code: Optional[str] = None,
    ) -> tuple[Optional[dict], Optional[str]]:
        """
        Fetch JSON with retry logic + circuit breaker (P0-5).

        Returns:
            Tuple of (data, error_message).
        """
        # Circuit breaker check
        if self._consecutive_errors >= self._circuit_breaker_threshold:
            logger.warning("[FOTMOB] Circuit breaker OPEN (%d errors), skipping %s",
                           self._consecutive_errors, identifier)
            return None, "circuit_breaker_open"

        for attempt in range(MAX_RETRIES):
            try:
                await self._rate_limit()
                client = await self._get_client(country_code)

                # Compute x-fm-req for this specific URL path
                from urllib.parse import urlparse as _urlparse
                parsed_url = _urlparse(url)
                path_with_query = parsed_url.path
                if parsed_url.query:
                    path_with_query += "?" + parsed_url.query
                xfm_header = self._compute_xfm_req(path_with_query)

                response = await client.get(
                    url,
                    headers={"x-fm-req": xfm_header},
                )

                # Soft fails (retryable)
                if response.status_code == 429:
                    delay = RETRY_DELAY_BASE * (2 ** attempt) + random.uniform(0, JITTER_MAX)
                    logger.warning("[FOTMOB] Rate limited (429) for %s, waiting %.1fs", identifier, delay)
                    await asyncio.sleep(delay)
                    continue

                if response.status_code >= 500:
                    delay = RETRY_DELAY_BASE * (2 ** attempt) + random.uniform(0, JITTER_MAX)
                    logger.warning("[FOTMOB] Server error %d for %s, retrying in %.1fs",
                                   response.status_code, identifier, delay)
                    await asyncio.sleep(delay)
                    continue

                # Hard fails
                if response.status_code == 404:
                    logger.debug("[FOTMOB] Not found (404) for %s", identifier)
                    self._consecutive_errors += 1
                    return None, "not_found"

                if response.status_code == 403:
                    logger.warning("[FOTMOB] Access denied (403) for %s", identifier)
                    self._consecutive_errors += 1
                    return None, "access_denied"

                response.raise_for_status()

                try:
                    data = response.json()
                    self._consecutive_errors = 0  # Reset on success
                    return data, None
                except Exception as e:
                    logger.error("[FOTMOB] JSON parse error for %s: %s", identifier, e)
                    self._consecutive_errors += 1
                    return None, "json_parse_error"

            except httpx.TimeoutException:
                delay = RETRY_DELAY_BASE * (2 ** attempt) + random.uniform(0, JITTER_MAX)
                logger.warning("[FOTMOB] Timeout for %s, attempt %d, waiting %.1fs",
                               identifier, attempt + 1, delay)
                await asyncio.sleep(delay)
                continue

            except httpx.RequestError as e:
                delay = RETRY_DELAY_BASE * (2 ** attempt) + random.uniform(0, JITTER_MAX)
                logger.warning("[FOTMOB] Request error for %s: %s, waiting %.1fs",
                               identifier, e, delay)
                await asyncio.sleep(delay)
                continue

            except Exception as e:
                logger.error("[FOTMOB] Unexpected error for %s: %s", identifier, e)
                self._consecutive_errors += 1
                return None, f"unexpected_error: {str(e)[:100]}"

        self._consecutive_errors += 1
        logger.error("[FOTMOB] Failed to fetch %s after %d attempts", identifier, MAX_RETRIES)
        return None, "max_retries_exceeded"

    # -----------------------------------------------------------------
    # xG Data (P0-3: parse by key name, NEVER by index)
    # -----------------------------------------------------------------

    async def get_match_xg(
        self,
        fotmob_match_id: int,
        country_code: Optional[str] = None,
    ) -> tuple[Optional[FotmobXGData], Optional[str]]:
        """
        Fetch and parse team-level xG for a match.

        P0-3: Stats are parsed by key name (title field), never by array index.
        P0-6: Only team-level xG/xGOT, no shotmap or player data.

        Args:
            fotmob_match_id: FotMob match ID.
            country_code: ISO 2-letter code for geo-proxy routing.

        Returns:
            Tuple of (FotmobXGData, error_message).
        """
        url = f"{FOTMOB_API_BASE}/matchDetails?matchId={fotmob_match_id}"
        data, error = await self._fetch_json(url, f"match_{fotmob_match_id}", country_code)

        if error:
            return None, error

        try:
            xg_data = self._parse_xg_stats(data)
            if xg_data is None:
                return None, "no_xg_data"
            return xg_data, None
        except Exception as e:
            logger.error("[FOTMOB] Schema break parsing xG for match %d: %s", fotmob_match_id, e)
            return None, f"schema_break: {str(e)[:100]}"

    def _parse_xg_stats(self, data: dict) -> Optional[FotmobXGData]:
        """
        Parse xG from matchDetails response.

        P0-3: NEVER parse by index, always find by title/key name.

        FotMob response structure (simplified):
        {
          "content": {
            "stats": {
              "Ede": {
                "stats": [
                  {
                    "title": "...",
                    "stats": [
                      {
                        "title": "Expected goals (xG)",
                        "stats": ["1.35", "0.87"]  // [home, away]
                      },
                      ...
                    ]
                  }
                ]
              }
            }
          }
        }
        """
        content = data.get("content", {})
        stats_section = content.get("stats", {})

        # Try multiple possible keys for the stats container
        stats_container = None
        for key in ("Ede", "stats"):
            if key in stats_section:
                stats_container = stats_section[key]
                break

        if not stats_container:
            # Some responses nest differently
            if isinstance(stats_section, dict) and "stats" not in stats_section:
                # Try treating stats_section itself as the container
                stats_container = stats_section
            else:
                return None

        all_groups = stats_container.get("stats", [])
        if not all_groups:
            return None

        # Build a flat map: normalized_title -> (home_value, away_value)
        xg_map: dict[str, tuple[Optional[str], Optional[str]]] = {}

        for group in all_groups:
            stats_items = group.get("stats", [])
            for stat in stats_items:
                title = (stat.get("title") or stat.get("key") or "").strip().lower()
                values = stat.get("stats", [])
                if len(values) >= 2:
                    xg_map[title] = (str(values[0]), str(values[1]))

        # Normalize synonyms
        _SYNONYMS = {
            "expected goals (xg)": "expected goals",
            "xg": "expected goals",
            "expected goals on target (xgot)": "expected goals on target",
            "xgot": "expected goals on target",
        }
        for alias, canonical in _SYNONYMS.items():
            if alias in xg_map and canonical not in xg_map:
                xg_map[canonical] = xg_map[alias]

        # Extract values
        xg_h, xg_a = xg_map.get("expected goals", (None, None))
        xgot_h, xgot_a = xg_map.get("expected goals on target", (None, None))
        xg_op_h, xg_op_a = xg_map.get("xg open play", (None, None))
        xg_sp_h, xg_sp_a = xg_map.get("xg set play", (None, None))

        # If no xG found at all, return None
        if xg_h is None and xg_a is None:
            return None

        return FotmobXGData(
            xg_home=_safe_float(xg_h),
            xg_away=_safe_float(xg_a),
            xgot_home=_safe_float(xgot_h),
            xgot_away=_safe_float(xgot_a),
            xg_open_play_home=_safe_float(xg_op_h),
            xg_open_play_away=_safe_float(xg_op_a),
            xg_set_play_home=_safe_float(xg_sp_h),
            xg_set_play_away=_safe_float(xg_sp_a),
            raw_stats=data.get("content", {}).get("stats"),
        )

    # -----------------------------------------------------------------
    # League fixtures (for match linking)
    # -----------------------------------------------------------------

    async def get_league_fixtures(
        self,
        fotmob_league_id: int,
        country_code: Optional[str] = None,
    ) -> tuple[list[FotmobFixture], Optional[str]]:
        """
        Fetch fixtures from FotMob league endpoint for match linking.

        Used by sync_fotmob_refs() to link internal matches to FotMob IDs.
        Returns finished matches for linking against our DB.

        Args:
            fotmob_league_id: FotMob league ID (e.g. 112 for Argentina).
            country_code: ISO 2-letter code for geo-proxy routing.

        Returns:
            Tuple of (list of FotmobFixture, error_message).
        """
        url = f"{FOTMOB_API_BASE}/leagues?id={fotmob_league_id}"
        data, error = await self._fetch_json(url, f"league_{fotmob_league_id}", country_code)

        if error:
            return [], error

        try:
            return self._parse_league_fixtures(data), None
        except Exception as e:
            logger.error("[FOTMOB] Schema break parsing fixtures for league %d: %s",
                         fotmob_league_id, e)
            return [], f"schema_break: {str(e)[:100]}"

    def _parse_league_fixtures(self, data: dict) -> list[FotmobFixture]:
        """Parse fixtures from FotMob league response."""
        fixtures = []

        # FotMob league response: matches in "matches" or "overview" section
        matches_section = data.get("matches", {})
        all_matches = matches_section.get("allMatches", [])

        if not all_matches:
            # Try alternative structure
            overview = data.get("overview", {})
            all_matches = overview.get("matches", [])

        for match in all_matches:
            try:
                match_id = match.get("id")
                if not match_id:
                    continue

                home_data = match.get("home", {})
                away_data = match.get("away", {})

                # Parse kickoff
                kickoff_utc = None
                utc_time = match.get("utcTime")
                if utc_time:
                    try:
                        # FotMob uses ISO 8601 format
                        kickoff_utc = datetime.fromisoformat(
                            utc_time.replace("Z", "+00:00")
                        ).replace(tzinfo=None)
                    except (ValueError, TypeError):
                        pass

                # Parse status
                status_data = match.get("status", {})
                status_code = None
                if isinstance(status_data, dict):
                    # finished, notstarted, ongoing, etc.
                    finished = status_data.get("finished", False)
                    started = status_data.get("started", False)
                    if finished:
                        status_code = "finished"
                    elif started:
                        status_code = "ongoing"
                    else:
                        status_code = "notstarted"
                elif isinstance(status_data, str):
                    status_code = status_data.lower()

                fixtures.append(FotmobFixture(
                    fotmob_id=int(match_id),
                    home_team=home_data.get("name", ""),
                    away_team=away_data.get("name", ""),
                    kickoff_utc=kickoff_utc,
                    status=status_code,
                    home_score=_safe_int(home_data.get("score")),
                    away_score=_safe_int(away_data.get("score")),
                ))
            except Exception as e:
                logger.debug("[FOTMOB] Skipping fixture parse error: %s", e)
                continue

        logger.debug("[FOTMOB] Parsed %d fixtures from league response", len(fixtures))
        return fixtures

    # -----------------------------------------------------------------
    # Lifecycle
    # -----------------------------------------------------------------

    async def close(self) -> None:
        """Close all open HTTP clients."""
        for key, client in self._clients.items():
            await client.aclose()
        n = len(self._clients)
        self._clients.clear()
        logger.debug("[FOTMOB] Provider closed (%d clients)", n)

    @property
    def is_circuit_open(self) -> bool:
        """Check if circuit breaker is currently open."""
        return self._consecutive_errors >= self._circuit_breaker_threshold

    def reset_circuit_breaker(self) -> None:
        """Manually reset the circuit breaker (e.g. at start of new job run)."""
        self._consecutive_errors = 0


# =============================================================================
# HELPERS
# =============================================================================

def _safe_float(val: Any) -> Optional[float]:
    """Safely convert a value to float, returning None on failure."""
    if val is None:
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def _safe_int(val: Any) -> Optional[int]:
    """Safely convert a value to int, returning None on failure."""
    if val is None:
        return None
    try:
        return int(val)
    except (ValueError, TypeError):
        return None
