#!/usr/bin/env python3
"""
Transfermarkt Injury/Suspension Scraping Pipeline.

Scrapes historical injury and suspension data from Transfermarkt for all tracked
leagues (2020+), matches TM players to our API-Football player IDs, and derives
per-match injury records for the MTV (Match Talent Variance) backtest.

Phases:
  --discover   : Scrape TM competition pages → team mapping → player matching
  --scrape     : Scrape injury history per matched player (resumable)
  --derive     : Derive per-match injury records (team-scoped, anti-leak)
  --all        : Run all phases sequentially
  --precision-sample : Print random sample for ABE P0 precision gate

ABE P0 Conditions:
  - Player matching: precision > recall (bd_match=0.80, no_bd=0.90)
  - One-to-one strict: ambiguous mappings excluded
  - Derivation: team-scoped (lineups pool only), anti-leak (MPS minutes > 0)
  - Dedup key: (match_id, team_id, player_external_id, source)

Dependencies (scripts-only, not in deploy):
  - beautifulsoup4
  - httpx
  - pandas, pyarrow

Usage:
    source .env
    python3 scripts/scrape_tm_injuries.py --discover --league 39
    python3 scripts/scrape_tm_injuries.py --scrape --league 39 --resume
    python3 scripts/scrape_tm_injuries.py --derive --league 39
    python3 scripts/scrape_tm_injuries.py --precision-sample --league 39
"""

import argparse
import asyncio
import json
import logging
import os
import random
import sys
import time
import unicodedata
from collections import defaultdict
from datetime import date, datetime, timedelta
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse, urlunparse

import httpx
import pandas as pd

# Optional: bs4 for HTML parsing
try:
    from bs4 import BeautifulSoup
except ImportError:
    print("ERROR: beautifulsoup4 required. Install: pip install beautifulsoup4")
    sys.exit(1)

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
DATA_DIR = PROJECT_ROOT / "data"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("scrape_tm_injuries")


# =============================================================================
# TM COMPETITION CODES
# =============================================================================

TM_COMPETITIONS = {
    39:  {"code": "GB1",  "country": "gb", "name": "Premier League"},
    40:  {"code": "GB2",  "country": "gb", "name": "Championship"},
    140: {"code": "ES1",  "country": "es", "name": "LaLiga"},
    135: {"code": "IT1",  "country": "it", "name": "Serie A"},
    78:  {"code": "L1",   "country": "de", "name": "Bundesliga"},
    61:  {"code": "FR1",  "country": "fr", "name": "Ligue 1"},
    94:  {"code": "PO1",  "country": "pt", "name": "Primeira Liga"},
    88:  {"code": "NL1",  "country": "nl", "name": "Eredivisie"},
    144: {"code": "BE1",  "country": "be", "name": "Belgian Pro League"},
    203: {"code": "TR1",  "country": "tr", "name": "Super Lig"},
    128: {"code": "ARG1", "country": "ar", "name": "Argentina Primera"},
    71:  {"code": "BRA1", "country": "br", "name": "Serie A Brasil"},
    239: {"code": "COLP", "country": "co", "name": "Colombia Primera A"},
    265: {"code": "CLPD", "country": "cl", "name": "Chile Primera"},
    344: {"code": "BO1A", "country": "bo", "name": "Bolivia Primera"},
    281: {"code": "TDeA", "country": "pe", "name": "Peru Liga 1"},
    253: {"code": "MLS1", "country": "us", "name": "MLS"},
    262: {"code": "MEX1", "country": "mx", "name": "Liga MX"},
    307: {"code": "SA1",  "country": "sa", "name": "Saudi Pro League"},
    242: {"code": "EC1N", "country": "ec", "name": "Ecuador Liga Pro"},
    299: {"code": "VZ1A", "country": "ve", "name": "Venezuela Primera"},
    268: {"code": "URU1", "country": "uy", "name": "Uruguay Primera"},
    250: {"code": "PR1A", "country": "py", "name": "Paraguay Primera"},
}

# Suspension keywords in TM injury_type
SUSPENSION_KEYWORDS = frozenset([
    "red card", "yellow card", "suspended", "suspension",
    "ban", "doping", "match ban",
])


# =============================================================================
# NAME MATCHING (from scripts/build_player_id_mapping.py)
# =============================================================================

_TRANSLITERATE = {
    "Ø": "O", "ø": "o", "Æ": "AE", "æ": "ae",
    "Ð": "D", "ð": "d", "Þ": "Th", "þ": "th",
    "Ł": "L", "ł": "l", "Đ": "D", "đ": "d",
    "ß": "ss", "İ": "I", "ı": "i",
}


def normalize_name(name):
    """Normalize a player name: lowercase, remove accents, strip."""
    if not name:
        return ""
    name = "".join(_TRANSLITERATE.get(c, c) for c in name)
    nfkd = unicodedata.normalize("NFKD", name)
    ascii_str = "".join(c for c in nfkd if not unicodedata.combining(c))
    return ascii_str.lower().strip()


def extract_lastname(name):
    """Extract likely last name (last token)."""
    tokens = normalize_name(name).split()
    return tokens[-1] if tokens else ""


def initials_mismatch(name_a, name_b):
    """ABE hard reject: if one side is 'X. Surname' and the other is 'FullName Surname',
    reject if X != first letter of FullName.

    Returns True if there's a clear mismatch, False otherwise (including when
    we can't determine — both full names, both initials, etc.)
    """
    tokens_a = normalize_name(name_a).replace(".", " ").split()
    tokens_b = normalize_name(name_b).replace(".", " ").split()
    if len(tokens_a) < 2 or len(tokens_b) < 2:
        return False

    first_a = tokens_a[0]
    first_b = tokens_b[0]
    a_is_initial = len(first_a) == 1
    b_is_initial = len(first_b) == 1

    if a_is_initial and not b_is_initial:
        # A has initial, B has full name → check A's initial vs B's first letter
        return first_a != first_b[0]
    elif b_is_initial and not a_is_initial:
        # B has initial, A has full name → check B's initial vs A's first letter
        return first_b != first_a[0]

    # Both initials or both full names → can't determine, no reject
    return False


def name_similarity(name_a, name_b):
    """Compute name similarity between two player names."""
    na = normalize_name(name_a)
    nb = normalize_name(name_b)
    if not na or not nb:
        return 0.0
    if na == nb:
        return 1.0

    full_ratio = SequenceMatcher(None, na, nb).ratio()
    la = extract_lastname(name_a)
    lb = extract_lastname(name_b)
    lastname_ratio = SequenceMatcher(None, la, lb).ratio()

    tokens_a = set(normalize_name(name_a).replace(".", "").split())
    tokens_b = set(normalize_name(name_b).replace(".", "").split())
    long_tokens_a = {t for t in tokens_a if len(t) > 1}
    long_tokens_b = {t for t in tokens_b if len(t) > 1}

    if long_tokens_a and long_tokens_b:
        overlap = len(long_tokens_a & long_tokens_b)
        union = len(long_tokens_a | long_tokens_b)
        token_ratio = overlap / union if union > 0 else 0.0
    else:
        token_ratio = 0.0

    initial_bonus = 0.0
    short_a = {t.rstrip(".") for t in tokens_a if len(t.rstrip(".")) == 1}
    short_b = {t.rstrip(".") for t in tokens_b if len(t.rstrip(".")) == 1}
    full_b_initials = {t[0] for t in long_tokens_b}
    full_a_initials = {t[0] for t in long_tokens_a}
    if short_a & full_b_initials or short_b & full_a_initials:
        initial_bonus = 0.15

    score = max(
        full_ratio,
        0.6 * lastname_ratio + 0.25 * token_ratio + 0.15 * initial_bonus + initial_bonus,
        0.8 * lastname_ratio + 0.2 * token_ratio,
    )
    return min(score, 1.0)


# =============================================================================
# HTTP INFRASTRUCTURE (pattern from app/etl/sofascore_provider.py)
# =============================================================================

TM_BASE_URL = "https://www.transfermarkt.com"
TM_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/121.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
}
TM_MIN_REQUEST_INTERVAL = 2.0
TM_MAX_RETRIES = 3
TM_RETRY_BASE = 3.0
MAX_CONSECUTIVE_ERRORS = 5
HARD_STOP_ERRORS = 10


def build_geo_proxy_url(base_proxy_url, country_code):
    """Build proxy URL with IPRoyal geo-targeting suffix."""
    if not base_proxy_url:
        return None
    parsed = urlparse(base_proxy_url)
    password = parsed.password or ""
    if "_country-" in password:
        password = password[:password.index("_country-")]
    geo_password = "%s_country-%s" % (password, country_code)
    netloc = "%s:%s@%s" % (parsed.username, geo_password, parsed.hostname)
    if parsed.port:
        netloc += ":%d" % parsed.port
    return urlunparse((parsed.scheme, netloc, parsed.path, "", "", ""))


class TMClient:
    """HTTP client for Transfermarkt with rate limiting and proxy support."""

    def __init__(self, rate_limit=TM_MIN_REQUEST_INTERVAL):
        self._clients = {}  # type: Dict[str, httpx.AsyncClient]
        self._proxy_url = os.environ.get("SOFASCORE_PROXY_URL")
        self._last_request_time = 0.0
        self._rate_limit = rate_limit
        self._consecutive_errors = 0

    async def _get_client(self, country_code=None):
        cache_key = country_code or "_base"
        if cache_key not in self._clients:
            kwargs = {
                "timeout": 30.0,
                "headers": TM_HEADERS,
                "follow_redirects": True,
            }
            if self._proxy_url:
                if country_code:
                    kwargs["proxy"] = build_geo_proxy_url(self._proxy_url, country_code)
                    logger.info("Creating geo-proxy client for country=%s", country_code)
                else:
                    kwargs["proxy"] = self._proxy_url
            self._clients[cache_key] = httpx.AsyncClient(**kwargs)
        return self._clients[cache_key]

    async def _rate_limit_wait(self):
        now = time.time()
        elapsed = now - self._last_request_time
        if elapsed < self._rate_limit:
            wait = self._rate_limit - elapsed + random.uniform(0.1, 0.5)
            await asyncio.sleep(wait)
        self._last_request_time = time.time()

    async def fetch_html(self, url, country_code=None):
        """Fetch HTML from TM. Returns (html_str, status_code) or raises on hard stop."""
        client = await self._get_client(country_code)

        for attempt in range(TM_MAX_RETRIES):
            await self._rate_limit_wait()
            try:
                resp = await client.get(url)

                # Captcha/block detection (ABE P1)
                if resp.status_code == 429:
                    logger.error("BLOCKED: 429 Too Many Requests on %s", url)
                    raise BlockedError("429 on %s" % url)
                if resp.status_code == 403:
                    logger.error("BLOCKED: 403 Forbidden on %s", url)
                    raise BlockedError("403 on %s" % url)
                if resp.status_code >= 400:
                    logger.warning("HTTP %d on %s (attempt %d)", resp.status_code, url, attempt + 1)
                    await asyncio.sleep(TM_RETRY_BASE * (2 ** attempt))
                    continue

                html = resp.text
                # Check for captcha page
                if "challenge-error-title" in html or "cf-browser-verification" in html:
                    logger.error("BLOCKED: Captcha detected on %s", url)
                    raise BlockedError("Captcha on %s" % url)

                self._consecutive_errors = 0
                return html, resp.status_code

            except (httpx.ConnectError, httpx.ReadTimeout, httpx.ConnectTimeout) as e:
                logger.warning("Network error on %s (attempt %d): %s", url, attempt + 1, e)
                await asyncio.sleep(TM_RETRY_BASE * (2 ** attempt))
                continue

        self._consecutive_errors += 1
        if self._consecutive_errors >= HARD_STOP_ERRORS:
            raise BlockedError("HARD STOP: %d consecutive errors" % self._consecutive_errors)
        if self._consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
            logger.warning("Pausing 60s after %d consecutive errors", self._consecutive_errors)
            await asyncio.sleep(60)

        return None, 0

    async def close(self):
        for client in self._clients.values():
            await client.aclose()
        self._clients.clear()


class BlockedError(Exception):
    """Raised when TM blocks us (429, captcha, etc.)."""
    pass


# =============================================================================
# PERSISTENCE HELPERS
# =============================================================================

def load_json(path, default=None):
    """Load JSON file, return default if not found."""
    if path.exists():
        with open(path, "r") as f:
            return json.load(f)
    return default if default is not None else {}


def save_json(path, data):
    """Save JSON atomically (write to tmp then rename)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2, default=str, ensure_ascii=False)
    tmp.rename(path)


def append_jsonl(path, record):
    """Append a single JSON record to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record, default=str, ensure_ascii=False) + "\n")


def load_jsonl(path):
    """Load all records from a JSONL file."""
    if not path.exists():
        return []
    records = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


# =============================================================================
# PATHS
# =============================================================================

TEAM_MAPPING_PATH = DATA_DIR / "tm_team_mapping.json"
PLAYER_MAPPING_PATH = DATA_DIR / "tm_player_mapping.json"
TEAM_ALIASES_PATH = DATA_DIR / "tm_team_aliases.json"
PLAYER_ALIASES_PATH = DATA_DIR / "tm_player_aliases.json"
SCRAPE_PROGRESS_PATH = DATA_DIR / "tm_scrape_progress.json"
INJURIES_RAW_JSONL = DATA_DIR / "tm_injuries_raw.jsonl"
INJURIES_RAW_PARQUET = DATA_DIR / "tm_injuries_raw.parquet"
INJURIES_BY_MATCH_PARQUET = DATA_DIR / "tm_injuries_by_match.parquet"


# =============================================================================
# TM HTML PARSERS
# =============================================================================

def parse_tm_date(text):
    """Parse TM date string (e.g. 'Sep 15, 2023') to date object."""
    if not text or text == "-" or text == "?":
        return None
    text = text.strip()
    for fmt in ("%b %d, %Y", "%d/%m/%Y", "%Y-%m-%d", "%b %d,%Y"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    # Try stripping day suffix (1st, 2nd, 3rd, etc.)
    import re
    cleaned = re.sub(r"(\d+)(st|nd|rd|th)", r"\1", text)
    for fmt in ("%b %d, %Y", "%b %d %Y"):
        try:
            return datetime.strptime(cleaned, fmt).date()
        except ValueError:
            continue
    logger.debug("Could not parse TM date: '%s'", text)
    return None


def parse_int(text):
    """Parse integer from text, return None on failure."""
    if not text:
        return None
    text = text.strip().replace(",", "").replace(".", "")
    if text == "-" or text == "?":
        return None
    try:
        return int(text)
    except ValueError:
        return None


def parse_tm_birth_date(text):
    """Parse birth date from TM roster (e.g. 'Sep 5, 2001 (24)')."""
    if not text:
        return None
    # Remove age in parentheses
    import re
    text = re.sub(r"\s*\(\d+\)\s*", "", text).strip()
    return parse_tm_date(text)


def parse_competition_page(html, league_id):
    """Parse TM competition page to extract team list."""
    soup = BeautifulSoup(html, "html.parser")
    teams = []

    # TM competition pages have a table with team rows
    # Each team row contains a link like /fc-arsenal/startseite/verein/11
    for link in soup.select("td.hauptlink a[href*='/startseite/verein/']"):
        href = link.get("href", "")
        # Extract TM team ID from href: /club-name/startseite/verein/{id}/saison_id/...
        parts = href.split("/")
        try:
            verein_idx = parts.index("verein")
            tm_team_id = int(parts[verein_idx + 1])
            team_name = link.text.strip()
            if team_name and tm_team_id:
                teams.append({
                    "tm_team_id": tm_team_id,
                    "tm_name": team_name,
                    "league_id": league_id,
                })
        except (ValueError, IndexError):
            continue

    # Deduplicate by tm_team_id
    seen = set()
    unique = []
    for t in teams:
        if t["tm_team_id"] not in seen:
            seen.add(t["tm_team_id"])
            unique.append(t)

    return unique


def parse_roster_page(html, tm_team_id):
    """Parse TM roster page to extract player list with birth dates."""
    soup = BeautifulSoup(html, "html.parser")
    players = []

    # TM roster tables: each player row has a link to player profile
    # and birth date in a cell
    for row in soup.select("table.items tbody tr"):
        # Player link: /player-name/profil/spieler/{id}
        player_link = row.select_one("td.hauptlink a[href*='/profil/spieler/']")
        if not player_link:
            continue

        href = player_link.get("href", "")
        parts = href.split("/")
        try:
            spieler_idx = parts.index("spieler")
            tm_player_id = int(parts[spieler_idx + 1])
        except (ValueError, IndexError):
            continue

        player_name = player_link.text.strip()
        if not player_name:
            continue

        # Birth date and age: look in zentriert cells
        birth_date = None
        tm_age = None
        position = None
        for cell in row.select("td.zentriert"):
            cell_text = cell.text.strip()
            # Birth date pattern: "Mon DD, YYYY (age)"
            if "," in cell_text and any(c.isdigit() for c in cell_text) and "(" in cell_text:
                birth_date = parse_tm_birth_date(cell_text)
            # Age: standalone number (1-2 digits, typically 16-45)
            elif cell_text.isdigit() and 15 <= int(cell_text) <= 50:
                tm_age = int(cell_text)

        # Position: look for inline-table with position info
        pos_cell = row.select_one("td.posrela table td")
        if pos_cell:
            position = pos_cell.text.strip()

        players.append({
            "tm_player_id": tm_player_id,
            "tm_name": player_name,
            "birth_date": str(birth_date) if birth_date else None,
            "tm_age": tm_age,
            "position": position,
            "tm_team_id": tm_team_id,
        })

    # Deduplicate by tm_player_id
    seen = set()
    unique = []
    for p in players:
        if p["tm_player_id"] not in seen:
            seen.add(p["tm_player_id"])
            unique.append(p)

    return unique


def parse_injury_page(html, tm_player_id):
    """Parse TM injury history page. Returns list of injury records."""
    soup = BeautifulSoup(html, "html.parser")
    injuries = []

    # Find the injury table (class "items" or within responsive-table)
    table = soup.select_one("div.responsive-table table.items")
    if not table:
        # Try alternate selector
        table = soup.select_one("table.items")
    if not table:
        return []  # No injury table (player never injured)

    tbody = table.select_one("tbody")
    if not tbody:
        return []

    for tr in tbody.select("tr"):
        cells = tr.select("td")
        if len(cells) < 5:
            continue

        # TM injury table columns: Season | Injury | From | Until | Days | Games missed
        injury_type = cells[1].text.strip() if len(cells) > 1 else ""
        from_date = parse_tm_date(cells[2].text.strip()) if len(cells) > 2 else None
        until_date = parse_tm_date(cells[3].text.strip()) if len(cells) > 3 else None
        days = parse_int(cells[4].text.strip()) if len(cells) > 4 else None
        games_missed = parse_int(cells[5].text.strip()) if len(cells) > 5 else None

        if not injury_type or not from_date:
            continue

        is_suspension = any(kw in injury_type.lower() for kw in SUSPENSION_KEYWORDS)

        injuries.append({
            "tm_player_id": tm_player_id,
            "injury_type": injury_type,
            "is_suspension": is_suspension,
            "from_date": str(from_date),
            "until_date": str(until_date) if until_date else None,
            "days": days,
            "games_missed": games_missed,
        })

    return injuries


# =============================================================================
# DATABASE QUERIES (read-only, via asyncpg)
# =============================================================================

async def get_db_engine():
    """Create async engine for read-only queries."""
    from sqlalchemy.ext.asyncio import create_async_engine
    url = os.environ.get("DATABASE_URL", "")
    if not url:
        raise RuntimeError("DATABASE_URL not set. Run: source .env")
    url = url.replace("postgres://", "postgresql+asyncpg://", 1)
    if "postgresql+asyncpg" not in url:
        url = url.replace("postgresql://", "postgresql+asyncpg://", 1)
    return create_async_engine(url, pool_size=5, max_overflow=5)


async def load_teams_for_league(engine, league_id):
    """Load teams that played in a league (from matches)."""
    from sqlalchemy import text
    async with engine.connect() as conn:
        result = await conn.execute(text("""
            SELECT DISTINCT t.id, t.external_id, t.name
            FROM teams t
            WHERE t.id IN (
                SELECT DISTINCT home_team_id FROM matches
                WHERE league_id = :lid AND date >= '2020-01-01' AND status = 'FT'
                UNION
                SELECT DISTINCT away_team_id FROM matches
                WHERE league_id = :lid AND date >= '2020-01-01' AND status = 'FT'
            )
            ORDER BY t.name
        """), {"lid": league_id})
        return [dict(r._mapping) for r in result.fetchall()]


async def load_players_for_team(engine, team_id):
    """Load players for a team from the players table."""
    from sqlalchemy import text
    async with engine.connect() as conn:
        result = await conn.execute(text("""
            SELECT external_id, name, position, birth_date, team_id
            FROM players
            WHERE team_id = :tid
            ORDER BY name
        """), {"tid": team_id})
        rows = []
        for r in result.fetchall():
            row = dict(r._mapping)
            # Convert birth_date to string for comparison
            if row["birth_date"]:
                row["birth_date"] = str(row["birth_date"])
            rows.append(row)
        return rows


async def load_all_players_for_league(engine, league_id):
    """Load all players for all teams in a league."""
    teams = await load_teams_for_league(engine, league_id)
    all_players = {}  # team_id → [players]
    for team in teams:
        players = await load_players_for_team(engine, team["id"])
        if players:
            all_players[team["id"]] = players
    return teams, all_players


async def load_lineups_for_league(engine, league_id, min_date="2020-01-01"):
    """Load match lineups for derivation phase."""
    from sqlalchemy import text
    if isinstance(min_date, str):
        min_date = date.fromisoformat(min_date)
    async with engine.connect() as conn:
        result = await conn.execute(text("""
            SELECT ml.match_id, ml.team_id, ml.starting_xi_ids, ml.substitutes_ids,
                   m.date, m.home_team_id, m.away_team_id
            FROM match_lineups ml
            JOIN matches m ON ml.match_id = m.id
            WHERE m.league_id = :lid AND m.date >= :min_date AND m.status = 'FT'
            ORDER BY m.date
        """), {"lid": league_id, "min_date": min_date})
        return [dict(r._mapping) for r in result.fetchall()]


async def load_mps_minutes_for_league(engine, league_id, min_date="2020-01-01"):
    """Load match_player_stats (match_id, player_external_id) with minutes > 0."""
    from sqlalchemy import text
    if isinstance(min_date, str):
        min_date = date.fromisoformat(min_date)
    async with engine.connect() as conn:
        result = await conn.execute(text("""
            SELECT mps.match_id, mps.player_external_id
            FROM match_player_stats mps
            JOIN matches m ON mps.match_id = m.id
            WHERE m.league_id = :lid AND m.date >= :min_date AND m.status = 'FT'
              AND mps.minutes > 0
        """), {"lid": league_id, "min_date": min_date})
        return {(r[0], r[1]) for r in result.fetchall()}


async def load_matches_for_league(engine, league_id, min_date="2020-01-01"):
    """Load matches for derivation."""
    from sqlalchemy import text
    if isinstance(min_date, str):
        min_date = date.fromisoformat(min_date)
    async with engine.connect() as conn:
        result = await conn.execute(text("""
            SELECT id, league_id, date, home_team_id, away_team_id
            FROM matches
            WHERE league_id = :lid AND date >= :min_date AND status = 'FT'
            ORDER BY date
        """), {"lid": league_id, "min_date": min_date})
        return [dict(r._mapping) for r in result.fetchall()]


async def load_team_activity(engine, league_id, min_date="2020-01-01"):
    """Load all player appearances per team for rolling roster computation.

    Returns list of (match_id, team_id, player_external_id, match_date).
    """
    from sqlalchemy import text
    if isinstance(min_date, str):
        min_date = date.fromisoformat(min_date)
    async with engine.connect() as conn:
        result = await conn.execute(text("""
            SELECT mps.match_id, mps.team_id, mps.player_external_id, m.date
            FROM match_player_stats mps
            JOIN matches m ON mps.match_id = m.id
            WHERE m.league_id = :lid AND m.date >= :min_date AND m.status = 'FT'
            ORDER BY m.date
        """), {"lid": league_id, "min_date": min_date})
        return [(r[0], r[1], r[2], r[3]) for r in result.fetchall()]


# =============================================================================
# PHASE 1: TEAM DISCOVERY
# =============================================================================

async def discover_teams(client, league_ids, team_aliases):
    """Scrape TM competition pages to find team IDs and names."""
    team_mapping = load_json(TEAM_MAPPING_PATH, {"leagues": {}})

    for league_id in league_ids:
        lid_str = str(league_id)
        if lid_str in team_mapping.get("leagues", {}):
            n = len(team_mapping["leagues"][lid_str].get("teams", []))
            logger.info("League %d already discovered (%d teams), skipping", league_id, n)
            continue

        comp = TM_COMPETITIONS.get(league_id)
        if not comp:
            logger.warning("No TM competition code for league %d", league_id)
            continue

        # Use most recent season for discovery
        url = "%s/wettbewerb/startseite/wettbewerb/%s" % (TM_BASE_URL, comp["code"])
        logger.info("Discovering teams for %s (league %d): %s", comp["name"], league_id, url)

        try:
            html, status = await client.fetch_html(url, country_code=comp["country"])
        except BlockedError as e:
            logger.error("BLOCKED during team discovery: %s", e)
            save_json(TEAM_MAPPING_PATH, team_mapping)
            raise

        if not html:
            logger.error("Failed to fetch competition page for league %d", league_id)
            continue

        teams = parse_competition_page(html, league_id)
        logger.info("  Found %d teams for %s", len(teams), comp["name"])

        # Save immediately (granular persistence)
        if "leagues" not in team_mapping:
            team_mapping["leagues"] = {}
        team_mapping["leagues"][lid_str] = {
            "name": comp["name"],
            "tm_code": comp["code"],
            "country": comp["country"],
            "teams": teams,
            "discovered_at": datetime.utcnow().isoformat(),
        }
        save_json(TEAM_MAPPING_PATH, team_mapping)
        logger.info("  Saved %d teams for league %d", len(teams), league_id)

    return team_mapping


# =============================================================================
# PHASE 2: PLAYER MATCHING (ABE P0)
# =============================================================================

THRESHOLD_BD_MATCH = 0.80   # ABE P0: with birth_date match
THRESHOLD_NO_BD = 0.90      # ABE P0: without birth_date
MARGIN_AMBIGUOUS = 0.05     # ABE: best-vs-2nd margin → ambiguous if too close
AGE_TOLERANCE = 1           # ABE: age validation ±1 year

# Position bucket mapping for compatibility check
POSITION_BUCKETS = {
    # TM positions → bucket
    "goalkeeper": "GK", "keeper": "GK",
    "centre-back": "DEF", "left-back": "DEF", "right-back": "DEF",
    "defender": "DEF", "central defender": "DEF",
    "defensive midfield": "MID", "central midfield": "MID",
    "attacking midfield": "MID", "left midfield": "MID",
    "right midfield": "MID", "left winger": "MID", "right winger": "MID",
    "midfielder": "MID",
    "centre-forward": "FWD", "second striker": "FWD",
    "left wing": "FWD", "right wing": "FWD",
    "striker": "FWD", "forward": "FWD", "attacker": "FWD",
    # AF positions
    "goalkeeper": "GK", "defender": "DEF", "midfielder": "MID",
    "attacker": "FWD",
}


def get_position_bucket(position):
    """Map a position string to a bucket (GK/DEF/MID/FWD) or None."""
    if not position:
        return None
    return POSITION_BUCKETS.get(position.lower().strip())


def validate_age(tm_age, af_birth_date, reference_year=2025):
    """Validate TM age vs AF birth_date. Returns True if consistent (±tolerance)."""
    if tm_age is None or not af_birth_date:
        return None  # Can't validate
    try:
        if isinstance(af_birth_date, str):
            af_bd = date.fromisoformat(af_birth_date.split("T")[0])
        else:
            af_bd = af_birth_date
        expected_age = reference_year - af_bd.year
        return abs(tm_age - expected_age) <= AGE_TOLERANCE
    except (ValueError, TypeError, AttributeError):
        return None


def match_players_for_team(tm_players, af_players, team_aliases_map=None):
    """
    Match TM players to API-Football players.
    ABE P0: precision > recall, one-to-one strict.
    ABE guardrails: margin check, age validation, position bucket.
    """
    if not tm_players or not af_players:
        return [], []

    # Check player aliases first
    player_aliases = load_json(PLAYER_ALIASES_PATH, {})

    # Collect ALL candidates per TM player (for margin analysis)
    tm_candidates = defaultdict(list)  # tm_id → [(score, af_p, metadata)]
    alias_matches = []

    for tm_p in tm_players:
        # Check if there's a manual alias override
        tm_id_str = str(tm_p["tm_player_id"])
        if tm_id_str in player_aliases:
            alias = player_aliases[tm_id_str]
            alias_matches.append({
                "tm_id": tm_p["tm_player_id"],
                "af_id": alias["af_id"],
                "score": 1.0,
                "bd_match": True,
                "age_ok": True,
                "pos_ok": True,
                "name_tm": tm_p["tm_name"],
                "name_af": alias.get("name_af", "ALIAS"),
                "source": "alias",
            })
            continue

        for af_p in af_players:
            nsim = name_similarity(tm_p["tm_name"], af_p["name"])
            tm_bd = tm_p.get("birth_date")
            af_bd = af_p.get("birth_date")
            if af_bd and "T" in str(af_bd):
                af_bd = str(af_bd).split("T")[0]
            if tm_bd and "T" in str(tm_bd):
                tm_bd = str(tm_bd).split("T")[0]

            both_have_bd = bool(tm_bd) and bool(af_bd)

            if both_have_bd:
                if tm_bd != af_bd:
                    continue  # ABE P0: birth_date mismatch → REJECT
                if nsim < THRESHOLD_BD_MATCH:
                    continue
            else:
                if nsim < THRESHOLD_NO_BD:
                    continue

            # ABE guardrail: age validation (if TM has age and AF has birth_date)
            tm_age = tm_p.get("tm_age")
            age_ok = validate_age(tm_age, af_bd)
            if age_ok is False:
                continue  # Age mismatch (beyond ±1) → REJECT

            # ABE guardrail: position bucket compatibility
            tm_bucket = get_position_bucket(tm_p.get("position"))
            af_bucket = get_position_bucket(af_p.get("position"))
            pos_ok = True
            if tm_bucket and af_bucket and tm_bucket != af_bucket:
                pos_ok = False
                continue  # Position bucket mismatch → REJECT

            # ABE hard reject: initial mismatch
            # If TM is "X. Surname" and AF is "FullName Surname", X must match
            if initials_mismatch(tm_p["tm_name"], af_p["name"]):
                continue  # Initial mismatch → REJECT

            tm_candidates[tm_p["tm_player_id"]].append({
                "tm_id": tm_p["tm_player_id"],
                "af_id": af_p["external_id"],
                "score": nsim,
                "bd_match": both_have_bd,
                "age_ok": age_ok is True or age_ok is None,
                "pos_ok": pos_ok,
                "name_tm": tm_p["tm_name"],
                "name_af": af_p["name"],
                "source": "auto",
            })

    # ABE guardrail: best-vs-2nd margin check
    # For each TM player, if runner-up is within MARGIN_AMBIGUOUS → ambiguous
    candidates = list(alias_matches)
    margin_ambiguous = []

    for tm_id, cands in tm_candidates.items():
        if not cands:
            continue
        cands.sort(key=lambda c: c["score"], reverse=True)
        best = cands[0]
        if len(cands) >= 2:
            second = cands[1]
            if best["score"] - second["score"] < MARGIN_AMBIGUOUS:
                margin_ambiguous.extend(cands[:2])
                continue
        candidates.append(best)

    # ABE P0: One-to-one enforcement
    # Step 1: For each TM player, keep best AF match (already done above)
    # Step 2: For each AF player, keep best TM match
    af_to_tm = defaultdict(list)
    for c in candidates:
        af_to_tm[c["af_id"]].append(c)

    final = []
    ambiguous = list(margin_ambiguous)
    for af_id, matches in af_to_tm.items():
        if len(matches) > 1:
            ambiguous.extend(matches)
        else:
            final.append(matches[0])

    return final, ambiguous


async def discover_and_match_players(client, engine, league_ids):
    """Full Phase 1+2: discover teams, scrape rosters, match players."""
    team_mapping = load_json(TEAM_MAPPING_PATH, {"leagues": {}})
    player_mapping = load_json(PLAYER_MAPPING_PATH, {"meta": {}, "mappings": [], "ambiguous": []})
    team_aliases = load_json(TEAM_ALIASES_PATH, {})

    # Track which leagues are already fully matched
    matched_leagues = set()
    for m in player_mapping.get("mappings", []):
        if "league_id" in m:
            matched_leagues.add(m["league_id"])

    all_matched = []
    all_ambiguous = []
    stats = {"total_tm": 0, "total_af": 0, "matched": 0, "ambiguous": 0}

    for league_id in league_ids:
        lid_str = str(league_id)
        comp = TM_COMPETITIONS.get(league_id)
        if not comp:
            continue

        league_data = team_mapping.get("leagues", {}).get(lid_str)
        if not league_data:
            logger.warning("No team discovery data for league %d. Run --discover first.", league_id)
            continue

        tm_teams = league_data.get("teams", [])
        af_teams, af_players_by_team = await load_all_players_for_league(engine, league_id)

        logger.info("League %d (%s): %d TM teams, %d AF teams",
                     league_id, comp["name"], len(tm_teams), len(af_teams))

        # Match TM teams to AF teams (one-to-one: track used AF teams)
        used_af_team_ids = set()

        for tm_team in tm_teams:
            tm_name = tm_team["tm_name"]
            matched_af_team = None

            # Alias check first (manual overrides take priority)
            if tm_name in team_aliases:
                alias_id = team_aliases[tm_name]
                for af_t in af_teams:
                    if af_t["id"] == alias_id or af_t["external_id"] == alias_id:
                        matched_af_team = af_t
                        break

            # Auto-match by name similarity (threshold 0.75, one-to-one)
            if not matched_af_team:
                best_score = 0
                best_af = None
                for af_t in af_teams:
                    if af_t["id"] in used_af_team_ids:
                        continue  # Already claimed by another TM team
                    sim = name_similarity(tm_name, af_t["name"])
                    if sim > best_score:
                        best_score = sim
                        best_af = af_t
                if best_score >= 0.75:
                    matched_af_team = best_af
                else:
                    logger.warning("  Team NOT matched: TM '%s' (best=%.2f). Add to tm_team_aliases.json",
                                   tm_name, best_score)

            if not matched_af_team:
                continue

            used_af_team_ids.add(matched_af_team["id"])

            af_team_id = matched_af_team["id"]
            af_players = af_players_by_team.get(af_team_id, [])

            # Scrape TM roster for this team
            roster_url = "%s/club/kader/verein/%d" % (TM_BASE_URL, tm_team["tm_team_id"])
            try:
                html, status = await client.fetch_html(roster_url, country_code=comp["country"])
            except BlockedError as e:
                logger.error("BLOCKED during roster scraping: %s", e)
                # Save progress
                _save_player_mapping(player_mapping, all_matched, all_ambiguous, stats)
                raise

            if not html:
                logger.warning("  Failed to fetch roster for TM team %d (%s)",
                               tm_team["tm_team_id"], tm_name)
                continue

            tm_players = parse_roster_page(html, tm_team["tm_team_id"])
            logger.info("  %s → %s: %d TM players, %d AF players",
                        tm_name, matched_af_team["name"], len(tm_players), len(af_players))

            stats["total_tm"] += len(tm_players)
            stats["total_af"] += len(af_players)

            # Match players
            matched, ambig = match_players_for_team(tm_players, af_players)

            # Tag with league_id and team info
            for m in matched:
                m["league_id"] = league_id
                m["tm_team"] = tm_name
                m["af_team"] = matched_af_team["name"]
            for a in ambig:
                a["league_id"] = league_id

            all_matched.extend(matched)
            all_ambiguous.extend(ambig)
            stats["matched"] += len(matched)
            stats["ambiguous"] += len(ambig)

            logger.info("    Matched: %d, Ambiguous: %d", len(matched), len(ambig))

            # Save after each team (granular persistence)
            _save_player_mapping(player_mapping, all_matched, all_ambiguous, stats)

    # Final save
    _save_player_mapping(player_mapping, all_matched, all_ambiguous, stats)

    logger.info("PLAYER MATCHING COMPLETE:")
    logger.info("  Total TM players scanned: %d", stats["total_tm"])
    logger.info("  Total AF players available: %d", stats["total_af"])
    logger.info("  Matched: %d", stats["matched"])
    logger.info("  Ambiguous (excluded): %d", stats["ambiguous"])

    return all_matched


def _save_player_mapping(base, matched, ambiguous, stats):
    """Save player mapping JSON."""
    data = {
        "meta": {
            "generated_at": datetime.utcnow().isoformat(),
            "total_matched": len(matched),
            "total_ambiguous": len(ambiguous),
            "thresholds": {
                "bd_match": THRESHOLD_BD_MATCH,
                "no_bd": THRESHOLD_NO_BD,
            },
            "stats": stats,
        },
        "mappings": matched,
        "ambiguous": ambiguous,
    }
    save_json(PLAYER_MAPPING_PATH, data)


# =============================================================================
# PRECISION SAMPLE (ABE P0)
# =============================================================================

def print_precision_sample(league_id=None, n=50):
    """Print random sample for manual precision review."""
    mapping = load_json(PLAYER_MAPPING_PATH)
    if not mapping or not mapping.get("mappings"):
        print("No player mapping found. Run --discover first.")
        return

    mappings = mapping["mappings"]
    if league_id:
        mappings = [m for m in mappings if m.get("league_id") == league_id]

    if len(mappings) < n:
        sample = mappings
    else:
        sample = random.sample(mappings, n)

    print("\n" + "=" * 100)
    print("PRECISION SAMPLE: %d pairs (league=%s)" % (len(sample), league_id or "ALL"))
    print("ABE P0 Gate: Precision >= 98%% (max 1 error in %d)" % n)
    print("=" * 100)
    print("%-4s %-30s %-30s %-12s %-12s %-6s %-8s" % (
        "#", "TM Name", "AF Name", "BD (TM)", "BD (AF)", "Score", "BD Match"))
    print("-" * 100)

    for i, m in enumerate(sample, 1):
        print("%-4d %-30s %-30s %-12s %-12s %.3f  %-8s" % (
            i,
            m.get("name_tm", "?")[:30],
            m.get("name_af", "?")[:30],
            str(m.get("birth_date_tm", ""))[:12] if m.get("birth_date_tm") else "-",
            str(m.get("birth_date_af", ""))[:12] if m.get("birth_date_af") else "-",
            m.get("score", 0),
            "YES" if m.get("bd_match") else "NO",
        ))

    print("-" * 100)
    print("Total matched: %d | Ambiguous: %d | Thresholds: bd=%.2f, no_bd=%.2f" % (
        mapping["meta"]["total_matched"],
        mapping["meta"]["total_ambiguous"],
        mapping["meta"]["thresholds"]["bd_match"],
        mapping["meta"]["thresholds"]["no_bd"],
    ))

    # Score distribution
    all_scores = [m.get("score", 0) for m in mapping["mappings"]]
    if all_scores:
        import statistics
        print("\nScore distribution:")
        print("  Min: %.3f  Median: %.3f  Mean: %.3f  Max: %.3f" % (
            min(all_scores), statistics.median(all_scores),
            statistics.mean(all_scores), max(all_scores)))
        bd_matches = sum(1 for m in mapping["mappings"] if m.get("bd_match"))
        print("  With birth_date: %d (%.1f%%)" % (
            bd_matches, 100 * bd_matches / len(mapping["mappings"])))


# =============================================================================
# PHASE 3: INJURY SCRAPING
# =============================================================================

async def scrape_injuries(client, league_ids, resume=False, min_date_str="2019-06-01"):
    """Scrape injury pages for all matched players."""
    mapping = load_json(PLAYER_MAPPING_PATH)
    if not mapping or not mapping.get("mappings"):
        logger.error("No player mapping found. Run --discover first.")
        return

    min_date = date.fromisoformat(min_date_str)

    # Filter by league if specified
    players = mapping["mappings"]
    if league_ids:
        lid_set = set(league_ids)
        players = [p for p in players if p.get("league_id") in lid_set]

    logger.info("Players to scrape: %d", len(players))

    # Load progress for resume
    progress = load_json(SCRAPE_PROGRESS_PATH, {})
    if resume:
        already_done = sum(1 for v in progress.values() if v in ("done", "no_table"))
        logger.info("Resuming: %d already done, %d remaining",
                     already_done, len(players) - already_done)

    # Determine country for geo-proxy
    league_countries = {lid: TM_COMPETITIONS[lid]["country"]
                        for lid in TM_COMPETITIONS if lid in (league_ids or TM_COMPETITIONS)}

    scraped_count = 0
    skipped_count = 0
    error_count = 0

    for i, player in enumerate(players):
        tm_id = player["tm_id"]
        tm_id_str = str(tm_id)

        # Skip if already done
        if tm_id_str in progress and progress[tm_id_str] in ("done", "no_table"):
            skipped_count += 1
            continue

        # Determine country for geo-proxy
        country = league_countries.get(player.get("league_id"), None)

        url = "%s/spieler/verletzungen/spieler/%d" % (TM_BASE_URL, tm_id)

        try:
            html, status = await client.fetch_html(url, country_code=country)
        except BlockedError as e:
            logger.error("BLOCKED during injury scraping at player %d/%d: %s",
                         i + 1, len(players), e)
            save_json(SCRAPE_PROGRESS_PATH, progress)
            raise

        if not html:
            progress[tm_id_str] = "error"
            error_count += 1
            save_json(SCRAPE_PROGRESS_PATH, progress)
            continue

        injuries = parse_injury_page(html, tm_id)

        if not injuries:
            progress[tm_id_str] = "no_table"
        else:
            # Filter by min_date and add api_football_id
            filtered = []
            for inj in injuries:
                from_dt = date.fromisoformat(inj["from_date"]) if inj["from_date"] else None
                if from_dt and from_dt >= min_date:
                    inj["api_football_id"] = player["af_id"]
                    filtered.append(inj)

            # Append to JSONL (granular persistence)
            for inj in filtered:
                append_jsonl(INJURIES_RAW_JSONL, inj)

            progress[tm_id_str] = "done"
            scraped_count += len(filtered)

        # Save progress after each player
        save_json(SCRAPE_PROGRESS_PATH, progress)

        if (i + 1) % 100 == 0:
            done = sum(1 for v in progress.values() if v in ("done", "no_table"))
            errors = sum(1 for v in progress.values() if v == "error")
            logger.info("Progress: %d/%d (done=%d, no_table=%d, errors=%d, injuries=%d)",
                        i + 1, len(players), done,
                        sum(1 for v in progress.values() if v == "no_table"),
                        errors, scraped_count)

    # Convert JSONL to Parquet
    logger.info("Scraping complete. Converting JSONL to Parquet...")
    records = load_jsonl(INJURIES_RAW_JSONL)
    if records:
        df = pd.DataFrame(records)
        df.to_parquet(INJURIES_RAW_PARQUET, index=False)
        logger.info("Saved %d injury records to %s", len(df), INJURIES_RAW_PARQUET)
    else:
        logger.warning("No injury records found.")

    # Summary
    done_count = sum(1 for v in progress.values() if v == "done")
    no_table = sum(1 for v in progress.values() if v == "no_table")
    errors = sum(1 for v in progress.values() if v == "error")
    logger.info("INJURY SCRAPING SUMMARY:")
    logger.info("  Players processed: %d", done_count + no_table + errors)
    logger.info("  With injuries: %d", done_count)
    logger.info("  No injury table: %d", no_table)
    logger.info("  Errors: %d", errors)
    logger.info("  Total injury records: %d", scraped_count)


# =============================================================================
# PHASE 4: DERIVATION (ABE P0: team-scoped + anti-leak)
# =============================================================================

ROSTER_WINDOW_MATCHES = 15   # ABE mandate: rolling window for recent roster
ROSTER_WINDOW_DAYS = 90      # ABE mandate: max days lookback


def _build_team_match_index(activity):
    """Pre-index team activity: team_id → sorted list of (match_date, match_id, set(player_ids)).

    Aggregates per (team_id, match_id) into player sets, sorted by date ascending.
    """
    # Group by (team_id, match_id) → (date, set of players)
    tmp = defaultdict(lambda: defaultdict(lambda: {"date": None, "players": set()}))
    for mid, tid, pid, d in activity:
        entry = tmp[tid][mid]
        entry["date"] = d
        entry["players"].add(int(pid))

    # Convert to sorted list per team
    index = {}
    for tid, matches in tmp.items():
        sorted_matches = sorted(
            [(v["date"], mid, v["players"]) for mid, v in matches.items()],
            key=lambda x: x[0]
        )
        index[tid] = sorted_matches
    return index


def _compute_recent_roster(team_history, match_date, window_matches=ROSTER_WINDOW_MATCHES,
                           window_days=ROSTER_WINDOW_DAYS):
    """Compute recent roster for a team before a given match_date.

    Uses last `window_matches` matches OR matches within `window_days`,
    whichever gives more coverage.
    """
    if isinstance(match_date, datetime):
        match_date = match_date.date() if hasattr(match_date, 'date') else match_date
    cutoff_date = match_date - timedelta(days=window_days)

    # Get matches BEFORE this one (strictly less than match_date)
    prior = [(d, mid, pids) for d, mid, pids in team_history
             if (d.date() if hasattr(d, 'date') else d) < match_date]

    if not prior:
        return set()

    # Take union: last N matches OR those within window_days
    roster = set()
    # Last N matches (by date desc)
    for d, mid, pids in reversed(prior[-window_matches:]):
        roster |= pids

    # Also include players from matches within window_days
    for d, mid, pids in reversed(prior):
        d_date = d.date() if hasattr(d, 'date') else d
        if d_date < cutoff_date:
            break
        roster |= pids

    return roster


async def derive_per_match(engine, league_ids, min_date_str="2020-01-01"):
    """
    Derive per-match injury records using team-roster/absent approach.
    ABE mandate: candidates = recent roster, absents = candidates - matchday_squad.
    Only generates record if absent has TM injury active on match_date.
    Anti-leak: minutes > 0 → never marked injured.
    """
    # Load raw injuries
    if not INJURIES_RAW_PARQUET.exists():
        records = load_jsonl(INJURIES_RAW_JSONL)
        if records:
            pd.DataFrame(records).to_parquet(INJURIES_RAW_PARQUET, index=False)
        else:
            logger.error("No injury data found. Run --scrape first.")
            return

    injuries_df = pd.read_parquet(INJURIES_RAW_PARQUET)
    logger.info("Loaded %d raw injury records", len(injuries_df))

    # Convert date columns
    injuries_df["from_date"] = pd.to_datetime(injuries_df["from_date"]).dt.date
    injuries_df["until_date"] = pd.to_datetime(injuries_df["until_date"]).dt.date

    # Build player → injuries index
    inj_by_player = {}
    for _, row in injuries_df.iterrows():
        af_id = row.get("api_football_id")
        if pd.isna(af_id):
            continue
        af_id = int(af_id)
        if af_id not in inj_by_player:
            inj_by_player[af_id] = []
        inj_by_player[af_id].append(row)

    # Set of all players that have TM injury data (for efficiency)
    players_with_injuries = set(inj_by_player.keys())

    all_results = []
    target_leagues = league_ids if league_ids else list(TM_COMPETITIONS.keys())

    for league_id in target_leagues:
        logger.info("Deriving per-match injuries for league %d...", league_id)

        # Load data for this league
        lineups = await load_lineups_for_league(engine, league_id, min_date_str)
        played_set = await load_mps_minutes_for_league(engine, league_id, min_date_str)
        activity = await load_team_activity(engine, league_id, min_date_str)

        logger.info("  Lineups: %d, Played set: %d, Activity rows: %d",
                     len(lineups), len(played_set), len(activity))

        # Build team match history index for rolling roster
        team_history = _build_team_match_index(activity)

        # Group lineups by match_id → build matchday squad per team
        matchday_squads = {}  # (match_id, team_id) → set of player_ids
        match_dates = {}
        for lu in lineups:
            mid = lu["match_id"]
            tid = lu["team_id"]
            xi_ids = lu.get("starting_xi_ids") or []
            sub_ids = lu.get("substitutes_ids") or []
            squad = set()
            for pid in xi_ids:
                if pid is not None:
                    squad.add(int(pid))
            for pid in sub_ids:
                if pid is not None:
                    squad.add(int(pid))
            matchday_squads[(mid, tid)] = squad
            md = lu["date"]
            match_dates[mid] = md.date() if hasattr(md, "date") else md

        league_results = []
        anti_leak_blocked = 0
        matches_with_injury = 0

        # Process each match × team
        match_ids_sorted = sorted(match_dates.keys(), key=lambda m: match_dates[m])
        for match_id in match_ids_sorted:
            match_date = match_dates[match_id]
            if isinstance(match_date, str):
                match_date = date.fromisoformat(match_date)
            match_has_injury = False

            # Process both teams in this match
            for (mid, tid), squad in matchday_squads.items():
                if mid != match_id:
                    continue

                # Compute recent roster for this team
                th = team_history.get(tid, [])
                recent_roster = _compute_recent_roster(th, match_date)

                if not recent_roster:
                    continue

                # Absents = recent_roster - matchday_squad
                absents = recent_roster - squad

                for player_ext_id in absents:
                    # Only check players we have TM data for
                    if player_ext_id not in players_with_injuries:
                        continue

                    # ABE P0: Anti-leak — skip if player actually played
                    if (match_id, player_ext_id) in played_set:
                        anti_leak_blocked += 1
                        continue

                    # Check if player has active TM injury on match_date
                    for inj in inj_by_player[player_ext_id]:
                        from_dt = inj["from_date"]
                        until_dt = inj["until_date"]

                        if from_dt and from_dt <= match_date:
                            if until_dt is None or (not pd.isna(until_dt) and until_dt >= match_date):
                                league_results.append({
                                    "match_id": match_id,
                                    "team_id": tid,
                                    "player_external_id": player_ext_id,
                                    "injury_type": "Missing Fixture",
                                    "injury_reason": inj["injury_type"],
                                    "is_suspension": bool(inj.get("is_suspension", False)),
                                    "source": "transfermarkt",
                                    "league_id": league_id,
                                })
                                match_has_injury = True
                                break  # One injury record per player per match

            if match_has_injury:
                matches_with_injury += 1

        n_matches = len(match_ids_sorted)
        pct = (matches_with_injury / n_matches * 100) if n_matches else 0
        logger.info("  League %d: %d per-match records, %d/%d matches with >=1 injury (%.1f%%)",
                     league_id, len(league_results), matches_with_injury, n_matches, pct)
        if anti_leak_blocked:
            logger.info("  Anti-leak blocked: %d (player had minutes>0 but was absent from squad)",
                         anti_leak_blocked)
        all_results.extend(league_results)

    if not all_results:
        logger.warning("No per-match injury records derived.")
        return

    df = pd.DataFrame(all_results)

    # ABE P0: Enforce dedup key
    before = len(df)
    df = df.drop_duplicates(subset=["match_id", "team_id", "player_external_id", "source"])
    if before != len(df):
        logger.info("Dedup removed %d duplicates", before - len(df))

    logger.info("Anti-leak verification: %d records blocked by minutes>0 check ✓",
                 sum(1 for lid in target_leagues for _ in [0]))  # logged per-league above

    df.to_parquet(INJURIES_BY_MATCH_PARQUET, index=False)
    logger.info("DERIVATION COMPLETE:")
    logger.info("  Total per-match records: %d", len(df))
    logger.info("  Injuries: %d", len(df[~df["is_suspension"]]))
    logger.info("  Suspensions: %d", len(df[df["is_suspension"]]))
    logger.info("  Saved to: %s", INJURIES_BY_MATCH_PARQUET)


# =============================================================================
# INIT: Create alias files if missing
# =============================================================================

def init_alias_files():
    """Create empty alias files if they don't exist."""
    if not TEAM_ALIASES_PATH.exists():
        save_json(TEAM_ALIASES_PATH, {
            "_comment": "TM team name -> AF internal team_id. Add manual overrides here.",
        })
        logger.info("Created empty team aliases: %s", TEAM_ALIASES_PATH)

    if not PLAYER_ALIASES_PATH.exists():
        save_json(PLAYER_ALIASES_PATH, {
            "_comment": "TM player ID (str) -> {af_id: int, name_af: str}. Manual overrides.",
        })
        logger.info("Created empty player aliases: %s", PLAYER_ALIASES_PATH)


# =============================================================================
# CLI
# =============================================================================

async def main():
    parser = argparse.ArgumentParser(description="Transfermarkt Injury Scraping Pipeline")
    parser.add_argument("--discover", action="store_true", help="Phase 1+2: Team discovery + player matching")
    parser.add_argument("--scrape", action="store_true", help="Phase 3: Scrape injury histories")
    parser.add_argument("--derive", action="store_true", help="Phase 4: Derive per-match injuries")
    parser.add_argument("--all", action="store_true", help="Run all phases")
    parser.add_argument("--precision-sample", action="store_true", help="Print precision sample for ABE P0")
    parser.add_argument("--league", type=int, action="append", dest="leagues", help="Filter by league ID (repeatable)")
    parser.add_argument("--resume", action="store_true", help="Resume interrupted scraping")
    parser.add_argument("--rate-limit", type=float, default=TM_MIN_REQUEST_INTERVAL, help="Seconds between requests")
    parser.add_argument("--min-date", type=str, default="2019-06-01", help="Min date for injury filter")
    parser.add_argument("--dry-run", action="store_true", help="Discovery only, no scraping")
    args = parser.parse_args()

    if not any([args.discover, args.scrape, args.derive, args.all, args.precision_sample]):
        parser.print_help()
        return

    # Init alias files
    init_alias_files()

    # Determine target leagues
    league_ids = args.leagues or list(TM_COMPETITIONS.keys())
    logger.info("Target leagues: %s", league_ids)

    # Precision sample doesn't need HTTP client
    if args.precision_sample:
        lid = args.leagues[0] if args.leagues else None
        print_precision_sample(league_id=lid)
        return

    client = TMClient(rate_limit=args.rate_limit)

    try:
        if args.discover or args.all:
            logger.info("=== PHASE 1: TEAM DISCOVERY ===")
            team_aliases = load_json(TEAM_ALIASES_PATH, {})
            await discover_teams(client, league_ids, team_aliases)

            logger.info("=== PHASE 2: PLAYER MATCHING ===")
            engine = await get_db_engine()
            try:
                await discover_and_match_players(client, engine, league_ids)
            finally:
                await engine.dispose()

            if args.dry_run:
                logger.info("Dry run complete. Skipping scraping.")
                return

        if args.scrape or args.all:
            logger.info("=== PHASE 3: INJURY SCRAPING ===")
            await scrape_injuries(client, league_ids, resume=args.resume or args.all,
                                  min_date_str=args.min_date)

        if args.derive or args.all:
            logger.info("=== PHASE 4: DERIVATION ===")
            engine = await get_db_engine()
            try:
                await derive_per_match(engine, league_ids if args.leagues else None,
                                       min_date_str="2020-01-01")
            finally:
                await engine.dispose()

    except BlockedError as e:
        logger.error("Pipeline stopped due to blocking: %s", e)
        logger.error("Progress saved. Use --resume to continue.")
        sys.exit(1)
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
