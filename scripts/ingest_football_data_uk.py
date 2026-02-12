#!/usr/bin/env python3
"""
Football-Data UK Odds Ingestion Pipeline

Reproducible, auditable pipeline for ingesting historical odds from Football-Data UK.
Supports two CSV formats:
  - Main leagues (EUR): per-season files at mmz4281/{season}/{code}.csv
  - Extra leagues (LATAM+): multi-season file at new/{code}.csv

Modes:
  - PHASE A (default): Warehouse-only mode - builds local DuckDB warehouse
  - PHASE B (optional): Postgres backfill - updates matches.opening_odds_* in DB

Usage:
    # Main leagues only (warehouse)
    python3 scripts/ingest_football_data_uk.py --warehouse-only

    # Extra leagues only (ARG, BRA, MEX, USA)
    python3 scripts/ingest_football_data_uk.py --only-extra

    # Both main + extra
    python3 scripts/ingest_football_data_uk.py --extra-leagues

    # Filter to specific leagues
    python3 scripts/ingest_football_data_uk.py --only-extra --leagues ARG

    # Postgres backfill (dry-run first!)
    python3 scripts/ingest_football_data_uk.py --extra-leagues --backfill-postgres --dry-run
    python3 scripts/ingest_football_data_uk.py --extra-leagues --backfill-postgres --no-dry-run

RESTRICTIONS:
    - Does NOT touch: odds_snapshots, market_movement_snapshots, lineup_movement_snapshots, odds_history
    - Does NOT modify any PIT/scheduler/ETL odds live logic
    - Backfill ONLY updates: matches.opening_odds_* WHERE NULL (never overwrites)
    - Sets opening_odds_source, opening_odds_kind, opening_odds_column for auditability
"""

import argparse
import csv
import json
import logging
import os
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from io import StringIO
from pathlib import Path
from typing import Optional

import httpx

# Optional imports for database operations
try:
    import duckdb
    HAS_DUCKDB = True
except ImportError:
    HAS_DUCKDB = False
    duckdb = None

try:
    import asyncpg
    import asyncio
    HAS_ASYNCPG = True
except ImportError:
    HAS_ASYNCPG = False
    asyncpg = None


# =============================================================================
# CONFIGURATION
# =============================================================================

# Division mapping: FDUK code -> API-Football league_id
# "Main leagues" use per-season CSVs: mmz4281/{season}/{code}.csv
DIVISION_MAPPING = {
    "E0": 39,    # EPL
    "SP1": 140,  # La Liga
    "I1": 135,   # Serie A
    "D1": 78,    # Bundesliga
    "F1": 61,    # Ligue 1
    "N1": 88,    # Eredivisie
    "B1": 144,   # Belgian Pro League
    "P1": 94,    # Primeira Liga
    "T1": 203,   # Super Lig
    "E1": 40,    # EFL Championship
}

# "Extra leagues" use multi-season CSVs: new/{code}.csv
# Different column format (Home/Away, PSCH/PSCD/PSCA, etc.)
EXTRA_LEAGUE_MAPPING = {
    "ARG": 128,  # Argentina Primera Division
    "BRA": 71,   # Brazil Serie A
    "MEX": 262,  # Mexico Liga MX
    "USA": 253,  # MLS
}

# Reverse mapping for logs
LEAGUE_NAMES = {
    39: "EPL",
    140: "LaLiga",
    135: "SerieA",
    78: "Bundesliga",
    61: "Ligue1",
    88: "Eredivisie",
    144: "BelgianPro",
    94: "PrimeiraLiga",
    203: "SuperLig",
    40: "Championship",
    128: "Argentina",
    71: "BrazilSerieA",
    262: "LigaMX",
    253: "MLS",
}

# Seasons to ingest (newest first)
SEASONS = [
    "2526",  # 2025-2026 (current, partial)
    "2425",  # 2024-2025
    "2324",  # 2023-2024
    "2223",  # 2022-2023
    "2122",  # 2021-2022
    "2021",  # 2020-2021
    "1920",  # 2019-2020
    "1819",  # 2018-2019
    "1718",  # 2017-2018
    "1617",  # 2016-2017
    "1516",  # 2015-2016
]

# FDUK base URLs
FDUK_BASE_URL = "https://www.football-data.co.uk/mmz4281"
FDUK_EXTRA_URL = "https://www.football-data.co.uk/new"

# Minimum season year for extra leagues (filter older data)
EXTRA_MIN_SEASON_YEAR = 2023

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
LOGS_DIR = PROJECT_ROOT / "logs"
ALIASES_PATH = DATA_DIR / "fduk_team_aliases.json"
WAREHOUSE_PATH = DATA_DIR / "fduk_odds.duckdb"


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class FDUKMatch:
    """Parsed match from FDUK CSV."""
    division: str
    season_code: str
    date: str  # ISO format YYYY-MM-DD
    home_team: str
    away_team: str
    fthg: Optional[int]
    ftag: Optional[int]
    ftr: Optional[str]
    odds_home: Optional[float]
    odds_draw: Optional[float]
    odds_away: Optional[float]
    odds_provider: str  # B365, PS, Avg
    overround: Optional[float]
    p_home: Optional[float]  # de-vigged probability
    p_draw: Optional[float]
    p_away: Optional[float]


@dataclass
class MatchResult:
    """Result of attempting to match FDUK row to DB."""
    fduk_match: FDUKMatch
    db_match_id: Optional[int]
    confidence: float
    match_method: str  # exact, alias, fuzzy, none


# =============================================================================
# LOGGING SETUP
# =============================================================================

def setup_logging(log_path: Path) -> logging.Logger:
    """Configure logging to file and console."""
    logger = logging.getLogger("fduk_ingest")
    logger.setLevel(logging.INFO)

    # File handler
    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(message)s"))

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


# =============================================================================
# TEAM ALIASES
# =============================================================================

def load_team_aliases(path: Path) -> dict:
    """Load team aliases from JSON file."""
    if not path.exists():
        raise FileNotFoundError(f"Team aliases file not found: {path}")

    with open(path) as f:
        data = json.load(f)

    # Flatten to {fduk_name_lower: team_id}
    aliases = {}
    for league, mappings in data.items():
        if league.startswith("_"):
            continue  # Skip metadata
        for fduk_name, team_id in mappings.items():
            aliases[fduk_name.lower()] = team_id

    return aliases


# =============================================================================
# CSV PARSING
# =============================================================================

def parse_date(date_str: str) -> Optional[str]:
    """Parse FDUK date format (DD/MM/YYYY or DD/MM/YY) to ISO format."""
    if not date_str:
        return None

    # Try full year format first (DD/MM/YYYY)
    try:
        dt = datetime.strptime(date_str, "%d/%m/%Y")
        return dt.strftime("%Y-%m-%d")
    except ValueError:
        pass

    # Try 2-digit year format (DD/MM/YY)
    try:
        dt = datetime.strptime(date_str, "%d/%m/%y")
        return dt.strftime("%Y-%m-%d")
    except ValueError:
        pass

    return None


def parse_float(value: str) -> Optional[float]:
    """Safely parse float from string."""
    if not value or value.strip() == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def parse_int(value: str) -> Optional[int]:
    """Safely parse int from string."""
    if not value or value.strip() == "":
        return None
    try:
        return int(value)
    except ValueError:
        return None


def get_odds_and_provider(row: dict) -> tuple[Optional[float], Optional[float], Optional[float], str]:
    """
    Extract odds with provider priority: B365 > PS (Pinnacle) > Avg.
    Returns (home, draw, away, provider).
    """
    # Try Bet365 first
    b365h = parse_float(row.get("B365H", ""))
    b365d = parse_float(row.get("B365D", ""))
    b365a = parse_float(row.get("B365A", ""))
    if b365h and b365d and b365a and b365h > 1 and b365d > 1 and b365a > 1:
        return b365h, b365d, b365a, "B365"

    # Try Pinnacle
    psh = parse_float(row.get("PSH", ""))
    psd = parse_float(row.get("PSD", ""))
    psa = parse_float(row.get("PSA", ""))
    if psh and psd and psa and psh > 1 and psd > 1 and psa > 1:
        return psh, psd, psa, "PS"

    # Fallback to Average
    avgh = parse_float(row.get("AvgH", ""))
    avgd = parse_float(row.get("AvgD", ""))
    avga = parse_float(row.get("AvgA", ""))
    if avgh and avgd and avga and avgh > 1 and avgd > 1 and avga > 1:
        return avgh, avgd, avga, "Avg"

    return None, None, None, "none"


def get_odds_and_provider_extra(row: dict) -> tuple[Optional[float], Optional[float], Optional[float], str]:
    """
    Extract closing odds from extra-league CSV format.
    Priority: Pinnacle Closing > Avg Closing > B365 Closing > Max Closing.
    Column names use 'C' suffix: PSCH, PSCD, PSCA, AvgCH, etc.
    """
    # Try Pinnacle Closing first (best for ML)
    psh = parse_float(row.get("PSCH", ""))
    psd = parse_float(row.get("PSCD", ""))
    psa = parse_float(row.get("PSCA", ""))
    if psh and psd and psa and psh > 1 and psd > 1 and psa > 1:
        return psh, psd, psa, "PS"

    # Try Average Closing
    avgh = parse_float(row.get("AvgCH", ""))
    avgd = parse_float(row.get("AvgCD", ""))
    avga = parse_float(row.get("AvgCA", ""))
    if avgh and avgd and avga and avgh > 1 and avgd > 1 and avga > 1:
        return avgh, avgd, avga, "Avg"

    # Try B365 Closing
    b365h = parse_float(row.get("B365CH", ""))
    b365d = parse_float(row.get("B365CD", ""))
    b365a = parse_float(row.get("B365CA", ""))
    if b365h and b365d and b365a and b365h > 1 and b365d > 1 and b365a > 1:
        return b365h, b365d, b365a, "B365"

    # Try Max Closing
    maxh = parse_float(row.get("MaxCH", ""))
    maxd = parse_float(row.get("MaxCD", ""))
    maxa = parse_float(row.get("MaxCA", ""))
    if maxh and maxd and maxa and maxh > 1 and maxd > 1 and maxa > 1:
        return maxh, maxd, maxa, "Max"

    return None, None, None, "none"


def calculate_devig(odds_h: float, odds_d: float, odds_a: float) -> tuple[float, float, float, float]:
    """
    Calculate de-vigged probabilities and overround.
    Returns (p_home, p_draw, p_away, overround).
    """
    imp_h = 1.0 / odds_h
    imp_d = 1.0 / odds_d
    imp_a = 1.0 / odds_a
    overround = imp_h + imp_d + imp_a

    # Normalize to true probabilities
    p_home = imp_h / overround
    p_draw = imp_d / overround
    p_away = imp_a / overround

    return p_home, p_draw, p_away, overround


def parse_csv(content: str, division: str, season_code: str) -> list[FDUKMatch]:
    """Parse FDUK CSV content into FDUKMatch objects."""
    matches = []
    reader = csv.DictReader(StringIO(content))

    for row in reader:
        date_iso = parse_date(row.get("Date", ""))
        if not date_iso:
            continue

        home_team = row.get("HomeTeam", "").strip()
        away_team = row.get("AwayTeam", "").strip()
        if not home_team or not away_team:
            continue

        odds_h, odds_d, odds_a, provider = get_odds_and_provider(row)

        # Calculate de-vigged probabilities
        p_home, p_draw, p_away, overround = None, None, None, None
        if odds_h and odds_d and odds_a:
            p_home, p_draw, p_away, overround = calculate_devig(odds_h, odds_d, odds_a)

        match = FDUKMatch(
            division=division,
            season_code=season_code,
            date=date_iso,
            home_team=home_team,
            away_team=away_team,
            fthg=parse_int(row.get("FTHG", "")),
            ftag=parse_int(row.get("FTAG", "")),
            ftr=row.get("FTR", "").strip() or None,
            odds_home=odds_h,
            odds_draw=odds_d,
            odds_away=odds_a,
            odds_provider=provider,
            overround=overround,
            p_home=p_home,
            p_draw=p_draw,
            p_away=p_away,
        )
        matches.append(match)

    return matches


def parse_extra_csv(content: str, division: str, min_season_year: int = EXTRA_MIN_SEASON_YEAR) -> list[FDUKMatch]:
    """Parse FDUK extra-league CSV content (multi-season format).

    Extra leagues use different column names:
      Home/Away (not HomeTeam/AwayTeam), HG/AG (not FTHG/FTAG), Res (not FTR).
    Season field is "2023/2024" format — filtered by min_season_year.
    """
    matches = []
    reader = csv.DictReader(StringIO(content))

    for row in reader:
        # Season filter: supports "2023/2024" or "2023" (single year)
        season_str = row.get("Season", "").strip()
        if not season_str:
            continue

        try:
            if "/" in season_str:
                # "2023/2024" → start_year=2023, season_code="2324"
                start_year = int(season_str.split("/")[0])
                season_code = season_str.split("/")[0][-2:] + season_str.split("/")[1][-2:]
            else:
                # "2023" → start_year=2023, season_code="2324"
                start_year = int(season_str)
                season_code = f"{start_year % 100:02d}{(start_year + 1) % 100:02d}"
        except ValueError:
            continue

        if start_year < min_season_year:
            continue

        date_iso = parse_date(row.get("Date", ""))
        if not date_iso:
            continue

        home_team = row.get("Home", "").strip()
        away_team = row.get("Away", "").strip()
        if not home_team or not away_team:
            continue

        odds_h, odds_d, odds_a, provider = get_odds_and_provider_extra(row)

        # Calculate de-vigged probabilities
        p_home, p_draw, p_away, overround = None, None, None, None
        if odds_h and odds_d and odds_a:
            p_home, p_draw, p_away, overround = calculate_devig(odds_h, odds_d, odds_a)

        match = FDUKMatch(
            division=division,
            season_code=season_code,
            date=date_iso,
            home_team=home_team,
            away_team=away_team,
            fthg=parse_int(row.get("HG", "")),
            ftag=parse_int(row.get("AG", "")),
            ftr=row.get("Res", "").strip() or None,
            odds_home=odds_h,
            odds_draw=odds_d,
            odds_away=odds_a,
            odds_provider=provider,
            overround=overround,
            p_home=p_home,
            p_draw=p_draw,
            p_away=p_away,
        )
        matches.append(match)

    return matches


# =============================================================================
# DATA FETCHING
# =============================================================================

def fetch_csv(division: str, season_code: str, logger: logging.Logger) -> Optional[str]:
    """Fetch CSV from Football-Data UK."""
    url = f"{FDUK_BASE_URL}/{season_code}/{division}.csv"

    try:
        response = httpx.get(url, timeout=30, follow_redirects=True)
        if response.status_code == 200:
            logger.info(f"  Fetched {division}/{season_code}: {len(response.text)} bytes")
            return response.text
        elif response.status_code == 404:
            logger.warning(f"  Not found: {url}")
            return None
        else:
            logger.error(f"  HTTP {response.status_code}: {url}")
            return None
    except Exception as e:
        logger.error(f"  Error fetching {url}: {e}")
        return None


def fetch_extra_csv(division: str, logger: logging.Logger) -> Optional[str]:
    """Fetch multi-season CSV from Football-Data UK extra leagues."""
    url = f"{FDUK_EXTRA_URL}/{division}.csv"

    try:
        response = httpx.get(url, timeout=30, follow_redirects=True)
        if response.status_code == 200:
            logger.info(f"  Fetched extra {division}: {len(response.text)} bytes")
            return response.text
        elif response.status_code == 404:
            logger.warning(f"  Not found: {url}")
            return None
        else:
            logger.error(f"  HTTP {response.status_code}: {url}")
            return None
    except Exception as e:
        logger.error(f"  Error fetching {url}: {e}")
        return None


# =============================================================================
# LEAGUE ID HELPERS
# =============================================================================

def get_league_id_for_division(division: str) -> Optional[int]:
    """Get league_id from either main or extra division mapping."""
    return DIVISION_MAPPING.get(division) or EXTRA_LEAGUE_MAPPING.get(division)


# Provider column mapping for metadata (main vs extra)
PROVIDER_COLUMN_MAP_MAIN = {"B365": "B365H", "PS": "PSH", "Avg": "AvgH"}
PROVIDER_COLUMN_MAP_EXTRA = {"PS": "PSCH", "Avg": "AvgCH", "B365": "B365CH", "Max": "MaxCH"}


def get_odds_metadata(fduk_match: FDUKMatch) -> tuple[str, str, str]:
    """Return (source, kind, column) metadata for the match.

    Main leagues: pre-closing odds (PSH, B365H, AvgH)
    Extra leagues: closing odds (PSCH, AvgCH, B365CH, MaxCH)
    """
    provider = fduk_match.odds_provider
    is_extra = fduk_match.division in EXTRA_LEAGUE_MAPPING

    if is_extra:
        col = PROVIDER_COLUMN_MAP_EXTRA.get(provider, f"{provider}CH")
        source = f"football-data.co.uk ({provider})"
        kind = "closing"
    else:
        col = PROVIDER_COLUMN_MAP_MAIN.get(provider, f"{provider}H")
        source = f"football-data.co.uk ({provider})"
        kind = "proxy_pre_closing"

    return source, kind, col


# =============================================================================
# WAREHOUSE (DuckDB)
# =============================================================================

def init_warehouse(db_path) -> "duckdb.DuckDBPyConnection":
    """Initialize DuckDB warehouse with schema."""
    if not HAS_DUCKDB:
        raise ImportError("DuckDB is required for warehouse mode. Install with: pip install duckdb")

    conn = duckdb.connect(str(db_path) if db_path != ":memory:" else ":memory:")

    # Drop and recreate for clean state (warehouse is regenerated each run)
    conn.execute("DROP TABLE IF EXISTS fduk_matches")

    conn.execute("""
        CREATE TABLE fduk_matches (
            division VARCHAR,
            season_code VARCHAR,
            date DATE,
            home_team VARCHAR,
            away_team VARCHAR,
            fthg INTEGER,
            ftag INTEGER,
            ftr VARCHAR,
            odds_home DOUBLE,
            odds_draw DOUBLE,
            odds_away DOUBLE,
            odds_provider VARCHAR,
            overround DOUBLE,
            p_home DOUBLE,
            p_draw DOUBLE,
            p_away DOUBLE,
            ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (division, season_code, date, home_team, away_team)
        )
    """)

    return conn


def insert_to_warehouse(conn: "duckdb.DuckDBPyConnection", matches: list[FDUKMatch]) -> int:
    """Insert matches to warehouse, skipping duplicates."""
    inserted = 0
    for match in matches:
        try:
            conn.execute("""
                INSERT OR IGNORE INTO fduk_matches (
                    division, season_code, date, home_team, away_team,
                    fthg, ftag, ftr, odds_home, odds_draw, odds_away,
                    odds_provider, overround, p_home, p_draw, p_away
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                match.division, match.season_code, match.date,
                match.home_team, match.away_team,
                match.fthg, match.ftag, match.ftr,
                match.odds_home, match.odds_draw, match.odds_away,
                match.odds_provider, match.overround,
                match.p_home, match.p_draw, match.p_away
            ])
            inserted += 1
        except Exception as e:
            # Duplicate or other error, skip
            pass

    return inserted


# =============================================================================
# MATCHING LOGIC
# =============================================================================

def normalize_team_name(name: str) -> str:
    """Normalize team name for matching."""
    return name.lower().strip()


def fuzzy_match_score(s1: str, s2: str) -> float:
    """Simple fuzzy match using Levenshtein-like ratio."""
    s1, s2 = s1.lower(), s2.lower()
    if s1 == s2:
        return 1.0

    # Check if one contains the other
    if s1 in s2 or s2 in s1:
        return 0.9

    # Simple character overlap
    set1, set2 = set(s1), set(s2)
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0.0


# =============================================================================
# POSTGRES BACKFILL
# =============================================================================

async def get_db_teams(conn) -> dict:
    """Fetch all teams from database."""
    rows = await conn.fetch("SELECT id, name FROM teams")
    return {row["name"].lower(): row["id"] for row in rows}


async def get_db_matches_for_league_date(
    conn, league_id: int, date_str: str, tolerance_days: int = 1
) -> list[dict]:
    """Fetch matches from DB for a league within date tolerance."""
    date_obj = datetime.strptime(date_str, "%Y-%m-%d").date()
    date_min = date_obj - timedelta(days=tolerance_days)
    date_max = date_obj + timedelta(days=tolerance_days)

    rows = await conn.fetch("""
        SELECT m.id, m.date, m.home_team_id, m.away_team_id,
               m.opening_odds_home, m.opening_odds_draw, m.opening_odds_away,
               th.name as home_name, ta.name as away_name
        FROM matches m
        JOIN teams th ON th.id = m.home_team_id
        JOIN teams ta ON ta.id = m.away_team_id
        WHERE m.league_id = $1
          AND m.date::date >= $2
          AND m.date::date <= $3
          AND m.status = 'FT'
    """, league_id, date_min, date_max)

    return [dict(row) for row in rows]


def match_fduk_to_db(
    fduk_match: FDUKMatch,
    db_matches: list[dict],
    aliases: dict,
    fuzzy_threshold: float
) -> MatchResult:
    """
    Attempt to match FDUK match to DB match.
    Returns MatchResult with confidence and method.
    """
    fduk_home = normalize_team_name(fduk_match.home_team)
    fduk_away = normalize_team_name(fduk_match.away_team)

    # Get team IDs from aliases
    home_id = aliases.get(fduk_home)
    away_id = aliases.get(fduk_away)

    best_match = None
    best_confidence = 0.0
    best_method = "none"

    for db_match in db_matches:
        db_home = normalize_team_name(db_match["home_name"])
        db_away = normalize_team_name(db_match["away_name"])

        # Method 1: Exact alias match
        if home_id and away_id:
            if db_match["home_team_id"] == home_id and db_match["away_team_id"] == away_id:
                return MatchResult(
                    fduk_match=fduk_match,
                    db_match_id=db_match["id"],
                    confidence=1.0,
                    match_method="alias"
                )

        # Method 2: Exact name match
        if fduk_home == db_home and fduk_away == db_away:
            return MatchResult(
                fduk_match=fduk_match,
                db_match_id=db_match["id"],
                confidence=1.0,
                match_method="exact"
            )

        # Method 3: Fuzzy match
        home_score = fuzzy_match_score(fduk_home, db_home)
        away_score = fuzzy_match_score(fduk_away, db_away)
        combined_score = (home_score + away_score) / 2

        if combined_score > best_confidence:
            best_confidence = combined_score
            best_match = db_match
            best_method = "fuzzy"

    if best_confidence >= fuzzy_threshold and best_match:
        return MatchResult(
            fduk_match=fduk_match,
            db_match_id=best_match["id"],
            confidence=best_confidence,
            match_method=best_method
        )

    return MatchResult(
        fduk_match=fduk_match,
        db_match_id=None,
        confidence=best_confidence,
        match_method="none"
    )


async def backfill_postgres(
    matches: list[FDUKMatch],
    database_url: str,
    aliases: dict,
    min_confidence: float,
    fuzzy_threshold: float,
    dry_run: bool,
    logger: logging.Logger
) -> dict:
    """
    Backfill opening_odds_* to Postgres matches table.
    Only updates rows where opening_odds_* are NULL.
    """
    if not HAS_ASYNCPG:
        raise ImportError("asyncpg is required for Postgres backfill. Install with: pip install asyncpg")

    conn = await asyncpg.connect(database_url)

    stats = {
        "processed": 0,
        "matched_high_conf": 0,
        "matched_fuzzy": 0,
        "unmatched": 0,
        "updated": 0,
        "skipped_has_odds": 0,
        "unmatched_teams": {},
    }

    try:
        # Group matches by league (both main + extra divisions)
        by_league = {}
        for m in matches:
            league_id = get_league_id_for_division(m.division)
            if league_id:
                by_league.setdefault(league_id, []).append(m)

        for league_id, league_matches in by_league.items():
            league_name = LEAGUE_NAMES.get(league_id, str(league_id))
            logger.info(f"  Processing {league_name}: {len(league_matches)} matches")

            for fduk_match in league_matches:
                stats["processed"] += 1

                # Skip if no odds
                if not fduk_match.odds_home:
                    continue

                # Get candidate DB matches
                db_matches = await get_db_matches_for_league_date(
                    conn, league_id, fduk_match.date
                )

                # Try to match
                result = match_fduk_to_db(fduk_match, db_matches, aliases, fuzzy_threshold)

                if result.db_match_id is None:
                    stats["unmatched"] += 1
                    key = f"{fduk_match.home_team} vs {fduk_match.away_team}"
                    stats["unmatched_teams"][key] = stats["unmatched_teams"].get(key, 0) + 1
                    continue

                if result.confidence < min_confidence:
                    stats["unmatched"] += 1
                    continue

                if result.match_method == "fuzzy":
                    stats["matched_fuzzy"] += 1
                else:
                    stats["matched_high_conf"] += 1

                # Check if already has odds
                db_match = next((m for m in db_matches if m["id"] == result.db_match_id), None)
                if db_match and db_match.get("opening_odds_home") is not None:
                    stats["skipped_has_odds"] += 1
                    continue

                # Build metadata
                source, kind, col = get_odds_metadata(fduk_match)
                match_date = datetime.strptime(fduk_match.date, "%Y-%m-%d")

                # Update
                if not dry_run:
                    await conn.execute("""
                        UPDATE matches
                        SET opening_odds_home = $1,
                            opening_odds_draw = $2,
                            opening_odds_away = $3,
                            opening_odds_source = $5,
                            opening_odds_kind = $6,
                            opening_odds_column = $7,
                            opening_odds_recorded_at = $8,
                            opening_odds_recorded_at_type = 'file_asof'
                        WHERE id = $4
                          AND opening_odds_home IS NULL
                    """, fduk_match.odds_home, fduk_match.odds_draw, fduk_match.odds_away,
                    result.db_match_id, source, kind, col, match_date)

                stats["updated"] += 1

    finally:
        await conn.close()

    return stats


# =============================================================================
# REPORTING
# =============================================================================

def generate_coverage_report(conn: "duckdb.DuckDBPyConnection") -> dict:
    """Generate coverage report from warehouse."""
    report = {
        "generated_at": datetime.now().isoformat(),
        "by_league_season": [],
        "by_provider": {},
        "overround_stats": {},
        "totals": {},
    }

    # Coverage by league/season
    rows = conn.execute("""
        SELECT
            division,
            season_code,
            COUNT(*) as total,
            COUNT(odds_home) as with_odds,
            ROUND(100.0 * COUNT(odds_home) / COUNT(*), 1) as pct_valid,
            MIN(date) as min_date,
            MAX(date) as max_date
        FROM fduk_matches
        GROUP BY division, season_code
        ORDER BY division, season_code DESC
    """).fetchall()

    for row in rows:
        report["by_league_season"].append({
            "division": row[0],
            "league": LEAGUE_NAMES.get(get_league_id_for_division(row[0]), row[0]),
            "season": row[1],
            "total": row[2],
            "with_odds": row[3],
            "pct_valid": row[4],
            "min_date": str(row[5]),
            "max_date": str(row[6]),
        })

    # Provider distribution
    rows = conn.execute("""
        SELECT
            division,
            odds_provider,
            COUNT(*) as count
        FROM fduk_matches
        WHERE odds_provider != 'none'
        GROUP BY division, odds_provider
    """).fetchall()

    for row in rows:
        div = row[0]
        if div not in report["by_provider"]:
            report["by_provider"][div] = {}
        report["by_provider"][div][row[1]] = row[2]

    # Overround stats
    rows = conn.execute("""
        SELECT
            division,
            PERCENTILE_CONT(0.10) WITHIN GROUP (ORDER BY overround) as p10,
            PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY overround) as p50,
            PERCENTILE_CONT(0.90) WITHIN GROUP (ORDER BY overround) as p90,
            COUNT(*) FILTER (WHERE overround > 1.12) as high_overround
        FROM fduk_matches
        WHERE overround IS NOT NULL
        GROUP BY division
    """).fetchall()

    for row in rows:
        report["overround_stats"][row[0]] = {
            "p10": round(row[1], 4) if row[1] else None,
            "p50": round(row[2], 4) if row[2] else None,
            "p90": round(row[3], 4) if row[3] else None,
            "count_over_1.12": row[4],
        }

    # Totals
    row = conn.execute("""
        SELECT
            COUNT(*) as total,
            COUNT(odds_home) as with_odds,
            ROUND(100.0 * COUNT(odds_home) / COUNT(*), 1) as pct
        FROM fduk_matches
    """).fetchone()

    report["totals"] = {
        "total_matches": row[0],
        "with_odds": row[1],
        "pct_valid": row[2],
    }

    return report


def generate_unmatched_report(stats: dict) -> dict:
    """Generate unmatched teams report."""
    sorted_teams = sorted(
        stats.get("unmatched_teams", {}).items(),
        key=lambda x: x[1],
        reverse=True
    )

    return {
        "generated_at": datetime.now().isoformat(),
        "total_unmatched": stats.get("unmatched", 0),
        "unique_unmatched_fixtures": len(sorted_teams),
        "top_unmatched": sorted_teams[:50],
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Football-Data UK Odds Ingestion")
    parser.add_argument("--warehouse-only", action="store_true", default=True,
                        help="Only build local warehouse (default)")
    parser.add_argument("--backfill-postgres", action="store_true",
                        help="Enable Postgres backfill (requires DATABASE_URL)")
    parser.add_argument("--dry-run", action="store_true", default=True,
                        help="Dry run for Postgres backfill (default)")
    parser.add_argument("--no-dry-run", action="store_true",
                        help="Actually write to Postgres")
    parser.add_argument("--min-match-confidence", type=float, default=0.90,
                        help="Minimum confidence for Postgres backfill")
    parser.add_argument("--fuzzy-threshold", type=float, default=0.85,
                        help="Fuzzy matching threshold")
    parser.add_argument("--seasons", nargs="+", default=SEASONS,
                        help="Seasons to process (main leagues only)")
    parser.add_argument("--extra-leagues", action="store_true",
                        help="Also ingest extra leagues (ARG, BRA, MEX, USA)")
    parser.add_argument("--only-extra", action="store_true",
                        help="Only ingest extra leagues (skip main)")
    parser.add_argument("--leagues", nargs="+", default=None,
                        help="Filter to specific division codes (e.g., ARG BRA E0)")
    parser.add_argument("--in-memory", action="store_true",
                        help="Use in-memory DuckDB (allows parallel runs)")
    args = parser.parse_args()

    # Resolve dry-run flag
    dry_run = not args.no_dry_run

    # Setup paths
    LOGS_DIR.mkdir(exist_ok=True)
    DATA_DIR.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOGS_DIR / f"fduk_ingest_{timestamp}.log"

    logger = setup_logging(log_path)

    logger.info("=" * 60)
    logger.info("FOOTBALL-DATA UK INGESTION PIPELINE")
    logger.info("=" * 60)
    logger.info(f"Mode: {'Warehouse + Backfill' if args.backfill_postgres else 'Warehouse only'}")
    logger.info(f"Dry run: {dry_run}")
    logger.info(f"Seasons: {args.seasons}")
    logger.info(f"Extra leagues: {args.extra_leagues or args.only_extra}")
    logger.info(f"League filter: {args.leagues or 'all'}")
    logger.info(f"Log: {log_path}")

    # Load aliases
    logger.info(f"\nLoading team aliases from {ALIASES_PATH}")
    aliases = load_team_aliases(ALIASES_PATH)
    logger.info(f"Loaded {len(aliases)} team aliases")

    # Initialize warehouse
    wh_path = ":memory:" if args.in_memory else WAREHOUSE_PATH
    logger.info(f"\nInitializing warehouse: {wh_path}")
    warehouse = init_warehouse(Path(wh_path) if wh_path != ":memory:" else wh_path)

    # Fetch and parse all data
    all_matches = []
    league_filter = set(args.leagues) if args.leagues else None

    # ── Main leagues (per-season CSVs) ──
    if not args.only_extra:
        for division, league_id in DIVISION_MAPPING.items():
            if league_filter and division not in league_filter:
                continue
            league_name = LEAGUE_NAMES.get(league_id, division)
            logger.info(f"\n{'=' * 50}")
            logger.info(f"Processing {league_name} ({division})")

            for season in args.seasons:
                csv_content = fetch_csv(division, season, logger)
                if csv_content:
                    matches = parse_csv(csv_content, division, season)
                    logger.info(f"    {season}: {len(matches)} matches parsed")

                    inserted = insert_to_warehouse(warehouse, matches)
                    logger.info(f"    {season}: {inserted} inserted to warehouse")

                    all_matches.extend(matches)

    # ── Extra leagues (multi-season CSVs) ──
    if args.extra_leagues or args.only_extra:
        for division, league_id in EXTRA_LEAGUE_MAPPING.items():
            if league_filter and division not in league_filter:
                continue
            league_name = LEAGUE_NAMES.get(league_id, division)
            logger.info(f"\n{'=' * 50}")
            logger.info(f"Processing EXTRA {league_name} ({division})")

            csv_content = fetch_extra_csv(division, logger)
            if csv_content:
                matches = parse_extra_csv(csv_content, division)
                logger.info(f"    {len(matches)} matches parsed (season >= {EXTRA_MIN_SEASON_YEAR})")

                inserted = insert_to_warehouse(warehouse, matches)
                logger.info(f"    {inserted} inserted to warehouse")

                all_matches.extend(matches)

    warehouse.commit()

    # Generate coverage report
    logger.info(f"\n{'=' * 50}")
    logger.info("Generating coverage report...")
    coverage = generate_coverage_report(warehouse)
    coverage_path = LOGS_DIR / f"fduk_coverage_report_{timestamp}.json"
    with open(coverage_path, "w") as f:
        json.dump(coverage, f, indent=2)
    logger.info(f"Coverage report: {coverage_path}")

    # Print summary
    logger.info(f"\nTOTALS:")
    logger.info(f"  Total matches: {coverage['totals']['total_matches']}")
    logger.info(f"  With odds: {coverage['totals']['with_odds']} ({coverage['totals']['pct_valid']}%)")

    # Postgres backfill
    backfill_stats = None
    if args.backfill_postgres:
        database_url = os.environ.get("DATABASE_URL")
        if not database_url:
            logger.error("DATABASE_URL not set. Cannot backfill to Postgres.")
            sys.exit(1)

        logger.info(f"\n{'=' * 50}")
        logger.info(f"POSTGRES BACKFILL {'(DRY RUN)' if dry_run else '(LIVE)'}")
        logger.info(f"Min confidence: {args.min_match_confidence}")
        logger.info(f"Fuzzy threshold: {args.fuzzy_threshold}")

        # Filter to matches with odds
        matches_with_odds = [m for m in all_matches if m.odds_home]
        logger.info(f"Matches with odds to process: {len(matches_with_odds)}")

        backfill_stats = asyncio.run(backfill_postgres(
            matches_with_odds,
            database_url,
            aliases,
            args.min_match_confidence,
            args.fuzzy_threshold,
            dry_run,
            logger
        ))

        # Save backfill summary
        backfill_path = LOGS_DIR / f"fduk_postgres_backfill_summary_{timestamp}.json"
        with open(backfill_path, "w") as f:
            json.dump(backfill_stats, f, indent=2)
        logger.info(f"Backfill summary: {backfill_path}")

        # Generate unmatched report
        unmatched_report = generate_unmatched_report(backfill_stats)
        unmatched_path = LOGS_DIR / f"fduk_unmatched_teams_{timestamp}.json"
        with open(unmatched_path, "w") as f:
            json.dump(unmatched_report, f, indent=2)
        logger.info(f"Unmatched report: {unmatched_path}")

        logger.info(f"\nBACKFILL STATS:")
        logger.info(f"  Processed: {backfill_stats['processed']}")
        logger.info(f"  Matched (high conf): {backfill_stats['matched_high_conf']}")
        logger.info(f"  Matched (fuzzy): {backfill_stats['matched_fuzzy']}")
        logger.info(f"  Unmatched: {backfill_stats['unmatched']}")
        logger.info(f"  Updated: {backfill_stats['updated']}")
        logger.info(f"  Skipped (has odds): {backfill_stats['skipped_has_odds']}")

    warehouse.close()

    logger.info(f"\n{'=' * 60}")
    logger.info("INGESTION COMPLETE")
    logger.info(f"{'=' * 60}")
    logger.info(f"Warehouse: {WAREHOUSE_PATH}")
    logger.info(f"Log: {log_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
