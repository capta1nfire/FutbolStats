#!/usr/bin/env python3
from __future__ import annotations
"""
Historical backfill of Understat xG data for top-5 EUR leagues.

FAST approach: uses /getLeagueData (1 req/season) which includes xG per match.
No need for individual /getMatchData calls (~3,700 reqs → ~12 reqs total).

Two tables populated:
  - match_external_refs (source='understat'): links our match_id to Understat ID
  - match_understat_team: xG data per match

Usage:
    source .env
    # Dry-run France (see what would be linked)
    python scripts/backfill_understat_historical.py --league-id 61 --dry-run

    # Execute France
    python scripts/backfill_understat_historical.py --league-id 61

    # Single season
    python scripts/backfill_understat_historical.py --league-id 61 --season 2020

    # All top-5 EUR dry-run
    python scripts/backfill_understat_historical.py --all --dry-run
"""

import argparse
import logging
import os
import sys
import time
import unicodedata
import re
from collections import defaultdict
from datetime import datetime, timedelta

import httpx
import psycopg2
import psycopg2.extras

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(message)s",
)
log = logging.getLogger("understat_historical")

# ── Understat config ──────────────────────────────────────────────────────────

UNDERSTAT_BASE = "https://understat.com"
RATE_LIMIT_SEC = 1.5  # polite rate limiting

# API-Football league_id → (Understat name, first season available)
LEAGUE_CONFIG = {
    39:  ("EPL",        2014),
    61:  ("Ligue_1",    2014),
    78:  ("Bundesliga", 2014),
    135: ("Serie_A",    2014),
    140: ("La_Liga",    2014),
}

# Manual name overrides: Understat name → normalized form matching our DB
# Only add entries where automatic matching fails
UNDERSTAT_NAME_OVERRIDES = {
    # France
    "Paris Saint Germain": "paris saint germain",
    "Saint-Etienne": "saint etienne",
    "Stade Rennais": "rennes",
    # Germany
    "Eintracht Frankfurt": "eintracht frankfurt",
    "Bayern Munich": "bayern munchen",
    "RasenBallsport Leipzig": "rb leipzig",
    "RB Leipzig": "rb leipzig",
    "Borussia M.Gladbach": "borussia monchengladbach",
    "Arminia Bielefeld": "arminia bielefeld",
    "FC Cologne": "1 koln",
    "Hertha Berlin": "hertha bsc",
    "Fortuna Duesseldorf": "fortuna dusseldorf",
    "Nuernberg": "1 nurnberg",
    "Greuther Fuerth": "spvgg greuther furth",
    # England
    "Manchester United": "manchester united",
    "Manchester City": "manchester city",
    "Wolverhampton Wanderers": "wolves",
    "Wolves": "wolves",
    "Sheffield United": "sheffield utd",
    "Sheffield Utd": "sheffield utd",
    # Spain
    "Atletico Madrid": "atletico madrid",
    "Athletic Club": "athletic club",
    "Celta Vigo": "celta vigo",
    # Italy
    "AC Milan": "ac milan",
    "Inter": "inter",
    "Hellas Verona": "hellas verona",
    "Verona": "hellas verona",
    "Parma Calcio 1913": "parma",
    "SPAL 2013": "spal",
}


# ── Name normalization (standalone, no imports needed) ────────────────────────

def _normalize(name: str) -> str:
    """Normalize team name for fuzzy matching."""
    if not name:
        return ""
    # Override first
    if name in UNDERSTAT_NAME_OVERRIDES:
        return UNDERSTAT_NAME_OVERRIDES[name]
    s = name.lower().strip()
    # Strip diacritics
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    # Replace punctuation with space
    s = re.sub(r"[.\-/&',()]", " ", s)
    # Remove juridical tokens
    for token in ("fc", "cf", "sc", "ac", "as", "ss", "us", "rc", "ca", "cd", "ssc"):
        s = re.sub(rf"\b{token}\b", "", s)
    # Collapse whitespace
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _names_match(name_a: str, name_b: str) -> bool:
    """Check if two team names match (normalized, with substring fallback)."""
    a = _normalize(name_a)
    b = _normalize(name_b)
    if not a or not b:
        return False
    if a == b:
        return True
    # Substring match (either direction)
    if len(a) >= 4 and len(b) >= 4:
        if a in b or b in a:
            return True
    return False


# ── Understat API ─────────────────────────────────────────────────────────────

def fetch_league_season(client: httpx.Client, league: str, season: int) -> list[dict]:
    """Fetch all matches for a league/season from Understat's internal API."""
    url = f"{UNDERSTAT_BASE}/getLeagueData/{league}/{season}"
    headers = {
        "X-Requested-With": "XMLHttpRequest",
        "Referer": f"{UNDERSTAT_BASE}/league/{league}/{season}",
    }

    for attempt in range(3):
        try:
            time.sleep(RATE_LIMIT_SEC)
            resp = client.get(url, headers=headers, timeout=30.0)

            if resp.status_code == 404:
                log.warning(f"  Season {season} not found (404)")
                return []
            if resp.status_code == 429:
                wait = 5 * (attempt + 1)
                log.warning(f"  Rate limited, waiting {wait}s...")
                time.sleep(wait)
                continue
            if resp.status_code >= 500:
                wait = 3 * (attempt + 1)
                log.warning(f"  Server error {resp.status_code}, retry in {wait}s...")
                time.sleep(wait)
                continue

            resp.raise_for_status()
            data = resp.json()

            dates = data.get("dates", [])
            matches = []
            for m in dates:
                if not m.get("isResult"):
                    continue
                xg = m.get("xG", {})
                xg_h = xg.get("h")
                xg_a = xg.get("a")
                # Skip matches without xG
                if xg_h is None or xg_a is None:
                    continue
                try:
                    matches.append({
                        "id": str(m["id"]),
                        "datetime": m.get("datetime", ""),
                        "home_team": m.get("h", {}).get("title", ""),
                        "away_team": m.get("a", {}).get("title", ""),
                        "home_goals": m.get("goals", {}).get("h"),
                        "away_goals": m.get("goals", {}).get("a"),
                        "xg_home": float(xg_h),
                        "xg_away": float(xg_a),
                    })
                except (ValueError, TypeError, KeyError) as e:
                    log.debug(f"  Skip match {m.get('id')}: {e}")
                    continue

            return matches

        except httpx.TimeoutException:
            log.warning(f"  Timeout for {league}/{season}, attempt {attempt + 1}")
            time.sleep(3)
        except Exception as e:
            log.error(f"  Error fetching {league}/{season}: {e}")
            time.sleep(3)

    return []


# ── DB helpers ────────────────────────────────────────────────────────────────

def load_db_matches(conn, league_id: int) -> list[dict]:
    """Load all FT matches for a league from DB."""
    sql = """
        SELECT m.id, m.season, m.date,
               t_home.name AS home_team, t_away.name AS away_team,
               m.home_goals, m.away_goals
        FROM matches m
        JOIN teams t_home ON m.home_team_id = t_home.id
        JOIN teams t_away ON m.away_team_id = t_away.id
        WHERE m.league_id = %s
          AND m.status IN ('FT', 'AET', 'PEN')
          AND m.home_goals IS NOT NULL
        ORDER BY m.date
    """
    with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
        cur.execute(sql, (league_id,))
        return [dict(r) for r in cur.fetchall()]


def load_existing_refs(conn, league_id: int) -> set[int]:
    """Load match_ids that already have Understat refs."""
    sql = """
        SELECT mer.match_id
        FROM match_external_refs mer
        JOIN matches m ON m.id = mer.match_id
        WHERE mer.source = 'understat'
          AND m.league_id = %s
    """
    with conn.cursor() as cur:
        cur.execute(sql, (league_id,))
        return {r[0] for r in cur.fetchall()}


def load_existing_xg(conn, league_id: int) -> set[int]:
    """Load match_ids that already have Understat xG."""
    sql = """
        SELECT mut.match_id
        FROM match_understat_team mut
        JOIN matches m ON m.id = mut.match_id
        WHERE m.league_id = %s
    """
    with conn.cursor() as cur:
        cur.execute(sql, (league_id,))
        return {r[0] for r in cur.fetchall()}


# ── Matching engine ───────────────────────────────────────────────────────────

def build_match_index(db_matches: list[dict]) -> dict:
    """Build index: (date_str, norm_home) → [match_dicts]."""
    index = defaultdict(list)
    for m in db_matches:
        dt = m["date"]
        if dt is None:
            continue
        date_str = dt.strftime("%Y-%m-%d")
        norm_home = _normalize(m["home_team"])
        index[(date_str, norm_home)].append(m)
    return index


def find_db_match(
    u_match: dict,
    match_index: dict,
    db_matches: list[dict],
) -> dict | None:
    """Find the DB match corresponding to an Understat match."""
    u_dt_str = u_match["datetime"]
    if not u_dt_str:
        return None

    try:
        u_dt = datetime.strptime(u_dt_str, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return None

    u_home = u_match["home_team"]
    u_away = u_match["away_team"]
    u_date_str = u_dt.strftime("%Y-%m-%d")

    # Try exact date, ±1 day
    for delta in (0, -1, 1):
        check_date = (u_dt + timedelta(days=delta)).strftime("%Y-%m-%d")
        # Try direct index lookup with normalized Understat home team
        u_home_norm = _normalize(u_home)
        candidates = match_index.get((check_date, u_home_norm), [])

        for c in candidates:
            if _names_match(u_away, c["away_team"]):
                return c

        # Fallback: scan all matches for this date (handles name normalization edge cases)
        for key, matches in match_index.items():
            if key[0] != check_date:
                continue
            for c in matches:
                if _names_match(u_home, c["home_team"]) and _names_match(u_away, c["away_team"]):
                    return c

    return None


# ── Main ──────────────────────────────────────────────────────────────────────

def run_backfill(
    league_id: int,
    seasons: list[int] | None = None,
    dry_run: bool = False,
):
    """Run historical Understat backfill for a league."""
    if league_id not in LEAGUE_CONFIG:
        log.error(f"League {league_id} not supported. Available: {list(LEAGUE_CONFIG.keys())}")
        return

    league_name, first_season = LEAGUE_CONFIG[league_id]
    current_year = datetime.utcnow().year
    if seasons is None:
        seasons = list(range(first_season, current_year + 1))

    log.info(f"{'[DRY-RUN] ' if dry_run else ''}Backfill {league_name} (league_id={league_id})")
    log.info(f"Seasons: {seasons[0]}-{seasons[-1]}")

    # Connect to DB
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        log.error("DATABASE_URL not set. Run: source .env")
        return
    conn = psycopg2.connect(db_url)
    conn.autocommit = False

    try:
        # Load all DB matches for this league
        db_matches = load_db_matches(conn, league_id)
        existing_refs = load_existing_refs(conn, league_id)
        existing_xg = load_existing_xg(conn, league_id)
        log.info(f"DB matches: {len(db_matches)} FT | Existing refs: {len(existing_refs)} | Existing xG: {len(existing_xg)}")

        # Build index for fast matching
        match_index = build_match_index(db_matches)

        # HTTP client
        client = httpx.Client(
            headers={
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                              "AppleWebKit/537.36 (KHTML, like Gecko) "
                              "Chrome/120.0.0.0 Safari/537.36",
                "Accept": "application/json, text/html, */*",
                "Accept-Language": "en-US,en;q=0.5",
            },
            follow_redirects=True,
        )

        # Totals
        total_fetched = 0
        total_matched = 0
        total_refs_new = 0
        total_xg_new = 0
        total_skipped_existing = 0
        total_unmatched = 0
        unmatched_examples = []

        for season in seasons:
            log.info(f"\n{'='*60}")
            log.info(f"Season {season}-{season+1} ({league_name})")
            log.info(f"{'='*60}")

            u_matches = fetch_league_season(client, league_name, season)
            if not u_matches:
                log.warning(f"  No matches returned for {season}")
                continue

            total_fetched += len(u_matches)
            season_matched = 0
            season_unmatched = 0
            season_refs_new = 0
            season_xg_new = 0

            refs_to_insert = []
            xg_to_insert = []

            for u_match in u_matches:
                db_match = find_db_match(u_match, match_index, db_matches)

                if db_match is None:
                    season_unmatched += 1
                    total_unmatched += 1
                    if len(unmatched_examples) < 20:
                        unmatched_examples.append(
                            f"  {u_match['datetime']} {u_match['home_team']} vs {u_match['away_team']} "
                            f"({u_match['home_goals']}-{u_match['away_goals']})"
                        )
                    continue

                season_matched += 1
                total_matched += 1
                match_id = db_match["id"]

                # Ref
                if match_id not in existing_refs:
                    refs_to_insert.append((
                        match_id,
                        "understat",
                        u_match["id"],
                        1.0,  # confidence
                        "historical_backfill+league_data",
                        datetime.utcnow(),
                    ))
                    existing_refs.add(match_id)  # prevent duplicates within run
                    season_refs_new += 1
                    total_refs_new += 1

                # xG
                if match_id not in existing_xg:
                    xg_h = u_match["xg_home"]
                    xg_a = u_match["xg_away"]
                    xg_to_insert.append((
                        match_id,
                        xg_h,
                        xg_a,
                        None,   # xpts_home
                        None,   # xpts_away
                        None,   # npxg_home (not available from league data)
                        None,   # npxg_away
                        xg_a,   # xga_home = xg_away
                        xg_h,   # xga_away = xg_home
                        datetime.utcnow(),
                        "understat_league_data_v1",
                    ))
                    existing_xg.add(match_id)
                    season_xg_new += 1
                    total_xg_new += 1
                else:
                    total_skipped_existing += 1

            # Write to DB
            if not dry_run and (refs_to_insert or xg_to_insert):
                with conn.cursor() as cur:
                    if refs_to_insert:
                        psycopg2.extras.execute_values(
                            cur,
                            """INSERT INTO match_external_refs
                               (match_id, source, source_match_id, confidence, matched_by, created_at)
                               VALUES %s
                               ON CONFLICT (match_id, source) DO NOTHING""",
                            refs_to_insert,
                        )

                    if xg_to_insert:
                        psycopg2.extras.execute_values(
                            cur,
                            """INSERT INTO match_understat_team
                               (match_id, xg_home, xg_away, xpts_home, xpts_away,
                                npxg_home, npxg_away, xga_home, xga_away,
                                captured_at, source_version)
                               VALUES %s
                               ON CONFLICT (match_id) DO NOTHING""",
                            xg_to_insert,
                        )

                conn.commit()

            log.info(f"  Understat: {len(u_matches)} matches | Matched: {season_matched} | "
                     f"Unmatched: {season_unmatched}")
            log.info(f"  New refs: {season_refs_new} | New xG: {season_xg_new}")

        # Summary
        log.info(f"\n{'='*60}")
        log.info(f"SUMMARY: {league_name} (league_id={league_id})")
        log.info(f"{'='*60}")
        log.info(f"  Understat matches fetched:  {total_fetched}")
        log.info(f"  Matched to DB:              {total_matched} ({100*total_matched/max(total_fetched,1):.1f}%)")
        log.info(f"  Unmatched:                  {total_unmatched}")
        log.info(f"  New refs inserted:          {total_refs_new}")
        log.info(f"  New xG inserted:            {total_xg_new}")
        log.info(f"  Skipped (existing):         {total_skipped_existing}")
        if dry_run:
            log.info(f"  [DRY-RUN] No data written to DB")
        log.info(f"{'='*60}")

        if unmatched_examples:
            log.info(f"\nUnmatched examples (first {len(unmatched_examples)}):")
            for ex in unmatched_examples:
                log.info(ex)

        # Post-verification
        if not dry_run and total_xg_new > 0:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT COUNT(*)
                    FROM match_understat_team mut
                    JOIN matches m ON m.id = mut.match_id
                    WHERE m.league_id = %s
                """, (league_id,))
                total_xg = cur.fetchone()[0]

                cur.execute("""
                    SELECT COUNT(*)
                    FROM matches m
                    WHERE m.league_id = %s AND m.status IN ('FT', 'AET', 'PEN')
                """, (league_id,))
                total_ft = cur.fetchone()[0]

            log.info(f"\nPost-verification:")
            log.info(f"  Total xG: {total_xg}/{total_ft} ({100*total_xg/max(total_ft,1):.1f}%)")

    except Exception as e:
        conn.rollback()
        log.error(f"Fatal error: {e}", exc_info=True)
        raise

    finally:
        client.close()
        conn.close()


def main():
    parser = argparse.ArgumentParser(
        description="Historical Understat xG backfill for top-5 EUR leagues"
    )
    parser.add_argument("--league-id", type=int, help="API-Football league ID (39, 61, 78, 135, 140)")
    parser.add_argument("--season", type=int, help="Single season to backfill (e.g. 2020)")
    parser.add_argument("--all", action="store_true", help="Backfill all 5 leagues")
    parser.add_argument("--dry-run", action="store_true", help="Match only, don't write to DB")
    args = parser.parse_args()

    if not args.league_id and not args.all:
        parser.error("Specify --league-id or --all")

    seasons = [args.season] if args.season else None

    if args.all:
        for lid in LEAGUE_CONFIG:
            run_backfill(lid, seasons=seasons, dry_run=args.dry_run)
    else:
        run_backfill(args.league_id, seasons=seasons, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
