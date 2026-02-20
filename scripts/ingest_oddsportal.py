#!/usr/bin/env python3
from __future__ import annotations
"""
OddsPortal Odds Ingestion Pipeline

Ingests historical closing odds from OddsPortal (via OddsHarvester JSON output)
into matches.opening_odds_* columns.

Modes:
  --dry-run     Report what would be updated without writing to DB (default)
  --no-dry-run  Actually write odds to DB

Usage:
    # Dry-run for Colombia
    python scripts/ingest_oddsportal.py --section ColombiaPrimeraA --dry-run

    # Real backfill
    python scripts/ingest_oddsportal.py --section ColombiaPrimeraA --no-dry-run

    # Process specific files
    python scripts/ingest_oddsportal.py --section ColombiaPrimeraA \
        --files data/oddsportal_raw/colombia-primera-a_2024.json

RESTRICTIONS:
    - Only updates matches.opening_odds_* WHERE opening_odds_home IS NULL
    - Never overwrites existing odds (FDUK or live pipeline)
    - GATE: aborts if any team name cannot be resolved via aliases
    - Score validation: rejects matches with score mismatch vs DB
"""

import argparse
import asyncio
import hashlib
import json
import logging
import os
import sys
from datetime import datetime
from glob import glob
from pathlib import Path

try:
    import asyncpg
except ImportError:
    print("ERROR: asyncpg required. Install: pip install asyncpg")
    sys.exit(1)

# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
ALIASES_PATH = PROJECT_ROOT / "data" / "oddsportal_team_aliases.json"
RAW_DATA_DIR = PROJECT_ROOT / "data" / "oddsportal_raw"

# Map alias section name → (league_ids[], file_glob_prefix)
SECTION_CONFIG = {
    "ColombiaPrimeraA":         {"league_ids": [239],      "prefix": "colombia-primera-a"},
    "ChilePrimeraDivision":     {"league_ids": [265],      "prefix": "chile-primera-division"},
    "EcuadorLigaPro":           {"league_ids": [242],      "prefix": "ecuador-liga-pro"},
    "UruguayPrimeraDivision":   {"league_ids": [268, 270], "prefix": "uruguay-primera-division"},
    "ParaguayPrimeraDivision":  {"league_ids": [250, 252], "prefix": "paraguay-primera-division"},
    "PeruLiga1":                {"league_ids": [281],      "prefix": "peru-liga-1"},
    "VenezuelaPrimeraDivision": {"league_ids": [299],      "prefix": "venezuela-primera-division"},
    "BoliviaDivisionProfesional": {"league_ids": [344],    "prefix": "bolivia-division-profesional"},
    "ArgentinaLigaProfesional": {"league_ids": [128],      "prefix": "argentina-liga-profesional"},
    "BrasilSerieA":             {"league_ids": [71],       "prefix": "brazil-serie-a"},
    "MexicoLigaMX":             {"league_ids": [262],      "prefix": "mexico-liga-mx"},
    "MLS":                      {"league_ids": [253],      "prefix": "usa-mls"},
    "SaudiProLeague":           {"league_ids": [307],      "prefix": "saudi"},
}

# Bookmaker priority for odds extraction (ABE P0)
BOOKMAKER_PRIORITY = [
    "Pinnacle",
    "bet365",
    "1xBet",
    "Marathon Bet",
    "MarathonBet",
    "William Hill",
    "Betfair",
    "Unibet",
    "bet-at-home",
    "BetInAsia",
]


# =============================================================================
# LOGGING
# =============================================================================

def setup_logging() -> logging.Logger:
    logger = logging.getLogger("oddsportal_ingest")
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(ch)
    return logger


# =============================================================================
# ALIAS LOADING (league-scoped — ABE P0)
# =============================================================================

def load_aliases_for_section(section: str) -> dict[str, int]:
    """Load aliases ONLY for a specific league section.
    Returns dict {normalized_name_lower: team_id}.
    ABE P0: league-scoped to prevent cross-league collisions.
    """
    with open(ALIASES_PATH) as f:
        all_aliases = json.load(f)

    section_data = all_aliases.get(section, {})
    result = {}
    for name, tid in section_data.items():
        if name.startswith("_"):
            continue  # Skip comments
        result[name.lower().strip()] = tid
    return result


# =============================================================================
# ODDS EXTRACTION
# =============================================================================

def extract_best_odds(match_data: dict) -> tuple[float, float, float, str] | None:
    """Extract best available 1x2 odds following bookmaker priority.
    Returns (home_odds, draw_odds, away_odds, source_name) or None.
    """
    market = match_data.get("1x2_market", [])
    if not market:
        return None

    # Index by bookmaker name (case-insensitive)
    by_bookie = {}
    for m in market:
        bname = m.get("bookmaker_name", "").strip()
        if bname:
            by_bookie[bname.lower()] = m

    # Try priority order
    for bookie in BOOKMAKER_PRIORITY:
        entry = by_bookie.get(bookie.lower())
        if entry:
            try:
                h = float(entry["1"])
                d = float(entry["X"])
                a = float(entry["2"])
                if h > 1.0 and d > 1.0 and a > 1.0:
                    return h, d, a, f"OddsPortal ({bookie})"
            except (ValueError, KeyError):
                continue

    # Fallback: average across all bookmakers with valid odds
    odds_h, odds_d, odds_a = [], [], []
    for entry in market:
        try:
            oh = float(entry["1"])
            od = float(entry["X"])
            oa = float(entry["2"])
            if oh > 1.0 and od > 1.0 and oa > 1.0:
                odds_h.append(oh)
                odds_d.append(od)
                odds_a.append(oa)
        except (ValueError, KeyError):
            continue

    if odds_h:
        avg_h = round(sum(odds_h) / len(odds_h), 4)
        avg_d = round(sum(odds_d) / len(odds_d), 4)
        avg_a = round(sum(odds_a) / len(odds_a), 4)
        return avg_h, avg_d, avg_a, f"OddsPortal (avg of {len(odds_h)})"

    return None


def extract_all_bookmaker_odds(match_data: dict) -> list[tuple]:
    """Extract ALL valid bookmaker odds from 1x2_market.
    Returns list of (bookmaker_name, odds_home, odds_draw, odds_away).
    """
    market = match_data.get("1x2_market", [])
    results = []
    for entry in market:
        bname = entry.get("bookmaker_name", "").strip()
        if not bname:
            continue
        try:
            h = float(entry["1"])
            d = float(entry["X"])
            a = float(entry["2"])
            if h > 1.0 and d > 1.0 and a > 1.0:
                results.append((bname, h, d, a))
        except (ValueError, KeyError):
            continue
    return results


# =============================================================================
# MATCH MATCHING (OddsPortal → DB)
# =============================================================================

async def match_op_to_db(
    conn,
    op_match: dict,
    league_ids: list[int],
    aliases: dict[str, int],
) -> tuple[int | None, str]:
    """Match an OddsPortal match to our matches table.

    Returns (match_id, reason) where reason explains the result.
    """
    home_name = op_match.get("home_team", "").strip()
    away_name = op_match.get("away_team", "").strip()

    # Resolve via league-scoped aliases
    home_id = aliases.get(home_name.lower())
    away_id = aliases.get(away_name.lower())

    if not home_id:
        return None, f"unresolved_home:{home_name}"
    if not away_id:
        return None, f"unresolved_away:{away_name}"

    # Parse match date (OddsHarvester appends " UTC" suffix)
    date_str = op_match.get("match_date", "").replace(" UTC", "").strip()
    match_date = None
    if date_str:
        try:
            match_date = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            try:
                match_date = datetime.strptime(date_str, "%Y-%m-%d")
            except ValueError:
                pass  # Fall through to dateless mode

    if match_date:
        # Date-based matching: ±2 day tolerance
        rows = await conn.fetch("""
            SELECT id, league_id, home_goals, away_goals, opening_odds_home
            FROM matches
            WHERE home_team_id = $1
              AND away_team_id = $2
              AND league_id = ANY($3::int[])
              AND ABS(EXTRACT(EPOCH FROM (date - $4::timestamp))) < 172800
              AND status IN ('FT', 'AET', 'PEN')
            ORDER BY ABS(EXTRACT(EPOCH FROM (date - $4::timestamp)))
            LIMIT 1
        """, home_id, away_id, league_ids, match_date)
    else:
        # Dateless matching: use season + score for disambiguation
        season = op_match.get("season")
        op_hs = op_match.get("home_score")
        op_as = op_match.get("away_score")
        if season and op_hs is not None and op_as is not None:
            rows = await conn.fetch("""
                SELECT id, league_id, home_goals, away_goals, opening_odds_home
                FROM matches
                WHERE home_team_id = $1
                  AND away_team_id = $2
                  AND league_id = ANY($3::int[])
                  AND season = $4
                  AND home_goals = $5
                  AND away_goals = $6
                  AND status IN ('FT', 'AET', 'PEN')
                ORDER BY date
            """, home_id, away_id, league_ids, season, int(op_hs), int(op_as))
            # Ambiguous: multiple matches with same teams+score+season → skip
            if len(rows) > 1:
                return None, f"ambiguous:{home_name} vs {away_name} season={season} score={op_hs}-{op_as} ({len(rows)} matches)"
        else:
            return None, f"bad_date:{date_str}"

    if not rows:
        return None, f"no_db_match:{home_name} vs {away_name} season={op_match.get('season', '?')}"

    row = rows[0]

    # Score validation (hard reject on mismatch)
    op_home_score = op_match.get("home_score")
    op_away_score = op_match.get("away_score")
    if op_home_score is not None and op_away_score is not None:
        try:
            op_hs = int(op_home_score)
            op_as = int(op_away_score)
            db_hs = row["home_goals"]
            db_as = row["away_goals"]
            if db_hs is not None and db_as is not None:
                if op_hs != db_hs or op_as != db_as:
                    return None, f"score_mismatch:OP={op_hs}-{op_as} DB={db_hs}-{db_as} (id={row['id']})"
        except (ValueError, TypeError):
            pass  # Can't validate, proceed

    return row["id"], "matched"


# =============================================================================
# MAIN INGESTION
# =============================================================================

async def ingest_section(
    conn,
    section: str,
    json_files: list[str],
    dry_run: bool,
    logger: logging.Logger,
) -> dict:
    """Ingest odds for one league section.

    Returns stats dict with GATE metrics.
    """
    config = SECTION_CONFIG[section]
    league_ids = config["league_ids"]
    aliases = load_aliases_for_section(section)

    if not aliases:
        logger.error(f"ABORT: No aliases found for section '{section}'")
        return {"abort": True, "reason": "no_aliases"}

    # Load all matches from JSON files
    all_op_matches = []
    for fpath in json_files:
        with open(fpath) as f:
            data = json.load(f)
        logger.info(f"  Loaded {len(data)} matches from {Path(fpath).name}")
        all_op_matches.extend(data)

    logger.info(f"  Total OddsPortal matches: {len(all_op_matches)}")

    # GATE check: collect all team names and verify resolution
    all_team_names = set()
    for m in all_op_matches:
        all_team_names.add(m.get("home_team", "").strip())
        all_team_names.add(m.get("away_team", "").strip())

    unresolved_teams = []
    for name in sorted(all_team_names):
        if name and name.lower() not in aliases:
            unresolved_teams.append(name)

    if unresolved_teams:
        logger.error(f"\n  GATE FAILED: {len(unresolved_teams)} unresolved team names:")
        for t in unresolved_teams:
            # Count occurrences
            count = sum(
                1 for m in all_op_matches
                if m.get("home_team", "").strip() == t or m.get("away_team", "").strip() == t
            )
            logger.error(f"    '{t}' (appears {count} times)")
        logger.error(f"\n  ACTION: Add missing teams to {ALIASES_PATH} section '{section}' and retry")
        return {
            "abort": True,
            "reason": "unresolved_teams",
            "unresolved": unresolved_teams,
        }

    resolved_pct = 100.0
    logger.info(f"  GATE: {len(all_team_names)} team names, 100% resolved")

    # Process matches
    stats = {
        "total_op_matches": len(all_op_matches),
        "resolved_teams_pct": resolved_pct,
        "matched": 0,
        "no_db_match": 0,
        "score_mismatch": 0,
        "no_odds_in_op": 0,
        "already_has_odds": 0,
        "updated": 0,
        "errors": 0,
        "unmatched_details": [],
    }

    for op_match in all_op_matches:
        # Extract best odds (for legacy dual-write)
        odds = extract_best_odds(op_match)
        # Extract ALL bookmaker odds (for raw_odds_1x2)
        all_odds = extract_all_bookmaker_odds(op_match)

        if not odds and not all_odds:
            stats["no_odds_in_op"] += 1
            continue

        home_odds, draw_odds, away_odds, source_name = odds if odds else (None, None, None, None)

        # Match to DB
        match_id, reason = await match_op_to_db(conn, op_match, league_ids, aliases)

        if match_id is None:
            if reason.startswith("score_mismatch"):
                stats["score_mismatch"] += 1
            else:
                stats["no_db_match"] += 1
            if len(stats["unmatched_details"]) < 50:
                stats["unmatched_details"].append(reason)

            # DLQ: write unmatched odds to raw_odds_1x2 with match_id = NULL
            # Uses deterministic hash for dedup on re-execution
            if not dry_run and all_odds:
                op_home = op_match.get("home_team", "?")
                op_away = op_match.get("away_team", "?")
                op_date = op_match.get("match_date", "")
                for bookie, oh, od, oa in all_odds:
                    event_hash = hashlib.md5(
                        f"oddsportal|{bookie}|{op_home}|{op_away}|{op_date}".encode()
                    ).hexdigest()
                    try:
                        await conn.execute("""
                            INSERT INTO raw_odds_1x2
                                (match_id, provider, bookmaker, odds_home, odds_draw, odds_away,
                                 odds_kind, external_event_hash, metadata)
                            VALUES (NULL, 'oddsportal', $1, $2, $3, $4, 'closing', $5,
                                    jsonb_build_object('home_team', $6, 'away_team', $7,
                                                       'match_date', $8, 'reason', $9,
                                                       'section', $10))
                            ON CONFLICT (external_event_hash)
                            WHERE match_id IS NULL AND external_event_hash IS NOT NULL
                            DO NOTHING
                        """, bookie, oh, od, oa, event_hash,
                        op_home, op_away, op_date, reason, section)
                    except Exception:
                        pass  # DLQ best-effort, don't block pipeline
                stats.setdefault("raw_dlq", 0)
                stats["raw_dlq"] += len(all_odds)

            continue

        stats["matched"] += 1

        # Fetch match_date once (used by both writes)
        match_date = None
        if not dry_run:
            match_date = await conn.fetchval(
                "SELECT date FROM matches WHERE id = $1", match_id
            )

        # Check if already has legacy odds
        existing = await conn.fetchval(
            "SELECT opening_odds_home FROM matches WHERE id = $1",
            match_id,
        )
        has_legacy_odds = existing is not None

        if not dry_run:
            try:
                # Legacy dual-write: UPDATE matches.opening_odds_* (only if NULL)
                if not has_legacy_odds and odds:
                    await conn.execute("""
                        UPDATE matches
                        SET opening_odds_home = $1,
                            opening_odds_draw = $2,
                            opening_odds_away = $3,
                            opening_odds_source = $4,
                            opening_odds_kind = 'closing',
                            opening_odds_column = '1x2',
                            opening_odds_recorded_at = $5,
                            opening_odds_recorded_at_type = 'match_date'
                        WHERE id = $6
                          AND opening_odds_home IS NULL
                    """, home_odds, draw_odds, away_odds, source_name,
                         match_date, match_id)

                # Raw write: INSERT ALL bookmaker odds into raw_odds_1x2
                for bookie, oh, od, oa in all_odds:
                    await conn.execute("""
                        INSERT INTO raw_odds_1x2
                            (match_id, provider, bookmaker, odds_home, odds_draw, odds_away,
                             odds_kind, recorded_at, metadata)
                        VALUES ($1, 'oddsportal', $2, $3, $4, $5, 'closing', $6,
                                jsonb_build_object('section', $7))
                        ON CONFLICT (match_id, provider, bookmaker, odds_kind)
                        WHERE match_id IS NOT NULL
                        DO UPDATE SET
                            odds_home = EXCLUDED.odds_home,
                            odds_draw = EXCLUDED.odds_draw,
                            odds_away = EXCLUDED.odds_away,
                            metadata = EXCLUDED.metadata
                    """, match_id, bookie, oh, od, oa, match_date, section)
                stats.setdefault("raw_inserted", 0)
                stats["raw_inserted"] += len(all_odds)

            except Exception as e:
                stats["errors"] += 1
                logger.warning(f"  DB error for match_id={match_id}: {e}")
                continue

        if has_legacy_odds:
            stats["already_has_odds"] += 1
        else:
            stats["updated"] += 1

    return stats


def print_gate_report(section: str, stats: dict, dry_run: bool, logger: logging.Logger):
    """Print GATE metrics report."""
    total = stats["total_op_matches"]
    matched = stats["matched"]
    match_rate = (matched / total * 100) if total > 0 else 0

    mode = "DRY RUN" if dry_run else "REAL"
    logger.info(f"\n{'='*60}")
    logger.info(f"GATE METRICS [{mode}] — {section}")
    logger.info(f"{'='*60}")
    logger.info(f"  total_op_matches:   {total}")
    logger.info(f"  resolved_teams:     100% (GATE passed)")
    logger.info(f"  matched_to_db:      {matched}")
    logger.info(f"  match_rate:         {match_rate:.1f}%")
    logger.info(f"  score_mismatches:   {stats['score_mismatch']}")
    logger.info(f"  no_odds_in_op:      {stats['no_odds_in_op']}")
    logger.info(f"  already_has_odds:   {stats['already_has_odds']}")
    logger.info(f"  would_update:       {stats['updated']}")
    logger.info(f"  raw_odds_inserted:  {stats.get('raw_inserted', 0)}")
    logger.info(f"  raw_dlq_unmatched:  {stats.get('raw_dlq', 0)}")
    logger.info(f"  errors:             {stats['errors']}")

    # Abort conditions
    if match_rate < 50 and total > 50:
        logger.warning(f"\n  WARNING: match_rate {match_rate:.1f}% < 50% — check league/date mapping!")

    score_mismatch_rate = (stats["score_mismatch"] / total * 100) if total > 0 else 0
    if score_mismatch_rate > 5:
        logger.warning(f"\n  WARNING: score_mismatch rate {score_mismatch_rate:.1f}% > 5%")

    # Show some unmatched details
    if stats.get("unmatched_details"):
        logger.info(f"\n  Unmatched samples (max 20):")
        for d in stats["unmatched_details"][:20]:
            logger.info(f"    {d}")


# =============================================================================
# CLI
# =============================================================================

async def main():
    parser = argparse.ArgumentParser(description="OddsPortal Odds Ingestion Pipeline")
    parser.add_argument("--section", required=True, choices=list(SECTION_CONFIG.keys()),
                        help="League section to process (matches alias file section)")
    parser.add_argument("--files", nargs="*",
                        help="Specific JSON files to process (default: auto-detect from section prefix)")
    parser.add_argument("--dry-run", action="store_true", default=True,
                        help="Report what would be updated without writing (default)")
    parser.add_argument("--no-dry-run", action="store_true",
                        help="Actually write odds to DB")
    args = parser.parse_args()

    dry_run = not args.no_dry_run
    logger = setup_logging()

    section = args.section
    config = SECTION_CONFIG[section]

    logger.info(f"OddsPortal Ingestion — {section}")
    logger.info(f"  League IDs: {config['league_ids']}")
    logger.info(f"  Mode: {'DRY RUN' if dry_run else 'REAL BACKFILL'}")

    # Discover JSON files
    if args.files:
        json_files = args.files
    else:
        pattern = str(RAW_DATA_DIR / f"{config['prefix']}_*.json")
        json_files = sorted(glob(pattern))
        if not json_files:
            logger.error(f"No JSON files found matching: {pattern}")
            sys.exit(1)

    logger.info(f"  Files: {len(json_files)}")
    for f in json_files:
        logger.info(f"    {Path(f).name}")

    # Connect to DB
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        logger.error("DATABASE_URL not set. Run: source .env")
        sys.exit(1)

    # asyncpg needs postgresql:// not postgres://
    if database_url.startswith("postgres://"):
        database_url = database_url.replace("postgres://", "postgresql://", 1)

    conn = await asyncpg.connect(database_url)

    try:
        stats = await ingest_section(conn, section, json_files, dry_run, logger)

        if stats.get("abort"):
            logger.error(f"\nABORT: {stats['reason']}")
            sys.exit(1)

        print_gate_report(section, stats, dry_run, logger)

    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(main())
