#!/usr/bin/env python3
"""
Build player_id_mapping: bipartite Hungarian matching between API-Football and Sofascore.

Phase 2 Ticket P2-01 — Player Entity Resolution.

Strategy:
  Phase A: Extract overlapping matches (both match_lineups + match_sofascore_player)
  Phase B: Fetch Sofascore player names from lineups API (cached to JSON)
  Phase C: Hungarian matching per match×team_side using name_similarity + position_compatibility
  Phase D: Cross-match aggregation → final confidence per (api_football_id, sofascore_id)
  Phase E: Insert into player_id_mapping table + audit report

Usage:
    source .env && python3 scripts/build_player_id_mapping.py [--dry-run] [--limit N] [--skip-fetch]
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time
import unicodedata
from collections import defaultdict
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import httpx
import numpy as np
from scipy.optimize import linear_sum_assignment

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("build_player_id_mapping")

# =============================================================================
# CONFIGURATION
# =============================================================================

SOFASCORE_API_BASE = "https://api.sofascore.com/api/v1"
SOFASCORE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Origin": "https://www.sofascore.com",
    "Referer": "https://www.sofascore.com/",
}
REQUEST_INTERVAL = 1.1  # seconds between requests
MAX_RETRIES = 3
NAMES_CACHE_PATH = PROJECT_ROOT / "data" / "sofascore_player_names.json"

# Position mapping: API-Football → normalized
AF_POS_MAP = {"G": "GK", "D": "DEF", "M": "MID", "F": "FWD"}

# Scoring weights for cost matrix
W_NAME = 0.60
W_POSITION = 0.30
W_LASTNAME = 0.10  # bonus for exact last-name match

# Minimum score to accept a pair from a single match
MIN_SINGLE_MATCH_SCORE = 0.40

# Minimum aggregated confidence to insert into DB
# 0.60 filters most false positives (single-match weak pairs with wrong names)
MIN_FINAL_CONFIDENCE = 0.60


# =============================================================================
# NAME NORMALIZATION & SIMILARITY
# =============================================================================

def normalize_name(name: str) -> str:
    """Normalize a player name for comparison: lowercase, remove accents, strip."""
    if not name:
        return ""
    # Pre-transliteration for characters NFKD doesn't decompose
    TRANSLITERATE = {
        "Ø": "O", "ø": "o", "Æ": "AE", "æ": "ae",
        "Ð": "D", "ð": "d", "Þ": "Th", "þ": "th",
        "Ł": "L", "ł": "l", "Đ": "D", "đ": "d",
        "ß": "ss", "İ": "I", "ı": "i",
    }
    name = "".join(TRANSLITERATE.get(c, c) for c in name)
    # NFKD decomposition, strip combining marks
    nfkd = unicodedata.normalize("NFKD", name)
    ascii_str = "".join(c for c in nfkd if not unicodedata.combining(c))
    return ascii_str.lower().strip()


def extract_lastname(name: str) -> str:
    """Extract likely last name (last token)."""
    tokens = normalize_name(name).split()
    if not tokens:
        return ""
    return tokens[-1]


def name_similarity(name_a: str, name_b: str) -> float:
    """
    Compute name similarity between two player names.

    Handles common football name patterns:
    - "A. Areola" vs "Alphonse Areola" (initial vs full first name)
    - "K. Mavropanos" vs "Konstantinos Mavropanos"
    """
    na = normalize_name(name_a)
    nb = normalize_name(name_b)

    if not na or not nb:
        return 0.0

    # Exact match
    if na == nb:
        return 1.0

    # SequenceMatcher on full normalized name
    full_ratio = SequenceMatcher(None, na, nb).ratio()

    # Last-name match bonus
    la = extract_lastname(name_a)
    lb = extract_lastname(name_b)
    lastname_ratio = SequenceMatcher(None, la, lb).ratio()

    # Token overlap: handles "A. Areola" vs "Alphonse Areola"
    tokens_a = set(normalize_name(name_a).replace(".", "").split())
    tokens_b = set(normalize_name(name_b).replace(".", "").split())

    # Remove single-char tokens (initials) for overlap
    long_tokens_a = {t for t in tokens_a if len(t) > 1}
    long_tokens_b = {t for t in tokens_b if len(t) > 1}

    if long_tokens_a and long_tokens_b:
        overlap = len(long_tokens_a & long_tokens_b)
        union = len(long_tokens_a | long_tokens_b)
        token_ratio = overlap / union if union > 0 else 0.0
    else:
        token_ratio = 0.0

    # Initial match: "A." matching "Alphonse" (first char)
    initial_bonus = 0.0
    short_a = {t.rstrip(".") for t in tokens_a if len(t.rstrip(".")) == 1}
    short_b = {t.rstrip(".") for t in tokens_b if len(t.rstrip(".")) == 1}
    full_b_initials = {t[0] for t in long_tokens_b}
    full_a_initials = {t[0] for t in long_tokens_a}
    if short_a & full_b_initials or short_b & full_a_initials:
        initial_bonus = 0.15

    # Weighted combination
    score = max(
        full_ratio,
        0.6 * lastname_ratio + 0.25 * token_ratio + 0.15 * initial_bonus + initial_bonus,
        0.8 * lastname_ratio + 0.2 * token_ratio,
    )

    return min(score, 1.0)


def position_compatibility(pos_af: str, pos_sc: str) -> float:
    """
    Position compatibility score.
    API-Football uses G/D/M/F. Sofascore uses GK/DEF/MID/FWD.
    """
    norm_af = AF_POS_MAP.get(pos_af, pos_af)
    norm_sc = pos_sc  # already GK/DEF/MID/FWD

    if norm_af == norm_sc:
        return 1.0

    # Adjacent positions (common misclassification)
    adjacent = {
        ("DEF", "MID"): 0.3,
        ("MID", "DEF"): 0.3,
        ("MID", "FWD"): 0.3,
        ("FWD", "MID"): 0.3,
    }
    return adjacent.get((norm_af, norm_sc), 0.0)


# =============================================================================
# SOFASCORE NAME FETCHER
# =============================================================================

async def fetch_sofascore_lineup_names(
    client: httpx.AsyncClient,
    event_id: str,
) -> Dict[str, str]:
    """
    Fetch player names from Sofascore lineups endpoint.

    Returns: {player_id_ext: player_name}
    """
    url = f"{SOFASCORE_API_BASE}/event/{event_id}/lineups"

    for attempt in range(MAX_RETRIES):
        try:
            resp = await client.get(url, headers=SOFASCORE_HEADERS, timeout=10.0)
            if resp.status_code == 404:
                return {}
            if resp.status_code == 403:
                logger.warning(f"  403 for event {event_id}, attempt {attempt+1}")
                await asyncio.sleep(5.0 * (attempt + 1))
                continue
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            logger.warning(f"  Error fetching event {event_id}: {e}")
            await asyncio.sleep(2.0 * (attempt + 1))
            continue

        names = {}
        for side in ("home", "away"):
            team_data = data.get(side, {})
            for entry in team_data.get("players", []):
                player_info = entry.get("player", {})
                pid = player_info.get("id")
                pname = player_info.get("name") or player_info.get("shortName", "")
                if pid and pname:
                    names[str(pid)] = pname
        return names

    return {}


async def resolve_sofascore_names(
    matches_with_events: List[Tuple[int, str]],
    skip_fetch: bool = False,
    limit: int = 0,
) -> Dict[str, str]:
    """
    Resolve Sofascore player_id_ext → player_name for all overlapping matches.

    Uses cache file to avoid re-fetching.
    Returns: {player_id_ext: player_name}
    """
    # Load cache
    cache = {}
    if NAMES_CACHE_PATH.exists():
        with open(NAMES_CACHE_PATH) as f:
            cache = json.load(f)
        logger.info(f"Loaded name cache: {len(cache)} players")

    if skip_fetch:
        logger.info("--skip-fetch: using cached names only")
        return cache

    # Determine which events still need fetching
    # Track which events we've already fetched (store in cache metadata)
    meta_key = "__fetched_events__"
    fetched_events = set(cache.get(meta_key, []))

    events_to_fetch = [
        (mid, eid) for mid, eid in matches_with_events
        if eid not in fetched_events
    ]

    if limit > 0:
        events_to_fetch = events_to_fetch[:limit]

    if not events_to_fetch:
        logger.info("All events already fetched, using cache")
        return {k: v for k, v in cache.items() if k != meta_key}

    logger.info(f"Fetching names for {len(events_to_fetch)} events "
                f"({len(fetched_events)} already cached)...")

    async with httpx.AsyncClient() as client:
        for i, (match_id, event_id) in enumerate(events_to_fetch):
            names = await fetch_sofascore_lineup_names(client, event_id)
            if names:
                cache.update(names)
                fetched_events.add(event_id)

            if (i + 1) % 50 == 0:
                logger.info(f"  Progress: {i+1}/{len(events_to_fetch)} events")

            await asyncio.sleep(REQUEST_INTERVAL)

    # Save cache
    cache[meta_key] = list(fetched_events)
    NAMES_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(NAMES_CACHE_PATH, "w") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved name cache: {len(cache) - 1} players, "
                f"{len(fetched_events)} events")

    return {k: v for k, v in cache.items() if k != meta_key}


# =============================================================================
# DATA EXTRACTION
# =============================================================================

async def extract_overlapping_matches(engine) -> List[dict]:
    """
    Extract all matches that have data in BOTH match_lineups and match_sofascore_player.

    Returns list of dicts with match_id, sofascore_event_id, and per-side data.
    """
    from sqlalchemy import text
    from sqlalchemy.ext.asyncio import AsyncSession
    from sqlalchemy.orm import sessionmaker

    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async with async_session() as session:
        # Get match IDs with Sofascore event IDs
        result = await session.execute(text("""
            SELECT DISTINCT msp.match_id, mer.source_match_id as sofascore_event_id
            FROM match_sofascore_player msp
            JOIN match_lineups ml ON ml.match_id = msp.match_id
            JOIN match_external_refs mer ON mer.match_id = msp.match_id AND mer.source = 'sofascore'
            WHERE msp.is_starter = true
              AND array_length(ml.starting_xi_ids, 1) >= 11
            ORDER BY msp.match_id
        """))
        match_events = [(r[0], r[1]) for r in result.fetchall()]
        logger.info(f"Found {len(match_events)} overlapping matches")

        # Get API-Football lineup data
        # match_lineups uses is_home (bool), not team_side
        result = await session.execute(text("""
            SELECT ml.match_id,
                   CASE WHEN ml.is_home THEN 'home' ELSE 'away' END as team_side,
                   ml.starting_xi_ids, ml.starting_xi_names, ml.starting_xi_positions
            FROM match_lineups ml
            WHERE ml.match_id IN (
                SELECT DISTINCT msp.match_id
                FROM match_sofascore_player msp
                JOIN match_lineups ml2 ON ml2.match_id = msp.match_id
                WHERE msp.is_starter = true AND array_length(ml2.starting_xi_ids, 1) >= 11
            )
            AND array_length(ml.starting_xi_ids, 1) >= 11
            ORDER BY ml.match_id, team_side
        """))
        af_lineups = {}
        for r in result.fetchall():
            key = (r[0], r[1])  # (match_id, team_side)
            af_lineups[key] = {
                "ids": r[2],      # int[]
                "names": r[3],    # text[]
                "positions": r[4] # text[]
            }

        # Get Sofascore player data
        result = await session.execute(text("""
            SELECT msp.match_id, msp.team_side,
                   msp.player_id_ext, msp.position
            FROM match_sofascore_player msp
            WHERE msp.match_id IN (
                SELECT DISTINCT msp2.match_id
                FROM match_sofascore_player msp2
                JOIN match_lineups ml ON ml.match_id = msp2.match_id
                WHERE msp2.is_starter = true AND array_length(ml.starting_xi_ids, 1) >= 11
            )
            AND msp.is_starter = true
            ORDER BY msp.match_id, msp.team_side
        """))
        sc_lineups = defaultdict(list)
        for r in result.fetchall():
            key = (r[0], r[1])
            sc_lineups[key].append({
                "player_id_ext": r[2],
                "position": r[3],
            })

    return match_events, af_lineups, sc_lineups


# =============================================================================
# HUNGARIAN MATCHING
# =============================================================================

def build_cost_matrix(
    af_players: List[dict],
    sc_players: List[dict],
    sc_names: Dict[str, str],
) -> Tuple[np.ndarray, List[dict], List[dict]]:
    """
    Build cost matrix for Hungarian algorithm.

    af_players: [{"id": int, "name": str, "position": str}, ...]
    sc_players: [{"player_id_ext": str, "position": str}, ...]
    sc_names: {player_id_ext: player_name}

    Returns: (cost_matrix, af_list, sc_list) where cost = 1 - score
    """
    n = len(af_players)
    m = len(sc_players)

    if n == 0 or m == 0:
        return np.array([[]]), af_players, sc_players

    cost = np.ones((n, m))  # default cost = 1.0 (score = 0.0)

    for i, af in enumerate(af_players):
        for j, sc in enumerate(sc_players):
            sc_name = sc_names.get(sc["player_id_ext"], "")
            af_name = af["name"] or ""

            # Name similarity
            ns = name_similarity(af_name, sc_name)

            # Position compatibility
            pc = position_compatibility(af["position"], sc["position"])

            # Last-name exact match bonus
            lb = 0.0
            if extract_lastname(af_name) and extract_lastname(sc_name):
                if extract_lastname(af_name) == extract_lastname(sc_name):
                    lb = 1.0

            score = W_NAME * ns + W_POSITION * pc + W_LASTNAME * lb
            cost[i][j] = 1.0 - score

    return cost, af_players, sc_players


def run_hungarian_matching(
    match_id: int,
    team_side: str,
    af_data: dict,
    sc_data: List[dict],
    sc_names: Dict[str, str],
) -> List[dict]:
    """
    Run Hungarian algorithm for one match×team_side.

    Returns list of matched pairs with scores.
    """
    # Build AF player list
    af_players = []
    ids = af_data["ids"]
    names = af_data["names"]
    positions = af_data["positions"]

    for k in range(len(ids)):
        af_players.append({
            "id": ids[k],
            "name": names[k] if k < len(names) else "",
            "position": positions[k] if k < len(positions) else "",
        })

    if not af_players or not sc_data:
        return []

    cost, af_list, sc_list = build_cost_matrix(af_players, sc_data, sc_names)

    if cost.size == 0:
        return []

    # Run Hungarian algorithm
    row_ind, col_ind = linear_sum_assignment(cost)

    pairs = []
    for r, c in zip(row_ind, col_ind):
        score = 1.0 - cost[r][c]
        if score >= MIN_SINGLE_MATCH_SCORE:
            af = af_list[r]
            sc = sc_list[c]
            pairs.append({
                "api_football_id": af["id"],
                "sofascore_id": sc["player_id_ext"],
                "af_name": af["name"],
                "sc_name": sc_names.get(sc["player_id_ext"], ""),
                "af_position": af["position"],
                "sc_position": sc["position"],
                "score": round(score, 4),
                "match_id": match_id,
                "team_side": team_side,
            })

    return pairs


# =============================================================================
# CROSS-MATCH AGGREGATION
# =============================================================================

def aggregate_pairs(all_pairs: List[dict]) -> List[dict]:
    """
    Aggregate matched pairs across all matches.

    For each (api_football_id, sofascore_id) pair:
    - count: number of matches where this pair was matched
    - avg_score: average single-match score
    - max_score: best single-match score
    - confidence: final confidence = f(count, avg_score)
    """
    pair_key = lambda p: (p["api_football_id"], p["sofascore_id"])

    groups = defaultdict(list)
    for pair in all_pairs:
        groups[pair_key(pair)].append(pair)

    # Also track: for each api_football_id, what sofascore_ids was it matched with?
    af_candidates = defaultdict(list)
    sc_candidates = defaultdict(list)

    for key, matches in groups.items():
        af_id, sc_id = key
        af_candidates[af_id].append((sc_id, len(matches)))
        sc_candidates[sc_id].append((af_id, len(matches)))

    results = []
    for key, matches in groups.items():
        af_id, sc_id = key
        count = len(matches)
        avg_score = sum(m["score"] for m in matches) / count
        max_score = max(m["score"] for m in matches)

        # Confidence formula:
        # - Base: avg_score
        # - Boost for repeated matches (diminishing returns)
        # - Penalty if this pair competes with other candidates
        count_boost = min(0.2, 0.05 * (count - 1))  # up to +0.2 for 5+ matches

        # Competition penalty: if af_id matched with multiple sc_ids
        af_comps = af_candidates[af_id]
        if len(af_comps) > 1:
            total_matches_af = sum(c for _, c in af_comps)
            dominance = count / total_matches_af
            competition_penalty = max(0, 0.15 * (1 - dominance))
        else:
            competition_penalty = 0.0

        confidence = min(1.0, avg_score + count_boost - competition_penalty)

        # Method determination
        if max_score >= 0.90:
            method = "bipartite_strong"
        elif max_score >= 0.70:
            method = "bipartite"
        else:
            method = "bipartite_weak"

        # Status
        if confidence >= 0.80:
            status = "active"
        elif confidence >= MIN_FINAL_CONFIDENCE:
            status = "pending_review"
        else:
            continue  # skip low-confidence pairs

        sample = matches[0]
        results.append({
            "api_football_id": af_id,
            "sofascore_id": sc_id,
            "player_name": sample["af_name"],
            "sc_name": sample["sc_name"],
            "position": AF_POS_MAP.get(sample["af_position"], sample["af_position"]),
            "confidence": round(confidence, 4),
            "method": method,
            "status": status,
            "match_count": count,
            "avg_score": round(avg_score, 4),
            "source_match_id": sample["match_id"],
        })

    # Resolve conflicts: if multiple sc_ids for same af_id, keep highest confidence
    # as active; secondary alternatives downgraded to pending_review for auditing
    results.sort(key=lambda x: -x["confidence"])
    seen_af = set()
    seen_sc = set()
    final = []
    for r in results:
        af = r["api_football_id"]
        sc = r["sofascore_id"]
        if af in seen_af or sc in seen_sc:
            # This is a secondary candidate — one side already has a better match
            # Keep as pending_review for auditing (not silently dropped)
            if af not in seen_af or sc not in seen_sc:
                # Only ONE side conflicts — plausible alternative, keep for review
                r["status"] = "pending_review"
                r["method"] += "_conflict"
                final.append(r)
            # If BOTH sides already seen → true duplicate pair, skip
            continue
        seen_af.add(af)
        seen_sc.add(sc)
        final.append(r)

    return final


# =============================================================================
# DB INSERTION
# =============================================================================

async def insert_mappings(engine, mappings: List[dict], dry_run: bool = False) -> int:
    """Insert final mappings into player_id_mapping table using batch inserts."""
    from sqlalchemy import text
    from sqlalchemy.ext.asyncio import AsyncSession
    from sqlalchemy.orm import sessionmaker

    if dry_run:
        logger.info(f"[DRY RUN] Would insert {len(mappings)} mappings")
        return len(mappings)

    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    # Filter out rows with NULL api_football_id (corrupt starting_xi_ids)
    valid_mappings = [m for m in mappings if m["api_football_id"] is not None]
    skipped_null = len(mappings) - len(valid_mappings)
    if skipped_null:
        logger.warning(f"Skipped {skipped_null} mappings with NULL api_football_id")

    BATCH_SIZE = 200
    inserted = 0
    async with async_session() as session:
        for batch_start in range(0, len(valid_mappings), BATCH_SIZE):
            batch = valid_mappings[batch_start:batch_start + BATCH_SIZE]
            # Build multi-row VALUES clause
            value_clauses = []
            params = {}
            for i, m in enumerate(batch):
                value_clauses.append(
                    f"(:af_{i}, :sc_{i}, :name_{i}, :pos_{i}, "
                    f":conf_{i}, :method_{i}, :status_{i}, :src_{i}, NOW(), NOW())"
                )
                params[f"af_{i}"] = m["api_football_id"]
                params[f"sc_{i}"] = m["sofascore_id"]
                params[f"name_{i}"] = m["player_name"]
                params[f"pos_{i}"] = m["position"]
                params[f"conf_{i}"] = m["confidence"]
                params[f"method_{i}"] = m["method"]
                params[f"status_{i}"] = m["status"]
                params[f"src_{i}"] = m["source_match_id"]

            sql = f"""
                INSERT INTO player_id_mapping
                    (api_football_id, sofascore_id, player_name, position,
                     confidence, method, status, source_match_id, created_at, updated_at)
                VALUES {", ".join(value_clauses)}
                ON CONFLICT (api_football_id, sofascore_id) DO UPDATE SET
                    confidence = EXCLUDED.confidence,
                    method = EXCLUDED.method,
                    status = EXCLUDED.status,
                    player_name = EXCLUDED.player_name,
                    updated_at = NOW()
            """
            try:
                await session.execute(text(sql), params)
                inserted += len(batch)
                logger.info(f"  Batch {batch_start//BATCH_SIZE + 1}: "
                            f"{len(batch)} rows ({inserted}/{len(valid_mappings)})")
            except Exception as e:
                logger.error(f"Batch insert error at offset {batch_start}: {e}")
                # Rollback aborted transaction before retrying
                await session.rollback()
                # Fallback to individual inserts for this batch
                for m in batch:
                    try:
                        await session.execute(text("""
                            INSERT INTO player_id_mapping
                                (api_football_id, sofascore_id, player_name, position,
                                 confidence, method, status, source_match_id, created_at, updated_at)
                            VALUES
                                (:af_id, :sc_id, :name, :pos,
                                 :conf, :method, :status, :src_match, NOW(), NOW())
                            ON CONFLICT (api_football_id, sofascore_id) DO UPDATE SET
                                confidence = EXCLUDED.confidence,
                                method = EXCLUDED.method,
                                status = EXCLUDED.status,
                                player_name = EXCLUDED.player_name,
                                updated_at = NOW()
                        """), {
                            "af_id": m["api_football_id"],
                            "sc_id": m["sofascore_id"],
                            "name": m["player_name"],
                            "pos": m["position"],
                            "conf": m["confidence"],
                            "method": m["method"],
                            "status": m["status"],
                            "src_match": m["source_match_id"],
                        })
                        inserted += 1
                    except Exception as e2:
                        logger.error(f"Error inserting {m['api_football_id']}↔{m['sofascore_id']}: {e2}")

        await session.commit()

    return inserted


# =============================================================================
# AUDIT REPORT
# =============================================================================

def print_audit_report(mappings: List[dict], all_pairs_count: int):
    """Print detailed audit report."""
    active = [m for m in mappings if m["status"] == "active"]
    pending = [m for m in mappings if m["status"] == "pending_review"]

    print("\n" + "=" * 72)
    print("PLAYER ID MAPPING — AUDIT REPORT")
    print("=" * 72)

    print(f"\nTotal raw pairs (pre-aggregation): {all_pairs_count}")
    print(f"Final unique mappings: {len(mappings)}")
    print(f"  Active (conf >= 0.80): {len(active)}")
    print(f"  Pending review (0.50-0.80): {len(pending)}")

    # Method breakdown
    methods = defaultdict(int)
    for m in mappings:
        methods[m["method"]] += 1
    print("\nMethod breakdown:")
    for method, count in sorted(methods.items(), key=lambda x: -x[1]):
        print(f"  {method}: {count}")

    # Confidence distribution
    confs = [m["confidence"] for m in mappings]
    if confs:
        print(f"\nConfidence distribution:")
        print(f"  Mean: {np.mean(confs):.4f}")
        print(f"  Median: {np.median(confs):.4f}")
        print(f"  P10: {np.percentile(confs, 10):.4f}")
        print(f"  P90: {np.percentile(confs, 90):.4f}")

    # Match count distribution
    match_counts = [m["match_count"] for m in mappings]
    if match_counts:
        print(f"\nMatch count per pair:")
        print(f"  Mean: {np.mean(match_counts):.1f}")
        print(f"  Median: {np.median(match_counts):.0f}")
        print(f"  Max: {max(match_counts)}")
        print(f"  Singles (1 match only): {sum(1 for c in match_counts if c == 1)}")

    # Position breakdown
    positions = defaultdict(int)
    for m in mappings:
        positions[m["position"]] += 1
    print(f"\nPosition breakdown:")
    for pos in ["GK", "DEF", "MID", "FWD"]:
        print(f"  {pos}: {positions.get(pos, 0)}")

    # Filter out NULLs for display
    valid = [m for m in mappings if m.get("api_football_id") is not None]

    # Top 20 highest confidence
    print(f"\nTop 20 highest-confidence mappings:")
    for m in sorted(valid, key=lambda x: -x["confidence"])[:20]:
        print(f"  AF {m['api_football_id']:>8} ↔ SC {m['sofascore_id']:>10} | "
              f"{m['player_name']:>25} ↔ {m['sc_name']:>25} | "
              f"{m['position']:>3} | conf={m['confidence']:.3f} | "
              f"n={m['match_count']} | {m['method']}")

    # Bottom 20 (lowest confidence, still included)
    valid_pending = [m for m in pending if m.get("api_football_id") is not None]
    if valid_pending:
        print(f"\nBottom 20 pending-review mappings:")
        for m in sorted(valid_pending, key=lambda x: x["confidence"])[:20]:
            print(f"  AF {m['api_football_id']:>8} ↔ SC {m['sofascore_id']:>10} | "
                  f"{m['player_name']:>25} ↔ {m['sc_name']:>25} | "
                  f"{m['position']:>3} | conf={m['confidence']:.3f} | "
                  f"n={m['match_count']} | {m['method']}")

    print("\n" + "=" * 72)


# =============================================================================
# MAIN
# =============================================================================

async def main():
    parser = argparse.ArgumentParser(description="Build player_id_mapping via Hungarian matching")
    parser.add_argument("--dry-run", action="store_true", help="Don't write to DB")
    parser.add_argument("--limit", type=int, default=0, help="Limit Sofascore API fetches")
    parser.add_argument("--skip-fetch", action="store_true", help="Use cached names only")
    parser.add_argument("--verbose", action="store_true", help="Verbose per-match logging")
    args = parser.parse_args()

    database_url = os.environ.get("DATABASE_URL", "")
    if not database_url:
        raise ValueError("DATABASE_URL not set")
    if database_url.startswith("postgresql://"):
        database_url = database_url.replace("postgresql://", "postgresql+asyncpg://", 1)

    from sqlalchemy.ext.asyncio import create_async_engine
    engine = create_async_engine(database_url, echo=False, pool_size=5)

    t0 = time.time()

    # Phase A: Extract overlapping matches
    logger.info("Phase A: Extracting overlapping matches...")
    match_events, af_lineups, sc_lineups = await extract_overlapping_matches(engine)
    logger.info(f"  {len(match_events)} matches, "
                f"{len(af_lineups)} AF lineups, "
                f"{len(sc_lineups)} SC lineups")

    # Phase B: Resolve Sofascore player names
    logger.info("Phase B: Resolving Sofascore player names...")
    sc_names = await resolve_sofascore_names(
        match_events,
        skip_fetch=args.skip_fetch,
        limit=args.limit,
    )
    logger.info(f"  Resolved {len(sc_names)} player names")

    # Phase C: Hungarian matching per match×team_side
    logger.info("Phase C: Running Hungarian matching...")
    all_pairs = []
    matches_processed = 0
    matches_skipped = 0

    for match_id, event_id in match_events:
        for team_side in ("home", "away"):
            key = (match_id, team_side)
            af = af_lineups.get(key)
            sc = sc_lineups.get(key, [])

            if not af or not sc:
                matches_skipped += 1
                continue

            pairs = run_hungarian_matching(match_id, team_side, af, sc, sc_names)
            all_pairs.extend(pairs)

            if args.verbose and pairs:
                logger.info(f"  Match {match_id} {team_side}: {len(pairs)} pairs "
                            f"(best={max(p['score'] for p in pairs):.3f})")

        matches_processed += 1

    logger.info(f"  Processed {matches_processed} matches, "
                f"skipped {matches_skipped} team-sides, "
                f"found {len(all_pairs)} raw pairs")

    # Phase D: Cross-match aggregation
    logger.info("Phase D: Aggregating across matches...")
    mappings = aggregate_pairs(all_pairs)
    logger.info(f"  {len(mappings)} unique mappings after aggregation")

    # Phase E: Insert into DB + audit
    logger.info("Phase E: Inserting into player_id_mapping...")
    inserted = await insert_mappings(engine, mappings, dry_run=args.dry_run)
    logger.info(f"  Inserted/updated: {inserted}")

    # Audit report
    print_audit_report(mappings, len(all_pairs))

    elapsed = time.time() - t0
    logger.info(f"Total elapsed: {elapsed:.1f}s")

    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
