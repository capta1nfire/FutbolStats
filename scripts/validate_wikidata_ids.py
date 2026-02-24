#!/usr/bin/env python3
"""
Wikidata QID Reconciliation v2 — Country-aware + Wikipedia-first.

4-phase approach:
  1. Wikipedia Search → get QID via wikibase_item (high confidence)
  2. SPARQL Validation → verify type + country (P17, P131*/P17 fallback)
  3. Fallback wbsearchentities → batch SPARQL with VALUES (1 req per team)
  4. Confidence scoring + auto-apply (HIGH/MED only if country_match)

Usage:
    python scripts/validate_wikidata_ids.py --scope latam --dry-run
    python scripts/validate_wikidata_ids.py --scope Bolivia --apply
    python scripts/validate_wikidata_ids.py --team-id 4325 --apply
    python scripts/validate_wikidata_ids.py --all --apply

Author: Master (per ABE reconciliation_v2 directive, 2026-02-23)
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import re

import httpx

# Shared constants
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wikidata_common import (
    COUNTRY_QID_MAP,
    LATAM_COUNTRIES,
    REVERSE_COUNTRY_MAP,
    SPARQL_ENDPOINT,
    VALID_TYPES,
    WIKIDATA_API,
    WIKIPEDIA_API,
)

# Database connection (P0-SEC: no hardcoded credentials)
DATABASE_URL = os.environ.get("DATABASE_URL")
if not DATABASE_URL:
    print("ERROR: DATABASE_URL environment variable is required", file=sys.stderr)
    sys.exit(1)
if DATABASE_URL.startswith("postgresql://"):
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://", 1)

# Rate limiting
WIKIPEDIA_DELAY = 0.2   # 5 req/sec
SPARQL_DELAY = 2.0      # Conservative for SPARQL
WBSEARCH_DELAY = 1.0    # Wikidata API

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("logs")
OUTPUT_DIR.mkdir(exist_ok=True)

USER_AGENT = "BonJogo/2.0 (wikidata-reconciliation-v2)"

# Stop words for name comparison (common prefixes/suffixes in club names)
_NAME_STOP = frozenset({
    "fc", "cf", "sc", "ac", "cd", "de", "del", "la", "el", "los", "las",
    "club", "deportivo", "deportiva", "atletico", "athletic", "united",
    "city", "real", "sporting", "sport", "futbol", "football", "soccer",
    "y", "e", "da", "do", "dos", "san", "santa",
})

# Description keywords that indicate a club/team entity
_CLUB_KEYWORDS = [
    "football club", "soccer club", "football team", "soccer team",
    "sports club", "sporting club", "association football",
]

# Description keywords that indicate a NON-club entity (person, season, etc.)
_EXCLUDED_KEYWORDS = [
    "footballer", "football player", "soccer player",
    "manager", "coach",
    " season", "tournament", "rivalry", "stadium",
    "national team",  # avoid matching national teams for club searches
]


def _name_overlap(team_name: str, wiki_title: str) -> bool:
    """
    Check that at least one significant word token (len≥3) overlaps
    between the team name and the Wikipedia article title.

    Prevents accepting articles about completely different entities
    (e.g., "Emelec" → "Universidad Católica" article).
    """
    def sig_tokens(s: str) -> set[str]:
        raw = set(re.sub(r"[^\w\s]", "", s.lower()).split())
        return {t for t in raw if len(t) >= 3} - _NAME_STOP

    t1 = sig_tokens(team_name)
    t2 = sig_tokens(wiki_title)

    if not t1:
        # Fallback: use all tokens ≥2 chars (for short names like "ADT")
        t1 = {t for t in re.sub(r"[^\w\s]", "", team_name.lower()).split() if len(t) >= 2}
    if not t2:
        t2 = {t for t in re.sub(r"[^\w\s]", "", wiki_title.lower()).split() if len(t) >= 2}

    return bool(t1 & t2)


def _is_football_club_desc(desc: str) -> bool:
    """
    Check if a Wikipedia description indicates a football club/team entity.

    Returns True only if it matches club keywords AND doesn't match exclusion
    patterns (persons, seasons, tournaments, stadiums).
    """
    desc_lower = desc.lower()
    is_club = any(kw in desc_lower for kw in _CLUB_KEYWORDS)
    is_excluded = any(kw in desc_lower for kw in _EXCLUDED_KEYWORDS)
    return is_club and not is_excluded


# =============================================================================
# Phase 1: Wikipedia Search (primary method)
# =============================================================================

async def wikipedia_search_qid(
    team_name: str,
    country: str,
    client: httpx.AsyncClient,
) -> Optional[dict]:
    """
    Search Wikipedia for team, extract QID via wikibase_item.

    Tries multiple search variants. Validates description contains football keywords.
    """
    search_url = f"{WIKIPEDIA_API}/w/api.php"
    summary_url = f"{WIKIPEDIA_API}/api/rest_v1/page/summary"

    # Search variants: most specific → least specific
    variants = [
        f"{team_name} {country} football club",
        f"{team_name} FC",
        team_name,
    ]

    for query in variants:
        params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "srlimit": 5,
            "format": "json",
        }
        try:
            resp = await client.get(search_url, params=params, timeout=30.0)
            if resp.status_code != 200:
                continue

            results = resp.json().get("query", {}).get("search", [])
            for result in results:
                title = result["title"].replace(" ", "_")
                await asyncio.sleep(WIKIPEDIA_DELAY)

                try:
                    summary_resp = await client.get(f"{summary_url}/{title}", timeout=30.0)
                    if summary_resp.status_code != 200:
                        continue
                    data = summary_resp.json()
                except Exception:
                    continue

                qid = data.get("wikibase_item")
                if not qid:
                    continue

                desc = data.get("description") or ""
                wiki_title_clean = data.get("title") or ""

                # Gate 1: Must be a football club/team (not person/season/tournament)
                if not _is_football_club_desc(desc):
                    continue

                # Gate 2: Name overlap — wiki article must relate to our team
                if not _name_overlap(team_name, wiki_title_clean):
                    logger.debug(f"    Name mismatch: '{team_name}' vs '{wiki_title_clean}' — skipping")
                    continue

                return {
                    "qid": qid,
                    "title": wiki_title_clean,
                    "description": desc,
                    "method": "wikipedia",
                }

        except Exception as e:
            logger.debug(f"Wikipedia search error for '{query}': {e}")
            continue

        await asyncio.sleep(WIKIPEDIA_DELAY)

    return None


# =============================================================================
# Phase 2: SPARQL Validation (type + country with P131*/P17 fallback)
# =============================================================================

SPARQL_VALIDATE_QUERY = """
SELECT ?type ?country ?locCountry ?stadium WHERE {{
  OPTIONAL {{ wd:{qid} wdt:P31/wdt:P279* ?type .
    FILTER (?type IN ({types})) }}
  OPTIONAL {{ wd:{qid} wdt:P17 ?country . }}
  OPTIONAL {{ wd:{qid} wdt:P131*/wdt:P17 ?locCountry . }}
  OPTIONAL {{ wd:{qid} wdt:P115 ?stadium . }}
}}
LIMIT 10
"""


async def sparql_validate_candidate(
    qid: str,
    expected_country: str,
    client: httpx.AsyncClient,
) -> dict:
    """
    Validate QID via SPARQL: check type, country (P17 + P131*/P17 fallback), stadium.

    If expected_country not in COUNTRY_QID_MAP → degrade to LOW without crash.
    """
    types_str = ", ".join(f"wd:{t}" for t in VALID_TYPES)
    query = SPARQL_VALIDATE_QUERY.format(qid=qid, types=types_str)

    result = {
        "type_valid": False,
        "country_match": False,
        "has_stadium": False,
        "country_qid": None,
    }

    try:
        resp = await client.get(
            SPARQL_ENDPOINT,
            params={"query": query, "format": "json"},
            headers={"User-Agent": USER_AGENT},
            timeout=60.0,
        )
        resp.raise_for_status()
        bindings = resp.json().get("results", {}).get("bindings", [])

        expected_qid = COUNTRY_QID_MAP.get(expected_country)
        if not expected_qid:
            logger.warning(f"Country '{expected_country}' not in COUNTRY_QID_MAP — will degrade to LOW")

        for b in bindings:
            # Type check
            if b.get("type", {}).get("value"):
                result["type_valid"] = True

            # Stadium check
            if b.get("stadium", {}).get("value"):
                result["has_stadium"] = True

            # Country check: P17 direct first, fallback P131*/P17
            country_uri = b.get("country", {}).get("value", "")
            loc_country_uri = b.get("locCountry", {}).get("value", "")

            for uri in [country_uri, loc_country_uri]:
                if uri and "/entity/Q" in uri:
                    found_qid = uri.split("/")[-1]
                    result["country_qid"] = found_qid
                    if expected_qid and found_qid == expected_qid:
                        result["country_match"] = True
                        break

            if result["country_match"]:
                break

    except Exception as e:
        logger.warning(f"SPARQL validation failed for {qid}: {e}")

    return result


# =============================================================================
# Phase 3: Fallback wbsearchentities + batch SPARQL validation
# =============================================================================

SPARQL_BATCH_VALIDATE = """
SELECT ?item ?type ?country ?locCountry ?stadium WHERE {{
  VALUES ?item {{ {items} }}
  OPTIONAL {{ ?item wdt:P31/wdt:P279* ?type .
    FILTER (?type IN ({types})) }}
  OPTIONAL {{ ?item wdt:P17 ?country . }}
  OPTIONAL {{ ?item wdt:P131*/wdt:P17 ?locCountry . }}
  OPTIONAL {{ ?item wdt:P115 ?stadium . }}
}}
"""


async def wbsearch_with_validation(
    team_name: str,
    country: str,
    client: httpx.AsyncClient,
    limit: int = 10,
) -> list[dict]:
    """
    Search Wikidata wbsearchentities, then batch-validate all candidates in 1 SPARQL.

    Returns ranked list of candidates with validation results.
    """
    # Step 1: wbsearchentities
    params = {
        "action": "wbsearchentities",
        "search": team_name,
        "language": "en",
        "type": "item",
        "format": "json",
        "limit": limit,
    }

    try:
        resp = await client.get(
            WIKIDATA_API,
            params=params,
            headers={"User-Agent": USER_AGENT},
            timeout=30.0,
        )
        resp.raise_for_status()
        candidates = resp.json().get("search", [])
    except Exception as e:
        logger.warning(f"wbsearchentities failed for '{team_name}': {e}")
        return []

    if not candidates:
        return []

    # Step 2: Batch SPARQL validation (1 request for all candidates)
    qids = [c["id"] for c in candidates]
    items_str = " ".join(f"wd:{q}" for q in qids)
    types_str = ", ".join(f"wd:{t}" for t in VALID_TYPES)
    query = SPARQL_BATCH_VALIDATE.format(items=items_str, types=types_str)

    validation_map: dict[str, dict] = {q: {"type_valid": False, "country_match": False, "has_stadium": False, "country_qid": None} for q in qids}

    try:
        await asyncio.sleep(SPARQL_DELAY)
        resp = await client.get(
            SPARQL_ENDPOINT,
            params={"query": query, "format": "json"},
            headers={"User-Agent": USER_AGENT},
            timeout=60.0,
        )
        resp.raise_for_status()
        bindings = resp.json().get("results", {}).get("bindings", [])

        expected_qid = COUNTRY_QID_MAP.get(country)

        for b in bindings:
            item_uri = b.get("item", {}).get("value", "")
            if "/entity/Q" not in item_uri:
                continue
            item_qid = item_uri.split("/")[-1]
            if item_qid not in validation_map:
                continue

            v = validation_map[item_qid]

            if b.get("type", {}).get("value"):
                v["type_valid"] = True
            if b.get("stadium", {}).get("value"):
                v["has_stadium"] = True

            for key in ["country", "locCountry"]:
                uri = b.get(key, {}).get("value", "")
                if uri and "/entity/Q" in uri:
                    found_qid = uri.split("/")[-1]
                    v["country_qid"] = found_qid
                    if expected_qid and found_qid == expected_qid:
                        v["country_match"] = True

    except Exception as e:
        logger.warning(f"Batch SPARQL validation failed: {e}")

    # Step 3: Build ranked results
    results = []
    for c in candidates:
        qid = c["id"]
        v = validation_map[qid]
        label = (c.get("label") or "").lower()

        # Name similarity bonus
        name_sim = 0.0
        if team_name.lower() in label or label in team_name.lower():
            name_sim = 0.10

        results.append({
            "qid": qid,
            "label": c.get("label"),
            "description": c.get("description"),
            "method": "wbsearch",
            **v,
            "name_sim": name_sim,
        })

    return results


# =============================================================================
# Phase 4: Confidence Scoring + Apply
# =============================================================================

def compute_confidence(
    method: str,
    type_valid: bool,
    country_match: bool,
    has_stadium: bool,
    name_sim: float,
) -> tuple[float, str]:
    """
    Compute confidence score and tier.

    Returns (score, tier) where tier is HIGH/MED/LOW.
    """
    if method == "wikipedia":
        score = 0.40  # Wikipedia-first base
    else:
        score = 0.20  # wbsearch base

    if type_valid:
        score += 0.25
    if country_match:
        score += 0.25
    if has_stadium:
        score += 0.10
    score += name_sim

    score = min(score, 1.0)

    # Tier assignment — LOW if no country match regardless of score
    if score >= 0.85:
        tier = "HIGH"
    elif score >= 0.60 and country_match:
        tier = "MED"
    else:
        tier = "LOW"

    return score, tier


async def process_team(
    team_id: int,
    team_name: str,
    country: str,
    current_qid: Optional[str],
    client: httpx.AsyncClient,
) -> dict:
    """
    Process a single team through the 4-phase reconciliation pipeline.
    """
    result = {
        "team_id": team_id,
        "team_name": team_name,
        "country": country,
        "current_qid": current_qid,
        "new_qid": None,
        "method": None,
        "confidence": 0.0,
        "tier": "LOW",
        "type_valid": False,
        "country_match": False,
        "has_stadium": False,
        "status": "not_found",
    }

    # Phase 1: Wikipedia Search
    wiki_result = await wikipedia_search_qid(team_name, country, client)

    if wiki_result:
        qid = wiki_result["qid"]
        logger.info(f"  Phase 1 (Wikipedia): {qid} — {wiki_result.get('description', '')[:60]}")

        # Phase 2: SPARQL Validation
        await asyncio.sleep(SPARQL_DELAY)
        validation = await sparql_validate_candidate(qid, country, client)

        score, tier = compute_confidence(
            "wikipedia",
            validation["type_valid"],
            validation["country_match"],
            validation["has_stadium"],
            0.0,
        )

        result.update({
            "new_qid": qid,
            "method": "wikipedia",
            "confidence": score,
            "tier": tier,
            "type_valid": validation["type_valid"],
            "country_match": validation["country_match"],
            "has_stadium": validation["has_stadium"],
            "wiki_title": wiki_result.get("title"),
            "wiki_description": wiki_result.get("description"),
        })

        if tier in ("HIGH", "MED"):
            result["status"] = "auto_applied"
            return result
        # Wikipedia found but LOW confidence — try wbsearch
        logger.info(f"  Phase 1 result LOW ({score:.2f}) — trying wbsearch fallback")

    # Phase 3: Fallback wbsearchentities
    await asyncio.sleep(WBSEARCH_DELAY)
    candidates = await wbsearch_with_validation(team_name, country, client)

    if candidates:
        # Rank by confidence
        ranked = []
        for c in candidates:
            score, tier = compute_confidence(
                "wbsearch",
                c["type_valid"],
                c["country_match"],
                c["has_stadium"],
                c["name_sim"],
            )
            ranked.append({**c, "confidence": score, "tier": tier})

        ranked.sort(key=lambda x: x["confidence"], reverse=True)
        best = ranked[0]

        logger.info(f"  Phase 3 (wbsearch): {best['qid']} ({best['confidence']:.2f}/{best['tier']}) — {(best.get('description') or '')[:60]}")

        # If wbsearch produced better result than Wikipedia LOW
        if best["confidence"] > result.get("confidence", 0):
            result.update({
                "new_qid": best["qid"],
                "method": "wbsearch",
                "confidence": best["confidence"],
                "tier": best["tier"],
                "type_valid": best["type_valid"],
                "country_match": best["country_match"],
                "has_stadium": best["has_stadium"],
                "wbsearch_label": best.get("label"),
                "wbsearch_description": best.get("description"),
                "candidates": [{"qid": c["qid"], "confidence": c.get("confidence", 0)} for c in ranked[:3]],
            })

    # Final status
    if result["new_qid"]:
        if result["tier"] in ("HIGH", "MED"):
            result["status"] = "auto_applied"
        elif result["tier"] == "LOW":
            result["status"] = "low_confidence"
        else:
            result["status"] = "manual_review"
    else:
        result["status"] = "not_found"

    return result


# =============================================================================
# Main orchestration
# =============================================================================

async def main():
    parser = argparse.ArgumentParser(description="Wikidata QID Reconciliation v2")
    parser.add_argument("--scope", type=str, default="latam",
                        help="Country scope: 'latam', 'all', or specific country name")
    parser.add_argument("--team-id", type=int, help="Process single team by ID")
    parser.add_argument("--apply", action="store_true", help="Apply corrections to DB")
    parser.add_argument("--dry-run", action="store_true", help="Dry run (default)")
    parser.add_argument("--force", action="store_true", help="Re-check even high-confidence teams")
    args = parser.parse_args()

    apply = args.apply and not args.dry_run
    if not apply:
        logger.info("DRY RUN mode (use --apply to commit changes)")

    from sqlalchemy import text
    from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
    from sqlalchemy.orm import sessionmaker

    engine = create_async_engine(DATABASE_URL, echo=False)
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    # Build scope query
    if args.team_id:
        scope_sql = "SELECT t.id, t.name, t.country, t.wikidata_id, t.wiki_source, t.wiki_confidence FROM teams t WHERE t.id = :team_id"
        params = {"team_id": args.team_id}
    else:
        force = args.force
        scope = args.scope

        conditions = ["t.country IS NOT NULL"]

        if not force:
            conditions.append("(t.wikidata_id IS NULL OR t.wiki_confidence IS NULL OR t.wiki_confidence < 0.85)")

        if scope == "latam":
            countries = ", ".join(f"'{c}'" for c in LATAM_COUNTRIES)
            conditions.append(f"t.country IN ({countries})")
        elif scope != "all":
            conditions.append(f"t.country = '{scope}'")

        where = " AND ".join(conditions)
        scope_sql = f"SELECT t.id, t.name, t.country, t.wikidata_id, t.wiki_source, t.wiki_confidence FROM teams t WHERE {where} ORDER BY t.country, t.id"
        params = {}

    # Fetch teams
    async with async_session() as session:
        result = await session.execute(text(scope_sql), params)
        teams = result.fetchall()

    logger.info(f"Found {len(teams)} teams to process (scope={args.scope or 'team-id'}, force={args.force})")

    # Process teams — collect all results first, then dedup, then apply
    all_results: list[dict] = []
    already_valid: list[dict] = []

    async with httpx.AsyncClient(headers={"User-Agent": USER_AGENT}) as client:
        for i, team in enumerate(teams):
            team_id, name, country, current_qid, wiki_source, wiki_confidence = (
                team.id, team.name, team.country, team.wikidata_id, team.wiki_source, team.wiki_confidence
            )

            logger.info(f"\n[{i+1}/{len(teams)}] {team_id} {name} ({country}) — current: {current_qid or 'NULL'}")

            # Skip if already validated at HIGH confidence and not forcing
            if not args.force and wiki_confidence and wiki_confidence >= 0.85 and current_qid:
                logger.info(f"  SKIP: already validated (confidence={wiki_confidence})")
                already_valid.append({
                    "team_id": team_id,
                    "team_name": name,
                    "country": country,
                    "qid": current_qid,
                    "confidence": wiki_confidence,
                })
                continue

            result = await process_team(team_id, name, country, current_qid, client)
            all_results.append(result)

            if result["status"] == "auto_applied":
                logger.info(f"  WOULD APPLY: {result['new_qid']} (conf={result['confidence']:.2f}, tier={result['tier']})")

    # ---------------------------------------------------------------
    # Duplicate QID dedup: if multiple teams got the same new_qid,
    # keep only the one with highest confidence; demote the rest to LOW
    # ---------------------------------------------------------------
    qid_owners: dict[str, list[dict]] = {}
    for r in all_results:
        qid = r.get("new_qid")
        if qid and r["status"] == "auto_applied":
            qid_owners.setdefault(qid, []).append(r)

    demoted_count = 0
    for qid, owners in qid_owners.items():
        if len(owners) <= 1:
            continue
        # Sort by confidence DESC — keep the best, demote the rest
        owners.sort(key=lambda x: x["confidence"], reverse=True)
        keeper = owners[0]
        for dup in owners[1:]:
            logger.warning(
                f"  DEDUP: {dup['team_id']} {dup['team_name']} shares QID {qid} "
                f"with {keeper['team_id']} {keeper['team_name']} — demoting to LOW"
            )
            dup["status"] = "low_confidence"
            dup["tier"] = "LOW"
            dup["dedup_note"] = f"QID conflict with team_id={keeper['team_id']} ({keeper['team_name']})"
            demoted_count += 1

    if demoted_count:
        logger.info(f"\n  Dedup: demoted {demoted_count} entries from auto_applied to LOW")

    # ---------------------------------------------------------------
    # Categorize results into report buckets + apply
    # ---------------------------------------------------------------
    report = {
        "auto_applied": [],
        "manual_review": [],
        "low_confidence": [],
        "not_found": [],
        "already_valid": already_valid,
    }

    for result in all_results:
        category = result["status"]
        if category == "auto_applied":
            report["auto_applied"].append(result)
        elif category == "low_confidence":
            report["low_confidence"].append(result)
        elif category == "manual_review":
            report["manual_review"].append(result)
        else:
            report["not_found"].append(result)

    # Pre-apply: check for QID collisions with existing DB entries
    if apply or not apply:  # always check, useful for dry-run report too
        async with async_session() as session:
            existing = await session.execute(
                text("SELECT id, wikidata_id FROM teams WHERE wikidata_id IS NOT NULL"),
            )
            db_qid_map: dict[str, int] = {row.wikidata_id: row.id for row in existing.fetchall()}

        collision_count = 0
        safe_auto = []
        for result in report["auto_applied"]:
            qid = result["new_qid"]
            team_id = result["team_id"]
            if qid and qid in db_qid_map and db_qid_map[qid] != team_id:
                owner_id = db_qid_map[qid]
                logger.warning(
                    f"  DB COLLISION: {team_id} {result['team_name']} wants {qid} "
                    f"but it belongs to team_id={owner_id} — demoting to LOW"
                )
                result["status"] = "low_confidence"
                result["tier"] = "LOW"
                result["dedup_note"] = f"QID {qid} already in DB for team_id={owner_id}"
                report["low_confidence"].append(result)
                collision_count += 1
            else:
                safe_auto.append(result)
        report["auto_applied"] = safe_auto

        if collision_count:
            logger.info(f"\n  DB collision check: demoted {collision_count} entries")

    # Apply if authorized
    if apply:
        applied_count = 0
        skipped_count = 0
        for result in report["auto_applied"]:
            if result["new_qid"]:
                team_id = result["team_id"]
                try:
                    async with async_session() as session:
                        async with session.begin():
                            await session.execute(
                                text("""
                                    UPDATE teams
                                    SET wikidata_id = :qid,
                                        wiki_source = 'reconciliation_v2',
                                        wiki_confidence = :confidence,
                                        wiki_matched_at = NOW()
                                    WHERE id = :team_id
                                """),
                                {"qid": result["new_qid"], "confidence": result["confidence"], "team_id": team_id},
                            )
                            await session.execute(
                                text("DELETE FROM team_wikidata_enrichment WHERE team_id = :team_id"),
                                {"team_id": team_id},
                            )
                    logger.info(f"  APPLIED: {result['team_id']} {result['team_name']} → {result['new_qid']} (conf={result['confidence']:.2f})")
                    applied_count += 1
                except Exception as e:
                    logger.error(f"  FAILED: {result['team_id']} {result['team_name']} → {result['new_qid']}: {e}")
                    skipped_count += 1
        logger.info(f"\n  Total applied: {applied_count}, skipped: {skipped_count}")

    await engine.dispose()

    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("SUMMARY")
    logger.info(f"  Auto-applied: {len(report['auto_applied'])}")
    logger.info(f"  Low confidence: {len(report['low_confidence'])}")
    logger.info(f"  Manual review: {len(report['manual_review'])}")
    logger.info(f"  Not found: {len(report['not_found'])}")
    logger.info(f"  Already valid: {len(report['already_valid'])}")

    # Save report
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    report_path = OUTPUT_DIR / f"reconciliation_v2_{ts}.json"

    # Clean up non-serializable fields
    def clean(obj):
        if isinstance(obj, dict):
            return {k: clean(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [clean(v) for v in obj]
        if isinstance(obj, float):
            return round(obj, 4)
        return obj

    with open(report_path, "w") as f:
        json.dump(clean(report), f, indent=2, default=str)
    logger.info(f"\nReport saved to {report_path}")


if __name__ == "__main__":
    asyncio.run(main())
