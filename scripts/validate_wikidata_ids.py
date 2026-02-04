#!/usr/bin/env python3
"""
Wikidata ID Validation & Reconciliation Script.

Based on Kimi recommendation: Hybrid approach (SPARQL validation + API reconciliation)

Phases:
1. SPARQL batch validation (50 QIDs per query) - identify invalid QIDs
2. API reconciliation (wbsearchentities) - get correct QIDs for invalid ones
3. Auto-accept high confidence, flag low confidence for manual review

Usage:
    python scripts/validate_wikidata_ids.py --phase 1  # Validate only
    python scripts/validate_wikidata_ids.py --phase 2  # Reconcile invalid
    python scripts/validate_wikidata_ids.py --phase 3  # Apply corrections
    python scripts/validate_wikidata_ids.py --all      # Run all phases

Author: Master (per Kimi recommendation)
Date: 2026-02-05
"""

import argparse
import asyncio
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import httpx

# Database connection
DATABASE_URL = "postgresql+asyncpg://postgres:heFyxqRYCUMNkVSCgcpXHprpjAPcfJAQ@maglev.proxy.rlwy.net:24997/railway"

# Wikidata endpoints
SPARQL_ENDPOINT = "https://query.wikidata.org/sparql"
WIKIDATA_API = "https://www.wikidata.org/w/api.php"

# Valid entity types for football teams
VALID_TYPES = [
    "Q476028",      # association football club
    "Q6979593",     # men's national association football team
    "Q15944511",    # women's national association football team
    "Q103229495",   # association football team (generic)
    "Q17270031",    # national under-21 football team
    "Q1194951",     # national under-23 football team
]

# Rate limiting
SPARQL_DELAY = 5.0      # 1 request every 5 seconds
RECONCILE_DELAY = 3.0   # 1 request every 3 seconds (conservative)

# Confidence thresholds
AUTO_ACCEPT_SCORE = 0.95
MANUAL_REVIEW_SCORE = 0.70

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Output files
OUTPUT_DIR = Path("logs")
OUTPUT_DIR.mkdir(exist_ok=True)


# =============================================================================
# Phase 1: SPARQL Batch Validation
# =============================================================================

SPARQL_VALIDATION_QUERY = """
SELECT ?item ?itemLabel ?type WHERE {{
  VALUES ?item {{ {qids} }}
  ?item wdt:P31/wdt:P279* ?type .
  FILTER (?type IN ({types}))
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en,es". }}
}}
"""


async def validate_qids_batch(
    qids: list[str],
    client: httpx.AsyncClient,
) -> set[str]:
    """
    Validate a batch of QIDs via SPARQL.

    Returns set of valid QIDs (those that are football teams).
    QIDs not in the result set are invalid.
    """
    # Format QIDs for SPARQL
    qids_str = " ".join(f"wd:{qid}" for qid in qids)
    types_str = ", ".join(f"wd:{t}" for t in VALID_TYPES)

    query = SPARQL_VALIDATION_QUERY.format(qids=qids_str, types=types_str)

    try:
        response = await client.get(
            SPARQL_ENDPOINT,
            params={"query": query, "format": "json"},
            headers={"User-Agent": "FutbolStats/1.0 (wikidata-validation)"},
            timeout=60.0,
        )
        response.raise_for_status()
        data = response.json()

        # Extract valid QIDs from results
        valid = set()
        for binding in data.get("results", {}).get("bindings", []):
            item_uri = binding.get("item", {}).get("value", "")
            if "/entity/Q" in item_uri:
                qid = item_uri.split("/")[-1]
                valid.add(qid)

        return valid

    except Exception as e:
        logger.error(f"SPARQL validation failed: {e}")
        return set()


async def phase1_validate_all(session) -> dict[str, Any]:
    """
    Phase 1: Validate all wikidata_ids in teams table.

    Returns dict with valid_qids, invalid_qids, and stats.
    """
    from sqlalchemy import text

    logger.info("=" * 60)
    logger.info("PHASE 1: SPARQL Batch Validation")
    logger.info("=" * 60)

    # Get all teams with wikidata_id
    result = await session.execute(text("""
        SELECT id, name, wikidata_id, country
        FROM teams
        WHERE wikidata_id IS NOT NULL
        ORDER BY
            CASE WHEN country IS NULL THEN 1 ELSE 0 END,  -- Clubs first
            id
    """))
    teams = result.fetchall()

    logger.info(f"Found {len(teams)} teams with wikidata_id")

    # Build team lookup
    team_lookup = {t.wikidata_id: {"id": t.id, "name": t.name, "country": t.country}
                   for t in teams}
    all_qids = list(team_lookup.keys())

    # Validate in batches of 50
    BATCH_SIZE = 50
    valid_qids = set()

    async with httpx.AsyncClient() as client:
        for i in range(0, len(all_qids), BATCH_SIZE):
            batch = all_qids[i:i + BATCH_SIZE]
            logger.info(f"Validating batch {i // BATCH_SIZE + 1}/{(len(all_qids) + BATCH_SIZE - 1) // BATCH_SIZE} ({len(batch)} QIDs)")

            batch_valid = await validate_qids_batch(batch, client)
            valid_qids.update(batch_valid)

            logger.info(f"  Valid: {len(batch_valid)}/{len(batch)}")

            # Rate limit
            if i + BATCH_SIZE < len(all_qids):
                await asyncio.sleep(SPARQL_DELAY)

    # Compute invalid
    invalid_qids = set(all_qids) - valid_qids

    # Build detailed results
    invalid_teams = []
    for qid in invalid_qids:
        team = team_lookup[qid]
        invalid_teams.append({
            "team_id": team["id"],
            "team_name": team["name"],
            "country": team["country"],
            "wikidata_id": qid,
            "is_national": team["country"] is None,
        })

    # Sort: clubs first (have country), then nationals
    invalid_teams.sort(key=lambda x: (x["is_national"], x["team_id"]))

    results = {
        "timestamp": datetime.utcnow().isoformat(),
        "total_teams": len(teams),
        "valid_count": len(valid_qids),
        "invalid_count": len(invalid_qids),
        "valid_pct": round(100 * len(valid_qids) / len(teams), 1),
        "invalid_teams": invalid_teams,
    }

    # Save results
    output_file = OUTPUT_DIR / f"wikidata_validation_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    logger.info("")
    logger.info(f"Results saved to: {output_file}")
    logger.info(f"Total: {len(teams)} | Valid: {len(valid_qids)} ({results['valid_pct']}%) | Invalid: {len(invalid_qids)}")

    # Show sample of invalid
    logger.info("")
    logger.info("Sample invalid teams (first 10):")
    for team in invalid_teams[:10]:
        logger.info(f"  {team['team_id']:5} | {team['team_name'][:30]:30} | {team['wikidata_id']} | {'NATIONAL' if team['is_national'] else team['country']}")

    return results


# =============================================================================
# Phase 2: Reconciliation (wbsearchentities API)
# =============================================================================

async def reconcile_team(
    team_name: str,
    is_national: bool,
    country: Optional[str],
    client: httpx.AsyncClient,
) -> Optional[dict[str, Any]]:
    """
    Reconcile a team name to Wikidata QID using wbsearchentities.

    Returns dict with qid, label, description, confidence.
    """
    # Build search term (heuristic for nationals)
    if is_national:
        search_term = f"{team_name} national football team"
    else:
        search_term = team_name

    params = {
        "action": "wbsearchentities",
        "search": search_term,
        "language": "en",
        "type": "item",
        "format": "json",
        "limit": 5,
    }

    try:
        response = await client.get(
            WIKIDATA_API,
            params=params,
            headers={"User-Agent": "FutbolStats/1.0 (wikidata-reconciliation)"},
            timeout=30.0,
        )
        response.raise_for_status()
        data = response.json()

        candidates = data.get("search", [])
        if not candidates:
            return None

        # Score candidates
        for c in candidates:
            desc = (c.get("description") or "").lower()
            label = (c.get("label") or "").lower()

            # Calculate confidence
            confidence = 0.5  # Base

            # Type match
            if is_national:
                if "national" in desc and ("football" in desc or "soccer" in desc):
                    confidence += 0.4
                elif "national team" in desc:
                    confidence += 0.3
            else:
                if "football club" in desc or "soccer club" in desc:
                    confidence += 0.4
                elif "football" in desc:
                    confidence += 0.2

            # Name similarity
            if team_name.lower() in label or label in team_name.lower():
                confidence += 0.1

            # Country match (for clubs)
            if country and country.lower() in desc:
                confidence += 0.1

            c["confidence"] = min(confidence, 1.0)

        # Sort by confidence
        candidates.sort(key=lambda x: x.get("confidence", 0), reverse=True)
        best = candidates[0]

        return {
            "qid": best.get("id"),
            "label": best.get("label"),
            "description": best.get("description"),
            "confidence": best.get("confidence", 0.5),
            "all_candidates": [{"qid": c["id"], "label": c.get("label"), "confidence": c.get("confidence")}
                              for c in candidates[:3]],
        }

    except Exception as e:
        logger.warning(f"Reconciliation failed for {team_name}: {e}")
        return None


async def phase2_reconcile(validation_file: str) -> dict[str, Any]:
    """
    Phase 2: Reconcile invalid QIDs.

    Reads validation results from Phase 1 and attempts to find correct QIDs.
    """
    logger.info("=" * 60)
    logger.info("PHASE 2: API Reconciliation")
    logger.info("=" * 60)

    # Load validation results
    with open(validation_file) as f:
        validation = json.load(f)

    invalid_teams = validation.get("invalid_teams", [])
    logger.info(f"Found {len(invalid_teams)} invalid teams to reconcile")

    # Separate by priority: clubs first, then nationals
    clubs = [t for t in invalid_teams if not t["is_national"]]
    nationals = [t for t in invalid_teams if t["is_national"]]

    logger.info(f"  Clubs: {len(clubs)}")
    logger.info(f"  Nationals: {len(nationals)}")

    # Process in priority order
    reconciled = []
    failed = []

    async with httpx.AsyncClient() as client:
        # Process clubs first
        for i, team in enumerate(clubs + nationals):
            logger.info(f"[{i+1}/{len(invalid_teams)}] Reconciling: {team['team_name']}")

            result = await reconcile_team(
                team["team_name"],
                team["is_national"],
                team.get("country"),
                client,
            )

            if result:
                entry = {
                    **team,
                    "old_qid": team["wikidata_id"],
                    "new_qid": result["qid"],
                    "new_label": result["label"],
                    "new_description": result["description"],
                    "confidence": result["confidence"],
                    "candidates": result["all_candidates"],
                    "status": "auto_accept" if result["confidence"] >= AUTO_ACCEPT_SCORE
                             else "manual_review" if result["confidence"] >= MANUAL_REVIEW_SCORE
                             else "low_confidence",
                }
                reconciled.append(entry)

                status_emoji = "✓" if entry["status"] == "auto_accept" else "?" if entry["status"] == "manual_review" else "✗"
                logger.info(f"  {status_emoji} {result['qid']} ({result['confidence']:.2f}) - {result['label'][:40]}")
            else:
                failed.append({
                    **team,
                    "status": "not_found",
                })
                logger.info(f"  ✗ No candidates found")

            # Rate limit
            if i + 1 < len(invalid_teams):
                await asyncio.sleep(RECONCILE_DELAY)

    # Summary
    auto_accept = [r for r in reconciled if r["status"] == "auto_accept"]
    manual_review = [r for r in reconciled if r["status"] == "manual_review"]
    low_confidence = [r for r in reconciled if r["status"] == "low_confidence"]

    results = {
        "timestamp": datetime.utcnow().isoformat(),
        "total_invalid": len(invalid_teams),
        "reconciled": len(reconciled),
        "failed": len(failed),
        "auto_accept": len(auto_accept),
        "manual_review": len(manual_review),
        "low_confidence": len(low_confidence),
        "corrections": reconciled,
        "not_found": failed,
    }

    # Save results
    output_file = OUTPUT_DIR / f"wikidata_reconciliation_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    logger.info("")
    logger.info(f"Results saved to: {output_file}")
    logger.info(f"Summary:")
    logger.info(f"  Auto-accept (>={AUTO_ACCEPT_SCORE}): {len(auto_accept)}")
    logger.info(f"  Manual review ({MANUAL_REVIEW_SCORE}-{AUTO_ACCEPT_SCORE}): {len(manual_review)}")
    logger.info(f"  Low confidence (<{MANUAL_REVIEW_SCORE}): {len(low_confidence)}")
    logger.info(f"  Not found: {len(failed)}")

    return results


# =============================================================================
# Phase 3: Apply Corrections
# =============================================================================

async def phase3_apply(reconciliation_file: str, dry_run: bool = True) -> dict[str, Any]:
    """
    Phase 3: Apply corrections to database.

    Only applies auto_accept corrections unless --force is specified.
    """
    from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy import text

    logger.info("=" * 60)
    logger.info(f"PHASE 3: Apply Corrections {'(DRY RUN)' if dry_run else '(LIVE)'}")
    logger.info("=" * 60)

    # Load reconciliation results
    with open(reconciliation_file) as f:
        reconciliation = json.load(f)

    corrections = reconciliation.get("corrections", [])
    auto_accept = [c for c in corrections if c["status"] == "auto_accept"]

    logger.info(f"Total corrections: {len(corrections)}")
    logger.info(f"Auto-accept: {len(auto_accept)}")

    if dry_run:
        logger.info("")
        logger.info("DRY RUN - No changes will be made")
        logger.info("Corrections that would be applied:")
        for c in auto_accept[:20]:
            logger.info(f"  {c['team_id']:5} | {c['team_name'][:25]:25} | {c['old_qid']} -> {c['new_qid']} ({c['confidence']:.2f})")
        if len(auto_accept) > 20:
            logger.info(f"  ... and {len(auto_accept) - 20} more")
        return {"dry_run": True, "would_apply": len(auto_accept)}

    # Apply corrections - commit each update individually to avoid transaction abort
    engine = create_async_engine(DATABASE_URL, echo=False)
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    applied = 0
    errors = 0
    skipped = 0

    for c in auto_accept:
        async with async_session() as session:
            try:
                # Update teams table
                result = await session.execute(
                    text("UPDATE teams SET wikidata_id = :new_qid WHERE id = :team_id AND wikidata_id = :old_qid"),
                    {"new_qid": c["new_qid"], "team_id": c["team_id"], "old_qid": c["old_qid"]}
                )

                if result.rowcount == 0:
                    skipped += 1
                    logger.info(f"  ~ {c['team_name']}: skipped (already updated or old_qid mismatch)")
                    await session.rollback()
                    continue

                # Delete old enrichment if exists
                await session.execute(
                    text("DELETE FROM team_wikidata_enrichment WHERE team_id = :team_id"),
                    {"team_id": c["team_id"]}
                )

                await session.commit()
                applied += 1
                logger.info(f"  ✓ {c['team_name']}: {c['old_qid']} -> {c['new_qid']}")

            except Exception as e:
                await session.rollback()
                errors += 1
                logger.error(f"  ✗ {c['team_name']}: {e}")

    await engine.dispose()
    logger.info(f"  Skipped (already updated): {skipped}")

    results = {
        "timestamp": datetime.utcnow().isoformat(),
        "applied": applied,
        "errors": errors,
    }

    logger.info("")
    logger.info(f"Applied: {applied} | Errors: {errors}")

    return results


# =============================================================================
# Main
# =============================================================================

async def main():
    parser = argparse.ArgumentParser(description="Wikidata ID Validation & Reconciliation")
    parser.add_argument("--phase", type=int, choices=[1, 2, 3], help="Run specific phase")
    parser.add_argument("--all", action="store_true", help="Run all phases")
    parser.add_argument("--validation-file", type=str, help="Validation file for phase 2")
    parser.add_argument("--reconciliation-file", type=str, help="Reconciliation file for phase 3")
    parser.add_argument("--apply", action="store_true", help="Actually apply corrections (phase 3)")
    args = parser.parse_args()

    if args.phase == 1 or args.all:
        from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
        from sqlalchemy.orm import sessionmaker

        engine = create_async_engine(DATABASE_URL, echo=False)
        async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

        async with async_session() as session:
            results = await phase1_validate_all(session)

        await engine.dispose()

        if args.all:
            validation_file = OUTPUT_DIR / f"wikidata_validation_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
            # Use the most recent validation file
            validation_files = sorted(OUTPUT_DIR.glob("wikidata_validation_*.json"))
            if validation_files:
                args.validation_file = str(validation_files[-1])

    if args.phase == 2 or args.all:
        if not args.validation_file:
            # Find most recent validation file
            validation_files = sorted(OUTPUT_DIR.glob("wikidata_validation_*.json"))
            if not validation_files:
                logger.error("No validation file found. Run phase 1 first.")
                return
            args.validation_file = str(validation_files[-1])

        logger.info(f"Using validation file: {args.validation_file}")
        results = await phase2_reconcile(args.validation_file)

        if args.all:
            reconciliation_files = sorted(OUTPUT_DIR.glob("wikidata_reconciliation_*.json"))
            if reconciliation_files:
                args.reconciliation_file = str(reconciliation_files[-1])

    if args.phase == 3 or args.all:
        if not args.reconciliation_file:
            reconciliation_files = sorted(OUTPUT_DIR.glob("wikidata_reconciliation_*.json"))
            if not reconciliation_files:
                logger.error("No reconciliation file found. Run phase 2 first.")
                return
            args.reconciliation_file = str(reconciliation_files[-1])

        logger.info(f"Using reconciliation file: {args.reconciliation_file}")
        results = await phase3_apply(args.reconciliation_file, dry_run=not args.apply)


if __name__ == "__main__":
    asyncio.run(main())
