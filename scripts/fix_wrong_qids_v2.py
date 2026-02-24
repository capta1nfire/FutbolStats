#!/usr/bin/env python3
"""
Fix 5 wrong Wikidata QIDs identified in CONMEBOL exhaustive audit (2026-02-23).

Root cause: wbsearchentities returns homonyms (global popularity, no country filter).
Fix: Verify correct QID via Wikipedia REST API, then apply.

Prerequisites: Overrides for 4227/4230 already migrated to team_enrichment_overrides.

Usage:
    python scripts/fix_wrong_qids_v2.py --dry-run   # Verify only (default)
    python scripts/fix_wrong_qids_v2.py --apply      # Apply corrections
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from datetime import datetime

import httpx

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DATABASE_URL = os.environ.get("DATABASE_URL")
if not DATABASE_URL:
    logger.error("DATABASE_URL environment variable is required")
    sys.exit(1)
if DATABASE_URL.startswith("postgresql://"):
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://", 1)

# 5 wrong QIDs pending fix (4229/4307 already fixed in prior session)
CORRECTIONS = {
    4325: {
        "old": "Q94571",
        "name": "Oriente Petrolero",
        "country": "Bolivia",
        "wiki_title": "Oriente_Petrolero",
    },
    4129: {
        "old": "Q1083312",
        "name": "Club Queretaro",
        "country": "Mexico",
        "wiki_title": "Querétaro_F.C.",
    },
    4227: {
        "old": "Q1103684",
        "name": "Libertad Asuncion",
        "country": "Paraguay",
        "wiki_title": "Club_Libertad",
    },
    4230: {
        "old": "Q1023208",
        "name": "Olimpia",
        "country": "Paraguay",
        "wiki_title": "Club_Olimpia",
    },
    6152: {
        "old": "Q122972609",
        "name": "Academia Anzoátegui",
        "country": "Venezuela",
        "wiki_title": "Anzoátegui_F.C.",  # Founded 2021, same org as Academia Anzoátegui
    },
}

WIKIPEDIA_API = "https://en.wikipedia.org/api/rest_v1/page/summary"


async def fetch_qid_from_wikipedia(client: httpx.AsyncClient, wiki_title: str) -> dict | None:
    """Fetch QID from Wikipedia REST API via wikibase_item field."""
    url = f"{WIKIPEDIA_API}/{wiki_title}"
    try:
        resp = await client.get(url)
        if resp.status_code == 200:
            data = resp.json()
            return {
                "qid": data.get("wikibase_item"),
                "title": data.get("title"),
                "description": data.get("description", ""),
                "extract": data.get("extract", "")[:200],
            }
        else:
            logger.warning(f"  Wikipedia API returned {resp.status_code} for {wiki_title}")
            return None
    except Exception as e:
        logger.error(f"  Error fetching {wiki_title}: {e}")
        return None


async def search_wikipedia_for_team(client: httpx.AsyncClient, team_name: str, country: str) -> dict | None:
    """Search Wikipedia for team when no wiki_title is available."""
    search_url = "https://en.wikipedia.org/w/api.php"
    queries = [
        f"{team_name} {country} football club",
        f"{team_name} football",
        team_name,
    ]
    for query in queries:
        params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "srlimit": 5,
            "format": "json",
        }
        try:
            resp = await client.get(search_url, params=params)
            if resp.status_code != 200:
                continue
            results = resp.json().get("query", {}).get("search", [])
            for result in results:
                title = result["title"].replace(" ", "_")
                wiki_data = await fetch_qid_from_wikipedia(client, title)
                if wiki_data and wiki_data.get("qid"):
                    desc = wiki_data.get("description", "").lower()
                    if any(kw in desc for kw in ["football", "soccer", "fútbol", "association"]):
                        logger.info(f"  Found via search '{query}': {wiki_data['title']} → {wiki_data['qid']}")
                        return wiki_data
                await asyncio.sleep(0.2)
        except Exception as e:
            logger.warning(f"  Search error for '{query}': {e}")
    return None


async def main(apply: bool = False):
    from sqlalchemy import text
    from sqlalchemy.ext.asyncio import create_async_engine

    engine = create_async_engine(DATABASE_URL, echo=False)
    results = {"verified": [], "failed": [], "skipped": []}

    async with httpx.AsyncClient(timeout=30, headers={"User-Agent": "BonJogo/1.0 (wikidata-fix)"}) as client:
        for team_id, info in CORRECTIONS.items():
            logger.info(f"\n{'='*60}")
            logger.info(f"Team {team_id}: {info['name']} ({info['country']})")
            logger.info(f"  Current QID: {info['old']}")

            wiki_data = None
            if info["wiki_title"]:
                wiki_data = await fetch_qid_from_wikipedia(client, info["wiki_title"])
            else:
                logger.info(f"  No wiki_title — searching Wikipedia...")
                wiki_data = await search_wikipedia_for_team(client, info["name"], info["country"])

            if not wiki_data or not wiki_data.get("qid"):
                logger.warning(f"  FAILED: Could not resolve QID for {info['name']}")
                results["failed"].append({
                    "team_id": team_id,
                    "name": info["name"],
                    "reason": "no_qid_found",
                })
                continue

            new_qid = wiki_data["qid"]
            logger.info(f"  Wikipedia QID: {new_qid}")
            logger.info(f"  Description: {wiki_data.get('description', 'N/A')}")

            if new_qid == info["old"]:
                logger.info(f"  SKIP: QID unchanged (already correct or same homonym)")
                results["skipped"].append({
                    "team_id": team_id,
                    "name": info["name"],
                    "qid": new_qid,
                    "reason": "qid_unchanged",
                })
                continue

            logger.info(f"  CORRECTION: {info['old']} → {new_qid}")
            results["verified"].append({
                "team_id": team_id,
                "name": info["name"],
                "old_qid": info["old"],
                "new_qid": new_qid,
                "wiki_title": wiki_data.get("title"),
                "description": wiki_data.get("description"),
            })

            if apply:
                async with engine.begin() as conn:
                    # Update teams table
                    await conn.execute(
                        text("""
                            UPDATE teams
                            SET wikidata_id = :new_qid,
                                wiki_source = 'reconciliation_v2_fix',
                                wiki_confidence = 1.0,
                                wiki_matched_at = NOW()
                            WHERE id = :team_id
                        """),
                        {"new_qid": new_qid, "team_id": team_id},
                    )
                    # Delete stale enrichment (overrides already migrated for 4227/4230)
                    result = await conn.execute(
                        text("DELETE FROM team_wikidata_enrichment WHERE team_id = :team_id"),
                        {"team_id": team_id},
                    )
                    logger.info(f"  APPLIED: teams.wikidata_id={new_qid}, deleted {result.rowcount} enrichment rows")

            await asyncio.sleep(0.3)

    await engine.dispose()

    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("SUMMARY")
    logger.info(f"  Verified & {'APPLIED' if apply else 'DRY-RUN'}: {len(results['verified'])}")
    logger.info(f"  Failed: {len(results['failed'])}")
    logger.info(f"  Skipped (unchanged): {len(results['skipped'])}")

    for r in results["verified"]:
        logger.info(f"  {r['team_id']} {r['name']}: {r['old_qid']} → {r['new_qid']}")
    for r in results["failed"]:
        logger.warning(f"  {r['team_id']} {r['name']}: {r['reason']}")

    # Save report
    os.makedirs("logs", exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    report_path = f"logs/fix_wrong_qids_v2_{ts}.json"
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nReport saved to {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fix wrong Wikidata QIDs (v2)")
    parser.add_argument("--apply", action="store_true", help="Apply corrections (default: dry-run)")
    parser.add_argument("--dry-run", action="store_true", help="Dry run only (default)")
    args = parser.parse_args()

    apply = args.apply and not args.dry_run
    if not apply:
        logger.info("DRY RUN mode (use --apply to commit changes)")

    asyncio.run(main(apply=apply))
