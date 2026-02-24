#!/usr/bin/env python3
"""
Post-hoc Wikidata Validation (E2).

Checks ALL teams with wikidata_id for:
  1. Country mismatch (P17 / P131*/P17 vs teams.country)
  2. Vandalism patterns in enrichment names
  3. Missing enrichment data

Usage:
    python scripts/validate_wikidata_posthoc.py
    python scripts/validate_wikidata_posthoc.py --country Bolivia

Author: Master (per ABE reconciliation_v2 E2, 2026-02-23)
"""

import argparse
import asyncio
import json
import logging
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import httpx

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wikidata_common import COUNTRY_QID_MAP, REVERSE_COUNTRY_MAP, SPARQL_ENDPOINT

# Database connection (P0-SEC)
DATABASE_URL = os.environ.get("DATABASE_URL")
if not DATABASE_URL:
    print("ERROR: DATABASE_URL environment variable is required", file=sys.stderr)
    sys.exit(1)
if DATABASE_URL.startswith("postgresql://"):
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://", 1)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("logs")
OUTPUT_DIR.mkdir(exist_ok=True)

USER_AGENT = "BonJogo/2.0 (wikidata-posthoc-validation)"

# Vandalism patterns in enrichment names
VANDALISM_PATTERNS = [
    re.compile(r'[^\w\s\.\-\'\(\)áéíóúñüàèìòùâêîôûãõçäöüßæøå,/&]'),  # unusual chars
    re.compile(r'^[A-Z\s]{10,}$'),      # ALL CAPS long string
    re.compile(r'(?i)fuck|shit|porn'),   # profanity
    re.compile(r'https?://'),            # URLs in name
    re.compile(r'\d{5,}'),              # long numbers
]

# Batch SPARQL for P17 with P131*/P17 fallback
SPARQL_BATCH_P17 = """
SELECT ?item ?country ?locCountry WHERE {{
  VALUES ?item {{ {items} }}
  OPTIONAL {{ ?item wdt:P17 ?country . }}
  OPTIONAL {{ ?item wdt:P131*/wdt:P17 ?locCountry . }}
}}
"""

BATCH_SIZE = 50


async def fetch_batch_p17(qids: list[str], client: httpx.AsyncClient) -> dict[str, str | None]:
    """Batch-fetch P17 country QIDs for a list of Wikidata QIDs."""
    items_str = " ".join(f"wd:{q}" for q in qids)
    query = SPARQL_BATCH_P17.format(items=items_str)

    result: dict[str, str | None] = {q: None for q in qids}

    try:
        resp = await client.get(
            SPARQL_ENDPOINT,
            params={"query": query, "format": "json"},
            headers={"User-Agent": USER_AGENT},
            timeout=60.0,
        )
        resp.raise_for_status()
        bindings = resp.json().get("results", {}).get("bindings", [])

        for b in bindings:
            item_uri = b.get("item", {}).get("value", "")
            if "/entity/Q" not in item_uri:
                continue
            item_qid = item_uri.split("/")[-1]
            if item_qid not in result:
                continue

            # P17 direct first, fallback P131*/P17
            for key in ["country", "locCountry"]:
                uri = b.get(key, {}).get("value", "")
                if uri and "/entity/Q" in uri:
                    found = uri.split("/")[-1]
                    if result[item_qid] is None:
                        result[item_qid] = found

    except Exception as e:
        logger.warning(f"SPARQL batch P17 failed: {e}")

    return result


def check_vandalism(text: str | None) -> str | None:
    """Check for vandalism patterns. Returns pattern description if found."""
    if not text:
        return None
    for pattern in VANDALISM_PATTERNS:
        if pattern.search(text):
            return f"Pattern match: {pattern.pattern[:40]}"
    return None


async def main():
    parser = argparse.ArgumentParser(description="Post-hoc Wikidata Validation")
    parser.add_argument("--country", type=str, help="Filter by country")
    args = parser.parse_args()

    from sqlalchemy import text
    from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
    from sqlalchemy.orm import sessionmaker

    engine = create_async_engine(DATABASE_URL, echo=False)
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    # Fetch all teams with wikidata_id
    country_filter = ""
    params = {}
    if args.country:
        country_filter = " AND t.country = :country"
        params = {"country": args.country}

    async with async_session() as session:
        result = await session.execute(
            text(f"""
                SELECT t.id, t.name, t.country, t.wikidata_id, t.wiki_source, t.wiki_confidence
                FROM teams t
                WHERE t.wikidata_id IS NOT NULL
                  AND t.country IS NOT NULL
                  {country_filter}
                ORDER BY t.country, t.id
            """),
            params,
        )
        teams = result.fetchall()

        # Fetch enrichment data for vandalism check
        enrichment_result = await session.execute(
            text("SELECT team_id, full_name, short_name FROM team_wikidata_enrichment"),
        )
        enrichments = {row.team_id: {"full_name": row.full_name, "short_name": row.short_name}
                       for row in enrichment_result.fetchall()}

    await engine.dispose()

    logger.info(f"Checking {len(teams)} teams with wikidata_id")

    # Build QID → team mapping
    team_lookup: dict[str, list[dict]] = {}
    for t in teams:
        qid = t.wikidata_id
        entry = {
            "team_id": t.id,
            "team_name": t.name,
            "country": t.country,
            "wiki_source": t.wiki_source,
            "wiki_confidence": t.wiki_confidence,
        }
        if qid not in team_lookup:
            team_lookup[qid] = []
        team_lookup[qid].append(entry)

    unique_qids = list(team_lookup.keys())
    logger.info(f"Unique QIDs to check: {len(unique_qids)}")

    # Batch SPARQL for P17
    p17_map: dict[str, str | None] = {}
    async with httpx.AsyncClient() as client:
        for i in range(0, len(unique_qids), BATCH_SIZE):
            batch = unique_qids[i:i + BATCH_SIZE]
            batch_num = i // BATCH_SIZE + 1
            total_batches = (len(unique_qids) + BATCH_SIZE - 1) // BATCH_SIZE
            logger.info(f"SPARQL batch {batch_num}/{total_batches} ({len(batch)} QIDs)")

            batch_result = await fetch_batch_p17(batch, client)
            p17_map.update(batch_result)

            if i + BATCH_SIZE < len(unique_qids):
                await asyncio.sleep(2.0)

    # Analyze results
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_checked": len(teams),
        "correct": [],
        "mismatch": [],
        "missing_p17": [],
        "unmapped_country": [],
        "vandalism_suspected": [],
        "no_enrichment": [],
    }

    for t in teams:
        team_id = t.id
        name = t.name
        country = t.country
        qid = t.wikidata_id

        # Country mismatch check
        expected_qid = COUNTRY_QID_MAP.get(country)
        actual_p17 = p17_map.get(qid)

        if not expected_qid:
            report["unmapped_country"].append({
                "team_id": team_id,
                "team_name": name,
                "country": country,
                "qid": qid,
                "note": f"Country '{country}' not in COUNTRY_QID_MAP",
            })
        elif actual_p17 is None:
            report["missing_p17"].append({
                "team_id": team_id,
                "team_name": name,
                "country": country,
                "qid": qid,
            })
        elif actual_p17 != expected_qid:
            actual_country_name = REVERSE_COUNTRY_MAP.get(actual_p17, f"Unknown ({actual_p17})")
            report["mismatch"].append({
                "team_id": team_id,
                "team_name": name,
                "country": country,
                "qid": qid,
                "expected_p17": expected_qid,
                "actual_p17": actual_p17,
                "actual_country": actual_country_name,
                "wiki_source": t.wiki_source,
            })
        else:
            report["correct"].append({
                "team_id": team_id,
                "team_name": name,
                "country": country,
                "qid": qid,
            })

        # Vandalism check
        enr = enrichments.get(team_id)
        if enr:
            for field in ["full_name", "short_name"]:
                issue = check_vandalism(enr.get(field))
                if issue:
                    report["vandalism_suspected"].append({
                        "team_id": team_id,
                        "team_name": name,
                        "field": field,
                        "value": enr[field],
                        "issue": issue,
                    })
                    break
        else:
            report["no_enrichment"].append({
                "team_id": team_id,
                "team_name": name,
                "qid": qid,
            })

    # Summary
    correct_count = len(report["correct"])
    logger.info(f"\n{'='*60}")
    logger.info("POST-HOC VALIDATION SUMMARY")
    logger.info(f"  Total checked: {len(teams)}")
    logger.info(f"  Correct (P17 match): {correct_count}")
    logger.info(f"  Country mismatch: {len(report['mismatch'])}")
    logger.info(f"  Missing P17: {len(report['missing_p17'])}")
    logger.info(f"  Unmapped country: {len(report['unmapped_country'])}")
    logger.info(f"  Vandalism suspected: {len(report['vandalism_suspected'])}")
    logger.info(f"  No enrichment data: {len(report['no_enrichment'])}")

    if report["mismatch"]:
        logger.info("\nCOUNTRY MISMATCHES:")
        for m in report["mismatch"]:
            logger.info(f"  {m['team_id']:5} {m['team_name'][:30]:30} | {m['country']:15} | {m['qid']} → P17={m['actual_country']}")

    if report["vandalism_suspected"]:
        logger.info("\nVANDALISM SUSPECTED:")
        for v in report["vandalism_suspected"]:
            logger.info(f"  {v['team_id']:5} {v['team_name'][:30]:30} | {v['field']}: {v['value'][:50]}")

    # Save report (exclude correct list for brevity)
    report_slim = {k: v for k, v in report.items() if k != "correct"}
    report_slim["correct_count"] = correct_count

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    report_path = OUTPUT_DIR / f"posthoc_validation_{ts}.json"
    with open(report_path, "w") as f:
        json.dump(report_slim, f, indent=2, default=str)
    logger.info(f"\nReport saved to {report_path}")


if __name__ == "__main__":
    asyncio.run(main())
