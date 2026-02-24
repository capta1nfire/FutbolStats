#!/usr/bin/env python3
"""
One-shot Wikidata enrichment catch-up.

Processes ALL teams with wikidata_id that don't have enrichment yet.
Rate limited to respect Wikidata API (0.2s/request = 5 req/sec).

Usage:
    python scripts/wikidata_catchup.py --dry-run
    python scripts/wikidata_catchup.py --apply
"""

import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from typing import Any, Optional
from urllib.parse import quote

import httpx
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# P0 ABE: NO hardcodear credenciales
DATABASE_URL = os.environ.get("DATABASE_URL")
if not DATABASE_URL:
    logger.error("DATABASE_URL environment variable is required")
    sys.exit(1)

# Rate limits
WIKIDATA_RATE_LIMIT = 0.2  # 5 req/sec
WIKIPEDIA_RATE_LIMIT = 0.2  # 5 req/sec

WIKIDATA_ENDPOINT = "https://query.wikidata.org/sparql"
WIKIPEDIA_API_BASE = "https://en.wikipedia.org/api/rest_v1/page/summary"

SPARQL_QUERY = """
SELECT ?team
       (SAMPLE(?teamLabel) AS ?teamLabel)
       (SAMPLE(?fullName) AS ?fullName)
       (SAMPLE(?shortName) AS ?shortName)
       (SAMPLE(?stadium) AS ?stadium)
       (SAMPLE(?stadiumLabel) AS ?stadiumLabel)
       (SAMPLE(?capacity) AS ?capacity)
       (SAMPLE(?altitude) AS ?altitude)
       (SAMPLE(?stadiumCoords) AS ?stadiumCoords)
       (SAMPLE(?adminLocation) AS ?adminLocation)
       (SAMPLE(?adminLocationLabel) AS ?adminLocationLabel)
       (SAMPLE(?website) AS ?website)
       (SAMPLE(?twitter) AS ?twitter)
       (SAMPLE(?instagram) AS ?instagram)
WHERE {{
  BIND(wd:{qid} AS ?team)

  OPTIONAL {{
    ?team wdt:P1448 ?fullName .
    FILTER(LANG(?fullName) IN ("en", "es", ""))
  }}
  OPTIONAL {{
    ?team wdt:P1813 ?shortName .
    FILTER(LANG(?shortName) IN ("en", "es", ""))
  }}
  OPTIONAL {{
    {{
      SELECT ?bestStadium (MAX(?cap) AS ?bestCapacity) (MAX(?hasCoords) AS ?coordsFlag) WHERE {{
        wd:{qid} wdt:P115 ?bestStadium .
        OPTIONAL {{ ?bestStadium wdt:P1083 ?cap . }}
        OPTIONAL {{ ?bestStadium wdt:P625 ?bcoords . }}
        BIND(IF(BOUND(?bcoords), 1, 0) AS ?hasCoords)
      }}
      GROUP BY ?bestStadium
      ORDER BY DESC(?coordsFlag) DESC(?bestCapacity)
      LIMIT 1
    }}
    BIND(?bestStadium AS ?stadium)
    BIND(?bestCapacity AS ?capacity)
    OPTIONAL {{ ?stadium wdt:P2044 ?altitude . }}
    OPTIONAL {{ ?stadium wdt:P625 ?stadiumCoords . }}
  }}
  OPTIONAL {{ ?team wdt:P131 ?adminLocation . }}
  OPTIONAL {{ ?team wdt:P856 ?website . }}
  OPTIONAL {{ ?team wdt:P2002 ?twitter . }}
  OPTIONAL {{ ?team wdt:P2003 ?instagram . }}

  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en,es". }}
}}
GROUP BY ?team
"""


async def fetch_wikidata(qid: str, client: httpx.AsyncClient) -> Optional[dict]:
    """Fetch from Wikidata SPARQL with retry."""
    query = SPARQL_QUERY.format(qid=qid)

    for attempt in range(3):
        try:
            response = await client.get(
                WIKIDATA_ENDPOINT,
                params={"query": query, "format": "json"},
                headers={"User-Agent": "FutbolStats/1.0 (contact@futbolstats.app)"},
                timeout=30.0,
            )

            if response.status_code == 429:
                retry_after = int(response.headers.get("Retry-After", 60))
                logger.warning(f"  429 rate limited, waiting {retry_after}s")
                await asyncio.sleep(retry_after)
                continue

            response.raise_for_status()
            data = response.json()

            bindings = data.get("results", {}).get("bindings", [])
            if not bindings:
                return None

            return parse_wikidata(bindings[0], data)

        except Exception as e:
            if attempt < 2:
                await asyncio.sleep(2 ** attempt)
                continue
            logger.warning(f"  Wikidata failed: {e}")
            return None

    return None


def parse_wikidata(binding: dict, raw: dict) -> dict:
    """Parse SPARQL response."""
    def get_val(key: str) -> Optional[str]:
        return binding.get(key, {}).get("value")

    def get_qid(key: str) -> Optional[str]:
        val = get_val(key)
        if val and "/entity/Q" in val:
            return val.split("/")[-1]
        return None

    # Parse coords
    lat, lon = None, None
    coords = get_val("stadiumCoords")
    if coords and coords.startswith("Point("):
        try:
            parts = coords[6:-1].split()
            lon, lat = float(parts[0]), float(parts[1])
        except:
            pass

    # Clean handles
    twitter = get_val("twitter")
    if twitter and "/" in twitter:
        twitter = twitter.rstrip("/").split("/")[-1]
    instagram = get_val("instagram")
    if instagram and "/" in instagram:
        instagram = instagram.rstrip("/").split("/")[-1]

    return {
        "raw_jsonb": raw,
        "full_name": get_val("fullName"),
        "short_name": get_val("shortName"),
        "stadium_name": get_val("stadiumLabel"),
        "stadium_wikidata_id": get_qid("stadium"),
        "stadium_capacity": int(get_val("capacity")) if get_val("capacity") else None,
        "stadium_altitude_m": int(float(get_val("altitude"))) if get_val("altitude") else None,
        "lat": lat,
        "lon": lon,
        "admin_location_label": get_val("adminLocationLabel"),
        "website": get_val("website"),
        "social_handles": {"twitter": twitter, "instagram": instagram},
        "enrichment_source": "wikidata",
    }


async def fetch_wikipedia(team_name: str, client: httpx.AsyncClient) -> Optional[dict]:
    """Fallback to Wikipedia REST API."""
    base_name = team_name.strip().replace(" ", "_")

    titles = [
        f"{base_name}_F.C.",
        f"{base_name}_FC",
        f"FC_{base_name}",
        f"{base_name}_(football_club)",
        base_name,
    ]

    for title in titles:
        try:
            encoded = quote(title, safe="")
            response = await client.get(
                f"{WIKIPEDIA_API_BASE}/{encoded}",
                headers={"User-Agent": "FutbolStats/1.0"},
                timeout=15.0,
                follow_redirects=True,
            )

            if response.status_code == 404:
                continue

            response.raise_for_status()
            data = response.json()

            if data.get("type") == "disambiguation":
                continue

            desc = (data.get("description") or "").lower()
            extract = (data.get("extract") or "").lower()

            football_words = ["football club", "soccer club", "football team", "fútbol"]
            if not any(w in desc or w in extract[:500] for w in football_words):
                continue

            # Extract full name
            full_name = None
            first_sentence = extract.split(".")[0] if extract else ""
            for pattern in ["commonly known as", "best known as", "also known as"]:
                if pattern in first_sentence.lower():
                    parts = first_sentence.split(",")
                    if parts:
                        full_name = parts[0].strip()
                    break

            return {
                "raw_jsonb": {"wikipedia": data},
                "full_name": full_name,
                "enrichment_source": "wikipedia",
            }

        except Exception:
            continue

    return None


def merge_data(wikidata: Optional[dict], wikipedia: Optional[dict]) -> dict:
    """Merge with wikidata priority."""
    base = {
        "raw_jsonb": None,
        "full_name": None,
        "short_name": None,
        "stadium_name": None,
        "stadium_wikidata_id": None,
        "stadium_capacity": None,
        "stadium_altitude_m": None,
        "lat": None,
        "lon": None,
        "admin_location_label": None,
        "website": None,
        "social_handles": {"twitter": None, "instagram": None},
        "enrichment_source": "none",
    }

    # Apply wikipedia first (lower priority)
    if wikipedia:
        for k, v in wikipedia.items():
            if k != "raw_jsonb" and v is not None and base.get(k) is None:
                base[k] = v

    # Apply wikidata (higher priority)
    if wikidata:
        for k, v in wikidata.items():
            if k == "social_handles" and v:
                for sk, sv in v.items():
                    if sv:
                        base["social_handles"][sk] = sv
            elif k != "raw_jsonb" and v is not None:
                base[k] = v

    # Composite provenance
    raw = {}
    if wikidata and wikidata.get("raw_jsonb"):
        raw["wikidata"] = wikidata["raw_jsonb"]
    if wikipedia and wikipedia.get("raw_jsonb"):
        raw["wikipedia"] = wikipedia["raw_jsonb"]
    base["raw_jsonb"] = raw if raw else None

    # Source
    if wikidata:
        base["enrichment_source"] = "wikidata+wikipedia" if wikipedia else "wikidata"
    elif wikipedia:
        base["enrichment_source"] = "wikipedia"

    return base


async def upsert(session: AsyncSession, team_id: int, wikidata_id: str, data: dict):
    """Insert or update enrichment."""
    await session.execute(
        text("""
        INSERT INTO team_wikidata_enrichment (
            team_id, wikidata_id, fetched_at, raw_jsonb,
            stadium_name, stadium_wikidata_id, stadium_capacity, stadium_altitude_m,
            lat, lon, admin_location_label,
            full_name, short_name, social_handles, website,
            enrichment_source
        ) VALUES (
            :team_id, :wikidata_id, NOW(), CAST(:raw_jsonb AS jsonb),
            :stadium_name, :stadium_wikidata_id, :stadium_capacity, :stadium_altitude_m,
            :lat, :lon, :admin_location_label,
            :full_name, :short_name, CAST(:social_handles AS jsonb), :website,
            :enrichment_source
        )
        ON CONFLICT (team_id) DO UPDATE SET
            wikidata_id = EXCLUDED.wikidata_id,
            fetched_at = NOW(),
            raw_jsonb = EXCLUDED.raw_jsonb,
            stadium_name = EXCLUDED.stadium_name,
            stadium_wikidata_id = EXCLUDED.stadium_wikidata_id,
            stadium_capacity = EXCLUDED.stadium_capacity,
            stadium_altitude_m = EXCLUDED.stadium_altitude_m,
            lat = EXCLUDED.lat,
            lon = EXCLUDED.lon,
            admin_location_label = EXCLUDED.admin_location_label,
            full_name = EXCLUDED.full_name,
            short_name = EXCLUDED.short_name,
            social_handles = EXCLUDED.social_handles,
            website = EXCLUDED.website,
            enrichment_source = EXCLUDED.enrichment_source
        """),
        {
            "team_id": team_id,
            "wikidata_id": wikidata_id,
            "raw_jsonb": json.dumps(data.get("raw_jsonb")),
            "stadium_name": data.get("stadium_name"),
            "stadium_wikidata_id": data.get("stadium_wikidata_id"),
            "stadium_capacity": data.get("stadium_capacity"),
            "stadium_altitude_m": data.get("stadium_altitude_m"),
            "lat": data.get("lat"),
            "lon": data.get("lon"),
            "admin_location_label": data.get("admin_location_label"),
            "full_name": data.get("full_name"),
            "short_name": data.get("short_name"),
            "social_handles": json.dumps(data.get("social_handles")),
            "website": data.get("website"),
            "enrichment_source": data.get("enrichment_source", "wikidata"),
        },
    )


async def main(dry_run: bool = True):
    logger.info("=" * 60)
    logger.info("Wikidata Enrichment Catch-up")
    logger.info("=" * 60)
    logger.info(f"Mode: {'DRY RUN' if dry_run else 'APPLY'}")

    engine = create_async_engine(DATABASE_URL, echo=False)
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    # Get pending teams
    async with async_session() as session:
        result = await session.execute(text("""
            SELECT t.id, t.wikidata_id, t.name
            FROM teams t
            LEFT JOIN team_wikidata_enrichment twe ON t.id = twe.team_id
            WHERE t.wikidata_id IS NOT NULL
              AND twe.team_id IS NULL
            ORDER BY t.id
        """))
        teams = result.fetchall()

    total = len(teams)
    logger.info(f"Found {total} teams pending enrichment")

    if total == 0:
        logger.info("Nothing to do!")
        return

    # Estimate time
    est_seconds = total * (WIKIDATA_RATE_LIMIT + 0.1)  # +0.1 for processing
    logger.info(f"Estimated time: {est_seconds/60:.1f} minutes")
    logger.info("")

    if dry_run:
        logger.info("DRY RUN - showing first 20 teams:")
        for t in teams[:20]:
            logger.info(f"  {t[0]:5} | {t[1]:12} | {t[2][:40]}")
        if total > 20:
            logger.info(f"  ... and {total - 20} more")
        await engine.dispose()
        return

    # Process
    metrics = {
        "total": total,
        "enriched": 0,
        "wikidata_ok": 0,
        "wikipedia_fallback": 0,
        "errors": 0,
        "started_at": datetime.utcnow().isoformat(),
    }

    async with httpx.AsyncClient() as client:
        for i, (team_id, wikidata_id, team_name) in enumerate(teams, 1):
            # Progress
            if i % 50 == 0 or i == total:
                pct = 100 * i / total
                logger.info(f"Progress: {i}/{total} ({pct:.1f}%) - enriched={metrics['enriched']}")

            # Rate limit
            await asyncio.sleep(WIKIDATA_RATE_LIMIT)

            # Fetch Wikidata
            wikidata = await fetch_wikidata(wikidata_id, client)
            if wikidata:
                metrics["wikidata_ok"] += 1

            # Wikipedia fallback if no full_name
            wikipedia = None
            if not wikidata or not wikidata.get("full_name"):
                await asyncio.sleep(WIKIPEDIA_RATE_LIMIT)
                wikipedia = await fetch_wikipedia(team_name, client)
                if wikipedia:
                    metrics["wikipedia_fallback"] += 1

            # Merge
            merged = merge_data(wikidata, wikipedia)

            # Upsert
            if wikidata or wikipedia:
                try:
                    async with async_session() as session:
                        await upsert(session, team_id, wikidata_id, merged)
                        await session.commit()
                    metrics["enriched"] += 1
                except Exception as e:
                    logger.error(f"  DB error for {team_id}: {e}")
                    metrics["errors"] += 1
            else:
                metrics["errors"] += 1

    await engine.dispose()

    metrics["finished_at"] = datetime.utcnow().isoformat()

    logger.info("")
    logger.info("=" * 60)
    logger.info("COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total processed: {metrics['total']}")
    logger.info(f"Enriched: {metrics['enriched']}")
    logger.info(f"Wikidata OK: {metrics['wikidata_ok']}")
    logger.info(f"Wikipedia fallback: {metrics['wikipedia_fallback']}")
    logger.info(f"Errors: {metrics['errors']}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", default=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    dry_run = not args.apply
    asyncio.run(main(dry_run=dry_run))
