#!/usr/bin/env python3
"""
Backfill enrichment data for Ecuador teams using corrected SPARQL query.

Tests Phase 1 fix (rdfs:label + MAX capacity heuristic).

Usage:
    python3 scripts/backfill_ecuador.py          # Dry-run
    python3 scripts/backfill_ecuador.py --apply  # Apply changes
"""

import argparse
import json
import logging
import sys
import time

import httpx

from _db import get_db_connection

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Silence httpx logs
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

SPARQL_ENDPOINT = "https://query.wikidata.org/sparql"

# Phase 1b corrected query
SPARQL_QUERY_TEMPLATE = """
SELECT ?team ?teamLabel ?fullName ?shortName
       ?stadium ?stadiumLabel ?capacity ?altitude ?stadiumCoords
       ?adminLocation ?adminLocationLabel
       ?website ?twitter ?instagram
WHERE {{
  VALUES ?team {{ wd:{qid} }}

  OPTIONAL {{
    ?team rdfs:label ?teamLabel .
    FILTER(LANG(?teamLabel) IN ("es", "en"))
  }}

  OPTIONAL {{
    ?team wdt:P1448 ?fullName .
    FILTER(LANG(?fullName) IN ("es", "en", ""))
  }}
  OPTIONAL {{
    ?team wdt:P1813 ?shortName .
    FILTER(LANG(?shortName) IN ("es", "en", ""))
  }}

  OPTIONAL {{
    {{
      SELECT ?stadium (MAX(?cap) AS ?capacity) WHERE {{
        wd:{qid} wdt:P115 ?stadium .
        OPTIONAL {{ ?stadium wdt:P1083 ?cap . }}
      }}
      GROUP BY ?stadium
      ORDER BY DESC(?capacity)
      LIMIT 1
    }}
    OPTIONAL {{
      ?stadium rdfs:label ?stadiumLabel .
      FILTER(LANG(?stadiumLabel) IN ("es", "en"))
    }}
    OPTIONAL {{ ?stadium wdt:P2044 ?altitude . }}
    OPTIONAL {{ ?stadium wdt:P625 ?stadiumCoords . }}
  }}

  OPTIONAL {{
    ?team wdt:P131 ?adminLocation .
    OPTIONAL {{
      ?adminLocation rdfs:label ?adminLocationLabel .
      FILTER(LANG(?adminLocationLabel) IN ("es", "en"))
    }}
  }}

  OPTIONAL {{ ?team wdt:P856 ?website . }}
  OPTIONAL {{ ?team wdt:P2002 ?twitter . }}
  OPTIONAL {{ ?team wdt:P2003 ?instagram . }}
}}
LIMIT 1
"""


def get_ecuador_teams(conn):
    """Get Ecuador teams with wikidata_id."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT t.id, t.name, t.wikidata_id
            FROM teams t
            WHERE t.country = 'Ecuador'
              AND t.wikidata_id IS NOT NULL
            ORDER BY t.name
        """)
        return cur.fetchall()


def fetch_wikidata(qid: str) -> dict:
    """Fetch data from Wikidata SPARQL."""
    query = SPARQL_QUERY_TEMPLATE.format(qid=qid)

    response = httpx.get(
        SPARQL_ENDPOINT,
        params={"query": query, "format": "json"},
        headers={"User-Agent": "FutbolStats/1.0"},
        timeout=30.0,
    )

    if response.status_code != 200:
        logger.warning(f"SPARQL error {response.status_code} for {qid}")
        return {}

    data = response.json()
    bindings = data.get("results", {}).get("bindings", [])

    if not bindings:
        return {}

    return parse_binding(bindings[0])


def parse_binding(b: dict) -> dict:
    """Parse SPARQL binding to dict."""
    def get_val(key):
        return b.get(key, {}).get("value")

    def get_qid(key):
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

    # Parse capacity/altitude
    capacity = None
    cap_str = get_val("capacity")
    if cap_str:
        try:
            capacity = int(float(cap_str))
        except:
            pass

    altitude = None
    alt_str = get_val("altitude")
    if alt_str:
        try:
            altitude = int(float(alt_str))
        except:
            pass

    # Social handles
    twitter = get_val("twitter")
    if twitter and "/" in twitter:
        twitter = twitter.rstrip("/").split("/")[-1]
    instagram = get_val("instagram")
    if instagram and "/" in instagram:
        instagram = instagram.rstrip("/").split("/")[-1]

    return {
        "team_label": get_val("teamLabel"),
        "full_name": get_val("fullName"),
        "short_name": get_val("shortName"),
        "stadium_qid": get_qid("stadium"),
        "stadium_name": get_val("stadiumLabel"),
        "stadium_capacity": capacity,
        "stadium_altitude_m": altitude,
        "lat": lat,
        "lon": lon,
        "admin_location_label": get_val("adminLocationLabel"),
        "website": get_val("website"),
        "twitter": twitter,
        "instagram": instagram,
    }


def update_enrichment(conn, team_id: int, wikidata_id: str, data: dict):
    """Upsert enrichment data."""
    social = json.dumps({"twitter": data.get("twitter"), "instagram": data.get("instagram")})

    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO team_wikidata_enrichment (
                team_id, wikidata_id, fetched_at,
                stadium_name, stadium_wikidata_id, stadium_capacity, stadium_altitude_m,
                lat, lon, admin_location_label,
                full_name, short_name, social_handles, website,
                enrichment_source
            ) VALUES (
                %s, %s, NOW(),
                %s, %s, %s, %s,
                %s, %s, %s,
                %s, %s, %s::jsonb, %s,
                'wikidata'
            )
            ON CONFLICT (team_id) DO UPDATE SET
                fetched_at = NOW(),
                stadium_name = COALESCE(EXCLUDED.stadium_name, team_wikidata_enrichment.stadium_name),
                stadium_wikidata_id = COALESCE(EXCLUDED.stadium_wikidata_id, team_wikidata_enrichment.stadium_wikidata_id),
                stadium_capacity = COALESCE(EXCLUDED.stadium_capacity, team_wikidata_enrichment.stadium_capacity),
                stadium_altitude_m = COALESCE(EXCLUDED.stadium_altitude_m, team_wikidata_enrichment.stadium_altitude_m),
                lat = COALESCE(EXCLUDED.lat, team_wikidata_enrichment.lat),
                lon = COALESCE(EXCLUDED.lon, team_wikidata_enrichment.lon),
                admin_location_label = COALESCE(EXCLUDED.admin_location_label, team_wikidata_enrichment.admin_location_label),
                full_name = COALESCE(EXCLUDED.full_name, team_wikidata_enrichment.full_name),
                short_name = COALESCE(EXCLUDED.short_name, team_wikidata_enrichment.short_name),
                social_handles = COALESCE(EXCLUDED.social_handles, team_wikidata_enrichment.social_handles),
                website = COALESCE(EXCLUDED.website, team_wikidata_enrichment.website)
        """, (
            team_id, wikidata_id,
            data.get("stadium_name"), data.get("stadium_qid"),
            data.get("stadium_capacity"), data.get("stadium_altitude_m"),
            data.get("lat"), data.get("lon"), data.get("admin_location_label"),
            data.get("full_name"), data.get("short_name"), social, data.get("website"),
        ))


def run_backfill(apply: bool = False):
    logger.info("=" * 60)
    logger.info("Ecuador Enrichment Backfill (Phase 1b SPARQL)")
    logger.info(f"Mode: {'APPLY' if apply else 'DRY-RUN'}")
    logger.info("=" * 60)

    conn = get_db_connection()
    teams = get_ecuador_teams(conn)

    logger.info(f"Found {len(teams)} Ecuador teams with wikidata_id")

    metrics = {"ok": 0, "fail": 0, "no_data": 0}

    for team_id, name, wikidata_id in teams:
        time.sleep(0.5)  # Rate limit

        data = fetch_wikidata(wikidata_id)

        if not data:
            metrics["no_data"] += 1
            logger.warning(f"  NO DATA: {name} ({wikidata_id})")
            continue

        # Log what we found
        stadium = data.get("stadium_name") or "-"
        full_name = data.get("full_name") or "-"
        short_name = data.get("short_name") or "-"
        city = data.get("admin_location_label") or "-"
        alt = data.get("stadium_altitude_m")
        alt_str = f"{alt}m" if alt else "-"

        if apply:
            try:
                update_enrichment(conn, team_id, wikidata_id, data)
                metrics["ok"] += 1
                logger.info(f"  OK: {name}")
                logger.info(f"      stadium={stadium}, alt={alt_str}")
                logger.info(f"      full_name={full_name}")
                logger.info(f"      short_name={short_name}, city={city}")
            except Exception as e:
                metrics["fail"] += 1
                logger.error(f"  FAIL: {name} - {e}")
        else:
            logger.info(f"  [DRY] {name}")
            logger.info(f"      stadium={stadium}, alt={alt_str}")
            logger.info(f"      full_name={full_name}")
            logger.info(f"      short_name={short_name}, city={city}")

    if apply:
        conn.commit()
        logger.info("Changes committed")

    conn.close()

    logger.info("=" * 60)
    logger.info(f"OK: {metrics['ok']}, FAIL: {metrics['fail']}, NO_DATA: {metrics['no_data']}")
    logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    run_backfill(apply=args.apply)


if __name__ == "__main__":
    main()
