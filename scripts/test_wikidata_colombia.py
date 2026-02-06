#!/usr/bin/env python3
"""
Test script: Wikidata enrichment for Colombian teams only.
Run manually to validate the pipeline before enabling the scheduler job.

Usage:
    python scripts/test_wikidata_colombia.py
"""

import asyncio
import json
import os
import sys
from datetime import datetime
from typing import Optional

import httpx

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://postgres:heFyxqRYCUMNkVSCgcpXHprpjAPcfJAQ@maglev.proxy.rlwy.net:24997/railway",
)

SPARQL_ENDPOINT = "https://query.wikidata.org/sparql"

SPARQL_QUERY_TEMPLATE = """
SELECT ?team ?teamLabel ?fullName ?shortName
       ?stadium ?stadiumLabel ?capacity ?altitude ?stadiumCoords
       ?stadiumLocation ?stadiumLocationLabel
       ?website ?twitter ?instagram
WHERE {{
  BIND(wd:{qid} AS ?team)

  OPTIONAL {{ ?team wdt:P1448 ?fullName . }}
  OPTIONAL {{ ?team wdt:P1813 ?shortName . }}

  OPTIONAL {{
    ?team wdt:P115 ?stadium .
    OPTIONAL {{ ?stadium wdt:P1083 ?capacity . }}
    OPTIONAL {{ ?stadium wdt:P2044 ?altitude . }}
    OPTIONAL {{ ?stadium wdt:P625 ?stadiumCoords . }}
    # Stadium location (P131) - more reliable than club P131
    OPTIONAL {{ ?stadium wdt:P131 ?stadiumLocation . }}
  }}

  OPTIONAL {{ ?team wdt:P856 ?website . }}
  OPTIONAL {{ ?team wdt:P2002 ?twitter . }}
  OPTIONAL {{ ?team wdt:P2003 ?instagram . }}

  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en,es". }}
}}
LIMIT 1
"""


def parse_wikidata_response(binding: dict, raw: dict) -> dict:
    """Parse SPARQL binding to structured dict."""

    def get_value(key: str):
        return binding.get(key, {}).get("value")

    def get_qid(key: str):
        val = get_value(key)
        if val and "/entity/Q" in val:
            return val.split("/")[-1]
        return None

    # Parse stadium coordinates
    lat, lon = None, None
    coords = get_value("stadiumCoords")
    if coords and coords.startswith("Point("):
        try:
            parts = coords[6:-1].split()
            lon, lat = float(parts[0]), float(parts[1])
        except (ValueError, IndexError):
            pass

    # Social handles cleanup
    twitter = get_value("twitter")
    if twitter and "/" in twitter:
        twitter = twitter.rstrip("/").split("/")[-1]
    instagram = get_value("instagram")
    if instagram and "/" in instagram:
        instagram = instagram.rstrip("/").split("/")[-1]

    # Parse capacity and altitude
    capacity = None
    capacity_str = get_value("capacity")
    if capacity_str:
        try:
            capacity = int(float(capacity_str))
        except (ValueError, TypeError):
            pass

    altitude = None
    altitude_str = get_value("altitude")
    if altitude_str:
        try:
            altitude = int(float(altitude_str))
        except (ValueError, TypeError):
            pass

    return {
        "raw_jsonb": raw,
        "full_name": get_value("fullName"),
        "short_name": get_value("shortName"),
        "stadium_name": get_value("stadiumLabel"),
        "stadium_wikidata_id": get_qid("stadium"),
        "stadium_capacity": capacity,
        "stadium_altitude_m": altitude,
        "lat": lat,
        "lon": lon,
        "admin_location_label": get_value("stadiumLocationLabel"),
        "website": get_value("website"),
        "social_handles": {"twitter": twitter, "instagram": instagram},
    }


async def fetch_wikidata(qid: str, client: httpx.AsyncClient) -> Optional[dict]:
    """Fetch data from Wikidata SPARQL."""
    query = SPARQL_QUERY_TEMPLATE.format(qid=qid)

    try:
        response = await client.get(
            SPARQL_ENDPOINT,
            params={"query": query, "format": "json"},
            headers={"User-Agent": "FutbolStats/1.0 (test script)"},
            timeout=30.0,
        )
        response.raise_for_status()
        data = response.json()

        bindings = data.get("results", {}).get("bindings", [])
        if not bindings:
            return None

        return parse_wikidata_response(bindings[0], raw=data)

    except Exception as e:
        print(f"  ERROR fetching {qid}: {e}")
        return None


async def main():
    import asyncpg

    print("=" * 60)
    print("WIKIDATA ENRICHMENT TEST - Colombian Teams")
    print("=" * 60)

    # Connect to DB
    conn = await asyncpg.connect(DATABASE_URL)

    # Get Colombian teams with wikidata_id
    teams = await conn.fetch("""
        SELECT id, name, wikidata_id
        FROM teams
        WHERE country = 'Colombia'
          AND wikidata_id IS NOT NULL
        ORDER BY name
    """)

    print(f"\nFound {len(teams)} Colombian teams with wikidata_id\n")

    enriched = 0
    errors = 0
    results = []

    async with httpx.AsyncClient() as client:
        for team in teams:
            team_id = team["id"]
            team_name = team["name"]
            wikidata_id = team["wikidata_id"]

            print(f"Processing: {team_name} ({wikidata_id})...", end=" ")

            # Rate limiting
            await asyncio.sleep(0.3)

            data = await fetch_wikidata(wikidata_id, client)

            if data:
                # Insert into DB
                try:
                    await conn.execute(
                        """
                        INSERT INTO team_wikidata_enrichment (
                            team_id, wikidata_id, fetched_at, raw_jsonb,
                            stadium_name, stadium_wikidata_id, stadium_capacity, stadium_altitude_m,
                            lat, lon, admin_location_label,
                            full_name, short_name, social_handles, website
                        ) VALUES (
                            $1, $2, NOW(), $3::jsonb,
                            $4, $5, $6, $7,
                            $8, $9, $10,
                            $11, $12, $13::jsonb, $14
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
                            website = EXCLUDED.website
                    """,
                        team_id,
                        wikidata_id,
                        json.dumps(data.get("raw_jsonb")),
                        data.get("stadium_name"),
                        data.get("stadium_wikidata_id"),
                        data.get("stadium_capacity"),
                        data.get("stadium_altitude_m"),
                        data.get("lat"),
                        data.get("lon"),
                        data.get("admin_location_label"),
                        data.get("full_name"),
                        data.get("short_name"),
                        json.dumps(data.get("social_handles")),
                        data.get("website"),
                    )

                    enriched += 1
                    status = "OK"

                    # Track what we got
                    results.append(
                        {
                            "team": team_name,
                            "stadium": data.get("stadium_name"),
                            "city": data.get("admin_location_label"),
                            "coords": f"{data.get('lat')}, {data.get('lon')}"
                            if data.get("lat")
                            else None,
                            "altitude": data.get("stadium_altitude_m"),
                        }
                    )

                except Exception as e:
                    errors += 1
                    status = f"DB ERROR: {e}"

                print(status)
            else:
                errors += 1
                print("NO DATA")

    await conn.close()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total teams:    {len(teams)}")
    print(f"Enriched:       {enriched}")
    print(f"Errors/No data: {errors}")

    print("\n" + "-" * 60)
    print("ENRICHMENT DETAILS:")
    print("-" * 60)
    for r in results:
        print(f"  {r['team']}")
        print(f"    Stadium: {r['stadium'] or 'N/A'}")
        print(f"    City:    {r['city'] or 'N/A'}")
        print(f"    Coords:  {r['coords'] or 'N/A'}")
        if r["altitude"]:
            print(f"    Altitude: {r['altitude']}m")
        print()


if __name__ == "__main__":
    asyncio.run(main())
