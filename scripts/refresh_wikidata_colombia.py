#!/usr/bin/env python3
"""
Re-enrich Colombian teams with improved SPARQL query.
Now extracts P131 from STADIUM (more reliable than club P131).

Usage:
    python scripts/refresh_wikidata_colombia.py
"""

import asyncio
import json
import os
import sys
from typing import Optional

import httpx

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATABASE_URL = os.environ.get("DATABASE_URL")
if not DATABASE_URL:
    print("ERROR: DATABASE_URL environment variable is required", file=sys.stderr)
    sys.exit(1)
if "+asyncpg" in DATABASE_URL:
    DATABASE_URL = DATABASE_URL.replace("postgresql+asyncpg://", "postgresql://", 1)

SPARQL_ENDPOINT = "https://query.wikidata.org/sparql"

# Improved query: extracts P131 from STADIUM, not club
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

    lat, lon = None, None
    coords = get_value("stadiumCoords")
    if coords and coords.startswith("Point("):
        try:
            parts = coords[6:-1].split()
            lon, lat = float(parts[0]), float(parts[1])
        except (ValueError, IndexError):
            pass

    twitter = get_value("twitter")
    if twitter and "/" in twitter:
        twitter = twitter.rstrip("/").split("/")[-1]
    instagram = get_value("instagram")
    if instagram and "/" in instagram:
        instagram = instagram.rstrip("/").split("/")[-1]

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
        # Now from STADIUM P131, not club P131
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
            headers={"User-Agent": "FutbolStats/1.0 (refresh script)"},
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
    print("REFRESH WIKIDATA - Colombian Teams (Stadium P131)")
    print("=" * 60)

    conn = await asyncpg.connect(DATABASE_URL)

    # Get Colombian teams that already have enrichment
    teams = await conn.fetch("""
        SELECT t.id, t.name, t.wikidata_id
        FROM teams t
        JOIN team_wikidata_enrichment twe ON t.id = twe.team_id
        WHERE t.country = 'Colombia'
        ORDER BY t.name
    """)

    print(f"\nRefreshing {len(teams)} Colombian teams with stadium P131...\n")

    updated = 0
    with_city = 0

    async with httpx.AsyncClient() as client:
        for team in teams:
            team_id = team["id"]
            team_name = team["name"]
            wikidata_id = team["wikidata_id"]

            print(f"Processing: {team_name}...", end=" ")

            await asyncio.sleep(0.3)  # Rate limit

            data = await fetch_wikidata(wikidata_id, client)

            if data:
                city = data.get("admin_location_label")

                # Update only admin_location_label
                await conn.execute(
                    """
                    UPDATE team_wikidata_enrichment
                    SET admin_location_label = $1,
                        fetched_at = NOW()
                    WHERE team_id = $2
                """,
                    city,
                    team_id,
                )

                updated += 1
                if city:
                    with_city += 1
                    print(f"OK - City: {city}")
                else:
                    print("OK - No city")
            else:
                print("NO DATA")

    await conn.close()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Teams refreshed:  {updated}")
    print(f"With city (P131): {with_city}")
    print(f"Without city:     {updated - with_city}")


if __name__ == "__main__":
    asyncio.run(main())
