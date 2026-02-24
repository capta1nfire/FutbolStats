#!/usr/bin/env python3
"""
Fix incorrect wikidata_ids for Colombian teams and re-enrich.

Identified issues:
- Envigado: Q1346968 (person) → Q332636 (correct football club)
- Union Magdalena: Q1763098 (asteroid) → Q2031920 (correct football club)
- La Equidad/Inter Bogota: Q137324985 is correct but lacks P115 (home venue) in Wikidata
  → Manual enrichment with Estadio Metropolitano de Techo coords (same as Fortaleza FC)

Usage:
    python scripts/fix_wikidata_ids_colombia.py
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

# Corrections to apply
WIKIDATA_ID_CORRECTIONS = {
    4209: {"old": "Q1346968", "new": "Q332636", "name": "Envigado"},
    4197: {"old": "Q1763098", "new": "Q2031920", "name": "Union Magdalena"},
}

# Manual enrichment for La Equidad (no P115 in Wikidata)
# Uses Estadio Metropolitano de Techo coords (same as Fortaleza FC)
MANUAL_ENRICHMENT = {
    4195: {
        "name": "La Equidad / Internacional de Bogota",
        "wikidata_id": "Q137324985",
        "stadium_name": "Estadio Metropolitano de Techo",
        "lat": 4.623545,
        "lon": -74.135611,
        "stadium_capacity": 7300,
        "admin_location_label": "Bogota",
    }
}

SPARQL_QUERY_TEMPLATE = """
SELECT ?team ?teamLabel ?fullName ?shortName
       ?stadium ?stadiumLabel ?capacity ?altitude ?stadiumCoords
       ?adminLocation ?adminLocationLabel
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
  }}

  OPTIONAL {{ ?team wdt:P131 ?adminLocation . }}

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
        "admin_location_label": get_value("adminLocationLabel"),
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
            headers={"User-Agent": "FutbolStats/1.0 (fix script)"},
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
    print("FIX WIKIDATA IDs - Colombian Teams")
    print("=" * 60)

    conn = await asyncpg.connect(DATABASE_URL)

    # Step 1: Correct wikidata_ids in teams table
    print("\n[STEP 1] Correcting wikidata_ids in teams table...")
    for team_id, correction in WIKIDATA_ID_CORRECTIONS.items():
        print(f"  {correction['name']}: {correction['old']} → {correction['new']}")
        await conn.execute(
            """
            UPDATE teams SET wikidata_id = $1 WHERE id = $2
        """,
            correction["new"],
            team_id,
        )
    print("  Done.")

    # Step 2: Delete old enrichment for these teams (will re-fetch)
    print("\n[STEP 2] Deleting old enrichment records...")
    team_ids_to_refresh = list(WIKIDATA_ID_CORRECTIONS.keys()) + list(
        MANUAL_ENRICHMENT.keys()
    )
    for team_id in team_ids_to_refresh:
        await conn.execute(
            """
            DELETE FROM team_wikidata_enrichment WHERE team_id = $1
        """,
            team_id,
        )
    print(f"  Deleted {len(team_ids_to_refresh)} records.")

    # Step 3: Re-fetch from Wikidata for corrected teams
    print("\n[STEP 3] Re-fetching from Wikidata...")
    async with httpx.AsyncClient() as client:
        for team_id, correction in WIKIDATA_ID_CORRECTIONS.items():
            qid = correction["new"]
            name = correction["name"]
            print(f"  Fetching {name} ({qid})...", end=" ")

            await asyncio.sleep(0.3)  # Rate limit
            data = await fetch_wikidata(qid, client)

            if data:
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
                """,
                    team_id,
                    qid,
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
                print(
                    f"OK - Stadium: {data.get('stadium_name')}, Coords: {data.get('lat')}, {data.get('lon')}"
                )
            else:
                print("NO DATA")

    # Step 4: Manual enrichment for La Equidad (no P115 in Wikidata)
    print("\n[STEP 4] Manual enrichment for teams without P115...")
    for team_id, manual in MANUAL_ENRICHMENT.items():
        print(f"  {manual['name']}...", end=" ")
        await conn.execute(
            """
            INSERT INTO team_wikidata_enrichment (
                team_id, wikidata_id, fetched_at, raw_jsonb,
                stadium_name, stadium_capacity,
                lat, lon, admin_location_label
            ) VALUES (
                $1, $2, NOW(), $3::jsonb,
                $4, $5,
                $6, $7, $8
            )
        """,
            team_id,
            manual["wikidata_id"],
            json.dumps({"source": "manual_enrichment", "reason": "P115 missing in Wikidata"}),
            manual["stadium_name"],
            manual["stadium_capacity"],
            manual["lat"],
            manual["lon"],
            manual["admin_location_label"],
        )
        print(
            f"OK - Stadium: {manual['stadium_name']}, Coords: {manual['lat']}, {manual['lon']}"
        )

    await conn.close()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"wikidata_ids corrected: {len(WIKIDATA_ID_CORRECTIONS)}")
    print(f"Re-enriched from Wikidata: {len(WIKIDATA_ID_CORRECTIONS)}")
    print(f"Manual enrichment: {len(MANUAL_ENRICHMENT)}")
    print("\nAll Colombian teams should now have stadium coordinates.")


if __name__ == "__main__":
    asyncio.run(main())
