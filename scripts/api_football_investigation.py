#!/usr/bin/env python3
"""
API-Football Payload Investigation Script.

Fetches raw responses from API-Football endpoints for analysis.
Does NOT modify any production code or database.

Usage:
    python scripts/api_football_investigation.py
"""

import asyncio
import json
import os
from datetime import datetime
from pathlib import Path

import httpx

# Load from environment (same as production)
RAPIDAPI_KEY = os.environ.get("RAPIDAPI_KEY", "")
RAPIDAPI_HOST = os.environ.get("RAPIDAPI_HOST", "api-football-v1.p.rapidapi.com")
BASE_URL = f"https://{RAPIDAPI_HOST}"

# Output directory
OUTPUT_DIR = Path("logs/api_football_samples")

# Sample fixtures to investigate (external_id from API-Football)
# Mix of: normal FT, AET with penalties, red cards
SAMPLE_FIXTURES = [
    # Need to find actual fixture IDs from recent matches
    # These will be populated from the /fixtures endpoint
]


async def fetch_endpoint(client: httpx.AsyncClient, endpoint: str, params: dict) -> dict:
    """Fetch a single endpoint and return raw response."""
    headers = {
        "X-RapidAPI-Key": RAPIDAPI_KEY,
        "X-RapidAPI-Host": RAPIDAPI_HOST,
    }
    url = f"{BASE_URL}/{endpoint}"

    response = await client.get(url, headers=headers, params=params)
    response.raise_for_status()
    return response.json()


async def find_sample_fixtures(client: httpx.AsyncClient) -> list[dict]:
    """Find recent finished fixtures with varied characteristics."""
    print("\n=== Finding sample fixtures ===")

    # Get recent finished fixtures from top leagues
    # Liga MX (262), Bundesliga (78), La Liga (140), Premier League (39)
    leagues = [262, 78, 140, 39, 143]  # Added Copa del Rey (143)
    season = 2025

    samples = []

    for league_id in leagues:
        try:
            data = await fetch_endpoint(client, "fixtures", {
                "league": league_id,
                "season": season,
                "status": "FT-AET-PEN",
                "last": 5,
            })

            fixtures = data.get("response", [])
            for fix in fixtures:
                fixture_id = fix.get("fixture", {}).get("id")
                status = fix.get("fixture", {}).get("status", {}).get("short")
                home = fix.get("teams", {}).get("home", {}).get("name")
                away = fix.get("teams", {}).get("away", {}).get("name")
                home_goals = fix.get("goals", {}).get("home")
                away_goals = fix.get("goals", {}).get("away")

                print(f"  {fixture_id}: {home} vs {away} ({home_goals}-{away_goals}) [{status}]")

                samples.append({
                    "fixture_id": fixture_id,
                    "status": status,
                    "home": home,
                    "away": away,
                    "score": f"{home_goals}-{away_goals}",
                    "league_id": league_id,
                })

        except Exception as e:
            print(f"  Error fetching league {league_id}: {e}")

    return samples


async def fetch_all_endpoints(client: httpx.AsyncClient, fixture_id: int, label: str) -> dict:
    """Fetch all relevant endpoints for a single fixture."""
    print(f"\n=== Fetching fixture {fixture_id} ({label}) ===")

    results = {
        "fixture_id": fixture_id,
        "label": label,
        "fetched_at": datetime.utcnow().isoformat() + "Z",
        "endpoints": {},
    }

    endpoints = [
        ("fixtures", {"id": fixture_id}),
        ("fixtures/statistics", {"fixture": fixture_id}),
        ("fixtures/events", {"fixture": fixture_id}),
        ("fixtures/lineups", {"fixture": fixture_id}),
        ("fixtures/players", {"fixture": fixture_id}),
    ]

    for endpoint, params in endpoints:
        try:
            data = await fetch_endpoint(client, endpoint, params)
            results["endpoints"][endpoint] = {
                "status": "ok",
                "response": data.get("response", []),
                "results": data.get("results", 0),
                "paging": data.get("paging", {}),
            }
            print(f"  ✓ {endpoint}: {data.get('results', '?')} results")
        except Exception as e:
            results["endpoints"][endpoint] = {
                "status": "error",
                "error": str(e),
            }
            print(f"  ✗ {endpoint}: {e}")

    return results


def analyze_fields(data: dict) -> dict:
    """Analyze available fields in the response."""
    analysis = {}

    for endpoint, content in data.get("endpoints", {}).items():
        if content.get("status") != "ok":
            analysis[endpoint] = {"status": "error"}
            continue

        response = content.get("response", [])
        if not response:
            analysis[endpoint] = {"status": "empty"}
            continue

        # Analyze first item structure
        first_item = response[0] if isinstance(response, list) else response

        def extract_keys(obj, prefix=""):
            """Recursively extract all keys."""
            keys = []
            if isinstance(obj, dict):
                for k, v in obj.items():
                    full_key = f"{prefix}.{k}" if prefix else k
                    keys.append(full_key)
                    keys.extend(extract_keys(v, full_key))
            elif isinstance(obj, list) and obj:
                keys.extend(extract_keys(obj[0], f"{prefix}[]"))
            return keys

        all_keys = extract_keys(first_item)

        analysis[endpoint] = {
            "status": "ok",
            "item_count": len(response) if isinstance(response, list) else 1,
            "top_level_keys": list(first_item.keys()) if isinstance(first_item, dict) else [],
            "all_keys_count": len(all_keys),
            "sample_keys": all_keys[:50],  # First 50 keys
        }

    return analysis


async def main():
    if not RAPIDAPI_KEY:
        print("ERROR: RAPIDAPI_KEY not set in environment")
        print("Export it from Railway or set manually for local testing")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    async with httpx.AsyncClient(timeout=30.0) as client:
        # Step 1: Find sample fixtures
        samples = await find_sample_fixtures(client)

        if not samples:
            print("No samples found!")
            return

        # Step 2: Select diverse samples (max 5)
        # Try to get: 1 AET/PEN, 1 high-scoring, 1 normal
        selected = []

        # Priority: AET/PEN first
        for s in samples:
            if s["status"] in ("AET", "PEN") and len(selected) < 5:
                selected.append(s)

        # Then add FT matches up to 5 total
        for s in samples:
            if s["status"] == "FT" and len(selected) < 5 and s not in selected:
                selected.append(s)

        print(f"\n=== Selected {len(selected)} fixtures for deep analysis ===")
        for s in selected:
            print(f"  - {s['fixture_id']}: {s['home']} vs {s['away']} [{s['status']}]")

        # Step 3: Fetch all endpoints for each selected fixture
        all_results = []
        for s in selected:
            result = await fetch_all_endpoints(
                client,
                s["fixture_id"],
                f"{s['home']} vs {s['away']}"
            )
            all_results.append(result)

            # Save individual fixture data
            output_file = OUTPUT_DIR / f"fixture_{s['fixture_id']}_raw.json"
            with open(output_file, "w") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"  Saved to {output_file}")

        # Step 4: Analyze fields across all fixtures
        print("\n=== Field Analysis ===")
        for result in all_results:
            analysis = analyze_fields(result)
            print(f"\nFixture {result['fixture_id']} ({result['label']}):")
            for endpoint, info in analysis.items():
                if info.get("status") == "ok":
                    print(f"  {endpoint}: {info['item_count']} items, {info['all_keys_count']} keys")
                    print(f"    Top-level: {info['top_level_keys']}")

        # Step 5: Save combined analysis
        combined = {
            "investigation_timestamp": datetime.utcnow().isoformat() + "Z",
            "fixtures_analyzed": len(all_results),
            "results": all_results,
        }

        combined_file = OUTPUT_DIR / "combined_analysis.json"
        with open(combined_file, "w") as f:
            json.dump(combined, f, indent=2, ensure_ascii=False)
        print(f"\nCombined analysis saved to {combined_file}")


if __name__ == "__main__":
    asyncio.run(main())
