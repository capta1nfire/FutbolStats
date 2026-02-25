"""
Inject real Open-Meteo climate normals into team_home_city_profile.
Replaces mocked data in climate_normals_by_month (JSONB) with 10-year averages.

Usage:
    source .env && python scripts/inject_climate_normals.py
"""

import asyncio
import json
import os

import asyncpg

# Manual overrides: DB key -> JSON key
# Direct aliases (spelling/accent variants)
# + proximity proxies (nearest city with data)
MANUAL_MAP: dict[str, str] = {
    "Orlando, Florida_USA": "Orlando_USA",
    "Seville_Spain": "Sevilla_Spain",
    "Sint- Truiden_Belgium": "Sint-Truiden_Belgium",
    "Belém_Brazil": "Belem_Brazil",
    "León de los Aldamas_Mexico": "Leon_Mexico",
    "Río Cuarto_Argentina": "Cordoba_Argentina",
    # Proximity proxies (nearest major city in JSON)
    "Aylesbury, Buckinghamshire_England": "London_England",
    "Deal, Kent_England": "London_England",
    "Hertford, Hertfordshire_England": "London_England",
    "Fareham, Hampshire_England": "Southampton_England",
    "Steyning, West Sussex_England": "Brighton_England",
    "Cantabria_Spain": "Santander_Spain",
    "Harrison_USA": "Bronx_USA",
    "Chicago_USA": "Bridgeview_USA",
}


async def main():
    # --- Load JSON ---
    json_path = os.path.join(os.path.dirname(__file__), "..", "data", "climate_normals.json")
    with open(json_path) as f:
        climate_data = json.load(f)
    print(f"Loaded {len(climate_data)} city entries from climate_normals.json")

    # --- Connect ---
    db_url = os.environ["DATABASE_URL"]
    # asyncpg needs postgresql:// not postgresql+asyncpg://
    db_url = db_url.replace("postgresql+asyncpg://", "postgresql://")
    conn = await asyncpg.connect(db_url)

    try:
        # --- Fetch all profiles with city + country ---
        rows = await conn.fetch("""
            SELECT p.team_id, p.home_city, t.country
            FROM team_home_city_profile p
            JOIN teams t ON t.id = p.team_id
            WHERE p.home_city != 'N/A'
              AND p.home_city IS NOT NULL
              AND t.country IS NOT NULL
        """)
        print(f"Found {len(rows)} eligible profiles (city != N/A, country not null)")

        updated = 0
        manual_hits = 0
        misses = []

        for row in rows:
            lookup_key = f"{row['home_city']}_{row['country']}"
            entry = climate_data.get(lookup_key)
            if entry is None:
                # Try manual override
                override = MANUAL_MAP.get(lookup_key)
                if override:
                    entry = climate_data.get(override)
                    if entry:
                        manual_hits += 1
            if entry is None:
                misses.append(lookup_key)
                continue

            normals = entry["monthly_normals"]
            await conn.execute(
                """
                UPDATE team_home_city_profile
                SET climate_normals_by_month = $1::jsonb,
                    last_updated_at = NOW()
                WHERE team_id = $2
                """,
                json.dumps(normals),
                row["team_id"],
            )
            updated += 1

        print(f"\nResults: {updated} updated ({manual_hits} via manual map), {len(misses)} misses")
        if misses:
            print(f"Misses ({len(misses)}):")
            for m in sorted(set(misses)):
                print(f"  - {m}")

    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(main())
