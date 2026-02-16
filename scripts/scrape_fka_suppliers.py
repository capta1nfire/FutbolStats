"""
Scrape Football Kit Archive for current kit supplier (manufacturer) per team.

Uses FKA's internal search API (api/search.php) to find team slugs,
then fetches the team kits page and extracts the current supplier from
the #timeline-suppliers section.

Loads existing team alias dictionaries (OddsPortal + FDUK) as fallback
search terms when the DB team name doesn't match in FKA.

Usage:
  source .env
  python3 /tmp/scrape_fka_suppliers.py --test          # 6 test teams
  python3 /tmp/scrape_fka_suppliers.py                  # all active teams
  python3 /tmp/scrape_fka_suppliers.py --output /tmp/kit_suppliers.json
"""
import argparse
import asyncio
import json
import logging
import os
import re
import sys
import time
from urllib.parse import quote_plus

sys.path.insert(0, "/Users/inseqio/FutbolStats")

import aiohttp

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)
log = logging.getLogger(__name__)

BASE = "https://www.footballkitarchive.com"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
    "Accept-Language": "en-US,en;q=0.9",
}

# Rate limit: be more polite to avoid 429
DELAY = 1.5
MAX_RETRIES = 3
BACKOFF_BASE = 30  # seconds

ALIAS_DIR = "/Users/inseqio/FutbolStats/data"


def load_alias_map():
    """Build team_id -> set of alias names from OddsPortal + FDUK dictionaries."""
    alias_map = {}  # team_id -> set(alias_names)

    for fname in ["oddsportal_team_aliases.json", "fduk_team_aliases.json"]:
        path = os.path.join(ALIAS_DIR, fname)
        if not os.path.exists(path):
            continue
        with open(path) as f:
            data = json.load(f)
        for league_key, mappings in data.items():
            if league_key.startswith("_"):
                continue
            if not isinstance(mappings, dict):
                continue
            for alias, team_id in mappings.items():
                if alias.startswith("_") or not isinstance(team_id, int):
                    continue
                alias_map.setdefault(team_id, set()).add(alias)

    log.info(f"Loaded aliases for {len(alias_map)} teams from dictionaries")
    return alias_map


# ---------- FKA search with retry ----------
async def fka_search(session, team_name):
    """Search FKA for a team, return list of {id, name, url}. Retries on 429."""
    url = f"{BASE}/api/search.php?filter={quote_plus(team_name)}"
    for attempt in range(MAX_RETRIES + 1):
        try:
            async with session.get(url, headers=HEADERS, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status == 429:
                    wait = BACKOFF_BASE * (2 ** attempt)
                    log.warning(f"429 rate-limited on '{team_name}', waiting {wait}s (attempt {attempt+1})")
                    await asyncio.sleep(wait)
                    continue
                if resp.status != 200:
                    log.warning(f"HTTP {resp.status} for '{team_name}'")
                    return []
                data = await resp.json(content_type=None)
                results = data.get("data", [])
                return [r for r in results if r.get("type") == "teams"]
        except Exception as e:
            log.warning(f"Search error for '{team_name}': {e}")
            return []
    log.warning(f"Max retries exceeded for '{team_name}'")
    return []


# ---------- Extract supplier from kits page ----------
async def extract_supplier(session, kits_url):
    """Fetch team kits page, extract current kit supplier from timeline. Retries on 429."""
    full_url = f"{BASE}{kits_url}" if kits_url.startswith("/") else kits_url
    for attempt in range(MAX_RETRIES + 1):
        try:
            async with session.get(full_url, headers=HEADERS, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                if resp.status == 429:
                    wait = BACKOFF_BASE * (2 ** attempt)
                    log.warning(f"429 rate-limited on '{kits_url}', waiting {wait}s (attempt {attempt+1})")
                    await asyncio.sleep(wait)
                    continue
                if resp.status != 200:
                    return None, None
                html = await resp.text()

                supplier_section = re.search(
                    r"id=['\"]timeline-suppliers['\"].*?</ol>", html, re.DOTALL
                )
                if not supplier_section:
                    return None, None

                section = supplier_section.group()
                match = re.search(
                    r"timeline-item-brand['\"][^>]*title=['\"]([^'\"]+)['\"].*?"
                    r"timeline-item-years['\"]>([^<]+)<",
                    section,
                    re.DOTALL,
                )
                if match:
                    return match.group(1).strip(), match.group(2).strip()
                return None, None
        except Exception as e:
            log.warning(f"Extract error for '{full_url}': {e}")
            return None, None
    log.warning(f"Max retries exceeded for page '{kits_url}'")
    return None, None


# ---------- Fuzzy match team name ----------
def normalize(name):
    """Normalize team name for comparison."""
    n = name.lower().strip()
    # Remove common prefixes/suffixes
    for prefix in ["fc ", "cf ", "ca ", "cd ", "sc ", "ac ", "ss ", "us ", "as ", "rc "]:
        if n.startswith(prefix):
            n = n[len(prefix):]
    for suffix in [" fc", " cf", " sc", " ac"]:
        if n.endswith(suffix):
            n = n[: -len(suffix)]
    # Remove accents (basic)
    replacements = {
        "á": "a", "é": "e", "í": "i", "ó": "o", "ú": "u",
        "ñ": "n", "ü": "u", "ç": "c", "ã": "a", "õ": "o",
        "ä": "a", "ö": "o", "ë": "e",
    }
    for k, v in replacements.items():
        n = n.replace(k, v)
    return n.strip()


def best_match(db_name, fka_results):
    """Pick the best FKA result for a DB team name."""
    if not fka_results:
        return None

    db_norm = normalize(db_name)

    # Exact match first
    for r in fka_results:
        if normalize(r["name"]) == db_norm:
            return r

    # Contains match
    for r in fka_results:
        fka_norm = normalize(r["name"])
        if db_norm in fka_norm or fka_norm in db_norm:
            return r

    # First word match (e.g., "Flamengo" matches "Flamengo RJ")
    db_first = db_norm.split()[0] if db_norm else ""
    for r in fka_results:
        fka_first = normalize(r["name"]).split()[0] if r.get("name") else ""
        if db_first and db_first == fka_first and len(db_first) > 3:
            return r

    # Fall back to first result if only one
    if len(fka_results) == 1:
        return fka_results[0]

    return None


# ---------- Main ----------
async def main(test_mode=False, output_path="/tmp/kit_suppliers.json"):
    from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy import text

    db_url = os.environ.get("DATABASE_URL_ASYNC")
    if not db_url:
        sync_url = os.environ.get("DATABASE_URL", "")
        db_url = sync_url.replace("postgresql://", "postgresql+asyncpg://")

    engine = create_async_engine(db_url, pool_size=3)
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    # Get teams
    async with async_session() as session:
        if test_mode:
            result = await session.execute(text("""
                SELECT DISTINCT t.id, t.external_id, t.name, t.country
                FROM teams t
                WHERE t.name IN ('River Plate', 'Boca Juniors', 'Barcelona',
                                 'Manchester United', 'Flamengo', 'Atletico Nacional',
                                 'Bayern Munich', 'Al Hilal', 'Colo-Colo')
                ORDER BY t.name
            """))
        else:
            result = await session.execute(text("""
                SELECT DISTINCT t.id, t.external_id, t.name, t.country
                FROM teams t
                WHERE t.id IN (
                    SELECT DISTINCT home_team_id FROM matches
                    WHERE league_id IN (SELECT league_id FROM admin_leagues WHERE is_active = true)
                    AND date >= '2024-01-01'
                    UNION
                    SELECT DISTINCT away_team_id FROM matches
                    WHERE league_id IN (SELECT league_id FROM admin_leagues WHERE is_active = true)
                    AND date >= '2024-01-01'
                )
                ORDER BY t.country, t.name
            """))
        teams = [dict(r._mapping) for r in result.fetchall()]

    await engine.dispose()

    # Load alias dictionaries for fallback searches
    alias_map = load_alias_map()

    # Resume: load existing results and skip already-processed teams
    existing_results = []
    processed_ids = set()
    if os.path.exists(output_path):
        with open(output_path) as f:
            existing_results = json.load(f)
        # Only keep entries that have supplier data (re-try the rest)
        existing_results = [r for r in existing_results if r.get("kit_supplier")]
        processed_ids = {r["team_id"] for r in existing_results}
        log.info(f"Resuming: {len(processed_ids)} teams with supplier data already in {output_path}")

    log.info(f"Processing {len(teams)} teams (test={test_mode}), skipping {len(processed_ids)} already done")

    results = list(existing_results)
    matched = len(existing_results)
    not_found = 0
    no_supplier = 0
    alias_saves = 0

    async with aiohttp.ClientSession() as http:
        for i, team in enumerate(teams):
            team_name = team["name"]
            team_id = team["id"]

            # Skip if already has supplier data from previous run
            if team_id in processed_ids:
                continue

            # Search FKA with DB name
            search_results = await fka_search(http, team_name)
            await asyncio.sleep(DELAY)

            match = best_match(team_name, search_results)
            if not match:
                # Try with country qualifier
                if team.get("country"):
                    search_results2 = await fka_search(http, f"{team_name} {team['country']}")
                    await asyncio.sleep(DELAY)
                    match = best_match(team_name, search_results2)

            # Fallback: try aliases from OddsPortal/FDUK dictionaries
            if not match and team_id in alias_map:
                aliases = alias_map[team_id] - {team_name}
                for alias in sorted(aliases, key=len, reverse=True)[:3]:
                    search_results3 = await fka_search(http, alias)
                    await asyncio.sleep(DELAY)
                    match = best_match(alias, search_results3)
                    if match:
                        alias_saves += 1
                        log.info(f"  ALIAS SAVE: '{team_name}' found via alias '{alias}'")
                        break

            if not match:
                log.warning(f"[{i+1}/{len(teams)}] NOT FOUND: {team_name} ({team['country']})")
                results.append({
                    "team_id": team["id"],
                    "external_id": team["external_id"],
                    "team_name": team_name,
                    "country": team.get("country"),
                    "fka_match": None,
                    "kit_supplier": None,
                    "supplier_since": None,
                })
                not_found += 1
                continue

            # Extract supplier from kits page
            supplier, period = await extract_supplier(http, match["url"])
            await asyncio.sleep(DELAY)

            if supplier:
                matched += 1
                log.info(f"[{i+1}/{len(teams)}] {team_name} → {match['name']} → {supplier} ({period})")
            else:
                no_supplier += 1
                log.warning(f"[{i+1}/{len(teams)}] {team_name} → {match['name']} → NO SUPPLIER DATA")

            results.append({
                "team_id": team["id"],
                "external_id": team["external_id"],
                "team_name": team_name,
                "country": team.get("country"),
                "fka_match": match["name"],
                "fka_id": match.get("id"),
                "fka_url": match.get("url"),
                "kit_supplier": supplier,
                "supplier_since": period,
            })

            # Progress every 50
            if (i + 1) % 50 == 0:
                log.info(f"Progress: {i+1}/{len(teams)} | matched={matched} not_found={not_found} no_supplier={no_supplier}")

    # Save results
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Summary
    print(f"\n{'='*60}")
    print(f"FKA KIT SUPPLIER SCRAPE COMPLETE")
    print(f"  Teams processed: {len(teams)}")
    print(f"  Matched with supplier: {matched}")
    print(f"  Rescued via alias fallback: {alias_saves}")
    print(f"  Not found in FKA: {not_found}")
    print(f"  Found but no supplier data: {no_supplier}")
    print(f"  Output: {output_path}")

    # Show supplier distribution
    suppliers = {}
    for r in results:
        s = r.get("kit_supplier")
        if s:
            suppliers[s] = suppliers.get(s, 0) + 1
    if suppliers:
        print(f"\n  Top suppliers:")
        for s, count in sorted(suppliers.items(), key=lambda x: -x[1])[:15]:
            print(f"    {s:20s}: {count}")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Test with 6 teams")
    parser.add_argument("--output", default="/tmp/kit_suppliers.json")
    args = parser.parse_args()
    asyncio.run(main(test_mode=args.test, output_path=args.output))
