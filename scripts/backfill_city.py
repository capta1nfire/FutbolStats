"""
One-off script: backfill admin_location_label for teams missing city.
Uses the updated SPARQL query (P131 + P159 fallback, en-priority languages).

Usage:
  source .env
  python scripts/backfill_city.py [--batch 50] [--dry-run]
"""
import asyncio
import argparse
import logging
import time
import urllib.request
import urllib.parse
import json
import os

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy import text

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

SPARQL_CITY_QUERY = """
SELECT ?adminLocationLabel WHERE {{
  {{
    SELECT ?adminLocation WHERE {{
      {{ wd:{qid} wdt:P131 ?adminLocation . }}
      UNION
      {{ wd:{qid} wdt:P159 ?adminLocation . }}
    }}
    LIMIT 1
  }}
  OPTIONAL {{ ?adminLocation rdfs:label ?adminLabel_en . FILTER(LANG(?adminLabel_en) = "en") }}
  OPTIONAL {{ ?adminLocation rdfs:label ?adminLabel_es . FILTER(LANG(?adminLabel_es) = "es") }}
  OPTIONAL {{ ?adminLocation rdfs:label ?adminLabel_local .
    FILTER(LANG(?adminLabel_local) IN ("it", "de", "fr", "pt", "nl", "tr"))
  }}
  BIND(COALESCE(?adminLabel_en, ?adminLabel_es, ?adminLabel_local) AS ?adminLocationLabel)
}}
LIMIT 1
"""

WIKIDATA_ENDPOINT = "https://query.wikidata.org/sparql"


from typing import Optional

def fetch_city(qid: str) -> Optional[str]:
    """Fetch city label from Wikidata for a given QID."""
    query = SPARQL_CITY_QUERY.format(qid=qid)
    params = urllib.parse.urlencode({"query": query, "format": "json"})
    req = urllib.request.Request(
        f"{WIKIDATA_ENDPOINT}?{params}",
        headers={"User-Agent": "FutbolStats/1.0 (city-backfill)"},
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())
            bindings = data.get("results", {}).get("bindings", [])
            if bindings and "adminLocationLabel" in bindings[0]:
                return bindings[0]["adminLocationLabel"]["value"]
    except Exception as e:
        log.warning(f"  SPARQL error for {qid}: {e}")
    return None


async def main(batch_size: int, dry_run: bool):
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        log.error("DATABASE_URL not set. Run: source .env")
        return

    # Convert postgres:// to postgresql+asyncpg://
    if db_url.startswith("postgres://"):
        db_url = db_url.replace("postgres://", "postgresql+asyncpg://", 1)
    elif db_url.startswith("postgresql://"):
        db_url = db_url.replace("postgresql://", "postgresql+asyncpg://", 1)

    engine = create_async_engine(db_url, echo=False)
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async with async_session() as session:
        result = await session.execute(text("""
            SELECT twe.team_id, twe.wikidata_id, t.name
            FROM team_wikidata_enrichment twe
            JOIN teams t ON t.id = twe.team_id
            WHERE (twe.admin_location_label IS NULL OR twe.admin_location_label = '')
              AND twe.wikidata_id IS NOT NULL
              AND t.team_type = 'club'
            ORDER BY twe.team_id
            LIMIT :batch_size
        """), {"batch_size": batch_size})
        teams = result.fetchall()

    log.info(f"Found {len(teams)} teams missing city (batch={batch_size}, dry_run={dry_run})")

    updated = 0
    skipped = 0
    errors = 0

    for team_id, wikidata_id, name in teams:
        city = fetch_city(wikidata_id)
        if city:
            log.info(f"  {name:30s} ({wikidata_id}) -> {city}")
            if not dry_run:
                async with async_session() as session:
                    await session.execute(
                        text("""
                            UPDATE team_wikidata_enrichment
                            SET admin_location_label = :city
                            WHERE team_id = :team_id
                        """),
                        {"city": city, "team_id": team_id},
                    )
                    await session.commit()
            updated += 1
        else:
            log.info(f"  {name:30s} ({wikidata_id}) -> NULL (skipped)")
            skipped += 1
        time.sleep(0.25)  # Rate limit: 4 req/sec

    log.info(f"Done: updated={updated}, skipped={skipped}, errors={errors}")
    await engine.dispose()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=50, help="Batch size")
    parser.add_argument("--dry-run", action="store_true", help="Don't write to DB")
    args = parser.parse_args()
    asyncio.run(main(args.batch, args.dry_run))
