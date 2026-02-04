"""
Wikidata Team Enrichment ETL.

Uses SPARQL to extract structured team data from Wikidata.
Guardrails: offline job, rate limiting, provenance, fail-open.

ABE Notes:
- Coords come from STADIUM (P115->P625), not the club
- admin_location_label from P131 is informational only, not normalized
- Timezone resolved downstream via _resolve_timezone_for_result()
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Any, Optional

import httpx
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


SPARQL_QUERY_TEMPLATE = """
SELECT ?team ?teamLabel ?fullName ?shortName
       ?stadium ?stadiumLabel ?capacity ?altitude ?stadiumCoords
       ?adminLocation ?adminLocationLabel
       ?website ?twitter ?instagram
WHERE {{
  BIND(wd:{qid} AS ?team)

  # Official names
  OPTIONAL {{ ?team wdt:P1448 ?fullName . }}
  OPTIONAL {{ ?team wdt:P1813 ?shortName . }}

  # PRIORITY: Stadium (P115) and its properties - coords from STADIUM
  OPTIONAL {{
    ?team wdt:P115 ?stadium .
    OPTIONAL {{ ?stadium wdt:P1083 ?capacity . }}
    OPTIONAL {{ ?stadium wdt:P2044 ?altitude . }}
    OPTIONAL {{ ?stadium wdt:P625 ?stadiumCoords . }}
  }}

  # Admin location (P131) - ONLY as informational label, NOT as truth
  # ABE: P131 is noisy, can be district/neighborhood/region
  OPTIONAL {{ ?team wdt:P131 ?adminLocation . }}

  # Web/Social
  OPTIONAL {{ ?team wdt:P856 ?website . }}
  OPTIONAL {{ ?team wdt:P2002 ?twitter . }}
  OPTIONAL {{ ?team wdt:P2003 ?instagram . }}

  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en,es". }}
}}
LIMIT 1
"""


async def fetch_wikidata_for_team(
    qid: str,
    client: httpx.AsyncClient,
) -> Optional[dict[str, Any]]:
    """
    Fetch structured data from Wikidata for a single team.

    Returns raw response for provenance + parsed fields.
    Fail-open: returns None on error (no crash).
    """
    query = SPARQL_QUERY_TEMPLATE.format(qid=qid)

    try:
        response = await client.get(
            settings.WIKIDATA_SPARQL_ENDPOINT,
            params={"query": query, "format": "json"},
            headers={"User-Agent": "FutbolStats/1.0 (contact@futbolstats.app)"},
            timeout=30.0,
        )
        response.raise_for_status()
        data = response.json()

        bindings = data.get("results", {}).get("bindings", [])
        if not bindings:
            logger.debug(f"[WIKIDATA] No results for {qid}")
            return None

        return _parse_wikidata_response(bindings[0], raw=data)

    except httpx.TimeoutException:
        logger.warning(f"[WIKIDATA] Timeout fetching {qid}")
        return None
    except httpx.HTTPStatusError as e:
        logger.warning(f"[WIKIDATA] HTTP error for {qid}: {e.response.status_code}")
        return None
    except Exception as e:
        logger.warning(f"[WIKIDATA] Fetch failed for {qid}: {e}")
        return None  # Fail-open


def _parse_wikidata_response(binding: dict, raw: dict) -> dict[str, Any]:
    """Parse SPARQL binding to structured dict."""

    def get_value(key: str) -> Optional[str]:
        return binding.get(key, {}).get("value")

    def get_qid(key: str) -> Optional[str]:
        val = get_value(key)
        if val and "/entity/Q" in val:
            return val.split("/")[-1]
        return None

    # Parse stadium coordinates (format: "Point(lon lat)")
    # ABE: Coords come from STADIUM (P115->P625), not the club
    lat, lon = None, None
    coords = get_value("stadiumCoords")
    if coords and coords.startswith("Point("):
        try:
            parts = coords[6:-1].split()
            lon, lat = float(parts[0]), float(parts[1])
        except (ValueError, IndexError):
            pass

    # Social handles cleanup (remove URLs, keep only username)
    twitter = get_value("twitter")
    if twitter and "/" in twitter:
        twitter = twitter.rstrip("/").split("/")[-1]
    instagram = get_value("instagram")
    if instagram and "/" in instagram:
        instagram = instagram.rstrip("/").split("/")[-1]

    # Parse capacity and altitude as integers
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
        "lat": lat,  # From stadium
        "lon": lon,  # From stadium
        # ABE: admin_location_label is informational, not normalized
        "admin_location_label": get_value("adminLocationLabel"),
        "website": get_value("website"),
        "social_handles": {
            "twitter": twitter,
            "instagram": instagram,
        },
    }


async def run_wikidata_enrichment_batch(
    session: AsyncSession,
    batch_size: int = 100,
    mode: str = "catch-up",
) -> dict[str, Any]:
    """
    Run batch enrichment for teams with wikidata_id.

    Modes:
    - catch-up: Process teams without enrichment (priority)
    - refresh: Process teams with enrichment > 30 days old

    736 teams / 100 batch = ~8 days for complete catch-up.
    """
    if mode == "catch-up":
        # Only teams without enrichment
        result = await session.execute(
            text("""
            SELECT t.id, t.wikidata_id
            FROM teams t
            LEFT JOIN team_wikidata_enrichment twe ON t.id = twe.team_id
            WHERE t.wikidata_id IS NOT NULL
              AND twe.team_id IS NULL
            LIMIT :batch_size
        """),
            {"batch_size": batch_size},
        )
    else:
        # Refresh: teams with enrichment > 30 days old
        result = await session.execute(
            text("""
            SELECT t.id, t.wikidata_id
            FROM teams t
            JOIN team_wikidata_enrichment twe ON t.id = twe.team_id
            WHERE twe.fetched_at < (NOW() AT TIME ZONE 'UTC') - INTERVAL '30 days'
            ORDER BY twe.fetched_at ASC
            LIMIT :batch_size
        """),
            {"batch_size": batch_size},
        )

    teams = result.fetchall()

    if not teams:
        return {
            "status": "ok",
            "mode": mode,
            "processed": 0,
            "message": "No teams to enrich",
        }

    enriched = 0
    errors = 0
    skipped = 0

    async with httpx.AsyncClient() as client:
        for team_id, wikidata_id in teams:
            # Rate limiting (conservative: 5 req/sec)
            await asyncio.sleep(settings.WIKIDATA_RATE_LIMIT_DELAY)

            data = await fetch_wikidata_for_team(wikidata_id, client)

            if data:
                await _upsert_enrichment(session, team_id, wikidata_id, data)
                enriched += 1
            else:
                # Fail-open: log but don't crash
                errors += 1

    await session.commit()

    logger.info(
        f"[WIKIDATA] Batch complete: mode={mode}, processed={len(teams)}, "
        f"enriched={enriched}, errors={errors}"
    )

    return {
        "status": "ok",
        "mode": mode,
        "processed": len(teams),
        "enriched": enriched,
        "errors": errors,
        "skipped": skipped,
    }


async def _upsert_enrichment(
    session: AsyncSession,
    team_id: int,
    wikidata_id: str,
    data: dict[str, Any],
) -> None:
    """
    UPSERT enrichment data.

    ABE: Explicit ::jsonb cast to avoid type ambiguity in SQLAlchemy text().
    """
    await session.execute(
        text("""
        INSERT INTO team_wikidata_enrichment (
            team_id, wikidata_id, fetched_at, raw_jsonb,
            stadium_name, stadium_wikidata_id, stadium_capacity, stadium_altitude_m,
            lat, lon, admin_location_label,
            full_name, short_name, social_handles, website
        ) VALUES (
            :team_id, :wikidata_id, (NOW() AT TIME ZONE 'UTC'), :raw_jsonb::jsonb,
            :stadium_name, :stadium_wikidata_id, :stadium_capacity, :stadium_altitude_m,
            :lat, :lon, :admin_location_label,
            :full_name, :short_name, :social_handles::jsonb, :website
        )
        ON CONFLICT (team_id) DO UPDATE SET
            wikidata_id = EXCLUDED.wikidata_id,
            fetched_at = (NOW() AT TIME ZONE 'UTC'),
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
        },
    )


async def get_enrichment_stats(session: AsyncSession) -> dict[str, Any]:
    """Get stats about wikidata enrichment coverage."""
    result = await session.execute(
        text("""
        SELECT
            (SELECT COUNT(*) FROM teams WHERE wikidata_id IS NOT NULL) as total_with_wikidata,
            (SELECT COUNT(*) FROM team_wikidata_enrichment) as enriched,
            (SELECT COUNT(*) FROM team_wikidata_enrichment WHERE lat IS NOT NULL) as with_coords,
            (SELECT COUNT(*) FROM team_wikidata_enrichment WHERE stadium_name IS NOT NULL) as with_stadium,
            (SELECT COUNT(*) FROM team_wikidata_enrichment WHERE admin_location_label IS NOT NULL) as with_admin_label
    """)
    )
    row = result.fetchone()

    total = row.total_with_wikidata or 0
    enriched = row.enriched or 0

    return {
        "total_with_wikidata": total,
        "enriched": enriched,
        "pct_complete": round(100.0 * enriched / total, 1) if total > 0 else 0,
        "with_coords": row.with_coords or 0,
        "with_stadium": row.with_stadium or 0,
        "with_admin_label": row.with_admin_label or 0,
    }
