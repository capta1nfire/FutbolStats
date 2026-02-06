#!/usr/bin/env python3
"""
Backfill stadium_altitude_m using Open-Meteo Elevation API.

ABE Approval: 2026-02-05
Guardrails:
- Only fill where lat/lon IS NOT NULL and stadium_altitude_m IS NULL
- Uses batch API (100 coords/request) for efficiency
- Dry-run mode by default
- Logging with conteos de candidatos/ok/fails

Usage:
    # Dry-run (default) - show what would be updated
    python3 scripts/backfill_stadium_altitude.py

    # Apply changes
    python3 scripts/backfill_stadium_altitude.py --apply

    # Limit batch size for testing
    python3 scripts/backfill_stadium_altitude.py --apply --limit 50
"""

import argparse
import logging
import sys
from typing import Optional
import requests

from _db import get_db_connection

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Open-Meteo Elevation API
ELEVATION_URL = "https://api.open-meteo.com/v1/elevation"

# Open-Meteo supports up to 100 coords per request
BATCH_SIZE = 100


def get_candidates(
    conn,
    limit: Optional[int] = None,
    force_overwrite: bool = False,
    country: Optional[str] = None,
) -> list:
    """
    Get teams with coords but no altitude.

    ABE P0: Only where lat/lon IS NOT NULL and stadium_altitude_m IS NULL.
    With --force-overwrite: include teams that already have altitude values.
    With --country: filter by specific country.
    """
    conditions = ["twe.lat IS NOT NULL", "twe.lon IS NOT NULL"]
    params = []

    if not force_overwrite:
        conditions.append("twe.stadium_altitude_m IS NULL")

    if country:
        conditions.append("t.country = %s")
        params.append(country)

    params.append(limit or 10000)

    query = f"""
        SELECT
            twe.team_id,
            t.name as team_name,
            t.country,
            twe.lat,
            twe.lon,
            twe.stadium_name
        FROM team_wikidata_enrichment twe
        JOIN teams t ON twe.team_id = t.id
        WHERE {' AND '.join(conditions)}
        ORDER BY t.country, t.name
        LIMIT %s
    """

    with conn.cursor() as cur:
        cur.execute(query, params)
        rows = cur.fetchall()

    return [
        {
            "team_id": row[0],
            "team_name": row[1],
            "country": row[2],
            "lat": row[3],
            "lon": row[4],
            "stadium_name": row[5],
        }
        for row in rows
    ]


def fetch_elevations_batch(coordinates: list) -> list:
    """
    Fetch elevations from Open-Meteo API (batch).

    Args:
        coordinates: List of (lat, lon) tuples.

    Returns:
        List of elevations (int or None).
    """
    if not coordinates:
        return []

    # Build comma-separated lat/lon strings
    lats = ",".join(str(lat) for lat, _ in coordinates)
    lons = ",".join(str(lon) for _, lon in coordinates)

    try:
        response = requests.get(
            ELEVATION_URL,
            params={"latitude": lats, "longitude": lons},
            timeout=30,
        )

        if response.status_code != 200:
            logger.warning(f"Open-Meteo API error: {response.status_code}")
            return [None] * len(coordinates)

        data = response.json()
        elevations_raw = data.get("elevation", [])

        # Parse elevations, handling None/NaN
        result = []
        for elev in elevations_raw:
            if elev is None or (isinstance(elev, float) and elev != elev):  # NaN check
                result.append(None)
            else:
                try:
                    result.append(int(round(elev)))
                except (ValueError, TypeError):
                    result.append(None)

        return result

    except requests.RequestException as e:
        logger.error(f"Elevation request failed: {e}")
        return [None] * len(coordinates)


def update_altitude(conn, team_id: int, altitude: int, force_overwrite: bool = False) -> bool:
    """Update stadium_altitude_m for a team."""
    try:
        with conn.cursor() as cur:
            if force_overwrite:
                cur.execute(
                    """
                    UPDATE team_wikidata_enrichment
                    SET stadium_altitude_m = %s
                    WHERE team_id = %s
                    """,
                    (altitude, team_id),
                )
            else:
                cur.execute(
                    """
                    UPDATE team_wikidata_enrichment
                    SET stadium_altitude_m = %s
                    WHERE team_id = %s
                      AND stadium_altitude_m IS NULL
                    """,
                    (altitude, team_id),
                )
        return True
    except Exception as e:
        logger.error(f"Failed to update team {team_id}: {e}")
        return False


def run_backfill(
    apply: bool = False,
    limit: Optional[int] = None,
    force_overwrite: bool = False,
    country: Optional[str] = None,
):
    """
    Main backfill logic.

    Args:
        apply: If True, apply changes. Otherwise dry-run.
        limit: Max number of teams to process.
        force_overwrite: If True, overwrite existing values.
        country: Filter by specific country.
    """
    logger.info("=" * 60)
    logger.info("Stadium Altitude Backfill via Open-Meteo")
    logger.info(f"Mode: {'APPLY' if apply else 'DRY-RUN'}")
    logger.info(f"Force Overwrite: {force_overwrite}")
    logger.info(f"Country Filter: {country or 'None'}")
    logger.info(f"Limit: {limit or 'None'}")
    logger.info("=" * 60)

    conn = get_db_connection()

    metrics = {
        "candidates": 0,
        "fetched_ok": 0,
        "fetched_fail": 0,
        "updated_ok": 0,
        "updated_fail": 0,
        "skipped_invalid": 0,
    }

    try:
        # Get candidates
        candidates = get_candidates(conn, limit, force_overwrite, country)
        metrics["candidates"] = len(candidates)

        logger.info(f"Found {len(candidates)} candidates")

        if not candidates:
            logger.info("No candidates to process")
            return metrics

        # Process in batches of 100 (Open-Meteo limit)
        for batch_start in range(0, len(candidates), BATCH_SIZE):
            batch = candidates[batch_start:batch_start + BATCH_SIZE]
            batch_num = (batch_start // BATCH_SIZE) + 1
            total_batches = (len(candidates) + BATCH_SIZE - 1) // BATCH_SIZE

            logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} teams)")

            # Extract coordinates
            coords = [(t["lat"], t["lon"]) for t in batch]

            # Fetch elevations (batch API call)
            elevations = fetch_elevations_batch(coords)

            # Process results
            for team, elevation in zip(batch, elevations):
                team_id = team["team_id"]
                team_name = team["team_name"]
                country = team["country"]

                if elevation is None:
                    metrics["fetched_fail"] += 1
                    logger.warning(f"  FAIL: {team_name} ({country}) - no elevation data")
                    continue

                # Sanity check: elevation should be reasonable (-500 to 6000m)
                if not (-500 <= elevation <= 6000):
                    metrics["skipped_invalid"] += 1
                    logger.warning(f"  SKIP: {team_name} ({country}) - invalid elevation {elevation}m")
                    continue

                metrics["fetched_ok"] += 1

                # High altitude indicator for logging
                altitude_tag = ""
                if elevation >= 2500:
                    altitude_tag = " [VERY HIGH]"
                elif elevation >= 1500:
                    altitude_tag = " [HIGH]"

                if apply:
                    success = update_altitude(conn, team_id, elevation, force_overwrite)
                    if success:
                        metrics["updated_ok"] += 1
                        logger.info(f"  OK: {team_name} ({country}) = {elevation}m{altitude_tag}")
                    else:
                        metrics["updated_fail"] += 1
                        logger.error(f"  FAIL: {team_name} ({country}) - update failed")
                else:
                    logger.info(f"  [DRY] {team_name} ({country}) = {elevation}m{altitude_tag}")

            # Commit batch
            if apply:
                conn.commit()
                logger.info(f"Batch {batch_num} committed")

    finally:
        conn.close()

    # Summary
    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Candidates:      {metrics['candidates']}")
    logger.info(f"Fetched OK:      {metrics['fetched_ok']}")
    logger.info(f"Fetched FAIL:    {metrics['fetched_fail']}")
    logger.info(f"Skipped Invalid: {metrics['skipped_invalid']}")
    if apply:
        logger.info(f"Updated OK:      {metrics['updated_ok']}")
        logger.info(f"Updated FAIL:    {metrics['updated_fail']}")
    else:
        logger.info("(DRY-RUN - no updates applied)")
    logger.info("=" * 60)

    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Backfill stadium_altitude_m using Open-Meteo Elevation API"
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply changes (default is dry-run)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of teams to process",
    )
    parser.add_argument(
        "--force-overwrite",
        action="store_true",
        help="Overwrite existing altitude values",
    )
    parser.add_argument(
        "--country",
        type=str,
        default=None,
        help="Filter by country name (e.g., 'Colombia')",
    )

    args = parser.parse_args()

    metrics = run_backfill(
        apply=args.apply,
        limit=args.limit,
        force_overwrite=args.force_overwrite,
        country=args.country,
    )

    # Exit with error if there were failures
    if metrics.get("updated_fail", 0) > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
