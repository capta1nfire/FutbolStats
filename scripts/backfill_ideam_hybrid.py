#!/usr/bin/env python3
"""
Hybrid backfill: IDEAM precipitation + Open-Meteo Archive (temp/humidity/wind/cloud).

For 1,809 Colombia matches (2019-2023) that only have IDEAM rain data,
fetch Open-Meteo Archive for temp/humidity/wind/cloud and combine with
IDEAM's more precise precipitation readings.

Insert directly into match_weather_canonical with source='open-meteo-archive+ideam'.

Usage:
    python3 scripts/backfill_ideam_hybrid.py              # dry-run
    python3 scripts/backfill_ideam_hybrid.py --apply       # insert into canonical
"""

import argparse
import json
import logging
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta

import requests

sys.path.insert(0, "/Users/inseqio/FutbolStats/scripts")
from _db import get_db_connection

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
HOURLY_VARS = [
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
    "cloud_cover",
    "surface_pressure",
    "is_day",
]

PROXY = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy") or os.environ.get("SOFASCORE_PROXY_URL")


def get_ideam_matches(conn):
    """Get IDEAM-only matches with coords and IDEAM rain data."""
    query = """
        SELECT h.match_id, m.date as kickoff_utc,
               twe.lat, twe.lon,
               (h.weather_data->'ideam'->'h0'->>'rain_mm')::float as ideam_rain_h0,
               (h.weather_data->'ideam'->>'dist_km')::float as station_dist_km,
               th.name as home_team,
               h.created_at
        FROM match_weather_hist h
        JOIN matches m ON m.id = h.match_id
        JOIN teams th ON m.home_team_id = th.id
        JOIN team_wikidata_enrichment twe ON twe.team_id = m.home_team_id
        WHERE h.source = 'ideam'
          AND twe.lat IS NOT NULL
          AND NOT EXISTS (
            SELECT 1 FROM match_weather_canonical c WHERE c.match_id = h.match_id
          )
        ORDER BY m.date
    """
    with conn.cursor() as cur:
        cur.execute(query)
        rows = cur.fetchall()
    return [
        {
            "match_id": r[0],
            "kickoff_utc": r[1],
            "lat": float(r[2]),
            "lon": float(r[3]),
            "ideam_rain_mm": r[4] if r[4] is not None else 0.0,
            "station_dist_km": r[5],
            "home_team": r[6],
            "captured_at": r[7],
        }
        for r in rows
    ]


def group_by_stadium(matches):
    """Group matches by unique (lat, lon) rounded to 4 decimals."""
    groups = defaultdict(list)
    for m in matches:
        key = (round(m["lat"], 4), round(m["lon"], 4))
        groups[key].append(m)
    return groups


def fetch_weather(lat, lon, start_date, end_date):
    """Fetch hourly weather from Open-Meteo Archive API."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join(HOURLY_VARS),
        "timezone": "UTC",
    }
    proxies = {"https": PROXY, "http": PROXY} if PROXY else None

    for attempt in range(3):
        try:
            resp = requests.get(ARCHIVE_URL, params=params, timeout=60, proxies=proxies)
            if resp.status_code == 429:
                wait = 30 * (attempt + 1)
                logger.warning("Rate limited, waiting %ds...", wait)
                time.sleep(wait)
                continue
            if resp.status_code != 200:
                logger.error("API error %d: %s", resp.status_code, resp.text[:200])
                return None

            data = resp.json()
            hourly = data.get("hourly", {})
            times = hourly.get("time", [])

            lookup = {}
            for i, t in enumerate(times):
                row = {}
                for var in HOURLY_VARS:
                    values = hourly.get(var, [])
                    row[var] = values[i] if i < len(values) else None
                lookup[t] = row
            return lookup

        except requests.RequestException as e:
            logger.error("Request failed (attempt %d): %s", attempt + 1, e)
            time.sleep(5)

    return None


def extract_kickoff_hour(weather_lookup, kickoff_utc):
    """Extract weather at kickoff hour (h0)."""
    kickoff_hour = kickoff_utc.replace(minute=0, second=0, microsecond=0)
    key = kickoff_hour.strftime("%Y-%m-%dT%H:%M")
    return weather_lookup.get(key)


def insert_canonical(conn, match_id, om_data, ideam_rain_mm, captured_at):
    """Insert hybrid row into match_weather_canonical."""
    if om_data is None:
        return False

    temp_c = om_data.get("temperature_2m")
    humidity_pct = om_data.get("relative_humidity_2m")
    wind_kmh = om_data.get("wind_speed_10m") or 0
    wind_ms = wind_kmh / 3.6
    cloudcover_pct = om_data.get("cloud_cover")
    pressure_hpa = om_data.get("surface_pressure")
    is_day = om_data.get("is_day")
    is_daylight = True if is_day == 1 else (False if is_day == 0 else None)

    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO match_weather_canonical (
                match_id, temp_c, humidity_pct, wind_ms, precip_mm,
                cloudcover_pct, pressure_hpa, is_daylight, precip_prob,
                kind, source, forecast_horizon_hours, captured_at
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, NULL,
                      'archive', 'open-meteo-archive+ideam', NULL, %s)
            ON CONFLICT (match_id) DO NOTHING
            """,
            (match_id, temp_c, humidity_pct, wind_ms, ideam_rain_mm,
             cloudcover_pct, pressure_hpa, is_daylight, captured_at),
        )
    return True


def main():
    parser = argparse.ArgumentParser(description="Hybrid IDEAM + Open-Meteo backfill")
    parser.add_argument("--apply", action="store_true", help="Actually insert (default: dry-run)")
    args = parser.parse_args()

    conn = get_db_connection()
    matches = get_ideam_matches(conn)
    logger.info("Found %d IDEAM-only matches not in canonical", len(matches))

    if not matches:
        logger.info("Nothing to do")
        conn.close()
        return

    stadium_groups = group_by_stadium(matches)
    logger.info("Unique stadiums: %d", len(stadium_groups))

    if PROXY:
        logger.info("Using proxy: %s", PROXY[:30] + "...")
    else:
        logger.warning("No HTTPS_PROXY set — may hit rate limits")

    inserted = 0
    api_calls = 0
    api_errors = 0
    no_data = 0

    for (lat, lon), group in stadium_groups.items():
        # Date range for this stadium
        dates = [m["kickoff_utc"] for m in group]
        min_date = (min(dates) - timedelta(days=1)).strftime("%Y-%m-%d")
        max_date_raw = max(dates) + timedelta(days=1)
        # Archive API limit: can't exceed yesterday
        yesterday = datetime.utcnow() - timedelta(days=1)
        if max_date_raw > yesterday:
            max_date_raw = yesterday
        max_date = max_date_raw.strftime("%Y-%m-%d")

        sample_team = group[0]["home_team"]
        logger.info(
            "Stadium (%.4f, %.4f) — %s — %d matches (%s to %s)",
            lat, lon, sample_team, len(group), min_date, max_date,
        )

        if not args.apply:
            for m in group:
                logger.info(
                    "  [DRY-RUN] match=%d %s ideam_rain=%.1f dist=%.1fkm",
                    m["match_id"], m["kickoff_utc"].strftime("%Y-%m-%d %H:%M"),
                    m["ideam_rain_mm"], m["station_dist_km"] or 0,
                )
            continue

        weather_lookup = fetch_weather(lat, lon, min_date, max_date)
        api_calls += 1

        if weather_lookup is None:
            api_errors += 1
            logger.error("  Failed to fetch weather for stadium")
            continue

        for m in group:
            h0 = extract_kickoff_hour(weather_lookup, m["kickoff_utc"])
            if h0 is None:
                no_data += 1
                continue

            ok = insert_canonical(
                conn, m["match_id"], h0, m["ideam_rain_mm"], m["captured_at"]
            )
            if ok:
                inserted += 1

        conn.commit()
        time.sleep(0.3)  # Be nice to Open-Meteo

    logger.info("=" * 60)
    logger.info("RESULTS: inserted=%d, api_calls=%d, api_errors=%d, no_data=%d",
                inserted, api_calls, api_errors, no_data)
    conn.close()


if __name__ == "__main__":
    main()
