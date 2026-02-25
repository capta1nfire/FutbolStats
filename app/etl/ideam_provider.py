"""
IDEAM Socrata Provider — Colombia weather from real stations.

Queries datos.gov.co SODA API (dataset 57sv-p2fu) for near-real-time
readings from IDEAM automatic telemetry stations (~2 min resolution).

Variables: temp (°C), humidity (%), wind (m/s), pressure (hPa), precip (mm).

Guardrails (ABE 2026-02-25):
- Only use when match is ≤3h from kickoff (nowcast, not forecast)
- captured_at = station reading timestamp (not fetch time), must be < t0
- kind='nowcast', source='ideam-socrata'

Usage:
    provider = IdeamSocrataProvider()
    data = await provider.get_station_reading(lat=4.6459, lon=-74.0775)
    await provider.close()
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from math import radians, cos, sin, asin, sqrt
from typing import Optional

import aiohttp

logger = logging.getLogger(__name__)

SOCRATA_BASE = "https://www.datos.gov.co/resource/57sv-p2fu.json"
SOCRATA_TIMEOUT = 10  # seconds — ABE guardrail

# Sensor description patterns → canonical field mapping
SENSOR_MAP = {
    "TEMPERATURA DEL AIRE A 2 m": "temp_c",
    "GPRS - TEMPERATURA DEL AIRE A 2 m": "temp_c",
    "HUMEDAD DEL AIRE A 2 m": "humidity",
    "GPRS - HUMEDAD DEL AIRE A 2 m": "humidity",
    "VELOCIDAD DEL VIENTO": "wind_ms",
    "GPRS - VELOCIDAD DEL VIENTO": "wind_ms",
    "PRESIÓN ATMOSFÉRICA": "pressure_hpa",
    "GPRS - PRESIÓN ATMOSFÉRICA": "pressure_hpa",
    "PRECIPITACIÓN": "precip_mm",
    "GPRS - PRECIPITACIÓN": "precip_mm",
}

# Minimum required fields for a "complete" reading
REQUIRED_FIELDS = {"temp_c", "humidity", "wind_ms", "precip_mm"}


@dataclass
class IdeamReading:
    """Station reading mapped to canonical weather fields."""
    temp_c: Optional[float] = None
    humidity: Optional[float] = None
    wind_ms: Optional[float] = None
    precip_mm: Optional[float] = None
    pressure_hpa: Optional[float] = None
    cloudcover: Optional[float] = None  # IDEAM doesn't provide this
    is_daylight: Optional[bool] = None  # Will be calculated if needed
    captured_at: Optional[datetime] = None
    station_name: Optional[str] = None
    station_dist_km: Optional[float] = None


def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Haversine distance in km."""
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return 2 * 6371 * asin(sqrt(a))


class IdeamSocrataProvider:
    """
    Provider for IDEAM station data via datos.gov.co Socrata SODA API.

    Dataset 57sv-p2fu: multi-variable, ~2min resolution, rolling ~48h window.
    """

    def __init__(self, timeout: int = SOCRATA_TIMEOUT):
        self.timeout = timeout
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            )
        return self._session

    async def get_station_reading(
        self,
        lat: float,
        lon: float,
        radius_km: float = 15.0,
    ) -> Optional[IdeamReading]:
        """
        Get the most recent station reading near a coordinate.

        Queries a bounding box around (lat, lon), finds the nearest station
        with the most complete variable set, and returns the latest readings.

        Args:
            lat: Stadium latitude.
            lon: Stadium longitude.
            radius_km: Search radius (default 15km, covers city-level).

        Returns:
            IdeamReading with station data, or None if unavailable.
        """
        try:
            session = await self._get_session()

            # Bounding box: ~0.15 degrees ≈ 15km at equator
            delta = radius_km / 111.0
            bbox_filter = (
                f"latitud between {lat - delta} and {lat + delta} "
                f"AND longitud between {lon - delta} and {lon + delta}"
            )

            params = {
                "$where": bbox_filter,
                "$order": "fechaobservacion DESC",
                "$limit": 100,  # Enough to get multiple stations × sensors
            }

            async with session.get(SOCRATA_BASE, params=params) as response:
                if response.status != 200:
                    logger.warning(f"[IDEAM] Socrata API error: {response.status}")
                    return None
                rows = await response.json()

            if not rows:
                logger.debug(f"[IDEAM] No stations within {radius_km}km of ({lat}, {lon})")
                return None

            return self._select_best_reading(rows, lat, lon)

        except aiohttp.ClientError as e:
            logger.warning(f"[IDEAM] Socrata request failed: {e}")
            return None
        except Exception as e:
            logger.error(f"[IDEAM] Unexpected error: {e}")
            return None

    def _select_best_reading(
        self,
        rows: list[dict],
        target_lat: float,
        target_lon: float,
    ) -> Optional[IdeamReading]:
        """
        From raw Socrata rows, select the best station and build an IdeamReading.

        Strategy:
        1. Group rows by station
        2. For each station, collect the latest reading per sensor type
        3. Score stations by: completeness (required fields) then distance
        4. Return the best station's reading
        """
        # Group by station code
        stations: dict[str, dict] = {}
        for row in rows:
            code = row.get("codigoestacion", "")
            sensor_desc = row.get("descripcionsensor", "")
            field = SENSOR_MAP.get(sensor_desc)
            if not field:
                continue

            if code not in stations:
                station_lat = float(row.get("latitud", 0))
                station_lon = float(row.get("longitud", 0))
                stations[code] = {
                    "name": row.get("nombreestacion", ""),
                    "lat": station_lat,
                    "lon": station_lon,
                    "dist_km": _haversine(target_lat, target_lon, station_lat, station_lon),
                    "readings": {},
                    "latest_ts": None,
                }

            # Only keep the first (most recent) reading per field per station
            if field not in stations[code]["readings"]:
                value_str = row.get("valorobservado", "")
                try:
                    value = float(value_str)
                except (ValueError, TypeError):
                    continue

                ts_str = row.get("fechaobservacion", "")
                try:
                    ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                    if ts.tzinfo is None:
                        ts = ts.replace(tzinfo=timezone.utc)
                except (ValueError, TypeError):
                    ts = None

                stations[code]["readings"][field] = value
                if ts and (stations[code]["latest_ts"] is None or ts > stations[code]["latest_ts"]):
                    stations[code]["latest_ts"] = ts

        if not stations:
            return None

        # Score: prioritize completeness of required fields, then distance
        def station_score(info: dict) -> tuple:
            fields = set(info["readings"].keys())
            completeness = len(fields & REQUIRED_FIELDS)
            return (-completeness, info["dist_km"])

        best_code = min(stations, key=lambda c: station_score(stations[c]))
        best = stations[best_code]

        readings = best["readings"]
        if not (readings.keys() & REQUIRED_FIELDS):
            # Not even one required field
            return None

        return IdeamReading(
            temp_c=readings.get("temp_c"),
            humidity=readings.get("humidity"),
            wind_ms=readings.get("wind_ms"),
            precip_mm=readings.get("precip_mm"),
            pressure_hpa=readings.get("pressure_hpa"),
            cloudcover=None,  # IDEAM doesn't provide
            captured_at=best["latest_ts"],
            station_name=best["name"],
            station_dist_km=round(best["dist_km"], 1),
        )

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()
        logger.debug("[IDEAM] Provider closed")
