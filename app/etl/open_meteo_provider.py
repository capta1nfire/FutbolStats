"""
Open-Meteo Provider for weather forecast and geocoding data.

Fetches weather forecasts and geocoding from Open-Meteo API (free, no API key required).
Used for weather features and venue geo resolution.

Usage:
    provider = OpenMeteoProvider()

    # Weather forecast
    data = await provider.get_forecast(lat=40.4168, lon=-3.7038, kickoff_utc=dt, horizon_hours=24)

    # Geocoding
    geo = await provider.geocode_city("Madrid", "Spain")

API: https://open-meteo.com/en/docs
Geocoding API: https://open-meteo.com/en/docs/geocoding-api
Reference: docs/ARCHITECTURE_SOTA.md section 1.3 (match_weather)
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

import aiohttp

logger = logging.getLogger(__name__)


@dataclass
class GeocodingResult:
    """
    Geocoding result for a city.

    Column names match venue_geo table schema.
    """
    venue_city: str
    country: str
    lat: float
    lon: float
    source: str = "open-meteo-geocoding"
    confidence: float = 0.9


@dataclass
class WeatherForecastData:
    """
    Weather forecast data for a match.

    Column names match EXACTLY the match_weather table schema.
    """
    temp_c: float
    humidity: float
    wind_ms: float
    precip_mm: float
    pressure_hpa: Optional[float] = None
    cloudcover: Optional[float] = None
    is_daylight: bool = False
    forecast_horizon_hours: int = 24
    captured_at: Optional[datetime] = None


class OpenMeteoProvider:
    """
    Provider for Open-Meteo weather forecast data.

    Open-Meteo is free and doesn't require an API key.
    Rate limits are generous (~10,000 requests/day).

    Forecast variables requested:
    - temperature_2m (°C)
    - relative_humidity_2m (%)
    - wind_speed_10m (m/s - converted from km/h)
    - precipitation (mm)
    - surface_pressure (hPa)
    - cloud_cover (%)
    - sunrise/sunset (for is_daylight calculation)
    """

    BASE_URL = "https://api.open-meteo.com/v1/forecast"

    def __init__(self, use_mock: bool = False, timeout: int = 30):
        """
        Initialize the Open-Meteo provider.

        Args:
            use_mock: If True, return mock data without network calls.
            timeout: HTTP timeout in seconds.
        """
        self.use_mock = use_mock
        self.timeout = timeout
        self._session: Optional[aiohttp.ClientSession] = None
        logger.info(f"OpenMeteoProvider initialized (mock={use_mock})")

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            )
        return self._session

    async def get_forecast(
        self,
        lat: float,
        lon: float,
        kickoff_utc: datetime,
        horizon_hours: int = 24,
    ) -> Optional[WeatherForecastData]:
        """
        Get weather forecast for a location at a specific time.

        Args:
            lat: Latitude of the venue.
            lon: Longitude of the venue.
            kickoff_utc: Match kickoff time in UTC.
            horizon_hours: Forecast horizon (default 24h before kickoff).

        Returns:
            WeatherForecastData or None if unavailable.
        """
        if self.use_mock:
            return self._get_mock_data(kickoff_utc, horizon_hours)

        try:
            session = await self._get_session()

            # Request forecast for the kickoff date
            # Open-Meteo uses ISO date format
            date_str = kickoff_utc.strftime("%Y-%m-%d")

            params = {
                "latitude": lat,
                "longitude": lon,
                "hourly": "temperature_2m,relative_humidity_2m,precipitation,surface_pressure,cloud_cover,wind_speed_10m",
                "daily": "sunrise,sunset",
                "timezone": "UTC",
                "start_date": date_str,
                "end_date": date_str,
            }

            async with session.get(self.BASE_URL, params=params) as response:
                if response.status != 200:
                    logger.warning(f"Open-Meteo API error: {response.status}")
                    return None

                data = await response.json()
                return self._parse_response(data, kickoff_utc, horizon_hours)

        except aiohttp.ClientError as e:
            logger.error(f"Open-Meteo request failed: {e}")
            return None
        except Exception as e:
            logger.error(f"Open-Meteo unexpected error: {e}")
            return None

    def _parse_response(
        self,
        data: dict,
        kickoff_utc: datetime,
        horizon_hours: int,
    ) -> Optional[WeatherForecastData]:
        """
        Parse Open-Meteo API response.

        Finds the hourly data closest to kickoff time.
        """
        try:
            hourly = data.get("hourly", {})
            daily = data.get("daily", {})

            if not hourly.get("time"):
                logger.warning("Open-Meteo: No hourly data in response")
                return None

            # Find the hour index closest to kickoff
            times = hourly["time"]
            kickoff_hour = kickoff_utc.hour
            hour_idx = min(kickoff_hour, len(times) - 1)

            # Extract values (with safe defaults)
            temp_c = hourly.get("temperature_2m", [None])[hour_idx]
            humidity = hourly.get("relative_humidity_2m", [None])[hour_idx]
            precip_mm = hourly.get("precipitation", [0])[hour_idx] or 0
            pressure_hpa = hourly.get("surface_pressure", [None])[hour_idx]
            cloudcover = hourly.get("cloud_cover", [None])[hour_idx]

            # Wind: Open-Meteo returns km/h, convert to m/s
            wind_kmh = hourly.get("wind_speed_10m", [0])[hour_idx] or 0
            wind_ms = round(wind_kmh / 3.6, 2)

            # Calculate is_daylight from sunrise/sunset
            is_daylight = self._calculate_is_daylight(
                kickoff_utc,
                daily.get("sunrise", [None])[0],
                daily.get("sunset", [None])[0],
            )

            # Validate required fields
            if temp_c is None or humidity is None:
                logger.warning("Open-Meteo: Missing required fields (temp/humidity)")
                return None

            return WeatherForecastData(
                temp_c=temp_c,
                humidity=humidity,
                wind_ms=wind_ms,
                precip_mm=precip_mm,
                pressure_hpa=pressure_hpa,
                cloudcover=cloudcover,
                is_daylight=is_daylight,
                forecast_horizon_hours=horizon_hours,
                captured_at=datetime.utcnow(),
            )

        except (KeyError, IndexError, TypeError) as e:
            logger.error(f"Open-Meteo parse error: {e}")
            return None

    def _calculate_is_daylight(
        self,
        kickoff_utc: datetime,
        sunrise_str: Optional[str],
        sunset_str: Optional[str],
    ) -> bool:
        """
        Calculate if kickoff is during daylight hours.

        Args:
            kickoff_utc: Kickoff time in UTC.
            sunrise_str: Sunrise time string (ISO format).
            sunset_str: Sunset time string (ISO format).

        Returns:
            True if kickoff is between sunrise and sunset.
        """
        if not sunrise_str or not sunset_str:
            # Best-effort: assume daylight if between 6am-8pm local
            # This is a fallback when sunrise/sunset data is unavailable
            hour = kickoff_utc.hour
            return 6 <= hour < 20

        try:
            # Parse ISO timestamps (Open-Meteo format: 2024-01-15T07:30)
            sunrise = datetime.fromisoformat(sunrise_str.replace("Z", "+00:00"))
            sunset = datetime.fromisoformat(sunset_str.replace("Z", "+00:00"))

            # Normalize to same date as kickoff for comparison
            # (sunrise/sunset are in local time from API)
            return sunrise.time() <= kickoff_utc.time() <= sunset.time()

        except (ValueError, AttributeError) as e:
            logger.debug(f"Could not parse sunrise/sunset: {e}")
            hour = kickoff_utc.hour
            return 6 <= hour < 20

    def _get_mock_data(
        self,
        kickoff_utc: datetime,
        horizon_hours: int,
    ) -> WeatherForecastData:
        """
        Return mock data for testing purposes.

        Values are reasonable ranges for European football.
        """
        import random

        return WeatherForecastData(
            temp_c=round(random.uniform(5, 25), 1),
            humidity=round(random.uniform(40, 85), 0),
            wind_ms=round(random.uniform(1, 8), 1),
            precip_mm=round(random.uniform(0, 5), 1),
            pressure_hpa=round(random.uniform(1000, 1030), 0),
            cloudcover=round(random.uniform(0, 100), 0),
            is_daylight=6 <= kickoff_utc.hour < 20,
            forecast_horizon_hours=horizon_hours,
            captured_at=datetime.utcnow(),
        )

    # =========================================================================
    # GEOCODING
    # =========================================================================

    GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"

    # Country name normalization for Open-Meteo API
    COUNTRY_NORMALIZATION = {
        "USA": "United States",
        "UK": "United Kingdom",
        "England": "United Kingdom",
        "Scotland": "United Kingdom",
        "Wales": "United Kingdom",
        "Northern-Ireland": "United Kingdom",
    }

    async def geocode_city(
        self,
        city: str,
        country: Optional[str] = None,
    ) -> Optional[GeocodingResult]:
        """
        Geocode a city to lat/lon coordinates.

        Uses Open-Meteo Geocoding API (free, no key required).

        Args:
            city: City name (e.g., "Madrid", "Manchester").
            country: Country name for disambiguation (e.g., "Spain", "England").

        Returns:
            GeocodingResult with lat/lon, or None if not found.
        """
        if self.use_mock:
            return self._get_mock_geocoding(city, country)

        if not city or not city.strip():
            return None

        try:
            session = await self._get_session()

            # Normalize country name if needed
            normalized_country = self.COUNTRY_NORMALIZATION.get(country, country) if country else None

            # Build search query: "city, country" for better results
            search_query = city.strip()
            if normalized_country:
                search_query = f"{city.strip()}, {normalized_country}"

            params = {
                "name": search_query,
                "count": 5,  # Get top 5 results for matching
                "language": "en",
                "format": "json",
            }

            async with session.get(self.GEOCODING_URL, params=params) as response:
                if response.status != 200:
                    logger.warning(f"Open-Meteo Geocoding error: {response.status} for {search_query}")
                    return None

                data = await response.json()
                return self._parse_geocoding_response(data, city, country)

        except aiohttp.ClientError as e:
            logger.error(f"Geocoding request failed for {city}: {e}")
            return None
        except Exception as e:
            logger.error(f"Geocoding unexpected error for {city}: {e}")
            return None

    def _parse_geocoding_response(
        self,
        data: dict,
        original_city: str,
        original_country: Optional[str],
    ) -> Optional[GeocodingResult]:
        """
        Parse Open-Meteo Geocoding API response.

        Returns the best match considering city and country.
        """
        results = data.get("results", [])

        if not results:
            logger.debug(f"Geocoding: No results for {original_city}, {original_country}")
            return None

        # Normalize for comparison
        city_lower = original_city.lower().strip()
        country_lower = (original_country or "").lower().strip()

        # Normalized country for matching
        normalized_country = self.COUNTRY_NORMALIZATION.get(original_country, original_country)
        normalized_lower = (normalized_country or "").lower().strip()

        best_match = None
        best_confidence = 0.0

        for result in results:
            result_city = (result.get("name") or "").lower()
            result_country = (result.get("country") or "").lower()
            result_admin1 = (result.get("admin1") or "").lower()  # State/region

            # Calculate confidence based on match quality
            confidence = 0.0

            # City match
            if result_city == city_lower:
                confidence += 0.5
            elif city_lower in result_city or result_city in city_lower:
                confidence += 0.3

            # Country match
            if original_country:
                if result_country == country_lower or result_country == normalized_lower:
                    confidence += 0.4
                elif country_lower in result_country or result_country in country_lower:
                    confidence += 0.25
                # UK special case: match "England" to "United Kingdom"
                elif original_country.lower() in ["england", "scotland", "wales"] and result_country == "united kingdom":
                    confidence += 0.35
            else:
                # No country filter, give partial credit
                confidence += 0.1

            # Population bonus (prefer larger cities - more likely to have stadiums)
            population = result.get("population", 0) or 0
            if population > 1_000_000:
                confidence += 0.1
            elif population > 100_000:
                confidence += 0.05

            if confidence > best_confidence:
                best_confidence = confidence
                best_match = result

        if not best_match or best_confidence < 0.3:
            logger.debug(f"Geocoding: Low confidence ({best_confidence:.2f}) for {original_city}, {original_country}")
            return None

        # Cap confidence at 0.95 (never 100% sure without manual verification)
        final_confidence = min(best_confidence, 0.95)

        return GeocodingResult(
            venue_city=original_city,
            country=original_country or best_match.get("country", "Unknown"),
            lat=best_match["latitude"],
            lon=best_match["longitude"],
            source="open-meteo-geocoding",
            confidence=round(final_confidence, 2),
        )

    def _get_mock_geocoding(
        self,
        city: str,
        country: Optional[str],
    ) -> GeocodingResult:
        """
        Return mock geocoding data for testing.

        Uses approximate coordinates for common football cities.
        """
        import random

        # Some known cities for testing
        known_cities = {
            "london": (51.5074, -0.1278),
            "madrid": (40.4168, -3.7038),
            "barcelona": (41.3851, 2.1734),
            "paris": (48.8566, 2.3522),
            "munich": (48.1351, 11.5820),
            "milan": (45.4642, 9.1900),
            "manchester": (53.4808, -2.2426),
            "liverpool": (53.4084, -2.9916),
        }

        city_lower = city.lower()
        if city_lower in known_cities:
            lat, lon = known_cities[city_lower]
        else:
            # Random European coordinates
            lat = round(random.uniform(40, 55), 4)
            lon = round(random.uniform(-5, 15), 4)

        return GeocodingResult(
            venue_city=city,
            country=country or "Unknown",
            lat=lat,
            lon=lon,
            source="open-meteo-geocoding-mock",
            confidence=0.85,
        )

    async def close(self) -> None:
        """Close the aiohttp session."""
        if self._session and not self._session.closed:
            await self._session.close()
        logger.debug("OpenMeteoProvider closed")
