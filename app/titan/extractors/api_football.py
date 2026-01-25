"""TITAN API-Football Extractor with PIT compliance.

Wraps the existing APIFootballProvider to add:
1. captured_at timestamps for PIT compliance
2. Idempotency key computation
3. Standardized ExtractionResult output
"""

import asyncio
import logging
import time
from datetime import datetime, date
from typing import Optional

import httpx

from app.config import get_settings
from app.titan.extractors.base import (
    TitanExtractor,
    ExtractionResult,
    compute_idempotency_key,
    compute_params_hash,
)

logger = logging.getLogger(__name__)
settings = get_settings()


class TitanAPIFootballExtractor(TitanExtractor):
    """TITAN-compliant API-Football extractor.

    Adds PIT compliance and idempotency to API-Football requests.
    Does NOT replace the existing APIFootballProvider - that continues
    to be used by the scheduler for live updates.

    This extractor is for TITAN batch extractions with full audit trail.
    """

    SOURCE_ID = "api_football"

    def __init__(self):
        super().__init__()

        # Detect if using API-Sports directly or RapidAPI
        host = settings.RAPIDAPI_HOST
        if "api-sports.io" in host:
            self.base_url = f"https://{host}"
            headers = {"x-apisports-key": settings.RAPIDAPI_KEY}
        else:
            self.base_url = f"https://{host}/v3"
            headers = {
                "X-RapidAPI-Key": settings.RAPIDAPI_KEY,
                "X-RapidAPI-Host": host,
            }

        self.client = httpx.AsyncClient(
            headers=headers,
            timeout=30.0,
        )
        self.requests_per_minute = settings.API_REQUESTS_PER_MINUTE

    async def extract(
        self,
        endpoint: str,
        params: dict,
        date_bucket: date,
    ) -> ExtractionResult:
        """Extract data from API-Football with PIT compliance.

        Args:
            endpoint: API endpoint (e.g., 'fixtures', 'odds', 'fixtures/statistics')
            params: Request parameters
            date_bucket: Logical date for partitioning

        Returns:
            ExtractionResult with captured_at and idempotency_key
        """
        job_id = self._generate_job_id()
        url = f"{self.base_url}/{endpoint}"

        # Compute idempotency key BEFORE request
        idempotency_key = compute_idempotency_key(
            self.SOURCE_ID, endpoint, params, date_bucket
        )
        params_hash = compute_params_hash(params)

        # Rate limiting delay
        delay = 60 / self.requests_per_minute

        start_time = time.time()
        try:
            response = await self.client.get(url, params=params)

            # CRITICAL: captured_at is AFTER response received
            captured_at = self._get_captured_at()

            response_time_ms = int((time.time() - start_time) * 1000)

            if response.status_code == 429:
                # Rate limited
                return ExtractionResult(
                    source_id=self.SOURCE_ID,
                    job_id=job_id,
                    url=url,
                    endpoint=endpoint,
                    params_hash=params_hash,
                    date_bucket=date_bucket,
                    response_type="error",
                    response_body=None,
                    http_status=429,
                    response_time_ms=response_time_ms,
                    captured_at=captured_at,
                    idempotency_key=idempotency_key,
                    error_type="rate_limit",
                    error_message="API rate limit exceeded (429)",
                )

            # Try to parse JSON
            try:
                data = response.json()

                # Check for API-level errors
                if data.get("errors"):
                    return ExtractionResult(
                        source_id=self.SOURCE_ID,
                        job_id=job_id,
                        url=url,
                        endpoint=endpoint,
                        params_hash=params_hash,
                        date_bucket=date_bucket,
                        response_type="error",
                        response_body=data,
                        http_status=response.status_code,
                        response_time_ms=response_time_ms,
                        captured_at=captured_at,
                        idempotency_key=idempotency_key,
                        error_type="api_error",
                        error_message=str(data["errors"]),
                    )

                # Success
                return ExtractionResult(
                    source_id=self.SOURCE_ID,
                    job_id=job_id,
                    url=url,
                    endpoint=endpoint,
                    params_hash=params_hash,
                    date_bucket=date_bucket,
                    response_type="json",
                    response_body=data,
                    http_status=response.status_code,
                    response_time_ms=response_time_ms,
                    captured_at=captured_at,
                    idempotency_key=idempotency_key,
                )

            except Exception as e:
                # JSON parse error
                return ExtractionResult(
                    source_id=self.SOURCE_ID,
                    job_id=job_id,
                    url=url,
                    endpoint=endpoint,
                    params_hash=params_hash,
                    date_bucket=date_bucket,
                    response_type="error",
                    response_body=None,
                    http_status=response.status_code,
                    response_time_ms=response_time_ms,
                    captured_at=captured_at,
                    idempotency_key=idempotency_key,
                    error_type="parse_error",
                    error_message=f"JSON parse error: {e}",
                )

        except httpx.TimeoutException as e:
            captured_at = self._get_captured_at()
            response_time_ms = int((time.time() - start_time) * 1000)
            return ExtractionResult(
                source_id=self.SOURCE_ID,
                job_id=job_id,
                url=url,
                endpoint=endpoint,
                params_hash=params_hash,
                date_bucket=date_bucket,
                response_type="error",
                response_body=None,
                http_status=0,
                response_time_ms=response_time_ms,
                captured_at=captured_at,
                idempotency_key=idempotency_key,
                error_type="timeout",
                error_message=str(e),
            )

        except httpx.RequestError as e:
            captured_at = self._get_captured_at()
            response_time_ms = int((time.time() - start_time) * 1000)
            return ExtractionResult(
                source_id=self.SOURCE_ID,
                job_id=job_id,
                url=url,
                endpoint=endpoint,
                params_hash=params_hash,
                date_bucket=date_bucket,
                response_type="error",
                response_body=None,
                http_status=0,
                response_time_ms=response_time_ms,
                captured_at=captured_at,
                idempotency_key=idempotency_key,
                error_type="request_error",
                error_message=str(e),
            )

        finally:
            # Respect rate limit
            await asyncio.sleep(delay)

    async def extract_odds(
        self,
        fixture_id: int,
        date_bucket: date,
    ) -> ExtractionResult:
        """Extract odds for a specific fixture.

        Convenience method for common odds extraction use case.

        Args:
            fixture_id: API-Football fixture ID
            date_bucket: Match date for partitioning

        Returns:
            ExtractionResult with odds data
        """
        return await self.extract(
            endpoint="odds",
            params={"fixture": fixture_id, "bookmaker": 8},  # Bet365
            date_bucket=date_bucket,
        )

    async def extract_fixture(
        self,
        fixture_id: int,
        date_bucket: date,
    ) -> ExtractionResult:
        """Extract fixture details.

        Args:
            fixture_id: API-Football fixture ID
            date_bucket: Match date for partitioning

        Returns:
            ExtractionResult with fixture data
        """
        return await self.extract(
            endpoint="fixtures",
            params={"id": fixture_id},
            date_bucket=date_bucket,
        )

    async def extract_fixtures_by_date(
        self,
        league_id: int,
        season: int,
        target_date: date,
    ) -> ExtractionResult:
        """Extract all fixtures for a date.

        Args:
            league_id: Competition ID
            season: Season year
            target_date: Date to extract

        Returns:
            ExtractionResult with fixtures data
        """
        return await self.extract(
            endpoint="fixtures",
            params={
                "league": league_id,
                "season": season,
                "date": target_date.isoformat(),
            },
            date_bucket=target_date,
        )

    async def close(self) -> None:
        """Close HTTP client."""
        await self.client.aclose()
