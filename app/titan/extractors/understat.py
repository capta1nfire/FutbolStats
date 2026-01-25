"""TITAN Understat Extractor - xG data extraction with PIT compliance.

Wraps the existing UnderstatProvider to add TITAN features:
- captured_at timestamps (timezone-aware UTC)
- Idempotency key computation
- Persistence to titan.raw_extractions

Strategy (per Auditor condition #3):
1. First try to compute xG from public.match_understat_team (PIT-safe)
2. Only if data is missing, fetch from Understat API and persist to raw_extractions
"""

import hashlib
import json
import logging
from dataclasses import asdict
from datetime import date, datetime, timezone
from typing import Optional
from uuid import UUID

from app.etl.understat_provider import UnderstatProvider, UnderstatMatchData
from app.titan.config import get_titan_settings
from app.titan.extractors.base import (
    TitanExtractor,
    ExtractionResult,
    compute_idempotency_key,
)

logger = logging.getLogger(__name__)
titan_settings = get_titan_settings()


def _utc_now() -> datetime:
    """Get current UTC timestamp (timezone-aware) for TIMESTAMPTZ compatibility."""
    return datetime.now(timezone.utc)


class TitanUnderstatExtractor(TitanExtractor):
    """TITAN-compliant wrapper for UnderstatProvider.

    Adds:
    - captured_at timestamps (timezone-aware UTC)
    - Idempotency key computation
    - Persistence to titan.raw_extractions

    Usage:
        extractor = TitanUnderstatExtractor()
        result = await extractor.extract_match_xg(
            understat_match_id="12345",
            date_bucket=date.today(),
        )
        await extractor.close()
    """

    SOURCE_ID = "understat"

    def __init__(self):
        super().__init__()
        self._provider = UnderstatProvider()

    async def extract_match_xg(
        self,
        understat_match_id: str,
        date_bucket: date,
    ) -> ExtractionResult:
        """Extract xG for a single match from Understat.

        Args:
            understat_match_id: The Understat match ID
            date_bucket: Date for idempotency bucketing

        Returns:
            ExtractionResult with xG data or error info
        """
        job_id = self._generate_job_id()
        endpoint = "match_xg"
        params = {"match_id": understat_match_id}
        url = f"https://understat.com/getMatchData/{understat_match_id}"

        # Compute idempotency key BEFORE making request
        idempotency_key = compute_idempotency_key(
            source_id=self.SOURCE_ID,
            endpoint=endpoint,
            params=params,
            date_bucket=date_bucket,
        )

        params_hash = hashlib.md5(
            json.dumps(params, sort_keys=True).encode()
        ).hexdigest()

        try:
            # Fetch from Understat API
            data = await self._provider.fetch_match_xg_by_understat_id(understat_match_id)

            # Capture timestamp AFTER response received
            captured_at = self._get_captured_at()

            if data is None:
                return ExtractionResult(
                    source_id=self.SOURCE_ID,
                    job_id=job_id,
                    url=url,
                    params_hash=params_hash,
                    date_bucket=date_bucket,
                    response_type="json",
                    response_body=None,
                    http_status=404,
                    captured_at=captured_at,
                    idempotency_key=idempotency_key,
                    is_success=False,
                    error_type="not_found",
                    error_message=f"Match {understat_match_id} not found on Understat",
                )

            # Convert dataclass to dict for JSON storage
            # Fix: use timezone-aware captured_at
            data.captured_at = captured_at
            response_body = {
                "xg_home": data.xg_home,
                "xg_away": data.xg_away,
                "npxg_home": data.npxg_home,
                "npxg_away": data.npxg_away,
                "xga_home": data.xga_home,
                "xga_away": data.xga_away,
                "xpts_home": data.xpts_home,
                "xpts_away": data.xpts_away,
                "source_version": data.source_version,
            }

            return ExtractionResult(
                source_id=self.SOURCE_ID,
                job_id=job_id,
                url=url,
                params_hash=params_hash,
                date_bucket=date_bucket,
                response_type="json",
                response_body=response_body,
                http_status=200,
                captured_at=captured_at,
                idempotency_key=idempotency_key,
                is_success=True,
            )

        except Exception as e:
            captured_at = self._get_captured_at()
            logger.error(f"Understat extraction failed for {understat_match_id}: {e}")

            return ExtractionResult(
                source_id=self.SOURCE_ID,
                job_id=job_id,
                url=url,
                params_hash=params_hash,
                date_bucket=date_bucket,
                response_type="json",
                response_body=None,
                http_status=500,
                captured_at=captured_at,
                idempotency_key=idempotency_key,
                is_success=False,
                error_type="extraction_error",
                error_message=str(e),
            )

    async def extract(
        self,
        endpoint: str,
        params: dict,
        date_bucket: date,
    ) -> ExtractionResult:
        """Generic extract method (required by ABC).

        For Understat, we only support 'match_xg' endpoint.
        """
        if endpoint == "match_xg":
            match_id = params.get("match_id")
            if not match_id:
                raise ValueError("match_id required in params")
            return await self.extract_match_xg(
                understat_match_id=str(match_id),
                date_bucket=date_bucket,
            )
        else:
            raise ValueError(f"Unknown endpoint: {endpoint}")

    async def close(self) -> None:
        """Close underlying provider connections."""
        await self._provider.close()
        logger.debug("TitanUnderstatExtractor closed")
