"""Abstract base class for TITAN extractors with PIT compliance.

All extractors MUST:
1. Capture timestamps at extraction time (captured_at)
2. Compute deterministic idempotency keys
3. Return standardized ExtractionResult objects
"""

import hashlib
import json
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, date, timezone
from typing import Any, Optional


@dataclass
class ExtractionResult:
    """Standardized result from any TITAN extractor.

    This maps directly to titan.raw_extractions table schema.
    """

    # Source identification
    source_id: str
    job_id: uuid.UUID

    # Request info
    url: str
    endpoint: str
    params_hash: str
    date_bucket: date

    # Response
    response_type: str  # 'json', 'html', 'error'
    response_body: Optional[dict]  # JSON response (None if error)
    http_status: int
    response_time_ms: Optional[int]

    # PIT Compliance (CRITICAL)
    captured_at: datetime

    # Idempotency (CRITICAL)
    idempotency_key: str

    # Error info (if failed)
    error_type: Optional[str] = None
    error_message: Optional[str] = None

    @property
    def is_success(self) -> bool:
        """Check if extraction was successful."""
        return 200 <= self.http_status < 300 and self.error_type is None

    def to_dict(self) -> dict:
        """Convert to dictionary for database insertion."""
        return {
            "source_id": self.source_id,
            "job_id": self.job_id,
            "url": self.url,
            "endpoint": self.endpoint,
            "params_hash": self.params_hash,
            "date_bucket": self.date_bucket,
            "response_type": self.response_type,
            "response_body": self.response_body,
            "http_status": self.http_status,
            "response_time_ms": self.response_time_ms,
            "captured_at": self.captured_at,
            "idempotency_key": self.idempotency_key,
        }


def compute_idempotency_key(
    source_id: str,
    endpoint: str,
    params: dict,
    date_bucket: date,
) -> str:
    """Compute deterministic idempotency key.

    Formula: SHA256(source_id|endpoint|sorted_params_json|date_bucket)[:32]

    Args:
        source_id: Source identifier (e.g., 'api_football')
        endpoint: API endpoint name (e.g., 'fixtures', 'odds')
        params: Request parameters (will be JSON-serialized sorted)
        date_bucket: Target date for the extraction

    Returns:
        32-character hex string (CHAR(32) in DB)
    """
    # Normalize params to deterministic JSON string
    normalized_params = json.dumps(params, sort_keys=True, default=str)

    # Build raw string for hashing
    raw = f"{source_id}|{endpoint}|{normalized_params}|{date_bucket.isoformat()}"

    # SHA256 and truncate to 32 chars
    return hashlib.sha256(raw.encode()).hexdigest()[:32]


def compute_params_hash(params: dict) -> str:
    """Compute MD5 hash of params for debugging.

    Not used for idempotency (SHA256 is), just for human-readable debugging.

    Args:
        params: Request parameters

    Returns:
        32-character hex string (MD5)
    """
    normalized = json.dumps(params, sort_keys=True, default=str)
    return hashlib.md5(normalized.encode()).hexdigest()


class TitanExtractor(ABC):
    """Abstract base class for all TITAN extractors.

    Extractors wrap external APIs and ensure:
    1. PIT compliance via captured_at timestamps
    2. Idempotency via deterministic keys
    3. Standardized output via ExtractionResult
    """

    # Must be set by subclasses
    SOURCE_ID: str = ""

    def __init__(self):
        if not self.SOURCE_ID:
            raise ValueError(f"{self.__class__.__name__} must define SOURCE_ID")

    def _generate_job_id(self) -> uuid.UUID:
        """Generate unique job ID for tracking."""
        return uuid.uuid4()

    def _get_captured_at(self) -> datetime:
        """Get current UTC timestamp for PIT compliance (timezone-aware).

        CRITICAL: This must be called AFTER the API response is received,
        representing the moment the data was captured.

        Returns timezone-aware datetime in UTC for TIMESTAMPTZ compatibility.
        """
        return datetime.now(timezone.utc)

    @abstractmethod
    async def extract(
        self,
        endpoint: str,
        params: dict,
        date_bucket: date,
    ) -> ExtractionResult:
        """Extract data from external API.

        Args:
            endpoint: API endpoint to call
            params: Request parameters
            date_bucket: Logical date for the extraction (for partitioning)

        Returns:
            ExtractionResult with all metadata and response
        """
        pass

    @abstractmethod
    async def close(self) -> None:
        """Close any open connections."""
        pass
