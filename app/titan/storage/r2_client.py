"""Cloudflare R2 Client for TITAN raw extraction storage.

Provides async S3-compatible operations for offloading large JSONB responses
from PostgreSQL to Cloudflare R2.

Usage:
    client = get_r2_client()
    if client:
        await client.put_object(key="source/date/job.json", body=json_str)
        data = await client.get_object(key="source/date/job.json")
        await client.close()
"""

import logging
from contextlib import asynccontextmanager
from typing import Optional

from app.titan.config import get_titan_settings

logger = logging.getLogger(__name__)
titan_settings = get_titan_settings()


class R2Client:
    """Async Cloudflare R2 client using aioboto3.

    S3-compatible API for storing and retrieving extraction responses.
    """

    def __init__(
        self,
        endpoint_url: str,
        access_key_id: str,
        secret_access_key: str,
        bucket: str,
    ):
        self.endpoint_url = endpoint_url
        self.access_key_id = access_key_id
        self.secret_access_key = secret_access_key
        self.bucket = bucket
        self._session = None

    async def _get_client(self):
        """Get or create aioboto3 S3 client."""
        if self._session is None:
            import aioboto3
            self._session = aioboto3.Session()
        return self._session.client(
            "s3",
            endpoint_url=self.endpoint_url,
            aws_access_key_id=self.access_key_id,
            aws_secret_access_key=self.secret_access_key,
        )

    async def put_object(
        self,
        key: str,
        body: str,
        content_type: str = "application/json",
    ) -> bool:
        """Upload object to R2.

        Args:
            key: Object key (e.g., "understat/2026-01-25/uuid.json")
            body: String content to upload
            content_type: MIME type (default: application/json)

        Returns:
            True if successful, False otherwise
        """
        try:
            async with await self._get_client() as client:
                await client.put_object(
                    Bucket=self.bucket,
                    Key=key,
                    Body=body.encode("utf-8"),
                    ContentType=content_type,
                )
                logger.debug(f"R2: Uploaded {key} ({len(body)} bytes)")
                return True
        except Exception as e:
            logger.error(f"R2: Failed to upload {key}: {e}")
            return False

    async def get_object(self, key: str) -> Optional[str]:
        """Download object from R2.

        Args:
            key: Object key

        Returns:
            String content or None if not found/error
        """
        try:
            async with await self._get_client() as client:
                response = await client.get_object(
                    Bucket=self.bucket,
                    Key=key,
                )
                body = await response["Body"].read()
                logger.debug(f"R2: Downloaded {key} ({len(body)} bytes)")
                return body.decode("utf-8")
        except Exception as e:
            # Handle NoSuchKey and other errors uniformly
            # botocore.exceptions.ClientError with Code='NoSuchKey' for missing objects
            error_str = str(e).lower()
            if "nosuchkey" in error_str or "not found" in error_str or "404" in error_str:
                logger.warning(f"R2: Object not found: {key}")
            else:
                logger.error(f"R2: Failed to download {key}: {e}")
            return None

    async def delete_object(self, key: str) -> bool:
        """Delete object from R2.

        Args:
            key: Object key

        Returns:
            True if successful, False otherwise
        """
        try:
            async with await self._get_client() as client:
                await client.delete_object(
                    Bucket=self.bucket,
                    Key=key,
                )
                logger.debug(f"R2: Deleted {key}")
                return True
        except Exception as e:
            logger.error(f"R2: Failed to delete {key}: {e}")
            return False

    async def object_exists(self, key: str) -> bool:
        """Check if object exists in R2.

        Args:
            key: Object key

        Returns:
            True if exists, False otherwise
        """
        try:
            async with await self._get_client() as client:
                await client.head_object(
                    Bucket=self.bucket,
                    Key=key,
                )
                return True
        except Exception:
            return False

    async def close(self) -> None:
        """Close the client session."""
        self._session = None
        logger.debug("R2: Client closed")


# Global client instance (lazy initialization)
_r2_client: Optional[R2Client] = None


def get_r2_client() -> Optional[R2Client]:
    """Get R2 client if enabled and configured.

    Returns:
        R2Client instance or None if R2 is disabled/not configured
    """
    global _r2_client

    if not titan_settings.R2_ENABLED:
        return None

    if not titan_settings.R2_ENDPOINT_URL:
        logger.warning("R2 enabled but TITAN_R2_ENDPOINT_URL not set")
        return None

    if _r2_client is None:
        _r2_client = R2Client(
            endpoint_url=titan_settings.R2_ENDPOINT_URL,
            access_key_id=titan_settings.R2_ACCESS_KEY_ID,
            secret_access_key=titan_settings.R2_SECRET_ACCESS_KEY,
            bucket=titan_settings.R2_BUCKET,
        )
        logger.info(f"R2: Client initialized (bucket={titan_settings.R2_BUCKET})")

    return _r2_client


def build_r2_key(source_id: str, date_bucket: str, job_id: str) -> str:
    """Build standardized R2 object key.

    Format: {source_id}/{date_bucket}/{job_id}.json

    Args:
        source_id: Extraction source (e.g., "understat", "api_football")
        date_bucket: Date string (e.g., "2026-01-25")
        job_id: UUID job identifier

    Returns:
        R2 object key string
    """
    return f"{source_id}/{date_bucket}/{job_id}.json"
