"""Cloudflare R2 Client for Logo Storage.

Provides async S3-compatible operations for storing and retrieving
team and competition logos.

Differences from TITAN R2 client:
- Handles binary data (images), not just JSON text
- Supports multiple content types (PNG, WebP)
- Methods tailored for logo operations

Usage:
    client = get_logos_r2_client()
    if client:
        await client.upload_logo(team_id=123, variant="front_3d", image_bytes=data)
        data = await client.download_logo(team_id=123, variant="front_3d")
"""

import logging
from typing import Optional

from app.logos.config import (
    get_logos_settings,
    build_team_logo_key,
    build_team_thumbnail_key,
    build_competition_logo_key,
    build_competition_thumbnail_key,
)

logger = logging.getLogger(__name__)
logos_settings = get_logos_settings()


class LogosR2Client:
    """Async Cloudflare R2 client for logo storage.

    S3-compatible API optimized for image storage and retrieval.
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

    # ==========================================================================
    # Low-level operations
    # ==========================================================================

    async def put_object(
        self,
        key: str,
        body: bytes,
        content_type: str = "image/png",
    ) -> bool:
        """Upload binary object to R2.

        Args:
            key: Object key
            body: Binary content to upload
            content_type: MIME type (default: image/png)

        Returns:
            True if successful, False otherwise
        """
        try:
            async with await self._get_client() as client:
                await client.put_object(
                    Bucket=self.bucket,
                    Key=key,
                    Body=body,
                    ContentType=content_type,
                )
                logger.debug(f"LogosR2: Uploaded {key} ({len(body)} bytes)")
                return True
        except Exception as e:
            logger.error(f"LogosR2: Failed to upload {key}: {e}")
            return False

    async def get_object(self, key: str) -> Optional[bytes]:
        """Download binary object from R2.

        Args:
            key: Object key

        Returns:
            Binary content or None if not found/error
        """
        try:
            async with await self._get_client() as client:
                response = await client.get_object(
                    Bucket=self.bucket,
                    Key=key,
                )
                body = await response["Body"].read()
                logger.debug(f"LogosR2: Downloaded {key} ({len(body)} bytes)")
                return body
        except Exception as e:
            error_str = str(e).lower()
            if "nosuchkey" in error_str or "not found" in error_str or "404" in error_str:
                logger.warning(f"LogosR2: Object not found: {key}")
            else:
                logger.error(f"LogosR2: Failed to download {key}: {e}")
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
                logger.debug(f"LogosR2: Deleted {key}")
                return True
        except Exception as e:
            logger.error(f"LogosR2: Failed to delete {key}: {e}")
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

    async def list_objects(self, prefix: str) -> list[str]:
        """List objects with given prefix.

        Args:
            prefix: Key prefix (e.g., "teams/123/")

        Returns:
            List of object keys
        """
        try:
            async with await self._get_client() as client:
                response = await client.list_objects_v2(
                    Bucket=self.bucket,
                    Prefix=prefix,
                )
                contents = response.get("Contents", [])
                return [obj["Key"] for obj in contents]
        except Exception as e:
            logger.error(f"LogosR2: Failed to list {prefix}: {e}")
            return []

    async def close(self) -> None:
        """Close the client session."""
        self._session = None
        logger.debug("LogosR2: Client closed")

    # ==========================================================================
    # High-level team logo operations
    # ==========================================================================

    async def upload_team_logo(
        self,
        team_id: int,
        variant: str,
        image_bytes: bytes,
        content_type: str = "image/png",
        apifb_id: Optional[int] = None,
        slug: Optional[str] = None,
        revision: int = 1,
    ) -> Optional[str]:
        """Upload team logo variant with immutable versioning.

        Args:
            team_id: Internal team ID
            variant: Logo variant (original, front_3d, facing_right, facing_left)
            image_bytes: PNG image data
            content_type: MIME type
            apifb_id: API-Football team ID (for traceability)
            slug: Team name slug (for readability)
            revision: Asset revision number

        Returns:
            R2 key if successful, None otherwise
        """
        ext = "webp" if "webp" in content_type else "png"
        key = build_team_logo_key(
            team_id, variant, ext,
            apifb_id=apifb_id, slug=slug, revision=revision
        )
        success = await self.put_object(key, image_bytes, content_type)
        return key if success else None

    async def download_team_logo(
        self,
        team_id: int,
        variant: str,
        ext: str = "png",
        apifb_id: Optional[int] = None,
        slug: Optional[str] = None,
        revision: int = 1,
    ) -> Optional[bytes]:
        """Download team logo variant.

        Args:
            team_id: Internal team ID
            variant: Logo variant
            ext: File extension
            apifb_id: API-Football team ID
            slug: Team name slug
            revision: Asset revision number

        Returns:
            Image bytes or None if not found
        """
        key = build_team_logo_key(
            team_id, variant, ext,
            apifb_id=apifb_id, slug=slug, revision=revision
        )
        return await self.get_object(key)

    async def upload_team_thumbnail(
        self,
        team_id: int,
        variant: str,
        size: int,
        image_bytes: bytes,
        apifb_id: Optional[int] = None,
        slug: Optional[str] = None,
        revision: int = 1,
    ) -> Optional[str]:
        """Upload team logo thumbnail with immutable versioning.

        Args:
            team_id: Internal team ID
            variant: Logo variant
            size: Thumbnail size (64, 128, 256, 512)
            image_bytes: WebP image data
            apifb_id: API-Football team ID
            slug: Team name slug
            revision: Asset revision number

        Returns:
            R2 key if successful, None otherwise
        """
        key = build_team_thumbnail_key(
            team_id, variant, size,
            apifb_id=apifb_id, slug=slug, revision=revision
        )
        success = await self.put_object(key, image_bytes, "image/webp")
        return key if success else None

    async def get_team_logo_urls(
        self,
        team_id: int,
        apifb_id: Optional[int] = None,
        slug: Optional[str] = None,
        revision: int = 1,
    ) -> dict:
        """Get all thumbnail URLs for a team.

        Args:
            team_id: Internal team ID
            apifb_id: API-Football team ID
            slug: Team name slug
            revision: Asset revision number

        Returns:
            Dict with structure:
            {
                "front": {"64": "url", "128": "url", ...},
                "right": {"64": "url", ...},
                "left": {"64": "url", ...}
            }
        """
        base_url = logos_settings.LOGOS_CDN_BASE_URL.rstrip("/")
        if not base_url:
            return {}

        sizes = logos_settings.LOGOS_THUMBNAIL_SIZES
        variants = ["front_3d", "facing_right", "facing_left"]
        variant_map = {"front_3d": "front", "facing_right": "right", "facing_left": "left"}

        urls = {}
        for variant in variants:
            key = variant_map[variant]
            urls[key] = {}
            for size in sizes:
                thumb_key = build_team_thumbnail_key(
                    team_id, variant, size,
                    apifb_id=apifb_id, slug=slug, revision=revision
                )
                urls[key][str(size)] = f"{base_url}/{thumb_key}"

        return urls

    # ==========================================================================
    # High-level competition logo operations
    # ==========================================================================

    async def upload_competition_logo(
        self,
        league_id: int,
        variant: str,
        image_bytes: bytes,
        content_type: str = "image/png",
        slug: Optional[str] = None,
        revision: int = 1,
    ) -> Optional[str]:
        """Upload competition logo with immutable versioning.

        Args:
            league_id: League ID
            variant: Logo variant (original, main)
            image_bytes: PNG image data
            content_type: MIME type
            slug: League name slug
            revision: Asset revision number

        Returns:
            R2 key if successful, None otherwise
        """
        ext = "webp" if "webp" in content_type else "png"
        key = build_competition_logo_key(
            league_id, variant, ext,
            slug=slug, revision=revision
        )
        success = await self.put_object(key, image_bytes, content_type)
        return key if success else None

    async def upload_competition_thumbnail(
        self,
        league_id: int,
        size: int,
        image_bytes: bytes,
        slug: Optional[str] = None,
        revision: int = 1,
    ) -> Optional[str]:
        """Upload competition logo thumbnail with immutable versioning.

        Args:
            league_id: League ID
            size: Thumbnail size
            image_bytes: WebP image data
            slug: League name slug
            revision: Asset revision number

        Returns:
            R2 key if successful, None otherwise
        """
        key = build_competition_thumbnail_key(
            league_id, size,
            slug=slug, revision=revision
        )
        success = await self.put_object(key, image_bytes, "image/webp")
        return key if success else None

    async def get_competition_logo_urls(
        self,
        league_id: int,
        slug: Optional[str] = None,
        revision: int = 1,
    ) -> dict:
        """Get all thumbnail URLs for a competition.

        Args:
            league_id: League ID
            slug: League name slug
            revision: Asset revision number

        Returns:
            Dict: {"64": "url", "128": "url", ...}
        """
        base_url = logos_settings.LOGOS_CDN_BASE_URL.rstrip("/")
        if not base_url:
            return {}

        sizes = logos_settings.LOGOS_THUMBNAIL_SIZES
        urls = {}
        for size in sizes:
            thumb_key = build_competition_thumbnail_key(
                league_id, size,
                slug=slug, revision=revision
            )
            urls[str(size)] = f"{base_url}/{thumb_key}"

        return urls

    # ==========================================================================
    # Cleanup operations
    # ==========================================================================

    async def delete_team_logos(self, team_id: int) -> int:
        """Delete all logos for a team.

        Args:
            team_id: Internal team ID

        Returns:
            Number of objects deleted
        """
        prefix = f"teams/{team_id}/"
        keys = await self.list_objects(prefix)
        deleted = 0
        for key in keys:
            if await self.delete_object(key):
                deleted += 1
        logger.info(f"LogosR2: Deleted {deleted} objects for team {team_id}")
        return deleted

    async def delete_competition_logos(self, league_id: int) -> int:
        """Delete all logos for a competition.

        Args:
            league_id: League ID

        Returns:
            Number of objects deleted
        """
        prefix = f"competitions/{league_id}/"
        keys = await self.list_objects(prefix)
        deleted = 0
        for key in keys:
            if await self.delete_object(key):
                deleted += 1
        logger.info(f"LogosR2: Deleted {deleted} objects for competition {league_id}")
        return deleted


# ==========================================================================
# Global client instance
# ==========================================================================

_logos_r2_client: Optional[LogosR2Client] = None


def get_logos_r2_client() -> Optional[LogosR2Client]:
    """Get Logos R2 client if enabled and configured.

    Returns:
        LogosR2Client instance or None if disabled/not configured
    """
    global _logos_r2_client

    if not logos_settings.LOGOS_R2_ENABLED:
        return None

    if not logos_settings.LOGOS_R2_ENDPOINT_URL:
        logger.warning("Logos R2 enabled but LOGOS_R2_ENDPOINT_URL not set")
        return None

    if _logos_r2_client is None:
        _logos_r2_client = LogosR2Client(
            endpoint_url=logos_settings.LOGOS_R2_ENDPOINT_URL,
            access_key_id=logos_settings.LOGOS_R2_ACCESS_KEY_ID,
            secret_access_key=logos_settings.LOGOS_R2_SECRET_ACCESS_KEY,
            bucket=logos_settings.LOGOS_R2_BUCKET,
        )
        logger.info(f"LogosR2: Client initialized (bucket={logos_settings.LOGOS_R2_BUCKET})")

    return _logos_r2_client
