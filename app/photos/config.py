"""Player Photos Pipeline Configuration.

Settings for photo scraping, validation, and processing.
Reuses LogosR2Client for R2 storage (same bucket, prefix players/).
"""

import hashlib
from functools import lru_cache
from typing import Optional

from pydantic_settings import BaseSettings


class PhotosSettings(BaseSettings):
    """Photos-specific settings."""

    # ==========================================================================
    # Feature flag
    # ==========================================================================

    PHOTOS_ENABLED: bool = False

    # ==========================================================================
    # PhotoRoom API (background removal)
    # ==========================================================================

    PHOTOROOM_API_KEY: str = ""
    PHOTOROOM_API_URL: str = "https://sdk.photoroom.com/v1/segment"

    # ==========================================================================
    # Gemini Vision (headshot validation)
    # ==========================================================================

    PHOTOS_GEMINI_VISION_ENABLED: bool = True
    GEMINI_API_KEY: str = ""

    # ==========================================================================
    # Validation thresholds
    # ==========================================================================

    PHOTOS_MIN_WIDTH: int = 100
    PHOTOS_MIN_HEIGHT: int = 100
    PHOTOS_MAX_FILE_SIZE_BYTES: int = 5 * 1024 * 1024  # 5MB (club site HQ photos)
    PHOTOS_MIN_FILE_SIZE_BYTES: int = 5 * 1024  # 5KB
    PHOTOS_ASPECT_RATIO_MIN: float = 0.4  # Club site body shots can be very vertical
    PHOTOS_ASPECT_RATIO_MAX: float = 1.0

    # ==========================================================================
    # Thumbnail sizes
    # ==========================================================================

    PHOTOS_THUMBNAIL_SIZES: list = [128, 256]
    PHOTOS_THUMBNAIL_FORMAT: str = "webp"
    PHOTOS_THUMBNAIL_QUALITY: int = 85

    # ==========================================================================
    # Identity matching
    # ==========================================================================

    PHOTOS_IDENTITY_THRESHOLD: int = 60

    # ==========================================================================
    # Rate limits
    # ==========================================================================

    PHOTOS_SOFASCORE_DELAY_SECS: float = 2.0
    PHOTOS_APIFB_DELAY_SECS: float = 0.5

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


@lru_cache
def get_photos_settings() -> PhotosSettings:
    """Get cached Photos settings instance."""
    return PhotosSettings()


# ==========================================================================
# R2 Key Builders (fix #1: immutable keys with content_hash)
# ==========================================================================


def build_player_photo_key(
    ext_id: int,
    content_hash: str,
    asset_type: str,
    style: str = "raw",
    ext: str = "png",
) -> str:
    """Build immutable R2 key for player photo.

    Format: players/{ext_id}/{hash[:12]}/{asset_type}_{style}.{ext}

    The content_hash prefix makes each version a unique key,
    enabling cache-busting and rollback without CDN purge.

    Args:
        ext_id: Player external ID (API-Football)
        content_hash: SHA-256 hex digest of image content
        asset_type: card | thumb
        style: raw | segmented | composed
        ext: File extension (png, webp)

    Returns:
        R2 key: players/12345/a1b2c3d4e5f6/card_segmented.png
    """
    return f"players/{ext_id}/{content_hash[:12]}/{asset_type}_{style}.{ext}"


def compute_content_hash(image_bytes: bytes) -> str:
    """Compute SHA-256 hash of image content.

    Used for:
    - Immutable R2 keys (dedup + cache-busting)
    - Known-bad image detection

    Args:
        image_bytes: Raw image bytes

    Returns:
        SHA-256 hex digest (64 chars)
    """
    return hashlib.sha256(image_bytes).hexdigest()
