"""3D Logo System Configuration.

Settings for R2 storage, CDN, and IA generation.
Uses same R2 credentials as TITAN but different bucket.
"""

from functools import lru_cache
from typing import Optional

from pydantic_settings import BaseSettings


class LogosSettings(BaseSettings):
    """Logos-specific settings (supplements main app Settings)."""

    # ==========================================================================
    # R2 Storage Configuration
    # ==========================================================================
    # Uses same Cloudflare account as TITAN, different bucket

    LOGOS_R2_ENABLED: bool = False
    LOGOS_R2_ENDPOINT_URL: str = ""  # https://<account_id>.r2.cloudflarestorage.com
    LOGOS_R2_ACCESS_KEY_ID: str = ""
    LOGOS_R2_SECRET_ACCESS_KEY: str = ""
    LOGOS_R2_BUCKET: str = "futbolstats-logos"

    # CDN configuration
    LOGOS_CDN_BASE_URL: str = ""  # https://logos.futbolstats.app or R2 public URL

    # ==========================================================================
    # Cloudflare API (for CDN invalidation)
    # ==========================================================================

    CLOUDFLARE_ZONE_ID: str = ""
    CLOUDFLARE_API_TOKEN: str = ""  # Token with cache purge permissions

    # ==========================================================================
    # IA Generation Configuration
    # ==========================================================================

    # Default IA model for generation
    LOGOS_IA_MODEL: str = "imagen-3"  # imagen-3, dall-e-3, sdxl

    # Google Gemini/Imagen
    GEMINI_API_KEY: str = ""

    # Free tier vs Paid tier
    # True = Google AI Studio (free, ~50 imgs/day limit)
    # False = Vertex AI ($0.03/img, no limit)
    LOGOS_USE_FREE_TIER: bool = True

    # OpenAI (DALL-E)
    OPENAI_API_KEY: str = ""

    # Replicate (SDXL)
    REPLICATE_API_TOKEN: str = ""

    # Generation limits
    LOGOS_BATCH_CONCURRENT_LIMIT: int = 5  # Max concurrent IA calls
    LOGOS_BATCH_RATE_LIMIT_RPM: int = 50  # Requests per minute

    # Cost Guard (Kimi recommendation)
    LOGOS_MAX_BATCH_COST_USD: float = 50.0  # Max cost per batch job
    LOGOS_COST_PER_IMAGE_DALLE: float = 0.04  # DALL-E 3 HD cost
    LOGOS_COST_PER_IMAGE_SDXL: float = 0.006  # SDXL via Replicate (p50)
    LOGOS_COST_PER_IMAGE_IMAGEN: float = 0.03  # Google Imagen (Vertex AI)
    LOGOS_COST_PER_IMAGE_IMAGEN_FREE: float = 0.0  # Google AI Studio (free tier)

    # Rate limits per IA model (requests per minute)
    LOGOS_DALLE_RPM: int = 5  # DALL-E 3 limit
    LOGOS_SDXL_RPM: int = 10  # SDXL limit
    LOGOS_IMAGEN_RPM: int = 10  # Imagen 3 limit

    # Retry policy
    LOGOS_IA_MAX_RETRIES: int = 3
    LOGOS_IA_RETRY_DELAY_SECONDS: int = 5

    # ==========================================================================
    # Validation Configuration
    # ==========================================================================

    LOGOS_MIN_WIDTH: int = 512
    LOGOS_MIN_HEIGHT: int = 512
    LOGOS_MAX_FILE_SIZE_BYTES: int = 5 * 1024 * 1024  # 5MB
    LOGOS_ASPECT_RATIO_TOLERANCE: float = 0.05  # 5% tolerance for 1:1

    # ==========================================================================
    # Thumbnail Configuration
    # ==========================================================================

    LOGOS_THUMBNAIL_SIZES: list[int] = [64, 128, 256, 512]
    LOGOS_THUMBNAIL_FORMAT: str = "webp"
    LOGOS_THUMBNAIL_QUALITY: int = 85

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache
def get_logos_settings() -> LogosSettings:
    """Get cached Logos settings instance."""
    return LogosSettings()


# ==========================================================================
# R2 Key Builders
# ==========================================================================


def build_team_logo_key(team_id: int, variant: str, ext: str = "png") -> str:
    """Build R2 key for team logo.

    Args:
        team_id: Internal team ID
        variant: Logo variant (original, front_3d, facing_right, facing_left)
        ext: File extension (png for originals, webp for thumbnails)

    Returns:
        R2 key: teams/{team_id}/{variant}.{ext}
    """
    return f"teams/{team_id}/{variant}.{ext}"


def build_team_thumbnail_key(team_id: int, variant: str, size: int) -> str:
    """Build R2 key for team logo thumbnail.

    Args:
        team_id: Internal team ID
        variant: Logo variant (front_3d, facing_right, facing_left)
        size: Thumbnail size (64, 128, 256, 512)

    Returns:
        R2 key: teams/{team_id}/{variant}_{size}.webp
    """
    return f"teams/{team_id}/{variant}_{size}.webp"


def build_competition_logo_key(league_id: int, variant: str, ext: str = "png") -> str:
    """Build R2 key for competition logo.

    Args:
        league_id: League ID from admin_leagues
        variant: Logo variant (original, main)
        ext: File extension

    Returns:
        R2 key: competitions/{league_id}/{variant}.{ext}
    """
    return f"competitions/{league_id}/{variant}.{ext}"


def build_competition_thumbnail_key(league_id: int, size: int) -> str:
    """Build R2 key for competition logo thumbnail.

    Args:
        league_id: League ID
        size: Thumbnail size

    Returns:
        R2 key: competitions/{league_id}/main_{size}.webp
    """
    return f"competitions/{league_id}/main_{size}.webp"
