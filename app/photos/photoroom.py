"""PhotoRoom API Client for Background Removal.

Uses PhotoRoom Segment API to remove backgrounds from player photos,
producing transparent PNGs suitable for lineup UI compositing.

API: POST https://sdk.photoroom.com/v1/segment
Auth: x-api-key header
Rate limit: 1 req/sec (Basic plan)
"""

import logging
from typing import Optional

import httpx

from app.photos.config import get_photos_settings

logger = logging.getLogger(__name__)
photos_settings = get_photos_settings()


async def remove_background(image_bytes: bytes) -> Optional[bytes]:
    """Remove background from player photo using PhotoRoom API.

    Args:
        image_bytes: Raw image bytes (PNG/JPEG)

    Returns:
        PNG bytes with transparent background, or None on failure
    """
    api_key = photos_settings.PHOTOROOM_API_KEY
    if not api_key:
        logger.warning("PHOTOROOM_API_KEY not set, skipping background removal")
        return None

    api_url = photos_settings.PHOTOROOM_API_URL

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                api_url,
                headers={"x-api-key": api_key},
                files={"image_file": ("photo.png", image_bytes, "image/png")},
            )
            resp.raise_for_status()

        result_bytes = resp.content
        logger.info(
            f"PhotoRoom: background removed ({len(image_bytes)} -> {len(result_bytes)} bytes)"
        )
        return result_bytes

    except httpx.HTTPStatusError as e:
        logger.error(f"PhotoRoom API error: {e.response.status_code} - {e.response.text[:200]}")
        return None
    except Exception as e:
        logger.error(f"PhotoRoom request failed: {e}")
        return None
