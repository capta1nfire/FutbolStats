"""Sofascore Player Photo Scraper (P3 source).

Fetches player headshots from Sofascore's API using player_id_mapping
for API-Football -> Sofascore ID resolution.

URL pattern: https://api.sofascore.com/api/v1/player/{sofascore_id}/image
Rate limit: 1 request / 2 seconds
"""

import logging
from dataclasses import dataclass
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

SOFASCORE_IMAGE_URL = "https://api.sofascore.com/api/v1/player/{sofascore_id}/image"

# Sofascore returns actual image bytes (JPEG/PNG), not JSON
SOFASCORE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
    "Accept": "image/webp,image/apng,image/*,*/*;q=0.8",
    "Referer": "https://www.sofascore.com/",
}


@dataclass
class ScrapedPhoto:
    """Result of a photo scrape attempt."""

    image_bytes: Optional[bytes] = None
    source: str = "sofascore"
    content_type: Optional[str] = None
    error: Optional[str] = None
    quality_cap: int = 80  # Source quality cap (Sofascore = high quality)


async def fetch_sofascore_photo(
    sofascore_id: str,
    timeout: float = 10.0,
) -> ScrapedPhoto:
    """Fetch player photo from Sofascore.

    Args:
        sofascore_id: Sofascore player ID (from player_id_mapping)
        timeout: Request timeout in seconds

    Returns:
        ScrapedPhoto with image bytes or error
    """
    url = SOFASCORE_IMAGE_URL.format(sofascore_id=sofascore_id)

    try:
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            resp = await client.get(url, headers=SOFASCORE_HEADERS)

            if resp.status_code == 404:
                return ScrapedPhoto(error="Photo not found (404)")

            resp.raise_for_status()

            content_type = resp.headers.get("content-type", "")
            if "image" not in content_type:
                return ScrapedPhoto(error=f"Unexpected content-type: {content_type}")

            image_bytes = resp.content
            if len(image_bytes) < 1000:
                return ScrapedPhoto(error=f"Image too small ({len(image_bytes)} bytes), likely placeholder")

            return ScrapedPhoto(
                image_bytes=image_bytes,
                content_type=content_type,
            )

    except httpx.HTTPStatusError as e:
        return ScrapedPhoto(error=f"HTTP {e.response.status_code}")
    except Exception as e:
        return ScrapedPhoto(error=str(e))
