"""API-Football Player Photo Scraper (P5 source — fallback).

Fetches player headshots from API-Football's media CDN.
Quality capped at 40 (low resolution, often outdated).

URL pattern: https://media.api-sports.io/football/players/{external_id}.png
"""

import logging
from dataclasses import dataclass
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

APIFB_IMAGE_URL = "https://media.api-sports.io/football/players/{external_id}.png"


@dataclass
class ScrapedPhoto:
    """Result of a photo scrape attempt."""

    image_bytes: Optional[bytes] = None
    source: str = "api_football"
    content_type: Optional[str] = None
    error: Optional[str] = None
    quality_cap: int = 40  # API-Football = low quality, capped at 40


async def fetch_apifb_photo(
    external_id: int,
    timeout: float = 10.0,
) -> ScrapedPhoto:
    """Fetch player photo from API-Football CDN.

    Args:
        external_id: API-Football player ID
        timeout: Request timeout in seconds

    Returns:
        ScrapedPhoto with image bytes or error
    """
    url = APIFB_IMAGE_URL.format(external_id=external_id)

    try:
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            resp = await client.get(url)

            if resp.status_code == 404:
                return ScrapedPhoto(error="Photo not found (404)")

            resp.raise_for_status()

            image_bytes = resp.content
            if len(image_bytes) < 1000:
                return ScrapedPhoto(error=f"Image too small ({len(image_bytes)} bytes), likely placeholder")

            content_type = resp.headers.get("content-type", "image/png")
            return ScrapedPhoto(
                image_bytes=image_bytes,
                content_type=content_type,
            )

    except httpx.HTTPStatusError as e:
        return ScrapedPhoto(error=f"HTTP {e.response.status_code}")
    except Exception as e:
        return ScrapedPhoto(error=str(e))
