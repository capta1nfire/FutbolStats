"""CDN Cache Invalidation for Logo Regenerations.

Purges Cloudflare cache when logos are regenerated to ensure
clients receive updated images.

Uses Cloudflare API v4 for cache purge operations.
"""

import logging
from typing import Optional

import httpx

from app.logos.config import get_logos_settings, build_team_thumbnail_key, build_competition_thumbnail_key

logger = logging.getLogger(__name__)
logos_settings = get_logos_settings()

# Cloudflare API base URL
CLOUDFLARE_API_BASE = "https://api.cloudflare.com/client/v4"


async def purge_cache_by_urls(urls: list[str]) -> tuple[bool, Optional[str]]:
    """Purge specific URLs from Cloudflare cache.

    Args:
        urls: List of full URLs to purge

    Returns:
        Tuple of (success, error_message)
    """
    zone_id = logos_settings.CLOUDFLARE_ZONE_ID
    api_token = logos_settings.CLOUDFLARE_API_TOKEN

    if not zone_id or not api_token:
        logger.warning("CDN purge skipped: CLOUDFLARE_ZONE_ID or CLOUDFLARE_API_TOKEN not configured")
        return False, "Cloudflare not configured"

    if not urls:
        return True, None

    # Cloudflare limits to 30 URLs per request
    MAX_URLS_PER_REQUEST = 30
    total_purged = 0
    errors = []

    async with httpx.AsyncClient() as client:
        for i in range(0, len(urls), MAX_URLS_PER_REQUEST):
            batch = urls[i : i + MAX_URLS_PER_REQUEST]

            try:
                response = await client.post(
                    f"{CLOUDFLARE_API_BASE}/zones/{zone_id}/purge_cache",
                    headers={
                        "Authorization": f"Bearer {api_token}",
                        "Content-Type": "application/json",
                    },
                    json={"files": batch},
                    timeout=30.0,
                )

                if response.status_code == 200:
                    result = response.json()
                    if result.get("success"):
                        total_purged += len(batch)
                        logger.debug(f"CDN: Purged {len(batch)} URLs")
                    else:
                        error = result.get("errors", [{}])[0].get("message", "Unknown error")
                        errors.append(error)
                        logger.error(f"CDN purge failed: {error}")
                else:
                    errors.append(f"HTTP {response.status_code}")
                    logger.error(f"CDN purge HTTP error: {response.status_code}")

            except httpx.TimeoutException:
                errors.append("Timeout")
                logger.error("CDN purge timeout")
            except Exception as e:
                errors.append(str(e))
                logger.error(f"CDN purge error: {e}")

    if errors:
        return False, "; ".join(errors)

    logger.info(f"CDN: Successfully purged {total_purged} URLs")
    return True, None


async def invalidate_team_logo_cdn(
    team_id: int,
    variants: Optional[list[str]] = None,
) -> tuple[bool, Optional[str]]:
    """Invalidate CDN cache for team logo.

    Purges all thumbnail sizes for specified variants.

    Args:
        team_id: Internal team ID
        variants: List of variants to purge (default: all)

    Returns:
        Tuple of (success, error_message)
    """
    if variants is None:
        variants = ["front_3d", "facing_right", "facing_left"]

    base_url = logos_settings.LOGOS_CDN_BASE_URL.rstrip("/")
    if not base_url:
        logger.warning("CDN invalidation skipped: LOGOS_CDN_BASE_URL not configured")
        return False, "CDN base URL not configured"

    sizes = logos_settings.LOGOS_THUMBNAIL_SIZES
    urls_to_purge = []

    for variant in variants:
        # Original PNG
        urls_to_purge.append(f"{base_url}/teams/{team_id}/{variant}.png")

        # WebP thumbnails
        for size in sizes:
            thumb_key = build_team_thumbnail_key(team_id, variant, size)
            urls_to_purge.append(f"{base_url}/{thumb_key}")

    success, error = await purge_cache_by_urls(urls_to_purge)

    if success:
        logger.info(f"CDN invalidated for team {team_id}: {len(urls_to_purge)} paths")

    return success, error


async def invalidate_competition_logo_cdn(league_id: int) -> tuple[bool, Optional[str]]:
    """Invalidate CDN cache for competition logo.

    Args:
        league_id: League ID

    Returns:
        Tuple of (success, error_message)
    """
    base_url = logos_settings.LOGOS_CDN_BASE_URL.rstrip("/")
    if not base_url:
        return False, "CDN base URL not configured"

    sizes = logos_settings.LOGOS_THUMBNAIL_SIZES
    urls_to_purge = []

    # Original PNG
    urls_to_purge.append(f"{base_url}/competitions/{league_id}/main.png")

    # WebP thumbnails
    for size in sizes:
        thumb_key = build_competition_thumbnail_key(league_id, size)
        urls_to_purge.append(f"{base_url}/{thumb_key}")

    success, error = await purge_cache_by_urls(urls_to_purge)

    if success:
        logger.info(f"CDN invalidated for competition {league_id}: {len(urls_to_purge)} paths")

    return success, error


async def invalidate_batch_cdn(team_ids: list[int]) -> dict[int, tuple[bool, Optional[str]]]:
    """Invalidate CDN cache for multiple teams.

    Args:
        team_ids: List of team IDs

    Returns:
        Dict mapping team_id to (success, error) tuple
    """
    results = {}

    for team_id in team_ids:
        success, error = await invalidate_team_logo_cdn(team_id)
        results[team_id] = (success, error)

    successful = sum(1 for s, _ in results.values() if s)
    logger.info(f"CDN batch invalidation: {successful}/{len(team_ids)} successful")

    return results
