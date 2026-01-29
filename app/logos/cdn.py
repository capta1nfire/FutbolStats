"""CDN Cache Invalidation for Logo Regenerations.

NOTE: With immutable URL versioning (v1, v2, etc.), cache invalidation
is NO LONGER REQUIRED. Each regeneration produces a new URL with
incremented revision, so browsers/CDN automatically fetch the new asset.

Strategy: Cache-Control: public, max-age=31536000, immutable
- Old URLs remain valid (previous versions)
- New URLs are fetched fresh (cache miss → origin)

The purge functions below are kept as NO-OPs for backwards compatibility.
"""

import logging
from typing import Optional

import httpx

from app.logos.config import get_logos_settings

logger = logging.getLogger(__name__)
# Settings kept for potential future use, but currently unused
# since invalidation functions are NO-OPs with immutable URLs
_logos_settings = get_logos_settings()

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

    NOTE: This is now a NO-OP. With immutable URL versioning, each
    regeneration produces a new URL (e.g., _v1 → _v2), so cache
    invalidation is unnecessary. The CDN serves old URLs indefinitely
    (which is fine) and fetches new URLs fresh on first request.

    Args:
        team_id: Internal team ID
        variants: List of variants to purge (ignored)

    Returns:
        Tuple of (True, None) - always succeeds as NO-OP
    """
    logger.debug(
        f"CDN invalidation skipped for team {team_id}: "
        "using immutable URL versioning strategy"
    )
    return True, None


async def invalidate_competition_logo_cdn(league_id: int) -> tuple[bool, Optional[str]]:
    """Invalidate CDN cache for competition logo.

    NOTE: This is now a NO-OP. With immutable URL versioning, each
    regeneration produces a new URL (e.g., _v1 → _v2), so cache
    invalidation is unnecessary.

    Args:
        league_id: League ID

    Returns:
        Tuple of (True, None) - always succeeds as NO-OP
    """
    logger.debug(
        f"CDN invalidation skipped for competition {league_id}: "
        "using immutable URL versioning strategy"
    )
    return True, None


async def invalidate_batch_cdn(team_ids: list[int]) -> dict[int, tuple[bool, Optional[str]]]:
    """Invalidate CDN cache for multiple teams.

    NOTE: This is now a NO-OP. With immutable URL versioning,
    cache invalidation is unnecessary.

    Args:
        team_ids: List of team IDs

    Returns:
        Dict mapping team_id to (True, None) for all teams
    """
    logger.debug(
        f"CDN batch invalidation skipped for {len(team_ids)} teams: "
        "using immutable URL versioning strategy"
    )
    return {team_id: (True, None) for team_id in team_ids}
