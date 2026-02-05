#!/usr/bin/env python3
"""
Extract social media handles from official team websites.

Strategy per Kimi recommendation + ABE P0 fixes:
1. Use website URL from team_wikidata_enrichment (P856)
2. Validate URL (SSRF protection)
3. Check robots.txt (best-effort)
4. Fetch homepage HTML (rate limit: 1 req/2s, max 3MB)
5. Parse JSON-LD sameAs first (most reliable per ABE)
6. Fall back to footer/header links
7. Validate handles (avoid generic like "share", "login")
8. Update social_handles in team_wikidata_enrichment (only NULLs)

P0 ABE Fixes:
- SSRF protection: Block private IPs, localhost
- Robots.txt: Best-effort check before fetch
- JSON-LD sameAs: Priority parser for reliability
- Intent URLs: Support twitter.com/intent/follow?screen_name=
- SQL: Use to_jsonb() for string casting
- Rate limit: Real 2s delay per request

Usage:
    python scripts/extract_social_from_websites.py --dry-run
    python scripts/extract_social_from_websites.py --apply --batch-size 50
"""

import argparse
import asyncio
import ipaddress
import json
import logging
import os
import re
import sys
from datetime import datetime
from typing import Any, Optional
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# P0 ABE: NO hardcodear credenciales - requerir env var explícito
DATABASE_URL = os.environ.get("DATABASE_URL")
if not DATABASE_URL:
    logger.error("DATABASE_URL environment variable is required")
    logger.error("Set it with: export DATABASE_URL='postgresql+asyncpg://...'")
    sys.exit(1)

# =============================================================================
# Configuration
# =============================================================================

# Rate limit: 1 request every 2 seconds (conservative per ABE)
RATE_LIMIT_DELAY = 2.0

# P1 ABE: Payload limit
MAX_HTML_SIZE = 3 * 1024 * 1024  # 3MB

# P0 ABE: SSRF Protection
BLOCKED_HOSTS = {"localhost", "127.0.0.1", "0.0.0.0"}
BLOCKED_NETWORKS = [
    ipaddress.ip_network("127.0.0.0/8"),  # Loopback
    ipaddress.ip_network("10.0.0.0/8"),  # Private A
    ipaddress.ip_network("172.16.0.0/12"),  # Private B
    ipaddress.ip_network("192.168.0.0/16"),  # Private C
    ipaddress.ip_network("169.254.0.0/16"),  # Link-local
]

# P0 ABE: Intent URLs FIRST (more specific), then direct profile
# Intent pattern: twitter.com/intent/follow?screen_name=handle
TWITTER_INTENT_PATTERN = re.compile(
    r"(?:twitter\.com|x\.com)/intent/(?:follow|user)\?(?:[^&]*&)*screen_name=([A-Za-z0-9_]{1,15})",
    re.IGNORECASE,
)
# Direct profile pattern: twitter.com/handle or x.com/handle
TWITTER_PROFILE_PATTERN = re.compile(
    r"(?:twitter\.com|x\.com)/([A-Za-z0-9_]{1,15})(?:\?|/|$)",
    re.IGNORECASE,
)

INSTAGRAM_PATTERN = re.compile(
    r"instagram\.com/([A-Za-z0-9_.]{1,30})(?:\?|/|$)",
    re.IGNORECASE,
)

# Handles to ignore (generic/spam)
INVALID_HANDLES = {
    "share",
    "login",
    "signup",
    "home",
    "intent",
    "sharer",
    "hashtag",
    "search",
    "explore",
    "p",
    "reel",
    "stories",
    "accounts",
    "oauth",
    "help",
    "about",
    "privacy",
    "terms",
    "contact",
}

# =============================================================================
# P0 ABE: SSRF Protection
# =============================================================================


def is_safe_url(url: str) -> bool:
    """
    Validate URL is safe (no SSRF).

    P0 ABE: Block private IPs, localhost, non-http schemes.
    """
    try:
        parsed = urlparse(url)

        # Must be http or https
        if parsed.scheme not in ("http", "https"):
            return False

        # Check blocked hostnames
        host = parsed.hostname or ""
        if host.lower() in BLOCKED_HOSTS:
            return False

        # Check if IP address in private ranges
        try:
            ip = ipaddress.ip_address(host)
            for network in BLOCKED_NETWORKS:
                if ip in network:
                    return False
        except ValueError:
            pass  # Not an IP, it's a hostname - OK

        return True
    except Exception:
        return False


# =============================================================================
# P0 ABE: Robots.txt Check
# =============================================================================


async def check_robots_allowed(
    url: str,
    client: httpx.AsyncClient,
) -> bool:
    """
    Best-effort robots.txt check.

    Returns True if allowed or unknown.
    P0 ABE: Simple check for "Disallow: /" under User-agent: *
    """
    try:
        parsed = urlparse(url)
        robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"

        response = await client.get(
            robots_url,
            timeout=5.0,
            follow_redirects=True,
        )

        if response.status_code != 200:
            return True  # No robots.txt = allowed

        text = response.text.lower()

        # Simple check: if "disallow: /" for User-agent: *, skip
        # This is best-effort, not a full parser
        if "user-agent: *" in text:
            lines = text.split("\n")
            in_star_block = False
            for line in lines:
                line = line.strip()
                if "user-agent: *" in line:
                    in_star_block = True
                elif line.startswith("user-agent:"):
                    in_star_block = False
                elif in_star_block and line == "disallow: /":
                    return False  # Blocks everything

        return True
    except Exception:
        return True  # On error, assume allowed (fail-open)


# =============================================================================
# P0 ABE: JSON-LD sameAs Parser (Most Reliable)
# =============================================================================


def extract_social_from_jsonld(html: str) -> dict[str, Optional[str]]:
    """
    Extract social handles from JSON-LD sameAs.

    P0 ABE: Most reliable source - structured data that clubs explicitly declare.
    """
    result: dict[str, Optional[str]] = {"twitter": None, "instagram": None}

    soup = BeautifulSoup(html, "html.parser")

    for script in soup.find_all("script", type="application/ld+json"):
        try:
            if not script.string:
                continue

            data = json.loads(script.string)

            # Handle both single object and array
            items = data if isinstance(data, list) else [data]

            for item in items:
                if not isinstance(item, dict):
                    continue

                same_as = item.get("sameAs", [])
                if isinstance(same_as, str):
                    same_as = [same_as]

                for url in same_as:
                    if not isinstance(url, str):
                        continue

                    if not result["twitter"]:
                        # P0 ABE: Try intent pattern first (more specific)
                        match = TWITTER_INTENT_PATTERN.search(url)
                        if not match:
                            match = TWITTER_PROFILE_PATTERN.search(url)
                        if match:
                            handle = match.group(1)
                            if handle and handle.lower() not in INVALID_HANDLES:
                                result["twitter"] = handle

                    if not result["instagram"]:
                        match = INSTAGRAM_PATTERN.search(url)
                        if match:
                            handle = match.group(1)
                            if handle and handle.lower() not in INVALID_HANDLES:
                                result["instagram"] = handle

        except (json.JSONDecodeError, TypeError):
            continue

    return result


# =============================================================================
# Twitter/Instagram Extractors (Fallback)
# =============================================================================


def extract_twitter_handle(html: str) -> Optional[str]:
    """
    Extract Twitter/X handle from HTML.

    Priority:
    1. Meta tag twitter:site
    2. Links in footer/header/nav
    3. Links anywhere in document
    """
    soup = BeautifulSoup(html, "html.parser")

    # Strategy 1: Look for meta tag twitter:site
    meta_twitter = soup.find("meta", attrs={"name": "twitter:site"})
    if meta_twitter and meta_twitter.get("content"):
        handle = str(meta_twitter["content"]).lstrip("@")
        if handle.lower() not in INVALID_HANDLES:
            return handle

    # Strategy 2: Look for links in footer/header/nav
    for area in ["footer", "header", "nav"]:
        container = soup.find(area) or soup.find(class_=re.compile(area, re.I))
        if container:
            for a in container.find_all("a", href=True):
                href = str(a["href"])
                if "twitter.com" in href or "x.com" in href:
                    # P0 ABE: Try intent pattern first
                    match = TWITTER_INTENT_PATTERN.search(href)
                    if not match:
                        match = TWITTER_PROFILE_PATTERN.search(href)
                    if match:
                        handle = match.group(1)
                        if handle and handle.lower() not in INVALID_HANDLES:
                            return handle

    # Strategy 3: Search entire document (last resort)
    for a in soup.find_all("a", href=True):
        href = str(a["href"])
        if "twitter.com" in href or "x.com" in href:
            # P0 ABE: Try intent pattern first
            match = TWITTER_INTENT_PATTERN.search(href)
            if not match:
                match = TWITTER_PROFILE_PATTERN.search(href)
            if match:
                handle = match.group(1)
                if handle and handle.lower() not in INVALID_HANDLES:
                    return handle

    return None


def extract_instagram_handle(html: str) -> Optional[str]:
    """
    Extract Instagram handle from HTML.

    Priority:
    1. Links in footer/header/nav
    2. Links anywhere in document
    """
    soup = BeautifulSoup(html, "html.parser")

    # Strategy 1: Look for links in footer/header/nav
    for area in ["footer", "header", "nav"]:
        container = soup.find(area) or soup.find(class_=re.compile(area, re.I))
        if container:
            for a in container.find_all("a", href=True):
                href = str(a["href"])
                if "instagram.com" in href:
                    match = INSTAGRAM_PATTERN.search(href)
                    if match:
                        handle = match.group(1)
                        if handle and handle.lower() not in INVALID_HANDLES:
                            return handle

    # Strategy 2: Search entire document
    for a in soup.find_all("a", href=True):
        href = str(a["href"])
        if "instagram.com" in href:
            match = INSTAGRAM_PATTERN.search(href)
            if match:
                handle = match.group(1)
                if handle and handle.lower() not in INVALID_HANDLES:
                    return handle

    return None


# =============================================================================
# Website Fetcher
# =============================================================================


async def fetch_website_html(
    url: str,
    client: httpx.AsyncClient,
    metrics: dict[str, int],
) -> Optional[str]:
    """
    Fetch website HTML with proper error handling.

    P0 ABE:
    - SSRF check before fetch
    - Robots.txt check (best-effort)
    - Size limit (3MB)
    """
    # Normalize URL
    if not url.startswith(("http://", "https://")):
        url = f"https://{url}"

    # P0: SSRF check
    if not is_safe_url(url):
        logger.debug(f"  SSRF blocked: {url}")
        metrics["blocked_ssrf"] += 1
        return None

    # P0: Robots check (best-effort)
    if not await check_robots_allowed(url, client):
        logger.debug(f"  Robots blocked: {url}")
        metrics["blocked_robots"] += 1
        return None

    try:
        response = await client.get(
            url,
            headers={
                "User-Agent": "FutbolStats/1.0 (contact@futbolstats.app; social-enrichment)",
                "Accept": "text/html",
                "Accept-Language": "en-US,en;q=0.9,es;q=0.8",
            },
            timeout=15.0,
            follow_redirects=True,
        )

        if response.status_code != 200:
            metrics["fetch_errors"] += 1
            return None

        # Only process HTML responses
        content_type = response.headers.get("content-type", "")
        if "text/html" not in content_type:
            metrics["fetch_errors"] += 1
            return None

        # P1: Size limit
        if len(response.content) > MAX_HTML_SIZE:
            logger.debug(f"  Too large ({len(response.content)} bytes): {url}")
            metrics["too_large"] += 1
            return None

        metrics["fetched_ok"] += 1
        return response.text

    except httpx.TimeoutException:
        metrics["timeouts"] += 1
        return None
    except Exception as e:
        logger.debug(f"  Fetch error: {url} - {e}")
        metrics["fetch_errors"] += 1
        return None


# =============================================================================
# Database Operations
# =============================================================================


async def get_candidates(session: AsyncSession, batch_size: int) -> list[tuple]:
    """
    Get teams with website but missing Twitter or Instagram.
    """
    result = await session.execute(
        text("""
            SELECT
                twe.team_id,
                t.name,
                twe.website,
                twe.social_handles->>'twitter' AS twitter,
                twe.social_handles->>'instagram' AS instagram
            FROM team_wikidata_enrichment twe
            JOIN teams t ON t.id = twe.team_id
            WHERE twe.website IS NOT NULL
              AND (
                twe.social_handles->>'twitter' IS NULL
                OR twe.social_handles->>'instagram' IS NULL
              )
            ORDER BY t.name
            LIMIT :batch_size
        """),
        {"batch_size": batch_size},
    )
    return result.fetchall()


async def update_social_handles(
    session: AsyncSession,
    team_id: int,
    new_twitter: Optional[str],
    new_instagram: Optional[str],
) -> bool:
    """
    Update social_handles - only fill NULL fields.

    P0 ABE: Use to_jsonb() for proper string casting.
    P0 ABE: Proper rollback on errors to avoid transaction abort.
    """
    if not new_twitter and not new_instagram:
        return False

    # Update enrichment_source to reflect website contribution
    source_update = """
        enrichment_source = CASE
            WHEN enrichment_source = 'wikidata' THEN 'wikidata+website'
            WHEN enrichment_source = 'wikipedia' THEN 'wikipedia+website'
            WHEN enrichment_source = 'wikidata+wikipedia' THEN 'wikidata+wikipedia+website'
            ELSE COALESCE(enrichment_source, 'website')
        END
    """

    # P0 ABE: Execute all updates in a single transaction with rollback on error
    try:
        if new_twitter:
            await session.execute(
                text("""
                    UPDATE team_wikidata_enrichment
                    SET social_handles = jsonb_set(
                        COALESCE(social_handles, '{}'::jsonb),
                        '{twitter}',
                        to_jsonb(:new_twitter::text),
                        true
                    )
                    WHERE team_id = :team_id
                      AND (social_handles->>'twitter' IS NULL)
                """),
                {"team_id": team_id, "new_twitter": new_twitter},
            )

        if new_instagram:
            await session.execute(
                text("""
                    UPDATE team_wikidata_enrichment
                    SET social_handles = jsonb_set(
                        COALESCE(social_handles, '{}'::jsonb),
                        '{instagram}',
                        to_jsonb(:new_instagram::text),
                        true
                    )
                    WHERE team_id = :team_id
                      AND (social_handles->>'instagram' IS NULL)
                """),
                {"team_id": team_id, "new_instagram": new_instagram},
            )

        # Update source
        await session.execute(
            text(f"""
                UPDATE team_wikidata_enrichment
                SET {source_update}
                WHERE team_id = :team_id
            """),
            {"team_id": team_id},
        )

        return True
    except Exception as e:
        # P0 ABE: Rollback on error to prevent transaction abort
        await session.rollback()
        logger.error(f"  DB error for team {team_id}: {e}")
        return False


# =============================================================================
# Main Logic
# =============================================================================


async def main(dry_run: bool = True, batch_size: int = 50, output_path: Optional[str] = None):
    # P0 ABE: Enforce max batch size of 100 per plan
    MAX_BATCH_SIZE = 100
    if batch_size > MAX_BATCH_SIZE:
        logger.warning(f"Batch size {batch_size} exceeds maximum {MAX_BATCH_SIZE}, capping")
        batch_size = MAX_BATCH_SIZE

    logger.info("=" * 60)
    logger.info("Website Social Media Extractor")
    logger.info("=" * 60)
    logger.info(f"Mode: {'DRY RUN' if dry_run else 'APPLY'}")
    logger.info(f"Batch size: {batch_size}")
    logger.info("")

    # Metrics for runlog
    metrics = {
        "total_candidates": 0,
        "fetched_ok": 0,
        "blocked_ssrf": 0,
        "blocked_robots": 0,
        "fetch_errors": 0,
        "timeouts": 0,
        "too_large": 0,
        "extracted_twitter": 0,
        "extracted_instagram": 0,
        "updated": 0,
        "errors": 0,
    }

    # Connect to DB
    engine = create_async_engine(DATABASE_URL, echo=False)
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async with async_session() as session:
        # Get candidates
        candidates = await get_candidates(session, batch_size)
        metrics["total_candidates"] = len(candidates)

        logger.info(f"Found {len(candidates)} candidates with website but missing social handles")
        logger.info("")

        if not candidates:
            logger.info("No candidates to process")
            await engine.dispose()
            return metrics

        extractions = []

        async with httpx.AsyncClient() as client:
            for row in candidates:
                team_id = row.team_id
                team_name = row.name
                website = row.website
                existing_twitter = row.twitter
                existing_instagram = row.instagram

                logger.info(f"Processing: {team_name}")

                # Rate limiting
                await asyncio.sleep(RATE_LIMIT_DELAY)

                # Fetch HTML
                html = await fetch_website_html(website, client, metrics)
                if not html:
                    continue

                # Extract social handles
                # P0 ABE: JSON-LD first (most reliable)
                jsonld_result = extract_social_from_jsonld(html)

                new_twitter = None
                new_instagram = None

                # Twitter: JSON-LD > meta > links
                if not existing_twitter:
                    new_twitter = jsonld_result.get("twitter")
                    if not new_twitter:
                        new_twitter = extract_twitter_handle(html)

                # Instagram: JSON-LD > links
                if not existing_instagram:
                    new_instagram = jsonld_result.get("instagram")
                    if not new_instagram:
                        new_instagram = extract_instagram_handle(html)

                if new_twitter:
                    metrics["extracted_twitter"] += 1
                    logger.info(f"  Twitter: @{new_twitter}")

                if new_instagram:
                    metrics["extracted_instagram"] += 1
                    logger.info(f"  Instagram: @{new_instagram}")

                if new_twitter or new_instagram:
                    extractions.append({
                        "team_id": team_id,
                        "team_name": team_name,
                        "website": website,
                        "new_twitter": new_twitter,
                        "new_instagram": new_instagram,
                    })

        logger.info("")
        logger.info(f"Extractions: {len(extractions)}")

        if dry_run:
            logger.info("")
            logger.info("DRY RUN - No changes applied")
            logger.info("Extractions that would be applied:")
            for e in extractions[:20]:
                tw = f"@{e['new_twitter']}" if e["new_twitter"] else "-"
                ig = f"@{e['new_instagram']}" if e["new_instagram"] else "-"
                logger.info(f"  {e['team_id']:5} | {e['team_name'][:25]:25} | TW: {tw:20} | IG: {ig}")
            if len(extractions) > 20:
                logger.info(f"  ... and {len(extractions) - 20} more")
        else:
            # Apply updates
            logger.info("")
            logger.info("Applying updates...")

            for e in extractions:
                success = await update_social_handles(
                    session,
                    e["team_id"],
                    e["new_twitter"],
                    e["new_instagram"],
                )
                if success:
                    metrics["updated"] += 1
                    logger.info(f"  Updated: {e['team_name']}")
                else:
                    metrics["errors"] += 1

            await session.commit()
            logger.info(f"  Updated: {metrics['updated']}")
            logger.info(f"  Errors: {metrics['errors']}")

    await engine.dispose()

    # P1 ABE: Emit runlog (now with output_path wired)
    emit_runlog(metrics, output_path)

    return metrics


def emit_runlog(metrics: dict[str, int], output_path: Optional[str] = None):
    """
    Emit execution summary for auditing.

    P1 ABE: Log summary and optionally write to JSONL.
    """
    summary = {
        "timestamp": datetime.utcnow().isoformat(),
        **metrics,
    }

    logger.info("")
    logger.info("=" * 60)
    logger.info("Run Summary")
    logger.info("=" * 60)
    logger.info(f"Candidates: {metrics['total_candidates']}")
    logger.info(f"Fetched OK: {metrics['fetched_ok']}")
    logger.info(f"Blocked (SSRF): {metrics['blocked_ssrf']}")
    logger.info(f"Blocked (robots): {metrics['blocked_robots']}")
    logger.info(f"Fetch errors: {metrics['fetch_errors']}")
    logger.info(f"Timeouts: {metrics['timeouts']}")
    logger.info(f"Too large: {metrics['too_large']}")
    logger.info(f"Extracted Twitter: {metrics['extracted_twitter']}")
    logger.info(f"Extracted Instagram: {metrics['extracted_instagram']}")
    logger.info(f"Updated: {metrics['updated']}")
    logger.info(f"Errors: {metrics['errors']}")

    if output_path:
        with open(output_path, "a") as f:
            f.write(json.dumps(summary) + "\n")
        logger.info(f"Runlog saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract social handles from team websites")
    parser.add_argument("--dry-run", action="store_true", default=True, help="Preview changes without applying")
    parser.add_argument("--apply", action="store_true", help="Apply changes to database")
    parser.add_argument("--batch-size", type=int, default=50, help="Number of teams to process")
    parser.add_argument("--output", type=str, help="Path for runlog JSONL output")

    args = parser.parse_args()

    dry_run = not args.apply
    asyncio.run(main(dry_run=dry_run, batch_size=args.batch_size, output_path=args.output))
