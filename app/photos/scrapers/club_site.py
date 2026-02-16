"""Club Website Player Photo Scraper (P1 source — best quality).

Fetches player headshots from official club websites.
Each team has a config entry in data/club_photo_sources.json with:
  - squad_url: URL to plantel/equipo page
  - strategy: extraction method (wordpress_grid, individual_pages, etc.)

Typical quality: 80-95 (official HQ studio photos).
"""

import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Optional
from urllib.parse import urljoin

import httpx
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

# Firecrawl API for WAF-protected sites (fallback when httpx gets 403)
FIRECRAWL_API_KEY = os.environ.get("FIRECRAWL_API_KEY", "")
FIRECRAWL_API_URL = "https://api.firecrawl.dev/v1/scrape"

# Path to config (relative to project root)
CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
    "data",
    "club_photo_sources.json",
)

# Reasonable headers to avoid blocks
DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "es-CO,es;q=0.9,en;q=0.5",
}


@dataclass
class ClubSitePlayer:
    """A player entry extracted from a club website."""

    name: str
    image_url: str
    jersey_number: Optional[int] = None
    position: Optional[str] = None
    page_url: Optional[str] = None  # Individual player page if available


@dataclass
class ClubSiteScrapeResult:
    """Result of scraping a club website's squad page."""

    team_external_id: int
    team_name: str
    squad_url: str
    players: list = field(default_factory=list)  # list of ClubSitePlayer
    error: Optional[str] = None
    quality_cap: int = 90


@dataclass
class ScrapedPhoto:
    """Result of downloading a specific player photo from club site."""

    image_bytes: Optional[bytes] = None
    source: str = "club_site"
    content_type: Optional[str] = None
    error: Optional[str] = None
    quality_cap: int = 90
    source_url: Optional[str] = None


def load_club_config():
    """Load club photo sources configuration.

    Returns:
        dict keyed by team external_id (as string), or empty dict if file missing.
    """
    if not os.path.exists(CONFIG_PATH):
        logger.warning(f"Club config not found: {CONFIG_PATH}")
        return {}
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _normalize_position(raw):
    """Normalize position text from club websites to standard codes."""
    if not raw:
        return None
    raw = raw.lower().strip()
    mapping = {
        "portero": "Goalkeeper",
        "arquero": "Goalkeeper",
        "defensa": "Defender",
        "defensor": "Defender",
        "lateral": "Defender",
        "zaguero": "Defender",
        "mediocampista": "Midfielder",
        "volante": "Midfielder",
        "centrocampista": "Midfielder",
        "delantero": "Attacker",
        "atacante": "Attacker",
        "extremo": "Attacker",
        "punta": "Attacker",
        "director tecnico": None,
        "dt": None,
        "entrenador": None,
        "preparador": None,
        "cuerpo tecnico": None,
    }
    for key, val in mapping.items():
        if key in raw:
            return val
    return None


def _clean_player_name(name):
    """Clean player name extracted from HTML.

    Strips leading jersey numbers (e.g. '3 Jan Angulo' -> 'Jan Angulo'),
    trailing numbers ('López2' -> 'López'), CMS suffixes, ALL CAPS to title case.
    """
    if not name:
        return name
    # Strip leading number + space (SportsPress pattern: "3 Jan Angulo")
    name = re.sub(r'^\d{1,3}\s+', '', name.strip())
    # Strip trailing digit glued to name ("López2")
    name = re.sub(r'(\w)\d$', r'\1', name)
    # Strip common CMS suffixes from alt text
    for suffix in ["equipo profesional", "plantel profesional", "primer equipo",
                    "jugador profesional", "jugador", "dim", "dim oficial"]:
        name = re.sub(r'\s+' + re.escape(suffix) + r'$', '', name, flags=re.IGNORECASE)
    # Convert ALL CAPS to Title Case (e.g. "DIEGO NOVOA" -> "Diego Novoa")
    if name == name.upper() and len(name) > 3:
        name = name.title()
    return name.strip()


def _extract_jersey_number(text):
    """Extract jersey number from text like '#10', 'No. 10', '10'."""
    if not text:
        return None
    m = re.search(r'(?:#|no\.?\s*|dorsal\s*)(\d{1,2})', text.lower())
    if m:
        return int(m.group(1))
    # Standalone 1-2 digit number
    m = re.match(r'^\s*(\d{1,2})\s*$', text.strip())
    if m:
        return int(m.group(1))
    return None


# ---------------------------------------------------------------------------
# Strategy: wordpress_grid
# ---------------------------------------------------------------------------
# Used for sites like Atlético Nacional where the squad page has a grid of
# player cards, each with an <img> tag and player info in nearby text.

def _parse_wordpress_grid(html, base_url):
    """Extract players from a WordPress-style grid layout.

    Looks for image elements with wp-content/uploads paths,
    then finds associated player names, jersey numbers, and positions.
    """
    soup = BeautifulSoup(html, "html.parser")
    players = []

    # Strategy 1: Look for article/div containers with player data
    # Common patterns: .elementor-team-member, .player-card, etc.
    # Generic: find all img tags with wp-content/uploads and reasonable sizes

    # First, try to find structured player containers
    # Look for common CMS patterns
    containers = soup.select(
        ".team-member, .player-card, .player-item, .player, "
        ".elementor-team-member, .wp-block-group, "
        "article.staff, .sp-template-player-gallery .gallery-item, "
        "div.sc_blogger_item.sp_player, "
        "article.eael-grid-post, "
        ".flipbox-container"
    )

    if containers:
        for container in containers:
            player = _extract_from_container(container, base_url)
            if player:
                players.append(player)

    if players:
        return players

    # Build a set of "staff section" elements so we can skip non-player images
    # (e.g. "CUERPO TÉCNICO" section headings)
    staff_section_elements = set()
    for heading in soup.find_all(["h1", "h2", "h3"]):
        text = heading.get_text(strip=True).lower()
        if any(kw in text for kw in ["cuerpo t", "staff", "director", "preparador"]):
            # Mark all siblings after this heading as staff
            parent = heading.find_parent(["section", "div"])
            if parent:
                staff_section_elements.add(id(parent))

    # Strategy 2: Elementor pattern — image widget followed by sibling section/div with h2s
    # Nacional uses: <div widget_type="image.default"> as sibling of <section> containing h2 name/position/number
    # Also handles: generic img in wp-content/uploads with nearby headings
    imgs = soup.find_all("img")
    photo_imgs = []
    for img in imgs:
        src = img.get("src") or ""
        # If src is a data: URI (lazy-load placeholder), prefer data-src
        if src.startswith("data:"):
            src = ""
        src = src or img.get("data-src") or img.get("data-lazy-src") or ""
        if not src:
            continue
        # Accept wp-content/uploads, i0.wp.com CDN, sites/default/files (Drupal), cdn.shopify.com
        if not any(p in src for p in ["wp-content/uploads", "sites/default/files", "i0.wp.com", "cdn.shopify.com"]):
            continue
        if any(skip in src.lower() for skip in [
            "logo", "icon", "banner", "sponsor", "escudo", "favicon", "cropped-",
            "betsson", "patrocinador", "nike", "postobon", "sportline",
            "cedimed", "jfk", "amazon", "stanley", "finandina",
            "oto-caps", "descargar", "snaau", "tvs", "aress", "mundo-unico",
            "trofeo", "palmares", "promo-front",
        ]):
            continue
        photo_imgs.append(img)

    for img in photo_imgs:
        # Skip images inside staff/cuerpo técnico sections
        if staff_section_elements:
            in_staff = False
            for ancestor in img.parents:
                if id(ancestor) in staff_section_elements:
                    in_staff = True
                    break
            if in_staff:
                continue

        src = img.get("src") or ""
        if src.startswith("data:"):
            src = ""
        src = src or img.get("data-src") or img.get("data-lazy-src") or ""
        image_url = urljoin(base_url, src)

        # Try alt text first
        name = _find_name_near_image(img)

        # Strategy 2a: Elementor widget sibling pattern
        # Walk up to the elementor-widget div, then check next sibling for headings
        if not name:
            widget_div = img.find_parent("div", attrs={"data-widget_type": True})
            if widget_div:
                next_sib = widget_div.find_next_sibling(["section", "div"])
                if next_sib:
                    headings = next_sib.find_all(["h2", "h3", "h4", "h5", "h6"])
                    for h in headings:
                        text = h.get_text(strip=True)
                        if not text or len(text) < 3 or len(text) > 60:
                            continue
                        if _normalize_position(text) is not None:
                            continue
                        if re.match(r'^\d{1,2}$', text.strip()):
                            continue
                        name = text
                        break

        # Strategy 2b: Generic — look in parent container headings
        if not name:
            parent = img.find_parent(["div", "article", "li", "figure"])
            if parent and parent.parent:
                headings = parent.parent.find_all(["h2", "h3", "h4", "h5", "h6"], recursive=False)
                for h in headings:
                    text = h.get_text(strip=True)
                    if text and 3 <= len(text) <= 60 and _normalize_position(text) is None and not re.match(r'^\d{1,2}$', text.strip()):
                        name = text
                        break

        if not name:
            continue

        # Clean name (strip leading jersey numbers, trailing digits)
        name = _clean_player_name(name)
        if not name or len(name) < 3:
            continue

        # Find jersey number and position from the same sibling section/div
        jersey = None
        position = None
        widget_div = img.find_parent("div", attrs={"data-widget_type": True})
        if widget_div:
            next_sib = widget_div.find_next_sibling(["section", "div"])
            if next_sib:
                for h in next_sib.find_all(["h2", "h3", "h4", "h5", "h6"]):
                    text = h.get_text(strip=True)
                    if not text:
                        continue
                    if jersey is None and re.match(r'^\d{1,2}$', text.strip()):
                        jersey = int(text.strip())
                        continue
                    if position is None:
                        p = _normalize_position(text)
                        if p is not None:
                            position = p
        else:
            # Generic: check parent context
            parent = img.find_parent(["div", "article", "li", "figure", "a"])
            if parent:
                text = parent.get_text(" ", strip=True)
                jersey = _extract_jersey_number(text)
                position = _normalize_position(text)

        players.append(ClubSitePlayer(
            name=name,
            image_url=image_url,
            jersey_number=jersey,
            position=position,
        ))

    return players


def _extract_from_container(container, base_url):
    """Extract player info from a structured container element."""
    # Find image (may be nested in sub-divs)
    img = container.find("img")
    if not img:
        return None

    src = img.get("src") or ""
    # If src is a data: URI (lazy-load placeholder), prefer data-src
    if src.startswith("data:"):
        src = ""
    src = src or img.get("data-src") or img.get("data-lazy-src") or ""
    if not src:
        return None
    # Skip non-photo images
    if any(skip in src.lower() for skip in ["logo", "icon", "banner", "sponsor", "escudo", "placeholder", "cropped-"]):
        return None

    image_url = urljoin(base_url, src)

    # Find name — try headings, then link text, then alt text
    name_el = container.find(["h2", "h3", "h4", "h5", "h6"])
    if not name_el:
        for tag in ["strong", "b"]:
            name_el = container.find(tag)
            if name_el:
                break

    name = name_el.get_text(strip=True) if name_el else None
    if not name:
        name = _find_name_near_image(img)
    if not name:
        return None

    name = _clean_player_name(name)
    # Skip if name looks like a section header
    if not name or len(name) < 3 or len(name) > 60:
        return None

    # Position — try structured spans first (e.g. span.item-role), then generic text
    position = None
    role_el = container.find("span", class_=lambda c: c and "role" in c)
    if role_el:
        position = _normalize_position(role_el.get_text(strip=True))
    if not position:
        text = container.get_text(" ", strip=True)
        position = _normalize_position(text)

    # Jersey number
    text = container.get_text(" ", strip=True)
    jersey = _extract_jersey_number(text)

    # Individual page link
    link = container.find("a", href=True)
    page_url = urljoin(base_url, link["href"]) if link else None

    return ClubSitePlayer(
        name=name,
        image_url=image_url,
        jersey_number=jersey,
        position=position,
        page_url=page_url,
    )


def _find_name_near_image(img):
    """Find a player name near an image element.

    Checks: alt text, title, sibling/parent headings.
    """
    # 1. Alt text (very common for player photos)
    alt = img.get("alt", "").strip()
    if alt and len(alt) > 2 and not any(x in alt.lower() for x in ["logo", "banner", "escudo", "imagen"]):
        return alt

    # 2. Title attribute
    title = img.get("title", "").strip()
    if title and len(title) > 2:
        return title

    # 3. Nearest heading (sibling or parent's child)
    parent = img.find_parent(["div", "figure", "article", "li", "a"])
    if parent:
        for tag in ["h2", "h3", "h4", "h5", "h6", "strong", "span.name", "p.name"]:
            heading = parent.find(tag)
            if heading:
                text = heading.get_text(strip=True)
                if text and len(text) > 2 and len(text) < 60:
                    return text

    return None


# ---------------------------------------------------------------------------
# Strategy: individual_pages
# ---------------------------------------------------------------------------
# Used for sites like Millonarios where images on the grid page are lazy-loaded,
# but individual player pages (/staff/{slug}) have the real image in og:image.

def _parse_individual_page(html, base_url):
    """Extract player photo from an individual player page.

    Uses og:image meta tag as the primary source.
    """
    soup = BeautifulSoup(html, "html.parser")

    # og:image is the most reliable source
    og_image = soup.find("meta", property="og:image")
    image_url = og_image["content"] if og_image and og_image.get("content") else None

    # og:title or page title for name
    og_title = soup.find("meta", property="og:title")
    name = og_title["content"].strip() if og_title and og_title.get("content") else None
    if not name:
        title_tag = soup.find("title")
        name = title_tag.get_text(strip=True) if title_tag else None

    if not image_url or not name:
        return None

    # Clean name (remove site name suffix like " - Millonarios FC")
    name = re.split(r'\s*[-–|]\s*', name)[0].strip()

    return ClubSitePlayer(
        name=name,
        image_url=urljoin(base_url, image_url),
        page_url=base_url,
    )


# ---------------------------------------------------------------------------
# Firecrawl fallback (WAF bypass)
# ---------------------------------------------------------------------------

async def _fetch_html_firecrawl(url, timeout=30.0):
    """Fetch HTML via Firecrawl API when httpx is blocked by WAF.

    Returns HTML string or None if Firecrawl is not configured or fails.
    """
    if not FIRECRAWL_API_KEY:
        return None
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(
                FIRECRAWL_API_URL,
                json={"url": url, "formats": ["html"], "onlyMainContent": True},
                headers={
                    "Authorization": f"Bearer {FIRECRAWL_API_KEY}",
                    "Content-Type": "application/json",
                },
            )
            resp.raise_for_status()
            data = resp.json()
            html = data.get("data", {}).get("html", "")
            if html:
                logger.info(f"  Firecrawl fallback OK for {url} ({len(html)} chars)")
                return html
            logger.warning(f"  Firecrawl returned no HTML for {url}")
            return None
    except Exception as e:
        logger.warning(f"  Firecrawl fallback failed for {url}: {e}")
        return None


# ---------------------------------------------------------------------------
# Main scrape function
# ---------------------------------------------------------------------------

async def scrape_club_squad(
    team_external_id,
    team_name=None,
    config=None,
    timeout=20.0,
):
    """Scrape squad photos from a club's official website.

    Args:
        team_external_id: API-Football team external ID
        team_name: Team name (for logging)
        config: Pre-loaded club config dict (or loads from file)
        timeout: HTTP timeout

    Returns:
        ClubSiteScrapeResult with list of players found
    """
    if config is None:
        config = load_club_config()

    key = str(team_external_id)
    if key not in config:
        return ClubSiteScrapeResult(
            team_external_id=team_external_id,
            team_name=team_name or "?",
            squad_url="",
            error=f"No club config for external_id={team_external_id}",
        )

    team_cfg = config[key]
    squad_url = team_cfg.get("squad_url", "")
    strategy = team_cfg.get("strategy", "wordpress_grid")
    quality_cap = team_cfg.get("quality_cap", 90)
    custom_headers = team_cfg.get("headers", {})

    if not squad_url:
        return ClubSiteScrapeResult(
            team_external_id=team_external_id,
            team_name=team_name or team_cfg.get("team_name", "?"),
            squad_url="",
            error="No squad_url configured",
        )

    headers = {**DEFAULT_HEADERS, **custom_headers}

    html = None
    used_firecrawl = False

    try:
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            resp = await client.get(squad_url, headers=headers)
            resp.raise_for_status()
            html = resp.text
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 403 and FIRECRAWL_API_KEY:
            logger.info(f"  httpx 403 for {squad_url}, trying Firecrawl fallback...")
            html = await _fetch_html_firecrawl(squad_url)
            used_firecrawl = True
        if not html:
            return ClubSiteScrapeResult(
                team_external_id=team_external_id,
                team_name=team_name or team_cfg.get("team_name", "?"),
                squad_url=squad_url,
                error=f"HTTP error: {e}",
            )
    except Exception as e:
        return ClubSiteScrapeResult(
            team_external_id=team_external_id,
            team_name=team_name or team_cfg.get("team_name", "?"),
            squad_url=squad_url,
            error=f"HTTP error: {e}",
        )

    try:
        if strategy == "wordpress_grid":
            players = _parse_wordpress_grid(html, squad_url)
        elif strategy == "individual_pages":
            soup = BeautifulSoup(html, "html.parser")
            player_links = []
            link_pattern = team_cfg.get("player_link_pattern", "/staff/")
            for a in soup.find_all("a", href=True):
                href = a["href"]
                if link_pattern in href:
                    full_url = urljoin(squad_url, href)
                    if full_url not in player_links:
                        player_links.append(full_url)

            players = []
            async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
                for purl in player_links:
                    try:
                        resp2 = await client.get(purl, headers=headers)
                        resp2.raise_for_status()
                        p = _parse_individual_page(resp2.text, purl)
                        if p:
                            players.append(p)
                    except Exception as e2:
                        logger.debug(f"  Failed to fetch {purl}: {e2}")
        else:
            players = _parse_wordpress_grid(html, squad_url)
    except Exception as e:
        return ClubSiteScrapeResult(
            team_external_id=team_external_id,
            team_name=team_name or team_cfg.get("team_name", "?"),
            squad_url=squad_url,
            error=f"Parse error: {e}",
        )

    return ClubSiteScrapeResult(
        team_external_id=team_external_id,
        team_name=team_name or team_cfg.get("team_name", "?"),
        squad_url=squad_url,
        players=players,
        quality_cap=quality_cap,
    )


async def fetch_club_photo(
    image_url,
    timeout=15.0,
    quality_cap=90,
):
    """Download a specific player photo from a club website.

    Args:
        image_url: Direct URL to the player photo
        timeout: HTTP timeout
        quality_cap: Quality cap for this source

    Returns:
        ScrapedPhoto with image bytes or error
    """
    try:
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            resp = await client.get(image_url, headers=DEFAULT_HEADERS)

            if resp.status_code == 404:
                return ScrapedPhoto(error="Photo not found (404)", source_url=image_url)

            resp.raise_for_status()

            content_type = resp.headers.get("content-type", "")
            if "image" not in content_type and "octet-stream" not in content_type:
                return ScrapedPhoto(
                    error=f"Not an image: {content_type}",
                    source_url=image_url,
                )

            image_bytes = resp.content
            if len(image_bytes) < 1000:
                return ScrapedPhoto(
                    error=f"Image too small ({len(image_bytes)} bytes)",
                    source_url=image_url,
                )

            return ScrapedPhoto(
                image_bytes=image_bytes,
                content_type=content_type,
                quality_cap=quality_cap,
                source_url=image_url,
            )

    except httpx.HTTPStatusError as e:
        return ScrapedPhoto(error=f"HTTP {e.response.status_code}", source_url=image_url)
    except Exception as e:
        return ScrapedPhoto(error=str(e), source_url=image_url)
