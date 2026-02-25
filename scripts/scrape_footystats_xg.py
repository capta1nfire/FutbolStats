#!/usr/bin/env python3
"""
FootyStats xG Scraping — Bolivia + Chile + Colombia + Ecuador + Peru + Venezuela

4-phase CLI:
  --discover   Team discovery + auto-generate aliases
  --scrape     Scrape xG from match pages (resumable)
  --validate   Validation report (season completeness, xG coverage)
  --ingest     Ingest xG to database (dry-run default)

Flags:
  --league {bolivia,chile,colombia,ecuador,peru,venezuela}  Target league (required for --scrape, optional filter for --ingest)
  --season LABEL            Single season canary/filter (e.g. "2024", "2013/14")
  --resume                  Continue from checkpoint
  --no-dry-run              Actually write to DB (--ingest only)
  --all-ingest-leagues      Ingest all configured leagues (default ingest scope: peru+venezuela)

Dependencies: playwright, asyncpg (pip install playwright asyncpg && playwright install chromium)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
import sys
import time
import unicodedata
from collections import defaultdict
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from playwright.async_api import Page, TimeoutError as PlaywrightTimeoutError
from playwright.async_api import async_playwright

try:
    import asyncpg
except ImportError:
    asyncpg = None  # Only needed for --discover and --ingest


# ── Paths ──────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "footystats_xg"
ALIASES_PATH = PROJECT_ROOT / "data" / "footystats_team_aliases.json"
PROGRESS_PATH = DATA_DIR / "progress.json"


# ── League Configuration ───────────────────────────────────────────────────

LEAGUES: Dict[str, Dict[str, Any]] = {
    "bolivia": {
        "league_id": 344,
        "fixtures_url": "https://footystats.org/bolivia/lfpb/fixtures",
        "country_slug": "bolivia",
        "seasons": [
            "2013/14", "2014/15", "2015/16", "2016/17",
            "2018", "2019", "2020", "2021", "2022", "2023", "2024", "2025",
        ],
        "db_season_map": {
            "2013/14": None, "2014/15": None, "2015/16": None, "2016/17": None,
            "2018": None,
            "2019": 2019, "2020": 2020, "2021": 2021, "2022": 2022,
            "2023": 2023, "2024": 2024, "2025": 2025,
        },
    },
    "chile": {
        "league_id": 265,
        "fixtures_url": "https://footystats.org/chile/primera-division/fixtures",
        "country_slug": "chile",
        "seasons": [
            "2014", "2015", "2016", "2017", "2018",
            "2019", "2020", "2021", "2022", "2023", "2024", "2025", "2026",
        ],
        "db_season_map": {
            "2014": None, "2015": None, "2016": None, "2017": None, "2018": None,
            "2019": 2019, "2020": 2020, "2021": 2021, "2022": 2022,
            "2023": 2023, "2024": 2024, "2025": 2025, "2026": 2026,
        },
    },
    "colombia": {
        "league_id": 239,
        "fixtures_url": "https://footystats.org/colombia/categoria-primera-a/fixtures",
        "country_slug": "colombia",
        "seasons": [
            "2013", "2014", "2015", "2016", "2017", "2018",
            "2019", "2020", "2021", "2022", "2023", "2024", "2025", "2026",
        ],
        "db_season_map": {
            "2013": None, "2014": None, "2015": None, "2016": None,
            "2017": None, "2018": None,
            "2019": 2019, "2020": 2020, "2021": 2021, "2022": 2022,
            "2023": 2023, "2024": 2024, "2025": 2025, "2026": 2026,
        },
    },
    "ecuador": {
        "league_id": 242,
        "fixtures_url": "https://footystats.org/ecuador/primera-categoria-serie-a/fixtures",
        "country_slug": "ecuador",
        "seasons": [
            "2016", "2017", "2018", "2019", "2020",
            "2021", "2022", "2023", "2024", "2025", "2026",
        ],
        "db_season_map": {
            "2016": None, "2017": None, "2018": None,
            "2019": 2019, "2020": 2020, "2021": 2021, "2022": 2022,
            "2023": 2023, "2024": 2024, "2025": 2025, "2026": 2026,
        },
    },
    "peru": {
        "league_id": 281,
        "fixtures_url": "https://footystats.org/peru/primera-division/fixtures",
        "country_slug": "peru",
        "seasons": [
            "2013", "2014", "2015", "2016", "2017", "2018",
            "2019", "2020", "2021", "2022", "2023", "2024", "2025", "2026",
        ],
        "db_season_map": {
            "2013": None, "2014": None, "2015": None, "2016": None,
            "2017": None, "2018": None,
            "2019": 2019, "2020": 2020, "2021": 2021, "2022": 2022,
            "2023": 2023, "2024": 2024, "2025": 2025, "2026": 2026,
        },
    },
    "venezuela": {
        "league_id": 299,
        "fixtures_url": "https://footystats.org/venezuela/primera-division/fixtures",
        "country_slug": "venezuela",
        "seasons": [
            "2013/14", "2014/15", "2015", "2016", "2017", "2018",
            "2019", "2020", "2021", "2022", "2023", "2024", "2025", "2026",
        ],
        "db_season_map": {
            "2013/14": None, "2014/15": None, "2015": None, "2016": None,
            "2017": None, "2018": None,
            "2019": 2019, "2020": 2020, "2021": 2021, "2022": 2022,
            "2023": 2023, "2024": 2024, "2025": 2025, "2026": 2026,
        },
    },
}

# Expected matches per season (min, max). Seasons outside range → WARNING.
EXPECTED_MATCHES: Dict[str, Dict[str, Tuple[int, int]]] = {
    "bolivia": {
        "default": (80, 450),
        "2020": (80, 200),      # COVID-shortened
        "2013/14": (50, 400),   # Older, less data possible
        "2014/15": (50, 400),
        "2015/16": (50, 400),
        "2016/17": (50, 400),
    },
    "chile": {
        "default": (80, 400),
        "2026": (5, 100),       # En curso
    },
    "colombia": {
        "default": (200, 500),
        "2020": (150, 300),     # COVID-shortened
        "2026": (5, 100),       # En curso
    },
    "ecuador": {
        "default": (80, 350),
        "2026": (5, 100),       # En curso
        "2016": (50, 300),
        "2017": (50, 300),
        "2018": (50, 300),
    },
    "peru": {
        "default": (200, 400),
        "2020": (150, 300),     # COVID-shortened
        "2021": (150, 300),     # COVID-affected
        "2026": (5, 100),       # En curso
    },
    "venezuela": {
        "default": (150, 400),
        "2020": (100, 200),     # COVID-shortened
        "2026": (5, 100),       # En curso
    },
}


# ── Constants ──────────────────────────────────────────────────────────────

NAV_TIMEOUT = 30_000
ELEM_TIMEOUT = 15_000
DELAY_MATCH = (3.0, 5.0)
DELAY_SEASON = (5.0, 8.0)
MAX_CONSEC_ERRORS = 10
SAVE_EVERY = 25
FIXTURES_SCROLL_MAX = 40
FIXTURES_STABLE_PASSES = 3
MATCH_HARD_TIMEOUT_S = 150.0
MATCH_RENDER_WAIT_MS = 12_000
MATCH_RENDER_POLL_MS = 1_000
MATCH_MAX_ATTEMPTS = 3
MATCH_RETRY_SLEEP = (1.5, 3.0)
DEFAULT_INGEST_LEAGUES = ("peru", "venezuela")
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/131.0.0.0 Safari/537.36"
)


# ── Utilities ──────────────────────────────────────────────────────────────

_TRANSLITERATE = {
    "Ø": "O", "ø": "o", "Æ": "AE", "æ": "ae",
    "Ð": "D", "ð": "d", "Þ": "Th", "þ": "th",
    "Ł": "L", "ł": "l", "Đ": "D", "đ": "d",
    "ß": "ss", "İ": "I", "ı": "i",
}


def normalize_name(name: str) -> str:
    """Lowercase, remove accents, strip — works for teams and players."""
    if not name:
        return ""
    name = "".join(_TRANSLITERATE.get(c, c) for c in name)
    nfkd = unicodedata.normalize("NFKD", name)
    ascii_str = "".join(c for c in nfkd if not unicodedata.combining(c))
    return ascii_str.lower().strip()


def team_similarity(a: str, b: str) -> float:
    """Fuzzy similarity for team names. Returns 0.0-1.0."""
    na = normalize_name(a)
    nb = normalize_name(b)
    if not na or not nb:
        return 0.0
    if na == nb:
        return 1.0
    seq_ratio = SequenceMatcher(None, na, nb).ratio()
    # Token overlap (handles "CD Universidad" vs "Club Universidad")
    ta = set(na.split())
    tb = set(nb.split())
    if ta and tb:
        overlap = len(ta & tb)
        union = len(ta | tb)
        token_ratio = overlap / union
    else:
        token_ratio = 0.0
    return max(seq_ratio, token_ratio)


def season_file_label(label: str) -> str:
    """Convert season label to filesystem-safe string: '2013/14' -> '2013-14'."""
    return str(label).replace("/", "-")


def raw_path(league_key: str, season_label: str) -> Path:
    """Path for raw season JSON."""
    return DATA_DIR / f"{league_key}_{season_file_label(season_label)}_raw.json"


def expected_range(league_key: str, season_label: str) -> Tuple[int, int]:
    """Get (min, max) expected matches for a season."""
    league_exp = EXPECTED_MATCHES.get(league_key, {})
    return league_exp.get(str(season_label), league_exp.get("default", (50, 500)))


# ── Aliases ────────────────────────────────────────────────────────────────

def load_aliases() -> Dict[str, Dict[str, int]]:
    """Load aliases file. Returns {league_key: {name_lower: team_id}}."""
    if not ALIASES_PATH.exists():
        return {}
    with ALIASES_PATH.open() as f:
        data = json.load(f)
    result = {}
    for league_key in LEAGUES:
        section = data.get(league_key, {})
        mapping = {}
        for name, tid in section.items():
            if name.startswith("_"):
                continue
            mapping[name.lower().strip()] = tid
        result[league_key] = mapping
    return result


def save_aliases(aliases_data: Dict[str, Any]) -> None:
    """Save aliases file."""
    ALIASES_PATH.parent.mkdir(parents=True, exist_ok=True)
    with ALIASES_PATH.open("w", encoding="utf-8") as f:
        json.dump(aliases_data, f, ensure_ascii=False, indent=2)


def check_aliases_gate(league_key: str) -> bool:
    """GATE: check that aliases exist and are non-empty for the league."""
    aliases = load_aliases()
    league_aliases = aliases.get(league_key, {})
    if not league_aliases:
        print(f"GATE FAILED: No aliases for '{league_key}'. Run --discover first.")
        return False
    return True


# ── Progress ───────────────────────────────────────────────────────────────

def progress_path(league_key: Optional[str] = None) -> Path:
    """
    Resolve checkpoint path.

    Per-league files avoid race conditions when two scrapers run in parallel.
    """
    if league_key:
        return DATA_DIR / f"progress_{league_key}.json"
    return PROGRESS_PATH


def load_progress(league_key: Optional[str] = None) -> Dict[str, Any]:
    """Load progress checkpoint (per league when provided)."""
    p = progress_path(league_key)
    if p.exists():
        with p.open() as f:
            return json.load(f)
    return {}


def save_progress(progress: Dict[str, Any], league_key: Optional[str] = None) -> None:
    """Save progress checkpoint (per league when provided)."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    p = progress_path(league_key)
    with p.open("w") as f:
        json.dump(progress, f, indent=2)


def load_all_progress() -> Dict[str, Any]:
    """Merge global + per-league progress files for validation/reporting."""
    merged: Dict[str, Any] = {"scraped": {}}
    paths = [PROGRESS_PATH] + sorted(DATA_DIR.glob("progress_*.json"))
    for p in paths:
        if not p.exists():
            continue
        try:
            with p.open() as f:
                data = json.load(f)
            merged["scraped"].update(data.get("scraped", {}))
        except Exception:
            continue
    return merged


def get_scraped_ids(progress: Dict, league_key: str, season_label: str) -> Set[str]:
    """Get set of already-scraped match IDs for a season."""
    key = f"{league_key}_{season_file_label(season_label)}"
    return set(progress.get("scraped", {}).get(key, {}).get("ids", []))


def update_progress(progress: Dict, league_key: str, season_label: str,
                    scraped_id: str, xg_found: bool) -> None:
    """Mark a match as scraped in progress."""
    key = f"{league_key}_{season_file_label(season_label)}"
    if "scraped" not in progress:
        progress["scraped"] = {}
    if key not in progress["scraped"]:
        progress["scraped"][key] = {"ids": [], "xg_found": 0, "xg_missing": 0}
    entry = progress["scraped"][key]
    if scraped_id not in entry["ids"]:
        entry["ids"].append(scraped_id)
    if xg_found:
        entry["xg_found"] = entry.get("xg_found", 0) + 1
    else:
        entry["xg_missing"] = entry.get("xg_missing", 0) + 1


# ── Playwright Core ────────────────────────────────────────────────────────

async def select_season(page: Page, season_label: str) -> bool:
    """Click season dropdown and select the given label. Returns success."""
    # Check if the page already shows this season (default/current season)
    try:
        current_el = page.locator("div.season .drop-down-parent")
        current_text = await current_el.inner_text(timeout=3_000)
        if season_label in current_text:
            print(f"    Season '{season_label}' already selected (default)")
            return True
    except Exception:
        pass

    try:
        await page.click("div.season .drop-down-parent", timeout=5_000)
        await page.wait_for_timeout(500)
    except PlaywrightTimeoutError:
        print(f"  WARNING: Could not open season dropdown")
        return False

    try:
        sel = f"div.season .drop-down li >> text='{season_label}'"
        await page.click(sel, timeout=5_000)
        await page.wait_for_timeout(3_000)
        return True
    except PlaywrightTimeoutError:
        print(f"  WARNING: Season '{season_label}' not found in dropdown")
        return False


async def collect_match_urls(page: Page) -> List[Dict[str, Any]]:
    """Scroll fixtures page and collect all completed match URLs with scores."""
    # Scroll until match anchors stabilize (faster on short pages, safer on slow pages)
    stable_passes = 0
    last_count = -1
    for _ in range(FIXTURES_SCROLL_MAX):
        await page.mouse.wheel(0, 3_000)
        await page.wait_for_timeout(700)
        count = await page.evaluate(
            "() => document.querySelectorAll('a[href*=\"-h2h-stats#\"]').length"
        )
        if count == last_count:
            stable_passes += 1
            if stable_passes >= FIXTURES_STABLE_PASSES:
                break
        else:
            stable_passes = 0
            last_count = count

    matches = await page.evaluate("""
        () => {
            const results = [];
            const anchors = document.querySelectorAll('a[href*="-h2h-stats#"]');
            for (const a of anchors) {
                const href = a.getAttribute('href') || '';
                const text = (a.textContent || '').trim();

                const hashIdx = href.indexOf('#');
                if (hashIdx < 0) continue;
                const matchId = href.substring(hashIdx + 1);
                const fullUrl = new URL(href, window.location.origin).href;

                // Score from text: "0 - 0FT" or "1 - 2FT"
                const scoreMatch = text.match(/(\\d+)\\s*-\\s*(\\d+)/);
                const homeScore = scoreMatch ? parseInt(scoreMatch[1]) : null;
                const awayScore = scoreMatch ? parseInt(scoreMatch[2]) : null;

                // Team slugs from URL (generic: /{country}/{home}-vs-{away}-h2h-stats)
                const pathMatch = href.match(/\\/[a-z-]+\\/(.+)-vs-(.+)-h2h-stats/);
                const homeSlug = pathMatch ? pathMatch[1] : null;
                const awaySlug = pathMatch ? pathMatch[2] : null;

                results.push({
                    url: fullUrl,
                    match_id: matchId,
                    home_slug: homeSlug,
                    away_slug: awaySlug,
                    home_score: homeScore,
                    away_score: awayScore,
                });
            }
            return results;
        }
    """)

    # Dedup by match_id
    seen = set()
    unique = []
    for m in matches:
        if m["match_id"] not in seen:
            seen.add(m["match_id"])
            unique.append(m)
    return unique


# JS for extracting xG + team names from H2H match page
EXTRACT_XG_JS = """
() => {
    const result = {
        team1: null, team2: null, date: null,
        team1_xg: null, team2_xg: null,
    };

    const tables = document.querySelectorAll('table');
    for (const table of tables) {
        const rows = table.querySelectorAll('tr');
        for (const row of rows) {
            const cells = row.querySelectorAll('td, th');
            if (cells.length >= 3) {
                const label = (cells[0].textContent || '').trim();
                const val1 = (cells[1].textContent || '').trim();
                const val2 = (cells[2].textContent || '').trim();

                if (label === 'Data' && val1 && val2 && !result.team1) {
                    result.team1 = val1;
                    result.team2 = val2;
                }
                if (label === 'xG' || label === 'xg' || label === 'Expected Goals') {
                    const xg1 = parseFloat(val1);
                    const xg2 = parseFloat(val2);
                    if (!isNaN(xg1) && !isNaN(xg2)) {
                        result.team1_xg = xg1;
                        result.team2_xg = xg2;
                    }
                }
            }
        }
        if (result.team1) break;
    }

    const dateEl = document.querySelector('.match-date, time, [datetime]');
    if (dateEl) {
        result.date = (dateEl.getAttribute('datetime') || dateEl.textContent || '').trim();
    }

    // Fallback: regex search in page text
    if (result.team1_xg === null) {
        const allText = document.body.innerText || '';
        const xgMatch = allText.match(/xG[\\s\\n]+([\\d.]+)[\\s\\n]+([\\d.]+)/i);
        if (xgMatch) {
            result.team1_xg = parseFloat(xgMatch[1]);
            result.team2_xg = parseFloat(xgMatch[2]);
        }
    }
    return result;
}
"""


async def extract_match_xg_with_context(
    browser, match_info: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Open a fresh browser context, visit H2H page, extract xG. Anti-detection."""
    url = match_info["url"]
    mid = match_info["match_id"]

    def _is_blocked_title(title: str) -> bool:
        t = (title or "").lower()
        return any(tag in t for tag in ("captcha", "blocked", "403", "just a moment"))

    async def _extract_with_polling(page: Page) -> tuple[Optional[Dict[str, Any]], str]:
        """Poll extraction for slow-rendered historical pages."""
        deadline = time.monotonic() + (MATCH_RENDER_WAIT_MS / 1000.0)
        last_data: Dict[str, Any] = {}
        while time.monotonic() < deadline:
            try:
                data = await page.evaluate(EXTRACT_XG_JS)
            except Exception:
                data = {}
            last_data = data or {}
            if last_data.get("team1_xg") is not None:
                return last_data, "ok"
            try:
                title = await page.title()
            except Exception:
                title = ""
            if _is_blocked_title(title):
                return last_data, "blocked"
            await page.wait_for_timeout(MATCH_RENDER_POLL_MS)
        return last_data, "timeout"

    for attempt in range(1, MATCH_MAX_ATTEMPTS + 1):
        # Fresh context per attempt — safer against anti-bot and stale page state.
        context = await browser.new_context(
            user_agent=USER_AGENT,
            viewport={"width": random.randint(1280, 1440), "height": random.randint(800, 1100)},
        )
        page = await context.new_page()

        try:
            # domcontentloaded is more reliable than networkidle on slow legacy pages.
            await page.goto(url, wait_until="domcontentloaded", timeout=NAV_TIMEOUT)
            try:
                await page.wait_for_selector("table", timeout=ELEM_TIMEOUT)
            except PlaywrightTimeoutError:
                # Keep polling extraction even if selector wait fails.
                pass

            data, status = await _extract_with_polling(page)
            title = await page.title()
            await context.close()

            if status == "ok":
                return {
                    "match_id": mid,
                    "team1": data.get("team1") or match_info.get("home_slug", ""),
                    "team2": data.get("team2") or match_info.get("away_slug", ""),
                    "home_slug": match_info.get("home_slug"),
                    "away_slug": match_info.get("away_slug"),
                    "date": data.get("date"),
                    "home_score": match_info.get("home_score"),
                    "away_score": match_info.get("away_score"),
                    "team1_xg": data["team1_xg"],
                    "team2_xg": data["team2_xg"],
                    "source_url": url,
                }

            if status == "blocked" or _is_blocked_title(title):
                print(f"    BLOCKED: match {mid} (title: {title[:60]})")
                return None

            has_team = bool((data or {}).get("team1"))
            if has_team:
                # Page rendered enough to identify teams; xG genuinely missing.
                print(
                    f"    NO-XG: match {mid} "
                    f"(teams: {data.get('team1')} vs {data.get('team2')}, no xG)"
                )
                return {"_no_xg": True}

            # EMPTY render: retry before declaring hard failure.
            if attempt < MATCH_MAX_ATTEMPTS:
                print(
                    f"    RETRY: match {mid} attempt {attempt}/{MATCH_MAX_ATTEMPTS} "
                    f"(slow/empty render)"
                )
                await asyncio.sleep(random.uniform(*MATCH_RETRY_SLEEP) * attempt)
                continue

            print(f"    EMPTY: match {mid} (page did not render stats)")
            return None

        except PlaywrightTimeoutError:
            await context.close()
            if attempt < MATCH_MAX_ATTEMPTS:
                print(
                    f"    RETRY: timeout match {mid} attempt {attempt}/{MATCH_MAX_ATTEMPTS}"
                )
                await asyncio.sleep(random.uniform(*MATCH_RETRY_SLEEP) * attempt)
                continue
            print(f"    TIMEOUT: match {mid}")
            return None
        except Exception as e:
            await context.close()
            if attempt < MATCH_MAX_ATTEMPTS:
                print(
                    f"    RETRY: error match {mid} attempt {attempt}/{MATCH_MAX_ATTEMPTS} ({e})"
                )
                await asyncio.sleep(random.uniform(*MATCH_RETRY_SLEEP) * attempt)
                continue
            print(f"    ERROR navigating/extracting match {mid}: {e}")
            return None

    return None


# ── Phase 1: Discovery ────────────────────────────────────────────────────

def deslugify(slug: str) -> str:
    """Convert URL slug to approximate team name: 'club-bolivar' -> 'Club Bolivar'."""
    # Remove common suffixes added by FootyStats (sa, fc, etc. at the end)
    return slug.replace("-", " ").strip().title()


async def discover_team_slugs(page: Page, league_key: str) -> Set[str]:
    """Scrape fixtures pages to collect all unique team slugs. Fast, no H2H visits."""
    cfg = LEAGUES[league_key]
    print(f"\n  Discovering team slugs for {league_key}...")

    # Use 3 recent seasons to cover promoted/relegated teams
    recent_seasons = cfg["seasons"][-3:]
    all_slugs = set()

    for season_label in recent_seasons:
        # Re-navigate to fixtures page before each season (prevents stale dropdown)
        await page.goto(cfg["fixtures_url"], wait_until="domcontentloaded", timeout=NAV_TIMEOUT)
        await page.wait_for_timeout(3_000)

        print(f"    Season {season_label}...")
        ok = await select_season(page, season_label)
        if not ok:
            continue

        matches = await collect_match_urls(page)
        if not matches:
            print(f"    No matches found for {season_label}")
            continue

        for m in matches:
            if m.get("home_slug"):
                all_slugs.add(m["home_slug"])
            if m.get("away_slug"):
                all_slugs.add(m["away_slug"])

        print(f"    {len(matches)} matches, {len(all_slugs)} unique slugs cumulative")
        await asyncio.sleep(random.uniform(*DELAY_SEASON))

    return all_slugs


async def cmd_discover() -> None:
    """Phase 1: Discover team names and auto-generate aliases."""
    if asyncpg is None:
        print("ERROR: asyncpg required for --discover. Install: pip install asyncpg")
        sys.exit(1)

    db_url = os.environ.get("DATABASE_URL", "")
    if not db_url:
        print("ERROR: DATABASE_URL not set. Run: source .env")
        sys.exit(1)
    if db_url.startswith("postgres://"):
        db_url = db_url.replace("postgres://", "postgresql://", 1)

    # Load existing aliases to merge
    existing = {}
    if ALIASES_PATH.exists():
        with ALIASES_PATH.open() as f:
            existing = json.load(f)

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            user_agent=USER_AGENT,
            viewport={"width": 1366, "height": 2000},
        )
        page = await context.new_page()

        # Discover team slugs from FootyStats (fast: no H2H visits)
        fs_slugs = {}
        for league_key in LEAGUES:
            slugs = await discover_team_slugs(page, league_key)
            fs_slugs[league_key] = slugs
            print(f"  {league_key}: {len(slugs)} unique slugs discovered")

        await browser.close()

    # Connect to DB and fetch our teams
    conn = await asyncpg.connect(db_url)
    try:
        aliases_out = {"_meta": {
            "version": "1.0.0",
            "generated_at": datetime.utcnow().isoformat(),
            "note": "FootyStats team name -> internal team_id. Review before use.",
        }}

        for league_key, cfg in LEAGUES.items():
            lid = cfg["league_id"]
            rows = await conn.fetch("""
                SELECT DISTINCT t.id, t.name
                FROM teams t
                JOIN matches m ON (t.id = m.home_team_id OR t.id = m.away_team_id)
                WHERE m.league_id = $1 AND m.status = 'FT'
            """, lid)
            db_teams = [(r["id"], r["name"]) for r in rows]

            section = {"_league_id": lid}
            unresolved = []
            existing_section = existing.get(league_key, {})
            existing_lower = {k.lower().strip(): (k, v) for k, v in existing_section.items()
                              if not k.startswith("_")}

            for slug in sorted(fs_slugs.get(league_key, [])):
                approx_name = deslugify(slug)

                # Check if slug (as approx name) already in existing aliases
                if approx_name.lower().strip() in existing_lower:
                    orig_key, tid = existing_lower[approx_name.lower().strip()]
                    section[orig_key] = tid
                    continue

                # Auto-match: try both the slug-derived name and the raw slug
                best_score = 0.0
                best_tid = None
                best_db_name = ""
                for tid, db_name in db_teams:
                    # Match against de-slugified name
                    s = team_similarity(approx_name, db_name)
                    if s > best_score:
                        best_score = s
                        best_tid = tid
                        best_db_name = db_name

                if best_score >= 0.70:
                    # Use the slug as alias key (it's what we'll see in URLs)
                    section[approx_name] = best_tid
                    tag = "AUTO" if best_score >= 0.80 else "WEAK"
                    print(f"    {tag}: slug='{slug}' -> '{best_db_name}' (id={best_tid}, sim={best_score:.2f})")
                else:
                    unresolved.append((slug, approx_name, best_db_name, best_score))
                    print(f"    UNRESOLVED: slug='{slug}' ('{approx_name}') ~ '{best_db_name}' (sim={best_score:.2f})")

            # Preserve any manual entries from existing file
            for k, v in existing_section.items():
                if not k.startswith("_") and k not in section:
                    section[k] = v

            aliases_out[league_key] = section

            n_resolved = sum(1 for k in section if not k.startswith("_"))
            n_total = n_resolved + len(unresolved)
            status = "PASS" if not unresolved else "FAIL"
            print(f"\n  {league_key}: {n_resolved}/{n_total} resolved ({status})")
            if unresolved:
                print(f"  UNRESOLVED ({len(unresolved)}) — add manually to aliases:")
                for slug, approx, db_name, score in unresolved:
                    print(f"    slug='{slug}' ('{approx}') ~ '{db_name}' (sim={score:.2f})")
    finally:
        await conn.close()

    save_aliases(aliases_out)
    print(f"\nAliases saved to {ALIASES_PATH}")
    print("If there are UNRESOLVED teams, add them manually to the aliases file and re-run --discover.")


# ── Phase 2: Scraping ─────────────────────────────────────────────────────

async def scrape_season(browser, league_key: str, season_label: str,
                        progress: Dict, resume: bool) -> Tuple[List[Dict[str, Any]], bool]:
    """Scrape all matches for one season. Returns (results, stopped_early)."""
    cfg = LEAGUES[league_key]
    rp = raw_path(league_key, season_label)

    # Load existing results if resuming
    existing_results = []
    if resume and rp.exists():
        with rp.open() as f:
            existing_results = json.load(f)

    scraped_ids = get_scraped_ids(progress, league_key, season_label)

    print(f"\n  Season {season_label} (resume={resume}, already={len(scraped_ids)} scraped)")

    # Phase 1: Collect match URLs (using a dedicated context)
    ctx = await browser.new_context(
        user_agent=USER_AGENT,
        viewport={"width": 1366, "height": 2000},
    )
    page = await ctx.new_page()
    # Warm-up: load FootyStats home to set anti-bot cookies before fixtures
    await page.goto("https://footystats.org/", wait_until="domcontentloaded", timeout=NAV_TIMEOUT)
    await page.wait_for_timeout(2_000)
    await page.goto(cfg["fixtures_url"], wait_until="domcontentloaded", timeout=NAV_TIMEOUT)
    await page.wait_for_timeout(3_000)
    ok = await select_season(page, season_label)
    if not ok:
        print(f"  SKIP: Could not select season {season_label}")
        await ctx.close()
        return existing_results, False

    matches = await collect_match_urls(page)
    await ctx.close()  # Close fixtures context
    print(f"  {len(matches)} match URLs collected")

    # Season completeness check
    lo, hi = expected_range(league_key, season_label)
    if len(matches) < lo:
        print(f"  WARNING: Only {len(matches)} matches (expected {lo}-{hi}) — possibly incomplete season")
    elif len(matches) > hi:
        print(f"  WARNING: {len(matches)} matches exceeds expected range ({lo}-{hi})")

    # Phase 2: Visit each match with fresh context per request
    pending = [m for m in matches if m["match_id"] not in scraped_ids]
    print(f"  {len(pending)} matches to scrape ({len(scraped_ids)} already done)")

    results = list(existing_results)
    consec_errors = 0
    xg_found = 0
    xg_missing = 0

    for i, match_info in enumerate(pending):
        mid = match_info["match_id"]

        try:
            result = await asyncio.wait_for(
                extract_match_xg_with_context(browser, match_info),
                timeout=MATCH_HARD_TIMEOUT_S,  # Hard cap includes retries for slow pages
            )
        except asyncio.TimeoutError:
            print(f"    GLOBAL-TIMEOUT: match {mid} (>{int(MATCH_HARD_TIMEOUT_S)}s)")
            result = None

        if result and not result.get("_no_xg"):
            result["season"] = season_label
            results.append(result)
            xg_found += 1
            consec_errors = 0
            update_progress(progress, league_key, season_label, mid, True)
        elif result and result.get("_no_xg"):
            # Page rendered but no xG data — NOT a hard error
            xg_missing += 1
            consec_errors = 0  # Reset: page worked, just no xG
            update_progress(progress, league_key, season_label, mid, False)
        else:
            # EMPTY/BLOCKED/TIMEOUT — hard failure
            xg_missing += 1
            consec_errors += 1
            update_progress(progress, league_key, season_label, mid, False)

        # Progress log
        if (i + 1) % 10 == 0 or (i + 1) == len(pending):
            print(f"    [{i+1}/{len(pending)}] xG: {xg_found} found, {xg_missing} missing")

        # Checkpoint save
        if (i + 1) % SAVE_EVERY == 0:
            save_progress(progress, league_key)
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            with rp.open("w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

        # Stop on too many consecutive errors (possible block/captcha)
        if consec_errors >= MAX_CONSEC_ERRORS:
            print(f"  STOP: {MAX_CONSEC_ERRORS} consecutive errors — possible block. Saving progress.")
            break

        await asyncio.sleep(random.uniform(*DELAY_MATCH))

    # Save season results + progress
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with rp.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    save_progress(progress, league_key)

    stopped_early = consec_errors >= MAX_CONSEC_ERRORS
    total = xg_found + xg_missing
    ratio = (xg_found / total * 100) if total > 0 else 0
    print(f"  Season {season_label} done: {xg_found}/{total} with xG ({ratio:.0f}%), {len(results)} total results")

    if total > 0 and ratio < 50:
        print(f"  WARNING: xG coverage < 50% — possible data quality issue for this season")

    return results, stopped_early


async def cmd_scrape(league_key: str, season_filter: Optional[str], resume: bool,
                     all_seasons: bool = False) -> None:
    """Phase 2: Scrape xG from match pages."""
    if league_key not in LEAGUES:
        print(f"ERROR: Unknown league '{league_key}'. Choose: {list(LEAGUES.keys())}")
        sys.exit(1)

    cfg = LEAGUES[league_key]
    seasons = cfg["seasons"]
    if season_filter:
        if season_filter not in seasons:
            print(f"ERROR: Season '{season_filter}' not in {seasons}")
            sys.exit(1)
        seasons = [season_filter]
    elif not all_seasons:
        # Default: skip seasons with no DB match (pre-2019) — they have no xG anyway
        seasons = [s for s in seasons if cfg["db_season_map"].get(s) is not None]
        skipped = len(cfg["seasons"]) - len(seasons)
        if skipped:
            print(f"Skipping {skipped} pre-DB seasons (use --all-seasons to include them)")

    progress = load_progress(league_key) if resume else {}

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)

        print(f"Scraping {league_key}: {len(seasons)} seasons")
        stopped_early = False

        for season_label in seasons:
            results, season_stopped = await scrape_season(
                browser, league_key, season_label, progress, resume,
            )

            # Check if season stopped due to consecutive hard errors (block/captcha)
            if season_stopped:
                print(f"\n  STOPPED EARLY in season {season_label}. Use --resume to continue later.")
                stopped_early = True
                break

            await asyncio.sleep(random.uniform(*DELAY_SEASON))

        await browser.close()

    save_progress(progress, league_key)

    # Summary
    print(f"\n{'='*60}")
    print(f"SCRAPING SUMMARY: {league_key}")
    print(f"{'='*60}")
    for season_label in cfg["seasons"]:
        rp_file = raw_path(league_key, season_label)
        if rp_file.exists():
            with rp_file.open() as f:
                data = json.load(f)
            pkey = f"{league_key}_{season_file_label(season_label)}"
            entry = progress.get("scraped", {}).get(pkey, {})
            print(f"  {season_label}: {len(data)} results "
                  f"(xG: {entry.get('xg_found', '?')}, missing: {entry.get('xg_missing', '?')})")

    if stopped_early:
        print("\nRun with --resume to continue from where we stopped.")


# ── Phase 3: Validation ───────────────────────────────────────────────────

def cmd_validate() -> None:
    """Phase 3: Print validation report for all scraped data."""
    print(f"\n{'='*65}")
    print(f"  FootyStats xG Scraping — Validation Report")
    print(f"{'='*65}")

    aliases = load_aliases()
    progress = load_all_progress()

    for league_key, cfg in LEAGUES.items():
        lid = cfg["league_id"]
        league_aliases = aliases.get(league_key, {})
        n_aliases = len(league_aliases)
        print(f"\n  {league_key.upper()} (league_id={lid}, {n_aliases} team aliases)")
        print(f"  {'Season':<10} {'Matches':>8} {'xG Found':>10} {'xG%':>6} {'Range':>12} {'Status'}")
        print(f"  {'-'*10} {'-'*8} {'-'*10} {'-'*6} {'-'*12} {'-'*10}")

        total_matches = 0
        total_xg = 0
        warnings = []

        for season_label in cfg["seasons"]:
            rp_file = raw_path(league_key, season_label)
            db_season = cfg["db_season_map"].get(season_label)

            if not rp_file.exists():
                print(f"  {season_label:<10} {'—':>8} {'—':>10} {'—':>6} {'—':>12} NOT SCRAPED")
                continue

            with rp_file.open() as f:
                data = json.load(f)

            n = len(data)
            n_xg = sum(1 for d in data if d.get("team1_xg") is not None)
            pct = (n_xg / n * 100) if n > 0 else 0
            lo, hi = expected_range(league_key, season_label)
            range_str = f"({lo}-{hi})"

            total_matches += n
            total_xg += n_xg

            # Determine status
            status = "OK"
            db_note = f"DB:{db_season}" if db_season else "raw-only"

            if n < lo:
                status = f"LOW ({db_note})"
                warnings.append(f"{season_label}: {n} matches < {lo} minimum")
            elif pct < 50 and n > 0:
                status = f"LOW-xG ({db_note})"
                warnings.append(f"{season_label}: xG coverage {pct:.0f}% < 50%")
            else:
                status = f"OK ({db_note})"

            print(f"  {season_label:<10} {n:>8} {n_xg:>10} {pct:>5.0f}% {range_str:>12} {status}")

        total_pct = (total_xg / total_matches * 100) if total_matches > 0 else 0
        print(f"  {'TOTAL':<10} {total_matches:>8} {total_xg:>10} {total_pct:>5.0f}%")

        if warnings:
            print(f"\n  WARNINGS:")
            for w in warnings:
                print(f"    - {w}")

        # Check team name coverage in aliases
        if rp_file.exists():
            all_teams_in_data = set()
            for season_label in cfg["seasons"]:
                rf = raw_path(league_key, season_label)
                if rf.exists():
                    with rf.open() as f:
                        for d in json.load(f):
                            if d.get("team1"):
                                all_teams_in_data.add(d["team1"])
                            if d.get("team2"):
                                all_teams_in_data.add(d["team2"])

            def can_resolve(name):
                """Check if name resolves via exact or fuzzy match."""
                if name.lower().strip() in league_aliases:
                    return True
                # Fuzzy fallback
                for alias_name in league_aliases:
                    if team_similarity(name, alias_name) >= 0.80:
                        return True
                return False

            unresolved = [t for t in all_teams_in_data if not can_resolve(t)]
            if unresolved:
                print(f"\n  UNRESOLVED TEAMS ({len(unresolved)}) — add to aliases before --ingest:")
                for t in sorted(unresolved):
                    print(f"    - '{t}'")
            else:
                print(f"\n  Team resolution: {len(all_teams_in_data)}/{len(all_teams_in_data)} (100%)")


# ── Phase 4: Ingestion ────────────────────────────────────────────────────

async def cmd_ingest(
    dry_run: bool,
    league_filter: Optional[str] = None,
    season_filter: Optional[str] = None,
    all_ingest_leagues: bool = False,
) -> None:
    """Phase 4: Ingest scraped xG into matches.xg_home/xg_away."""
    if asyncpg is None:
        print("ERROR: asyncpg required for --ingest. Install: pip install asyncpg")
        sys.exit(1)

    db_url = os.environ.get("DATABASE_URL", "")
    if not db_url:
        print("ERROR: DATABASE_URL not set. Run: source .env")
        sys.exit(1)
    if db_url.startswith("postgres://"):
        db_url = db_url.replace("postgres://", "postgresql://", 1)

    aliases = load_aliases()

    if season_filter and not league_filter:
        print("ERROR: --season for --ingest requires --league (season labels can be ambiguous).")
        sys.exit(1)

    if league_filter:
        if league_filter not in LEAGUES:
            print(f"ERROR: Unknown league '{league_filter}'. Choose: {list(LEAGUES.keys())}")
            sys.exit(1)
        target_leagues = [league_filter]
    elif all_ingest_leagues:
        target_leagues = list(LEAGUES.keys())
    else:
        target_leagues = [lk for lk in DEFAULT_INGEST_LEAGUES if lk in LEAGUES]

    if not target_leagues:
        print("ERROR: No target leagues configured for ingest.")
        sys.exit(1)

    # Check aliases gate only for target leagues
    for league_key in target_leagues:
        if not check_aliases_gate(league_key):
            sys.exit(1)

    mode_str = "DRY-RUN" if dry_run else "LIVE WRITE"
    print(f"\n{'='*60}")
    print(f"  FootyStats xG Ingestion ({mode_str})")
    print(f"  Scope leagues: {', '.join(target_leagues)}")
    if season_filter:
        print(f"  Scope season: {season_filter}")
    print(f"{'='*60}")

    conn = await asyncpg.connect(db_url)
    try:
        for league_key in target_leagues:
            cfg = LEAGUES[league_key]
            lid = cfg["league_id"]
            league_aliases = aliases.get(league_key, {})

            stats = defaultdict(int)

            season_labels = cfg["seasons"]
            if season_filter:
                if season_filter not in season_labels:
                    print(
                        f"ERROR: Season '{season_filter}' not configured for league '{league_key}'. "
                        f"Available: {season_labels}"
                    )
                    sys.exit(1)
                season_labels = [season_filter]

            for season_label in season_labels:
                db_season = cfg["db_season_map"].get(season_label)
                if db_season is None:
                    continue  # Pre-2019, no DB match

                rp_file = raw_path(league_key, season_label)
                if not rp_file.exists():
                    continue

                with rp_file.open() as f:
                    data = json.load(f)

                for rec in data:
                    stats["total"] += 1

                    team1 = rec.get("team1", "")
                    team2 = rec.get("team2", "")

                    def resolve_team(name, slug):
                        """Try exact alias → slug deslugified → fuzzy."""
                        # 1. Exact alias match
                        tid = league_aliases.get(name.lower().strip())
                        if tid:
                            return tid
                        # 2. Slug deslugified
                        if slug:
                            approx = deslugify(slug)
                            tid = league_aliases.get(approx.lower().strip())
                            if tid:
                                return tid
                        # 3. Fuzzy match (threshold 0.80)
                        best_score = 0.0
                        best_tid = None
                        for alias_name, alias_tid in league_aliases.items():
                            s = team_similarity(name, alias_name)
                            if s > best_score:
                                best_score = s
                                best_tid = alias_tid
                        if best_score >= 0.80:
                            return best_tid
                        return None

                    tid1 = resolve_team(team1, rec.get("home_slug"))
                    tid2 = resolve_team(team2, rec.get("away_slug"))

                    if not tid1:
                        stats["unresolved_team"] += 1
                        continue
                    if not tid2:
                        stats["unresolved_team"] += 1
                        continue

                    date_str = rec.get("date", "")
                    match_date = None
                    if date_str:
                        for fmt in ("%Y-%m-%dT%H:%M", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
                            try:
                                match_date = datetime.strptime(date_str.strip()[:19], fmt)
                                break
                            except ValueError:
                                continue

                    if match_date:
                        # Date-based: either team order, ±2 days
                        rows = await conn.fetch("""
                            SELECT id, home_team_id, away_team_id, home_goals, away_goals,
                                   xg_home, xg_away
                            FROM matches
                            WHERE league_id = $1
                              AND (
                                (home_team_id = $2 AND away_team_id = $3)
                                OR (home_team_id = $3 AND away_team_id = $2)
                              )
                              AND ABS(EXTRACT(EPOCH FROM (date - $4::timestamp))) < 172800
                              AND status IN ('FT', 'AET', 'PEN')
                            ORDER BY ABS(EXTRACT(EPOCH FROM (date - $4::timestamp)))
                            LIMIT 2
                        """, lid, tid1, tid2, match_date)
                    else:
                        # Dateless fallback: season + teams + score
                        hs = rec.get("home_score")
                        aws = rec.get("away_score")
                        if hs is not None and aws is not None:
                            rows = await conn.fetch("""
                                SELECT id, home_team_id, away_team_id, home_goals, away_goals,
                                       xg_home, xg_away
                                FROM matches
                                WHERE league_id = $1
                                  AND (
                                    (home_team_id = $2 AND away_team_id = $3)
                                    OR (home_team_id = $3 AND away_team_id = $2)
                                  )
                                  AND season = $4
                                  AND status IN ('FT', 'AET', 'PEN')
                                ORDER BY date
                                LIMIT 3
                            """, lid, tid1, tid2, db_season)
                        else:
                            stats["bad_date"] += 1
                            continue

                    if not rows:
                        stats["no_db_match"] += 1
                        continue

                    # Tie-breaker: if >1 candidate, take closest date; if still ambiguous, skip
                    if len(rows) > 1 and not match_date:
                        stats["ambiguous"] += 1
                        continue

                    row = rows[0]

                    # Score validation
                    hs_fs = rec.get("home_score")
                    as_fs = rec.get("away_score")
                    db_hg = row["home_goals"]
                    db_ag = row["away_goals"]

                    if hs_fs is not None and as_fs is not None and db_hg is not None and db_ag is not None:
                        # FootyStats score is from fixtures page (home/away per URL order)
                        # DB match has home_team_id/away_team_id
                        # We need to check if the URL home matches DB home
                        # Since we searched with either order, we can't directly compare
                        # Instead: total goals must match at minimum
                        total_fs = int(hs_fs) + int(as_fs)
                        total_db = int(db_hg) + int(db_ag)
                        if total_fs != total_db:
                            stats["score_mismatch"] += 1
                            continue
                        # Exact check: try both orderings
                        exact = (int(hs_fs) == int(db_hg) and int(as_fs) == int(db_ag))
                        reverse = (int(hs_fs) == int(db_ag) and int(as_fs) == int(db_hg))
                        if not exact and not reverse:
                            stats["score_mismatch"] += 1
                            continue

                    # Skip if already has xG
                    if row["xg_home"] is not None or row["xg_away"] is not None:
                        stats["already_has_xg"] += 1
                        continue

                    # Assign xG: team1/team2 → home/away based on DB match
                    db_home_tid = row["home_team_id"]
                    if db_home_tid == tid1:
                        xg_home = rec["team1_xg"]
                        xg_away = rec["team2_xg"]
                    else:
                        xg_home = rec["team2_xg"]
                        xg_away = rec["team1_xg"]

                    if not dry_run:
                        await conn.execute("""
                            UPDATE matches
                            SET xg_home = $1, xg_away = $2, xg_source = 'footystats'
                            WHERE id = $3 AND xg_home IS NULL AND xg_away IS NULL
                        """, float(xg_home), float(xg_away), row["id"])

                    stats["updated"] += 1

            # Per-league report
            print(f"\n  {league_key.upper()} (league_id={lid}):")
            for k in ["total", "updated", "no_db_match", "score_mismatch",
                       "already_has_xg", "unresolved_team", "ambiguous", "bad_date"]:
                v = stats[k]
                if v > 0:
                    print(f"    {k}: {v}")

            if stats["total"] > 0:
                match_rate = stats["updated"] / stats["total"] * 100
                mismatch_rate = stats["score_mismatch"] / stats["total"] * 100
                print(f"    match_rate: {match_rate:.1f}%")
                print(f"    score_mismatch_rate: {mismatch_rate:.1f}%")

                if match_rate < 90:
                    print(f"    WARNING: match_rate < 90% — check aliases or date mapping")
                if mismatch_rate > 2:
                    print(f"    WARNING: score_mismatch > 2% — data quality concern")

    finally:
        await conn.close()

    if dry_run:
        print(f"\nDRY-RUN complete. No changes written. Use --ingest --no-dry-run to write.")


# ── CLI ────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="FootyStats xG Scraping — Bolivia + Chile + Ecuador + Peru + Venezuela",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 scripts/scrape_footystats_xg.py --discover
  python3 scripts/scrape_footystats_xg.py --scrape --league peru --season 2024
  python3 scripts/scrape_footystats_xg.py --scrape --league venezuela --resume
  python3 scripts/scrape_footystats_xg.py --validate
  python3 scripts/scrape_footystats_xg.py --ingest  # default scope: peru + venezuela
  python3 scripts/scrape_footystats_xg.py --ingest --league peru --no-dry-run
  python3 scripts/scrape_footystats_xg.py --ingest --all-ingest-leagues
        """,
    )
    parser.add_argument("--discover", action="store_true", help="Phase 1: Team discovery + aliases")
    parser.add_argument("--scrape", action="store_true", help="Phase 2: Scrape xG (resumable)")
    parser.add_argument("--validate", action="store_true", help="Phase 3: Validation report")
    parser.add_argument("--ingest", action="store_true", help="Phase 4: Ingest to DB (dry-run default)")
    parser.add_argument("--league", type=str, choices=list(LEAGUES.keys()),
                        help="Target league (required for --scrape, optional filter for --ingest)")
    parser.add_argument("--season", type=str, help="Single season for canary/filter (e.g. '2024', '2013/14')")
    parser.add_argument("--resume", action="store_true", help="Continue from checkpoint")
    parser.add_argument("--no-dry-run", action="store_true", help="Actually write to DB (--ingest only)")
    parser.add_argument("--all-seasons", action="store_true",
                        help="Include pre-DB seasons (db_season_map=None). Default: skip them.")
    parser.add_argument("--all-ingest-leagues", action="store_true",
                        help="For --ingest: process all configured leagues (default is peru+venezuela).")

    args = parser.parse_args()

    if not any([args.discover, args.scrape, args.validate, args.ingest]):
        parser.print_help()
        sys.exit(1)

    if args.discover:
        asyncio.run(cmd_discover())
    elif args.scrape:
        if not args.league:
            print("ERROR: --league required for --scrape")
            sys.exit(1)
        asyncio.run(cmd_scrape(args.league, args.season, args.resume, args.all_seasons))
    elif args.validate:
        cmd_validate()
    elif args.ingest:
        dry_run = not args.no_dry_run
        asyncio.run(cmd_ingest(
            dry_run=dry_run,
            league_filter=args.league,
            season_filter=args.season,
            all_ingest_leagues=args.all_ingest_leagues,
        ))


if __name__ == "__main__":
    main()
