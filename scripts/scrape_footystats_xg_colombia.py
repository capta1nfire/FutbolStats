#!/usr/bin/env python3
"""
Scrape per-match xG from FootyStats for Colombia Categoria Primera A (2020-2024).

Strategy:
  Phase 1: Navigate fixtures page, click each season, collect all FT match URLs
  Phase 2: Visit each match H2H page (with #match_id fragment), extract xG

Output: JSON array compatible with the xG ingestion pipeline.
"""

from __future__ import annotations

import asyncio
import json
import random
import re
import sys
from pathlib import Path
from typing import Any, Optional

from playwright.async_api import Page, TimeoutError as PlaywrightTimeoutError
from playwright.async_api import async_playwright

# ── Configuration ──────────────────────────────────────────────────────────
FIXTURES_URL = "https://footystats.org/colombia/categoria-primera-a/fixtures"
SEASONS = [2020, 2021, 2022, 2023, 2024]

NAVIGATION_TIMEOUT_MS = 30_000
ELEMENT_TIMEOUT_MS = 15_000
DELAY_BETWEEN_MATCHES = (1.5, 3.0)  # seconds
DELAY_BETWEEN_SEASONS = (3.0, 5.0)

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/131.0.0.0 Safari/537.36"
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = PROJECT_ROOT / "data" / "footystats_xg" / "colombia_xg_raw.json"
CHECKPOINT_PATH = PROJECT_ROOT / "data" / "footystats_xg" / "checkpoint.json"


# ── Phase 1: Collect match URLs from fixtures ──────────────────────────────

async def collect_match_urls_for_season(page: Page, year: int) -> list[dict[str, Any]]:
    """Click season selector and collect all FT match URLs for that year."""

    # Click the season dropdown to open it
    try:
        await page.click("div.season .drop-down-parent", timeout=5_000)
        await page.wait_for_timeout(500)
    except PlaywrightTimeoutError:
        print(f"  WARNING: Could not open season dropdown for {year}")
        return []

    # Click the year option
    try:
        # The dropdown items contain just the year text
        year_selector = f"div.season .drop-down li >> text='{year}'"
        await page.click(year_selector, timeout=5_000)
        await page.wait_for_timeout(3_000)  # wait for data to load
    except PlaywrightTimeoutError:
        print(f"  WARNING: Could not select year {year}")
        return []

    # Scroll down to load all matches (lazy loading)
    for _ in range(10):
        await page.mouse.wheel(0, 3_000)
        await page.wait_for_timeout(800)

    # Extract all completed match links with their scores
    matches = await page.evaluate("""
        () => {
            const results = [];
            // Find all anchor tags with FT scores and match IDs
            const anchors = document.querySelectorAll('a[href*="-h2h-stats#"]');
            for (const a of anchors) {
                const href = a.getAttribute('href') || '';
                const text = (a.textContent || '').trim();

                // Extract match ID from fragment
                const hashIdx = href.indexOf('#');
                if (hashIdx < 0) continue;
                const matchId = href.substring(hashIdx + 1);

                // Build full URL
                const fullUrl = new URL(href, window.location.origin).href;

                // Extract score from text like "0 - 0FT" or "1 - 2FT"
                const scoreMatch = text.match(/(\\d+)\\s*-\\s*(\\d+)/);
                const homeScore = scoreMatch ? parseInt(scoreMatch[1]) : null;
                const awayScore = scoreMatch ? parseInt(scoreMatch[2]) : null;

                // Extract team slugs from URL path
                const pathMatch = href.match(/\\/colombia\\/(.+)-vs-(.+)-h2h-stats/);
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

    print(f"  Season {year}: {len(unique)} completed matches found")
    return unique


async def collect_all_match_urls(page: Page) -> dict[int, list[dict[str, Any]]]:
    """Collect match URLs for all target seasons."""
    print("Phase 1: Collecting match URLs from fixtures page...")
    await page.goto(FIXTURES_URL, wait_until="domcontentloaded", timeout=NAVIGATION_TIMEOUT_MS)
    await page.wait_for_timeout(3_000)

    all_seasons = {}
    for year in SEASONS:
        matches = await collect_match_urls_for_season(page, year)
        all_seasons[year] = matches
        if year != SEASONS[-1]:
            await asyncio.sleep(random.uniform(*DELAY_BETWEEN_SEASONS))

    total = sum(len(v) for v in all_seasons.values())
    print(f"Phase 1 complete: {total} matches across {len(SEASONS)} seasons\n")
    return all_seasons


# ── Phase 2: Extract xG from individual match pages ───────────────────────

async def extract_match_xg(page: Page, match_info: dict[str, Any]) -> Optional[dict[str, Any]]:
    """Visit a match H2H page and extract xG + team names + date."""
    url = match_info["url"]
    match_id = match_info["match_id"]

    try:
        await page.goto(url, wait_until="domcontentloaded", timeout=NAVIGATION_TIMEOUT_MS)
        await page.wait_for_timeout(2_000)
    except PlaywrightTimeoutError:
        print(f"    TIMEOUT navigating to match {match_id}")
        return None

    # Extract match data including xG, team names, date
    data = await page.evaluate("""
        () => {
            const result = {
                team1: null,
                team2: null,
                date: null,
                team1_xg: null,
                team2_xg: null,
            };

            // Extract team names + xG from stats table.
            // The table has a header row: | Data | Team A | Team B |
            // and a stats row:           | xG   | 1.34   | 1.60   |
            // NOTE: team order in table != URL order. We store as team1/team2
            // and let the ingestion script resolve home/away via our DB.
            const tables = document.querySelectorAll('table');
            for (const table of tables) {
                const rows = table.querySelectorAll('tr');
                for (const row of rows) {
                    const cells = row.querySelectorAll('td, th');
                    if (cells.length >= 3) {
                        const label = (cells[0].textContent || '').trim();
                        const val1 = (cells[1].textContent || '').trim();
                        const val2 = (cells[2].textContent || '').trim();

                        // Header row with team names
                        if (label === 'Data' && val1 && val2 && !result.team1) {
                            result.team1 = val1;
                            result.team2 = val2;
                        }

                        // xG row
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
                // Stop after first table that has a Data header (match stats table)
                if (result.team1) break;
            }

            // Extract date from the page
            const dateEl = document.querySelector('.match-date, time, [datetime]');
            if (dateEl) {
                result.date = (dateEl.getAttribute('datetime') || dateEl.textContent || '').trim();
            }

            // Fallback xG: search all text nodes
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
    """)

    if data["team1_xg"] is None:
        return None

    return {
        "match_id": match_id,
        "team1": data["team1"] or match_info.get("home_slug", ""),
        "team2": data["team2"] or match_info.get("away_slug", ""),
        "date": data["date"],
        "home_score": match_info.get("home_score"),
        "away_score": match_info.get("away_score"),
        "team1_xg": data["team1_xg"],
        "team2_xg": data["team2_xg"],
        "source_url": url,
    }


# ── Checkpoint logic ──────────────────────────────────────────────────────

def load_checkpoint() -> set[str]:
    """Load set of already-scraped match IDs."""
    if CHECKPOINT_PATH.exists():
        with CHECKPOINT_PATH.open() as f:
            return set(json.load(f))
    return set()


def save_checkpoint(scraped_ids: set[str]) -> None:
    """Persist scraped match IDs."""
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CHECKPOINT_PATH.open("w") as f:
        json.dump(sorted(scraped_ids), f)


def load_existing_results() -> list[dict[str, Any]]:
    """Load previously scraped results for incremental append."""
    if OUTPUT_PATH.exists():
        with OUTPUT_PATH.open() as f:
            return json.load(f)
    return []


def save_results(results: list[dict[str, Any]]) -> None:
    """Save results to JSON."""
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


# ── Main ──────────────────────────────────────────────────────────────────

async def main() -> None:
    scraped_ids = load_checkpoint()
    all_results = load_existing_results()
    initial_count = len(all_results)

    if scraped_ids:
        print(f"Resuming: {len(scraped_ids)} matches already scraped, {initial_count} results on disk\n")

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            user_agent=USER_AGENT,
            viewport={"width": 1366, "height": 2000},
        )
        page = await context.new_page()

        # Phase 1: collect match URLs
        seasons_data = await collect_all_match_urls(page)

        # Phase 2: visit each match and extract xG
        print("Phase 2: Extracting xG from individual matches...")
        total_matches = sum(len(v) for v in seasons_data.values())
        processed = 0
        xg_found = 0
        xg_missing = 0

        for year in SEASONS:
            matches = seasons_data.get(year, [])
            print(f"\n  Season {year}: {len(matches)} matches to process")

            for i, match_info in enumerate(matches):
                mid = match_info["match_id"]
                processed += 1

                if mid in scraped_ids:
                    continue

                result = await extract_match_xg(page, match_info)

                if result:
                    result["season"] = year
                    all_results.append(result)
                    xg_found += 1
                    scraped_ids.add(mid)
                else:
                    xg_missing += 1
                    scraped_ids.add(mid)  # mark as attempted

                # Progress log every 10 matches
                if (i + 1) % 10 == 0 or (i + 1) == len(matches):
                    print(
                        f"    {year} [{i+1}/{len(matches)}] "
                        f"xG found: {xg_found}, missing: {xg_missing} "
                        f"(total: {processed}/{total_matches})"
                    )

                # Save checkpoint every 25 matches
                if processed % 25 == 0:
                    save_checkpoint(scraped_ids)
                    save_results(all_results)

                await asyncio.sleep(random.uniform(*DELAY_BETWEEN_MATCHES))

        await browser.close()

    # Final save
    save_checkpoint(scraped_ids)
    save_results(all_results)

    new_results = len(all_results) - initial_count
    print(f"\n{'='*60}")
    print(f"DONE: {new_results} new results ({len(all_results)} total)")
    print(f"xG found: {xg_found}, xG missing: {xg_missing}")
    print(f"Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
