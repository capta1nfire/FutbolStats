#!/usr/bin/env python3
"""
Scrape 1X2 odds from OddsPortal for any configured league.

Usage:
    python scripts/scrape_oddsportal_league.py --league paraguay --years 2019-2026
    python scripts/scrape_oddsportal_league.py --league uruguay --years 2019-2026
    python scripts/scrape_oddsportal_league.py --league paraguay --years 2023-2025

Output format is compatible with the existing ingest_oddsportal.py pipeline.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
from pathlib import Path
from typing import Any

from playwright.async_api import TimeoutError as PlaywrightTimeoutError
from playwright.async_api import async_playwright

# ── League URL configurations ────────────────────────────────────────────
# OddsPortal renames leagues over the years. Map each year to its URL slug.

LEAGUE_CONFIGS: dict[str, dict[str, Any]] = {
    "paraguay": {
        "country": "paraguay",
        "seasons": {
            2026: "copa-de-primera",
            2025: "copa-de-primera-2025",
            2024: "primera-division-2024",
            2023: "primera-division-2023",
            2022: "primera-division-2022",
            2021: "primera-division-2021",
            2020: "primera-division-2020",
            2019: "primera-division-2019",
        },
    },
    "uruguay": {
        "country": "uruguay",
        "seasons": {
            2026: "liga-auf-uruguaya",
            2025: "liga-auf-uruguaya-2025",
            2024: "primera-division-2024",
            2023: "primera-division-2023",
            2022: "primera-division-2022",
            2021: "primera-division-2021",
            2020: "primera-division-2020",
            2019: "primera-division-2019",
        },
    },
    "colombia": {
        "country": "colombia",
        "seasons": {
            2026: "primera-a",
            2025: "primera-a-2025",
            2024: "primera-a-2024",
            2023: "primera-a-2023",
            2022: "primera-a-2022",
            2021: "primera-a-2021",
            2020: "primera-a-2020",
            2019: "primera-a-2019",
        },
    },
    "chile": {
        "country": "chile",
        "seasons": {
            2025: "primera-division",
            2024: "primera-division-2024",
            2023: "primera-division-2023",
            2022: "primera-division-2022",
            2021: "primera-division-2021",
            2020: "primera-division-2020",
            2019: "primera-division-2019",
        },
    },
    "bolivia": {
        "country": "bolivia",
        "seasons": {
            2025: "liga-profesional",
            2024: "liga-profesional-2024",
            2023: "primera-division-2023",
            2022: "primera-division-2022",
            2021: "primera-division-2021",
            2020: "primera-division-2020",
            2019: "primera-division-2019",
        },
    },
    "peru": {
        "country": "peru",
        "seasons": {
            2025: "liga-1",
            2024: "liga-1-2024",
            2023: "liga-1-2023",
            2022: "liga-1-2022",
            2021: "liga-1-2021",
            2020: "liga-1-2020",
            2019: "liga-1-2019",
        },
    },
    "venezuela": {
        "country": "venezuela",
        "seasons": {
            2025: "liga-futve",
            2024: "liga-futve-2024",
            2023: "primera-division-2023",
            2022: "primera-division-2022",
            2021: "primera-division-2021",
            2020: "primera-division-2020",
            2019: "primera-division-2019",
        },
    },
    "ecuador": {
        "country": "ecuador",
        "seasons": {
            2025: "liga-pro",
            2024: "liga-pro-2024",
            2023: "liga-pro-2023",
            2022: "liga-pro-2022",
            2021: "liga-pro-2021",
            2020: "liga-pro-2020",
            2019: "liga-pro-2019",
        },
    },
    "saudi": {
        "country": "saudi-arabia",
        "seasons": {
            2026: "saudi-professional-league",
            2025: "saudi-professional-league-2024-2025",
            2024: "saudi-professional-league-2023-2024",
            2023: "saudi-professional-league-2022-2023",
            2022: "saudi-professional-league-2021-2022",
            2021: "saudi-professional-league-2020-2021",
            2020: "saudi-professional-league-2019-2020",
        },
    },
}

MAX_PAGES_PER_SEASON = 15
NAVIGATION_TIMEOUT_MS = 30_000
EVENT_ROWS_TIMEOUT_MS = 30_000
SCROLL_PASSES = 3
SCROLL_WAIT_MS = 3_000

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/131.0.0.0 Safari/537.36"
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ── JS extraction (identical to Colombia scraper) ─────────────────────

EXTRACT_JS = """
() => {
  const clean = (value) => (value || "").replace(/\\s+/g, " ").trim();

  const normalizeUrl = (href) => {
    if (!href) return null;
    try { return new URL(href, window.location.origin).href; }
    catch { return null; }
  };

  const getRowLines = (text) =>
    (text || "").split(/\\n+/).map((line) => line.trim()).filter(Boolean);

  const isScoreSeparator = (token) => token === "–" || token === "-" || token === "—";
  const isIntegerToken = (token) => /^\\d+$/.test(token || "");
  const isOddsLike = (v) => Number.isFinite(v) && v >= 1.01 && v <= 50.0;

  const parseAmericanOddsToken = (txt) => {
    if (!/^[+-]\\d{2,4}$/.test(txt || "")) return null;
    const american = Number(txt);
    if (!Number.isFinite(american) || american === 0) return null;
    const decimal = american < 0
      ? 1 + 100 / Math.abs(american)
      : 1 + american / 100;
    if (!isOddsLike(decimal)) return null;
    return Number(decimal.toFixed(2));
  };

  const parseDecimalOddsToken = (txt) => {
    if (!/^\\d{1,3}[.,]\\d{1,3}$/.test(txt || "")) return null;
    const decimal = Number(txt.replace(",", "."));
    if (!isOddsLike(decimal)) return null;
    return Number(decimal.toFixed(2));
  };

  const parseOddsToken = (txt) => {
    const raw = clean(txt || "");
    return parseAmericanOddsToken(raw) ?? parseDecimalOddsToken(raw);
  };

  const getMatchAnchor = (row) => {
    const anchors = Array.from(row.querySelectorAll("a[href]"));
    if (!anchors.length) return null;
    const footballAnchor = anchors.find((a) => {
      const href = a.getAttribute("href") || "";
      return href.includes("/football/");
    });
    return footballAnchor || anchors[0];
  };

  const extractTeamsAndScore = (row) => {
    const lines = getRowLines(row.innerText);
    const sepIdx = lines.findIndex((token) => isScoreSeparator(token));
    if (
      sepIdx >= 2 && sepIdx + 2 < lines.length &&
      isIntegerToken(lines[sepIdx - 1]) && isIntegerToken(lines[sepIdx + 1])
    ) {
      return {
        homeTeam: lines[sepIdx - 2], awayTeam: lines[sepIdx + 2],
        homeScore: Number(lines[sepIdx - 1]), awayScore: Number(lines[sepIdx + 1]),
        lines, sepIdx,
      };
    }
    return { homeTeam: null, awayTeam: null, homeScore: null, awayScore: null, lines, sepIdx: -1 };
  };

  const extractTeamsFallback = (row) => {
    const candidates = Array.from(row.querySelectorAll("a"))
      .map((a) => clean(a.textContent))
      .filter(Boolean)
      .filter((txt) => !isIntegerToken(txt) && parseOddsToken(txt) === null);
    const uniq = [];
    for (const name of candidates) { if (!uniq.includes(name)) uniq.push(name); }
    if (uniq.length >= 2) return [uniq[0], uniq[1]];
    return [null, null];
  };

  const chooseBestTriplet = (values) => {
    if (values.length < 3) return [null, null, null];
    let best = null, bestScore = -Infinity, bestIdx = -1;
    for (let i = 0; i <= values.length - 3; i++) {
      const h = values[i], d = values[i + 1], a = values[i + 2];
      if (!isOddsLike(h) || !isOddsLike(d) || !isOddsLike(a)) continue;
      const overround = (1 / h) + (1 / d) + (1 / a);
      if (overround < 0.85 || overround > 1.40) continue;
      let score = -Math.abs(overround - 1.05);
      if (d >= h && d >= a) score += 0.25;
      if (score > bestScore || (score === bestScore && i > bestIdx)) {
        bestScore = score; best = [h, d, a]; bestIdx = i;
      }
    }
    if (best) return best;
    return [values[0], values[1], values[2]];
  };

  const extractOdds = (row, structured) => {
    const sel = '[data-testid="odd-container-default"], [data-testid="odd-container-winning"]';
    const containerNodes = Array.from(row.querySelectorAll(sel))
      .filter(node => !node.querySelector(sel));
    const containerValues = [];
    for (const node of containerNodes) {
      const value = parseOddsToken(node.textContent);
      if (value !== null) containerValues.push(value);
    }
    if (containerValues.length >= 3) return chooseBestTriplet(containerValues);

    const values = [];
    let tokens = structured?.lines || getRowLines(row.innerText);
    if (structured && structured.sepIdx >= 0) tokens = tokens.slice(structured.sepIdx + 3);
    for (const token of tokens) {
      const value = parseOddsToken(token);
      if (value !== null) values.push(value);
    }

    if (values.length < 3) {
      const rawTokens = (row.innerText || "").match(/[+-]\\d{2,4}|\\d{1,3}[.,]\\d{1,3}/g) || [];
      for (const token of rawTokens) {
        const value = parseOddsToken(token);
        if (value !== null) values.push(value);
      }
    }
    return chooseBestTriplet(values);
  };

  const rows = Array.from(document.querySelectorAll('div[class^="eventRow"]'));
  const results = [], dedupe = new Set();

  for (const row of rows) {
    const matchAnchor = getMatchAnchor(row);
    if (!matchAnchor) continue;
    const matchLink = normalizeUrl(matchAnchor.getAttribute("href"));
    if (!matchLink) continue;

    const structured = extractTeamsAndScore(row);
    let homeTeam = structured.homeTeam, awayTeam = structured.awayTeam;
    let homeScore = structured.homeScore, awayScore = structured.awayScore;

    if (!homeTeam || !awayTeam) {
      const [fH, fA] = extractTeamsFallback(row);
      homeTeam = homeTeam || fH; awayTeam = awayTeam || fA;
    }
    if (!homeTeam || !awayTeam) continue;

    const [odds1, oddsX, odds2] = extractOdds(row, structured);

    const key = `${matchLink}|${homeTeam}|${awayTeam}`;
    if (dedupe.has(key)) continue;
    dedupe.add(key);

    const market = [];
    if (odds1 !== null && oddsX !== null && odds2 !== null) {
      market.push({
        "1": odds1.toFixed(2), "X": oddsX.toFixed(2), "2": odds2.toFixed(2),
        bookmaker_name: "avg", period: "FullTime",
      });
    }

    results.push({
      home_team: homeTeam, away_team: awayTeam,
      home_score: homeScore, away_score: awayScore,
      match_link: matchLink, "1x2_market": market,
    });
  }
  return results;
}
"""


async def scrape_page(page, url: str, page_number: int, label: str) -> list[dict[str, Any]]:
    """Navigate and scrape one results page."""
    try:
        await page.goto(url, wait_until="domcontentloaded", timeout=NAVIGATION_TIMEOUT_MS)
        if page_number > 1:
            await page.wait_for_function(
                f"window.location.href.includes('/#/page/{page_number}/')",
                timeout=8_000,
            )
        await page.wait_for_selector(
            'div[class^="eventRow"]',
            timeout=EVENT_ROWS_TIMEOUT_MS,
        )
    except PlaywrightTimeoutError:
        print(f"  {label} p{page_number}: no eventRows (timeout) — stopping season")
        return []

    for _ in range(SCROLL_PASSES):
        await page.mouse.wheel(0, 3_000)
        await page.wait_for_timeout(SCROLL_WAIT_MS)

    matches = await page.evaluate(EXTRACT_JS)
    print(f"  {label} p{page_number}: {len(matches)} matches")
    return matches


async def scrape_season(
    browser, country: str, slug: str, year: int
) -> list[dict[str, Any]]:
    """Scrape all pages for one season, auto-detecting page count."""
    base_url = f"https://www.oddsportal.com/football/{country}/{slug}/results/"
    label = f"[{country}/{year}]"
    season_matches: list[dict[str, Any]] = []
    global_dedupe: set[str] = set()

    for page_number in range(1, MAX_PAGES_PER_SEASON + 1):
        if page_number == 1:
            url = base_url
        else:
            url = f"{base_url}#/page/{page_number}/"

        context = await browser.new_context(
            user_agent=USER_AGENT,
            viewport={"width": 1366, "height": 2000},
        )
        page = await context.new_page()

        page_matches = await scrape_page(page, url, page_number, label)

        if not page_matches:
            await context.close()
            break

        # Dedup across pages
        new_matches = []
        for m in page_matches:
            link = m.get("match_link", "")
            if link and link not in global_dedupe:
                global_dedupe.add(link)
                new_matches.append(m)

        dupes = len(page_matches) - len(new_matches)
        if dupes > 0:
            print(f"  {label} p{page_number}: skipped {dupes} duplicates")

        season_matches.extend(new_matches)
        await context.close()

        # If we got very few matches, likely last page
        if len(page_matches) < 10:
            break

        await asyncio.sleep(random.uniform(2.5, 4.5))

    print(f"{label} TOTAL: {len(season_matches)} matches")
    return season_matches


async def main(league: str, years: list[int]) -> None:
    config = LEAGUE_CONFIGS[league]
    country = config["country"]
    seasons = config["seasons"]

    output_dir = PROJECT_ROOT / "data" / "oddsportal_raw"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_matches: list[dict[str, Any]] = []

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)

        for year in sorted(years, reverse=True):
            slug = seasons.get(year)
            if not slug:
                print(f"WARNING: No URL config for {league}/{year}, skipping")
                continue

            season_matches = await scrape_season(browser, country, slug, year)

            # Tag each match with season year
            for m in season_matches:
                m["season"] = year

            all_matches.extend(season_matches)

            # Save incrementally after each season
            partial_path = output_dir / f"{league}_partial.json"
            with partial_path.open("w", encoding="utf-8") as f:
                json.dump(all_matches, f, ensure_ascii=False, indent=2)

            await asyncio.sleep(random.uniform(3.0, 6.0))

        await browser.close()

    # Final output
    output_path = output_dir / f"{league}_all.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(all_matches, f, ensure_ascii=False, indent=2)

    # Summary
    with_odds = sum(1 for m in all_matches if m.get("1x2_market"))
    print(f"\n{'='*60}")
    print(f"DONE: {league}")
    print(f"Total matches: {len(all_matches)}")
    print(f"With odds: {with_odds} ({100*with_odds/max(len(all_matches),1):.1f}%)")
    print(f"Output: {output_path}")

    # Per-season breakdown
    from collections import Counter
    by_year = Counter(m["season"] for m in all_matches)
    odds_by_year = Counter(m["season"] for m in all_matches if m.get("1x2_market"))
    for y in sorted(by_year):
        print(f"  {y}: {by_year[y]} matches, {odds_by_year[y]} with odds")


def parse_args():
    parser = argparse.ArgumentParser(description="Scrape OddsPortal league results")
    parser.add_argument(
        "--league",
        required=True,
        choices=list(LEAGUE_CONFIGS.keys()),
        help="League to scrape",
    )
    parser.add_argument(
        "--years",
        required=True,
        help="Year range (e.g., 2019-2026 or 2023-2025)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    start, end = args.years.split("-")
    years = list(range(int(start), int(end) + 1))
    asyncio.run(main(args.league, years))
