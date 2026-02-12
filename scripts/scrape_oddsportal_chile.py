#!/usr/bin/env python3
"""
Scrape 1X2 odds from OddsPortal Chile Primera División / Liga de Primera (2019-2026).

Note: League renamed in 2025 from "primera-division" to "liga-de-primera".
URL slugs differ by year.

Output: data/oddsportal_raw/chile-primera-division_{year}.json
"""

from __future__ import annotations

import asyncio
import json
import random
from pathlib import Path
from typing import Any

from playwright.async_api import TimeoutError as PlaywrightTimeoutError
from playwright.async_api import async_playwright

# ── Configuration ─────────────────────────────────────────────────────────

SEASONS = [2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026]

def season_url(year: int, page: int) -> str:
    if year == 2026:
        slug = "liga-de-primera"
    elif year == 2025:
        slug = "liga-de-primera-2025"
    else:
        slug = f"primera-division-{year}"
    return f"https://www.oddsportal.com/football/chile/{slug}/results/#/page/{page}/"


MAX_PAGES_PER_SEASON = 10

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
OUTPUT_DIR = PROJECT_ROOT / "data" / "oddsportal_raw"
CHECKPOINT_PATH = OUTPUT_DIR / "chile_checkpoint.json"


# ── JavaScript extraction ─────────────────────────────────────────────────

EXTRACT_JS = """
() => {
  const clean = (value) => (value || "").replace(/\\s+/g, " ").trim();

  const normalizeUrl = (href) => {
    if (!href) return null;
    try {
      return new URL(href, window.location.origin).href;
    } catch {
      return null;
    }
  };

  const getRowLines = (text) =>
    (text || "")
      .split(/\\n+/)
      .map((line) => line.trim())
      .filter(Boolean);

  const isScoreSeparator = (token) => token === "–" || token === "-" || token === "—";
  const isIntegerToken = (token) => /^\\d+$/.test(token || "");
  const isOddsLike = (v) => Number.isFinite(v) && v >= 1.01 && v <= 50.0;

  const parseAmericanOddsToken = (txt) => {
    if (!/^[+-]\\d{2,4}$/.test(txt || "")) return null;
    const american = Number(txt);
    if (!Number.isFinite(american) || american === 0) return null;
    const decimal =
      american < 0
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
      return href.includes("/football/chile/");
    });
    return footballAnchor || anchors.find((a) =>
      (a.getAttribute("href") || "").includes("/football/")
    ) || anchors[0];
  };

  const extractTeamsAndScore = (row) => {
    const lines = getRowLines(row.innerText);
    const sepIdx = lines.findIndex((token) => isScoreSeparator(token));
    if (
      sepIdx >= 2 &&
      sepIdx + 2 < lines.length &&
      isIntegerToken(lines[sepIdx - 1]) &&
      isIntegerToken(lines[sepIdx + 1])
    ) {
      return {
        homeTeam: lines[sepIdx - 2],
        awayTeam: lines[sepIdx + 2],
        homeScore: Number(lines[sepIdx - 1]),
        awayScore: Number(lines[sepIdx + 1]),
        lines,
        sepIdx,
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
    for (const name of candidates) {
      if (!uniq.includes(name)) uniq.push(name);
    }
    if (uniq.length >= 2) return [uniq[0], uniq[1]];
    return [null, null];
  };

  const chooseBestTriplet = (values) => {
    if (values.length < 3) return [null, null, null];
    let best = null;
    let bestScore = -Infinity;
    let bestIdx = -1;
    for (let i = 0; i <= values.length - 3; i++) {
      const h = values[i];
      const d = values[i + 1];
      const a = values[i + 2];
      if (!isOddsLike(h) || !isOddsLike(d) || !isOddsLike(a)) continue;
      const overround = (1 / h) + (1 / d) + (1 / a);
      if (overround < 0.85 || overround > 1.40) continue;
      let score = -Math.abs(overround - 1.05);
      if (d >= h && d >= a) score += 0.25;
      if (score > bestScore || (score === bestScore && i > bestIdx)) {
        bestScore = score;
        best = [h, d, a];
        bestIdx = i;
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
    if (containerValues.length >= 3) {
      return chooseBestTriplet(containerValues);
    }

    const values = [];
    let tokens = structured?.lines || getRowLines(row.innerText);
    if (structured && structured.sepIdx >= 0) {
      tokens = tokens.slice(structured.sepIdx + 3);
    }
    for (const token of tokens) {
      const value = parseOddsToken(token);
      if (value !== null) values.push(value);
    }

    if (values.length < 3) {
      const rawTokens =
        (row.innerText || "").match(/[+-]\\d{2,4}|\\d{1,3}[.,]\\d{1,3}/g) || [];
      for (const token of rawTokens) {
        const value = parseOddsToken(token);
        if (value !== null) values.push(value);
      }
    }

    return chooseBestTriplet(values);
  };

  const MONTHS = {
    'Jan': '01', 'Feb': '02', 'Mar': '03', 'Apr': '04',
    'May': '05', 'Jun': '06', 'Jul': '07', 'Aug': '08',
    'Sep': '09', 'Oct': '10', 'Nov': '11', 'Dec': '12'
  };

  const parseDateHeader = (text) => {
    const m = (text || "").match(/(\\d{1,2})\\s+(\\w{3})\\s+(\\d{4})/);
    if (!m) return null;
    const day = m[1].padStart(2, '0');
    const month = MONTHS[m[2]];
    const year = m[3];
    if (!month) return null;
    return `${year}-${month}-${day}`;
  };

  const dateForRow = new Map();
  let currentDate = null;

  const allRows = document.querySelectorAll('div[class^="eventRow"]');
  for (const row of allRows) {
    const dateDiv = row.querySelector('div[class*="bg-gray-light"]');
    if (dateDiv) {
      const dateText = parseDateHeader(dateDiv.textContent);
      if (dateText) currentDate = dateText;
    }
    dateForRow.set(row, currentDate);
  }

  const rows = Array.from(document.querySelectorAll('div[class^="eventRow"]'));
  const results = [];
  const dedupe = new Set();

  for (const row of rows) {
    const matchAnchor = getMatchAnchor(row);
    if (!matchAnchor) continue;
    const matchLink = normalizeUrl(matchAnchor.getAttribute("href"));
    if (!matchLink) continue;

    const structured = extractTeamsAndScore(row);
    let homeTeam = structured.homeTeam;
    let awayTeam = structured.awayTeam;
    let homeScore = structured.homeScore;
    let awayScore = structured.awayScore;

    if (!homeTeam || !awayTeam) {
      const [fallbackHome, fallbackAway] = extractTeamsFallback(row);
      homeTeam = homeTeam || fallbackHome;
      awayTeam = awayTeam || fallbackAway;
    }
    if (!homeTeam || !awayTeam) continue;

    const [odds1, oddsX, odds2] = extractOdds(row, structured);

    const key = `${matchLink}|${homeTeam}|${awayTeam}`;
    if (dedupe.has(key)) continue;
    dedupe.add(key);

    const matchDate = dateForRow.get(row) || null;

    const market = [];
    if (odds1 !== null && oddsX !== null && odds2 !== null) {
      market.push({
        "1": odds1.toFixed(2),
        "X": oddsX.toFixed(2),
        "2": odds2.toFixed(2),
        bookmaker_name: "avg",
        period: "FullTime",
      });
    }

    results.push({
      home_team: homeTeam,
      away_team: awayTeam,
      home_score: homeScore,
      away_score: awayScore,
      match_date: matchDate,
      match_link: matchLink,
      "1x2_market": market,
    });
  }

  return results;
}
"""


def load_checkpoint() -> dict[str, Any]:
    if CHECKPOINT_PATH.exists():
        with CHECKPOINT_PATH.open() as f:
            return json.load(f)
    return {"completed_seasons": [], "partial": {}}


def save_checkpoint(ckpt: dict[str, Any]) -> None:
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CHECKPOINT_PATH.open("w") as f:
        json.dump(ckpt, f, indent=2)


async def scrape_page(page, year: int, page_number: int) -> list[dict[str, Any]]:
    url = season_url(year, page_number)
    try:
        await page.goto(url, wait_until="domcontentloaded", timeout=NAVIGATION_TIMEOUT_MS)
        await page.wait_for_function(
            f"window.location.href.includes('/#/page/{page_number}/')",
            timeout=8_000,
        )
        await page.wait_for_selector(
            'div[class^="eventRow"]',
            timeout=EVENT_ROWS_TIMEOUT_MS,
        )
    except PlaywrightTimeoutError:
        return []

    for _ in range(SCROLL_PASSES):
        await page.mouse.wheel(0, 3_000)
        await page.wait_for_timeout(SCROLL_WAIT_MS)

    matches = await page.evaluate(EXTRACT_JS)
    return matches


async def scrape_season(browser, year: int, ckpt: dict) -> list[dict[str, Any]]:
    all_matches: list[dict[str, Any]] = []
    global_dedupe: set[str] = set()

    start_page = 1
    partial = ckpt.get("partial", {})
    if str(year) in partial:
        start_page = partial[str(year)].get("next_page", 1)
        output_path = OUTPUT_DIR / f"chile-primera-division_{year}.json"
        if output_path.exists():
            with output_path.open() as f:
                existing = json.load(f)
            all_matches = existing
            for m in existing:
                link = m.get("match_link", "")
                if link:
                    global_dedupe.add(link)
            print(f"  Resuming season {year} from page {start_page} ({len(existing)} matches on disk)")

    for page_number in range(start_page, MAX_PAGES_PER_SEASON + 1):
        context = await browser.new_context(
            user_agent=USER_AGENT,
            viewport={"width": 1366, "height": 2000},
        )
        page = await context.new_page()

        page_matches = await scrape_page(page, year, page_number)
        await context.close()

        if not page_matches:
            print(f"  Page {page_number}: 0 matches (end of season)")
            break

        new_matches = []
        for m in page_matches:
            link = m.get("match_link", "")
            if link and link not in global_dedupe:
                global_dedupe.add(link)
                new_matches.append(m)

        dupes = len(page_matches) - len(new_matches)
        odds_count = sum(1 for m in new_matches if m.get("1x2_market"))
        dates_count = sum(1 for m in new_matches if m.get("match_date"))

        print(
            f"  Page {page_number}: {len(new_matches)} new matches "
            f"(odds: {odds_count}, dates: {dates_count}"
            f"{f', {dupes} dupes' if dupes > 0 else ''})"
        )

        all_matches.extend(new_matches)

        ckpt["partial"][str(year)] = {"next_page": page_number + 1}
        save_checkpoint(ckpt)

        output_path = OUTPUT_DIR / f"chile-primera-division_{year}.json"
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(all_matches, f, ensure_ascii=False, indent=2)

        if page_number < MAX_PAGES_PER_SEASON:
            await asyncio.sleep(random.uniform(3.0, 5.0))

    return all_matches


async def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt = load_checkpoint()

    completed = set(ckpt.get("completed_seasons", []))
    remaining = [y for y in SEASONS if y not in completed]

    if not remaining:
        print("All seasons already completed! Delete checkpoint to re-scrape.")
        return

    print(f"Chile Primera División OddsPortal Scraper")
    print(f"Seasons to scrape: {remaining}")
    if completed:
        print(f"Already completed: {sorted(completed)}")
    print()

    grand_total = 0
    grand_odds = 0

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)

        for year in remaining:
            print(f"\n{'='*50}")
            print(f"Season {year}")
            print(f"{'='*50}")

            matches = await scrape_season(browser, year, ckpt)

            n_odds = sum(1 for m in matches if m.get("1x2_market"))
            n_dates = sum(1 for m in matches if m.get("match_date"))

            if matches:
                print(f"\n  Season {year} DONE: {len(matches)} matches, "
                      f"{n_odds} with odds ({n_odds/len(matches)*100:.0f}%), "
                      f"{n_dates} with dates")
            else:
                print(f"\n  Season {year} DONE: 0 matches")

            grand_total += len(matches)
            grand_odds += n_odds

            ckpt["completed_seasons"] = sorted(set(ckpt.get("completed_seasons", [])) | {year})
            if str(year) in ckpt.get("partial", {}):
                del ckpt["partial"][str(year)]
            save_checkpoint(ckpt)

            if year != remaining[-1]:
                await asyncio.sleep(random.uniform(4.0, 7.0))

        await browser.close()

    print(f"\n{'='*60}")
    print(f"GRAND TOTAL: {grand_total} matches, {grand_odds} with odds")
    for year in SEASONS:
        output_path = OUTPUT_DIR / f"chile-primera-division_{year}.json"
        if output_path.exists():
            with output_path.open() as f:
                data = json.load(f)
            n = len(data)
            n_o = sum(1 for m in data if m.get("1x2_market"))
            if n > 0:
                print(f"  {year}: {n} matches, {n_o} with odds ({n_o/n*100:.0f}%)")
            else:
                print(f"  {year}: 0 matches")
    print(f"\nOutput: {OUTPUT_DIR}/chile-primera-division_*.json")


if __name__ == "__main__":
    asyncio.run(main())
