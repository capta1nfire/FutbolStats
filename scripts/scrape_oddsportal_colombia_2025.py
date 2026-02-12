#!/usr/bin/env python3
"""
Scrape 1X2 odds from OddsPortal Colombia Primera A 2025 results pages.

Output format is compatible with the existing odds ingestion pipeline.
"""

from __future__ import annotations

import asyncio
import json
import random
from pathlib import Path
from typing import Any

from playwright.async_api import TimeoutError as PlaywrightTimeoutError
from playwright.async_api import async_playwright


BASE_URL_TEMPLATE = (
    "https://www.oddsportal.com/football/colombia/primera-a-2025/results/#/page/{page}/"
)
START_PAGE = 1
END_PAGE = 9
TOTAL_PAGES = END_PAGE - START_PAGE + 1

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
OUTPUT_PATH = (
    PROJECT_ROOT / "data" / "oddsportal_raw" / "colombia-primera-a_2025_rescrape.json"
)


async def extract_matches_from_page(page) -> list[dict[str, Any]]:
    """Extract all match rows from the current page."""
    return await page.evaluate(
        """
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

            const preferred = anchors.find((a) => {
              const href = a.getAttribute("href") || "";
              return href.includes("/football/colombia/primera-a-2025/");
            });
            if (preferred) return preferred;

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

            return {
              homeTeam: null,
              awayTeam: null,
              homeScore: null,
              awayScore: null,
              lines,
              sepIdx: -1,
            };
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

              // Desempate: preferir la tripleta más a la derecha del row.
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
            // 1) Odds explícitas por data-testid (leaf-only: no parent containers).
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

            // 2) Fallback: desde tokens estructurados del row (después del away team).
            const values = [];
            let tokens = structured?.lines || getRowLines(row.innerText);
            if (structured && structured.sepIdx >= 0) {
              tokens = tokens.slice(structured.sepIdx + 3);
            }
            for (const token of tokens) {
              const value = parseOddsToken(token);
              if (value !== null) values.push(value);
            }

            // 3) Último fallback por regex en texto completo.
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
              match_link: matchLink,
              "1x2_market": market,
            });
          }

          return results;
        }
        """
    )


async def scrape_page(page, page_number: int) -> list[dict[str, Any]]:
    """Navigate and scrape one results page."""
    url = BASE_URL_TEMPLATE.format(page=page_number)
    try:
        await page.goto(
            url,
            wait_until="domcontentloaded",
            timeout=NAVIGATION_TIMEOUT_MS,
        )
        # OddsPortal usa hash routing (#/page/N/). Confirmamos que cambió al hash esperado.
        await page.wait_for_function(
            f"window.location.href.includes('/#/page/{page_number}/')",
            timeout=8_000,
        )
        await page.wait_for_selector(
            'div[class^="eventRow"]',
            timeout=EVENT_ROWS_TIMEOUT_MS,
        )
    except PlaywrightTimeoutError:
        print(
            f"WARNING: Page {page_number}/{TOTAL_PAGES} sin eventRows en 30s. Se omite."
        )
        return []

    for _ in range(SCROLL_PASSES):
        await page.mouse.wheel(0, 3_000)
        await page.wait_for_timeout(SCROLL_WAIT_MS)

    matches = await extract_matches_from_page(page)
    print(f"Page {page_number}/{TOTAL_PAGES}: {len(matches)} matches extracted")
    return matches


async def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    all_matches: list[dict[str, Any]] = []
    global_dedupe: set[str] = set()

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)

        for page_number in range(START_PAGE, END_PAGE + 1):
            # Fresh context+page per page number to avoid SPA cache
            context = await browser.new_context(
                user_agent=USER_AGENT,
                viewport={"width": 1366, "height": 2000},
            )
            page = await context.new_page()

            page_matches = await scrape_page(page, page_number)

            # Dedup across pages by match_link
            new_matches = []
            for m in page_matches:
                link = m.get("match_link", "")
                if link and link not in global_dedupe:
                    global_dedupe.add(link)
                    new_matches.append(m)

            dupes = len(page_matches) - len(new_matches)
            if dupes > 0:
                print(f"  (skipped {dupes} cross-page duplicates)")
            all_matches.extend(new_matches)

            await context.close()

            if page_number < END_PAGE:
                await asyncio.sleep(random.uniform(3.0, 5.0))

        await browser.close()

    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(all_matches, f, ensure_ascii=False, indent=2)

    print(f"Total matches extraídos: {len(all_matches)}")
    print(f"Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
