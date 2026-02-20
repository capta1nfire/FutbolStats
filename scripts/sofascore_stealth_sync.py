#!/usr/bin/env python3
"""
Sofascore Stealth Sync — Playwright-based data collection.

Replaces httpx-based sofascore_provider for data fetching.
Uses browser stealth to bypass Sofascore's anti-bot protection (Varnish/CF).

ATI Directive (2026-02-20):
- playwright-stealth for headless evasion
- Random delays (2-6s) between requests
- Block images/CSS/fonts to reduce bandwidth
- page.on('response') to intercept API JSON
- Local execution with batch DB ingestion (NOT Railway Docker)

Usage:
    source .env
    python scripts/sofascore_stealth_sync.py --test           # verify stealth works
    python scripts/sofascore_stealth_sync.py --refs           # sync refs (scheduled events)
    python scripts/sofascore_stealth_sync.py --stats          # backfill stats for FT matches
    python scripts/sofascore_stealth_sync.py --lineups        # capture XI for upcoming matches
    python scripts/sofascore_stealth_sync.py --ratings        # backfill ratings for FT matches
    python scripts/sofascore_stealth_sync.py --all            # all sync modes
"""

import argparse
import asyncio
import json
import logging
import os
import random
import sys
import time
from datetime import datetime, timedelta
from typing import Any, Optional

# Add project root to path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("sofascore_stealth")

# ─── Config ───────────────────────────────────────────────────────────────────
SOFASCORE_WEB = "https://www.sofascore.com"
SOFASCORE_API = "https://api.sofascore.com/api/v1"

DELAY_MIN = 2.0
DELAY_MAX = 6.0
SESSION_ROTATE_EVERY = 25
MAX_CONSECUTIVE_ERRORS = 5
BLOCKED_RESOURCES = {"image", "stylesheet", "font", "media"}

USER_AGENTS = [
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36",
]

# SOFASCORE_SUPPORTED_LEAGUES from sota_constants (subset used for refs)
SUPPORTED_LEAGUE_IDS = {
    2, 3, 39, 40, 61, 78, 88, 94, 128, 135, 140, 203,
    239, 253, 262, 265, 268, 281, 299, 307, 344,
}


# ═════════════════════════════════════════════════════════════════════════════
# STEALTH FETCHER
# ═════════════════════════════════════════════════════════════════════════════

class StealthFetcher:
    """Playwright-based stealth fetcher for Sofascore API data.

    Two fetch strategies:
      1. Primary: After establishing session on sofascore.com, navigate directly
         to API URLs in the browser (cookies carry over from session).
      2. Fallback: Navigate to Sofascore web page, intercept API responses
         via page.on('response').
    """

    def __init__(self, headless=True):
        self._headless = headless
        self._pw = None
        self._browser = None
        self._context = None
        self._page = None
        self._request_count = 0
        self._consecutive_errors = 0
        self._stealth = None

    async def start(self):
        """Launch browser with stealth configuration."""
        from playwright.async_api import async_playwright
        from playwright_stealth import Stealth

        self._stealth = Stealth(
            navigator_platform_override="MacIntel",
            navigator_vendor_override="Google Inc.",
        )
        self._pw = await async_playwright().start()
        self._browser = await self._pw.chromium.launch(
            headless=self._headless,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--no-sandbox",
            ],
        )
        await self._new_session()

    async def _simulate_human_behavior(self):
        """Simulate human mouse movements and scrolling."""
        logger.info("Simulating human interaction (mouse & scroll)...")
        try:
            # Random mouse movements
            for _ in range(random.randint(3, 6)):
                x = random.randint(100, 800)
                y = random.randint(100, 600)
                await self._page.mouse.move(x, y, steps=random.randint(5, 15))
                await asyncio.sleep(random.uniform(0.1, 0.5))
            
            # Random scrolling
            for _ in range(random.randint(2, 4)):
                await self._page.mouse.wheel(0, random.randint(200, 600))
                await asyncio.sleep(random.uniform(0.5, 1.5))
                
        except Exception as e:
            logger.warning("Human simulation error: %s", e)

    async def _new_session(self):
        """Create fresh context + page, navigate to sofascore.com for session cookies."""
        if self._context:
            try:
                await self._context.close()
            except Exception:
                pass

        self._context = await self._browser.new_context(
            viewport={
                "width": random.randint(1280, 1920),
                "height": random.randint(800, 1080),
            },
            user_agent=random.choice(USER_AGENTS),
            locale="en-US",
        )

        # Apply stealth to context (all pages inherit)
        await self._stealth.apply_stealth_async(self._context)

        self._page = await self._context.new_page()

        # Block images/CSS/fonts per ATI directive
        await self._page.route("**/*", self._block_resources)

        # Establish session — navigate to sofascore.com
        logger.info("Establishing session on sofascore.com...")
        try:
            await self._page.goto(SOFASCORE_WEB, wait_until="domcontentloaded", timeout=45000)
        except Exception as e:
            logger.warning("Initial navigation slow (%s), continuing anyway", type(e).__name__)
        
        await self._simulate_human_behavior()
        await asyncio.sleep(random.uniform(1, 3))

        cookies = await self._context.cookies()
        logger.info("Session established (%d cookies)", len(cookies))
        self._request_count = 0
        self._consecutive_errors = 0

    async def _block_resources(self, route):
        """Block images, CSS, fonts, media to reduce bandwidth."""
        if route.request.resource_type in BLOCKED_RESOURCES:
            await route.abort()
        else:
            await route.continue_()

    async def fetch_api(self, api_path: str) -> Optional[dict]:
        """Fetch API endpoint using an injected XHR/fetch from the page context.

        Instead of navigating the browser to the JSON endpoint (which triggers
        'Sec-Fetch-Dest: document' and lacks proper Referer), we inject a fetch()
        call into the existing sofascore.com page. This makes the request look
        exactly like the real frontend SPA making an API call.
        """
        # Rotate session periodically
        if self._request_count >= SESSION_ROTATE_EVERY:
            logger.info("Rotating session after %d requests", self._request_count)
            await self._new_session()

        # Human-like delay with jitter
        await asyncio.sleep(random.uniform(DELAY_MIN, DELAY_MAX))
        
        # Ocasionalmente añadir un micro-movimiento antes de pedir datos
        if random.random() < 0.3:
            try:
                await self._page.mouse.move(random.randint(100, 800), random.randint(100, 600))
            except Exception:
                pass

        url = f"{SOFASCORE_API}{api_path}"
        try:
            # Inject fetch into the zombie page (URL passed as argument, not interpolated)
            data = await self._page.evaluate('''async (url) => {
                const res = await fetch(url);
                if (!res.ok) throw new Error("HTTP " + res.status);
                return await res.json();
            }''', url)
            
            self._request_count += 1
            self._consecutive_errors = 0
            return data

        except Exception as e:
            logger.error("fetch_api error for %s: %s", api_path, e)
            self._consecutive_errors += 1

            # Rotate session on repeated errors
            if self._consecutive_errors >= 3:
                logger.info("3+ consecutive errors, rotating session")
                await self._new_session()

            return None

    async def fetch_via_intercept(self, page_url: str, api_pattern: str) -> Optional[dict]:
        """Navigate to a Sofascore web page and intercept matching API response.

        This is the fallback strategy per ATI directive:
        page.on('response') to capture the JSON from the API call
        that the page makes during rendering.
        """
        captured = {}

        async def on_response(response):
            if api_pattern in response.url and response.status == 200:
                try:
                    captured["data"] = await response.json()
                except Exception:
                    pass

        page = await self._context.new_page()
        await self._stealth.apply_stealth_async(page)
        await page.route("**/*", self._block_resources)
        page.on("response", on_response)

        try:
            await page.goto(page_url, wait_until="domcontentloaded", timeout=45000)
            # Wait extra for API calls to fire after page load
            await asyncio.sleep(random.uniform(5, 8))
        except Exception as e:
            logger.warning("intercept navigation error for %s: %s", page_url, e)
        finally:
            await page.close()

        self._request_count += 1
        return captured.get("data")

    @property
    def consecutive_errors(self):
        return self._consecutive_errors

    async def close(self):
        """Clean up browser resources."""
        if self._context:
            try:
                await self._context.close()
            except Exception:
                pass
        if self._browser:
            await self._browser.close()
        if self._pw:
            await self._pw.stop()
        logger.info("Browser closed")


# ═════════════════════════════════════════════════════════════════════════════
# PARSING HELPERS (reuse from sofascore_provider.py)
# ═════════════════════════════════════════════════════════════════════════════

def parse_scheduled_events(data: dict) -> list[dict]:
    """Parse scheduled-events API response into event dicts."""
    events = []
    for event in data.get("events", []):
        event_id = event.get("id")
        if not event_id:
            continue
        home = event.get("homeTeam", {})
        away = event.get("awayTeam", {})
        tournament = event.get("tournament", {})
        start_ts = event.get("startTimestamp")
        kickoff = None
        if start_ts:
            try:
                kickoff = datetime.utcfromtimestamp(start_ts)
            except (ValueError, TypeError):
                pass
        events.append({
            "event_id": str(event_id),
            "home_team": home.get("name", ""),
            "away_team": away.get("name", ""),
            "kickoff_utc": kickoff,
            "league_name": tournament.get("name", ""),
            "home_team_slug": home.get("slug", ""),
            "away_team_slug": away.get("slug", ""),
        })
    return events


def parse_lineup_data(data: dict) -> dict:
    """Parse lineups API response into structured dict."""
    from app.etl.sofascore_provider import normalize_position

    result = {"home": None, "away": None}
    for side in ("home", "away"):
        team_data = data.get(side, {})
        if not team_data:
            continue
        formation = team_data.get("formation")
        players = []
        for entry in team_data.get("players", []):
            player_info = entry.get("player", {})
            stats = entry.get("statistics", {})
            pid = player_info.get("id")
            if not pid:
                continue
            raw_pos = player_info.get("position", "")
            rating = stats.get("rating")
            avg_rating = player_info.get("averageRating")
            players.append({
                "player_id_ext": str(pid),
                "position": normalize_position(raw_pos),
                "is_starter": not entry.get("substitute", False),
                "rating": float(rating) if rating else None,
                "rating_pre_match": float(avg_rating) if avg_rating else None,
                "name": player_info.get("name"),
            })
        result[side] = {"formation": formation, "players": players}
    return result


def parse_statistics(data: dict) -> dict:
    """Parse statistics API response. Reuses SofascoreProvider logic."""
    from app.etl.sofascore_provider import SofascoreProvider
    parser = SofascoreProvider(use_mock=True)
    return parser._parse_statistics_response(data)


# ═════════════════════════════════════════════════════════════════════════════
# TEST MODE
# ═════════════════════════════════════════════════════════════════════════════

async def run_test(fetcher: StealthFetcher):
    """Test stealth approaches against Sofascore API."""
    today = datetime.utcnow().strftime("%Y-%m-%d")
    tomorrow = (datetime.utcnow() + timedelta(days=1)).strftime("%Y-%m-%d")

    print("\n" + "=" * 60)
    print("SOFASCORE STEALTH TEST")
    print("=" * 60)

    # Test 1: Direct API navigation (primary approach)
    print(f"\n[1] Direct API navigation: scheduled-events/{tomorrow}")
    data = await fetcher.fetch_api(f"/sport/football/scheduled-events/{tomorrow}")
    if data:
        events = data.get("events", [])
        print(f"    ✓ SUCCESS — {len(events)} events")
        if events:
            e = events[0]
            ht = e.get("homeTeam", {}).get("name", "?")
            at = e.get("awayTeam", {}).get("name", "?")
            print(f"    First: {ht} vs {at}")
        approach_1_works = True
    else:
        print("    ✗ FAILED — 403 or error")
        approach_1_works = False

    # Test 2: Page interception (fallback approach)
    print(f"\n[2] Page interception: football/{tomorrow}")
    data2 = await fetcher.fetch_via_intercept(
        f"{SOFASCORE_WEB}/football/{tomorrow}",
        "scheduled-events",
    )
    if data2:
        events2 = data2.get("events", [])
        print(f"    ✓ SUCCESS — intercepted {len(events2)} events")
        approach_2_works = True
    else:
        print("    ✗ FAILED — no API response intercepted")
        approach_2_works = False

    # Test 3: Single event lineups (if we have events)
    test_events = data or data2
    if test_events and test_events.get("events"):
        # Find a recent/upcoming match
        event_id = test_events["events"][0]["id"]
        print(f"\n[3] Event lineups: event/{event_id}/lineups")
        lineup_data = await fetcher.fetch_api(f"/event/{event_id}/lineups")
        if lineup_data:
            parsed = parse_lineup_data(lineup_data)
            home_n = len(parsed["home"]["players"]) if parsed["home"] else 0
            away_n = len(parsed["away"]["players"]) if parsed["away"] else 0
            print(f"    ✓ SUCCESS — home:{home_n} away:{away_n} players")
        else:
            print("    ✗ FAILED or no lineup available")

        # Test 4: Event statistics
        print(f"\n[4] Event statistics: event/{event_id}/statistics")
        stats_data = await fetcher.fetch_api(f"/event/{event_id}/statistics")
        if stats_data:
            parsed_stats = parse_statistics(stats_data)
            xg = parsed_stats.get("xg_home")
            poss = parsed_stats.get("possession_home")
            print(f"    ✓ SUCCESS — xG_home={xg}, poss_home={poss}%")
        else:
            print("    ✗ FAILED or no stats available (match may be NS)")
    else:
        print("\n[3-4] Skipped — no events to test with")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Direct API navigation:  {'✓ WORKS' if approach_1_works else '✗ BLOCKED'}")
    print(f"  Page interception:      {'✓ WORKS' if approach_2_works else '✗ BLOCKED'}")
    if approach_1_works:
        print("  → Use direct API navigation (faster for batch)")
    elif approach_2_works:
        print("  → Use page interception (slower but works)")
    else:
        print("  → BOTH APPROACHES BLOCKED — need different strategy")
    print()


# ═════════════════════════════════════════════════════════════════════════════
# DB HELPERS
# ═════════════════════════════════════════════════════════════════════════════

async def get_db_pool():
    """Create asyncpg connection pool."""
    import asyncpg
    url = os.environ.get("DATABASE_URL")
    if not url:
        logger.error("DATABASE_URL not set. Run: source .env")
        sys.exit(1)
    # asyncpg needs postgresql:// not postgres://
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql://", 1)
    return await asyncpg.create_pool(url, min_size=1, max_size=3)


# ═════════════════════════════════════════════════════════════════════════════
# MODE: REFS SYNC
# ═════════════════════════════════════════════════════════════════════════════

async def sync_refs(fetcher: StealthFetcher):
    """Sync Sofascore event refs for upcoming matches."""
    from app.etl.sofascore_provider import calculate_match_score, get_sofascore_threshold
    from app.etl.sofascore_aliases import build_alias_index

    pool = await get_db_pool()
    alias_index = build_alias_index()

    dates = [(datetime.utcnow() + timedelta(days=d)).strftime("%Y-%m-%d") for d in range(4)]
    total_linked = 0
    total_scanned = 0

    for date_str in dates:
        logger.info("[REFS] Fetching scheduled events for %s", date_str)
        data = await fetcher.fetch_api(f"/sport/football/scheduled-events/{date_str}")
        if not data:
            # Try interception fallback
            data = await fetcher.fetch_via_intercept(
                f"{SOFASCORE_WEB}/football/{date_str}",
                "scheduled-events",
            )
        if not data:
            logger.warning("[REFS] Failed to fetch events for %s", date_str)
            continue

        events = parse_scheduled_events(data)
        logger.info("[REFS] %d events found for %s", len(events), date_str)

        # Get our matches for this date range
        rows = await pool.fetch("""
            SELECT m.id, m.date, t_home.name as home_name, t_away.name as away_name,
                   m.league_id
            FROM matches m
            JOIN teams t_home ON t_home.id = m.home_team_id
            JOIN teams t_away ON t_away.id = m.away_team_id
            WHERE m.date::date = $1::date
              AND m.league_id = ANY($2)
        """, datetime.strptime(date_str, "%Y-%m-%d"), list(SUPPORTED_LEAGUE_IDS))

        # Check existing refs
        existing = set()
        if rows:
            match_ids = [r["id"] for r in rows]
            existing_rows = await pool.fetch("""
                SELECT match_id FROM match_external_refs
                WHERE source = 'sofascore' AND match_id = ANY($1)
            """, match_ids)
            existing = {r["match_id"] for r in existing_rows}

        for match_row in rows:
            total_scanned += 1
            if match_row["id"] in existing:
                continue

            best_score = 0.0
            best_event = None
            best_matched_by = ""

            for event in events:
                if not event["kickoff_utc"]:
                    continue
                score, matched_by = calculate_match_score(
                    our_home=match_row["home_name"],
                    our_away=match_row["away_name"],
                    our_kickoff=match_row["date"],
                    sf_home=event["home_team"],
                    sf_away=event["away_team"],
                    sf_kickoff=event["kickoff_utc"],
                    alias_index=alias_index,
                )
                if score > best_score:
                    best_score = score
                    best_event = event
                    best_matched_by = matched_by

            threshold = get_sofascore_threshold(match_row["league_id"])
            if best_score >= threshold and best_event:
                await pool.execute("""
                    INSERT INTO match_external_refs (match_id, source, source_match_id, confidence, matched_by, created_at)
                    VALUES ($1, 'sofascore', $2, $3, $4, NOW())
                    ON CONFLICT (match_id, source) DO NOTHING
                """, match_row["id"], best_event["event_id"], best_score, best_matched_by)
                total_linked += 1
                logger.info("  Linked match %d → event %s (%.3f, %s)",
                            match_row["id"], best_event["event_id"], best_score, best_matched_by)

    await pool.close()
    logger.info("[REFS] Complete: scanned=%d, linked=%d", total_scanned, total_linked)


# ═════════════════════════════════════════════════════════════════════════════
# MODE: STATS BACKFILL
# ═════════════════════════════════════════════════════════════════════════════

async def sync_stats(fetcher: StealthFetcher):
    """Backfill post-match statistics for FT matches with Sofascore refs."""
    pool = await get_db_pool()

    # FT matches with ref but no stats (last 7 days)
    rows = await pool.fetch("""
        SELECT m.id as match_id, mer.source_match_id as event_id, m.league_id
        FROM matches m
        JOIN match_external_refs mer ON mer.match_id = m.id AND mer.source = 'sofascore'
        LEFT JOIN match_sofascore_stats mss ON mss.match_id = m.id
        WHERE m.status IN ('FT', 'AET', 'PEN')
          AND m.date > NOW() - INTERVAL '7 days'
          -- Jitter dinámico: Esperar entre 25 mins y ~4 horas DESPUÉS de que termina el partido
          -- (Asumiendo 115 mins de duración de partido + 25 mins base + (0-215 mins aleatorios por ID))
          AND m.date + (140 + (m.id % 215)) * INTERVAL '1 minute' < NOW()
          AND mss.match_id IS NULL
        ORDER BY m.date DESC
        LIMIT 100
    """)

    logger.info("[STATS] %d FT matches to backfill", len(rows))
    updated = 0
    errors = 0

    for row in rows:
        event_id = row["event_id"]
        data = await fetcher.fetch_api(f"/event/{event_id}/statistics")
        if not data:
            errors += 1
            if fetcher.consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                logger.error("[STATS] Too many consecutive errors, stopping")
                break
            continue

        stats = parse_statistics(data)

        await pool.execute("""
            INSERT INTO match_sofascore_stats (
                match_id, possession_home, possession_away,
                total_shots_home, total_shots_away,
                shots_on_target_home, shots_on_target_away,
                xg_home, xg_away, corners_home, corners_away,
                fouls_home, fouls_away,
                big_chances_home, big_chances_away,
                big_chances_missed_home, big_chances_missed_away,
                accurate_passes_home, accurate_passes_away,
                pass_accuracy_home, pass_accuracy_away,
                raw_stats, captured_at
            ) VALUES (
                $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11,
                $12, $13, $14, $15, $16, $17, $18, $19, $20, $21,
                $22, NOW()
            )
            ON CONFLICT (match_id) DO NOTHING
        """,
            row["match_id"],
            stats.get("possession_home"), stats.get("possession_away"),
            stats.get("total_shots_home"), stats.get("total_shots_away"),
            stats.get("shots_on_target_home"), stats.get("shots_on_target_away"),
            stats.get("xg_home"), stats.get("xg_away"),
            stats.get("corners_home"), stats.get("corners_away"),
            stats.get("fouls_home"), stats.get("fouls_away"),
            stats.get("big_chances_home"), stats.get("big_chances_away"),
            stats.get("big_chances_missed_home"), stats.get("big_chances_missed_away"),
            stats.get("accurate_passes_home"), stats.get("accurate_passes_away"),
            stats.get("pass_accuracy_home"), stats.get("pass_accuracy_away"),
            json.dumps(stats.get("raw_stats", {})),
        )
        updated += 1
        logger.info("  Stats captured for match %d (event %s)", row["match_id"], event_id)

    await pool.close()
    logger.info("[STATS] Complete: updated=%d, errors=%d", updated, errors)


# ═════════════════════════════════════════════════════════════════════════════
# MODE: LINEUPS CAPTURE
# ═════════════════════════════════════════════════════════════════════════════

async def sync_lineups(fetcher: StealthFetcher):
    """Capture pre-kickoff lineups for NS matches with Sofascore refs."""
    pool = await get_db_pool()

    # NS matches in next 48h with ref but no lineup
    rows = await pool.fetch("""
        SELECT m.id as match_id, mer.source_match_id as event_id, m.league_id
        FROM matches m
        JOIN match_external_refs mer ON mer.match_id = m.id AND mer.source = 'sofascore'
        LEFT JOIN match_sofascore_lineup msl ON msl.match_id = m.id
        WHERE m.status = 'NS'
          AND m.date BETWEEN NOW() AND NOW() + INTERVAL '48 hours'
          AND msl.match_id IS NULL
        ORDER BY m.date ASC
        LIMIT 50
    """)

    logger.info("[LINEUPS] %d NS matches to check", len(rows))
    captured = 0
    errors = 0
    now = datetime.utcnow()

    for row in rows:
        event_id = row["event_id"]
        data = await fetcher.fetch_api(f"/event/{event_id}/lineups")
        if not data:
            errors += 1
            if fetcher.consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                logger.error("[LINEUPS] Too many consecutive errors, stopping")
                break
            continue

        parsed = parse_lineup_data(data)

        for side in ("home", "away"):
            team = parsed.get(side)
            if not team or not team.get("players"):
                continue

            starters = [p for p in team["players"] if p["is_starter"]]
            if len(starters) < 7:
                continue  # Not a real lineup yet

            # Upsert lineup
            await pool.execute("""
                INSERT INTO match_sofascore_lineup (match_id, team_side, formation, captured_at)
                VALUES ($1, $2, $3, $4)
                ON CONFLICT (match_id, team_side) DO UPDATE SET
                    formation = EXCLUDED.formation,
                    captured_at = EXCLUDED.captured_at
            """, row["match_id"], side, team.get("formation"), now)

            # Upsert players
            for p in team["players"]:
                await pool.execute("""
                    INSERT INTO match_sofascore_player (
                        match_id, team_side, player_id_ext, position,
                        is_starter, rating_pre_match, rating_recent_form, captured_at
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                    ON CONFLICT (match_id, team_side, player_id_ext) DO UPDATE SET
                        position = EXCLUDED.position,
                        is_starter = EXCLUDED.is_starter,
                        rating_pre_match = EXCLUDED.rating_pre_match,
                        rating_recent_form = EXCLUDED.rating_recent_form,
                        captured_at = EXCLUDED.captured_at
                """,
                    row["match_id"], side, p["player_id_ext"], p["position"],
                    p["is_starter"], p.get("rating_pre_match"), p.get("rating"),
                    now,
                )

            captured += 1
            logger.info("  Lineup captured for match %d %s (%d players, formation=%s)",
                        row["match_id"], side, len(team["players"]), team.get("formation"))

    await pool.close()
    logger.info("[LINEUPS] Complete: captured=%d (sides), errors=%d", captured, errors)


# ═════════════════════════════════════════════════════════════════════════════
# MODE: RATINGS BACKFILL
# ═════════════════════════════════════════════════════════════════════════════

async def sync_ratings(fetcher: StealthFetcher):
    """Backfill post-match player ratings for FT matches."""
    pool = await get_db_pool()

    # FT matches with ref but no rating history (last 7 days)
    rows = await pool.fetch("""
        SELECT m.id as match_id, mer.source_match_id as event_id,
               m.date as match_date, m.league_id
        FROM matches m
        JOIN match_external_refs mer ON mer.match_id = m.id AND mer.source = 'sofascore'
        WHERE m.status IN ('FT', 'AET', 'PEN')
          AND m.date > NOW() - INTERVAL '7 days'
          -- Jitter dinámico: Esperar entre 25 mins y ~4 horas DESPUÉS de que termina el partido
          AND m.date + (140 + (m.id % 215)) * INTERVAL '1 minute' < NOW()
          AND NOT EXISTS (
              SELECT 1 FROM sofascore_player_rating_history h
              WHERE h.match_id = m.id
          )
        ORDER BY m.date DESC
        LIMIT 100
    """)

    logger.info("[RATINGS] %d FT matches to backfill", len(rows))
    updated = 0
    errors = 0
    now = datetime.utcnow()

    for row in rows:
        event_id = row["event_id"]
        data = await fetcher.fetch_api(f"/event/{event_id}/lineups")
        if not data:
            errors += 1
            if fetcher.consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                logger.error("[RATINGS] Too many consecutive errors, stopping")
                break
            continue

        parsed = parse_lineup_data(data)
        match_ratings = 0

        for side in ("home", "away"):
            team = parsed.get(side)
            if not team or not team.get("players"):
                continue
            for p in team["players"]:
                rating = p.get("rating")
                if not rating:
                    continue
                await pool.execute("""
                    INSERT INTO sofascore_player_rating_history (
                        player_id_ext, match_id, team_side, position,
                        rating, is_starter, match_date, captured_at
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                    ON CONFLICT (player_id_ext, match_id) DO UPDATE SET
                        rating = EXCLUDED.rating,
                        captured_at = EXCLUDED.captured_at
                """,
                    p["player_id_ext"], row["match_id"], side, p["position"],
                    rating, p["is_starter"], row["match_date"], now,
                )
                match_ratings += 1

        if match_ratings > 0:
            updated += 1
            logger.info("  Ratings captured for match %d: %d players", row["match_id"], match_ratings)

    await pool.close()
    logger.info("[RATINGS] Complete: matches=%d, errors=%d", updated, errors)


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

async def main():
    parser = argparse.ArgumentParser(description="Sofascore Stealth Sync")
    parser.add_argument("--test", action="store_true", help="Test stealth approach")
    parser.add_argument("--refs", action="store_true", help="Sync refs (scheduled events)")
    parser.add_argument("--stats", action="store_true", help="Backfill stats for FT matches")
    parser.add_argument("--lineups", action="store_true", help="Capture lineups for NS matches")
    parser.add_argument("--ratings", action="store_true", help="Backfill ratings for FT matches")
    parser.add_argument("--all", action="store_true", help="Run all sync modes")
    parser.add_argument("--headed", action="store_true", help="Run with visible browser (debug)")
    args = parser.parse_args()

    if not any([args.test, args.refs, args.stats, args.lineups, args.ratings, args.all]):
        parser.print_help()
        return

    fetcher = StealthFetcher(headless=not args.headed)
    await fetcher.start()

    try:
        if args.test:
            await run_test(fetcher)

        if args.refs or args.all:
            await sync_refs(fetcher)

        if args.stats or args.all:
            await sync_stats(fetcher)

        if args.lineups or args.all:
            await sync_lineups(fetcher)

        if args.ratings or args.all:
            await sync_ratings(fetcher)

    finally:
        await fetcher.close()


if __name__ == "__main__":
    asyncio.run(main())
