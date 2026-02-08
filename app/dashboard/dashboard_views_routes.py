"""Dashboard Views API — PIT, TITAN, feature coverage, tables, predictions, analytics.

23 endpoints under /dashboard/* (various paths, no single prefix).
All protected by dashboard token auth.
Extracted from main.py Step 4b.
"""

import json
import logging
import os
import time
from datetime import datetime, timedelta
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from sqlalchemy import column, func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.database import AsyncSessionLocal, get_async_session
from app.etl.sota_constants import SOFASCORE_SUPPORTED_LEAGUES, UNDERSTAT_SUPPORTED_LEAGUES
from app.models import (
    JobRun, Match, OddsHistory, PITReport, PostMatchAudit,
    Prediction, PredictionOutcome, SensorPrediction, ShadowPrediction, Team,
)
from app.security import verify_dashboard_token_bool
from app.state import ml_engine

router = APIRouter(tags=["dashboard-views"])

logger = logging.getLogger(__name__)
settings = get_settings()


def _check_token(request: Request):
    """Verify dashboard token (header + session + query param dev)."""
    if not verify_dashboard_token_bool(request):
        raise HTTPException(status_code=401, detail="Dashboard access requires valid token.")


# =============================================================================
# PIT DASHBOARD (Minimalista - No DB queries)
# =============================================================================

# In-memory cache for dashboard data
_pit_dashboard_cache = {
    "data": None,
    "timestamp": 0,
    "ttl": 45,  # 45 seconds cache
}


def _load_latest_pit_report() -> dict:
    """
    Load the most recent PIT report from DB first, then filesystem fallback.

    Priority: DB (pit_reports table) > filesystem (logs/)
    This ensures reports survive Railway deploys.
    """
    import os
    from glob import glob

    result = {
        "weekly": None,
        "daily": None,
        "source": None,
        "error": None,
        "report_date": None,
        "created_at": None,
    }

    # Try DB first (persistent across deploys)
    try:
        db_result = _load_pit_reports_from_db()
        if db_result.get("daily") or db_result.get("weekly"):
            return db_result
    except Exception as e:
        # Log but don't fail - fall through to filesystem
        logger.debug(f"DB PIT lookup failed (will try filesystem): {e}")

    # Fallback to filesystem (for backwards compatibility)
    logs_dir = "logs"
    if not os.path.exists(logs_dir):
        result["error"] = "No PIT reports found (DB empty, no logs directory)"
        return result

    # Find latest weekly report
    weekly_files = glob(f"{logs_dir}/pit_weekly_*.json")
    if weekly_files:
        latest_weekly = max(weekly_files, key=os.path.getmtime)
        try:
            import json
            with open(latest_weekly) as f:
                result["weekly"] = json.load(f)
                result["weekly"]["_file"] = os.path.basename(latest_weekly)
                result["source"] = "filesystem_weekly"
        except Exception as e:
            result["error"] = f"Error reading weekly: {e}"

    # Find latest daily report (legacy fallbacks)
    # - pit_evaluation_live_only_*.json (current script)
    # - pit_evaluation_*.json (older/legacy)
    daily_files = glob(f"{logs_dir}/pit_evaluation_live_only_*.json") + glob(f"{logs_dir}/pit_evaluation_*.json")
    if daily_files:
        latest_daily = max(daily_files, key=os.path.getmtime)
        try:
            import json
            with open(latest_daily) as f:
                result["daily"] = json.load(f)
                result["daily"]["_file"] = os.path.basename(latest_daily)
                if not result["weekly"]:
                    result["source"] = "filesystem_daily"
        except Exception as e:
            if not result["error"]:
                result["error"] = f"Error reading daily: {e}"

    if not result["weekly"] and not result["daily"]:
        result["error"] = "No PIT reports found"

    return result


async def _load_pit_reports_from_db_async() -> dict:
    """
    Load latest PIT reports from pit_reports table (async version).
    Returns dict with weekly/daily payloads and metadata.
    """
    from sqlalchemy import text

    result = {
        "weekly": None,
        "daily": None,
        "source": None,
        "error": None,
        "report_date": None,
        "created_at": None,
    }

    try:
        async with AsyncSessionLocal() as session:
            # Get latest daily - process result INSIDE the session context
            daily_result = await session.execute(text("""
                SELECT payload, report_date, created_at, source
                FROM pit_reports
                WHERE report_type = 'daily'
                ORDER BY report_date DESC
                LIMIT 1
            """))
            daily_row = daily_result.fetchone()

            # Extract daily data while still in session
            if daily_row:
                result["daily"] = daily_row[0]  # payload is JSON
                result["report_date"] = str(daily_row[1])
                result["created_at"] = str(daily_row[2])
                result["source"] = f"db_{daily_row[3]}"

            # Get latest weekly
            weekly_result = await session.execute(text("""
                SELECT payload, report_date, created_at, source
                FROM pit_reports
                WHERE report_type = 'weekly'
                ORDER BY report_date DESC
                LIMIT 1
            """))
            weekly_row = weekly_result.fetchone()

            # Extract weekly data while still in session
            if weekly_row:
                result["weekly"] = weekly_row[0]
                if not result["source"]:
                    result["report_date"] = str(weekly_row[1])
                    result["created_at"] = str(weekly_row[2])
                    result["source"] = f"db_{weekly_row[3]}"

        # Check after session closes
        if not result["weekly"] and not result["daily"]:
            result["error"] = "No PIT reports in database"

    except Exception as e:
        logger.warning(f"PIT DB load error: {e}")
        result["error"] = f"DB error: {str(e)[:100]}"

    return result


def _load_pit_reports_from_db() -> dict:
    """
    Sync wrapper for _load_pit_reports_from_db_async.
    Used by cached functions that need sync interface.
    """
    import asyncio
    try:
        # Try to get existing event loop
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # We're inside an async context, can't use run_until_complete
            # Return empty to fall through to filesystem fallback
            return {"error": "Cannot run sync in async context"}
        return loop.run_until_complete(_load_pit_reports_from_db_async())
    except RuntimeError:
        # No event loop exists, create new one
        return asyncio.run(_load_pit_reports_from_db_async())


async def _get_cached_pit_data_async() -> dict:
    """Get PIT data with caching (async version for DB access)."""
    now = time.time()
    if _pit_dashboard_cache["data"] and (now - _pit_dashboard_cache["timestamp"]) < _pit_dashboard_cache["ttl"]:
        return _pit_dashboard_cache["data"]

    # Try DB first (async)
    data = await _load_pit_reports_from_db_async()
    if data.get("daily") or data.get("weekly"):
        _pit_dashboard_cache["data"] = data
        _pit_dashboard_cache["timestamp"] = now
        return data

    # Fallback to filesystem
    data = _load_latest_pit_report()
    _pit_dashboard_cache["data"] = data
    _pit_dashboard_cache["timestamp"] = now
    return data


def _get_cached_pit_data() -> dict:
    """Get PIT data with caching (sync fallback, uses filesystem only)."""
    now = time.time()
    if _pit_dashboard_cache["data"] and (now - _pit_dashboard_cache["timestamp"]) < _pit_dashboard_cache["ttl"]:
        return _pit_dashboard_cache["data"]

    data = _load_latest_pit_report()
    _pit_dashboard_cache["data"] = data
    _pit_dashboard_cache["timestamp"] = now
    return data


# _has_valid_session, _verify_dashboard_token, _get_dashboard_token_from_request,
# _verify_debug_token — all imported from app.security (see imports at top)






@router.get("/dashboard/pit.json")
async def pit_dashboard_json(request: Request):
    """
    PIT Dashboard JSON - Raw data for programmatic access.

    Returns the latest weekly/daily PIT report data.
    """
    if not verify_dashboard_token_bool(request):
        raise HTTPException(
            status_code=401,
            detail="Dashboard access requires valid token.",
        )

    data = await _get_cached_pit_data_async()
    return {
        "source": data.get("source"),
        "error": data.get("error"),
        "report_date": data.get("report_date"),
        "created_at": data.get("created_at"),
        "weekly": data.get("weekly"),
        "daily": data.get("daily"),
        "cache_age_seconds": round(time.time() - _pit_dashboard_cache["timestamp"], 1) if _pit_dashboard_cache["timestamp"] else None,
    }


# =============================================================================
# TITAN OMNISCIENCE Dashboard
# =============================================================================


@router.get("/dashboard/titan.json")
async def titan_dashboard_json(request: Request):
    """
    TITAN OMNISCIENCE Dashboard JSON - Enterprise scraping status.

    Returns comprehensive TITAN operational status including:
    - Extraction counts and coverage
    - DLQ (Dead Letter Queue) status
    - PIT (Point-in-Time) compliance metrics
    - Feature matrix statistics

    Protected by X-Dashboard-Token (same auth as /dashboard/ops.json).
    """
    if not verify_dashboard_token_bool(request):
        raise HTTPException(
            status_code=401,
            detail="Dashboard access requires valid token.",
        )

    from app.titan.dashboard import get_titan_status

    async with AsyncSessionLocal() as session:
        return await get_titan_status(session)


# =============================================================================
# Feature Coverage Matrix (SOTA Dashboard)
# =============================================================================
# Cache for feature coverage matrix (expensive query, TTL 30 min)
_feature_coverage_cache = {
    "data": None,
    "timestamp": 0,
    "ttl": 1800,  # 30 minutes
}


@router.get("/dashboard/feature-coverage.json")
async def dashboard_feature_coverage_json(request: Request):
    """
    Feature Coverage Matrix - Shows % of non-NULL features by league and season.

    Used by SOTA dashboard to identify which leagues have sufficient data quality
    for ML training (avoid imputing missing values with 0).

    Windows:
    - 23/24: 2023-08-01 to 2024-07-31
    - 24/25: 2024-08-01 to 2025-07-31
    - 25/26: 2025-08-01 to 2026-07-31 (current season)

    Features (30 total):
    - Tier 1 (14): Core features from public.matches [PROD]
    - Tier 1b (6): xG features from titan.feature_matrix [TITAN]
    - Tier 1c (4): Lineup features from titan.feature_matrix [TITAN]
    - Tier 1d (6): XI Depth features from titan.feature_matrix [TITAN]

    Auth: X-Dashboard-Token header.
    Cache: 30 minutes TTL.
    """
    if not verify_dashboard_token_bool(request):
        raise HTTPException(
            status_code=401,
            detail="Dashboard access requires valid token.",
        )

    now = time.time()
    cached = False
    cache_age = None

    # Check cache
    if _feature_coverage_cache["data"] and (now - _feature_coverage_cache["timestamp"]) < _feature_coverage_cache["ttl"]:
        cached = True
        cache_age = round(now - _feature_coverage_cache["timestamp"], 1)
        return {
            "generated_at": _feature_coverage_cache["data"]["generated_at"],
            "cached": cached,
            "cache_age_seconds": cache_age,
            "data": _feature_coverage_cache["data"]["data"],
        }

    # Calculate fresh data
    async with AsyncSessionLocal() as session:
        data = await _calculate_feature_coverage(session)

    # Update cache
    _feature_coverage_cache["data"] = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "data": data,
    }
    _feature_coverage_cache["timestamp"] = now

    return {
        "generated_at": _feature_coverage_cache["data"]["generated_at"],
        "cached": False,
        "cache_age_seconds": None,
        "data": data,
    }


async def _calculate_feature_coverage(session) -> dict:
    """
    Calculate feature coverage matrix for all leagues and windows.

    Returns structured data matching the frontend contract.
    """
    from app.etl.competitions import COMPETITIONS

    # Define windows
    windows = [
        {"key": "23/24", "from": "2023-08-01", "to": "2024-07-31"},
        {"key": "24/25", "from": "2024-08-01", "to": "2025-07-31"},
        {"key": "25/26", "from": "2025-08-01", "to": "2026-07-31"},
    ]

    # Define tiers
    tiers = [
        {"id": "tier1", "label": "[PROD] Tier 1 - Core", "badge": "PROD"},
        {"id": "tier1b", "label": "[TITAN] Tier 1b - xG", "badge": "TITAN"},
        {"id": "tier1c", "label": "[TITAN] Tier 1c - Lineup", "badge": "TITAN"},
        {"id": "tier1d", "label": "[TITAN] Tier 1d - XI Depth", "badge": "TITAN"},
    ]

    # Define features with tier mapping
    features = [
        # Tier 1 - Core (calculated from matches data)
        {"key": "home_goals_scored_avg", "tier_id": "tier1", "badge": "PROD", "source": "public.matches"},
        {"key": "home_goals_conceded_avg", "tier_id": "tier1", "badge": "PROD", "source": "public.matches"},
        {"key": "home_shots_avg", "tier_id": "tier1", "badge": "PROD", "source": "public.matches.stats"},
        {"key": "home_corners_avg", "tier_id": "tier1", "badge": "PROD", "source": "public.matches.stats"},
        {"key": "home_rest_days", "tier_id": "tier1", "badge": "PROD", "source": "public.matches"},
        {"key": "home_matches_played", "tier_id": "tier1", "badge": "PROD", "source": "public.matches"},
        {"key": "away_goals_scored_avg", "tier_id": "tier1", "badge": "PROD", "source": "public.matches"},
        {"key": "away_goals_conceded_avg", "tier_id": "tier1", "badge": "PROD", "source": "public.matches"},
        {"key": "away_shots_avg", "tier_id": "tier1", "badge": "PROD", "source": "public.matches.stats"},
        {"key": "away_corners_avg", "tier_id": "tier1", "badge": "PROD", "source": "public.matches.stats"},
        {"key": "away_rest_days", "tier_id": "tier1", "badge": "PROD", "source": "public.matches"},
        {"key": "away_matches_played", "tier_id": "tier1", "badge": "PROD", "source": "public.matches"},
        {"key": "goal_diff_avg", "tier_id": "tier1", "badge": "PROD", "source": "derived"},
        {"key": "rest_diff", "tier_id": "tier1", "badge": "PROD", "source": "derived"},
        # Tier 1b - xG
        {"key": "xg_home_last5", "tier_id": "tier1b", "badge": "TITAN", "source": "titan.feature_matrix"},
        {"key": "xga_home_last5", "tier_id": "tier1b", "badge": "TITAN", "source": "titan.feature_matrix"},
        {"key": "npxg_home_last5", "tier_id": "tier1b", "badge": "TITAN", "source": "titan.feature_matrix"},
        {"key": "xg_away_last5", "tier_id": "tier1b", "badge": "TITAN", "source": "titan.feature_matrix"},
        {"key": "xga_away_last5", "tier_id": "tier1b", "badge": "TITAN", "source": "titan.feature_matrix"},
        {"key": "npxg_away_last5", "tier_id": "tier1b", "badge": "TITAN", "source": "titan.feature_matrix"},
        # Tier 1c - Lineup
        {"key": "sofascore_home_formation", "tier_id": "tier1c", "badge": "TITAN", "source": "titan.feature_matrix"},
        {"key": "sofascore_away_formation", "tier_id": "tier1c", "badge": "TITAN", "source": "titan.feature_matrix"},
        {"key": "lineup_home_starters_count", "tier_id": "tier1c", "badge": "TITAN", "source": "titan.feature_matrix"},
        {"key": "lineup_away_starters_count", "tier_id": "tier1c", "badge": "TITAN", "source": "titan.feature_matrix"},
        # Tier 1d - XI Depth
        {"key": "xi_home_def_count", "tier_id": "tier1d", "badge": "TITAN", "source": "titan.feature_matrix"},
        {"key": "xi_home_mid_count", "tier_id": "tier1d", "badge": "TITAN", "source": "titan.feature_matrix"},
        {"key": "xi_home_fwd_count", "tier_id": "tier1d", "badge": "TITAN", "source": "titan.feature_matrix"},
        {"key": "xi_away_def_count", "tier_id": "tier1d", "badge": "TITAN", "source": "titan.feature_matrix"},
        {"key": "xi_away_mid_count", "tier_id": "tier1d", "badge": "TITAN", "source": "titan.feature_matrix"},
        {"key": "xi_away_fwd_count", "tier_id": "tier1d", "badge": "TITAN", "source": "titan.feature_matrix"},
    ]

    # Build leagues list from COMPETITIONS
    leagues = [
        {"league_id": comp.league_id, "name": comp.name}
        for comp in COMPETITIONS.values()
    ]

    # =========================================================================
    # Query 1: Get match counts and Tier 1 coverage per league/window
    # =========================================================================
    tier1_query = text("""
        WITH match_counts AS (
            SELECT
                league_id,
                CASE
                    WHEN date >= '2023-08-01' AND date < '2024-08-01' THEN '23/24'
                    WHEN date >= '2024-08-01' AND date < '2025-08-01' THEN '24/25'
                    WHEN date >= '2025-08-01' AND date < '2026-08-01' THEN '25/26'
                END as time_window,
                COUNT(*) as total,
                -- Goals are always available for FT matches
                COUNT(*) as with_goals,
                -- Stats JSON with shots/corners (check if total_shots exists)
                COUNT(*) FILTER (WHERE
                    stats IS NOT NULL
                    AND stats::text NOT IN ('null', '{}', '')
                    AND (stats->'home'->>'total_shots') IS NOT NULL
                ) as with_stats
            FROM matches
            WHERE status = 'FT'
              AND date >= '2023-08-01'
              AND date < '2026-08-01'
            GROUP BY league_id, time_window
        )
        SELECT * FROM match_counts WHERE time_window IS NOT NULL
        ORDER BY league_id, time_window
    """)

    tier1_result = await session.execute(tier1_query)
    tier1_rows = tier1_result.fetchall()

    # Build tier1 data structure
    tier1_data = {}  # {league_id: {window: {total, with_goals, with_stats}}}
    for row in tier1_rows:
        lid = row.league_id
        win = row.time_window
        if lid not in tier1_data:
            tier1_data[lid] = {}
        tier1_data[lid][win] = {
            "total": int(row.total),
            "with_goals": int(row.with_goals),
            "with_stats": int(row.with_stats),
        }

    # =========================================================================
    # Query 2: Get TITAN feature coverage per league/window
    # =========================================================================
    titan_query = text("""
        SELECT
            competition_id as league_id,
            CASE
                WHEN kickoff_utc >= '2023-08-01' AND kickoff_utc < '2024-08-01' THEN '23/24'
                WHEN kickoff_utc >= '2024-08-01' AND kickoff_utc < '2025-08-01' THEN '24/25'
                WHEN kickoff_utc >= '2025-08-01' AND kickoff_utc < '2026-08-01' THEN '25/26'
            END as time_window,
            COUNT(*) as total,
            -- Tier 1b: xG features
            COUNT(*) FILTER (WHERE xg_home_last5 IS NOT NULL) as xg_home_last5,
            COUNT(*) FILTER (WHERE xga_home_last5 IS NOT NULL) as xga_home_last5,
            COUNT(*) FILTER (WHERE npxg_home_last5 IS NOT NULL) as npxg_home_last5,
            COUNT(*) FILTER (WHERE xg_away_last5 IS NOT NULL) as xg_away_last5,
            COUNT(*) FILTER (WHERE xga_away_last5 IS NOT NULL) as xga_away_last5,
            COUNT(*) FILTER (WHERE npxg_away_last5 IS NOT NULL) as npxg_away_last5,
            -- Tier 1c: Lineup features
            COUNT(*) FILTER (WHERE sofascore_home_formation IS NOT NULL) as sofascore_home_formation,
            COUNT(*) FILTER (WHERE sofascore_away_formation IS NOT NULL) as sofascore_away_formation,
            COUNT(*) FILTER (WHERE lineup_home_starters_count IS NOT NULL) as lineup_home_starters_count,
            COUNT(*) FILTER (WHERE lineup_away_starters_count IS NOT NULL) as lineup_away_starters_count,
            -- Tier 1d: XI Depth features
            COUNT(*) FILTER (WHERE xi_home_def_count IS NOT NULL) as xi_home_def_count,
            COUNT(*) FILTER (WHERE xi_home_mid_count IS NOT NULL) as xi_home_mid_count,
            COUNT(*) FILTER (WHERE xi_home_fwd_count IS NOT NULL) as xi_home_fwd_count,
            COUNT(*) FILTER (WHERE xi_away_def_count IS NOT NULL) as xi_away_def_count,
            COUNT(*) FILTER (WHERE xi_away_mid_count IS NOT NULL) as xi_away_mid_count,
            COUNT(*) FILTER (WHERE xi_away_fwd_count IS NOT NULL) as xi_away_fwd_count
        FROM titan.feature_matrix
        WHERE kickoff_utc >= '2023-08-01' AND kickoff_utc < '2026-08-01'
        GROUP BY competition_id, time_window
        HAVING CASE
            WHEN kickoff_utc >= '2023-08-01' AND kickoff_utc < '2024-08-01' THEN '23/24'
            WHEN kickoff_utc >= '2024-08-01' AND kickoff_utc < '2025-08-01' THEN '24/25'
            WHEN kickoff_utc >= '2025-08-01' AND kickoff_utc < '2026-08-01' THEN '25/26'
        END IS NOT NULL
        ORDER BY competition_id, time_window
    """)

    try:
        titan_result = await session.execute(titan_query)
        titan_rows = titan_result.fetchall()
    except Exception as e:
        logger.warning(f"TITAN feature_matrix query failed (schema may not exist): {e}")
        titan_rows = []

    # Build titan data structure
    titan_data = {}  # {league_id: {window: {feature: count}}}
    titan_features = [
        "xg_home_last5", "xga_home_last5", "npxg_home_last5",
        "xg_away_last5", "xga_away_last5", "npxg_away_last5",
        "sofascore_home_formation", "sofascore_away_formation",
        "lineup_home_starters_count", "lineup_away_starters_count",
        "xi_home_def_count", "xi_home_mid_count", "xi_home_fwd_count",
        "xi_away_def_count", "xi_away_mid_count", "xi_away_fwd_count",
    ]

    for row in titan_rows:
        lid = row.league_id
        win = row.time_window
        if lid not in titan_data:
            titan_data[lid] = {}
        titan_data[lid][win] = {"total": int(row.total)}
        for feat in titan_features:
            titan_data[lid][win][feat] = int(getattr(row, feat, 0) or 0)

    # =========================================================================
    # Build coverage and league_summaries structures
    # =========================================================================
    coverage = {}  # {feature_key: {league_id: {window: {pct, n}}}}
    league_summaries = {}  # {league_id: {window: {matches_total, avg_pct}}}

    # Initialize coverage structure
    for feat in features:
        coverage[feat["key"]] = {}

    # Process each league
    for league in leagues:
        lid = league["league_id"]
        league_summaries[str(lid)] = {}

        for win_def in windows:
            win = win_def["key"]

            # Get base match count from tier1_data
            t1 = tier1_data.get(lid, {}).get(win, {"total": 0, "with_goals": 0, "with_stats": 0})
            matches_total = t1["total"]

            # Calculate coverage for each feature
            feature_pcts = []

            for feat in features:
                fkey = feat["key"]

                if str(lid) not in coverage[fkey]:
                    coverage[fkey][str(lid)] = {}

                if matches_total == 0:
                    pct = 0.0
                    n = 0
                elif feat["tier_id"] == "tier1":
                    # Tier 1 features - use tier1_data
                    if fkey in ["home_shots_avg", "away_shots_avg", "home_corners_avg", "away_corners_avg"]:
                        # Stats-dependent features
                        n = t1["with_stats"]
                    else:
                        # Goals/rest/matches_played - always available for FT matches
                        n = t1["with_goals"]
                    pct = round(100.0 * n / matches_total, 1)
                else:
                    # Tier 1b/1c/1d - use titan_data
                    titan_win = titan_data.get(lid, {}).get(win, {})
                    # Use titan total as denominator if available, else matches_total
                    titan_total = titan_win.get("total", 0)
                    if titan_total > 0:
                        n = titan_win.get(fkey, 0)
                        pct = round(100.0 * n / titan_total, 1)
                    else:
                        n = 0
                        pct = 0.0

                coverage[fkey][str(lid)][win] = {"pct": pct, "n": n}
                feature_pcts.append(pct)

            # Get titan total for this window (for transparency in league_summaries)
            titan_win = titan_data.get(lid, {}).get(win, {})
            titan_total = titan_win.get("total", 0)

            # Calculate league summary (avg across all 30 features)
            avg_pct = round(sum(feature_pcts) / len(feature_pcts), 1) if feature_pcts else 0.0
            league_summaries[str(lid)][win] = {
                "matches_total_ft": matches_total,
                "matches_total_titan": titan_total,
                "avg_pct": avg_pct,
            }

        # Calculate total (combined windows) with correct denominators per tier
        # Tier 1: uses FT matches as denominator
        total_matches_ft = sum(
            league_summaries[str(lid)].get(w["key"], {}).get("matches_total_ft", 0)
            for w in windows
        )
        # TITAN tiers: uses titan.feature_matrix rows as denominator
        total_matches_titan = sum(
            titan_data.get(lid, {}).get(w["key"], {}).get("total", 0)
            for w in windows
        )

        total_pcts = []
        for feat in features:
            fkey = feat["key"]
            total_n = sum(
                coverage[fkey].get(str(lid), {}).get(w["key"], {}).get("n", 0)
                for w in windows
            )

            # Use correct denominator based on tier (ABE fix: avoid mixing denominators)
            if feat["tier_id"] == "tier1":
                denominator = total_matches_ft
            else:
                denominator = total_matches_titan

            total_pct = round(100.0 * total_n / denominator, 1) if denominator > 0 else 0.0
            coverage[fkey][str(lid)]["total"] = {"pct": total_pct, "n": total_n}
            total_pcts.append(total_pct)

        league_summaries[str(lid)]["total"] = {
            "matches_total_ft": total_matches_ft,
            "matches_total_titan": total_matches_titan,
            "avg_pct": round(sum(total_pcts) / len(total_pcts), 1) if total_pcts else 0.0,
        }

    return {
        "windows": windows,
        "tiers": tiers,
        "features": features,
        "leagues": leagues,
        "league_summaries": league_summaries,
        "coverage": coverage,
    }


# =============================================================================
# ML Health Dashboard (ATI v1.1)
# =============================================================================
# Cache for ML health (60s TTL - early warning for pipeline issues)
_ml_health_cache = {
    "data": None,
    "timestamp": 0,
    "ttl": 60,  # 60 seconds
    "previous_snapshot": None,  # For future top_regressions calculation
    "previous_snapshot_at": None,
}


@router.get("/dashboard/ml_health.json")
async def dashboard_ml_health_json(request: Request):
    """
    ML Health Dashboard - Comprehensive ML pipeline health metrics.

    ATI v1.1 - Addresses "vuelo a ciegas" finding where XGBoost ran
    with 0% coverage for shots/corners in 23/24 season.

    Returns:
    - fuel_gauge: Overall health status (ok/warn/error) with reasons
    - sota_stats_coverage: Stats coverage by season and league (P0 - root cause)
    - titan_coverage: TITAN feature_matrix coverage by tier
    - pit_compliance: Point-in-Time violations
    - freshness: Data staleness (age_hours_now for early warning)
    - prediction_confidence: Entropy and tier distribution
    - top_regressions: Not implemented yet (requires baseline)

    Fail-soft: If any section fails, returns partial data with health="partial"

    Auth: X-Dashboard-Token header.
    """
    import time

    if not verify_dashboard_token_bool(request):
        raise HTTPException(
            status_code=401,
            detail="Dashboard access requires valid token.",
        )

    now = time.time()

    # Check cache
    if _ml_health_cache["data"] and (now - _ml_health_cache["timestamp"]) < _ml_health_cache["ttl"]:
        cached_data = _ml_health_cache["data"]
        return {
            "generated_at": cached_data["generated_at"],
            "cached": True,
            "cache_age_seconds": round(now - _ml_health_cache["timestamp"], 1),
            "health": cached_data["health"],
            "data": cached_data["data"],
        }

    # Build fresh data
    from app.ml.health import build_ml_health_data

    async with AsyncSessionLocal() as session:
        result = await build_ml_health_data(session)

    # Update cache + snapshot for future regressions
    _ml_health_cache["previous_snapshot"] = _ml_health_cache["data"]
    _ml_health_cache["previous_snapshot_at"] = _ml_health_cache["timestamp"]
    _ml_health_cache["data"] = result
    _ml_health_cache["timestamp"] = now

    return {
        "generated_at": result["generated_at"],
        "cached": False,
        "cache_age_seconds": None,
        "health": result["health"],
        "data": result["data"],
    }



# Admin Panel endpoints moved to app/dashboard/admin_routes.py

# Football Navigation API (P3) moved to app/dashboard/football_routes.py


@router.get("/dashboard/pit/debug")
async def pit_dashboard_debug(request: Request):
    """
    Debug endpoint - shows raw pit_reports table content.
    Protected by dashboard token.
    """
    if not verify_dashboard_token_bool(request):
        raise HTTPException(status_code=401, detail="Dashboard access requires valid token.")

    from sqlalchemy import text

    result = {
        "table_contents": [],
        "weekly_count": 0,
        "daily_count": 0,
        "error": None,
    }

    try:
        async with AsyncSessionLocal() as session:
            # Get all rows
            rows_result = await session.execute(text("""
                SELECT id, report_type, report_date, source, created_at, updated_at,
                       LENGTH(payload::text) as payload_size
                FROM pit_reports
                ORDER BY report_date DESC, report_type DESC
                LIMIT 20
            """))
            rows = rows_result.fetchall()

            result["table_contents"] = [
                {
                    "id": row[0],
                    "report_type": row[1],
                    "report_date": str(row[2]),
                    "source": row[3],
                    "created_at": str(row[4]),
                    "updated_at": str(row[5]),
                    "payload_size": row[6],
                }
                for row in rows
            ]

            # Counts
            daily_result = await session.execute(text("SELECT COUNT(*) FROM pit_reports WHERE report_type='daily'"))
            result["daily_count"] = daily_result.scalar()

            weekly_result = await session.execute(text("SELECT COUNT(*) FROM pit_reports WHERE report_type='weekly'"))
            result["weekly_count"] = weekly_result.scalar()

    except Exception as e:
        result["error"] = str(e)

    return result


@router.get("/dashboard/debug/experiment-gating/{match_id}")
async def debug_experiment_gating(
    match_id: int,
    request: Request,
    variant: str = Query("A", regex="^[ABCD]$"),
    session: AsyncSession = Depends(get_async_session),
):
    """
    Debug endpoint for ext-A/B/C/D experiment gating. Read-only.

    Explains why a match does/doesn't have an ext prediction.
    Uses EXACT same gating logic as the job (strict inequalities).

    ATI: Observability endpoint - no side effects.
    """
    if not verify_dashboard_token_bool(request):
        raise HTTPException(status_code=401, detail="Dashboard access requires valid token.")

    from pathlib import Path

    # Config por variante
    variant_config = {
        "A": (settings.EXTA_SHADOW_ENABLED, settings.EXTA_SHADOW_MODEL_VERSION, settings.EXTA_SHADOW_MODEL_PATH),
        "B": (settings.EXTB_SHADOW_ENABLED, settings.EXTB_SHADOW_MODEL_VERSION, settings.EXTB_SHADOW_MODEL_PATH),
        "C": (settings.EXTC_SHADOW_ENABLED, settings.EXTC_SHADOW_MODEL_VERSION, settings.EXTC_SHADOW_MODEL_PATH),
        "D": (settings.EXTD_SHADOW_ENABLED, settings.EXTD_SHADOW_MODEL_VERSION, settings.EXTD_SHADOW_MODEL_PATH),
    }
    enabled, model_version, model_path = variant_config[variant]
    start_at = settings.EXT_SHADOW_START_AT

    checks = []
    failure_reason = None

    # 1. Get match info
    match_result = await session.execute(text("""
        SELECT m.id, m.date as kickoff_at, m.status, ht.name as home_team, at.name as away_team
        FROM matches m
        LEFT JOIN teams ht ON m.home_team_id = ht.id
        LEFT JOIN teams at ON m.away_team_id = at.id
        WHERE m.id = :match_id
    """), {"match_id": match_id})
    match_row = match_result.fetchone()
    if not match_row:
        raise HTTPException(status_code=404, detail=f"Match {match_id} not found")

    kickoff_at = match_row[1]

    # 2. Check lineup_confirmed snapshot exists
    snapshot_result = await session.execute(text("""
        SELECT os.id, os.snapshot_at FROM odds_snapshots os
        WHERE os.match_id = :match_id AND os.snapshot_type = 'lineup_confirmed'
        ORDER BY os.snapshot_at DESC LIMIT 1
    """), {"match_id": match_id})
    snapshot_row = snapshot_result.fetchone()

    has_lineup = snapshot_row is not None
    checks.append({"check": "lineup_confirmed_exists", "status": "PASS" if has_lineup else "FAIL"})
    if not has_lineup:
        failure_reason = "no_lineup_confirmed"

    snapshot_id = snapshot_row[0] if has_lineup else None
    snapshot_at = snapshot_row[1] if has_lineup else None
    delta_minutes = None
    has_pred = None

    if has_lineup:
        delta_minutes = (kickoff_at - snapshot_at).total_seconds() / 60

        # 3. Check snapshot_after_start_at
        start_dt = datetime.fromisoformat(start_at)
        is_after_start = snapshot_at >= start_dt
        checks.append({
            "check": "snapshot_after_start_at",
            "status": "PASS" if is_after_start else "FAIL",
            "detail": f"snapshot_at={snapshot_at.isoformat()}, start_at={start_at}"
        })
        if not is_after_start and not failure_reason:
            failure_reason = "snapshot_before_start_at"

        # 4. Check window (STRICT: > 10 and < 90, matching job logic)
        # Job uses: m.date > os.snapshot_at + INTERVAL '10 minutes'
        #           m.date < os.snapshot_at + INTERVAL '90 minutes'
        in_window = delta_minutes > 10 and delta_minutes < 90
        window_detail = f"delta={round(delta_minutes, 1)}min"
        if delta_minutes <= 10:
            window_detail += " (<=10min, too late)"
        elif delta_minutes >= 90:
            window_detail += " (>=90min, too early)"
        checks.append({
            "check": "window_10_90_min_strict",
            "status": "PASS" if in_window else "FAIL",
            "detail": window_detail
        })
        if not in_window and not failure_reason:
            failure_reason = "outside_window_too_late" if delta_minutes <= 10 else "outside_window_too_early"

        # 5. Check model exists
        model_exists = Path(model_path).exists()
        checks.append({
            "check": "model_exists",
            "status": "PASS" if model_exists else "FAIL",
            "detail": f"path={model_path}"
        })
        if not model_exists and not failure_reason:
            failure_reason = "model_not_found"

        # 6. Check existing prediction (PIT-safe: by match_id + model_version with snapshot_at <= kickoff)
        # ATI FIX: No buscar por snapshot_id, buscar por match_id + model_version
        pred_result = await session.execute(text("""
            SELECT pe.id, pe.snapshot_at
            FROM predictions_experiments pe
            WHERE pe.match_id = :match_id
              AND pe.model_version = :model_version
              AND pe.snapshot_at <= :kickoff_at
            ORDER BY pe.snapshot_at DESC
            LIMIT 1
        """), {"match_id": match_id, "model_version": model_version, "kickoff_at": kickoff_at})
        pred_row = pred_result.fetchone()
        has_pred = pred_row is not None
        checks.append({
            "check": "has_pit_safe_prediction",
            "status": "PASS" if has_pred else "FAIL",
            "detail": f"prediction_snapshot_at={pred_row[1].isoformat() if has_pred else 'none'}"
        })

    return {
        "match_id": match_id,
        "variant": variant,
        "variant_enabled": enabled,
        "model_version": model_version,
        "kickoff_at": kickoff_at.isoformat() if kickoff_at else None,
        "match_info": {"home_team": match_row[3], "away_team": match_row[4], "status": match_row[2]},
        "latest_lineup_confirmed_snapshot_at": snapshot_at.isoformat() if snapshot_at else None,
        "snapshot_id": snapshot_id,
        "delta_minutes_to_kickoff": round(delta_minutes, 1) if delta_minutes else None,
        "start_at": start_at,
        "has_prediction_experiment": has_pred,
        "failure_reason": failure_reason,
        "checks": checks,
    }


@router.post("/dashboard/pit/trigger")
async def pit_trigger_evaluation(request: Request):
    """
    Manually trigger PIT evaluation (for testing).
    Protected by dashboard token.
    """
    if not verify_dashboard_token_bool(request):
        raise HTTPException(status_code=401, detail="Dashboard access requires valid token.")

    from app.scheduler import daily_pit_evaluation

    try:
        await daily_pit_evaluation()
        return {"status": "ok", "message": "PIT evaluation triggered"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


# =============================================================================
# DASHBOARD V2 ENDPOINTS (wrapper-compliant)
# Wrapper: { generated_at, cached, cache_age_seconds, data }
# =============================================================================

def _make_v2_wrapper(data: dict, cached: bool = False, cache_age_seconds: float = 0) -> dict:
    """Build standard v2 wrapper for dashboard endpoints."""
    return {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "cached": cached,
        "cache_age_seconds": round(cache_age_seconds, 1),
        "data": data,
    }


# Cache for /dashboard/overview/rollup.json
_rollup_cache: dict = {"data": None, "timestamp": 0, "ttl": 60}


@router.get("/dashboard/overview/rollup.json")
async def dashboard_overview_rollup(request: Request):
    """
    V2 endpoint: Overview rollup data with standard wrapper.
    TTL: 60s
    Auth: X-Dashboard-Token
    """
    if not verify_dashboard_token_bool(request):
        raise HTTPException(status_code=401, detail="Dashboard access requires valid token.")

    import time as time_module
    now_ts = time_module.time()

    # Check cache
    if _rollup_cache["data"] and (now_ts - _rollup_cache["timestamp"]) < _rollup_cache["ttl"]:
        cache_age = now_ts - _rollup_cache["timestamp"]
        return _make_v2_wrapper(_rollup_cache["data"], cached=True, cache_age_seconds=cache_age)

    # Best-effort: get ops data and extract rollup subset
    try:
        from app.dashboard.ops_routes import get_cached_ops_data  # lazy import

        ops_data = await get_cached_ops_data()

        # Extract stable rollup fields - use REAL keys from ops_data
        # Required fields (core dashboard)
        rollup = {
            "generated_at": ops_data.get("generated_at"),
            "budget": ops_data.get("budget"),
            "sentry": ops_data.get("sentry"),
            "jobs_health": ops_data.get("jobs_health"),
            "predictions_health": ops_data.get("predictions_health"),
            "fastpath_health": ops_data.get("fastpath_health"),
            "pit": ops_data.get("pit"),
            "movement": ops_data.get("movement"),
            "llm_cost": ops_data.get("llm_cost"),
            "sota_enrichment": ops_data.get("sota_enrichment"),
            "providers": ops_data.get("providers"),
        }

        # Optional fields (include if present)
        optional_keys = [
            "coverage_by_league",
            "shadow_mode",
            "sensor_b",
            "rerun_serving",
            "ml_model",
            "telemetry",
        ]
        for key in optional_keys:
            if key in ops_data and ops_data[key] is not None:
                rollup[key] = ops_data[key]

        _rollup_cache["data"] = rollup
        _rollup_cache["timestamp"] = now_ts

        return _make_v2_wrapper(rollup, cached=False, cache_age_seconds=0)

    except Exception as e:
        logger.warning(f"Rollup endpoint error: {e}")
        return _make_v2_wrapper(
            {"status": "degraded", "note": f"upstream error: {str(e)[:50]}"},
            cached=False,
            cache_age_seconds=0,
        )


# Cache for /dashboard/sentry/issues.json
_sentry_issues_cache: dict = {"data": None, "timestamp": 0, "ttl": 90, "params": None}


@router.get("/dashboard/sentry/issues.json")
async def dashboard_sentry_issues(
    request: Request,
    range: str = "24h",
    page: int = 1,
    limit: int = 50,
):
    """
    V2 endpoint: Sentry issues with standard wrapper.
    TTL: 90s
    Auth: X-Dashboard-Token
    Query params: range (24h|1h|7d), page, limit (max 100)
    """
    if not verify_dashboard_token_bool(request):
        raise HTTPException(status_code=401, detail="Dashboard access requires valid token.")

    import time as time_module
    import httpx
    from datetime import timedelta

    # Clamp limit
    limit = min(max(1, limit), 100)
    page = max(1, page)

    # Validate range
    valid_ranges = {"1h": 1, "24h": 24, "7d": 168}
    if range not in valid_ranges:
        range = "24h"
    hours = valid_ranges[range]

    now_ts = time_module.time()
    cache_key = f"{range}:{page}:{limit}"

    # Check cache (only if same params)
    if (_sentry_issues_cache["data"] and
        _sentry_issues_cache["params"] == cache_key and
        (now_ts - _sentry_issues_cache["timestamp"]) < _sentry_issues_cache["ttl"]):
        cache_age = now_ts - _sentry_issues_cache["timestamp"]
        return _make_v2_wrapper(_sentry_issues_cache["data"], cached=True, cache_age_seconds=cache_age)

    # Base response (degraded fallback)
    base_data = {
        "issues": [],
        "total": 0,
        "page": page,
        "limit": limit,
        "pages": 0,
        "status": "degraded",
        "note": "upstream unavailable",
    }

    # Check credentials
    if not settings.SENTRY_AUTH_TOKEN or not settings.SENTRY_ORG or not settings.SENTRY_PROJECT_SLUG:
        base_data["note"] = "Sentry credentials not configured"
        return _make_v2_wrapper(base_data, cached=False, cache_age_seconds=0)

    org_slug = settings.SENTRY_ORG
    project_slug = settings.SENTRY_PROJECT_SLUG
    auth_token = settings.SENTRY_AUTH_TOKEN
    env_filter = settings.SENTRY_ENV or "production"

    headers = {
        "Authorization": f"Bearer {auth_token}",
        "Content-Type": "application/json",
    }

    try:
        async with httpx.AsyncClient(headers=headers, timeout=10.0) as client:
            issues_url = f"https://sentry.io/api/0/projects/{org_slug}/{project_slug}/issues/"

            # Time boundary
            now_dt = datetime.utcnow()
            time_ago = (now_dt - timedelta(hours=hours)).strftime("%Y-%m-%dT%H:%M:%S")

            env_query = f"environment:{env_filter}" if env_filter else ""

            # Fetch issues
            params = {
                "query": f"is:unresolved {env_query}".strip(),
                "sort": "date",
                "limit": 100,  # Fetch more, then paginate
            }
            resp = await client.get(issues_url, params=params)

            if resp.status_code != 200:
                base_data["note"] = f"Sentry API returned {resp.status_code}"
                return _make_v2_wrapper(base_data, cached=False, cache_age_seconds=0)

            all_issues = resp.json() if isinstance(resp.json(), list) else []

            # Filter by time range (lastSeen within range)
            filtered_issues = [
                i for i in all_issues
                if i.get("lastSeen", "") >= time_ago
            ]

            # Build issue list (no PII, no stacktraces)
            issues = []
            for issue in filtered_issues:
                title = issue.get("title", "Unknown")[:100]
                title = title.replace("@", "[at]")  # Basic sanitization
                issue_id = issue.get("id")

                issues.append({
                    "id": str(issue_id) if issue_id else None,
                    "title": title,
                    "level": issue.get("level", "error"),
                    "count": int(issue.get("count", 0)),
                    "last_seen_at": issue.get("lastSeen"),
                    "first_seen_at": issue.get("firstSeen"),
                    "issue_url": f"https://sentry.io/organizations/{org_slug}/issues/{issue_id}/" if issue_id else None,
                })

            # Sort by count descending
            issues.sort(key=lambda x: x["count"], reverse=True)

            # Paginate
            total = len(issues)
            pages = (total + limit - 1) // limit if total > 0 else 0
            start = (page - 1) * limit
            end = start + limit
            paginated_issues = issues[start:end]

            # Determine status
            status = "ok"
            if len([i for i in issues if i["level"] == "error"]) >= 5:
                status = "warn"
            if len([i for i in issues if i["level"] == "error"]) >= 20:
                status = "red"

            result = {
                "issues": paginated_issues,
                "total": total,
                "page": page,
                "limit": limit,
                "pages": pages,
                "status": status,
            }

            # Cache
            _sentry_issues_cache["data"] = result
            _sentry_issues_cache["timestamp"] = now_ts
            _sentry_issues_cache["params"] = cache_key

            return _make_v2_wrapper(result, cached=False, cache_age_seconds=0)

    except Exception as e:
        logger.warning(f"Sentry issues endpoint error: {e}")
        base_data["note"] = f"fetch error: {str(e)[:50]}"
        return _make_v2_wrapper(base_data, cached=False, cache_age_seconds=0)


# Cache for /dashboard/predictions/missing.json
_missing_preds_cache: dict = {"data": None, "timestamp": 0, "ttl": 60, "params": None}


@router.get("/dashboard/predictions/missing.json")
async def dashboard_predictions_missing(
    request: Request,
    hours: int = 48,
    league_ids: str = None,  # comma-separated
    page: int = 1,
    limit: int = 50,
    session: AsyncSession = Depends(get_async_session),
):
    """
    V2 endpoint: Matches missing predictions with standard wrapper.
    TTL: 60s
    Auth: X-Dashboard-Token
    Query params: hours (1-72), league_ids (comma-separated), page, limit (max 100)
    """
    if not verify_dashboard_token_bool(request):
        raise HTTPException(status_code=401, detail="Dashboard access requires valid token.")

    import time as time_module

    # Clamp params
    hours = min(max(1, hours), 72)
    limit = min(max(1, limit), 100)
    page = max(1, page)

    # Parse league_ids
    league_filter = []
    if league_ids:
        try:
            league_filter = [int(x.strip()) for x in league_ids.split(",") if x.strip().isdigit()]
        except Exception:
            pass

    now_ts = time_module.time()
    cache_key = f"{hours}:{','.join(map(str, league_filter))}:{page}:{limit}"

    # Check cache
    if (_missing_preds_cache["data"] and
        _missing_preds_cache["params"] == cache_key and
        (now_ts - _missing_preds_cache["timestamp"]) < _missing_preds_cache["ttl"]):
        cache_age = now_ts - _missing_preds_cache["timestamp"]
        return _make_v2_wrapper(_missing_preds_cache["data"], cached=True, cache_age_seconds=cache_age)

    base_data = {
        "missing": [],
        "missing_total": 0,
        "matches_total": 0,
        "coverage_pct": None,
        "total": 0,
        "page": page,
        "limit": limit,
        "pages": 0,
        "status": "degraded",
        "note": "upstream unavailable",
    }

    try:
        from sqlalchemy import text

        now_dt = datetime.utcnow()
        cutoff = now_dt + timedelta(hours=hours)

        # Build query for matches without predictions
        league_clause = ""
        if league_filter:
            league_clause = f"AND m.league_id IN ({','.join(map(str, league_filter))})"

        # Count total matches in window
        total_query = f"""
            SELECT COUNT(*) FROM matches m
            WHERE m.status = 'NS'
              AND m.date BETWEEN NOW() AND :cutoff
              {league_clause}
        """
        total_result = await session.execute(text(total_query), {"cutoff": cutoff})
        matches_total = total_result.scalar() or 0

        # Get matches missing predictions
        missing_query = f"""
            SELECT m.id, m.date, m.league_id, ht.name as home, at.name as away
            FROM matches m
            JOIN teams ht ON ht.id = m.home_team_id
            JOIN teams at ON at.id = m.away_team_id
            LEFT JOIN predictions p ON p.match_id = m.id
            WHERE m.status = 'NS'
              AND m.date BETWEEN NOW() AND :cutoff
              AND p.id IS NULL
              {league_clause}
            ORDER BY m.date ASC
        """
        missing_result = await session.execute(text(missing_query), {"cutoff": cutoff})
        all_missing = missing_result.fetchall()

        missing_total = len(all_missing)

        # Build missing list
        missing_list = []
        for row in all_missing:
            missing_list.append({
                "match_id": row[0],
                "kickoff_utc": row[1].isoformat() + "Z" if row[1] else None,
                "league_id": row[2],
                "home": row[3],
                "away": row[4],
            })

        # Paginate
        pages = (missing_total + limit - 1) // limit if missing_total > 0 else 0
        start = (page - 1) * limit
        end = start + limit
        paginated = missing_list[start:end]

        # Calculate coverage
        coverage_pct = None
        if matches_total > 0:
            coverage_pct = round(((matches_total - missing_total) / matches_total) * 100, 1)

        # Determine status
        status = "ok"
        if missing_total > 0:
            status = "warn"
        if matches_total > 0 and (missing_total / matches_total) > 0.1:
            status = "red"

        result = {
            "missing": paginated,
            "missing_total": missing_total,
            "matches_total": matches_total,
            "coverage_pct": coverage_pct,
            "total": missing_total,
            "page": page,
            "limit": limit,
            "pages": pages,
            "status": status,
        }

        _missing_preds_cache["data"] = result
        _missing_preds_cache["timestamp"] = now_ts
        _missing_preds_cache["params"] = cache_key

        return _make_v2_wrapper(result, cached=False, cache_age_seconds=0)

    except Exception as e:
        logger.warning(f"Missing predictions endpoint error: {e}")
        base_data["note"] = f"db error: {str(e)[:50]}"
        return _make_v2_wrapper(base_data, cached=False, cache_age_seconds=0)


# Cache for /dashboard/movement/recent.json (activity by recency)
_movement_recent_cache: dict = {"data": None, "timestamp": 0, "ttl": 60, "params": None}


@router.get("/dashboard/movement/recent.json")
async def dashboard_movement_recent(
    request: Request,
    range: str = "24h",
    type: str = None,  # "lineup" or "market"
    limit: int = 50,
    session: AsyncSession = Depends(get_async_session),
):
    """
    V2 endpoint: Recent lineup/market activity ordered by recency.
    TTL: 60s
    Auth: X-Dashboard-Token
    Query params: range (24h|7d), type (lineup|market), limit (max 100)
    """
    if not verify_dashboard_token_bool(request):
        raise HTTPException(status_code=401, detail="Dashboard access requires valid token.")

    import time as time_module
    from sqlalchemy import text

    # Clamp params
    limit = min(max(1, limit), 100)

    # Validate range
    valid_ranges = {"24h": 24, "7d": 168}
    if range not in valid_ranges:
        range = "24h"
    hours = valid_ranges[range]

    # Validate type
    valid_types = ["lineup", "market", None]
    if type not in valid_types:
        type = None

    now_ts = time_module.time()
    cache_key = f"{range}:{type}:{limit}"

    # Check cache
    if (_movement_recent_cache["data"] and
        _movement_recent_cache["params"] == cache_key and
        (now_ts - _movement_recent_cache["timestamp"]) < _movement_recent_cache["ttl"]):
        cache_age = now_ts - _movement_recent_cache["timestamp"]
        return _make_v2_wrapper(_movement_recent_cache["data"], cached=True, cache_age_seconds=cache_age)

    base_data = {
        "items": [],
        "status": "degraded",
        "note": "upstream unavailable",
    }

    try:
        now_dt = datetime.utcnow()
        cutoff = now_dt - timedelta(hours=hours)

        items = []

        # Get lineup changes (from match_sofascore_lineup captured_at)
        if type is None or type == "lineup":
            lineup_query = """
                SELECT DISTINCT ON (m.id) m.id, m.date, m.league_id,
                       ht.name as home, at.name as away,
                       msl.captured_at
                FROM match_sofascore_lineup msl
                JOIN matches m ON m.id = msl.match_id
                JOIN teams ht ON ht.id = m.home_team_id
                JOIN teams at ON at.id = m.away_team_id
                WHERE msl.captured_at > :cutoff
                ORDER BY m.id, msl.captured_at DESC
                LIMIT :limit
            """
            lineup_result = await session.execute(
                text(lineup_query),
                {"cutoff": cutoff, "limit": limit}
            )
            for row in lineup_result.fetchall():
                items.append({
                    "match_id": row[0],
                    "kickoff_utc": row[1].isoformat() + "Z" if row[1] else None,
                    "league_id": row[2],
                    "home": row[3],
                    "away": row[4],
                    "type": "lineup",
                    "captured_at": row[5].isoformat() + "Z" if row[5] else None,
                    "source": "sofascore",
                })

        # Get market/odds captures (from odds_history)
        if type is None or type == "market":
            market_query = """
                SELECT DISTINCT ON (m.id) m.id, m.date, m.league_id,
                       ht.name as home, at.name as away,
                       oh.recorded_at, oh.source
                FROM odds_history oh
                JOIN matches m ON m.id = oh.match_id
                JOIN teams ht ON ht.id = m.home_team_id
                JOIN teams at ON at.id = m.away_team_id
                WHERE oh.recorded_at > :cutoff
                  AND NOT COALESCE(oh.quarantined, false)
                  AND oh.source NOT IN ('consensus')
                ORDER BY m.id, oh.recorded_at DESC
                LIMIT :limit
            """
            market_result = await session.execute(
                text(market_query),
                {"cutoff": cutoff, "limit": limit}
            )
            for row in market_result.fetchall():
                items.append({
                    "match_id": row[0],
                    "kickoff_utc": row[1].isoformat() + "Z" if row[1] else None,
                    "league_id": row[2],
                    "home": row[3],
                    "away": row[4],
                    "type": "market",
                    "captured_at": row[5].isoformat() + "Z" if row[5] else None,
                    "source": row[6] if row[6] else "unknown",
                })

        # Sort by captured_at (most recent first) and limit
        items.sort(key=lambda x: x["captured_at"] or "", reverse=True)
        items = items[:limit]

        result = {
            "items": items,
            "status": "ok" if items else "warn",
            "note": "recent activity ordered by recency",
        }

        _movement_recent_cache["data"] = result
        _movement_recent_cache["timestamp"] = now_ts
        _movement_recent_cache["params"] = cache_key

        return _make_v2_wrapper(result, cached=False, cache_age_seconds=0)

    except Exception as e:
        logger.warning(f"Movement recent endpoint error: {e}")
        base_data["note"] = f"db error: {str(e)[:50]}"
        return _make_v2_wrapper(base_data, cached=False, cache_age_seconds=0)


# Cache for /dashboard/movement/top.json (top movers by magnitude)
_movement_top_cache: dict = {"data": None, "timestamp": 0, "ttl": 60, "params": None}


@router.get("/dashboard/movement/top.json")
async def dashboard_movement_top(
    request: Request,
    range: str = "24h",
    type: str = None,  # "lineup" or "market"
    limit: int = 50,
    session: AsyncSession = Depends(get_async_session),
):
    """
    V2 endpoint: Top movers by movement magnitude.
    TTL: 60s
    Auth: X-Dashboard-Token
    Query params: range (24h|7d), type (lineup|market), limit (max 100)

    Movement magnitude:
    - market: max |Δimplied_prob| between first and last snapshot in window
    - lineup: placeholder (no magnitude metric yet, returns empty for type=lineup)
    """
    if not verify_dashboard_token_bool(request):
        raise HTTPException(status_code=401, detail="Dashboard access requires valid token.")

    import time as time_module
    from sqlalchemy import text

    # Clamp params
    limit = min(max(1, limit), 100)

    # Validate range
    valid_ranges = {"24h": 24, "7d": 168}
    if range not in valid_ranges:
        range = "24h"
    hours = valid_ranges[range]

    # Validate type
    valid_types = ["lineup", "market", None]
    if type not in valid_types:
        type = None

    now_ts = time_module.time()
    cache_key = f"{range}:{type}:{limit}"

    # Check cache
    if (_movement_top_cache["data"] and
        _movement_top_cache["params"] == cache_key and
        (now_ts - _movement_top_cache["timestamp"]) < _movement_top_cache["ttl"]):
        cache_age = now_ts - _movement_top_cache["timestamp"]
        return _make_v2_wrapper(_movement_top_cache["data"], cached=True, cache_age_seconds=cache_age)

    base_data = {
        "movers": [],
        "status": "degraded",
        "note": "upstream unavailable",
    }

    try:
        now_dt = datetime.utcnow()
        cutoff = now_dt - timedelta(hours=hours)

        movers = []

        # Market movers: calculate magnitude as max delta in implied probs
        if type is None or type == "market":
            # Get matches with multiple odds snapshots and calculate movement
            market_query = """
                WITH match_odds_range AS (
                    SELECT
                        match_id,
                        FIRST_VALUE(implied_home) OVER w AS first_implied_home,
                        FIRST_VALUE(implied_draw) OVER w AS first_implied_draw,
                        FIRST_VALUE(implied_away) OVER w AS first_implied_away,
                        LAST_VALUE(implied_home) OVER w AS last_implied_home,
                        LAST_VALUE(implied_draw) OVER w AS last_implied_draw,
                        LAST_VALUE(implied_away) OVER w AS last_implied_away,
                        MAX(recorded_at) OVER (PARTITION BY match_id) AS last_recorded_at,
                        ROW_NUMBER() OVER (PARTITION BY match_id ORDER BY recorded_at DESC) AS rn
                    FROM odds_history
                    WHERE recorded_at > :cutoff
                      AND NOT COALESCE(quarantined, false)
                      AND implied_home IS NOT NULL
                      AND source = 'Bet365'
                    WINDOW w AS (PARTITION BY match_id ORDER BY recorded_at
                                 ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING)
                ),
                match_movement AS (
                    SELECT
                        match_id,
                        last_recorded_at,
                        GREATEST(
                            ABS(COALESCE(last_implied_home, 0) - COALESCE(first_implied_home, 0)),
                            ABS(COALESCE(last_implied_draw, 0) - COALESCE(first_implied_draw, 0)),
                            ABS(COALESCE(last_implied_away, 0) - COALESCE(first_implied_away, 0))
                        ) AS magnitude
                    FROM match_odds_range
                    WHERE rn = 1
                )
                SELECT m.id, m.date, m.league_id, ht.name, at.name,
                       mm.magnitude, mm.last_recorded_at
                FROM match_movement mm
                JOIN matches m ON m.id = mm.match_id
                JOIN teams ht ON ht.id = m.home_team_id
                JOIN teams at ON at.id = m.away_team_id
                WHERE mm.magnitude > 0.01  -- Filter noise (>1% movement)
                ORDER BY mm.magnitude DESC
                LIMIT :limit
            """
            try:
                market_result = await session.execute(
                    text(market_query),
                    {"cutoff": cutoff, "limit": limit}
                )
                for row in market_result.fetchall():
                    movers.append({
                        "match_id": row[0],
                        "kickoff_utc": row[1].isoformat() + "Z" if row[1] else None,
                        "league_id": row[2],
                        "home": row[3],
                        "away": row[4],
                        "type": "market",
                        "value": round(float(row[5]) * 100, 2) if row[5] else 0,  # As percentage points
                        "captured_at": row[6].isoformat() + "Z" if row[6] else None,
                        "source": "api-football",
                    })
            except Exception as market_err:
                logger.debug(f"Market movers query failed: {market_err}")
                # Continue without market data

        # Lineup movers: no magnitude metric yet
        # For type=lineup, return empty with note
        if type == "lineup":
            result = {
                "movers": [],
                "status": "warn",
                "note": "lineup magnitude metric not implemented yet; use /movement/recent.json for lineup activity",
            }
            return _make_v2_wrapper(result, cached=False, cache_age_seconds=0)

        # Sort by magnitude (value) descending
        movers.sort(key=lambda x: abs(x.get("value", 0)), reverse=True)
        movers = movers[:limit]

        # Determine status
        status = "ok" if movers else "warn"
        note = "top movers by implied probability change (percentage points)"
        if not movers:
            note = "no significant market movements detected in window"

        result = {
            "movers": movers,
            "status": status,
            "note": note,
        }

        _movement_top_cache["data"] = result
        _movement_top_cache["timestamp"] = now_ts
        _movement_top_cache["params"] = cache_key

        return _make_v2_wrapper(result, cached=False, cache_age_seconds=0)

    except Exception as e:
        logger.warning(f"Movement top endpoint error: {e}")
        base_data["note"] = f"db error: {str(e)[:50]}"
        return _make_v2_wrapper(base_data, cached=False, cache_age_seconds=0)


# =============================================================================
# END DASHBOARD V2 ENDPOINTS
# =============================================================================


@router.post("/dashboard/predictions/trigger")
async def predictions_trigger_save(request: Request):
    """
    Manually trigger daily_save_predictions (for recovery).
    Protected by dashboard token.
    Use when predictions_health is RED/WARN.
    Returns detailed diagnostics for debugging.
    """
    if not verify_dashboard_token_bool(request):
        raise HTTPException(status_code=401, detail="Dashboard access requires valid token.")

    from app.db_utils import upsert
    from app.features import FeatureEngineer
    from app.ml.shadow import is_shadow_enabled, log_shadow_prediction
    from app.ml.sensor import log_sensor_prediction
    from app.config import get_settings
    from app.ops.audit import log_ops_action, OpsActionTimer

    sensor_settings = get_settings()
    start_time = time.time()

    diagnostics = {
        "status": "unknown",
        "model_loaded": False,
        "matches_found": 0,
        "ns_matches": 0,
        "predictions_generated": 0,
        "predictions_saved": 0,
        "shadow_logged": 0,
        "shadow_errors": 0,
        "sensor_logged": 0,
        "sensor_errors": 0,
        "errors": [],
    }

    try:
        async with AsyncSessionLocal() as session:
            # Step 1: Use global ml_engine (already loaded at startup)
            if not ml_engine.is_loaded:
                diagnostics["status"] = "error"
                diagnostics["errors"].append("Global ML model not loaded")
                return diagnostics
            diagnostics["model_loaded"] = True
            diagnostics["model_version"] = ml_engine.model_version

            # Step 2: Get features
            feature_engineer = FeatureEngineer(session=session)
            df = await feature_engineer.get_upcoming_matches_features()
            diagnostics["matches_found"] = len(df)

            if len(df) == 0:
                diagnostics["status"] = "ok"
                diagnostics["errors"].append("No upcoming matches found")
                return diagnostics

            # Filter to NS only
            df_ns = df[df["status"] == "NS"].copy()
            diagnostics["ns_matches"] = len(df_ns)

            if len(df_ns) == 0:
                diagnostics["status"] = "ok"
                diagnostics["errors"].append("No NS matches to predict")
                return diagnostics

            # Step 3: Generate predictions
            predictions = ml_engine.predict(df_ns)
            diagnostics["predictions_generated"] = len(predictions)

            # Step 4: Save to database with shadow + sensor logging
            saved = 0
            shadow_logged = 0
            shadow_errors = 0
            sensor_logged = 0
            sensor_errors = 0
            for idx, pred in enumerate(predictions):
                match_id = pred.get("match_id")
                if not match_id:
                    continue

                probs = pred["probabilities"]
                try:
                    await upsert(
                        session,
                        Prediction,
                        values={
                            "match_id": match_id,
                            "model_version": ml_engine.model_version,
                            "home_prob": probs["home"],
                            "draw_prob": probs["draw"],
                            "away_prob": probs["away"],
                        },
                        conflict_columns=["match_id", "model_version"],
                        update_columns=["home_prob", "draw_prob", "away_prob"],
                    )
                    saved += 1

                    # Shadow prediction: log parallel two-stage prediction
                    if is_shadow_enabled():
                        try:
                            match_df = df_ns.iloc[[idx]]
                            shadow_result = await log_shadow_prediction(
                                session=session,
                                match_id=match_id,
                                df=match_df,
                                baseline_engine=ml_engine,
                                skip_commit=True,
                            )
                            if shadow_result:
                                shadow_logged += 1
                        except Exception as shadow_err:
                            shadow_errors += 1
                            logger.warning(f"Shadow prediction failed for match {match_id}: {shadow_err}")

                    # Sensor B: log A vs B predictions (internal diagnostics only)
                    if sensor_settings.SENSOR_ENABLED:
                        try:
                            import numpy as np
                            match_df = df_ns.iloc[[idx]]
                            model_a_probs = np.array([probs["home"], probs["draw"], probs["away"]])
                            sensor_result = await log_sensor_prediction(
                                session=session,
                                match_id=match_id,
                                df=match_df,
                                model_a_probs=model_a_probs,
                                model_a_version=ml_engine.model_version,
                            )
                            if sensor_result:
                                sensor_logged += 1
                        except Exception as sensor_err:
                            sensor_errors += 1
                            logger.warning(f"Sensor prediction failed for match {match_id}: {sensor_err}")

                except Exception as e:
                    diagnostics["errors"].append(f"Match {match_id}: {str(e)[:50]}")

            await session.commit()
            diagnostics["predictions_saved"] = saved
            diagnostics["shadow_logged"] = shadow_logged
            diagnostics["shadow_errors"] = shadow_errors
            diagnostics["sensor_logged"] = sensor_logged
            diagnostics["sensor_errors"] = sensor_errors
            diagnostics["status"] = "ok" if saved > 0 else "no_new_predictions"

            shadow_info = f", shadow: {shadow_logged}" if is_shadow_enabled() else ""
            sensor_info = f", sensor: {sensor_logged}" if sensor_settings.SENSOR_ENABLED else ""
            logger.info(f"Predictions trigger complete: {saved} saved from {len(df_ns)} NS matches{shadow_info}{sensor_info}")

    except Exception as e:
        diagnostics["status"] = "error"
        diagnostics["errors"].append(str(e))
        logger.error(f"Predictions trigger failed: {e}")

    # Audit log
    duration_ms = int((time.time() - start_time) * 1000)
    try:
        async with AsyncSessionLocal() as audit_session:
            await log_ops_action(
                session=audit_session,
                request=request,
                action="predictions_trigger",
                params=None,
                result="ok" if diagnostics["status"] == "ok" else "error",
                result_detail={
                    "predictions_saved": diagnostics.get("predictions_saved", 0),
                    "ns_matches": diagnostics.get("ns_matches", 0),
                    "shadow_logged": diagnostics.get("shadow_logged", 0),
                    "sensor_logged": diagnostics.get("sensor_logged", 0),
                },
                error_message=diagnostics["errors"][0] if diagnostics["errors"] else None,
                duration_ms=duration_ms,
            )
    except Exception as audit_err:
        logger.warning(f"Failed to log audit for predictions_trigger: {audit_err}")

    return diagnostics


@router.post("/dashboard/predictions/trigger-fase0")
async def predictions_trigger_fase0(request: Request):
    """
    Trigger predictions using the EXACT same code path as the scheduler.

    FASE 0 ATI AUDIT: This endpoint calls daily_save_predictions() directly
    to ensure kill-switch router and league_only features are exercised.

    Unlike /dashboard/predictions/trigger (recovery endpoint), this one:
    - Uses league_only=True for feature engineering
    - Applies kill-switch router (filters teams with <5 league matches)
    - Emits [KILL-SWITCH] logs and Prometheus metrics
    - Returns detailed metrics for audit verification

    Protected by dashboard token.
    """
    if not verify_dashboard_token_bool(request):
        raise HTTPException(status_code=401, detail="Dashboard access requires valid token.")

    from app.scheduler import daily_save_predictions
    from app.ops.audit import log_ops_action

    start_time = time.time()

    try:
        # Call the SAME function that the scheduler uses
        # This ensures kill-switch router is exercised with identical code path
        result = await daily_save_predictions(return_metrics=True)

        # Handle case where result is None (shouldn't happen with return_metrics=True)
        if result is None:
            result = {"status": "unknown", "error": "No metrics returned"}

    except Exception as e:
        logger.error(f"[TRIGGER-FASE0] Failed: {e}")
        result = {"status": "error", "error": str(e)}

    # Audit log
    duration_ms = int((time.time() - start_time) * 1000)
    try:
        async with AsyncSessionLocal() as audit_session:
            await log_ops_action(
                session=audit_session,
                request=request,
                action="predictions_trigger_fase0",
                params=None,
                result="ok" if result.get("status") == "ok" else "error",
                result_detail={
                    "n_matches_total": result.get("n_matches_total", 0),
                    "n_eligible": result.get("n_eligible", 0),
                    "n_filtered": result.get("n_filtered", 0),
                    "filtered_by_reason": result.get("filtered_by_reason", {}),
                    "saved": result.get("saved", 0),
                },
                error_message=result.get("error"),
                duration_ms=duration_ms,
            )
    except Exception as audit_err:
        logger.warning(f"Failed to log audit for trigger-fase0: {audit_err}")

    return result




# -----------------------------------------------------------------------------
# Upcoming Matches for Dashboard Overview Card
# -----------------------------------------------------------------------------
_upcoming_matches_cache = {
    "data": None,
    "timestamp": 0,
    "ttl": 60,  # seconds - recommended by auditor
}


@router.get("/dashboard/upcoming_matches.json")
async def get_upcoming_matches_dashboard(
    request: Request,
    hours: int = 24,
    limit: int = 20,
    session: AsyncSession = Depends(get_async_session),
):
    """
    Upcoming scheduled matches for dashboard Overview card.

    Returns next N hours of scheduled matches with prediction status.
    Uses EXISTS subquery to avoid N+1 queries (P0 requirement).

    Auth: X-Dashboard-Token header (read-only, separate from X-API-Key).

    Response:
    {
        "generated_at": "2026-01-22T...",
        "cached": true/false,
        "cache_age_seconds": 45,
        "data": {
            "upcoming": [
                {
                    "id": 12345,
                    "home": "Real Madrid",
                    "away": "Barcelona",
                    "kickoff_iso": "2026-01-22T20:00:00Z",
                    "league_name": "La Liga",
                    "has_prediction": true
                }
            ]
        }
    }
    """
    if not verify_dashboard_token_bool(request):
        raise HTTPException(status_code=401, detail="Dashboard access requires valid token.")

    # Enforce limits (P1 recommendation)
    hours = min(max(hours, 1), 72)  # 1-72 hours
    limit = min(max(limit, 1), 50)  # 1-50 matches

    now = time.time()

    # Check cache
    if (
        _upcoming_matches_cache["data"] is not None
        and (now - _upcoming_matches_cache["timestamp"]) < _upcoming_matches_cache["ttl"]
    ):
        cached_data = _upcoming_matches_cache["data"]
        return {
            "generated_at": cached_data["generated_at"],
            "cached": True,
            "cache_age_seconds": round(now - _upcoming_matches_cache["timestamp"], 1),
            "data": cached_data["data"],
        }

    # Build query with EXISTS subquery to avoid N+1 (P0 requirement)
    from sqlalchemy import exists, literal_column
    from sqlalchemy.orm import aliased

    now_dt = datetime.utcnow()
    cutoff_dt = now_dt + timedelta(hours=hours)

    # Subquery: EXISTS prediction for this match
    has_prediction_subq = (
        exists()
        .where(Prediction.match_id == Match.id)
        .correlate(Match)
    )

    # Main query with team names via JOIN
    home_team = aliased(Team, name="home_team")
    away_team = aliased(Team, name="away_team")

    query = (
        select(
            Match.id,
            Match.date,
            Match.league_id,
            Match.home_team_id,
            Match.away_team_id,
            home_team.name.label("home_name"),
            away_team.name.label("away_name"),
            has_prediction_subq.label("has_prediction"),
        )
        .join(home_team, Match.home_team_id == home_team.id)
        .join(away_team, Match.away_team_id == away_team.id)
        .where(Match.status == "NS")
        .where(Match.date >= now_dt)
        .where(Match.date <= cutoff_dt)
        .order_by(Match.date)
        .limit(limit)
    )

    result = await session.execute(query)
    rows = result.all()

    # Batch resolve display_names (COALESCE: override > wikidata > name)
    all_team_ids = list({r.home_team_id for r in rows} | {r.away_team_id for r in rows}) if rows else []
    display_map: dict[int, str] = {}
    if all_team_ids:
        dn_result = await session.execute(
            text("""
                SELECT t.id AS team_id,
                       COALESCE(teo.short_name, twe.short_name, t.name) AS display_name
                FROM teams t
                LEFT JOIN team_enrichment_overrides teo ON t.id = teo.team_id
                LEFT JOIN team_wikidata_enrichment twe ON t.id = twe.team_id
                WHERE t.id = ANY(:team_ids)
            """),
            {"team_ids": all_team_ids}
        )
        display_map = {r.team_id: r.display_name for r in dn_result.fetchall()}

    # Build league name lookup from COMPETITIONS (single source of truth)
    from app.etl.competitions import COMPETITIONS
    league_name_by_id: dict[int, str] = {
        lid: comp.name for lid, comp in COMPETITIONS.items() if comp.name
    }

    # Format response
    upcoming = []
    for row in rows:
        upcoming.append({
            "id": row.id,
            "home": row.home_name,
            "away": row.away_name,
            "home_display_name": display_map.get(row.home_team_id, row.home_name),
            "away_display_name": display_map.get(row.away_team_id, row.away_name),
            "kickoff_iso": row.date.isoformat() + "Z" if row.date else None,
            "league_name": league_name_by_id.get(row.league_id, f"League {row.league_id}"),
            "has_prediction": bool(row.has_prediction),
        })

    generated_at = datetime.utcnow().isoformat() + "Z"

    # Update cache
    _upcoming_matches_cache["data"] = {
        "generated_at": generated_at,
        "data": {"upcoming": upcoming},
    }
    _upcoming_matches_cache["timestamp"] = now

    return {
        "generated_at": generated_at,
        "cached": False,
        "cache_age_seconds": 0,
        "data": {"upcoming": upcoming},
    }


# -----------------------------------------------------------------------------
# Dashboard Matches Table Endpoint
# -----------------------------------------------------------------------------

# Cache for matches table - keyed by query params
# TTL: 15s for LIVE, 60s for NS/FT (P1 auditor: 10-15s LIVE for Ops responsiveness)
_matches_table_cache: dict[str, dict] = {}
_MATCHES_CACHE_TTL_LIVE = 15  # seconds
_MATCHES_CACHE_TTL_DEFAULT = 60  # seconds


def _get_matches_cache_key(
    status: str,
    hours: int,
    league_id: int | None,
    page: int,
    limit: int,
    match_id: int | None = None,
    from_time: str | None = None,
    to_time: str | None = None,
) -> str:
    """Generate cache key including all query params (P1 guardrail)."""
    if from_time and to_time:
        return f"matches:{status}:range:{from_time}:{to_time}:{league_id}:{page}:{limit}:{match_id}"
    return f"matches:{status}:{hours}:{league_id}:{page}:{limit}:{match_id}"


@router.get("/dashboard/matches.json")
async def get_matches_dashboard(
    request: Request,
    status: str = "NS",  # NS, LIVE, FT, or ALL
    hours: int = 168,  # 7 days default for table view
    from_time: str | None = None,  # ISO8601 UTC start time (overrides hours)
    to_time: str | None = None,  # ISO8601 UTC end time (overrides hours)
    league_id: int | None = None,
    match_id: int | None = None,  # optional exact match lookup (deep-link support)
    page: int = 1,
    limit: int = 50,
    session: AsyncSession = Depends(get_async_session),
):
    """
    Matches table for dashboard /matches page.

    Returns paginated matches with filtering by status, time window, and league.
    Uses EXISTS subquery to avoid N+1 queries.

    Auth: X-Dashboard-Token header (read-only).

    Query params:
    - status: NS (scheduled), LIVE (in-play), FT (finished), ALL
    - hours: time window (1-168, default 168 = 7 days)
    - from_time: ISO8601 UTC start time (overrides hours if both from_time and to_time provided)
    - to_time: ISO8601 UTC end time (overrides hours if both from_time and to_time provided)
    - league_id: optional filter by league
    - page: pagination (1-indexed)
    - limit: rows per page (1-100)

    Response:
    {
        "generated_at": "2026-01-22T...",
        "cached": true/false,
        "cache_age_seconds": 15,
        "data": {
            "matches": [...],
            "total": 234,
            "page": 1,
            "limit": 50,
            "pages": 5
        }
    }
    """
    if not verify_dashboard_token_bool(request):
        raise HTTPException(status_code=401, detail="Dashboard access requires valid token.")

    # Normalize and validate params
    status = status.upper()
    valid_statuses = {"NS", "LIVE", "FT", "ALL"}
    if status not in valid_statuses:
        status = "ALL"

    hours = min(max(hours, 1), 168)  # 1-168 hours (7 days max)
    page = max(page, 1)
    limit = min(max(limit, 1), 100)

    # Parse from_time/to_time if provided (ISO8601 UTC)
    parsed_from_time = None
    parsed_to_time = None
    if from_time and to_time:
        try:
            # Parse ISO8601 format (e.g., 2026-01-23T00:00:00Z)
            parsed_from_time = datetime.fromisoformat(from_time.replace("Z", "+00:00")).replace(tzinfo=None)
            parsed_to_time = datetime.fromisoformat(to_time.replace("Z", "+00:00")).replace(tzinfo=None)
        except ValueError:
            # Invalid format, ignore and use hours-based window
            pass

    # Cache TTL based on status
    cache_ttl = _MATCHES_CACHE_TTL_LIVE if status == "LIVE" else _MATCHES_CACHE_TTL_DEFAULT
    # Include from_time/to_time in cache key when used
    cache_key = _get_matches_cache_key(
        status, hours, league_id, page, limit, match_id=match_id,
        from_time=from_time if parsed_from_time else None,
        to_time=to_time if parsed_to_time else None
    )

    now = time.time()

    # Check cache
    if cache_key in _matches_table_cache:
        cached_entry = _matches_table_cache[cache_key]
        age = now - cached_entry["timestamp"]
        if age < cache_ttl:
            return {
                "generated_at": cached_entry["generated_at"],
                "cached": True,
                "cache_age_seconds": round(age, 1),
                "data": cached_entry["data"],
            }

    # Build query with GROUP BY to deduplicate prediction rows
    from sqlalchemy import func
    from sqlalchemy.orm import aliased

    now_dt = datetime.utcnow()

    # Time window: use from_time/to_time if provided, otherwise calculate from hours
    if parsed_from_time and parsed_to_time:
        # Explicit date range from calendar view
        start_dt = parsed_from_time
        end_dt = parsed_to_time
    elif status == "NS":
        # Upcoming: now to +hours
        start_dt = now_dt
        end_dt = now_dt + timedelta(hours=hours)
    elif status == "FT":
        # Finished: -hours to now
        start_dt = now_dt - timedelta(hours=hours)
        end_dt = now_dt
    else:
        # LIVE or ALL: both directions
        start_dt = now_dt - timedelta(hours=hours)
        end_dt = now_dt + timedelta(hours=hours)

    # Team aliases
    home_team = aliased(Team, name="home_team")
    away_team = aliased(Team, name="away_team")

    # Team display names subquery (for use_short_names toggle)
    # COALESCE: override.short_name > wikidata.short_name > team.name
    display_names_subq = text("""
        SELECT
            t.id AS team_id,
            COALESCE(teo.short_name, twe.short_name, t.name) AS display_name
        FROM teams t
        LEFT JOIN team_enrichment_overrides teo ON t.id = teo.team_id
        LEFT JOIN team_wikidata_enrichment twe ON t.id = twe.team_id
    """).columns(
        column("team_id"),
        column("display_name"),
    ).subquery("display_names")

    # Aliases for home/away display names
    home_display = aliased(display_names_subq, name="home_display")
    away_display = aliased(display_names_subq, name="away_display")

    # Base query with LEFT JOINs for predictions and weather
    # Use GROUP BY and MAX to get one row per match when there are multiple predictions
    # Weather: use raw SQL subquery with DISTINCT ON to get latest forecast per match
    weather_subq = text("""
        SELECT DISTINCT ON (match_id)
            match_id,
            temp_c,
            humidity,
            wind_ms,
            precip_mm,
            precip_prob,
            cloudcover,
            is_daylight
        FROM match_weather
        ORDER BY match_id, forecast_horizon_hours ASC
    """).columns(
        column("match_id"),
        column("temp_c"),
        column("humidity"),
        column("wind_ms"),
        column("precip_mm"),
        column("precip_prob"),
        column("cloudcover"),
        column("is_daylight"),
    ).subquery("weather")

    # Experimental predictions (ext-A/B/C) from predictions_experiments
    # Uses DISTINCT ON with PIT guard (snapshot_at <= kickoff) and tie-break
    ext_subq = text("""
        WITH latest AS (
          SELECT DISTINCT ON (pe.match_id, pe.model_version)
            pe.match_id,
            pe.model_version,
            pe.home_prob,
            pe.draw_prob,
            pe.away_prob
          FROM predictions_experiments pe
          JOIN matches m ON m.id = pe.match_id
          WHERE pe.model_version IN ('v1.0.2-ext-A','v1.0.2-ext-B','v1.0.2-ext-C','v1.0.1-league-only-20260202')
            AND pe.snapshot_at <= m.date
          ORDER BY pe.match_id, pe.model_version, pe.snapshot_at DESC, pe.created_at DESC
        )
        SELECT
          match_id,
          MAX(CASE WHEN model_version='v1.0.2-ext-A' THEN home_prob END) AS ext_a_home,
          MAX(CASE WHEN model_version='v1.0.2-ext-A' THEN draw_prob END) AS ext_a_draw,
          MAX(CASE WHEN model_version='v1.0.2-ext-A' THEN away_prob END) AS ext_a_away,
          MAX(CASE WHEN model_version='v1.0.2-ext-B' THEN home_prob END) AS ext_b_home,
          MAX(CASE WHEN model_version='v1.0.2-ext-B' THEN draw_prob END) AS ext_b_draw,
          MAX(CASE WHEN model_version='v1.0.2-ext-B' THEN away_prob END) AS ext_b_away,
          MAX(CASE WHEN model_version='v1.0.2-ext-C' THEN home_prob END) AS ext_c_home,
          MAX(CASE WHEN model_version='v1.0.2-ext-C' THEN draw_prob END) AS ext_c_draw,
          MAX(CASE WHEN model_version='v1.0.2-ext-C' THEN away_prob END) AS ext_c_away,
          MAX(CASE WHEN model_version='v1.0.1-league-only-20260202' THEN home_prob END) AS ext_d_home,
          MAX(CASE WHEN model_version='v1.0.1-league-only-20260202' THEN draw_prob END) AS ext_d_draw,
          MAX(CASE WHEN model_version='v1.0.1-league-only-20260202' THEN away_prob END) AS ext_d_away
        FROM latest
        GROUP BY match_id
    """).columns(
        column("match_id"),
        column("ext_a_home"),
        column("ext_a_draw"),
        column("ext_a_away"),
        column("ext_b_home"),
        column("ext_b_draw"),
        column("ext_b_away"),
        column("ext_c_home"),
        column("ext_c_draw"),
        column("ext_c_away"),
        column("ext_d_home"),
        column("ext_d_draw"),
        column("ext_d_away"),
    ).subquery("ext")

    base_query = (
        select(
            Match.id,
            Match.date,
            Match.league_id,
            Match.round,
            Match.status,
            Match.home_goals,
            Match.away_goals,
            Match.elapsed,
            Match.elapsed_extra,
            Match.venue_name,
            Match.venue_city,
            Match.home_team_id,
            Match.away_team_id,
            home_team.name.label("home_name"),
            away_team.name.label("away_name"),
            # Display names for use_short_names toggle
            home_display.c.display_name.label("home_display_name"),
            away_display.c.display_name.label("away_display_name"),
            # Model A (production) prediction - use MAX to pick one value
            func.max(Prediction.home_prob).label("model_a_home"),
            func.max(Prediction.draw_prob).label("model_a_draw"),
            func.max(Prediction.away_prob).label("model_a_away"),
            # Market odds (frozen at prediction time)
            func.max(Prediction.frozen_odds_home).label("market_home"),
            func.max(Prediction.frozen_odds_draw).label("market_draw"),
            func.max(Prediction.frozen_odds_away).label("market_away"),
            # Shadow/Two-Stage prediction
            func.max(ShadowPrediction.shadow_home_prob).label("shadow_home"),
            func.max(ShadowPrediction.shadow_draw_prob).label("shadow_draw"),
            func.max(ShadowPrediction.shadow_away_prob).label("shadow_away"),
            # Sensor B prediction
            func.max(SensorPrediction.b_home_prob).label("sensor_b_home"),
            func.max(SensorPrediction.b_draw_prob).label("sensor_b_draw"),
            func.max(SensorPrediction.b_away_prob).label("sensor_b_away"),
            # Weather forecast (use direct columns, added to GROUP BY since subquery has DISTINCT ON)
            weather_subq.c.temp_c.label("weather_temp_c"),
            weather_subq.c.humidity.label("weather_humidity"),
            weather_subq.c.wind_ms.label("weather_wind_ms"),
            weather_subq.c.precip_mm.label("weather_precip_mm"),
            weather_subq.c.precip_prob.label("weather_precip_prob"),
            weather_subq.c.cloudcover.label("weather_cloudcover"),
            weather_subq.c.is_daylight.label("weather_is_daylight"),
            # Ext-A/B/C experimental predictions (use MAX to avoid GROUP BY issues)
            func.max(ext_subq.c.ext_a_home).label("ext_a_home"),
            func.max(ext_subq.c.ext_a_draw).label("ext_a_draw"),
            func.max(ext_subq.c.ext_a_away).label("ext_a_away"),
            func.max(ext_subq.c.ext_b_home).label("ext_b_home"),
            func.max(ext_subq.c.ext_b_draw).label("ext_b_draw"),
            func.max(ext_subq.c.ext_b_away).label("ext_b_away"),
            func.max(ext_subq.c.ext_c_home).label("ext_c_home"),
            func.max(ext_subq.c.ext_c_draw).label("ext_c_draw"),
            func.max(ext_subq.c.ext_c_away).label("ext_c_away"),
            func.max(ext_subq.c.ext_d_home).label("ext_d_home"),
            func.max(ext_subq.c.ext_d_draw).label("ext_d_draw"),
            func.max(ext_subq.c.ext_d_away).label("ext_d_away"),
        )
        .join(home_team, Match.home_team_id == home_team.id)
        .join(away_team, Match.away_team_id == away_team.id)
        .outerjoin(home_display, home_display.c.team_id == Match.home_team_id)
        .outerjoin(away_display, away_display.c.team_id == Match.away_team_id)
        .outerjoin(Prediction, (Prediction.match_id == Match.id) & (Prediction.model_version == settings.MODEL_VERSION))
        .outerjoin(ShadowPrediction, ShadowPrediction.match_id == Match.id)
        .outerjoin(SensorPrediction, SensorPrediction.match_id == Match.id)
        .outerjoin(weather_subq, weather_subq.c.match_id == Match.id)
        .outerjoin(ext_subq, ext_subq.c.match_id == Match.id)
        .group_by(
            Match.id,
            Match.date,
            Match.league_id,
            Match.round,
            Match.status,
            Match.home_goals,
            Match.away_goals,
            Match.elapsed,
            Match.elapsed_extra,
            Match.venue_name,
            Match.venue_city,
            Match.home_team_id,
            Match.away_team_id,
            home_team.name,
            away_team.name,
            # Display names
            home_display.c.display_name,
            away_display.c.display_name,
            # Weather fields (from DISTINCT ON subquery, so safe to group by)
            weather_subq.c.temp_c,
            weather_subq.c.humidity,
            weather_subq.c.wind_ms,
            weather_subq.c.precip_mm,
            weather_subq.c.precip_prob,
            weather_subq.c.cloudcover,
            weather_subq.c.is_daylight,
        )
    )

    # Optional exact match lookup (deep-link support).
    # When match_id is provided we ignore time window/status/league filters to avoid false negatives.
    if match_id is not None:
        base_query = base_query.where(Match.id == match_id)
    else:
        base_query = base_query.where(Match.date >= start_dt).where(Match.date <= end_dt)

    # Status filter
    if match_id is None and status == "LIVE":
        # LIVE includes: 1H, HT, 2H, ET, BT, P, SUSP, INT, LIVE
        live_statuses = ["1H", "HT", "2H", "ET", "BT", "P", "SUSP", "INT", "LIVE"]
        base_query = base_query.where(Match.status.in_(live_statuses))
    elif match_id is None and status == "NS":
        base_query = base_query.where(Match.status == "NS")
    elif match_id is None and status == "FT":
        # FT includes: FT, AET, PEN
        ft_statuses = ["FT", "AET", "PEN"]
        base_query = base_query.where(Match.status.in_(ft_statuses))
    # ALL: no status filter

    # League filter
    if match_id is None and league_id is not None:
        base_query = base_query.where(Match.league_id == league_id)

    # Count total for pagination (count distinct match IDs)
    # Note: We just count Match rows, no need to join Team tables for count
    count_query = (
        select(func.count(Match.id))
    )

    if match_id is not None:
        count_query = count_query.where(Match.id == match_id)
    else:
        count_query = count_query.where(Match.date >= start_dt).where(Match.date <= end_dt)

    if match_id is None and status == "LIVE":
        count_query = count_query.where(Match.status.in_(["1H", "HT", "2H", "ET", "BT", "P", "SUSP", "INT", "LIVE"]))
    elif match_id is None and status == "NS":
        count_query = count_query.where(Match.status == "NS")
    elif match_id is None and status == "FT":
        count_query = count_query.where(Match.status.in_(["FT", "AET", "PEN"]))
    if match_id is None and league_id is not None:
        count_query = count_query.where(Match.league_id == league_id)

    total_result = await session.execute(count_query)
    total = total_result.scalar() or 0

    # Order and paginate
    # For match_id we always return a single match (if found).
    if match_id is not None:
        page = 1
        limit = 1
        offset = 0
        query = base_query.limit(1)
    else:
        offset = (page - 1) * limit
        query = base_query.order_by(Match.date.desc() if status == "FT" else Match.date).offset(offset).limit(limit)

    result = await session.execute(query)
    rows = result.all()

    # League names from COMPETITIONS with extended fallback
    from app.etl.competitions import COMPETITIONS

    # Extended fallback for leagues not in COMPETITIONS
    LEAGUE_NAMES_EXTENDED: dict[int, str] = {
        1: "World Cup", 2: "Champions League", 3: "Europa League",
        39: "Premier League", 40: "Championship", 61: "Ligue 1",
        78: "Bundesliga", 135: "Serie A", 140: "La Liga",
        94: "Primeira Liga", 88: "Eredivisie", 203: "Süper Lig",
        239: "Liga BetPlay", 253: "MLS", 262: "Liga MX",
        128: "Argentina Primera", 71: "Brasileirão",
        848: "Conference League", 45: "FA Cup", 143: "Copa del Rey",
        242: "Ecuador Liga Pro", 250: "Paraguay Primera",
    }

    # League ID to country mapping for flags
    LEAGUE_COUNTRY: dict[int, str] = {
        # International
        1: "World", 2: "World", 3: "World", 848: "World",
        4: "World", 5: "World", 6: "World", 7: "World",
        9: "World", 10: "World", 11: "World", 13: "World",
        22: "World", 29: "World", 30: "World", 31: "World",
        32: "World", 33: "World", 34: "World", 37: "World",
        # Europe Top 5
        39: "England", 40: "England", 45: "England",
        140: "Spain", 143: "Spain",
        135: "Italy",
        78: "Germany",
        61: "France",
        # Europe Secondary
        94: "Portugal", 88: "Netherlands", 144: "Belgium",
        203: "Turkey",
        # Americas
        253: "USA", 262: "Mexico",
        128: "Argentina", 71: "Brazil",
        239: "Colombia", 242: "Ecuador", 250: "Paraguay",
        265: "Chile", 268: "Uruguay", 281: "Peru",
        299: "Venezuela", 344: "Bolivia",
        # Middle East
        307: "Saudi-Arabia",
    }

    # Build league name lookup: COMPETITIONS takes priority, then extended fallback
    league_name_by_id: dict[int, str] = LEAGUE_NAMES_EXTENDED.copy()
    for lid, comp in COMPETITIONS.items():
        if comp.name:
            league_name_by_id[lid] = comp.name

    # Format response
    matches = []
    for row in rows:
        match_data = {
            "id": row.id,
            "kickoff_iso": row.date.isoformat() + "Z" if row.date else None,
            "league_id": row.league_id,
            "league_name": league_name_by_id.get(row.league_id, f"League {row.league_id}"),
            "league_country": LEAGUE_COUNTRY.get(row.league_id, ""),
            "round": row.round,
            "home": row.home_name,
            "away": row.away_name,
            "home_team_id": row.home_team_id,
            "away_team_id": row.away_team_id,
            "home_display_name": row.home_display_name or row.home_name,
            "away_display_name": row.away_display_name or row.away_name,
            "status": row.status,
        }

        # Venue (stadium name and city)
        if row.venue_name or row.venue_city:
            match_data["venue"] = {
                "name": row.venue_name,
                "city": row.venue_city,
            }

        # Weather forecast (if available)
        if row.weather_temp_c is not None:
            match_data["weather"] = {
                "temp_c": round(row.weather_temp_c, 1),
                "humidity": round(row.weather_humidity, 0) if row.weather_humidity else None,
                "wind_ms": round(row.weather_wind_ms, 1) if row.weather_wind_ms else None,
                "precip_mm": round(row.weather_precip_mm, 1) if row.weather_precip_mm else None,
                "precip_prob": round(row.weather_precip_prob, 0) if row.weather_precip_prob else None,
                "cloudcover": round(row.weather_cloudcover, 0) if row.weather_cloudcover else None,
                "is_daylight": bool(row.weather_is_daylight) if row.weather_is_daylight is not None else None,
            }

        # Score (only if played/playing)
        if row.home_goals is not None and row.away_goals is not None:
            match_data["score"] = {"home": row.home_goals, "away": row.away_goals}

        # Elapsed (only if live)
        if row.elapsed is not None:
            match_data["elapsed"] = row.elapsed
            if row.elapsed_extra is not None:
                match_data["elapsed_extra"] = row.elapsed_extra

        # Market odds (converted from decimal odds to implied probabilities)
        if row.market_home is not None:
            match_data["market"] = {
                "home": round(1 / row.market_home, 3) if row.market_home > 0 else None,
                "draw": round(1 / row.market_draw, 3) if row.market_draw and row.market_draw > 0 else None,
                "away": round(1 / row.market_away, 3) if row.market_away and row.market_away > 0 else None,
            }

        # Model A prediction
        if row.model_a_home is not None:
            match_data["model_a"] = {
                "home": round(row.model_a_home, 3),
                "draw": round(row.model_a_draw, 3),
                "away": round(row.model_a_away, 3),
            }

        # Shadow/Two-Stage prediction
        if row.shadow_home is not None:
            match_data["shadow"] = {
                "home": round(row.shadow_home, 3),
                "draw": round(row.shadow_draw, 3),
                "away": round(row.shadow_away, 3),
            }

        # Sensor B prediction
        if row.sensor_b_home is not None:
            match_data["sensor_b"] = {
                "home": round(row.sensor_b_home, 3),
                "draw": round(row.sensor_b_draw, 3),
                "away": round(row.sensor_b_away, 3),
            }

        # Ext-A experimental prediction
        if row.ext_a_home is not None:
            match_data["extA"] = {
                "home": round(float(row.ext_a_home), 3),
                "draw": round(float(row.ext_a_draw), 3),
                "away": round(float(row.ext_a_away), 3),
            }

        # Ext-B experimental prediction
        if row.ext_b_home is not None:
            match_data["extB"] = {
                "home": round(float(row.ext_b_home), 3),
                "draw": round(float(row.ext_b_draw), 3),
                "away": round(float(row.ext_b_away), 3),
            }

        # Ext-C experimental prediction
        if row.ext_c_home is not None:
            match_data["extC"] = {
                "home": round(float(row.ext_c_home), 3),
                "draw": round(float(row.ext_c_draw), 3),
                "away": round(float(row.ext_c_away), 3),
            }

        # Ext-D experimental prediction (league-only retrained)
        if row.ext_d_home is not None:
            match_data["extD"] = {
                "home": round(float(row.ext_d_home), 3),
                "draw": round(float(row.ext_d_draw), 3),
                "away": round(float(row.ext_d_away), 3),
            }

        matches.append(match_data)

    generated_at = datetime.utcnow().isoformat() + "Z"
    pages = (total + limit - 1) // limit if limit > 0 else 1

    response_data = {
        "matches": matches,
        "total": total,
        "page": page,
        "limit": limit,
        "pages": pages,
    }

    # Update cache
    _matches_table_cache[cache_key] = {
        "generated_at": generated_at,
        "timestamp": now,
        "data": response_data,
    }

    # Clean old cache entries (simple LRU: keep last 50)
    if len(_matches_table_cache) > 50:
        oldest_key = min(_matches_table_cache.keys(), key=lambda k: _matches_table_cache[k]["timestamp"])
        del _matches_table_cache[oldest_key]

    return {
        "generated_at": generated_at,
        "cached": False,
        "cache_age_seconds": 0,
        "data": response_data,
    }


# -----------------------------------------------------------------------------
# Dashboard Jobs History Endpoint
# -----------------------------------------------------------------------------

# Cache for jobs table
_jobs_table_cache: dict[str, dict] = {}
_JOBS_CACHE_TTL = 30  # seconds


def _get_jobs_cache_key(status: str | None, job_name: str | None, hours: int, page: int, limit: int) -> str:
    """Generate cache key including all query params."""
    return f"jobs:{status}:{job_name}:{hours}:{page}:{limit}"


@router.get("/dashboard/jobs.json")
async def get_jobs_dashboard(
    request: Request,
    status: str | None = None,  # ok, error, rate_limited, budget_exceeded
    job_name: str | None = None,  # Filter by job name
    hours: int = 24,  # Time window (default 24h)
    page: int = 1,
    limit: int = 50,
    session: AsyncSession = Depends(get_async_session),
):
    """
    Job runs history for dashboard /jobs page.

    Returns paginated job runs with filtering by status and job name.

    Auth: X-Dashboard-Token header (read-only).

    Query params:
    - status: ok, error, rate_limited, budget_exceeded (optional)
    - job_name: stats_backfill, odds_sync, fastpath, etc. (optional)
    - hours: time window (1-168, default 24)
    - page: pagination (1-indexed)
    - limit: rows per page (1-100)

    Response:
    {
        "generated_at": "2026-01-22T...",
        "cached": true/false,
        "cache_age_seconds": 15,
        "data": {
            "runs": [...],
            "total": 234,
            "page": 1,
            "limit": 50,
            "pages": 5,
            "jobs_summary": {...}  // Per-job health summary
        }
    }
    """
    if not verify_dashboard_token_bool(request):
        raise HTTPException(status_code=401, detail="Dashboard access requires valid token.")

    # Normalize and validate params
    hours = min(max(hours, 1), 168)  # 1-168 hours (7 days max)
    page = max(page, 1)
    limit = min(max(limit, 1), 100)

    cache_key = _get_jobs_cache_key(status, job_name, hours, page, limit)
    now = time.time()

    # Check cache
    if cache_key in _jobs_table_cache:
        cached_entry = _jobs_table_cache[cache_key]
        age = now - cached_entry["timestamp"]
        if age < _JOBS_CACHE_TTL:
            return {
                "generated_at": cached_entry["generated_at"],
                "cached": True,
                "cache_age_seconds": round(age, 1),
                "data": cached_entry["data"],
            }

    from sqlalchemy import func

    now_dt = datetime.utcnow()
    cutoff_dt = now_dt - timedelta(hours=hours)

    # Base query
    base_query = (
        select(JobRun)
        .where(JobRun.started_at >= cutoff_dt)
    )

    # Status filter
    if status:
        base_query = base_query.where(JobRun.status == status)

    # Job name filter
    if job_name:
        base_query = base_query.where(JobRun.job_name == job_name)

    # Count total for pagination
    count_query = select(func.count()).select_from(base_query.subquery())
    total_result = await session.execute(count_query)
    total = total_result.scalar() or 0

    # Order and paginate
    offset = (page - 1) * limit
    query = base_query.order_by(JobRun.started_at.desc()).offset(offset).limit(limit)

    result = await session.execute(query)
    rows = result.scalars().all()

    # Format response
    runs = []
    for row in rows:
        run_data = {
            "id": row.id,
            "job_name": row.job_name,
            "status": row.status,
            "started_at": row.started_at.isoformat() + "Z" if row.started_at else None,
            "finished_at": row.finished_at.isoformat() + "Z" if row.finished_at else None,
            "duration_ms": row.duration_ms,
        }

        # Only include error if failed
        if row.status == "error" and row.error_message:
            run_data["error"] = row.error_message[:500]  # Truncate long errors

        # Include metrics if present
        if row.metrics:
            run_data["metrics"] = row.metrics

        runs.append(run_data)

    # Get jobs summary (per-job health) using existing function
    from app.jobs.tracking import get_jobs_health_from_db
    jobs_summary = await get_jobs_health_from_db(session)

    generated_at = datetime.utcnow().isoformat() + "Z"
    pages = (total + limit - 1) // limit if limit > 0 else 1

    response_data = {
        "runs": runs,
        "total": total,
        "page": page,
        "limit": limit,
        "pages": pages,
        "jobs_summary": jobs_summary,
    }

    # Update cache
    _jobs_table_cache[cache_key] = {
        "generated_at": generated_at,
        "timestamp": now,
        "data": response_data,
    }

    # Clean old cache entries (simple LRU: keep last 20)
    if len(_jobs_table_cache) > 20:
        oldest_key = min(_jobs_table_cache.keys(), key=lambda k: _jobs_table_cache[k]["timestamp"])
        del _jobs_table_cache[oldest_key]

    return {
        "generated_at": generated_at,
        "cached": False,
        "cache_age_seconds": 0,
        "data": response_data,
    }


# -----------------------------------------------------------------------------
# Dashboard Data Quality Endpoint (tabular checks for /data-quality page)
# -----------------------------------------------------------------------------
_data_quality_cache: dict[str, dict] = {}
_DATA_QUALITY_CACHE_TTL = 45  # seconds (auditor: 30–60s)
_data_quality_detail_cache: dict[str, dict] = {}
_DATA_QUALITY_DETAIL_CACHE_TTL = 45  # seconds


def _normalize_multi_param(values: list[str] | None) -> list[str]:
    """
    Normalize multi-select query params.
    Supports repeated params (?status=a&status=b) and comma-separated (?status=a,b).
    """
    if not values:
        return []
    out: list[str] = []
    for v in values:
        if not v:
            continue
        parts = [p.strip() for p in v.split(",")] if "," in v else [v.strip()]
        out.extend([p for p in parts if p])
    # Deduplicate, preserve order
    seen = set()
    deduped = []
    for v in out:
        k = v.lower()
        if k in seen:
            continue
        seen.add(k)
        deduped.append(v)
    return deduped


def _dq_status_from_count(
    count: int | None,
    *,
    fail_if_gt: int = 0,
    warn_if_gt: int = 0,
) -> str:
    """Map numeric check value to passing|warning|failing."""
    if count is None:
        return "warning"
    if count > fail_if_gt:
        return "failing"
    if count > warn_if_gt:
        return "warning"
    return "passing"


async def _build_data_quality_checks(session: AsyncSession) -> list[dict]:
    """
    Build a stable list of Data Quality checks.
    Best-effort: if a single source fails, degrade that check (do not fail whole endpoint).
    """
    now_iso = datetime.utcnow().isoformat() + "Z"
    checks: list[dict] = []

    # Helper to append checks with consistent shape
    def add_check(
        *,
        check_id: str,
        name: str,
        category: str,
        status: str,
        current_value,
        threshold,
        affected_count: int,
        description: str | None,
    ) -> None:
        checks.append(
            {
                "id": check_id,
                "name": name,
                "category": category,
                "status": status,
                "last_run_at": now_iso,
                "current_value": current_value,
                "threshold": threshold,
                "affected_count": affected_count,
                "description": description,
            }
        )

    # 1) Quarantined odds (24h) - should be 0
    try:
        res = await session.execute(
            text(
                """
                SELECT COUNT(*) FROM odds_history
                WHERE quarantined = true
                  AND recorded_at > NOW() - INTERVAL '24 hours'
                """
            )
        )
        quarantined_odds_24h = int(res.scalar() or 0)
        add_check(
            check_id="dq_quarantined_odds_24h",
            name="Quarantined odds (24h)",
            category="odds",
            status=_dq_status_from_count(quarantined_odds_24h, fail_if_gt=0, warn_if_gt=0),
            current_value=quarantined_odds_24h,
            threshold=0,
            affected_count=quarantined_odds_24h,
            description="Odds snapshots flagged as quarantined in the last 24h (should be 0).",
        )
    except Exception as e:
        add_check(
            check_id="dq_quarantined_odds_24h",
            name="Quarantined odds (24h)",
            category="odds",
            status="warning",
            current_value=None,
            threshold=0,
            affected_count=0,
            description=f"Degraded: could not query odds_history ({str(e)[:80]}).",
        )

    # 2) Tainted matches (7d) - should be 0
    try:
        res = await session.execute(
            text(
                """
                SELECT COUNT(*) FROM matches
                WHERE tainted = true
                  AND date > NOW() - INTERVAL '7 days'
                """
            )
        )
        tainted_matches_7d = int(res.scalar() or 0)
        add_check(
            check_id="dq_tainted_matches_7d",
            name="Tainted matches (7d)",
            category="consistency",
            status=_dq_status_from_count(tainted_matches_7d, fail_if_gt=0, warn_if_gt=0),
            current_value=tainted_matches_7d,
            threshold=0,
            affected_count=tainted_matches_7d,
            description="Matches flagged as tainted in the last 7 days (should be 0).",
        )
    except Exception as e:
        add_check(
            check_id="dq_tainted_matches_7d",
            name="Tainted matches (7d)",
            category="consistency",
            status="warning",
            current_value=None,
            threshold=0,
            affected_count=0,
            description=f"Degraded: could not query matches.tainted ({str(e)[:80]}).",
        )

    # 3) Unmapped teams (missing logo_url) - warning if > 0
    try:
        res = await session.execute(
            text(
                """
                SELECT COUNT(DISTINCT t.id) FROM teams t
                WHERE t.logo_url IS NULL
                """
            )
        )
        unmapped_teams = int(res.scalar() or 0)
        add_check(
            check_id="dq_unmapped_teams",
            name="Unmapped teams (missing logo)",
            category="completeness",
            status=_dq_status_from_count(unmapped_teams, fail_if_gt=999999999, warn_if_gt=0),
            current_value=unmapped_teams,
            threshold=0,
            affected_count=unmapped_teams,
            description="Teams missing logo_url (proxy for incomplete mapping).",
        )
    except Exception as e:
        add_check(
            check_id="dq_unmapped_teams",
            name="Unmapped teams (missing logo)",
            category="completeness",
            status="warning",
            current_value=None,
            threshold=0,
            affected_count=0,
            description=f"Degraded: could not query teams.logo_url ({str(e)[:80]}).",
        )

    # 4) Odds desync near-term (6h) - warning if > 0
    try:
        res = await session.execute(
            text(
                """
                SELECT COUNT(DISTINCT m.id)
                FROM matches m
                JOIN odds_snapshots os ON os.match_id = m.id
                WHERE m.status = 'NS'
                  AND m.date BETWEEN NOW() AND NOW() + INTERVAL '6 hours'
                  AND os.odds_freshness = 'live'
                  AND os.snapshot_type = 'lineup_confirmed'
                  AND os.snapshot_at >= NOW() - INTERVAL '120 minutes'
                  AND (m.odds_home IS NULL OR m.odds_draw IS NULL OR m.odds_away IS NULL)
                """
            )
        )
        odds_desync_6h = int(res.scalar() or 0)
        add_check(
            check_id="dq_odds_desync_6h",
            name="Odds desync (next 6h)",
            category="freshness",
            status=_dq_status_from_count(odds_desync_6h, fail_if_gt=999999999, warn_if_gt=0),
            current_value=odds_desync_6h,
            threshold=0,
            affected_count=odds_desync_6h,
            description="NS matches in next 6h with live lineup_confirmed snapshot but NULL odds in matches.",
        )
    except Exception as e:
        add_check(
            check_id="dq_odds_desync_6h",
            name="Odds desync (next 6h)",
            category="freshness",
            status="warning",
            current_value=None,
            threshold=0,
            affected_count=0,
            description=f"Degraded: could not query odds_snapshots/matches odds fields ({str(e)[:80]}).",
        )

    # 5) Odds desync critical (90m) - failing if > 0
    try:
        res = await session.execute(
            text(
                """
                SELECT COUNT(DISTINCT m.id)
                FROM matches m
                JOIN odds_snapshots os ON os.match_id = m.id
                WHERE m.status = 'NS'
                  AND m.date BETWEEN NOW() AND NOW() + INTERVAL '90 minutes'
                  AND os.odds_freshness = 'live'
                  AND os.snapshot_type = 'lineup_confirmed'
                  AND os.snapshot_at >= NOW() - INTERVAL '120 minutes'
                  AND (m.odds_home IS NULL OR m.odds_draw IS NULL OR m.odds_away IS NULL)
                """
            )
        )
        odds_desync_90m = int(res.scalar() or 0)
        add_check(
            check_id="dq_odds_desync_90m",
            name="Odds desync (next 90m)",
            category="odds",
            status=_dq_status_from_count(odds_desync_90m, fail_if_gt=0, warn_if_gt=0),
            current_value=odds_desync_90m,
            threshold=0,
            affected_count=odds_desync_90m,
            description="CRITICAL: NS matches in next 90m missing odds despite recent live lineup_confirmed snapshot.",
        )
    except Exception as e:
        add_check(
            check_id="dq_odds_desync_90m",
            name="Odds desync (next 90m)",
            category="odds",
            status="warning",
            current_value=None,
            threshold=0,
            affected_count=0,
            description=f"Degraded: could not query odds_snapshots/matches odds fields ({str(e)[:80]}).",
        )

    # =========================================================================
    # SOTA ENRICHMENT CHECKS (Understat, Weather, Venue Geo, Team Profiles)
    # =========================================================================

    # 6) Understat coverage: FT matches in last 14d (Top-5 leagues) with xG
    understat_league_ids = ",".join(str(lid) for lid in UNDERSTAT_SUPPORTED_LEAGUES)
    try:
        res = await session.execute(
            text(
                f"""
                SELECT
                    COUNT(*) FILTER (WHERE mut.match_id IS NOT NULL) AS with_xg,
                    COUNT(*) AS total_ft
                FROM matches m
                LEFT JOIN match_understat_team mut ON m.id = mut.match_id
                WHERE m.status IN ('FT', 'AET', 'PEN')
                  AND m.date >= NOW() - INTERVAL '14 days'
                  AND m.league_id IN ({understat_league_ids})
                """
            )
        )
        row = res.first()
        with_xg = int(row[0] or 0) if row else 0
        total_ft = int(row[1] or 0) if row else 0
        missing_xg = total_ft - with_xg
        coverage_pct = round(with_xg / total_ft * 100, 1) if total_ft > 0 else 0.0
        # Status: warn if coverage < 60%, fail if < 30%
        if coverage_pct < 30:
            status = "failing"
        elif coverage_pct < 60:
            status = "warning"
        else:
            status = "passing"
        add_check(
            check_id="dq_understat_coverage_ft_14d",
            name="Understat xG coverage (14d)",
            category="coverage",
            status=status,
            current_value=f"{coverage_pct}%",
            threshold="≥60%",
            affected_count=missing_xg,
            description=f"FT matches (Top-5 leagues, 14d) with xG data: {with_xg}/{total_ft}.",
        )
    except Exception as e:
        add_check(
            check_id="dq_understat_coverage_ft_14d",
            name="Understat xG coverage (14d)",
            category="coverage",
            status="warning",
            current_value=None,
            threshold="≥60%",
            affected_count=0,
            description=f"Degraded: could not query match_understat_team ({str(e)[:80]}).",
        )

    # 7) Weather coverage: NS matches in next 48h with forecasts
    try:
        res = await session.execute(
            text(
                """
                SELECT
                    COUNT(*) FILTER (WHERE mw.match_id IS NOT NULL) AS with_weather,
                    COUNT(*) AS total_ns
                FROM matches m
                LEFT JOIN match_weather mw ON m.id = mw.match_id
                WHERE m.status = 'NS'
                  AND m.date >= NOW()
                  AND m.date < NOW() + INTERVAL '48 hours'
                """
            )
        )
        row = res.first()
        with_weather = int(row[0] or 0) if row else 0
        total_ns = int(row[1] or 0) if row else 0
        missing_weather = total_ns - with_weather
        coverage_pct = round(with_weather / total_ns * 100, 1) if total_ns > 0 else 0.0
        # Status: warn if coverage < 30%, fail if < 10% (weather is optional SOTA feature)
        if coverage_pct < 10:
            status = "failing"
        elif coverage_pct < 30:
            status = "warning"
        else:
            status = "passing"
        add_check(
            check_id="dq_weather_coverage_ns_48h",
            name="Weather coverage (NS 48h)",
            category="coverage",
            status=status,
            current_value=f"{coverage_pct}%",
            threshold="≥30%",
            affected_count=missing_weather,
            description=f"NS matches (48h) with weather forecast: {with_weather}/{total_ns}.",
        )
    except Exception as e:
        add_check(
            check_id="dq_weather_coverage_ns_48h",
            name="Weather coverage (NS 48h)",
            category="coverage",
            status="warning",
            current_value=None,
            threshold="≥30%",
            affected_count=0,
            description=f"Degraded: could not query match_weather ({str(e)[:80]}).",
        )

    # 8) Venue geo coverage: venues from last 30d matches with coordinates
    # Note: venue_geo uses venue_city as key
    try:
        res = await session.execute(
            text(
                """
                SELECT
                    COUNT(DISTINCT vg.venue_city) AS with_geo,
                    COUNT(DISTINCT m.venue_city) AS total_venues
                FROM matches m
                LEFT JOIN venue_geo vg ON m.venue_city = vg.venue_city
                WHERE m.venue_city IS NOT NULL
                  AND m.date >= NOW() - INTERVAL '30 days'
                """
            )
        )
        row = res.first()
        with_geo = int(row[0] or 0) if row else 0
        total_venues = int(row[1] or 0) if row else 0
        missing_geo = total_venues - with_geo
        coverage_pct = round(with_geo / total_venues * 100, 1) if total_venues > 0 else 0.0
        # Status: warn if coverage < 30%, fail if < 10%
        if coverage_pct < 10:
            status = "failing"
        elif coverage_pct < 30:
            status = "warning"
        else:
            status = "passing"
        add_check(
            check_id="dq_venue_geo_coverage",
            name="Venue geo coverage (30d)",
            category="coverage",
            status=status,
            current_value=f"{coverage_pct}%",
            threshold="≥30%",
            affected_count=missing_geo,
            description=f"Venues from recent matches with coordinates: {with_geo}/{total_venues}.",
        )
    except Exception as e:
        add_check(
            check_id="dq_venue_geo_coverage",
            name="Venue geo coverage (30d)",
            category="coverage",
            status="warning",
            current_value=None,
            threshold="≥30%",
            affected_count=0,
            description=f"Degraded: could not query venue_geo ({str(e)[:80]}).",
        )

    # 9) Team home city profile coverage
    try:
        res = await session.execute(
            text(
                """
                SELECT
                    COUNT(*) FILTER (WHERE thcp.team_id IS NOT NULL) AS with_profile,
                    COUNT(*) AS total_teams
                FROM teams t
                LEFT JOIN team_home_city_profile thcp ON t.id = thcp.team_id
                """
            )
        )
        row = res.first()
        with_profile = int(row[0] or 0) if row else 0
        total_teams = int(row[1] or 0) if row else 0
        missing_profile = total_teams - with_profile
        coverage_pct = round(with_profile / total_teams * 100, 1) if total_teams > 0 else 0.0
        # Status: warn if coverage < 20%, fail if < 5%
        if coverage_pct < 5:
            status = "failing"
        elif coverage_pct < 20:
            status = "warning"
        else:
            status = "passing"
        add_check(
            check_id="dq_team_profile_coverage",
            name="Team profile coverage",
            category="coverage",
            status=status,
            current_value=f"{coverage_pct}%",
            threshold="≥20%",
            affected_count=missing_profile,
            description=f"Teams with home city profile: {with_profile}/{total_teams}.",
        )
    except Exception as e:
        add_check(
            check_id="dq_team_profile_coverage",
            name="Team profile coverage",
            category="coverage",
            status="warning",
            current_value=None,
            threshold="≥20%",
            affected_count=0,
            description=f"Degraded: could not query team_home_city_profile ({str(e)[:80]}).",
        )

    # 10) Sofascore XI coverage: NS matches in next 48h (supported leagues) with XI data
    try:
        # Check if table exists first
        table_check = await session.execute(
            text("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables
                    WHERE table_name = 'match_sofascore_lineup'
                )
            """)
        )
        tables_exist = table_check.scalar()

        if tables_exist:
            sofascore_league_ids = ",".join(str(lid) for lid in SOFASCORE_SUPPORTED_LEAGUES)
            res = await session.execute(
                text(
                    f"""
                    SELECT
                        COUNT(*) FILTER (WHERE msl.match_id IS NOT NULL) AS with_xi,
                        COUNT(*) AS total_ns
                    FROM matches m
                    LEFT JOIN match_sofascore_lineup msl ON m.id = msl.match_id
                    WHERE m.status = 'NS'
                      AND m.date >= NOW()
                      AND m.date < NOW() + INTERVAL '48 hours'
                      AND m.league_id IN ({sofascore_league_ids})
                    """
                )
            )
            row = res.first()
            with_xi = int(row[0] or 0) if row else 0
            total_ns = int(row[1] or 0) if row else 0
            missing_xi = total_ns - with_xi
            coverage_pct = round(with_xi / total_ns * 100, 1) if total_ns > 0 else 0.0
            # Status: warn if coverage < 20%, fail if < 5% (XI is optional SOTA feature)
            if coverage_pct < 5:
                status = "failing"
            elif coverage_pct < 20:
                status = "warning"
            else:
                status = "passing"
            add_check(
                check_id="dq_sofascore_xi_coverage_ns_48h",
                name="Sofascore XI coverage (NS 48h)",
                category="coverage",
                status=status,
                current_value=f"{coverage_pct}%",
                threshold="≥20%",
                affected_count=missing_xi,
                description=f"NS matches (48h, supported leagues) with XI data: {with_xi}/{total_ns}.",
            )
        else:
            # Tables not deployed yet
            add_check(
                check_id="dq_sofascore_xi_coverage_ns_48h",
                name="Sofascore XI coverage (NS 48h)",
                category="coverage",
                status="warning",
                current_value="pending",
                threshold="≥20%",
                affected_count=0,
                description="Tables not deployed yet (migration 030 pending).",
            )
    except Exception as e:
        add_check(
            check_id="dq_sofascore_xi_coverage_ns_48h",
            name="Sofascore XI coverage (NS 48h)",
            category="coverage",
            status="warning",
            current_value=None,
            threshold="≥20%",
            affected_count=0,
            description=f"Degraded: could not query match_sofascore_lineup ({str(e)[:80]}).",
        )

    return checks


@router.get("/dashboard/data_quality.json")
async def dashboard_data_quality_json(
    request: Request,
    status: list[str] | None = Query(default=None),  # passing|warning|failing (multi)
    category: list[str] | None = Query(default=None),  # coverage|consistency|completeness|freshness|odds (multi)
    q: str | None = None,
    page: int = 1,
    limit: int = 50,
    session: AsyncSession = Depends(get_async_session),
):
    """
    Data Quality checks (tabular) for dashboard /data-quality page.
    Auth: X-Dashboard-Token header required.
    """
    if not verify_dashboard_token_bool(request):
        raise HTTPException(status_code=401, detail="Dashboard access requires valid token.")

    page = max(page, 1)
    limit = min(max(limit, 1), 100)

    status_filters = [s.lower() for s in _normalize_multi_param(status)]
    category_filters = [c.lower() for c in _normalize_multi_param(category)]
    q_norm = (q or "").strip().lower()

    cache_key = f"dq:{','.join(status_filters) or '-'}:{','.join(category_filters) or '-'}:{q_norm or '-'}:{page}:{limit}"
    now = time.time()

    if cache_key in _data_quality_cache:
        cached_entry = _data_quality_cache[cache_key]
        age = now - cached_entry["timestamp"]
        if age < _DATA_QUALITY_CACHE_TTL:
            return {
                "generated_at": cached_entry["generated_at"],
                "cached": True,
                "cache_age_seconds": round(age, 1),
                "data": cached_entry["data"],
            }

    checks = await _build_data_quality_checks(session)

    # Filter
    filtered = []
    for c in checks:
        c_status = str(c.get("status") or "").lower()
        c_cat = str(c.get("category") or "").lower()
        if status_filters and c_status not in status_filters:
            continue
        if category_filters and c_cat not in category_filters:
            continue
        if q_norm:
            hay = " ".join(
                [
                    str(c.get("id") or ""),
                    str(c.get("name") or ""),
                    str(c.get("description") or ""),
                    str(c.get("category") or ""),
                    str(c.get("status") or ""),
                ]
            ).lower()
            if q_norm not in hay:
                continue
        filtered.append(c)

    total = len(filtered)
    pages = (total + limit - 1) // limit if limit > 0 else 1
    start = (page - 1) * limit
    end = start + limit
    page_checks = filtered[start:end]

    generated_at = datetime.utcnow().isoformat() + "Z"
    response_data = {
        "checks": page_checks,
        "total": total,
        "page": page,
        "limit": limit,
        "pages": pages,
    }

    _data_quality_cache[cache_key] = {
        "generated_at": generated_at,
        "timestamp": now,
        "data": response_data,
    }
    # Simple cache pruning
    if len(_data_quality_cache) > 50:
        oldest_key = min(_data_quality_cache.keys(), key=lambda k: _data_quality_cache[k]["timestamp"])
        del _data_quality_cache[oldest_key]

    return {
        "generated_at": generated_at,
        "cached": False,
        "cache_age_seconds": 0,
        "data": response_data,
    }


@router.get("/dashboard/data_quality/{check_id}.json")
async def dashboard_data_quality_check_json(
    check_id: str,
    request: Request,
    limit: int = 50,
    session: AsyncSession = Depends(get_async_session),
):
    """
    Data Quality check detail endpoint.
    Best-effort and no PII. Returns affected_items (limited) and empty history for now.
    Auth: X-Dashboard-Token header required.
    """
    if not verify_dashboard_token_bool(request):
        raise HTTPException(status_code=401, detail="Dashboard access requires valid token.")

    limit = min(max(limit, 1), 100)
    cache_key = f"dq_detail:{check_id}:{limit}"
    now = time.time()

    if cache_key in _data_quality_detail_cache:
        cached_entry = _data_quality_detail_cache[cache_key]
        age = now - cached_entry["timestamp"]
        if age < _DATA_QUALITY_DETAIL_CACHE_TTL:
            return {
                "generated_at": cached_entry["generated_at"],
                "cached": True,
                "cache_age_seconds": round(age, 1),
                "data": cached_entry["data"],
            }

    # Build current check list and locate the check
    checks = await _build_data_quality_checks(session)
    check = next((c for c in checks if str(c.get("id")) == check_id), None)
    if not check:
        raise HTTPException(status_code=404, detail="Unknown check_id")

    affected_items: list[dict] = []

    try:
        if check_id == "dq_quarantined_odds_24h":
            res = await session.execute(
                text(
                    """
                    SELECT id, match_id, recorded_at, source
                    FROM odds_history
                    WHERE quarantined = true
                      AND recorded_at > NOW() - INTERVAL '24 hours'
                    ORDER BY recorded_at DESC
                    LIMIT :limit
                    """
                ),
                {"limit": limit},
            )
            affected_items = [
                {
                    "odds_history_id": int(r[0]),
                    "match_id": int(r[1]) if r[1] is not None else None,
                    "recorded_at": r[2].isoformat() + "Z" if r[2] else None,
                    "source": r[3],
                }
                for r in res.fetchall()
            ]

        elif check_id == "dq_tainted_matches_7d":
            res = await session.execute(
                text(
                    """
                    SELECT id, date, league_id, status
                    FROM matches
                    WHERE tainted = true
                      AND date > NOW() - INTERVAL '7 days'
                    ORDER BY date DESC
                    LIMIT :limit
                    """
                ),
                {"limit": limit},
            )
            affected_items = [
                {
                    "match_id": int(r[0]),
                    "date": r[1].isoformat() + "Z" if r[1] else None,
                    "league_id": int(r[2]) if r[2] is not None else None,
                    "status": r[3],
                }
                for r in res.fetchall()
            ]

        elif check_id == "dq_unmapped_teams":
            res = await session.execute(
                text(
                    """
                    SELECT id, name, country
                    FROM teams
                    WHERE logo_url IS NULL
                    ORDER BY id ASC
                    LIMIT :limit
                    """
                ),
                {"limit": limit},
            )
            affected_items = [
                {
                    "team_id": int(r[0]),
                    "name": r[1],
                    "country": r[2],
                }
                for r in res.fetchall()
            ]

        elif check_id in ("dq_odds_desync_6h", "dq_odds_desync_90m"):
            window = "6 hours" if check_id == "dq_odds_desync_6h" else "90 minutes"
            res = await session.execute(
                text(
                    f"""
                    SELECT DISTINCT m.id, m.date, m.league_id
                    FROM matches m
                    JOIN odds_snapshots os ON os.match_id = m.id
                    WHERE m.status = 'NS'
                      AND m.date BETWEEN NOW() AND NOW() + INTERVAL '{window}'
                      AND os.odds_freshness = 'live'
                      AND os.snapshot_type = 'lineup_confirmed'
                      AND os.snapshot_at >= NOW() - INTERVAL '120 minutes'
                      AND (m.odds_home IS NULL OR m.odds_draw IS NULL OR m.odds_away IS NULL)
                    ORDER BY m.date ASC
                    LIMIT :limit
                    """
                ),
                {"limit": limit},
            )
            affected_items = [
                {
                    "match_id": int(r[0]),
                    "kickoff_utc": r[1].isoformat() + "Z" if r[1] else None,
                    "league_id": int(r[2]) if r[2] is not None else None,
                }
                for r in res.fetchall()
            ]
    except Exception as e:
        # Best-effort: don't fail endpoint if affected_items query fails
        affected_items = [{"error": f"degraded: {str(e)[:120]}"}]

    generated_at = datetime.utcnow().isoformat() + "Z"
    response_data = {
        "check": check,
        "affected_items": affected_items,
        "history": [],  # Placeholder for future: time-series snapshots per check
    }

    _data_quality_detail_cache[cache_key] = {
        "generated_at": generated_at,
        "timestamp": now,
        "data": response_data,
    }
    if len(_data_quality_detail_cache) > 50:
        oldest_key = min(_data_quality_detail_cache.keys(), key=lambda k: _data_quality_detail_cache[k]["timestamp"])
        del _data_quality_detail_cache[oldest_key]

    return {
        "generated_at": generated_at,
        "cached": False,
        "cache_age_seconds": 0,
        "data": response_data,
    }

# Dashboard Settings endpoints moved to app/dashboard/settings_routes.py



# =============================================================================
# DASHBOARD PREDICTIONS (read-only, for ops dashboard)
# =============================================================================

_dashboard_predictions_cache: dict = {"data": None, "timestamp": 0, "ttl": 45}


@router.get("/dashboard/predictions.json")
async def dashboard_predictions_json(
    request: Request,
    league_ids: str | None = Query(default=None, description="Comma-separated league IDs"),
    status: str | None = Query(default=None, description="Match status filter (NS,LIVE,FT,etc)"),
    model: str | None = Query(default=None, description="Model filter (baseline,shadow)"),
    q: str | None = Query(default=None, description="Search by team name"),
    days_back: int = Query(default=0, ge=0, le=30, description="Days back from today"),
    days_ahead: int = Query(default=3, ge=0, le=14, description="Days ahead from today"),
    page: int = Query(default=1, ge=1, description="Page number"),
    limit: int = Query(default=50, ge=1, le=100, description="Items per page"),
    session: AsyncSession = Depends(get_async_session),
):
    """
    Read-only predictions list for Dashboard.

    Auth: X-Dashboard-Token required.
    TTL: 45s cache.

    Query params:
    - league_ids: Comma-separated league IDs (e.g., "39,140,135")
    - status: Match status filter (NS, LIVE, FT, etc.)
    - model: Model filter (baseline, shadow)
    - q: Search by team name
    - days_back: Days back from today (default 0)
    - days_ahead: Days ahead from today (default 3)
    - page: Page number (default 1)
    - limit: Items per page (default 50, max 100)

    SECURITY: No secrets/PII. Public match data only.
    """
    if not verify_dashboard_token_bool(request):
        raise HTTPException(status_code=401, detail="Dashboard access requires valid token.")

    now = time.time()
    generated_at = datetime.utcnow().isoformat() + "Z"

    try:
        from sqlalchemy import text

        # Build date range
        today = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        from_date = today - timedelta(days=days_back)
        to_date = today + timedelta(days=days_ahead + 1)  # +1 to include full day

        # Build query with filters
        filters = ["m.date >= :from_date", "m.date < :to_date"]
        params: dict = {"from_date": from_date, "to_date": to_date}

        # League filter
        if league_ids:
            try:
                league_list = [int(lid.strip()) for lid in league_ids.split(",") if lid.strip()]
                if league_list:
                    filters.append(f"m.league_id IN ({','.join(str(l) for l in league_list)})")
            except ValueError:
                pass  # Invalid league_ids, ignore filter

        # Status filter
        if status:
            status_list = [s.strip().upper() for s in status.split(",") if s.strip()]
            if status_list:
                status_placeholders = ",".join(f"'{s}'" for s in status_list)
                filters.append(f"m.status IN ({status_placeholders})")

        # Model filter (baseline vs shadow)
        # IMPORTANT: Shadow predictions are read from shadow_predictions table (canonical source)
        use_shadow_table = model and model.lower() == "shadow"

        model_filter_sql = ""
        if model and model.lower() == "baseline":
            model_filter_sql = "AND p.model_version NOT LIKE '%shadow%' AND p.model_version NOT LIKE '%two_stage%'"
        # Note: shadow case is handled separately below with shadow_predictions table

        # Search filter
        if q:
            filters.append("(t_home.name ILIKE :q OR t_away.name ILIKE :q)")
            params["q"] = f"%{q}%"

        where_clause = " AND ".join(filters)

        # League names fallback (no competitions table)
        league_names = {
            1: "World Cup", 2: "Champions League", 3: "Europa League",
            39: "Premier League", 40: "Championship", 61: "Ligue 1",
            78: "Bundesliga", 135: "Serie A", 140: "La Liga",
            94: "Primeira Liga", 88: "Eredivisie", 203: "Süper Lig",
            239: "Liga BetPlay", 253: "MLS", 262: "Liga MX",
            128: "Argentina Primera", 71: "Brasileirão",
            848: "Conference League", 45: "FA Cup", 143: "Copa del Rey",
        }

        # =================================================================
        # SHADOW PATH: Read from shadow_predictions table (canonical source)
        # =================================================================
        if use_shadow_table:
            # Build shadow-specific filters (using sp alias)
            shadow_filters = ["m.date >= :from_date", "m.date < :to_date"]

            # League filter
            if league_ids:
                try:
                    league_list = [int(lid.strip()) for lid in league_ids.split(",") if lid.strip()]
                    if league_list:
                        shadow_filters.append(f"m.league_id IN ({','.join(str(l) for l in league_list)})")
                except ValueError:
                    pass

            # Status filter
            if status:
                status_list = [s.strip().upper() for s in status.split(",") if s.strip()]
                if status_list:
                    status_placeholders = ",".join(f"'{s}'" for s in status_list)
                    shadow_filters.append(f"m.status IN ({status_placeholders})")

            # Search filter
            if q:
                shadow_filters.append("(t_home.name ILIKE :q OR t_away.name ILIKE :q)")

            shadow_where = " AND ".join(shadow_filters)

            # Count from shadow_predictions
            shadow_count_query = f"""
                SELECT COUNT(DISTINCT sp.id)
                FROM shadow_predictions sp
                JOIN matches m ON sp.match_id = m.id
                JOIN teams t_home ON m.home_team_id = t_home.id
                JOIN teams t_away ON m.away_team_id = t_away.id
                WHERE {shadow_where}
            """
            count_result = await session.execute(text(shadow_count_query), params)
            total = int(count_result.scalar() or 0)

            pages = (total + limit - 1) // limit if limit > 0 else 1
            offset_val = (page - 1) * limit

            # Fetch shadow predictions
            shadow_data_query = f"""
                SELECT
                    sp.id,
                    sp.match_id,
                    m.league_id,
                    m.date AS kickoff_utc,
                    t_home.name AS home_team,
                    t_away.name AS away_team,
                    m.status,
                    m.home_goals,
                    m.away_goals,
                    sp.shadow_version,
                    sp.shadow_architecture,
                    sp.shadow_home_prob,
                    sp.shadow_draw_prob,
                    sp.shadow_away_prob,
                    sp.shadow_predicted,
                    sp.shadow_correct,
                    sp.created_at
                FROM shadow_predictions sp
                JOIN matches m ON sp.match_id = m.id
                JOIN teams t_home ON m.home_team_id = t_home.id
                JOIN teams t_away ON m.away_team_id = t_away.id
                WHERE {shadow_where}
                ORDER BY m.date ASC, sp.created_at DESC
                LIMIT :limit OFFSET :offset
            """
            params["limit"] = limit
            params["offset"] = offset_val

            result = await session.execute(text(shadow_data_query), params)
            rows = result.fetchall()

            # Format shadow predictions
            # Map shadow_predicted to pick format (home/draw/away)
            # Handles both formats: "H/D/A" or "home/draw/away"
            def map_predicted_to_pick(predicted: str | None) -> str | None:
                if not predicted:
                    return None
                p = predicted.lower()
                if p in ("home", "h"):
                    return "home"
                elif p in ("draw", "d"):
                    return "draw"
                elif p in ("away", "a"):
                    return "away"
                return None

            predictions = []
            for row in rows:
                probs = {
                    "home": round(row.shadow_home_prob, 3) if row.shadow_home_prob else None,
                    "draw": round(row.shadow_draw_prob, 3) if row.shadow_draw_prob else None,
                    "away": round(row.shadow_away_prob, 3) if row.shadow_away_prob else None,
                }

                # Use shadow_predicted from DB (avoids rounding issues with co-pick)
                pick = map_predicted_to_pick(row.shadow_predicted)

                predictions.append({
                    "id": row.id,
                    "match_id": row.match_id,
                    "league_id": row.league_id,
                    "league_name": league_names.get(row.league_id, f"League {row.league_id}"),
                    "kickoff_utc": row.kickoff_utc.isoformat() + "Z" if row.kickoff_utc else None,
                    "home_team": row.home_team,
                    "away_team": row.away_team,
                    "status": row.status,
                    "score": f"{row.home_goals}-{row.away_goals}" if row.home_goals is not None else None,
                    "model": "shadow",
                    "model_version": row.shadow_version,
                    "architecture": row.shadow_architecture,
                    "pick": pick,
                    "probs": probs,
                    "predicted": row.shadow_predicted,
                    "is_correct": row.shadow_correct,
                    "is_frozen": None,  # Shadow doesn't use frozen concept
                    "frozen_at": None,
                    "confidence_tier": None,
                    "created_at": row.created_at.isoformat() + "Z" if row.created_at else None,
                })

            return {
                "generated_at": generated_at,
                "cached": False,
                "cache_age_seconds": 0,
                "source": "shadow_predictions",  # ABE requirement: trazabilidad
                "data": {
                    "predictions": predictions,
                    "total": total,
                    "page": page,
                    "limit": limit,
                    "pages": pages,
                    "filters_applied": {
                        "league_ids": league_ids,
                        "status": status,
                        "model": model,
                        "q": q,
                        "days_back": days_back,
                        "days_ahead": days_ahead,
                    },
                },
            }
        # =================================================================
        # END SHADOW PATH
        # =================================================================

        # Count total
        count_query = f"""
            SELECT COUNT(DISTINCT p.id)
            FROM predictions p
            JOIN matches m ON p.match_id = m.id
            JOIN teams t_home ON m.home_team_id = t_home.id
            JOIN teams t_away ON m.away_team_id = t_away.id
            WHERE {where_clause} {model_filter_sql}
        """
        count_result = await session.execute(text(count_query), params)
        total = int(count_result.scalar() or 0)

        # Calculate pagination
        pages = (total + limit - 1) // limit if limit > 0 else 1
        offset = (page - 1) * limit

        # Fetch predictions with pagination
        data_query = f"""
            SELECT
                p.id,
                p.match_id,
                m.league_id,
                m.date AS kickoff_utc,
                t_home.name AS home_team,
                t_away.name AS away_team,
                m.status,
                m.home_goals,
                m.away_goals,
                p.model_version,
                p.home_prob,
                p.draw_prob,
                p.away_prob,
                p.is_frozen,
                p.frozen_at,
                p.frozen_confidence_tier,
                p.created_at
            FROM predictions p
            JOIN matches m ON p.match_id = m.id
            JOIN teams t_home ON m.home_team_id = t_home.id
            JOIN teams t_away ON m.away_team_id = t_away.id
            WHERE {where_clause} {model_filter_sql}
            ORDER BY m.date ASC, p.created_at DESC
            LIMIT :limit OFFSET :offset
        """
        params["limit"] = limit
        params["offset"] = offset

        result = await session.execute(text(data_query), params)
        rows = result.fetchall()

        # Format predictions
        predictions = []
        for row in rows:
            # Determine pick based on highest probability
            probs = {
                "home": round(row.home_prob, 3) if row.home_prob else None,
                "draw": round(row.draw_prob, 3) if row.draw_prob else None,
                "away": round(row.away_prob, 3) if row.away_prob else None,
            }

            pick = None
            if probs["home"] and probs["draw"] and probs["away"]:
                max_prob = max(probs["home"], probs["draw"], probs["away"])
                if probs["home"] == max_prob:
                    pick = "home"
                elif probs["draw"] == max_prob:
                    pick = "draw"
                else:
                    pick = "away"

            # Determine model type
            model_version = row.model_version or ""
            if "shadow" in model_version.lower() or "two_stage" in model_version.lower():
                model_type = "shadow"
            else:
                model_type = "baseline"

            predictions.append({
                "id": row.id,
                "match_id": row.match_id,
                "league_id": row.league_id,
                "league_name": league_names.get(row.league_id, f"League {row.league_id}"),
                "kickoff_utc": row.kickoff_utc.isoformat() + "Z" if row.kickoff_utc else None,
                "home_team": row.home_team,
                "away_team": row.away_team,
                "status": row.status,
                "score": f"{row.home_goals}-{row.away_goals}" if row.home_goals is not None else None,
                "model": model_type,
                "model_version": model_version,
                "pick": pick,
                "probs": probs,
                "is_frozen": row.is_frozen,
                "frozen_at": row.frozen_at.isoformat() + "Z" if row.frozen_at else None,
                "confidence_tier": row.frozen_confidence_tier,
                "created_at": row.created_at.isoformat() + "Z" if row.created_at else None,
            })

        return {
            "generated_at": generated_at,
            "cached": False,
            "cache_age_seconds": 0,
            "source": "predictions",  # Canonical source for baseline models
            "data": {
                "predictions": predictions,
                "total": total,
                "page": page,
                "limit": limit,
                "pages": pages,
                "filters_applied": {
                    "league_ids": league_ids,
                    "status": status,
                    "model": model,
                    "q": q,
                    "days_back": days_back,
                    "days_ahead": days_ahead,
                },
            },
        }

    except Exception as e:
        logger.error(f"[DASHBOARD] predictions.json error: {e}", exc_info=True)
        return {
            "generated_at": generated_at,
            "cached": False,
            "cache_age_seconds": 0,
            "data": {
                "predictions": [],
                "total": 0,
                "page": page,
                "limit": limit,
                "pages": 0,
                "status": "degraded",
                "error": str(e)[:100],
            },
        }


# =============================================================================
# DASHBOARD ANALYTICS REPORTS (catalog for analytics table)
# =============================================================================

_analytics_reports_cache: dict = {"data": None, "timestamp": 0, "ttl": 120}


async def _build_analytics_reports(session: AsyncSession) -> list[dict]:
    """
    Build the list of analytics reports from various sources.

    Sources:
    - prediction_performance_reports (model_performance)
    - pit_reports (prediction_accuracy)
    - ops_daily_rollups (system_metrics)
    - job_runs / api usage metrics (api_usage)

    Returns list of report dicts with stable IDs.
    """
    from sqlalchemy import text

    reports = []

    # 1) Model Performance Reports (from prediction_performance_reports)
    try:
        result = await session.execute(text("""
            SELECT id, generated_at, window_days, report_date, payload, source
            FROM prediction_performance_reports
            ORDER BY generated_at DESC
            LIMIT 10
        """))
        rows = result.fetchall()

        for row in rows:
            payload = row.payload or {}
            overall = payload.get("overall", {})
            accuracy = overall.get("accuracy")
            total_matches = overall.get("total_matches", 0)

            # Determine status
            status = "ok"
            if accuracy is not None and accuracy < 0.35:
                status = "warning"
            elif row.generated_at and (datetime.utcnow() - row.generated_at).days > 2:
                status = "stale"

            reports.append({
                "id": f"model_perf_{row.window_days}d_{row.id}",
                "type": "model_performance",
                "title": f"Model Performance ({row.window_days}d)",
                "subtitle": f"Report date: {row.report_date}" if row.report_date else None,
                "status": status,
                "updated_at": row.generated_at.isoformat() + "Z" if row.generated_at else None,
                "tags": ["xgb", f"{row.window_days}d", row.source or "auto"],
                "summary": {
                    "primary_label": "Accuracy",
                    "primary_value": f"{accuracy:.1%}" if accuracy else "N/A",
                    "secondary_label": "Matches",
                    "secondary_value": total_matches,
                },
                "links": [
                    {"title": "Performance API", "url": f"/dashboard/ops/predictions_performance.json?window_days={row.window_days}"},
                ],
            })
    except Exception as e:
        logger.warning(f"[ANALYTICS] Failed to load model_performance reports: {e}")
        reports.append({
            "id": "model_perf_error",
            "type": "model_performance",
            "title": "Model Performance",
            "subtitle": "Error loading reports",
            "status": "warning",
            "updated_at": None,
            "tags": ["error"],
            "summary": {"primary_label": "Status", "primary_value": "unavailable"},
            "links": [],
        })

    # 2) Prediction Accuracy Reports (from pit_reports)
    try:
        result = await session.execute(text("""
            SELECT id, report_type, report_date, payload, source, created_at, updated_at
            FROM pit_reports
            WHERE report_type IN ('daily', 'weekly')
            ORDER BY created_at DESC
            LIMIT 10
        """))
        rows = result.fetchall()

        for row in rows:
            payload = row.payload or {}
            summary_data = payload.get("summary", {})
            accuracy = summary_data.get("accuracy")
            total = summary_data.get("total_predictions", 0)

            # Determine status
            status = "ok"
            if row.created_at and (datetime.utcnow() - row.created_at).days > 3:
                status = "stale"
            elif accuracy is not None and accuracy < 0.30:
                status = "warning"

            reports.append({
                "id": f"pit_{row.report_type}_{row.id}",
                "type": "prediction_accuracy",
                "title": f"PIT Report ({row.report_type.capitalize()})",
                "subtitle": f"Date: {row.report_date.strftime('%Y-%m-%d')}" if row.report_date else None,
                "status": status,
                "updated_at": (row.updated_at or row.created_at).isoformat() + "Z" if (row.updated_at or row.created_at) else None,
                "tags": ["pit", row.report_type, row.source or "auto"],
                "summary": {
                    "primary_label": "Accuracy",
                    "primary_value": f"{accuracy:.1%}" if accuracy else "N/A",
                    "secondary_label": "Predictions",
                    "secondary_value": total,
                },
                "links": [
                    {"title": "PIT Protocol", "url": "docs/PIT_EVALUATION_PROTOCOL.md"},
                ],
            })
    except Exception as e:
        logger.warning(f"[ANALYTICS] Failed to load pit_reports: {e}")

    # 3) System Metrics (from ops_daily_rollups)
    try:
        result = await session.execute(text("""
            SELECT day, payload, created_at, updated_at
            FROM ops_daily_rollups
            ORDER BY day DESC
            LIMIT 7
        """))
        rows = result.fetchall()

        for row in rows:
            payload = row.payload or {}
            api_calls = payload.get("api_calls_total", 0)
            predictions_generated = payload.get("predictions_generated", 0)
            job_status = payload.get("jobs_status", "unknown")

            # Determine status
            status = "ok"
            if job_status == "error":
                status = "warning"
            elif row.day and (datetime.utcnow().date() - row.day).days > 2:
                status = "stale"

            reports.append({
                "id": f"ops_rollup_{row.day.isoformat()}",
                "type": "system_metrics",
                "title": f"Daily Ops Rollup",
                "subtitle": f"Date: {row.day.isoformat()}",
                "status": status,
                "updated_at": (row.updated_at or row.created_at).isoformat() + "Z" if (row.updated_at or row.created_at) else None,
                "tags": ["ops", "daily", "system"],
                "summary": {
                    "primary_label": "API Calls",
                    "primary_value": api_calls,
                    "secondary_label": "Predictions",
                    "secondary_value": predictions_generated,
                },
                "links": [
                    {"title": "History API", "url": "/dashboard/ops/history.json?days=30"},
                ],
            })
    except Exception as e:
        logger.warning(f"[ANALYTICS] Failed to load ops_daily_rollups: {e}")

    # 4) API Usage / Job Health (synthetic from job_runs)
    try:
        result = await session.execute(text("""
            SELECT
                job_name,
                COUNT(*) as runs_24h,
                COUNT(*) FILTER (WHERE status = 'ok') as success_count,
                MAX(started_at) as last_run
            FROM job_runs
            WHERE started_at > NOW() - INTERVAL '24 hours'
            GROUP BY job_name
            ORDER BY runs_24h DESC
            LIMIT 5
        """))
        rows = result.fetchall()

        if rows:
            total_runs = sum(r.runs_24h for r in rows)
            total_success = sum(r.success_count for r in rows)
            success_rate = total_success / total_runs if total_runs > 0 else 0

            status = "ok" if success_rate >= 0.95 else ("warning" if success_rate >= 0.80 else "stale")

            reports.append({
                "id": "api_usage_24h",
                "type": "api_usage",
                "title": "Job Executions (24h)",
                "subtitle": f"Top jobs: {', '.join(r.job_name[:20] for r in rows[:3])}",
                "status": status,
                "updated_at": datetime.utcnow().isoformat() + "Z",
                "tags": ["jobs", "api", "24h"],
                "summary": {
                    "primary_label": "Success Rate",
                    "primary_value": f"{success_rate:.1%}",
                    "secondary_label": "Total Runs",
                    "secondary_value": total_runs,
                },
                "links": [
                    {"title": "Job Runs API", "url": "/dashboard/ops/job_runs.json"},
                ],
            })
    except Exception as e:
        logger.warning(f"[ANALYTICS] Failed to load job_runs metrics: {e}")

    # 5) SOTA Enrichment Summary (synthetic)
    try:
        result = await session.execute(text("""
            SELECT
                (SELECT COUNT(*) FROM match_sofascore_lineup) as sofascore_lineups,
                (SELECT COUNT(*) FROM match_weather) as weather_forecasts,
                (SELECT COUNT(*) FROM venue_geo) as venue_geos,
                (SELECT COUNT(*) FROM match_understat_team) as understat_xg
        """))
        row = result.fetchone()

        if row:
            total_enrichments = (row.sofascore_lineups or 0) + (row.weather_forecasts or 0) + (row.venue_geos or 0) + (row.understat_xg or 0)

            reports.append({
                "id": "sota_enrichment_summary",
                "type": "system_metrics",
                "title": "SOTA Enrichment Summary",
                "subtitle": f"XI: {row.sofascore_lineups}, Weather: {row.weather_forecasts}, Geo: {row.venue_geos}",
                "status": "ok" if total_enrichments > 100 else "warning",
                "updated_at": datetime.utcnow().isoformat() + "Z",
                "tags": ["sota", "enrichment", "coverage"],
                "summary": {
                    "primary_label": "Total Enrichments",
                    "primary_value": total_enrichments,
                    "secondary_label": "Understat xG",
                    "secondary_value": row.understat_xg or 0,
                },
                "links": [
                    {"title": "Ops Dashboard", "url": "/dashboard/ops.json"},
                ],
            })
    except Exception as e:
        logger.warning(f"[ANALYTICS] Failed to load SOTA enrichment summary: {e}")

    return reports


@router.get("/dashboard/analytics/reports.json")
async def dashboard_analytics_reports(
    request: Request,
    type: str | None = Query(default=None, description="Report type filter"),
    q: str | None = Query(default=None, description="Search by title/subtitle"),
    status: str | None = Query(default=None, description="Status filter (ok|warning|stale)"),
    page: int = Query(default=1, ge=1, description="Page number"),
    limit: int = Query(default=50, ge=1, le=100, description="Items per page"),
    session: AsyncSession = Depends(get_async_session),
):
    """
    Analytics reports catalog for Dashboard table.

    Auth: X-Dashboard-Token required.
    TTL: 120s cache.

    Query params:
    - type: Filter by report type (model_performance, prediction_accuracy, system_metrics, api_usage)
    - q: Search by title or subtitle
    - status: Filter by status (ok, warning, stale)
    - page: Page number (default 1)
    - limit: Items per page (default 50, max 100)

    SECURITY: No secrets/PII. Aggregated metrics only.
    """
    if not verify_dashboard_token_bool(request):
        raise HTTPException(status_code=401, detail="Dashboard access requires valid token.")

    now = time.time()
    cache = _analytics_reports_cache
    generated_at = datetime.utcnow().isoformat() + "Z"

    try:
        # Check cache (only for unfiltered requests)
        cache_key = f"{type}:{q}:{status}:{page}:{limit}"
        use_cache = (type is None and q is None and status is None and page == 1 and limit == 50)

        if use_cache and cache["data"] and (now - cache["timestamp"]) < cache["ttl"]:
            return {
                "generated_at": cache["data"]["generated_at"],
                "cached": True,
                "cache_age_seconds": round(now - cache["timestamp"], 1),
                "data": cache["data"]["payload"],
            }

        # Build reports list
        all_reports = await _build_analytics_reports(session)

        # Apply filters
        filtered = all_reports

        if type:
            filtered = [r for r in filtered if r["type"] == type]

        if status:
            filtered = [r for r in filtered if r["status"] == status]

        if q:
            q_lower = q.lower()
            filtered = [
                r for r in filtered
                if q_lower in (r["title"] or "").lower()
                or q_lower in (r["subtitle"] or "").lower()
                or any(q_lower in tag.lower() for tag in r.get("tags", []))
            ]

        # Pagination
        total = len(filtered)
        pages = (total + limit - 1) // limit if limit > 0 else 1
        start = (page - 1) * limit
        end = start + limit
        paginated = filtered[start:end]

        payload = {
            "reports": paginated,
            "total": total,
            "page": page,
            "limit": limit,
            "pages": pages,
            "note": "best-effort, aggregated server-side",
        }

        # Update cache for default requests
        if use_cache:
            cache["data"] = {"generated_at": generated_at, "payload": payload}
            cache["timestamp"] = now

        return {
            "generated_at": generated_at,
            "cached": False,
            "cache_age_seconds": 0,
            "data": payload,
        }

    except Exception as e:
        logger.error(f"[ANALYTICS] reports.json error: {e}", exc_info=True)
        return {
            "generated_at": generated_at,
            "cached": False,
            "cache_age_seconds": 0,
            "data": {
                "reports": [],
                "total": 0,
                "page": page,
                "limit": limit,
                "pages": 0,
                "status": "degraded",
                "error": str(e)[:100],
                "note": "best-effort, aggregated server-side",
            },
        }


# =============================================================================
# AUDIT LOGS ENDPOINT
# =============================================================================
_audit_logs_cache: dict = {"data": None, "timestamp": 0}
_AUDIT_LOGS_TTL = 90  # 90 seconds cache


def _parse_range_to_hours(range_str: str | None) -> int:
    """Convert range string to hours. Default 24h."""
    if not range_str:
        return 24
    range_map = {"1h": 1, "24h": 24, "7d": 168, "30d": 720}
    return range_map.get(range_str, 24)


def _derive_severity(result: str | None, error_message: str | None) -> str:
    """Derive severity from result/error_message."""
    if error_message or result == "error":
        return "error"
    if result == "warning" or result == "partial":
        return "warning"
    return "info"


def _derive_actor_kind(actor: str | None) -> str:
    """Derive actor_kind from actor field."""
    if not actor:
        return "system"
    actor_lower = actor.lower()
    if "scheduler" in actor_lower or "job" in actor_lower or "system" in actor_lower:
        return "system"
    if "token" in actor_lower or "key" in actor_lower or "user" in actor_lower:
        return "user"
    return "system"


async def _build_audit_events(
    session: AsyncSession,
    hours: int,
    types: list[str] | None,
    severities: list[str] | None,
    actor_kinds: list[str] | None,
    search: str | None,
) -> list[dict]:
    """Build audit events from multiple sources."""
    events = []
    cutoff = datetime.utcnow() - timedelta(hours=hours)

    # Source 1: ops_audit_log
    try:
        query = text("""
            SELECT id, action, request_id, actor, actor_id, result, error_message,
                   created_at, duration_ms
            FROM ops_audit_log
            WHERE created_at >= :cutoff
            ORDER BY created_at DESC
            LIMIT 500
        """)
        result = await session.execute(query, {"cutoff": cutoff})
        rows = result.mappings().all()

        for row in rows:
            severity = _derive_severity(row.get("result"), row.get("error_message"))
            actor_kind = _derive_actor_kind(row.get("actor"))

            # Build display message
            action = row.get("action", "unknown")
            result_str = row.get("result", "")
            duration = row.get("duration_ms")
            msg_parts = [f"{action}"]
            if result_str:
                msg_parts.append(f"→ {result_str}")
            if duration:
                msg_parts.append(f"({duration}ms)")
            message = " ".join(msg_parts)

            # Actor display (redact full IDs)
            actor_id = row.get("actor_id", "")
            actor_display = row.get("actor", "system")
            if actor_id and len(actor_id) > 4:
                actor_display = f"{actor_display}:{actor_id[:4]}…"

            events.append({
                "id": f"audit_{row['id']}",
                "type": action,
                "severity": severity,
                "actor_kind": actor_kind,
                "actor_display": actor_display,
                "message": message,
                "created_at": row["created_at"].isoformat() + "Z" if row.get("created_at") else None,
                "correlation_id": row.get("request_id"),
                "runbook_url": None,
            })
    except Exception as e:
        logger.warning(f"[AUDIT] Failed to load ops_audit_log: {e}")

    # Source 2: job_runs (scheduler jobs)
    try:
        query = text("""
            SELECT id, job_name, status, started_at, finished_at, duration_ms, error_message
            FROM job_runs
            WHERE created_at >= :cutoff
            ORDER BY created_at DESC
            LIMIT 500
        """)
        result = await session.execute(query, {"cutoff": cutoff})
        rows = result.mappings().all()

        for row in rows:
            status = row.get("status", "")
            severity = "error" if status == "error" else ("warning" if status == "partial" else "info")

            job_name = row.get("job_name", "unknown_job")
            duration = row.get("duration_ms")
            msg_parts = [f"job:{job_name}", f"→ {status}"]
            if duration:
                msg_parts.append(f"({duration}ms)")
            if row.get("error_message"):
                # Truncate error message
                err = str(row["error_message"])[:50]
                msg_parts.append(f"- {err}")
            message = " ".join(msg_parts)

            events.append({
                "id": f"job_{row['id']}",
                "type": f"job_{job_name}",
                "severity": severity,
                "actor_kind": "system",
                "actor_display": "scheduler",
                "message": message,
                "created_at": row["started_at"].isoformat() + "Z" if row.get("started_at") else None,
                "correlation_id": None,
                "runbook_url": "/docs/OPS_RUNBOOK.md" if severity == "error" else None,
            })
    except Exception as e:
        logger.warning(f"[AUDIT] Failed to load job_runs: {e}")

    # Sort all events by created_at DESC
    events.sort(key=lambda x: x.get("created_at") or "", reverse=True)

    # Apply filters
    if types:
        type_set = set(types)
        events = [e for e in events if e["type"] in type_set]

    if severities:
        sev_set = set(severities)
        events = [e for e in events if e["severity"] in sev_set]

    if actor_kinds:
        ak_set = set(actor_kinds)
        events = [e for e in events if e["actor_kind"] in ak_set]

    if search:
        search_lower = search.lower()
        events = [
            e for e in events
            if search_lower in (e.get("message") or "").lower()
            or search_lower in (e.get("type") or "").lower()
            or search_lower in (e.get("actor_display") or "").lower()
        ]

    return events


@router.get("/dashboard/audit_logs.json")
async def dashboard_audit_logs(
    request: Request,
    type: str | None = Query(default=None, description="Type filter (comma-separated for multi)"),
    severity: str | None = Query(default=None, description="Severity filter: info,warning,error (comma-separated)"),
    actor_kind: str | None = Query(default=None, description="Actor kind filter: system,user (comma-separated)"),
    q: str | None = Query(default=None, description="Search in message/type/actor"),
    range: str | None = Query(default="24h", description="Time range: 1h, 24h, 7d, 30d"),
    page: int = Query(default=1, ge=1, description="Page number"),
    limit: int = Query(default=50, ge=1, le=100, description="Items per page"),
    session: AsyncSession = Depends(get_async_session),
):
    """
    Audit logs for Dashboard monitoring.

    Auth: X-Dashboard-Token required.
    TTL: 90s cache.

    Query params:
    - type: Filter by event type (comma-separated for multi)
    - severity: Filter by severity: info, warning, error (comma-separated)
    - actor_kind: Filter by actor kind: system, user (comma-separated)
    - q: Search in message, type, or actor_display
    - range: Time range - 1h, 24h (default), 7d, 30d
    - page: Page number (default 1)
    - limit: Items per page (default 50, max 100)

    SECURITY: No PII/secrets/payloads. Redacted actor IDs.
    """
    if not verify_dashboard_token_bool(request):
        raise HTTPException(status_code=401, detail="Dashboard access requires valid token.")

    generated_at = datetime.utcnow().isoformat() + "Z"

    # Parse multi-value filters
    types = [t.strip() for t in type.split(",")] if type else None
    severities = [s.strip() for s in severity.split(",")] if severity else None
    actor_kinds = [a.strip() for a in actor_kind.split(",")] if actor_kind else None
    hours = _parse_range_to_hours(range)

    # Check if we can use cache (only for default/simple requests)
    use_cache = not types and not severities and not actor_kinds and not q and range == "24h"
    now = time.time()
    cache = _audit_logs_cache

    try:
        if use_cache and cache["data"] and (now - cache["timestamp"]) < _AUDIT_LOGS_TTL:
            cached_data = cache["data"]
            # Apply pagination to cached data
            all_events = cached_data["all_events"]
            total = len(all_events)
            pages = (total + limit - 1) // limit if limit > 0 else 1
            start = (page - 1) * limit
            end = start + limit
            paginated = all_events[start:end]

            return {
                "generated_at": cached_data["generated_at"],
                "cached": True,
                "cache_age_seconds": int(now - cache["timestamp"]),
                "data": {
                    "events": paginated,
                    "total": total,
                    "page": page,
                    "limit": limit,
                    "pages": pages,
                },
            }

        # Build events list
        all_events = await _build_audit_events(session, hours, types, severities, actor_kinds, q)

        # Pagination
        total = len(all_events)
        pages = (total + limit - 1) // limit if limit > 0 else 1
        start = (page - 1) * limit
        end = start + limit
        paginated = all_events[start:end]

        payload = {
            "events": paginated,
            "total": total,
            "page": page,
            "limit": limit,
            "pages": pages,
        }

        # Update cache for default requests
        if use_cache:
            cache["data"] = {"generated_at": generated_at, "all_events": all_events}
            cache["timestamp"] = now

        return {
            "generated_at": generated_at,
            "cached": False,
            "cache_age_seconds": 0,
            "data": payload,
        }

    except Exception as e:
        logger.error(f"[AUDIT] audit_logs.json error: {e}", exc_info=True)
        return {
            "generated_at": generated_at,
            "cached": False,
            "cache_age_seconds": 0,
            "data": {
                "events": [],
                "total": 0,
                "page": page,
                "limit": limit,
                "pages": 0,
                "status": "degraded",
                "error": str(e)[:100],
            },
        }


# =============================================================================
# TEAM LOGOS ENDPOINT
# =============================================================================
_team_logos_cache: dict = {"data": None, "timestamp": 0}
_TEAM_LOGOS_TTL = 3600  # 1 hour cache (logos rarely change)


@router.get("/dashboard/team_logos.json")
async def dashboard_team_logos(
    request: Request,
    session: AsyncSession = Depends(get_async_session),
):
    """
    Team logos map for Dashboard.

    Auth: X-Dashboard-Token required.
    TTL: 3600s (1 hour) cache - logos rarely change.

    Returns a map of team_name -> logo_url for efficient frontend lookup.
    """
    if not verify_dashboard_token_bool(request):
        raise HTTPException(status_code=401, detail="Dashboard access requires valid token.")

    generated_at = datetime.utcnow().isoformat() + "Z"
    now = time.time()
    cache = _team_logos_cache

    # Check cache
    if cache["data"] and (now - cache["timestamp"]) < _TEAM_LOGOS_TTL:
        return {
            "generated_at": cache["data"]["generated_at"],
            "cached": True,
            "cache_age_seconds": int(now - cache["timestamp"]),
            "teams": cache["data"]["teams"],
            "count": cache["data"]["count"],
        }

    try:
        # Fetch all teams with logos
        result = await session.execute(
            text("SELECT name, logo_url FROM teams WHERE logo_url IS NOT NULL ORDER BY name")
        )
        rows = result.fetchall()

        # Build name -> logo_url map
        teams_map = {}
        for row in rows:
            name, logo_url = row
            if name and logo_url:
                teams_map[name] = logo_url

        # Update cache
        cache["data"] = {
            "generated_at": generated_at,
            "teams": teams_map,
            "count": len(teams_map),
        }
        cache["timestamp"] = now

        return {
            "generated_at": generated_at,
            "cached": False,
            "cache_age_seconds": 0,
            "teams": teams_map,
            "count": len(teams_map),
        }

    except Exception as e:
        logger.error(f"[TEAM_LOGOS] Error fetching team logos: {e}", exc_info=True)
        return {
            "generated_at": generated_at,
            "cached": False,
            "cache_age_seconds": 0,
            "teams": {},
            "count": 0,
            "status": "degraded",
            "error": str(e)[:100],
        }
