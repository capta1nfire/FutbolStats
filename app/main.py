"""FastAPI application for FutbolStat MVP."""

import json
import logging
import os
import time
from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel, Field, model_validator
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from sqlalchemy import bindparam, func, select, text, column
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.database import close_db, get_async_session, init_db, AsyncSessionLocal, get_pool_status
from app.etl import APIFootballProvider, ETLPipeline
from app.etl.competitions import ALL_LEAGUE_IDS, COMPETITIONS
from app.etl.sota_constants import SOFASCORE_SUPPORTED_LEAGUES, UNDERSTAT_SUPPORTED_LEAGUES
from app.features import FeatureEngineer
from app.ml.persistence import load_active_model, persist_model_snapshot
from app.models import JobRun, Match, OddsHistory, OpsAlert, PITReport, PostMatchAudit, Prediction, PredictionOutcome, SensorPrediction, ShadowPrediction, Team, TeamAdjustment, TeamOverride
from app.teams.overrides import preload_team_overrides, resolve_team_display
from app.scheduler import start_scheduler, stop_scheduler, get_last_sync_time, get_sync_leagues, SYNC_LEAGUES, global_sync_window
from app.security import (
    limiter, verify_api_key, verify_api_key_or_ops_session,
    verify_dashboard_token_bool as _verify_dashboard_token,
    verify_debug_token as _verify_debug_token,
    _has_valid_ops_session as _has_valid_session,
    _get_dashboard_token_from_request,
)
from app.state import ml_engine, _telemetry, _incr
from app.telemetry.sentry import init_sentry, sentry_job_context, is_sentry_enabled
from app.logos.routes import router as logos_router
from app.dashboard.model_benchmark import router as model_benchmark_router
from app.dashboard.football_routes import router as football_routes_router
from app.dashboard.admin_routes import router as admin_routes_router
from app.routes.core import router as core_router
from app.utils.standings import (
    select_standings_view, StandingsGroupNotFound, apply_zones,
    group_standings_by_name, select_default_standings_group,
    classify_group_type,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Initialize Sentry for error tracking (before FastAPI app creation)
# Only activates if SENTRY_DSN is set in environment
init_sentry()

settings = get_settings()

# =============================================================================
# OPS LOG BUFFER (in-memory, filtered)
# =============================================================================

OPS_LOG_BUFFER_MAX = int(os.environ.get("OPS_LOG_BUFFER_MAX", "500"))
OPS_LOG_DEFAULT_LIMIT = int(os.environ.get("OPS_LOG_DEFAULT_LIMIT", "200"))
OPS_LOG_DEFAULT_SINCE_MINUTES = int(os.environ.get("OPS_LOG_DEFAULT_SINCE_MINUTES", "1440"))  # 24h

_ops_log_buffer: "deque[dict]" = deque(maxlen=OPS_LOG_BUFFER_MAX)
_ops_log_handler_installed = False
_seen_scheduler_started = False  # Deduplicate "Scheduler started" (show only once per deploy)


def _is_relevant_ops_log(record: logging.LogRecord, message: str) -> bool:
    """
    Decide if a log line is relevant for the operator dashboard.
    Goal: include capture/sync/budget/errors, exclude noisy scheduler/httpx spam.
    """
    name = record.name or ""
    msg = message or ""

    # Always include warnings/errors
    if record.levelno >= logging.WARNING:
        return True

    # Exclude very noisy loggers
    if name.startswith("apscheduler"):
        return False
    if name.startswith("httpx"):
        return False
    if name.startswith("uvicorn.access"):
        return False

    # Exclude common spam patterns
    spam_substrings = [
        "executed successfully",
        "Job ",
        "HTTP Request: GET",
        "HTTP Request: POST",
        # Global sync spam (keep only "complete" and "Filtered to")
        "Global sync: Fetching all fixtures",
        "Global sync: Received",
    ]
    if any(s in msg for s in spam_substrings):
        return False

    # Include key operational events (high signal)
    include_substrings = [
        "Global sync complete",
        "Lineup confirmed for match",
        "Captured lineup_confirmed odds",
        "PIT_SNAPSHOT_CREATED",
        "Market movement",
        "lineup_movement",
        "finished_match_stats",
        "Stats backfill",
        "BUDGET",
        "APIBudgetExceeded",
        "budget exceeded",
        "Rate limited",
        "ERROR",
        "Error ",
        "WARNING",
    ]
    if any(s in msg for s in include_substrings):
        return True

    # Include some internal modules even at INFO
    if name.startswith("app.scheduler") or name.startswith("app.etl.api_football"):
        # Keep only if it doesn't look like low-signal spam
        low_signal = ["Adding job tentatively", "Database tables created successfully"]
        if any(s in msg for s in low_signal):
            return False
        return True

    return False


class OpsLogBufferHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        global _seen_scheduler_started

        try:
            message = record.getMessage()
        except Exception:
            message = "<unprintable>"

        if not _is_relevant_ops_log(record, message):
            return

        # Deduplicate "Scheduler started" - show only once per deploy
        if "Scheduler started:" in message:
            if _seen_scheduler_started:
                return  # Skip duplicate
            _seen_scheduler_started = True

        ts = getattr(record, "created", None)
        dt = datetime.utcfromtimestamp(ts) if isinstance(ts, (int, float)) else datetime.utcnow()

        _ops_log_buffer.append(
            {
                "ts_utc": dt.isoformat() + "Z",
                "level": record.levelname,
                "logger": record.name,
                "message": message,
            }
        )


def _install_ops_log_handler() -> None:
    global _ops_log_handler_installed
    if _ops_log_handler_installed:
        return

    root = logging.getLogger()
    handler = OpsLogBufferHandler(level=logging.INFO)
    root.addHandler(handler)
    _ops_log_handler_installed = True


def _get_ops_logs(
    since_minutes: int = OPS_LOG_DEFAULT_SINCE_MINUTES,
    limit: int = OPS_LOG_DEFAULT_LIMIT,
    level: Optional[str] = None,
    compact: bool = False,
) -> list[dict]:
    # Copy without holding locks (deque ops are atomic-ish for append; acceptable for dashboard use)
    rows = list(_ops_log_buffer)

    # Filter by since (UTC)
    cutoff = datetime.utcnow() - timedelta(minutes=since_minutes)
    filtered = []
    for r in rows:
        ts_str = r.get("ts_utc")
        try:
            dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00")).replace(tzinfo=None) if ts_str else None
        except Exception:
            dt = None
        if dt and dt < cutoff:
            continue
        filtered.append(r)

    # Filter by minimum level if provided
    if level:
        level = level.upper()
        order = {"DEBUG": 10, "INFO": 20, "WARNING": 30, "ERROR": 40, "CRITICAL": 50}
        min_lvl = order.get(level)
        if min_lvl is not None:
            filtered = [r for r in filtered if order.get(r.get("level", "INFO"), 20) >= min_lvl]

    # Compact mode: group by message, show count + first/last timestamp
    if compact:
        from collections import OrderedDict
        groups: OrderedDict[str, dict] = OrderedDict()
        for r in filtered:
            msg = r.get("message", "")
            # Normalize message for grouping (remove variable parts like match counts)
            # Keep first 80 chars for grouping key
            key = msg[:80] if len(msg) > 80 else msg
            if key not in groups:
                groups[key] = {
                    "message": msg,
                    "level": r.get("level"),
                    "logger": r.get("logger"),
                    "count": 1,
                    "first_ts": r.get("ts_utc"),
                    "last_ts": r.get("ts_utc"),
                }
            else:
                groups[key]["count"] += 1
                groups[key]["last_ts"] = r.get("ts_utc")
        # Sort by last_ts descending (newest groups first)
        result = list(groups.values())
        result.sort(key=lambda x: x.get("last_ts", ""), reverse=True)
        return result[: max(1, int(limit))]

    # Newest first
    filtered.reverse()
    return filtered[: max(1, int(limit))]


# ml_engine, _telemetry, _incr imported from app.state (singleton-by-import)


# Simple in-memory cache for predictions
# TTL reduced to 60s for better live match updates (elapsed, goals)
_predictions_cache = {
    "data": None,
    "timestamp": 0,
    "ttl": 60,  # 1 minute cache (was 5 min, reduced for live matches)
}

# Standings cache: keyed by (league_id, season), stores standings list
# TTL 30 minutes - standings don't change frequently during a match detail view
# Standings: DB-first with L1 memory cache
# Architecture: memory cache (30min TTL) -> DB (6h TTL) -> provider fallback
_standings_cache = {}  # type: dict  # L1 cache: {(league_id, season): {"data": list, "timestamp": float}}
_STANDINGS_CACHE_TTL = 1800  # 30 minutes (L1 memory)
_STANDINGS_DB_TTL = 21600  # 6 hours (DB refresh threshold)


# Many LATAM leagues run on a calendar-year season (Jan-Dec). Our default season logic
# (Jul-Jun) is correct for most European leagues but wrong for these.
_CALENDAR_YEAR_SEASON_LEAGUES = {
    71,   # Brazil - Serie A
    128,  # Argentina - Primera División
    239,  # Colombia Primera A
    242,  # Ecuador Liga Pro
    253,  # USA - MLS (calendar)
    265,  # Chile Primera Division
    268,  # Uruguay Primera - Apertura
    270,  # Uruguay Primera - Clausura
    281,  # Peru Primera Division
    299,  # Venezuela Primera Division
    344,  # Bolivia Primera Division
}

# Leagues where we should NOT filter teams by "Relegation" in previous standings,
# because they use different systems (averages) or have no relegation.
# For these leagues, we use the full previous roster in placeholder generation.
_NO_RELEGATION_FILTER_LEAGUES = {
    239,  # Colombia - relegation by multi-season averages (not single table)
    262,  # Mexico - no relegation (varies by season, assume none)
}


def _season_for_league(league_id: Optional[int], dt: datetime) -> int:
    """
    Determine API-Football 'season' parameter for a league at a given date.

    - Default: European-style season year (Jul-Jun): Jan 2026 -> 2025.
    - Calendar-year leagues (LATAM/MLS): Jan 2026 -> 2026.
    """
    if league_id is not None and league_id in _CALENDAR_YEAR_SEASON_LEAGUES:
        return dt.year
    return dt.year if dt.month >= 7 else dt.year - 1


def _get_cached_standings(league_id: int, season: int) -> Optional[list]:
    """Get standings from L1 memory cache if still valid.

    Returns shallow copies of each entry to prevent mutations
    (e.g., external→internal ID translation) from corrupting the cache.
    """
    key = (league_id, season)
    if key in _standings_cache:
        entry = _standings_cache[key]
        if time.time() - entry["timestamp"] < _STANDINGS_CACHE_TTL:
            return [dict(e) for e in entry["data"]]
    return None


def _set_cached_standings(league_id: int, season: int, data: list) -> None:
    """Store standings in L1 memory cache."""
    key = (league_id, season)
    _standings_cache[key] = {"data": data, "timestamp": time.time()}


async def _get_standings_from_db(session, league_id: int, season: int) -> Optional[list]:
    """Get standings from DB (L2). Returns None if not found or expired."""
    from datetime import timedelta
    result = await session.execute(
        text("""
            SELECT standings, captured_at
            FROM league_standings
            WHERE league_id = :league_id AND season = :season
        """),
        {"league_id": league_id, "season": season}
    )
    row = result.fetchone()
    if row:
        standings, captured_at = row
        # Check if data is stale (older than 6h)
        if captured_at and (datetime.now() - captured_at).total_seconds() < _STANDINGS_DB_TTL:
            return standings
    return None


async def _save_standings_to_db(session, league_id: int, season: int, standings: list) -> None:
    """Persist standings to DB with upsert."""
    from datetime import timedelta
    expires_at = datetime.now() + timedelta(seconds=_STANDINGS_DB_TTL)
    await session.execute(
        text("""
            INSERT INTO league_standings (league_id, season, standings, captured_at, expires_at, source)
            VALUES (:league_id, :season, :standings, NOW(), :expires_at, 'warmup')
            ON CONFLICT (league_id, season)
            DO UPDATE SET standings = :standings, captured_at = NOW(), expires_at = :expires_at, source = 'warmup'
        """),
        {"league_id": league_id, "season": season, "standings": json.dumps(standings), "expires_at": expires_at}
    )
    await session.commit()


# Standings calculated cache: shorter TTL (15 min) for calculated standings
_STANDINGS_CALCULATED_TTL = 900  # 15 minutes


async def _calculate_standings_from_results(session, league_id: int, season: int) -> list:
    """
    Calculate standings from FT match results when API-Football has no data yet.

    Guardrails (per Auditor approval):
    - Only activates if FT_count >= 2 in this league/season
    - Only for league competitions (not cups/knockouts)
    - Returns source='calculated', is_calculated=True for transparency
    - Uses shorter cache TTL (15 min)
    - Priority: API standings > calculated > placeholder

    Sorting: points DESC, goal_diff DESC, goals_for DESC, team_name ASC

    Args:
        session: Database session
        league_id: League ID
        season: Season year

    Returns:
        List of standings dicts with calculated stats, or empty list if not eligible.
    """
    # Check FT count threshold (guardrail: need at least 2 finished matches)
    ft_count_result = await session.execute(
        text("""
            SELECT COUNT(*)
            FROM matches
            WHERE league_id = :league_id
              AND EXTRACT(YEAR FROM date) = :season
              AND status IN ('FT', 'AET', 'PEN', 'AWD')
              AND home_goals IS NOT NULL
              AND away_goals IS NOT NULL
        """),
        {"league_id": league_id, "season": season}
    )
    ft_count = ft_count_result.scalar() or 0

    if ft_count < 2:
        logger.debug(f"Calculated standings skipped: league {league_id} season {season} has only {ft_count} FT matches (need >= 2)")
        return []

    # Get all teams from season fixtures (includes teams with 0 matches played)
    teams_result = await session.execute(
        text("""
            SELECT DISTINCT t.id, t.external_id, t.name, t.logo_url
            FROM teams t
            JOIN matches m ON (t.id = m.home_team_id OR t.id = m.away_team_id)
            WHERE m.league_id = :league_id
              AND EXTRACT(YEAR FROM m.date) = :season
              AND t.team_type = 'club'
        """),
        {"league_id": league_id, "season": season}
    )
    teams = {row[0]: {"id": row[0], "external_id": row[1], "name": row[2], "logo_url": row[3]} for row in teams_result.fetchall()}

    if not teams:
        return []

    # Initialize stats for all teams
    stats = {}
    for team_id, team_data in teams.items():
        stats[team_id] = {
            "team_id": team_data["external_id"],  # Use external_id for consistency with API-Football
            "team_name": team_data["name"],
            "team_logo": team_data["logo_url"],
            "points": 0,
            "played": 0,
            "won": 0,
            "drawn": 0,
            "lost": 0,
            "goals_for": 0,
            "goals_against": 0,
            "goal_diff": 0,
            "form": "",  # Last 5 results
            "form_results": [],  # For building form string
        }

    # Calculate stats from FT matches
    matches_result = await session.execute(
        text("""
            SELECT home_team_id, away_team_id, home_goals, away_goals, date
            FROM matches
            WHERE league_id = :league_id
              AND EXTRACT(YEAR FROM date) = :season
              AND status IN ('FT', 'AET', 'PEN', 'AWD')
              AND home_goals IS NOT NULL
              AND away_goals IS NOT NULL
            ORDER BY date ASC
        """),
        {"league_id": league_id, "season": season}
    )

    for home_id, away_id, home_goals, away_goals, match_date in matches_result.fetchall():
        if home_id not in stats or away_id not in stats:
            continue

        # Home team stats
        stats[home_id]["played"] += 1
        stats[home_id]["goals_for"] += home_goals
        stats[home_id]["goals_against"] += away_goals

        # Away team stats
        stats[away_id]["played"] += 1
        stats[away_id]["goals_for"] += away_goals
        stats[away_id]["goals_against"] += home_goals

        if home_goals > away_goals:
            # Home win
            stats[home_id]["won"] += 1
            stats[home_id]["points"] += 3
            stats[home_id]["form_results"].append("W")
            stats[away_id]["lost"] += 1
            stats[away_id]["form_results"].append("L")
        elif home_goals < away_goals:
            # Away win
            stats[away_id]["won"] += 1
            stats[away_id]["points"] += 3
            stats[away_id]["form_results"].append("W")
            stats[home_id]["lost"] += 1
            stats[home_id]["form_results"].append("L")
        else:
            # Draw
            stats[home_id]["drawn"] += 1
            stats[home_id]["points"] += 1
            stats[home_id]["form_results"].append("D")
            stats[away_id]["drawn"] += 1
            stats[away_id]["points"] += 1
            stats[away_id]["form_results"].append("D")

    # Calculate goal diff and form string (last 5)
    for team_id in stats:
        stats[team_id]["goal_diff"] = stats[team_id]["goals_for"] - stats[team_id]["goals_against"]
        stats[team_id]["form"] = "".join(stats[team_id]["form_results"][-5:])
        del stats[team_id]["form_results"]  # Remove helper field

    # Sort: points DESC, goal_diff DESC, goals_for DESC, team_name ASC
    sorted_teams = sorted(
        stats.values(),
        key=lambda x: (-x["points"], -x["goal_diff"], -x["goals_for"], x["team_name"])
    )

    # Build final standings with positions
    standings = []
    for idx, team_stats in enumerate(sorted_teams, start=1):
        standings.append({
            "position": idx,
            "team_id": team_stats["team_id"],
            "team_name": team_stats["team_name"],
            "team_logo": team_stats["team_logo"],
            "points": team_stats["points"],
            "played": team_stats["played"],
            "won": team_stats["won"],
            "drawn": team_stats["drawn"],
            "lost": team_stats["lost"],
            "goals_for": team_stats["goals_for"],
            "goals_against": team_stats["goals_against"],
            "goal_diff": team_stats["goal_diff"],
            "form": team_stats["form"],
            "group": None,
            "is_calculated": True,  # Transparency flag
            "source": "calculated",
        })

    logger.info(f"Calculated standings for league {league_id} season {season}: {len(standings)} teams from {ft_count} FT matches")
    return standings


async def _calculate_reclasificacion(session, league_id: int, season: int) -> dict | None:
    """
    Calculate reclasificación table (accumulated Apertura + Clausura).

    Phase 3 of League Format Configuration system.
    Only called when rules_json.reclasificacion.enabled = true.

    ABE P0 Guardrails:
    - Only regular phase matches (exclude Quadrangulares/Play Offs/Final)
    - Return None if either Apertura or Clausura has 0 matches (missing_phase)
    - Fail-closed on team_id duplicates
    - Single query + in-memory aggregation (no N+1)

    Args:
        session: Database session
        league_id: League ID
        season: Season year

    Returns:
        Dict with data + metadata, or None if not available.
    """
    # Single query: fetch all regular-phase FT matches with team info
    result = await session.execute(
        text("""
            SELECT m.home_team_id, m.away_team_id, m.home_goals, m.away_goals,
                   ht.name, ht.logo_url,
                   awt.name, awt.logo_url,
                   m.round
            FROM matches m
            JOIN teams ht ON ht.id = m.home_team_id
            JOIN teams awt ON awt.id = m.away_team_id
            WHERE m.league_id = :league_id
              AND m.season = :season
              AND m.status IN ('FT', 'AET', 'PEN', 'AWD')
              AND m.home_goals IS NOT NULL
              AND m.away_goals IS NOT NULL
              AND (m.round ILIKE 'Apertura - %' OR m.round ILIKE 'Clausura - %')
              AND m.round NOT ILIKE '%Quadrangular%'
              AND m.round NOT ILIKE '%Play Offs%'
              AND m.round NOT ILIKE '%Final%'
        """),
        {"league_id": league_id, "season": season},
    )
    rows = result.fetchall()

    # Early return: not enough matches (< 20 = less than 1 full matchday)
    if len(rows) < 20:
        logger.info(
            f"[RECLASIFICACION] Skipped league {league_id}: only {len(rows)} "
            f"regular-phase matches (need >= 20)"
        )
        return None

    # Count matches per phase (round is column index 8)
    apertura_count = sum(1 for r in rows if r[8].startswith("Apertura"))
    clausura_count = sum(1 for r in rows if r[8].startswith("Clausura"))

    # ABE P0: Missing phase → null + log
    if apertura_count == 0:
        logger.warning(
            f"[RECLASIFICACION] missing_phase: Apertura for league {league_id} season {season}"
        )
        return None
    if clausura_count == 0:
        logger.warning(
            f"[RECLASIFICACION] missing_phase: Clausura for league {league_id} season {season}"
        )
        return None

    # Aggregate stats in-memory (keyed by internal team_id)
    stats: dict[int, dict] = {}

    for row in rows:
        home_id, away_id, hg, ag = row[0], row[1], row[2], row[3]
        # row[4..5] = home team name, logo
        # row[6..7] = away team name, logo

        # Initialize teams if not seen (ABE P0-1: team_id = internal id, not external)
        for tid, name, logo in [
            (home_id, row[4], row[5]),
            (away_id, row[6], row[7]),
        ]:
            if tid not in stats:
                stats[tid] = {
                    "team_id": tid,
                    "team_name": name,
                    "team_logo": logo,
                    "points": 0, "played": 0, "won": 0, "drawn": 0, "lost": 0,
                    "goals_for": 0, "goals_against": 0, "goal_diff": 0,
                }

        # Home team stats
        stats[home_id]["played"] += 1
        stats[home_id]["goals_for"] += hg
        stats[home_id]["goals_against"] += ag

        # Away team stats
        stats[away_id]["played"] += 1
        stats[away_id]["goals_for"] += ag
        stats[away_id]["goals_against"] += hg

        if hg > ag:
            stats[home_id]["won"] += 1
            stats[home_id]["points"] += 3
            stats[away_id]["lost"] += 1
        elif hg < ag:
            stats[away_id]["won"] += 1
            stats[away_id]["points"] += 3
            stats[home_id]["lost"] += 1
        else:
            stats[home_id]["drawn"] += 1
            stats[home_id]["points"] += 1
            stats[away_id]["drawn"] += 1
            stats[away_id]["points"] += 1

    # Calculate goal_diff
    for s in stats.values():
        s["goal_diff"] = s["goals_for"] - s["goals_against"]

    # ABE P0: Validate no duplicate team_id (fail-closed)
    team_ids = [s["team_id"] for s in stats.values()]
    if len(team_ids) != len(set(team_ids)):
        logger.error(
            f"[RECLASIFICACION] duplicate team_id detected for league {league_id} "
            f"season {season}. Aborting reclasificacion."
        )
        return None

    # Sort: points DESC, goal_diff DESC, goals_for DESC, team_name ASC
    sorted_teams = sorted(
        stats.values(),
        key=lambda x: (-x["points"], -x["goal_diff"], -x["goals_for"], x["team_name"]),
    )

    # Add position
    data = []
    for idx, team in enumerate(sorted_teams, start=1):
        data.append({"position": idx, **team})

    logger.info(
        f"[RECLASIFICACION] Calculated for league {league_id} season {season}: "
        f"{len(data)} teams, apertura={apertura_count} clausura={clausura_count} "
        f"total={len(rows)}"
    )

    return {
        "data": data,
        "apertura_matches": apertura_count,
        "clausura_matches": clausura_count,
        "total_matches": len(rows),
    }


async def _get_season_team_stats_from_standings(
    session, league_id: int, season: int
) -> dict[int, dict] | None:
    """
    Get per-team points/played for a season from stored standings.

    Looks for Apertura + Clausura groups and sums them.
    Returns {internal_team_id: {points, played, goals_for, goals_against, ...}} or None.

    ABE P0: Translates external_id (API-Football) → internal id (teams.id)
    before returning to avoid ID collisions across leagues.
    """
    result = await session.execute(
        text("SELECT standings FROM league_standings WHERE league_id = :lid AND season = :s"),
        {"lid": league_id, "s": season},
    )
    row = result.fetchone()
    if not row or not row.standings:
        return None

    standings = row.standings if isinstance(row.standings, list) else []
    if not standings:
        return None

    groups = group_standings_by_name(standings)

    # Find Apertura and Clausura groups (case-insensitive)
    apertura_entries = None
    clausura_entries = None
    for name, entries in groups.items():
        name_lower = name.lower()
        if "apertura" in name_lower and "group" not in name_lower:
            apertura_entries = entries
        elif "clausura" in name_lower and "group" not in name_lower:
            clausura_entries = entries

    if not apertura_entries and not clausura_entries:
        # No Apertura/Clausura found; try selecting main group as fallback
        selected_group, _ = select_default_standings_group(groups, {})
        if selected_group and selected_group in groups:
            entries = groups[selected_group]
            team_stats = {}
            for e in entries:
                tid = e.get("team_id")
                if tid is not None:
                    team_stats[tid] = {
                        "points": int(e.get("points") or 0),
                        "played": int(e.get("played") or 0),
                        "goals_for": int(e.get("goals_for") or 0),
                        "goals_against": int(e.get("goals_against") or 0),
                    }
            if team_stats:
                return await _translate_ext_to_int_ids(session, team_stats)
            return None
        return None

    # Sum Apertura + Clausura per team
    team_stats: dict[int, dict] = {}
    for entries in [apertura_entries, clausura_entries]:
        if not entries:
            continue
        for e in entries:
            tid = e.get("team_id")
            if tid is None:
                continue
            if tid not in team_stats:
                team_stats[tid] = {
                    "points": 0, "played": 0,
                    "goals_for": 0, "goals_against": 0,
                }
            team_stats[tid]["points"] += int(e.get("points") or 0)
            team_stats[tid]["played"] += int(e.get("played") or 0)
            team_stats[tid]["goals_for"] += int(e.get("goals_for") or 0)
            team_stats[tid]["goals_against"] += int(e.get("goals_against") or 0)

    if not team_stats:
        return None
    return await _translate_ext_to_int_ids(session, team_stats)


async def _translate_ext_to_int_ids(
    session, team_stats: dict[int, dict]
) -> dict[int, dict] | None:
    """
    Translate team_stats keyed by external_id to internal_id.

    ABE P0: Prevents ID collisions where a Colombian team's external_id
    matches a European team's internal_id (e.g., Jaguares ext=1133 vs Espanyol id=1133).
    """
    ext_ids = list(team_stats.keys())
    if not ext_ids:
        return None

    id_result = await session.execute(
        text("SELECT id, external_id FROM teams WHERE external_id IN :eids").bindparams(
            bindparam("eids", expanding=True)
        ),
        {"eids": ext_ids},
    )
    ext_to_int = {r.external_id: r.id for r in id_result.fetchall()}

    translated: dict[int, dict] = {}
    for ext_id, stats in team_stats.items():
        internal_id = ext_to_int.get(ext_id)
        if internal_id is not None:
            translated[internal_id] = stats
        else:
            # Team not found in our DB — keep external_id as fallback
            logger.warning(f"[DESCENSO] No internal_id for external_id {ext_id}")
            translated[ext_id] = stats

    return translated if translated else None


async def _get_season_team_stats_from_matches(
    session, league_id: int, season: int
) -> dict[int, dict] | None:
    """
    Get per-team points/played for a season from matches.

    Only used when rounds are properly labeled (not NULL).
    Filters to regular phase only (excludes playoffs/quadrangulares/finals).
    Returns {internal_team_id: {points, played, ...}} or None.
    """
    # First check if this season has labeled rounds
    round_check = await session.execute(
        text("""
            SELECT COUNT(*) FILTER (WHERE round IS NOT NULL) as labeled,
                   COUNT(*) as total
            FROM matches
            WHERE league_id = :lid AND season = :s
              AND status IN ('FT', 'AET', 'PEN', 'AWD')
        """),
        {"lid": league_id, "s": season},
    )
    rc = round_check.fetchone()
    if not rc or rc.total == 0:
        return None
    # ABE P0: Fail-closed if rounds are mostly NULL (can't filter playoffs)
    if rc.labeled < rc.total * 0.5:
        logger.warning(
            f"[DESCENSO] Season {season} league {league_id}: {rc.labeled}/{rc.total} "
            f"matches have labeled rounds. Fail-closed."
        )
        return None

    # Fetch regular-phase matches only
    result = await session.execute(
        text("""
            SELECT m.home_team_id, m.away_team_id, m.home_goals, m.away_goals
            FROM matches m
            WHERE m.league_id = :lid
              AND m.season = :s
              AND m.status IN ('FT', 'AET', 'PEN', 'AWD')
              AND m.home_goals IS NOT NULL
              AND m.away_goals IS NOT NULL
              AND m.round IS NOT NULL
              AND m.round NOT ILIKE '%Quadrangular%'
              AND m.round NOT ILIKE '%Play Offs%'
              AND m.round NOT ILIKE '%Final%'
              AND m.round NOT ILIKE '%Quarter%'
              AND m.round NOT ILIKE '%Semi%'
              AND m.round NOT ILIKE '%8th Finals%'
              AND m.round NOT ILIKE '%Round of 16%'
        """),
        {"lid": league_id, "s": season},
    )
    rows = result.fetchall()
    if len(rows) < 20:
        return None

    stats: dict[int, dict] = {}
    for home_id, away_id, hg, ag in rows:
        for tid in [home_id, away_id]:
            if tid not in stats:
                stats[tid] = {"points": 0, "played": 0, "goals_for": 0, "goals_against": 0}

        stats[home_id]["played"] += 1
        stats[home_id]["goals_for"] += hg
        stats[home_id]["goals_against"] += ag
        stats[away_id]["played"] += 1
        stats[away_id]["goals_for"] += ag
        stats[away_id]["goals_against"] += hg

        if hg > ag:
            stats[home_id]["points"] += 3
        elif hg < ag:
            stats[away_id]["points"] += 3
        else:
            stats[home_id]["points"] += 1
            stats[away_id]["points"] += 1

    return stats if stats else None


async def _calculate_descenso(
    session,
    league_id: int,
    season: int,
    relegation_config: dict,
    all_standings: list[dict],
) -> dict | None:
    """
    Calculate relegation risk table (tabla de promedios).

    Phase 4 of League Format Configuration system.
    Two paths:
    - Path A: API-Football provides "Promedios" group → use directly
    - Path B: Calculate from standings + matches hybrid

    ABE P0 Guardrails:
    - No matches with NULL rounds (fail-closed)
    - Zone = "relegation_risk" (not "relegation") — informational only
    - team_id = internal id
    """
    relegation_count = int(relegation_config.get("count", 2))
    years = int(relegation_config.get("years", 3))

    # --- Path A: API-Football "Promedios" group ---
    promedios = [s for s in all_standings if "promedios" in (s.get("group") or "").lower()]
    if promedios:
        # Build data from API promedios group
        data = []
        for entry in sorted(promedios, key=lambda x: int(x.get("position") or 999)):
            pos = int(entry.get("position") or 0)
            points = int(entry.get("points") or 0)
            played = int(entry.get("played") or 0)
            avg = round(points / played, 4) if played > 0 else 0.0
            total = len(promedios)
            zone = None
            if pos > total - relegation_count:
                zone = {"type": "relegation_risk", "style": "red"}
            data.append({
                "position": pos,
                "team_id": entry.get("team_id"),  # external_id — translated later
                "team_name": entry.get("team_name"),
                "team_logo": entry.get("team_logo"),
                "points": points,
                "played": played,
                "average": avg,
                "goals_for": int(entry.get("goals_for") or 0),
                "goals_against": int(entry.get("goals_against") or 0),
                "goal_diff": int(entry.get("goal_diff") or 0),
                "zone": zone,
            })

        # ABE P0-2 (Phase 5 audit): all_standings already has internal IDs
        # (translated at endpoint level, line ~4394). No re-translation needed.
        # Previous code here caused collisions (same bug as Phase 4 Espanyol fix).

        logger.info(
            f"[DESCENSO] Path A (API) for league {league_id}: {len(data)} teams"
        )
        return {
            "data": data,
            "method": "average_3y",
            "source": "api",
            "relegation_count": relegation_count,
        }

    # --- Path B: Calculate from standings + matches hybrid ---
    seasons = list(range(season - years + 1, season + 1))
    seasons_used = []
    season_data: dict[int, dict[int, dict]] = {}  # {season: {team_id: stats}}

    # Step 1: Collect per-season stats
    # Try both standings and matches, use whichever is more complete.
    # Handles split-season leagues (e.g. Colombia) where standings may only
    # have one half (Clausura) while matches have both (Apertura + Clausura).
    for s in seasons:
        standings_stats = await _get_season_team_stats_from_standings(session, league_id, s)
        matches_stats = await _get_season_team_stats_from_matches(session, league_id, s)

        # Pick the more complete source (higher avg PJ = more complete)
        if standings_stats and matches_stats:
            avg_st_pj = sum(v["played"] for v in standings_stats.values()) / len(standings_stats)
            avg_mt_pj = sum(v["played"] for v in matches_stats.values()) / len(matches_stats)
            if avg_mt_pj > avg_st_pj * 1.3:
                season_stats, source_type = matches_stats, "matches"
            else:
                season_stats, source_type = standings_stats, "standings"
        elif standings_stats:
            season_stats, source_type = standings_stats, "standings"
        elif matches_stats:
            season_stats, source_type = matches_stats, "matches"
        else:
            logger.warning(
                f"[DESCENSO] No data for league {league_id} season {s}. Skipping."
            )
            continue

        seasons_used.append(s)
        season_data[s] = season_stats
        logger.info(
            f"[DESCENSO] Season {s}: {len(season_stats)} teams from {source_type}"
        )

    if len(seasons_used) < 2:
        logger.warning(
            f"[DESCENSO] Only {len(seasons_used)} seasons available for league "
            f"{league_id}. Need >= 2. Returning null."
        )
        return None

    # Step 2: Current primera = teams in current season's data
    current_primera_ids = set(season_data.get(season, {}).keys())
    if not current_primera_ids:
        logger.warning(f"[DESCENSO] No current season ({season}) data. Cannot filter.")
        return None

    # Step 3: Continuous stint per team — only count consecutive seasons
    # going backwards from current. A gap (team not in primera) resets the clock.
    team_stint_start: dict[int, int] = {}
    for tid in current_primera_ids:
        stint_start = season
        for s in sorted(seasons_used, reverse=True):
            if s == season:
                continue
            if tid in season_data.get(s, {}):
                stint_start = s
            else:
                break  # Gap breaks continuity
        team_stint_start[tid] = stint_start

    # Step 4: Accumulate only within each team's stint
    accumulated: dict[int, dict] = {}
    for tid in current_primera_ids:
        stint_start = team_stint_start[tid]
        accumulated[tid] = {"points": 0, "played": 0, "goals_for": 0, "goals_against": 0}
        for s in seasons_used:
            if s >= stint_start and tid in season_data.get(s, {}):
                stats = season_data[s][tid]
                accumulated[tid]["points"] += stats["points"]
                accumulated[tid]["played"] += stats["played"]
                accumulated[tid]["goals_for"] += stats.get("goals_for", 0)
                accumulated[tid]["goals_against"] += stats.get("goals_against", 0)

    logger.info(
        f"[DESCENSO] Stint analysis: {len(current_primera_ids)} current teams, "
        f"stints: {dict(sorted(((tid, team_stint_start[tid]) for tid in list(current_primera_ids)[:5]), key=lambda x: x[1]))}"
    )

    if len(accumulated) < 10:
        return None

    # Calculate average and goal_diff
    for stats in accumulated.values():
        stats["average"] = (
            round(stats["points"] / stats["played"], 4)
            if stats["played"] > 0 else 0.0
        )
        stats["goal_diff"] = stats["goals_for"] - stats["goals_against"]

    # Resolve team names/logos with display_name (COALESCE pattern from TEAM_ENRICHMENT_SYSTEM.md)
    # ABE P0: All IDs in accumulated are now internal (translated in standings helper)
    all_ids = list(accumulated.keys())
    team_info_result = await session.execute(
        text("""
            SELECT t.id, t.name, t.logo_url,
                   COALESCE(teo.short_name, twe.short_name, t.name) AS display_name
            FROM teams t
            LEFT JOIN team_enrichment_overrides teo ON t.id = teo.team_id
            LEFT JOIN team_wikidata_enrichment twe ON t.id = twe.team_id
            WHERE t.id IN :ids
        """).bindparams(
            bindparam("ids", expanding=True)
        ),
        {"ids": all_ids},
    )
    team_rows = team_info_result.fetchall()

    # Build lookup: internal id → team info
    id_lookup: dict[int, dict] = {}
    for r in team_rows:
        id_lookup[r[0]] = {"name": r[1], "logo": r[2], "display_name": r[3]}

    # Build normalized data (all IDs already internal, no merging needed)
    normalized: dict[int, dict] = {}
    for tid, stats in accumulated.items():
        info = id_lookup.get(tid)
        if not info:
            logger.warning(f"[DESCENSO] Team id {tid} not found in teams table. Skipping.")
            continue
        normalized[tid] = {
            **stats,
            "team_id": tid,
            "team_name": info["name"],
            "display_name": info["display_name"],
            "team_logo": info["logo"],
        }

    # ABE P0: Validate no duplicate team_id
    if len(normalized) < 10:
        return None

    # Sort: average DESC (best first), goal_diff DESC, goals_for DESC
    sorted_teams = sorted(
        normalized.values(),
        key=lambda x: (-x["average"], -x["goal_diff"], -x["goals_for"], x["team_name"]),
    )

    # Add position and zone marking
    total = len(sorted_teams)
    data = []
    for idx, team in enumerate(sorted_teams, start=1):
        zone = None
        if idx > total - relegation_count:
            zone = {"type": "relegation_risk", "style": "red"}
        data.append({
            "position": idx,
            "team_id": team["team_id"],
            "team_name": team["team_name"],
            "display_name": team["display_name"],
            "team_logo": team["team_logo"],
            "points": team["points"],
            "played": team["played"],
            "average": team["average"],
            "goals_for": team["goals_for"],
            "goals_against": team["goals_against"],
            "goal_diff": team["goal_diff"],
            "zone": zone,
        })

    logger.info(
        f"[DESCENSO] Path B (calculated) for league {league_id}: {len(data)} teams, "
        f"seasons={seasons_used}"
    )
    return {
        "data": data,
        "method": "average_3y",
        "source": "calculated",
        "relegation_count": relegation_count,
        "seasons": seasons_used,
    }


async def _generate_placeholder_standings(session, league_id: int, season: int) -> list:
    """
    Generate placeholder standings for a league when API data is not yet available.

    Strategy (in order of priority):
    1. Use teams from fixtures of the new season (most accurate - reflects actual roster)
    2. Use teams from previous season standings, filtering relegated teams
    3. Fall back to teams from recent matches (least accurate)

    Returns teams with zero stats, ordered alphabetically.

    Args:
        session: Database session
        league_id: League ID
        season: Season year

    Returns:
        List of standings dicts with all zeros, ordered alphabetically by team name.
    """
    teams_data = []

    # Strategy 1: Use teams from fixtures of the target season (most accurate)
    # This reflects the actual roster including promotions/relegations
    new_season_result = await session.execute(
        text("""
            SELECT DISTINCT t.id, t.name, t.logo_url
            FROM teams t
            JOIN matches m ON (t.id = m.home_team_id OR t.id = m.away_team_id)
            WHERE m.league_id = :league_id
              AND EXTRACT(YEAR FROM m.date) = :season
              AND t.team_type = 'club'
            ORDER BY t.name
        """),
        {"league_id": league_id, "season": season}
    )
    for row in new_season_result.fetchall():
        teams_data.append({
            "id": row[0],  # Use internal ID
            "name": row[1],
            "logo_url": row[2],
        })

    if teams_data:
        logger.info(f"Using {len(teams_data)} teams from {season} fixtures for placeholder")
    else:
        # Strategy 2: Use teams from previous season standings, filtering relegated teams
        prev_standings_result = await session.execute(
            text("""
                SELECT standings
                FROM league_standings
                WHERE league_id = :league_id
                  AND season < :season
                  AND json_array_length(standings) > 0
                ORDER BY season DESC
                LIMIT 1
            """),
            {"league_id": league_id, "season": season}
        )
        prev_row = prev_standings_result.fetchone()

        if prev_row and prev_row[0]:
            prev_standings = prev_row[0]

            # Filter to main group first to avoid duplicates from multi-group leagues
            # (e.g. Ecuador has Serie A + Championship Round + Qualifying Round + Relegation Round
            # with overlapping teams). Get rules_json for heuristic selection.
            rules_result = await session.execute(
                text("SELECT rules_json FROM admin_leagues WHERE league_id = :lid"),
                {"lid": league_id}
            )
            rules_row = rules_result.fetchone()
            prev_rules_json = (
                rules_row.rules_json if rules_row and isinstance(rules_row.rules_json, dict)
                else {}
            )
            prev_view = select_standings_view(prev_standings, prev_rules_json)
            filtered_prev = prev_view.standings

            relegated_teams = []
            seen_team_ids = set()
            for s in filtered_prev:
                desc = s.get("description") or ""
                # Only filter by "Relegation" if the league uses traditional table-based relegation.
                # Skip filtering for leagues with averages-based or no relegation system.
                if league_id not in _NO_RELEGATION_FILTER_LEAGUES:
                    if "relegation" in desc.lower():
                        relegated_teams.append(s.get("team_name"))
                        continue  # Skip relegated teams
                # Deduplicate by team_id as safety net
                tid = s.get("team_id")
                if tid and tid in seen_team_ids:
                    continue
                if tid:
                    seen_team_ids.add(tid)
                teams_data.append({
                    "id": tid,  # Note: may be external_id from old data, will be translated later
                    "name": s.get("team_name"),
                    "logo_url": s.get("team_logo"),
                })
            teams_data.sort(key=lambda x: x.get("name", ""))
            if relegated_teams:
                logger.info(f"Excluded {len(relegated_teams)} relegated teams: {relegated_teams}")
            logger.info(f"Using {len(teams_data)} teams from previous standings for placeholder")

    # Strategy 3: Fallback to teams from recent matches (less accurate)
    if not teams_data:
        result = await session.execute(
            text("""
                SELECT DISTINCT t.id, t.name, t.logo_url
                FROM teams t
                JOIN matches m ON (t.id = m.home_team_id OR t.id = m.away_team_id)
                WHERE m.league_id = :league_id
                  AND m.date > NOW() - INTERVAL '1 year'
                  AND t.team_type = 'club'
                ORDER BY t.name
            """),
            {"league_id": league_id}
        )
        for row in result.fetchall():
            teams_data.append({
                "id": row[0],  # Use internal ID
                "name": row[1],
                "logo_url": row[2],
            })
        logger.info(f"Using {len(teams_data)} teams from recent matches for placeholder")

    if not teams_data:
        return []

    # Build placeholder standings ordered alphabetically (position = row number)
    standings = []
    for idx, team in enumerate(teams_data, start=1):
        standings.append({
            "position": idx,
            "team_id": team["id"],  # Use internal ID
            "team_name": team["name"],
            "team_logo": team.get("logo_url"),
            "points": 0,
            "played": 0,
            "won": 0,
            "drawn": 0,
            "lost": 0,
            "goals_for": 0,
            "goals_against": 0,
            "goal_diff": 0,
            "form": "",
            "group": None,
            "is_placeholder": True,
        })

    logger.info(f"Generated placeholder standings for league {league_id} season {season}: {len(standings)} teams")
    return standings


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    import asyncio

    # Startup
    logger.info("Starting FutbolStat MVP...")
    _install_ops_log_handler()
    await init_db()

    # Ensure ops_daily_rollups table exists (idempotent)
    async with AsyncSessionLocal() as session:
        await session.execute(text("""
            CREATE TABLE IF NOT EXISTS ops_daily_rollups (
                day DATE PRIMARY KEY,
                payload JSONB NOT NULL,
                created_at TIMESTAMP DEFAULT NOW(),
                updated_at TIMESTAMP DEFAULT NOW()
            )
        """))
        # Ensure league_standings table exists (DB-first architecture)
        await session.execute(text("""
            CREATE TABLE IF NOT EXISTS league_standings (
                id SERIAL PRIMARY KEY,
                league_id INTEGER NOT NULL,
                season INTEGER NOT NULL,
                standings JSONB NOT NULL,
                captured_at TIMESTAMP DEFAULT NOW(),
                source VARCHAR(50) DEFAULT 'api_football',
                expires_at TIMESTAMP,
                UNIQUE(league_id, season)
            )
        """))
        await session.execute(text("""
            CREATE INDEX IF NOT EXISTS ix_league_standings_league_id ON league_standings(league_id)
        """))
        await session.execute(text("""
            CREATE INDEX IF NOT EXISTS ix_league_standings_season ON league_standings(season)
        """))
        await session.commit()

    # Try to load model from PostgreSQL first (fast path)
    model_loaded = False
    async with AsyncSessionLocal() as session:
        logger.info("Checking for model in PostgreSQL...")
        model_loaded = await load_active_model(session, ml_engine)

    if model_loaded:
        logger.info("ML model loaded from PostgreSQL (fast startup)")
    elif ml_engine.load_model():
        # Fallback: try loading from local filesystem (legacy)
        logger.info("ML model loaded from filesystem (legacy)")
    elif settings.SKIP_AUTO_TRAIN:
        logger.warning(
            "No ML model found, but SKIP_AUTO_TRAIN=true. "
            "Use POST /model/train to train manually."
        )
    else:
        # Start training in background to avoid startup timeout
        logger.info("No ML model found. Starting background training...")
        asyncio.create_task(_train_model_background())

    # Initialize shadow engine if configured (FASE 2: two-stage shadow mode)
    async with AsyncSessionLocal() as session:
        from app.ml.shadow import init_shadow_engine
        shadow_initialized = await init_shadow_engine(session)
        if shadow_initialized:
            logger.info("Shadow engine initialized (two-stage model for A/B comparison)")

    # Start background scheduler for weekly sync/train
    start_scheduler(ml_engine)

    # Warm up standings cache for active leagues (non-blocking)
    asyncio.create_task(_warmup_standings_cache())

    # Predictions catch-up on startup (P2 resilience)
    # If predictions job hasn't run recently and there are upcoming matches,
    # trigger a catch-up to avoid gaps from deploys interrupting the daily cron
    asyncio.create_task(_predictions_catchup_on_startup())

    yield

    # Shutdown
    logger.info("Shutting down...")
    stop_scheduler()
    await close_db()


async def _train_model_background():
    """Train the ML model in background after startup and save to PostgreSQL."""
    import asyncio
    from concurrent.futures import ThreadPoolExecutor

    # Small delay to let server fully start
    await asyncio.sleep(2)

    try:
        logger.info("Background training started...")
        async with AsyncSessionLocal() as session:
            feature_engineer = FeatureEngineer(session=session)
            df = await feature_engineer.build_training_dataset()

            if len(df) < 100:
                logger.error(f"Insufficient training data: {len(df)} samples. Need at least 100.")
                return

            # Train in executor to avoid blocking the event loop
            loop = asyncio.get_event_loop()
            with ThreadPoolExecutor() as executor:
                result = await loop.run_in_executor(executor, ml_engine.train, df)

            # Save model to PostgreSQL for fast startup on future deploys
            snapshot_id = await persist_model_snapshot(
                session=session,
                engine=ml_engine,
                brier_score=result["brier_score"],
                cv_scores=result["cv_scores"],
                samples_trained=result["samples_trained"],
            )

            logger.info(
                f"Background training complete: {ml_engine.model_version} with {len(df)} samples. "
                f"Saved to DB as snapshot {snapshot_id}"
            )
    except Exception as e:
        logger.error(f"Background training failed: {e}")


async def _warmup_standings_cache():
    """Pre-warm standings for leagues with upcoming matches.

    DB-first architecture: fetches from provider and persists to DB + L1 cache.
    Runs at startup to ensure most match_details requests hit cache/DB.
    This is fire-and-forget - failures don't affect app health.
    """
    import asyncio

    # Small delay to let server fully start
    await asyncio.sleep(2)
    _t_start = time.time()

    try:
        async with AsyncSessionLocal() as session:
            # Get unique league_ids from matches in the next 7 days
            from datetime import timedelta
            now = datetime.now()
            week_ahead = now + timedelta(days=7)

            result = await session.execute(
                text("""
                    SELECT DISTINCT league_id
                    FROM matches
                    WHERE league_id IS NOT NULL
                      AND date >= :start_date
                      AND date <= :end_date
                      AND status = 'NS'
                    LIMIT 20
                """),
                {"start_date": now.date(), "end_date": week_ahead.date()}
            )
            league_ids = [row[0] for row in result.fetchall()]

            if not league_ids:
                logger.info("[WARMUP] No upcoming leagues to warm up")
                return

            logger.info(f"[WARMUP] Warming up standings for {len(league_ids)} leagues: {league_ids}")

            provider = APIFootballProvider()
            warmed = 0
            skipped_cache = 0
            skipped_db = 0
            failed = 0
            consecutive_failures = 0
            max_consecutive_failures = 3  # Stop if API seems down

            for league_id in league_ids:
                season = _season_for_league(league_id, now)
                # Skip if already in L1 cache
                if _get_cached_standings(league_id, season) is not None:
                    skipped_cache += 1
                    continue

                # Skip if already in DB (fresh)
                db_standings = await _get_standings_from_db(session, league_id, season)
                if db_standings is not None:
                    # Populate L1 cache from DB
                    _set_cached_standings(league_id, season, db_standings)
                    skipped_db += 1
                    continue

                # Abort if too many consecutive failures (API budget/rate limit)
                if consecutive_failures >= max_consecutive_failures:
                    logger.warning(f"[WARMUP] Aborting after {consecutive_failures} consecutive failures")
                    break

                try:
                    standings = await provider.get_standings(league_id, season)
                    # Persist to DB (primary storage)
                    await _save_standings_to_db(session, league_id, season, standings)
                    # Populate L1 cache
                    _set_cached_standings(league_id, season, standings)
                    warmed += 1
                    consecutive_failures = 0  # Reset on success
                    # Rate limit: 0.5s between calls
                    await asyncio.sleep(0.5)
                except Exception as e:
                    failed += 1
                    consecutive_failures += 1
                    logger.warning(f"[WARMUP] Failed league {league_id}: {e}")
                    # Exponential backoff on failure: 1s, 2s, 4s
                    await asyncio.sleep(min(2 ** consecutive_failures, 4))

            await provider.close()
            elapsed_ms = int((time.time() - _t_start) * 1000)
            logger.info(
                f"[WARMUP] Complete: warmed={warmed}, skipped_cache={skipped_cache}, "
                f"skipped_db={skipped_db}, failed={failed}, total_leagues={len(league_ids)}, elapsed_ms={elapsed_ms}"
            )

    except Exception as e:
        logger.error(f"[WARMUP] Standings warmup failed: {e}")


async def _predictions_catchup_on_startup():
    """
    Predictions catch-up on startup (P2 resilience).

    Handles missed daily_save_predictions runs due to deploys/restarts.
    Conditions to trigger:
    - hours_since_last_prediction_saved > 6
    - ns_next_48h > 0 (there are upcoming matches to predict)

    This is fire-and-forget, idempotent (upsert), and non-blocking.
    """
    import asyncio

    # Small delay to let server fully start and ML model load
    await asyncio.sleep(5)

    try:
        async with AsyncSessionLocal() as session:
            now = datetime.utcnow()

            # 1) Check hours since last prediction saved
            res = await session.execute(
                text("SELECT MAX(created_at) FROM predictions")
            )
            last_pred_at = res.scalar()

            hours_since_last = None
            if last_pred_at:
                delta = now - last_pred_at
                hours_since_last = delta.total_seconds() / 3600

            # 2) Check NS matches in next 48h
            res = await session.execute(
                text("""
                    SELECT COUNT(*) FROM matches
                    WHERE status = 'NS'
                      AND date > NOW()
                      AND date <= NOW() + INTERVAL '48 hours'
                """)
            )
            ns_next_48h = int(res.scalar() or 0)

            # 3) Evaluate conditions
            should_catchup = (
                (hours_since_last is None or hours_since_last > 6)
                and ns_next_48h > 0
            )

            hours_str = f"{hours_since_last:.1f}" if hours_since_last else "N/A"

            if not should_catchup:
                logger.info(
                    f"[STARTUP] Predictions catch-up skipped: "
                    f"hours_since_last={hours_str}, ns_next_48h={ns_next_48h}"
                )
                return

            # 4) Trigger catch-up
            logger.warning(
                f"[OPS_ALERT] predictions catch-up on startup triggered: "
                f"hours_since_last={hours_str}, ns_next_48h={ns_next_48h}"
            )

            # Use same logic as /dashboard/predictions/trigger endpoint
            from app.db_utils import upsert

            # Check ML model is loaded
            if not ml_engine.is_loaded:
                logger.error("[STARTUP] Predictions catch-up aborted: ML model not loaded")
                return

            # Get features for upcoming matches
            feature_engineer = FeatureEngineer(session=session)
            df = await feature_engineer.get_upcoming_matches_features()

            if len(df) == 0:
                logger.info("[STARTUP] Predictions catch-up: no upcoming matches found")
                return

            # Filter to NS only
            df_ns = df[df["status"] == "NS"].copy()

            if len(df_ns) == 0:
                logger.info("[STARTUP] Predictions catch-up: no NS matches to predict")
                return

            # Generate predictions
            predictions = ml_engine.predict(df_ns)

            # Save to database (idempotent upsert)
            saved = 0
            for pred in predictions:
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
                except Exception as e:
                    logger.warning(f"[STARTUP] Predictions catch-up: match {match_id} failed: {e}")

            await session.commit()
            logger.info(
                f"[STARTUP] Predictions catch-up complete: saved={saved}, "
                f"ns_matches={len(df_ns)}, model={ml_engine.model_version}"
            )

    except Exception as e:
        logger.error(f"[STARTUP] Predictions catch-up failed: {e}")


app = FastAPI(
    title="FutbolStat MVP",
    description="Football Prediction System for FIFA World Cup",
    version="1.0.0",
    lifespan=lifespan,
)

# Add session middleware for OPS console login
# Secret key is required in production; fallback to random for dev
_session_secret = settings.OPS_SESSION_SECRET or os.urandom(32).hex()
if not settings.OPS_SESSION_SECRET and os.getenv("RAILWAY_PROJECT_ID"):
    logger.warning("[SECURITY] OPS_SESSION_SECRET not set in production - sessions will be invalidated on restart")

from starlette.middleware.sessions import SessionMiddleware
app.add_middleware(
    SessionMiddleware,
    secret_key=_session_secret,
    session_cookie="ops_session",
    max_age=settings.OPS_SESSION_TTL_HOURS * 3600,
    same_site="lax",
    https_only=os.getenv("RAILWAY_PROJECT_ID") is not None,  # Secure cookie in prod
)

# Add rate limiting
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Include routers
app.include_router(core_router)
app.include_router(logos_router)
app.include_router(model_benchmark_router)
app.include_router(football_routes_router)
app.include_router(admin_routes_router)


# Request/Response Models (HealthResponse moved to app/routes/core.py)

class ETLSyncRequest(BaseModel):
    league_ids: list[int]
    season: int
    fetch_odds: bool = False


class ETLSyncResponse(BaseModel):
    matches_synced: int
    teams_synced: int
    details: list[dict]


class TrainRequest(BaseModel):
    min_date: Optional[str] = None  # YYYY-MM-DD
    max_date: Optional[str] = None
    league_ids: Optional[list[int]] = None


class TrainResponse(BaseModel):
    model_version: str
    brier_score: float
    samples_trained: int
    feature_importance: dict


class PredictionItem(BaseModel):
    """Prediction item with contextual intelligence for iOS consumption."""
    match_id: Optional[int] = None
    match_external_id: Optional[int] = None
    home_team: str
    away_team: str
    home_team_logo: Optional[str] = None
    away_team_logo: Optional[str] = None
    date: datetime
    status: Optional[str] = None  # Match status: NS, FT, 1H, 2H, HT, etc.
    elapsed: Optional[int] = None  # Current minute for live matches (e.g., 32)
    elapsed_extra: Optional[int] = None  # Added/injury time (e.g., 3 for 90+3)
    home_goals: Optional[int] = None  # Final score (nil if not played)
    away_goals: Optional[int] = None  # Final score (nil if not played)
    league_id: Optional[int] = None
    venue: Optional[dict] = None  # Stadium: {"name": str, "city": str} or None
    events: Optional[list[dict]] = None  # Match events (goals, cards) for live timeline

    # Model pick derived from probabilities (home, draw, away)
    pick: Optional[str] = None

    # Adjusted probabilities (after team adjustments)
    probabilities: dict

    @model_validator(mode='after')
    def derive_pick_from_probabilities(self) -> 'PredictionItem':
        """Derive pick from probabilities if not set.

        Deterministic tie-breaker: home > draw > away (matches betting convention).
        """
        if self.pick is None and self.probabilities:
            probs = self.probabilities
            h = probs.get("home", 0)
            d = probs.get("draw", 0)
            a = probs.get("away", 0)
            if h or d or a:  # At least one prob exists
                # Deterministic: priority home > draw > away on ties
                if h >= d and h >= a:
                    self.pick = "home"
                elif d >= a:
                    self.pick = "draw"
                else:
                    self.pick = "away"
        return self
    # Raw model output before adjustments
    raw_probabilities: Optional[dict] = None

    fair_odds: dict
    market_odds: Optional[dict] = None

    # Confidence tier with degradation tracking
    confidence_tier: Optional[str] = None  # gold, silver, copper
    original_tier: Optional[str] = None    # Original tier before degradation

    # Value betting
    value_bets: Optional[list[dict]] = None
    has_value_bet: Optional[bool] = False
    best_value_bet: Optional[dict] = None

    # Contextual adjustments applied
    adjustment_applied: Optional[bool] = False
    adjustments: Optional[dict] = None

    # Reasoning engine (human-readable insights)
    prediction_insights: Optional[list[str]] = None
    warnings: Optional[list[str]] = None

    # Frozen prediction data (for finished matches)
    is_frozen: Optional[bool] = False
    frozen_at: Optional[str] = None  # ISO datetime when prediction was frozen
    frozen_ev: Optional[dict] = None  # EV values at freeze time

    # Rerun serving (DB-first gated)
    served_from_rerun: Optional[bool] = None  # True if served from DB rerun prediction
    rerun_model_version: Optional[str] = None  # Model version of rerun prediction


class PredictionsResponse(BaseModel):
    predictions: list[PredictionItem]
    model_version: str
    # Metadata about contextual filters applied
    context_applied: Optional[dict] = None


# /health, /telemetry, /metrics moved to app/routes/core.py

@app.get("/sync/status")
async def get_sync_status():
    """
    Get current sync status for iOS display.

    Returns last sync timestamp and API budget info.
    Used by mobile app to show data freshness.
    """
    last_sync = get_last_sync_time()

    # Best-effort: expose real budget numbers (prefer internal guardrail, optionally enrich with cached /status)
    daily_budget = int(getattr(settings, "API_DAILY_BUDGET", 0) or 0) or 75000
    daily_used = None
    remaining_pct = None
    api_account_status = None

    try:
        from app.etl.api_football import get_api_budget_status, get_api_account_status  # type: ignore

        internal = get_api_budget_status()
        # Internal is authoritative for guardrail; can be None early in day before first request
        daily_used = internal.get("budget_used")
        daily_budget = internal.get("budget_total") or daily_budget

        # Enrich with real API status (cached 10 min); don't fail endpoint if unavailable
        api_account_status = await get_api_account_status()  # type: ignore
        ext_used = api_account_status.get("requests_today")
        ext_limit = api_account_status.get("requests_limit")
        if isinstance(ext_used, int) and ext_used >= 0:
            # Use the max to avoid under-reporting after process restart
            if isinstance(daily_used, int):
                daily_used = max(daily_used, ext_used)
            else:
                daily_used = ext_used
        if isinstance(ext_limit, int) and ext_limit > 0:
            daily_budget = ext_limit
    except Exception:
        pass

    if isinstance(daily_used, int) and daily_budget > 0:
        remaining_pct = round((1 - (daily_used / daily_budget)) * 100, 1)

    return {
        "last_sync_at": last_sync.isoformat() if last_sync else None,
        "sync_interval_seconds": 60,
        "daily_api_calls": daily_used,
        "daily_budget": daily_budget,
        "budget_remaining_percent": remaining_pct,
        "leagues": get_sync_leagues(),
        "api_account_status": api_account_status,  # optional debug visibility for iOS
    }


@app.get("/")
@limiter.limit("60/minute")
async def root(request: Request):
    """Root endpoint with API info."""
    return {
        "name": "FutbolStat MVP",
        "version": "1.0.0",
        "description": "Football Prediction System for FIFA World Cup",
        "endpoints": {
            "health": "/health",
            "etl_sync": "POST /etl/sync",
            "train": "POST /model/train",
            "predictions": "GET /predictions/upcoming",
        },
    }


@app.post("/etl/sync", response_model=ETLSyncResponse)
@limiter.limit("10/minute")
async def etl_sync(
    request: Request,
    body: ETLSyncRequest,
    session: AsyncSession = Depends(get_async_session),
    _: bool = Depends(verify_api_key),
):
    """
    Sync fixtures from API-Football.

    Fetches matches for specified leagues and season.
    Requires API key authentication.
    """
    logger.info(f"ETL sync request: {body}")

    # Validate league IDs
    for league_id in body.league_ids:
        if league_id not in COMPETITIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown league ID: {league_id}. Valid IDs: {ALL_LEAGUE_IDS}",
            )

    provider = APIFootballProvider()
    try:
        pipeline = ETLPipeline(provider=provider, session=session)
        result = await pipeline.sync_multiple_leagues(
            league_ids=body.league_ids,
            season=body.season,
            fetch_odds=body.fetch_odds,
        )

        return ETLSyncResponse(
            matches_synced=result["total_matches_synced"],
            teams_synced=result["total_teams_synced"],
            details=result["details"],
        )
    finally:
        await provider.close()


@app.post("/etl/sync-historical")
@limiter.limit("5/minute")
async def etl_sync_historical(
    request: Request,
    start_year: int = 2018,
    end_year: Optional[int] = None,
    league_ids: Optional[list[int]] = None,
    session: AsyncSession = Depends(get_async_session),
    _: bool = Depends(verify_api_key),
):
    """
    Sync historical data for multiple seasons.

    This is a long-running operation. Use for initial data loading.
    Requires API key authentication.
    """
    if league_ids is None:
        league_ids = ALL_LEAGUE_IDS

    provider = APIFootballProvider()
    try:
        pipeline = ETLPipeline(provider=provider, session=session)
        result = await pipeline.sync_historical_data(
            league_ids=league_ids,
            start_year=start_year,
            end_year=end_year,
        )
        return result
    finally:
        await provider.close()


@app.post("/etl/sync-window")
@limiter.limit("5/minute")
async def etl_sync_window(
    request: Request,
    days_ahead: int = 10,
    days_back: int = 1,
    _: bool = Depends(verify_api_key),
):
    """
    Sync fixtures by date window (not by season).

    This endpoint triggers the global_sync_window job which loads fixtures
    for a range of dates regardless of season. Useful for loading LATAM 2026
    fixtures when CURRENT_SEASON is still set to 2025.

    Args:
        days_ahead: Days ahead to sync (default: 10)
        days_back: Days back to sync (default: 1)

    Requires API key authentication.
    """
    from app.ops.audit import log_ops_action

    logger.info(f"[ETL] sync-window request: days_back={days_back}, days_ahead={days_ahead}")

    start_time = time.time()
    result = await global_sync_window(days_ahead=days_ahead, days_back=days_back)
    duration_ms = int((time.time() - start_time) * 1000)

    # Audit log
    try:
        async with AsyncSessionLocal() as audit_session:
            await log_ops_action(
                session=audit_session,
                request=request,
                action="sync_window",
                params={"days_ahead": days_ahead, "days_back": days_back},
                result="ok" if not result.get("error") else "error",
                result_detail={
                    "matches_synced": result.get("matches_synced", 0),
                    "days_processed": result.get("days_processed", 0),
                },
                error_message=result.get("error"),
                duration_ms=duration_ms,
            )
    except Exception as audit_err:
        logger.warning(f"Failed to log audit for sync_window: {audit_err}")

    return {
        "status": "ok",
        "matches_synced": result.get("matches_synced", 0),
        "days_processed": result.get("days_processed", 0),
        "window": result.get("window", {}),
        "by_date": result.get("by_date", {}),
        "error": result.get("error"),
    }


@app.post("/etl/refresh-aggregates")
@limiter.limit("5/minute")
async def etl_refresh_aggregates(
    request: Request,
    session: AsyncSession = Depends(get_async_session),
    _: bool = Depends(verify_api_key),
):
    """
    Manually trigger league aggregates refresh.

    Computes league baselines and team profiles for all leagues with
    sufficient data. This is the same job that runs daily at 06:30 UTC.

    Returns metrics about the refresh operation.
    Requires API key authentication.
    """
    logger.info("[AGGREGATES] Manual refresh triggered via API")

    from app.aggregates.refresh_job import refresh_all_aggregates, get_aggregates_status

    # Get status before
    status_before = await get_aggregates_status(session)

    # Run refresh
    result = await refresh_all_aggregates(session)

    # Get status after
    status_after = await get_aggregates_status(session)

    return {
        "status": "ok",
        "refresh_result": result,
        "status_before": status_before,
        "status_after": status_after,
    }


@app.get("/aggregates/status")
@limiter.limit("30/minute")
async def get_aggregates_status_endpoint(
    request: Request,
    session: AsyncSession = Depends(get_async_session),
    _: bool = Depends(verify_api_key),
):
    """
    Get current status of league aggregates tables.

    Returns counts and latest computation timestamps.
    Requires API key authentication.
    """
    from app.aggregates.refresh_job import get_aggregates_status

    status = await get_aggregates_status(session)
    return {"status": "ok", **status}


@app.get("/aggregates/breakdown")
@limiter.limit("30/minute")
async def get_aggregates_breakdown(
    request: Request,
    session: AsyncSession = Depends(get_async_session),
    _: bool = Depends(verify_api_key),
):
    """
    Get detailed breakdown of aggregates by dimension.

    Clarifies what baselines_created and profiles_created represent.
    Requires API key authentication.
    """
    from sqlalchemy import select, func, distinct
    from app.models import LeagueSeasonBaseline, LeagueTeamProfile

    # Baselines breakdown
    total_baselines = (await session.execute(
        select(func.count(LeagueSeasonBaseline.id))
    )).scalar() or 0

    distinct_leagues = (await session.execute(
        select(func.count(distinct(LeagueSeasonBaseline.league_id)))
    )).scalar() or 0

    distinct_seasons = (await session.execute(
        select(func.count(distinct(LeagueSeasonBaseline.season)))
    )).scalar() or 0

    distinct_dates = (await session.execute(
        select(func.count(distinct(LeagueSeasonBaseline.as_of_date)))
    )).scalar() or 0

    # Profiles breakdown
    total_profiles = (await session.execute(
        select(func.count(LeagueTeamProfile.id))
    )).scalar() or 0

    distinct_teams = (await session.execute(
        select(func.count(distinct(LeagueTeamProfile.team_id)))
    )).scalar() or 0

    profiles_with_min_sample = (await session.execute(
        select(func.count(LeagueTeamProfile.id))
        .where(LeagueTeamProfile.min_sample_ok == True)
    )).scalar() or 0

    # Season distribution
    seasons_result = await session.execute(
        select(
            LeagueSeasonBaseline.season,
            func.count(LeagueSeasonBaseline.id)
        )
        .group_by(LeagueSeasonBaseline.season)
        .order_by(LeagueSeasonBaseline.season.desc())
    )
    seasons_breakdown = {str(row[0]): row[1] for row in seasons_result}

    return {
        "status": "ok",
        "baselines": {
            "total_rows": total_baselines,
            "distinct_league_id": distinct_leagues,
            "distinct_season": distinct_seasons,
            "distinct_as_of_date": distinct_dates,
            "note": "Each row = one (league_id, season, as_of_date) combination",
        },
        "profiles": {
            "total_rows": total_profiles,
            "distinct_team_id": distinct_teams,
            "with_min_sample_ok": profiles_with_min_sample,
            "note": "Each row = one (league_id, season, team_id, as_of_date) combination",
        },
        "seasons_breakdown": seasons_breakdown,
    }


@app.post("/model/train", response_model=TrainResponse)
@limiter.limit("5/minute")
async def train_model(
    request: Request,
    body: TrainRequest = None,
    session: AsyncSession = Depends(get_async_session),
    _: bool = Depends(verify_api_key),
):
    """
    Train the prediction model.

    Uses historical match data to train XGBoost model.
    Requires API key authentication.
    """
    body = body or TrainRequest()

    logger.info("Starting model training...")

    # Parse dates
    min_date = None
    max_date = None
    if body.min_date:
        min_date = datetime.strptime(body.min_date, "%Y-%m-%d")
    if body.max_date:
        max_date = datetime.strptime(body.max_date, "%Y-%m-%d")

    # Build training dataset
    feature_engineer = FeatureEngineer(session=session)
    df = await feature_engineer.build_training_dataset(
        min_date=min_date,
        max_date=max_date,
        league_ids=body.league_ids,
    )

    if len(df) < 100:
        raise HTTPException(
            status_code=400,
            detail=f"Insufficient training data: {len(df)} samples. Need at least 100.",
        )

    # Train model in executor to avoid blocking the event loop
    import asyncio
    from concurrent.futures import ThreadPoolExecutor

    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as executor:
        result = await loop.run_in_executor(executor, ml_engine.train, df)

    # Save model to PostgreSQL for fast startup on future deploys
    snapshot_id = await persist_model_snapshot(
        session=session,
        engine=ml_engine,
        brier_score=result["brier_score"],
        cv_scores=result["cv_scores"],
        samples_trained=result["samples_trained"],
    )
    logger.info(f"Model saved to PostgreSQL as snapshot {snapshot_id}")

    return TrainResponse(
        model_version=result["model_version"],
        brier_score=result["brier_score"],
        samples_trained=result["samples_trained"],
        feature_importance=result["feature_importance"],
    )


@app.get("/predictions/upcoming", response_model=PredictionsResponse)
@limiter.limit("30/minute")
async def get_predictions(
    request: Request,
    league_ids: Optional[str] = None,  # comma-separated
    days: int = 7,  # Legacy: applies to both back and ahead if specific params not set
    days_back: Optional[int] = None,  # Past N days (finished matches with scores)
    days_ahead: Optional[int] = None,  # Future N days (upcoming matches)
    save: bool = False,  # Save predictions to database
    with_context: bool = True,  # Apply contextual intelligence
    session: AsyncSession = Depends(get_async_session),
    _: bool = Depends(verify_api_key),
):
    """
    Get predictions for upcoming matches with contextual intelligence.

    Returns probabilities, fair odds, and reasoning insights for matches.
    Applies team adjustments, league drift detection, and market movement analysis.

    Args:
        league_ids: Comma-separated league IDs to filter
        days: Legacy param - applies to both directions if days_back/days_ahead not set
        days_back: Past N days for finished matches (overrides 'days' for past)
        days_ahead: Future N days for upcoming matches (overrides 'days' for future)
        save: Persist predictions to database for auditing
        with_context: Apply contextual intelligence (team adjustments, drift, odds)

    Priority window example: ?days_back=1&days_ahead=1 → yesterday/today/tomorrow
    Full window example: ?days_back=7&days_ahead=7 → 15-day range

    Uses in-memory caching (5 min TTL) for faster responses.
    """
    global _predictions_cache

    if not ml_engine.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Train a model first with POST /model/train",
        )

    # Resolve actual days_back and days_ahead (new params override legacy 'days')
    actual_days_back = days_back if days_back is not None else days
    actual_days_ahead = days_ahead if days_ahead is not None else days
    logger.info(f"Predictions params: days={days}, days_back={days_back}, days_ahead={days_ahead} -> actual_back={actual_days_back}, actual_ahead={actual_days_ahead}")

    # Cache key based on parameters
    cache_key = f"{league_ids or 'all'}_{actual_days_back}_{actual_days_ahead}_{with_context}"
    now = time.time()

    # Check cache (only for default full requests without league filter)
    is_default_full = (
        league_ids is None
        and actual_days_back == 7
        and actual_days_ahead == 7
        and not save
        and with_context
    )
    if is_default_full and _predictions_cache["data"] is not None:
        if now - _predictions_cache["timestamp"] < _predictions_cache["ttl"]:
            logger.info("Returning cached predictions")
            return _predictions_cache["data"]

    # Priority optimization: serve any cacheable request from full (7+7) cache
    # This applies to priority requests (1+1) or any subset of the default window
    is_cacheable_subset = (
        league_ids is None
        and actual_days_back <= 7
        and actual_days_ahead <= 7
        and not save
        and with_context
    )

    # Helper to filter predictions by date range
    def _filter_predictions_by_range(
        cached_response: PredictionsResponse,
        days_back: int,
        days_ahead: int
    ) -> PredictionsResponse:
        from datetime import timezone
        today_start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        range_start = today_start - timedelta(days=days_back)
        range_end = today_start + timedelta(days=days_ahead + 1)  # +1 to include full day

        filtered = [
            p for p in cached_response.predictions
            if range_start <= p.date.replace(tzinfo=timezone.utc) < range_end
        ]
        return PredictionsResponse(
            predictions=filtered,
            model_version=cached_response.model_version,
            context_applied=cached_response.context_applied,
        )

    # If cache is warm and request is a subset, filter and return immediately
    if is_cacheable_subset and _predictions_cache["data"] is not None:
        if now - _predictions_cache["timestamp"] < _predictions_cache["ttl"]:
            if is_default_full:
                # Full request with warm cache - return as-is (already handled above, but safety)
                _incr("predictions_cache_hit_full")
                logger.info("predictions_cache | cache_hit | type=full, count=%d", len(_predictions_cache["data"].predictions))
                return _predictions_cache["data"]
            else:
                # Subset request - filter from cache
                result = _filter_predictions_by_range(
                    _predictions_cache["data"],
                    actual_days_back,
                    actual_days_ahead
                )
                _incr("predictions_cache_hit_priority")
                logger.info(
                    "predictions_cache | cache_hit | type=priority, filtered_count=%d, full_count=%d, days_back=%d, days_ahead=%d",
                    len(result.predictions), len(_predictions_cache["data"].predictions),
                    actual_days_back, actual_days_ahead
                )
                return result

    # Parse league IDs
    league_id_list = None
    if league_ids:
        league_id_list = [int(x.strip()) for x in league_ids.split(",")]

    # Cold start optimization: when cache is cold and request is a subset,
    # always fetch full (7+7) to populate cache, then filter result
    # This ensures first priority request warms the cache for subsequent requests
    fetch_days_back = actual_days_back
    fetch_days_ahead = actual_days_ahead
    needs_filtering = False

    if is_cacheable_subset and not is_default_full and league_id_list is None:
        # Subset request with cold cache - fetch full range to warm cache
        _incr("predictions_cache_miss_priority_upgrade")
        logger.info(
            "predictions_cache | cache_miss | type=priority_upgrade, requested_days=%d+%d, fetching=7+7",
            actual_days_back, actual_days_ahead
        )
        fetch_days_back = 7
        fetch_days_ahead = 7
        needs_filtering = True
    elif is_cacheable_subset and is_default_full:
        _incr("predictions_cache_miss_full")
        logger.info("predictions_cache | cache_miss | type=full")
    else:
        logger.info("predictions_cache | cache_bypass | league_ids=%s, save=%s, with_context=%s",
                    league_ids, save, with_context)

    # Track compute time for cache miss (with per-stage timing)
    _compute_start = time.time()
    _stage_times = {}

    # Get features for upcoming matches
    # iOS progressive loading:
    #   Priority: days_back=1, days_ahead=1 → yesterday/today/tomorrow (~50-100 matches)
    #   Full: days_back=7, days_ahead=7 → 15-day window (~300 matches)
    _t0 = time.time()
    feature_engineer = FeatureEngineer(session=session)
    df = await feature_engineer.get_upcoming_matches_features(
        league_ids=league_id_list,
        include_recent_days=fetch_days_back,  # Past N days for finished matches
        days_ahead=fetch_days_ahead,  # Future N days for upcoming matches
    )
    _stage_times["features_ms"] = (time.time() - _t0) * 1000
    logger.info(f"Predictions query: days_back={fetch_days_back}, days_ahead={fetch_days_ahead}, matches={len(df)}")

    if len(df) == 0:
        return PredictionsResponse(
            predictions=[],
            model_version=ml_engine.model_version,
        )

    # Load contextual intelligence data
    team_adjustments = None
    context = None
    context_metadata = {
        "team_adjustments_loaded": False,
        "unstable_leagues": 0,
        "odds_movements_detected": 0,
    }

    if with_context:
        from app.ml.recalibration import RecalibrationEngine, load_team_adjustments, get_drift_cache_stats
        _t1 = time.time()

        try:
            # Load team adjustments (includes raw data to avoid duplicate query)
            _t_adj = time.time()
            team_adjustments = await load_team_adjustments(session)
            _stage_times["adjustments_ms"] = (time.time() - _t_adj) * 1000
            context_metadata["team_adjustments_loaded"] = True

            # Initialize recalibrator for context gathering
            recalibrator = RecalibrationEngine(session)

            # Detect unstable leagues (with TTL cache)
            _drift_stats_before = get_drift_cache_stats()
            _t_drift = time.time()
            drift_result = await recalibrator.detect_league_drift()
            unstable_leagues = {alert["league_id"] for alert in drift_result.get("drift_alerts", [])}
            _stage_times["drift_ms"] = (time.time() - _t_drift) * 1000
            _drift_stats_after = get_drift_cache_stats()
            _drift_was_hit = _drift_stats_after["hits"] > _drift_stats_before["hits"]
            _stage_times["drift_cache"] = "HIT" if _drift_was_hit else "MISS"
            context_metadata["unstable_leagues"] = len(unstable_leagues)

            # Check odds movements for upcoming matches (batch query, no N+1)
            _t_odds = time.time()
            odds_result = await recalibrator.check_all_upcoming_odds_movements(days_ahead=days)
            odds_movements = {
                alert["match_id"]: alert
                for alert in odds_result.get("alerts", [])
            }
            _stage_times["odds_ms"] = (time.time() - _t_odds) * 1000
            context_metadata["odds_movements_detected"] = len(odds_movements)

            # Build team details from already-loaded adjustments (no duplicate query)
            team_details = {}
            for adj in team_adjustments.get("raw", []):
                home_anomaly_rate = adj.home_anomalies / adj.home_predictions if adj.home_predictions > 0 else 0
                away_anomaly_rate = adj.away_anomalies / adj.away_predictions if adj.away_predictions > 0 else 0
                team_details[adj.team_id] = {
                    "home_anomaly_rate": home_anomaly_rate,
                    "away_anomaly_rate": away_anomaly_rate,
                    "consecutive_minimal": adj.consecutive_minimal_count,
                    "international_penalty": adj.international_penalty,
                }

            # Build context dictionary
            context = {
                "unstable_leagues": unstable_leagues,
                "odds_movements": odds_movements,
                "international_commitments": {},  # Filled from team_details
                "team_details": team_details,
            }

            # Add international commitments from team_details
            for team_id, details in team_details.items():
                if details["international_penalty"] < 1.0:
                    context["international_commitments"][team_id] = {
                        "penalty": details["international_penalty"],
                        "days": 3,  # Approximation
                    }

            _stage_times["context_ms"] = (time.time() - _t1) * 1000
            logger.info(
                f"Context loaded: {len(unstable_leagues)} unstable leagues, "
                f"{len(odds_movements)} odds movements"
            )

        except Exception as e:
            _stage_times["context_ms"] = (time.time() - _t1) * 1000
            logger.warning(f"Error loading context: {e}. Predictions will be made without context.")

    # Make predictions with context
    _t2 = time.time()
    predictions = ml_engine.predict(df, team_adjustments=team_adjustments, context=context)
    _stage_times["predict_ms"] = (time.time() - _t2) * 1000

    # For finished matches, overlay frozen prediction data if available
    _t3 = time.time()
    predictions = await _overlay_frozen_predictions(session, predictions)
    _stage_times["overlay_ms"] = (time.time() - _t3) * 1000

    # For NS matches, overlay rerun predictions if PREFER_RERUN_PREDICTIONS=true
    _t3b = time.time()
    # Build match_dates dict from DataFrame for freshness check
    match_dates = {}
    if "match_id" in df.columns and "date" in df.columns:
        for _, row in df.iterrows():
            mid = row.get("match_id")
            mdate = row.get("date")
            if mid and mdate:
                match_dates[int(mid)] = mdate if isinstance(mdate, datetime) else datetime.fromisoformat(str(mdate).replace("Z", "+00:00"))
    predictions, rerun_stats = await _overlay_rerun_predictions(session, predictions, match_dates)
    _stage_times["rerun_overlay_ms"] = (time.time() - _t3b) * 1000
    if rerun_stats.get("db_hits", 0) > 0 or rerun_stats.get("db_stale", 0) > 0:
        logger.info(
            f"rerun_serving | db_hits={rerun_stats['db_hits']} db_stale={rerun_stats['db_stale']} "
            f"live_fallback={rerun_stats['live_fallback']} total_ns={rerun_stats['total_ns']}"
        )

    # Apply team identity overrides (rebranding, e.g., La Equidad → Internacional de Bogotá)
    _t4 = time.time()
    predictions = await _apply_team_overrides(session, predictions)
    _stage_times["overrides_ms"] = (time.time() - _t4) * 1000

    # ═══════════════════════════════════════════════════════════════
    # FASE 1: Apply draw cap to value bets (portfolio level)
    # ═══════════════════════════════════════════════════════════════
    _t5 = time.time()
    from app.ml.policy import apply_draw_cap, get_policy_config
    policy_config = get_policy_config()
    predictions, policy_metadata = apply_draw_cap(
        predictions,
        max_draw_share=policy_config["max_draw_share"],
        enabled=policy_config["draw_cap_enabled"],
    )
    _stage_times["policy_cap_ms"] = (time.time() - _t5) * 1000
    if policy_metadata.get("cap_applied"):
        logger.info(
            f"policy_draw_cap | applied | draws={policy_metadata['n_draws_original']}→{policy_metadata['n_draws_after']} "
            f"share={policy_metadata['draw_share_original']}%→{policy_metadata['draw_share_after']}%"
        )
    # ═══════════════════════════════════════════════════════════════

    # Save predictions to database if requested
    if save:
        saved_count = await _save_predictions_to_db(session, predictions, ml_engine.model_version)
        logger.info(f"Saved {saved_count} predictions to database")

    # Convert to response model
    prediction_items = []
    for pred in predictions:
        item = PredictionItem(
            match_id=pred.get("match_id"),
            match_external_id=pred.get("match_external_id"),
            home_team=pred["home_team"],
            away_team=pred["away_team"],
            home_team_logo=pred.get("home_team_logo"),
            away_team_logo=pred.get("away_team_logo"),
            date=pred["date"],
            status=pred.get("status"),
            elapsed=pred.get("elapsed"),
            elapsed_extra=pred.get("elapsed_extra"),
            home_goals=pred.get("home_goals"),
            away_goals=pred.get("away_goals"),
            league_id=pred.get("league_id"),
            venue=pred.get("venue"),
            events=pred.get("events"),
            probabilities=pred["probabilities"],
            raw_probabilities=pred.get("raw_probabilities"),
            fair_odds=pred["fair_odds"],
            market_odds=pred.get("market_odds"),
            confidence_tier=pred.get("confidence_tier"),
            original_tier=pred.get("original_tier"),
            value_bets=pred.get("value_bets"),
            has_value_bet=pred.get("has_value_bet", False),
            best_value_bet=pred.get("best_value_bet"),
            adjustment_applied=pred.get("adjustment_applied", False),
            adjustments=pred.get("adjustments"),
            prediction_insights=pred.get("prediction_insights"),
            warnings=pred.get("warnings"),
            # Frozen prediction fields
            is_frozen=pred.get("is_frozen", False),
            frozen_at=pred.get("frozen_at"),
            frozen_ev=pred.get("frozen_ev"),
            # Rerun serving fields
            served_from_rerun=pred.get("served_from_rerun"),
            rerun_model_version=pred.get("rerun_model_version"),
        )
        prediction_items.append(item)

    response = PredictionsResponse(
        predictions=prediction_items,
        model_version=ml_engine.model_version,
        context_applied=context_metadata if with_context else None,
    )

    # Compute time for telemetry
    _compute_ms = (time.time() - _compute_start) * 1000
    _stage_times["total_ms"] = _compute_ms

    # Log per-stage timing breakdown for performance monitoring
    # Separate numeric timings from string metadata (like drift_cache hit/miss)
    _timing_parts = [f"{k}={v:.0f}" for k, v in _stage_times.items() if isinstance(v, (int, float))]
    _meta_parts = [f"{k}={v}" for k, v in _stage_times.items() if isinstance(v, str)]
    logger.info(
        "predictions_timing | %s | %s | matches=%d",
        " | ".join(_timing_parts),
        " | ".join(_meta_parts) if _meta_parts else "no_meta",
        len(prediction_items)
    )

    # Cache the response (for 7+7 requests or upgraded priority requests)
    # This ensures the cache is always populated with full data
    should_cache = is_default_full or needs_filtering
    if should_cache:
        _predictions_cache["data"] = response
        _predictions_cache["timestamp"] = now
        logger.info(
            "predictions_cache | cached | compute_ms=%.1f, full_count=%d, type=%s",
            _compute_ms, len(prediction_items),
            "priority_upgrade" if needs_filtering else "full"
        )

    # If this was an upgraded priority request, filter the result before returning
    if needs_filtering:
        filtered_response = _filter_predictions_by_range(
            response,
            actual_days_back,
            actual_days_ahead
        )
        logger.info(
            "predictions_cache | filtered | filtered_count=%d, full_count=%d, days_back=%d, days_ahead=%d",
            len(filtered_response.predictions), len(response.predictions),
            actual_days_back, actual_days_ahead
        )
        return filtered_response

    return response


async def _save_predictions_to_db(
    session: AsyncSession,
    predictions: list[dict],
    model_version: str,
) -> int:
    """Save predictions to database for later auditing."""
    from app.db_utils import upsert

    saved = 0
    for pred in predictions:
        match_id = pred.get("match_id")
        if not match_id:
            continue

        probs = pred["probabilities"]

        try:
            # Use generic upsert for cross-database compatibility
            await upsert(
                session,
                Prediction,
                values={
                    "match_id": match_id,
                    "model_version": model_version,
                    "home_prob": probs["home"],
                    "draw_prob": probs["draw"],
                    "away_prob": probs["away"],
                },
                conflict_columns=["match_id", "model_version"],
                update_columns=["home_prob", "draw_prob", "away_prob"],
            )
            saved += 1
        except Exception as e:
            logger.warning(f"Error saving prediction for match {match_id}: {e}")

    await session.commit()
    return saved


# Metrics counters for rerun serving (DB-first vs live fallback)
_rerun_serving_stats = {
    "db_hits": 0,
    "db_stale": 0,
    "live_fallback": 0,
    "total_ns_served": 0,
}


async def _overlay_rerun_predictions(
    session: AsyncSession,
    predictions: list[dict],
    match_dates: dict[int, datetime],  # match_id -> match date for freshness check
) -> tuple[list[dict], dict]:
    """
    Overlay rerun predictions from DB for NS matches.

    DISABLED (2025-01): Per audit directive, serving is baseline-only.
    Rerun/shadow predictions are for evaluation only, not production serving.
    This function now always returns predictions unchanged (baseline).

    The PREFER_RERUN_PREDICTIONS flag and rerun infrastructure remain for
    OPS/analysis endpoints but do not affect public prediction serving.

    Returns:
        tuple: (unchanged predictions, empty stats dict)
    """
    # AUDIT P0: Baseline-only serving - always return unchanged
    stats = {"db_hits": 0, "db_stale": 0, "live_fallback": 0, "total_ns": 0}
    return predictions, stats

    if not predictions:
        return predictions, stats

    # Get NS match IDs
    ns_match_ids = [
        p.get("match_id") for p in predictions
        if p.get("match_id") and p.get("status") == "NS"
    ]
    if not ns_match_ids:
        return predictions, stats

    stats["total_ns"] = len(ns_match_ids)

    # Query rerun predictions (those with run_id, most recent per match)
    # Using a subquery to get the latest prediction per match_id with run_id
    result = await session.execute(
        text("""
            SELECT DISTINCT ON (match_id)
                match_id, model_version, home_prob, draw_prob, away_prob,
                created_at, run_id
            FROM predictions
            WHERE match_id = ANY(:match_ids)
              AND run_id IS NOT NULL
            ORDER BY match_id, created_at DESC
        """),
        {"match_ids": ns_match_ids}
    )
    rerun_preds = {row[0]: row for row in result.fetchall()}

    if not rerun_preds:
        stats["live_fallback"] = len(ns_match_ids)
        return predictions, stats

    # Freshness threshold
    freshness_hours = settings.RERUN_FRESHNESS_HOURS
    now = datetime.utcnow()

    # Overlay rerun predictions where fresh
    for pred in predictions:
        match_id = pred.get("match_id")
        status = pred.get("status")

        if status != "NS" or match_id not in rerun_preds:
            if status == "NS":
                stats["live_fallback"] += 1
            continue

        db_pred = rerun_preds[match_id]
        pred_created_at = db_pred[5]  # created_at
        match_date = match_dates.get(match_id)

        # Freshness check: prediction must be within RERUN_FRESHNESS_HOURS of now
        # OR within RERUN_FRESHNESS_HOURS before match kickoff
        is_fresh = False
        hours_since_pred = (now - pred_created_at).total_seconds() / 3600

        if hours_since_pred <= freshness_hours:
            is_fresh = True
        elif match_date:
            # Also fresh if match is soon and pred was made recently enough
            hours_to_kickoff = (match_date - now).total_seconds() / 3600
            if hours_to_kickoff > 0 and hours_since_pred <= freshness_hours * 2:
                is_fresh = True

        if is_fresh:
            # Overlay DB prediction
            pred["probabilities"] = {
                "home": float(db_pred[2]),
                "draw": float(db_pred[3]),
                "away": float(db_pred[4]),
            }
            # Recalculate fair odds from new probabilities
            pred["fair_odds"] = {
                "home": round(1.0 / db_pred[2], 2) if db_pred[2] > 0 else None,
                "draw": round(1.0 / db_pred[3], 2) if db_pred[3] > 0 else None,
                "away": round(1.0 / db_pred[4], 2) if db_pred[4] > 0 else None,
            }
            # Mark as served from rerun
            pred["served_from_rerun"] = True
            pred["rerun_model_version"] = db_pred[1]
            stats["db_hits"] += 1
        else:
            stats["db_stale"] += 1
            stats["live_fallback"] += 1

    # Update global counters
    _rerun_serving_stats["db_hits"] += stats["db_hits"]
    _rerun_serving_stats["db_stale"] += stats["db_stale"]
    _rerun_serving_stats["live_fallback"] += stats["live_fallback"]
    _rerun_serving_stats["total_ns_served"] += stats["total_ns"]

    # Record Prometheus metrics
    try:
        from app.telemetry.metrics import record_rerun_serving_batch
        record_rerun_serving_batch(
            db_hits=stats["db_hits"],
            db_stale=stats["db_stale"],
            live_fallback=stats["live_fallback"],
            total_ns=stats["total_ns"],
        )
    except Exception as e:
        logger.warning(f"Failed to record rerun serving metrics: {e}")

    return predictions, stats


async def _overlay_frozen_predictions(
    session: AsyncSession,
    predictions: list[dict],
) -> list[dict]:
    """
    Overlay frozen prediction data for finished matches.

    For matches that have frozen predictions (is_frozen=True), we replace
    the dynamically calculated values with the frozen values. This ensures
    users see the ORIGINAL prediction they saw before the match, not a
    recalculated one after model retraining.

    Frozen data includes:
    - frozen_odds_home/draw/away: Bookmaker odds at freeze time
    - frozen_ev_home/draw/away: EV calculations at freeze time
    - frozen_confidence_tier: Confidence tier at freeze time
    - frozen_value_bets: Value bets at freeze time
    """
    if not predictions:
        return predictions

    # Get match IDs to look up frozen predictions
    match_ids = [p.get("match_id") for p in predictions if p.get("match_id")]
    if not match_ids:
        return predictions

    # Query frozen predictions for these matches
    result = await session.execute(
        select(Prediction)
        .where(
            Prediction.match_id.in_(match_ids),
            Prediction.is_frozen == True,  # noqa: E712
        )
    )
    frozen_preds = {p.match_id: p for p in result.scalars().all()}

    if not frozen_preds:
        return predictions

    # Overlay frozen data for finished matches
    for pred in predictions:
        match_id = pred.get("match_id")
        status = pred.get("status")

        # Only overlay for finished matches with frozen predictions
        if match_id in frozen_preds and status not in ("NS", None):
            frozen = frozen_preds[match_id]

            # Overlay frozen odds if available (from when prediction was frozen)
            if frozen.frozen_odds_home is not None:
                pred["market_odds"] = {
                    "home": frozen.frozen_odds_home,
                    "draw": frozen.frozen_odds_draw,
                    "away": frozen.frozen_odds_away,
                    "is_frozen": True,  # Flag to indicate these are frozen odds
                }

            # Overlay frozen confidence tier
            if frozen.frozen_confidence_tier:
                # Keep original for reference but use frozen as main
                pred["original_tier"] = pred.get("confidence_tier")
                pred["confidence_tier"] = frozen.frozen_confidence_tier

            # Overlay frozen value bets
            if frozen.frozen_value_bets:
                pred["value_bets"] = frozen.frozen_value_bets
                pred["has_value_bet"] = len(frozen.frozen_value_bets) > 0
                if frozen.frozen_value_bets:
                    # Find best value bet (highest EV) - support both old "ev" and new "expected_value" keys
                    best = max(frozen.frozen_value_bets, key=lambda x: x.get("expected_value", x.get("ev", 0)))
                    pred["best_value_bet"] = best

            # Add frozen metadata
            pred["is_frozen"] = True
            pred["frozen_at"] = frozen.frozen_at.isoformat() if frozen.frozen_at else None

            # Add frozen EV values for reference
            if frozen.frozen_ev_home is not None:
                pred["frozen_ev"] = {
                    "home": frozen.frozen_ev_home,
                    "draw": frozen.frozen_ev_draw,
                    "away": frozen.frozen_ev_away,
                }

    return predictions


async def _apply_team_overrides(
    session: AsyncSession,
    predictions: list[dict],
) -> list[dict]:
    """
    Apply team identity overrides to predictions.

    For rebranded teams (e.g., La Equidad → Internacional de Bogotá),
    replaces display names/logos based on match date and effective_from.

    Args:
        session: Database session.
        predictions: List of prediction dicts with team info.

    Returns:
        Predictions with overridden team names/logos where applicable.
    """
    if not predictions:
        return predictions

    # Collect all unique external team IDs
    external_ids = set()
    for pred in predictions:
        home_ext = pred.get("home_team_external_id")
        away_ext = pred.get("away_team_external_id")
        if home_ext:
            external_ids.add(home_ext)
        if away_ext:
            external_ids.add(away_ext)

    if not external_ids:
        return predictions

    # Batch load all overrides (single query)
    overrides = await preload_team_overrides(session, list(external_ids))

    if not overrides:
        return predictions

    # Apply overrides to each prediction
    override_count = 0
    for pred in predictions:
        match_date = pred.get("date")
        if not match_date:
            continue

        # Convert to datetime if string
        if isinstance(match_date, str):
            match_date = datetime.fromisoformat(match_date.replace("Z", "+00:00"))

        # Home team override
        home_ext = pred.get("home_team_external_id")
        if home_ext:
            home_display = resolve_team_display(
                overrides,
                home_ext,
                match_date,
                pred.get("home_team", "Unknown"),
                pred.get("home_team_logo"),
            )
            if home_display.is_override:
                pred["home_team"] = home_display.name
                if home_display.logo_url:
                    pred["home_team_logo"] = home_display.logo_url
                override_count += 1

        # Away team override
        away_ext = pred.get("away_team_external_id")
        if away_ext:
            away_display = resolve_team_display(
                overrides,
                away_ext,
                match_date,
                pred.get("away_team", "Unknown"),
                pred.get("away_team_logo"),
            )
            if away_display.is_override:
                pred["away_team"] = away_display.name
                if away_display.logo_url:
                    pred["away_team_logo"] = away_display.logo_url
                override_count += 1

    if override_count > 0:
        logger.info(f"Applied {override_count} team identity overrides to predictions")

    return predictions


@app.get("/predictions/match/{match_id}")
async def get_match_prediction(
    match_id: int,
    session: AsyncSession = Depends(get_async_session),
    _: bool = Depends(verify_api_key),
):
    """Get prediction for a specific match."""
    if not ml_engine.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Train a model first.",
        )

    # Get match
    result = await session.execute(select(Match).where(Match.id == match_id))
    match = result.scalar_one_or_none()

    if not match:
        raise HTTPException(status_code=404, detail="Match not found")

    # Get features
    feature_engineer = FeatureEngineer(session=session)
    features = await feature_engineer.get_match_features(match)

    # Get team names
    home_team = await session.get(Team, match.home_team_id)
    away_team = await session.get(Team, match.away_team_id)

    features["home_team_name"] = home_team.name if home_team else "Unknown"
    features["away_team_name"] = away_team.name if away_team else "Unknown"
    features["odds_home"] = match.odds_home
    features["odds_draw"] = match.odds_draw
    features["odds_away"] = match.odds_away

    import pandas as pd

    df = pd.DataFrame([features])
    predictions = ml_engine.predict(df)

    return predictions[0]


# =============================================================================
# LIVE SUMMARY ENDPOINT (iOS Live Score Polling)
# =============================================================================

# Cache for live summary (L1 cache with 5s TTL per Auditor requirement)
_live_summary_cache: dict = {
    "data": None,
    "timestamp": 0.0,
    "ttl": 5.0,  # 5 second TTL
}

# Live statuses that indicate a match is currently being played
LIVE_STATUSES = frozenset(["1H", "HT", "2H", "ET", "BT", "P", "LIVE", "INT", "SUSP"])


@app.get("/live-summary")
@limiter.limit("60/minute")  # Rate limit: 60 req/min per IP (4 req/15s is comfortable)
async def get_live_summary(
    request: Request,
    session: AsyncSession = Depends(get_async_session),
    _: bool = Depends(verify_api_key),  # Require API key authentication
):
    """
    Ultra-light endpoint for live score polling (iOS LiveScoreManager).

    Returns only LIVE matches with minimal payload (~50 bytes/match).
    Designed for 15s polling interval from iOS clients.

    Response schema (v2 - FASE 1: includes events):
    {
        "ts": 1705500000,  // Unix timestamp of cache
        "matches": {
            "12345": {
                "s": "2H", "e": 67, "ex": 0, "h": 2, "a": 1,
                "ev": [
                    {"m": 23, "t": "Goal", "d": "Normal Goal", "tm": 529, "p": "Messi", "a": "Di Maria"},
                    {"m": 45, "x": 2, "t": "Card", "d": "Yellow Card", "tm": 530, "p": "Martinez"}
                ]
            }
        }
    }

    Fields:
    - s: status (1H, HT, 2H, ET, FT, etc.)
    - e: elapsed minutes
    - ex: elapsed_extra (injury time, e.g., 3 for 90+3)
    - h: home goals
    - a: away goals
    - ev: events array (optional, only if events exist)
      - m: minute
      - x: extra minute (injury time)
      - t: type (Goal, Card)
      - d: detail (Normal Goal, Yellow Card, Red Card, Penalty, Own Goal, etc.)
      - tm: team_id
      - p: player name
      - a: assist name (goals only)

    Auth: Requires X-API-Key header.
    Rate limit: 60 requests/minute per IP.
    """
    from app.telemetry.metrics import record_live_summary_request

    start_time = time.time()
    now = time.time()

    try:
        # Check L1 cache (5s TTL)
        if (
            _live_summary_cache["data"] is not None
            and now - _live_summary_cache["timestamp"] < _live_summary_cache["ttl"]
        ):
            # Cache hit - return immediately
            cached_data = _live_summary_cache["data"]
            latency_ms = (time.time() - start_time) * 1000
            record_live_summary_request(
                status="ok",
                latency_ms=latency_ms,
                matches_count=len(cached_data.get("matches", {})),
            )
            return cached_data

        # Cache miss - query DB (FASE 1: now includes events column)
        query = text("""
            SELECT id, status, elapsed, elapsed_extra, home_goals, away_goals, events
            FROM matches
            WHERE status IN ('1H', 'HT', '2H', 'ET', 'BT', 'P', 'LIVE', 'INT', 'SUSP')
            LIMIT 50
        """)

        result = await session.execute(query)
        rows = result.fetchall()

        # Build compact response (keyed by internal match_id per Auditor requirement)
        # FASE 1: now includes events (ev) when available
        matches_dict = {}
        for row in rows:
            match_id = row[0]
            match_data = {
                "s": row[1],  # status
                "e": row[2] or 0,  # elapsed
                "ex": row[3] or 0,  # elapsed_extra
                "h": row[4] or 0,  # home_goals
                "a": row[5] or 0,  # away_goals
            }
            # FASE 1: Convert FULL schema events to COMPACT format for iOS
            # DB stores: {type, detail, minute, extra_minute, team_id, team_name, player_name, assist_name}
            # iOS expects: {m, x, t, d, tm, p, a}
            events = row[6]
            if events:
                # events is already JSON from DB, parse if string
                if isinstance(events, str):
                    try:
                        events = json.loads(events)
                    except json.JSONDecodeError:
                        events = None
                if events:
                    # Convert to compact format (only Goal and Card for iOS timeline)
                    compact_events = []
                    for ev in events:
                        ev_type = ev.get("type")
                        if ev_type not in ("Goal", "Card"):
                            continue
                        compact_events.append({
                            "m": ev.get("minute"),
                            "x": ev.get("extra_minute"),
                            "t": ev_type,
                            "d": ev.get("detail"),
                            "tm": ev.get("team_id"),
                            "p": ev.get("player_name"),
                            "a": ev.get("assist_name"),
                        })
                    if compact_events:
                        match_data["ev"] = compact_events
            matches_dict[match_id] = match_data

        response_data = {
            "ts": int(now),
            "matches": matches_dict,
        }

        # Update L1 cache
        _live_summary_cache["data"] = response_data
        _live_summary_cache["timestamp"] = now

        latency_ms = (time.time() - start_time) * 1000
        record_live_summary_request(
            status="ok",
            latency_ms=latency_ms,
            matches_count=len(matches_dict),
        )

        logger.debug(f"[live-summary] Returned {len(matches_dict)} live matches in {latency_ms:.1f}ms")

        return response_data

    except Exception as e:
        latency_ms = (time.time() - start_time) * 1000
        record_live_summary_request(status="error", latency_ms=latency_ms, matches_count=0)
        logger.error(f"[live-summary] Error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/teams")
async def list_teams(
    team_type: Optional[str] = None,
    limit: int = 100,
    session: AsyncSession = Depends(get_async_session),
    _: bool = Depends(verify_api_key),
):
    """List teams in the database."""
    query = select(Team)
    if team_type:
        query = query.where(Team.team_type == team_type)
    query = query.limit(limit)

    result = await session.execute(query)
    teams = result.scalars().all()

    return [
        {
            "id": t.id,
            "external_id": t.external_id,
            "name": t.name,
            "country": t.country,
            "team_type": t.team_type,
            "logo_url": t.logo_url,
        }
        for t in teams
    ]


@app.get("/matches")
async def list_matches(
    league_id: Optional[int] = None,
    status: Optional[str] = None,
    limit: int = 50,
    session: AsyncSession = Depends(get_async_session),
    _: bool = Depends(verify_api_key),
):
    """List matches in the database with eager loading to avoid N+1 queries."""
    from sqlalchemy.orm import selectinload

    query = (
        select(Match)
        .options(
            selectinload(Match.home_team),
            selectinload(Match.away_team),
        )
        .order_by(Match.date.desc())
    )

    if league_id:
        query = query.where(Match.league_id == league_id)
    if status:
        query = query.where(Match.status == status)

    query = query.limit(limit)

    result = await session.execute(query)
    matches = result.scalars().all()

    # Build response using eager-loaded relationships
    return [
        {
            "id": m.id,
            "external_id": m.external_id,
            "date": m.date,
            "league_id": m.league_id,
            "home_team": m.home_team.name if m.home_team else "Unknown",
            "away_team": m.away_team.name if m.away_team else "Unknown",
            "home_goals": m.home_goals,
            "away_goals": m.away_goals,
            "status": m.status,
            "match_type": m.match_type,
        }
        for m in matches
    ]


@app.get("/competitions")
async def list_competitions(
    _: bool = Depends(verify_api_key),
):
    """List available competitions."""
    return [
        {
            "league_id": comp.league_id,
            "name": comp.name,
            "match_type": comp.match_type,
            "priority": comp.priority.value,
            "match_weight": comp.match_weight,
        }
        for comp in COMPETITIONS.values()
    ]


@app.get("/model/info")
async def model_info():
    """Get information about the current model."""
    if not ml_engine.is_loaded:
        return {
            "loaded": False,
            "message": "No model loaded. Train one with POST /model/train",
        }

    return {
        "loaded": True,
        "version": ml_engine.model_version,
        "features": ml_engine.FEATURE_COLUMNS,
    }


@app.get("/model/shadow-report")
@limiter.limit("30/minute")
async def get_shadow_report_endpoint(
    request: Request,
    session: AsyncSession = Depends(get_async_session),
    _: bool = Depends(verify_api_key_or_ops_session),
):
    """
    Get shadow model A/B comparison report.

    Returns accuracy and Brier score comparison between baseline and shadow (two-stage) model.
    Includes per-outcome breakdown and GO/NO-GO recommendation.
    Requires API key authentication.
    """
    from app.ml.shadow import is_shadow_enabled, get_shadow_report

    if not is_shadow_enabled():
        return {
            "status": "disabled",
            "message": "Shadow mode not enabled. Set MODEL_SHADOW_ARCHITECTURE=two_stage to enable.",
        }

    report = await get_shadow_report(session)
    return report


@app.get("/model/sensor-report")
@limiter.limit("30/minute")
async def get_sensor_report_endpoint(
    request: Request,
    session: AsyncSession = Depends(get_async_session),
    _: bool = Depends(verify_api_key_or_ops_session),
):
    """
    Get Sensor B (LogReg L2) calibration diagnostics report.

    Returns Model A vs Model B comparison with Brier scores, accuracy,
    signal score, and window analysis. INTERNAL USE ONLY - does not affect production.
    Requires API key authentication.
    """
    from app.ml.sensor import get_sensor_report
    from app.config import get_settings

    sensor_settings = get_settings()
    if not sensor_settings.SENSOR_ENABLED:
        return {
            "status": "disabled",
            "message": "Sensor B not enabled. Set SENSOR_ENABLED=true to enable.",
        }

    report = await get_sensor_report(session)
    return report


@app.post("/odds/refresh")
async def refresh_odds(
    session: AsyncSession = Depends(get_async_session),
):
    """
    Refresh odds for all upcoming matches.

    Fetches latest pre-match odds from API-Football for matches with status 'NS'.
    Prioritizes Bet365, Pinnacle for reliable odds.
    """
    # Get all upcoming matches
    query = select(Match).where(Match.status == "NS")
    result = await session.execute(query)
    matches = result.scalars().all()

    if not matches:
        return {"message": "No upcoming matches found", "updated": 0}

    provider = APIFootballProvider()
    updated_count = 0
    errors = []

    try:
        for match in matches:
            try:
                odds = await provider.get_odds(match.external_id)
                if odds:
                    match.odds_home = odds.get("odds_home")
                    match.odds_draw = odds.get("odds_draw")
                    match.odds_away = odds.get("odds_away")
                    updated_count += 1
                    logger.info(f"Updated odds for match {match.id}: H={match.odds_home}, D={match.odds_draw}, A={match.odds_away}")
            except Exception as e:
                errors.append({"match_id": match.id, "error": str(e)})
                logger.error(f"Error fetching odds for match {match.id}: {e}")

        await session.commit()

    finally:
        await provider.close()

    return {
        "message": f"Odds refresh complete",
        "total_matches": len(matches),
        "updated": updated_count,
        "errors": errors if errors else None,
    }


@app.get("/teams/{team_id}/history")
async def get_team_history(
    team_id: int,
    limit: int = 5,
    session: AsyncSession = Depends(get_async_session),
    _: bool = Depends(verify_api_key),
):
    """
    Get recent match history for a team.

    Returns the last N matches played by the team with results.
    Uses eager loading to avoid N+1 queries.
    """
    from sqlalchemy import or_
    from sqlalchemy.orm import selectinload

    # Get team info
    team = await session.get(Team, team_id)
    if not team:
        raise HTTPException(status_code=404, detail="Team not found")

    # Get last matches with eager loading of both teams (avoids N+1 queries)
    query = (
        select(Match)
        .where(
            or_(
                Match.home_team_id == team_id,
                Match.away_team_id == team_id,
            ),
            Match.status == "FT",  # Only finished matches
        )
        .options(
            selectinload(Match.home_team),
            selectinload(Match.away_team),
        )
        .order_by(Match.date.desc())
        .limit(limit)
    )

    result = await session.execute(query)
    matches = result.scalars().all()

    history = []
    for match in matches:
        # Get opponent from eager-loaded relationship
        if match.home_team_id == team_id:
            opponent = match.away_team
            team_goals = match.home_goals
            opponent_goals = match.away_goals
            is_home = True
        else:
            opponent = match.home_team
            team_goals = match.away_goals
            opponent_goals = match.home_goals
            is_home = False

        # Determine result
        if team_goals > opponent_goals:
            result_str = "W"
        elif team_goals < opponent_goals:
            result_str = "L"
        else:
            result_str = "D"

        history.append({
            "match_id": match.id,
            "date": match.date.isoformat() if match.date else None,
            "opponent": opponent.name if opponent else "Unknown",
            "opponent_logo": opponent.logo_url if opponent else None,
            "is_home": is_home,
            "team_goals": team_goals,
            "opponent_goals": opponent_goals,
            "result": result_str,
            "league_id": match.league_id,
        })

    return {
        "team_id": team_id,
        "team_name": team.name,
        "team_logo": team.logo_url,
        "matches": history,
    }


@app.get("/matches/{match_id}/details")
async def get_match_details(
    match_id: int,
    session: AsyncSession = Depends(get_async_session),
    _: bool = Depends(verify_api_key),
):
    """
    Get full match details including both teams' recent history and standings.

    Returns match info, prediction, standings positions, and last 5 matches for each team.
    """
    import time
    import asyncio
    _t_start = time.time()
    _timings = {}

    # Get match
    _t0 = time.time()
    match = await session.get(Match, match_id)
    _timings["get_match"] = int((time.time() - _t0) * 1000)
    if not match:
        raise HTTPException(status_code=404, detail="Match not found")

    # Get teams (parallel)
    _t0 = time.time()
    home_team, away_team = await asyncio.gather(
        session.get(Team, match.home_team_id),
        session.get(Team, match.away_team_id),
    )
    _timings["get_teams"] = int((time.time() - _t0) * 1000)

    # Get display_names for short name toggle (COALESCE: override > wikidata > name)
    home_display_name = home_team.name if home_team else "Unknown"
    away_display_name = away_team.name if away_team else "Unknown"
    team_ids = [t.id for t in [home_team, away_team] if t]
    if team_ids:
        display_result = await session.execute(
            text("""
                SELECT
                    t.id AS team_id,
                    COALESCE(teo.short_name, twe.short_name, t.name) AS display_name
                FROM teams t
                LEFT JOIN team_enrichment_overrides teo ON t.id = teo.team_id
                LEFT JOIN team_wikidata_enrichment twe ON t.id = twe.team_id
                WHERE t.id = ANY(:team_ids)
            """),
            {"team_ids": team_ids}
        )
        display_map = {row.team_id: row.display_name for row in display_result.fetchall()}
        if home_team and home_team.id in display_map:
            home_display_name = display_map[home_team.id]
        if away_team and away_team.id in display_map:
            away_display_name = display_map[away_team.id]

    # Determine season for standings lookup
    current_date = match.date or datetime.now()
    season = _season_for_league(match.league_id, current_date)

    # NON-BLOCKING standings: L1 cache -> DB -> skip (never call external API in hot path)
    # This ensures endpoint always responds <400ms regardless of league
    _t0 = time.time()
    standings = None
    standings_status = "skipped"  # skipped | cache_hit | db_hit | miss
    standings_source = None
    if match.league_id:
        # L1: memory cache (check truthiness - empty list means no data)
        standings = _get_cached_standings(match.league_id, season)
        if standings:
            standings_status = "cache_hit"
            standings_source = "cache"
            _incr("standings_source_cache")
        else:
            # L2: database
            standings = await _get_standings_from_db(session, match.league_id, season)
            if standings:
                standings_status = "db_hit"
                standings_source = "db"
                _incr("standings_source_db")
                # Populate L1 cache for next request
                _set_cached_standings(match.league_id, season, standings)
            else:
                # L3: Try calculated standings from FT results first
                standings = await _calculate_standings_from_results(session, match.league_id, season)
                if standings:
                    standings_status = "calculated"
                    standings_source = "calculated"
                    _incr("standings_source_calculated")
                    _set_cached_standings(match.league_id, season, standings)
                else:
                    # L4: Generate placeholder standings (zero stats, alphabetical order)
                    standings = await _generate_placeholder_standings(session, match.league_id, season)
                    if standings:
                        standings_status = "placeholder"
                        standings_source = "placeholder"
                        _incr("standings_source_placeholder")
                        _set_cached_standings(match.league_id, season, standings)
                    else:
                        standings_status = "miss"
                        _incr("standings_source_miss")
    _timings["get_standings"] = int((time.time() - _t0) * 1000)
    _timings["standings_status"] = standings_status
    if standings_source:
        _timings["standings_source"] = standings_source

    # Get history for both teams (parallel)
    _t0 = time.time()
    home_history, away_history = await asyncio.gather(
        get_team_history(match.home_team_id, limit=5, session=session),
        get_team_history(match.away_team_id, limit=5, session=session),
    )
    _timings["get_history"] = int((time.time() - _t0) * 1000)

    # Extract standings positions (only if we have cached data)
    home_position = None
    away_position = None
    home_league_points = None
    away_league_points = None

    # Only use standings for club teams when cache hit
    # Note: standings now use internal team_id (teams.id), not external_id
    # ABE P0: Apply group filtering to avoid duplicates from multi-group standings
    if home_team and home_team.team_type == "club" and standings:
        try:
            # Get rules_json for group selection
            rules_result = await session.execute(
                text("SELECT rules_json FROM admin_leagues WHERE league_id = :lid"),
                {"lid": match.league_id}
            )
            rules_row = rules_result.fetchone()
            rules_json = (
                rules_row.rules_json if rules_row and isinstance(rules_row.rules_json, dict)
                else {}
            )

            # Filter standings to selected group (ABE P0: avoid duplicates)
            view_result = select_standings_view(
                standings=standings,
                rules_json=rules_json,
                requested_group=None,  # Use heuristic, no override
            )
            filtered_standings = view_result.standings

            for standing in filtered_standings:
                if home_team and standing.get("team_id") == home_team.id:
                    home_position = standing.get("position")
                    home_league_points = standing.get("points")
                if away_team and standing.get("team_id") == away_team.id:
                    away_position = standing.get("position")
                    away_league_points = standing.get("points")
        except Exception as e:
            logger.warning(f"Could not process standings: {e}")

    # Get prediction if model is loaded and match not played
    prediction = None
    if ml_engine.is_loaded and match.status == "NS":
        try:
            _t0 = time.time()
            feature_engineer = FeatureEngineer(session=session)
            features = await feature_engineer.get_match_features(match)
            _timings["get_features"] = int((time.time() - _t0) * 1000)

            features["home_team_name"] = home_team.name if home_team else "Unknown"
            features["away_team_name"] = away_team.name if away_team else "Unknown"

            _t0 = time.time()
            import pandas as pd
            df = pd.DataFrame([features])
            predictions = ml_engine.predict(df)
            prediction = predictions[0] if predictions else None
            _timings["ml_predict"] = int((time.time() - _t0) * 1000)
        except Exception as e:
            logger.error(f"Error getting prediction: {e}")

    _timings["total"] = int((time.time() - _t_start) * 1000)
    logger.info(f"[PERF] match_details match_id={match_id} timings={_timings}")

    # Resolve team display names/logos (handles rebranding like La Equidad → Internacional de Bogotá)
    home_name = home_team.name if home_team else "Unknown"
    home_logo = home_team.logo_url if home_team else None
    away_name = away_team.name if away_team else "Unknown"
    away_logo = away_team.logo_url if away_team else None

    # Apply team overrides if match date is after effective_from
    if match.date and (home_team or away_team):
        external_ids = []
        if home_team and home_team.external_id:
            external_ids.append(home_team.external_id)
        if away_team and away_team.external_id:
            external_ids.append(away_team.external_id)

        if external_ids:
            overrides = await preload_team_overrides(session, external_ids)
            if overrides:
                if home_team and home_team.external_id:
                    home_display = resolve_team_display(
                        overrides, home_team.external_id, match.date, home_name, home_logo
                    )
                    if home_display.is_override:
                        home_name = home_display.name
                        home_logo = home_display.logo_url or home_logo

                if away_team and away_team.external_id:
                    away_display = resolve_team_display(
                        overrides, away_team.external_id, match.date, away_name, away_logo
                    )
                    if away_display.is_override:
                        away_name = away_display.name
                        away_logo = away_display.logo_url or away_logo

    return {
        "match": {
            "id": match.id,
            "date": match.date.isoformat() if match.date else None,
            "league_id": match.league_id,
            "status": match.status,
            "home_goals": match.home_goals,
            "away_goals": match.away_goals,
            "venue": {
                "name": match.venue_name,
                "city": match.venue_city,
            } if match.venue_name else None,
        },
        "home_team": {
            "id": home_team.external_id if home_team else None,
            "name": home_name,
            "display_name": home_display_name,
            "logo": home_logo,
            "history": home_history["matches"],
            "position": home_position,
            "league_points": home_league_points,
        },
        "away_team": {
            "id": away_team.external_id if away_team else None,
            "name": away_name,
            "display_name": away_display_name,
            "logo": away_logo,
            "history": away_history["matches"],
            "position": away_position,
            "league_points": away_league_points,
        },
        "prediction": prediction,
        "standings_status": standings_status,  # hit | miss | skipped
    }


@app.get("/matches/{match_id}/insights")
async def get_match_insights(
    match_id: int,
    session: AsyncSession = Depends(get_async_session),
    _: bool = Depends(verify_api_key),
):
    """
    Get narrative insights for a finished match.

    Returns human-readable explanations of why the prediction succeeded or failed,
    including analysis of efficiency, clinical finishing, goalkeeper heroics, etc.

    Only available for matches that have been audited (finished + processed).
    """
    # Get match
    match = await session.get(Match, match_id)
    if not match:
        raise HTTPException(status_code=404, detail="Match not found")

    # Only for finished matches
    if match.status not in ("FT", "AET", "PEN"):
        raise HTTPException(
            status_code=400,
            detail=f"Insights only available for finished matches. Status: {match.status}"
        )

    # Get prediction outcome and audit for this match (canonical path)
    result = await session.execute(
        select(PredictionOutcome, PostMatchAudit)
        .join(PostMatchAudit, PredictionOutcome.id == PostMatchAudit.outcome_id)
        .where(PredictionOutcome.match_id == match_id)
    )
    row = result.first()

    if not row:
        # Fallback: generate narrative insights on-demand (non-canonical) so iOS can display something
        # even if the daily audit has not run yet or failed.
        #
        # Important: Keep response shape identical to MatchInsightsResponse in iOS (no optionals).
        from app.audit.service import PostMatchAuditService

        # Prefer a frozen prediction if available; otherwise use latest prediction.
        pred = None
        pred_res = await session.execute(
            select(Prediction)
            .where(Prediction.match_id == match_id)
            .where(Prediction.is_frozen == True)  # noqa: E712
            .order_by(Prediction.frozen_at.desc().nullslast(), Prediction.created_at.desc())
            .limit(1)
        )
        pred = pred_res.scalar_one_or_none()

        if pred is None:
            pred_res = await session.execute(
                select(Prediction)
                .where(Prediction.match_id == match_id)
                .order_by(Prediction.created_at.desc())
                .limit(1)
            )
            pred = pred_res.scalar_one_or_none()

        # If no saved prediction exists, compute one from features (best-effort).
        if pred is None:
            try:
                # Load team names for context
                home_team = await session.get(Team, match.home_team_id)
                away_team = await session.get(Team, match.away_team_id)
                feature_engineer = FeatureEngineer(session=session)
                features = await feature_engineer.get_match_features(match)
                features["home_team_name"] = home_team.name if home_team else "Local"
                features["away_team_name"] = away_team.name if away_team else "Visitante"

                import pandas as pd

                df = pd.DataFrame([features])
                preds = ml_engine.predict(df)
                p0 = preds[0] if preds else None
                probs = (p0 or {}).get("probabilities") or {}
                hp = float(probs.get("home") or 0.0)
                dp = float(probs.get("draw") or 0.0)
                ap = float(probs.get("away") or 0.0)
                pred = Prediction(
                    match_id=match_id,
                    model_version=ml_engine.model_version,
                    home_prob=hp,
                    draw_prob=dp,
                    away_prob=ap,
                )
            except Exception:
                # No prediction available; return empty insights but keep schema stable.
                pred = Prediction(
                    match_id=match_id,
                    model_version=ml_engine.model_version,
                    home_prob=0.0,
                    draw_prob=0.0,
                    away_prob=0.0,
                )

        service = PostMatchAuditService(session)
        try:
            predicted_result, confidence = service._get_predicted_result(pred)
        except Exception:
            predicted_result, confidence = ("draw", 0.0)

        actual_result = "draw"
        if match.home_goals is not None and match.away_goals is not None:
            if match.home_goals > match.away_goals:
                actual_result = "home"
            elif match.home_goals < match.away_goals:
                actual_result = "away"

        prediction_correct = predicted_result == actual_result

        home_team = await session.get(Team, match.home_team_id)
        away_team = await session.get(Team, match.away_team_id)

        narrative_result = service.generate_narrative_insights(
            prediction=pred,
            actual_result=actual_result,
            home_goals=match.home_goals or 0,
            away_goals=match.away_goals or 0,
            stats=match.stats or {},
            home_team_name=home_team.name if home_team else "Local",
            away_team_name=away_team.name if away_team else "Visitante",
            home_position=None,
            away_position=None,
        )

        await service.close()

        fallback_response = {
            "match_id": match_id,
            "prediction_correct": prediction_correct,
            "predicted_result": predicted_result,
            "actual_result": actual_result,
            "confidence": confidence,
            "deviation_type": "pending_audit",
            "insights": narrative_result.get("insights") or [],
            "momentum_analysis": narrative_result.get("momentum_analysis"),
            # No LLM narrative available - indicates match had no pre-match prediction
            "llm_narrative_status": "no_prediction",
        }
        # Include match stats for UI stats table
        if match.stats:
            fallback_response["match_stats"] = match.stats
        if match.events:
            fallback_response["match_events"] = match.events
        return fallback_response

    outcome, audit = row

    response = {
        "match_id": match_id,
        "prediction_correct": outcome.prediction_correct,
        "predicted_result": outcome.predicted_result,
        "actual_result": outcome.actual_result,
        "confidence": outcome.confidence,
        "deviation_type": audit.deviation_type,
        "insights": audit.narrative_insights or [],
        "momentum_analysis": audit.momentum_analysis,
    }

    # Include LLM narrative if available
    if audit.llm_narrative_status == "ok" and audit.llm_narrative_json:
        response["llm_narrative"] = audit.llm_narrative_json
        response["llm_narrative_status"] = "ok"
    elif audit.llm_narrative_status:
        response["llm_narrative_status"] = audit.llm_narrative_status

    # Include match stats for UI stats table (renders independently of narrative)
    if match.stats:
        response["match_stats"] = match.stats

    # Include events for UI
    if match.events:
        response["match_events"] = match.events

    return response


@app.get("/matches/{match_id}/timeline")
async def get_match_timeline(
    match_id: int,
    session: AsyncSession = Depends(get_async_session),
    _: bool = Depends(verify_api_key),
):
    """
    Get timeline data for a finished match.

    Returns goal events with minutes, and compares against our prediction
    to show when the prediction was "in line" vs "out of line" with the score.

    Only available for finished matches (FT, AET, PEN) with a saved prediction.
    """
    # Get match
    match = await session.get(Match, match_id)
    if not match:
        raise HTTPException(status_code=404, detail="Match not found")

    # Only for finished matches
    if match.status not in ("FT", "AET", "PEN"):
        raise HTTPException(
            status_code=400,
            detail=f"Timeline only available for finished matches. Status: {match.status}"
        )

    # Get saved prediction for this match
    # Use the FIRST frozen prediction (original baseline model)
    # The two_stage shadow model was added later for A/B testing but shouldn't
    # replace the original prediction for evaluation purposes
    prediction_source = "frozen_original"
    result = await session.execute(
        select(Prediction)
        .where(Prediction.match_id == match_id)
        .where(Prediction.is_frozen == True)
        .order_by(Prediction.created_at.asc())  # First/original prediction
        .limit(1)
    )
    prediction = result.scalar_one_or_none()

    # Last fallback: any prediction (mark as low confidence)
    if not prediction:
        prediction_source = "unfrozen_fallback"
        result = await session.execute(
            select(Prediction)
            .where(Prediction.match_id == match_id)
            .order_by(Prediction.created_at.asc())
            .limit(1)
        )
        prediction = result.scalar_one_or_none()

    if not prediction:
        raise HTTPException(
            status_code=404,
            detail="No prediction saved for this match"
        )

    # Determine what we predicted
    predicted_outcome = "home"
    if prediction.away_prob > prediction.home_prob and prediction.away_prob > prediction.draw_prob:
        predicted_outcome = "away"
    elif prediction.draw_prob > prediction.home_prob and prediction.draw_prob > prediction.away_prob:
        predicted_outcome = "draw"

    # Get goal events - prefer DB, fallback to API
    import time
    _t0 = time.time()
    events = []
    events_source = "none"

    # Try DB first (for finished matches, events should be cached)
    if match.events and len(match.events) > 0:
        events = match.events
        events_source = "db"
        _incr("timeline_source_db")
        logger.info(f"[PERF] timeline match_id={match_id} events_source=db count={len(events)} time_ms={int((time.time() - _t0) * 1000)}")
    else:
        # Fallback to API (and persist for next time)
        _incr("timeline_source_api_fallback")
        logger.info(f"[PERF] timeline match_id={match_id} events_source=api_fallback (db events empty)")
        provider = APIFootballProvider()
        try:
            events = await provider.get_fixture_events(match.external_id)
            events_source = "api"
            # Persist to DB for future requests (best-effort)
            if events:
                try:
                    match.events = events
                    await session.commit()
                    logger.info(f"[PERF] timeline match_id={match_id} persisted {len(events)} events to DB")
                except Exception as persist_err:
                    logger.warning(f"[PERF] timeline match_id={match_id} failed to persist events: {persist_err}")
        finally:
            await provider.close()
        logger.info(f"[PERF] timeline match_id={match_id} events_source=api count={len(events)} time_ms={int((time.time() - _t0) * 1000)}")

    # Filter only goals
    goals = [
        e for e in events
        if e.get("type") == "Goal"
    ]

    # Sort by minute
    goals.sort(key=lambda g: (g.get("minute") or 0, g.get("extra_minute") or 0))

    # Get team IDs
    home_team = await session.get(Team, match.home_team_id)
    away_team = await session.get(Team, match.away_team_id)
    home_external_id = home_team.external_id if home_team else None
    away_external_id = away_team.external_id if away_team else None

    # Build timeline segments
    # Each segment: {start_minute, end_minute, home_score, away_score, status}
    # status: "correct" (prediction in line), "neutral" (draw when we predicted win), "wrong" (losing)
    segments = []
    current_home = 0
    current_away = 0
    last_minute = 0

    # Calculate total match duration based on goals (including added time)
    # Default to 90, but extend if there are goals in added time
    total_minutes = 90
    if goals:
        # Consider both base minute and extra time (e.g., 90+3 = 93 effective)
        max_effective = max(
            (g.get("minute") or 0) + (g.get("extra_minute") or 0)
            for g in goals
        )
        # Use the maximum between 90 and the last goal's effective minute
        total_minutes = max(90, max_effective)

    for goal in goals:
        minute = goal.get("minute") or 0
        extra = goal.get("extra_minute") or 0
        effective_minute = minute + (extra * 0.1)  # For sorting 90+1, 90+2, etc.

        # Add segment before this goal
        if minute > last_minute:
            status = _calculate_segment_status(
                current_home, current_away, predicted_outcome
            )
            segments.append({
                "start_minute": last_minute,
                "end_minute": minute,
                "home_goals": current_home,
                "away_goals": current_away,
                "status": status,
            })

        # Determine which team scored (prefer team_id, fallback to team_name match)
        is_home_team = False
        is_away_team = False
        used_legacy_fallback = False

        if goal.get("team_id"):
            is_home_team = goal.get("team_id") == home_external_id
            is_away_team = goal.get("team_id") == away_external_id
        else:
            # Legacy fallback: match by team name
            used_legacy_fallback = True
            goal_team_name = goal.get("team_name") or goal.get("team")
            if goal_team_name:
                is_home_team = goal_team_name == (home_team.name if home_team else None)
                is_away_team = goal_team_name == (away_team.name if away_team else None)
            logger.info(f"[TIMELINE] match_id={match_id} legacy_fallback goal_team_name={goal_team_name} matched={'home' if is_home_team else 'away' if is_away_team else 'none'}")

        # Update score
        if is_home_team:
            if goal.get("detail") == "Own Goal":
                current_away += 1
            else:
                current_home += 1
        elif is_away_team:
            if goal.get("detail") == "Own Goal":
                current_home += 1
            else:
                current_away += 1

        last_minute = minute

    # Add final segment
    if last_minute < total_minutes:
        status = _calculate_segment_status(
            current_home, current_away, predicted_outcome
        )
        segments.append({
            "start_minute": last_minute,
            "end_minute": total_minutes,
            "home_goals": current_home,
            "away_goals": current_away,
            "status": status,
        })

    # Calculate time in correct prediction
    correct_minutes = sum(
        (s["end_minute"] - s["start_minute"]) for s in segments if s["status"] == "correct"
    )
    total_match_minutes = total_minutes
    correct_percentage = (correct_minutes / total_match_minutes) * 100 if total_match_minutes > 0 else 0

    # Determine final result
    final_result = "draw"
    if match.home_goals > match.away_goals:
        final_result = "home"
    elif match.away_goals > match.home_goals:
        final_result = "away"

    return {
        "match_id": match_id,
        "status": match.status,
        "final_score": {
            "home": match.home_goals,
            "away": match.away_goals,
        },
        "prediction": {
            "outcome": predicted_outcome,
            "home_prob": round(prediction.home_prob, 4),
            "draw_prob": round(prediction.draw_prob, 4),
            "away_prob": round(prediction.away_prob, 4),
            "correct": predicted_outcome == final_result,
        },
        "total_minutes": total_minutes,
        "goals": [
            {
                "minute": g.get("minute"),
                "extra_minute": g.get("extra_minute"),
                # Determine team: prefer team_id match, fallback to team_name match for legacy events
                "team": (
                    "home" if g.get("team_id") == home_external_id
                    else "away" if g.get("team_id") == away_external_id
                    else "home" if g.get("team_name") == (home_team.name if home_team else None) or g.get("team") == (home_team.name if home_team else None)
                    else "away"
                ),
                "team_name": g.get("team_name") or g.get("team"),  # Support legacy "team" field
                "player": g.get("player_name") or g.get("player"),  # Support legacy "player" field
                "is_own_goal": g.get("detail") == "Own Goal",
                "is_penalty": g.get("detail") == "Penalty",
            }
            for g in goals
        ],
        "segments": segments,
        "summary": {
            "correct_minutes": round(correct_minutes, 1),
            "correct_percentage": round(correct_percentage, 1),
        },
        "_meta": {
            "events_source": events_source,
            "events_count": len(events),
            "prediction_source": prediction_source,
        },
    }


def _calculate_segment_status(home_score: int, away_score: int, predicted: str) -> str:
    """
    Calculate segment status based on current score vs prediction.

    Returns:
        "correct": Score aligns with prediction
        "neutral": Draw when we predicted a win (gray area)
        "wrong": Losing team is the one we predicted to win
    """
    if home_score == away_score:
        # It's a draw
        if predicted == "draw":
            return "correct"
        else:
            return "neutral"  # We predicted a win but it's tied

    if home_score > away_score:
        # Home is winning
        if predicted == "home":
            return "correct"
        elif predicted == "away":
            return "wrong"
        else:  # predicted draw
            return "neutral"
    else:
        # Away is winning
        if predicted == "away":
            return "correct"
        elif predicted == "home":
            return "wrong"
        else:  # predicted draw
            return "neutral"


@app.get("/matches/{match_id}/odds-history")
async def get_match_odds_history(
    match_id: int,
    session: AsyncSession = Depends(get_async_session),
    _: bool = Depends(verify_api_key),
):
    """
    Get odds history for a match showing how odds changed over time.

    Returns all recorded odds snapshots for the match, ordered by time.
    Useful for:
    - Analyzing line movements before the match
    - Seeing opening vs closing odds
    - Detecting sharp money movements
    """
    # Get match
    match = await session.get(Match, match_id)
    if not match:
        raise HTTPException(status_code=404, detail="Match not found")

    # Get team names
    home_team = await session.get(Team, match.home_team_id)
    away_team = await session.get(Team, match.away_team_id)

    # Get odds history
    result = await session.execute(
        select(OddsHistory)
        .where(OddsHistory.match_id == match_id)
        .order_by(OddsHistory.recorded_at.asc())
    )
    history = result.scalars().all()

    # Calculate line movement if we have opening and current odds
    movement = None
    if len(history) >= 2:
        opening = history[0]
        current = history[-1]
        if opening.odds_home and current.odds_home:
            movement = {
                "home_change": round(current.odds_home - opening.odds_home, 2),
                "draw_change": round((current.odds_draw or 0) - (opening.odds_draw or 0), 2),
                "away_change": round((current.odds_away or 0) - (opening.odds_away or 0), 2),
                "home_pct": round((current.odds_home - opening.odds_home) / opening.odds_home * 100, 1),
                "draw_pct": round(((current.odds_draw or 0) - (opening.odds_draw or 0)) / (opening.odds_draw or 1) * 100, 1) if opening.odds_draw else None,
                "away_pct": round(((current.odds_away or 0) - (opening.odds_away or 0)) / (opening.odds_away or 1) * 100, 1) if opening.odds_away else None,
            }

    return {
        "match_id": match_id,
        "home_team": home_team.name if home_team else "Unknown",
        "away_team": away_team.name if away_team else "Unknown",
        "match_date": match.date.isoformat() if match.date else None,
        "status": match.status,
        "current_odds": {
            "home": match.odds_home,
            "draw": match.odds_draw,
            "away": match.odds_away,
            "recorded_at": match.odds_recorded_at.isoformat() if match.odds_recorded_at else None,
        },
        "history": [
            {
                "recorded_at": h.recorded_at.isoformat(),
                "odds_home": h.odds_home,
                "odds_draw": h.odds_draw,
                "odds_away": h.odds_away,
                "implied_home": round(h.implied_home, 4) if h.implied_home else None,
                "implied_draw": round(h.implied_draw, 4) if h.implied_draw else None,
                "implied_away": round(h.implied_away, 4) if h.implied_away else None,
                "overround": round(h.overround, 4) if h.overround else None,
                "is_opening": h.is_opening,
                "is_closing": h.is_closing,
                "source": h.source,
            }
            for h in history
        ],
        "movement": movement,
        "total_snapshots": len(history),
    }


@app.get("/standings/{league_id}")
async def get_league_standings(
    league_id: int,
    season: int = None,
    group: Optional[str] = None,
    session: AsyncSession = Depends(get_async_session),
    _: bool = Depends(verify_api_key),
):
    """
    Get full league standings/table for a given league.

    DB-first architecture: serves from DB, falls back to provider on miss.
    Returns all teams with position, points, matches played, goals, form, etc.

    Query params:
    - season: Year (default: current season for league)
    - group: Specific group name to filter (default: auto-selected via heuristic)

    Response includes `meta` field with:
    - available_groups: All groups in standings
    - selected_group: Currently shown group
    - selection_reason: Why this group was selected
    - tie_warning: List of groups if TIE detected (requires manual config)
    """
    _t_start = time.time()
    source = None

    try:
        # Determine season if not provided
        if season is None:
            current_date = datetime.now()
            season = _season_for_league(league_id, current_date)

        # L1: Memory cache (check truthiness - empty list means no data)
        # IMPORTANT: deepcopy to avoid mutating cache when applying translations
        cached_standings = _get_cached_standings(league_id, season)
        if cached_standings:
            import copy
            standings = copy.deepcopy(cached_standings)
            source = "cache"
        else:
            # L2: Database
            standings = await _get_standings_from_db(session, league_id, season)
            if standings:
                source = "db"
                # Populate L1 cache
                _set_cached_standings(league_id, season, standings)
            else:
                # L3: Provider fallback (and persist)
                source = "api_fallback"
                provider = APIFootballProvider()
                try:
                    standings = await provider.get_standings(league_id, season)
                    if standings:
                        # Persist to DB
                        await _save_standings_to_db(session, league_id, season, standings)
                        # Populate L1 cache
                        _set_cached_standings(league_id, season, standings)
                finally:
                    await provider.close()

        # L3.5: Calculated standings from FT results (when API has no data yet)
        # Priority: API > calculated > placeholder
        # Guardrails: FT_count >= 2, transparency via is_calculated flag
        if not standings:
            standings = await _calculate_standings_from_results(session, league_id, season)
            if standings:
                source = "calculated"
                # Use shorter TTL for calculated standings (15 min)
                _set_cached_standings(league_id, season, standings)
                logger.info(f"Using calculated standings for league {league_id} season {season}")

        # L4: Placeholder fallback - generate zero-stats standings from known teams
        if not standings:
            standings = await _generate_placeholder_standings(session, league_id, season)
            if standings:
                source = "placeholder"
                # Cache placeholder standings (shorter TTL handled by is_placeholder flag)
                _set_cached_standings(league_id, season, standings)

        if not standings:
            raise HTTPException(
                status_code=404,
                detail=f"Standings not available yet for season {season}. No teams found for this league.",
            )

        # Apply team identity overrides (e.g., La Equidad -> Internacional de Bogotá)
        from app.teams.overrides import apply_team_overrides_to_standings
        standings = await apply_team_overrides_to_standings(
            session, standings, league_id, season
        )

        # Translate external_id (API-Football) to internal id
        # This ensures team_id in response matches teams.id, not teams.external_id
        external_ids = [s.get("team_id") for s in standings if s.get("team_id")]
        if external_ids:
            result = await session.execute(
                select(Team.id, Team.external_id).where(Team.external_id.in_(external_ids))
            )
            ext_to_internal = {row.external_id: row.id for row in result.all()}
            for standing in standings:
                ext_id = standing.get("team_id")
                if ext_id and ext_id in ext_to_internal:
                    standing["team_id"] = ext_to_internal[ext_id]

        # Enrich with display_name for use_short_names toggle
        # (uses internal team_id, so must run after external->internal translation)
        from app.teams.overrides import enrich_standings_with_display_names
        standings = await enrich_standings_with_display_names(session, standings)

        # Get rules_json for standings view selection (ABE P0: DB-first filtering)
        rules_result = await session.execute(
            text("SELECT rules_json FROM admin_leagues WHERE league_id = :lid"),
            {"lid": league_id}
        )
        rules_row = rules_result.fetchone()
        rules_json = (
            rules_row.rules_json if rules_row and isinstance(rules_row.rules_json, dict)
            else {}
        )

        # Apply standings view selection (filter by group)
        try:
            view_result = select_standings_view(
                standings=standings,
                rules_json=rules_json,
                requested_group=group,
            )
        except StandingsGroupNotFound as e:
            # ABE P0: Return 404 with available_groups in body AND header
            raise HTTPException(
                status_code=404,
                detail={
                    "message": f"Group '{e.requested}' not found",
                    "available_groups": e.available,
                },
                headers={"X-Available-Groups": ",".join(e.available)},
            )

        elapsed_ms = int((time.time() - _t_start) * 1000)
        logger.info(
            f"[PERF] get_standings league_id={league_id} season={season} "
            f"source={source} group={view_result.selected_group} time_ms={elapsed_ms}"
        )

        # Determine if standings are placeholder or calculated
        is_placeholder = source == "placeholder" or (
            view_result.standings and view_result.standings[0].get("is_placeholder", False)
        )
        is_calculated = source == "calculated" or (
            view_result.standings and view_result.standings[0].get("is_calculated", False)
        )

        # Phase 2: Apply zones/badges to standings entries
        zones_config = rules_json.get("zones", {})
        if zones_config.get("enabled", False):
            apply_zones(view_result.standings, zones_config)

        # Phase 3: Reclasificación (accumulated Apertura + Clausura)
        reclasificacion = None
        reclasificacion_config = rules_json.get("reclasificacion", {})
        if reclasificacion_config.get("enabled", False):
            try:
                reclasificacion = await _calculate_reclasificacion(
                    session=session,
                    league_id=league_id,
                    season=season,
                )
            except Exception as e:
                logger.error(
                    f"[STANDINGS] Error calculating reclasificacion for league {league_id}: {e}"
                )

        # Phase 4: Descenso por promedio
        descenso = None
        relegation_config = rules_json.get("relegation", {})
        if (
            relegation_config.get("enabled", False)
            and relegation_config.get("method") == "average_3y"
        ):
            try:
                descenso = await _calculate_descenso(
                    session=session,
                    league_id=league_id,
                    season=season,
                    relegation_config=relegation_config,
                    all_standings=standings,
                )
            except Exception as e:
                logger.error(
                    f"[STANDINGS] Error calculating descenso for league {league_id}: {e}"
                )

        # Phase 5: Build available_tables metadata
        # ABE P0-4: Reuse group_standings_by_name for team_count
        all_groups = group_standings_by_name(standings)
        available_tables = []
        for gname in view_result.available_groups:
            gtype = classify_group_type(gname, rules_json)
            # ABE P0-2: If descenso exists, exclude native "descenso" groups (avoid duplication)
            if gtype == "descenso" and descenso:
                continue
            available_tables.append({
                "group": gname,
                "team_count": len(all_groups.get(gname, [])),
                "type": gtype,
                "is_current": gname == view_result.selected_group,
            })
        # Add virtual tables (reclasificación/descenso) if they exist
        if reclasificacion:
            available_tables.append({
                "group": "Reclasificación",
                "team_count": len(reclasificacion.get("data", [])),
                "type": "reclasificacion",
                "is_current": False,
            })
        if descenso:
            available_tables.append({
                "group": "Descenso por Promedio",
                "team_count": len(descenso.get("data", [])),
                "type": "descenso",
                "is_current": False,
            })
        # ABE P1: Stable ordering (regular → group_stage → playoff → virtual)
        _TYPE_ORDER = {"regular": 0, "group_stage": 1, "playoff": 2, "reclasificacion": 3, "descenso": 4}
        available_tables.sort(key=lambda t: _TYPE_ORDER.get(t["type"], 99))

        # ABE P0: Backwards-compatible response with added `meta` field
        return {
            "league_id": league_id,
            "season": season,
            "standings": view_result.standings,
            "source": source,
            "is_placeholder": is_placeholder,
            "is_calculated": is_calculated,
            "meta": {
                "available_groups": view_result.available_groups,
                "available_tables": available_tables,
                "selected_group": view_result.selected_group,
                "selection_reason": view_result.selection_reason,
                "tie_warning": view_result.tie_warning,
                "zones_source": zones_config.get("source") if zones_config.get("enabled", False) else None,
                "is_group_stage": rules_json.get("standings", {}).get("is_group_stage", False),
            },
            "reclasificacion": reclasificacion,
            "descenso": descenso,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching standings: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch standings")


# ============================================================================
# AUDIT ENDPOINTS - Post-match analysis and model evaluation
# ============================================================================


class AuditResponse(BaseModel):
    matches_audited: int
    correct_predictions: int
    accuracy: float
    anomalies_detected: int
    period_days: int


class AuditSummaryResponse(BaseModel):
    total_outcomes: int
    correct_predictions: int
    overall_accuracy: float
    by_tier: dict
    by_deviation_type: dict
    recent_anomalies: list


@app.post("/audit/run", response_model=AuditResponse)
async def run_audit(
    days: int = 7,
    session: AsyncSession = Depends(get_async_session),
):
    """
    Run post-match audit for completed matches.

    Analyzes predictions vs actual results for matches finished in the last N days.
    Fetches xG, events (red cards, penalties, VAR) and classifies deviations.
    """
    from app.audit import create_audit_service

    logger.info(f"Running audit for last {days} days...")

    try:
        audit_service = await create_audit_service(session)
        result = await audit_service.audit_recent_matches(days=days)
        await audit_service.close()

        return AuditResponse(**result)

    except Exception as e:
        logger.error(f"Audit failed: {e}")
        raise HTTPException(status_code=500, detail="Audit failed. Check server logs for details.")


@app.get("/audit/summary", response_model=AuditSummaryResponse)
async def get_audit_summary(
    days: Optional[int] = None,
    session: AsyncSession = Depends(get_async_session),
):
    """
    Get summary of audit results.

    Returns accuracy by confidence tier, deviation distribution, and recent anomalies.
    """
    from sqlalchemy import func

    from app.models import PredictionOutcome, PostMatchAudit

    # Base query
    query = select(PredictionOutcome)

    if days:
        from datetime import timedelta
        cutoff = datetime.utcnow() - timedelta(days=days)
        query = query.where(PredictionOutcome.audited_at >= cutoff)

    result = await session.execute(query)
    outcomes = result.scalars().all()

    if not outcomes:
        return AuditSummaryResponse(
            total_outcomes=0,
            correct_predictions=0,
            overall_accuracy=0.0,
            by_tier={},
            by_deviation_type={},
            recent_anomalies=[],
        )

    # Calculate metrics
    total = len(outcomes)
    correct = sum(1 for o in outcomes if o.prediction_correct)
    overall_accuracy = (correct / total * 100) if total > 0 else 0

    # By tier
    tiers = {}
    for tier in ["gold", "silver", "copper"]:
        tier_outcomes = [o for o in outcomes if o.confidence_tier == tier]
        tier_correct = sum(1 for o in tier_outcomes if o.prediction_correct)
        tier_total = len(tier_outcomes)
        tiers[tier] = {
            "total": tier_total,
            "correct": tier_correct,
            "accuracy": (tier_correct / tier_total * 100) if tier_total > 0 else 0,
        }

    # Get audits for deviation breakdown
    outcome_ids = [o.id for o in outcomes]
    audit_result = await session.execute(
        select(PostMatchAudit).where(PostMatchAudit.outcome_id.in_(outcome_ids))
    )
    audits = audit_result.scalars().all()

    # By deviation type
    deviation_types = {}
    for dtype in ["minimal", "expected", "anomaly"]:
        count = sum(1 for a in audits if a.deviation_type == dtype)
        deviation_types[dtype] = count

    # Recent anomalies
    anomaly_audits = [a for a in audits if a.deviation_type == "anomaly"]
    recent_anomalies = []

    for audit in anomaly_audits[:10]:  # Last 10 anomalies
        outcome = next((o for o in outcomes if o.id == audit.outcome_id), None)
        if outcome:
            # Get match info
            match_result = await session.execute(
                select(Match).where(Match.id == outcome.match_id)
            )
            match = match_result.scalar_one_or_none()

            if match:
                home_team = await session.get(Team, match.home_team_id)
                away_team = await session.get(Team, match.away_team_id)

                recent_anomalies.append({
                    "match_id": match.id,
                    "date": match.date.isoformat() if match.date else None,
                    "home_team": home_team.name if home_team else "Unknown",
                    "away_team": away_team.name if away_team else "Unknown",
                    "score": f"{outcome.actual_home_goals}-{outcome.actual_away_goals}",
                    "predicted": outcome.predicted_result,
                    "actual": outcome.actual_result,
                    "confidence": round(outcome.confidence * 100, 1),
                    "primary_factor": audit.primary_factor,
                    "xg_home": outcome.xg_home,
                    "xg_away": outcome.xg_away,
                })

    return AuditSummaryResponse(
        total_outcomes=total,
        correct_predictions=correct,
        overall_accuracy=round(overall_accuracy, 2),
        by_tier=tiers,
        by_deviation_type=deviation_types,
        recent_anomalies=recent_anomalies,
    )


# ============================================================================
# RECALIBRATION ENDPOINTS - Model auto-adjustment and team confidence
# ============================================================================


class RecalibrationStatusResponse(BaseModel):
    current_model_version: str
    baseline_brier_score: float
    current_brier_score: Optional[float]
    last_retrain_date: Optional[str]
    gold_accuracy_current: float
    gold_accuracy_threshold: float
    retrain_needed: bool
    retrain_reason: str
    teams_with_adjustments: int


class TeamAdjustmentResponse(BaseModel):
    team_id: int
    team_name: str
    confidence_multiplier: float
    total_predictions: int
    correct_predictions: int
    anomaly_count: int
    avg_deviation_score: float
    last_updated: str
    reason: Optional[str]


class ModelSnapshotResponse(BaseModel):
    id: int
    model_version: str
    model_path: str
    brier_score: float
    samples_trained: int
    is_active: bool
    is_baseline: bool
    created_at: str


@app.get("/recalibration/status", response_model=RecalibrationStatusResponse)
async def get_recalibration_status(
    session: AsyncSession = Depends(get_async_session),
):
    """
    Get current recalibration status.

    Returns model health metrics, thresholds, and whether retraining is needed.
    """
    from app.ml.recalibration import RecalibrationEngine

    try:
        recalibrator = RecalibrationEngine(session)
        status = await recalibrator.get_recalibration_status()
        return RecalibrationStatusResponse(**status)
    except Exception as e:
        logger.error(f"Error getting recalibration status: {e}")
        raise HTTPException(status_code=500, detail="Failed to get recalibration status")


@app.get("/recalibration/team-adjustments", response_model=list[TeamAdjustmentResponse])
async def get_team_adjustments(
    session: AsyncSession = Depends(get_async_session),
):
    """
    Get all teams with confidence adjustments.

    Returns teams whose predictions are being adjusted due to high anomaly rates.
    """
    from app.ml.recalibration import RecalibrationEngine

    try:
        recalibrator = RecalibrationEngine(session)
        adjustments = await recalibrator.get_team_adjustments()
        return [TeamAdjustmentResponse(**adj) for adj in adjustments]
    except Exception as e:
        logger.error(f"Error getting team adjustments: {e}")
        raise HTTPException(status_code=500, detail="Failed to get team adjustments")


@app.post("/recalibration/calculate-adjustments")
async def calculate_team_adjustments(
    days: int = 30,
    session: AsyncSession = Depends(get_async_session),
):
    """
    Manually trigger team adjustment calculation.

    Analyzes recent prediction outcomes and updates confidence multipliers.
    """
    from app.ml.recalibration import RecalibrationEngine

    try:
        recalibrator = RecalibrationEngine(session)
        result = await recalibrator.calculate_team_adjustments(days=days)
        return result
    except Exception as e:
        logger.error(f"Error calculating adjustments: {e}")
        raise HTTPException(status_code=500, detail="Failed to calculate adjustments")


@app.get("/recalibration/league-drift")
async def get_league_drift(
    session: AsyncSession = Depends(get_async_session),
):
    """
    Detect league-level accuracy drift.

    Compares weekly GOLD accuracy per league against historical baseline.
    Leagues with 15%+ accuracy drop are marked as 'Unstable'.

    Use this to identify structural changes in specific leagues.
    """
    from app.ml.recalibration import RecalibrationEngine

    try:
        recalibrator = RecalibrationEngine(session)
        result = await recalibrator.detect_league_drift()
        return result
    except Exception as e:
        logger.error(f"Error detecting league drift: {e}")
        raise HTTPException(status_code=500, detail="Failed to detect league drift")


@app.get("/recalibration/odds-movement")
async def get_odds_movements(
    days_ahead: int = 3,
    session: AsyncSession = Depends(get_async_session),
):
    """
    Check for significant market odds movements.

    Compares current market odds with our fair odds at prediction time.
    Movement of 25%+ triggers tier degradation warning.

    Returns matches with unusual market activity that may indicate
    information we don't have (injuries, lineup changes, etc).
    """
    from app.ml.recalibration import RecalibrationEngine

    try:
        recalibrator = RecalibrationEngine(session)
        result = await recalibrator.check_all_upcoming_odds_movements(days_ahead=days_ahead)
        return result
    except Exception as e:
        logger.error(f"Error checking odds movements: {e}")
        raise HTTPException(status_code=500, detail="Failed to check odds movements")


@app.get("/recalibration/odds-movement/{match_id}")
async def get_match_odds_movement(
    match_id: int,
    session: AsyncSession = Depends(get_async_session),
):
    """
    Check odds movement for a specific match.

    Returns detailed analysis of market movement and tier degradation recommendation.
    """
    from app.ml.recalibration import RecalibrationEngine

    try:
        recalibrator = RecalibrationEngine(session)
        result = await recalibrator.check_odds_movement(match_id)
        return result
    except Exception as e:
        logger.error(f"Error checking match odds movement: {e}")
        raise HTTPException(status_code=500, detail="Failed to check match odds movement")


@app.get("/recalibration/lineup/{match_external_id}")
async def check_match_lineup(
    match_external_id: int,
    session: AsyncSession = Depends(get_async_session),
):
    """
    Check lineup validation for a specific match (Fase 3).

    Fetches announced lineup from API-Football (available ~60min before kickoff)
    and compares with expected best XI.

    Returns:
    - available: Whether lineups are announced
    - lineup_data: Formation and starters count for each team
    - tier_degradation: Recommended tier reduction (0, 1, or 2)
    - warnings: List of warnings (LINEUP_ROTATION_HOME, LINEUP_ROTATION_SEVERE_AWAY, etc.)
    - insights: Human-readable rotation analysis

    Variance thresholds:
    - 30%+ rotation = 1 tier degradation
    - 50%+ rotation = 2 tier degradation (severe)
    """
    from app.ml.recalibration import RecalibrationEngine

    try:
        recalibrator = RecalibrationEngine(session)
        result = await recalibrator.check_lineup_for_match(match_external_id)
        return result
    except Exception as e:
        logger.error(f"Error checking match lineup: {e}")
        raise HTTPException(status_code=500, detail="Failed to check match lineup")


@app.get("/matches/{match_id}/lineup")
async def get_match_lineup(
    match_id: int,
    session: AsyncSession = Depends(get_async_session),
    _: bool = Depends(verify_api_key),
):
    """
    Get full lineup information for a match.

    Fetches starting XI and substitutes for both teams.
    Available approximately 60 minutes before kickoff.
    """
    from app.etl.api_football import APIFootballProvider

    # Get match to find external ID
    match = await session.get(Match, match_id)
    if not match:
        raise HTTPException(status_code=404, detail="Match not found")

    provider = APIFootballProvider()
    try:
        lineup_data = await provider.get_lineups(match.external_id)

        if not lineup_data:
            return {
                "available": False,
                "match_id": match_id,
                "external_id": match.external_id,
                "message": "Lineups not yet announced (typically available ~60min before kickoff)",
            }

        return {
            "available": True,
            "match_id": match_id,
            "external_id": match.external_id,
            "home": lineup_data.get("home"),
            "away": lineup_data.get("away"),
        }
    finally:
        await provider.close()


@app.get("/recalibration/snapshots", response_model=list[ModelSnapshotResponse])
async def get_model_snapshots(
    session: AsyncSession = Depends(get_async_session),
):
    """
    Get all model snapshots.

    Returns history of model versions for rollback capability.
    """
    from app.models import ModelSnapshot

    try:
        query = select(ModelSnapshot).order_by(ModelSnapshot.created_at.desc())
        result = await session.execute(query)
        snapshots = result.scalars().all()

        return [
            ModelSnapshotResponse(
                id=s.id,
                model_version=s.model_version,
                model_path=s.model_path,
                brier_score=s.brier_score,
                samples_trained=s.samples_trained,
                is_active=s.is_active,
                is_baseline=s.is_baseline,
                created_at=s.created_at.isoformat(),
            )
            for s in snapshots
        ]
    except Exception as e:
        logger.error(f"Error getting snapshots: {e}")
        raise HTTPException(status_code=500, detail="Failed to get model snapshots")


@app.post("/recalibration/create-baseline")
async def create_baseline_snapshot(
    brier_score: float = 0.2063,
    session: AsyncSession = Depends(get_async_session),
):
    """
    Create a baseline snapshot for the current model.

    This sets the reference point for model validation.
    New models must beat this Brier score to be deployed.
    """
    from app.ml.recalibration import RecalibrationEngine
    from app.config import get_settings
    from pathlib import Path

    settings = get_settings()

    try:
        recalibrator = RecalibrationEngine(session)

        # Find current model file
        model_path = Path(settings.MODEL_PATH)
        model_files = list(model_path.glob("xgb_*.json"))

        if not model_files:
            raise HTTPException(status_code=404, detail="No model files found")

        latest_model = max(model_files, key=lambda p: p.stat().st_mtime)

        snapshot = await recalibrator.create_snapshot(
            model_version=settings.MODEL_VERSION,
            model_path=str(latest_model),
            brier_score=brier_score,
            cv_scores=[brier_score],  # Single value for baseline
            samples_trained=0,  # Unknown for existing model
            is_baseline=True,
        )

        return {
            "message": "Baseline snapshot created",
            "snapshot_id": snapshot.id,
            "model_version": snapshot.model_version,
            "brier_score": snapshot.brier_score,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating baseline: {e}")
        raise HTTPException(status_code=500, detail="Failed to create baseline snapshot")


# ============================================================================
# LINEUP ARBITRAGE - Real-time odds capture at lineup announcement
# ============================================================================


@app.post("/lineup/monitor")
@limiter.limit("10/minute")
async def trigger_lineup_monitoring(
    request: Request,
    _: bool = Depends(verify_api_key),
):
    """
    Manually trigger lineup monitoring to capture odds at lineup_confirmed time.

    This is the same job that runs every 5 minutes automatically.
    Use this endpoint to test or force capture for matches in the next 90 minutes.

    The job:
    1. Finds matches starting within 90 minutes
    2. Checks if lineups are announced (11 players per team)
    3. If lineup is confirmed and no snapshot exists:
       - Captures current odds as 'lineup_confirmed' snapshot
       - Records exact timestamp for model evaluation

    This data is CRITICAL for evaluating the Lineup Arbitrage hypothesis:
    Can we beat the market odds AT THE MOMENT lineups are announced?

    Requires API key authentication.
    """
    from app.scheduler import monitor_lineups_and_capture_odds

    result = await monitor_lineups_and_capture_odds()
    return result


@app.get("/lineup/snapshots")
async def get_lineup_snapshots(
    days: int = 7,
    session: AsyncSession = Depends(get_async_session),
):
    """
    Get lineup_confirmed odds snapshots for recent matches.

    Returns matches where we captured odds at the moment of lineup announcement.
    This data is used to evaluate the Lineup Arbitrage model.

    Each snapshot includes:
    - match_id, date
    - snapshot_at: When we detected the lineup
    - odds at that moment (H/D/A)
    - implied probabilities (normalized)
    - TIMING METRICS: delta_to_kickoff, odds_freshness
    """
    from sqlalchemy import text

    cutoff = datetime.utcnow() - timedelta(days=days)

    result = await session.execute(text("""
        SELECT
            os.match_id,
            os.snapshot_at,
            os.odds_home,
            os.odds_draw,
            os.odds_away,
            os.prob_home,
            os.prob_draw,
            os.prob_away,
            os.overround,
            os.bookmaker,
            os.kickoff_time,
            os.delta_to_kickoff_seconds,
            os.odds_freshness,
            m.date as match_date,
            m.status,
            m.home_goals,
            m.away_goals,
            ht.name as home_team,
            at.name as away_team
        FROM odds_snapshots os
        JOIN matches m ON os.match_id = m.id
        LEFT JOIN teams ht ON m.home_team_id = ht.id
        LEFT JOIN teams at ON m.away_team_id = at.id
        WHERE os.snapshot_type = 'lineup_confirmed'
          AND os.snapshot_at >= :cutoff
        ORDER BY os.snapshot_at DESC
    """), {"cutoff": cutoff})

    snapshots = result.fetchall()

    # Calculate timing distribution
    deltas = [s.delta_to_kickoff_seconds for s in snapshots if s.delta_to_kickoff_seconds is not None]
    freshness_counts = {}
    for s in snapshots:
        f = s.odds_freshness or "unknown"
        freshness_counts[f] = freshness_counts.get(f, 0) + 1

    timing_stats = None
    if deltas:
        sorted_deltas = sorted(deltas)
        p50_idx = len(sorted_deltas) // 2
        p90_idx = int(len(sorted_deltas) * 0.9)
        timing_stats = {
            "count": len(deltas),
            "min_minutes": round(min(deltas) / 60, 1),
            "max_minutes": round(max(deltas) / 60, 1),
            "p50_minutes": round(sorted_deltas[p50_idx] / 60, 1),
            "p90_minutes": round(sorted_deltas[p90_idx] / 60, 1) if p90_idx < len(sorted_deltas) else None,
            "mean_minutes": round(sum(deltas) / len(deltas) / 60, 1),
        }

    return {
        "count": len(snapshots),
        "days": days,
        "timing_stats": timing_stats,
        "freshness_distribution": freshness_counts,
        "snapshots": [
            {
                "match_id": s.match_id,
                "home_team": s.home_team,
                "away_team": s.away_team,
                "match_date": s.match_date.isoformat() if s.match_date else None,
                "kickoff_time": s.kickoff_time.isoformat() if s.kickoff_time else None,
                "status": s.status,
                "final_score": f"{s.home_goals}-{s.away_goals}" if s.home_goals is not None else None,
                "snapshot_at": s.snapshot_at.isoformat() if s.snapshot_at else None,
                "delta_to_kickoff_minutes": round(s.delta_to_kickoff_seconds / 60, 1) if s.delta_to_kickoff_seconds else None,
                "odds_freshness": s.odds_freshness,
                "odds": {
                    "home": float(s.odds_home) if s.odds_home else None,
                    "draw": float(s.odds_draw) if s.odds_draw else None,
                    "away": float(s.odds_away) if s.odds_away else None,
                },
                "implied_probs": {
                    "home": float(s.prob_home) if s.prob_home else None,
                    "draw": float(s.prob_draw) if s.prob_draw else None,
                    "away": float(s.prob_away) if s.prob_away else None,
                },
                "overround": float(s.overround) if s.overround else None,
                "source": s.bookmaker,
            }
            for s in snapshots
        ],
    }


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






@app.get("/dashboard/pit.json")
async def pit_dashboard_json(request: Request):
    """
    PIT Dashboard JSON - Raw data for programmatic access.

    Returns the latest weekly/daily PIT report data.
    """
    if not _verify_dashboard_token(request):
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


@app.get("/dashboard/titan.json")
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
    if not _verify_dashboard_token(request):
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


@app.get("/dashboard/feature-coverage.json")
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
    if not _verify_dashboard_token(request):
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


@app.get("/dashboard/ml_health.json")
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

    if not _verify_dashboard_token(request):
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


@app.get("/dashboard/pit/debug")
async def pit_dashboard_debug(request: Request):
    """
    Debug endpoint - shows raw pit_reports table content.
    Protected by dashboard token.
    """
    if not _verify_dashboard_token(request):
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


@app.get("/dashboard/debug/experiment-gating/{match_id}")
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
    if not _verify_dashboard_token(request):
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


@app.post("/dashboard/pit/trigger")
async def pit_trigger_evaluation(request: Request):
    """
    Manually trigger PIT evaluation (for testing).
    Protected by dashboard token.
    """
    if not _verify_dashboard_token(request):
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


@app.get("/dashboard/overview/rollup.json")
async def dashboard_overview_rollup(request: Request):
    """
    V2 endpoint: Overview rollup data with standard wrapper.
    TTL: 60s
    Auth: X-Dashboard-Token
    """
    if not _verify_dashboard_token(request):
        raise HTTPException(status_code=401, detail="Dashboard access requires valid token.")

    import time as time_module
    now_ts = time_module.time()

    # Check cache
    if _rollup_cache["data"] and (now_ts - _rollup_cache["timestamp"]) < _rollup_cache["ttl"]:
        cache_age = now_ts - _rollup_cache["timestamp"]
        return _make_v2_wrapper(_rollup_cache["data"], cached=True, cache_age_seconds=cache_age)

    # Best-effort: get ops data and extract rollup subset
    try:
        ops_data = await _get_cached_ops_data()

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


@app.get("/dashboard/sentry/issues.json")
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
    if not _verify_dashboard_token(request):
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


@app.get("/dashboard/predictions/missing.json")
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
    if not _verify_dashboard_token(request):
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


@app.get("/dashboard/movement/recent.json")
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
    if not _verify_dashboard_token(request):
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
                       oh.recorded_at
                FROM odds_history oh
                JOIN matches m ON m.id = oh.match_id
                JOIN teams ht ON ht.id = m.home_team_id
                JOIN teams at ON at.id = m.away_team_id
                WHERE oh.recorded_at > :cutoff
                  AND NOT COALESCE(oh.quarantined, false)
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
                    "source": "api-football",
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


@app.get("/dashboard/movement/top.json")
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
    if not _verify_dashboard_token(request):
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


@app.post("/dashboard/predictions/trigger")
async def predictions_trigger_save(request: Request):
    """
    Manually trigger daily_save_predictions (for recovery).
    Protected by dashboard token.
    Use when predictions_health is RED/WARN.
    Returns detailed diagnostics for debugging.
    """
    if not _verify_dashboard_token(request):
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


@app.post("/dashboard/predictions/trigger-fase0")
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
    if not _verify_dashboard_token(request):
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


# =============================================================================
# OPS DASHBOARD (DB-backed, cached)
# =============================================================================

_ops_dashboard_cache = {
    "data": None,
    "timestamp": 0,
    "ttl": 45,  # seconds
    # Refresh state (avoid doing heavy DB work inside HTTP requests)
    "refreshing": False,
    "last_refresh_reason": None,
    "last_refresh_error": None,
    "last_refresh_started_at": None,   # epoch seconds
    "last_refresh_finished_at": None,  # epoch seconds
    "last_refresh_duration_ms": None,
    # Backoff (prevent tight retry loops if DB is unhealthy)
    "refresh_failures": 0,
    "next_refresh_after": 0,  # epoch seconds
}

# Best-effort handle for the latest refresh task (debug/visibility only)
_ops_dashboard_refresh_task = None

# Rate-limit OPS_ALERT logging (once per 5 minutes max)
_predictions_health_alert_last: float = 0
_PREDICTIONS_HEALTH_ALERT_COOLDOWN = 300  # 5 minutes


async def _calculate_telemetry_summary(session) -> dict:
    """
    Calculate Data Quality Telemetry summary for ops dashboard.

    Queries DB for quarantine/taint/unmapped counts (24h window).
    Returns status: OK/WARN/RED based on data quality flags.

    Thresholds (conservative, protecting training/backtest/value):
    - RED: tainted_matches > 0 OR quarantined_odds > 0
    - WARN: unmapped_entities > 0 (and tainted/quarantined == 0)
    - OK: all counters == 0
    """
    now = datetime.utcnow()

    # NOTE (2026-02): Keep /dashboard/ops.json cheap.
    # Consolidate multiple counts into a single statement to reduce
    # "Consecutive DB Queries" noise and DB round-trips.
    quarantined_odds_24h = 0
    tainted_matches_24h = 0
    unmapped_entities_24h = 0
    odds_desync_6h = 0
    odds_desync_90m = 0

    try:
        res = await session.execute(
            text("""
                SELECT
                  -- 1) Quarantined odds in last 24h
                  (SELECT COUNT(*)
                     FROM odds_history
                    WHERE quarantined = true
                      AND recorded_at > NOW() - INTERVAL '24 hours'
                  ) AS quarantined_odds_24h,

                  -- 2) Tainted matches (recent)
                  (SELECT COUNT(*)
                     FROM matches
                    WHERE tainted = true
                      AND date > NOW() - INTERVAL '7 days'
                  ) AS tainted_matches_7d,

                  -- 3) Unmapped entities (teams without logo)
                  (SELECT COUNT(DISTINCT t.id)
                     FROM teams t
                    WHERE t.logo_url IS NULL
                  ) AS unmapped_entities,

                  -- 4a) Odds desync 6h window (early warning)
                  (SELECT COUNT(DISTINCT m.id)
                     FROM matches m
                     JOIN odds_snapshots os ON os.match_id = m.id
                    WHERE m.status = 'NS'
                      AND m.date BETWEEN NOW() AND NOW() + INTERVAL '6 hours'
                      AND os.odds_freshness = 'live'
                      AND os.snapshot_type = 'lineup_confirmed'
                      AND os.snapshot_at >= NOW() - INTERVAL '120 minutes'
                      AND (m.odds_home IS NULL OR m.odds_draw IS NULL OR m.odds_away IS NULL)
                  ) AS odds_desync_6h,

                  -- 4b) Odds desync 90m window (near kickoff, critical)
                  (SELECT COUNT(DISTINCT m.id)
                     FROM matches m
                     JOIN odds_snapshots os ON os.match_id = m.id
                    WHERE m.status = 'NS'
                      AND m.date BETWEEN NOW() AND NOW() + INTERVAL '90 minutes'
                      AND os.odds_freshness = 'live'
                      AND os.snapshot_type = 'lineup_confirmed'
                      AND os.snapshot_at >= NOW() - INTERVAL '120 minutes'
                      AND (m.odds_home IS NULL OR m.odds_draw IS NULL OR m.odds_away IS NULL)
                  ) AS odds_desync_90m
            """)
        )
        row = res.first()
        if row:
            quarantined_odds_24h = int(row[0] or 0)
            tainted_matches_24h = int(row[1] or 0)
            unmapped_entities_24h = int(row[2] or 0)
            odds_desync_6h = int(row[3] or 0)
            odds_desync_90m = int(row[4] or 0)
    except Exception:
        # Fail-soft: keep ops dashboard alive if a table/column is missing.
        # (The older multi-query approach was more granular; this is the minimal-safe fallback.)
        pass

    # Determine status
    # RED: desync near kickoff (90m) OR tainted/quarantined
    # WARN: desync in 6h window OR unmapped entities
    if odds_desync_90m > 0 or tainted_matches_24h > 0 or quarantined_odds_24h > 0:
        status = "RED"
    elif odds_desync_6h > 0 or unmapped_entities_24h > 0:
        status = "WARN"
    else:
        status = "OK"

    # Build Grafana links from env vars (only if configured)
    links = []
    grafana_urls = {
        "Availability": os.environ.get("GRAFANA_DQ_AVAIL_URL"),
        "Freshness/Lag": os.environ.get("GRAFANA_DQ_LAG_URL"),
        "Market Integrity": os.environ.get("GRAFANA_DQ_MARKET_URL"),
        "Mapping Coverage": os.environ.get("GRAFANA_DQ_MAPPING_URL"),
    }
    for title, url in grafana_urls.items():
        if url:
            links.append({"title": f"Grafana: {title}", "url": url})

    return {
        "status": status,
        "updated_at": now.isoformat(),
        "summary": {
            "quarantined_odds_24h": quarantined_odds_24h,
            "tainted_matches_24h": tainted_matches_24h,
            "unmapped_entities_24h": unmapped_entities_24h,
            "odds_desync_6h": odds_desync_6h,
            "odds_desync_90m": odds_desync_90m,
        },
        "links": links,
    }


async def _refresh_ops_dashboard_cache(reason: str = "unknown") -> None:
    """Refresh ops dashboard cache in background (fail-soft)."""
    start_ts = time.time()
    _ops_dashboard_cache["refreshing"] = True
    _ops_dashboard_cache["last_refresh_reason"] = reason
    _ops_dashboard_cache["last_refresh_started_at"] = start_ts
    _ops_dashboard_cache["last_refresh_error"] = None

    try:
        data = await _load_ops_data()
        _ops_dashboard_cache["data"] = data
        _ops_dashboard_cache["timestamp"] = time.time()
        _ops_dashboard_cache["refresh_failures"] = 0
        _ops_dashboard_cache["next_refresh_after"] = 0
    except Exception as e:
        # Backoff: exponential up to 5 minutes
        failures = int(_ops_dashboard_cache.get("refresh_failures") or 0) + 1
        _ops_dashboard_cache["refresh_failures"] = failures
        backoff_seconds = min(300, 2 ** min(failures, 8))
        _ops_dashboard_cache["next_refresh_after"] = time.time() + backoff_seconds
        _ops_dashboard_cache["last_refresh_error"] = f"{type(e).__name__}: {e}"
        logger.warning(
            f"[OPS_DASHBOARD] Cache refresh failed ({reason}) ({type(e).__name__}): {e!r}",
            exc_info=True,
        )
    finally:
        end_ts = time.time()
        _ops_dashboard_cache["last_refresh_finished_at"] = end_ts
        _ops_dashboard_cache["last_refresh_duration_ms"] = int((end_ts - start_ts) * 1000)
        _ops_dashboard_cache["refreshing"] = False


def _schedule_ops_dashboard_cache_refresh(reason: str = "stale") -> None:
    """
    Schedule a cache refresh without inheriting the current request context.

    Why: Sentry performance tracing can attribute async child tasks to the
    current HTTP transaction (contextvars propagation), re-triggering the
    "Consecutive DB Queries" alert even if we don't await the refresh.
    """
    import asyncio
    import contextvars

    global _ops_dashboard_refresh_task

    # Avoid duplicate refreshes
    if _ops_dashboard_cache.get("refreshing"):
        return

    # Respect backoff window after failures
    now = time.time()
    next_after = float(_ops_dashboard_cache.get("next_refresh_after") or 0)
    if next_after and now < next_after:
        return

    # Mark refreshing early to prevent thundering herd scheduling
    _ops_dashboard_cache["refreshing"] = True

    try:
        # Run task creation in a fresh (empty) Context to detach request scope.
        ctx = contextvars.Context()
        _ops_dashboard_refresh_task = ctx.run(
            asyncio.create_task,
            _refresh_ops_dashboard_cache(reason=reason),
        )
    except Exception as e:
        _ops_dashboard_cache["refreshing"] = False
        _ops_dashboard_cache["last_refresh_error"] = f"schedule_failed: {type(e).__name__}: {e}"
        logger.warning(f"[OPS_DASHBOARD] Could not schedule cache refresh: {e!r}")


async def _calculate_shadow_mode_summary(session) -> dict:
    """
    Calculate Shadow Mode summary for ops dashboard.

    Returns state, counts, metrics, and recommendation for A/B model comparison.
    """
    from app.ml.shadow import is_shadow_enabled, get_shadow_engine
    from app.config import get_settings

    settings = get_settings()
    now = datetime.utcnow()

    # State info
    shadow_arch = settings.MODEL_SHADOW_ARCHITECTURE
    enabled = bool(shadow_arch)
    shadow_engine = get_shadow_engine()
    engine_loaded = shadow_engine is not None and shadow_engine.is_loaded if shadow_engine else False

    state = {
        "enabled": enabled,
        "shadow_architecture": shadow_arch or None,
        "shadow_model_version": shadow_engine.model_version if engine_loaded else None,
        "baseline_model_version": settings.MODEL_VERSION,
        "last_evaluation_at": None,
        "evaluation_job_interval_minutes": 30,
    }

    # Thresholds from settings
    min_samples = settings.SHADOW_MIN_SAMPLES
    brier_improvement_min = settings.SHADOW_BRIER_IMPROVEMENT_MIN
    accuracy_drop_max = settings.SHADOW_ACCURACY_DROP_MAX
    window_days = settings.SHADOW_WINDOW_DAYS

    # Default response if disabled
    if not enabled or not engine_loaded:
        return {
            "state": state,
            "counts": None,
            "metrics": None,
            "gating": {
                "min_samples_required": min_samples,
                "samples_evaluated": 0,
            },
            "recommendation": {
                "status": "DISABLED" if not enabled else "NOT_LOADED",
                "reason": "Shadow mode not configured" if not enabled else "Shadow model not loaded",
            },
        }

    # Counts query (window)
    counts = {
        "shadow_predictions_total": 0,
        "shadow_predictions_evaluated": 0,
        "shadow_predictions_pending": 0,
        "shadow_predictions_last_24h": 0,
        "shadow_evaluations_last_24h": 0,
    }

    try:
        # Rollback any previous failed transaction state
        await session.rollback()

        # Total predictions
        res = await session.execute(
            text(f"""
                SELECT COUNT(*) FROM shadow_predictions
                WHERE created_at > NOW() - INTERVAL '{window_days} days'
            """)
        )
        counts["shadow_predictions_total"] = int(res.scalar() or 0)

        # Evaluated vs pending
        res = await session.execute(
            text(f"""
                SELECT
                    COUNT(*) FILTER (WHERE evaluated_at IS NOT NULL) AS evaluated,
                    COUNT(*) FILTER (WHERE evaluated_at IS NULL) AS pending
                FROM shadow_predictions
                WHERE created_at > NOW() - INTERVAL '{window_days} days'
            """)
        )
        row = res.first()
        if row:
            counts["shadow_predictions_evaluated"] = int(row[0] or 0)
            counts["shadow_predictions_pending"] = int(row[1] or 0)

        # Last 24h predictions
        res = await session.execute(
            text("""
                SELECT COUNT(*) FROM shadow_predictions
                WHERE created_at > NOW() - INTERVAL '24 hours'
            """)
        )
        counts["shadow_predictions_last_24h"] = int(res.scalar() or 0)

        # Last 24h evaluations
        res = await session.execute(
            text("""
                SELECT COUNT(*) FROM shadow_predictions
                WHERE evaluated_at > NOW() - INTERVAL '24 hours'
            """)
        )
        counts["shadow_evaluations_last_24h"] = int(res.scalar() or 0)

        # Errors in last 24h (shadow prediction failures)
        res = await session.execute(
            text("""
                SELECT COUNT(*) FROM shadow_predictions
                WHERE error_code IS NOT NULL
                  AND created_at > NOW() - INTERVAL '24 hours'
            """)
        )
        counts["shadow_errors_last_24h"] = int(res.scalar() or 0)

        # Last evaluation timestamp
        res = await session.execute(
            text("SELECT MAX(evaluated_at) FROM shadow_predictions")
        )
        last_eval = res.scalar()
        if last_eval:
            state["last_evaluation_at"] = last_eval.isoformat() if hasattr(last_eval, 'isoformat') else str(last_eval)

    except Exception as e:
        logger.warning(f"Shadow mode counts query failed: {e}")

    # Metrics (only if enough evaluated samples)
    samples_evaluated = counts["shadow_predictions_evaluated"]
    metrics = None
    head_to_head = None
    draws_info = None

    if samples_evaluated >= min_samples:
        try:
            # Accuracy and Brier
            res = await session.execute(
                text(f"""
                    SELECT
                        AVG(CASE WHEN baseline_correct THEN 1.0 ELSE 0.0 END) AS baseline_acc,
                        AVG(CASE WHEN shadow_correct THEN 1.0 ELSE 0.0 END) AS shadow_acc,
                        AVG(baseline_brier) AS baseline_brier,
                        AVG(shadow_brier) AS shadow_brier,
                        SUM(CASE WHEN shadow_correct AND NOT baseline_correct THEN 1 ELSE 0 END) AS shadow_better,
                        SUM(CASE WHEN baseline_correct AND NOT shadow_correct THEN 1 ELSE 0 END) AS baseline_better,
                        SUM(CASE WHEN shadow_correct AND baseline_correct THEN 1 ELSE 0 END) AS both_correct,
                        SUM(CASE WHEN NOT shadow_correct AND NOT baseline_correct THEN 1 ELSE 0 END) AS both_wrong
                    FROM shadow_predictions
                    WHERE evaluated_at IS NOT NULL
                      AND created_at > NOW() - INTERVAL '{window_days} days'
                """)
            )
            row = res.first()
            if row:
                baseline_acc = float(row[0] or 0)
                shadow_acc = float(row[1] or 0)
                baseline_brier = float(row[2] or 0)
                shadow_brier = float(row[3] or 0)

                metrics = {
                    "baseline_accuracy": round(baseline_acc, 4),
                    "shadow_accuracy": round(shadow_acc, 4),
                    "baseline_brier": round(baseline_brier, 4),
                    "shadow_brier": round(shadow_brier, 4),
                    "delta_accuracy": round(shadow_acc - baseline_acc, 4),
                    "delta_brier": round(shadow_brier - baseline_brier, 4),
                }

                head_to_head = {
                    "shadow_better": int(row[4] or 0),
                    "baseline_better": int(row[5] or 0),
                    "both_correct": int(row[6] or 0),
                    "both_wrong": int(row[7] or 0),
                }

            # Draw prediction stats
            res = await session.execute(
                text(f"""
                    SELECT
                        AVG(CASE WHEN shadow_predicted = 'draw' THEN 1.0 ELSE 0.0 END) AS shadow_draw_pct,
                        AVG(CASE WHEN actual_result = 'draw' THEN 1.0 ELSE 0.0 END) AS actual_draw_pct
                    FROM shadow_predictions
                    WHERE evaluated_at IS NOT NULL
                      AND created_at > NOW() - INTERVAL '{window_days} days'
                """)
            )
            row = res.first()
            if row:
                draws_info = {
                    "shadow_draw_predicted_pct": round(float(row[0] or 0) * 100, 1),
                    "actual_draw_pct": round(float(row[1] or 0) * 100, 1),
                }

        except Exception as e:
            logger.warning(f"Shadow mode metrics query failed: {e}")

    # Recommendation logic
    if samples_evaluated < min_samples:
        recommendation = {
            "status": "NO_DATA",
            "reason": f"Need {min_samples} evaluated samples, have {samples_evaluated}",
        }
    elif metrics:
        delta_brier = metrics["delta_brier"]
        delta_acc = metrics["delta_accuracy"]

        # GO: shadow improves brier AND doesn't hurt accuracy too much
        if delta_brier <= -brier_improvement_min and delta_acc >= -accuracy_drop_max:
            recommendation = {
                "status": "GO",
                "reason": f"Shadow improves Brier by {-delta_brier:.4f}, accuracy delta {delta_acc:+.1%}",
            }
        # NO_GO: shadow degrades accuracy significantly
        elif delta_acc < -accuracy_drop_max:
            recommendation = {
                "status": "NO_GO",
                "reason": f"Shadow degrades accuracy by {-delta_acc:.1%} (max allowed: {accuracy_drop_max:.1%})",
            }
        # NO_GO: shadow makes brier worse
        elif delta_brier > brier_improvement_min:
            recommendation = {
                "status": "NO_GO",
                "reason": f"Shadow degrades Brier by {delta_brier:.4f}",
            }
        # HOLD: not enough improvement to switch
        else:
            recommendation = {
                "status": "HOLD",
                "reason": f"Shadow comparable to baseline (Brier delta: {delta_brier:+.4f}, accuracy delta: {delta_acc:+.1%})",
            }
    else:
        recommendation = {
            "status": "NO_DATA",
            "reason": "Metrics not available",
        }

    # Health signals for telemetry
    health = {
        "pending_ft_to_evaluate": 0,
        "eval_lag_minutes": 0.0,
        "stale_threshold_minutes": settings.SHADOW_EVAL_STALE_MINUTES,
        "is_stale": False,
    }
    try:
        from app.ml.shadow import get_shadow_health_metrics
        health_data = await get_shadow_health_metrics(session)
        health["pending_ft_to_evaluate"] = health_data.get("pending_ft", 0)
        health["eval_lag_minutes"] = health_data.get("eval_lag_minutes", 0.0)
        health["is_stale"] = health["eval_lag_minutes"] > settings.SHADOW_EVAL_STALE_MINUTES
    except Exception as e:
        logger.warning(f"Shadow health metrics query failed: {e}")

    return {
        "state": state,
        "counts": counts,
        "metrics": metrics,
        "head_to_head": head_to_head,
        "draws": draws_info,
        "gating": {
            "min_samples_required": min_samples,
            "samples_evaluated": samples_evaluated,
            "brier_improvement_min": brier_improvement_min,
            "accuracy_drop_max": accuracy_drop_max,
            "window_days": window_days,
        },
        "recommendation": recommendation,
        "health": health,
    }


async def _calculate_sensor_b_summary(session) -> dict:
    """
    Calculate Sensor B summary for ops dashboard.

    Sensor B is INTERNAL DIAGNOSTICS ONLY - never affects production picks.
    Returns flat structure for easy card rendering.

    States (Auditor-approved):
    - DISABLED: SENSOR_ENABLED=false
    - LEARNING: <min_samples evaluated, not ready to report metrics
    - TRACKING: >=min_samples, no conclusive signal yet
    - SIGNAL_DETECTED: signal_score > threshold, A may be stale
    - OVERFITTING_SUSPECTED: signal_score < threshold, B is noise
    - ERROR: exception during computation
    """
    from app.ml.sensor import get_sensor_report
    from app.config import get_settings

    sensor_settings = get_settings()

    if not sensor_settings.SENSOR_ENABLED:
        return {
            "state": "DISABLED",
            "reason": "SENSOR_ENABLED=false",
        }

    try:
        # Rollback any previous failed transaction
        await session.rollback()

        report = await get_sensor_report(session)

        # Determine state for card display (Auditor-approved statuses)
        # AUDIT P0: Derive state from recommendation.status (source of truth)
        rec = report.get("recommendation", {})
        rec_status = rec.get("status", "NO_DATA")
        report_status = report.get("status", "")

        if report_status in ("NO_DATA", "INSUFFICIENT_DATA", "DISABLED"):
            state = "LEARNING"
        elif rec_status in ("SIGNAL_DETECTED", "OVERFITTING_SUSPECTED", "TRACKING"):
            state = rec_status
        elif rec_status == "LEARNING":
            state = "LEARNING"
        else:
            state = "LEARNING"

        # Extract metrics for flat card display
        metrics = report.get("metrics", {})
        gating = report.get("gating", {})
        sensor_info = report.get("sensor_info", {})
        counts = report.get("counts", {})

        # Health signals for telemetry
        health = {
            "pending_ft_to_evaluate": 0,
            "eval_lag_minutes": 0.0,
            "stale_threshold_minutes": sensor_settings.SENSOR_EVAL_STALE_MINUTES,
            "is_stale": False,
        }
        try:
            from app.ml.sensor import get_sensor_health_metrics
            health_data = await get_sensor_health_metrics(session)
            health["pending_ft_to_evaluate"] = health_data.get("pending_ft", 0)
            health["eval_lag_minutes"] = health_data.get("eval_lag_minutes", 0.0)
            health["is_stale"] = health["eval_lag_minutes"] > sensor_settings.SENSOR_EVAL_STALE_MINUTES
        except Exception as he:
            logger.warning(f"Sensor health metrics query failed: {he}")

        # Compute accuracy percentages (only if samples >= min_samples)
        # AUDIT P0: Use evaluated_with_b for A vs B comparison (where sensor produced predictions)
        samples_evaluated = counts.get("evaluated_with_b", 0)
        samples_evaluated_total = counts.get("evaluated_total", 0)
        samples_pending = counts.get("pending_with_b", 0)
        samples_pending_total = counts.get("pending_total", 0)
        # AUDIT: Expose missing B predictions (sensor was LEARNING when logged)
        missing_b_evaluated = counts.get("missing_b_evaluated", 0)
        missing_b_pending = counts.get("missing_b_pending", 0)
        min_samples = gating.get("min_samples_required", 50)
        has_enough_samples = samples_evaluated >= min_samples

        # Accuracy fields (null if not enough samples)
        accuracy_a_pct = None
        accuracy_b_pct = None
        delta_accuracy_pct = None

        if has_enough_samples:
            a_acc = metrics.get("a_accuracy")
            b_acc = metrics.get("b_accuracy")
            if a_acc is not None and b_acc is not None:
                accuracy_a_pct = round(a_acc * 100, 1)
                accuracy_b_pct = round(b_acc * 100, 1)
                delta_accuracy_pct = round((b_acc - a_acc) * 100, 1)

        return {
            "state": state,
            "reason": rec.get("reason", ""),
            # Counts - AUDIT P0: expose both total and with_b, use with_b for card
            "samples_evaluated": samples_evaluated,  # evaluated_with_b (A vs B comparison)
            "samples_evaluated_total": samples_evaluated_total,  # all evaluated
            "samples_pending": samples_pending,  # pending_with_b (will have B to compare)
            "samples_pending_total": samples_pending_total,  # all pending
            # AUDIT: Expose missing B predictions (sensor was LEARNING when logged)
            # These records are excluded from A vs B comparison metrics
            "missing_b_evaluated": missing_b_evaluated,
            "missing_b_pending": missing_b_pending,
            "missing_b_total": missing_b_evaluated + missing_b_pending,
            # Warning if sensor is ready but there are missing B predictions (needs retry)
            "has_missing_b": (missing_b_evaluated + missing_b_pending) > 0,
            "missing_b_warning": (
                "retry needed" if sensor_info.get("is_ready") and (missing_b_evaluated + missing_b_pending) > 0
                else None
            ),
            "min_samples": min_samples,
            # Accuracy A vs B (Auditor card) - only present if samples >= min_samples
            "accuracy_a_pct": accuracy_a_pct,
            "accuracy_b_pct": accuracy_b_pct,
            "delta_accuracy_pct": delta_accuracy_pct,
            "window_days": sensor_settings.SENSOR_EVAL_WINDOW_DAYS,
            "note": "solo FT evaluados (apples-to-apples con Model A)",
            # Metrics (only show if we have enough samples - gating)
            "signal_score": metrics.get("signal_score") if state != "LEARNING" else None,
            "brier_a": metrics.get("a_brier") if state != "LEARNING" else None,
            "brier_b": metrics.get("b_brier") if state != "LEARNING" else None,
            "delta_brier": metrics.get("delta_brier") if state != "LEARNING" else None,
            "accuracy_a": metrics.get("a_accuracy") if state != "LEARNING" else None,
            "accuracy_b": metrics.get("b_accuracy") if state != "LEARNING" else None,
            # Sensor info
            "window_size": sensor_info.get("window_size", 50),
            "last_retrain_at": sensor_info.get("last_trained"),
            "retrain_interval_hours": sensor_settings.SENSOR_RETRAIN_INTERVAL_HOURS,
            "model_version": sensor_info.get("model_version"),
            "is_ready": sensor_info.get("is_ready", False),
            # Health (telemetry)
            "health": health,
            # Sanity check (P0 ATI/ADA): detect overconfidence in last 24h
            "sanity": report.get("sanity"),
        }
    except Exception as e:
        logger.warning(f"Sensor B summary failed: {e}")
        return {
            "state": "ERROR",
            "reason": str(e)[:100],
        }


async def _calculate_extc_shadow_summary(session) -> dict:
    """
    Calculate ext-C shadow model summary for ops dashboard.

    ext-C is an experimental model evaluation job that generates
    shadow predictions for the v1.0.2-ext-C model variant.

    States:
    - DISABLED: EXTC_SHADOW_ENABLED=false
    - ACTIVE: Job is enabled and running
    - ERROR: Issue with job execution
    """
    from app.config import get_settings
    from app.telemetry.metrics import (
        EXTC_SHADOW_INSERTED,
        EXTC_SHADOW_ERRORS,
        EXTC_SHADOW_LAST_SUCCESS,
    )

    extc_settings = get_settings()

    if not extc_settings.EXTC_SHADOW_ENABLED:
        return {
            "state": "DISABLED",
            "reason": "EXTC_SHADOW_ENABLED=false",
            "model_version": extc_settings.EXTC_SHADOW_MODEL_VERSION,
        }

    try:
        # Get metrics from Prometheus
        inserted_total = 0
        errors_total = 0
        last_success_ts = None

        try:
            inserted_total = int(EXTC_SHADOW_INSERTED._value.get() or 0)
        except Exception:
            pass

        try:
            errors_total = int(EXTC_SHADOW_ERRORS._value.get() or 0)
        except Exception:
            pass

        try:
            ts = EXTC_SHADOW_LAST_SUCCESS._value.get()
            if ts and ts > 0:
                last_success_ts = datetime.utcfromtimestamp(ts).isoformat() + "Z"
        except Exception:
            pass

        # Get count from predictions_experiments table
        predictions_count = 0
        try:
            result = await session.execute(text("""
                SELECT COUNT(*) as n
                FROM predictions_experiments
                WHERE model_version = :model_version
            """), {"model_version": extc_settings.EXTC_SHADOW_MODEL_VERSION})
            row = result.first()
            predictions_count = int(row[0]) if row else 0
        except Exception as e:
            logger.debug(f"[EXTC_SHADOW] Count query failed: {e}")

        return {
            "state": "ACTIVE",
            "model_version": extc_settings.EXTC_SHADOW_MODEL_VERSION,
            "model_path": extc_settings.EXTC_SHADOW_MODEL_PATH,
            "interval_minutes": extc_settings.EXTC_SHADOW_INTERVAL_MINUTES,
            "batch_size": extc_settings.EXTC_SHADOW_BATCH_SIZE,
            "oos_only": extc_settings.EXTC_SHADOW_OOS_ONLY,
            "start_at": extc_settings.EXTC_SHADOW_START_AT,
            "predictions_count": predictions_count,
            "inserted_total": inserted_total,
            "errors_total": errors_total,
            "last_success_at": last_success_ts,
        }

    except Exception as e:
        logger.warning(f"ext-C shadow summary failed: {e}")
        return {
            "state": "ERROR",
            "reason": str(e)[:100],
            "model_version": extc_settings.EXTC_SHADOW_MODEL_VERSION,
        }


async def _calculate_jobs_health_summary(session) -> dict:
    """
    Calculate P0 jobs health summary for OPS dashboard.

    Monitors the three critical jobs:
    - stats_backfill: Fetches match stats for finished matches
    - odds_sync: Syncs 1X2 odds for upcoming matches
    - fastpath: Generates LLM narratives for finished matches

    Each job reports:
    - last_success_at: Timestamp of last OK run
    - minutes_since_success: Gap since last OK
    - ft_pending (stats_backfill): FT matches without stats
    - backlog_ready (fastpath): Audits ready for narratives
    - status: ok/warn/red based on thresholds

    P1-B: Falls back to DB (job_runs table) when Prometheus metrics are unavailable
    (e.g., cold-start after deploy).
    """
    from app.telemetry.metrics import (
        job_last_success_timestamp,
        stats_backfill_ft_pending_gauge,
        fastpath_backlog_ready_gauge,
    )

    now = datetime.utcnow()

    # P1-B: Preload DB fallback data for all jobs
    db_fallback = {}
    try:
        from app.jobs.tracking import get_jobs_health_from_db
        db_fallback = await get_jobs_health_from_db(session)
    except Exception as e:
        logger.debug(f"[JOBS_HEALTH] DB fallback unavailable: {e}")

    # Helper to format timestamp and calculate age
    def job_status(job_name: str, max_gap_minutes: int) -> dict:
        last_success = None
        source = "prometheus"

        # Try Prometheus first
        try:
            ts = job_last_success_timestamp.labels(job=job_name)._value.get()
            if ts and ts > 0:
                last_success = datetime.utcfromtimestamp(ts)
        except Exception:
            pass

        # P1-B: Fallback to DB if Prometheus has no data
        if last_success is None and job_name in db_fallback:
            db_data = db_fallback[job_name]
            if db_data.get("last_success_at"):
                try:
                    # Parse ISO format
                    ts_str = db_data["last_success_at"].rstrip("Z")
                    last_success = datetime.fromisoformat(ts_str)
                    source = "db_fallback"
                except Exception:
                    pass

        if last_success:
            gap_minutes = (now - last_success).total_seconds() / 60
            status = "ok"
            if gap_minutes > max_gap_minutes * 2:
                status = "red"
            elif gap_minutes > max_gap_minutes:
                status = "warn"
            return {
                "last_success_at": last_success.isoformat() + "Z",
                "minutes_since_success": round(gap_minutes, 1),
                "status": status,
                "source": source,  # For debugging
            }

        return {
            "last_success_at": None,
            "minutes_since_success": None,
            "status": "unknown",
        }

    # Stats backfill: runs hourly, warn if >2h, red if >3h
    stats_health = job_status("stats_backfill", max_gap_minutes=120)
    try:
        ft_pending = int(stats_backfill_ft_pending_gauge._value.get() or 0)
        stats_health["ft_pending"] = ft_pending
        # Idle override: if no pending work, job is healthy regardless of timer
        if ft_pending == 0:
            stats_health["status"] = "ok"
            stats_health["note"] = "idle_no_pending"
        else:
            # Escalate status based on pending count
            if ft_pending > 10:
                stats_health["status"] = "red"
            elif ft_pending > 5 and stats_health["status"] == "ok":
                stats_health["status"] = "warn"
    except Exception:
        stats_health["ft_pending"] = None

    # Odds sync: runs every 6h, warn if >12h, red if >18h
    odds_health = job_status("odds_sync", max_gap_minutes=720)

    # Fastpath: runs every 2min, warn if >5min, red if >10min
    fastpath_health = job_status("fastpath", max_gap_minutes=5)
    try:
        backlog_ready = int(fastpath_backlog_ready_gauge._value.get() or 0)
        fastpath_health["backlog_ready"] = backlog_ready
        # Escalate status based on backlog
        if backlog_ready > 5:
            fastpath_health["status"] = "red"
        elif backlog_ready > 3 and fastpath_health["status"] == "ok":
            fastpath_health["status"] = "warn"
    except Exception:
        fastpath_health["backlog_ready"] = None

    # Overall status: worst of the three
    statuses = [stats_health["status"], odds_health["status"], fastpath_health["status"]]
    if "red" in statuses:
        overall = "red"
    elif "warn" in statuses:
        overall = "warn"
    elif all(s == "ok" for s in statuses):
        overall = "ok"
    else:
        overall = "unknown"

    # Add help URLs for oncall quick reference
    runbook_base = "docs/GRAFANA_ALERTS_CHECKLIST.md"
    stats_health["help_url"] = f"{runbook_base}#stats-backfill-job"
    odds_health["help_url"] = f"{runbook_base}#odds-sync-job"
    fastpath_health["help_url"] = f"{runbook_base}#fastpath-llm-narratives-job"

    # Build top_alert for warn/red status (Auditor Dashboard enhancement)
    top_alert = None
    alerts_count = 0

    # Helper: compute stable incident_id for a job (same hash as _aggregate_incidents)
    # Uses "jobs:" prefix — canonical: id = md5("jobs:<job_key>")
    def _job_incident_id(job_key: str) -> int:
        import hashlib
        h = hashlib.md5(f"jobs:{job_key}".encode()).hexdigest()
        return int(h[:8], 16)

    # Add incident_id to each job for deep-linking from dashboard
    for _jk, _jd in [("stats_backfill", stats_health), ("odds_sync", odds_health), ("fastpath", fastpath_health)]:
        _jd["incident_id"] = _job_incident_id(_jk)

    if overall in ("warn", "red"):
        # Collect all jobs with their severity for ranking
        job_alerts = []
        jobs_meta = {
            "stats_backfill": {"data": stats_health, "label": "Stats Backfill"},
            "odds_sync": {"data": odds_health, "label": "Odds Sync"},
            "fastpath": {"data": fastpath_health, "label": "Fast-Path Narratives"},
        }

        for job_key, meta in jobs_meta.items():
            job_data = meta["data"]
            job_status_val = job_data.get("status", "unknown")

            if job_status_val in ("warn", "red"):
                alerts_count += 1
                minutes_since = job_data.get("minutes_since_success")

                # Build reason string
                if minutes_since is not None:
                    if minutes_since >= 60:
                        hours_ago = round(minutes_since / 60, 1)
                        reason = f"Last success {hours_ago}h ago"
                    else:
                        reason = f"Last success {int(minutes_since)}m ago"
                else:
                    reason = "No recent success recorded"

                # Add context for specific jobs
                if job_key == "stats_backfill" and job_data.get("ft_pending"):
                    reason += f" ({job_data['ft_pending']} FT pending)"
                elif job_key == "fastpath" and job_data.get("backlog_ready"):
                    reason += f" ({job_data['backlog_ready']} backlog)"

                job_alerts.append({
                    "job_key": job_key,
                    "label": meta["label"],
                    "severity": job_status_val,
                    "reason": reason,
                    "minutes_since_success": minutes_since,
                    "runbook_url": job_data.get("help_url"),
                    "incident_id": _job_incident_id(job_key),
                    # Sort key: red=2, warn=1; then by minutes_since_success desc
                    "_sort_key": (2 if job_status_val == "red" else 1, minutes_since or 0),
                })

        # Select worst job as top_alert
        if job_alerts:
            job_alerts.sort(key=lambda x: x["_sort_key"], reverse=True)
            worst = job_alerts[0]
            top_alert = {
                "job_key": worst["job_key"],
                "label": worst["label"],
                "severity": worst["severity"],
                "reason": worst["reason"],
                "minutes_since_success": worst["minutes_since_success"],
                "runbook_url": worst["runbook_url"],
                "incident_id": worst["incident_id"],
            }

    result = {
        "status": overall,
        "runbook_url": f"{runbook_base}#p0-jobs-health-scheduler-jobs",
        "stats_backfill": stats_health,
        "odds_sync": odds_health,
        "fastpath": fastpath_health,
    }

    # Only include top_alert fields when there are alerts
    if top_alert:
        result["top_alert"] = top_alert
        result["alerts_count"] = alerts_count

    return result


async def _calculate_sota_enrichment_summary(session) -> dict:
    """
    Calculate SOTA enrichment coverage metrics for OPS dashboard.

    Reports coverage and staleness for:
    - Understat xG data (match_understat_team)
    - Weather forecasts (match_weather)
    - Venue geo coordinates (venue_geo)
    - Team home city profiles (team_home_city_profile)
    - Sofascore XI lineups (match_sofascore_lineup)

    All metrics are best-effort: query failures return "unavailable" status.

    STANDARDIZED SHAPE (per component):
    {
        "status": "ok" | "warn" | "red" | "unavailable",
        "coverage_pct": float,
        "total": int,
        "with_data": int,
        "staleness_hours": float | null,
        "note": string | null
    }
    """
    now = datetime.utcnow()
    result = {"status": "ok", "generated_at": now.isoformat()}

    # Helper to build unavailable response
    def _unavailable(note: str) -> dict:
        return {
            "status": "unavailable",
            "coverage_pct": 0.0,
            "total": 0,
            "with_data": 0,
            "staleness_hours": None,
            "note": note,
        }

    # 1) Understat coverage: FT matches in last 14 days with xG data
    understat_league_ids = ",".join(str(lid) for lid in UNDERSTAT_SUPPORTED_LEAGUES)
    try:
        res = await session.execute(
            text(f"""
                SELECT
                    COUNT(*) FILTER (WHERE mut.match_id IS NOT NULL) AS with_xg,
                    COUNT(*) AS total_ft
                FROM matches m
                LEFT JOIN match_understat_team mut ON m.id = mut.match_id
                WHERE m.status IN ('FT', 'AET', 'PEN')
                  AND m.date >= NOW() - INTERVAL '14 days'
                  AND m.league_id IN ({understat_league_ids})
            """)
        )
        row = res.first()
        with_data = int(row[0] or 0) if row else 0
        total = int(row[1] or 0) if row else 0
        coverage_pct = round(with_data / total * 100, 1) if total > 0 else 0.0

        # Get staleness (latest captured_at)
        res_stale = await session.execute(
            text("""
                SELECT MAX(captured_at) FROM match_understat_team
                WHERE captured_at > NOW() - INTERVAL '7 days'
            """)
        )
        latest_capture = res_stale.scalar()
        staleness_hours = None
        if latest_capture:
            staleness_hours = round((now - latest_capture).total_seconds() / 3600, 1)

        result["understat"] = {
            "status": "ok" if coverage_pct >= 50 else ("warn" if coverage_pct >= 20 else "red"),
            "coverage_pct": coverage_pct,
            "total": total,
            "with_data": with_data,
            "staleness_hours": staleness_hours,
            "note": "FT matches last 14d (top 5 leagues)",
        }
    except Exception as e:
        logger.debug(f"[SOTA] Understat metrics unavailable: {e}")
        result["understat"] = _unavailable(f"Query failed: {str(e)[:50]}")

    # 2) Weather coverage: NS matches in next 48h with weather forecasts
    try:
        res = await session.execute(
            text("""
                SELECT
                    COUNT(*) FILTER (WHERE mw.match_id IS NOT NULL) AS with_weather,
                    COUNT(*) AS total_ns
                FROM matches m
                LEFT JOIN match_weather mw ON m.id = mw.match_id
                WHERE m.status = 'NS'
                  AND m.date >= NOW()
                  AND m.date < NOW() + INTERVAL '48 hours'
            """)
        )
        row = res.first()
        with_data = int(row[0] or 0) if row else 0
        total = int(row[1] or 0) if row else 0
        coverage_pct = round(with_data / total * 100, 1) if total > 0 else 0.0

        result["weather"] = {
            "status": "ok" if coverage_pct >= 50 else ("warn" if coverage_pct >= 10 else "red"),
            "coverage_pct": coverage_pct,
            "total": total,
            "with_data": with_data,
            "staleness_hours": None,  # weather is forward-looking, no staleness
            "note": "NS matches next 48h",
        }
    except Exception as e:
        logger.debug(f"[SOTA] Weather metrics unavailable: {e}")
        result["weather"] = _unavailable(f"Query failed: {str(e)[:50]}")

    # 3) Venue geo coverage: venues from recent matches with coordinates
    try:
        res = await session.execute(
            text("""
                SELECT
                    COUNT(DISTINCT vg.venue_city) AS with_geo,
                    COUNT(DISTINCT m.venue_city) AS total_venues
                FROM matches m
                LEFT JOIN venue_geo vg ON m.venue_city = vg.venue_city
                WHERE m.venue_city IS NOT NULL
                  AND m.date >= NOW() - INTERVAL '30 days'
            """)
        )
        row = res.first()
        with_data = int(row[0] or 0) if row else 0
        total = int(row[1] or 0) if row else 0
        coverage_pct = round(with_data / total * 100, 1) if total > 0 else 0.0

        result["venue_geo"] = {
            "status": "ok" if coverage_pct >= 50 else ("warn" if coverage_pct >= 20 else "red"),
            "coverage_pct": coverage_pct,
            "total": total,
            "with_data": with_data,
            "staleness_hours": None,  # static data, no staleness
            "note": "Venues from matches last 30d",
        }
    except Exception as e:
        logger.debug(f"[SOTA] Venue geo metrics unavailable: {e}")
        result["venue_geo"] = _unavailable(f"Query failed: {str(e)[:50]}")

    # 4) Team profiles coverage: teams with home city profiles
    # ABE 2026-01-25: Report both "all teams" and "active teams (30d)" for clarity
    try:
        # Query 1: All teams (historical debt metric)
        res_all = await session.execute(
            text("""
                SELECT
                    COUNT(*) FILTER (WHERE thcp.team_id IS NOT NULL) AS with_profile,
                    COUNT(*) AS total_teams
                FROM teams t
                LEFT JOIN team_home_city_profile thcp ON t.id = thcp.team_id
            """)
        )
        row_all = res_all.first()
        with_data_all = int(row_all[0] or 0) if row_all else 0
        total_all = int(row_all[1] or 0) if row_all else 0
        coverage_pct_all = round(with_data_all / total_all * 100, 1) if total_all > 0 else 0.0

        # Query 2: Active teams (last 30d) - operational metric for dashboard card
        res_active = await session.execute(
            text("""
                SELECT
                    COUNT(DISTINCT CASE WHEN thcp.team_id IS NOT NULL THEN t.id END) AS with_profile,
                    COUNT(DISTINCT t.id) AS total_teams
                FROM teams t
                JOIN matches m ON t.id = m.home_team_id OR t.id = m.away_team_id
                LEFT JOIN team_home_city_profile thcp ON t.id = thcp.team_id
                WHERE m.date >= NOW() - INTERVAL '30 days'
            """)
        )
        row_active = res_active.first()
        with_data_active = int(row_active[0] or 0) if row_active else 0
        total_active = int(row_active[1] or 0) if row_active else 0
        coverage_pct_active = round(with_data_active / total_active * 100, 1) if total_active > 0 else 0.0

        # Primary metric: active teams (better operational signal)
        result["team_profiles"] = {
            "status": "ok" if coverage_pct_active >= 30 else ("warn" if coverage_pct_active >= 10 else "red"),
            "coverage_pct": coverage_pct_active,
            "total": total_active,
            "with_data": with_data_active,
            "staleness_hours": None,  # static data, no staleness
            "note": f"Active teams (30d). All teams: {with_data_all}/{total_all} ({coverage_pct_all}%)",
            # Detailed breakdown for audit
            "all_teams": {
                "with_data": with_data_all,
                "total": total_all,
                "coverage_pct": coverage_pct_all,
            },
            "active_teams_30d": {
                "with_data": with_data_active,
                "total": total_active,
                "coverage_pct": coverage_pct_active,
            },
        }
    except Exception as e:
        logger.debug(f"[SOTA] Team profiles metrics unavailable: {e}")
        result["team_profiles"] = _unavailable(f"Query failed: {str(e)[:50]}")

    # 5) Sofascore XI coverage: NS matches in next 48h with XI data
    try:
        # Check if tables exist first (they may not be deployed yet)
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
                text(f"""
                    SELECT
                        COUNT(*) FILTER (WHERE msl.match_id IS NOT NULL) AS with_xi,
                        COUNT(*) AS total_ns
                    FROM matches m
                    LEFT JOIN match_sofascore_lineup msl ON m.id = msl.match_id
                    WHERE m.status = 'NS'
                      AND m.date >= NOW()
                      AND m.date < NOW() + INTERVAL '48 hours'
                      AND m.league_id IN ({sofascore_league_ids})
                """)
            )
            row = res.first()
            with_data = int(row[0] or 0) if row else 0
            total = int(row[1] or 0) if row else 0
            coverage_pct = round(with_data / total * 100, 1) if total > 0 else 0.0

            # Get staleness (latest captured_at)
            res_stale = await session.execute(
                text("""
                    SELECT MAX(captured_at) FROM match_sofascore_lineup
                    WHERE captured_at > NOW() - INTERVAL '7 days'
                """)
            )
            latest_capture = res_stale.scalar()
            staleness_hours = None
            if latest_capture:
                staleness_hours = round((now - latest_capture).total_seconds() / 3600, 1)

            result["sofascore_xi"] = {
                "status": "ok" if coverage_pct >= 30 else ("warn" if coverage_pct >= 10 else "red"),
                "coverage_pct": coverage_pct,
                "total": total,
                "with_data": with_data,
                "staleness_hours": staleness_hours,
                "note": "NS matches next 48h (SOTA leagues)",
            }
        else:
            # Tables don't exist yet - waiting for migration
            result["sofascore_xi"] = _unavailable("Tables not deployed yet (migration 030)")
    except Exception as e:
        logger.debug(f"[SOTA] Sofascore XI metrics unavailable: {e}")
        result["sofascore_xi"] = _unavailable(f"Query failed: {str(e)[:50]}")

    # Overall status: worst of components (excluding unavailable)
    component_statuses = []
    for key in ["understat", "weather", "venue_geo", "team_profiles", "sofascore_xi"]:
        if result.get(key, {}).get("status") in ("ok", "warn", "red"):
            component_statuses.append(result[key]["status"])

    if "red" in component_statuses:
        result["status"] = "red"
    elif "warn" in component_statuses:
        result["status"] = "warn"
    elif component_statuses:
        result["status"] = "ok"
    else:
        result["status"] = "unavailable"

    return result


async def _calculate_titan_summary() -> dict:
    """
    Calculate TITAN OMNISCIENCE summary for OPS dashboard.

    Reports:
    - feature_matrix row count and tier coverage
    - Job status (last run, success/fail)
    - Progress toward N=50/200 gate

    Lightweight query for ops.json inclusion.
    Creates its own session to avoid scope issues.
    """
    now = datetime.utcnow()
    result = {
        "status": "ok",
        "generated_at": now.isoformat(),
        "feature_matrix": {},
        "job": {},
        "gate": {},
    }

    try:
        async with AsyncSessionLocal() as session:
            # Check if titan schema exists
            schema_check = await session.execute(text("""
                SELECT EXISTS (
                    SELECT FROM information_schema.schemata
                    WHERE schema_name = 'titan'
                )
            """))
            schema_exists = schema_check.scalar()

            if not schema_exists:
                result["status"] = "unavailable"
                result["note"] = "TITAN schema not deployed"
                return result

            # Feature matrix stats
            fm_stats = await session.execute(text("""
                SELECT
                    COUNT(*) as total,
                    COUNT(*) FILTER (WHERE tier1_complete = TRUE) as tier1,
                    COUNT(*) FILTER (WHERE tier1b_complete = TRUE) as tier1b,
                    COUNT(*) FILTER (WHERE tier1c_complete = TRUE) as tier1c,
                    COUNT(*) FILTER (WHERE tier1d_complete = TRUE) as tier1d,
                    COUNT(*) FILTER (WHERE outcome IS NOT NULL) as with_outcome
                FROM titan.feature_matrix
            """))
            row = fm_stats.first()
            if row:
                total = int(row[0] or 0)
                tier1 = int(row[1] or 0)
                tier1b = int(row[2] or 0)
                tier1c = int(row[3] or 0)
                tier1d = int(row[4] or 0)
                with_outcome = int(row[5] or 0)

                result["feature_matrix"] = {
                    "total_rows": total,
                    "tier1_complete": tier1,
                    "tier1b_complete": tier1b,
                    "tier1c_complete": tier1c,
                    "tier1d_complete": tier1d,
                    "with_outcome": with_outcome,
                    "tier1b_pct": round(tier1b / total * 100, 1) if total > 0 else 0,
                    "tier1c_pct": round(tier1c / total * 100, 1) if total > 0 else 0,
                }

                # Gate progress (ABE-defined thresholds)
                result["gate"] = {
                    "n_current": with_outcome,
                    "n_target_pilot": 50,
                    "n_target_prelim": 200,
                    "n_target_formal": 500,
                    "ready_for_pilot": with_outcome >= 50,
                    "ready_for_prelim": with_outcome >= 200,
                    "ready_for_formal": with_outcome >= 500,
                    "pct_to_pilot": round(min(100, with_outcome / 50 * 100), 1),
                    "pct_to_prelim": round(min(100, with_outcome / 200 * 100), 1),
                    "pct_to_formal": round(min(100, with_outcome / 500 * 100), 1),
                }

            # Job status (last TITAN runner run)
            job_stats = await session.execute(text("""
                SELECT status, started_at, metrics
                FROM job_runs
                WHERE job_name = 'titan_feature_matrix_runner'
                ORDER BY started_at DESC
                LIMIT 1
            """))
            job_row = job_stats.first()
            if job_row:
                result["job"] = {
                    "last_status": job_row[0],
                    "last_run_at": job_row[1].isoformat() if job_row[1] else None,
                    "last_metrics": job_row[2] if job_row[2] else {},
                }
            else:
                result["job"] = {
                    "last_status": "never_run",
                    "last_run_at": None,
                    "note": "Job has not run yet - will start on next 2h interval",
                }

            # Determine overall status
            if result["feature_matrix"].get("total_rows", 0) == 0:
                result["status"] = "warn"
                result["note"] = "No data in feature_matrix yet"
            elif result["gate"].get("ready_for_formal"):
                result["status"] = "ok"
                result["note"] = f"Ready for formal eval (N={with_outcome})"
            elif result["gate"].get("ready_for_prelim"):
                result["status"] = "ok"
                result["note"] = f"Ready for preliminary eval (N={with_outcome}/500)"
            elif result["gate"].get("ready_for_pilot"):
                result["status"] = "ok"
                result["note"] = f"Ready for pilot eval (N={with_outcome}/500)"
            else:
                result["status"] = "building"
                result["note"] = f"Accumulating data: {with_outcome}/500 for formal gate"

    except Exception as e:
        logger.warning(f"[TITAN] Summary calculation failed: {e}")
        result["status"] = "error"
        result["error"] = str(e)[:100]

    return result


async def _calculate_rerun_serving_summary(session) -> dict:
    """
    Calculate rerun serving metrics for OPS dashboard.

    Shows canary status: how many predictions are served from DB (two-stage)
    vs live baseline, plus active rerun info.
    """
    settings = get_settings()
    try:
        # Check if enabled
        enabled = settings.PREFER_RERUN_PREDICTIONS
        freshness_hours = settings.RERUN_FRESHNESS_HOURS

        # Get active rerun info
        res = await session.execute(
            text("""
                SELECT run_id, architecture_after, model_version_after,
                       matches_total, created_at
                FROM prediction_reruns
                WHERE is_active = true
                ORDER BY created_at DESC
                LIMIT 1
            """)
        )
        active_rerun = res.fetchone()

        # Count NS matches with rerun predictions (fresh)
        res = await session.execute(
            text("""
                SELECT COUNT(DISTINCT p.match_id)
                FROM predictions p
                JOIN matches m ON m.id = p.match_id
                WHERE p.run_id IS NOT NULL
                  AND m.status = 'NS'
                  AND m.date > NOW()
                  AND p.created_at > NOW() - make_interval(hours => :hours)
            """),
            {"hours": freshness_hours * 2}  # Use 2x freshness for counting
        )
        ns_with_rerun = int(res.scalar() or 0)

        # Total NS matches in window
        res = await session.execute(
            text("""
                SELECT COUNT(*)
                FROM matches
                WHERE status = 'NS'
                  AND date > NOW()
                  AND date < NOW() + INTERVAL '7 days'
            """)
        )
        total_ns = int(res.scalar() or 0)

        # Get in-memory serving stats
        from_rerun_pct = round(100.0 * ns_with_rerun / total_ns, 1) if total_ns > 0 else 0.0
        from_baseline_pct = round(100.0 - from_rerun_pct, 1)

        return {
            "enabled": enabled,
            "freshness_hours": freshness_hours,
            "active_rerun": {
                "run_id": str(active_rerun[0]) if active_rerun else None,
                "architecture": active_rerun[1] if active_rerun else None,
                "model_version": active_rerun[2] if active_rerun else None,
                "matches_total": active_rerun[3] if active_rerun else None,
                "created_at": active_rerun[4].isoformat() if active_rerun else None,
            } if active_rerun else None,
            "coverage": {
                "ns_with_rerun": ns_with_rerun,
                "total_ns_7d": total_ns,
                "from_rerun_pct": from_rerun_pct,
                "from_baseline_pct": from_baseline_pct,
            },
            "in_memory_stats": _rerun_serving_stats.copy(),
        }
    except Exception as e:
        logger.warning(f"Rerun serving summary failed: {e}")
        return {
            "enabled": settings.PREFER_RERUN_PREDICTIONS,
            "error": str(e)[:100],
        }


async def _calculate_predictions_health(session) -> dict:
    """
    Calculate predictions health metrics for P0 observability.

    Detects when daily_save_predictions isn't running/persisting.
    Returns status: ok/warn/red based on recency and coverage.

    Smart logic: If there are no upcoming NS matches scheduled, we don't
    raise WARN/RED for stale predictions (false positive in low-activity periods).

    PERF: Single CTE query replaces 8 sequential queries (fixes N+1 pattern).
    """
    now = datetime.utcnow()

    # Single consolidated query using CTEs (8 queries → 1)
    res = await session.execute(
        text("""
            WITH
              pred_stats AS (
                SELECT
                  MAX(created_at) as last_pred_at,
                  COUNT(*) FILTER (WHERE created_at > NOW() - INTERVAL '24 hours') as preds_last_24h,
                  COUNT(*) FILTER (WHERE created_at::date = CURRENT_DATE) as preds_today
                FROM predictions
              ),
              ft_with_pred AS (
                SELECT
                  m.id,
                  p.id as pred_id
                FROM matches m
                LEFT JOIN predictions p ON p.match_id = m.id
                WHERE m.status IN ('FT', 'AET', 'PEN')
                  AND m.date > NOW() - INTERVAL '48 hours'
              ),
              ft_stats AS (
                SELECT
                  COUNT(*) as ft_48h,
                  COUNT(*) FILTER (WHERE pred_id IS NULL) as ft_48h_missing
                FROM ft_with_pred
              ),
              ns_with_pred AS (
                SELECT
                  m.id,
                  m.date,
                  p.id as pred_id
                FROM matches m
                LEFT JOIN predictions p ON p.match_id = m.id
                WHERE m.status = 'NS'
                  AND m.date > NOW()
              ),
              ns_stats AS (
                SELECT
                  COUNT(*) FILTER (WHERE date <= NOW() + INTERVAL '48 hours') as ns_next_48h,
                  COUNT(*) FILTER (WHERE date <= NOW() + INTERVAL '48 hours' AND pred_id IS NULL) as ns_next_48h_missing,
                  MIN(date) as next_ns_date
                FROM ns_with_pred
              )
            SELECT
              pred_stats.last_pred_at,
              pred_stats.preds_last_24h,
              pred_stats.preds_today,
              ft_stats.ft_48h,
              ft_stats.ft_48h_missing,
              ns_stats.ns_next_48h,
              ns_stats.ns_next_48h_missing,
              ns_stats.next_ns_date
            FROM pred_stats, ft_stats, ns_stats
        """)
    )
    row = res.fetchone()

    # Extract values from single row result
    last_pred_at = row[0]
    preds_last_24h = int(row[1] or 0)
    preds_today = int(row[2] or 0)
    ft_48h = int(row[3] or 0)
    ft_48h_missing = int(row[4] or 0)
    ns_next_48h = int(row[5] or 0)
    ns_next_48h_missing = int(row[6] or 0)
    next_ns_date = row[7]

    # Coverage percentages
    coverage_48h_pct = 0.0
    if ft_48h > 0:
        coverage_48h_pct = round(((ft_48h - ft_48h_missing) / ft_48h) * 100, 1)

    ns_coverage_pct = 100.0
    if ns_next_48h > 0:
        ns_coverage_pct = round(((ns_next_48h - ns_next_48h_missing) / ns_next_48h) * 100, 1)

    # Calculate hours since last prediction (informational only)
    hours_since_last = None
    if last_pred_at:
        delta = now - last_pred_at
        hours_since_last = round(delta.total_seconds() / 3600, 1)

    # Determine status with smart logic
    # Primary metric: NS coverage (do upcoming matches have predictions?)
    # Secondary metric: FT coverage (did past matches have predictions?)
    status = "ok"
    status_reason = None

    # Smart bypass: no upcoming matches = no expectation of predictions
    if ns_next_48h == 0:
        status = "ok"
        status_reason = "No upcoming NS matches in 48h (low activity period)"
    # NEW: Check if prediction job hasn't run recently (stale job detection)
    # P1: Only WARN for staleness if coverage is NOT 100% - if all NS and FT have predictions,
    # staleness is just informational (DAILY-SAVE runs once/day)
    elif hours_since_last and hours_since_last > 12 and ns_next_48h > 0:
        # Check if coverage is perfect - if so, staleness is just informational
        if ns_next_48h_missing == 0 and ft_48h_missing == 0:
            # Coverage is 100%, staleness is OK (just means daily job ran earlier)
            status = "ok"
            status_reason = None  # No alert needed
        else:
            status = "warn"
            status_reason = f"Predictions job stale: {hours_since_last:.1f}h since last save with {ns_next_48h} NS upcoming"
    # Primary check: upcoming NS matches should have predictions
    elif ns_coverage_pct < 50:
        status = "red"
        status_reason = f"NS coverage {ns_coverage_pct}% < 50% ({ns_next_48h_missing}/{ns_next_48h} missing)"
    elif ns_coverage_pct < 80:
        status = "warn"
        status_reason = f"NS coverage {ns_coverage_pct}% < 80% ({ns_next_48h_missing}/{ns_next_48h} missing)"
    # Secondary check: past FT matches coverage
    elif coverage_48h_pct < 50:
        status = "red"
        status_reason = f"FT coverage {coverage_48h_pct}% < 50% threshold"
    elif coverage_48h_pct < 80:
        status = "warn"
        status_reason = f"FT coverage {coverage_48h_pct}% < 80% threshold"

    # Log OPS_ALERT if red/warn (rate-limited to avoid spam)
    global _predictions_health_alert_last
    import time as _time
    now_ts = _time.time()

    if status in ("red", "warn") and (now_ts - _predictions_health_alert_last) > _PREDICTIONS_HEALTH_ALERT_COOLDOWN:
        _predictions_health_alert_last = now_ts
        if status == "red":
            logger.error(
                f"[OPS_ALERT] predictions_health=RED: {status_reason}. "
                f"last_pred={last_pred_at}, preds_24h={preds_last_24h}, "
                f"ft_48h={ft_48h}, missing={ft_48h_missing}, ns_next_48h={ns_next_48h}"
            )
        else:
            logger.warning(
                f"[OPS_ALERT] predictions_health=WARN: {status_reason}. "
                f"last_pred={last_pred_at}, preds_24h={preds_last_24h}, ns_next_48h={ns_next_48h}"
            )

    # Emit Prometheus metrics for Grafana alerting (P1)
    try:
        from app.telemetry.metrics import set_predictions_health_metrics
        set_predictions_health_metrics(
            hours_since_last=hours_since_last,
            ns_next_48h=ns_next_48h,
            ns_missing_next_48h=ns_next_48h_missing,
            coverage_ns_pct=ns_coverage_pct,
            status=status,
        )
    except Exception as e:
        logger.debug(f"Failed to emit predictions health metrics: {e}")

    return {
        "status": status,
        "status_reason": status_reason,
        # NS (upcoming) metrics - primary
        "ns_matches_next_48h": ns_next_48h,
        "ns_matches_next_48h_missing_prediction": ns_next_48h_missing,
        "ns_coverage_pct": ns_coverage_pct,
        "next_ns_match_utc": next_ns_date.isoformat() if next_ns_date else None,
        # FT (past) metrics - secondary
        "ft_matches_last_48h": ft_48h,
        "ft_matches_last_48h_missing_prediction": ft_48h_missing,
        "ft_coverage_pct": coverage_48h_pct,
        # Informational (not used for status determination)
        "last_prediction_saved_at": last_pred_at.isoformat() if last_pred_at else None,
        "hours_since_last_prediction": hours_since_last,
        "predictions_saved_last_24h": preds_last_24h,
        "predictions_saved_today_utc": preds_today,
        "thresholds": {
            "ns_coverage_warn_pct": 80,
            "ns_coverage_red_pct": 50,
            "ft_coverage_warn_pct": 80,
            "ft_coverage_red_pct": 50,
        },
    }


async def _calculate_model_performance(session) -> dict:
    """
    Calculate model performance summary for OPS dashboard card.

    Returns compact summary from the latest 7d performance report:
    - Brier score
    - Skill vs market
    - Recommendation (OK/WATCH/INVESTIGATE)
    - Status color (green/yellow/red)
    """
    from app.ml.performance_metrics import get_latest_report

    try:
        report = await get_latest_report(session, window_days=7)

        if not report:
            return {
                "status": "gray",
                "status_reason": "No report available yet",
                "brier_score": None,
                "skill_vs_market": None,
                "recommendation": None,
                "n_predictions": 0,
                "confidence": "none",
                "report_generated_at": None,
            }

        global_metrics = report.get("global", {})
        metrics = global_metrics.get("metrics", {})
        diagnostics = report.get("diagnostics", {})
        market = metrics.get("market_comparison", {})

        brier = metrics.get("brier_score")
        skill = market.get("skill_vs_market") if market else None
        recommendation = diagnostics.get("recommendation", "OK")
        n = global_metrics.get("n", 0)
        confidence = report.get("confidence", "low")

        # Determine status color based on recommendation
        if "INVESTIGATE" in recommendation:
            status = "red"
        elif "MONITOR" in recommendation or "WATCH" in recommendation:
            status = "yellow"
        else:
            status = "green"

        # Format skill for display
        skill_display = None
        if skill is not None:
            skill_display = f"{skill * 100:+.1f}%"

        return {
            "status": status,
            "status_reason": recommendation,
            "brier_score": round(brier, 4) if brier else None,
            "skill_vs_market": skill_display,
            "skill_vs_market_raw": round(skill, 4) if skill is not None else None,
            "recommendation": recommendation,
            "n_predictions": n,
            "confidence": confidence,
            "report_generated_at": report.get("generated_at"),
            "endpoint": "/dashboard/ops/predictions_performance.json?window_days=7",
        }

    except Exception as e:
        logger.warning(f"Error calculating model performance: {e}")
        return {
            "status": "gray",
            "status_reason": f"Error: {str(e)[:50]}",
            "brier_score": None,
            "skill_vs_market": None,
            "recommendation": None,
            "n_predictions": 0,
            "confidence": "error",
            "report_generated_at": None,
        }


async def _calculate_fastpath_health(session) -> dict:
    """
    Calculate fast-path LLM narrative health metrics.

    Monitors the fast-path job that generates narratives within minutes
    of match completion instead of waiting for daily audit.

    Returns status: ok/warn/red based on tick recency and error rates.
    """
    from app.config import get_settings

    settings = get_settings()
    now = datetime.utcnow()

    # Read ticks from DB (canonical source - survives restarts/multi-process)
    last_tick_at = None
    last_tick_result = {}
    ticks_total = 0
    ticks_with_activity = 0
    db_unavailable = False

    try:
        # PERF: Single query for tick stats (2 queries → 1)
        res = await session.execute(
            text("""
                WITH recent_tick AS (
                    SELECT tick_at, selected, refreshed, ready, enqueued, completed, errors, skipped
                    FROM fastpath_ticks
                    ORDER BY tick_at DESC
                    LIMIT 1
                ),
                hour_stats AS (
                    SELECT
                        COUNT(*) as total,
                        COUNT(*) FILTER (WHERE selected > 0 OR enqueued > 0 OR completed > 0) as with_activity
                    FROM fastpath_ticks
                    WHERE tick_at > NOW() - INTERVAL '1 hour'
                )
                SELECT
                    rt.tick_at, rt.selected, rt.refreshed, rt.ready, rt.enqueued, rt.completed, rt.errors, rt.skipped,
                    hs.total, hs.with_activity
                FROM hour_stats hs
                LEFT JOIN recent_tick rt ON true
            """)
        )
        row = res.fetchone()
        if row:
            if row[0]:  # tick_at exists
                last_tick_at = row[0]
                last_tick_result = {
                    "selected": row[1], "refreshed": row[2], "stats_ready": row[3],
                    "enqueued": row[4], "completed": row[5], "errors": row[6], "skipped": row[7]
                }
            ticks_total = row[8] or 0
            ticks_with_activity = row[9] or 0
    except Exception as db_err:
        # DB unavailable - mark as red status (don't use in-memory fallback in prod)
        logger.warning(f"fastpath_ticks DB unavailable: {db_err}")
        db_unavailable = True

    # Check if fast-path is enabled
    enabled = os.environ.get("FASTPATH_ENABLED", str(settings.FASTPATH_ENABLED)).lower()
    is_enabled = enabled not in ("false", "0", "no")

    # Calculate minutes since last tick
    minutes_since_tick = None
    if last_tick_at:
        delta = now - last_tick_at
        minutes_since_tick = round(delta.total_seconds() / 60, 1)

    # Query DB for LLM stats in last 60 minutes
    llm_60m = {"ok": 0, "ok_retry": 0, "error": 0, "skipped": 0, "in_queue": 0, "running": 0}
    error_codes_60m = {}
    pending_ready = 0

    try:
        # LLM status breakdown (last 60 min)
        res = await session.execute(
            text("""
                SELECT llm_narrative_status, COUNT(*) as cnt
                FROM post_match_audits
                WHERE created_at > NOW() - INTERVAL '60 minutes'
                  AND llm_narrative_status IS NOT NULL
                GROUP BY llm_narrative_status
            """)
        )
        for row in res.fetchall():
            status_key = row[0] or "unknown"
            llm_60m[status_key] = int(row[1])

        # Error codes breakdown (last 60 min)
        res = await session.execute(
            text("""
                SELECT llm_narrative_error_code, COUNT(*) as cnt
                FROM post_match_audits
                WHERE created_at > NOW() - INTERVAL '60 minutes'
                  AND llm_narrative_error_code IS NOT NULL
                GROUP BY llm_narrative_error_code
                ORDER BY cnt DESC
                LIMIT 5
            """)
        )
        for row in res.fetchall():
            error_codes_60m[row[0]] = int(row[1])

        # Pending ready: FT matches with stats_ready but no successful narrative
        # Use COALESCE(finished_at, date) to align with fast-path selector logic
        res = await session.execute(
            text("""
                SELECT COUNT(*)
                FROM matches m
                WHERE m.status IN ('FT', 'AET', 'PEN')
                  AND COALESCE(m.finished_at, m.date) > NOW() - INTERVAL '180 minutes'
                  AND m.stats_ready_at IS NOT NULL
                  AND NOT EXISTS (
                      SELECT 1 FROM prediction_outcomes po
                      JOIN post_match_audits pma ON pma.outcome_id = po.id
                      WHERE po.match_id = m.id
                        AND pma.llm_narrative_status = 'ok'
                  )
            """)
        )
        pending_ready = int(res.scalar() or 0)

    except Exception as e:
        logger.warning(f"Error calculating fastpath_health DB metrics: {e}")

    # Calculate total errors and success
    total_ok = llm_60m.get("ok", 0) + llm_60m.get("ok_retry", 0)
    total_errors = llm_60m.get("error", 0)
    total_processed = total_ok + total_errors + llm_60m.get("skipped", 0)
    error_rate_60m = 0.0
    if total_processed > 0:
        error_rate_60m = round((total_errors / total_processed) * 100, 1)

    # Determine status
    status = "ok"
    status_reason = None

    if db_unavailable:
        status = "red"
        status_reason = "fastpath_ticks table unavailable"
    elif not is_enabled:
        status = "disabled"
        status_reason = "FASTPATH_ENABLED=false"
    elif minutes_since_tick is None:
        status = "warn"
        status_reason = "No tick recorded yet (job may not have run)"
    elif minutes_since_tick > 10:
        status = "red"
        status_reason = f"No tick in {minutes_since_tick:.0f} min (>10 min threshold)"
    elif error_rate_60m > 50:
        status = "red"
        status_reason = f"Error rate {error_rate_60m}% > 50% threshold"
    elif error_rate_60m > 20:
        status = "warn"
        status_reason = f"Error rate {error_rate_60m}% > 20% threshold"
    elif total_ok == 0 and llm_60m.get("skipped", 0) > 5:
        status = "warn"
        status_reason = f"0 ok, {llm_60m.get('skipped', 0)} skipped (gating issues?)"

    return {
        "status": status,
        "status_reason": status_reason,
        "enabled": is_enabled,
        "last_tick_at": last_tick_at.isoformat() if last_tick_at else None,
        "minutes_since_tick": minutes_since_tick,
        "last_tick_result": last_tick_result,
        "ticks_total": ticks_total,
        "ticks_with_activity": ticks_with_activity,
        "last_60m": {
            "ok": llm_60m.get("ok", 0),
            "ok_retry": llm_60m.get("ok_retry", 0),
            "error": llm_60m.get("error", 0),
            "skipped": llm_60m.get("skipped", 0),
            "in_queue": llm_60m.get("in_queue", 0),
            "running": llm_60m.get("running", 0),
            "total_processed": total_processed,
            "error_rate_pct": error_rate_60m,
        },
        "top_error_codes_60m": error_codes_60m,
        "pending_ready": pending_ready,
        "config": {
            "interval_seconds": settings.FASTPATH_INTERVAL_SECONDS,
            "lookback_minutes": settings.FASTPATH_LOOKBACK_MINUTES,
            "max_concurrent_jobs": settings.FASTPATH_MAX_CONCURRENT_JOBS,
        },
    }


@app.get("/dashboard/ops/fastpath_debug.json")
async def fastpath_debug_endpoint(
    request: Request,
    session: AsyncSession = Depends(get_async_session),
):
    """Debug endpoint to see skipped audits and their reasons."""
    _verify_debug_token(request)

    try:
        # Get skipped audits from last 60 min
        res = await session.execute(
            text("""
                SELECT
                    pma.id as audit_id,
                    po.match_id,
                    pma.llm_narrative_status,
                    pma.llm_narrative_error_code,
                    pma.llm_narrative_error_detail,
                    pma.created_at,
                    m.home_goals,
                    m.away_goals,
                    m.stats IS NOT NULL as has_stats,
                    m.stats_ready_at IS NOT NULL as stats_ready
                FROM post_match_audits pma
                JOIN prediction_outcomes po ON po.id = pma.outcome_id
                JOIN matches m ON m.id = po.match_id
                WHERE pma.created_at > NOW() - INTERVAL '60 minutes'
                  AND pma.llm_narrative_status = 'skipped'
                ORDER BY pma.created_at DESC
                LIMIT 20
            """)
        )
        skipped = []
        for r in res.fetchall():
            skipped.append({
                "audit_id": r[0],
                "match_id": r[1],
                "status": r[2],
                "error_code": r[3],
                "error_detail": r[4],
                "created_at": r[5].isoformat() if r[5] else None,
                "goals": f"{r[6]}-{r[7]}",
                "has_stats": r[8],
                "stats_ready": r[9],
            })

        # Get status breakdown
        res2 = await session.execute(
            text("""
                SELECT llm_narrative_status, COUNT(*)
                FROM post_match_audits
                WHERE created_at > NOW() - INTERVAL '60 minutes'
                GROUP BY llm_narrative_status
                ORDER BY COUNT(*) DESC
            """)
        )
        breakdown = {r[0]: r[1] for r in res2.fetchall()}

        return {
            "skipped_audits": skipped,
            "status_breakdown_60m": breakdown,
        }
    except Exception as e:
        logger.error(f"fastpath_debug error: {e}")
        return {"error": str(e)}


@app.get("/dashboard/ops/llm_audit/{match_id}.json")
async def llm_audit_endpoint(
    request: Request,
    match_id: int,
    session: AsyncSession = Depends(get_async_session),
):
    """
    Debug endpoint for LLM traceability.

    Returns the exact payload sent to Qwen for a specific match,
    allowing quick RCA for hallucination issues.
    """
    _verify_debug_token(request)

    try:
        # Get audit with traceability data
        res = await session.execute(
            text("""
                SELECT
                    pma.id as audit_id,
                    po.match_id,
                    pma.llm_prompt_version,
                    pma.llm_prompt_input_hash,
                    pma.llm_prompt_input_json,
                    pma.llm_output_raw,
                    pma.llm_validation_errors,
                    pma.llm_narrative_status,
                    pma.llm_narrative_error_code,
                    pma.llm_narrative_error_detail,
                    pma.llm_narrative_request_id,
                    pma.llm_narrative_json,
                    pma.llm_narrative_generated_at,
                    pma.llm_narrative_tokens_in,
                    pma.llm_narrative_tokens_out,
                    pma.llm_narrative_exec_ms,
                    pma.created_at
                FROM post_match_audits pma
                JOIN prediction_outcomes po ON po.id = pma.outcome_id
                WHERE po.match_id = :match_id
            """),
            {"match_id": match_id}
        )
        row = res.fetchone()

        if not row:
            return {"error": f"No audit found for match_id {match_id}"}

        return {
            "audit_id": row[0],
            "match_id": row[1],
            "prompt_version": row[2],
            "prompt_input_hash": row[3],
            "prompt_input_json": row[4],
            "output_raw_preview": row[5][:500] if row[5] else None,
            "output_raw_len": len(row[5]) if row[5] else 0,
            "validation_errors": row[6],
            "status": row[7],
            "error_code": row[8],
            "error_detail": row[9],
            "runpod_job_id": row[10],
            "narrative_json": row[11],
            "generated_at": row[12].isoformat() if row[12] else None,
            "tokens_in": row[13],
            "tokens_out": row[14],
            "exec_ms": row[15],
            "audit_created_at": row[16].isoformat() if row[16] else None,
        }
    except Exception as e:
        logger.error(f"llm_audit error for match {match_id}: {e}")
        return {"error": str(e)}


@app.get("/dashboard/ops/match_data.json")
async def match_data_debug_endpoint(
    request: Request,
    match_id: int = Query(...),
    session: AsyncSession = Depends(get_async_session),
):
    """Debug endpoint to see exact match_data sent to LLM."""
    _verify_debug_token(request)

    try:
        match = await session.get(Match, match_id)
        if not match:
            return {"error": f"Match {match_id} not found"}

        home_team = await session.get(Team, match.home_team_id)
        away_team = await session.get(Team, match.away_team_id)

        # Get prediction
        pred_result = await session.execute(
            select(Prediction)
            .where(Prediction.match_id == match_id)
            .order_by(Prediction.created_at.desc())
            .limit(1)
        )
        prediction = pred_result.scalar_one_or_none()

        # Build the exact match_data that would be sent to LLM
        probs = {}
        predicted_result = None
        confidence = None
        if prediction:
            probs = {
                "home": prediction.home_prob,
                "draw": prediction.draw_prob,
                "away": prediction.away_prob,
            }
            predicted_result = max(probs, key=probs.get)
            confidence = probs[predicted_result]

        home_goals = match.home_goals or 0
        away_goals = match.away_goals or 0
        if home_goals > away_goals:
            actual_result = "home"
        elif away_goals > home_goals:
            actual_result = "away"
        else:
            actual_result = "draw"

        match_data = {
            "match_id": match.id,
            "home_team": home_team.name if home_team else "Local",
            "away_team": away_team.name if away_team else "Visitante",
            "league_name": "",
            "date": match.date.isoformat() if match.date else "",
            "home_goals": home_goals,
            "away_goals": away_goals,
            "stats": match.stats or {},
            "events": match.events or [],
            "prediction": {
                "probabilities": probs,
                "predicted_result": predicted_result,
                "confidence": confidence,
                "correct": predicted_result == actual_result if predicted_result else None,
            },
            "market_odds": {
                "home": match.odds_home,
                "draw": match.odds_draw,
                "away": match.odds_away,
            } if match.odds_home else {},
        }

        return {
            "match_data_sent_to_llm": match_data,
            "raw_match_stats": match.stats,
            "stats_ready_at": match.stats_ready_at.isoformat() if match.stats_ready_at else None,
            "stats_last_checked_at": match.stats_last_checked_at.isoformat() if match.stats_last_checked_at else None,
        }
    except Exception as e:
        logger.error(f"match_data_debug error: {e}")
        return {"error": str(e)}


@app.get("/dashboard/ops/stats_rca.json")
async def stats_rca_endpoint(
    request: Request,
    match_id: int = Query(...),
    session: AsyncSession = Depends(get_async_session),
):
    """
    RCA endpoint: fetch stats from API-Football and show full diagnostic.
    Tests: API response, parsing, persistence.
    """
    _verify_debug_token(request)

    from app.etl.api_football import APIFootballProvider

    result = {
        "match_id": match_id,
        "steps": {},
        "diagnosis": None,
    }

    try:
        # Step 1: Get match from DB
        match = await session.get(Match, match_id)
        if not match:
            return {"error": f"Match {match_id} not found"}

        result["steps"]["1_match_info"] = {
            "id": match.id,
            "external_id": match.external_id,
            "stats_before": match.stats,
            "stats_ready_at": str(match.stats_ready_at) if match.stats_ready_at else None,
        }

        if not match.external_id:
            result["diagnosis"] = "NO_EXTERNAL_ID"
            return result

        # Step 2: Fetch from API-Football
        provider = APIFootballProvider()
        try:
            stats_data = await provider._rate_limited_request(
                "fixtures/statistics",
                {"fixture": match.external_id},
                entity="stats"
            )
            await provider.close()
        except Exception as api_err:
            result["steps"]["2_api_call"] = {"error": str(api_err)}
            result["diagnosis"] = "API_CALL_FAILED"
            return result

        response = stats_data.get("response", [])
        result["steps"]["2_api_call"] = {
            "raw_response_keys": list(stats_data.keys()),
            "response_len": len(response),
            "response_teams": [r.get("team", {}).get("name") for r in response] if response else [],
        }

        if len(response) < 2:
            result["diagnosis"] = "API_RESPONSE_EMPTY_OR_INCOMPLETE"
            result["steps"]["2_api_call"]["raw_response"] = stats_data
            return result

        # Step 3: Show raw statistics structure
        result["steps"]["3_raw_stats_structure"] = {
            "team_0_name": response[0].get("team", {}).get("name"),
            "team_0_statistics_count": len(response[0].get("statistics", [])),
            "team_0_statistics_sample": response[0].get("statistics", [])[:5],
            "team_1_name": response[1].get("team", {}).get("name"),
            "team_1_statistics_count": len(response[1].get("statistics", [])),
            "team_1_statistics_sample": response[1].get("statistics", [])[:5],
        }

        # Step 4: Parse stats using our key_map
        key_map = {
            "Ball Possession": "ball_possession",
            "Total Shots": "total_shots",
            "Shots on Goal": "shots_on_goal",
            "Shots off Goal": "shots_off_goal",
            "Blocked Shots": "blocked_shots",
            "Shots insidebox": "shots_insidebox",
            "Shots outsidebox": "shots_outsidebox",
            "Fouls": "fouls",
            "Corner Kicks": "corner_kicks",
            "Offsides": "offsides",
            "Yellow Cards": "yellow_cards",
            "Red Cards": "red_cards",
            "Goalkeeper Saves": "goalkeeper_saves",
            "Total passes": "total_passes",
            "Passes accurate": "passes_accurate",
            "Passes %": "passes_pct",
            "expected_goals": "expected_goals",
        }

        def parse_team_stats(stats_list):
            parsed = {}
            for stat in stats_list:
                stat_type = stat.get("type")
                stat_value = stat.get("value")
                if stat_type in key_map:
                    parsed[key_map[stat_type]] = stat_value
            return parsed

        home_stats = parse_team_stats(response[0].get("statistics", []))
        away_stats = parse_team_stats(response[1].get("statistics", []))

        result["steps"]["4_parsed_stats"] = {
            "home_stats": home_stats,
            "away_stats": away_stats,
            "home_keys": list(home_stats.keys()),
            "away_keys": list(away_stats.keys()),
        }

        # Step 5: Test persistence
        new_stats = {"home": home_stats, "away": away_stats}
        match.stats = new_stats
        await session.flush()
        await session.commit()

        # Step 6: Re-query to verify persistence
        await session.refresh(match)
        result["steps"]["5_persistence"] = {
            "stats_after_commit": match.stats,
            "persisted_successfully": match.stats is not None and match.stats != {},
        }

        if match.stats and match.stats != {}:
            result["diagnosis"] = "SUCCESS - Stats fetched, parsed, and persisted"
        else:
            result["diagnosis"] = "PERSISTENCE_FAILED - Stats were set but did not persist"

        return result

    except Exception as e:
        logger.error(f"stats_rca error: {e}", exc_info=True)
        result["diagnosis"] = f"EXCEPTION: {str(e)}"
        return result


@app.get("/dashboard/ops/bulk_stats_backfill.json")
async def bulk_stats_backfill_endpoint(
    request: Request,
    since_date: str = Query("2026-01-03", description="Start date YYYY-MM-DD"),
    limit: int = Query(50, description="Max matches to process per call"),
    dry_run: bool = Query(True, description="If true, only list matches without fetching"),
    session: AsyncSession = Depends(get_async_session),
):
    """
    Bulk backfill stats for all FT matches since a given date that are missing stats.
    Use dry_run=true first to see how many matches need backfill.
    """
    _verify_debug_token(request)

    from app.etl.api_football import APIFootballProvider
    from datetime import datetime
    import json as json_lib

    result = {
        "since_date": since_date,
        "dry_run": dry_run,
        "limit": limit,
        "matches_found": 0,
        "matches_processed": 0,
        "successes": [],
        "failures": [],
    }

    try:
        # Parse date string to date object
        from datetime import datetime as dt
        since_date_parsed = dt.strptime(since_date, "%Y-%m-%d").date()

        # Find all FT matches since date with missing stats
        res = await session.execute(text("""
            SELECT id, external_id, date, home_team_id, away_team_id
            FROM matches
            WHERE status IN ('FT', 'AET', 'PEN')
              AND date >= :since_date
              AND (stats IS NULL OR stats::text = '{}' OR stats::text = 'null')
            ORDER BY date ASC
            LIMIT :limit
        """), {"since_date": since_date_parsed, "limit": limit})

        matches = res.fetchall()
        result["matches_found"] = len(matches)

        if dry_run:
            result["matches_to_process"] = [
                {"id": m[0], "external_id": m[1], "date": str(m[2])}
                for m in matches
            ]
            # Also count total
            res_total = await session.execute(text("""
                SELECT COUNT(*) FROM matches
                WHERE status IN ('FT', 'AET', 'PEN')
                  AND date >= :since_date
                  AND (stats IS NULL OR stats::text = '{}' OR stats::text = 'null')
            """), {"since_date": since_date_parsed})
            result["total_missing"] = res_total.scalar()
            return result

        # Process matches
        provider = APIFootballProvider()
        key_map = {
            "Ball Possession": "ball_possession",
            "Total Shots": "total_shots",
            "Shots on Goal": "shots_on_goal",
            "Shots off Goal": "shots_off_goal",
            "Blocked Shots": "blocked_shots",
            "Shots insidebox": "shots_insidebox",
            "Shots outsidebox": "shots_outsidebox",
            "Fouls": "fouls",
            "Corner Kicks": "corner_kicks",
            "Offsides": "offsides",
            "Yellow Cards": "yellow_cards",
            "Red Cards": "red_cards",
            "Goalkeeper Saves": "goalkeeper_saves",
            "Total passes": "total_passes",
            "Passes accurate": "passes_accurate",
            "Passes %": "passes_pct",
            "expected_goals": "expected_goals",
        }

        def parse_team_stats(stats_list):
            parsed = {}
            for stat in stats_list:
                stat_type = stat.get("type")
                stat_value = stat.get("value")
                if stat_type in key_map:
                    parsed[key_map[stat_type]] = stat_value
            return parsed

        for match_row in matches:
            match_id, external_id, match_date, home_id, away_id = match_row

            if not external_id:
                result["failures"].append({"id": match_id, "reason": "NO_EXTERNAL_ID"})
                continue

            try:
                stats_data = await provider._rate_limited_request(
                    "fixtures/statistics",
                    {"fixture": external_id},
                    entity="stats"
                )
                response = stats_data.get("response", [])

                if len(response) < 2:
                    result["failures"].append({"id": match_id, "external_id": external_id, "reason": "API_EMPTY"})
                    continue

                home_stats = parse_team_stats(response[0].get("statistics", []))
                away_stats = parse_team_stats(response[1].get("statistics", []))

                new_stats = {"home": home_stats, "away": away_stats}

                # Update in DB
                await session.execute(
                    text("UPDATE matches SET stats = :stats, stats_ready_at = NOW() WHERE id = :id"),
                    {"stats": json_lib.dumps(new_stats), "id": match_id}
                )
                result["successes"].append({"id": match_id, "external_id": external_id})
                result["matches_processed"] += 1

            except Exception as e:
                result["failures"].append({"id": match_id, "external_id": external_id, "reason": str(e)})

        await provider.close()
        await session.commit()

        return result

    except Exception as e:
        logger.error(f"bulk_stats_backfill error: {e}", exc_info=True)
        return {"error": str(e)}


@app.get("/dashboard/ops/fetch_events.json")
async def fetch_events_endpoint(
    request: Request,
    match_id: int = Query(...),
    session: AsyncSession = Depends(get_async_session),
):
    """
    Fetch events from API-Football for a specific match and persist.
    Used for testing/verification.
    """
    _verify_debug_token(request)

    from app.etl.api_football import APIFootballProvider
    from app.llm.fastpath import FastPathService

    result = {
        "match_id": match_id,
        "events_before": None,
        "events_after": None,
        "api_response_count": 0,
        "diagnosis": None,
    }

    try:
        match = await session.get(Match, match_id)
        if not match:
            return {"error": f"Match {match_id} not found"}

        result["events_before"] = match.events
        result["external_id"] = match.external_id

        if not match.external_id:
            result["diagnosis"] = "NO_EXTERNAL_ID"
            return result

        # Fetch events from API-Football
        provider = APIFootballProvider()
        try:
            events_data = await provider._rate_limited_request(
                "fixtures/events",
                {"fixture": match.external_id},
                entity="events"
            )
            await provider.close()
        except Exception as api_err:
            result["diagnosis"] = f"API_CALL_FAILED: {api_err}"
            return result

        events_response = events_data.get("response", [])
        result["api_response_count"] = len(events_response)

        if not events_response:
            result["diagnosis"] = "API_RESPONSE_EMPTY"
            return result

        # Parse events using FastPathService method
        fastpath = FastPathService(session)
        parsed_events = fastpath._parse_events(events_response)
        result["parsed_events_count"] = len(parsed_events)
        result["parsed_events"] = parsed_events

        # Persist
        match.events = parsed_events
        await session.commit()
        await session.refresh(match)

        result["events_after"] = match.events
        result["diagnosis"] = "SUCCESS" if match.events else "PERSISTENCE_FAILED"

        return result

    except Exception as e:
        logger.error(f"fetch_events error: {e}", exc_info=True)
        result["diagnosis"] = f"EXCEPTION: {str(e)}"
        return result


@app.get("/dashboard/ops/audit_metrics.json")
async def audit_metrics_endpoint(
    request: Request,
    session: AsyncSession = Depends(get_async_session),
):
    """
    Audit endpoint: cross-check dashboard metrics with direct DB queries.
    Returns raw query results for manual verification.
    """
    _verify_debug_token(request)

    from sqlalchemy import text

    result = {
        "generated_at": datetime.utcnow().isoformat(),
        "audits": {},
    }

    # P0.1: fastpath_ticks verification
    try:
        # Last 5 ticks
        res = await session.execute(text("""
            SELECT tick_at, selected, refreshed, ready, enqueued, completed, errors, skipped
            FROM fastpath_ticks
            ORDER BY tick_at DESC
            LIMIT 5
        """))
        ticks = [{"tick_at": str(r[0]), "selected": r[1], "refreshed": r[2], "ready": r[3],
                  "enqueued": r[4], "completed": r[5], "errors": r[6], "skipped": r[7]}
                 for r in res.fetchall()]
        result["audits"]["fastpath_ticks_last_5"] = ticks

        # Tick count last hour
        res = await session.execute(text("""
            SELECT COUNT(*), COUNT(*) FILTER (WHERE selected > 0 OR enqueued > 0 OR completed > 0)
            FROM fastpath_ticks WHERE tick_at > NOW() - INTERVAL '1 hour'
        """))
        row = res.fetchone()
        result["audits"]["ticks_1h"] = {"total": row[0], "with_activity": row[1]}
    except Exception as e:
        result["audits"]["fastpath_ticks_error"] = str(e)

    # P0.1: pending_ready verification - sample 5 match_ids
    try:
        res = await session.execute(text("""
            SELECT m.id, m.status, m.stats_ready_at, m.finished_at, m.date,
                   COALESCE(m.finished_at, m.date) as effective_finished
            FROM matches m
            WHERE m.status IN ('FT', 'AET', 'PEN')
              AND COALESCE(m.finished_at, m.date) > NOW() - INTERVAL '180 minutes'
              AND m.stats_ready_at IS NOT NULL
              AND NOT EXISTS (
                  SELECT 1 FROM prediction_outcomes po
                  JOIN post_match_audits pma ON pma.outcome_id = po.id
                  WHERE po.match_id = m.id AND pma.llm_narrative_status = 'ok'
              )
            LIMIT 5
        """))
        pending = [{"match_id": r[0], "status": r[1], "stats_ready_at": str(r[2]) if r[2] else None,
                    "finished_at": str(r[3]) if r[3] else None, "date": str(r[4]) if r[4] else None}
                   for r in res.fetchall()]
        result["audits"]["pending_ready_sample"] = pending
        result["audits"]["pending_ready_count"] = len(pending)
    except Exception as e:
        result["audits"]["pending_ready_error"] = str(e)

    # P0.2: LLM status breakdown last 60m - direct query
    try:
        res = await session.execute(text("""
            SELECT llm_narrative_status, COUNT(*) as cnt
            FROM post_match_audits
            WHERE created_at > NOW() - INTERVAL '60 minutes'
              AND llm_narrative_status IS NOT NULL
            GROUP BY llm_narrative_status
        """))
        llm_breakdown = {r[0]: r[1] for r in res.fetchall()}
        result["audits"]["llm_60m_direct"] = llm_breakdown

        # Sample of audits with error
        res = await session.execute(text("""
            SELECT pma.id, pma.outcome_id, pma.llm_narrative_status, pma.llm_narrative_error_code, pma.created_at
            FROM post_match_audits pma
            WHERE pma.created_at > NOW() - INTERVAL '60 minutes'
              AND pma.llm_narrative_status = 'error'
            LIMIT 5
        """))
        errors = [{"audit_id": r[0], "outcome_id": r[1], "status": r[2], "error_code": r[3], "created_at": str(r[4])}
                  for r in res.fetchall()]
        result["audits"]["llm_errors_sample"] = errors
    except Exception as e:
        result["audits"]["llm_60m_error"] = str(e)

    # P1.1: predictions_health verification
    try:
        # Last prediction saved
        res = await session.execute(text("SELECT MAX(created_at) FROM predictions"))
        last_pred = res.scalar()
        result["audits"]["last_prediction_saved_at"] = str(last_pred) if last_pred else None

        # FT matches last 48h
        res = await session.execute(text("""
            SELECT COUNT(*) FROM matches
            WHERE status IN ('FT', 'AET', 'PEN')
              AND COALESCE(finished_at, date) > NOW() - INTERVAL '48 hours'
        """))
        ft_48h = res.scalar()
        result["audits"]["ft_matches_48h"] = ft_48h

        # FT matches missing prediction
        res = await session.execute(text("""
            SELECT COUNT(*) FROM matches m
            WHERE m.status IN ('FT', 'AET', 'PEN')
              AND COALESCE(m.finished_at, m.date) > NOW() - INTERVAL '48 hours'
              AND NOT EXISTS (SELECT 1 FROM predictions p WHERE p.match_id = m.id)
        """))
        missing = res.scalar()
        result["audits"]["ft_missing_prediction_48h"] = missing

        # Sample of missing
        res = await session.execute(text("""
            SELECT m.id, m.status, m.date, m.home_team_id, m.away_team_id
            FROM matches m
            WHERE m.status IN ('FT', 'AET', 'PEN')
              AND COALESCE(m.finished_at, m.date) > NOW() - INTERVAL '48 hours'
              AND NOT EXISTS (SELECT 1 FROM predictions p WHERE p.match_id = m.id)
            LIMIT 10
        """))
        missing_sample = [{"match_id": r[0], "status": r[1], "date": str(r[2]) if r[2] else None}
                          for r in res.fetchall()]
        result["audits"]["ft_missing_prediction_sample"] = missing_sample
    except Exception as e:
        result["audits"]["predictions_health_error"] = str(e)

    # P1.2: stats_backfill verification
    try:
        # Matches 72h with stats (stats is JSON, cast to text for comparison)
        res = await session.execute(text("""
            SELECT COUNT(*) FROM matches
            WHERE status IN ('FT', 'AET', 'PEN')
              AND COALESCE(finished_at, date) > NOW() - INTERVAL '72 hours'
              AND stats IS NOT NULL
              AND stats::text != '{}'
              AND stats::text != 'null'
        """))
        with_stats = res.scalar()
        result["audits"]["finished_72h_with_stats"] = with_stats

        # Matches 72h missing stats
        res = await session.execute(text("""
            SELECT COUNT(*) FROM matches
            WHERE status IN ('FT', 'AET', 'PEN')
              AND COALESCE(finished_at, date) > NOW() - INTERVAL '72 hours'
              AND (stats IS NULL OR stats::text = '{}' OR stats::text = 'null')
        """))
        missing_stats = res.scalar()
        result["audits"]["finished_72h_missing_stats"] = missing_stats

        # Sample missing stats
        res = await session.execute(text("""
            SELECT id, status, date, stats
            FROM matches
            WHERE status IN ('FT', 'AET', 'PEN')
              AND COALESCE(finished_at, date) > NOW() - INTERVAL '72 hours'
              AND (stats IS NULL OR stats::text = '{}' OR stats::text = 'null')
            LIMIT 10
        """))
        missing_sample = [{"match_id": r[0], "status": r[1], "date": str(r[2]) if r[2] else None,
                           "stats": r[3]} for r in res.fetchall()]
        result["audits"]["missing_stats_sample"] = missing_sample
    except Exception as e:
        result["audits"]["stats_backfill_error"] = str(e)

    # P0.3: Stats integrity check - matches with stats that might be overwritten
    try:
        res = await session.execute(text("""
            SELECT id, status, stats_ready_at, stats IS NOT NULL as has_stats,
                   events IS NOT NULL as has_events
            FROM matches
            WHERE status IN ('FT', 'AET', 'PEN')
              AND stats_ready_at IS NOT NULL
              AND stats IS NOT NULL AND stats::text != '{}'
            ORDER BY stats_ready_at DESC
            LIMIT 5
        """))
        integrity = [{"match_id": r[0], "status": r[1], "stats_ready_at": str(r[2]) if r[2] else None,
                      "has_stats": r[3], "has_events": r[4]} for r in res.fetchall()]
        result["audits"]["stats_integrity_sample"] = integrity
    except Exception as e:
        result["audits"]["stats_integrity_error"] = str(e)

    return result


@app.get("/dashboard/ops/predictions_performance.json")
async def predictions_performance_endpoint(
    request: Request,
    window_days: int = Query(default=7, ge=1, le=30),
    regenerate: bool = Query(default=False),
    session: AsyncSession = Depends(get_async_session),
):
    """
    Prediction performance report: proper probability metrics for model evaluation.

    Returns Brier score, log loss, calibration, and market comparison.
    Use this to distinguish variance from bugs.

    Args:
        window_days: 7 or 14 (default 7)
        regenerate: If True, generates fresh report instead of returning cached

    Auth: X-Dashboard-Token header required.
    """
    _verify_debug_token(request)

    from app.ml.performance_metrics import (
        generate_performance_report,
        get_latest_report,
        save_performance_report,
    )

    if regenerate:
        # Generate fresh report
        report = await generate_performance_report(session, window_days)
        await save_performance_report(session, report, window_days, source="api")
        return report

    # Try to get cached report
    cached = await get_latest_report(session, window_days)
    if cached:
        return cached

    # No cached report, generate one
    report = await generate_performance_report(session, window_days)
    await save_performance_report(session, report, window_days, source="api")
    return report


# =============================================================================
# SENTRY HEALTH (server-side aggregation for ops dashboard)
# =============================================================================

_sentry_health_cache: dict = {
    "data": None,
    "timestamp": 0,
    "ttl": 90,  # 90 seconds cache (balance between freshness and API limits)
}

# Sentry API thresholds for status determination
_SENTRY_CRITICAL_THRESHOLD_1H = 3  # active_issues_1h >= 3 → critical
_SENTRY_WARNING_THRESHOLD_24H = 1  # active_issues_24h >= 1 → warning


async def _fetch_sentry_health() -> dict:
    """
    Fetch Sentry health metrics via Sentry API (server-side only).

    Best-effort: returns degraded status if credentials missing or API fails.
    Uses in-memory cache with 90s TTL to avoid API rate limits.

    Sentry API endpoints used:
    - GET /api/0/projects/{org}/{project}/issues/ (for issue counts)
    - Query params: statsPeriod=1h, statsPeriod=24h, query=is:unresolved

    Returns:
        dict with status, counts, top_issues, etc.
    """
    import time as time_module
    import httpx

    now_ts = time_module.time()
    now_iso = datetime.utcnow().isoformat()

    # Check cache first
    if _sentry_health_cache["data"] and (now_ts - _sentry_health_cache["timestamp"]) < _sentry_health_cache["ttl"]:
        cached = _sentry_health_cache["data"].copy()
        cached["cached"] = True
        cached["cache_age_seconds"] = int(now_ts - _sentry_health_cache["timestamp"])
        return cached

    # Base response structure (degraded fallback)
    base_response = {
        "status": "degraded",
        "cached": False,
        "cache_age_seconds": 0,
        "generated_at": now_iso,
        "project": {
            "org_slug": settings.SENTRY_ORG or None,
            "project_slug": settings.SENTRY_PROJECT_SLUG or None,
            "env": settings.SENTRY_ENV or "production",
        },
        "counts": {
            "new_issues_1h": 0,
            "new_issues_24h": 0,
            "active_issues_1h": 0,
            "active_issues_24h": 0,
            "open_issues": 0,
        },
        "last_event_at": None,
        "top_issues": [],
        "note": "best-effort, aggregated server-side",
    }

    # Check if credentials are configured
    if not settings.SENTRY_AUTH_TOKEN or not settings.SENTRY_ORG or not settings.SENTRY_PROJECT_SLUG:
        base_response["error"] = "Sentry credentials not configured"
        _sentry_health_cache["data"] = base_response
        _sentry_health_cache["timestamp"] = now_ts
        return base_response

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

            # Calculate time boundaries for filtering by lastSeen/firstSeen
            from datetime import timedelta
            now_dt = datetime.utcnow()
            one_hour_ago = (now_dt - timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%S")
            one_day_ago = (now_dt - timedelta(hours=24)).strftime("%Y-%m-%dT%H:%M:%S")

            # Build query - try with env filter first, fallback without if empty
            env_query = f"environment:{env_filter}" if env_filter else ""

            # 1) Fetch ALL unresolved issues (open_issues - no time filter)
            #    Sort by lastSeen desc to get most recent activity first
            params_open = {
                "query": f"is:unresolved {env_query}".strip(),
                "sort": "date",  # sort by lastSeen descending
                "limit": 100,
            }
            resp_open = await client.get(issues_url, params=params_open)

            # Parse open issues response
            all_issues = []
            env_filter_excluded = False

            if resp_open.status_code == 200:
                all_issues = resp_open.json() if isinstance(resp_open.json(), list) else []

                # If env filter returned 0, try without env to check if filter excluded results
                if len(all_issues) == 0 and env_query:
                    params_no_env = {
                        "query": "is:unresolved",
                        "sort": "date",
                        "limit": 100,
                    }
                    resp_no_env = await client.get(issues_url, params=params_no_env)
                    if resp_no_env.status_code == 200:
                        issues_no_env = resp_no_env.json() if isinstance(resp_no_env.json(), list) else []
                        if len(issues_no_env) > 0:
                            env_filter_excluded = True
                            all_issues = issues_no_env  # Use unfiltered results

            # Calculate counts from the fetched issues
            open_issues = len(all_issues)
            new_issues_1h = 0
            new_issues_24h = 0
            active_issues_1h = 0
            active_issues_24h = 0
            last_event_at = None
            top_issues = []

            for issue in all_issues:
                first_seen = issue.get("firstSeen", "")
                last_seen = issue.get("lastSeen", "")

                # New issues (by firstSeen - when issue was created)
                if first_seen and first_seen >= one_hour_ago:
                    new_issues_1h += 1
                if first_seen and first_seen >= one_day_ago:
                    new_issues_24h += 1

                # Active issues (by lastSeen - recent activity)
                if last_seen and last_seen >= one_hour_ago:
                    active_issues_1h += 1
                if last_seen and last_seen >= one_day_ago:
                    active_issues_24h += 1

            # Get last_event_at from most recent issue (already sorted by lastSeen desc)
            if all_issues:
                last_event_at = all_issues[0].get("lastSeen")

                # Extract top issues (top 3 by recent activity, from active_24h)
                active_24h_issues = [
                    i for i in all_issues
                    if i.get("lastSeen", "") >= one_day_ago
                ]
                # Sort by count descending for top issues
                sorted_active = sorted(
                    active_24h_issues,
                    key=lambda x: int(x.get("count", 0)),
                    reverse=True
                )[:3]

                for issue in sorted_active:
                    title = issue.get("title", "Unknown")[:80]
                    title = title.replace("@", "[at]")  # Basic sanitization
                    top_issues.append({
                        "title": title,
                        "count": int(issue.get("count", 0)),
                        "level": issue.get("level", "error"),
                        "last_seen": issue.get("lastSeen"),
                    })

            # Determine status based on activity thresholds (use active, not new)
            status = "ok"
            if active_issues_1h >= _SENTRY_CRITICAL_THRESHOLD_1H:
                status = "critical"
            elif active_issues_24h >= _SENTRY_WARNING_THRESHOLD_24H:
                status = "warning"

            # Build note
            note = "best-effort, aggregated server-side"
            if env_filter_excluded:
                note += f"; env filter '{env_filter}' excluded results, showing all"

            result = {
                "status": status,
                "cached": False,
                "cache_age_seconds": 0,
                "generated_at": now_iso,
                "project": {
                    "org_slug": org_slug,
                    "project_slug": project_slug,
                    "env": env_filter if not env_filter_excluded else "(all)",
                },
                "counts": {
                    "new_issues_1h": new_issues_1h,
                    "new_issues_24h": new_issues_24h,
                    "active_issues_1h": active_issues_1h,
                    "active_issues_24h": active_issues_24h,
                    "open_issues": open_issues,
                },
                "last_event_at": last_event_at,
                "top_issues": top_issues,
                "note": note,
            }

            # Update cache
            _sentry_health_cache["data"] = result
            _sentry_health_cache["timestamp"] = now_ts

            return result

    except httpx.TimeoutException:
        base_response["error"] = "Sentry API timeout"
        logger.warning("Sentry health fetch timeout")
    except httpx.HTTPStatusError as e:
        base_response["error"] = f"Sentry API HTTP {e.response.status_code}"
        logger.warning(f"Sentry health fetch HTTP error: {e}")
    except Exception as e:
        base_response["error"] = f"Sentry fetch error: {str(e)[:50]}"
        logger.warning(f"Sentry health fetch error: {e}")

    # Cache degraded response too (to avoid hammering on errors)
    _sentry_health_cache["data"] = base_response
    _sentry_health_cache["timestamp"] = now_ts
    return base_response


def _get_providers_health() -> dict:
    """Get health status for external data providers."""
    try:
        from app.etl.api_football import get_provider_health
        api_football = get_provider_health()
    except Exception as e:
        logger.warning(f"Could not get API-Football provider health: {e}")
        api_football = {"status": "unknown", "error": str(e)}

    return {
        "api_football": api_football,
    }


async def _load_ops_data() -> dict:
    """
    Ops dashboard: read-only aggregated metrics from DB + in-process state.
    Designed to be lightweight (few aggregated queries) and cached.
    """
    now = datetime.utcnow()

    # Budget status - fetch real API account status from API-Football
    budget_status: dict = {"status": "unavailable"}
    try:
        from app.etl.api_football import get_api_account_status

        budget_status = await get_api_account_status()
    except Exception as e:
        logger.warning(f"Could not fetch API account status: {e}")
        budget_status = {"status": "unavailable", "error": str(e)}

    # Observational metadata: API-Football daily budget refresh time (approx).
    # User-reported: ~4:00pm America/Los_Angeles. Best-effort only (ops UX).
    try:
        from zoneinfo import ZoneInfo

        tz_name = "America/Los_Angeles"
        reset_hour = 16
        reset_minute = 0

        now_utc = datetime.utcnow().replace(tzinfo=ZoneInfo("UTC"))
        now_la = now_utc.astimezone(ZoneInfo(tz_name))
        next_reset_la = now_la.replace(hour=reset_hour, minute=reset_minute, second=0, microsecond=0)
        if next_reset_la <= now_la:
            next_reset_la = next_reset_la + timedelta(days=1)
        next_reset_utc = next_reset_la.astimezone(ZoneInfo("UTC"))

        if not isinstance(budget_status, dict):
            budget_status = {"status": "unavailable"}

        budget_status.update(
            {
                "tokens_reset_tz": tz_name,
                "tokens_reset_local_time": f"{reset_hour:02d}:{reset_minute:02d}",
                "tokens_reset_at_la": next_reset_la.isoformat(),
                "tokens_reset_at_utc": next_reset_utc.isoformat(),
                "tokens_reset_note": "Observed daily refresh around 4:00pm America/Los_Angeles",
            }
        )
    except Exception:
        pass

    # Sentry health - fetch aggregated metrics (best-effort, cached)
    sentry_health: dict = await _fetch_sentry_health()

    league_mode = os.environ.get("LEAGUE_MODE", "tracked").strip().lower()
    last_sync = get_last_sync_time()

    async with AsyncSessionLocal() as session:
        # Tracked leagues (distinct league_id)
        res = await session.execute(text("SELECT COUNT(DISTINCT league_id) FROM matches WHERE league_id IS NOT NULL"))
        tracked_leagues_count = int(res.scalar() or 0)

        # Upcoming matches (next 24h)
        res = await session.execute(
            text(
                """
                SELECT league_id, COUNT(*) AS upcoming
                FROM matches
                WHERE league_id IS NOT NULL
                  AND date >= NOW()
                  AND date < NOW() + INTERVAL '24 hours'
                GROUP BY league_id
                ORDER BY upcoming DESC
                LIMIT 20
                """
            )
        )
        upcoming_by_league = [{"league_id": int(r[0]), "upcoming_24h": int(r[1])} for r in res.fetchall()]

        # PIT snapshots (live, lineup_confirmed)
        res = await session.execute(
            text(
                """
                SELECT COUNT(*)
                FROM odds_snapshots
                WHERE snapshot_type = 'lineup_confirmed'
                  AND odds_freshness = 'live'
                  AND snapshot_at > NOW() - INTERVAL '60 minutes'
                """
            )
        )
        pit_live_60m = int(res.scalar() or 0)

        res = await session.execute(
            text(
                """
                SELECT COUNT(*)
                FROM odds_snapshots
                WHERE snapshot_type = 'lineup_confirmed'
                  AND odds_freshness = 'live'
                  AND snapshot_at > NOW() - INTERVAL '24 hours'
                """
            )
        )
        pit_live_24h = int(res.scalar() or 0)

        # ΔKO distribution (last 60m)
        res = await session.execute(
            text(
                """
                SELECT ROUND(delta_to_kickoff_seconds / 60.0) AS min_to_ko, COUNT(*) AS c
                FROM odds_snapshots
                WHERE snapshot_type = 'lineup_confirmed'
                  AND odds_freshness = 'live'
                  AND snapshot_at > NOW() - INTERVAL '60 minutes'
                  AND delta_to_kickoff_seconds IS NOT NULL
                GROUP BY 1
                ORDER BY 1
                """
            )
        )
        pit_dko_60m = [{"min_to_ko": int(r[0]), "count": int(r[1])} for r in res.fetchall()]

        # Latest PIT snapshots (last 10, any freshness)
        res = await session.execute(
            text(
                """
                SELECT os.snapshot_at, os.match_id, m.league_id, os.odds_freshness, os.delta_to_kickoff_seconds,
                       os.odds_home, os.odds_draw, os.odds_away, os.bookmaker
                FROM odds_snapshots os
                JOIN matches m ON m.id = os.match_id
                WHERE os.snapshot_type = 'lineup_confirmed'
                ORDER BY os.snapshot_at DESC
                LIMIT 10
                """
            )
        )
        latest_pit = []
        for r in res.fetchall():
            latest_pit.append(
                {
                    "snapshot_at": r[0].isoformat() if r[0] else None,
                    "match_id": int(r[1]) if r[1] is not None else None,
                    "league_id": int(r[2]) if r[2] is not None else None,
                    "odds_freshness": r[3],
                    "delta_to_kickoff_minutes": round(float(r[4]) / 60.0, 1) if r[4] is not None else None,
                    "odds": {
                        "home": float(r[5]) if r[5] is not None else None,
                        "draw": float(r[6]) if r[6] is not None else None,
                        "away": float(r[7]) if r[7] is not None else None,
                    },
                    "bookmaker": r[8],
                }
            )

        # Movement snapshots (last 24h)
        lineup_movement_24h = None
        market_movement_24h = None
        try:
            res = await session.execute(
                text(
                    """
                    SELECT COUNT(*)
                    FROM lineup_movement_snapshots
                    WHERE captured_at > NOW() - INTERVAL '24 hours'
                    """
                )
            )
            lineup_movement_24h = int(res.scalar() or 0)
        except Exception:
            lineup_movement_24h = None

        try:
            res = await session.execute(
                text(
                    """
                    SELECT COUNT(*)
                    FROM market_movement_snapshots
                    WHERE captured_at > NOW() - INTERVAL '24 hours'
                    """
                )
            )
            market_movement_24h = int(res.scalar() or 0)
        except Exception:
            market_movement_24h = None

        # Stats backfill health (last 72h finished matches)
        # Use COALESCE(finished_at, date) to get matches that FINISHED in last 72h
        # not matches that STARTED in last 72h (date is kickoff time)
        res = await session.execute(
            text(
                """
                SELECT
                    COUNT(*) FILTER (WHERE stats IS NOT NULL AND stats::text != '{}' AND stats::text != 'null') AS with_stats,
                    COUNT(*) FILTER (WHERE stats IS NULL OR stats::text = '{}' OR stats::text = 'null') AS missing_stats
                FROM matches
                WHERE status IN ('FT', 'AET', 'PEN')
                  AND COALESCE(finished_at, date) > NOW() - INTERVAL '72 hours'
                """
            )
        )
        row = res.first()
        stats_with = int(row[0] or 0) if row else 0
        stats_missing = int(row[1] or 0) if row else 0

        # =============================================================
        # PROGRESS METRICS (for re-test / Alpha readiness)
        # =============================================================
        # Configurable targets via env vars
        # Per PIT Protocol v2: Piloto=50, Preliminar=200, Formal=500
        # Aligned with TITAN formal gate for consistency
        TARGET_PIT_SNAPSHOTS_30D = int(os.environ.get("TARGET_PIT_SNAPSHOTS_30D", "500"))
        TARGET_PIT_BETS_30D = int(os.environ.get("TARGET_PIT_BETS_30D", "500"))
        TARGET_BASELINE_COVERAGE_PCT = int(os.environ.get("TARGET_BASELINE_COVERAGE_PCT", "60"))

        # 1) PIT snapshots (30 days) - lineup_confirmed with live odds
        pit_snapshots_30d = 0
        try:
            res = await session.execute(
                text(
                    """
                    SELECT COUNT(*)
                    FROM odds_snapshots
                    WHERE snapshot_type = 'lineup_confirmed'
                      AND odds_freshness = 'live'
                      AND snapshot_at > NOW() - INTERVAL '30 days'
                    """
                )
            )
            pit_snapshots_30d = int(res.scalar() or 0)
        except Exception:
            pit_snapshots_30d = 0

        # 2) PIT with predictions as-of (evaluable bets, 30 days)
        # Count PIT snapshots that have a prediction created BEFORE the snapshot
        pit_bets_30d = 0
        try:
            res = await session.execute(
                text(
                    """
                    SELECT COUNT(DISTINCT os.id)
                    FROM odds_snapshots os
                    WHERE os.snapshot_type = 'lineup_confirmed'
                      AND os.odds_freshness = 'live'
                      AND os.snapshot_at > NOW() - INTERVAL '30 days'
                      AND EXISTS (
                          SELECT 1 FROM predictions p
                          WHERE p.match_id = os.match_id
                            AND p.created_at < os.snapshot_at
                      )
                    """
                )
            )
            pit_bets_30d = int(res.scalar() or 0)
        except Exception:
            pit_bets_30d = 0

        # 3) Baseline coverage (% of recent PIT matches with market_movement pre-KO)
        # This measures how many PIT snapshots have baseline odds for CLV proxy
        baseline_coverage_pct = 0
        pit_with_baseline = 0
        pit_total_for_baseline = 0
        try:
            res = await session.execute(
                text(
                    """
                    SELECT
                        COUNT(*) FILTER (WHERE has_baseline) AS with_baseline,
                        COUNT(*) AS total
                    FROM (
                        SELECT os.id,
                               EXISTS (
                                   SELECT 1 FROM market_movement_snapshots mms
                                   WHERE mms.match_id = os.match_id
                                     AND mms.captured_at < (
                                         SELECT m.date FROM matches m WHERE m.id = os.match_id
                                     )
                               ) AS has_baseline
                        FROM odds_snapshots os
                        WHERE os.snapshot_type = 'lineup_confirmed'
                          AND os.odds_freshness = 'live'
                          AND os.snapshot_at > NOW() - INTERVAL '30 days'
                    ) sub
                    """
                )
            )
            row = res.first()
            if row:
                pit_with_baseline = int(row[0] or 0)
                pit_total_for_baseline = int(row[1] or 0)
                if pit_total_for_baseline > 0:
                    baseline_coverage_pct = round((pit_with_baseline / pit_total_for_baseline) * 100, 1)
        except Exception:
            baseline_coverage_pct = 0
            pit_with_baseline = 0
            pit_total_for_baseline = 0

        progress_metrics = {
            "pit_snapshots_30d": pit_snapshots_30d,
            "target_pit_snapshots_30d": TARGET_PIT_SNAPSHOTS_30D,
            "pit_bets_30d": pit_bets_30d,
            "target_pit_bets_30d": TARGET_PIT_BETS_30D,
            "baseline_coverage_pct": baseline_coverage_pct,
            "pit_with_baseline": pit_with_baseline,
            "pit_total_for_baseline": pit_total_for_baseline,
            "target_baseline_coverage_pct": TARGET_BASELINE_COVERAGE_PCT,
            "ready_for_retest": (
                pit_bets_30d >= TARGET_PIT_BETS_30D and
                baseline_coverage_pct >= TARGET_BASELINE_COVERAGE_PCT
            ),
        }

        # =============================================================
        # PREDICTIONS HEALTH (P0 observability - detect scheduler issues)
        # =============================================================
        predictions_health = await _calculate_predictions_health(session)

        # =============================================================
        # FAST-PATH HEALTH (LLM narrative generation monitoring)
        # =============================================================
        fastpath_health = await _calculate_fastpath_health(session)

        # =============================================================
        # MODEL PERFORMANCE (7d probability metrics summary)
        # =============================================================
        model_performance = await _calculate_model_performance(session)

        # =============================================================
        # DATA QUALITY TELEMETRY (quarantine/taint/unmapped summary)
        # =============================================================
        telemetry_data = await _calculate_telemetry_summary(session)

        # =============================================================
        # SHADOW MODE (A/B model comparison monitoring)
        # =============================================================
        shadow_mode_data = await _calculate_shadow_mode_summary(session)

        # =============================================================
        # SENSOR B (LogReg L2 calibration diagnostics - INTERNAL ONLY)
        # =============================================================
        sensor_b_data = await _calculate_sensor_b_summary(session)

        # =============================================================
        # EXTC SHADOW (experimental ext-C model evaluation)
        # =============================================================
        extc_shadow_data = await _calculate_extc_shadow_summary(session)

        # =============================================================
        # RERUN SERVING (DB-first canary for two-stage)
        # =============================================================
        rerun_serving_data = await _calculate_rerun_serving_summary(session)

        # =============================================================
        # JOBS HEALTH (P0 jobs monitoring - stats_backfill, odds_sync, fastpath)
        # =============================================================
        jobs_health_data = await _calculate_jobs_health_summary(session)

        # =============================================================
        # SOTA ENRICHMENT (Understat xG, Weather, Venue Geo coverage)
        # =============================================================
        sota_enrichment_data = await _calculate_sota_enrichment_summary(session)

        # =============================================================
        # LLM COST (Gemini token usage from PostMatchAudit)
        # =============================================================
        llm_cost_data = {"provider": "gemini", "status": "unavailable"}
        try:
            # Rollback any previous failed transaction state
            await session.rollback()

            # Use pricing from settings (single source of truth)
            MODEL_PRICING = settings.GEMINI_PRICING
            DEFAULT_PRICE_IN = settings.GEMINI_PRICE_INPUT
            DEFAULT_PRICE_OUT = settings.GEMINI_PRICE_OUTPUT

            # Build dynamic CASE statements from MODEL_PRICING
            # Groups models by same pricing to reduce SQL complexity
            def build_pricing_case_sql() -> str:
                """Generate SQL CASE for model-specific pricing from settings."""
                # Group models by pricing tuple (input, output)
                pricing_groups: dict[tuple[float, float], list[str]] = {}
                for model, prices in MODEL_PRICING.items():
                    key = (prices["input"], prices["output"])
                    if key not in pricing_groups:
                        pricing_groups[key] = []
                    pricing_groups[key].append(model)

                case_parts = []
                for (price_in, price_out), models in pricing_groups.items():
                    models_sql = ", ".join(f"'{m}'" for m in models)
                    case_parts.append(
                        f"WHEN llm_narrative_model IN ({models_sql}) THEN "
                        f"(COALESCE(llm_narrative_tokens_in, 0) * {price_in} + "
                        f"COALESCE(llm_narrative_tokens_out, 0) * {price_out}) / 1000000.0"
                    )

                # Add ELSE for unknown models (uses default pricing via params)
                case_parts.append(
                    "ELSE (COALESCE(llm_narrative_tokens_in, 0) * :default_in + "
                    "COALESCE(llm_narrative_tokens_out, 0) * :default_out) / 1000000.0"
                )

                return "CASE " + " ".join(case_parts) + " END"

            pricing_case_sql = build_pricing_case_sql()

            # Helper function to build LLM cost query with specific interval
            # Note: INTERVAL cannot be parameterized in PostgreSQL, must use literal
            def llm_cost_query(interval_literal: str) -> str:
                return f"""
                    SELECT
                        COUNT(*) AS request_count,
                        COALESCE(SUM(llm_narrative_tokens_in), 0) AS tokens_in,
                        COALESCE(SUM(llm_narrative_tokens_out), 0) AS tokens_out,
                        COALESCE(SUM({pricing_case_sql}), 0) AS cost_usd
                    FROM post_match_audits
                    WHERE llm_narrative_model LIKE 'gemini%'
                      AND (COALESCE(llm_narrative_tokens_in, 0) > 0
                           OR COALESCE(llm_narrative_tokens_out, 0) > 0)
                      AND created_at > NOW() - INTERVAL '{interval_literal}'
                """

            query_params = {"default_in": DEFAULT_PRICE_IN, "default_out": DEFAULT_PRICE_OUT}

            # 24h metrics
            res_24h = await session.execute(
                text(llm_cost_query("24 hours")), query_params
            )
            row_24h = res_24h.first()
            requests_24h = int(row_24h[0] or 0) if row_24h else 0
            tokens_in_24h = int(row_24h[1] or 0) if row_24h else 0
            tokens_out_24h = int(row_24h[2] or 0) if row_24h else 0
            cost_24h = float(row_24h[3] or 0) if row_24h else 0.0

            # 7d metrics
            res_7d = await session.execute(
                text(llm_cost_query("7 days")), query_params
            )
            row_7d = res_7d.first()
            requests_7d = int(row_7d[0] or 0) if row_7d else 0
            tokens_in_7d = int(row_7d[1] or 0) if row_7d else 0
            tokens_out_7d = int(row_7d[2] or 0) if row_7d else 0
            cost_7d = float(row_7d[3] or 0) if row_7d else 0.0

            # 28d metrics (matches Google AI Studio billing window)
            res_28d = await session.execute(
                text(llm_cost_query("28 days")), query_params
            )
            row_28d = res_28d.first()
            requests_28d = int(row_28d[0] or 0) if row_28d else 0
            tokens_in_28d = int(row_28d[1] or 0) if row_28d else 0
            tokens_out_28d = int(row_28d[2] or 0) if row_28d else 0
            cost_28d = float(row_28d[3] or 0) if row_28d else 0.0

            # Total accumulated cost (all time)
            res_total = await session.execute(
                text(
                    f"""
                    SELECT
                        COUNT(*) AS request_count,
                        COALESCE(SUM(llm_narrative_tokens_in), 0) AS tokens_in,
                        COALESCE(SUM(llm_narrative_tokens_out), 0) AS tokens_out,
                        COALESCE(SUM({pricing_case_sql}), 0) AS cost_usd
                    FROM post_match_audits
                    WHERE llm_narrative_model LIKE 'gemini%'
                      AND (COALESCE(llm_narrative_tokens_in, 0) > 0
                           OR COALESCE(llm_narrative_tokens_out, 0) > 0)
                    """
                ),
                query_params,
            )
            row_total = res_total.first()
            requests_total = int(row_total[0] or 0) if row_total else 0
            tokens_in_total = int(row_total[1] or 0) if row_total else 0
            tokens_out_total = int(row_total[2] or 0) if row_total else 0
            cost_total = float(row_total[3] or 0) if row_total else 0.0

            # Calculate avg cost per request
            avg_cost_per_request = cost_24h / requests_24h if requests_24h > 0 else 0.0

            # Status: warn if cost_24h > $1 or avg_cost > $0.01
            status = "ok"
            if cost_24h > 1.0 or avg_cost_per_request > 0.01:
                status = "warn"

            # Model usage breakdown by window (for tooltip/audit)
            # Best-effort: if fails, omit model_usage_* (don't break llm_cost)
            model_usage_28d = None
            model_usage_7d = None
            model_usage_24h = None
            try:
                def model_usage_query(interval_literal: str) -> str:
                    return f"""
                        SELECT
                            llm_narrative_model,
                            COUNT(*) AS requests,
                            COALESCE(SUM(llm_narrative_tokens_in), 0) AS tokens_in,
                            COALESCE(SUM(llm_narrative_tokens_out), 0) AS tokens_out
                        FROM post_match_audits
                        WHERE llm_narrative_model LIKE 'gemini%'
                          AND (COALESCE(llm_narrative_tokens_in, 0) > 0
                               OR COALESCE(llm_narrative_tokens_out, 0) > 0)
                          AND created_at > NOW() - INTERVAL '{interval_literal}'
                        GROUP BY llm_narrative_model
                        ORDER BY requests DESC
                    """

                def parse_model_usage(rows) -> dict | None:
                    if not rows:
                        return None
                    models = {}
                    top_model = None
                    max_requests = 0
                    for row in rows:
                        model_name = row[0]
                        req_count = int(row[1] or 0)
                        t_in = int(row[2] or 0)
                        t_out = int(row[3] or 0)
                        models[model_name] = {
                            "requests": req_count,
                            "tokens_in": t_in,
                            "tokens_out": t_out,
                        }
                        if req_count > max_requests:
                            max_requests = req_count
                            top_model = model_name
                    if not models:
                        return None
                    return {"top_model": top_model, "models": models}

                # 28d usage
                res_usage_28d = await session.execute(text(model_usage_query("28 days")))
                model_usage_28d = parse_model_usage(res_usage_28d.fetchall())

                # 7d usage
                res_usage_7d = await session.execute(text(model_usage_query("7 days")))
                model_usage_7d = parse_model_usage(res_usage_7d.fetchall())

                # 24h usage
                res_usage_24h = await session.execute(text(model_usage_query("24 hours")))
                model_usage_24h = parse_model_usage(res_usage_24h.fetchall())

            except Exception as usage_err:
                logger.debug(f"Could not calculate model usage breakdown: {usage_err}")
                # Continue without model_usage_* (best-effort)

            # Get current model pricing for transparency
            current_model = settings.GEMINI_MODEL
            current_pricing = MODEL_PRICING.get(
                current_model, {"input": DEFAULT_PRICE_IN, "output": DEFAULT_PRICE_OUT}
            )

            llm_cost_data = {
                "provider": "gemini",
                "model": current_model,
                # Pricing transparency for auditing (current model)
                "pricing_input_per_1m": current_pricing["input"],
                "pricing_output_per_1m": current_pricing["output"],
                # All model pricing for reference
                "model_pricing": MODEL_PRICING,
                # Cost metrics (calculated with per-model pricing)
                "cost_24h_usd": round(cost_24h, 4),
                "cost_7d_usd": round(cost_7d, 4),
                "cost_28d_usd": round(cost_28d, 4),
                "cost_total_usd": round(cost_total, 2),
                # Request counts (all requests with tokens, not filtered by status)
                "requests_24h": requests_24h,
                "requests_7d": requests_7d,
                "requests_28d": requests_28d,
                "requests_total": requests_total,
                # Legacy fields for backward compatibility
                "requests_ok_24h": requests_24h,
                "requests_ok_7d": requests_7d,
                "requests_ok_total": requests_total,
                "avg_cost_per_ok_24h": round(avg_cost_per_request, 6),
                # Token breakdown
                "tokens_in_24h": tokens_in_24h,
                "tokens_out_24h": tokens_out_24h,
                "tokens_in_7d": tokens_in_7d,
                "tokens_out_7d": tokens_out_7d,
                "tokens_in_28d": tokens_in_28d,
                "tokens_out_28d": tokens_out_28d,
                "tokens_in_total": tokens_in_total,
                "tokens_out_total": tokens_out_total,
                "status": status,
                "note": "Cost calculated per-model from settings.GEMINI_PRICING. 28d window matches Google AI Studio billing.",
                "pricing_source": "config.GEMINI_PRICING",
                # Model usage breakdown (best-effort, may be None)
                **({"model_usage_28d": model_usage_28d} if model_usage_28d else {}),
                **({"model_usage_7d": model_usage_7d} if model_usage_7d else {}),
                **({"model_usage_24h": model_usage_24h} if model_usage_24h else {}),
            }
        except Exception as e:
            logger.warning(f"Could not calculate LLM cost: {e}")
            llm_cost_data = {"provider": "gemini", "status": "error", "error": str(e)}

    # League names - comprehensive fallback for all known leagues
    # Includes EXTENDED_LEAGUES and other common leagues from API-Football
    LEAGUE_NAMES_FALLBACK: dict[int, str] = {
        1: "World Cup",
        2: "Champions League",
        3: "Europa League",
        4: "Euro",
        5: "Nations League",
        9: "Copa América",
        10: "Friendlies",
        11: "Sudamericana",
        13: "Libertadores",
        22: "Gold Cup",
        # Legacy: league_id=28 was previously (incorrectly) used for WCQ CONMEBOL in code.
        # In production DB it may contain SAFF Championship fixtures. Keep explicit to avoid confusion.
        28: "SAFF Championship (legacy)",
        # WC 2026 Qualifiers (correct API-Football league IDs)
        29: "WCQ CAF",
        30: "WCQ AFC",
        31: "WCQ CONCACAF",
        32: "WCQ UEFA",
        33: "WCQ OFC",
        34: "WCQ CONMEBOL",
        37: "WCQ Intercontinental Play-offs",
        39: "Premier League",
        45: "FA Cup",
        61: "Ligue 1",
        71: "Brazil Serie A",
        78: "Bundesliga",
        88: "Eredivisie",
        94: "Primeira Liga",
        128: "Argentina Primera",
        135: "Serie A",
        140: "La Liga",
        143: "Copa del Rey",
        203: "Super Lig",
        239: "Colombia Primera A",
        242: "Ecuador Liga Pro",
        250: "Paraguay Primera - Apertura",
        252: "Paraguay Primera - Clausura",
        253: "MLS",
        262: "Liga MX",
        265: "Chile Primera División",
        268: "Uruguay Primera - Apertura",
        270: "Uruguay Primera - Clausura",
        281: "Peru Primera División",
        299: "Venezuela Primera División",
        344: "Bolivia Primera División",
        848: "Conference League",
    }

    # Merge with COMPETITIONS (if available)
    league_name_by_id: dict[int, str] = LEAGUE_NAMES_FALLBACK.copy()
    try:
        for league_id, comp in (COMPETITIONS or {}).items():
            if league_id is not None and comp is not None:
                name = getattr(comp, "name", None)
                if name:
                    league_name_by_id[int(league_id)] = name
    except Exception:
        pass  # Keep fallback names

    for item in upcoming_by_league:
        lid = item["league_id"]
        item["league_name"] = league_name_by_id.get(lid)

    for item in latest_pit:
        lid = item.get("league_id")
        if isinstance(lid, int):
            item["league_name"] = league_name_by_id.get(lid)

    # =============================================================
    # LIVE SUMMARY STATS (iOS Live Score Polling)
    # =============================================================
    live_summary_stats = {
        "cache_ttl_seconds": _live_summary_cache["ttl"],
        "cache_timestamp": _live_summary_cache["timestamp"],
        "cache_age_seconds": round(time.time() - _live_summary_cache["timestamp"], 1) if _live_summary_cache["timestamp"] else None,
        "cached_live_matches": len(_live_summary_cache["data"]["matches"]) if _live_summary_cache["data"] else 0,
    }

    # =============================================================
    # ML MODEL STATUS
    # =============================================================
    ml_model_info = {
        "loaded": ml_engine.model is not None,
        "version": ml_engine.model_version,
        "source": "file",  # Currently only file-based loading
        "model_path": str(ml_engine.model_path),
    }
    if ml_engine.model is not None:
        try:
            ml_model_info["n_features"] = ml_engine.model.n_features_in_
        except AttributeError:
            pass

    # =============================================================
    # COVERAGE BY LEAGUE (NS matches in next 48h with predictions/odds)
    # =============================================================
    coverage_by_league = []
    try:
        async with AsyncSessionLocal() as session:
            res = await session.execute(
                text(
                    """
                    SELECT
                        m.league_id,
                        COUNT(*) AS total_ns,
                        COUNT(p.id) AS with_prediction,
                        COUNT(m.odds_home) AS with_odds
                    FROM matches m
                    LEFT JOIN predictions p ON p.match_id = m.id
                    WHERE m.status = 'NS'
                      AND m.date >= NOW()
                      AND m.date < NOW() + INTERVAL '48 hours'
                      AND m.league_id IS NOT NULL
                    GROUP BY m.league_id
                    ORDER BY COUNT(*) DESC
                    LIMIT 15
                    """
                )
            )
            for row in res.fetchall():
                lid = int(row[0])
                total = int(row[1])
                with_pred = int(row[2])
                with_odds = int(row[3])
                coverage_by_league.append({
                    "league_id": lid,
                    "league_name": league_name_by_id.get(lid, f"League {lid}"),
                    "total_ns": total,
                    "with_prediction": with_pred,
                    "with_odds": with_odds,
                    "pred_pct": round(with_pred / total * 100, 1) if total > 0 else 0,
                    "odds_pct": round(with_odds / total * 100, 1) if total > 0 else 0,
                })
    except Exception as e:
        logger.warning(f"Could not calculate coverage by league: {e}")

    return {
        "generated_at": now.isoformat(),
        "league_mode": league_mode,
        "tracked_leagues_count": tracked_leagues_count,
        "last_sync_at": last_sync.isoformat() if last_sync else None,
        "budget": budget_status,
        "sentry": sentry_health,
        "pit": {
            "live_60m": pit_live_60m,
            "live_24h": pit_live_24h,
            "delta_to_kickoff_60m": pit_dko_60m,
            "latest": latest_pit,
        },
        "movement": {
            "lineup_movement_24h": lineup_movement_24h,
            "market_movement_24h": market_movement_24h,
        },
        "stats_backfill": {
            "finished_72h_with_stats": stats_with,
            "finished_72h_missing_stats": stats_missing,
        },
        "upcoming": {
            "by_league_24h": upcoming_by_league,
        },
        "progress": progress_metrics,
        "predictions_health": predictions_health,
        "fastpath_health": fastpath_health,
        "model_performance": model_performance,
        "telemetry": telemetry_data,
        "llm_cost": llm_cost_data,
        "shadow_mode": shadow_mode_data,
        "sensor_b": sensor_b_data,
        "extc_shadow": extc_shadow_data,
        "rerun_serving": rerun_serving_data,
        "jobs_health": jobs_health_data,
        "sota_enrichment": sota_enrichment_data,
        "titan": await _calculate_titan_summary(),
        "coverage_by_league": coverage_by_league,
        "ml_model": ml_model_info,
        "live_summary": live_summary_stats,
        "db_pool": get_pool_status(),
        "providers": _get_providers_health(),
    }


async def _get_cached_ops_data(blocking: bool = True) -> dict:
    now = time.time()
    if _ops_dashboard_cache["data"] and (now - _ops_dashboard_cache["timestamp"]) < _ops_dashboard_cache["ttl"]:
        return _ops_dashboard_cache["data"]

    # Non-blocking mode: return stale cache and refresh in background.
    # Used by /dashboard/ops.json to avoid heavy DB work in request path.
    if not blocking:
        if _ops_dashboard_cache["data"] is not None:
            _schedule_ops_dashboard_cache_refresh(reason="stale_nonblocking")
            return _ops_dashboard_cache["data"]
        # Cold start: schedule refresh and return a minimal placeholder
        _schedule_ops_dashboard_cache_refresh(reason="cold_start_nonblocking")
        return {
            "generated_at": datetime.utcnow().isoformat(),
            "status": "warming_cache",
            "note": "OPS cache warming in background (non-blocking). Retry in a few seconds.",
        }

    # Blocking mode (HTML dashboard): refresh cache, but avoid duplicate heavy work
    # if a background refresh is already running.
    if _ops_dashboard_cache.get("refreshing") and _ops_dashboard_cache.get("data") is not None:
        return _ops_dashboard_cache["data"]

    # Respect backoff window after failures: serve stale if we have it.
    next_after = float(_ops_dashboard_cache.get("next_refresh_after") or 0)
    if next_after and now < next_after and _ops_dashboard_cache.get("data") is not None:
        return _ops_dashboard_cache["data"]

    await _refresh_ops_dashboard_cache(reason="blocking_request")
    if _ops_dashboard_cache.get("data") is not None:
        return _ops_dashboard_cache["data"]

    # Extremely rare: still no data (DB down on cold start). Fail-soft.
    return {
        "generated_at": datetime.utcnow().isoformat(),
        "status": "unavailable",
        "note": "Could not load OPS data (DB unavailable).",
    }






@app.get("/dashboard/ops.json")
async def ops_dashboard_json(request: Request):
    if not _verify_dashboard_token(request):
        raise HTTPException(status_code=401, detail="Dashboard access requires valid token.")
    # Non-blocking: avoid running the heavy ops query bundle in the request path.
    # Return stale cache (if any) and refresh in background.
    data = await _get_cached_ops_data(blocking=False)
    return {
        "data": data,
        "cache_age_seconds": round(time.time() - _ops_dashboard_cache["timestamp"], 1) if _ops_dashboard_cache["timestamp"] else None,
        "cache_ttl_seconds": _ops_dashboard_cache.get("ttl"),
        "cache_refreshing": bool(_ops_dashboard_cache.get("refreshing")),
        "cache_last_refresh_error": _ops_dashboard_cache.get("last_refresh_error"),
        "cache_last_refresh_duration_ms": _ops_dashboard_cache.get("last_refresh_duration_ms"),
        "cache_last_refresh_finished_at": _ops_dashboard_cache.get("last_refresh_finished_at"),
    }


# -----------------------------------------------------------------------------
# Upcoming Matches for Dashboard Overview Card
# -----------------------------------------------------------------------------
_upcoming_matches_cache = {
    "data": None,
    "timestamp": 0,
    "ttl": 60,  # seconds - recommended by auditor
}


@app.get("/dashboard/upcoming_matches.json")
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
    if not _verify_dashboard_token(request):
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


@app.get("/dashboard/matches.json")
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
    if not _verify_dashboard_token(request):
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
        .outerjoin(Prediction, Prediction.match_id == Match.id)
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


@app.get("/dashboard/jobs.json")
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
    if not _verify_dashboard_token(request):
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


@app.get("/dashboard/data_quality.json")
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
    if not _verify_dashboard_token(request):
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


@app.get("/dashboard/data_quality/{check_id}.json")
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
    if not _verify_dashboard_token(request):
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


# =============================================================================
# DASHBOARD SETTINGS (read-only operational settings, no secrets)
# =============================================================================

# Cache for settings endpoints
_settings_summary_cache: dict = {"data": None, "timestamp": 0, "ttl": 300}
_settings_flags_cache: dict = {"data": None, "timestamp": 0, "ttl": 60}
_settings_models_cache: dict = {"data": None, "timestamp": 0, "ttl": 300}

# SECURITY: Secrets that MUST NEVER appear in settings responses
_SETTINGS_SECRET_KEYS = frozenset({
    "DATABASE_URL", "RAPIDAPI_KEY", "API_KEY", "DASHBOARD_TOKEN",
    "RUNPOD_API_KEY", "GEMINI_API_KEY", "METRICS_BEARER_TOKEN",
    "SMTP_PASSWORD", "OPS_ADMIN_PASSWORD", "OPS_SESSION_SECRET",
    "SENTRY_AUTH_TOKEN", "FUTBOLSTATS_API_KEY", "X_API_KEY",
})


def _is_env_configured(key: str) -> bool:
    """Check if an environment variable is configured (non-empty)."""
    import os
    val = os.environ.get(key, "")
    return bool(val and val.strip())


def _get_known_feature_flags() -> list[dict]:
    """
    Return list of known feature flags with metadata.

    NOTE: Values are bool or None, never secret strings.
    """
    import os

    flags = [
        # LLM/Narratives
        {"key": "FASTPATH_ENABLED", "scope": "llm", "description": "Enable FastPath LLM narrative generation"},
        {"key": "FASTPATH_DRY_RUN", "scope": "llm", "description": "FastPath dry-run mode (no writes)"},
        {"key": "GEMINI_ENABLED", "scope": "llm", "description": "Use Gemini as LLM provider"},
        {"key": "RUNPOD_ENABLED", "scope": "llm", "description": "Use RunPod as LLM provider"},
        # SOTA
        {"key": "SOTA_SOFASCORE_ENABLED", "scope": "sota", "description": "Enable Sofascore XI capture"},
        {"key": "SOTA_SOFASCORE_REFS_ENABLED", "scope": "sota", "description": "Enable Sofascore refs discovery"},
        {"key": "SOTA_WEATHER_ENABLED", "scope": "sota", "description": "Enable weather capture"},
        {"key": "SOTA_VENUE_GEO_ENABLED", "scope": "sota", "description": "Enable venue geocoding"},
        {"key": "SOTA_UNDERSTAT_ENABLED", "scope": "sota", "description": "Enable Understat xG capture"},
        # Sensor/Shadow
        {"key": "SENSOR_B_ENABLED", "scope": "sensor", "description": "Enable Sensor B calibration"},
        {"key": "SHADOW_MODE_ENABLED", "scope": "sensor", "description": "Enable shadow mode predictions"},
        {"key": "SHADOW_TWO_STAGE_ENABLED", "scope": "sensor", "description": "Enable two-stage shadow architecture"},
        # Jobs
        {"key": "SCHEDULER_ENABLED", "scope": "jobs", "description": "Enable background scheduler"},
        {"key": "ODDS_SYNC_ENABLED", "scope": "jobs", "description": "Enable odds sync job"},
        {"key": "STATS_BACKFILL_ENABLED", "scope": "jobs", "description": "Enable stats backfill job"},
        # Predictions
        {"key": "PREDICTIONS_ENABLED", "scope": "predictions", "description": "Enable prediction generation"},
        {"key": "PREDICTIONS_TWO_STAGE", "scope": "predictions", "description": "Use two-stage prediction model"},
        # Other
        {"key": "DEBUG", "scope": "other", "description": "Debug mode enabled"},
        {"key": "SENTRY_ENABLED", "scope": "other", "description": "Enable Sentry error tracking"},
    ]

    result = []
    for flag in flags:
        key = flag["key"]
        raw_val = os.environ.get(key, "").lower().strip()

        # Determine enabled state
        if raw_val in ("true", "1", "yes", "on"):
            enabled = True
        elif raw_val in ("false", "0", "no", "off"):
            enabled = False
        elif raw_val == "":
            enabled = None  # Not set
        else:
            enabled = None  # Unknown value

        result.append({
            "key": key,
            "enabled": enabled,
            "scope": flag["scope"],
            "description": flag["description"],
            "source": "env" if raw_val else "default",
        })

    return result


@app.get("/dashboard/settings/summary.json")
async def dashboard_settings_summary(request: Request):
    """
    Read-only summary of operational settings.

    Auth: X-Dashboard-Token required.
    TTL: 300s cache.

    SECURITY: No secrets or PII in response. Only configured: true/false.
    """
    if not _verify_dashboard_token(request):
        raise HTTPException(status_code=401, detail="Dashboard access requires valid token.")

    now = time.time()
    cache = _settings_summary_cache

    # Check cache
    if cache["data"] and (now - cache["timestamp"]) < cache["ttl"]:
        return {
            "generated_at": cache["data"]["generated_at"],
            "cached": True,
            "cache_age_seconds": round(now - cache["timestamp"], 1),
            "data": cache["data"]["payload"],
        }

    # Build fresh data
    generated_at = datetime.utcnow().isoformat() + "Z"

    try:
        integrations = {
            "rapidapi": {
                "configured": _is_env_configured("RAPIDAPI_KEY") or _is_env_configured("API_FOOTBALL_KEY"),
                "source": "env",
            },
            "sentry": {
                "configured": _is_env_configured("SENTRY_DSN"),
                "source": "env",
            },
            "metrics": {
                "configured": _is_env_configured("METRICS_BEARER_TOKEN"),
                "source": "env",
            },
            "gemini": {
                "configured": _is_env_configured("GEMINI_API_KEY"),
                "source": "env",
            },
            "runpod": {
                "configured": _is_env_configured("RUNPOD_API_KEY"),
                "source": "env",
            },
            "database": {
                "configured": _is_env_configured("DATABASE_URL"),
                "source": "env",
            },
        }

        payload = {
            "readonly": True,
            "sections": ["general", "feature_flags", "model_versions", "integrations"],
            "notes": "Read-only operational settings. No secrets returned.",
            "links": [
                {"title": "Ops Dashboard", "url": "/dashboard/ops.json"},
                {"title": "Data Quality", "url": "/dashboard/data_quality.json"},
                {"title": "Feature Flags", "url": "/dashboard/settings/feature_flags.json"},
                {"title": "Model Versions", "url": "/dashboard/settings/model_versions.json"},
            ],
            "integrations": integrations,
        }

        # Update cache
        cache["data"] = {"generated_at": generated_at, "payload": payload}
        cache["timestamp"] = now

        return {
            "generated_at": generated_at,
            "cached": False,
            "cache_age_seconds": 0,
            "data": payload,
        }

    except Exception as e:
        logger.error(f"[SETTINGS] summary.json error: {e}")
        return {
            "generated_at": generated_at,
            "cached": False,
            "cache_age_seconds": 0,
            "data": {
                "readonly": True,
                "sections": [],
                "notes": "Read-only operational settings. No secrets returned.",
                "status": "degraded",
                "error": str(e)[:100],
            },
        }


@app.get("/dashboard/settings/feature_flags.json")
async def dashboard_settings_feature_flags(
    request: Request,
    q: str | None = None,
    enabled: bool | None = None,
    scope: str | None = None,
    page: int = 1,
    limit: int = 50,
):
    """
    Read-only list of feature flags.

    Auth: X-Dashboard-Token required.
    TTL: 60s cache (flags can change with deploy).

    Query params:
    - q: Search by key or description
    - enabled: Filter by true/false
    - scope: Filter by scope (llm, sota, jobs, sensor, predictions, other)
    - page: Page number (default 1)
    - limit: Items per page (default 50, max 100)

    SECURITY: No secrets. Only boolean enabled state.
    """
    if not _verify_dashboard_token(request):
        raise HTTPException(status_code=401, detail="Dashboard access requires valid token.")

    # Clamp limit
    limit = min(max(1, limit), 100)
    page = max(1, page)

    now = time.time()
    cache = _settings_flags_cache
    generated_at = datetime.utcnow().isoformat() + "Z"

    try:
        # Get all flags (always fresh since they depend on env)
        all_flags = _get_known_feature_flags()

        # Apply filters
        filtered = all_flags

        if q:
            q_lower = q.lower()
            filtered = [
                f for f in filtered
                if q_lower in f["key"].lower() or q_lower in f["description"].lower()
            ]

        if enabled is not None:
            filtered = [f for f in filtered if f["enabled"] == enabled]

        if scope:
            filtered = [f for f in filtered if f["scope"] == scope]

        # Pagination
        total = len(filtered)
        pages = (total + limit - 1) // limit if limit > 0 else 1
        start = (page - 1) * limit
        end = start + limit
        paginated = filtered[start:end]

        return {
            "generated_at": generated_at,
            "cached": False,  # Always fresh for flags
            "cache_age_seconds": 0,
            "data": {
                "flags": paginated,
                "total": total,
                "page": page,
                "limit": limit,
                "pages": pages,
            },
        }

    except Exception as e:
        logger.error(f"[SETTINGS] feature_flags.json error: {e}")
        return {
            "generated_at": generated_at,
            "cached": False,
            "cache_age_seconds": 0,
            "data": {
                "flags": [],
                "total": 0,
                "page": page,
                "limit": limit,
                "pages": 0,
                "status": "degraded",
                "error": str(e)[:100],
            },
        }


@app.get("/dashboard/settings/model_versions.json")
async def dashboard_settings_model_versions(request: Request):
    """
    Read-only list of ML model versions.

    Auth: X-Dashboard-Token required.
    TTL: 300s cache.

    SECURITY: No secrets. Only version strings.
    """
    if not _verify_dashboard_token(request):
        raise HTTPException(status_code=401, detail="Dashboard access requires valid token.")

    now = time.time()
    cache = _settings_models_cache
    generated_at = datetime.utcnow().isoformat() + "Z"

    # Check cache
    if cache["data"] and (now - cache["timestamp"]) < cache["ttl"]:
        return {
            "generated_at": cache["data"]["generated_at"],
            "cached": True,
            "cache_age_seconds": round(now - cache["timestamp"], 1),
            "data": cache["data"]["payload"],
        }

    try:
        import os

        # Get model versions from settings/env
        baseline_version = getattr(settings, "MODEL_VERSION", None) or os.environ.get("MODEL_VERSION", "v1.0.0")
        architecture = os.environ.get("MODEL_ARCHITECTURE", "baseline")
        shadow_version = os.environ.get("SHADOW_MODEL_VERSION", "disabled")
        shadow_architecture = os.environ.get("SHADOW_ARCHITECTURE", "disabled")

        # Try to get updated_at from model files if available
        model_updated_at = None
        try:
            import glob
            model_files = glob.glob("models/xgb_*.json")
            if model_files:
                import os.path
                latest_file = max(model_files, key=os.path.getmtime)
                mtime = os.path.getmtime(latest_file)
                model_updated_at = datetime.utcfromtimestamp(mtime).isoformat() + "Z"
        except Exception:
            pass

        models = [
            {
                "name": "baseline",
                "version": baseline_version,
                "source": "settings",
                "updated_at": model_updated_at,
            },
            {
                "name": "architecture",
                "version": architecture,
                "source": "env",
                "updated_at": None,
            },
            {
                "name": "shadow_version",
                "version": shadow_version,
                "source": "env",
                "updated_at": None,
            },
            {
                "name": "shadow_architecture",
                "version": shadow_architecture,
                "source": "env",
                "updated_at": None,
            },
        ]

        payload = {"models": models}

        # Update cache
        cache["data"] = {"generated_at": generated_at, "payload": payload}
        cache["timestamp"] = now

        return {
            "generated_at": generated_at,
            "cached": False,
            "cache_age_seconds": 0,
            "data": payload,
        }

    except Exception as e:
        logger.error(f"[SETTINGS] model_versions.json error: {e}")
        return {
            "generated_at": generated_at,
            "cached": False,
            "cache_age_seconds": 0,
            "data": {
                "models": [],
                "status": "degraded",
                "error": str(e)[:100],
            },
        }


# =============================================================================
# IA FEATURES SETTINGS (dynamic LLM configuration)
# =============================================================================

_ia_features_cache: dict = {"data": None, "timestamp": 0, "ttl": 30}


@app.get("/dashboard/settings/ia-features.json")
async def dashboard_settings_ia_features_get(request: Request):
    """
    Get IA Features configuration.

    Auth: X-Dashboard-Token required.
    TTL: 30s cache (short for config changes).

    Returns:
      - narratives_enabled: bool | null (null = inherit from env)
      - narrative_feedback_enabled: bool (placeholder for Phase 2)
      - primary_model: str (model key from LLM_MODELS)
      - temperature: float
      - max_tokens: int
      - available_models: list of model info with pricing
    """
    if not _verify_dashboard_token(request):
        raise HTTPException(status_code=401, detail="Dashboard access requires valid token.")

    now = time.time()
    cache = _ia_features_cache
    generated_at = datetime.utcnow().isoformat() + "Z"

    # Check cache
    if cache["data"] and (now - cache["timestamp"]) < cache["ttl"]:
        return {
            "generated_at": cache["data"]["generated_at"],
            "cached": True,
            "cache_age_seconds": round(now - cache["timestamp"], 1),
            "data": cache["data"]["payload"],
        }

    try:
        from app.config import get_ia_features_config, LLM_MODELS

        async with AsyncSessionLocal() as session:
            ia_config = await get_ia_features_config(session)

        # Build available_models list from catalog
        available_models = [
            {
                "id": model_id,
                "display_name": info["display_name"],
                "provider": info["provider"],
                "input_price": info["input_price_per_1m"],
                "output_price": info["output_price_per_1m"],
                "max_tokens": info["max_tokens"],
            }
            for model_id, info in LLM_MODELS.items()
        ]

        # Compute effective state (for UI display)
        effective_enabled = ia_config.get("narratives_enabled")
        if effective_enabled is None:
            effective_enabled = settings.FASTPATH_ENABLED

        payload = {
            **ia_config,
            "effective_enabled": effective_enabled,  # Resolved value after inheritance
            "env_fastpath_enabled": settings.FASTPATH_ENABLED,  # For "Inherit" display
            "available_models": available_models,
        }

        # Update cache
        cache["data"] = {"generated_at": generated_at, "payload": payload}
        cache["timestamp"] = now

        return {
            "generated_at": generated_at,
            "cached": False,
            "cache_age_seconds": 0,
            "data": payload,
        }

    except Exception as e:
        logger.error(f"[SETTINGS] ia-features GET error: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch IA features config")


@app.patch("/dashboard/settings/ia-features.json")
async def dashboard_settings_ia_features_patch(request: Request):
    """
    Update IA Features configuration.

    Auth: X-Dashboard-Token required.

    Allowed fields:
      - narratives_enabled: bool | null
      - primary_model: str (must be valid key in LLM_MODELS)
      - temperature: float (0.0 - 1.0)
      - max_tokens: int (100 - 131072)

    Note: narrative_feedback_enabled is read-only (Phase 2 placeholder).
    """
    if not _verify_dashboard_token(request):
        raise HTTPException(status_code=401, detail="Dashboard access requires valid token.")

    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    from app.config import get_ia_features_config, LLM_MODELS
    from sqlalchemy import text

    # Whitelist of updatable fields
    allowed_fields = {"narratives_enabled", "primary_model", "temperature", "max_tokens"}
    updates = {k: v for k, v in body.items() if k in allowed_fields}

    if not updates:
        raise HTTPException(status_code=400, detail="No valid fields to update")

    # Validate primary_model if provided
    if "primary_model" in updates:
        if updates["primary_model"] not in LLM_MODELS:
            valid_models = list(LLM_MODELS.keys())
            raise HTTPException(
                status_code=400,
                detail=f"Invalid model. Valid options: {valid_models}"
            )

    # Validate temperature if provided
    if "temperature" in updates:
        temp = updates["temperature"]
        if not isinstance(temp, (int, float)) or temp < 0.0 or temp > 1.0:
            raise HTTPException(status_code=400, detail="temperature must be 0.0 - 1.0")

    # Validate max_tokens if provided
    if "max_tokens" in updates:
        tokens = updates["max_tokens"]
        if not isinstance(tokens, int) or tokens < 100 or tokens > 131072:
            raise HTTPException(status_code=400, detail="max_tokens must be 100 - 131072")

    # Validate narratives_enabled (must be bool or null)
    if "narratives_enabled" in updates:
        val = updates["narratives_enabled"]
        if val is not None and not isinstance(val, bool):
            raise HTTPException(status_code=400, detail="narratives_enabled must be true, false, or null")

    try:
        async with AsyncSessionLocal() as session:
            # Get current config
            current = await get_ia_features_config(session)

            # Merge updates
            new_config = {**current, **updates}

            # Upsert ops_settings
            await session.execute(
                text("""
                    INSERT INTO ops_settings (key, value, updated_at, updated_by)
                    VALUES ('ia_features', :value, NOW(), 'dashboard')
                    ON CONFLICT (key) DO UPDATE SET
                        value = :value,
                        updated_at = NOW(),
                        updated_by = 'dashboard'
                """),
                {"value": json.dumps(new_config)}
            )
            await session.commit()

            # Invalidate cache
            _ia_features_cache["data"] = None

            logger.info(f"[SETTINGS] IA Features updated: {updates}")

            return {
                "generated_at": datetime.utcnow().isoformat() + "Z",
                "data": new_config,
                "updated_fields": list(updates.keys()),
            }

    except Exception as e:
        logger.error(f"[SETTINGS] ia-features PATCH error: {e}")
        raise HTTPException(status_code=500, detail="Failed to update IA features config")


# -----------------------------------------------------------------------------
# IA Features: Visibility Endpoints (Fase 2)
# -----------------------------------------------------------------------------


@app.get("/dashboard/settings/ia-features/prompt-template.json")
async def ia_features_prompt_template(request: Request):
    """
    Returns the current LLM prompt template for narrative generation.

    Auth: X-Dashboard-Token required.

    Returns:
    - version: Current prompt version (e.g., "v11")
    - prompt_template: Full prompt string (with placeholders)
    - char_count: Character count
    - notes: Description
    """
    if not _verify_dashboard_token(request):
        raise HTTPException(status_code=401, detail="Dashboard access requires valid token.")

    try:
        from app.llm.narrative_generator import build_narrative_prompt

        # Build prompt with dummy data to show template structure
        dummy_match_data = {
            "match_id": 0,
            "home_team": "{HOME_TEAM}",
            "away_team": "{AWAY_TEAM}",
            "home_team_id": None,
            "away_team_id": None,
            "league_name": "{LEAGUE}",
            "date": "{DATE}",
            "home_goals": 0,
            "away_goals": 0,
            "venue": {"name": "{VENUE}", "city": "{CITY}"},
            "stats": {
                "home": {"possession": "{POSS_H}", "shots": "{SHOTS_H}"},
                "away": {"possession": "{POSS_A}", "shots": "{SHOTS_A}"},
            },
            "prediction": {
                "selection": "{SELECTION}",
                "confidence": 0.0,
                "home_prob": 0.0,
                "draw_prob": 0.0,
                "away_prob": 0.0,
            },
            "events": [],
            "market_odds": {"home": 0.0, "draw": 0.0, "away": 0.0},
            "derived_facts": {},
            "narrative_style": {},
        }

        prompt, _, _ = build_narrative_prompt(dummy_match_data)

        return {
            "version": "v11",
            "prompt_template": prompt,
            "char_count": len(prompt),
            "notes": "Prompt v11 para narrativas post-partido. Placeholders marcados con {PLACEHOLDER}.",
        }

    except Exception as e:
        logger.error(f"[SETTINGS] prompt-template error: {e}")
        raise HTTPException(status_code=500, detail="Failed to get prompt template")


@app.get("/dashboard/settings/ia-features/preview/{match_id}.json")
async def ia_features_preview(
    request: Request,
    match_id: int,
    session: AsyncSession = Depends(get_async_session),
):
    """
    Preview the LLM payload for a specific match (without calling the LLM).

    Auth: X-Dashboard-Token required.

    Returns:
    - match_id: Match ID
    - match_label: "Home vs Away"
    - prompt_preview: Full prompt that would be sent to LLM
    - match_data: Structured data used to build the prompt
    """
    if not _verify_dashboard_token(request):
        raise HTTPException(status_code=401, detail="Dashboard access requires valid token.")

    try:
        from sqlalchemy.orm import selectinload
        from app.llm.narrative_generator import build_narrative_prompt

        # Get match with teams
        result = await session.execute(
            select(Match)
            .options(selectinload(Match.home_team), selectinload(Match.away_team))
            .where(Match.id == match_id)
        )
        match = result.scalar_one_or_none()

        if not match:
            raise HTTPException(status_code=404, detail=f"Match {match_id} not found")

        # Get prediction
        pred_result = await session.execute(
            select(Prediction)
            .where(Prediction.match_id == match_id)
            .order_by(Prediction.created_at.desc())
            .limit(1)
        )
        prediction = pred_result.scalar_one_or_none()

        # Get odds from odds_history
        odds_result = await session.execute(
            select(OddsHistory)
            .where(OddsHistory.match_id == match_id)
            .order_by(OddsHistory.recorded_at.desc())
            .limit(1)
        )
        odds_row = odds_result.scalar_one_or_none()

        # Build match_data dict
        home_name = match.home_team.name if match.home_team else "Local"
        away_name = match.away_team.name if match.away_team else "Visitante"

        # Stats come from match.stats JSON field
        stats_dict = {"home": {}, "away": {}}
        if match.stats:
            stats_dict = match.stats

        # Events come from match.events JSON field (limit to 10)
        events_list = []
        if match.events:
            events_list = match.events[:10]

        # Market odds from odds_history or match.odds_*
        market_odds = {}
        if odds_row:
            market_odds = {
                "home": odds_row.odds_home,
                "draw": odds_row.odds_draw,
                "away": odds_row.odds_away,
            }
        elif match.odds_home:
            market_odds = {
                "home": match.odds_home,
                "draw": match.odds_draw,
                "away": match.odds_away,
            }

        prediction_dict = {}
        if prediction:
            # Calculate selection and confidence from probabilities
            probs = {
                "home": prediction.home_prob,
                "draw": prediction.draw_prob,
                "away": prediction.away_prob,
            }
            selection = max(probs, key=probs.get)
            confidence = probs[selection]
            prediction_dict = {
                "selection": selection,
                "confidence": confidence,
                "home_prob": prediction.home_prob,
                "draw_prob": prediction.draw_prob,
                "away_prob": prediction.away_prob,
            }

        # Get league name from COMPETITIONS constant
        league_name = ""
        league_info = COMPETITIONS.get(match.league_id)
        if league_info:
            league_name = league_info.name or ""

        match_data = {
            "match_id": match.id,
            "home_team": home_name,
            "away_team": away_name,
            "home_team_id": match.home_team_id,
            "away_team_id": match.away_team_id,
            "league_name": league_name,
            "date": match.date.isoformat() if match.date else "",
            "home_goals": match.home_goals or 0,
            "away_goals": match.away_goals or 0,
            "venue": {"name": match.venue_name, "city": match.venue_city} if match.venue_name else {},
            "stats": stats_dict,
            "prediction": prediction_dict,
            "events": events_list,
            "market_odds": market_odds,
            "derived_facts": {},
            "narrative_style": {},
        }

        # Build prompt
        prompt, _, _ = build_narrative_prompt(match_data)

        return {
            "match_id": match.id,
            "match_label": f"{home_name} vs {away_name}",
            "status": match.status,
            "prompt_preview": prompt,
            "match_data": match_data,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[SETTINGS] preview error for match {match_id}: {type(e).__name__}: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate preview")


@app.get("/dashboard/settings/ia-features/call-history.json")
async def ia_features_call_history(
    request: Request,
    limit: int = Query(default=20, ge=1, le=100),
    session: AsyncSession = Depends(get_async_session),
):
    """
    Returns recent LLM narrative generation calls from post_match_audits.

    Auth: X-Dashboard-Token required.

    Query params:
    - limit: Max items to return (default 20, max 100)

    Returns:
    - items: List of recent narrative generations with metrics
    - total: Total count of narratives generated
    """
    if not _verify_dashboard_token(request):
        raise HTTPException(status_code=401, detail="Dashboard access requires valid token.")

    try:
        from app.config import LLM_MODELS

        # Query recent audits with narratives
        query = (
            select(
                PostMatchAudit.outcome_id,
                PostMatchAudit.llm_narrative_model,
                PostMatchAudit.llm_narrative_tokens_in,
                PostMatchAudit.llm_narrative_tokens_out,
                PostMatchAudit.llm_narrative_delay_ms,
                PostMatchAudit.llm_narrative_exec_ms,
                PostMatchAudit.llm_narrative_generated_at,
                PostMatchAudit.llm_narrative_status,
                PostMatchAudit.llm_prompt_version,
                Match.id.label("match_id"),
                Team.name.label("home_team_name"),
            )
            .join(PredictionOutcome, PostMatchAudit.outcome_id == PredictionOutcome.id)
            .join(Match, PredictionOutcome.match_id == Match.id)
            .join(Team, Match.home_team_id == Team.id)
            .where(PostMatchAudit.llm_narrative_generated_at.isnot(None))
            .order_by(PostMatchAudit.llm_narrative_generated_at.desc())
            .limit(limit)
        )

        result = await session.execute(query)
        rows = result.all()

        # Get away team names separately
        match_ids = [r.match_id for r in rows]
        away_query = (
            select(Match.id, Team.name.label("away_team_name"))
            .join(Team, Match.away_team_id == Team.id)
            .where(Match.id.in_(match_ids))
        )
        away_result = await session.execute(away_query)
        away_names = {r.id: r.away_team_name for r in away_result.all()}

        # Count total
        count_query = select(func.count()).select_from(PostMatchAudit).where(
            PostMatchAudit.llm_narrative_generated_at.isnot(None)
        )
        total = (await session.execute(count_query)).scalar() or 0

        items = []
        for row in rows:
            # Calculate cost
            model_key = row.llm_narrative_model or "gemini-2.5-flash-lite"
            model_info = LLM_MODELS.get(model_key, LLM_MODELS.get("gemini-2.5-flash-lite", {}))
            tokens_in = row.llm_narrative_tokens_in or 0
            tokens_out = row.llm_narrative_tokens_out or 0
            cost_usd = (
                (tokens_in * model_info.get("input_price_per_1m", 0.10) / 1_000_000)
                + (tokens_out * model_info.get("output_price_per_1m", 0.40) / 1_000_000)
            )

            away_name = away_names.get(row.match_id, "Visitante")

            items.append({
                "match_id": row.match_id,
                "match_label": f"{row.home_team_name} vs {away_name}",
                "generated_at": row.llm_narrative_generated_at.isoformat() if row.llm_narrative_generated_at else None,
                "model": row.llm_narrative_model,
                "prompt_version": row.llm_prompt_version,
                "tokens_in": tokens_in,
                "tokens_out": tokens_out,
                "latency_ms": row.llm_narrative_delay_ms,
                "exec_ms": row.llm_narrative_exec_ms,
                "cost_usd": round(cost_usd, 6),
                "status": row.llm_narrative_status or "success",
                "audit_url": f"/dashboard/ops/llm_audit/{row.match_id}.json",
            })

        return {
            "items": items,
            "total": total,
            "limit": limit,
        }

    except Exception as e:
        logger.error(f"[SETTINGS] call-history error: {e}")
        raise HTTPException(status_code=500, detail="Failed to get call history")


# Rate limiting for playground (in-memory, resets on restart)
_playground_rate_limits: dict[str, list[datetime]] = {}


def _check_playground_rate_limit(token: str) -> tuple[bool, int, datetime]:
    """
    Check rate limit for playground endpoint.

    Returns: (allowed, remaining, reset_at)
    """
    now = datetime.utcnow()
    hour_ago = now - timedelta(hours=1)

    # Clean old calls
    calls = _playground_rate_limits.get(token, [])
    calls = [c for c in calls if c > hour_ago]

    if len(calls) >= 10:
        reset_at = calls[0] + timedelta(hours=1)
        return False, 0, reset_at

    calls.append(now)
    _playground_rate_limits[token] = calls
    return True, 10 - len(calls), now + timedelta(hours=1)


class PlaygroundRequest(BaseModel):
    """Request body for playground endpoint."""

    match_id: int = Field(..., description="Match ID to generate narrative for")
    temperature: float | None = Field(default=None, ge=0.0, le=1.0, description="Temperature (0.0-1.0)")
    max_tokens: int | None = Field(default=None, ge=100, le=131072, description="Max tokens")
    model: str | None = Field(default=None, description="Model to use")


@app.post("/dashboard/settings/ia-features/playground")
async def ia_features_playground(
    request: Request,
    body: PlaygroundRequest,
    session: AsyncSession = Depends(get_async_session),
):
    """
    LLM Playground: Generate a narrative for a match with custom parameters.

    Auth: X-Dashboard-Token required.
    Rate limit: 10 calls/hour per token.

    This endpoint actually calls the LLM and incurs costs.
    Narratives generated here are NOT persisted to the database.
    """
    if not _verify_dashboard_token(request):
        raise HTTPException(status_code=401, detail="Dashboard access requires valid token.")

    try:
        from sqlalchemy.orm import selectinload
        from app.llm.narrative_generator import build_narrative_prompt
        from app.config import LLM_MODELS, get_ia_features_config
        import time as time_module

        # Get token for rate limiting
        token = request.headers.get("X-Dashboard-Token", "anonymous")

        # Check rate limit
        allowed, remaining, reset_at = _check_playground_rate_limit(token)
        if not allowed:
            return JSONResponse(
                status_code=429,
                content={
                    "error": "Rate limit exceeded (10 calls/hour)",
                    "rate_limit": {
                        "remaining": 0,
                        "reset_at": reset_at.isoformat(),
                    },
                },
            )

        # Validate match exists
        result = await session.execute(
            select(Match)
            .options(selectinload(Match.home_team), selectinload(Match.away_team))
            .where(Match.id == body.match_id)
        )
        match = result.scalar_one_or_none()

        if not match:
            raise HTTPException(status_code=404, detail=f"Match {body.match_id} not found")

        # Validate match is finished
        if match.status not in ("FT", "AET", "PEN"):
            raise HTTPException(
                status_code=400,
                detail=f"Match must be finished (status: {match.status})"
            )

        # Validate match has stats
        if not match.stats:
            raise HTTPException(
                status_code=400,
                detail="Match has no stats available"
            )

        # Get prediction
        pred_result = await session.execute(
            select(Prediction)
            .where(Prediction.match_id == body.match_id)
            .order_by(Prediction.created_at.desc())
            .limit(1)
        )
        prediction = pred_result.scalar_one_or_none()

        # Get odds
        odds_result = await session.execute(
            select(OddsHistory)
            .where(OddsHistory.match_id == body.match_id)
            .order_by(OddsHistory.recorded_at.desc())
            .limit(1)
        )
        odds_row = odds_result.scalar_one_or_none()

        # Build match_data dict
        home_name = match.home_team.name if match.home_team else "Local"
        away_name = match.away_team.name if match.away_team else "Visitante"

        events_list = match.events[:10] if match.events else []

        market_odds = {}
        if odds_row:
            market_odds = {
                "home": odds_row.odds_home,
                "draw": odds_row.odds_draw,
                "away": odds_row.odds_away,
            }
        elif match.odds_home:
            market_odds = {
                "home": match.odds_home,
                "draw": match.odds_draw,
                "away": match.odds_away,
            }

        prediction_dict = {}
        if prediction:
            # Calculate selection and confidence from probabilities
            probs = {
                "home": prediction.home_prob,
                "draw": prediction.draw_prob,
                "away": prediction.away_prob,
            }
            selection = max(probs, key=probs.get)
            confidence = probs[selection]
            prediction_dict = {
                "selection": selection,
                "confidence": confidence,
                "home_prob": prediction.home_prob,
                "draw_prob": prediction.draw_prob,
                "away_prob": prediction.away_prob,
            }

        # Get league name
        league_name = ""
        league_info = COMPETITIONS.get(match.league_id)
        if league_info:
            league_name = league_info.name or ""

        match_data = {
            "match_id": match.id,
            "home_team": home_name,
            "away_team": away_name,
            "home_team_id": match.home_team_id,
            "away_team_id": match.away_team_id,
            "league_name": league_name,
            "date": match.date.isoformat() if match.date else "",
            "home_goals": match.home_goals or 0,
            "away_goals": match.away_goals or 0,
            "venue": {"name": match.venue_name, "city": match.venue_city} if match.venue_name else {},
            "stats": match.stats,
            "prediction": prediction_dict,
            "events": events_list,
            "market_odds": market_odds,
            "derived_facts": {},
            "narrative_style": {},
        }

        # Get current config for defaults
        ia_config = await get_ia_features_config(session)

        # Resolve parameters
        model_to_use = body.model or ia_config.get("primary_model", "gemini-2.5-flash-lite")
        temperature_to_use = body.temperature if body.temperature is not None else ia_config.get("temperature", 0.7)
        max_tokens_to_use = body.max_tokens or ia_config.get("max_tokens", 4096)

        # Validate model exists
        if model_to_use not in LLM_MODELS:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown model: {model_to_use}"
            )

        model_info = LLM_MODELS[model_to_use]

        # Build prompt
        prompt, _, _ = build_narrative_prompt(match_data)

        # Generate narrative using GeminiClient directly
        # (Playground only supports Gemini models for now)
        from app.llm.gemini_client import GeminiClient
        from app.llm.narrative_generator import parse_json_response

        start_time = time_module.time()

        # Create Gemini client and generate
        client = GeminiClient()
        try:
            result = await client.generate(
                prompt=prompt,
                max_tokens=max_tokens_to_use,
                temperature=temperature_to_use,
            )
        finally:
            await client.close()

        latency_ms = int((time_module.time() - start_time) * 1000)

        if result.status != "COMPLETED":
            raise HTTPException(
                status_code=500,
                detail=f"LLM generation failed: {result.error or result.status}"
            )

        tokens_in = result.tokens_in
        tokens_out = result.tokens_out

        parsed = parse_json_response(result.text)

        if not parsed:
            raise HTTPException(
                status_code=500,
                detail="Failed to parse LLM response"
            )

        # Calculate cost
        cost_usd = (
            (tokens_in * model_info.get("input_price_per_1m", 0.10) / 1_000_000)
            + (tokens_out * model_info.get("output_price_per_1m", 0.40) / 1_000_000)
        )

        # Extract narrative
        narrative_data = parsed.get("narrative", {})

        return {
            "narrative": {
                "title": narrative_data.get("title", ""),
                "body": narrative_data.get("body", ""),
                "key_factors": narrative_data.get("key_factors", []),
            },
            "model_used": model_to_use,
            "metrics": {
                "tokens_in": tokens_in,
                "tokens_out": tokens_out,
                "latency_ms": latency_ms,
                "cost_usd": round(cost_usd, 6),
            },
            "warnings": [],
            "rate_limit": {
                "remaining": remaining,
                "reset_at": reset_at.isoformat(),
            },
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[SETTINGS] playground error: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate narrative")


# =============================================================================
# DASHBOARD PREDICTIONS (read-only, for ops dashboard)
# =============================================================================

_dashboard_predictions_cache: dict = {"data": None, "timestamp": 0, "ttl": 45}


@app.get("/dashboard/predictions.json")
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
    if not _verify_dashboard_token(request):
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


@app.get("/dashboard/analytics/reports.json")
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
    if not _verify_dashboard_token(request):
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


@app.get("/dashboard/audit_logs.json")
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
    if not _verify_dashboard_token(request):
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


@app.get("/dashboard/team_logos.json")
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
    if not _verify_dashboard_token(request):
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


@app.get("/dashboard/ops/logs.json")
async def ops_dashboard_logs_json(
    request: Request,
    limit: int = OPS_LOG_DEFAULT_LIMIT,
    since_minutes: int = OPS_LOG_DEFAULT_SINCE_MINUTES,
    level: Optional[str] = None,
    mode: Optional[str] = None,
):
    """Filtered in-memory ops logs (copy/paste friendly). Use mode=compact for grouped view."""
    if not _verify_dashboard_token(request):
        raise HTTPException(status_code=401, detail="Dashboard access requires valid token.")
    compact = mode == "compact"
    return {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "limit": limit,
        "since_minutes": since_minutes,
        "level": level,
        "mode": mode,
        "entries": _get_ops_logs(since_minutes=since_minutes, limit=limit, level=level, compact=compact),
    }




# =============================================================================
# OPS ADMIN ENDPOINTS (manual triggers, protected by token)
# =============================================================================


@app.post("/dashboard/ops/rollup")
async def trigger_ops_rollup(request: Request):
    """
    Manually trigger the daily ops rollup job.

    Protected by dashboard token. Use for testing/validation.
    """
    if not _verify_dashboard_token(request):
        raise HTTPException(status_code=401, detail="Dashboard access requires valid token.")

    from app.scheduler import daily_ops_rollup

    result = await daily_ops_rollup()
    return {
        "status": "executed",
        "result": result,
    }


@app.post("/dashboard/ops/odds_sync")
async def trigger_odds_sync(request: Request):
    """
    Manually trigger the odds sync job for upcoming matches.

    Protected by dashboard token. Use for testing/validation or immediate sync.
    """
    if not _verify_dashboard_token(request):
        raise HTTPException(status_code=401, detail="Dashboard access requires valid token.")

    from app.scheduler import sync_odds_for_upcoming_matches
    from app.ops.audit import log_ops_action

    start_time = time.time()
    result = await sync_odds_for_upcoming_matches()
    duration_ms = int((time.time() - start_time) * 1000)

    # Audit log
    try:
        async with AsyncSessionLocal() as audit_session:
            await log_ops_action(
                session=audit_session,
                request=request,
                action="odds_sync",
                params=None,
                result="ok" if result.get("status") == "completed" else "error",
                result_detail={
                    "scanned": result.get("scanned", 0),
                    "updated": result.get("updated", 0),
                    "api_calls": result.get("api_calls", 0),
                },
                duration_ms=duration_ms,
            )
    except Exception as audit_err:
        logger.warning(f"Failed to log audit for odds_sync: {audit_err}")

    return {
        "status": "executed",
        "result": result,
    }


@app.post("/dashboard/ops/sensor_retrain")
async def trigger_sensor_retrain(request: Request):
    """
    Manually trigger Sensor B retrain job.

    Protected by dashboard token. Use after deploy to force immediate retrain
    instead of waiting for the 6h interval.
    """
    if not _verify_dashboard_token(request):
        raise HTTPException(status_code=401, detail="Dashboard access requires valid token.")

    from app.scheduler import retrain_sensor_model

    start_time = time.time()
    result = await retrain_sensor_model()
    duration_ms = int((time.time() - start_time) * 1000)

    return {
        "status": "executed",
        "duration_ms": duration_ms,
        "result": result,
    }


@app.post("/dashboard/ops/sensor_eval")
async def trigger_sensor_eval(request: Request):
    """
    Manually trigger Sensor B evaluation job.

    Protected by dashboard token. Evaluates pending FT matches against
    sensor predictions.
    """
    if not _verify_dashboard_token(request):
        raise HTTPException(status_code=401, detail="Dashboard access requires valid token.")

    from app.scheduler import evaluate_sensor_predictions_job

    start_time = time.time()
    result = await evaluate_sensor_predictions_job()
    duration_ms = int((time.time() - start_time) * 1000)

    return {
        "status": "executed",
        "duration_ms": duration_ms,
        "result": result,
    }


@app.post("/dashboard/ops/shadow_eval")
async def trigger_shadow_eval(request: Request):
    """
    Manually trigger Shadow mode evaluation job.

    Protected by dashboard token. Evaluates pending FT matches against
    shadow predictions.
    """
    if not _verify_dashboard_token(request):
        raise HTTPException(status_code=401, detail="Dashboard access requires valid token.")

    from app.scheduler import evaluate_shadow_predictions

    start_time = time.time()
    result = await evaluate_shadow_predictions()
    duration_ms = int((time.time() - start_time) * 1000)

    return {
        "status": "executed",
        "duration_ms": duration_ms,
        "result": result,
    }


@app.post("/dashboard/ops/stats_backfill")
async def trigger_stats_backfill(request: Request):
    """
    Manually trigger stats backfill job.

    Protected by dashboard token. Use after deploy to force immediate execution
    instead of waiting for the 60min interval.
    """
    if not _verify_dashboard_token(request):
        raise HTTPException(status_code=401, detail="Dashboard access requires valid token.")

    from app.scheduler import capture_finished_match_stats

    start_time = time.time()
    result = await capture_finished_match_stats()
    duration_ms = int((time.time() - start_time) * 1000)

    return {
        "status": "executed",
        "duration_ms": duration_ms,
        "result": result,
    }


@app.post("/dashboard/ops/historical_stats_backfill")
async def trigger_historical_stats_backfill(request: Request):
    """
    Trigger historical stats backfill job for matches since 2023-08-01.

    This endpoint calls the scheduler job which:
    - Processes 500 matches per run (configurable via HISTORICAL_STATS_BACKFILL_BATCH_SIZE)
    - Marks matches without stats as {"_no_stats": true} to skip on future runs
    - Auto-advances through all leagues

    Protected by dashboard token.
    """
    if not _verify_dashboard_token(request):
        raise HTTPException(status_code=401, detail="Dashboard access requires valid token.")

    from app.scheduler import historical_stats_backfill

    start_time = time.time()
    result = await historical_stats_backfill()
    duration_ms = int((time.time() - start_time) * 1000)

    return {
        "status": "executed",
        "duration_ms": duration_ms,
        "result": result,
    }


@app.post("/dashboard/ops/match_link")
async def link_match_to_api_football(
    request: Request,
    match_id: int,
    external_id: int,
    fetch_stats: bool = True,
):
    """
    Link an orphan match to its API-Football fixture_id.

    Orphan matches (external_id=NULL) cannot receive odds or stats from API-Football.
    This endpoint allows manually linking them when the fixture_id is known.

    Args:
        match_id: Our internal match ID
        external_id: API-Football fixture ID
        fetch_stats: If True, also fetch and update stats from API-Football
    """
    if not _verify_dashboard_token(request):
        raise HTTPException(status_code=401, detail="Dashboard access requires valid token.")

    from sqlalchemy import text
    import json

    start_time = time.time()
    result = {"match_id": match_id, "external_id": external_id}

    try:
        # Update external_id first
        async with AsyncSessionLocal() as session:
            # Check if external_id already exists on another match
            check_result = await session.execute(text("""
                SELECT id FROM matches WHERE external_id = :external_id AND id != :match_id
            """), {"match_id": match_id, "external_id": external_id})
            existing = check_result.scalar()
            if existing:
                return {"status": "error", "error": f"external_id {external_id} already exists on match {existing}"}

            update_result = await session.execute(text("""
                UPDATE matches
                SET external_id = :external_id
                WHERE id = :match_id
            """), {"match_id": match_id, "external_id": external_id})
            await session.commit()
            result["external_id_updated"] = True
            result["rows_affected"] = update_result.rowcount

        # Optionally fetch and update stats in separate transaction
        if fetch_stats:
            from app.etl.api_football import APIFootballProvider
            provider = APIFootballProvider()
            try:
                stats_data = await provider.get_fixture_statistics(external_id)
                if stats_data:
                    async with AsyncSessionLocal() as session2:
                        await session2.execute(text("""
                            UPDATE matches
                            SET stats = CAST(:stats_json AS JSON)
                            WHERE id = :match_id
                        """), {"match_id": match_id, "stats_json": json.dumps(stats_data)})
                        await session2.commit()
                    result["stats_updated"] = True
                    result["stats_keys"] = list(stats_data.get("home", {}).keys())
                else:
                    result["stats_updated"] = False
                    result["stats_error"] = "No stats returned from API"
            finally:
                await provider.close()

        duration_ms = int((time.time() - start_time) * 1000)
        return {"status": "ok", "duration_ms": duration_ms, "result": result}

    except Exception as e:
        duration_ms = int((time.time() - start_time) * 1000)
        return {"status": "error", "duration_ms": duration_ms, "error": str(e)}


@app.patch("/dashboard/matches/{match_id}/odds")
async def update_match_odds_manual(
    request: Request,
    match_id: int,
    odds_home: float,
    odds_draw: float,
    odds_away: float,
    source: str = "manual_audit",
):
    """
    Manually update 1X2 odds for a match (audit/backfill purposes).

    Use when API-Football doesn't have odds but we need them for tracking.
    Records source for audit trail.

    Args:
        match_id: Internal match ID
        odds_home: Home win odds (decimal, e.g. 2.50)
        odds_draw: Draw odds (decimal, e.g. 3.20)
        odds_away: Away win odds (decimal, e.g. 2.80)
        source: Source of odds (e.g. "manual_audit_bet365", "sportsgambler")
    """
    if not _verify_dashboard_token(request):
        raise HTTPException(status_code=401, detail="Dashboard access requires valid token.")

    from sqlalchemy import text
    from app.ops.audit import log_ops_action

    # Validate odds are reasonable (1.01 to 100.0)
    for name, value in [("odds_home", odds_home), ("odds_draw", odds_draw), ("odds_away", odds_away)]:
        if not (1.01 <= value <= 100.0):
            raise HTTPException(status_code=400, detail=f"{name} must be between 1.01 and 100.0")

    start_time = time.time()

    try:
        async with AsyncSessionLocal() as session:
            # Verify match exists
            check = await session.execute(
                text("SELECT id, status FROM matches WHERE id = :mid"),
                {"mid": match_id}
            )
            match = check.fetchone()
            if not match:
                raise HTTPException(status_code=404, detail=f"Match {match_id} not found")

            # Update odds
            await session.execute(
                text("""
                    UPDATE matches
                    SET odds_home = :oh, odds_draw = :od, odds_away = :oa,
                        odds_recorded_at = NOW()
                    WHERE id = :mid
                """),
                {"mid": match_id, "oh": odds_home, "od": odds_draw, "oa": odds_away}
            )
            await session.commit()

        duration_ms = int((time.time() - start_time) * 1000)

        # Audit log
        try:
            async with AsyncSessionLocal() as audit_session:
                await log_ops_action(
                    session=audit_session,
                    request=request,
                    action="manual_odds_update",
                    params={"match_id": match_id, "source": source},
                    result="ok",
                    result_detail={
                        "odds_home": odds_home,
                        "odds_draw": odds_draw,
                        "odds_away": odds_away,
                    },
                    duration_ms=duration_ms,
                )
        except Exception as audit_err:
            logger.warning(f"Failed to log audit for manual_odds_update: {audit_err}")

        return {
            "status": "ok",
            "match_id": match_id,
            "odds": {"home": odds_home, "draw": odds_draw, "away": odds_away},
            "source": source,
            "duration_ms": duration_ms,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update odds for match {match_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/dashboard/ops/stats_refresh")
async def trigger_stats_refresh(request: Request, lookback_hours: int = 48, max_calls: int = 100):
    """
    Manually trigger stats refresh for recently finished matches.

    Unlike stats_backfill, this re-fetches stats for ALL recent FT matches,
    even if they already have stats. Captures late events like red cards.

    Args:
        lookback_hours: Hours to look back (default 48 for manual runs)
        max_calls: Max API calls (default 100)
    """
    if not _verify_dashboard_token(request):
        raise HTTPException(status_code=401, detail="Dashboard access requires valid token.")

    from app.scheduler import refresh_recent_ft_stats

    start_time = time.time()
    result = await refresh_recent_ft_stats(lookback_hours=lookback_hours, max_calls=max_calls)
    duration_ms = int((time.time() - start_time) * 1000)

    return {
        "status": "executed",
        "duration_ms": duration_ms,
        "result": result,
    }


@app.post("/dashboard/ops/narratives_regenerate")
async def trigger_narratives_regenerate(
    request: Request,
    lookback_hours: int = 48,
    max_matches: int = 100,
    force: bool = False,
):
    """
    Regenerate LLM narratives for recently finished matches.

    This endpoint resets narratives for matches that had stats refreshed,
    allowing FastPath to regenerate them with updated data.

    Args:
        lookback_hours: Hours to look back for finished matches (default 48)
        max_matches: Maximum matches to process (default 100)
        force: If True, regenerate even if narrative already exists
    """
    if not _verify_dashboard_token(request):
        raise HTTPException(status_code=401, detail="Dashboard access requires valid token.")

    from sqlalchemy import text

    start_time = time.time()
    metrics = {
        "checked": 0,
        "reset": 0,
        "already_pending": 0,
        "no_audit": 0,
        "errors": 0,
    }

    try:
        async with AsyncSessionLocal() as session:
            # Find FT matches in lookback window that have prediction_outcomes and post_match_audits
            result = await session.execute(text("""
                SELECT
                    m.id as match_id,
                    po.id as outcome_id,
                    pma.id as audit_id,
                    pma.llm_narrative_status,
                    ht.name as home_team,
                    at.name as away_team
                FROM matches m
                JOIN prediction_outcomes po ON po.match_id = m.id
                JOIN post_match_audits pma ON pma.outcome_id = po.id
                JOIN teams ht ON ht.id = m.home_team_id
                JOIN teams at ON at.id = m.away_team_id
                WHERE m.status IN ('FT', 'AET', 'PEN')
                  AND m.finished_at >= NOW() - INTERVAL ':lookback hours'
                  AND m.stats IS NOT NULL
                  AND m.stats::text != '{}'
                ORDER BY m.finished_at DESC
                LIMIT :max_matches
            """.replace(":lookback", str(lookback_hours))), {
                "max_matches": max_matches,
            })

            rows = result.fetchall()
            metrics["checked"] = len(rows)

            reset_ids = []
            match_ids = []
            for row in rows:
                if row.llm_narrative_status == "pending":
                    metrics["already_pending"] += 1
                    continue

                if row.llm_narrative_status == "ok" and not force:
                    # Skip if already has narrative and force=False
                    continue

                reset_ids.append(row.audit_id)
                match_ids.append(row.match_id)

            if reset_ids:
                # Reset narrative status to allow regeneration
                await session.execute(text("""
                    UPDATE post_match_audits
                    SET llm_narrative_status = 'pending',
                        llm_narrative_attempts = 0,
                        llm_narrative_json = NULL,
                        llm_output_raw = NULL,
                        llm_prompt_version = NULL,
                        llm_validation_errors = NULL
                    WHERE id = ANY(:ids)
                """), {"ids": reset_ids})

                # Trick: update finished_at to NOW() so FastPath picks them up
                # FastPath has a 90-min lookback, this makes old matches eligible
                await session.execute(text("""
                    UPDATE matches
                    SET finished_at = NOW()
                    WHERE id = ANY(:ids)
                """), {"ids": match_ids})

                await session.commit()
                metrics["reset"] = len(reset_ids)

        duration_ms = int((time.time() - start_time) * 1000)
        return {
            "status": "executed",
            "duration_ms": duration_ms,
            "result": {
                **metrics,
                "message": f"Reset {metrics['reset']} narratives. FastPath will regenerate in next ticks (~2 min each).",
            },
        }

    except Exception as e:
        duration_ms = int((time.time() - start_time) * 1000)
        metrics["errors"] = 1
        return {
            "status": "error",
            "duration_ms": duration_ms,
            "result": {**metrics, "error": str(e)},
        }


# =============================================================================
# PREDICTION RERUN (controlled two-stage model promotion)
# =============================================================================


class PredictionRerunRequest(BaseModel):
    """Request body for predictions rerun endpoint."""
    window_hours: int = Field(default=168, ge=24, le=336, description="Time window (24-336h)")
    dry_run: bool = Field(default=True, description="If True, compute stats but don't save")
    architecture: str = Field(default="two_stage", description="Target architecture")
    max_matches: int = Field(default=500, ge=1, le=1000, description="Max matches to rerun")
    notes: Optional[str] = Field(default=None, description="Optional notes for audit")


@app.post("/dashboard/ops/predictions_rerun")
async def predictions_rerun(request: Request, body: PredictionRerunRequest):
    """
    Manual re-prediction of NS matches with two-stage architecture.

    ONE-OFF operation for controlled model promotion. Requires:
    - Dashboard token authentication
    - dry_run=true first to review changes
    - Saves before/after stats to prediction_reruns table

    Rollback: Set is_active=false on the rerun record (or PREFER_RERUN_PREDICTIONS=false).
    """
    import uuid
    from sqlalchemy import text

    if not _verify_dashboard_token(request):
        raise HTTPException(status_code=401, detail="Dashboard access requires valid token.")

    # GUARDRAIL: Prevent two_stage/shadow predictions from going to predictions table
    # Shadow/two-stage models must use shadow_predictions table, not predictions.
    # This prevents data inconsistency issues discovered in Model Benchmark Tile.
    forbidden_archs = ['two_stage', 'shadow', 'twostage']
    if any(arch in body.architecture.lower() for arch in forbidden_archs):
        raise HTTPException(
            status_code=400,
            detail=(
                f"Architecture '{body.architecture}' cannot write to predictions table. "
                "Shadow/two-stage predictions must use shadow_predictions table. "
                "Use the shadow mode pipeline instead of this endpoint."
            )
        )

    from app.ml.shadow import get_shadow_engine, is_shadow_enabled
    from app.models import Prediction, PredictionRerun
    from app.features import FeatureEngineer

    settings = get_settings()
    run_id = uuid.uuid4()

    async with AsyncSessionLocal() as session:
        try:
            # 1. Validate shadow engine is available
            shadow_engine = get_shadow_engine()
            if not shadow_engine or not shadow_engine.is_loaded:
                return {
                    "status": "error",
                    "error": "Shadow engine (two-stage) not loaded. Train it first.",
                    "hint": "Set MODEL_SHADOW_ARCHITECTURE=two_stage and trigger shadow training."
                }

            # 2. Query NS matches in window
            result = await session.execute(text("""
                SELECT m.id, m.external_id, m.date, m.league_id,
                       m.odds_home, m.odds_draw, m.odds_away
                FROM matches m
                WHERE m.status = 'NS'
                  AND m.date >= NOW()
                  AND m.date <= NOW() + make_interval(hours => :window_hours)
                ORDER BY m.date ASC
                LIMIT :max_matches
            """), {
                "window_hours": body.window_hours,
                "max_matches": body.max_matches,
            })
            ns_matches = result.fetchall()

            if not ns_matches:
                return {
                    "status": "no_matches",
                    "message": f"No NS matches found in {body.window_hours}h window",
                    "run_id": str(run_id),
                }

            match_ids = [m[0] for m in ns_matches]
            matches_with_odds = sum(1 for m in ns_matches if m[4] is not None)

            # 3. Get BEFORE predictions (baseline)
            result = await session.execute(text("""
                SELECT p.match_id, p.model_version, p.home_prob, p.draw_prob, p.away_prob
                FROM predictions p
                WHERE p.match_id = ANY(:match_ids)
                  AND p.model_version = :version
            """), {
                "match_ids": match_ids,
                "version": settings.MODEL_VERSION,
            })
            before_preds = {row[0]: {"home": row[2], "draw": row[3], "away": row[4]} for row in result.fetchall()}

            # 4. Compute BEFORE stats
            before_stats = _compute_prediction_stats(before_preds, "before")

            # 5. Get features for NS matches
            feature_engineer = FeatureEngineer(session=session)
            df = await feature_engineer.get_upcoming_matches_features()

            # Filter to our NS matches
            df_ns = df[df["match_id"].isin(match_ids)].copy()

            if len(df_ns) == 0:
                return {
                    "status": "no_features",
                    "error": "Could not compute features for NS matches",
                    "matches_total": len(match_ids),
                }

            # 6. Generate AFTER predictions (two-stage)
            after_preds = {}
            after_probas = shadow_engine.predict_proba(df_ns)

            for idx, (_, row) in enumerate(df_ns.iterrows()):
                match_id = row["match_id"]
                after_preds[match_id] = {
                    "home": float(after_probas[idx][0]),
                    "draw": float(after_probas[idx][1]),
                    "away": float(after_probas[idx][2]),
                }

            # 7. Compute AFTER stats
            after_stats = _compute_prediction_stats(after_preds, "after")

            # 8. Compute top deltas (largest draw probability changes)
            top_deltas = []
            for match_id in after_preds:
                if match_id in before_preds:
                    delta_draw = after_preds[match_id]["draw"] - before_preds[match_id]["draw"]
                    top_deltas.append({
                        "match_id": match_id,
                        "delta_draw": round(delta_draw, 4),
                        "before": {k: round(v, 4) for k, v in before_preds[match_id].items()},
                        "after": {k: round(v, 4) for k, v in after_preds[match_id].items()},
                    })

            top_deltas.sort(key=lambda x: abs(x["delta_draw"]), reverse=True)
            top_deltas = top_deltas[:20]  # Keep top 20

            # 9. Build response
            response = {
                "status": "dry_run" if body.dry_run else "executed",
                "run_id": str(run_id),
                "window_hours": body.window_hours,
                "architecture_before": settings.MODEL_ARCHITECTURE,
                "architecture_after": body.architecture,
                "model_version_before": settings.MODEL_VERSION,
                "model_version_after": f"v1.1.0-{body.architecture}",
                "matches_total": len(match_ids),
                "matches_with_features": len(df_ns),
                "matches_with_odds": matches_with_odds,
                "matches_with_before_pred": len(before_preds),
                "stats_before": before_stats,
                "stats_after": after_stats,
                "top_deltas": top_deltas[:10],  # Show top 10 in response
            }

            # 10. If not dry_run, save to database
            if not body.dry_run:
                # Save new predictions with run_id
                saved = 0
                errors = 0
                model_version_after = f"v1.1.0-{body.architecture}"

                for match_id, probs in after_preds.items():
                    try:
                        # Insert new prediction (with different model_version, so no conflict)
                        await session.execute(text("""
                            INSERT INTO predictions (match_id, model_version, home_prob, draw_prob, away_prob, run_id, created_at)
                            VALUES (:match_id, :model_version, :home_prob, :draw_prob, :away_prob, :run_id, NOW())
                            ON CONFLICT (match_id, model_version)
                            DO UPDATE SET
                                home_prob = EXCLUDED.home_prob,
                                draw_prob = EXCLUDED.draw_prob,
                                away_prob = EXCLUDED.away_prob,
                                run_id = EXCLUDED.run_id,
                                created_at = NOW()
                        """), {
                            "match_id": match_id,
                            "model_version": model_version_after,
                            "home_prob": probs["home"],
                            "draw_prob": probs["draw"],
                            "away_prob": probs["away"],
                            "run_id": run_id,
                        })
                        saved += 1
                    except Exception as e:
                        errors += 1
                        logger.warning(f"Rerun: failed to save match {match_id}: {e}")

                # Create audit record
                rerun_record = PredictionRerun(
                    run_id=run_id,
                    run_type="manual_rerun",
                    window_hours=body.window_hours,
                    architecture_before=settings.MODEL_ARCHITECTURE,
                    architecture_after=body.architecture,
                    model_version_before=settings.MODEL_VERSION,
                    model_version_after=model_version_after,
                    matches_total=len(match_ids),
                    matches_with_odds=matches_with_odds,
                    stats_before=before_stats,
                    stats_after=after_stats,
                    top_deltas=top_deltas,
                    is_active=True,
                    triggered_by="dashboard_ops",
                    notes=body.notes,
                )
                session.add(rerun_record)
                await session.commit()

                response["saved"] = saved
                response["errors"] = errors
                response["audit_record_created"] = True

                logger.info(
                    f"[RERUN] Predictions rerun complete: run_id={run_id}, "
                    f"saved={saved}, errors={errors}, matches={len(match_ids)}"
                )

            return response

        except Exception as e:
            logger.error(f"[RERUN] Predictions rerun failed: {e}")
            raise HTTPException(status_code=500, detail="Rerun failed. Check server logs for details.")


def _compute_prediction_stats(preds: dict, label: str) -> dict:
    """Compute stats for a set of predictions."""
    if not preds:
        return {"n": 0, "label": label}

    home_probs = [p["home"] for p in preds.values()]
    draw_probs = [p["draw"] for p in preds.values()]
    away_probs = [p["away"] for p in preds.values()]

    # Max prob for each prediction (confidence)
    max_probs = [max(p["home"], p["draw"], p["away"]) for p in preds.values()]

    # Count picks
    picks = {"home": 0, "draw": 0, "away": 0}
    for p in preds.values():
        if p["home"] >= p["draw"] and p["home"] >= p["away"]:
            picks["home"] += 1
        elif p["draw"] >= p["home"] and p["draw"] >= p["away"]:
            picks["draw"] += 1
        else:
            picks["away"] += 1

    n = len(preds)
    return {
        "label": label,
        "n": n,
        "avg_p_home": round(sum(home_probs) / n, 4),
        "avg_p_draw": round(sum(draw_probs) / n, 4),
        "avg_p_away": round(sum(away_probs) / n, 4),
        "avg_p_max": round(sum(max_probs) / n, 4),
        "draw_share_pct": round(100.0 * picks["draw"] / n, 2),
        "home_picks": picks["home"],
        "draw_picks": picks["draw"],
        "away_picks": picks["away"],
    }


@app.post("/dashboard/ops/predictions_rerun_rollback")
async def predictions_rerun_rollback(request: Request, run_id: str):
    """
    Rollback a prediction rerun by setting is_active=False.

    This doesn't delete data - it just marks the rerun as inactive,
    so the serving logic will prefer baseline predictions.
    """
    import uuid
    from sqlalchemy import text

    if not _verify_dashboard_token(request):
        raise HTTPException(status_code=401, detail="Dashboard access requires valid token.")

    try:
        parsed_run_id = uuid.UUID(run_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid run_id format")

    async with AsyncSessionLocal() as session:
        result = await session.execute(text("""
            UPDATE prediction_reruns
            SET is_active = FALSE
            WHERE run_id = :run_id
            RETURNING id, run_type, matches_total
        """), {"run_id": parsed_run_id})

        row = result.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Rerun not found")

        await session.commit()

        logger.info(f"[RERUN] Rollback executed: run_id={run_id}")

        return {
            "status": "rolled_back",
            "run_id": run_id,
            "rerun_id": row[0],
            "run_type": row[1],
            "matches_affected": row[2],
            "message": "Rerun deactivated. Baseline predictions will be served.",
        }


@app.get("/dashboard/ops/predictions_reruns.json")
async def list_prediction_reruns(request: Request):
    """List all prediction reruns with their status."""
    from sqlalchemy import text

    if not _verify_dashboard_token(request):
        raise HTTPException(status_code=401, detail="Dashboard access requires valid token.")

    async with AsyncSessionLocal() as session:
        result = await session.execute(text("""
            SELECT
                run_id, run_type, window_hours,
                architecture_before, architecture_after,
                model_version_before, model_version_after,
                matches_total, matches_with_odds,
                stats_before, stats_after,
                is_active, triggered_by, notes,
                created_at, evaluated_at, evaluated_matches
            FROM prediction_reruns
            ORDER BY created_at DESC
            LIMIT 20
        """))

        reruns = []
        for row in result.fetchall():
            reruns.append({
                "run_id": str(row[0]),
                "run_type": row[1],
                "window_hours": row[2],
                "architecture_before": row[3],
                "architecture_after": row[4],
                "model_version_before": row[5],
                "model_version_after": row[6],
                "matches_total": row[7],
                "matches_with_odds": row[8],
                "draw_share_before": row[9].get("draw_share_pct") if row[9] else None,
                "draw_share_after": row[10].get("draw_share_pct") if row[10] else None,
                "is_active": row[11],
                "triggered_by": row[12],
                "notes": row[13],
                "created_at": row[14].isoformat() if row[14] else None,
                "evaluated_at": row[15].isoformat() if row[15] else None,
                "evaluated_matches": row[16],
            })

        return {"reruns": reruns, "count": len(reruns)}


# =============================================================================
# ALPHA PROGRESS SNAPSHOTS (track Re-test/Alpha evolution over time)
# =============================================================================


@app.post("/dashboard/ops/progress_snapshot")
async def capture_progress_snapshot(request: Request, milestone: str | None = None):
    """
    Capture current Alpha Progress state to DB for auditing.

    Creates a snapshot with: generated_at, league_mode, tracked_leagues_count,
    progress metrics, and budget subset.
    Protected by dashboard token.

    Optional query param:
    - milestone: Label for this capture (e.g., "baseline_0", "pit_75", "pit_100", "bets_100", "ready_true")
    """
    import os
    from app.models import AlphaProgressSnapshot

    if not _verify_dashboard_token(request):
        raise HTTPException(status_code=401, detail="Dashboard access requires valid token.")

    # Get current ops data
    data = await _get_cached_ops_data()

    # Extract relevant fields for the snapshot
    payload = {
        "generated_at": data.get("generated_at"),
        "league_mode": data.get("league_mode"),
        "tracked_leagues_count": data.get("tracked_leagues_count"),
        "progress": data.get("progress"),
        "budget": {
            "status": data.get("budget", {}).get("status"),
            "plan": data.get("budget", {}).get("plan"),
            "requests_today": data.get("budget", {}).get("requests_today"),
            "requests_limit": data.get("budget", {}).get("requests_limit"),
        },
        "pit": {
            "live_60m": data.get("pit", {}).get("live_60m"),
            "live_24h": data.get("pit", {}).get("live_24h"),
        },
    }

    # Add milestone label if provided
    if milestone:
        payload["milestone"] = milestone

    # Get git commit SHA from env if available
    app_commit = os.environ.get("RAILWAY_GIT_COMMIT_SHA") or os.environ.get("GIT_COMMIT_SHA")

    # Save to DB
    async with AsyncSessionLocal() as session:
        snapshot = AlphaProgressSnapshot(
            payload=payload,
            source="dashboard_manual" if milestone else "dashboard_manual",
            app_commit=app_commit[:40] if app_commit else None,
        )
        session.add(snapshot)
        await session.commit()
        await session.refresh(snapshot)

        return {
            "status": "captured",
            "id": snapshot.id,
            "milestone": milestone,
            "captured_at": snapshot.captured_at.isoformat(),
            "source": snapshot.source,
            "app_commit": snapshot.app_commit,
        }


@app.get("/dashboard/ops/progress_snapshots.json")
async def get_progress_snapshots(request: Request, limit: int = 50):
    """
    Get historical Alpha Progress snapshots for auditing.

    Returns list of snapshots ordered by captured_at DESC (most recent first).
    Protected by dashboard token.
    """
    import json

    if not _verify_dashboard_token(request):
        raise HTTPException(status_code=401, detail="Dashboard access requires valid token.")

    async with AsyncSessionLocal() as session:
        result = await session.execute(text("""
            SELECT id, captured_at, payload, source, app_commit
            FROM alpha_progress_snapshots
            ORDER BY captured_at DESC
            LIMIT :limit
        """), {"limit": limit})

        rows = result.fetchall()
        snapshots = []
        for row in rows:
            payload = row[2]
            if isinstance(payload, str):
                payload = json.loads(payload)

            snapshots.append({
                "id": row[0],
                "captured_at": row[1].isoformat() if row[1] else None,
                "payload": payload,
                "source": row[3],
                "app_commit": row[4],
            })

        return {
            "count": len(snapshots),
            "limit": limit,
            "snapshots": snapshots,
        }


# =============================================================================
# OPS HISTORY ENDPOINTS (KPI rollups from ops_daily_rollups table)
# =============================================================================


async def _get_ops_history(days: int = 30) -> list[dict]:
    """Fetch recent daily rollups from ops_daily_rollups table."""
    import json

    async with AsyncSessionLocal() as session:
        result = await session.execute(text("""
            SELECT day, payload, updated_at
            FROM ops_daily_rollups
            ORDER BY day DESC
            LIMIT :days
        """), {"days": days})

        rows = result.fetchall()
        history = []
        for row in rows:
            day = row[0]
            payload = row[1]
            updated_at = row[2]

            # Parse payload if it's a string
            if isinstance(payload, str):
                payload = json.loads(payload)

            history.append({
                "day": str(day),
                "payload": payload,
                "updated_at": updated_at.isoformat() if updated_at else None,
            })

        return history


@app.get("/dashboard/ops/history.json")
async def ops_history_json(request: Request, days: int = 30):
    """JSON endpoint for historical daily KPIs."""
    if not _verify_dashboard_token(request):
        raise HTTPException(status_code=401, detail="Dashboard access requires valid token.")

    history = await _get_ops_history(days=days)
    return {
        "days_requested": days,
        "days_available": len(history),
        "history": history,
    }




# =============================================================================
# OPS CONSOLE LOGIN / LOGOUT
# =============================================================================


def _render_login_page(error: str = "") -> str:
    """Render the OPS console login page HTML."""
    error_html = f'<div class="error">{error}</div>' if error else ""
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>OPS Console Login</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
        }}
        .login-container {{
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 16px;
            padding: 40px;
            width: 100%;
            max-width: 400px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        }}
        h1 {{
            color: #fff;
            text-align: center;
            margin-bottom: 8px;
            font-size: 24px;
        }}
        .subtitle {{
            color: rgba(255, 255, 255, 0.6);
            text-align: center;
            margin-bottom: 32px;
            font-size: 14px;
        }}
        .error {{
            background: rgba(239, 68, 68, 0.2);
            border: 1px solid rgba(239, 68, 68, 0.5);
            color: #fca5a5;
            padding: 12px;
            border-radius: 8px;
            margin-bottom: 20px;
            font-size: 14px;
            text-align: center;
        }}
        .form-group {{
            margin-bottom: 20px;
        }}
        label {{
            display: block;
            color: rgba(255, 255, 255, 0.8);
            margin-bottom: 8px;
            font-size: 14px;
        }}
        input[type="password"] {{
            width: 100%;
            padding: 12px 16px;
            border: 1px solid rgba(255, 255, 255, 0.2);
            border-radius: 8px;
            background: rgba(255, 255, 255, 0.1);
            color: #fff;
            font-size: 16px;
            transition: border-color 0.2s;
        }}
        input[type="password"]:focus {{
            outline: none;
            border-color: #3b82f6;
        }}
        button {{
            width: 100%;
            padding: 14px;
            background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
            border: none;
            border-radius: 8px;
            color: #fff;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            transition: transform 0.2s, box-shadow 0.2s;
        }}
        button:hover {{
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(59, 130, 246, 0.4);
        }}
        .footer {{
            text-align: center;
            margin-top: 24px;
            color: rgba(255, 255, 255, 0.4);
            font-size: 12px;
        }}
    </style>
</head>
<body>
    <div class="login-container">
        <h1>FutbolStats OPS</h1>
        <p class="subtitle">Admin Console</p>
        {error_html}
        <form method="POST" action="/ops/login">
            <div class="form-group">
                <label for="password">Password</label>
                <input type="password" id="password" name="password" required autofocus>
            </div>
            <button type="submit">Sign In</button>
        </form>
        <p class="footer">Secure access only</p>
    </div>
</body>
</html>"""


@app.get("/ops/login")
async def ops_login_page(request: Request, error: str = ""):
    """Display the OPS console login form."""
    from fastapi.responses import HTMLResponse, RedirectResponse

    # If already logged in, redirect to dashboard JSON
    if _has_valid_session(request):
        return RedirectResponse(url="/dashboard/ops.json", status_code=302)

    # Check if login is enabled
    if not settings.OPS_ADMIN_PASSWORD:
        raise HTTPException(
            status_code=503,
            detail="OPS login disabled. Set OPS_ADMIN_PASSWORD env var."
        )

    return HTMLResponse(content=_render_login_page(error))


@app.post("/ops/login")
@limiter.limit("10/minute")
async def ops_login_submit(request: Request):
    """Process OPS console login."""
    from fastapi.responses import HTMLResponse, RedirectResponse

    # Check if login is enabled
    if not settings.OPS_ADMIN_PASSWORD:
        raise HTTPException(status_code=503, detail="OPS login disabled")

    # Parse form data
    form = await request.form()
    password = form.get("password", "")

    # Validate password
    if password != settings.OPS_ADMIN_PASSWORD:
        logger.warning(f"[OPS_LOGIN] Failed login attempt from {request.client.host}")
        return HTMLResponse(
            content=_render_login_page("Invalid password"),
            status_code=401
        )

    # Create session
    request.session["ops_authenticated"] = True
    request.session["issued_at"] = datetime.utcnow().isoformat()
    logger.info(f"[OPS_LOGIN] Successful login from {request.client.host}")

    # Redirect to dashboard JSON
    return RedirectResponse(url="/dashboard/ops.json", status_code=302)


@app.get("/ops/logout")
async def ops_logout(request: Request):
    """Logout from OPS console."""
    from fastapi.responses import RedirectResponse

    # Clear session
    request.session.clear()
    logger.info(f"[OPS_LOGIN] Logout from {request.client.host}")

    return RedirectResponse(url="/ops/login", status_code=302)


@app.get("/dashboard")
async def dashboard_home(request: Request):
    """Unified dashboard entrypoint (redirects to Ops)."""
    from fastapi.responses import RedirectResponse

    if not _verify_dashboard_token(request):
        # Redirect to login instead of 401 for better UX
        return RedirectResponse(url="/ops/login", status_code=302)

    # Preserve token query param ONLY in development (not in prod - security risk)
    target = "/dashboard/ops.json"
    if not os.getenv("RAILWAY_PROJECT_ID"):
        token = request.query_params.get("token")
        if token:
            target = f"{target}?token={token}"
    return RedirectResponse(url=target, status_code=307)


# =============================================================================
# DEBUG ENDPOINT: Daily Counts (temporary for ops monitoring)
# =============================================================================


@app.get("/dashboard/ops/daily_counts.json")
async def ops_daily_counts(
    request: Request,
    date: str = None,
    session: AsyncSession = Depends(get_async_session),
):
    """
    Get daily counts for predictions, audits, and LLM narratives.

    Args:
        date: Date in YYYY-MM-DD format. Defaults to today (UTC).
    """
    if not _verify_dashboard_token(request):
        raise HTTPException(status_code=401, detail="Dashboard access requires valid token.")

    import re

    target_date = date or datetime.utcnow().strftime("%Y-%m-%d")
    # Validate date format (YYYY-MM-DD) to prevent SQL injection
    if not re.match(r"^\d{4}-\d{2}-\d{2}$", target_date):
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD.")

    # A) Predictions
    predictions_created_today = await session.execute(
        text(f"SELECT COUNT(*) FROM predictions WHERE created_at::date = '{target_date}'")
    )
    pred_created = predictions_created_today.scalar() or 0

    predictions_for_matches_today = await session.execute(
        text(f"""
            SELECT COUNT(*) FROM predictions p
            JOIN matches m ON p.match_id = m.id
            WHERE m.date::date = '{target_date}'
        """)
    )
    pred_for_matches = predictions_for_matches_today.scalar() or 0

    # B) Audits
    ft_matches_today = await session.execute(
        text(f"""
            SELECT COUNT(*) FROM matches
            WHERE status IN ('FT', 'AET', 'PEN')
            AND date::date = '{target_date}'
        """)
    )
    ft_count = ft_matches_today.scalar() or 0

    with_prediction_outcome = await session.execute(
        text(f"""
            SELECT COUNT(*) FROM prediction_outcomes po
            JOIN matches m ON po.match_id = m.id
            WHERE m.status IN ('FT', 'AET', 'PEN')
            AND m.date::date = '{target_date}'
        """)
    )
    po_count = with_prediction_outcome.scalar() or 0

    with_post_match_audit = await session.execute(
        text(f"""
            SELECT COUNT(*) FROM post_match_audits pma
            JOIN prediction_outcomes po ON pma.outcome_id = po.id
            JOIN matches m ON po.match_id = m.id
            WHERE m.status IN ('FT', 'AET', 'PEN')
            AND m.date::date = '{target_date}'
        """)
    )
    pma_count = with_post_match_audit.scalar() or 0

    # C) LLM Narratives
    llm_ok_today = await session.execute(
        text(f"""
            SELECT COUNT(*) FROM post_match_audits pma
            JOIN prediction_outcomes po ON pma.outcome_id = po.id
            JOIN matches m ON po.match_id = m.id
            WHERE m.date::date = '{target_date}'
            AND pma.llm_narrative_status = 'ok'
        """)
    )
    llm_ok = llm_ok_today.scalar() or 0

    llm_breakdown = await session.execute(
        text(f"""
            SELECT pma.llm_narrative_status, COUNT(*) as count
            FROM post_match_audits pma
            JOIN prediction_outcomes po ON pma.outcome_id = po.id
            JOIN matches m ON po.match_id = m.id
            WHERE m.date::date = '{target_date}'
            GROUP BY pma.llm_narrative_status
            ORDER BY count DESC
        """)
    )
    breakdown = {row[0] or "null": row[1] for row in llm_breakdown.all()}

    # D) LLM Error Details (for debugging)
    llm_error_details = await session.execute(
        text(f"""
            SELECT
                po.match_id,
                pma.id as audit_id,
                pma.llm_narrative_status,
                pma.llm_narrative_delay_ms,
                pma.llm_narrative_exec_ms,
                pma.llm_narrative_tokens_in,
                pma.llm_narrative_tokens_out,
                pma.llm_narrative_worker_id,
                pma.llm_narrative_model,
                pma.llm_narrative_generated_at,
                CASE WHEN pma.llm_narrative_json IS NULL THEN true ELSE false END as json_is_null,
                SUBSTRING(pma.llm_narrative_json::text, 1, 500) as json_preview,
                pma.llm_narrative_error_code,
                pma.llm_narrative_error_detail,
                pma.llm_narrative_request_id,
                pma.llm_narrative_attempts
            FROM post_match_audits pma
            JOIN prediction_outcomes po ON pma.outcome_id = po.id
            JOIN matches m ON po.match_id = m.id
            WHERE m.date::date = '{target_date}'
            AND (pma.llm_narrative_status IS NULL OR pma.llm_narrative_status != 'ok')
            ORDER BY po.match_id
        """)
    )
    error_rows = []
    for row in llm_error_details.all():
        error_rows.append({
            "match_id": row[0],
            "audit_id": row[1],
            "status": row[2],
            "delay_ms": row[3],
            "exec_ms": row[4],
            "tokens_in": row[5],
            "tokens_out": row[6],
            "worker_id": row[7],
            "model": row[8],
            "generated_at": str(row[9]) if row[9] else None,
            "json_is_null": row[10],
            "json_preview": row[11],
            "error_code": row[12],
            "error_detail": row[13],
            "request_id": row[14],
            "attempts": row[15],
        })

    return {
        "date": target_date,
        "predictions": {
            "created_today": pred_created,
            "for_matches_today": pred_for_matches,
        },
        "audits": {
            "ft_matches_today": ft_count,
            "with_prediction_outcome": po_count,
            "with_post_match_audit": pma_count,
        },
        "llm_narratives": {
            "ok_today": llm_ok,
            "breakdown": breakdown,
            "error_details": error_rows,
        },
    }


# =============================================================================
# DAILY COMPARISON: Model A vs Shadow vs Sensor B vs Market
# =============================================================================


@app.get("/dashboard/ops/daily_comparison.json")
async def ops_daily_comparison(
    request: Request,
    date: str = None,
    league_id: int = None,
    session: AsyncSession = Depends(get_async_session),
):
    """
    Get daily comparison of finished matches: Real vs Model A vs Shadow vs Sensor B vs Market.

    Args:
        date: Date in YYYY-MM-DD format (America/Los_Angeles timezone). Defaults to today.
        league_id: Optional filter by league ID.

    Returns:
        List of matches with predictions from all sources for comparison.
    """
    import pytz
    import re

    if not _verify_dashboard_token(request):
        raise HTTPException(status_code=401, detail="Dashboard access requires valid token.")

    # Default to today in LA timezone
    la_tz = pytz.timezone("America/Los_Angeles")
    if date:
        # Validate date format
        if not re.match(r"^\d{4}-\d{2}-\d{2}$", date):
            raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD.")
        target_date = datetime.strptime(date, "%Y-%m-%d")
    else:
        # Today in LA timezone
        target_date = datetime.now(la_tz).replace(hour=0, minute=0, second=0, microsecond=0)
        target_date = target_date.replace(tzinfo=None)  # Make naive for localize

    # Convert date_la to UTC range (CRITICAL for index usage per Auditor)
    start_la = la_tz.localize(target_date.replace(hour=0, minute=0, second=0))
    end_la = la_tz.localize(target_date.replace(hour=23, minute=59, second=59))
    start_utc = start_la.astimezone(pytz.UTC).replace(tzinfo=None)  # Naive for DB
    end_utc = (end_la.astimezone(pytz.UTC) + timedelta(seconds=1)).replace(tzinfo=None)

    # Build query with UTC range filter
    query = """
        SELECT
            match_id,
            kickoff_utc,
            match_day_la,
            league_id,
            status,
            home_team,
            away_team,
            home_goals,
            away_goals,
            actual_outcome,
            a_home_prob,
            a_draw_prob,
            a_away_prob,
            a_pick,
            a_version,
            a_is_frozen,
            shadow_home_prob,
            shadow_draw_prob,
            shadow_away_prob,
            shadow_pick,
            shadow_version,
            sensor_home_prob,
            sensor_draw_prob,
            sensor_away_prob,
            sensor_pick,
            sensor_version,
            sensor_state,
            market_bookmaker,
            market_odds_home,
            market_odds_draw,
            market_odds_away,
            market_implied_home,
            market_implied_draw,
            market_implied_away,
            market_pick
        FROM v_daily_match_comparison
        WHERE kickoff_utc >= :start_utc AND kickoff_utc < :end_utc
    """
    params = {"start_utc": start_utc, "end_utc": end_utc}

    if league_id:
        query += " AND league_id = :league_id"
        params["league_id"] = league_id

    query += " ORDER BY kickoff_utc"

    result = await session.execute(text(query), params)
    matches = result.mappings().all()

    # Calculate summary stats
    total = len(matches)
    a_correct = sum(1 for m in matches if m["a_pick"] == m["actual_outcome"])
    shadow_correct = sum(1 for m in matches if m["shadow_pick"] == m["actual_outcome"])
    sensor_correct = sum(1 for m in matches if m["sensor_pick"] == m["actual_outcome"])
    market_correct = sum(1 for m in matches if m["market_pick"] == m["actual_outcome"])

    return {
        "date_la": target_date.strftime("%Y-%m-%d"),
        "start_utc": start_utc.isoformat(),
        "end_utc": end_utc.isoformat(),
        "total_matches": total,
        "summary": {
            "model_a": {
                "correct": a_correct,
                "accuracy": round(a_correct / total, 3) if total > 0 else 0,
            },
            "shadow": {
                "correct": shadow_correct,
                "accuracy": round(shadow_correct / total, 3) if total > 0 else 0,
            },
            "sensor_b": {
                "correct": sensor_correct,
                "accuracy": round(sensor_correct / total, 3) if total > 0 else 0,
            },
            "market": {
                "correct": market_correct,
                "accuracy": round(market_correct / total, 3) if total > 0 else 0,
            },
        },
        "matches": [dict(m) for m in matches],
    }






@app.get("/dashboard/ops/team_overrides.json")
async def ops_team_overrides(
    request: Request,
    external_team_id: int = None,
    session: AsyncSession = Depends(get_async_session),
):
    """
    List all team identity overrides configured in the system.

    Args:
        external_team_id: Optional filter by API-Football team ID (e.g., 1134 for La Equidad).

    Used to verify rebranding configurations like La Equidad → Internacional de Bogotá.
    """
    if not _verify_dashboard_token(request):
        raise HTTPException(status_code=401, detail="Dashboard access requires valid token.")

    query = """
        SELECT
            id, provider, external_team_id, display_name, display_logo_url,
            effective_from, effective_to, reason, updated_by, created_at, updated_at
        FROM team_overrides
    """
    params = {}
    if external_team_id:
        query += " WHERE external_team_id = :external_team_id"
        params["external_team_id"] = external_team_id
    query += " ORDER BY external_team_id, effective_from DESC"

    result = await session.execute(text(query), params)
    rows = result.fetchall()

    overrides = []
    for row in rows:
        overrides.append({
            "id": row[0],
            "provider": row[1],
            "external_team_id": row[2],
            "display_name": row[3],
            "display_logo_url": row[4],
            "effective_from": str(row[5]) if row[5] else None,
            "effective_to": str(row[6]) if row[6] else None,
            "reason": row[7],
            "updated_by": row[8],
            "created_at": str(row[9]) if row[9] else None,
            "updated_at": str(row[10]) if row[10] else None,
        })

    return {
        "count": len(overrides),
        "overrides": overrides,
        "note": "These overrides replace team names/logos for matches on or after effective_from date.",
    }


@app.get("/dashboard/ops/job_runs.json")
async def ops_job_runs(
    request: Request,
    job_name: str = None,
    limit: int = 20,
    session: AsyncSession = Depends(get_async_session),
):
    """
    List recent job runs from the job_runs table (P1-B fallback).

    Args:
        job_name: Optional filter by job name (stats_backfill, odds_sync, fastpath).
        limit: Max rows to return (default 20).

    Used to verify job tracking is working and debug jobs_health fallback.
    """
    if not _verify_dashboard_token(request):
        raise HTTPException(status_code=401, detail="Dashboard access requires valid token.")

    # Check if table exists
    try:
        check_result = await session.execute(text("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables
                WHERE table_name = 'job_runs'
            )
        """))
        table_exists = check_result.scalar()
        if not table_exists:
            return {
                "count": 0,
                "runs": [],
                "note": "job_runs table does not exist. Run migration 028_job_runs.py first.",
            }
    except Exception as e:
        return {"error": str(e), "note": "Failed to check table existence"}

    query = """
        SELECT id, job_name, status, started_at, finished_at, duration_ms, error_message, metrics
        FROM job_runs
    """
    params = {"limit": min(limit, 100)}  # Cap at 100

    if job_name:
        query += " WHERE job_name = :job_name"
        params["job_name"] = job_name

    query += " ORDER BY finished_at DESC LIMIT :limit"

    result = await session.execute(text(query), params)
    rows = result.fetchall()

    runs = []
    for row in rows:
        runs.append({
            "id": row[0],
            "job_name": row[1],
            "status": row[2],
            "started_at": row[3].isoformat() + "Z" if row[3] else None,
            "finished_at": row[4].isoformat() + "Z" if row[4] else None,
            "duration_ms": row[5],
            "error_message": row[6],
            "metrics": row[7],
        })

    # Get last success per job for summary
    summary_result = await session.execute(text("""
        SELECT DISTINCT ON (job_name)
            job_name,
            finished_at as last_success_at
        FROM job_runs
        WHERE status = 'ok'
        ORDER BY job_name, finished_at DESC
    """))
    summary_rows = summary_result.fetchall()
    summary = {
        row[0]: row[1].isoformat() + "Z" if row[1] else None
        for row in summary_rows
    }

    return {
        "count": len(runs),
        "runs": runs,
        "last_success_by_job": summary,
        "note": "Job runs tracked for ops dashboard fallback when Prometheus is cold.",
    }


@app.post("/dashboard/ops/migrate_llm_error_fields")
async def migrate_llm_error_fields(
    request: Request,
    session: AsyncSession = Depends(get_async_session),
):
    """
    One-time migration to add LLM error observability fields.
    Safe to run multiple times (uses IF NOT EXISTS).
    """
    if not _verify_dashboard_token(request):
        raise HTTPException(status_code=401, detail="Dashboard access requires valid token.")

    migrations = [
        "ALTER TABLE post_match_audits ADD COLUMN IF NOT EXISTS llm_narrative_error_code VARCHAR(50)",
        "ALTER TABLE post_match_audits ADD COLUMN IF NOT EXISTS llm_narrative_error_detail VARCHAR(500)",
        "ALTER TABLE post_match_audits ADD COLUMN IF NOT EXISTS llm_narrative_request_id VARCHAR(100)",
        "ALTER TABLE post_match_audits ADD COLUMN IF NOT EXISTS llm_narrative_attempts INTEGER",
    ]

    results = []
    for sql in migrations:
        try:
            await session.execute(text(sql))
            results.append({"sql": sql[:60] + "...", "status": "ok"})
        except Exception as e:
            results.append({"sql": sql[:60] + "...", "status": "error", "error": str(e)})

    await session.commit()

    # Verify columns exist
    verify = await session.execute(
        text("""
            SELECT column_name FROM information_schema.columns
            WHERE table_name='post_match_audits'
            AND column_name LIKE 'llm_narrative_error%'
            OR column_name IN ('llm_narrative_request_id', 'llm_narrative_attempts')
            ORDER BY column_name
        """)
    )
    columns = [row[0] for row in verify.all()]

    return {
        "status": "ok",
        "migrations": results,
        "verified_columns": columns,
    }


@app.post("/dashboard/ops/migrate_fastpath_fields")
async def migrate_fastpath_fields(
    request: Request,
    session: AsyncSession = Depends(get_async_session),
):
    """
    One-time migration to add fast-path tracking fields to matches.
    Safe to run multiple times (uses IF NOT EXISTS).
    """
    if not _verify_dashboard_token(request):
        raise HTTPException(status_code=401, detail="Dashboard access requires valid token.")

    migrations = [
        "ALTER TABLE matches ADD COLUMN IF NOT EXISTS finished_at TIMESTAMP",
        "ALTER TABLE matches ADD COLUMN IF NOT EXISTS stats_ready_at TIMESTAMP",
        "ALTER TABLE matches ADD COLUMN IF NOT EXISTS stats_last_checked_at TIMESTAMP",
        "ALTER TABLE matches ADD COLUMN IF NOT EXISTS events JSON",
        """CREATE INDEX IF NOT EXISTS idx_matches_fastpath_candidates
           ON matches(finished_at, stats_ready_at)
           WHERE finished_at IS NOT NULL AND stats_ready_at IS NULL""",
        """CREATE INDEX IF NOT EXISTS idx_matches_finished_at
           ON matches(finished_at)
           WHERE finished_at IS NOT NULL""",
    ]

    results = []
    for sql in migrations:
        try:
            await session.execute(text(sql))
            results.append({"sql": sql[:60] + "...", "status": "ok"})
        except Exception as e:
            results.append({"sql": sql[:60] + "...", "status": "error", "error": str(e)})

    await session.commit()

    # Verify columns exist
    verify = await session.execute(
        text("""
            SELECT column_name FROM information_schema.columns
            WHERE table_name='matches'
            AND column_name IN ('finished_at', 'stats_ready_at', 'stats_last_checked_at')
            ORDER BY column_name
        """)
    )
    columns = [row[0] for row in verify.all()]

    return {
        "status": "ok",
        "migrations": results,
        "verified_columns": columns,
    }


# =============================================================================
# OPS ALERTS: Grafana Webhook → Bell + Toast Notifications
# =============================================================================


def _verify_alerts_webhook_secret(request: Request) -> bool:
    """Verify webhook authentication via X-Alerts-Secret header or Authorization header.

    Supports two formats:
    1. X-Alerts-Secret: <token>  (direct header)
    2. Authorization: X-Alerts-Secret <token>  (Grafana webhook format)
    """
    settings = get_settings()
    if not settings.ALERTS_WEBHOOK_SECRET:
        return False  # Webhook disabled if no secret configured

    # Try direct header first
    provided = request.headers.get("X-Alerts-Secret", "")
    if provided == settings.ALERTS_WEBHOOK_SECRET:
        return True

    # Try Authorization header with custom scheme (Grafana format)
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("X-Alerts-Secret "):
        provided = auth_header[len("X-Alerts-Secret "):]
        return provided == settings.ALERTS_WEBHOOK_SECRET

    return False


@app.post("/dashboard/ops/alerts/webhook")
async def ops_alerts_webhook(
    request: Request,
    session: AsyncSession = Depends(get_async_session),
):
    """
    Receive alerts from Grafana Alerting webhook.

    Auth: X-Alerts-Secret header (dedicated secret, not dashboard token).

    Expects Grafana Unified Alerting format. Tolerant parsing.
    Upserts by dedupe_key (fingerprint) for idempotence.
    """
    if not _verify_alerts_webhook_secret(request):
        raise HTTPException(status_code=401, detail="Invalid or missing X-Alerts-Secret")

    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON payload")

    # Grafana sends alerts in different formats depending on version
    # Handle both single alert and array of alerts
    alerts = []
    if isinstance(payload, list):
        alerts = payload
    elif isinstance(payload, dict):
        # Grafana Unified Alerting format: { alerts: [...] }
        if "alerts" in payload:
            alerts = payload.get("alerts", [])
        else:
            # Single alert object
            alerts = [payload]

    if not alerts:
        return {"status": "ok", "processed": 0, "message": "No alerts in payload"}

    processed = 0
    errors = []

    for alert in alerts:
        try:
            # Extract fields with fallbacks
            labels = alert.get("labels", {})
            annotations = alert.get("annotations", {})

            # Dedupe key: prefer fingerprint, fallback to alertname + labels hash
            fingerprint = alert.get("fingerprint", "")
            if not fingerprint:
                # Generate from alertname + sorted labels
                import hashlib
                alertname = labels.get("alertname", "unknown")
                labels_str = str(sorted(labels.items()))
                fingerprint = hashlib.sha256(f"{alertname}:{labels_str}".encode()).hexdigest()[:32]

            # Status: firing or resolved
            status = alert.get("status", "firing").lower()
            if status not in ("firing", "resolved"):
                status = "firing"

            # Severity from labels
            severity = labels.get("severity", "warning").lower()
            if severity not in ("critical", "warning", "info"):
                severity = "warning"

            # Title: prefer annotations.summary, fallback to labels.alertname
            title = (
                annotations.get("summary")
                or labels.get("alertname")
                or alert.get("title")
                or "Unknown Alert"
            )[:500]  # Truncate

            # Message: prefer annotations.description
            message = annotations.get("description") or annotations.get("message") or ""
            if len(message) > 1000:
                message = message[:997] + "..."

            # Timestamps (convert to naive UTC for DB compatibility)
            # Helper: normalize any timestamp to UTC naive (repo convention)
            def _to_utc_naive(value: str | datetime | None) -> datetime | None:
                """
                Convert timestamp to UTC naive datetime.

                - None -> None
                - ISO string with tz -> parse, convert to UTC, strip tzinfo
                - datetime aware -> convert to UTC, strip tzinfo
                - datetime naive -> assume UTC, return as-is
                """
                from datetime import timezone

                if value is None:
                    return None

                if isinstance(value, str):
                    if not value or value == "0001-01-01T00:00:00Z":
                        return None
                    try:
                        # Parse ISO format, handle Z suffix
                        value = datetime.fromisoformat(value.replace("Z", "+00:00"))
                    except ValueError:
                        logger.warning(f"[ALERTS] Invalid timestamp format: {value[:50]}")
                        return None

                if isinstance(value, datetime):
                    if value.tzinfo is not None:
                        # Convert to UTC then strip tzinfo
                        value = value.astimezone(timezone.utc).replace(tzinfo=None)
                    # else: already naive, assume UTC
                    return value

                return None

            starts_at = _to_utc_naive(alert.get("startsAt"))
            ends_at = _to_utc_naive(alert.get("endsAt"))

            # Source URL (Grafana panel/alert link)
            source_url = (
                alert.get("generatorURL")
                or alert.get("silenceURL")
                or annotations.get("runbook_url")
            )

            # Guardrail: ensure timestamps are naive UTC before DB insert
            # (asyncpg will reject aware datetimes for TIMESTAMP WITHOUT TIME ZONE)
            if starts_at is not None and starts_at.tzinfo is not None:
                logger.warning(f"[ALERTS] starts_at still has tzinfo after normalization, forcing naive: {fingerprint}")
                starts_at = starts_at.replace(tzinfo=None)
            if ends_at is not None and ends_at.tzinfo is not None:
                logger.warning(f"[ALERTS] ends_at still has tzinfo after normalization, forcing naive: {fingerprint}")
                ends_at = ends_at.replace(tzinfo=None)

            # Upsert into ops_alerts
            now = datetime.utcnow()

            # Check if exists
            existing = await session.execute(
                select(OpsAlert).where(OpsAlert.dedupe_key == fingerprint)
            )
            existing_alert = existing.scalar_one_or_none()

            if existing_alert:
                # Update existing
                existing_alert.status = status
                existing_alert.severity = severity
                existing_alert.title = title
                existing_alert.message = message
                existing_alert.labels = labels
                existing_alert.annotations = annotations
                existing_alert.starts_at = starts_at or existing_alert.starts_at
                existing_alert.ends_at = ends_at
                existing_alert.source_url = source_url or existing_alert.source_url
                existing_alert.last_seen_at = now
                existing_alert.updated_at = now
                # If resolved, mark as read (auto-clear)
                if status == "resolved":
                    existing_alert.is_read = True
            else:
                # Insert new
                new_alert = OpsAlert(
                    dedupe_key=fingerprint,
                    status=status,
                    severity=severity,
                    title=title,
                    message=message,
                    labels=labels,
                    annotations=annotations,
                    starts_at=starts_at,
                    ends_at=ends_at,
                    source="grafana",
                    source_url=source_url,
                    first_seen_at=now,
                    last_seen_at=now,
                    is_read=False,
                    is_ack=False,
                    created_at=now,
                    updated_at=now,
                )
                session.add(new_alert)

            processed += 1

        except Exception as e:
            errors.append(str(e)[:100])
            logger.warning(f"Failed to process alert: {e}")

    await session.commit()

    return {
        "status": "ok",
        "processed": processed,
        "errors": errors if errors else None,
    }


@app.get("/dashboard/ops/alerts.json")
async def ops_alerts_list(
    request: Request,
    limit: int = 50,
    status: str = "all",  # firing, resolved, all
    unread_only: bool = False,
    session: AsyncSession = Depends(get_async_session),
):
    """
    Get ops alerts for bell dropdown.

    Auth: X-Dashboard-Token (same as ops.json).

    Returns unread_count and list of recent alerts.
    """
    if not _verify_dashboard_token(request):
        raise HTTPException(status_code=401, detail="Dashboard access requires valid token.")

    # Clamp limit
    limit = min(max(1, limit), 100)

    # Build query
    query = select(OpsAlert).order_by(OpsAlert.last_seen_at.desc())

    # Status filter
    if status == "firing":
        query = query.where(OpsAlert.status == "firing")
    elif status == "resolved":
        query = query.where(OpsAlert.status == "resolved")
    # else: all

    # Unread filter
    if unread_only:
        query = query.where(OpsAlert.is_read == False)

    query = query.limit(limit)

    result = await session.execute(query)
    alerts = result.scalars().all()

    # Get unread count (always firing + unread)
    unread_result = await session.execute(
        select(func.count(OpsAlert.id)).where(
            OpsAlert.is_read == False,
            OpsAlert.status == "firing"
        )
    )
    unread_count = unread_result.scalar() or 0

    # Format response
    items = []
    for a in alerts:
        items.append({
            "id": a.id,
            "dedupe_key": a.dedupe_key,
            "status": a.status,
            "severity": a.severity,
            "title": a.title,
            "message": a.message[:200] if a.message else None,  # Truncate for list
            "starts_at": a.starts_at.isoformat() if a.starts_at else None,
            "ends_at": a.ends_at.isoformat() if a.ends_at else None,
            "last_seen_at": a.last_seen_at.isoformat() if a.last_seen_at else None,
            "source_url": a.source_url,
            "is_read": a.is_read,
            "is_ack": a.is_ack,
        })

    return {
        "unread_count": unread_count,
        "items": items,
    }


@app.post("/dashboard/ops/alerts/ack")
async def ops_alerts_ack(
    request: Request,
    session: AsyncSession = Depends(get_async_session),
):
    """
    Mark alerts as read/acknowledged.

    Auth: X-Dashboard-Token.

    Body: { "ids": [1,2,3] } or { "ack_all": true }
    """
    if not _verify_dashboard_token(request):
        raise HTTPException(status_code=401, detail="Dashboard access requires valid token.")

    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    now = datetime.utcnow()
    updated = 0

    if body.get("ack_all"):
        # Mark all unread firing alerts as read
        result = await session.execute(
            text("""
                UPDATE ops_alerts
                SET is_read = true, is_ack = true, updated_at = :now
                WHERE is_read = false AND status = 'firing'
            """),
            {"now": now}
        )
        updated = result.rowcount
    elif body.get("ids"):
        ids = body.get("ids", [])
        if not isinstance(ids, list):
            raise HTTPException(status_code=400, detail="ids must be an array")
        # Mark specific alerts as read
        result = await session.execute(
            text("""
                UPDATE ops_alerts
                SET is_read = true, is_ack = true, updated_at = :now
                WHERE id = ANY(:ids)
            """),
            {"now": now, "ids": ids}
        )
        updated = result.rowcount
    else:
        raise HTTPException(status_code=400, detail="Provide 'ids' array or 'ack_all': true")

    await session.commit()

    return {
        "status": "ok",
        "updated": updated,
    }


@app.post("/ops/migrate-weather-precip-prob", include_in_schema=False)
async def migrate_weather_precip_prob(
    session: AsyncSession = Depends(get_async_session),
    _: None = Depends(_verify_dashboard_token),
):
    """
    One-time migration to add precipitation probability field to match_weather.
    Also triggers a backfill for upcoming matches.
    """
    from sqlalchemy import text

    migrations = [
        "ALTER TABLE match_weather ADD COLUMN IF NOT EXISTS precip_prob double precision",
    ]

    results = []
    for sql in migrations:
        try:
            await session.execute(text(sql))
            await session.commit()
            results.append({"sql": sql[:60] + "...", "status": "ok"})
        except Exception as e:
            results.append({"sql": sql[:60] + "...", "status": "error", "error": str(e)})

    # Verify column was added
    verify = await session.execute(
        text("""
            SELECT column_name FROM information_schema.columns
            WHERE table_name = 'match_weather'
            ORDER BY ordinal_position
        """)
    )
    columns = [row[0] for row in verify.all()]

    return {
        "status": "ok",
        "migrations": results,
        "verified_columns": columns,
        "note": "Backfill will happen automatically on next weather_sync job run",
    }


@app.post("/ops/trigger-weather-sync", include_in_schema=False)
async def trigger_weather_sync(
    hours: int = 48,
    limit: int = 100,
    _: None = Depends(_verify_dashboard_token),
):
    """
    Manually trigger weather forecast capture for upcoming matches.

    Args:
        hours: Lookahead window (default 48h)
        limit: Max matches to process (default 100)

    Returns:
        Stats from the weather capture job.
    """
    from app.etl.sota_jobs import capture_weather_prekickoff

    async with AsyncSessionLocal() as session:
        stats = await capture_weather_prekickoff(
            session,
            hours=hours,
            limit=limit,
            horizon=24,
        )
        await session.commit()

    return {
        "status": "ok",
        "stats": stats,
    }


# ---------------------------------------------------------------------------
# Debug Log Endpoint (for iOS performance instrumentation)
# ---------------------------------------------------------------------------

from pathlib import Path
from datetime import datetime as dt

# Environment flag: DEBUG_LOG_ENABLED=true allows logging without token (dev mode ONLY)
# SECURITY: In production (Railway), this flag is IGNORED - auth is always required
_DEBUG_LOG_ENABLED = os.getenv("DEBUG_LOG_ENABLED", "false").lower() == "true"
_IS_PRODUCTION = os.getenv("RAILWAY_PROJECT_ID") is not None


@app.post("/debug/log")
async def debug_log(request: Request):
    """
    Receives performance logs from iOS instrumentation.

    Security:
    - In production: always require valid X-Dashboard-Token (fail-closed)
    - In development: DEBUG_LOG_ENABLED=true allows without token

    Rate limit: handled by global rate limiter
    """
    # SECURITY: In production, ALWAYS require auth (ignore DEBUG_LOG_ENABLED)
    skip_auth = _DEBUG_LOG_ENABLED and not _IS_PRODUCTION
    if not skip_auth:
        token = request.headers.get("X-Dashboard-Token")
        expected = os.getenv("DASHBOARD_TOKEN", "")
        if not expected:
            # Fail-closed: no token configured = deny all
            return JSONResponse({"error": "service misconfigured"}, status_code=503)
        if not token or token != expected:
            return JSONResponse({"error": "unauthorized"}, status_code=401)

    # Parse and validate payload
    try:
        payload = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid json"}, status_code=400)

    if not isinstance(payload, dict):
        return JSONResponse({"error": "payload must be object"}, status_code=400)

    if "component" not in payload:
        return JSONResponse({"error": "missing component field"}, status_code=400)

    # Ensure logs directory exists
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / "perf_debug.log"

    # Build log entry (no token/headers logged)
    entry = {
        "ts": dt.utcnow().isoformat() + "Z",
        "component": payload.get("component"),
        "endpoint": payload.get("endpoint"),
        "message": payload.get("message"),
        "data": payload.get("data"),
        "hypothesisId": payload.get("hypothesisId"),
    }

    # Append to log file
    import json as _json
    with open(log_file, "a") as f:
        f.write(_json.dumps(entry) + "\n")

    return {"status": "ok"}


@app.get("/debug/log")
async def get_debug_logs(request: Request, tail: int = 50):
    """
    Returns the last N lines of perf_debug.log for analysis.

    Security: requires valid X-Dashboard-Token (always, even in debug mode)
    """
    token = request.headers.get("X-Dashboard-Token")
    expected = os.getenv("DASHBOARD_TOKEN", "")
    if not token or token != expected:
        return JSONResponse({"error": "unauthorized"}, status_code=401)

    log_file = Path("logs") / "perf_debug.log"
    if not log_file.exists():
        return {"logs": [], "count": 0, "message": "No logs yet"}

    import json as _json
    lines = []
    with open(log_file, "r") as f:
        all_lines = f.readlines()
        # Get last N lines
        for line in all_lines[-tail:]:
            try:
                lines.append(_json.loads(line.strip()))
            except Exception:
                lines.append({"raw": line.strip()})

    return {"logs": lines, "count": len(lines), "total": len(all_lines)}


# -----------------------------------------------------------------------------
# Dashboard Incidents Endpoint
# -----------------------------------------------------------------------------
_incidents_cache = {
    "data": None,
    "timestamp": 0,
    "ttl": 30,  # 30 seconds cache (balance freshness vs load)
}


_RESOLVE_GRACE_MINUTES = 30  # Auto-resolve after 30 min not seen (per ABE guardrail)


def _make_incident_id(source: str, key: str) -> int:
    """Generate stable incident ID from source + key (MD5 first 8 hex → int)."""
    import hashlib
    h = hashlib.md5(f"{source}:{key}".encode()).hexdigest()
    return int(h[:8], 16)


async def _detect_active_incidents(session) -> list[dict]:
    """
    Detect currently active incidents from all sources.

    Returns list of dicts with: id, source, source_key, severity, type, title,
    description, runbook_url, details.

    This is the "detection" phase only — does NOT persist anything.
    """
    incidents = []
    now = datetime.utcnow()
    now_iso = now.isoformat() + "Z"

    # =========================================================================
    # OPTIMIZATION: Launch HTTP-only sources as tasks
    # =========================================================================
    import asyncio
    sentry_task = asyncio.create_task(_fetch_sentry_health())

    # =========================================================================
    # SOURCE 1: Sentry Issues
    # =========================================================================
    try:
        sentry_data = await sentry_task
        if sentry_data.get("status") != "degraded":
            top_issues = sentry_data.get("top_issues", [])
            for issue in top_issues[:10]:
                title = issue.get("title", "Unknown Sentry Issue")
                level = issue.get("level", "error")
                count = issue.get("count", 0)
                last_seen = issue.get("last_seen")

                severity = "warning"
                if level in ("error", "fatal"):
                    severity = "critical"
                elif level == "warning":
                    severity = "warning"
                else:
                    severity = "info"

                _PERF_KEYWORDS = ("consecutive db", "n+1", "slow db", "slow http",
                                  "large http", "large render", "file io on main")
                if level == "info" and any(k in title.lower() for k in _PERF_KEYWORDS):
                    severity = "warning"

                source_key = title[:50]
                incidents.append({
                    "id": _make_incident_id("sentry", source_key),
                    "source": "sentry",
                    "source_key": source_key,
                    "severity": severity,
                    "type": "sentry",
                    "title": title[:80],
                    "description": f"Sentry: {count} events. Level: {level}."[:200],
                    "runbook_url": None,
                    "details": {"level": level, "count": count, "last_seen": last_seen},
                })
    except Exception as e:
        logger.warning(f"Could not fetch Sentry incidents: {e}")

    # =========================================================================
    # SOURCE 2: Predictions Health
    # =========================================================================
    try:
        pred_health = await _calculate_predictions_health(session)
        status_val = pred_health.get("status", "ok")
        if status_val in ("warn", "warning", "critical"):
            reason = pred_health.get("status_reason", "Predictions health degraded")
            ns_missing = pred_health.get("ns_matches_next_48h_missing_prediction", 0)
            ns_total = pred_health.get("ns_matches_next_48h", 0)
            coverage = pred_health.get("ns_coverage_pct", 100)
            severity = "warning" if status_val == "warn" else status_val

            incidents.append({
                "id": _make_incident_id("predictions", "health"),
                "source": "predictions",
                "source_key": "health",
                "severity": severity,
                "type": "predictions",
                "title": f"Predictions coverage at {coverage}%"[:80],
                "description": f"{reason}. {ns_missing}/{ns_total} NS matches missing predictions."[:200],
                "runbook_url": "docs/OPS_RUNBOOK.md#predictions-health",
                "details": {"coverage_pct": coverage, "ns_missing": ns_missing, "ns_total": ns_total},
            })
    except Exception as e:
        logger.warning(f"Could not check predictions health: {e}")

    # =========================================================================
    # SOURCE 3: Jobs Health
    # =========================================================================
    try:
        jobs_health = await _calculate_jobs_health_summary(session)

        for job_name in ["stats_backfill", "odds_sync", "fastpath"]:
            job_data = jobs_health.get(job_name, {})
            job_status = job_data.get("status", "ok")
            if job_status in ("warn", "warning", "red", "critical"):
                mins_since = job_data.get("minutes_since_success")
                help_url = job_data.get("help_url")

                severity = {
                    "warn": "warning", "warning": "warning",
                    "red": "critical", "critical": "critical",
                }.get(job_status, "warning")
                time_str = f"{int(mins_since)}m" if mins_since and mins_since < 60 else (
                    f"{int(mins_since/60)}h" if mins_since else "unknown"
                )

                job_labels = {
                    "stats_backfill": "Stats Backfill",
                    "odds_sync": "Odds Sync",
                    "fastpath": "Fast-Path Narratives",
                }
                expected_intervals = {
                    "stats_backfill": 120, "odds_sync": 720, "fastpath": 5,
                }
                job_label = job_labels.get(job_name, job_name)
                expected_min = expected_intervals.get(job_name)
                ft_pending = job_data.get("ft_pending")
                backlog_ready = job_data.get("backlog_ready")
                last_success_at = job_data.get("last_success_at")
                data_source = job_data.get("source", "unknown")

                desc_parts = [f"Job '{job_label}' last succeeded {time_str} ago (status: {job_status})."]
                if expected_min:
                    desc_parts.append(f"Expected interval: {expected_min}min.")
                if ft_pending is not None:
                    desc_parts.append(f"FT pending stats: {ft_pending}.")
                if backlog_ready is not None:
                    desc_parts.append(f"Backlog ready: {backlog_ready}.")
                desc_parts.append(f"Source: {data_source}.")

                details = {
                    "job_key": job_name,
                    "job_label": job_label,
                    "status": job_status,
                    "minutes_since_success": mins_since,
                    "expected_interval_min": expected_min,
                    "last_success_at": last_success_at,
                    "source": data_source,
                    "runbook_url": help_url,
                }
                if ft_pending is not None:
                    details["ft_pending"] = ft_pending
                if backlog_ready is not None:
                    details["backlog_ready"] = backlog_ready

                # Canonical: id = md5("jobs:<job_name>"), source="jobs"
                incidents.append({
                    "id": _make_incident_id("jobs", job_name),
                    "source": "jobs",
                    "source_key": job_name,
                    "severity": severity,
                    "type": "scheduler",
                    "title": f"Job '{job_label}' unhealthy"[:80],
                    "description": " ".join(desc_parts)[:300],
                    "runbook_url": help_url,
                    "details": details,
                })
    except Exception as e:
        logger.warning(f"Could not check jobs health: {e}")

    # =========================================================================
    # SOURCE 4: FastPath Health (LLM narratives)
    # =========================================================================
    try:
        fp_health = await _calculate_fastpath_health(session)
        fp_status = fp_health.get("status", "ok")
        if fp_status in ("warn", "warning", "red", "critical"):
            error_rate = fp_health.get("last_60m", {}).get("error_rate_pct", 0)
            in_queue = fp_health.get("last_60m", {}).get("in_queue", 0)
            reason = fp_health.get("status_reason", "Fastpath degraded")
            severity = {"warn": "warning", "warning": "warning", "red": "critical", "critical": "critical"}.get(fp_status, "warning")

            incidents.append({
                "id": _make_incident_id("fastpath", "health"),
                "source": "fastpath",
                "source_key": "health",
                "severity": severity,
                "type": "llm",
                "title": f"Fastpath error rate {error_rate}%"[:80],
                "description": f"{reason}. Queue: {in_queue}."[:200],
                "runbook_url": "docs/OPS_RUNBOOK.md#fastpath-health",
                "details": {"error_rate_pct": error_rate, "in_queue": in_queue},
            })
    except Exception as e:
        logger.warning(f"Could not check fastpath health: {e}")

    # =========================================================================
    # SOURCE 5: API Budget (not yet implemented)
    # =========================================================================
    # TODO: implement _fetch_api_football_budget() to enable this source

    # =========================================================================
    # SOURCE 6: Team Profile Coverage (data quality)
    # Excludes nationals from denominator (source='excluded_national')
    # =========================================================================
    try:
        profile_result = await session.execute(text("""
            WITH active_clubs AS (
                SELECT DISTINCT t.id
                FROM teams t
                JOIN matches m ON t.id = m.home_team_id OR t.id = m.away_team_id
                WHERE t.team_type = 'club'
                  AND t.country IS NOT NULL
                  AND m.date >= NOW() - INTERVAL '30 days'
            )
            SELECT
                COUNT(*) AS total,
                COUNT(thcp.team_id) AS with_profile
            FROM active_clubs ac
            LEFT JOIN team_home_city_profile thcp
              ON ac.id = thcp.team_id
              AND thcp.source != 'excluded_national'
        """))
        profile_row = profile_result.fetchone()
        if profile_row and profile_row[0] > 0:
            total_clubs = profile_row[0]
            with_profile = profile_row[1]
            coverage_pct = round(with_profile / total_clubs * 100, 1)

            if coverage_pct < 80:
                incidents.append({
                    "id": _make_incident_id("data_quality", "team_profile_coverage"),
                    "source": "data_quality",
                    "source_key": "team_profile_coverage",
                    "severity": "warning",
                    "type": "data_quality",
                    "title": f"Team profile coverage {coverage_pct}% (target 80%)"[:80],
                    "description": (
                        f"{total_clubs - with_profile} of {total_clubs} active clubs "
                        f"missing home city profile. Run cascade sync."
                    )[:200],
                    "runbook_url": None,
                    "details": {
                        "coverage_pct": coverage_pct,
                        "total_clubs": total_clubs,
                        "with_profile": with_profile,
                        "missing": total_clubs - with_profile,
                    },
                })
    except Exception as e:
        logger.warning(f"Could not check team profile coverage: {e}")

    return incidents


async def _upsert_incidents(session, detected: list[dict]) -> None:
    """
    Upsert detected incidents into ops_incidents table (batch, set-based).

    Single INSERT...ON CONFLICT DO UPDATE — no Python loops, no N+1 queries.

    - INSERT new incidents with timeline "created" event (built in SQL).
    - UPDATE existing: refresh title/description/details/severity, set last_seen_at.
    - REOPEN resolved incidents that reappear (status → active, timeline "reopened").
    - Does NOT touch user-set acknowledged status (unless reopening from resolved).
    """
    if not detected:
        return

    import json as _json
    now = datetime.utcnow()
    now_iso = now.isoformat() + "Z"

    # Build parallel arrays for unnest (ABE: text[] for JSONB to avoid asyncpg type issues)
    ids: list[int] = []
    sources: list[str] = []
    source_keys: list[str] = []
    severities: list[str] = []
    types: list[str] = []
    titles: list[str] = []
    descriptions: list[str | None] = []
    details_json: list[str | None] = []
    runbook_urls: list[str | None] = []
    titles_short: list[str] = []  # for "created" timeline message

    for inc in detected:
        ids.append(inc["id"])
        sources.append(inc["source"])
        source_keys.append(inc["source_key"])
        severities.append(inc["severity"])
        types.append(inc["type"])
        titles.append(inc["title"])
        descriptions.append(inc.get("description"))
        details_json.append(
            _json.dumps(inc["details"]) if inc.get("details") else None
        )
        runbook_urls.append(inc.get("runbook_url"))
        titles_short.append(inc["title"][:100])

    await session.execute(
        text("""
            WITH data AS (
                SELECT *
                FROM unnest(
                    :ids    ::BIGINT[],
                    :sources ::TEXT[],
                    :source_keys ::TEXT[],
                    :severities  ::TEXT[],
                    :types       ::TEXT[],
                    :titles      ::TEXT[],
                    :descriptions ::TEXT[],
                    :details_json ::TEXT[],
                    :runbook_urls ::TEXT[],
                    :titles_short ::TEXT[]
                ) AS t(id, source, source_key, severity, type, title,
                       description, details_json, runbook_url, title_short)
            )
            INSERT INTO ops_incidents
                (id, source, source_key, severity, status, type, title,
                 description, details, runbook_url, timeline,
                 created_at, last_seen_at, updated_at)
            SELECT
                d.id, d.source, d.source_key, d.severity, 'active', d.type, d.title,
                d.description,
                CASE WHEN d.details_json IS NOT NULL
                     THEN d.details_json::jsonb ELSE NULL END,
                d.runbook_url,
                jsonb_build_array(jsonb_build_object(
                    'ts',      :now_iso ::TEXT,
                    'message', 'Incident detected: ' || d.title_short,
                    'actor',   'system',
                    'action',  'created'
                )),
                :now ::TIMESTAMPTZ, :now ::TIMESTAMPTZ, :now ::TIMESTAMPTZ
            FROM data d
            ON CONFLICT (source, source_key) DO UPDATE SET
                severity     = EXCLUDED.severity,
                title        = EXCLUDED.title,
                description  = EXCLUDED.description,
                details      = EXCLUDED.details,
                runbook_url  = EXCLUDED.runbook_url,
                last_seen_at = EXCLUDED.last_seen_at,
                updated_at   = EXCLUDED.updated_at,
                status = CASE
                    WHEN ops_incidents.status = 'resolved' THEN 'active'
                    ELSE ops_incidents.status
                END,
                resolved_at = CASE
                    WHEN ops_incidents.status = 'resolved' THEN NULL
                    ELSE ops_incidents.resolved_at
                END,
                acknowledged_at = CASE
                    WHEN ops_incidents.status = 'resolved' THEN NULL
                    ELSE ops_incidents.acknowledged_at
                END,
                timeline = CASE
                    WHEN ops_incidents.status = 'resolved' THEN
                        COALESCE(ops_incidents.timeline, '[]'::jsonb)
                        || jsonb_build_array(jsonb_build_object(
                            'ts',      :now_iso ::TEXT,
                            'message', 'Incident reopened (detected again)',
                            'actor',   'system',
                            'action',  'reopened'
                        ))
                    ELSE ops_incidents.timeline
                END
        """),
        {
            "ids": ids,
            "sources": sources,
            "source_keys": source_keys,
            "severities": severities,
            "types": types,
            "titles": titles,
            "descriptions": descriptions,
            "details_json": details_json,
            "runbook_urls": runbook_urls,
            "titles_short": titles_short,
            "now": now,
            "now_iso": now_iso,
        },
    )

    await session.commit()


async def _auto_resolve_stale_incidents(session) -> int:
    """
    Auto-resolve incidents not seen within grace window (single UPDATE, no loops).

    Only resolves active/acknowledged incidents where last_seen_at is older
    than RESOLVE_GRACE_MINUTES. Appends "auto_resolved" timeline event in SQL.

    Returns count of auto-resolved incidents.
    """
    now = datetime.utcnow()
    now_iso = now.isoformat() + "Z"
    grace_cutoff = now - timedelta(minutes=_RESOLVE_GRACE_MINUTES)

    result = await session.execute(
        text("""
            UPDATE ops_incidents
            SET status      = 'resolved',
                resolved_at = :now ::TIMESTAMPTZ,
                updated_at  = :now ::TIMESTAMPTZ,
                timeline    = COALESCE(timeline, '[]'::jsonb)
                              || jsonb_build_array(jsonb_build_object(
                                  'ts',      :now_iso ::TEXT,
                                  'message', :resolve_msg ::TEXT,
                                  'actor',   'system',
                                  'action',  'auto_resolved'
                              ))
            WHERE status IN ('active', 'acknowledged')
              AND last_seen_at < :cutoff ::TIMESTAMPTZ
            RETURNING id
        """),
        {
            "now": now,
            "now_iso": now_iso,
            "resolve_msg": f"Auto-resolved (not seen for {_RESOLVE_GRACE_MINUTES}+ min)",
            "cutoff": grace_cutoff,
        },
    )
    resolved_ids = result.fetchall()

    if resolved_ids:
        await session.commit()

    return len(resolved_ids)


async def _aggregate_incidents(session) -> list[dict]:
    """
    Aggregate incidents from multiple sources, persist to ops_incidents,
    and return the full list from DB.

    Flow:
    1. Detect active incidents from all sources
    2. Upsert into ops_incidents (create/update/reopen)
    3. Auto-resolve stale incidents (grace window)
    4. Read all non-resolved from DB and return as dicts
    """
    # Phase 1: Detect
    detected = await _detect_active_incidents(session)

    # Phase 2: Upsert
    try:
        await _upsert_incidents(session, detected)
    except Exception as e:
        logger.error(f"Failed to upsert incidents: {e}")
        # Rollback and continue with read-only
        await session.rollback()

    # Phase 3: Auto-resolve stale
    try:
        resolved_count = await _auto_resolve_stale_incidents(session)
        if resolved_count > 0:
            logger.info(f"Auto-resolved {resolved_count} stale incidents (grace={_RESOLVE_GRACE_MINUTES}m)")
    except Exception as e:
        logger.warning(f"Failed to auto-resolve incidents: {e}")
        await session.rollback()

    # Phase 4: Read all from DB
    try:
        result = await session.execute(
            text("""
                SELECT id, source, source_key, severity, status, type, title,
                       description, details, runbook_url, timeline,
                       created_at, last_seen_at, acknowledged_at, resolved_at, updated_at
                FROM ops_incidents
                ORDER BY
                    CASE severity WHEN 'critical' THEN 0 WHEN 'warning' THEN 1 ELSE 2 END,
                    created_at DESC
            """)
        )
        rows = result.mappings().all()
        incidents = []

        def _ts_iso(dt) -> str | None:
            """Serialize TIMESTAMPTZ to ISO 8601 with Z suffix (no +00:00 duplication)."""
            if dt is None:
                return None
            s = dt.isoformat()
            # asyncpg returns aware datetimes with +00:00; replace with Z for JS compat
            if s.endswith("+00:00"):
                s = s[:-6] + "Z"
            elif not s.endswith("Z"):
                s += "Z"
            return s

        for row in rows:
            inc = {
                "id": row["id"],
                "severity": row["severity"],
                "status": row["status"],
                "type": row["type"],
                "title": row["title"],
                "description": row["description"] or "",
                "created_at": _ts_iso(row["created_at"]),
                "updated_at": _ts_iso(row["updated_at"]),
                "runbook_url": row["runbook_url"],
                "details": row["details"],
                "timeline": row["timeline"] or [],
                "acknowledged_at": _ts_iso(row["acknowledged_at"]),
                "resolved_at": _ts_iso(row["resolved_at"]),
                "source": row["source"],
                "last_seen_at": _ts_iso(row["last_seen_at"]),
            }
            incidents.append(inc)
        return incidents
    except Exception as e:
        logger.error(f"Failed to read incidents from DB: {e}")
        # Fallback: return detected incidents as dicts (ephemeral, like before)
        return detected


@app.get("/dashboard/incidents.json")
async def get_incidents_dashboard(
    request: Request,
    status: list[str] = Query(default=[]),
    severity: list[str] = Query(default=[]),
    type: str = Query(default=None, alias="type"),
    q: str = Query(default=None),
    page: int = Query(default=1, ge=1),
    limit: int = Query(default=50, ge=1, le=100),
    session: AsyncSession = Depends(get_async_session),
):
    """
    Unified incidents endpoint for Dashboard.

    Aggregates incidents from multiple sources:
    - Sentry issues (errors/exceptions)
    - Predictions health alerts
    - Scheduler jobs health
    - FastPath/LLM health
    - API budget warnings

    Query params:
    - status: active|acknowledged|resolved (multi-select)
    - severity: info|warning|critical (multi-select)
    - type: sentry|predictions|scheduler|llm|api_budget
    - q: search substring in title/description
    - page: pagination (default 1)
    - limit: page size (default 50, max 100)

    Response:
    {
        "generated_at": "2026-01-23T...",
        "cached": true,
        "cache_age_seconds": 12,
        "data": {
            "incidents": [...],
            "total": 15,
            "page": 1,
            "limit": 50,
            "pages": 1
        }
    }

    Auth: X-Dashboard-Token header required.
    """
    if not _verify_dashboard_token(request):
        raise HTTPException(status_code=401, detail="Dashboard access requires valid token.")

    now = time.time()
    now_iso = datetime.utcnow().isoformat() + "Z"

    # Check cache
    if (
        _incidents_cache["data"] is not None
        and (now - _incidents_cache["timestamp"]) < _incidents_cache["ttl"]
    ):
        all_incidents = _incidents_cache["data"]
        cached = True
        cache_age = round(now - _incidents_cache["timestamp"], 1)
    else:
        # Fetch fresh data
        try:
            all_incidents = await _aggregate_incidents(session)
            _incidents_cache["data"] = all_incidents
            _incidents_cache["timestamp"] = now
            cached = False
            cache_age = 0
        except Exception as e:
            logger.error(f"Failed to aggregate incidents: {e}")
            all_incidents = []
            cached = False
            cache_age = 0

    # Apply filters
    filtered = all_incidents

    # Filter by status (multi-select)
    if status:
        valid_statuses = {"active", "acknowledged", "resolved"}
        status_filter = set(s.lower() for s in status if s.lower() in valid_statuses)
        if status_filter:
            filtered = [i for i in filtered if i["status"] in status_filter]

    # Filter by severity (multi-select)
    if severity:
        valid_severities = {"info", "warning", "critical"}
        severity_filter = set(s.lower() for s in severity if s.lower() in valid_severities)
        if severity_filter:
            filtered = [i for i in filtered if i["severity"] in severity_filter]

    # Filter by type
    if type:
        filtered = [i for i in filtered if i["type"] == type]

    # Filter by search query (substring in title or description)
    if q:
        q_lower = q.lower()
        filtered = [
            i for i in filtered
            if q_lower in i["title"].lower() or q_lower in i["description"].lower()
        ]

    # Pagination
    total = len(filtered)
    pages = max(1, (total + limit - 1) // limit)
    page = min(page, pages)  # Clamp to valid range
    start = (page - 1) * limit
    end = start + limit
    paginated = filtered[start:end]

    return {
        "generated_at": now_iso,
        "cached": cached,
        "cache_age_seconds": cache_age,
        "data": {
            "incidents": paginated,
            "total": total,
            "page": page,
            "limit": limit,
            "pages": pages,
        },
    }


@app.patch("/dashboard/incidents/{incident_id}")
async def patch_incident(
    incident_id: int,
    request: Request,
    session: AsyncSession = Depends(get_async_session),
):
    """
    Update incident status (acknowledge/resolve).

    Body: {"status": "acknowledged"|"resolved"}

    - Persists status change to ops_incidents.
    - Sets acknowledged_at / resolved_at timestamps.
    - Appends timeline event with actor="user".

    Auth: X-Dashboard-Token header required.
    """
    if not _verify_dashboard_token(request):
        raise HTTPException(status_code=401, detail="Dashboard access requires valid token.")

    import json as _json

    body = await request.json()
    new_status = body.get("status")
    if new_status not in ("acknowledged", "resolved"):
        raise HTTPException(status_code=400, detail="status must be 'acknowledged' or 'resolved'")

    # Fetch current incident
    result = await session.execute(
        text("SELECT id, status, timeline FROM ops_incidents WHERE id = :id"),
        {"id": incident_id},
    )
    row = result.mappings().first()
    if not row:
        raise HTTPException(status_code=404, detail="Incident not found")

    current_status = row["status"]
    if current_status == new_status:
        return {"ok": True, "message": f"Already {new_status}"}

    # Validate transition
    if new_status == "acknowledged" and current_status != "active":
        raise HTTPException(status_code=400, detail="Can only acknowledge active incidents")
    if new_status == "resolved" and current_status == "resolved":
        raise HTTPException(status_code=400, detail="Already resolved")

    now = datetime.utcnow()
    now_iso = now.isoformat() + "Z"

    # Build timeline event
    old_timeline = row["timeline"] or []
    if isinstance(old_timeline, str):
        old_timeline = _json.loads(old_timeline)
    new_timeline = list(old_timeline)
    new_timeline.append({
        "ts": now_iso,
        "message": f"Status changed: {current_status} → {new_status}",
        "actor": "user",
        "action": new_status,
    })

    # Update
    if new_status == "acknowledged":
        await session.execute(
            text("""
                UPDATE ops_incidents
                SET status = 'acknowledged', acknowledged_at = :now,
                    updated_at = :now, timeline = CAST(:timeline AS jsonb)
                WHERE id = :id
            """),
            {"id": incident_id, "now": now, "timeline": _json.dumps(new_timeline)},
        )
    elif new_status == "resolved":
        await session.execute(
            text("""
                UPDATE ops_incidents
                SET status = 'resolved', resolved_at = :now,
                    updated_at = :now, timeline = CAST(:timeline AS jsonb)
                WHERE id = :id
            """),
            {"id": incident_id, "now": now, "timeline": _json.dumps(new_timeline)},
        )

    await session.commit()

    # Invalidate cache
    _incidents_cache["data"] = None
    _incidents_cache["timestamp"] = 0

    return {"ok": True, "status": new_status, "updated_at": now_iso}
