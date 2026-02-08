"""Public API endpoints — predictions, matches, standings, teams, ETL, recalibration.

39 endpoints under various paths (/, /predictions/*, /matches/*, /standings/*,
/teams/*, /etl/*, /model/*, /odds/*, /audit/*, /recalibration/*, /lineup/*).
Auth: mix of verify_api_key (protected) and public (no auth).
Extracted from main.py Step 5 (final extraction).
"""

import json
import logging
import os
import time
from datetime import datetime, timedelta
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, model_validator
from sqlalchemy import bindparam, func, select, text, column
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.database import AsyncSessionLocal, get_async_session, get_pool_status
from app.etl import APIFootballProvider, ETLPipeline
from app.etl.competitions import ALL_LEAGUE_IDS, COMPETITIONS
from app.etl.sota_constants import SOFASCORE_SUPPORTED_LEAGUES, UNDERSTAT_SUPPORTED_LEAGUES
from app.features import FeatureEngineer
from app.ml.persistence import load_active_model, persist_model_snapshot
from app.models import (
    JobRun, Match, OddsHistory, OpsAlert, PITReport, PostMatchAudit,
    Prediction, PredictionOutcome, SensorPrediction, ShadowPrediction,
    Team, TeamAdjustment, TeamOverride,
)
from app.teams.overrides import preload_team_overrides, resolve_team_display
from app.scheduler import get_last_sync_time, get_sync_leagues, SYNC_LEAGUES, global_sync_window
from app.security import limiter, verify_api_key, verify_api_key_or_ops_session
from app.state import ml_engine, _telemetry, _incr, _live_summary_cache
from app.utils.standings import (
    select_standings_view, StandingsGroupNotFound, apply_zones,
    group_standings_by_name, select_default_standings_group,
    classify_group_type,
)

router = APIRouter(tags=["api"])

logger = logging.getLogger(__name__)
settings = get_settings()

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
                    await session.execute(text("SAVEPOINT sp_pred"))
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
                    await session.execute(text("RELEASE SAVEPOINT sp_pred"))
                    saved += 1
                except Exception as e:
                    try:
                        await session.execute(text("ROLLBACK TO SAVEPOINT sp_pred"))
                    except Exception:
                        pass
                    logger.warning(f"[STARTUP] Predictions catch-up: match {match_id} failed: {e}")

            await session.commit()
            logger.info(
                f"[STARTUP] Predictions catch-up complete: saved={saved}, "
                f"ns_matches={len(df_ns)}, model={ml_engine.model_version}"
            )

    except Exception as e:
        logger.error(f"[STARTUP] Predictions catch-up failed: {e}")



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

@router.get("/sync/status")
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


@router.get("/")
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


@router.post("/etl/sync", response_model=ETLSyncResponse)
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


@router.post("/etl/sync-historical")
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


@router.post("/etl/sync-window")
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


@router.post("/etl/refresh-aggregates")
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


@router.get("/aggregates/status")
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


@router.get("/aggregates/breakdown")
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


@router.post("/model/train", response_model=TrainResponse)
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


@router.get("/predictions/upcoming", response_model=PredictionsResponse)
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

    # ═══════════════════════════════════════════════════════════════
    # FASE 2: Market anchor — blend model probs with market for low-signal leagues
    # ═══════════════════════════════════════════════════════════════
    _t5b = time.time()
    from app.ml.policy import apply_market_anchor
    predictions, anchor_metadata = apply_market_anchor(
        predictions,
        alpha_default=policy_config["market_anchor_alpha_default"],
        league_overrides=policy_config["market_anchor_league_overrides"],
        enabled=policy_config["market_anchor_enabled"],
    )
    _stage_times["market_anchor_ms"] = (time.time() - _t5b) * 1000
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


@router.get("/predictions/match/{match_id}")
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

# _live_summary_cache imported from app.state (shared with ops_routes.py)

# Live statuses that indicate a match is currently being played
LIVE_STATUSES = frozenset(["1H", "HT", "2H", "ET", "BT", "P", "LIVE", "INT", "SUSP"])


@router.get("/live-summary")
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


@router.get("/teams")
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


@router.get("/matches")
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


@router.get("/competitions")
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


@router.get("/model/info")
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


@router.get("/model/shadow-report")
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


@router.get("/model/sensor-report")
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


@router.post("/odds/refresh")
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


@router.get("/teams/{team_id}/history")
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


@router.get("/matches/{match_id}/details")
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


@router.get("/matches/{match_id}/insights")
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


@router.get("/matches/{match_id}/timeline")
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


@router.get("/matches/{match_id}/odds-history")
async def get_match_odds_history(
    match_id: int,
    source: Optional[str] = Query(None, description="Filter by bookmaker source (e.g. Bet365, consensus, Pinnacle)"),
    session: AsyncSession = Depends(get_async_session),
    _: bool = Depends(verify_api_key),
):
    """
    Get odds history for a match showing how odds changed over time.

    Returns all recorded odds snapshots for the match, ordered by time.
    Optional ?source= filter to view a single bookmaker's history.
    """
    # Get match
    match = await session.get(Match, match_id)
    if not match:
        raise HTTPException(status_code=404, detail="Match not found")

    # Get team names
    home_team = await session.get(Team, match.home_team_id)
    away_team = await session.get(Team, match.away_team_id)

    # Get odds history (optionally filtered by source)
    query = select(OddsHistory).where(OddsHistory.match_id == match_id)
    if source:
        query = query.where(OddsHistory.source == source)
    query = query.order_by(OddsHistory.recorded_at.asc())
    result = await session.execute(query)
    history = result.scalars().all()

    # Available sources for client discovery
    if source:
        # Need a separate query for all sources
        all_sources_result = await session.execute(
            select(OddsHistory.source)
            .where(OddsHistory.match_id == match_id)
            .distinct()
        )
        available_sources = sorted([row[0] for row in all_sources_result.fetchall()])
    else:
        available_sources = sorted(set(h.source for h in history))

    # P0-1: For movement, use best available source by priority (not hardcoded Bet365)
    priority_order = ["Bet365", "Pinnacle", "1xBet", "Unibet", "William Hill",
                      "Betfair", "Bwin", "888sport"]
    movement_source = source  # If client specified, use that
    if not movement_source:
        # Pick best available by priority
        for pb in priority_order:
            if pb in available_sources:
                movement_source = pb
                break
        if not movement_source and available_sources:
            movement_source = available_sources[0]

    # Filter history to single source for movement calculation
    movement_entries = [h for h in history if h.source == movement_source] if movement_source else []
    movement = None
    if len(movement_entries) >= 2:
        opening = movement_entries[0]
        current = movement_entries[-1]
        if opening.odds_home and current.odds_home:
            movement = {
                "source": movement_source,
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
        "available_sources": available_sources,
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


@router.get("/standings/{league_id}")
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


@router.post("/audit/run", response_model=AuditResponse)
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


@router.get("/audit/summary", response_model=AuditSummaryResponse)
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


@router.get("/recalibration/status", response_model=RecalibrationStatusResponse)
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


@router.get("/recalibration/team-adjustments", response_model=list[TeamAdjustmentResponse])
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


@router.post("/recalibration/calculate-adjustments")
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


@router.get("/recalibration/league-drift")
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


@router.get("/recalibration/odds-movement")
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


@router.get("/recalibration/odds-movement/{match_id}")
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


@router.get("/recalibration/lineup/{match_external_id}")
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


@router.get("/matches/{match_id}/lineup")
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


@router.get("/recalibration/snapshots", response_model=list[ModelSnapshotResponse])
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


@router.post("/recalibration/create-baseline")
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


@router.post("/lineup/monitor")
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


@router.get("/lineup/snapshots")
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



# Dashboard views (PIT, TITAN, tables, predictions, analytics) moved to app/dashboard/dashboard_views_routes.py
