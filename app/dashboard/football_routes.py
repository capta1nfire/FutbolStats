"""Football Navigation API (P3) — Dashboard endpoints.

13 endpoints for football navigation: countries, leagues, nationals,
tournaments, World Cup 2026. All protected by dashboard token.
"""

import time
from datetime import datetime
from urllib.parse import unquote

from fastapi import APIRouter, HTTPException, Request

from app.database import AsyncSessionLocal
from app.security import verify_dashboard_token_bool

router = APIRouter(prefix="/dashboard/football", tags=["football"])

# Cache for football nav endpoints
_football_nav_cache: dict = {}
FOOTBALL_NAV_CACHE_TTL = 120  # 2 minutes for lists
FOOTBALL_NAV_DETAIL_CACHE_TTL = 60  # 1 minute for details


def _get_football_cache(key: str, ttl: int) -> tuple:
    """Get cached data if valid. Returns (data, age_seconds) or (None, None)."""
    if key in _football_nav_cache:
        cached = _football_nav_cache[key]
        age = time.time() - cached["timestamp"]
        if age < ttl:
            return cached["data"], int(age)
    return None, None


def _set_football_cache(key: str, data: dict):
    """Set cache entry."""
    _football_nav_cache[key] = {"data": data, "timestamp": time.time()}


def invalidate_football_cache(key: str):
    """Invalidate a specific football cache entry. Called from admin endpoints."""
    if key in _football_nav_cache:
        del _football_nav_cache[key]


def _check_token(request: Request):
    """Verify dashboard token (header + session + query param dev)."""
    if not verify_dashboard_token_bool(request):
        raise HTTPException(status_code=401, detail="Dashboard access requires valid token.")


@router.get("/nav.json")
async def dashboard_football_nav(request: Request):
    """
    Football Navigation - Top-level categories.

    P3: Returns categories for Col 2 top area.
    """
    _check_token(request)

    cache_key = "football_nav"
    cached_data, cache_age = _get_football_cache(cache_key, FOOTBALL_NAV_CACHE_TTL)

    if cached_data:
        return {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "cached": True,
            "cache_age_seconds": cache_age,
            "data": cached_data,
        }

    from app.dashboard.football_nav import build_nav

    async with AsyncSessionLocal() as session:
        data = await build_nav(session)

    _set_football_cache(cache_key, data)

    return {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "cached": False,
        "cache_age_seconds": None,
        "data": data,
    }


@router.get("/leagues/countries.json")
async def dashboard_football_countries(request: Request):
    """
    Football Navigation - Countries with active leagues.

    P3: For Col 2 when category=leagues_by_country.
    """
    _check_token(request)

    cache_key = "football_countries"
    cached_data, cache_age = _get_football_cache(cache_key, FOOTBALL_NAV_CACHE_TTL)

    if cached_data:
        return {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "cached": True,
            "cache_age_seconds": cache_age,
            "data": cached_data,
        }

    from app.dashboard.football_nav import build_countries_list

    async with AsyncSessionLocal() as session:
        data = await build_countries_list(session)

    _set_football_cache(cache_key, data)

    return {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "cached": False,
        "cache_age_seconds": None,
        "data": data,
    }


@router.get("/leagues/country/{country}.json")
async def dashboard_football_country_detail(request: Request, country: str):
    """
    Football Navigation - Leagues for a specific country.

    P3: For Col 4 when a country is selected.
    """
    _check_token(request)

    country = unquote(country)

    cache_key = f"football_country_{country}"
    cached_data, cache_age = _get_football_cache(cache_key, FOOTBALL_NAV_DETAIL_CACHE_TTL)

    if cached_data:
        return {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "cached": True,
            "cache_age_seconds": cache_age,
            "data": cached_data,
        }

    from app.dashboard.football_nav import build_country_detail

    async with AsyncSessionLocal() as session:
        data = await build_country_detail(session, country)

    if data is None:
        raise HTTPException(status_code=404, detail=f"Country '{country}' not found or has no active leagues")

    _set_football_cache(cache_key, data)

    return {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "cached": False,
        "cache_age_seconds": None,
        "data": data,
    }


@router.get("/league/{league_id}.json")
async def dashboard_football_league_detail(request: Request, league_id: int):
    """
    Football Navigation - League detail.

    P3: For Col 4 drilldown of a specific league.
    """
    _check_token(request)

    cache_key = f"football_league_{league_id}"
    cached_data, cache_age = _get_football_cache(cache_key, FOOTBALL_NAV_DETAIL_CACHE_TTL)

    if cached_data:
        return {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "cached": True,
            "cache_age_seconds": cache_age,
            "data": cached_data,
        }

    from app.dashboard.football_nav import build_league_nav_detail

    async with AsyncSessionLocal() as session:
        data = await build_league_nav_detail(session, league_id)

    if data is None:
        raise HTTPException(status_code=404, detail=f"League {league_id} not found or not active")

    _set_football_cache(cache_key, data)

    return {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "cached": False,
        "cache_age_seconds": None,
        "data": data,
    }


@router.get("/group/{group_id}.json")
async def dashboard_football_group_detail(request: Request, group_id: int):
    """
    Football Navigation - League group detail (paired leagues).

    P3: For Col 4 drilldown of a paired league group.
    """
    _check_token(request)

    cache_key = f"football_group_{group_id}"
    cached_data, cache_age = _get_football_cache(cache_key, FOOTBALL_NAV_DETAIL_CACHE_TTL)

    if cached_data:
        return {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "cached": True,
            "cache_age_seconds": cache_age,
            "data": cached_data,
        }

    from app.dashboard.football_nav import build_group_nav_detail

    async with AsyncSessionLocal() as session:
        data = await build_group_nav_detail(session, group_id)

    if data is None:
        raise HTTPException(status_code=404, detail=f"Group {group_id} not found or has no active members")

    _set_football_cache(cache_key, data)

    return {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "cached": False,
        "cache_age_seconds": None,
        "data": data,
    }


@router.get("/overview.json")
async def dashboard_football_overview(request: Request):
    """
    Football Navigation - Overview.

    P3.1: Shows summary counts, upcoming matches, top leagues, and alerts.
    All filtered by admin_leagues.is_active = true.
    """
    _check_token(request)

    cache_key = "football_overview"
    cached_data, cache_age = _get_football_cache(cache_key, FOOTBALL_NAV_CACHE_TTL)

    if cached_data:
        return {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "cached": True,
            "cache_age_seconds": cache_age,
            "data": cached_data,
        }

    from app.dashboard.football_nav import build_football_overview

    async with AsyncSessionLocal() as session:
        data = await build_football_overview(session)

    _set_football_cache(cache_key, data)

    return {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "cached": False,
        "cache_age_seconds": None,
        "data": data,
    }


# =============================================================================
# National Teams (P3.3)
# =============================================================================


@router.get("/nationals/countries.json")
async def dashboard_football_nationals_countries(request: Request):
    """
    Football Navigation - List countries with national teams.

    P3.3: Returns countries with national teams that have matches in active
    international competitions. Ordered by total_matches DESC.
    """
    _check_token(request)

    cache_key = "football_nationals_countries"
    cached_data, cache_age = _get_football_cache(cache_key, FOOTBALL_NAV_CACHE_TTL)

    if cached_data:
        return {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "cached": True,
            "cache_age_seconds": cache_age,
            "data": cached_data,
        }

    from app.dashboard.football_nav import build_nationals_countries_list

    async with AsyncSessionLocal() as session:
        data = await build_nationals_countries_list(session)

    _set_football_cache(cache_key, data)

    return {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "cached": False,
        "cache_age_seconds": None,
        "data": data,
    }


@router.get("/nationals/country/{country}.json")
async def dashboard_football_nationals_country(request: Request, country: str):
    """
    Football Navigation - Country detail with national teams.

    P3.3: The {country} parameter is the team name (e.g., "Portugal", "Spain").
    Returns teams, competitions, recent matches, and stats.
    """
    _check_token(request)

    country = unquote(country)

    cache_key = f"football_nationals_country_{country}"
    cached_data, cache_age = _get_football_cache(cache_key, FOOTBALL_NAV_DETAIL_CACHE_TTL)

    if cached_data:
        return {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "cached": True,
            "cache_age_seconds": cache_age,
            "data": cached_data,
        }

    from app.dashboard.football_nav import build_nationals_country_detail

    async with AsyncSessionLocal() as session:
        data = await build_nationals_country_detail(session, country)

    if data is None:
        raise HTTPException(status_code=404, detail=f"Country '{country}' not found")

    _set_football_cache(cache_key, data)

    return {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "cached": False,
        "cache_age_seconds": None,
        "data": data,
    }


@router.get("/nationals/team/{team_id}.json")
async def dashboard_football_nationals_team(request: Request, team_id: int):
    """
    Football Navigation - Team 360 for national team.

    P3.3: Returns full team details including competitions, stats (overall and
    by competition), recent matches, and head-to-head records.
    Only works for teams with team_type='national'.
    """
    _check_token(request)

    cache_key = f"football_nationals_team_{team_id}"
    cached_data, cache_age = _get_football_cache(cache_key, FOOTBALL_NAV_DETAIL_CACHE_TTL)

    if cached_data:
        return {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "cached": True,
            "cache_age_seconds": cache_age,
            "data": cached_data,
        }

    from app.dashboard.football_nav import build_nationals_team_detail

    async with AsyncSessionLocal() as session:
        data = await build_nationals_team_detail(session, team_id)

    if data is None:
        raise HTTPException(status_code=404, detail=f"National team {team_id} not found")

    _set_football_cache(cache_key, data)

    return {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "cached": False,
        "cache_age_seconds": None,
        "data": data,
    }


# =============================================================================
# Tournaments & Cups (P3.4)
# =============================================================================


@router.get("/tournaments.json")
async def dashboard_football_tournaments(request: Request):
    """
    Football Navigation - List tournaments, cups and international competitions.

    P3.4: Returns tournaments filtered by kind IN ('cup', 'international', 'friendly')
    AND is_active=true. Includes stats per tournament: total_matches, matches_30d,
    seasons_range, last_match, next_match, coverage percentages, participants_count.
    """
    _check_token(request)

    cache_key = "football_tournaments"
    cached_data, cache_age = _get_football_cache(cache_key, FOOTBALL_NAV_CACHE_TTL)

    if cached_data:
        return {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "cached": True,
            "cache_age_seconds": cache_age,
            "data": cached_data,
        }

    from app.dashboard.football_nav import build_tournaments_list

    async with AsyncSessionLocal() as session:
        data = await build_tournaments_list(session)

    _set_football_cache(cache_key, data)

    return {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "cached": False,
        "cache_age_seconds": None,
        "data": data,
    }


# =============================================================================
# World Cup 2026 (P3.5)
# =============================================================================


@router.get("/world-cup-2026/overview.json")
async def dashboard_football_world_cup_overview(request: Request):
    """
    Football Navigation - World Cup 2026 overview.

    P3.5: Returns overview with summary, alerts, and upcoming matches.
    Status can be: "ok", "not_ready", or "disabled".
    Fail-soft: returns status="not_ready" if no data, never 500.
    """
    _check_token(request)

    cache_key = "football_world_cup_overview"
    cached_data, cache_age = _get_football_cache(cache_key, FOOTBALL_NAV_CACHE_TTL)

    if cached_data:
        return {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "cached": True,
            "cache_age_seconds": cache_age,
            "data": cached_data,
        }

    from app.dashboard.football_nav import build_world_cup_overview

    async with AsyncSessionLocal() as session:
        data = await build_world_cup_overview(session)

    _set_football_cache(cache_key, data)

    return {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "cached": False,
        "cache_age_seconds": None,
        "data": data,
    }


@router.get("/world-cup-2026/groups.json")
async def dashboard_football_world_cup_groups(request: Request):
    """
    Football Navigation - World Cup 2026 groups list.

    P3.5: Returns all groups with team standings.
    Status can be: "ok", "not_ready", or "disabled".
    Fail-soft: returns empty groups if no standings data.
    """
    _check_token(request)

    cache_key = "football_world_cup_groups"
    cached_data, cache_age = _get_football_cache(cache_key, FOOTBALL_NAV_DETAIL_CACHE_TTL)

    if cached_data:
        return {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "cached": True,
            "cache_age_seconds": cache_age,
            "data": cached_data,
        }

    from app.dashboard.football_nav import build_world_cup_groups

    async with AsyncSessionLocal() as session:
        data = await build_world_cup_groups(session)

    _set_football_cache(cache_key, data)

    return {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "cached": False,
        "cache_age_seconds": None,
        "data": data,
    }


@router.get("/world-cup-2026/group/{group}.json")
async def dashboard_football_world_cup_group_detail(request: Request, group: str):
    """
    Football Navigation - World Cup 2026 group detail.

    P3.5: Returns standings and matches for a specific group.
    Parameter group is URL-decoded (e.g., "Group A" or "Group%20A").
    Returns 404 if group not found.
    """
    _check_token(request)

    group = unquote(group)

    cache_key = f"football_world_cup_group_{group}"
    cached_data, cache_age = _get_football_cache(cache_key, FOOTBALL_NAV_DETAIL_CACHE_TTL)

    if cached_data:
        return {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "cached": True,
            "cache_age_seconds": cache_age,
            "data": cached_data,
        }

    from app.dashboard.football_nav import build_world_cup_group_detail

    async with AsyncSessionLocal() as session:
        data = await build_world_cup_group_detail(session, group)

    if data is None:
        raise HTTPException(status_code=404, detail=f"Group '{group}' not found")

    _set_football_cache(cache_key, data)

    return {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "cached": False,
        "cache_age_seconds": None,
        "data": data,
    }
