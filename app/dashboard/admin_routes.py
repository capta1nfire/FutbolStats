"""Admin Panel API — Dashboard endpoints for admin CRUD.

12 endpoints for admin panel: overview, leagues, teams, enrichment,
audit, league-groups. All protected by dashboard token.
"""

import re
import time
from datetime import datetime

from fastapi import APIRouter, HTTPException, Request
from sqlalchemy import text

from app.database import AsyncSessionLocal
from app.security import verify_dashboard_token_bool

router = APIRouter(prefix="/dashboard/admin", tags=["admin"])

# Cache for admin endpoints (120s for lists, 60s for details)
_admin_cache = {
    "overview": {"data": None, "timestamp": 0, "ttl": 120},
    "leagues": {"data": None, "timestamp": 0, "ttl": 120},
    "league_detail": {},  # keyed by league_id
    "teams": {},  # keyed by filter params
    "team_detail": {},  # keyed by team_id
}


def _check_token(request: Request):
    """Verify dashboard token (header + session + query param dev)."""
    if not verify_dashboard_token_bool(request):
        raise HTTPException(status_code=401, detail="Dashboard access requires valid token.")


def invalidate_admin_cache(section: str, key: str = None):
    """Invalidate admin cache entry. Exported for cross-module use."""
    if key is None:
        if isinstance(_admin_cache.get(section), dict) and "data" in _admin_cache[section]:
            _admin_cache[section]["data"] = None
        elif section in _admin_cache:
            _admin_cache[section] = {}
    else:
        if section in _admin_cache and key in _admin_cache[section]:
            del _admin_cache[section][key]


# =============================================================================
# Admin Panel P0 (Read-only)
# =============================================================================


@router.get("/overview.json")
async def dashboard_admin_overview(request: Request):
    """Admin Panel - System overview with counts and coverage summary."""
    _check_token(request)

    now = time.time()
    cache = _admin_cache["overview"]

    if cache["data"] and (now - cache["timestamp"]) < cache["ttl"]:
        return {
            "generated_at": cache["data"]["generated_at"],
            "cached": True,
            "cache_age_seconds": round(now - cache["timestamp"], 1),
            "data": cache["data"]["data"],
        }

    from app.dashboard.admin import build_overview

    async with AsyncSessionLocal() as session:
        data = await build_overview(session)

    result = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "data": data,
    }
    cache["data"] = result
    cache["timestamp"] = now

    return {
        "generated_at": result["generated_at"],
        "cached": False,
        "cache_age_seconds": None,
        "data": data,
    }


@router.get("/leagues.json")
async def dashboard_admin_leagues(request: Request):
    """Admin Panel - List all leagues with configured vs observed distinction."""
    _check_token(request)

    now = time.time()
    cache = _admin_cache["leagues"]

    if cache["data"] and (now - cache["timestamp"]) < cache["ttl"]:
        return {
            "generated_at": cache["data"]["generated_at"],
            "cached": True,
            "cache_age_seconds": round(now - cache["timestamp"], 1),
            "data": cache["data"]["data"],
        }

    from app.dashboard.admin import build_leagues_list

    async with AsyncSessionLocal() as session:
        data = await build_leagues_list(session)

    result = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "data": data,
    }
    cache["data"] = result
    cache["timestamp"] = now

    return {
        "generated_at": result["generated_at"],
        "cached": False,
        "cache_age_seconds": None,
        "data": data,
    }


@router.get("/league/{league_id}.json")
async def dashboard_admin_league_detail(request: Request, league_id: int):
    """Admin Panel - Detail for a specific league."""
    _check_token(request)

    now = time.time()
    cache_key = str(league_id)
    cache = _admin_cache["league_detail"]

    if cache_key in cache and cache[cache_key]["data"] and (now - cache[cache_key]["timestamp"]) < 60:
        cached = cache[cache_key]
        return {
            "generated_at": cached["data"]["generated_at"],
            "cached": True,
            "cache_age_seconds": round(now - cached["timestamp"], 1),
            "data": cached["data"]["data"],
        }

    from app.dashboard.admin import build_league_detail

    async with AsyncSessionLocal() as session:
        data = await build_league_detail(session, league_id)

    if data is None:
        raise HTTPException(status_code=404, detail=f"League {league_id} not found.")

    result = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "data": data,
    }
    cache[cache_key] = {"data": result, "timestamp": now}

    return {
        "generated_at": result["generated_at"],
        "cached": False,
        "cache_age_seconds": None,
        "data": data,
    }


@router.get("/teams.json")
async def dashboard_admin_teams(
    request: Request,
    type: str = "all",
    country: str = None,
    search: str = None,
    limit: int = 100,
    offset: int = 0,
):
    """Admin Panel - List teams with optional filters and search.

    Args:
        type: Filter by team type (all, club, national)
        country: Filter by country name
        search: Search by team name (case-insensitive partial match)
        limit: Max results (1-500)
        offset: Pagination offset
    """
    _check_token(request)

    now = time.time()
    cache_key = f"{type}:{country}:{search}:{limit}:{offset}"
    cache = _admin_cache["teams"]

    # Skip cache for search queries (fresh results)
    if not search and cache_key in cache and cache[cache_key]["data"] and (now - cache[cache_key]["timestamp"]) < 120:
        cached = cache[cache_key]
        return {
            "generated_at": cached["data"]["generated_at"],
            "cached": True,
            "cache_age_seconds": round(now - cached["timestamp"], 1),
            "data": cached["data"]["data"],
        }

    from app.dashboard.admin import build_teams_list

    team_type = type if type != "all" else None

    async with AsyncSessionLocal() as session:
        data = await build_teams_list(session, team_type=team_type, country=country, search=search, limit=limit, offset=offset)

    result = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "data": data,
    }

    # Only cache non-search requests (avoid cache bloat from typeahead)
    if not search:
        cache[cache_key] = {"data": result, "timestamp": now}

    return {
        "generated_at": result["generated_at"],
        "cached": False,
        "cache_age_seconds": None,
        "data": data,
    }


@router.get("/team/{team_id}.json")
async def dashboard_admin_team_detail(request: Request, team_id: int):
    """Admin Panel - Detail for a specific team."""
    _check_token(request)

    now = time.time()
    cache_key = str(team_id)
    cache = _admin_cache["team_detail"]

    if cache_key in cache and cache[cache_key]["data"] and (now - cache[cache_key]["timestamp"]) < 60:
        cached = cache[cache_key]
        return {
            "generated_at": cached["data"]["generated_at"],
            "cached": True,
            "cache_age_seconds": round(now - cached["timestamp"], 1),
            "data": cached["data"]["data"],
        }

    from app.dashboard.admin import build_team_detail

    async with AsyncSessionLocal() as session:
        data = await build_team_detail(session, team_id)

    if data is None:
        raise HTTPException(status_code=404, detail=f"Team {team_id} not found.")

    result = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "data": data,
    }
    cache[cache_key] = {"data": result, "timestamp": now}

    return {
        "generated_at": result["generated_at"],
        "cached": False,
        "cache_age_seconds": None,
        "data": data,
    }


# =============================================================================
# Team Enrichment Override (Manual Corrections)
# =============================================================================


@router.put("/team/{team_id}/enrichment")
async def dashboard_admin_put_team_enrichment(request: Request, team_id: int):
    """
    Create or update manual override for team enrichment data.

    P0 ABE Semantics:
    - Empty string "" -> NULL (clear override for that field)
    - If all fields become NULL -> DELETE the override row
    - Only non-null fields are stored as overrides
    - Audit log for every write
    """
    from app.ops.audit import log_ops_action, OpsActionTimer

    _check_token(request)

    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    # Validate team exists
    async with AsyncSessionLocal() as session:
        team_result = await session.execute(
            text("SELECT id, name FROM teams WHERE id = :tid"),
            {"tid": team_id}
        )
        team = team_result.fetchone()
        if not team:
            raise HTTPException(status_code=404, detail=f"Team {team_id} not found")

        # P0 ABE: Normalize empty strings to None
        def normalize(val):
            if val is None or (isinstance(val, str) and val.strip() == ""):
                return None
            return val.strip() if isinstance(val, str) else val

        full_name = normalize(body.get("full_name"))
        short_name = normalize(body.get("short_name"))
        stadium_name = normalize(body.get("stadium_name"))
        stadium_capacity = body.get("stadium_capacity")
        website = normalize(body.get("website"))
        twitter_handle = normalize(body.get("twitter_handle"))
        instagram_handle = normalize(body.get("instagram_handle"))
        source = normalize(body.get("source")) or "manual"
        notes = normalize(body.get("notes"))

        # Normalize capacity
        if stadium_capacity is not None:
            if isinstance(stadium_capacity, str) and stadium_capacity.strip() == "":
                stadium_capacity = None
            else:
                try:
                    stadium_capacity = int(stadium_capacity)
                    if stadium_capacity < 0 or stadium_capacity >= 200000:
                        raise HTTPException(status_code=400, detail="stadium_capacity must be 0-199999")
                except (ValueError, TypeError):
                    raise HTTPException(status_code=400, detail="stadium_capacity must be a valid integer")

        # P0 ABE: Validate handles (strip @ and spaces)
        if twitter_handle:
            twitter_handle = twitter_handle.lstrip("@").strip()
            if not re.match(r'^[A-Za-z0-9_]{1,15}$', twitter_handle):
                raise HTTPException(status_code=400, detail="Invalid Twitter handle format")

        if instagram_handle:
            instagram_handle = instagram_handle.lstrip("@").strip()
            if not re.match(r'^[A-Za-z0-9_.]{1,30}$', instagram_handle):
                raise HTTPException(status_code=400, detail="Invalid Instagram handle format")

        # P0 ABE: Validate website (only http/https)
        if website:
            if not website.startswith(("http://", "https://")):
                raise HTTPException(status_code=400, detail="Website must start with http:// or https://")
            # Block javascript: and data: schemes
            if website.lower().startswith(("javascript:", "data:")):
                raise HTTPException(status_code=400, detail="Invalid website URL scheme")

        # P0 ABE: Check if all fields are NULL -> DELETE instead
        all_null = all([
            full_name is None,
            short_name is None,
            stadium_name is None,
            stadium_capacity is None,
            website is None,
            twitter_handle is None,
            instagram_handle is None,
        ])

        with OpsActionTimer() as timer:
            if all_null:
                # Delete override if exists
                await session.execute(
                    text("DELETE FROM team_enrichment_overrides WHERE team_id = :tid"),
                    {"tid": team_id}
                )
                action_result = "deleted"
            else:
                # Upsert override
                await session.execute(
                    text("""
                        INSERT INTO team_enrichment_overrides (
                            team_id, full_name, short_name, stadium_name, stadium_capacity,
                            website, twitter_handle, instagram_handle, source, notes,
                            created_at, updated_at
                        ) VALUES (
                            :tid, :full_name, :short_name, :stadium_name, :stadium_capacity,
                            :website, :twitter_handle, :instagram_handle, :source, :notes,
                            NOW(), NOW()
                        )
                        ON CONFLICT (team_id) DO UPDATE SET
                            full_name = EXCLUDED.full_name,
                            short_name = EXCLUDED.short_name,
                            stadium_name = EXCLUDED.stadium_name,
                            stadium_capacity = EXCLUDED.stadium_capacity,
                            website = EXCLUDED.website,
                            twitter_handle = EXCLUDED.twitter_handle,
                            instagram_handle = EXCLUDED.instagram_handle,
                            source = EXCLUDED.source,
                            notes = EXCLUDED.notes,
                            updated_at = NOW()
                    """),
                    {
                        "tid": team_id,
                        "full_name": full_name,
                        "short_name": short_name,
                        "stadium_name": stadium_name,
                        "stadium_capacity": stadium_capacity,
                        "website": website,
                        "twitter_handle": twitter_handle,
                        "instagram_handle": instagram_handle,
                        "source": source,
                        "notes": notes,
                    }
                )
                action_result = "upserted"

            await session.commit()

        # P0 ABE: Audit log
        await log_ops_action(
            session=session,
            request=request,
            action="team_enrichment_override",
            params={
                "team_id": team_id,
                "team_name": team.name,
                "fields_set": [k for k, v in {
                    "full_name": full_name, "short_name": short_name,
                    "stadium_name": stadium_name, "stadium_capacity": stadium_capacity,
                    "website": website, "twitter": twitter_handle, "instagram": instagram_handle,
                }.items() if v is not None],
                "source": source,
            },
            result="ok",
            result_detail={"action": action_result, "team_id": team_id},
            duration_ms=timer.duration_ms,
        )

        # Invalidate team detail cache
        if str(team_id) in _admin_cache["team_detail"]:
            del _admin_cache["team_detail"][str(team_id)]

        return {
            "status": "ok",
            "team_id": team_id,
            "action": action_result,
        }


@router.delete("/team/{team_id}/enrichment")
async def dashboard_admin_delete_team_enrichment(request: Request, team_id: int):
    """
    Delete all manual overrides for a team (revert to automatic data).

    P0 ABE: Audit log for every delete.
    """
    from app.ops.audit import log_ops_action, OpsActionTimer

    _check_token(request)

    async with AsyncSessionLocal() as session:
        # Validate team exists
        team_result = await session.execute(
            text("SELECT id, name FROM teams WHERE id = :tid"),
            {"tid": team_id}
        )
        team = team_result.fetchone()
        if not team:
            raise HTTPException(status_code=404, detail=f"Team {team_id} not found")

        with OpsActionTimer() as timer:
            result = await session.execute(
                text("DELETE FROM team_enrichment_overrides WHERE team_id = :tid RETURNING team_id"),
                {"tid": team_id}
            )
            deleted = result.fetchone() is not None
            await session.commit()

        # P0 ABE: Audit log
        await log_ops_action(
            session=session,
            request=request,
            action="team_enrichment_override_delete",
            params={"team_id": team_id, "team_name": team.name},
            result="ok",
            result_detail={"deleted": deleted, "team_id": team_id},
            duration_ms=timer.duration_ms,
        )

        # Invalidate team detail cache
        if str(team_id) in _admin_cache["team_detail"]:
            del _admin_cache["team_detail"][str(team_id)]

        return {
            "status": "ok",
            "team_id": team_id,
            "deleted": deleted,
        }


# =============================================================================
# P2B - Admin Mutations
# =============================================================================


@router.patch("/leagues/{league_id}.json")
async def dashboard_admin_patch_league(request: Request, league_id: int):
    """
    Admin Panel - Update a league configuration.

    P2B: PATCH mutations with audit trail.
    Whitelist: is_active, country, kind, priority, match_type, match_weight,
               display_order, tags, rules_json, group_id, name
    """
    _check_token(request)

    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    from app.dashboard.admin import patch_league, ValidationError

    try:
        async with AsyncSessionLocal() as session:
            result = await patch_league(session, league_id, body, actor="dashboard")
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    # Invalidate caches
    _admin_cache["overview"]["data"] = None
    _admin_cache["leagues"]["data"] = None
    if str(league_id) in _admin_cache["league_detail"]:
        del _admin_cache["league_detail"][str(league_id)]

    # P0 ABE: Invalidate football navigation cache for tags updates
    from app.dashboard.football_routes import invalidate_football_cache
    invalidate_football_cache(f"football_league_{league_id}")

    return {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "data": result,
    }


@router.patch("/team/{team_id}.json")
async def dashboard_admin_patch_team(request: Request, team_id: int):
    """
    Admin Panel - Patch team wiki fields.

    Supported fields (P0):
      - wiki_url (nullable string)
      - wikidata_id (nullable string)

    Returns (unwrapped, by dashboard client contract):
      { team_id, updated_fields, wiki }
    """
    _check_token(request)

    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    from app.dashboard.admin import ValidationError, patch_team_wiki
    from sqlalchemy.exc import IntegrityError

    try:
        async with AsyncSessionLocal() as session:
            result = await patch_team_wiki(session, team_id, body, actor="dashboard")
    except NotImplementedError as e:
        raise HTTPException(status_code=501, detail=str(e))
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except IntegrityError as e:
        # Handle unique constraint violations (e.g., duplicate wikidata_id)
        if "wikidata_id" in str(e):
            raise HTTPException(status_code=409, detail="wikidata_id already assigned to another team")
        raise HTTPException(status_code=409, detail="Data conflict: value already exists")

    # Invalidate team detail cache
    try:
        if str(team_id) in _admin_cache["team_detail"]:
            del _admin_cache["team_detail"][str(team_id)]
    except Exception:
        pass

    return result


# =============================================================================
# Audit & League Groups
# =============================================================================


@router.get("/audit.json")
async def dashboard_admin_audit(
    request: Request,
    entity_type: str = None,
    entity_id: str = None,
    limit: int = 50,
    offset: int = 0,
):
    """
    Admin Panel - View audit log entries.

    P2B: Audit trail for mutations.
    Optional filters: entity_type, entity_id
    Supported entity_types: admin_leagues, admin_league_groups
    """
    _check_token(request)

    from app.dashboard.admin import get_audit_log, ValidationError

    try:
        async with AsyncSessionLocal() as session:
            data = await get_audit_log(
                session,
                entity_type=entity_type,
                entity_id=entity_id,
                limit=limit,
                offset=offset
            )
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "cached": False,
        "data": data,
    }


@router.get("/league-groups.json")
async def dashboard_admin_league_groups(request: Request):
    """
    Admin Panel - List league groups with aggregated metrics.

    P2C: Paired leagues (Apertura/Clausura) as navigable entities.
    """
    _check_token(request)

    from app.dashboard.admin import build_league_groups_list

    async with AsyncSessionLocal() as session:
        data = await build_league_groups_list(session)

    return {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "cached": False,
        "cache_age_seconds": None,
        "data": data,
    }


@router.get("/league-group/{group_id}.json")
async def dashboard_admin_league_group_detail(request: Request, group_id: int):
    """
    Admin Panel - League group detail with member leagues.

    P2C: Full details for a paired league group.
    """
    _check_token(request)

    from app.dashboard.admin import build_league_group_detail

    async with AsyncSessionLocal() as session:
        data = await build_league_group_detail(session, group_id)

    if data is None:
        raise HTTPException(status_code=404, detail=f"Group {group_id} not found")

    return {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "cached": False,
        "cache_age_seconds": None,
        "data": data,
    }
