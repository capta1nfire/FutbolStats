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
    "match_squad": {},  # keyed by match_id
    "team_squad": {},  # keyed by team_id
    "team_squad_stats": {},  # keyed by team_id:season
    "players_managers": {},  # keyed by view:league_id:limit
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
        stadium_wikidata_id = normalize(body.get("stadium_wikidata_id"))
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

        # Validate stadium_wikidata_id (Q-number format)
        if stadium_wikidata_id:
            if not re.match(r'^Q\d{1,10}$', stadium_wikidata_id):
                raise HTTPException(status_code=400, detail="stadium_wikidata_id must be a Wikidata Q-number (e.g. Q12345)")

        # P0 ABE: Check if all fields are NULL -> DELETE instead
        all_null = all([
            full_name is None,
            short_name is None,
            stadium_name is None,
            stadium_capacity is None,
            stadium_wikidata_id is None,
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
                            stadium_wikidata_id, website, twitter_handle, instagram_handle,
                            source, notes, created_at, updated_at
                        ) VALUES (
                            :tid, :full_name, :short_name, :stadium_name, :stadium_capacity,
                            :stadium_wikidata_id, :website, :twitter_handle, :instagram_handle,
                            :source, :notes, NOW(), NOW()
                        )
                        ON CONFLICT (team_id) DO UPDATE SET
                            full_name = EXCLUDED.full_name,
                            short_name = EXCLUDED.short_name,
                            stadium_name = EXCLUDED.stadium_name,
                            stadium_capacity = EXCLUDED.stadium_capacity,
                            stadium_wikidata_id = EXCLUDED.stadium_wikidata_id,
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
                        "stadium_wikidata_id": stadium_wikidata_id,
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

    # Invalidate coverage map cache (season_start_month, display_name changes)
    from app.dashboard.dashboard_views_routes import _coverage_map_cache
    _coverage_map_cache["data"] = None
    _coverage_map_cache["params"] = None

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


# =============================================================================
# Players & Managers — Squad Data (P4)
# Consumed by Football > Players, Football > Managers, Match Detail > Squad,
# TeamDrawer > overview. All protected by dashboard token.
# =============================================================================


async def _get_match_squad_side(session, team_id: int, team_name: str,
                                fixture_ext_id, match_date) -> dict:
    """Build squad side (injuries + manager) for one team in a match. PIT-strict."""
    # Injuries: PIT (captured_at < match.date)
    injuries = []
    if fixture_ext_id:
        inj_result = await session.execute(
            text("""
                SELECT player_name, injury_type, injury_reason
                FROM player_injuries
                WHERE team_id = :tid
                  AND fixture_external_id = :fid
                  AND captured_at < :match_date
                ORDER BY injury_type, player_name
            """),
            {"tid": team_id, "fid": fixture_ext_id, "match_date": match_date},
        )
        injuries = [
            {
                "player_name": r.player_name,
                "injury_type": r.injury_type,
                "injury_reason": r.injury_reason,
            }
            for r in inj_result.fetchall()
        ]

    # Manager: PIT (detected_at < match.date, active at match date)
    mgr_result = await session.execute(
        text("""
            SELECT tmh.manager_external_id, tmh.manager_name, tmh.start_date,
                   mg.nationality, mg.photo_url,
                   ((:match_date)::date - tmh.start_date) AS tenure_days
            FROM team_manager_history tmh
            LEFT JOIN managers mg ON mg.external_id = tmh.manager_external_id
            WHERE tmh.team_id = :tid
              AND tmh.start_date <= (:match_date)::date
              AND (tmh.end_date IS NULL OR tmh.end_date > (:match_date)::date)
              AND tmh.detected_at < :match_date
            ORDER BY tmh.start_date DESC
            LIMIT 1
        """),
        {"tid": team_id, "match_date": match_date},
    )
    mgr = mgr_result.fetchone()
    manager = None
    if mgr:
        manager = {
            "external_id": mgr.manager_external_id,
            "name": mgr.manager_name,
            "nationality": mgr.nationality,
            "photo_url": mgr.photo_url,
            "start_date": mgr.start_date.isoformat() if mgr.start_date else None,
            "tenure_days": mgr.tenure_days if mgr.tenure_days is not None else None,
        }

    return {
        "team_id": team_id,
        "team_name": team_name,
        "manager": manager,
        "injuries": injuries,
    }


@router.get("/match/{match_id}/squad.json")
async def dashboard_admin_match_squad(request: Request, match_id: int):
    """Match squad: injuries + managers for both teams (PIT-strict).

    ABE P0-2: captured_at < match.date for injuries, detected_at < match.date for managers.
    """
    _check_token(request)

    now = time.time()
    cache_key = str(match_id)
    cache = _admin_cache["match_squad"]

    if cache_key in cache and cache[cache_key]["data"] and (now - cache[cache_key]["timestamp"]) < 120:
        cached = cache[cache_key]
        return {
            "generated_at": cached["data"]["generated_at"],
            "cached": True,
            "cache_age_seconds": round(now - cached["timestamp"], 1),
            "data": cached["data"]["data"],
        }

    async with AsyncSessionLocal() as session:
        match_result = await session.execute(
            text("""
                SELECT m.id, m.external_id, m.date, m.status,
                       m.home_team_id, m.away_team_id,
                       ht.name AS home_team_name, at.name AS away_team_name
                FROM matches m
                JOIN teams ht ON m.home_team_id = ht.id
                JOIN teams at ON m.away_team_id = at.id
                WHERE m.id = :mid
            """),
            {"mid": match_id},
        )
        match = match_result.fetchone()

        if not match:
            raise HTTPException(status_code=404, detail=f"Match {match_id} not found.")

        home = await _get_match_squad_side(
            session, match.home_team_id, match.home_team_name,
            match.external_id, match.date,
        )
        away = await _get_match_squad_side(
            session, match.away_team_id, match.away_team_name,
            match.external_id, match.date,
        )

    data = {"match_id": match_id, "home": home, "away": away}
    result_obj = {"generated_at": datetime.utcnow().isoformat() + "Z", "data": data}
    cache[cache_key] = {"data": result_obj, "timestamp": now}

    return {
        "generated_at": result_obj["generated_at"],
        "cached": False,
        "cache_age_seconds": None,
        "data": data,
    }


@router.get("/team/{team_id}/squad.json")
async def dashboard_admin_team_squad(request: Request, team_id: int):
    """Team squad: current manager + history + active injuries (upcoming 14d window).

    ABE P0-3: active absences = fixture_date in [NOW(), NOW()+14d].
    """
    _check_token(request)

    now = time.time()
    cache_key = str(team_id)
    cache = _admin_cache["team_squad"]

    if cache_key in cache and cache[cache_key]["data"] and (now - cache[cache_key]["timestamp"]) < 120:
        cached = cache[cache_key]
        return {
            "generated_at": cached["data"]["generated_at"],
            "cached": True,
            "cache_age_seconds": round(now - cached["timestamp"], 1),
            "data": cached["data"]["data"],
        }

    async with AsyncSessionLocal() as session:
        team_result = await session.execute(
            text("SELECT id, name FROM teams WHERE id = :tid"),
            {"tid": team_id},
        )
        team = team_result.fetchone()
        if not team:
            raise HTTPException(status_code=404, detail=f"Team {team_id} not found.")

        # Current manager (end_date IS NULL)
        mgr_result = await session.execute(
            text("""
                SELECT tmh.manager_external_id, tmh.manager_name, tmh.start_date,
                       mg.nationality, mg.photo_url,
                       (CURRENT_DATE - tmh.start_date) AS tenure_days
                FROM team_manager_history tmh
                LEFT JOIN managers mg ON mg.external_id = tmh.manager_external_id
                WHERE tmh.team_id = :tid AND tmh.end_date IS NULL
                ORDER BY tmh.start_date DESC
                LIMIT 1
            """),
            {"tid": team_id},
        )
        mgr = mgr_result.fetchone()
        current_manager = None
        if mgr:
            current_manager = {
                "external_id": mgr.manager_external_id,
                "name": mgr.manager_name,
                "nationality": mgr.nationality,
                "photo_url": mgr.photo_url,
                "start_date": mgr.start_date.isoformat() if mgr.start_date else None,
                "tenure_days": mgr.tenure_days if mgr.tenure_days is not None else None,
            }

        # Manager history (last 10 stints)
        history_result = await session.execute(
            text("""
                SELECT tmh.manager_external_id, tmh.manager_name, tmh.start_date,
                       tmh.end_date, mg.nationality, mg.photo_url
                FROM team_manager_history tmh
                LEFT JOIN managers mg ON mg.external_id = tmh.manager_external_id
                WHERE tmh.team_id = :tid
                ORDER BY tmh.start_date DESC
                LIMIT 10
            """),
            {"tid": team_id},
        )
        manager_history = [
            {
                "external_id": r.manager_external_id,
                "name": r.manager_name,
                "nationality": r.nationality,
                "photo_url": r.photo_url,
                "start_date": r.start_date.isoformat() if r.start_date else None,
                "end_date": r.end_date.isoformat() if r.end_date else None,
            }
            for r in history_result.fetchall()
        ]

        # Active injuries: P0-3 upcoming 14d window
        inj_result = await session.execute(
            text("""
                SELECT DISTINCT ON (player_external_id)
                       player_name, injury_type, injury_reason, fixture_date
                FROM player_injuries
                WHERE team_id = :tid
                  AND fixture_date >= NOW()
                  AND fixture_date <= NOW() + INTERVAL '14 days'
                ORDER BY player_external_id, captured_at DESC
            """),
            {"tid": team_id},
        )
        current_injuries = [
            {
                "player_name": r.player_name,
                "injury_type": r.injury_type,
                "injury_reason": r.injury_reason,
                "fixture_date": r.fixture_date.isoformat() if r.fixture_date else None,
            }
            for r in inj_result.fetchall()
        ]

    data = {
        "team_id": team_id,
        "team_name": team.name,
        "current_manager": current_manager,
        "manager_history": manager_history,
        "current_injuries": current_injuries,
    }
    result_obj = {"generated_at": datetime.utcnow().isoformat() + "Z", "data": data}
    cache[cache_key] = {"data": result_obj, "timestamp": now}

    return {
        "generated_at": result_obj["generated_at"],
        "cached": False,
        "cache_age_seconds": None,
        "data": data,
    }


@router.get("/team/{team_id}/squad-stats.json")
async def dashboard_admin_team_squad_stats(
    request: Request,
    team_id: int,
    season: int = None,
):
    """
    Team squad stats: per-player seasonal aggregates from match_player_stats.

    Data source: match_player_stats (API-Football /fixtures/players backfill).

    Notes:
    - Ratings are post-match. They are safe for dashboard display, but PIT-critical
      for modeling: NEVER use same-match ratings for pre-match PTS.
    - This endpoint is admin-only (dashboard token).
    """
    _check_token(request)

    now = time.time()
    cache_key = f"{team_id}:{season or 'latest'}"
    cache = _admin_cache["team_squad_stats"]

    if cache_key in cache and cache[cache_key]["data"] and (now - cache[cache_key]["timestamp"]) < 120:
        cached = cache[cache_key]
        return {
            "generated_at": cached["data"]["generated_at"],
            "cached": True,
            "cache_age_seconds": round(now - cached["timestamp"], 1),
            "data": cached["data"]["data"],
        }

    async with AsyncSessionLocal() as session:
        team_result = await session.execute(
            text("SELECT id, name, external_id FROM teams WHERE id = :tid"),
            {"tid": team_id},
        )
        team = team_result.fetchone()
        if not team:
            raise HTTPException(status_code=404, detail=f"Team {team_id} not found.")

        team_external_id = team.external_id
        if team_external_id is None:
            # Team exists but has no external mapping — cannot query match_player_stats
            data = {
                "team_id": team_id,
                "team_external_id": None,
                "team_name": team.name,
                "season": season,
                "available_seasons": [],
                "players": [],
            }
            result_obj = {"generated_at": datetime.utcnow().isoformat() + "Z", "data": data}
            cache[cache_key] = {"data": result_obj, "timestamp": now}
            return {
                "generated_at": result_obj["generated_at"],
                "cached": False,
                "cache_age_seconds": None,
                "data": data,
            }

        # Available seasons for this team (desc)
        seasons_result = await session.execute(
            text("""
                SELECT DISTINCT m.season
                FROM match_player_stats mps
                JOIN matches m ON m.id = mps.match_id
                WHERE mps.team_external_id = :team_ext
                  AND m.season IS NOT NULL
                ORDER BY m.season DESC
            """),
            {"team_ext": team_external_id},
        )
        available_seasons = [int(r.season) for r in seasons_result.fetchall() if r.season is not None]

        selected_season = season
        if selected_season is None and available_seasons:
            selected_season = available_seasons[0]

        if selected_season is not None and available_seasons and selected_season not in available_seasons:
            raise HTTPException(
                status_code=400,
                detail=f"Season {selected_season} not available for team {team_id}.",
            )

        # Total finished matches for this team in the season
        team_matches_played = 0
        if selected_season is not None:
            tm_result = await session.execute(
                text("""
                    SELECT COUNT(*) AS cnt
                    FROM matches
                    WHERE (home_team_id = :tid OR away_team_id = :tid)
                      AND season = :season
                      AND status = 'FT'
                """),
                {"tid": team_id, "season": selected_season},
            )
            team_matches_played = int(tm_result.scalar() or 0)

        players = []
        if selected_season is not None:
            result = await session.execute(
                text("""
                    SELECT
                        mps.player_external_id,
                        MAX(mps.player_name) AS player_name,
                        COALESCE(MODE() WITHIN GROUP (ORDER BY mps.position), 'U') AS position,
                        COALESCE(MAX(p.jersey_number), MAX((mps.raw_json #>> '{statistics,games,number}')::int)) AS jersey_number,
                        COUNT(*) FILTER (WHERE COALESCE(mps.minutes, 0) > 0) AS appearances,
                        -- Weighted rating by minutes (PTS-aligned)
                        ROUND(
                            SUM(mps.rating * mps.minutes)
                            / NULLIF(SUM(mps.minutes) FILTER (WHERE mps.rating IS NOT NULL AND mps.minutes IS NOT NULL AND mps.minutes > 0), 0),
                            2
                        ) AS avg_rating,
                        SUM(COALESCE(mps.minutes, 0)) AS total_minutes,
                        SUM(COALESCE(mps.goals, 0)) AS goals,
                        SUM(COALESCE(mps.assists, 0)) AS assists,
                        SUM(COALESCE(mps.saves, 0)) AS saves,
                        SUM(COALESCE(mps.yellow_cards, 0)) AS yellows,
                        SUM(COALESCE(mps.red_cards, 0)) AS reds,
                        SUM(COALESCE(mps.passes_key, 0)) AS key_passes,
                        SUM(COALESCE(mps.tackles, 0)) AS tackles,
                        SUM(COALESCE(mps.interceptions, 0)) AS interceptions,
                        SUM(COALESCE(mps.shots_total, 0)) AS shots_total,
                        SUM(COALESCE(mps.shots_on_target, 0)) AS shots_on_target,
                        SUM(COALESCE(mps.passes_total, 0)) AS passes_total,
                        ROUND(
                            AVG(mps.passes_accuracy) FILTER (WHERE mps.passes_accuracy IS NOT NULL AND mps.minutes > 0),
                            0
                        ) AS passes_accuracy,
                        SUM(COALESCE(mps.blocks, 0)) AS blocks,
                        SUM(COALESCE(mps.duels_total, 0)) AS duels_total,
                        SUM(COALESCE(mps.duels_won, 0)) AS duels_won,
                        SUM(COALESCE(mps.dribbles_attempts, 0)) AS dribbles_attempts,
                        SUM(COALESCE(mps.dribbles_success, 0)) AS dribbles_success,
                        SUM(COALESCE(mps.fouls_drawn, 0)) AS fouls_drawn,
                        SUM(COALESCE(mps.fouls_committed, 0)) AS fouls_committed,
                        BOOL_OR(COALESCE(mps.is_captain, false)) AS ever_captain,
                        -- Bio fields from players table
                        MAX(p.firstname) AS firstname,
                        MAX(p.lastname) AS lastname,
                        MAX(p.birth_date::text) AS birth_date,
                        MAX(p.birth_place) AS birth_place,
                        MAX(p.birth_country) AS birth_country,
                        MAX(p.nationality) AS nationality,
                        MAX(p.height) AS height,
                        MAX(p.weight) AS weight,
                        MAX(p.photo_url) AS photo_url,
                        -- HQ photo URLs: best available (contextual > global)
                        (SELECT a.cdn_url FROM player_photo_assets a
                         WHERE a.player_external_id = mps.player_external_id
                           AND a.is_active = true AND a.asset_type = 'thumb'
                           AND a.style IN ('segmented', 'raw')
                         ORDER BY (a.context_team_id = :team_id)::int DESC, a.updated_at DESC
                         LIMIT 1) AS photo_url_thumb_hq,
                        (SELECT a.cdn_url FROM player_photo_assets a
                         WHERE a.player_external_id = mps.player_external_id
                           AND a.is_active = true AND a.asset_type = 'card'
                           AND a.style IN ('segmented', 'raw')
                         ORDER BY (a.context_team_id = :team_id)::int DESC, a.updated_at DESC
                         LIMIT 1) AS photo_url_card_hq
                    FROM match_player_stats mps
                    JOIN matches m ON m.id = mps.match_id
                    LEFT JOIN players p ON p.external_id = mps.player_external_id
                    WHERE mps.team_external_id = :team_ext
                      AND m.season = :season
                    GROUP BY mps.player_external_id
                    HAVING SUM(COALESCE(mps.minutes, 0)) > 0
                    ORDER BY appearances DESC, total_minutes DESC
                """),
                {"team_ext": team_external_id, "season": selected_season, "team_id": team_id},
            )
            for r in result.fetchall():
                players.append({
                    "player_external_id": int(r.player_external_id),
                    "player_name": r.player_name or f"Player#{int(r.player_external_id)}",
                    "photo_url": r.photo_url,
                    "photo_url_thumb_hq": r.photo_url_thumb_hq,
                    "photo_url_card_hq": r.photo_url_card_hq,
                    "position": r.position,
                    "jersey_number": int(r.jersey_number) if r.jersey_number is not None else None,
                    "appearances": int(r.appearances or 0),
                    "avg_rating": float(r.avg_rating) if r.avg_rating is not None else None,
                    "total_minutes": int(r.total_minutes or 0),
                    "goals": int(r.goals or 0),
                    "assists": int(r.assists or 0),
                    "saves": int(r.saves or 0),
                    "yellows": int(r.yellows or 0),
                    "reds": int(r.reds or 0),
                    "key_passes": int(r.key_passes or 0),
                    "tackles": int(r.tackles or 0),
                    "interceptions": int(r.interceptions or 0),
                    "shots_total": int(r.shots_total or 0),
                    "shots_on_target": int(r.shots_on_target or 0),
                    "passes_total": int(r.passes_total or 0),
                    "passes_accuracy": int(r.passes_accuracy) if r.passes_accuracy is not None else None,
                    "blocks": int(r.blocks or 0),
                    "duels_total": int(r.duels_total or 0),
                    "duels_won": int(r.duels_won or 0),
                    "dribbles_attempts": int(r.dribbles_attempts or 0),
                    "dribbles_success": int(r.dribbles_success or 0),
                    "fouls_drawn": int(r.fouls_drawn or 0),
                    "fouls_committed": int(r.fouls_committed or 0),
                    "ever_captain": bool(r.ever_captain),
                    "firstname": r.firstname,
                    "lastname": r.lastname,
                    "birth_date": r.birth_date,
                    "birth_place": r.birth_place,
                    "birth_country": r.birth_country,
                    "nationality": r.nationality,
                    "height": r.height,
                    "weight": r.weight,
                })

    data = {
        "team_id": team_id,
        "team_external_id": int(team_external_id) if team_external_id is not None else None,
        "team_name": team.name,
        "season": selected_season,
        "available_seasons": available_seasons,
        "team_matches_played": team_matches_played,
        "players": players,
    }
    result_obj = {"generated_at": datetime.utcnow().isoformat() + "Z", "data": data}
    cache[cache_key] = {"data": result_obj, "timestamp": now}

    return {
        "generated_at": result_obj["generated_at"],
        "cached": False,
        "cache_age_seconds": None,
        "data": data,
    }


@router.get("/players-managers.json")
async def dashboard_admin_players_managers(
    request: Request,
    view: str = "injuries",
    league_id: int = None,
    limit: int = 200,
):
    """Global view for Players and Managers categories.

    Args:
        view: 'injuries' (active absences) or 'managers' (current managers)
        league_id: Optional filter by league (ABE P1-2)
        limit: Max results (1-500)
    """
    _check_token(request)

    if view not in ("injuries", "managers"):
        raise HTTPException(status_code=400, detail="view must be 'injuries' or 'managers'")

    limit = max(1, min(limit, 500))

    now = time.time()
    cache_key = f"{view}:{league_id}:{limit}"
    cache = _admin_cache["players_managers"]

    if cache_key in cache and cache[cache_key]["data"] and (now - cache[cache_key]["timestamp"]) < 120:
        cached = cache[cache_key]
        return {
            "generated_at": cached["data"]["generated_at"],
            "cached": True,
            "cache_age_seconds": round(now - cached["timestamp"], 1),
            "data": cached["data"]["data"],
        }

    async with AsyncSessionLocal() as session:
        if view == "injuries":
            data = await _build_injuries_view(session, league_id, limit)
        else:
            data = await _build_managers_view(session, league_id, limit)

    result_obj = {"generated_at": datetime.utcnow().isoformat() + "Z", "data": data}
    cache[cache_key] = {"data": result_obj, "timestamp": now}

    return {
        "generated_at": result_obj["generated_at"],
        "cached": False,
        "cache_age_seconds": None,
        "data": data,
    }


async def _build_injuries_view(session, league_id=None, limit=1000) -> dict:
    """Active injuries grouped by league -> team. P0-3: upcoming 14d window.

    ABE P1-B: filter by admin_leagues.is_active for consistency with Football nav.
    ABE P1-A: global totals independent of LIMIT so badge/UI don't diverge.
    """
    params: dict = {"limit": limit}
    league_filter = ""
    if league_id:
        league_filter = "AND pi.league_id = :league_id"
        params["league_id"] = league_id

    # Global total (independent of LIMIT) for accurate badge count
    total_params: dict = {}
    total_league_filter = ""
    if league_id:
        total_league_filter = "AND pi.league_id = :league_id"
        total_params["league_id"] = league_id

    total_result = await session.execute(
        text(f"""
            SELECT COUNT(*) AS cnt
            FROM (
                SELECT DISTINCT ON (player_external_id, team_id)
                       player_external_id, team_id, league_id
                FROM player_injuries
                WHERE fixture_date >= NOW()
                  AND fixture_date <= NOW() + INTERVAL '14 days'
                ORDER BY player_external_id, team_id, captured_at DESC
            ) pi
            JOIN admin_leagues al ON al.league_id = pi.league_id AND al.is_active = true
            WHERE 1=1 {total_league_filter}
        """),
        total_params,
    )
    global_total = (total_result.scalar() or 0)

    result = await session.execute(
        text(f"""
            SELECT
                pi.league_id,
                COALESCE(al.display_name, al.name) AS league_name,
                pi.team_id,
                t.name AS team_name,
                pi.player_name,
                pi.injury_type,
                pi.injury_reason
            FROM (
                SELECT DISTINCT ON (player_external_id, team_id)
                       player_external_id, team_id, league_id,
                       player_name, injury_type, injury_reason
                FROM player_injuries
                WHERE fixture_date >= NOW()
                  AND fixture_date <= NOW() + INTERVAL '14 days'
                ORDER BY player_external_id, team_id, captured_at DESC
            ) pi
            JOIN teams t ON t.id = pi.team_id
            JOIN admin_leagues al ON al.league_id = pi.league_id AND al.is_active = true
            WHERE 1=1 {league_filter}
            ORDER BY al.name, t.name, pi.player_name
            LIMIT :limit
        """),
        params,
    )
    rows = result.fetchall()

    # Group by league -> team
    leagues_dict: dict = {}
    for r in rows:
        lid = r.league_id
        if lid not in leagues_dict:
            leagues_dict[lid] = {
                "league_id": lid,
                "name": r.league_name or f"League {lid}",
                "teams": {},
            }
        tid = r.team_id
        if tid not in leagues_dict[lid]["teams"]:
            leagues_dict[lid]["teams"][tid] = {
                "team_id": tid,
                "name": r.team_name,
                "injuries": [],
            }
        leagues_dict[lid]["teams"][tid]["injuries"].append({
            "player_name": r.player_name,
            "injury_type": r.injury_type,
            "injury_reason": r.injury_reason,
        })

    leagues = []
    for league_data in leagues_dict.values():
        teams = sorted(league_data["teams"].values(), key=lambda t: t["name"])
        leagues.append({
            "league_id": league_data["league_id"],
            "name": league_data["name"],
            "teams": teams,
            "absences_count": sum(len(t["injuries"]) for t in teams),
        })
    leagues.sort(key=lambda lg: lg["absences_count"], reverse=True)

    return {
        "leagues": leagues,
        "total_absences": global_total,
    }


async def _build_managers_view(session, league_id=None, limit=1000) -> dict:
    """Active managers with tenure. Flag is_new for tenure < 60d.

    ABE P1-A: global totals independent of LIMIT so badge/UI don't diverge.
    """
    params: dict = {"limit": limit}
    league_filter = ""
    if league_id:
        league_filter = "AND tpl.league_id = :league_id"
        params["league_id"] = league_id

    # CTE used by both total and detail queries
    tpl_cte = """
        team_primary_league AS (
            SELECT DISTINCT ON (x.team_id)
                x.team_id, x.league_id
            FROM (
                SELECT home_team_id AS team_id, league_id, date
                FROM matches
                WHERE date >= NOW() - INTERVAL '365 days'
                UNION ALL
                SELECT away_team_id AS team_id, league_id, date
                FROM matches
                WHERE date >= NOW() - INTERVAL '365 days'
            ) x
            JOIN admin_leagues al ON al.league_id = x.league_id
                AND al.is_active = true AND al.kind = 'league'
            ORDER BY x.team_id, x.date DESC
        )
    """

    # Global totals (independent of LIMIT) for accurate badge counts
    total_params: dict = {}
    total_league_filter = ""
    if league_id:
        total_league_filter = "AND tpl.league_id = :league_id"
        total_params["league_id"] = league_id

    total_result = await session.execute(
        text(f"""
            WITH {tpl_cte}
            SELECT
                COUNT(*) AS total,
                COUNT(*) FILTER (WHERE (CURRENT_DATE - tmh.start_date) < 60) AS new_count
            FROM team_manager_history tmh
            JOIN team_primary_league tpl ON tpl.team_id = tmh.team_id
            WHERE tmh.end_date IS NULL
              {total_league_filter}
        """),
        total_params,
    )
    totals_row = total_result.fetchone()
    global_total = totals_row.total if totals_row else 0
    global_new = totals_row.new_count if totals_row else 0

    result = await session.execute(
        text(f"""
            WITH {tpl_cte}
            SELECT
                tmh.team_id,
                t.name AS team_name,
                tpl.league_id,
                COALESCE(al.display_name, al.name) AS league_name,
                tmh.manager_external_id,
                tmh.manager_name,
                mg.nationality,
                mg.photo_url,
                tmh.start_date,
                (CURRENT_DATE - tmh.start_date) AS tenure_days
            FROM team_manager_history tmh
            JOIN teams t ON t.id = tmh.team_id
            LEFT JOIN managers mg ON mg.external_id = tmh.manager_external_id
            JOIN team_primary_league tpl ON tpl.team_id = tmh.team_id
            LEFT JOIN admin_leagues al ON al.league_id = tpl.league_id
            WHERE tmh.end_date IS NULL
              {league_filter}
            ORDER BY (CURRENT_DATE - tmh.start_date) ASC
            LIMIT :limit
        """),
        params,
    )
    rows = result.fetchall()

    managers = [
        {
            "team_id": r.team_id,
            "team_name": r.team_name,
            "league_id": r.league_id,
            "league_name": r.league_name,
            "manager": {
                "external_id": r.manager_external_id,
                "name": r.manager_name,
                "nationality": r.nationality,
                "photo_url": r.photo_url,
                "start_date": r.start_date.isoformat() if r.start_date else None,
            },
            "tenure_days": r.tenure_days if r.tenure_days is not None else None,
            "is_new": (r.tenure_days or 999) < 60,
        }
        for r in rows
    ]

    return {
        "managers": managers,
        "total_managers": global_total,
        "new_managers_count": global_new,
    }
