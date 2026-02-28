"""
Admin Panel - Read/write endpoints for leagues/teams management.

P2A: DB-first reads from admin_leagues table.
P2B: PATCH mutations with audit trail.
All functions return data dicts (no cache handling - that's in main.py).
"""

import json
import logging
import re
from datetime import datetime, timezone
from typing import Any, List, Optional

from sqlalchemy import text
from sqlalchemy.exc import ProgrammingError
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)

# In-memory cache for admin_leagues (refreshed on each request, lightweight)
_league_cache: dict[int, dict] = {}


async def _load_league_cache(session: AsyncSession) -> dict[int, dict]:
    """Load admin_leagues into memory cache for fast lookups."""
    global _league_cache

    result = await session.execute(
        text("""
            SELECT
                league_id, name, display_name, logo_url, wikipedia_url,
                country, kind, is_active,
                priority, match_type, match_weight, group_id, source, rules_json,
                season_start_month
            FROM admin_leagues
        """)
    )
    rows = result.fetchall()

    _league_cache = {}
    for r in rows:
        _league_cache[r.league_id] = {
            "league_id": r.league_id,
            "name": r.name,
            "display_name": r.display_name,
            "effective_name": r.display_name or r.name,
            "logo_url": r.logo_url,
            "wikipedia_url": r.wikipedia_url,
            "country": r.country,
            "kind": r.kind,
            "is_active": r.is_active,
            "priority": r.priority,
            "match_type": r.match_type,
            "match_weight": r.match_weight,
            "group_id": r.group_id,
            "source": r.source,
            "rules_json": r.rules_json if isinstance(r.rules_json, dict) else {},
            "season_start_month": r.season_start_month or 8,
            # configured = source in ('seed', 'override')
            "configured": r.source in ("seed", "override"),
        }

    return _league_cache


def get_league_info_sync(league_id: int) -> dict:
    """Get league info from cache (sync version, must call _load_league_cache first)."""
    if league_id in _league_cache:
        entry = _league_cache[league_id]
        return {
            "name": entry["effective_name"],
            "priority": entry["priority"],
            "match_type": entry["match_type"],
            "match_weight": entry["match_weight"],
            "configured": entry["configured"],
        }
    return {
        "name": f"League {league_id}",
        "priority": None,
        "match_type": None,
        "match_weight": None,
        "configured": False,
    }


# =============================================================================
# Overview
# =============================================================================

async def build_overview(session: AsyncSession) -> dict:
    """Build admin overview with counts and coverage summary."""

    # Load league cache first (DB-first)
    await _load_league_cache(session)

    # Admin leagues counts (from admin_leagues table)
    leagues_query = text("""
        SELECT
            COUNT(*) as total,
            COUNT(*) FILTER (WHERE is_active = true) as active,
            COUNT(*) FILTER (WHERE source = 'seed') as seed,
            COUNT(*) FILTER (WHERE source = 'observed') as observed
        FROM admin_leagues
    """)
    leagues_result = await session.execute(leagues_query)
    leagues_row = leagues_result.fetchone()

    # Counts from matches
    counts_query = text("""
        SELECT
            COUNT(DISTINCT league_id) as total_leagues,
            COUNT(DISTINCT CASE WHEN date >= NOW() - INTERVAL '30 days' THEN league_id END) as active_30d,
            COUNT(*) as total_matches,
            COUNT(*) FILTER (WHERE date >= '2025-08-01') as matches_25_26,
            COUNT(*) FILTER (WHERE stats IS NOT NULL AND stats::text != '{}' AND (stats->>'_no_stats') IS NULL) as with_stats
        FROM matches
    """)
    result = await session.execute(counts_query)
    row = result.fetchone()

    # Teams counts
    teams_query = text("""
        SELECT
            COUNT(*) as total,
            COUNT(*) FILTER (WHERE team_type = 'club') as clubs,
            COUNT(*) FILTER (WHERE team_type = 'national') as national
        FROM teams
    """)
    teams_result = await session.execute(teams_query)
    teams_row = teams_result.fetchone()

    # Predictions count
    pred_query = text("SELECT COUNT(*) as total FROM predictions")
    pred_result = await session.execute(pred_query)
    pred_row = pred_result.fetchone()

    # TITAN tier1 coverage for 25/26 (fail-soft)
    titan_tier1_pct = None
    try:
        titan_query = text("""
            SELECT
                ROUND(100.0 * COUNT(*) FILTER (WHERE tier1_complete = true) / NULLIF(COUNT(*), 0), 1) as tier1_pct
            FROM titan.feature_matrix
            WHERE kickoff_utc >= '2025-08-01'
        """)
        titan_result = await session.execute(titan_query)
        titan_row = titan_result.fetchone()
        titan_tier1_pct = float(titan_row.tier1_pct) if titan_row and titan_row.tier1_pct else None
    except Exception as e:
        logger.warning(f"TITAN query failed (fail-soft): {e}")

    # Top leagues by matches
    top_query = text("""
        SELECT league_id, COUNT(*) as matches
        FROM matches
        GROUP BY league_id
        ORDER BY matches DESC
        LIMIT 5
    """)
    top_result = await session.execute(top_query)
    top_rows = top_result.fetchall()

    top_leagues = []
    for r in top_rows:
        info = get_league_info_sync(r.league_id)
        top_leagues.append({
            "league_id": r.league_id,
            "name": info["name"],
            "matches": r.matches,
        })

    return {
        "counts": {
            "leagues_total": leagues_row.total,
            "leagues_active": leagues_row.active,  # is_active=true (product decision)
            "leagues_seed": leagues_row.seed,
            "leagues_observed": leagues_row.observed,
            "leagues_in_matches_30d": row.active_30d,
            "teams_total": teams_row.total,
            "teams_clubs": teams_row.clubs,
            "teams_national": teams_row.national,
            "matches_total": row.total_matches,
            "matches_25_26": row.matches_25_26,
            "predictions_total": pred_row.total,
        },
        "coverage_summary": {
            "matches_with_stats_pct": round(100.0 * row.with_stats / row.total_matches, 1) if row.total_matches else 0,
            "titan_tier1_25_26_pct": titan_tier1_pct,
        },
        "top_leagues_by_matches": top_leagues,
    }


# =============================================================================
# Leagues List
# =============================================================================

async def build_leagues_list(session: AsyncSession) -> dict:
    """Build leagues list from admin_leagues (DB-first)."""

    # Load league cache (DB-first)
    await _load_league_cache(session)

    # Get all observed leagues from matches
    observed_query = text("""
        SELECT
            league_id,
            COUNT(*) as total_matches,
            COUNT(*) FILTER (WHERE status IN ('FT', 'AET', 'PEN')) as finished,
            COUNT(*) FILTER (WHERE stats IS NOT NULL AND stats::text != '{}' AND (stats->>'_no_stats') IS NULL) as with_stats,
            COUNT(*) FILTER (WHERE date >= '2025-08-01') as matches_25_26,
            COUNT(DISTINCT home_team_id) as unique_teams,
            MIN(season) as first_season,
            MAX(season) as last_season,
            MAX(date) as last_match
        FROM matches
        GROUP BY league_id
        ORDER BY total_matches DESC
    """)
    result = await session.execute(observed_query)
    observed_rows = result.fetchall()

    observed_ids = {r.league_id for r in observed_rows}
    observed_data = {r.league_id: r for r in observed_rows}

    # Get TITAN coverage (fail-soft)
    titan_data = {}
    try:
        titan_query = text("""
            SELECT
                competition_id as league_id,
                COUNT(*) as total,
                COUNT(*) FILTER (WHERE tier1_complete = true) as tier1,
                COUNT(*) FILTER (WHERE tier1b_complete = true) as tier1b
            FROM titan.feature_matrix
            WHERE kickoff_utc >= '2025-08-01'
            GROUP BY competition_id
        """)
        titan_result = await session.execute(titan_query)
        for r in titan_result.fetchall():
            titan_data[r.league_id] = {
                "total": r.total,
                "tier1": r.tier1,
                "tier1b": r.tier1b,
                "tier1_pct": round(100.0 * r.tier1 / r.total, 1) if r.total else 0,
                "tier1b_pct": round(100.0 * r.tier1b / r.total, 1) if r.total else 0,
            }
    except Exception as e:
        logger.warning(f"TITAN query failed (fail-soft): {e}")

    # Get paired league groups
    groups_query = text("""
        SELECT group_id, group_key, name as group_name, country as group_country
        FROM admin_league_groups
    """)
    groups_result = await session.execute(groups_query)
    groups_data = {r.group_id: {"key": r.group_key, "name": r.group_name, "country": r.group_country}
                   for r in groups_result.fetchall()}

    # Build leagues list from admin_leagues (DB-first)
    leagues = []
    all_ids = set(_league_cache.keys()) | observed_ids

    for league_id in sorted(all_ids):
        db_entry = _league_cache.get(league_id)
        obs = observed_data.get(league_id)
        titan = titan_data.get(league_id)

        if db_entry:
            league_entry = {
                "league_id": league_id,
                "name": db_entry["effective_name"],
                "display_name": db_entry["display_name"],
                "logo_url": db_entry["logo_url"],
                "wikipedia_url": db_entry["wikipedia_url"],
                "country": db_entry["country"],
                "kind": db_entry["kind"],
                "is_active": db_entry["is_active"],
                "configured": db_entry["configured"],
                "source": db_entry["source"],
                "priority": db_entry["priority"],
                "match_type": db_entry["match_type"],
                "match_weight": db_entry["match_weight"],
                "season_start_month": db_entry["season_start_month"],
                "observed": league_id in observed_ids,
            }
            # Add group info if paired (with paired_handling from rules_json)
            if db_entry["group_id"] and db_entry["group_id"] in groups_data:
                group_info = groups_data[db_entry["group_id"]].copy()
                rules = db_entry.get("rules_json") or {}
                group_info["paired_handling"] = rules.get("paired_handling", "grouped")
                league_entry["group"] = group_info
        else:
            # League in matches but not in admin_leagues (should not happen after sync)
            league_entry = {
                "league_id": league_id,
                "name": f"League {league_id}",
                "country": None,
                "kind": "league",
                "is_active": False,
                "configured": False,
                "source": "unknown",
                "priority": None,
                "match_type": None,
                "match_weight": None,
                "observed": True,
            }

        # Add stats if observed
        if obs:
            league_entry["stats"] = {
                "total_matches": obs.total_matches,
                "finished_matches": obs.finished,
                "matches_25_26": obs.matches_25_26,
                "with_stats_pct": round(100.0 * obs.with_stats / obs.total_matches, 1) if obs.total_matches else 0,
                "unique_teams": obs.unique_teams,
                "seasons_range": [obs.first_season, obs.last_season],
                "last_match": obs.last_match.isoformat() if obs.last_match else None,
            }

        # Add TITAN if available
        if titan:
            league_entry["titan"] = titan

        leagues.append(league_entry)

    # Sort: is_active first, then configured, then by matches
    leagues.sort(key=lambda x: (
        not x["is_active"],  # active first
        not x["configured"],  # then configured
        -(x.get("stats", {}).get("total_matches", 0))  # then by matches desc
    ))

    # Count by source
    seed_count = sum(1 for e in _league_cache.values() if e["source"] == "seed")
    observed_count = sum(1 for e in _league_cache.values() if e["source"] == "observed")
    active_count = sum(1 for e in _league_cache.values() if e["is_active"])

    # Unmapped = in matches but not in admin_leagues (should be 0 after sync)
    unmapped = [lid for lid in observed_ids if lid not in _league_cache]

    return {
        "leagues": leagues,
        "totals": {
            "total_in_db": len(_league_cache),
            "active": active_count,
            "seed": seed_count,
            "observed": observed_count,
            "in_matches": len(observed_ids),
            "with_titan_data": len(titan_data),
        },
        "unmapped_in_matches": sorted(unmapped),
        "groups": list(groups_data.values()),
    }


# =============================================================================
# League Detail
# =============================================================================

async def build_league_detail(session: AsyncSession, league_id: int) -> Optional[dict]:
    """Build detail for a specific league."""

    # Load league cache (DB-first)
    await _load_league_cache(session)

    db_entry = _league_cache.get(league_id)

    # Check if league has any matches
    check_query = text("SELECT COUNT(*) as cnt FROM matches WHERE league_id = :lid")
    check_result = await session.execute(check_query, {"lid": league_id})
    if check_result.fetchone().cnt == 0 and not db_entry:
        return None  # League not found

    # Get group info if paired (with paired_handling from rules_json)
    group_info = None
    if db_entry and db_entry.get("group_id"):
        group_query = text("""
            SELECT group_key, name, country
            FROM admin_league_groups
            WHERE group_id = :gid
        """)
        group_result = await session.execute(group_query, {"gid": db_entry["group_id"]})
        gr = group_result.fetchone()
        if gr:
            rules = db_entry.get("rules_json") or {}
            group_info = {
                "key": gr.group_key,
                "name": gr.name,
                "country": gr.country,
                "paired_handling": rules.get("paired_handling", "grouped")
            }

    # Stats by season
    season_query = text("""
        SELECT
            season,
            COUNT(*) as total_matches,
            COUNT(*) FILTER (WHERE status IN ('FT', 'AET', 'PEN')) as finished,
            COUNT(*) FILTER (WHERE stats IS NOT NULL AND stats::text != '{}' AND (stats->>'_no_stats') IS NULL) as with_stats,
            COUNT(*) FILTER (WHERE odds_home IS NOT NULL) as with_odds
        FROM matches
        WHERE league_id = :lid
        GROUP BY season
        ORDER BY season DESC
        LIMIT 10
    """)
    season_result = await session.execute(season_query, {"lid": league_id})

    stats_by_season = []
    for r in season_result.fetchall():
        stats_by_season.append({
            "season": r.season,
            "total_matches": r.total_matches,
            "finished": r.finished,
            "with_stats_pct": round(100.0 * r.with_stats / r.total_matches, 1) if r.total_matches else 0,
            "with_odds_pct": round(100.0 * r.with_odds / r.total_matches, 1) if r.total_matches else 0,
        })

    # Teams in league
    teams_query = text("""
        SELECT
            t.id as team_id,
            t.external_id,
            t.name,
            COUNT(m.id) as matches_in_league
        FROM teams t
        JOIN matches m ON t.id = m.home_team_id OR t.id = m.away_team_id
        WHERE m.league_id = :lid
        GROUP BY t.id
        ORDER BY matches_in_league DESC
        LIMIT 30
    """)
    teams_result = await session.execute(teams_query, {"lid": league_id})

    teams = []
    for r in teams_result.fetchall():
        teams.append({
            "team_id": r.team_id,
            "external_id": r.external_id,
            "name": r.name,
            "matches_in_league": r.matches_in_league,
        })

    # TITAN coverage (fail-soft)
    titan_coverage = {"status": "unavailable"}
    try:
        titan_query = text("""
            SELECT
                COUNT(*) as total,
                COUNT(*) FILTER (WHERE tier1_complete = true) as tier1,
                COUNT(*) FILTER (WHERE tier1b_complete = true) as tier1b,
                COUNT(*) FILTER (WHERE tier1c_complete = true) as tier1c,
                COUNT(*) FILTER (WHERE tier1d_complete = true) as tier1d
            FROM titan.feature_matrix
            WHERE competition_id = :lid
        """)
        titan_result = await session.execute(titan_query, {"lid": league_id})
        tr = titan_result.fetchone()
        if tr and tr.total > 0:
            titan_coverage = {
                "total": tr.total,
                "tier1": tr.tier1,
                "tier1b": tr.tier1b,
                "tier1c": tr.tier1c,
                "tier1d": tr.tier1d,
            }
    except Exception as e:
        logger.warning(f"TITAN query failed for league {league_id}: {e}")

    # Recent matches
    recent_query = text("""
        SELECT
            m.id as match_id,
            m.date,
            ht.name as home,
            at.name as away,
            m.status,
            CASE
                WHEN m.status NOT IN ('FT', 'AET', 'PEN') THEN null
                WHEN m.stats IS NOT NULL AND m.stats::text != '{}' AND (m.stats->>'_no_stats') IS NULL THEN true
                ELSE false
            END as has_stats,
            CASE WHEN p.id IS NOT NULL THEN true ELSE false END as has_prediction
        FROM matches m
        JOIN teams ht ON m.home_team_id = ht.id
        JOIN teams at ON m.away_team_id = at.id
        LEFT JOIN predictions p ON m.id = p.match_id
        WHERE m.league_id = :lid
        ORDER BY m.date DESC
        LIMIT 10
    """)
    recent_result = await session.execute(recent_query, {"lid": league_id})

    recent_matches = []
    for r in recent_result.fetchall():
        recent_matches.append({
            "match_id": r.match_id,
            "date": r.date.isoformat() if r.date else None,
            "home": r.home,
            "away": r.away,
            "status": r.status,
            "has_stats": r.has_stats,
            "has_prediction": r.has_prediction,
        })

    league_info = {
        "league_id": league_id,
        "observed": len(stats_by_season) > 0,
    }

    if db_entry:
        league_info.update({
            "name": db_entry["effective_name"],
            "display_name": db_entry["display_name"],
            "logo_url": db_entry["logo_url"],
            "wikipedia_url": db_entry["wikipedia_url"],
            "country": db_entry["country"],
            "kind": db_entry["kind"],
            "is_active": db_entry["is_active"],
            "configured": db_entry["configured"],
            "source": db_entry["source"],
            "priority": db_entry["priority"],
            "match_type": db_entry["match_type"],
            "match_weight": db_entry["match_weight"],
            "season_start_month": db_entry["season_start_month"],
        })
        if group_info:
            league_info["group"] = group_info
    else:
        league_info.update({
            "name": f"League {league_id}",
            "country": None,
            "kind": "league",
            "is_active": False,
            "configured": False,
            "source": "unknown",
            "priority": None,
            "match_type": None,
            "match_weight": None,
        })

    return {
        "league": league_info,
        "stats_by_season": stats_by_season,
        "teams": teams,
        "titan_coverage": titan_coverage,
        "recent_matches": recent_matches,
    }


# =============================================================================
# Teams List
# =============================================================================

async def build_teams_list(
    session: AsyncSession,
    team_type: Optional[str] = None,
    country: Optional[str] = None,
    search: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
) -> dict:
    """Build paginated teams list with optional filters.

    Args:
        session: Database session
        team_type: Filter by team type (club, national)
        country: Filter by country
        search: Search by team name (case-insensitive)
        limit: Max results (1-500)
        offset: Pagination offset

    Note:
        When search is provided, uses a lightweight query (no JOINs/stats)
        optimized for typeahead performance. Stats are loaded on-demand
        when opening TeamDrawer.
    """

    # Validate params
    limit = min(max(limit, 1), 500)
    offset = max(offset, 0)

    # Build WHERE clause
    where_parts = []
    params = {"limit": limit, "offset": offset}

    if team_type and team_type != "all":
        where_parts.append("t.team_type = :team_type")
        params["team_type"] = team_type

    if country:
        where_parts.append("t.country = :country")
        params["country"] = country

    is_search = search and search.strip()
    if is_search:
        # Case-insensitive search by name
        where_parts.append("t.name ILIKE :search")
        params["search"] = f"%{search.strip()}%"

    where_clause = "WHERE " + " AND ".join(where_parts) if where_parts else ""

    # Count total
    count_query = text(f"""
        SELECT COUNT(*) as total
        FROM teams t
        {where_clause}
    """)
    count_result = await session.execute(count_query, params)
    total = count_result.fetchone().total

    # FAST PATH: Lightweight query for search (minimal JOINs for display_name)
    # Optimized for typeahead dropdown performance
    if is_search:
        teams_query = text(f"""
            SELECT
                t.id as team_id,
                t.external_id,
                t.name,
                t.country,
                t.team_type,
                t.logo_url,
                COALESCE(o.short_name, twe.short_name) as short_name
            FROM teams t
            LEFT JOIN team_wikidata_enrichment twe ON t.id = twe.team_id
            LEFT JOIN team_enrichment_overrides o ON t.id = o.team_id
            {where_clause}
            ORDER BY t.name ASC
            LIMIT :limit OFFSET :offset
        """)
        teams_result = await session.execute(teams_query, params)

        teams = []
        for r in teams_result.fetchall():
            # display_name: COALESCE(override.short_name, wikidata.short_name, teams.name)
            display_name = r.short_name if r.short_name else r.name
            teams.append({
                "team_id": r.team_id,
                "external_id": r.external_id,
                "name": r.name,
                "display_name": display_name,  # ADB P0: short_name cascade for UI
                "country": r.country,
                "team_type": r.team_type,
                "logo_url": r.logo_url,
                "stats": {
                    "total_matches": 0,
                    "matches_25_26": 0,
                    "leagues_played": 0,
                },
            })

        return {
            "teams": teams,
            "pagination": {
                "total": total,
                "limit": limit,
                "offset": offset,
                "has_more": offset + limit < total,
            },
            "filters_applied": {
                "type": team_type or "all",
                "country": country,
                "search": search.strip() if search else None,
            },
        }

    # FULL PATH: Query with stats (for browsing, not search)
    teams_query = text(f"""
        SELECT
            t.id as team_id,
            t.external_id,
            t.name,
            t.country,
            t.team_type,
            t.logo_url,
            COUNT(m.id) as total_matches,
            COUNT(m.id) FILTER (WHERE m.date >= '2025-08-01') as matches_25_26,
            COUNT(DISTINCT m.league_id) as leagues_played
        FROM teams t
        LEFT JOIN matches m ON t.id = m.home_team_id OR t.id = m.away_team_id
        {where_clause}
        GROUP BY t.id
        ORDER BY total_matches DESC
        LIMIT :limit OFFSET :offset
    """)
    teams_result = await session.execute(teams_query, params)

    teams = []
    for r in teams_result.fetchall():
        teams.append({
            "team_id": r.team_id,
            "external_id": r.external_id,
            "name": r.name,
            "country": r.country,
            "team_type": r.team_type,
            "logo_url": r.logo_url,
            "stats": {
                "total_matches": r.total_matches,
                "matches_25_26": r.matches_25_26,
                "leagues_played": r.leagues_played,
            },
        })

    return {
        "teams": teams,
        "pagination": {
            "total": total,
            "limit": limit,
            "offset": offset,
            "has_more": offset + limit < total,
        },
        "filters_applied": {
            "type": team_type or "all",
            "country": country,
        },
    }


# =============================================================================
# Team Detail
# =============================================================================


def _get_source_badge(enrichment_source: str) -> dict:
    """
    Generate badge indicator for data source (per Kimi recommendation).

    Returns dict with emoji, label, and tooltip for dashboard display.
    """
    badges = {
        "wikidata": {
            "emoji": "check",
            "label": "Wikidata",
            "tooltip": "Data from Wikidata SPARQL API",
            "color": "green",
        },
        "wikipedia": {
            "emoji": "warning",
            "label": "Wikipedia",
            "tooltip": "Fallback: Data from Wikipedia REST API (Wikidata incomplete)",
            "color": "yellow",
        },
        "wikidata+wikipedia": {
            "emoji": "check",
            "label": "Wikidata+WP",
            "tooltip": "Primary from Wikidata, some fields from Wikipedia fallback",
            "color": "green",
        },
        "wikidata+website": {
            "emoji": "check",
            "label": "Wikidata+Web",
            "tooltip": "Wikidata enrichment + social handles from official website",
            "color": "green",
        },
        "wikipedia+website": {
            "emoji": "warning",
            "label": "Wikipedia+Web",
            "tooltip": "Wikipedia fallback + social handles from official website",
            "color": "yellow",
        },
        "wikidata+wikipedia+website": {
            "emoji": "check",
            "label": "WD+WP+Web",
            "tooltip": "Wikidata + Wikipedia fallback + social handles from official website",
            "color": "green",
        },
        "website": {
            "emoji": "globe",
            "label": "Website",
            "tooltip": "Data extracted from official team website only",
            "color": "blue",
        },
    }

    # Handle override sources (format: "override:manual", "override:transfermarkt")
    if enrichment_source and enrichment_source.startswith("override:"):
        override_type = enrichment_source.split(":")[1] if ":" in enrichment_source else "manual"
        return {
            "emoji": "edit",
            "label": f"Override ({override_type})",
            "tooltip": f"Manual correction: {override_type}",
            "color": "blue",
        }

    return badges.get(enrichment_source, {
        "emoji": "question",
        "label": enrichment_source or "Unknown",
        "tooltip": "Unknown data source",
        "color": "gray",
    })


async def build_team_detail(session: AsyncSession, team_id: int) -> Optional[dict]:
    """Build detail for a specific team."""

    # Load league cache (DB-first)
    await _load_league_cache(session)

    # Get team info (attempt v2 with optional wiki fields; fallback if migration not applied)
    wiki_supported = True
    team_row = None
    team_query_v2 = text("""
        SELECT
            id, external_id, name, country, team_type, logo_url,
            kit_supplier, kit_supplier_since,
            wiki_url, wikidata_id, wiki_title, wiki_lang,
            wiki_url_cached, wiki_source, wiki_confidence, wiki_matched_at
        FROM teams
        WHERE id = :tid
    """)
    try:
        team_result = await session.execute(team_query_v2, {"tid": team_id})
        team_row = team_result.fetchone()
    except ProgrammingError:
        wiki_supported = False
        team_query_v1 = text("""
            SELECT id, external_id, name, country, team_type, logo_url,
                   kit_supplier, kit_supplier_since
            FROM teams
            WHERE id = :tid
        """)
        team_result = await session.execute(team_query_v1, {"tid": team_id})
        team_row = team_result.fetchone()

    if not team_row:
        return None

    # Wikidata enrichment with manual overrides (optional; fail-soft if migration not applied)
    # P0 ABE: Use COALESCE for override > wikidata cascade
    enrichment_supported = True
    enrichment_row = None
    override_row = None
    enrichment_query = text("""
        SELECT
            twe.wikidata_id,
            twe.fetched_at,
            -- Merged values (override > wikidata)
            COALESCE(o.stadium_name, twe.stadium_name) AS stadium_name,
            COALESCE(o.stadium_wikidata_id, twe.stadium_wikidata_id) AS stadium_wikidata_id,
            COALESCE(o.stadium_capacity, twe.stadium_capacity) AS stadium_capacity,
            twe.stadium_altitude_m,
            COALESCE(o.admin_location_label, twe.admin_location_label) AS admin_location_label,
            COALESCE(o.lat, twe.lat) AS lat,
            COALESCE(o.lon, twe.lon) AS lon,
            COALESCE(o.full_name, twe.full_name) AS full_name,
            COALESCE(o.short_name, twe.short_name) AS short_name,
            COALESCE(o.website, twe.website) AS website,
            COALESCE(o.twitter_handle, twe.social_handles->>'twitter') AS twitter,
            COALESCE(o.instagram_handle, twe.social_handles->>'instagram') AS instagram,
            -- Enrichment source: override:{source} if any override field is set, else wikidata source
            CASE
                WHEN o.team_id IS NOT NULL AND (
                    o.full_name IS NOT NULL OR o.short_name IS NOT NULL OR
                    o.stadium_name IS NOT NULL OR o.stadium_capacity IS NOT NULL OR
                    o.stadium_wikidata_id IS NOT NULL OR
                    o.website IS NOT NULL OR o.twitter_handle IS NOT NULL OR
                    o.instagram_handle IS NOT NULL OR o.lat IS NOT NULL OR o.lon IS NOT NULL OR
                    o.admin_location_label IS NOT NULL
                ) THEN 'override:' || COALESCE(o.source, 'manual')
                ELSE COALESCE(twe.enrichment_source, 'wikidata')
            END AS enrichment_source,
            -- P0 ABE: has_override = at least one override field is non-null
            CASE
                WHEN o.team_id IS NOT NULL AND (
                    o.full_name IS NOT NULL OR o.short_name IS NOT NULL OR
                    o.stadium_name IS NOT NULL OR o.stadium_capacity IS NOT NULL OR
                    o.stadium_wikidata_id IS NOT NULL OR
                    o.website IS NOT NULL OR o.twitter_handle IS NOT NULL OR
                    o.instagram_handle IS NOT NULL OR o.lat IS NOT NULL OR o.lon IS NOT NULL OR
                    o.admin_location_label IS NOT NULL
                ) THEN true
                ELSE false
            END AS has_override,
            -- Raw override values for edit form
            o.full_name AS override_full_name,
            o.short_name AS override_short_name,
            o.stadium_name AS override_stadium_name,
            o.stadium_capacity AS override_stadium_capacity,
            o.stadium_wikidata_id AS override_stadium_wikidata_id,
            o.admin_location_label AS override_city,
            o.website AS override_website,
            o.twitter_handle AS override_twitter,
            o.instagram_handle AS override_instagram,
            o.source AS override_source,
            o.notes AS override_notes,
            o.updated_at AS override_updated_at
        FROM team_wikidata_enrichment twe
        LEFT JOIN team_enrichment_overrides o ON twe.team_id = o.team_id
        WHERE twe.team_id = :tid
    """)
    try:
        enrichment_result = await session.execute(enrichment_query, {"tid": team_id})
        enrichment_row = enrichment_result.fetchone()
    except ProgrammingError:
        enrichment_supported = False

    # Leagues played
    leagues_query = text("""
        SELECT
            m.league_id,
            COUNT(*) as matches,
            MIN(m.season) as first_season,
            MAX(m.season) as last_season
        FROM matches m
        WHERE m.home_team_id = :tid OR m.away_team_id = :tid
        GROUP BY m.league_id
        ORDER BY matches DESC
    """)
    leagues_result = await session.execute(leagues_query, {"tid": team_id})

    leagues_played = []
    for r in leagues_result.fetchall():
        info = get_league_info_sync(r.league_id)
        leagues_played.append({
            "league_id": r.league_id,
            "name": info["name"],
            "matches": r.matches,
            "seasons_range": [r.first_season, r.last_season],
        })

    # Stats by season (wins/draws/losses)
    stats_query = text("""
        SELECT
            m.season,
            COUNT(*) as matches,
            COUNT(*) FILTER (WHERE
                (m.home_team_id = :tid AND m.home_goals > m.away_goals) OR
                (m.away_team_id = :tid AND m.away_goals > m.home_goals)
            ) as wins,
            COUNT(*) FILTER (WHERE m.home_goals = m.away_goals AND m.status IN ('FT', 'AET', 'PEN')) as draws,
            COUNT(*) FILTER (WHERE
                (m.home_team_id = :tid AND m.home_goals < m.away_goals) OR
                (m.away_team_id = :tid AND m.away_goals < m.home_goals)
            ) as losses
        FROM matches m
        WHERE (m.home_team_id = :tid OR m.away_team_id = :tid)
          AND m.status IN ('FT', 'AET', 'PEN')
        GROUP BY m.season
        ORDER BY m.season DESC
        LIMIT 5
    """)
    stats_result = await session.execute(stats_query, {"tid": team_id})

    stats_by_season = []
    for r in stats_result.fetchall():
        stats_by_season.append({
            "season": r.season,
            "matches": r.matches,
            "wins": r.wins,
            "draws": r.draws,
            "losses": r.losses,
        })

    # Recent matches
    recent_query = text("""
        SELECT
            m.id as match_id,
            m.date,
            m.league_id,
            CASE WHEN m.home_team_id = :tid THEN at.name ELSE ht.name END as opponent,
            CASE WHEN m.home_team_id = :tid THEN 'home' ELSE 'away' END as home_away,
            CONCAT(m.home_goals, '-', m.away_goals) as result,
            m.status
        FROM matches m
        JOIN teams ht ON m.home_team_id = ht.id
        JOIN teams at ON m.away_team_id = at.id
        WHERE m.home_team_id = :tid OR m.away_team_id = :tid
        ORDER BY m.date DESC
        LIMIT 10
    """)
    recent_result = await session.execute(recent_query, {"tid": team_id})

    recent_matches = []
    for r in recent_result.fetchall():
        info = get_league_info_sync(r.league_id)
        recent_matches.append({
            "match_id": r.match_id,
            "date": r.date.isoformat() if r.date else None,
            "opponent": r.opponent,
            "home_away": r.home_away,
            "result": r.result if r.status in ("FT", "AET", "PEN") else None,
            "league_name": info["name"],
        })

    # Predictions stats - P0 safe version (no accuracy without ground truth table)
    predictions_query = text("""
        SELECT
            COUNT(*) as total,
            COUNT(*) FILTER (WHERE p.created_at >= NOW() - INTERVAL '30 days') as last_30d,
            COUNT(*) FILTER (WHERE p.frozen_confidence_tier = 'gold') as gold,
            COUNT(*) FILTER (WHERE p.frozen_confidence_tier = 'silver') as silver,
            COUNT(*) FILTER (WHERE p.frozen_confidence_tier = 'copper') as copper
        FROM predictions p
        JOIN matches m ON p.match_id = m.id
        WHERE m.home_team_id = :tid OR m.away_team_id = :tid
    """)
    pred_result = await session.execute(predictions_query, {"tid": team_id})
    pr = pred_result.fetchone()

    predictions_stats = {
        "total": pr.total,
        "last_30d": pr.last_30d,
        "confidence_tier_distribution": {
            "gold": pr.gold,
            "silver": pr.silver,
            "copper": pr.copper,
        },
        "accuracy": {
            "status": "not_available",
            "note": "Requires prediction_outcomes table for ground truth",
        },
    }

    # Compute display_name: COALESCE(override.short_name, wikidata.short_name, teams.name)
    # This provides a friendly short name for UI display
    display_name = team_row.name  # fallback
    if enrichment_row:
        short_name = getattr(enrichment_row, "short_name", None)
        if short_name:
            display_name = short_name

    payload = {
        "team": {
            "team_id": team_row.id,
            "external_id": team_row.external_id,
            "name": team_row.name,
            "display_name": display_name,  # ADB P0: short_name cascade for UI
            "country": team_row.country,
            "team_type": team_row.team_type,
            "logo_url": team_row.logo_url,
            "kit_supplier": getattr(team_row, "kit_supplier", None),
            "kit_supplier_since": getattr(team_row, "kit_supplier_since", None),
        },
        "leagues_played": leagues_played,
        "stats_by_season": stats_by_season,
        "recent_matches": recent_matches,
        "predictions_stats": predictions_stats,
    }

    if wiki_supported:
        wiki_matched_at = getattr(team_row, "wiki_matched_at", None)
        payload["team"]["wiki"] = {
            "wiki_url": getattr(team_row, "wiki_url", None),
            "wikidata_id": getattr(team_row, "wikidata_id", None),
            "wiki_title": getattr(team_row, "wiki_title", None),
            "wiki_lang": getattr(team_row, "wiki_lang", None),
            "wiki_url_cached": getattr(team_row, "wiki_url_cached", None),
            "wiki_source": getattr(team_row, "wiki_source", None),
            "wiki_confidence": getattr(team_row, "wiki_confidence", None),
            "wiki_matched_at": (wiki_matched_at.isoformat()) if wiki_matched_at else None,
        }

    if enrichment_supported:
        if enrichment_row:
            fetched_at = getattr(enrichment_row, "fetched_at", None)
            enrichment_source = getattr(enrichment_row, "enrichment_source", "wikidata")
            has_override = getattr(enrichment_row, "has_override", False)
            override_updated_at = getattr(enrichment_row, "override_updated_at", None)

            payload["wikidata_enrichment"] = {
                "wikidata_id": getattr(enrichment_row, "wikidata_id", None),
                "wikidata_updated_at": (fetched_at.isoformat()) if fetched_at else None,
                # Merged effective values (COALESCE override > wikidata)
                "stadium_name": getattr(enrichment_row, "stadium_name", None),
                "stadium_wikidata_id": getattr(enrichment_row, "stadium_wikidata_id", None),
                "stadium_capacity": getattr(enrichment_row, "stadium_capacity", None),
                "stadium_altitude_m": getattr(enrichment_row, "stadium_altitude_m", None),
                "city": getattr(enrichment_row, "admin_location_label", None),
                "lat": getattr(enrichment_row, "lat", None),
                "lon": getattr(enrichment_row, "lon", None),
                "full_name": getattr(enrichment_row, "full_name", None),
                "short_name": getattr(enrichment_row, "short_name", None),
                "website": getattr(enrichment_row, "website", None),
                "twitter": getattr(enrichment_row, "twitter", None),
                "instagram": getattr(enrichment_row, "instagram", None),
                "enrichment_source": enrichment_source,
                # Badge indicators for ADB
                "source_badge": _get_source_badge(enrichment_source),
                # P0 ABE: has_override = at least one override field is non-null
                "has_override": has_override,
                # Raw override values for edit form (what user has set)
                "override": {
                    "full_name": getattr(enrichment_row, "override_full_name", None),
                    "short_name": getattr(enrichment_row, "override_short_name", None),
                    "stadium_name": getattr(enrichment_row, "override_stadium_name", None),
                    "stadium_capacity": getattr(enrichment_row, "override_stadium_capacity", None),
                    "stadium_wikidata_id": getattr(enrichment_row, "override_stadium_wikidata_id", None),
                    "city": getattr(enrichment_row, "override_city", None),
                    "website": getattr(enrichment_row, "override_website", None),
                    "twitter": getattr(enrichment_row, "override_twitter", None),
                    "instagram": getattr(enrichment_row, "override_instagram", None),
                    "source": getattr(enrichment_row, "override_source", None),
                    "notes": getattr(enrichment_row, "override_notes", None),
                    "updated_at": (override_updated_at.isoformat()) if override_updated_at else None,
                } if has_override else None,
            }
        else:
            payload["wikidata_enrichment"] = None

    # Feature coverage (ATI P0 - fail-soft)
    payload["feature_coverage"] = await _build_feature_coverage(session, team_id)

    return payload


async def _build_feature_coverage(session: AsyncSession, team_id: int) -> Optional[dict]:
    """
    Build feature coverage data for a team (ATI P0).

    3 layers: kill-switch status, source coverage, TITAN tiers.
    Uses dynamic config from get_settings() for lookback/min thresholds.
    Fail-soft: returns None if titan schema doesn't exist or query fails.
    """
    try:
        from app.config import get_settings
        _cfg = get_settings()
        lookback_days = _cfg.KILLSWITCH_LOOKBACK_DAYS
        min_league_matches = _cfg.KILLSWITCH_MIN_LEAGUE_MATCHES
    except Exception:
        lookback_days = 90
        min_league_matches = 5

    try:
        # Query 1: Kill-switch status (ATI P0-1: JOIN admin_leagues, P0-2: dynamic config)
        ks_result = await session.execute(text("""
            SELECT COUNT(*) as ft_league_matches
            FROM (
                SELECT m.date
                FROM matches m
                JOIN admin_leagues al ON m.league_id = al.league_id AND al.kind = 'league'
                WHERE m.status = 'FT'
                  AND m.date >= NOW() - MAKE_INTERVAL(days => :lookback_days)
                  AND (m.home_team_id = :tid OR m.away_team_id = :tid)
            ) sub
        """), {"tid": team_id, "lookback_days": lookback_days})
        ft_count = int(ks_result.scalar() or 0)

        if ft_count >= min_league_matches + 3:
            ks_status = "ok"
        elif ft_count >= min_league_matches:
            ks_status = "warning"
        else:
            ks_status = "blocked"

        killswitch = {
            "ft_league_matches": ft_count,
            "lookback_days": lookback_days,
            "min_required": min_league_matches,
            "status": ks_status,
        }
    except Exception as e:
        logger.warning(f"[FEATURE-COVERAGE] kill-switch query failed for team {team_id}: {e}")
        killswitch = None

    try:
        # Query 2: TITAN coverage (ATI P0-3: league-only, P0-4: past matches only)
        cov_result = await session.execute(text("""
            SELECT
                COUNT(*) as total_matches,
                COUNT(*) FILTER (WHERE fm.odds_home_close IS NOT NULL) as with_odds,
                COUNT(*) FILTER (WHERE fm.xg_home_last5 IS NOT NULL OR fm.xg_away_last5 IS NOT NULL) as with_xg,
                COUNT(*) FILTER (WHERE fm.sofascore_home_formation IS NOT NULL OR fm.sofascore_away_formation IS NOT NULL) as with_lineup,
                COUNT(*) FILTER (WHERE fm.xi_home_def_count IS NOT NULL OR fm.xi_away_def_count IS NOT NULL) as with_xi_depth,
                COUNT(*) FILTER (WHERE fm.form_home_last5 IS NOT NULL OR fm.form_away_last5 IS NOT NULL) as with_form,
                COUNT(*) FILTER (WHERE fm.h2h_total_matches IS NOT NULL AND fm.h2h_total_matches > 0) as with_h2h,
                COUNT(*) FILTER (WHERE fm.tier1_complete) as tier1_ok,
                COUNT(*) FILTER (WHERE fm.tier1b_complete) as tier1b_ok,
                COUNT(*) FILTER (WHERE fm.tier1c_complete) as tier1c_ok,
                COUNT(*) FILTER (WHERE fm.tier1d_complete) as tier1d_ok,
                COUNT(*) FILTER (WHERE fm.tier2_complete) as tier2_ok,
                COUNT(*) FILTER (WHERE fm.tier3_complete) as tier3_ok
            FROM titan.feature_matrix fm
            JOIN admin_leagues al ON fm.competition_id = al.league_id AND al.kind = 'league'
            WHERE (fm.home_team_id = :tid OR fm.away_team_id = :tid)
              AND fm.kickoff_utc <= NOW()
              AND fm.kickoff_utc >= NOW() - MAKE_INTERVAL(days => :lookback_days)
        """), {"tid": team_id, "lookback_days": lookback_days})
        row = cov_result.fetchone()

        if not row or row.total_matches == 0:
            return {"killswitch": killswitch, "sources": None, "tiers": None}

        total = row.total_matches

        def _pct(count: int) -> float:
            return round(100.0 * count / total, 1) if total > 0 else 0.0

        sources = {
            "total_matches": total,
            "odds": {"count": row.with_odds, "pct": _pct(row.with_odds)},
            "xg": {"count": row.with_xg, "pct": _pct(row.with_xg)},
            "lineup": {"count": row.with_lineup, "pct": _pct(row.with_lineup)},
            "xi_depth": {"count": row.with_xi_depth, "pct": _pct(row.with_xi_depth)},
            "form": {"count": row.with_form, "pct": _pct(row.with_form)},
            "h2h": {"count": row.with_h2h, "pct": _pct(row.with_h2h)},
        }

        tiers = {
            "tier1": {"count": row.tier1_ok, "pct": _pct(row.tier1_ok)},
            "tier1b": {"count": row.tier1b_ok, "pct": _pct(row.tier1b_ok)},
            "tier1c": {"count": row.tier1c_ok, "pct": _pct(row.tier1c_ok)},
            "tier1d": {"count": row.tier1d_ok, "pct": _pct(row.tier1d_ok)},
            "tier2": {"count": row.tier2_ok, "pct": _pct(row.tier2_ok)},
            "tier3": {"count": row.tier3_ok, "pct": _pct(row.tier3_ok)},
        }

        return {"killswitch": killswitch, "sources": sources, "tiers": tiers}

    except Exception as e:
        logger.warning(f"[FEATURE-COVERAGE] TITAN query failed for team {team_id}: {e}")
        return {"killswitch": killswitch, "sources": None, "tiers": None} if killswitch else None


# =============================================================================
# P2B - Mutations
# =============================================================================

# Whitelist of fields allowed in PATCH
PATCH_ALLOWED_FIELDS = {
    "is_active", "country", "kind", "priority", "match_type", "match_weight",
    "display_order", "tags", "rules_json", "group_id", "name",
    "display_name", "logo_url", "wikipedia_url", "season_start_month",
}

# Valid values for enums
VALID_KINDS = {"league", "cup", "international", "friendly"}
VALID_PRIORITIES = {"high", "medium", "low", None}
VALID_MATCH_TYPES = {"official", "friendly", None}
VALID_CHANNELS = {"ios", "android", "web"}

# rules_json v1 schema validation
VALID_SEASON_MODELS = {"aug_jul", "calendar"}
VALID_PAIRED_HANDLING = {"grouped", "separate"}


class ValidationError(Exception):
    """Raised when patch validation fails."""
    pass


# =============================================================================
# P0 - Team Wiki mutation helpers
# =============================================================================

_WIKIDATA_ID_RE = re.compile(r"^Q\d+$")


def _normalize_wiki_url(value: Any) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValidationError("wiki_url must be a string or null")
    v = value.strip()
    return v or None


def _validate_wiki_url(value: Optional[str]) -> None:
    if value is None:
        return
    if len(value) > 2000:
        raise ValidationError("wiki_url too long")
    if not value.startswith("https://"):
        raise ValidationError("wiki_url must start with https://")
    if "m.wikipedia.org" in value:
        raise ValidationError("wiki_url must not be a mobile URL (m.wikipedia.org)")
    if "/Special:" in value:
        raise ValidationError("wiki_url must not be a Special: page")
    if "?" in value or "#" in value:
        raise ValidationError("wiki_url must not include query params or fragments")
    if ".wikipedia.org/wiki/" not in value:
        raise ValidationError("wiki_url must be a Wikipedia /wiki/ URL")


def _normalize_wikidata_id(value: Any) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValidationError("wikidata_id must be a string or null")
    v = value.strip().upper()
    return v or None


def _validate_wikidata_id(value: Optional[str]) -> None:
    if value is None:
        return
    if not _WIKIDATA_ID_RE.match(value):
        raise ValidationError("wikidata_id must match format Q<digits> (e.g., Q42)")


async def patch_team_wiki(
    session: AsyncSession,
    team_id: int,
    patch_data: dict,
    actor: str = "dashboard",
) -> dict:
    """
    PATCH wiki fields for a team (P0).

    Allowed fields:
      - wiki_url (nullable string)
      - wikidata_id (nullable string)

    PATCH semantics: Only fields present in the payload are updated.
    Absent fields retain their current values.

    Returns:
      { team_id, updated_fields, wiki }
    """
    if not isinstance(patch_data, dict):
        raise ValidationError("Invalid JSON body")

    # Check which fields are present in the payload (PATCH semantics)
    has_wiki_url = "wiki_url" in patch_data
    has_wikidata_id = "wikidata_id" in patch_data

    if not has_wiki_url and not has_wikidata_id:
        raise ValidationError("No valid fields to update (wiki_url or wikidata_id)")

    # Normalize and validate only fields that are present
    wiki_url = _normalize_wiki_url(patch_data["wiki_url"]) if has_wiki_url else None
    wikidata_id = _normalize_wikidata_id(patch_data["wikidata_id"]) if has_wikidata_id else None

    if has_wiki_url:
        _validate_wiki_url(wiki_url)
    if has_wikidata_id:
        _validate_wikidata_id(wikidata_id)

    # Ensure migration applied + team exists
    try:
        current = await session.execute(
            text("""
                SELECT id, wiki_url, wikidata_id
                FROM teams
                WHERE id = :tid
            """),
            {"tid": team_id},
        )
    except ProgrammingError as e:
        raise NotImplementedError("Teams wiki fields not available (migration missing)") from e

    row = current.fetchone()
    if not row:
        raise ValueError(f"Team {team_id} not found")

    # Determine which fields actually changed (only check fields present in payload)
    updated_fields: List[str] = []
    if has_wiki_url and (row.wiki_url or None) != (wiki_url or None):
        updated_fields.append("wiki_url")
    if has_wikidata_id and (row.wikidata_id or None) != (wikidata_id or None):
        updated_fields.append("wikidata_id")

    if not updated_fields:
        return {
            "team_id": team_id,
            "updated_fields": [],
            "wiki": {
                "wiki_url": row.wiki_url,
                "wikidata_id": row.wikidata_id,
            },
        }

    # Compute final values: use new value if present, else keep existing
    final_wiki_url = wiki_url if has_wiki_url else row.wiki_url
    final_wikidata_id = wikidata_id if has_wikidata_id else row.wikidata_id

    # If both are now cleared, clear all derived fields too
    clear_all = final_wiki_url is None and final_wikidata_id is None
    now = datetime.now(timezone.utc)

    update_sql = text("""
        UPDATE teams
        SET
            wiki_url = :wiki_url,
            wikidata_id = :wikidata_id,
            wiki_source = :wiki_source,
            wiki_matched_at = :wiki_matched_at,
            wiki_title = :wiki_title,
            wiki_lang = :wiki_lang,
            wiki_url_cached = :wiki_url_cached,
            wiki_confidence = :wiki_confidence
        WHERE id = :tid
        RETURNING
            wiki_url, wikidata_id, wiki_title, wiki_lang,
            wiki_url_cached, wiki_source, wiki_confidence, wiki_matched_at
    """)

    params = {
        "tid": team_id,
        "wiki_url": final_wiki_url,
        "wikidata_id": final_wikidata_id,
        "wiki_source": None if clear_all else "manual",
        "wiki_matched_at": None if clear_all else now,
        # Derived fields not computed yet (P0); clear on change
        "wiki_title": None,
        "wiki_lang": None,
        "wiki_url_cached": None,
        "wiki_confidence": None,
    }

    result = await session.execute(update_sql, params)
    updated = result.fetchone()
    await session.commit()

    wiki_matched_at = updated.wiki_matched_at
    return {
        "team_id": team_id,
        "updated_fields": updated_fields,
        "wiki": {
            "wiki_url": updated.wiki_url,
            "wikidata_id": updated.wikidata_id,
            "wiki_title": updated.wiki_title,
            "wiki_lang": updated.wiki_lang,
            "wiki_url_cached": updated.wiki_url_cached,
            "wiki_source": updated.wiki_source,
            "wiki_confidence": updated.wiki_confidence,
            "wiki_matched_at": (wiki_matched_at.isoformat()) if wiki_matched_at else None,
        },
    }


def _validate_rules_json_v1(rules: dict) -> None:
    """
    Validate rules_json against v1 schema.
    Empty dict is valid. Unknown fields are ignored (forward compatibility).
    Raises ValidationError on type/value violations.
    """
    if not rules:
        return  # Empty is valid

    # team_count_expected
    if "team_count_expected" in rules:
        tc = rules["team_count_expected"]
        if not isinstance(tc, int) or tc <= 0:
            raise ValidationError("rules_json.team_count_expected must be a positive integer")

    # season_model
    if "season_model" in rules:
        sm = rules["season_model"]
        if sm not in VALID_SEASON_MODELS:
            raise ValidationError(f"rules_json.season_model must be one of: {sorted(VALID_SEASON_MODELS)}")

    # promotion_relegation
    if "promotion_relegation" in rules:
        pr = rules["promotion_relegation"]
        if not isinstance(pr, dict):
            raise ValidationError("rules_json.promotion_relegation must be an object")
        if "promote" in pr and (not isinstance(pr["promote"], int) or pr["promote"] < 0):
            raise ValidationError("rules_json.promotion_relegation.promote must be >= 0")
        if "relegate" in pr and (not isinstance(pr["relegate"], int) or pr["relegate"] < 0):
            raise ValidationError("rules_json.promotion_relegation.relegate must be >= 0")
        if "playoffs" in pr and not isinstance(pr["playoffs"], bool):
            raise ValidationError("rules_json.promotion_relegation.playoffs must be boolean")

    # qualification
    if "qualification" in rules:
        qual = rules["qualification"]
        if not isinstance(qual, dict):
            raise ValidationError("rules_json.qualification must be an object")
        if "targets" in qual:
            if not isinstance(qual["targets"], list):
                raise ValidationError("rules_json.qualification.targets must be an array")
            for i, t in enumerate(qual["targets"]):
                if not isinstance(t, dict):
                    raise ValidationError(f"rules_json.qualification.targets[{i}] must be an object")
                if "target_league_id" in t and not isinstance(t["target_league_id"], int):
                    raise ValidationError(f"rules_json.qualification.targets[{i}].target_league_id must be int")
                if "slots" in t and (not isinstance(t["slots"], int) or t["slots"] < 0):
                    raise ValidationError(f"rules_json.qualification.targets[{i}].slots must be >= 0")

    # paired_handling
    if "paired_handling" in rules:
        ph = rules["paired_handling"]
        if ph not in VALID_PAIRED_HANDLING:
            raise ValidationError(f"rules_json.paired_handling must be one of: {sorted(VALID_PAIRED_HANDLING)}")


def _validate_patch(patch: dict) -> dict:
    """
    Validate and sanitize patch data.
    Returns sanitized patch dict.
    Raises ValidationError on invalid data.
    """
    sanitized = {}

    for key, value in patch.items():
        if key not in PATCH_ALLOWED_FIELDS:
            continue  # Silently ignore unknown fields

        if key == "kind":
            if value not in VALID_KINDS:
                raise ValidationError(f"kind must be one of: {VALID_KINDS}")
            sanitized[key] = value

        elif key == "priority":
            if value is not None and value not in VALID_PRIORITIES:
                raise ValidationError(f"priority must be one of: {VALID_PRIORITIES}")
            sanitized[key] = value

        elif key == "match_type":
            if value is not None and value not in VALID_MATCH_TYPES:
                raise ValidationError(f"match_type must be one of: {VALID_MATCH_TYPES}")
            sanitized[key] = value

        elif key == "match_weight":
            if value is not None:
                try:
                    weight = float(value)
                    if weight < 0 or weight > 1:
                        raise ValidationError("match_weight must be between 0 and 1")
                    sanitized[key] = weight
                except (TypeError, ValueError):
                    raise ValidationError("match_weight must be a number")
            else:
                sanitized[key] = None

        elif key == "is_active":
            if not isinstance(value, bool):
                raise ValidationError("is_active must be a boolean")
            sanitized[key] = value

        elif key == "display_order":
            if value is not None:
                try:
                    sanitized[key] = int(value)
                except (TypeError, ValueError):
                    raise ValidationError("display_order must be an integer")
            else:
                sanitized[key] = None

        elif key == "group_id":
            if value is not None:
                try:
                    sanitized[key] = int(value)
                except (TypeError, ValueError):
                    raise ValidationError("group_id must be an integer")
            else:
                sanitized[key] = None

        elif key == "tags":
            if not isinstance(value, dict):
                raise ValidationError("tags must be a JSON object")
            # Validate channels if present
            if "channels" in value:
                channels = value["channels"]
                if not isinstance(channels, list):
                    raise ValidationError("tags.channels must be an array")
                invalid = set(channels) - VALID_CHANNELS
                if invalid:
                    raise ValidationError(f"tags.channels contains invalid values: {invalid}")
            sanitized[key] = value

        elif key == "rules_json":
            if not isinstance(value, dict):
                raise ValidationError("rules_json must be a JSON object")
            _validate_rules_json_v1(value)
            sanitized[key] = value

        elif key == "name":
            if value is not None and not isinstance(value, str):
                raise ValidationError("name must be a string")
            if value is not None and len(value.strip()) == 0:
                raise ValidationError("name cannot be empty")
            sanitized[key] = value.strip() if value else None

        elif key == "country":
            if value is not None and not isinstance(value, str):
                raise ValidationError("country must be a string")
            sanitized[key] = value.strip() if value else None

        elif key == "season_start_month":
            if value is not None:
                try:
                    month = int(value)
                    if month < 1 or month > 12:
                        raise ValidationError("season_start_month must be between 1 and 12")
                    sanitized[key] = month
                except (TypeError, ValueError):
                    raise ValidationError("season_start_month must be an integer")
            else:
                sanitized[key] = None

        elif key in ("display_name", "logo_url", "wikipedia_url"):
            if value is not None and not isinstance(value, str):
                raise ValidationError(f"{key} must be a string")
            sanitized[key] = value.strip() if value else None

    return sanitized


async def _write_audit_log(
    session: AsyncSession,
    entity_type: str,
    entity_id: str,
    action: str,
    before_json: Optional[dict],
    after_json: Optional[dict],
    actor: str = "dashboard"
) -> int:
    """Write an audit log entry. Returns the audit log id."""
    result = await session.execute(
        text("""
            INSERT INTO admin_audit_log (entity_type, entity_id, action, actor, before_json, after_json)
            VALUES (:entity_type, :entity_id, :action, :actor, :before_json, :after_json)
            RETURNING id
        """),
        {
            "entity_type": entity_type,
            "entity_id": entity_id,
            "action": action,
            "actor": actor,
            "before_json": json.dumps(before_json) if before_json else None,
            "after_json": json.dumps(after_json) if after_json else None,
        }
    )
    row = result.fetchone()
    return row[0]


async def patch_league(
    session: AsyncSession,
    league_id: int,
    patch_data: dict,
    actor: str = "dashboard"
) -> dict:
    """
    Apply a patch to an admin_leagues row.

    Args:
        session: Database session
        league_id: The league to update
        patch_data: Dict of fields to update (whitelist enforced)
        actor: Who is making the change

    Returns:
        Dict with updated league and audit_id

    Raises:
        ValidationError: If patch data is invalid
        ValueError: If league not found
    """
    # Validate patch
    sanitized = _validate_patch(patch_data)

    if not sanitized:
        raise ValidationError("No valid fields to update")

    # Get current state
    current_query = text("""
        SELECT
            league_id, sport, name, display_name, logo_url, wikipedia_url,
            country, kind, is_active,
            priority, match_type, match_weight, display_order,
            group_id, tags, rules_json, source, season_start_month,
            created_at, updated_at
        FROM admin_leagues
        WHERE league_id = :lid
    """)
    result = await session.execute(current_query, {"lid": league_id})
    row = result.fetchone()

    if not row:
        raise ValueError(f"League {league_id} not found")

    # Build before state
    before_state = {
        "league_id": row.league_id,
        "sport": row.sport,
        "name": row.name,
        "display_name": row.display_name,
        "logo_url": row.logo_url,
        "wikipedia_url": row.wikipedia_url,
        "country": row.country,
        "kind": row.kind,
        "is_active": row.is_active,
        "priority": row.priority,
        "match_type": row.match_type,
        "match_weight": row.match_weight,
        "display_order": row.display_order,
        "group_id": row.group_id,
        "tags": row.tags if isinstance(row.tags, dict) else {},
        "rules_json": row.rules_json if isinstance(row.rules_json, dict) else {},
        "source": row.source,
        "season_start_month": row.season_start_month,
    }

    # Build SET clause dynamically
    set_parts = []
    params = {"lid": league_id}

    for key, value in sanitized.items():
        if key in ("tags", "rules_json"):
            set_parts.append(f"{key} = CAST(:{key} AS jsonb)")
            params[key] = json.dumps(value)
        else:
            set_parts.append(f"{key} = :{key}")
            params[key] = value

    # Update source to 'override' if it was 'seed'
    if before_state["source"] == "seed":
        set_parts.append("source = 'override'")

    set_clause = ", ".join(set_parts)

    # Execute update
    update_query = text(f"""
        UPDATE admin_leagues
        SET {set_clause}
        WHERE league_id = :lid
        RETURNING
            league_id, sport, name, display_name, logo_url, wikipedia_url,
            country, kind, is_active,
            priority, match_type, match_weight, display_order,
            group_id, tags, rules_json, source, season_start_month,
            created_at, updated_at
    """)

    result = await session.execute(update_query, params)
    updated_row = result.fetchone()

    # Build after state
    after_state = {
        "league_id": updated_row.league_id,
        "sport": updated_row.sport,
        "name": updated_row.name,
        "display_name": updated_row.display_name,
        "logo_url": updated_row.logo_url,
        "wikipedia_url": updated_row.wikipedia_url,
        "country": updated_row.country,
        "kind": updated_row.kind,
        "is_active": updated_row.is_active,
        "priority": updated_row.priority,
        "match_type": updated_row.match_type,
        "match_weight": updated_row.match_weight,
        "display_order": updated_row.display_order,
        "group_id": updated_row.group_id,
        "tags": updated_row.tags if isinstance(updated_row.tags, dict) else {},
        "rules_json": updated_row.rules_json if isinstance(updated_row.rules_json, dict) else {},
        "source": updated_row.source,
        "season_start_month": updated_row.season_start_month,
    }

    # Write audit log
    audit_id = await _write_audit_log(
        session,
        entity_type="admin_leagues",
        entity_id=str(league_id),
        action="update",
        before_json=before_state,
        after_json=after_state,
        actor=actor
    )

    await session.commit()

    # Invalidate cache
    global _league_cache
    _league_cache = {}

    return {
        "league": after_state,
        "audit_id": audit_id,
        "changes_applied": list(sanitized.keys()),
    }


# =============================================================================
# P2B - Audit Log
# =============================================================================

# Allowlist of valid entity types for audit queries (security hardening)
VALID_AUDIT_ENTITY_TYPES = {"admin_leagues", "admin_league_groups"}


async def get_audit_log(
    session: AsyncSession,
    entity_type: Optional[str] = None,
    entity_id: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
) -> dict:
    """
    Get audit log entries with optional filters.

    Args:
        session: Database session
        entity_type: Filter by entity type (e.g., 'admin_leagues')
        entity_id: Filter by entity ID
        limit: Max entries to return (default 50, max 200)
        offset: Pagination offset

    Returns:
        Dict with entries list and pagination info

    Raises:
        ValidationError: If entity_type is not in allowlist
    """
    # Validate entity_type against allowlist (security hardening)
    if entity_type and entity_type not in VALID_AUDIT_ENTITY_TYPES:
        raise ValidationError(
            f"invalid entity_type: must be one of {sorted(VALID_AUDIT_ENTITY_TYPES)}"
        )

    limit = min(max(limit, 1), 200)
    offset = max(offset, 0)

    where_parts = []
    params = {"limit": limit, "offset": offset}

    if entity_type:
        where_parts.append("entity_type = :entity_type")
        params["entity_type"] = entity_type

    if entity_id:
        where_parts.append("entity_id = :entity_id")
        params["entity_id"] = entity_id

    where_clause = "WHERE " + " AND ".join(where_parts) if where_parts else ""

    # Count total
    count_query = text(f"""
        SELECT COUNT(*) as total
        FROM admin_audit_log
        {where_clause}
    """)
    count_result = await session.execute(count_query, params)
    total = count_result.fetchone().total

    # Get entries
    entries_query = text(f"""
        SELECT
            id, entity_type, entity_id, action, actor,
            before_json, after_json, created_at
        FROM admin_audit_log
        {where_clause}
        ORDER BY created_at DESC
        LIMIT :limit OFFSET :offset
    """)
    result = await session.execute(entries_query, params)

    entries = []
    for row in result.fetchall():
        entries.append({
            "id": row.id,
            "entity_type": row.entity_type,
            "entity_id": row.entity_id,
            "action": row.action,
            "actor": row.actor,
            "before": row.before_json if isinstance(row.before_json, dict) else None,
            "after": row.after_json if isinstance(row.after_json, dict) else None,
            "created_at": row.created_at.isoformat() if row.created_at else None,
        })

    return {
        "entries": entries,
        "pagination": {
            "total": total,
            "limit": limit,
            "offset": offset,
            "has_more": offset + limit < total,
        },
        "filters": {
            "entity_type": entity_type,
            "entity_id": entity_id,
        },
    }


# =============================================================================
# League Groups Endpoints (P2C)
# =============================================================================


async def build_league_groups_list(session: AsyncSession) -> dict:
    """
    Build list of league groups with aggregated metrics.
    """
    # Get all groups
    groups_query = text("""
        SELECT g.group_id, g.group_key, g.name, g.country, g.tags
        FROM admin_league_groups g
        ORDER BY g.name
    """)
    groups_result = await session.execute(groups_query)
    groups_rows = groups_result.fetchall()

    # Get member leagues for each group
    members_query = text("""
        SELECT
            al.group_id,
            al.league_id,
            al.name as league_name,
            al.is_active
        FROM admin_leagues al
        WHERE al.group_id IS NOT NULL
        ORDER BY al.group_id, al.name
    """)
    members_result = await session.execute(members_query)
    members_by_group = {}
    for row in members_result.fetchall():
        if row.group_id not in members_by_group:
            members_by_group[row.group_id] = []
        members_by_group[row.group_id].append({
            "league_id": row.league_id,
            "name": row.league_name,
            "is_active": row.is_active
        })

    # Get aggregated stats for each group
    stats_query = text("""
        SELECT
            al.group_id,
            COUNT(m.id) as total_matches,
            COUNT(m.id) FILTER (WHERE m.date >= '2025-08-01') as matches_25_26,
            MAX(m.date) as last_match,
            MIN(m.season) as first_season,
            MAX(m.season) as last_season,
            COUNT(m.id) FILTER (WHERE m.stats IS NOT NULL AND m.stats::text != '{}') as with_stats,
            COUNT(m.id) FILTER (WHERE m.odds_home IS NOT NULL) as with_odds
        FROM admin_leagues al
        LEFT JOIN matches m ON m.league_id = al.league_id
        WHERE al.group_id IS NOT NULL
        GROUP BY al.group_id
    """)
    stats_result = await session.execute(stats_query)
    stats_by_group = {row.group_id: row for row in stats_result.fetchall()}

    # Build response
    groups = []
    for g in groups_rows:
        members = members_by_group.get(g.group_id, [])
        stats = stats_by_group.get(g.group_id)

        total = stats.total_matches if stats else 0
        with_stats = stats.with_stats if stats else 0
        with_odds = stats.with_odds if stats else 0

        groups.append({
            "group_id": g.group_id,
            "group_key": g.group_key,
            "name": g.name,
            "country": g.country,
            "leagues": members,
            "is_active_any": any(m["is_active"] for m in members),
            "is_active_all": all(m["is_active"] for m in members) if members else False,
            "stats": {
                "total_matches": total,
                "matches_25_26": stats.matches_25_26 if stats else 0,
                "last_match": stats.last_match.isoformat() if stats and stats.last_match else None,
                "seasons_range": [stats.first_season, stats.last_season] if stats and stats.first_season else None,
                "with_stats_pct": round(with_stats * 100 / total, 1) if total > 0 else None,
                "with_odds_pct": round(with_odds * 100 / total, 1) if total > 0 else None,
            }
        })

    return {
        "groups": groups,
        "total": len(groups)
    }


async def build_league_group_detail(session: AsyncSession, group_id: int) -> Optional[dict]:
    """
    Build detailed view of a league group.
    Returns None if group not found.
    """
    # Get group info
    group_query = text("""
        SELECT group_id, group_key, name, country, tags, created_at, updated_at
        FROM admin_league_groups
        WHERE group_id = :gid
    """)
    result = await session.execute(group_query, {"gid": group_id})
    group_row = result.fetchone()

    if not group_row:
        return None

    # Get member leagues with full stats
    members_query = text("""
        SELECT
            al.league_id, al.name, al.country, al.kind, al.is_active,
            al.priority, al.match_type, al.match_weight, al.rules_json,
            COUNT(m.id) as total_matches,
            COUNT(m.id) FILTER (WHERE m.date >= '2025-08-01') as matches_25_26,
            COUNT(m.id) FILTER (WHERE m.status IN ('FT', 'AET', 'PEN')) as finished,
            MAX(m.date) as last_match
        FROM admin_leagues al
        LEFT JOIN matches m ON m.league_id = al.league_id
        WHERE al.group_id = :gid
        GROUP BY al.league_id, al.name, al.country, al.kind, al.is_active,
                 al.priority, al.match_type, al.match_weight, al.rules_json
        ORDER BY al.name
    """)
    members_result = await session.execute(members_query, {"gid": group_id})

    member_leagues = []
    for m in members_result.fetchall():
        member_leagues.append({
            "league_id": m.league_id,
            "name": m.name,
            "country": m.country,
            "kind": m.kind,
            "is_active": m.is_active,
            "priority": m.priority,
            "match_type": m.match_type,
            "match_weight": m.match_weight,
            "rules_json": m.rules_json if isinstance(m.rules_json, dict) else {},
            "stats": {
                "total_matches": m.total_matches,
                "matches_25_26": m.matches_25_26,
                "finished": m.finished,
                "last_match": m.last_match.isoformat() if m.last_match else None
            }
        })

    # Aggregated stats by season
    league_ids = [m["league_id"] for m in member_leagues]
    if league_ids:
        stats_by_season_query = text("""
            SELECT
                season,
                COUNT(*) as total_matches,
                COUNT(*) FILTER (WHERE status IN ('FT', 'AET', 'PEN')) as finished,
                COUNT(*) FILTER (WHERE stats IS NOT NULL AND stats::text != '{}') as with_stats,
                COUNT(*) FILTER (WHERE odds_home IS NOT NULL) as with_odds
            FROM matches
            WHERE league_id = ANY(:lids)
            GROUP BY season
            ORDER BY season DESC
            LIMIT 10
        """)
        stats_result = await session.execute(stats_by_season_query, {"lids": league_ids})
        stats_by_season = [
            {
                "season": r.season,
                "total_matches": r.total_matches,
                "finished": r.finished,
                "with_stats_pct": round(r.with_stats * 100 / r.total_matches, 1) if r.total_matches > 0 else None,
                "with_odds_pct": round(r.with_odds * 100 / r.total_matches, 1) if r.total_matches > 0 else None
            }
            for r in stats_result.fetchall()
        ]
    else:
        stats_by_season = []

    # Distinct teams across all member leagues
    if league_ids:
        teams_query = text("""
            SELECT DISTINCT t.id as team_id, t.name, t.country
            FROM teams t
            JOIN matches m ON t.id = m.home_team_id OR t.id = m.away_team_id
            WHERE m.league_id = ANY(:lids)
            ORDER BY t.name
            LIMIT 100
        """)
        teams_result = await session.execute(teams_query, {"lids": league_ids})
        teams = [{"team_id": r.team_id, "name": r.name, "country": r.country}
                 for r in teams_result.fetchall()]
    else:
        teams = []

    # Recent matches across all member leagues
    if league_ids:
        recent_query = text("""
            SELECT
                m.id as match_id, m.date, m.league_id, m.status,
                ht.name as home_team, at.name as away_team,
                m.home_goals, m.away_goals
            FROM matches m
            JOIN teams ht ON m.home_team_id = ht.id
            JOIN teams at ON m.away_team_id = at.id
            WHERE m.league_id = ANY(:lids)
            ORDER BY m.date DESC
            LIMIT 20
        """)
        recent_result = await session.execute(recent_query, {"lids": league_ids})
        recent_matches = [
            {
                "match_id": r.match_id,
                "date": r.date.isoformat() if r.date else None,
                "league_id": r.league_id,
                "status": r.status,
                "home_team": r.home_team,
                "away_team": r.away_team,
                "score": f"{r.home_goals}-{r.away_goals}" if r.home_goals is not None else None
            }
            for r in recent_result.fetchall()
        ]
    else:
        recent_matches = []

    return {
        "group": {
            "group_id": group_row.group_id,
            "group_key": group_row.group_key,
            "name": group_row.name,
            "country": group_row.country,
            "tags": group_row.tags if isinstance(group_row.tags, dict) else {},
        },
        "member_leagues": member_leagues,
        "is_active_any": any(m["is_active"] for m in member_leagues),
        "is_active_all": all(m["is_active"] for m in member_leagues) if member_leagues else False,
        "stats_by_season": stats_by_season,
        "teams_count": len(teams),
        "teams": teams,
        "recent_matches": recent_matches
    }
