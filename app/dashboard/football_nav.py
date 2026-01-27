"""
Football Navigation API - Read-only endpoints for hierarchical UX navigation.

P3: DB-driven navigation for:
- TopBar Sport=Football → Col 2 categories → Col 2 list → Col 4 content → Col 5 drawer

All data sourced from admin_leagues + admin_league_groups (NOT COMPETITIONS dict).
Filters by is_active=true for "served" leagues.
"""

import logging
from datetime import datetime
from typing import Optional

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


# =============================================================================
# Navigation Categories
# =============================================================================

NAV_CATEGORIES = [
    {
        "id": "overview",
        "label": "Overview",
        "enabled": True,
        "note": None,
    },
    {
        "id": "leagues_by_country",
        "label": "Leagues by Country",
        "enabled": True,
        "note": None,
    },
    {
        "id": "national_teams",
        "label": "National Teams",
        "enabled": True,
        "note": None,
    },
    {
        "id": "tournaments_competitions",
        "label": "Tournaments & Cups",
        "enabled": True,
        "note": None,
    },
    {
        "id": "world_cup_2026",
        "label": "World Cup 2026",
        "enabled": False,
        "note": "Coming soon",
    },
    {
        "id": "players",
        "label": "Players",
        "enabled": False,
        "note": "Coming soon",
    },
]


async def build_nav(session: AsyncSession) -> dict:
    """
    Build top-level navigation categories for Football.
    Returns static categories with dynamic counts.
    """
    # Get counts for badges
    counts_query = text("""
        SELECT
            COUNT(*) FILTER (WHERE is_active = true AND kind = 'league') as leagues,
            COUNT(*) FILTER (WHERE is_active = true AND kind = 'cup') as cups,
            COUNT(*) FILTER (WHERE is_active = true AND kind = 'international') as international,
            COUNT(DISTINCT country) FILTER (WHERE is_active = true AND country IS NOT NULL) as countries
        FROM admin_leagues
    """)
    result = await session.execute(counts_query)
    row = result.fetchone()

    # Add counts to categories
    categories = []
    for cat in NAV_CATEGORIES:
        cat_copy = cat.copy()
        if cat["id"] == "leagues_by_country":
            cat_copy["count"] = row.countries if row else 0
        elif cat["id"] == "tournaments_competitions":
            cat_copy["count"] = row.cups if row else 0
        elif cat["id"] == "national_teams":
            cat_copy["count"] = row.international if row else 0
        categories.append(cat_copy)

    return {
        "sport": "football",
        "categories": categories,
    }


# =============================================================================
# Leagues by Country
# =============================================================================

async def build_countries_list(session: AsyncSession) -> dict:
    """
    Build list of countries with active leagues.
    For Col 2 when category=leagues_by_country.
    """
    # Get countries with active leagues
    countries_query = text("""
        SELECT
            al.country,
            COUNT(*) as leagues_count,
            COUNT(DISTINCT al.group_id) FILTER (WHERE al.group_id IS NOT NULL) as groups_count,
            ARRAY_AGG(
                JSON_BUILD_OBJECT(
                    'league_id', al.league_id,
                    'name', al.name,
                    'kind', al.kind,
                    'priority', al.priority,
                    'group_id', al.group_id
                ) ORDER BY
                    CASE WHEN al.priority = 'high' THEN 1
                         WHEN al.priority = 'medium' THEN 2
                         ELSE 3 END,
                    al.display_order NULLS LAST,
                    al.name
            ) as leagues
        FROM admin_leagues al
        WHERE al.is_active = true
          AND al.country IS NOT NULL
          AND al.kind IN ('league', 'cup')
        GROUP BY al.country
        ORDER BY
            COUNT(*) FILTER (WHERE al.priority = 'high') DESC,
            COUNT(*) DESC,
            al.country
    """)
    result = await session.execute(countries_query)

    countries = []
    for row in result.fetchall():
        leagues_list = []
        for lg in row.leagues:
            league_entry = {
                "league_id": lg["league_id"],
                "name": lg["name"],
                "kind": lg["kind"],
            }
            if lg["group_id"]:
                league_entry["group_id"] = lg["group_id"]
            leagues_list.append(league_entry)

        countries.append({
            "country": row.country,
            "leagues_count": row.leagues_count,
            "groups_count": row.groups_count,
            "leagues": leagues_list,
        })

    return {
        "countries": countries,
        "total": len(countries),
    }


async def build_country_detail(session: AsyncSession, country: str) -> Optional[dict]:
    """
    Build detailed view of leagues for a specific country.
    For Col 4 when a country is selected.
    """
    # Get leagues for country
    leagues_query = text("""
        SELECT
            al.league_id, al.name, al.kind, al.is_active,
            al.priority, al.match_type, al.match_weight,
            al.group_id, al.rules_json, al.display_order
        FROM admin_leagues al
        WHERE al.is_active = true
          AND al.country = :country
        ORDER BY
            CASE WHEN al.priority = 'high' THEN 1
                 WHEN al.priority = 'medium' THEN 2
                 ELSE 3 END,
            al.display_order NULLS LAST,
            al.name
    """)
    result = await session.execute(leagues_query, {"country": country})
    leagues_rows = result.fetchall()

    if not leagues_rows:
        return None

    # Get groups in this country
    groups_query = text("""
        SELECT DISTINCT g.group_id, g.group_key, g.name
        FROM admin_league_groups g
        JOIN admin_leagues al ON al.group_id = g.group_id
        WHERE al.country = :country AND al.is_active = true
    """)
    groups_result = await session.execute(groups_query, {"country": country})
    groups_map = {r.group_id: {"key": r.group_key, "name": r.name} for r in groups_result.fetchall()}

    # Get league IDs for stats query
    league_ids = [r.league_id for r in leagues_rows]

    # Get stats for all leagues
    stats_query = text("""
        SELECT
            m.league_id,
            COUNT(*) as total_matches,
            MIN(m.season) as first_season,
            MAX(m.season) as last_season,
            MAX(m.date) as last_match,
            COUNT(*) FILTER (WHERE m.stats IS NOT NULL AND m.stats::text != '{}' AND (m.stats->>'_no_stats') IS NULL) as with_stats,
            COUNT(*) FILTER (WHERE m.odds_home IS NOT NULL) as with_odds
        FROM matches m
        WHERE m.league_id = ANY(:lids)
        GROUP BY m.league_id
    """)
    stats_result = await session.execute(stats_query, {"lids": league_ids})
    stats_by_league = {r.league_id: r for r in stats_result.fetchall()}

    # Get TITAN coverage
    titan_query = text("""
        SELECT
            competition_id as league_id,
            COUNT(*) as total,
            SUM(CASE WHEN tier1_complete THEN 1 ELSE 0 END) as tier1
        FROM titan.feature_matrix
        WHERE competition_id = ANY(:lids) AND season = 2025
        GROUP BY competition_id
    """)
    try:
        titan_result = await session.execute(titan_query, {"lids": league_ids})
        titan_by_league = {r.league_id: r for r in titan_result.fetchall()}
    except Exception:
        titan_by_league = {}

    # Build league entries
    leagues = []
    grouped_league_ids = set()

    # First pass: identify grouped leagues
    for r in leagues_rows:
        if r.group_id:
            grouped_league_ids.add(r.league_id)

    # Build groups first
    groups_added = set()
    for r in leagues_rows:
        if r.group_id and r.group_id not in groups_added:
            group_info = groups_map.get(r.group_id, {})
            # Get all members of this group
            members = [lr for lr in leagues_rows if lr.group_id == r.group_id]

            # Aggregate stats for group
            group_total = 0
            group_with_stats = 0
            group_with_odds = 0
            group_last_match = None
            group_seasons = []

            for m in members:
                stats = stats_by_league.get(m.league_id)
                if stats:
                    group_total += stats.total_matches
                    group_with_stats += stats.with_stats
                    group_with_odds += stats.with_odds
                    if stats.last_match:
                        if not group_last_match or stats.last_match > group_last_match:
                            group_last_match = stats.last_match
                    if stats.first_season:
                        group_seasons.append(stats.first_season)
                    if stats.last_season:
                        group_seasons.append(stats.last_season)

            leagues.append({
                "type": "group",
                "group_id": r.group_id,
                "group_key": group_info.get("key"),
                "name": group_info.get("name", f"Group {r.group_id}"),
                "member_count": len(members),
                "members": [
                    {"league_id": m.league_id, "name": m.name, "kind": m.kind}
                    for m in members
                ],
                "stats": {
                    "total_matches": group_total,
                    "seasons_range": [min(group_seasons), max(group_seasons)] if group_seasons else None,
                    "last_match": group_last_match.isoformat() if group_last_match else None,
                    "with_stats_pct": round(group_with_stats * 100 / group_total, 1) if group_total > 0 else None,
                    "with_odds_pct": round(group_with_odds * 100 / group_total, 1) if group_total > 0 else None,
                },
            })
            groups_added.add(r.group_id)

    # Add non-grouped leagues
    for r in leagues_rows:
        if r.league_id not in grouped_league_ids:
            stats = stats_by_league.get(r.league_id)
            titan = titan_by_league.get(r.league_id)

            league_entry = {
                "type": "league",
                "league_id": r.league_id,
                "name": r.name,
                "kind": r.kind,
                "priority": r.priority,
                "match_type": r.match_type,
                "stats": {
                    "total_matches": stats.total_matches if stats else 0,
                    "seasons_range": [stats.first_season, stats.last_season] if stats and stats.first_season else None,
                    "last_match": stats.last_match.isoformat() if stats and stats.last_match else None,
                    "with_stats_pct": round(stats.with_stats * 100 / stats.total_matches, 1) if stats and stats.total_matches > 0 else None,
                    "with_odds_pct": round(stats.with_odds * 100 / stats.total_matches, 1) if stats and stats.total_matches > 0 else None,
                },
            }
            if titan:
                league_entry["titan"] = {
                    "total": titan.total,
                    "tier1": titan.tier1,
                    "tier1_pct": round(titan.tier1 * 100 / titan.total, 1) if titan.total > 0 else None,
                }
            leagues.append(league_entry)

    return {
        "country": country,
        "competitions": leagues,
        "total": len(leagues),
    }


# =============================================================================
# League Detail (for navigation)
# =============================================================================

async def build_league_nav_detail(session: AsyncSession, league_id: int) -> Optional[dict]:
    """
    Build navigation detail for a specific league.
    For Col 4 drilldown.
    """
    # Get league info
    league_query = text("""
        SELECT
            al.league_id, al.name, al.country, al.kind, al.is_active,
            al.priority, al.match_type, al.match_weight,
            al.group_id, al.rules_json, al.tags, al.source
        FROM admin_leagues al
        WHERE al.league_id = :lid AND al.is_active = true
    """)
    result = await session.execute(league_query, {"lid": league_id})
    league_row = result.fetchone()

    if not league_row:
        return None

    # Get group info if applicable
    group_info = None
    if league_row.group_id:
        group_query = text("""
            SELECT group_id, group_key, name, country
            FROM admin_league_groups
            WHERE group_id = :gid
        """)
        group_result = await session.execute(group_query, {"gid": league_row.group_id})
        gr = group_result.fetchone()
        if gr:
            rules = league_row.rules_json if isinstance(league_row.rules_json, dict) else {}
            group_info = {
                "group_id": gr.group_id,
                "key": gr.group_key,
                "name": gr.name,
                "country": gr.country,
                "paired_handling": rules.get("paired_handling", "grouped"),
            }

    # Get stats by season
    stats_query = text("""
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
        LIMIT 5
    """)
    stats_result = await session.execute(stats_query, {"lid": league_id})
    stats_by_season = [
        {
            "season": r.season,
            "total_matches": r.total_matches,
            "finished": r.finished,
            "with_stats_pct": round(r.with_stats * 100 / r.total_matches, 1) if r.total_matches > 0 else None,
            "with_odds_pct": round(r.with_odds * 100 / r.total_matches, 1) if r.total_matches > 0 else None,
        }
        for r in stats_result.fetchall()
    ]

    # Get TITAN coverage for current season
    titan_query = text("""
        SELECT
            COUNT(*) as total,
            SUM(CASE WHEN tier1_complete THEN 1 ELSE 0 END) as tier1,
            SUM(CASE WHEN tier1b_complete THEN 1 ELSE 0 END) as tier1b,
            SUM(CASE WHEN tier1c_complete THEN 1 ELSE 0 END) as tier1c,
            SUM(CASE WHEN tier1d_complete THEN 1 ELSE 0 END) as tier1d
        FROM titan.feature_matrix
        WHERE competition_id = :lid AND season = 2025
    """)
    try:
        titan_result = await session.execute(titan_query, {"lid": league_id})
        titan_row = titan_result.fetchone()
        titan = None
        if titan_row and titan_row.total > 0:
            titan = {
                "total": titan_row.total,
                "tier1": titan_row.tier1,
                "tier1b": titan_row.tier1b,
                "tier1c": titan_row.tier1c,
                "tier1d": titan_row.tier1d,
                "tier1_pct": round(titan_row.tier1 * 100 / titan_row.total, 1),
            }
    except Exception:
        titan = None

    # Get recent matches
    recent_query = text("""
        SELECT
            m.id as match_id, m.date, m.status,
            ht.name as home_team, at.name as away_team,
            m.home_goals, m.away_goals
        FROM matches m
        JOIN teams ht ON m.home_team_id = ht.id
        JOIN teams at ON m.away_team_id = at.id
        WHERE m.league_id = :lid
        ORDER BY m.date DESC
        LIMIT 10
    """)
    recent_result = await session.execute(recent_query, {"lid": league_id})
    recent_matches = [
        {
            "match_id": r.match_id,
            "date": r.date.isoformat() if r.date else None,
            "status": r.status,
            "home_team": r.home_team,
            "away_team": r.away_team,
            "score": f"{r.home_goals}-{r.away_goals}" if r.home_goals is not None else None,
        }
        for r in recent_result.fetchall()
    ]

    return {
        "league": {
            "league_id": league_row.league_id,
            "name": league_row.name,
            "country": league_row.country,
            "kind": league_row.kind,
            "priority": league_row.priority,
            "match_type": league_row.match_type,
            "match_weight": league_row.match_weight,
            "rules_json": league_row.rules_json if isinstance(league_row.rules_json, dict) else {},
        },
        "group": group_info,
        "stats_by_season": stats_by_season,
        "titan": titan,
        "recent_matches": recent_matches,
        "standings": {
            "status": "not_available",
            "note": "Standings table coming soon",
        },
    }


# =============================================================================
# Group Detail (for navigation)
# =============================================================================

async def build_group_nav_detail(session: AsyncSession, group_id: int) -> Optional[dict]:
    """
    Build navigation detail for a league group (paired leagues).
    For Col 4 drilldown.
    """
    # Get group info
    group_query = text("""
        SELECT group_id, group_key, name, country, tags
        FROM admin_league_groups
        WHERE group_id = :gid
    """)
    result = await session.execute(group_query, {"gid": group_id})
    group_row = result.fetchone()

    if not group_row:
        return None

    # Get member leagues (only active)
    members_query = text("""
        SELECT
            al.league_id, al.name, al.kind, al.is_active,
            al.priority, al.match_type, al.match_weight, al.rules_json
        FROM admin_leagues al
        WHERE al.group_id = :gid AND al.is_active = true
        ORDER BY al.name
    """)
    members_result = await session.execute(members_query, {"gid": group_id})
    members_rows = members_result.fetchall()

    if not members_rows:
        return None  # No active members

    # Get paired_handling from first member's rules_json
    paired_handling = "grouped"
    for m in members_rows:
        rules = m.rules_json if isinstance(m.rules_json, dict) else {}
        if "paired_handling" in rules:
            paired_handling = rules["paired_handling"]
            break

    member_leagues = []
    league_ids = []
    for m in members_rows:
        league_ids.append(m.league_id)
        member_leagues.append({
            "league_id": m.league_id,
            "name": m.name,
            "kind": m.kind,
            "priority": m.priority,
            "match_type": m.match_type,
        })

    # Get aggregated stats by season
    stats_query = text("""
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
        LIMIT 5
    """)
    stats_result = await session.execute(stats_query, {"lids": league_ids})
    stats_by_season = [
        {
            "season": r.season,
            "total_matches": r.total_matches,
            "finished": r.finished,
            "with_stats_pct": round(r.with_stats * 100 / r.total_matches, 1) if r.total_matches > 0 else None,
            "with_odds_pct": round(r.with_odds * 100 / r.total_matches, 1) if r.total_matches > 0 else None,
        }
        for r in stats_result.fetchall()
    ]

    # Get TITAN coverage
    titan_query = text("""
        SELECT
            COUNT(*) as total,
            SUM(CASE WHEN tier1_complete THEN 1 ELSE 0 END) as tier1
        FROM titan.feature_matrix
        WHERE competition_id = ANY(:lids) AND season = 2025
    """)
    try:
        titan_result = await session.execute(titan_query, {"lids": league_ids})
        titan_row = titan_result.fetchone()
        titan = None
        if titan_row and titan_row.total > 0:
            titan = {
                "total": titan_row.total,
                "tier1": titan_row.tier1,
                "tier1_pct": round(titan_row.tier1 * 100 / titan_row.total, 1),
            }
    except Exception:
        titan = None

    # Get recent matches across all members
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
        LIMIT 10
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
            "score": f"{r.home_goals}-{r.away_goals}" if r.home_goals is not None else None,
        }
        for r in recent_result.fetchall()
    ]

    return {
        "group": {
            "group_id": group_row.group_id,
            "group_key": group_row.group_key,
            "name": group_row.name,
            "country": group_row.country,
            "paired_handling": paired_handling,
        },
        "member_leagues": member_leagues,
        "is_active_all": all(m.is_active for m in members_rows),
        "stats_by_season": stats_by_season,
        "titan": titan,
        "recent_matches": recent_matches,
        "standings": {
            "status": "not_available",
            "note": "Standings table coming soon",
        },
    }
