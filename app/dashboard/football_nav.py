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
            m.home_team_id, m.away_team_id,
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
            "home_team_id": r.home_team_id,
            "away_team_id": r.away_team_id,
            "home_team": r.home_team,
            "away_team": r.away_team,
            "score": f"{r.home_goals}-{r.away_goals}" if r.home_goals is not None else None,
        }
        for r in recent_result.fetchall()
    ]

    # Get participants (top 50 teams by matches in this league)
    participants_query = text("""
        SELECT
            t.id as team_id,
            t.name,
            t.country,
            t.logo_url,
            COUNT(m.id) as matches_in_league,
            MIN(m.season) as first_season,
            MAX(m.season) as last_season
        FROM teams t
        JOIN matches m ON t.id = m.home_team_id OR t.id = m.away_team_id
        WHERE m.league_id = :lid
        GROUP BY t.id, t.name, t.country, t.logo_url
        ORDER BY COUNT(m.id) DESC
        LIMIT 50
    """)
    participants_result = await session.execute(participants_query, {"lid": league_id})
    participants = [
        {
            "team_id": r.team_id,
            "name": r.name,
            "country": r.country,
            "logo_url": r.logo_url,
            "matches_in_league": r.matches_in_league,
            "seasons_range": [r.first_season, r.last_season] if r.first_season else None,
        }
        for r in participants_result.fetchall()
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
        "participants": participants,
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
            m.home_team_id, m.away_team_id,
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
            "home_team_id": r.home_team_id,
            "away_team_id": r.away_team_id,
            "home_team": r.home_team,
            "away_team": r.away_team,
            "score": f"{r.home_goals}-{r.away_goals}" if r.home_goals is not None else None,
        }
        for r in recent_result.fetchall()
    ]

    # Get participants (top 60 teams by matches across all member leagues)
    participants_query = text("""
        SELECT
            t.id as team_id,
            t.name,
            t.country,
            t.logo_url,
            COUNT(m.id) as matches_in_group,
            MIN(m.season) as first_season,
            MAX(m.season) as last_season
        FROM teams t
        JOIN matches m ON t.id = m.home_team_id OR t.id = m.away_team_id
        WHERE m.league_id = ANY(:lids)
        GROUP BY t.id, t.name, t.country, t.logo_url
        ORDER BY COUNT(m.id) DESC
        LIMIT 60
    """)
    participants_result = await session.execute(participants_query, {"lids": league_ids})
    participants = [
        {
            "team_id": r.team_id,
            "name": r.name,
            "country": r.country,
            "logo_url": r.logo_url,
            "matches_in_group": r.matches_in_group,
            "seasons_range": [r.first_season, r.last_season] if r.first_season else None,
        }
        for r in participants_result.fetchall()
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
        "participants": participants,
        "standings": {
            "status": "not_available",
            "note": "Standings table coming soon",
        },
    }


# =============================================================================
# Football Overview (P3.1)
# =============================================================================

async def build_football_overview(session: AsyncSession) -> dict:
    """
    Build football overview for category=overview.
    Shows summary counts, upcoming matches, top leagues, and alerts.
    All filtered by admin_leagues.is_active = true.
    """
    # 1. Summary counts (single query for efficiency)
    summary_query = text("""
        WITH active_leagues AS (
            SELECT league_id FROM admin_leagues WHERE is_active = true
        )
        SELECT
            (SELECT COUNT(*) FROM admin_leagues WHERE is_active = true) as leagues_active,
            (SELECT COUNT(DISTINCT country) FROM admin_leagues WHERE is_active = true AND country IS NOT NULL) as countries_active,
            (SELECT COUNT(*) FROM matches WHERE league_id IN (SELECT league_id FROM active_leagues)
                AND date >= NOW() AND date < NOW() + INTERVAL '7 days') as matches_next_7d,
            (SELECT COUNT(*) FROM matches WHERE league_id IN (SELECT league_id FROM active_leagues)
                AND status IN ('1H', '2H', 'HT', 'LIVE', 'ET', 'BT', 'P')) as matches_live,
            (SELECT COUNT(*) FROM matches WHERE league_id IN (SELECT league_id FROM active_leagues)
                AND status IN ('FT', 'AET', 'PEN') AND date >= NOW() - INTERVAL '24 hours') as matches_finished_24h
    """)
    summary_result = await session.execute(summary_query)
    summary_row = summary_result.fetchone()

    # 2. Teams active count (separate for clarity)
    teams_query = text("""
        SELECT COUNT(DISTINCT t.id) as teams_active
        FROM teams t
        JOIN matches m ON t.id = m.home_team_id OR t.id = m.away_team_id
        JOIN admin_leagues al ON m.league_id = al.league_id AND al.is_active = true
        WHERE m.date >= NOW() - INTERVAL '365 days'
    """)
    teams_result = await session.execute(teams_query)
    teams_row = teams_result.fetchone()

    summary = {
        "leagues_active_count": summary_row.leagues_active,
        "countries_active_count": summary_row.countries_active,
        "matches_next_7d_count": summary_row.matches_next_7d,
        "matches_live_count": summary_row.matches_live,
        "matches_finished_24h_count": summary_row.matches_finished_24h,
        "teams_active_count": teams_row.teams_active,
    }

    # 3. Upcoming matches (próximos 20)
    upcoming_query = text("""
        SELECT
            m.id as match_id, m.date, m.league_id, m.status,
            al.name as league_name,
            ht.name as home_team, at.name as away_team,
            EXISTS(SELECT 1 FROM predictions p WHERE p.match_id = m.id) as has_prediction
        FROM matches m
        JOIN admin_leagues al ON m.league_id = al.league_id AND al.is_active = true
        JOIN teams ht ON m.home_team_id = ht.id
        JOIN teams at ON m.away_team_id = at.id
        WHERE m.date >= NOW() AND m.status = 'NS'
        ORDER BY m.date ASC
        LIMIT 20
    """)
    upcoming_result = await session.execute(upcoming_query)
    upcoming = [
        {
            "match_id": r.match_id,
            "date": r.date.isoformat() if r.date else None,
            "league_id": r.league_id,
            "league_name": r.league_name,
            "home_team": r.home_team,
            "away_team": r.away_team,
            "status": r.status,
            "has_prediction": r.has_prediction,
        }
        for r in upcoming_result.fetchall()
    ]

    # 4. Top 10 leagues by matches_30d, then matches_total
    leagues_query = text("""
        SELECT
            al.league_id, al.name, al.country,
            COUNT(*) FILTER (WHERE m.date >= NOW() - INTERVAL '30 days') as matches_30d,
            COUNT(*) as matches_total,
            COUNT(*) FILTER (WHERE m.stats IS NOT NULL AND m.stats::text != '{}' AND (m.stats->>'_no_stats') IS NULL) as with_stats,
            COUNT(*) FILTER (WHERE m.odds_home IS NOT NULL) as with_odds
        FROM admin_leagues al
        LEFT JOIN matches m ON m.league_id = al.league_id
        WHERE al.is_active = true
        GROUP BY al.league_id, al.name, al.country
        ORDER BY COUNT(*) FILTER (WHERE m.date >= NOW() - INTERVAL '30 days') DESC, COUNT(*) DESC
        LIMIT 10
    """)
    leagues_result = await session.execute(leagues_query)
    leagues = [
        {
            "league_id": r.league_id,
            "name": r.name,
            "country": r.country,
            "matches_30d": r.matches_30d,
            "matches_total": r.matches_total,
            "with_stats_pct": round(r.with_stats * 100 / r.matches_total, 1) if r.matches_total > 0 else None,
            "with_odds_pct": round(r.with_odds * 100 / r.matches_total, 1) if r.matches_total > 0 else None,
        }
        for r in leagues_result.fetchall()
    ]

    # 5. Alerts: ligas activas con with_stats_pct < 50 o with_odds_pct < 50
    alerts_query = text("""
        WITH league_coverage AS (
            SELECT
                al.league_id, al.name,
                COUNT(*) as total,
                COUNT(*) FILTER (WHERE m.stats IS NOT NULL AND m.stats::text != '{}' AND (m.stats->>'_no_stats') IS NULL) as with_stats,
                COUNT(*) FILTER (WHERE m.odds_home IS NOT NULL) as with_odds
            FROM admin_leagues al
            LEFT JOIN matches m ON m.league_id = al.league_id
            WHERE al.is_active = true
            GROUP BY al.league_id, al.name
            HAVING COUNT(*) > 0
        )
        SELECT league_id, name,
            ROUND(with_stats * 100.0 / total, 1) as stats_pct,
            ROUND(with_odds * 100.0 / total, 1) as odds_pct
        FROM league_coverage
        WHERE with_stats * 100.0 / total < 50
           OR with_odds * 100.0 / total < 50
        ORDER BY with_stats * 100.0 / total ASC
        LIMIT 5
    """)
    alerts_result = await session.execute(alerts_query)
    alerts = []
    for r in alerts_result.fetchall():
        if r.stats_pct < 50:
            alerts.append({
                "type": "low_stats_coverage",
                "league_id": r.league_id,
                "league_name": r.name,
                "message": f"Stats coverage below 50%: {r.stats_pct}%",
                "value": float(r.stats_pct),
            })
        if r.odds_pct < 50:
            alerts.append({
                "type": "low_odds_coverage",
                "league_id": r.league_id,
                "league_name": r.name,
                "message": f"Odds coverage below 50%: {r.odds_pct}%",
                "value": float(r.odds_pct),
            })

    # 6. TITAN coverage (fail-soft)
    titan = None
    try:
        titan_query = text("""
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN tier1_complete THEN 1 ELSE 0 END) as tier1,
                SUM(CASE WHEN tier1b_complete THEN 1 ELSE 0 END) as tier1b
            FROM titan.feature_matrix fm
            JOIN admin_leagues al ON fm.competition_id = al.league_id AND al.is_active = true
            WHERE fm.season = 2025
        """)
        titan_result = await session.execute(titan_query)
        titan_row = titan_result.fetchone()
        if titan_row and titan_row.total > 0:
            titan = {
                "total": titan_row.total,
                "tier1": titan_row.tier1,
                "tier1b": titan_row.tier1b,
                "tier1_pct": round(titan_row.tier1 * 100 / titan_row.total, 1),
                "tier1b_pct": round(titan_row.tier1b * 100 / titan_row.total, 1),
            }
    except Exception as e:
        logger.warning(f"TITAN query failed (fail-soft): {e}")
        titan = None

    return {
        "summary": summary,
        "upcoming": upcoming,
        "leagues": leagues,
        "alerts": alerts,
        "titan": titan,
    }


# =============================================================================
# National Teams API (P3.3)
# =============================================================================


async def build_nationals_countries_list(session: AsyncSession) -> dict:
    """
    Build list of countries with national teams.
    Only includes teams with matches in active international competitions.
    Fail-soft: returns empty list if no international competitions.
    """
    # Get active international competitions
    intl_leagues_query = text("""
        SELECT league_id, name FROM admin_leagues
        WHERE kind = 'international' AND is_active = true
    """)
    intl_result = await session.execute(intl_leagues_query)
    intl_leagues = {r.league_id: r.name for r in intl_result.fetchall()}
    intl_league_ids = list(intl_leagues.keys())

    # Fail-soft: no international competitions
    if not intl_league_ids:
        return {
            "countries": [],
            "totals": {"countries_count": 0, "teams_count": 0, "competitions_count": 0}
        }

    # Get national teams with match stats in international competitions
    # For national teams, teams.name IS the country (teams.country may be NULL)
    countries_query = text("""
        SELECT
            t.name as country,
            COUNT(DISTINCT t.id) as teams_count,
            COUNT(m.id) as total_matches,
            COUNT(DISTINCT m.league_id) as competitions_count,
            MAX(m.date) as last_match
        FROM teams t
        LEFT JOIN matches m ON (t.id = m.home_team_id OR t.id = m.away_team_id)
            AND m.league_id = ANY(:intl_ids)
        WHERE t.team_type = 'national'
        GROUP BY t.name
        HAVING COUNT(m.id) > 0
        ORDER BY COUNT(m.id) DESC
    """)
    result = await session.execute(countries_query, {"intl_ids": intl_league_ids})

    countries = []
    total_teams = 0
    for r in result.fetchall():
        countries.append({
            "country": r.country,
            "teams_count": r.teams_count,
            "total_matches": r.total_matches,
            "competitions_count": r.competitions_count,
            "last_match": r.last_match.isoformat() + "Z" if r.last_match else None
        })
        total_teams += r.teams_count

    return {
        "countries": countries,
        "totals": {
            "countries_count": len(countries),
            "teams_count": total_teams,
            "competitions_count": len(intl_leagues)
        }
    }


async def build_nationals_country_detail(session: AsyncSession, country: str) -> Optional[dict]:
    """
    Build detail for a country's national teams.
    The country parameter is the team name (e.g., "Portugal", "Spain").
    Returns None if no team found with that name.
    """
    # Get teams for this country (name matches country)
    teams_query = text("""
        SELECT id, name, logo_url
        FROM teams
        WHERE team_type = 'national' AND name = :country
    """)
    teams_result = await session.execute(teams_query, {"country": country})
    teams_rows = teams_result.fetchall()

    if not teams_rows:
        return None

    team_ids = [t.id for t in teams_rows]

    # Get international leagues
    intl_query = text("""
        SELECT league_id, name FROM admin_leagues
        WHERE kind = 'international' AND is_active = true
    """)
    intl_result = await session.execute(intl_query)
    intl_leagues = {r.league_id: r.name for r in intl_result.fetchall()}
    intl_ids = list(intl_leagues.keys()) or [-1]  # Placeholder to avoid empty array

    # Get stats per team
    stats_query = text("""
        SELECT
            t.id as team_id,
            t.name,
            t.logo_url,
            COUNT(m.id) as total_matches,
            COUNT(m.id) FILTER (WHERE m.date >= '2025-08-01') as matches_25_26,
            array_agg(DISTINCT m.league_id) FILTER (WHERE m.league_id IS NOT NULL) as competition_ids
        FROM teams t
        LEFT JOIN matches m ON (t.id = m.home_team_id OR t.id = m.away_team_id)
            AND m.league_id = ANY(:intl_ids)
        WHERE t.id = ANY(:tids)
        GROUP BY t.id, t.name, t.logo_url
    """)
    stats_result = await session.execute(stats_query, {"tids": team_ids, "intl_ids": intl_ids})

    teams = []
    for r in stats_result.fetchall():
        comp_names = [intl_leagues.get(lid, f"Competition {lid}") for lid in (r.competition_ids or [])]
        teams.append({
            "team_id": r.team_id,
            "name": r.name,
            "logo_url": r.logo_url,
            "total_matches": r.total_matches,
            "matches_25_26": r.matches_25_26,
            "competitions": comp_names
        })

    # Competitions with match counts
    comps_query = text("""
        SELECT
            m.league_id,
            COUNT(*) as matches_count
        FROM matches m
        WHERE (m.home_team_id = ANY(:tids) OR m.away_team_id = ANY(:tids))
            AND m.league_id = ANY(:intl_ids)
        GROUP BY m.league_id
        ORDER BY COUNT(*) DESC
    """)
    comps_result = await session.execute(comps_query, {"tids": team_ids, "intl_ids": intl_ids})
    competitions = [
        {
            "league_id": r.league_id,
            "name": intl_leagues.get(r.league_id, f"Competition {r.league_id}"),
            "matches_count": r.matches_count
        }
        for r in comps_result.fetchall()
    ]

    # Recent matches
    recent_query = text("""
        SELECT
            m.id as match_id, m.date, m.league_id, m.status,
            m.home_goals, m.away_goals,
            ht.name as home_team, at.name as away_team
        FROM matches m
        JOIN teams ht ON m.home_team_id = ht.id
        JOIN teams at ON m.away_team_id = at.id
        WHERE (m.home_team_id = ANY(:tids) OR m.away_team_id = ANY(:tids))
            AND m.league_id = ANY(:intl_ids)
        ORDER BY m.date DESC
        LIMIT 20
    """)
    recent_result = await session.execute(recent_query, {"tids": team_ids, "intl_ids": intl_ids})
    recent_matches = [
        {
            "match_id": r.match_id,
            "date": r.date.isoformat() + "Z" if r.date else None,
            "competition_name": intl_leagues.get(r.league_id, "Unknown"),
            "home_team": r.home_team,
            "away_team": r.away_team,
            "status": r.status,
            "score": f"{r.home_goals}-{r.away_goals}" if r.home_goals is not None else None
        }
        for r in recent_result.fetchall()
    ]

    # Overall stats (wins/draws/losses)
    overall_query = text("""
        SELECT
            COUNT(*) as total,
            COUNT(*) FILTER (WHERE
                (m.home_team_id = ANY(:tids) AND m.home_goals > m.away_goals) OR
                (m.away_team_id = ANY(:tids) AND m.away_goals > m.home_goals)
            ) as wins,
            COUNT(*) FILTER (WHERE m.home_goals = m.away_goals) as draws,
            COUNT(*) FILTER (WHERE
                (m.home_team_id = ANY(:tids) AND m.home_goals < m.away_goals) OR
                (m.away_team_id = ANY(:tids) AND m.away_goals < m.home_goals)
            ) as losses
        FROM matches m
        WHERE (m.home_team_id = ANY(:tids) OR m.away_team_id = ANY(:tids))
            AND m.league_id = ANY(:intl_ids)
            AND m.status IN ('FT', 'AET', 'PEN')
    """)
    overall_result = await session.execute(overall_query, {"tids": team_ids, "intl_ids": intl_ids})
    overall = overall_result.fetchone()

    return {
        "country": country,
        "teams": teams,
        "competitions": competitions,
        "recent_matches": recent_matches,
        "stats": {
            "total_matches": overall.total if overall else 0,
            "wins": overall.wins if overall else 0,
            "draws": overall.draws if overall else 0,
            "losses": overall.losses if overall else 0
        }
    }


async def build_nationals_team_detail(session: AsyncSession, team_id: int) -> Optional[dict]:
    """
    Build Team 360 for a national team.
    Returns None if team not found or not a national team.
    """
    # Get team (must be national)
    team_query = text("""
        SELECT id, name, logo_url, team_type
        FROM teams
        WHERE id = :tid AND team_type = 'national'
    """)
    team_result = await session.execute(team_query, {"tid": team_id})
    team_row = team_result.fetchone()

    if not team_row:
        return None

    # Get international leagues
    intl_query = text("""
        SELECT league_id, name FROM admin_leagues
        WHERE kind = 'international' AND is_active = true
    """)
    intl_result = await session.execute(intl_query)
    intl_leagues = {r.league_id: r.name for r in intl_result.fetchall()}
    intl_ids = list(intl_leagues.keys()) or [-1]

    # Stats by competition
    comp_stats_query = text("""
        SELECT
            m.league_id,
            COUNT(*) as total,
            COUNT(*) FILTER (WHERE
                (m.home_team_id = :tid AND m.home_goals > m.away_goals) OR
                (m.away_team_id = :tid AND m.away_goals > m.home_goals)
            ) as wins,
            COUNT(*) FILTER (WHERE m.home_goals = m.away_goals) as draws,
            COUNT(*) FILTER (WHERE
                (m.home_team_id = :tid AND m.home_goals < m.away_goals) OR
                (m.away_team_id = :tid AND m.away_goals < m.home_goals)
            ) as losses,
            MAX(m.date) as last_match
        FROM matches m
        WHERE (m.home_team_id = :tid OR m.away_team_id = :tid)
            AND m.league_id = ANY(:intl_ids)
            AND m.status IN ('FT', 'AET', 'PEN')
        GROUP BY m.league_id
        ORDER BY COUNT(*) DESC
    """)
    comp_stats_result = await session.execute(comp_stats_query, {"tid": team_id, "intl_ids": intl_ids})

    competitions = []
    stats_by_competition = []
    total_wins, total_draws, total_losses, total_matches = 0, 0, 0, 0

    for r in comp_stats_result.fetchall():
        name = intl_leagues.get(r.league_id, f"Competition {r.league_id}")
        competitions.append({
            "league_id": r.league_id,
            "name": name,
            "matches_count": r.total,
            "last_match": r.last_match.strftime("%Y-%m-%d") if r.last_match else None
        })
        stats_by_competition.append({
            "league_id": r.league_id,
            "name": name,
            "matches": r.total,
            "wins": r.wins,
            "draws": r.draws,
            "losses": r.losses
        })
        total_wins += r.wins
        total_draws += r.draws
        total_losses += r.losses
        total_matches += r.total

    # Goals stats
    goals_query = text("""
        SELECT
            SUM(CASE WHEN m.home_team_id = :tid THEN m.home_goals ELSE m.away_goals END) as goals_for,
            SUM(CASE WHEN m.home_team_id = :tid THEN m.away_goals ELSE m.home_goals END) as goals_against
        FROM matches m
        WHERE (m.home_team_id = :tid OR m.away_team_id = :tid)
            AND m.league_id = ANY(:intl_ids)
            AND m.status IN ('FT', 'AET', 'PEN')
    """)
    goals_result = await session.execute(goals_query, {"tid": team_id, "intl_ids": intl_ids})
    goals = goals_result.fetchone()

    # Recent matches
    recent_query = text("""
        SELECT
            m.id as match_id, m.date, m.league_id, m.status,
            m.home_team_id, m.away_team_id,
            m.home_goals, m.away_goals,
            ht.name as home_team, at.name as away_team
        FROM matches m
        JOIN teams ht ON m.home_team_id = ht.id
        JOIN teams at ON m.away_team_id = at.id
        WHERE (m.home_team_id = :tid OR m.away_team_id = :tid)
            AND m.league_id = ANY(:intl_ids)
        ORDER BY m.date DESC
        LIMIT 20
    """)
    recent_result = await session.execute(recent_query, {"tid": team_id, "intl_ids": intl_ids})

    recent_matches = []
    for r in recent_result.fetchall():
        is_home = r.home_team_id == team_id
        opponent = r.away_team if is_home else r.home_team
        result = None
        if r.home_goals is not None:
            if is_home:
                result = f"{r.home_goals}-{r.away_goals}"
            else:
                result = f"{r.away_goals}-{r.home_goals}"

        recent_matches.append({
            "match_id": r.match_id,
            "date": r.date.isoformat() + "Z" if r.date else None,
            "competition_id": r.league_id,
            "competition_name": intl_leagues.get(r.league_id, "Unknown"),
            "opponent": opponent,
            "home_away": "home" if is_home else "away",
            "result": result,
            "status": r.status
        })

    # Head to head (top 10 opponents)
    h2h_query = text("""
        SELECT
            CASE WHEN m.home_team_id = :tid THEN m.away_team_id ELSE m.home_team_id END as opponent_id,
            COUNT(*) as total,
            COUNT(*) FILTER (WHERE
                (m.home_team_id = :tid AND m.home_goals > m.away_goals) OR
                (m.away_team_id = :tid AND m.away_goals > m.home_goals)
            ) as wins,
            COUNT(*) FILTER (WHERE m.home_goals = m.away_goals) as draws,
            COUNT(*) FILTER (WHERE
                (m.home_team_id = :tid AND m.home_goals < m.away_goals) OR
                (m.away_team_id = :tid AND m.away_goals < m.home_goals)
            ) as losses
        FROM matches m
        WHERE (m.home_team_id = :tid OR m.away_team_id = :tid)
            AND m.league_id = ANY(:intl_ids)
            AND m.status IN ('FT', 'AET', 'PEN')
        GROUP BY opponent_id
        ORDER BY COUNT(*) DESC
        LIMIT 10
    """)
    h2h_result = await session.execute(h2h_query, {"tid": team_id, "intl_ids": intl_ids})
    h2h_rows = h2h_result.fetchall()

    # Get opponent names
    opponent_ids = [r.opponent_id for r in h2h_rows]
    if opponent_ids:
        names_query = text("SELECT id, name FROM teams WHERE id = ANY(:ids)")
        names_result = await session.execute(names_query, {"ids": opponent_ids})
        opponent_names = {r.id: r.name for r in names_result.fetchall()}
    else:
        opponent_names = {}

    head_to_head = [
        {
            "opponent_id": r.opponent_id,
            "opponent_name": opponent_names.get(r.opponent_id, "Unknown"),
            "total_matches": r.total,
            "wins": r.wins,
            "draws": r.draws,
            "losses": r.losses
        }
        for r in h2h_rows
    ]

    return {
        "team": {
            "team_id": team_row.id,
            "name": team_row.name,
            "logo_url": team_row.logo_url,
            "team_type": team_row.team_type
        },
        "competitions": competitions,
        "stats_overall": {
            "total_matches": total_matches,
            "wins": total_wins,
            "draws": total_draws,
            "losses": total_losses,
            "goals_for": int(goals.goals_for or 0) if goals else 0,
            "goals_against": int(goals.goals_against or 0) if goals else 0
        },
        "stats_by_competition": stats_by_competition,
        "recent_matches": recent_matches,
        "head_to_head": head_to_head
    }
