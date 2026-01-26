"""
Admin Panel P0 - Read-only endpoints for leagues/teams visibility.

All functions return data dicts (no cache handling - that's in main.py).
"""

import logging
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.etl.competitions import COMPETITIONS
from app.ml.health import LEAGUE_NAMES

logger = logging.getLogger(__name__)


def get_league_info(league_id: int) -> dict:
    """Get league info from COMPETITIONS dict or fallback to LEAGUE_NAMES."""
    comp = COMPETITIONS.get(league_id)
    if comp:
        return {
            "name": comp.name,
            "priority": comp.priority.value,
            "match_type": comp.match_type,
            "match_weight": comp.match_weight,
            "configured": True,
        }
    return {
        "name": LEAGUE_NAMES.get(league_id, f"League {league_id}"),
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
        info = get_league_info(r.league_id)
        top_leagues.append({
            "league_id": r.league_id,
            "name": info["name"],
            "matches": r.matches,
        })

    return {
        "counts": {
            "leagues_configured": len(COMPETITIONS),
            "leagues_observed": row.total_leagues,
            "leagues_active_30d": row.active_30d,
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
    """Build leagues list with configured vs observed distinction."""

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

    # Build leagues list: merge configured + observed
    leagues = []
    all_ids = set(COMPETITIONS.keys()) | observed_ids

    for league_id in sorted(all_ids):
        info = get_league_info(league_id)
        obs = observed_data.get(league_id)
        titan = titan_data.get(league_id)

        league_entry = {
            "league_id": league_id,
            "name": info["name"],
            "configured": info["configured"],
            "observed": league_id in observed_ids,
        }

        # Add config fields only if configured
        if info["configured"]:
            league_entry["priority"] = info["priority"]
            league_entry["match_type"] = info["match_type"]
            league_entry["match_weight"] = info["match_weight"]

        # Add stats if observed
        if obs:
            league_entry["stats"] = {
                "total_matches": obs.total_matches,
                "finished_matches": obs.finished,
                "matches_25_26": obs.matches_25_26,
                "with_stats_pct": round(100.0 * obs.with_stats / obs.total_matches, 1) if obs.total_matches else 0,
                "unique_teams": obs.unique_teams,
                "seasons": [obs.first_season, obs.last_season],
                "last_match": obs.last_match.isoformat() if obs.last_match else None,
            }

        # Add TITAN if available
        if titan:
            league_entry["titan"] = titan

        leagues.append(league_entry)

    # Sort: configured first, then by matches
    leagues.sort(key=lambda x: (
        not x["configured"],  # configured first
        -(x.get("stats", {}).get("total_matches", 0))  # then by matches desc
    ))

    # Unmapped observed (in DB but not in COMPETITIONS)
    unmapped = [lid for lid in observed_ids if lid not in COMPETITIONS]

    return {
        "leagues": leagues,
        "totals": {
            "configured": len(COMPETITIONS),
            "observed_in_db": len(observed_ids),
            "with_titan_data": len(titan_data),
        },
        "unmapped_observed": sorted(unmapped),
    }


# =============================================================================
# League Detail
# =============================================================================

async def build_league_detail(session: AsyncSession, league_id: int) -> Optional[dict]:
    """Build detail for a specific league."""

    info = get_league_info(league_id)

    # Check if league has any matches
    check_query = text("SELECT COUNT(*) as cnt FROM matches WHERE league_id = :lid")
    check_result = await session.execute(check_query, {"lid": league_id})
    if check_result.fetchone().cnt == 0 and not info["configured"]:
        return None  # League not found

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
            CASE WHEN m.stats IS NOT NULL AND m.stats::text != '{}' THEN true ELSE false END as has_stats,
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

    return {
        "league": {
            "league_id": league_id,
            "name": info["name"],
            "configured": info["configured"],
            "observed": len(stats_by_season) > 0,
            "priority": info["priority"],
            "match_type": info["match_type"],
            "match_weight": info["match_weight"],
        },
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
    limit: int = 100,
    offset: int = 0,
) -> dict:
    """Build paginated teams list with optional filters."""

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

    where_clause = "WHERE " + " AND ".join(where_parts) if where_parts else ""

    # Count total
    count_query = text(f"""
        SELECT COUNT(*) as total
        FROM teams t
        {where_clause}
    """)
    count_result = await session.execute(count_query, params)
    total = count_result.fetchone().total

    # Get teams with stats
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

async def build_team_detail(session: AsyncSession, team_id: int) -> Optional[dict]:
    """Build detail for a specific team."""

    # Get team info
    team_query = text("""
        SELECT id, external_id, name, country, team_type, logo_url
        FROM teams
        WHERE id = :tid
    """)
    team_result = await session.execute(team_query, {"tid": team_id})
    team_row = team_result.fetchone()

    if not team_row:
        return None

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
        info = get_league_info(r.league_id)
        leagues_played.append({
            "league_id": r.league_id,
            "name": info["name"],
            "matches": r.matches,
            "seasons": [r.first_season, r.last_season],
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
        info = get_league_info(r.league_id)
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

    return {
        "team": {
            "team_id": team_row.id,
            "external_id": team_row.external_id,
            "name": team_row.name,
            "country": team_row.country,
            "team_type": team_row.team_type,
            "logo_url": team_row.logo_url,
        },
        "leagues_played": leagues_played,
        "stats_by_season": stats_by_season,
        "recent_matches": recent_matches,
        "predictions_stats": predictions_stats,
    }
