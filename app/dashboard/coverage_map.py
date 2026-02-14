"""
Coverage Map calculator — computes data coverage per league/country.

Contract: docs/COVERAGE_MAP_CONTRACT.md
Endpoint: GET /dashboard/coverage-map.json
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

COUNTRY_TO_ISO3 = {
    "Argentina": "ARG",
    "Belgium": "BEL",
    "Bolivia": "BOL",
    "Brazil": "BRA",
    "Chile": "CHL",
    "Colombia": "COL",
    "Ecuador": "ECU",
    "England": "GBR",
    "France": "FRA",
    "Germany": "DEU",
    "Italy": "ITA",
    "Mexico": "MEX",
    "Netherlands": "NLD",
    "Paraguay": "PRY",
    "Peru": "PER",
    "Portugal": "PRT",
    "Saudi-Arabia": "SAU",
    "Spain": "ESP",
    "Turkey": "TUR",
    "Uruguay": "URY",
    "USA": "USA",
    "Venezuela": "VEN",
}

ISO3_TO_COUNTRY = {v: k for k, v in COUNTRY_TO_ISO3.items()}

TIER_WEIGHTS = {"p0": 0.60, "p1": 0.25, "p2": 0.15}

P0_KEYS = ["xg", "odds_closing", "odds_opening", "lineups"]
P1_KEYS = [
    "weather", "bio_adaptability", "sofascore_xi_ratings",
    "external_refs", "freshness", "join_health",
]
P2_KEYS = [
    "match_stats", "match_events", "venue", "referees",
    "player_injuries", "managers", "squad_catalog", "standings",
]

ALL_DIM_KEYS = P0_KEYS + P1_KEYS + P2_KEYS

DIMENSIONS_META = [
    {"key": "xg", "label": "xG", "priority": "P0", "contributes_to_score": True,
     "source_tables": ["match_understat_team", "match_fotmob_stats"], "pit_guardrail": "data presence (post-match by nature)"},
    {"key": "odds_closing", "label": "Closing Odds", "priority": "P0", "contributes_to_score": True,
     "source_tables": ["matches", "odds_history"], "pit_guardrail": "data presence (PIT via freshness dim)"},
    {"key": "odds_opening", "label": "Opening Odds", "priority": "P0", "contributes_to_score": True,
     "source_tables": ["matches", "odds_history"], "pit_guardrail": "data presence (PIT via freshness dim)"},
    {"key": "lineups", "label": "Lineups", "priority": "P0", "contributes_to_score": True,
     "source_tables": ["match_lineups"], "pit_guardrail": "lineup_confirmed_at < match.date"},
    {"key": "weather", "label": "Weather", "priority": "P1", "contributes_to_score": True,
     "source_tables": ["match_weather"], "pit_guardrail": "captured_at < match.date"},
    {"key": "bio_adaptability", "label": "Bio-Adaptability", "priority": "P1", "contributes_to_score": True,
     "source_tables": ["team_home_city_profile"], "pit_guardrail": "static profile"},
    {"key": "sofascore_xi_ratings", "label": "Sofascore XI Ratings", "priority": "P1", "contributes_to_score": True,
     "source_tables": ["match_sofascore_player"], "pit_guardrail": "data presence"},
    {"key": "external_refs", "label": "External Refs", "priority": "P1", "contributes_to_score": True,
     "source_tables": ["match_external_refs"], "pit_guardrail": "confidence >= 0.90"},
    {"key": "freshness", "label": "Data Freshness", "priority": "P1", "contributes_to_score": True,
     "source_tables": ["matches", "match_understat_team", "match_fotmob_stats", "match_lineups"],
     "pit_guardrail": "timeliness meta-dimension"},
    {"key": "join_health", "label": "Join Health", "priority": "P1", "contributes_to_score": True,
     "source_tables": ["match_external_refs"], "pit_guardrail": ">=2 distinct sources with confidence >= 0.90"},
    {"key": "match_stats", "label": "Match Stats", "priority": "P2", "contributes_to_score": True,
     "source_tables": ["matches.stats"], "pit_guardrail": "none"},
    {"key": "match_events", "label": "Match Events", "priority": "P2", "contributes_to_score": True,
     "source_tables": ["matches.events"], "pit_guardrail": "none"},
    {"key": "venue", "label": "Venue", "priority": "P2", "contributes_to_score": True,
     "source_tables": ["matches"], "pit_guardrail": "none"},
    {"key": "referees", "label": "Referees/Coaches", "priority": "P2", "contributes_to_score": True,
     "source_tables": ["match_lineups.coach_id"], "pit_guardrail": "none"},
    {"key": "player_injuries", "label": "Player Injuries", "priority": "P2", "contributes_to_score": True,
     "source_tables": ["player_injuries"], "pit_guardrail": "captured_at < match.date"},
    {"key": "managers", "label": "Managers", "priority": "P2", "contributes_to_score": True,
     "source_tables": ["team_manager_history"], "pit_guardrail": "start_date <= match.date"},
    {"key": "squad_catalog", "label": "Squad Catalog", "priority": "P2", "contributes_to_score": True,
     "source_tables": ["players"], "pit_guardrail": "none"},
    {"key": "standings", "label": "Standings", "priority": "P2", "contributes_to_score": True,
     "source_tables": ["league_standings"], "pit_guardrail": "season = match.season"},
    {"key": "data_quality_flags", "label": "Data Quality", "priority": "DIAG", "contributes_to_score": False,
     "source_tables": ["matches.tainted", "odds_history.quarantined"], "pit_guardrail": "diagnostic only"},
]

COLOR_SCALE = [
    {"min": 0, "max": 24.9, "color": "#7f1d1d"},
    {"min": 25, "max": 49.9, "color": "#b45309"},
    {"min": 50, "max": 69.9, "color": "#0369a1"},
    {"min": 70, "max": 84.9, "color": "#15803d"},
    {"min": 85, "max": 100, "color": "#22c55e"},
]


# ---------------------------------------------------------------------------
# Date range resolution
# ---------------------------------------------------------------------------

def resolve_date_range(
    window: str,
    season: Optional[int] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
) -> tuple:
    """Resolve window/season/from/to into (date_from, date_to) strings."""
    now = datetime.utcnow()
    if window == "custom":
        return date_from, date_to
    if window == "since_2023":
        return "2023-01-01", now.strftime("%Y-%m-%d")
    if window == "last_365d":
        start = now - timedelta(days=365)
        return start.strftime("%Y-%m-%d"), now.strftime("%Y-%m-%d")
    if window == "season_to_date":
        if season is None:
            season = now.year if now.month >= 7 else now.year - 1
        return f"{season}-07-01", now.strftime("%Y-%m-%d")
    # fallback
    return "2023-01-01", now.strftime("%Y-%m-%d")


# ---------------------------------------------------------------------------
# SQL builder
# ---------------------------------------------------------------------------

def _build_coverage_sql(
    league_ids: Optional[List[int]] = None,
    country_names: Optional[List[str]] = None,
) -> tuple:
    """Build the big CTE query. Returns (sql_string, params_dict).

    Uses bound parameters for all user-supplied values (no string interpolation).
    """
    # Dynamic WHERE clauses for optional filters
    extra_where = ""
    params: Dict[str, Any] = {}

    if league_ids:
        extra_where += "\n    AND m.league_id = ANY(:league_ids)"
        params["league_ids"] = league_ids
    if country_names:
        extra_where += "\n    AND al.country = ANY(:country_names)"
        params["country_names"] = country_names

    sql = f"""
WITH eligible AS (
  SELECT m.id, m.league_id, m.season, m.home_team_id, m.away_team_id, m.date,
         m.odds_home, m.odds_draw, m.odds_away, m.odds_recorded_at,
         m.opening_odds_home, m.opening_odds_draw, m.opening_odds_away, m.opening_odds_recorded_at,
         m.stats, m.events, m.venue_name, m.venue_city, m.tainted
  FROM matches m
  JOIN admin_leagues al ON m.league_id = al.league_id AND al.is_active = true
  WHERE m.status IN ('FT','AET','PEN')
    AND m.date >= CAST(:date_from AS timestamp)
    AND m.date < CAST(:date_to AS timestamp){extra_where}
),

-- ========== P0 ==========

-- xG: Understat OR FotMob — data presence (xG is post-match, PIT N/A)
dim_xg AS (
  SELECT DISTINCT e.id AS match_id
  FROM eligible e
  LEFT JOIN match_understat_team ust ON ust.match_id = e.id
  LEFT JOIN match_fotmob_stats fmt
    ON fmt.match_id = e.id
    AND fmt.xg_home IS NOT NULL AND fmt.xg_away IS NOT NULL
  WHERE ust.match_id IS NOT NULL OR fmt.match_id IS NOT NULL
),

-- Closing odds: matches table OR odds_history(is_closing) — data presence
dim_odds_closing AS (
  SELECT DISTINCT e.id AS match_id FROM eligible e
  LEFT JOIN odds_history ohc
    ON ohc.match_id = e.id AND ohc.is_closing = true
    AND ohc.odds_home > 1.0 AND ohc.odds_draw > 1.0 AND ohc.odds_away > 1.0
    AND (ohc.quarantined IS NULL OR ohc.quarantined = false)
  WHERE (e.odds_home > 1.0 AND e.odds_draw > 1.0 AND e.odds_away > 1.0)
     OR ohc.match_id IS NOT NULL
),

-- Opening odds: matches table OR odds_history(is_opening) — data presence
dim_odds_opening AS (
  SELECT DISTINCT e.id AS match_id FROM eligible e
  LEFT JOIN odds_history oho
    ON oho.match_id = e.id AND oho.is_opening = true
    AND oho.odds_home > 1.0 AND oho.odds_draw > 1.0 AND oho.odds_away > 1.0
    AND (oho.quarantined IS NULL OR oho.quarantined = false)
  WHERE (e.opening_odds_home > 1.0 AND e.opening_odds_draw > 1.0 AND e.opening_odds_away > 1.0)
     OR oho.match_id IS NOT NULL
),

-- Lineups: both sides with >=7 starters, PIT-safe, robust to duplicates
dim_lineups AS (
  SELECT ml.match_id
  FROM match_lineups ml
  JOIN eligible e ON e.id = ml.match_id
  WHERE array_length(ml.starting_xi_ids, 1) >= 7
    AND (ml.lineup_confirmed_at IS NULL OR ml.lineup_confirmed_at < e.date)
  GROUP BY ml.match_id
  HAVING COUNT(DISTINCT ml.is_home) = 2
),

-- ========== P1 ==========

-- Weather: exists PIT-safe
dim_weather AS (
  SELECT DISTINCT mw.match_id
  FROM match_weather mw
  JOIN eligible e ON e.id = mw.match_id
  WHERE mw.captured_at < e.date
),

-- Bio-adaptability: both teams have timezone, away has climate normals
dim_bio AS (
  SELECT e.id AS match_id FROM eligible e
  JOIN team_home_city_profile hp ON hp.team_id = e.home_team_id AND hp.timezone IS NOT NULL
  JOIN team_home_city_profile ap ON ap.team_id = e.away_team_id AND ap.timezone IS NOT NULL
    AND ap.climate_normals_by_month IS NOT NULL
),

-- Sofascore XI: >=11 rated starters per side — data presence
dim_sofascore AS (
  SELECT match_id FROM (
    SELECT msp.match_id, msp.team_side,
           COUNT(*) FILTER (
             WHERE msp.is_starter = true
               AND (msp.rating_pre_match IS NOT NULL OR msp.rating_recent_form IS NOT NULL)
           ) AS rated_starters
    FROM match_sofascore_player msp
    JOIN eligible e ON e.id = msp.match_id
    GROUP BY msp.match_id, msp.team_side
  ) sub
  WHERE rated_starters >= 11
  GROUP BY match_id
  HAVING COUNT(*) = 2
),

-- External refs: at least 1 high-confidence ref from xG/rating sources
dim_ext_refs AS (
  SELECT DISTINCT mer.match_id
  FROM match_external_refs mer
  JOIN eligible e ON e.id = mer.match_id
  WHERE mer.source IN ('understat', 'fotmob', 'sofascore')
    AND mer.confidence >= 0.90
),

-- Freshness: odds captured pre-kickoff AND (xG captured OR lineups confirmed) pre-kickoff
-- This measures real-time pipeline timeliness, NOT data presence
dim_freshness AS (
  SELECT e.id AS match_id FROM eligible e
  WHERE e.odds_recorded_at IS NOT NULL AND e.odds_recorded_at < e.date
    AND (
      EXISTS (
        SELECT 1 FROM match_understat_team ust
        WHERE ust.match_id = e.id AND ust.captured_at < e.date
      )
      OR EXISTS (
        SELECT 1 FROM match_fotmob_stats fmt
        WHERE fmt.match_id = e.id AND fmt.xg_home IS NOT NULL AND fmt.captured_at < e.date
      )
      OR EXISTS (
        SELECT 1 FROM match_lineups ml
        WHERE ml.match_id = e.id AND array_length(ml.starting_xi_ids, 1) >= 7
          AND (ml.lineup_confirmed_at IS NULL OR ml.lineup_confirmed_at < e.date)
      )
    )
),

-- Join health: >=2 distinct high-confidence sources (cross-provider breadth)
dim_join_health AS (
  SELECT mer.match_id
  FROM match_external_refs mer
  JOIN eligible e ON e.id = mer.match_id
  WHERE mer.confidence >= 0.90
  GROUP BY mer.match_id
  HAVING COUNT(DISTINCT mer.source) >= 2
),

-- ========== P2 ==========

-- Match stats: non-empty JSON with required keys
dim_stats AS (
  SELECT e.id AS match_id FROM eligible e
  WHERE e.stats IS NOT NULL
    AND e.stats::text NOT IN ('{{}}', 'null', '')
    AND e.stats::text LIKE '%shots_on_goal%'
    AND e.stats::text LIKE '%corner_kicks%'
),

-- Match events: non-empty
dim_events AS (
  SELECT e.id AS match_id FROM eligible e
  WHERE e.events IS NOT NULL
    AND e.events::text NOT IN ('[]', 'null', '')
),

-- Venue: both name and city
dim_venue AS (
  SELECT e.id AS match_id FROM eligible e
  WHERE e.venue_name IS NOT NULL AND e.venue_city IS NOT NULL
),

-- Referees: fallback to coach presence in lineups (no dedicated referee columns)
dim_referees AS (
  SELECT DISTINCT ml.match_id
  FROM match_lineups ml
  JOIN eligible e ON e.id = ml.match_id
  WHERE ml.coach_id IS NOT NULL
),

-- Player injuries: at least 1 injury record PIT-safe
dim_injuries AS (
  SELECT DISTINCT pi.match_id
  FROM player_injuries pi
  JOIN eligible e ON e.id = pi.match_id
  WHERE pi.match_id IS NOT NULL
    AND pi.captured_at IS NOT NULL AND pi.captured_at < e.date
),

-- Managers: both teams have active manager at match date
dim_managers AS (
  SELECT e.id AS match_id FROM eligible e
  WHERE EXISTS (
    SELECT 1 FROM team_manager_history tmh
    WHERE tmh.team_id = e.home_team_id
      AND tmh.start_date <= e.date::date
      AND (tmh.end_date IS NULL OR tmh.end_date >= e.date::date)
  ) AND EXISTS (
    SELECT 1 FROM team_manager_history tmh
    WHERE tmh.team_id = e.away_team_id
      AND tmh.start_date <= e.date::date
      AND (tmh.end_date IS NULL OR tmh.end_date >= e.date::date)
  )
),

-- Squad catalog: both teams have >=11 players
dim_squad AS (
  SELECT e.id AS match_id FROM eligible e
  WHERE (SELECT COUNT(*) FROM players p WHERE p.team_id = e.home_team_id) >= 11
    AND (SELECT COUNT(*) FROM players p WHERE p.team_id = e.away_team_id) >= 11
),

-- Standings: season-safe (PIT N/A — snapshot data backfilled)
dim_standings AS (
  SELECT DISTINCT e.id AS match_id
  FROM eligible e
  JOIN league_standings ls
    ON ls.league_id = e.league_id
    AND ls.season = e.season
),

-- ========== DIAGNOSTIC ==========

-- Data quality: not tainted AND no quarantined odds
dim_quality AS (
  SELECT e.id AS match_id FROM eligible e
  WHERE (e.tainted IS NULL OR e.tainted = false)
    AND NOT EXISTS (
      SELECT 1 FROM odds_history oh
      WHERE oh.match_id = e.id AND oh.quarantined = true
    )
)

-- ========== FINAL AGGREGATION ==========
SELECT
  e.league_id,
  al.name AS league_name,
  al.country,
  COUNT(*) AS eligible_matches,
  -- P0 numerators
  COUNT(d_xg.match_id) AS xg_n,
  COUNT(d_oc.match_id) AS odds_closing_n,
  COUNT(d_oo.match_id) AS odds_opening_n,
  COUNT(d_lu.match_id) AS lineups_n,
  -- P1 numerators
  COUNT(d_we.match_id) AS weather_n,
  COUNT(d_bi.match_id) AS bio_adaptability_n,
  COUNT(d_sc.match_id) AS sofascore_xi_ratings_n,
  COUNT(d_er.match_id) AS external_refs_n,
  COUNT(d_fr.match_id) AS freshness_n,
  COUNT(d_jh.match_id) AS join_health_n,
  -- P2 numerators
  COUNT(d_st.match_id) AS match_stats_n,
  COUNT(d_ev.match_id) AS match_events_n,
  COUNT(d_ve.match_id) AS venue_n,
  COUNT(d_re.match_id) AS referees_n,
  COUNT(d_in.match_id) AS player_injuries_n,
  COUNT(d_mg.match_id) AS managers_n,
  COUNT(d_sq.match_id) AS squad_catalog_n,
  COUNT(d_sd.match_id) AS standings_n,
  -- Diagnostic
  COUNT(d_dq.match_id) AS data_quality_n
FROM eligible e
JOIN admin_leagues al ON al.league_id = e.league_id
LEFT JOIN dim_xg d_xg ON d_xg.match_id = e.id
LEFT JOIN dim_odds_closing d_oc ON d_oc.match_id = e.id
LEFT JOIN dim_odds_opening d_oo ON d_oo.match_id = e.id
LEFT JOIN dim_lineups d_lu ON d_lu.match_id = e.id
LEFT JOIN dim_weather d_we ON d_we.match_id = e.id
LEFT JOIN dim_bio d_bi ON d_bi.match_id = e.id
LEFT JOIN dim_sofascore d_sc ON d_sc.match_id = e.id
LEFT JOIN dim_ext_refs d_er ON d_er.match_id = e.id
LEFT JOIN dim_freshness d_fr ON d_fr.match_id = e.id
LEFT JOIN dim_join_health d_jh ON d_jh.match_id = e.id
LEFT JOIN dim_stats d_st ON d_st.match_id = e.id
LEFT JOIN dim_events d_ev ON d_ev.match_id = e.id
LEFT JOIN dim_venue d_ve ON d_ve.match_id = e.id
LEFT JOIN dim_referees d_re ON d_re.match_id = e.id
LEFT JOIN dim_injuries d_in ON d_in.match_id = e.id
LEFT JOIN dim_managers d_mg ON d_mg.match_id = e.id
LEFT JOIN dim_squad d_sq ON d_sq.match_id = e.id
LEFT JOIN dim_standings d_sd ON d_sd.match_id = e.id
LEFT JOIN dim_quality d_dq ON d_dq.match_id = e.id
GROUP BY e.league_id, al.name, al.country
ORDER BY COUNT(*) DESC
"""
    return sql, params


# ---------------------------------------------------------------------------
# Score & tier computation
# ---------------------------------------------------------------------------

def _pct(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return round(100.0 * numerator / denominator, 1)


def _compute_league_data(row) -> dict:
    """Transform a SQL row into a league coverage dict with all dimensions."""
    total = row.eligible_matches

    dims = {}
    # Map SQL column suffixes to dimension keys
    col_map = [
        ("xg_n", "xg"),
        ("odds_closing_n", "odds_closing"),
        ("odds_opening_n", "odds_opening"),
        ("lineups_n", "lineups"),
        ("weather_n", "weather"),
        ("bio_adaptability_n", "bio_adaptability"),
        ("sofascore_xi_ratings_n", "sofascore_xi_ratings"),
        ("external_refs_n", "external_refs"),
        ("freshness_n", "freshness"),
        ("join_health_n", "join_health"),
        ("match_stats_n", "match_stats"),
        ("match_events_n", "match_events"),
        ("venue_n", "venue"),
        ("referees_n", "referees"),
        ("player_injuries_n", "player_injuries"),
        ("managers_n", "managers"),
        ("squad_catalog_n", "squad_catalog"),
        ("standings_n", "standings"),
    ]
    for col, key in col_map:
        n = getattr(row, col, 0) or 0
        dims[key] = {"pct": _pct(n, total), "numerator": n, "denominator": total}

    # Data quality (diagnostic, not in score)
    dq_n = getattr(row, "data_quality_n", 0) or 0
    dims["data_quality_flags"] = {"pct": _pct(dq_n, total), "numerator": dq_n, "denominator": total}

    return {
        "league_id": row.league_id,
        "league_name": row.league_name,
        "country": row.country,
        "country_iso3": COUNTRY_TO_ISO3.get(row.country),
        "eligible_matches": total,
        "dimensions": dims,
    }


def _compute_scores(league: dict) -> dict:
    """Calculate p0/p1/p2/total scores and universe tier."""
    dims = league["dimensions"]
    total = league["eligible_matches"]

    # Per-tier averages
    p0_pct = sum(dims[k]["pct"] for k in P0_KEYS) / len(P0_KEYS)
    p1_pct = sum(dims[k]["pct"] for k in P1_KEYS) / len(P1_KEYS)
    p2_pct = sum(dims[k]["pct"] for k in P2_KEYS) / len(P2_KEYS)
    coverage_total = (
        TIER_WEIGHTS["p0"] * p0_pct
        + TIER_WEIGHTS["p1"] * p1_pct
        + TIER_WEIGHTS["p2"] * p2_pct
    )

    league["p0_pct"] = round(p0_pct, 1)
    league["p1_pct"] = round(p1_pct, 1)
    league["p2_pct"] = round(p2_pct, 1)
    league["coverage_total_pct"] = round(coverage_total, 1)

    # Universe coverage
    odds_pct = dims["odds_closing"]["pct"]
    xg_pct = dims["xg"]["pct"]
    lineups_pct = dims["lineups"]["pct"]

    # Intersection percentages (approximate using min since we don't have exact intersections)
    # For precise values we'd need additional SQL, but min is a good lower bound
    odds_xg_pct = min(odds_pct, xg_pct)
    xi_odds_xg_pct = min(lineups_pct, odds_pct, xg_pct)

    league["universe_coverage"] = {
        "base_pct": 100.0,
        "odds_pct": round(odds_pct, 1),
        "xg_pct": round(xg_pct, 1),
        "odds_xg_pct": round(odds_xg_pct, 1),
        "xi_odds_xg_pct": round(xi_odds_xg_pct, 1),
    }

    # Universe tier assignment
    if total == 0:
        league["universe_tier"] = "insufficient_data"
    elif xi_odds_xg_pct >= 70:
        league["universe_tier"] = "xi_odds_xg"
    elif odds_xg_pct >= 70:
        league["universe_tier"] = "odds_xg"
    elif xg_pct >= 70:
        league["universe_tier"] = "xg"
    elif odds_pct >= 70:
        league["universe_tier"] = "odds"
    else:
        league["universe_tier"] = "base"

    return league


def _aggregate_by_country(leagues: List[dict]) -> List[dict]:
    """Aggregate league data into country-level summaries (weighted by eligible_matches)."""
    by_country = {}
    for lg in leagues:
        iso3 = lg.get("country_iso3")
        if not iso3:
            continue
        if iso3 not in by_country:
            by_country[iso3] = {
                "country_iso3": iso3,
                "country_name": lg["country"],
                "league_count": 0,
                "eligible_matches": 0,
                "_weighted_total": 0.0,
                "_weighted_p0": 0.0,
                "_weighted_p1": 0.0,
                "_weighted_p2": 0.0,
                "_dim_numerators": {k: 0 for k in ALL_DIM_KEYS},
                "_dim_numerators_dq": 0,
                "_dim_denominator": 0,
            }
        c = by_country[iso3]
        n = lg["eligible_matches"]
        c["league_count"] += 1
        c["eligible_matches"] += n
        c["_weighted_total"] += lg["coverage_total_pct"] * n
        c["_weighted_p0"] += lg["p0_pct"] * n
        c["_weighted_p1"] += lg["p1_pct"] * n
        c["_weighted_p2"] += lg["p2_pct"] * n
        c["_dim_denominator"] += n
        for k in ALL_DIM_KEYS:
            c["_dim_numerators"][k] += lg["dimensions"][k]["numerator"]
        c["_dim_numerators_dq"] += lg["dimensions"]["data_quality_flags"]["numerator"]

    countries = []
    for iso3, c in sorted(by_country.items(), key=lambda x: x[1]["eligible_matches"], reverse=True):
        total = c["eligible_matches"]
        country = {
            "country_iso3": iso3,
            "country_name": c["country_name"],
            "league_count": c["league_count"],
            "eligible_matches": total,
            "coverage_total_pct": round(c["_weighted_total"] / total, 1) if total else 0.0,
            "p0_pct": round(c["_weighted_p0"] / total, 1) if total else 0.0,
            "p1_pct": round(c["_weighted_p1"] / total, 1) if total else 0.0,
            "p2_pct": round(c["_weighted_p2"] / total, 1) if total else 0.0,
        }
        # Compute universe tier from aggregated P0 dims
        odds_pct = _pct(c["_dim_numerators"]["odds_closing"], total)
        xg_pct = _pct(c["_dim_numerators"]["xg"], total)
        lu_pct = _pct(c["_dim_numerators"]["lineups"], total)
        odds_xg_pct = min(odds_pct, xg_pct)
        xi_odds_xg_pct = min(lu_pct, odds_pct, xg_pct)

        country["universe_tier"] = "base"
        if xi_odds_xg_pct >= 70:
            country["universe_tier"] = "xi_odds_xg"
        elif odds_xg_pct >= 70:
            country["universe_tier"] = "odds_xg"
        elif xg_pct >= 70:
            country["universe_tier"] = "xg"
        elif odds_pct >= 70:
            country["universe_tier"] = "odds"

        country["universe_coverage"] = {
            "base_pct": 100.0,
            "odds_pct": round(odds_pct, 1),
            "xg_pct": round(xg_pct, 1),
            "odds_xg_pct": round(odds_xg_pct, 1),
            "xi_odds_xg_pct": round(xi_odds_xg_pct, 1),
        }

        # Country-level dimension summary (aggregated numerators)
        dims = {}
        for k in ALL_DIM_KEYS:
            n = c["_dim_numerators"][k]
            dims[k] = {"pct": _pct(n, total), "numerator": n, "denominator": total}
        dims["data_quality_flags"] = {
            "pct": _pct(c["_dim_numerators_dq"], total),
            "numerator": c["_dim_numerators_dq"],
            "denominator": total,
        }
        country["dimensions"] = dims
        countries.append(country)

    return countries


# ---------------------------------------------------------------------------
# Main calculator
# ---------------------------------------------------------------------------

async def build_coverage_map(
    session: AsyncSession,
    window: str = "since_2023",
    season: Optional[int] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    league_ids: Optional[List[int]] = None,
    country_iso3: Optional[List[str]] = None,
    group_by: str = "country",
    min_matches: int = 30,
    include_leagues: bool = True,
    include_quality_flags: bool = True,
) -> dict:
    """Build the full coverage map response."""
    # Resolve dates
    resolved_from, resolved_to = resolve_date_range(window, season, date_from, date_to)

    # Map ISO3 codes to country names for SQL filter
    country_names = None
    if country_iso3:
        country_names = [ISO3_TO_COUNTRY.get(c) for c in country_iso3 if c in ISO3_TO_COUNTRY]
        if not country_names:
            country_names = None

    # Build and execute SQL
    sql, params = _build_coverage_sql(league_ids=league_ids, country_names=country_names)
    params["date_from"] = resolved_from
    params["date_to"] = resolved_to

    logger.info(
        "coverage_map | window=%s from=%s to=%s leagues=%s countries=%s",
        window, resolved_from, resolved_to, league_ids, country_iso3,
    )

    result = await session.execute(text(sql), params)
    rows = result.fetchall()

    # Transform rows into league dicts
    leagues = []
    for row in rows:
        lg = _compute_league_data(row)
        lg = _compute_scores(lg)

        # Apply min_matches filter
        if lg["eligible_matches"] < min_matches:
            lg["universe_tier"] = "insufficient_data"

        # Strip quality flags from dimensions if not requested
        if not include_quality_flags:
            lg["dimensions"].pop("data_quality_flags", None)

        leagues.append(lg)

    # Aggregate by country
    countries = _aggregate_by_country(leagues)

    # Build response
    response = {
        "contract_version": "coverage-map.v1",
        "request": {
            "window": window,
            "from": resolved_from,
            "to": resolved_to,
            "group_by": group_by,
            "league_ids": league_ids or [],
            "country_iso3": country_iso3 or [],
            "min_matches": min_matches,
            "include_leagues": include_leagues,
            "include_quality_flags": include_quality_flags,
        },
        "weights": TIER_WEIGHTS,
        "dimensions": DIMENSIONS_META,
        "color_scale": COLOR_SCALE,
        "countries": countries,
    }

    if include_leagues:
        response["leagues"] = leagues

    response["summary"] = {
        "countries": len(countries),
        "leagues": len(leagues),
        "eligible_matches": sum(lg["eligible_matches"] for lg in leagues),
        "coverage_total_pct_mean": (
            round(sum(lg["coverage_total_pct"] for lg in leagues) / len(leagues), 1)
            if leagues else 0.0
        ),
    }

    return response
