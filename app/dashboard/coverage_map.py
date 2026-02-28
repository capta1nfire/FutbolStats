"""
Coverage Map calculator — computes data coverage per league/country.

Contract: docs/COVERAGE_MAP_CONTRACT.md
Endpoint: GET /dashboard/coverage-map.json
"""

import logging
from datetime import datetime, timedelta, timezone
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
    "geo",
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
     "source_tables": ["match_weather_canonical"], "pit_guardrail": "forecast: captured_at < match.date; archive: post-hoc (kind column)"},
    {"key": "bio_adaptability", "label": "Bio-Adaptability", "priority": "P1", "contributes_to_score": True,
     "source_tables": ["team_home_city_profile"], "pit_guardrail": "static profile"},
    {"key": "sofascore_xi_ratings", "label": "Sofascore XI Ratings", "priority": "P1", "contributes_to_score": True,
     "source_tables": ["sofascore_player_rating_history"], "pit_guardrail": "post-match ratings (is_starter + rating NOT NULL)"},
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
    {"key": "geo", "label": "Geo (Altitude)", "priority": "P2", "contributes_to_score": True,
     "source_tables": ["team_wikidata_enrichment"], "pit_guardrail": "static profile"},
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
    """Resolve window/season/from/to into (date_from, date_to) strings.

    Returns (None, None) for per-league season modes (current_season,
    prev_season, prev_season_2) — date filtering is done per-league in SQL.
    """
    now = datetime.now(timezone.utc)
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
    if window in ("current_season", "prev_season", "prev_season_2"):
        return None, None  # per-league — SQL handles dates
    # fallback
    return "2023-01-01", now.strftime("%Y-%m-%d")


# ---------------------------------------------------------------------------
# SQL builder
# ---------------------------------------------------------------------------

def _build_coverage_sql(
    league_ids: Optional[List[int]] = None,
    country_names: Optional[List[str]] = None,
    per_league_season: bool = False,
    season_offset: int = 0,
) -> tuple:
    """Build coverage query using FILTER + EXISTS (no CTEs, no LEFT JOINs).

    Returns (sql_string, params_dict).
    Uses bound parameters for all user-supplied values (no string interpolation).

    When per_league_season=True, the date filter uses al.season_start_month
    to compute per-league season boundaries instead of global date_from/date_to.
    """
    extra_where = ""
    params: Dict[str, Any] = {}

    if league_ids:
        extra_where += "\n    AND m.league_id = ANY(:league_ids)"
        params["league_ids"] = league_ids
    if country_names:
        extra_where += "\n    AND al.country = ANY(:country_names)"
        params["country_names"] = country_names

    # Date filter: per-league season mode vs global date range
    if per_league_season:
        # Per-league: season boundaries computed from al.season_start_month.
        # base_year = year when current season started for this league.
        # Uses CURRENT_DATE (date, not timestamptz) and make_date (date) to
        # avoid timestamp/timestamptz coercion with matches.date (timestamp).
        # No LEAST(…, NOW()) needed: status IN ('FT','AET','PEN') already
        # excludes future matches.
        date_filter = """
  AND m.date >= make_date(
      (CASE WHEN EXTRACT(MONTH FROM CURRENT_DATE) >= al.season_start_month
            THEN EXTRACT(YEAR FROM CURRENT_DATE)::int
            ELSE EXTRACT(YEAR FROM CURRENT_DATE)::int - 1 END) - :season_offset,
      al.season_start_month, 1)
  AND m.date < make_date(
      (CASE WHEN EXTRACT(MONTH FROM CURRENT_DATE) >= al.season_start_month
            THEN EXTRACT(YEAR FROM CURRENT_DATE)::int
            ELSE EXTRACT(YEAR FROM CURRENT_DATE)::int - 1 END) - :season_offset + 1,
      al.season_start_month, 1)"""
        params["season_offset"] = season_offset
    else:
        date_filter = """
  AND m.date >= :date_from
  AND m.date < :date_to"""

    sql = f"""
SELECT
  m.league_id,
  COALESCE(al.display_name, al.name) AS league_name,
  al.country,
  al.logo_url,
  al.wikipedia_url,
  al.kind,
  COUNT(*) AS eligible_matches,

  -- ===== P0 =====

  -- xG: Canonical SSOT (match_canonical_xg, excl. quarantined)
  COUNT(*) FILTER (WHERE
    EXISTS (SELECT 1 FROM match_canonical_xg cxg
            WHERE cxg.match_id = m.id AND cxg.source NOT LIKE '%_quarantined')
  ) AS xg_n,

  -- Canonical odds: match_canonical_odds (single source of truth, cascade-resolved)
  COUNT(*) FILTER (WHERE
    EXISTS (SELECT 1 FROM match_canonical_odds co
            WHERE co.match_id = m.id
            AND co.odds_home > 1.0 AND co.odds_draw > 1.0 AND co.odds_away > 1.0)
  ) AS odds_canonical_n,

  -- Canonical closing odds (is_closing=true, FT/AET only)
  COUNT(*) FILTER (WHERE
    EXISTS (SELECT 1 FROM match_canonical_odds co
            WHERE co.match_id = m.id AND co.is_closing = true
            AND co.odds_home > 1.0 AND co.odds_draw > 1.0 AND co.odds_away > 1.0)
  ) AS odds_closing_n,

  -- Legacy opening odds (retained for backward compat, pending DROP)
  COUNT(*) FILTER (WHERE
    (m.opening_odds_home > 1.0 AND m.opening_odds_draw > 1.0 AND m.opening_odds_away > 1.0)
  ) AS odds_opening_n,

  -- Lineups: both sides with >=7 starters
  COUNT(*) FILTER (WHERE EXISTS (
    SELECT 1 FROM (
      SELECT ml.match_id FROM match_lineups ml
      WHERE ml.match_id = m.id AND array_length(ml.starting_xi_ids, 1) >= 7
        AND (ml.lineup_confirmed_at IS NULL OR ml.lineup_confirmed_at < m.date)
      GROUP BY ml.match_id HAVING COUNT(DISTINCT ml.is_home) = 2
    ) sub
  )) AS lineups_n,

  -- ===== P1 =====

  -- Weather: canonical (forecast preferred, archive fallback)
  COUNT(*) FILTER (WHERE EXISTS (
    SELECT 1 FROM match_weather_canonical mwc WHERE mwc.match_id = m.id
  )) AS weather_n,

  -- Bio-adaptability: both teams timezone, away has climate normals
  COUNT(*) FILTER (WHERE
    EXISTS (SELECT 1 FROM team_home_city_profile hp
            WHERE hp.team_id = m.home_team_id AND hp.timezone IS NOT NULL)
    AND EXISTS (SELECT 1 FROM team_home_city_profile ap
                WHERE ap.team_id = m.away_team_id AND ap.timezone IS NOT NULL
                AND ap.climate_normals_by_month IS NOT NULL)
  ) AS bio_adaptability_n,

  -- Sofascore XI: >=11 rated starters per side (post-match ratings from history)
  COUNT(*) FILTER (WHERE EXISTS (
    SELECT 1 FROM (
      SELECT sprh.match_id FROM sofascore_player_rating_history sprh
      WHERE sprh.match_id = m.id AND sprh.is_starter = true AND sprh.rating IS NOT NULL
      GROUP BY sprh.match_id, sprh.team_side
      HAVING COUNT(*) >= 11
    ) sub
    GROUP BY sub.match_id HAVING COUNT(*) = 2
  )) AS sofascore_xi_ratings_n,

  -- External refs: >=1 high-confidence from xG/rating sources
  COUNT(*) FILTER (WHERE EXISTS (
    SELECT 1 FROM match_external_refs mer
    WHERE mer.match_id = m.id AND mer.source IN ('understat','fotmob','sofascore')
    AND mer.confidence >= 0.90
  )) AS external_refs_n,

  -- Freshness: odds pre-kickoff (real-time OR canonical closing) + supporting data
  COUNT(*) FILTER (WHERE
    (
      (m.odds_recorded_at IS NOT NULL AND m.odds_recorded_at < m.date)
      OR EXISTS (SELECT 1 FROM match_canonical_odds co
                 WHERE co.match_id = m.id AND co.is_closing = true
                 AND co.odds_home > 1.0 AND co.odds_draw > 1.0 AND co.odds_away > 1.0)
    )
    AND (
      EXISTS (SELECT 1 FROM match_canonical_xg cxg
              WHERE cxg.match_id = m.id AND cxg.source NOT LIKE '%_quarantined')
      OR EXISTS (SELECT 1 FROM match_lineups ml
                 WHERE ml.match_id = m.id AND array_length(ml.starting_xi_ids, 1) >= 7
                 AND (ml.lineup_confirmed_at IS NULL OR ml.lineup_confirmed_at < m.date))
    )
  ) AS freshness_n,

  -- Join health: >=2 distinct high-confidence sources
  COUNT(*) FILTER (WHERE EXISTS (
    SELECT 1 FROM match_external_refs mer
    WHERE mer.match_id = m.id AND mer.confidence >= 0.90
    GROUP BY mer.match_id HAVING COUNT(DISTINCT mer.source) >= 2
  )) AS join_health_n,

  -- ===== P2 =====

  -- Match stats: non-empty with required keys
  COUNT(*) FILTER (WHERE
    m.stats IS NOT NULL AND m.stats::text NOT IN ('{{}}', 'null', '')
    AND m.stats::text LIKE '%shots_on_goal%' AND m.stats::text LIKE '%corner_kicks%'
  ) AS match_stats_n,

  -- Match events: non-empty
  COUNT(*) FILTER (WHERE
    m.events IS NOT NULL AND m.events::text NOT IN ('[]', 'null', '')
  ) AS match_events_n,

  -- Venue
  COUNT(*) FILTER (WHERE m.venue_name IS NOT NULL AND m.venue_city IS NOT NULL) AS venue_n,

  -- Referees/Coaches: coach presence in lineups
  COUNT(*) FILTER (WHERE EXISTS (
    SELECT 1 FROM match_lineups ml WHERE ml.match_id = m.id AND ml.coach_id IS NOT NULL
  )) AS referees_n,

  -- Player injuries (PIT-safe)
  COUNT(*) FILTER (WHERE EXISTS (
    SELECT 1 FROM player_injuries pi
    WHERE pi.match_id = m.id AND pi.captured_at IS NOT NULL AND pi.captured_at < m.date
  )) AS player_injuries_n,

  -- Managers: both teams have active manager at match date
  COUNT(*) FILTER (WHERE
    EXISTS (SELECT 1 FROM team_manager_history tmh
            WHERE tmh.team_id = m.home_team_id AND tmh.start_date <= m.date::date
            AND (tmh.end_date IS NULL OR tmh.end_date >= m.date::date))
    AND EXISTS (SELECT 1 FROM team_manager_history tmh
                WHERE tmh.team_id = m.away_team_id AND tmh.start_date <= m.date::date
                AND (tmh.end_date IS NULL OR tmh.end_date >= m.date::date))
  ) AS managers_n,

  -- Squad catalog: both teams >=11 players
  COUNT(*) FILTER (WHERE
    (SELECT COUNT(*) FROM players p WHERE p.team_id = m.home_team_id) >= 11
    AND (SELECT COUNT(*) FROM players p WHERE p.team_id = m.away_team_id) >= 11
  ) AS squad_catalog_n,

  -- Standings: season-safe
  COUNT(*) FILTER (WHERE EXISTS (
    SELECT 1 FROM league_standings ls WHERE ls.league_id = m.league_id AND ls.season = m.season
  )) AS standings_n,

  -- Geo: both teams have lat/lon/altitude in team_wikidata_enrichment
  COUNT(*) FILTER (WHERE
    EXISTS (SELECT 1 FROM team_wikidata_enrichment twh
            WHERE twh.team_id = m.home_team_id
            AND twh.lat IS NOT NULL AND twh.lon IS NOT NULL AND twh.stadium_altitude_m IS NOT NULL)
    AND EXISTS (SELECT 1 FROM team_wikidata_enrichment twa
                WHERE twa.team_id = m.away_team_id
                AND twa.lat IS NOT NULL AND twa.lon IS NOT NULL AND twa.stadium_altitude_m IS NOT NULL)
  ) AS geo_n,

  -- ===== INTERSECTIONS (exact, not approximated) =====

  -- Odds AND xG
  COUNT(*) FILTER (WHERE
    (
      (m.odds_home > 1.0 AND m.odds_draw > 1.0 AND m.odds_away > 1.0)
      OR EXISTS (SELECT 1 FROM odds_history ohc
                 WHERE ohc.match_id = m.id AND ohc.is_closing = true
                 AND ohc.odds_home > 1.0 AND ohc.odds_draw > 1.0 AND ohc.odds_away > 1.0
                 AND (ohc.quarantined IS NULL OR ohc.quarantined = false))
    )
    AND (
      EXISTS (SELECT 1 FROM match_canonical_xg cxg
              WHERE cxg.match_id = m.id AND cxg.source NOT LIKE '%_quarantined')
    )
  ) AS odds_xg_n,

  -- Lineups AND Odds AND xG
  COUNT(*) FILTER (WHERE
    (
      (m.odds_home > 1.0 AND m.odds_draw > 1.0 AND m.odds_away > 1.0)
      OR EXISTS (SELECT 1 FROM odds_history ohc
                 WHERE ohc.match_id = m.id AND ohc.is_closing = true
                 AND ohc.odds_home > 1.0 AND ohc.odds_draw > 1.0 AND ohc.odds_away > 1.0
                 AND (ohc.quarantined IS NULL OR ohc.quarantined = false))
    )
    AND (
      EXISTS (SELECT 1 FROM match_canonical_xg cxg
              WHERE cxg.match_id = m.id AND cxg.source NOT LIKE '%_quarantined')
    )
    AND EXISTS (
      SELECT 1 FROM (
        SELECT ml.match_id FROM match_lineups ml
        WHERE ml.match_id = m.id AND array_length(ml.starting_xi_ids, 1) >= 7
          AND (ml.lineup_confirmed_at IS NULL OR ml.lineup_confirmed_at < m.date)
        GROUP BY ml.match_id HAVING COUNT(DISTINCT ml.is_home) = 2
      ) sub
    )
  ) AS xi_odds_xg_n,

  -- ===== DIAGNOSTIC =====

  -- Data quality: not tainted, no quarantined odds
  COUNT(*) FILTER (WHERE
    (m.tainted IS NULL OR m.tainted = false)
    AND NOT EXISTS (SELECT 1 FROM odds_history oh WHERE oh.match_id = m.id AND oh.quarantined = true)
  ) AS data_quality_n

FROM matches m
JOIN admin_leagues al ON m.league_id = al.league_id AND al.is_active = true
WHERE m.status IN ('FT','AET','PEN'){date_filter}{extra_where}
GROUP BY m.league_id, al.display_name, al.name, al.country, al.logo_url, al.wikipedia_url, al.kind
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
        ("odds_canonical_n", "odds_canonical"),
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
        ("geo_n", "geo"),
    ]
    for col, key in col_map:
        n = getattr(row, col, 0) or 0
        dims[key] = {"pct": _pct(n, total), "numerator": n, "denominator": total}

    # Data quality (diagnostic, not in score)
    dq_n = getattr(row, "data_quality_n", 0) or 0
    dims["data_quality_flags"] = {"pct": _pct(dq_n, total), "numerator": dq_n, "denominator": total}

    # Intersection counts (exact from SQL, not approximated)
    odds_xg_n = getattr(row, "odds_xg_n", 0) or 0
    xi_odds_xg_n = getattr(row, "xi_odds_xg_n", 0) or 0

    return {
        "league_id": row.league_id,
        "league_name": row.league_name,
        "country": row.country,
        "country_iso3": COUNTRY_TO_ISO3.get(row.country),
        "logo_url": getattr(row, "logo_url", None),
        "wikipedia_url": getattr(row, "wikipedia_url", None),
        "kind": getattr(row, "kind", "league") or "league",
        "eligible_matches": total,
        "dimensions": dims,
        "_odds_xg_n": odds_xg_n,
        "_xi_odds_xg_n": xi_odds_xg_n,
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

    # Exact intersection percentages from SQL (not approximated with min())
    odds_xg_pct = _pct(league.get("_odds_xg_n", 0), total)
    xi_odds_xg_pct = _pct(league.get("_xi_odds_xg_n", 0), total)

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
                "_odds_xg_n": 0,
                "_xi_odds_xg_n": 0,
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
        c["_odds_xg_n"] += lg.get("_odds_xg_n", 0)
        c["_xi_odds_xg_n"] += lg.get("_xi_odds_xg_n", 0)

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
        odds_xg_pct = _pct(c["_odds_xg_n"], total)
        xi_odds_xg_pct = _pct(c["_xi_odds_xg_n"], total)

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
    # Determine per-league season mode
    per_league_season = window in ("current_season", "prev_season", "prev_season_2")
    season_offset = {"current_season": 0, "prev_season": 1, "prev_season_2": 2}.get(window, 0)

    # Resolve dates (returns None, None for per-league modes)
    resolved_from, resolved_to = resolve_date_range(window, season, date_from, date_to)

    # Map ISO3 codes to country names for SQL filter
    country_names = None
    if country_iso3:
        country_names = [ISO3_TO_COUNTRY.get(c) for c in country_iso3 if c in ISO3_TO_COUNTRY]
        if not country_names:
            country_names = None

    # Build and execute SQL
    sql, params = _build_coverage_sql(
        league_ids=league_ids,
        country_names=country_names,
        per_league_season=per_league_season,
        season_offset=season_offset,
    )
    if not per_league_season:
        params["date_from"] = datetime.strptime(resolved_from, "%Y-%m-%d")
        params["date_to"] = datetime.strptime(resolved_to, "%Y-%m-%d")

    logger.info(
        "coverage_map | window=%s from=%s to=%s per_league=%s offset=%s leagues=%s countries=%s",
        window, resolved_from, resolved_to, per_league_season, season_offset,
        league_ids, country_iso3,
    )

    try:
        # Disable JIT: 19-dim FILTER+EXISTS query triggers excessive JIT compilation
        # (~1.2s overhead for 130 functions). Query runs in ~1.1s without JIT.
        await session.execute(text("SET LOCAL jit = off"))
        result = await session.execute(text(sql), params)
        rows = result.fetchall()
    except Exception as exc:
        logger.error("coverage_map SQL error: %s", str(exc)[:500])
        raise

    # Transform rows into league dicts
    leagues = []
    found_league_ids = set()
    for row in rows:
        lg = _compute_league_data(row)
        lg = _compute_scores(lg)

        # Apply min_matches filter
        if lg["eligible_matches"] < min_matches:
            lg["universe_tier"] = "insufficient_data"

        leagues.append(lg)
        found_league_ids.add(lg["league_id"])

    # Off-season fallback: for current_season (offset=0), calendar-year leagues
    # may have 0 finished matches if the new season hasn't started yet (e.g.
    # Bolivia in Feb 2026). Re-query with offset=1 for missing leagues so they
    # still appear on the map with their most recent completed season.
    if per_league_season and season_offset == 0:
        try:
            active_result = await session.execute(text(
                "SELECT league_id FROM admin_leagues WHERE is_active = true"
            ))
            active_ids = {r.league_id for r in active_result.fetchall()}
            missing_ids = sorted(active_ids - found_league_ids)
        except Exception:
            missing_ids = []

        if missing_ids:
            logger.info("coverage_map | off-season fallback for %d leagues: %s",
                        len(missing_ids), missing_ids[:10])
            fb_sql, fb_params = _build_coverage_sql(
                league_ids=missing_ids,
                country_names=None,
                per_league_season=True,
                season_offset=1,
            )
            fb_params["season_offset"] = 1
            try:
                fb_result = await session.execute(text(fb_sql), fb_params)
                for row in fb_result.fetchall():
                    lg = _compute_league_data(row)
                    lg = _compute_scores(lg)
                    if lg["eligible_matches"] < min_matches:
                        lg["universe_tier"] = "insufficient_data"
                    leagues.append(lg)
            except Exception as exc:
                logger.warning("coverage_map fallback query error: %s", str(exc)[:200])

    # Aggregate by country (BEFORE stripping data_quality_flags — aggregation needs it)
    countries = _aggregate_by_country(leagues)

    # Strip quality flags AFTER aggregation
    if not include_quality_flags:
        for lg in leagues:
            lg["dimensions"].pop("data_quality_flags", None)
        for c in countries:
            c["dimensions"].pop("data_quality_flags", None)

    # Strip internal intersection counts (not part of the API contract)
    for lg in leagues:
        lg.pop("_odds_xg_n", None)
        lg.pop("_xi_odds_xg_n", None)

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
