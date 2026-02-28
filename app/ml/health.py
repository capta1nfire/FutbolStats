"""
ML Health Dashboard - Core Logic

ATI v1.1: Queries SOTA stats coverage + TITAN coverage por season/league,
freshness con age_hours_now, fail-soft por sección.

Endpoint: GET /dashboard/ml_health.json
"""

import logging
from datetime import datetime, timezone
from typing import Any, Callable, Coroutine

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.etl.sota_constants import UNDERSTAT_SUPPORTED_LEAGUES

logger = logging.getLogger(__name__)

# Season definitions (fixed ranges - European football seasons Aug-Jul)
SEASONS = {
    "23/24": ("2023-08-01", "2024-08-01"),
    "24/25": ("2024-08-01", "2025-08-01"),
    "25/26": ("2025-08-01", "2026-08-01"),
}

# Top 5 European leagues
TOP_LEAGUES = [140, 39, 135, 78, 61]  # La Liga, PL, Serie A, Bundesliga, Ligue 1

# League name mapping (no leagues table exists)
LEAGUE_NAMES = {
    # UEFA Competitions
    2: "Champions League",
    3: "Europa League",
    848: "Conference League",
    # Top 5 European Leagues
    39: "Premier League",
    140: "La Liga",
    135: "Serie A",
    78: "Bundesliga",
    61: "Ligue 1",
    # Other European Leagues
    40: "Championship",
    94: "Primeira Liga",
    88: "Eredivisie",
    203: "Super Lig",
    45: "FA Cup",
    143: "Copa del Rey",
    307: "Saudi Pro League",
    # South America
    71: "Brasil Serie A",
    128: "Argentina Primera",
    239: "Colombia Primera A",
    242: "Ecuador Liga Pro",
    250: "Paraguay Primera - Apertura",
    252: "Paraguay Primera - Clausura",
    253: "MLS",
    262: "Liga MX",
    265: "Chile Primera División",
    268: "Uruguay Primera - Apertura",
    270: "Uruguay Primera - Clausura",
    281: "Peru Primera División",
    299: "Venezuela Primera División",
    344: "Bolivia Primera División",
    # World Cup Qualifiers
    10: "Amistosos",
    29: "WCQ CAF",
    30: "WCQ AFC",
    31: "WCQ CONCACAF",
    32: "WCQ UEFA",
    33: "WCQ OFC",
    34: "WCQ CONMEBOL",
    37: "WCQ Intercontinental Play-offs",
    # Other
    1: "World Cup",
    4: "Euro",
    11: "Copa America",
    13: "Nations League",
}


async def build_ml_health_data(session: AsyncSession) -> dict:
    """
    Build complete ML health data with fail-soft per section.

    Returns dict with:
    - generated_at: ISO timestamp
    - health: "ok" | "partial" | "error"
    - data: all health metrics
    """
    degraded_sections: list[str] = []
    data: dict[str, Any] = {}

    # 1. SOTA Stats Coverage (P0 - causa raíz del "vuelo a ciegas")
    data["sota_stats_coverage"] = await _safe_query(
        "sota_stats_coverage",
        lambda: _query_sota_stats_coverage(session),
        {"by_season": {}, "by_league": [], "status": "unknown"},
        degraded_sections,
    )

    # 2. TITAN Coverage por season
    data["titan_coverage"] = await _safe_query(
        "titan_coverage",
        lambda: _query_titan_coverage(session),
        {"by_season": {}, "by_league": [], "status": "unknown"},
        degraded_sections,
    )

    # 3. PIT Compliance
    data["pit_compliance"] = await _safe_query(
        "pit_compliance",
        lambda: _query_pit_compliance(session),
        {"total_rows": 0, "violations": 0, "violation_pct": 0.0, "status": "unknown"},
        degraded_sections,
    )

    # 4. Freshness (age_hours_now + lead_time_hours)
    data["freshness"] = await _safe_query(
        "freshness",
        lambda: _query_freshness(session),
        {"age_hours_now": {}, "lead_time_hours": {}, "historical_7d": {"age_hours_now": {}, "lead_time_hours": {}}, "status": "unknown"},
        degraded_sections,
    )

    # 5. Prediction Confidence
    data["prediction_confidence"] = await _safe_query(
        "prediction_confidence",
        lambda: _query_prediction_confidence(session),
        {"entropy": {}, "tier_distribution": {}, "sample_n": 0, "window_days": 30},
        degraded_sections,
    )

    # 6. Top Regressions (placeholder - requires baseline snapshot)
    data["top_regressions"] = {
        "status": "not_ready",
        "note": "Requires baseline snapshot - will compare current vs previous window after 48h of data",
    }

    # 7. Compute Fuel Gauge
    data["fuel_gauge"] = _compute_fuel_gauge(data, degraded_sections)

    # 8. Overall health
    health = "ok"
    if degraded_sections:
        health = "partial"
    if data["fuel_gauge"]["status"] == "error":
        health = "error"

    return {
        "generated_at": datetime.now(timezone.utc).isoformat() + "Z",
        "health": health,
        "data": data,
    }


async def _safe_query(
    section_name: str,
    query_fn: Callable[[], Coroutine[Any, Any, dict]],
    default_value: dict,
    degraded_sections: list[str],
) -> dict:
    """Execute query with fail-soft: on error, return default + mark degraded."""
    try:
        return await query_fn()
    except Exception as e:
        logger.warning(f"ML Health section '{section_name}' degraded: {e}")
        degraded_sections.append(section_name)
        return {**default_value, "_degraded": True, "_error": str(e)[:100]}


# =============================================================================
# Query Functions
# =============================================================================


async def _query_sota_stats_coverage(session: AsyncSession) -> dict:
    """
    Query SOTA stats coverage from matches table.

    P0 - This is the root cause of "vuelo a ciegas":
    XGBoost model ran with 0% shots/corners coverage in 23/24.
    """
    # Coverage by season
    _understat_ids = ",".join(str(lid) for lid in UNDERSTAT_SUPPORTED_LEAGUES)
    season_query = text(f"""
        SELECT
            CASE
                WHEN date >= '2023-08-01' AND date < '2024-08-01' THEN '23/24'
                WHEN date >= '2024-08-01' AND date < '2025-08-01' THEN '24/25'
                WHEN date >= '2025-08-01' AND date < '2026-08-01' THEN '25/26'
            END as season,
            COUNT(*) as total_matches_ft,
            ROUND(100.0 * COUNT(*) FILTER (
                WHERE stats IS NOT NULL
                AND stats::text != '{{}}'
                AND (stats->>'_no_stats') IS NULL
            ) / NULLIF(COUNT(*), 0), 1) as with_stats_pct,
            ROUND(100.0 * COUNT(*) FILTER (
                WHERE (stats->>'_no_stats')::boolean = true
            ) / NULLIF(COUNT(*), 0), 1) as marked_no_stats_pct,
            ROUND(100.0 * COUNT(*) FILTER (
                WHERE stats->'home'->>'total_shots' IS NOT NULL
            ) / NULLIF(COUNT(*), 0), 1) as shots_present_pct
        FROM matches
        WHERE status IN ('FT', 'AET', 'PEN')
          AND date >= '2023-08-01'
          AND league_id IN ({_understat_ids})
        GROUP BY 1
        ORDER BY 1
    """)

    result = await session.execute(season_query)
    rows = result.fetchall()

    by_season = {}
    for row in rows:
        if row.season:
            by_season[row.season] = {
                "total_matches_ft": row.total_matches_ft,
                "with_stats_pct": float(row.with_stats_pct or 0),
                "marked_no_stats_pct": float(row.marked_no_stats_pct or 0),
                "shots_present_pct": float(row.shots_present_pct or 0),
            }

    # Coverage by league (current season only)
    _understat_ids = ",".join(str(lid) for lid in UNDERSTAT_SUPPORTED_LEAGUES)
    league_query = text(f"""
        SELECT
            league_id,
            ROUND(100.0 * COUNT(*) FILTER (
                WHERE stats IS NOT NULL
                AND stats::text != '{{}}'
                AND (stats->>'_no_stats') IS NULL
            ) / NULLIF(COUNT(*), 0), 1) as with_stats_pct
        FROM matches
        WHERE status IN ('FT', 'AET', 'PEN')
          AND date >= '2025-08-01'
          AND league_id IN ({_understat_ids})
        GROUP BY league_id
        ORDER BY with_stats_pct DESC
    """)

    result = await session.execute(league_query)
    league_rows = result.fetchall()

    by_league = []
    for row in league_rows:
        by_league.append({
            "league_id": row.league_id,
            "name": LEAGUE_NAMES.get(row.league_id, f"League {row.league_id}"),
            "with_stats_pct": float(row.with_stats_pct or 0),
        })

    # Determine status based on current season coverage
    current_season_pct = by_season.get("25/26", {}).get("with_stats_pct", 0)
    if current_season_pct >= 70:
        status = "ok"
    elif current_season_pct >= 50:
        status = "warn"
    else:
        status = "error"

    return {
        "by_season": by_season,
        "by_league": by_league,
        "status": status,
    }


async def _query_titan_coverage(session: AsyncSession) -> dict:
    """Query TITAN feature_matrix coverage by season and tier."""
    # Coverage by season
    season_query = text("""
        SELECT
            CASE
                WHEN kickoff_utc >= '2025-08-01' AND kickoff_utc < '2026-08-01' THEN '25/26'
                WHEN kickoff_utc >= '2024-08-01' AND kickoff_utc < '2025-08-01' THEN '24/25'
            END as season,
            COUNT(*) as total,
            COUNT(*) FILTER (WHERE tier1_complete = true) as tier1_count,
            COUNT(*) FILTER (WHERE tier1b_complete = true) as tier1b_count,
            COUNT(*) FILTER (WHERE tier1c_complete = true) as tier1c_count,
            COUNT(*) FILTER (WHERE tier1d_complete = true) as tier1d_count,
            ROUND(100.0 * COUNT(*) FILTER (WHERE tier1_complete = true) / NULLIF(COUNT(*), 0), 1) as tier1_pct,
            ROUND(100.0 * COUNT(*) FILTER (WHERE tier1b_complete = true) / NULLIF(COUNT(*), 0), 1) as tier1b_pct,
            ROUND(100.0 * COUNT(*) FILTER (WHERE tier1c_complete = true) / NULLIF(COUNT(*), 0), 1) as tier1c_pct,
            ROUND(100.0 * COUNT(*) FILTER (WHERE tier1d_complete = true) / NULLIF(COUNT(*), 0), 1) as tier1d_pct
        FROM titan.feature_matrix
        WHERE kickoff_utc >= '2024-08-01'
        GROUP BY 1
        ORDER BY 1
    """)

    result = await session.execute(season_query)
    rows = result.fetchall()

    by_season = {}
    for row in rows:
        if row.season:
            by_season[row.season] = {
                "tier1": {"complete": row.tier1_count, "total": row.total, "pct": float(row.tier1_pct or 0)},
                "tier1b": {"complete": row.tier1b_count, "total": row.total, "pct": float(row.tier1b_pct or 0)},
                "tier1c": {"complete": row.tier1c_count, "total": row.total, "pct": float(row.tier1c_pct or 0)},
                "tier1d": {"complete": row.tier1d_count, "total": row.total, "pct": float(row.tier1d_pct or 0)},
            }

    # Coverage by league (current season) - uses competition_id, not league_id
    league_query = text("""
        SELECT
            competition_id as league_id,
            ROUND(100.0 * COUNT(*) FILTER (WHERE tier1_complete = true) / NULLIF(COUNT(*), 0), 1) as tier1_pct,
            ROUND(100.0 * COUNT(*) FILTER (WHERE tier1b_complete = true) / NULLIF(COUNT(*), 0), 1) as tier1b_pct
        FROM titan.feature_matrix
        WHERE kickoff_utc >= '2025-08-01'
        GROUP BY competition_id
        ORDER BY tier1_pct DESC
    """)

    result = await session.execute(league_query)
    league_rows = result.fetchall()

    by_league = []
    for row in league_rows:
        by_league.append({
            "league_id": row.league_id,
            "name": LEAGUE_NAMES.get(row.league_id, f"League {row.league_id}"),
            "tier1_pct": float(row.tier1_pct or 0),
            "tier1b_pct": float(row.tier1b_pct or 0),
        })

    # Determine status
    current_tier1_pct = by_season.get("25/26", {}).get("tier1", {}).get("pct", 0)
    if current_tier1_pct >= 80:
        status = "ok"
    elif current_tier1_pct >= 50:
        status = "warn"
    else:
        status = "error"

    return {
        "by_season": by_season,
        "by_league": by_league,
        "status": status,
    }


async def _query_pit_compliance(session: AsyncSession) -> dict:
    """Query PIT (Point-in-Time) compliance from TITAN feature_matrix."""
    query = text("""
        SELECT
            COUNT(*) as total_rows,
            COUNT(*) FILTER (WHERE pit_max_captured_at >= kickoff_utc) as violations,
            ROUND(100.0 * COUNT(*) FILTER (WHERE pit_max_captured_at >= kickoff_utc) / NULLIF(COUNT(*), 0), 2) as violation_pct
        FROM titan.feature_matrix
        WHERE kickoff_utc >= '2025-08-01'
    """)

    result = await session.execute(query)
    row = result.fetchone()

    if not row:
        return {"total_rows": 0, "violations": 0, "violation_pct": 0.0, "status": "ok"}

    violations = row.violations or 0
    status = "error" if violations > 0 else "ok"

    return {
        "total_rows": row.total_rows or 0,
        "violations": violations,
        "violation_pct": float(row.violation_pct or 0),
        "status": status,
    }


async def _query_freshness(session: AsyncSession) -> dict:
    """
    Query data freshness with two scopes:

    1. **upcoming** (P0 — used for status & fuel_gauge):
       Only NS matches with kickoff in the next 48h.
       This is the "early warning" signal: if the pipeline stops refreshing,
       upcoming matches will have stale odds/xG and p95 will spike.

    2. **historical_7d** (P1 — observability only):
       All matches with kickoff in the last 7 days (including FT).
       NOT used for status/fuel_gauge — provided for dashboards and debugging.

    Each scope reports:
    - age_hours_now: NOW() - captured_at  (staleness)
    - lead_time_hours: kickoff_utc - captured_at  (operational context)
    """

    def _build_age_query(where_clause: str) -> str:
        return f"""
            SELECT
                'odds' as tier,
                ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP (
                    ORDER BY EXTRACT(EPOCH FROM (NOW() - odds_captured_at))/3600
                )::numeric, 1) as p50,
                ROUND(PERCENTILE_CONT(0.95) WITHIN GROUP (
                    ORDER BY EXTRACT(EPOCH FROM (NOW() - odds_captured_at))/3600
                )::numeric, 1) as p95,
                ROUND(MAX(EXTRACT(EPOCH FROM (NOW() - odds_captured_at))/3600)::numeric, 1) as max,
                COUNT(*) as n
            FROM titan.feature_matrix
            WHERE odds_captured_at IS NOT NULL
              AND {where_clause}

            UNION ALL

            SELECT
                'xg' as tier,
                ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP (
                    ORDER BY EXTRACT(EPOCH FROM (NOW() - xg_captured_at))/3600
                )::numeric, 1) as p50,
                ROUND(PERCENTILE_CONT(0.95) WITHIN GROUP (
                    ORDER BY EXTRACT(EPOCH FROM (NOW() - xg_captured_at))/3600
                )::numeric, 1) as p95,
                ROUND(MAX(EXTRACT(EPOCH FROM (NOW() - xg_captured_at))/3600)::numeric, 1) as max,
                COUNT(*) as n
            FROM titan.feature_matrix
            WHERE xg_captured_at IS NOT NULL
              AND {where_clause}
        """

    def _build_lead_query(where_clause: str) -> str:
        return f"""
            SELECT
                'odds' as tier,
                ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP (
                    ORDER BY EXTRACT(EPOCH FROM (kickoff_utc - odds_captured_at))/3600
                )::numeric, 1) as p50,
                ROUND(PERCENTILE_CONT(0.95) WITHIN GROUP (
                    ORDER BY EXTRACT(EPOCH FROM (kickoff_utc - odds_captured_at))/3600
                )::numeric, 1) as p95,
                ROUND(MAX(EXTRACT(EPOCH FROM (kickoff_utc - odds_captured_at))/3600)::numeric, 1) as max,
                COUNT(*) as n
            FROM titan.feature_matrix
            WHERE odds_captured_at IS NOT NULL
              AND {where_clause}

            UNION ALL

            SELECT
                'xg' as tier,
                ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP (
                    ORDER BY EXTRACT(EPOCH FROM (kickoff_utc - xg_captured_at))/3600
                )::numeric, 1) as p50,
                ROUND(PERCENTILE_CONT(0.95) WITHIN GROUP (
                    ORDER BY EXTRACT(EPOCH FROM (kickoff_utc - xg_captured_at))/3600
                )::numeric, 1) as p95,
                ROUND(MAX(EXTRACT(EPOCH FROM (kickoff_utc - xg_captured_at))/3600)::numeric, 1) as max,
                COUNT(*) as n
            FROM titan.feature_matrix
            WHERE xg_captured_at IS NOT NULL
              AND {where_clause}
        """

    def _parse_rows(rows) -> dict:
        result = {}
        for row in rows:
            if row.tier:
                result[row.tier] = {
                    "p50": float(row.p50) if row.p50 is not None else None,
                    "p95": float(row.p95) if row.p95 is not None else None,
                    "max": float(row.max) if row.max is not None else None,
                    "n": int(row.n) if row.n is not None else 0,
                }
        return result

    # --- Scope 1: Upcoming (P0 — used for status & fuel_gauge) ---
    upcoming_where = "kickoff_utc BETWEEN NOW() AND NOW() + INTERVAL '48 hours'"

    result = await session.execute(text(_build_age_query(upcoming_where)))
    upcoming_age = _parse_rows(result.fetchall())

    result = await session.execute(text(_build_lead_query(upcoming_where)))
    upcoming_lead = _parse_rows(result.fetchall())

    # --- Scope 2: Historical 7d (P1 — observability only) ---
    historical_where = "kickoff_utc >= NOW() - INTERVAL '7 days'"

    result = await session.execute(text(_build_age_query(historical_where)))
    historical_age = _parse_rows(result.fetchall())

    result = await session.execute(text(_build_lead_query(historical_where)))
    historical_lead = _parse_rows(result.fetchall())

    # --- Status based on UPCOMING only (early warning) ---
    odds_p95 = upcoming_age.get("odds", {}).get("p95")
    xg_p95 = upcoming_age.get("xg", {}).get("p95")

    status = "ok"
    if odds_p95 and odds_p95 > 24:
        status = "error"
    elif odds_p95 and odds_p95 > 6:
        status = "warn"
    if xg_p95 and xg_p95 > 72:
        status = "error"
    elif xg_p95 and xg_p95 > 24 and status != "error":
        status = "warn"

    return {
        "age_hours_now": upcoming_age,
        "lead_time_hours": upcoming_lead,
        "historical_7d": {
            "age_hours_now": historical_age,
            "lead_time_hours": historical_lead,
        },
        "status": status,
    }


async def _query_prediction_confidence(session: AsyncSession) -> dict:
    """Query prediction confidence metrics (entropy and tier distribution)."""
    query = text("""
        WITH prediction_entropy AS (
            SELECT
                -1 * (home_prob * LN(home_prob) + draw_prob * LN(draw_prob) + away_prob * LN(away_prob)) / LN(3) as normalized_entropy,
                frozen_confidence_tier
            FROM predictions
            WHERE home_prob > 0 AND draw_prob > 0 AND away_prob > 0
              AND created_at > NOW() - INTERVAL '30 days'
        )
        SELECT
            COUNT(*) as sample_n,
            ROUND(AVG(normalized_entropy)::numeric, 3) as avg,
            ROUND(PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY normalized_entropy)::numeric, 3) as p25,
            ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY normalized_entropy)::numeric, 3) as p50,
            ROUND(PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY normalized_entropy)::numeric, 3) as p75,
            ROUND(PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY normalized_entropy)::numeric, 3) as p95,
            COUNT(*) FILTER (WHERE frozen_confidence_tier = 'gold') as gold,
            COUNT(*) FILTER (WHERE frozen_confidence_tier = 'silver') as silver,
            COUNT(*) FILTER (WHERE frozen_confidence_tier = 'copper') as copper
        FROM prediction_entropy
    """)

    result = await session.execute(query)
    row = result.fetchone()

    if not row or row.sample_n == 0:
        return {
            "entropy": {},
            "tier_distribution": {},
            "sample_n": 0,
            "window_days": 30,
        }

    return {
        "entropy": {
            "avg": float(row.avg) if row.avg is not None else None,
            "p25": float(row.p25) if row.p25 is not None else None,
            "p50": float(row.p50) if row.p50 is not None else None,
            "p75": float(row.p75) if row.p75 is not None else None,
            "p95": float(row.p95) if row.p95 is not None else None,
        },
        "tier_distribution": {
            "gold": row.gold or 0,
            "silver": row.silver or 0,
            "copper": row.copper or 0,
        },
        "sample_n": row.sample_n or 0,
        "window_days": 30,
    }


def _compute_fuel_gauge(data: dict, degraded_sections: list[str]) -> dict:
    """
    Compute fuel gauge from collected metrics.

    ATI v1.1: Uses correct paths from data dict, includes SOTA stats coverage,
    uses age_hours_now for staleness detection.
    """
    reasons = []
    status = "ok"

    # Degraded sections (fail-soft triggered)
    if degraded_sections:
        status = "warn"
        for section in degraded_sections:
            reasons.append(f"Degraded section: {section}")

    # PIT violations (critical)
    pit = data.get("pit_compliance", {})
    if pit.get("violations", 0) > 0:
        status = "error"
        reasons.append(f"PIT violations: {pit['violations']}")

    # SOTA stats coverage (current season 25/26) - CRITICAL for XGBoost
    sota = data.get("sota_stats_coverage", {}).get("by_season", {}).get("25/26", {})
    sota_pct = sota.get("with_stats_pct", 0)
    if sota_pct < 50:
        status = "error"
        reasons.append(f"SOTA stats coverage critical: {sota_pct}%")
    elif sota_pct < 70 and status != "error":
        status = "warn"
        reasons.append(f"SOTA stats coverage low: {sota_pct}%")

    # TITAN tier1 coverage
    titan = data.get("titan_coverage", {}).get("by_season", {}).get("25/26", {}).get("tier1", {})
    tier1_pct = titan.get("pct", 0)
    if tier1_pct < 50:
        status = "error"
        reasons.append(f"TITAN tier1 coverage critical: {tier1_pct}%")
    elif tier1_pct < 80 and status != "error":
        status = "warn"
        reasons.append(f"TITAN tier1 coverage low: {tier1_pct}%")

    # Freshness - age_hours_now (early warning for pipeline down)
    freshness = data.get("freshness", {}).get("age_hours_now", {})
    odds_p95 = freshness.get("odds", {}).get("p95")
    if odds_p95 and odds_p95 > 24:
        status = "error"
        reasons.append(f"Odds staleness critical: p95={odds_p95}h ago")
    elif odds_p95 and odds_p95 > 6 and status != "error":
        status = "warn"
        reasons.append(f"Odds staleness elevated: p95={odds_p95}h ago")

    xg_p95 = freshness.get("xg", {}).get("p95")
    if xg_p95 and xg_p95 > 72:
        if status != "error":
            status = "error"
        reasons.append(f"xG staleness critical: p95={xg_p95}h ago")
    elif xg_p95 and xg_p95 > 24 and status != "error":
        status = "warn"
        reasons.append(f"xG staleness elevated: p95={xg_p95}h ago")

    if not reasons:
        reasons = ["All systems nominal"]

    return {
        "status": status,
        "reasons": reasons,
        "as_of_utc": datetime.now(timezone.utc).isoformat() + "Z",
    }
