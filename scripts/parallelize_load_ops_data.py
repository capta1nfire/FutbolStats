#!/usr/bin/env python3
"""Parallelize _load_ops_data in ops_routes.py.

Extracts 6 helper functions from the sequential implementation and
rewrites _load_ops_data to use asyncio.gather for ~16 concurrent sections.

Each _calculate_* function gets its own DB session (pool: 10+20=30, uses ~14).
Expected result: ops dashboard cache refresh ~3-5x faster.
"""

FILE = "app/dashboard/ops_routes.py"


def main():
    with open(FILE) as f:
        lines = f.readlines()

    # Find function boundaries (0-indexed)
    load_start = None
    cached_start = None
    for i, line in enumerate(lines):
        if line.startswith("async def _load_ops_data(") and load_start is None:
            load_start = i
        elif line.startswith("async def _get_cached_ops_data(") and cached_start is None:
            cached_start = i

    assert load_start is not None, "Could not find _load_ops_data"
    assert cached_start is not None, "Could not find _get_cached_ops_data"

    old_func = lines[load_start:cached_start]
    print(f"_load_ops_data: line {load_start + 1}")
    print(f"_get_cached_ops_data: line {cached_start + 1}")
    print(f"Old function: {len(old_func)} lines")

    # Extract LLM cost block body (at 12-space indent inside try block)
    llm_body_start = None
    llm_body_end = None
    for i, line in enumerate(old_func):
        if "# Use pricing from settings (single source of truth)" in line and llm_body_start is None:
            llm_body_start = i
        if llm_body_start is not None and llm_body_end is None:
            if "except Exception as e:" in line.strip():
                if i + 1 < len(old_func) and "Could not calculate LLM cost" in old_func[i + 1]:
                    llm_body_end = i
                    break

    assert llm_body_start is not None, "Could not find LLM cost body start"
    assert llm_body_end is not None, "Could not find LLM cost body end"

    llm_body = "".join(old_func[llm_body_start:llm_body_end])
    print(f"LLM cost body: {llm_body_end - llm_body_start} lines")

    # Build replacement code
    new_code = build_replacement(llm_body)

    # Splice into file
    new_lines = lines[:load_start] + new_code.splitlines(keepends=True) + lines[cached_start:]

    with open(FILE, "w") as f:
        f.writelines(new_lines)

    print(f"Before: {len(lines)} lines")
    print(f"After: {len(new_lines)} lines")
    print(f"Delta: {len(new_lines) - len(lines)}")
    print("Done.")


def build_replacement(llm_body: str) -> str:
    """Build helper functions + rewritten _load_ops_data."""

    return (
        '''# =====================================================================
# League names fallback (module-level for _build_league_name_map)
# =====================================================================
_LEAGUE_NAMES_FALLBACK: dict[int, str] = {
    1: "World Cup", 2: "Champions League", 3: "Europa League",
    4: "Euro", 5: "Nations League", 9: "Copa Am\\u00e9rica",
    10: "Friendlies", 11: "Sudamericana", 13: "Libertadores",
    22: "Gold Cup",
    # Legacy: league_id=28 was previously (incorrectly) used for WCQ CONMEBOL in code.
    # In production DB it may contain SAFF Championship fixtures.
    28: "SAFF Championship (legacy)",
    # WC 2026 Qualifiers (correct API-Football league IDs)
    29: "WCQ CAF", 30: "WCQ AFC", 31: "WCQ CONCACAF",
    32: "WCQ UEFA", 33: "WCQ OFC", 34: "WCQ CONMEBOL",
    37: "WCQ Intercontinental Play-offs",
    39: "Premier League", 45: "FA Cup", 61: "Ligue 1",
    71: "Brazil Serie A", 78: "Bundesliga", 88: "Eredivisie",
    94: "Primeira Liga", 128: "Argentina Primera", 135: "Serie A",
    140: "La Liga", 143: "Copa del Rey", 203: "Super Lig",
    239: "Colombia Primera A", 242: "Ecuador Liga Pro",
    250: "Paraguay Primera - Apertura", 252: "Paraguay Primera - Clausura",
    253: "MLS", 262: "Liga MX", 265: "Chile Primera Divisi\\u00f3n",
    268: "Uruguay Primera - Apertura", 270: "Uruguay Primera - Clausura",
    281: "Peru Primera Divisi\\u00f3n", 299: "Venezuela Primera Divisi\\u00f3n",
    344: "Bolivia Primera Divisi\\u00f3n", 848: "Conference League",
}


def _build_league_name_map() -> dict[int, str]:
    """Build league_id -> name map from fallback + COMPETITIONS."""
    from app.etl.competitions import COMPETITIONS

    league_name_by_id = _LEAGUE_NAMES_FALLBACK.copy()
    try:
        for league_id, comp in (COMPETITIONS or {}).items():
            if league_id is not None and comp is not None:
                name = getattr(comp, "name", None)
                if name:
                    league_name_by_id[int(league_id)] = name
    except Exception:
        pass
    return league_name_by_id


# =====================================================================
# Parallel helpers for _load_ops_data (each opens its own DB session)
# =====================================================================

async def _fetch_budget_status() -> dict:
    """Fetch API-Football budget status (HTTP call + timezone enrichment)."""
    budget_status: dict = {"status": "unavailable"}
    try:
        from app.etl.api_football import get_api_account_status
        budget_status = await get_api_account_status()
    except Exception as e:
        logger.warning(f"Could not fetch API account status: {e}")
        budget_status = {"status": "unavailable", "error": str(e)}

    try:
        from zoneinfo import ZoneInfo
        tz_name = "America/Los_Angeles"
        reset_hour, reset_minute = 16, 0
        now_utc = datetime.utcnow().replace(tzinfo=ZoneInfo("UTC"))
        now_la = now_utc.astimezone(ZoneInfo(tz_name))
        next_reset_la = now_la.replace(hour=reset_hour, minute=reset_minute, second=0, microsecond=0)
        if next_reset_la <= now_la:
            next_reset_la += timedelta(days=1)
        next_reset_utc = next_reset_la.astimezone(ZoneInfo("UTC"))
        if not isinstance(budget_status, dict):
            budget_status = {"status": "unavailable"}
        budget_status.update({
            "tokens_reset_tz": tz_name,
            "tokens_reset_local_time": f"{reset_hour:02d}:{reset_minute:02d}",
            "tokens_reset_at_la": next_reset_la.isoformat(),
            "tokens_reset_at_utc": next_reset_utc.isoformat(),
            "tokens_reset_note": "Observed daily refresh around 4:00pm America/Los_Angeles",
        })
    except Exception:
        pass
    return budget_status


async def _run_inline_queries() -> dict:
    """Run simple inline DB queries for ops dashboard (single session)."""
    async with AsyncSessionLocal() as session:
        # Tracked leagues (distinct league_id)
        res = await session.execute(text("SELECT COUNT(DISTINCT league_id) FROM matches WHERE league_id IS NOT NULL"))
        tracked_leagues_count = int(res.scalar() or 0)

        # Upcoming matches (next 24h)
        res = await session.execute(
            text("""
                SELECT league_id, COUNT(*) AS upcoming
                FROM matches
                WHERE league_id IS NOT NULL
                  AND date >= NOW()
                  AND date < NOW() + INTERVAL '24 hours'
                GROUP BY league_id
                ORDER BY upcoming DESC
                LIMIT 20
            """)
        )
        upcoming_by_league = [{"league_id": int(r[0]), "upcoming_24h": int(r[1])} for r in res.fetchall()]

        # PIT snapshots (live, lineup_confirmed)
        res = await session.execute(
            text("""
                SELECT COUNT(*)
                FROM odds_snapshots
                WHERE snapshot_type = 'lineup_confirmed'
                  AND odds_freshness = 'live'
                  AND snapshot_at > NOW() - INTERVAL '60 minutes'
            """)
        )
        pit_live_60m = int(res.scalar() or 0)

        res = await session.execute(
            text("""
                SELECT COUNT(*)
                FROM odds_snapshots
                WHERE snapshot_type = 'lineup_confirmed'
                  AND odds_freshness = 'live'
                  AND snapshot_at > NOW() - INTERVAL '24 hours'
            """)
        )
        pit_live_24h = int(res.scalar() or 0)

        # DKO distribution (last 60m)
        res = await session.execute(
            text("""
                SELECT ROUND(delta_to_kickoff_seconds / 60.0) AS min_to_ko, COUNT(*) AS c
                FROM odds_snapshots
                WHERE snapshot_type = 'lineup_confirmed'
                  AND odds_freshness = 'live'
                  AND snapshot_at > NOW() - INTERVAL '60 minutes'
                  AND delta_to_kickoff_seconds IS NOT NULL
                GROUP BY 1
                ORDER BY 1
            """)
        )
        pit_dko_60m = [{"min_to_ko": int(r[0]), "count": int(r[1])} for r in res.fetchall()]

        # Latest PIT snapshots (last 10, any freshness)
        res = await session.execute(
            text("""
                SELECT os.snapshot_at, os.match_id, m.league_id, os.odds_freshness, os.delta_to_kickoff_seconds,
                       os.odds_home, os.odds_draw, os.odds_away, os.bookmaker
                FROM odds_snapshots os
                JOIN matches m ON m.id = os.match_id
                WHERE os.snapshot_type = 'lineup_confirmed'
                ORDER BY os.snapshot_at DESC
                LIMIT 10
            """)
        )
        latest_pit = []
        for r in res.fetchall():
            latest_pit.append({
                "snapshot_at": r[0].isoformat() if r[0] else None,
                "match_id": int(r[1]) if r[1] is not None else None,
                "league_id": int(r[2]) if r[2] is not None else None,
                "odds_freshness": r[3],
                "delta_to_kickoff_minutes": round(float(r[4]) / 60.0, 1) if r[4] is not None else None,
                "odds": {
                    "home": float(r[5]) if r[5] is not None else None,
                    "draw": float(r[6]) if r[6] is not None else None,
                    "away": float(r[7]) if r[7] is not None else None,
                },
                "bookmaker": r[8],
            })

        # Movement snapshots (last 24h)
        lineup_movement_24h = None
        market_movement_24h = None
        try:
            res = await session.execute(
                text("SELECT COUNT(*) FROM lineup_movement_snapshots WHERE captured_at > NOW() - INTERVAL '24 hours'")
            )
            lineup_movement_24h = int(res.scalar() or 0)
        except Exception:
            lineup_movement_24h = None
        try:
            res = await session.execute(
                text("SELECT COUNT(*) FROM market_movement_snapshots WHERE captured_at > NOW() - INTERVAL '24 hours'")
            )
            market_movement_24h = int(res.scalar() or 0)
        except Exception:
            market_movement_24h = None

        # Stats backfill health (last 72h finished matches)
        res = await session.execute(
            text("""
                SELECT
                    COUNT(*) FILTER (WHERE stats IS NOT NULL AND stats::text != '{}' AND stats::text != 'null') AS with_stats,
                    COUNT(*) FILTER (WHERE stats IS NULL OR stats::text = '{}' OR stats::text = 'null') AS missing_stats
                FROM matches
                WHERE status IN ('FT', 'AET', 'PEN')
                  AND COALESCE(finished_at, date) > NOW() - INTERVAL '72 hours'
            """)
        )
        row = res.first()
        stats_with = int(row[0] or 0) if row else 0
        stats_missing = int(row[1] or 0) if row else 0

        # =============================================================
        # PROGRESS METRICS (for re-test / Alpha readiness)
        # =============================================================
        TARGET_PIT_SNAPSHOTS_30D = int(os.environ.get("TARGET_PIT_SNAPSHOTS_30D", "500"))
        TARGET_PIT_BETS_30D = int(os.environ.get("TARGET_PIT_BETS_30D", "500"))
        TARGET_BASELINE_COVERAGE_PCT = int(os.environ.get("TARGET_BASELINE_COVERAGE_PCT", "60"))

        pit_snapshots_30d = 0
        try:
            res = await session.execute(
                text("""
                    SELECT COUNT(*)
                    FROM odds_snapshots
                    WHERE snapshot_type = 'lineup_confirmed'
                      AND odds_freshness = 'live'
                      AND snapshot_at > NOW() - INTERVAL '30 days'
                """)
            )
            pit_snapshots_30d = int(res.scalar() or 0)
        except Exception:
            pit_snapshots_30d = 0

        pit_bets_30d = 0
        try:
            res = await session.execute(
                text("""
                    SELECT COUNT(DISTINCT os.id)
                    FROM odds_snapshots os
                    WHERE os.snapshot_type = 'lineup_confirmed'
                      AND os.odds_freshness = 'live'
                      AND os.snapshot_at > NOW() - INTERVAL '30 days'
                      AND EXISTS (
                          SELECT 1 FROM predictions p
                          WHERE p.match_id = os.match_id
                            AND p.created_at < os.snapshot_at
                      )
                """)
            )
            pit_bets_30d = int(res.scalar() or 0)
        except Exception:
            pit_bets_30d = 0

        baseline_coverage_pct = 0
        pit_with_baseline = 0
        pit_total_for_baseline = 0
        try:
            res = await session.execute(
                text("""
                    SELECT
                        COUNT(*) FILTER (WHERE has_baseline) AS with_baseline,
                        COUNT(*) AS total
                    FROM (
                        SELECT os.id,
                               EXISTS (
                                   SELECT 1 FROM market_movement_snapshots mms
                                   WHERE mms.match_id = os.match_id
                                     AND mms.captured_at < (
                                         SELECT m.date FROM matches m WHERE m.id = os.match_id
                                     )
                               ) AS has_baseline
                        FROM odds_snapshots os
                        WHERE os.snapshot_type = 'lineup_confirmed'
                          AND os.odds_freshness = 'live'
                          AND os.snapshot_at > NOW() - INTERVAL '30 days'
                    ) sub
                """)
            )
            row = res.first()
            if row:
                pit_with_baseline = int(row[0] or 0)
                pit_total_for_baseline = int(row[1] or 0)
                if pit_total_for_baseline > 0:
                    baseline_coverage_pct = round((pit_with_baseline / pit_total_for_baseline) * 100, 1)
        except Exception:
            baseline_coverage_pct = 0
            pit_with_baseline = 0
            pit_total_for_baseline = 0

        progress_metrics = {
            "pit_snapshots_30d": pit_snapshots_30d,
            "target_pit_snapshots_30d": TARGET_PIT_SNAPSHOTS_30D,
            "pit_bets_30d": pit_bets_30d,
            "target_pit_bets_30d": TARGET_PIT_BETS_30D,
            "baseline_coverage_pct": baseline_coverage_pct,
            "pit_with_baseline": pit_with_baseline,
            "pit_total_for_baseline": pit_total_for_baseline,
            "target_baseline_coverage_pct": TARGET_BASELINE_COVERAGE_PCT,
            "ready_for_retest": (
                pit_bets_30d >= TARGET_PIT_BETS_30D and
                baseline_coverage_pct >= TARGET_BASELINE_COVERAGE_PCT
            ),
        }

    return {
        "tracked_leagues_count": tracked_leagues_count,
        "upcoming_by_league": upcoming_by_league,
        "pit_live_60m": pit_live_60m,
        "pit_live_24h": pit_live_24h,
        "pit_dko_60m": pit_dko_60m,
        "latest_pit": latest_pit,
        "lineup_movement_24h": lineup_movement_24h,
        "market_movement_24h": market_movement_24h,
        "stats_with": stats_with,
        "stats_missing": stats_missing,
        "progress_metrics": progress_metrics,
    }


async def _run_llm_cost_queries() -> dict:
    """Calculate LLM cost metrics for ops dashboard."""
    llm_cost_data: dict = {"provider": "gemini", "status": "unavailable"}
    try:
        async with AsyncSessionLocal() as session:
'''
        + llm_body
        + '''    except Exception as e:
        logger.warning(f"Could not calculate LLM cost: {e}")
        llm_cost_data = {"provider": "gemini", "status": "error", "error": str(e)}
    return llm_cost_data


async def _run_coverage_queries(league_name_by_id: dict[int, str]) -> list:
    """Coverage by league (NS matches in next 48h with predictions/odds)."""
    coverage_by_league = []
    try:
        async with AsyncSessionLocal() as session:
            res = await session.execute(
                text("""
                    SELECT
                        m.league_id,
                        COUNT(*) AS total_ns,
                        COUNT(p.id) AS with_prediction,
                        COUNT(m.odds_home) AS with_odds
                    FROM matches m
                    LEFT JOIN predictions p ON p.match_id = m.id
                    WHERE m.status = 'NS'
                      AND m.date >= NOW()
                      AND m.date < NOW() + INTERVAL '48 hours'
                      AND m.league_id IS NOT NULL
                    GROUP BY m.league_id
                    ORDER BY COUNT(*) DESC
                    LIMIT 15
                """)
            )
            for row in res.fetchall():
                lid = int(row[0])
                total = int(row[1])
                with_pred = int(row[2])
                with_odds = int(row[3])
                coverage_by_league.append({
                    "league_id": lid,
                    "league_name": league_name_by_id.get(lid, f"League {lid}"),
                    "total_ns": total,
                    "with_prediction": with_pred,
                    "with_odds": with_odds,
                    "pred_pct": round(with_pred / total * 100, 1) if total > 0 else 0,
                    "odds_pct": round(with_odds / total * 100, 1) if total > 0 else 0,
                })
    except Exception as e:
        logger.warning(f"Could not calculate coverage by league: {e}")
    return coverage_by_league


async def _load_ops_data() -> dict:
    """
    Ops dashboard: read-only aggregated metrics from DB + in-process state.
    Parallelized with asyncio.gather -- ~16 independent sections run concurrently,
    each with its own DB session. Pool: 10+20=30, uses ~14 concurrent sessions.
    """
    from app.main import _live_summary_cache  # lazy import (P0-11: no top-level app.main)
    from app.scheduler import get_last_sync_time

    now = datetime.utcnow()
    league_mode = os.environ.get("LEAGUE_MODE", "tracked").strip().lower()
    last_sync = get_last_sync_time()
    league_name_by_id = _build_league_name_map()

    # Helper: run a _calculate_* function with its own DB session
    async def _calc(fn):
        async with AsyncSessionLocal() as s:
            return await fn(s)

    # Run all independent sections in parallel
    (
        budget_status,
        sentry_health,
        inline,
        predictions_health,
        fastpath_health,
        model_performance,
        telemetry_data,
        shadow_mode_data,
        sensor_b_data,
        extc_shadow_data,
        rerun_serving_data,
        jobs_health_data,
        sota_enrichment_data,
        titan_data,
        llm_cost_data,
        coverage_by_league,
    ) = await asyncio.gather(
        _fetch_budget_status(),
        _fetch_sentry_health(),
        _run_inline_queries(),
        _calc(_calculate_predictions_health),
        _calc(_calculate_fastpath_health),
        _calc(_calculate_model_performance),
        _calc(_calculate_telemetry_summary),
        _calc(_calculate_shadow_mode_summary),
        _calc(_calculate_sensor_b_summary),
        _calc(_calculate_extc_shadow_summary),
        _calc(_calculate_rerun_serving_summary),
        _calc(_calculate_jobs_health_summary),
        _calc(_calculate_sota_enrichment_summary),
        _calculate_titan_summary(),
        _run_llm_cost_queries(),
        _run_coverage_queries(league_name_by_id),
    )

    # Post-processing: enrich with league names
    for item in inline["upcoming_by_league"]:
        item["league_name"] = league_name_by_id.get(item["league_id"])
    for item in inline["latest_pit"]:
        lid = item.get("league_id")
        if isinstance(lid, int):
            item["league_name"] = league_name_by_id.get(lid)

    # Live summary stats (from main.py cache, lazy imported)
    live_summary_stats = {
        "cache_ttl_seconds": _live_summary_cache["ttl"],
        "cache_timestamp": _live_summary_cache["timestamp"],
        "cache_age_seconds": round(time.time() - _live_summary_cache["timestamp"], 1) if _live_summary_cache["timestamp"] else None,
        "cached_live_matches": len(_live_summary_cache["data"]["matches"]) if _live_summary_cache["data"] else 0,
    }

    # ML model status
    ml_model_info = {
        "loaded": ml_engine.model is not None,
        "version": ml_engine.model_version,
        "source": "file",
        "model_path": str(ml_engine.model_path),
    }
    if ml_engine.model is not None:
        try:
            ml_model_info["n_features"] = ml_engine.model.n_features_in_
        except AttributeError:
            pass

    return {
        "generated_at": now.isoformat(),
        "league_mode": league_mode,
        "tracked_leagues_count": inline["tracked_leagues_count"],
        "last_sync_at": last_sync.isoformat() if last_sync else None,
        "budget": budget_status,
        "sentry": sentry_health,
        "pit": {
            "live_60m": inline["pit_live_60m"],
            "live_24h": inline["pit_live_24h"],
            "delta_to_kickoff_60m": inline["pit_dko_60m"],
            "latest": inline["latest_pit"],
        },
        "movement": {
            "lineup_movement_24h": inline["lineup_movement_24h"],
            "market_movement_24h": inline["market_movement_24h"],
        },
        "stats_backfill": {
            "finished_72h_with_stats": inline["stats_with"],
            "finished_72h_missing_stats": inline["stats_missing"],
        },
        "upcoming": {
            "by_league_24h": inline["upcoming_by_league"],
        },
        "progress": inline["progress_metrics"],
        "predictions_health": predictions_health,
        "fastpath_health": fastpath_health,
        "model_performance": model_performance,
        "telemetry": telemetry_data,
        "llm_cost": llm_cost_data,
        "shadow_mode": shadow_mode_data,
        "sensor_b": sensor_b_data,
        "extc_shadow": extc_shadow_data,
        "rerun_serving": rerun_serving_data,
        "jobs_health": jobs_health_data,
        "sota_enrichment": sota_enrichment_data,
        "titan": titan_data,
        "coverage_by_league": coverage_by_league,
        "ml_model": ml_model_info,
        "live_summary": live_summary_stats,
        "db_pool": get_pool_status(),
        "providers": _get_providers_health(),
    }


'''
    )


if __name__ == "__main__":
    main()
