"""
Auto-Lab Online — PASO 2: Advisory feature evaluation per league.

Runs Feature Lab FAST tests for a single league, persists results to DB,
and exposes them via dashboard. Does NOT modify league_serving_config
automatically (advisory only — ABE decides when to promote).

Uses FeatureEngineer from app/features/engineering.py (no duplication).
Feature set definitions imported from scripts/feature_lab.py constants.

Distributed lock: pg_try_advisory_lock(777001) prevents duplicate runs
across Gunicorn workers.
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.ml.metrics import calculate_brier_score

logger = logging.getLogger("futbolstats.auto_lab")

# ═══════════════════════════════════════════════════════════════════════════
# FAST TEST FEATURE SETS — Reference same constants as feature_lab.py
# ═══════════════════════════════════════════════════════════════════════════

# Baseline 14 (production model v1.0.0)
_BASELINE_14 = [
    "home_goals_scored_avg", "home_goals_conceded_avg",
    "home_shots_avg", "home_corners_avg",
    "home_rest_days", "home_matches_played",
    "away_goals_scored_avg", "away_goals_conceded_avg",
    "away_shots_avg", "away_corners_avg",
    "away_rest_days", "away_matches_played",
    "goal_diff_avg", "rest_diff",
]

# Elo variants
_ELO = ["elo_home", "elo_away", "elo_diff"]
_ELO_GW = ["elo_gw_home", "elo_gw_away", "elo_gw_diff"]

# Form
_FORM_CORE = ["home_win_rate5", "away_win_rate5", "form_diff"]

# xG
_XG_CORE = ["home_xg_for_avg", "away_xg_for_avg", "xg_diff_avg"]

# Odds-derived (implicit probs)
_ODDS = ["odds_home", "odds_draw", "odds_away"]

# Opponent-adjusted
_OPP_ADJ = ["opp_att_home", "opp_def_home", "opp_att_away", "opp_def_away",
             "opp_rating_diff"]

# Overperformance
_OVERPERF = ["overperf_home", "overperf_away", "overperf_diff"]

FAST_TESTS = {
    # Anchor: production model
    "baseline_14":      _BASELINE_14,
    # Elo-only (minimal)
    "elo_only":         _ELO,
    # Elo + Form (Optuna champion in many leagues)
    "elo_gw_form":      _FORM_CORE + _ELO_GW,
    # Baseline + Elo (best combo in Feature Lab)
    "baseline_elo":     _BASELINE_14 + _ELO,
    # Baseline + Elo + xG (if available)
    "baseline_elo_xg":  _BASELINE_14 + _ELO + _XG_CORE,
    # Baseline + Elo + Form + OppAdj (ABE combo)
    "abe_full":         _BASELINE_14 + _ELO + _FORM_CORE + _OPP_ADJ + _OVERPERF,
}

# XGBoost hyperparams (conservative, same as feature_lab FAST mode)
_XGB_PARAMS = {
    "objective": "multi:softprob",
    "num_class": 3,
    "max_depth": 3,
    "learning_rate": 0.05,
    "n_estimators": 100,
    "min_child_weight": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "verbosity": 0,
    "tree_method": "hist",
}

# Advisory lock ID (same pattern as Event Bus Sweeper)
_LAB_ADVISORY_LOCK_ID = 777001

# Number of random seeds for stability (3 vs 5 in full lab)
_N_SEEDS = 3


# ═══════════════════════════════════════════════════════════════════════════
# DISTRIBUTED LOCK
# ═══════════════════════════════════════════════════════════════════════════

async def _acquire_lab_lock(session) -> bool:
    result = await session.execute(
        text("SELECT pg_try_advisory_lock(:lock_id)"),
        {"lock_id": _LAB_ADVISORY_LOCK_ID},
    )
    return result.scalar()


async def _release_lab_lock(session):
    await session.execute(
        text("SELECT pg_advisory_unlock(:lock_id)"),
        {"lock_id": _LAB_ADVISORY_LOCK_ID},
    )


# ═══════════════════════════════════════════════════════════════════════════
# CADENCE: Which league to run next?
# ═══════════════════════════════════════════════════════════════════════════

async def _pick_next_league(session) -> dict:
    """Pick the league most overdue for a lab run.

    Returns dict with league_id, name, ft_90d, last_run_at, days_since_run
    or None if no league is due.
    """
    settings = get_settings()
    # Get all active leagues with FT count in last 90 days
    rows = (await session.execute(text("""
        SELECT al.league_id, al.name,
               COUNT(m.id) FILTER (WHERE m.date >= NOW() - INTERVAL '90 days') AS ft_90d,
               COUNT(m.id) AS ft_total,
               (SELECT MAX(lr.finished_at) FROM league_lab_runs lr
                WHERE lr.league_id = al.league_id AND lr.status = 'completed') AS last_run_at
        FROM admin_leagues al
        JOIN matches m ON m.league_id = al.league_id AND m.status = 'FT'
        WHERE al.kind = 'league' AND al.is_active = true
        GROUP BY al.league_id, al.name
        HAVING COUNT(m.id) >= :min_matches
        ORDER BY al.league_id
    """), {"min_matches": settings.AUTO_LAB_MIN_MATCHES_TOTAL}))).fetchall()

    if not rows:
        return None

    now = datetime.utcnow()
    best = None
    best_overdue = -1

    for r in rows:
        league_id, name, ft_90d, ft_total, last_run_at = r

        # Determine cadence based on activity level
        if ft_90d >= 180:
            cadence_days = settings.AUTO_LAB_CADENCE_HIGH_DAYS
        elif ft_90d >= 120:
            cadence_days = settings.AUTO_LAB_CADENCE_MID_DAYS
        else:
            cadence_days = settings.AUTO_LAB_CADENCE_LOW_DAYS

        if last_run_at is None:
            days_since = 999  # Never run
        else:
            days_since = (now - last_run_at.replace(tzinfo=None)).days

        overdue = days_since - cadence_days
        if overdue > best_overdue:
            best_overdue = overdue
            best = {
                "league_id": league_id,
                "name": name,
                "ft_90d": ft_90d,
                "ft_total": ft_total,
                "last_run_at": last_run_at,
                "days_since_run": days_since,
                "cadence_days": cadence_days,
                "overdue_days": overdue,
            }

    if best and best["overdue_days"] <= 0:
        logger.info("[AUTO_LAB] No league overdue (best: %s, %d days until due)",
                     best["name"], -best["overdue_days"])
        return None

    return best


# ═══════════════════════════════════════════════════════════════════════════
# EVALUATION: Train + eval XGBoost per feature set
# ═══════════════════════════════════════════════════════════════════════════

def _compute_market_brier(df):
    """Compute Brier score of the market (implied probabilities from odds)."""
    odds_cols = ["odds_home", "odds_draw", "odds_away"]
    if not all(c in df.columns for c in odds_cols):
        return None

    valid = df.dropna(subset=odds_cols)
    if len(valid) < 50:
        return None

    # De-vig (proportional)
    implied = 1.0 / valid[odds_cols].values
    total = implied.sum(axis=1, keepdims=True)
    market_probs = implied / total

    y_true = valid["result"].values
    return float(calculate_brier_score(y_true, market_probs))


def _evaluate_feature_set(df, features, test_name):
    """Train XGBoost with TimeSeriesSplit and evaluate.

    Returns dict with metrics or None if insufficient data.
    """
    import xgboost as xgb
    from sklearn.model_selection import TimeSeriesSplit

    # Filter to available features
    available = [f for f in features if f in df.columns]
    missing = [f for f in features if f not in df.columns]

    if len(available) < 2:
        return None

    # Drop rows with all-NaN features
    subset = df[available + ["result"]].dropna(subset=available, how="all")
    if len(subset) < 100:
        return None

    X = subset[available].fillna(0).values
    y = subset["result"].values

    # TimeSeriesSplit with 3 folds
    tscv = TimeSeriesSplit(n_splits=3)
    brier_scores = []

    for seed_offset in range(_N_SEEDS):
        params = dict(_XGB_PARAMS)
        params["random_state"] = 42 + seed_offset

        fold_briers = []
        for train_idx, val_idx in tscv.split(X):
            model = xgb.XGBClassifier(**params)
            model.fit(X[train_idx], y[train_idx], verbose=False)
            proba = model.predict_proba(X[val_idx])

            # Ensure 3 classes (XGBoost may drop a class if not in training)
            if proba.shape[1] < 3:
                n_pad = 3 - proba.shape[1]
                proba = np.hstack([proba, np.zeros((proba.shape[0], n_pad))])

            fold_briers.append(calculate_brier_score(y[val_idx], proba))

        brier_scores.append(float(np.mean(fold_briers)))

    brier_ensemble = float(np.mean(brier_scores))
    brier_std = float(np.std(brier_scores))

    # Accuracy (last fold, last seed — indicative only)
    last_preds = np.argmax(proba, axis=1)
    accuracy = float(np.mean(last_preds == y[val_idx]))

    return {
        "test_name": test_name,
        "brier_ensemble": round(brier_ensemble, 5),
        "brier_std": round(brier_std, 5),
        "accuracy": round(accuracy, 4),
        "n_features": len(available),
        "n_missing": len(missing),
        "missing_features": missing[:5],  # First 5 for diagnostics
        "n_train": int(len(X) * 0.67),  # Approximate
        "n_test": int(len(X) * 0.33),
        "n_total": len(X),
    }


# ═══════════════════════════════════════════════════════════════════════════
# PERSISTENCE: Save run + results to DB
# ═══════════════════════════════════════════════════════════════════════════

async def _create_run(session, league_id, trigger_reason="scheduled"):
    """Insert a new league_lab_runs row. Returns run_id."""
    result = await session.execute(text("""
        INSERT INTO league_lab_runs (league_id, status, mode, trigger_reason)
        VALUES (:league_id, 'running', 'fast', :trigger)
        RETURNING id
    """), {"league_id": league_id, "trigger": trigger_reason})
    await session.commit()
    return result.scalar()


async def _complete_run(session, run_id, results, market_brier, duration_ms,
                        n_matches, error_msg=None):
    """Update run with final results."""
    best = None
    if results:
        best = min(results, key=lambda r: r["brier_ensemble"])

    status = "completed" if not error_msg else "error"

    await session.execute(text("""
        UPDATE league_lab_runs SET
            finished_at = NOW(),
            status = :status,
            n_matches_used = :n_matches,
            n_tests_run = :n_tests,
            duration_ms = :duration_ms,
            error_message = :error_msg,
            best_test_name = :best_name,
            best_brier = :best_brier,
            market_brier = :market_brier,
            delta_vs_market = :delta
        WHERE id = :run_id
    """), {
        "run_id": run_id,
        "status": status,
        "n_matches": n_matches,
        "n_tests": len(results) if results else 0,
        "duration_ms": duration_ms,
        "error_msg": error_msg,
        "best_name": best["test_name"] if best else None,
        "best_brier": best["brier_ensemble"] if best else None,
        "market_brier": market_brier,
        "delta": round(best["brier_ensemble"] - market_brier, 5) if best and market_brier else None,
    })
    await session.commit()


async def _save_results(session, run_id, league_id, results, market_brier):
    """Insert individual test results."""
    for r in results:
        delta = round(r["brier_ensemble"] - market_brier, 5) if market_brier else None
        await session.execute(text("""
            INSERT INTO league_lab_results
                (run_id, league_id, test_name, test_type,
                 brier_ensemble, brier_market, delta_vs_market,
                 accuracy, n_train, n_test, n_features, result_json)
            VALUES
                (:run_id, :league_id, :test_name, 'feature_set',
                 :brier, :market_brier, :delta,
                 :accuracy, :n_train, :n_test, :n_features,
                 CAST(:result_json AS JSONB))
        """), {
            "run_id": run_id,
            "league_id": league_id,
            "test_name": r["test_name"],
            "brier": r["brier_ensemble"],
            "market_brier": market_brier,
            "delta": delta,
            "accuracy": r["accuracy"],
            "n_train": r["n_train"],
            "n_test": r["n_test"],
            "n_features": r["n_features"],
            "result_json": __import__("json").dumps({
                "brier_std": r["brier_std"],
                "missing_features": r.get("missing_features", []),
                "n_missing": r.get("n_missing", 0),
            }),
        })
    await session.commit()


# ═══════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

async def run_fast_lab(session, league_id, trigger_reason="scheduled"):
    """Run FAST feature lab for a single league.

    Uses FeatureEngineer.build_training_dataset() for data extraction
    (no duplication of feature engineering logic).

    Returns dict with run summary or None on error.
    """
    from app.features.engineering import FeatureEngineer

    settings = get_settings()
    t0 = time.monotonic()
    run_id = await _create_run(session, league_id, trigger_reason)

    try:
        # Extract data using shared FeatureEngineer
        fe = FeatureEngineer(session)
        min_date = datetime(2023, 1, 1)  # TRAINING_MIN_DATE
        df = await fe.build_training_dataset(
            min_date=min_date,
            league_ids=[league_id],
            league_only=True,
        )

        if df is None or len(df) < settings.AUTO_LAB_MIN_MATCHES_TOTAL:
            duration_ms = int((time.monotonic() - t0) * 1000)
            await _complete_run(session, run_id, [], None, duration_ms, len(df) if df is not None else 0,
                                error_msg="insufficient_data")
            return {"run_id": run_id, "status": "insufficient_data", "n_matches": len(df) if df is not None else 0}

        n_matches = len(df)

        # Enrich with derived features (Elo, Form, OppAdj, Overperf)
        from app.features.lab_features import enrich_for_lab
        df = enrich_for_lab(df)

        logger.info("[AUTO_LAB] League %d: %d matches, %d features available",
                     league_id, n_matches, len(df.columns))

        # Market benchmark
        market_brier = _compute_market_brier(df)

        # Run FAST tests
        results = []
        for test_name, features in FAST_TESTS.items():
            try:
                r = _evaluate_feature_set(df, features, test_name)
                if r:
                    results.append(r)
                    logger.info("[AUTO_LAB] %s: Brier=%.5f (n_feat=%d)",
                                 test_name, r["brier_ensemble"], r["n_features"])
            except Exception as e:
                logger.warning("[AUTO_LAB] Test %s failed: %s", test_name, e)

        duration_ms = int((time.monotonic() - t0) * 1000)

        # Persist
        await _save_results(session, run_id, league_id, results, market_brier)
        await _complete_run(session, run_id, results, market_brier, duration_ms, n_matches)

        best = min(results, key=lambda r: r["brier_ensemble"]) if results else None
        summary = {
            "run_id": run_id,
            "league_id": league_id,
            "status": "completed",
            "n_matches": n_matches,
            "n_tests": len(results),
            "market_brier": market_brier,
            "best_test": best["test_name"] if best else None,
            "best_brier": best["brier_ensemble"] if best else None,
            "delta_vs_market": round(best["brier_ensemble"] - market_brier, 5) if best and market_brier else None,
            "duration_ms": duration_ms,
        }
        logger.info("[AUTO_LAB] League %d DONE: %s", league_id, summary)
        return summary

    except asyncio.TimeoutError:
        duration_ms = int((time.monotonic() - t0) * 1000)
        await _complete_run(session, run_id, [], None, duration_ms, 0,
                            error_msg="timeout")
        return {"run_id": run_id, "status": "timeout"}

    except Exception as e:
        duration_ms = int((time.monotonic() - t0) * 1000)
        logger.error("[AUTO_LAB] League %d FAILED: %s", league_id, e, exc_info=True)
        try:
            await _complete_run(session, run_id, [], None, duration_ms, 0,
                                error_msg=str(e)[:500])
        except Exception:
            pass
        return {"run_id": run_id, "status": "error", "error": str(e)}


async def auto_lab_scheduler_job():
    """Entry point for APScheduler. Acquires advisory lock, picks league, runs lab."""
    settings = get_settings()
    if not settings.AUTO_LAB_ENABLED:
        return

    from app.database import AsyncSessionLocal

    async with AsyncSessionLocal() as session:
        if not await _acquire_lab_lock(session):
            logger.info("[AUTO_LAB] Another worker already running, skipping")
            return

        try:
            # Check daily quota
            today_count = (await session.execute(text("""
                SELECT COUNT(*) FROM league_lab_runs
                WHERE started_at >= CURRENT_DATE AND status != 'error'
            """))).scalar()

            if today_count >= settings.AUTO_LAB_MAX_PER_DAY:
                logger.info("[AUTO_LAB] Daily quota reached (%d/%d)",
                             today_count, settings.AUTO_LAB_MAX_PER_DAY)
                return

            # Pick next league
            league = await _pick_next_league(session)
            if not league:
                logger.info("[AUTO_LAB] No league due for evaluation")
                return

            logger.info("[AUTO_LAB] Selected: %s (league_id=%d, overdue=%d days)",
                         league["name"], league["league_id"], league["overdue_days"])

            # Run with timeout
            timeout_s = settings.AUTO_LAB_TIMEOUT_MIN * 60
            result = await asyncio.wait_for(
                run_fast_lab(session, league["league_id"]),
                timeout=timeout_s,
            )

            if result:
                logger.info("[AUTO_LAB] Completed: %s", result.get("status"))

        except asyncio.TimeoutError:
            logger.error("[AUTO_LAB] Global timeout (%d min)", settings.AUTO_LAB_TIMEOUT_MIN)
        except Exception as e:
            logger.error("[AUTO_LAB] Scheduler job failed: %s", e, exc_info=True)
        finally:
            try:
                await _release_lab_lock(session)
            except Exception:
                pass
