#!/usr/bin/env python3
"""
PIT Evaluation v3 - Enhanced with Auditor Metrics

Implements PIT Evaluation Protocol v3 with:
- Segmentation by timing bucket (10-20, 20-30, 30-45, 45-60, 60-75, 75-90)
- LogLoss and ECE (Expected Calibration Error)
- Paired differential (model vs market) with bootstrap CI
- CLV (Closing Line Value) using T5 as closing line
- Betting audit: by pick, by odds range, by edge decile

FASE 3C (ABE 2026-01-25):
- --devig: De-vig method (proportional=baseline, power=alternative)
- --calibrator: Calibration method (none=baseline, isotonic, temperature)
- --calib-train-end: Date to split calibration train/test (CRITICAL: no leakage)

Usage:
    # Baseline (default)
    DATABASE_URL=... python3 scripts/evaluate_pit_v3.py --min-snapshot-date 2026-01-07

    # With power de-vig
    DATABASE_URL=... python3 scripts/evaluate_pit_v3.py --devig power

    # With isotonic calibration (requires --calib-train-end)
    DATABASE_URL=... python3 scripts/evaluate_pit_v3.py --calibrator isotonic --calib-train-end 2026-01-06

Output:
    logs/pit_evaluation_v3_YYYYMMDD_HHMMSS.json
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import random
from datetime import datetime
from decimal import Decimal
from math import isnan
from pathlib import Path

import asyncpg
import numpy as np


def to_float(val):
    """Convert Decimal or other numeric to float."""
    if val is None:
        return None
    if isinstance(val, Decimal):
        return float(val)
    return float(val)


def normalize_probs(p_home, p_draw, p_away) -> tuple[float, float, float]:
    """
    Clamp and renormalize probabilities to sum to 1.
    Handles edge cases: negatives, >1, not summing to 1.
    """
    # Clamp to [0, 1]
    p_home = max(0.0, min(1.0, p_home or 0.0))
    p_draw = max(0.0, min(1.0, p_draw or 0.0))
    p_away = max(0.0, min(1.0, p_away or 0.0))

    total = p_home + p_draw + p_away

    # If total is 0 or very small, return uniform
    if total < 0.001:
        return (1/3, 1/3, 1/3)

    # Renormalize to sum to 1
    return (p_home / total, p_draw / total, p_away / total)


# Protocol v3 constants
PROTOCOL_VERSION = "3.0"
TIMING_WINDOW_VALID_MIN = 10  # minutes pre-kickoff
TIMING_WINDOW_VALID_MAX = 90  # minutes pre-kickoff
TIMING_WINDOW_IDEAL_MIN = 45
TIMING_WINDOW_IDEAL_MAX = 75
EDGE_THRESHOLD = 0.05  # 5% edge required to bet
BOOTSTRAP_ITERATIONS = 1000
MIN_BETS_FOR_CI = 30


def _parse_iso_date(date_str: str | None) -> str | None:
    """Parse ISO date (YYYY-MM-DD) as string guardrail."""
    if date_str is None:
        return None
    # Keep strict-ish: 10 chars and simple digits/hyphens
    s = date_str.strip()
    if len(s) != 10:
        raise ValueError(f"Invalid date format (expected YYYY-MM-DD): {date_str!r}")
    # Basic sanity; asyncpg will still handle actual date parsing if used as param,
    # but we embed dates into SQL strings elsewhere in this script.
    yyyy, mm, dd = s.split("-")
    if not (yyyy.isdigit() and mm.isdigit() and dd.isdigit()):
        raise ValueError(f"Invalid date format (expected YYYY-MM-DD): {date_str!r}")
    return s


async def fetch_pit_data(
    conn,
    min_snapshot_date: str | None = None,
    max_snapshot_date: str | None = None,
    league_ids: list[int] | None = None,
) -> list[dict]:
    """
    Fetch PIT-eligible snapshots with match results.
    Only reads from odds_snapshots + matches.

    Args:
        min_snapshot_date: Optional ISO date string (e.g., '2026-01-13') to filter snapshots
        max_snapshot_date: Optional ISO date string (e.g., '2026-01-20') to cap snapshots (exclusive)
        league_ids: Optional list of league IDs to filter (for feature coverage analysis)
    """
    where_clause = "WHERE os.snapshot_type = 'lineup_confirmed'"
    if min_snapshot_date:
        where_clause += f" AND os.snapshot_at >= '{min_snapshot_date}'"
    if max_snapshot_date:
        where_clause += f" AND os.snapshot_at < '{max_snapshot_date}'"
    if league_ids:
        ids_str = ",".join(str(lid) for lid in league_ids)
        where_clause += f" AND m.league_id IN ({ids_str})"

    query = f"""
        SELECT
            os.id as snapshot_id,
            os.match_id,
            os.snapshot_at,
            os.snapshot_type,
            os.odds_home,
            os.odds_draw,
            os.odds_away,
            os.prob_home,
            os.prob_draw,
            os.prob_away,
            os.odds_freshness,
            os.delta_to_kickoff_seconds,
            os.bookmaker,
            m.date as kickoff_time,
            m.status,
            m.home_goals,
            m.away_goals,
            m.league_id,
            m.season
        FROM odds_snapshots os
        JOIN matches m ON m.id = os.match_id
        {where_clause}
        ORDER BY os.snapshot_at DESC
    """
    rows = await conn.fetch(query)
    return [dict(r) for r in rows]


async def fetch_predictions(conn) -> tuple[list, dict]:
    """
    Fetch all model predictions if available.
    Returns tuple:
        - list: all prediction rows [{match_id, home_prob, draw_prob, away_prob, model_version, created_at?}, ...]
        - dict: metadata about prediction integrity
    """
    metadata = {
        'table_exists': False,
        'has_created_at': False,
        'pit_prediction_integrity': 'unknown',
    }

    # Check if predictions table exists
    check_query = """
        SELECT EXISTS (
            SELECT FROM information_schema.tables
            WHERE table_name = 'predictions'
        )
    """
    exists = await conn.fetchval(check_query)

    if not exists:
        return [], metadata

    metadata['table_exists'] = True

    # Check if created_at column exists
    col_check = """
        SELECT EXISTS (
            SELECT FROM information_schema.columns
            WHERE table_name = 'predictions' AND column_name = 'created_at'
        )
    """
    has_created_at = await conn.fetchval(col_check)
    metadata['has_created_at'] = has_created_at

    if has_created_at:
        query = """
            SELECT match_id, home_prob, draw_prob, away_prob, model_version, created_at
            FROM predictions
            ORDER BY match_id, created_at DESC
        """
        # Will be set to 'enforced' after filtering
        metadata['pit_prediction_integrity'] = 'timestamped_not_enforced'
    else:
        query = """
            SELECT match_id, home_prob, draw_prob, away_prob, model_version
            FROM predictions
        """
        metadata['pit_prediction_integrity'] = 'unknown_no_timestamp'

    try:
        rows = await conn.fetch(query)
        return [dict(r) for r in rows], metadata
    except Exception:
        return [], metadata


async def fetch_predictions_experiments(conn, model_version: str | None = None) -> tuple[list, dict]:
    """
    Fetch predictions from predictions_experiments table.

    Returns predictions keyed by snapshot_id for 1:1 matching with snapshots.
    Used when --source experiments is specified.
    """
    metadata = {
        'table_exists': False,
        'has_created_at': True,  # Always true for experiments
        'pit_prediction_integrity': 'enforced',  # Constraint in table
        'source': 'predictions_experiments',
        'keyed_by': 'snapshot_id',
    }

    # Check if predictions_experiments table exists
    check_query = """
        SELECT EXISTS (
            SELECT FROM information_schema.tables
            WHERE table_name = 'predictions_experiments'
        )
    """
    exists = await conn.fetchval(check_query)

    if not exists:
        return [], metadata

    metadata['table_exists'] = True

    # Build query with optional model_version filter (parameterized to avoid SQL injection)
    if model_version:
        query = """
            SELECT
                pe.snapshot_id,
                pe.match_id,
                pe.home_prob,
                pe.draw_prob,
                pe.away_prob,
                pe.model_version,
                pe.created_at,
                pe.snapshot_at
            FROM predictions_experiments pe
            WHERE pe.model_version = $1
            ORDER BY pe.snapshot_at
        """
        query_args = [model_version]
    else:
        query = """
            SELECT
                pe.snapshot_id,
                pe.match_id,
                pe.home_prob,
                pe.draw_prob,
                pe.away_prob,
                pe.model_version,
                pe.created_at,
                pe.snapshot_at
            FROM predictions_experiments pe
            ORDER BY pe.snapshot_at
        """
        query_args = []

    try:
        rows = await conn.fetch(query, *query_args)
        predictions = [dict(r) for r in rows]
        metadata['n_unique_snapshots'] = len(set(p['snapshot_id'] for p in predictions))
        return predictions, metadata
    except Exception as e:
        metadata['error'] = str(e)
        return [], metadata


async def fetch_shadow_predictions(conn, shadow_version: str | None = None) -> tuple[list, dict]:
    """
    Fetch predictions from shadow_predictions table (canonical source for shadow models).

    This is the authoritative source for shadow model predictions, per ABE directive.
    Uses asyncpg placeholders ($1) for parameterized queries.

    Args:
        conn: Database connection
        shadow_version: Optional shadow_version to filter (e.g., "v1.1.0-two_stage")

    Returns:
        - list: predictions [{match_id, home_prob, draw_prob, away_prob, model_version, created_at}, ...]
        - dict: metadata about prediction integrity
    """
    metadata = {
        'table_exists': False,
        'has_created_at': True,
        'pit_prediction_integrity': 'enforced',
        'source': 'shadow_predictions',
    }

    # Check if shadow_predictions table exists
    check_query = """
        SELECT EXISTS (
            SELECT FROM information_schema.tables
            WHERE table_name = 'shadow_predictions'
        )
    """
    exists = await conn.fetchval(check_query)

    if not exists:
        metadata['error'] = 'shadow_predictions table does not exist'
        return [], metadata

    metadata['table_exists'] = True

    # Build query with optional shadow_version filter (asyncpg placeholders $1)
    if shadow_version:
        query = """
            SELECT
                sp.match_id,
                sp.shadow_home_prob as home_prob,
                sp.shadow_draw_prob as draw_prob,
                sp.shadow_away_prob as away_prob,
                sp.shadow_version as model_version,
                sp.shadow_predicted as predicted,
                sp.shadow_correct as is_correct,
                sp.created_at
            FROM shadow_predictions sp
            WHERE sp.shadow_version = $1
            ORDER BY sp.match_id, sp.created_at DESC
        """
        query_args = [shadow_version]
    else:
        query = """
            SELECT
                sp.match_id,
                sp.shadow_home_prob as home_prob,
                sp.shadow_draw_prob as draw_prob,
                sp.shadow_away_prob as away_prob,
                sp.shadow_version as model_version,
                sp.shadow_predicted as predicted,
                sp.shadow_correct as is_correct,
                sp.created_at
            FROM shadow_predictions sp
            ORDER BY sp.match_id, sp.created_at DESC
        """
        query_args = []

    try:
        rows = await conn.fetch(query, *query_args)
        predictions = [dict(r) for r in rows]
        metadata['n_predictions'] = len(predictions)

        # Count unique shadow_versions
        version_counts = {}
        for p in predictions:
            mv = p.get('model_version', 'unknown')
            version_counts[mv] = version_counts.get(mv, 0) + 1
        metadata['shadow_versions_available'] = version_counts

        return predictions, metadata
    except Exception as e:
        metadata['error'] = str(e)
        return [], metadata


async def fetch_predictions_experiments_many(conn, model_versions: list[str]) -> tuple[list, dict]:
    """
    Fetch predictions from predictions_experiments for a set of model_versions.

    Designed for ablation/compare runs where we need multiple models on the same snapshot_ids.
    Returns raw rows; caller groups by model_version and snapshot_id.
    """
    metadata = {
        'table_exists': False,
        'has_created_at': True,
        'pit_prediction_integrity': 'enforced',
        'source': 'predictions_experiments',
        'keyed_by': 'snapshot_id',
    }

    if not model_versions:
        metadata['error'] = 'model_versions empty'
        return [], metadata

    check_query = """
        SELECT EXISTS (
            SELECT FROM information_schema.tables
            WHERE table_name = 'predictions_experiments'
        )
    """
    exists = await conn.fetchval(check_query)
    if not exists:
        return [], metadata
    metadata['table_exists'] = True

    query = """
        SELECT
            pe.snapshot_id,
            pe.match_id,
            pe.home_prob,
            pe.draw_prob,
            pe.away_prob,
            pe.model_version,
            pe.created_at,
            pe.snapshot_at
        FROM predictions_experiments pe
        WHERE pe.model_version = ANY($1)
        ORDER BY pe.snapshot_at
    """
    try:
        rows = await conn.fetch(query, model_versions)
        preds = [dict(r) for r in rows]
        metadata['n_predictions'] = len(preds)
        metadata['n_unique_snapshots'] = len(set(p['snapshot_id'] for p in preds))
        metadata['model_versions_requested'] = model_versions
        return preds, metadata
    except Exception as e:
        metadata['error'] = str(e)
        return [], metadata


def is_shadow_model_version(model_version: str | None) -> bool:
    """Check if model_version indicates a shadow model.

    Handles naming variants: shadow, two_stage, two-stage, twostage
    """
    if not model_version:
        return False
    mv_lower = model_version.lower()
    return (
        'shadow' in mv_lower or
        'two_stage' in mv_lower or
        'two-stage' in mv_lower or
        'twostage' in mv_lower
    )


def get_pit_safe_prediction(predictions_list: list, match_id: int, snapshot_at) -> dict | None:
    """
    Get the most recent prediction for match_id where created_at <= snapshot_at.
    Returns None if no PIT-safe prediction exists.

    Args:
        predictions_list: List of all predictions (sorted by match_id, created_at DESC)
        match_id: The match to find prediction for
        snapshot_at: The snapshot timestamp (prediction must be created before this)

    Returns:
        The PIT-safe prediction dict, or None if not found
    """
    # Filter predictions for this match
    match_preds = [p for p in predictions_list if p['match_id'] == match_id]

    if not match_preds:
        return None

    # If no created_at column, we can't enforce PIT - return latest (but flag it)
    if 'created_at' not in match_preds[0] or match_preds[0].get('created_at') is None:
        return match_preds[0]  # Return first (most recent), but integrity won't be 'enforced'

    # Handle timezone-naive comparison
    snapshot_at_naive = snapshot_at
    if hasattr(snapshot_at, 'tzinfo') and snapshot_at.tzinfo is not None:
        snapshot_at_naive = snapshot_at.replace(tzinfo=None)

    # Find most recent prediction created before snapshot_at
    for pred in match_preds:  # Already sorted DESC by created_at
        pred_created = pred['created_at']
        if pred_created is None:
            continue

        # Handle timezone
        if hasattr(pred_created, 'tzinfo') and pred_created.tzinfo is not None:
            pred_created = pred_created.replace(tzinfo=None)

        if pred_created < snapshot_at_naive:
            return pred

    # No prediction exists before snapshot_at
    return None


def calculate_delta_minutes(snapshot: dict) -> float:
    """Calculate delta_KO = kickoff_time - snapshot_at in minutes."""
    # Use pre-calculated delta if available
    if snapshot.get('delta_to_kickoff_seconds') is not None:
        return float(snapshot['delta_to_kickoff_seconds']) / 60.0

    # Calculate from timestamps
    kickoff = snapshot.get('kickoff_time')
    snapshot_at = snapshot.get('snapshot_at')

    if kickoff is None or snapshot_at is None:
        return -1  # Invalid

    # Handle timezone-naive comparison
    if hasattr(kickoff, 'tzinfo') and kickoff.tzinfo is not None:
        kickoff = kickoff.replace(tzinfo=None)
    if hasattr(snapshot_at, 'tzinfo') and snapshot_at.tzinfo is not None:
        snapshot_at = snapshot_at.replace(tzinfo=None)

    delta_seconds = (kickoff - snapshot_at).total_seconds()
    return delta_seconds / 60.0


def is_pit_valid(snapshot: dict, delta_min: float) -> bool:
    """Check if snapshot is a valid PIT per Protocol v2."""
    # Must be live odds
    if snapshot.get('odds_freshness') != 'live':
        return False

    # Must have valid odds
    odds_h = to_float(snapshot.get('odds_home'))
    odds_d = to_float(snapshot.get('odds_draw'))
    odds_a = to_float(snapshot.get('odds_away'))

    if not all([odds_h, odds_d, odds_a]):
        return False
    if not (odds_h > 1.0 and odds_d > 1.0 and odds_a > 1.0):
        return False

    # Must have FT result
    if snapshot.get('status') != 'FT':
        return False
    if snapshot.get('home_goals') is None or snapshot.get('away_goals') is None:
        return False

    # Timing: 10-90 min pre-kickoff
    if delta_min < TIMING_WINDOW_VALID_MIN or delta_min > TIMING_WINDOW_VALID_MAX:
        return False

    return True


def is_timing_ideal(delta_min: float) -> bool:
    """Check if timing is in ideal window (45-75 min)."""
    return TIMING_WINDOW_IDEAL_MIN <= delta_min <= TIMING_WINDOW_IDEAL_MAX


def get_result(snapshot: dict) -> int:
    """Get match result: 0=home win, 1=draw, 2=away win."""
    hg = snapshot['home_goals']
    ag = snapshot['away_goals']
    if hg > ag:
        return 0
    elif hg == ag:
        return 1
    else:
        return 2


def odds_to_probs_devig(odds_h, odds_d, odds_a) -> tuple[float, float, float]:
    """Convert odds to de-vigged probabilities."""
    odds_h = to_float(odds_h)
    odds_d = to_float(odds_d)
    odds_a = to_float(odds_a)
    raw_h = 1.0 / odds_h
    raw_d = 1.0 / odds_d
    raw_a = 1.0 / odds_a
    total = raw_h + raw_d + raw_a
    return (raw_h / total, raw_d / total, raw_a / total)


def multiclass_brier(y_true: list[int], y_proba: list[tuple]) -> float:
    """Calculate multiclass Brier score."""
    n = len(y_true)
    if n == 0:
        return None

    total = 0.0
    for i, (true_label, probs) in enumerate(zip(y_true, y_proba)):
        for j in range(3):
            actual = 1.0 if j == true_label else 0.0
            total += (probs[j] - actual) ** 2

    return total / n


def multiclass_logloss(y_true: list[int], y_proba: list[tuple], eps: float = 1e-15) -> float:
    """Calculate multiclass log loss (cross-entropy)."""
    n = len(y_true)
    if n == 0:
        return None

    total = 0.0
    for true_label, probs in zip(y_true, y_proba):
        p = max(eps, min(1 - eps, probs[true_label]))
        total -= np.log(p)

    return total / n


def calculate_ece(y_true: list[int], y_proba: list[tuple], n_bins: int = 10) -> dict:
    """Calculate Expected Calibration Error (ECE) and Maximum Calibration Error (MCE)."""
    n = len(y_true)
    if n == 0:
        return {'ece': None, 'mce': None}

    predictions = []
    for true_label, probs in zip(y_true, y_proba):
        for j in range(3):
            actual = 1 if j == true_label else 0
            predictions.append({'confidence': probs[j], 'correct': actual})

    predictions.sort(key=lambda x: x['confidence'])
    bin_size = len(predictions) // n_bins

    ece = 0.0
    mce = 0.0

    for i in range(n_bins):
        start = i * bin_size
        end = (i + 1) * bin_size if i < n_bins - 1 else len(predictions)
        bin_preds = predictions[start:end]

        if not bin_preds:
            continue

        avg_confidence = np.mean([p['confidence'] for p in bin_preds])
        accuracy = np.mean([p['correct'] for p in bin_preds])
        bin_weight = len(bin_preds) / len(predictions)
        calibration_error = abs(accuracy - avg_confidence)

        ece += bin_weight * calibration_error
        mce = max(mce, calibration_error)

    return {'ece': round(ece, 4), 'mce': round(mce, 4)}


def calculate_paired_differential(y_true, y_proba_model, y_proba_market, n_bootstrap=1000):
    """Calculate paired differential (model - market) with bootstrap CI."""
    n = len(y_true)
    if n < 10:
        return {'status': 'insufficient_n', 'n': n}

    brier_diffs = []
    logloss_diffs = []
    eps = 1e-15

    for true_label, model_probs, market_probs in zip(y_true, y_proba_model, y_proba_market):
        brier_model = sum((model_probs[j] - (1.0 if j == true_label else 0.0))**2 for j in range(3))
        brier_market = sum((market_probs[j] - (1.0 if j == true_label else 0.0))**2 for j in range(3))
        brier_diffs.append(brier_model - brier_market)

        logloss_model = -np.log(max(eps, model_probs[true_label]))
        logloss_market = -np.log(max(eps, market_probs[true_label]))
        logloss_diffs.append(logloss_model - logloss_market)

    brier_bootstrap = []
    logloss_bootstrap = []
    for _ in range(n_bootstrap):
        indices = [random.randint(0, n-1) for _ in range(n)]
        brier_bootstrap.append(np.mean([brier_diffs[i] for i in indices]))
        logloss_bootstrap.append(np.mean([logloss_diffs[i] for i in indices]))

    brier_bootstrap.sort()
    logloss_bootstrap.sort()
    ci_low = int(0.025 * n_bootstrap)
    ci_high = int(0.975 * n_bootstrap)

    return {
        'n': n,
        'brier_diff_mean': round(np.mean(brier_diffs), 4),
        'brier_diff_ci95': [round(brier_bootstrap[ci_low], 4), round(brier_bootstrap[ci_high], 4)],
        'logloss_diff_mean': round(np.mean(logloss_diffs), 4),
        'logloss_diff_ci95': [round(logloss_bootstrap[ci_low], 4), round(logloss_bootstrap[ci_high], 4)],
        'status': 'ok'
    }


TIMING_BUCKETS = [(10, 20, '10-20'), (20, 30, '20-30'), (30, 45, '30-45'), (45, 60, '45-60'), (60, 75, '60-75'), (75, 90, '75-90')]


def get_timing_bucket(delta_min: float) -> str:
    for low, high, label in TIMING_BUCKETS:
        if low <= delta_min < high:
            return label
    return 'other'


async def fetch_closing_odds(conn, match_ids: list[int]) -> dict:
    """Fetch closing odds (T5 from market_movement_snapshots)."""
    if not match_ids:
        return {}

    query = """
        SELECT match_id, odds_home, odds_draw, odds_away, prob_home, prob_draw, prob_away
        FROM market_movement_snapshots
        WHERE match_id = ANY($1) AND snapshot_type = 'T5'
        ORDER BY match_id, captured_at DESC
    """
    rows = await conn.fetch(query, match_ids)

    closing = {}
    for r in rows:
        mid = r['match_id']
        if mid not in closing:
            closing[mid] = {
                'odds_home': to_float(r['odds_home']),
                'odds_draw': to_float(r['odds_draw']),
                'odds_away': to_float(r['odds_away']),
                'prob_home': to_float(r['prob_home']),
                'prob_draw': to_float(r['prob_draw']),
                'prob_away': to_float(r['prob_away']),
            }
    return closing


def calculate_clv(bet_prob_devigged: float, close_prob_devigged: float) -> float:
    """
    Calculate CLV = (close_prob / bet_prob) - 1. Positive = good.

    IMPORTANT: Both probs must be de-vigged (normalized) for fair comparison.
    If bet_prob > close_prob, CLV is negative (we bought at worse price).
    If bet_prob < close_prob, CLV is positive (we got value).
    """
    if close_prob_devigged is None or close_prob_devigged <= 0:
        return None
    if bet_prob_devigged is None or bet_prob_devigged <= 0:
        return None
    return (close_prob_devigged / bet_prob_devigged) - 1


def calculate_betting_metrics(bets: list[dict]) -> dict:
    """
    Calculate ROI and observable metrics from list of bets.

    FASE 3C.0 (ABE revised):
    - ev_model is NOT a GO/NO-GO metric (it's model's belief, not observable)
    - Observable metrics: ROI, CLV, skill_vs_market
    - Gate 3C.0 checks: PIT integrity, n_bets >= 30 for CI, CLV availability
    """
    if not bets:
        return {
            'n_bets': 0,
            'roi': None,
            'ev_model': None,  # Renamed: model's expected value (NOT observable)
            'total_staked': 0,
            'total_returns': 0,
        }

    total_staked = len(bets)  # Flat 1 unit each
    total_returns = sum(b['returns'] for b in bets)

    roi = (total_returns - total_staked) / total_staked if total_staked > 0 else 0

    # ev_model: model's projected EV (p_model * odds - 1)
    # NOTE: This is NOT an observable metric. It reflects model's belief.
    # With miscalibrated model, ev_model can be high while profit is negative.
    # Kept for diagnostic purposes only, NOT for GO/NO-GO decisions.
    ev_model = np.mean([b['ev'] for b in bets]) if bets else 0

    return {
        'n_bets': len(bets),
        'roi': roi,
        'ev_model': round(ev_model, 4),  # Diagnostic only, not GO/NO-GO
        'total_staked': total_staked,
        'total_returns': total_returns,
    }


def bootstrap_ci(values: list[float], n_iterations: int = BOOTSTRAP_ITERATIONS, ci: float = 0.95) -> tuple:
    """Calculate bootstrap confidence interval."""
    if len(values) < MIN_BETS_FOR_CI:
        return (None, None, "insufficient_n")

    n = len(values)
    bootstrap_means = []

    for _ in range(n_iterations):
        sample = [random.choice(values) for _ in range(n)]
        bootstrap_means.append(np.mean(sample))

    bootstrap_means.sort()
    lower_idx = int((1 - ci) / 2 * n_iterations)
    upper_idx = int((1 + ci) / 2 * n_iterations)

    return (bootstrap_means[lower_idx], bootstrap_means[upper_idx], "ok")


MIN_PREDICTIONS_FOR_STABLE_METRICS = 30


def holm_bonferroni_adjust(p_values: list[float | None]) -> list[float | None]:
    """
    Holm-Bonferroni adjusted p-values.

    Keeps None as None.
    """
    indexed = [(i, p) for i, p in enumerate(p_values) if p is not None]
    if not indexed:
        return [None for _ in p_values]

    m = len(indexed)
    indexed.sort(key=lambda x: x[1])  # ascending by p
    adjusted = [None for _ in p_values]

    # Step-down procedure with monotonicity enforcement
    prev_adj = 0.0
    for rank, (idx, p) in enumerate(indexed, start=1):
        adj = (m - rank + 1) * float(p)
        adj = min(1.0, max(prev_adj, adj))
        adjusted[idx] = adj
        prev_adj = adj
    return adjusted


def _two_sided_p_from_bootstrap(deltas: list[float]) -> float | None:
    """Two-sided p-value from bootstrap delta distribution around 0."""
    if not deltas:
        return None
    arr = np.array(deltas, dtype=float)
    # Handle NaNs
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return None
    p_le = float(np.mean(arr <= 0.0))
    p_ge = float(np.mean(arr >= 0.0))
    return float(min(1.0, 2.0 * min(p_le, p_ge)))


def _percentile_ci(values: list[float], alpha: float = 0.05) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    arr = np.array(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return None, None
    low = float(np.quantile(arr, alpha / 2))
    high = float(np.quantile(arr, 1 - alpha / 2))
    return low, high


def _simulate_bet_for_snapshot(
    *,
    odds: tuple[float, float, float],
    probs_model: tuple[float, float, float],
    probs_market: tuple[float, float, float],
    result: int,
    edge_threshold: float,
    require_ev_positive: bool,
) -> dict:
    """
    Deterministic bet policy:
    - pick outcome with max edge (p_model - p_market)
    - bet if edge >= threshold and (optionally) EV>0
    Returns per-snapshot bet outcome with pnl, stake, clv placeholder.
    """
    edges = [probs_model[i] - probs_market[i] for i in range(3)]
    best_idx = int(np.argmax(edges))
    best_edge = float(edges[best_idx])
    ev_best = float(probs_model[best_idx] * odds[best_idx] - 1.0)

    place_bet = best_edge >= edge_threshold and (ev_best > 0 if require_ev_positive else True)
    if not place_bet:
        return {
            'bet': False,
            'bet_outcome': None,
            'edge': best_edge,
            'ev': ev_best,
            'stake': 0.0,
            'pnl': 0.0,
            'returns': 0.0,
        }

    won = (result == best_idx)
    returns = float(odds[best_idx] if won else 0.0)
    pnl = returns - 1.0  # flat 1 unit stake

    return {
        'bet': True,
        'bet_outcome': best_idx,
        'edge': best_edge,
        'ev': ev_best,
        'stake': 1.0,
        'pnl': pnl,
        'returns': returns,
    }


def _bootstrap_paired_delta(
    *,
    baseline_series: dict,
    candidate_series: dict,
    metric: str,
    n_bootstrap: int,
    seed: int,
) -> dict:
    """
    Paired bootstrap over snapshot indices.

    baseline_series / candidate_series are dicts with precomputed per-snapshot arrays:
      - brier_contrib: float per snapshot (sum squared error across 3 classes)
      - logloss_contrib: float per snapshot (-log p_true)
      - stake: 1 or 0 per snapshot
      - pnl: returns-1 per snapshot (0 when no bet)
      - clv: float or NaN per snapshot (NaN when missing)

    Supported metrics for bootstrap delta:
      - roi
      - clv_mean
      - brier_model
      - logloss_model
    """
    rng = np.random.default_rng(seed)
    n = int(baseline_series['n'])
    if n <= 0:
        return {'status': 'insufficient_n', 'n': n}

    idx_samples = rng.integers(0, n, size=(n_bootstrap, n), endpoint=False)

    def compute_metric(series: dict, sample_idx: np.ndarray) -> float:
        if metric == 'brier_model':
            return float(np.mean(series['brier_contrib'][sample_idx]))
        if metric == 'logloss_model':
            return float(np.mean(series['logloss_contrib'][sample_idx]))
        if metric == 'roi':
            stake_sum = float(np.sum(series['stake'][sample_idx]))
            if stake_sum <= 0:
                return float('nan')
            pnl_sum = float(np.sum(series['pnl'][sample_idx]))
            return pnl_sum / stake_sum
        if metric == 'clv_mean':
            clv = series['clv'][sample_idx]
            clv = clv[~np.isnan(clv)]
            if clv.size == 0:
                return float('nan')
            return float(np.mean(clv))
        raise ValueError(f"Unsupported metric for bootstrap: {metric}")

    deltas = []
    for b in range(n_bootstrap):
        sample_idx = idx_samples[b]
        base_v = compute_metric(baseline_series, sample_idx)
        cand_v = compute_metric(candidate_series, sample_idx)
        deltas.append(float(cand_v - base_v))

    ci_low, ci_high = _percentile_ci(deltas)
    p_val = _two_sided_p_from_bootstrap(deltas)

    # Point estimate on full sample (not bootstrapped)
    full_idx = np.arange(n)
    base_full = compute_metric(baseline_series, full_idx)
    cand_full = compute_metric(candidate_series, full_idx)
    delta_full = float(cand_full - base_full)

    return {
        'status': 'ok',
        'n': n,
        'metric': metric,
        'baseline': base_full,
        'candidate': cand_full,
        'delta': delta_full,
        'delta_ci95': [ci_low, ci_high],
        'p_value': p_val,
        'n_bootstrap': n_bootstrap,
    }


def generate_interpretation(phase: str, brier: dict, betting: dict) -> dict:
    """
    Generate interpretation block based on explicit rules.

    Rules:
    1. If phase=insufficient OR betting.roi_ci_status=insufficient_n => confidence='low', verdict='HOLD'
    2. If brier.skill_vs_market < 0 => add note 'model worse than market (early signal)'
    3. If ROI CI95 lower bound > 0 => confidence='high', verdict='GO (alpha)'
    4. If brier.n_with_predictions < 30 => add variance warning note

    Returns:
        {confidence: str, verdict: str, bullet_notes: list[str]}
    """
    confidence = "medium"  # default
    verdict = "HOLD"  # default (conservative)
    bullet_notes = []

    n_bets = betting.get('n_bets', 0)
    roi_ci_status = betting.get('roi_ci_status', 'no_bets')
    roi_ci95_low = betting.get('roi_ci95_low')
    skill_vs_market = brier.get('skill_vs_market')
    n_with_predictions = brier.get('n_with_predictions', 0)

    # Rule 1: insufficient data => low confidence, HOLD
    if phase == 'insufficient' or roi_ci_status == 'insufficient_n':
        confidence = "low"
        verdict = "HOLD"
        bullet_notes.append(f"insufficient_n: phase={phase}, n_bets={n_bets}, min=50 for CI")

    # Rule 2: model worse than market => add warning note
    if skill_vs_market is not None and skill_vs_market < 0:
        bullet_notes.append(f"model worse than market (early signal): skill_vs_market={skill_vs_market:.2%}")

    # Rule 3: ROI CI95 lower bound > 0 => high confidence, GO
    # Only applies if we have sufficient data (not insufficient phase)
    if roi_ci95_low is not None and roi_ci95_low > 0 and phase != 'insufficient':
        confidence = "high"
        verdict = "GO (alpha)"
        bullet_notes.append(f"ROI CI95 lower bound positive: {roi_ci95_low:.2%}")

    # Rule 4: low N predictions => high variance warning
    if n_with_predictions is not None and n_with_predictions < MIN_PREDICTIONS_FOR_STABLE_METRICS:
        bullet_notes.append(f"low_n_predictions: n_with_predictions={n_with_predictions} (<{MIN_PREDICTIONS_FOR_STABLE_METRICS}), metrics have high variance")

    # Additional context notes
    if roi_ci_status == 'no_bets':
        bullet_notes.append("no_bets: edge threshold not met by any prediction")
    elif roi_ci_status == 'ok' and confidence != "high":
        # Sufficient data but CI includes 0 => medium confidence
        if roi_ci95_low is not None and roi_ci95_low <= 0:
            bullet_notes.append(f"ROI CI95 includes zero: [{roi_ci95_low:.2%}, {betting.get('roi_ci95_high', 0):.2%}]")

    return {
        "confidence": confidence,
        "verdict": verdict,
        "bullet_notes": bullet_notes,
    }


async def run_evaluation(
    min_snapshot_date: str | None = None,
    max_snapshot_date: str | None = None,
    edge_threshold: float | None = None,
    devig_method: str = "proportional",
    calibrator_method: str = "none",
    calib_train_end: str | None = None,
    league_ids: list[int] | None = None,
    model_version: str | None = None,
    source: str = "predictions",
    require_ev_positive: bool = True,
) -> dict:
    """Main evaluation logic.

    Args:
        min_snapshot_date: Optional ISO date string (e.g., '2026-01-13') to filter snapshots
        max_snapshot_date: Optional ISO date string (exclusive) to cap snapshots
        edge_threshold: Optional edge threshold override (default: EDGE_THRESHOLD constant)
        devig_method: De-vig method ("proportional"=baseline, "power"=alternative)
        calibrator_method: Calibration method ("none"=baseline, "isotonic", "temperature")
        calib_train_end: Date to split calibration train/test (required if calibrator != none)
        league_ids: Optional list of league IDs to filter (for feature coverage analysis)
        model_version: Optional model version to filter predictions (e.g., "v1.0.0")

    FASE 3C (ABE): Calibration requires --calib-train-end to prevent data leakage.
    """
    # Use custom threshold if provided
    threshold = edge_threshold if edge_threshold is not None else EDGE_THRESHOLD

    # FASE 3C: Import de-vig and calibration modules
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from app.ml.devig import get_devig_function
    from app.ml.calibration import get_calibrator

    devig_fn = get_devig_function(devig_method)
    calibrator = get_calibrator(calibrator_method)

    # Validate calibration params
    if calibrator is not None and calib_train_end is None:
        return {
            'error': 'Calibration requires --calib-train-end to prevent data leakage',
            'generated_at': datetime.now().isoformat(),
        }
    database_url = os.environ.get('DATABASE_URL', '')
    if not database_url:
        return {
            'error': 'DATABASE_URL not set',
            'generated_at': datetime.now().isoformat(),
        }

    # Connect
    conn = await asyncpg.connect(database_url)

    try:
        # Fetch data
        snapshots = await fetch_pit_data(conn, min_snapshot_date, max_snapshot_date, league_ids)

        # Fetch predictions based on source
        # ABE directive: Shadow models must read from shadow_predictions table (canonical source)
        if source == "experiments":
            # Use predictions_experiments table (TITAN tier comparison)
            predictions_list, pred_metadata = await fetch_predictions_experiments(conn, model_version)
            pred_metadata['source'] = 'predictions_experiments'
        elif is_shadow_model_version(model_version):
            # Shadow model: read from shadow_predictions table (canonical source)
            predictions_list, pred_metadata = await fetch_shadow_predictions(conn, model_version)
            pred_metadata['source'] = 'shadow_predictions'
            pred_metadata['model_version_filter'] = model_version
            print(f"[INFO] Using shadow_predictions table for model_version={model_version}")
        else:
            # Use standard predictions table
            predictions_list, pred_metadata = await fetch_predictions(conn)
            pred_metadata['source'] = 'predictions'

            # Count model versions before filtering (only for standard predictions)
            model_version_counts = {}
            for p in predictions_list:
                mv = p.get('model_version', 'unknown')
                model_version_counts[mv] = model_version_counts.get(mv, 0) + 1
            pred_metadata['model_versions_available'] = model_version_counts

            # ABE P0-1: Auto-select model_version from model_snapshots.is_active (canonical)
            # Fallback: most recent created_at if no active snapshot (degraded integrity)
            if not model_version and len(model_version_counts) > 1:
                active_row = await conn.fetchrow(
                    "SELECT model_version FROM model_snapshots "
                    "WHERE is_active = true ORDER BY created_at DESC LIMIT 1"
                )
                if active_row:
                    auto_selected = active_row['model_version']
                    pred_metadata['model_version_source'] = 'model_snapshots.is_active'
                else:
                    # Fallback: most recent created_at in predictions (degraded)
                    latest_by_version = {}
                    for p in predictions_list:
                        mv = p.get('model_version', 'unknown')
                        cat = p.get('created_at')
                        if cat and (mv not in latest_by_version or cat > latest_by_version[mv]):
                            latest_by_version[mv] = cat
                    auto_selected = max(latest_by_version, key=latest_by_version.get) if latest_by_version else max(model_version_counts, key=model_version_counts.get)
                    pred_metadata['model_version_source'] = 'fallback_most_recent'
                    print(f"[WARNING] No active snapshot found — using fallback (integrity=degraded)")
                print(f"[WARNING] Multiple model_versions detected: {model_version_counts}")
                print(f"[WARNING] Auto-selecting: {auto_selected} (source: {pred_metadata.get('model_version_source', 'unknown')})")
                print(f"[WARNING] Use --model-version to explicitly choose a cohort.")
                model_version = auto_selected
                pred_metadata['model_version_auto_selected'] = True

            # Filter by model_version if specified (or auto-selected)
            if model_version:
                predictions_list = [p for p in predictions_list if p.get('model_version') == model_version]
                pred_metadata['model_version_filter'] = model_version
                pred_metadata['n_predictions_after_filter'] = len(predictions_list)

        # Coverage stats
        n_total = len(snapshots)
        n_live = sum(1 for s in snapshots if s.get('odds_freshness') == 'live')

        # Calculate delta for each snapshot
        for s in snapshots:
            s['delta_min'] = calculate_delta_minutes(s)

        n_pre_kickoff = sum(1 for s in snapshots if s['delta_min'] > 0)

        # Filter to valid PIT
        valid_pit = [s for s in snapshots if is_pit_valid(s, s['delta_min'])]
        n_valid_10_90 = len(valid_pit)
        n_valid_ideal = sum(1 for s in valid_pit if is_timing_ideal(s['delta_min']))

        # Match predictions to snapshots using PIT-safe logic
        # Only use predictions created BEFORE the snapshot timestamp
        pit_safe_predictions = {}  # match_id -> prediction (or snapshot_id for experiments)
        n_no_prediction_asof = 0
        n_with_prediction_any = 0  # Has prediction but maybe not PIT-safe

        can_enforce_pit = pred_metadata.get('has_created_at', False)

        if source == "experiments":
            # MODE: experiments - emparejamiento 1:1 por snapshot_id
            # CRITICAL: Key by snapshot_id (NOT match_id) for true 1:1 matching
            predictions_by_snapshot = {p['snapshot_id']: p for p in predictions_list}

            for s in valid_pit:
                snapshot_id = s['snapshot_id']  # odds_snapshots.id
                pred = predictions_by_snapshot.get(snapshot_id)

                if pred:
                    # PIT is enforced by table constraint (created_at <= snapshot_at)
                    # Key by snapshot_id for 1:1 matching
                    pit_safe_predictions[snapshot_id] = pred
                else:
                    n_no_prediction_asof += 1

            pred_metadata['keyed_by'] = 'snapshot_id'
        else:
            # MODE: predictions - emparejamiento por match_id (código original)
            for s in valid_pit:
                match_id = s['match_id']
                snapshot_at = s.get('snapshot_at')

                if can_enforce_pit and snapshot_at:
                    # PIT-safe: only use predictions created before snapshot
                    pred = get_pit_safe_prediction(predictions_list, match_id, snapshot_at)
                    if pred:
                        pit_safe_predictions[match_id] = pred
                    else:
                        # Check if there's ANY prediction for this match (just not PIT-safe)
                        any_pred = [p for p in predictions_list if p['match_id'] == match_id]
                        if any_pred:
                            n_with_prediction_any += 1
                        n_no_prediction_asof += 1
                else:
                    # No timestamps - can't enforce PIT, use any prediction
                    match_preds = [p for p in predictions_list if p['match_id'] == match_id]
                    if match_preds:
                        pit_safe_predictions[match_id] = match_preds[0]
                    else:
                        n_no_prediction_asof += 1

        # Update integrity status
        if can_enforce_pit:
            pred_metadata['pit_prediction_integrity'] = 'enforced'
        pred_metadata['n_predictions_pit_safe'] = len(pit_safe_predictions)
        pred_metadata['n_no_prediction_asof'] = n_no_prediction_asof
        pred_metadata['n_had_prediction_but_not_pit_safe'] = n_with_prediction_any

        # Timing distribution
        valid_deltas = [s['delta_min'] for s in valid_pit]
        if valid_deltas:
            timing_dist = {
                'p10': float(np.percentile(valid_deltas, 10)),
                'p50': float(np.percentile(valid_deltas, 50)),
                'p90': float(np.percentile(valid_deltas, 90)),
                'min': float(min(valid_deltas)),
                'max': float(max(valid_deltas)),
            }
        else:
            timing_dist = {'p10': None, 'p50': None, 'p90': None, 'min': None, 'max': None}

        # Breakdown by league
        league_counts = {}
        for s in valid_pit:
            lid = s.get('league_id')
            league_counts[lid] = league_counts.get(lid, 0) + 1
        top_leagues = sorted(league_counts.items(), key=lambda x: -x[1])[:10]

        # Breakdown by bookmaker
        bookmaker_counts = {}
        for s in valid_pit:
            bm = s.get('bookmaker', 'unknown')
            bookmaker_counts[bm] = bookmaker_counts.get(bm, 0) + 1
        top_bookmakers = sorted(bookmaker_counts.items(), key=lambda x: -x[1])[:10]

        # Brier calculation (for snapshots with PIT-safe model predictions)
        # Use snapshot_id for experiments (1:1), match_id for standard predictions
        if source == "experiments":
            pit_with_preds = [s for s in valid_pit if s['snapshot_id'] in pit_safe_predictions]
        else:
            pit_with_preds = [s for s in valid_pit if s['match_id'] in pit_safe_predictions]

        # FASE 3C: Split data for calibration if needed
        calib_train_data = None
        calib_test_data = pit_with_preds
        n_calib_train = 0
        n_calib_test = len(pit_with_preds)

        if calibrator is not None and calib_train_end:
            # Split: train on data < calib_train_end, test on data >= min_snapshot_date
            calib_train_data = []
            calib_test_data = []

            for s in pit_with_preds:
                snapshot_date = s.get('snapshot_at')
                if snapshot_date:
                    snapshot_date_str = snapshot_date.strftime('%Y-%m-%d') if hasattr(snapshot_date, 'strftime') else str(snapshot_date)[:10]
                    if snapshot_date_str < calib_train_end:
                        calib_train_data.append(s)
                    else:
                        calib_test_data.append(s)
                else:
                    calib_test_data.append(s)

            n_calib_train = len(calib_train_data)
            n_calib_test = len(calib_test_data)

            # Train calibrator if we have enough training data
            if n_calib_train >= 30:
                train_probs = []
                train_outcomes = []
                for s in calib_train_data:
                    pred_key = s['snapshot_id'] if source == "experiments" else s['match_id']
                    pred = pit_safe_predictions[pred_key]
                    raw_probs = (
                        to_float(pred.get('home_prob', 1/3)),
                        to_float(pred.get('draw_prob', 1/3)),
                        to_float(pred.get('away_prob', 1/3)),
                    )
                    train_probs.append(normalize_probs(*raw_probs))
                    train_outcomes.append(get_result(s))

                calibrator.fit(np.array(train_probs), np.array(train_outcomes))
            else:
                # Not enough training data, disable calibrator
                calibrator = None

        brier_results = {
            'n_with_predictions': len(calib_test_data),
            'brier_model': None,
            'brier_uniform': None,
            'brier_market': None,
            'skill_vs_uniform': None,
            'skill_vs_market': None,
            # FASE 3C: Calibration metadata
            'n_calib_train': n_calib_train,
            'n_calib_test': n_calib_test,
            'calibrator_used': calibrator_method if calibrator is not None else 'none',
            'devig_method': devig_method,
        }

        if calib_test_data:
            y_true = []
            y_proba_model = []
            y_proba_market = []
            y_proba_uniform = []

            for s in calib_test_data:
                pred_key = s['snapshot_id'] if source == "experiments" else s['match_id']
                pred = pit_safe_predictions[pred_key]
                result = get_result(s)
                y_true.append(result)

                # Model probabilities (normalized to sum to 1)
                raw_probs = (
                    to_float(pred.get('home_prob', 1/3)),
                    to_float(pred.get('draw_prob', 1/3)),
                    to_float(pred.get('away_prob', 1/3)),
                )
                model_probs = normalize_probs(*raw_probs)

                # FASE 3C: Apply calibration if fitted
                if calibrator is not None and calibrator.is_fitted:
                    calibrated = calibrator.transform(np.array([model_probs]))
                    model_probs = tuple(calibrated[0])

                y_proba_model.append(model_probs)

                # Market probabilities (using selected de-vig method)
                mkt_probs = devig_fn(
                    to_float(s['odds_home']),
                    to_float(s['odds_draw']),
                    to_float(s['odds_away'])
                )
                y_proba_market.append(mkt_probs)

                # Uniform
                y_proba_uniform.append((1/3, 1/3, 1/3))

            brier_model = multiclass_brier(y_true, y_proba_model)
            brier_market = multiclass_brier(y_true, y_proba_market)
            brier_uniform = multiclass_brier(y_true, y_proba_uniform)

            brier_results['brier_model'] = brier_model
            brier_results['brier_market'] = brier_market
            brier_results['brier_uniform'] = brier_uniform

            if brier_model and brier_uniform:
                brier_results['skill_vs_uniform'] = 1 - brier_model / brier_uniform
            if brier_model and brier_market:
                brier_results['skill_vs_market'] = 1 - brier_model / brier_market

            # LogLoss
            logloss_model = multiclass_logloss(y_true, y_proba_model)
            logloss_market = multiclass_logloss(y_true, y_proba_market)
            brier_results['logloss_model'] = logloss_model
            brier_results['logloss_market'] = logloss_market
            if logloss_model and logloss_market:
                brier_results['logloss_skill_vs_market'] = 1 - logloss_model / logloss_market

            # ECE
            ece_results = calculate_ece(y_true, y_proba_model)
            brier_results['ece'] = ece_results['ece']
            brier_results['mce'] = ece_results['mce']

            # Paired differential
            paired_diff = calculate_paired_differential(y_true, y_proba_model, y_proba_market)
            brier_results['paired_differential'] = paired_diff

        # Segmentation by timing bucket
        bucket_metrics = {}
        for s in pit_with_preds:
            bucket = get_timing_bucket(s['delta_min'])
            if bucket not in bucket_metrics:
                bucket_metrics[bucket] = {'y_true': [], 'y_model': [], 'y_market': []}

            pred_key = s['snapshot_id'] if source == "experiments" else s['match_id']
            pred = pit_safe_predictions[pred_key]
            result = get_result(s)
            raw_probs = (to_float(pred.get('home_prob', 1/3)), to_float(pred.get('draw_prob', 1/3)), to_float(pred.get('away_prob', 1/3)))
            model_probs = normalize_probs(*raw_probs)
            market_probs = devig_fn(to_float(s['odds_home']), to_float(s['odds_draw']), to_float(s['odds_away']))

            bucket_metrics[bucket]['y_true'].append(result)
            bucket_metrics[bucket]['y_model'].append(model_probs)
            bucket_metrics[bucket]['y_market'].append(market_probs)

        segmented_by_bucket = []
        for bucket in ['10-20', '20-30', '30-45', '45-60', '60-75', '75-90']:
            if bucket not in bucket_metrics or not bucket_metrics[bucket]['y_true']:
                segmented_by_bucket.append({'bucket': bucket, 'n': 0})
                continue

            bm = bucket_metrics[bucket]
            n = len(bm['y_true'])
            brier_m = multiclass_brier(bm['y_true'], bm['y_model'])
            brier_mkt = multiclass_brier(bm['y_true'], bm['y_market'])
            logloss_m = multiclass_logloss(bm['y_true'], bm['y_model'])
            ece_m = calculate_ece(bm['y_true'], bm['y_model'])

            segmented_by_bucket.append({
                'bucket': bucket, 'n': n,
                'brier_model': round(brier_m, 4) if brier_m else None,
                'brier_market': round(brier_mkt, 4) if brier_mkt else None,
                'skill_vs_market': round(1 - brier_m / brier_mkt, 4) if brier_m and brier_mkt else None,
                'logloss_model': round(logloss_m, 4) if logloss_m else None,
                'ece': ece_m['ece'],
            })

        # Fetch closing odds for CLV
        match_ids_for_clv = [s['match_id'] for s in calib_test_data]
        closing_odds = await fetch_closing_odds(conn, match_ids_for_clv)

        # Betting simulation (FASE 3C: use calib_test_data, not pit_with_preds)
        bets = []
        for s in calib_test_data:
            # Use PIT-safe prediction selected above
            pred_key = s['snapshot_id'] if source == "experiments" else s['match_id']
            pred = pit_safe_predictions[pred_key]
            odds = [to_float(s['odds_home']), to_float(s['odds_draw']), to_float(s['odds_away'])]

            # Normalize model probabilities
            raw_probs = (
                to_float(pred.get('home_prob', 0)),
                to_float(pred.get('draw_prob', 0)),
                to_float(pred.get('away_prob', 0)),
            )
            model_probs = list(normalize_probs(*raw_probs))

            # FASE 3C: Apply calibration if fitted
            if calibrator is not None and calibrator.is_fitted:
                calibrated = calibrator.transform(np.array([model_probs]))
                model_probs = list(calibrated[0])

            probs_model = model_probs

            # FASE 3C: Use selected de-vig method
            probs_market = devig_fn(odds[0], odds[1], odds[2])

            result = get_result(s)
            bet_outcome = _simulate_bet_for_snapshot(
                odds=(odds[0], odds[1], odds[2]),
                probs_model=(probs_model[0], probs_model[1], probs_model[2]),
                probs_market=(probs_market[0], probs_market[1], probs_market[2]),
                result=result,
                edge_threshold=threshold,
                require_ev_positive=require_ev_positive,
            )

            if bet_outcome['bet']:
                best_idx = bet_outcome['bet_outcome']
                best_edge = bet_outcome['edge']
                ev_best = bet_outcome['ev']
                returns = bet_outcome['returns']

                # CLV calculation (CRITICAL: both sides must use SAME devig method)
                # ABE fix: recalculate close_probs with devig_fn, don't use stored prob_*
                clv = None
                match_close = closing_odds.get(s['match_id'])
                if match_close:
                    # Use odds from closing snapshot and apply SAME devig_fn
                    close_odds_h = match_close.get('odds_home')
                    close_odds_d = match_close.get('odds_draw')
                    close_odds_a = match_close.get('odds_away')
                    if close_odds_h and close_odds_d and close_odds_a:
                        close_probs = devig_fn(close_odds_h, close_odds_d, close_odds_a)
                        bet_prob_devigged = probs_market[best_idx]
                        close_prob_devigged = close_probs[best_idx]
                        if close_prob_devigged and bet_prob_devigged:
                            clv = calculate_clv(bet_prob_devigged, close_prob_devigged)

                bets.append({
                    'match_id': s['match_id'],
                    'bet_outcome': best_idx,
                    'odds': odds[best_idx],
                    'ev': ev_best,
                    'edge': best_edge,
                    'won': (returns > 0),
                    'returns': returns,
                    'delta_min': s['delta_min'],
                    'bucket': get_timing_bucket(s['delta_min']),
                    'clv': clv,
                })

        betting_metrics = calculate_betting_metrics(bets)

        # Bootstrap CI for ROI (observable) and ev_model (diagnostic only)
        if bets:
            roi_values = [(b['returns'] - 1) for b in bets]  # P&L per bet

            roi_ci_low, roi_ci_high, roi_ci_status = bootstrap_ci(roi_values)

            betting_metrics['roi_ci95_low'] = roi_ci_low
            betting_metrics['roi_ci95_high'] = roi_ci_high
            betting_metrics['roi_ci_status'] = roi_ci_status
            # Note: ev_model CI removed from main output (not GO/NO-GO metric per ABE)

            # Win rate
            betting_metrics['win_rate'] = sum(1 for b in bets if b['won']) / len(bets)

            # CLV metrics (FASE 3C.0 ABE: CLV is observable, use for GO/NO-GO)
            clv_values = [b['clv'] for b in bets if b['clv'] is not None]
            if clv_values:
                betting_metrics['clv_mean'] = round(np.mean(clv_values), 4)
                betting_metrics['clv_median'] = round(np.median(clv_values), 4)
                betting_metrics['clv_positive_pct'] = round(sum(1 for c in clv_values if c > 0) / len(clv_values), 4)
                betting_metrics['clv_n'] = len(clv_values)
                # CLV CI95 bootstrap (ABE: observable metric for GO/NO-GO)
                clv_ci_low, clv_ci_high, clv_ci_status = bootstrap_ci(clv_values)
                betting_metrics['clv_ci95_low'] = clv_ci_low
                betting_metrics['clv_ci95_high'] = clv_ci_high
                betting_metrics['clv_ci_status'] = clv_ci_status
            else:
                betting_metrics['clv_mean'] = None
                betting_metrics['clv_n'] = 0
                betting_metrics['clv_ci95_low'] = None
                betting_metrics['clv_ci95_high'] = None
                betting_metrics['clv_ci_status'] = 'no_clv_data'

            # Betting audit: by pick
            pick_dist = {'home': 0, 'draw': 0, 'away': 0}
            pick_roi = {'home': [], 'draw': [], 'away': []}
            for b in bets:
                pick_name = ['home', 'draw', 'away'][b['bet_outcome']]
                pick_dist[pick_name] += 1
                pick_roi[pick_name].append(b['returns'] - 1)
            betting_metrics['pick_distribution'] = pick_dist
            betting_metrics['pick_roi'] = {k: round(np.mean(v), 4) if v else None for k, v in pick_roi.items()}

            # By odds range
            odds_ranges = {'<2.0': [], '2.0-3.0': [], '3.0-4.0': [], '>4.0': []}
            for b in bets:
                o = b['odds']
                pnl = b['returns'] - 1
                if o < 2.0:
                    odds_ranges['<2.0'].append(pnl)
                elif o < 3.0:
                    odds_ranges['2.0-3.0'].append(pnl)
                elif o < 4.0:
                    odds_ranges['3.0-4.0'].append(pnl)
                else:
                    odds_ranges['>4.0'].append(pnl)
            betting_metrics['roi_by_odds_range'] = {k: {'n': len(v), 'roi': round(np.mean(v), 4) if v else None} for k, v in odds_ranges.items()}

            # By timing bucket
            bucket_roi = {}
            for b in bets:
                bkt = b['bucket']
                if bkt not in bucket_roi:
                    bucket_roi[bkt] = []
                bucket_roi[bkt].append(b['returns'] - 1)
            betting_metrics['roi_by_bucket'] = {k: {'n': len(v), 'roi': round(np.mean(v), 4) if v else None} for k, v in bucket_roi.items()}

            # By edge decile (ABE: include CLV for observable monotonicidad)
            bets_sorted = sorted(bets, key=lambda x: x['edge'])
            n_bets = len(bets_sorted)
            decile_size = max(1, n_bets // 10)
            edge_deciles = []
            for i in range(10):
                start = i * decile_size
                end = (i + 1) * decile_size if i < 9 else n_bets
                db = bets_sorted[start:end]
                if db:
                    decile_clv = [b['clv'] for b in db if b['clv'] is not None]
                    edge_deciles.append({
                        'decile': i + 1, 'n': len(db),
                        'edge_range': [round(db[0]['edge'], 4), round(db[-1]['edge'], 4)],
                        'roi': round(np.mean([b['returns'] - 1 for b in db]), 4),
                        'clv_mean': round(np.mean(decile_clv), 4) if decile_clv else None,
                        'clv_n': len(decile_clv),
                    })
            betting_metrics['roi_by_edge_decile'] = edge_deciles

        else:
            betting_metrics['roi_ci95_low'] = None
            betting_metrics['roi_ci95_high'] = None
            betting_metrics['roi_ci_status'] = 'no_bets'
            betting_metrics['win_rate'] = None
            betting_metrics['clv_ci95_low'] = None
            betting_metrics['clv_ci95_high'] = None
            betting_metrics['clv_ci_status'] = 'no_bets'

        # Determine phase first (needed for interpretation)
        phase = (
            'insufficient' if n_valid_10_90 < 50 else
            'piloto' if n_valid_10_90 < 200 else
            'preliminar' if n_valid_10_90 < 500 else
            'formal'
        )

        # Generate interpretation based on rules
        interpretation = generate_interpretation(phase, brier_results, betting_metrics)

        # Build report
        filters_dict = {
            'snapshot_type': 'lineup_confirmed',
            'odds_freshness': 'live',
            'timing_window_valid': f'{TIMING_WINDOW_VALID_MIN}-{TIMING_WINDOW_VALID_MAX} min',
            'timing_window_ideal': f'{TIMING_WINDOW_IDEAL_MIN}-{TIMING_WINDOW_IDEAL_MAX} min',
            'edge_threshold': threshold,
            'require_ev_positive': require_ev_positive,
        }
        if min_snapshot_date:
            filters_dict['min_snapshot_date'] = min_snapshot_date
        if max_snapshot_date:
            filters_dict['max_snapshot_date'] = max_snapshot_date
        if league_ids:
            filters_dict['league_ids'] = league_ids
        if model_version:
            filters_dict['model_version'] = model_version

        report = {
            'generated_at': datetime.now().isoformat(),
            'protocol_version': PROTOCOL_VERSION,
            'filters': filters_dict,
            'counts': {
                'n_total_snapshots': n_total,
                'n_live': n_live,
                'n_pre_kickoff': n_pre_kickoff,
                'n_pit_valid_10_90': n_valid_10_90,
                'n_pit_valid_ideal_45_75': n_valid_ideal,
                'n_with_pit_safe_predictions': len(pit_with_preds),
                'n_no_prediction_asof': n_no_prediction_asof,
            },
            'timing_distribution': timing_dist,
            'breakdown_by_league': [{'league_id': lid, 'n': n} for lid, n in top_leagues],
            'breakdown_by_bookmaker': [{'bookmaker': bm, 'n': n} for bm, n in top_bookmakers],
            'brier': brier_results,
            'segmented_by_bucket': segmented_by_bucket,
            'betting': betting_metrics,
            'phase': phase,
            'interpretation': interpretation,
            'prediction_integrity': pred_metadata,
            'clv_note': 'CLV uses T5 (5 min pre-KO) from market_movement_snapshots as closing line',
            'notes': 'read-only evaluation; no writes to DB; probs normalized; PIT integrity enforced (pred.created_at < snapshot_at)',
        }

        return report

    finally:
        await conn.close()


def print_summary(report: dict):
    """Print short summary to stdout."""
    print("=" * 60)
    print("PIT EVALUATION - LIVE ODDS ONLY")
    print(f"Protocol v{report.get('protocol_version', '?')}")
    print("=" * 60)

    counts = report.get('counts', {})
    pred_integrity = report.get('prediction_integrity', {})
    print(f"\nCoverage (Auditor-traceable):")
    print(f"  N_total_snapshots:   {counts.get('n_total_snapshots', 0)}")
    print(f"  N_live:              {counts.get('n_live', 0)}")
    print(f"  N_valid_window:      {counts.get('n_pit_valid_10_90', 0)}  (10-90 min pre-KO)")
    print(f"  N_ideal_window:      {counts.get('n_pit_valid_ideal_45_75', 0)}  (45-75 min)")
    print(f"  N_pit_safe:          {counts.get('n_with_pit_safe_predictions', 0)}  (pred.created_at < snapshot_at)")
    print(f"  N_any_pred:          {pred_integrity.get('n_predictions_pit_safe', 0) + pred_integrity.get('n_had_prediction_but_not_pit_safe', 0)}")
    print(f"  N_no_pred_asof:      {counts.get('n_no_prediction_asof', 0)}")
    print(f"  N_not_pit_safe:      {pred_integrity.get('n_had_prediction_but_not_pit_safe', 0)}  (pred after snapshot)")
    print(f"  PIT integrity:       {pred_integrity.get('pit_prediction_integrity', 'unknown')}")

    brier = report.get('brier', {})
    # FASE 3C: Show calibration config
    if brier.get('calibrator_used') or brier.get('devig_method'):
        print(f"\nFASE 3C Config:")
        print(f"  De-vig method:       {brier.get('devig_method', 'proportional')}")
        print(f"  Calibrator:          {brier.get('calibrator_used', 'none')}")
        if brier.get('n_calib_train', 0) > 0:
            print(f"  N train (calib):     {brier.get('n_calib_train', 0)}")
        print(f"  N test (eval):       {brier.get('n_calib_test', brier.get('n_with_predictions', 0))}")

    if brier.get('brier_model') is not None:
        print(f"\nBrier (calibration):")
        print(f"  Model:               {brier['brier_model']:.4f}")
        print(f"  Market:              {brier.get('brier_market', 0):.4f}")
        print(f"  Uniform:             {brier.get('brier_uniform', 0):.4f}")
        print(f"  Skill vs uniform:    {brier.get('skill_vs_uniform', 0):.2%}")
        print(f"  Skill vs market:     {brier.get('skill_vs_market', 0):.2%}")

    betting = report.get('betting', {})
    print(f"\nBetting (primary - observable metrics):")
    print(f"  N bets:              {betting.get('n_bets', 0)}")
    if betting.get('roi') is not None:
        print(f"  ROI:                 {betting['roi']:.2%}")
        if betting.get('roi_ci95_low') is not None:
            print(f"  ROI CI95%:           [{betting['roi_ci95_low']:.2%}, {betting['roi_ci95_high']:.2%}]")
        print(f"  Win rate:            {betting.get('win_rate', 0):.2%}")

    # CLV metrics (ABE: observable, use for GO/NO-GO)
    if betting.get('clv_n', 0) > 0:
        print(f"\nCLV (Closing Line Value - observable):")
        print(f"  CLV mean:            {betting.get('clv_mean', 0):.4f}")
        print(f"  CLV median:          {betting.get('clv_median', 0):.4f}")
        if betting.get('clv_ci95_low') is not None:
            print(f"  CLV CI95%:           [{betting['clv_ci95_low']:.4f}, {betting['clv_ci95_high']:.4f}]")
        print(f"  CLV positive %:      {betting.get('clv_positive_pct', 0):.2%}")
        print(f"  CLV n:               {betting.get('clv_n', 0)}")

    # ev_model (diagnostic only, NOT GO/NO-GO)
    if betting.get('ev_model') is not None:
        print(f"\nEV Model (diagnostic only - NOT observable):")
        print(f"  ev_model:            {betting.get('ev_model', 0):.4f}  (model's belief, not real profit)")

    print(f"\nPhase: {report.get('phase', 'unknown')}")

    interpretation = report.get('interpretation', {})
    if interpretation:
        print(f"\nInterpretation:")
        print(f"  Confidence:          {interpretation.get('confidence', 'unknown')}")
        print(f"  Verdict:             {interpretation.get('verdict', 'unknown')}")
        for note in interpretation.get('bullet_notes', []):
            print(f"  • {note}")
    print("=" * 60)


async def main():
    parser = argparse.ArgumentParser(description='PIT Evaluation - Live Odds Only')
    parser.add_argument(
        '--min-snapshot-date',
        type=str,
        default=None,
        help='Minimum snapshot date (ISO format, e.g., 2026-01-13). Excludes earlier snapshots.'
    )
    parser.add_argument(
        '--max-snapshot-date',
        type=str,
        default=None,
        help='Maximum snapshot date (ISO format, exclusive). Caps snapshots to < this date.'
    )
    parser.add_argument(
        '--edge-threshold',
        type=float,
        default=None,
        help='Edge threshold for betting (e.g., 0.10 for 10%%). Overrides default 5%%.'
    )
    # FASE 3C: De-vig and calibration flags
    parser.add_argument(
        '--devig',
        type=str,
        choices=['proportional', 'power'],
        default='proportional',
        help='De-vig method: proportional (baseline) or power (alternative)'
    )
    parser.add_argument(
        '--calibrator',
        type=str,
        choices=['none', 'isotonic', 'temperature'],
        default='none',
        help='Calibration method: none (baseline), isotonic, or temperature'
    )
    parser.add_argument(
        '--calib-train-end',
        type=str,
        default=None,
        help='Date to split calibration train/test (ISO format). REQUIRED if --calibrator != none'
    )
    parser.add_argument(
        '--league-ids',
        type=str,
        default=None,
        help='Comma-separated list of league IDs to filter (e.g., "40,39,135")'
    )
    parser.add_argument(
        '--model-version',
        type=str,
        default=None,
        help='Filter predictions by model_version (e.g., "v1.0.0", "v1.1.0-two_stage")'
    )
    parser.add_argument(
        '--source',
        type=str,
        choices=['predictions', 'experiments'],
        default='predictions',
        help='Source table for predictions (default: predictions, use experiments for tier comparison)'
    )
    parser.add_argument(
        '--require-ev-positive',
        action='store_true',
        help='Require EV>0 in addition to edge threshold for placing a bet (default: enabled).'
    )
    parser.add_argument(
        '--allow-ev-negative',
        action='store_true',
        help='Allow betting even if EV<=0 (still requires edge threshold).'
    )
    parser.add_argument(
        '--compare',
        action='store_true',
        help='Compare baseline vs multiple model_versions (ablation-style) using predictions_experiments.'
    )
    parser.add_argument(
        '--baseline-model-version',
        type=str,
        default=None,
        help='Baseline model_version for --compare (required).'
    )
    parser.add_argument(
        '--compare-model-versions',
        type=str,
        default=None,
        help='Comma-separated candidate model_versions for --compare (required).'
    )
    parser.add_argument(
        '--compare-metric',
        type=str,
        choices=['roi', 'clv_mean', 'brier_model', 'logloss_model'],
        default='roi',
        help='Metric for paired bootstrap delta and Holm-Bonferroni p-adjust in --compare.'
    )
    parser.add_argument(
        '--compare-seed',
        type=int,
        default=42,
        help='Random seed for paired bootstrap in --compare.'
    )
    parser.add_argument(
        '--compare-iterations',
        type=int,
        default=1000,
        help='Bootstrap iterations for --compare.'
    )
    parser.add_argument(
        '--compare-output-csv',
        type=str,
        default=None,
        help='Optional CSV output path for --compare table.'
    )
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Normalize date inputs early (guardrail)
    try:
        args.min_snapshot_date = _parse_iso_date(args.min_snapshot_date)
        args.max_snapshot_date = _parse_iso_date(args.max_snapshot_date)
        args.calib_train_end = _parse_iso_date(args.calib_train_end) if args.calib_train_end else None
    except Exception as e:
        print(f"ERROR: {e}")
        return 1

    # FASE 3C: Validate calibration params
    if args.calibrator != 'none' and args.calib_train_end is None:
        print("ERROR: --calib-train-end is REQUIRED when using --calibrator")
        print("       This prevents data leakage in calibration.")
        return 1

    # Parse league_ids
    league_ids = None
    if args.league_ids:
        league_ids = [int(lid.strip()) for lid in args.league_ids.split(",")]
        print(f"Filtering to {len(league_ids)} leagues: {league_ids[:5]}{'...' if len(league_ids) > 5 else ''}")

    if args.model_version:
        print(f"Filtering predictions to model_version: {args.model_version}")

    if args.min_snapshot_date:
        print(f"Running PIT evaluation (live odds only, snapshot >= {args.min_snapshot_date})...")
    else:
        print("Running PIT evaluation (live odds only)...")
    if args.max_snapshot_date:
        print(f"  snapshot < {args.max_snapshot_date} (cap)")

    # Resolve bet policy flags
    require_ev_positive = True
    if args.allow_ev_negative:
        require_ev_positive = False
    if args.require_ev_positive:
        require_ev_positive = True

    # Print config
    print(f"Config: devig={args.devig}, calibrator={args.calibrator}, require_ev_positive={require_ev_positive}", end="")
    if args.calib_train_end:
        print(f", calib_train_end={args.calib_train_end}")
    else:
        print()

    if args.edge_threshold:
        print(f"Using custom edge threshold: {args.edge_threshold:.0%}")

    if args.source == 'experiments':
        print("Using predictions_experiments table")

    # --compare mode (ablation-style): baseline vs candidates (paired bootstrap deltas)
    if args.compare:
        if args.source != 'experiments':
            print("ERROR: --compare currently requires --source experiments (1:1 snapshot_id matching).")
            return 1
        if not args.baseline_model_version or not args.compare_model_versions:
            print("ERROR: --compare requires --baseline-model-version and --compare-model-versions.")
            return 1

        baseline_mv = args.baseline_model_version.strip()
        candidates = [s.strip() for s in args.compare_model_versions.split(",") if s.strip()]
        if not candidates:
            print("ERROR: --compare-model-versions is empty after parsing.")
            return 1

        database_url = os.environ.get('DATABASE_URL', '')
        if not database_url:
            print("ERROR: DATABASE_URL not set")
            return 1

        conn = await asyncpg.connect(database_url)
        try:
            snapshots = await fetch_pit_data(conn, args.min_snapshot_date, args.max_snapshot_date, league_ids)
            for s in snapshots:
                s['delta_min'] = calculate_delta_minutes(s)
            valid_pit = [s for s in snapshots if is_pit_valid(s, s['delta_min'])]

            # Fetch needed model_versions from experiments
            all_versions = [baseline_mv] + candidates
            preds_rows, preds_meta = await fetch_predictions_experiments_many(conn, all_versions)
            if not preds_meta.get('table_exists'):
                print("ERROR: predictions_experiments table not available.")
                return 1

            preds_by_model: dict[str, dict[int, dict]] = {}
            for r in preds_rows:
                mv = r.get('model_version')
                sid = r.get('snapshot_id')
                if not mv or sid is None:
                    continue
                preds_by_model.setdefault(mv, {})[sid] = r

            # Cohort: intersection across baseline + all candidates (strict paired)
            snapshot_ids = [s['snapshot_id'] for s in valid_pit]
            cohort = set(snapshot_ids)
            for mv in all_versions:
                cohort &= set(preds_by_model.get(mv, {}).keys())

            cohort_list = sorted(cohort)
            if not cohort_list:
                print("ERROR: Empty paired cohort (no common snapshot_ids across versions).")
                return 1

            # Build snapshot lookup for cohort
            snap_by_id = {s['snapshot_id']: s for s in valid_pit}
            cohort_snaps = [snap_by_id[sid] for sid in cohort_list if sid in snap_by_id]
            # If some ids missing in valid_pit map, shrink cohort (shouldn't happen)
            cohort_ids_final = [s['snapshot_id'] for s in cohort_snaps]

            # Shared market quantities per snapshot (contract fixed)
            # Import de-vig & calibrator hooks from app (same as run_evaluation)
            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent))
            from app.ml.devig import get_devig_function

            devig_fn = get_devig_function(args.devig)
            threshold = args.edge_threshold if args.edge_threshold is not None else EDGE_THRESHOLD

            # Closing odds for CLV
            match_ids_for_clv = [s['match_id'] for s in cohort_snaps]
            closing_odds = await fetch_closing_odds(conn, match_ids_for_clv)

            def build_series(mv: str) -> dict:
                brier_contrib = []
                logloss_contrib = []
                stake = []
                pnl = []
                clv = []

                eps = 1e-15
                preds_map = preds_by_model[mv]
                for s in cohort_snaps:
                    sid = s['snapshot_id']
                    pred = preds_map[sid]
                    result = get_result(s)

                    raw_probs = (
                        to_float(pred.get('home_prob', 1/3)),
                        to_float(pred.get('draw_prob', 1/3)),
                        to_float(pred.get('away_prob', 1/3)),
                    )
                    model_probs = normalize_probs(*raw_probs)
                    odds = (to_float(s['odds_home']), to_float(s['odds_draw']), to_float(s['odds_away']))
                    market_probs = devig_fn(odds[0], odds[1], odds[2])

                    # Per-snapshot brier/logloss contributions
                    bc = sum((model_probs[j] - (1.0 if j == result else 0.0)) ** 2 for j in range(3))
                    brier_contrib.append(float(bc))
                    ll = -np.log(max(eps, float(model_probs[result])))
                    logloss_contrib.append(float(ll))

                    bet_outcome = _simulate_bet_for_snapshot(
                        odds=odds,
                        probs_model=model_probs,
                        probs_market=market_probs,
                        result=result,
                        edge_threshold=threshold,
                        require_ev_positive=require_ev_positive,
                    )
                    stake.append(float(bet_outcome['stake']))
                    pnl.append(float(bet_outcome['pnl']))

                    # CLV only when bet placed and closing snapshot exists
                    clv_val = float('nan')
                    if bet_outcome['bet']:
                        match_close = closing_odds.get(s['match_id'])
                        if match_close:
                            close_odds = (
                                match_close.get('odds_home'),
                                match_close.get('odds_draw'),
                                match_close.get('odds_away'),
                            )
                            if close_odds[0] and close_odds[1] and close_odds[2]:
                                close_probs = devig_fn(close_odds[0], close_odds[1], close_odds[2])
                                bet_probs = market_probs
                                bet_prob_devigged = bet_probs[int(bet_outcome['bet_outcome'])]
                                close_prob_devigged = close_probs[int(bet_outcome['bet_outcome'])]
                                clv_calc = calculate_clv(bet_prob_devigged, close_prob_devigged)
                                if clv_calc is not None:
                                    clv_val = float(clv_calc)
                    clv.append(clv_val)

                return {
                    'n': len(cohort_snaps),
                    'brier_contrib': np.array(brier_contrib, dtype=float),
                    'logloss_contrib': np.array(logloss_contrib, dtype=float),
                    'stake': np.array(stake, dtype=float),
                    'pnl': np.array(pnl, dtype=float),
                    'clv': np.array(clv, dtype=float),
                }

            baseline_series = build_series(baseline_mv)
            candidate_series_map = {mv: build_series(mv) for mv in candidates}

            # Compute paired deltas (bootstrap) for each candidate vs baseline
            comparisons = []
            pvals = []
            for mv in candidates:
                res = _bootstrap_paired_delta(
                    baseline_series=baseline_series,
                    candidate_series=candidate_series_map[mv],
                    metric=args.compare_metric,
                    n_bootstrap=int(args.compare_iterations),
                    seed=int(args.compare_seed),
                )
                comparisons.append({
                    'baseline_model_version': baseline_mv,
                    'candidate_model_version': mv,
                    **res,
                })
                pvals.append(res.get('p_value'))

            p_adj = holm_bonferroni_adjust(pvals)
            for i, adj in enumerate(p_adj):
                comparisons[i]['p_adj_holm'] = adj

            # Optional CSV
            if args.compare_output_csv:
                out_path = Path(args.compare_output_csv)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                with open(out_path, "w", newline="") as f:
                    writer = csv.DictWriter(
                        f,
                        fieldnames=[
                            "baseline_model_version",
                            "candidate_model_version",
                            "metric",
                            "n",
                            "baseline",
                            "candidate",
                            "delta",
                            "delta_ci95_low",
                            "delta_ci95_high",
                            "p_value",
                            "p_adj_holm",
                            "n_bootstrap",
                            "status",
                        ],
                    )
                    writer.writeheader()
                    for row in comparisons:
                        ci = row.get('delta_ci95') or [None, None]
                        writer.writerow({
                            "baseline_model_version": row.get("baseline_model_version"),
                            "candidate_model_version": row.get("candidate_model_version"),
                            "metric": row.get("metric"),
                            "n": row.get("n"),
                            "baseline": row.get("baseline"),
                            "candidate": row.get("candidate"),
                            "delta": row.get("delta"),
                            "delta_ci95_low": ci[0],
                            "delta_ci95_high": ci[1],
                            "p_value": row.get("p_value"),
                            "p_adj_holm": row.get("p_adj_holm"),
                            "n_bootstrap": row.get("n_bootstrap"),
                            "status": row.get("status"),
                        })
                print(f"\n[OK] Compare CSV saved to: {out_path}")

            # Save JSON report alongside normal logs (without running run_evaluation)
            logs_dir = Path(__file__).parent.parent / "logs"
            logs_dir.mkdir(exist_ok=True)
            suffix = f"_from_{args.min_snapshot_date}" if args.min_snapshot_date else ""
            if args.max_snapshot_date:
                suffix += f"_to_{args.max_snapshot_date}"
            json_path = logs_dir / f"pit_compare_v3_{timestamp}{suffix}_{baseline_mv.replace('.', '_')}.json"
            report = {
                "generated_at": datetime.now().isoformat(),
                "mode": "compare",
                "protocol_version": PROTOCOL_VERSION,
                "filters": {
                    "min_snapshot_date": args.min_snapshot_date,
                    "max_snapshot_date": args.max_snapshot_date,
                    "league_ids": league_ids,
                    "source": "predictions_experiments",
                    "devig_method": args.devig,
                    "edge_threshold": threshold,
                    "require_ev_positive": require_ev_positive,
                    "cohort_mode": "intersection_all_versions",
                    "paired_n_snapshots": len(cohort_snaps),
                    "compare_metric": args.compare_metric,
                    "compare_seed": args.compare_seed,
                    "compare_iterations": args.compare_iterations,
                },
                "prediction_integrity": preds_meta,
                "baseline_model_version": baseline_mv,
                "candidate_model_versions": candidates,
                "comparisons": comparisons,
                "notes": "read-only compare; paired bootstrap over snapshot_id intersection; ROI uses bet-only denominator; CLV uses T5 as closing line",
            }
            with open(json_path, "w") as f:
                json.dump(report, f, indent=2, default=str)
            print(f"\nResultados guardados en: {json_path}")
            return 0
        finally:
            await conn.close()

    try:
        report = await run_evaluation(
            min_snapshot_date=args.min_snapshot_date,
            max_snapshot_date=args.max_snapshot_date,
            edge_threshold=args.edge_threshold,
            devig_method=args.devig,
            calibrator_method=args.calibrator,
            calib_train_end=args.calib_train_end,
            league_ids=league_ids,
            model_version=args.model_version,
            source=args.source,
            require_ev_positive=require_ev_positive,
        )
    except Exception as e:
        report = {
            'generated_at': datetime.now().isoformat(),
            'error': str(e),
            'phase': 'error',
        }
        print(f"ERROR: {e}")

    # Print summary
    print_summary(report)

    # Save JSON
    logs_dir = Path(__file__).parent.parent / "logs"
    logs_dir.mkdir(exist_ok=True)

    suffix = f"_from_{args.min_snapshot_date}" if args.min_snapshot_date else ""
    if league_ids:
        suffix += f"_filtered_{len(league_ids)}leagues"
    if args.model_version:
        suffix += f"_{args.model_version.replace('.', '_')}"
    json_path = logs_dir / f"pit_evaluation_v3_{timestamp}{suffix}.json"
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\nResultados guardados en: {json_path}")

    # Exit 0 even if insufficient data (per spec)
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
