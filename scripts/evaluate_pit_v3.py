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
import json
import os
import random
from datetime import datetime
from decimal import Decimal
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


async def fetch_pit_data(conn, min_snapshot_date: str | None = None) -> list[dict]:
    """
    Fetch PIT-eligible snapshots with match results.
    Only reads from odds_snapshots + matches.

    Args:
        min_snapshot_date: Optional ISO date string (e.g., '2026-01-13') to filter snapshots
    """
    where_clause = "WHERE os.snapshot_type = 'lineup_confirmed'"
    if min_snapshot_date:
        where_clause += f" AND os.snapshot_at >= '{min_snapshot_date}'"

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
    edge_threshold: float | None = None,
    devig_method: str = "proportional",
    calibrator_method: str = "none",
    calib_train_end: str | None = None,
) -> dict:
    """Main evaluation logic.

    Args:
        min_snapshot_date: Optional ISO date string (e.g., '2026-01-13') to filter snapshots
        edge_threshold: Optional edge threshold override (default: EDGE_THRESHOLD constant)
        devig_method: De-vig method ("proportional"=baseline, "power"=alternative)
        calibrator_method: Calibration method ("none"=baseline, "isotonic", "temperature")
        calib_train_end: Date to split calibration train/test (required if calibrator != none)

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
        snapshots = await fetch_pit_data(conn, min_snapshot_date)
        predictions_list, pred_metadata = await fetch_predictions(conn)

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
        pit_safe_predictions = {}  # match_id -> prediction
        n_no_prediction_asof = 0
        n_with_prediction_any = 0  # Has prediction but maybe not PIT-safe

        can_enforce_pit = pred_metadata.get('has_created_at', False)

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
                    pred = pit_safe_predictions[s['match_id']]
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
                pred = pit_safe_predictions[s['match_id']]
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

            pred = pit_safe_predictions[s['match_id']]
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
            pred = pit_safe_predictions[s['match_id']]
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

            # Calculate edges
            edges = [probs_model[i] - probs_market[i] for i in range(3)]
            best_idx = np.argmax(edges)
            best_edge = edges[best_idx]

            # EV for best outcome
            ev_best = probs_model[best_idx] * odds[best_idx] - 1

            # Only bet if edge >= threshold and EV > 0
            if best_edge >= threshold and ev_best > 0:
                result = get_result(s)
                won = (result == best_idx)
                returns = odds[best_idx] if won else 0

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
                    'won': won,
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
        }
        if min_snapshot_date:
            filters_dict['min_snapshot_date'] = min_snapshot_date

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
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # FASE 3C: Validate calibration params
    if args.calibrator != 'none' and args.calib_train_end is None:
        print("ERROR: --calib-train-end is REQUIRED when using --calibrator")
        print("       This prevents data leakage in calibration.")
        return 1

    if args.min_snapshot_date:
        print(f"Running PIT evaluation (live odds only, snapshot >= {args.min_snapshot_date})...")
    else:
        print("Running PIT evaluation (live odds only)...")

    # Print FASE 3C config
    print(f"Config: devig={args.devig}, calibrator={args.calibrator}", end="")
    if args.calib_train_end:
        print(f", calib_train_end={args.calib_train_end}")
    else:
        print()

    if args.edge_threshold:
        print(f"Using custom edge threshold: {args.edge_threshold:.0%}")

    try:
        report = await run_evaluation(
            min_snapshot_date=args.min_snapshot_date,
            edge_threshold=args.edge_threshold,
            devig_method=args.devig,
            calibrator_method=args.calibrator,
            calib_train_end=args.calib_train_end,
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
    json_path = logs_dir / f"pit_evaluation_v3_{timestamp}{suffix}.json"
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\nResultados guardados en: {json_path}")

    # Exit 0 even if insufficient data (per spec)
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
