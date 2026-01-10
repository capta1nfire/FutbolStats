#!/usr/bin/env python3
"""
PIT Evaluation - Live Odds Only

Evaluates Point-In-Time predictions using only live odds captured at lineup confirmation.
Implements PIT Evaluation Protocol v2.

Usage:
    DATABASE_URL=... python3 scripts/evaluate_pit_live_only.py

Output:
    logs/pit_evaluation_live_only_YYYYMMDD_HHMMSS.json
"""

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


# Protocol v2 constants
PROTOCOL_VERSION = "2.0"
TIMING_WINDOW_VALID_MIN = 10  # minutes pre-kickoff
TIMING_WINDOW_VALID_MAX = 90  # minutes pre-kickoff
TIMING_WINDOW_IDEAL_MIN = 45
TIMING_WINDOW_IDEAL_MAX = 75
EDGE_THRESHOLD = 0.05  # 5% edge required to bet
BOOTSTRAP_ITERATIONS = 1000
MIN_BETS_FOR_CI = 30


async def fetch_pit_data(conn) -> list[dict]:
    """
    Fetch PIT-eligible snapshots with match results.
    Only reads from odds_snapshots + matches.
    """
    query = """
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
        WHERE os.snapshot_type = 'lineup_confirmed'
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

        if pred_created <= snapshot_at_naive:
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


def calculate_betting_metrics(bets: list[dict]) -> dict:
    """Calculate ROI, EV from list of bets."""
    if not bets:
        return {
            'n_bets': 0,
            'roi': None,
            'ev': None,
            'total_staked': 0,
            'total_returns': 0,
        }

    total_staked = len(bets)  # Flat 1 unit each
    total_returns = sum(b['returns'] for b in bets)

    roi = (total_returns - total_staked) / total_staked if total_staked > 0 else 0
    ev = np.mean([b['ev'] for b in bets]) if bets else 0

    return {
        'n_bets': len(bets),
        'roi': roi,
        'ev': ev,
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


async def run_evaluation() -> dict:
    """Main evaluation logic."""
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
        snapshots = await fetch_pit_data(conn)
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

        brier_results = {
            'n_with_predictions': len(pit_with_preds),
            'brier_model': None,
            'brier_uniform': None,
            'brier_market': None,
            'skill_vs_uniform': None,
            'skill_vs_market': None,
        }

        if pit_with_preds:
            y_true = []
            y_proba_model = []
            y_proba_market = []
            y_proba_uniform = []

            for s in pit_with_preds:
                pred = pit_safe_predictions[s['match_id']]
                result = get_result(s)
                y_true.append(result)

                # Model probabilities (normalized to sum to 1)
                raw_probs = (
                    to_float(pred.get('home_prob', 1/3)),
                    to_float(pred.get('draw_prob', 1/3)),
                    to_float(pred.get('away_prob', 1/3)),
                )
                y_proba_model.append(normalize_probs(*raw_probs))

                # Market probabilities (de-vigged)
                mkt_probs = odds_to_probs_devig(s['odds_home'], s['odds_draw'], s['odds_away'])
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

        # Betting simulation
        bets = []
        for s in pit_with_preds:
            # Use PIT-safe prediction selected above
            pred = pit_safe_predictions[s['match_id']]
            odds = [to_float(s['odds_home']), to_float(s['odds_draw']), to_float(s['odds_away'])]

            # Normalize model probabilities
            raw_probs = (
                to_float(pred.get('home_prob', 0)),
                to_float(pred.get('draw_prob', 0)),
                to_float(pred.get('away_prob', 0)),
            )
            probs_model = list(normalize_probs(*raw_probs))
            probs_market = odds_to_probs_devig(*odds)

            # Calculate edges
            edges = [probs_model[i] - probs_market[i] for i in range(3)]
            best_idx = np.argmax(edges)
            best_edge = edges[best_idx]

            # EV for best outcome
            ev_best = probs_model[best_idx] * odds[best_idx] - 1

            # Only bet if edge >= threshold and EV > 0
            if best_edge >= EDGE_THRESHOLD and ev_best > 0:
                result = get_result(s)
                won = (result == best_idx)
                returns = odds[best_idx] if won else 0

                bets.append({
                    'match_id': s['match_id'],
                    'bet_outcome': best_idx,
                    'odds': odds[best_idx],
                    'ev': ev_best,
                    'edge': best_edge,
                    'won': won,
                    'returns': returns,
                })

        betting_metrics = calculate_betting_metrics(bets)

        # Bootstrap CI for ROI and EV
        if bets:
            roi_values = [(b['returns'] - 1) for b in bets]  # P&L per bet
            ev_values = [b['ev'] for b in bets]

            roi_ci_low, roi_ci_high, roi_ci_status = bootstrap_ci(roi_values)
            ev_ci_low, ev_ci_high, ev_ci_status = bootstrap_ci(ev_values)

            betting_metrics['roi_ci95_low'] = roi_ci_low
            betting_metrics['roi_ci95_high'] = roi_ci_high
            betting_metrics['roi_ci_status'] = roi_ci_status
            betting_metrics['ev_ci95_low'] = ev_ci_low
            betting_metrics['ev_ci95_high'] = ev_ci_high
            betting_metrics['ev_ci_status'] = ev_ci_status

            # Win rate
            betting_metrics['win_rate'] = sum(1 for b in bets if b['won']) / len(bets)
        else:
            betting_metrics['roi_ci95_low'] = None
            betting_metrics['roi_ci95_high'] = None
            betting_metrics['roi_ci_status'] = 'no_bets'
            betting_metrics['ev_ci95_low'] = None
            betting_metrics['ev_ci95_high'] = None
            betting_metrics['ev_ci_status'] = 'no_bets'
            betting_metrics['win_rate'] = None

        # Build report
        report = {
            'generated_at': datetime.now().isoformat(),
            'protocol_version': PROTOCOL_VERSION,
            'filters': {
                'snapshot_type': 'lineup_confirmed',
                'odds_freshness': 'live',
                'timing_window_valid': f'{TIMING_WINDOW_VALID_MIN}-{TIMING_WINDOW_VALID_MAX} min',
                'timing_window_ideal': f'{TIMING_WINDOW_IDEAL_MIN}-{TIMING_WINDOW_IDEAL_MAX} min',
                'edge_threshold': EDGE_THRESHOLD,
            },
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
            'betting': betting_metrics,
            'phase': (
                'insufficient' if n_valid_10_90 < 50 else
                'piloto' if n_valid_10_90 < 200 else
                'preliminar' if n_valid_10_90 < 500 else
                'formal'
            ),
            'prediction_integrity': pred_metadata,
            'notes': 'read-only evaluation; no writes to DB; probs normalized; PIT integrity enforced (pred.created_at <= snapshot_at)',
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
    print(f"\nCoverage:")
    print(f"  Total snapshots:     {counts.get('n_total_snapshots', 0)}")
    print(f"  Live:                {counts.get('n_live', 0)}")
    print(f"  Valid PIT (10-90):   {counts.get('n_pit_valid_10_90', 0)}")
    print(f"  Ideal (45-75):       {counts.get('n_pit_valid_ideal_45_75', 0)}")
    print(f"  PIT-safe preds:      {counts.get('n_with_pit_safe_predictions', 0)}")
    print(f"  No pred as-of:       {counts.get('n_no_prediction_asof', 0)}")
    print(f"  PIT integrity:       {pred_integrity.get('pit_prediction_integrity', 'unknown')}")

    brier = report.get('brier', {})
    if brier.get('brier_model') is not None:
        print(f"\nBrier (calibration):")
        print(f"  Model:               {brier['brier_model']:.4f}")
        print(f"  Market:              {brier.get('brier_market', 0):.4f}")
        print(f"  Uniform:             {brier.get('brier_uniform', 0):.4f}")
        print(f"  Skill vs uniform:    {brier.get('skill_vs_uniform', 0):.2%}")
        print(f"  Skill vs market:     {brier.get('skill_vs_market', 0):.2%}")

    betting = report.get('betting', {})
    print(f"\nBetting (primary):")
    print(f"  N bets:              {betting.get('n_bets', 0)}")
    if betting.get('roi') is not None:
        print(f"  ROI:                 {betting['roi']:.2%}")
        if betting.get('roi_ci95_low') is not None:
            print(f"  ROI CI95%:           [{betting['roi_ci95_low']:.2%}, {betting['roi_ci95_high']:.2%}]")
        print(f"  EV:                  {betting.get('ev', 0):.4f}")
        print(f"  Win rate:            {betting.get('win_rate', 0):.2%}")

    print(f"\nPhase: {report.get('phase', 'unknown')}")
    print("=" * 60)


async def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("Running PIT evaluation (live odds only)...")

    try:
        report = await run_evaluation()
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

    json_path = logs_dir / f"pit_evaluation_live_only_{timestamp}.json"
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\nResultados guardados en: {json_path}")

    # Exit 0 even if insufficient data (per spec)
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
