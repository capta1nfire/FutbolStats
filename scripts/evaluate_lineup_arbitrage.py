#!/usr/bin/env python3
"""
Evaluate Lineup Arbitrage model against REAL lineup_confirmed odds snapshots.

This script evaluates the hypothesis:
    "Can we beat the market odds AT THE MOMENT lineups are announced (~T-60min)?"

Requirements:
- At least N >= 200 lineup_confirmed snapshots (ideally 500+)
- Snapshots should have odds_freshness='live' (from production system)

Outputs:
- Delta Brier vs lineup_confirmed baseline
- 95% CI via bootstrap
- Gate coverage and effectiveness
- Decision: CONTINUE or CLOSE the project

Usage:
    DATABASE_URL="..." python scripts/evaluate_lineup_arbitrage.py --min-snapshots 200
"""

import argparse
import asyncio
import logging
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def calculate_brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Calculate Brier score for multiclass (H/D/A)."""
    return np.mean(np.sum((y_true - y_prob) ** 2, axis=1))


def bootstrap_ci(metric_func, y_true, y_pred_model, y_pred_market, n_bootstrap=1000, ci=0.95):
    """Bootstrap confidence interval for delta = model_brier - market_brier."""
    n = len(y_true)
    deltas = []

    for _ in range(n_bootstrap):
        idx = np.random.choice(n, size=n, replace=True)
        model_brier = metric_func(y_true[idx], y_pred_model[idx])
        market_brier = metric_func(y_true[idx], y_pred_market[idx])
        deltas.append(model_brier - market_brier)

    deltas = np.array(deltas)
    alpha = (1 - ci) / 2
    lower = np.percentile(deltas, alpha * 100)
    upper = np.percentile(deltas, (1 - alpha) * 100)

    return np.mean(deltas), lower, upper


async def load_lineup_confirmed_data(session, min_snapshots: int, freshness_filter: str = None):
    """Load matches with lineup_confirmed snapshots for evaluation."""

    # Build freshness filter
    freshness_clause = ""
    if freshness_filter:
        freshness_clause = f"AND os.odds_freshness = '{freshness_filter}'"

    query = f"""
        SELECT
            m.id as match_id,
            m.date,
            m.home_goals,
            m.away_goals,

            -- Lineup confirmed odds (our TRUE baseline)
            os.prob_home as lineup_prob_home,
            os.prob_draw as lineup_prob_draw,
            os.prob_away as lineup_prob_away,
            os.delta_to_kickoff_seconds,
            os.odds_freshness,

            -- Count starters to estimate rotation (if available)
            COALESCE(array_length(ml_home.starting_xi_ids, 1), 0) as home_xi_count,
            COALESCE(array_length(ml_away.starting_xi_ids, 1), 0) as away_xi_count

        FROM matches m
        JOIN odds_snapshots os ON m.id = os.match_id
            AND os.snapshot_type = 'lineup_confirmed'
            {freshness_clause}
        LEFT JOIN match_lineups ml_home ON m.id = ml_home.match_id
            AND ml_home.is_home = true
        LEFT JOIN match_lineups ml_away ON m.id = ml_away.match_id
            AND ml_away.is_home = false
        WHERE m.status = 'FT'
          AND m.home_goals IS NOT NULL
          AND os.prob_home IS NOT NULL
        ORDER BY m.date DESC
    """

    result = await session.execute(text(query))
    rows = result.fetchall()

    if len(rows) < min_snapshots:
        return None, f"Insufficient data: {len(rows)} < {min_snapshots} required"

    df = pd.DataFrame(rows, columns=[
        'match_id', 'date', 'home_goals', 'away_goals',
        'lineup_prob_home', 'lineup_prob_draw', 'lineup_prob_away',
        'delta_to_kickoff_seconds', 'odds_freshness',
        'home_xi_count', 'away_xi_count'
    ])

    return df, None


def create_outcome_matrix(df: pd.DataFrame) -> np.ndarray:
    """Create one-hot encoded outcome matrix [N x 3] for H/D/A."""
    y = np.zeros((len(df), 3))

    for i, row in df.iterrows():
        if row['home_goals'] > row['away_goals']:
            y[i, 0] = 1  # Home win
        elif row['home_goals'] < row['away_goals']:
            y[i, 2] = 1  # Away win
        else:
            y[i, 1] = 1  # Draw

    return y


def simple_lineup_adjustment(df: pd.DataFrame) -> np.ndarray:
    """
    Simple lineup-based probability adjustment.

    NOTE: Without jaccard/rotation features, we currently just return market probs.
    In production, this would use lineup deviation metrics.
    """
    # For now, just return the market probs as "model" probs
    # This tests if we can even beat market with NO adjustment
    # (baseline sanity check)
    probs = df[['lineup_prob_home', 'lineup_prob_draw', 'lineup_prob_away']].values.copy()

    # TODO: When jaccard/rotation features are available:
    # - Detect rotation (jaccard < 0.7 or rotation > 3)
    # - Shift probability toward opponent

    return probs


async def main(args):
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        raise ValueError("DATABASE_URL required")

    if database_url.startswith("postgresql://"):
        database_url = database_url.replace("postgresql://", "postgresql+asyncpg://", 1)

    engine = create_async_engine(database_url)
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async with async_session() as session:
        logger.info(f"Loading lineup_confirmed data (min={args.min_snapshots})...")

        # Try live first, fallback to all
        df, error = await load_lineup_confirmed_data(
            session, args.min_snapshots,
            freshness_filter='live' if args.live_only else None
        )

        if error:
            logger.warning(f"Live-only failed: {error}")
            if args.live_only:
                logger.error("--live-only specified but insufficient live data. Exiting.")
                return

            logger.info("Trying with all freshness levels...")
            df, error = await load_lineup_confirmed_data(session, args.min_snapshots)

            if error:
                logger.error(error)
                logger.error("\n❌ INSUFFICIENT DATA - Cannot evaluate yet.")
                logger.info("   Run the system for 2-4 weeks to accumulate lineup_confirmed snapshots.")
                return

        logger.info(f"Loaded {len(df)} matches with lineup_confirmed snapshots")

        # Data quality report
        logger.info("\n" + "="*60)
        logger.info("DATA QUALITY REPORT")
        logger.info("="*60)

        freshness_dist = df['odds_freshness'].value_counts()
        logger.info(f"Odds Freshness Distribution:")
        for f, cnt in freshness_dist.items():
            pct = cnt / len(df) * 100
            logger.info(f"  {f or 'unknown'}: {cnt} ({pct:.1f}%)")

        delta_stats = df['delta_to_kickoff_seconds'].dropna() / 60
        if len(delta_stats) > 0:
            logger.info(f"\nTiming (minutes before kickoff):")
            logger.info(f"  p10: {delta_stats.quantile(0.1):.1f}")
            logger.info(f"  p50: {delta_stats.quantile(0.5):.1f}")
            logger.info(f"  p90: {delta_stats.quantile(0.9):.1f}")
            logger.info(f"  mean: {delta_stats.mean():.1f}")

        # Check lineup data availability
        df_with_lineups = df[(df['home_xi_count'] > 0) & (df['away_xi_count'] > 0)]
        logger.info(f"\nMatches with lineup data: {len(df_with_lineups)} ({len(df_with_lineups)/len(df)*100:.1f}%)")

        # Create outcome matrix
        y_true = create_outcome_matrix(df)

        # Market baseline (lineup_confirmed odds) - convert Decimal to float
        market_probs = df[['lineup_prob_home', 'lineup_prob_draw', 'lineup_prob_away']].astype(float).values

        # Model prediction (simple adjustment)
        model_probs = simple_lineup_adjustment(df).astype(float)

        # Calculate metrics
        logger.info("\n" + "="*60)
        logger.info("EVALUATION RESULTS")
        logger.info("="*60)

        market_brier = calculate_brier_score(y_true, market_probs)
        model_brier = calculate_brier_score(y_true, model_probs)
        delta = model_brier - market_brier

        logger.info(f"\nBrier Scores (lower is better):")
        logger.info(f"  Market (lineup_confirmed): {market_brier:.5f}")
        logger.info(f"  Model (lineup-adjusted):   {model_brier:.5f}")
        logger.info(f"  Delta:                     {delta:+.5f}")

        # Bootstrap CI
        logger.info(f"\nBootstrapping 95% CI ({args.bootstrap_n} iterations)...")
        mean_delta, ci_lower, ci_upper = bootstrap_ci(
            calculate_brier_score, y_true, model_probs, market_probs,
            n_bootstrap=args.bootstrap_n
        )
        logger.info(f"  Delta mean: {mean_delta:+.5f}")
        logger.info(f"  95% CI: [{ci_lower:+.5f}, {ci_upper:+.5f}]")

        # Gate analysis (placeholder - needs rotation/jaccard features)
        logger.info("\n" + "="*60)
        logger.info("GATE ANALYSIS")
        logger.info("="*60)
        logger.info("⚠️ Gate analysis requires jaccard/rotation features in match_lineups.")
        logger.info("   These features will be computed when production lineup monitoring starts.")
        logger.info("   For now, we evaluate the FULL dataset (no gating).")

        # Bias check: Compare live vs stale samples
        logger.info("\n" + "="*60)
        logger.info("FRESHNESS BIAS CHECK")
        logger.info("="*60)

        live_mask = df['odds_freshness'] == 'live'
        stale_mask = df['odds_freshness'].isin(['stale', 'unknown'])

        live_count = live_mask.sum()
        stale_count = stale_mask.sum()

        if live_count >= 30 and stale_count >= 30:
            # Compare Brier scores between live and stale
            live_brier = calculate_brier_score(y_true[live_mask], market_probs[live_mask])
            stale_brier = calculate_brier_score(y_true[stale_mask], market_probs[stale_mask])

            logger.info(f"Live samples: {live_count}, Stale samples: {stale_count}")
            logger.info(f"Market Brier (live):  {live_brier:.5f}")
            logger.info(f"Market Brier (stale): {stale_brier:.5f}")
            logger.info(f"Difference: {stale_brier - live_brier:+.5f}")

            if abs(stale_brier - live_brier) > 0.01:
                logger.warning("⚠️ SIGNIFICANT BIAS: Stale and live samples have different baseline quality!")
                logger.warning("   → Recommend using --live-only for evaluation when sufficient live data exists.")
            else:
                logger.info("✅ No significant bias detected between live and stale samples.")
        elif live_count < 30:
            logger.info(f"⚠️ Insufficient live samples ({live_count}) for bias check. Need ≥30.")
            logger.info("   → Run production system for more data before using --live-only.")
        else:
            logger.info(f"✅ All samples are live ({live_count}). No bias check needed.")

        # Decision
        logger.info("\n" + "="*60)
        logger.info("DECISION")
        logger.info("="*60)

        if ci_upper < 0:
            logger.info("✅ ALPHA CONFIRMED: 95% CI entirely negative")
            logger.info("   → CONTINUE the Lineup Arbitrage project")
        elif ci_lower > 0:
            logger.info("❌ NO ALPHA: 95% CI entirely positive (worse than market)")
            logger.info("   → CLOSE the Lineup Arbitrage project")
        else:
            logger.info("⚠️ INCONCLUSIVE: 95% CI crosses zero")
            logger.info(f"   → Need more data. Current N={len(df)}, recommend N >= 500")
            if freshness_dist.get('live', 0) < len(df) * 0.5:
                logger.info("   → Most snapshots are not 'live' - quality concern")

    await engine.dispose()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Lineup Arbitrage model")
    parser.add_argument("--min-snapshots", type=int, default=200, help="Minimum snapshots required")
    parser.add_argument("--live-only", action="store_true", help="Only use live odds (not stale)")
    parser.add_argument("--bootstrap-n", type=int, default=1000, help="Bootstrap iterations")
    args = parser.parse_args()

    asyncio.run(main(args))
