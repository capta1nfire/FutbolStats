#!/usr/bin/env python3
"""
Generate experimental predictions for TITAN tier comparison.

For each snapshot in the evaluation window, generates a prediction using
the specified tier's model and inserts into predictions_experiments.

Usage:
    python scripts/generate_tier_preds.py --tier baseline --since 2026-01-07
    python scripts/generate_tier_preds.py --tier T1b --since 2026-01-07

Prerequisites:
    1. Run migrations/titan_009_predictions_experiments.sql
    2. Train models with train_titan_tier.py

Output:
    Inserts into predictions_experiments table with PIT-safe created_at
"""

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import xgboost as xgb
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

from app.ml.engine import XGBoostEngine
from app.features.engineering import FeatureEngineer
from app.models import Match

# Feature definitions
BASELINE_FEATURES = XGBoostEngine.FEATURE_COLUMNS

# PRODUCCIÓN v1.0.0: Exactamente 14 features (sin los 3 de FASE 1)
BASELINE_V1_FEATURES = [
    "home_goals_scored_avg",
    "home_goals_conceded_avg",
    "home_shots_avg",
    "home_corners_avg",
    "home_rest_days",
    "home_matches_played",
    "away_goals_scored_avg",
    "away_goals_conceded_avg",
    "away_shots_avg",
    "away_corners_avg",
    "away_rest_days",
    "away_matches_played",
    "goal_diff_avg",
    "rest_diff",
]

# BUG #3 FIX: Added missing flags to distinguish real 0 vs imputed 0
TITAN_T1B_FEATURES = [
    "xg_home_last5", "xg_away_last5", "xga_home_last5", "xga_away_last5",
    "npxg_home_last5", "npxg_away_last5",
    "xg_home_missing", "xg_away_missing"  # 1 if <5 matches available
]
# T1b_v2: Differential features
TITAN_T1B_V2_FEATURES = [
    "xg_diff_last5",      # xg_home_last5 - xg_away_last5
    "xga_diff_last5",     # xga_home_last5 - xga_away_last5
    "npxg_diff_last5",    # npxg_home_last5 - npxg_away_last5
    "xg_net_home_last5",  # xg_home_last5 - xga_home_last5
    "xg_net_away_last5",  # xg_away_last5 - xga_away_last5
    "net_diff_last5",     # xg_net_home_last5 - xg_net_away_last5
    "xg_home_missing", "xg_away_missing"  # 1 if <5 matches available
]
TITAN_T1C_FEATURES = [
    "sofascore_lineup_integrity_score",
    "lineup_home_starters_count", "lineup_away_starters_count"
]
TITAN_T1D_FEATURES = [
    "xi_home_def_count", "xi_home_mid_count", "xi_home_fwd_count",
    "xi_away_def_count", "xi_away_mid_count", "xi_away_fwd_count",
    "xi_formation_mismatch_flag"
]

TIER_FEATURES = {
    "baseline": list(BASELINE_FEATURES),
    "T1b": list(BASELINE_FEATURES) + TITAN_T1B_FEATURES,
    "T1b_v2": list(BASELINE_FEATURES) + TITAN_T1B_V2_FEATURES,
    "T1c": list(BASELINE_FEATURES) + TITAN_T1C_FEATURES,
    "T1d": list(BASELINE_FEATURES) + TITAN_T1D_FEATURES,
    # v1 tiers: usan exactamente los 14 features de producción v1.0.0
    "baseline_v1": list(BASELINE_V1_FEATURES),
    "T1b_v2_on_v1": list(BASELINE_V1_FEATURES) + TITAN_T1B_V2_FEATURES,
}


async def get_snapshots(session: AsyncSession, since: datetime) -> list[dict]:
    """Get all lineup_confirmed snapshots since the specified date."""
    query = text("""
        SELECT os.id as snapshot_id, os.match_id, os.snapshot_at
        FROM odds_snapshots os
        WHERE os.snapshot_type = 'lineup_confirmed'
          AND os.snapshot_at >= :since
        ORDER BY os.snapshot_at
    """)
    result = await session.execute(query, {"since": since})
    return [dict(r._mapping) for r in result.fetchall()]


async def get_match(session: AsyncSession, match_id: int) -> Optional[Match]:
    """Get match by ID."""
    from sqlalchemy import select
    result = await session.execute(
        select(Match).where(Match.id == match_id)
    )
    return result.scalar_one_or_none()


async def build_features_asof(
    session: AsyncSession,
    engineer: FeatureEngineer,
    match: Match,
    snapshot_at: datetime,
    tier: str
) -> list[float]:
    """
    Build feature vector as-of snapshot time.

    PIT-STRICT: Uses snapshot_at (not match.date) for temporal calculations.
    This ensures rest_days and other features are computed as-of the snapshot,
    preventing information leakage from the time between snapshot and kickoff.
    """
    # 1. Get baseline features using PIT-strict method
    # CRITICAL: Use get_match_features_asof to calculate features as-of snapshot_at
    try:
        all_features = await engineer.get_match_features_asof(match, snapshot_at)
    except Exception as e:
        print(f"Warning: Failed to get features for match {match.id}: {e}")
        all_features = {}

    # Extract only the features for this tier (in order)
    feature_cols = TIER_FEATURES[tier]
    feature_vector = []

    # Determine which baseline features to use (14 for v1 tiers, 17 for others)
    if tier in ["baseline_v1", "T1b_v2_on_v1"]:
        baseline_cols = BASELINE_V1_FEATURES
    else:
        baseline_cols = BASELINE_FEATURES

    # Baseline features
    for col in baseline_cols:
        feature_vector.append(float(all_features.get(col, 0.0)))

    # 2. For TITAN tiers, add extra features
    if tier in ["T1b", "T1b_v2", "T1c", "T1d", "T1b_v2_on_v1"]:

        if tier in ["T1b", "T1b_v2", "T1b_v2_on_v1"]:
            # BUG FIXES (ATI):
            #   - #2: Use direct AVG of last 5 matches (no window function ambiguity)
            #   - #3: Add missing flags
            xg_query = text("""
                WITH team_xg_history AS (
                    SELECT m.home_team_id as team_id, m.id as match_id, m.date,
                           mut.xg_home as xg_for, mut.xg_away as xg_against,
                           mut.npxg_home as npxg_for
                    FROM matches m
                    JOIN match_understat_team mut ON mut.match_id = m.id
                    WHERE mut.xg_home IS NOT NULL
                    UNION ALL
                    SELECT m.away_team_id, m.id, m.date,
                           mut.xg_away, mut.xg_home, mut.npxg_away
                    FROM matches m
                    JOIN match_understat_team mut ON mut.match_id = m.id
                    WHERE mut.xg_away IS NOT NULL
                )
                SELECT
                    -- Home team rolling
                    home_roll.xg_last5 as xg_home_last5,
                    home_roll.xga_last5 as xga_home_last5,
                    home_roll.npxg_last5 as npxg_home_last5,
                    CASE WHEN home_roll.match_count < 5 THEN 1 ELSE 0 END as xg_home_missing,
                    -- Away team rolling
                    away_roll.xg_last5 as xg_away_last5,
                    away_roll.xga_last5 as xga_away_last5,
                    away_roll.npxg_last5 as npxg_away_last5,
                    CASE WHEN away_roll.match_count < 5 THEN 1 ELSE 0 END as xg_away_missing
                FROM matches m
                -- Home team: AVG of last 5 matches STRICTLY BEFORE snapshot_at
                LEFT JOIN LATERAL (
                    SELECT
                        AVG(xg_for) as xg_last5,
                        AVG(xg_against) as xga_last5,
                        AVG(npxg_for) as npxg_last5,
                        COUNT(*) as match_count
                    FROM (
                        SELECT xg_for, xg_against, npxg_for
                        FROM team_xg_history
                        WHERE team_id = m.home_team_id
                          AND date < :snapshot_at  -- STRICTLY before snapshot
                        ORDER BY date DESC, match_id DESC
                        LIMIT 5
                    ) last5
                ) home_roll ON TRUE
                -- Away team: AVG of last 5 matches STRICTLY BEFORE snapshot_at
                LEFT JOIN LATERAL (
                    SELECT
                        AVG(xg_for) as xg_last5,
                        AVG(xg_against) as xga_last5,
                        AVG(npxg_for) as npxg_last5,
                        COUNT(*) as match_count
                    FROM (
                        SELECT xg_for, xg_against, npxg_for
                        FROM team_xg_history
                        WHERE team_id = m.away_team_id
                          AND date < :snapshot_at
                        ORDER BY date DESC, match_id DESC
                        LIMIT 5
                    ) last5
                ) away_roll ON TRUE
                WHERE m.id = :match_id
            """)
            xg_result = await session.execute(
                xg_query,
                {"match_id": match.id, "snapshot_at": snapshot_at}
            )
            xg_row = xg_result.fetchone()

            if xg_row:
                row = dict(xg_row._mapping)
                xg_home = float(row.get('xg_home_last5') or 0.0)
                xg_away = float(row.get('xg_away_last5') or 0.0)
                xga_home = float(row.get('xga_home_last5') or 0.0)
                xga_away = float(row.get('xga_away_last5') or 0.0)
                npxg_home = float(row.get('npxg_home_last5') or 0.0)
                npxg_away = float(row.get('npxg_away_last5') or 0.0)
                xg_home_missing = int(row.get('xg_home_missing') or 1)
                xg_away_missing = int(row.get('xg_away_missing') or 1)

                if tier == "T1b":
                    # T1b: Raw xG features + missing flags
                    feature_vector.extend([xg_home, xg_away, xga_home, xga_away, npxg_home, npxg_away,
                                           xg_home_missing, xg_away_missing])
                else:
                    # T1b_v2: Differential features + missing flags
                    xg_diff = xg_home - xg_away
                    xga_diff = xga_home - xga_away
                    npxg_diff = npxg_home - npxg_away
                    xg_net_home = xg_home - xga_home
                    xg_net_away = xg_away - xga_away
                    net_diff = xg_net_home - xg_net_away
                    feature_vector.extend([xg_diff, xga_diff, npxg_diff, xg_net_home, xg_net_away, net_diff,
                                           xg_home_missing, xg_away_missing])
            else:
                # No xG data available - all zeros with missing=1
                if tier == "T1b":
                    feature_vector.extend([0.0] * 6 + [1, 1])  # 6 xG features + 2 missing flags
                else:
                    feature_vector.extend([0.0] * 6 + [1, 1])  # 6 diff features + 2 missing flags

        elif tier in ["T1c", "T1d"]:
            # T1c/T1d: Use titan.feature_matrix
            titan_query = text("""
                SELECT *
                FROM titan.feature_matrix fm
                WHERE fm.match_id = :match_id
                  AND fm.pit_max_captured_at <= :snapshot_at
                ORDER BY fm.pit_max_captured_at DESC
                LIMIT 1
            """)
            titan_result = await session.execute(
                titan_query,
                {"match_id": match.id, "snapshot_at": snapshot_at}
            )
            titan_row = titan_result.fetchone()

            if tier == "T1c":
                if titan_row:
                    row = dict(titan_row._mapping)
                    feature_vector.extend([
                        float(row.get('sofascore_lineup_integrity_score') or 0.0),
                        float(row.get('lineup_home_starters_count') or 0),
                        float(row.get('lineup_away_starters_count') or 0),
                    ])
                else:
                    feature_vector.extend([0.0] * len(TITAN_T1C_FEATURES))

            elif tier == "T1d":
                if titan_row:
                    row = dict(titan_row._mapping)
                    feature_vector.extend([
                        float(row.get('xi_home_def_count') or 0),
                        float(row.get('xi_home_mid_count') or 0),
                        float(row.get('xi_home_fwd_count') or 0),
                        float(row.get('xi_away_def_count') or 0),
                        float(row.get('xi_away_mid_count') or 0),
                        float(row.get('xi_away_fwd_count') or 0),
                        float(row.get('xi_formation_mismatch_flag') or 0),
                    ])
                else:
                    feature_vector.extend([0.0] * len(TITAN_T1D_FEATURES))

    return feature_vector


async def generate_predictions(tier: str, since: str):
    """Generate predictions for all snapshots since the specified date."""

    database_url = os.environ.get('DATABASE_URL')
    if not database_url:
        raise ValueError("DATABASE_URL environment variable required")

    if database_url.startswith('postgresql://'):
        database_url = database_url.replace('postgresql://', 'postgresql+asyncpg://')

    engine = create_async_engine(database_url)
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    # Load trained model
    model_path = Path(f"models/xgb_{tier}_latest.json")
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}. Run train_titan_tier.py first.")

    model = xgb.XGBClassifier()
    model.load_model(str(model_path))
    print(f"Loaded model: {model_path}")

    since_date = datetime.fromisoformat(since)
    model_version = f"v1.0.0+{tier}" if tier != "baseline" else "v1.0.0-exp"
    feature_list = TIER_FEATURES[tier]

    async with async_session() as session:
        # Get all snapshots
        snapshots = await get_snapshots(session, since_date)
        print(f"Found {len(snapshots)} snapshots since {since}")

        # Create FeatureEngineer
        engineer = FeatureEngineer(session)

        inserted = 0
        skipped = 0
        errors = 0

        for i, snap in enumerate(snapshots):
            if i % 50 == 0:
                print(f"Processing snapshot {i}/{len(snapshots)}...")

            try:
                # Get match
                match = await get_match(session, snap['match_id'])
                if not match:
                    print(f"Warning: Match {snap['match_id']} not found")
                    errors += 1
                    continue

                # Build features as-of snapshot_at
                features = await build_features_asof(
                    session, engineer, match, snap['snapshot_at'], tier
                )

                # Predict
                X = np.array([features])
                probs = model.predict_proba(X)[0]  # [home, draw, away]

                # Insert with PIT-safe created_at
                created_at = snap['snapshot_at'] - timedelta(seconds=1)

                insert_query = text("""
                    INSERT INTO predictions_experiments
                    (snapshot_id, match_id, snapshot_at, model_version,
                     home_prob, draw_prob, away_prob, feature_set, created_at)
                    VALUES (:snapshot_id, :match_id, :snapshot_at, :model_version,
                            :home_prob, :draw_prob, :away_prob, :feature_set, :created_at)
                    ON CONFLICT (snapshot_id, model_version) DO NOTHING
                """)

                await session.execute(insert_query, {
                    'snapshot_id': snap['snapshot_id'],
                    'match_id': snap['match_id'],
                    'snapshot_at': snap['snapshot_at'],
                    'model_version': model_version,
                    'home_prob': float(probs[0]),
                    'draw_prob': float(probs[1]),
                    'away_prob': float(probs[2]),
                    'feature_set': json.dumps(feature_list),
                    'created_at': created_at,
                })

                inserted += 1

            except Exception as e:
                print(f"Error processing snapshot {snap['snapshot_id']}: {e}")
                errors += 1
                continue

        await session.commit()

    await engine.dispose()

    print(f"\nGeneration complete:")
    print(f"  Tier: {tier}")
    print(f"  Model version: {model_version}")
    print(f"  Inserted: {inserted}")
    print(f"  Skipped: {skipped}")
    print(f"  Errors: {errors}")


def main():
    parser = argparse.ArgumentParser(description="Generate TITAN tier predictions")
    parser.add_argument('--tier', required=True, choices=list(TIER_FEATURES.keys()),
                        help='Tier to generate: baseline, T1b, T1c, T1d')
    parser.add_argument('--since', required=True,
                        help='Start date for evaluation window (ISO format, e.g., 2026-01-07)')
    args = parser.parse_args()

    print(f"Generating predictions for tier={args.tier}, since={args.since}")
    print("=" * 60)

    asyncio.run(generate_predictions(args.tier, args.since))


if __name__ == "__main__":
    main()
