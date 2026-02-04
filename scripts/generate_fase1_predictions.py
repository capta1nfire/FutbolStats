#!/usr/bin/env python3
"""
Generate Fase 1 experimental predictions using OPTIMIZED batch SQL.

Creates predictions for the new v1.0.1-league-only model
trained with league_only=True features (NO SKEW).

OPTIMIZATION: Uses batch SQL for feature calculation, avoiding N+1 queries.

Usage:
    python scripts/generate_fase1_predictions.py --since 2026-01-08

Output:
    Inserts into predictions_experiments with PIT-safe created_at = snapshot_at - 1s
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from datetime import datetime, timedelta
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import glob
import numpy as np
import xgboost as xgb
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Model configuration
MODEL_VERSION = "v1.0.1-league-only-trained"
MODEL_PATH_PATTERN = "models/xgb_v1.0.1-league-only_*.json"

# The 14 features expected by the model
FEATURE_COLUMNS = [
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

ROLLING_WINDOW = 10


async def generate_predictions(since: str, until: Optional[str] = None):
    """Generate predictions for all lineup_confirmed snapshots since date."""

    database_url = os.environ.get('DATABASE_URL')
    if not database_url:
        raise ValueError("DATABASE_URL required")

    if database_url.startswith('postgresql://'):
        database_url = database_url.replace('postgresql://', 'postgresql+asyncpg://')

    engine = create_async_engine(database_url)
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    # Load model
    matching_models = sorted(glob.glob(MODEL_PATH_PATTERN))
    if not matching_models:
        raise RuntimeError(f"No model found matching: {MODEL_PATH_PATTERN}")
    model_path = matching_models[-1]  # Latest
    logger.info(f"Loading model: {model_path}")

    model = xgb.XGBClassifier()
    model.load_model(model_path)
    logger.info(f"Model loaded: n_features={model.n_features_in_}")

    since_date = datetime.fromisoformat(since)
    until_date = datetime.fromisoformat(until) if until else None

    async with async_session() as session:
        # Step 1: Get all snapshots to process
        snapshots_query = text("""
            SELECT os.id as snapshot_id, os.match_id, os.snapshot_at,
                   m.home_team_id, m.away_team_id, m.date as match_date
            FROM odds_snapshots os
            JOIN matches m ON os.match_id = m.id
            WHERE os.snapshot_type = 'lineup_confirmed'
              AND os.snapshot_at >= :since
        """ + (" AND os.snapshot_at < :until" if until_date else "") + """
            ORDER BY os.snapshot_at
        """)

        params = {"since": since_date}
        if until_date:
            params["until"] = until_date

        result = await session.execute(snapshots_query, params)
        snapshots = [dict(r._mapping) for r in result.fetchall()]
        logger.info(f"Found {len(snapshots)} snapshots to process")

        if not snapshots:
            return {"n_inserted": 0}

        # Step 2: Get all league matches for feature calculation (batch)
        all_team_ids = list(set(
            [s['home_team_id'] for s in snapshots] +
            [s['away_team_id'] for s in snapshots]
        ))

        # Find earliest date we need data for
        min_snapshot_date = min(s['snapshot_at'] for s in snapshots)
        earliest_needed = min_snapshot_date - timedelta(days=365)

        league_matches_query = text("""
            SELECT
                m.id as match_id,
                m.date,
                m.home_team_id,
                m.away_team_id,
                m.home_goals,
                m.away_goals,
                COALESCE((m.stats->'home'->>'total_shots')::int, 0) as home_shots,
                COALESCE((m.stats->'away'->>'total_shots')::int, 0) as away_shots,
                COALESCE((m.stats->'home'->>'corner_kicks')::int, 0) as home_corners,
                COALESCE((m.stats->'away'->>'corner_kicks')::int, 0) as away_corners
            FROM matches m
            JOIN admin_leagues al ON m.league_id = al.league_id
            WHERE m.status = 'FT'
              AND m.home_goals IS NOT NULL
              AND m.away_goals IS NOT NULL
              AND (m.tainted IS NULL OR m.tainted = false)
              AND al.kind = 'league'
              AND m.date >= :earliest
              AND (m.home_team_id = ANY(:team_ids) OR m.away_team_id = ANY(:team_ids))
            ORDER BY m.date
        """)

        result = await session.execute(league_matches_query, {
            "earliest": earliest_needed,
            "team_ids": all_team_ids
        })
        all_league_matches = [dict(r._mapping) for r in result.fetchall()]
        logger.info(f"Fetched {len(all_league_matches)} league matches for features")

        # Step 3: Build team match index
        team_matches = {}
        for m in all_league_matches:
            for team_id in [m['home_team_id'], m['away_team_id']]:
                if team_id not in team_matches:
                    team_matches[team_id] = []
                team_matches[team_id].append((m['date'], m))

        for team_id in team_matches:
            team_matches[team_id].sort(key=lambda x: x[0], reverse=True)

        logger.info(f"Built match index for {len(team_matches)} teams")

        # Step 4: Generate predictions for each snapshot
        n_inserted = 0
        n_errors = 0

        for i, snap in enumerate(snapshots):
            if (i + 1) % 100 == 0:
                logger.info(f"Processing snapshot {i + 1}/{len(snapshots)}")

            try:
                # Calculate features as-of snapshot_at
                features = calculate_features_asof(
                    snap['home_team_id'],
                    snap['away_team_id'],
                    snap['snapshot_at'],
                    team_matches
                )

                # Build feature vector
                feature_vector = [float(features.get(col, 0.0)) for col in FEATURE_COLUMNS]
                X = np.array([feature_vector])

                # Predict
                probs = model.predict_proba(X)[0]

                # Insert with PIT-safe created_at
                created_at = snap['snapshot_at'] - timedelta(seconds=1)

                await session.execute(text("""
                    INSERT INTO predictions_experiments
                    (snapshot_id, match_id, snapshot_at, model_version,
                     home_prob, draw_prob, away_prob, feature_set, created_at)
                    VALUES (:snapshot_id, :match_id, :snapshot_at, :model_version,
                            :home_prob, :draw_prob, :away_prob, :feature_set, :created_at)
                    ON CONFLICT (snapshot_id, model_version) DO UPDATE SET
                        home_prob = EXCLUDED.home_prob,
                        draw_prob = EXCLUDED.draw_prob,
                        away_prob = EXCLUDED.away_prob,
                        feature_set = EXCLUDED.feature_set,
                        created_at = EXCLUDED.created_at
                """), {
                    'snapshot_id': snap['snapshot_id'],
                    'match_id': snap['match_id'],
                    'snapshot_at': snap['snapshot_at'],
                    'model_version': MODEL_VERSION,
                    'home_prob': float(probs[0]),
                    'draw_prob': float(probs[1]),
                    'away_prob': float(probs[2]),
                    'feature_set': json.dumps({
                        'features': FEATURE_COLUMNS,
                        'league_only': True,
                        'model_path': model_path,
                    }),
                    'created_at': created_at,
                })

                n_inserted += 1

            except Exception as e:
                logger.error(f"Error processing snapshot {snap['snapshot_id']}: {e}")
                n_errors += 1
                continue

        await session.commit()

    await engine.dispose()

    # Print summary
    print(f"\n{'='*60}")
    print(f"PREDICTION GENERATION COMPLETE: {MODEL_VERSION}")
    print(f"{'='*60}")
    print(f"  n_snapshots: {len(snapshots)}")
    print(f"  n_inserted: {n_inserted}")
    print(f"  n_errors: {n_errors}")
    print(f"  model_path: {model_path}")

    return {
        'model_version': MODEL_VERSION,
        'n_snapshots': len(snapshots),
        'n_inserted': n_inserted,
        'n_errors': n_errors,
    }


def calculate_features_asof(home_id: int, away_id: int, asof_dt: datetime, team_matches: dict) -> dict:
    """Calculate features for a match as-of a specific datetime."""

    # Get home team history before asof_dt
    home_history = []
    if home_id in team_matches:
        for dt, m in team_matches[home_id]:
            if dt < asof_dt:
                home_history.append(m)
                if len(home_history) >= ROLLING_WINDOW:
                    break

    # Get away team history before asof_dt
    away_history = []
    if away_id in team_matches:
        for dt, m in team_matches[away_id]:
            if dt < asof_dt:
                away_history.append(m)
                if len(away_history) >= ROLLING_WINDOW:
                    break

    features = {}

    # Home features
    features.update(calculate_team_stats(home_id, home_history, "home"))
    features["home_rest_days"] = calculate_rest_days(home_history, asof_dt)

    # Away features
    features.update(calculate_team_stats(away_id, away_history, "away"))
    features["away_rest_days"] = calculate_rest_days(away_history, asof_dt)

    # Derived features
    features["goal_diff_avg"] = features["home_goals_scored_avg"] - features["away_goals_scored_avg"]
    features["rest_diff"] = features["home_rest_days"] - features["away_rest_days"]

    return features


def calculate_team_stats(team_id: int, history: list, prefix: str) -> dict:
    """Calculate rolling average stats for a team."""
    if not history:
        return {
            f"{prefix}_goals_scored_avg": 0.0,
            f"{prefix}_goals_conceded_avg": 0.0,
            f"{prefix}_shots_avg": 0.0,
            f"{prefix}_corners_avg": 0.0,
            f"{prefix}_matches_played": 0,
        }

    goals_scored = []
    goals_conceded = []
    shots = []
    corners = []

    for m in history:
        if m['home_team_id'] == team_id:
            goals_scored.append(m['home_goals'] or 0)
            goals_conceded.append(m['away_goals'] or 0)
            shots.append(m['home_shots'] or 0)
            corners.append(m['home_corners'] or 0)
        else:
            goals_scored.append(m['away_goals'] or 0)
            goals_conceded.append(m['home_goals'] or 0)
            shots.append(m['away_shots'] or 0)
            corners.append(m['away_corners'] or 0)

    return {
        f"{prefix}_goals_scored_avg": np.mean(goals_scored) if goals_scored else 0.0,
        f"{prefix}_goals_conceded_avg": np.mean(goals_conceded) if goals_conceded else 0.0,
        f"{prefix}_shots_avg": np.mean(shots) if shots else 0.0,
        f"{prefix}_corners_avg": np.mean(corners) if corners else 0.0,
        f"{prefix}_matches_played": len(history),
    }


def calculate_rest_days(history: list, asof_dt: datetime) -> float:
    """Calculate days since last match."""
    if not history:
        return 7.0

    last_match_date = history[0]['date']
    if isinstance(last_match_date, str):
        last_match_date = datetime.fromisoformat(last_match_date.replace('Z', '+00:00'))

    # Handle timezone
    if hasattr(asof_dt, 'tzinfo') and asof_dt.tzinfo is not None:
        if hasattr(last_match_date, 'tzinfo') and last_match_date.tzinfo is None:
            last_match_date = last_match_date.replace(tzinfo=asof_dt.tzinfo)

    delta = asof_dt - last_match_date
    return max(1.0, min(30.0, delta.days))


def main():
    parser = argparse.ArgumentParser(description="Generate Fase 1 predictions (optimized)")
    parser.add_argument('--since', required=True, help='Start date (ISO format)')
    parser.add_argument('--until', default=None, help='End date (ISO format, optional)')
    args = parser.parse_args()

    result = asyncio.run(generate_predictions(args.since, args.until))

    print(f"\n--- JSON Summary ---")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
