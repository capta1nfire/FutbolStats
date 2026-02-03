#!/usr/bin/env python3
"""
Generate experimental predictions for extended range variants.

Usage:
    python scripts/generate_ext_predictions.py --variant A --since 2026-01-08
    python scripts/generate_ext_predictions.py --variant all --since 2026-01-08

Output:
    Inserts into predictions_experiments table with model_version = v1.0.2-ext-{A|B|C}
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import xgboost as xgb
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Feature list (same as training)
ORIGINAL_14_FEATURES = [
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

VARIANTS = {
    "A": "v1.0.2-ext-A",
    "B": "v1.0.2-ext-B",
    "C": "v1.0.2-ext-C",
}


async def get_snapshots_with_features(session: AsyncSession, since: str) -> list:
    """
    Get lineup_confirmed snapshots with features calculated at snapshot_at.
    """
    # Get snapshots
    query = text("""
        SELECT
            os.id as snapshot_id,
            os.match_id,
            os.snapshot_at,
            m.home_team_id,
            m.away_team_id,
            m.date as match_date
        FROM odds_snapshots os
        JOIN matches m ON os.match_id = m.id
        WHERE os.snapshot_type = 'lineup_confirmed'
          AND os.snapshot_at >= :since
          AND os.odds_home IS NOT NULL
          AND os.odds_draw IS NOT NULL
          AND os.odds_away IS NOT NULL
        ORDER BY os.snapshot_at
    """)

    since_dt = datetime.fromisoformat(since) if isinstance(since, str) else since
    result = await session.execute(query, {"since": since_dt})
    snapshots = [dict(r._mapping) for r in result.fetchall()]
    logger.info(f"Found {len(snapshots)} lineup_confirmed snapshots since {since}")

    if not snapshots:
        return []

    # Get all team IDs
    all_team_ids = set()
    for s in snapshots:
        all_team_ids.add(s['home_team_id'])
        all_team_ids.add(s['away_team_id'])

    # Get league matches for feature calculation
    earliest_date = datetime.fromisoformat(since) - timedelta(days=365)

    league_query = text("""
        SELECT
            m.id,
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
          AND al.kind = 'league'
          AND m.date >= :earliest
          AND (m.home_team_id = ANY(:team_ids) OR m.away_team_id = ANY(:team_ids))
        ORDER BY m.date
    """)

    result = await session.execute(league_query, {
        "earliest": earliest_date,
        "team_ids": list(all_team_ids)
    })
    league_matches = [dict(r._mapping) for r in result.fetchall()]
    logger.info(f"Fetched {len(league_matches)} league matches for feature calculation")

    # Build team match index
    from collections import defaultdict
    team_index = defaultdict(list)
    for m in league_matches:
        for tid in [m['home_team_id'], m['away_team_id']]:
            team_index[tid].append((m['date'], m))

    # Sort by date for each team
    for tid in team_index:
        team_index[tid].sort(key=lambda x: x[0])

    # Calculate features for each snapshot
    results = []
    for s in snapshots:
        snapshot_at = s['snapshot_at']
        home_id = s['home_team_id']
        away_id = s['away_team_id']

        # Get team history BEFORE snapshot_at
        home_history = []
        for dt, m in reversed(team_index.get(home_id, [])):
            if dt < snapshot_at:
                home_history.append(m)
                if len(home_history) >= 10:
                    break

        away_history = []
        for dt, m in reversed(team_index.get(away_id, [])):
            if dt < snapshot_at:
                away_history.append(m)
                if len(away_history) >= 10:
                    break

        # Calculate features
        features = {}

        # Home features
        if home_history:
            goals_s, goals_c, shots, corners = [], [], [], []
            for m in home_history:
                if m['home_team_id'] == home_id:
                    goals_s.append(m['home_goals'] or 0)
                    goals_c.append(m['away_goals'] or 0)
                    shots.append(m['home_shots'] or 0)
                    corners.append(m['home_corners'] or 0)
                else:
                    goals_s.append(m['away_goals'] or 0)
                    goals_c.append(m['home_goals'] or 0)
                    shots.append(m['away_shots'] or 0)
                    corners.append(m['away_corners'] or 0)

            features['home_goals_scored_avg'] = np.mean(goals_s)
            features['home_goals_conceded_avg'] = np.mean(goals_c)
            features['home_shots_avg'] = np.mean(shots)
            features['home_corners_avg'] = np.mean(corners)
            features['home_matches_played'] = len(home_history)

            # Rest days
            last_date = home_history[0]['date']
            delta = (snapshot_at - last_date).days if hasattr(snapshot_at, 'days') else (snapshot_at - last_date).total_seconds() / 86400
            features['home_rest_days'] = max(1, min(30, delta))
        else:
            features['home_goals_scored_avg'] = 0
            features['home_goals_conceded_avg'] = 0
            features['home_shots_avg'] = 0
            features['home_corners_avg'] = 0
            features['home_rest_days'] = 7
            features['home_matches_played'] = 0

        # Away features
        if away_history:
            goals_s, goals_c, shots, corners = [], [], [], []
            for m in away_history:
                if m['home_team_id'] == away_id:
                    goals_s.append(m['home_goals'] or 0)
                    goals_c.append(m['away_goals'] or 0)
                    shots.append(m['home_shots'] or 0)
                    corners.append(m['home_corners'] or 0)
                else:
                    goals_s.append(m['away_goals'] or 0)
                    goals_c.append(m['home_goals'] or 0)
                    shots.append(m['away_shots'] or 0)
                    corners.append(m['away_corners'] or 0)

            features['away_goals_scored_avg'] = np.mean(goals_s)
            features['away_goals_conceded_avg'] = np.mean(goals_c)
            features['away_shots_avg'] = np.mean(shots)
            features['away_corners_avg'] = np.mean(corners)
            features['away_matches_played'] = len(away_history)

            last_date = away_history[0]['date']
            delta = (snapshot_at - last_date).days if hasattr(snapshot_at, 'days') else (snapshot_at - last_date).total_seconds() / 86400
            features['away_rest_days'] = max(1, min(30, delta))
        else:
            features['away_goals_scored_avg'] = 0
            features['away_goals_conceded_avg'] = 0
            features['away_shots_avg'] = 0
            features['away_corners_avg'] = 0
            features['away_rest_days'] = 7
            features['away_matches_played'] = 0

        # Derived features
        features['goal_diff_avg'] = features['home_goals_scored_avg'] - features['away_goals_scored_avg']
        features['rest_diff'] = features['home_rest_days'] - features['away_rest_days']

        s['features'] = features
        results.append(s)

    return results


async def generate_predictions(variant: str, since: str):
    """Generate predictions for a variant."""
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant: {variant}")

    model_version = VARIANTS[variant]
    model_path = f"models/xgb_{model_version}_20260201.json"

    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    # Load model
    model = xgb.XGBClassifier()
    model.load_model(model_path)
    logger.info(f"Loaded model: {model_path}")

    database_url = os.environ.get('DATABASE_URL')
    if database_url.startswith('postgresql://'):
        database_url = database_url.replace('postgresql://', 'postgresql+asyncpg://')

    engine = create_async_engine(database_url)
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async with async_session() as session:
        # Get snapshots with features
        snapshots = await get_snapshots_with_features(session, since)

        if not snapshots:
            logger.warning("No snapshots found")
            return 0

        # Generate predictions
        inserted = 0
        for s in snapshots:
            features = s['features']
            feature_vector = np.array([[features.get(f, 0) for f in ORIGINAL_14_FEATURES]])

            probs = model.predict_proba(feature_vector)[0]

            # Insert into predictions_experiments
            insert_query = text("""
                INSERT INTO predictions_experiments
                (snapshot_id, match_id, snapshot_at, model_version,
                 home_prob, draw_prob, away_prob, feature_set, created_at)
                VALUES (:snapshot_id, :match_id, :snapshot_at, :model_version,
                        :home_prob, :draw_prob, :away_prob, :feature_set, :created_at)
                ON CONFLICT (snapshot_id, model_version) DO NOTHING
            """)

            await session.execute(insert_query, {
                "snapshot_id": s['snapshot_id'],
                "match_id": s['match_id'],
                "snapshot_at": s['snapshot_at'],
                "model_version": model_version,
                "home_prob": float(probs[0]),
                "draw_prob": float(probs[1]),
                "away_prob": float(probs[2]),
                "feature_set": json.dumps(ORIGINAL_14_FEATURES),
                "created_at": s['snapshot_at'] - timedelta(seconds=1),
            })
            inserted += 1

        await session.commit()
        logger.info(f"Inserted {inserted} predictions for {model_version}")
        return inserted

    await engine.dispose()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', required=True, choices=['A', 'B', 'C', 'all'])
    parser.add_argument('--since', required=True, help='Min snapshot date (ISO)')
    args = parser.parse_args()

    if args.variant == 'all':
        variants = ['A', 'B', 'C']
    else:
        variants = [args.variant]

    for v in variants:
        count = asyncio.run(generate_predictions(v, args.since))
        print(f"Variant {v}: {count} predictions generated")


if __name__ == "__main__":
    main()
