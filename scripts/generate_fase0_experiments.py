#!/usr/bin/env python3
"""
Generate Fase 0 A/B experimental predictions.

Creates predictions for the SAME cohort of snapshots with different pipelines:
- v1.0.0-control: league_only=False, no kill-switch (pre-Fase0 baseline)
- v1.0.1-league-only: league_only=True, no kill-switch (isolate feature effect)
- v1.0.1-killswitch: league_only=True, with kill-switch (full Fase0)

All variants use the same v1.0.0 model weights - only the feature pipeline differs.

Usage:
    # Generate all 3 variants
    python scripts/generate_fase0_experiments.py --variant control --since 2026-01-07
    python scripts/generate_fase0_experiments.py --variant league-only --since 2026-01-07
    python scripts/generate_fase0_experiments.py --variant killswitch --since 2026-01-07

Output:
    Inserts into predictions_experiments with PIT-safe created_at = snapshot_at - 1s
"""

import argparse
import asyncio
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

from app.ml.engine import XGBoostEngine
from app.features.engineering import FeatureEngineer
from app.models import Match

# Variant configurations
VARIANTS = {
    "control": {
        "model_version": "v1.0.0-control",
        "model_path": None,  # Use default v1.0.0
        "league_only": False,
        "killswitch": False,
        "description": "Pre-Fase0 baseline: all matches, all competitions"
    },
    "league-only": {
        "model_version": "v1.0.1-league-only",
        "model_path": None,  # Use default v1.0.0 (skew experiment)
        "league_only": True,
        "killswitch": False,
        "description": "Fase0 features: league-only rolling averages, no gating (SKEW)"
    },
    "killswitch": {
        "model_version": "v1.0.1-killswitch",
        "model_path": None,  # Use default v1.0.0 (skew experiment)
        "league_only": True,
        "killswitch": True,
        "description": "Full Fase0: league-only features + kill-switch gating (SKEW)"
    },
    # FASE 1: New model trained with league_only=True (NO SKEW)
    "league-only-new": {
        "model_version": "v1.0.1-league-only-trained",
        "model_path": "models/xgb_v1.0.1-league-only_*.json",  # New model
        "league_only": True,
        "killswitch": False,
        "description": "FASE 1: Model trained with league_only=True (NO SKEW)"
    },
}

# Kill-switch parameters (same as scheduler.py)
MIN_LEAGUE_MATCHES = 5
LOOKBACK_DAYS = 90


async def get_snapshots(session: AsyncSession, since: datetime, until: Optional[datetime] = None) -> list[dict]:
    """Get all lineup_confirmed snapshots in the date range."""
    query = """
        SELECT os.id as snapshot_id, os.match_id, os.snapshot_at,
               m.home_team_id, m.away_team_id, m.date as match_date
        FROM odds_snapshots os
        JOIN matches m ON os.match_id = m.id
        WHERE os.snapshot_type = 'lineup_confirmed'
          AND os.snapshot_at >= :since
    """
    params = {"since": since}

    if until:
        query += " AND os.snapshot_at < :until"
        params["until"] = until

    query += " ORDER BY os.snapshot_at"

    result = await session.execute(text(query), params)
    return [dict(r._mapping) for r in result.fetchall()]


async def get_team_league_history(session: AsyncSession, team_ids: list[int], earliest_date: datetime) -> dict:
    """
    Get league match history for kill-switch evaluation.
    Returns dict: team_id -> [list of match dates]
    """
    query = text("""
        SELECT team_id, match_date
        FROM (
            SELECT home_team_id as team_id, date as match_date
            FROM matches m
            JOIN admin_leagues al ON m.league_id = al.league_id
            WHERE m.status = 'FT'
              AND al.kind = 'league'
              AND m.date >= :earliest_date
            UNION ALL
            SELECT away_team_id as team_id, date as match_date
            FROM matches m
            JOIN admin_leagues al ON m.league_id = al.league_id
            WHERE m.status = 'FT'
              AND al.kind = 'league'
              AND m.date >= :earliest_date
        ) sub
        WHERE team_id = ANY(:team_ids)
        ORDER BY team_id, match_date DESC
    """)

    result = await session.execute(query, {"team_ids": team_ids, "earliest_date": earliest_date})

    team_match_dates = defaultdict(list)
    for row in result.fetchall():
        team_match_dates[row.team_id].append(row.match_date)

    return team_match_dates


def check_killswitch(home_id: int, away_id: int, match_date: datetime,
                     team_match_dates: dict) -> tuple[bool, str]:
    """
    Check if match passes kill-switch.
    Returns (is_eligible, reason) where reason is set if not eligible.
    """
    cutoff = match_date - timedelta(days=LOOKBACK_DAYS)

    home_count = sum(
        1 for d in team_match_dates.get(home_id, [])
        if cutoff <= d < match_date
    )
    away_count = sum(
        1 for d in team_match_dates.get(away_id, [])
        if cutoff <= d < match_date
    )

    home_ok = home_count >= MIN_LEAGUE_MATCHES
    away_ok = away_count >= MIN_LEAGUE_MATCHES

    if home_ok and away_ok:
        return True, ""
    elif not home_ok and not away_ok:
        return False, "both_insufficient"
    elif not home_ok:
        return False, "home_insufficient"
    else:
        return False, "away_insufficient"


async def get_match(session: AsyncSession, match_id: int) -> Optional[Match]:
    """Get match by ID."""
    from sqlalchemy import select
    result = await session.execute(
        select(Match).where(Match.id == match_id)
    )
    return result.scalar_one_or_none()


async def generate_variant(variant: str, since: str, until: Optional[str] = None):
    """Generate predictions for a specific variant."""

    config = VARIANTS[variant]
    print(f"\n{'='*60}")
    print(f"Generating: {config['model_version']}")
    print(f"Description: {config['description']}")
    print(f"league_only={config['league_only']}, killswitch={config['killswitch']}")
    print(f"{'='*60}\n")

    database_url = os.environ.get('DATABASE_URL')
    if not database_url:
        raise ValueError("DATABASE_URL environment variable required")

    if database_url.startswith('postgresql://'):
        database_url = database_url.replace('postgresql://', 'postgresql+asyncpg://')

    engine_db = create_async_engine(database_url)
    async_session = sessionmaker(engine_db, class_=AsyncSession, expire_on_commit=False)

    # Load model - either default v1.0.0 or specific path for new variants
    ml_engine = XGBoostEngine()

    model_path = config.get('model_path')
    if model_path and '*' in model_path:
        # Glob pattern - find latest matching model
        import glob
        matching = sorted(glob.glob(model_path))
        if not matching:
            raise RuntimeError(f"No model found matching: {model_path}")
        model_path = matching[-1]  # Latest
        print(f"Using model: {model_path}")

    if model_path:
        if not ml_engine.load_model(model_path):
            raise RuntimeError(f"Could not load model from {model_path}")
    else:
        if not ml_engine.load_model():
            raise RuntimeError("Could not load ML model v1.0.0")

    # ATI AUDIT: Log model↔features alignment
    expected_features = ml_engine._get_model_expected_features()
    print(f"Loaded model: {ml_engine.model_version}")
    print(f"  n_expected_features: {len(expected_features)}")
    print(f"  expected_features: {expected_features}")
    print(f"  FEATURE_COLUMNS (code): {len(ml_engine.FEATURE_COLUMNS)}")

    since_date = datetime.fromisoformat(since)
    until_date = datetime.fromisoformat(until) if until else None

    # Counters
    n_snapshots_total = 0
    n_inserted = 0
    n_skipped_killswitch = 0
    skipped_by_reason = defaultdict(int)
    n_errors = 0

    async with async_session() as session:
        # Get all snapshots
        snapshots = await get_snapshots(session, since_date, until_date)
        n_snapshots_total = len(snapshots)
        print(f"Found {n_snapshots_total} snapshots since {since}")

        # For kill-switch variant, pre-fetch team league history
        team_match_dates = {}
        if config['killswitch']:
            all_team_ids = list(set(
                [s['home_team_id'] for s in snapshots] +
                [s['away_team_id'] for s in snapshots]
            ))
            if snapshots:
                min_match_date = min(s['match_date'] for s in snapshots)
                earliest_needed = min_match_date - timedelta(days=LOOKBACK_DAYS + 7)
                team_match_dates = await get_team_league_history(
                    session, all_team_ids, earliest_needed
                )
                print(f"Fetched league history for {len(all_team_ids)} teams")

        # Create FeatureEngineer
        feature_engineer = FeatureEngineer(session=session)

        for i, snap in enumerate(snapshots):
            if i % 50 == 0:
                print(f"Processing snapshot {i}/{n_snapshots_total}...")

            try:
                # Kill-switch check (only for killswitch variant)
                if config['killswitch']:
                    is_eligible, reason = check_killswitch(
                        snap['home_team_id'],
                        snap['away_team_id'],
                        snap['match_date'],
                        team_match_dates
                    )
                    if not is_eligible:
                        n_skipped_killswitch += 1
                        skipped_by_reason[reason] += 1
                        continue

                # Get match
                match = await get_match(session, snap['match_id'])
                if not match:
                    n_errors += 1
                    continue

                # Get features as-of snapshot_at (PIT-safe)
                # league_only controls whether rolling averages use only league matches
                features = await feature_engineer.get_match_features_asof(
                    match, snap['snapshot_at'], league_only=config['league_only']
                )

                # Build feature vector for model
                # Use _get_model_expected_features() for backward compatibility
                # (v1.0.0 has 14 features, FEATURE_COLUMNS now has 17)
                expected_cols = ml_engine._get_model_expected_features()
                feature_vector = [float(features.get(col, 0.0)) for col in expected_cols]

                # Predict
                X = np.array([feature_vector])
                probs = ml_engine.model.predict_proba(X)[0]  # [home, draw, away]

                # Insert with PIT-safe created_at
                created_at = snap['snapshot_at'] - timedelta(seconds=1)

                insert_query = text("""
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
                """)

                await session.execute(insert_query, {
                    'snapshot_id': snap['snapshot_id'],
                    'match_id': snap['match_id'],
                    'snapshot_at': snap['snapshot_at'],
                    'model_version': config['model_version'],
                    'home_prob': float(probs[0]),
                    'draw_prob': float(probs[1]),
                    'away_prob': float(probs[2]),
                    'feature_set': json.dumps({
                        'features': list(ml_engine.FEATURE_COLUMNS),
                        'league_only': config['league_only'],
                        'killswitch': config['killswitch'],
                    }),
                    'created_at': created_at,
                })

                n_inserted += 1

            except Exception as e:
                print(f"Error processing snapshot {snap['snapshot_id']}: {e}")
                n_errors += 1
                continue

        await session.commit()

    await engine_db.dispose()

    # Print summary
    print(f"\n{'='*60}")
    print(f"GENERATION COMPLETE: {config['model_version']}")
    print(f"{'='*60}")
    print(f"  n_snapshots_total: {n_snapshots_total}")
    print(f"  n_inserted: {n_inserted}")
    if config['killswitch']:
        print(f"  n_skipped_killswitch: {n_skipped_killswitch}")
        for reason, count in sorted(skipped_by_reason.items()):
            print(f"    - {reason}: {count}")
    print(f"  n_errors: {n_errors}")

    return {
        'model_version': config['model_version'],
        'n_snapshots_total': n_snapshots_total,
        'n_inserted': n_inserted,
        'n_skipped_killswitch': n_skipped_killswitch,
        'skipped_by_reason': dict(skipped_by_reason),
        'n_errors': n_errors,
    }


def main():
    parser = argparse.ArgumentParser(description="Generate Fase 0 A/B experiments")
    parser.add_argument('--variant', required=True, choices=list(VARIANTS.keys()),
                        help='Variant to generate: control, league-only, killswitch, league-only-new')
    parser.add_argument('--since', required=True,
                        help='Start date (ISO format, e.g., 2026-01-07)')
    parser.add_argument('--until', default=None,
                        help='End date (ISO format, optional)')
    args = parser.parse_args()

    result = asyncio.run(generate_variant(args.variant, args.since, args.until))

    # Print JSON summary for ATI
    print(f"\n--- JSON Summary ---")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
