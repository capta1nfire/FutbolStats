#!/usr/bin/env python3
"""
Build training dataset with TITAN features for tier comparison.

PARIDAD BASELINE: Usa FeatureEngineer de producción para calcular
FEATURE_COLUMNS exactamente igual que v1.0.0.

Usage:
    python scripts/build_titan_dataset.py --tier baseline --cutoff 2026-01-06
    python scripts/build_titan_dataset.py --tier T1b --cutoff 2026-01-06

Output:
    data/train_{tier}_{cutoff}.parquet
"""

import argparse
import asyncio
import os
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

# CRITICAL: Import from production to ensure parity
from app.ml.engine import XGBoostEngine
from app.features.engineering import FeatureEngineer
from app.models import Match

# Get FEATURE_COLUMNS from the engine class (XGBoostEngine, not MLEngine)
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

# TITAN features adicionales
# BUG #3 FIX: Added missing flags to distinguish real 0 vs imputed 0
TITAN_T1B_FEATURES = [
    "xg_home_last5", "xg_away_last5", "xga_home_last5", "xga_away_last5",
    "npxg_home_last5", "npxg_away_last5",
    "xg_home_missing", "xg_away_missing"  # 1 if <5 matches available
]
# T1b_v2: Differential features (ATI request for controlled iteration)
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


async def build_dataset(tier: str, cutoff: str, min_date: str = None, league_ids: list[int] = None) -> pd.DataFrame:
    """
    Build training dataset for specified tier.

    Uses FeatureEngineer from production to calculate BASELINE_FEATURES
    (exactly the same as v1.0.0).

    Args:
        tier: Feature tier (baseline, T1b, T1c, T1d)
        cutoff: Max date for training data (exclusive)
        min_date: Optional min date for training data (for smoke tests)
        league_ids: Optional list of league IDs to filter (e.g., [39,78,135,140,61] for Top 5)
    """
    database_url = os.environ.get('DATABASE_URL')
    if not database_url:
        raise ValueError("DATABASE_URL environment variable required")

    # Convert to async URL if needed
    if database_url.startswith('postgresql://'):
        database_url = database_url.replace('postgresql://', 'postgresql+asyncpg://')

    engine = create_async_engine(database_url)
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    cutoff_date = datetime.fromisoformat(cutoff)
    min_date_dt = datetime.fromisoformat(min_date) if min_date else None

    async with async_session() as session:
        # 1. Get completed matches before cutoff (optionally after min_date)
        query = select(Match).where(
            Match.status == "FT",
            Match.home_goals.isnot(None),
            Match.away_goals.isnot(None),
            Match.date < cutoff_date,
        )
        if min_date_dt:
            query = query.where(Match.date >= min_date_dt)
        if league_ids:
            query = query.where(Match.league_id.in_(league_ids))
        query = query.order_by(Match.date)

        result = await session.execute(query)
        matches = list(result.scalars().all())

        print(f"Found {len(matches)} matches before {cutoff}")

        # 2. Use FeatureEngineer from production for BASELINE
        engineer = FeatureEngineer(session)
        features_list = []

        for i, match in enumerate(matches):
            if i % 100 == 0:
                print(f"Processing match {i}/{len(matches)}...")

            try:
                # get_match_features calculates EXACTLY the same features as v1.0.0
                # It uses match.date internally for as-of calculations
                all_features = await engineer.get_match_features(match)

                # Extract only BASELINE_FEATURES (paridad with v1.0.0)
                row = {col: all_features.get(col, 0.0) for col in BASELINE_FEATURES}
                row['match_id'] = match.id

                # Add outcome
                if match.home_goals > match.away_goals:
                    row['outcome'] = 'H'
                elif match.home_goals == match.away_goals:
                    row['outcome'] = 'D'
                else:
                    row['outcome'] = 'A'

                features_list.append(row)

            except Exception as e:
                print(f"Warning: Failed to get features for match {match.id}: {e}")
                continue

        df = pd.DataFrame(features_list)
        print(f"Built baseline dataset with {len(df)} rows")

        # 3. For TITAN tiers, add extra features via JOIN
        if tier in ["T1b", "T1b_v2", "T1c", "T1d"]:
            print(f"Enriching with TITAN features for tier {tier}...")

            # T1b/T1b_v2: Calculate rolling xG directly from match_understat_team
            # BUG FIXES (ATI):
            #   - #2: Use direct AVG of last 5 matches (no window function ambiguity)
            #   - #3: Add missing flags (xg_home_missing, xg_away_missing)
            # NOTE: captured_at filter not applied because all xG was backfilled
            #       in Jan 2026. We assume xG available post-match in production.
            if tier in ["T1b", "T1b_v2"]:
                xg_query = text("""
                    WITH team_xg_history AS (
                        -- Consolidated xG history per team (home + away matches)
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
                        m.id as match_id,
                        -- Home team rolling (direct AVG of last 5 matches < m.date)
                        home_roll.xg_last5 as xg_home_last5,
                        home_roll.xga_last5 as xga_home_last5,
                        home_roll.npxg_last5 as npxg_home_last5,
                        home_roll.match_count as home_xg_match_count,
                        CASE WHEN home_roll.match_count < 5 THEN 1 ELSE 0 END as xg_home_missing,
                        -- Away team rolling
                        away_roll.xg_last5 as xg_away_last5,
                        away_roll.xga_last5 as xga_away_last5,
                        away_roll.npxg_last5 as npxg_away_last5,
                        away_roll.match_count as away_xg_match_count,
                        CASE WHEN away_roll.match_count < 5 THEN 1 ELSE 0 END as xg_away_missing
                    FROM matches m
                    -- Home team: AVG of last 5 matches STRICTLY BEFORE m.date
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
                              AND date < m.date  -- STRICTLY before
                            ORDER BY date DESC, match_id DESC  -- Deterministic tie-breaker
                            LIMIT 5
                        ) last5
                    ) home_roll ON TRUE
                    -- Away team: AVG of last 5 matches STRICTLY BEFORE m.date
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
                              AND date < m.date
                            ORDER BY date DESC, match_id DESC
                            LIMIT 5
                        ) last5
                    ) away_roll ON TRUE
                    WHERE m.date < :cutoff
                      AND m.status = 'FT'
                    ORDER BY m.id
                """)

                xg_result = await session.execute(xg_query, {"cutoff": cutoff_date})
                xg_rows = xg_result.fetchall()

                if xg_rows:
                    xg_df = pd.DataFrame([dict(r._mapping) for r in xg_rows])
                    n_with_xg = xg_df['xg_home_last5'].notna().sum()
                    n_home_full = (xg_df['xg_home_missing'] == 0).sum()
                    n_away_full = (xg_df['xg_away_missing'] == 0).sum()
                    print(f"Found {len(xg_df)} matches, {n_with_xg} with xG features ({100*n_with_xg/len(xg_df):.1f}%)")
                    print(f"  Home full coverage (5 matches): {n_home_full} ({100*n_home_full/len(xg_df):.1f}%)")
                    print(f"  Away full coverage (5 matches): {n_away_full} ({100*n_away_full/len(xg_df):.1f}%)")
                    df = df.merge(xg_df, on='match_id', how='left')

                    # Fill NaN with 0 for raw xG features (graceful degradation)
                    for col in TITAN_T1B_FEATURES:
                        if col in df.columns:
                            df[col] = df[col].fillna(0.0)

                    # BUG #3 FIX: Fill missing flags (1 = missing/incomplete)
                    df['xg_home_missing'] = df['xg_home_missing'].fillna(1).astype(int)
                    df['xg_away_missing'] = df['xg_away_missing'].fillna(1).astype(int)

                    # T1b_v2: Calculate differential features
                    if tier == "T1b_v2":
                        df['xg_diff_last5'] = df['xg_home_last5'] - df['xg_away_last5']
                        df['xga_diff_last5'] = df['xga_home_last5'] - df['xga_away_last5']
                        df['npxg_diff_last5'] = df['npxg_home_last5'] - df['npxg_away_last5']
                        df['xg_net_home_last5'] = df['xg_home_last5'] - df['xga_home_last5']
                        df['xg_net_away_last5'] = df['xg_away_last5'] - df['xga_away_last5']
                        df['net_diff_last5'] = df['xg_net_home_last5'] - df['xg_net_away_last5']
                        print(f"Calculated T1b_v2 differential features")
                else:
                    print("Warning: No xG data found in match_understat_team")
                    for col in TITAN_T1B_FEATURES:
                        df[col] = 0.0
                    df['xg_home_missing'] = 1
                    df['xg_away_missing'] = 1
                    if tier == "T1b_v2":
                        for col in TITAN_T1B_V2_FEATURES:
                            df[col] = 0.0

            # T1c/T1d: Use titan.feature_matrix (for now - may need similar treatment)
            elif tier in ["T1c", "T1d"]:
                titan_query = text("""
                    SELECT DISTINCT ON (fm.match_id)
                           fm.match_id,
                           fm.sofascore_lineup_integrity_score,
                           fm.lineup_home_starters_count, fm.lineup_away_starters_count,
                           fm.xi_home_def_count, fm.xi_home_mid_count, fm.xi_home_fwd_count,
                           fm.xi_away_def_count, fm.xi_away_mid_count, fm.xi_away_fwd_count,
                           fm.xi_formation_mismatch_flag
                    FROM titan.feature_matrix fm
                    JOIN matches m ON fm.match_id = m.id
                    WHERE m.date < :cutoff
                      AND fm.pit_max_captured_at <= m.date
                    ORDER BY fm.match_id, fm.pit_max_captured_at DESC
                """)

                titan_result = await session.execute(titan_query, {"cutoff": cutoff_date})
                titan_rows = titan_result.fetchall()

                if titan_rows:
                    titan_df = pd.DataFrame([dict(r._mapping) for r in titan_rows])
                    print(f"Found {len(titan_df)} TITAN feature rows")
                    df = df.merge(titan_df, on='match_id', how='left')

                    # Fill NaN with 0 for TITAN features (graceful degradation)
                    titan_cols = TITAN_T1C_FEATURES + TITAN_T1D_FEATURES
                    for col in titan_cols:
                        if col in df.columns:
                            df[col] = df[col].fillna(0.0)
                else:
                    print("Warning: No TITAN features found in feature_matrix")

        return df

    await engine.dispose()


async def main():
    parser = argparse.ArgumentParser(description="Build TITAN tier dataset")
    parser.add_argument('--tier', required=True, choices=list(TIER_FEATURES.keys()),
                        help='Tier to build: baseline, T1b, T1c, T1d')
    parser.add_argument('--cutoff', required=True,
                        help='Training cutoff date (ISO format, e.g., 2026-01-06)')
    parser.add_argument('--min-date', required=False, default=None,
                        help='Optional min date for training data (for smoke tests)')
    parser.add_argument('--league-ids', required=False, default=None,
                        help='Comma-separated list of league IDs (e.g., "39,78,135,140,61" for Top 5)')
    args = parser.parse_args()

    # Parse league IDs if provided
    league_ids = None
    if args.league_ids:
        league_ids = [int(x.strip()) for x in args.league_ids.split(',')]

    print(f"Building dataset for tier={args.tier}, cutoff={args.cutoff}")
    if args.min_date:
        print(f"  min_date={args.min_date}")
    if league_ids:
        print(f"  league_ids={league_ids}")
    print(f"BASELINE_FEATURES ({len(BASELINE_FEATURES)}): {BASELINE_FEATURES}")

    df = await build_dataset(args.tier, args.cutoff, args.min_date, league_ids)

    # Save to parquet
    output_dir = Path('data')
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / f"train_{args.tier}_{args.cutoff.replace('-', '')}.parquet"

    df.to_parquet(output_path, index=False)
    print(f"\nDataset saved: {output_path}")
    print(f"Shape: {df.shape}")
    print(f"Features: {list(df.columns)}")


if __name__ == "__main__":
    asyncio.run(main())
