#!/usr/bin/env python3
"""
Evaluación Preliminar del Modelo - NOT PIT (Point-In-Time)

IMPORTANTE: Esta evaluación usa odds de opening/closing, NO odds capturados en tiempo real
al momento de lineup_confirmed. Los resultados son PRELIMINARES y NO deben usarse
para decisiones de negocio sobre alpha real.

Pre-registered configuration (principal):
- Threshold: 0.02
- Stake: fijo (1 unidad)
- Gate: actual (rotation/meta)
- Todo lo demás: exploratorio

Métricas:
- Brier Score vs Market
- ROI/EV por apuestas ejecutadas
- Coverage (% de partidos en los que apuestas)
- Bootstrap CI95%

Run with:
    DATABASE_URL="postgresql://..." python scripts/evaluate_model_not_pit.py
"""

import asyncio
import os
import sys
import json
from datetime import datetime
from typing import Optional
import warnings

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

DATABASE_URL = os.environ.get("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("DATABASE_URL environment variable is required")

# Convert to async URL if needed
if DATABASE_URL.startswith("postgresql://"):
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://")


# =============================================================================
# CONFIGURATION - PRE-REGISTERED
# =============================================================================

# Principal configuration (pre-registered, avoid p-hacking)
PRINCIPAL_CONFIG = {
    "name": "PRINCIPAL",
    "edge_threshold": 0.02,  # p_model - implied_prob > 0.02
    "stake": 1.0,  # Fixed stake
    "use_gate": True,  # Use rotation/meta gate
    "odds_type": "opening",  # opening or closing
}

# Exploratory configurations (clearly labeled)
EXPLORATORY_CONFIGS = [
    {"name": "threshold_0.01", "edge_threshold": 0.01, "stake": 1.0, "use_gate": True, "odds_type": "opening"},
    {"name": "threshold_0.03", "edge_threshold": 0.03, "stake": 1.0, "use_gate": True, "odds_type": "opening"},
    {"name": "threshold_0.05", "edge_threshold": 0.05, "stake": 1.0, "use_gate": True, "odds_type": "opening"},
    {"name": "no_gate_0.02", "edge_threshold": 0.02, "stake": 1.0, "use_gate": False, "odds_type": "opening"},
    {"name": "closing_0.02", "edge_threshold": 0.02, "stake": 1.0, "use_gate": True, "odds_type": "closing"},
]

# Bootstrap settings
N_BOOTSTRAP = 1000
CONFIDENCE_LEVEL = 0.95


# =============================================================================
# DATA LOADING
# =============================================================================

async def load_evaluation_data(engine, season: int = 2024) -> pd.DataFrame:
    """
    Load matches with predictions, odds, and outcomes for evaluation.

    Returns DataFrame with:
    - match_id, date, home_team, away_team
    - prob_home, prob_draw, prob_away (model predictions)
    - opening_odds_home/draw/away or closing odds
    - actual_outcome (0=home, 1=draw, 2=away)
    - gate features (rotation, congestion, etc.)
    """
    async with engine.connect() as conn:
        result = await conn.execute(text("""
            SELECT
                m.id as match_id,
                m.date,
                m.season,
                ht.name as home_team,
                at.name as away_team,
                m.home_goals,
                m.away_goals,
                -- Model predictions (from predictions table)
                p.home_prob as prob_home,
                p.draw_prob as prob_draw,
                p.away_prob as prob_away,
                -- Opening odds
                m.opening_odds_home,
                m.opening_odds_draw,
                m.opening_odds_away,
                -- Closing odds (current odds as proxy)
                m.odds_home as closing_odds_home,
                m.odds_draw as closing_odds_draw,
                m.odds_away as closing_odds_away,
                -- Lineup features from matches table (for gate)
                m.home_lineup_surprise_index,
                m.away_lineup_surprise_index,
                m.home_missing_starter_count,
                m.away_missing_starter_count
            FROM matches m
            JOIN teams ht ON m.home_team_id = ht.id
            JOIN teams at ON m.away_team_id = at.id
            LEFT JOIN predictions p ON m.id = p.match_id
            WHERE m.status = 'FT'
              AND m.season >= :season
              AND m.home_goals IS NOT NULL
              AND m.away_goals IS NOT NULL
              AND m.opening_odds_home IS NOT NULL
            ORDER BY m.date
        """), {"season": season})

        rows = result.fetchall()
        columns = result.keys()

        df = pd.DataFrame(rows, columns=columns)

        # Calculate actual outcome
        df['actual_outcome'] = df.apply(
            lambda r: 0 if r['home_goals'] > r['away_goals']
                     else (1 if r['home_goals'] == r['away_goals'] else 2),
            axis=1
        )

        # Fill missing predictions with market implied probs
        for col, odds_col in [('prob_home', 'opening_odds_home'),
                               ('prob_draw', 'opening_odds_draw'),
                               ('prob_away', 'opening_odds_away')]:
            mask = df[col].isna() & df[odds_col].notna()
            if mask.any():
                # De-vig using power method
                df.loc[mask, col] = 1 / df.loc[mask, odds_col]

        # Normalize probabilities to sum to 1
        prob_cols = ['prob_home', 'prob_draw', 'prob_away']
        prob_sum = df[prob_cols].sum(axis=1)
        for col in prob_cols:
            df[col] = df[col] / prob_sum

        return df


# =============================================================================
# METRICS CALCULATION
# =============================================================================

def calculate_implied_probs(odds_h, odds_d, odds_a):
    """Convert odds to implied probabilities with de-vig."""
    raw_probs = np.array([1/odds_h, 1/odds_d, 1/odds_a])
    # De-vig by normalizing
    return raw_probs / raw_probs.sum()


def brier_score_multiclass(probs, actual_outcome):
    """
    Calculate multi-class Brier score using OvA (One-vs-All) averaging.

    This matches the formula in app/ml/metrics.py for consistency.
    Uses sklearn.brier_score_loss for each class, then averages.

    probs: array of shape (n_samples, 3) with [prob_home, prob_draw, prob_away]
    actual_outcome: array of shape (n_samples,) with 0=home, 1=draw, 2=away

    Returns:
        Average Brier score across all classes (lower is better).
        Perfect = 0, random = ~0.22 for 3-class balanced.
    """
    from sklearn.metrics import brier_score_loss

    n_classes = probs.shape[1]
    brier_scores = []

    for cls in range(n_classes):
        y_true_binary = (actual_outcome == cls).astype(int)
        y_proba_cls = probs[:, cls]
        score = brier_score_loss(y_true_binary, y_proba_cls)
        brier_scores.append(score)

    return np.mean(brier_scores)


def brier_score_sum_squares(probs, actual_outcome):
    """
    Calculate Brier score using sum of squared errors (classic formula).

    This is the traditional Brier formula: mean(sum((p_i - y_i)^2))
    Often used in betting/sports analytics literature.

    Returns:
        Mean squared error sum (lower is better).
        Perfect = 0, random = ~0.67 for 3-class balanced.
    """
    n = len(actual_outcome)
    actual_one_hot = np.zeros((n, 3))
    actual_one_hot[np.arange(n), actual_outcome] = 1

    return np.mean(np.sum((probs - actual_one_hot) ** 2, axis=1))


def calculate_roi(bets, outcomes, odds, stakes):
    """
    Calculate ROI from bets.
    bets: array of bet types (0=home, 1=draw, 2=away, -1=no bet)
    outcomes: actual outcomes
    odds: odds for each outcome (n_samples, 3)
    stakes: stake per bet
    """
    total_stake = 0
    total_return = 0

    for i, bet in enumerate(bets):
        if bet >= 0:
            stake = stakes if isinstance(stakes, (int, float)) else stakes[i]
            total_stake += stake
            if outcomes[i] == bet:
                total_return += stake * odds[i, bet]

    if total_stake == 0:
        return 0, 0, 0

    profit = total_return - total_stake
    roi = profit / total_stake

    return roi, profit, total_stake


def apply_gate(df, use_gate: bool):
    """Apply gate filter based on lineup surprise and missing starters."""
    if not use_gate:
        return df

    # Gate: exclude matches with high lineup surprise or too many missing starters
    # This is a conservative filter to avoid betting on matches with unpredictable lineups
    mask = pd.Series(True, index=df.index)

    # Exclude if lineup surprise index is too high (> 0.3 = 30% unexpected players)
    if 'home_lineup_surprise_index' in df.columns and df['home_lineup_surprise_index'].notna().any():
        mask &= (df['home_lineup_surprise_index'].fillna(0) <= 0.3) & (df['away_lineup_surprise_index'].fillna(0) <= 0.3)

    # Exclude if too many starters are missing (> 3)
    if 'home_missing_starter_count' in df.columns and df['home_missing_starter_count'].notna().any():
        mask &= (df['home_missing_starter_count'].fillna(0) <= 3) & (df['away_missing_starter_count'].fillna(0) <= 3)

    return df[mask]


def identify_value_bets(df, config: dict) -> np.ndarray:
    """
    Identify value bets based on edge threshold.
    Returns array of bet types (-1=no bet, 0=home, 1=draw, 2=away)
    """
    odds_type = config.get("odds_type", "opening")
    threshold = config["edge_threshold"]

    if odds_type == "opening":
        odds_cols = ['opening_odds_home', 'opening_odds_draw', 'opening_odds_away']
    else:
        odds_cols = ['closing_odds_home', 'closing_odds_draw', 'closing_odds_away']

    # Filter to rows with valid odds
    valid_mask = df[odds_cols].notna().all(axis=1)

    bets = np.full(len(df), -1)  # -1 = no bet

    for i, row in df[valid_mask].iterrows():
        idx = df.index.get_loc(i)

        # Get odds and implied probs
        odds = np.array([row[odds_cols[0]], row[odds_cols[1]], row[odds_cols[2]]])
        implied_probs = 1 / odds
        implied_probs = implied_probs / implied_probs.sum()  # de-vig

        # Get model probs
        model_probs = np.array([row['prob_home'], row['prob_draw'], row['prob_away']])

        # Calculate edge for each outcome
        edges = model_probs - implied_probs

        # Find best value bet (if any exceeds threshold)
        best_outcome = np.argmax(edges)
        if edges[best_outcome] > threshold:
            # Additional check: positive EV
            ev = model_probs[best_outcome] * odds[best_outcome] - 1
            if ev > 0:
                bets[idx] = best_outcome

    return bets


def bootstrap_ci(data, statistic_func, n_bootstrap=1000, confidence=0.95):
    """
    Calculate bootstrap confidence interval.
    Uses block bootstrap for time series data if data is ordered by time.
    """
    n = len(data)

    if n < 10:
        return np.nan, np.nan, np.nan

    # Block size for block bootstrap (weekly blocks ~ 7 matches)
    block_size = min(7, n // 10)
    if block_size < 1:
        block_size = 1

    boot_stats = []
    n_blocks = n // block_size

    for _ in range(n_bootstrap):
        # Block bootstrap
        if block_size > 1 and n_blocks > 1:
            block_indices = np.random.choice(n_blocks, size=n_blocks, replace=True)
            indices = []
            for bi in block_indices:
                start = bi * block_size
                end = min(start + block_size, n)
                indices.extend(range(start, end))
            indices = np.array(indices[:n])  # Trim to original size
        else:
            indices = np.random.choice(n, size=n, replace=True)

        try:
            stat = statistic_func(data.iloc[indices] if isinstance(data, pd.DataFrame) else data[indices])
            boot_stats.append(stat)
        except Exception:
            continue

    if len(boot_stats) < 100:
        return np.nan, np.nan, np.nan

    boot_stats = np.array(boot_stats)
    alpha = 1 - confidence
    ci_lower = np.percentile(boot_stats, alpha/2 * 100)
    ci_upper = np.percentile(boot_stats, (1 - alpha/2) * 100)

    return np.mean(boot_stats), ci_lower, ci_upper


# =============================================================================
# MAIN EVALUATION
# =============================================================================

async def evaluate_config(df: pd.DataFrame, config: dict) -> dict:
    """Evaluate a single configuration."""

    # Apply gate if configured
    df_filtered = apply_gate(df.copy(), config.get("use_gate", True))

    if len(df_filtered) < 50:
        return {
            "config": config["name"],
            "error": f"Insufficient data after filtering: {len(df_filtered)} matches",
            "n_matches": len(df_filtered),
        }

    # Get odds type
    odds_type = config.get("odds_type", "opening")
    if odds_type == "opening":
        odds_cols = ['opening_odds_home', 'opening_odds_draw', 'opening_odds_away']
    else:
        odds_cols = ['closing_odds_home', 'closing_odds_draw', 'closing_odds_away']

    # Filter to matches with valid odds
    valid_odds = df_filtered[odds_cols].notna().all(axis=1)
    df_eval = df_filtered[valid_odds].copy()

    if len(df_eval) < 50:
        return {
            "config": config["name"],
            "error": f"Insufficient data with valid odds: {len(df_eval)} matches",
            "n_matches": len(df_eval),
        }

    # Calculate Brier scores
    model_probs = df_eval[['prob_home', 'prob_draw', 'prob_away']].values
    actual = df_eval['actual_outcome'].values

    model_brier = brier_score_multiclass(model_probs, actual)

    # Market Brier
    market_probs = np.zeros_like(model_probs)
    for i, row in df_eval.iterrows():
        idx = df_eval.index.get_loc(i)
        odds = np.array([row[odds_cols[0]], row[odds_cols[1]], row[odds_cols[2]]])
        market_probs[idx] = calculate_implied_probs(odds[0], odds[1], odds[2])

    market_brier = brier_score_multiclass(market_probs, actual)
    delta_brier = model_brier - market_brier  # Negative = model better

    # Identify value bets
    bets = identify_value_bets(df_eval, config)
    n_bets = np.sum(bets >= 0)
    coverage = n_bets / len(df_eval)

    # Calculate ROI
    odds_array = df_eval[odds_cols].values
    roi, profit, total_stake = calculate_roi(bets, actual, odds_array, config["stake"])

    # Bootstrap CIs
    def brier_diff_func(data):
        m_probs = data[['prob_home', 'prob_draw', 'prob_away']].values
        mkt_probs = np.zeros_like(m_probs)
        for i, (_, row) in enumerate(data.iterrows()):
            odds = np.array([row[odds_cols[0]], row[odds_cols[1]], row[odds_cols[2]]])
            mkt_probs[i] = calculate_implied_probs(odds[0], odds[1], odds[2])
        actual_boot = data['actual_outcome'].values
        return brier_score_multiclass(m_probs, actual_boot) - brier_score_multiclass(mkt_probs, actual_boot)

    def roi_func(data):
        bets_boot = identify_value_bets(data, config)
        actual_boot = data['actual_outcome'].values
        odds_boot = data[odds_cols].values
        roi_boot, _, _ = calculate_roi(bets_boot, actual_boot, odds_boot, config["stake"])
        return roi_boot

    # Only bootstrap if we have enough data
    if len(df_eval) >= 100:
        brier_mean, brier_ci_low, brier_ci_high = bootstrap_ci(df_eval, brier_diff_func, N_BOOTSTRAP, CONFIDENCE_LEVEL)
        roi_mean, roi_ci_low, roi_ci_high = bootstrap_ci(df_eval, roi_func, N_BOOTSTRAP, CONFIDENCE_LEVEL)
    else:
        brier_mean, brier_ci_low, brier_ci_high = delta_brier, np.nan, np.nan
        roi_mean, roi_ci_low, roi_ci_high = roi, np.nan, np.nan

    # Decision criterion
    if np.isnan(roi_ci_low) or np.isnan(brier_ci_high):
        decision = "INCONCLUSIVE - Insufficient data for CI"
    elif roi_ci_low > 0 and brier_ci_high < 0:
        decision = "CONTINUE - Positive signal"
    elif roi_ci_high < 0:
        decision = "CLOSE - Negative ROI"
    elif brier_ci_low > 0:
        decision = "CLOSE - Model worse than market"
    else:
        decision = "INCONCLUSIVE - CI crosses zero"

    return {
        "config": config["name"],
        "n_matches": len(df_eval),
        "n_bets": n_bets,
        "coverage": coverage,
        "model_brier": model_brier,
        "market_brier": market_brier,
        "delta_brier": delta_brier,
        "delta_brier_ci": (brier_ci_low, brier_ci_high),
        "roi": roi,
        "roi_ci": (roi_ci_low, roi_ci_high),
        "profit": profit,
        "total_stake": total_stake,
        "decision": decision,
    }


async def main():
    print("=" * 80)
    print("EVALUACIÓN PRELIMINAR DEL MODELO - NOT PIT")
    print("=" * 80)
    print(f"Timestamp: {datetime.utcnow().isoformat()}")
    print()
    print("⚠️  ADVERTENCIA: Esta evaluación usa odds de opening/closing.")
    print("⚠️  NO es Point-In-Time. Los resultados son PRELIMINARES.")
    print("⚠️  No tomar decisiones de negocio basadas en esto.")
    print()

    engine = create_async_engine(DATABASE_URL)

    # Load data
    print("Cargando datos...")
    df = await load_evaluation_data(engine, season=2024)
    print(f"Total partidos cargados: {len(df)}")
    print(f"Rango de fechas: {df['date'].min()} a {df['date'].max()}")
    print()

    # Check data quality
    has_predictions = df['prob_home'].notna().sum()
    has_opening = df['opening_odds_home'].notna().sum()
    has_closing = df['closing_odds_home'].notna().sum()

    print("Calidad de datos:")
    print(f"  Con predicciones del modelo: {has_predictions} ({has_predictions/len(df)*100:.1f}%)")
    print(f"  Con opening odds: {has_opening} ({has_opening/len(df)*100:.1f}%)")
    print(f"  Con closing odds: {has_closing} ({has_closing/len(df)*100:.1f}%)")
    print()

    results = []

    # ==========================================================================
    # PRINCIPAL CONFIGURATION (PRE-REGISTERED)
    # ==========================================================================
    print("=" * 80)
    print("CONFIGURACIÓN PRINCIPAL (PRE-REGISTERED)")
    print("=" * 80)
    print(f"  Threshold: {PRINCIPAL_CONFIG['edge_threshold']}")
    print(f"  Stake: fijo ({PRINCIPAL_CONFIG['stake']})")
    print(f"  Gate: {'Sí' if PRINCIPAL_CONFIG['use_gate'] else 'No'}")
    print(f"  Odds: {PRINCIPAL_CONFIG['odds_type']}")
    print()

    principal_result = await evaluate_config(df, PRINCIPAL_CONFIG)
    results.append(principal_result)

    if "error" not in principal_result:
        print(f"Partidos evaluados: {principal_result['n_matches']}")
        print(f"Apuestas realizadas: {principal_result['n_bets']} ({principal_result['coverage']*100:.1f}% coverage)")
        print()
        print("Métricas:")
        print(f"  Brier Score (modelo): {principal_result['model_brier']:.4f}")
        print(f"  Brier Score (mercado): {principal_result['market_brier']:.4f}")
        print(f"  Delta Brier: {principal_result['delta_brier']:.4f} {'✅ modelo mejor' if principal_result['delta_brier'] < 0 else '❌ mercado mejor'}")
        ci_low, ci_high = principal_result['delta_brier_ci']
        print(f"  Delta Brier CI95%: [{ci_low:.4f}, {ci_high:.4f}]")
        print()
        print(f"  ROI: {principal_result['roi']*100:.2f}%")
        roi_ci_low, roi_ci_high = principal_result['roi_ci']
        print(f"  ROI CI95%: [{roi_ci_low*100:.2f}%, {roi_ci_high*100:.2f}%]")
        print(f"  Profit: {principal_result['profit']:.2f} unidades")
        print(f"  Total apostado: {principal_result['total_stake']:.2f} unidades")
        print()
        print(f"📊 DECISIÓN: {principal_result['decision']}")
    else:
        print(f"❌ Error: {principal_result['error']}")

    # ==========================================================================
    # EXPLORATORY CONFIGURATIONS
    # ==========================================================================
    print()
    print("=" * 80)
    print("CONFIGURACIONES EXPLORATORIAS")
    print("=" * 80)
    print("(Solo para análisis interno, no para decisiones)")
    print()

    for config in EXPLORATORY_CONFIGS:
        result = await evaluate_config(df, config)
        results.append(result)

        print(f"\n--- {config['name']} ---")
        if "error" not in result:
            print(f"  N: {result['n_matches']}, Bets: {result['n_bets']}, Coverage: {result['coverage']*100:.1f}%")
            print(f"  Delta Brier: {result['delta_brier']:.4f}")
            print(f"  ROI: {result['roi']*100:.2f}% CI95%: [{result['roi_ci'][0]*100:.2f}%, {result['roi_ci'][1]*100:.2f}%]")
        else:
            print(f"  Error: {result['error']}")

    # ==========================================================================
    # SUMMARY TABLE
    # ==========================================================================
    print()
    print("=" * 80)
    print("RESUMEN")
    print("=" * 80)

    print(f"\n{'Config':<20} {'N':<6} {'Bets':<6} {'Cov%':<6} {'ΔBrier':<8} {'ROI%':<8} {'Decision'}")
    print("-" * 80)

    for r in results:
        if "error" not in r:
            print(f"{r['config']:<20} {r['n_matches']:<6} {r['n_bets']:<6} {r['coverage']*100:<6.1f} {r['delta_brier']:<8.4f} {r['roi']*100:<8.2f} {r['decision'][:20]}")
        else:
            print(f"{r['config']:<20} {r.get('n_matches', 0):<6} {'ERROR':<6}")

    # Save results to JSON
    output_file = f"scripts/evaluation_not_pit_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump({
            "timestamp": datetime.utcnow().isoformat(),
            "warning": "NOT PIT - PRELIMINARY RESULTS ONLY",
            "results": results,
        }, f, indent=2, default=str)

    print(f"\nResultados guardados en: {output_file}")

    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
