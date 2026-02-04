#!/usr/bin/env python3
"""
Evaluate policy with draw cap - Fase 1

Compara:
- Policy actual: best_edge >= 0.05 (sin cap)
- Policy con cap: MAX_DRAW_SHARE = 0.35

Feature flags:
- POLICY_DRAW_CAP_ENABLED = True/False
- POLICY_MAX_DRAW_SHARE = 0.35

ATI request - 2026-02-01
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from collections import defaultdict
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════
# POLICY FEATURE FLAGS
# ═══════════════════════════════════════════════════════════════
POLICY_DRAW_CAP_ENABLED = True
POLICY_MAX_DRAW_SHARE = 0.35
POLICY_EDGE_THRESHOLD = 0.05


def devig_proportional(odds_home: float, odds_draw: float, odds_away: float) -> list[float]:
    """Convert odds to fair probabilities using proportional devig."""
    implied = [1/odds_home, 1/odds_draw, 1/odds_away]
    total = sum(implied)
    return [p/total for p in implied]


def apply_draw_cap(selected_bets: list, max_draw_share: float = POLICY_MAX_DRAW_SHARE) -> list:
    """
    Apply draw cap to selected bets.

    Ensures draws <= max_draw_share of FINAL total.
    Formula: max_draws = floor(n_others / (1 - max_draw_share) * max_draw_share)

    Example: If max_draw_share=0.35 and n_others=135:
    - Final total should have 35% draws
    - n_others = 65% of total → total = 135/0.65 = 207.7
    - max_draws = 207.7 * 0.35 = 72.7 → 72

    Args:
        selected_bets: List of dicts with 'pick', 'edge', 'odds', etc.
        max_draw_share: Maximum fraction of bets that can be draws in FINAL result

    Returns:
        Filtered list of bets, n_dropped
    """
    if not selected_bets:
        return selected_bets, 0

    draws = [b for b in selected_bets if b['pick'] == 'draw']
    others = [b for b in selected_bets if b['pick'] != 'draw']

    n_others = len(others)
    n_draws = len(draws)

    if n_others == 0:
        # All bets are draws - just take top by edge
        max_draws = max(1, int(len(draws) * max_draw_share))
        draws_sorted = sorted(draws, key=lambda b: b['edge'], reverse=True)
        return draws_sorted[:max_draws], len(draws) - max_draws

    # Calculate max draws so that final concentration = max_draw_share
    # max_draws / (max_draws + n_others) = max_draw_share
    # max_draws = max_draw_share * (max_draws + n_others)
    # max_draws * (1 - max_draw_share) = max_draw_share * n_others
    # max_draws = (max_draw_share * n_others) / (1 - max_draw_share)
    max_draws = int((max_draw_share * n_others) / (1 - max_draw_share))

    if n_draws <= max_draws:
        return selected_bets, 0

    # Sort draws by edge descending, keep top max_draws
    draws_sorted = sorted(draws, key=lambda b: b['edge'], reverse=True)
    draws_kept = draws_sorted[:max_draws]
    draws_dropped = len(draws) - max_draws

    return draws_kept + others, draws_dropped


def run_policy(probs: np.ndarray, market_probs: np.ndarray, y_true: np.ndarray,
               odds_rows: list, edge_threshold: float = POLICY_EDGE_THRESHOLD,
               apply_cap: bool = False, max_draw_share: float = POLICY_MAX_DRAW_SHARE) -> dict:
    """
    Run betting policy and calculate ROI.

    Args:
        probs: Model probabilities (n, 3)
        market_probs: Market probabilities (n, 3)
        y_true: Actual outcomes (n,)
        odds_rows: List of dicts with odds_home, odds_draw, odds_away
        edge_threshold: Minimum edge to place bet
        apply_cap: Whether to apply draw cap
        max_draw_share: Max fraction of draws if cap enabled

    Returns:
        Dict with policy results
    """
    n = len(probs)

    # Step 1: Select all bets with edge >= threshold
    selected_bets = []

    for i in range(n):
        model_p = probs[i]
        market_p = market_probs[i]

        edges = [model_p[j] - market_p[j] for j in range(3)]
        best_pick = int(np.argmax(edges))

        if edges[best_pick] >= edge_threshold:
            odds = [float(odds_rows[i]['odds_home']),
                    float(odds_rows[i]['odds_draw']),
                    float(odds_rows[i]['odds_away'])][best_pick]
            won = 1 if y_true[i] == best_pick else 0
            returns = odds if won else 0
            pick_name = ["home", "draw", "away"][best_pick]

            selected_bets.append({
                "idx": i,
                "pick": pick_name,
                "pick_class": best_pick,
                "edge": edges[best_pick],
                "odds": odds,
                "p_model": model_p[best_pick],
                "p_market": market_p[best_pick],
                "won": won,
                "returns": returns,
            })

    # Step 2: Apply draw cap if enabled
    n_draws_dropped = 0
    if apply_cap and selected_bets:
        selected_bets, n_draws_dropped = apply_draw_cap(selected_bets, max_draw_share)

    # Step 3: Calculate metrics
    if not selected_bets:
        return {
            "n_bets": 0,
            "n_draws_dropped": n_draws_dropped,
            "concentration": {"home": 0, "draw": 0, "away": 0},
            "roi": {"total": 0, "home": 0, "draw": 0, "away": 0},
            "win_rate": {"total": 0, "home": 0, "draw": 0, "away": 0},
        }

    # Group by pick
    by_pick = defaultdict(list)
    for b in selected_bets:
        by_pick[b['pick']].append(b)

    total_staked = len(selected_bets)
    total_returns = sum(b['returns'] for b in selected_bets)
    total_won = sum(b['won'] for b in selected_bets)

    # Concentration
    concentration = {
        "home": round(100 * len(by_pick['home']) / total_staked, 2),
        "draw": round(100 * len(by_pick['draw']) / total_staked, 2),
        "away": round(100 * len(by_pick['away']) / total_staked, 2),
    }

    # ROI by pick
    roi = {"total": round(100 * (total_returns - total_staked) / total_staked, 2)}
    win_rate = {"total": round(100 * total_won / total_staked, 2)}

    for pick in ["home", "draw", "away"]:
        bets = by_pick[pick]
        if bets:
            staked = len(bets)
            returns = sum(b['returns'] for b in bets)
            won = sum(b['won'] for b in bets)
            roi[pick] = round(100 * (returns - staked) / staked, 2)
            win_rate[pick] = round(100 * won / staked, 2)
        else:
            roi[pick] = None
            win_rate[pick] = None

    # Edge stats
    edge_stats = {
        "mean": round(np.mean([b['edge'] for b in selected_bets]), 4),
        "by_pick": {}
    }
    for pick in ["home", "draw", "away"]:
        bets = by_pick[pick]
        if bets:
            edge_stats["by_pick"][pick] = {
                "mean": round(np.mean([b['edge'] for b in bets]), 4),
                "min": round(min(b['edge'] for b in bets), 4),
                "max": round(max(b['edge'] for b in bets), 4),
            }

    return {
        "n_bets": total_staked,
        "n_draws_dropped": n_draws_dropped,
        "concentration": concentration,
        "roi": roi,
        "win_rate": win_rate,
        "edge_stats": edge_stats,
        "by_pick_detail": {
            pick: {
                "n_bets": len(bets),
                "mean_odds": round(np.mean([b['odds'] for b in bets]), 3) if bets else None,
                "mean_p_model": round(np.mean([b['p_model'] for b in bets]), 4) if bets else None,
                "mean_p_market": round(np.mean([b['p_market'] for b in bets]), 4) if bets else None,
            }
            for pick, bets in by_pick.items()
        }
    }


def bootstrap_roi_ci(selected_bets: list, n_bootstrap: int = 1000, ci: float = 0.95) -> tuple:
    """Calculate bootstrap CI for ROI."""
    if not selected_bets or len(selected_bets) < 10:
        return (None, None)

    rng = np.random.default_rng(42)
    rois = []

    for _ in range(n_bootstrap):
        sample = rng.choice(selected_bets, size=len(selected_bets), replace=True)
        staked = len(sample)
        returns = sum(b['returns'] for b in sample)
        rois.append(100 * (returns - staked) / staked)

    alpha = (1 - ci) / 2
    lower = np.percentile(rois, 100 * alpha)
    upper = np.percentile(rois, 100 * (1 - alpha))

    return (round(lower, 2), round(upper, 2))


async def run_evaluation(min_snapshot_date: str):
    """Run policy evaluation with and without draw cap."""

    database_url = os.environ.get('DATABASE_URL')
    if not database_url:
        raise ValueError("DATABASE_URL required")

    if database_url.startswith('postgresql://'):
        database_url = database_url.replace('postgresql://', 'postgresql+asyncpg://')

    engine = create_async_engine(database_url)
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    results = {
        "generated_at": datetime.utcnow().isoformat(),
        "evaluation_type": "Policy Draw Cap Comparison",
        "min_snapshot_date": min_snapshot_date,
        "policy_config": {
            "edge_threshold": POLICY_EDGE_THRESHOLD,
            "max_draw_share": POLICY_MAX_DRAW_SHARE,
        },
    }

    async with async_session() as session:

        # Get valid cohort
        min_date_dt = datetime.fromisoformat(min_snapshot_date)

        query = text("""
            SELECT
                os.id as snapshot_id,
                m.id as match_id,
                CASE
                    WHEN m.home_goals > m.away_goals THEN 0
                    WHEN m.home_goals = m.away_goals THEN 1
                    ELSE 2
                END as result,
                os.odds_home,
                os.odds_draw,
                os.odds_away,
                pe_new.home_prob as new_home,
                pe_new.draw_prob as new_draw,
                pe_new.away_prob as new_away
            FROM odds_snapshots os
            JOIN matches m ON os.match_id = m.id
            JOIN predictions_experiments pe_new ON pe_new.snapshot_id = os.id
                AND pe_new.model_version = 'v1.0.1-league-only-trained'
            WHERE m.status = 'FT'
              AND m.home_goals IS NOT NULL
              AND m.away_goals IS NOT NULL
              AND os.odds_home IS NOT NULL
              AND os.odds_draw IS NOT NULL
              AND os.odds_away IS NOT NULL
              AND os.odds_freshness = 'live'
              AND EXTRACT(EPOCH FROM (m.date - os.snapshot_at)) / 60 BETWEEN 10 AND 90
              AND os.snapshot_at >= :min_date
            ORDER BY os.snapshot_at
        """)

        result = await session.execute(query, {"min_date": min_date_dt})
        rows = [dict(r._mapping) for r in result.fetchall()]

        n = len(rows)
        logger.info(f"Cohort size: n={n}")

        results["cohort"] = {"n": n}

        # Build arrays
        y_true = np.array([r['result'] for r in rows])

        new_probs = np.array([[float(r['new_home']), float(r['new_draw']), float(r['new_away'])] for r in rows])

        market_probs = np.array([
            devig_proportional(float(r['odds_home']), float(r['odds_draw']), float(r['odds_away']))
            for r in rows
        ])

        # Normalize
        new_probs = new_probs / new_probs.sum(axis=1, keepdims=True)

        # ═══════════════════════════════════════════════════════════════
        # POLICY COMPARISON
        # ═══════════════════════════════════════════════════════════════

        logger.info("Running policy WITHOUT draw cap...")
        policy_no_cap = run_policy(
            new_probs, market_probs, y_true, rows,
            edge_threshold=POLICY_EDGE_THRESHOLD,
            apply_cap=False
        )

        logger.info("Running policy WITH draw cap...")
        policy_with_cap = run_policy(
            new_probs, market_probs, y_true, rows,
            edge_threshold=POLICY_EDGE_THRESHOLD,
            apply_cap=True,
            max_draw_share=POLICY_MAX_DRAW_SHARE
        )

        # Bootstrap CI for ROI
        # Re-run to get bet lists for bootstrap
        def get_bets_for_bootstrap(probs, market_probs, y_true, odds_rows, apply_cap, max_draw_share):
            selected_bets = []
            for i in range(len(probs)):
                model_p = probs[i]
                market_p = market_probs[i]
                edges = [model_p[j] - market_p[j] for j in range(3)]
                best_pick = int(np.argmax(edges))
                if edges[best_pick] >= POLICY_EDGE_THRESHOLD:
                    odds = [float(odds_rows[i]['odds_home']),
                            float(odds_rows[i]['odds_draw']),
                            float(odds_rows[i]['odds_away'])][best_pick]
                    won = 1 if y_true[i] == best_pick else 0
                    returns = odds if won else 0
                    pick_name = ["home", "draw", "away"][best_pick]
                    selected_bets.append({
                        "pick": pick_name,
                        "edge": edges[best_pick],
                        "odds": odds,
                        "won": won,
                        "returns": returns,
                    })
            if apply_cap:
                selected_bets, _ = apply_draw_cap(selected_bets, max_draw_share)
            return selected_bets

        bets_no_cap = get_bets_for_bootstrap(new_probs, market_probs, y_true, rows, False, 0)
        bets_with_cap = get_bets_for_bootstrap(new_probs, market_probs, y_true, rows, True, POLICY_MAX_DRAW_SHARE)

        ci_no_cap = bootstrap_roi_ci(bets_no_cap)
        ci_with_cap = bootstrap_roi_ci(bets_with_cap)

        policy_no_cap["roi_ci95"] = ci_no_cap
        policy_with_cap["roi_ci95"] = ci_with_cap

        results["policy_comparison"] = {
            "without_cap": policy_no_cap,
            "with_cap": policy_with_cap,
        }

        # ═══════════════════════════════════════════════════════════════
        # DELTA ANALYSIS
        # ═══════════════════════════════════════════════════════════════

        delta = {
            "n_bets_change": policy_with_cap["n_bets"] - policy_no_cap["n_bets"],
            "n_draws_dropped": policy_with_cap["n_draws_dropped"],
            "concentration_draw_change": policy_with_cap["concentration"]["draw"] - policy_no_cap["concentration"]["draw"],
            "roi_total_change": policy_with_cap["roi"]["total"] - policy_no_cap["roi"]["total"],
        }

        results["delta"] = delta

        # ═══════════════════════════════════════════════════════════════
        # VERDICT
        # ═══════════════════════════════════════════════════════════════

        draw_share_ok = policy_with_cap["concentration"]["draw"] <= 35
        roi_not_worse = policy_with_cap["roi"]["total"] >= policy_no_cap["roi"]["total"] - 5  # Allow 5% margin

        verdict = {
            "draw_share_reduced": draw_share_ok,
            "roi_acceptable": roi_not_worse,
            "recommendation": "GO" if (draw_share_ok and roi_not_worse) else "REVIEW"
        }

        results["verdict"] = verdict

    await engine.dispose()
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate policy with draw cap")
    parser.add_argument('--min-snapshot-date', required=True, help='Min snapshot date (ISO)')
    args = parser.parse_args()

    results = asyncio.run(run_evaluation(args.min_snapshot_date))

    # Save JSON
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"logs/policy_draw_cap_eval_{timestamp}.json"

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n{'='*70}")
    print(f"POLICY EVALUATION COMPLETE: {output_path}")
    print(f"{'='*70}")

    # Print comparison
    print("\n--- POLICY COMPARISON ---\n")

    no_cap = results["policy_comparison"]["without_cap"]
    with_cap = results["policy_comparison"]["with_cap"]

    print(f"{'Metric':<25} {'Sin Cap':>15} {'Con Cap (35%)':>15} {'Delta':>10}")
    print("-" * 70)

    print(f"{'n_bets':<25} {no_cap['n_bets']:>15} {with_cap['n_bets']:>15} {with_cap['n_bets'] - no_cap['n_bets']:>+10}")
    print(f"{'n_draws_dropped':<25} {'-':>15} {with_cap['n_draws_dropped']:>15} {'-':>10}")
    print()

    print("Concentration (%):")
    for pick in ["home", "draw", "away"]:
        nc = no_cap['concentration'][pick]
        wc = with_cap['concentration'][pick]
        print(f"  {pick:<23} {nc:>15.1f} {wc:>15.1f} {wc - nc:>+10.1f}")
    print()

    print("ROI (%):")
    print(f"  {'total':<23} {no_cap['roi']['total']:>15.2f} {with_cap['roi']['total']:>15.2f} {with_cap['roi']['total'] - no_cap['roi']['total']:>+10.2f}")
    for pick in ["home", "draw", "away"]:
        nc = no_cap['roi'].get(pick)
        wc = with_cap['roi'].get(pick)
        if nc is not None and wc is not None:
            print(f"  {pick:<23} {nc:>15.2f} {wc:>15.2f} {wc - nc:>+10.2f}")
    print()

    print("Win Rate (%):")
    print(f"  {'total':<23} {no_cap['win_rate']['total']:>15.2f} {with_cap['win_rate']['total']:>15.2f}")
    print()

    print("ROI CI95:")
    print(f"  {'sin cap':<23} [{no_cap['roi_ci95'][0]}, {no_cap['roi_ci95'][1]}]")
    print(f"  {'con cap':<23} [{with_cap['roi_ci95'][0]}, {with_cap['roi_ci95'][1]}]")
    print()

    print("--- VERDICT ---")
    v = results["verdict"]
    print(f"  Draw share reduced: {v['draw_share_reduced']}")
    print(f"  ROI acceptable: {v['roi_acceptable']}")
    print(f"  Recommendation: {v['recommendation']}")

    return output_path


if __name__ == "__main__":
    main()
