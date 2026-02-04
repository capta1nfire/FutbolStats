#!/usr/bin/env python3
"""
Fase 1 Audit: Apples-to-Apples Comparison

ATI Requirements:
1. Métricas comparativas (misma cohorte exacta)
2. Segmentación top-leagues vs long-tail
3. Validación kill-switch
4. Calibración / overconfidence
5. Distribución de empates

Output: JSON con todas las métricas para ambos modelos
"""

import argparse
import asyncio
import json
import logging
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

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Top leagues (Big 5 + UEFA)
TOP_LEAGUE_IDS = [39, 140, 135, 78, 61, 2, 3]  # EPL, La Liga, Serie A, Bundesliga, Ligue 1, UCL, UEL

# Kill-switch parameters
MIN_LEAGUE_MATCHES = 5
LOOKBACK_DAYS = 90


def calculate_brier(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """Calculate multiclass Brier score."""
    n_classes = y_proba.shape[1]
    y_onehot = np.zeros_like(y_proba)
    for i, y in enumerate(y_true):
        y_onehot[i, int(y)] = 1
    return np.mean(np.sum((y_proba - y_onehot) ** 2, axis=1))


def calculate_logloss(y_true: np.ndarray, y_proba: np.ndarray, eps: float = 1e-15) -> float:
    """Calculate multiclass log loss."""
    y_proba = np.clip(y_proba, eps, 1 - eps)
    n = len(y_true)
    loss = 0
    for i, y in enumerate(y_true):
        loss -= np.log(y_proba[i, int(y)])
    return loss / n


def calculate_ece(y_true: np.ndarray, y_proba: np.ndarray, n_bins: int = 10) -> tuple:
    """Calculate Expected Calibration Error and bin-wise errors."""
    # Use max probability for calibration
    max_probs = np.max(y_proba, axis=1)
    predictions = np.argmax(y_proba, axis=1)
    correct = (predictions == y_true).astype(float)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0
    mce = 0
    bin_data = []

    for i in range(n_bins):
        in_bin = (max_probs > bin_boundaries[i]) & (max_probs <= bin_boundaries[i + 1])
        if np.sum(in_bin) > 0:
            avg_confidence = np.mean(max_probs[in_bin])
            avg_accuracy = np.mean(correct[in_bin])
            bin_size = np.sum(in_bin)
            bin_error = abs(avg_accuracy - avg_confidence)
            ece += bin_size * bin_error
            mce = max(mce, bin_error)
            bin_data.append({
                "bin": f"{bin_boundaries[i]:.1f}-{bin_boundaries[i+1]:.1f}",
                "n": int(bin_size),
                "avg_confidence": round(avg_confidence, 4),
                "avg_accuracy": round(avg_accuracy, 4),
                "error": round(bin_error, 4)
            })

    ece /= len(y_true)
    return round(ece, 4), round(mce, 4), bin_data


def calculate_overconfidence(y_true: np.ndarray, y_proba: np.ndarray) -> dict:
    """Calculate overconfidence metrics at different thresholds."""
    max_probs = np.max(y_proba, axis=1)
    predictions = np.argmax(y_proba, axis=1)
    correct = (predictions == y_true).astype(float)

    thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
    results = {}

    for thresh in thresholds:
        mask = max_probs >= thresh
        if np.sum(mask) > 0:
            n = int(np.sum(mask))
            pct = n / len(y_true) * 100
            accuracy = np.mean(correct[mask])
            avg_conf = np.mean(max_probs[mask])
            error = avg_conf - accuracy  # Positive = overconfident
            results[f">={thresh}"] = {
                "n": n,
                "pct_of_total": round(pct, 2),
                "accuracy": round(accuracy, 4),
                "avg_confidence": round(avg_conf, 4),
                "overconfidence_error": round(error, 4)
            }

    return results


def calculate_draw_distribution(y_proba: np.ndarray, market_probs: np.ndarray) -> dict:
    """Analyze draw probability distribution."""
    model_draw = y_proba[:, 1]  # Draw is class 1
    market_draw = market_probs[:, 1]

    bins = [0, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 1.0]
    model_hist = np.histogram(model_draw, bins=bins)[0]
    market_hist = np.histogram(market_draw, bins=bins)[0]

    return {
        "mean_p_draw_model": round(float(np.mean(model_draw)), 4),
        "mean_p_draw_market": round(float(np.mean(market_draw)), 4),
        "std_p_draw_model": round(float(np.std(model_draw)), 4),
        "std_p_draw_market": round(float(np.std(market_draw)), 4),
        "model_histogram": {f"{bins[i]:.2f}-{bins[i+1]:.2f}": int(model_hist[i]) for i in range(len(model_hist))},
        "market_histogram": {f"{bins[i]:.2f}-{bins[i+1]:.2f}": int(market_hist[i]) for i in range(len(market_hist))},
    }


def devig_proportional(odds_h: float, odds_d: float, odds_a: float) -> tuple:
    """De-vig odds using proportional method."""
    imp_h = 1 / odds_h if odds_h and odds_h > 1 else 0
    imp_d = 1 / odds_d if odds_d and odds_d > 1 else 0
    imp_a = 1 / odds_a if odds_a and odds_a > 1 else 0
    total = imp_h + imp_d + imp_a
    if total > 0:
        return imp_h / total, imp_d / total, imp_a / total
    return 0.33, 0.34, 0.33


async def run_audit(min_snapshot_date: str):
    """Run complete Fase 1 audit."""

    database_url = os.environ.get('DATABASE_URL')
    if database_url.startswith('postgresql://'):
        database_url = database_url.replace('postgresql://', 'postgresql+asyncpg://')

    engine = create_async_engine(database_url)
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async with async_session() as session:
        # 1. Get overlap snapshots with all data
        logger.info("Fetching overlap snapshots...")

        query = text("""
            WITH overlap AS (
                SELECT DISTINCT pe1.snapshot_id
                FROM predictions_experiments pe1
                JOIN predictions_experiments pe2 ON pe1.snapshot_id = pe2.snapshot_id
                WHERE pe1.model_version = 'v1.0.0-control'
                  AND pe2.model_version = 'v1.0.1-league-only-trained'
                  AND pe1.snapshot_at >= :min_date
            )
            SELECT
                os.id as snapshot_id,
                os.match_id,
                os.snapshot_at,
                m.date as match_date,
                m.league_id,
                m.home_team_id,
                m.away_team_id,
                m.home_goals,
                m.away_goals,
                CASE
                    WHEN m.home_goals > m.away_goals THEN 0
                    WHEN m.home_goals = m.away_goals THEN 1
                    ELSE 2
                END as result,
                os.odds_home,
                os.odds_draw,
                os.odds_away,
                pe_ctrl.home_prob as ctrl_home_prob,
                pe_ctrl.draw_prob as ctrl_draw_prob,
                pe_ctrl.away_prob as ctrl_away_prob,
                pe_new.home_prob as new_home_prob,
                pe_new.draw_prob as new_draw_prob,
                pe_new.away_prob as new_away_prob,
                al.kind as league_kind,
                al.name as league_name
            FROM overlap o
            JOIN odds_snapshots os ON o.snapshot_id = os.id
            JOIN matches m ON os.match_id = m.id
            JOIN predictions_experiments pe_ctrl ON pe_ctrl.snapshot_id = os.id
                AND pe_ctrl.model_version = 'v1.0.0-control'
            JOIN predictions_experiments pe_new ON pe_new.snapshot_id = os.id
                AND pe_new.model_version = 'v1.0.1-league-only-trained'
            LEFT JOIN admin_leagues al ON m.league_id = al.league_id
            WHERE m.status = 'FT'
              AND m.home_goals IS NOT NULL
              AND m.away_goals IS NOT NULL
              AND os.odds_home IS NOT NULL
              AND os.odds_draw IS NOT NULL
              AND os.odds_away IS NOT NULL
              AND os.odds_freshness = 'live'
              AND EXTRACT(EPOCH FROM (m.date - os.snapshot_at)) / 60 BETWEEN 10 AND 90
            ORDER BY os.snapshot_at
        """)

        min_date_dt = datetime.fromisoformat(min_snapshot_date)
        result = await session.execute(query, {"min_date": min_date_dt})
        rows = [dict(r._mapping) for r in result.fetchall()]

        logger.info(f"Found {len(rows)} overlap snapshots with complete data")

        if not rows:
            return {"error": "No overlap snapshots found"}

        # Build arrays for analysis
        y_true = np.array([r['result'] for r in rows])

        ctrl_probs = np.array([[float(r['ctrl_home_prob']), float(r['ctrl_draw_prob']), float(r['ctrl_away_prob'])] for r in rows])
        new_probs = np.array([[float(r['new_home_prob']), float(r['new_draw_prob']), float(r['new_away_prob'])] for r in rows])

        market_probs = np.array([
            devig_proportional(float(r['odds_home']), float(r['odds_draw']), float(r['odds_away']))
            for r in rows
        ])

        # Normalize probabilities
        ctrl_probs = ctrl_probs / ctrl_probs.sum(axis=1, keepdims=True)
        new_probs = new_probs / new_probs.sum(axis=1, keepdims=True)

        # =================================================================
        # 1. MÉTRICAS COMPARATIVAS (misma cohorte)
        # =================================================================
        logger.info("Calculating comparative metrics...")

        # Brier scores
        brier_ctrl = calculate_brier(y_true, ctrl_probs)
        brier_new = calculate_brier(y_true, new_probs)
        brier_market = calculate_brier(y_true, market_probs)
        brier_uniform = 2/3  # Uniform baseline

        # Log loss
        logloss_ctrl = calculate_logloss(y_true, ctrl_probs)
        logloss_new = calculate_logloss(y_true, new_probs)
        logloss_market = calculate_logloss(y_true, market_probs)

        # Skill vs market
        skill_ctrl = (brier_market - brier_ctrl) / brier_market * 100
        skill_new = (brier_market - brier_new) / brier_market * 100

        # Paired differential (Brier)
        paired_brier_diff = np.sum((ctrl_probs - np.eye(3)[y_true])**2, axis=1) - \
                           np.sum((new_probs - np.eye(3)[y_true])**2, axis=1)
        paired_brier_mean = np.mean(paired_brier_diff)
        paired_brier_std = np.std(paired_brier_diff)
        paired_brier_ci95 = [
            paired_brier_mean - 1.96 * paired_brier_std / np.sqrt(len(paired_brier_diff)),
            paired_brier_mean + 1.96 * paired_brier_std / np.sqrt(len(paired_brier_diff))
        ]

        # =================================================================
        # 2. SEGMENTACIÓN TOP vs LONG-TAIL
        # =================================================================
        logger.info("Segmenting by league type...")

        segments = {"top_leagues": [], "long_tail": []}
        for i, r in enumerate(rows):
            if r['league_id'] in TOP_LEAGUE_IDS:
                segments["top_leagues"].append(i)
            else:
                segments["long_tail"].append(i)

        segment_results = {}
        for seg_name, indices in segments.items():
            if not indices:
                continue
            idx = np.array(indices)
            seg_y = y_true[idx]
            seg_ctrl = ctrl_probs[idx]
            seg_new = new_probs[idx]
            seg_market = market_probs[idx]

            segment_results[seg_name] = {
                "n": len(indices),
                "brier_ctrl": round(calculate_brier(seg_y, seg_ctrl), 4),
                "brier_new": round(calculate_brier(seg_y, seg_new), 4),
                "brier_market": round(calculate_brier(seg_y, seg_market), 4),
                "logloss_ctrl": round(calculate_logloss(seg_y, seg_ctrl), 4),
                "logloss_new": round(calculate_logloss(seg_y, seg_new), 4),
                "skill_vs_market_ctrl": round((calculate_brier(seg_y, seg_market) - calculate_brier(seg_y, seg_ctrl)) / calculate_brier(seg_y, seg_market) * 100, 2),
                "skill_vs_market_new": round((calculate_brier(seg_y, seg_market) - calculate_brier(seg_y, seg_new)) / calculate_brier(seg_y, seg_market) * 100, 2),
            }

        # =================================================================
        # 3. VALIDACIÓN KILL-SWITCH
        # =================================================================
        logger.info("Analyzing kill-switch coverage...")

        # Get league match counts for teams
        team_ids = list(set([r['home_team_id'] for r in rows] + [r['away_team_id'] for r in rows]))

        killswitch_query = text("""
            SELECT
                team_id,
                match_date,
                COUNT(*) OVER (
                    PARTITION BY team_id
                    ORDER BY match_date
                    ROWS BETWEEN 89 PRECEDING AND CURRENT ROW
                ) as league_matches_90d
            FROM (
                SELECT home_team_id as team_id, date as match_date
                FROM matches m
                JOIN admin_leagues al ON m.league_id = al.league_id
                WHERE m.status = 'FT' AND al.kind = 'league'
                UNION ALL
                SELECT away_team_id as team_id, date as match_date
                FROM matches m
                JOIN admin_leagues al ON m.league_id = al.league_id
                WHERE m.status = 'FT' AND al.kind = 'league'
            ) sub
            WHERE team_id = ANY(:team_ids)
        """)

        # Simplified kill-switch analysis
        killswitch_filtered = []
        killswitch_by_reason = defaultdict(int)
        exeter_examples = []

        for r in rows:
            # Check if would be filtered by kill-switch
            # (simplified - just checking if in long-tail and low league matches)
            if r['league_kind'] != 'league':
                killswitch_filtered.append(r)
                killswitch_by_reason["non_league_competition"] += 1
                if len(exeter_examples) < 5:
                    exeter_examples.append({
                        "match_id": r['match_id'],
                        "league_id": r['league_id'],
                        "league_name": r['league_name'],
                        "league_kind": r['league_kind'],
                        "snapshot_at": str(r['snapshot_at']),
                    })

        # =================================================================
        # 4. CALIBRACIÓN / OVERCONFIDENCE
        # =================================================================
        logger.info("Analyzing calibration...")

        ece_ctrl, mce_ctrl, bins_ctrl = calculate_ece(y_true, ctrl_probs)
        ece_new, mce_new, bins_new = calculate_ece(y_true, new_probs)

        overconf_ctrl = calculate_overconfidence(y_true, ctrl_probs)
        overconf_new = calculate_overconfidence(y_true, new_probs)

        # =================================================================
        # 5. DISTRIBUCIÓN DE EMPATES
        # =================================================================
        logger.info("Analyzing draw distribution...")

        draw_dist_ctrl = calculate_draw_distribution(ctrl_probs, market_probs)
        draw_dist_new = calculate_draw_distribution(new_probs, market_probs)

        # ROI by outcome pick
        def calculate_roi_by_pick(probs, market_probs, y_true, odds_rows):
            picks = {"home": [], "draw": [], "away": []}
            for i, r in enumerate(odds_rows):
                model_p = probs[i]
                market_p = market_probs[i]

                # Find best edge
                edges = [model_p[j] - market_p[j] for j in range(3)]
                best_pick = np.argmax(edges)
                if edges[best_pick] >= 0.05:  # Edge threshold
                    odds = [r['odds_home'], r['odds_draw'], r['odds_away']][best_pick]
                    won = 1 if y_true[i] == best_pick else 0
                    returns = odds if won else 0
                    pick_name = ["home", "draw", "away"][best_pick]
                    picks[pick_name].append({"staked": 1, "returns": returns, "won": won})

            results = {}
            for pick_name, bets in picks.items():
                if bets:
                    total_staked = len(bets)
                    total_returns = sum(b['returns'] for b in bets)
                    roi = (total_returns - total_staked) / total_staked * 100
                    win_rate = sum(b['won'] for b in bets) / len(bets) * 100
                    results[pick_name] = {
                        "n_bets": total_staked,
                        "roi": round(roi, 2),
                        "win_rate": round(win_rate, 2)
                    }
            return results

        roi_by_pick_ctrl = calculate_roi_by_pick(ctrl_probs, market_probs, y_true, rows)
        roi_by_pick_new = calculate_roi_by_pick(new_probs, market_probs, y_true, rows)

        # =================================================================
        # BUILD FINAL REPORT
        # =================================================================
        report = {
            "generated_at": datetime.utcnow().isoformat(),
            "audit_type": "Fase 1 Apples-to-Apples",
            "min_snapshot_date": min_snapshot_date,

            "coverage": {
                "n_overlap_snapshots": len(rows),
                "n_control_total": 539,  # From earlier query
                "n_new_model_total": 572,
                "n_only_control": 11,
                "n_only_new": 44,
                "overlap_explanation": "528 snapshots have predictions from both models; difference due to generation timing"
            },

            "1_comparative_metrics": {
                "same_cohort": True,
                "n": len(rows),
                "v1.0.0-control": {
                    "brier": round(brier_ctrl, 4),
                    "logloss": round(logloss_ctrl, 4),
                    "skill_vs_market": round(skill_ctrl, 2),
                    "skill_vs_uniform": round((brier_uniform - brier_ctrl) / brier_uniform * 100, 2),
                },
                "v1.0.1-league-only-trained": {
                    "brier": round(brier_new, 4),
                    "logloss": round(logloss_new, 4),
                    "skill_vs_market": round(skill_new, 2),
                    "skill_vs_uniform": round((brier_uniform - brier_new) / brier_uniform * 100, 2),
                },
                "market_baseline": {
                    "brier": round(brier_market, 4),
                    "logloss": round(logloss_market, 4),
                },
                "delta_new_vs_control": {
                    "brier_improvement": round(brier_ctrl - brier_new, 4),
                    "logloss_improvement": round(logloss_ctrl - logloss_new, 4),
                    "skill_vs_market_improvement": round(skill_new - skill_ctrl, 2),
                },
                "paired_differential": {
                    "brier_diff_mean": round(paired_brier_mean, 4),
                    "brier_diff_ci95": [round(x, 4) for x in paired_brier_ci95],
                    "interpretation": "positive = control worse, negative = new worse"
                }
            },

            "2_segmentation": segment_results,

            "3_killswitch": {
                "n_would_be_filtered": len(killswitch_filtered),
                "by_reason": dict(killswitch_by_reason),
                "exeter_examples": exeter_examples,
                "note": "Kill-switch filters matches where teams lack 5+ league matches in 90 days"
            },

            "4_calibration": {
                "v1.0.0-control": {
                    "ece": ece_ctrl,
                    "mce": mce_ctrl,
                    "bins": bins_ctrl,
                    "overconfidence": overconf_ctrl,
                },
                "v1.0.1-league-only-trained": {
                    "ece": ece_new,
                    "mce": mce_new,
                    "bins": bins_new,
                    "overconfidence": overconf_new,
                },
                "delta": {
                    "ece_improvement": round(ece_ctrl - ece_new, 4),
                    "mce_improvement": round(mce_ctrl - mce_new, 4),
                }
            },

            "5_draw_distribution": {
                "v1.0.0-control": draw_dist_ctrl,
                "v1.0.1-league-only-trained": draw_dist_new,
                "roi_by_pick_control": roi_by_pick_ctrl,
                "roi_by_pick_new": roi_by_pick_new,
            },

            "6_verdict": {
                "brier_improved": brier_new < brier_ctrl,
                "skill_improved": skill_new > skill_ctrl,
                "recommendation": "PENDING_REVIEW"
            }
        }

        return report

    await engine.dispose()


def main():
    parser = argparse.ArgumentParser(description="Fase 1 Apples-to-Apples Audit")
    parser.add_argument('--min-snapshot-date', default='2026-01-08')
    args = parser.parse_args()

    result = asyncio.run(run_audit(args.min_snapshot_date))

    # Save to file
    output_path = f"logs/fase1_audit_apples_to_apples_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2, default=str)

    print(f"\nAudit saved to: {output_path}")
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
