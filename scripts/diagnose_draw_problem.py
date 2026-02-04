#!/usr/bin/env python3
"""
Diagnóstico completo del "draw problem" - Fase 1

Separar modelo vs policy para entender:
1. Mismatch 528 vs 501 (tabla de exclusiones)
2. Distribución argmax picks (modelo puro)
3. Métricas por clase (Brier/LogLoss por outcome)
4. Policy EV (edge distribution y regla de selección)
5. Kill-switch razones separadas

ATI audit request - 2026-02-01
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


def devig_proportional(odds_home: float, odds_draw: float, odds_away: float) -> list[float]:
    """Convert odds to fair probabilities using proportional devig."""
    implied = [1/odds_home, 1/odds_draw, 1/odds_away]
    total = sum(implied)
    return [p/total for p in implied]


def brier_score(y_true: np.ndarray, probs: np.ndarray) -> float:
    """Calculate Brier score for multiclass."""
    n = len(y_true)
    score = 0.0
    for i in range(n):
        for c in range(3):
            target = 1.0 if y_true[i] == c else 0.0
            score += (probs[i, c] - target) ** 2
    return score / n


def logloss(y_true: np.ndarray, probs: np.ndarray) -> float:
    """Calculate log loss."""
    eps = 1e-15
    n = len(y_true)
    score = 0.0
    for i in range(n):
        p = max(eps, min(1-eps, probs[i, y_true[i]]))
        score -= np.log(p)
    return score / n


async def run_diagnosis(min_snapshot_date: str):
    """Run full diagnosis."""

    database_url = os.environ.get('DATABASE_URL')
    if not database_url:
        raise ValueError("DATABASE_URL required")

    if database_url.startswith('postgresql://'):
        database_url = database_url.replace('postgresql://', 'postgresql+asyncpg://')

    engine = create_async_engine(database_url)
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    results = {
        "generated_at": datetime.utcnow().isoformat(),
        "diagnosis_type": "Draw Problem Deep Dive",
        "min_snapshot_date": min_snapshot_date,
    }

    async with async_session() as session:

        # =================================================================
        # SECTION 1: EXCLUSION ANALYSIS (528 vs 501)
        # =================================================================
        logger.info("=== Section 1: Exclusion Analysis ===")

        min_date_dt = datetime.fromisoformat(min_snapshot_date)

        # Step 1a: All snapshots with BOTH predictions
        both_preds_query = text("""
            SELECT
                os.id as snapshot_id,
                os.match_id,
                os.snapshot_at,
                m.status,
                m.home_goals,
                m.away_goals,
                os.odds_home,
                os.odds_draw,
                os.odds_away,
                os.odds_freshness,
                EXTRACT(EPOCH FROM (m.date - os.snapshot_at)) / 60 as minutes_before_kickoff
            FROM odds_snapshots os
            JOIN matches m ON os.match_id = m.id
            JOIN predictions_experiments pe_ctrl ON pe_ctrl.snapshot_id = os.id
                AND pe_ctrl.model_version = 'v1.0.0-control'
            JOIN predictions_experiments pe_new ON pe_new.snapshot_id = os.id
                AND pe_new.model_version = 'v1.0.1-league-only-trained'
            WHERE os.snapshot_at >= :min_date
            ORDER BY os.snapshot_at
        """)

        result = await session.execute(both_preds_query, {"min_date": min_date_dt})
        all_overlap = [dict(r._mapping) for r in result.fetchall()]

        logger.info(f"Total snapshots with BOTH predictions: {len(all_overlap)}")

        # Categorize exclusions
        exclusions = {
            "total_with_both_predictions": len(all_overlap),
            "exclusion_reasons": defaultdict(int),
            "final_valid": 0,
        }

        valid_snapshots = []
        for row in all_overlap:
            reasons = []

            # Check status
            if row['status'] != 'FT':
                reasons.append("match_not_finished")

            # Check goals
            if row['home_goals'] is None or row['away_goals'] is None:
                reasons.append("missing_goals")

            # Check odds
            if row['odds_home'] is None or row['odds_draw'] is None or row['odds_away'] is None:
                reasons.append("missing_odds")

            # Check freshness
            if row['odds_freshness'] != 'live':
                reasons.append("odds_not_live")

            # Check timing window (10-90 min before kickoff)
            mins = row['minutes_before_kickoff']
            if mins is None or mins < 10 or mins > 90:
                reasons.append("timing_window_invalid")

            if reasons:
                for r in reasons:
                    exclusions["exclusion_reasons"][r] += 1
            else:
                valid_snapshots.append(row['snapshot_id'])

        exclusions["final_valid"] = len(valid_snapshots)
        exclusions["exclusion_reasons"] = dict(exclusions["exclusion_reasons"])

        results["1_exclusion_analysis"] = exclusions
        logger.info(f"Final valid snapshots: {len(valid_snapshots)}")
        logger.info(f"Exclusion breakdown: {exclusions['exclusion_reasons']}")

        # =================================================================
        # SECTION 2: MODEL ARGMAX DISTRIBUTION (pure model, NOT policy)
        # =================================================================
        logger.info("=== Section 2: Model Argmax Distribution ===")

        # Get valid cohort with all data
        valid_query = text("""
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
                pe_ctrl.home_prob as ctrl_home,
                pe_ctrl.draw_prob as ctrl_draw,
                pe_ctrl.away_prob as ctrl_away,
                pe_new.home_prob as new_home,
                pe_new.draw_prob as new_draw,
                pe_new.away_prob as new_away
            FROM odds_snapshots os
            JOIN matches m ON os.match_id = m.id
            JOIN predictions_experiments pe_ctrl ON pe_ctrl.snapshot_id = os.id
                AND pe_ctrl.model_version = 'v1.0.0-control'
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

        result = await session.execute(valid_query, {"min_date": min_date_dt})
        rows = [dict(r._mapping) for r in result.fetchall()]

        n = len(rows)
        logger.info(f"Valid cohort for analysis: n={n}")

        # Build arrays
        y_true = np.array([r['result'] for r in rows])

        ctrl_probs = np.array([[float(r['ctrl_home']), float(r['ctrl_draw']), float(r['ctrl_away'])] for r in rows])
        new_probs = np.array([[float(r['new_home']), float(r['new_draw']), float(r['new_away'])] for r in rows])

        market_probs = np.array([
            devig_proportional(float(r['odds_home']), float(r['odds_draw']), float(r['odds_away']))
            for r in rows
        ])

        # Normalize
        ctrl_probs = ctrl_probs / ctrl_probs.sum(axis=1, keepdims=True)
        new_probs = new_probs / new_probs.sum(axis=1, keepdims=True)

        # Argmax distribution (MODEL prediction, not betting policy)
        ctrl_argmax = np.argmax(ctrl_probs, axis=1)
        new_argmax = np.argmax(new_probs, axis=1)
        market_argmax = np.argmax(market_probs, axis=1)

        def argmax_dist(arr):
            return {
                "pct_home": round(100 * np.mean(arr == 0), 2),
                "pct_draw": round(100 * np.mean(arr == 1), 2),
                "pct_away": round(100 * np.mean(arr == 2), 2),
                "n_home": int(np.sum(arr == 0)),
                "n_draw": int(np.sum(arr == 1)),
                "n_away": int(np.sum(arr == 2)),
            }

        # p_draw statistics
        def draw_stats(probs):
            p_draw = probs[:, 1]
            return {
                "mean": round(float(np.mean(p_draw)), 4),
                "std": round(float(np.std(p_draw)), 4),
                "p25": round(float(np.percentile(p_draw, 25)), 4),
                "p50": round(float(np.percentile(p_draw, 50)), 4),
                "p75": round(float(np.percentile(p_draw, 75)), 4),
                "p90": round(float(np.percentile(p_draw, 90)), 4),
            }

        # Histogram of p_draw
        def draw_histogram(probs):
            p_draw = probs[:, 1]
            bins = [(0.00, 0.20), (0.20, 0.25), (0.25, 0.30), (0.30, 0.35), (0.35, 0.40), (0.40, 0.50), (0.50, 1.00)]
            hist = {}
            for lo, hi in bins:
                count = int(np.sum((p_draw >= lo) & (p_draw < hi)))
                hist[f"{lo:.2f}-{hi:.2f}"] = count
            return hist

        # High/low confidence draw picks (ATI request)
        def draw_confidence_breakdown(probs, argmax_arr):
            p_draw = probs[:, 1]
            is_draw_pick = (argmax_arr == 1)

            high_conf = is_draw_pick & (p_draw > 0.40)
            low_conf = is_draw_pick & (p_draw < 0.35)

            n_draw_picks = int(np.sum(is_draw_pick))
            return {
                "n_draw_picks": n_draw_picks,
                "n_high_conf_draw_gt_40pct": int(np.sum(high_conf)),
                "pct_high_conf": round(100 * np.sum(high_conf) / max(1, n_draw_picks), 2),
                "n_low_conf_draw_lt_35pct": int(np.sum(low_conf)),
                "pct_low_conf": round(100 * np.sum(low_conf) / max(1, n_draw_picks), 2),
            }

        results["2_argmax_distribution"] = {
            "n": n,
            "v1.0.0-control": {
                "argmax_picks": argmax_dist(ctrl_argmax),
                "p_draw_stats": draw_stats(ctrl_probs),
                "p_draw_histogram": draw_histogram(ctrl_probs),
                "draw_confidence": draw_confidence_breakdown(ctrl_probs, ctrl_argmax),
            },
            "v1.0.1-league-only": {
                "argmax_picks": argmax_dist(new_argmax),
                "p_draw_stats": draw_stats(new_probs),
                "p_draw_histogram": draw_histogram(new_probs),
                "draw_confidence": draw_confidence_breakdown(new_probs, new_argmax),
            },
            "market": {
                "argmax_picks": argmax_dist(market_argmax),
                "p_draw_stats": draw_stats(market_probs),
                "p_draw_histogram": draw_histogram(market_probs),
            },
            "actual_outcomes": {
                "n_home": int(np.sum(y_true == 0)),
                "n_draw": int(np.sum(y_true == 1)),
                "n_away": int(np.sum(y_true == 2)),
                "pct_home": round(100 * np.mean(y_true == 0), 2),
                "pct_draw": round(100 * np.mean(y_true == 1), 2),
                "pct_away": round(100 * np.mean(y_true == 2), 2),
            }
        }

        # =================================================================
        # SECTION 3: METRICS BY CLASS (Brier/LogLoss per outcome)
        # =================================================================
        logger.info("=== Section 3: Metrics by Class ===")

        def metrics_by_class(y_true, probs, name):
            """Calculate Brier and LogLoss per class."""
            eps = 1e-15
            n = len(y_true)

            class_metrics = {}
            for c, label in enumerate(["home", "draw", "away"]):
                mask = (y_true == c)
                n_class = int(np.sum(mask))

                if n_class == 0:
                    class_metrics[label] = {"n": 0, "brier": None, "logloss": None}
                    continue

                # Brier for this class: (p_c - 1)^2 when true, (p_c - 0)^2 when false
                # Average over samples where outcome=c
                brier_class = float(np.mean((probs[mask, c] - 1) ** 2))

                # LogLoss for this class
                logloss_class = float(-np.mean(np.log(np.clip(probs[mask, c], eps, 1-eps))))

                # Accuracy: when model predicts this class, is it right?
                pred_is_c = (np.argmax(probs, axis=1) == c)
                if np.sum(pred_is_c) > 0:
                    acc_when_pred = float(np.mean(y_true[pred_is_c] == c))
                else:
                    acc_when_pred = None

                class_metrics[label] = {
                    "n_actual": n_class,
                    "n_predicted": int(np.sum(pred_is_c)),
                    "brier": round(brier_class, 4),
                    "logloss": round(logloss_class, 4),
                    "accuracy_when_predicted": round(acc_when_pred, 4) if acc_when_pred else None,
                    "mean_prob_when_actual": round(float(np.mean(probs[mask, c])), 4),
                }

            # Total metrics
            total_brier = brier_score(y_true, probs)
            total_logloss = logloss(y_true, probs)

            return {
                "by_class": class_metrics,
                "total_brier": round(total_brier, 4),
                "total_logloss": round(total_logloss, 4),
            }

        results["3_metrics_by_class"] = {
            "v1.0.0-control": metrics_by_class(y_true, ctrl_probs, "control"),
            "v1.0.1-league-only": metrics_by_class(y_true, new_probs, "new"),
            "market": metrics_by_class(y_true, market_probs, "market"),
        }

        # Confusion-style breakdown
        def confusion_breakdown(y_true, probs, name):
            """Simple confusion: predicted vs actual."""
            pred = np.argmax(probs, axis=1)
            labels = ["home", "draw", "away"]

            confusion = {}
            for pred_c in range(3):
                for actual_c in range(3):
                    key = f"pred_{labels[pred_c]}_actual_{labels[actual_c]}"
                    confusion[key] = int(np.sum((pred == pred_c) & (y_true == actual_c)))

            return confusion

        results["3_metrics_by_class"]["confusion_control"] = confusion_breakdown(y_true, ctrl_probs, "control")
        results["3_metrics_by_class"]["confusion_new"] = confusion_breakdown(y_true, new_probs, "new")

        # =================================================================
        # SECTION 4: POLICY EV ANALYSIS (edge distribution)
        # =================================================================
        logger.info("=== Section 4: Policy EV Analysis ===")

        EDGE_THRESHOLD = 0.05  # Same as in audit script

        def analyze_policy(probs, market_probs, y_true, odds_rows, name):
            """Analyze betting policy behavior."""

            # Calculate edges
            edges_home = probs[:, 0] - market_probs[:, 0]
            edges_draw = probs[:, 1] - market_probs[:, 1]
            edges_away = probs[:, 2] - market_probs[:, 2]

            # Edge statistics
            edge_stats = {
                "home": {
                    "mean": round(float(np.mean(edges_home)), 4),
                    "p50": round(float(np.percentile(edges_home, 50)), 4),
                    "p75": round(float(np.percentile(edges_home, 75)), 4),
                    "p90": round(float(np.percentile(edges_home, 90)), 4),
                    "n_above_threshold": int(np.sum(edges_home >= EDGE_THRESHOLD)),
                },
                "draw": {
                    "mean": round(float(np.mean(edges_draw)), 4),
                    "p50": round(float(np.percentile(edges_draw, 50)), 4),
                    "p75": round(float(np.percentile(edges_draw, 75)), 4),
                    "p90": round(float(np.percentile(edges_draw, 90)), 4),
                    "n_above_threshold": int(np.sum(edges_draw >= EDGE_THRESHOLD)),
                },
                "away": {
                    "mean": round(float(np.mean(edges_away)), 4),
                    "p50": round(float(np.percentile(edges_away, 50)), 4),
                    "p75": round(float(np.percentile(edges_away, 75)), 4),
                    "p90": round(float(np.percentile(edges_away, 90)), 4),
                    "n_above_threshold": int(np.sum(edges_away >= EDGE_THRESHOLD)),
                },
            }

            # Simulate betting policy (same as audit)
            bets_by_pick = {"home": [], "draw": [], "away": []}

            for i in range(len(probs)):
                model_p = probs[i]
                market_p = market_probs[i]

                edges = [model_p[j] - market_p[j] for j in range(3)]
                best_pick = int(np.argmax(edges))

                if edges[best_pick] >= EDGE_THRESHOLD:
                    odds = [float(odds_rows[i]['odds_home']), float(odds_rows[i]['odds_draw']), float(odds_rows[i]['odds_away'])][best_pick]
                    won = 1 if y_true[i] == best_pick else 0
                    returns = odds if won else 0
                    pick_name = ["home", "draw", "away"][best_pick]

                    bets_by_pick[pick_name].append({
                        "odds": odds,
                        "edge": edges[best_pick],
                        "p_model": model_p[best_pick],
                        "p_market": market_p[best_pick],
                        "won": won,
                        "returns": returns,
                    })

            # Aggregate bet stats
            bet_summary = {}
            for pick_name, bets in bets_by_pick.items():
                if not bets:
                    bet_summary[pick_name] = {"n_bets": 0}
                    continue

                total_staked = len(bets)
                total_returns = sum(b['returns'] for b in bets)
                roi = (total_returns - total_staked) / total_staked * 100
                win_rate = 100 * sum(b['won'] for b in bets) / total_staked

                bet_summary[pick_name] = {
                    "n_bets": total_staked,
                    "roi": round(roi, 2),
                    "win_rate": round(win_rate, 2),
                    "mean_odds": round(np.mean([b['odds'] for b in bets]), 3),
                    "mean_edge": round(np.mean([b['edge'] for b in bets]), 4),
                    "mean_p_model": round(np.mean([b['p_model'] for b in bets]), 4),
                    "mean_p_market": round(np.mean([b['p_market'] for b in bets]), 4),
                    "mean_p_diff": round(np.mean([b['p_model'] - b['p_market'] for b in bets]), 4),
                }

            return {
                "edge_threshold": EDGE_THRESHOLD,
                "edge_stats": edge_stats,
                "bet_summary": bet_summary,
                "total_bets": sum(len(bets) for bets in bets_by_pick.values()),
            }

        odds_rows = rows  # Same rows have odds

        results["4_policy_analysis"] = {
            "v1.0.0-control": analyze_policy(ctrl_probs, market_probs, y_true, odds_rows, "control"),
            "v1.0.1-league-only": analyze_policy(new_probs, market_probs, y_true, odds_rows, "new"),
            "policy_rule": {
                "description": "Select best edge if edge >= threshold",
                "edge_threshold": EDGE_THRESHOLD,
                "same_rule_both_models": True,
            }
        }

        # =================================================================
        # SECTION 5: KILL-SWITCH DETAILED BREAKDOWN
        # =================================================================
        logger.info("=== Section 5: Kill-Switch Analysis ===")

        # Get all matches that would be affected by kill-switch
        killswitch_query = text("""
            WITH team_league_counts AS (
                -- Count league matches per team in last 90 days before each upcoming match
                SELECT
                    t.id as team_id,
                    t.name as team_name,
                    COUNT(DISTINCT m.id) as league_matches_90d
                FROM teams t
                LEFT JOIN matches m ON (m.home_team_id = t.id OR m.away_team_id = t.id)
                LEFT JOIN admin_leagues al ON m.league_id = al.league_id
                WHERE m.status = 'FT'
                  AND al.kind = 'league'
                  AND m.date >= NOW() - INTERVAL '90 days'
                GROUP BY t.id, t.name
            )
            SELECT
                os.id as snapshot_id,
                os.match_id,
                m.date,
                al.name as league_name,
                al.kind as league_kind,
                ht.name as home_team,
                at.name as away_team,
                COALESCE(htc.league_matches_90d, 0) as home_league_matches,
                COALESCE(atc.league_matches_90d, 0) as away_league_matches
            FROM odds_snapshots os
            JOIN matches m ON os.match_id = m.id
            JOIN admin_leagues al ON m.league_id = al.league_id
            JOIN teams ht ON m.home_team_id = ht.id
            JOIN teams at ON m.away_team_id = at.id
            LEFT JOIN team_league_counts htc ON htc.team_id = ht.id
            LEFT JOIN team_league_counts atc ON atc.team_id = at.id
            WHERE os.snapshot_at >= :min_date
              AND (al.kind != 'league' OR COALESCE(htc.league_matches_90d, 0) < 5 OR COALESCE(atc.league_matches_90d, 0) < 5)
            ORDER BY os.snapshot_at
            LIMIT 200
        """)

        result = await session.execute(killswitch_query, {"min_date": min_date_dt})
        ks_rows = [dict(r._mapping) for r in result.fetchall()]

        # Categorize
        ks_reasons = defaultdict(int)
        ks_examples = defaultdict(list)

        for row in ks_rows:
            if row['league_kind'] != 'league':
                reason = "non_league_competition"
            elif row['home_league_matches'] < 5 and row['away_league_matches'] < 5:
                reason = "both_insufficient_history"
            elif row['home_league_matches'] < 5:
                reason = "home_insufficient_history"
            else:
                reason = "away_insufficient_history"

            ks_reasons[reason] += 1

            if len(ks_examples[reason]) < 3:
                ks_examples[reason].append({
                    "match_id": row['match_id'],
                    "league": row['league_name'],
                    "league_kind": row['league_kind'],
                    "home_team": row['home_team'],
                    "away_team": row['away_team'],
                    "home_matches": row['home_league_matches'],
                    "away_matches": row['away_league_matches'],
                })

        results["5_killswitch_detailed"] = {
            "n_would_be_filtered": len(ks_rows),
            "by_reason": dict(ks_reasons),
            "examples_by_reason": dict(ks_examples),
            "threshold": {
                "min_league_matches": 5,
                "lookback_days": 90,
            }
        }

        # =================================================================
        # SECTION 6: DIAGNOSIS SUMMARY
        # =================================================================
        logger.info("=== Section 6: Diagnosis Summary ===")

        # Is the problem in model or policy?
        ctrl_argmax_draw_pct = results["2_argmax_distribution"]["v1.0.0-control"]["argmax_picks"]["pct_draw"]
        new_argmax_draw_pct = results["2_argmax_distribution"]["v1.0.1-league-only"]["argmax_picks"]["pct_draw"]

        ctrl_policy_draw_bets = results["4_policy_analysis"]["v1.0.0-control"]["bet_summary"].get("draw", {}).get("n_bets", 0)
        new_policy_draw_bets = results["4_policy_analysis"]["v1.0.1-league-only"]["bet_summary"].get("draw", {}).get("n_bets", 0)

        ctrl_total_bets = results["4_policy_analysis"]["v1.0.0-control"]["total_bets"]
        new_total_bets = results["4_policy_analysis"]["v1.0.1-league-only"]["total_bets"]

        ctrl_policy_draw_pct = 100 * ctrl_policy_draw_bets / max(1, ctrl_total_bets)
        new_policy_draw_pct = 100 * new_policy_draw_bets / max(1, new_total_bets)

        results["6_diagnosis_summary"] = {
            "model_argmax_draw_inflation": {
                "control_pct": ctrl_argmax_draw_pct,
                "new_pct": new_argmax_draw_pct,
                "delta": round(new_argmax_draw_pct - ctrl_argmax_draw_pct, 2),
                "interpretation": "Positive delta = new model predicts draw more often as argmax"
            },
            "policy_draw_inflation": {
                "control_pct": round(ctrl_policy_draw_pct, 2),
                "new_pct": round(new_policy_draw_pct, 2),
                "delta": round(new_policy_draw_pct - ctrl_policy_draw_pct, 2),
                "interpretation": "Positive delta = policy bets on draw more often with new model"
            },
            "root_cause_hypothesis": None,
            "recommended_action": None,
        }

        # Determine root cause
        model_delta = new_argmax_draw_pct - ctrl_argmax_draw_pct
        policy_delta = new_policy_draw_pct - ctrl_policy_draw_pct

        if model_delta > 10:
            results["6_diagnosis_summary"]["root_cause_hypothesis"] = "MODEL: New model significantly inflates p_draw in argmax predictions"
            results["6_diagnosis_summary"]["recommended_action"] = "Retrain with reduced draw_weight or regularization targeting draw class"
        elif policy_delta > 20 and model_delta < 10:
            results["6_diagnosis_summary"]["root_cause_hypothesis"] = "POLICY: Edge selector is over-selecting draws due to market mispricing"
            results["6_diagnosis_summary"]["recommended_action"] = "Adjust policy: cap draw share, penalize draws with large p_draw deviation from market"
        else:
            results["6_diagnosis_summary"]["root_cause_hypothesis"] = "MIXED: Both model and policy contribute"
            results["6_diagnosis_summary"]["recommended_action"] = "Address model first (draw_weight), then tune policy if needed"

    await engine.dispose()
    return results


def main():
    parser = argparse.ArgumentParser(description="Diagnose draw problem - Fase 1")
    parser.add_argument('--min-snapshot-date', required=True, help='Min snapshot date (ISO)')
    args = parser.parse_args()

    results = asyncio.run(run_diagnosis(args.min_snapshot_date))

    # Save JSON
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"logs/draw_diagnosis_{timestamp}.json"

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n{'='*70}")
    print(f"DIAGNOSIS COMPLETE: {output_path}")
    print(f"{'='*70}")

    # Print key findings
    print("\n--- KEY FINDINGS ---\n")

    print("1. Exclusion Analysis:")
    exc = results["1_exclusion_analysis"]
    print(f"   Total with both predictions: {exc['total_with_both_predictions']}")
    print(f"   Final valid: {exc['final_valid']}")
    print(f"   Exclusion reasons: {exc['exclusion_reasons']}")

    print("\n2. Argmax Distribution (MODEL, not policy):")
    for model in ["v1.0.0-control", "v1.0.1-league-only", "market"]:
        picks = results["2_argmax_distribution"][model]["argmax_picks"]
        print(f"   {model}: H={picks['pct_home']}% D={picks['pct_draw']}% A={picks['pct_away']}%")

    print("\n   Actual outcomes:")
    actual = results["2_argmax_distribution"]["actual_outcomes"]
    print(f"   Actual: H={actual['pct_home']}% D={actual['pct_draw']}% A={actual['pct_away']}%")

    print("\n3. Metrics by Class:")
    for model in ["v1.0.0-control", "v1.0.1-league-only", "market"]:
        mc = results["3_metrics_by_class"][model]
        print(f"   {model}:")
        print(f"     Total Brier: {mc['total_brier']}, Total LogLoss: {mc['total_logloss']}")
        for cls in ["home", "draw", "away"]:
            c = mc["by_class"][cls]
            print(f"     {cls}: n_actual={c['n_actual']}, n_pred={c['n_predicted']}, brier={c['brier']}, acc={c['accuracy_when_predicted']}")

    print("\n4. Policy Analysis:")
    for model in ["v1.0.0-control", "v1.0.1-league-only"]:
        pa = results["4_policy_analysis"][model]
        print(f"   {model}: total_bets={pa['total_bets']}")
        for pick in ["home", "draw", "away"]:
            bs = pa["bet_summary"].get(pick, {})
            if bs.get("n_bets", 0) > 0:
                print(f"     {pick}: n={bs['n_bets']}, roi={bs['roi']}%, edge={bs['mean_edge']}, p_diff={bs['mean_p_diff']}")

    print("\n5. Kill-Switch:")
    ks = results["5_killswitch_detailed"]
    print(f"   Would filter: {ks['n_would_be_filtered']}")
    print(f"   By reason: {ks['by_reason']}")

    print("\n6. Diagnosis Summary:")
    ds = results["6_diagnosis_summary"]
    print(f"   Model argmax draw: ctrl={ds['model_argmax_draw_inflation']['control_pct']}% → new={ds['model_argmax_draw_inflation']['new_pct']}%")
    print(f"   Policy draw bets: ctrl={ds['policy_draw_inflation']['control_pct']}% → new={ds['policy_draw_inflation']['new_pct']}%")
    print(f"\n   ROOT CAUSE: {ds['root_cause_hypothesis']}")
    print(f"   RECOMMENDED: {ds['recommended_action']}")

    return output_path


if __name__ == "__main__":
    main()
