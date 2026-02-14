"""
SteamChaserModel — Line Movement Prediction Pipeline (Phase 2, P2-13).

Predicts whether the closing line will collapse beyond the vig-adjusted threshold
after lineup confirmation. Binary classification (XGBoost).

Target (ATI directive — vig-adjusted):
    y = 1 if max(|prob_close_k - prob_T60_k|) > overround_T60 / 2
    This ensures we only chase movements that overcome bookmaker margin.

Status: SHADOW MODE / DATA COLLECTION ONLY.
    - Only ~644 training pairs available (Jan 2026+), 10 positives (1.6%)
    - Minimum ~2,000 pairs needed before meaningful training
    - Pipeline collects data, defines target, builds features
    - Training deferred until data accumulates (~mid-April 2026)

Features:
    Available now: overround_T60, prob_T60_{home,draw,away}, league_id
    Forward-only: talent_delta_{home,away}, shock_magnitude, xi_continuity
"""

import logging
from datetime import datetime
from typing import Optional, Dict, List, Tuple

import numpy as np

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger("futbolstats.steamchaser")

# ATI directive: threshold must beat vig friction
# Movement must exceed half the bookmaker margin to represent +EV
VIG_DIVISOR = 2  # threshold = overround / VIG_DIVISOR

# Minimum samples before allowing training
MIN_TRAINING_SAMPLES = 500
MIN_POSITIVE_SAMPLES = 30


async def collect_training_data(session: AsyncSession) -> Dict:
    """
    Collect T60→T5 pairs with features for SteamChaser training.

    Returns dict with:
        - pairs: list of feature dicts
        - stats: summary statistics
    """
    result = await session.execute(text("""
        WITH pairs AS (
            SELECT
                t60.match_id,
                m.league_id,
                m.date AS match_date,
                -- T60 state (features available at prediction time)
                t60.prob_home AS p60_h, t60.prob_draw AS p60_d, t60.prob_away AS p60_a,
                t60.overround AS overround_t60,
                t60.bookmaker AS bookmaker_t60,
                -- T5 state (proxy for closing line)
                t5.prob_home AS p5_h, t5.prob_draw AS p5_d, t5.prob_away AS p5_a,
                t5.overround AS overround_t5,
                -- Max drift across outcomes
                GREATEST(
                    ABS(t5.prob_home - t60.prob_home),
                    ABS(t5.prob_draw - t60.prob_draw),
                    ABS(t5.prob_away - t60.prob_away)
                ) AS max_drift,
                -- Directional drift (which outcome moved most)
                CASE
                    WHEN ABS(t5.prob_home - t60.prob_home) = GREATEST(
                        ABS(t5.prob_home - t60.prob_home),
                        ABS(t5.prob_draw - t60.prob_draw),
                        ABS(t5.prob_away - t60.prob_away)
                    ) THEN 'home'
                    WHEN ABS(t5.prob_draw - t60.prob_draw) = GREATEST(
                        ABS(t5.prob_home - t60.prob_home),
                        ABS(t5.prob_draw - t60.prob_draw),
                        ABS(t5.prob_away - t60.prob_away)
                    ) THEN 'draw'
                    ELSE 'away'
                END AS drift_outcome
            FROM market_movement_snapshots t60
            JOIN market_movement_snapshots t5
                ON t5.match_id = t60.match_id AND t5.snapshot_type = 'T5'
            JOIN matches m ON m.id = t60.match_id
            WHERE t60.snapshot_type = 'T60'
              AND t60.prob_home IS NOT NULL
              AND t5.prob_home IS NOT NULL
              AND t60.overround IS NOT NULL
            ORDER BY m.date
        )
        SELECT *,
               -- Target: big move = drift exceeds vig friction (overround/2)
               CASE WHEN max_drift > overround_t60 / :vig_divisor THEN 1 ELSE 0 END AS target
        FROM pairs
    """), {"vig_divisor": VIG_DIVISOR})

    rows = result.fetchall()
    if not rows:
        return {"pairs": [], "stats": {"n": 0, "n_positive": 0, "pct_positive": 0.0}}

    pairs = []
    for row in rows:
        pairs.append({
            "match_id": row.match_id,
            "league_id": row.league_id,
            "match_date": row.match_date,
            # Features
            "overround_t60": float(row.overround_t60),
            "prob_t60_home": float(row.p60_h),
            "prob_t60_draw": float(row.p60_d),
            "prob_t60_away": float(row.p60_a),
            "prob_t60_fav": float(max(row.p60_h, row.p60_d, row.p60_a)),
            "prob_t60_range": float(max(row.p60_h, row.p60_d, row.p60_a) - min(row.p60_h, row.p60_d, row.p60_a)),
            # Target info
            "max_drift": float(row.max_drift),
            "drift_outcome": row.drift_outcome,
            "target": int(row.target),
            "vig_threshold": float(row.overround_t60) / VIG_DIVISOR,
        })

    n_positive = sum(1 for p in pairs if p["target"] == 1)
    stats = {
        "n": len(pairs),
        "n_positive": n_positive,
        "n_negative": len(pairs) - n_positive,
        "pct_positive": round(100.0 * n_positive / len(pairs), 2),
        "avg_overround": round(np.mean([p["overround_t60"] for p in pairs]), 5),
        "avg_vig_threshold": round(np.mean([p["vig_threshold"] for p in pairs]), 5),
        "avg_max_drift": round(np.mean([p["max_drift"] for p in pairs]), 5),
        "p90_max_drift": round(float(np.percentile([p["max_drift"] for p in pairs], 90)), 5),
        "data_range": f"{pairs[0]['match_date'].date()} to {pairs[-1]['match_date'].date()}",
        "ready_for_training": n_positive >= MIN_POSITIVE_SAMPLES and len(pairs) >= MIN_TRAINING_SAMPLES,
    }

    return {"pairs": pairs, "stats": stats}


async def collect_forward_features(
    session: AsyncSession,
    match_id: int,
) -> Optional[Dict]:
    """
    Collect SteamChaser features for a single match (forward inference).

    Used by cascade handler for shadow prediction.
    Returns None if T60 snapshot not available.
    """
    result = await session.execute(text("""
        SELECT
            mms.prob_home, mms.prob_draw, mms.prob_away, mms.overround,
            m.league_id
        FROM market_movement_snapshots mms
        JOIN matches m ON m.id = mms.match_id
        WHERE mms.match_id = :match_id
          AND mms.snapshot_type = 'T60'
          AND mms.prob_home IS NOT NULL
        ORDER BY mms.captured_at DESC
        LIMIT 1
    """), {"match_id": match_id})
    row = result.fetchone()

    if not row or row.overround is None:
        return None

    return {
        "overround_t60": float(row.overround),
        "prob_t60_home": float(row.prob_home),
        "prob_t60_draw": float(row.prob_draw),
        "prob_t60_away": float(row.prob_away),
        "prob_t60_fav": float(max(row.prob_home, row.prob_draw, row.prob_away)),
        "prob_t60_range": float(max(row.prob_home, row.prob_draw, row.prob_away) - min(row.prob_home, row.prob_draw, row.prob_away)),
        "league_id": row.league_id,
        "vig_threshold": float(row.overround) / VIG_DIVISOR,
    }


def build_feature_matrix(pairs: List[Dict]) -> Tuple:
    """
    Build X, y arrays from collected pairs for XGBoost training.

    Features:
        0: overround_t60
        1: prob_t60_home
        2: prob_t60_draw
        3: prob_t60_away
        4: prob_t60_fav (max probability — market confidence)
        5: prob_t60_range (spread between fav and underdog)

    Future features (when data accumulates):
        - talent_delta_home, talent_delta_away
        - shock_magnitude
        - xi_continuity_home, xi_continuity_away

    Returns (X, y, feature_names).
    """
    feature_names = [
        "overround_t60",
        "prob_t60_home",
        "prob_t60_draw",
        "prob_t60_away",
        "prob_t60_fav",
        "prob_t60_range",
    ]

    X = np.array([
        [p[f] for f in feature_names]
        for p in pairs
    ], dtype=np.float64)

    y = np.array([p["target"] for p in pairs], dtype=np.int32)

    return X, y, feature_names


def evaluate_model(y_true, y_pred_proba) -> Dict:
    """
    Evaluate SteamChaser model with ATI-mandated metrics.

    Returns AUC and LogLoss (NOT accuracy — ATI directive P2-14).
    """
    from sklearn.metrics import roc_auc_score, log_loss

    metrics = {}

    n_pos = int(np.sum(y_true))
    n_neg = len(y_true) - n_pos
    metrics["n"] = len(y_true)
    metrics["n_positive"] = n_pos
    metrics["n_negative"] = n_neg
    metrics["pct_positive"] = round(100.0 * n_pos / len(y_true), 2)

    if n_pos == 0 or n_neg == 0:
        metrics["auc"] = None
        metrics["logloss"] = None
        metrics["status"] = "INSUFFICIENT_CLASSES"
        return metrics

    try:
        metrics["auc"] = round(float(roc_auc_score(y_true, y_pred_proba)), 5)
    except Exception:
        metrics["auc"] = None

    try:
        metrics["logloss"] = round(float(log_loss(y_true, y_pred_proba)), 5)
    except Exception:
        metrics["logloss"] = None

    metrics["status"] = "OK"
    return metrics


def run_oot_evaluation(pairs: List[Dict], oot_fraction: float = 0.3) -> Dict:
    """
    Out-of-Time evaluation of SteamChaser (P2-14).

    Splits data chronologically, trains XGBoost on early period,
    evaluates on later period with AUC + LogLoss (ATI mandate: NO accuracy).

    Also evaluates a naive baseline (predict mean positive rate)
    to provide a reference for the model's discriminative ability.

    Returns evaluation report dict.
    """
    if len(pairs) < 50:
        return {
            "status": "INSUFFICIENT_DATA",
            "n_pairs": len(pairs),
            "min_required": 50,
        }

    # Sort by date (should already be sorted from SQL)
    sorted_pairs = sorted(pairs, key=lambda p: p["match_date"])

    # OOT split
    split_idx = int(len(sorted_pairs) * (1 - oot_fraction))
    train_pairs = sorted_pairs[:split_idx]
    test_pairs = sorted_pairs[split_idx:]

    X_train, y_train, feature_names = build_feature_matrix(train_pairs)
    X_test, y_test, _ = build_feature_matrix(test_pairs)

    n_pos_train = int(np.sum(y_train))
    n_pos_test = int(np.sum(y_test))

    report = {
        "split": {
            "train_n": len(train_pairs),
            "train_positive": n_pos_train,
            "train_date_range": f"{train_pairs[0]['match_date'].date()} to {train_pairs[-1]['match_date'].date()}",
            "test_n": len(test_pairs),
            "test_positive": n_pos_test,
            "test_date_range": f"{test_pairs[0]['match_date'].date()} to {test_pairs[-1]['match_date'].date()}",
        },
        "features": feature_names,
        "vig_divisor": VIG_DIVISOR,
    }

    # Check if enough positives in both sets
    if n_pos_train < 3 or n_pos_test < 1:
        report["status"] = "INSUFFICIENT_POSITIVES"
        report["model_metrics"] = None
        report["baseline_metrics"] = None
        report["conclusion"] = (
            f"Only {n_pos_train} positives in train, {n_pos_test} in test. "
            f"Need >=3 train, >=1 test. Accumulate more data."
        )
        return report

    # Train XGBoost (handle imbalance with scale_pos_weight)
    try:
        import xgboost as xgb

        scale = (len(y_train) - n_pos_train) / max(n_pos_train, 1)
        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            scale_pos_weight=scale,
            eval_metric="logloss",
            use_label_encoder=False,
            random_state=42,
        )
        model.fit(X_train, y_train, verbose=False)

        # Predict probabilities on test set
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        # Model metrics
        report["model_metrics"] = evaluate_model(y_test, y_pred_proba)

        # Feature importance
        importances = model.feature_importances_
        report["feature_importance"] = {
            feature_names[i]: round(float(importances[i]), 4)
            for i in np.argsort(importances)[::-1]
        }

    except Exception as e:
        report["model_metrics"] = {"status": "TRAINING_FAILED", "error": str(e)}

    # Baseline: predict mean positive rate for all (no discrimination)
    base_rate = n_pos_train / len(y_train)
    baseline_pred = np.full(len(y_test), base_rate)
    report["baseline_metrics"] = evaluate_model(y_test, baseline_pred)

    # Conclusion
    model_auc = (report.get("model_metrics") or {}).get("auc")
    base_auc = (report.get("baseline_metrics") or {}).get("auc")
    if model_auc and base_auc:
        delta_auc = model_auc - 0.5  # AUC of random = 0.5
        report["conclusion"] = (
            f"Model AUC={model_auc:.4f} (baseline=0.5, delta=+{delta_auc:.4f}). "
            f"{'Signal detected' if model_auc > 0.55 else 'No meaningful signal yet'}. "
            f"Positive rate: train={100*n_pos_train/len(y_train):.1f}%, test={100*n_pos_test/len(y_test):.1f}%."
        )
    else:
        report["conclusion"] = "Evaluation inconclusive (insufficient classes or training failure)."

    report["status"] = "OK"
    return report


async def training_readiness_check(session: AsyncSession) -> Dict:
    """
    Check if we have enough data to train SteamChaser.

    Returns readiness status and data stats.
    """
    data = await collect_training_data(session)
    stats = data["stats"]

    return {
        "ready": stats.get("ready_for_training", False),
        "n_pairs": stats["n"],
        "n_positive": stats["n_positive"],
        "pct_positive": stats["pct_positive"],
        "min_required_samples": MIN_TRAINING_SAMPLES,
        "min_required_positives": MIN_POSITIVE_SAMPLES,
        "gap_samples": max(0, MIN_TRAINING_SAMPLES - stats["n"]),
        "gap_positives": max(0, MIN_POSITIVE_SAMPLES - stats["n_positive"]),
        "avg_overround": stats.get("avg_overround"),
        "vig_threshold": stats.get("avg_vig_threshold"),
        "data_range": stats.get("data_range"),
        "recommendation": (
            "READY: Proceed with training"
            if stats.get("ready_for_training")
            else f"ACCUMULATE: Need {max(0, MIN_TRAINING_SAMPLES - stats['n'])} more samples "
                 f"and {max(0, MIN_POSITIVE_SAMPLES - stats['n_positive'])} more positives"
        ),
    }
