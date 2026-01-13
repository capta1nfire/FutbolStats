"""
Prediction Performance Metrics Module.

Calculates proper probability metrics for model evaluation:
- Brier score (primary)
- Log loss (secondary)
- Calibration by bins
- Market comparison

These metrics allow distinguishing variance from bugs.
"""

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


@dataclass
class PredictionRecord:
    """Single prediction with outcome for evaluation."""
    match_id: int
    league_id: int
    finished_at: datetime
    # Model probabilities
    home_prob: float
    draw_prob: float
    away_prob: float
    # Actual outcome (one-hot encoded)
    actual_home: int  # 1 if home won, else 0
    actual_draw: int  # 1 if draw, else 0
    actual_away: int  # 1 if away won, else 0
    # Confidence tier
    confidence_tier: Optional[str] = None
    # Market odds (optional)
    odds_home: Optional[float] = None
    odds_draw: Optional[float] = None
    odds_away: Optional[float] = None


@dataclass
class CalibrationBin:
    """Calibration bin for reliability diagram."""
    bin_start: float
    bin_end: float
    count: int = 0
    avg_confidence: float = 0.0
    empirical_accuracy: float = 0.0


@dataclass
class ConfusionMatrix:
    """3x3 confusion matrix for home/draw/away."""
    # Rows = predicted, Columns = actual
    # [[pred_home_act_home, pred_home_act_draw, pred_home_act_away], ...]
    matrix: list = field(default_factory=lambda: [[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    labels: list = field(default_factory=lambda: ["home", "draw", "away"])


def calculate_brier_score(records: list[PredictionRecord]) -> float:
    """
    Calculate multiclass Brier score.

    Formula: (1/N) * sum((p_home - y_home)^2 + (p_draw - y_draw)^2 + (p_away - y_away)^2)

    Range: [0, 2] for 3 classes. Lower is better.
    - Perfect: 0
    - Random uniform (0.33, 0.33, 0.33): ~0.667
    - Worst case (confident wrong): 2.0
    """
    if not records:
        return float('nan')

    total = 0.0
    for r in records:
        # Squared errors for each class
        err_home = (r.home_prob - r.actual_home) ** 2
        err_draw = (r.draw_prob - r.actual_draw) ** 2
        err_away = (r.away_prob - r.actual_away) ** 2
        total += err_home + err_draw + err_away

    return total / len(records)


def calculate_log_loss(records: list[PredictionRecord], eps: float = 1e-15) -> float:
    """
    Calculate multiclass log loss (cross-entropy).

    Formula: -(1/N) * sum(y_home*log(p_home) + y_draw*log(p_draw) + y_away*log(p_away))

    Range: [0, inf). Lower is better.
    - Perfect: 0
    - Random uniform: ~1.099 (log(3))
    """
    if not records:
        return float('nan')

    total = 0.0
    for r in records:
        # Clip probabilities to avoid log(0)
        p_home = max(eps, min(1 - eps, r.home_prob))
        p_draw = max(eps, min(1 - eps, r.draw_prob))
        p_away = max(eps, min(1 - eps, r.away_prob))

        # Only the actual class contributes
        loss = -(
            r.actual_home * math.log(p_home) +
            r.actual_draw * math.log(p_draw) +
            r.actual_away * math.log(p_away)
        )
        total += loss

    return total / len(records)


def calculate_accuracy(records: list[PredictionRecord]) -> tuple[int, int, float]:
    """
    Calculate argmax accuracy.

    Returns: (correct_count, total_count, accuracy_pct)
    """
    if not records:
        return 0, 0, 0.0

    correct = 0
    for r in records:
        probs = [r.home_prob, r.draw_prob, r.away_prob]
        predicted = probs.index(max(probs))  # 0=home, 1=draw, 2=away
        actual = [r.actual_home, r.actual_draw, r.actual_away].index(1)
        if predicted == actual:
            correct += 1

    return correct, len(records), round(100 * correct / len(records), 2)


def calculate_calibration_bins(
    records: list[PredictionRecord],
    n_bins: int = 7
) -> list[CalibrationBin]:
    """
    Calculate calibration bins for reliability diagram.

    Bins predictions by max probability and compares:
    - avg_confidence: mean of max(p) in bin
    - empirical_accuracy: actual hit rate when model predicted that class

    Well-calibrated model: avg_confidence ≈ empirical_accuracy
    """
    # Define bin edges (focus on typical prediction ranges)
    # [0.33-0.40, 0.40-0.50, 0.50-0.60, 0.60-0.70, 0.70-0.80, 0.80-0.90, 0.90-1.0]
    bin_edges = [0.33, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.01]

    bins = []
    for i in range(len(bin_edges) - 1):
        bins.append(CalibrationBin(
            bin_start=bin_edges[i],
            bin_end=bin_edges[i + 1]
        ))

    # Assign each prediction to a bin
    bin_data = {i: {"confidences": [], "correct": 0, "total": 0} for i in range(len(bins))}

    for r in records:
        probs = [r.home_prob, r.draw_prob, r.away_prob]
        max_prob = max(probs)
        predicted_idx = probs.index(max_prob)
        actuals = [r.actual_home, r.actual_draw, r.actual_away]
        is_correct = actuals[predicted_idx] == 1

        # Find bin
        for i, b in enumerate(bins):
            if b.bin_start <= max_prob < b.bin_end:
                bin_data[i]["confidences"].append(max_prob)
                bin_data[i]["total"] += 1
                if is_correct:
                    bin_data[i]["correct"] += 1
                break

    # Calculate bin statistics
    for i, b in enumerate(bins):
        data = bin_data[i]
        b.count = data["total"]
        if data["total"] > 0:
            b.avg_confidence = round(sum(data["confidences"]) / data["total"], 4)
            b.empirical_accuracy = round(data["correct"] / data["total"], 4)

    return bins


def calculate_confusion_matrix(records: list[PredictionRecord]) -> ConfusionMatrix:
    """
    Calculate 3x3 confusion matrix.

    Rows = predicted class (argmax)
    Columns = actual class
    """
    cm = ConfusionMatrix()

    for r in records:
        probs = [r.home_prob, r.draw_prob, r.away_prob]
        predicted = probs.index(max(probs))
        actual = [r.actual_home, r.actual_draw, r.actual_away].index(1)
        cm.matrix[predicted][actual] += 1

    return cm


def calculate_market_brier(records: list[PredictionRecord]) -> Optional[float]:
    """
    Calculate Brier score using market implied probabilities.

    Implied probs are normalized: p_i = (1/odds_i) / sum(1/odds_j)

    Returns None if insufficient odds data.
    """
    records_with_odds = [r for r in records if r.odds_home and r.odds_draw and r.odds_away]

    if len(records_with_odds) < 10:
        return None

    total = 0.0
    for r in records_with_odds:
        # Calculate implied probabilities (normalized)
        inv_home = 1.0 / r.odds_home
        inv_draw = 1.0 / r.odds_draw
        inv_away = 1.0 / r.odds_away
        total_inv = inv_home + inv_draw + inv_away

        p_home = inv_home / total_inv
        p_draw = inv_draw / total_inv
        p_away = inv_away / total_inv

        # Squared errors
        err_home = (p_home - r.actual_home) ** 2
        err_draw = (p_draw - r.actual_draw) ** 2
        err_away = (p_away - r.actual_away) ** 2
        total += err_home + err_draw + err_away

    return total / len(records_with_odds)


def calculate_skill_vs_market(model_brier: float, market_brier: float) -> Optional[float]:
    """
    Calculate skill score vs market.

    Formula: skill = 1 - (model_brier / market_brier)

    Interpretation:
    - skill > 0: Model beats market
    - skill = 0: Model equals market
    - skill < 0: Model worse than market

    Example: skill = 0.05 means model is 5% better than market
    """
    if market_brier is None or market_brier == 0:
        return None

    return round(1 - (model_brier / market_brier), 4)


async def fetch_prediction_records(
    session: AsyncSession,
    window_days: int,
    min_n: int = 10
) -> list[PredictionRecord]:
    """
    Fetch prediction records for evaluation from database.

    Joins matches, predictions, and prediction_outcomes for finished matches.
    Uses COALESCE(finished_at, date) for window calculation.
    """
    cutoff = datetime.utcnow() - timedelta(days=window_days)

    query = text("""
        SELECT
            m.id as match_id,
            m.league_id,
            COALESCE(m.finished_at, m.date) as finished_at,
            p.home_prob,
            p.draw_prob,
            p.away_prob,
            m.home_goals,
            m.away_goals,
            po.confidence_tier,
            m.odds_home,
            m.odds_draw,
            m.odds_away
        FROM matches m
        JOIN predictions p ON p.match_id = m.id
        LEFT JOIN prediction_outcomes po ON po.match_id = m.id
        WHERE m.status IN ('FT', 'AET', 'PEN')
          AND m.home_goals IS NOT NULL
          AND m.away_goals IS NOT NULL
          AND COALESCE(m.finished_at, m.date) >= :cutoff
          AND COALESCE(m.finished_at, m.date) <= NOW()
        ORDER BY COALESCE(m.finished_at, m.date) DESC
    """)

    result = await session.execute(query, {"cutoff": cutoff})
    rows = result.fetchall()

    records = []
    for row in rows:
        # Determine actual outcome (one-hot)
        home_goals = row.home_goals
        away_goals = row.away_goals

        if home_goals > away_goals:
            actual_home, actual_draw, actual_away = 1, 0, 0
        elif home_goals == away_goals:
            actual_home, actual_draw, actual_away = 0, 1, 0
        else:
            actual_home, actual_draw, actual_away = 0, 0, 1

        records.append(PredictionRecord(
            match_id=row.match_id,
            league_id=row.league_id,
            finished_at=row.finished_at,
            home_prob=row.home_prob,
            draw_prob=row.draw_prob,
            away_prob=row.away_prob,
            actual_home=actual_home,
            actual_draw=actual_draw,
            actual_away=actual_away,
            confidence_tier=row.confidence_tier,
            odds_home=row.odds_home,
            odds_draw=row.odds_draw,
            odds_away=row.odds_away,
        ))

    logger.info(f"Fetched {len(records)} prediction records for {window_days}d window")
    return records


def segment_by_league(
    records: list[PredictionRecord],
    top_n: int = 5
) -> dict[int, list[PredictionRecord]]:
    """Segment records by league_id, return top N by volume."""
    by_league = {}
    for r in records:
        if r.league_id not in by_league:
            by_league[r.league_id] = []
        by_league[r.league_id].append(r)

    # Sort by volume and take top N
    sorted_leagues = sorted(by_league.items(), key=lambda x: len(x[1]), reverse=True)
    return dict(sorted_leagues[:top_n])


def segment_by_tier(records: list[PredictionRecord]) -> dict[str, list[PredictionRecord]]:
    """Segment records by confidence tier."""
    by_tier = {"gold": [], "silver": [], "copper": [], "unknown": []}
    for r in records:
        tier = r.confidence_tier or "unknown"
        if tier in by_tier:
            by_tier[tier].append(r)
        else:
            by_tier["unknown"].append(r)
    return by_tier


def calculate_segment_metrics(records: list[PredictionRecord]) -> dict:
    """Calculate all metrics for a segment of records."""
    if not records:
        return {
            "n": 0,
            "n_with_odds": 0,
            "coverage_pct": 0.0,
            "metrics": None
        }

    n_with_odds = sum(1 for r in records if r.odds_home and r.odds_draw and r.odds_away)

    correct, total, accuracy = calculate_accuracy(records)
    brier = calculate_brier_score(records)
    logloss = calculate_log_loss(records)
    calibration = calculate_calibration_bins(records)
    confusion = calculate_confusion_matrix(records)
    market_brier = calculate_market_brier(records)
    skill = calculate_skill_vs_market(brier, market_brier) if market_brier else None

    return {
        "n": len(records),
        "n_with_odds": n_with_odds,
        "coverage_pct": round(100 * n_with_odds / len(records), 1) if records else 0,
        "metrics": {
            "accuracy": {
                "correct": correct,
                "total": total,
                "pct": accuracy
            },
            "brier_score": round(brier, 4) if not math.isnan(brier) else None,
            "log_loss": round(logloss, 4) if not math.isnan(logloss) else None,
            "calibration": [
                {
                    "bin": f"{b.bin_start:.2f}-{b.bin_end:.2f}",
                    "count": b.count,
                    "avg_confidence": b.avg_confidence,
                    "empirical_accuracy": b.empirical_accuracy,
                    "calibration_error": round(abs(b.avg_confidence - b.empirical_accuracy), 4) if b.count > 0 else None
                }
                for b in calibration
            ],
            "confusion_matrix": {
                "labels": confusion.labels,
                "matrix": confusion.matrix,
                "description": "Rows=predicted, Cols=actual"
            },
            "market_comparison": {
                "market_brier": round(market_brier, 4) if market_brier else None,
                "skill_vs_market": skill,
                "interpretation": "skill > 0 means model beats market"
            } if market_brier else None
        }
    }


async def generate_performance_report(
    session: AsyncSession,
    window_days: int,
    min_n_confidence: int = 30
) -> dict:
    """
    Generate complete performance report.

    Args:
        session: Database session
        window_days: 7 or 14
        min_n_confidence: Minimum samples for high confidence conclusions

    Returns:
        Complete report dict ready for JSON storage
    """
    logger.info(f"Generating performance report for {window_days}d window")

    records = await fetch_prediction_records(session, window_days)

    # Global metrics
    global_metrics = calculate_segment_metrics(records)

    # Confidence flag
    confidence = "high" if len(records) >= min_n_confidence else "low"

    # By league (top 5)
    by_league = segment_by_league(records, top_n=5)
    league_metrics = {}
    for league_id, league_records in by_league.items():
        league_metrics[str(league_id)] = calculate_segment_metrics(league_records)

    # By confidence tier
    by_tier = segment_by_tier(records)
    tier_metrics = {}
    for tier, tier_records in by_tier.items():
        if tier_records:  # Only include non-empty tiers
            tier_metrics[tier] = calculate_segment_metrics(tier_records)

    # Diagnostic signals
    diagnostics = _calculate_diagnostics(global_metrics, records)

    report = {
        "window_days": window_days,
        "generated_at": datetime.utcnow().isoformat(),
        "confidence": confidence,
        "min_n_for_high_confidence": min_n_confidence,
        "global": global_metrics,
        "by_league": league_metrics,
        "by_confidence_tier": tier_metrics,
        "diagnostics": diagnostics,
        "interpretation_guide": {
            "brier_score": "Lower is better. Perfect=0, Uniform=0.667, Worst=2.0",
            "log_loss": "Lower is better. Perfect=0, Uniform=1.099",
            "calibration_error": "Low error (<0.05) = well calibrated. High error = model overconfident or underconfident",
            "skill_vs_market": "Positive = beating market, Negative = worse than market",
            "variance_vs_bug": "High log_loss + poor calibration = bug. Normal log_loss + poor accuracy = variance"
        }
    }

    logger.info(f"Report generated: {global_metrics['n']} predictions, confidence={confidence}")
    return report


def _calculate_diagnostics(global_metrics: dict, records: list[PredictionRecord]) -> dict:
    """
    Calculate diagnostic signals to distinguish variance from bugs.

    Key signals:
    - Calibration drift: Are we systematically over/underconfident?
    - Log loss explosion: Is the model making confident wrong predictions?
    - Class imbalance: Are we over-predicting one outcome?
    """
    diagnostics = {
        "signals": [],
        "recommendation": None
    }

    if not global_metrics.get("metrics"):
        diagnostics["signals"].append({
            "type": "insufficient_data",
            "severity": "warn",
            "message": "Not enough data for diagnostics"
        })
        return diagnostics

    metrics = global_metrics["metrics"]

    # 1. Log loss explosion check
    logloss = metrics.get("log_loss")
    if logloss and logloss > 1.2:  # Worse than uniform + margin
        diagnostics["signals"].append({
            "type": "logloss_explosion",
            "severity": "red",
            "value": logloss,
            "threshold": 1.2,
            "message": "Log loss significantly worse than random - possible bug in probability calculation"
        })

    # 2. Calibration drift check
    calibration = metrics.get("calibration", [])
    high_error_bins = [
        b for b in calibration
        if b.get("count", 0) >= 5 and b.get("calibration_error", 0) > 0.15
    ]
    if high_error_bins:
        diagnostics["signals"].append({
            "type": "calibration_drift",
            "severity": "warn",
            "bins_affected": len(high_error_bins),
            "message": f"{len(high_error_bins)} bins with calibration error > 15%"
        })

    # 3. Class imbalance in predictions
    confusion = metrics.get("confusion_matrix", {}).get("matrix", [[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    pred_home = sum(confusion[0])
    pred_draw = sum(confusion[1])
    pred_away = sum(confusion[2])
    total_preds = pred_home + pred_draw + pred_away

    if total_preds > 0:
        # Check if draw predictions are suspiciously low or high
        draw_pct = pred_draw / total_preds
        if draw_pct < 0.10:
            diagnostics["signals"].append({
                "type": "draw_underprediction",
                "severity": "info",
                "value": round(draw_pct * 100, 1),
                "message": f"Only {round(draw_pct * 100, 1)}% predictions are draws - might be underweighting draws"
            })
        elif draw_pct > 0.40:
            diagnostics["signals"].append({
                "type": "draw_overprediction",
                "severity": "warn",
                "value": round(draw_pct * 100, 1),
                "message": f"{round(draw_pct * 100, 1)}% predictions are draws - unusual pattern"
            })

    # 4. Market comparison
    market = metrics.get("market_comparison")
    if market and market.get("skill_vs_market") is not None:
        skill = market["skill_vs_market"]
        if skill < -0.10:
            diagnostics["signals"].append({
                "type": "worse_than_market",
                "severity": "warn",
                "value": skill,
                "message": f"Model {abs(skill)*100:.1f}% worse than market implied probabilities"
            })
        elif skill > 0.05:
            diagnostics["signals"].append({
                "type": "beating_market",
                "severity": "ok",
                "value": skill,
                "message": f"Model {skill*100:.1f}% better than market"
            })

    # Generate recommendation
    red_signals = [s for s in diagnostics["signals"] if s.get("severity") == "red"]
    warn_signals = [s for s in diagnostics["signals"] if s.get("severity") == "warn"]

    if red_signals:
        diagnostics["recommendation"] = "INVESTIGATE: Red signals detected - likely bug, not variance"
    elif len(warn_signals) >= 2:
        diagnostics["recommendation"] = "MONITOR: Multiple warnings - could be systematic issue"
    elif warn_signals:
        diagnostics["recommendation"] = "WATCH: Single warning - likely variance, monitor next report"
    else:
        diagnostics["recommendation"] = "OK: No concerning signals - performance within expected variance"

    return diagnostics


async def save_performance_report(
    session: AsyncSession,
    report: dict,
    window_days: int,
    source: str = "scheduler"
) -> int:
    """
    Save performance report to database.

    Uses upsert to replace existing report for same window/date.
    Returns report ID.
    """
    import json
    from datetime import date

    today = date.today()

    result = await session.execute(
        text("""
            INSERT INTO prediction_performance_reports
                (generated_at, window_days, report_date, payload, source)
            VALUES (NOW(), :window_days, :report_date, :payload, :source)
            ON CONFLICT (window_days, report_date)
            DO UPDATE SET
                generated_at = NOW(),
                payload = :payload,
                source = :source
            RETURNING id
        """),
        {
            "window_days": window_days,
            "report_date": today,
            "payload": json.dumps(report),
            "source": source
        }
    )

    report_id = result.scalar()
    await session.commit()

    logger.info(f"Saved performance report id={report_id} for {window_days}d window")
    return report_id


async def get_latest_report(
    session: AsyncSession,
    window_days: int
) -> Optional[dict]:
    """Get the most recent performance report for given window."""
    import json

    result = await session.execute(
        text("""
            SELECT payload, generated_at
            FROM prediction_performance_reports
            WHERE window_days = :window_days
            ORDER BY report_date DESC, generated_at DESC
            LIMIT 1
        """),
        {"window_days": window_days}
    )

    row = result.fetchone()
    if not row:
        return None

    payload = row.payload
    if isinstance(payload, str):
        payload = json.loads(payload)

    return payload
