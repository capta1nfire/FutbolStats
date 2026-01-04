"""
Recalibration engine for automatic model adjustment.

This module handles:
1. Per-team confidence adjustments based on anomaly rates
2. Trigger evaluation for model retraining
3. Model validation before deployment
4. Snapshot creation and rollback
"""

import logging
from datetime import datetime, timedelta
from typing import Optional

from sqlalchemy import func, select, and_, Integer, case
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.dialects.postgresql import insert as pg_insert

from app.models import (
    Team,
    Match,
    PredictionOutcome,
    PostMatchAudit,
    TeamAdjustment,
    ModelSnapshot,
)

logger = logging.getLogger(__name__)


# Trigger thresholds
GOLD_ACCURACY_THRESHOLD = 0.65      # Retrain if gold tier < 65%
ANOMALY_RATE_THRESHOLD = 0.20       # Retrain if anomaly rate > 20%
MIN_PREDICTIONS_FOR_TRIGGER = 20    # Minimum predictions before evaluating triggers

# Validation thresholds
BRIER_SCORE_BASELINE = 0.2063       # Don't deploy if >= this value
BRIER_SCORE_IMPROVEMENT_MIN = 0.005 # Minimum improvement required

# Team adjustment thresholds (based on anomaly rate)
ADJUSTMENT_SEVERE = 0.85    # anomaly_rate > 30%
ADJUSTMENT_MODERATE = 0.90  # anomaly_rate > 20%
ADJUSTMENT_MILD = 0.95      # anomaly_rate > 10%
MIN_TEAM_PREDICTIONS = 5    # Minimum predictions before adjusting team


class RecalibrationEngine:
    """
    Engine for automatic model recalibration.

    Monitors prediction performance and triggers adjustments when needed.
    """

    def __init__(self, session: AsyncSession):
        self.session = session

    async def calculate_team_adjustments(self, days: int = 30) -> dict:
        """
        Calculate and update per-team confidence adjustments.

        Analyzes prediction outcomes for each team and adjusts confidence
        based on anomaly rates.

        Args:
            days: Number of days to look back for analysis

        Returns:
            Dictionary with adjustment results
        """
        cutoff_date = datetime.utcnow() - timedelta(days=days)

        # Get all outcomes with their audits, grouped by team
        # We need to analyze both home and away performances
        query = (
            select(
                Match.home_team_id.label("team_id"),
                func.count(PredictionOutcome.id).label("total"),
                func.sum(
                    case((PredictionOutcome.prediction_correct == True, 1), else_=0)
                ).label("correct"),
                func.sum(
                    case((PostMatchAudit.deviation_type == "anomaly", 1), else_=0)
                ).label("anomalies"),
                func.avg(PostMatchAudit.deviation_score).label("avg_deviation"),
            )
            .select_from(PredictionOutcome)
            .join(Match, PredictionOutcome.match_id == Match.id)
            .outerjoin(PostMatchAudit, PredictionOutcome.id == PostMatchAudit.outcome_id)
            .where(PredictionOutcome.audited_at >= cutoff_date)
            .group_by(Match.home_team_id)
        )

        home_results = await self.session.execute(query)
        home_stats = {row.team_id: row for row in home_results.all()}

        # Same for away teams
        query_away = (
            select(
                Match.away_team_id.label("team_id"),
                func.count(PredictionOutcome.id).label("total"),
                func.sum(
                    case((PredictionOutcome.prediction_correct == True, 1), else_=0)
                ).label("correct"),
                func.sum(
                    case((PostMatchAudit.deviation_type == "anomaly", 1), else_=0)
                ).label("anomalies"),
                func.avg(PostMatchAudit.deviation_score).label("avg_deviation"),
            )
            .select_from(PredictionOutcome)
            .join(Match, PredictionOutcome.match_id == Match.id)
            .outerjoin(PostMatchAudit, PredictionOutcome.id == PostMatchAudit.outcome_id)
            .where(PredictionOutcome.audited_at >= cutoff_date)
            .group_by(Match.away_team_id)
        )

        away_results = await self.session.execute(query_away)
        away_stats = {row.team_id: row for row in away_results.all()}

        # Merge home and away stats
        all_team_ids = set(home_stats.keys()) | set(away_stats.keys())
        adjustments_made = []

        for team_id in all_team_ids:
            home = home_stats.get(team_id)
            away = away_stats.get(team_id)

            # Safely get values with defaults
            home_total = home.total if home else 0
            away_total = away.total if away else 0
            total = home_total + away_total

            home_correct = (home.correct or 0) if home else 0
            away_correct = (away.correct or 0) if away else 0
            correct = home_correct + away_correct

            home_anomalies = (home.anomalies or 0) if home else 0
            away_anomalies = (away.anomalies or 0) if away else 0
            anomalies = home_anomalies + away_anomalies

            # Calculate avg deviation (weighted by count)
            home_dev_score = (home.avg_deviation or 0) if home else 0
            away_dev_score = (away.avg_deviation or 0) if away else 0
            home_dev = home_dev_score * home_total
            away_dev = away_dev_score * away_total
            avg_deviation = (home_dev + away_dev) / total if total > 0 else 0

            # Skip teams with too few predictions
            if total < MIN_TEAM_PREDICTIONS:
                continue

            # Calculate anomaly rate
            anomaly_rate = anomalies / total if total > 0 else 0

            # Determine confidence multiplier
            if anomaly_rate > 0.30:
                multiplier = ADJUSTMENT_SEVERE
                reason = f"High anomaly rate ({anomaly_rate:.1%})"
            elif anomaly_rate > 0.20:
                multiplier = ADJUSTMENT_MODERATE
                reason = f"Moderate anomaly rate ({anomaly_rate:.1%})"
            elif anomaly_rate > 0.10:
                multiplier = ADJUSTMENT_MILD
                reason = f"Mild anomaly rate ({anomaly_rate:.1%})"
            else:
                multiplier = 1.0
                reason = None

            # Upsert team adjustment
            stmt = pg_insert(TeamAdjustment).values(
                team_id=team_id,
                confidence_multiplier=multiplier,
                total_predictions=total,
                correct_predictions=correct or 0,
                anomaly_count=anomalies or 0,
                avg_deviation_score=avg_deviation,
                last_updated=datetime.utcnow(),
                adjustment_reason=reason,
            ).on_conflict_do_update(
                index_elements=["team_id"],
                set_={
                    "confidence_multiplier": multiplier,
                    "total_predictions": total,
                    "correct_predictions": correct or 0,
                    "anomaly_count": anomalies or 0,
                    "avg_deviation_score": avg_deviation,
                    "last_updated": datetime.utcnow(),
                    "adjustment_reason": reason,
                }
            )

            await self.session.execute(stmt)

            if multiplier != 1.0:
                # Get team name for logging
                team = await self.session.get(Team, team_id)
                team_name = team.name if team else f"Team {team_id}"
                adjustments_made.append({
                    "team_id": team_id,
                    "team_name": team_name,
                    "multiplier": multiplier,
                    "anomaly_rate": anomaly_rate,
                    "reason": reason,
                })
                logger.info(f"Adjusted {team_name}: {multiplier} ({reason})")

        await self.session.commit()

        return {
            "teams_analyzed": len(all_team_ids),
            "adjustments_made": len(adjustments_made),
            "adjustments": adjustments_made,
        }

    async def should_trigger_retrain(self, days: int = 7) -> tuple[bool, str]:
        """
        Evaluate if model retraining should be triggered.

        Checks:
        1. Gold tier accuracy < 65%
        2. Anomaly rate > 20%

        Args:
            days: Number of days to evaluate

        Returns:
            Tuple of (should_retrain, reason)
        """
        cutoff_date = datetime.utcnow() - timedelta(days=days)

        # Get recent outcomes with audits
        query = (
            select(PredictionOutcome, PostMatchAudit)
            .outerjoin(PostMatchAudit, PredictionOutcome.id == PostMatchAudit.outcome_id)
            .where(PredictionOutcome.audited_at >= cutoff_date)
        )

        result = await self.session.execute(query)
        rows = result.all()

        if len(rows) < MIN_PREDICTIONS_FOR_TRIGGER:
            return False, f"Insufficient data: {len(rows)} predictions (need {MIN_PREDICTIONS_FOR_TRIGGER})"

        # Calculate gold tier accuracy
        gold_outcomes = [r[0] for r in rows if r[0].confidence_tier == "gold"]
        gold_correct = sum(1 for o in gold_outcomes if o.prediction_correct)
        gold_accuracy = gold_correct / len(gold_outcomes) if gold_outcomes else 1.0

        # Calculate anomaly rate
        anomalies = sum(1 for r in rows if r[1] and r[1].deviation_type == "anomaly")
        anomaly_rate = anomalies / len(rows)

        # Check triggers
        if gold_accuracy < GOLD_ACCURACY_THRESHOLD:
            return True, f"Gold accuracy {gold_accuracy:.1%} < {GOLD_ACCURACY_THRESHOLD:.0%} threshold"

        if anomaly_rate > ANOMALY_RATE_THRESHOLD:
            return True, f"Anomaly rate {anomaly_rate:.1%} > {ANOMALY_RATE_THRESHOLD:.0%} threshold"

        return False, f"Metrics OK (Gold: {gold_accuracy:.1%}, Anomaly: {anomaly_rate:.1%})"

    async def validate_new_model(
        self,
        new_brier: float,
        baseline_brier: Optional[float] = None,
    ) -> tuple[bool, str]:
        """
        Validate if a new model should be deployed.

        The new model must have a Brier score better than the baseline.

        Args:
            new_brier: Brier score of the new model
            baseline_brier: Baseline to compare against (defaults to BRIER_SCORE_BASELINE)

        Returns:
            Tuple of (is_valid, message)
        """
        if baseline_brier is None:
            # Get baseline from database
            query = select(ModelSnapshot).where(ModelSnapshot.is_baseline == True)
            result = await self.session.execute(query)
            baseline_snapshot = result.scalar_one_or_none()

            if baseline_snapshot:
                baseline_brier = baseline_snapshot.brier_score
            else:
                baseline_brier = BRIER_SCORE_BASELINE

        # Validate
        if new_brier >= baseline_brier:
            return False, f"New Brier ({new_brier:.4f}) >= baseline ({baseline_brier:.4f}) - REJECTED"

        improvement = baseline_brier - new_brier
        if improvement < BRIER_SCORE_IMPROVEMENT_MIN:
            return False, f"Improvement ({improvement:.4f}) < minimum ({BRIER_SCORE_IMPROVEMENT_MIN}) - REJECTED"

        return True, f"Improvement: {baseline_brier:.4f} → {new_brier:.4f} ({improvement:.4f}) - APPROVED"

    async def create_snapshot(
        self,
        model_version: str,
        model_path: str,
        brier_score: float,
        cv_scores: list[float],
        samples_trained: int,
        training_config: Optional[dict] = None,
        is_baseline: bool = False,
    ) -> ModelSnapshot:
        """
        Create a snapshot of the current model.

        Args:
            model_version: Version string (e.g., "v1.0.0")
            model_path: Path to the saved model file
            brier_score: Cross-validation Brier score
            cv_scores: Per-fold Brier scores
            samples_trained: Number of training samples
            training_config: Hyperparameters used
            is_baseline: Whether this is the baseline model

        Returns:
            The created ModelSnapshot
        """
        # Deactivate current active snapshot
        await self.session.execute(
            ModelSnapshot.__table__.update()
            .where(ModelSnapshot.is_active == True)
            .values(is_active=False)
        )

        # Create new snapshot
        snapshot = ModelSnapshot(
            model_version=model_version,
            model_path=model_path,
            brier_score=brier_score,
            cv_brier_scores={"scores": cv_scores},
            samples_trained=samples_trained,
            is_active=True,
            is_baseline=is_baseline,
            training_config=training_config,
        )

        self.session.add(snapshot)
        await self.session.commit()
        await self.session.refresh(snapshot)

        logger.info(f"Created snapshot: {model_version} (Brier: {brier_score:.4f})")
        return snapshot

    async def get_active_snapshot(self) -> Optional[ModelSnapshot]:
        """Get the currently active model snapshot."""
        query = select(ModelSnapshot).where(ModelSnapshot.is_active == True)
        result = await self.session.execute(query)
        return result.scalar_one_or_none()

    async def get_baseline_snapshot(self) -> Optional[ModelSnapshot]:
        """Get the baseline model snapshot."""
        query = select(ModelSnapshot).where(ModelSnapshot.is_baseline == True)
        result = await self.session.execute(query)
        return result.scalar_one_or_none()

    async def rollback_to_snapshot(self, snapshot_id: int) -> bool:
        """
        Rollback to a previous model snapshot.

        Args:
            snapshot_id: ID of the snapshot to rollback to

        Returns:
            True if successful
        """
        # Get target snapshot
        snapshot = await self.session.get(ModelSnapshot, snapshot_id)
        if not snapshot:
            logger.error(f"Snapshot {snapshot_id} not found")
            return False

        # Deactivate all snapshots
        await self.session.execute(
            ModelSnapshot.__table__.update()
            .values(is_active=False)
        )

        # Activate target snapshot
        snapshot.is_active = True
        await self.session.commit()

        logger.info(f"Rolled back to snapshot {snapshot_id}: {snapshot.model_version}")
        return True

    async def get_team_adjustments(self) -> list[dict]:
        """
        Get all current team adjustments.

        Returns:
            List of team adjustments with team names
        """
        query = (
            select(TeamAdjustment, Team)
            .join(Team, TeamAdjustment.team_id == Team.id)
            .where(TeamAdjustment.confidence_multiplier != 1.0)
            .order_by(TeamAdjustment.confidence_multiplier)
        )

        result = await self.session.execute(query)
        rows = result.all()

        return [
            {
                "team_id": adj.team_id,
                "team_name": team.name,
                "confidence_multiplier": adj.confidence_multiplier,
                "total_predictions": adj.total_predictions,
                "correct_predictions": adj.correct_predictions,
                "anomaly_count": adj.anomaly_count,
                "avg_deviation_score": round(adj.avg_deviation_score, 4),
                "last_updated": adj.last_updated.isoformat(),
                "reason": adj.adjustment_reason,
            }
            for adj, team in rows
        ]

    async def get_recalibration_status(self) -> dict:
        """
        Get current recalibration status.

        Returns:
            Status dictionary with metrics and thresholds
        """
        # Get active snapshot
        active = await self.get_active_snapshot()
        baseline = await self.get_baseline_snapshot()

        # Check if retrain needed
        should_retrain, reason = await self.should_trigger_retrain()

        # Count team adjustments
        adj_query = select(func.count(TeamAdjustment.id)).where(
            TeamAdjustment.confidence_multiplier != 1.0
        )
        adj_result = await self.session.execute(adj_query)
        teams_adjusted = adj_result.scalar() or 0

        # Get recent metrics
        cutoff = datetime.utcnow() - timedelta(days=7)
        outcome_query = (
            select(PredictionOutcome)
            .where(PredictionOutcome.audited_at >= cutoff)
        )
        outcomes_result = await self.session.execute(outcome_query)
        recent_outcomes = outcomes_result.scalars().all()

        gold_outcomes = [o for o in recent_outcomes if o.confidence_tier == "gold"]
        gold_correct = sum(1 for o in gold_outcomes if o.prediction_correct)
        gold_accuracy = (gold_correct / len(gold_outcomes) * 100) if gold_outcomes else 0

        return {
            "current_model_version": active.model_version if active else "unknown",
            "baseline_brier_score": baseline.brier_score if baseline else BRIER_SCORE_BASELINE,
            "current_brier_score": active.brier_score if active else None,
            "last_retrain_date": active.created_at.isoformat() if active else None,
            "gold_accuracy_current": round(gold_accuracy, 1),
            "gold_accuracy_threshold": GOLD_ACCURACY_THRESHOLD * 100,
            "retrain_needed": should_retrain,
            "retrain_reason": reason,
            "teams_with_adjustments": teams_adjusted,
        }


async def load_team_adjustments(session: AsyncSession) -> dict[int, float]:
    """
    Load all team adjustments as a simple dictionary.

    Args:
        session: Database session

    Returns:
        Dictionary mapping team_id to confidence_multiplier
    """
    query = select(TeamAdjustment)
    result = await session.execute(query)
    adjustments = result.scalars().all()

    return {adj.team_id: adj.confidence_multiplier for adj in adjustments}
