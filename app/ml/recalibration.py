"""
Recalibration engine with contextual intelligence.

This module handles:
1. Per-team confidence adjustments with home/away split
2. Recovery factor (5 consecutive MINIMAL = conservative forgiveness)
3. International commitment look-ahead with squad depth intelligence
4. League drift detection for structural changes
5. Market movement filter for odds volatility
6. Trigger evaluation for model retraining
7. Model validation before deployment
"""

import logging
from datetime import datetime, timedelta
from typing import Optional

from sqlalchemy import func, select, and_, or_, case
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.dialects.postgresql import insert as pg_insert

from app.models import (
    Team,
    Match,
    Prediction,
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
MIN_TEAM_PREDICTIONS = 3    # Minimum predictions before adjusting team

# Recovery factor - CONSERVATIVE (Fase 2.2)
RECOVERY_THRESHOLD = 5      # 5 consecutive MINIMAL audits = forgiveness (was 3)
RECOVERY_INCREMENT = 0.01   # +0.01 per recovery step (was 0.02)

# International penalty
INTERNATIONAL_PENALTY_KNOCKOUT = 0.85   # Elimination match ±3 days
INTERNATIONAL_PENALTY_GROUP = 0.92      # Group stage ±3 days
INTERNATIONAL_PENALTY_QUALIFIED = 0.76  # Already qualified = DOUBLE penalty (0.88^2 ≈ 0.76)

# Lineup validation (Fase 3)
LINEUP_VARIANCE_THRESHOLD = 0.30   # 30% variance triggers tier degradation
LINEUP_VARIANCE_SEVERE = 0.50      # 50% variance = severe (2 tier degradation)
LINEUP_MIN_STARTERS = 7            # Minimum starters to calculate variance

# International league IDs (Champions League, Europa League, etc.)
INTERNATIONAL_LEAGUE_IDS = [2, 3, 848, 531]  # UCL, UEL, UECL, Super Cup

# League Drift Detection (Fase 2.2)
LEAGUE_DRIFT_THRESHOLD = 0.15       # 15% accuracy drop triggers alert
LEAGUE_DRIFT_MIN_MATCHES = 10       # Minimum matches to evaluate drift
LEAGUE_DRIFT_LOOKBACK_WEEKS = 8     # Historical baseline period

# Market Movement Filter (Fase 2.2)
ODDS_MOVEMENT_THRESHOLD = 0.25      # 25% odds movement triggers tier degradation
ODDS_MOVEMENT_SEVERE = 0.40         # 40% movement = severe warning


class RecalibrationEngine:
    """
    Engine for contextual model recalibration.

    Implements anticipatory intelligence for better predictions.
    """

    def __init__(self, session: AsyncSession):
        self.session = session

    # =========================================================================
    # CORE: Home/Away Split Team Adjustments
    # =========================================================================

    async def calculate_team_adjustments(self, days: int = 30) -> dict:
        """
        Calculate per-team confidence adjustments with home/away split.

        Analyzes prediction outcomes separately for home and away performances.
        Updates recovery counter for forgiveness logic.

        Args:
            days: Number of days to look back for analysis

        Returns:
            Dictionary with adjustment results
        """
        cutoff_date = datetime.utcnow() - timedelta(days=days)

        # Get HOME outcomes with their audits
        home_query = (
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

        home_results = await self.session.execute(home_query)
        home_stats = {row.team_id: row for row in home_results.all()}

        # Get AWAY outcomes with their audits
        away_query = (
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

        away_results = await self.session.execute(away_query)
        away_stats = {row.team_id: row for row in away_results.all()}

        # Process each team
        all_team_ids = set(home_stats.keys()) | set(away_stats.keys())
        adjustments_made = []

        for team_id in all_team_ids:
            home = home_stats.get(team_id)
            away = away_stats.get(team_id)

            # Get existing adjustment for recovery tracking
            existing = await self.session.get(TeamAdjustment, team_id)

            # Calculate home stats
            home_total = home.total if home else 0
            home_correct = (home.correct or 0) if home else 0
            home_anomalies = (home.anomalies or 0) if home else 0

            # Calculate away stats
            away_total = away.total if away else 0
            away_correct = (away.correct or 0) if away else 0
            away_anomalies = (away.anomalies or 0) if away else 0

            # Total stats
            total = home_total + away_total
            correct = home_correct + away_correct
            anomalies = home_anomalies + away_anomalies

            # Average deviation
            home_dev_score = (home.avg_deviation or 0) if home else 0
            away_dev_score = (away.avg_deviation or 0) if away else 0
            avg_deviation = 0
            if total > 0:
                avg_deviation = (home_dev_score * home_total + away_dev_score * away_total) / total

            # Skip teams with too few predictions
            if total < MIN_TEAM_PREDICTIONS:
                continue

            # Calculate HOME multiplier
            home_multiplier = 1.0
            home_reason = None
            if home_total >= MIN_TEAM_PREDICTIONS:
                home_anomaly_rate = home_anomalies / home_total if home_total > 0 else 0
                home_multiplier, home_reason = self._calculate_multiplier(home_anomaly_rate, "home")

            # Calculate AWAY multiplier
            away_multiplier = 1.0
            away_reason = None
            if away_total >= MIN_TEAM_PREDICTIONS:
                away_anomaly_rate = away_anomalies / away_total if away_total > 0 else 0
                away_multiplier, away_reason = self._calculate_multiplier(away_anomaly_rate, "away")

            # Apply recovery factor if consecutive MINIMALs
            consecutive_minimal = await self._count_consecutive_minimal(team_id)
            recovery_applied = False

            if consecutive_minimal >= RECOVERY_THRESHOLD:
                recovery_bonus = RECOVERY_INCREMENT * (consecutive_minimal // RECOVERY_THRESHOLD)
                if home_multiplier < 1.0:
                    home_multiplier = min(1.0, home_multiplier + recovery_bonus)
                    recovery_applied = True
                if away_multiplier < 1.0:
                    away_multiplier = min(1.0, away_multiplier + recovery_bonus)
                    recovery_applied = True

            # Combined reason
            reason_parts = []
            if home_reason:
                reason_parts.append(f"H:{home_reason}")
            if away_reason:
                reason_parts.append(f"A:{away_reason}")
            if recovery_applied:
                reason_parts.append(f"Recovery +{RECOVERY_INCREMENT * (consecutive_minimal // RECOVERY_THRESHOLD):.2f}")

            combined_reason = "; ".join(reason_parts) if reason_parts else None

            # Legacy combined multiplier (average of home/away)
            combined_multiplier = (home_multiplier + away_multiplier) / 2

            # Upsert team adjustment
            stmt = pg_insert(TeamAdjustment).values(
                team_id=team_id,
                confidence_multiplier=combined_multiplier,
                home_multiplier=home_multiplier,
                away_multiplier=away_multiplier,
                consecutive_minimal_count=consecutive_minimal if consecutive_minimal >= RECOVERY_THRESHOLD else 0,
                total_predictions=total,
                correct_predictions=correct,
                anomaly_count=anomalies,
                avg_deviation_score=avg_deviation,
                home_predictions=home_total,
                home_correct=home_correct,
                home_anomalies=home_anomalies,
                away_predictions=away_total,
                away_correct=away_correct,
                away_anomalies=away_anomalies,
                last_updated=datetime.utcnow(),
                adjustment_reason=combined_reason,
            ).on_conflict_do_update(
                index_elements=["team_id"],
                set_={
                    "confidence_multiplier": combined_multiplier,
                    "home_multiplier": home_multiplier,
                    "away_multiplier": away_multiplier,
                    "consecutive_minimal_count": consecutive_minimal if consecutive_minimal >= RECOVERY_THRESHOLD else 0,
                    "total_predictions": total,
                    "correct_predictions": correct,
                    "anomaly_count": anomalies,
                    "avg_deviation_score": avg_deviation,
                    "home_predictions": home_total,
                    "home_correct": home_correct,
                    "home_anomalies": home_anomalies,
                    "away_predictions": away_total,
                    "away_correct": away_correct,
                    "away_anomalies": away_anomalies,
                    "last_updated": datetime.utcnow(),
                    "adjustment_reason": combined_reason,
                }
            )

            await self.session.execute(stmt)

            if home_multiplier != 1.0 or away_multiplier != 1.0:
                team = await self.session.get(Team, team_id)
                team_name = team.name if team else f"Team {team_id}"
                adjustments_made.append({
                    "team_id": team_id,
                    "team_name": team_name,
                    "home_multiplier": home_multiplier,
                    "away_multiplier": away_multiplier,
                    "recovery_applied": recovery_applied,
                    "reason": combined_reason,
                })
                logger.info(f"Adjusted {team_name}: H={home_multiplier:.2f}, A={away_multiplier:.2f} ({combined_reason})")

        await self.session.commit()

        return {
            "teams_analyzed": len(all_team_ids),
            "adjustments_made": len(adjustments_made),
            "adjustments": adjustments_made,
        }

    def _calculate_multiplier(self, anomaly_rate: float, context: str) -> tuple[float, str]:
        """Calculate multiplier based on anomaly rate."""
        if anomaly_rate > 0.30:
            return ADJUSTMENT_SEVERE, f"{anomaly_rate:.0%} anomalies"
        elif anomaly_rate > 0.20:
            return ADJUSTMENT_MODERATE, f"{anomaly_rate:.0%} anomalies"
        elif anomaly_rate > 0.10:
            return ADJUSTMENT_MILD, f"{anomaly_rate:.0%} anomalies"
        return 1.0, None

    # =========================================================================
    # RECOVERY FACTOR: Forgiveness after consecutive good audits
    # =========================================================================

    async def _count_consecutive_minimal(self, team_id: int) -> int:
        """
        Count consecutive MINIMAL audits for a team (most recent first).

        Returns count of consecutive MINIMALs until first non-MINIMAL.
        """
        query = (
            select(PostMatchAudit.deviation_type)
            .join(PredictionOutcome, PostMatchAudit.outcome_id == PredictionOutcome.id)
            .join(Match, PredictionOutcome.match_id == Match.id)
            .where(
                or_(
                    Match.home_team_id == team_id,
                    Match.away_team_id == team_id
                )
            )
            .order_by(PostMatchAudit.created_at.desc())
            .limit(10)  # Check last 10 audits
        )

        result = await self.session.execute(query)
        deviation_types = [row[0] for row in result.all()]

        consecutive = 0
        for dtype in deviation_types:
            if dtype == "minimal":
                consecutive += 1
            else:
                break  # Stop at first non-MINIMAL

        return consecutive

    async def update_recovery_counters(self) -> dict:
        """
        Update recovery counters for all teams based on recent audits.

        Called after daily_audit to update consecutive_minimal_count.
        """
        # Get all teams with adjustments
        query = select(TeamAdjustment)
        result = await self.session.execute(query)
        adjustments = result.scalars().all()

        updates = []
        for adj in adjustments:
            consecutive = await self._count_consecutive_minimal(adj.team_id)

            if consecutive != adj.consecutive_minimal_count:
                adj.consecutive_minimal_count = consecutive
                adj.last_updated = datetime.utcnow()

                # Apply recovery if threshold met
                if consecutive >= RECOVERY_THRESHOLD and (adj.home_multiplier < 1.0 or adj.away_multiplier < 1.0):
                    recovery_bonus = RECOVERY_INCREMENT * (consecutive // RECOVERY_THRESHOLD)
                    new_home = min(1.0, adj.home_multiplier + recovery_bonus)
                    new_away = min(1.0, adj.away_multiplier + recovery_bonus)

                    if new_home != adj.home_multiplier or new_away != adj.away_multiplier:
                        adj.home_multiplier = new_home
                        adj.away_multiplier = new_away
                        adj.confidence_multiplier = (new_home + new_away) / 2
                        adj.adjustment_reason = f"Recovery applied: {consecutive} consecutive MINIMAL"

                        team = await self.session.get(Team, adj.team_id)
                        team_name = team.name if team else f"Team {adj.team_id}"
                        updates.append({
                            "team_id": adj.team_id,
                            "team_name": team_name,
                            "new_home_multiplier": new_home,
                            "new_away_multiplier": new_away,
                            "consecutive_minimal": consecutive,
                        })
                        logger.info(f"Recovery applied to {team_name}: H={new_home:.2f}, A={new_away:.2f}")

        await self.session.commit()

        return {
            "teams_updated": len(adjustments),
            "teams_checked": len(adjustments),
            "forgiveness_applied": len(updates),
            "recoveries_applied": len(updates),
            "updates": updates,
        }

    # =========================================================================
    # INTERNATIONAL COMMITMENT LOOK-AHEAD
    # =========================================================================

    async def check_international_commitments(
        self,
        team_id: int,
        match_date: datetime,
        days_window: int = 3,
    ) -> dict:
        """
        Check if team has international commitments within ±days of match.

        Returns penalty factor and context.
        """
        start_date = match_date - timedelta(days=days_window)
        end_date = match_date + timedelta(days=days_window)

        # Find international matches for this team
        query = (
            select(Match)
            .where(
                and_(
                    or_(
                        Match.home_team_id == team_id,
                        Match.away_team_id == team_id
                    ),
                    Match.league_id.in_(INTERNATIONAL_LEAGUE_IDS),
                    Match.date >= start_date,
                    Match.date <= end_date,
                    Match.date != match_date,  # Exclude the match itself
                )
            )
            .order_by(Match.date)
        )

        result = await self.session.execute(query)
        international_matches = result.scalars().all()

        if not international_matches:
            return {
                "has_commitment": False,
                "penalty": 1.0,
                "reason": None,
            }

        # Determine penalty based on match importance
        # For now, treat all international matches equally
        # Could be enhanced with knockout stage detection
        penalty = INTERNATIONAL_PENALTY_GROUP
        reason = f"International match within {days_window} days"

        return {
            "has_commitment": True,
            "penalty": penalty,
            "reason": reason,
            "matches": [
                {
                    "id": m.id,
                    "date": m.date.isoformat(),
                    "league_id": m.league_id,
                }
                for m in international_matches
            ],
        }

    async def apply_international_penalties(self, days_ahead: int = 7) -> dict:
        """
        Apply international penalties to upcoming matches.

        Updates international_penalty field in team_adjustments.

        Args:
            days_ahead: How many days ahead to look for matches
        """
        now = datetime.utcnow()
        upcoming_window = now + timedelta(days=days_ahead)

        # Get upcoming domestic matches
        query = (
            select(Match)
            .where(
                and_(
                    Match.date >= now,
                    Match.date <= upcoming_window,
                    Match.status == "NS",
                    ~Match.league_id.in_(INTERNATIONAL_LEAGUE_IDS),
                )
            )
        )

        result = await self.session.execute(query)
        upcoming_matches = result.scalars().all()

        penalties_applied = []

        for match in upcoming_matches:
            # Check home team
            home_commitment = await self.check_international_commitments(
                match.home_team_id, match.date
            )
            if home_commitment["has_commitment"]:
                await self._update_international_penalty(
                    match.home_team_id,
                    home_commitment["penalty"],
                    home_commitment["reason"],
                )
                penalties_applied.append({
                    "team_id": match.home_team_id,
                    "match_id": match.id,
                    "penalty": home_commitment["penalty"],
                })

            # Check away team
            away_commitment = await self.check_international_commitments(
                match.away_team_id, match.date
            )
            if away_commitment["has_commitment"]:
                await self._update_international_penalty(
                    match.away_team_id,
                    away_commitment["penalty"],
                    away_commitment["reason"],
                )
                penalties_applied.append({
                    "team_id": match.away_team_id,
                    "match_id": match.id,
                    "penalty": away_commitment["penalty"],
                })

        await self.session.commit()

        # Count unique teams checked
        teams_checked = set()
        for match in upcoming_matches:
            teams_checked.add(match.home_team_id)
            teams_checked.add(match.away_team_id)

        return {
            "matches_checked": len(upcoming_matches),
            "teams_checked": len(teams_checked),
            "penalties_applied": len(penalties_applied),
            "details": penalties_applied,
        }

    async def _update_international_penalty(
        self,
        team_id: int,
        penalty: float,
        reason: str,
    ):
        """Update international penalty for a team."""
        stmt = pg_insert(TeamAdjustment).values(
            team_id=team_id,
            international_penalty=penalty,
            last_updated=datetime.utcnow(),
        ).on_conflict_do_update(
            index_elements=["team_id"],
            set_={
                "international_penalty": penalty,
                "last_updated": datetime.utcnow(),
            }
        )
        await self.session.execute(stmt)

    # =========================================================================
    # LEAGUE DRIFT DETECTION (Fase 2.2)
    # =========================================================================

    async def detect_league_drift(self) -> dict:
        """
        Detect structural changes in league prediction accuracy.

        Compares weekly GOLD accuracy per league against historical baseline.
        If accuracy drops 15%+ from baseline, marks league as 'Unstable'.

        Returns:
            Dictionary with drift analysis per league
        """
        now = datetime.utcnow()
        week_ago = now - timedelta(days=7)
        historical_cutoff = now - timedelta(weeks=LEAGUE_DRIFT_LOOKBACK_WEEKS)

        # Get recent week's outcomes by league
        recent_query = (
            select(
                Match.league_id,
                func.count(PredictionOutcome.id).label("total"),
                func.sum(
                    case((PredictionOutcome.prediction_correct == True, 1), else_=0)
                ).label("correct"),
            )
            .select_from(PredictionOutcome)
            .join(Match, PredictionOutcome.match_id == Match.id)
            .where(
                and_(
                    PredictionOutcome.audited_at >= week_ago,
                    PredictionOutcome.confidence_tier == "gold",
                )
            )
            .group_by(Match.league_id)
        )

        recent_result = await self.session.execute(recent_query)
        recent_stats = {row.league_id: row for row in recent_result.all()}

        # Get historical baseline by league
        historical_query = (
            select(
                Match.league_id,
                func.count(PredictionOutcome.id).label("total"),
                func.sum(
                    case((PredictionOutcome.prediction_correct == True, 1), else_=0)
                ).label("correct"),
            )
            .select_from(PredictionOutcome)
            .join(Match, PredictionOutcome.match_id == Match.id)
            .where(
                and_(
                    PredictionOutcome.audited_at >= historical_cutoff,
                    PredictionOutcome.audited_at < week_ago,
                    PredictionOutcome.confidence_tier == "gold",
                )
            )
            .group_by(Match.league_id)
        )

        historical_result = await self.session.execute(historical_query)
        historical_stats = {row.league_id: row for row in historical_result.all()}

        # Analyze drift per league
        drift_alerts = []
        stable_leagues = []

        for league_id, recent in recent_stats.items():
            if recent.total < LEAGUE_DRIFT_MIN_MATCHES:
                continue  # Not enough data

            recent_accuracy = recent.correct / recent.total if recent.total > 0 else 0
            historical = historical_stats.get(league_id)

            if historical and historical.total >= LEAGUE_DRIFT_MIN_MATCHES:
                historical_accuracy = historical.correct / historical.total
                drift = historical_accuracy - recent_accuracy

                if drift >= LEAGUE_DRIFT_THRESHOLD:
                    drift_alerts.append({
                        "league_id": league_id,
                        "status": "unstable",
                        "recent_accuracy": round(recent_accuracy * 100, 1),
                        "historical_accuracy": round(historical_accuracy * 100, 1),
                        "drift_percentage": round(drift * 100, 1),
                        "recent_matches": recent.total,
                        "insight": f"ALERT: {drift*100:.1f}% accuracy drop detected",
                    })
                    logger.warning(
                        f"League {league_id} DRIFT DETECTED: "
                        f"{historical_accuracy:.1%} → {recent_accuracy:.1%} "
                        f"(drop: {drift:.1%})"
                    )
                else:
                    stable_leagues.append({
                        "league_id": league_id,
                        "status": "stable",
                        "recent_accuracy": round(recent_accuracy * 100, 1),
                        "historical_accuracy": round(historical_accuracy * 100, 1),
                    })

        return {
            "leagues_analyzed": len(recent_stats),
            "unstable_leagues": len(drift_alerts),
            "stable_leagues": len(stable_leagues),
            "drift_alerts": drift_alerts,
            "details": stable_leagues,
        }

    async def get_unstable_leagues(self) -> set[int]:
        """
        Get set of league IDs currently marked as unstable.

        Used by prediction endpoints to degrade tier for matches in these leagues.
        """
        drift_result = await self.detect_league_drift()
        return {alert["league_id"] for alert in drift_result["drift_alerts"]}

    # =========================================================================
    # MARKET MOVEMENT FILTER (Fase 2.2)
    # =========================================================================

    async def check_odds_movement(self, match_id: int) -> dict:
        """
        Check if market odds have moved significantly since prediction was made.

        Compares current match odds with odds at prediction time.
        Movement of 25%+ triggers tier degradation.

        Args:
            match_id: The match to check

        Returns:
            Dictionary with movement analysis and tier adjustment recommendation
        """
        # Get the prediction and current match data
        pred_query = (
            select(Prediction, Match)
            .join(Match, Prediction.match_id == Match.id)
            .where(Prediction.match_id == match_id)
            .order_by(Prediction.created_at.desc())
            .limit(1)
        )

        result = await self.session.execute(pred_query)
        row = result.first()

        if not row:
            return {"has_movement": False, "reason": "No prediction found"}

        prediction, match = row

        # Get current market odds from match
        if not match.odds_home or not match.odds_draw or not match.odds_away:
            return {"has_movement": False, "reason": "No market odds available"}

        # Calculate implied probabilities from current odds
        current_implied_home = 1 / match.odds_home
        current_implied_draw = 1 / match.odds_draw
        current_implied_away = 1 / match.odds_away

        # Our fair odds at prediction time
        fair_home = 1 / prediction.home_prob if prediction.home_prob > 0 else 999
        fair_draw = 1 / prediction.draw_prob if prediction.draw_prob > 0 else 999
        fair_away = 1 / prediction.away_prob if prediction.away_prob > 0 else 999

        # Calculate movement for each outcome
        movements = []

        # Home odds movement
        if fair_home < 999:
            home_movement = abs(match.odds_home - fair_home) / fair_home
            movements.append(("home", home_movement, match.odds_home, fair_home))

        # Draw odds movement
        if fair_draw < 999:
            draw_movement = abs(match.odds_draw - fair_draw) / fair_draw
            movements.append(("draw", draw_movement, match.odds_draw, fair_draw))

        # Away odds movement
        if fair_away < 999:
            away_movement = abs(match.odds_away - fair_away) / fair_away
            movements.append(("away", away_movement, match.odds_away, fair_away))

        # Find max movement
        if not movements:
            return {"has_movement": False, "reason": "Could not calculate movement"}

        max_movement = max(movements, key=lambda x: x[1])
        outcome, movement_pct, current_odds, fair_odds = max_movement

        result = {
            "match_id": match_id,
            "has_movement": False,
            "max_movement_outcome": outcome,
            "movement_percentage": round(movement_pct * 100, 1),
            "current_odds": current_odds,
            "fair_odds_at_prediction": round(fair_odds, 2),
            "tier_degradation": None,
            "insight": None,
        }

        # Check thresholds
        if movement_pct >= ODDS_MOVEMENT_SEVERE:
            result["has_movement"] = True
            result["tier_degradation"] = 2  # GOLD → COPPER
            result["insight"] = (
                f"SEVERE: {movement_pct*100:.1f}% market movement on {outcome}. "
                f"Odds: {fair_odds:.2f} → {current_odds:.2f}. "
                f"High risk detected - double tier degradation recommended."
            )
            logger.warning(f"Match {match_id}: SEVERE odds movement ({movement_pct:.1%})")

        elif movement_pct >= ODDS_MOVEMENT_THRESHOLD:
            result["has_movement"] = True
            result["tier_degradation"] = 1  # GOLD → SILVER or SILVER → COPPER
            result["insight"] = (
                f"WARNING: {movement_pct*100:.1f}% market movement on {outcome}. "
                f"Odds: {fair_odds:.2f} → {current_odds:.2f}. "
                f"Unusual activity detected - tier degradation recommended."
            )
            logger.info(f"Match {match_id}: Significant odds movement ({movement_pct:.1%})")

        return result

    async def check_all_upcoming_odds_movements(self, days_ahead: int = 3) -> dict:
        """
        Check odds movements for all upcoming matches.

        Returns summary of matches with significant market movement.
        """
        now = datetime.utcnow()
        cutoff = now + timedelta(days=days_ahead)

        # Get upcoming matches with predictions
        query = (
            select(Prediction.match_id)
            .join(Match, Prediction.match_id == Match.id)
            .where(
                and_(
                    Match.date >= now,
                    Match.date <= cutoff,
                    Match.status == "NS",
                )
            )
            .distinct()
        )

        result = await self.session.execute(query)
        match_ids = [row[0] for row in result.all()]

        alerts = []
        for match_id in match_ids:
            movement = await self.check_odds_movement(match_id)
            if movement["has_movement"]:
                alerts.append(movement)

        return {
            "matches_checked": len(match_ids),
            "movements_detected": len(alerts),
            "alerts": alerts,
        }

    def degrade_tier(self, current_tier: str, levels: int = 1) -> str:
        """
        Degrade confidence tier by specified levels.

        Args:
            current_tier: Current tier (gold, silver, copper)
            levels: Number of levels to degrade

        Returns:
            New tier after degradation
        """
        tier_order = ["gold", "silver", "copper"]

        try:
            current_idx = tier_order.index(current_tier.lower())
            new_idx = min(current_idx + levels, len(tier_order) - 1)
            return tier_order[new_idx]
        except ValueError:
            return "copper"  # Default to lowest tier if unknown

    # =========================================================================
    # LINEUP VALIDATION (Fase 3)
    # =========================================================================

    async def validate_lineup(
        self,
        match_id: int,
        lineup_data: dict,
        expected_starters: dict,
    ) -> dict:
        """
        Validate starting XI against expected best XI ("Equipo de Gala").

        Compares announced lineup with historical best starters for each team.
        If variance exceeds threshold, triggers tier degradation.

        Args:
            match_id: The match ID
            lineup_data: Dict with 'home' and 'away' lineup data from API
            expected_starters: Dict with 'home' and 'away' expected player IDs

        Returns:
            Dictionary with validation results and tier degradation recommendations
        """
        result = {
            "match_id": match_id,
            "home_validation": None,
            "away_validation": None,
            "tier_degradation": 0,
            "warnings": [],
            "insights": [],
        }

        # Validate home lineup
        if lineup_data.get("home") and expected_starters.get("home"):
            home_result = self._calculate_lineup_variance(
                lineup_data["home"],
                expected_starters["home"],
                "home",
            )
            result["home_validation"] = home_result

            if home_result["variance"] >= LINEUP_VARIANCE_SEVERE:
                result["tier_degradation"] = max(result["tier_degradation"], 2)
                result["warnings"].append("LINEUP_ROTATION_SEVERE_HOME")
                result["insights"].append(
                    f"{lineup_data['home'].get('team_name', 'Local')}: "
                    f"Rotación masiva {home_result['variance']:.0%} ({home_result['missing_starters']} titulares ausentes)"
                )
            elif home_result["variance"] >= LINEUP_VARIANCE_THRESHOLD:
                result["tier_degradation"] = max(result["tier_degradation"], 1)
                result["warnings"].append("LINEUP_ROTATION_HOME")
                result["insights"].append(
                    f"{lineup_data['home'].get('team_name', 'Local')}: "
                    f"Rotación significativa {home_result['variance']:.0%}"
                )

        # Validate away lineup
        if lineup_data.get("away") and expected_starters.get("away"):
            away_result = self._calculate_lineup_variance(
                lineup_data["away"],
                expected_starters["away"],
                "away",
            )
            result["away_validation"] = away_result

            if away_result["variance"] >= LINEUP_VARIANCE_SEVERE:
                result["tier_degradation"] = max(result["tier_degradation"], 2)
                result["warnings"].append("LINEUP_ROTATION_SEVERE_AWAY")
                result["insights"].append(
                    f"{lineup_data['away'].get('team_name', 'Visitante')}: "
                    f"Rotación masiva {away_result['variance']:.0%} ({away_result['missing_starters']} titulares ausentes)"
                )
            elif away_result["variance"] >= LINEUP_VARIANCE_THRESHOLD:
                result["tier_degradation"] = max(result["tier_degradation"], 1)
                result["warnings"].append("LINEUP_ROTATION_AWAY")
                result["insights"].append(
                    f"{lineup_data['away'].get('team_name', 'Visitante')}: "
                    f"Rotación significativa {away_result['variance']:.0%}"
                )

        return result

    def _calculate_lineup_variance(
        self,
        lineup: dict,
        expected_player_ids: set[int],
        context: str,
    ) -> dict:
        """
        Calculate variance between announced lineup and expected starters.

        Args:
            lineup: Lineup data with starting_xi
            expected_player_ids: Set of expected starter player IDs
            context: 'home' or 'away' for logging

        Returns:
            Dictionary with variance metrics
        """
        starting_xi = lineup.get("starting_xi", [])
        actual_ids = {p["id"] for p in starting_xi if p.get("id")}

        if len(actual_ids) < LINEUP_MIN_STARTERS:
            return {
                "variance": 0,
                "missing_starters": 0,
                "unexpected_starters": 0,
                "reason": "Insufficient lineup data",
            }

        # Calculate how many expected starters are missing
        missing = expected_player_ids - actual_ids
        unexpected = actual_ids - expected_player_ids

        # Variance = % of expected starters not in lineup
        variance = len(missing) / len(expected_player_ids) if expected_player_ids else 0

        return {
            "variance": variance,
            "missing_starters": len(missing),
            "unexpected_starters": len(unexpected),
            "missing_ids": list(missing),
            "formation": lineup.get("formation"),
        }

    async def get_expected_starters(self, team_id: int, num_matches: int = 5) -> set[int]:
        """
        Determine expected best XI based on recent match history.

        Analyzes last N matches to identify most frequently started players.
        Players who started in >60% of matches are considered expected starters.

        Args:
            team_id: Team ID to analyze
            num_matches: Number of recent matches to analyze

        Returns:
            Set of player IDs that form the expected best XI
        """
        # This would require storing lineup history in the database
        # For now, return empty set - will be populated when lineups are tracked
        # In production, query stored lineups and count appearances
        return set()

    async def check_lineup_for_match(self, match_external_id: int) -> dict:
        """
        Check lineup status for a specific match.

        Fetches lineup from API (if available ~60min before kickoff),
        compares with expected starters, and returns validation result.

        Args:
            match_external_id: External match ID for API lookup

        Returns:
            Dictionary with lineup validation or status if not yet available
        """
        from app.etl.api_football import APIFootballProvider

        provider = APIFootballProvider()
        try:
            lineup_data = await provider.get_lineups(match_external_id)

            if not lineup_data or not lineup_data.get("home"):
                return {
                    "available": False,
                    "reason": "Lineups not yet announced (typically available ~60min before kickoff)",
                }

            # Get match from database to find team IDs
            query = select(Match).where(Match.external_id == match_external_id)
            result = await self.session.execute(query)
            match = result.scalar_one_or_none()

            if not match:
                return {"available": False, "reason": "Match not found in database"}

            # Get expected starters for both teams
            home_expected = await self.get_expected_starters(match.home_team_id)
            away_expected = await self.get_expected_starters(match.away_team_id)

            # Validate lineups
            validation = await self.validate_lineup(
                match_id=match.id,
                lineup_data=lineup_data,
                expected_starters={
                    "home": home_expected,
                    "away": away_expected,
                },
            )

            validation["available"] = True
            validation["lineup_data"] = {
                "home": {
                    "team_name": lineup_data["home"].get("team_name"),
                    "formation": lineup_data["home"].get("formation"),
                    "starters_count": len(lineup_data["home"].get("starting_xi", [])),
                },
                "away": {
                    "team_name": lineup_data["away"].get("team_name"),
                    "formation": lineup_data["away"].get("formation"),
                    "starters_count": len(lineup_data["away"].get("starting_xi", [])),
                },
            }

            return validation

        finally:
            await provider.close()

    # =========================================================================
    # RETRAIN TRIGGERS AND VALIDATION
    # =========================================================================

    async def should_trigger_retrain(self, days: int = 7) -> tuple[bool, str]:
        """
        Evaluate if model retraining should be triggered.
        """
        cutoff_date = datetime.utcnow() - timedelta(days=days)

        query = (
            select(PredictionOutcome, PostMatchAudit)
            .outerjoin(PostMatchAudit, PredictionOutcome.id == PostMatchAudit.outcome_id)
            .where(PredictionOutcome.audited_at >= cutoff_date)
        )

        result = await self.session.execute(query)
        rows = result.all()

        if len(rows) < MIN_PREDICTIONS_FOR_TRIGGER:
            return False, f"Insufficient data: {len(rows)} predictions (need {MIN_PREDICTIONS_FOR_TRIGGER})"

        gold_outcomes = [r[0] for r in rows if r[0].confidence_tier == "gold"]
        gold_correct = sum(1 for o in gold_outcomes if o.prediction_correct)
        gold_accuracy = gold_correct / len(gold_outcomes) if gold_outcomes else 1.0

        anomalies = sum(1 for r in rows if r[1] and r[1].deviation_type == "anomaly")
        anomaly_rate = anomalies / len(rows)

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
        """Validate if a new model should be deployed."""
        if baseline_brier is None:
            query = select(ModelSnapshot).where(ModelSnapshot.is_baseline == True)
            result = await self.session.execute(query)
            baseline_snapshot = result.scalar_one_or_none()
            baseline_brier = baseline_snapshot.brier_score if baseline_snapshot else BRIER_SCORE_BASELINE

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
        """Create a snapshot of the current model."""
        await self.session.execute(
            ModelSnapshot.__table__.update()
            .where(ModelSnapshot.is_active == True)
            .values(is_active=False)
        )

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
        """Rollback to a previous model snapshot."""
        snapshot = await self.session.get(ModelSnapshot, snapshot_id)
        if not snapshot:
            logger.error(f"Snapshot {snapshot_id} not found")
            return False

        await self.session.execute(
            ModelSnapshot.__table__.update().values(is_active=False)
        )

        snapshot.is_active = True
        await self.session.commit()

        logger.info(f"Rolled back to snapshot {snapshot_id}: {snapshot.model_version}")
        return True

    async def get_team_adjustments(self) -> list[dict]:
        """Get all current team adjustments with home/away split."""
        query = (
            select(TeamAdjustment, Team)
            .join(Team, TeamAdjustment.team_id == Team.id)
            .where(
                or_(
                    TeamAdjustment.home_multiplier != 1.0,
                    TeamAdjustment.away_multiplier != 1.0,
                )
            )
            .order_by(TeamAdjustment.confidence_multiplier)
        )

        result = await self.session.execute(query)
        rows = result.all()

        return [
            {
                "team_id": adj.team_id,
                "team_name": team.name,
                "home_multiplier": adj.home_multiplier,
                "away_multiplier": adj.away_multiplier,
                "confidence_multiplier": adj.confidence_multiplier,
                "international_penalty": adj.international_penalty,
                "consecutive_minimal": adj.consecutive_minimal_count,
                "total_predictions": adj.total_predictions,
                "correct_predictions": adj.correct_predictions,
                "anomaly_count": adj.anomaly_count,
                "home_predictions": adj.home_predictions,
                "home_anomalies": adj.home_anomalies,
                "away_predictions": adj.away_predictions,
                "away_anomalies": adj.away_anomalies,
                "avg_deviation_score": round(adj.avg_deviation_score, 4),
                "last_updated": adj.last_updated.isoformat(),
                "reason": adj.adjustment_reason,
            }
            for adj, team in rows
        ]

    async def get_recalibration_status(self) -> dict:
        """Get current recalibration status."""
        active = await self.get_active_snapshot()
        baseline = await self.get_baseline_snapshot()

        should_retrain, reason = await self.should_trigger_retrain()

        adj_query = select(func.count(TeamAdjustment.id)).where(
            or_(
                TeamAdjustment.home_multiplier != 1.0,
                TeamAdjustment.away_multiplier != 1.0,
            )
        )
        adj_result = await self.session.execute(adj_query)
        teams_adjusted = adj_result.scalar() or 0

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


# =========================================================================
# HELPER FUNCTIONS
# =========================================================================

async def load_team_adjustments(session: AsyncSession) -> dict:
    """
    Load all team adjustments as home/away dictionaries.

    Returns:
        Dictionary with 'home' and 'away' keys, each mapping team_id to multiplier
    """
    query = select(TeamAdjustment)
    result = await session.execute(query)
    adjustments = result.scalars().all()

    return {
        "home": {adj.team_id: adj.home_multiplier * adj.international_penalty for adj in adjustments},
        "away": {adj.team_id: adj.away_multiplier * adj.international_penalty for adj in adjustments},
    }
