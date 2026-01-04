"""
Post-match audit service for analyzing prediction outcomes.

This service:
1. Fetches completed matches that have predictions
2. Retrieves detailed stats from API-Football (xG, events, etc.)
3. Records outcomes and classifies deviations
4. Generates insights for model improvement
"""

import logging
from datetime import datetime, timedelta
from typing import Optional

from sqlalchemy import select, and_
from sqlalchemy.ext.asyncio import AsyncSession

from app.etl.api_football import APIFootballProvider
from app.models import (
    Match,
    Prediction,
    PredictionOutcome,
    PostMatchAudit,
)

logger = logging.getLogger(__name__)


# Confidence tier thresholds
GOLD_THRESHOLD = 0.55  # >= 55% confidence
SILVER_THRESHOLD = 0.45  # >= 45% confidence
# Below 45% is Copper


class PostMatchAuditService:
    """Service for auditing prediction outcomes after matches complete."""

    def __init__(self, session: AsyncSession):
        self.session = session
        self.provider = APIFootballProvider()

    async def close(self):
        """Close API client."""
        await self.provider.close()

    def _get_confidence_tier(self, confidence: float) -> str:
        """Determine confidence tier based on probability."""
        if confidence >= GOLD_THRESHOLD:
            return "gold"
        elif confidence >= SILVER_THRESHOLD:
            return "silver"
        return "copper"

    def _get_predicted_result(self, prediction: Prediction) -> tuple[str, float]:
        """Get the predicted result and confidence from a prediction."""
        probs = {
            "home": prediction.home_prob,
            "draw": prediction.draw_prob,
            "away": prediction.away_prob,
        }
        predicted = max(probs, key=probs.get)
        confidence = probs[predicted]
        return predicted, confidence

    def _get_actual_result(self, home_goals: int, away_goals: int) -> str:
        """Determine actual result from goals."""
        if home_goals > away_goals:
            return "home"
        elif home_goals < away_goals:
            return "away"
        return "draw"

    def _get_xg_result(self, xg_home: Optional[float], xg_away: Optional[float]) -> Optional[str]:
        """Determine what result xG would predict."""
        if xg_home is None or xg_away is None:
            return None
        diff = xg_home - xg_away
        if diff > 0.3:  # Significant xG advantage
            return "home"
        elif diff < -0.3:
            return "away"
        return "draw"

    def _calculate_deviation_score(
        self,
        prediction: Prediction,
        actual_result: str,
        xg_home: Optional[float],
        xg_away: Optional[float],
    ) -> tuple[str, float]:
        """
        Calculate deviation type and score.

        Returns:
            tuple: (deviation_type, deviation_score)
            - deviation_type: 'minimal', 'expected', or 'anomaly'
            - deviation_score: 0-1 where 1 is maximum deviation
        """
        predicted_result, confidence = self._get_predicted_result(prediction)
        is_correct = predicted_result == actual_result

        # Get probability assigned to actual result
        actual_prob = getattr(prediction, f"{actual_result}_prob")

        if is_correct:
            # Correct prediction
            if confidence >= GOLD_THRESHOLD:
                return "minimal", 0.0  # High confidence, correct
            else:
                return "minimal", 0.2  # Low confidence, but still correct

        # Incorrect prediction - analyze why
        xg_result = self._get_xg_result(xg_home, xg_away)

        # Score based on how unlikely the actual result was
        deviation_score = 1.0 - actual_prob  # Higher score = more unexpected

        # Check if xG aligned with actual result (explains the deviation)
        if xg_result == actual_result and xg_result != predicted_result:
            # xG predicted correctly but we didn't - expected deviation
            return "expected", min(deviation_score, 0.6)

        # High confidence wrong prediction = anomaly
        if confidence >= GOLD_THRESHOLD and deviation_score > 0.6:
            return "anomaly", deviation_score

        # Low confidence wrong prediction = expected
        if confidence < SILVER_THRESHOLD:
            return "expected", deviation_score * 0.8

        return "expected", deviation_score

    def _identify_primary_factor(
        self,
        had_red_card: bool,
        had_penalty: bool,
        had_var_decision: bool,
        xg_home: Optional[float],
        xg_away: Optional[float],
        actual_home_goals: int,
        actual_away_goals: int,
        predicted_result: str,
        actual_result: str,
    ) -> Optional[str]:
        """Identify the primary factor explaining a deviation."""
        if predicted_result == actual_result:
            return None  # No deviation to explain

        # Check disruption factors in order of impact
        if had_red_card:
            return "red_card"

        if had_var_decision and had_penalty:
            return "var_penalty"

        if had_penalty:
            return "penalty"

        # Check xG mismatch
        if xg_home is not None and xg_away is not None:
            xg_result = self._get_xg_result(xg_home, xg_away)
            if xg_result == actual_result:
                # xG aligned with result, we just predicted wrong
                return "model_error"

            # xG didn't match result either - finishing quality issue
            home_diff = actual_home_goals - xg_home
            away_diff = actual_away_goals - xg_away

            if abs(home_diff) > 1.5 or abs(away_diff) > 1.5:
                return "finishing_variance"

        return "unknown"

    async def audit_match(self, match: Match, prediction: Prediction) -> Optional[PredictionOutcome]:
        """
        Audit a single match's prediction.

        Args:
            match: The completed match
            prediction: The prediction made for this match

        Returns:
            PredictionOutcome if created, None if already audited
        """
        # Check if already audited
        existing = await self.session.execute(
            select(PredictionOutcome).where(PredictionOutcome.prediction_id == prediction.id)
        )
        if existing.scalar_one_or_none():
            logger.debug(f"Match {match.id} already audited")
            return None

        # Fetch detailed stats from API
        stats = await self.provider.get_fixture_statistics(match.external_id)
        events = await self._fetch_events(match.external_id)

        # Parse stats
        xg_home = None
        xg_away = None
        home_possession = None
        total_shots_home = None
        total_shots_away = None
        shots_on_target_home = None
        shots_on_target_away = None

        if stats:
            home_stats = stats.get("home", {})
            away_stats = stats.get("away", {})

            xg_home = self._parse_float(home_stats.get("expected_goals"))
            xg_away = self._parse_float(away_stats.get("expected_goals"))
            home_possession = self._parse_float(home_stats.get("ball_possession", "").replace("%", ""))
            total_shots_home = self._parse_int(home_stats.get("total_shots"))
            total_shots_away = self._parse_int(away_stats.get("total_shots"))
            shots_on_target_home = self._parse_int(home_stats.get("shots_on_goal"))
            shots_on_target_away = self._parse_int(away_stats.get("shots_on_goal"))

        # Parse events for disruption factors
        had_red_card = False
        had_penalty = False
        had_var_decision = False
        red_card_minute = None

        for event in events:
            event_type = event.get("type", "").lower()
            detail = event.get("detail", "").lower()

            if event_type == "card" and "red" in detail:
                had_red_card = True
                if red_card_minute is None:
                    red_card_minute = event.get("time", {}).get("elapsed")

            if event_type == "goal" and "penalty" in detail:
                had_penalty = True

            if event_type == "var":
                had_var_decision = True

        # Calculate results
        predicted_result, confidence = self._get_predicted_result(prediction)
        actual_result = self._get_actual_result(match.home_goals, match.away_goals)
        is_correct = predicted_result == actual_result
        confidence_tier = self._get_confidence_tier(confidence)

        # Create outcome
        outcome = PredictionOutcome(
            prediction_id=prediction.id,
            match_id=match.id,
            actual_result=actual_result,
            actual_home_goals=match.home_goals,
            actual_away_goals=match.away_goals,
            predicted_result=predicted_result,
            prediction_correct=is_correct,
            confidence=confidence,
            confidence_tier=confidence_tier,
            xg_home=xg_home,
            xg_away=xg_away,
            xg_diff=(xg_home - xg_away) if xg_home and xg_away else None,
            had_red_card=had_red_card,
            had_penalty=had_penalty,
            had_var_decision=had_var_decision,
            red_card_minute=red_card_minute,
            home_possession=home_possession,
            total_shots_home=total_shots_home,
            total_shots_away=total_shots_away,
            shots_on_target_home=shots_on_target_home,
            shots_on_target_away=shots_on_target_away,
        )

        self.session.add(outcome)
        await self.session.flush()  # Get the ID

        # Create audit record
        deviation_type, deviation_score = self._calculate_deviation_score(
            prediction, actual_result, xg_home, xg_away
        )

        xg_result = self._get_xg_result(xg_home, xg_away)
        xg_result_aligned = xg_result == actual_result if xg_result else False
        xg_prediction_aligned = xg_result == predicted_result if xg_result else False

        primary_factor = self._identify_primary_factor(
            had_red_card,
            had_penalty,
            had_var_decision,
            xg_home,
            xg_away,
            match.home_goals,
            match.away_goals,
            predicted_result,
            actual_result,
        )

        # Secondary factors
        secondary_factors = {}
        if had_red_card and primary_factor != "red_card":
            secondary_factors["red_card"] = True
        if had_penalty and primary_factor != "penalty":
            secondary_factors["penalty"] = True
        if had_var_decision and primary_factor != "var_penalty":
            secondary_factors["var"] = True

        audit = PostMatchAudit(
            outcome_id=outcome.id,
            deviation_type=deviation_type,
            deviation_score=deviation_score,
            primary_factor=primary_factor,
            secondary_factors=secondary_factors if secondary_factors else None,
            xg_result_aligned=xg_result_aligned,
            xg_prediction_aligned=xg_prediction_aligned,
            goals_vs_xg_home=(match.home_goals - xg_home) if xg_home else None,
            goals_vs_xg_away=(match.away_goals - xg_away) if xg_away else None,
            should_adjust_model=deviation_type == "anomaly",
            adjustment_notes=f"Anomaly: predicted {predicted_result}, actual {actual_result}" if deviation_type == "anomaly" else None,
        )

        self.session.add(audit)
        logger.info(
            f"Audited match {match.id}: {predicted_result} vs {actual_result} "
            f"({'✓' if is_correct else '✗'}) - {deviation_type}"
        )

        return outcome

    async def _fetch_events(self, fixture_id: int) -> list:
        """Fetch match events from API."""
        data = await self.provider._rate_limited_request(
            "fixtures/events", {"fixture": fixture_id}
        )
        return data.get("response", [])

    def _parse_float(self, value) -> Optional[float]:
        """Safely parse a float value."""
        if value is None:
            return None
        try:
            return float(value)
        except (ValueError, TypeError):
            return None

    def _parse_int(self, value) -> Optional[int]:
        """Safely parse an int value."""
        if value is None:
            return None
        try:
            return int(value)
        except (ValueError, TypeError):
            return None

    async def audit_recent_matches(self, days: int = 7) -> dict:
        """
        Audit all completed matches from the last N days that have predictions.

        Args:
            days: Number of days to look back

        Returns:
            Summary of audit results
        """
        cutoff_date = datetime.utcnow() - timedelta(days=days)

        # Find completed matches with predictions that haven't been audited
        query = (
            select(Match, Prediction)
            .join(Prediction, Match.id == Prediction.match_id)
            .outerjoin(PredictionOutcome, Prediction.id == PredictionOutcome.prediction_id)
            .where(
                and_(
                    Match.status == "FT",
                    Match.date >= cutoff_date,
                    Match.home_goals.isnot(None),
                    Match.away_goals.isnot(None),
                    PredictionOutcome.id.is_(None),  # Not yet audited
                )
            )
        )

        result = await self.session.execute(query)
        matches_to_audit = result.all()

        logger.info(f"Found {len(matches_to_audit)} matches to audit from last {days} days")

        audited = 0
        correct = 0
        anomalies = 0

        for match, prediction in matches_to_audit:
            try:
                outcome = await self.audit_match(match, prediction)
                if outcome:
                    audited += 1
                    if outcome.prediction_correct:
                        correct += 1
                    # Check for anomaly
                    audit_result = await self.session.execute(
                        select(PostMatchAudit).where(PostMatchAudit.outcome_id == outcome.id)
                    )
                    audit = audit_result.scalar_one_or_none()
                    if audit and audit.deviation_type == "anomaly":
                        anomalies += 1
            except Exception as e:
                logger.error(f"Error auditing match {match.id}: {e}")
                continue

        await self.session.commit()

        accuracy = (correct / audited * 100) if audited > 0 else 0

        return {
            "matches_audited": audited,
            "correct_predictions": correct,
            "accuracy": round(accuracy, 2),
            "anomalies_detected": anomalies,
            "period_days": days,
        }


async def create_audit_service(session: AsyncSession) -> PostMatchAuditService:
    """Factory function to create audit service."""
    return PostMatchAuditService(session)
