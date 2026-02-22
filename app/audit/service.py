"""
Post-match audit service for analyzing prediction outcomes.

This service:
1. Fetches completed matches that have predictions
2. Retrieves detailed stats from API-Football (xG, events, etc.)
3. Records outcomes and classifies deviations
4. Generates NARRATIVE INSIGHTS for understanding WHY predictions failed
5. Provides learning signals for model improvement
"""

import logging
import math
import re
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
from sqlalchemy import select, and_, text
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.etl.api_football import APIFootballProvider
from app.ml.justice import compute_y_soft, compute_justice_weight
from app.ml.autopsy import classify_autopsy
from app.models import (
    Match,
    Prediction,
    PredictionOutcome,
    PostMatchAudit,
)
from app.config import get_settings, get_ia_features_config, should_generate_narratives

logger = logging.getLogger(__name__)


async def _get_llm_enabled(session: AsyncSession) -> bool:
    """Check if LLM narrative is enabled (env + dashboard toggle)."""
    try:
        settings = get_settings()
        if not settings.NARRATIVE_LLM_ENABLED or not settings.RUNPOD_API_KEY:
            return False
        # Respect dashboard toggle (ops_settings.narratives_enabled)
        ia_config = await get_ia_features_config(session)
        return should_generate_narratives(ia_config, settings)
    except Exception:
        return False


# Confidence tier thresholds
GOLD_THRESHOLD = 0.55  # >= 55% confidence
SILVER_THRESHOLD = 0.45  # >= 45% confidence
# Below 45% is Copper

# Justice Index alpha (GDT starting recommendation)
JUSTICE_ALPHA = 0.5


async def resolve_xg_for_match(
    session: AsyncSession, match_id: int,
) -> tuple[Optional[float], Optional[float], Optional[str]]:
    """Resolve best available xG for a match.

    GDT-mandated priority chain:
      1. match_understat_team (Opta-grade, Big 5)
      2. match_fotmob_stats  (FotMob raw — massive Tier-2/3 coverage)
      3. matches.xg_home     (FootyStats / API-Football generic fallback)
      4. match_sofascore_stats (Sofascore, last resort)
    """
    # 1. Understat (Opta-grade, Big 5)
    r = await session.execute(text(
        "SELECT xg_home, xg_away FROM match_understat_team "
        "WHERE match_id = :mid AND xg_home IS NOT NULL LIMIT 1"
    ), {"mid": match_id})
    row = r.fetchone()
    if row and row[0] is not None:
        return float(row[0]), float(row[1]), "understat"

    # 2. match_fotmob_stats (GDT Priority 2 — ~11K matches, Tier-2/3)
    r2 = await session.execute(text(
        "SELECT xg_home, xg_away FROM match_fotmob_stats "
        "WHERE match_id = :mid AND xg_home IS NOT NULL LIMIT 1"
    ), {"mid": match_id})
    row2 = r2.fetchone()
    if row2 and row2[0] is not None:
        return float(row2[0]), float(row2[1]), "fotmob_stats"

    # 3. matches table (FootyStats / API-Football generic fallback)
    r3 = await session.execute(text(
        "SELECT xg_home, xg_away, xg_source FROM matches "
        "WHERE id = :mid AND xg_home IS NOT NULL"
    ), {"mid": match_id})
    row3 = r3.fetchone()
    if row3 and row3[0] is not None:
        return float(row3[0]), float(row3[1]), row3[2]

    # 4. match_sofascore_stats (Sofascore, ~650 matches)
    r4 = await session.execute(text(
        "SELECT xg_home, xg_away FROM match_sofascore_stats "
        "WHERE match_id = :mid AND xg_home IS NOT NULL LIMIT 1"
    ), {"mid": match_id})
    row4 = r4.fetchone()
    if row4 and row4[0] is not None:
        return float(row4[0]), float(row4[1]), "sofascore_stats"

    return None, None, None


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
        Calculate deviation type and score with ALGORITHMIC HUMILITY.

        Philosophy:
        - All in-game events (VAR, penalties, red cards) are legitimate football outcomes
        - When we predict wrong, especially with high confidence, it's MODEL ERROR
        - Only classify as 'anomaly' for truly extra-sporting events or impossible
          statistical disparity (e.g., team with xG 0.05 beats team with xG 4.0)

        Returns:
            tuple: (deviation_type, deviation_score)
            - deviation_type: 'minimal', 'expected', or 'model_error'
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

        # === INCORRECT PREDICTION - ALGORITHMIC HUMILITY ===
        # When we're wrong, we admit it. No excuses.

        xg_result = self._get_xg_result(xg_home, xg_away)

        # Score based on how unlikely the actual result was
        deviation_score = 1.0 - actual_prob  # Higher score = more unexpected

        # Check for impossible statistical disparity (true anomaly)
        # This is the ONLY case where we classify as anomaly:
        # When xG strongly favored one team but the opposite won
        if xg_home is not None and xg_away is not None:
            xg_winner = xg_home if actual_result == "home" else (xg_away if actual_result == "away" else max(xg_home, xg_away))
            xg_loser = xg_away if actual_result == "home" else (xg_home if actual_result == "away" else min(xg_home, xg_away))

            # Truly anomalous: winner had < 0.3 xG AND loser had > 2.5 xG
            # This represents statistically impossible luck, not football merit
            if actual_result != "draw" and xg_winner < 0.3 and xg_loser > 2.5:
                return "anomaly", deviation_score

        # HIGH CONFIDENCE WRONG = MODEL ERROR (not anomaly!)
        # The model was confident and wrong - this is OUR fault, not random variance
        if confidence >= GOLD_THRESHOLD:
            return "model_error", deviation_score

        # SILVER confidence wrong = still model error, but less severe
        if confidence >= SILVER_THRESHOLD:
            return "model_error", deviation_score * 0.9

        # Low confidence wrong = expected variance (copper tier)
        # We weren't confident, so being wrong is within expectations
        return "expected", deviation_score * 0.8

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
        """
        Identify the primary factor for INFORMATIONAL purposes only.

        ALGORITHMIC HUMILITY:
        - This is NOT an excuse mechanism - it's contextual data
        - VAR, penalties, red cards are LEGITIMATE football outcomes
        - The primary factor helps understand the match context for retraining
        - When we're wrong, the default is MODEL_ERROR (we own our mistakes)

        Returns factor as informational context, not as justification.
        """
        if predicted_result == actual_result:
            return None  # No deviation to explain

        # === ALGORITHMIC HUMILITY: MODEL ERROR IS THE DEFAULT ===
        # When we predict wrong, it's our fault. Period.
        # Events below are CONTEXT, not EXCUSES.

        # Check xG alignment first - this tells us if the model's logic was sound
        if xg_home is not None and xg_away is not None:
            xg_result = self._get_xg_result(xg_home, xg_away)

            # Calculate finishing variance
            home_diff = actual_home_goals - xg_home
            away_diff = actual_away_goals - xg_away

            if xg_result == actual_result:
                # xG aligned with result - our model simply missed
                # This is PURE model error, no external factors needed
                return "model_error"

            # xG also didn't predict correctly - exceptional finishing
            if abs(home_diff) > 1.5 or abs(away_diff) > 1.5:
                # Someone finished WAY above or below expectation
                # Still not an excuse - good teams finish their chances
                return "finishing_quality"

        # === INFORMATIONAL CONTEXT (not excuses) ===
        # These events DID happen and SHOULD be recorded for analysis
        # But they don't justify our prediction failure

        # Note: We record these as context for potential feature engineering
        # A red card IS data - it means we should factor in discipline risk
        # A penalty IS data - it means we should factor in box presence

        if had_red_card:
            # Context: discipline played a role
            # Learning: factor in team discipline history
            return "context_red_card"

        if had_penalty:
            # Context: set piece/VAR situation
            # Learning: factor in penalty area aggression
            return "context_penalty"

        # Default: pure model error
        # We had no excuse - we simply predicted wrong
        return "model_error"

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

        # Fetch detailed stats from API (best-effort; do NOT fail the audit if API is unavailable/budget-limited)
        try:
            stats = await self.provider.get_fixture_statistics(match.external_id)
        except Exception as e:
            logger.warning(f"Audit stats fetch failed for fixture {match.external_id}: {e}")
            stats = {}

        try:
            events = await self._fetch_events(match.external_id)
        except Exception as e:
            logger.warning(f"Audit events fetch failed for fixture {match.external_id}: {e}")
            events = []

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

        # === Justice Statistics (Post-Match Auditor Module 1) ===
        try:
            xg_h_resolved, xg_a_resolved, xg_src = await resolve_xg_for_match(
                self.session, match.id
            )
            if xg_h_resolved is not None and xg_a_resolved is not None:
                p_h, p_d, p_a = compute_y_soft(xg_h_resolved, xg_a_resolved)
                w = float(compute_justice_weight(
                    np.array([float(match.home_goals)]),
                    np.array([float(match.away_goals)]),
                    np.array([xg_h_resolved]),
                    np.array([xg_a_resolved]),
                    alpha=JUSTICE_ALPHA,
                )[0])
                outcome.y_soft_home = round(p_h, 6)
                outcome.y_soft_draw = round(p_d, 6)
                outcome.y_soft_away = round(p_a, 6)
                outcome.justice_weight = round(w, 6)
                outcome.justice_alpha = JUSTICE_ALPHA
                outcome.xg_source_ysoft = xg_src
        except Exception as e:
            logger.warning(f"Justice stats failed for match {match.id}: {e}")
            xg_h_resolved, xg_a_resolved, xg_src = None, None, None

        # === Financial Autopsy (Post-Match Auditor Module 2) ===
        # GDT Override 4: NO sweeper. If CLV doesn't exist yet, tag stays NULL.
        # CLV Job is responsible for updating autopsy_tag when it completes.
        autopsy_tag = None
        try:
            clv_row = await self.session.execute(text(
                "SELECT clv_home, clv_draw, clv_away, selected_outcome, clv_selected "
                "FROM prediction_clv WHERE prediction_id = :pid LIMIT 1"
            ), {"pid": prediction.id})
            clv_data = clv_row.fetchone()

            if clv_data:
                sel_clv = None
                # Backfill selected_outcome if NULL
                if clv_data[3] is None:
                    clv_map = {"home": clv_data[0], "draw": clv_data[1], "away": clv_data[2]}
                    sel_clv_raw = clv_map.get(predicted_result)
                    sel_clv = float(sel_clv_raw) if sel_clv_raw is not None else None
                    await self.session.execute(text(
                        "UPDATE prediction_clv SET selected_outcome = :sel, clv_selected = :clv "
                        "WHERE prediction_id = :pid"
                    ), {"sel": predicted_result, "clv": sel_clv, "pid": prediction.id})
                else:
                    sel_clv = float(clv_data[4]) if clv_data[4] is not None else None

                # Use best available xG (resolved > API-Football stats)
                best_xg_h = xg_h_resolved if xg_h_resolved is not None else xg_home
                best_xg_a = xg_a_resolved if xg_a_resolved is not None else xg_away
                autopsy_tag = classify_autopsy(
                    prediction_correct=is_correct,
                    predicted_result=predicted_result,
                    clv_selected=sel_clv,
                    xg_home=best_xg_h,
                    xg_away=best_xg_a,
                ).value
        except Exception as e:
            logger.warning(f"Autopsy classification failed for match {match.id}: {e}")

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

        # Secondary factors - INFORMATIONAL context only
        # These are recorded for analysis, not as excuses
        secondary_factors = {}
        if had_red_card and "red_card" not in (primary_factor or ""):
            secondary_factors["red_card"] = True
        if had_penalty and "penalty" not in (primary_factor or ""):
            secondary_factors["penalty"] = True
        if had_var_decision:
            secondary_factors["var"] = True

        # === ALGORITHMIC HUMILITY: LEARNING FROM MODEL ERRORS ===
        # We learn from model_error (our mistakes), not from "anomalies"
        # Anomalies are truly rare statistical impossibilities
        # Model errors are where we improve
        should_learn = deviation_type == "model_error"

        # Build adjustment notes with humility
        if deviation_type == "model_error":
            adjustment_notes = (
                f"Model error: predicted {predicted_result} "
                f"(conf: {confidence:.1%}), actual {actual_result}. "
                f"Context: {primary_factor or 'none'}"
            )
        elif deviation_type == "anomaly":
            adjustment_notes = (
                f"Statistical anomaly: {predicted_result} vs {actual_result}. "
                f"xG strongly contradicted result - rare variance event."
            )
        else:
            adjustment_notes = None

        # === EARLY SEASON CAVEAT (GDT/ABE Directive) ===
        # Suppress should_adjust_model for early-season matches (matchday < 5).
        # Features like form_diff, elo_k10_diff use 5-match windows — unreliable before J5.
        _round_nums = re.findall(r'\d+', str(match.round or ''))
        effective_matchday = int(_round_nums[-1]) if _round_nums else None

        # Fallback for leagues without matchday in round string (e.g., MLS "Regular season")
        if effective_matchday is None:
            _md_result = await self.session.execute(text(
                "SELECT COUNT(DISTINCT date) + 1 FROM matches "
                "WHERE league_id = :lid AND season = :season "
                "AND status IN ('FT','AET','PEN') AND date < :mdate"
            ), {"lid": match.league_id, "season": match.season, "mdate": match.date})
            effective_matchday = _md_result.scalar() or 1

        is_early_season = effective_matchday < 5

        if is_early_season:
            should_learn = False
            if secondary_factors is None:
                secondary_factors = {}
            secondary_factors["early_season_caveat"] = True
            secondary_factors["effective_matchday"] = effective_matchday
            if adjustment_notes:
                adjustment_notes = f"[EARLY SEASON VARIANCE] {adjustment_notes}"

        # Generate narrative insights for the match
        narrative_result = self.generate_narrative_insights(
            prediction=prediction,
            actual_result=actual_result,
            home_goals=match.home_goals,
            away_goals=match.away_goals,
            stats=stats or {},
            home_team_name=match.home_team.name if match.home_team else "Local",
            away_team_name=match.away_team.name if match.away_team else "Visitante",
            home_position=None,  # Positions require extra API call, skip for now
            away_position=None,
        )

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
            autopsy_tag=autopsy_tag,
            should_adjust_model=should_learn,
            adjustment_notes=adjustment_notes,
            narrative_insights=narrative_result.get("insights"),
            momentum_analysis=narrative_result.get("momentum_analysis"),
        )

        # Generate LLM narrative (best-effort, doesn't block audit)
        if await _get_llm_enabled(self.session):
            try:
                llm_result = await self._generate_llm_narrative(
                    match=match,
                    prediction=prediction,
                    stats=stats or {},
                    events=events,
                    is_correct=is_correct,
                )
                audit.llm_narrative_status = llm_result.status
                audit.llm_narrative_json = llm_result.narrative_json
                audit.llm_narrative_generated_at = datetime.utcnow()
                audit.llm_narrative_model = llm_result.model
                audit.llm_narrative_delay_ms = llm_result.delay_ms
                audit.llm_narrative_exec_ms = llm_result.exec_ms
                audit.llm_narrative_tokens_in = llm_result.tokens_in
                audit.llm_narrative_tokens_out = llm_result.tokens_out
                audit.llm_narrative_worker_id = llm_result.worker_id
                audit.llm_narrative_error_code = llm_result.error_code
                audit.llm_narrative_error_detail = llm_result.error[:500] if llm_result.error else None
                audit.llm_narrative_request_id = llm_result.request_id or None
                audit.llm_narrative_attempts = llm_result.attempts
                logger.info(f"LLM narrative for match {match.id}: {llm_result.status} (error_code={llm_result.error_code})")
            except Exception as e:
                logger.warning(f"LLM narrative failed for match {match.id}: {e}")
                audit.llm_narrative_status = "error"
                audit.llm_narrative_error_code = "exception"
                audit.llm_narrative_error_detail = str(e)[:500]

        self.session.add(audit)
        logger.info(
            f"Audited match {match.id}: {predicted_result} vs {actual_result} "
            f"({'✓' if is_correct else '✗'}) - {deviation_type} "
            f"({len(narrative_result.get('insights', []))} insights)"
        )

        return outcome

    async def _fetch_events(self, fixture_id: int) -> list:
        """Fetch match events from API."""
        data = await self.provider._rate_limited_request(
            "fixtures/events", {"fixture": fixture_id}, entity="events"
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

    # =========================================================================
    # LLM NARRATIVE GENERATION
    # =========================================================================

    async def _generate_llm_narrative(
        self,
        match: Match,
        prediction: Prediction,
        stats: dict,
        events: list,
        is_correct: bool,
    ):
        """
        Generate LLM narrative for a match using RunPod/Qwen.

        Args:
            match: The Match object with team relationships loaded.
            prediction: The Prediction object.
            stats: Match statistics dict.
            events: Match events list.
            is_correct: Whether prediction was correct.

        Returns:
            NarrativeResult from the LLM generator.
        """
        from app.llm.narrative_generator import NarrativeGenerator

        # Build compact match data for prompt
        home_team_name = match.home_team.name if match.home_team else "Local"
        away_team_name = match.away_team.name if match.away_team else "Visitante"

        predicted_result, confidence = self._get_predicted_result(prediction)

        match_data = {
            "match_id": match.id,
            "home_team": home_team_name,
            "away_team": away_team_name,
            "league_name": f"Liga {match.league_id}",
            "date": match.date.isoformat() if match.date else "",
            "home_goals": match.home_goals or 0,
            "away_goals": match.away_goals or 0,
            "stats": stats,
            "prediction": {
                "probabilities": {
                    "home": prediction.home_prob,
                    "draw": prediction.draw_prob,
                    "away": prediction.away_prob,
                },
                "predicted_result": predicted_result,
                "confidence": confidence,
                "correct": is_correct,
            },
            "events": [
                {
                    "minute": e.get("time", {}).get("elapsed"),
                    "type": e.get("type", ""),
                    "detail": e.get("detail", ""),
                }
                for e in (events or [])[:5]  # Cap at 5 events
            ],
        }

        generator = NarrativeGenerator()
        try:
            return await generator.generate(match_data)
        finally:
            await generator.close()

    # =========================================================================
    # NARRATIVE REASONING ENGINE
    # =========================================================================

    # Historical big teams in La Liga (Top 10 histórico)
    BIG_TEAMS = {
        "Real Madrid", "Barcelona", "Atletico Madrid", "Atlético Madrid",
        "Sevilla", "Valencia", "Villarreal", "Athletic Club", "Athletic Bilbao",
        "Real Sociedad", "Real Betis",
    }

    def generate_narrative_insights(
        self,
        prediction: Prediction,
        actual_result: str,
        home_goals: int,
        away_goals: int,
        stats: dict,
        home_team_name: str,
        away_team_name: str,
        home_position: Optional[int] = None,
        away_position: Optional[int] = None,
        home_penalties: int = 0,
        away_penalties: int = 0,
    ) -> dict:
        """
        Generate human-readable narrative insights explaining WHY the result happened.

        This is the "Reasoning Engine" that provides rich context beyond simple
        model error classification. It analyzes:
        1. Efficiency (shots on target vs total shots)
        2. Clinical finishing (goals vs xG)
        3. npxG (Non-Penalty xG) - real open-play danger
        4. Defensive solidity (goalkeeper saves, blocked shots)
        5. Table position context (Urgency Factor)
        6. Localía en Crisis (Big team home collapse)
        7. Momentum analysis

        Returns dict with insights list and momentum_analysis.
        """
        insights = []
        home_stats = stats.get("home", {})
        away_stats = stats.get("away", {})

        # Parse key metrics
        home_sot = self._parse_int(home_stats.get("shots_on_goal")) or 0
        away_sot = self._parse_int(away_stats.get("shots_on_goal")) or 0
        home_shots = self._parse_int(home_stats.get("total_shots")) or 0
        away_shots = self._parse_int(away_stats.get("total_shots")) or 0
        home_xg = self._parse_float(home_stats.get("expected_goals"))
        away_xg = self._parse_float(away_stats.get("expected_goals"))
        home_saves = self._parse_int(home_stats.get("goalkeeper_saves")) or 0
        away_saves = self._parse_int(away_stats.get("goalkeeper_saves")) or 0

        # Get predicted result
        predicted_result, confidence = self._get_predicted_result(prediction)
        is_correct = predicted_result == actual_result

        # Determine winner name
        if actual_result == "home":
            winner_name = home_team_name
            loser_name = away_team_name
        elif actual_result == "away":
            winner_name = away_team_name
            loser_name = home_team_name
        else:
            winner_name = None
            loser_name = None

        # === EFFICIENCY ANALYSIS ===
        # Who was more clinical with their chances?
        if home_shots > 0 and away_shots > 0:
            home_accuracy = home_sot / home_shots
            away_accuracy = away_sot / away_shots

            if away_accuracy > home_accuracy * 1.5 and actual_result == "away":
                # Away team was significantly more clinical
                insights.append({
                    "type": "analysis",
                    "icon": "target",
                    "message": (
                        f"{away_team_name} aprovechó cada oportunidad. "
                        f"Con {away_sot} tiros al arco de {away_shots} intentos, "
                        f"fueron más efectivos que {home_team_name} ({home_sot} de {home_shots})."
                    ),
                    "priority": 3,  # Analysis priority
                })
            elif home_accuracy > away_accuracy * 1.5 and actual_result == "home":
                insights.append({
                    "type": "analysis",
                    "icon": "target",
                    "message": (
                        f"{home_team_name} aprovechó cada oportunidad. "
                        f"Con {home_sot} tiros al arco de {home_shots} intentos, "
                        f"fueron más efectivos que {away_team_name} ({away_sot} de {away_shots})."
                    ),
                    "priority": 3,
                })

        # === CLINICAL FINISHING (Goals vs xG) ===
        if home_xg and away_xg:
            home_overperform = home_goals - home_xg
            away_overperform = away_goals - away_xg

            if away_overperform > 0.5 and actual_result == "away":
                insights.append({
                    "type": "analysis",
                    "icon": "flame.fill",
                    "message": (
                        f"{away_team_name} fue letal en el área. "
                        f"Convirtió {away_goals} goles con pocas chances claras."
                    ),
                    "priority": 3,
                })
            elif home_overperform > 0.5 and actual_result == "home":
                insights.append({
                    "type": "analysis",
                    "icon": "flame.fill",
                    "message": (
                        f"{home_team_name} fue letal en el área. "
                        f"Convirtió {home_goals} goles con pocas chances claras."
                    ),
                    "priority": 3,
                })

            # STERILE favorite analysis - WARNING type has high priority (caution)
            if not is_correct and predicted_result == "home" and home_sot <= 2:
                insights.append({
                    "type": "caution",
                    "icon": "exclamationmark.triangle.fill",
                    "message": (
                        f"{home_team_name} llegó mucho pero sin peligro real. "
                        f"De {home_shots} intentos, solo {home_sot} fueron al arco."
                    ),
                    "priority": 0,  # Highest priority - warning/caution
                })
            elif not is_correct and predicted_result == "away" and away_sot <= 2:
                insights.append({
                    "type": "caution",
                    "icon": "exclamationmark.triangle.fill",
                    "message": (
                        f"{away_team_name} llegó mucho pero sin peligro real. "
                        f"De {away_shots} intentos, solo {away_sot} fueron al arco."
                    ),
                    "priority": 0,
                })

        # === npxG ANALYSIS (Non-Penalty Expected Goals) ===
        # A penalty has xG of ~0.78. If total xG is inflated by penalties,
        # the team's real open-play danger was much lower.
        PENALTY_XG = 0.78
        NPXG_LOW_THRESHOLD = 1.0  # npxG < 1.0 is low real danger

        if home_xg and not is_correct:
            home_npxg = home_xg - (home_penalties * PENALTY_XG)
            away_npxg = (away_xg or 0) - (away_penalties * PENALTY_XG)

            # Favorite had inflated xG due to penalties
            if predicted_result == "home" and home_penalties > 0 and home_npxg < NPXG_LOW_THRESHOLD:
                insights.append({
                    "type": "caution",
                    "icon": "sportscourt.fill",
                    "message": (
                        f"{home_team_name} dependió de jugadas a balón parado. "
                        f"En juego abierto generó poco peligro real."
                    ),
                    "priority": 0,
                })
            elif predicted_result == "away" and away_penalties > 0 and away_npxg < NPXG_LOW_THRESHOLD:
                insights.append({
                    "type": "caution",
                    "icon": "sportscourt.fill",
                    "message": (
                        f"{away_team_name} dependió de jugadas a balón parado. "
                        f"En juego abierto generó poco peligro real."
                    ),
                    "priority": 0,
                })

        # === DEFENSIVE ANALYSIS ===
        # Portero Heroico: Saves >= 5 is exceptional
        HEROIC_SAVES_THRESHOLD = 5
        SOLID_SAVES_THRESHOLD = 3

        if winner_name and loser_name:
            if actual_result == "away":
                if away_saves >= HEROIC_SAVES_THRESHOLD:
                    insights.append({
                        "type": "analysis",
                        "icon": "shield.fill",
                        "message": (
                            f"El arquero de {away_team_name} fue una muralla. "
                            f"Realizó {away_saves} atajadas clave para asegurar la victoria."
                        ),
                        "priority": 1,  # High priority - heroic performance
                    })
                elif away_saves >= SOLID_SAVES_THRESHOLD:
                    insights.append({
                        "type": "analysis",
                        "icon": "hand.raised.fill",
                        "message": (
                            f"Buena actuación del arquero de {away_team_name} "
                            f"con {away_saves} atajadas importantes."
                        ),
                        "priority": 3,
                    })
            elif actual_result == "home":
                if home_saves >= HEROIC_SAVES_THRESHOLD:
                    insights.append({
                        "type": "analysis",
                        "icon": "shield.fill",
                        "message": (
                            f"El arquero de {home_team_name} fue una muralla. "
                            f"Realizó {home_saves} atajadas clave para asegurar la victoria."
                        ),
                        "priority": 1,
                    })
                elif home_saves >= SOLID_SAVES_THRESHOLD:
                    insights.append({
                        "type": "analysis",
                        "icon": "hand.raised.fill",
                        "message": (
                            f"Buena actuación del arquero de {home_team_name} "
                            f"con {home_saves} atajadas importantes."
                        ),
                        "priority": 3,
                    })

        # === URGENCY FACTOR (Table Position) ===
        if home_position and away_position:
            # Relegation zone is typically positions 18-20
            RELEGATION_ZONE = 17  # Position 17+ is danger zone

            if away_position >= RELEGATION_ZONE and actual_result == "away":
                # Team in relegation danger won
                insights.append({
                    "type": "context",
                    "icon": "flame.circle.fill",
                    "message": (
                        f"{away_team_name} está peleando por no descender (puesto {away_position}). "
                        f"Esa necesidad de puntos se notó en la cancha."
                    ),
                    "priority": 4,  # Context priority
                })
            elif home_position >= RELEGATION_ZONE and actual_result == "home":
                insights.append({
                    "type": "context",
                    "icon": "flame.circle.fill",
                    "message": (
                        f"{home_team_name} está peleando por no descender (puesto {home_position}). "
                        f"Esa necesidad de puntos se notó en la cancha."
                    ),
                    "priority": 4,
                })

            # Position differential insight
            if abs(home_position - away_position) >= 5:
                higher_team = home_team_name if home_position < away_position else away_team_name
                lower_team = away_team_name if home_position < away_position else home_team_name
                higher_pos = min(home_position, away_position)
                lower_pos = max(home_position, away_position)

                if (actual_result == "away" and away_position > home_position) or \
                   (actual_result == "home" and home_position > away_position):
                    # Underdog won
                    insights.append({
                        "type": "context",
                        "icon": "arrow.up.circle.fill",
                        "message": (
                            f"Sorpresa: {lower_team} (puesto {lower_pos}) "
                            f"venció a {higher_team} (puesto {higher_pos})."
                        ),
                        "priority": 4,
                    })

            # === LOCALÍA EN CRISIS (Big Team Home Collapse) ===
            # Big teams losing at home to underdogs (position 15+) indicates
            # psychological collapse under stadium pressure
            UNDERDOG_POSITION = 15

            is_home_big_team = home_team_name in self.BIG_TEAMS
            is_away_underdog = away_position >= UNDERDOG_POSITION if away_position else False

            if (is_home_big_team and is_away_underdog and
                actual_result == "away" and not is_correct):
                # Big team lost at home to underdog
                goal_diff = away_goals - home_goals
                if goal_diff >= 3:
                    severity_msg = "Una derrota difícil de explicar."
                else:
                    severity_msg = "No pudieron con la presión."

                insights.append({
                    "type": "context",
                    "icon": "person.3.sequence.fill",
                    "message": (
                        f"{home_team_name} se bloqueó ante su gente. "
                        f"Perdió en casa contra {away_team_name} por {home_goals}-{away_goals}. "
                        f"{severity_msg}"
                    ),
                    "priority": 1,  # High priority - significant event
                })

        # === MODEL ERROR ADMISSION ===
        if not is_correct:
            # Translate result to Spanish (simple terms)
            result_es = {"home": "local", "away": "visitante", "draw": "empate"}
            confidence_pct = int(confidence * 100)

            insights.append({
                "type": "admission",
                "icon": "brain.head.profile",
                "message": (
                    f"Nos equivocamos. Apostamos por victoria {result_es.get(predicted_result, predicted_result)} "
                    f"con {confidence_pct}% de confianza, pero ganó el {result_es.get(actual_result, actual_result)}."
                ),
                "priority": 2,  # Admission comes after caution, before analysis
            })

        # === SUMMARY INSIGHT ===
        if not is_correct and winner_name:
            if away_sot > home_sot and actual_result == "away":
                insights.append({
                    "type": "summary",
                    "icon": "checkmark.seal.fill",
                    "message": (
                        f"Resultado justo. {winner_name} tuvo más llegadas claras "
                        f"({away_sot} tiros al arco vs {home_sot})."
                    ),
                    "priority": 5,  # Summary comes last
                })
            elif home_sot > away_sot and actual_result == "home":
                insights.append({
                    "type": "summary",
                    "icon": "checkmark.seal.fill",
                    "message": (
                        f"Resultado justo. {winner_name} tuvo más llegadas claras "
                        f"({home_sot} tiros al arco vs {away_sot})."
                    ),
                    "priority": 5,
                })

        # === SORT BY PRIORITY FOR iOS ===
        # Order: caution (0) → heroic/collapse (1) → admission (2) → analysis (3) → context (4) → summary (5)
        insights.sort(key=lambda x: x.get("priority", 99))

        # === MOMENTUM ANALYSIS ===
        # Detect if the losing team "gave up" after conceding
        # Approximation: If favorite lost badly (3+ goals) and had very few SOT, they collapsed
        momentum_analysis = None

        if not is_correct and loser_name:
            goal_diff = abs(home_goals - away_goals)
            loser_sot = home_sot if actual_result == "away" else away_sot

            # Collapse detection: lost by 2+ goals AND had <= 2 SOT
            if goal_diff >= 2 and loser_sot <= 2:
                momentum_analysis = {
                    "type": "collapse",
                    "icon": "arrow.down.right.circle.fill",
                    "message": (
                        f"{loser_name} no reaccionó después de ir abajo. "
                        f"Solo {loser_sot} tiro(s) al arco en todo el partido."
                    ),
                }
            elif goal_diff >= 3:
                # Heavy defeat even with some shots
                momentum_analysis = {
                    "type": "overwhelmed",
                    "icon": "waveform.path.ecg",
                    "message": (
                        f"{loser_name} fue superado en todos los aspectos. "
                        f"El {home_goals}-{away_goals} final lo dice todo."
                    ),
                }
            elif loser_sot >= home_sot + away_sot - loser_sot:
                # Loser had more SOT but still lost - unlucky
                momentum_analysis = {
                    "type": "unlucky",
                    "icon": "dice.fill",
                    "message": (
                        f"{loser_name} tuvo más chances ({loser_sot} tiros al arco) "
                        f"pero no las convirtió. Faltó efectividad."
                    ),
                }

        return {
            "insights": insights,
            "momentum_analysis": momentum_analysis,
        }

    def generate_learning_signals(
        self,
        prediction: Prediction,
        actual_result: str,
        stats: dict,
    ) -> list[str]:
        """
        Generate specific learning signals for model retraining.

        These are actionable insights about what the model should learn
        from this match to improve future predictions.
        """
        signals = []
        home_stats = stats.get("home", {})
        away_stats = stats.get("away", {})

        predicted_result, confidence = self._get_predicted_result(prediction)
        is_correct = predicted_result == actual_result

        if is_correct:
            return []  # No learning needed from correct predictions

        # Parse metrics
        home_sot = self._parse_int(home_stats.get("shots_on_goal")) or 0
        away_sot = self._parse_int(away_stats.get("shots_on_goal")) or 0

        # Learning signal: Efficiency matters more than possession
        if predicted_result == "home" and actual_result == "away" and away_sot > home_sot:
            signals.append(
                "LEARN: La eficiencia en definición (SOT ratio) puede superar la ventaja de local"
            )

        if predicted_result == "away" and actual_result == "home" and home_sot > away_sot:
            signals.append(
                "LEARN: La presión local con disparos a puerta puede superar la calidad visitante"
            )

        # Learning signal: Low SOT indicates sterility
        if predicted_result == "home" and home_sot <= 2:
            signals.append(
                f"LEARN: Solo {home_sot} SOT del favorito local indica problema de generación ofensiva"
            )

        if predicted_result == "away" and away_sot <= 2:
            signals.append(
                f"LEARN: Solo {away_sot} SOT del favorito visitante indica problema de generación ofensiva"
            )

        return signals

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
        # Eager load home_team and away_team for narrative insights
        query = (
            select(Match, Prediction)
            .join(Prediction, Match.id == Prediction.match_id)
            .outerjoin(PredictionOutcome, Prediction.id == PredictionOutcome.prediction_id)
            .options(
                selectinload(Match.home_team),
                selectinload(Match.away_team),
            )
            .where(
                and_(
                    Match.status.in_(("FT", "AET", "PEN")),
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
