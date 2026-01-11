"""
Fast-Path Narrative Generation Service.

Generates LLM narratives within minutes of match completion instead of waiting
for the daily audit job (08:00 UTC).

Architecture:
1. Selector: Find recently finished matches with predictions
2. Stats Refresh: Fetch missing stats from API-Football with backoff
3. Enqueue: Submit batch of ready matches to RunPod
4. Poll: Check completions and persist results
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Optional

from sqlalchemy import select, and_, or_
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.config import get_settings
from app.models import Match, Prediction, PostMatchAudit, PredictionOutcome
from app.llm.narrative_generator import (
    NarrativeGenerator,
    NarrativeResult,
    check_stats_gating,
    build_narrative_prompt,
    parse_json_response,
    validate_narrative_json,
    log_llm_evaluation,
)
from app.llm.runpod_client import RunPodClient, RunPodError

logger = logging.getLogger(__name__)

# Backoff profile for stats refresh (minutes since FT -> seconds between checks)
STATS_BACKOFF_PROFILE = [
    (5, 60),      # < 5 min since FT: check every 1 min
    (15, 120),    # < 15 min: every 2 min
    (60, 300),    # < 60 min: every 5 min
    (90, 900),    # < 90 min: every 15 min
]


def _get_backoff_interval(minutes_since_ft: float) -> int:
    """Get seconds until next stats check based on time since FT."""
    for threshold_min, interval_sec in STATS_BACKOFF_PROFILE:
        if minutes_since_ft < threshold_min:
            return interval_sec
    return 900  # Default: 15 min


def _should_check_stats(match: Match, now: datetime) -> bool:
    """Determine if we should refresh stats for this match."""
    if not match.finished_at:
        return True  # Always check if we don't know when it finished

    minutes_since_ft = (now - match.finished_at).total_seconds() / 60
    if minutes_since_ft > 90:
        return False  # Stop checking after 90 min

    if not match.stats_last_checked_at:
        return True

    # Check if enough time has passed based on backoff profile
    seconds_since_check = (now - match.stats_last_checked_at).total_seconds()
    required_interval = _get_backoff_interval(minutes_since_ft)
    return seconds_since_check >= required_interval


class FastPathService:
    """Service for fast-path narrative generation."""

    def __init__(self, session: AsyncSession):
        self.session = session
        self.settings = get_settings()
        self.runpod = RunPodClient()
        self._api_provider = None

    async def close(self):
        """Close resources."""
        await self.runpod.close()
        if self._api_provider:
            await self._api_provider.close()

    async def _get_api_provider(self):
        """Lazy-load API provider."""
        if self._api_provider is None:
            from app.etl import APIFootballProvider
            self._api_provider = APIFootballProvider()
        return self._api_provider

    async def run_tick(self) -> dict:
        """
        Run one tick of the fast-path job.

        Returns:
            Dict with metrics: selected, refreshed, ready, enqueued, completed, errors
        """
        now = datetime.utcnow()
        lookback = now - timedelta(minutes=self.settings.FASTPATH_LOOKBACK_MINUTES)

        metrics = {
            "selected": 0,
            "refreshed": 0,
            "stats_ready": 0,
            "enqueued": 0,
            "completed": 0,
            "errors": 0,
        }

        # 1. Select recently finished matches needing narrative
        candidates = await self._select_candidates(lookback, now)
        metrics["selected"] = len(candidates)
        logger.info(f"[FASTPATH] Selected {len(candidates)} candidates for narrative generation")

        if not candidates:
            return metrics

        # 2. Refresh stats for matches that need it
        refreshed = await self._refresh_stats_batch(candidates, now)
        metrics["refreshed"] = refreshed

        # 3. Check which matches are now ready (stats gating passed)
        ready_matches = []
        for match in candidates:
            if match.stats_ready_at:
                ready_matches.append(match)
            else:
                # Check if stats now pass gating
                stats_data = {"stats": match.stats or {}}
                passes, _ = check_stats_gating(stats_data)
                if passes:
                    match.stats_ready_at = now
                    ready_matches.append(match)

        metrics["stats_ready"] = len(ready_matches)
        await self.session.commit()

        if not ready_matches:
            logger.info(f"[FASTPATH] No matches ready (stats gating). Refreshed {refreshed} stats.")
            return metrics

        # 4. Enqueue narratives for ready matches
        enqueued = await self._enqueue_narratives(ready_matches, now)
        metrics["enqueued"] = enqueued

        # 5. Poll for completions
        completed, errors = await self._poll_completions()
        metrics["completed"] = completed
        metrics["errors"] = errors

        logger.info(
            f"[FASTPATH] Tick complete: selected={metrics['selected']}, "
            f"refreshed={metrics['refreshed']}, ready={metrics['stats_ready']}, "
            f"enqueued={metrics['enqueued']}, completed={metrics['completed']}, errors={metrics['errors']}"
        )

        return metrics

    async def _select_candidates(self, lookback: datetime, now: datetime) -> list[Match]:
        """
        Select matches that:
        - Status is FT/AET/PEN
        - Finished within lookback window
        - Have a prediction
        - Don't have a successful LLM narrative yet
        """
        # Get matches with predictions that finished recently
        result = await self.session.execute(
            select(Match)
            .options(selectinload(Match.predictions))
            .options(selectinload(Match.home_team))
            .options(selectinload(Match.away_team))
            .where(
                and_(
                    Match.status.in_(["FT", "AET", "PEN"]),
                    Match.date >= lookback,
                    Match.date <= now,
                )
            )
            .order_by(Match.date.desc())
        )
        matches = result.scalars().all()

        # Filter to matches with predictions and without successful narrative
        candidates = []
        for match in matches:
            if not match.predictions:
                continue

            # Check if already has successful narrative via outcome/audit
            outcome_result = await self.session.execute(
                select(PredictionOutcome)
                .join(PostMatchAudit, PredictionOutcome.id == PostMatchAudit.outcome_id)
                .where(
                    and_(
                        PredictionOutcome.match_id == match.id,
                        PostMatchAudit.llm_narrative_status == "ok",
                    )
                )
            )
            if outcome_result.scalar_one_or_none():
                continue  # Already has narrative

            # Set finished_at if not set
            if not match.finished_at:
                match.finished_at = now

            candidates.append(match)

        return candidates

    async def _refresh_stats_batch(self, matches: list[Match], now: datetime) -> int:
        """Refresh stats for matches that need it. Returns count of refreshed."""
        refreshed = 0
        provider = await self._get_api_provider()

        for match in matches:
            if not _should_check_stats(match, now):
                continue

            if not match.external_id:
                continue

            try:
                # Fetch stats from API-Football
                stats_data = await provider._rate_limited_request(
                    "fixtures/statistics",
                    {"fixture": match.external_id}
                )
                response = stats_data.get("response", [])

                if response and len(response) >= 2:
                    # Parse stats into our format
                    home_stats = self._parse_team_stats(response[0].get("statistics", []))
                    away_stats = self._parse_team_stats(response[1].get("statistics", []))

                    match.stats = {
                        "home": home_stats,
                        "away": away_stats,
                    }
                    refreshed += 1
                    logger.debug(f"[FASTPATH] Refreshed stats for match {match.id}")

                match.stats_last_checked_at = now

            except Exception as e:
                logger.warning(f"[FASTPATH] Failed to refresh stats for match {match.id}: {e}")
                match.stats_last_checked_at = now

        await self.session.commit()
        return refreshed

    def _parse_team_stats(self, stats_list: list) -> dict:
        """Parse API-Football stats list into dict."""
        result = {}
        key_map = {
            "Ball Possession": "ball_possession",
            "Total Shots": "total_shots",
            "Shots on Goal": "shots_on_goal",
            "Shots off Goal": "shots_off_goal",
            "Blocked Shots": "blocked_shots",
            "Shots insidebox": "shots_insidebox",
            "Shots outsidebox": "shots_outsidebox",
            "Fouls": "fouls",
            "Corner Kicks": "corner_kicks",
            "Offsides": "offsides",
            "Yellow Cards": "yellow_cards",
            "Red Cards": "red_cards",
            "Goalkeeper Saves": "goalkeeper_saves",
            "Total passes": "total_passes",
            "Passes accurate": "passes_accurate",
            "Passes %": "passes_pct",
            "expected_goals": "expected_goals",
        }
        for stat in stats_list:
            stat_type = stat.get("type", "")
            value = stat.get("value")
            if stat_type in key_map and value is not None:
                # Clean percentage values
                if isinstance(value, str) and value.endswith("%"):
                    try:
                        value = float(value.rstrip("%"))
                    except ValueError:
                        pass
                result[key_map[stat_type]] = value
        return result

    async def _enqueue_narratives(self, matches: list[Match], now: datetime) -> int:
        """
        Enqueue narrative generation for ready matches.

        Creates outcome/audit records if needed and submits to RunPod.
        """
        enqueued = 0
        generator = NarrativeGenerator()

        try:
            for match in matches[:self.settings.FASTPATH_MAX_CONCURRENT_JOBS]:
                # Get or create outcome/audit
                outcome, audit = await self._get_or_create_audit(match)
                if not outcome or not audit:
                    continue

                # Skip if already processing or done
                if audit.llm_narrative_status in ("ok", "in_queue", "running"):
                    continue

                # Get frozen/latest prediction
                prediction = await self._get_best_prediction(match.id)
                if not prediction:
                    continue

                # Build match data for prompt
                match_data = self._build_match_data(match, prediction)

                # Check gating
                passes, reason = check_stats_gating(match_data)
                if not passes:
                    audit.llm_narrative_status = "skipped"
                    audit.llm_narrative_error_code = "gating_skipped"
                    audit.llm_narrative_error_detail = reason
                    continue

                # Submit to RunPod
                try:
                    prompt = build_narrative_prompt(match_data)
                    job_id = await self.runpod.run_job(prompt)

                    audit.llm_narrative_status = "in_queue"
                    audit.llm_narrative_request_id = job_id
                    audit.llm_narrative_attempts = 1
                    audit.llm_narrative_model = "qwen-vllm"
                    enqueued += 1

                    logger.info(f"[FASTPATH] Enqueued match {match.id} -> job {job_id}")

                except RunPodError as e:
                    audit.llm_narrative_status = "error"
                    audit.llm_narrative_error_code = "runpod_http_error"
                    audit.llm_narrative_error_detail = str(e)[:500]
                    logger.error(f"[FASTPATH] Failed to enqueue match {match.id}: {e}")

            await self.session.commit()

        finally:
            await generator.close()

        return enqueued

    async def _get_or_create_audit(self, match: Match) -> tuple:
        """Get or create PredictionOutcome and PostMatchAudit for a match."""
        # Check for existing outcome
        result = await self.session.execute(
            select(PredictionOutcome)
            .options(selectinload(PredictionOutcome.audit))
            .where(PredictionOutcome.match_id == match.id)
        )
        outcome = result.scalar_one_or_none()

        if outcome and outcome.audit:
            return outcome, outcome.audit

        # Need to create - get prediction first
        prediction = await self._get_best_prediction(match.id)
        if not prediction:
            return None, None

        # Create outcome if needed
        if not outcome:
            # Determine actual result
            home_goals = match.home_goals or 0
            away_goals = match.away_goals or 0
            if home_goals > away_goals:
                actual = "home"
            elif away_goals > home_goals:
                actual = "away"
            else:
                actual = "draw"

            # Determine predicted result
            probs = {
                "home": prediction.home_prob,
                "draw": prediction.draw_prob,
                "away": prediction.away_prob,
            }
            predicted = max(probs, key=probs.get)
            confidence = probs[predicted]

            # Determine confidence tier
            if confidence >= 0.50:
                tier = "gold"
            elif confidence >= 0.40:
                tier = "silver"
            else:
                tier = "copper"

            outcome = PredictionOutcome(
                match_id=match.id,
                prediction_id=prediction.id,
                predicted_result=predicted,
                actual_result=actual,
                actual_home_goals=home_goals,
                actual_away_goals=away_goals,
                confidence=confidence,
                confidence_tier=tier,
                prediction_correct=(predicted == actual),
            )
            self.session.add(outcome)
            await self.session.flush()

        # Create audit if needed
        audit = outcome.audit if outcome.audit else None
        if not audit:
            audit = PostMatchAudit(
                outcome_id=outcome.id,
                deviation_type="pending_fastpath",
                deviation_score=0.0,
                xg_result_aligned=False,
                xg_prediction_aligned=False,
            )
            self.session.add(audit)
            await self.session.flush()

        return outcome, audit

    async def _get_best_prediction(self, match_id: int) -> Optional[Prediction]:
        """Get frozen prediction, or latest if none frozen."""
        # Try frozen first
        result = await self.session.execute(
            select(Prediction)
            .where(
                and_(
                    Prediction.match_id == match_id,
                    Prediction.is_frozen == True,
                )
            )
            .order_by(Prediction.frozen_at.desc().nullslast())
            .limit(1)
        )
        prediction = result.scalar_one_or_none()
        if prediction:
            return prediction

        # Fall back to latest
        result = await self.session.execute(
            select(Prediction)
            .where(Prediction.match_id == match_id)
            .order_by(Prediction.created_at.desc())
            .limit(1)
        )
        return result.scalar_one_or_none()

    def _build_match_data(self, match: Match, prediction: Prediction) -> dict:
        """Build match_data dict for narrative prompt."""
        return {
            "match_id": match.id,
            "home_team": match.home_team.name if match.home_team else "Local",
            "away_team": match.away_team.name if match.away_team else "Visitante",
            "league_name": "",  # Could add league lookup if needed
            "date": match.date.isoformat() if match.date else "",
            "home_goals": match.home_goals or 0,
            "away_goals": match.away_goals or 0,
            "stats": match.stats or {},
            "events": [],  # Could fetch events if needed
            "prediction": {
                "predicted_result": max(
                    [("home", prediction.home_prob), ("draw", prediction.draw_prob), ("away", prediction.away_prob)],
                    key=lambda x: x[1]
                )[0].upper(),
                "confidence": max(prediction.home_prob, prediction.draw_prob, prediction.away_prob),
                "home_prob": prediction.home_prob,
                "draw_prob": prediction.draw_prob,
                "away_prob": prediction.away_prob,
            },
            "market_odds": {
                "home": match.odds_home,
                "draw": match.odds_draw,
                "away": match.odds_away,
            } if match.odds_home else {},
        }

    async def _poll_completions(self) -> tuple[int, int]:
        """
        Poll RunPod for job completions.

        Returns:
            Tuple of (completed_count, error_count)
        """
        completed = 0
        errors = 0

        # Find audits with in_queue/running status
        result = await self.session.execute(
            select(PostMatchAudit)
            .join(PredictionOutcome, PostMatchAudit.outcome_id == PredictionOutcome.id)
            .where(
                and_(
                    PostMatchAudit.llm_narrative_status.in_(["in_queue", "running"]),
                    PostMatchAudit.llm_narrative_request_id.isnot(None),
                )
            )
        )
        pending_audits = result.scalars().all()

        for audit in pending_audits:
            try:
                job_data = await self.runpod.poll_job(audit.llm_narrative_request_id)
                status = job_data.get("status")

                if status == "COMPLETED":
                    # Extract and validate response
                    try:
                        text = self.runpod.extract_text(job_data)
                        tokens_in, tokens_out = self.runpod.extract_usage(job_data)
                        meta = self.runpod.extract_metadata(job_data)

                        parsed = parse_json_response(text)
                        # Get match_id for validation
                        outcome = await self.session.get(PredictionOutcome, audit.outcome_id)
                        match_id = outcome.match_id if outcome else 0

                        if parsed and validate_narrative_json(parsed, match_id):
                            audit.llm_narrative_status = "ok"
                            audit.llm_narrative_json = parsed
                            audit.llm_narrative_generated_at = datetime.utcnow()
                            audit.llm_narrative_delay_ms = meta["delay_ms"]
                            audit.llm_narrative_exec_ms = meta["exec_ms"]
                            audit.llm_narrative_tokens_in = tokens_in
                            audit.llm_narrative_tokens_out = tokens_out
                            audit.llm_narrative_worker_id = meta["worker_id"]
                            completed += 1

                            logger.info(
                                f"[FASTPATH] Completed narrative for audit {audit.id}: "
                                f"tokens={tokens_in}/{tokens_out}, exec={meta['exec_ms']}ms"
                            )
                        else:
                            audit.llm_narrative_status = "error"
                            audit.llm_narrative_error_code = "schema_invalid"
                            audit.llm_narrative_error_detail = "JSON validation failed"
                            errors += 1

                    except RunPodError as e:
                        audit.llm_narrative_status = "error"
                        audit.llm_narrative_error_code = "empty_output"
                        audit.llm_narrative_error_detail = str(e)[:500]
                        errors += 1

                elif status in ("FAILED", "CANCELLED"):
                    error_msg = job_data.get("error", "Unknown error")
                    audit.llm_narrative_status = "error"
                    audit.llm_narrative_error_code = "runpod_http_error"
                    audit.llm_narrative_error_detail = error_msg[:500]
                    errors += 1

                elif status == "IN_PROGRESS":
                    audit.llm_narrative_status = "running"

                # else: still IN_QUEUE, no change

            except asyncio.TimeoutError:
                # Job still running, will check again next tick
                pass
            except Exception as e:
                logger.warning(f"[FASTPATH] Error polling job {audit.llm_narrative_request_id}: {e}")

        await self.session.commit()
        return completed, errors


async def run_fastpath_tick(session: AsyncSession) -> dict:
    """Convenience function to run one fast-path tick."""
    service = FastPathService(session)
    try:
        return await service.run_tick()
    finally:
        await service.close()
