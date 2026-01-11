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
from app.models import Match, Prediction, PostMatchAudit, PredictionOutcome, Team
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
        logger.info("[FASTPATH] Selecting candidates...")
        try:
            candidates = await self._select_candidates(lookback, now)
            metrics["selected"] = len(candidates)
            logger.info(f"[FASTPATH] Selected {len(candidates)} candidates for narrative generation")
        except Exception as e:
            logger.error(f"[FASTPATH] Failed to select candidates: {e}", exc_info=True)
            metrics["errors"] = 1
            return metrics

        if not candidates:
            return metrics

        # 2. Refresh stats for matches that need it
        logger.info(f"[FASTPATH] Refreshing stats for candidates without stats")
        try:
            refreshed = await self._refresh_stats_batch(candidates, now)
            logger.info(f"[FASTPATH] Refreshed stats for {refreshed} matches")
        except Exception as stats_err:
            logger.error(f"[FASTPATH] Stats refresh failed: {stats_err}", exc_info=True)
            refreshed = 0
        metrics["refreshed"] = refreshed

        # 3. Check which matches are now ready (stats gating passed)
        logger.info(f"[FASTPATH] Checking stats gating for {len(candidates)} candidates")
        ready_matches = []
        gating_errors = 0
        gating_failed = 0
        already_ready = 0
        orphan_reset = 0
        for match in candidates:
            try:
                # Check for orphaned stats_ready_at (marked ready but stats empty)
                if match.stats_ready_at and not match.stats:
                    logger.warning(f"[FASTPATH] Match {match.id} has stats_ready_at but empty stats - resetting")
                    match.stats_ready_at = None
                    orphan_reset += 1

                if match.stats_ready_at:
                    ready_matches.append(match)
                    already_ready += 1
                else:
                    # Check if stats now pass gating
                    stats_data = {"stats": match.stats or {}}
                    passes, reason = check_stats_gating(stats_data)
                    if passes:
                        match.stats_ready_at = now
                        ready_matches.append(match)
                    else:
                        gating_failed += 1
                        logger.debug(f"[FASTPATH] Match {match.id} gating failed: {reason}")
            except Exception as loop_err:
                logger.error(f"[FASTPATH] Error checking match {match.id}: {loop_err}", exc_info=True)
                gating_errors += 1
                continue
        if orphan_reset > 0:
            logger.warning(f"[FASTPATH] Reset {orphan_reset} orphaned stats_ready_at (stats were empty)")
        logger.info(
            f"[FASTPATH] Stats gating complete: {len(ready_matches)} ready "
            f"(already_ready={already_ready}, newly_ready={len(ready_matches)-already_ready}), "
            f"gating_failed={gating_failed}, errors={gating_errors}"
        )

        metrics["stats_ready"] = len(ready_matches)
        # Note: commit deferred to _enqueue_narratives to avoid greenlet issues

        if not ready_matches:
            # Commit any finished_at/stats_ready_at changes
            try:
                await self.session.commit()
            except Exception as commit_err:
                logger.error(f"[FASTPATH] Commit failed: {commit_err}", exc_info=True)
            logger.info(f"[FASTPATH] No matches ready (stats gating). Refreshed {refreshed} stats.")
            return metrics

        # 4. Enqueue narratives for ready matches
        logger.info(f"[FASTPATH] Enqueuing {len(ready_matches)} ready matches")
        try:
            enqueued = await self._enqueue_narratives(ready_matches, now)
            metrics["enqueued"] = enqueued
        except Exception as enqueue_err:
            logger.error(f"[FASTPATH] Enqueue failed: {enqueue_err}", exc_info=True)
            metrics["errors"] += 1
            return metrics

        # 5. Poll for completions
        logger.info("[FASTPATH] Polling for completions")
        try:
            completed, errors = await self._poll_completions()
            metrics["completed"] = completed
            metrics["errors"] = errors
        except Exception as poll_err:
            logger.error(f"[FASTPATH] Poll failed: {poll_err}", exc_info=True)
            metrics["errors"] += 1

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
        - Finished within lookback window (using finished_at if available, else kickoff date)
        - Have a prediction
        - Don't have a successful LLM narrative yet
        """
        # Subquery: match IDs that already have successful LLM narrative
        matches_with_narrative = (
            select(PredictionOutcome.match_id)
            .join(PostMatchAudit, PredictionOutcome.id == PostMatchAudit.outcome_id)
            .where(PostMatchAudit.llm_narrative_status == "ok")
            .scalar_subquery()
        )

        # Subquery: match IDs that have predictions
        matches_with_predictions = (
            select(Prediction.match_id)
            .distinct()
            .scalar_subquery()
        )

        # Main query: finished matches that have predictions but no successful narrative
        result = await self.session.execute(
            select(Match)
            .options(selectinload(Match.predictions))
            .options(selectinload(Match.home_team))
            .options(selectinload(Match.away_team))
            .where(
                and_(
                    Match.status.in_(["FT", "AET", "PEN"]),
                    Match.id.in_(matches_with_predictions),
                    ~Match.id.in_(matches_with_narrative),
                    or_(
                        # Primary: use finished_at timestamp (accurate)
                        and_(
                            Match.finished_at.isnot(None),
                            Match.finished_at >= lookback,
                            Match.finished_at <= now,
                        ),
                        # Fallback for bootstrap: finished_at not set yet, use kickoff date
                        # (kickoff within last 24h covers most matches that finished recently)
                        and_(
                            Match.finished_at.is_(None),
                            Match.date >= now - timedelta(hours=24),
                            Match.date <= now,
                        ),
                    ),
                )
            )
            .order_by(Match.date.desc())
        )
        matches = result.scalars().all()

        # Set finished_at for matches that don't have it (bootstrap)
        for match in matches:
            if not match.finished_at:
                match.finished_at = now

        return matches

    async def _refresh_stats_batch(self, matches: list[Match], now: datetime) -> int:
        """Refresh stats for matches that need it. Returns count of refreshed."""
        refreshed = 0
        provider = await self._get_api_provider()

        for match in matches:
            # Force refresh if stats_ready_at is set but stats are empty (orphan recovery)
            has_stats_ready = match.stats_ready_at is not None
            has_stats = match.stats is not None and match.stats != {}
            force_refresh = has_stats_ready and not has_stats

            if force_refresh:
                logger.info(f"[FASTPATH] Force refreshing stats for orphaned match {match.id} (stats_ready_at={match.stats_ready_at}, stats={match.stats})")

            should_check = _should_check_stats(match, now)
            if not force_refresh and not should_check:
                logger.debug(f"[FASTPATH] Skipping match {match.id}: force_refresh={force_refresh}, should_check={should_check}")
                continue

            if not match.external_id:
                logger.warning(f"[FASTPATH] Match {match.id} has no external_id, skipping stats refresh")
                continue

            logger.info(f"[FASTPATH] Fetching stats for match {match.id} (external_id={match.external_id})")

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
                    logger.info(f"[FASTPATH] Refreshed stats for match {match.id}: home_keys={list(home_stats.keys())[:3]}")
                else:
                    logger.warning(f"[FASTPATH] No stats in API response for match {match.id} (response len={len(response)})")

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

        # Pre-load team names to avoid lazy loading issues
        team_ids = set()
        for match in matches[:self.settings.FASTPATH_MAX_CONCURRENT_JOBS]:
            if match.home_team_id:
                team_ids.add(match.home_team_id)
            if match.away_team_id:
                team_ids.add(match.away_team_id)

        team_names = {}
        if team_ids:
            result = await self.session.execute(
                select(Team.id, Team.name).where(Team.id.in_(team_ids))
            )
            for team_id, team_name in result.all():
                team_names[team_id] = team_name

        try:
            for match in matches[:self.settings.FASTPATH_MAX_CONCURRENT_JOBS]:
                # Get team names from pre-loaded cache
                home_team_name = team_names.get(match.home_team_id, "Local")
                away_team_name = team_names.get(match.away_team_id, "Visitante")

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
                match_data = self._build_match_data(match, prediction, home_team_name, away_team_name)

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
        """Get or create PredictionOutcome and PostMatchAudit for a match.

        Uses explicit queries to avoid lazy loading / greenlet issues.
        """
        # Query outcome and audit with explicit JOIN (no lazy loading)
        result = await self.session.execute(
            select(PredictionOutcome, PostMatchAudit)
            .outerjoin(PostMatchAudit, PredictionOutcome.id == PostMatchAudit.outcome_id)
            .where(PredictionOutcome.match_id == match.id)
        )
        row = result.first()

        if row:
            outcome, audit = row
            if outcome and audit:
                return outcome, audit
            # Have outcome but no audit - will create audit below
            if outcome:
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

        # No outcome exists - need to create both
        prediction = await self._get_best_prediction(match.id)
        if not prediction:
            return None, None

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

        # Create audit
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

    def _build_match_data(
        self,
        match: Match,
        prediction: Prediction,
        home_team_name: str,
        away_team_name: str,
    ) -> dict:
        """Build match_data dict for narrative prompt.

        Team names passed explicitly to avoid lazy loading issues.
        """
        # Determine predicted result and if correct
        probs = {
            "home": prediction.home_prob,
            "draw": prediction.draw_prob,
            "away": prediction.away_prob,
        }
        predicted_result = max(probs, key=probs.get)
        confidence = probs[predicted_result]

        # Determine actual result
        home_goals = match.home_goals or 0
        away_goals = match.away_goals or 0
        if home_goals > away_goals:
            actual_result = "home"
        elif away_goals > home_goals:
            actual_result = "away"
        else:
            actual_result = "draw"

        return {
            "match_id": match.id,
            "home_team": home_team_name,
            "away_team": away_team_name,
            "league_name": "",  # Could add league lookup if needed
            "date": match.date.isoformat() if match.date else "",
            "home_goals": home_goals,
            "away_goals": away_goals,
            "stats": match.stats or {},
            "events": [],  # Could fetch events if needed
            "prediction": {
                "probabilities": {
                    "home": prediction.home_prob,
                    "draw": prediction.draw_prob,
                    "away": prediction.away_prob,
                },
                "predicted_result": predicted_result,
                "confidence": confidence,
                "correct": predicted_result == actual_result,
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
        logger.info(f"[FASTPATH] Found {len(pending_audits)} audits to poll for completions")

        # Debug: check for orphaned audits (in_queue but no request_id)
        orphan_result = await self.session.execute(
            select(PostMatchAudit.id, PostMatchAudit.llm_narrative_status)
            .where(
                and_(
                    PostMatchAudit.llm_narrative_status.in_(["in_queue", "running"]),
                    or_(
                        PostMatchAudit.llm_narrative_request_id.is_(None),
                        PostMatchAudit.llm_narrative_request_id == "",
                    ),
                )
            )
        )
        orphans = orphan_result.all()
        if orphans:
            orphan_ids = [o[0] for o in orphans]
            logger.warning(f"[FASTPATH] Found {len(orphans)} orphaned audits (in_queue/running but no request_id): {orphan_ids}")
            # Reset orphaned audits so they can be re-enqueued
            for orphan_id in orphan_ids:
                orphan_audit = await self.session.get(PostMatchAudit, orphan_id)
                if orphan_audit:
                    orphan_audit.llm_narrative_status = None
                    orphan_audit.llm_narrative_error_code = None
                    orphan_audit.llm_narrative_error_detail = None
                    logger.info(f"[FASTPATH] Reset orphaned audit {orphan_id} for re-enqueue")

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
