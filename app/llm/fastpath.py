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
from app.teams.overrides import preload_team_overrides, resolve_team_display
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
from app.llm.gemini_client import GeminiClient, GeminiError
from app.llm.claim_validator import (
    PROMPT_VERSION,
    sanitize_payload_for_storage,
    compute_payload_hash,
    validate_narrative_claims,
    should_reject_narrative,
    get_rejection_reason,
    sanitize_narrative_body,
)
from app.telemetry.metrics import (
    llm_unsupported_claims_total,
    llm_narratives_validated_total,
    llm_requests_total,
    llm_latency_ms,
    llm_tokens_total,
    llm_cost_usd,
)

logger = logging.getLogger(__name__)

# Circuit breaker state for LLM provider fallback
# Prevents "spamming" fallback to RunPod when Gemini fails repeatedly
_circuit_breaker = {
    "gemini_consecutive_failures": 0,
    "gemini_circuit_open": False,
    "gemini_circuit_opened_at": None,
    "threshold": 3,  # Open circuit after 3 consecutive failures
    "reset_after_seconds": 300,  # Try Gemini again after 5 minutes
}


def _check_circuit_breaker() -> tuple[bool, str]:
    """
    Check if circuit breaker allows Gemini requests.

    Returns:
        (should_use_gemini, reason)
    """
    if not _circuit_breaker["gemini_circuit_open"]:
        return True, "circuit_closed"

    # Check if enough time has passed to retry
    opened_at = _circuit_breaker["gemini_circuit_opened_at"]
    if opened_at:
        elapsed = (datetime.utcnow() - opened_at).total_seconds()
        if elapsed >= _circuit_breaker["reset_after_seconds"]:
            # Half-open: allow one request to test
            logger.info("[CIRCUIT] Half-open: allowing Gemini test request")
            return True, "half_open"

    return False, "circuit_open"


def _record_gemini_success():
    """Record successful Gemini request, reset circuit breaker."""
    _circuit_breaker["gemini_consecutive_failures"] = 0
    if _circuit_breaker["gemini_circuit_open"]:
        logger.info("[CIRCUIT] Gemini recovered, closing circuit breaker")
        _circuit_breaker["gemini_circuit_open"] = False
        _circuit_breaker["gemini_circuit_opened_at"] = None

    # Emit metrics
    _update_circuit_breaker_metrics()


def _record_gemini_failure():
    """Record failed Gemini request, potentially open circuit."""
    _circuit_breaker["gemini_consecutive_failures"] += 1
    failures = _circuit_breaker["gemini_consecutive_failures"]
    threshold = _circuit_breaker["threshold"]

    if failures >= threshold and not _circuit_breaker["gemini_circuit_open"]:
        logger.warning(
            f"[CIRCUIT] Opening circuit breaker after {failures} consecutive Gemini failures. "
            f"Will retry in {_circuit_breaker['reset_after_seconds']}s"
        )
        _circuit_breaker["gemini_circuit_open"] = True
        _circuit_breaker["gemini_circuit_opened_at"] = datetime.utcnow()

    # Emit metrics
    _update_circuit_breaker_metrics()


def _update_circuit_breaker_metrics():
    """Update Prometheus gauges for circuit breaker state."""
    try:
        from app.telemetry.metrics import set_circuit_breaker_state
        set_circuit_breaker_state(
            provider="gemini",
            is_open=_circuit_breaker["gemini_circuit_open"],
            consecutive_failures=_circuit_breaker["gemini_consecutive_failures"],
        )
    except Exception as e:
        logger.debug(f"[METRICS] Failed to update circuit breaker metrics: {e}")


# Backoff profile for stats refresh (minutes since FT -> seconds between checks)
STATS_BACKOFF_PROFILE = [
    (5, 60),      # < 5 min since FT: check every 1 min
    (15, 120),    # < 15 min: every 2 min
    (60, 300),    # < 60 min: every 5 min
    (90, 900),    # < 90 min: every 15 min
]

# Events configuration
MAX_EVENTS_PER_MATCH = 10  # Cap events to avoid token bloat
EVENT_PRIORITY = {
    "Goal": 1,
    "Card": 2,
    "Var": 3,
    "Penalty": 1,  # Same as goal
    "subst": 4,
}


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
        self.provider_name = self.settings.NARRATIVE_PROVIDER.lower()

        # Initialize appropriate LLM client based on feature flag
        if self.provider_name == "gemini":
            self.llm_client = GeminiClient()
            logger.info("FastPath using Gemini provider")
        else:
            self.llm_client = RunPodClient()
            logger.info("FastPath using RunPod provider")

        # Keep runpod reference for backward compatibility
        self.runpod = self.llm_client if self.provider_name == "runpod" else None
        self._api_provider = None

    async def close(self):
        """Close resources."""
        await self.llm_client.close()
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
        - Haven't exceeded max retry attempts (3)
        """
        MAX_LLM_ATTEMPTS = 3

        # Subquery: match IDs that already have successful LLM narrative
        matches_with_narrative = (
            select(PredictionOutcome.match_id)
            .join(PostMatchAudit, PredictionOutcome.id == PostMatchAudit.outcome_id)
            .where(PostMatchAudit.llm_narrative_status == "ok")
            .scalar_subquery()
        )

        # Subquery: match IDs that have exceeded max retry attempts
        matches_max_retries = (
            select(PredictionOutcome.match_id)
            .join(PostMatchAudit, PredictionOutcome.id == PostMatchAudit.outcome_id)
            .where(
                and_(
                    PostMatchAudit.llm_narrative_status == "error",
                    PostMatchAudit.llm_narrative_attempts >= MAX_LLM_ATTEMPTS,
                )
            )
            .scalar_subquery()
        )

        # Subquery: match IDs that have predictions
        matches_with_predictions = (
            select(Prediction.match_id)
            .distinct()
            .scalar_subquery()
        )

        # Main query: finished matches that have predictions but no successful narrative
        # and haven't exceeded max retry attempts
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
                    ~Match.id.in_(matches_max_retries),  # Exclude max-retried matches
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

            # Also need to fetch if events are missing
            needs_events = not match.events

            if force_refresh:
                logger.info(f"[FASTPATH] Force refreshing stats for orphaned match {match.id} (stats_ready_at={match.stats_ready_at}, stats={match.stats})")

            should_check = _should_check_stats(match, now)
            if not force_refresh and not should_check and not needs_events:
                logger.debug(f"[FASTPATH] Skipping match {match.id}: force_refresh={force_refresh}, should_check={should_check}, needs_events={needs_events}")
                continue

            if not match.external_id:
                logger.warning(f"[FASTPATH] Match {match.id} has no external_id, skipping stats refresh")
                continue

            logger.info(f"[FASTPATH] Fetching data for match {match.id} (external_id={match.external_id}, needs_stats={should_check or force_refresh}, needs_events={needs_events})")

            try:
                # Only fetch stats if needed (not just for events)
                needs_stats = should_check or force_refresh
                if needs_stats:
                    stats_data = await provider._rate_limited_request(
                        "fixtures/statistics",
                        {"fixture": match.external_id},
                        entity="stats"
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

                # Fetch events if not already present
                if needs_events:
                    try:
                        events_data = await provider._rate_limited_request(
                            "fixtures/events",
                            {"fixture": match.external_id},
                            entity="events"
                        )
                        events_response = events_data.get("response", [])
                        if events_response:
                            parsed_events = self._parse_events(events_response)
                            match.events = parsed_events
                            logger.info(f"[FASTPATH] Fetched {len(parsed_events)} events for match {match.id}")
                        else:
                            logger.warning(f"[FASTPATH] No events in API response for match {match.id}")
                    except Exception as events_err:
                        logger.warning(f"[FASTPATH] Failed to fetch events for match {match.id}: {events_err}")

            except Exception as e:
                logger.warning(f"[FASTPATH] Failed to refresh data for match {match.id}: {e}")
                if needs_stats:
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

    def _parse_events(self, events_list: list) -> list:
        """
        Parse API-Football events into compact format.

        Prioritizes: Goals > Cards (Red > Yellow) > VAR > Subst
        Returns up to MAX_EVENTS_PER_MATCH events sorted by priority then minute.
        """
        parsed = []
        for event in events_list:
            event_type = event.get("type", "")
            detail = event.get("detail", "")

            # Determine priority (lower = more important)
            if event_type == "Goal":
                priority = 1
            elif event_type == "Card":
                priority = 2 if detail == "Red Card" else 3  # Red before Yellow
            elif event_type == "Var":
                priority = 4
            elif event_type == "subst":
                priority = 5
            else:
                priority = 6  # Other events

            parsed.append({
                "minute": event.get("time", {}).get("elapsed"),
                "extra_minute": event.get("time", {}).get("extra"),
                "type": event_type,
                "detail": detail,
                "team_id": event.get("team", {}).get("id"),  # External team ID for timeline mapping
                "team_name": event.get("team", {}).get("name"),
                "player_name": event.get("player", {}).get("name"),
                "assist": event.get("assist", {}).get("name"),
                "_priority": priority,
            })

        # Sort by priority (ascending), then by minute (ascending)
        parsed.sort(key=lambda x: (x.get("_priority", 99), x.get("minute") or 0))

        # Take top events and remove internal priority field
        result = []
        for e in parsed[:MAX_EVENTS_PER_MATCH]:
            event_clean = {k: v for k, v in e.items() if k != "_priority" and v is not None}
            result.append(event_clean)

        return result

    async def _enqueue_narratives(self, matches: list[Match], now: datetime) -> int:
        """
        Enqueue narrative generation for ready matches.

        Creates outcome/audit records if needed and submits to RunPod.
        """
        enqueued = 0
        generator = NarrativeGenerator()

        # Pre-load team info to avoid lazy loading issues
        team_ids = set()
        for match in matches[:self.settings.FASTPATH_MAX_CONCURRENT_JOBS]:
            if match.home_team_id:
                team_ids.add(match.home_team_id)
            if match.away_team_id:
                team_ids.add(match.away_team_id)

        team_info = {}  # team_id -> {"name": str, "external_id": int, "original_name": str}
        external_ids_for_overrides = []
        if team_ids:
            result = await self.session.execute(
                select(Team.id, Team.name, Team.external_id).where(Team.id.in_(team_ids))
            )
            for team_id, team_name, external_id in result.all():
                team_info[team_id] = {"name": team_name, "external_id": external_id, "original_name": team_name}
                if external_id:
                    external_ids_for_overrides.append(external_id)

        # Load team overrides (e.g., La Equidad → Internacional de Bogotá)
        team_overrides = {}
        if external_ids_for_overrides:
            try:
                team_overrides = await preload_team_overrides(self.session, external_ids_for_overrides)
            except Exception as override_err:
                logger.warning(f"[FASTPATH] Failed to load team overrides: {override_err}")

        try:
            for match in matches[:self.settings.FASTPATH_MAX_CONCURRENT_JOBS]:
                # Get team info from pre-loaded cache
                home_info = team_info.get(match.home_team_id, {"name": "Local", "external_id": None, "original_name": "Local"})
                away_info = team_info.get(match.away_team_id, {"name": "Visitante", "external_id": None, "original_name": "Visitante"})

                # Apply team overrides based on match date
                if team_overrides and match.date:
                    home_ext_id = home_info.get("external_id")
                    away_ext_id = away_info.get("external_id")
                    if home_ext_id:
                        home_display = resolve_team_display(
                            team_overrides, home_ext_id, match.date,
                            home_info.get("original_name", "Local"),
                        )
                        if home_display.is_override:
                            home_info["name"] = home_display.name
                            logger.debug(f"[FASTPATH] Override applied: {home_display.original_name} → {home_display.name}")
                    if away_ext_id:
                        away_display = resolve_team_display(
                            team_overrides, away_ext_id, match.date,
                            away_info.get("original_name", "Visitante"),
                        )
                        if away_display.is_override:
                            away_info["name"] = away_display.name
                            logger.debug(f"[FASTPATH] Override applied: {away_display.original_name} → {away_display.name}")

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

                # Fetch league and team context for narratives (best-effort)
                league_context = None
                home_team_context = None
                away_team_context = None
                try:
                    from app.aggregates import get_league_context, get_team_context
                    league_context = await get_league_context(
                        self.session, match.league_id, match.season
                    )
                    if match.home_team_id:
                        home_team_context = await get_team_context(
                            self.session, match.league_id, match.season, match.home_team_id
                        )
                    if match.away_team_id:
                        away_team_context = await get_team_context(
                            self.session, match.league_id, match.season, match.away_team_id
                        )
                except Exception as ctx_err:
                    logger.debug(f"[FASTPATH] Failed to fetch context for match {match.id}: {ctx_err}")

                # Build match data for prompt (with team aliases and context)
                match_data = self._build_match_data(
                    match, prediction, home_info, away_info,
                    league_context=league_context,
                    home_team_context=home_team_context,
                    away_team_context=away_team_context,
                )

                # Check gating
                passes, reason = check_stats_gating(match_data)
                if not passes:
                    audit.llm_narrative_status = "skipped"
                    audit.llm_narrative_error_code = "gating_skipped"
                    audit.llm_narrative_error_detail = reason
                    continue

                # Submit to LLM provider
                try:
                    prompt, home_pack, away_pack = build_narrative_prompt(match_data)

                    # Persist payload for traceability (before sending to LLM)
                    sanitized_payload = sanitize_payload_for_storage(match_data)
                    sanitized_payload["home_alias_pack"] = home_pack
                    sanitized_payload["away_alias_pack"] = away_pack
                    payload_hash = compute_payload_hash(sanitized_payload)

                    audit.llm_prompt_version = PROMPT_VERSION
                    audit.llm_prompt_input_json = sanitized_payload
                    audit.llm_prompt_input_hash = payload_hash

                    if self.provider_name == "gemini":
                        # Check circuit breaker before calling Gemini
                        use_gemini, circuit_state = _check_circuit_breaker()

                        if not use_gemini:
                            # Circuit is open - skip this match (don't fallback to avoid double cost)
                            logger.warning(
                                f"[FASTPATH] Skipping match {match.id}: Gemini circuit breaker open. "
                                f"Set NARRATIVE_PROVIDER=runpod for immediate fallback."
                            )
                            llm_requests_total.labels(provider="gemini", status="skipped").inc()
                            audit.llm_narrative_status = "skipped"
                            audit.llm_narrative_error_code = "circuit_breaker_open"
                            audit.llm_narrative_error_detail = (
                                "Gemini circuit breaker open after consecutive failures. "
                                "Match will be retried when circuit resets."
                            )
                            continue

                        # Gemini is synchronous - generate immediately
                        result = await self.llm_client.generate(prompt)

                        # Record metrics (latency and tokens regardless of outcome)
                        llm_latency_ms.labels(provider="gemini").observe(result.exec_ms)
                        llm_tokens_total.labels(provider="gemini", direction="input").inc(result.tokens_in)
                        llm_tokens_total.labels(provider="gemini", direction="output").inc(result.tokens_out)
                        # Gemini 2.0 Flash pricing: $0.075/1M input, $0.30/1M output
                        cost = (result.tokens_in * 0.075 + result.tokens_out * 0.30) / 1_000_000
                        llm_cost_usd.labels(provider="gemini").inc(cost)

                        if result.status == "COMPLETED":
                            _record_gemini_success()
                            llm_requests_total.labels(provider="gemini", status="ok").inc()

                            # Parse and validate Gemini response (same as RunPod)
                            text = result.text
                            audit.llm_output_raw = text[:5000] if text else None

                            parsed = parse_json_response(text)
                            outcome = await self.session.get(PredictionOutcome, audit.outcome_id)
                            match_id_for_validation = outcome.match_id if outcome else 0

                            if parsed and validate_narrative_json(parsed, match_id_for_validation):
                                # Sanitize narrative body
                                narrative_obj = parsed.get("narrative", {})
                                if isinstance(narrative_obj, dict) and "body" in narrative_obj:
                                    original_body = narrative_obj.get("body", "")
                                    sanitized_body, _ = sanitize_narrative_body(original_body)
                                    if sanitized_body != original_body:
                                        narrative_obj["body"] = sanitized_body
                                        parsed["narrative"] = narrative_obj

                                audit.llm_narrative_status = "ok"
                                audit.llm_narrative_json = parsed
                            else:
                                # Schema validation failed
                                audit.llm_narrative_status = "error"
                                audit.llm_narrative_error_code = "schema_invalid"
                                audit.llm_narrative_error_detail = f"JSON validation failed. Text len: {len(text) if text else 0}"
                                logger.warning(f"[FASTPATH] Gemini response invalid for match {match.id}")

                            audit.llm_narrative_request_id = f"gemini-{match.id}"
                            audit.llm_narrative_model = "gemini-2.0-flash"
                            audit.llm_narrative_generated_at = datetime.utcnow()
                            audit.llm_narrative_delay_ms = 0
                            audit.llm_narrative_exec_ms = result.exec_ms
                            audit.llm_narrative_tokens_in = result.tokens_in
                            audit.llm_narrative_tokens_out = result.tokens_out
                            enqueued += 1
                            logger.info(f"[FASTPATH] Gemini completed match {match.id} in {result.exec_ms}ms")
                        else:
                            _record_gemini_failure()
                            llm_requests_total.labels(provider="gemini", status="error").inc()
                            audit.llm_narrative_status = "error"
                            audit.llm_narrative_error_code = "gemini_error"
                            audit.llm_narrative_error_detail = result.error[:500] if result.error else "Unknown error"
                            logger.error(f"[FASTPATH] Gemini failed for match {match.id}: {result.error}")
                    else:
                        # RunPod async pattern
                        job_id = await self.llm_client.run_job(prompt)
                        llm_requests_total.labels(provider="runpod", status="enqueued").inc()
                        audit.llm_narrative_status = "in_queue"
                        audit.llm_narrative_request_id = job_id
                        audit.llm_narrative_model = "qwen-vllm"
                        enqueued += 1
                        logger.info(f"[FASTPATH] Enqueued match {match.id} -> job {job_id}, hash={payload_hash[:12]}")

                    # Increment attempt counter (or set to 1 if first attempt)
                    current_attempts = audit.llm_narrative_attempts or 0
                    audit.llm_narrative_attempts = current_attempts + 1

                except (RunPodError, GeminiError) as e:
                    audit.llm_narrative_status = "error"
                    audit.llm_narrative_error_code = f"{self.provider_name}_http_error"
                    audit.llm_narrative_error_detail = str(e)[:500]
                    # Increment attempt counter on error
                    current_attempts = audit.llm_narrative_attempts or 0
                    audit.llm_narrative_attempts = current_attempts + 1
                    logger.error(f"[FASTPATH] Failed to process match {match.id}: {e}")

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
        """Get frozen baseline prediction (the one user saw), or latest baseline if none frozen.

        IMPORTANT: Must use baseline model (MODEL_VERSION), not shadow/experimental models.
        Shadow predictions (two_stage) are for A/B testing only and should never be shown to users.
        """
        from app.config import get_settings
        settings = get_settings()
        baseline_version = settings.MODEL_VERSION  # e.g., "v1.0.0"

        # Try frozen baseline prediction first (this is what user saw before match)
        result = await self.session.execute(
            select(Prediction)
            .where(
                and_(
                    Prediction.match_id == match_id,
                    Prediction.is_frozen == True,
                    Prediction.model_version == baseline_version,
                )
            )
            .order_by(Prediction.frozen_at.desc().nullslast())
            .limit(1)
        )
        prediction = result.scalar_one_or_none()
        if prediction:
            return prediction

        # Fall back to latest baseline (not frozen yet)
        result = await self.session.execute(
            select(Prediction)
            .where(
                and_(
                    Prediction.match_id == match_id,
                    Prediction.model_version == baseline_version,
                )
            )
            .order_by(Prediction.created_at.desc())
            .limit(1)
        )
        return result.scalar_one_or_none()

    def _build_match_data(
        self,
        match: Match,
        prediction: Prediction,
        home_info: dict,
        away_info: dict,
        league_context: Optional[dict] = None,
        home_team_context: Optional[dict] = None,
        away_team_context: Optional[dict] = None,
    ) -> dict:
        """Build match_data dict for narrative prompt.

        Team info passed explicitly to avoid lazy loading issues.
        Includes team_aliases for LLM to use (prevents hallucinated nicknames).
        Includes derived_facts for verifiable pre-computed facts (P1 anti-hallucination).
        Includes league_context and team_context for relative comparisons (P0/P1 aggregates).
        """
        from app.llm.team_aliases import get_team_aliases
        from app.llm.derived_facts import build_derived_facts

        home_team_name = home_info.get("name", "Local")
        away_team_name = away_info.get("name", "Visitante")
        home_external_id = home_info.get("external_id")
        away_external_id = away_info.get("external_id")

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

        # Build team aliases (only these can be used by LLM)
        team_aliases = {
            "home": get_team_aliases(home_external_id, home_team_name, is_home=True),
            "away": get_team_aliases(away_external_id, away_team_name, is_home=False),
        }

        return {
            "match_id": match.id,
            "home_team": home_team_name,
            "away_team": away_team_name,
            "home_team_id": home_external_id,
            "away_team_id": away_external_id,
            "team_aliases": team_aliases,
            "league_name": "",  # Could add league lookup if needed
            "date": match.date.isoformat() if match.date else "",
            "home_goals": home_goals,
            "away_goals": away_goals,
            "stats": match.stats or {},
            "events": match.events or [],
            "venue": {
                "name": match.venue_name,
                "city": match.venue_city,
            } if match.venue_name else {},
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
            # P1/P2: Pre-computed verifiable facts (reduces inference hallucinations)
            "derived_facts": build_derived_facts(
                home_goals=home_goals,
                away_goals=away_goals,
                home_team=home_team_name,
                away_team=away_team_name,
                events=match.events or [],
                stats=match.stats or {},
                match_status=match.status,
                # P2: Extended context for richer narratives
                market_odds={
                    "home": match.odds_home,
                    "draw": match.odds_draw,
                    "away": match.odds_away,
                } if match.odds_home else None,
                model_probs={
                    "home": prediction.home_prob,
                    "draw": prediction.draw_prob,
                    "away": prediction.away_prob,
                },
                value_bet=prediction.frozen_value_bets[0] if prediction.frozen_value_bets else None,
                prediction_correct=(predicted_result == actual_result),
                # P0/P1: League and team context for relative comparisons
                league_context=league_context,
                home_team_context=home_team_context,
                away_team_context=away_team_context,
            ),
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

                        # Debug: log raw response when empty or short
                        if not text or len(text) < 50:
                            logger.warning(f"[FASTPATH] Empty/short LLM response for audit {audit.id}: len={len(text) if text else 0}, text={text[:200] if text else 'None'}")

                        # Store raw output for debugging
                        audit.llm_output_raw = text[:5000] if text else None

                        parsed = parse_json_response(text)
                        # Get match_id for validation
                        outcome = await self.session.get(PredictionOutcome, audit.outcome_id)
                        match_id = outcome.match_id if outcome else 0

                        if parsed and validate_narrative_json(parsed, match_id):
                            # Sanitize narrative body to remove control tokens (P0 fix)
                            narrative_obj = parsed.get("narrative", {})
                            if isinstance(narrative_obj, dict) and "body" in narrative_obj:
                                original_body = narrative_obj.get("body", "")
                                sanitized_body, token_warnings = sanitize_narrative_body(original_body)
                                if sanitized_body != original_body:
                                    narrative_obj["body"] = sanitized_body
                                    parsed["narrative"] = narrative_obj
                                    logger.info(
                                        f"[FASTPATH] Sanitized {len(token_warnings)} control tokens "
                                        f"from audit {audit.id}"
                                    )
                            else:
                                token_warnings = []

                            # Schema is valid, now validate claims against payload
                            # P0 FIX: narrative can be dict {title, body} or str
                            narrative_obj = parsed.get("narrative", "")
                            if isinstance(narrative_obj, dict):
                                narrative_text = narrative_obj.get("body", "") or ""
                            else:
                                narrative_text = str(narrative_obj or "")
                            payload_for_claims = audit.llm_prompt_input_json or {}

                            claim_errors = validate_narrative_claims(
                                narrative_text,
                                payload_for_claims,
                                strict=True
                            )

                            # Add token sanitization warnings to claim_errors
                            if token_warnings:
                                claim_errors = claim_errors or []
                                claim_errors.extend(token_warnings)

                            # Check for normalization warning from validate_narrative_json
                            normalization_warning = parsed.pop("_normalization_warning", None)
                            if normalization_warning:
                                claim_errors = claim_errors or []
                                claim_errors.append(normalization_warning)
                                logger.info(
                                    f"[FASTPATH] Narrative normalized for audit {audit.id}: "
                                    f"{normalization_warning.get('extra_keys_count')} extra keys"
                                )

                            audit.llm_validation_errors = claim_errors if claim_errors else None

                            if should_reject_narrative(claim_errors):
                                # Narrative has unsupported claims - reject
                                rejection_reason = get_rejection_reason(claim_errors)
                                audit.llm_narrative_status = "error"
                                audit.llm_narrative_error_code = "unsupported_claim"
                                audit.llm_narrative_error_detail = rejection_reason[:500]
                                audit.llm_narrative_json = parsed  # Store anyway for debugging
                                errors += 1

                                # Telemetry: track claim types + request status
                                llm_requests_total.labels(provider="runpod", status="rejected").inc()
                                llm_narratives_validated_total.labels(status="rejected").inc()
                                for err in claim_errors:
                                    if err.get("severity") == "error":
                                        llm_unsupported_claims_total.labels(
                                            claim_type=err.get("claim", "unknown")
                                        ).inc()

                                logger.warning(
                                    f"[FASTPATH] Narrative rejected for audit {audit.id}: {rejection_reason}"
                                )
                            else:
                                # All validations passed
                                audit.llm_narrative_status = "ok"
                                audit.llm_narrative_json = parsed
                                audit.llm_narrative_generated_at = datetime.utcnow()
                                audit.llm_narrative_delay_ms = meta["delay_ms"]
                                audit.llm_narrative_exec_ms = meta["exec_ms"]
                                audit.llm_narrative_tokens_in = tokens_in
                                audit.llm_narrative_tokens_out = tokens_out
                                audit.llm_narrative_worker_id = meta["worker_id"]
                                completed += 1

                                # Telemetry: track successful validation + full metrics
                                llm_requests_total.labels(provider="runpod", status="ok").inc()
                                llm_narratives_validated_total.labels(status="ok").inc()
                                llm_latency_ms.labels(provider="runpod").observe(meta["exec_ms"])
                                llm_tokens_total.labels(provider="runpod", direction="input").inc(tokens_in)
                                llm_tokens_total.labels(provider="runpod", direction="output").inc(tokens_out)
                                # RunPod/Qwen pricing estimate: $0.20/1M tokens
                                cost = (tokens_in + tokens_out) * 0.20 / 1_000_000
                                llm_cost_usd.labels(provider="runpod").inc(cost)

                                logger.info(
                                    f"[FASTPATH] Completed narrative for audit {audit.id}: "
                                    f"tokens={tokens_in}/{tokens_out}, exec={meta['exec_ms']}ms"
                                )
                        else:
                            # Log what we received for debugging
                            logger.warning(
                                f"[FASTPATH] Schema validation failed for audit {audit.id}: "
                                f"text_len={len(text)}, parsed={'yes' if parsed else 'no'}, "
                                f"first_100={text[:100] if text else 'None'}"
                            )
                            audit.llm_narrative_status = "error"
                            audit.llm_narrative_error_code = "schema_invalid"
                            audit.llm_narrative_error_detail = f"JSON validation failed. Text len: {len(text)}"
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
                    llm_requests_total.labels(provider="runpod", status="error").inc()
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
