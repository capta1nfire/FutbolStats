"""
Event Bus — DB-backed event dispatch + Sweeper Queue.

Phase 2, P2-09.

Design:
- asyncio.Queue for immediate in-process dispatch
- DB state (match_lineups.lineup_detected_at + predictions.asof_timestamp) is source of truth
- Sweeper reconciles every 2min: finds matches with lineup but no post-lineup prediction
- FOR UPDATE SKIP LOCKED prevents concurrent sweeper runs from double-processing

ATI Directive: FOR UPDATE SKIP LOCKED mandatory for sweeper dedupe.
GDT Directive: DB-backed — queue loss on crash is recoverable via sweeper.
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

from sqlalchemy import text

logger = logging.getLogger("futbolstats.events")

# ── Event type constants ─────────────────────────────────────────────────────
LINEUP_CONFIRMED = "LINEUP_CONFIRMED"


# ── Event ────────────────────────────────────────────────────────────────────
class Event:
    """Immutable event payload."""

    __slots__ = ("event_type", "payload", "created_at")

    def __init__(self, event_type: str, payload: Dict[str, Any]):
        self.event_type = event_type
        self.payload = payload
        self.created_at = datetime.now(timezone.utc)

    def __repr__(self):
        return f"Event({self.event_type}, match_id={self.payload.get('match_id')}, src={self.payload.get('source', '?')})"


# ── EventBus ─────────────────────────────────────────────────────────────────
class EventBus:
    """
    In-memory event bus with async consumer.

    Events are dispatched to registered handlers in subscription order.
    If no handler is registered for an event type, a warning is logged.
    On crash, the Sweeper Queue recovers missed events from DB state.
    """

    def __init__(self, max_queue_size: int = 1000):
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=max_queue_size)
        self._handlers: Dict[str, List[Callable]] = {}
        self._task: Optional[asyncio.Task] = None
        self._running = False
        # In-progress match IDs to prevent concurrent handler execution
        self._processing: set = set()

    def subscribe(self, event_type: str, handler: Callable):
        """Register an async handler for an event type."""
        if event_type not in self._handlers:
            self._handlers[event_type] = []
        self._handlers[event_type].append(handler)
        logger.info(f"EventBus: subscribed {handler.__name__} to {event_type}")

    async def emit(self, event_type: str, payload: Dict[str, Any]):
        """Emit an event to the queue for async processing."""
        event = Event(event_type, payload)
        try:
            self._queue.put_nowait(event)
            logger.info(f"EventBus: emitted {event}")
        except asyncio.QueueFull:
            logger.error(f"EventBus: queue full ({self._queue.maxsize}), dropping {event}")

    async def start(self):
        """Start the background consumer task."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._consumer_loop())
        logger.info("EventBus: started consumer loop")

    async def stop(self):
        """Graceful shutdown: drain queue then stop."""
        self._running = False
        if self._task:
            # Sentinel to unblock the consumer
            await self._queue.put(None)
            try:
                await asyncio.wait_for(self._task, timeout=10.0)
            except asyncio.TimeoutError:
                self._task.cancel()
                logger.warning("EventBus: consumer did not finish in 10s, cancelled")
            self._task = None
        logger.info(f"EventBus: stopped (pending={self._queue.qsize()})")

    async def _consumer_loop(self):
        """Process events sequentially from the queue."""
        while self._running:
            try:
                event = await self._queue.get()
                if event is None:
                    break
                await self._dispatch(event)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"EventBus: consumer loop error: {e}", exc_info=True)

    async def _dispatch(self, event: Event):
        """Dispatch event to all registered handlers with in-progress guard."""
        match_id = event.payload.get("match_id")

        # Dedupe: skip if this match is already being processed
        if match_id and match_id in self._processing:
            logger.info(f"EventBus: match_id={match_id} already in-progress, skipping")
            return

        handlers = self._handlers.get(event.event_type, [])
        if not handlers:
            logger.warning(f"EventBus: no handlers for {event.event_type}")
            return

        if match_id:
            self._processing.add(match_id)
        try:
            for handler in handlers:
                try:
                    await handler(event)
                except Exception as e:
                    logger.error(
                        f"EventBus: handler {handler.__name__} failed for {event}: {e}",
                        exc_info=True,
                    )
        finally:
            if match_id:
                self._processing.discard(match_id)

    @property
    def pending_count(self) -> int:
        return self._queue.qsize()

    @property
    def processing_count(self) -> int:
        return len(self._processing)


# ── Sweeper Queue ────────────────────────────────────────────────────────────

async def sweep_missed_lineups(session_factory) -> List[Dict[str, Any]]:
    """
    Find matches with confirmed lineups but no post-lineup prediction.

    Uses FOR UPDATE OF m SKIP LOCKED to prevent concurrent sweeper runs
    from processing the same match.

    Returns list of dicts with match_id and lineup_detected_at.
    """
    async with session_factory() as session:
        result = await session.execute(text("""
            SELECT m.id AS match_id,
                   ml.lineup_detected_at
            FROM matches m
            -- match_lineups has 2 rows per match (home/away). Join ONLY home row to avoid duplicates.
            JOIN match_lineups ml ON ml.match_id = m.id AND ml.team_id = m.home_team_id
            WHERE m.date BETWEEN NOW() AND NOW() + INTERVAL '65 minutes'
              AND m.status = 'NS'
              AND ml.lineup_detected_at IS NOT NULL
              AND NOT EXISTS (
                  SELECT 1 FROM predictions p
                  WHERE p.match_id = m.id
                    AND p.asof_timestamp >= ml.lineup_detected_at
              )
            FOR UPDATE OF m SKIP LOCKED
        """))
        rows = result.fetchall()
        # Commit immediately to release FOR UPDATE locks
        await session.commit()

        matches = [
            {"match_id": row.match_id, "lineup_detected_at": row.lineup_detected_at}
            for row in rows
        ]

        if matches:
            ids = [m["match_id"] for m in matches]
            logger.info(f"[SWEEPER] Found {len(matches)} matches needing post-lineup prediction: {ids}")
        else:
            logger.debug("[SWEEPER] No missed lineups found")

        return matches


async def run_sweeper(bus: EventBus, session_factory) -> int:
    """
    Run the sweeper and emit LINEUP_CONFIRMED events for missed matches.

    Called by the scheduler every 2 minutes.
    Returns count of events emitted.
    """
    try:
        matches = await sweep_missed_lineups(session_factory)
        emitted = 0
        for match in matches:
            await bus.emit(LINEUP_CONFIRMED, {
                "match_id": match["match_id"],
                "lineup_detected_at": match["lineup_detected_at"],
                "source": "sweeper",
            })
            emitted += 1
        return emitted
    except Exception as e:
        logger.error(f"[SWEEPER] Failed: {e}", exc_info=True)
        return 0


# ── Singleton ────────────────────────────────────────────────────────────────

_bus_instance: Optional[EventBus] = None


def get_event_bus() -> EventBus:
    """Get or create the singleton EventBus instance."""
    global _bus_instance
    if _bus_instance is None:
        _bus_instance = EventBus()
    return _bus_instance
