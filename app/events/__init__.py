"""
Event-driven infrastructure for Phase 2 lineup cascade.

Architecture:
- asyncio.Queue for immediate dispatch (in-memory, fast)
- DB as source of truth (match_lineups.lineup_detected_at + predictions.asof_timestamp)
- Sweeper Queue every 2min: reconciles missed events via FOR UPDATE SKIP LOCKED

Usage:
    from app.events import get_event_bus, LINEUP_CONFIRMED

    bus = get_event_bus()
    bus.subscribe(LINEUP_CONFIRMED, cascade_handler)
    await bus.start()
    await bus.emit(LINEUP_CONFIRMED, {"match_id": 123})
"""

from app.events.bus import (
    Event,
    EventBus,
    LINEUP_CONFIRMED,
    get_event_bus,
    run_sweeper,
    sweep_missed_lineups,
)

__all__ = [
    "Event",
    "EventBus",
    "LINEUP_CONFIRMED",
    "get_event_bus",
    "run_sweeper",
    "sweep_missed_lineups",
]
