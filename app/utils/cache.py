"""Simple TTL cache for single-value endpoints.

Replaces the repetitive dict pattern:
    _cache = {"data": None, "timestamp": 0, "ttl": 60}

Usage:
    _cache = SimpleCache(ttl=60)

    # Read
    hit, data = _cache.get()
    if hit:
        return data

    # Write
    data = expensive_query()
    _cache.set(data)

    # Param-aware (cache invalidates when params change)
    hit, data = _cache.get(params=filter_key)
    _cache.set(data, params=filter_key)

    # Invalidate
    _cache.invalidate()
"""

import time


_UNSET = object()


class SimpleCache:
    """TTL-based single-value cache with optional param-aware invalidation."""

    __slots__ = ("ttl", "data", "timestamp", "params")

    def __init__(self, ttl: float):
        self.ttl = ttl
        self.data = None
        self.timestamp: float = 0.0
        self.params = None

    def get(self, *, params=_UNSET) -> tuple[bool, object]:
        """Return (hit, data). Check TTL and optional params match."""
        if self.data is None:
            return False, None
        if time.time() - self.timestamp >= self.ttl:
            return False, None
        if params is not _UNSET and self.params != params:
            return False, None
        return True, self.data

    def set(self, data: object, *, params=None) -> None:
        """Store data with current timestamp."""
        self.data = data
        self.timestamp = time.time()
        if params is not None:
            self.params = params

    def invalidate(self) -> None:
        """Clear cached data."""
        self.data = None
        self.timestamp = 0.0
        self.params = None

    @property
    def age(self) -> "float | None":
        """Seconds since last set, or None if empty."""
        if self.timestamp == 0.0:
            return None
        return time.time() - self.timestamp
