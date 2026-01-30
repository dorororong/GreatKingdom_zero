import threading
from collections import OrderedDict

from metrics import CacheMetrics


class EvalCache:
    """LRU cache with optional in-flight dedup."""

    def __init__(self, max_entries=50000):
        self.max_entries = max_entries
        self._cache = OrderedDict()
        self._inflight = {}
        self._lock = threading.Lock()
        self.metrics = CacheMetrics()

    def clear(self):
        with self._lock:
            self._cache.clear()
            self._inflight.clear()

    def get(self, key):
        with self._lock:
            cached = self._cache.get(key)
            if cached is None:
                self.metrics.miss()
                return None
            self.metrics.hit()
            self._cache.move_to_end(key)
            return cached

    def reserve(self, key):
        """Return (cached, event, is_owner)."""
        with self._lock:
            cached = self._cache.get(key)
            if cached is not None:
                self.metrics.hit()
                self._cache.move_to_end(key)
                return cached, None, False
            self.metrics.miss()
            evt = self._inflight.get(key)
            if evt is None:
                evt = threading.Event()
                self._inflight[key] = evt
                return None, evt, True
            self.metrics.inflight()
            return None, evt, False

    def store(self, key, value):
        with self._lock:
            self._cache[key] = value
            self._cache.move_to_end(key)
            if self.max_entries and len(self._cache) > self.max_entries:
                self._cache.popitem(last=False)
            evt = self._inflight.pop(key, None)
            if evt is not None:
                evt.set()

    def wait_for(self, key, evt):
        evt.wait()
        with self._lock:
            cached = self._cache.get(key)
            if cached is None:
                return None
            self._cache.move_to_end(key)
            return cached
