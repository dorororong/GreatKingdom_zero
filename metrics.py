from dataclasses import dataclass


@dataclass
class CacheMetrics:
    hits: int = 0
    misses: int = 0
    deduped: int = 0
    inflight_waits: int = 0

    def hit(self, n=1):
        self.hits += int(n)

    def miss(self, n=1):
        self.misses += int(n)

    def dedup(self, n=1):
        self.deduped += int(n)

    def inflight(self, n=1):
        self.inflight_waits += int(n)

    def summary(self):
        total = self.hits + self.misses
        hit_rate = self.hits / total if total else 0.0
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
            "deduped": self.deduped,
            "inflight_waits": self.inflight_waits
        }
