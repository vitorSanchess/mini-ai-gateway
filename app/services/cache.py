"""Thread-safe in-memory TTL + LRU cache for repeated LLM prompts."""

from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import json
import threading
import time

from app.services.base import ProviderResult


@dataclass(frozen=True)
class _CacheEntry:
    value: ProviderResult
    expires_at: float


class ResponseCache:
    """Bounded in-memory cache with per-entry TTL and LRU eviction."""

    def __init__(self, ttl_seconds: int = 300, max_size: int = 256) -> None:
        self._ttl_seconds = ttl_seconds
        self._max_size = max_size
        self._store: OrderedDict[str, _CacheEntry] = OrderedDict()
        self._lock = threading.Lock()

    @staticmethod
    def make_key(
        prompt: str,
        system_prompt: str | None,
        requested_model: str | None,
    ) -> str:
        """Build a deterministic SHA-256 key from normalized request inputs."""
        canonical = json.dumps(
            {
                "prompt": prompt.strip(),
                "system_prompt": (system_prompt or "").strip(),
                "model": (requested_model or "auto").strip(),
            },
            sort_keys=True,
            ensure_ascii=False,
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def get(self, key: str) -> ProviderResult | None:
        """Return a cached ProviderResult if present and not expired."""
        now = time.monotonic()
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            if now >= entry.expires_at:
                del self._store[key]
                return None
            self._store.move_to_end(key)
            return entry.value

    def set(self, key: str, value: ProviderResult) -> None:
        """Store a ProviderResult and evict the least-recently-used entry if full."""
        if self._ttl_seconds <= 0 or self._max_size <= 0:
            return
        expires_at = time.monotonic() + self._ttl_seconds
        with self._lock:
            self._store[key] = _CacheEntry(value=value, expires_at=expires_at)
            self._store.move_to_end(key)
            while len(self._store) > self._max_size:
                self._store.popitem(last=False)

    def clear(self) -> None:
        """Remove all entries from the cache."""
        with self._lock:
            self._store.clear()

    @property
    def size(self) -> int:
        """Return the number of active (non-expired) cache entries."""
        now = time.monotonic()
        with self._lock:
            expired = [k for k, v in self._store.items() if now >= v.expires_at]
            for key in expired:
                del self._store[key]
            return len(self._store)
