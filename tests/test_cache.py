"""Unit tests for the thread-safe TTL + LRU ResponseCache."""

import time

from app.schemas import TokenUsage
from app.services.base import ProviderResult
from app.services.cache import ResponseCache


def _sample_result(text: str = "hello") -> ProviderResult:
    return ProviderResult(
        text=text,
        provider="mock",
        model="mock-v1",
        latency_ms=10.0,
        tokens=TokenUsage(input_tokens=4, output_tokens=6, total_tokens=10),
    )


def test_deterministic_key_normalizes_whitespace() -> None:
    key_a = ResponseCache.make_key("  hello world  ", "  sys ", "auto")
    key_b = ResponseCache.make_key("hello world", "sys", "auto")
    key_c = ResponseCache.make_key("hello world", "other", "auto")

    assert key_a == key_b
    assert key_a != key_c


def test_cache_stores_and_expires_after_ttl(monkeypatch) -> None:
    now = [100.0]
    monkeypatch.setattr(time, "monotonic", lambda: now[0])

    cache = ResponseCache(ttl_seconds=60, max_size=10)
    key = cache.make_key("ping", None, None)
    cache.set(key, _sample_result("pong"))

    assert cache.get(key) is not None
    assert cache.size == 1

    # Advance virtual clock past TTL
    now[0] = 161.0
    assert cache.get(key) is None
    assert cache.size == 0


def test_cache_evicts_least_recently_used_entry() -> None:
    cache = ResponseCache(ttl_seconds=300, max_size=2)
    k1 = cache.make_key("p1", None, None)
    k2 = cache.make_key("p2", None, None)
    k3 = cache.make_key("p3", None, None)

    cache.set(k1, _sample_result("r1"))
    cache.set(k2, _sample_result("r2"))

    # Touch k1 so k2 becomes the least recently used
    assert cache.get(k1) is not None

    cache.set(k3, _sample_result("r3"))

    assert cache.size == 2
    assert cache.get(k1) is not None
    assert cache.get(k2) is None
    assert cache.get(k3) is not None
