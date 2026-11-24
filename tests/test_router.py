"""Unit tests for GatewayRouter routing, fallback, and cache behavior."""

import pytest

from app.schemas import TokenUsage
from app.services.base import (
    AllProvidersFailedError,
    LLMProvider,
    ProviderCallError,
    ProviderResult,
)
from app.services.cache import ResponseCache
from app.services.router import GatewayRouter, RouteTarget


class FakeProvider(LLMProvider):
    """Configurable test double for LLMProvider."""

    def __init__(self, provider_name: str, should_fail: bool = False) -> None:
        self._name = provider_name
        self._should_fail = should_fail
        self.calls: list[tuple[str, str | None, str | None]] = []

    @property
    def name(self) -> str:
        return self._name

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        model: str | None = None,
    ) -> ProviderResult:
        self.calls.append((prompt, system_prompt, model))
        if self._should_fail:
            raise ProviderCallError(f"{self._name}/{model} simulated 503 error")
        return ProviderResult(
            text=f"reply from {self._name}/{model}",
            provider=self._name,
            model=model or "default",
            latency_ms=12.5,
            tokens=TokenUsage(input_tokens=5, output_tokens=7, total_tokens=12),
        )


def test_primary_success_and_cache_hit() -> None:
    primary = FakeProvider("gemini")
    router = GatewayRouter(
        routes=[RouteTarget(primary, "gemini-3.8-flash")],
        cache=ResponseCache(ttl_seconds=60, max_size=10),
    )

    first = router.execute("What is an AI Gateway?")
    second = router.execute("What is an AI Gateway?")

    assert first.cache_hit is False
    assert first.fallback_used is False
    assert first.attempts == 1
    assert first.result.provider == "gemini"

    assert second.cache_hit is True
    assert second.fallback_used is False
    assert second.attempts == 0
    assert second.result.text == first.result.text
    assert len(primary.calls) == 1


def test_bypass_cache_when_use_cache_false() -> None:
    primary = FakeProvider("gemini")
    router = GatewayRouter(
        routes=[RouteTarget(primary, "gemini-3.8-flash")],
        cache=ResponseCache(ttl_seconds=60, max_size=10),
    )

    router.execute("repeat me", use_cache=False)
    router.execute("repeat me", use_cache=False)

    assert len(primary.calls) == 2


def test_automatic_fallback_when_primary_fails() -> None:
    primary = FakeProvider("gemini", should_fail=True)
    backup = FakeProvider("ollama", should_fail=False)
    router = GatewayRouter(
        routes=[
            RouteTarget(primary, "gemini-3.8-flash"),
            RouteTarget(backup, "llama3.2:1b"),
        ],
        cache=ResponseCache(ttl_seconds=60, max_size=10),
    )

    result = router.execute("test fallback")

    assert result.fallback_used is True
    assert result.attempts == 2
    assert result.result.provider == "ollama"
    assert result.result.model == "llama3.2:1b"
    assert len(primary.calls) == 1
    assert len(backup.calls) == 1


def test_explicit_requested_model_is_tried_first() -> None:
    primary = FakeProvider("gemini")
    backup = FakeProvider("ollama")
    router = GatewayRouter(
        routes=[
            RouteTarget(primary, "gemini-3.8-flash"),
            RouteTarget(backup, "llama3.2:1b"),
        ],
        cache=ResponseCache(ttl_seconds=60, max_size=10),
    )

    result = router.execute("route directly", requested_model="llama3.2:1b")

    assert result.fallback_used is False
    assert result.attempts == 1
    assert result.result.provider == "ollama"
    assert len(primary.calls) == 0
    assert len(backup.calls) == 1


def test_raises_when_all_providers_fail() -> None:
    p1 = FakeProvider("gemini", should_fail=True)
    p2 = FakeProvider("ollama", should_fail=True)
    router = GatewayRouter(
        routes=[
            RouteTarget(p1, "gemini-3.8-flash"),
            RouteTarget(p2, "llama3.2:1b"),
        ],
        cache=ResponseCache(ttl_seconds=60, max_size=10),
    )

    with pytest.raises(AllProvidersFailedError) as exc_info:
        router.execute("fail everywhere")

    assert len(exc_info.value.errors) == 2
