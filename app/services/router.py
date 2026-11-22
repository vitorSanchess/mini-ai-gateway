"""Routing, automatic fallback, and caching orchestration for the AI Gateway."""

from dataclasses import dataclass
import logging
import time

from app.services.base import (
    AllProvidersFailedError,
    LLMProvider,
    ProviderCallError,
    ProviderResult,
)
from app.services.cache import ResponseCache

logger = logging.getLogger("mini_ai_gateway")


@dataclass(frozen=True)
class RouteTarget:
    """A concrete (provider, model) pair in the fallback chain."""

    provider: LLMProvider
    model: str

    @property
    def label(self) -> str:
        return f"{self.provider.name}/{self.model}"


@dataclass(frozen=True)
class GatewayResult:
    """Enriched result returned by the router to the API layer."""

    result: ProviderResult
    latency_ms: float
    cache_hit: bool
    fallback_used: bool
    attempts: int


class GatewayRouter:
    """Executes chat requests with cache lookup and ordered provider fallback."""

    def __init__(self, routes: list[RouteTarget], cache: ResponseCache) -> None:
        if not routes:
            raise ValueError("GatewayRouter requires at least one RouteTarget.")
        self._routes = routes
        self._cache = cache

    @property
    def route_labels(self) -> list[str]:
        return [route.label for route in self._routes]

    @property
    def cache_entries(self) -> int:
        return self._cache.size

    def _resolve_plan(self, requested_model: str | None) -> list[RouteTarget]:
        """Build the ordered route list for a request, honoring an explicit model override."""
        if not requested_model:
            return list(self._routes)

        requested = requested_model.strip()
        for idx, route in enumerate(self._routes):
            if route.model == requested or route.label == requested:
                return [route] + self._routes[:idx] + self._routes[idx + 1 :]

        # If caller asked for a custom model not in the default list, try it on the primary provider first.
        custom_target = RouteTarget(
            provider=self._routes[0].provider,
            model=requested,
        )
        return [custom_target] + list(self._routes)

    def execute(
        self,
        prompt: str,
        system_prompt: str | None = None,
        requested_model: str | None = None,
        use_cache: bool = True,
    ) -> GatewayResult:
        """Check cache first; on miss, try routes in order until one succeeds."""
        cache_key = self._cache.make_key(prompt, system_prompt, requested_model)

        if use_cache:
            start = time.perf_counter()
            cached = self._cache.get(cache_key)
            if cached is not None:
                elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
                return GatewayResult(
                    result=cached,
                    latency_ms=elapsed_ms,
                    cache_hit=True,
                    fallback_used=False,
                    attempts=0,
                )

        plan = self._resolve_plan(requested_model)
        errors: list[str] = []

        for attempt_idx, target in enumerate(plan, start=1):
            try:
                provider_result = target.provider.generate(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    model=target.model,
                )
            except ProviderCallError as exc:
                errors.append(str(exc))
                logger.warning(
                    "provider_route_failed",
                    extra={
                        "event": "provider_route_failed",
                        "provider": target.provider.name,
                        "model": target.model,
                        "attempt": attempt_idx,
                        "has_next_route": attempt_idx < len(plan),
                        "error": str(exc),
                    },
                )
                continue

            if use_cache:
                self._cache.set(cache_key, provider_result)

            return GatewayResult(
                result=provider_result,
                latency_ms=provider_result.latency_ms,
                cache_hit=False,
                fallback_used=attempt_idx > 1,
                attempts=attempt_idx,
            )

        raise AllProvidersFailedError(errors)
