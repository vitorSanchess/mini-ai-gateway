"""Thread-safe in-memory metrics collector exposed via GET /metrics."""

from collections import Counter
import threading
import time

from app.schemas import LatencyStats, MetricsResponse, TokenUsage


class MetricsCollector:
    """Tracks gateway traffic, cache efficiency, fallbacks, PII redactions, tokens, and latency."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._started_at = time.monotonic()

        self._total_requests = 0
        self._successful_requests = 0
        self._failed_requests = 0
        self._cache_hits = 0
        self._cache_misses = 0
        self._fallback_count = 0
        self._pii_redacted_requests = 0

        self._pii_redactions_by_type: Counter[str] = Counter()
        self._requests_by_provider: Counter[str] = Counter()
        self._requests_by_model: Counter[str] = Counter()

        self._input_tokens = 0
        self._output_tokens = 0
        self._total_tokens = 0

        self._latency_sum_ms = 0.0
        self._latency_count = 0
        self._latency_min_ms: float | None = None
        self._latency_max_ms: float | None = None

    def record_success(
        self,
        *,
        provider: str,
        model: str,
        latency_ms: float,
        tokens: TokenUsage,
        cache_hit: bool,
        fallback: bool,
        pii_redactions: dict[str, int],
    ) -> None:
        """Record a completed /v1/chat request."""
        with self._lock:
            self._total_requests += 1
            self._successful_requests += 1

            if cache_hit:
                self._cache_hits += 1
            else:
                self._cache_misses += 1
                self._input_tokens += tokens.input_tokens
                self._output_tokens += tokens.output_tokens
                self._total_tokens += tokens.total_tokens

            if fallback:
                self._fallback_count += 1

            if pii_redactions:
                self._pii_redacted_requests += 1
                self._pii_redactions_by_type.update(pii_redactions)

            self._requests_by_provider[provider] += 1
            self._requests_by_model[model] += 1

            self._latency_sum_ms += latency_ms
            self._latency_count += 1
            self._latency_min_ms = (
                latency_ms
                if self._latency_min_ms is None
                else min(self._latency_min_ms, latency_ms)
            )
            self._latency_max_ms = (
                latency_ms
                if self._latency_max_ms is None
                else max(self._latency_max_ms, latency_ms)
            )

    def record_failure(self, *, pii_redactions: dict[str, int]) -> None:
        """Record a failed /v1/chat request."""
        with self._lock:
            self._total_requests += 1
            self._failed_requests += 1
            if pii_redactions:
                self._pii_redacted_requests += 1
                self._pii_redactions_by_type.update(pii_redactions)

    def snapshot(self, *, cache_entries: int) -> MetricsResponse:
        """Return a consistent point-in-time snapshot of all metrics."""
        with self._lock:
            cache_lookups = self._cache_hits + self._cache_misses
            cache_hit_rate = (
                round(self._cache_hits / cache_lookups, 4) if cache_lookups > 0 else 0.0
            )
            avg_latency = (
                round(self._latency_sum_ms / self._latency_count, 2)
                if self._latency_count > 0
                else 0.0
            )

            return MetricsResponse(
                uptime_seconds=round(time.monotonic() - self._started_at, 2),
                total_requests=self._total_requests,
                successful_requests=self._successful_requests,
                failed_requests=self._failed_requests,
                cache_hits=self._cache_hits,
                cache_misses=self._cache_misses,
                cache_hit_rate=cache_hit_rate,
                cache_entries=cache_entries,
                fallback_count=self._fallback_count,
                pii_redacted_requests=self._pii_redacted_requests,
                pii_redactions_by_type=dict(self._pii_redactions_by_type),
                requests_by_provider=dict(self._requests_by_provider),
                requests_by_model=dict(self._requests_by_model),
                tokens_consumed=TokenUsage(
                    input_tokens=self._input_tokens,
                    output_tokens=self._output_tokens,
                    total_tokens=self._total_tokens,
                ),
                latency_ms=LatencyStats(
                    avg=avg_latency,
                    min=round(self._latency_min_ms, 2) if self._latency_min_ms is not None else 0.0,
                    max=round(self._latency_max_ms, 2) if self._latency_max_ms is not None else 0.0,
                ),
            )
