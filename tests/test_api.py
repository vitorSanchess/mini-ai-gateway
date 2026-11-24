"""Integration tests for FastAPI endpoints using mocked providers."""

from collections.abc import Generator
import pytest
from fastapi.testclient import TestClient

from app.config import Settings, get_settings
from app.main import app, get_metrics, get_router
from app.schemas import TokenUsage
from app.services.base import LLMProvider, ProviderCallError, ProviderResult
from app.services.cache import ResponseCache
from app.services.metrics import MetricsCollector
from app.services.router import GatewayRouter, RouteTarget


class MockProvider(LLMProvider):
    """In-memory provider recording every sanitized prompt it receives."""

    def __init__(self, provider_name: str, should_fail: bool = False) -> None:
        self._name = provider_name
        self._should_fail = should_fail
        self.received_prompts: list[str] = []

    @property
    def name(self) -> str:
        return self._name

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        model: str | None = None,
    ) -> ProviderResult:
        self.received_prompts.append(prompt)
        if self._should_fail:
            raise ProviderCallError(f"{self._name}/{model} unavailable")
        return ProviderResult(
            text=f"ok:{model}",
            provider=self._name,
            model=model or "mock-model",
            latency_ms=20.0,
            tokens=TokenUsage(input_tokens=10, output_tokens=5, total_tokens=15),
        )


@pytest.fixture(autouse=True)
def reset_dependency_overrides() -> Generator[None, None, None]:
    yield
    app.dependency_overrides.clear()


def _configure_app(
    routes: list[RouteTarget],
    pii_enabled: bool = True,
) -> tuple[TestClient, MetricsCollector]:
    router = GatewayRouter(
        routes=routes,
        cache=ResponseCache(ttl_seconds=300, max_size=64),
    )
    metrics = MetricsCollector()
    settings = Settings(
        app_name="Mini AI Gateway Test",
        app_env="test",
        gemini_api_key="test-key",
        pii_guardrail_enabled=pii_enabled,
    )

    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_router] = lambda: router
    app.dependency_overrides[get_metrics] = lambda: metrics

    return TestClient(app), metrics


def test_health_endpoint_reports_routes_and_guardrail() -> None:
    provider = MockProvider("gemini")
    client, _ = _configure_app([RouteTarget(provider, "gemini-3.8-flash")])

    response = client.get("/health")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["routes"] == ["gemini/gemini-3.8-flash"]
    assert data["cache_entries"] == 0
    assert data["pii_guardrail_enabled"] is True


def test_chat_masks_pii_caches_response_and_updates_metrics() -> None:
    provider = MockProvider("gemini")
    client, _ = _configure_app([RouteTarget(provider, "gemini-3.8-flash")])

    payload = {"prompt": "Cliente CPF 123.456.789-00 e email teste@banco.com"}

    first = client.post("/v1/chat", json=payload)
    assert first.status_code == 200
    body1 = first.json()
    assert first.headers["X-Request-ID"] == body1["request_id"]
    assert body1["pii_redacted"] is True
    assert body1["redactions"] == {"cpf": 1, "email": 1}
    assert body1["cache_hit"] is False
    assert body1["attempts"] == 1

    # Verify raw PII never reached the upstream provider
    assert len(provider.received_prompts) == 1
    assert "123.456.789-00" not in provider.received_prompts[0]
    assert "teste@banco.com" not in provider.received_prompts[0]
    assert "[CPF_REDACTED]" in provider.received_prompts[0]
    assert "[EMAIL_REDACTED]" in provider.received_prompts[0]

    # Second identical request must be served from cache without double-counting billed tokens
    second = client.post("/v1/chat", json=payload)
    assert second.status_code == 200
    body2 = second.json()
    assert body2["cache_hit"] is True
    assert body2["attempts"] == 0
    assert len(provider.received_prompts) == 1

    metrics_resp = client.get("/metrics")
    assert metrics_resp.status_code == 200
    m = metrics_resp.json()
    assert m["total_requests"] == 2
    assert m["successful_requests"] == 2
    assert m["failed_requests"] == 0
    assert m["cache_hits"] == 1
    assert m["cache_misses"] == 1
    assert m["cache_hit_rate"] == 0.5
    assert m["tokens_consumed"]["total_tokens"] == 15
    assert m["pii_redacted_requests"] == 2
    assert m["pii_redactions_by_type"] == {"cpf": 2, "email": 2}


def test_chat_fallback_and_metrics_tracking() -> None:
    primary = MockProvider("gemini", should_fail=True)
    backup = MockProvider("ollama", should_fail=False)
    client, _ = _configure_app(
        [
            RouteTarget(primary, "gemini-3.8-flash"),
            RouteTarget(backup, "llama3.2:1b"),
        ]
    )

    resp = client.post("/v1/chat", json={"prompt": "Olá"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["provider"] == "ollama"
    assert body["model"] == "llama3.2:1b"
    assert body["fallback"] is True
    assert body["attempts"] == 2

    m = client.get("/metrics").json()
    assert m["fallback_count"] == 1
    assert m["requests_by_provider"] == {"ollama": 1}


def test_chat_returns_502_when_all_providers_fail() -> None:
    failing = MockProvider("gemini", should_fail=True)
    client, _ = _configure_app([RouteTarget(failing, "gemini-3.8-flash")])

    resp = client.post("/v1/chat", json={"prompt": "Olá"})
    assert resp.status_code == 502

    m = client.get("/metrics").json()
    assert m["total_requests"] == 1
    assert m["failed_requests"] == 1
    assert m["successful_requests"] == 0


def test_chat_rejects_empty_prompt_with_422() -> None:
    provider = MockProvider("gemini")
    client, _ = _configure_app([RouteTarget(provider, "gemini-3.8-flash")])

    resp = client.post("/v1/chat", json={"prompt": ""})
    assert resp.status_code == 422
