"""FastAPI entrypoint for the Mini AI Gateway (Routing, Fallback, Cache, Guardrails & Observability)."""

from functools import lru_cache
import uuid

from fastapi import Depends, FastAPI, HTTPException, Request, Response, status
from fastapi.responses import JSONResponse

from app.config import Settings, get_settings
from app.schemas import ChatRequest, ChatResponse, HealthResponse, MetricsResponse
from app.services.base import AllProvidersFailedError, ProviderConfigurationError
from app.services.cache import ResponseCache
from app.services.gemini_client import GeminiClient
from app.services.guardrails import GuardrailResult, PIIGuardrail
from app.services.logging_config import configure_logging
from app.services.metrics import MetricsCollector
from app.services.ollama_client import OllamaClient
from app.services.router import GatewayRouter, RouteTarget

logger = configure_logging(get_settings().log_level)

app = FastAPI(
    title="Mini AI Gateway",
    version="0.3.0",
    description="Lightweight AI Gateway with routing, fallback, caching, PII guardrails, and observability.",
)


@app.exception_handler(ProviderConfigurationError)
def handle_provider_config_error(
    _request: Request,
    exc: ProviderConfigurationError,
) -> JSONResponse:
    """Return a clean JSON 500 error if no LLM providers are configured."""
    logger.error(
        "provider_configuration_error",
        extra={"event": "provider_configuration_error", "error": str(exc)},
    )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": str(exc)},
    )


@lru_cache
def get_router() -> GatewayRouter:
    """Build a singleton GatewayRouter with the configured provider fallback chain and cache."""
    settings: Settings = get_settings()
    routes: list[RouteTarget] = []

    if settings.gemini_api_key and settings.gemini_api_key not in {
        "your_gemini_api_key_here",
        "gemini-api-key",
    }:
        gemini_client = GeminiClient(
            api_key=settings.gemini_api_key,
            default_model=settings.gemini_model,
        )
        routes.append(RouteTarget(provider=gemini_client, model=settings.gemini_model))

        if (
            settings.gemini_fallback_model
            and settings.gemini_fallback_model != settings.gemini_model
        ):
            routes.append(
                RouteTarget(
                    provider=gemini_client,
                    model=settings.gemini_fallback_model,
                )
            )

    if settings.ollama_model:
        ollama_client = OllamaClient(
            base_url=settings.ollama_base_url,
            default_model=settings.ollama_model,
            timeout_seconds=settings.ollama_timeout_seconds,
        )
        routes.append(RouteTarget(provider=ollama_client, model=settings.ollama_model))

    if not routes:
        raise ProviderConfigurationError(
            "No LLM providers configured. Set GEMINI_API_KEY and/or OLLAMA_MODEL in .env."
        )

    cache = ResponseCache(
        ttl_seconds=settings.cache_ttl_seconds,
        max_size=settings.cache_max_size,
    )
    return GatewayRouter(routes=routes, cache=cache)


@lru_cache
def get_guardrail() -> PIIGuardrail:
    """Return a singleton PIIGuardrail instance."""
    return PIIGuardrail()


@lru_cache
def get_metrics() -> MetricsCollector:
    """Return a singleton MetricsCollector instance."""
    return MetricsCollector()


@app.get("/health", response_model=HealthResponse, tags=["System"])
def health_check(
    settings: Settings = Depends(get_settings),
    router: GatewayRouter = Depends(get_router),
) -> HealthResponse:
    """Verify the service is running and inspect active routes, cache size, and guardrail status."""
    return HealthResponse(
        status="ok",
        app_name=settings.app_name,
        environment=settings.app_env,
        routes=router.route_labels,
        cache_entries=router.cache_entries,
        pii_guardrail_enabled=settings.pii_guardrail_enabled,
    )


@app.get("/metrics", response_model=MetricsResponse, tags=["Observability"])
def metrics_endpoint(
    router: GatewayRouter = Depends(get_router),
    metrics: MetricsCollector = Depends(get_metrics),
) -> MetricsResponse:
    """Expose gateway metrics: traffic, cache hit rate, fallbacks, PII redactions, tokens, and latency."""
    return metrics.snapshot(cache_entries=router.cache_entries)


@app.post("/v1/chat", response_model=ChatResponse, tags=["Gateway"])
def chat(
    payload: ChatRequest,
    response: Response,
    settings: Settings = Depends(get_settings),
    router: GatewayRouter = Depends(get_router),
    guardrail: PIIGuardrail = Depends(get_guardrail),
    metrics: MetricsCollector = Depends(get_metrics),
) -> ChatResponse:
    """Sanitize PII, route through cache + fallback chain, record metrics, and emit JSON logs."""
    request_id = str(uuid.uuid4())
    response.headers["X-Request-ID"] = request_id

    if settings.pii_guardrail_enabled:
        guardrail_result = guardrail.sanitize(
            prompt=payload.prompt,
            system_prompt=payload.system_prompt,
        )
    else:
        guardrail_result = GuardrailResult(
            sanitized_prompt=payload.prompt,
            sanitized_system_prompt=payload.system_prompt,
            pii_detected=False,
            redactions={},
        )

    if guardrail_result.pii_detected:
        logger.warning(
            "pii_redacted",
            extra={
                "event": "pii_redacted",
                "request_id": request_id,
                "redactions": guardrail_result.redactions,
            },
        )

    try:
        gateway_result = router.execute(
            prompt=guardrail_result.sanitized_prompt,
            system_prompt=guardrail_result.sanitized_system_prompt,
            requested_model=payload.model,
            use_cache=payload.use_cache,
        )
    except AllProvidersFailedError as exc:
        metrics.record_failure(pii_redactions=guardrail_result.redactions)
        logger.error(
            "chat_request_failed",
            extra={
                "event": "chat_request_failed",
                "request_id": request_id,
                "status_code": status.HTTP_502_BAD_GATEWAY,
                "pii_redacted": guardrail_result.pii_detected,
                "error": str(exc),
            },
        )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(exc),
        ) from exc

    metrics.record_success(
        provider=gateway_result.result.provider,
        model=gateway_result.result.model,
        latency_ms=gateway_result.latency_ms,
        tokens=gateway_result.result.tokens,
        cache_hit=gateway_result.cache_hit,
        fallback=gateway_result.fallback_used,
        pii_redactions=guardrail_result.redactions,
    )

    logger.info(
        "chat_request_completed",
        extra={
            "event": "chat_request_completed",
            "request_id": request_id,
            "status_code": status.HTTP_200_OK,
            "provider": gateway_result.result.provider,
            "model": gateway_result.result.model,
            "latency_ms": gateway_result.latency_ms,
            "cache_hit": gateway_result.cache_hit,
            "fallback": gateway_result.fallback_used,
            "attempts": gateway_result.attempts,
            "tokens_total": gateway_result.result.tokens.total_tokens,
            "pii_redacted": guardrail_result.pii_detected,
            "redactions": guardrail_result.redactions,
        },
    )

    return ChatResponse(
        request_id=request_id,
        answer=gateway_result.result.text,
        provider=gateway_result.result.provider,
        model=gateway_result.result.model,
        latency_ms=gateway_result.latency_ms,
        tokens_used=gateway_result.result.tokens,
        cache_hit=gateway_result.cache_hit,
        fallback=gateway_result.fallback_used,
        attempts=gateway_result.attempts,
        pii_redacted=guardrail_result.pii_detected,
        redactions=guardrail_result.redactions,
    )
