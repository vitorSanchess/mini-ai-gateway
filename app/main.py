"""FastAPI entrypoint for the Mini AI Gateway (Routing, Fallback & Cache)."""

from functools import lru_cache

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse

from app.config import Settings, get_settings
from app.schemas import ChatRequest, ChatResponse, HealthResponse
from app.services.base import AllProvidersFailedError, ProviderConfigurationError
from app.services.cache import ResponseCache
from app.services.gemini_client import GeminiClient
from app.services.ollama_client import OllamaClient
from app.services.router import GatewayRouter, RouteTarget

app = FastAPI(
    title="Mini AI Gateway",
    version="0.2.0",
    description="Lightweight AI Gateway with routing, automatic fallback, and in-memory caching.",
)


@app.exception_handler(ProviderConfigurationError)
def handle_provider_config_error(
    _request: Request,
    exc: ProviderConfigurationError,
) -> JSONResponse:
    """Return a clean JSON 500 error if no LLM providers are configured."""
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


@app.get("/health", response_model=HealthResponse, tags=["System"])
def health_check(
    settings: Settings = Depends(get_settings),
    router: GatewayRouter = Depends(get_router),
) -> HealthResponse:
    """Verify the service is running and inspect active routes and cache size."""
    return HealthResponse(
        status="ok",
        app_name=settings.app_name,
        environment=settings.app_env,
        routes=router.route_labels,
        cache_entries=router.cache_entries,
    )


@app.post("/v1/chat", response_model=ChatResponse, tags=["Gateway"])
def chat(
    payload: ChatRequest,
    router: GatewayRouter = Depends(get_router),
) -> ChatResponse:
    """Route a prompt through cache + ordered provider fallback chain."""
    try:
        gateway_result = router.execute(
            prompt=payload.prompt,
            system_prompt=payload.system_prompt,
            requested_model=payload.model,
            use_cache=payload.use_cache,
        )
    except ProviderConfigurationError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc
    except AllProvidersFailedError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(exc),
        ) from exc

    return ChatResponse(
        answer=gateway_result.result.text,
        provider=gateway_result.result.provider,
        model=gateway_result.result.model,
        latency_ms=gateway_result.latency_ms,
        tokens_used=gateway_result.result.tokens,
        cache_hit=gateway_result.cache_hit,
        fallback=gateway_result.fallback_used,
        attempts=gateway_result.attempts,
    )
    