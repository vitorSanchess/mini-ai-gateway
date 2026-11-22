"""Pydantic request and response schemas for the API."""

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Incoming payload for POST /v1/chat."""

    prompt: str = Field(
        ...,
        min_length=1,
        max_length=16000,
        description="User prompt to send to the LLM.",
    )
    system_prompt: str | None = Field(
        default=None,
        max_length=4000,
        description="Optional system instruction to guide model behavior.",
    )
    model: str | None = Field(
        default=None,
        max_length=120,
        description="Optional preferred model override for primary routing.",
    )
    use_cache: bool = Field(
        default=True,
        description="Whether to read/write the in-memory response cache.",
    )


class TokenUsage(BaseModel):
    """Token breakdown returned by the LLM provider."""

    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)


class ChatResponse(BaseModel):
    """Normalized response returned by POST /v1/chat."""

    request_id: str
    answer: str
    provider: str
    model: str
    latency_ms: float = Field(..., ge=0)
    tokens_used: TokenUsage
    cache_hit: bool = False
    fallback: bool = False
    attempts: int = Field(default=1, ge=0)
    pii_redacted: bool = False
    redactions: dict[str, int] = Field(default_factory=dict)


class HealthResponse(BaseModel):
    """Service health status for GET /health."""

    status: str
    app_name: str
    environment: str
    routes: list[str]
    cache_entries: int
    pii_guardrail_enabled: bool


class LatencyStats(BaseModel):
    """Aggregated latency metrics in milliseconds."""

    avg: float = 0.0
    min: float = 0.0
    max: float = 0.0


class MetricsResponse(BaseModel):
    """Operational metrics exposed via GET /metrics."""

    uptime_seconds: float
    total_requests: int
    successful_requests: int
    failed_requests: int
    cache_hits: int
    cache_misses: int
    cache_hit_rate: float
    cache_entries: int
    fallback_count: int
    pii_redacted_requests: int
    pii_redactions_by_type: dict[str, int]
    requests_by_provider: dict[str, int]
    requests_by_model: dict[str, int]
    tokens_consumed: TokenUsage
    latency_ms: LatencyStats
