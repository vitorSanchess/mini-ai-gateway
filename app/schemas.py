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

    answer: str
    provider: str
    model: str
    latency_ms: float = Field(..., ge=0)
    tokens_used: TokenUsage
    cache_hit: bool = False
    fallback: bool = False
    attempts: int = Field(default=1, ge=0)


class HealthResponse(BaseModel):
    """Service health status for GET /health."""

    status: str
    app_name: str
    environment: str
    routes: list[str]
    cache_entries: int
