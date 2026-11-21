"""Gemini API client wrapper using the official google-genai SDK."""

import time

from google import genai
from google.genai import types

from app.schemas import TokenUsage
from app.services.base import (
    LLMProvider,
    ProviderCallError,
    ProviderConfigurationError,
    ProviderResult,
)


class GeminiClient(LLMProvider):
    """Encapsulates calls to Google's Gemini API."""

    def __init__(self, api_key: str, default_model: str) -> None:
        if not api_key or api_key in {"your_gemini_api_key_here", "gemini-api-key"}:
            raise ProviderConfigurationError(
                "GEMINI_API_KEY is not configured. Set it in .env or environment variables."
            )
        self._client = genai.Client(api_key=api_key)
        self._default_model = default_model

    @property
    def name(self) -> str:
        return "gemini"

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        model: str | None = None,
    ) -> ProviderResult:
        """Send a prompt to Gemini and return text, latency, and token usage."""
        target_model = model or self._default_model
        config = (
            types.GenerateContentConfig(system_instruction=system_prompt)
            if system_prompt
            else None
        )

        start = time.perf_counter()
        try:
            response = self._client.models.generate_content(
                model=target_model,
                contents=prompt,
                config=config,
            )
        except Exception as exc:
            raise ProviderCallError(
                f"gemini/{target_model} failed: {exc}"
            ) from exc
        elapsed_ms = round((time.perf_counter() - start) * 1000, 2)

        usage = getattr(response, "usage_metadata", None)
        tokens = TokenUsage(
            input_tokens=(getattr(usage, "prompt_token_count", 0) or 0),
            output_tokens=(getattr(usage, "candidates_token_count", 0) or 0),
            total_tokens=(getattr(usage, "total_token_count", 0) or 0),
        )

        return ProviderResult(
            text=(response.text or "").strip(),
            provider=self.name,
            model=target_model,
            latency_ms=elapsed_ms,
            tokens=tokens,
        )
