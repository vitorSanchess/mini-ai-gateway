"""Local Ollama HTTP client used as a resilient fallback provider."""

import time

import httpx

from app.schemas import TokenUsage
from app.services.base import LLMProvider, ProviderCallError, ProviderResult


class OllamaClient(LLMProvider):
    """Calls a local or remote Ollama server via its REST /api/generate endpoint."""

    def __init__(
        self,
        base_url: str,
        default_model: str,
        timeout_seconds: float = 30.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._default_model = default_model
        self._timeout = timeout_seconds

    @property
    def name(self) -> str:
        return "ollama"

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        model: str | None = None,
    ) -> ProviderResult:
        """Send a prompt to Ollama and normalize its response & token counts."""
        target_model = model or self._default_model
        payload: dict[str, object] = {
            "model": target_model,
            "prompt": prompt,
            "stream": False,
        }
        if system_prompt:
            payload["system"] = system_prompt

        start = time.perf_counter()
        try:
            response = httpx.post(
                f"{self._base_url}/api/generate",
                json=payload,
                timeout=self._timeout,
            )
            response.raise_for_status()
            data = response.json()
        except Exception as exc:
            raise ProviderCallError(
                f"ollama/{target_model} failed: {exc}"
            ) from exc
        elapsed_ms = round((time.perf_counter() - start) * 1000, 2)

        input_tokens = int(data.get("prompt_eval_count") or 0)
        output_tokens = int(data.get("eval_count") or 0)
        tokens = TokenUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
        )

        return ProviderResult(
            text=str(data.get("response", "")).strip(),
            provider=self.name,
            model=target_model,
            latency_ms=elapsed_ms,
            tokens=tokens,
        )
