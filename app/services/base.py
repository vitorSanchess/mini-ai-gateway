"""Shared provider interfaces, result models, and domain exceptions."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

from app.schemas import TokenUsage


class ProviderConfigurationError(RuntimeError):
    """Raised when required provider credentials or settings are missing."""


class ProviderCallError(RuntimeError):
    """Raised when a single upstream LLM provider call fails."""


class AllProvidersFailedError(RuntimeError):
    """Raised when every provider in the routing/fallback chain fails."""

    def __init__(self, errors: list[str]) -> None:
        self.errors = errors
        summary = " | ".join(errors)
        super().__init__(f"All configured LLM routes failed: {summary}")


@dataclass(frozen=True)
class ProviderResult:
    """Internal normalized result returned by any LLM provider."""

    text: str
    provider: str
    model: str
    latency_ms: float
    tokens: TokenUsage


class LLMProvider(ABC):
    """Contract that every LLM provider client must implement."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier for the provider (e.g., 'gemini', 'ollama')."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        model: str | None = None,
    ) -> ProviderResult:
        """Execute a prompt against the provider and return a ProviderResult."""
