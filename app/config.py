"""Application configuration loaded from environment variables or .env."""

from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Typed runtime settings for the Mini AI Gateway."""

    app_name: str = "Mini AI Gateway"
    app_env: str = "development"

    # Primary & secondary Gemini models
    gemini_api_key: str = ""
    gemini_model: str = "gemini-4-hydrogen"
    gemini_fallback_model: str = ""

    # Optional local Ollama fallback
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = ""
    ollama_timeout_seconds: float = 30.0

    # In-memory TTL + LRU cache
    cache_ttl_seconds: int = 300
    cache_max_size: int = 256

    # Security & observability
    pii_guardrail_enabled: bool = True
    log_level: str = "INFO"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance so .env is only read once."""
    return Settings()
