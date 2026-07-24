"""Application settings, loaded from environment variables / .env file.

Uses pydantic-settings so all configuration is validated once at process
startup and shared everywhere via `get_settings()` (a cached singleton).
"""

from functools import lru_cache
from typing import Literal, Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Strongly-typed environment configuration.

    See `.env.example` at the package root for a documented list of every
    variable and its default.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    # --- Auth -----------------------------------------------------------
    # Shared-secret bearer token the NestJS backend must send in the
    # `Authorization: Bearer <token>` header on every /api/v1/* call.
    AI_SERVICE_TOKEN: str

    # --- Server -----------------------------------------------------------
    PORT: int = 8000

    # --- Optional LLM upgrade path -----------------------------------------
    # "none" keeps the service fully rule-based (zero external calls, zero
    # API keys required). Set to "openai" or "anthropic" to opt into
    # LLM-assisted symptom extraction / chat replies, with automatic
    # fallback to the rule-based engine on any failure.
    LLM_PROVIDER: Literal["none", "openai", "anthropic"] = "none"
    LLM_API_KEY: Optional[str] = None
    LLM_API_BASE_URL: Optional[str] = None
    LLM_MODEL: Optional[str] = None

    # --- Computer vision (CLIP zero-shot) ----------------------------------
    ENABLE_CV: bool = True
    CV_MODEL_NAME: str = "openai/clip-vit-base-patch32"

    # --- Networking ---------------------------------------------------------
    HTTP_TIMEOUT_SECONDS: float = 10.0


@lru_cache
def get_settings() -> Settings:
    """Cached settings singleton (env is only parsed once per process)."""
    return Settings()  # type: ignore[call-arg]
