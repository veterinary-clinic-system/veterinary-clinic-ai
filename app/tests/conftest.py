"""Shared pytest fixtures.

Ensures every test runs against a clean, deterministic `Settings()`
regardless of what a developer's local `.env` file (if any) contains:
explicit `monkeypatch.setenv` calls always take precedence over `.env`
values in pydantic-settings, and we clear the `get_settings()` lru_cache
before/after each test so changes actually take effect.

`ENABLE_CV=false` is forced here so the test suite never attempts to
import/load `transformers`/`torch`, matching the task requirement that
the endpoint tests must pass without the heavy CV dependencies installed.
"""

import pytest

from app.core.config import get_settings

TEST_TOKEN = "test-service-token"


@pytest.fixture(autouse=True)
def clean_settings(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("AI_SERVICE_TOKEN", TEST_TOKEN)
    monkeypatch.setenv("LLM_PROVIDER", "none")
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.setenv("ENABLE_CV", "false")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
