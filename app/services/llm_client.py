"""Thin HTTP client for the optional LLM upgrade path (OpenAI / Anthropic).

Only ever exercised when `LLM_PROVIDER` is set to "openai" or "anthropic"
*and* `LLM_API_KEY` is configured -- see `app/core/config.py`. Every public
function here returns `None` on any failure (missing config, network
error, timeout, unexpected response shape) instead of raising, so callers
(`nlp_triage.llm_extract`, `chat_engine`) can transparently fall back to
their rule-based paths. This module is safe to import even when `httpx`
calls are never exercised (e.g. LLM_PROVIDER="none").
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

import httpx

from app.core.config import Settings

logger = logging.getLogger(__name__)

OPENAI_DEFAULT_BASE_URL = "https://api.openai.com/v1"
ANTHROPIC_DEFAULT_BASE_URL = "https://api.anthropic.com/v1"
ANTHROPIC_API_VERSION = "2023-06-01"

DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
DEFAULT_ANTHROPIC_MODEL = "claude-3-5-haiku-20241022"


async def complete(
    *,
    messages: List[Dict[str, str]],
    settings: Settings,
    system_prompt: Optional[str] = None,
    json_mode: bool = False,
    max_tokens: int = 500,
) -> Optional[str]:
    """Request a chat-style completion from the configured LLM provider.

    `messages` is a list of `{"role": "user"|"assistant", "content": str}`
    dicts (oldest first). Returns the assistant's raw text reply, or
    `None` on any failure. Never raises.
    """
    if settings.LLM_PROVIDER == "openai":
        return await _call_openai(messages, settings, system_prompt, json_mode, max_tokens)
    if settings.LLM_PROVIDER == "anthropic":
        return await _call_anthropic(messages, settings, system_prompt, max_tokens)
    return None


async def _call_openai(
    messages: List[Dict[str, str]],
    settings: Settings,
    system_prompt: Optional[str],
    json_mode: bool,
    max_tokens: int,
) -> Optional[str]:
    if not settings.LLM_API_KEY:
        return None

    base_url = (settings.LLM_API_BASE_URL or OPENAI_DEFAULT_BASE_URL).rstrip("/")
    url = f"{base_url}/chat/completions"

    payload_messages: List[Dict[str, str]] = []
    if system_prompt:
        payload_messages.append({"role": "system", "content": system_prompt})
    payload_messages.extend(messages)

    payload: Dict[str, Any] = {
        "model": settings.LLM_MODEL or DEFAULT_OPENAI_MODEL,
        "messages": payload_messages,
        "max_tokens": max_tokens,
        "temperature": 0.3,
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}

    headers = {
        "Authorization": f"Bearer {settings.LLM_API_KEY}",
        "Content-Type": "application/json",
    }

    try:
        async with httpx.AsyncClient(timeout=settings.HTTP_TIMEOUT_SECONDS) as client:
            response = await client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]
    except Exception:  # noqa: BLE001 - any failure must degrade gracefully
        logger.warning("OpenAI-compatible completion request failed", exc_info=True)
        return None


async def _call_anthropic(
    messages: List[Dict[str, str]],
    settings: Settings,
    system_prompt: Optional[str],
    max_tokens: int,
) -> Optional[str]:
    if not settings.LLM_API_KEY:
        return None

    base_url = (settings.LLM_API_BASE_URL or ANTHROPIC_DEFAULT_BASE_URL).rstrip("/")
    url = f"{base_url}/messages"

    payload: Dict[str, Any] = {
        "model": settings.LLM_MODEL or DEFAULT_ANTHROPIC_MODEL,
        "max_tokens": max_tokens,
        "messages": messages,
    }
    if system_prompt:
        payload["system"] = system_prompt

    headers = {
        "x-api-key": settings.LLM_API_KEY,
        "anthropic-version": ANTHROPIC_API_VERSION,
        "Content-Type": "application/json",
    }

    try:
        async with httpx.AsyncClient(timeout=settings.HTTP_TIMEOUT_SECONDS) as client:
            response = await client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()
            content_blocks = data.get("content", [])
            texts = [block.get("text", "") for block in content_blocks if block.get("type") == "text"]
            joined = "".join(texts).strip()
            return joined or None
    except Exception:  # noqa: BLE001 - any failure must degrade gracefully
        logger.warning("Anthropic completion request failed", exc_info=True)
        return None


def parse_json_object(raw: str) -> Optional[Dict[str, Any]]:
    """Defensively extract a JSON object from an LLM's raw text reply.

    Handles the common case of a model wrapping JSON in a ```json fence
    despite instructions not to, or prefixing it with stray text. Returns
    `None` (never raises) if no valid JSON object could be recovered.
    """
    text = raw.strip()

    fence_match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1)
    else:
        brace_match = re.search(r"\{.*\}", text, re.DOTALL)
        if brace_match:
            text = brace_match.group(0)

    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None

    if not isinstance(parsed, dict):
        return None
    return parsed
