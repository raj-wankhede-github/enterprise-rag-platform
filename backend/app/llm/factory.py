"""Selecting an LLM provider from configuration."""

from __future__ import annotations

from app.core.config import Settings
from app.llm.base import LLMProvider
from app.llm.none import NoLLMProvider


def build_llm(settings: Settings) -> LLMProvider:
    """Construct the configured provider.

    ``anthropic`` without an API key returns the offline provider rather than failing to start.
    Every caller already has a deterministic fallback, so a missing key degrades quality with a
    log line instead of taking the service down.
    """
    if settings.llm_provider == "anthropic":
        from app.llm.anthropic_provider import AnthropicProvider

        provider = AnthropicProvider(settings)
        return provider if provider.available else NoLLMProvider()
    return NoLLMProvider()
