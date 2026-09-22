"""Anthropic provider.

Uses the official SDK. Three API details this code depends on, each of which has a failure mode
if got wrong:

* **``output_config.format``** with a JSON schema, not the deprecated ``output_format`` and not
  a "reply with JSON" instruction. The schema is enforced, so the first text block is valid JSON
  and there is no repair path to maintain.
* **Assistant prefill is rejected** on this model family with a 400. Response shape is controlled
  by the schema instead, which is better anyway -- a prefill constrains the opening tokens and
  the schema constrains the whole object.
* **A refusal is an HTTP 200** carrying ``stop_reason == "refusal"`` and no usable content.
  Reading ``content[0]`` without checking is an IndexError in production at the worst moment.

Cache breakpoints are placed by the caller, not here. Contextualisation needs the document as a
cached prefix with per-batch chunks after it, and only the caller knows which part is stable.
"""

from __future__ import annotations

import json
from typing import Any

from app.core.config import Settings
from app.llm.base import (
    LLMRefusalError,
    LLMUnavailableError,
    StructuredResult,
    Usage,
)


class AnthropicProvider:
    name = "anthropic"

    def __init__(self, settings: Settings) -> None:
        self.model = settings.anthropic_model
        self._key = settings.anthropic_api_key.get_secret_value() if settings.anthropic_api_key else None
        self._client: Any | None = None

    @property
    def available(self) -> bool:
        return bool(self._key)

    def _ensure_client(self) -> Any:
        if self._client is not None:
            return self._client
        if not self._key:
            raise LLMUnavailableError("ANTHROPIC_API_KEY is not configured")
        try:
            from anthropic import AsyncAnthropic
        except ImportError as exc:  # pragma: no cover - dependency is declared
            raise LLMUnavailableError("the anthropic SDK is not installed") from exc
        self._client = AsyncAnthropic(api_key=self._key, max_retries=2)
        return self._client

    async def structured(
        self,
        *,
        system: list[dict[str, Any]],
        messages: list[dict[str, Any]],
        schema: dict[str, Any],
        max_tokens: int = 4096,
        effort: str | None = None,
    ) -> StructuredResult:
        client = self._ensure_client()

        request: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": messages,
            "output_config": {"format": {"type": "json_schema", "schema": schema}},
        }
        if effort:
            request["output_config"]["effort"] = effort

        try:
            from anthropic import APIError

            response = await client.messages.create(**request)
        except Exception as exc:  # the SDK raises a family of APIError subclasses
            if type(exc).__name__ == "APIStatusError" or isinstance(exc, APIError):
                raise LLMUnavailableError(f"anthropic request failed: {exc}") from exc
            raise LLMUnavailableError(f"anthropic request failed: {exc}") from exc

        if getattr(response, "stop_reason", None) == "refusal":
            details = getattr(response, "stop_details", None)
            raise LLMRefusalError(
                category=getattr(details, "category", None),
                explanation=getattr(details, "explanation", None),
            )

        text = next((block.text for block in response.content if block.type == "text"), None)
        if text is None:
            raise LLMUnavailableError("the response contained no text block")

        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            # Should be unreachable: output_config.format constrains the output. If it ever
            # fires, the schema and the model disagree and that is worth surfacing loudly
            # rather than silently repairing.
            raise LLMUnavailableError(f"schema-constrained output was not valid JSON: {exc}") from exc

        return StructuredResult(data=data, usage=_usage_of(response), model=self.model)


def _usage_of(response: Any) -> Usage:
    usage = getattr(response, "usage", None)
    if usage is None:
        return Usage(calls=1)
    return Usage(
        input_tokens=getattr(usage, "input_tokens", 0) or 0,
        output_tokens=getattr(usage, "output_tokens", 0) or 0,
        cache_read_input_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
        cache_creation_input_tokens=getattr(usage, "cache_creation_input_tokens", 0) or 0,
        calls=1,
    )


def cached(text: str, *, ttl: str | None = None) -> dict[str, Any]:
    """A system block with a cache breakpoint on it.

    ``ttl="1h"`` is worth paying for when a single document is contextualised across many
    batches spread over minutes; the default five minutes is enough when the worker processes a
    document's batches back to back, which the scheduler guarantees.
    """
    control: dict[str, Any] = {"type": "ephemeral"}
    if ttl:
        control["ttl"] = ttl
    return {"type": "text", "text": text, "cache_control": control}


def plain(text: str) -> dict[str, Any]:
    return {"type": "text", "text": text}
