"""The offline provider.

Not a mock in the testing sense -- it is the shipped behaviour when no model is configured, and
it is what CI runs. Every call raises, which is deliberate: a caller that reaches the model
without a model configured has a bug, and returning plausible empty output would hide it.

Callers that can degrade check ``available`` first. The contextualiser falls back to its
template, the query planner falls back to rules, and the answer path falls back to extractive
generation -- all without a branch on provider type.
"""

from __future__ import annotations

from typing import Any

from app.llm.base import LLMUnavailableError, StructuredResult


class NoLLMProvider:
    name = "none"
    model = "none"

    @property
    def available(self) -> bool:
        return False

    async def structured(
        self,
        *,
        system: list[dict[str, Any]],
        messages: list[dict[str, Any]],
        schema: dict[str, Any],
        max_tokens: int = 4096,
        effort: str | None = None,
    ) -> StructuredResult:
        raise LLMUnavailableError(
            "LLM_PROVIDER=none. Callers must check `available` and use their deterministic "
            "fallback rather than reaching the model."
        )
