"""The LLM contract and its cost accounting.

Two things every provider must report, because both are product requirements rather than
telemetry niceties:

* **Usage, split by cache state.** Contextual retrieval is the largest ingestion line item, and
  its viability rests entirely on prompt caching. A cache that silently stops hitting -- one
  changed byte in the prefix does it -- turns a cheap run into a bill nobody sees until the
  invoice. ``cache_read_input_tokens`` is how that becomes visible on the run that causes it.
* **Refusals as a first-class outcome.** A refusal returns HTTP 200 with
  ``stop_reason == "refusal"`` and no usable content. Code that reads ``content`` without
  checking gets an IndexError in production, at the worst moment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

#: Published per-million-token prices, used for the cost figures the eval reports. Cache reads
#: are ~10% of input and cache writes ~125%, which is what makes the document-prefix pattern in
#: contextualisation worth the complexity.
PRICING: dict[str, dict[str, float]] = {
    "claude-opus-5": {"input": 5.00, "output": 25.00},
    "claude-sonnet-5": {"input": 2.00, "output": 10.00},
    "claude-haiku-4-5": {"input": 1.00, "output": 5.00},
}
CACHE_READ_MULTIPLIER = 0.10
CACHE_WRITE_MULTIPLIER = 1.25


@dataclass(slots=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0
    calls: int = 0

    def add(self, other: Usage) -> None:
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.cache_read_input_tokens += other.cache_read_input_tokens
        self.cache_creation_input_tokens += other.cache_creation_input_tokens
        self.calls += other.calls

    @property
    def cache_hit_rate(self) -> float:
        """Share of input tokens served from cache.

        The number to watch. If it collapses between runs, something is invalidating the prefix
        -- a timestamp in the system prompt, a reordered dict, a worker interleaving documents.
        """
        total = self.input_tokens + self.cache_read_input_tokens
        return self.cache_read_input_tokens / total if total else 0.0

    def cost_usd(self, model: str) -> float:
        prices = PRICING.get(model)
        if not prices:
            return 0.0
        per_token_in = prices["input"] / 1_000_000
        per_token_out = prices["output"] / 1_000_000
        return (
            self.input_tokens * per_token_in
            + self.cache_read_input_tokens * per_token_in * CACHE_READ_MULTIPLIER
            + self.cache_creation_input_tokens * per_token_in * CACHE_WRITE_MULTIPLIER
            + self.output_tokens * per_token_out
        )


class LLMRefusalError(Exception):
    """The model declined. Returned as HTTP 200, so it must be checked rather than caught."""

    def __init__(self, category: str | None, explanation: str | None) -> None:
        self.category = category
        self.explanation = explanation
        super().__init__(f"model refused the request (category={category})")


class LLMUnavailableError(Exception):
    """Transport, rate limit or server failure. Callers degrade rather than fail."""


@dataclass(slots=True)
class StructuredResult:
    data: dict[str, Any]
    usage: Usage = field(default_factory=Usage)
    model: str = ""


@runtime_checkable
class LLMProvider(Protocol):
    name: str
    model: str

    async def structured(
        self,
        *,
        system: list[dict[str, Any]],
        messages: list[dict[str, Any]],
        schema: dict[str, Any],
        max_tokens: int = 4096,
        effort: str | None = None,
    ) -> StructuredResult:
        """One request constrained to a JSON schema.

        ``system`` is a list of content blocks rather than a string so the caller controls
        exactly where the cache breakpoint sits -- which is the whole mechanism behind cheap
        batched contextualisation.
        """
        ...

    @property
    def available(self) -> bool: ...
