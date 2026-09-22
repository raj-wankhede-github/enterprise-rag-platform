"""LLM contextualisation, its caching contract, and its degradation.

The caching tests are the ones that matter commercially. Contextual retrieval is the largest
ingestion cost and it is only affordable because the document sits in a cached prefix; a change
that breaks prefix stability does not fail anything, it just multiplies the bill.
"""

from __future__ import annotations

from datetime import date
from typing import Any

import pytest

from app.ingestion.contextualize import DocumentContext, TemplateContextualizer, prefixed
from app.ingestion.llm_contextualize import (
    CONTEXTUALIZER_VERSION,
    LLMContextualizer,
    estimated_cost_per_1k_chunks,
)
from app.ingestion.types import ChildChunk
from app.llm.base import LLMRefusalError, LLMUnavailableError, StructuredResult, Usage
from app.llm.none import NoLLMProvider

DOCUMENT = DocumentContext(
    title="Travel Policy",
    doc_type="policy",
    effective_from=date(2025, 1, 1),
    source_system="sharepoint",
    full_text="# Travel Policy\n\n## Per Diem\n\nThe rate is 120 EUR.\n\n## Receipts\n\nOver 50 EUR.",
)


def chunk(ordinal: int, text: str) -> ChildChunk:
    return ChildChunk(
        ordinal=ordinal,
        parent_ordinal=0,
        text=text,
        heading_path="Travel Policy > Per Diem",
        token_count=20,
    )


class RecordingProvider:
    """Captures every request so the cache layout can be asserted."""

    name = "recording"
    model = "claude-opus-5"

    def __init__(self, *, fail: Exception | None = None, skip_indices: set[int] | None = None) -> None:
        self.requests: list[dict[str, Any]] = []
        self.fail = fail
        self.skip_indices = skip_indices or set()

    @property
    def available(self) -> bool:
        return True

    async def structured(
        self,
        *,
        system: list[dict[str, Any]],
        messages: list[dict[str, Any]],
        schema: dict[str, Any],
        max_tokens: int = 4096,
        effort: str | None = None,
    ) -> StructuredResult:
        self.requests.append({"system": system, "messages": messages, "schema": schema})
        if self.fail:
            raise self.fail

        content = messages[0]["content"]
        count = content.count("\n\n[") + (1 if content.startswith("Excerpts:\n\n[") else 0)
        contexts = [{"index": i, "context": f"model context {i}"} for i in range(count) if i not in self.skip_indices]
        # A cache read on every call after the first: what the real provider reports once the
        # document prefix is warm.
        usage = Usage(
            input_tokens=100,
            output_tokens=50,
            cache_read_input_tokens=0 if len(self.requests) == 1 else 5000,
            cache_creation_input_tokens=5000 if len(self.requests) == 1 else 0,
            calls=1,
        )
        return StructuredResult(data={"contexts": contexts}, usage=usage, model=self.model)


# --------------------------------------------------------------------------------------------
# The caching contract
# --------------------------------------------------------------------------------------------


async def test_the_document_sits_behind_a_cache_breakpoint() -> None:
    """The entire economic argument for this design."""
    provider = RecordingProvider()
    await LLMContextualizer(provider).contextualize(DOCUMENT, [chunk(0, "a"), chunk(1, "b")])

    system = provider.requests[0]["system"]
    cached_blocks = [block for block in system if "cache_control" in block]
    assert len(cached_blocks) == 1
    assert DOCUMENT.full_text in cached_blocks[0]["text"]
    assert cached_blocks[0]["cache_control"]["type"] == "ephemeral"


async def test_the_cached_prefix_is_byte_identical_across_batches() -> None:
    """One changed byte anywhere in the prefix invalidates it and multiplies the bill."""
    provider = RecordingProvider()
    chunks = [chunk(i, f"chunk text number {i}") for i in range(45)]
    await LLMContextualizer(provider, batch_size=20).contextualize(DOCUMENT, chunks)

    assert len(provider.requests) == 3
    prefixes = {repr(request["system"]) for request in provider.requests}
    assert len(prefixes) == 1, "the system prefix differed between batches; the cache will miss"


async def test_only_the_chunks_vary_between_batches() -> None:
    provider = RecordingProvider()
    chunks = [chunk(i, f"text {i}") for i in range(40)]
    await LLMContextualizer(provider, batch_size=20).contextualize(DOCUMENT, chunks)

    bodies = [request["messages"][0]["content"] for request in provider.requests]
    assert bodies[0] != bodies[1]
    assert "text 0" in bodies[0] and "text 0" not in bodies[1]


async def test_batches_respect_the_configured_size() -> None:
    provider = RecordingProvider()
    await LLMContextualizer(provider, batch_size=5).contextualize(DOCUMENT, [chunk(i, f"t{i}") for i in range(12)])
    assert len(provider.requests) == 3


async def test_a_long_ttl_is_passed_through_when_asked_for() -> None:
    provider = RecordingProvider()
    await LLMContextualizer(provider, cache_ttl="1h").contextualize(DOCUMENT, [chunk(0, "a")])
    cached_block = next(b for b in provider.requests[0]["system"] if "cache_control" in b)
    assert cached_block["cache_control"]["ttl"] == "1h"


async def test_cache_usage_is_accumulated_for_cost_reporting() -> None:
    provider = RecordingProvider()
    contextualizer = LLMContextualizer(provider, batch_size=10)
    await contextualizer.contextualize(DOCUMENT, [chunk(i, f"t{i}") for i in range(30)])

    assert contextualizer.usage.calls == 3
    assert contextualizer.usage.cache_creation_input_tokens == 5000
    assert contextualizer.usage.cache_read_input_tokens == 10000
    assert contextualizer.usage.cache_hit_rate > 0.9


# --------------------------------------------------------------------------------------------
# Output handling
# --------------------------------------------------------------------------------------------


async def test_context_lines_come_back_in_chunk_order() -> None:
    provider = RecordingProvider()
    lines = await LLMContextualizer(provider, batch_size=2).contextualize(
        DOCUMENT, [chunk(i, f"t{i}") for i in range(4)]
    )
    assert lines == ["model context 0", "model context 1", "model context 0", "model context 1"]


async def test_a_skipped_excerpt_falls_back_rather_than_indexing_an_empty_context() -> None:
    """An empty context line means a chunk indexed with nothing situating it."""
    provider = RecordingProvider(skip_indices={1})
    lines = await LLMContextualizer(provider).contextualize(DOCUMENT, [chunk(0, "a"), chunk(1, "b")])
    assert lines[0] == "model context 0"
    assert lines[1]
    assert "Travel Policy" in lines[1]


async def test_the_schema_requires_an_index_and_a_context() -> None:
    provider = RecordingProvider()
    await LLMContextualizer(provider).contextualize(DOCUMENT, [chunk(0, "a")])
    schema = provider.requests[0]["schema"]
    item = schema["properties"]["contexts"]["items"]
    assert item["required"] == ["index", "context"]
    assert item["additionalProperties"] is False


# --------------------------------------------------------------------------------------------
# Degradation
# --------------------------------------------------------------------------------------------


async def test_no_provider_uses_the_template() -> None:
    lines = await LLMContextualizer(NoLLMProvider()).contextualize(DOCUMENT, [chunk(0, "a")])
    assert lines == await TemplateContextualizer().contextualize(DOCUMENT, [chunk(0, "a")])


@pytest.mark.parametrize(
    "failure",
    [LLMUnavailableError("down"), LLMRefusalError(category="cyber", explanation=None)],
)
async def test_a_failure_degrades_to_the_template_rather_than_failing_the_ingest(
    failure: Exception,
) -> None:
    provider = RecordingProvider(fail=failure)
    lines = await LLMContextualizer(provider).contextualize(DOCUMENT, [chunk(0, "a")])
    assert len(lines) == 1
    assert "Travel Policy" in lines[0]


async def test_a_mid_document_failure_keeps_the_batches_already_paid_for() -> None:
    """Discarding completed batches would throw away work already billed."""

    class FailsAfterFirst(RecordingProvider):
        async def structured(self, **kwargs: Any) -> StructuredResult:
            if len(self.requests) >= 1:
                self.requests.append({"system": [], "messages": [], "schema": {}})
                raise LLMUnavailableError("rate limited")
            return await super().structured(**kwargs)

    provider = FailsAfterFirst()
    lines = await LLMContextualizer(provider, batch_size=2).contextualize(
        DOCUMENT, [chunk(i, f"t{i}") for i in range(4)]
    )
    assert lines[0] == "model context 0"
    assert "Travel Policy" in lines[2]


async def test_an_oversized_document_uses_the_template() -> None:
    """The prefix would dominate every batch even at cache-read prices."""
    huge = DocumentContext(title="Big", full_text="x" * 500_000)
    provider = RecordingProvider()
    lines = await LLMContextualizer(provider, max_document_chars=1000).contextualize(huge, [chunk(0, "a")])
    assert provider.requests == []
    assert lines[0]


async def test_no_chunks_means_no_call() -> None:
    provider = RecordingProvider()
    assert await LLMContextualizer(provider).contextualize(DOCUMENT, []) == []
    assert provider.requests == []


# --------------------------------------------------------------------------------------------
# Prompt and versioning
# --------------------------------------------------------------------------------------------


async def test_the_document_is_labelled_as_reference_not_instructions() -> None:
    """Indirect prompt injection: an ingested document must not be able to issue orders."""
    provider = RecordingProvider()
    await LLMContextualizer(provider).contextualize(DOCUMENT, [chunk(0, "a")])
    instructions = provider.requests[0]["system"][0]["text"].lower()
    assert "not as instructions" in instructions
    assert "content to describe" in instructions


async def test_the_prompt_forbids_adding_facts() -> None:
    provider = RecordingProvider()
    await LLMContextualizer(provider).contextualize(DOCUMENT, [chunk(0, "a")])
    instructions = provider.requests[0]["system"][0]["text"].lower()
    assert "never add facts" in instructions


def test_the_version_is_exposed_for_the_generation_fingerprint() -> None:
    """A prompt edit must force a rebuild rather than silently splitting the index."""
    assert LLMContextualizer(NoLLMProvider()).version == CONTEXTUALIZER_VERSION
    assert TemplateContextualizer().version != CONTEXTUALIZER_VERSION


# --------------------------------------------------------------------------------------------
# Cost
# --------------------------------------------------------------------------------------------


def test_cost_per_1k_chunks_reflects_the_cache_hit_rate() -> None:
    """The number that decides whether contextual retrieval is affordable at corpus scale."""
    cold = Usage(input_tokens=1_000_000, output_tokens=10_000, calls=50)
    warm = Usage(input_tokens=50_000, cache_read_input_tokens=950_000, output_tokens=10_000, calls=50)
    cold_cost = estimated_cost_per_1k_chunks(cold, 1000, "claude-opus-5")
    warm_cost = estimated_cost_per_1k_chunks(warm, 1000, "claude-opus-5")
    assert warm_cost < cold_cost * 0.4


def test_cost_of_no_chunks_is_zero() -> None:
    assert estimated_cost_per_1k_chunks(Usage(), 0, "claude-opus-5") == 0.0


def test_an_unknown_model_costs_zero_rather_than_guessing() -> None:
    assert Usage(input_tokens=1000).cost_usd("some-future-model") == 0.0


def test_prefixed_joins_the_context_line_to_the_chunk() -> None:
    assert prefixed("context here", "body text") == "context here\n\nbody text"
    assert prefixed("", "body text") == "body text"
