"""Contextual retrieval: a one-line summary of where each chunk sits in its document.

Prepended to the text used for **both** the embedding and BM25, but stored in its own
``context_line`` field so citations can quote only verbatim source text. Anthropic's published
tests put the effect at roughly halving failed retrievals, and about two-thirds when combined
with reranking.

The dominant failure it fixes is mundane: a chunk that reads "The limit is 120 EUR per night"
does not say which policy, which year, or which grade. Embedded alone it matches almost nothing;
prefixed with "Clause 4.2 of the 2025 Travel Policy, on per-diem limits" it matches the question
people actually ask.

``TemplateContextualizer`` is the CI and BYOC-without-LLM default. It is deterministic and free,
and it already recovers much of the benefit, because most of the missing context is metadata the
document already carries rather than anything that needs a model.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from typing import Protocol, runtime_checkable

from app.ingestion.types import ChildChunk


@dataclass(frozen=True, slots=True)
class DocumentContext:
    """What a contextualizer knows about the document a chunk came from."""

    title: str
    doc_type: str | None = None
    source_system: str | None = None
    author: str | None = None
    effective_from: date | None = None
    publication_date: date | None = None
    full_text: str = ""


@runtime_checkable
class Contextualizer(Protocol):
    #: Part of the generation fingerprint: editing a prompt must force a rebuild, because every
    #: stored vector was computed with the previous wording.
    version: str

    async def contextualize(self, document: DocumentContext, chunks: Sequence[ChildChunk]) -> list[str]: ...


class TemplateContextualizer:
    """Deterministic context lines built from metadata the document already has.

    No LLM, no cost, byte-identical across runs -- which is what lets the ``+contextual`` row of
    the ablation table be produced in CI rather than only against a paid API.
    """

    version = "template-1"

    async def contextualize(self, document: DocumentContext, chunks: Sequence[ChildChunk]) -> list[str]:
        return [self.line(document, chunk) for chunk in chunks]

    def line(self, document: DocumentContext, chunk: ChildChunk) -> str:
        parts: list[str] = []

        # The heading path minus the document root: the root is already the title.
        section = chunk.heading_path
        _, _, tail = section.partition(" > ")
        if tail:
            parts.append(f"{tail}")

        parts.append(f"in {document.title}")

        if document.doc_type:
            parts.append(f"({document.doc_type})")
        if document.effective_from:
            parts.append(f"effective {document.effective_from.isoformat()}")
        elif document.publication_date:
            parts.append(f"published {document.publication_date.isoformat()}")
        if document.source_system and document.source_system != "upload":
            parts.append(f"from {document.source_system}")

        return " ".join(parts).strip()


def prefixed(context_line: str, text: str) -> str:
    """The string that actually gets embedded and BM25-indexed.

    Kept as a function rather than inlined so that the embedding path and the indexing path
    cannot drift apart -- if they did, the vector would describe text that BM25 never saw.
    """
    line = context_line.strip()
    return f"{line}\n\n{text}" if line else text


def build_contextualizer(settings: object, provider: object | None = None) -> Contextualizer:
    """Select a contextualiser from configuration.

    ``llm`` without an available provider falls back to the template rather than failing to
    start. Contextual retrieval is a quality stage: a missing key should cost a few points of
    recall with a log line, not prevent ingestion entirely.
    """
    kind = getattr(settings, "contextualizer", "template")
    if kind == "llm" and provider is not None and getattr(provider, "available", False):
        from app.ingestion.llm_contextualize import LLMContextualizer

        return LLMContextualizer(
            provider,  # type: ignore[arg-type]
            batch_size=int(getattr(settings, "contextualize_batch_size", 20)),
        )
    return TemplateContextualizer()
